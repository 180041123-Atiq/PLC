import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F

from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import Levenshtein

class ImageEncoder(nn.Module):
    def __init__(self, backbone="resnet50", embed_dim=512):
        super().__init__()
        
        # Use ResNet without the classifier head
        resnet = models.resnet50(weights="IMAGENET1K_V1")
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        
        # Project features to embedding dimension for the decoder
        self.proj = nn.Conv2d(2048, embed_dim, kernel_size=1)
        
    def forward(self, x):
        """
        x: (B, 3, H, W) input image
        returns:
          seq_features: (B, L, D) sequence of patch embeddings
        """
        feats = self.feature_extractor(x)        # (B, 2048, H/32, W/32)
        feats = self.proj(feats)                 # (B, D, H/32, W/32)
        
        B, D, H, W = feats.shape
        seq_features = feats.flatten(2).permute(0, 2, 1) # (B, L=H*W, D)
        return seq_features

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len, num_layers=6, num_heads=8):
        super().__init__()
        
        # Token embedding (learned representations for {0,1,2,3})
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        
        # Positional encoding (so model knows step order)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Final classifier
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, encoder_feats, tgt_seq):
        """
        encoder_feats: (B, L, D) image features
        tgt_seq: (B, T) previously generated tokens (teacher forcing during training)
        """
        B, T = tgt_seq.shape
        positions = torch.arange(0, T, device=tgt_seq.device).unsqueeze(0)
        
        # Embed tokens + positions
        tgt_emb = self.token_emb(tgt_seq) + self.pos_emb(positions)
        
        # Transformer decoder expects shape (T, B, D)
        tgt_emb = tgt_emb.permute(1, 0, 2)
        memory = encoder_feats.permute(1, 0, 2)
        
        decoded = self.decoder(tgt_emb, memory)  # (T, B, D)
        
        # Back to (B, T, vocab_size)
        out = self.fc_out(decoded.permute(1, 0, 2))
        return out
    
class Img2SeqModel(nn.Module):
    def __init__(self, vocab_size=5, max_len=1401, embed_dim=512):
        super().__init__()
        self.encoder = ImageEncoder(embed_dim=embed_dim)
        self.decoder = TransformerDecoder(vocab_size=vocab_size, embed_dim=embed_dim, max_len=max_len)
    
    def forward(self, images, tgt_seq):
        """
        images: (B, 3, H, W)
        tgt_seq: (B, T) ground-truth sequence (used for teacher forcing)
        """
        enc_feats = self.encoder(images)
        out = self.decoder(enc_feats, tgt_seq)
        return out  # (B, T, vocab_size)

class PLCloss():
    def __init__(self, compound_loss=2, pad_token=3):
        self.compound_loss = compound_loss
        self.pad_token = pad_token

    def masked_ce_loss(self, preds, targets):
        """
        preds: (B, T, vocab_size)
        targets: (B, T)
        """
        preds = preds.reshape(-1, preds.size(-1))   # (B*T, vocab_size)
        targets = targets.reshape(-1)               # (B*T,)
        
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_token)
        return loss_fn(preds, targets)
    
    def grammar_loss(self,preds):
        
        probs = F.softmax(preds, dim=-1)
        predClasses = torch.argmax(preds,dim=-1)
        
        (batch_size,seq_len,vocub_size) = preds.size()

        mask = torch.ones_like(probs)
        
        stack_list = []
        bos_list = []
        for bb in range(batch_size):
            stack = []
            for sq in range(seq_len):
                if predClasses[bb][sq].item() == 4:
                    bos_list.append((bb,sq,4))
                if predClasses[bb][sq].item() == 0:
                    stack.append((0,sq))
                elif predClasses[bb][sq].item() == 1:
                    stack.append((1,sq))
                elif predClasses[bb][sq].item() == 2:
                    if len(stack) > 0:
                        stack.pop()
                    else:
                        stack.append((2,sq))
                elif predClasses[bb][sq].item() == 3:
                    break
            stack_list.append(stack)
        

        for bb in range(len(stack_list)):
            for sq in range(len(stack_list[bb])):
                cls,sqq = stack_list[bb][sq]
                mask[bb,sqq,cls] = 0

        for bs in range(len(bos_list)):
            bt,sqq,cls = bos_list[bs]
            mask[bt,sqq,cls] = 0

        # Probability assigned to illegal actions
        illegal_prob = probs * (1 - mask)
        loss = illegal_prob.sum(dim=2).mean()  # avg over batch
        
        return loss

    def end_token_loss(self, preds, targets):
        B, L, V = preds.size()
        predClasses = torch.argmax(preds,dim=-1)

        loss = 0.0
        for b in range(B):
            # First EOS in prediction
            pred_eos_idxs = (predClasses[b] == self.pad_token).nonzero(as_tuple=True)[0]
            first_pred_eos = pred_eos_idxs[0] if len(pred_eos_idxs) > 0 else L  # L = if no EOS predicted
            
            # First EOS in target
            target_eos_idxs = (targets[b] == self.pad_token).nonzero(as_tuple=True)[0]
            first_target_eos = target_eos_idxs[0] if len(target_eos_idxs) > 0 else L
            
            # Absolute difference loss
            loss += torch.abs(first_pred_eos - first_target_eos).float()
        
        return loss / B

    def genLoss(self,preds,tgt_seq):

        if self.compound_loss == 0:
            return self.masked_ce_loss(preds,tgt_seq[:,1:])
        elif self.compound_loss == 1:
            return self.masked_ce_loss(preds,tgt_seq[:,1:])+\
            self.grammar_loss(preds)
        elif self.compound_loss == 2:
            ce_loss = self.masked_ce_loss(preds,tgt_seq[:,1:])
            grm_loss = self.grammar_loss(preds)
            mx_loss = max(ce_loss,grm_loss)
            ed_loss = min(self.end_token_loss(preds,tgt_seq),mx_loss)
            return ce_loss+grm_loss+ed_loss
            
        

class WebpageDataset(Dataset):
    def __init__(self, 
            img_dir='webcode2m_plc/image', 
            seq_dir='webcode2m_plc/layout', 
            max_len=1401, 
            transform=None
        ):
        self.img_dir = img_dir
        self.seq_dir = seq_dir
        self.max_len = max_len
        self.transform = transform if transform else T.Compose([
            T.Resize((224, 224)),   # match encoder input size
            T.ToTensor(),
        ])
        
        # match image filenames and sequence filenames
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
        self.seq_files = sorted([f for f in os.listdir(seq_dir) if f.endswith(".txt")])
        assert len(self.img_files) == len(self.seq_files), "Mismatch between images and sequences"
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        
        # Load sequence (line-separated integers)
        seq_path = os.path.join(self.seq_dir, self.seq_files[idx])
        with open(seq_path, "r") as f:
            seq = [int(line.strip()) for line in f if line.strip()]  # ignore empty lines
        
        # Pad or truncate
        if len(seq) < self.max_len:
            seq = seq + [3] * (self.max_len - len(seq))  # pad with EOS=3
        else:
            seq = seq[:self.max_len]
        
        seq = torch.tensor(seq, dtype=torch.long)
        
        return img, seq

class PLCtrainer:
    def __init__(self):
        self.epochs = 5
        self.max_len = 1401
        self.bos_token = 4
        self.eos_token = 3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainloader,self.testloader = self.getLoaders()
        self.model = Img2SeqModel(max_len=self.max_len)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=500, 
            num_training_steps= len(self.trainloader)*self.epochs
        )
        self.loss_fn = PLCloss()
        
        self.train_token_acc = []
        self.train_edit_acc = []
        self.val_token_acc = []
        self.val_edit_acc = []

    def getLoaders(self):
        dataset = WebpageDataset(max_len=self.max_len)
        train_dataset, test_dataset = random_split(dataset, [900, 100])
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
        test_loader  = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=2)
        return train_loader,test_loader

    def greedy_decode(self, img):
        batch_size = img.size(0)
        generated = torch.full(
            (batch_size, 1), 
            self.bos_token, 
            dtype=torch.long, 
            device=self.device
        )

        for _ in range(self.max_len-1):
            out = self.model(img, generated)       # [batch, seq_len, vocab_size]
            next_token = out[:, -1, :].argmax(dim=-1)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

            if (next_token == self.eos_token).all():
                break

        return generated

    def greedy_val(self, epoch):
        self.model.eval()
        all_token_acc = []
        all_edit_acc = []

        with torch.no_grad():
            loop = tqdm(self.testloader, desc=f"Val for Epoch {epoch+1}", leave=False)
            for imgs, tgt_seq in loop:
                imgs, tgt_seq = imgs.to(self.device), tgt_seq.to(self.device)
                preds = self.greedy_decode(
                    img=imgs
                )                

                token_acc = self.calcTokenAcc(preds, tgt_seq)

                edit_acc = self.calcEditAcc(preds,tgt_seq)
                all_token_acc.append(token_acc.item())
                all_edit_acc.append(edit_acc)

        return (sum(all_token_acc)/len(all_token_acc)), \
        (sum(all_edit_acc)/len(all_edit_acc))

    def teacherForcedVal(self, epoch):
        self.model.eval()
        all_token_acc = []
        all_edit_acc = []

        with torch.no_grad():
            loop = tqdm(self.testloader, desc=f"Val for Epoch {epoch+1}", leave=False)
            for imgs, tgt_seq in loop:
                imgs, tgt_seq = imgs.to(self.device), tgt_seq.to(self.device)
                preds = self.model(imgs,tgt_seq[:,:-1])                

                token_acc = self.calcTokenAcc(
                    preds=torch.argmax(preds, dim=-1),
                    targets=tgt_seq
                )

                edit_acc = self.calcEditAcc(
                    preds=torch.argmax(preds, dim=-1),
                    targets=tgt_seq
                )
                all_token_acc.append(token_acc.item())
                all_edit_acc.append(edit_acc)

        return (sum(all_token_acc)/len(all_token_acc)), \
        (sum(all_edit_acc)/len(all_edit_acc))

    def calcTokenAcc(self, preds, targets):

        pad_len = self.max_len - preds.size(1)
        preds_pad = F.pad(preds, (0, pad_len), value=3) 
        preds_pad = preds_pad[:, 1:]
        targets_pad = targets[:, 1:]  # (B, 1400)

        mask = (targets_pad != 3)
        
        token_acc = (((preds_pad==targets_pad) & mask).sum() / mask.sum() )*100
            
        return token_acc
    
    def calcEditAcc(self, preds, targets):
        
        preds_trimmed = []
        targets_trimmed = []
        targets = targets[:,1:]
        preds = preds[:,1:]
        batch_size = targets.size()[0]
        for ii in range(batch_size):

            tgt_eos_idxs = (targets[ii] == self.eos_token).nonzero(as_tuple=True)[0]

            if len(tgt_eos_idxs) > 0:
                targets_trimmed.append(targets[ii, :tgt_eos_idxs[0]])
            else:
                targets_trimmed.append(targets[ii])

            pd_eos_idxs = (preds[ii] == self.eos_token).nonzero(as_tuple=True)[0]

            if len(pd_eos_idxs) > 0:
                preds_trimmed.append(preds[ii, :pd_eos_idxs[0]])
            else:
                preds_trimmed.append(preds[ii])


        total_acc = 0
        for p, t in zip(preds_trimmed, targets_trimmed):
            dist = Levenshtein.distance(p.tolist(), t.tolist())
            max_len = max(len(p), len(t))
            acc = 1 - dist/max_len
            total_acc += acc

        return (total_acc / targets.size()[0])*100.0

    def plotTrainValAcc(self):
        epochs = range(1, self.epochs + 1)

        plt.figure(figsize=(8, 5))

        # Token accuracy
        plt.plot(epochs, self.train_token_acc, label="Train Token Acc", marker="o")
        plt.plot(epochs, self.val_token_acc, label="Val Token Acc", marker="s")

        # Edit distance accuracy
        plt.plot(epochs, self.train_edit_acc, label="Train Edit Acc", marker="^")
        plt.plot(epochs, self.val_edit_acc, label="Val Edit Acc", marker="d")

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Training vs Validation Accuracy (Token & Edit Distance)")
        plt.legend()
        plt.grid(True)

        # Save before show
        plt.savefig("TrainValAccuracyPlot.png", dpi=300, bbox_inches="tight")
        plt.show()


    def train(self):
        self.model.to(self.device)
        best_val_score_token = -100
        best_val_score_edit = -100
        for epoch in range(self.epochs):
            self.model.train()
            loop = tqdm(self.trainloader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)
            all_token_acc = []
            all_edit_acc = []
            for imgs, tgt_seq in loop:
                imgs, tgt_seq = imgs.to(self.device), tgt_seq.to(self.device)
                preds = self.model(imgs, tgt_seq[:, :-1])   
                loss = self.loss_fn.genLoss(preds, tgt_seq)  
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                loop.set_postfix(loss=loss.item())
                token_acc = self.calcTokenAcc(
                    preds=torch.argmax(preds, dim=-1),
                    targets=tgt_seq
                )
                edit_acc = self.calcEditAcc(
                    preds=torch.argmax(preds, dim=-1),
                    targets=tgt_seq[:,1:]
                )
                all_token_acc.append(token_acc.item())
                all_edit_acc.append(edit_acc)
            try:
                self.train_token_acc.append(sum(all_token_acc)/len(all_token_acc))  
                self.train_edit_acc.append(sum(all_edit_acc)/len(all_edit_acc))  
            except Exception as e:
                print(f"An error {e}")
            val_scores = self.teacherForcedVal(epoch=epoch)
            self.val_token_acc.append(val_scores[0])
            self.val_edit_acc.append(val_scores[1])

            if val_scores[0] > best_val_score_token or \
            val_scores[1] > best_val_score_edit:
                best_val_score_edit = val_scores[1]
                best_val_score_token = val_scores[0]

                print(f"\nFor epoch: {epoch+1}")
                print(f"Best Token Level Accuracy: {best_val_score_token}")
                print(f"Best Levenshtein Score: {best_val_score_edit}")

        self.endTrainingTasks()

    def endTrainingTasks(self):
        self.plotTrainValAcc()

if __name__ == '__main__':
    plc = PLCtrainer()
    plc.train()
    # exp = PLCloss(compound_loss=2)
    # print(exp.genLoss())
    # edit_acc = plc.calcTokenAcc(torch.ones(2,1400),torch.ones(2,1401))
    # print(edit_acc)
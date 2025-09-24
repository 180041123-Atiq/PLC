from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, infer_device
import torch

device = f"{infer_device}:0"

processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf") 
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    dtype=torch.float16,
    device_map=device
)

# prepare image and text prompt, using the appropriate prompt template
url = "webcode2m_plc/image"
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": url},
            {
                "type": "text", 
                "text": "You are an assistant that converts "+
                "webpage screenshots into clean HTML+inline CSS. "+
                "Output only code in a single markdown code block."
            },
        ],
    },
]
inputs = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
inputs = inputs.to(model.device, torch.float16)

# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(output[0], skip_special_tokens=True))
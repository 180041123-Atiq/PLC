from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from PIL import Image
import torch

processor = AutoProcessor.from_pretrained("tanvirb/websight-7B")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForVision2Seq.from_pretrained(
    "tanvirb/websight-7B",
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True
)

url = "webcode2m_plc/image/0.png" 

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text", 
                "text": "You are an assistant that converts webpage screenshots into clean HTML. "
                        "Output only code in a single markdown code block."
            },
        ],
    },
]

url = "webcode2m_plc/image/0.png"
image = Image.open(url)

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = processor(text=[text], images=[image])

generate_ids = model.generate(inputs.input_ids, max_length=30)

print(processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
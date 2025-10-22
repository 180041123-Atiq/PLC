import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, BitsAndBytesConfig

model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"

# Load processor
processor = AutoProcessor.from_pretrained(model_id)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load 4-bit quantized model with automatic device placement
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True
)

torch.cuda.empty_cache()
print("✅ Model loaded successfully on:", model.device)

url = "webcode2m_plc/image/0.png"   # path to your screenshot

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": url},
            {"type": "text",
             "text": "You are an assistant that converts webpage screenshots into clean HTML+inline CSS. "
                     "Output only code in a single markdown code block."},
        ],
    },
]

# Apply chat template
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

output = model.generate(**inputs, max_new_tokens=30)  # 60 is safe on T4

decoded = processor.decode(output[0], skip_special_tokens=True)
print("\n🔹 Generated Output:\n")
print(decoded)

del model
torch.cuda.empty_cache()
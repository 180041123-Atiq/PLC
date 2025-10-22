from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch

from prompts import gen_layout_prompt

# Load model
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Aquiles-ai/Qwen2.5-VL-3B-Instruct-Img2Code",
#     dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",  # Requires flash-attn
#     device_map="auto"
# )
def gen_qwen_output(index,layout=False):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # Without flash-attn:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Aquiles-ai/Qwen2.5-VL-3B-Instruct-Img2Code",
        quantization_config=bnb_config,
        dtype="auto",
        device_map="auto"
    )

    # Load processor
    processor = AutoProcessor.from_pretrained(
        "Aquiles-ai/Qwen2.5-VL-3B-Instruct-Img2Code",
        min_pixels=256*28*28,
        max_pixels=1280*28*28
    )

    # Load image
    image = Image.open(f"webcode2m_plc/image/{index}.png")

    # Prepare messages

    if layout:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text", 
                        "text": gen_layout_prompt(index)
                    },
                ],
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": "Generate the HTML/CSS code for this webpage screenshot."},
                ],
            }
        ]


    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)

    return output_text


if __name__ == '__main__':
    pass
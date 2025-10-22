from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, TextIteratorStreamer
from qwen_vl_utils import process_vision_info
import torch
from threading import Thread

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Aquiles-ai/Qwen2.5-VL-3B-Instruct-Img2Code",
    quantization_config=bnb_config,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # Requires flash-attn
    device_map="auto"
)

# Without flash-attn:
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Aquiles-ai/Qwen2.5-VL-3B-Instruct-Img2Code",
#     quantization_config=bnb_config,
#     dtype="auto",
#     device_map="auto"
# )

# Load processor
processor = AutoProcessor.from_pretrained(
    "Aquiles-ai/Qwen2.5-VL-3B-Instruct-Img2Code",
    min_pixels=256*28*28,
    max_pixels=1280*28*28
)

def gen_qwen_output(html_image,prompt):
    

    # Load image
    image = html_image

    # Prepare messages

    
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
                    "text": prompt
                },
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

    # # Inference: Generation of the output
    # generated_ids = model.generate(**inputs, max_new_tokens=2048)
    # generated_ids_trimmed = [
    #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    # ]
    # output_text = processor.batch_decode(
    #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    # )
    # # print(output_text)

    # return output_text[0]

    streamer = TextIteratorStreamer(
        processor,
        skip_prompt=True,
        skip_special_tokens=True
    )
    
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=2048
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    print("Generating code:")
    for new_text in streamer:
        print(new_text, end="", flush=True)
    
    thread.join()


if __name__ == '__main__':
    pass
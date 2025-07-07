import sys

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, TextStreamer

max_tokens = 512
default_prompt = f"""
You will act as a remote sensing analyst to describe the content of the image.
Please provide a detailed description of the image in 1 to 3 sentences, including the following aspects:
1. The general content of the image, such as the type of scene (e.g., urban, rural, forest, water body, etc.).
2. The specific objects or features present in the image, such as buildings, roads, vegetation, water bodies, etc.
3. The spatial distribution of the objects or features, such as their arrangement, placements, density, and patterns.
4. Any other relevant information that can help understand the image, such as the time of capture, weather conditions, or any notable geographical feature or events.
Please ensure that your description is clear, concise, and informative, the total words should **not exceed {max_tokens}**.
Do not use the markdown format.
Do not use the bullet points or numbered lists.
Do not include any personal opinions or subjective views.
"""


def test_gemma3_caption():
    """
    Test the Gemma3 model for image captioning.
    """
    # Load the model and processor

    ckpt = "src/stage2/generative/tools/conditions/caption/weights/gemma-3-4b-it"
    model = Gemma3ForConditionalGeneration.from_pretrained(
        ckpt,
        device_map="cuda:1",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).eval()
    processor = AutoProcessor.from_pretrained(ckpt)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "url": "data/Disaterm3/train_images/train_images/beriut_explosion_post_6_9.png",
                },
                {"type": "text", "text": default_prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    streamer = TextStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        stream=sys.stdout,
    )

    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            streamer=streamer,
        )

    # generation = generation[0][input_len:]
    # decoded = processor.decode(generation, skip_special_tokens=True)
    # print(decoded)

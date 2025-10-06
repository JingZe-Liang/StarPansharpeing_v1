import asyncio
import base64
import io
import os
import sys
from collections.abc import Sequence

import numpy as np
import toml
import torch
from openai import AsyncOpenAI
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, TextStreamer
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionAttention

from src.utilities.logging import log

setattr(Qwen2_5_VLVisionAttention, "is_causal", False)
type InputImageType = str | np.ndarray | Image.Image

max_tokens = 300
default_prompt = f"""
You will act as a remote sensing analyst to describe the content of the image.
Please provide a detailed description of the image in 1 to 3 sentences, including the following aspects:
1. The general content of the remote sensing image, such as the type of scene (e.g., urban, rural, forest, water body, etc.).
2. The specific objects or features present in the image, such as buildings, roads, vegetation, water bodies, etc.
3. The spatial distribution of the objects or features, such as their arrangement, placements, density, and patterns.
4. Any other relevant information that can help understand the image, such as the time of capture, weather conditions, or any notable geographical feature or events.
Please ensure that your description is clear, concise, and informative, the total words should **not exceed {max_tokens}**.
The image provided to you is a remote sensing image from a satellite. NOT a normal photo. the photo is taken from a top-down view.
Do not use the markdown format.
Do not use the bullet points or numbered lists.
All reponse and answers must be in English, NO other languages.
Do not include any personal opinions or subjective views. 
"""


def img_to_base64(img: np.ndarray | str | Image.Image) -> str:
    """
    Convert a numpy array or PIL Image to base64 string.
    """
    if isinstance(img, str):
        with open(img, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    elif isinstance(img, Image.Image):
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        return base64.b64encode(img_bytes.getvalue()).decode("utf-8")

    assert img.ndim == 3 and img.dtype == np.uint8, (
        "Image must be a 3D numpy array with dtype uint8."
    )
    img_bytes = io.BytesIO()
    Image.fromarray(img).save(img_bytes, format="PNG")
    img_base64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
    return img_base64


def array_img_to_pil(img: np.ndarray, denorm=False) -> Image.Image:
    """
    Convert a numpy array image to PIL Image.
    """

    # image: 0 .. 1
    if denorm:
        img = (img * 255).astype(np.uint8)

    assert img.ndim == 3 and img.dtype == np.uint8, (
        "Image must be a 3D numpy array with dtype uint8."
    )

    return Image.fromarray(img).convert("RGB")


local_qwen_ckpt = "src/stage2/generative/tools/conditions/caption/weights/Qwen2.5VL"
remote_qwen_ckpt = "Qwen/Qwen2.5-VL-7B-Instruct"


def get_qwen25vl_model(
    ckpt: str = remote_qwen_ckpt,
    max_tokens: int = max_tokens,
    prompt: str = default_prompt,
    encode=True,
    device="cuda",
    stream=False,
):
    # default: Load the model on the available device(s)
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    # )

    from src.stage2.generative.tools.conditions.caption.gemma2_caption_encode import (
        gemma2_caption_encode,
    )

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        ckpt,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        device_map=device,
    ).eval()

    # default processor
    processor = AutoProcessor.from_pretrained(ckpt)
    # The default range for the number of visual tokens per image in the model is 4-16384.
    # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels
    )
    print(f"Loaded Qwen2.5-VL model from {ckpt} and processor")

    text_encode_func = None
    if encode:
        text_encode_func = gemma2_caption_encode(device=device, return_truncated=True)
        print(f"Loaded Gemma2 text encoder from {ckpt}")

    def process_img(img: str | np.ndarray | Image.Image):
        if isinstance(img, np.ndarray):
            # img = array_img_to_base64(img)
            # to PIL Image
            img = array_img_to_pil(img)
        elif isinstance(img, str):
            assert os.path.exists(img), f"Image path {img} does not exist."
        else:
            raise ValueError(f"Invalid image path: {img}")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": img},
                ],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        vision_info = process_vision_info(messages)
        image_inputs = vision_info[0] if len(vision_info) > 0 else None
        video_inputs = vision_info[1] if len(vision_info) > 1 else None
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # Inference: Generation of the output
        streamer = (
            TextStreamer(
                processor.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                stream=sys.stdout,
            )
            if stream
            else None
        )
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs, max_new_tokens=max_tokens, streamer=streamer
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        embeds, attn_mask, valid_length = None, None, None
        if encode and text_encode_func is not None:
            embeds, attn_mask, valid_length = text_encode_func(output_text[0])

        return {
            "caption": output_text[0],
            "caption_feature": embeds,
            "attention_mask": attn_mask,
            "valid_length": valid_length,
        }

    return model, process_img


def get_qwen25vl_max_api(max_current_tasks=5):
    client = AsyncOpenAI(
        api_key=toml.load("env.toml")["DASHSCOPE_API_KEY"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    async def image_captioning(img) -> str:
        try:
            # Encode image to base64
            base64_image = img_to_base64(img)
            response = await client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                            {"type": "text", "text": default_prompt},
                        ],
                    }
                ],
                model="qwen-vl-plus",
                extra_body={"enable_thinking": False},
            )
            content = response.choices[0].message.content
            return content if content is not None else ""
        except Exception as e:
            log(f"Error in image_captioning: {e}", level="error")
            return ""

    semaphore = asyncio.Semaphore(max_current_tasks)

    async def limited_image_captioning(img):
        async with semaphore:
            return await image_captioning(img)

    async def captioning_async_main(images) -> list[str]:
        tasks = [limited_image_captioning(image) for image in images]
        results = await asyncio.gather(*tasks)
        return results

    def process_img(img: InputImageType | Sequence[InputImageType]) -> str | list[str]:
        # Process single image or sequence of images
        if isinstance(img, (tuple, list)):
            # Handle sequence of images
            img_list = []
            for single_img in img:
                if isinstance(single_img, np.ndarray):
                    img_list.append(array_img_to_pil(single_img))
                elif isinstance(single_img, str):
                    assert os.path.exists(single_img), (
                        f"Image path {single_img} does not exist."
                    )
                    img_list.append(single_img)
                elif isinstance(single_img, Image.Image):
                    img_list.append(single_img)
                else:
                    raise ValueError(f"Invalid image type: {type(single_img)}")
        else:
            # Handle single image
            if isinstance(img, np.ndarray):
                img_list = [array_img_to_pil(img)]
            elif isinstance(img, str):
                assert os.path.exists(img), f"Image path {img} does not exist."
                img_list = [img]
            elif isinstance(img, Image.Image):
                img_list = [img]
            else:
                raise ValueError(f"Invalid image type: {type(img)}")

        # Run async captioning for all images at once
        results = asyncio.run(captioning_async_main(img_list))

        # Return single result if input was single image, otherwise return list
        return results[0] if len(img_list) == 1 else results

    return process_img


if __name__ == "__main__":
    import time

    import PIL.Image as Image

    # Initialize the API
    process_img = get_qwen25vl_max_api()

    # Test with single image
    img_path = "data/YuZhongDataset/OpenEarthMap/OpenEarthMap_wo_xBD/abancay/images/abancay_1.tif"
    if os.path.exists(img_path):
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img)

        print("Testing single image...")
        start_time = time.time()
        result = process_img(img_array)
        end_time = time.time()
        print(f"Single image result: {result}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")

        # Test with multiple images (concurrent processing)
        print("\nTesting multiple images (concurrent)...")
        # Explicitly type the list to help type checker
        images: list[np.ndarray] = [img_array] * 3  # Same image 3 times for testing

        start_time = time.time()
        results = process_img(images)
        end_time = time.time()

        for i, res in enumerate(results):
            print(f"Image {i + 1} result: {res}")
        print(f"Time taken for 3 images: {end_time - start_time:.2f} seconds")
        print(f"Average time per image: {(end_time - start_time) / 3:.2f} seconds")
    else:
        print(f"Test image not found: {img_path}")
        print("Please provide a valid image path for testing.")

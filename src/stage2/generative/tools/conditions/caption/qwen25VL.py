import array
import base64
import io
import os
import sys

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, TextStreamer
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionAttention

from src.stage2.generative.tools.conditions.caption.gemma2_caption_encode import (
    gemma2_caption_encode,
)

setattr(Qwen2_5_VLVisionAttention, "is_causal", False)

max_tokens = 300
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


def array_img_to_base64(img: np.ndarray) -> str:
    """
    Convert a numpy array image to base64 string.
    """
    assert (
        img.ndim == 3 and img.dtype == np.uint8
    ), "Image must be a 3D numpy array with dtype uint8."
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

    assert (
        img.ndim == 3 and img.dtype == np.uint8
    ), "Image must be a 3D numpy array with dtype uint8."

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

    if encode:
        text_encode = gemma2_caption_encode(device=device, return_truncated=True)
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
        image_inputs, video_inputs = process_vision_info(messages)
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
        if encode:
            embeds, attn_mask, valid_length = text_encode(output_text[0])

        return {
            "caption": output_text[0],
            "caption_feature": embeds,
            "attention_mask": attn_mask,
            "valid_length": valid_length,
        }

    return model, process_img


if __name__ == "__main__":
    _, inferencer = get_qwen25vl_model()
    import PIL.Image as Image

    # Example usage
    img_path = "data/TEOChatlas/eval/External_images/AID/airport_37.jpg"
    img = Image.open(img_path).convert("RGB")
    img = np.array(img)

    print(inferencer(img_path))

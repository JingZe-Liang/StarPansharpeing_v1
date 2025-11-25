import asyncio
import base64
import io
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import toml
import torch
from openai import AsyncOpenAI
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, TextStreamer
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionAttention

from src.utilities.logging import log
from src.utilities.train_utils.visualization import get_rgb_image

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

    assert img.ndim == 3 and img.dtype == np.uint8, "Image must be a 3D numpy array with dtype uint8."
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

    assert img.ndim == 3 and img.dtype == np.uint8, "Image must be a 3D numpy array with dtype uint8."

    return Image.fromarray(img).convert("RGB")


local_qwen_ckpt = "/Data/ZiHanCao/checkpoints/Qwen/Qwen2___5-VL-7B-Instruct"
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

    text_encode_func = None
    if encode:
        from src.stage2.generative.tools.conditions.caption.gemma2_caption_encode import (
            gemma2_caption_encode,
        )

        text_encode_func = gemma2_caption_encode(device=device, return_truncated=True)
        print(f"Loaded Gemma2 text encoder from {ckpt}")

    def process_img(img: str | np.ndarray | Image.Image):
        if isinstance(img, np.ndarray):
            # img = array_img_to_base64(img)
            # to PIL Image
            img = array_img_to_pil(img, denorm=True)
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
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
            generated_ids = model.generate(**inputs, max_new_tokens=max_tokens, streamer=streamer)

        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
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


def captioning_dataloader_img(dl, process_img, rgb_channels: list[int] | str = "mean"):
    """
    Process images from a dataloader and generate captions.

    Args:
        dl: Dataloader that yields samples with 'img' and '__key__' fields
        process_img: Function to process images and generate captions
        rgb_channels: Channel selection for RGB conversion

    Yields:
        dict: Dictionary containing image id, image data, and caption results
    """

    for sample in dl:
        img = sample["img"]
        assert img.ndim == 4, f"Image batch must be 4D numpy array, got {img.ndim}D."
        assert img.shape[0] == 1, f"Only support batch size 1 for captioning, got {img.shape[0]}."
        img_id = sample["__key__"]

        # to RGB
        if img.ndim == 4 and img.shape[1] == 3:
            img_rgb = img
        else:
            img_rgb = get_rgb_image(img, rgb_channels=rgb_channels, use_linstretch=True)  # (bs, c, h, w)
        img = img_rgb[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 3) numpy array

        results = process_img(img)

        yield {
            "id": img_id,
            "image": img,
            "caption": results["caption"],
            "caption_feature": results["caption_feature"],
            "attention_mask": results["attention_mask"],
            "valid_length": results["valid_length"],
        }


def main_process_dataloader_img(
    dl,
    rgb_channels: list[int] | str = "mean",
    save_dir: str = "tmp/captions_qwen25vl",
    device="cuda",
    file_type: str = "jsonl",
):
    """
    Main function to process images from a dataloader and save captions.

    Args:
        dl: Dataloader that yields samples with 'img' and '__key__' fields
        rgb_channels: Channel selection for RGB conversion
        save_dir: Directory to save caption files
        device: Device to run the model on
        file_type: File type to save captions ('txt' or 'jsonl')
    """
    import os

    try:
        import jsonlines as jsl
    except ImportError:
        raise ImportError("jsonlines package is required for saving captions in jsonl format")

    from loguru import logger
    from tqdm import tqdm

    model, process_img = get_qwen25vl_model(
        ckpt=local_qwen_ckpt,
        max_tokens=max_tokens,
        prompt=default_prompt,
        encode=False,
        device=device,
        stream=True,
    )

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    for res in tqdm(
        captioning_dataloader_img(dl, process_img, rgb_channels=rgb_channels),
        desc="Captioning ...",
        disable=True,
    ):
        img_id = res["id"][0] if isinstance(res["id"], list) else res["id"]
        caption = res["caption"]
        logger.info(f"Image ID: {img_id}\nCaption: {caption}\n" + "-" * 60)

        if file_type == "txt":
            file_path = os.path.join(save_dir, f"{img_id}.txt")
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w") as f:
                f.write(caption)
        elif file_type == "jsonl":
            file_path = os.path.join(save_dir, f"{img_id}.jsonl")
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with jsl.open(file_path, mode="w") as writer:
                writer.write({"id": img_id, "caption": caption})
        else:
            raise ValueError(f"Invalid file type: {file_type}")


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
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
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
                    assert os.path.exists(single_img), f"Image path {single_img} does not exist."
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
    from braceexpand import braceexpand

    # Test dataloader functionality if possible
    from src.data.hyperspectral_loader import get_hyperspectral_dataloaders

    # Test with a sample dataloader
    print("\nTesting dataloader functionality...")
    tar_file = list(braceexpand("data/BigEarthNet_S2/hyper_images/BigEarthNet_data_{0003..0006}.tar"))
    _, dl = get_hyperspectral_dataloaders(
        wds_paths=tar_file,
        batch_size=1,
        num_workers=1,
        to_neg_1_1=False,
        permute=True,
        resample=False,
        per_channel_norm=False,
    )

    print("Successfully created dataloader. Testing captioning...")
    # Use the main_process_dataloader_img function to process the dataloader
    main_process_dataloader_img(
        dl,
        rgb_channels=[3, 2, 1],  # Common RGB channels for satellite imagery
        save_dir="data/BigEarthNet_S2/condition_captions/0003-0006",
        device="cuda" if torch.cuda.is_available() else "cpu",
        file_type="jsonl",
    )
    print("Dataloader captioning test completed.")

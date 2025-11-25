import base64
import io
import os
import sys
from collections.abc import Sequence

import jsonlines as jsl
import numpy as np
import toml
import torch
from loguru import logger
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration, TextStreamer

from src.utilities.logging import log
from src.utilities.train_utils.visualization import get_rgb_image

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

    assert img.ndim == 3 and img.dtype == np.uint8, (
        f"Image must be a 3D numpy array with dtype uint8, but got shape {img.shape} and dtype {img.dtype}."
    )

    return Image.fromarray(img).convert("RGB")


"""
run the command: hf download Qwen/Qwen3-VL-30B-A3B-Instruct
"""

# blob version
local_qwen_ckpt = "src/stage2/generative/tools/conditions/caption/caption_weights/Qwen3vl-30B-A3B"
remote_qwen_ckpt = "Qwen/Qwen3-VL-30B-A3B-Instruct"


def get_qwen3vl_model(
    ckpt: str = remote_qwen_ckpt,
    max_tokens: int = max_tokens,
    prompt: str = default_prompt,
    encode=False,
    device="cuda",
    stream=False,
):
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        ckpt,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device,
    ).eval()

    # default processor
    processor = AutoProcessor.from_pretrained(ckpt)
    # The default range for the number of visual tokens per image in the model is 4-16384.
    # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")

    print(f"Loaded Qwen3-VL model from {ckpt} and processor")

    text_encode_func = None
    if encode:
        from src.stage2.generative.tools.conditions.caption.gemma2_caption_encode import (
            gemma2_caption_encode,
        )

        text_encode_func = gemma2_caption_encode(device=device, return_truncated=True)
        print(f"Loaded Gemma2 text encoder from {ckpt}")

    def process_img(img: str | np.ndarray | Image.Image):
        """
        img: (0, 1) range numpy array or image path or PIL Image
        """
        if isinstance(img, np.ndarray):
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
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
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
    for sample in dl:
        img = sample["img"]  # (B, H, W, C) numpy array
        assert img.ndim == 4, f"Image batch must be 4D numpy array, got {img.ndim}D."
        assert img.shape[0] == 1, f"Only support batch size 1 for captioning, got {img.shape[0]}."
        img_id = sample["__key__"]

        # to RGB
        img_rgb = get_rgb_image(img, rgb_channels=rgb_channels, use_linstretch=True)
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
    save_dir: str = "tmp/captions_qwen3vl",
    device="cuda",
    file_type: str = "txt",
):
    model, process_img = get_qwen3vl_model(
        ckpt=remote_qwen_ckpt,
        max_tokens=max_tokens,
        prompt=default_prompt,
        encode=False,
        device=device,
        stream=True,
    )
    # 1, 2, 4, 7
    # save caption to file
    os.makedirs(save_dir, exist_ok=True)

    for res in tqdm(
        captioning_dataloader_img(dl, process_img, rgb_channels=rgb_channels), desc="Captioning ...", disable=True
    ):
        img_id = res["id"][0]
        caption = res["caption"]
        logger.info(f"Image ID: {img_id}\nCaption: {caption}\n" + "-" * 60)

        if file_type == "txt":
            with open(os.path.join(save_dir, f"{img_id}.txt"), "w") as f:
                f.write(caption)
        elif file_type == "jsonl":
            with jsl.open(os.path.join(save_dir, f"{img_id}.jsonl"), mode="w") as writer:
                writer.write({"id": img_id, "caption": caption})
        else:
            raise ValueError(f"Invalid file type: {file_type}")


if __name__ == "__main__":
    from src.data.hyperspectral_loader import get_hyperspectral_dataloaders

    # Create dataloader - note that get_hyperspectral_dataloaders returns (dataset, dataloader)
    tar_file = "data/BigEarthNet_S2/hyper_images/BigEarthNet_data_0000.tar"
    _, dl = get_hyperspectral_dataloaders(
        wds_paths=tar_file,
        batch_size=1,
        num_workers=1,
        to_neg_1_1=False,
        permute=False,
        resample=False,
        per_channel_norm=False,
    )

    main_process_dataloader_img(dl, [3, 2, 1])

import asyncio
import base64
import io
import math
import os
import sys
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from threading import Thread

import jsonlines as jsl
import numpy as np
import toml
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from loguru import logger
from openai import AsyncOpenAI
from PIL import Image
from qwen_vl_utils import process_vision_info
from torchvision.transforms.functional import InterpolationMode
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    TextIteratorStreamer,
    TextStreamer,
)

from src.utilities.func.redirect_stderr import suppress_stdout_stderr
from src.utilities.logging import log
from src.utilities.train_utils.visualization import get_rgb_image

remote_path = "OpenGVLab/InternVL3_5-8B"
local_path = "src/stage2/generative/tools/conditions/caption/caption_weights/models--OpenGVLab--InternVL3_5-8B/snapshots/9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79"
max_tokens = 300
default_prompt = f"""
You will act as a remote sensing analyst to describe the content of the image.
Please provide a detailed description of the image in 1 to 3 sentences, including the following aspects:
1. The general content of the remote sensing image, such as the type of scene (e.g., urban, rural, forest, water body, etc.).
2. The specific objects or features present in the image, such as buildings, roads, vegetation, water bodies, etc.
3. The spatial distribution of the objects or features, such as their arrangement, placements, density, and patterns.
4. Any other relevant information that can help understand the image, such as the time of capture, weather conditions, or any notable geographical feature or events.
Please ensure that your description is clear, concise, and informative, the total words should **not exceed {max_tokens}**.
The image provided to you is a remote sensing image from a satellite.
NOT a normal photo. the photo is taken from a top-down view.
Do not use the markdown format.
Do not use the bullet points or numbered lists.
All reponse and answers must be in English, NO other languages.
Do not include any personal opinions or subjective views. 
<image>\n
Now, please provide your description based on the image.
"""

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file: np.ndarray | str, input_size=448, max_num=12):
    if isinstance(image_file, str) and os.path.exists(image_file):
        image = Image.open(image_file).convert("RGB")
    else:
        assert isinstance(image_file, np.ndarray), "Invalid image file, got {}".format(
            type(image_file)
        )
        image = array_img_to_pil(image_file, denorm=True)
    # image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def get_intervl35_model(
    ckpt: str = local_path,
    max_tokens: int = max_tokens,
    prompt: str = default_prompt,
    encode=True,
    device="cuda",
    stream=False,
):
    model = (
        AutoModel.from_pretrained(
            ckpt,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        )
        .eval()
        .cuda()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt, trust_remote_code=True, use_fast=False
    )
    logger.info("Loaded model done.")

    def process_img(img):
        pixel_values = load_image(img).to(torch.bfloat16).to(device)

        # Define the generation configuration
        question = prompt

        if stream:
            streamer = TextIteratorStreamer(
                tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=10
            )
            generation_config = dict(
                max_new_tokens=1024,
                do_sample=False,
                streamer=streamer if stream else None,
            )
            # Start the model chat in a separate thread
            thread = Thread(
                target=model.chat,
                kwargs=dict(
                    tokenizer=tokenizer,
                    pixel_values=pixel_values,
                    question=question,
                    history=None,
                    return_history=False,
                    generation_config=generation_config,
                ),
            )
            thread.start()
        else:
            generation_config = dict(
                max_new_tokens=1024,
                do_sample=False,
            )

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            with suppress_stdout_stderr():
                response = model.chat(
                    tokenizer,
                    pixel_values,
                    question,
                    generation_config,
                    history=None,
                    return_history=False,
                )

        gen_txt = ""
        if stream:
            for new_text in streamer:
                if new_text == model.conv_template.sep:
                    break
                gen_txt += new_text
                print(
                    new_text, end="", flush=True
                )  # Print each new chunk of generated text on the same line

        yield {
            "caption": response,
            "valid_length": len(response),
        }

    return model, process_img


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


def captioning_dataloader_img(
    dl, process_img, rgb_channels: list[int] | str = "mean", resume_from=None
):
    """
    Process images from a dataloader and generate captions.

    Args:
        dl: Dataloader that yields samples with 'img' and '__key__' fields
        process_img: Function to process images and generate captions
        rgb_channels: Channel selection for RGB conversion

    Yields:
        dict: Dictionary containing image id, image data, and caption results
    """
    skipped = 0
    processed = 0
    resumed = not resume_from is not None  # str: False; None: True
    for sample in dl:
        img = sample["img"]
        assert img.ndim == 4, f"Image batch must be 4D numpy array, got {img.ndim}D."
        assert img.shape[0] == 1, (
            f"Only support batch size 1 for captioning, got {img.shape[0]}."
        )
        img_id = sample["__key__"]

        if isinstance(resume_from, str):
            if img_id[0] == resume_from:
                resumed = True
            else:
                skipped += 1
                logger.debug(
                    f"Skipping {img_id[0]} until {resume_from}. Skipped {skipped} files."
                )
                yield {
                    "id": img_id,
                    "processed": processed,
                    "skipped": skipped,
                    "is_skipped": True,
                }
                continue
            if resume_from is not None and resumed:
                logger.info(f"Resuming from {img_id[0]}.")
        elif isinstance(resume_from, set):
            img_id_0 = "JPEGImages/" + img_id[0]  # litdata specific
            if img_id_0 in resume_from:
                skipped += 1
                # logger.debug(f"Skipping {img_id[0]}. Skipped {skipped} files.")
                # continue  # skip already processed files
                yield {
                    "id": img_id,
                    "processed": processed,
                    "skipped": skipped,
                    "is_skipped": True,
                }
                continue

        # Convert to RGB
        if img.ndim == 4 and img.shape[1] == 3:
            img_rgb = img
        else:
            img_rgb = get_rgb_image(
                img, rgb_channels=rgb_channels, use_linstretch=True
            )  # (bs, c, h, w)
        img = img_rgb[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 3) numpy array

        results_g = process_img(img)
        results = next(results_g)

        processed += 1

        yield {
            "id": img_id,
            "image": img,
            "caption": results["caption"],
            "valid_length": results["valid_length"],
            "skipped": skipped,
            "processed": processed,
            "is_skipped": False,
        }


def main_process_dataloader_img(
    dl,
    rgb_channels: list[int] | str = "mean",
    save_dir: str = "tmp/captions_qwen25vl",
    device="cuda",
    file_type: str = "jsonl",
    resume_from: set | str | None = None,
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
        raise ImportError(
            "jsonlines package is required for saving captions in jsonl format"
        )

    from loguru import logger
    from tqdm import tqdm

    model, process_img = get_intervl35_model(
        ckpt=local_path,
        max_tokens=max_tokens,
        prompt=default_prompt,
        encode=False,
        device=device,
        stream=False,
    )

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    tbar: tqdm = tqdm(  # type: ignore
        captioning_dataloader_img(
            dl, process_img, rgb_channels=rgb_channels, resume_from=resume_from
        ),
        desc="Captioning ...",
        total=410475,
        disable=False,
    )

    for res in tbar:
        tbar.set_postfix(
            {
                "Processed": res["processed"],
                "Skipped": res["skipped"],
                "Is_Skipped": res["is_skipped"],
            }
        )
        tbar.set_description(f"Id: {res['id'][0]}")

        processed_resumed_n = (
            res["processed"]
            + res["skipped"]
            + (len(resume_from) if isinstance(resume_from, (set, list)) else 0)
        )
        tbar.n = processed_resumed_n
        tbar.refresh()

        if not res["is_skipped"]:
            img_id = res["id"][0] if isinstance(res["id"], list) else res["id"]
            caption = res["caption"]
            # logger.info(f"Image ID: {img_id}\nCaption: {caption}\n" + "-" * 60)

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


if __name__ == "__main__":
    # Test dataloader functionality if possible
    from src.data.hyperspectral_loader import get_hyperspectral_dataloaders

    _resumed = True
    resumed_set = None

    if _resumed:
        saved_resume_path = "data/RemoteSAM270k/RemoteSAM-270K/captions/JPEGImages"
        assert Path(saved_resume_path).exists(), (
            f"Resume path {saved_resume_path} does not exist."
        )
        saved_jsonl_files = list(Path(saved_resume_path).glob("*"))
        # remove extensions
        resumed_set = set(  # 'JPEGImages/xxx'
            "/".join(saved_resume_path.with_suffix("").parts[-2:])
            for saved_resume_path in saved_jsonl_files
        )
        logger.info(f"Already processed files: {len(resumed_set)}")

    # Test with a sample dataloader
    print("\nTesting dataloader functionality...")
    tar_file = "data/RemoteSAM270k/RemoteSAM-270K/RemoteSAM270K.tar"
    litdata_dir = "data2/RemoteSAM270k/LitData_hyper_images"

    if litdata_dir is not None:
        from litdata.streaming import StreamingDataLoader

        from src.data.litdata_hyperloader import ImageStreamingDataset

        ds = ImageStreamingDataset(
            input_dir=litdata_dir,
            resize_before_transform=None,
            to_neg_1_1=False,
            force_to_rgb=True,
        )
        dl = StreamingDataLoader(ds, batch_size=1, num_workers=6, shuffle=False)

    # if os.path.exists(tar_file):
    #     _, dl = get_hyperspectral_dataloaders(
    #         wds_paths=tar_file,
    #         batch_size=1,
    #         num_workers=1,
    #         to_neg_1_1=False,
    #         permute=True,
    #         resample=False,
    #         per_channel_norm=False,
    #         shuffle_size=-1,
    #         transform_prob=0.0,
    #     )
    # else:
    #     raise ValueError(f"Tar file {tar_file} does not exist.")

    print("Successfully created dataloader. Testing captioning...")
    # Use the main_process_dataloader_img function to process the dataloader
    main_process_dataloader_img(
        dl,
        rgb_channels=[0, 1, 2],  # Common RGB channels for satellite imagery
        save_dir="data/RemoteSAM270k/RemoteSAM-270K/captions/tmp",
        # "tmp/internvl35_captions/",
        device="cuda" if torch.cuda.is_available() else "cpu",
        file_type="jsonl",
        resume_from=resumed_set,
    )

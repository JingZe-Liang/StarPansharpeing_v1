import os
import warnings
from itertools import zip_longest
from pathlib import Path
from typing import Any, Dict, List, Literal, Union, cast

import numpy as np
import torch
import webdataset as wds
import wids
from braceexpand import braceexpand
from natsort import natsorted
from PIL import Image
from tqdm import tqdm

from data.hyperspectral_loader import get_hyperspectral_dataloaders
from src.data.codecs import safetensors_codec_io
from src.data.tar_utils import TarSinkManager
from src.stage2.generative.tools.condition_prepare import (
    prepare_condition_from_webdataset,
)
from src.utilities.logging import log_print

warnings.filterwarnings("ignore", module="torch.utils.checkpoint")


def webdataset_conditions_prepare(
    datasets: wds.WebDataset | list,
    base_dir: str,
    tar_name: str,
    nums=None,
    conditions: Union[str, List[str]] = "all",
    rgb_channels: list[int] | None = None,
    device="cuda",
    to_pil: bool = True,
    save_original_rgb: bool = True,
    condition_save_format: Literal["png", "jpg", "safetensors"] = "png",
    caption_save_format: Literal["txt", "json", "safetensors"] = "txt",
):
    """
    Process webdataset to generate condition images and captions.

    Args:
        ds (wds.WebDataset): Input webdataset containing hyperspectral images
        base_dir (str): Base directory for saving output
        tar_name (str): Name of the output tar file
        tar_rel_path (str): Relative path for the tar file
        conditions (str or list): Conditions to generate ("all" or list of specific conditions)
        rgb_channels (list[int], optional): RGB channels to extract from hyperspectral data
        device (str): Device for processing ("cuda" or "cpu")
        to_pil (bool): Whether to convert outputs to PIL images
        save_original_rgb (bool): Whether to save the original RGB image
        condition_save_format (str): Format for saving condition images
        caption_save_format (str): Format for saving captions
    """

    log_print(f"Starting condition preparation for dataset: {tar_name}")
    log_print(f"Output directory: {base_dir}")
    log_print(f"Conditions: {conditions}")
    log_print(f"RGB channels: {rgb_channels}")
    log_print(f"Device: {device}")
    torch.cuda.set_device(device)  # Set the device for torch operations

    total_n = 0
    sink_man = TarSinkManager(base_dir)

    if not isinstance(datasets, list):
        datasets = [datasets]

    if nums is None:
        nums = [None] * len(datasets)

    nums: list[int | None]
    for ds, num in zip_longest(datasets, nums):
        # Wrap the condition preparation generator with tqdm for progress tracking
        condition_generator = prepare_condition_from_webdataset(
            ds,
            conditions,
            rgb_channels,
            device="cuda",
            to_pil=to_pil,  # type: ignore
        )

        # Note: Since we don't know the total count, we'll use an unbounded progress bar
        progress_bar = tqdm(
            condition_generator,
            desc="Processing conditions",
            unit="sample",
            total=num,
        )
        for sample_idx, (sample, condition_data) in enumerate(zip(ds, progress_bar)):
            # Get the sample key from the original webdataset sample
            sample_key = sample.get("__key__", None)
            url = sample.get("__url__", None)
            tar_name = Path(url[0]).name
            tar_rel_path = f"conditions/{tar_name}"
            sink = sink_man.get_sink(tar_name, tar_rel_path)

            assert url is not None, (
                "Sample does not have a URL which should not be happened."
            )
            assert sample_key is not None, (
                "Sample does not have a key which should not be happened."
            )

            # Prepare the output dictionary for webdataset
            assert len(sample_key) == 1, "Sample key must be a single string."
            output_sample = {"__key__": sample_key[0]}

            # Save original RGB image if requested
            if save_original_rgb and "img" in sample:
                img = sample["img"]
                if hasattr(img, "cpu"):  # torch tensor
                    img = img.cpu().numpy()
                if img.ndim == 4:  # Remove batch dimension if present
                    img = img[0]
                if img.ndim == 3 and img.shape[0] > 3:  # Extract RGB channels
                    if rgb_channels is not None:
                        img = img[rgb_channels]
                    else:
                        img = img[:3]  # Take first 3 channels as RGB

                # Convert to HWC format and normalize to [0, 255]
                if img.shape[0] == 3:  # CHW format
                    img = img.transpose(1, 2, 0)  # Convert to HWC
                img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

                # Save as PNG
                rgb_image = Image.fromarray(img)
                output_sample["rgb.png"] = rgb_image

            # Process and save condition images
            for condition_name, condition_output in condition_data.items():
                if condition_name == "caption":
                    # Handle caption separately
                    if caption_save_format == "txt":
                        output_sample[f"{condition_name}.txt"] = str(
                            condition_output
                        ).encode("utf-8")
                    elif caption_save_format == "json":
                        import json

                        output_sample[f"{condition_name}.json"] = json.dumps(
                            {"caption": str(condition_output)}
                        ).encode("utf-8")
                    elif caption_save_format == "safetensors":
                        # Convert caption to tensor (encode as bytes then to tensor)
                        caption_bytes = str(condition_output).encode("utf-8")
                        caption_tensor = torch.frombuffer(
                            caption_bytes, dtype=torch.uint8
                        )
                        output_sample[f"{condition_name}.safetensors"] = (
                            safetensors_codec_io({"caption": caption_tensor})
                        )
                else:
                    # Handle image conditions
                    if condition_save_format == "png":
                        if isinstance(condition_output, Image.Image):
                            output_sample[f"{condition_name}.png"] = condition_output
                        elif isinstance(condition_output, np.ndarray):
                            output_sample[f"{condition_name}.png"] = Image.fromarray(
                                condition_output
                            )
                        else:
                            log_print(
                                f"Unsupported condition output type for {condition_name}: {type(condition_output)}",
                                level="warning",
                            )
                    elif condition_save_format == "jpg":
                        if isinstance(condition_output, Image.Image):
                            output_sample[f"{condition_name}.jpg"] = condition_output
                        elif isinstance(condition_output, np.ndarray):
                            output_sample[f"{condition_name}.jpg"] = Image.fromarray(
                                condition_output
                            )
                    elif condition_save_format == "safetensors":
                        if isinstance(condition_output, Image.Image):
                            condition_array = np.array(condition_output)
                        elif isinstance(condition_output, np.ndarray):
                            condition_array = condition_output
                        else:
                            log_print(
                                f"Cannot convert {condition_name} to array for safetensors",
                                level="warning",
                            )
                            continue

                        # Convert numpy array to torch tensor
                        condition_tensor = torch.from_numpy(condition_array)
                        output_sample[f"{condition_name}.safetensors"] = (
                            safetensors_codec_io({condition_name: condition_tensor})
                        )

            # Write the processed sample to tar
            if output_sample:
                sink.write(output_sample)
                total_n += 1

                # Update progress bar description periodically
                progress_bar.set_postfix({"samples_processed": total_n})
                progress_bar.set_description(f"tar: {tar_name}")
                # if total_n % 100 == 0:
                #     progress_bar.set_postfix({"samples_processed": total_n})
                #     log_print(f"Processed {total_n} samples.")
            else:
                log_print(
                    f"Empty output for sample {sample_key}, skipping.", level="warning"
                )

            # log_print(f"{total_n} samples written to {base_dir}/{tar_rel_path}")
        progress_bar.close()

    # Always close the sink manager
    sink_man.close_all()

    log_print(f"Finished processing.")
    return total_n


# --- Main entry with Hydra support --- #

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf


def main_with_hydra_config(cfg: DictConfig) -> None:
    """
    Main function for condition preparation with Hydra configuration.

    Expected config structure:
    data:
      wds_paths: [...] # WebDataset paths
      # other data configuration
    output:
      output_dir: str
      tar_name: str
      tar_rel_path: str
    processor:
      conditions: "all" or list
      rgb_channels: list or null
      device: str
      to_pil: bool
      save_original_rgb: bool
      condition_save_format: str
      caption_save_format: str
    """
    log_print("Starting condition preparation with configuration:")
    log_print(OmegaConf.to_yaml(cfg, resolve=True))

    # Setup WebDataset pipeline

    # Get the dataloader
    if "_target_" in cfg.data:
        nums = None
        _, input_dataloader = hydra.utils.instantiate(cfg.data)
        log_print(f"Setting up WebDataset pipeline: {cfg.data.wds_paths}")

    elif isinstance(cfg.data, (str, list, ListConfig)):
        if isinstance(cfg.data, str):
            if Path(cfg.data).is_dir():
                cfg.data = [str(p) for p in Path(cfg.data).glob("*.tar")]

        input_dataloader = []
        nums = []
        data_s: list[str] = (
            list(braceexpand(cfg.data)) if isinstance(cfg.data, str) else cfg.data  # type: ignore
        )
        data_s = natsorted(data_s)  # Sort the dataset paths naturally
        log_print(f"Set up WebDataset pipeline: {data_s}")
        for path in data_s:
            log_print(f"Computing smaples for {path}...")
            num = wids.wids.compute_num_samples(path)
            nums.append(num)
            log_print(f"<green>Number of samples in {path}: {num}</>")

            _, dl = get_hyperspectral_dataloaders(
                path,
                batch_size=1,
                num_workers=0,
                shuffle_size=-1,
                to_neg_1_1=False,
                transform_prob=0.0,
                resample=False,
                permute=False,
            )
            input_dataloader.append(dl)

    else:
        raise ValueError(f"cfg.data must be a dict or a list, got {type(cfg.data)}")

    # Run condition preparation
    total_samples = webdataset_conditions_prepare(
        datasets=input_dataloader,
        base_dir=cfg.output.output_dir,
        tar_name=cfg.output.tar_name,
        nums=nums,
        conditions=cfg.processor.get("conditions", "all"),
        rgb_channels=cfg.processor.get("rgb_channels", None),
        device=cfg.processor.get("device", "cuda"),
        to_pil=cfg.processor.get("to_pil", True),
        save_original_rgb=cfg.processor.get("save_original_rgb", True),
        condition_save_format=cfg.processor.get("condition_save_format", "png"),
        caption_save_format=cfg.processor.get("caption_save_format", "txt"),
    )

    log_print(
        f"Condition preparation completed. Total samples processed: {total_samples}"
    )


def main_with_args():
    """
    Main function for condition preparation with command line arguments.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Generate conditions from webdataset")
    parser.add_argument(
        "--wds_paths",
        type=str,
        nargs="+",
        required=True,
        help="WebDataset tar file paths",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for condition files",
    )
    parser.add_argument(
        "--tar_name", type=str, required=True, help="Output tar file name"
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default="all",
        help="Conditions to generate (comma-separated or 'all')",
    )
    parser.add_argument(
        "--rgb_channels",
        type=int,
        nargs=3,
        default=None,
        help="RGB channels to extract (3 integers)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device for processing"
    )
    parser.add_argument(
        "--condition_save_format",
        type=str,
        default="png",
        choices=["png", "jpg", "safetensors"],
        help="Format for saving condition images",
    )
    parser.add_argument(
        "--caption_save_format",
        type=str,
        default="txt",
        choices=["txt", "json", "safetensors"],
        help="Format for saving captions",
    )
    parser.add_argument(
        "--no_original_rgb", action="store_true", help="Don't save original RGB images"
    )

    args = parser.parse_args()

    # Create simple webdataset
    import webdataset as wds

    ds = wds.WebDataset(args.wds_paths).decode("torch")

    # Parse conditions
    if args.conditions == "all":
        conditions = "all"
    else:
        conditions = [c.strip() for c in args.conditions.split(",")]

    # Run condition preparation
    total_samples = webdataset_conditions_prepare(
        datasets=ds,
        base_dir=args.output_dir,
        tar_name=args.tar_name,
        conditions=conditions,
        rgb_channels=args.rgb_channels,
        device=args.device,
        to_pil=True,
        save_original_rgb=not args.no_original_rgb,
        condition_save_format=args.condition_save_format,
        caption_save_format=args.caption_save_format,
    )

    log_print(
        f"Condition preparation completed. Total samples processed: {total_samples}"
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--hydra":
        # Remove the --hydra flag and run with Hydra
        sys.argv = [sys.argv[0]] + sys.argv[2:]

        @hydra.main(
            config_path="../scripts/configs/condition_preparation",
            config_name="hyperspectral_full",
            version_base=None,
        )
        def hydra_main(cfg: DictConfig) -> None:
            from loguru import logger

            with logger.catch():
                main_with_hydra_config(cfg)

        hydra_main()
    else:
        # Run with command line arguments
        from loguru import logger

        with logger.catch():
            main_with_args()

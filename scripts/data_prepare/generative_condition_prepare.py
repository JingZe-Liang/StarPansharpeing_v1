import json
import os
import warnings
from itertools import zip_longest
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence, Union, cast

import numpy as np
import torch
import webdataset as wds
import wids
from braceexpand import braceexpand
from litdata.streaming.writer import BinaryWriter
from natsort import natsorted
from PIL import Image
from tqdm import tqdm

from data.tar_utils import TarSinkManager
from src.data.codecs import rgb_codec_io, safetensors_codec_io
from src.data.hyperspectral_loader import get_hyperspectral_dataloaders
from src.stage2.generative.tools.condition_prepare import (
    prepare_condition_from_webdataset,
)
from src.utilities.logging import log_print

warnings.filterwarnings("ignore", module="torch.utils.checkpoint")


def dir_save_sample(sink: str, sample: dict[str, Any]):
    img_name = sample["__key__"]
    if isinstance(img_name, list):
        assert len(img_name) == 1, "Image name list must contain a single element."
        img_name = img_name[0]

    for k, v in sample.items():
        save_path = os.path.join(sink, f"{img_name}.{k}")
        if k.endswith((".jpeg", ".png", ".jpg")):
            if isinstance(v, Image.Image):
                v.save(save_path)
            elif isinstance(v, bytes):
                with open(save_path, "wb") as f:
                    f.write(v)
        elif k.endswith((".txt", ".json")):
            # Handle both string and bytes data
            if isinstance(v, str):
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(v)
            elif isinstance(v, bytes):
                with open(save_path, "wb") as f:
                    f.write(v)
        # else:
        #     log_print(f"Unknown file type: {k}", level="warning")


def prepare_fn(
    datasets,
    nums,
    conditions,
    rgb_channels,
    relative_data_dir,
    sink_man: TarSinkManager | BinaryWriter | str,
    resume_from: str | int | None = None,
    save_original_rgb: bool = False,
    to_pil: bool = False,
    condition_save_format: Literal["png", "jpg", "safetensors"] = "png",
    caption_save_format: Literal["txt", "json"] = "json",
    save_attn_mask: bool = False,
    use_linstretch=False,
    dataset_type: str = "webdataset",
):
    total_n = 0
    for ds, num in zip_longest(datasets, nums):
        # Wrap the condition preparation generator with tqdm for progress tracking
        condition_generator = prepare_condition_from_webdataset(
            ds,
            conditions,
            rgb_channels,
            device="cuda",
            to_pil=to_pil,  # type: ignore
            resume_from=resume_from,
            use_linstretch=use_linstretch,
        )

        # Note: Since we don't know the total count, we'll use an unbounded progress bar
        progress_bar = tqdm(
            condition_generator,
            desc="Processing conditions",
            unit="sample",
            total=num,
        )

        rel_data_dir = relative_data_dir if resume_from is None else f"{relative_data_dir}_resumed"
        for sample_idx, (sample, condition_data) in enumerate(progress_bar):
            if condition_data is None:
                continue

            # Get the sample key from the original webdataset sample
            if dataset_type == "webdataset":
                sample_key = sample.get("__key__", None)
                url = sample.get("__url__", None)
                tar_name = Path(url[0]).name
                tar_rel_path = f"{rel_data_dir}/{tar_name}"
                # Type assertion: sink_man must be TarSinkManager for webdataset
                assert not isinstance(sink_man, (str, BinaryWriter)), "sink_man must be TarSinkManager for webdataset"
                sink = sink_man.get_sink(tar_name, tar_rel_path)

                assert url is not None, "Sample does not have a URL which should not be happened."
                assert sample_key is not None, "Sample does not have a key which should not be happened."
                # Prepare the output dictionary for webdataset
                assert len(sample_key) == 1, "Sample key must be a single string."
                output_sample = {"__key__": sample_key[0]}
            elif dataset_type == "litdata":
                # sink: BinaryWriter = sink_man
                sink = cast(BinaryWriter, sink_man)
                sample_key = sample.get("__key__", str(total_n))
                output_sample = {"__key__": sample_key}
            elif dataset_type == "dir":
                # save in dir
                sink = cast(str, os.path.join(sink_man, rel_data_dir))
                os.makedirs(sink, exist_ok=True)
                assert isinstance(sink, str), "For 'dir' dataset_type, sink_man must be a directory path string."
                sample_key = sample.get("__key__", str(total_n))
                output_sample = {"__key__": sample_key}
            else:
                raise ValueError(f"Unsupported dataset type: {dataset_type}")

            # Save original RGB image if requested
            if save_original_rgb and "img" in sample:
                img = sample["img"]
                if hasattr(img, "cpu"):  # torch tensor
                    img = img.cpu().numpy()
                if img.ndim == 4:  # Remove batch dimension if present
                    img = img[0]
                if img.ndim == 3 and img.shape[0] > 3:  # Extract RGB channels
                    if isinstance(rgb_channels, (list, tuple)):
                        img = img[rgb_channels]
                    elif rgb_channels == "mean":
                        c_3 = img.shape[0] // 3
                        bands = [img[i * c_3 : (i + 1) * c_3, :, :].mean(0) for i in range(3)]
                        img = np.stack(bands, axis=0)
                    else:
                        img = img[:3]  # Take first 3 channels as RGB

                # Convert to HWC format and normalize to [0, 255]
                if img.shape[0] == 3:  # CHW format
                    img = img.transpose(1, 2, 0)  # Convert to HWC
                img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

                # Save as PNG
                rgb_image = Image.fromarray(img)
                if dataset_type == "webdataset":
                    output_sample["rgb.jpg"] = rgb_image
                elif dataset_type == "litdata_bin":
                    output_sample["rgb"] = rgb_image
                elif dataset_type == "dir":
                    output_sample["rgb.jpg"] = rgb_image

            # > Process and save condition images or captions
            for condition_name, condition_output in condition_data.items():
                ######### Captions ###########
                if condition_name == "caption":
                    condition_output = cast(Dict[str, Any], condition_output)
                    valid_length = condition_output["valid_length"]

                    # Handle caption separately
                    if caption_save_format == "txt":
                        output_sample[f"{condition_name}.txt"] = str(condition_output["caption"]).encode("utf-8")
                    elif caption_save_format == "json":
                        output_sample[f"{condition_name}.json"] = json.dumps(
                            {
                                "caption": str(condition_output["caption"]),
                                "valid_length": str(valid_length),
                            }
                        ).encode("utf-8")
                    else:
                        raise ValueError(f"Unsupported condition type: {condition_name}")

                    embeds, mask = (
                        condition_output["caption_feature"],
                        condition_output["attention_mask"],
                    )
                    if embeds is not None and mask is not None:
                        saved_name = "features.safetensors"
                        # trunc the caption embed
                        saved = {"caption_feature": embeds.to(torch.bfloat16)}
                        if save_attn_mask:
                            saved_name = "features_and_mask.safetensors"
                            saved["attention_mask"] = torch.as_tensor(mask).to(torch.uint8)
                        output_sample[saved_name] = safetensors_codec_io(saved)

                ########## image conditions ###########
                else:
                    if isinstance(condition_output, np.ndarray) and to_pil:
                        condition_output = Image.fromarray(condition_output)

                    # Handle image conditions
                    if condition_name != "segmentation":
                        if to_pil:
                            condition_output = condition_output.convert("L")
                        else:
                            condition_output = Image.fromarray(condition_output).convert("L")
                            condition_output = np.array(condition_output)

                    if condition_save_format == "png":
                        if dataset_type in ("webdataset", "dir"):
                            key_ = f"{condition_name}.png"
                        else:
                            key_ = condition_name
                        output_sample[key_] = (
                            condition_output
                            if isinstance(condition_output, Image.Image)
                            else Image.fromarray(condition_output)
                        )
                    elif condition_save_format == "jpg":
                        if isinstance(condition_output, Image.Image):
                            # compression jpeg quality
                            if dataset_type in ("webdataset", "dir"):
                                key_ = f"{condition_name}.jpg"
                            else:
                                key_ = condition_name
                            output_sample[key_] = rgb_codec_io(
                                img=np.array(condition_output)
                                if isinstance(condition_output, Image.Image)
                                else condition_output,
                                format="jpeg",
                                quality=85,  # Default JPEG quality
                            )

            # Write the processed sample to tar
            if output_sample:
                if dataset_type == "webdataset":
                    sink.write(output_sample)
                    progress_bar.set_description(f"tar: {tar_name}")
                elif dataset_type in ("litdata", "litdata_bin"):
                    sink.add_item(total_n, output_sample)
                else:  # dir
                    # save items separately
                    assert isinstance(sink, str), "For 'dir' dataset_type, sink must be a directory path string."
                    dir_save_sample(sink, output_sample)

                total_n += 1

                # Update progress bar description periodically
                progress_bar.set_postfix({"samples_processed": total_n})
            else:
                log_print(f"Empty output for sample {sample_key}, skipping.", level="warning")

        progress_bar.close()
        if dataset_type == "litdata":
            sink.done()
            sink.merge()
            log_print("Finished writing litdata binary.", level="info")

    return total_n


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
    caption_save_format: Literal["txt", "json"] = "json",
    resume_from: str | int | None = None,
    relative_data_dir: str = "conditions",
    save_attn_mask: bool = False,
    use_linstretch=False,
    dataset_type: str = "webdataset",
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

    # lazy load
    from src.data.tar_utils import TarSinkManager

    log_print(f"Starting condition preparation for dataset: {tar_name}")
    log_print(f"Output directory: {base_dir}")
    log_print(f"Conditions: {conditions}")
    log_print(f"RGB channels: {rgb_channels}, use linstretch: {use_linstretch}")
    log_print(f"Device: {device}")
    torch.cuda.set_device(device)  # Set the device for torch operations

    if dataset_type == "litdata":
        sink_man = BinaryWriter(base_dir, chunk_bytes="512Mb")
    elif dataset_type == "webdataset":
        sink_man = TarSinkManager(base_dir)
    elif dataset_type == "dir":
        sink_man = base_dir  # type: ignore
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    if not isinstance(datasets, list):
        datasets = [datasets]

    if nums is None or isinstance(nums, int):
        nums = [None] * len(datasets)

    nums = cast(list[int | None], nums)
    total_n = 0
    # > do the condition preparation
    try:
        total_n = prepare_fn(
            datasets=datasets,
            nums=nums,
            conditions=conditions,
            rgb_channels=rgb_channels,
            relative_data_dir=relative_data_dir,
            sink_man=sink_man,
            resume_from=resume_from,
            save_original_rgb=save_original_rgb,
            to_pil=to_pil,
            condition_save_format=condition_save_format,
            caption_save_format=caption_save_format,
            save_attn_mask=save_attn_mask,
            use_linstretch=use_linstretch,
            dataset_type=dataset_type,
        )
    except Exception as e:
        log_print(f"Error: {e}", level="critical")
        raise RuntimeError(f"Condition preparation failed: {e}")
    finally:
        if dataset_type == "litdata":
            log_print(f"Finalizing litdata binary writer ...")
        elif dataset_type == "webdataset":
            log_print(f"Closing all sinks ...")
            sink_man.close_all()
        else:
            log_print(f"Stop processing.")

    log_print(f"Finished processing.")
    return total_n


# * --- Main entry with Hydra support --- #

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
    log_print(OmegaConf.to_yaml(cfg))

    # Setup WebDataset pipeline

    # Get the dataloader
    if "_target_" in cfg.data:
        input_dataloader = hydra.utils.instantiate(cfg.data)
        nums = None
        try:
            nums = len(input_dataloader)
        except:
            pass
        # log_print(f"Setting up dataset pipeline: {cfg.data.wds_paths}")

    elif isinstance(cfg.data, (str, list, ListConfig)):
        # TODO: add litdata dataset support

        if isinstance(cfg.data, str):
            if Path(cfg.data).is_dir():
                cfg.data = [str(p) for p in Path(cfg.data).glob("*.tar")]

        input_dataloader = []
        data_s: list[str] = (
            list(braceexpand(cfg.data)) if isinstance(cfg.data, str) else cfg.data  # type: ignore
        )
        data_s = natsorted(data_s)  # Sort the dataset paths naturally

        nums = [] if cfg.processor.count_tar_num else None
        log_print(f"Set up dataset pipeline: {data_s}")
        for path in data_s:
            if cfg.processor.count_tar_num:
                nums = cast(list[int], nums)
                nums.append(len(list(Path(path).glob("*.tar"))))
                log_print(f"Computing number of samples for {path}...")
                num = wids.wids.compute_num_samples(path)
                nums.append(num)
                log_print(f"<green>Number of samples in {path}: {num}</>")

            _, dl = get_hyperspectral_dataloaders(
                path,
                batch_size=1,
                num_workers=cfg.loader.num_workers,
                img_key=cfg.loader.img_key,
                tgt_key=cfg.loader.tgt_key,
                shuffle_size=-1,
                to_neg_1_1=False,
                transform_prob=0.0,
                resample=False,
                permute=getattr(cfg.loader, "permute", True),
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
        resume_from=cfg.processor.get("resume_from", None),
        relative_data_dir=cfg.processor.get("relative_data_dir", "conditions"),
        use_linstretch=cfg.processor.get("use_linstretch", False),
        dataset_type=cfg.processor.get("dataset_type", "webdataset"),
    )

    log_print(f"Condition preparation completed. Total samples processed: {total_samples}")


if __name__ == "__main__":
    # Hydra condition preparation
    from loguru import logger

    @hydra.main(
        config_path="../configs/condition_preparation",
        config_name="hyperspectral_full",
        version_base=None,
    )
    def hydra_main(cfg: DictConfig) -> None:
        with logger.catch():
            main_with_hydra_config(cfg)

    hydra_main()

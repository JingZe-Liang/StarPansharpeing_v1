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


def prepare_fn(
    datasets,
    nums,
    conditions,
    rgb_channels,
    relative_data_dir,
    sink_man: TarSinkManager | BinaryWriter,
    resume_from: str | None = None,
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
                sink = sink_man.get_sink(tar_name, tar_rel_path)

                assert url is not None, "Sample does not have a URL which should not be happened."
                assert sample_key is not None, "Sample does not have a key which should not be happened."
                # Prepare the output dictionary for webdataset
                assert len(sample_key) == 1, "Sample key must be a single string."
                output_sample = {"__key__": sample_key[0]}
            elif dataset_type == "litdata":
                sink: BinaryWriter = sink_man
                sample_key = sample.get("__key__", str(total_n))
                output_sample = {"__key__": sample_key}
            else:
                raise ValueError(f"Unsupported dataset type: {dataset_type}")

            # > Save original RGB image if requested
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

            # > Process and save condition images or captions
            for condition_name, condition_output in condition_data.items():
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

                # * image conditions
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
                        if dataset_type == "webdataset":
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
                            if dataset_type == "webdataset":
                                key_ = f"{condition_name}.jpg"
                            else:
                                key_ = condition_name
                            output_sample[key_] = rgb_codec_io(
                                img=np.array(condition_output)
                                if isinstance(condition_output, Image.Image)
                                else condition_output,
                                format="jpeg",
                                quality=80,  # Default JPEG quality
                            )

                    # elif condition_save_format == "safetensors":
                    #     if isinstance(condition_output, Image.Image):
                    #         condition_array = np.array(condition_output)
                    #     elif isinstance(condition_output, np.ndarray):
                    #         condition_array = condition_output
                    #     else:
                    #         log_print(
                    #             f"Cannot convert {condition_name} to array for safetensors",
                    #             level="warning",
                    #         )
                    #         continue

                    #     # Convert numpy array to torch tensor
                    #     condition_tensor = torch.from_numpy(condition_array)
                    #     output_sample[f"{condition_name}.safetensors"] = (
                    #         safetensors_codec_io({condition_name: condition_tensor})
                    #     )

            # Write the processed sample to tar
            if output_sample:
                if dataset_type == "webdataset":
                    sink.write(output_sample)
                    progress_bar.set_description(f"tar: {tar_name}")
                else:
                    sink.add_item(total_n, output_sample)
                total_n += 1

                # Update progress bar description periodically
                progress_bar.set_postfix({"samples_processed": total_n})
            else:
                log_print(f"Empty output for sample {sample_key}, skipping.", level="warning")

        progress_bar.close()
        if dataset_type == "litdata":
            sink.done()
            sink.merge()
            logger.info("Finished writing litdata binary.")

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
    resume_from: str | None = None,
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
    else:
        sink_man = TarSinkManager(base_dir)

    if not isinstance(datasets, list):
        datasets = [datasets]

    if nums is None:
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
    finally:
        if dataset_type == "litdata":
            log_print(f"Finalizing litdata binary writer ...")
        else:
            log_print(f"Closing all sinks ...")
            sink_man.close_all()

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
        nums = None
        _, input_dataloader = hydra.utils.instantiate(cfg.data)
        log_print(f"Setting up dataset pipeline: {cfg.data.wds_paths}")

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
    parser.add_argument("--tar_name", type=str, required=True, help="Output tar file name")
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
    parser.add_argument("--device", type=str, default="cuda", help="Device for processing")
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
    parser.add_argument("--no_original_rgb", action="store_true", help="Don't save original RGB images")

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

    log_print(f"Condition preparation completed. Total samples processed: {total_samples}")


# * --- utilities --- #

from tarfile import TarFile

import natsort
from rich.progress import Progress

from src.data.tar_utils import extract_tar_files_safe, read_tar_filenames_safe
from src.utilities import logging


def list_conditions_grouped(
    file_dir: str | None = None,
    file_tar: str | None = None,
    condition_n=5,
    check_file=True,
):
    if file_dir is not None:
        file_list = [p.stem for p in Path(file_dir).glob("*")]
    elif file_tar is not None:
        file_list = read_tar_filenames_safe(file_tar, close_tar=True, check_file=check_file)
        assert isinstance(file_list, list), "Expected a list of file names from tar."
    else:
        raise ValueError("Either file_dir or file_tar must be provided.")

    # sort
    if file_dir is not None:
        print(f"file dir is provided, not tar file, the list of files should be sorted.")
        file_list = natsort.natsorted(file_list)
        print("sorted done.")

    grouped_files: list[list[str]] = []
    group: list[str] = []
    _last_group_name = None

    for file in (tbar := tqdm(file_list)):
        stem_name = file.split(".")[0]
        if _last_group_name is None or stem_name != _last_group_name:
            if _last_group_name is not None:
                grouped_files.append(group)
            group = [file]
            # print(f"new group: {stem_name}, groups found: {len(grouped_files)}")
            tbar.set_postfix({"group": len(grouped_files)})
            _last_group_name = stem_name
        else:
            group.append(file)

    if len(group) == condition_n:
        grouped_files.append(group)

    # check the groups
    need_fixed_groups = []
    for group in grouped_files:
        if len(group) != condition_n:
            log_print(
                f"Group {group[0]} has {len(group)} files, expected {condition_n}.",
                "warning",
            )
            need_fixed_groups.append(group)

    log_print(f"{len(need_fixed_groups)} groups need to be fixed.")
    if len(need_fixed_groups) > 0:
        with open("need_fixed_groups.txt", "w") as f:
            for group in need_fixed_groups:
                for file in group:
                    f.write(str(file) + "\n")

    return grouped_files


def list_tar_hyper_images(
    tar_file: str,
    finds: list[str] | None = None,
    sort: bool = True,
    stop_until_find: bool = True,
):
    # names = tar.getnames()
    names = read_tar_filenames_safe(tar_file, close_tar=True)
    assert isinstance(names, list), "Expected a list of file names from tar."
    print(f"Total files in tar: {len(names)}")

    find_indices_map = {}
    if finds is not None:
        for i, n in enumerate(names):
            n = n.split(".")[0]
            if n in finds:
                print(f"Found {n} in tar file with index {i}")
                find_indices_map[n] = i

    if stop_until_find and finds:
        # finds_list = list(find_indices_map.keys())
        find_indices = list(find_indices_map.values())
        # not include min found index
        print(
            f"Stop until find {min(find_indices_map, key=find_indices_map.get)} "  # type: ignore
            f"in tar file with index {min(find_indices)}"
        )
        names = names[: min(find_indices)] if find_indices else names

    file_list = [Path(p).stem for p in names]

    if sort:
        file_list = natsort.natsorted(file_list)
        print("sorted done.")

    return file_list, names


def comp_conditions_hyper_images_names(
    tar_file: str,
    condition_dir: str | None = None,
    condition_tar: str | None = None,
    condition_n=4,
):
    print(">>> listing conditions ...")
    groups = list_conditions_grouped(
        condition_dir,
        condition_tar,
        condition_n=condition_n,
        check_file=True,  # no check condition file in tar
    )

    print(">>> listing hyper images ...")
    tar_name, _ = list_tar_hyper_images(tar_file, sort=False)

    print(f"\n\nTar file contains {len(tar_name)} images.")
    print(f"Condition directory contains {len(groups)} groups of images.\n\n")

    for i, (tname, group) in enumerate(zip_longest(tar_name, groups)):
        tname = str(tname).split(".")[0] if tname else None
        gname = str(group[0]).split(".")[0] if group else None
        if tname != gname:
            print(
                f"Mismatch: Tar name {tname} does not match group name {gname} at index [{i}/{max(len(tar_name), len(groups))}]"
            )
            print("------------------")
        # else:
        #     print(f"Match: Tar name {tname} matches group name {gname}")

    print(">>> Comparison completed.\n")


def re_tar_from_dir(
    src_img_tar: str,
    conditions_dir,
    output_file,
    condition_names=["hed", "segmentation", "sketch", "mlsd"],
):
    assert output_file.endswith(".tar"), "Output file must be a .tar file"
    stems = [p.split(".")[0] for p in read_tar_filenames_safe(src_img_tar, close_tar=True, progress=True)]

    output_tar = TarFile(output_file, "w")
    for stem in (tbar := tqdm(stems, desc="Re-tarring conditions")):
        for c_name in condition_names:
            condition_file = Path(conditions_dir, f"{stem}.{c_name}.png")

            if not condition_file.exists():
                tbar.clear()
                print(f"Condition file {condition_file} does not exist, skipping.")
                tbar.refresh()
                continue

            output_tar.add(condition_file, arcname=f"{stem}.{c_name}.png")
            tbar.set_postfix({"file": stem})

    # > compare the keys
    output_tar.close()
    print(f"Re-tar completed. Output file: {output_file}")
    comp_conditions_hyper_images_names(src_img_tar, condition_tar=output_file)


def concate_tars(*src_tars, output_tar: str, repeat_find=True):
    from src.data.tar_utils import extract_tar_files_safe, read_tar_filenames_safe

    n_total = 0
    if repeat_find:
        s = set()
    with TarFile(output_tar, "w") as out_tar:
        for tar_file in src_tars:
            assert Path(tar_file).exists(), f"Tar file {tar_file} does not exist."
            tar = TarFile(tar_file, "r")
            log_print("merged tar:{}".format(tar.name))

            tar_files = read_tar_filenames_safe(tar_path=tar_file, close_tar=True)
            tbar: tqdm = tqdm(
                enumerate(extract_tar_files_safe(tar=tar, close_tar=False)),
                total=len(tar_files),
                desc="Merging tar members",
            )
            for i, (member, file_data) in tbar:
                if repeat_find:
                    if member.name in s:
                        log_print(
                            f"Skipping duplicate member {member.name} in tar {tar.name}",
                            "warning",
                        )
                        continue
                    else:
                        s.add(member.name)

                try:
                    out_tar.addfile(member, file_data)
                    n_total += 1
                    tbar.set_description(f"Extract {member.name}")
                except Exception as e:
                    log_print(f"Failed to add {member.name} to output tar: {e}", "error")
                    continue
            tar.close()

    log_print(f"Concatenated {len(src_tars)} tar files into {output_tar}, total {n_total} files.")


if __name__ == "__main__":
    # > hydra or args condition preparation
    import sys

    from loguru import logger

    if len(sys.argv) > 1 and sys.argv[1] == "--hydra":
        # Remove the --hydra flag and run with Hydra
        sys.argv = [sys.argv[0]] + sys.argv[2:]

        @hydra.main(
            config_path="../configs/condition_preparation",
            config_name="hyperspectral_full",
            version_base=None,
        )
        def hydra_main(cfg: DictConfig) -> None:
            with logger.catch():
                main_with_hydra_config(cfg)

        hydra_main()
    else:
        # Run with command line arguments
        with logger.catch():
            main_with_args()

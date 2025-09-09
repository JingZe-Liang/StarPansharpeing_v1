import os
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import webdataset as wds

from src.data.codecs import (
    img_decode_io,
    mat_decode_io,
    npz_decode_io,
    safetensors_decode_io,
    tiff_decode_io,
    wids_image_decode,
)
from src.data.multimodal_loader import MultimodalityDataloader
from src.data.utils import norm_img_ as norm_img_fn
from src.data.utils import permute_img_to_chw, remove_extension, to_tensor
from src.utilities.config_utils import function_config_to_basic_types
from src.utilities.logging import log_print

SUPPORTED_SATELLITES = {"WV2", "WV3", "WV4", "IKONOS"}


# * --- Webdataset dataloader --- #


def img_dict_mapper_with_ext(
    sample: dict, to_neg_1_1: bool = True, latent_ext="safetensors"
):
    # keys: ['hrms', 'lrms', 'pan', 'ms', 'hrms_latent', 'lrms_latent', 'pan_latent', 'ms_latent']
    hrms = permute_img_to_chw(to_tensor(sample["hrms"]))
    lrms = permute_img_to_chw(to_tensor(sample["lrms"]))
    pan = permute_img_to_chw(to_tensor(sample["pan"]))  # h, w
    if pan.ndim == 2:
        pan = pan[None]  # add channel dim

    # normalize
    hrms, lrms, pan = map(
        partial(
            norm_img_fn,
            to_neg_1_1=to_neg_1_1,
            per_channel=False,
        ),
        [hrms, lrms, pan],
    )

    # if has latents
    if "latents" in sample or "hrms_latent" in sample:
        if latent_ext == "npz":
            sample_dict = sample["latents"]
            hrms_latent, lrms_latent, pan_latent = map(
                to_tensor,
                [
                    sample_dict["hrms_latent"],
                    sample_dict["lrms_latent"],
                    sample_dict["pan_latent"],
                ],
            )
        elif latent_ext == "safetensors":
            sample_dict = sample["latents"]
            hrms_latent = sample_dict["hrms_latent"].float()
            lrms_latent = sample_dict["lrms_latent"].float()
            pan_latent = sample_dict["pan_latent"].float()
        else:
            sample_dict = sample["latents"]
            hrms_latent, lrms_latent, pan_latent = map(
                to_tensor,
                [
                    sample["hrms_latent"],
                    sample["lrms_latent"],
                    sample["pan_latent"],
                ],
            )

        sample_new = {
            "hrms": hrms,
            "lrms": lrms,
            "pan": pan,
            "hrms_latent": hrms_latent,
            "lrms_latent": lrms_latent,
            "pan_latent": pan_latent,
        }

    else:
        sample_new = {"hrms": hrms, "lrms": lrms, "pan": pan}
    sample.update(sample_new)

    return sample


def satellite_name_add(sample: dict):
    url = sample["__url__"]
    file_stem = Path(url).stem
    sat_name = file_stem.split("_")[-1]
    assert sat_name in SUPPORTED_SATELLITES, (
        f"Satellite name {sat_name} not in supported list {SUPPORTED_SATELLITES}"
    )
    sample["satellite"] = sat_name
    return sample


@function_config_to_basic_types
def get_pansharp_lantent_dataloader(
    wds_paths: str | list[str],
    batch_size: int,
    num_workers: int,
    shuffle_size: int = 100,
    to_neg_1_1: bool = True,
    resample: bool = True,
    latent_ext: Literal["safetensors", "npy", "npz"] = "safetensors",
    shardshuffle: bool = False,
    add_satellite_name=False,
):
    """
    Compact with script 'scripts/data_prepare/pansharpening_data_simulation.py'

    When using as the pansharpening trainer dataloader, assume that we aleady normalize the images.
    """

    if isinstance(wds_paths, str):
        wds_paths = [wds_paths]

    dataset = wds.WebDataset(
        wds_paths,
        resampled=resample,
        shardshuffle=shardshuffle,
        handler=wds.warn_and_continue,
        nodesplitter=wds.shardlists.split_by_node,
        workersplitter=wds.split_by_worker,
        seed=2025,
        verbose=True,
        detshuffle=True if shuffle_size > 0 else False,
    )

    dataset = dataset.decode(
        wds.handle_extension("tif tiff", tiff_decode_io),
        wds.handle_extension("jpg png jpeg", img_decode_io),
        wds.handle_extension("mat", mat_decode_io),
        wds.handle_extension("safetensors", safetensors_decode_io),
        wds.handle_extension("npz", npz_decode_io),
        "torch",
    )

    # * --- dict mapper of images and latents ---

    dataset = dataset.map(remove_extension)
    dataset = dataset.map(
        partial(img_dict_mapper_with_ext, to_neg_1_1=to_neg_1_1, latent_ext=latent_ext)
    )
    if add_satellite_name:
        dataset = dataset.map(satellite_name_add)

    # since we do not use any transforms, the batch and unbatch is useless.

    dataloader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=None if num_workers == 0 else 6,
        drop_last=False,
    )

    dataloader = dataloader.with_length(10_000)  # 10k pairs of the dataloader

    log_print(f"[Pansharpening Dataset]: constructed the dataloader")

    return dataset, dataloader


# * --- Wids dataloader --- #


@function_config_to_basic_types
def get_wids_mat_full_resolution_dataloder(
    wds_paths: dict[Literal["lrms", "pan"], str],
    batch_size: int,
    num_workers: int = 0,
    shuffle_size: int = 0,
    to_neg_1_1: bool = False,
    **__discarded_kwargs,
):
    """
    Left tokenizer encoding in the trainer (on-the-fly encoding).
    full-resolution pansharpening only.
    """
    img_decode_fn = partial(
        wids_image_decode,
        to_neg_1_1=to_neg_1_1,
        mat_is_single_img=True,
        process_img_keys="ALL",
    )
    pan_decode_fn = partial(
        wids_image_decode,
        to_neg_1_1=to_neg_1_1,
        mat_is_single_img=True,
        process_img_keys="ALL",
        permute=False,
    )
    codecs = {"lrms": img_decode_fn, "pan": pan_decode_fn}

    # compactibility with pansharpening simulator
    codecs["img"] = img_decode_fn

    dataset, dataloader = MultimodalityDataloader.create_loader(
        wds_paths,
        to_neg_1_1=to_neg_1_1,
        permute=True,
        return_classed_dict=True,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle_size=shuffle_size,
        codecs=codecs,
    )

    return dataset, dataloader


# * --- Test --- #


def test_wds_reduced_resolution_loading():
    ds, dl = get_pansharp_lantent_dataloader(
        "data/WorldView3/pansharpening_reduced/Pansharpening_WV3.tar",
        batch_size=2,
        shuffle_size=0,
        num_workers=0,
        add_satellite_name=True,
    )
    for sample in dl:
        print(sample)


def test_mat_full_resolution_loading():
    wds_paths = {
        "lrms": "data/WorldView4/pansharpening_full/MS_shardindex.json",
        "pan": "data/WorldView4/pansharpening_full/PAN_shardindex.json",
    }
    dataset, dataloader = get_wids_mat_full_resolution_dataloder(
        wds_paths, 2, 0, to_neg_1_1=True
    )
    for sample in dataloader:
        print(sample.keys())


if __name__ == "__main__":
    # test_mat_full_resolution_loading()
    test_wds_reduced_resolution_loading()

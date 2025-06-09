import sys
from functools import partial
from typing import Literal

import numpy as np
import torch
import webdataset as wds

from src.data.codecs import (
    mat_decode_io,
    npz_decode_io,
    safetensors_decode_io,
    tiff_decode_io,
)
from src.data.utils import (
    flatten_sub_dict,
    merge_modalities,
    norm_img,
    remove_extension,
    remove_meta_data,
)
from src.utilities.logging import log_print


def get_panshap_dataloaders(
    wds_paths: str | list[str],
    batch_size: int,
    num_workers: int,
    shuffle_size: int = 100,
    to_neg_1_1: bool = True,
    resample: bool = True,
    latent_ext: Literal["safetensors", "npy", "npz"] = "safetensors",
    shardshuffle: bool = True,
):
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
        wds.handle_extension("mat", mat_decode_io),
        wds.handle_extension("safetensors", safetensors_decode_io),
        wds.handle_extension("npz", npz_decode_io),
        "torch",
    )

    # * --- dict mapper of images and latents ---

    def img_dict_mapper_with_ext(sample: dict):
        # keys: ['hrms', 'lrms', 'pan', 'ms', 'hrms_latent', 'lrms_latent', 'pan_latent', 'ms_latent']
        hrms = torch.as_tensor(sample["hrms.tiff"]).float()
        lrms = torch.as_tensor(sample["lrms.tiff"]).float()
        pan = torch.as_tensor(sample["pan.tiff"]).float()

        # normalize
        hrms = hrms / hrms.max()
        lrms = lrms / lrms.max()
        pan = pan / pan.max()

        if to_neg_1_1:
            hrms = hrms * 2 - 1
            lrms = lrms * 2 - 1
            pan = pan * 2 - 1

        # if has latents
        if f"hrms_latents.{latent_ext}" in sample or f"hrms_latent.npy" in sample:
            if latent_ext == "npz":
                sample_dict = sample["latenets.npz"]
                hrms_latent = torch.as_tensor(
                    sample_dict["hrms_latent"], dtype=torch.float32
                )
                lrms_latent = torch.as_tensor(
                    sample_dict["lrms_latent"], dtype=torch.float32
                )
                pan_latent = torch.as_tensor(
                    sample_dict["pan_latent"], dtype=torch.float32
                )
            elif latent_ext == "safetensors":
                sample_dict = sample["latents.safetensors"]
                hrms_latent = sample_dict["hrms_latent"].float()
                lrms_latent = sample_dict["lrms_latent"].float()
                pan_latent = sample_dict["pan_latent"].float()
            else:
                sample_dict = sample["latents.npz"]
                hrms_latent = torch.as_tensor(
                    sample["hrms_latent"], dtype=torch.float32
                )
                lrms_latent = torch.as_tensor(
                    sample["lrms_latent"], dtype=torch.float32
                )
                pan_latent = torch.as_tensor(sample["pan_latent"], dtype=torch.float32)

            sample = {
                "hrms": hrms,
                "lrms": lrms,
                "pan": pan,
                "hrms_latent": hrms_latent,
                "lrms_latent": lrms_latent,
                "pan_latent": pan_latent,
            }

        else:
            sample = {
                "hrms": hrms,
                "lrms": lrms,
                "pan": pan,
            }
        return sample

    dataset = dataset.map(img_dict_mapper_with_ext)
    dataset: wds.WebDataset

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


def get_multimodal_dataloaders(
    wds_paths: str | list[str],
    batch_size: int,
    num_workers: int,
    shuffle_size: int = 100,
    to_neg_1_1: bool = True,
    resample: bool = True,
    prefetch_factor: int = 6,
    latent_ext: Literal["safetensors", "npy", "npz"] = "safetensors",
    pin_memory: bool = True,
    remove_meta: bool = False,
    normed_keys: list[str] = ["img", "hrms", "lrms", "pan"],
) -> tuple[wds.DataPipeline, wds.WebLoader]:
    dataset = wds.DataPipeline(
        wds.ResampledShards(wds_paths) if resample else wds.SimpleShardList(wds_paths),
        merge_modalities,
        wds.decode(
            wds.handle_extension("tif tiff", tiff_decode_io),
            wds.handle_extension("npz", npz_decode_io),
            wds.handle_extension("safetensors", safetensors_decode_io),
            "torch",
        ),
        wds.map(remove_extension),
        wds.map(flatten_sub_dict(regardless_of_any_collisions=True)),
        wds.map(
            partial(
                norm_img,
                to_neg_1_1=to_neg_1_1,
                norm_keys=normed_keys,
                permute=False,
            )
        ),
        wds.shuffle(shuffle_size),
        # wds.batched(batch_size, partial=True),
    )
    if remove_meta:
        dataset = dataset.append(wds.map(remove_meta_data))

    dataloader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=False,
        pin_memory=pin_memory,
    )
    dataloader = dataloader.with_length(10_000)  # 10k pairs of the dataloader
    # dataloader = dataloader.unbatched()

    # if shuffle_size > 0:
    #     dataloader = dataloader.shuffle(shuffle_size)
    # dataloader = dataloader.batched(batch_size)

    log_print(f"[Pansharpening Dataset]: constructed the dataloader")

    return dataset, dataloader


if __name__ == "__main__":
    # _, loader = get_panshap_dataloaders(
    #     wds_paths=[
    #         "data/pansharpening/MMSeg_YREB/dataset_0000.tar",
    #         # "data/pansharpening/MMSeg_YREB/dataset_0001.tar",
    #     ],
    #     batch_size=4,
    #     num_workers=1,
    #     latent_ext="npz",
    # )
    # for i, sample in enumerate(loader):
    #     # print(sample)
    #     # break
    #     print(i)

    _, loader = get_multimodal_dataloaders(
        wds_paths=[
            "data/MMSeg_YREB/[hyper_images,pansharpening_pairs,latents]/MMSeg_YREB_train_part-12_bands-MSI-{0000..0003}.tar"
        ],
        batch_size=32,
        num_workers=3,
        resample=True,
        prefetch_factor=6,
        pin_memory=False,
        shuffle_size=-1,
    )

    from tqdm import tqdm

    for sample in tqdm(loader):
        print(sample.keys())
        # pass

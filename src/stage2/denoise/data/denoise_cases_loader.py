from functools import partial

import accelerate
import numpy as np
import torch
import webdataset as wds
import wids
from beartype import beartype

from src.data.codecs import tiff_decode_io
from src.data.utils import norm_img, remove_meta_data


@beartype
def create_denoise_WDC_dataset(
    wds_paths: str | list[str],
    batch_size: int,
    num_workers: int,
    shuffle_size: int = 0,
    to_neg_1_1: bool = False,  # downstream task does not need to norm image to (-1, 1)
    resample: bool = True,
    pin_memory: bool = False,
    shuffle: bool = False,
    remove_meta: bool = False,
    prefetch_factor: int | None = None,
):
    dataset = wds.WebDataset(
        wds_paths,
        resampled=resample,  # no need `iter(dataloader)` for `next` function
        shardshuffle=False,
        nodesplitter=wds.shardlists.single_node_only
        if not accelerate.state.PartialState().use_distributed
        else wds.shardlists.split_by_node,  # split_by_node if is multi-node training
        workersplitter=wds.shardlists.split_by_worker,
        seed=2025,
        verbose=True,
        detshuffle=True if shuffle_size > 0 else False,
        empty_check=False,
        handler=wds.warn_and_continue,
    )
    dataset = dataset.decode(
        *[wds.handle_extension("clean noisy", tiff_decode_io), "torch"],
        handler=wds.reraise_exception,
    )
    dataset = dataset.map(
        partial(norm_img, to_neg_1_1=to_neg_1_1, norm_keys=["clean", "noisy"])
    )
    if remove_meta:
        dataset = dataset.map(remove_meta_data)

    if shuffle and shuffle_size > 0:
        dataset = dataset.shuffle(shuffle_size)
    dataloader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=pin_memory,
        drop_last=False,
        shuffle=False,
    )

    return dataset, dataloader


# * --- Test --- #


def test_loader():
    ds, dl = create_denoise_WDC_dataset(
        "data/Downstreams/WDCDenoise/denoise_cases/WDC_191_bands_px_192-case1.tar",
        batch_size=1,
        num_workers=0,
        shuffle_size=0,
        resample=True,
        to_neg_1_1=False,
        remove_meta=False,
    )

    print("=== Testing Denoise DataLoader ===")
    for i, batch in enumerate(dl):
        img_noisy = batch["noisy"]
        img_clean = batch["clean"]
        print(f"Batch {i} for dataset {batch['__url__'][0]}:")
        print(
            f"  Noisy image shape: {img_noisy.shape}, dtype: {img_noisy.dtype}, min: {img_noisy.min()}, max: {img_noisy.max()}"
        )
        print(
            f"  Clean image shape: {img_clean.shape}, dtype: {img_clean.dtype}, min: {img_clean.min()}, max: {img_clean.max()}"
        )
        if i >= 2:  # Only test first 3 samples
            break


if __name__ == "__main__":
    test_loader()

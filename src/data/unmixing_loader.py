from functools import partial
from typing import Literal, Sequence, cast

import accelerate
import numpy as np
import torch
import webdataset as wds
from torch import Tensor

from src.data.utils import norm_img, remove_extension, remove_meta_data


def get_unmixing_dataloader(
    wds_paths: str | list[str],
    batch_size: int,
    num_workers: int,
    shuffle_size: int = 0,
    to_neg_1_1: bool = True,
    resample: bool = True,
    pin_memory=True,
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
    dataset = dataset.decode("torch", handler=wds.warn_and_continue)
    dataset = dataset.map(remove_extension)
    dataset = dataset.map(partial(norm_img, clip_zero=False, to_neg_1_1=to_neg_1_1))

    dataloader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        shuffle=False,
    )

    return dataset, dataloader


# * --- Test --- * #


def test_loader():
    path = "data/UrbanUnmixing/Urban_188_em4_init.tar"
    ds, dl = get_unmixing_dataloader(path, 1, 0)
    for sample in dl:
        # print(sample.keys())
        for k, v in sample.items():
            if not k.startswith("__"):
                print(f"{k}: {v.shape}")
        break


if __name__ == "__main__":
    test_loader()

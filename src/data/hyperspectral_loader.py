import io

import tifffile
import torch
import webdataset as wds
from accelerate.state import PartialState


def tiff_decoder(key, x):
    if key.endswith(".tiff"):
        return tifffile.imread(io.BytesIO(x))
    else:
        return x


def get_dict_tensor_mapper(to_neg_1_1=True):
    def wds_to_dict_tensor_mapper(sample):
        img = torch.as_tensor(sample["img.tiff"]).float()
        img = img / img.max()
        if to_neg_1_1:
            img = img * 2 - 1
        img = img.permute(-1, 0, 1)

        return {"img": img}

    return wds_to_dict_tensor_mapper


def get_hyperspectral_dataloaders(
    wds_paths: str | list[str],
    batch_size: int,
    num_workers: int,
    shuffle_size: int = 100,
    to_neg_1_1: bool = True,
):
    dict_mapper = get_dict_tensor_mapper(to_neg_1_1)

    part_state = PartialState()
    is_ddp = part_state.use_distributed

    dataset = wds.WebDataset(
        wds_paths,
        resampled=True,
        shardshuffle=True if is_ddp else False,
        cache_size=shuffle_size,
        nodesplitter=wds.shardlists.split_by_node
        if is_ddp
        else wds.shardlists.single_node_only,
        seed=2025,
        verbose=True,
        detshuffle=True if shuffle_size > 0 else False,
    )
    dataset = dataset.decode(tiff_decoder)
    dataset = dataset.map(dict_mapper)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=6,
        drop_last=False,
    )

    return dataset, dataloader

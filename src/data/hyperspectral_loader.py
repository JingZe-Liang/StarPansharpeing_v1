import io
from typing import Sequence

import accelerate
import numpy as np
import tifffile
import torch
import torch.distributed
import webdataset as wds
from accelerate.state import PartialState
from kornia.augmentation import (
    AugmentationSequential,
    RandomBoxBlur,
    RandomChannelShuffle,
    RandomCutMixV2,
    RandomHorizontalFlip,
    RandomRotation,
    RandomSharpness,
    RandomVerticalFlip,
)
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from loguru import logger


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


class HyperRandomGrayScale(IntensityAugmentationBase2D):
    def __init__(self, p=0.5):
        super().__init__(p)

    def apply_transform(self, input, params, flags, transform=None):
        assert input.ndim == 4
        c = input.shape[1]
        gray = input.mean(dim=1, keepdim=True).repeat_interleave(c, dim=1)
        return gray


def hyper_transform(
    op_list: tuple[str],
    probs: tuple[float] | float = 0.5,
    random_apply: int | tuple[int] = 2,
):
    if isinstance(probs, float):
        probs = [probs] * len(op_list)
    assert len(probs) == len(op_list)

    _op_list_cls = dict(
        grayscale=lambda p: HyperRandomGrayScale(p=p),
        channel_shuffle=lambda p: RandomChannelShuffle(p=p),
        sharpness=lambda p: RandomSharpness(p=p, sharpness=[0.5, 1.0]),
        rotation=lambda p: RandomRotation((-30, 30), p=p),
        horizontal_flip=lambda p: RandomHorizontalFlip(p=p),
        vertical_flip=lambda p: RandomVerticalFlip(p=p),
        cutmix=lambda p: RandomCutMixV2(num_mix=1, p=p, cut_size=(0.4, 0.6)),
        blur=lambda p: RandomBoxBlur((3, 3), p=p),
    )

    ops = []
    for op_str, prob in zip(op_list, probs):
        op = _op_list_cls[op_str]
        ops.append(op(prob))

    op_seq = AugmentationSequential(
        *ops,
        data_keys=["input"],
        random_apply=tuple(random_apply)
        if isinstance(random_apply, Sequence)
        else random_apply,
        same_on_batch=False,
        keepdim=True,
    )

    def dict_mapper(sample):
        sample = op_seq(sample)
        return sample

    return dict_mapper


def get_hyperspectral_dataloaders(
    wds_paths: str | list[str],
    batch_size: int,
    num_workers: int,
    shuffle_size: int = 100,
    to_neg_1_1: bool = True,
    hyper_transforms_lst: tuple[str] = (
        "grayscale",
        "channel_shuffle",
        "rotation",
        "cutmix",
        "horizontal_flip",
        "vertical_flip",
    ),
    transform_prob: tuple[float] | float = 0.2,
    random_apply: int | tuple[int] = 1,
):
    dict_mapper = get_dict_tensor_mapper(to_neg_1_1)
    use_transf = (
        hyper_transforms_lst is not None
        and len(hyper_transforms_lst) > 0
        and transform_prob > 0
    )
    if use_transf:
        transform = hyper_transform(hyper_transforms_lst, transform_prob, random_apply)
        logger.info(
            f"[HyperWebdataset]: use augmentations {hyper_transforms_lst} with prob {transform_prob}"
        )

    part_state = PartialState()
    is_ddp = part_state.use_distributed
    if is_ddp:
        logger.info("[HyperWebdataset]: using DDP, split the datset by node")

    dataset = wds.WebDataset(
        wds_paths,
        resampled=True,  # no need `iter(dataloader)` for `next` function
        shardshuffle=shuffle_size if is_ddp else False,
        nodesplitter=wds.shardlists.split_by_node
        if is_ddp
        else wds.shardlists.single_node_only,
        seed=2025,
        verbose=True,
        detshuffle=True if shuffle_size > 0 else False,
    )
    dataset = dataset.decode(tiff_decoder)
    dataset = dataset.map(dict_mapper)
    dataset = dataset.batched(batch_size)
    if use_transf:
        dataset = dataset.map_dict(img=transform)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=6,
        drop_last=False,
    )

    # unbatch, shuffle, and rebatch within different workers
    dataloader = dataloader.unbatched()
    if shuffle_size > 0:
        dataloader = dataloader.shuffle(shuffle_size)
    dataloader = dataloader.batched(batch_size)

    logger.info(
        f"[HyperDataset]: batch size: {batch_size}, num workers: {num_workers}, use transformations: {hyper_transforms_lst}"
    )

    return dataset, dataloader


def get_fast_test_hyperspectral_data(
    data_type: str = "DCF",
    batch_size: int = 1,
):
    """
    get a test data for model/module/function testing.
    """
    wds_paths = {
        "DCF": "data/DCF_2019_Track_2-8_bands-px_512-MSI-0000.tar",
    }[data_type]

    _, dataloader = get_hyperspectral_dataloaders(
        wds_paths,
        batch_size=batch_size,
        num_workers=1,
        shuffle_size=100,
        to_neg_1_1=True,
    )

    return dataloader


if __name__ == "__main__":
    # Test config
    test_wds_path = [
        "data/DCF_2019_Track_2-8_bands-px_512-MSI-0000.tar",
        "data/DCF_2019_Track_2-8_bands-px_512-MSI-0001.tar",
        "data/DCF_2019_Track_2-8_bands-px_512-MSI-0002.tar",
        "data/DCF_2019_Track_2-8_bands-px_512-MSI-0003.tar",
        "data/DCF_2019_Track_2-8_bands-px_512-MSI-0004.tar",
        "data/DCF_2019_Track_2-8_bands-px_512-MSI-0005.tar",
        "data/DCF_2019_Track_2-8_bands-px_512-MSI-0006.tar",
    ]
    test_batch_size = 32
    test_num_workers = 2
    test_shuffle_size = 300
    accelerator = accelerate.Accelerator()

    # Get test dataloader
    test_dataset, test_loader = get_hyperspectral_dataloaders(
        wds_paths=test_wds_path,
        batch_size=test_batch_size,
        num_workers=test_num_workers,
        shuffle_size=test_shuffle_size,
        to_neg_1_1=True,
        transform_prob=1.0,
        random_apply=(1, 2),
    )

    import matplotlib.pyplot as plt
    import torchvision.utils

    # * for loop the images
    for batch in test_loader:
        img_tensor = batch["img"]  # shape: [N, C, H, W]
        print(f"proc={torch.distributed.get_rank()} - {img_tensor.shape}")

    # * plot the grid of images
    # # Get a batch of data
    # batch = next(iter(test_loader))
    # img_tensor = batch["img"]  # shape: [N, C, H, W]

    # # Select [4,2,0] channels for RGB visualization for all images in the batch
    # rgb_imgs = img_tensor[:, [4, 2, 0], :, :]

    # # Normalize from [-1, 1] to [0, 1]
    # rgb_imgs = (rgb_imgs + 1) / 2

    # # Create image grid using make_grid
    # grid = torchvision.utils.make_grid(rgb_imgs, nrow=8, padding=2, normalize=False)

    # # Convert to numpy and adjust dimension order for plotting
    # grid_img = grid.permute(1, 2, 0).numpy()

    # # Display and save the grid
    # plt.figure(figsize=(15, 15))
    # plt.imshow(grid_img)
    # plt.axis("off")
    # plt.savefig("multispectral_grid.png", bbox_inches="tight", pad_inches=0, dpi=300)
    # plt.close()

    # print(f"Batch shape: {img_tensor.shape}")
    # print(f"Range: min={img_tensor.min():.2f}, max={img_tensor.max():.2f}")
    # print(f"Data type: {img_tensor.dtype}")

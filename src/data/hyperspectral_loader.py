import io

import tifffile
import torch
import webdataset as wds
from accelerate.state import PartialState
from kornia.augmentation import (
    AugmentationSequential,
    RandomChannelShuffle,
    RandomCutMixV2,
    RandomRotation,
    RandomSharpness,
)
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D


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
    op_list: list[str], probs: list[float] | float = 0.5, random_apply: int = 2
):
    if isinstance(probs, float):
        probs = [probs] * len(op_list)
    assert len(probs) == len(op_list)

    _op_list_cls = dict(
        grayscale=lambda p: HyperRandomGrayScale(p=p),
        channel_shuffle=lambda p: RandomChannelShuffle(p=p),
        sharpness=lambda p: RandomSharpness(p=p, sharpness=[0.5, 1.0]),
        rotation=lambda p: RandomRotation((-30, 30), p=p),
        cutmix=lambda p: RandomCutMixV2(num_mix=1, p=p),
    )

    ops = []
    for op_str, prob in zip(op_list, probs):
        op = _op_list_cls[op_str]
        ops.append(op(prob))

    op_seq = AugmentationSequential(
        *ops, data_keys=["input"], random_apply=random_apply
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
    hyper_transforms_lst: list[str] = [
        "grayscale",
        "channel_shuffle",
        "sharpness",
        "rotation",
        "cutmix",
    ],
    transform_prob: list[float] | float = 0.2,
    random_apply: int = 2,
):
    dict_mapper = get_dict_tensor_mapper(to_neg_1_1)

    part_state = PartialState()
    is_ddp = part_state.use_distributed
    transform = hyper_transform(hyper_transforms_lst, transform_prob, random_apply)

    dataset = wds.WebDataset(
        wds_paths,
        resampled=True,
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

    dataloader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=6,
        drop_last=False,
    )
    dataloader = dataloader.map_dict(img=transform)

    return dataset, dataloader


if __name__ == "__main__":
    # Test config
    test_wds_path = [
        "/HardDisk/ZiHanCao/datasets/Multispectral_webdatasets/MMSeg_YREB_train_part-12_bands-MSI-0000.tar",
    ]
    test_batch_size = 32
    test_num_workers = 2
    test_shuffle_size = 300

    # Get test dataloader
    test_dataset, test_loader = get_hyperspectral_dataloaders(
        wds_paths=test_wds_path,
        batch_size=test_batch_size,
        num_workers=test_num_workers,
        shuffle_size=test_shuffle_size,
        to_neg_1_1=True,
    )

    # Test multiple batches
    num_batches_to_test = 5
    for i, batch in enumerate(test_loader):
        if i >= num_batches_to_test:
            break
        img_tensor = batch["img"]
        print(f"\nBatch {i+1}:")
        print(f"Shape: {img_tensor.shape}")
        print(f"Value range: min={img_tensor.min():.2f}, max={img_tensor.max():.2f}")
        print(f"Data type: {img_tensor.dtype}")

import torch
from torch import Tensor
from kornia.augmentation import (
    AugmentationSequential,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomVerticalFlip,
    RandomBrightness,
    RandomGamma,
)
from kornia.constants import DataKey


def rgb_nir_transform(p, data_keys: list[str | DataKey] = ["input", "input"], size=(512, 512)):
    return AugmentationSequential(
        RandomHorizontalFlip(p=p),
        RandomVerticalFlip(p=p),
        RandomResizedCrop(p=p, size=size, scale=(0.8, 1.2), ratio=(0.8, 1.2)),
        RandomBrightness(p=p, brightness=(0.6, 1.6)),
        RandomGamma(gamma=(0.5, 2.0), gain=(1.5, 1.5), p=p),
        data_keys=data_keys,  # type: ignore
        keepdim=True,
    )
    return


class CRTransformInterface:
    def __class_getitem__(cls): ...

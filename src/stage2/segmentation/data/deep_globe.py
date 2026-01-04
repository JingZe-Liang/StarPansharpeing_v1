import numpy as np
import torch
from torch import Tensor
from litdata import StreamingDataLoader, StreamingDataset
from kornia.augmentation import (
    AugmentationSequential,
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomBoxBlur,
    Normalize,
)
from functools import partial
from src.data import _BaseStreamingDataset


class DeepGlobeRoadExtractionStreamingDataset(_BaseStreamingDataset):
    def __init__(
        self,
        img_size: int = 512,
        augmentation_prob=0.5,
        norm_mean_std=[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],  # to (-1, 1)
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.img_size = img_size

        # Define Augmentations
        self.resize = AugmentationSequential(
            Normalize(mean=norm_mean_std[0], std=norm_mean_std[1]),
            RandomResizedCrop(
                size=(img_size, img_size),
                scale=(0.5, 1.0),
                ratio=(3 / 4, 4 / 3),
                p=1.0,
                keepdim=True,
            ),
            data_keys=["input", "mask"],
            keepdim=True,
        )

        self.transforms = None
        if augmentation_prob > 0:
            self.transforms = AugmentationSequential(
                RandomHorizontalFlip(p=augmentation_prob),
                RandomVerticalFlip(p=augmentation_prob),
                random_apply=1,
                data_keys=["input", "mask"],
                keepdim=True,
            )
            # Disable auto conversion to ensure we handle tensors directly
            if hasattr(self.transforms, "_disable_features"):
                self.transforms._disable_features = True

    def _apply_transforms(self, img: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Apply transforms to image and mask."""
        # Kornia automatically applies Nearest Neighbor interpolation for "mask" data key
        img, mask = self.resize(img, mask)
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
        return img, mask

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        img = sample["img"]
        mask = sample.get("mask")

        # 1. Image Preprocessing
        if not isinstance(img, Tensor):
            img = torch.tensor(img)

        # Assuming img is 0-255 uint8, convert to float 0-1
        img = img.float() / 255.0

        # 2. Mask Preprocessing
        if mask is not None:
            if not isinstance(mask, Tensor):
                mask = torch.tensor(mask)

            # Binarize: 0 stays 0, 255 becomes 1
            mask = (mask >= 128).long()

            if mask.ndim == 2:
                mask = mask.unsqueeze(0)

            # mask is confirmed to be in CHW format
            if mask.ndim == 3 and mask.shape[0] > 1:
                mask = mask[0:1, ...]

        if img.ndim == 2:
            img = img.unsqueeze(0)
        elif img.ndim == 3 and img.shape[0] > 4 and img.shape[-1] <= 4:
            img = img.permute(2, 0, 1)

        # 3. Augmentation
        if mask is not None:
            # Convert to float for kornia transforms to be safe, then back to long
            mask = mask.float()
            img, mask = self._apply_transforms(img, mask)

            # Squeeze channel dim if it's 1
            if mask.shape[0] == 1:
                mask = mask.squeeze(0)
            sample["mask"] = mask.long()
        else:
            # If no mask, we might need different handling or just resize img
            # Ideally we should have a resize_image_only pipeline if mask is None
            pass

        sample["img"] = img
        sample.pop("__key__", None)

        return sample


def __test_dataloader():
    import lovely_tensors as lt

    lt.monkey_patch()
    ds = DeepGlobeRoadExtractionStreamingDataset(input_dir="data/Downstreams/RoadExtraction/DeepGlobe_Litdata/train")
    print(len(ds))

    sample = ds[0]
    print(sample["img"].shape)
    print(sample["mask"].shape)


if __name__ == "__main__":
    __test_dataloader()

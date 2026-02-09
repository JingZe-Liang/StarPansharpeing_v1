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
from collections.abc import Sequence
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

        mean, std = _normalize_mean_std(norm_mean_std)

        # Define Augmentations
        self.resize = AugmentationSequential(
            Normalize(mean=mean, std=std),
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
            mask = mask.long()
            sample["gt"] = mask
            sample.pop("mask", None)
        else:
            # If no mask, we might need different handling or just resize img
            # Ideally we should have a resize_image_only pipeline if mask is None
            pass

        sample["img"] = img
        sample.pop("__key__", None)

        return sample


def _normalize_mean_std(norm_mean_std: Sequence[Sequence[float]]) -> tuple[list[float], list[float]]:
    if len(norm_mean_std) != 2:
        raise ValueError("norm_mean_std must be a sequence with [mean, std].")
    mean = [float(v) for v in norm_mean_std[0]]
    std = [float(v) for v in norm_mean_std[1]]
    return mean, std


def test_dataloader():
    import lovely_tensors as lt
    import matplotlib.pyplot as plt

    lt.monkey_patch()
    ds = DeepGlobeRoadExtractionStreamingDataset(input_dir="data/Downstreams/RoadExtraction/DeepGlobe_Litdata/train")
    print(len(ds))  # type: ignore[arg-type]

    sample = ds[100]
    print(sample["img"].shape)
    print(sample["gt"].shape)
    print(sample["gt"].unique())

    # 绘制图像和掩码
    img = sample["img"]  # (C, H, W)
    mask = sample["gt"]  # (H, W)

    # 转换为 numpy 并调整维度 (C, H, W) -> (H, W, C)
    img_np = img.permute(1, 2, 0).numpy()
    # 反归一化: 从 (-1, 1) 到 (0, 1)
    img_np = (img_np + 1) / 2
    img_np = img_np.clip(0, 1)

    mask_np = mask.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(img_np)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(mask_np, cmap="gray")
    axes[1].set_title("Mask")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("deep_globe_sample.webp", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved to deep_globe_sample.webp")


if __name__ == "__main__":
    test_dataloader()

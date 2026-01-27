"""
EuroSAT dataset class

Data folder is origanized as,
❯ tree -L 1 .
.
├── AnnualCrop
├── Forest
├── HerbaceousVegetation
├── Highway
├── Industrial
├── Pasture
├── PermanentCrop
├── Residential
├── River
└── SeaLake

"""

import os
import tempfile
import numpy as np
import tifffile
import torch
from torchvision.datasets import ImageFolder
from typing import Callable, Optional
import kornia.augmentation as K

# EuroSAT Statistics (Sentinel-2 13 bands)
EUROSAT_MEAN = [1353.7, 1117.2, 1041.4, 946.5, 1199.1, 2003.0, 2374.0, 2301.2, 732.9, 12.0, 1820.6, 1118.2, 2599.7]
EUROSAT_STD = [65.4, 154.0, 188.1, 278.9, 222.3, 234.6, 305.6, 300.4, 489.4, 411.7, 526.5, 288.6, 375.8]


def tiff_loader(path: str) -> torch.Tensor:
    """Load a tiff image using tifffile and convert to torch tensor (C, H, W)."""
    img = tifffile.imread(path).astype(np.float32)
    img = torch.from_numpy(img)

    # Handle shape: Ensure (C, H, W)
    if img.ndim == 3 and img.shape[2] <= 13 and img.shape[2] < img.shape[0]:  # (H, W, C)
        img = img.permute(2, 0, 1)

    return img


class EuroSAT(ImageFolder):
    """
    EuroSAT Dataset using ImageFolder with custom tiff loader and Kornia augmentations.

    Args:
        root: Path to dataset root directory
        split: 'train' or 'val'/'test' (affects augmentation)
        to_neg_1_1: If True, normalize to [-1, 1]; if False, use z-score normalization
        transform: Custom transform (if None, uses default Kornia augmentations)
        target_transform: Transform for labels
        is_valid_file: Function to filter valid files
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        to_neg_1_1: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        self.split = split
        self.to_neg_1_1 = to_neg_1_1

        # Build default transform if not provided
        if transform is None:
            aug_list = []

            # Add augmentations for training
            if split == "train":
                aug_list.extend(
                    [
                        K.RandomRotation(degrees=30.0, p=0.5),
                        K.RandomResizedCrop(size=(64, 64), scale=(0.8, 1.0), p=0.5),
                        K.RandomVerticalFlip(p=0.5),
                    ]
                )

            # Add normalization
            if to_neg_1_1:
                # Normalize to [0, 1] then to [-1, 1]
                aug_list.append(K.Normalize(mean=[0.5] * 13, std=[0.5] * 13))
            else:
                # Z-score normalization with EuroSAT stats
                aug_list.append(K.Normalize(mean=EUROSAT_MEAN, std=EUROSAT_STD))

            transform = K.AugmentationSequential(
                *aug_list,
                data_keys=["input"],
                keepdim=True,
            )

        self._transform = transform

        super().__init__(
            root,
            transform=None,  # We'll apply transform in __getitem__
            target_transform=target_transform,
            loader=tiff_loader,
            is_valid_file=is_valid_file,
        )

    def __getitem__(self, index: int):
        """
        Args:
            index: Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        # For [-1, 1] normalization, first scale to [0, 1]
        # EuroSAT raw values are typically 0-10000 (Sentinel-2 DN values)
        if self.to_neg_1_1:
            sample = sample / 10000.0
            sample = sample.clamp(0, 1)

        # Apply transform
        if self._transform is not None:
            sample = self._transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


def test_image_folder():
    """Test EuroSAT with Kornia augmentations."""
    print("Testing EuroSAT with simplified Kornia transform...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        classes = ["AnnualCrop", "Forest"]
        for cls in classes:
            os.makedirs(os.path.join(tmp_dir, cls), exist_ok=True)

        # Create dummy tiff files (H, W, C)
        dummy_data = np.random.uniform(0, 10000, (64, 64, 13)).astype(np.float32)

        tifffile.imwrite(os.path.join(tmp_dir, "AnnualCrop", "img1.tif"), dummy_data)
        tifffile.imwrite(os.path.join(tmp_dir, "Forest", "img1.tif"), dummy_data)

        # Test z-score normalization
        ds_zscore = EuroSAT(root=tmp_dir, split="train", to_neg_1_1=False)
        img_z, label = ds_zscore[0]
        print(f"Z-score norm - Shape: {img_z.shape}, Label: {label}")
        print(f"Z-score range: {img_z.min():.2f} to {img_z.max():.2f}")

        # Test [-1, 1] normalization
        ds_neg11 = EuroSAT(root=tmp_dir, split="train", to_neg_1_1=True)
        img_n, _ = ds_neg11[0]
        print(f"-1~1 norm - Shape: {img_n.shape}")
        print(f"-1~1 range: {img_n.min():.2f} to {img_n.max():.2f}")

        # Test val (no aug)
        ds_val = EuroSAT(root=tmp_dir, split="val", to_neg_1_1=True)
        img_val, _ = ds_val[0]
        print(f"Val - Shape: {img_val.shape}")

        print("✓ EuroSAT test passed!")


if __name__ == "__main__":
    test_image_folder()

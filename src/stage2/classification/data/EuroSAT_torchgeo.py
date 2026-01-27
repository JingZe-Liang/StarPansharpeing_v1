"""
EuroSAT dataset using TorchGeo

This uses the torchgeo library which automatically handles:
- Official train/val/test splits from the paper
- Automatic download of split files and dataset
- Proper band indexing and selection

Data folder structure after download:
data/
└── ds/
    └── images/
        └── remote_sensing/
            └── otherDatasets/
                └── sentinel_2/
                    └── tif/
                        ├── AnnualCrop/
                        ├── Forest/
                        ...
"""

import torch
import kornia.augmentation as K
from torchgeo.datasets import EuroSAT as TorchGeoEuroSAT
from typing import Any

# EuroSAT Statistics (Sentinel-2 13 bands)
EUROSAT_MEAN = [1353.7, 1117.2, 1041.4, 946.5, 1199.1, 2003.0, 2374.0, 2301.2, 732.9, 12.0, 1820.6, 1118.2, 2599.7]
EUROSAT_STD = [65.4, 154.0, 188.1, 278.9, 222.3, 234.6, 305.6, 300.4, 489.4, 411.7, 526.5, 288.6, 375.8]


class EuroSATWithAug(TorchGeoEuroSAT):
    """
    EuroSAT dataset with Kornia augmentations.

    Args:
        root: Path to dataset root directory
        split: 'train', 'val', or 'test'
        to_neg_1_1: If True, normalize to [-1, 1]; if False, use z-score normalization
        bands: Sequence of band names to load (default: all 13 bands)
        download: If True, download dataset and split files automatically
        checksum: If True, verify MD5 checksums
    """

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        to_neg_1_1: bool = False,
        bands: tuple[str, ...] = TorchGeoEuroSAT.BAND_SETS["all"],
        download: bool = True,
        checksum: bool = False,
    ):
        self.to_neg_1_1 = to_neg_1_1
        self.split_name = split

        # Build augmentation pipeline
        aug_list = []

        # Add training augmentations
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
            # Normalize to [-1, 1]
            # EuroSAT values are typically 0-10000, so we'll normalize after scaling
            aug_list.append(K.Normalize(mean=[0.5] * len(bands), std=[0.5] * len(bands)))
        else:
            # Z-score normalization with EuroSAT stats
            aug_list.append(K.Normalize(mean=EUROSAT_MEAN[: len(bands)], std=EUROSAT_STD[: len(bands)]))

        self.aug = (
            K.AugmentationSequential(
                *aug_list,
                data_keys=["input"],
                keepdim=True,
            )
            if aug_list
            else None
        )

        # Initialize parent class without transforms (we'll apply them ourselves)
        super().__init__(
            root=root,
            split=split,
            bands=bands,
            transforms=None,
            download=download,
            checksum=checksum,
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """
        Returns:
            tuple: (image, label) where image is (C, H, W) tensor
        """
        # Get sample from parent class
        sample = super().__getitem__(index)
        image = sample["image"]  # Already a float tensor from torchgeo
        label = sample["label"]

        # Scale to [0, 1] for [-1, 1] normalization
        # EuroSAT raw values are typically 0-10000 (Sentinel-2 DN values)
        if self.to_neg_1_1:
            image = image / 10000.0
            image = image.clamp(0, 1)

        # Apply augmentations
        if self.aug is not None:
            image = self.aug(image)

        return image, label


def test_torchgeo_eurosat():
    """Test TorchGeo EuroSAT dataset with augmentations."""
    print("Testing TorchGeo EuroSAT...")

    # Test all three splits
    for split in ["train", "val", "test"]:
        print(f"\n--- Testing {split} split ---")

        # Z-score normalization
        ds_zscore = EuroSATWithAug(
            root="data/Downstreams/EuroSAT_torchgeo",
            split=split,
            to_neg_1_1=False,
            download=True,
        )

        print(f"Dataset length: {len(ds_zscore)}")

        img, label = ds_zscore[0]
        print(f"Image shape: {img.shape}, Label: {label}")
        print(f"Z-score range: {img.min():.2f} to {img.max():.2f}")

        # [-1, 1] normalization
        ds_neg11 = EuroSATWithAug(
            root="data/Downstreams/EuroSAT_torchgeo",
            split=split,
            to_neg_1_1=True,
            download=True,
        )

        img_n, _ = ds_neg11[0]
        print(f"-1~1 range: {img_n.min():.2f} to {img_n.max():.2f}")

    print("\n✓ TorchGeo EuroSAT test passed!")


if __name__ == "__main__":
    test_torchgeo_eurosat()

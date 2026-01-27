"""
UC Merced dataset using TorchGeo with augmentations and custom normalization.
"""

import torch
import kornia.augmentation as K
from torchgeo.datasets import UCMerced as TorchGeoUCMerced
from typing import Any


class UCMerced(TorchGeoUCMerced):
    """
    UC Merced Land Use dataset with Kornia augmentations and [-1, 1] normalization option.

    The dataset consists of 2100 256x256 RGB images with 21 land use classes.
    """

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        to_neg_1_1: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            root: Root directory where dataset can be found.
            split: One of "train", "val", or "test".
            to_neg_1_1: If True, normalize image values to [-1, 1].
            download: If True, download dataset and store it in the root directory.
            checksum: If True, check the MD5 of the downloaded files.
        """
        self.to_neg_1_1 = to_neg_1_1

        # Build augmentation pipeline using Kornia
        aug_list: list[Any] = []
        if split == "train":
            aug_list.extend(
                [
                    K.RandomRotation(degrees=30.0, p=0.5),
                    K.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0), p=0.5),
                    K.RandomVerticalFlip(p=0.5),
                    K.RandomHorizontalFlip(p=0.5),
                ]
            )

        # Normalization to [-1, 1] if requested
        # Expects input in [0, 1]
        if self.to_neg_1_1:
            aug_list.append(
                K.Normalize(
                    mean=torch.tensor([0.5, 0.5, 0.5]),
                    std=torch.tensor([0.5, 0.5, 0.5]),
                )
            )

        self.aug = (
            K.AugmentationSequential(
                *aug_list,
                data_keys=["input"],
                keepdim=True,
            )
            if aug_list
            else None
        )

        super().__init__(
            root=root,
            split=split,
            transforms=None,  # We will apply augmentations in __getitem__
            download=download,
            checksum=checksum,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Get a sample from the dataset.

        Args:
            index: Index of the sample to fetch.

        Returns:
            A dictionary containing the image and label.
        """
        sample = super().__getitem__(index)
        image = sample["image"]  # torch.Tensor, usually (3, H, W)

        # Convert to float and scale to [0, 1] if needed
        # TorchGeo UCMerced usually returns uint8 if not transformed
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        elif image.max() > 1.0:
            image = image.float() / 255.0
        else:
            image = image.float()

        # Apply augmentations and normalization
        if self.aug is not None:
            # Kornia expects (B, C, H, W), so we unsqueeze and squeeze
            image = self.aug(image.unsqueeze(0)).squeeze(0)

        sample["image"] = image
        return sample


def test_ucmerced() -> None:
    """
    Test the UCMerced dataset implementation.
    """
    import os

    # Note: TorchGeo will try to find the dataset in 'root'
    root = "data/Downstreams/UCMerced"
    try:
        ds = UCMerced(root=root, split="train", to_neg_1_1=True, download=True)
        print(f"Dataset length: {len(ds)}")
        if len(ds) > 0:
            sample = ds[0]
            image = sample["image"]
            label = sample["label"]
            print(f"Image shape: {image.shape}")
            print(f"Label: {label}")
            print(f"Image range: {image.min().item():.2f} to {image.max().item():.2f}")

            assert image.shape == (3, 256, 256)
            if ds.to_neg_1_1:
                assert image.min() >= -1.1 and image.max() <= 1.1
    except Exception as e:
        print(f"Error during testing: {e}")


if __name__ == "__main__":
    test_ucmerced()

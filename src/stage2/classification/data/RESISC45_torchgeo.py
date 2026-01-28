"""
NWPU-RESISC45 dataset using TorchGeo with augmentations and optional [-1, 1] normalization.
"""

from typing import Any

import kornia.augmentation as K
import torch
from torchgeo.datasets import RESISC45 as TorchGeoRESISC45


class RESISC45(TorchGeoRESISC45):
    """
    NWPU-RESISC45 dataset with Kornia augmentations and optional [-1, 1] normalization.

    The dataset contains 256x256 RGB images with 45 classes.
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
            transforms=None,
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
        image = sample["image"]

        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        elif image.max() > 1.0:
            image = image.float() / 255.0
        else:
            image = image.float()

        if self.aug is not None:
            image = self.aug(image.unsqueeze(0)).squeeze(0)

        label = sample["label"]
        return {"image": image, "label": label}


def test_resisc45() -> None:
    root = "data/Downstreams/RESISC45"
    ds = RESISC45(root=root, split="train", to_neg_1_1=True, download=True)
    print(f"Dataset length: {len(ds)}")
    if len(ds) > 0:
        sample = ds[0]
        image = sample["image"]
        label = sample["label"]
        print(f"Image shape: {image.shape}")
        print(f"Label: {label}")
        print(f"Image range: {image.min().item():.2f} to {image.max().item():.2f}")


if __name__ == "__main__":
    test_resisc45()

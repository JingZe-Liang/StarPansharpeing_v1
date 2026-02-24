"""
NWPU-RESISC45 dataset using TorchGeo with augmentations and optional [-1, 1] normalization.
"""

from collections import defaultdict
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
        propotion: float = 1.0,
        sampling_seed: int = 2025,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            root: Root directory where dataset can be found.
            split: One of "train", "val", or "test".
            to_neg_1_1: If True, normalize image values to [-1, 1].
            propotion: Balanced sampling ratio in (0, 1]. For example, 0.1 means
                sampling 10% with equal count per class.
            sampling_seed: Random seed for balanced sampling index selection.
            download: If True, download dataset and store it in the root directory.
            checksum: If True, check the MD5 of the downloaded files.
        """
        self._validate_propotion(propotion)
        self.to_neg_1_1 = to_neg_1_1
        self.propotion = propotion
        self.sampling_seed = sampling_seed

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
        labels = self._extract_labels()
        self.sample_indices = self._build_balanced_indices(
            labels=labels,
            propotion=self.propotion,
            seed=self.sampling_seed,
        )

    @staticmethod
    def _validate_propotion(propotion: float) -> None:
        if not (0.0 < propotion <= 1.0):
            raise ValueError(f"propotion must be in (0, 1], got {propotion}.")

    def _extract_labels(self) -> list[int]:
        if hasattr(self, "targets"):
            return [int(label) for label in self.targets]
        return [int(label) for _, label in self.samples]

    @staticmethod
    def _build_balanced_indices(labels: list[int], propotion: float, seed: int) -> list[int]:
        RESISC45._validate_propotion(propotion)
        if not labels:
            return []
        if propotion == 1.0:
            return list(range(len(labels)))

        grouped_indices: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            grouped_indices[label].append(idx)

        min_class_count = min(len(class_indices) for class_indices in grouped_indices.values())
        samples_per_class = max(1, int(min_class_count * propotion))

        generator = torch.Generator().manual_seed(seed)
        sampled_indices: list[int] = []
        for class_id in sorted(grouped_indices):
            class_indices = grouped_indices[class_id]
            perm = torch.randperm(len(class_indices), generator=generator)
            selected = [class_indices[i] for i in perm[:samples_per_class].tolist()]
            sampled_indices.extend(selected)

        sampled_indices.sort()
        return sampled_indices

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Get a sample from the dataset.

        Args:
            index: Index of the sample to fetch.

        Returns:
            A dictionary containing the image and label.
        """
        dataset_index = self.sample_indices[index]
        sample = super().__getitem__(dataset_index)
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

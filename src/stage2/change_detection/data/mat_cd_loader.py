"""
Hyperspectral Change Detection Dataset Loader

This module provides dataset loaders for hyperspectral change detection datasets
including Bay Area, Hermiston, and Santa Barbara datasets.

Author: Zihan Cao
Date: 2025/09/05
"""

import os
import warnings
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from einops import rearrange
from scipy import io as sio
from torch.utils.data import DataLoader, Dataset

from src.data.window_slider import WindowSlider
from src.utilities.logging import log

from .label_centric_patcher import label_centrical_patcher

# Suppress warnings from scipy.io
warnings.filterwarnings("ignore", category=FutureWarning)


class HyperspectralChangeDetectionDataset(Dataset):
    """
    Base dataset class for hyperspectral change detection.

    Args:
        data_root (str): Root directory containing dataset folders
        dataset_name (str): Name of the dataset ('bayArea', 'Hermiston', 'santaBarbara')
        patch_size (Optional[int]): Size of patches to extract. If None, return full images.
        stride (int): Stride for patch extraction
        transform (Optional[Any]): Optional transform to apply to data
        normalize (bool): Whether to normalize the data
    """

    DATASET_CONFIGS = {
        "bayArea": {
            "image1_path": "bayArea/mat/Bay_Area_2013.mat",
            "image2_path": "bayArea/mat/Bay_Area_2015.mat",
            "gt_path": "bayArea/gt/bayArea_gtChanges2.mat.mat",
            "image_key": "HypeRvieW",
            "gt_key": "HypeRvieW",
            "shape": (600, 500, 224),
            "num_classes": 3,
            "class_names": ["unknown", "changed", "unchanged"],
        },
        "Hermiston": {
            "image1_path": "Hermiston/mat/hermiston2004.mat",
            "image2_path": "Hermiston/mat/hermiston2007.mat",
            "gt_path": "Hermiston/gt/rdChangesHermiston_5classes.mat",
            "image_key": "HypeRvieW",
            "gt_key": "gt5clasesHermiston",
            "shape": (390, 200, 242),
            "num_classes": 6,
            "class_names": ["unchanged", "type1", "type2", "type3", "type4", "type5"],
        },
        "santaBarbara": {
            "image1_path": "santaBarbara/mat/barbara_2013.mat",
            "image2_path": "santaBarbara/mat/barbara_2014.mat",
            "gt_path": "santaBarbara/gt/barbara_gtChanges.mat",
            "image_key": "HypeRvieW",
            "gt_key": "HypeRvieW",
            "shape": (984, 740, 224),
            "num_classes": 3,
            "class_names": ["unknown", "changed", "unchanged"],
        },
        "FarmLand": {
            "image1_path": "FarmLand/mat/Farm1.mat",
            "image2_path": "FarmLand/mat/Farm2.mat",
            "gt_path": "FarmLand/gt/GTChina1.mat",
            "image1_key": "imgh",
            "image2_key": "imghl",
            "gt_key": "label",
            "shape": (450, 140, 155),
            "class_names": ["unknown", "unchanged", "changed"],
        },
    }

    def __init__(
        self,
        data_root: str,
        dataset_name: str,
        patch_size: Optional[int] = None,
        stride: int = 1,
        transform: Optional[Any] = None,
        normalize: bool = True,
    ):
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.normalize = normalize

        if dataset_name not in self.DATASET_CONFIGS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available datasets: {list(self.DATASET_CONFIGS.keys())}"
            )

        self.config = self.DATASET_CONFIGS[dataset_name]

        # Load data
        self.image1, self.image2, self.gt = self._load_data()

        # Generate patches or use full image mode
        if patch_size is not None:
            self.patches = self._generate_patches()
            self.full_image_mode = False
            log(f"Loaded {dataset_name} dataset in patch mode:", level="debug")
            log(f"  Image1 shape: {self.image1.shape}", level="debug")
            log(f"  Image2 shape: {self.image2.shape}", level="debug")
            log(f"  GT shape: {self.gt.shape}", level="debug")
            log(f"  Number of patches: {len(self.patches)}", level="debug")
            log(f"  Classes: {np.unique(self.gt)}", level="debug")
        else:
            self.full_image_mode = True
            log(f"Loaded {dataset_name} dataset in full image mode:", level="debug")
            log(f"  Image1 shape: {self.image1.shape}", level="debug")
            log(f"  Image2 shape: {self.image2.shape}", level="debug")
            log(f"  GT shape: {self.gt.shape}", level="debug")
            log(f"  Classes: {np.unique(self.gt)}", level="debug")

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load hyperspectral images and ground truth."""
        # Load first image (t1)
        image1_path = os.path.join(self.data_root, self.config["image1_path"])
        image1_data = sio.loadmat(image1_path)
        image1 = image1_data[
            self.config.get("image_key", self.config.get("image1_key"))
        ]

        # Load second image (t2)
        image2_path = os.path.join(self.data_root, self.config["image2_path"])
        image2_data = sio.loadmat(image2_path)
        image2 = image2_data[
            self.config.get("image_key", self.config.get("image2_key"))
        ]

        # Load ground truth
        gt_path = os.path.join(self.data_root, self.config["gt_path"])
        gt_data = sio.loadmat(gt_path)
        gt = gt_data[self.config["gt_key"]]

        # All to 0: unknown, 1: changed, 2: unchanged
        if self.dataset_name == "FarmLand":
            gt_index_1 = gt == 1
            gt_index_2 = gt == 2
            gt[gt_index_1] = 2
            gt[gt_index_2] = 1
        elif self.dataset_name == "Hermiston":
            gt[gt > 0] = 1
            gt[gt == 0] = 2

        # Ensure consistent shapes
        if image1.shape != image2.shape:
            raise ValueError(
                f"Image shapes don't match: {image1.shape} vs {image2.shape}"
            )

        if gt.shape != image1.shape[:2]:
            raise ValueError(
                f"GT shape {gt.shape} doesn't match image spatial shape {image1.shape[:2]}"
            )

        # Normalize if requested
        if self.normalize:
            image1, image2 = self._normalize_image(image1, image2)

        return image1, image2, gt

    def _normalize_image(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        quantile_q: float = 0.99,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize image to [0, 1] range using shared min-max normalization across both images."""
        # Convert to torch tensors
        img1 = torch.as_tensor(image1, dtype=torch.float32)
        img2 = torch.as_tensor(image2, dtype=torch.float32)

        # Stack images along time dimension to compute shared min/max
        # Shape will be (H, W, C, T) where T=2 for the two time points
        img_t = torch.stack([img1, img2], dim=-1)

        if quantile_q < 1.0:
            flatten_ = partial(rearrange, pattern="h w c t -> (h w t) c")
            # Use quantile-based min/max to reduce outlier impact
            # (1, c) -> (1, 1, c, 1) for broadcasting
            q_max = torch.quantile(flatten_(img_t), q=quantile_q, dim=0, keepdim=True)
            img_t = img_t.clamp_(max=q_max[:, None, :, None])

        # Compute shared min and max across all dimensions except channels
        # Result shape will be (1, 1, C, 1) or (1, 1, 1, 1) depending on per_channel
        img_shard_min = img_t.amin((0, 1, 3), keepdim=True).squeeze(-1)
        img_shard_max = img_t.amax((0, 1, 3), keepdim=True).squeeze(-1)

        # Normalize each image using shared min/max values
        img1.sub_(img_shard_min)
        img2.sub_(img_shard_min)

        # Avoid division by zero
        img_range = img_shard_max - img_shard_min
        img_range = torch.where(img_range < 1e-6, torch.tensor(1e-6), img_range)

        img1.div_(img_range)
        img2.div_(img_range)

        # Clip to [0, 1] range to handle numerical errors
        img1 = img1.clamp_(0.0, 1.0)
        img2 = img2.clamp_(0.0, 1.0)

        return img1.numpy(), img2.numpy()

    def _generate_patches(self) -> List[Dict]:
        """Generate patch coordinates for training/testing."""
        height, width = self.gt.shape
        patches = []

        # Ensure patch_size is not None
        if self.patch_size is None:
            raise ValueError("patch_size must be specified for patch generation")

        for i in range(0, height - self.patch_size + 1, self.stride):
            for j in range(0, width - self.patch_size + 1, self.stride):
                # Get center pixel label (for classification)
                center_i = i + self.patch_size // 2
                center_j = j + self.patch_size // 2

                # Skip unknown pixels (label 0 typically means unknown/no data)
                # if self.gt[center_i, center_j] == 0:
                #     continue

                patches.append(
                    {
                        "i": i,
                        "j": j,
                        "center_i": center_i,
                        "center_j": center_j,
                        # "label": self.gt[center_i, center_j],
                    }
                )

        return patches

    def __len__(self) -> int:
        """Return dataset length."""
        if self.full_image_mode:
            return 1  # Only one full image pair
        else:
            return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a patch pair or full images and their labels."""
        if self.full_image_mode:
            # Return full images
            image1 = torch.from_numpy(self.image1).float().permute(2, 0, 1)  # (C, H, W)
            image2 = torch.from_numpy(self.image2).float().permute(2, 0, 1)  # (C, H, W)
            gt = torch.from_numpy(self.gt).long()  # (H, W)

            # Apply transforms if provided
            if self.transform is not None:
                image1 = self.transform(image1)
                image2 = self.transform(image2)

            return image1, image2, gt
        else:
            # Return patch
            patch_info = self.patches[idx]
            i, j = patch_info["i"], patch_info["j"]
            # label = patch_info["label"]

            # Extract patches from both images
            patch1 = self.image1[i : i + self.patch_size, j : j + self.patch_size, :]
            patch2 = self.image2[i : i + self.patch_size, j : j + self.patch_size, :]
            gt_patch = self.gt[i : i + self.patch_size, j : j + self.patch_size]

            # Convert to tensors and rearrange dimensions to (C, H, W)
            patch1 = torch.from_numpy(patch1).float().permute(2, 0, 1)
            patch2 = torch.from_numpy(patch2).float().permute(2, 0, 1)
            label = torch.tensor(gt_patch, dtype=torch.long)

            # Apply transforms if provided
            if self.transform is not None:
                patch1 = self.transform(patch1)
                patch2 = self.transform(patch2)

            return patch1, patch2, label


class FullImageChangeDetectionDataset(Dataset):
    """
    Dataset class for full image hyperspectral change detection.
    This class always returns full images without patch extraction.

    Args:
        data_root (str): Root directory containing dataset folders
        dataset_name (str): Name of the dataset ('bayArea', 'Hermiston', 'santaBarbara')
        transform (Optional[Any]): Optional transform to apply to data
        normalize (bool): Whether to normalize the data
    """

    def __init__(
        self,
        data_root: str,
        dataset_name: str,
        transform: Optional[Any] = None,
        normalize: bool = True,
    ):
        # Use the base dataset with patch_size=None for full image mode
        self.base_dataset = HyperspectralChangeDetectionDataset(
            data_root=data_root,
            dataset_name=dataset_name,
            patch_size=None,
            transform=transform,
            normalize=normalize,
        )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.base_dataset[idx]


def create_change_detection_dataloader(
    data_root: str,
    dataset_name: str,
    patch_size: Optional[int] = 15,
    stride: int = 1,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    transform: Optional[Any] = None,
    normalize: bool = True,
):
    """
    Create a DataLoader for hyperspectral change detection dataset.

    Args:
        data_root (str): Root directory containing dataset folders
        dataset_name (str): Name of the dataset ('bayArea', 'Hermiston', 'santaBarbara')
        patch_size (Optional[int]): Size of patches to extract. If None, return full images.
        stride (int): Stride for patch extraction
        batch_size (int): Batch size for DataLoader
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of worker processes for data loading
        transform (Optional[Any]): Optional transform to apply to data
        normalize (bool): Whether to normalize the data

    Returns:
        DataLoader: Configured DataLoader for the dataset
    """
    dataset = HyperspectralChangeDetectionDataset(
        data_root=data_root,
        dataset_name=dataset_name,
        patch_size=patch_size,
        stride=stride,
        transform=transform,
        normalize=normalize,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataset, dataloader


def create_changed_detection_patch_loader(
    data_root: str,
    dataset_name: str,
    patch_size: int = 32,
    stride: int = 1,
    batch_size: int = 16,
    micro_batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    transform: Optional[Any] = None,
    normalize: bool = True,
    changed_label=1,
    unchanged_label=2,
):
    dataset = HyperspectralChangeDetectionDataset(
        data_root=data_root,
        dataset_name=dataset_name,
        patch_size=None,
        stride=stride,
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    def yield_fn():
        for img1, img2, label in dataloader:
            for img1_p, img2_p, label_p in label_centrical_patcher(
                img1,
                img2,
                label,
                changed_label=changed_label,
                unchanged_label=unchanged_label,
                micro_batch_size=micro_batch_size,
                patch_size=patch_size,
            ):
                yield img1_p, img2_p, label_p

    yield_gen_loader = yield_fn()
    return dataset, yield_gen_loader


def create_full_image_dataloader(
    data_root: str,
    dataset_name: str,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 4,
    transform: Optional[Any] = None,
    normalize: bool = True,
):
    """
    Create a DataLoader for full image hyperspectral change detection.

    Args:
        data_root (str): Root directory containing dataset folders
        dataset_name (str): Name of the dataset ('bayArea', 'Hermiston', 'santaBarbara')
        batch_size (int): Batch size for DataLoader (typically 1 for full images)
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of worker processes for data loading
        transform (Optional[Any]): Optional transform to apply to data
        normalize (bool): Whether to normalize the data

    Returns:
        DataLoader: Configured DataLoader for the dataset
    """
    dataset = FullImageChangeDetectionDataset(
        data_root=data_root,
        dataset_name=dataset_name,
        transform=transform,
        normalize=normalize,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataset, dataloader


# * --- Test --- #


def test_full_patcher_dataloader():
    from src.stage2.change_detection.data.label_centric_patcher import (
        label_centrical_patcher,
    )

    ds, dl = create_full_image_dataloader(
        "data/Downstreams/ChangeDetection/mat_2",
        dataset_name="FarmLand",
        batch_size=1,
        shuffle=False,
    )
    for batch in dl:
        img1, img2, label = batch
        for img1_p, img2_p, label_p in label_centrical_patcher(
            img1,
            img2,
            label,
            changed_label=1,
            unchanged_label=2,
            micro_batch_size=16,
            patch_size=32,
        ):
            print(img1_p.shape, img2_p.shape, label_p.shape)


# Example usage
if __name__ == "__main__":
    test_full_patcher_dataloader()

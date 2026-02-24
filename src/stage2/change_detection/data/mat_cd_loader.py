"""
Hyperspectral Change Detection Dataset Loader

This module provides dataset loaders for hyperspectral change detection datasets
including Bay Area, Hermiston, and Santa Barbara datasets.

Author: Zihan Cao
Date: 2025/09/05
"""

import math
import os
import warnings
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from einops import rearrange
from kornia.augmentation import (
    AugmentationSequential,
    RandomHorizontalFlip,
    RandomRotation,
    RandomSolarize,
    RandomVerticalFlip,
)
from loguru import logger
from scipy import io as sio
from torch.utils.data import DataLoader, Dataset

from src.data.window_slider import WindowSlider

from .label_centric_patcher import label_centrical_patcher

# Suppress warnings from scipy.io
warnings.filterwarnings("ignore", category=FutureWarning)


def get_default_transform():
    """Get default augmentation transform for hyperspectral images."""
    transform = AugmentationSequential(
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        # RandomRotation(degrees=90, p=0.5, align_corners=False),
        data_keys=["input", "input", "mask"],
        same_on_batch=False,
    )
    return transform


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
            "num_classes": 3,
            "class_names": ["unknown", "unchanged", "changed"],
        },
    }

    def __init__(
        self,
        data_root: str,
        dataset_name: str,
        patch_size: Optional[int] = None,
        patch_mode: str = "random_0.2",
        fixed_changed_num: int = 500,
        fixed_unchanged_num: int = 500,
        stride: int = 1,
        transform: Optional[Any] | Literal["default"] = "default",
        normalize: bool = True,
        to_neg_1_1: bool = False,
        train_patch_upsample_to: int | None = None,
    ):
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.patch_size = patch_size
        self.patch_mode = patch_mode
        self.fixed_changed_num = int(fixed_changed_num)
        self.fixed_unchanged_num = int(fixed_unchanged_num)
        self.stride = stride
        if transform == "default":
            transform = get_default_transform()
        if transform is not None:
            logger.info(f"[CD Dataset]: use transform: {transform}")
        self.transform = transform
        self.normalize = normalize
        self.to_neg_1_1 = to_neg_1_1
        self.train_patch_upsample_to = train_patch_upsample_to

        if dataset_name not in self.DATASET_CONFIGS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. Available datasets: {list(self.DATASET_CONFIGS.keys())}"
            )

        self.config = edict(self.DATASET_CONFIGS[dataset_name])

        # Load data
        self.image1, self.image2, self.gt = self._load_data()

        # Generate patches or use full image mode
        if patch_size is not None:
            if patch_mode == "slide":
                self.patches = self._generate_patches_slide_windows()
            elif patch_mode.startswith("fixed"):
                changed_num, unchanged_num = self._resolve_fixed_patch_counts(patch_mode)
                self.patches = self._generate_patches_with_fixed_num(changed=changed_num, unchanged=unchanged_num)
            elif patch_mode.startswith("random"):
                ratio = float(patch_mode.split("_")[1])
                self.patches = self._generate_patches_rnd_splits(train_splits=ratio)
            else:
                raise ValueError(f"Unknown patch mode: {patch_mode}")
            self.full_image_mode = False
            logger.debug(f"Loaded {dataset_name} dataset in patch mode:")
            logger.debug(f"  Image1 shape: {self.image1.shape}")
            logger.debug(f"  Image2 shape: {self.image2.shape}")
            logger.debug(f"  GT shape: {self.gt.shape}")
            logger.debug(f"  Number of patches: {len(self.patches)}")
            logger.debug(f"  Classes: {np.unique(self.gt)}")
        else:
            self.full_image_mode = True
            logger.debug(f"Loaded {dataset_name} dataset in full image mode:")
            logger.debug(f"  Image1 shape: {self.image1.shape}")
            logger.debug(f"  Image2 shape: {self.image2.shape}")
            logger.debug(f"  GT shape: {self.gt.shape}")
            logger.debug(f"  Classes: {np.unique(self.gt)}")

    def _resolve_fixed_patch_counts(self, patch_mode: str) -> tuple[int, int]:
        """Resolve fixed changed/unchanged patch counts from patch_mode."""
        # Supported:
        # - fixed: use fixed_changed_num / fixed_unchanged_num
        # - fixed_500: use same number for changed and unchanged
        # - fixed_300_700: changed=300, unchanged=700
        parts = patch_mode.split("_")
        if len(parts) == 1:
            changed_num = self.fixed_changed_num
            unchanged_num = self.fixed_unchanged_num
        elif len(parts) == 2:
            changed_num = int(parts[1])
            unchanged_num = int(parts[1])
        elif len(parts) == 3:
            changed_num = int(parts[1])
            unchanged_num = int(parts[2])
        else:
            raise ValueError(
                f"Invalid fixed patch mode: {patch_mode}. Use `fixed`, `fixed_<num>`, or `fixed_<changed>_<unchanged>`."
            )

        if changed_num <= 0 or unchanged_num <= 0:
            raise ValueError(f"Fixed sample counts must be > 0, got changed={changed_num}, unchanged={unchanged_num}.")
        return changed_num, unchanged_num

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load hyperspectral images and ground truth."""
        # Load first image (t1)
        image1_path = os.path.join(self.data_root, self.config["image1_path"])
        image1_data = sio.loadmat(image1_path)
        image1 = image1_data[self.config.get("image_key", self.config.get("image1_key"))]

        # Load second image (t2)
        image2_path = os.path.join(self.data_root, self.config["image2_path"])
        image2_data = sio.loadmat(image2_path)
        image2 = image2_data[self.config.get("image_key", self.config.get("image2_key"))]

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
            # 0 to unchanged ?
            gt[gt == 0] = 2
        elif self.dataset_name == "Hermiston":
            gt[gt > 0] = 1
            gt[gt == 0] = 2

        # Final mapping: unknown->255, changed->1, unchanged->0
        gt[gt == 0] = 255
        gt[gt == 2] = 0
        gt = gt.astype(np.int32)

        # Ensure consistent shapes
        if image1.shape != image2.shape:
            raise ValueError(f"Image shapes don't match: {image1.shape} vs {image2.shape}")

        if gt.shape != image1.shape[:2]:
            raise ValueError(f"GT shape {gt.shape} doesn't match image spatial shape {image1.shape[:2]}")

        # Normalize if requested
        if self.normalize:
            image1, image2 = self._normalize_image(image1, image2, to_neg_1_1=self.to_neg_1_1)

        return image1, image2, gt

    def _normalize_image(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        quantile_q: float = 1.0,
        to_neg_1_1: bool = False,
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

        if to_neg_1_1:
            img1.mul_(2.0).sub_(1.0)
            img2.mul_(2.0).sub_(1.0)

        return img1.numpy(), img2.numpy()

    def _generate_patches_with_fixed_num(self, changed: int = 500, unchanged: int = 500) -> list[dict[str, int]]:
        """Generate fixed-number patches for changed and unchanged classes."""
        h, w = self.gt.shape[-2:]
        ps = self.patch_size
        if ps is None:
            raise ValueError("patch_size must be specified for fixed sampling.")

        max_i = h - ps
        max_j = w - ps
        if max_i < 0 or max_j < 0:
            raise ValueError(f"Image size ({h}, {w}) is smaller than patch size {ps}")

        invalid_labels = {255, -1}
        changed_label, unchanged_label = 1, 0
        positions_by_label: dict[int, list[dict[str, int]]] = {
            changed_label: [],
            unchanged_label: [],
        }

        for i in range(max_i + 1):
            for j in range(max_j + 1):
                center_i = i + ps // 2
                center_j = j + ps // 2
                center_label = int(self.gt[center_i, center_j])
                if center_label in invalid_labels:
                    continue
                if center_label not in positions_by_label:
                    continue
                positions_by_label[center_label].append(
                    {
                        "i": i,
                        "j": j,
                        "center_i": center_i,
                        "center_j": center_j,
                    }
                )

        generator = torch.Generator().manual_seed(2025)
        patches: list[dict[str, int]] = []
        specs = [
            (changed_label, changed, "changed"),
            (unchanged_label, unchanged, "unchanged"),
        ]
        for class_label, target_num, class_name in specs:
            class_positions = positions_by_label[class_label]
            available_num = len(class_positions)
            if available_num == 0:
                logger.warning(f"No valid positions found for {class_name} class.")
                continue

            if available_num <= target_num:
                selected = class_positions
                if available_num < target_num:
                    logger.warning(
                        f"{class_name} class only has {available_num} samples, fewer than requested {target_num}.",
                    )
            else:
                selected_indices = torch.randperm(available_num, generator=generator)[:target_num].tolist()
                selected = [class_positions[int(i)] for i in selected_indices]
            patches.extend(selected)
            logger.info(f"Selected {len(selected)} {class_name} patches from {available_num} candidates.")

        logger.info(f"Fixed sampling generated {len(patches)} total patches.")
        return patches

    def _generate_patches_rnd_splits(self, train_splits: float = 0.2):
        """Splits images into patches in one training-split ratio.

        Generate random patch positions ensuring all patches stay within image boundaries.
        Samples patches separately for changed and unchanged pixels to maintain class balance.

        Parameters
        ----------
        train_splits : float
            Ratio of valid patch positions to sample (0.0 to 1.0)

        Returns
        -------
        List[Dict]
            List of patch dictionaries with keys 'i', 'j', 'center_i', 'center_j'
            representing patch top-left corner and center coordinates
        """
        h, w = self.gt.shape[-2:]
        ps = self.patch_size
        assert ps is not None

        # Calculate valid patch top-left corner ranges
        # Patch covers [i:i+ps, j:j+ps], so i can be 0 to h-ps, j can be 0 to w-ps
        max_i = h - ps
        max_j = w - ps

        if max_i < 0 or max_j < 0:
            raise ValueError(f"Image size ({h}, {w}) is smaller than patch size {ps}")

        # Get all valid patch positions and their center labels
        # Ignore invalid labels (e.g., 255)
        INVALID_LABELS = [255, -1]  # Labels to ignore

        # Group positions by their center pixel label
        positions_by_label = {}  # Dict to store positions for each label

        for i in range(max_i + 1):
            for j in range(max_j + 1):
                center_i = i + ps // 2
                center_j = j + ps // 2
                center_label = self.gt[center_i, center_j]

                # Skip invalid labels
                if center_label in INVALID_LABELS:
                    continue

                position = {
                    "i": i,
                    "j": j,
                    "center_i": center_i,
                    "center_j": center_j,
                }

                # Group positions by label
                if center_label not in positions_by_label:
                    positions_by_label[center_label] = []
                positions_by_label[center_label].append(position)

        # Get unique valid classes
        valid_classes = sorted(positions_by_label.keys())
        n_classes = len(valid_classes)

        if n_classes == 0:
            logger.warning("No valid positions found for sampling")
            return []

        # Log class distribution
        class_info = {label: len(positions_by_label[label]) for label in valid_classes}
        logger.info(f"Found {n_classes} classes with distribution: {class_info}")

        # Calculate total target patches
        total_valid_positions = sum(len(positions) for positions in positions_by_label.values())
        total_target_patches = int(total_valid_positions * train_splits)

        # For balanced sampling, take equal numbers from each class
        # If some classes have fewer samples, sample all from those classes
        min_class_size = min(len(positions) for positions in positions_by_label.values())

        if min_class_size == 0:
            logger.warning("Some classes have no valid positions")
            return []

        # Calculate patches per class for balanced sampling
        patches_per_class = min(min_class_size, total_target_patches // n_classes)

        logger.info(
            f"Sampling {patches_per_class} patches from each of {n_classes} classes (total: {patches_per_class * n_classes})"
        )

        generator = torch.Generator().manual_seed(2025)
        patches = []

        # Sample from each class
        for class_label in valid_classes:
            class_positions = positions_by_label[class_label]

            if len(class_positions) <= patches_per_class:
                # Take all positions if fewer than requested
                selected_positions = class_positions
                logger.info(f"Selected all {len(selected_positions)} patches from class {class_label}")
            else:
                # Randomly sample positions
                indices = torch.randperm(len(class_positions), generator=generator)[:patches_per_class]
                selected_positions = [class_positions[idx] for idx in indices]
                logger.info(f"Selected {len(selected_positions)} patches from class {class_label}")

            patches.extend(selected_positions)

        return patches

    def _generate_patches_slide_windows(self) -> List[Dict]:
        """Generate patch coordinates for training/testing."""
        height, width = self.gt.shape
        patches = []

        # Ensure patch_size is not None
        if self.patch_size is None:
            raise ValueError("patch_size must be specified for patch generation")

        # If not divisible, adjust the image using resize
        # if (height - self.patch_size) % self.stride != 0 or (width - self.patch_size) % self.stride != 0:
        #     # resize larger
        #     lh, lw = (
        #         math.ceil(height / self.patch_size) * self.patch_size,
        #         math.ceil(width / self.patch_size) * self.patch_size,
        #     )
        #     self.image1 = resize(self.image1, (lh, lw), order=2, mode="constant", cval=0)
        #     self.image2 = resize(self.image2, (lh, lw), order=2, mode="constant", cval=0)
        #     self.gt = resize(self.gt, (lh, lw), order=0, mode="constant", cval=255)  # 255 is ignored label
        #     logger.info(f"Resize images from ({height}, {width}) to ({lh}, {lw}) for patch extraction.")

        for i in range(0, height - self.patch_size + 1, self.stride):
            for j in range(0, width - self.patch_size + 1, self.stride):
                # Get center pixel label (for classification)
                center_i = i + self.patch_size // 2
                center_j = j + self.patch_size // 2

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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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

            return {"img1": image1, "img2": image2, "gt": gt}
        else:
            # Return patch
            patch_info = self.patches[idx]
            i, j = patch_info["i"], patch_info["j"]
            ps = self.patch_size
            if ps is None:
                raise ValueError("patch_size must be specified in patch mode.")
            # label = patch_info["label"]

            # Extract patches from both images
            patch1 = self.image1[i : i + ps, j : j + ps, :]
            patch2 = self.image2[i : i + ps, j : j + ps, :]
            gt_patch = self.gt[i : i + ps, j : j + ps]

            # Convert to tensors and rearrange dimensions to (C, H, W)
            patch1 = torch.from_numpy(patch1).float().permute(2, 0, 1)
            patch2 = torch.from_numpy(patch2).float().permute(2, 0, 1)
            label = torch.tensor(gt_patch, dtype=torch.long)

            # Apply transforms if provided
            if self.transform is not None:
                patch1 = self.transform(patch1)
                patch2 = self.transform(patch2)

            if self.train_patch_upsample_to is not None:
                tgt_size = int(self.train_patch_upsample_to)
                if tgt_size <= 0:
                    raise ValueError(f"train_patch_upsample_to must be > 0, got {tgt_size}")
                patch1 = F.interpolate(
                    patch1.unsqueeze(0),
                    size=(tgt_size, tgt_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                patch2 = F.interpolate(
                    patch2.unsqueeze(0),
                    size=(tgt_size, tgt_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                label = (
                    F.interpolate(
                        label.unsqueeze(0).unsqueeze(0).float(),
                        size=(tgt_size, tgt_size),
                        mode="nearest",
                    )
                    .squeeze(0)
                    .squeeze(0)
                    .long()
                )

            return {"img1": patch1, "img2": patch2, "gt": label}


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
        to_neg_1_1: bool = False,
        interp_to_div_by: int | None = None,
    ):
        # Use the base dataset with patch_size=None for full image mode
        self.base_dataset = HyperspectralChangeDetectionDataset(
            data_root=data_root,
            dataset_name=dataset_name,
            patch_size=None,
            transform=transform,
            normalize=normalize,
            to_neg_1_1=to_neg_1_1,
        )

        self.interp_to_div_by = interp_to_div_by

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t1, t2, gt = self.base_dataset[idx].values()
        if self.interp_to_div_by is not None:
            h, w = t1.shape[-2:]
            r = self.interp_to_div_by
            h_i, w_i = math.ceil(h / r) * r, math.ceil(w / r) * r
            t1 = F.interpolate(t1[None], size=(h_i, w_i), mode="bilinear")[0]
            t2 = F.interpolate(t2[None], size=(h_i, w_i), mode="bilinear")[0]
            gt = F.interpolate(gt[None, None].float(), size=(h_i, w_i), mode="nearest")[0, 0]
        return {"img1": t1, "img2": t2, "gt": gt}


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
    to_neg_1_1: bool = False,
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
        to_neg_1_1=to_neg_1_1,
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
    patch_mode: str = "random_0.2",
    fixed_changed_num: int = 500,
    fixed_unchanged_num: int = 500,
    stride: int = 1,
    batch_size: int = 16,
    micro_batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    transform: Optional[Any] = None,
    normalize: bool = True,
    changed_label=1,
    unchanged_label=0,
    to_neg_1_1=False,
    patch_outside=False,
    train_patch_upsample_to: int | None = None,
):
    dataset = HyperspectralChangeDetectionDataset(
        data_root=data_root,
        dataset_name=dataset_name,
        patch_size=patch_size,
        patch_mode=patch_mode,
        fixed_changed_num=fixed_changed_num,
        fixed_unchanged_num=fixed_unchanged_num,
        stride=stride,
        transform=transform,
        normalize=normalize,
        to_neg_1_1=to_neg_1_1,
        train_patch_upsample_to=train_patch_upsample_to,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def yield_fn():
        """
        Dataloader return pre-clipped patches or full images,
        and in this function, clip another patches that around the changed labels.
        """
        for batch in dataloader:
            img1, img2, label = batch["img1"], batch["img2"], batch["gt"]
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

    if patch_outside:
        yield_gen_loader = yield_fn()
        return dataset, yield_gen_loader
    else:
        return dataset, dataloader


def create_full_image_dataloader(
    data_root: str,
    dataset_name: str,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 4,
    transform: Optional[Any] = None,
    normalize: bool = True,
    to_neg_1_1=False,
    interp_to_div_by: int | None = None,
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
        to_neg_1_1=to_neg_1_1,
        interp_to_div_by=interp_to_div_by,
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


def test_change_detection_dataloader():
    ds, dl = create_changed_detection_patch_loader(
        "data/Downstreams/ChangeDetection/mat_2",
        dataset_name="Hermiston",
        patch_size=128,
        stride=1,
        batch_size=16,
        shuffle=False,
        patch_mode="random_0.2",
    )
    print(f"Dataset has {len(ds)} samples.")
    for batch in dl:
        img1, img2, label = batch["img1"], batch["img2"], batch["gt"]
        print(img1.shape, img2.shape, label.shape)
        # break  # 只测试第一个batch


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
    """
    python -m src.stage2.change_detection.data.mat_cd_loader
    """
    test_change_detection_dataloader()
    # test_full_patcher_dataloader()

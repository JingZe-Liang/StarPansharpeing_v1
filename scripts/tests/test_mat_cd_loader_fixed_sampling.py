from __future__ import annotations

import numpy as np
import torch

from src.stage2.change_detection.data.mat_cd_loader import HyperspectralChangeDetectionDataset


def _count_available_centers(gt: np.ndarray, patch_size: int, label: int) -> int:
    h, w = gt.shape
    count = 0
    for i in range(h - patch_size + 1):
        for j in range(w - patch_size + 1):
            ci = i + patch_size // 2
            cj = j + patch_size // 2
            if int(gt[ci, cj]) == label:
                count += 1
    return count


def test_fixed_sampling_returns_requested_balanced_counts() -> None:
    ds = HyperspectralChangeDetectionDataset.__new__(HyperspectralChangeDetectionDataset)
    ds.patch_size = 3
    ds.gt = (np.indices((12, 12)).sum(axis=0) % 2).astype(np.int32)

    patches = ds._generate_patches_with_fixed_num(changed=10, unchanged=12)
    labels = [int(ds.gt[p["center_i"], p["center_j"]]) for p in patches]

    assert len(patches) == 22
    assert labels.count(1) == 10
    assert labels.count(0) == 12


def test_fixed_sampling_falls_back_when_class_insufficient() -> None:
    ds = HyperspectralChangeDetectionDataset.__new__(HyperspectralChangeDetectionDataset)
    ds.patch_size = 3
    ds.gt = np.zeros((10, 10), dtype=np.int32)
    ds.gt[3, 3] = 1
    ds.gt[6, 6] = 1

    patches = ds._generate_patches_with_fixed_num(changed=10, unchanged=5)
    labels = [int(ds.gt[p["center_i"], p["center_j"]]) for p in patches]

    available_changed = _count_available_centers(ds.gt, ds.patch_size, label=1)
    assert labels.count(1) == available_changed
    assert labels.count(0) == 5


def test_train_patch_upsample_only_affects_patch_output() -> None:
    ds = HyperspectralChangeDetectionDataset.__new__(HyperspectralChangeDetectionDataset)
    ds.full_image_mode = False
    ds.patch_size = 3
    ds.transform = None
    ds.train_patch_upsample_to = 8
    ds.patches = [{"i": 0, "j": 0, "center_i": 1, "center_j": 1}]
    ds.image1 = np.random.rand(3, 3, 4).astype(np.float32)
    ds.image2 = np.random.rand(3, 3, 4).astype(np.float32)
    ds.gt = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=np.int32,
    )

    out = ds[0]

    assert out["img1"].shape == (4, 8, 8)
    assert out["img2"].shape == (4, 8, 8)
    assert out["gt"].shape == (8, 8)
    assert out["gt"].dtype == torch.long
    assert set(out["gt"].unique().tolist()).issubset({0, 1})


def test_fixed_patch_mode_count_parsing() -> None:
    ds = HyperspectralChangeDetectionDataset.__new__(HyperspectralChangeDetectionDataset)
    ds.fixed_changed_num = 123
    ds.fixed_unchanged_num = 456

    assert ds._resolve_fixed_patch_counts("fixed") == (123, 456)
    assert ds._resolve_fixed_patch_counts("fixed_50") == (50, 50)
    assert ds._resolve_fixed_patch_counts("fixed_60_40") == (60, 40)

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
import torch

from src.stage2.segmentation.data.atlantic_forest import (
    AtlanticForestSegmentationDataset,
    AtlanticForestKeyTransform,
    get_atlantic_forest_dataloader,
)


def _write_pair(image_path: Path, mask_path: Path) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((8, 8, 4), 5000, dtype=np.uint16)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 1
    tifffile.imwrite(image_path, image)
    tifffile.imwrite(mask_path, mask)


def test_atlantic_forest_dataset_neg11_and_keys(tmp_path: Path) -> None:
    root = tmp_path / "AtlanticForeastSegmentation"
    _write_pair(
        root / "Training" / "image" / "a.tif",
        root / "Training" / "label" / "a.tif",
    )

    dataset = AtlanticForestSegmentationDataset(
        root=root,
        split="train",
        transform=AtlanticForestKeyTransform(),
        is_neg_1_1=True,
        norm_const=10000.0,
        augmentation=False,
    )
    sample = dataset[0]
    assert "img" in sample and "gt" in sample
    img = sample["img"]
    gt = sample["gt"]
    assert img.shape == (4, 8, 8)
    assert gt.shape == (8, 8)
    assert float(img.min()) >= -1.0 and float(img.max()) <= 1.0
    assert set(np.unique(gt.numpy()).tolist()) <= {0, 1}


def test_atlantic_forest_dataloader_val_split(tmp_path: Path) -> None:
    root = tmp_path / "AtlanticForeastSegmentation"
    _write_pair(
        root / "Validation" / "images" / "b.tif",
        root / "Validation" / "masks" / "b.tif",
    )

    _, dataloader = get_atlantic_forest_dataloader(
        batch_size=1,
        num_workers=0,
        split="val",
        root=str(root),
        transform=AtlanticForestKeyTransform(),
        is_neg_1_1=False,
        norm_const=10000.0,
        augmentation=False,
    )
    batch = next(iter(dataloader))
    assert "img" in batch and "gt" in batch
    assert batch["img"].shape == (1, 4, 8, 8)
    assert batch["gt"].shape == (1, 8, 8)
    assert float(batch["img"].min()) >= 0.0 and float(batch["img"].max()) <= 1.0


def test_atlantic_forest_norm_const_none_uses_min_max(tmp_path: Path) -> None:
    root = tmp_path / "AtlanticForeastSegmentation"
    img_path = root / "Training" / "image" / "c.tif"
    mask_path = root / "Training" / "label" / "c.tif"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    image = np.arange(8 * 8 * 4, dtype=np.uint16).reshape(8, 8, 4)
    mask = np.zeros((8, 8), dtype=np.uint8)
    tifffile.imwrite(img_path, image)
    tifffile.imwrite(mask_path, mask)

    dataset = AtlanticForestSegmentationDataset(
        root=root,
        split="train",
        transform=AtlanticForestKeyTransform(),
        is_neg_1_1=False,
        norm_const=None,
        augmentation=False,
    )
    sample = dataset[0]
    img = sample["img"]
    assert torch.isclose(img.min(), torch.tensor(0.0))
    assert torch.isclose(img.max(), torch.tensor(1.0))

from pathlib import Path
from typing import Literal

import pytest
from PIL import Image
import numpy as np

from src.stage2.change_detection.data.DSIFN import DSIFNChangeDetectionDataset


@pytest.mark.parametrize("split", ["train", "val"])
def test_dsifn_dataset_basic(split: Literal["train", "val"]) -> None:
    data_root = Path("data/Downstreams/ChangeDetection-DSIFN")
    if not data_root.exists():
        pytest.skip(f"DSIFN dataset not found at {data_root}")  # type: ignore

    dataset = DSIFNChangeDetectionDataset(
        data_root=data_root,
        split=split,
        mask_dir="mask",
        mask_subdir=None,
        transform=None,
        normalize=True,
        resize_to_mask=True,
    )

    assert len(dataset) > 0
    sample = dataset[0]
    assert set(sample.keys()) == {"img1", "img2", "gt"}

    img1 = sample["img1"]
    img2 = sample["img2"]
    gt = sample["gt"]

    assert img1.shape == img2.shape
    assert img1.shape[0] == 3
    assert gt.shape[0] == 1
    assert img1.shape[-2:] == gt.shape[-2:]

    gt_unique = gt.unique().tolist()
    assert set(gt_unique).issubset({0, 1})


def test_dsifn_dataset_all_split_with_explicit_layout() -> None:
    data_root = Path("data/Downstreams/ChangeDetection-DSIFN")
    if not data_root.exists() or not (data_root / "train").exists():
        pytest.skip("DSIFN explicit split layout not found.")  # type: ignore

    dataset = DSIFNChangeDetectionDataset(
        data_root=data_root,
        split="all",
        mask_dir="mask",
        mask_subdir=None,
        transform=None,
        normalize=False,
        resize_to_mask=False,
    )
    assert len(dataset) > 0


def _write_rgb(path: Path, value: int, hw: tuple[int, int] = (8, 8)) -> None:
    arr = np.full((hw[0], hw[1], 3), value, dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _write_mask(path: Path, value: int, hw: tuple[int, int] = (8, 8)) -> None:
    arr = np.full((hw[0], hw[1]), value, dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _build_minimal_split(
    root: Path,
    split: str,
    sid: str,
    image_hw: tuple[int, int] = (8, 8),
    mask_hw: tuple[int, int] = (8, 8),
) -> None:
    split_root = root / split
    for folder in ("t1", "t2", "mask"):
        (split_root / folder).mkdir(parents=True, exist_ok=True)
    _write_rgb(split_root / "t1" / f"{sid}.png", value=32, hw=image_hw)
    _write_rgb(split_root / "t2" / f"{sid}.png", value=96, hw=image_hw)
    _write_mask(split_root / "mask" / f"{sid}.png", value=255, hw=mask_hw)


def test_dsifn_explicit_split_layout_ignores_ratio() -> None:
    data_root = Path("outputs/tests/tmp_dsifn_explicit")
    _build_minimal_split(data_root, "train", "0001")
    _build_minimal_split(data_root, "val", "0002")
    _build_minimal_split(data_root, "test", "0003")

    dataset = DSIFNChangeDetectionDataset(
        data_root=data_root,
        split="train",
        mask_dir="mask",
        transform=None,
        train_ratio=1.5,
        val_ratio=0.9,
        normalize=False,
    )

    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["img1"].shape[-2:] == sample["gt"].shape[-2:]


def test_dsifn_resize_to_resizes_image_and_mask() -> None:
    data_root = Path("outputs/tests/tmp_dsifn_resize_to")
    _build_minimal_split(data_root, "train", "0001", image_hw=(10, 14), mask_hw=(6, 9))
    _build_minimal_split(data_root, "val", "0002", image_hw=(10, 14), mask_hw=(6, 9))
    _build_minimal_split(data_root, "test", "0003", image_hw=(10, 14), mask_hw=(6, 9))

    dataset = DSIFNChangeDetectionDataset(
        data_root=data_root,
        split="train",
        mask_dir="mask",
        transform=None,
        normalize=False,
        resize_to_mask=False,
        resize_to=12,
    )

    sample = dataset[0]
    assert sample["img1"].shape[-2:] == (12, 12)
    assert sample["img2"].shape[-2:] == (12, 12)
    assert sample["gt"].shape[-2:] == (12, 12)

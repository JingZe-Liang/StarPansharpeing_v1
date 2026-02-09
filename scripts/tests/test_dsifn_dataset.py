from pathlib import Path
from typing import Literal

import pytest

from src.stage2.change_detection.data.DSIFN import DSIFNChangeDetectionDataset


@pytest.mark.parametrize("split", ["train", "val"])
def test_dsifn_dataset_basic(split: Literal["train", "val"]) -> None:
    data_root = Path("data/Downstreams/ChangeDetection-DSIFN")
    if not data_root.exists():
        pytest.skip(f"DSIFN dataset not found at {data_root}")

    dataset = DSIFNChangeDetectionDataset(
        data_root=data_root,
        split=split,
        mask_dir="mask_256",
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
        pytest.skip("DSIFN explicit split layout not found.")

    dataset = DSIFNChangeDetectionDataset(
        data_root=data_root,
        split="all",
        mask_dir="mask_256",
        mask_subdir=None,
        transform=None,
        normalize=False,
        resize_to_mask=False,
    )
    assert len(dataset) > 0

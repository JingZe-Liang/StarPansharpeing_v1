from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.stage2.change_detection.data.gvlm_land_slide import (
    GVLMLandslideDataset,
    build_sample_paths,
    get_default_transform,
    load_label,
    load_rgb,
    read_list,
)

DATA_ROOT = Path("data/Downstreams/滑坡检测-GVLM/GVLM_CD256_0.3neg")


def _has_gvlm_data() -> bool:
    required = [
        DATA_ROOT / "list/train.txt",
        DATA_ROOT / "list/val.txt",
        DATA_ROOT / "list/test.txt",
        DATA_ROOT / "A",
        DATA_ROOT / "B",
        DATA_ROOT / "label",
    ]
    return all(path.exists() for path in required)


@pytest.mark.skipif(not _has_gvlm_data(), reason="GVLM landslide data not found.")
def test_read_list():
    train_list = DATA_ROOT / "list/train.txt"
    names = read_list(train_list)
    assert len(names) > 0
    assert all(name.endswith(".png") for name in names)


@pytest.mark.skipif(not _has_gvlm_data(), reason="GVLM landslide data not found.")
def test_build_sample_paths():
    samples = build_sample_paths(DATA_ROOT, "list", "train")
    assert len(samples) > 0
    sample = samples[0]
    assert sample.img1.exists()
    assert sample.img2.exists()
    assert sample.label.exists()
    assert sample.name.endswith(".png")


@pytest.mark.skipif(not _has_gvlm_data(), reason="GVLM landslide data not found.")
def test_load_rgb_and_label():
    samples = build_sample_paths(DATA_ROOT, "list", "train")
    sample = samples[0]

    img1 = load_rgb(sample.img1)
    img2 = load_rgb(sample.img2)
    label = load_label(sample.label)

    assert img1.shape == img2.shape
    assert len(img1.shape) == 3 and img1.shape[2] == 3  # H, W, C
    assert len(label.shape) == 2  # H, W
    assert img1.shape[:2] == label.shape


@pytest.mark.skipif(not _has_gvlm_data(), reason="GVLM landslide data not found.")
def test_dataset_length():
    ds_train = GVLMLandslideDataset(DATA_ROOT, split="train")
    ds_val = GVLMLandslideDataset(DATA_ROOT, split="val")
    ds_test = GVLMLandslideDataset(DATA_ROOT, split="test")

    assert len(ds_train) > 0
    assert len(ds_val) > 0
    assert len(ds_test) > 0


@pytest.mark.skipif(not _has_gvlm_data(), reason="GVLM landslide data not found.")
def test_dataset_getitem_shapes_and_types():
    ds = GVLMLandslideDataset(DATA_ROOT, split="train", transform=None)
    sample = ds[0]

    img1 = sample["img1"]
    img2 = sample["img2"]
    gt = sample["gt"]

    assert isinstance(img1, torch.Tensor)
    assert isinstance(img2, torch.Tensor)
    assert isinstance(gt, torch.Tensor)

    # Check shapes: C, H, W
    assert img1.ndim == 3
    assert img2.ndim == 3
    assert gt.ndim == 3  # 1, H, W before squeeze in original, but kept as 3D

    assert img1.shape[0] == 3  # RGB channels
    assert img2.shape[0] == 3
    assert gt.shape[0] == 1  # Single channel mask

    assert img1.shape[1:] == gt.shape[1:]
    assert img1.shape == img2.shape


@pytest.mark.skipif(not _has_gvlm_data(), reason="GVLM landslide data not found.")
def test_dataset_normalization():
    ds_normalized = GVLMLandslideDataset(DATA_ROOT, split="train", transform=None, normalize=True)
    ds_unnormalized = GVLMLandslideDataset(DATA_ROOT, split="train", transform=None, normalize=False)

    sample_norm = ds_normalized[0]
    sample_unnorm = ds_unnormalized[0]

    # Normalized should be in [0, 1]
    assert sample_norm["img1"].min() >= 0 and sample_norm["img1"].max() <= 1
    assert sample_norm["img2"].min() >= 0 and sample_norm["img2"].max() <= 1

    # Unnormalized should be in [0, 255]
    assert sample_unnorm["img1"].min() >= 0 and sample_unnorm["img1"].max() <= 255
    assert sample_unnorm["img2"].min() >= 0 and sample_unnorm["img2"].max() <= 255


@pytest.mark.skipif(not _has_gvlm_data(), reason="GVLM landslide data not found.")
def test_dataset_label_threshold():
    ds = GVLMLandslideDataset(DATA_ROOT, split="train", transform=None, label_threshold=127)
    sample = ds[0]
    gt = sample["gt"]

    # After thresholding, labels should be binary (0 or 1)
    unique_values = torch.unique(gt)
    assert all(v in [0, 1] for v in unique_values.tolist())


@pytest.mark.skipif(not _has_gvlm_data(), reason="GVLM landslide data not found.")
def test_dataset_return_name():
    ds_with_name = GVLMLandslideDataset(DATA_ROOT, split="train", transform=None, return_name=True)
    ds_without_name = GVLMLandslideDataset(DATA_ROOT, split="train", transform=None, return_name=False)

    sample_with = ds_with_name[0]
    sample_without = ds_without_name[0]

    assert "name" in sample_with
    assert "name" not in sample_without
    assert sample_with["name"].endswith(".png")


@pytest.mark.skipif(not _has_gvlm_data(), reason="GVLM landslide data not found.")
def test_dataset_with_default_transform():
    ds = GVLMLandslideDataset(DATA_ROOT, split="train", transform="default")
    sample = ds[0]

    assert "img1" in sample
    assert "img2" in sample
    assert "gt" in sample

    # Shapes should still be correct after transform
    assert sample["img1"].shape[0] == 3
    assert sample["img2"].shape[0] == 3
    assert sample["gt"].shape[0] == 1


@pytest.mark.skipif(not _has_gvlm_data(), reason="GVLM landslide data not found.")
def test_get_default_transform():
    transform = get_default_transform(prob=0.5)
    assert transform is not None


@pytest.mark.skipif(not _has_gvlm_data(), reason="GVLM landslide data not found.")
def test_dataset_gt_is_long():
    ds = GVLMLandslideDataset(DATA_ROOT, split="train", transform=None)
    sample = ds[0]
    gt = sample["gt"]

    assert gt.dtype == torch.long

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from torch import Tensor

from src.stage2.change_detection.data.xview2 import XView2ChangeDetectionDataset


def _to_uint8_rgb(img: Tensor) -> np.ndarray:
    arr = img.detach().cpu().numpy()
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(f"Expected CHW RGB tensor, got shape={arr.shape}")
    if arr.min() >= -1.0 and arr.max() <= 1.0:
        arr = (arr + 1.0) * 0.5
    elif arr.max() > 1.5:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    return arr


def _to_color_mask(mask: Tensor) -> np.ndarray:
    arr = mask.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected HW or 1HW mask tensor, got shape={mask.shape}")
    arr = arr.astype(np.int64)
    out = np.zeros((*arr.shape, 3), dtype=np.uint8)
    out[arr == 0] = np.array([0, 0, 0], dtype=np.uint8)
    out[arr != 0] = np.array([255, 0, 0], dtype=np.uint8)
    return out


def _save_triplet_image(sample: dict[str, Tensor | str], save_path: Path) -> None:
    img1 = _to_uint8_rgb(sample["img1"])  # type: ignore[arg-type]
    img2 = _to_uint8_rgb(sample["img2"])  # type: ignore[arg-type]
    gt = _to_color_mask(sample["gt"])  # type: ignore[arg-type]
    merged = np.concatenate([img1, img2, gt], axis=1)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(merged).save(save_path)


@pytest.mark.parametrize("split", ["train", "test"])
def test_xview2_dataset_shapes(split: str) -> None:
    data_root = Path("data/Downstreams/xView2_raw")
    if not data_root.exists():
        pytest.skip(f"xView2 data root not found: {data_root}")
    ds = XView2ChangeDetectionDataset(
        data_root=data_root,
        split=split,  # type: ignore[arg-type]
        include_hold_in_train=True,
        transform=None,
        normalize=True,
        img_to_neg1_1=False,
        resize_to_mask=True,
        binarize_gt=True,
        return_name=True,
    )
    assert len(ds) > 0
    sample = ds[0]
    img1 = sample["img1"]
    img2 = sample["img2"]
    gt = sample["gt"]
    assert isinstance(img1, Tensor) and isinstance(img2, Tensor) and isinstance(gt, Tensor)
    assert img1.shape == img2.shape
    assert gt.ndim == 3 and gt.shape[0] == 1
    assert img1.shape[-2:] == gt.shape[-2:]


def test_xview2_plot_triplet() -> None:
    data_root = Path("data/Downstreams/xView2_raw")
    if not data_root.exists():
        pytest.skip(f"xView2 data root not found: {data_root}")

    ds = XView2ChangeDetectionDataset(
        data_root=data_root,
        split="train",
        include_hold_in_train=True,
        transform=None,
        normalize=True,
        img_to_neg1_1=False,
        resize_to_mask=True,
        binarize_gt=True,
        return_name=True,
    )
    sample = ds[0]
    out_path = Path("tmp/xview2_triplet_sample.png")
    _save_triplet_image(sample, out_path)
    assert out_path.exists()

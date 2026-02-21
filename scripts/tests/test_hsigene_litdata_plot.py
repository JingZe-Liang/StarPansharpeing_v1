from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pytest
import torch
import tifffile
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.litdata_hyperloader import *


def _decode_field(value: Any, prefer_tiff: bool) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, Image.Image):
        return np.array(value)
    if isinstance(value, bytes):
        if prefer_tiff:
            return tifffile.imread(io.BytesIO(value))
        with Image.open(io.BytesIO(value)) as im:
            return np.array(im)
    raise TypeError(f"Unsupported sample field type: {type(value)}")


def _to_chw(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Only 2D/3D arrays are supported, got shape={arr.shape}")

    c_first = arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]
    c_last = arr.shape[2] < arr.shape[0] and arr.shape[2] < arr.shape[1]

    if c_first:
        return arr
    if c_last:
        return np.transpose(arr, (2, 0, 1))
    return arr


def _normalize_01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max <= x_min:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min + 1e-8)


def _hsi_to_rgb(chw: np.ndarray, rgb_channels: list[int]) -> np.ndarray:
    if chw.shape[0] == 1:
        g = _normalize_01(chw[0])
        return np.stack([g, g, g], axis=-1)

    if chw.shape[0] == 2:
        c0 = _normalize_01(chw[0])
        c1 = _normalize_01(chw[1])
        c2 = (c0 + c1) * 0.5
        return np.stack([c0, c1, c2], axis=-1)

    idx = [min(max(i, 0), chw.shape[0] - 1) for i in rgb_channels]
    rgb = np.stack([_normalize_01(chw[i]) for i in idx], axis=-1)
    return rgb


def _cond_to_display(arr: np.ndarray) -> np.ndarray:
    chw = _to_chw(arr)
    if chw.shape[0] == 1:
        g = _normalize_01(chw[0])
        return np.stack([g, g, g], axis=-1)
    if chw.shape[0] >= 3:
        return np.stack([_normalize_01(chw[0]), _normalize_01(chw[1]), _normalize_01(chw[2])], axis=-1)
    g = _normalize_01(chw[0])
    return np.stack([g, g, g], axis=-1)


def _save_one_visual(
    sample_key: str,
    img_rgb: np.ndarray,
    cond_map: dict[str, np.ndarray],
    caption_text: str,
    save_path: Path,
) -> None:
    cond_names = ["hed", "segmentation", "sketch", "mlsd"]
    fig, axes = plt.subplots(1, 2 + len(cond_names), figsize=(24, 4))
    axes[0].imshow(np.clip(img_rgb, 0.0, 1.0))
    axes[0].set_title("img")
    axes[0].axis("off")

    for i, name in enumerate(cond_names, start=1):
        arr = cond_map.get(name)
        if arr is None:
            axes[i].text(0.5, 0.5, f"{name}\nmissing", ha="center", va="center")
            axes[i].set_title(name)
            axes[i].axis("off")
            continue
        axes[i].imshow(np.clip(arr, 0.0, 1.0))
        axes[i].set_title(name)
        axes[i].axis("off")

    caption_show = caption_text.replace("\n", " ").strip()
    if len(caption_show) > 320:
        caption_show = caption_show[:317] + "..."
    axes[-1].set_title("caption")
    axes[-1].axis("off")
    axes[-1].text(
        0.02,
        0.98,
        caption_show if caption_show else "(empty)",
        va="top",
        ha="left",
        wrap=True,
        fontsize=9,
    )

    fig.suptitle(sample_key)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_split(
    root: Path,
    split: str,
    n_samples: int,
    rgb_channels: list[int],
    out_root: Path,
) -> list[Path]:
    try:
        litdata = pytest.importorskip("litdata")
    except Exception as e:  # pragma: no cover
        pytest.skip(f"Cannot import litdata in current environment: {e}")
    img_dir = root / "LitData_hyper_images" / split
    cond_dir = root / "LitData_conditions" / split
    cap_dir = root / "LitData_image_captions" / split
    if not img_dir.exists() or not cond_dir.exists() or not cap_dir.exists():
        pytest.skip(f"Missing split dirs: {img_dir} or {cond_dir} or {cap_dir}")

    img_ds = litdata.StreamingDataset(input_dir=str(img_dir))
    cond_ds = litdata.StreamingDataset(input_dir=str(cond_dir))
    cap_ds = litdata.StreamingDataset(input_dir=str(cap_dir))

    n = min(n_samples, len(img_ds), len(cond_ds), len(cap_ds))
    if n <= 0:
        pytest.skip(f"No samples in split={split}")

    saved_paths: list[Path] = []
    for idx in range(n):
        img_sample = img_ds[idx]
        cond_sample = cond_ds[idx]
        cap_sample = cap_ds[idx]
        key = str(img_sample.get("__key__", f"{split}_{idx:05d}"))

        img_arr = _decode_field(img_sample["img"], prefer_tiff=True)
        img_chw = _to_chw(img_arr)
        img_rgb = _hsi_to_rgb(img_chw, rgb_channels=rgb_channels)

        cond_map: dict[str, np.ndarray] = {}
        for name in ["hed", "segmentation", "sketch", "mlsd"]:
            if name not in cond_sample:
                continue
            cond_arr = _decode_field(cond_sample[name], prefer_tiff=False)
            cond_map[name] = _cond_to_display(cond_arr)

        caption_val = cap_sample.get("caption", "")
        if isinstance(caption_val, dict):
            caption_text = str(caption_val.get("caption", ""))
        else:
            caption_text = str(caption_val)

        save_path = out_root / split / f"{idx:03d}_{key}.png"
        _save_one_visual(
            sample_key=key,
            img_rgb=img_rgb,
            cond_map=cond_map,
            caption_text=caption_text,
            save_path=save_path,
        )
        saved_paths.append(save_path)

    return saved_paths


@pytest.mark.parametrize("split", ["train", "test"])
def test_plot_hsigene_litdata_samples(split: str) -> None:
    root = Path(os.environ.get("HSIGENE_DATA_ROOT", "data2/HSIGene_dataset"))
    out_root = Path(os.environ.get("HSIGENE_VIS_OUT", "tmp/hsigene_litdata_vis"))
    n_samples = int(os.environ.get("HSIGENE_VIS_N", "4"))
    rgb_channels = [int(x.strip()) for x in os.environ.get("HSIGENE_RGB", "20,12,5").split(",")]
    if len(rgb_channels) != 3:
        raise ValueError("HSIGENE_RGB must have three comma-separated integers.")

    saved_paths = _plot_split(root, split, n_samples, rgb_channels, out_root)
    assert len(saved_paths) > 0
    assert all(p.exists() for p in saved_paths)

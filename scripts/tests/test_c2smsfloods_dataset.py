from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytest
import tifffile
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.stage2.change_detection.data.c2smsfloods import (
    C2SMSFloodChangeDetectionDataset,
    create_c2smsfloods_change_detection_dataloader,
)


def _make_s1_chip(chip_dir: Path) -> None:
    chip_dir.mkdir(parents=True, exist_ok=True)
    vv = np.linspace(-20.0, 1.0, num=16, dtype=np.float32).reshape(4, 4)
    vh = np.linspace(-25.0, -1.0, num=16, dtype=np.float32).reshape(4, 4)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1

    tifffile.imwrite(chip_dir / "VV.tif", vv)
    tifffile.imwrite(chip_dir / "VH.tif", vh)
    tifffile.imwrite(chip_dir / "LabelWater.tif", mask)


def _make_s2_chip(chip_dir: Path) -> None:
    chip_dir.mkdir(parents=True, exist_ok=True)

    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[..., 0] = 10
    rgb[..., 1] = 40
    rgb[..., 2] = 80

    swir = np.zeros((4, 4, 3), dtype=np.uint8)
    swir[..., 0] = 20
    swir[..., 1] = 60
    swir[..., 2] = 100

    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0:2, 0:2] = 1

    Image.fromarray(rgb).save(chip_dir / "RGB.png")
    Image.fromarray(swir).save(chip_dir / "SWIR.png")
    tifffile.imwrite(chip_dir / "LabelWater.tif", mask)

    # Use different shapes to verify resize in s2_all mode.
    tifffile.imwrite(chip_dir / "B1.tif", np.full((2, 2), 100, dtype=np.float32))
    tifffile.imwrite(chip_dir / "B2.tif", np.full((4, 4), 500, dtype=np.float32))
    tifffile.imwrite(chip_dir / "B3.tif", np.full((4, 4), 600, dtype=np.float32))


def _build_minimal_dataset_root(tmp_path: Path, chips_dir_name: str = "ms-dataset-chips") -> Path:
    root = tmp_path / "c2smsfloods"
    chips = root / chips_dir_name

    scene_ids = ["scene_a", "scene_b"]
    # Ensure paired s1/s2 chips share position suffix for sensor=all matching.
    s1_names = ["S1_xxx_00000-00000", "S1_xxx_00016-00016"]
    s2_names = ["S2_xxx_00000-00000", "S2_xxx_00016-00016"]

    for sid, s1_name, s2_name in zip(scene_ids, s1_names, s2_names, strict=True):
        _make_s1_chip(chips / sid / "s1" / s1_name)
        _make_s2_chip(chips / sid / "s2" / s2_name)

    return root


def _to_display_img(chw: torch.Tensor) -> np.ndarray:
    arr = chw.detach().cpu().float().numpy()
    if arr.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape={arr.shape}")
    c, h, w = arr.shape
    if c == 1:
        img = arr[0]
    elif c == 2:
        pseudo = np.zeros((3, h, w), dtype=np.float32)
        pseudo[0] = arr[0]
        pseudo[1] = arr[1]
        pseudo[2] = arr[0] - arr[1]
        img = np.transpose(pseudo, (1, 2, 0))
    else:
        img = np.transpose(arr[:3], (1, 2, 0))

    img_min = float(np.min(img))
    img_max = float(np.max(img))
    scale = max(img_max - img_min, 1e-6)
    img = (img - img_min) / scale
    return np.clip(img, 0.0, 1.0)


def _save_modalities_figure(sample: dict[str, torch.Tensor | str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not isinstance(sample["s1"], torch.Tensor):
        raise TypeError("sample['s1'] must be Tensor")
    if not isinstance(sample["s2_rgb"], torch.Tensor):
        raise TypeError("sample['s2_rgb'] must be Tensor")
    if not isinstance(sample["s2_all"], torch.Tensor):
        raise TypeError("sample['s2_all'] must be Tensor")
    if not isinstance(sample["gt"], torch.Tensor):
        raise TypeError("sample['gt'] must be Tensor")

    s1 = _to_display_img(sample["s1"])
    s2_rgb_pre = _to_display_img(sample["s2_rgb"][:3])
    s2_rgb_post = _to_display_img(sample["s2_rgb"][3:])
    s2_all = _to_display_img(sample["s2_all"])
    gt = sample["gt"].squeeze(0).detach().cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.reshape(-1)
    axes[0].imshow(s1, cmap="gray" if s1.ndim == 2 else None)
    axes[0].set_title("S1 (VV+VH)")
    axes[1].imshow(s2_rgb_pre)
    axes[1].set_title("S2 RGB")
    axes[2].imshow(s2_rgb_post)
    axes[2].set_title("S2 SWIR")
    axes[3].imshow(s2_all)
    axes[3].set_title("S2 ALL Bands")
    axes[4].imshow(gt, cmap="gray")
    axes[4].set_title("GT")
    axes[5].axis("off")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def test_c2smsfloods_s1_dataset_output(tmp_path: Path) -> None:
    root = _build_minimal_dataset_root(tmp_path)

    ds = C2SMSFloodChangeDetectionDataset(
        data_root=root,
        split="all",
        sensor="s1",
        transform=None,
        normalize=True,
        return_name=True,
    )

    sample = ds[0]
    assert set(sample) == {"img", "gt", "name"}
    assert sample["img"].shape == (2, 4, 4)
    assert sample["gt"].shape == (1, 4, 4)


def test_c2smsfloods_s2_rgb_swir_output(tmp_path: Path) -> None:
    root = _build_minimal_dataset_root(tmp_path)

    ds = C2SMSFloodChangeDetectionDataset(
        data_root=root,
        split="all",
        sensor="s2_rgb_swir",
        transform=None,
        normalize=False,
    )

    sample = ds[0]
    assert set(sample) == {"img", "gt"}
    assert sample["img"].shape == (6, 4, 4)
    assert sample["gt"].shape == (1, 4, 4)


def test_c2smsfloods_s2_all_output_resized_and_cat(tmp_path: Path) -> None:
    root = _build_minimal_dataset_root(tmp_path)

    ds = C2SMSFloodChangeDetectionDataset(
        data_root=root,
        split="all",
        sensor="s2_all",
        transform=None,
        normalize=False,
    )

    sample = ds[0]
    assert set(sample) == {"img", "gt"}
    assert sample["img"].shape == (3, 4, 4)
    assert sample["gt"].shape == (1, 4, 4)


def test_c2smsfloods_all_mode_output(tmp_path: Path) -> None:
    root = _build_minimal_dataset_root(tmp_path)

    ds = C2SMSFloodChangeDetectionDataset(
        data_root=root,
        split="all",
        sensor="all",
        transform=None,
        normalize=False,
    )

    sample = ds[0]
    assert set(sample) == {"s1", "s2_rgb", "s2_all", "gt"}
    assert sample["s1"].shape == (2, 4, 4)
    assert sample["s2_rgb"].shape == (6, 4, 4)
    assert sample["s2_all"].shape == (3, 4, 4)
    assert sample["gt"].shape == (1, 4, 4)


def test_c2smsfloods_supports_custom_chips_dir_name(tmp_path: Path) -> None:
    root = _build_minimal_dataset_root(tmp_path, chips_dir_name="chips")

    ds = C2SMSFloodChangeDetectionDataset(
        data_root=root,
        split="all",
        sensor="s1",
        chips_dir_name="chips",
        transform=None,
    )
    assert len(ds) == 2


def test_c2smsfloods_s2_rgb_swir_dataloader_output(tmp_path: Path) -> None:
    root = _build_minimal_dataset_root(tmp_path)

    dataset, loader = create_c2smsfloods_change_detection_dataloader(
        data_root=root,
        split="train",
        sensor="s2_rgb_swir",
        train_ratio=0.5,
        val_ratio=0.0,
        batch_size=1,
        num_workers=0,
        transform=None,
        normalize=False,
    )

    assert len(dataset) == 1
    batch = next(iter(loader))
    assert set(batch) == {"img", "gt"}
    assert batch["img"].shape == (1, 6, 4, 4)
    assert batch["gt"].shape == (1, 1, 4, 4)


def test_c2smsfloods_all_mode_dataloader_output(tmp_path: Path) -> None:
    root = _build_minimal_dataset_root(tmp_path)

    dataset, loader = create_c2smsfloods_change_detection_dataloader(
        data_root=root,
        split="train",
        sensor="all",
        train_ratio=0.5,
        val_ratio=0.0,
        batch_size=1,
        num_workers=0,
        transform=None,
        normalize=False,
    )

    assert len(dataset) == 1
    batch = next(iter(loader))
    assert set(batch) == {"s1", "s2_rgb", "s2_all", "gt"}
    assert batch["s1"].shape == (1, 2, 4, 4)
    assert batch["s2_rgb"].shape == (1, 6, 4, 4)
    assert batch["s2_all"].shape == (1, 3, 4, 4)
    assert batch["gt"].shape == (1, 1, 4, 4)


def test_c2smsfloods_persistent_split_no_overlap(tmp_path: Path) -> None:
    root = _build_minimal_dataset_root(tmp_path)
    split_file = tmp_path / "splits" / "scene_split.json"

    train_ds = C2SMSFloodChangeDetectionDataset(
        data_root=root,
        split="train",
        sensor="s1",
        split_file=split_file,
        train_ratio=0.5,
        val_ratio=0.0,
        transform=None,
    )
    test_ds = C2SMSFloodChangeDetectionDataset(
        data_root=root,
        split="test",
        sensor="s1",
        split_file=split_file,
        train_ratio=0.5,
        val_ratio=0.0,
        transform=None,
    )

    train_names = {s.name for s in train_ds.samples}
    test_names = {s.name for s in test_ds.samples}
    assert train_names.isdisjoint(test_names)
    assert split_file.exists()


def test_c2smsfloods_crop_resize_output_shape(tmp_path: Path) -> None:
    root = _build_minimal_dataset_root(tmp_path)
    ds = C2SMSFloodChangeDetectionDataset(
        data_root=root,
        split="all",
        sensor="s1",
        crop_size=2,
        resize_to=3,
        random_crop=False,
        transform=None,
        normalize=False,
    )
    sample = ds[0]
    assert sample["img"].shape == (2, 3, 3)
    assert sample["gt"].shape == (1, 3, 3)


def test_c2smsfloods_plot_modalities_to_tmp(tmp_path: Path) -> None:
    real_root = Path("data/Downstreams/c2smsfloods")
    if not real_root.exists():
        pytest.skip("Real c2smsfloods data not found at data/Downstreams/c2smsfloods")

    ds = C2SMSFloodChangeDetectionDataset(
        data_root=real_root,
        chips_dir_name="chips",
        split="all",
        sensor="all",
        transform=None,
        normalize=True,
    )
    if len(ds) == 0:
        pytest.skip("No paired samples available for sensor='all' on real data")

    sample = ds[0]
    tensor_sample = {
        "s1": sample["s1"],
        "s2_rgb": sample["s2_rgb"],
        "s2_all": sample["s2_all"],
        "gt": sample["gt"],
    }
    out_path = Path("tmp") / "c2smsfloods_modalities" / "sample_modalities_real.png"
    _save_modalities_figure(tensor_sample, out_path=out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


# Touch torch symbol to avoid accidental import removal by future edits.
assert torch.__version__

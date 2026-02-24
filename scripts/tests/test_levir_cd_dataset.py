from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest
import torch

from src.stage2.change_detection.data.levir_cd import (
    LEVIRCDDataset,
    build_sample_paths,
    create_levir_cd_dataloader,
    read_list,
)

matplotlib.use("Agg")

DATA_ROOT = Path("data/Downstreams/ChangeDetection/LEVIR-CD/LEVIR-CD256")


def _has_levir_data() -> bool:
    required = [
        DATA_ROOT / "A",
        DATA_ROOT / "B",
        DATA_ROOT / "label",
        DATA_ROOT / "list/train.txt",
        DATA_ROOT / "list/val.txt",
        DATA_ROOT / "list/test.txt",
        DATA_ROOT / "list/trainval.txt",
    ]
    return all(path.exists() for path in required)


def _to_rgb_display(img: torch.Tensor) -> torch.Tensor:
    x = img.detach().cpu().float()
    if x.ndim != 3 or x.shape[0] != 3:
        raise ValueError(f"Expected image as [3,H,W], got {tuple(x.shape)}")
    if float(x.max()) > 1.0 or float(x.min()) < 0.0:
        x = x / 255.0
    return x.clamp(0.0, 1.0).permute(1, 2, 0)


@pytest.mark.skipif(not _has_levir_data(), reason="LEVIR-CD256 data not found.")
def test_read_train_list() -> None:
    names = read_list(DATA_ROOT / "list" / "train.txt")
    assert len(names) > 0
    assert all(name.endswith(".png") for name in names)


@pytest.mark.skipif(not _has_levir_data(), reason="LEVIR-CD256 data not found.")
def test_build_sample_paths_with_trainval_alias() -> None:
    samples = build_sample_paths(DATA_ROOT, "list", "train_val")
    assert len(samples) > 0
    assert samples[0].img1.exists()
    assert samples[0].img2.exists()
    assert samples[0].label.exists()


@pytest.mark.skipif(not _has_levir_data(), reason="LEVIR-CD256 data not found.")
def test_dataset_getitem_shapes() -> None:
    ds = LEVIRCDDataset(
        data_root=DATA_ROOT,
        split="train",
        transform=None,
        normalize=True,
        img_to_neg1_1=False,
    )
    sample = ds[0]
    assert isinstance(sample["img1"], torch.Tensor)
    assert isinstance(sample["img2"], torch.Tensor)
    assert isinstance(sample["gt"], torch.Tensor)
    assert sample["img1"].shape[0] == 3
    assert sample["img2"].shape[0] == 3
    assert sample["gt"].shape[0] == 1
    assert sample["img1"].shape[1:] == sample["gt"].shape[1:]


@pytest.mark.skipif(not _has_levir_data(), reason="LEVIR-CD256 data not found.")
def test_dataset_default_transform() -> None:
    ds = LEVIRCDDataset(data_root=DATA_ROOT, split="train", transform="default")
    sample = ds[0]
    assert sample["img1"].shape[0] == 3
    assert sample["img2"].shape[0] == 3
    assert sample["gt"].shape[0] == 1


@pytest.mark.skipif(not _has_levir_data(), reason="LEVIR-CD256 data not found.")
def test_dataloader_batch() -> None:
    dataset, loader = create_levir_cd_dataloader(
        data_root=DATA_ROOT,
        split="val",
        batch_size=2,
        shuffle=False,
        num_workers=0,
        transform=None,
        normalize=True,
    )
    assert len(dataset) > 0
    batch = next(iter(loader))
    assert batch["img1"].ndim == 4
    assert batch["img2"].ndim == 4
    assert batch["gt"].ndim == 4
    assert batch["img1"].shape[1] == 3
    assert batch["gt"].shape[1] == 1


@pytest.mark.skipif(not _has_levir_data(), reason="LEVIR-CD256 data not found.")
def test_plot_sample_to_tmp(tmp_path: Path) -> None:
    ds = LEVIRCDDataset(data_root=DATA_ROOT, split="train", transform=None, normalize=True, return_name=True)
    sample = ds[0]

    if not isinstance(sample["img1"], torch.Tensor):
        raise TypeError("img1 must be Tensor")
    if not isinstance(sample["img2"], torch.Tensor):
        raise TypeError("img2 must be Tensor")
    if not isinstance(sample["gt"], torch.Tensor):
        raise TypeError("gt must be Tensor")

    img1 = _to_rgb_display(sample["img1"])
    img2 = _to_rgb_display(sample["img2"])
    gt = sample["gt"].squeeze(0).detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img1.numpy())
    axes[0].set_title("img1")
    axes[1].imshow(img2.numpy())
    axes[1].set_title("img2")
    axes[2].imshow(gt, cmap="gray")
    axes[2].set_title("gt")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()

    out_path = tmp_path / "levir_cd_sample.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    assert out_path.exists()
    assert out_path.stat().st_size > 0
    print(f"save test image at {out_path}")

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest
from torch import Tensor

from src.stage2.change_detection.data.DSIFN import create_dsifn_change_detection_dataloader

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_ROOT = Path("data/Downstreams/ChangeDetection-DSIFN")
OUT_DIR = Path("outputs/tests/dsifn_preview")


def _to_plot_rgb(image: Tensor) -> np.ndarray:
    arr = image.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
    min_v = float(arr.min())
    max_v = float(arr.max())
    if max_v > 1.0 or min_v < 0.0:
        denom = max_v - min_v
        if denom > 0.0:
            arr = (arr - min_v) / denom
        else:
            arr = np.zeros_like(arr, dtype=np.float32)
    return np.clip(arr, 0.0, 1.0)


def _to_plot_mask(mask: Tensor) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[0]
    arr = mask.detach().cpu().numpy().astype(np.float32)
    return np.clip(arr, 0.0, 1.0)


def _save_batch_preview(img1: Tensor, img2: Tensor, gt: Tensor, batch_idx: int, out_dir: Path) -> Path:
    batch_size = int(img1.shape[0])
    fig, axes = plt.subplots(batch_size, 3, figsize=(11, 3.8 * batch_size), squeeze=False)

    for row in range(batch_size):
        axes[row, 0].imshow(_to_plot_rgb(img1[row]))
        axes[row, 0].set_title(f"batch{batch_idx} sample{row} t1")
        axes[row, 1].imshow(_to_plot_rgb(img2[row]))
        axes[row, 1].set_title(f"batch{batch_idx} sample{row} t2")
        axes[row, 2].imshow(_to_plot_mask(gt[row]), cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 2].set_title(f"batch{batch_idx} sample{row} mask")
        for col in range(3):
            axes[row, col].axis("off")

    fig.tight_layout()
    out_path = out_dir / f"batch_{batch_idx:02d}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def test_dsifn_dataloader_visualize_batches() -> None:
    if not DATA_ROOT.exists():
        pytest.skip(f"DSIFN dataset not found at {DATA_ROOT}")  # type: ignore

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _, dataloader = create_dsifn_change_detection_dataloader(
        data_root=DATA_ROOT,
        split="test",
        mask_dir="mask",
        batch_size=2,
        shuffle=False,
        num_workers=0,
        transform=None,
        normalize=True,
        img_to_neg1_1=False,
        resize_to_mask=False,
        resize_to=128,
        pin_memory=False,
        drop_last=False,
    )

    max_batches = 3
    saved_paths: list[Path] = []
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        saved_paths.append(
            _save_batch_preview(
                img1=batch["img1"],
                img2=batch["img2"],
                gt=batch["gt"],
                batch_idx=batch_idx,
                out_dir=OUT_DIR,
            )
        )

    assert saved_paths, "No DSIFN preview images were saved."
    for path in saved_paths:
        assert path.exists()

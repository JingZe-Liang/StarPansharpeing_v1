from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
import torch


def _to_chw(img: torch.Tensor) -> torch.Tensor:
    if img.ndim != 3:
        return img
    if img.shape[0] in (2, 13):
        return img  # CHW
    if img.shape[-1] in (2, 13):
        return img.permute(2, 0, 1)  # HWC -> CHW
    if img.shape[1] in (2, 13):
        return img.permute(1, 2, 0)  # WCH -> CHW (litdata tiff transpose artifact)
    return img


def _quantile_stretch(
    img: torch.Tensor,
    *,
    q_low: float = 0.02,
    q_high: float = 0.98,
    eps: float = 1e-6,
) -> torch.Tensor:
    flat = img.flatten()
    lo = torch.quantile(flat, q_low)
    hi = torch.quantile(flat, q_high)
    img = img.clamp(lo, hi)
    return (img - lo) / (hi - lo + eps)


def s2_to_rgb_for_display(
    s2: torch.Tensor,
    *,
    rgb_bands_0based: tuple[int, int, int] = (3, 2, 1),  # B04,B03,B02
    q_low: float = 0.02,
    q_high: float = 0.98,
    gamma: float = 2.2,
) -> torch.Tensor:
    """Convert S2 (CHW/HWC/WCH) to RGB HWC in [0, 1] for visualization."""
    s2 = _to_chw(s2).float()
    if s2.ndim != 3:
        raise ValueError(f"Expected S2 with 3 dims, got {s2.shape}")
    if s2.shape[0] <= max(rgb_bands_0based):
        raise ValueError(f"Invalid rgb bands {rgb_bands_0based} for S2 shape {s2.shape}")

    rgb = s2[list(rgb_bands_0based)]
    rgb = _quantile_stretch(rgb, q_low=q_low, q_high=q_high)
    if gamma > 0:
        rgb = rgb.clamp(0, 1) ** (1.0 / gamma)
    return rgb.permute(1, 2, 0).contiguous()


def s1_to_gray_for_display(
    s1: torch.Tensor,
    *,
    channel: Literal["vv", "vh", "mean"] = "vv",
    domain: Literal["db", "db10"] = "db",
    linear_compress: Literal["none", "log1p", "sqrt"] = "log1p",
    q_low: float = 0.02,
    q_high: float = 0.98,
    smooth: Literal["none", "median", "gaussian"] = "median",
    kernel_size: int = 3,
) -> torch.Tensor:
    """Convert S1 (CHW/HWC/WCH) to grayscale HW in [0, 1] for visualization."""
    s1 = _to_chw(s1).float()
    if s1.ndim != 3:
        raise ValueError(f"Expected S1 with 3 dims, got {s1.shape}")
    if s1.shape[0] != 2:
        raise ValueError(f"Expected S1 with 2 channels, got {s1.shape}")

    if channel == "vv":
        img = s1[0]
    elif channel == "vh":
        img = s1[1]
    else:
        img = s1.mean(dim=0)

    if domain == "db10":
        if img.min() < -1.0:
            img = torch.pow(10.0, img / 10.0)
            if linear_compress == "log1p":
                img = torch.log1p(img)
            elif linear_compress == "sqrt":
                img = torch.sqrt(torch.clamp_min(img, 0.0))
            elif linear_compress != "none":
                raise ValueError(f"Unsupported linear_compress: {linear_compress}")

    img = _quantile_stretch(img, q_low=q_low, q_high=q_high)

    if smooth != "none":
        import kornia.filters as kf

        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError(f"kernel_size must be a positive odd number, got {kernel_size}")
        img_4d = img[None, None, ...]
        if smooth == "median":
            img_4d = kf.median_blur(img_4d, (kernel_size, kernel_size))
        else:
            img_4d = kf.gaussian_blur2d(img_4d, (kernel_size, kernel_size), (1.0, 1.0))
        img = img_4d[0, 0]

    return img.clamp(0, 1).contiguous()


def _read_tiff_chw(path: Path) -> torch.Tensor:
    with rasterio.open(path) as ds:
        arr = ds.read()
    return torch.from_numpy(arr)


def load_sen12_triplet(
    *,
    base_dir: str | Path,
    season: str,
    scene_id: int,
    patch_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base_dir = Path(base_dir)
    s1 = base_dir / season / f"{season}_s1" / f"s1_{scene_id}" / f"{season}_s1_{scene_id}_p{patch_id}.tif"
    s2 = base_dir / season / f"{season}_s2" / f"s2_{scene_id}" / f"{season}_s2_{scene_id}_p{patch_id}.tif"
    s2c = (
        base_dir
        / season
        / f"{season}_s2_cloudy"
        / f"s2_cloudy_{scene_id}"
        / f"{season}_s2_cloudy_{scene_id}_p{patch_id}.tif"
    )
    return _read_tiff_chw(s1), _read_tiff_chw(s2), _read_tiff_chw(s2c)


def plot_sen12_triplet(
    s1: torch.Tensor,
    s2: torch.Tensor,
    s2_cloudy: torch.Tensor,
    title: str | None = None,
    save_path: str | Path | None = "tmp/sen12_cr_preview.png",
    s1_domain: Literal["db", "db10"] = "db10",
    s1_linear_compress: Literal["none", "log1p", "sqrt"] = "log1p",
    s1_smooth: Literal["none", "median", "gaussian"] = "median",
    s1_kernel_size: int = 3,
) -> None:
    s1_img = s1_to_gray_for_display(
        s1,
        channel="mean",
        domain=s1_domain,
        smooth=s1_smooth,
        kernel_size=s1_kernel_size,
        linear_compress=s1_linear_compress,
    )
    s2_rgb = s2_to_rgb_for_display(s2)
    s2c_rgb = s2_to_rgb_for_display(s2_cloudy)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(s1_img.cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("S1 (Mean)")
    axes[1].imshow(s2_rgb.cpu().numpy())
    axes[1].set_title("S2 (Clean) RGB")
    axes[2].imshow(s2c_rgb.cpu().numpy())
    axes[2].set_title("S2 (Cloudy) RGB")

    for ax in axes:
        ax.axis("off")
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()

    if save_path is None:
        plt.show()
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

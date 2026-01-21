from __future__ import annotations
from duckdb.experimental.spark.sql.functions import printf

import os
from pathlib import Path
from typing import cast

import hydra
import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torchmetrics.functional.image import peak_signal_noise_ratio

_IMPORT_ERROR: Exception | None = None
# try:
from src.stage2.cloud_removal.data.CUHK_CR import (
    CUHK_CR_StreamingDataset,
    CUHK_CR_StreamingDataset_RandomKey_For_tokenizerPEFT,
)
from src.stage2.cloud_removal.diffusion.vae import CosmosRSVAE
# except PermissionError as exc:
#     _IMPORT_ERROR = exc
#     CUHK_CR_StreamingDataset = None  # type: ignore[assignment]
#     CosmosRSVAE = None  # type: ignore[assignment]

from loguru import logger

logger.disable("src.stage1.cosmos")


def _skip_if_import_failed() -> None:
    print("Skip ...")
    if _IMPORT_ERROR is not None:
        pytest.skip(f"Environment PermissionError during import (likely multiprocessing semaphore): {_IMPORT_ERROR}")


def _find_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Cannot find repo root containing pyproject.toml")


def _to_0_1(x: Tensor, *, is_neg_1_1: bool) -> Tensor:
    x = x.float()
    if is_neg_1_1:
        return ((x + 1.0) / 2.0).clamp(0.0, 1.0)
    return x.clamp(0.0, 1.0)


def _choose_device() -> torch.device:
    forced = os.getenv("CUHK_CR_VAE_PSNR_DEVICE", "").strip().lower()
    if forced in {"cpu", "cuda"}:
        if forced == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(forced)

    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        # 如果空闲显存太少（例如 <4GiB），直接用 CPU 避免 OOM
        if free_bytes < 4 * 1024**3:
            return torch.device("cpu")
    except Exception:
        # mem_get_info 可能在某些环境不可用；保守起见仍尝试 cuda
        pass

    return torch.device("cuda")


def _parse_int_list(csv: str) -> list[int]:
    parts = [p.strip() for p in csv.split(",") if p.strip()]
    return [int(p) for p in parts]


def _to_display_rgb(x_01: Tensor, *, rgb_channels: list[int] | None = None) -> Tensor:
    if x_01.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape={tuple(x_01.shape)}")

    b, c, _, _ = x_01.shape
    if c == 3:
        return x_01
    if c == 1:
        return x_01.repeat(1, 3, 1, 1)

    if rgb_channels is None:
        rgb_channels = [0, 1, 2]

    if len(rgb_channels) != 3:
        raise ValueError(f"rgb_channels must have length 3, got {rgb_channels}")

    rgb_channels = [ch if ch >= 0 else c + ch for ch in rgb_channels]
    if any((ch < 0 or ch >= c) for ch in rgb_channels):
        raise ValueError(f"rgb_channels out of range for c={c}: {rgb_channels}")

    return x_01[:, rgb_channels]


def _save_batch_recon_figure(
    *,
    out_path: Path,
    img_01: Tensor,
    gt_01: Tensor,
    rec_01: Tensor,
    max_rows: int = 4,
    rgb_channels: list[int] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    if not (img_01.shape == gt_01.shape == rec_01.shape):
        raise ValueError(
            f"Expected img/gt/rec to have the same shape, got {tuple(img_01.shape)=}, {tuple(gt_01.shape)=}, {tuple(rec_01.shape)=}"
        )

    img_rgb = _to_display_rgb(img_01, rgb_channels=rgb_channels)
    gt_rgb = _to_display_rgb(gt_01, rgb_channels=rgb_channels)
    rec_rgb = _to_display_rgb(rec_01, rgb_channels=rgb_channels)
    diff_rgb = (gt_rgb - rec_rgb).abs().clamp(0.0, 1.0)

    rows = min(int(img_rgb.shape[0]), int(max_rows))
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 4.0), dpi=140)
    if rows == 1:
        axes = axes[None, :]

    titles = ["img(input)", "gt", "recon(gt)", "|gt-recon|"]
    for r in range(rows):
        for c in range(cols):
            ax = axes[r][c]
            ax.axis("off")
            if r == 0:
                ax.set_title(titles[c])

        for ax, x in zip(axes[r], [img_rgb[r], gt_rgb[r], rec_rgb[r], diff_rgb[r]], strict=True):
            ax.imshow(x.permute(1, 2, 0).detach().cpu().numpy())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path.as_posix(), bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def test_v1() -> None:
    # _skip_if_import_failed()
    repo_root = _find_repo_root(Path(__file__).resolve())

    vae_cfg_path = repo_root / "scripts" / "configs" / "cloud_removal" / "vae" / "cosmos_rs.yaml"
    vae_cfg = cast(DictConfig, OmegaConf.load(vae_cfg_path))

    model_path = Path(str(vae_cfg["model_path"]))
    if not model_path.is_absolute():
        model_path = repo_root / model_path
    if not model_path.exists():
        pytest.skip(f"VAE checkpoint not found: {model_path}")

    device = _choose_device()

    vae = hydra.utils.instantiate(vae_cfg)
    if not isinstance(vae, CosmosRSVAE):
        raise TypeError(f"Expected CosmosRSVAE, got {type(vae)}")
    vae = vae.to(device=device)
    vae.eval().requires_grad_(False)

    data_root = repo_root / "data" / "Downstreams" / "CUHK-CR" / "litdata_out"
    if not data_root.exists():
        pytest.skip(f"CUHK-CR litdata not found: {data_root}")

    batch_size = int(os.getenv("CUHK_CR_VAE_PSNR_BATCH", "8"))
    max_batches = int(os.getenv("CUHK_CR_VAE_PSNR_NUM_BATCHES", "3"))

    _, dataloader = CUHK_CR_StreamingDataset_RandomKey_For_tokenizerPEFT.create_dataloader(
        input_dir=str(data_root),
        stream_ds_kwargs={
            # "name": "cr1",
            # "split": "train",
            "shuffle": False,
            "to_neg_1_1": True,
            "getitem_random": False,
            "interp_to": 512,
        },
        loader_kwargs={
            "batch_size": batch_size,
            "num_workers": 0,
            "pin_memory": device.type == "cuda",
        },
    )

    psnr_sum = 0.0
    psnr_batches = 0

    seen_batches = 0
    print(f"[CUHK-v1]: will eval {max_batches} batches, batch size {batch_size}")
    for batch in dataloader:
        if seen_batches >= max_batches:
            break

        if device.type == "cuda":
            torch.cuda.empty_cache()
        x_in = batch["gt"].to(device=device)
        x_cloudy = batch["img"].to(device=device)

        z = vae.encode(x_in)
        x_rec = vae.decode(z, input_shape=x_in.shape)

        x_01 = _to_0_1(x_in, is_neg_1_1=True)
        x_rec_01 = _to_0_1(x_rec, is_neg_1_1=True)
        x_cloudy_01 = _to_0_1(x_cloudy, is_neg_1_1=True)

        psnr_batch = peak_signal_noise_ratio(x_rec_01, x_01, data_range=1.0)
        psnr_sum += float(psnr_batch.mean().detach().cpu().item())
        psnr_batches += 1
        seen_batches += 1

    if seen_batches == 0:
        pytest.skip("No batches were read from dataloader")

    psnr_val = psnr_sum / max(psnr_batches, 1)
    assert psnr_val == psnr_val, "PSNR is NaN"
    print(f"[CUHK-CR1] VAE recon PSNR: {psnr_val:.4f} dB (avg over {seen_batches} batches, bs={batch_size})")

    # vis the last batch
    out_path = f"cuhk_vis_bs={batch_size}_px={x_01.shape[-1]}.png"
    _save_batch_recon_figure(
        out_path=Path(out_path),
        img_01=x_cloudy_01,
        gt_01=x_01,
        rec_01=x_rec_01,
        rgb_channels=None,
    )
    print(f"save path at {out_path}")


@torch.no_grad()
def test_v2() -> None:
    repo_root = _find_repo_root(Path(__file__).resolve())

    vae_cfg_path = repo_root / "scripts" / "configs" / "cloud_removal" / "vae" / "cosmos_rs.yaml"
    vae_cfg = cast(DictConfig, OmegaConf.load(vae_cfg_path))

    device = _choose_device()

    vae = hydra.utils.instantiate(vae_cfg)
    if not isinstance(vae, CosmosRSVAE):
        raise TypeError(f"Expected CosmosRSVAE, got {type(vae)}")
    vae = vae.to(device=device)
    vae.eval().requires_grad_(False)

    data_root = repo_root / "data" / "Downstreams" / "CUHK-CR" / "litdata_out"
    if not data_root.exists():
        pytest.skip(f"CUHK-CR litdata not found: {data_root}")

    batch_size = int(os.getenv("CUHK_CR_VAE_VIS_BATCH", "8"))
    rgb_channels_env = os.getenv("CUHK_CR_VAE_VIS_RGB_CHANNELS", "").strip()
    rgb_channels = _parse_int_list(rgb_channels_env) if rgb_channels_env else None

    _, dataloader = CUHK_CR_StreamingDataset.create_dataloader(
        input_dir=str(data_root),
        stream_ds_kwargs={
            "name": "cr2",
            "split": "train",
            "shuffle": False,
            "to_neg_1_1": True,
            "interp_to": None,
        },
        loader_kwargs={
            "batch_size": batch_size,
            "num_workers": 0,
            "pin_memory": device.type == "cuda",
        },
    )
    batch = next(iter(dataloader))

    x_img = batch["img"].to(device=device)
    x_gt = batch["gt"].to(device=device)

    z = vae.encode(x_gt)
    x_rec = vae.decode(z, input_shape=x_gt.shape)

    img_01 = _to_0_1(x_img, is_neg_1_1=True)
    gt_01 = _to_0_1(x_gt, is_neg_1_1=True)
    rec_01 = _to_0_1(x_rec, is_neg_1_1=True)

    out_path = f"cuhk_cr2_vae_recon_bs{batch_size}.png"
    _save_batch_recon_figure(
        out_path=Path(out_path),
        img_01=img_01,
        gt_01=gt_01,
        rec_01=rec_01,
        rgb_channels=rgb_channels,
    )
    print(f"save path at {out_path}")


if __name__ == "__main__":
    test_v1()
    # test_v2()

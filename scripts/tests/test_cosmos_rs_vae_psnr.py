from __future__ import annotations

import os
import csv
from pathlib import Path

import pytest
import torch
from torch import Tensor

from src.stage2.cloud_removal.diffusion.vae import CosmosRSVAE


def _to_0_1(x: Tensor, *, assume_neg_1_1: bool) -> Tensor:
    if assume_neg_1_1:
        return (x + 1.0) / 2.0
    return x


def _psnr(x: Tensor, y: Tensor, *, data_range: float = 1.0, eps: float = 1e-12) -> Tensor:
    mse = torch.mean((x - y) ** 2)
    mse = torch.clamp(mse, min=eps)
    return 20.0 * torch.log10(torch.tensor(data_range, device=x.device, dtype=x.dtype)) - 10.0 * torch.log10(mse)


def _has_sen12_meta_csv(data_root: Path, *, split: str) -> bool:
    return (data_root / "litdata_meta" / f"{split}.csv").exists()


def _iter_s2_paths_from_meta(data_root: Path, *, split: str) -> list[Path]:
    meta_path = data_root / "litdata_meta" / f"{split}.csv"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    paths: list[Path] = []
    with meta_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "s2_path" not in (reader.fieldnames or []):
            raise ValueError(f"Missing 's2_path' column in {meta_path}")
        for row in reader:
            s2_rel = row.get("s2_path")
            if not s2_rel:
                continue
            paths.append(data_root / s2_rel)
    return paths


def _to_chw(x: Tensor) -> Tensor:
    if x.ndim != 3:
        raise ValueError(f"Expected 3D tensor for S2 tif, got shape={tuple(x.shape)}")
    if x.shape[0] in (13, 2):
        return x
    if x.shape[-1] in (13, 2):
        return x.permute(2, 0, 1)
    if x.shape[1] in (13, 2):
        return x.permute(1, 2, 0)
    return x


def _normalize_s2_default(x: Tensor, *, to_neg_1_1: bool) -> Tensor:
    x = torch.nan_to_num(x.float())
    x = torch.clamp(x, 0.0, 10000.0) / 10000.0
    if to_neg_1_1:
        x = x * 2.0 - 1.0
    return x


def test_sen12_s2_clean_vae_recon_psnr() -> None:
    """
    这个测试用于“测一下当前 s2 clean 输入到 VAE 重建的 PSNR”。
    建议运行：pytest -s scripts/tests/test_cosmos_rs_vae_psnr.py
    """

    data_root = Path(os.getenv("SEN12_CR_ROOT", "data/SEN12MS-CR"))
    split = os.getenv("SEN12_CR_SPLIT", "val")
    if not _has_sen12_meta_csv(data_root, split=split):
        pytest.skip(f"SEN12MS-CR meta csv not found under {data_root} for split={split}.")

    model_path = Path(
        os.getenv(
            "COSMOS_RS_VAE_PATH",
            "runs/stage1_cosmos_nested/2025-12-21_02-01-17_cosmos_f8c16p1_litdata_one_loader_irepa-spatial-norm_noisy_latent_aug/ema/tokenizer/model.safetensors",
        )
    )
    if not model_path.exists():
        pytest.skip(f"Cosmos_RS VAE weights not found: {model_path}")

    num_samples = int(os.getenv("PSNR_NUM_SAMPLES", "4"))
    to_neg_1_1 = os.getenv("SEN12_TO_NEG_1_1", "1") not in {"0", "false", "False"}

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = "bf16" if device.type == "cuda" else "fp32"

    s2_paths = _iter_s2_paths_from_meta(data_root, split=split)
    if not s2_paths:
        pytest.skip(f"No S2 paths found in meta csv for split={split}.")

    try:
        import tifffile as tiff
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("tifffile is required to load SEN12MS-CR S2 tifs") from exc

    x0 = torch.from_numpy(tiff.imread(s2_paths[0]))
    x0 = _normalize_s2_default(_to_chw(x0), to_neg_1_1=to_neg_1_1)
    in_channels = int(x0.shape[0])

    vae = CosmosRSVAE(
        model_path=model_path.as_posix(),
        dtype=dtype,
    ).to(device=device)
    vae.eval().requires_grad_(False)

    psnrs: list[Tensor] = []
    for tif_path in s2_paths[: min(num_samples, len(s2_paths))]:
        x = torch.from_numpy(tiff.imread(tif_path))
        x = _normalize_s2_default(_to_chw(x), to_neg_1_1=to_neg_1_1)
        x = x.unsqueeze(0).to(device=device)
        z = vae.encode(x)
        x_rec = vae.decode(z, input_shape=x.shape)

        x_01 = _to_0_1(x, assume_neg_1_1=to_neg_1_1)
        x_rec_01 = _to_0_1(x_rec, assume_neg_1_1=to_neg_1_1)
        psnrs.append(_psnr(x_01, x_rec_01, data_range=1.0).detach().cpu())

    psnr_mean = torch.stack(psnrs).mean()
    print(
        f"[VAE PSNR] split={split} n={len(psnrs)} c={in_channels} to_neg_1_1={to_neg_1_1} psnr={psnr_mean.item():.4f}"
    )
    assert torch.isfinite(psnr_mean)

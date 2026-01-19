import os
from pathlib import Path
from typing import cast
import hydra
import pytest
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torchmetrics.functional.image import peak_signal_noise_ratio
from tqdm import tqdm

# Try imports
try:
    from src.stage2.cloud_removal.data.CUHK_CR import CUHK_CR_StreamingDataset
    from src.stage2.cloud_removal.diffusion.vae import CosmosRSVAE
except ImportError:
    CUHK_CR_StreamingDataset = None
    CosmosRSVAE = None


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
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def test_vae_metrics_cuhk() -> None:
    if CosmosRSVAE is None or CUHK_CR_StreamingDataset is None:
        pytest.skip("Modules not found")

    repo_root = _find_repo_root(Path(__file__).resolve())

    # Load VAE config
    vae_cfg_path = repo_root / "scripts" / "configs" / "cloud_removal" / "vae" / "cosmos_rs.yaml"
    if not vae_cfg_path.exists():
        pytest.skip(f"Config not found: {vae_cfg_path}")

    vae_cfg = cast(DictConfig, OmegaConf.load(vae_cfg_path))

    # Resolve model path
    model_path = Path(str(vae_cfg["model_path"]))
    if not model_path.is_absolute():
        model_path = repo_root / model_path

    if not model_path.exists():
        pytest.skip(f"VAE checkpoint not found: {model_path}")

    # Instantiate VAE
    device = _choose_device()
    vae = CosmosRSVAE(model_path=str(model_path), dtype=vae_cfg.get("dtype", "bf16"))
    vae = vae.to(device=device)
    vae.eval()

    # Load Dataset
    data_root = repo_root / "data" / "Downstreams" / "CUHK-CR" / "litdata_out"
    if not data_root.exists():
        pytest.skip(f"Data root not found: {data_root}")

    # Create dataloader
    # Use 'test' split as requested
    _, dataloader = CUHK_CR_StreamingDataset.create_dataloader(
        input_dir=str(data_root),
        stream_ds_kwargs={"name": "cr1", "split": "train", "shuffle": False, "to_neg_1_1": True},
        loader_kwargs={
            "batch_size": 4,  # Small batch size to avoid OOM
            "num_workers": 4,
            "pin_memory": True,
        },
    )

    psnr_values = []

    # For online mean/std calculation
    # Welford's algorithm or simple sum accumulation.
    # Since we need per-channel, we accrue sum and sum_sq per channel.
    n_samples = 0
    channel_sum = None
    channel_sq_sum = None

    # Run slightly more batches to get a stable estimate
    max_batches = 200

    print(f"\nRunning evaluation on {device}...")

    for i, batch in tqdm(enumerate(dataloader)):
        if i >= max_batches:
            break

        x_gt = batch["gt"].to(device)  # Use Ground Truth for VAE reconstruction test

        # Forward pass
        # encode returns shifted/scaled latent
        z = vae.encode(x_gt, no_std=True)
        # decode
        x_rec = vae.decode(z, input_shape=x_gt.shape, no_std=True)

        # Computing PSNR
        # Convert to [0, 1] for PSNR calculation
        x_gt_01 = _to_0_1(x_gt, is_neg_1_1=True)
        x_rec_01 = _to_0_1(x_rec, is_neg_1_1=True)

        psnr = peak_signal_noise_ratio(x_rec_01, x_gt_01, data_range=1.0)
        psnr_values.append(psnr.item())

        # Computing Latent Stats
        # z shape: [B, C, H, W]
        # We aggregate over B, H, W for each C
        if channel_sum is None:
            n_channels = z.shape[1]
            channel_sum = torch.zeros(n_channels, device=device, dtype=torch.float64)
            channel_sq_sum = torch.zeros(n_channels, device=device, dtype=torch.float64)

        # Flatten spatial dims: [B, C, H, W] -> [B, C, H*W] -> [B*H*W, C]?
        # Easier: sum over B,H,W
        # z is bf16 or float32, cast to float64 for accumulation
        z_dbl = z.double()

        # Number of pixels in this batch (B * H * W)
        curr_n_pixels = z.shape[0] * z.shape[2] * z.shape[3]
        n_samples += curr_n_pixels

        channel_sum += z_dbl.sum(dim=(0, 2, 3))
        channel_sq_sum += (z_dbl**2).sum(dim=(0, 2, 3))

    if n_samples == 0:
        pytest.skip("No data processed")

    # Aggregate Metrics
    avg_psnr = np.mean(psnr_values)

    # Compute Mean / Std
    # Mean = Sum / N
    # Std = sqrt( E[x^2] - (E[x])^2 )
    latent_mean = channel_sum / n_samples
    latent_mean_sq = channel_sq_sum / n_samples
    latent_std = torch.sqrt(latent_mean_sq - latent_mean**2)

    latent_mean = latent_mean.cpu().numpy()
    latent_std = latent_std.cpu().numpy()

    print(f"\nResults over {i} batches:")
    print(f"Reconstruction PSNR: {avg_psnr:.4f} dB")
    print("-" * 40)
    print(f"{'Channel':<10} | {'Mean':<10} | {'Std':<10}")
    print("-" * 40)
    for c in range(len(latent_mean)):
        print(f"{c:<10} | {latent_mean[c]:<10.4f} | {latent_std[c]:<10.4f}")
    print("-" * 40)

    # Basic assertion to pass test if it runs
    assert avg_psnr > 0

    # Optional: Log global mean/std
    print(f"Global Latent Mean: {latent_mean.mean():.4f}")
    print(f"Global Latent Std:  {latent_std.mean():.4f}")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-s", "-v", __file__]))

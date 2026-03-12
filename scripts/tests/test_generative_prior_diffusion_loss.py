from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.stage1.self_supervised.generative_prior_loss.diffusion_loss import DiffusionLoss


class _DummyModel(nn.Module):
    def forward(self, z_t: torch.Tensor, t: torch.Tensor, **model_kwargs: object) -> torch.Tensor:
        _ = t, model_kwargs
        return z_t


def test_diffusion_loss_forward_scalar() -> None:
    model = _DummyModel()
    loss_fn = DiffusionLoss(
        model=model,
        lambda0=5.0,
        end_logsnr=-15.0,
        schedule="linear_logsnr_vp",
    )
    latent_clean = torch.randn(4, 8, 16, 16)
    loss = loss_fn(latent_clean)
    assert loss.ndim == 0
    assert torch.isfinite(loss).item()


def test_diffusion_loss_forward_return_terms() -> None:
    model = _DummyModel()
    loss_fn = DiffusionLoss(model=model)
    latent_clean = torch.randn(3, 4, 8, 8)
    terms = loss_fn(latent_clean, reduction="none", return_terms=True)
    assert "loss" in terms
    assert "pred" in terms
    assert "target" in terms
    assert "t" in terms
    assert "elbo_coef" in terms
    assert "z_t" in terms
    assert terms["loss"].shape == (latent_clean.shape[0],)
    assert torch.all(torch.isfinite(terms["elbo_coef"])).item()
    assert torch.all(terms["elbo_coef"] > 0).item()


def test_diffusion_loss_boundary_noise_behavior() -> None:
    model = _DummyModel()
    loss_fn = DiffusionLoss(model=model, lambda0=5.0, end_logsnr=-15.0)
    latent_clean = torch.randn(2, 4, 8, 8)
    near_zero_t = torch.full((latent_clean.shape[0],), 1e-4)
    near_one_t = torch.full((latent_clean.shape[0],), 1 - 1e-4)

    terms_low_noise = loss_fn(latent_clean, t_forced=near_zero_t, reduction="none", return_terms=True)
    terms_high_noise = loss_fn(latent_clean, t_forced=near_one_t, reduction="none", return_terms=True)

    low_noise_dist = (terms_low_noise["z_t"] - latent_clean).abs().mean()
    high_noise_dist = (terms_high_noise["z_t"] - latent_clean).abs().mean()
    assert high_noise_dist > low_noise_dist

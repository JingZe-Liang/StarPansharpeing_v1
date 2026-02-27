from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.stage1.self_supervised.generative_prior_loss.diffusion_loss import DiffusionLoss


class _DummyModel(nn.Module):
    def forward(self, x_t: torch.Tensor, t: torch.Tensor, **model_kwargs: object) -> torch.Tensor:
        _ = t, model_kwargs
        return torch.zeros_like(x_t)


def test_diffusion_loss_forward_scalar() -> None:
    model = _DummyModel()
    loss_fn = DiffusionLoss(
        model=model,
        model_type="x1",
        path_type="linear",
        loss_type="none",
    )
    latent = torch.randn(4, 8, 16, 16)
    loss = loss_fn(latent)
    assert loss.ndim == 0
    assert torch.isfinite(loss).item()


def test_diffusion_loss_forward_return_terms() -> None:
    model = _DummyModel()
    loss_fn = DiffusionLoss(
        model=model,
        model_type="velocity",
        path_type="linear",
        loss_type="none",
    )
    latent = torch.randn(3, 4, 8, 8)
    terms = loss_fn(latent, reduction="none", return_terms=True)
    assert "loss" in terms
    assert "pred" in terms
    assert terms["loss"].shape == (latent.shape[0],)

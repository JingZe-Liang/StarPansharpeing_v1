from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.stage1.self_supervised.generative_prior_loss.diffusion_loss import DiffusionLoss


class _DummyModel(nn.Module):
    def forward(self, z_t: torch.Tensor, t: torch.Tensor, **model_kwargs: object) -> torch.Tensor:
        _ = t, model_kwargs
        return torch.zeros_like(z_t)


def test_diffusion_loss_forward_with_runtime_model() -> None:
    loss_fn = DiffusionLoss(model=None)
    latent_clean = torch.randn(2, 8, 16, 16)
    loss = loss_fn(latent_clean, model=_DummyModel())
    assert loss.ndim == 0
    assert torch.isfinite(loss).item()


def test_diffusion_loss_forward_without_model_raises() -> None:
    loss_fn = DiffusionLoss(model=None)
    latent_clean = torch.randn(2, 8, 16, 16)
    with pytest.raises(ValueError, match="requires a model"):
        loss_fn(latent_clean)

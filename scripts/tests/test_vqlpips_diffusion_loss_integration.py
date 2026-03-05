from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[2]))

import src.stage1.utilities.losses.gan_loss.loss as gan_loss_module
from src.stage1.utilities.losses.gan_loss.loss import VQLPIPSWithDiscriminator


class _DummyDiffusionLoss(nn.Module):
    def __init__(self, model=None, **kwargs: Any) -> None:
        super().__init__()
        _ = model, kwargs

    def forward(
        self,
        latent: torch.Tensor,
        model_kwargs: dict[str, Any] | None = None,
        *,
        model: nn.Module | None = None,
        reduction: str = "mean",
        t_forced: torch.Tensor | None = None,
        x0_forced: torch.Tensor | None = None,
        return_terms: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        _ = latent, model_kwargs, model, reduction, t_forced, x0_forced
        if return_terms:
            return {"loss": torch.tensor(2.0)}
        return torch.tensor(2.0)


class _DummyAcceleratorState:
    mixed_precision = "no"


class _DummyPartialState:
    def __init__(self) -> None:
        self.device = torch.device("cpu")


@pytest.fixture(autouse=True)
def _disable_autocast(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(gan_loss_module.torch, "autocast", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(gan_loss_module, "AcceleratorState", _DummyAcceleratorState)
    monkeypatch.setattr(gan_loss_module, "PartialState", _DummyPartialState)


def test_vqlpips_diffusion_loss_included_in_gen_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gan_loss_module, "DiffusionLoss", _DummyDiffusionLoss)

    loss_fn = VQLPIPSWithDiscriminator(
        disc_network_type="none",
        disc_weight=0.0,
        perceptual_weight=0.0,
        quantizer_type=None,
        ssim_weight=0.0,
        diffusion_loss_weight=3.0,
        diffusion_loss_options={
            "model_type": "x1",
            "path_type": "linear",
            "loss_type": "none",
        },
    )

    inputs = torch.zeros(2, 3, 8, 8)
    reconstructions = torch.zeros(2, 3, 8, 8)
    latent = torch.zeros(2, 8, 2, 2)
    dummy_denoiser = nn.Conv2d(8, 8, kernel_size=1)

    losses, logs = loss_fn(
        inputs=inputs,
        reconstructions=reconstructions,
        latent=latent,
        diffusion_model=dummy_denoiser,
        optimizer_idx=0,
        global_step=0,
    )

    assert "gen_loss" in losses
    assert "diffusion_loss" in logs
    assert torch.isclose(logs["diffusion_loss"], torch.tensor(6.0))

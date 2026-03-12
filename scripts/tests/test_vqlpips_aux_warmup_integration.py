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


class _DummyREPALoss(nn.Module):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        _ = kwargs
        self.repa_encoder = None

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        _ = args, kwargs
        return torch.tensor(2.0)


class _DummyVFLoss(nn.Module):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        _ = kwargs
        self.repa_encoder = None

    def forward(
        self,
        img: torch.Tensor,
        feat: torch.Tensor | list[torch.Tensor] | None,
        nll_loss: torch.Tensor,
        enc_last_layer: nn.Parameter | None,
    ) -> torch.Tensor:
        _ = img, feat, nll_loss, enc_last_layer
        return torch.tensor(3.0)


class _DummySIGReg:
    def __init__(self, **kwargs: Any) -> None:
        _ = kwargs

    def __call__(self, latent_1d: torch.Tensor) -> torch.Tensor:
        _ = latent_1d
        return torch.tensor(5.0)


class _DummyLatentSparsityLoss(nn.Module):
    def __init__(self, dim_z: int, **kwargs: Any) -> None:
        super().__init__()
        _ = kwargs
        self.dim_z = dim_z

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _ = latent
        return torch.tensor(7.0), {}


def _dummy_lcr_loss(latent: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    _ = latent, kwargs
    return torch.tensor(0.0), torch.tensor(2.0)


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


def test_vqlpips_repa_sem_wait_for_own_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gan_loss_module, "REPALoss", _DummyREPALoss)

    loss_fn = VQLPIPSWithDiscriminator(
        disc_network_type="none",
        disc_weight=0.0,
        perceptual_weight=0.0,
        quantizer_type=None,
        ssim_weight=0.0,
        repa_loss_weight=3.0,
        repa_start_for_g=10,
        sem_distill_weight=5.0,
        sem_distill_start_for_g=10,
    )

    losses_warmup, logs_warmup = loss_fn(
        inputs=torch.zeros(2, 3, 8, 8),
        reconstructions=torch.zeros(2, 3, 8, 8),
        optimizer_idx=0,
        global_step=5,
    )

    assert "gen_loss" in losses_warmup
    assert torch.isclose(logs_warmup["repa_loss"], torch.tensor(0.0))
    assert torch.isclose(logs_warmup["sem_dist_loss"], torch.tensor(0.0))

    _, logs_after = loss_fn(
        inputs=torch.zeros(2, 3, 8, 8),
        reconstructions=torch.zeros(2, 3, 8, 8),
        tokenizer_feat=torch.zeros(2, 4, 2, 2),
        tokenizer_feat2=torch.zeros(2, 4, 2, 2),
        optimizer_idx=0,
        global_step=10,
    )

    assert torch.isclose(logs_after["repa_loss"], torch.tensor(6.0))
    assert torch.isclose(logs_after["sem_dist_loss"], torch.tensor(10.0))


def test_vqlpips_vf_wait_for_own_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gan_loss_module, "VFLoss", _DummyVFLoss)

    loss_fn = VQLPIPSWithDiscriminator(
        disc_network_type="none",
        disc_weight=0.0,
        perceptual_weight=0.0,
        quantizer_type=None,
        ssim_weight=0.0,
        vf_loss_weight=4.0,
        vf_start_for_g=10,
    )

    losses_warmup, logs_warmup = loss_fn(
        inputs=torch.zeros(2, 3, 8, 8),
        reconstructions=torch.zeros(2, 3, 8, 8),
        optimizer_idx=0,
        global_step=5,
    )

    assert "gen_loss" in losses_warmup
    assert torch.isclose(logs_warmup["vf_loss"], torch.tensor(0.0))

    _, logs_after = loss_fn(
        inputs=torch.zeros(2, 3, 8, 8),
        reconstructions=torch.zeros(2, 3, 8, 8),
        tokenizer_feat=torch.zeros(2, 4, 2, 2),
        optimizer_idx=0,
        global_step=10,
    )

    assert torch.isclose(logs_after["vf_loss"], torch.tensor(12.0))


def test_vqlpips_other_reg_wait_for_shared_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gan_loss_module, "SIGReg", _DummySIGReg)
    monkeypatch.setattr(gan_loss_module, "LatentSparsityLoss", _DummyLatentSparsityLoss)
    monkeypatch.setattr(gan_loss_module, "lcr_loss", _dummy_lcr_loss)

    loss_fn = VQLPIPSWithDiscriminator(
        disc_network_type="none",
        disc_weight=0.0,
        perceptual_weight=0.0,
        quantizer_type=None,
        ssim_weight=0.0,
        other_reg_start_for_g=10,
        lcr_loss_weight=2.0,
        latent_sparsity_weight=3.0,
        sigreg_weight=4.0,
    )
    loss_fn._calculate_adaptive_weight = lambda nll_loss, g_loss, last_layer=None: torch.tensor(1.0)  # type: ignore[method-assign]

    _, logs_warmup = loss_fn(
        inputs=torch.zeros(2, 3, 8, 8),
        reconstructions=torch.zeros(2, 3, 8, 8),
        optimizer_idx=0,
        global_step=5,
    )

    assert torch.isclose(logs_warmup["lcr_loss"], torch.tensor(0.0))
    assert torch.isclose(logs_warmup["latent_sparsity_loss"], torch.tensor(0.0))
    assert torch.isclose(logs_warmup["sigreg_loss"], torch.tensor(0.0))

    _, logs_after = loss_fn(
        inputs=torch.zeros(2, 3, 8, 8),
        reconstructions=torch.zeros(2, 3, 8, 8),
        latent=torch.zeros(2, 8, 2, 2),
        optimizer_idx=0,
        global_step=10,
    )

    assert torch.isclose(logs_after["lcr_loss"], torch.tensor(4.0))
    assert torch.isclose(logs_after["latent_sparsity_loss"], torch.tensor(7.0))
    assert torch.isclose(logs_after["sigreg_loss"], torch.tensor(20.0))

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
from src.stage1.utilities.losses.repa.repa_feature_loss import PhiSMultipleTeacherDistillLoss


class _DummyPhiSMultipleTeacherDistillLoss(nn.Module):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        _ = kwargs

    def move_teachers_to(self, device: torch.device, dtype: torch.dtype | None = None) -> None:
        _ = device, dtype

    def reset_phi_stats(self) -> None:
        return None

    def update_phi_stats_from_image(self, img: torch.Tensor) -> None:
        _ = img
        return None

    def finalize_phi_from_stats(self, distributed: bool = True) -> None:
        _ = distributed
        return None

    def load_phi_from_cache(self, cache_path: str, broadcast: bool = True) -> None:
        _ = cache_path, broadcast
        return None

    def save_phi_to_cache(self, cache_path: str) -> None:
        _ = cache_path
        return None

    def forward(self, img: torch.Tensor, student_feature: dict[str, torch.Tensor | list[torch.Tensor]]) -> torch.Tensor:
        _ = img, student_feature
        return torch.tensor(2.0)


class _DummyTeacherAdapter:
    def __init__(self) -> None:
        self.encoder = nn.Conv2d(3, 4, kernel_size=1)
        self.processor = None

    def forward_features(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        *,
        get_interm_feats: bool,
        detach: bool,
    ) -> list[torch.Tensor]:
        _ = x, get_interm_feats, detach
        return [torch.zeros(1, 4, 2, 2)]

    def encode(
        self,
        img: torch.Tensor,
        *,
        get_interm_feats: bool,
        use_linstretch: bool,
        detach: bool,
        repa_fixed_bs: int | None,
    ) -> list[torch.Tensor]:
        _ = get_interm_feats, use_linstretch, detach, repa_fixed_bs
        return [torch.zeros(img.shape[0], 4, 2, 2, device=img.device, dtype=img.dtype)]


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


def test_vqlpips_phis_loss_included_in_gen_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gan_loss_module, "PhiSMultipleTeacherDistillLoss", _DummyPhiSMultipleTeacherDistillLoss)

    loss_fn = VQLPIPSWithDiscriminator(
        disc_network_type="none",
        disc_weight=0.0,
        perceptual_weight=0.0,
        quantizer_type=None,
        ssim_weight=0.0,
        phis_loss_weight=3.0,
        phis_loss_options={"teacher_configs": {"dino": {}}, "teacher_dims": {"dino": [4]}},
    )

    inputs = torch.zeros(2, 3, 8, 8)
    reconstructions = torch.zeros(2, 3, 8, 8)
    losses, logs = loss_fn(
        inputs=inputs,
        reconstructions=reconstructions,
        optimizer_idx=0,
        global_step=0,
        phis_student_feature={"dino": [torch.zeros(2, 4, 2, 2)]},
    )

    assert "gen_loss" in losses
    assert "phis_loss" in logs
    assert torch.isclose(logs["phis_loss"], torch.tensor(6.0))


def test_vqlpips_phis_mutex_with_repa_vf_gram(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gan_loss_module, "PhiSMultipleTeacherDistillLoss", _DummyPhiSMultipleTeacherDistillLoss)

    with pytest.raises(AssertionError):
        VQLPIPSWithDiscriminator(
            disc_network_type="none",
            disc_weight=0.0,
            perceptual_weight=0.0,
            quantizer_type=None,
            ssim_weight=0.0,
            phis_loss_weight=1.0,
            phis_loss_options={"teacher_configs": {"dino": {}}, "teacher_dims": {"dino": [4]}},
            repa_loss_weight=1.0,
        )

    with pytest.raises(AssertionError):
        VQLPIPSWithDiscriminator(
            disc_network_type="none",
            disc_weight=0.0,
            perceptual_weight=0.0,
            quantizer_type=None,
            ssim_weight=0.0,
            phis_loss_weight=1.0,
            phis_loss_options={"teacher_configs": {"dino": {}}, "teacher_dims": {"dino": [4]}},
            vf_loss_weight=1.0,
        )

    with pytest.raises(AssertionError):
        VQLPIPSWithDiscriminator(
            disc_network_type="none",
            disc_weight=0.0,
            perceptual_weight=0.0,
            quantizer_type=None,
            ssim_weight=0.0,
            phis_loss_weight=1.0,
            phis_loss_options={"teacher_configs": {"dino": {}}, "teacher_dims": {"dino": [4]}},
            gram_loss_weight=1.0,
        )


def test_vqlpips_phis_requires_student_feature_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gan_loss_module, "PhiSMultipleTeacherDistillLoss", _DummyPhiSMultipleTeacherDistillLoss)

    loss_fn = VQLPIPSWithDiscriminator(
        disc_network_type="none",
        disc_weight=0.0,
        perceptual_weight=0.0,
        quantizer_type=None,
        ssim_weight=0.0,
        phis_loss_weight=1.0,
        phis_loss_options={"teacher_configs": {"dino": {}}, "teacher_dims": {"dino": [4]}},
    )

    with pytest.raises(AssertionError):
        loss_fn(
            inputs=torch.zeros(2, 3, 8, 8),
            reconstructions=torch.zeros(2, 3, 8, 8),
            optimizer_idx=0,
            global_step=0,
        )


def test_vqlpips_phis_loss_waits_for_its_own_start_step(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gan_loss_module, "PhiSMultipleTeacherDistillLoss", _DummyPhiSMultipleTeacherDistillLoss)

    loss_fn = VQLPIPSWithDiscriminator(
        disc_network_type="none",
        disc_weight=0.0,
        phis_start_for_g=10,
        perceptual_weight=0.0,
        quantizer_type=None,
        ssim_weight=0.0,
        phis_loss_weight=3.0,
        phis_loss_options={"teacher_configs": {"dino": {}}, "teacher_dims": {"dino": [4]}},
    )

    losses, logs = loss_fn(
        inputs=torch.zeros(2, 3, 8, 8),
        reconstructions=torch.zeros(2, 3, 8, 8),
        optimizer_idx=0,
        global_step=5,
    )

    assert "gen_loss" in losses
    assert "phis_loss" in logs
    assert torch.isclose(logs["phis_loss"], torch.tensor(0.0))


def test_phis_teacher_not_in_state_dict_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.stage1.utilities.losses.repa.repa_feature_loss as repa_module

    monkeypatch.setattr(repa_module, "build_teacher_adapter", lambda **kwargs: _DummyTeacherAdapter())

    loss_fn = PhiSMultipleTeacherDistillLoss(
        teacher_configs={"dino": {"repa_model_type": "dinov3", "repa_model_name": "dinov3_vitl16"}},
        teacher_dims={"dino": [4]},
    )

    keys = list(loss_fn.state_dict().keys())
    assert any(key.startswith("phi_loss.") for key in keys)
    assert not any(key.startswith("teacher_encoders.") for key in keys)
    assert not any("teacher_adapters" in key for key in keys)

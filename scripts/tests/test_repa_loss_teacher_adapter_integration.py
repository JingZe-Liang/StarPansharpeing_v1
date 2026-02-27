from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.stage1.utilities.losses.repa.repa_feature_loss import PhiSMultipleTeacherDistillLoss


class _DummyTeacherAdapter:
    def __init__(self, dims: list[int]) -> None:
        self.dims = dims
        self.encoder = nn.Identity()
        self.processor = None

    def forward_features(
        self, x: torch.Tensor | dict[str, torch.Tensor], *, get_interm_feats: bool, detach: bool
    ) -> list[torch.Tensor]:
        _ = x, get_interm_feats, detach
        raise NotImplementedError

    def encode(
        self,
        img: torch.Tensor,
        *,
        get_interm_feats: bool,
        use_linstretch: bool,
        detach: bool,
        repa_fixed_bs: int | None,
    ) -> list[torch.Tensor]:
        _ = use_linstretch, repa_fixed_bs
        b = img.shape[0]
        out: list[torch.Tensor] = []
        for i, dim in enumerate(self.dims):
            feat = torch.ones(b, dim, 2, 2, device=img.device, dtype=img.dtype) * float(i + 1)
            out.append(feat.detach() if detach else feat)
        if get_interm_feats:
            return out
        return [out[-1]]


def test_phis_multiple_teacher_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    teacher_dims = {
        "dino": [4, 4],
        "siglip": [6],
        "pe": [8, 8, 8],
    }

    def _fake_build_teacher_adapter(**kwargs: Any) -> _DummyTeacherAdapter:
        model_type = kwargs["repa_model_type"]
        if model_type == "dinov3":
            return _DummyTeacherAdapter(teacher_dims["dino"])
        if model_type == "siglip2":
            return _DummyTeacherAdapter(teacher_dims["siglip"])
        if model_type == "pe":
            return _DummyTeacherAdapter(teacher_dims["pe"])
        raise ValueError(model_type)

    import src.stage1.utilities.losses.repa.repa_feature_loss as repa_module

    monkeypatch.setattr(repa_module, "build_teacher_adapter", _fake_build_teacher_adapter)

    loss_mod = PhiSMultipleTeacherDistillLoss(
        teacher_configs={
            "dino": {"repa_model_type": "dinov3", "repa_model_name": "dinov3_vitl16"},
            "siglip": {"repa_model_type": "siglip2", "repa_model_name": "google/siglip2-so400m-patch16-naflex"},
            "pe": {"repa_model_type": "pe", "repa_model_name": "PE-Core-B16-224"},
        },
        teacher_dims=teacher_dims,
    )

    img = torch.randn(2, 5, 16, 16)
    loss_mod.reset_phi_stats()
    loss_mod.update_phi_stats_from_image(img)
    loss_mod.finalize_phi_from_stats(distributed=False)

    student_feature = {
        "dino": [torch.zeros(2, 4, 2, 2), torch.zeros(2, 4, 2, 2)],
        "siglip": [torch.zeros(2, 6, 2, 2)],
        "pe": [torch.zeros(2, 8, 2, 2), torch.zeros(2, 8, 2, 2), torch.zeros(2, 8, 2, 2)],
    }

    loss = loss_mod(img, student_feature)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_phis_multiple_teacher_key_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    teacher_dims = {"dino": [4]}

    def _fake_build_teacher_adapter(**kwargs: Any) -> _DummyTeacherAdapter:
        _ = kwargs
        return _DummyTeacherAdapter([4])

    import src.stage1.utilities.losses.repa.repa_feature_loss as repa_module

    monkeypatch.setattr(repa_module, "build_teacher_adapter", _fake_build_teacher_adapter)

    loss_mod = PhiSMultipleTeacherDistillLoss(
        teacher_configs={"dino": {"repa_model_type": "dinov3", "repa_model_name": "dinov3_vitl16"}},
        teacher_dims=teacher_dims,
    )

    img = torch.randn(1, 5, 16, 16)
    loss_mod.reset_phi_stats()
    loss_mod.update_phi_stats_from_image(img)
    loss_mod.finalize_phi_from_stats(distributed=False)

    with pytest.raises(KeyError):
        loss_mod(img, {"siglip": [torch.zeros(1, 4, 2, 2)]})

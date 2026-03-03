from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import src.stage1.utilities.losses.repa.repa_feature_loss as repa_feature_loss


class _DummyMultiLayerPhiLoss(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def update_phi_stats(self, teacher_feats: dict[str, list[torch.Tensor]]) -> None:
        return None

    def reset_phi_stats(self) -> None:
        return None

    def finalize_phi_from_stats(self, *, distributed: bool = True) -> None:
        return None

    def load_phi_from_cache(self, cache_path: str | Path, *, broadcast: bool = True) -> None:
        return None

    def save_phi_to_cache(self, cache_path: str | Path) -> None:
        return None

    def forward(
        self,
        student_feats: dict[str, list[torch.Tensor]],
        teacher_feats: dict[str, list[torch.Tensor]],
    ) -> torch.Tensor:
        return torch.tensor(0.0)


def _dummy_build_teacher_adapter(**kwargs):
    encoder = nn.Identity()
    return SimpleNamespace(
        encoder=encoder,
        processor=None,
        encode=lambda *args, **inner_kwargs: [torch.zeros(1, 4, 2, 2)],
    )


def test_phis_auto_load_required_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(repa_feature_loss, "build_teacher_adapter", _dummy_build_teacher_adapter)
    monkeypatch.setattr(repa_feature_loss, "MultiLayersPhiSDistillLoss", _DummyMultiLayerPhiLoss)

    missing_path = tmp_path / "missing_phi_buffer.pt"
    with pytest.raises(FileNotFoundError, match="PhiS cache path does not exist"):
        repa_feature_loss.PhiSMultipleTeacherDistillLoss(
            teacher_configs={"dinov3": {"repa_model_type": "dinov3", "repa_model_name": "dinov3_vitl16"}},
            teacher_dims={"dinov3": [4]},
            phi_cache_path=missing_path,
            phi_cache_required=True,
            phi_cache_load_on_init=True,
        )

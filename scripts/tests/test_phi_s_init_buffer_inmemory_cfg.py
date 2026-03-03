from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.stage1.utilities.losses.distill import phi_s_init_buffer as init_buf


class _DummyPhiLoss:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def to(self, device: torch.device) -> "_DummyPhiLoss":
        return self

    def move_teachers_to(self, device: torch.device) -> None:
        return None

    def reset_phi_stats(self) -> None:
        return None

    def update_phi_stats_from_image(self, img: torch.Tensor) -> None:
        return None

    def finalize_phi_from_stats(self, *, distributed: bool = True) -> None:
        return None

    def save_phi_to_cache(self, cache_path: str | Path) -> None:
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"ok": True}, path)


def test_phi_s_init_buffer_inmemory_cfg(tmp_path: Path, monkeypatch) -> None:
    output_path = tmp_path / "phi_buffer.pt"
    cfg = OmegaConf.create(
        {
            "loader": {"_target_": "ignored.target"},
            "phis_loss_options": {
                "teacher_configs": {"dinov3": {"repa_model_type": "dinov3", "repa_model_name": "dinov3_vitl16"}},
                "teacher_dims": {"dinov3": [4]},
            },
            "runtime": {
                "seed": 2025,
                "device": "cpu",
                "max_batches": 2,
                "log_every": 1,
                "output_path": str(output_path),
            },
        }
    )
    assert cfg is not None

    monkeypatch.setattr(init_buf, "register_new_resolvers", lambda: None)
    monkeypatch.setattr(init_buf, "_default_cfg", lambda: cfg)
    monkeypatch.setattr(
        init_buf,
        "_build_loader",
        lambda _cfg: [{"img": torch.randn(2, 3, 8, 8)}, {"img": torch.randn(2, 3, 8, 8)}],
    )
    monkeypatch.setattr(init_buf, "PhiSMultipleTeacherDistillLoss", _DummyPhiLoss)

    init_buf.main()
    assert output_path.exists()

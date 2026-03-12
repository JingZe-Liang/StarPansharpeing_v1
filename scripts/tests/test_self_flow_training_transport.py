from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.utilities.transport.self_flow.src.sampling import (
    ModelType,
    PathType,
    Transport,
    WeightType,
    create_transport,
)


class DualTimeProbeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.last_t: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, **kwargs: object) -> torch.Tensor:
        _ = kwargs
        self.last_t = timesteps
        return x * self.scale


def _build_transport(model_type: ModelType = ModelType.VELOCITY) -> Transport:
    return Transport(
        model_type=model_type,
        path_type=PathType.LINEAR,
        loss_type=WeightType.NONE,
        train_eps=0.0,
        sample_eps=0.0,
        time_sample_type="uniform",
    )


def test_training_losses_self_flow_image_input_runs_and_has_expected_keys() -> None:
    transport = _build_transport()
    student = DualTimeProbeModel()
    teacher = DualTimeProbeModel()

    x1 = torch.randn(2, 3, 4, 4)
    terms = transport.training_losses_self_flow(
        student,
        teacher,
        x1,
        model_kwargs={},
        teacher_model_kwargs={},
        mask_ratio=0.25,
    )

    expected_keys = {"loss", "gen_loss", "rep_loss", "pred", "t_primary", "t_secondary", "tau_min"}
    assert expected_keys.issubset(terms.keys())
    assert torch.isfinite(terms["loss"])
    assert terms["loss"].ndim == 0
    assert terms["pred"].shape == x1.shape
    assert terms["tau"].shape == (2, 16)
    assert terms["tau_min"].shape == (2, 16)
    assert student.last_t is not None and student.last_t.shape == (2, 16)
    assert teacher.last_t is not None and teacher.last_t.shape == (2, 16)

    terms["loss"].backward()
    assert student.scale.grad is not None
    assert teacher.scale.grad is None


def test_training_losses_self_flow_token_input_uses_token_level_tau() -> None:
    transport = _build_transport()
    student = DualTimeProbeModel()
    teacher = DualTimeProbeModel()

    x1 = torch.randn(3, 7, 5)
    t_forced = torch.tensor([0.2, 0.4, 0.6], dtype=x1.dtype)
    s_forced = torch.tensor([0.8, 0.1, 0.3], dtype=x1.dtype)

    terms = transport.training_losses_self_flow(
        student,
        teacher,
        x1,
        mask_ratio=0.0,
        t_forced=t_forced,
        s_forced=s_forced,
    )

    expected_tau = t_forced.unsqueeze(1).expand(-1, x1.shape[1])
    expected_tau_min = torch.minimum(t_forced, s_forced).unsqueeze(1).expand(-1, x1.shape[1])

    torch.testing.assert_close(terms["tau"], expected_tau)
    torch.testing.assert_close(terms["tau_min"], expected_tau_min)
    assert student.last_t is not None and student.last_t.shape == (3, 7)
    assert teacher.last_t is not None and teacher.last_t.shape == (3, 7)


def test_training_losses_self_flow_non_velocity_raises() -> None:
    transport = _build_transport(model_type=ModelType.NOISE)
    student = DualTimeProbeModel()
    teacher = DualTimeProbeModel()
    x1 = torch.randn(2, 4, 8, 8)

    with pytest.raises(NotImplementedError, match="velocity"):
        _ = transport.training_losses_self_flow(student, teacher, x1)


def test_training_fm_loss_velocity_runs_and_returns_expected_fields() -> None:
    transport = _build_transport(model_type=ModelType.VELOCITY)
    model = DualTimeProbeModel()
    x1 = torch.randn(4, 3, 8, 8)
    t_forced = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=x1.dtype)

    terms = transport.training_fm_loss(
        model=model,
        x1=x1,
        model_kwargs={},
        get_pred_x_clean=True,
        t_forced=t_forced,
    )

    assert {"loss", "pred", "t", "pred_x_clean"}.issubset(terms.keys())
    assert terms["loss"].shape == (4,)
    assert terms["pred"].shape == x1.shape
    torch.testing.assert_close(terms["t"], t_forced)
    assert terms["pred_x_clean"].shape == x1.shape
    assert torch.isfinite(terms["loss"]).all()


def test_training_losses_alias_matches_training_fm_loss() -> None:
    transport = _build_transport(model_type=ModelType.VELOCITY)
    model = DualTimeProbeModel()
    x1 = torch.randn(2, 3, 4, 4)
    t_forced = torch.tensor([0.2, 0.7], dtype=x1.dtype)
    x0_forced = torch.randn_like(x1)

    terms_fm = transport.training_fm_loss(
        model=model,
        x1=x1,
        model_kwargs={},
        get_pred_x_clean=True,
        t_forced=t_forced,
        x0_forced=x0_forced,
    )
    terms_alias = transport.training_losses(
        model=model,
        x1=x1,
        model_kwargs={},
        get_pred_x_clean=True,
        t_forced=t_forced,
        x0_forced=x0_forced,
    )

    torch.testing.assert_close(terms_fm["t"], terms_alias["t"])
    torch.testing.assert_close(terms_fm["loss"], terms_alias["loss"])
    torch.testing.assert_close(terms_fm["pred"], terms_alias["pred"])
    torch.testing.assert_close(terms_fm["pred_x_clean"], terms_alias["pred_x_clean"])


@pytest.mark.parametrize("model_type", [ModelType.NOISE, ModelType.SCORE])
def test_training_fm_loss_noise_and_score_run(model_type: ModelType) -> None:
    transport = Transport(
        model_type=model_type,
        path_type=PathType.LINEAR,
        loss_type=WeightType.NONE,
        train_eps=0.0,
        sample_eps=0.0,
        time_sample_type="uniform",
    )
    model = DualTimeProbeModel()
    x1 = torch.randn(3, 5, 6, 6)
    terms = transport.training_fm_loss(model=model, x1=x1, model_kwargs={})

    assert {"loss", "pred", "t", "pred_x_clean"}.issubset(terms.keys())
    assert terms["loss"].shape == (3,)
    assert torch.isfinite(terms["loss"]).all()
    assert terms["pred"].shape == x1.shape
    assert terms["pred_x_clean"].shape == x1.shape


def test_create_transport_respects_prediction_and_loss_weight() -> None:
    tr_noise = create_transport(path_type="Linear", prediction="noise", loss_weight="likelihood")
    tr_score = create_transport(path_type="Linear", prediction="score", loss_weight="velocity")
    tr_vel = create_transport(path_type="Linear", prediction="velocity", loss_weight=None)

    assert tr_noise.model_type == ModelType.NOISE
    assert tr_noise.loss_type == WeightType.LIKELIHOOD
    assert tr_score.model_type == ModelType.SCORE
    assert tr_score.loss_type == WeightType.VELOCITY
    assert tr_vel.model_type == ModelType.VELOCITY
    assert tr_vel.loss_type == WeightType.NONE

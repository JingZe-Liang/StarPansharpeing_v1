from __future__ import annotations

import torch

from src.utilities.transport.flow_matching.transport import ModelType, PathType, Sampler, Transport, WeightType


def test_transport_training_losses_x1_runs() -> None:
    transport = Transport(
        model_type=ModelType.X1,
        path_type=PathType.LINEAR,
        loss_type=WeightType.NONE,
        train_eps=1e-3,
        sample_eps=1e-3,
        time_sample_type="uniform",
    )

    x1 = torch.randn(4, 8, 16, 16)
    conditions = torch.randn(4, 2, 16, 16)

    def model_fn(x_t: torch.Tensor, t: torch.Tensor, **model_kwargs: object) -> torch.Tensor:
        _ = t, model_kwargs
        return torch.zeros_like(x_t)

    terms = transport.training_losses(model_fn, x1, model_kwargs={"conditions": conditions}, t_forced=torch.rand(4))
    loss = terms["loss"]
    assert loss.ndim == 0
    assert torch.isfinite(loss).all()


def test_sampler_sample_ode_x1_shape() -> None:
    transport = Transport(
        model_type=ModelType.X1,
        path_type=PathType.LINEAR,
        loss_type=WeightType.NONE,
        train_eps=1e-3,
        sample_eps=1e-3,
        time_sample_type="uniform",
    )
    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(num_steps=6, sampling_time_type="uniform", clip_for_x1_pred=False)

    x0 = torch.randn(2, 3, 8, 8)

    def model_fn(x_t: torch.Tensor, t: torch.Tensor, **model_kwargs: object) -> torch.Tensor:
        _ = x_t, t, model_kwargs
        return torch.ones_like(x0)

    samples = sample_fn(x0, model_fn)
    assert samples.shape[0] >= 6
    assert samples.shape[1:] == x0.shape
    assert samples[-1].shape == x0.shape

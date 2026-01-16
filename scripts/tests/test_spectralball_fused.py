from typing import cast

import torch

from src.utilities.optim.spectral_ball_fused import SpectralBallFused, split_spectralball_oned_params


def test_split_spectralball_oned_params_by_ndim_and_ignore_patterns() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 8, bias=True), torch.nn.LayerNorm(8))
    linear0 = cast(torch.nn.Linear, model[0])
    norm1 = cast(torch.nn.LayerNorm, model[1])

    groups = split_spectralball_oned_params(model.named_parameters())
    assert any(p is linear0.weight for p in groups.spectralball["params"])
    assert any(p is linear0.bias for p in groups.oned["params"])
    assert any(p is norm1.weight for p in groups.oned["params"])
    assert any(p is norm1.bias for p in groups.oned["params"])

    groups_ignored = split_spectralball_oned_params(
        model.named_parameters(), ignored_keys_for_spectralball=("0.weight",)
    )
    assert any(p is linear0.weight for p in groups_ignored.oned["params"])
    assert len(groups_ignored.spectralball["params"]) == 0


def test_spectralball_fused_step_runs() -> None:
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8, bias=True), torch.nn.LayerNorm(8))
    linear0 = cast(torch.nn.Linear, model[0])

    optimizer = SpectralBallFused.create_spectralball_optimizer(
        model.named_parameters(),
        spectralball_params_defaults={
            "lr": 1e-3,
            "momentum_beta": 0.9,
            "weight_decay": 0.0,
            "power_iteration_steps": 1,
            "msign_steps": 1,
            "solver_tolerance_f": 1e-2,
            "solver_max_iterations": 2,
            "radius_mode": "identity",
            "retract_mode": "hard",
        },
        oned_params_defaults={"lr": 1e-3, "weight_decay": 0.0},
        oned_param_algo="adamw",
    )

    x = torch.randn(2, 4)
    before = linear0.weight.detach().clone()
    loss = model(x).sum()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    assert not torch.allclose(linear0.weight.detach(), before)


def test_spectralball_fused_all_oned_when_ignored() -> None:
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8, bias=True), torch.nn.LayerNorm(8))
    linear0 = cast(torch.nn.Linear, model[0])

    optimizer = SpectralBallFused.create_spectralball_optimizer(
        model.named_parameters(),
        ignored_keys_for_spectralball=("0.weight",),
        spectralball_params_defaults={
            "lr": 1e-3,
            "momentum_beta": 0.9,
            "weight_decay": 0.0,
            "power_iteration_steps": 1,
            "msign_steps": 1,
            "solver_tolerance_f": 1e-2,
            "solver_max_iterations": 2,
            "radius_mode": "identity",
            "retract_mode": "hard",
        },
        oned_params_defaults={"lr": 1e-3, "weight_decay": 0.0},
        oned_param_algo="adamw",
    )

    x = torch.randn(2, 4)
    before = linear0.weight.detach().clone()
    loss = model(x).sum()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    assert not torch.allclose(linear0.weight.detach(), before)

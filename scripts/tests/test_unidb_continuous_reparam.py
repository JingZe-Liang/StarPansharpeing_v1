import torch

from src.utilities.transport.UniDB.plan_continuous import (
    LinearInterpolant,
    LinearInterpolantConfig,
    UniDBContinuous,
)


def test_unidb_continuous_reparam_roundtrip_epsilon_x0() -> None:
    torch.manual_seed(0)
    sde = UniDBContinuous(lambda_square=0.01, gamma=1.0, T=64, schedule="cosine", eps=0.01)

    batch, channels, height, width = 4, 3, 8, 8
    x0 = torch.randn(batch, channels, height, width)
    mu = torch.randn_like(x0)

    t = sde.sample_time(batch, device=x0.device, dtype=x0.dtype, t_min=1e-2)
    xt, eps = sde.sample_xt(x0, mu, t)

    eps_rec = sde.epsilon_from_xt_x0(xt, x0, mu, t)
    torch.testing.assert_close(eps_rec, eps, rtol=1e-4, atol=1e-4)

    x0_rec = sde.x0_from_xt_epsilon(xt, mu, t, eps)
    torch.testing.assert_close(x0_rec, x0, rtol=1e-4, atol=1e-4)


def test_unidb_continuous_velocity_matches_two_forms() -> None:
    torch.manual_seed(0)
    sde = UniDBContinuous(lambda_square=0.01, gamma=1.0, T=64, schedule="cosine", eps=0.01)

    batch, channels, height, width = 4, 3, 8, 8
    x0 = torch.randn(batch, channels, height, width)
    mu = torch.randn_like(x0)

    t = sde.sample_time(batch, device=x0.device, dtype=x0.dtype, t_min=1e-2)
    xt, eps = sde.sample_xt(x0, mu, t)

    v_direct = sde.velocity_target(x0, mu, t, eps)
    v_ab = sde.velocity_from_epsilon(xt, mu, t, eps)
    torch.testing.assert_close(v_direct, v_ab, rtol=1e-4, atol=1e-4)


def test_linear_interpolant_velocity_matches_two_forms() -> None:
    torch.manual_seed(0)
    interpolant = LinearInterpolant(LinearInterpolantConfig(sigma_max=0.1, sigma_schedule="quadratic"))

    batch, channels, height, width = 4, 3, 8, 8
    x0 = torch.randn(batch, channels, height, width)
    x1 = torch.randn_like(x0)

    t = UniDBContinuous.sample_time(batch, device=x0.device, dtype=x0.dtype, t_min=1e-2)
    xt, eps = interpolant.sample_xt(x0, x1, t)

    v_direct = interpolant.velocity_target(x0, x1, t, eps)
    v_ab = interpolant.velocity_from_epsilon(xt, x1, t, eps)
    torch.testing.assert_close(v_direct, v_ab, rtol=1e-4, atol=1e-4)

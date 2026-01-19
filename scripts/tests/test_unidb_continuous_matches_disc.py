import torch

from src.utilities.transport.UniDB.plan_continuous import UniDBContinuous
from src.utilities.transport.UniDB.plan_disc import UniDB as UniDBDisc


def test_unidb_continuous_matches_disc_at_grid_points() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float32

    lambda_square = 0.01
    gamma = 1.0
    T = 64
    schedule = "cosine"
    eps = 0.01

    disc = UniDBDisc(lambda_square=lambda_square, gamma=gamma, T=T, schedule=schedule, eps=eps, device=device)
    cont = UniDBContinuous(lambda_square=lambda_square, gamma=gamma, T=T, schedule=schedule, eps=eps)

    torch.testing.assert_close(cont._dt.to(device=device, dtype=dtype), disc.dt.to(device=device, dtype=dtype))

    test_steps = [1, 2, 5, T // 2, T - 1, T]
    for t in test_steps:
        t01 = torch.tensor([t / T], device=device, dtype=dtype)

        m_cont = cont.m(t01, device=device, dtype=dtype)
        m_disc = disc.m(t).reshape(1)
        torch.testing.assert_close(m_cont, m_disc.to(dtype=dtype), rtol=0.0, atol=0.0)

        sigma_cont = cont.sigma(t01, device=device, dtype=dtype)
        sigma_disc = disc.f_sigma(t).reshape(1)
        torch.testing.assert_close(sigma_cont, sigma_disc.to(dtype=dtype), rtol=0.0, atol=0.0)

        x0 = torch.randn(2, 3, 4, 4, device=device, dtype=dtype)
        mu = torch.randn_like(x0)
        eps_noise = torch.randn_like(x0)
        disc.set_mu(mu)

        xt_cont = (
            _broadcast(m_cont, x0) * x0 + (1.0 - _broadcast(m_cont, x0)) * mu + _broadcast(sigma_cont, x0) * eps_noise
        )
        xt_disc = disc.f_mean(x0, t) + disc.f_sigma(t) * eps_noise
        torch.testing.assert_close(xt_cont, xt_disc.to(dtype=dtype), rtol=0.0, atol=0.0)


def _broadcast(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return t.view(-1, *([1] * (x.ndim - 1)))

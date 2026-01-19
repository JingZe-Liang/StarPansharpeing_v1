import torch

from src.utilities.transport.UniDB.plan_continuous import UniDBContinuous, UniDBLoss
from src.utilities.transport.UniDB.plan_disc import UniDB as UniDBDisc


def test_unidb_paper_means_match_disc_exactly() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float32

    cont = UniDBContinuous(lambda_square=0.01, gamma=1.0, T=64, schedule="cosine", eps=0.01)
    disc = UniDBDisc(
        lambda_square=cont.lambda_square,
        gamma=cont.gamma,
        T=cont.T,
        schedule=cont.schedule,
        eps=cont.eps,
        device=device,
    )

    batch, channels, height, width = 4, 3, 8, 8
    x0 = torch.randn(batch, channels, height, width, device=device, dtype=dtype)
    mu = torch.randn_like(x0)
    xt = torch.randn_like(x0)
    eps_hat = torch.randn_like(x0)

    t_index = 10
    disc.set_mu(mu)
    score = -eps_hat / disc.f_sigma(t_index)

    mu_theta_disc = disc.reverse_mean_ode_step(xt, score, t_index)
    mu_theta_cont = cont.reverse_mean_theta_adjacent(xt, mu, t_index, eps_hat)
    torch.testing.assert_close(mu_theta_cont, mu_theta_disc.to(dtype=dtype), rtol=0.0, atol=0.0)

    mu_gamma_disc = disc.reverse_optimum_step(xt, x0, t_index)
    mu_gamma_cont = cont.reverse_mean_gamma_adjacent(xt, x0, mu, t_index)
    torch.testing.assert_close(mu_gamma_cont, mu_gamma_disc.to(dtype=dtype), rtol=0.0, atol=0.0)

    class _ConstModel:
        def __init__(self, out: torch.Tensor):
            self.out = out

        def __call__(self, xt: torch.Tensor, t01: torch.Tensor, **kwargs) -> torch.Tensor:
            return self.out

    obj = UniDBLoss(cont, prediction="epsilon", loss_type="mean", norm="l1", weighting="inv")
    gen = torch.Generator(device=device).manual_seed(0)
    loss = obj.loss(
        _ConstModel(eps_hat),
        x0=x0,
        mu=mu,
        model_kwargs={},
        generator=gen,
    )
    assert torch.isfinite(loss).item()

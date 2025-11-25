import math

import torch
from torch.distributions import Beta


def l2_norm(x: torch.Tensor, eps: float = 1e-7, dim=-1) -> torch.Tensor:
    return x / torch.clamp(x.norm(dim=dim, keepdim=True), min=eps)


class PowerSphericalDistribution:
    def __init__(self, mu: torch.Tensor, kappa: torch.Tensor, eps: float = 1e-7, dim=-1):
        self.eps = eps
        self.mu = l2_norm(mu, eps)  # [..., m]
        self.kappa = torch.clamp(kappa, min=0.0)

        self.m = self.mu.shape[dim]
        self.d = self.m - 1
        beta_const = 0.5 * self.d
        self.alpha = self.kappa + beta_const  # [...,]
        self.beta = torch.as_tensor(beta_const, dtype=self.kappa.dtype, device=self.kappa.device).expand_as(self.kappa)
        self.dim = dim
        assert dim in (-1, 1), "only support dim=-1 or dim=1"

    def _log_normalizer(self) -> torch.Tensor:
        # log N_X(κ,d) = -[ (α+β)log 2 + β log π + lgamma(α) - lgamma(α+β) ]
        return (
            -(self.alpha + self.beta) * math.log(2.0)
            - self.beta * math.log(math.pi)
            - torch.lgamma(self.alpha)
            + torch.lgamma(self.alpha + self.beta)
        )

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        dot = (self.mu * x).sum(dim=self.dim).clamp(-1.0, 1.0)
        return self._log_normalizer() + self.kappa * torch.log1p(dot)

    def entropy(self) -> torch.Tensor:
        # H = -[ log N_X + κ ( log 2 + ψ(α) - ψ(α+β) ) ]
        return -(
            self._log_normalizer()
            + self.kappa * (math.log(2.0) + (torch.digamma(self.alpha) - torch.digamma(self.alpha + self.beta)))
        )

    def kl_to_uniform(self) -> torch.Tensor:
        # KL(q || U(S^{d})) = -H(q) + log |S^{d}|
        d = torch.as_tensor(self.d, dtype=self.kappa.dtype, device=self.kappa.device)
        log_area = math.log(2.0) + 0.5 * (d + 1.0) * math.log(math.pi) - torch.lgamma(0.5 * (d + 1.0))
        return -self.entropy() + log_area

    def rsample(self):
        Z = Beta(self.alpha, self.beta).rsample()  # [*S, *B]
        t = (2.0 * Z - 1.0).unsqueeze(self.dim)  # [*S, *B, 1]

        # 2) v ~ U(S^{m-2})
        if self.dim == -1:
            v = torch.randn(
                *self.mu.shape[:-1],
                self.m - 1,
                device=self.mu.device,
                dtype=self.mu.dtype,
            )  # [*S, *B, m-1]
        elif self.dim == 1:
            v = torch.randn(
                self.mu.shape[0],
                self.m - 1,
                *self.mu.shape[2:],
                device=self.mu.device,
                dtype=self.mu.dtype,
            )  # [N, m-1, H, W]
        v = l2_norm(v, self.eps, dim=self.dim)

        y = torch.cat([t, torch.sqrt(torch.clamp(1 - t**2, min=0.0)) * v], dim=self.dim)  # [*S, *B, m]

        e1 = torch.zeros_like(self.mu)
        e1[..., 0] = 1.0
        u = l2_norm(e1 - self.mu, self.eps, dim=self.dim)
        if u.dim() < y.dim():
            u = u.view((1,) * (y.dim() - u.dim()) + u.shape)
        z = y - 2.0 * (y * u).sum(dim=self.dim, keepdim=True) * u

        parallel = (self.mu - e1).abs().sum(dim=self.dim, keepdim=True) < 1e-6
        if parallel.any():
            p = parallel
            if p.dim() < y.dim() - 1:
                p = p.view((1,) * (y.dim() - 1 - p.dim()) + p.shape)
            z = torch.where(p, y, z)
        return z


if __name__ == "__main__":
    mu = torch.randn(2, 16, 32, 32)
    kappa = torch.randn(2, 32, 32)
    dim = 1

    # the same
    mu = l2_norm(mu, dim=dim)
    kappa = torch.nn.functional.softplus(kappa) + 1

    psd = PowerSphericalDistribution(mu, kappa, dim=dim)
    loss = psd.kl_to_uniform()
    print(loss.mean())

    x_r = psd.rsample()
    print(x_r.shape)

    # to 1d
    mu = mu.flatten(2).permute(0, 2, 1)
    kappa = kappa.flatten(2)

    psd = PowerSphericalDistribution(mu, kappa, dim=-1)
    loss = psd.kl_to_uniform()
    print(loss.mean())

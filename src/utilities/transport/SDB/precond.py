import torch
import torch.nn as nn
from torch import Tensor

from .plan import DiffusionTarget, expand_t_as, SDBPlan


class EDMPrecond(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        plan: SDBPlan,
        sigma_0_pow_2: float = 0.25,
        sigma_1_pow_2: float = 0.25,
        sigma_01: float = 0.125,
    ):
        super().__init__()

        self.plan = plan
        self.model = model
        assert self.plan.plan_tgt == DiffusionTarget.x_0, (
            "EDM preconditioner is only for x_0 target, but got {}".format(self.plan.plan_tgt)
        )

        self.sigma_0 = torch.tensor(sigma_0_pow_2) ** 0.5
        self.sigma_1 = torch.tensor(sigma_1_pow_2) ** 0.5
        self.sigma_01 = torch.tensor(sigma_01)

    def forward(self, x_t: Tensor, t: Tensor, **model_kwargs):
        # Expand t to match input dimensions for the next sde parameters
        t = expand_t_as(t, x_t, dim_not_match_raise=False)

        alpha_t, _ = self.plan.alpha_t_with_derivative(t)
        beta_t, _ = self.plan.beta_t_with_derivative(t)
        gamma_t, _ = self.plan.gamma_t_with_derivative(t)

        # fmt: off
        c_in = 1 / (
            alpha_t ** 2 * self.sigma_0 ** 2 +
            beta_t ** 2 * self.sigma_1 ** 2 +
            2 * alpha_t * beta_t * self.sigma_01 +
            gamma_t ** 2
        ).sqrt()

        c_skip = (alpha_t * self.sigma_0 ** 2 +
                  beta_t * self.sigma_01) * (c_in ** 2)

        c_out = (
            beta_t ** 2 * self.sigma_0 ** 2 * self.sigma_1 ** 2 -
            beta_t ** 2 * self.sigma_01 ** 2 +
            gamma_t ** 2 * self.sigma_0 ** 2
        ).sqrt() * c_in
        # fmt: on

        weight = 1 / (c_out**2)

        c_noise = 0.25 * t.log()
        model_out = self.model(x=c_in * x_t, timesteps=c_noise.flatten(0), **model_kwargs)
        if isinstance(model_out, tuple):
            aux_loss = model_out[1]
            model_out = model_out[0]
        else:
            aux_loss = torch.as_tensor(0.0, device=x_t.device)

        pred_x_0 = c_skip * x_t + c_out * model_out

        return pred_x_0, aux_loss, weight

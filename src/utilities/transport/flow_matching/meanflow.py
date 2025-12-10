from functools import partial
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn


def is_list_tuple(c):
    return isinstance(c, (list, tuple))


def slice_conditions(conditions: list[torch.Tensor] | torch.Tensor, first_n: int):
    if is_list_tuple(conditions):
        return [c[:first_n] for c in conditions]
    else:
        return conditions[:first_n]


def where_select_cond(
    conditions: list[torch.Tensor] | torch.Tensor, mask: torch.Tensor, null_cond: Tensor | None = None
):
    if is_list_tuple(conditions):
        if null_cond is None:
            null_cond = torch.zeros_like(conditions[0])
        return [torch.where(mask, c, null_cond) for c in conditions]
    else:
        if null_cond is None:
            null_cond = torch.zeros_like(conditions)
        return torch.where(mask, conditions, null_cond)


class Normalizer:
    # minmax for raw image, mean_std for vae latent
    def __init__(self, mode="minmax", mean=None, std=None):
        assert mode in ["minmax", "mean_std"], "mode must be 'minmax' or 'mean_std'"
        self.mode = mode

        if mode == "mean_std":
            if mean is None or std is None:
                raise ValueError("mean and std must be provided for 'mean_std' mode")
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

    @classmethod
    def from_list(cls, config):
        """
        config: [mode, mean, std]
        """
        mode, mean, std = config
        return cls(mode, mean, std)

    def norm(self, x):
        if self.mode == "minmax":
            return x * 2 - 1
        elif self.mode == "mean_std":
            return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def unnorm(self, x):
        if self.mode == "minmax":
            x = x.clip(-1, 1)
            return (x + 1) * 0.5
        elif self.mode == "mean_std":
            return x * self.std.to(x.device) + self.mean.to(x.device)


def stopgrad(x):
    return x.detach()


def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """
    Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
    Args:
        error: Tensor of shape (B, C, W, H)
        gamma: Power used in original ||Δ||^{2γ} loss
        c: Small constant for stability
    Returns:
        Scalar loss
    """
    delta_sq = torch.mean(error**2, dim=(1, 2, 3), keepdim=False)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq  # ||Δ||^2
    return (stopgrad(w) * loss).mean()


def expand_as(t, x):
    dims_x = x.dim()
    ep_dims = dims_x - t.dim()
    t = t.expand(x.shape[0], *[1] * ep_dims)
    return t


class MeanFlow:
    def __init__(
        self,
        channels=1,
        image_size=32,
        normalizer=["minmax", None, None],
        # mean flow settings
        flow_ratio=0.50,
        # time distribution, mu, sigma
        time_dist=["lognorm", -0.4, 1.0],
        cfg_ratio=0.10,
        # set scale as none to disable CFG distill
        cfg_scale: float | str = 2.0,
        use_improved: bool = False,
        network_inp_cfg: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.use_improved = use_improved
        self._cfg_learnable = isinstance(cfg_scale, nn.Parameter)

        self.normer = Normalizer.from_list(normalizer)

        self.flow_ratio = flow_ratio
        self.time_dist = time_dist
        self.cfg_ratio = cfg_ratio
        self.network_inp_cfg = network_inp_cfg
        self.w = cfg_scale

    def sample_cfg(self, bs: int, device: str | torch.device = "cuda"):
        if isinstance(self.w, str):
            assert self.network_inp_cfg
            if self.w == "imf_cfg":
                omega_max, beta = 8.0, 1.0
                u = torch.rand(bs, device=device)
                if abs(beta - 1.0) < 1e-6:
                    w = torch.exp(u * torch.log(torch.tensor(omega_max)))
                else:
                    numerator = (omega_max ** (1 - beta) - 1) * u + 1
                    w = numerator ** (1 / (1 - beta))
                return w
            else:
                raise ValueError(f"Unknown cfg_scale: {self.w}")
        else:
            return self.w

    # fix: r should be always not larger than t
    def sample_t_r(self, batch_size, device):
        if self.time_dist[0] == "uniform":
            samples = np.random.rand(batch_size, 2).astype(np.float32)

        elif self.time_dist[0] == "lognorm":
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))  # Apply sigmoid

        # Assign t = max, r = min, for each pair
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        # Select only some samples to apply the Meanflow
        # else the original flow matching for stability
        num_selected = int(self.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r

    def _get_velocity(self, x):
        batch_size = x.shape[0]
        device = x.device

        t, r = self.sample_t_r(batch_size, device)

        t_ = expand_as(t, x).detach().clone()
        r_ = expand_as(r, x).detach().clone()

        e = torch.randn_like(x)
        x = self.normer.norm(x)

        z = (1 - t_) * x + t_ * e
        v = e - x

        return (t, r), (t_, r_), z, v

    def _imf_loss(self, model, x, c=None, null_c=None, **model_kwargs):
        (t, r), (t_, r_), z, v = self._get_velocity(x)

        # IMF network velocity
        v_net = model(z, t, t, c, v_pred=True, **model_kwargs)  # forward v head if it has

        # CFG
        if c is not None and self.cfg_ratio > 0.0:
            w = self.sample_cfg(x.shape[0], device=x.device)
            bs = x.shape[0]
            bs_g = int(bs * self.cfg_ratio)

            # Forward with condition
            cfg_mask = torch.rand(bs) < self.cfg_ratio
            if self.network_inp_cfg:
                v_c = model(z, t, t, w, c, **model_kwargs)
                v_u = model(z, t, t, w, null_c, **model_kwargs)
            else:
                v_c = model(z, t, t, c, **model_kwargs)
                v_u = model(z, t, t, null_c, **model_kwargs)

            v_g = v + (1 - 1 / w) * (v_c - v_u)

        else:
            v_g = v_net

        # JVP
        model_partial = partial(model, cond=c) if not self.network_inp_cfg else partial(model, cfg_w=w, cond=c)
        u_t, dudt = torch.autograd.functional.jvp(
            model_partial,
            (z, t, r),
            (v_net, torch.ones_like(t), torch.zeros_like(r)),
        )

        # IMF compound velocity
        comp_v = u_t + (t_ - r_) * stopgrad(dudt)
        error = comp_v - v_g

        # IMF loss
        loss_dict = {}
        loss_total = 0.0

        loss_imf = adaptive_l2_loss(error)  # loss_imf -> u head -> backbone
        loss_dict["loss_imf"] = loss_imf.item()

        if self.network_has_v_head:
            loss_fm = F.mse_loss(v_net, v_g)  # loss_fm -> v head -> backbone
            loss_dict["loss_fm"] = loss_fm.item()
            loss_total += loss_fm * 0.2  # TODO: add weight in init.
        else:
            loss = loss_imf

        return loss, loss_dict

    def _mf_loss(self, model, x, c=None, null_c=None, **model_kwargs):
        (t, r), (t_, r_), z, v = self._get_velocity(x)

        # CFG
        if c is not None and self.cfg_ratio > 0.0:
            assert isinstance(self.w, float)

            bs = x.shape[0]
            bs_g = int(bs * self.cfg_ratio)

            # Forward with condition
            z_g = z[:bs_g]
            t_g = t[:bs_g]
            v_g_ = v[:bs_g]
            c_g = slice_conditions(c, bs_g)

            # Conditional velocity
            v_cond = model(z_g, t_g, t_g, c_g, **model_kwargs)
            v_uncond = model(z_g, t_g, t_g, null_c, **model_kwargs)
            v_g = v_g_ * self.w + (1 - self.w) * v_uncond
            v_g = torch.cat([v_g, v[bs_g:]], dim=0)

            # assert null_c is not None
            # uncond = null_c
            # bs = c.shape[0] if not is_list_tuple(c) else c[0].shape[0]
            # cfg_mask = torch.rand_like(bs) < self.cfg_ratio

            # # CFG conditions
            # c = torch.where(cfg_mask, uncond, c)

            # if self.w is not None:
            #     with torch.no_grad():
            #         u_t = model(z, t, t, uncond)
            #     v_g = self.w * v + (1 - self.w) * u_t
            #     # offical JAX repo uses original v for unconditional items
            #     cfg_mask = rearrange(cfg_mask, "b -> b 1 1 1").bool()
            #     v_g = torch.where(cfg_mask, v, v_g)
            # else:
            #     v_g = v
        else:
            v_g = v

        # forward pass
        # u = model(z, t, r, y=c)
        model_partial = partial(model, cond=c)
        jvp_args = (
            lambda z, t, r: model_partial(z, t, r),
            (z, t, r),
            (v_g, torch.ones_like(t), torch.zeros_like(r)),
        )
        u, dudt = torch.autograd.functional.jvp(*jvp_args)

        # Mean velocity target
        u_tgt = v_g - (t_ - r_) * dudt

        error = u - stopgrad(u_tgt)
        loss = adaptive_l2_loss(error)

        mse_val = (stopgrad(error) ** 2).mean()

        # Reconstruction
        # u = replace some of with cfg uncondition
        u_hat_pred = u + (t_ - r_) * dudt  # ~= v_hat
        recon = v + x  # only apporx e
        return loss, mse_val, recon

    @torch.no_grad()
    def sample_each_class(self, model, n_per_class, classes=None, sample_steps=5, device="cuda"):
        model.eval()

        if classes is None:
            c = torch.arange(self.num_classes, device=device).repeat(n_per_class)
        else:
            c = torch.tensor(classes, device=device).repeat(n_per_class)

        z = torch.randn(c.shape[0], self.channels, self.image_size, self.image_size, device=device)

        t_vals = torch.linspace(1.0, 0.0, sample_steps + 1, device=device)

        for i in range(sample_steps):
            t = torch.full((z.size(0),), t_vals[i], device=device)
            r = torch.full((z.size(0),), t_vals[i + 1], device=device)

            # print(f"t: {t[0].item():.4f};  r: {r[0].item():.4f}")

            t_ = rearrange(t, "b -> b 1 1 1").detach().clone()
            r_ = rearrange(r, "b -> b 1 1 1").detach().clone()

            v = model(z, t, r, c)
            z = z - (t_ - r_) * v

        z = self.normer.unnorm(z)
        return z

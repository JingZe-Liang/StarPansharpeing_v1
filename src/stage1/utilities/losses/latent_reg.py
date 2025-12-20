"""
Reimplementation of SSVAE paper: Delving into Latent Spectral Biasing of Video VAEs for Superior Diffusability
https://github.com/zai-org/SSVAE

Regularizations of VAE/AE latent space.
Including Latent Masked Reconstruction (LMR) and LCR.

Regularization of channel nested dropping from DC-VAE2.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from jaxtyping import Float
import numpy as np
from einops import rearrange
from loguru import logger
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "NestChannelDrop",
    "lcr_loss",
    "lmr_apply",
]


class NestChannelDrop(nn.Module):
    def __init__(
        self,
        drop_type: str | list[int] = "uniform_4",
        max_channels: int = 16,
        drop_prob: float = 0.5,
    ):
        super().__init__()
        self.max_channels = max_channels
        self.drop_prob = drop_prob

        if isinstance(drop_type, str):
            drop_type, args = drop_type.lower().split("_")
            self.drop_type = drop_type
            if drop_type == "exp":
                self.sample_kwargs = {"lambda": float(args)}
            elif drop_type == "uniform":
                assert args.isdigit(), "args should be an int"
                self.sample_kwargs = {"low": int(args)}
            else:
                raise ValueError(f"drop_type {drop_type} not supported, only exp and uniform are supported")
        else:  # list
            self.drop_list = drop_type
            assert max(self.drop_list) < self.max_channels, (
                f"max_channels {self.max_channels} should be larger than the max of drop_list {max(self.drop_list)}"
            )
            self.drop_type = "prefixed"
        logger.info(
            f"[NestChannelDrop]: drop_type={self.drop_type}, max_channels={self.max_channels}, drop_prob={self.drop_prob}"
        )

        self.channel_arange = nn.Buffer(torch.arange(self.max_channels, dtype=torch.int32), persistent=False)

    def exponential_sampling(self, lambda_val, size=1):
        u = np.random.uniform(size=size)
        k = -np.log(1 - u) / lambda_val
        return torch.as_tensor(np.floor(k).astype(int)).clip_(0, self.max_channels).unsqueeze(-1)

    def uniform_sampling(self, low: int, size: int = 1):
        # (bs, 1)
        k = torch.randint(low=low, high=self.max_channels, size=(size, 1))
        return k

    def prefixed_sampling(self, size: int = 1):
        drop_list = self.drop_list
        leave_channels = np.random.choice(drop_list, size=size, replace=True)
        return torch.as_tensor(leave_channels).unsqueeze(-1)

    def forward(self, z, inference_channels: int | None = None) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Apply channel drop to the input tensor.

        Args:
            z: Input tensor of shape (bs, c, h, w)
            inference_channels: Number of channels to keep during inference

        Returns:
            If applying channel drop: tuple of (z, mask) where mask is the channel mask
            If not applying channel drop: z tensor only
        """
        if (self.training and np.random.random() > self.drop_prob) or (
            not self.training and inference_channels is None
        ):
            return z

        if inference_channels is not None:
            assert not self.training
            assert inference_channels <= self.max_channels
            return z[:, :inference_channels]

        assert self.max_channels == z.shape[1]

        bs = z.shape[0]
        if self.drop_type == "exp":
            leave_channels = self.exponential_sampling(size=bs, **self.sample_kwargs)
        elif self.drop_type == "uniform":
            leave_channels = self.uniform_sampling(size=bs, **self.sample_kwargs)  # type: ignore
        elif self.drop_type == "prefixed":
            leave_channels = self.prefixed_sampling(size=bs)
        else:
            raise ValueError(f"drop_type {self.drop_type} not supported, only exp and uniform are supported")

        # drop channels

        # 1. expand the cached empty z
        # if self.dropped_x.shape[-2:] != z.shape[-2:]:
        #     if self.learnable:
        #         z_empty = nn.functional.interpolate(
        #             self.dropped_x,
        #             size=z.shape[-2:],
        #             mode="bilinear",
        #             align_corners=False,
        #         )
        #         z_empty = z_empty.expand(bs, -1, -1, -1)
        #     else:
        #         z_empty = torch.zeros_like(z)
        # else:
        #     z_empty = self.dropped_x.expand(bs, -1, -1, -1)
        z_zeros = torch.zeros_like(z)

        # 2. drop channels
        channels = self.channel_arange[None].expand(bs, -1)  # type: ignore
        cond = channels < leave_channels.to(channels)
        mask = cond.unsqueeze(-1).unsqueeze(-1).expand_as(z)  # (bs, c, h, w)
        z = torch.where(mask, z, z_zeros)

        return z, mask


def _sample_t_distributional(
    bs: int,
    device: str | torch.device = "cuda",
    noise_type_cfg: str = "beta_1_3",
    timestep_range: tuple[int, int] = (0, 1),
):
    """Sample a latent noising time-step in specific distribution"""
    # NOTE: helper sub-functions are defined below; this function only parses the string and dispatches.
    # Output shape: (bs, 1, 1, 1); value range depends on distribution.

    # Backward compatibility: map patterns like "exp_0.2" to "exp_max_0.2_lambda_5"
    if noise_type_cfg.startswith("exp_") and not noise_type_cfg.startswith("exp_max_"):
        _val = float(noise_type_cfg.split("_")[1])
        noise_type_cfg = f"exp_max_{_val}_lambda_5"

    # Uniform in [0, max_val]
    elif noise_type_cfg.startswith("uniform_max_"):
        max_val = float(noise_type_cfg.split("_")[-1])
        t = torch.rand((bs,), device=device) * max_val

    # Truncated exponential on [0, T]
    elif noise_type_cfg.startswith("exp_max_"):
        parts = noise_type_cfg.split("_")
        # exp_max_{T} or exp_max_{T}_lambda_{lam}
        T = float(parts[2])
        lam = 5.0
        if len(parts) >= 5 and parts[3] == "lambda":
            lam = float(parts[4])

        u = torch.rand((bs,), device=device)
        exp_term = torch.exp(torch.tensor(-lam * T, device=device))
        t = -torch.log1p(-u * (1.0 - exp_term)) / lam

    # Beta(a, b) scaled to [0,1]
    elif noise_type_cfg.startswith("beta_"):
        parts = noise_type_cfg.split("_")
        a = float(parts[1])
        b = float(parts[2])
        dist = torch.distributions.Beta(
            torch.tensor(a, device=device),
            torch.tensor(b, device=device),
        )
        t = dist.sample((bs,))
        return t

    # Default: standard normal, absolute value then clamp to [0,1] for stability
    else:
        t = torch.abs(torch.rand((bs,), device=device))

    return t


def _calculate_pearson_autocorr(x, renorm=True):
    # v: (B, N, D)
    # return: scalar p
    # treat v as a batch B of N random variables, each has D observations
    assert x.dim() == 3, f"x should be 3D tensor, got {x.shape=}"

    D = x.size(-1)
    if renorm:
        x = x - x.mean(dim=-1, keepdim=True)
        norm = torch.norm(x, dim=-1, keepdim=True) + 1e-8
        x = x / norm
    sim = torch.matmul(x, x.transpose(1, 2))  # (B, N, N)
    N = x.size(1)
    mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1)
    mean_corr = sim[:, mask == 1]
    mean_corr = mean_corr.mean()
    if not renorm:
        mean_corr = mean_corr / D
    return mean_corr


def lcr_loss(
    z: Float[Tensor, "b c h w"],
    thresh: float = 0.75,
    windows_size: list[int] = [2, 2],
    weight_type: str = "average",
    renorm: bool = True,
):
    """
    Compute the LCR loss on the latent.
    $\\mathcal{L}_{LCR} = \text{ReLU}(\alpha - \\mathbb{E}[R(\\mathbf{p})])$
    """
    # Compute the local correlation
    H, W = z.shape[-2:]
    H_win, W_win = windows_size
    assert H % H_win == 0 and W % W_win == 0, f"H and W should be divisible by {H_win} and {W_win}"

    # 1. to local windows, [B, h, w, hw, ww, c]
    z_win = rearrange(z, "b c (h h_win) (w w_win) -> b (h w) (h_win w_win) c", h_win=H_win, w_win=W_win)

    # 2. call pearson autocorr
    z_win_merged = rearrange(z_win, "b n hw c -> (b n) hw c")
    local_corr = _calculate_pearson_autocorr(z_win_merged, renorm=renorm)

    # 3. compute loss
    lcr_loss = F.relu(thresh - local_corr)

    return local_corr, lcr_loss

@dataclass
class LatentMaskConfig:
    """Any type compatible with OmegaConf structures."""
    mask_ratios: Any = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75])
    block_sizes: Any = field(default_factory=lambda: {16: [1, 1], 32: [2, 2]})
    mask_probs: Any = field(default_factory=lambda: {16: [0.7, 0.1, 0.1, 0.1], 32: [0.6, 0.1, 0.15, 0.15]})


def lmr_apply(
    z: Float[Tensor, "b c h w"],
    mask_ratios: list[float] = [0.0, 0.25, 0.5, 0.75],
    block_sizes: dict[int, list[int]] = {16: [1, 1], 32: [2, 2]},
    mask_probs: dict[int, list[float]] = {16: [0.7, 0.1, 0.1, 0.1], 32: [0.6, 0.1, 0.15, 0.15]},
):
    size = z.shape[-1]
    block_size = block_sizes.get(size, [1, 1])

    mask_prob_weight = mask_probs.get(size, [0.7, 0.1, 0.1, 0.1])
    mask_ratio = float(np.random.choice(mask_ratios, p=mask_prob_weight))
    _least_mask_prob = mask_ratios[1]

    # Not masking
    if mask_ratio < _least_mask_prob:
        return z

    h_blk, w_blk = block_size
    B, C, H, W = z.shape
    N = H * W
    nh_blk, nw_blk = H // h_blk, W // w_blk
    total_blks = nh_blk * nw_blk
    masked_area_per_blk = h_blk * w_blk
    num_mask_blocks = max(1, int(N * mask_ratio) // masked_area_per_blk)
    mask_blk_indices = torch.randperm(total_blks, device=z.device)[:num_mask_blocks]

    mask = torch.zeros(B, 1, H, W, device=z.device, dtype=torch.bool)

    # cal mask coordinates
    for blk_idx in mask_blk_indices:
        blk_h = blk_idx // nw_blk
        blk_w = blk_idx % nw_blk
        h_start = blk_h * h_blk
        h_end = h_start + h_blk
        w_start = blk_w * w_blk
        w_end = w_start + w_blk
        mask[:, :, h_start:h_end, w_start:w_end] = True

    # Apply mask to z: set masked regions to 0
    z_masked = z * (~mask).float()

    return z_masked, mask


# ------ Tests ------ #


def __test_lmr_apply():
    b, c, h, w = 2, 16, 32, 32
    z = torch.randn(b, c, h, w)
    z_masked, mask = lmr_apply(z, mask_ratios=[0.0, 0.5], block_sizes={32: [4, 4]}, mask_probs={32: [0.0, 1.0]})
    print("z_masked:", z_masked)
    print("mask:", mask)
    print("masked ratio:", mask.float().mean().item())


def __test_lcr_loss():
    b, c, h, w = 2, 16, 32, 32
    z = torch.randn(b, c, h, w)
    local_corr, lcr = lcr_loss(z, thresh=0.75, windows_size=[4, 4], weight_type="average", renorm=True)
    print("local_corr:", local_corr)
    print("lcr_loss:", lcr)


if __name__ == "__main__":
    __test_lmr_apply()
    __test_lcr_loss()

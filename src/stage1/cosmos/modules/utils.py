# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared utilities for the networks module."""

from collections.abc import Callable
from functools import partial
from inspect import Parameter, isclass, isfunction, signature
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from einops import pack, rearrange, unpack

from .rmsnorm_triton import TritonRMSNorm2dFunc


def time2batch(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    batch_size = x.shape[0]
    return rearrange(x, "b c t h w -> (b t) c h w"), batch_size


def batch2time(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    return rearrange(x, "(b t) c h w -> b c t h w", b=batch_size)


def space2batch(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    batch_size, height = x.shape[0], x.shape[-2]
    return rearrange(x, "b c t h w -> (b h w) c t"), batch_size, height


def batch2space(x: torch.Tensor, batch_size: int, height: int) -> torch.Tensor:
    return rearrange(x, "(b h w) c t -> b c t h w", b=batch_size, h=height)


def cast_tuple(t: Any, length: int = 1) -> Any:
    return t if isinstance(t, tuple) else ((t,) * length)


def replication_pad(x):
    return torch.cat([x[:, :, :1, ...], x], dim=2)


def divisible_by(num: int, den: int) -> bool:
    return (num % den) == 0


def is_odd(n: int) -> bool:
    return not divisible_by(n, 2)


def nonlinearity(x, mp=False):
    if mp:
        return torch.nn.functional.silu(x) / 0.596
    return torch.nn.functional.silu(x)


def gelu_nonlinear(x, clamp=None):
    x = torch.nn.functional.gelu(x, approximate="tanh")
    if clamp is not None:
        x = torch.clamp(x, min=-clamp, max=clamp)
    return x


def mp_sum(a, b, t=0.5):
    return a.lerp(b, t) / np.sqrt((1 - t) ** 2 + t**2)


# * --- Norm --- #


class RMSNorm2d(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        scale: float | None = None,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.use_bias = bias

        if self.elementwise_affine:
            self.weight = torch.nn.parameter.Parameter(torch.ones(self.num_features))
            if bias:
                self.bias = torch.nn.parameter.Parameter(torch.zeros(self.num_features))
            else:
                self.bias = 0.0
        else:
            self.weight = 1.0
            self.bias = 0.0

        self.scale = scale if scale is not None else num_features**0.5

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.normalize(x, dim=1, eps=self.eps)
        w = self.weight.view(1, -1, 1, 1) if self.elementwise_affine else 1
        b = (
            self.bias.view(1, -1, 1, 1)
            if (self.elementwise_affine and self.use_bias)
            else 0.0
        )
        x = x * self.scale * w + b

        return x


class LayerNorm2d(torch.nn.LayerNorm):
    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x - torch.mean(x, dim=1, keepdim=True)
        out = out / torch.sqrt(torch.square(out).mean(dim=1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out


class TritonRMSNorm2d(torch.nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return TritonRMSNorm2dFunc.apply(x, self.weight, self.bias, self.eps)


def unit_magnitude_normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


class AdaptiveGroupNorm32(nn.Module):
    def __init__(self, z_channel, in_filters, eps=1e-6):
        super().__init__()
        self.gn = nn.GroupNorm(
            num_groups=32, num_channels=in_filters, eps=eps, affine=False
        )
        # self.lin = nn.Linear(z_channels, in_filters * 2)
        self.gamma = nn.Linear(z_channel, in_filters)
        self.beta = nn.Linear(z_channel, in_filters)
        self.eps = eps

    def forward(self, x, quantizer):
        B, C, _, _ = x.shape
        # quantizer = F.adaptive_avg_pool2d(quantizer, (1, 1))
        # calcuate var for scale
        scale = rearrange(quantizer, "b c h w -> b c (h w)")
        scale = scale.var(dim=-1) + self.eps  # not unbias
        scale = scale.sqrt()
        scale = self.gamma(scale).view(B, C, 1, 1)

        # calculate mean for bias
        bias = rearrange(quantizer, "b c h w -> b c (h w)")
        bias = bias.mean(dim=-1)
        bias = self.beta(bias).view(B, C, 1, 1)

        x = self.gn(x)
        x = scale * x + bias

        return x


def Normalize(
    in_channels,
    norm_type: str
    | Literal["gn", "bn2d", "ln2d", "rms_native", "rms_triton", "unit_vec_norm", "none"]
    | None = "gn",
    **norm_kwargs,
):
    if norm_type == "gn":
        return torch.nn.GroupNorm(
            num_channels=in_channels,
            eps=1e-6,
            affine=True,
            num_groups=norm_kwargs.get("num_groups", 32),
        )
    elif norm_type == "bn2d":
        cls = torch.nn.BatchNorm2d
    elif norm_type == "ln2d":
        cls = LayerNorm2d
    elif norm_type == "unit_vec_norm":
        return partial(
            unit_magnitude_normalize, dim=1, eps=norm_kwargs.get("eps", 1e-4)
        )
    elif norm_type == "rms_native":
        cls = RMSNorm2d
    elif norm_type == "rms_triton":
        cls = TritonRMSNorm2d
    elif norm_type in (None, "none"):
        return torch.nn.Identity()
    else:
        raise ValueError(
            f"Unknown normalization type: {norm_type}. Supported types are: 'gn', 'bn2d', 'ln2d', 'rms_native', "
            "'rms_triton', None or 'none'."
        )

    return cls(in_channels, **extract_needed_kwargs(norm_kwargs, cls))


class CausalNormalize(torch.nn.Module):
    def __init__(self, in_channels, num_groups=1):
        super().__init__()
        self.norm = torch.nn.GroupNorm(
            num_groups=num_groups,
            num_channels=in_channels,
            eps=1e-6,
            affine=True,
        )
        self.num_groups = num_groups

    def forward(self, x):
        # if num_groups !=1, we apply a spatio-temporal groupnorm for backward compatibility purpose.
        # All new models should use num_groups=1, otherwise causality is not guaranteed.
        if self.num_groups == 1:
            x, batch_size = time2batch(x)
            return batch2time(self.norm(x), batch_size)
        return self.norm(x)


# * --- Utilities --- #


def exists(v):
    return v is not None


def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def round_ste(z: torch.Tensor) -> torch.Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


def log(t, eps=1e-5):
    return t.clamp(min=eps).log()


def entropy(prob):
    return (-prob * log(prob)).sum(dim=-1)


def extract_needed_kwargs(
    kwargs: dict, cls: Callable | type, include_default: bool = False
) -> dict:
    """
    Extracts the subset of `kwargs` that match the parameters of a given class's __init__ method or a function.

    If a parameter is not provided in `kwargs` but has a default value, the default is used.
    Missing required parameters will raise a ValueError.

    Args:
        kwargs (dict): A dictionary of keyword arguments to filter.
        cls (type or function): A class or function whose signature is used to extract the needed kwargs.

    Returns:
        dict: A dictionary containing only the relevant keyword arguments.

    Raises:
        AssertionError: If `cls` is a class without an __init__ method.
        ValueError: If a required argument is missing or an unsupported type is passed.
    """
    needed_kwargs = {}
    if isclass(cls):
        assert hasattr(cls, "__init__"), f"{cls} does not have an __init__ method."
        sig = signature(cls.__init__)
    elif isfunction(cls):
        sig = signature(cls)
    else:
        raise ValueError(f"Expected a class or function, got {type(cls)}.")

    for param in sig.parameters.values():
        if param.name == "self":
            continue
        if param.name in kwargs:
            needed_kwargs[param.name] = kwargs[param.name]
        elif include_default and param.default is not Parameter.empty:
            needed_kwargs[param.name] = param.default
        # else:
        #     raise ValueError(
        #         f"Missing required argument '{param.name}' for {cls.__name__}."
        #     )
    return needed_kwargs


def val2tuple(x: list | tuple | Any, min_len: int = 1) -> tuple:
    if isinstance(x, (list, tuple)):
        x = list(x)
    else:
        x = [x] * min_len

    return tuple(x)

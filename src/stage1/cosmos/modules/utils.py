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

from typing import Any

import torch
from einops import pack, rearrange, unpack

from .rmsnorm_triton import TritonRMSNorm2dFunc


def time2batch(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    batch_size = x.shape[0]
    return rearrange(x, "b c t h w -> (b t) c h w"), batch_size


def batch2time(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    return rearrange(x, "(b t) c h w -> b c t h w", b=batch_size)


def space2batch(x: torch.Tensor) -> tuple[torch.Tensor, int]:
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


def nonlinearity(x):
    return x * torch.sigmoid(x)


# * --- Norm --- #


class RMSNorm2d(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = torch.nn.parameter.Parameter(torch.ones(self.num_features))
            if bias:
                self.bias = torch.nn.parameter.Parameter(torch.zeros(self.num_features))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (
            x / torch.sqrt(torch.square(x.float()).mean(dim=1, keepdim=True) + self.eps)
        ).to(x.dtype)
        if self.elementwise_affine:
            x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x


class LayerNorm2d(torch.nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x - torch.mean(x, dim=1, keepdim=True)
        out = out / torch.sqrt(torch.square(out).mean(dim=1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out


class TritonRMSNorm2d(torch.nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return TritonRMSNorm2dFunc.apply(x, self.weight, self.bias, self.eps)


def Normalize(in_channels, norm_type="gn", **norm_kwargs):
    if norm_type == "gn":
        return torch.nn.GroupNorm(
            num_channels=in_channels,
            eps=1e-6,
            affine=True,
            num_groups=norm_kwargs.get("num_groups", 32),
        )
    elif norm_type == "ln":
        norm_kwargs.pop("num_groups", None)
        return LayerNorm2d(in_channels)
    elif norm_type == "rms_native":
        norm_kwargs.pop("num_groups", None)
        return RMSNorm2d(in_channels, **norm_kwargs)
    elif norm_type == "rms_triton":
        norm_kwargs.pop("num_groups", None)
        return TritonRMSNorm2d(in_channels, **norm_kwargs)
    else:
        raise ValueError(
            f"Unknown normalization type: {norm_type}. Supported types are: 'gn', 'ln', 'rms'."
        )


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

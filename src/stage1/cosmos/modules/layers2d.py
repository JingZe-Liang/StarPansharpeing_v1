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

"""The model definition for Continuous 2D layers

Adapted from: https://github.com/CompVis/stable-diffusion/blob/
21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/model.py

[Copyright (c) 2022 Robin Rombach and Patrick Esser and contributors]
https://github.com/CompVis/stable-diffusion/blob/
21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/LICENSE
"""

import math
import sys
from functools import partial
from inspect import signature
from typing import Any, Callable, Literal, Optional, Sequence, Union

import numpy as np

# pytorch_diffusion + derived encoder decoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint
from transformers.activations import ACT2FN
from typing_extensions import deprecated

from src.stage1.cosmos.modules.patching import Patcher, UnPatcher
from src.stage1.cosmos.modules.utils import (
    Normalize,
    gelu_nonlinear,
    nonlinearity,
    val2tuple,
)
from src.stage1.MoEs.deepseek_moe.moe_layer import DeepseekECMoE
from src.stage1.MoEs.deepseek_moe.moe_layer import DeepseekV2MoE as DeepSeekTCMoE
from src.utilities.logging import log_print

# * --- Upsample and Downsample --- #


class UpsampleRepeatConv(nn.Module):
    def __init__(self, in_channels: int, padding_mode: str = "zeros"):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode=padding_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        return self.conv(x)


def get_same_padding(
    kernel_size: Union[int, tuple[int, ...]],
) -> Union[int, tuple[int, ...]]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


class DownsamplePadConv(nn.Module):
    def __init__(self, in_channels: int, padding_mode: str = "constant"):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=2, padding=0
        )
        self.padding_mode = padding_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (0, 1, 0, 1)
        if self.padding_mode != "constant":
            x = F.pad(x, pad, mode=self.padding_mode)
        else:
            x = F.pad(x, pad, mode="constant", value=0)
        return self.conv(x)


class ConvPixelUnshuffleDownSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.factor = factor
        out_ratio = factor**2
        assert out_channels % out_ratio == 0
        self.conv = nn.Conv2d(
            in_channels,
            out_channels // out_ratio,
            kernel_size,
            padding=get_same_padding(kernel_size),
            padding_mode=padding_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = F.pixel_unshuffle(x, self.factor)
        return x


class PixelUnshuffleChannelAveragingDownSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
        # group_size: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert in_channels * factor**2 % out_channels == 0
        self.group_size = in_channels * factor**2 // out_channels
        # hidden = out_channels * group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pixel_unshuffle(
            x, self.factor
        )  # c * factor ** 2 -> hidden = out_c * group_size
        B, C, H, W = x.shape
        x = x.view(B, self.out_channels, self.group_size, H, W)
        x = x.mean(dim=2)
        return x


class ConvPixelShuffleUpSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.factor = factor
        out_ratio = factor**2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * out_ratio,
            kernel_size,
            padding=get_same_padding(kernel_size),
            padding_mode=padding_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = F.pixel_shuffle(x, self.factor)
        return x


class InterpolateConvUpSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
        mode: str = "nearest",
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.factor = factor
        self.mode = mode
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            1,
            padding=get_same_padding(kernel_size),
            padding_mode=padding_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, scale_factor=self.factor, mode=self.mode)
        x = self.conv(x)
        return x


class ChannelDuplicatingPixelUnshuffleUpSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert out_channels * factor**2 % in_channels == 0
        self.repeats = out_channels * factor**2 // in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = F.pixel_shuffle(x, self.factor)
        return x


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        use_bias: bool = False,
        dropout: float = 0,
        norm: str = "bn2d",
        act_func: str = "relu",
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()

        padding = get_same_padding(kernel_size)
        padding *= dilation

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
            padding_mode=padding_mode,
        )
        self.norm = Normalize(in_channels=out_channels, norm_type=norm)
        self.act = build_act(act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


# TODO: this version causes the norm is high
class GLUMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        use_bias=False,
        norm=(None, None, "ln2d"),
        act_func=("silu", "silu", None),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act_func = val2tuple(act_func, 3)

        mid_channels = (
            round(in_channels * expand_ratio) if mid_channels is None else mid_channels
        )

        self.glu_act = build_act(act_func[1], inplace=False)
        self.inverted_conv = ConvLayer(
            in_channels,
            mid_channels * 2,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.depth_conv = ConvLayer(
            mid_channels * 2,
            mid_channels * 2,
            kernel_size,
            stride=stride,
            groups=mid_channels * 2,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=None,
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            use_bias=use_bias[2],
            norm=norm[2],
            act_func=act_func[2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)

        x, gate = torch.chunk(x, 2, dim=1)
        gate = self.glu_act(gate)
        x = x * gate

        x = self.point_conv(x)
        return x


# register activation function here
REGISTERED_ACT_DICT: dict[str, type] = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "hswish": nn.Hardswish,
    "silu": nn.SiLU,
    "gelu": partial(nn.GELU, approximate="tanh"),
}


def build_kwargs_from_config(config: dict, target_func: Callable):
    valid_keys = list(signature(target_func).parameters)
    kwargs = {}
    for key in config:
        if key in valid_keys:
            kwargs[key] = config[key]
    return kwargs


def build_act(name: str, **kwargs) -> Optional[nn.Module]:
    if name in REGISTERED_ACT_DICT:
        act_cls = REGISTERED_ACT_DICT[name]
        args = build_kwargs_from_config(kwargs, act_cls)
        return act_cls(**args)
    else:
        return None


class ResidualBlock(nn.Module):
    def __init__(
        self,
        main: Optional[nn.Module],
        shortcut: Optional[nn.Module],
        post_act=None,
        pre_norm: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)

    # @torch.compile
    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm is None:
            return self.main(x)
        else:
            return self.main(self.pre_norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            res = self.forward_main(x) + self.shortcut(x)
            if self.post_act:
                res = self.post_act(res)
        return res


# * --- Upsample and downsample entries --- #


def build_upsample_block(
    block_type: str,
    in_channels: int,
    out_channels: int,
    shortcut: Optional[str],
    padding_mode: str = "zeros",
) -> nn.Module:
    if block_type == "ConvPixelShuffle":
        block = ConvPixelShuffleUpSampleLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            factor=2,
            padding_mode=padding_mode,
        )
    elif block_type == "RepeatConv":
        block = UpsampleRepeatConv(in_channels, padding_mode=padding_mode)
    elif block_type == "InterpolateConv":
        block = InterpolateConvUpSampleLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            factor=2,
            padding_mode=padding_mode,
        )
    else:
        raise ValueError(f"block_type {block_type} is not supported for upsampling")

    if shortcut is None:
        pass
    elif shortcut == "duplicating":
        shortcut_block = ChannelDuplicatingPixelUnshuffleUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=2
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for upsample")
    return block


def build_downsample_block(
    block_type: str,
    in_channels: int,
    out_channels: int,
    shortcut: Optional[str],
    padding_mode: str = "zeros",
) -> nn.Module:
    if block_type == "Conv":
        block = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode=padding_mode,
        )
    elif block_type == "PadConv":
        block = DownsamplePadConv(in_channels=in_channels)
    elif block_type == "ConvPixelUnshuffle":
        block = ConvPixelUnshuffleDownSampleLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            factor=2,
            padding_mode=padding_mode,
        )
    else:
        raise ValueError(f"block_type {block_type} is not supported for downsampling")

    if shortcut is None:
        pass
    elif shortcut == "averaging":
        shortcut_block = PixelUnshuffleChannelAveragingDownSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=2
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for downsample")
    return block


# * --- Blocks --- #


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int | None = None,
        dropout: float,
        use_residual_factor: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        padding_mode = kwargs.get("padding_mode", "zeros")
        norm_type = kwargs.get("norm_type", "gn")
        gn_norm_groups = kwargs.get("num_groups", 32)

        self.norm1 = Normalize(
            in_channels, num_groups=gn_norm_groups, norm_type=norm_type
        )
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode=padding_mode,
        )
        self.norm2 = Normalize(
            out_channels, num_groups=gn_norm_groups, norm_type=norm_type
        )
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode=padding_mode,
        )
        self.nin_shortcut = (
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act_checkpoint = kwargs.get("act_checkpoint", False)
        self.use_residual_factor = use_residual_factor
        if use_residual_factor:
            self.residual_factor = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward_fn(
        self,
        x: torch.Tensor,
        slots: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        x = self.nin_shortcut(x)
        if self.use_residual_factor:
            h = h * self.residual_factor

        return x + h

    def forward(
        self,
        x: torch.Tensor,
        slots: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # slots, t not used, compacted with ResnetBlockSlotsInjected

        if self.act_checkpoint and self.training:
            return checkpoint(self.forward_fn, x, use_reentrant=True)  # type: ignore
        return self.forward_fn(x)


class ResnetBlockSlotsInjected(ResnetBlock):
    def __init__(
        self,
        *,
        in_channels: int,
        slot_dim: int,
        time_dim: int,
        out_channels: int | None = None,
        dropout: float = 0.0,
        act_checkpoint: bool = False,
        use_residual_factor: bool = False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            act_checkpoint=act_checkpoint,
            use_residual_factor=use_residual_factor,
        )
        self.in_channels = in_channels
        self.slot_dim = slot_dim
        self.time_dim = time_dim
        self.time_to_hidden = nn.Linear(time_dim, in_channels)
        self.slot_to_hidden = nn.Conv2d(
            slot_dim,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="replicate",
        )
        self.slots_t_to_mod = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels * 2, 3, 1, 1),
        )

        # zero out the last conv for condition
        self.slots_t_to_mod[-1].weight.data.zero_()
        self.slots_t_to_mod[-1].bias.data.zero_()

    def forward_fn(self, x, slots, t):
        # interpolate 2d
        slots = F.interpolate(slots, size=x.shape[-2:], mode="nearest")
        slots = self.slot_to_hidden(slots)
        t = self.time_to_hidden(t)
        slots_t = slots + t[..., None, None]
        scale, shift = self.slots_t_to_mod(slots_t).chunk(2, dim=1)

        # base forward
        h = x
        h = self.norm1(h)
        h = h * (1 + scale) + shift

        h = nonlinearity(h)
        h = self.conv1(h)

        # base forward
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        x = self.nin_shortcut(x)

        if self.use_residual_factor:
            h = h * self.residual_factor

        return x + h

    def forward(
        self, x: torch.Tensor, slots: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        if self.act_checkpoint and self.training:
            return checkpoint(self.forward_fn, x, slots, t, use_reentrant=True)
        return self.forward_fn(x, slots, t)


class AttnBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        act_checkpoint: bool,
        use_residual_factor: bool = False,
        norm_type: str = "gn",
        norm_groups: int = 32,
    ):
        super().__init__()

        self.norm = Normalize(in_channels, norm_type=norm_type, num_groups=norm_groups)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

        self.act_checkpoint = act_checkpoint
        self.use_residual_factor = use_residual_factor
        if self.use_residual_factor:
            self.residual_factor = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def forward_fn(self, x: torch.Tensor) -> torch.Tensor:
        # TODO (freda): Consider reusing implementations in Attn `imaginaire`,
        # since than one is gonna be based on TransformerEngine's attn op,
        # w/c could ease CP implementations.
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)
        if self.use_residual_factor:
            h_ = h_ * self.residual_factor

        return x + h_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_checkpoint:
            return checkpoint(self.forward_fn, x, use_reentrant=True)
        return self.forward_fn(x)


@deprecated(
    "this class does not work with FSDP, please specify the FSDP wrapped module directly"
    "and the accelerator will handle the wrapping automatically"
)
class FSDPNoWarpModule(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.wrap_mod = module

    def forward(self, x):
        return self.wrap_mod(x)


# * --- MoEs 2D --- #


class MoE2DBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        n_experts: int = 4,
        n_selected: int = 1,
        n_shared_experts: int = 1,
        n_token_ec: int = 64 * 64,  # 2d image, 64x64
        moe_type: Literal["tc", "ec", "tc+ec"] = "tc",
        act_checkpoint: bool = False,
    ):
        super().__init__()
        self.moe_type = moe_type
        self.act_checkpoint = act_checkpoint

        moe_tc_fn = partial(
            DeepSeekTCMoE,
            num_experts_per_tok=n_selected,
            n_routed_experts=n_experts,
            hidden_size=in_channels,
            moe_intermediate_size=hidden_channels,
            n_shared_experts=n_shared_experts,
        )
        moe_ec_fn = partial(
            DeepseekECMoE,
            expert_capacity_per_batch=n_token_ec,
            n_routed_experts=n_experts,
            moe_intermediate_size=hidden_channels,
            hidden_size=in_channels,
            n_shared_experts=n_shared_experts,
        )

        if self.moe_type == "tc":  # token choice
            self.moe = nn.ModuleDict({"moe_tc": moe_tc_fn()})
        elif self.moe_type == "ec":  # expert choice
            assert n_token_ec is not None, "n_token_ec should be set for expert choice"
            self.moe = nn.ModuleDict({"moe_ed": moe_ec_fn()})
        elif self.moe_type == "tc+ec":  # token choice + expert choice
            assert n_token_ec is not None, "n_token_ec should be set for expert choice"
            self.moe = nn.ModuleDict({"moe_tc": moe_tc_fn(), "moe_ec": moe_ec_fn()})
        else:
            raise ValueError(f"[MoE2DBlockTC] Unknown moe_type={self.moe_type}")

    # @torch.compile
    def _forward_fn(self, x: torch.Tensor):
        # x: (B, C, H, W)
        h, w = x.shape[-2:]
        x = rearrange(x, "b c h w -> b (h w) c")
        if self.moe_type == "tc":
            x = self.moe["moe_tc"](x)
        elif self.moe_type == "ec":
            x = self.moe["moe_ec"](x)
        else:
            x_ec = self.moe["moe_ec"](x)
            x_tc = self.moe["moe_tc"](x)
            x = x_ec + x_tc

        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        return x

    def forward(self, x):
        x = x.contiguous()
        if self.training and self.act_checkpoint:
            return checkpoint(self._forward_fn, x, use_reentrant=True)
        return self._forward_fn(x)


class DiCoCompactChannelAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.body = nn.Conv2d(in_channels, in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.global_avg(x)
        h = self.body(h)
        h = self.sigmoid(h) * x
        return h


class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act="gelu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Conv2d(self.hidden_size, self.intermediate_size, 1, 1, 0)
        self.up_proj = nn.Conv2d(self.hidden_size, self.intermediate_size, 1, 1, 0)
        self.down_proj = nn.Conv2d(self.intermediate_size, self.hidden_size, 1, 1, 0)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class LlamaFFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.w1 = nn.Conv2d(hidden_size, intermediate_size, 1, 1, 0)
        self.w2 = nn.Conv2d(intermediate_size, hidden_size, 1, 1, 0)
        self.w3 = nn.Conv2d(hidden_size, intermediate_size, 1, 1, 0)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x) * self.w3(x)))


class DiCoBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int | None = None,
        dropout: float = 0.0,
        norm_type: str = "gn",
        norm_groups: int = 32,
        use_residual: bool = False,
        padding_mode: str = "zeros",
        act_checkpoint: bool = False,
        use_ffn: bool = True,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.act_checkpoint = act_checkpoint
        self.use_ffn = use_ffn
        if out_channels is None:
            out_channels = in_channels

        log_print(
            f"[Dico block]: in: {in_channels}"
            f"out: {out_channels}"
            f"hidden: {hidden_channels}",
            "debug",
        )

        self.norm = Normalize(in_channels, norm_type=norm_type, num_groups=norm_groups)

        self.dropout = nn.Dropout(dropout)

        # conv module
        self.body = nn.Sequential(
            # point conv
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            # depth conv
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode=padding_mode,
                groups=out_channels,
            ),
            # nn.Conv2d(
            #     in_channels,
            #     out_channels,
            #     kernel_size=3,
            #     stride=1,
            #     padding=1,
            #     padding_mode=padding_mode,
            # ),
            nn.GELU(approximate="tanh"),
        )

        # cca
        self.cca = DiCoCompactChannelAttention(out_channels)
        self.body_out = nn.Conv2d(
            out_channels, out_channels, kernel_size=1, stride=1, padding=0
        )

        # ffn
        if self.use_ffn:
            self.ffn = FeedForward(out_channels, hidden_channels)
            # self.ffn = LlamaFFN(hidden_size=out_channels, intermediate_size=hidden_channels)
            # self.ffn = nn.Conv2d(
            #     out_channels,
            #     hidden_channels,
            #     kernel_size=3,
            #     stride=1,
            #     padding=1,
            #     padding_mode=padding_mode,
            # )
            self.norm_ffn = Normalize(
                out_channels, norm_type=norm_type, num_groups=norm_groups
            )

        self.nin_shortcut = (
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    @torch.compile
    def forward_fn(self, x: torch.Tensor) -> torch.Tensor:
        h = x

        # token mixer
        h = self.norm(h)
        h = self.body(h)
        h = self.cca(h)
        h = self.body_out(h) + self.nin_shortcut(x)

        # ffn
        if self.use_ffn:
            h1 = h
            h = self.norm_ffn(h)
            h = self.dropout(h)
            h = self.ffn(h) + h1

        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_checkpoint and self.training:
            return checkpoint(self.forward_fn, x, use_reentrant=True)
        return self.forward_fn(x)


class ResnetBlockMoE2D(nn.Module):
    # variant for ResnetBlock
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        drop_out: float = 0.0,
        n_experts: int = 4,
        n_selected: int = 1,
        n_shared_experts: int = 1,
        n_token_ec: int = 64 * 64,  # 2d image, 64x64
        hidden_factor: int = 4,
        moe_type: Literal["tc", "ec", "tc+ec"] = "tc",
        act_checkpoint: bool = False,
        padding_mode: str = "zeros",
        norm_type: str = "gn",
        norm_groups: int = 32,
        token_mixer_type: Literal["resblock", "dico_block"] = "resblock",
        **resnet_kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        log_print(f"[ResnetBlockMoE2D] using token mixer: {token_mixer_type}", "debug")
        if token_mixer_type == "resblock":
            self.token_mixer = ResnetBlock(
                in_channels=in_channels,
                out_channels=self.out_channels,
                dropout=drop_out,
                act_checkpoint=act_checkpoint,
                use_residual_factor=False,
                padding_mode=padding_mode,
                norm_type=norm_type,
                **resnet_kwargs,
            )
        elif token_mixer_type == "dico_block":
            self.token_mixer = DiCoBlock(
                in_channels=in_channels,
                out_channels=self.out_channels,
                hidden_channels=int(hidden_factor * self.out_channels),
                dropout=drop_out,
                act_checkpoint=act_checkpoint,
                use_residual=False,
                padding_mode=padding_mode,
                norm_type=norm_type,
                use_ffn=False,  # moe serves as ffn
                norm_groups=resnet_kwargs.get("num_groups", 32),
            )
        self.moe_prenorm = Normalize(
            self.out_channels, norm_type=norm_type, num_groups=norm_groups
        )
        self.moe = MoE2DBlock(
            in_channels=self.out_channels,
            hidden_channels=int(hidden_factor * self.out_channels),
            act_checkpoint=act_checkpoint,
            n_experts=n_experts,
            n_selected=n_selected,
            n_shared_experts=n_shared_experts,
            n_token_ec=n_token_ec,
            moe_type=moe_type,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # token mixer
        h = self.token_mixer(x)  # shortcut in the block
        # ffn
        h = self.moe(self.moe_prenorm(h)) + h
        return h


# * --- Input and output convs with different bands images --- #


class DiffBandsInputConvIn(nn.Module):
    def __init__(
        self,
        band_lst: list[int],
        hidden_dim: int,
        basic_module: Literal["conv", "resnet", "inv_bottleneck", "moe"] = "conv",
        padding_mode: str = "zeros",
    ):
        super().__init__()

        self.band_lst = band_lst
        self.hidden_dim = hidden_dim

        self.in_modules = nn.ModuleDict()
        for c in band_lst:
            if basic_module == "conv":
                module = nn.Conv2d(
                    in_channels=c,
                    out_channels=hidden_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="replicate",
                )
            elif basic_module == "mobile":
                module = nn.Sequential(
                    nn.Conv2d(c, hidden_dim, 1, 1, 0),
                    nn.Conv2d(
                        hidden_dim,
                        hidden_dim,
                        3,
                        1,
                        1,
                        padding_mode="replicate",
                        groups=hidden_dim,
                    ),
                )
            elif basic_module == "inv_bottleneck":
                module = ResidualBlock(
                    GLUMBConv(
                        in_channels=c,
                        out_channels=hidden_dim,
                        kernel_size=3,
                        stride=1,
                        expand_ratio=4,
                        use_bias=True,
                        norm=(None, None, "rms_triton"),
                        act_func=("silu", "silu", None),
                    ),
                    nn.Conv2d(c, hidden_dim, kernel_size=1, stride=1, padding=0),
                )
            elif basic_module == "resnet":
                module = ResnetBlock(
                    in_channels=c,
                    out_channels=hidden_dim,
                    num_groups=1,
                    dropout=0.0,
                    act_checkpoint=False,
                    padding_mode=padding_mode,
                    norm_type="gn",
                )
            elif basic_module == "moe":
                module = nn.Sequential(
                    nn.Conv2d(
                        in_channels=c,
                        out_channels=hidden_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        padding_mode="replicate",
                    ),
                    MoE2DBlock(
                        in_channels=hidden_dim,
                        hidden_channels=hidden_dim,
                        n_experts=4,
                        n_selected=1,
                        n_shared_experts=1,
                        moe_type="tc",
                        act_checkpoint=False,
                    ),
                    # to patcher dim
                    # nn.Conv2d(c, hidden_dim, kernel_size=1, stride=1, padding=0),
                )
            else:
                raise ValueError(
                    f"[DiffBandsInputConvIn] Unknown basic_module={basic_module}"
                )

            self.in_modules["conv_in_{}".format(c)] = module
            # keep the modules has grad in FSDP, DDP training
            # self.register_buffer(
            #     f"buf_{c}",
            #     torch.zeros(1, c, 3, 3),
            #     persistent=False,
            # )
            log_print(
                f"[DiffBandsInputConvIn] set conv to hidden module and buffer for channel {c}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c_ = x.shape[1]
        module = getattr(self.in_modules, "conv_in_{}".format(c_))
        if module is None:
            raise ValueError(
                f"[DiffBandsInputConvIn] no module for channel {c_}, please check the channel list"
            )
        h = module(x)

        if self.training:
            for c in self.band_lst:
                if c != c_:
                    # buf = getattr(self, f"buf_{c}")
                    buf = torch.zeros(1, c, 3, 3, device=x.device, dtype=x.dtype)
                    m = self.in_modules["conv_in_{}".format(c)]
                    no_use_h = m(buf)
                    h = h + no_use_h.mean() * 0.0

        return h


class DiffBandsInputConvOut(nn.Module):
    def __init__(
        self,
        band_lst: list[int],
        hidden_dim: int,
        basic_module: str = "conv",
    ):
        super().__init__()

        self.band_lst = band_lst
        self.hidden_dim = hidden_dim
        self.basic_module = basic_module

        self.in_modules = nn.ModuleDict()
        for c in band_lst:
            if basic_module == "conv":
                module = nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=c,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="replicate",
                )
            elif basic_module == "mobile":
                module = nn.Sequential(
                    nn.Conv2d(
                        hidden_dim,
                        hidden_dim,
                        3,
                        1,
                        1,
                        padding_mode="replicate",
                        groups=hidden_dim,
                    ),
                    nn.Conv2d(hidden_dim, c, 1, 1, 0),
                )
            elif basic_module == "inv_bottleneck":
                module = ResidualBlock(
                    GLUMBConv(
                        in_channels=hidden_dim,
                        out_channels=c,
                        kernel_size=3,
                        stride=1,
                        expand_ratio=4,
                        use_bias=True,
                        norm=(None, None, "rms_triton"),
                        act_func=("silu", "silu", None),
                    ),
                    nn.Conv2d(hidden_dim, c, kernel_size=1, stride=1, padding=0),
                )
            elif basic_module == "resnet":
                module = ResnetBlock(
                    in_channels=hidden_dim,
                    out_channels=c,
                    dropout=0.0,
                    act_checkpoint=False,
                    num_groups=1,
                    norm_type="gn",
                )
            elif basic_module == "moe":
                module = nn.Sequential(
                    nn.Conv2d(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        padding_mode="replicate",
                    ),
                    MoE2DBlock(
                        in_channels=hidden_dim,
                        hidden_channels=hidden_dim,
                        n_experts=4,
                        n_selected=1,
                        n_shared_experts=1,
                        moe_type="tc",
                        act_checkpoint=False,
                    ),
                    # to patcher dim
                    nn.Conv2d(hidden_dim, c, kernel_size=1, stride=1, padding=0),
                )
            else:
                raise ValueError(
                    f"[DiffBandsInputConvIn] Unknown basic_module={basic_module}"
                )

            self.in_modules["conv_out_{}".format(c)] = module

            # keep the modules has grad in FSDP, DDP training
            log_print(
                f"[DiffBandsInputConvOut] set conv to hidden module for channel {c}"
            )

        # self.register_buffer(
        #     f"buf_out",
        #     torch.zeros(1, 256, 3, 3),
        #     persistent=False,
        # )

        self.out_channel = None

    def forward(self, x: torch.Tensor, out_channel: int) -> torch.Tensor:
        self.out_channel = out_channel

        module = getattr(self.in_modules, f"conv_out_{out_channel}", None)
        if module is None:
            raise ValueError(
                f"[DiffBandsInputConvOut] No module for out_channel={out_channel}. Available: {list(self.in_modules.keys())}",
            )
        h = module(x)

        if self.training:
            for c in self.band_lst:
                if c != out_channel:
                    # buf = self.buf_out
                    buf = torch.zeros(1, 256, 3, 3, device=x.device, dtype=x.dtype)
                    m = self.in_modules["conv_out_{}".format(c)]
                    no_use_h = m(buf)
                    h = h + no_use_h.mean() * 0.0

        return h

    @property
    def weight(self):
        # used to get the weight of the conv_out module for GAN loss
        assert self.out_channel is not None, (
            "out_channel is not set, please call forward first"
        )
        module = getattr(self.in_modules, f"conv_out_{self.out_channel}", None)
        if module is None:
            raise ValueError(
                f"[DiffBandsInputConvOut] No module for out_channel={self.out_channel}. Available: {list(self.in_modules.keys())}",
            )

        match self.basic_module:
            case "conv":
                return module.weight
            case "resnet":
                return module.conv2.weight
            case "moe":
                return module[-1].weight
            case "inv_bottleneck":
                return module.main.depth_conv.conv.weight
            case "mobile":
                return module[-1].weight
            case _:
                raise ValueError(
                    f"[DiffBandsInputConvOut] Unknown basic_module={self.basic_module}. Available: conv, resnet, moe, inv_bottleneck"
                )


# * --- Encoder and decoder --- #


def make_block_fn(
    block_name: str = "resblock",
    moe_n_experts=4,
    act_checkpoint=False,
    use_residual_factor=False,
    moe_n_selected=1,
    moe_n_shared_experts=1,
    hidden_factor=2,
    moe_type="tc",
    padding_mode: str = "zeros",
    norm_type: str = "gn",
    token_mixer_type: Literal["resblock", "dico_block"] = "resblock",
    **kwargs,
):
    if block_name == "res_moe":

        def block_fn(block_in, block_out, dropout, curr_res):
            return ResnetBlockMoE2D(
                in_channels=block_in,
                out_channels=block_out,
                drop_out=dropout,
                n_experts=moe_n_experts,
                n_selected=moe_n_selected,
                n_shared_experts=moe_n_shared_experts,
                n_token_ec=int(curr_res * curr_res / 2),
                hidden_factor=hidden_factor,
                moe_type=moe_type,
                act_checkpoint=act_checkpoint,
                padding_mode=padding_mode,
                norm_type=norm_type,
                token_mixer_type=token_mixer_type,
                **kwargs,
            )

    elif block_name == "dico_block":

        def block_fn(block_in, block_out, dropout, curr_res):
            return DiCoBlock(
                in_channels=block_in,
                hidden_channels=block_out * hidden_factor,
                out_channels=block_out,
                dropout=dropout,
                norm_type=norm_type,
                norm_groups=kwargs.get("num_groups", 32),
                use_residual=True,
                padding_mode=padding_mode,
                act_checkpoint=act_checkpoint,
            )

    elif block_name == "res_block":

        def block_fn(block_in, block_out, dropout, curr_res):
            return ResnetBlock(
                in_channels=block_in,
                out_channels=block_out,
                dropout=dropout,
                act_checkpoint=act_checkpoint,
                use_residual_factor=use_residual_factor,
                padding_mode=padding_mode,
                norm_type=norm_type,
                **kwargs,
            )

    else:
        raise ValueError(
            f"block_name {block_name} is not supported. Supported: 'res_block', 'res_moe', 'dico_block'"
        )

    return block_fn


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int | list[int],
        channels: int,
        channels_mult: list[int],
        num_res_blocks: int,
        attn_resolutions: list[int],
        dropout: float,
        resolution: int,
        z_channels: int,
        spatial_compression: int,
        act_checkpoint: bool = False,
        use_residual_factor: bool = False,
        downsample_type: str = "PadConv",  # used in original cosmos tokenizer
        downsample_shortcut: str | None = None,
        force_not_attn: bool = False,
        patch_size: int = 4,
        patch_method: str = "haar",
        conv_in_module: Literal["conv", "resnet", "inv_bottleneck", "moe"] = "conv",
        block_name: Literal["res_block", "dico_block", "res_moe"] = "res_block",
        # if block_name != 'moe', does not use
        moe_n_experts: int = 4,
        moe_n_selected: int = 1,
        moe_n_shared_experts: int = 1,
        hidden_factor: int = 2,
        moe_type: Literal["tc", "ec", "tc+ec"] = "tc",
        moe_token_mixer_type: Literal["resblock", "dico_block"] = "resblock",
        # padding and norm
        padding_mode: str = "zeros",
        norm_type: str = "gn",
        norm_groups: int = 32,
        **ignore_kwargs,
    ):
        super().__init__()
        self.num_resolutions = len(channels_mult)
        self.num_res_blocks = num_res_blocks
        self.moe_n_experts = moe_n_experts
        self.moe_n_selected = moe_n_selected
        self.moe_n_shared_experts = moe_n_shared_experts
        self.hidden_factor = hidden_factor
        self.moe_type = moe_type

        log_print(
            f"[Encoder]: padding mode: {padding_mode}, norm type: {norm_type}, norm groups: {norm_groups}, "
            f"use activation checkpoint: {act_checkpoint}"
        )
        log_print(f"[Encoder]: z_channels: {z_channels}, patch size: {patch_size}")
        log_print(f"[Encoder]: Using block name: {block_name}")

        # Patcher.
        self.patcher = Patcher(patch_size, patch_method)
        log_print(
            f"[Encoder]: in_channels: {in_channels}, patch_size: {patch_size}, "
            f"patch_method: {patch_method}"
        )

        # calculate the number of downsample operations
        self.num_downsamples = int(math.log2(spatial_compression)) - int(
            math.log2(patch_size)
        )
        assert self.num_downsamples <= self.num_resolutions, (
            f"we can only downsample {self.num_resolutions} times at most"
        )

        # input conv
        self.use_diffbands_input = isinstance(in_channels, Sequence)
        if self.use_diffbands_input:
            assert isinstance(in_channels, list)
            in_channels = [c * patch_size * patch_size for c in in_channels]
            self.conv_in = DiffBandsInputConvIn(
                band_lst=in_channels,
                hidden_dim=channels,
                basic_module=conv_in_module,
                padding_mode=padding_mode,
            )
        else:
            in_channels = in_channels * patch_size * patch_size
            self.conv_in = torch.nn.Conv2d(
                in_channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode=padding_mode,
            )

        # downsampling
        curr_res = resolution // patch_size
        in_ch_mult = (1,) + tuple(channels_mult)
        self.in_ch_mult = in_ch_mult

        block_fn = make_block_fn(
            block_name=block_name,
            moe_n_experts=self.moe_n_experts,
            moe_n_selected=self.moe_n_selected,
            moe_n_shared_experts=self.moe_n_shared_experts,
            hidden_factor=self.hidden_factor,
            moe_type=self.moe_type,
            act_checkpoint=act_checkpoint,
            use_residual_factor=use_residual_factor,
            padding_mode=padding_mode,
            norm_type=norm_type,
            num_groups=norm_groups,
            token_mixer_type=moe_token_mixer_type,
        )
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = channels * in_ch_mult[i_level]
            block_out = channels * channels_mult[i_level]
            for _ in range(self.num_res_blocks):
                res_block = block_fn(block_in, block_out, dropout, curr_res)
                block.append(res_block)
                block_in = block_out
                if curr_res in attn_resolutions and not force_not_attn:
                    log_print(f"[Encoder]: use attn at {curr_res}")
                    attn.append(
                        AttnBlock(
                            block_in,
                            act_checkpoint=act_checkpoint,
                            use_residual_factor=use_residual_factor,
                            norm_groups=norm_groups,
                        )
                    )
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level < self.num_downsamples:
                down.downsample = build_downsample_block(
                    downsample_type, block_in, block_in, shortcut=downsample_shortcut
                )
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = block_fn(block_in, block_out, dropout, curr_res)
        self.mid.attn_1 = (
            AttnBlock(
                block_in,
                act_checkpoint=act_checkpoint,
                use_residual_factor=use_residual_factor,
                norm_type=norm_type,
                norm_groups=norm_groups,
            )
            if not force_not_attn
            else nn.Identity()
        )
        self.mid.block_2 = block_fn(block_in, block_out, dropout, curr_res)
        # end
        self.norm_out = Normalize(block_in, norm_type=norm_type, num_groups=norm_groups)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode=padding_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patcher(x)

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level < self.num_downsamples:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels: int | list[int],
        channels: int,
        channels_mult: list[int],
        num_res_blocks: int,
        attn_resolutions: list[int],
        dropout: float,
        resolution: int,
        z_channels: int,
        spatial_compression: int,
        act_checkpoint: bool = False,
        use_residual_factor: bool = False,
        upsample_type: str = "RepeatConv",
        upsample_shortcut: str | None = None,
        conv_out_module: Literal["conv", "resnet", "inv_bottleneck", "moe"] = "conv",
        force_not_attn: bool = False,
        block_name: Literal["res_block", "dico_block", "res_moe"] = "res_block",
        moe_n_experts: int = 4,
        moe_n_selected: int = 1,
        moe_n_shared_experts: int = 1,
        hidden_factor: int = 2,
        moe_type: Literal["tc", "ec", "tc+ec"] = "tc",
        moe_token_mixer_type: Literal["resblock", "dico_block"] = "resblock",
        padding_mode: str = "zeros",
        norm_type: str = "gn",
        norm_groups: int = 32,
        **ignore_kwargs,
    ):
        super().__init__()
        self.num_resolutions = len(channels_mult)
        self.num_res_blocks = num_res_blocks
        self.moe_n_experts = moe_n_experts
        self.moe_n_selected = moe_n_selected
        self.moe_n_shared_experts = moe_n_shared_experts
        self.hidden_factor = hidden_factor
        self.moe_type = moe_type

        log_print(
            f"[Decoder]: padding mode: {padding_mode}, norm type: {norm_type}, norm_groups: {norm_groups}, "
            f"use activation checkpoint: {act_checkpoint}"
        )
        log_print(f"[Decoder]: z_channels: {z_channels}")
        log_print(f"[Decoder]: Using block type: {block_name} ")

        # UnPatcher.
        self.patch_size = patch_size = ignore_kwargs.get("patch_size", 1)
        self.unpatcher = UnPatcher(
            patch_size, ignore_kwargs.get("patch_method", "rearrange")
        )

        # calculate the number of upsample operations
        self.num_upsamples = int(math.log2(spatial_compression)) - int(
            math.log2(patch_size)
        )
        assert self.num_upsamples <= self.num_resolutions, (
            f"we can only upsample {self.num_resolutions} times at most"
        )

        block_in = channels * channels_mult[self.num_resolutions - 1]
        curr_res = (resolution // patch_size) // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        log_print(
            "Working with z of shape {} = {} dimensions.".format(
                self.z_shape, np.prod(self.z_shape)
            )
        )

        # z to block_in
        self.conv_in = torch.nn.Conv2d(
            z_channels,
            block_in,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode=padding_mode,
        )

        # block fn
        block_fn = make_block_fn(
            block_name=block_name,
            moe_n_experts=self.moe_n_experts,
            moe_n_selected=self.moe_n_selected,
            moe_n_shared_experts=self.moe_n_shared_experts,
            hidden_factor=self.hidden_factor,
            moe_type=self.moe_type,
            act_checkpoint=act_checkpoint,
            use_residual_factor=use_residual_factor,
            padding_mode=padding_mode,
            norm_type=norm_type,
            num_groups=norm_groups,
            token_mixer_type=moe_token_mixer_type,
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = block_fn(block_in, block_in, dropout, curr_res)
        self.mid.attn_1 = (
            AttnBlock(block_in, act_checkpoint=act_checkpoint, norm_groups=norm_groups)
            if not force_not_attn
            else nn.Identity()
        )
        self.mid.block_2 = block_fn(block_in, block_in, dropout, curr_res)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = channels * channels_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(block_fn(block_in, block_out, dropout, curr_res))
                block_in = block_out
                if curr_res in attn_resolutions and not force_not_attn:
                    log_print(f"[Decoder]: use attn at {curr_res}")
                    attn.append(
                        AttnBlock(
                            block_in,
                            act_checkpoint=act_checkpoint,
                            use_residual_factor=use_residual_factor,
                            norm_type=norm_type,
                            norm_groups=norm_groups,
                        )
                    )
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level >= (self.num_resolutions - self.num_upsamples):
                up.upsample = build_upsample_block(
                    upsample_type,
                    block_in,
                    block_out,
                    shortcut=upsample_shortcut,
                    padding_mode=padding_mode,
                )
                curr_res = curr_res * 2
            self.up.insert(0, up)

        # end
        self.norm_out = Normalize(block_in, norm_type=norm_type, num_groups=norm_groups)

        self.use_diffbands_input = isinstance(out_channels, list)
        if self.use_diffbands_input:
            log_print("[Decoder]: use diffbands input")
            out_ch = [c * patch_size * patch_size for c in out_channels]
            conv_out = DiffBandsInputConvOut(
                band_lst=out_ch, hidden_dim=block_in, basic_module=conv_out_module
            )
        else:
            out_ch = out_channels * patch_size * patch_size
            conv_out = torch.nn.Conv2d(
                block_in,
                out_ch,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode=padding_mode,
            )

        # fsdp warpper, but not used
        _wrap_fsdp_last_layer = ignore_kwargs.get("wrap_fsdp_last_layer", False)
        self._wrap_fsdp_last_layer = _wrap_fsdp_last_layer
        if _wrap_fsdp_last_layer:
            self.conv_out = FSDPNoWarpModule(conv_out)
            log_print("[Decoder] use FSDPNoWarpModule")
        else:
            self.conv_out = conv_out

    def forward(self, z: torch.Tensor, out_channels: int | None = None) -> torch.Tensor:
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        # decoder.up.0.block.2.conv1.weight
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level >= (self.num_resolutions - self.num_upsamples):
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        if not self.use_diffbands_input:
            conv_out_h = (h,)
        else:
            assert out_channels is not None, "out_channels should be provided"
            conv_out_h = (h, out_channels * self.patch_size * self.patch_size)
        h = self.conv_out(*conv_out_h)
        h = self.unpatcher(h)
        return h

    def forward_with_cfg(self, x, t, z, y=None, cfg_scale=1.0):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, z)
        eps, rest = model_out[:, : self.out_channels], model_out[:, self.out_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def get_last_layer(self):
        if not self._wrap_fsdp_last_layer:
            return self.conv_out.weight
        else:
            return self.conv_out.wrap_mod.weight


# *==============================================================
# * Decoder Diffusion version
# *==============================================================


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(
        self, hidden_size, frequency_embedding_size=256, time_scale: float = 1.0
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.register_buffer(
            "time_scale",
            torch.tensor(time_scale),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(
            t * self.time_scale, self.frequency_embedding_size
        )
        t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        return t_emb


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(
        self,
        hidden_size,
        patch_size,
        out_channels,
        z_channels,
        t_channels,
    ):
        super().__init__()
        self.norm_final = Normalize(hidden_size)
        self.conv = nn.Conv2d(
            hidden_size, patch_size * patch_size * out_channels, 3, 1, 1
        )
        self.t_embd = nn.Linear(t_channels, hidden_size, bias=True)
        self.z_embd = nn.Conv2d(z_channels, hidden_size, 1, 1)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Conv2d(hidden_size, 2 * hidden_size, 1, 1)
        )

    def forward(self, x, z, t):
        c = (
            self.z_embd(F.interpolate(z, size=x.shape[-2:], mode="nearest"))
            + self.t_embd(t)[..., None, None]
        )
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x) * (1 + scale) + shift
        x = self.conv(x)
        return x


class DecoderDiff(nn.Module):
    def __init__(
        self,
        out_channels: int,  # rgb or hyperspectral image channels
        channels: int,
        channels_mult: list[int],
        num_res_blocks: int,
        attn_resolutions: int,
        dropout: float,
        resolution: int,
        z_channels: int,
        spatial_compression: int,
        act_checkpoint: bool = False,
        z_cfg_drop: float = 0.1,
        learn_sigma: bool = False,
        diff_cond_inject_strategy: Literal[
            "cat", "inject_part", "inject_full"
        ] = "inject_part",
        decoder_patch_size: int = 4,
        patch_method: str = "rearrange",
        unpatch_type: Literal["upsample", "unpatch"] = "unpatch",
        time_scale: float = 1.0,
        use_residual_factor: bool = False,
        **_discard_kwargs,
    ):
        super().__init__()
        self.num_resolutions = len(channels_mult)
        self.num_res_blocks = num_res_blocks
        self.learn_sigma = learn_sigma
        self.diff_cond_inject_strategy = diff_cond_inject_strategy
        self.unpatch_type = unpatch_type
        self.z_cfg_drop = z_cfg_drop
        self.out_channels = out_channels

        log_print(
            f"[Decoder]: use activation checkpoint: {act_checkpoint}\n"
            f"diffusion conditioning inject strategy: {diff_cond_inject_strategy}\n"
            f"use_residual_factor: {use_residual_factor}",
        )

        # Patcher
        self.patcher = Patcher(
            decoder_patch_size,
            patch_method,
        )
        conv_in_ch = out_channels * decoder_patch_size * decoder_patch_size

        # z embedding
        self.z_embedding = nn.Conv2d(z_channels, conv_in_ch, 1, 1, 0)

        z_res = resolution // spatial_compression
        self.z_shape = (1, z_channels, z_res, z_res)
        curr_res = resolution // decoder_patch_size
        log_print(
            "Working with z of shape {} = {} dimensions.".format(
                self.z_shape, np.prod(self.z_shape)
            )
        )

        # UnPatcher.
        if self.unpatch_type == "unpatch":
            self.unpatcher = UnPatcher(
                decoder_patch_size,
                patch_method,
            )
            # out_channels * decoder_patch_size * decoder_patch_size
            self.num_upsamples = 0
        elif self.unpatch_type == "upsample":
            log_print(
                f"[Decoder Unpatcher]: unpatcher is set to use upsample, may cause GPU OOM",
                "warning",
            )
            unpatcher_sz = 1
            # use upsample to unpatch
            self.unpatcher = UnPatcher(
                unpatcher_sz,
                patch_method,
            )
            # out_channels * unpatcher_sz * unpatcher_sz

            # calculate the number of upsample operations
            self.num_upsamples = int(math.log2(decoder_patch_size))
            assert self.num_upsamples <= self.num_resolutions, (
                f"we can only upsample {self.num_resolutions} times at most"
            )
        else:
            raise NotImplementedError("unpatcher_type must be 'upsample' or 'unpatch'")

        block_in = channels * channels_mult[self.num_resolutions - 1]
        t_in = block_in

        # z to block_in
        cat_x_z_in = conv_in_ch * 2
        self.conv_in = torch.nn.Conv2d(
            cat_x_z_in, block_in, kernel_size=3, stride=1, padding=1
        )

        # timestep embedder
        self.t_embedder = TimestepEmbedder(t_in, time_scale=time_scale)  # base channels

        # block class
        if diff_cond_inject_strategy in ("inject_part", "inject_full"):
            block_class = partial(
                ResnetBlockSlotsInjected,
                slot_dim=conv_in_ch,
                time_dim=t_in,
                use_residual_factor=use_residual_factor,
            )
        elif diff_cond_inject_strategy == "cat":
            block_class = partial(
                ResnetBlock,
                use_residual_factor=use_residual_factor,
            )
        else:
            raise ValueError(
                "diff_cond_inject_strategy must be either 'inject' or 'cat'"
            )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = block_class(
            in_channels=block_in,
            out_channels=t_in,
            dropout=dropout,
            act_checkpoint=act_checkpoint,
        )
        # self.mid.attn_1 = AttnBlock(block_in, act_checkpoint=act_checkpoint)
        if diff_cond_inject_strategy == "inject_part":
            mid_blk_2_cls = ResnetBlock
        elif diff_cond_inject_strategy == "inject_full":
            mid_blk_2_cls = block_class
            log_print(
                f"[Decoder Block]: diffusion condition injection strategy is {diff_cond_inject_strategy}, "
                "it will inject condition in every residual block, may cause GPU usage high",
                "warning",
            )
        else:
            mid_blk_2_cls = block_class
        self.mid.block_2 = mid_blk_2_cls(  # do not use cond
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            act_checkpoint=act_checkpoint,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):  # 2, 1, 0
            block = nn.ModuleList()
            # attn = nn.ModuleList()
            block_out = channels * channels_mult[i_level]
            for blk_level in range(self.num_res_blocks + 1):
                if diff_cond_inject_strategy == "inject_part":
                    inner_class = (
                        block_class
                        if blk_level == 0
                        else partial(
                            ResnetBlock,
                            use_residual_factor=use_residual_factor,
                        )
                    )
                else:
                    inner_class = block_class
                block.append(
                    inner_class(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        act_checkpoint=act_checkpoint,
                    )
                )
                block_in = block_out
                # if curr_res in attn_resolutions:
                #     attn.append(AttnBlock(block_in, act_checkpoint=act_checkpoint))
            up = nn.Module()
            up.block = block
            # up.attn = attn
            if i_level >= (
                self.num_resolutions - self.num_upsamples
            ):  # [2,1] upsample, 1
                # last layers upsample
                up.upsample = UpsampleRepeatConv(block_in)
                log_print(f"upsample {i_level}")
                curr_res = curr_res * 2

            # [128*4*4*2, 128*4, 128*2]
            self.up.insert(0, up)

        # end
        # self.norm_out = Normalize(block_in)
        out_mul = 2.0 if learn_sigma else 1.0
        # self.conv_out = torch.nn.Conv2d(
        #     block_in, int(out_ch * out_mul), kernel_size=3, stride=1, padding=1
        # )
        self.final_layer = FinalLayer(
            block_in,
            decoder_patch_size,
            int(out_channels * out_mul),
            conv_in_ch,
            t_in,
        )
        # zero-out the final layer
        self.final_layer.conv.weight.data.zero_()
        self.final_layer.conv.bias.data.zero_()

        # cfg null condition
        self.null_cond = nn.Parameter(torch.zeros(1, z_channels, z_res, z_res))
        torch.nn.init.normal_(self.null_cond, std=0.02)

    def embed_cond(self, z):
        bs = z.shape[0]
        null_cond_interp = self.null_cond
        if null_cond_interp.shape[-2] != z.shape[-2]:
            if self.training:
                raise ValueError(
                    f"null_cond_interp.shape[-2] ({null_cond_interp.shape[-2:]})!= z.shape[-2] ({z.shape[-2:]})"
                )

            null_cond_interp = F.interpolate(
                null_cond_interp, size=z.shape[-2], mode="bicubic"
            )

        if self.training:
            drop_ids = torch.rand(bs, 1, 1, 1).to(z) < self.z_cfg_drop
            z = torch.where(
                drop_ids,
                null_cond_interp.repeat(bs, 1, 1, 1),
                z,
            )

        return self.z_embedding(z)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        x = self.patcher(x)
        z = self.embed_cond(z)
        h = torch.cat(
            [x, F.interpolate(z, size=x.shape[-2:], mode="nearest")],
            dim=1,
        )

        h = self.conv_in(h)
        t = self.t_embedder(t)

        # middle
        h = self.mid.block_1(h, z, t)
        # h = self.mid.attn_1(h)
        h = self.mid.block_2(h, z, t)

        # upsampling
        # decoder.up.0.block.2.conv1.weight
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, z, t)
                # if len(self.up[i_level].attn) > 0:
                #     h = self.up[i_level].attn[i_block](h)
            if i_level >= (self.num_resolutions - self.num_upsamples):
                h = self.up[i_level].upsample(h)

        # h = self.norm_out(h)
        # h = nonlinearity(h)
        h = self.final_layer(h, z, t)
        h = self.unpatcher(h)
        return h

    def forward_with_cfg(self, x, t, z, y=None, cfg_scale=1.0):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, z)
        eps, rest = model_out[:, : self.out_channels], model_out[:, self.out_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def get_last_layer(self):
        return self.final_layer.conv.weight


if __name__ == "__main__":
    import time

    from tqdm import trange

    def func_mem_wrapper(func):
        def wrapper(*args, **kwargs):
            # 记录初始显存占用
            torch.cuda.reset_peak_memory_stats()  # reset the peak memory stats
            initial_memory = torch.cuda.memory_allocated()

            ret = func(*args, **kwargs)

            # 执行 tokenizer 并记录显存占用
            allocated_memory = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()

            # 计算显存增量
            memory_usage = allocated_memory - initial_memory

            # 打印显存占用信息
            print(f"Initial memory allocated: {initial_memory / 1024**2:.2f} MB")
            print(
                f"Memory allocated after forward pass: {allocated_memory / 1024**2:.2f} MB"
            )
            print(f"Peak memory allocated: {peak_memory / 1024**2:.2f} MB")
            print(f"Memory usage: {memory_usage / 1024**2:.2f} MB")

            print(torch.cuda.memory_summary(torch.cuda.current_device()))

            return ret

        return wrapper

    def func_speed_wrapper(test_num=100):
        def inner_func_wrapper(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()

                for _ in trange(test_num):
                    ret = func(*args, **kwargs)

                end_time = time.time()
                total_time = end_time - start_time
                average_time = total_time / test_num

                print(f"Function {func.__name__} executed {test_num} times.")
                print(f"Total time: {total_time:.4f} seconds")
                print(f"Average time per execution: {average_time:.4f} seconds")

                return ret

            return wrapper

        return inner_func_wrapper

    @func_mem_wrapper
    def test_diff_enc_dec():
        encoder = Encoder(
            8,
            128,
            [2, 2, 4],
            2,
            [32],
            0.1,
            512,
            16,
            16,
            True,
            patch_size=4,
        )
        decoder = DecoderDiff(
            8,
            128,
            [2, 2, 4],
            2,
            [32],
            0.1,
            512,
            16,
            16,
            True,
            diff_cond_inject_strategy="inject_full",
            decoder_patch_size=4,
            z_cfg_drop=0.3,
            unpatch_type="unpatch",
        )
        dtype = torch.bfloat16
        device = torch.device("cuda:1")
        encoder = encoder.to(device, dtype)
        decoder = decoder.to(device, dtype)

        bs = 1
        img = torch.randn(bs, 8, 512, 512).to(device, dtype)
        xt = torch.randn(bs, 8, 512, 512).to(device, dtype)
        t = torch.randint(0, 1000, (bs,)).to(device, dtype)

        slots = encoder(img)
        recon = decoder(xt, t, slots)

        opt = torch.optim.AdamW(
            [*encoder.parameters(), *decoder.parameters()],
            lr=1e-3,
        )

        print(recon.shape)

        opt.zero_grad()
        recon.mean().backward()

        enc_n = 0
        dec_n = 0

        for n, p in encoder.named_parameters():
            if p.grad is None:
                print(f"{n} has no grad")
            enc_n += p.numel()

        for n, p in decoder.named_parameters():
            if p.grad is None:
                print(f"{n} has no grad")
            dec_n += p.numel()

        print(f"encoder params: {enc_n / 1e6}, dec params: {dec_n / 1e6}")

        opt.step()

    @func_mem_wrapper
    def test_auto_enc_dec():
        img_size = 256
        # 1024/4=256
        # 256/2=128
        encoder = Encoder(
            8,
            128,
            [2, 2, 4],
            2,
            [32],
            0.1,
            img_size,
            16,
            8,
            act_checkpoint=False,
            patch_size=4,
            # downsample_type="ConvPixelUnshuffle",
            # downsample_shortcut="averaging",
            block_name="dico_block",
            moe_token_mixer_type="dico_block",
            force_not_attn=True,
            norm_type="gn",
            hidden_factor=3,
        )
        decoder = Decoder(
            8,
            128,
            [2, 2, 4],
            2,
            [32],
            0.1,
            img_size,
            16,
            8,
            act_checkpoint=False,
            patch_size=4,
            # upsample_type="ConvPixelShuffle",
            # upsample_shortcut="duplicating",
            block_name="res_moe",
            force_not_attn=True,
            norm_type="gn",
            moe_token_mixer_type="dico_block",
            hidden_factor=3,
        )
        dtype = torch.bfloat16
        device = torch.device("cuda:1")
        encoder = encoder.to(device, dtype)
        decoder = decoder.to(device, dtype)

        bs = 16
        img = torch.randn(bs, 8, *[img_size] * 2).to(device, dtype)
        opt = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=1e-3)
        opt.zero_grad()

        slots = encoder(img)
        recon = decoder(slots)

        print(recon.shape)

        recon.mean().backward()
        opt.step()

        enc_n = 0
        dec_n = 0

        for n, p in encoder.named_parameters():
            if p.grad is None:
                print(f"{n} has no grad")
            enc_n += p.numel()

        for n, p in decoder.named_parameters():
            if p.grad is None:
                print(f"{n} has no grad")
            dec_n += p.numel()

        print(f"encoder params: {enc_n / 1e6}, dec params: {dec_n / 1e6}")
        import time

        time.sleep(10)

    @func_mem_wrapper
    def test_multi_bands_enc_dec(optimize=False):
        import accelerate

        accelerator = accelerate.Accelerator(mixed_precision="bf16")
        device = accelerator.device

        img_size = 512
        # 1024/4=256
        # 256/2=128
        encoder = Encoder(
            [4, 8, 16, 24],
            128,
            [2, 2, 4],
            2,
            [32],
            0.1,
            img_size,
            16,
            8,
            patch_size=4,
            act_checkpoint=True,
            force_not_attn=True,
            norm_type="gn",
            block_name="dico_block",
            hidden_factor=2,
        )
        decoder = Decoder(
            [4, 8, 16, 24],
            128,
            [2, 2, 4],
            2,
            [32],
            0.1,
            img_size,
            16,
            8,
            act_checkpoint=True,
            patch_size=4,
            moe_type="tc+ec",
            force_not_attn=True,
            norm_type="gn",
            block_name="dico_block",
            hidden_factor=2,
        )
        dtype = torch.bfloat16
        encoder = encoder.to(device, dtype)
        decoder = decoder.to(device, dtype)

        from fvcore.nn import parameter_count_table

        log_print(parameter_count_table(encoder))
        log_print(parameter_count_table(decoder))

        bs = 8
        img = torch.randn(bs, 8, *[img_size] * 2).to(device, dtype)

        from contextlib import nullcontext

        if optimize:
            optimizer = torch.optim.AdamW(
                [*encoder.parameters(), *decoder.parameters()],
                lr=1e-3,
            )
            optimizer.zero_grad()
            context = nullcontext
        else:
            context = nullcontext  # torch.no_grad

        with accelerator.autocast():
            with context():
                slots = encoder(img)
                recon = decoder(slots, img.shape[1])

        print(recon.shape)

        if optimize:
            loss = recon.mean()
            accelerator.backward(loss)
            optimizer.step()

        accelerator.backward(recon.mean())
        p_norms = {}
        for n, p in encoder.named_parameters():
            p_norms[n] = p.norm().item()

        # sort norms
        p_norms = {
            k: v
            for k, v in sorted(p_norms.items(), key=lambda item: item[1], reverse=True)
        }
        largest_n = 300
        # print the largest n norms
        for k, v in p_norms.items():
            print(f"{k}: {v:.4f}")
            if largest_n > 0:
                largest_n -= 1
            else:
                break

        # enc_n = 0
        # dec_n = 0

        # for n, p in encoder.named_parameters():
        #     if p.grad is None:
        #         print(f"{n} has no grad")
        #     enc_n += p.numel()

        # for n, p in decoder.named_parameters():
        #     if p.grad is None:
        #         print(f"{n} has no grad")
        #     dec_n += p.numel()
        # print(f"encoder params: {enc_n / 1e6}, dec params: {dec_n / 1e6}")

        import time

        time.sleep(10)

    def test_moe_layer():
        moe_block = MoE2DBlock(32, 64, 4, 1, 1, 32 * 32, "tc+ec").cuda()
        x = torch.randn(1, 32, 128, 128).cuda()
        print("input shape: ", x.shape)
        out = moe_block(x)
        print("output shape: ", out.shape)

    test_auto_enc_dec()
    # test_diff_enc_dec()
    # test_multi_bands_enc_dec()
    # test_moe_layer()

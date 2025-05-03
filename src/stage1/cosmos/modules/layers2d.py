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
from typing import Callable, Literal, Optional, Union

import numpy as np

# pytorch_diffusion + derived encoder decoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer")
from src.stage1.cosmos.modules.patching import Patcher, UnPatcher
from src.stage1.cosmos.modules.utils import Normalize, nonlinearity
from src.utilities.logging import log_print

# * ==========================================================
# * Norm


class RMSNorm2d(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)[None, :, None, None])

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# * ==========================================================
# * Upsample and downsamples


class UpsampleRepeatConv(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
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
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=2, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        return self.conv(x)


class ConvPixelUnshuffleDownSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
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
    ):
        super().__init__()
        self.factor = factor
        out_ratio = factor**2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * out_ratio,
            kernel_size,
            padding=get_same_padding(kernel_size),
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


# * ==========================================================
# * Upsample and downsample entries


def build_upsample_block(
    block_type: str, in_channels: int, out_channels: int, shortcut: Optional[str]
) -> nn.Module:
    if block_type == "ConvPixelShuffle":
        block = ConvPixelShuffleUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, factor=2
        )
    elif block_type == "RepeatConv":
        block = UpsampleRepeatConv(in_channels)
    elif block_type == "InterpolateConv":
        block = InterpolateConvUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, factor=2
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
    block_type: str, in_channels: int, out_channels: int, shortcut: Optional[str]
) -> nn.Module:
    if block_type == "Conv":
        block = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
    elif block_type == "PadConv":
        block = DownsamplePadConv(in_channels=in_channels)
    elif block_type == "ConvPixelUnshuffle":
        block = ConvPixelUnshuffleDownSampleLayer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, factor=2
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


# * ==========================================================
# * Blocks


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int = None,
        dropout: float,
        use_residual_factor: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.nin_shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
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
            return checkpoint(self.forward_fn, x, use_reentrant=True)
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
            slot_dim, in_channels, kernel_size=3, stride=1, padding=1
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
        self, in_channels: int, act_checkpoint: bool, use_residual_factor: bool = False
    ):
        super().__init__()

        self.norm = Normalize(in_channels)
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


class FSDPNoWarpModule(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.wrap_mod = module

    def forward(self, x):
        return self.wrap_mod(x)


# * ==========================================================
# * Encoder and decoder


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
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
        downsample_type: str = "PadConv",
        downsample_shortcut: str | None = None,
        force_not_attn: bool = False,
        patch_size: int = 4,
        patch_method: str = "haar",
        **ignore_kwargs,
    ):
        super().__init__()
        self.num_resolutions = len(channels_mult)
        self.num_res_blocks = num_res_blocks

        log_print(f"[Encoder]: use activation checkpoint: {act_checkpoint}")
        log_print(f"[Encoder]: z_channels: {z_channels}, patch size: {patch_size}")

        # Patcher.
        self.patcher = Patcher(patch_size, patch_method)
        in_channels = in_channels * patch_size * patch_size
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

        # downsampling
        self.conv_in = torch.nn.Conv2d(
            in_channels, channels, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution // patch_size
        in_ch_mult = (1,) + tuple(channels_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = channels * in_ch_mult[i_level]
            block_out = channels * channels_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        act_checkpoint=act_checkpoint,
                        use_residual_factor=use_residual_factor,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions and not force_not_attn:
                    log_print(f"[Encoder]: use attn at {curr_res}")
                    attn.append(
                        AttnBlock(
                            block_in,
                            act_checkpoint=act_checkpoint,
                            use_residual_factor=use_residual_factor,
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
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            act_checkpoint=act_checkpoint,
            use_residual_factor=use_residual_factor,
        )
        self.mid.attn_1 = AttnBlock(
            block_in,
            act_checkpoint=act_checkpoint,
            use_residual_factor=use_residual_factor,
        )
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            act_checkpoint=act_checkpoint,
            use_residual_factor=use_residual_factor,
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in, z_channels, kernel_size=3, stride=1, padding=1
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
        out_channels: int,
        channels: int,
        channels_mult: list[int],
        num_res_blocks: int,
        attn_resolutions: int,
        dropout: float,
        resolution: int,
        z_channels: int,
        spatial_compression: int,
        act_checkpoint: bool = False,
        use_residual_factor: bool = False,
        upsample_type: str = "RepeatConv",
        upsample_shortcut: str | None = None,
        force_not_attn: bool = False,
        **ignore_kwargs,
    ):
        super().__init__()
        self.num_resolutions = len(channels_mult)
        self.num_res_blocks = num_res_blocks

        log_print(f"[Decoder]: use activation checkpoint: {act_checkpoint}")
        log_print(f"[Decoder]: z_channels: {z_channels}")

        # UnPatcher.
        patch_size = ignore_kwargs.get("patch_size", 1)
        self.unpatcher = UnPatcher(
            patch_size, ignore_kwargs.get("patch_method", "rearrange")
        )
        out_ch = out_channels * patch_size * patch_size

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
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            act_checkpoint=act_checkpoint,
            use_residual_factor=use_residual_factor,
        )
        self.mid.attn_1 = AttnBlock(block_in, act_checkpoint=act_checkpoint)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            act_checkpoint=act_checkpoint,
            use_residual_factor=use_residual_factor,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = channels * channels_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        act_checkpoint=act_checkpoint,
                        use_residual_factor=use_residual_factor,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions and not force_not_attn:
                    log_print(f"[Decoder]: use attn at {curr_res}")
                    attn.append(
                        AttnBlock(
                            block_in,
                            act_checkpoint=act_checkpoint,
                            use_residual_factor=use_residual_factor,
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
                )
                curr_res = curr_res * 2
            self.up.insert(0, up)

        # end
        self.norm_out = Normalize(block_in)

        _wrap_fsdp_last_layer = ignore_kwargs.get("wrap_fsdp_last_layer", False)
        self._wrap_fsdp_last_layer = _wrap_fsdp_last_layer
        if _wrap_fsdp_last_layer:
            self.conv_out = FSDPNoWarpModule(
                torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)
            )
            log_print("[Decoder] use FSDPNoWarpModule")
        else:
            self.conv_out = torch.nn.Conv2d(
                block_in, out_ch, kernel_size=3, stride=1, padding=1
            )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
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
        h = self.conv_out(h)
        h = self.unpatcher(h)
        return h

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
            out_channels * decoder_patch_size * decoder_patch_size
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
            out_channels * unpatcher_sz * unpatcher_sz

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
        img_size = 512
        # 1024/4=256
        # 256/2=128
        encoder = Encoder(
            8,
            128,
            [2, 2, 2, 4],
            2,
            [32],
            0.1,
            img_size,
            16,
            8,
            True,
            patch_size=4,
            downsample_type="ConvPixelUnshuffle",
            downsample_shortcut="averaging",
        )
        decoder = Decoder(
            8,
            128,
            [2, 2, 2, 4],
            2,
            [32],
            0.1,
            img_size,
            16,
            8,
            act_checkpoint=True,
            patch_size=4,
            upsample_type="ConvPixelShuffle",
            upsample_shortcut="duplicating",
        )
        dtype = torch.bfloat16
        device = torch.device("cuda:1")
        encoder = encoder.to(device, dtype)
        decoder = decoder.to(device, dtype)

        bs = 4
        img = torch.randn(bs, 8, *[img_size] * 2).to(device, dtype)

        slots = encoder(img)
        recon = decoder(slots)

        print(recon.shape)

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
        import time

        time.sleep(10)

    test_auto_enc_dec()
    # test_diff_enc_dec()

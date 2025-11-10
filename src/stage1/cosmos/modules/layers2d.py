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
from functools import partial
from typing import Any, Callable, Literal, Optional, Sequence, Union, no_type_check

import numpy as np

# pytorch_diffusion + derived encoder decoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import create_conv2d, create_norm_layer
from timm.layers.layer_scale import LayerScale2d
from typing_extensions import deprecated

from src.utilities.logging import log_print

from .blocks import (
    AdaptiveInputConvLayer,
    AdaptiveOutputConvLayer,
    ConvNeXtBlock,
    DiCoBlock,
    DiffBandsInputConvIn,
    DiffBandsInputConvOut,
    FinalLayer,
    Normalize,
    ResnetBlock,
    ResnetBlockMoE2D,
    ResnetBlockSlotsInjected,
    TimestepEmbedder,
    make_attn,
    nonlinearity,
)
from .patching import Patcher, UnPatcher
from .resample import build_downsample_block, build_upsample_block


def is_list_tuple(x: Any) -> bool:
    return isinstance(x, (list, tuple))


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


# * --- Encoder and decoder --- #


def make_block_fn(
    block_name: Literal[
        "res_block",
        "dico_block",
        "convnext",
        "res_moe",
    ] = "res_block",
    moe_n_experts=4,
    act_checkpoint=False,
    use_residual_factor=False,
    moe_n_selected=1,
    moe_n_shared_experts=1,
    hidden_factor=2,
    moe_type="tc",
    padding_mode: str = "zeros",
    norm_type: str = "gn",
    token_mixer_type: Literal["res_block", "dico_block", "convnext"] = "res_block",
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
                ffn_type="glu",
                use_ffn=True,
            )

    elif block_name == "convnext":

        def block_fn(block_in, block_out, dropout, curr_res):
            return ConvNeXtBlock(
                block_in,
                block_in * hidden_factor,
                block_out,
                norm_type=norm_type,
                act_checkpoint=act_checkpoint,
                drop_path=dropout,
                **kwargs,
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
                use_dico_cca=False,
                act_type=("gelu", "gelu"),
                nin_shortcut_norm=True,
                **kwargs,
            )

    else:
        raise ValueError(
            f"block_name {block_name} is not supported. Supported: 'res_block', 'res_moe', 'dico_block', 'convnext'"
        )

    return block_fn


class Encoder(nn.Module):
    def __init__(
        self,
        # list of channels using the diffbandsblock (different experts); int otherwise
        # using a nested conv layer.
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
        # downsampling
        downsample_type: Literal[
            "PadConv", "Conv", "ConvPixelUnshuffle"
        ] = "PadConv",  # used in original cosmos tokenizer
        downsample_shortcut: Literal["averaging"] | None = None,
        downsample_kwargs: dict = {"padconv_use_manually_pad": False},
        # patch size, patcher, and blocks
        patch_size: int = 4,
        patch_method: str = "haar",
        conv_in_module: Literal["conv", "resnet", "inv_bottleneck", "moe"] = "conv",
        block_name: Literal["res_block", "dico_block", "res_moe"] = "res_block",
        attn_type: str = "attn_vanilla",
        # if block_name != 'moe', does not use
        moe_n_experts: int = 4,
        moe_n_selected: int = 1,
        moe_n_shared_experts: int = 1,
        hidden_factor: int = 2,
        moe_type: Literal["tc", "ec", "tc+ec"] = "tc",
        moe_token_mixer_type: Literal["res_block", "dico_block"] = "res_block",
        # padding and norm
        padding_mode: str = "zeros",
        norm_type: str = "gn",
        norm_groups: int = 32,
        resample_norm_keep: bool = False,
        adaptive_mode: str = "slice",
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
        self.block_name = block_name

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
            in_channels = [c * patch_size * patch_size for c in in_channels]
            self.conv_in = DiffBandsInputConvIn(
                band_lst=in_channels,
                hidden_dim=channels,
                basic_module=conv_in_module,
                padding_mode=padding_mode,
            )
        else:
            in_channels = in_channels * patch_size * patch_size

            # Use a nested conv here
            assert isinstance(in_channels, int)
            self.conv_in = AdaptiveInputConvLayer(
                in_channels=in_channels,
                out_channels=channels,
                use_bias=True,
                mode=adaptive_mode,
            )

            # NOTE: normally a conv
            # self.conv_in = torch.nn.Conv2d(
            #     in_channels,
            #     channels,
            #     kernel_size=3,
            #     stride=1,
            #     padding=1,
            #     padding_mode=padding_mode,
            # )

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
                if curr_res in attn_resolutions:
                    log_print(f"[Encoder]: use attn at {curr_res}")
                    attn.append(
                        make_attn(
                            block_in, attn_type=attn_type, act_checkpoint=act_checkpoint
                        )
                    )
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level < self.num_downsamples:
                down.downsample = build_downsample_block(
                    downsample_type,
                    block_in,
                    block_in,
                    shortcut=downsample_shortcut,
                    padding_mode=padding_mode,
                    norm_keep=resample_norm_keep,
                    **downsample_kwargs,
                    # padconv_use_manually_pad=downsample_manually_pad,
                )
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = block_fn(block_in, block_out, dropout, curr_res)
        self.mid.attn_1 = make_attn(
            block_in, attn_type=attn_type, act_checkpoint=act_checkpoint
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

    @no_type_check
    def forward(
        self, x: torch.Tensor, ret_interm_feats: bool | tuple | list = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        x: [bs, c, h, w], input images.
        ret_interm_feats: bool or list/tuple of int, intermidates features from given indices (list or tuple)
            or all encoder features, and additional middle block feauture.
        """
        x = self.patcher(x)
        feats = []

        # downsampling
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level < self.num_downsamples:
                h = self.down[i_level].downsample(h)
            if ret_interm_feats is True or (
                is_list_tuple(ret_interm_feats) and i_level in ret_interm_feats
            ):
                feats.append(h)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        if ret_interm_feats is True or (
            # -1 is the middle block
            is_list_tuple(ret_interm_feats) and -1 == ret_interm_feats[-1]
        ):
            feats.append(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h if not ret_interm_feats else (h, feats)


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
        upsample_type: Literal[
            "RepeatConv", "ConvPixelShuffle", "InterpolateConv"
        ] = "RepeatConv",
        upsample_shortcut: Literal["duplicating"] | None = None,
        upsample_kwargs: dict = {"interp_type": "xy_repeat"},
        conv_out_module: Literal["conv", "resnet", "inv_bottleneck", "moe"] = "conv",
        attn_type: str = "attn_vanilla",
        block_name: Literal["res_block", "dico_block", "res_moe"] = "res_block",
        moe_n_experts: int = 4,
        moe_n_selected: int = 1,
        moe_n_shared_experts: int = 1,
        hidden_factor: int = 2,
        moe_type: Literal["tc", "ec", "tc+ec"] = "tc",
        moe_token_mixer_type: Literal["res_block", "dico_block"] = "res_block",
        padding_mode: str = "zeros",
        norm_type: str = "gn",
        norm_groups: int = 32,
        patch_size: int = 4,
        patch_method: str = "haar",
        resample_norm_keep: bool = False,
        adaptive_mode: str = "slice",
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
        self.block_name = block_name

        log_print(
            f"[Decoder]: padding mode: {padding_mode}, norm type: {norm_type}, norm_groups: {norm_groups}, "
            f"use activation checkpoint: {act_checkpoint}"
        )
        log_print(f"[Decoder]: z_channels: {z_channels}")
        log_print(f"[Decoder]: Using block type: {block_name} ")

        # UnPatcher.
        self.patch_size = patch_size
        self.unpatcher = UnPatcher(patch_size, patch_method)

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
        self.mid.attn_1 = make_attn(
            block_in,
            attn_type=attn_type,
            act_checkpoint=act_checkpoint,
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
                if curr_res in attn_resolutions:
                    log_print(f"[Decoder]: use attn at {curr_res}")
                    attn.append(
                        make_attn(
                            block_in,
                            attn_type=attn_type,
                            act_checkpoint=act_checkpoint,
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
                    norm_keep=resample_norm_keep,
                    **upsample_kwargs,
                )
                curr_res = curr_res * 2
            self.up.insert(0, up)

        # end
        self.norm_out = Normalize(block_in, norm_type=norm_type, num_groups=norm_groups)

        if isinstance(out_channels, list):
            log_print("[Decoder]: use diffbands input")
            out_ch = [c * patch_size * patch_size for c in out_channels]
            conv_out = DiffBandsInputConvOut(
                band_lst=out_ch, hidden_dim=block_in, basic_module=conv_out_module
            )
        else:
            out_ch = out_channels * patch_size * patch_size
            # Use a nested conv
            assert isinstance(out_ch, int)
            conv_out = AdaptiveOutputConvLayer(
                in_channels=block_in,
                out_channels=out_ch,
                use_bias=True,
                mode=adaptive_mode,
            )

        # Ignore it: fsdp warpper, but not used
        _wrap_fsdp_last_layer = ignore_kwargs.get(
            "wrap_fsdp_last_layer", False
        )  # TODO: remove this
        self._wrap_fsdp_last_layer = _wrap_fsdp_last_layer
        if _wrap_fsdp_last_layer:
            self.conv_out = FSDPNoWarpModule(conv_out)
            log_print("[Decoder] use FSDPNoWarpModule")
        else:
            self.conv_out = conv_out

    @no_type_check
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
        assert out_channels is not None, "out_channels should be provided"
        conv_out_h = (h, out_channels * self.patch_size * self.patch_size)
        h = self.conv_out(*conv_out_h)
        h = self.unpatcher(h)
        return h

    def get_last_layer(self):
        if not self._wrap_fsdp_last_layer:
            return self.conv_out.weight
        else:
            return self.conv_out.wrap_mod.weight


# *==============================================================
# * Generative Decoder
# *==============================================================
from src.stage1.cosmos.modules.utils import AdaptiveGroupNorm


class GenerativeDecoder(Decoder):
    """Adapted from WeTok"""

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
        upsample_type: Literal[
            "RepeatConv", "ConvPixelShuffle", "InterpolateConv"
        ] = "RepeatConv",
        upsample_shortcut: Literal["duplicating"] | None = None,
        upsample_kwargs: dict = {"interp_type": "xy_repeat"},
        conv_out_module: Literal["conv", "resnet", "inv_bottleneck", "moe"] = "conv",
        attn_type: str = "attn_vanilla",
        block_name: Literal["res_block", "dico_block", "res_moe"] = "res_block",
        moe_n_experts: int = 4,
        moe_n_selected: int = 1,
        moe_n_shared_experts: int = 1,
        hidden_factor: int = 2,
        moe_type: Literal["tc", "ec", "tc+ec"] = "tc",
        moe_token_mixer_type: Literal["res_block", "dico_block"] = "res_block",
        padding_mode: str = "zeros",
        norm_type: str = "gn",
        norm_groups: int = 32,
        patch_size: int = 4,
        patch_method: str = "haar",
        resample_norm_keep: bool = False,
        adaptive_mode: str = "slice",
        per_layer_noise: bool = False,
        **ignore_kwargs,
    ):
        super().__init__(
            out_channels=out_channels,
            channels=channels,
            channels_mult=channels_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            resolution=resolution,
            z_channels=z_channels,
            spatial_compression=spatial_compression,
            act_checkpoint=act_checkpoint,
            use_residual_factor=use_residual_factor,
            upsample_type=upsample_type,
            upsample_shortcut=upsample_shortcut,
            upsample_kwargs=upsample_kwargs,
            conv_out_module=conv_out_module,
            attn_type=attn_type,
            block_name=block_name,
            moe_n_experts=moe_n_experts,
            moe_n_selected=moe_n_selected,
            moe_n_shared_experts=moe_n_shared_experts,
            hidden_factor=hidden_factor,
            moe_type=moe_type,
            moe_token_mixer_type=moe_token_mixer_type,
            padding_mode=padding_mode,
            norm_type=norm_type,
            norm_groups=norm_groups,
            patch_size=patch_size,
            patch_method=patch_method,
            resample_norm_keep=resample_norm_keep,
            adaptive_mode=adaptive_mode,
            **ignore_kwargs,
        )

        self.per_layer_noise = per_layer_noise

        # First conv_in should be [z, noise]
        block_in = channels * channels_mult[self.num_resolutions - 1]
        self.conv_in = create_conv2d(
            z_channels * 2, block_in, 3, padding_mode=padding_mode
        )

        # Code conditioning blocks
        self.cond_layers = nn.ModuleList()
        self.noise_ls = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block_out = channels * channels_mult[i_level]
            # adap_gn = AdaptiveGroupNorm(z_channels, block_in, eps=1e-6)
            adap_gn = create_norm_layer(
                "adaptivegn", z_channels, in_chan=block_in, eps=1e-6
            )
            self.cond_layers.append(adap_gn)
            if self.per_layer_noise:
                layer_noise_proj = nn.ModuleDict(
                    {
                        # ls(cat(h, noise)) init relative small
                        "ls": LayerScale2d(
                            block_in * 2, init_values=1e-4, inplace=True
                        ),
                        "norm": create_norm_layer(
                            "groupnorm", z_channels, num_groups=32
                        ),
                        "proj": nn.Sequential(
                            nn.SiLU(),
                            create_conv2d(
                                2 * block_in,
                                block_in,
                                3,
                                padding_mode=padding_mode,
                            ),
                        ),
                    }
                )
                self.noise_ls.append(layer_noise_proj)

    @no_type_check
    def forward(self, z: torch.Tensor, out_channels: int | None = None) -> torch.Tensor:
        cond = z.clone()

        # first cat the noise with z as the condition
        noise = torch.rand_like(z)
        z = torch.cat([z, noise], dim=1)  # cat z with an additional noise
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            # layerscale a noise and inject into the hiddens
            if self.per_layer_noise:
                layer_noise_proj = self.noise_ls[i_level]
                layer_noise = torch.randn_like(h)
                # norm, cat, ls, proj
                h_normed = layer_noise_proj["norm"](h)
                h_noisy = torch.cat([h_normed, layer_noise], dim=1)
                h_scaled = layer_noise_proj["ls"](h_noisy)
                h = layer_noise_proj["proj"](h_scaled) + h

            # conditioned on the input z
            h = self.cond_layers[i_level](h, cond)  # adaGN

            # blocks
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)

            # upsample
            if i_level >= (self.num_resolutions - self.num_upsamples):
                h = self.up[i_level].upsample(h)

        # out layer
        h = self.norm_out(h)
        h = nonlinearity(h)
        assert out_channels is not None, "out_channels should be provided"
        conv_out_h = (h, out_channels * self.patch_size * self.patch_size)
        h = self.conv_out(*conv_out_h)
        h = self.unpatcher(h)
        return h


# *==============================================================
# * Decoder Diffusion version
# *==============================================================


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
            norm_groups=1,
            hidden_factor=2,
            resample_norm_keep=True,
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
            block_name="dico_block",
            force_not_attn=True,
            norm_type="rms_triton",
            moe_token_mixer_type="dico_block",
            hidden_factor=2,
            resample_norm_keep=True,
        )
        dtype = torch.bfloat16
        device = torch.device("cuda:0")
        encoder = encoder.to(device, dtype)
        decoder = decoder.to(device, dtype)

        bs = 2
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
        # device = accelerator.device
        torch.cuda.set_device(1)
        device = torch.device("cuda")

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
            act_checkpoint=False,
            attn_type="none",
            norm_type="gn",
            norm_groups=32,
            block_name="res_block",
            hidden_factor=2,
            downsample_manually_pad=True,
            resample_norm_keep=False,
            conv_in_module="conv",
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
            patch_size=4,
            act_checkpoint=False,
            moe_type="tc+ec",
            attn_type="none",
            norm_type="gn",
            norm_groups=32,
            block_name="res_block",
            hidden_factor=2,
            resample_norm_keep=False,
            conv_out_module="conv",
        )
        dtype = torch.bfloat16
        encoder = encoder.to(device, dtype)
        decoder = decoder.to(device, dtype)

        optimizer = torch.optim.Adam(
            [*encoder.parameters(), *decoder.parameters()],
            lr=1e-4,
        )

        for _ in range(200):
            bs = 4
            # img = torch.randn(bs, 8, *[img_size] * 2).to(device, dtype).clip(-1,1)
            x = (
                torch.randint(0, 255, (bs, 8, 512, 512), dtype=torch.float32).to(
                    device, dtype
                )
                / 255.0
            )
            x = x * 2 - 1
            img = x.clip(-1, 1)

            with accelerator.autocast():
                z = encoder(img)
                recon = decoder(z, img.shape[1])

            print(recon.shape)

            if optimize:
                optimizer.zero_grad()
                loss = F.mse_loss(recon, img)
                log_print(f"<red>loss: {loss.item():.4f}</>")
                accelerator.backward(loss)
                optimizer.step()

        # accelerator.backward(recon.mean())
        # p_norms = {}
        # for n, p in encoder.named_parameters():
        #     p_norms[n] = p.norm().item()

        # # sort norms
        # p_norms = {
        #     k: v
        #     for k, v in sorted(p_norms.items(), key=lambda item: item[1], reverse=True)
        # }
        # largest_n = 300
        # # print the largest n norms
        # for k, v in p_norms.items():
        #     print(f"{k}: {v:.4f}")
        #     if largest_n > 0:
        #         largest_n -= 1
        #     else:
        #         break

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

    def test_generative_decoder():
        torch.cuda.set_device("cuda:1")

        decoder = GenerativeDecoder(
            16,
            128,
            [2, 4, 4],
            2,
            [],
            0.0,
            256,
            16,
            8,
            False,
            patch_size=1,
            padding_mode="reflect",
            per_layer_noise=True,
        ).cuda()

        x = torch.randn(1, 16, 32, 32).cuda()
        with torch.no_grad():
            out = decoder(x)

        print(out.shape)

    def test_nested_conv():
        conv_in = AdaptiveInputConvLayer(300, 32)
        x = torch.randn(1, 256, 32, 32)
        out = conv_in(x)
        print(out.shape)

        out.mean().backward()
        for n, p in conv_in.named_parameters():
            if p.grad is None:
                print(f"{n} has no grad")
            else:
                print(f"{n} grad sum: {p.grad.shape}")

    """
        python -m src.stage1.cosmos.modules.layers2d
    """

    # test_auto_enc_dec()
    # test_diff_enc_dec()
    # test_multi_bands_enc_dec(True)
    # test_generative_decoder()

    test_nested_conv()

    # test_moe_layer()

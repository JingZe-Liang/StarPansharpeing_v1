"""
UViT decoder that takes z, noise, and t (optional r) as inputs
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Optional, Union

import torch
import torch.nn as nn
from timm.layers.create_act import create_act_layer
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from typing_extensions import Annotated, Any

from src.utilities.config_utils import dataclass_from_dict

from .blocks import (
    AdaptiveInputConvLayer,
    AdaptiveOutputConvLayer,
    DiffBandsInputConvIn,
    DiffBandsInputConvOut,
)
from .t_blocks.conv_res import DownBlock2D, ResnetBlock2D, UpBlock2D
from .t_blocks.embeddings import (
    LearnedPositionalEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from .t_blocks.transformer_block import TransformerBlock, VisionTransformer
from .t_blocks.utils import init_weights, init_zero

# * --- Config --- #


@dataclass
class UViTDecoderConfig:
    in_channels: Any = 3  # int or list[int]
    z_dim: int = 4
    channels: int = 128
    ch_mult: tuple = (1, 2, 4, 4)
    act_fn: str = "silu"
    vit_act_fn: str = "geglu"
    layers_per_block: int = 2
    num_attention_heads: Optional[int] = None
    dropout: float = 0.0
    norm_num_groups: int = 32
    time_scale_shift: bool = True
    mid_nlayers: int = 12
    mid_theta: float = 100.0
    attn_window: int = 8
    eps: float = 1e-5
    ada_norm: bool = True
    learned_pos_embed: bool = False
    image_size: tuple | None = None
    relative_pos_embed: bool = True
    time_cond_type: str = "t-r"
    init: Optional[str] = None
    use_act_ckpt: bool = False


# * --- Model --- #


class UViTDecoder(nn.Module):
    # fmt: off
    SIZES = {
        "XS": {"channels": 32, "num_attention_heads": 4, "mid_nlayers": 8, "ch_mult": (1, 2, 3, 3)},
        "S": {"channels": 48, "num_attention_heads": 4, "mid_nlayers": 8, "ch_mult": (1, 2, 3, 3)},
        "B": {"channels": 64, "mid_nlayers": 10, "ch_mult": (1, 2, 3, 3)},
        "M": {"channels": 96, "mid_nlayers": 12, "ch_mult": (1, 2, 3, 3)},
        "L": {"channels": 96, "mid_nlayers": 16, "ch_mult": (1, 2, 4, 4)},
        "XL": {"channels": 128, "mid_nlayers": 16, "ch_mult": (1, 2, 4, 4)},
        "H": {"channels": 192, "mid_nlayers": 16, "ch_mult": (1, 2, 4, 4)},
        "XH": {"channels": 256, "mid_nlayers": 16, "ch_mult": (1, 2, 4, 4)},
        "G": {"channels": 384, "mid_nlayers": 16, "ch_mult": (1, 2, 4, 4)},
    }
    # fmt: on

    # Based on code from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_2d_condition.py
    def __init__(
        self,
        in_channels=3,
        z_dim=4,
        channels=128,
        ch_mult=(1, 2, 4, 4),
        act_fn: str = "silu",
        vit_act_fn: str = "geglu",
        layers_per_block=2,
        num_attention_heads: Optional[int] = None,
        total_resolutions: int = 8,
        dropout=0.0,
        norm_num_groups=32,
        time_scale_shift=True,
        mid_nlayers=12,
        mid_theta=100.0,
        attn_window=8,
        eps=1e-5,
        ada_norm=True,
        learned_pos_embed=False,
        image_size: int | None = None,
        relative_pos_embed=True,
        time_cond_type="t-r",
        init: Optional[Mapping] = None,
        use_act_ckpt: bool = False,
        **_kwargs,
    ):
        ### Config ###
        self.out_dim = in_channels
        self.ada_norm = ada_norm
        self.grad_checkpointing = use_act_ckpt

        # Compute appropriate number of channels for each level, adjust for GroupNorm
        self.ch_level = [
            math.ceil(channels * ch_f / norm_num_groups) * norm_num_groups
            for ch_f in ch_mult
        ]
        channels = self.ch_level[0]  # The first channel is the input channel
        self.mid_dim = self.ch_level[-1]

        if isinstance(dropout, (int, float)):
            dropout = [dropout] * len(self.ch_level)

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(self.ch_level)

        super().__init__()

        ### Input ###
        z_conv_in = nn.Conv2d(z_dim, channels, kernel_size=3, padding=1)
        if isinstance(in_channels, (list, tuple)):
            noise_conv_in = DiffBandsInputConvIn(
                in_channels, channels, padding_mode="reflect"
            )
        else:
            noise_conv_in = AdaptiveInputConvLayer(in_channels, channels)
        fused_conv = nn.Conv2d(
            channels * 2, channels, kernel_size=3, padding=1, groups=channels
        )
        self.conv_in = nn.ModuleDict(
            {
                "z_conv_in": z_conv_in,
                "noise_conv_in": noise_conv_in,
                "fused_conv_in": fused_conv,
            }
        )

        ### Time ###
        time_embed_dim = channels * 4
        self.time_proj = Timesteps(
            channels, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_cond_type = time_cond_type
        self.use_delta_t_embed = time_cond_type in ["t-r", "r", "t,t-r", "r,t-r", "t,r,t-r"]  # fmt: skip
        if self.use_delta_t_embed:
            self.delta_t_proj = Timesteps(
                channels, flip_sin_to_cos=True, downscale_freq_shift=0
            )
        self.time_embedding = TimestepEmbedding(channels, time_embed_dim, act_fn=act_fn)

        ### AdaNorm Embedding ###
        if ada_norm:
            self.ada_ctx_proj = torch.nn.Sequential(
                torch.nn.Conv2d(z_dim, channels, kernel_size=3, stride=1, padding=1),
                torch.nn.SiLU(),
                torch.nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            )

        ### Down blocks ###
        assert (n_resamples := math.log2(total_resolutions)) % 1 == 0, (
            "total_resamples should be a power of 2"
        )
        n_resamples = int(n_resamples)
        self.down_blocks = nn.ModuleList([])
        output_channel = channels
        for i_level, ch in enumerate(self.ch_level):
            input_channel = output_channel
            output_channel = ch
            is_final_block = i_level == len(self.ch_level) - 1

            if layers_per_block[i_level] != 0:
                self.down_blocks.append(
                    DownBlock2D(
                        num_layers=layers_per_block[i_level],
                        in_channels=input_channel,
                        out_channels=output_channel,
                        temb_channels=time_embed_dim,
                        dropout=dropout[i_level],
                        add_downsample=not is_final_block
                        and (i_level < n_resamples),  # FIXME: add downsample earlier?
                        resnet_act_fn=act_fn,
                        resnet_groups=norm_num_groups,
                        time_scale_shift=time_scale_shift,
                        resnet_eps=eps,
                        ada_norm=ada_norm,
                        ada_emb_dim=channels,
                    )
                )

        # Mid block ###
        down_scale = total_resolutions  # (2 ** (len(self.ch_level) - 1),)
        self.mid_block = UViTMiddleTransformer(
            inner_dim=output_channel,
            dropout=dropout[-1],
            num_layers=mid_nlayers,
            norm_num_groups=norm_num_groups,
            num_attention_heads=num_attention_heads,
            rope_theta=mid_theta,
            attn_window=attn_window,
            eps=eps,
            ada_norm=ada_norm,
            ada_emb_dim=channels,
            learned_pos_embed=learned_pos_embed,
            sample_size=(image_size // down_scale, image_size // down_scale)
            if learned_pos_embed
            else None,
            relative_pos_embed=relative_pos_embed,
            act_fn=vit_act_fn,
        )

        ### Up blocks ###
        self.up_blocks = nn.ModuleList([])

        for i_level, ch in enumerate(reversed(self.ch_level)):
            input_channel = (
                self.ch_level[-i_level - 2]
                if i_level < len(self.ch_level) - 1
                else self.ch_level[0]
            )
            prev_output_channel = output_channel
            output_channel = ch

            is_final_block = i_level == len(self.ch_level) - 1

            if layers_per_block[-i_level - 1] != 0:
                self.up_blocks.append(
                    UpBlock2D(
                        num_layers=layers_per_block[-i_level - 1] + 1,
                        in_channels=input_channel,
                        out_channels=output_channel,
                        prev_output_channel=prev_output_channel,
                        temb_channels=time_embed_dim,
                        resolution_idx=i_level,
                        dropout=dropout[-i_level - 1],
                        add_upsample=(not is_final_block) and (i_level < n_resamples),
                        resnet_act_fn=act_fn,
                        resnet_groups=norm_num_groups,
                        time_scale_shift=time_scale_shift,
                        resnet_eps=eps,
                        ada_norm=ada_norm,
                        ada_emb_dim=channels,
                    )
                )

        ### Output ###
        self.conv_norm_out = nn.GroupNorm(
            num_channels=channels, num_groups=norm_num_groups, eps=eps
        )
        self.conv_out_act = create_act_layer(act_fn)
        if isinstance(in_channels, (list, tuple)):
            self.conv_out = DiffBandsInputConvOut(
                in_channels, channels, padding_mode="reflect"
            )
        else:
            self.conv_out = AdaptiveOutputConvLayer(channels, in_channels)

        ### Null condition h ###
        self.null_cond_h = nn.Parameter(torch.randn(1, z_dim, 1, 1) * 0.2)

        ### Weights init ###
        self.init_weights(**(init or {}))

    @classmethod
    def create_model(cls, size: str, **overrides):
        cfg = dataclass_from_dict(UViTDecoderConfig, overrides)
        if size is not None:
            if size in cls.SIZES:
                kwargs = {**cls.SIZES[size], **(asdict(cfg) or {})}
            else:
                raise ValueError(
                    f"Unknown size '{size}' for UViTDecoder. Available sizes: {list(cls.SIZES.keys())}"
                )
        return cls(**kwargs)

    def init_weights(self, method="xavier_uniform", ckpt_module="decoder", **kwargs):
        for m in self.modules():
            if isinstance(m, ResnetBlock2D):
                init_zero(m.conv2)
            elif isinstance(m, TransformerBlock):
                init_zero(m.ff.proj_out)

        init_weights(self, method=method, ckpt_module=ckpt_module, **kwargs)

    def _expand_time(self, timestep, bs: int, device):
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.float64, device=device)
        elif len(timestep.shape) == 0:
            timestep = timestep[None].to(device)
        timestep = timestep.expand(bs)
        return timestep

    def get_time_embed(
        self, sample: torch.Tensor, timesteps: tuple | torch.Tensor | float
    ) -> torch.Tensor:
        bs, device = sample.shape[0], sample.device
        if isinstance(timesteps, (list, tuple)):
            timesteps = [self._expand_time(t, bs, device) for t in timesteps]
        else:
            timesteps = self._expand_time(timesteps, bs, device)

        t_emb = self._get_delta_embed(timesteps)
        t_emb = self.time_embedding(t_emb, None)
        return t_emb

    def _get_delta_embed(self, ts):
        if isinstance(ts, (list, tuple)):
            t, r = ts
        else:
            assert not self.use_delta_t_embed, "timestep should be a tuple of (t, r)"
            t = ts
            return self.time_proj(t)

        if self.use_delta_t_embed:
            delta_embedder = self.delta_t_proj
        else:
            delta_embedder = self.time_proj

        if self.time_cond_type == "t-r":
            delta_embed = delta_embedder(t - r)
        elif self.time_cond_type == "r":
            delta_embed = delta_embedder(r)
        elif self.time_cond_type == "t,r":
            delta_embed = self.time_proj(t) + delta_embedder(r)
        elif self.time_cond_type == "t,t-r":
            delta_embed = self.time_proj(t) + delta_embedder(t - r)
        elif self.time_cond_type == "r,t-r":
            delta_embed = self.time_proj(r) + delta_embedder(t - r)
        elif self.time_cond_type == "t,r,t-r":
            delta_embed = self.time_proj(t) + self.time_proj(r) + delta_embedder(t - r)
        else:
            raise NotImplementedError(
                f"Time cond type {self.time_cond_type} not implemented"
            )
        return delta_embed

    def _forward_downs(self, x, t_emb, ctx_emb, use_act_ckpt=False):
        # 2. Down blocks
        down_block_res = [x]
        for downsample_block in self.down_blocks:
            if use_act_ckpt:
                x, res_samples = checkpoint(
                    downsample_block, x, t_emb, ctx_emb, use_reentrant=False
                )
            else:
                x, res_samples = downsample_block(
                    hidden_states=x, temb=t_emb, ctx_emb=ctx_emb
                )
            down_block_res.extend(res_samples)
        return x, down_block_res

    def _forward_ups(
        self, x, t_emb, ctx_emb, down_block_res, use_act_ckpt=False
    ) -> torch.Tensor:
        for upsample_block in self.up_blocks:
            res_samples = down_block_res[-len(upsample_block.resnets) :]
            down_block_res = down_block_res[: -len(upsample_block.resnets)]

            if use_act_ckpt:
                x = checkpoint(
                    upsample_block,
                    x,
                    res_samples,
                    t_emb,
                    None,
                    ctx_emb,
                    use_reentrant=False,
                )
            else:
                x = upsample_block(
                    hidden_states=x,
                    res_hidden_states_tuple=res_samples,
                    temb=t_emb,
                    ctx_emb=ctx_emb,
                )
        return x  # type: ignore[return-value]

    def _forward_mids(self, x, t_emb, ctx_emb, use_act_ckpt=False):
        # TODO: middle transformer need to conditioned on t_emb and z.
        x = self.mid_block(x, ctx_emb=ctx_emb, use_act_ckpt=use_act_ckpt)
        return x

    def forward(
        self,
        x: Tensor,
        t,
        r=None,
        z: Tensor | None = None,
        inp_shape: Annotated[torch.Size | int, "bs,c,h,w or c"] | None = None,
        return_zs=False,  # always be False
        derivative=False,
    ) -> Tensor:
        # t: timestep, t or (t, r)
        out_chan = (
            inp_shape[1] if isinstance(inp_shape, (torch.Size, tuple)) else inp_shape
        )

        ### Prepare input ###

        # Concat with z and project
        z_expanded = torch.nn.functional.interpolate(
            z, size=(x.shape[-2], x.shape[-1]), mode="nearest"
        )

        # conv ins
        z_expanded = self.conv_in["z_conv_in"](z_expanded)
        x = self.conv_in["noise_conv_in"](x)
        x = self.conv_in["fused_conv_in"](torch.cat([x, z_expanded], dim=1))

        ctx_emb = None
        if self.ada_norm:
            ctx_emb = self.ada_ctx_proj(z)

        ### Forward pass ###
        use_act_ckpt = self.grad_checkpointing and self.training and not derivative

        # 1. Time embedding
        if r is not None:
            t = (t, r)
        t_emb = self.get_time_embed(sample=x, timesteps=t)

        # 2. Down blocks
        x, enc_res = self._forward_downs(x, t_emb, ctx_emb, use_act_ckpt=use_act_ckpt)

        # 4. Mid block
        x = self._forward_mids(x, t_emb, ctx_emb, use_act_ckpt=use_act_ckpt)

        # 5. Up blocks
        x = self._forward_ups(x, t_emb, ctx_emb, enc_res, use_act_ckpt=use_act_ckpt)

        # 6. Output
        if self.conv_norm_out:
            x = self.conv_norm_out(x)
        x = self.conv_out_act(x)
        x = self.conv_out(x, out_chan)

        return x

    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable


class UViTMiddleTransformer(VisionTransformer):
    def __init__(
        self,
        *args,
        sample_size=None,
        act_fn: str = "geglu",
        learned_pos_embed=True,
        norm_num_groups: int = 32,
        out_norm: bool = False,
        **kwargs,
    ):
        super().__init__(
            *args,
            learned_pos_embed=False,
            sample_size=sample_size,
            act_fn=act_fn,
            out_norm=out_norm,
            **kwargs,
        )

        self.norm = torch.nn.GroupNorm(
            num_groups=norm_num_groups,
            num_channels=self.in_channels,
            eps=1e-6,
            affine=True,
        )

        self.pre_pos_embeddings = None
        if learned_pos_embed:
            self.pre_pos_embeddings = LearnedPositionalEmbedding(
                (self.inner_dim, *sample_size)
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        ctx_emb: Optional[torch.Tensor] = None,
        use_act_ckpt: bool = False,
    ) -> torch.Tensor:
        residual = hidden_states
        if self.pre_pos_embeddings is not None:
            hidden_states = self.pre_pos_embeddings(hidden_states)

        hidden_states = self.norm(hidden_states)
        if ctx_emb is not None and ctx_emb.shape[-2:] != hidden_states.shape[-2:]:
            ctx_emb = nn.functional.interpolate(
                ctx_emb, size=hidden_states.shape[-2:], mode="nearest"
            )

        hidden_states = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            ctx_emb=ctx_emb,
            out_2d_map=True,
            use_act_ckpt=use_act_ckpt,
        )
        output = hidden_states + residual
        return output


def test_uvit_decoder():
    # Test different model sizes
    sizes = ["S", "M", "L"]

    for size in sizes:
        print(f"\n=== Testing {size} model ===")

        # Create decoder
        decoder = UViTDecoder.create_model(
            size=size,
            in_channels=3,
            z_dim=4,
            image_size=(64, 64),
            time_cond_type="t-r",
            use_act_ckpt=False,
        ).cuda()

        print(f"Model {size} created successfully")

        # Test data
        bs, c, h, w = 2, 3, 64, 64
        z_dim = 4

        x = torch.randn(bs, c, h, w).cuda()
        z = torch.randn(bs, z_dim, h // 4, w // 4).cuda()
        t = torch.randint(0, 1000, (bs,)).cuda()
        r = torch.randint(0, 1000, (bs,)).cuda()

        print(f"Input shapes: x={x.shape}, z={z.shape}")

        # Test forward pass with different modes
        decoder.eval()
        with torch.no_grad():
            # Test mode 2: with t and r
            print("Testing forward with t and r...")
            output2 = decoder(x, t, r=r, z=z)
            print(f"Output shape (t, r): {output2.shape}")

        # Test training mode with gradient
        print("Testing training mode with gradient...")
        decoder.train()
        x.requires_grad_(True)

        output = decoder(x, t, r=r, z=z)
        loss = output.mean()
        loss.backward()

        print(f"Training loss: {loss.item():.6f}")
        print(f"Input grad norm: {x.grad.norm().item():.6f}")

        # Test with activation checkpointing
        print("Testing with activation checkpointing...")
        decoder_ckpt = UViTDecoder.create_model(
            size=size,
            in_channels=3,
            z_dim=4,
            image_size=(64, 64),
            time_cond_type="t-r",
            use_act_ckpt=True,
        ).cuda()

        decoder_ckpt.train()
        output_ckpt = decoder_ckpt(x, t, r=r, z=z)
        loss_ckpt = output_ckpt.mean()
        loss_ckpt.backward()

        print(f"Checkpoint loss: {loss_ckpt.item():.6f}")

        # Compare outputs
        diff = torch.abs(output - output_ckpt).max().item()
        print(f"Max difference between normal and checkpoint: {diff:.2e}")

        print(f"Model {size} test completed successfully")

    print("\n=== All tests completed ===")


if __name__ == "__main__":
    """
    python -m src.stage1.cosmos.modules.uvit_decoder
    """
    test_uvit_decoder()

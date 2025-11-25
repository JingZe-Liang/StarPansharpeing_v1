import functools

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.checkpoint import checkpoint

from .blocks import Spatial2DNATBlock, Spatial2DNATBlockConditional


class Spatial2DNatStage(nn.Module):
    def __init__(
        self,
        in_chans: int,
        embed_dim: list[int],
        depths: list[int],
        cond_width: int | None,
        out_chans: int | None,
        norm_layer: str = "layernorm2d",
        drop_path: float = 0.0,
        norm_eps: float = 1e-6,
        k_size=8,
        stride=2,
        dilation=2,
        n_heads=8,
        ffn_ratio: float | int = 2,
        qkv_bias=True,
        qk_norm="layernorm2d",
    ):
        super().__init__()
        n_layers = len(depths)
        self.grad_checkpointing = False
        self._use_condition = cond_width is not None

        layers = nn.ModuleList()
        for s, dim in enumerate(embed_dim):
            stage_in_chs = embed_dim[s - 1] if s > 0 else in_chans
            block_kwargs = dict(
                k_size=k_size,
                stride=stride,
                dilation=dilation,
                n_heads=n_heads,
                ffn_ratio=ffn_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                norm_layer=norm_layer,
                norm_eps=norm_eps,
                drop_path=drop_path,
            )
            block_cls = Spatial2DNATBlockConditional if self._use_condition else Spatial2DNATBlock
            blocks = [
                block_cls(
                    in_channels=stage_in_chs if i == 0 else dim,
                    out_channels=dim,
                    cond_chs=cond_width,
                    **block_kwargs,
                )
                for i in range(depths[s])
            ]
            layers.append(nn.ModuleList(blocks))
        self.layers = layers

        # Output projection
        if out_chans is not None:
            self.out_conv = nn.Conv2d(embed_dim[-1], out_chans, kernel_size=1)
        else:
            self.out_conv = nn.Identity()

    def forward(self, x, cond=None):
        # interpolate the condition
        if cond is not None:
            cond = nn.functional.interpolate(cond, size=x.shape[2:], mode="bilinear", align_corners=False)

        def _closure(x, cond):
            for stage in self.layers:
                for block in stage:
                    if self._use_condition:
                        x = block(x, cond)
                    else:
                        x = block(x)
            return x

        # Checkpointing
        if self.grad_checkpointing and self.training:
            x = checkpoint(_closure, x, cond, use_reentrant=False)
        else:
            x = _closure(x, cond)

        x = self.out_conv(x)
        return x

    def set_grad_checkpointing(self, enable: bool = True):
        self.grad_checkpointing = enable

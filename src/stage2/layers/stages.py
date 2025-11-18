from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Tuple, Union

import torch as th
import torch.nn as nn
from timm.layers import (
    DropPath,
    create_act_layer,
    get_norm_act_layer,
)
from timm.layers.create_conv2d import create_conv2d
from timm.layers.create_norm import create_norm_layer
from timm.layers.helpers import make_divisible
from timm.models import checkpoint_seq, named_apply
from timm.models.vitamin import (
    Downsample2d,
    GeGluMlp,
    StridedConv,
    _init_conv,
)
from torch.utils.checkpoint import checkpoint

from .blocks import Spatial2DNATBlock, Spatial2DNATBlockConditional, build_block
from .conv import MBConv, MbConvLNBlock, MBStem


class _BasicStages(nn.Module):
    def __init__(self):
        super().__init__()
        self.grad_checkpointing = False
        self._use_condition = False

    def set_grad_checkpointing(self, enable: bool = True):
        self.grad_checkpointing = enable

    def forward(self, x, cond=None):
        # interpolate the condition
        if cond is not None:
            cond = th.nn.functional.interpolate(
                cond, size=x.shape[2:], mode="bilinear", align_corners=False
            )

        # stages
        for stage in self.stages:
            for block in stage:
                if self.grad_checkpointing and self.training:
                    x = checkpoint(block, x, cond, use_reentrant=False)
                else:
                    x = block(x, cond)

        return self.out_conv(x)


class MbConvSequentialCond(_BasicStages):
    def __init__(
        self,
        in_chans: int,
        embed_dim: list[int],
        depths: list[int],
        cond_width: int | None,
        out_chans: int | None = None,
        stride: int = 1,
        drop_path: float = 0.0,
        kernel_size: int = 3,
        norm_layer: str = "layernorm2d",
        norm_eps: float = 1e-6,
        act_layer: str = "gelu",
        expand_ratio: float = 4.0,
        **kwargs,
    ):
        super().__init__()
        self.grad_checkpointing = False

        stages = []
        self.num_stages = len(embed_dim)
        for s, dim in enumerate(embed_dim):  # stage
            stage_in_chs = embed_dim[s - 1] if s > 0 else in_chans
            blocks = [
                MbConvLNBlock(
                    in_chs=stage_in_chs if d == 0 else dim,
                    out_chs=dim,
                    cond_chs=cond_width,
                    stride=stride,  # 2 if d == 0 else 1,
                    drop_path=drop_path,
                    norm_layer=norm_layer,
                    norm_eps=norm_eps,
                    act_layer=act_layer,
                    expand_ratio=expand_ratio,
                )
                for d in range(depths[s])
            ]
            stages.append(nn.ModuleList(blocks))
        self.stages = nn.ModuleList(stages)
        if out_chans:
            self.out_conv = nn.Conv2d(embed_dim[-1], out_chans, kernel_size=1)
        else:
            self.out_conv = nn.Identity()

        self._use_condition = cond_width is not None


class MbConvStagesCond(_BasicStages):
    """MobileConv for stage 1 and stage 2 of ViTamin"""

    def __init__(
        self,
        in_chans: int,
        stem_width: int,
        embed_dim: list[int],
        depths: list[int],
        cond_width: int,
        stride: int = 1,
        drop_path: float = 0.0,
        kernel_size: int = 3,
        norm_layer: str = "layernorm2d",
        norm_eps: float = 1e-6,
        act_layer: str = "gelu",
        expand_ratio: float = 4.0,
        **kwargs,
    ):
        super().__init__()
        self.grad_checkpointing = False

        # FIXME: difference with MbConvSequentialCond?
        self.stem = MBStem(in_chs=in_chans, out_chs=stem_width)

        stages = []
        self.num_stages = len(embed_dim)
        for s, dim in enumerate(embed_dim):  # stage
            stage_in_chs = embed_dim[s - 1] if s > 0 else stem_width
            blocks = [
                MbConvLNBlock(
                    in_chs=stage_in_chs if d == 0 else dim,
                    out_chs=dim,
                    cond_chs=stem_width,
                    stride=stride,  # 2 if d == 0 else 1,
                    drop_path=drop_path,
                    norm_layer=norm_layer,
                    norm_eps=norm_eps,
                    act_layer=act_layer,
                    expand_ratio=expand_ratio,
                )
                for d in range(depths[s])
            ]
            stages.append(nn.ModuleList(blocks))
        self.stages = nn.ModuleList(stages)
        self.out_conv = (
            nn.Identity()
        )  # Add output conv to match parent class expectation

    def forward(self, x, cond):
        x = self.stem(x)
        return super().forward(x, cond)


class Spatial2DNatStage(_BasicStages):
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
        self._use_condition = cond_width is not None

        stages = nn.ModuleList()
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
            block_cls = (
                Spatial2DNATBlockConditional
                if self._use_condition
                else Spatial2DNATBlock
            )
            blocks = [
                block_cls(
                    in_channels=stage_in_chs if i == 0 else dim,
                    out_channels=dim,
                    cond_chs=cond_width,
                    **block_kwargs,
                )
                for i in range(depths[s])
            ]
            stages.append(nn.ModuleList(blocks))
        self.stages = stages

        # Output projection
        if out_chans is not None:
            self.out_conv = nn.Conv2d(embed_dim[-1], out_chans, kernel_size=1)
        else:
            self.out_conv = nn.Identity()


class ResBlockStage(_BasicStages):
    """
    ResNet-style block stage for processing image features.

    This stage consists of multiple ResNet blocks organized in different stages
    with optional conditional inputs. It supports gradient checkpointing for
    memory-efficient training.
    """

    def __init__(
        self,
        in_chans: int,
        embed_dim: list[int],
        depths: list[int],
        cond_width: int | None = None,
        out_chans: int | None = None,
        norm_layer: str = "layernorm2d",
        norm_eps: float = 1e-6,
        act_layer: str = "relu6",
        drop_path: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()

        self._use_condition = cond_width is not None
        self.num_stages = len(embed_dim)

        # Build stages
        stages = []
        for s, dim in enumerate(embed_dim):
            stage_in_chs = embed_dim[s - 1] if s > 0 else in_chans
            blocks = [
                self._build_resblock(
                    in_chs=stage_in_chs if i == 0 else dim,
                    out_chs=dim,
                    cond_chs=cond_width,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depths[s])
            ]
            stages.append(nn.ModuleList(blocks))

        self.stages = nn.ModuleList(stages)

        # Output projection
        if out_chans is not None:
            self.out_conv = nn.Conv2d(embed_dim[-1], out_chans, kernel_size=1)
        else:
            self.out_conv = nn.Identity()

    def _build_resblock(
        self,
        in_chs: int,
        out_chs: int,
        cond_chs: int | None = None,
        norm_layer: str = "layernorm2d",
        act_layer: str = "relu6",
    ) -> nn.Module:
        blk = build_block(
            "ResBlock",
            in_chs,
            out_chs,
            norm=norm_layer,
            act=act_layer,
            cond_channels=cond_chs,
        )

        return blk

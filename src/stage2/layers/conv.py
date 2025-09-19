from functools import partial

import torch.nn as nn
import torch.utils.checkpoint
from timm.layers.create_act import create_act_layer, get_act_layer
from timm.layers.create_conv2d import create_conv2d
from timm.layers.create_norm import create_norm_layer
from timm.layers.create_norm_act import get_norm_act_layer
from timm.layers.drop import DropPath
from timm.layers.helpers import make_divisible
from timm.models import checkpoint_seq, named_apply
from timm.models.convnext import ConvNeXtBlock
from timm.models.vitamin import (
    Downsample2d,
    GeGluMlp,
    StridedConv,
    _init_conv,
)
from torch.utils.checkpoint import checkpoint


class Stem(nn.Module):
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        act_layer: str = "gelu",
        norm_layer: str = "layernorm2d",
        norm_eps: float = 1e-6,
        bias: bool = True,
    ):
        super().__init__()
        norm_act_layer = partial(
            get_norm_act_layer(norm_layer, act_layer), eps=norm_eps
        )
        self.out_chs = out_chs

        self.conv1 = create_conv2d(in_chs, out_chs, 3, stride=1, bias=bias)
        self.norm1 = norm_act_layer(out_chs)
        self.conv2 = create_conv2d(out_chs, out_chs, 3, stride=1, bias=bias)

        named_apply(_init_conv, self)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        return x


class MbConvLNBlock(nn.Module):
    """Pre-Norm Conv Block - 1x1 - kxk - 1x1, w/ inverted bottleneck (expand)"""

    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        cond_chs: int,
        stride: int = 1,
        drop_path: float = 0.0,
        kernel_size: int = 3,
        norm_layer: str = "layernorm2d",
        norm_eps: float = 1e-6,
        act_layer: str = "gelu",
        expand_ratio: float = 4.0,
    ):
        super(MbConvLNBlock, self).__init__()
        self.grad_checkpointing = False

        self.stride, self.in_chs, self.out_chs = stride, in_chs, out_chs
        mid_chs = make_divisible(out_chs * expand_ratio)
        prenorm_act_layer = partial(
            get_norm_act_layer(norm_layer, act_layer), eps=norm_eps
        )

        if stride == 2:
            self.shortcut = Downsample2d(in_chs, out_chs, pool_type="avg", bias=True)
        elif in_chs != out_chs:
            self.shortcut = nn.Conv2d(in_chs, out_chs, 1, bias=True)
        else:
            self.shortcut = nn.Identity()

        self.pre_norm = prenorm_act_layer(in_chs, apply_act=False)
        self.down = nn.Identity()
        self.conv1_1x1 = create_conv2d(in_chs, mid_chs, 1, stride=1, bias=True)
        self.act1 = create_act_layer(act_layer, inplace=True)
        self.conv2_kxk = create_conv2d(
            mid_chs,
            mid_chs,
            kernel_size,
            stride=stride,
            dilation=1,
            groups=mid_chs,
            bias=True,
        )
        self.act2 = create_act_layer(act_layer, inplace=True)
        self.conv3_1x1 = create_conv2d(mid_chs, out_chs, 1, bias=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.cond_conv_kxk = create_conv2d(
            cond_chs, mid_chs, 3, stride=1, padding=1, bias=True
        )

    def forward_(self, x, cond):
        shortcut = self.shortcut(x)

        x = self.pre_norm(x)
        x = self.down(x)  # nn.Identity()

        # 1x1 expansion conv & act
        x = self.conv1_1x1(x)
        x = self.act1(x)

        # (strided) depthwise 3x3 conv & act
        cond = self.cond_conv_kxk(cond)
        x = self.conv2_kxk(x) + cond
        x = self.act2(x)

        # 1x1 linear projection to output width
        x = self.conv3_1x1(x)
        x = self.drop_path(x) + shortcut

        return x

    def forward(self, x, cond):
        if self.grad_checkpointing:
            return torch.utils.checkpoint.checkpoint(
                self.forward_, x, cond, use_reentrant=False
            )
        else:
            return self.forward_(x, cond)

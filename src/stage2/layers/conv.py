from functools import partial
from typing import Any, Optional, Union

import torch.nn as nn
import torch.utils.checkpoint
from timm.layers.create_act import create_act_layer, get_act_layer
from timm.layers.create_conv2d import create_conv2d
from timm.layers.create_norm import create_norm_layer
from timm.layers.create_norm_act import create_norm_act_layer, get_norm_act_layer
from timm.layers.drop import DropPath
from timm.layers.helpers import make_divisible, to_2tuple, to_3tuple, to_ntuple
from timm.models import checkpoint_seq, named_apply
from timm.models.convnext import ConvNeXtBlock
from timm.models.vitamin import (
    Downsample2d,
    _init_conv,
)
from torch.utils.checkpoint import checkpoint

from .functional import ConditionalBlock


class MBStem(nn.Module):
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
        cond_chs: int | None = None,
        stride: int = 1,
        drop_path: float = 0.0,
        kernel_size: int = 3,
        norm_layer: str = "layernorm2d",
        norm_eps: float = 1e-6,
        act_layer: str = "gelu",
        expand_ratio: float = 4.0,
        cond_type: str = "add",
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

        self.cond_type = cond_type
        if cond_chs is not None:
            cond_out_chs = out_chs * 2 if cond_type == "adaln3" else out_chs
            self.cond_conv_kxk = nn.Sequential(
                create_conv2d(cond_chs, out_chs, 1, bias=True),
                create_norm_act_layer("layernorm2d", mid_chs // 2, "gelu"),
                create_conv2d(out_chs, cond_out_chs, 3, bias=False, groups=out_chs),
            )

    def forward_(self, x, cond=None):
        shortcut = self.shortcut(x)

        x = self.pre_norm(x)
        x = self.down(x)

        # 1x1 expansion conv & act
        x = self.conv1_1x1(x)
        x = self.act1(x)

        # (strided) depthwise 3x3 conv & act
        x = self.conv2_kxk(x)
        x = self.act2(x)

        # 1x1 linear projection to output width
        x = self.conv3_1x1(x)

        # conditioning
        if cond is not None and hasattr(self, "cond_conv_kxk"):
            # modulate
            if self.cond_type == "adaln3":
                scale, shift = self.cond_conv_kxk(cond).chunk(2, dim=1)
                x = x * (1 + scale) + shift
            else:
                x = x + self.cond_conv_kxk(cond)

        # output
        x = self.drop_path(x) + shortcut

        return x

    @torch.compile
    def forward(self, x, cond=None):
        if self.grad_checkpointing:
            return torch.utils.checkpoint.checkpoint(
                self.forward_, x, cond, use_reentrant=False
            )
        else:
            return self.forward_(x, cond)


# *==============================================================
# * MB layers
# * taken from https://github.com/dc-ai-projects/DC-Gen/blob/main/dc_gen/
# *==============================================================

# alias for the timm norms
from timm.layers import create_norm

for new_name, orig_name in [
    ("bn2d", "batchnorm2d"),
    ("ln2d", "layernorm2d"),
    ("trms2d", "tritonrmsnorm2d"),
]:
    create_norm._NORM_MAP[new_name] = create_norm._NORM_MAP[orig_name]


def get_same_padding(kernel_size: int | tuple[int, ...]) -> int | tuple[int, ...]:
    if isinstance(kernel_size, (tuple, list)):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


# basic conv layer class
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
        norm: Optional[str] = "bn2d",
        act_func: Optional[str] = "relu",
    ):
        super(ConvLayer, self).__init__()

        padding = get_same_padding(kernel_size)
        padding *= dilation

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None

        # pad same
        self.conv = create_conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            bias=use_bias,
            groups=groups,
            dilation=dilation,
            padding=padding,
        )

        self.norm = create_norm_layer(norm, out_channels) if norm else None
        self.act = create_act_layer(act_func) if act_func else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class LinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias=True,
        dropout=0,
        norm=None,
        act_func=None,
        try_squeeze=True,
    ):
        super(LinearLayer, self).__init__()

        self.dropout = nn.Dropout(dropout, inplace=False) if dropout > 0 else None
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.norm = create_norm_layer(norm, num_features=out_features) if norm else None
        self.act = create_act_layer(act_func)
        self.try_squeeze = try_squeeze

    def _try_squeeze(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.try_squeeze:
            x = self._try_squeeze(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.linear(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


# * --- MB layers --- #


# Depthwise Separable Convolution
class DSConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super(DSConv, self).__init__()

        use_bias = to_2tuple(use_bias)
        norm = to_2tuple(norm)
        act_func = to_2tuple(act_func)

        self.depth_conv = ConvLayer(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.point_conv = ConvLayer(
            in_channels,
            out_channels,
            1,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        use_bias=False,
        norm=("bn2d", "bn2d", "bn2d"),
        act_func=("relu6", "relu6", None),
    ):
        super(MBConv, self).__init__()

        use_bias = to_2tuple(use_bias)
        norm = to_2tuple(norm)
        act_func = to_2tuple(act_func)
        mid_channels = (
            round(in_channels * expand_ratio) if mid_channels is None else mid_channels
        )

        self.inverted_conv = ConvLayer(
            in_channels,
            mid_channels,
            1,
            stride=1,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.depth_conv = ConvLayer(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=mid_channels,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            norm=norm[2],
            act_func=act_func[2],
            use_bias=use_bias[2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class FusedMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        groups=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super().__init__()
        use_bias = to_2tuple(use_bias)
        norm = to_2tuple(norm)
        act_func = to_2tuple(act_func)

        mid_channels = (
            round(in_channels * expand_ratio) if mid_channels is None else mid_channels
        )

        self.spatial_conv = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            groups=groups,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_conv(x)
        x = self.point_conv(x)
        return x


class GLUMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels: Optional[int] = None,
        expand_ratio: float = 6,
        use_bias: bool | tuple = False,
        norm=(None, None, "ln2d"),
        act_func=("silu", "silu", None),
    ):
        super().__init__()
        use_bias = to_2tuple(use_bias)
        norm = to_2tuple(norm)
        act_func = to_2tuple(act_func)

        mid_channels = (
            round(in_channels * expand_ratio) if mid_channels is None else mid_channels
        )

        self.glu_act = create_act_layer(act_func[1])
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


class GLUMBConv1D(GLUMBConv):
    def forward(
        self, x: torch.Tensor, HW: Optional[tuple[int, int]] = None
    ) -> torch.Tensor:
        B, N, C = x.shape
        if HW is None:
            H = W = int(N**0.5)
        else:
            H, W = HW

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = super().forward(x)
        x = x.reshape(B, C, N).permute(0, 2, 1)

        return x


# * --- Submodules --- #


class SEModule_(nn.Module):
    "https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/senet.py"

    def __init__(self, channels: int, reduction: int):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.silu = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x)
        x = self.silu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class CoordAttnModule_(nn.Module):
    "https://github.com/houqb/CoordAttention/blob/main/mbv2_ca.py"

    def __init__(self, inp, oup, groups=4):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.norm = create_norm_layer("tritonrmsnorm2d", mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.norm(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = identity * x_w * x_h

        return y


# * --- Resblock --- #


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        mid_channels: Optional[int] = None,
        expand_ratio: float = 1,
        use_bias: bool | tuple = False,
        norm: tuple[Optional[str], ...] = ("bn2d", "bn2d"),
        act_func: tuple[Optional[str], ...] = ("relu6", None),
    ):
        super().__init__()
        use_bias = to_2tuple(use_bias)
        norm = to_2tuple(norm)
        act_func = to_2tuple(act_func)

        mid_channels = (
            round(in_channels * expand_ratio) if mid_channels is None else mid_channels
        )

        self.conv1 = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.conv2 = ConvLayer(
            mid_channels,
            out_channels,
            kernel_size,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResBlockCondition(ResBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        mid_channels: Optional[int] = None,
        expand_ratio: float = 1,
        use_bias: bool | tuple = False,
        norm: tuple[Optional[str], ...] = ("bn2d", "bn2d"),
        act_func: tuple[Optional[str], ...] = ("relu6", None),
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            mid_channels,
            expand_ratio,
            use_bias,
            norm,
            act_func,
        )
        mid_channels = (
            round(in_channels * expand_ratio) if mid_channels is None else mid_channels
        )
        cond_layer = nn.Sequential(
            create_conv2d(cond_channels, out_channels, kernel_size=1),
            create_norm_act_layer(
                "layernorm2d", out_channels, act_layer="silu", eps=1e-6
            ),
            create_conv2d(
                out_channels, out_channels * 3, kernel_size=3, groups=out_channels
            ),
        )
        self.conv2 = ConditionalBlock(
            self.conv2,
            cond_layer,
            condition_types="modulate_3",
            process_cond_before="interpolate_as_x",
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x, cond)
        return x


class GLUResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        gate_kernel_size: int = 3,
        stride: int = 1,
        mid_channels: Optional[int] = None,
        expand_ratio: float = 1,
        use_bias: bool | tuple = False,
        norm: tuple[Optional[str], ...] = (None, "trms2d"),
        act_func: tuple[Optional[str], ...] = ("silu", None),
    ):
        super().__init__()
        use_bias = to_3tuple(use_bias)
        norm = to_3tuple(norm)
        act_func = to_3tuple(act_func)

        mid_channels = (
            round(in_channels * expand_ratio) if mid_channels is None else mid_channels
        )

        self.conv1 = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=None,
        )
        self.conv_gate = ConvLayer(
            in_channels,
            mid_channels,
            gate_kernel_size,
            stride,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.conv2 = ConvLayer(
            mid_channels,
            out_channels,
            kernel_size,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x) * self.conv_gate(x)
        x = self.conv2(x)
        return x


class ChannelAttentionResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        mid_channels: Optional[int] = None,
        expand_ratio: float = 1,
        use_bias: bool | tuple = False,
        norm: tuple[Optional[str], ...] = ("bn2d", "bn2d"),
        act_func: tuple[Optional[str], ...] = ("relu6", None),
        channel_attention_operation: str = "SEModule",
        channel_attention_position: int = 2,
    ):
        super().__init__()
        use_bias = to_2tuple(use_bias)
        norm = to_2tuple(norm)
        act_func = to_2tuple(act_func)

        mid_channels = (
            round(in_channels * expand_ratio) if mid_channels is None else mid_channels
        )

        self.conv1 = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.conv2 = ConvLayer(
            mid_channels,
            out_channels,
            kernel_size,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )
        if channel_attention_operation == "SEModule":
            self.channel_attention = SEModule_(out_channels, reduction=4)
        elif channel_attention_operation == "CoordAttnModule":
            self.channel_attention = CoordAttnModule_(
                out_channels, out_channels, groups=4
            )
        else:
            raise ValueError(
                f"channel_attention_operation {channel_attention_operation} is not supported"
            )
        self.channel_attention_position = channel_attention_position

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        if self.channel_attention_position == 1:
            x = self.channel_attention(x)
        x = self.conv2(x)
        if self.channel_attention_position == 2:
            x = self.channel_attention(x)
        return x


if __name__ == "__main__":
    """
    python -m src.stage2.layers.conv
    """
    cond_blk = ResBlockCondition(64, 64, 32)
    x = torch.randn(2, 64, 32, 32)
    cond = torch.randn(2, 32, 16, 16)
    y = cond_blk(x, cond)
    print(y.shape)

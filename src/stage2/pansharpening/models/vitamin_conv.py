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
        self.stride, self.in_chs, self.out_chs = stride, in_chs, out_chs
        mid_chs = make_divisible(out_chs * expand_ratio)
        # breakpoint()
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

    def init_weights(self, scheme=""):
        named_apply(partial(_init_conv, scheme=scheme), self)

    def forward(self, x, cond):
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


class MbConvStages(nn.Module):
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

        self.stem = Stem(
            in_chs=in_chans,
            out_chs=stem_width,
        )
        stages = {}
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
            stages[f"stage_{s}"] = nn.ModuleList(blocks)
        self.stages = nn.ModuleDict(stages)

    def forward(self, x, cond):
        x = self.stem(x)

        # interpolate the condition
        cond = th.nn.functional.interpolate(
            cond, size=x.shape[2:], mode="bilinear", align_corners=False
        )

        # stages
        for stage in self.stages.values():
            for block in stage:
                if self.grad_checkpointing and self.training:
                    x = checkpoint(block, x, cond, use_reentrant=False)
                else:
                    x = block(x, cond)

        return x


@dataclass
class ConvCfg:
    expand_ratio: float = 4.0
    expand_output: bool = (
        True  # calculate expansion channels from output (vs input chs)
    )
    kernel_size: int = 3
    group_size: int = 1  # 1 == depthwise
    pre_norm_act: bool = False  # activation after pre-norm
    stride_mode: str = "dw"  # stride done via one of 'pool', '1x1', 'dw'
    pool_type: str = "avg2"
    downsample_pool_type: str = "avg2"
    act_layer: str = "gelu"  # stem & stage 1234
    norm_layer: str = ""
    norm_eps: float = 1e-5
    down_shortcut: bool = True
    mlp: str = "mlp"


@dataclass
class VitaminCfg:
    stem_width: int
    embed_dim: list[int]
    depths: list[int]
    pan_channel: int = 1
    ms_channel: int = 8
    condition_channel: int = 256
    use_residual: bool = False
    conv_cfg: ConvCfg = field(default_factory=ConvCfg)


def create_conv3x3_same(in_chan, out_chan):
    return create_conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1)


class VitaminModel(nn.Module):
    def __init__(self, cfg: VitaminCfg):
        super().__init__()
        self.cfg = cfg

        patchers = nn.ModuleDict()
        patchers["pan_conv"] = create_conv3x3_same(cfg.pan_channel, cfg.stem_width)
        patchers["ms_conv"] = create_conv3x3_same(cfg.ms_channel, cfg.stem_width)
        patchers["fused_conv"] = create_conv3x3_same(cfg.stem_width * 2, cfg.stem_width)
        patchers["condition_conv"] = nn.Sequential(
            create_norm_layer(cfg.conv_cfg.norm_layer, cfg.condition_channel),
            create_conv3x3_same(cfg.condition_channel, cfg.stem_width),
        )
        self.patchers = patchers

        self.stages = MbConvStages(
            cfg.stem_width,
            cfg.stem_width,
            cfg.embed_dim,
            cfg.depths,
            cond_width=cfg.condition_channel,
            **asdict(cfg.conv_cfg),
        )
        self.out_conv = create_conv3x3_same(cfg.embed_dim[-1], cfg.ms_channel)

    def forward(self, ms, pan, cond):
        x_ms = self.patchers["ms_conv"](ms)
        x_pan = self.patchers["pan_conv"](pan)
        x = th.cat([x_ms, x_pan], dim=1)
        x = self.patchers["fused_conv"](x)
        cond = self.patchers["condition_conv"](cond)

        # stages
        x = self.stages(x, cond)
        x = self.out_conv(x)

        if self.cfg.use_residual:
            x = x + ms

        return x


if __name__ == "__main__":
    import torch

    # Example usage of ConvCfg dataclass
    conv_cfg = ConvCfg(
        expand_ratio=2.0, kernel_size=3, act_layer="gelu", norm_layer="layernorm2d"
    )
    print("ConvCfg example:", conv_cfg)

    # Example usage of VitaminCfg dataclass
    vitamin_cfg = VitaminCfg(
        stem_width=32,
        embed_dim=[64, 192, 192],
        depths=[2, 2, 2],
        pan_channel=1,
        ms_channel=8,
        condition_channel=256,
        use_residual=True,
        conv_cfg=conv_cfg,
    )
    print("VitaminCfg example:", vitamin_cfg)

    # Example usage of VitaminModel
    model = VitaminModel(vitamin_cfg)

    # Create sample inputs
    batch_size, height, width = 2, 64, 64
    ms = torch.randn(batch_size, vitamin_cfg.ms_channel, height, width)
    pan = torch.randn(batch_size, vitamin_cfg.pan_channel, height, width)
    cond = torch.randn(batch_size, vitamin_cfg.condition_channel, height, width)

    # Forward pass
    with torch.no_grad():
        output = model(ms, pan, cond)
        print(f"Input shape: MS {ms.shape}, PAN {pan.shape}, Cond {cond.shape}")
        print(f"Output shape: {output.shape}")
        print("Model output computed successfully!")

    # Print infos of the network
    from fvcore.nn import parameter_count_table

    print(parameter_count_table(model, max_depth=2))

from dataclasses import asdict, dataclass, field

import torch as th
import torch.nn as nn
from timm.layers.create_conv2d import create_conv2d
from timm.layers.create_norm import create_norm_layer

from src.stage2.layers import MbConvStages


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
    norm_layer: str = "layernorm2d"
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

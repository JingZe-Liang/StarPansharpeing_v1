from dataclasses import dataclass, field
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.create_conv2d import create_conv2d
from timm.layers.create_norm import create_norm_layer, get_norm_layer

from src.stage2.layers.naf import NAFBlock, NAFBlockConditional
from src.stage2.layers.patcher import create_patcher, create_unpatcher
from src.utilities.config_utils import dataclass_from_dict
from src.utilities.logging import log


@dataclass
class PansharpeningNAFNetConfig:
    pan_channel: int = 1
    ms_channel: int = 8
    width: int = 256
    middle_blk_num: int = 1
    enc_blk_nums: list = field(default_factory=lambda: [1, 1, 1, 1])
    dec_blk_nums: list = field(default_factory=lambda: [1, 1, 1, 1])
    condition_channel: int = 16
    use_residual: bool = True
    dw_expand: int = 2
    ffn_expand: int = 2
    patch_size: int = 1
    condition_on_decoder: bool = False
    block_drop: float = 0.0
    output_rescale: bool = True
    is_neg_1_1: bool = True


class PansharpeningNAFNet(nn.Module):
    def __init__(self, cfg: PansharpeningNAFNetConfig):
        super().__init__()
        patchers = nn.ModuleDict()
        patchers["pan_conv"] = create_patcher(
            cfg.pan_channel, cfg.width, cfg.patch_size
        )
        patchers["ms_conv"] = create_patcher(cfg.ms_channel, cfg.width, cfg.patch_size)
        patchers["fused_conv"] = create_patcher(cfg.width * 2, cfg.width, 1)
        patchers["condition_conv"] = nn.Sequential(
            create_norm_layer("layernorm2d", cfg.condition_channel),
            create_conv2d(
                cfg.condition_channel, cfg.width, 3, stride=1, padding=1, bias=True
            ),
        )
        self.cfg = cfg
        self.patchers = patchers
        self.unpatcher = create_unpatcher(cfg.width, cfg.width, cfg.patch_size)
        self.head = self._create_head()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = cfg.width
        enc_block_partial = partial(
            NAFBlockConditional,
            cond_chs=cfg.width,
            DW_Expand=cfg.dw_expand,
            FFN_Expand=cfg.ffn_expand,
            drop_out_rate=cfg.block_drop,
        )
        dec_block_partial = (
            partial(
                NAFBlock,
                DW_Expand=cfg.dw_expand,
                FFN_Expand=cfg.ffn_expand,
                drop_out_rate=cfg.block_drop,
            )
            if not cfg.condition_on_decoder
            else enc_block_partial
        )
        for num in cfg.enc_blk_nums:
            self.encoders.append(
                nn.ModuleList([enc_block_partial(chan) for _ in range(num)])
            )
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.ModuleList(
            [dec_block_partial(chan) for _ in range(cfg.middle_blk_num)]
        )

        for num in cfg.dec_blk_nums:
            self.ups.append(
                nn.ConvTranspose2d(chan, chan // 2, 2, 2)
                # nn.Sequential(
                #     nn.Conv2d(chan, chan * 2, 1, bias=False),
                #     nn.PixelShuffle(2),
                # )
            )
            chan = chan // 2
            self.decoders.append(
                nn.ModuleList([dec_block_partial(chan) for _ in range(num)])
            )

        self.padder_size = 2 ** len(self.encoders)

    def _create_head(self):
        out_chan = self.cfg.ms_channel
        embed_dim = self.cfg.width
        head_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1),
        )
        head_out = nn.Conv2d(embed_dim, out_chan, 1)
        return nn.Sequential(head_conv, head_out)

    def _patching_inputs(self, ms, pan, cond):
        x_ms = self.patchers["ms_conv"](ms)
        x_pan = self.patchers["pan_conv"](pan)
        x = torch.cat([x_ms, x_pan], dim=1)
        x = self.patchers["fused_conv"](x)
        cond = self.patchers["condition_conv"](cond)
        return x, cond

    def forward(self, ms, pan, cond):
        B, C, H, W = ms.shape
        x, cond = self._patching_inputs(ms, pan, cond)

        # backbone
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            for block in encoder:
                x = block(x, cond)
            encs.append(x)
            x = down(x)

        for block in self.middle_blks:
            x = block(x, cond)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            for block in decoder:
                x = block(x, cond)

        # unpatcher
        x = self.unpatcher(x)

        # heads
        x = self.head(x)

        if self.cfg.use_residual:
            x = x + ms

        if self.cfg.output_rescale:
            if self.cfg.is_neg_1_1:
                x = torch.tanh(x)
            else:
                x = torch.sigmoid(x)
        return x

    @classmethod
    def create_model(cls, **kwargs):
        cfg = dataclass_from_dict(PansharpeningNAFNetConfig, kwargs)
        model = cls(cfg)
        return model


def test_pansharpening_nafnet():
    """Test PansharpeningNAFNet model with sample inputs."""
    # Create model with valid configuration
    model = PansharpeningNAFNet.create_model(
        ms_channel=8,
        pan_channel=1,
        condition_channel=8,
        width=64,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1],
        dec_blk_nums=[1, 1, 1],
        use_residual=True,
        patch_size=2,
        dw_expand=1,
        ffn_expand=1,
    )
    model.train()

    from fvcore.nn import parameter_count_table

    print(parameter_count_table(model))

    # Create sample inputs
    ms = torch.randn(1, 8, 256, 256)
    pan = torch.randn(1, 1, 256, 256)
    # cond = torch.randn(1, 16, 32, 32)
    cond = torch.randn(1, 8, 256, 256)

    # Test forward pass
    output = model(ms, pan, cond)

    # Verify output shape
    assert output.shape == ms.shape, (
        f"Expected output shape {ms.shape}, got {output.shape}"
    )

    # Test inference mode
    model.eval()
    with torch.no_grad():
        output_eval = model(ms, pan, cond)

    # Verify evaluation output shape
    assert output_eval.shape == ms.shape, (
        f"Expected eval output shape {ms.shape}, got {output_eval.shape}"
    )

    print("✓ PansharpeningNAFNet test passed!")
    print(f"Input MS shape: {ms.shape}")
    print(f"Input PAN shape: {pan.shape}")
    print(f"Input condition shape: {cond.shape}")
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    test_pansharpening_nafnet()

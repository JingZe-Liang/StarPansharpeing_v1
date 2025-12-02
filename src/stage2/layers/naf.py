# NAFNet compatibility
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.create_conv2d import create_conv2d
from timm.layers.create_norm import create_norm_layer, get_norm_layer
from timm.layers.create_norm_act import create_norm_act_layer
from timm.layers.patch_embed import PatchEmbed
from torch.utils.checkpoint import checkpoint

from src.utilities.config_utils import dataclass_from_dict
from src.utilities.logging import log

from .cross_attn import CrossAttention
from .patcher import create_unpatcher


def modulate(x, shift, scale):
    return x * (scale + 1) + shift


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0, **__discarded_kwargs):
        super().__init__()
        self.grad_checkpointing = False
        dw_channel = c * DW_Expand
        self.conv1 = create_conv2d(c, dw_channel, 1)
        self.conv2 = create_conv2d(dw_channel, dw_channel, 3, stride=1, padding=1, depthwise=True)
        self.conv3 = create_conv2d(dw_channel // 2, c, 1)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            create_conv2d(dw_channel // 2, dw_channel // 2, 1),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = create_conv2d(c, ffn_channel, 1)
        self.conv5 = create_conv2d(ffn_channel // 2, c, 3, stride=1, padding=1)

        self.norm1 = create_norm_layer("layernorm2d", c)
        self.norm2 = create_norm_layer("layernorm2d", c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()

        self.beta = nn.Parameter(1e-4 * torch.ones((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(1e-4 * torch.ones((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp, *args, **kwargs):
        def forward_closure(inp):
            x = inp
            x = self.norm1(x)
            
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.sg(x)
            x = x * self.sca(x)
            x = self.conv3(x)
            x = self.dropout1(x)

            y = inp + x * self.beta

            x = self.conv4(self.norm2(y))
            x = self.sg(x)
            x = self.conv5(x)
            x = self.dropout2(x)

            return y + x * self.gamma

        if self.grad_checkpointing and self.training:
            return checkpoint(forward_closure, inp, use_reentrant=False)
        else:
            return forward_closure(inp)

    def init_weights(self):
        """Initialize weights following the original NAFNet implementation."""
        convs = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]
        norms = [self.norm1, self.norm2]

        with torch.no_grad():
            for c in convs:
                nn.init.kaiming_normal_(c.weight)
                if c.bias is not None:
                    nn.init.zeros_(c.bias)
            for n in norms:
                nn.init.ones_(n.weight)
                if getattr(n, "bias", None) is not None:
                    nn.init.zeros_(n.bias)


class NAFBlockConditional(NAFBlock):
    def __init__(
        self,
        c,
        cond_chs=256,
        DW_Expand=2,
        FFN_Expand=2,
        drop_out_rate=0.0,
        ms_cond_chans=None,
        **__discarded_kwargs,
    ):
        super().__init__(c, DW_Expand, FFN_Expand, drop_out_rate)
        ffn_channel = FFN_Expand * c
        if ms_cond_chans is not None:
            ms_cond_chans = cond_chs
            self.ms_conv_before_add = create_conv2d(ms_cond_chans, c, 3)
        # lora-like modulation
        self.modulation = nn.Sequential(
            create_conv2d(cond_chs, ffn_channel // 2, 1, bias=True),
            create_norm_act_layer("layernorm2d", ffn_channel // 2, act_layer="silu", eps=1e-6),
            create_conv2d(
                ffn_channel // 2,
                ffn_channel * 2,
                3,
                stride=1,
                bias=False,
                groups=ffn_channel // 2,
            ),
        )

    def forward(self, inp, mod_cond, ms_cond=None):
        """input, latent_modulation, multispectral_add_condition"""

        def forward_closure(inp, latent, ms_cond=None):
            x = inp

            # Add multispectral condition
            if hasattr(self, "ms_conv_before_add") and ms_cond is not None:
                ms_cond = F.interpolate(ms_cond, size=x.shape[2:], mode="bilinear", align_corners=False)
                ms_cond = self.ms_conv_before_add(ms_cond)
                x = x + ms_cond

            # Conv stage 1
            x = self.norm1(x)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.sg(x)
            x = x * self.sca(x)
            x = self.conv3(x)
            x = self.dropout1(x)
            y = inp + x * self.beta

            # Latent modulation
            l_cond = F.interpolate(latent, size=y.shape[-2:], mode="bilinear", align_corners=False)
            scale, shift = self.modulation(l_cond).chunk(2, dim=1)

            # Conv stage 2
            x = self.conv4(self.norm2(y))
            x = modulate(x, shift, scale)
            x = self.sg(x)
            x = self.conv5(x)
            x = self.dropout2(x)

            return y + x * self.gamma

        if self.grad_checkpointing and self.training:
            return checkpoint(forward_closure, inp, mod_cond, ms_cond, use_reentrant=False)
        else:
            return forward_closure(inp, mod_cond, ms_cond)


class NAFCrossAttentionConditional(NAFBlock):
    def __init__(
        self,
        c,
        cond_chs=256,
        DW_Expand=2,
        FFN_Expand=2,
        drop_out_rate=0.0,
        cross_attn: bool = False,
        cross_attn_kwargs: dict = dict(
            n_q_heads=8,
            n_kv_heads=8,
            qk_norm="zero_mean_rmsnorm",
            use_gate=True,
            cross_attn_patch_size=2,
        ),
        **__discarded_kwargs,
    ):
        super().__init__(c, DW_Expand, FFN_Expand, drop_out_rate)
        ffn_channel = FFN_Expand * c
        # lora-like modulation
        self.modulation = nn.Sequential(
            create_conv2d(cond_chs, ffn_channel // 2, 1, bias=True),
            create_norm_act_layer("layernorm2d", ffn_channel // 2, act_layer="silu", eps=1e-6),
            create_conv2d(
                ffn_channel // 2,
                ffn_channel * 2,
                3,
                stride=1,
                bias=False,
                groups=ffn_channel // 2,
            ),
        )
        self.use_cross_attn = cross_attn
        if cross_attn:
            ps = cross_attn_kwargs.pop("cross_attn_patch_size", 2)
            self.ca_patch_size = ps
            self.cond_to_x_dim = create_conv2d(cond_chs, c, 1, bias=True)
            self.cross_attn = CrossAttention(dim=c, **cross_attn_kwargs if cross_attn_kwargs else {})
            self.ca_q_patcher = PatchEmbed(
                patch_size=ps,
                norm_layer=get_norm_layer("layernorm"),
                in_chans=c,
                embed_dim=c,
                output_fmt="NLC",
                strict_img_size=False,
            )
            self.ca_q_unpatcher = create_unpatcher(c, c, ps, depthwise=True)

    def _forward_cross_attn(self, x, cond):
        B, C, H, W = x.shape
        qh, qw = H // self.ca_patch_size, W // self.ca_patch_size
        x_in = x
        q = self.ca_q_patcher(x)  # (B, Hq*Wq, C)
        cond = self.cond_to_x_dim(cond)
        cond = cond.flatten(2).transpose(1, 2)  # (B, Hc*Wc, C)
        q = self.cross_attn(q, cond)
        q = q.transpose(1, 2).view(B, C, qh, qw)
        x = self.ca_q_unpatcher(q)
        x = x + x_in
        return x

    def forward(self, inp, cond_modulate, cond_cross_attn):
        def forward_closure(inp, cond_modulate, cond_cross_attn):
            x = inp
            # Conv stage 1
            x = self.norm1(x)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.sg(x)
            x = x * self.sca(x)
            x = self.conv3(x)
            x = self.dropout1(x)
            y = inp + x * self.beta

            # latent cross-attention
            if self.use_cross_attn:
                y = self._forward_cross_attn(y, cond_cross_attn)

            # ms modulation
            cond_modulate = F.interpolate(cond_modulate, size=y.shape[-2:], mode="bilinear", align_corners=False)
            scale, shift = self.modulation(cond_modulate).chunk(2, dim=1)

            # Conv stage 2
            x = self.conv4(self.norm2(y))
            x = modulate(x, shift, scale)
            x = self.sg(x)
            x = self.conv5(x)
            x = self.dropout2(x)

            return y + x * self.gamma

        if self.grad_checkpointing and self.training:
            return checkpoint(
                forward_closure,
                inp,
                cond_modulate,
                cond_cross_attn,
                use_reentrant=False,
            )
        else:
            return forward_closure(inp, cond_modulate, cond_cross_attn)


@dataclass
class NAFNetConfig:
    img_channel: int = 3
    width: int = 256
    middle_blk_num: int = 1
    enc_blk_nums: list = field(default_factory=lambda: [1, 1, 1, 1])
    dec_blk_nums: list = field(default_factory=lambda: [1, 1, 1, 1])


class NAFNet(nn.Module):
    def __init__(self, cfg: NAFNetConfig):
        super().__init__()
        self.intro = nn.Conv2d(
            in_channels=cfg.img_channel,
            out_channels=cfg.width,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )
        self.ending = create_conv2d(cfg.width, cfg.img_channel, 3, padding=1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = cfg.width
        for num in cfg.enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(cfg.middle_blk_num)])

        for num in cfg.dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x

    @classmethod
    def create_model(cls, **kwargs):
        cfg = dataclass_from_dict(NAFNetConfig, kwargs)
        model = cls(cfg)
        return model


def test_model():
    model = NAFNet.create_model(
        img_channel=4,
        width=64,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1, 1],
        dec_blk_nums=[1, 1, 1, 1],
    )
    x = torch.randn(2, 4, 256, 256)
    with torch.no_grad():
        y = model(x)
    print(y.shape)
    assert y.shape == x.shape


if __name__ == "__main__":
    test_model()

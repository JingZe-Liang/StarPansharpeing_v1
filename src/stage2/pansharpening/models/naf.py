import math
from dataclasses import dataclass, field, fields, replace
from functools import partial
from typing import Annotated

import black
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.create_conv2d import create_conv2d
from timm.layers.create_norm import create_norm_layer, get_norm_layer
from timm.layers.weight_init import lecun_normal_
from timm.models._builder import build_model_with_cfg
from timm.models._manipulate import named_apply
from timm.models._registry import register_model
from torch import Tensor

from src.stage2.layers.naf import (
    NAFBlock,
    NAFBlockConditional,
    NAFCrossAttentionConditional,
)
from src.stage2.layers.patcher import create_patcher, create_unpatcher
from src.stage2.layers.rescale import RescaleOutput, ValueRange
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
    condition_on_decoder: bool = True
    block_drop: float = 0.0
    output_rescale: bool = True
    is_neg_1_1: bool = True
    residual_type: str = "ms"


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
            create_conv2d(cfg.condition_channel, cfg.width, 3, bias=True),
        )
        self.cfg = cfg
        self.patchers = patchers
        self.unpatcher = create_unpatcher(cfg.width, cfg.width, cfg.patch_size)
        self.head = self._create_head()

        # Add output rescaling layer
        if cfg.output_rescale:
            out_val_range = (
                ValueRange.MINUS_ONE_ONE if cfg.is_neg_1_1 else ValueRange.ZERO_ONE
            )
            self.output_rescale = RescaleOutput(
                rescale=True, out_val_range=out_val_range
            )
        else:
            self.output_rescale = nn.Identity()  # do nothing

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
                # nn.ConvTranspose2d(chan, chan // 2, 2, 2)
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.ModuleList([dec_block_partial(chan) for _ in range(num)])
            )

        self.padder_size = 2 ** len(self.encoders)

        self._init_weights()

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

    def forward(
        self,
        ms,
        pan,
        cond: Annotated[Tensor, "latents or decoded images"],
    ):
        B, C, H, W = ms.shape
        x, cond_ = self._patching_inputs(ms, pan, cond)

        # backbone
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            for block in encoder:
                x = block(x, cond_)
            encs.append(x)
            x = down(x)

        for block in self.middle_blks:
            x = block(x, cond_)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            for block in decoder:
                x = block(x, cond_)

        # unpatcher
        x = self.unpatcher(x)

        # heads
        x = self.head(x)

        res_type = self.cfg.residual_type
        if res_type == "condition":
            assert cond.shape[1] == x.shape[1], (
                f"Condition channel {cond.shape[1]} must match output channel {x.shape[1]}"
            )
            x = x + cond
        elif res_type == "ms":
            assert ms.shape[-2:] == x.shape[-2:]
            x = x + ms
        elif res_type == "pan":
            assert pan.shape[1] == 1
            x = x + pan

        # Apply output rescaling using RescaleOutput layer
        x = self.output_rescale(x)
        return x

    @classmethod
    def create_model(cls, **kwargs):
        cfg = dataclass_from_dict(PansharpeningNAFNetConfig, kwargs)
        model = cls(cfg)
        return model

    def _init_weights(self):
        def _apply(module):
            if isinstance(module, nn.Conv2d):
                # nn.init.kaiming_normal_(module.weight, mode="fan_in")
                lecun_normal_(module.weight)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # init weights
        self.apply(_apply)
        # zeros out head
        nn.init.zeros_(self.head[1].weight)
        nn.init.zeros_(self.head[1].bias)

        log("[PansharpeningNAFNet] Initialized weights")


# *==============================================================
# * Decoder only NAFNet for pansharpening
# *==============================================================


@dataclass
class PansharpeningNAFDecoderNetConfig:
    pan_channel: int = 1
    ms_channel: int = 8
    width: int = 256
    middle_blk_num: int = 1
    dec_blk_nums: list = field(default_factory=lambda: [1, 1, 1, 1])
    condition_channel: int = 16
    residual_type: str = "none"  # ms, pan, condition, none
    dw_expand: int = 2
    ffn_expand: int = 2
    patch_size: int = 1
    block_drop: float = 0.0
    output_rescale: bool = True
    is_neg_1_1: bool = True
    up_ratio: int = 4
    decoder_cross_attn_idx: list = field(default_factory=lambda: [0, 1])
    cross_attn_kwargs: dict = field(
        default_factory=lambda: {
            "n_q_heads": 4,
            "n_kv_heads": 4,
            "cross_attn_patch_size": 2,
        }
    )


class PansharpeningNAFDecoderNet(nn.Module):
    def __init__(self, cfg: PansharpeningNAFDecoderNetConfig):
        super().__init__()
        patchers = nn.ModuleDict()
        patchers["pan_conv"] = create_patcher(
            cfg.pan_channel, cfg.width, cfg.patch_size
        )
        patchers["ms_conv"] = create_patcher(cfg.ms_channel, cfg.width, cfg.patch_size)
        patchers["condition_conv"] = nn.Sequential(
            create_norm_layer("layernorm2d", cfg.condition_channel),
            create_conv2d(
                cfg.condition_channel, cfg.width, 3, stride=1, padding=1, bias=True
            ),
        )
        self.cfg = cfg
        self.n_up_most = math.log2(cfg.up_ratio)
        self.patchers = patchers
        dec_out_dim = cfg.width // (2 ** len(self.cfg.dec_blk_nums))
        self.unpatcher = create_unpatcher(dec_out_dim, dec_out_dim, cfg.patch_size)
        self.head = self._create_head(dec_out_dim)

        # Add output rescaling layer
        if cfg.output_rescale:
            out_val_range = (
                ValueRange.MINUS_ONE_ONE if cfg.is_neg_1_1 else ValueRange.ZERO_ONE
            )
            self.output_rescale = RescaleOutput(
                rescale=True, out_val_range=out_val_range
            )
        else:
            self.output_rescale = nn.Identity()  # do nothing

        self.middle_blks = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()

        chan = cfg.width
        dec_block_partial = partial(
            NAFCrossAttentionConditional,
            DW_Expand=cfg.dw_expand,
            FFN_Expand=cfg.ffn_expand,
            drop_out_rate=cfg.block_drop,
            cond_chs=chan,
            cross_attn_kwargs=cfg.cross_attn_kwargs,
        )
        # Middle blocks
        self.middle_blks = nn.ModuleList(
            [
                dec_block_partial(chan, cross_attn=True)
                for _ in range(cfg.middle_blk_num)
            ]
        )
        # Decoder blocks
        for layer_idx, num in enumerate(cfg.dec_blk_nums):
            if layer_idx < self.n_up_most:
                self.ups.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode="nearest"),
                        nn.Conv2d(chan, chan // 2, 1, 1, bias=False),
                    )
                )
            else:
                self.ups.append(create_conv2d(chan, chan // 2, 3))
            chan = chan // 2
            self.decoders.append(
                nn.ModuleList(
                    [
                        dec_block_partial(
                            chan, cross_attn=layer_idx in cfg.decoder_cross_attn_idx
                        )
                        for _ in range(num)
                    ]
                )
            )

        self._init_weights()

    def _create_head(self, dec_out_dim: int):
        out_chan = self.cfg.ms_channel
        embed_dim = dec_out_dim
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
        if ms.shape[-1] == pan.shape[-1]:
            # dissolve the MS upsampled low-quality issue
            ms = F.interpolate(
                ms,
                scale_factor=1 / self.cfg.up_ratio,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        assert ms.shape[-1] == pan.shape[-1] // self.cfg.up_ratio, (
            f"MS width {ms.shape[-1]} must be {self.cfg.up_ratio} times smaller "
            "than PAN width {pan.shape[-1]}"
        )

        x_ms = self.patchers["ms_conv"](ms)
        x_pan = self.patchers["pan_conv"](pan)
        cond = self.patchers["condition_conv"](cond)
        return x_ms, x_pan, cond

    def forward(
        self,
        ms,
        pan,
        cond: Annotated[Tensor, "latents or decoded images"],
    ):
        B, C, H, W = ms.shape
        x, x_pan, cond_attn = self._patching_inputs(ms, pan, cond)

        for block in self.middle_blks:
            x = block(x, x_pan, cond_attn)

        for decoder, up in zip(self.decoders, self.ups):
            x = up(x)
            for block in decoder:
                x = block(x, x_pan, cond_attn)

        # unpatcher
        x = self.unpatcher(x)

        # heads
        x = self.head(x)

        res_type = self.cfg.residual_type
        if res_type == "condition":
            assert cond.shape[1] == x.shape[1], (
                f"Condition channel {cond.shape[1]} must match output channel {x.shape[1]}"
            )
            x = x + cond
        elif res_type == "ms":
            assert ms.shape[-2:] == x.shape[-2:]
            x = x + ms
        elif res_type == "pan":
            assert pan.shape[1] == 1
            x = x + pan

        # Apply output rescaling using RescaleOutput layer
        x = self.output_rescale(x)
        return x

    @classmethod
    def create_model(cls, **kwargs):
        cfg = dataclass_from_dict(PansharpeningNAFDecoderNetConfig, kwargs)
        model = cls(cfg)
        return model

    def _init_weights(self):
        def _apply(module):
            if isinstance(module, nn.Conv2d):
                # nn.init.kaiming_normal_(module.weight, mode="fan_in")
                lecun_normal_(module.weight)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # init weights
        self.apply(_apply)
        # zeros out head
        nn.init.zeros_(self.head[1].weight)
        nn.init.zeros_(self.head[1].bias)

        log("[PansharpeningNAFDecoderNet] Initialized weights")


# *==============================================================
# * Timm register model
# *==============================================================


def _create_nafnet(variant: str, pretrained=False, **kwargs):
    log(f"Creating model {variant}, pretrained={pretrained}")
    if kwargs.get("pretrained_cfg", "") == "none":
        kwargs.setdefault("pretrained_strict", False)

    cfg_keys = [f.name for f in fields(PansharpeningNAFNetConfig)]
    cfg_args = {}
    for k in list(kwargs.keys()):
        if k in cfg_keys:
            cfg_args[k] = kwargs.pop(k)
    model_cfg = replace(PansharpeningNAFNetConfig(), **cfg_args)

    model = build_model_with_cfg(
        PansharpeningNAFNet,
        variant,
        pretrained,
        cfg=model_cfg,
        **kwargs,
    )
    return model


@register_model
def pansharpening_nafnet_small(pretrained=False, **kwargs):
    model_args = {
        "width": 64,
        "middle_blk_num": 1,
        "enc_blk_nums": [1, 1, 1],
        "dec_blk_nums": [1, 1, 1],
        "condition_channel": 256,  # comes from the de-tokenizer
        "patch_size": 1,
        "use_residual": True,
        "dw_expand": 1,
        "ffn_expand": 2,
        "is_neg_1_1": True,
        "output_rescale": True,
    }
    model = _create_nafnet(
        "pansharpening_nafnet_small",
        pretrained,
        **dict(model_args, **kwargs),
    )
    return model


def test_pansharpening_nafnet():
    """Test PansharpeningNAFNet model with sample inputs."""
    # Create model with valid configuration
    model = PansharpeningNAFNet.create_model(
        ms_channel=8,
        pan_channel=1,
        condition_channel=256,
        width=64,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1],
        dec_blk_nums=[1, 1, 1],
        use_residual=True,
        patch_size=2,
        dw_expand=1,
        ffn_expand=1,
        condition_on_decoder=True,
        residual_type="ms",
        is_neg_1_1=True,
        output_rescale=True,
    )
    model.train()

    from fvcore.nn import parameter_count_table

    print(parameter_count_table(model))

    # Create sample inputs
    ms = torch.randn(1, 8, 256, 256)
    pan = torch.randn(1, 1, 256, 256)
    cond = torch.randn(1, 256, 32, 32)  # latents
    # cond = torch.randn(1, 8, 256, 256)  # images

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


def test_pansharpening_naf_decoder_net():
    from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count_table

    # Create sample inputs
    bs = 4
    ms = torch.randn(bs, 8, 256, 256).cuda()
    pan = torch.randn(bs, 1, 256, 256).cuda()
    cond = torch.randn(bs, 16, 32, 32).cuda()  # latents
    # cond = torch.randn(1, 8, 256, 256)  # images

    cfg = PansharpeningNAFDecoderNetConfig(
        width=384,
        middle_blk_num=1,
        dec_blk_nums=[1, 1, 1],
        decoder_cross_attn_idx=[0],
    )
    model = PansharpeningNAFDecoderNet(cfg).cuda()
    model.train()

    print(flop_count_table(FlopCountAnalysis(model, (ms, pan, cond))))

    # optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    import heavyball

    optim = heavyball.ForeachAdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1.0e-3,
        betas=(0.9, 0.999),
        caution=True,
        foreach=True,
    )

    # Test forward pass
    from tqdm import trange

    for _ in trange(200):
        output = model(ms, pan, cond)
        output.mean().backward()
        # optim.step()
        # optim.zero_grad()

        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is None:
                print("name", n, "has not grad")

        # import time

        # time.sleep(0.1)


if __name__ == "__main__":
    test_pansharpening_nafnet()
    # test_pansharpening_naf_decoder_net()

    # import timm

    # print(timm.models.list_models("*naf*"))

    # model = timm.create_model(
    #     "pansharpening_nafnet_small",
    #     ms_channel=8,
    #     pan_channel=1,
    #     condition_channel=16,
    #     pretrained=False,
    # )
    # print(model)

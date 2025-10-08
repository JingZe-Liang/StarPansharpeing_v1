import math
from dataclasses import asdict, dataclass, field, fields, replace
from functools import partial
from typing import Annotated, Any, Optional

import numpy as np
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

from src.stage2.layers.blocks import Spatial2DNATBlock, Spatial2DNATBlockConditional
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
class NATBlockConfig:
    """Configuration for NAT block parameters."""

    n_heads: int = 8
    k_size: int = 8
    stride: int = 2
    dilation: int = 2


@dataclass
class DenoiseNAFNetConfig:
    lr_channels: int = 8
    width: int = 256
    middle_blk_num: int = 1
    middle_blk_type: str = "nat"
    enc_blk_nums: list = field(default_factory=lambda: [1, 1, 1, 1])
    dec_blk_nums: list = field(default_factory=lambda: [1, 1, 1, 1])
    enc_blk_type: list = field(default_factory=lambda: ["naf", "naf", "naf", "naf"])
    dec_blk_type: list = field(default_factory=lambda: ["naf", "naf", "naf", "naf"])
    condition_channel: int = 16  # latent channels
    use_residual: bool = True
    dw_expand: int = 2
    ffn_expand: int = 2
    patch_size: int = 1
    condition_on_decoder: bool = True
    block_drop: float = 0.0
    output_rescale: bool = True
    is_neg_1_1: bool = True
    residual_type: Optional[str] = "lr"
    nat_enc_kwargs: list[NATBlockConfig | None] = field(default_factory=lambda: [])
    nat_mid_kwargs: NATBlockConfig | None = None
    nat_dec_kwargs: list[NATBlockConfig | None] = field(default_factory=lambda: [])


class DenoiseNAFNet(nn.Module):
    """
    Denoise Network using NAFNet architecture with conditional blocks.
    """

    def __init__(self, cfg: DenoiseNAFNetConfig):
        super().__init__()

        # Validate configuration
        assert len(cfg.enc_blk_nums) == len(cfg.enc_blk_type), (
            f"enc_blk_nums length ({len(cfg.enc_blk_nums)}) must match "
            f"enc_blk_type length ({len(cfg.enc_blk_type)})"
        )
        assert len(cfg.dec_blk_nums) == len(cfg.dec_blk_type), (
            f"dec_blk_nums length ({len(cfg.dec_blk_nums)}) must match "
            f"dec_blk_type length ({len(cfg.dec_blk_type)})"
        )
        assert cfg.middle_blk_type in [
            "naf",
            "nat",
        ], f"middle_blk_type must be 'naf' or 'nat', got {cfg.middle_blk_type}"

        # =====================================================================
        # Input Processing Layers
        # =====================================================================

        # Patch embedding layers for different input modalities
        patchers = nn.ModuleDict()
        patchers["lr_conv"] = create_patcher(
            cfg.lr_channels, cfg.width, cfg.patch_size
        )  # LR image patch embedding
        patchers["latent_conv"] = nn.Sequential(
            create_norm_layer("layernorm2d", cfg.condition_channel),
            create_conv2d(cfg.condition_channel, cfg.width, 3, bias=True),
        )  # Latent condition processing

        self.cfg = cfg
        self.patchers = patchers
        self.unpatcher = create_unpatcher(
            cfg.width, cfg.width, cfg.patch_size
        )  # Output unpatching
        self.head = self._create_head()  # Output head

        # =====================================================================
        # Output Processing
        # =====================================================================

        # Add output rescaling layer for value range normalization
        if cfg.output_rescale:
            out_val_range = (
                ValueRange.MINUS_ONE_ONE if cfg.is_neg_1_1 else ValueRange.ZERO_ONE
            )
            self.output_rescale = RescaleOutput(
                rescale=True, out_val_range=out_val_range
            )
        else:
            self.output_rescale = nn.Identity()  # do nothing

        # =====================================================================
        # Network Architecture Components
        # =====================================================================

        # Main network components
        self.encoders = nn.ModuleList()  # Encoder blocks (downsampling path)
        self.decoders = nn.ModuleList()  # Decoder blocks (upsampling path)
        self.middle_blks = nn.ModuleList()  # Middle bottleneck blocks
        self.ups = nn.ModuleList()  # Upsampling layers
        self.downs = nn.ModuleList()  # Downsampling layers

        # Start with base channel dimension
        chan = cfg.width

        # =====================================================================
        # Block Configuration and Mapping
        # =====================================================================

        # Partial functions for different block types with shared parameters
        naf_enc_block_partial = partial(
            NAFBlockConditional,
            cond_chs=cfg.width,
            DW_Expand=cfg.dw_expand,
            FFN_Expand=cfg.ffn_expand,
            drop_out_rate=cfg.block_drop,
            ms_cond_chans=cfg.width,
        )  # NAF encoder block with conditional processing
        nat_enc_block_partial = partial(
            Spatial2DNATBlockConditional,
            cond_chs=cfg.width,
            ms_cond_chans=cfg.width,
            ffn_ratio=cfg.ffn_expand,
            drop_path=cfg.block_drop,
            # defaults
            k_size=4,
            stride=2,
            dilation=2,
            n_heads=8,
        )  # Neighborhood Attention encoder block with conditional processing
        naf_dec_block_partial = (
            partial(
                NAFBlock,
                DW_Expand=cfg.dw_expand,
                FFN_Expand=cfg.ffn_expand,
                drop_out_rate=cfg.block_drop,
            )
            if not cfg.condition_on_decoder
            else naf_enc_block_partial
        )  # NAF decoder block (conditional or unconditional)
        nat_dec_block_partial = (
            partial(
                Spatial2DNATBlock,
                dim=cfg.width,
                ffn_ratio=cfg.ffn_expand,
                drop_path=cfg.block_drop,
                k_size=4,
                stride=2,
                dilation=2,
                n_heads=8,
            )
            if not cfg.condition_on_decoder
            else nat_enc_block_partial
        )  # Neighborhood Attention decoder block (conditional or unconditional)

        # Block type mappings for dynamic architecture selection
        enc_mapping = {
            "naf": naf_enc_block_partial,
            "nat": nat_enc_block_partial,
        }  # Encoder block mapping
        dec_mapping = {
            "naf": naf_dec_block_partial,
            "nat": nat_dec_block_partial,
        }  # Decoder block mapping

        # =====================================================================
        # Encoder Path (Downsampling)
        # =====================================================================

        for i, num in enumerate(cfg.enc_blk_nums):
            enc_fn = enc_mapping[cfg.enc_blk_type[i]]
            log(f"[NAFNet Encoder Block {i}] Type: {enc_fn.func}, Num: {num}")
            enc_fn = self._replace_nat_defaults(enc_fn, cfg.nat_enc_kwargs, i)
            self.encoders.append(nn.ModuleList([enc_fn(chan) for _ in range(num)]))
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )  # Downsampling with stride 2
            chan = chan * 2  # Double channels after downsampling

        # =====================================================================
        # Middle Bottleneck Blocks
        # =====================================================================

        # Middle blocks using configurable type (highest resolution features)
        middle_mapping = {
            "naf": naf_enc_block_partial,
            "nat": nat_enc_block_partial,
        }
        middle_fn = middle_mapping[cfg.middle_blk_type]
        middle_fn = self._replace_nat_defaults(middle_fn, cfg.nat_mid_kwargs, None)
        log(f"[NAFNet Middile Block] Type: {middle_fn.func}, Num: {cfg.middle_blk_num}")
        self.middle_blks = nn.ModuleList(
            [middle_fn(chan) for _ in range(cfg.middle_blk_num)]
        )

        # =====================================================================
        # Decoder Path (Upsampling)
        # =====================================================================

        for i, num in enumerate(cfg.dec_blk_nums):
            # Upsampling using PixelShuffle (sub-pixel convolution)
            self.ups.append(
                # nn.ConvTranspose2d(chan, chan // 2, 2, 2)
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2),  # Upsample by factor of 2
                )
            )
            chan = chan // 2  # Halve channels after upsampling
            dec_fn = dec_mapping[cfg.dec_blk_type[i]]
            log(f"[NAFNet Decoder Block {i}] Type: {dec_fn.func}, Num: {num}")
            dec_fn = self._replace_nat_defaults(dec_fn, cfg.nat_dec_kwargs, i)
            self.decoders.append(nn.ModuleList([dec_fn(chan) for _ in range(num)]))

        # Calculate padding size for input processing
        self.padder_size = 2 ** len(self.encoders)

        self._init_weights()

    def _replace_nat_defaults(
        self,
        nat_partial: partial,
        nat_kwgs: list[NATBlockConfig | None] | NATBlockConfig | None,
        layer_idx: int | None = None,
    ):
        if nat_kwgs is None:
            return nat_partial

        # Handle list case
        if isinstance(nat_kwgs, list):
            if len(nat_kwgs) == 0:
                return nat_partial

            if layer_idx is not None:
                assert layer_idx < len(nat_kwgs), (
                    f"Layer index {layer_idx} exceeds provided NAT kwargs length {len(nat_kwgs)}: {nat_kwgs=}"
                )
                nat_kwgs = nat_kwgs[layer_idx]
            else:
                # If no layer_idx specified and it's a list, return unchanged
                return nat_partial

        # Now nat_kwgs is either NATBlockConfig or None
        if nat_kwgs is None:
            return nat_partial

        assert isinstance(nat_kwgs, NATBlockConfig), (
            f"{nat_kwgs=} is not a NATBlockConfig"
        )
        # Get existing partial arguments
        existing_kwargs = nat_partial.keywords.copy()
        # Update with NATBlockConfig parameters, but preserve existing ones
        nat_kwargs_dict = asdict(nat_kwgs)
        existing_kwargs.update(nat_kwargs_dict)
        nat_func = nat_partial.func
        nat_partial_new = partial(nat_func, **existing_kwargs)
        return nat_partial_new

    def _create_head(self):
        out_chan = self.cfg.lr_channels
        embed_dim = self.cfg.width
        # assert (
        #     embed_dim // 4 > out_chan
        # ), f"embed_dim // 4 > out_chan, {embed_dim//4=}> {out_chan=}"
        head_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1),
        )
        head_out = nn.Conv2d(embed_dim, out_chan, 1)
        lr_shortcut = create_unpatcher(
            embed_dim, embed_dim, patch_size=self.cfg.patch_size
        )

        head = nn.ModuleDict()
        head["head_conv"] = head_conv
        head["head_out"] = head_out
        head["lr_shortcut"] = lr_shortcut
        return head

    def _patching_inputs(self, lr, cond):
        # Embed individual modalities
        x_lr = self.patchers["lr_conv"](lr)  # MS patch embedding

        # Process latent conditions
        l_cond = self.patchers["latent_conv"](cond)  # Latent condition processing

        return x_lr, l_cond

    def _forward_head(self, x, lr_c):
        breakpoint()
        x = self.head["head_conv"](x)
        x = x + self.head["lr_shortcut"](lr_c)
        x = self.head["head_out"](x)
        return x

    def _to_out(self, x, lr, latent):
        res_type = self.cfg.residual_type
        if res_type == "condition":
            assert latent.shape[1] == x.shape[1], (
                f"Condition channel {latent.shape[1]} must match output channel {x.shape[1]}"
            )
            x = x + latent
        elif res_type == "lr":
            assert lr.shape[-2:] == x.shape[-2:]
            x = x + lr

        # Apply output rescaling using RescaleOutput layer
        x = self.output_rescale(x)
        return x

    def _forward_backbone(self, x, lr_c, latent_c):
        encs = []  # Store encoder outputs for skip connections

        # =====================================================================
        # Encoder Path (Downsampling with skip connections)
        # =====================================================================
        for i, (encoder, down) in enumerate(zip(self.encoders, self.downs)):
            # Process through encoder blocks at current resolution
            for block in encoder:
                x = block(x, latent_c, lr_c)  # Conditional block processing
            encs.append(x)  # Store for skip connection
            x = down(x)  # Downsample to next resolution level

        # =====================================================================
        # Middle Bottleneck Processing
        # =====================================================================
        for block in self.middle_blks:
            x = block(x, latent_c, lr_c)  # Process at lowest resolution

        # =====================================================================
        # Decoder Path (Upsampling with skip connections)
        # =====================================================================
        for i, (decoder, up, enc_skip) in enumerate(
            zip(self.decoders, self.ups, encs[::-1])
        ):
            x = up(x)  # Upsample to higher resolution
            x = x + enc_skip  # Add skip connection from encoder
            for block in decoder:
                x = block(x, latent_c, lr_c)  # Conditional block processing

        return x

    def forward(
        self,
        lr,
        cond: Annotated[Tensor, "latents or decoded images"],
    ):
        # Step 1: Process inputs through patch embedding layers
        x, latent_c = self._patching_inputs(lr, cond)
        lr_c = x.clone()

        # Step 2: Process through encoder-decoder backbone
        x = self._forward_backbone(x, lr_c, latent_c)

        # Step 3: Convert back to spatial resolution
        x = self.unpatcher(x)

        # Step 4: Apply output head for final feature processing
        x = self._forward_head(x, lr_c)

        # Step 5: Apply residual connections and output scaling
        x = self._to_out(x, lr, cond)

        return x

    @classmethod
    def create_model(cls, **kwargs):
        cfg = dataclass_from_dict(DenoiseNAFNetConfig, kwargs)
        model = cls(cfg)
        return model

    def _init_weights(self):
        """
        Initialize network weights using proper initialization schemes.

        Uses different initialization strategies for different layer types:
        - Conv2d: LeCun normal initialization
        - Linear: Xavier uniform initialization
        - LayerNorm: Ones for weights, zeros for bias
        - Output head: Zero initialization for stable training start
        """

        def _apply(module):
            if isinstance(module, nn.Conv2d):
                # Convolutional layers: LeCun normal initialization
                # nn.init.kaiming_normal_(module.weight, mode="fan_in")
                lecun_normal_(module.weight)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                # Linear layers: Xavier uniform initialization
                nn.init.xavier_uniform_(module.weight)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                # Layer normalization: standard initialization
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Apply weight initialization to all modules
        self.apply(_apply)

        # Initialize output head with zeros for stable training start
        if hasattr(self.head["head_out"], "weight"):
            nn.init.zeros_(self.head["head_out"].weight)
        if hasattr(self.head["head_out"], "bias"):
            nn.init.zeros_(self.head["head_out"].bias)

        log("[PansharpeningNAFNet] Initialized weights", level="info")


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


if __name__ == "__main__":
    model = DenoiseNAFNet.create_model(
        is_neg_1_1=False,
        width=64,
        enc_blk_nums=[1, 1, 1],
        dec_blk_nums=[1, 1, 1],
        enc_blk_type=["naf", "nat", "nat"],
        dec_blk_type=["nat", "nat", "naf"],
        condition_channel=16,
        patch_size=2,
        block_drop=0.1,
    )
    dtype = torch.bfloat16
    device = "cuda:0"

    model.to(device=device, dtype=dtype)
    lr = torch.randn(1, 8, 256, 256).to(device=device, dtype=dtype)
    latent = torch.randn(1, 16, 32, 32).to(device=device, dtype=dtype)

    with torch.autocast(device_type="cuda", dtype=dtype):
        out = model(lr, latent)
    print(out.shape)

    # fvcore
    from fvcore.nn import FlopCountAnalysis, flop_count_table

    """
    model, 0.372G, 0.803T
    """
    print(flop_count_table(FlopCountAnalysis(model, (lr, latent))))

    out.mean().backward()
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is None:
            print("name", n, "has not grad")

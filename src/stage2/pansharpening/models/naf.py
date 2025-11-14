import math
from dataclasses import asdict, dataclass, field, fields, replace
from functools import partial
from typing import Annotated, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
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
class PansharpeningNAFNetConfig:
    pan_channel: int = 1
    ms_channel: int = 8
    width: int = 256
    middle_blk_num: int = 1
    middle_blk_type: str = "nat"
    enc_blk_nums: list = field(default_factory=lambda: [1, 1, 1, 1])
    dec_blk_nums: list = field(default_factory=lambda: [1, 1, 1, 1])
    enc_blk_type: list = field(default_factory=lambda: ["naf", "naf", "naf", "naf"])
    dec_blk_type: list = field(default_factory=lambda: ["naf", "naf", "naf", "naf"])
    condition_channel: int = 16  # latent channels
    ms_cond_chans: int = 9  # wv3: 8+1
    use_residual: bool = True
    dw_expand: int = 2
    ffn_expand: int = 2
    patch_size: int = 1
    condition_on_decoder: bool = True
    block_drop: float = 0.0
    output_rescale: bool = True
    is_neg_1_1: bool = True
    residual_type: str = "ms"
    nat_enc_kwargs: list[NATBlockConfig | None] = field(default_factory=lambda: [])
    nat_mid_kwargs: NATBlockConfig | None = None
    nat_dec_kwargs: list[NATBlockConfig | None] = field(default_factory=lambda: [])


class PansharpeningNAFNet(nn.Module):
    """
    Pansharpening Network using NAFNet architecture with conditional blocks.

    This network performs pansharpening by fusing high-resolution panchromatic (PAN) images
    with low-resolution multispectral (MS) images to produce high-resolution MS images.
    The architecture uses encoder-decoder structure with conditional attention blocks.
    """

    def __init__(self, cfg: PansharpeningNAFNetConfig):
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
        patchers["pan_conv"] = create_patcher(
            cfg.pan_channel, cfg.width, cfg.patch_size
        )  # PAN image patch embedding
        patchers["ms_conv"] = create_patcher(
            cfg.ms_channel, cfg.width, cfg.patch_size
        )  # MS image patch embedding
        patchers["fused_conv"] = create_patcher(
            cfg.width * 2, cfg.width, 1
        )  # Fused feature processing
        patchers["latent_conv"] = nn.Sequential(
            create_norm_layer("layernorm2d", cfg.condition_channel),
            create_conv2d(cfg.condition_channel, cfg.width, 3, bias=True),
        )  # Latent condition processing
        patchers["ms_cond_conv"] = create_conv2d(
            cfg.width * 2, cfg.width, 1
        )  # MS condition processing

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
                # nn.Sequential(
                #     nn.Upsample(scale_factor=2, mode="nearest"),
                #     nn.Conv2d(chan, chan // 2, 3, 1, 1),
                # )
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

        ## Weight initialization
        self.init_weights()

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
        out_chan = self.cfg.ms_channel
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
        # ms_shortcut = create_conv2d(embed_dim, embed_dim, 1)
        ms_shortcut = create_unpatcher(
            embed_dim, embed_dim, patch_size=self.cfg.patch_size
        )

        head = nn.ModuleDict()
        head["head_conv"] = head_conv
        head["head_out"] = head_out
        head["ms_shortcut"] = ms_shortcut
        return head

    def _patching_inputs(self, ms, pan, cond):
        """
        Process input modalities through patch embedding layers.

        Args:
            ms: Low-resolution multispectral input
            pan: High-resolution panchromatic input
            cond: Latent conditions

        Returns:
            x: Fused features from MS and PAN inputs
            ms_cond: MS condition features
            l_cond: Latent condition features
        """
        # Embed individual modalities
        x_ms = self.patchers["ms_conv"](ms)  # MS patch embedding
        x_pan = self.patchers["pan_conv"](pan)  # PAN patch embedding

        # Fuse MS and PAN features
        x = torch.cat([x_ms, x_pan], dim=1)  # Concatenate along channel dimension
        ms_cond = self.patchers["ms_cond_conv"](x)  # MS condition processing
        x = self.patchers["fused_conv"](x)  # Fused feature processing

        # Process latent conditions
        l_cond = self.patchers["latent_conv"](cond)  # Latent condition processing

        return x, ms_cond, l_cond

    def _forward_head(self, x, ms_cond):
        x = self.head["head_conv"](x)
        x = x + self.head["ms_shortcut"](ms_cond)
        x = self.head["head_out"](x)
        return x

    def _to_out(self, x, ms, pan, latent):
        res_type = self.cfg.residual_type
        if res_type == "condition":
            assert latent.shape[1] == x.shape[1], (
                f"Condition channel {latent.shape[1]} must match output channel {x.shape[1]}"
            )
            x = x + latent
        elif res_type == "ms":
            assert ms.shape[-2:] == x.shape[-2:]
            x = x + ms
        elif res_type == "pan":
            assert pan.shape[1] == 1
            x = x + pan

        # Apply output rescaling using RescaleOutput layer
        x = self.output_rescale(x)
        return x

    def _forward_backbone(self, x, ms_c, l_c):
        """
        Process features through encoder-decoder backbone with skip connections.

        Args:
            x: Fused input features
            ms_c: MS condition features
            l_c: Latent condition features

        Returns:
            Processed features after full encoder-decoder pass
        """
        encs = []  # Store encoder outputs for skip connections

        # =====================================================================
        # Encoder Path (Downsampling with skip connections)
        # =====================================================================
        for i, (encoder, down) in enumerate(zip(self.encoders, self.downs)):
            # Process through encoder blocks at current resolution
            for block in encoder:
                x = block(x, l_c, ms_c)  # Conditional block processing
            encs.append(x)  # Store for skip connection
            x = down(x)  # Downsample to next resolution level

        # =====================================================================
        # Middle Bottleneck Processing
        # =====================================================================
        for block in self.middle_blks:
            x = block(x, l_c, ms_c)  # Process at lowest resolution

        # =====================================================================
        # Decoder Path (Upsampling with skip connections)
        # =====================================================================
        for i, (decoder, up, enc_skip) in enumerate(
            zip(self.decoders, self.ups, encs[::-1])
        ):
            x = up(x)  # Upsample to higher resolution
            x = x + enc_skip  # Add skip connection from encoder
            for block in decoder:
                x = block(x, l_c, ms_c)  # Conditional block processing

        return x

    def forward(
        self,
        ms,
        pan,
        cond: Annotated[Tensor, "latents or decoded images"],
    ):
        """
        Forward pass of the pansharpening network.

        Args:
            ms: Low-resolution multispectral input
            pan: High-resolution panchromatic input
            cond: Latent conditions or decoded images for guidance

        Returns:
            High-resolution pansharpened multispectral output
        """
        # tmp
        # cond = cond * 0.0
        # print('-- max value', ms.abs().max(), pan.abs().max())
        # print(cond.mean(), cond.std())

        # Step 1: Process inputs through patch embedding layers
        x, ms_c, l_c = self._patching_inputs(ms, pan, cond)

        # Step 2: Process through encoder-decoder backbone
        x = self._forward_backbone(x, ms_c, l_c)

        # Step 3: Convert back to spatial resolution
        x = self.unpatcher(x)

        # Step 4: Apply output head for final feature processing
        x = self._forward_head(x, ms_c)

        # Step 5: Apply residual connections and output scaling
        x = self._to_out(x, ms, pan, cond)

        return x

    @classmethod
    def create_model(cls, **kwargs):
        cfg = dataclass_from_dict(PansharpeningNAFNetConfig, kwargs)
        model = cls(cfg)
        return model

    def init_weights(self):
        """
        Initialize network weights using proper initialization schemes.

        Uses different initialization strategies for different layer types:
        - Conv2d: LeCun normal initialization
        - Linear: Xavier uniform initialization
        - LayerNorm: Ones for weights, zeros for bias
        - Output head: Zero initialization for stable training start
        """
        norms = [
            get_norm_layer(n)
            for n in ["layernorm2d", "rmsnorm2d", "layernorm", "rmsnorm"]
        ]

        @torch.no_grad()
        def _apply(module, name):
            logger.debug(f"init module: {name}")
            if hasattr(module, "init_weights"):
                module.init_weights()
            elif isinstance(module, nn.Conv2d):
                # Convolutional layers: LeCun normal initialization
                # nn.init.kaiming_normal_(
                #     module.weight, mode="fan_out", nonlinearity="relu"
                # )
                lecun_normal_(module.weight)
                # nn.init.xavier_normal_(module.weight)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                # Linear layers: Xavier uniform initialization
                # nn.init.xavier_normal_(module.weight)
                nn.init.trunc_normal_(module.weight, std=0.02)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, tuple(norms)):
                # Layer normalization: standard initialization
                nn.init.ones_(module.weight)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Apply weight initialization to all modules
        # self.apply(_apply)
        named_apply(_apply, self)

        # Initialize output head with zeros for stable training start
        if self.cfg.residual_type is not None:
            if hasattr(self.head["head_out"], "weight"):
                nn.init.zeros_(self.head["head_out"].weight)
            if hasattr(self.head["head_out"], "bias"):
                nn.init.zeros_(self.head["head_out"].bias)

        # Assert all the weights are initialized
        for name, param in self.named_parameters():
            data = param.data
            assert not torch.any(torch.isnan(data)), f"NaN found in parameter {name}"

        logger.log("NOTE", "[PansharpeningNAFNet] Initialized weights")


# *==============================================================
# * Decoder only NAFNet for Pansharpening
# *==============================================================


@dataclass
class PansharpeningNAFDecoderNetConfig:
    pan_channel: int = 1
    ms_channel: int = 8
    width: int = 256
    middle_blk_num: int = 1
    middle_blk_type: str = "naf"
    dec_blk_nums: list = field(default_factory=lambda: [1, 1, 1, 1])
    dec_blk_type: list = field(default_factory=lambda: ["naf", "naf", "naf", "naf"])
    condition_channel: int = 16
    residual_type: str = "none"  # ms, pan, condition, none
    dw_expand: int = 2
    ffn_expand: int = 2
    patch_size: int = 1
    block_drop: float = 0.0
    output_rescale: bool = True
    is_neg_1_1: bool = True
    up_ratio: int = 4
    condition_on_decoder: bool = True
    nat_mid_kwargs: NATBlockConfig | None = None
    nat_dec_kwargs: list[NATBlockConfig | None] = field(default_factory=lambda: [])
    up_ratio: int = 4
    head_interp: str = "transpose"


class PansharpeningNAFDecoderNet(nn.Module):
    def __init__(self, cfg: PansharpeningNAFDecoderNetConfig):
        super().__init__()

        # Validate configuration
        assert len(cfg.dec_blk_nums) == len(cfg.dec_blk_type), (
            f"dec_blk_nums length ({len(cfg.dec_blk_nums)}) must match "
            f"dec_blk_type length ({len(cfg.dec_blk_type)})"
        )
        assert cfg.middle_blk_type in [
            "naf",
            "nat",
        ], f"middle_blk_type must be 'naf' or 'nat', got {cfg.middle_blk_type}"

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

        # =====================================================================
        # Block Configuration and Mapping
        # =====================================================================

        chan = cfg.width

        # Partial functions for different block types with shared parameters
        naf_block_partial = partial(
            NAFBlockConditional if cfg.condition_on_decoder else NAFBlock,
            cond_chs=cfg.width if cfg.condition_on_decoder else None,
            DW_Expand=cfg.dw_expand,
            FFN_Expand=cfg.ffn_expand,
            drop_out_rate=cfg.block_drop,
            ms_cond_chans=cfg.width if cfg.condition_on_decoder else None,
        )

        nat_block_partial = partial(
            Spatial2DNATBlockConditional
            if cfg.condition_on_decoder
            else Spatial2DNATBlock,
            cond_chs=cfg.width if cfg.condition_on_decoder else None,
            ms_cond_chans=cfg.width if cfg.condition_on_decoder else None,
            ffn_ratio=cfg.ffn_expand,
            drop_path=cfg.block_drop,
            # defaults
            k_size=4,
            stride=2,
            dilation=2,
            n_heads=8,
        )

        # Block type mappings for dynamic architecture selection
        dec_mapping = {
            "naf": naf_block_partial,
            "nat": nat_block_partial,
        }

        middle_mapping = {
            "naf": naf_block_partial,
            "nat": nat_block_partial,
        }

        # =====================================================================
        # Middle Bottleneck Blocks
        # =====================================================================
        middle_fn = middle_mapping[cfg.middle_blk_type]
        middle_fn = self._replace_nat_defaults_decoder(
            middle_fn, cfg.nat_mid_kwargs, None
        )
        log(
            f"[NAFDecoderNet Middle Block] Type: {middle_fn.func}, Num: {cfg.middle_blk_num}"
        )
        self.middle_blks = nn.ModuleList(
            [middle_fn(chan) for _ in range(cfg.middle_blk_num)]
        )

        # =====================================================================
        # Decoder Path (Upsampling)
        # =====================================================================
        for i, num in enumerate(cfg.dec_blk_nums):
            if i < self.n_up_most:
                self.ups.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode="nearest"),
                        nn.Conv2d(chan, chan // 2, 1, 1, bias=False),
                    )
                )
            else:
                self.ups.append(create_conv2d(chan, chan // 2, 3))
            chan = chan // 2

            dec_fn = dec_mapping[cfg.dec_blk_type[i]]
            log(f"[NAFDecoderNet Decoder Block {i}] Type: {dec_fn.func}, Num: {num}")
            dec_fn = self._replace_nat_defaults_decoder(dec_fn, cfg.nat_dec_kwargs, i)
            self.decoders.append(nn.ModuleList([dec_fn(chan) for _ in range(num)]))

        self._init_weights()

    def _create_head(self, dec_out_dim: int):
        out_chan = self.cfg.ms_channel
        embed_dim = dec_out_dim
        fused_in_dim = dec_out_dim * 2 + self.cfg.width
        # assert (
        #     embed_dim // 4 > out_chan
        # ), f"embed_dim // 4 > out_chan, {embed_dim//4=}> {out_chan=}"
        assert ((n_ups := np.log2(self.cfg.up_ratio)) % 1) == 0, (
            f"Up ratio must be a power of 2, got {self.cfg.up_ratio}"
        )
        if self.cfg.head_interp == "transpose":
            ms_in_ups = nn.Sequential(
                nn.Conv2d(self.cfg.width, embed_dim, 1),
                *[
                    nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2, groups=embed_dim)
                    for _ in range(int(n_ups))
                ],  # fmt: skip
            )
        elif self.cfg.head_interp == "upsample":
            create_up2 = lambda: nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                create_conv2d(embed_dim, embed_dim, 3, depth_wise=True),
            )
            ms_in_ups = nn.Sequential(
                nn.Conv2d(self.cfg.width, embed_dim, 1),
                *[create_up2() for _ in range(int(n_ups))],
            )
        else:
            raise ValueError(f"Invalid head interpolation mode: {self.cfg.head_interp}")
        head_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1),
        )
        head_out = nn.Conv2d(embed_dim, out_chan, 1)
        ms_shortcut = create_conv2d(embed_dim, embed_dim, 1)
        # Add fused conv for creating MS condition from x and x_pan
        fused_conv = create_conv2d(fused_in_dim, embed_dim, 1)

        head = nn.ModuleDict()
        head["ms_ups"] = ms_in_ups
        head["head_conv"] = head_conv
        head["head_out"] = head_out
        head["ms_shortcut"] = ms_shortcut
        head["fused_conv"] = fused_conv
        return head

    def _forward_head(self, x, fused_cond):
        fused_cond = self.head["fused_conv"](fused_cond)
        x = self.head["head_conv"](x)
        x = x + self.head["ms_shortcut"](fused_cond)
        x = self.head["head_out"](x)
        return x

    def _to_out(self, x, ms, pan, cond):
        res_type = self.cfg.residual_type
        if res_type == "condition":
            assert cond.shape == x.shape, (
                f"Condition channel {cond.shape} must match output channel {x.shape}"
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
        x_ms, x_pan, cond_attn = self._patching_inputs(ms, pan, cond)

        x = x_ms
        for block in self.middle_blks:
            x = block(x, x_pan, cond_attn)

        for i, (decoder, up) in enumerate(zip(self.decoders, self.ups)):
            x = up(x)
            for block in decoder:
                x = block(x, x_pan, cond_attn)

        # unpatcher
        x = self.unpatcher(x)

        # Create MS condition for head shortcut (similar to PansharpeningNAFNet)
        # x_ms_interp = F.interpolate(
        #     x_ms, size=x.shape[-2:], mode="bilinear", antialias=True
        # )
        x_ms_interp = self.head["ms_ups"](x_ms)
        x_fused = torch.cat([x, x_ms_interp, x_pan], dim=1)

        # Apply head with shortcut connection
        x = self._forward_head(x, x_fused)

        # Apply residual connections and output scaling
        x = self._to_out(x, ms, pan, cond)
        return x

    def _replace_nat_defaults_decoder(
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

    @classmethod
    def create_model(cls, **kwargs):
        cfg = dataclass_from_dict(PansharpeningNAFDecoderNetConfig, kwargs)
        model = cls(cfg)
        return model

    def _init_weights(self):
        def _apply(module):
            if isinstance(module, nn.Conv2d):
                # nn.init.kaiming_normal_(module.weight)
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
        if hasattr(self.head["head_out"], "weight"):
            nn.init.zeros_(self.head["head_out"].weight)
        if hasattr(self.head["head_out"], "bias"):
            nn.init.zeros_(self.head["head_out"].bias)

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
        enc_blk_nums=[1, 1, 1, 1],  # Match default enc_blk_type length
        dec_blk_nums=[1, 1, 1, 1],  # Match default dec_blk_type length
        use_residual=True,
        patch_size=1,
        dw_expand=2,
        ffn_expand=2,
        condition_on_decoder=True,
        residual_type="ms",
        is_neg_1_1=False,
        output_rescale=False,
    )
    model.train()

    from fvcore.nn import parameter_count_table

    print(parameter_count_table(model))

    # Move model to CUDA and use bfloat16 with autocast
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    for i in range(1000):
        # Create sample inputs
        ms = torch.randn(1, 8, 256, 256, device=device)
        pan = torch.randn(1, 1, 256, 256, device=device)
        cond = torch.randn(1, 256, 32, 32, device=device)  # latents

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Test forward pass with bfloat16 autocast
        if device == "cuda" and torch.cuda.is_bf16_supported():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(ms, pan, cond)
        else:
            # Fallback to float32 if bfloat16 not supported
            output = model(ms, pan, cond)

        # Verify output shape
        assert output.shape == ms.shape, (
            f"Expected output shape {ms.shape}, got {output.shape}"
        )

        optimizer.zero_grad()
        ng = {}

        gt = torch.randn_like(output)
        loss = F.l1_loss(output, gt)
        print(loss)

        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                ng[name] = param.grad

        # sort the dict grad
        if i == 0:
            print("------------------")
            ng = {
                k: v
                for k, v in sorted(
                    ng.items(), key=lambda item: item[1].norm().item(), reverse=True
                )
            }
            # print the max first 10 grads
            for i, (k, v) in enumerate(ng.items()):
                if i < 10:
                    print(f"grad {i}: name={k}, grad_norm={v.norm().item()}")

        # step
        optimizer.step()

    # Test inference mode
    # model.eval()
    # with torch.no_grad():
    #     with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    #         output_eval = model(ms, pan, cond)

    # Verify evaluation output shape
    # assert output_eval.shape == ms.shape, (
    #     f"Expected eval output shape {ms.shape}, got {output_eval.shape}"
    # )

    # print("✓ PansharpeningNAFNet test passed!")
    # print(f"Input MS shape: {ms.shape}")
    # print(f"Input PAN shape: {pan.shape}")
    # print(f"Input condition shape: {cond.shape}")
    # print(f"Output shape: {output.shape}")


def test_pansharpening_naf_decoder_net():
    from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count_table

    # Create sample inputs in bf16 format
    bs = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda"
    ms = torch.randn(bs, 8, 256, 256, device=device, dtype=torch.bfloat16)
    pan = torch.randn(bs, 1, 256, 256, device=device, dtype=torch.bfloat16)
    cond = torch.randn(bs, 16, 32, 32, device=device, dtype=torch.bfloat16)  # latents

    cfg = PansharpeningNAFDecoderNetConfig(
        width=512,
        middle_blk_num=1,
        dec_blk_nums=[1, 1, 1],
        dec_blk_type=["nat", "nat", "naf"],
        middle_blk_type="nat",
        condition_on_decoder=True,
        nat_mid_kwargs=NATBlockConfig(n_heads=8, k_size=4, stride=2, dilation=2),
        nat_dec_kwargs=[
            NATBlockConfig(n_heads=8, k_size=4, stride=2, dilation=2),
            NATBlockConfig(n_heads=8, k_size=4, stride=2, dilation=2),
            # None,
            None,
        ],
    )
    model = PansharpeningNAFDecoderNet(cfg).to(device).to(dtype=torch.bfloat16)
    model.train()

    # print(flop_count_table(FlopCountAnalysis(model, (ms, pan, cond))))

    # optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    import heavyball

    heavyball.utils.compile_mode = None
    optim = heavyball.ForeachAdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1,
        betas=(0.9, 0.999),
        caution=True,
        foreach=True,
    )

    # Test forward pass
    from tqdm import trange

    for _ in trange(200):
        # Model and inputs are already in bf16, no need for autocast
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(ms, pan, cond)
        # loss = output.mean()
        # loss.backward()
        # optim.step()
        # optim.zero_grad()

        # for n, p in model.named_parameters():
        #     if p.requires_grad and p.grad is None:
        #         print("name", n, "has not grad")


def test_nat_blocks_only():
    """Test PansharpeningNAFNet with NAT blocks for both encoder and decoder."""
    print("Testing PansharpeningNAFNet with NAT blocks only...")

    # Create model using only NAT blocks
    model = PansharpeningNAFNet.create_model(
        ms_channel=8,
        pan_channel=1,
        condition_channel=256,
        width=64,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1, 1],
        dec_blk_nums=[1, 1, 1, 1],
        enc_blk_type=["nat", "nat", "nat", "nat"],  # Use NAT for encoder
        dec_blk_type=["nat", "nat", "nat", "nat"],  # Use NAT for decoder
        middle_blk_type="nat",  # Use NAT for middle blocks
        use_residual=True,
        patch_size=1,
        dw_expand=1,
        ffn_expand=2,
        condition_on_decoder=True,
        residual_type="ms",
        is_neg_1_1=True,
        output_rescale=True,
    )
    model.train()
    model = model.to("cuda").bfloat16()

    print("✓ Model created successfully with NAT blocks")
    print(f"Model configuration:")
    print(f"  - Encoder block types: {model.cfg.enc_blk_type}")
    print(f"  - Decoder block types: {model.cfg.dec_blk_type}")
    print(f"  - Middle block type: {model.cfg.middle_blk_type}")

    # Create sample inputs
    device = "cuda"
    ms = torch.randn(1, 8, 256, 256, device=device).bfloat16()
    pan = torch.randn(1, 1, 256, 256, device=device).bfloat16()
    cond = torch.randn(1, 256, 32, 32, device=device).bfloat16()

    # Test forward pass with bf16 autocast
    print("Testing forward pass...")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(ms, pan, cond)
    print(f"✓ Forward pass successful")
    print(f"  - Input MS shape: {ms.shape}")
    print(f"  - Input PAN shape: {pan.shape}")
    print(f"  - Input condition shape: {cond.shape}")
    print(f"  - Output shape: {output.shape}")

    # Verify output shape
    assert output.shape == ms.shape, (
        f"Expected output shape {ms.shape}, got {output.shape}"
    )
    print("✓ Output shape correct")

    # Test inference mode
    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output_eval = model(ms, pan, cond)
    print("✓ Inference mode successful")

    # Test gradient flow
    model.train()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(ms, pan, cond)
        loss = output.mean()
    loss.backward()
    print("✓ Gradient flow successful")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")

    print("\n✅ NAT blocks test passed!")
    return True


def test_mixed_blocks():
    """Test PansharpeningNAFNet with mixed NAF and NAT blocks."""
    print("\nTesting PansharpeningNAFNet with mixed NAF and NAT blocks...")

    # Create model with mixed blocks
    model = PansharpeningNAFNet.create_model(
        ms_channel=8,
        pan_channel=1,
        condition_channel=256,
        width=64,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1],
        dec_blk_nums=[1, 1, 1],
        enc_blk_type=["naf", "nat", "naf"],  # Mixed encoder
        dec_blk_type=["nat", "nat", "naf"],  # Mixed decoder
        middle_blk_type="nat",
        use_residual=True,
        patch_size=1,
        dw_expand=1,
        ffn_expand=2,
        condition_on_decoder=True,
        residual_type="ms",
        is_neg_1_1=True,
        output_rescale=True,
        nat_enc_kwargs=[
            None,
            NATBlockConfig(n_heads=8, k_size=4, stride=2, dilation=2),
            NATBlockConfig(n_heads=8, k_size=8, stride=2, dilation=1),
        ],
        nat_mid_kwargs=NATBlockConfig(n_heads=8, k_size=8, stride=2, dilation=1),
        nat_dec_kwargs=[
            NATBlockConfig(n_heads=8, k_size=8, stride=2, dilation=1),
            NATBlockConfig(n_heads=8, k_size=4, stride=2, dilation=2),
            None,
        ],
    )
    model.train()
    model = model.to("cuda", torch.bfloat16)

    print("✓ Mixed block model created successfully")

    # Create sample inputs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ms = torch.randn(1, 8, 256, 256, device=device, dtype=torch.bfloat16)
    pan = torch.randn(1, 1, 256, 256, device=device, dtype=torch.bfloat16)
    cond = torch.randn(1, 256, 32, 32, device=device, dtype=torch.bfloat16)

    # Test forward pass with bf16 autocast
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(ms, pan, cond)
    print(f"✓ Mixed blocks forward pass successful")
    print(f"  - Output shape: {output.shape}")
    assert output.shape == ms.shape
    print("✅ Mixed blocks test passed!")
    return True


if __name__ == "__main__":
    print("decoder")
    # test_pansharpening_naf_decoder_net()
    test_pansharpening_nafnet()

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import einops
import torch
import torch.nn as nn
from timm.layers.create_norm import get_norm_layer
from timm.layers.helpers import to_2tuple
from timm.layers.patch_embed import PatchEmbed
from timm.models.naflexvit import NaFlexEmbeds, NaFlexVit


@dataclass
class NaFlexVitCfg:
    """Configuration for FlexVit model.

    This dataclass contains the bulk of model configuration parameters,
    with core parameters (img_size, in_chans, num_classes, etc.) remaining
    as direct constructor arguments for API compatibility.
    """

    # Architecture parameters
    patch_size: Union[int, Tuple[int, int]] = 1  # force to be 1
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    scale_mlp_norm: bool = False  # Apply scaling norm to MLP

    # Attention parameters
    qkv_bias: bool = True
    qk_norm: bool = False
    proj_bias: bool = True
    attn_drop_rate: float = 0.0
    scale_attn_inner_norm: bool = False  # Apply scaling norm to attn context

    # Regularization
    init_values: Optional[float] = (
        None  # Layer-scale init values (layer-scale enabled if not None)
    )
    drop_rate: float = 0.0  # Dropout rate for classifier
    pos_drop_rate: float = 0.0  # Dropout rate for position embeddings
    patch_drop_rate: float = 0.0  # Dropout rate for patch tokens
    proj_drop_rate: float = 0.0  # Dropout rate for linear projections
    drop_path_rate: float = 0.0  # Stochastic depth drop rate

    # Prefix token configuration
    class_token: bool = False  # Use class token
    reg_tokens: int = 0  # Number of register tokens

    # Position embedding configuration
    pos_embed: str = "learned"  # Type of position embedding ('learned', 'factorized', 'rope', 'none')
    pos_embed_grid_size: Optional[Tuple[int, int]] = (
        16,
        16,
    )  # Grid size for position embedding initialization
    pos_embed_interp_mode: str = (
        "bicubic"  # Interpolation mode for position embedding resizing
    )
    pos_embed_ar_preserving: bool = False  # Whether to preserve aspect ratio during position embedding interpolation
    pos_embed_use_grid_sample: bool = (
        False  # Whether to use grid_sample for naflex position embedding interpolation
    )

    # ROPE specific configuration
    rope_type: str = "axial"  # ROPE type: '' or 'none' for no ROPE, 'axial' for standard, 'mixed' for learnable frequencies
    rope_temperature: float = 10000.0  # Temperature for ROPE frequency computation
    rope_ref_feat_shape: Optional[Tuple[int, int]] = None
    rope_grid_offset: float = 0.0  # Grid offset for non-pixel ROPE mode
    rope_grid_indexing: str = "ij"  # Grid indexing mode for ROPE ('ij' or 'xy')

    # Image processing
    dynamic_img_pad: bool = (
        False  # Whether to enable dynamic padding for variable resolution
    )

    # Other architecture choices
    pre_norm: bool = True  # Whether to apply normalization before attention/MLP layers (start of blocks)
    final_norm: bool = True  # Whether to apply final normalization before pooling and classifier (end of blocks)
    fc_norm: Optional[bool] = (
        None  # Whether to normalize features before final classifier (after pooling)
    )

    # Global pooling setup
    global_pool: str = ""  # Type of global pooling for final sequence  # * no pooling
    pool_include_prefix: bool = (
        False  # Whether to include class/register prefix tokens in global pooling
    )
    attn_pool_num_heads: Optional[int] = None  # Override num_heads for attention pool
    attn_pool_mlp_ratio: Optional[float] = None  # Override mlp_ratio for attention pool

    # Weight initialization
    weight_init: str = ""  # Weight initialization scheme
    fix_init: bool = True  # Apply weight initialization fix (scaling w/ layer index)

    # Embedding configuration
    embed_proj_type: str = "linear"  # Type of embedding layer ('conv' or 'linear')
    input_norm_layer: Optional[str] = (
        None  # Normalization layer for embeddings input (before input projection)
    )
    embed_norm_layer: Optional[str] = (
        None  # Normalization layer for embeddings (after input projection)
    )

    # Layer implementations
    norm_layer: Optional[str] = None  # Normalization layer for transformer blocks
    act_layer: Optional[str] = None  # Activation layer for MLP blocks
    block_fn: Optional[str] = None  # Transformer block implementation class name
    mlp_layer: Optional[str] = None  # MLP implementation class name

    # EVA-specific parameters
    attn_type: str = "eva"  # Attention type: 'standard', 'eva', 'rope'
    swiglu_mlp: bool = True  # Use SwiGLU MLP variant
    qkv_fused: bool = True  # Whether to use fused QKV projections

    # Variable patch size support
    enable_patch_interpolator: bool = True  # Enable dynamic patch size support


@dataclass
class NaFlexVitCfgAdpoted:
    """
    Adpoted from timm.models.naflexvit.NaFlexVitCfg
    """

    img_size: int = 32
    in_chans: int = 16
    patch_embed_dim: int = 16
    out_chans: int = 256
    patch_size: int = 2


class NaFlexVitAdpoted(NaFlexVit):
    def __init__(self, cfg: NaFlexVitCfg, cfg2: NaFlexVitCfgAdpoted):
        super().__init__(cfg, in_chans=cfg2.patch_embed_dim, img_size=cfg2.img_size)
        self.cfg, self.cfg2 = cfg, cfg2
        cfg.patch_size = 1  # force to be 1
        self._build_input_patcher(cfg, cfg2)
        self._build_head(cfg, cfg2)

    def _build_head(self, cfg: NaFlexVitCfg, cfg2: NaFlexVitCfgAdpoted):
        norm_layer = get_norm_layer(cfg.norm_layer) or nn.LayerNorm
        self.head = nn.Sequential(
            norm_layer(cfg.embed_dim),
            nn.Linear(cfg.embed_dim, cfg2.out_chans * cfg2.patch_size**2, bias=True),
        )

    def _build_input_patcher(self, cfg: NaFlexVitCfg, cfg2: NaFlexVitCfgAdpoted):
        patch_size = cfg2.patch_size

        self.patch_size = to_2tuple(patch_size)
        self.patch_embed = PatchEmbed(
            img_size=cfg2.img_size,
            patch_size=patch_size,
            in_chans=cfg2.in_chans,
            embed_dim=cfg2.patch_embed_dim,
            bias=True,
            strict_img_size=False,
            output_fmt="NCHW",
        )
        self.fuse_stem = nn.Conv2d(cfg2.patch_embed_dim * 2, cfg2.patch_embed_dim, 1)

    def unpatchify(self, x: torch.Tensor):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.cfg2.out_chans
        p = self.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = einops.rearrange(
            x, "bs (h w) (p1 p2 c) -> bs c (h p1) (w p2)", h=h, w=w, p1=p, p2=p, c=c
        )
        return x

    def _forward_additional_patcher(self, ms, pan):
        latents = (ms, pan)
        ys = []
        for latent in latents:
            y = self.patch_embed(latent)
            ys.append(y)

        # fuse all latent
        y = self.fuse_stem(torch.cat(ys, dim=1))
        return y

    def _forward_after_backbone(self, x):
        x = self.head(x)
        x = self.unpatchify(x)
        return x

    def forward(self, ms, pan):
        x = self._forward_additional_patcher(ms, pan)
        x = self.forward_features(x)
        x = self._forward_after_backbone(x)
        return x


def test_naflex_vit_pansharpening_model():
    cfg = NaFlexVitCfg(
        embed_dim=256,
        depth=8,
        num_heads=8,
    )
    cfg2 = NaFlexVitCfgAdpoted(
        img_size=32,
        in_chans=16,
        out_chans=256,
        patch_size=2,
    )

    ms = torch.randn(1, 16, 64, 64)
    pan = torch.randn(1, 16, 64, 64)

    model = NaFlexVitAdpoted(cfg, cfg2)
    out = model(ms, pan)
    print(out.shape)  # [1, 256, 64, 64]

    out.mean().backward()

    # Check parameters
    from fvcore.nn import FlopCountAnalysis, flop_count_table

    flops = FlopCountAnalysis(model, (ms, pan))
    print(flop_count_table(flops))

    # Check gradients
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            print(f"Parameter {name} has no gradient!")


if __name__ == "__main__":
    test_naflex_vit_pansharpening_model()

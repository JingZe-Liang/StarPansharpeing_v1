from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import einops
import torch
import torch.nn as nn
from timm.layers.create_norm import get_norm_layer
from timm.layers.helpers import to_2tuple
from timm.layers.patch_embed import PatchEmbed
from timm.models.naflexvit import NaFlexEmbeds, NaFlexVit

from src.utilities.config_utils import function_config_to_basic_types

from .norm import *  # register custom norms


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
    qk_norm: bool = True
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
    weight_init: str = "jax"  # Weight initialization scheme
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
    norm_layer: Optional[str] = (
        "flarmsnorm"  # Normalization layer for transformer blocks
    )
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
    z_dim: int = 256
    out_chans: int = 16
    out_2d_latent: bool = True


class Transformer(NaFlexVit):
    def __init__(self, cfg: NaFlexVitCfg, cfg2: NaFlexVitCfgAdpoted):
        super().__init__(cfg, in_chans=cfg2.z_dim, img_size=cfg2.img_size)
        self.cfg, self.cfg2 = cfg, cfg2
        self._build_head(cfg, cfg2)

    def _build_head(self, cfg: NaFlexVitCfg, cfg2: NaFlexVitCfgAdpoted):
        norm_layer = get_norm_layer(cfg.norm_layer) or nn.LayerNorm
        self.head = nn.Sequential(
            norm_layer(cfg.embed_dim),
            nn.Linear(cfg.embed_dim, cfg2.out_chans * cfg.patch_size**2, bias=True),
        )

    def unpatchify(self, x: torch.Tensor, hw=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.cfg2.out_chans
        p = self.cfg.patch_size
        if hw is None:
            h = w = int(x.shape[1] ** 0.5)
            assert h * w == x.shape[1]
        else:
            h, w = hw

        x = einops.rearrange(
            x, "bs (h w) (p1 p2 c) -> bs c (h p1) (w p2)", h=h, w=w, p1=p, p2=p, c=c
        )
        return x

    def _forward_after_backbone(self, x, hw):
        x = self.head(x)
        if self.cfg2.out_2d_latent:
            x = self.unpatchify(x, hw)
        return x

    def forward(self, x):
        hw = x.shape[-2:]
        x = self.forward_features(x)
        x = x[:, self.cfg.reg_tokens :]
        x = self._forward_after_backbone(x, hw)
        return x

    @function_config_to_basic_types
    @staticmethod
    def create_model(cfg1_kwargs: dict = {}, cfg2_kwargs: dict = {}):
        cfg1 = NaFlexVitCfg(**cfg1_kwargs)
        cfg2 = NaFlexVitCfgAdpoted(**cfg2_kwargs)
        model = Transformer(cfg1, cfg2)
        return model


def test_naflex_vit_pansharpening_model():
    torch.cuda.set_device(1)
    cfg = NaFlexVitCfg(
        embed_dim=512,
        depth=8,
        num_heads=8,
        rope_type="axial",
        pos_embed="learned",
        pos_embed_grid_size=(32, 32),
        reg_tokens=8,
    )
    cfg2 = NaFlexVitCfgAdpoted(
        img_size=32,
        in_chans=16,
        out_chans=256,
    )

    x = torch.randn(1, 256, 32, 32).cuda()

    model = Transformer(cfg, cfg2).cuda()
    out = model(x)
    print(out.shape)  # [1, 256, 64, 64]

    out.mean().backward()

    # Check parameters
    from fvcore.nn import FlopCountAnalysis, flop_count_table

    flops = FlopCountAnalysis(model, (x,))
    print(flop_count_table(flops))

    # Check gradients
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            print(f"Parameter {name} has no gradient!")


if __name__ == "__main__":
    """
    python -m src.stage1.cosmos.modules.naflex
    """
    test_naflex_vit_pansharpening_model()

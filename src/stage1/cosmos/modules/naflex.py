from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import einops
import torch
import torch.nn as nn
from loguru import logger
from timm.layers.create_norm import get_norm_layer
from timm.layers.helpers import to_2tuple
from timm.layers.patch_embed import PatchEmbed
from timm.layers.pos_embed import resample_abs_pos_embed
from timm.layers.pos_embed_sincos import build_fourier_pos_embed
from timm.models.naflexvit import NaFlexEmbeds, NaFlexVit

from src.utilities.config_utils import (
    dataclass_from_dict,
    function_config_to_basic_types,
)

from .norm import *  # register custom norms


@dataclass
class NaFlexVitCfg:
    """Configuration for FlexVit model.

    This dataclass contains the bulk of model configuration parameters,
    with core parameters (img_size, in_chans, num_classes, etc.) remaining
    as direct constructor arguments for API compatibility.
    """

    # Architecture parameters
    patch_size: int = 1  # force to be 1
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
    norm_layer: Optional[str] = "rmsnorm"  # Normalization layer for transformer blocks
    act_layer: Optional[str] = None  # Activation layer for MLP blocks
    block_fn: Optional[str] = None  # Transformer block implementation class name
    mlp_layer: Optional[str] = None  # MLP implementation class name

    # EVA-specific parameters
    attn_type: str = "eva"  # Attention type: 'standard', 'eva', 'rope'
    swiglu_mlp: bool = True  # Use SwiGLU MLP variant
    qkv_fused: bool = True  # Whether to use fused QKV projections

    # Variable patch size support
    enable_patch_interpolator: bool = True  # Enable dynamic patch size support

    #  Tokenization related
    img_size: int = 32
    z_dim: int = 256
    out_chans: int = 16
    out_2d_latent: bool = True
    unpatch_size: Optional[int] = None  # if None, use patch_size
    compile_model: bool = False

    # for cross-attention
    cross_attn_tokens: int = -1
    cross_attn_ratio: float = 0.5


class Transformer(NaFlexVit):
    def __init__(self, cfg: NaFlexVitCfg):
        super().__init__(cfg, in_chans=cfg.z_dim, img_size=cfg.img_size)
        self.cfg = cfg
        self._build_head(cfg)

        if cfg.compile_model:
            logger.info(f"[Naflex Transformer]: Compiling model ...")
            for i in range(len(self.blocks)):
                self.blocks[i] = torch.compile(self.blocks[i])
            self.head = torch.compile(self.head)

    def _build_head(self, cfg: NaFlexVitCfg):
        norm_layer = get_norm_layer(cfg.norm_layer) or nn.LayerNorm
        self.patch_size = cfg.patch_size
        self.unpatch_size = cfg.unpatch_size or cfg.patch_size
        self.head = nn.Sequential(
            norm_layer(cfg.embed_dim),
            nn.Linear(cfg.embed_dim, cfg.out_chans * self.unpatch_size**2, bias=True),
        )

    def unpatchify(self, x: torch.Tensor, hw=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.cfg.out_chans
        p = self.unpatch_size
        if hw is None:
            h = w = int(x.shape[1] ** 0.5)
            assert h * w == x.shape[1]
        else:
            h, w = hw

        x = einops.rearrange(
            x,
            "bs (h w) (p1 p2 c) -> bs c (h p1) (w p2)",
            h=h // p,
            w=w // p,
            p1=p,
            p2=p,
            c=c,
        )
        return x

    def _forward_after_backbone(self, x, hw):
        x = self.head(x)
        if self.cfg.out_2d_latent:
            x = self.unpatchify(x, hw)
        return x

    def _get_output_shape(self, x):
        hw = x.shape[-2:]
        if self.cfg.unpatch_size is not None:
            out_hw = (torch.tensor(hw) // self.patch_size * self.unpatch_size).tolist()
        else:
            out_hw = hw

    def forward(self, x):
        # Output HW
        out_hw = self._get_output_shape(x)

        # Features
        x = self.forward_features(x)
        x = x[:, self.cfg.reg_tokens :]

        # Head
        x = self._forward_after_backbone(x, out_hw)

        return x

    @function_config_to_basic_types
    @staticmethod
    def create_model(**overrides):
        cfg = dataclass_from_dict(NaFlexVitCfg, overrides)
        model = Transformer(cfg)
        return model


def test_naflex_vit_pansharpening_model():
    torch.cuda.set_device(0)
    cfg = NaFlexVitCfg(
        embed_dim=512,
        depth=8,
        num_heads=8,
        rope_type="axial",
        pos_embed="learned",
        pos_embed_grid_size=(32, 32),
        reg_tokens=8,
        patch_size=2,
        img_size=32,
        out_chans=256,
        unpatch_size=2,
        z_dim=256,
    )

    x = torch.randn(1, 256, 32, 32).cuda()

    model = Transformer(cfg).cuda()

    # Test standard forward
    print("=== Testing standard forward ===")
    out = model(x)
    print(f"Standard forward output shape: {out.shape}")  # [1, 256, 64, 64]

    # Test forward_intermediates
    print("\n=== Testing forward_intermediates ===")

    # Test with different indices configurations
    test_configs = [
        {"indices": None, "desc": "all layers"},
        {"indices": [2, 4, 6], "desc": "specific layers [2,4,6]"},
        {"indices": 3, "desc": "last 3 layers"},
        {"indices": [0, 7], "desc": "first and last layers"},
    ]

    for config in test_configs:
        print(f"\n--- Testing {config['desc']} ---")
        try:
            result = model.forward_intermediates(
                x,
                indices=config["indices"],
                return_prefix_tokens=True,
                norm=True,
                output_fmt="NCHW",
                output_dict=True,
            )

            if isinstance(result, dict):
                print(f"Output keys: {list(result.keys())}")
                if "image_features" in result:
                    print(f"Final features shape: {result['image_features'].shape}")
                if "image_intermediates" in result:
                    intermediates = result["image_intermediates"]
                    print(f"Number of intermediate layers: {len(intermediates)}")
                    for i, feat in enumerate(intermediates):
                        print(
                            f"  Layer {config['indices'][i] if config['indices'] is not None and isinstance(config['indices'], list) else i}: {feat.shape}"
                        )
                if "image_intermediates_prefix" in result:
                    prefix_intermediates = result["image_intermediates_prefix"]
                    print(
                        f"Prefix intermediates shape: {[p.shape for p in prefix_intermediates]}"
                    )
            else:
                print(f"Return type: {type(result)}")

        except Exception as e:
            print(f"Error with {config['desc']}: {e}")

    # Test gradient computation
    print("\n=== Testing gradient computation ===")
    out.mean().backward()

    # Check gradients
    missing_grads = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            missing_grads.append(name)

    if missing_grads:
        print(f"Parameters without gradients: {missing_grads}")
    else:
        print("All parameters have gradients!")

    # Check FLOPs
    print("\n=== Computing FLOPs ===")
    try:
        from fvcore.nn import FlopCountAnalysis, flop_count_table

        flops = FlopCountAnalysis(model, (x,))
        print(flop_count_table(flops))
    except ImportError:
        print("fvcore not available, skipping FLOP analysis")


if __name__ == "__main__":
    """
    export LOVELY_TENSORS=1
    python -m src.stage1.cosmos.modules.naflex
    """
    with logger.catch():
        test_naflex_vit_pansharpening_model()

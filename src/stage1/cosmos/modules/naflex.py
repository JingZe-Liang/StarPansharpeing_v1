from timm.layers.create_norm import create_norm_layer
import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import einops
import torch

import torch.nn as nn
from torch import Tensor
from easydict import EasyDict as edict
from einx import get_at
from loguru import logger
from timm.layers import apply_keep_indices_nlc, to_2tuple
from timm.layers.create_norm import get_norm_layer
from timm.layers.helpers import to_2tuple
from timm.layers.patch_embed import PatchEmbed
from timm.layers.pos_embed import resample_abs_pos_embed
from timm.layers import get_act_layer, Mlp, LayerScale
from timm.models import eva, naflexvit
from timm.models._manipulate import named_apply
from timm.models.eva import AttentionRope, DropPath, EvaAttention, GluMlp, Mlp, SwiGLU
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from timm.models.naflexvit import (
    Block,
    NaFlexEmbeds,
    NaFlexVit,
    # checkpoint,
    create_attention_mask,
    feature_take_indices,
    get_init_weights_vit,
)
from timm.layers.drop import calculate_drop_path_rates
from diffusers.models.embeddings import get_2d_sincos_pos_embed

from src.utilities.config_utils import dataclass_from_dict, function_config_to_basic_types
from src.utilities.network_utils import compile_decorator

from .norm import *  # register custom norms
from .transformer import GatedAttention


def nepa_prediction_loss(h_in: Tensor, h_out: Tensor, shift: bool = True) -> Tensor:
    """
    Next Embedding Prediction loss (negative cosine similarity).

    This loss encourages the model to predict the next position's input embedding
    from the current position's output hidden state, similar to autoregressive
    language modeling but in latent space.

    Args:
        h_in:  [B, T, D] input embeddings (target, will be detached)
        h_out: [B, T, D] output hidden states (prediction)
        shift: if True, predict h_in[i+1] from h_out[i] (next token prediction)
               if False, predict h_in[i] from h_out[i] (position-wise matching)

    Returns:
        loss: scalar, negative cosine similarity in range [-1, 1]
    """
    # Detach target to prevent gradient flow
    h_in = h_in.detach()

    if shift:
        # Next token prediction: h_out[i] predicts h_in[i+1]
        p = h_out[:, :-1, :]  # positions 0 to T-2
        z = h_in[:, 1:, :]  # positions 1 to T-1
    else:
        # Position-wise matching
        p = h_out
        z = h_in

    # L2 normalize along feature dimension
    p = torch.nn.functional.normalize(p, dim=-1)
    z = torch.nn.functional.normalize(z, dim=-1)

    # Negative cosine similarity (minimize to maximize similarity)
    loss = -(p * z).sum(dim=-1).mean()
    return loss


class GatedAttentionTimmWrapped(GatedAttention):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qkv_fused: bool = True,
        qkv_bias_separate: bool = False,
        num_prefix_tokens: int = 1,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_head_dim: Optional[int] = None,
        norm_layer: Optional[Callable] = None,
        qk_norm: bool = False,
        scale_norm: bool = True,
        rotate_half: bool = False,
        is_causal: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__(
            dim,
            num_heads,
            num_heads,
            norm_layer,
            qk_norm,
            qkv_bias,
            num_prefix_tokens=num_prefix_tokens,
            attention_dropout=attn_drop,
            headwise_attn_output_gate=True,
            elementwise_attn_output_gate=False,
            is_causal=is_causal,
        )


class EvaBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        qkv_fused: bool = True,
        mlp_ratio: float = 4.0,
        swiglu_mlp: bool = False,
        swiglu_align_to: int = 0,
        scale_mlp: bool = False,
        scale_attn_inner: bool = False,
        num_prefix_tokens: int = 1,
        attn_type: str = "eva",
        rotate_half: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: Optional[float] = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        attn_head_dim: Optional[int] = None,
        is_causal: bool = False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        """Initialize the EVA transformer block.

        Args:
          dim: Input dimension of the token embeddings
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias terms in query, key, value projections
            qkv_fused: Whether to use a single projection for query, key, value
            mlp_ratio: Ratio of MLP hidden dimension to input dimension
            swiglu_mlp: Whether to use SwiGLU activation in the MLP
            scale_mlp: Whether to use normalization in the MLP
            scale_attn_inner: Whether to use normalization within the attention mechanism
            num_prefix_tokens: Number of tokens at the beginning of the sequence (class tokens, etc.)
            attn_type: Type of attention module to use ('eva' or 'rope')
            proj_drop: Dropout rate for projection layers
            attn_drop: Dropout rate for attention matrix
            drop_path: Stochastic depth rate
            init_values: Initial value for LayerScale, None = no LayerScale
            act_layer: Activation layer constructor
            norm_layer: Normalization layer constructor
            attn_head_dim: Dimension of each attention head (if None, computed as dim // num_heads)
        """
        dd = {"device": device, "dtype": dtype}
        super().__init__()

        self.norm1 = norm_layer(dim, **dd)
        logger.debug(f"Layer uses attention type {attn_type}")
        attn_cls = {"rope": AttentionRope, "eva": EvaAttention, "gated": GatedAttentionTimmWrapped}[attn_type]

        attn_kwargs = {}
        if attn_type == "gated":
            attn_kwargs["is_causal"] = is_causal

        self.attn = attn_cls(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_fused=qkv_fused,
            num_prefix_tokens=num_prefix_tokens,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_head_dim=attn_head_dim,
            norm_layer=norm_layer,
            scale_norm=scale_attn_inner,
            rotate_half=rotate_half,
            **attn_kwargs,
            **dd,
        )
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim, **dd)) if init_values is not None else None
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim, **dd)
        hidden_features = int(dim * mlp_ratio)
        if swiglu_mlp:
            if scale_mlp or swiglu_align_to:
                # when norm in SwiGLU used or alignment enabled, an impl with separate fc for gate & x is used
                self.mlp = SwiGLU(
                    in_features=dim,
                    hidden_features=hidden_features,
                    norm_layer=norm_layer if scale_mlp else None,
                    drop=proj_drop,
                    align_to=swiglu_align_to,
                    **dd,
                )
            else:
                # w/o any extra norm, an impl with packed weights is used
                self.mlp = GluMlp(
                    in_features=dim,
                    hidden_features=hidden_features * 2,
                    norm_layer=norm_layer if scale_mlp else None,
                    act_layer=nn.SiLU,
                    gate_last=False,
                    drop=proj_drop,
                    **dd,
                )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=hidden_features,
                act_layer=act_layer,
                norm_layer=norm_layer if scale_mlp else None,
                drop=proj_drop,
                **dd,
            )
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim, **dd)) if init_values is not None else None
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    @compile_decorator
    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.gamma_1 is None:
            x = x + self.drop_path1(self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask))
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask))
            x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


eva.EvaBlock = EvaBlock  # type: ignore[invalid-assignment]


def get_block_fn(cfg) -> Callable:
    """Get appropriate block function based on configuration.

    Returns a partially applied block constructor with EVA-specific
    or conflicting parameters pre-configured if needed.
    """
    # Check if we need EVA block features
    use_eva_features = (
        cfg.attn_type in ("eva", "rope", "gated")
        or cfg.rope_type not in ("", "none")  # Any ROPE type requires EVA blocks
        or cfg.swiglu_mlp
    )

    if use_eva_features:
        # Determine attention type based on rope_type if not explicitly set
        attn_type = cfg.attn_type
        if attn_type == "standard" and cfg.rope_type not in ("", "none"):
            attn_type = "rope"

        num_prefix_tokens = (1 if cfg.class_token else 0) + cfg.reg_tokens
        return partial(
            EvaBlock,
            attn_type=attn_type,
            swiglu_mlp=cfg.swiglu_mlp,
            scale_mlp=cfg.scale_mlp_norm,
            scale_attn_inner=cfg.scale_attn_inner_norm,
            qkv_fused=cfg.qkv_fused,
            num_prefix_tokens=num_prefix_tokens,
            is_causal=getattr(cfg, "is_causal", False),  # despite the 'nepa' pretrained task, `is_causal` is False
        )
    else:
        # Standard ViT block
        block_fn = cfg.block_fn or Block
        if cfg.scale_mlp_norm or cfg.scale_attn_inner_norm:
            # param names differ between EVA vs non-EVA block types
            block_fn = partial(
                block_fn,
                scale_mlp_norm=cfg.scale_mlp_norm,
                scale_attn_norm=cfg.scale_attn_inner_norm,
            )
        return block_fn


naflexvit.get_block_fn = get_block_fn  # type: ignore

# -------------- Naflex Config ------------------ #


@dataclass
class NaFlexVitCfg:
    """Configuration for FlexVit model.

    This dataclass contains the bulk of model configuration parameters,
    with core parameters (img_size, in_chans, num_classes, etc.) remaining
    as direct constructor arguments for API compatibility.
    """

    # Architecture parameters
    patch_size: int = 1
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
    init_values: Optional[float] = None  # Layer-scale init values (layer-scale enabled if not None)
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
    # Grid size for position embedding initialization
    pos_embed_grid_size: Optional[Tuple[int, int]] = (16, 16)
    pos_embed_interp_mode: str = "bicubic"  # Interpolation mode for position embedding resizing
    pos_embed_ar_preserving: bool = False  # Whether to preserve aspect ratio during position embedding interpolation
    pos_embed_use_grid_sample: bool = False  # Whether to use grid_sample for naflex position embedding interpolation

    # ROPE specific configuration
    rope_type: str = (
        "axial"  # ROPE type: '' or 'none' for no ROPE, 'axial' for standard, 'mixed' for learnable frequencies
    )
    rope_temperature: float = 10000.0  # Temperature for ROPE frequency computation
    rope_ref_feat_shape: Optional[Tuple[int, int]] = None
    rope_grid_offset: float = 0.0  # Grid offset for non-pixel ROPE mode
    rope_grid_indexing: str = "ij"  # Grid indexing mode for ROPE ('ij' or 'xy')

    # Image processing
    dynamic_img_pad: bool = False  # Whether to enable dynamic padding for variable resolution

    # Other architecture choices
    pre_norm: bool = True  # Whether to apply normalization before attention/MLP layers (start of blocks)
    final_norm: bool = True  # Whether to apply final normalization before pooling and classifier (end of blocks)
    fc_norm: Optional[bool] = None  # Whether to normalize features before final classifier (after pooling)

    # Global pooling setup
    global_pool: str = ""  # Type of global pooling for final sequence
    pool_include_prefix: bool = False  # Whether to include class/register prefix tokens in global pooling
    attn_pool_num_heads: Optional[int] = None  # Override num_heads for attention pool
    attn_pool_mlp_ratio: Optional[float] = None  # Override mlp_ratio for attention pool

    # Weight initialization
    weight_init: str = "jax"  # Weight initialization scheme
    fix_init: bool = True  # Apply weight initialization fix (scaling w/ layer index)

    # Embedding configuration
    embed_proj_type: str = "linear"  # Type of embedding layer ('conv' or 'linear')
    input_norm_layer: Optional[str] = None  # Normalization layer for embeddings input (before input projection)
    embed_norm_layer: Optional[str] = None  # Normalization layer for embeddings (after input projection)

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

    # Tokenization related
    img_size: int = 32
    in_chans: int = 256
    out_chans: int = 16
    out_2d_latent: bool = True
    unpatch_size: Optional[int] = None  # if None, use patch_size
    compile_model: bool = False  # control torch.compile

    # for cross-attention
    cross_attn_tokens: int = -1
    cross_attn_ratio: float = 0.5

    # adaptive generation decoder
    is_first_cat_noise: bool = False

    # 'ijepa', 'lejepa', None for no pretrained task
    pretrained_type: Any = None  # Union[str, list[str]]

    # IBOT head cfgs
    ibot_n_prototypes: int = 65536
    ibot_head_hidden_dim: int = 2048
    ibot_bottleneck_dim: int = 256
    ibot_nlayers: int = 3

    # MAE head cfgs
    mae_decoder_dim: int = 768
    mae_decoder_pe_init_type: str = "trunc_normal"  # trunc_normal or sincos
    mae_decoder_depth: int = 8
    mae_mask_type: str = "kaiming"  # 'kaiming' or 'pixio'
    mae_mask_ratio: float = 0.8
    mae_decoder_head: str = "seperated"  # 'shared' or 'seperated'
    mae_pixio_mask_grid: int = 2
    mae_latent_size: int = 14  # latent size for mae

    # NePA (Next Embedding Prediction) cfgs
    nepa_is_causal: bool = True  # Whether to use causal attention for NePA
    nepa_shift_prediction: bool = True  # Whether to predict next position (shift by 1)
    # attn_mask_type: 'is_causal' (implicit, requires gated attn) or 'explicit' (default) or None (auto)
    nepa_attn_mask_type: Optional[str] = "explicit"
    is_causal: bool = False  # Global model causality (if True, applies to all tasks)


def ffn_init_fn(module: nn.Module, d_model: int, d_ffn: int, layer_id: int | None = None):
    std = 1.0 / math.sqrt(d_model)
    torch.nn.init.trunc_normal_(module.layer1.weight, std=std, a=-3 * std, b=3 * std)

    # scale init by depth as in https://arxiv.org/abs/1908.11365 -- worked slightly better.
    std = 1.0 / math.sqrt(d_ffn)
    if layer_id is not None:
        std = std / math.sqrt(2 * (layer_id + 1))
    torch.nn.init.trunc_normal_(module.layer2.weight, std=std, a=-3 * std, b=3 * std)


def attention_init_fn(module, layer_id: int | None = None):
    std = 1.0 / math.sqrt(module.q_dim)
    torch.nn.init.trunc_normal_(module.q_proj.weight, std=std, a=-3 * std, b=3 * std)
    std = 1.0 / math.sqrt(module.ctx_dim)
    torch.nn.init.trunc_normal_(module.k_proj.weight, std=std, a=-3 * std, b=3 * std)
    torch.nn.init.trunc_normal_(module.v_proj.weight, std=std, a=-3 * std, b=3 * std)

    std = 1.0 / math.sqrt(module.inner_dim)
    torch.nn.init.trunc_normal_(module.output_proj.weight, std=std, a=-3 * std, b=3 * std)

    for layer in module.q_norm, module.k_norm, module.v_norm:
        if hasattr(layer, "init_weights"):
            layer.init_weights()


class Transformer(NaFlexVit):
    def __init__(self, cfg: NaFlexVitCfg):
        super().__init__(cfg, in_chans=cfg.in_chans, img_size=cfg.img_size)
        self.cfg = cfg
        self._build_head(cfg)

        if cfg.compile_model:
            logger.log("NOTE", f"[Naflex Transformer]: Compiling model ...")
            for i in range(len(self.blocks)):
                self.blocks[i] = torch.compile(self.blocks[i])
            # self.head = torch.compile(self.head)

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

    def _forward_after_backbone(self, x, hw: list | None):
        x = self.head(x)
        if self.cfg.out_2d_latent and hw is not None:
            x = self.unpatchify(x, hw)
        else:
            assert hw is None and self.unpatch_size == 1, (
                f"HW is not None or the unpatch_size is not 1, when force no patchify"
            )
            # Let the output be 1D tensor
        return x

    def _get_output_shape(self, x):
        hw = x.shape[-2:]
        if self.cfg.unpatch_size is not None:
            out_hw = (torch.tensor(hw) // self.patch_size * self.unpatch_size).tolist()
        else:
            out_hw = hw
        return out_hw

    def forward(self, x, output_type: str | None = None, **_ignored_kwargs):
        # Output HW
        if output_type in (None, "2d"):
            out_hw = self._get_output_shape(x)
        else:
            out_hw = None  # keep the output to be 1D tensor

        # Features
        x = self.forward_features(x)
        x = cast(torch.Tensor, x)
        x = x[:, self.num_prefix_tokens :]

        # Head
        x = self._forward_after_backbone(x, out_hw)

        return x

    @classmethod
    @function_config_to_basic_types
    def create_model(cls, **overrides):
        cfg: NaFlexVitCfg = dataclass_from_dict(NaFlexVitCfg, overrides)

        # Support for NePA implicit causal mask
        # Configure attn_mask_type: 'is_causal' (implicit) or 'explicit'
        if cfg.pretrained_type is not None and "nepa" in cfg.pretrained_type and cfg.nepa_is_causal:
            mask_type = cfg.nepa_attn_mask_type

            # Auto-selection if None
            if mask_type is None:
                mask_type = "is_causal" if cfg.attn_type == "gated" else "explicit"

            if mask_type == "is_causal":
                if cfg.attn_type == "gated":
                    cfg.is_causal = True
                    logger.info("[NePA]: Enabled implicit causal attention (is_causal=True for GatedAttention)")
                else:
                    logger.warning(
                        f"[NePA]: 'is_causal' mask requested but attn_type='{cfg.attn_type}' "
                        "does not support it. Falling back to explicit mask."
                    )
                    cfg.is_causal = False
            else:
                # Explicit mode
                cfg.is_causal = False

        model = cls(cfg)
        return model

    def init_weights(self, mode="jax"):
        super().init_weights(mode=mode)

        def rescale(p, layer_id):
            p.div_(math.sqrt(2.0 * layer_id))

        # Rescale the depth
        rescale_layer = True
        if rescale_layer:
            for layer_id, blk in enumerate(self.blocks, 1):
                rescale(blk.attn.proj.weight.data, layer_id)
                rescale(blk.mlp.fc2.weight.data, layer_id)
            logger.info("[Naflex Transformer]: Rescale the layers initialization for more stable training")


class IJEPANaFlexViT(Transformer):
    def __init__(self, cfg: NaFlexVitCfg):
        super().__init__(cfg)

        self.pretrained_type: str = cfg.pretrained_type

        # --------- Build the heads or decoders --------- #
        if "lejepa" in self.pretrained_type:
            self._build_lejepa_head(cfg)
        if "ibot" in self.pretrained_type:
            self._build_ibot_head(cfg)
        if "latent_mae" in self.pretrained_type or "pixel_mae" in self.pretrained_type:
            self._build_mae_decoder(cfg)
        if "nepa" in self.pretrained_type:
            self._build_nepa_head(cfg)

    def _build_mae_decoder(self, cfg: NaFlexVitCfg):
        """
        Latent MAE is more like JEPA,
            image x -> CNN encoder -> latent -> mask inside Naflex transformer (MAE encoder) ->
                -> merge masked tokens in  Naflex transformer (MAE decoder) -> predict the masked tokens (MSE loss)
            this type MAE works at latent, so called latent_mae

        Pixel MAE is more like the original pixel-space MAE,
            image x -> CNN encoder -> latent -> mask inside Naflex transformer (MAE encoder) ->
                -> merge masked tokens in  Naflex transformer (MAE decoder) -> CNN decoder -> reconstruct the image
            this type MAE works at pixel, so called pixel_mae
        """
        assert "latent_mae" in cfg.pretrained_type or "pixel_mae" in cfg.pretrained_type, "MAE decoder is not supported"
        embed_dim = cfg.mae_decoder_dim
        mae_decoder_depth = cfg.mae_decoder_depth
        block_fn = get_block_fn(cfg)
        dpr = calculate_drop_path_rates(cfg.drop_path_rate, cfg.depth)  # stochastic depth decay rule
        norm_layer = get_norm_layer(cfg.norm_layer) or nn.LayerNorm
        act_layer = get_act_layer(cfg.act_layer) or nn.GELU
        mlp_layer = cfg.mlp_layer or Mlp
        dd: dict = {"device": None, "dtype": None}
        self.mae_decoder = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    qkv_bias=cfg.qkv_bias,
                    qk_norm=cfg.qk_norm,
                    proj_bias=cfg.proj_bias,
                    init_values=cfg.init_values,
                    proj_drop=cfg.proj_drop_rate,
                    attn_drop=cfg.attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                    **dd,
                )
                for i in range(mae_decoder_depth)
            ]
        )
        logger.debug("Build MAE decoder", name="MAE Naflex Transformer")

        # --------- Decoder embeddings, mask token, and decoder norm/predictor ----------- #

        # Init the decoder blocks
        init_mode = "timm"
        init_fn = get_init_weights_vit(mode=init_mode)
        # mae implement this using jax init fn
        self.mae_decoder.apply(init_fn)

        n_patches = (cfg.mae_latent_size // cfg.patch_size) ** 2
        # the pixio implementation fix the norm at the encoder's head
        # here we move the norm with mae embedder
        self.mae_embedder = nn.Sequential(
            create_norm_layer(cfg.norm_layer, cfg.embed_dim),
            nn.Linear(cfg.embed_dim, cfg.mae_decoder_dim, bias=True),
        )
        self.mae_mask_token = nn.Parameter(torch.zeros(1, 1, cfg.mae_decoder_dim))
        self.mae_pos_embed = nn.Parameter(torch.zeros(1, n_patches + self.num_prefix_tokens, cfg.mae_decoder_dim))
        if cfg.mae_decoder_head == "seperated":
            self.mae_head = nn.Sequential(
                norm_layer(cfg.mae_decoder_dim),
                nn.Linear(cfg.mae_decoder_dim, cfg.patch_size**2 * cfg.out_chans, bias=True),
            )

        # init weights
        pos_embed_init_type: str = cfg.mae_decoder_pe_init_type  # [trunc_norm, sincos]
        if pos_embed_init_type == "trunc_normal":
            nn.init.trunc_normal_(self.mae_pos_embed, std=0.02)
        elif pos_embed_init_type == "sincos":
            init_pe = get_2d_sincos_pos_embed(
                cfg.mae_decoder_dim,
                grid_size=cfg.img_size // cfg.patch_size,
                cls_token=True if self.num_prefix_tokens > 0 else False,
                extra_tokens=self.num_prefix_tokens,
                output_type="pt",
            )
            with torch.no_grad():
                self.mae_pos_embed.copy_(init_pe.unsqueeze(0))
        nn.init.normal_(self.mae_mask_token, std=0.02)
        logger.debug("Init the MAE decoder", name="MAE Naflex Transformer")

    def _build_lejepa_head(self, cfg):
        from src.stage1.self_supervised.lejepa_aug import create_lejepa_projector

        self.lejepa_projector = create_lejepa_projector(
            cfg.embed_dim,
            cfg.embed_dim,
            mean_out_hw=False,
            # mean_out_hw=not cfg.class_token,  # if use class, not mean out the spatial tokens
        )
        logger.info("[IJEPA Naflex Transformer]: Build LeJEPA head")

    def _build_ibot_head(self, cfg):
        from src.stage1.self_supervised.dino.layers.dino_head import DINOHead

        self.ibot_head = DINOHead(
            in_dim=cfg.embed_dim,
            out_dim=cfg.ibot_n_prototypes,
            hidden_dim=cfg.ibot_head_hidden_dim,
            bottleneck_dim=cfg.ibot_bottleneck_dim,
            nlayers=cfg.ibot_nlayers,
        )
        self.mask_token = nn.Parameter(torch.zeros(1, cfg.embed_dim))
        logger.info("[IBOT Naflex Transformer]: Build iBOT head")

    def _build_nepa_head(self, cfg: NaFlexVitCfg):
        """Build NePA (Next Embedding Prediction) pretraining components.

        NePA doesn't require an additional head - the loss is computed directly
        on the embedding space using cosine similarity between transformer output
        and (shifted) input embeddings.
        """
        self.nepa_is_causal = cfg.nepa_is_causal
        self.nepa_shift = cfg.nepa_shift_prediction
        logger.info(
            f"[NePA NaFlex Transformer]: Build NePA pretraining support "
            f"(causal={self.nepa_is_causal}, shift={self.nepa_shift}, nepa_attn_mask_type={cfg.nepa_attn_mask_type})"
        )

    def _forward_nepa_backbone(
        self,
        x: torch.Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass for NePA pretraining with optional causal attention.

        NePA (Next Embedding Prediction) trains the model to predict the next
        position's input embedding from the current position's output. This is
        similar to autoregressive language modeling but in latent space.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            h_in: [B, T, D] input embeddings before transformer blocks
            h_out: [B, T, D] output hidden states after transformer blocks
        """
        naflex_mode = False

        # 1. Get embeddings (before transformer blocks)
        embeds = self._forward_embeds(x, patch_coord=None, patch_valid=None, attn_mask=None, masks=None)
        h_in = embeds["patches"].clone()  # [B, T, D] - input embeddings
        x_forward = embeds["patches"]
        rope_embeds = embeds.get("rope_embeds", None)

        # 2. Create causal attention mask if enabled
        causal_mask = None

        # Check if we should use implicit causal implementation
        # Implicit mode is active if model.cfg.is_causal is True.
        # This was configured in create_model based on nepa_attn_mask_type.
        is_model_causal = getattr(self.cfg, "is_causal", True)

        # Only create explicit mask if NePA implies causal AND model is NOT natively causal
        if getattr(self, "nepa_is_causal", True) and not is_model_causal:
            seq_len = x_forward.shape[1]
            # Upper triangular mask: position i cannot attend to positions > i
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x_forward.device, dtype=x_forward.dtype),
                diagonal=1,
            )
            # Convert to attention mask format (0 = attend, -inf = mask)
            causal_mask = causal_mask * torch.finfo(x_forward.dtype).min

        # 3. Forward through transformer blocks with causal attention
        do_checkpointing = self.grad_checkpointing and self.training and not torch.jit.is_scripting()
        for blk in self.blocks:
            if do_checkpointing:
                x_forward = torch_checkpoint(blk, x_forward, rope_embeds, causal_mask, use_reentrant=False)
            else:
                x_forward = blk(x_forward, rope=rope_embeds, attn_mask=causal_mask)

        # 4. Apply final layer norm
        h_out = self.norm(x_forward)  # [B, T, D] - output hidden states

        return h_in, h_out

    def _prepare_masks(self, masks=None):
        """Ensure the masks are list of tensors"""
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        return masks

    def _ibot_apply_masks(self, x, masks: torch.BoolTensor):
        if masks is not None:
            assert torch.is_tensor(masks)
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
        return x

    def _jepa_apply_masks(self, x, masks: list[Tensor]):
        all_x = []
        for m in masks:
            mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
            all_x += [torch.gather(x, dim=1, index=mask_keep)]
        return torch.cat(all_x, dim=0)

    def _mae_apply_masks(self, x, masks: torch.IntTensor):
        """masks is a Tensor of indices of [B, S_masked]"""
        D = x.size(-1)
        x = torch.gather(x, dim=1, index=masks.unsqueeze(-1).repeat(1, 1, D))
        return x

    def _forward_embeds(
        self,
        x,
        patch_coord,
        patch_valid,
        attn_mask,
        masks: List[Tensor] | Tensor | None = None,
    ):
        """Forward pass through patch / abs pos / rope pos embeds and patch dropout

        IJEPA masking strategy: mask out the patches, and drop them.
            For this instance, `masks` is a list of Int32 mask indices shaped as [B, S_masked]
                e.g., [tensor([[ 14,  28,  42,  56,  70,  84,  98, 112, 126, 140, 154],
                        [ 30,  31,  32,  44,  45,  46,  58,  59,  60,  72,  73]])]
                that can be torch.gather indexed.
            RoPE actions:
            Equals at
            # axial rope: [S, headD * 2]
            # rope: [S, headD * 2] -> [B, 1, S, headD * 2]
            rope_embeds = rope_embeds[None, None].repeat(x.shape[0], 1, 1, 1)
            # masks -> [B, S_masked] -> [B, 1, S_masked, headD * 2]
            # gather -> [B, S_masked, headD * 2]
            m_repeated = m[:, None, :, None].repeat(
                1, 1, 1, rope_embeds.size(-1)
            )
            rope_masked += [rope_embeds.gather(-2, m_repeated)]

        IBOT masking strategy: mask the patches with a learnable token and not drop them.
            where(masks, mask_token, x)
        """
        naflex_mode = patch_coord is not None
        # patch embed, abs pos embed, returns global grid size as calculated from 'standard' NCHW batches
        x, grid_size = self.embeds(x, patch_coord=patch_coord, patch_valid=patch_valid)

        # Generate ROPE embeddings at model level
        rope_embeds = None
        if self.rope is not None:
            if patch_coord is not None:
                # NaFlex mode - variable grid sizes
                rope_embeds = self._generate_rope_naflex(x, patch_coord)
            elif grid_size is not None:
                # Standard mode - fixed grid size
                rope_embeds = self.rope.get_embed(shape=grid_size)
            else:
                assert False, "Expected one of patch_coord or grid_size to be valid"

        # Apply patch dropout with coordinated updates
        keep_indices: Optional[torch.Tensor] = None
        if self.training and self.patch_drop is not None:
            x, keep_indices = self.patch_drop(x)
            # keep_indices excludes prefix tokens, can use directly on patch_valid & rope embeds
            if patch_valid is not None:
                patch_valid = patch_valid.gather(1, keep_indices)
            if rope_embeds is not None and not self.rope_is_mixed:
                # Update ROPE embeddings to match dropped tokens (only for axial mode)
                # Batch dim already present in NaFlex mode, but will be added in standard mode.
                rope_embeds = apply_keep_indices_nlc(x, rope_embeds, keep_indices, pos_embed_has_batch=naflex_mode)
                if not naflex_mode:
                    # B, N, dim -> B, 1, N, dim. Need head dim added for standard mode, already added in NaFlex.
                    rope_embeds = rope_embeds.unsqueeze(1)

        # Create attention mask from patch_valid after patch dropout applied
        if attn_mask is None:
            attn_mask = create_attention_mask(patch_valid, num_prefix_tokens=self.num_prefix_tokens, dtype=x.dtype)

        # ------------ Apply masks ------------ #
        if masks is not None:
            # Apply masks
            prefixed_tokens, x = x[:, : self.num_prefix_tokens], x[:, self.num_prefix_tokens :]
            if "ibot" in self.pretrained_type:
                x = self._ibot_apply_masks(x, masks=masks)
            elif "ijepa" in self.pretrained_type:
                x = self._jepa_apply_masks(x, masks=masks)
            elif self._is_mae():
                x = self._mae_apply_masks(x, masks=masks)

            x = torch.cat([prefixed_tokens, x], dim=1)

            # Rope related, rope is applied in attention module
            if rope_embeds is not None:
                ##### IJEPA masking strategy: mask out
                if "ijepa" in self.pretrained_type:
                    rope_masked = []
                    for m in masks:
                        # m: [B, S_masked] indices
                        assert not self.rope_is_mixed, "mixed rope is not supported in JEPA training"
                        rope_masked += [get_at("[S] ropeD, B S_masked -> B 1 S_masked ropeD", rope_embeds, m)]
                    rope_embeds = torch.cat(rope_masked, dim=0)  # [B*n_masks, 1, S_masked, ropeD]
                elif self._is_mae():
                    # masks: [B, S_masked] indices, can be cat since the MAE has the same length token to keep
                    # for each sample in a batch
                    assert torch.is_tensor(masks), f"masks should be a tensor, got {type(masks)}"
                    rope_embeds = get_at("[S] ropeD, B S_masked -> B 1 S_masked ropeD", rope_embeds, masks)
                else:
                    # Do nothing to rope, only mask the embeded tokens
                    masks = cast(Tensor, masks)

        x = self.norm_pre(x)
        return {
            "patches": x,
            "patch_valid": patch_valid,
            "rope_embeds": rope_embeds,
            "attn_mask": attn_mask,
            "keep_indices": keep_indices,
        }

    def forward_features(
        self,
        patches: torch.Tensor,
        patch_coord: Optional[torch.Tensor] = None,
        patch_valid: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        masks: Optional[Tensor | List[Tensor]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """ """
        naflex_mode = patch_coord is not None

        # Pass through patch & abs position embedding module with patch coordinate/type support
        embeds = self._forward_embeds(
            patches,
            patch_coord=patch_coord,
            patch_valid=patch_valid,
            attn_mask=attn_mask,
            masks=masks,
        )
        x = embeds["patches"]
        rope_embeds = embeds.get("rope_embeds", None)
        keep_indices = embeds.get("keep_indices", None)
        attn_mask = embeds.get("attn_mask", None)

        # Apply transformer blocks with masked attention and/or ROPE if provided
        do_checkpointing = self.grad_checkpointing and not torch.jit.is_scripting()
        if self.rope_is_mixed and rope_embeds is not None:
            # Mixed mode with per-layer embeddings (list or iterator)
            for i, (blk, rope_embed) in enumerate(zip(self.blocks, rope_embeds)):
                if self.training and self.patch_drop is not None and keep_indices is not None:
                    # Apply patch dropout to rope_embed if needed (batch dim already present in naflex mode)
                    rope_embed = apply_keep_indices_nlc(
                        x,
                        rope_embed,
                        keep_indices,
                        pos_embed_has_batch=naflex_mode,
                    )
                if do_checkpointing:
                    x = torch_checkpoint(blk, x, rope_embed, attn_mask, use_reentrant=False)
                else:
                    x = blk(x, rope=rope_embed, attn_mask=attn_mask)
        elif rope_embeds is not None:
            # Axial ROPE mode with shared embeddings
            for blk in self.blocks:
                if do_checkpointing:
                    x = torch_checkpoint(blk, x, rope_embeds, attn_mask, use_reentrant=False)
                else:
                    x = blk(x, rope=rope_embeds, attn_mask=attn_mask)
        else:
            for blk in self.blocks:
                if do_checkpointing:
                    x = torch_checkpoint(blk, x, None, attn_mask, use_reentrant=False)
                else:
                    x = blk(x, attn_mask=attn_mask)

        x = self.norm(x)

        if naflex_mode:
            return {"patches": x, "patch_valid": embeds.get("patch_valid", None)}

        return x

    def forward(  # type: ignore[invalid-method-override]
        self,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
        patch_coord: Optional[torch.Tensor] = None,
        patch_valid: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        output_type: str | None = None,
        masks: Optional[Tensor | List[Tensor]] = None,  # IJEPA masks
    ):
        """Forward with JEPA masks support"""
        if masks is not None and self.pretrained_type == "ijepa":
            masks = self._prepare_masks(masks=masks)

        input_is_dict = isinstance(x, Dict)
        naflex_mode = input_is_dict or patch_coord is not None
        if naflex_mode:
            assert masks is None, "JEPA does not support naflex mode."

            if input_is_dict:
                # Handle dictionary input from NaFlex collator, dict inputs take priority over args
                patches = x["patches"]
                patch_valid = x.get("patch_valid", patch_valid)
                patch_coord = x.get("patch_coord", patch_coord)
                attn_mask = x.get("attn_mask", attn_mask)
            else:
                patches = x
            assert patch_coord is not None, "patch_coord is required in naflex mode"
            assert patch_valid is not None, "patch_valid is required in naflex mode"

            features = self.forward_features(
                patches=patches,
                patch_valid=patch_valid,
                patch_coord=patch_coord,
                attn_mask=attn_mask,
                ##### ! is naflex mode, do not input the jepa masks
            )

            # Pass patches & patch_valid to forward_head for masked pooling
            x = self.forward_head(**features)
        else:
            # * This is the Tensor input x forward pass, not naflex mode ##################

            if output_type in (None, "2d"):
                out_hw = self._get_output_shape(x)
            else:
                out_hw = None  # keep the output to be 1D tensor

            # Features
            x = self.forward_features(x, masks=masks)
            x = cast(torch.Tensor, x)
            x = x[:, self.num_prefix_tokens :]

            # Head
            x = self._forward_after_backbone(x, out_hw)

        return x

    def _forward_after_backbone(self, x, hw: list | None) -> tuple[Tensor, Optional[Dict[str, Tensor]]] | Tensor:
        head_out = self.head(x)
        if self.cfg.out_2d_latent and hw is not None:
            out = self.unpatchify(head_out, hw)
        else:
            assert hw is None and self.unpatch_size == 1, (
                f"HW is not None or the unpatch_size is not 1, when force no patchify"
            )
            # Let the output be 1D tensor
            out = head_out

        return out

    def _is_mae(self, pretrained_task: list[str] | None = None):
        pretrained_task = cast(list[str], pretrained_task or self.pretrained_type)
        return "latent_mae" in pretrained_task or "pixel_mae" in pretrained_task

    def _forward_pretrained_backbone(
        self,
        x: torch.Tensor,
        output_type: str = "1d",  # fixed it.
        masks: Optional[Tensor | List[Tensor]] = None,
        masks_indices: Optional[Tensor] = None,
        *,
        pretrained_task: list[str] | None = None,
        **kwargs,
    ):
        terms = {}
        pretrained_task = [] if pretrained_task is None else pretrained_task

        # if output_type in (None, "2d"):
        out_hw = self._get_output_shape(x)
        # else:
        #     out_hw = None  # keep the output to be 1D tensor

        # ---------- forward backbone ---------- #
        x = cast(torch.Tensor, self.forward_features(x, masks=masks))

        # ---------- forward different pretrained heads / decoders ---------- #
        ######### IJepa features ########
        if "ijepa" in self.pretrained_type and "ijepa" in pretrained_task:
            # x is the backbone's out
            terms["ijepa_feat"] = x[:, self.num_prefix_tokens :]

        ######### MAE decoders ########
        if self._is_mae(pretrained_task):
            ids_restore = kwargs.get("ids_restore", None)
            assert ids_restore is not None, "ids_restore is required for MAE decoder"

            # x is masked inside `forward_features`
            x_masked = x
            # 1. embed to decoder dim
            x_dec = self.mae_embedder(x_masked)
            # 2. add mask_token to the masked positions (following pixio pattern)
            x_cls_reg = x_dec[:, : self.num_prefix_tokens, :]
            x_patches_masked = x_dec[:, self.num_prefix_tokens :]
            mask_tokens = self.mae_mask_token.repeat(x.shape[0], ids_restore.shape[1] - x_patches_masked.shape[1], 1)

            # Concatenate visible tokens with mask tokens, then gather (unshuffle)
            x_dec_spatial = torch.cat([x_patches_masked, mask_tokens], dim=1)
            x_dec_spatial = torch.gather(
                x_dec_spatial, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_dec_spatial.shape[2])
            )

            # Append prefix tokens after gather operation
            x_dec = torch.cat([x_cls_reg, x_dec_spatial], dim=1)

            # 3. add decoder's positional embedding
            # Calculate spatial size excluding prefix tokens
            spatial_tokens = x_dec.shape[1] - self.num_prefix_tokens
            hw = int(spatial_tokens**0.5)
            abs_pe = resample_abs_pos_embed(
                self.mae_pos_embed, new_size=[hw, hw], num_prefix_tokens=self.num_prefix_tokens
            )
            x_dec = x_dec + abs_pe
            x_dec = x_dec.contiguous()
            # 4. call decoder blocks and head
            do_checkpointing = self.grad_checkpointing and self.training and not torch.jit.is_scripting()
            for blk in self.mae_decoder:
                if do_checkpointing:
                    x_dec = torch_checkpoint(blk, x_dec, None, None, use_reentrant=False)
                else:
                    x_dec = blk(x_dec)

            if self.cfg.mae_decoder_head == "shared":
                x_dec = self.head(x_dec)
            else:
                x_dec = self.mae_head(x_dec)
            # 5. unpatchify if needed
            # assert out_hw is not None
            p, c, h, w = self.unpatch_size, self.cfg.out_chans, out_hw[0], out_hw[1]
            x_dec_2d = self.unpatchify(x_dec[:, self.num_prefix_tokens :], hw=out_hw)
            # x_dec_2d = einops.rearrange(
            #     x_dec[:, self.num_prefix_tokens:],
            #     "bs (h w) (p1 p2 c) -> bs c (h p1) (w p2)",
            #     h=h // p,
            #     w=w // p,
            #     p1=p,
            #     p2=p,
            #     c=c
            # )
            # 6. form the output
            terms["mae_decode_out"] = x_dec[:, self.num_prefix_tokens :]
            terms["mae_decode_out_2d"] = x_dec_2d

        ######### Lejepa projector #########
        if hasattr(self, "lejepa_projector") and "lejepa" in self.cfg.pretrained_type and "lejepa" in pretrained_task:
            # NOTE: may use all spatial tokens to compute the lejepa loss?
            x_pool = self._pool(x)
            lejepa_proj = self.lejepa_projector(x_pool)  # x is the backbone's out
            terms["lejepa_proj"] = lejepa_proj
            # spatial tokens only
            terms["prefixed_tokens"] = x[:, : self.num_prefix_tokens]

        ########## IBOT projector ##########
        if hasattr(self, "ibot_head") and "ibot" in self.cfg.pretrained_type and "ibot" in pretrained_task:
            terms["ibot_feat"] = x_tokens = x[:, self.num_prefix_tokens :]

            # Check mask shape alignment if masks provided
            if masks is not None and isinstance(masks, Tensor):
                B, N = masks.shape
                assert x_tokens.shape[:2] == (B, N), (
                    f"Mask shape {masks.shape} does not match feature shape {x_tokens.shape[:2]}"
                )

            if masks_indices is not None:
                # IBOT teacher does
                x_tokens = torch.index_select(x_tokens.flatten(0, 1), dim=0, index=masks_indices)
            terms["ibot_proj"] = self.ibot_head(x_tokens)

        ########## NePA (Next Embedding Prediction) ##########
        if hasattr(self, "nepa_is_causal") and "nepa" in self.cfg.pretrained_type and "nepa" in pretrained_task:
            # Use dedicated NePA forward pass with causal attention
            h_in, h_out = self._forward_nepa_backbone(kwargs.get("nepa_input", x))
            # Compute NePA loss
            shift = getattr(self, "nepa_shift", True)
            nepa_loss = nepa_prediction_loss(h_in, h_out, shift=shift)
            terms["nepa_loss"] = nepa_loss
            terms["nepa_h_in"] = h_in
            terms["nepa_h_out"] = h_out

        # Patch tokens
        x_tokens = x[:, self.num_prefix_tokens :]
        terms["x_patch_tokens"] = x_tokens

        # Return the 1d features as the backbone output
        # not forward by the head
        # terms: [prefixed_tokens, ijepa_feat, lejepa_proj, ibot_feat, ibot_proj, x_patch_tokens]
        return x_tokens, edict(terms)

    def forward_intermedieates(
        self,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
        indices: Optional[Union[int, List[int]]] = None,
        return_prefix_tokens: bool = False,
        norm: bool = False,
        stop_early: bool = False,
        output_fmt: str = "NCHW",
        intermediates_only: bool = False,
        output_dict: bool = False,
        patch_coord: Optional[torch.Tensor] = None,
        patch_valid: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        masks: Optional[Tensor | List[Tensor]] = None,
    ):
        """Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            return_prefix_tokens: Return both prefix and spatial intermediate tokens
            norm: Apply norm layer to all intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
            output_dict: Return outputs as a dictionary with 'image_features' and 'image_intermediates' keys
            patch_coord: Optional patch coordinates [B, N, 2] for NaFlex mode
            patch_valid: Optional patch type indicators (1=patch, 0=padding) for NaFlex
            attn_mask: Optional attention mask for masked attention
        Returns:
            A tuple with (final_features, intermediates), a list of intermediate features, or a dictionary containing
            'image_features' and 'image_intermediates' (and optionally 'image_intermediates_prefix')
        """

        # Prepare JEPA masks
        if masks is not None:
            raise ValueError(f"Input masks are not supported for getting the intermidate features")
            masks = None

        return super().forward_intermediates(
            x,
            indices,
            return_prefix_tokens,
            norm,
            stop_early,
            output_fmt,
            intermediates_only,
            output_dict,
            patch_coord,
            patch_valid,
            attn_mask,
        )


class MAENaFlexViT(Transformer):
    def __init__(self): ...


class DINONaFlexVit(Transformer): ...


def __test_jepa_naflex():
    from src.stage1.self_supervised.ijepa.src.models.vision_transformer import (
        VisionTransformerPredictor,
        apply_masks,
        repeat_interleave_batch,
        vit_predictor,
    )
    from src.stage1.self_supervised.jepa_blockutils import MaskCollator

    x = torch.randn(2, 3, 224, 224)
    x = [xi for xi in x]
    collator = MaskCollator(
        patch_size=16,
        npred=4,
        nenc=1,
        min_keep=10,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
    )
    x, mask_enc, mask_pred = collator(x)

    x = x.to("cuda", torch.bfloat16)
    mask_enc = [m.to("cuda", torch.int32) for m in mask_enc]
    mask_pred = [m.to("cuda", torch.int32) for m in mask_pred]

    # model
    cfg = NaFlexVitCfg(
        patch_size=16,
        embed_dim=768,
        depth=8,
        num_heads=8,
        pos_embed="learned",
        rope_type="axial",
        in_chans=3,
        out_chans=768,
        unpatch_size=1,
        reg_tokens=4,
    )
    model = IJEPANaFlexViT(cfg).to("cuda", torch.bfloat16)

    # predictor
    predictor = vit_predictor(
        num_patches=(224 // 16) ** 2,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=4,
        num_heads=8,
    ).to("cuda", torch.bfloat16)

    # Forward
    with torch.autocast("cuda", torch.bfloat16):
        # Target
        with torch.no_grad():
            h = model._forward_pretrained_backbone(x)[0]
            h = torch.layer_norm(h, (h.size(-1),))
            B = len(h)
            h = apply_masks(h, mask_pred)
            h_tgt = repeat_interleave_batch(h, B, repeat=len(mask_enc))
            print(h_tgt.shape)

        # Context
        h_ctx = model._forward_pretrained_backbone(x, jepa_masks=mask_enc)[0]
        print(h_ctx.shape)
        h_pred = predictor(h_ctx, mask_enc, mask_pred)
        print(h_pred.shape)

        # Loss
        loss = torch.nn.functional.smooth_l1_loss(h_pred, h_tgt)
        print(f"loss: {loss}")


if __name__ == "__main__":
    """
    python -m src.stage1.cosmos.modules.naflex
    """
    __test_jepa_naflex()

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import einops
import torch
import torch.nn as nn
from einx import get_at
from loguru import logger
from timm.layers import apply_keep_indices_nlc, to_2tuple
from timm.layers.create_norm import get_norm_layer
from timm.layers.helpers import to_2tuple
from timm.layers.patch_embed import PatchEmbed
from timm.layers.pos_embed import resample_abs_pos_embed
from timm.models.naflexvit import (
    NaFlexEmbeds,
    NaFlexVit,
    checkpoint,
    create_attention_mask,
)
from torch import Tensor

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

    # is low-level feature skip the semantic encoder
    # straight through to CNN decoder
    latent_straight_through_skip: bool = False


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
        return out_hw

    def forward(self, x):
        # Output HW
        out_hw = self._get_output_shape(x)

        # Features
        x = self.forward_features(x)
        x = cast(torch.Tensor, x)
        x = x[:, self.cfg.reg_tokens :]

        # Head
        x = self._forward_after_backbone(x, out_hw)

        return x

    @function_config_to_basic_types
    @classmethod
    def create_model(cls, **overrides):
        cfg = dataclass_from_dict(NaFlexVitCfg, overrides)
        model = cls(cfg)
        return model


class IJEPANaFlexViT(Transformer):
    def _prepare_masks(self, masks=None):
        """Ensure the masks are list of tensors"""
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        return masks

    def _forward_embeds(
        self,
        x,
        patch_coord,
        patch_valid,
        attn_mask,
        masks: List[Tensor] | None,
    ):
        """Forward pass through patch / abs pos / rope pos embeds and patch dropout"""
        naflex_mode = patch_coord is not None

        # patch embed, abs pos embed, returns global grid size as calculated from 'standard' NCHW batches
        x, grid_size = self.embeds(
            x,
            patch_coord=patch_coord,
            patch_valid=patch_valid,
        )

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
                rope_embeds = apply_keep_indices_nlc(
                    x, rope_embeds, keep_indices, pos_embed_has_batch=naflex_mode
                )
                if not naflex_mode:
                    # B, N, dim -> B, 1, N, dim. Need head dim added for standard mode, already added in NaFlex.
                    rope_embeds = rope_embeds.unsqueeze(1)

        # Create attention mask from patch_valid after patch dropout applied
        if attn_mask is None:
            attn_mask = create_attention_mask(
                patch_valid, num_prefix_tokens=self.num_prefix_tokens, dtype=x.dtype
            )

        # JEPA masks
        if masks is not None:
            x = apply_masks(x, masks=masks)
            # Rope related, rope is applied in attention module
            if rope_embeds is not None:
                rope_masked = []
                for m in masks:
                    # m: [B, S_masked] indices
                    assert not self.rope_is_mixed, (
                        "mixed rope is not supported in JEPA training"
                    )
                    """
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
                    """

                    rope_masked += [
                        get_at(
                            "[S] ropeD, B S_masked -> B 1 S_masked ropeD",
                            rope_embeds,
                            m,
                        )
                    ]
                rope_embeds = torch.cat(rope_masked, dim=0)  # [B, 1, S_masked, ropeD]

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
        jepa_masks: Optional[List[Tensor]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """ """
        naflex_mode = patch_coord is not None

        # Pass through patch & abs position embedding module with patch coordinate/type support
        embeds = self._forward_embeds(
            patches,
            patch_coord=patch_coord,
            patch_valid=patch_valid,
            attn_mask=attn_mask,
            masks=jepa_masks,
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
                if (
                    self.training
                    and self.patch_drop is not None
                    and keep_indices is not None
                ):
                    # Apply patch dropout to rope_embed if needed (batch dim already present in naflex mode)
                    rope_embed = apply_keep_indices_nlc(
                        x,
                        rope_embed,
                        keep_indices,
                        pos_embed_has_batch=naflex_mode,
                    )
                if do_checkpointing:
                    x = checkpoint(blk, x, rope=rope_embed, attn_mask=attn_mask)
                else:
                    x = blk(x, rope=rope_embed, attn_mask=attn_mask)
        elif rope_embeds is not None:
            # Axial ROPE mode with shared embeddings
            for blk in self.blocks:
                if do_checkpointing:
                    x = checkpoint(blk, x, rope=rope_embeds, attn_mask=attn_mask)
                else:
                    x = blk(x, rope=rope_embeds, attn_mask=attn_mask)
        else:
            for blk in self.blocks:
                if do_checkpointing:
                    x = checkpoint(blk, x, attn_mask=attn_mask)
                else:
                    x = blk(x, attn_mask=attn_mask)

        x = self.norm(x)

        if naflex_mode:
            return {
                "patches": x,
                "patch_valid": embeds.get("patch_valid", None),
            }

        return x

    def forward(
        self,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
        patch_coord: Optional[torch.Tensor] = None,
        patch_valid: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        jepa_masks: Optional[Tensor | List[Tensor]] = None,  # IJEPA masks
    ):
        """Forward with JEPA masks support"""
        if jepa_masks is not None:
            jepa_masks = self._prepare_masks(masks=jepa_masks)

        input_is_dict = isinstance(x, Dict)
        naflex_mode = input_is_dict or patch_coord is not None
        if naflex_mode:
            assert jepa_masks is None, "JEPA does not support naflex mode."
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
            )

            # Pass patches & patch_valid to forward_head for masked pooling
            x = self.forward_head(**features)
        else:
            x = self.forward_features(x, jepa_masks=jepa_masks)
            x = self.forward_head(x)
        return x


class MAENaFlexViT(Transformer):
    def __init__(self): ...


def mode_support_jepa(model):
    import inspect

    sig = inspect.signature(model.forward)
    params = sig.parameters

    return "jepa_masks" in params


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
            h = model(x)
            h = torch.layer_norm(h, (h.size(-1),))
            B = len(h)
            h = apply_masks(h, mask_pred)
            h_tgt = repeat_interleave_batch(h, B, repeat=len(mask_enc))
            print(h_tgt.shape)

        # Context
        h_ctx = model(x, jepa_masks=mask_enc)
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

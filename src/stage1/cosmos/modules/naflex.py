import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import einops
import torch
import torch.nn as nn
from easydict import EasyDict as edict
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
    feature_take_indices,
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
    # enable_jepa: bool = False  # enable jepa training
    # enable_lejepa: bool = False  # enable lejepa training
    # 'ijepa', 'lejepa', None for no pretrained task
    pretrained_type: Any = None  # Union[str, list[str]]


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

    def init_weights(self, mode: str = "jax"):
        super().init_weights(mode=mode)

        def rescale(p, layer_id):
            p.div_(math.sqrt(2.0 * layer_id))

        # Rescale the depth
        rescale_layer = True
        if rescale_layer:
            for layer_id, blk in enumerate(self.blocks, 1):
                rescale(blk.attn.proj.weight.data, layer_id)
                rescale(blk.mlp.fc2.weight.data, layer_id)
            logger.info("Rescale the layers initialization for more stable training")


class IJEPANaFlexViT(Transformer):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Build the LeJEPA head
        if "lejepa" in cfg.pretrained_type:
            self._build_jepa_head(cfg)

    def _build_jepa_head(self, cfg):
        from src.stage1.self_supervised.lejepa_aug import create_lejepa_projector

        self.lejepa_projector = create_lejepa_projector(
            cfg.embed_dim,
            cfg.embed_dim,
            mean_out_hw=False,
            # mean_out_hw=not cfg.class_token,  # if use class, not mean out the spatial tokens
        )
        logger.info("[IJEPA Naflex Transformer]: Build LeJEPA head")

    def _prepare_masks(self, masks=None):
        """Ensure the masks are list of tensors"""
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        return masks

    def _jepa_apply_masks(self, x, masks):
        all_x = []
        for m in masks:
            mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
            all_x += [torch.gather(x, dim=1, index=mask_keep)]
        return torch.cat(all_x, dim=0)

    def _forward_embeds(
        self,
        x,
        patch_coord,
        patch_valid,
        attn_mask,
        masks: List[Tensor] | None = None,
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

        ########## Apply JEPA masks
        if masks is not None:
            x = self._jepa_apply_masks(x, masks=masks)
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
        output_type: str | None = None,
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
            x = self.forward_features(x, jepa_masks=jepa_masks)
            x = cast(torch.Tensor, x)
            x = x[:, self.cfg.reg_tokens :]

            # Head
            x = self._forward_after_backbone(x, out_hw)

        return x

    def _forward_after_backbone(
        self, x, hw: list | None
    ) -> tuple[Tensor, Optional[Dict[str, Tensor]]] | Tensor:
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

    def _forward_pretrained_backbone(
        self,
        x: torch.Tensor,
        output_type: str = "1d",  # fixed it.
        jepa_masks: Optional[Tensor | List[Tensor]] = None,  # IJEPA masks
    ):
        others = None

        if output_type in (None, "2d"):
            out_hw = self._get_output_shape(x)
        else:
            out_hw = None  # keep the output to be 1D tensor

        # Features
        x = self.forward_features(x, jepa_masks=jepa_masks)
        x = cast(torch.Tensor, x)
        x = x[:, self.cfg.reg_tokens :]

        ######### IJepa features ########
        if self.cfg.pretrained_type == "ijepa":
            # x is the backbone's out
            others = edict({"ijepa_feat": x})

        ######### Lejepa projector #########
        if hasattr(self, "lejepa_projector") and self.cfg.pretrained_type == "lejepa":
            x = self._pool(x)
            lejepa_proj = self.lejepa_projector(x)  # x is the backbone's out
            others = edict({"lejepa_proj": lejepa_proj})

        # Return the 1d features as the backbone output
        # not forward by the head
        return x, others

    def forward_intermediates(
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
        jepa_masks: Optional[Tensor | List[Tensor]] = None,  # IJEPA masks
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
        if jepa_masks is not None:
            # jepa_masks = self._prepare_masks(masks=jepa_masks)
            raise ValueError(
                f"Input masks are not supported for getting the intermidate features"
            )
            jepa_masks = None

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

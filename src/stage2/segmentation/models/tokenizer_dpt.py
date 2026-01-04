"""
Hybrid Tokenizer + DPT Segmentation Model.

This module implements a segmentation network that uses:
1. A Hybrid Tokenizer backbone (CNN encoder + Transformer encoder)
2. A DPT head for fusing multi-scale features

The model extracts 4 feature scales:
- 2 from CNN encoder (low-level, high resolution)
- 2 from Transformer encoder (semantic, lower resolution)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from loguru import logger
from omegaconf import OmegaConf
from timm.layers import get_act_layer, get_norm_layer
from timm.models._manipulate import named_apply
from torch import Tensor
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.module import _IncompatibleKeys

from src.stage1.cosmos.cosmos_hybrid import CosmosHybridTokenizer
from src.utilities.config_utils import function_config_to_basic_types

from ...layers.hybrid_dpt import DPTSegmentationHead, HybridDPTHead


def _create_default_cfg():
    """Create default configuration for TokenizerHybridDPT."""
    cfg_dict = dict(
        tokenizer=dict(
            cnn_cfg=dict(
                model=dict(
                    resolution=1024,
                    in_channels=512,
                    out_channels=512,
                    z_channels=768,
                    latent_channels=32,
                    channels=128,
                    channels_mult=[2, 4, 4],
                    num_res_blocks=2,
                    attn_resolutions=[],
                    dropout=0.0,
                    spatial_compression=8,
                    patch_size=1,
                    block_name="res_block",
                    norm_type="rmsnorm2d",
                    act_type="silu",
                    norm_groups=32,
                    adaptive_mode="interp",
                    downsample_kwargs=dict(padconv_use_manually_pad=False),
                    upsample_kwargs=dict(interp_type="nearest_interp"),
                ),
                quantizer_type=None,
                vf_on_z_or_module="z",
                use_repa_loss=False,
                dino_feature_dim=1024,
            ),
            trans_enc_cfg=dict(
                embed_dim=1152,
                depth=24,
                num_heads=16,
                mlp_ratio=4.0,
                qkv_bias=True,
                patch_size=2,
                norm_layer="flarmsnorm",
                pos_embed="learned",
                rope_type="axial",
                pos_embed_grid_size=[32, 32],
                img_size=32,
                in_chans=768,
                out_chans=768,
                unpatch_size=2,
                reg_tokens=4,
                attn_type="gated",
            ),
            trans_dec_cfg=None,
            distill_cfg=dict(
                dino_feature_dim=1024,
                semantic_feature_dim=1024,
                cache_layers=dict(
                    low_level=[0, 1, 2, -1],
                    semantic=[5, 11, 17, 23],  # 4 semantic layers
                ),
            ),
            hybrid_tokenizer_cfg=dict(
                latent_bottleneck_type="before_semantic",
                latent_straight_through_skip=True,
            ),
        ),
        dpt_head=dict(
            feature_dim=256,
            norm_layer="layernorm2d",
            head_type="hybrid",  # "hybrid" or "simple"
            n_blocks=1,
        ),
        # CNN feature indices to use (from cache_layers.low_level)
        cnn_feature_indices=[1, 2],  # 2nd and 3rd CNN features
        # Transformer feature indices to use (from cache_layers.semantic)
        vit_feature_indices=[1, 3],  # 2nd and 4th semantic features
        tokenizer_pretrained_path=None,
        input_channels=155,
        num_classes=24,
        freeze_backbone=True,
        _debug=False,
    )
    return OmegaConf.create(cfg_dict)


class HybridFeatureExtractor(nn.Module):
    """
    Wrapper around CosmosHybridTokenizer to extract multi-scale features.

    Extracts features from both CNN encoder (low-level) and Transformer
    encoder (semantic) at specified layer indices.
    """

    def __init__(
        self,
        backbone: CosmosHybridTokenizer,
        cnn_feature_indices: list[int],
        vit_feature_indices: list[int],
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.cnn_feature_indices = cnn_feature_indices
        self.vit_feature_indices = vit_feature_indices
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            self.backbone.requires_grad_(False)
            logger.info("[HybridFeatureExtractor] Backbone frozen.")

        # Get channel dimensions from backbone config
        self._setup_channel_info()

    def _setup_channel_info(self):
        """Setup channel dimension information from backbone."""
        # CNN encoder channel dimensions based on channels_mult
        base_ch = self.backbone.cnn_cfg.model.channels
        ch_mult = self.backbone.cnn_cfg.model.channels_mult

        # CNN features: after each downsample block
        # Index 0: base_ch * ch_mult[0], Index 1: base_ch * ch_mult[1], etc.
        self.cnn_channels = [base_ch * m for m in ch_mult]
        # Add mid-block channel (same as last downsample)
        self.cnn_channels.append(self.cnn_channels[-1])

        # Transformer semantic features: all have same embed_dim
        self.vit_channels = [self.backbone.trans_enc_cfg.embed_dim] * 4

        # Get selected channels
        self.selected_cnn_channels = [self.cnn_channels[i] for i in self.cnn_feature_indices]
        self.selected_vit_channels = [self.vit_channels[i] for i in self.vit_feature_indices]

        self.out_channels = self.selected_cnn_channels + self.selected_vit_channels

        logger.info(f"[HybridFeatureExtractor] CNN channels (all): {self.cnn_channels}")
        logger.info(f"[HybridFeatureExtractor] Selected CNN channels: {self.selected_cnn_channels}")
        logger.info(f"[HybridFeatureExtractor] Selected ViT channels: {self.selected_vit_channels}")
        logger.info(f"[HybridFeatureExtractor] Total output channels: {self.out_channels}")

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Extract multi-scale features.

        Args:
            x: Input image tensor (B, C, H, W)

        Returns:
            List of feature tensors [cnn_feat1, cnn_feat2, vit_feat1, vit_feat2]
        """
        grad_ctx = torch.no_grad if self.freeze_backbone else torch.enable_grad

        if self.freeze_backbone:
            self.backbone.eval()

        with torch.autocast("cuda", torch.bfloat16):
            with grad_ctx():
                enc_out = self.backbone.encode(x, get_intermediate_features=True)
                all_cnn_feats = enc_out.low_lvl_z
                all_vit_feats = enc_out.sem_z

        # Clear backbone cache to free memory
        if hasattr(self.backbone, "z"):
            self.backbone.z = None  # type: ignore[assignment]
        if hasattr(self.backbone, "sem_z"):
            self.backbone.sem_z = None  # type: ignore[assignment]

        # Select specified features
        selected_cnn = [
            F.interpolate(all_cnn_feats[i], size=x.shape[-1] // 16, mode="bilinear", align_corners=False)
            for i in self.cnn_feature_indices
        ]
        selected_vit = []
        for i in self.vit_feature_indices:
            feat = all_vit_feats[i]
            # Reshape ViT features from (B, L, C) to (B, C, H, W) if needed
            if feat.ndim == 3:
                b, l, c = feat.shape
                h = w = int(l**0.5)
                feat = rearrange(feat, "b (h w) c -> b c h w", h=h, w=w)
            selected_vit.append(feat)

        # Combine: CNN features first (higher res), then ViT features
        features = selected_cnn + selected_vit

        return features


class TokenizerHybridDPT(nn.Module):
    """
    Segmentation network with Hybrid Tokenizer backbone and DPT head.

    Architecture:
    1. HybridFeatureExtractor: Extracts multi-scale CNN + ViT features
    2. HybridDPTHead/DPTSegmentationHead: Fuses features for dense prediction

    Args:
        cfg: Configuration dict/OmegaConf with model settings
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_cfg = cfg.tokenizer
        self.dpt_cfg = cfg.dpt_head

        self._debug = cfg._debug

        # Create backbone
        self.backbone = self._create_backbone()

        # Create feature extractor
        self.feature_extractor = HybridFeatureExtractor(
            backbone=self.backbone,
            cnn_feature_indices=cfg.cnn_feature_indices,
            vit_feature_indices=cfg.vit_feature_indices,
            freeze_backbone=cfg.freeze_backbone,
        )

        # Create DPT head
        in_channels = self.feature_extractor.out_channels
        if self.dpt_cfg.head_type == "hybrid":
            self.head = HybridDPTHead(
                num_classes=cfg.num_classes,
                in_channels=in_channels,
                feature_dim=self.dpt_cfg.feature_dim,
                norm_layer=self.dpt_cfg.norm_layer,
                n_blocks=self.dpt_cfg.get("n_blocks", 1),
            )
        else:
            self.head = DPTSegmentationHead(
                num_classes=cfg.num_classes,
                in_channels=in_channels,
                feature_dim=self.dpt_cfg.feature_dim,
                norm_layer=self.dpt_cfg.norm_layer,
            )

        # Initialize weights
        self._init_weights()

        logger.info(f"[TokenizerHybridDPT] Created with {len(in_channels)} feature scales")

    def _create_backbone(self) -> CosmosHybridTokenizer:
        """Create tokenizer backbone."""
        t_cfg = self.tok_cfg

        backbone = CosmosHybridTokenizer.create_model(
            cnn_cfg=t_cfg.cnn_cfg,
            trans_enc_cfg=t_cfg.trans_enc_cfg,
            trans_dec_cfg=t_cfg.trans_dec_cfg,
            distillation_cfg=t_cfg.distill_cfg,
            hybrid_tokenizer_cfg=t_cfg.get("hybrid_tokenizer_cfg", None),
        )

        if self.cfg.tokenizer_pretrained_path is not None:
            backbone.load_pretrained(self.cfg.tokenizer_pretrained_path)
            logger.info(f"[TokenizerHybridDPT] Loaded backbone from: {self.cfg.tokenizer_pretrained_path}")
        elif self._debug:
            logger.warning("[TokenizerHybridDPT] Debug mode: using random backbone weights")
        else:
            raise ValueError("tokenizer_pretrained_path must be specified")

        return backbone

    def _init_weights(self):
        """Initialize weights for non-backbone modules."""

        def _apply(module, name: str):
            if "backbone" not in name:
                if hasattr(module, "init_weights"):
                    module.init_weights()
                elif isinstance(module, _ConvNd):
                    if "output_conv" in name or "seg_head" in name:
                        nn.init.xavier_uniform_(module.weight, gain=0.01)
                    else:
                        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                    nn.init.ones_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        named_apply(_apply, self)
        logger.info("[TokenizerHybridDPT] Initialized weights (except backbone)")

    def forward(self, x: Float[Tensor, "b c h w"]) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input image tensor (B, C, H, W)

        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        input_size = x.shape[-2:]

        # Extract multi-scale features
        features = self.feature_extractor(x)

        # Fuse and predict
        logits = self.head(features)

        # Upsample to input resolution
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)

        return logits

    def _filter_backbone_params(self, k: str) -> bool:
        return "backbone" in k

    def parameters(self, *args, **kwargs):
        for name, param in self.named_parameters(*args, **kwargs):
            if "backbone" in name:
                param.requires_grad = False
            yield param

    def named_parameters(self, *args, **kwargs):
        for name, param in super().named_parameters(*args, **kwargs):
            if "backbone" in name:
                param.requires_grad = False
            yield name, param

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        # Remove backbone parameters
        backbone_keys = [k for k in state_dict.keys() if self._filter_backbone_params(k)]
        for k in backbone_keys:
            del state_dict[k]
        logger.info(f"[TokenizerHybridDPT] State dict: {len(state_dict)} params (backbone excluded)")
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = False, *args, **kwargs):
        # Filter out backbone parameters
        backbone_keys = [k for k in state_dict.keys() if self._filter_backbone_params(k)]
        for k in backbone_keys:
            del state_dict[k]
        missing_ks, unexpected_ks = super().load_state_dict(state_dict, strict=strict)

        missing_ks = [k for k in missing_ks if not self._filter_backbone_params(k)]
        unexpected_ks = [k for k in unexpected_ks if not self._filter_backbone_params(k)]

        if missing_ks:
            logger.warning(f"Missing keys: {missing_ks}")
        if unexpected_ks:
            logger.warning(f"Unexpected keys: {unexpected_ks}")

        return _IncompatibleKeys(missing_ks, unexpected_ks)

    @classmethod
    @function_config_to_basic_types
    def create_model(cls, cfg=None, **overrides):
        """Create model from configuration."""
        if cfg is None:
            cfg = _create_default_cfg()

        if overrides:
            cfg = OmegaConf.merge(cfg, overrides)

        return cls(cfg)


def __test_model():
    """Test the TokenizerHybridDPT model."""
    from src.data.hyperspectral_loader import get_fast_test_hyperspectral_data
    from fvcore.nn import parameter_count_table

    dl = get_fast_test_hyperspectral_data("fmow_RGB")
    sample = next(iter(dl))
    x = sample["img"].cuda()
    x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)

    cfg = _create_default_cfg()
    cfg._debug = True
    cfg.tokenizer_pretrained_path = (
        "runs/stage1_cosmos_hybrid/2025-12-21_23-52-12_hybrid_cosmos_f16c64_ijepa_pretrained_sem_no_lejepa"
        "/ema/tokenizer/model.safetensors"
    )

    model = TokenizerHybridDPT(cfg).cuda()
    model.train()  # switch to train mode for backward
    print(parameter_count_table(model))

    with torch.autocast("cuda", torch.bfloat16):
        y = model(x)
        loss = y.sum()
        loss.backward()

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    # Check gradients
    backbone_has_grad = False
    head_has_grad = False
    for name, param in model.named_parameters():
        if "backbone" in name and param.grad is not None:
            backbone_has_grad = True
            print(f"Error: Backbone param {name} has gradient!")
        if "head" in name and param.grad is not None:
            head_has_grad = True

    if not backbone_has_grad and head_has_grad:
        print("Gradient check passed: Backbone is frozen, Head has gradients.")
    else:
        print(f"Gradient check failed: {backbone_has_grad=}, {head_has_grad=}")

    assert y.shape == (x.shape[0], cfg.num_classes, x.shape[2], x.shape[3])
    print("Test passed!")


if __name__ == "__main__":
    """
    MODEL_COMPILED=0 python -m src.stage2.segmentation.models.tokenizer_dpt
    """
    with logger.catch():
        __test_model()

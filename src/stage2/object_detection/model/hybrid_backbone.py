from __future__ import annotations

from typing import Any, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from timm.layers import get_act_layer, get_norm_layer

from src.stage1.cosmos.cosmos_hybrid import CosmosHybridTokenizer
from src.stage2.segmentation.models.adapter import DINOv3EncoderAdapter
from src.stage2.segmentation.models.tokenizer_backbone_adapted import (
    HybridTokenizerEncoderAdapter,
    TOKENIZER_INTERACTION_INDEXES,
)
from src.utilities.config_utils import function_config_to_basic_types

logger = logger.bind(_name_="hybrid_det_backbone")
logger.disable("src.stage1")


def _create_default_cfg() -> DictConfig:
    yaml_string = """
    tokenizer:
        cnn_cfg:
            model:
                resolution: 1024
                in_channels: 512
                out_channels: 512
                z_channels: 768
                latent_channels: 32
                channels: 128
                channels_mult: [2, 4, 4]
                num_res_blocks: 2
                attn_resolutions: []
                dropout: 0.0
                spatial_compression: 8
                patch_size: 1
                block_name: res_block
                norm_type: rmsnorm2d
                act_type: silu
                norm_groups: 32
                adaptive_mode: interp
                downsample_kwargs:
                    padconv_use_manually_pad: false
                upsample_kwargs:
                    interp_type: nearest_interp
            quantizer_type: null
            vf_on_z_or_module: z
            use_repa_loss: false
            dino_feature_dim: 1024
        trans_enc_cfg:
            embed_dim: 1152
            depth: 24
            num_heads: 16
            mlp_ratio: 4.0
            qkv_bias: true
            patch_size: 2
            norm_layer: flarmsnorm
            pos_embed: learned
            rope_type: axial
            pos_embed_grid_size: [32, 32]
            img_size: 32
            in_chans: 768
            out_chans: 768
            unpatch_size: 2
            reg_tokens: 4
            attn_type: gated
        trans_dec_cfg: null
        distill_cfg:
            dino_feature_dim: 1024
            semantic_feature_dim: 1024
            cache_layers:
                low_level: [0, 1, 2, -1]
                semantic: [5, 11, 14, 19]
        hybrid_tokenizer_cfg:
            latent_bottleneck_type: before_semantic
            latent_straight_through_skip: true
    tokenizer_feature:
        pretrained_path: null
        model_name: hybrid_tokenizer_b16
        pretrained_size: 512
        in_channels: 155
        interp_ratio: null
        interaction_indexes: [1, 2, 4, 5]
        layer_in_channels: [512, 512, 1152, 1152]
        features_per_stage: [256, 384, 512, 512]
        conv_inplane: 64
        drop_path_rate: 0.3
        with_cffn: true
        cffn_ratio: 0.25
        deform_num_heads: 16
        deform_ratio: 0.5
        add_vit_feature: true
        use_extra_extractor: true
        with_cp: true
        select_in_all_layers: true
        extractor_type: deform_attention
        extractor_kwargs: {}
    adapter:
        latent_width: 32
        n_conv_per_stage: 2
        depth_per_stage: 2
        norm: layernorm2d
        act: gelu
        drop: 0.0
        act_first: false
        conv_bias: false
        block_types: [mbconv, mbconv, mbconv, mbconv]
    tokenizer_pretrained_path: null
    input_channels: ${tokenizer_feature.in_channels}
    feature_keys: [res2, res3, res4, res5]
    encoder_feature_strides: [4, 8, 16, 32]
    feature_strides: [2, 4, 8, 16]
    fpn_out_features: [p2, p3, p4, p5]
    fpn_out_channels: 256
    num_classes: 2
    pixel_mean: 0.0
    pixel_std: 1.0
    freeze_backbone: true
    device: cpu
    _debug: false
    """
    cfg = OmegaConf.create(yaml_string)
    if not isinstance(cfg, DictConfig):
        raise TypeError("Default hybrid backbone config must be a DictConfig")
    return cfg


def _ensure_cfg(cfg: DictConfig | dict[str, Any]) -> DictConfig:
    if isinstance(cfg, DictConfig):
        return cfg
    return OmegaConf.create(cfg)


def _resolve_interaction_indexes(model_name: str, interaction_indexes: list[int] | None) -> list[int]:
    if interaction_indexes is not None:
        return list(interaction_indexes)
    if model_name not in TOKENIZER_INTERACTION_INDEXES:
        raise KeyError(f"Unknown tokenizer model_name: {model_name}")
    return list(TOKENIZER_INTERACTION_INDEXES[model_name])


@function_config_to_basic_types
def build_hybrid_tokenizer_encoder(cfg: DictConfig) -> DINOv3EncoderAdapter:
    cfg = _ensure_cfg(cfg)
    tok_cfg = cfg.tokenizer
    f_cfg = cfg.tokenizer_feature
    a_cfg = cfg.adapter

    if int(f_cfg.in_channels) != int(cfg.input_channels):
        raise ValueError(f"input_channels mismatch: {cfg.input_channels=} vs {f_cfg.in_channels=}")

    interaction_indexes = _resolve_interaction_indexes(
        model_name=f_cfg.model_name,
        interaction_indexes=getattr(f_cfg, "interaction_indexes", None),
    )
    logger.info(f"Creating tokenizer encoder: {f_cfg.model_name}")

    tok_backbone = CosmosHybridTokenizer.create_model(
        cnn_cfg=tok_cfg.cnn_cfg,
        trans_enc_cfg=tok_cfg.trans_enc_cfg,
        trans_dec_cfg=tok_cfg.trans_dec_cfg,
        distillation_cfg=tok_cfg.distill_cfg,
        hybrid_tokenizer_cfg=getattr(tok_cfg, "hybrid_tokenizer_cfg", None),
    )

    if cfg.tokenizer_pretrained_path is not None:
        tok_backbone.load_pretrained(cfg.tokenizer_pretrained_path)
        logger.info(f"Loaded tokenizer backbone from: {cfg.tokenizer_pretrained_path}")
    elif cfg._debug:
        logger.warning("Using debug mode, tokenizer backbone will use random weights.")
    else:
        raise ValueError("tokenizer_pretrained_path must be specified unless _debug is true")

    dinov3_adapter = HybridTokenizerEncoderAdapter(
        backbone=tok_backbone,
        in_channels=int(f_cfg.in_channels),
        conv_inplane=int(f_cfg.conv_inplane),
        interaction_indexes=interaction_indexes,
        deform_num_heads=int(f_cfg.deform_num_heads),
        drop_path_rate=float(f_cfg.drop_path_rate),
        with_cffn=bool(f_cfg.with_cffn),
        cffn_ratio=float(f_cfg.cffn_ratio),
        deform_ratio=float(f_cfg.deform_ratio),
        add_vit_feature=bool(f_cfg.add_vit_feature),
        use_extra_extractor=bool(f_cfg.use_extra_extractor),
        with_cp=bool(f_cfg.with_cp),
        pretrain_size=int(f_cfg.pretrained_size),
        n_points=4,
        init_values=0.0,
        use_bn=False,
        select_in_all_layers=bool(getattr(f_cfg, "select_in_all_layers", False)),
        interp_ratio=getattr(f_cfg, "interp_ratio", None),
        layer_in_channels=getattr(f_cfg, "layer_in_channels", None),
        freeze_backbone=bool(cfg.freeze_backbone),
    )

    encoder_adapter = DINOv3EncoderAdapter(
        dinov3_adapter=dinov3_adapter,
        target_channels=list(f_cfg.features_per_stage),
        conv_op=nn.Conv2d,
        norm_op=get_norm_layer(str(a_cfg.norm)),
        nonlin=get_act_layer(str(a_cfg.act)),
        dropout_op=float(a_cfg.drop),
        conv_bias=bool(a_cfg.conv_bias),
    )

    logger.info("Created hybrid tokenizer encoder adapter.")
    return encoder_adapter


class FeatureEncoder(Protocol):
    output_channels: list[int]

    def __call__(self, x: torch.Tensor) -> list[torch.Tensor] | tuple[list[torch.Tensor], Any]: ...


class HybridTokenizerFeatureExtractor(nn.Module):
    def __init__(
        self,
        encoder: FeatureEncoder,
        feature_keys: list[str],
        source_strides: list[int] | None = None,
        target_strides: list[int] | None = None,
    ) -> None:
        super().__init__()
        if len(feature_keys) != len(encoder.output_channels):
            raise ValueError(
                f"feature_keys length must match encoder output channels: {len(feature_keys)} != {len(encoder.output_channels)}"
            )
        if source_strides is not None and len(source_strides) != len(feature_keys):
            raise ValueError(
                f"source_strides length must match feature_keys: {len(source_strides)} != {len(feature_keys)}"
            )
        if target_strides is not None and len(target_strides) != len(feature_keys):
            raise ValueError(
                f"target_strides length must match feature_keys: {len(target_strides)} != {len(feature_keys)}"
            )
        self.encoder = encoder
        self.feature_keys = list(feature_keys)
        self.feature_channels = list(encoder.output_channels)
        self._source_strides = list(source_strides) if source_strides is not None else None
        self._target_strides = list(target_strides) if target_strides is not None else None

    def _resize_feature(self, feat: torch.Tensor, src_stride: int, dst_stride: int) -> torch.Tensor:
        if src_stride == dst_stride:
            return feat
        if src_stride % dst_stride != 0 and dst_stride % src_stride != 0:
            raise ValueError(f"Stride ratio must be integer: {src_stride=} vs {dst_stride=}")
        scale = float(src_stride) / float(dst_stride)
        height = int(round(feat.shape[-2] * scale))
        width = int(round(feat.shape[-1] * scale))
        return F.interpolate(feat, size=(height, width), mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.encoder(x)
        if isinstance(feats, tuple):
            feats = feats[0]
        if len(feats) != len(self.feature_keys):
            raise ValueError(f"encoder returned {len(feats)} features, expected {len(self.feature_keys)}")
        if self._source_strides is None or self._target_strides is None:
            return dict(zip(self.feature_keys, feats, strict=True))
        resized = [
            self._resize_feature(feat, int(src_stride), int(dst_stride))
            for feat, src_stride, dst_stride in zip(feats, self._source_strides, self._target_strides, strict=True)
        ]
        return dict(zip(self.feature_keys, resized, strict=True))

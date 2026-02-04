from __future__ import annotations

from typing import Any

import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.utilities.config_utils import function_config_to_basic_types

from .hybrid_fcos import (
    _compute_fpn_strides,
    _create_default_backbone_cfg,
    _ensure_cfg,
    _init_mmdet_default_scope,
    _load_base_mmdet_cfg,
    _patch_common_model_cfg,
    require_mmdet,
)

logger = logger.bind(_name_="hybrid_det_mmdet_rcnn")


def _create_default_cfg() -> DictConfig:
    cfg = _create_default_backbone_cfg()
    cfg.encoder_feature_strides = [4, 8, 16, 32]
    cfg.feature_strides = [2, 4, 8, 16]

    cfg.detector = OmegaConf.create()
    cfg.detector.base_config_path = None
    cfg.detector.fpn_num_outs = 5
    return cfg


def _patch_rcnn_heads_cfg(base_cfg: Any, hybrid_cfg: DictConfig) -> None:
    model_cfg: dict[str, Any] = base_cfg.model
    num_outs = int(getattr(hybrid_cfg.detector, "fpn_num_outs", 5))
    fpn_strides = _compute_fpn_strides([int(s) for s in hybrid_cfg.feature_strides], num_outs)

    # RPN: align anchor/prior strides with feature strides.
    rpn_head = model_cfg.get("rpn_head")
    if isinstance(rpn_head, dict):
        rpn_head["in_channels"] = int(hybrid_cfg.fpn_out_channels)
        if isinstance(rpn_head.get("anchor_generator"), dict):
            rpn_head["anchor_generator"]["strides"] = fpn_strides
        if isinstance(rpn_head.get("prior_generator"), dict):
            rpn_head["prior_generator"]["strides"] = fpn_strides

    # RoI: align featmap strides and num_classes.
    roi_head = model_cfg.get("roi_head")
    if not isinstance(roi_head, dict):
        raise KeyError("base_cfg.model.roi_head is missing or not a dict (expected a Faster R-CNN style config)")

    bbox_roi_extractor = roi_head.get("bbox_roi_extractor")
    if isinstance(bbox_roi_extractor, dict):
        bbox_roi_extractor["featmap_strides"] = fpn_strides[:4]

    mask_roi_extractor = roi_head.get("mask_roi_extractor")
    if isinstance(mask_roi_extractor, dict):
        mask_roi_extractor["featmap_strides"] = fpn_strides[:4]

    bbox_head = roi_head.get("bbox_head")
    _set_num_classes(bbox_head, int(hybrid_cfg.num_classes))

    # Neck: ensure it outputs enough levels for RPN.
    neck = model_cfg.get("neck")
    if isinstance(neck, dict):
        neck["num_outs"] = num_outs


def _set_num_classes(head_cfg: Any, num_classes: int) -> None:
    if isinstance(head_cfg, dict):
        if "num_classes" in head_cfg:
            head_cfg["num_classes"] = num_classes
        # Cascade-style heads can nest bbox_head fields.
        if "bbox_head" in head_cfg:
            _set_num_classes(head_cfg["bbox_head"], num_classes)
        return
    if isinstance(head_cfg, list):
        for item in head_cfg:
            _set_num_classes(item, num_classes)
        return


def build_hybrid_rcnn_model(cfg: DictConfig) -> nn.Module:
    require_mmdet()
    cfg = _ensure_cfg(cfg)
    _init_mmdet_default_scope()

    base_cfg_source = getattr(cfg.detector, "base_cfg", None)
    if base_cfg_source not in (None, ""):
        base_cfg = _load_base_mmdet_cfg(base_cfg_source)
    else:
        base_cfg_path = getattr(cfg.detector, "base_config_path", None)
        if base_cfg_path in (None, ""):
            raise ValueError(
                "You must set either cfg.detector.base_cfg (a dict/DictConfig) or cfg.detector.base_config_path "
                "(an MMDetection config file path, e.g. /path/to/mmdetection/configs/faster_rcnn/*.py)."
            )
        base_cfg = _load_base_mmdet_cfg(base_cfg_path)
    base_cfg.setdefault("default_scope", "mmdet")

    _patch_common_model_cfg(base_cfg, cfg)
    _patch_rcnn_heads_cfg(base_cfg, cfg)

    from mmdet.registry import MODELS  # type: ignore[import-not-found]

    model = MODELS.build(base_cfg.model)
    if hasattr(model, "init_weights"):
        model.init_weights()
    return model


class HybridRCNN(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = build_hybrid_rcnn_model(cfg)

    def forward(self, *args: Any, **kwargs: Any):
        return self.model(*args, **kwargs)

    @classmethod
    @function_config_to_basic_types
    def create_model(cls, cfg: DictConfig = _create_default_cfg(), **overrides: Any) -> "HybridRCNN":
        if overrides:
            cfg.merge_with(overrides)
        logger.info("Creating HybridRCNN (MMDetection) model.")
        return cls(cfg)

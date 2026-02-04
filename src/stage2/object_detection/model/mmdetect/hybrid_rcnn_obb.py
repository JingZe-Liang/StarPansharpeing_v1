from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.utilities.config_utils import function_config_to_basic_types

from .hybrid_fcos_obb import require_mmrotate
from .hybrid_fcos import (
    _compute_fpn_strides,
    _create_default_backbone_cfg,
    _ensure_cfg,
    _init_mmdet_default_scope,
    _load_base_mmdet_cfg,
    _patch_common_model_cfg,
    require_mmdet,
)

logger = logger.bind(_name_="hybrid_det_mmdet_rcnn_obb")


def _create_default_cfg() -> DictConfig:
    cfg = _create_default_backbone_cfg()
    cfg.encoder_feature_strides = [4, 8, 16, 32]
    cfg.feature_strides = [2, 4, 8, 16]

    cfg.obb = OmegaConf.create()
    cfg.obb.angle_version = "le90"

    cfg.detector = OmegaConf.create()
    cfg.detector.base_cfg = None
    cfg.detector.base_config_path = None
    cfg.detector.fpn_num_outs = 5
    return cfg


def _patch_rcnn_obb_heads_cfg(base_cfg: Any, hybrid_cfg: DictConfig) -> None:
    model_cfg: dict[str, Any] = base_cfg.model
    num_outs = int(getattr(hybrid_cfg.detector, "fpn_num_outs", 5))
    fpn_strides = _compute_fpn_strides([int(s) for s in hybrid_cfg.feature_strides], num_outs)

    rpn_head = model_cfg.get("rpn_head")
    if isinstance(rpn_head, dict):
        rpn_head["in_channels"] = int(hybrid_cfg.fpn_out_channels)
        if isinstance(rpn_head.get("anchor_generator"), dict):
            rpn_head["anchor_generator"]["strides"] = fpn_strides
        if isinstance(rpn_head.get("prior_generator"), dict):
            rpn_head["prior_generator"]["strides"] = fpn_strides

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

    angle_version = getattr(hybrid_cfg, "obb", {}).get("angle_version", None)
    if angle_version is not None:
        _set_angle_version(bbox_head, angle_version)

    neck = model_cfg.get("neck")
    if isinstance(neck, dict):
        neck["num_outs"] = num_outs


def _set_num_classes(head_cfg: Any, num_classes: int) -> None:
    if isinstance(head_cfg, dict):
        if "num_classes" in head_cfg:
            head_cfg["num_classes"] = num_classes
        if "bbox_head" in head_cfg:
            _set_num_classes(head_cfg["bbox_head"], num_classes)
        return
    if isinstance(head_cfg, list):
        for item in head_cfg:
            _set_num_classes(item, num_classes)


def _set_angle_version(head_cfg: Any, angle_version: str) -> None:
    if isinstance(head_cfg, dict):
        if "angle_version" in head_cfg:
            head_cfg["angle_version"] = angle_version
        if "bbox_head" in head_cfg:
            _set_angle_version(head_cfg["bbox_head"], angle_version)
        return
    if isinstance(head_cfg, list):
        for item in head_cfg:
            _set_angle_version(item, angle_version)


@function_config_to_basic_types
@logger.catch()
def build_hybrid_rcnn_obb_model(cfg: dict[str, Any] | DictConfig) -> Any:
    cfg = _ensure_cfg(cfg)
    require_mmdet()
    require_mmrotate()
    _init_mmdet_default_scope()

    base_cfg_source = cfg.detector.base_cfg or cfg.detector.base_config_path
    if base_cfg_source is None:
        raise ValueError("detector.base_cfg or detector.base_config_path must be set")

    base_cfg = _load_base_mmdet_cfg(base_cfg_source)
    _patch_common_model_cfg(base_cfg, cfg)
    _patch_rcnn_obb_heads_cfg(base_cfg, cfg)

    from mmdet.registry import MODELS  # type: ignore[import-not-found]

    model = MODELS.build(base_cfg.model)
    model.cfg = base_cfg
    return model


def load_default_cfg() -> DictConfig:
    return _create_default_cfg()


def load_cfg_from_yaml(cfg_path: str | Path) -> DictConfig:
    cfg = OmegaConf.load(str(cfg_path))
    return _ensure_cfg(cfg)

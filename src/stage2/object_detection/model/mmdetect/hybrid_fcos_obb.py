from __future__ import annotations

from pathlib import Path
from typing import Any

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

logger = logger.bind(_name_="hybrid_det_mmdet_fcos_obb")


def require_mmrotate() -> None:
    try:
        import mmrotate  # type: ignore[import-not-found]  # noqa: F401
    except Exception as exc:
        raise ImportError(
            "mmrotate is required for OBB models. Install mmrotate and ensure it is on PYTHONPATH."
        ) from exc


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


def _patch_fcos_obb_head_cfg(base_cfg: Any, hybrid_cfg: DictConfig) -> None:
    model_cfg: dict[str, Any] = base_cfg.model
    num_outs = int(getattr(hybrid_cfg.detector, "fpn_num_outs", 5))
    fpn_strides = _compute_fpn_strides([int(s) for s in hybrid_cfg.feature_strides], num_outs)

    bbox_head = model_cfg.get("bbox_head")
    if not isinstance(bbox_head, dict):
        raise KeyError("base_cfg.model.bbox_head is missing or not a dict (expected an FCOS-style config)")

    bbox_head["num_classes"] = int(hybrid_cfg.num_classes)
    bbox_head["in_channels"] = int(hybrid_cfg.fpn_out_channels)
    if "strides" in bbox_head:
        bbox_head["strides"] = fpn_strides

    angle_version = getattr(hybrid_cfg, "obb", {}).get("angle_version", None)
    if angle_version is not None and "angle_version" in bbox_head:
        bbox_head["angle_version"] = angle_version


@function_config_to_basic_types
@logger.catch()
def build_hybrid_fcos_obb_model(cfg: dict[str, Any] | DictConfig) -> Any:
    cfg = _ensure_cfg(cfg)
    require_mmdet()
    require_mmrotate()
    _init_mmdet_default_scope()

    base_cfg_source = cfg.detector.base_cfg or cfg.detector.base_config_path
    if base_cfg_source is None:
        raise ValueError("detector.base_cfg or detector.base_config_path must be set")

    base_cfg = _load_base_mmdet_cfg(base_cfg_source)
    _patch_common_model_cfg(base_cfg, cfg)
    _patch_fcos_obb_head_cfg(base_cfg, cfg)

    from mmdet.registry import MODELS  # type: ignore[import-not-found]

    model = MODELS.build(base_cfg.model)
    model.cfg = base_cfg
    return model


def load_default_cfg() -> DictConfig:
    return _create_default_cfg()


def load_cfg_from_yaml(cfg_path: str | Path) -> DictConfig:
    cfg = OmegaConf.load(str(cfg_path))
    return _ensure_cfg(cfg)

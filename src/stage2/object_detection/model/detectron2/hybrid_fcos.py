from __future__ import annotations

from typing import Any

import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.utilities.config_utils import function_config_to_basic_types

from .detectron2_utils import DETECTRON2_AVAILABLE, build_detectron2_cfg, build_detectron2_model
from .hybrid_backbone import _create_default_cfg as _create_default_backbone_cfg

logger = logger.bind(_name_="hybrid_fcos")


def _create_default_cfg() -> DictConfig:
    cfg = _create_default_backbone_cfg()
    cfg.detector = OmegaConf.create()
    cfg.detector.model_zoo_config = "COCO-Detection/fcos_R_50_FPN_1x.yaml"
    cfg.detector.score_thresh = 0.05
    cfg.detector.max_detections_per_image = 100
    cfg.detector.fcos_in_features = ["p3", "p4", "p5", "p6", "p7"]
    return cfg


def build_hybrid_fcos_model(cfg: DictConfig) -> nn.Module:
    if not DETECTRON2_AVAILABLE:
        raise RuntimeError("detectron2 is required to build HybridFCOS")
    d2_cfg = build_detectron2_cfg(cfg, model_type="fcos")
    return build_detectron2_model(d2_cfg)


class HybridFCOS(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = build_hybrid_fcos_model(cfg)

    def forward(self, *args: Any, **kwargs: Any):
        return self.model(*args, **kwargs)

    @classmethod
    @function_config_to_basic_types
    def create_model(cls, cfg: DictConfig = _create_default_cfg(), **overrides: Any) -> "HybridFCOS":
        if overrides:
            cfg.merge_with(overrides)
        logger.info("Creating HybridFCOS model.")
        return cls(cfg)

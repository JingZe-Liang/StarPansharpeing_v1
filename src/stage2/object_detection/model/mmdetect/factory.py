from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from torch import nn

from .hybrid_fcos import build_hybrid_fcos_model
from .hybrid_fcos_obb import build_hybrid_fcos_obb_model
from .hybrid_rcnn import build_hybrid_rcnn_model
from .hybrid_rcnn_obb import build_hybrid_rcnn_obb_model


def build_hybrid_model_from_yaml(
    model_type: str,
    cfg_path: str | Path,
    overrides: dict[str, Any] | DictConfig | None = None,
) -> nn.Module:
    cfg = OmegaConf.load(str(cfg_path))
    if not isinstance(cfg, DictConfig):
        raise TypeError("Model config must be a mapping (DictConfig).")
    if overrides is not None:
        cfg = OmegaConf.merge(cfg, overrides)
        if not isinstance(cfg, DictConfig):
            raise TypeError("Merged model config must be a mapping (DictConfig).")

    model_key = model_type.lower()
    if model_key == "fcos":
        return build_hybrid_fcos_model(cfg)
    if model_key == "rcnn":
        return build_hybrid_rcnn_model(cfg)
    if model_key == "fcos_obb":
        return build_hybrid_fcos_obb_model(cfg)
    if model_key == "rcnn_obb":
        return build_hybrid_rcnn_obb_model(cfg)
    raise ValueError(f"Unknown model_type: {model_type}")

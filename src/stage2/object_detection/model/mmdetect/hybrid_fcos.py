from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.utilities.config_utils import function_config_to_basic_types

# Reuse the existing hybrid tokenizer encoder implementation.
from ..hybrid_backbone import HybridTokenizerFeatureExtractor, build_hybrid_tokenizer_encoder
from ..hybrid_backbone import _create_default_cfg as _create_default_backbone_cfg

logger = logger.bind(_name_="hybrid_det_mmdet_fcos")


def _ensure_cfg(cfg: DictConfig | dict[str, Any] | Any) -> DictConfig:
    if isinstance(cfg, DictConfig):
        return cfg

    try:
        from mmengine.config import ConfigDict  # type: ignore[import-not-found]
    except Exception:
        ConfigDict = None  # type: ignore[assignment]

    if ConfigDict is not None and isinstance(cfg, ConfigDict):
        return OmegaConf.create(cfg.to_dict())
    return OmegaConf.create(cfg)


def _resolve_pixel_stats(value: float | list[float], length: int, name: str) -> list[float]:
    if isinstance(value, list):
        if len(value) == 1:
            return [float(value[0])] * length
        if len(value) != length:
            raise ValueError(f"{name} length must be 1 or {length}, got {len(value)}")
        return [float(v) for v in value]
    return [float(value)] * length


def _compute_fpn_strides(in_strides: list[int], num_outs: int) -> list[int]:
    if not in_strides:
        raise ValueError("in_strides must be non-empty")
    if num_outs <= 0:
        raise ValueError(f"num_outs must be positive, got {num_outs}")

    strides = list(in_strides[: min(len(in_strides), num_outs)])
    while len(strides) < num_outs:
        strides.append(int(strides[-1]) * 2)
    return strides


try:
    from mmengine.config import Config  # type: ignore[import-not-found]
    from mmengine.model import BaseModule  # type: ignore[import-not-found]
    from mmengine.registry import DefaultScope  # type: ignore[import-not-found]

    from mmdet.registry import MODELS  # type: ignore[import-not-found]

    MMDET_AVAILABLE = True
    _MMDET_IMPORT_ERROR: Exception | None = None
except Exception as exc:
    Config = object  # type: ignore[assignment]
    BaseModule = nn.Module  # type: ignore[assignment]
    DefaultScope = object  # type: ignore[assignment]
    MODELS = None  # type: ignore[assignment]
    MMDET_AVAILABLE = False
    _MMDET_IMPORT_ERROR = exc


def require_mmdet() -> None:
    if MMDET_AVAILABLE:
        return
    raise RuntimeError(f"mmdetection/mmengine is required but not available: {_MMDET_IMPORT_ERROR}")


def _init_mmdet_default_scope() -> None:
    # MMEngine uses a global default scope for registries. Base configs usually set it,
    # but we keep a small guard here for programmatic builds.
    require_mmdet()
    current = DefaultScope.get_current_instance()
    if current is None or getattr(current, "scope_name", None) != "mmdet":
        DefaultScope.get_instance("mmdet", scope_name="mmdet")


if MMDET_AVAILABLE:

    @MODELS.register_module()
    class HybridTokenizerBackboneMMDet(BaseModule):
        """MMDetection backbone wrapper around the project's hybrid tokenizer encoder.

        It returns a tuple of multi-scale feature maps (B, C, H, W) in the order of
        `feature_keys`. Strides are configured externally in the model config.
        """

        def __init__(self, hybrid_cfg: dict[str, Any] | DictConfig) -> None:
            super().__init__()
            cfg = _ensure_cfg(hybrid_cfg)

            encoder = build_hybrid_tokenizer_encoder(cfg)
            self._feature_keys = list(cfg.feature_keys)
            self._feature_strides = [int(s) for s in cfg.feature_strides]
            source_strides = list(getattr(cfg, "encoder_feature_strides", cfg.feature_strides))
            self._feature_extractor = HybridTokenizerFeatureExtractor(
                encoder=encoder,
                feature_keys=self._feature_keys,
                source_strides=[int(s) for s in source_strides],
                target_strides=self._feature_strides,
            )

            self.output_channels = list(self._feature_extractor.feature_channels)

        @property
        def feature_strides(self) -> list[int]:
            return list(self._feature_strides)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
            feats = self._feature_extractor(x)
            # Preserve order to match FPN in_channels.
            return tuple(feats[k] for k in self._feature_keys)


else:

    class HybridTokenizerBackboneMMDet(nn.Module):
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_mmdet()


def _create_default_cfg() -> DictConfig:
    cfg = _create_default_backbone_cfg()
    # A more typical detection stride setup. Override freely.
    cfg.encoder_feature_strides = [1, 2, 4, 8]
    cfg.feature_strides = [2, 4, 8, 16]

    cfg.detector = OmegaConf.create()
    # You can either provide a file path to an MMDetection config, or provide a config dict directly.
    # If `base_cfg` does not include a top-level "model" key, it will be treated as a "model" dict.
    cfg.detector.base_cfg = None
    # You can set this to a valid MMDetection config file (e.g., from your mmdetection repo).
    cfg.detector.base_config_path = None
    cfg.detector.fpn_num_outs = 5
    return cfg


def _normalize_base_cfg_dict(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    if "model" in cfg_dict:
        return cfg_dict
    return {"model": cfg_dict}


def _load_base_mmdet_cfg(cfg_source: str | Path | dict[str, Any] | DictConfig) -> Any:
    require_mmdet()
    if isinstance(cfg_source, DictConfig):
        container = OmegaConf.to_container(cfg_source, resolve=True)
        if not isinstance(container, dict):
            raise TypeError("cfg.detector.base_cfg must resolve to a dict")
        cfg_dict: dict[str, Any] = {str(k): v for k, v in container.items()}
        return Config(_normalize_base_cfg_dict(cfg_dict))

    if isinstance(cfg_source, dict):
        cfg_dict: dict[str, Any] = cfg_source
        return Config(_normalize_base_cfg_dict(cfg_dict))

    path = Path(cfg_source)
    if not path.exists():
        raise FileNotFoundError(f"base_config_path does not exist: {path}")
    return Config.fromfile(str(path))


def _patch_common_model_cfg(base_cfg: Any, hybrid_cfg: DictConfig) -> None:
    model_cfg: dict[str, Any] = base_cfg.model

    hybrid_cfg_dict = OmegaConf.to_container(hybrid_cfg, resolve=True)
    if not isinstance(hybrid_cfg_dict, dict):
        raise TypeError("hybrid_cfg must resolve to a dict")

    # Replace backbone with our hybrid backbone wrapper.
    model_cfg["backbone"] = {"type": "HybridTokenizerBackboneMMDet", "hybrid_cfg": hybrid_cfg_dict}

    # Patch data preprocessor stats to match hyperspectral channel count.
    if "data_preprocessor" in model_cfg and isinstance(model_cfg["data_preprocessor"], dict):
        dp = model_cfg["data_preprocessor"]
        in_ch = int(hybrid_cfg.input_channels)
        dp["mean"] = _resolve_pixel_stats(hybrid_cfg.pixel_mean, in_ch, "pixel_mean")
        dp["std"] = _resolve_pixel_stats(hybrid_cfg.pixel_std, in_ch, "pixel_std")
        # Hyperspectral is not BGR.
        dp["bgr_to_rgb"] = False

    # Patch neck channels to match the hybrid encoder outputs.
    if "neck" in model_cfg and isinstance(model_cfg["neck"], dict):
        neck = model_cfg["neck"]
        neck["in_channels"] = list(hybrid_cfg.tokenizer_feature.features_per_stage)
        neck["out_channels"] = int(hybrid_cfg.fpn_out_channels)


def _patch_fcos_head_cfg(base_cfg: Any, hybrid_cfg: DictConfig) -> None:
    model_cfg: dict[str, Any] = base_cfg.model
    num_outs = int(getattr(hybrid_cfg.detector, "fpn_num_outs", 5))
    fpn_strides = _compute_fpn_strides([int(s) for s in hybrid_cfg.feature_strides], num_outs)

    bbox_head = model_cfg.get("bbox_head")
    if not isinstance(bbox_head, dict):
        raise KeyError("base_cfg.model.bbox_head is missing or not a dict (expected an FCOS config)")

    bbox_head["num_classes"] = int(hybrid_cfg.num_classes)
    bbox_head["in_channels"] = int(hybrid_cfg.fpn_out_channels)
    bbox_head["strides"] = fpn_strides
    if "strides" in bbox_head:
        bbox_head["strides"] = fpn_strides

    # Ensure neck produces the number of feature levels expected by FCOS head.
    neck = model_cfg.get("neck")
    if isinstance(neck, dict):
        neck["num_outs"] = num_outs
        neck.setdefault("add_extra_convs", "on_output")


def build_hybrid_fcos_model(cfg: DictConfig) -> nn.Module:
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
                "(an MMDetection config file path, e.g. /path/to/mmdetection/configs/fcos/*.py)."
            )
        base_cfg = _load_base_mmdet_cfg(base_cfg_path)
    base_cfg.setdefault("default_scope", "mmdet")

    _patch_common_model_cfg(base_cfg, cfg)
    _patch_fcos_head_cfg(base_cfg, cfg)

    model = MODELS.build(base_cfg.model)
    if hasattr(model, "init_weights"):
        model.init_weights()
    return model


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
        logger.info("Creating HybridFCOS (MMDetection) model.")
        return cls(cfg)

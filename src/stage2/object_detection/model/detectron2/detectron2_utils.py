from __future__ import annotations

from typing import Any, Literal

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from .hybrid_backbone import HybridTokenizerFeatureExtractor, build_hybrid_tokenizer_encoder

logger = logger.bind(_name_="hybrid_det_detectron2")

HYBRID_BACKBONE_NAME = "build_hybrid_tokenizer_fpn_backbone"

try:
    from detectron2 import model_zoo  # type: ignore[import-not-found]
    from detectron2.config import CfgNode, get_cfg  # type: ignore[import-not-found]
    from detectron2.modeling import (  # type: ignore[import-not-found]
        BACKBONE_REGISTRY,
        Backbone,
        ShapeSpec,
        build_model,
    )
    from detectron2.modeling.backbone import FPN  # type: ignore[import-not-found]
    from detectron2.modeling.backbone.fpn import LastLevelMaxPool  # type: ignore[import-not-found]

    DETECTRON2_AVAILABLE = True
    _DETECTRON2_IMPORT_ERROR: Exception | None = None
except Exception as exc:

    class _MissingDependency:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(f"detectron2 is required but not available: {exc}")

    def _missing_callable(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(f"detectron2 is required but not available: {exc}")

    class _MissingModelZoo:
        def get_config_file(self, *_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError(f"detectron2 is required but not available: {exc}")

    class _MissingRegistry:
        def register(self):
            def _decorator(fn):
                return fn

            return _decorator

    model_zoo = _MissingModelZoo()
    CfgNode = _MissingDependency  # type: ignore[assignment]
    get_cfg = _missing_callable  # type: ignore[assignment]
    BACKBONE_REGISTRY = _MissingRegistry()  # type: ignore[assignment]
    Backbone = object  # type: ignore[assignment]
    ShapeSpec = _MissingDependency  # type: ignore[assignment]
    FPN = _MissingDependency  # type: ignore[assignment]
    LastLevelMaxPool = _MissingDependency  # type: ignore[assignment]
    build_model = _missing_callable  # type: ignore[assignment]
    DETECTRON2_AVAILABLE = False
    _DETECTRON2_IMPORT_ERROR = exc


def require_detectron2() -> None:
    if DETECTRON2_AVAILABLE:
        return
    raise RuntimeError(f"detectron2 is required but not available: {_DETECTRON2_IMPORT_ERROR}")


def _maybe_register():
    if not DETECTRON2_AVAILABLE:

        def _decorator(fn):
            return fn

        return _decorator
    return BACKBONE_REGISTRY.register()


def _resolve_pixel_stats(value: float | list[float], length: int, name: str) -> list[float]:
    if isinstance(value, list):
        if len(value) == 1:
            return [float(value[0])] * length
        if len(value) != length:
            raise ValueError(f"{name} length must be 1 or {length}, got {len(value)}")
        return [float(v) for v in value]
    return [float(value)] * length


def add_hybrid_backbone_config(d2_cfg: CfgNode, hybrid_cfg: DictConfig) -> None:
    require_detectron2()
    cfg_container = OmegaConf.create(OmegaConf.to_container(hybrid_cfg, resolve=True))
    if not isinstance(cfg_container, DictConfig):
        raise TypeError("hybrid_cfg must resolve to a DictConfig")
    hybrid_cfg = cfg_container

    d2_cfg.MODEL.HYBRID_BACKBONE = CfgNode()
    d2_cfg.MODEL.HYBRID_BACKBONE.TOKENIZER = OmegaConf.to_container(hybrid_cfg.tokenizer, resolve=True)
    d2_cfg.MODEL.HYBRID_BACKBONE.TOKENIZER_FEATURE = OmegaConf.to_container(hybrid_cfg.tokenizer_feature, resolve=True)
    d2_cfg.MODEL.HYBRID_BACKBONE.ADAPTER = OmegaConf.to_container(hybrid_cfg.adapter, resolve=True)
    d2_cfg.MODEL.HYBRID_BACKBONE.PRETRAINED_PATH = hybrid_cfg.tokenizer_pretrained_path
    d2_cfg.MODEL.HYBRID_BACKBONE.INPUT_CHANNELS = int(hybrid_cfg.input_channels)
    d2_cfg.MODEL.HYBRID_BACKBONE.OUT_FEATURES = list(hybrid_cfg.feature_keys)
    d2_cfg.MODEL.HYBRID_BACKBONE.OUT_STRIDES = list(hybrid_cfg.feature_strides)
    d2_cfg.MODEL.HYBRID_BACKBONE.OUT_CHANNELS = list(hybrid_cfg.tokenizer_feature.features_per_stage)
    d2_cfg.MODEL.HYBRID_BACKBONE.FPN_OUT_CHANNELS = int(hybrid_cfg.fpn_out_channels)
    d2_cfg.MODEL.HYBRID_BACKBONE.DEBUG = bool(hybrid_cfg._debug)
    d2_cfg.MODEL.HYBRID_BACKBONE.FREEZE_BACKBONE = bool(hybrid_cfg.freeze_backbone)


class HybridTokenizerBackboneD2(Backbone):
    def __init__(
        self,
        feature_extractor: HybridTokenizerFeatureExtractor,
        out_features: list[str],
        out_strides: list[int],
    ) -> None:
        if not DETECTRON2_AVAILABLE:
            raise RuntimeError("detectron2 is required to build HybridTokenizerBackboneD2")
        super().__init__()
        if len(out_features) != len(out_strides):
            raise ValueError(f"out_features length must match out_strides: {len(out_features)} != {len(out_strides)}")
        if len(out_features) != len(feature_extractor.feature_channels):
            raise ValueError(
                f"out_features length must match feature channels: {len(out_features)} != {len(feature_extractor.feature_channels)}"
            )
        self.feature_extractor = feature_extractor
        self._out_features = list(out_features)
        self._out_feature_strides = dict(zip(out_features, out_strides, strict=True))
        self._out_feature_channels = dict(zip(out_features, feature_extractor.feature_channels, strict=True))

    def forward(self, x):
        return self.feature_extractor(x)

    def output_shape(self) -> dict[str, ShapeSpec]:
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }


@_maybe_register()
def build_hybrid_tokenizer_fpn_backbone(cfg: CfgNode, input_shape: Any):
    require_detectron2()
    hy_cfg = cfg.MODEL.HYBRID_BACKBONE
    backbone_cfg = OmegaConf.create(
        {
            "tokenizer": hy_cfg.TOKENIZER,
            "tokenizer_feature": hy_cfg.TOKENIZER_FEATURE,
            "adapter": hy_cfg.ADAPTER,
            "tokenizer_pretrained_path": hy_cfg.PRETRAINED_PATH,
            "input_channels": hy_cfg.INPUT_CHANNELS,
            "_debug": hy_cfg.DEBUG,
            "freeze_backbone": hy_cfg.FREEZE_BACKBONE,
        }
    )

    encoder = build_hybrid_tokenizer_encoder(backbone_cfg)
    feature_extractor = HybridTokenizerFeatureExtractor(
        encoder=encoder,
        feature_keys=list(hy_cfg.OUT_FEATURES),
    )
    bottom_up = HybridTokenizerBackboneD2(
        feature_extractor=feature_extractor,
        out_features=list(hy_cfg.OUT_FEATURES),
        out_strides=list(hy_cfg.OUT_STRIDES),
    )

    return FPN(
        bottom_up=bottom_up,
        in_features=list(hy_cfg.OUT_FEATURES),
        out_channels=int(hy_cfg.FPN_OUT_CHANNELS),
        norm="",
        top_block=LastLevelMaxPool(),
    )


def build_detectron2_cfg(hybrid_cfg: DictConfig, model_type: Literal["rcnn", "fcos"]) -> CfgNode:
    require_detectron2()
    cfg_container = OmegaConf.create(OmegaConf.to_container(hybrid_cfg, resolve=True))
    if not isinstance(cfg_container, DictConfig):
        raise TypeError("hybrid_cfg must resolve to a DictConfig")
    hybrid_cfg = cfg_container

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(hybrid_cfg.detector.model_zoo_config))
    add_hybrid_backbone_config(cfg, hybrid_cfg)

    cfg.MODEL.BACKBONE.NAME = HYBRID_BACKBONE_NAME
    cfg.MODEL.FPN.IN_FEATURES = list(hybrid_cfg.feature_keys)
    cfg.MODEL.FPN.OUT_CHANNELS = int(hybrid_cfg.fpn_out_channels)

    cfg.MODEL.PIXEL_MEAN = _resolve_pixel_stats(hybrid_cfg.pixel_mean, int(hybrid_cfg.input_channels), "pixel_mean")
    cfg.MODEL.PIXEL_STD = _resolve_pixel_stats(hybrid_cfg.pixel_std, int(hybrid_cfg.input_channels), "pixel_std")
    cfg.MODEL.DEVICE = str(hybrid_cfg.device)

    if model_type == "rcnn":
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = int(hybrid_cfg.num_classes)
        cfg.MODEL.ROI_HEADS.IN_FEATURES = list(hybrid_cfg.detector.roi_in_features)
        cfg.MODEL.RPN.IN_FEATURES = list(hybrid_cfg.detector.rpn_in_features)
        if hybrid_cfg.detector.score_thresh is not None:
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(hybrid_cfg.detector.score_thresh)
    elif model_type == "fcos":
        cfg.MODEL.FCOS.NUM_CLASSES = int(hybrid_cfg.num_classes)
        cfg.MODEL.FCOS.IN_FEATURES = list(hybrid_cfg.detector.fcos_in_features)
        if hybrid_cfg.detector.score_thresh is not None:
            cfg.MODEL.FCOS.SCORE_THRESH_TEST = float(hybrid_cfg.detector.score_thresh)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if hybrid_cfg.detector.max_detections_per_image is not None:
        cfg.TEST.DETECTIONS_PER_IMAGE = int(hybrid_cfg.detector.max_detections_per_image)

    cfg.freeze()
    return cfg


def build_detectron2_model(cfg: CfgNode):
    require_detectron2()
    model = build_model(cfg)
    logger.info("Built detectron2 model with hybrid backbone.")
    return model

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from loguru import logger


def _append_dinov3_repo_to_path() -> None:
    repo_dir = Path(__file__).resolve().parents[2] / "src" / "stage1" / "utilities" / "losses" / "dinov3"
    if repo_dir.exists():
        sys.path.insert(0, str(repo_dir))


_append_dinov3_repo_to_path()

from src.stage2.object_detection.model.mmdetect import hybrid_rcnn as hr


def _ensure_dict_config(cfg: DictConfig | Any) -> DictConfig:
    if not isinstance(cfg, DictConfig):
        raise TypeError("Expected DictConfig")
    return cfg


def _load_yaml_cfg(filename: str) -> DictConfig:
    cfg_path = (
        Path(__file__).resolve().parents[2] / "src" / "stage2" / "object_detection" / "model" / "mmdetect" / filename
    )
    cfg = OmegaConf.load(cfg_path)
    return _ensure_dict_config(cfg)


def _build_hybrid_cfg() -> DictConfig:
    yaml_string = """
    tokenizer_feature:
        features_per_stage: [1152, 1152, 1152, 1152]
    input_channels: 155
    pixel_mean: 0.0
    pixel_std: 1.0
    feature_keys: [res2, res3, res4, res5]
    encoder_feature_strides: [4, 8, 16, 32]
    feature_strides: [2, 4, 8, 16]
    fpn_out_channels: 256
    num_classes: 3
    detector:
        fpn_num_outs: 5
    """
    return _ensure_dict_config(OmegaConf.create(yaml_string))


def _build_base_cfg() -> SimpleNamespace:
    model: dict[str, Any] = {
        "data_preprocessor": {"mean": [0.5], "std": [0.25], "bgr_to_rgb": True},
        "backbone": {"type": "ResNet"},
        "neck": {"in_channels": [1, 2, 3, 4], "out_channels": 64, "num_outs": 4},
        "rpn_head": {
            "in_channels": 128,
            "anchor_generator": {"strides": [4, 8, 16, 32, 64]},
            "prior_generator": {"strides": [4, 8, 16, 32, 64]},
        },
        "roi_head": {
            "bbox_roi_extractor": {"featmap_strides": [4, 8, 16, 32]},
            "mask_roi_extractor": {"featmap_strides": [4, 8, 16, 32]},
            "bbox_head": {"num_classes": 80},
        },
    }
    return SimpleNamespace(model=model)


def test_patch_rcnn_config_updates_fields() -> None:
    base_cfg = _build_base_cfg()
    hybrid_cfg = _build_hybrid_cfg()

    hr._patch_common_model_cfg(base_cfg, hybrid_cfg)
    hr._patch_rcnn_heads_cfg(base_cfg, hybrid_cfg)

    model = base_cfg.model
    expected_strides = hr._compute_fpn_strides(
        [int(s) for s in hybrid_cfg.feature_strides], int(hybrid_cfg.detector.fpn_num_outs)
    )

    assert model["backbone"]["type"] == "HybridTokenizerBackboneMMDet"
    assert model["data_preprocessor"]["mean"] == [0.0] * int(hybrid_cfg.input_channels)
    assert model["data_preprocessor"]["std"] == [1.0] * int(hybrid_cfg.input_channels)
    assert model["data_preprocessor"]["bgr_to_rgb"] is False
    assert model["neck"]["in_channels"] == list(hybrid_cfg.tokenizer_feature.features_per_stage)
    assert model["neck"]["out_channels"] == int(hybrid_cfg.fpn_out_channels)
    assert model["neck"]["num_outs"] == int(hybrid_cfg.detector.fpn_num_outs)
    assert model["rpn_head"]["anchor_generator"]["strides"] == expected_strides
    assert model["rpn_head"]["prior_generator"]["strides"] == expected_strides
    assert model["roi_head"]["bbox_roi_extractor"]["featmap_strides"] == expected_strides[:4]
    assert model["roi_head"]["mask_roi_extractor"]["featmap_strides"] == expected_strides[:4]
    assert model["roi_head"]["bbox_head"]["num_classes"] == int(hybrid_cfg.num_classes)


def _build_debug_rcnn_cfg(cfg: DictConfig) -> DictConfig:
    container = OmegaConf.to_container(cfg, resolve=True)
    new_cfg = _ensure_dict_config(OmegaConf.create(container))
    new_cfg.detector.fpn_num_outs = int(getattr(new_cfg.detector, "fpn_num_outs", 5))
    new_cfg.encoder_feature_strides = [4, 8, 16, 32]
    new_cfg.feature_strides = [2, 4, 8, 16]
    new_cfg.tokenizer_feature.features_per_stage = [1152, 1152, 1152, 1152]
    new_cfg.input_channels = 3
    new_cfg.tokenizer_feature.in_channels = 3
    new_cfg.num_classes = 3
    new_cfg._debug = True
    new_cfg.tokenizer_pretrained_path = None
    return new_cfg


def _build_forward_rcnn_cfg(cfg: DictConfig) -> DictConfig:
    container = OmegaConf.to_container(cfg, resolve=True)
    new_cfg = _ensure_dict_config(OmegaConf.create(container))
    new_cfg.detector.fpn_num_outs = int(getattr(new_cfg.detector, "fpn_num_outs", 5))
    new_cfg.encoder_feature_strides = [4, 8, 16, 32]
    new_cfg.feature_strides = [2, 4, 8, 16]
    new_cfg.tokenizer_feature.features_per_stage = [8, 8, 8, 8]
    new_cfg.fpn_out_channels = 256
    new_cfg.input_channels = 3
    new_cfg.tokenizer_feature.in_channels = 3
    new_cfg._debug = True
    new_cfg.tokenizer_pretrained_path = None
    return new_cfg


def _register_dummy_backbone() -> None:
    from mmengine.model import BaseModule
    from mmdet.registry import MODELS  # type: ignore[import-not-found]
    import torch.nn as nn
    import torch.nn.functional as F

    @MODELS.register_module(name="HybridTokenizerBackboneMMDet", force=True)
    class DummyHybridBackbone(BaseModule):
        def __init__(self, hybrid_cfg: dict[str, Any]) -> None:
            super().__init__()
            tokenizer_feature = hybrid_cfg.get("tokenizer_feature", {})
            self.out_channels = list(tokenizer_feature.get("features_per_stage", [8, 8, 8, 8]))
            self.strides = list(hybrid_cfg.get("feature_strides", [2, 4, 8, 16]))
            in_ch = int(hybrid_cfg.get("input_channels", 3))
            self.proj = nn.ModuleList([nn.Conv2d(in_ch, ch, kernel_size=1) for ch in self.out_channels])

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
            feats = []
            for stride, proj in zip(self.strides, self.proj, strict=False):
                if stride > 1:
                    feat = F.avg_pool2d(x, kernel_size=stride, stride=stride)
                else:
                    feat = x
                feats.append(proj(feat))
            return tuple(feats)


def _build_dummy_samples(batch_size: int, img_shape: tuple[int, int, int]) -> list[Any]:
    from mmdet.structures import DetDataSample  # type: ignore[import-not-found]

    samples = []
    for _ in range(batch_size):
        sample = DetDataSample()
        sample.set_metainfo({"img_shape": img_shape, "ori_shape": img_shape, "scale_factor": (1.0, 1.0)})
        samples.append(sample)
    return samples


@logger.catch()
def test_rcnn_param_count() -> None:
    from fvcore.nn import parameter_count_table

    cfg = _build_debug_rcnn_cfg(_load_yaml_cfg("hybrid_mmdet_rcnn.yaml"))
    model = hr.build_hybrid_rcnn_model(cfg)
    print(parameter_count_table(model))


@logger.catch()
def test_rcnn_forward() -> None:
    _register_dummy_backbone()

    cfg = _build_forward_rcnn_cfg(_load_yaml_cfg("hybrid_mmdet_rcnn.yaml"))
    model = hr.build_hybrid_rcnn_model(cfg)
    model.eval()
    inputs = torch.randn(1, int(cfg.input_channels), 64, 64)

    img_shape = (int(inputs.shape[2]), int(inputs.shape[3]), int(inputs.shape[1]))
    data_samples = _build_dummy_samples(int(inputs.shape[0]), img_shape)
    with torch.no_grad():
        outputs = model.forward(inputs, data_samples=data_samples, mode="tensor")

    assert outputs is not None

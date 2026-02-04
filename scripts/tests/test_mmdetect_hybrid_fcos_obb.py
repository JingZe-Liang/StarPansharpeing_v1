from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.stage2.object_detection.model.mmdetect import hybrid_fcos_obb as hfo


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
    obb:
        angle_version: le90
    """
    return _ensure_dict_config(OmegaConf.create(yaml_string))


def _build_base_cfg() -> SimpleNamespace:
    model: dict[str, Any] = {
        "data_preprocessor": {"mean": [0.5], "std": [0.25], "bgr_to_rgb": True},
        "backbone": {"type": "ResNet"},
        "neck": {"in_channels": [1, 2, 3, 4], "out_channels": 64, "num_outs": 4},
        "bbox_head": {
            "type": "RotatedFCOSHead",
            "num_classes": 80,
            "in_channels": 128,
            "strides": [8, 16, 32, 64, 128],
            "angle_version": "le90",
        },
    }
    return SimpleNamespace(model=model)


def test_patch_fcos_obb_config_updates_fields() -> None:
    base_cfg = _build_base_cfg()
    hybrid_cfg = _build_hybrid_cfg()

    hfo._patch_common_model_cfg(base_cfg, hybrid_cfg)
    hfo._patch_fcos_obb_head_cfg(base_cfg, hybrid_cfg)

    model = base_cfg.model
    expected_strides = hfo._compute_fpn_strides(
        [int(s) for s in hybrid_cfg.feature_strides], int(hybrid_cfg.detector.fpn_num_outs)
    )

    assert model["backbone"]["type"] == "HybridTokenizerBackboneMMDet"
    assert model["data_preprocessor"]["mean"] == [0.0] * int(hybrid_cfg.input_channels)
    assert model["data_preprocessor"]["std"] == [1.0] * int(hybrid_cfg.input_channels)
    assert model["data_preprocessor"]["bgr_to_rgb"] is False
    assert model["neck"]["in_channels"] == list(hybrid_cfg.tokenizer_feature.features_per_stage)
    assert model["neck"]["out_channels"] == int(hybrid_cfg.fpn_out_channels)
    assert model["neck"]["num_outs"] == int(hybrid_cfg.detector.fpn_num_outs)
    assert model["bbox_head"]["num_classes"] == int(hybrid_cfg.num_classes)
    assert model["bbox_head"]["in_channels"] == int(hybrid_cfg.fpn_out_channels)
    assert model["bbox_head"]["strides"] == expected_strides
    assert model["bbox_head"]["angle_version"] == str(hybrid_cfg.obb.angle_version)


def test_fcos_obb_yaml_loads() -> None:
    cfg = _load_yaml_cfg("hybrid_mmdet_fcos_obb.yaml")
    assert cfg.obb.angle_version

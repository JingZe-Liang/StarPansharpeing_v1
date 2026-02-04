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

from src.stage2.object_detection.model.mmdetect import hybrid_fcos as hf


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
        "bbox_head": {
            "num_classes": 80,
            "in_channels": 128,
            "strides": [8, 16, 32, 64, 128],
        },
    }
    return SimpleNamespace(model=model)


def test_patch_fcos_config_updates_fields() -> None:
    base_cfg = _build_base_cfg()
    hybrid_cfg = _build_hybrid_cfg()

    hf._patch_common_model_cfg(base_cfg, hybrid_cfg)
    hf._patch_fcos_head_cfg(base_cfg, hybrid_cfg)

    model = base_cfg.model
    expected_strides = hf._compute_fpn_strides(
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
    assert model["bbox_head"]["strides"] == expected_strides


def _build_debug_fcos_cfg(cfg: DictConfig) -> DictConfig:
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


def _build_forward_fcos_cfg(cfg: DictConfig) -> DictConfig:
    container = OmegaConf.to_container(cfg, resolve=True)
    new_cfg = _ensure_dict_config(OmegaConf.create(container))
    new_cfg.detector.fpn_num_outs = int(getattr(new_cfg.detector, "fpn_num_outs", 5))
    new_cfg.encoder_feature_strides = [4, 8, 16, 32]
    new_cfg.feature_strides = [2, 4, 8, 16]
    new_cfg.tokenizer_feature.features_per_stage = [8, 8, 8, 8]
    new_cfg.fpn_out_channels = 8
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


def test_fcos_param_count() -> None:
    from fvcore.nn import parameter_count_table

    cfg = _build_debug_fcos_cfg(_load_yaml_cfg("hybrid_mmdet_fcos.yaml"))
    model = hf.build_hybrid_fcos_model(cfg)
    print(parameter_count_table(model))


@logger.catch()
def test_fcos_forward() -> None:
    _register_dummy_backbone()

    cfg = _build_forward_fcos_cfg(_load_yaml_cfg("hybrid_mmdet_fcos.yaml"))
    model = hf.build_hybrid_fcos_model(cfg)
    model.eval()
    inputs = torch.randn(1, int(cfg.input_channels), 64, 64)

    img_shape = (int(inputs.shape[2]), int(inputs.shape[3]), int(inputs.shape[1]))
    data_samples = _build_dummy_samples(int(inputs.shape[0]), img_shape)
    with torch.no_grad():
        outputs = model.forward(inputs, data_samples=data_samples, mode="tensor")

    assert outputs is not None


def test_full_model_forward() -> None:
    cfg = _load_yaml_cfg("hybrid_mmdet_fcos.yaml")
    cfg._debug = True
    cfg.tokenizer_pretrained_path = None
    device = "cuda:1"

    backbone = hf.HybridTokenizerBackboneMMDet(cfg).to(device)
    backbone.eval()

    img_size = int(cfg.tokenizer.trans_enc_cfg.img_size)
    patch_size = int(cfg.tokenizer.trans_enc_cfg.patch_size)
    spatial = img_size * patch_size

    inputs = torch.randn(1, int(cfg.input_channels), spatial, spatial).to(device)
    with torch.no_grad() and torch.autocast(device, torch.bfloat16):
        outputs = tuple(backbone(inputs))

    output_channels: list[int] = [int(ch) for ch in getattr(backbone, "output_channels", [])]
    assert len(outputs) == len(cfg.feature_keys)
    for feat, channels in zip(outputs, output_channels, strict=True):
        assert int(feat.shape[1]) == int(channels)

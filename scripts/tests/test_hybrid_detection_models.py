from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.stage2.object_detection.model.detectron2_utils import DETECTRON2_AVAILABLE
from src.stage2.object_detection.model.hybrid_backbone import HybridTokenizerFeatureExtractor
from src.stage2.object_detection.model.hybrid_fcos import build_hybrid_fcos_model, _create_default_cfg as fcos_cfg
from src.stage2.object_detection.model.hybrid_rcnn import build_hybrid_rcnn_model, _create_default_cfg as rcnn_cfg


class DummyEncoder(nn.Module):
    def __init__(self, out_channels: list[int], strides: list[int]) -> None:
        super().__init__()
        if len(out_channels) != len(strides):
            raise ValueError("out_channels and strides must have same length")
        self.output_channels = list(out_channels)
        self._strides = list(strides)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        b, _, h, w = x.shape
        feats: list[torch.Tensor] = []
        for ch, stride in zip(self.output_channels, self._strides, strict=True):
            feats.append(torch.zeros((b, ch, h // stride, w // stride), dtype=x.dtype))
        return feats


def test_feature_extractor_outputs_dict() -> None:
    encoder = DummyEncoder([8, 16, 32], [1, 2, 4])
    extractor = HybridTokenizerFeatureExtractor(encoder, feature_keys=["res2", "res3", "res4"])
    x = torch.randn(2, 3, 64, 64)
    feats = extractor(x)
    assert list(feats.keys()) == ["res2", "res3", "res4"]
    assert feats["res2"].shape == (2, 8, 64, 64)
    assert feats["res3"].shape == (2, 16, 32, 32)
    assert feats["res4"].shape == (2, 32, 16, 16)


def test_feature_extractor_key_mismatch() -> None:
    encoder = DummyEncoder([8, 16], [1, 2])
    with pytest.raises(ValueError):
        _ = HybridTokenizerFeatureExtractor(encoder, feature_keys=["res2"])


@pytest.mark.skipif(DETECTRON2_AVAILABLE, reason="detectron2 available; skipping missing-dependency checks")
def test_hybrid_rcnn_requires_detectron2() -> None:
    cfg = rcnn_cfg()
    with pytest.raises(RuntimeError):
        _ = build_hybrid_rcnn_model(cfg)


@pytest.mark.skipif(DETECTRON2_AVAILABLE, reason="detectron2 available; skipping missing-dependency checks")
def test_hybrid_fcos_requires_detectron2() -> None:
    cfg = fcos_cfg()
    with pytest.raises(RuntimeError):
        _ = build_hybrid_fcos_model(cfg)

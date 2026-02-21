from __future__ import annotations

import pytest
import torch

from src.stage2.segmentation.models.upernet_four_features import UPerNetFourFeatureDecoder


def _make_features(batch_size: int = 2) -> list[torch.Tensor]:
    return [
        torch.randn(batch_size, 64, 64, 64),
        torch.randn(batch_size, 128, 32, 32),
        torch.randn(batch_size, 256, 16, 16),
        torch.randn(batch_size, 512, 8, 8),
    ]


def test_upernet_four_features_eval_output_shape() -> None:
    model = UPerNetFourFeatureDecoder(in_channels=[64, 128, 256, 512], num_classes=6)
    model.eval()
    with torch.no_grad():
        out = model(_make_features())
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 6, 64, 64)


def test_upernet_four_features_train_returns_main_and_aux() -> None:
    model = UPerNetFourFeatureDecoder(in_channels=[64, 128, 256, 512], num_classes=6)
    model.train()
    out = model(_make_features())
    assert isinstance(out, list)
    assert len(out) == 2
    main_logits, aux_logits = out
    assert main_logits.shape == (2, 6, 64, 64)
    assert aux_logits.shape == (2, 6, 64, 64)


def test_upernet_four_features_invalid_feature_count() -> None:
    model = UPerNetFourFeatureDecoder(in_channels=[64, 128, 256, 512], num_classes=6)
    with pytest.raises(ValueError, match="exactly 4"):
        _ = model(_make_features()[:3])


def test_upernet_four_features_channel_mismatch() -> None:
    model = UPerNetFourFeatureDecoder(in_channels=[64, 128, 256, 512], num_classes=6)
    features = _make_features()
    features[1] = torch.randn(2, 129, 32, 32)
    with pytest.raises(ValueError, match="channel mismatch"):
        _ = model(features)


def test_upernet_four_features_batch_mismatch() -> None:
    model = UPerNetFourFeatureDecoder(in_channels=[64, 128, 256, 512], num_classes=6)
    features = _make_features()
    features[2] = torch.randn(3, 256, 16, 16)
    with pytest.raises(ValueError, match="batch size mismatch"):
        _ = model(features)

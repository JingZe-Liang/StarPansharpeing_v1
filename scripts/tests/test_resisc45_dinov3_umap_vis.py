import torch

from scripts.infer.resisc45_dinov3_umap_vis import _select_adapter_feature


def test_select_adapter_feature_returns_last_feature_for_negative_index() -> None:
    features = [
        torch.zeros(2, 4, 3, 3),
        torch.ones(2, 4, 3, 3),
        torch.full((2, 4, 3, 3), 2.0),
    ]

    selected = _select_adapter_feature(features, feature_index=-1)

    assert torch.equal(selected, features[-1])


def test_select_adapter_feature_raises_on_empty_features() -> None:
    try:
        _select_adapter_feature([], feature_index=0)
    except ValueError as exc:
        assert "no features" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError for empty feature list.")

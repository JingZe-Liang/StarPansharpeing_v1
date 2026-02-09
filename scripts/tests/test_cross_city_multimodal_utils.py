from __future__ import annotations

import numpy as np
import pytest
import torch

from src.stage2.segmentation.data.cross_city_multimodal import (
    generate_patch_coords,
    generate_sliding_positions,
    remap_background_to_ignore,
)
from src.stage2.segmentation.models.tokenizer_backbone_adapted_multimodal import split_modal_tensor


def test_generate_sliding_positions_cover_last_tile() -> None:
    pos = generate_sliding_positions(size=10, patch_size=4, stride=3)
    assert pos == [0, 3, 6]


def test_generate_patch_coords_count() -> None:
    coords = generate_patch_coords(height=8, width=8, patch_size=4, stride=2)
    assert len(coords) == 9
    assert coords[0] == (0, 0)
    assert coords[-1] == (4, 4)


def test_remap_background_to_ignore() -> None:
    label = np.array([[0, 1, 2], [3, 0, 4]], dtype=np.int64)
    mapped = remap_background_to_ignore(label, ignore_index=255)
    expected = np.array([[255, 0, 1], [2, 255, 3]], dtype=np.int64)
    np.testing.assert_array_equal(mapped, expected)


def test_split_modal_tensor() -> None:
    x = torch.randn(2, 16, 32, 32)
    hsi, msi, sar = split_modal_tensor(x, [10, 4, 2])
    assert hsi.shape == (2, 10, 32, 32)
    assert msi.shape == (2, 4, 32, 32)
    assert sar.shape == (2, 2, 32, 32)


def test_split_modal_tensor_channel_mismatch() -> None:
    x = torch.randn(1, 8, 16, 16)
    with pytest.raises(ValueError, match="Input channels mismatch"):
        _ = split_modal_tensor(x, [4, 4, 2])

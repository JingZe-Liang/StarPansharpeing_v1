from __future__ import annotations

import pytest
import torch

from src.stage2.generative.Sana.diffusion.model.nets.sana_blocks import (
    MultiHeadCrossAttention,
)


def test_mask_to_kv_seqlens_with_2d_mask() -> None:
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.int64)
    seqlens = MultiHeadCrossAttention._mask_to_kv_seqlens(mask, batch_size=2)
    assert seqlens == [3, 4]


def test_mask_to_kv_seqlens_with_4d_mask() -> None:
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)[:, None, None, :]
    seqlens = MultiHeadCrossAttention._mask_to_kv_seqlens(mask, batch_size=2)
    assert seqlens == [2, 3]


def test_mask_to_kv_seqlens_rejects_invalid_ndim() -> None:
    mask = torch.ones((2, 1, 3), dtype=torch.int64)
    with pytest.raises(ValueError, match="Unsupported mask ndim"):
        MultiHeadCrossAttention._mask_to_kv_seqlens(mask, batch_size=2)

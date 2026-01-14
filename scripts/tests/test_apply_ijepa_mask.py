import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.stage1.cosmos.modules.transformer import TransformerTokenizer
from src.stage1.self_supervised import MaskCollator
from src.stage1.self_supervised import jepa_blockutils


def test_mask_collator_lengths_and_ranges(monkeypatch) -> None:
    class _DummyLock:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _DummyValue:
        def __init__(self, _typecode, value: int):
            self.value = value
            self._lock = _DummyLock()

        def get_lock(self):
            return self._lock

    monkeypatch.setattr(jepa_blockutils, "Value", _DummyValue)

    batch_size = 2
    img_size = 64
    patch_size = 8
    nenc = 2
    npred = 3

    x = torch.randn(batch_size, 3, img_size, img_size)
    collator = MaskCollator(
        input_size=(img_size, img_size),
        patch_size=patch_size,
        enc_mask_scale=(0.8, 1.0),
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        nenc=nenc,
        npred=npred,
        min_keep=4,
        allow_overlap=False,
    )
    x_out, masks_enc, masks_pred = collator(x)

    assert tuple(x_out.shape) == tuple(x.shape)
    assert isinstance(masks_enc, list) and len(masks_enc) == nenc
    assert isinstance(masks_pred, list) and len(masks_pred) == npred

    num_patches = (img_size // patch_size) * (img_size // patch_size)
    for m in [*masks_enc, *masks_pred]:
        assert m.ndim == 2 and m.shape[0] == batch_size
        assert m.min().item() >= 0
        assert m.max().item() < num_patches


def test_apply_ijepa_mask_prefix_alignment_with_multiple_masks() -> None:
    batch_size = 2
    num_prefix_tokens = 2
    num_patch_tokens = 9
    embed_dim = 4
    rope_dim = 8

    # Make prefix tokens and patch tokens constant per-sample so we can verify alignment.
    sample_values = torch.arange(1, batch_size + 1, dtype=torch.float32)[:, None, None]
    prefixed_tokens = sample_values.repeat(1, num_prefix_tokens, embed_dim)
    patch_tokens = sample_values.repeat(1, num_patch_tokens, embed_dim)
    x = torch.cat([prefixed_tokens, patch_tokens], dim=1)

    n_masks = 3
    num_keep = 5
    g = torch.Generator().manual_seed(0)
    masks = [
        torch.stack([torch.randperm(num_patch_tokens, generator=g)[:num_keep] for _ in range(batch_size)], dim=0)
        for _ in range(n_masks)
    ]
    rope = torch.randn(num_patch_tokens, rope_dim)

    dummy = SimpleNamespace(num_prefix_tokens=num_prefix_tokens, _rope_is_mixed=False)
    dummy._jepa_apply_masks = TransformerTokenizer._jepa_apply_masks.__get__(dummy, type(dummy))  # type: ignore[attr-defined]

    x_masked, rope_masked = TransformerTokenizer._apply_ijepa_mask(dummy, x, masks, rope)  # type: ignore[arg-type]

    assert tuple(x_masked.shape) == (batch_size * n_masks, num_prefix_tokens + num_keep, embed_dim)
    assert rope_masked is not None
    assert tuple(rope_masked.shape) == (batch_size * n_masks, 1, num_keep, rope_dim)

    prefix_mean = x_masked[:, :num_prefix_tokens].mean(dim=(1, 2))
    patch_mean = x_masked[:, num_prefix_tokens:].mean(dim=(1, 2))
    assert torch.allclose(prefix_mean, patch_mean)

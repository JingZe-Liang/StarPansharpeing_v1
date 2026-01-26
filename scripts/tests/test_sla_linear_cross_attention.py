import torch
import pytest

from src.stage2.layers.SLA import LinearCrossAttention


def test_linear_cross_attention_supports_different_lengths() -> None:
    torch.manual_seed(0)
    b, h, lq, lk, d = 2, 4, 128, 257, 32
    q = torch.randn(b, h, lq, d, requires_grad=True)
    k = torch.randn(b, h, lk, d, requires_grad=True)
    v = torch.randn(b, h, lk, d, requires_grad=True)

    attn = LinearCrossAttention(feature_map="relu", use_bf16=True)
    out = attn(q, k, v)

    assert out.shape == (b, h, lq, d)

    out.mean().backward()
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None


def test_linear_cross_attention_requires_kv_same_length() -> None:
    b, h, lq, lk, d = 1, 2, 8, 9, 16
    q = torch.randn(b, h, lq, d)
    k = torch.randn(b, h, lk, d)
    v = torch.randn(b, h, lk + 1, d)
    attn = LinearCrossAttention()

    with pytest.raises(ValueError):
        _ = attn(q, k, v)

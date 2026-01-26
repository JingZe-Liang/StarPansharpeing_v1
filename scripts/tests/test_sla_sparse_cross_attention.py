import pytest
import torch

from src.stage2.layers.SLA import SparseLinearCrossAttention


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton SLA kernels.")
def test_sla_sparse_cross_attention_forward_backward_cuda() -> None:
    pytest.importorskip("triton")

    torch.manual_seed(0)

    # Keep shapes small to reduce Triton compilation time on first run.
    b, h, lq, lk, d = 1, 2, 128, 257, 64

    q = torch.randn(b, h, lq, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(b, h, lk, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(b, h, lk, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    attn = SparseLinearCrossAttention(
        head_dim=d,
        topk=0.25,
        feature_map="relu",
        BLKQ=64,
        BLKK=64,
        use_bf16=True,
    ).cuda()

    out = attn(q, k, v)
    assert out.shape == (b, h, lq, d)

    loss = out.float().square().mean()
    loss.backward()

    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton SLA kernels.")
def test_sla_sparse_cross_attention_matches_dense_when_topk_is_1_cuda() -> None:
    """
    If topk=1.0, sparse block selection should include all key blocks, reducing to dense attention.
    This checks the sparse softmax branch output (linear branch projection is zero-initialized).
    """
    pytest.importorskip("triton")

    torch.manual_seed(0)

    b, h, lq, lk, d = 1, 2, 128, 257, 64
    q = torch.randn(b, h, lq, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, h, lk, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(b, h, lk, d, device="cuda", dtype=torch.bfloat16)

    attn = SparseLinearCrossAttention(
        head_dim=d,
        topk=1.0,
        feature_map="relu",
        BLKQ=64,
        BLKK=64,
        use_bf16=True,
    ).cuda()

    out_sparse = attn(q, k, v)
    out_dense = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

    # BF16 + different kernels: use a tolerant threshold.
    max_err = (out_sparse - out_dense).abs().max().item()
    assert max_err < 0.01

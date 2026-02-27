from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

from src.stage1.cosmos.third_party.FlashWindowAttention.flash_swin_attn.func_flash_swin import flash_swin_attn_func
from src.stage1.cosmos.third_party.FlashWindowAttention.flash_swin_attn.func_swin import mha_core


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for FlashSwin tests")


def _make_shift_mask(num_windows: int, seq: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((num_windows, seq, seq), dtype=dtype, device=device)
    third = seq // 3
    mask[:, :third, third : 2 * third] = -100.0
    mask[:, third : 2 * third, :third] = -100.0
    mask[:, -third:, :-third] = -100.0
    return mask


def _expand_window_mask(mask: torch.Tensor, batch: int) -> torch.Tensor:
    n_w, seq, _ = mask.shape
    return mask.unsqueeze(0).expand(batch // n_w, n_w, seq, seq).reshape(batch, seq, seq).contiguous()


def _run_eager(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor | None,
    scale_qk: float,
) -> torch.Tensor:
    return mha_core(q, k, v, bias, mask, scale_qk)


def _run_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor | None,
    scale_qk: float,
) -> torch.Tensor:
    attn_bias = bias.unsqueeze(0)
    if mask is not None:
        attn_bias = attn_bias + _expand_window_mask(mask, q.size(0)).unsqueeze(1)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=0.0, scale=scale_qk)


def _run_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor | None,
    scale_qk: float,
) -> torch.Tensor:
    window_mask = _expand_window_mask(mask, q.size(0)) if mask is not None else None
    return flash_swin_attn_func(q, k, v, bias, window_mask, scale_qk)


def _forward_backward(
    fn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor | None,
    scale_qk: float,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    q_i = q.detach().clone().requires_grad_(True)
    k_i = k.detach().clone().requires_grad_(True)
    v_i = v.detach().clone().requires_grad_(True)
    b_i = bias.detach().clone().requires_grad_(True)

    out = fn(q_i, k_i, v_i, b_i, mask, scale_qk)
    loss = out.float().square().mean() + 0.05 * out.float().abs().mean()
    loss.backward()

    return out.detach(), (q_i.grad.detach(), k_i.grad.detach(), v_i.grad.detach(), b_i.grad.detach())


def _tolerance(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float16:
        return 2e-2, 2e-3
    return 8e-2, 2e-2


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_shift_mask", [False, True])
def test_flash_swin_backends_forward_backward(dtype: torch.dtype, use_shift_mask: bool) -> None:
    torch.manual_seed(7)

    batch, heads, seq, head_dim = 4, 4, 49, 32
    scale_qk = head_dim**-0.5
    device = torch.device("cuda")

    q = torch.randn((batch, heads, seq, head_dim), device=device, dtype=dtype)
    k = torch.randn((batch, heads, seq, head_dim), device=device, dtype=dtype)
    v = torch.randn((batch, heads, seq, head_dim), device=device, dtype=dtype)
    bias = torch.randn((heads, seq, seq), device=device, dtype=dtype)

    mask = _make_shift_mask(num_windows=2, seq=seq, dtype=dtype, device=device) if use_shift_mask else None

    out_eager, grads_eager = _forward_backward(_run_eager, q, k, v, bias, mask, scale_qk)
    out_sdpa, grads_sdpa = _forward_backward(_run_sdpa, q, k, v, bias, mask, scale_qk)
    out_triton, grads_triton = _forward_backward(_run_triton, q, k, v, bias, mask, scale_qk)

    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(out_sdpa.float(), out_eager.float(), rtol=rtol, atol=atol)
    torch.testing.assert_close(out_triton.float(), out_eager.float(), rtol=rtol, atol=atol)

    for grad_sdpa, grad_eager in zip(grads_sdpa, grads_eager, strict=True):
        torch.testing.assert_close(grad_sdpa.float(), grad_eager.float(), rtol=rtol, atol=atol)
    for grad_triton, grad_eager in zip(grads_triton, grads_eager, strict=True):
        torch.testing.assert_close(grad_triton.float(), grad_eager.float(), rtol=rtol, atol=atol)

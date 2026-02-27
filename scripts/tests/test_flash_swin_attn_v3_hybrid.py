from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from src.stage1.cosmos.third_party.FlashWindowAttention.flash_swin_attn.func_flash_swin_hybrid import (
    hybrid_sdpa_fwd_flash_swin_v3_bwd,
)
from src.stage1.cosmos.third_party.FlashWindowAttention.flash_swin_attn.func_flash_swin_v2 import (
    flash_swin_attn_func_v2,
)
from src.stage1.cosmos.third_party.FlashWindowAttention.flash_swin_attn.func_flash_swin_v3 import (
    flash_swin_attn_func_v3,
)
from src.stage1.cosmos.third_party.FlashWindowAttention.flash_swin_attn.func_swin import mha_core


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for FlashSwin Triton v3 tests")


def _make_shift_mask(num_windows: int, seq: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((num_windows, seq, seq), dtype=dtype, device=device)
    block = seq // 4
    mask[:, :block, block : 3 * block] = -100.0
    mask[:, block : 3 * block, :block] = -100.0
    mask[:, -block:, :-block] = -100.0
    return mask


def _expand_window_mask(mask: torch.Tensor, batch: int) -> torch.Tensor:
    n_w, seq, _ = mask.shape
    return mask.unsqueeze(0).expand(batch // n_w, n_w, seq, seq).reshape(batch, seq, seq).contiguous()


def _forward_backward(
    fn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask_nw: torch.Tensor,
    scale_qk: float,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    q_i = q.detach().clone().requires_grad_(True)
    k_i = k.detach().clone().requires_grad_(True)
    v_i = v.detach().clone().requires_grad_(True)
    b_i = bias.detach().clone().requires_grad_(True)

    out = fn(q_i, k_i, v_i, b_i, mask_nw, scale_qk)
    loss = out.float().square().mean() + 0.1 * out.float().abs().mean()
    loss.backward()

    return out.detach(), (q_i.grad.detach(), k_i.grad.detach(), v_i.grad.detach(), b_i.grad.detach())


def _run_eager(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask_nw: torch.Tensor,
    scale_qk: float,
) -> torch.Tensor:
    return mha_core(q, k, v, bias, mask_nw, scale_qk)


def _run_triton_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask_nw: torch.Tensor,
    scale_qk: float,
) -> torch.Tensor:
    return flash_swin_attn_func_v2(q, k, v, bias, _expand_window_mask(mask_nw, q.size(0)), scale_qk)


def _run_triton_v3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask_nw: torch.Tensor,
    scale_qk: float,
) -> torch.Tensor:
    return flash_swin_attn_func_v3(q, k, v, bias, _expand_window_mask(mask_nw, q.size(0)), scale_qk)


def _run_hybrid_v3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask_nw: torch.Tensor,
    scale_qk: float,
) -> torch.Tensor:
    return hybrid_sdpa_fwd_flash_swin_v3_bwd(q, k, v, bias, _expand_window_mask(mask_nw, q.size(0)), scale_qk)


def _tolerance(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float16:
        return 2e-2, 2e-3
    return 8e-2, 2e-2


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_swin_triton_v3_forward_backward(dtype: torch.dtype) -> None:
    torch.manual_seed(53)

    batch, heads, seq, head_dim = 4, 6, 49, 32
    device = torch.device("cuda")
    scale_qk = head_dim**-0.5

    q = torch.randn((batch, heads, seq, head_dim), device=device, dtype=dtype)
    k = torch.randn((batch, heads, seq, head_dim), device=device, dtype=dtype)
    v = torch.randn((batch, heads, seq, head_dim), device=device, dtype=dtype)
    bias = torch.randn((heads, seq, seq), device=device, dtype=dtype)
    mask_nw = _make_shift_mask(num_windows=2, seq=seq, dtype=dtype, device=device)

    out_eager, grads_eager = _forward_backward(_run_eager, q, k, v, bias, mask_nw, scale_qk)
    out_v3, grads_v3 = _forward_backward(_run_triton_v3, q, k, v, bias, mask_nw, scale_qk)

    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(out_v3.float(), out_eager.float(), rtol=rtol, atol=atol)
    for grad_v3, grad_eager in zip(grads_v3, grads_eager, strict=True):
        torch.testing.assert_close(grad_v3.float(), grad_eager.float(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_swin_triton_v3_matches_v2(dtype: torch.dtype) -> None:
    torch.manual_seed(59)

    batch, heads, seq, head_dim = 4, 4, 49, 32
    device = torch.device("cuda")
    scale_qk = head_dim**-0.5

    q = torch.randn((batch, heads, seq, head_dim), device=device, dtype=dtype)
    k = torch.randn((batch, heads, seq, head_dim), device=device, dtype=dtype)
    v = torch.randn((batch, heads, seq, head_dim), device=device, dtype=dtype)
    bias = torch.randn((heads, seq, seq), device=device, dtype=dtype)
    mask_nw = _make_shift_mask(num_windows=2, seq=seq, dtype=dtype, device=device)

    out_v2, grads_v2 = _forward_backward(_run_triton_v2, q, k, v, bias, mask_nw, scale_qk)
    out_v3, grads_v3 = _forward_backward(_run_triton_v3, q, k, v, bias, mask_nw, scale_qk)

    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(out_v3.float(), out_v2.float(), rtol=rtol, atol=atol)
    for grad_v3, grad_v2 in zip(grads_v3, grads_v2, strict=True):
        torch.testing.assert_close(grad_v3.float(), grad_v2.float(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_swin_hybrid_v3_backward(dtype: torch.dtype) -> None:
    torch.manual_seed(67)

    batch, heads, seq, head_dim = 4, 4, 49, 32
    device = torch.device("cuda")
    scale_qk = head_dim**-0.5

    q = torch.randn((batch, heads, seq, head_dim), device=device, dtype=dtype)
    k = torch.randn((batch, heads, seq, head_dim), device=device, dtype=dtype)
    v = torch.randn((batch, heads, seq, head_dim), device=device, dtype=dtype)
    bias = torch.randn((heads, seq, seq), device=device, dtype=dtype)
    mask_nw = _make_shift_mask(num_windows=2, seq=seq, dtype=dtype, device=device)

    out_eager, grads_eager = _forward_backward(_run_eager, q, k, v, bias, mask_nw, scale_qk)
    out_hybrid, grads_hybrid = _forward_backward(_run_hybrid_v3, q, k, v, bias, mask_nw, scale_qk)

    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(out_hybrid.float(), out_eager.float(), rtol=rtol, atol=atol)
    for grad_hybrid, grad_eager in zip(grads_hybrid, grads_eager, strict=True):
        torch.testing.assert_close(grad_hybrid.float(), grad_eager.float(), rtol=rtol, atol=atol)

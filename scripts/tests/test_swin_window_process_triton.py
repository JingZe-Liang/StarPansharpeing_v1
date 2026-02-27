from __future__ import annotations

import pytest
import torch

from src.stage1.cosmos.third_party.SwinTransformer.kernels.window_process.window_process import (
    WINDOW_PROCESS_BACKEND,
    WindowProcess,
    WindowProcessReverse,
)


def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def _window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


def _fwd_pyt(x: torch.Tensor, shift_size: int, window_size: int) -> torch.Tensor:
    shifted = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2)) if shift_size > 0 else x
    return _window_partition(shifted, window_size)


def _rev_pyt(x: torch.Tensor, shift_size: int, window_size: int, H: int, W: int) -> torch.Tensor:
    shifted = _window_reverse(x, window_size, H, W)
    return torch.roll(shifted, shifts=(shift_size, shift_size), dims=(1, 2)) if shift_size > 0 else shifted


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shift_size", [0, 2])
def test_roll_partition_forward_backward(dtype: torch.dtype, shift_size: int) -> None:
    B, H, W, C = 4, 14, 14, 32
    ws = 7
    x1 = torch.randn((B, H, W, C), device="cuda", dtype=dtype, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)

    y1 = _fwd_pyt(x1, shift_size=shift_size, window_size=ws)
    y2 = WindowProcess.apply(x2, B, H, W, C, -shift_size, ws)
    atol = 1e-3 if dtype == torch.float16 else 1e-5
    rtol = 1e-3 if dtype == torch.float16 else 1e-5
    assert torch.allclose(y1, y2, atol=atol, rtol=rtol)

    grad = torch.randn_like(y1)
    y1.backward(grad)
    y2.backward(grad)
    assert x1.grad is not None and x2.grad is not None
    assert torch.allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shift_size", [0, 2])
def test_merge_roll_forward_backward(dtype: torch.dtype, shift_size: int) -> None:
    B, H, W, C = 4, 14, 14, 32
    ws = 7
    nH, nW = H // ws, W // ws
    x1 = torch.randn((B * nH * nW, ws, ws, C), device="cuda", dtype=dtype, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)

    y1 = _rev_pyt(x1, shift_size=shift_size, window_size=ws, H=H, W=W)
    y2 = WindowProcessReverse.apply(x2, B, H, W, C, shift_size, ws)
    atol = 1e-3 if dtype == torch.float16 else 1e-5
    rtol = 1e-3 if dtype == torch.float16 else 1e-5
    assert torch.allclose(y1, y2, atol=atol, rtol=rtol)

    grad = torch.randn_like(y1)
    y1.backward(grad)
    y2.backward(grad)
    assert x1.grad is not None and x2.grad is not None
    assert torch.allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_backend_resolves() -> None:
    assert WINDOW_PROCESS_BACKEND in {"cuda_ext", "triton"}

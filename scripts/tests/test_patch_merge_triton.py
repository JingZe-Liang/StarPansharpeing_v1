from __future__ import annotations

import pytest
import torch

from src.stage1.cosmos.modules.swin_op.patch_merge.patch_merge_triton import (
    patch_merge_blc_pytorch,
    patch_merge_blc_triton,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("hw_shape", [(8, 8), (7, 9), (17, 19)])
def test_patch_merge_triton_forward_matches_pytorch(dtype: torch.dtype, hw_shape: tuple[int, int]) -> None:
    b, c = 4, 32
    h, w = hw_shape
    x = torch.randn((b, h * w, c), device="cuda", dtype=dtype)

    y_ref, hw_ref = patch_merge_blc_pytorch(x, (h, w))
    y_tri, hw_tri = patch_merge_blc_triton(x, (h, w))

    assert hw_ref == hw_tri
    assert y_ref.shape == y_tri.shape
    assert torch.allclose(y_ref, y_tri, atol=1e-3 if dtype == torch.float16 else 1e-6, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("hw_shape", [(8, 8), (7, 9), (17, 19)])
def test_patch_merge_triton_backward_matches_pytorch(dtype: torch.dtype, hw_shape: tuple[int, int]) -> None:
    b, c = 2, 64
    h, w = hw_shape
    x_ref = torch.randn((b, h * w, c), device="cuda", dtype=dtype, requires_grad=True)
    x_tri = x_ref.detach().clone().requires_grad_(True)

    y_ref, _ = patch_merge_blc_pytorch(x_ref, (h, w))
    y_tri, _ = patch_merge_blc_triton(x_tri, (h, w))

    grad = torch.randn_like(y_ref)
    y_ref.backward(grad)
    y_tri.backward(grad)

    assert x_ref.grad is not None and x_tri.grad is not None
    assert torch.allclose(x_ref.grad, x_tri.grad, atol=1e-3 if dtype == torch.float16 else 1e-6, rtol=0.0)

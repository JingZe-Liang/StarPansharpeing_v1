import importlib.util
from pathlib import Path

import pytest
import torch


def _load_triton_rmsnorm2d_func() -> object | None:
    module_path = Path(__file__).resolve().parents[2] / "src/stage1/cosmos/modules/rmsnorm_triton.py"
    spec = importlib.util.spec_from_file_location("test_rmsnorm_triton_module", module_path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return None
    return getattr(module, "TritonRMSNorm2dFunc", None)


TritonRMSNorm2dFunc = _load_triton_rmsnorm2d_func()


def _reference_rmsnorm2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, eps: float) -> torch.Tensor:
    y = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=1, keepdim=True) + eps)
    y = y * weight.view(1, -1, 1, 1)
    if bias is not None:
        y = y + bias.view(1, -1, 1, 1)
    return y


@pytest.mark.skipif(
    not torch.cuda.is_available() or TritonRMSNorm2dFunc is None,
    reason="CUDA and Triton are required",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("use_bias", [False, True])
def test_triton_rmsnorm2d_func_matches_reference_with_optional_bias(dtype: torch.dtype, use_bias: bool) -> None:
    torch.manual_seed(0)
    b, c, h, w = 2, 32, 8, 8
    eps = 1e-5
    forward_atol = 2.5e-3 if dtype == torch.float16 else 1e-5
    grad_atol = 8e-3 if dtype == torch.float16 else 1e-5
    rtol = 1e-3 if dtype == torch.float16 else 1e-5

    weight_ref = torch.randn((c,), device="cuda", dtype=torch.float32, requires_grad=True)
    weight_tri = weight_ref.detach().clone().requires_grad_(True)

    bias_ref = torch.randn((c,), device="cuda", dtype=torch.float32, requires_grad=True) if use_bias else None
    bias_tri = bias_ref.detach().clone().requires_grad_(True) if bias_ref is not None else None

    x_ref = torch.randn((b, c, h, w), device="cuda", dtype=dtype, requires_grad=True)
    x_tri = x_ref.detach().clone().requires_grad_(True)

    y_ref = _reference_rmsnorm2d(x_ref, weight_ref, bias_ref, eps)
    y_tri = TritonRMSNorm2dFunc.apply(x_tri, weight_tri, bias_tri, eps)

    assert y_ref.shape == y_tri.shape
    assert torch.allclose(y_ref.float(), y_tri.float(), atol=forward_atol, rtol=rtol)

    grad = torch.randn_like(y_ref)
    y_ref.backward(grad)
    y_tri.backward(grad.to(y_tri.dtype))

    assert x_ref.grad is not None
    assert x_tri.grad is not None
    assert torch.allclose(x_ref.grad, x_tri.grad, atol=grad_atol, rtol=rtol)

    assert weight_ref.grad is not None
    assert weight_tri.grad is not None
    assert torch.allclose(weight_ref.grad, weight_tri.grad, atol=grad_atol, rtol=rtol)

    if use_bias:
        assert bias_ref is not None
        assert bias_tri is not None
        assert bias_ref.grad is not None
        assert bias_tri.grad is not None
        assert torch.allclose(bias_ref.grad, bias_tri.grad, atol=grad_atol, rtol=rtol)

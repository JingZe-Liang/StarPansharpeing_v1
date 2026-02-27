from __future__ import annotations

import pytest
import torch

from src.stage1.cosmos.modules.swin_op.swin_transformer import Mlp, PatchMerging, SwinTransformerBlock
from src.stage1.cosmos.modules.variants.mlp import SwiGLU


def test_swin_block_out_dim_and_output_2d() -> None:
    torch.manual_seed(0)
    b, h, w, c = 2, 8, 8, 64
    out_dim = 96
    block = SwinTransformerBlock(
        dim=c,
        out_dim=out_dim,
        input_resolution=(h, w),
        num_heads=8,
        window_size=4,
        shift_size=2,
        is_flash=False,
        attn_backend="py",
        window_backend="py",
        output_2d=True,
    )
    x = torch.randn((b, h * w, c), dtype=torch.float32, requires_grad=True)
    y = block(x)
    assert y.shape == (b, out_dim, h, w)
    y.mean().backward()
    assert x.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("mlp_cls", "mlp_kwargs"),
    [
        (Mlp, None),
        (SwiGLU, {"is_fused": None, "use_conv": False}),
    ],
)
def test_swin_block_py_vs_triton_forward_backward(
    mlp_cls: type[torch.nn.Module], mlp_kwargs: dict[str, object] | None
) -> None:
    torch.manual_seed(0)
    b, h, w, c = 4, 14, 14, 96
    heads = 6
    ws = 7
    shift = 3

    block_py = SwinTransformerBlock(
        dim=c,
        input_resolution=(h, w),
        num_heads=heads,
        window_size=ws,
        shift_size=shift,
        is_flash=False,
        attn_backend="py",
        window_backend="py",
        mlp_cls=mlp_cls,
        mlp_kwargs=mlp_kwargs,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
    ).cuda()
    block_py = block_py
    block_triton = SwinTransformerBlock(
        dim=c,
        input_resolution=(h, w),
        num_heads=heads,
        window_size=ws,
        shift_size=shift,
        is_flash=True,
        attn_backend="triton_v3",
        window_backend="py",
        mlp_cls=mlp_cls,
        mlp_kwargs=mlp_kwargs,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
    ).cuda()
    block_triton = block_triton
    block_triton.load_state_dict(block_py.state_dict(), strict=True)

    # block_py.eval()
    # block_triton.eval()

    dtype = torch.float32

    x_ref = torch.randn((b, h * w, c), device="cuda", dtype=dtype, requires_grad=True)
    x_tri = x_ref.detach().clone().requires_grad_(True)

    y_ref = block_py(x_ref)
    y_tri = block_triton(x_tri)

    assert y_ref.shape == y_tri.shape
    out_diff = (y_ref.float() - y_tri.float()).abs().detach()
    out_mae = float(out_diff.mean().item())
    out_max = float(out_diff.max().item())

    grad = torch.randn_like(y_ref)
    y_ref.backward(grad)
    y_tri.backward(grad)
    assert x_ref.grad is not None and x_tri.grad is not None
    grad_diff = (x_ref.grad.float() - x_tri.grad.float()).abs().detach()
    grad_mae = float(grad_diff.mean().item())
    grad_max = float(grad_diff.max().item())

    name = mlp_cls.__name__
    print(
        f"[swin-block-diff] {name} out_mae={out_mae:.6e} out_max={out_max:.6e} "
        f"grad_mae={grad_mae:.6e} grad_max={grad_max:.6e}"
    )

    assert out_mae < 2.0e-4
    assert out_max < 1.0e-2
    assert grad_mae < 2.0e-4
    assert grad_max < 1.0e-2
    y_ref = block_py(x_ref)
    y_tri = block_triton(x_tri)

    assert y_ref.shape == y_tri.shape
    out_diff = (y_ref.float() - y_tri.float()).abs().detach()
    out_mae = float(out_diff.mean().item())
    out_max = float(out_diff.max().item())

    grad = torch.randn_like(y_ref)
    y_ref.backward(grad)
    y_tri.backward(grad)
    assert x_ref.grad is not None and x_tri.grad is not None
    grad_diff = (x_ref.grad.float() - x_tri.grad.float()).abs().detach()
    grad_mae = float(grad_diff.mean().item())
    grad_max = float(grad_diff.max().item())

    name = mlp_cls.__name__
    print(
        f"[swin-block-diff] {name} out_mae={out_mae:.6e} out_max={out_max:.6e} "
        f"grad_mae={grad_mae:.6e} grad_max={grad_max:.6e}"
    )

    assert out_mae < 2.0e-4
    assert out_max < 1.0e-2
    assert grad_mae < 2.0e-4
    assert grad_max < 1.0e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_patch_merge_class_restored_and_runs() -> None:
    torch.manual_seed(0)
    b, h, w, c = 2, 7, 9, 32
    x = torch.randn((b, h * w, c), device="cuda", dtype=torch.float16, requires_grad=True)
    merge = PatchMerging(input_resolution=(h, w), dim=c, merge_backend="triton").cuda()
    merge = merge.half()
    y = merge(x)
    assert y.shape == (b, ((h + 1) // 2) * ((w + 1) // 2), c * 2)
    y.float().square().mean().backward()
    assert x.grad is not None

import importlib.util
import time
from pathlib import Path

import pytest
import torch
from timm.layers.pos_embed_sincos import apply_rot_embed_cat, build_fourier_pos_embed


def _load_rope_bhsd_module():
    module_path = Path(__file__).resolve().parents[2] / "src/stage1/cosmos/modules/rope/kernels/rope_bhsd.py"
    spec = importlib.util.spec_from_file_location("test_rope_bhsd_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rope_bhsd = _load_rope_bhsd_module()


def _dtype_cases() -> list[torch.dtype]:
    dtypes = [torch.float32, torch.float16]
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtypes.append(torch.bfloat16)
    return dtypes


def _build_timm_raw_sin_cos(seqlen: int, half_dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    sin_half, cos_half = build_fourier_pos_embed(
        [seqlen],
        num_bands=half_dim,
        in_pixels=False,
        linear_bands=False,
        device=device,
        dtype=torch.float32,
    )
    return sin_half.reshape(seqlen, half_dim), cos_half.reshape(seqlen, half_dim)


def _expand_for_timm(
    sin_half: torch.Tensor, cos_half: torch.Tensor, timm_half: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    if timm_half:
        sin_full = torch.cat([sin_half, sin_half], dim=-1)
        cos_full = torch.cat([cos_half, cos_half], dim=-1)
    else:
        sin_full = sin_half.repeat_interleave(2, dim=-1)
        cos_full = cos_half.repeat_interleave(2, dim=-1)
    return sin_full, cos_full


def _build_test_tables(
    seqlen_ro: int,
    half_dim: int,
    batch_size: int,
    batch_specific: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    base_len = seqlen_ro + batch_size + 3
    sin_base, cos_base = _build_timm_raw_sin_cos(base_len, half_dim, device=device)
    if not batch_specific:
        return cos_base[:seqlen_ro].contiguous(), sin_base[:seqlen_ro].contiguous()

    positions = (
        torch.arange(seqlen_ro, device=device, dtype=torch.long)[None, :]
        + torch.arange(batch_size, device=device, dtype=torch.long)[:, None]
    )
    gather_index = positions.unsqueeze(-1).expand(-1, -1, half_dim)
    cos = torch.gather(cos_base.unsqueeze(0).expand(batch_size, -1, -1), dim=1, index=gather_index)
    sin = torch.gather(sin_base.unsqueeze(0).expand(batch_size, -1, -1), dim=1, index=gather_index)
    return cos.contiguous(), sin.contiguous()


def _resolve_positions(seqlen: int, batch_size: int, device: torch.device, offsets: int | torch.Tensor) -> torch.Tensor:
    base = torch.arange(seqlen, device=device, dtype=torch.long)
    if isinstance(offsets, int):
        return base.unsqueeze(0).expand(batch_size, -1) + offsets
    return base.unsqueeze(0) + offsets.to(device=device, dtype=torch.long).unsqueeze(1)


def _gather_aligned_half(
    table: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    if table.ndim == 4:
        if table.shape[1] != 1:
            raise ValueError(f"Expected table.shape[1] == 1 for 4D input, got {table.shape}")
        table = table.squeeze(1)
    if table.ndim == 2:
        expanded = table.unsqueeze(0).expand(positions.shape[0], -1, -1)
    else:
        expanded = table
    gather_index = positions.unsqueeze(-1).expand(-1, -1, expanded.shape[-1])
    return torch.gather(expanded, dim=1, index=gather_index)


def _apply_timm_reference(
    x: torch.Tensor,
    cos_half: torch.Tensor,
    sin_half: torch.Tensor,
    interleaved: bool,
    rotary_dim: int | None = None,
    seqlen_offsets: int | torch.Tensor = 0,
) -> torch.Tensor:
    if rotary_dim is None:
        rotary_dim = x.shape[-1]
    batch = x.shape[0]
    seqlen = x.shape[2]
    positions = _resolve_positions(seqlen, batch, x.device, seqlen_offsets)
    timm_half = not interleaved
    cos_aligned_half = _gather_aligned_half(cos_half, positions)
    sin_aligned_half = _gather_aligned_half(sin_half, positions)
    sin_full, cos_full = _expand_for_timm(sin_aligned_half, cos_aligned_half, timm_half=timm_half)
    rope = torch.cat([sin_full, cos_full], dim=-1)[:, None, :, :]
    x_rot = apply_rot_embed_cat(x[..., :rotary_dim], rope, half=timm_half)
    if rotary_dim == x.shape[-1]:
        return x_rot
    return torch.cat([x_rot, x[..., rotary_dim:]], dim=-1)


def _benchmark_ms(fn, warmup: int = 5, iters: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iters


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", _dtype_cases())
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("rotary_ratio", [1.0, 0.5])
@pytest.mark.parametrize("batch_specific", [False, True])
def test_apply_rope_bhsd_matches_timm_reference(
    dtype: torch.dtype, interleaved: bool, rotary_ratio: float, batch_specific: bool
) -> None:
    torch.manual_seed(0)
    b, h, s, d = 2, 4, 13, 32
    rotary_dim = int(d * rotary_ratio)
    half_dim = rotary_dim // 2
    seqlen_ro = s + 4
    atol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    rtol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5

    x_ref = torch.randn((b, h, s, d), device="cuda", dtype=dtype, requires_grad=True)
    x_tri = x_ref.detach().clone().requires_grad_(True)
    seqlen_offsets = torch.tensor([1, 3], device="cuda", dtype=torch.int32)
    cos, sin = _build_test_tables(seqlen_ro, half_dim, batch_size=b, batch_specific=batch_specific, device=x_ref.device)

    y_ref = _apply_timm_reference(
        x_ref,
        cos_half=cos,
        sin_half=sin,
        rotary_dim=rotary_dim,
        interleaved=interleaved,
        seqlen_offsets=seqlen_offsets,
    )
    y_tri = rope_bhsd.apply_rope_bhsd(
        x_tri,
        cos=cos,
        sin=sin,
        rotary_dim=rotary_dim,
        interleaved=interleaved,
        seqlen_offsets=seqlen_offsets,
    )

    assert y_ref.shape == y_tri.shape
    assert torch.allclose(y_ref.float(), y_tri.float(), atol=atol, rtol=rtol)

    grad = torch.randn_like(y_ref)
    y_ref.backward(grad)
    y_tri.backward(grad.to(y_tri.dtype))

    assert x_ref.grad is not None
    assert x_tri.grad is not None
    assert torch.allclose(x_ref.grad.float(), x_tri.grad.float(), atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", _dtype_cases())
def test_apply_rope_bhsd_normalizes_batched_4d_cos_sin(dtype: torch.dtype) -> None:
    torch.manual_seed(1)
    b, h, s, d = 2, 3, 11, 16
    half_dim = d // 2
    atol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    rtol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5

    cos_half, sin_half = _build_test_tables(s, half_dim, batch_size=b, batch_specific=True, device=torch.device("cuda"))
    x = torch.randn((b, h, s, d), device="cuda", dtype=dtype)
    cos = cos_half.unsqueeze(1)
    sin = sin_half.unsqueeze(1)

    y_ref = _apply_timm_reference(x, cos_half=cos, sin_half=sin, interleaved=False)
    y_tri = rope_bhsd.apply_rope_bhsd(x, cos=cos, sin=sin)

    assert torch.allclose(y_ref.float(), y_tri.float(), atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", _dtype_cases())
@pytest.mark.parametrize("interleaved", [False, True])
def test_apply_rope_qk_bhsd_matches_timm_reference_with_prefix(dtype: torch.dtype, interleaved: bool) -> None:
    torch.manual_seed(2)
    b, h, s, d = 2, 4, 12, 32
    prefix = 3
    rotary_dim = 24
    half_dim = rotary_dim // 2
    seqlen_ro = s - prefix + 5
    atol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    rtol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5

    q_ref = torch.randn((b, h, s, d), device="cuda", dtype=dtype, requires_grad=True)
    k_ref = torch.randn((b, h, s, d), device="cuda", dtype=dtype, requires_grad=True)
    q_tri = q_ref.detach().clone().requires_grad_(True)
    k_tri = k_ref.detach().clone().requires_grad_(True)

    cos, sin = _build_test_tables(seqlen_ro, half_dim, batch_size=b, batch_specific=True, device=q_ref.device)
    offsets = torch.tensor([1, 2], device="cuda", dtype=torch.int32)

    q_ref_out = torch.cat(
        [
            q_ref[:, :, :prefix, :],
            _apply_timm_reference(
                q_ref[:, :, prefix:, :],
                cos_half=cos,
                sin_half=sin,
                rotary_dim=rotary_dim,
                interleaved=interleaved,
                seqlen_offsets=offsets,
            ),
        ],
        dim=2,
    )
    k_ref_out = torch.cat(
        [
            k_ref[:, :, :prefix, :],
            _apply_timm_reference(
                k_ref[:, :, prefix:, :],
                cos_half=cos,
                sin_half=sin,
                rotary_dim=rotary_dim,
                interleaved=interleaved,
                seqlen_offsets=offsets,
            ),
        ],
        dim=2,
    )

    q_tri_out, k_tri_out = rope_bhsd.apply_rope_qk_bhsd(
        q_tri,
        k_tri,
        cos=cos.unsqueeze(1),
        sin=sin.unsqueeze(1),
        rotary_dim=rotary_dim,
        interleaved=interleaved,
        num_prefix_tokens=prefix,
        seqlen_offsets=offsets,
    )

    assert torch.allclose(q_ref_out.float(), q_tri_out.float(), atol=atol, rtol=rtol)
    assert torch.allclose(k_ref_out.float(), k_tri_out.float(), atol=atol, rtol=rtol)
    assert torch.equal(q_ref_out[:, :, :prefix, :], q_tri_out[:, :, :prefix, :])
    assert torch.equal(k_ref_out[:, :, :prefix, :], k_tri_out[:, :, :prefix, :])

    grad_q = torch.randn_like(q_ref_out)
    grad_k = torch.randn_like(k_ref_out)
    q_ref_out.backward(grad_q, retain_graph=True)
    k_ref_out.backward(grad_k)
    q_tri_out.backward(grad_q.to(q_tri_out.dtype), retain_graph=True)
    k_tri_out.backward(grad_k.to(k_tri_out.dtype))

    assert q_ref.grad is not None
    assert q_tri.grad is not None
    assert k_ref.grad is not None
    assert k_tri.grad is not None
    assert torch.allclose(q_ref.grad.float(), q_tri.grad.float(), atol=atol, rtol=rtol)
    assert torch.allclose(k_ref.grad.float(), k_tri.grad.float(), atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_apply_rope_bhsd_large_seq_benchmark() -> None:
    torch.manual_seed(3)
    b, h, s, d = 2, 8, 4096, 64
    rotary_dim = 64
    half_dim = rotary_dim // 2
    dtype = torch.float16
    cos, sin = _build_test_tables(s, half_dim, batch_size=b, batch_specific=False, device=torch.device("cuda"))

    x_timm = torch.randn((b, h, s, d), device="cuda", dtype=dtype)
    x_triton = x_timm.detach().clone()
    grad_timm = torch.randn((b, h, s, d), device="cuda", dtype=dtype)
    grad_triton = grad_timm.detach().clone()

    def run_timm_forward() -> None:
        _apply_timm_reference(x_timm, cos_half=cos, sin_half=sin, rotary_dim=rotary_dim, interleaved=False)

    def run_triton_forward() -> None:
        rope_bhsd.apply_rope_bhsd(x_triton, cos=cos, sin=sin, rotary_dim=rotary_dim, interleaved=False)

    def run_timm_forward_backward() -> None:
        x = x_timm.detach().requires_grad_(True)
        y = _apply_timm_reference(x, cos_half=cos, sin_half=sin, rotary_dim=rotary_dim, interleaved=False)
        y.backward(grad_timm)

    def run_triton_forward_backward() -> None:
        x = x_triton.detach().requires_grad_(True)
        y = rope_bhsd.apply_rope_bhsd(x, cos=cos, sin=sin, rotary_dim=rotary_dim, interleaved=False)
        y.backward(grad_triton)

    timm_fwd_ms = _benchmark_ms(run_timm_forward, warmup=5, iters=30)
    triton_fwd_ms = _benchmark_ms(run_triton_forward, warmup=5, iters=30)
    timm_ms = _benchmark_ms(run_timm_forward_backward, warmup=3, iters=10)
    triton_ms = _benchmark_ms(run_triton_forward_backward, warmup=3, iters=10)

    print(
        f"[rope-bhsd-benchmark] shape=({b},{h},{s},{d}) dtype={dtype} "
        f"timm_fwd_ms={timm_fwd_ms:.3f} triton_fwd_ms={triton_fwd_ms:.3f} fwd_speedup={timm_fwd_ms / triton_fwd_ms:.3f}x "
        f"timm_fwd_bwd_ms={timm_ms:.3f} triton_fwd_bwd_ms={triton_ms:.3f} fwd_bwd_speedup={timm_ms / triton_ms:.3f}x"
    )

    assert timm_fwd_ms > 0
    assert triton_fwd_ms > 0
    assert timm_ms > 0
    assert triton_ms > 0

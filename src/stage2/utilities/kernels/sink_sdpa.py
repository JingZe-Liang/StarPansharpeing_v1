import math
import torch

from typing import Optional, Tuple


# Reference implementation
def sdpa_with_sink(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    logits = torch.matmul(q, k.transpose(-2, -1)) / (D**0.5)  # [B, H, S, S]
    exp_logits = torch.exp(logits)  # [B, H, S, S]
    mass = exp_logits.sum(dim=-1, keepdim=True) + s.exp()[:, None, None]  # [B, H, S, 1]
    a = exp_logits / mass  # [B, H, S, S]
    return a @ v  # [B, H, S, D]


class AttentionSinkAtHome(torch.autograd.Function):
    @staticmethod
    @torch.compile
    def alter_out_with_lse_grad(dOut, dLSE, out, eps: float = 1e-12):
        dOut_f, dLSE_f, out_f = (
            dOut.to(torch.float32),
            dLSE.to(torch.float32),
            out.to(torch.float32),
        )
        denom = dOut_f.square().sum(-1, keepdim=True) + eps
        return (out_f - dLSE_f[..., None] * dOut_f / denom).to(out.dtype)

    @staticmethod
    def flash_attention_forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (out, lse, cum_q, cum_k, max_q, max_k, seed, offset, _) = torch.ops.aten._scaled_dot_product_flash_attention(
            q, k, v, ctx.dropout_p, ctx.is_causal, False, scale=ctx.scale
        )
        # Save for backward
        ctx.save_for_backward(q, k, v, out, lse, cum_q, cum_k, seed, offset)
        ctx.max_q = int(max_q)
        ctx.max_k = int(max_k)
        return out, lse

    @staticmethod
    def flash_attention_backward(
        ctx, dOut: torch.Tensor, dLSE: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v, out, lse, cum_q, cum_k, seed, offset = ctx.saved_tensors
        out = AttentionSinkAtHome.alter_out_with_lse_grad(dOut, dLSE, out, ctx.eps)
        dQ, dK, dV = torch.ops.aten._scaled_dot_product_flash_attention_backward(
            dOut,
            q,
            k,
            v,
            out,
            lse,
            cum_q,
            cum_k,
            ctx.max_q,
            ctx.max_k,
            ctx.dropout_p,
            ctx.is_causal,
            seed,
            offset,
        )
        return dQ, dK, dV

    @staticmethod
    def cudnn_attention_forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (out, lse, cum_q, cum_k, max_q, max_k, philox_seed, philox_offset, _) = (
            torch.ops.aten._scaled_dot_product_cudnn_attention(
                q,
                k,
                v,
                None,
                True,
                ctx.dropout_p,
                ctx.is_causal,
                False,
                scale=ctx.scale,
            )
        )
        # Save for backward
        ctx.save_for_backward(q, k, v, out, lse, cum_q, cum_k, philox_seed, philox_offset)
        ctx.max_q = int(max_q)
        ctx.max_k = int(max_k)
        return out, lse

    @staticmethod
    def cudnn_attention_backward(
        ctx, dOut: torch.Tensor, dLSE: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v, out, lse, cum_q, cum_k, seed, offset = ctx.saved_tensors
        out = AttentionSinkAtHome.alter_out_with_lse_grad(dOut, dLSE, out, ctx.eps)
        dQ, dK, dV = torch.ops.aten._scaled_dot_product_cudnn_attention_backward(
            dOut,
            q,
            k,
            v,
            out,
            lse,
            seed,
            offset,
            None,
            cum_q,
            cum_k,
            ctx.max_q,
            ctx.max_k,
            ctx.dropout_p,
            ctx.is_causal,
            scale=ctx.scale,
        )
        return dQ, dK, dV

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        eps: float = 1e-12,
        backend: str = "flash",  # flash, cudnn
    ):
        assert q.is_cuda and k.is_cuda and v.is_cuda, "CUDA tensors required"
        if scale is None:
            scale = 1.0 / math.sqrt(q.size(-1))
        # Save params
        ctx.dropout_p = float(dropout_p)
        ctx.is_causal = bool(is_causal)
        ctx.scale = float(scale)
        ctx.eps = float(eps)
        # Call internal torch kernels
        match backend:
            case "flash":
                out, lse = AttentionSinkAtHome.flash_attention_forward(ctx, q, k, v)
            case "cudnn":
                out, lse = AttentionSinkAtHome.cudnn_attention_forward(ctx, q, k, v)
        ctx.backend = backend
        return out, lse

    @staticmethod
    def backward(ctx, dOut, dLSE):
        if dOut is None:
            raise RuntimeError("This wrapper requires grad wrt 'out' (dOut) for LSE grads injection.")
        if dLSE is None:
            dLSE = torch.zeros_like(ctx.saved_tensors[4])  # like lse

        match ctx.backend:
            case "flash":
                dQ, dK, dV = AttentionSinkAtHome.flash_attention_backward(ctx, dOut, dLSE)
            case "cudnn":
                dQ, dK, dV = AttentionSinkAtHome.cudnn_attention_backward(ctx, dOut, dLSE)

        return dQ, dK, dV, None, None, None, None, None


def attention_sink_at_home(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: Optional[torch.Tensor] = None,
    *,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    eps: float = 1e-12,
    backend: str = "flash",
) -> torch.Tensor:
    o, lse = AttentionSinkAtHome.apply(q, k, v, float(dropout_p), bool(is_causal), scale, eps, backend)
    if s is not None:
        o = (torch.sigmoid(lse[..., None] - s[None, :, None, None]) * o.to(torch.float32)).to(
            o.dtype
        )  # By @YouJiacheng: https://x.com/YouJiacheng/status/1957764436623847807
    return o


B, H, S, D = 32, 4, 256, 128
q = torch.randn(B, H, S, D).cuda().bfloat16()
k = torch.randn(B, H, S, D).cuda().bfloat16()
v = torch.randn(B, H, S, D).cuda().bfloat16()
s = torch.randn(H).cuda().bfloat16()
out_grad = torch.randn_like(v)
rg = lambda t: t.detach().clone().requires_grad_(True)
atol, rtol = torch.finfo(torch.bfloat16).eps * 2, 3e-2

q1, k1, v1 = rg(q), rg(k), rg(v)
vanilla_sdpa_torch = torch.nn.functional.scaled_dot_product_attention(q1, k1, v1)
vanilla_sdpa_torch.backward(out_grad)

q2, k2, v2 = rg(q), rg(k), rg(v)
vanilla_sdpa_at_home = attention_sink_at_home(q2, k2, v2)
vanilla_sdpa_at_home.backward(out_grad)

torch.testing.assert_close(vanilla_sdpa_torch, vanilla_sdpa_at_home, atol=atol, rtol=rtol)
torch.testing.assert_close(q2.grad, q1.grad, atol=atol, rtol=rtol)
torch.testing.assert_close(k2.grad, k1.grad, atol=atol, rtol=rtol)
torch.testing.assert_close(v2.grad, v1.grad, atol=atol, rtol=rtol)

q1, k1, v1, s1 = rg(q), rg(k), rg(v), rg(s)
sink_sdpa_baseline = sdpa_with_sink(q1, k1, v1, s1)
sink_sdpa_baseline.backward(out_grad)

q2, k2, v2, s2 = rg(q), rg(k), rg(v), rg(s)
sink_sdpa_at_home = attention_sink_at_home(q2, k2, v2, s2)
sink_sdpa_at_home.backward(out_grad)

torch.testing.assert_close(sink_sdpa_baseline, sink_sdpa_at_home, atol=atol, rtol=rtol)
torch.testing.assert_close(q2.grad, q1.grad, atol=atol, rtol=rtol)
torch.testing.assert_close(k2.grad, k1.grad, atol=atol, rtol=rtol)
torch.testing.assert_close(v2.grad, v1.grad, atol=atol, rtol=rtol)

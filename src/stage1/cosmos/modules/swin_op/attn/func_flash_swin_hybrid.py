from __future__ import annotations

import torch
import torch.nn.functional as F

from .func_flash_swin_v3 import flash_swin_attn_bwd_func_v3


class HybridSDPAForwardFlashSwinV3Backward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: torch.Tensor,
        window_mask: torch.Tensor | None,
        scale_qk: float,
    ) -> torch.Tensor:
        attn_bias = bias.unsqueeze(0)
        if window_mask is not None:
            attn_bias = attn_bias + window_mask.unsqueeze(1)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=0.0, scale=scale_qk)
        ctx.save_for_backward(q, k, v, bias)
        ctx.window_mask = window_mask
        ctx.scale_qk = scale_qk
        return out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        d_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        q, k, v, bias = ctx.saved_tensors
        d_q, d_k, d_v, d_bias = flash_swin_attn_bwd_func_v3(
            q=q,
            k=k,
            v=v,
            bias=bias,
            window_mask=ctx.window_mask,
            d_out=d_out,
            scale_qk=ctx.scale_qk,
        )
        if d_bias is None:
            d_bias = torch.zeros_like(bias)
        return d_q, d_k, d_v, d_bias, None, None


hybrid_sdpa_fwd_flash_swin_v3_bwd = HybridSDPAForwardFlashSwinV3Backward.apply

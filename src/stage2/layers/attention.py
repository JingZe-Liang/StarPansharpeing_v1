from typing import Callable

import lazy_loader
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Array, Float
from timm.layers.attention import Attention as Attention_
from torch import Tensor

natten = lazy_loader.load("natten")


def _float16_clip_value(x):
    if torch.get_autocast_dtype("cuda") == torch.float16:
        x = x.clip(-65504, 65504)

    return x


class NatAttention(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size=8,
        stride=2,
        dilation=2,
        num_heads=8,
        qkv_bias=True,
        qk_norm: type[nn.Module] | None = None,
        norm_layer=None,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        scale_norm: bool = False,
        torch_compile=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.torch_compile = torch_compile
        self.is_causal = False

        self.qkv = nn.Conv2d(dim, dim * 3, 3, 1, 1, groups=dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = (
            norm_layer(dim) if scale_norm and norm_layer is not None else nn.Identity()
        )
        self.proj = nn.Conv2d(dim, dim, 1, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        if qk_norm is not None:
            self.q_norm = qk_norm(self.head_dim)
            self.k_norm = qk_norm(self.head_dim)
        else:
            self.q_norm = self.k_norm = nn.Identity()

    def _apply_rope(self, q: Float[Tensor, "b h w nh hd"], k, rope):
        if rope:
            h, w = q.shape[1:3]
            q = rearrange(q, "b h w nh hd -> b (h w) nh hd")
            k = rearrange(k, "b h w nh hd -> b (h w) nh hd")

            q, k = rope(q, k)

            q = rearrange(q, "b (h w) nh hd -> b h w nh hd", h=h, w=w)
            k = rearrange(k, "b (h w) nh hd -> b h w nh hd", h=h, w=w)

        return q, k

    def forward(self, x: Float[Tensor, "b c h w"], mask=None, rope=None):
        B, C, H, W = x.shape

        qkv = self.qkv(x)
        qkv = rearrange(qkv, "b (qkv c) h w -> b qkv c h w", qkv=3)
        q, k, v = qkv.unbind(1)
        dtype = q.dtype

        q = self.q_norm(q)  # (bs, c, h, w)
        k = self.k_norm(k)

        q = rearrange(q, "b (nh hd) h w -> b h w nh hd", nh=self.num_heads)
        k = rearrange(k, "b (nh hd) h w -> b h w nh hd", nh=self.num_heads)
        v = rearrange(v, "b (nh hd) h w -> b h w nh hd", nh=self.num_heads)

        q, k = self._apply_rope(q, k, rope)

        x = natten.na2d(  # (bs, h, w, nh, hd)  # type: ignore
            q,
            k,
            v,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            is_causal=False,
            torch_compile=self.torch_compile,
        )

        x = rearrange(x, "b h w nh hd -> b (nh hd) h w")
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = _float16_clip_value(x)

        return x


class Attention(Attention_):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        qk_norm: type[nn.Module] | None = None,
        **block_kwargs,
    ):
        super().__init__(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            **block_kwargs,
        )

        if qk_norm:
            self.q_norm = qk_norm(dim)
            self.k_norm = qk_norm(dim)
        else:
            self.q_norm = self.k_norm = nn.Identity()

    def forward(self, x, mask=None, rope: Callable | None = None):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, C)
        q, k, v = qkv.unbind(2)
        dtype = q.dtype

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.reshape(B, N, self.num_heads, C // self.num_heads).to(dtype)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).to(dtype)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).to(dtype)

        # RoPE
        if rope is not None:
            q, k = rope(q, k)

        use_fp32_attention = getattr(
            self, "fp32_attention", False
        )  # necessary for NAN loss
        if use_fp32_attention:
            q, k, v = q.float(), k.float(), v.float()

        # sdpa
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None and mask.ndim == 2:
            mask = (1 - mask.to(x.dtype)) * -10000.0
            mask = mask[:, None, None].repeat(1, self.num_heads, 1, 1)

        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
        )
        x = x.transpose(1, 2)

        x = x.view(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = _float16_clip_value(x)

        return x

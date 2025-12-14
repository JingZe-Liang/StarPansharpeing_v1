# --------------------------------------------------------
# ViT^3: Unlocking Test-Time Training in Vision
# Written by Dongchen Han
# --------------------------------------------------------

import math

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from diffusers.models.embeddings import get_1d_rotary_pos_embed, get_2d_rotary_pos_embed, apply_rotary_emb
from einops import rearrange
from timm.models.layers import trunc_normal_
from timm.layers.mlp import SwiGLU

from flash_attn import flash_attn_func


class TTT2DWithoutAttention(nn.Module):
    r"""Test-Time Training block for ViT^3 model.
        - https://arxiv.org/abs/2512.01643

    This block implements test-time inner training of two parallel sub-modules:
        1. Simplified SwiGLU inner module, i.e., SwiGLU with identity output layer
        2. 3x3 depth-wise convolution (3x3dwc) inner module

    Note:
        The TTT inner loss is a per-head / per-sample vector-valued loss (shape [B, num_heads]).
        The torch.autograd.backward only supports scalar losses, so here we implement a hand-derived
        backward (closed-form gradient expressions) that directly computes parameter gradients.
        Alternative efficient implementations are welcome and appreciated.

    Args:
        dim (int): Number of input channels
        num_heads (int): Number of attention heads
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, num_heads, qkv_bias=True, **kwargs):
        super().__init__()
        head_dim = dim // num_heads
        self.dim = dim
        self.num_heads = num_heads

        self.qkv = nn.Linear(dim, dim * 3 + head_dim * 3, bias=qkv_bias)
        self.w1 = nn.Parameter(torch.zeros(1, self.num_heads, head_dim, head_dim))
        self.w2 = nn.Parameter(torch.zeros(1, self.num_heads, head_dim, head_dim))
        self.w3 = nn.Parameter(torch.zeros(head_dim, 1, 3, 3))
        trunc_normal_(self.w1, std=0.02)
        trunc_normal_(self.w2, std=0.02)
        trunc_normal_(self.w3, std=0.02)
        self.proj = nn.Linear(dim + head_dim, dim)

        equivalent_head_dim = 9
        self.scale = equivalent_head_dim**-0.5
        # The equivalent head_dim of 3x3dwc branch is 1x(3x3)=9 (1 channel, 3x3 kernel)
        # We used this equivalent_head_dim to compute self.scale in our earlier experiments
        # Using self.scale=head_dim**-0.5 (head_dim of simplified SwiGLU branch) leads to similar performance

    def inner_train_simplified_swiglu(self, k, v, w1, w2, lr=1.0):
        """
        Args:
            k (torch.Tensor): Key tensor of shape [B, num_heads, N, head_dim]
            v (torch.Tensor): Value tensor of shape [B, num_heads, N, head_dim]
            w1 (torch.Tensor): First weight matrix of shape [1, num_heads, head_dim, head_dim]
            w2 (torch.Tensor): Second weight matrix of shape [1, num_heads, head_dim, head_dim]
            lr (float, optional): Learning rate for inner-loop update. Default: 1.0

        Returns:
            tuple: Updated w1 and w2
        """
        # --- Forward ---
        z1 = k @ w1
        z2 = k @ w2
        sig = F.sigmoid(z2)
        a = z2 * sig
        # v_hat = a
        # l = (v_hat * v).sum(dim=3).mean(dim=2) * self.scale
        # Notably, v_hat and l are not computed here because
        # they are unnecessary for deriving the gradient expression below.
        # We directly compute e = dl/dv_hat for the backward pass.

        # --- Backward ---
        e = -v / float(v.shape[2]) * self.scale
        g1 = k.transpose(-2, -1) @ (e * a)
        g2 = k.transpose(-2, -1) @ (e * z1 * (sig * (1.0 + z2 * (1.0 - sig))))

        # --- Clip gradient (for stability) ---
        g1 = g1 / (g1.norm(dim=-2, keepdim=True) + 1.0)
        g2 = g2 / (g2.norm(dim=-2, keepdim=True) + 1.0)

        # --- Step ---
        w1, w2 = w1 - lr * g1, w2 - lr * g2
        return w1, w2

    def inner_train_3x3dwc(self, k, v, w, lr=1.0, implementation="prod"):
        """
        Args:
            k (torch.Tensor): Spatial key tensor of shape [B, C, H, W]
            v (torch.Tensor): Spatial value tensor of shape [B, C, H, W]
            w (torch.Tensor): 3x3 convolution weights of shape [C, 1, 3, 3]
            lr (float, optional): Learning rate for inner-loop update. Default: 1.0
            implementation (str, optional): Implementation method, 'conv' or 'prod'. Default: 'prod'

        Returns:
            torch.Tensor: Updated convolution weights
        """
        # --- Forward ---
        # v_hat = F.conv2d(k, w, padding=1, groups=C)
        # l = - (v_hat * v).mean(dim=[-2, -1]) * self.scale
        # Notably, v_hat and l are not computed here because
        # they are unnecessary for deriving the gradient expression below.
        # We directly compute e = dl/dv_hat for the backward pass.

        # --- Backward ---
        # Two equivalent implementations. The 'prod' implementation appears to be slightly faster
        B, C, H, W = k.shape
        e = -v / float(v.shape[2] * v.shape[3]) * self.scale
        if implementation == "conv":
            g = F.conv2d(k.reshape(1, B * C, H, W), e.reshape(B * C, 1, H, W), padding=1, groups=B * C)
            g = g.transpose(0, 1)
        elif implementation == "prod":
            k = F.pad(k, (1, 1, 1, 1))
            outs = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ys = 1 + dy
                    xs = 1 + dx
                    dot = (k[:, :, ys : ys + H, xs : xs + W] * e).sum(dim=(-2, -1))
                    outs.append(dot)
            g = torch.stack(outs, dim=-1).reshape(B * C, 1, 3, 3)
        else:
            raise NotImplementedError

        # --- Clip gradient (for stability) ---
        g = g / (g.norm(dim=[-2, -1], keepdim=True) + 1.0)

        # --- Step ---
        w = w.repeat(B, 1, 1, 1) - lr * g
        return w

    def forward(self, x, h, w, rope=None):
        """
        Args:
            x (torch.Tensor): Input features with shape of (B, N, C)
            h (int): Feature map height
            w (int): Feature map width
            rope (nn.Module, optional): Rotary Position Embedding
        """
        b, n, c = x.shape
        d = c // self.num_heads

        # Prepare q/k/v
        q1, k1, v1, q2, k2, v2 = torch.split(self.qkv(x), [c, c, c, d, d, d], dim=-1)
        if rope is not None:
            q1 = rope(q1.reshape(b, h, w, c)).reshape(b, n, self.num_heads, d).transpose(1, 2)
            k1 = rope(k1.reshape(b, h, w, c)).reshape(b, n, self.num_heads, d).transpose(1, 2)
        else:
            q1 = q1.reshape(b, n, self.num_heads, d).transpose(1, 2)
            k1 = k1.reshape(b, n, self.num_heads, d).transpose(1, 2)
        v1 = v1.reshape(b, n, self.num_heads, d).transpose(1, 2)
        q2 = q2.reshape(b, h, w, d).permute(0, 3, 1, 2)
        k2 = k2.reshape(b, h, w, d).permute(0, 3, 1, 2)
        v2 = v2.reshape(b, h, w, d).permute(0, 3, 1, 2)

        # Inner training using (k, v)
        w1, w2 = self.inner_train_simplified_swiglu(k1, v1, self.w1, self.w2)
        w3 = self.inner_train_3x3dwc(k2, v2, self.w3, implementation="prod")

        # Apply updated inner module to q
        x1 = (q1 @ w1) * F.silu(q1 @ w2)
        x1 = x1.transpose(1, 2).reshape(b, n, c)
        x2 = F.conv2d(q2.reshape(1, b * d, h, w), w3, padding=1, groups=b * d)
        x2 = x2.reshape(b, d, n).transpose(1, 2)

        # Output proj
        x = torch.cat([x1, x2], dim=-1)
        x = self.proj(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}"


# *==============================================================
# * LaCT TTT Module
# *==============================================================


@torch.compile()
def silu_backprop(dy: torch.Tensor, x: torch.Tensor):
    """
    Args:
        dy: [b, d, l], gradient of the outer loss wrt the y
        x: [b, d, l], input of the silu activation
    outs:
        dx: [b, d, l], gradient of the outer loss wrt the x
        dx = dy * sigma * (1 + x * (1 - sigma))
    """
    sigma = torch.sigmoid(x)
    dx = dy * sigma * (1 + x * (1 - sigma))
    return dx


@torch.compile()
def l2_norm(x: torch.Tensor):
    """
    x: [b, l, d]
    """
    x_type = x.dtype
    ret = x / (x.norm(dim=-1, keepdim=True) + 1e-5)  # norm will upcast to float32
    return ret.type(x_type)


@torch.compile()
def zeropower_via_newtonschulz5(G):
    """
    This is an updated version of the zeropower_via_newtonschulz5 function in here:
    https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt_medium.py#L26
    The code is modified from https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py#L49, which contains the original muon implementation.
    Major change: G is [b, d, d] rather than [d, d]
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    Args:
        G: [b, d, d']
    Returns:
        X: [b, d, d']
    FLOPS:  When d=d', Total FLOPS=30 * b * d^3
    """
    assert len(G.shape) == 3
    X = G.bfloat16()
    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for a, b, c in [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]:
        A = X @ X.transpose(1, 2)
        B = b * A + c * A @ A  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    return X


@torch.compile
def bidirectional_lact_swiglu(
    w0: torch.Tensor,  # [b, dh, dk]
    w1: torch.Tensor,  # [b, dv, dh]
    w2: torch.Tensor,  # [b, dh, dk]
    q: torch.Tensor,  # [b, l, dk]
    k: torch.Tensor,  # [b, l, dk]
    v: torch.Tensor,  # [b, l, dv]
    lr0: torch.Tensor,  # [b, l, 1]
    lr1: torch.Tensor,  # [b, l, 1]
    lr2: torch.Tensor,  # [b, l, 1]
    use_muon: bool = True,
) -> torch.Tensor:
    """
    Bidirectional LaCT with SwiGLU fast weight function.
    w0, w1, w2 are the fast weights. f(x) =  w1 @ (silu(w0 @ x) * (w2 @ x))

    About precision:
        w0, w1, w2 are mostly likely fp32.
        q, k, v are fp16.
        lr0, lr1, lr2 are fp32.
        The forward, backward produce bf16 gradients, updated fast weights are fp32.
        The final output are bf16.


    FLOPS:
        (assume dk=dv denoted as D, hidden dimension of swiglu-mlp is H, ignore muon)
        Forward pass with key: 4 * D * H * L * B
        Backward pass: 8 * D * H * L * B
        Forward with Query: 6 * D * H * L * B
        Total: 18 * D * H * L * B
    Outputs:
        o: [b, l, dv]
    """

    # adding detach here sometimes improves stability.
    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)

    q = q.transpose(1, 2)  # [b, dk, l]
    v = v.transpose(1, 2)

    ######### update the fast weight w0, w1, w2 with test-time training #########

    #### Forward pass with key
    # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
    gate_before_act = torch.bmm(w0, k.transpose(1, 2))
    hidden_before_mul = torch.bmm(w2, k.transpose(1, 2))
    hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

    #### Backward pass to compute fast weight gradients
    # [b, dh, dv] @ [b, dv, l] -> [b, dh, l]
    dhidden = torch.bmm(w1.transpose(1, 2), v)

    dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
    dgate = dhidden * hidden_before_mul
    dgate_before_act = silu_backprop(dgate, gate_before_act)

    # [b, dv, l] @ [b, l, dh] -> [b, dv, dh]
    dw1 = torch.bmm(v, (hidden.transpose(1, 2) * lr1).type_as(v))  # [b, d, d]
    # [b, dh, l] @ [b, l, dk] -> [b, dh, dk]
    dw0 = torch.bmm(dgate_before_act, (k * lr0).type_as(dgate_before_act))
    dw2 = torch.bmm(dhidden_before_mul, (k * lr2).type_as(dhidden_before_mul))

    if use_muon:
        w0 = zeropower_via_newtonschulz5(dw0)
        w1 = zeropower_via_newtonschulz5(dw1)
        w2 = zeropower_via_newtonschulz5(dw2)

    w1 = w1 + dw1
    w0 = w0 + dw0
    w2 = w2 + dw2

    w0 = w0 / (w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
    w1 = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
    w2 = w2 / (w2.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

    ######### apply the updated fast weights to the query #########

    # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
    h = torch.bmm(w2, q)
    gate = F.silu(torch.bmm(w0, q), inplace=True)
    # [b, dv, dh] @ [b, dh, l] -> [b, dv, l] -> [b, l, dv]
    o = torch.bmm(w1, gate * h).transpose(1, 2)

    return o


def inv_softplus(x):
    if isinstance(x, torch.Tensor):
        y = x + torch.log(-torch.expm1(-x))
    else:
        y = x + math.log(-math.expm1(-x))
    return y


class BidirectionalLaCTSwiGLU(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        inter_multi: float = 1,
        use_o_norm: bool = True,  # recommended to be True
        qk_l2_norm: bool = True,  # recommended to be True
        use_muon: bool = True,  # if your seq len > head_dim * 2, recommended to be True
        base_lr: float = 1e-2,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.inter_multi = inter_multi
        self.use_o_norm = use_o_norm
        self.qk_l2_norm = qk_l2_norm

        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = dim // head_dim

        self.to_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.lr_dim = 1  # single scalar learning rate for each head
        self.lr_proj = nn.Linear(dim, self.lr_dim * 3 * self.num_heads, bias=False)
        self.base_lr = base_lr
        self.base_lr_inv = inv_softplus(base_lr)

        # create initial fast weights
        d_in, d_out = self.head_dim, self.head_dim
        d_h = int(self.head_dim * self.inter_multi)

        self.w0 = nn.Parameter(torch.randn(self.num_heads, d_h, d_in) / math.sqrt(d_in))
        self.w1 = nn.Parameter(torch.randn(self.num_heads, d_out, d_h) / math.sqrt(d_h))
        self.w2 = nn.Parameter(torch.randn(self.num_heads, d_h, d_in) / math.sqrt(d_in))

        self.qk_l2_norm = qk_l2_norm
        self.use_muon = use_muon

        self.use_o_norm = use_o_norm
        if self.use_o_norm:
            self.o_norm = nn.RMSNorm(head_dim, eps=1e-5, elementwise_affine=True)
        else:
            self.o_norm = nn.Identity()

    def forward(self, x: torch.Tensor, rope: tuple[Tensor, Tensor] | None = None) -> torch.Tensor:
        """
        x: [b, l, d]
        """

        qkv = F.silu(self.to_qkv(x), inplace=True)  # SiLU - Linear

        # [b * num_heads, l, head_dim]
        q, k, v = rearrange(
            qkv,
            "b l (qkv h d) -> qkv (b h) l d",
            qkv=3,
            h=self.num_heads,
            d=self.head_dim,
        )

        if rope is not None:
            q = rearrange(q, "(b h) l d -> b l h d", h=self.num_heads, b=x.shape[0])
            k = rearrange(k, "(b h) l d -> b l h d", h=self.num_heads, b=x.shape[0])
            q, k = LacTAttentionFFNBlock._apply_rope(q, k, rope)
            q = rearrange(q, "b l h d -> (b h) l d", h=self.num_heads)
            k = rearrange(k, "b l h d -> (b h) l d", h=self.num_heads)

        if self.qk_l2_norm:
            q = l2_norm(q)
            k = l2_norm(k)

        # better to have float32 for lr.
        # For muon, I found that float16 is still very good.
        with torch.autocast(device_type="cuda", enabled=False):
            lr = self.lr_proj(x)  # [b, l, lr_dim]

        lr = torch.nn.functional.softplus(lr.float() + self.base_lr_inv)

        # [b * num_heads, l, 1] for each lr
        lr0, lr1, lr2 = rearrange(lr, "b l (h lrs d) -> lrs (b h) l d", lrs=3, h=self.num_heads, d=self.lr_dim)

        # [nh, d, d] -> [b * nh, d, d]
        w0 = self.w0.repeat(x.shape[0], 1, 1)
        w1 = self.w1.repeat(x.shape[0], 1, 1)
        w2 = self.w2.repeat(x.shape[0], 1, 1)

        # [b * num_heads, l, head_dim]
        output = bidirectional_lact_swiglu(w0, w1, w2, q, k, v, lr0, lr1, lr2, self.use_muon)

        output = self.o_norm(output)
        output = rearrange(output, "(b h) l d -> b l (h d)", h=self.num_heads, b=x.shape[0])
        output = self.o_proj(output)

        # [b, l, d]
        return output


class LacTAttentionFFNBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        lact_inter_multi: float = 1.0,
        lact_use_o_norm: bool = True,
        lact_qk_l2_norm: bool = True,
        lact_use_muon: bool = True,
        lact_base_lr: float = 1e-2,
        use_ffn: bool = True,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        ffn_drop: float = 0.0,
        rms_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"`dim` must be divisible by `num_heads`, got dim={dim}, num_heads={num_heads}.")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.softmax_scale = self.head_dim**-0.5

        self.use_ffn = use_ffn

        self.lact_norm = nn.RMSNorm(dim, eps=rms_norm_eps, elementwise_affine=True)
        self.attn_norm = nn.RMSNorm(dim, eps=rms_norm_eps, elementwise_affine=True)
        self.ffn_norm = nn.RMSNorm(dim, eps=rms_norm_eps, elementwise_affine=True) if use_ffn else nn.Identity()

        self.lact = BidirectionalLaCTSwiGLU(
            dim=dim,
            head_dim=self.head_dim,
            inter_multi=lact_inter_multi,
            use_o_norm=lact_use_o_norm,
            qk_l2_norm=lact_qk_l2_norm,
            use_muon=lact_use_muon,
            base_lr=lact_base_lr,
        )

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=False)

        self.ffn = nn.Identity()
        if use_ffn:
            ffn_hidden_dim = int(dim * ffn_ratio)
            self.ffn = SwiGLU(
                in_features=dim,
                hidden_features=ffn_hidden_dim,
                out_features=dim,
                bias=False,
                drop=float(ffn_drop),
            )

        self.attn_drop_p = float(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    @staticmethod
    def _apply_rope(
        q: Tensor,
        k: Tensor,
        rope: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        """
        Apply real-valued RoPE (cos, sin) to q and k.

        Parameters
        ----------
        q : Tensor
            Query tensor with shape [B, L, H, D].
        k : Tensor
            Key tensor with shape [B, L, H, D].
        rope : tuple[Tensor, Tensor]
            (cos, sin) with shape [S, D_rope].

        Returns
        -------
        tuple[Tensor, Tensor]
            Rotated (q, k) with the same shapes as inputs.
        """
        cos, sin = rope
        b, l, h, d = q.shape

        if cos.ndim != 2 or sin.ndim != 2:
            raise ValueError(f"`rope` must be 2D (cos, sin), got cos.ndim={cos.ndim}, sin.ndim={sin.ndim}.")
        if cos.shape != sin.shape:
            raise ValueError(f"`rope` cos/sin must have same shape, got {cos.shape} vs {sin.shape}.")
        if cos.shape[0] < l:
            raise ValueError(f"`rope` sequence dim too small, got S={cos.shape[0]} < L={l}.")

        cos = cos[:l]
        sin = sin[:l]
        rotary_dim = int(cos.shape[1])
        if rotary_dim > d:
            raise ValueError(f"`rope` dim too large, got D_rope={rotary_dim} > head_dim={d}.")
        if rotary_dim % 2 != 0:
            raise ValueError(f"`rope` dim must be even, got D_rope={rotary_dim}.")

        q = q.transpose(1, 2)  # [B, H, L, D]
        k = k.transpose(1, 2)

        if rotary_dim == d:
            q = apply_rotary_emb(q, (cos, sin), use_real=True, sequence_dim=2)
            k = apply_rotary_emb(k, (cos, sin), use_real=True, sequence_dim=2)
        else:
            q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
            k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
            q_rot = apply_rotary_emb(q_rot, (cos, sin), use_real=True, sequence_dim=2)
            k_rot = apply_rotary_emb(k_rot, (cos, sin), use_real=True, sequence_dim=2)
            q = torch.cat([q_rot, q_pass], dim=-1)
            k = torch.cat([k_rot, k_pass], dim=-1)

        q = q.transpose(1, 2)  # [B, L, H, D]
        k = k.transpose(1, 2)
        return q, k

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor] | None = None) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Input tensor with shape [B, L, C].
        rope : tuple[Tensor, Tensor] | None
            Real RoPE (cos, sin). Each is [S, D_rope]. If provided, it will be applied to q/k.
        """
        x = x + self.lact(self.lact_norm(x), rope=rope)

        x_attn = self.attn_norm(x)
        qkv = self.qkv(x_attn)
        q, k, v = qkv.chunk(3, dim=-1)

        q = rearrange(q, "b l (h d) -> b l h d", h=self.num_heads, d=self.head_dim)
        k = rearrange(k, "b l (h d) -> b l h d", h=self.num_heads, d=self.head_dim)
        v = rearrange(v, "b l (h d) -> b l h d", h=self.num_heads, d=self.head_dim)

        if rope is not None:
            q, k = self._apply_rope(q, k, rope)

        attn_out = flash_attn_func(
            q,
            k,
            v,
            dropout_p=self.attn_drop_p,
            softmax_scale=self.softmax_scale,
            causal=False,
        )
        attn_out = rearrange(attn_out, "b l h d -> b l (h d)")
        attn_out = self.proj(attn_out)
        x = x + self.proj_drop(attn_out)

        if self.use_ffn:
            x_ffn = self.ffn_norm(x)
            x = x + self.ffn(x_ffn)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}, head_dim={self.head_dim}, use_ffn={self.use_ffn}"


LacTAttentionFFNLayer = LacTAttentionFFNBlock


def _test_layer():
    B, L, D, HeadDim = 4, 32768, 2048, 512

    layer = BidirectionalLaCTSwiGLU(D, HeadDim, use_muon=True)

    layer = layer.to("cuda")

    x = torch.randn(B, L, D).to("cuda")

    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        output = layer(x)
    print(output.shape, output.dtype)
    print("Input norm", x.norm(), "Output norm", output.norm())


def _test_rope_and_lact_cpu() -> None:
    """Sanity check RoPE application and LaCT forward on CPU."""
    b, l, dim, num_heads = 2, 32, 64, 4
    head_dim = dim // num_heads

    x = torch.randn(b, l, dim)
    cos, sin = get_1d_rotary_pos_embed(dim=head_dim, pos=torch.arange(l), use_real=True)

    q = torch.randn(b, l, num_heads, head_dim)
    k = torch.randn(b, l, num_heads, head_dim)
    q2, k2 = LacTAttentionFFNBlock._apply_rope(q, k, (cos, sin))
    assert q2.shape == q.shape
    assert k2.shape == k.shape

    lact = BidirectionalLaCTSwiGLU(dim=dim, head_dim=head_dim, use_muon=False)
    y_no_rope = lact(x)
    y_rope = lact(x, rope=(cos, sin))
    assert y_no_rope.shape == x.shape
    assert y_rope.shape == x.shape

    block_ffn = LacTAttentionFFNBlock(dim=dim, num_heads=num_heads, use_ffn=True, lact_use_muon=False)
    assert block_ffn.use_ffn is True
    block_no_ffn = LacTAttentionFFNBlock(dim=dim, num_heads=num_heads, use_ffn=False, lact_use_muon=False)
    assert block_no_ffn.use_ffn is False


if __name__ == "__main__":
    _test_rope_and_lact_cpu()
    if torch.cuda.is_available():
        _test_layer()

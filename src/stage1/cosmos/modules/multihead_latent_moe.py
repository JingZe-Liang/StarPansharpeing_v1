"""
multihead_latent_moe.py
=======================

Standalone **Multi-Head LatentMoE** + **Head Parallel (HP)** reference implementation
(arXiv:2602.04870) in pure PyTorch, with an optional FlexAttention-based expert
computation path.

This file is self-contained and can be imported as a module or executed as a
script to run included training / inference demos.

What this implements (high-level)
---------------------------------
Multi-Head LatentMoE projects each token into Nh sub-tokens (heads), then runs an
independent MoE per head, concatenates outputs, and projects back:

    [x_{t,1},...,x_{t,Nh}] = split(W_in x_t)
    o_t = W_out concat( f_1(x_{t,1}), ..., f_{Nh}(x_{t,Nh}) )

Head Parallel (HP) moves the all-to-all **before routing** by redistributing the
head dimension across GPUs, so each GPU can do routing + expert compute locally
for its assigned heads. Communication becomes O(1) in k (top-k), balanced, and
deterministic.

This file provides:
- MultiHeadLatentMoEConfig (dataclass)
- ProcessMesh2D helper (DP × HP "meshgrid")
- MultiHeadLatentMoELayer with optional head-parallel all_to_all
- Optional IO-aware routing forward (streaming top-k in blocks, pure PyTorch)
- Optional FlexAttention-based expert computation (pure PyTorch)
- Tiny demo LM + train/infer demos

⚠️ Practical notes
------------------
- The paper's IO-aware routing/expert compute target HBM/SRAM behavior and uses
  Triton/CUDA kernels. Here we provide **a functional PyTorch reference** that
  matches the algorithms conceptually, but it is **not** expected to match the
  kernel-level performance results.
- HP in this implementation assumes **Nh is divisible by hp_size** and that each
  rank in an HP group processes the same (B,T) shape (standard in distributed training).

---------------------------------------------------------------------------
Quickstart (single process)
---------------------------------------------------------------------------
>>> import torch
>>> from multihead_latent_moe import MultiHeadLatentMoEConfig, MultiHeadLatentMoELayer
>>>
>>> x = torch.randn(2, 128, 256)
>>> cfg = MultiHeadLatentMoEConfig(
...     d_model=256,
...     num_heads=8,
...     head_dim=32,
...     num_experts=8,
...     top_k=2,
...     d_hidden=64,
... )
>>> layer = MultiHeadLatentMoELayer(cfg)
>>> y = layer(x)
>>> y.shape
torch.Size([2, 128, 256])

---------------------------------------------------------------------------
Distributed demo (Head Parallel on CPU)
---------------------------------------------------------------------------
Run with hp_size=2:

    torchrun --standalone --nproc_per_node=2 multihead_latent_moe.py --demo train --dp_size 1 --hp_size 2

Generation demo:

    torchrun --standalone --nproc_per_node=2 multihead_latent_moe.py --demo infer --dp_size 1 --hp_size 2

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import math
import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# FlexAttention is available in PyTorch 2.5+
try:
    from torch.nn.attention.flex_attention import flex_attention
except Exception:  # pragma: no cover
    flex_attention = None


# -----------------------------
# Distributed "meshgrid" helper
# -----------------------------


class ProcessMesh2D:
    """
    Simple 2D process mesh: DP × MP.

    In this file, MP is interpreted as HP (head-parallel).

        dp_rank = rank // mp_size
        mp_rank = rank % mp_size
    """

    def __init__(self, dp_size: int, mp_size: int, backend_hint: Optional[str] = None):
        if dist.is_initialized():
            world = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world = 1
            rank = 0

        if dp_size * mp_size != world:
            raise ValueError(f"dp_size*mp_size must equal world_size. Got dp={dp_size}, mp={mp_size}, world={world}")

        self.dp_size = dp_size
        self.mp_size = mp_size
        self.world_size = world
        self.rank = rank

        self.dp_rank = rank // mp_size
        self.mp_rank = rank % mp_size

        self._mp_groups: List[dist.ProcessGroup] = []
        self._dp_groups: List[dist.ProcessGroup] = []

        if dist.is_initialized() and world > 1:
            for d in range(dp_size):
                ranks = [d * mp_size + i for i in range(mp_size)]
                self._mp_groups.append(dist.new_group(ranks=ranks, backend=backend_hint))
            for m in range(mp_size):
                ranks = [d * mp_size + m for d in range(dp_size)]
                self._dp_groups.append(dist.new_group(ranks=ranks, backend=backend_hint))
            self.mp_group = self._mp_groups[self.dp_rank]
            self.dp_group = self._dp_groups[self.mp_rank]
        else:
            self.mp_group = None
            self.dp_group = None

    def mp_world_size(self) -> int:
        return self.mp_size

    def mp_rank_in_group(self) -> int:
        return self.mp_rank

    def all_reduce_grads(self, module: nn.Module, average: bool = True) -> None:
        """
        Synchronize gradients for a 2D DP×MP mesh (here MP is Head Parallel).

        Mark MP-sharded parameters by setting:
            param._mesh_mp_sharded = True

        Replicated parameters are reduced across MP group and DP group (global on the 2D mesh).
        MP-sharded parameters are reduced across DP group only.
        """
        if not dist.is_initialized():
            return

        mp_group = self.mp_group
        dp_group = self.dp_group

        for p in module.parameters():
            if p.grad is None:
                continue

            is_mp_sharded = bool(getattr(p, "_mesh_mp_sharded", False))

            if is_mp_sharded:
                if self.dp_size > 1 and dp_group is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=dp_group)
                    if average:
                        p.grad.div_(self.dp_size)
                continue

            if self.mp_size > 1 and mp_group is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=mp_group)
            if self.dp_size > 1 and dp_group is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=dp_group)

            if average:
                p.grad.div_(self.dp_size * self.mp_size)


# -----------------------------
# MoE building blocks
# -----------------------------


@dataclass
class MultiHeadLatentMoEConfig:
    """
    Configuration for Multi-Head LatentMoE.

    Args:
      d_model:    model hidden size d
      num_heads:  Nh
      head_dim:   dh (often d_model/num_heads)
      num_experts: Ne per head
      top_k:      k
      d_hidden:   de (neurons per expert) for the 2-layer expert MLP
      dropout:    dropout applied after projections

    Routing / compute options:
      - router_fp32: compute router logits in FP32 (stability)
      - use_io_aware_routing: streaming top-k in blocks (exact, pure PyTorch)
      - expert_impl: "loop" | "flex"
          "loop": standard per-expert token gather + matmul
          "flex": per-expert FlexAttention-based gelu FFN (exact)
    """

    d_model: int
    num_heads: int
    head_dim: int
    num_experts: int
    top_k: int
    d_hidden: int
    dropout: float = 0.0

    router_fp32: bool = True
    router_bias: bool = False
    router_jitter_noise: float = 0.0

    use_io_aware_routing: bool = False
    io_route_token_block: int = 256
    io_route_expert_block: int = 64

    expert_impl: str = "loop"  # "loop" or "flex"
    expert_activation: str = "gelu"  # for "flex" path we assume GELU


def _seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def init_distributed_from_env() -> None:
    if dist.is_initialized():
        return
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)


class TopKRouter(nn.Module):
    def __init__(self, d_in: int, num_experts: int, fp32: bool = True, bias: bool = False, jitter_noise: float = 0.0):
        super().__init__()
        self.num_experts = num_experts
        self.fp32 = fp32
        self.jitter_noise = float(jitter_noise)
        self.linear = nn.Linear(d_in, num_experts, bias=bias)

    def forward_dense(self, x: torch.Tensor, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.linear(x.float() if self.fp32 else x)
        if self.jitter_noise > 0:
            logits = logits + torch.randn_like(logits) * self.jitter_noise
        k = min(int(top_k), self.num_experts)
        vals, idx = torch.topk(logits, k=k, dim=-1)
        w = F.softmax(vals, dim=-1).to(dtype=x.dtype)
        return idx, w

    def forward_io_aware(
        self,
        x: torch.Tensor,
        top_k: int,
        token_block: int,
        expert_block: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Exact streaming top-k in blocks (conceptually aligned with Algorithm 1).

        This is a pure-PyTorch version:
          - loops over expert blocks
          - maintains running top-k values/indices for each token
        """
        T, d_in = x.shape
        Ne = self.num_experts
        k = min(int(top_k), Ne)

        # We compute logits in blocks to reduce peak activation memory.
        W = self.linear.weight  # (Ne, d_in)
        b = self.linear.bias  # (Ne,) or None

        # Running top-k (values and indices) for each token.
        top_vals = torch.full((T, k), -float("inf"), device=x.device, dtype=torch.float32)
        top_idx = torch.zeros((T, k), device=x.device, dtype=torch.int64)

        # Process experts in chunks.
        for e0 in range(0, Ne, expert_block):
            e1 = min(Ne, e0 + expert_block)
            Wb = W[e0:e1, :]  # (Eb, d_in)
            logits_b = x.float().matmul(Wb.t())  # (T, Eb)
            if b is not None:
                logits_b = logits_b + b[e0:e1].float()

            # local top-k within block
            local_k = min(k, e1 - e0)
            vals_b, idx_b = torch.topk(logits_b, k=local_k, dim=-1)  # (T,local_k)
            idx_b = idx_b + e0

            # merge with running top-k
            cand_vals = torch.cat([top_vals, vals_b], dim=-1)  # (T, k+local_k)
            cand_idx = torch.cat([top_idx, idx_b], dim=-1)

            new_vals, new_pos = torch.topk(cand_vals, k=k, dim=-1)
            new_idx = torch.gather(cand_idx, dim=-1, index=new_pos)

            top_vals, top_idx = new_vals, new_idx

        weights = F.softmax(top_vals, dim=-1).to(dtype=x.dtype)
        return top_idx, weights

    def forward(
        self, x: torch.Tensor, top_k: int, io_aware: bool = False, token_block: int = 256, expert_block: int = 64
    ):
        if io_aware:
            return self.forward_io_aware(x, top_k=top_k, token_block=token_block, expert_block=expert_block)
        return self.forward_dense(x, top_k=top_k)


class ExpertFFN(nn.Module):
    """
    A simple 2-layer expert MLP in head_dim:

        y = gelu(x W_in^T) W_out

    Shapes:
      W_in : (d_hidden, head_dim)
      W_out: (d_hidden, head_dim)

    This matches the attention-duality used in the paper's FlexAttention method.
    """

    def __init__(self, head_dim: int, d_hidden: int):
        super().__init__()
        self.W_in = nn.Parameter(torch.empty(d_hidden, head_dim))
        self.W_out = nn.Parameter(torch.empty(d_hidden, head_dim))
        nn.init.normal_(self.W_in, mean=0.0, std=0.02)
        nn.init.normal_(self.W_out, mean=0.0, std=0.02)

    def forward_loop(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, head_dim)
        hidden = F.gelu(x.matmul(self.W_in.t()))
        return hidden.matmul(self.W_out)

    def forward_flex(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute gelu(x W_in^T) W_out using FlexAttention as in Eq.(7)-(8) discussion.

        This is *exact* but is a reference implementation, not a fused kernel.
        """
        if flex_attention is None:
            raise RuntimeError("flex_attention is not available in this PyTorch build.")

        # Attention expects (B,H,Q,E). We'll use B=1,H=1.
        q = x.unsqueeze(0).unsqueeze(0)  # (1,1,T,dh)
        k = self.W_in.unsqueeze(0).unsqueeze(0)  # (1,1,de,dh)
        v = self.W_out.unsqueeze(0).unsqueeze(0)  # (1,1,de,dh)

        # score_mod signature: (score, b, h, q_idx, kv_idx) -> modified score
        def score_mod(score, b, h, q_idx, kv_idx):
            # log(gelu(score)+1) so softmax becomes (gelu(score)+1)/ℓ
            return torch.log(F.gelu(score) + 1.0)

        out, lse = flex_attention(q, k, v, score_mod=score_mod, block_mask=None, return_lse=True)
        # out: (1,1,T,dh) = (gelu(score)+1)V / ℓ
        # lse: (1,1,T) = log ℓ
        ell = torch.exp(lse).unsqueeze(-1)  # (1,1,T,1)
        out = out * ell  # (gelu(score)+1)V

        # Subtract bias term 1V = sum_j V_j for this expert
        v_sum = v.sum(dim=-2, keepdim=True)  # (1,1,1,dh)
        out = out - v_sum  # gelu(score)V

        return out.squeeze(0).squeeze(0)  # (T,dh)

    def forward(self, x: torch.Tensor, impl: str = "loop") -> torch.Tensor:
        if impl == "loop":
            return self.forward_loop(x)
        if impl == "flex":
            return self.forward_flex(x)
        raise ValueError(f"Unknown impl: {impl}")


class MoEHead(nn.Module):
    """
    One head's MoE: router + a set of experts.

    This is an **intra-rank** MoE (no expert-parallel all-to-all). It gathers
    tokens per expert locally and scatters outputs back.
    """

    def __init__(
        self,
        head_dim: int,
        num_experts: int,
        top_k: int,
        d_hidden: int,
        router_fp32: bool,
        router_bias: bool,
        router_jitter_noise: float,
        use_io_aware_routing: bool,
        io_route_token_block: int,
        io_route_expert_block: int,
        expert_impl: str,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = TopKRouter(
            head_dim, num_experts, fp32=router_fp32, bias=router_bias, jitter_noise=router_jitter_noise
        )
        self.use_io_aware_routing = use_io_aware_routing
        self.io_route_token_block = io_route_token_block
        self.io_route_expert_block = io_route_expert_block

        self.expert_impl = expert_impl
        self.experts = nn.ModuleList([ExpertFFN(head_dim, d_hidden) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (T, head_dim)
        returns: (T, head_dim)
        """
        T, dh = x.shape
        idx, w = self.router(
            x,
            top_k=self.top_k,
            io_aware=self.use_io_aware_routing,
            token_block=self.io_route_token_block,
            expert_block=self.io_route_expert_block,
        )  # (T,k), (T,k)

        k = idx.shape[1]
        token_ids = torch.arange(T, device=x.device, dtype=torch.int64).unsqueeze(1).expand(T, k).reshape(-1)
        expert_ids = idx.reshape(-1).to(torch.int64)
        weights = w.reshape(-1)
        x_rep = x.unsqueeze(1).expand(T, k, dh).reshape(-1, dh)

        # Group by expert id for local compute
        order = torch.argsort(expert_ids)
        expert_ids = expert_ids[order]
        token_ids = token_ids[order]
        weights = weights[order]
        x_rep = x_rep[order]

        out_rep = torch.empty_like(x_rep)
        # loop over experts
        for e in range(self.num_experts):
            m = expert_ids == e
            if m.any():
                out_rep[m] = self.experts[e](x_rep[m], impl=self.expert_impl)

        out_rep = out_rep * weights.unsqueeze(-1)

        # Scatter-add back
        y = torch.zeros((T, dh), device=x.device, dtype=x.dtype)
        y.index_add_(0, token_ids, out_rep)
        return y


# -----------------------------
# Multi-Head LatentMoE + Head Parallel
# -----------------------------


class MultiHeadLatentMoELayer(nn.Module):
    """
    Multi-Head LatentMoE layer with optional Head Parallel (HP).

    x: (B, T, d_model) -> y: (B, T, d_model)

    If `mesh` is provided and mesh.mp_size>1 (interpreted as HP):
      - project+split into (B*T, Nh, dh)
      - all_to_all across HP group to redistribute heads
      - compute local heads' MoE for all received tokens
      - reverse all_to_all to send results back
      - concat heads, project out
    """

    def __init__(self, cfg: MultiHeadLatentMoEConfig, mesh: Optional[ProcessMesh2D] = None):
        super().__init__()
        self.cfg = cfg
        self.mesh = mesh

        if cfg.num_heads * cfg.head_dim != cfg.d_model:
            raise ValueError("This reference assumes num_heads*head_dim == d_model for simplicity.")

        # Projections in/out of "latent head space"
        self.w_in = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.w_out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

        # Head sharding
        hp_size = self.hp_world_size
        if cfg.num_heads % hp_size != 0:
            raise ValueError(f"num_heads={cfg.num_heads} must be divisible by hp_size={hp_size}")
        self.local_heads = cfg.num_heads // hp_size
        self.head_offset = self.hp_rank * self.local_heads

        # Instantiate only local heads (parameters live only on head-owner rank)
        self.heads = nn.ModuleList(
            [
                MoEHead(
                    head_dim=cfg.head_dim,
                    num_experts=cfg.num_experts,
                    top_k=cfg.top_k,
                    d_hidden=cfg.d_hidden,
                    router_fp32=cfg.router_fp32,
                    router_bias=cfg.router_bias,
                    router_jitter_noise=cfg.router_jitter_noise,
                    use_io_aware_routing=cfg.use_io_aware_routing,
                    io_route_token_block=cfg.io_route_token_block,
                    io_route_expert_block=cfg.io_route_expert_block,
                    expert_impl=cfg.expert_impl,
                )
                for _ in range(self.local_heads)
            ]
        )

        # Mark per-head MoE parameters as MP-sharded (heads live only on the head-owner rank)
        for _p in self.heads.parameters():
            setattr(_p, "_mesh_mp_sharded", True)

    @property
    def hp_world_size(self) -> int:
        if self.mesh is None:
            return 1
        return self.mesh.mp_world_size()

    @property
    def hp_rank(self) -> int:
        if self.mesh is None:
            return 0
        return self.mesh.mp_rank_in_group()

    @property
    def hp_group(self) -> Optional[dist.ProcessGroup]:
        if self.mesh is None:
            return None
        return self.mesh.mp_group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        cfg = self.cfg

        x_proj = self.w_in(x)  # (B,T,D)
        x_heads = x_proj.view(B * T, cfg.num_heads, cfg.head_dim)  # (Ntok,Nh,dh)

        if self.hp_world_size == 1 or not dist.is_initialized():
            # Local heads path (no head-parallel)
            y_heads = self._forward_local_heads(x_heads)  # (Ntok,Nh,dh)
        else:
            y_heads = self._forward_head_parallel(x_heads)  # (Ntok,Nh,dh)

        y = y_heads.reshape(B, T, D)
        y = self.w_out(y)
        y = self.drop(y)
        return y

    def _forward_local_heads(self, x_heads: torch.Tensor) -> torch.Tensor:
        """
        x_heads: (Ntok, Nh, dh) computed locally.
        """
        Ntok, Nh, dh = x_heads.shape
        outs = []
        for h in range(Nh):
            head = self.heads[h]  # if hp_world_size==1, local_heads==Nh
            outs.append(head(x_heads[:, h, :]))
        y = torch.stack(outs, dim=1)
        return y

    def _forward_head_parallel(self, x_heads: torch.Tensor) -> torch.Tensor:
        """
        Head Parallel (HP) implementation.

        Assumes:
          - same Ntok = B*T on each rank inside hp_group
          - Nh divisible by hp_size

        Steps:
          1) all_to_all to send each rank's head slices to head-owner ranks
          2) compute MoE per local head for all received tokens
          3) reverse all_to_all to return results to token-owner ranks
          4) concatenate head slices in rank order to restore Nh
        """
        cfg = self.cfg
        hp = self.hp_world_size
        Ntok, Nh, dh = x_heads.shape
        local_h = self.local_heads
        device = x_heads.device

        # Pack send buffer: concat head slices for each destination rank
        # x_heads: (Ntok, Nh, dh) -> (hp, Ntok*local_h, dh)
        send = x_heads.view(Ntok, hp, local_h, dh).transpose(0, 1).reshape(hp, Ntok * local_h, dh).contiguous()
        send_flat = send.reshape(hp * Ntok * local_h, dh)

        split = [Ntok * local_h] * hp
        recv_flat = torch.empty_like(send_flat)

        dist.all_to_all_single(
            output=recv_flat,
            input=send_flat,
            output_split_sizes=split,
            input_split_sizes=split,
            group=self.hp_group,
        )
        recv = (
            recv_flat.view(hp, Ntok * local_h, dh).view(hp, Ntok, local_h, dh).contiguous()
        )  # (src, Ntok, local_h, dh)

        # Compute local heads for each source rank's tokens
        out = torch.empty_like(recv)
        for j in range(local_h):
            head_mod = self.heads[j]  # local head j corresponds to global head (head_offset + j)
            # tokens for this head from all sources: (hp, Ntok, dh) -> flatten for compute
            xj = recv[:, :, j, :].reshape(hp * Ntok, dh)
            yj = head_mod(xj)  # (hp*Ntok, dh)
            out[:, :, j, :] = yj.view(hp, Ntok, dh)

        # Reverse all-to-all: send outputs for each source rank back
        # out: (src, Ntok, local_h, dh) -> flat segments ordered by src
        send2 = out.view(hp, Ntok * local_h, dh).reshape(hp * Ntok * local_h, dh).contiguous()
        recv2 = torch.empty_like(send2)

        dist.all_to_all_single(
            output=recv2,
            input=send2,
            output_split_sizes=split,
            input_split_sizes=split,
            group=self.hp_group,
        )

        # recv2 segments are ordered by sender (head-owner ranks).
        recv2 = recv2.view(hp, Ntok, local_h, dh).contiguous()  # (sender, Ntok, local_h, dh)

        # Reassemble Nh heads by concatenating sender slices in rank order
        # sender r corresponds to global head slice [r*local_h:(r+1)*local_h]
        y_heads = recv2.permute(1, 0, 2, 3).reshape(Ntok, Nh, dh).contiguous()
        return y_heads


# -----------------------------
# Tiny LM demo (training & inference)
# -----------------------------


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def reshape(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,T,dh)

        q = reshape(q)
        k = reshape(k)
        v = reshape(v)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)


class MHMoEBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff: MultiHeadLatentMoELayer, dropout: float = 0.0):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, dropout=dropout)
        self.ff = ff
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class TinyMHMoELM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ff_cfg: MultiHeadLatentMoEConfig,
        mesh: Optional[ProcessMesh2D] = None,
        dropout: float = 0.0,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        blocks = []
        for _ in range(n_layers):
            ff = MultiHeadLatentMoELayer(ff_cfg, mesh=mesh)
            blocks.append(MHMoEBlock(d_model, n_heads, ff=ff, dropout=dropout))
        self.blocks = nn.ModuleList(blocks)
        self.norm_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        if T > self.max_seq_len:
            raise ValueError("Sequence too long for pos_emb")
        pos = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len :]
            logits = self(idx_cond)
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.argmax(probs, dim=-1, keepdim=True)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


def train_demo(dp_size: int, hp_size: int, steps: int = 50, seed: int = 123, device: Optional[str] = None) -> None:
    init_distributed_from_env()
    _seed_all(seed + (dist.get_rank() if dist.is_initialized() else 0))

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    mesh = ProcessMesh2D(dp_size=dp_size, mp_size=hp_size)

    vocab = 256
    d_model = 128
    n_heads_attn = 4
    n_layers = 2
    seq_len = 64
    batch = 8

    ff_cfg = MultiHeadLatentMoEConfig(
        d_model=d_model,
        num_heads=8,
        head_dim=16,
        num_experts=8,
        top_k=2,
        d_hidden=32,
        dropout=0.0,
        use_io_aware_routing=True,  # demo the streaming top-k
        expert_impl="loop",  # "flex" also works but is slower
    )

    model = TinyMHMoELM(
        vocab_size=vocab,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads_attn,
        ff_cfg=ff_cfg,
        mesh=mesh,
        dropout=0.0,
        max_seq_len=seq_len,
    ).to(device_t)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for step in range(steps):
        model.train()
        idx = torch.randint(0, vocab, (batch, seq_len), device=device_t)
        logits = model(idx)
        loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, vocab), idx[:, 1:].reshape(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        mesh.all_reduce_grads(model, average=True)
        opt.step()

        if (not dist.is_initialized()) or dist.get_rank() == 0:
            if step % 10 == 0 or step == steps - 1:
                print(f"[train] step={step:04d} loss={loss.item():.4f}")


def infer_demo(dp_size: int, hp_size: int, seed: int = 123, device: Optional[str] = None) -> None:
    init_distributed_from_env()
    _seed_all(seed + (dist.get_rank() if dist.is_initialized() else 0))

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    mesh = ProcessMesh2D(dp_size=dp_size, mp_size=hp_size)

    vocab = 64
    d_model = 128
    n_heads_attn = 4
    n_layers = 2
    seq_len = 64

    ff_cfg = MultiHeadLatentMoEConfig(
        d_model=d_model,
        num_heads=8,
        head_dim=16,
        num_experts=8,
        top_k=2,
        d_hidden=32,
        dropout=0.0,
        use_io_aware_routing=False,
        expert_impl="loop",
    )

    model = TinyMHMoELM(
        vocab_size=vocab,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads_attn,
        ff_cfg=ff_cfg,
        mesh=mesh,
        dropout=0.0,
        max_seq_len=seq_len,
    ).to(device_t)

    prompt = torch.zeros((1, 8), dtype=torch.long, device=device_t)
    out = model.generate(prompt, max_new_tokens=16)

    if (not dist.is_initialized()) or dist.get_rank() == 0:
        print("[infer] generated ids:", out.tolist())


def _parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Multi-Head LatentMoE + Head Parallel demo")
    p.add_argument("--demo", choices=["train", "infer"], default="train")
    p.add_argument("--dp_size", type=int, default=1)
    p.add_argument("--hp_size", type=int, default=1)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.demo == "train":
        train_demo(dp_size=args.dp_size, hp_size=args.hp_size, steps=args.steps, seed=args.seed, device=args.device)
    else:
        infer_demo(dp_size=args.dp_size, hp_size=args.hp_size, seed=args.seed, device=args.device)

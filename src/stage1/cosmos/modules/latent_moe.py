"""
latent_moe.py
=============

Standalone **LatentMoE** implementation (arXiv:2601.18089) with a practical,
pure-PyTorch Expert-Parallel (EP) dispatch/collect path and a simple 2D
process-mesh ("meshgrid") helper.

This file is meant to be **drop-in**: you can import `LatentMoELayer` as a
feed-forward replacement inside a Transformer block, or run this file as a
script to execute the included training / inference demos.

Key features
------------
- LatentMoE layer with:
  - routed experts computed in **latent dimension** `d_latent` (ℓ in the paper)
  - router computed in original hidden dimension `d_model` (d in the paper)
  - optional **shared experts** computed in `d_model`
- Expert Parallel (EP) implementation with:
  - token replication (top-k)
  - variable-size **all_to_all_single** dispatch and return
  - deterministic packing (sorted by destination rank)
- 2D process mesh helper (DP × EP) for "meshgrid" style distributed setups
- Minimal toy decoder-only LM + training loop + greedy generation demo
  (pure-PyTorch, no external datasets needed)

⚠️ Notes (scope & engineering reality)
-------------------------------------
- This is a **reference implementation** optimized for correctness and clarity,
  not peak performance.
- Modern production MoE stacks fuse permutation, grouped GEMM, overlap comm/comp,
  and implement load-balancing carefully. You can still use this as a working
  baseline to validate ideas and extend it.

---------------------------------------------------------------------------
Quickstart (single process)
---------------------------------------------------------------------------
>>> import torch
>>> from latent_moe import LatentMoEConfig, LatentMoELayer
>>>
>>> x = torch.randn(2, 128, 256)  # (batch, seq, d_model)
>>> cfg = LatentMoEConfig(
...     d_model=256,
...     d_latent=64,
...     d_hidden=128,
...     num_experts=8,
...     top_k=2,
...     num_shared_experts=1,
... )
>>> moe = LatentMoELayer(cfg)
>>> y = moe(x)
>>> y.shape
torch.Size([2, 128, 256])

---------------------------------------------------------------------------
Distributed demo (CPU, gloo)
---------------------------------------------------------------------------
Run a tiny training loop with EP=2 ranks:

    torchrun --standalone --nproc_per_node=2 latent_moe.py --demo train --dp_size 1 --ep_size 2

Run a tiny generation demo:

    torchrun --standalone --nproc_per_node=2 latent_moe.py --demo infer --dp_size 1 --ep_size 2

You can set dp_size>1 to replicate the model across DP replicas; in that case,
gradients for each EP shard are all-reduced across the DP group automatically.

---------------------------------------------------------------------------
Public API
---------------------------------------------------------------------------
- ProcessMesh2D: creates DP and EP process groups from a 2D "meshgrid"
- LatentMoEConfig: dataclass config
- LatentMoELayer: the MoE feed-forward module
- TinyMoELM: tiny decoder-only LM using LatentMoE blocks
- train_demo / infer_demo: runnable demos

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import math
import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Distributed "meshgrid" helper
# -----------------------------


class ProcessMesh2D:
    """
    A lightweight 2D process-mesh helper.

    We assume `world_size == dp_size * mp_size` and map ranks to coordinates:

        dp_rank = rank // mp_size
        mp_rank = rank % mp_size

    Then:
      - `mp_group` (a.k.a. model-parallel group) = ranks with same dp_rank
      - `dp_group` (data-parallel group)         = ranks with same mp_rank

    For LatentMoE:
      - interpret mp_size as EP (expert-parallel) size.

    This is intentionally simple and avoids `DeviceMesh` / DTensor dependencies.
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

        # Create groups if distributed, else use WORLD.
        if dist.is_initialized() and world > 1:
            # mp groups: rows (same dp_rank)
            for d in range(dp_size):
                ranks = [d * mp_size + i for i in range(mp_size)]
                self._mp_groups.append(dist.new_group(ranks=ranks, backend=backend_hint))
            # dp groups: columns (same mp_rank)
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

    def dp_world_size(self) -> int:
        return self.dp_size

    def dp_rank_in_group(self) -> int:
        # rank within dp_group is dp_rank (since group = ranks with same mp_rank).
        return self.dp_rank

    def all_reduce_grads(self, module: nn.Module, average: bool = True) -> None:
        """
        Synchronize gradients for a 2D DP×MP mesh.

        We treat parameters as either:
          1) **MP-sharded** (exist only on a subset of MP ranks, e.g. experts/heads shards)
             -> reduce across the DP group only (replicas of the same shard)
          2) **Replicated across MP ranks** (the usual case: embeddings, attention, routers, projections, etc.)
             -> reduce across MP group *and* DP group (equivalent to a world all-reduce on the 2D mesh)

        Mark a parameter as MP-sharded by setting:
            param._mesh_mp_sharded = True

        If torch.distributed is not initialized, this is a no-op.
        """
        if not dist.is_initialized():
            return

        # Groups may be None in single-process mode.
        mp_group = self.mp_group
        dp_group = self.dp_group

        for p in module.parameters():
            if p.grad is None:
                continue

            is_mp_sharded = bool(getattr(p, "_mesh_mp_sharded", False))

            if is_mp_sharded:
                # Only DP replicas share this shard.
                if self.dp_size > 1 and dp_group is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=dp_group)
                    if average:
                        p.grad.div_(self.dp_size)
                continue

            # Replicated param: all-reduce across the full 2D mesh
            if self.mp_size > 1 and mp_group is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=mp_group)
            if self.dp_size > 1 and dp_group is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=dp_group)

            if average:
                p.grad.div_(self.dp_size * self.mp_size)


# -----------------------------
# Core MoE components
# -----------------------------


@dataclass
class LatentMoEConfig:
    """
    Configuration for LatentMoE.

    Parameters correspond to the paper's notation:
      - d_model  : model hidden size (d)
      - d_latent : latent size used inside routed experts (ℓ)
      - d_hidden : expert MLP hidden size (m)
      - num_experts : number of routed experts (N')
      - top_k       : number of active experts per token (K')
      - num_shared_experts : number of dense/shared experts (S)

    The paper describes two scaling recipes:
      - ℓ-MoEeff: increase N' while keeping K' fixed -> cheaper
      - ℓ-MoEacc: increase both N' and K' by d/ℓ -> accuracy at ~constant cost

    In this implementation, you set `num_experts` and `top_k` explicitly.
    """

    d_model: int
    d_latent: int
    d_hidden: int
    num_experts: int
    top_k: int
    num_shared_experts: int = 0
    dropout: float = 0.0

    # Router options
    router_fp32: bool = True
    router_jitter_noise: float = 0.0  # e.g. 1e-2

    # Expert MLP options
    activation: str = "swiglu"  # "swiglu" | "gelu" | "relu"
    expert_bias: bool = False


def _activation_fn(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    name = name.lower()
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
    if name == "silu":
        return F.silu
    raise ValueError(f"Unknown activation: {name}")


class TopKRouter(nn.Module):
    """
    Simple top-k router. Returns (indices, weights) of shape (T, k).

    Router logits are computed as x @ W_r^T (implemented with nn.Linear).
    Weights are softmax over the top-k logits.
    """

    def __init__(self, d_model: int, num_experts: int, fp32: bool = True, jitter_noise: float = 0.0):
        super().__init__()
        self.num_experts = num_experts
        self.fp32 = fp32
        self.jitter_noise = float(jitter_noise)
        self.linear = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (T, d_model)
            top_k: int

        Returns:
            topk_idx: (T, k) int64
            topk_w  : (T, k) float32/float16 (same dtype as x)
        """
        T, _ = x.shape
        k = min(int(top_k), self.num_experts)
        logits = self.linear(x.float() if self.fp32 else x)

        if self.jitter_noise > 0:
            logits = logits + torch.randn_like(logits) * self.jitter_noise

        topk_val, topk_idx = torch.topk(logits, k=k, dim=-1)  # (T,k)
        topk_w = F.softmax(topk_val, dim=-1).to(dtype=x.dtype)
        return topk_idx, topk_w


class LatentExpertMLP(nn.Module):
    """
    One routed expert operating in latent dimension.

    Default is SwiGLU-style:
        y = W2( silu(Wg x) * (W1 x) )

    Alternate:
        activation in {"gelu", "relu"}:
        y = W2( act(W1 x) )
    """

    def __init__(self, d_in: int, d_hidden: int, d_out: int, activation: str = "swiglu", bias: bool = False):
        super().__init__()
        self.activation = activation.lower()
        if self.activation == "swiglu":
            self.w1 = nn.Linear(d_in, d_hidden, bias=bias)
            self.wg = nn.Linear(d_in, d_hidden, bias=bias)
            self.w2 = nn.Linear(d_hidden, d_out, bias=bias)
        else:
            act = _activation_fn(self.activation)
            self.act = act
            self.w1 = nn.Linear(d_in, d_hidden, bias=bias)
            self.w2 = nn.Linear(d_hidden, d_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            return self.w2(F.silu(self.wg(x)) * self.w1(x))
        return self.w2(self.act(self.w1(x)))


class DenseSharedFFN(nn.Module):
    """
    Dense FFN used as a "shared expert" operating in d_model.
    """

    def __init__(
        self, d_model: int, d_hidden: int, activation: str = "swiglu", bias: bool = False, dropout: float = 0.0
    ):
        super().__init__()
        self.ffn = LatentExpertMLP(d_model, d_hidden, d_model, activation=activation, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.ffn(x))


def _all_gather_counts_2d(counts: torch.Tensor, group: Optional[dist.ProcessGroup]) -> torch.Tensor:
    """
    Gather a 1D tensor (len=group_size) from all ranks in `group`.
    Returns a (group_size, group_size) tensor where row r is counts from rank r.
    """
    if not dist.is_initialized():
        return counts.unsqueeze(0)

    world = dist.get_world_size(group=group)
    out = [torch.empty_like(counts) for _ in range(world)]
    dist.all_gather(out, counts, group=group)
    return torch.stack(out, dim=0)


def _all_to_all_variable(
    send: torch.Tensor,
    send_counts: Sequence[int],
    recv_counts: Sequence[int],
    group: Optional[dist.ProcessGroup],
) -> torch.Tensor:
    """
    Variable-size all_to_all_single along dim0.
    """
    if not dist.is_initialized():
        return send

    if sum(send_counts) != send.shape[0]:
        raise ValueError(f"send_counts sum {sum(send_counts)} != send.shape[0] {send.shape[0]}")
    out_shape = (sum(recv_counts),) + send.shape[1:]
    recv = torch.empty(out_shape, device=send.device, dtype=send.dtype)
    dist.all_to_all_single(
        output=recv,
        input=send,
        output_split_sizes=list(map(int, recv_counts)),
        input_split_sizes=list(map(int, send_counts)),
        group=group,
    )
    return recv


class LatentMoELayer(nn.Module):
    """
    LatentMoE feed-forward layer.

    Forward shape:
        x: (B, T, d_model) -> y: (B, T, d_model)

    Distributed EP support:
        - If `mesh` is provided and mesh.mp_size>1 (interpreted as EP),
          experts are sharded across EP ranks inside each DP replica.
        - Routed tokens are dispatched with all_to_all_single, experts computed
          locally, and results returned with another all_to_all_single.

    Expected expert sharding:
        num_experts % ep_size == 0
        each EP rank owns a contiguous shard of experts.
    """

    def __init__(self, cfg: LatentMoEConfig, mesh: Optional[ProcessMesh2D] = None):
        super().__init__()
        self.cfg = cfg
        self.mesh = mesh

        self.router = TopKRouter(
            cfg.d_model, cfg.num_experts, fp32=cfg.router_fp32, jitter_noise=cfg.router_jitter_noise
        )

        # Down/up projection (routed path)
        self.down = nn.Linear(cfg.d_model, cfg.d_latent, bias=False)
        self.up = nn.Linear(cfg.d_latent, cfg.d_model, bias=False)

        # Shared experts (dense path, in d_model)
        self.shared: nn.ModuleList = nn.ModuleList(
            [
                DenseSharedFFN(
                    cfg.d_model, cfg.d_hidden, activation=cfg.activation, bias=cfg.expert_bias, dropout=cfg.dropout
                )
                for _ in range(cfg.num_shared_experts)
            ]
        )

        # Expert sharding
        ep_size = self.ep_world_size
        if cfg.num_experts % ep_size != 0:
            raise ValueError(f"num_experts={cfg.num_experts} must be divisible by ep_size={ep_size}")
        self.local_experts = cfg.num_experts // ep_size
        self.expert_offset = self.ep_rank * self.local_experts

        self.experts: nn.ModuleList = nn.ModuleList(
            [
                LatentExpertMLP(
                    cfg.d_latent, cfg.d_hidden, cfg.d_latent, activation=cfg.activation, bias=cfg.expert_bias
                )
                for _ in range(self.local_experts)
            ]
        )
        # Mark expert parameters as MP-sharded (do NOT all-reduce across MP ranks)
        for _p in self.experts.parameters():
            setattr(_p, "_mesh_mp_sharded", True)

        self.dropout = nn.Dropout(cfg.dropout)

    @property
    def ep_world_size(self) -> int:
        if self.mesh is None:
            return 1
        return self.mesh.mp_world_size()

    @property
    def ep_rank(self) -> int:
        if self.mesh is None:
            return 0
        return self.mesh.mp_rank_in_group()

    @property
    def ep_group(self) -> Optional[dist.ProcessGroup]:
        if self.mesh is None:
            return None
        return self.mesh.mp_group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            y: (B, T, d_model)
        """
        B, T, D = x.shape
        cfg = self.cfg
        x_flat = x.reshape(B * T, D)

        # Shared experts (dense)
        shared_out = 0.0
        for se in self.shared:
            shared_out = shared_out + se(x_flat)

        # Routing in d_model (paper routes in d, then projects to ℓ for expert compute)
        topk_idx, topk_w = self.router(x_flat, cfg.top_k)  # (Ntok,k), (Ntok,k)

        # Project to latent ℓ for expert compute
        x_lat = self.down(x_flat)  # (Ntok, d_latent)

        # Routed expert compute (distributed or local)
        y_lat = self._routed_forward_ep(x_lat, topk_idx, topk_w)  # (Ntok, d_latent)

        y = self.up(y_lat) + shared_out
        y = self.dropout(y)
        return y.reshape(B, T, D)

    def _routed_forward_ep(
        self,
        x_lat: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_w: torch.Tensor,
    ) -> torch.Tensor:
        """
        EP-routed expert forward.

        x_lat:     (Ntok, d_latent)
        topk_idx:  (Ntok, k) global expert ids
        topk_w:    (Ntok, k) weights
        """
        device = x_lat.device
        Ntok, d_lat = x_lat.shape
        k = topk_idx.shape[1]

        # Create per-assignment buffers (token replicated k times)
        token_ids = (
            torch.arange(Ntok, device=device, dtype=torch.int64).unsqueeze(1).expand(Ntok, k).reshape(-1)
        )  # (Ntok*k)
        expert_ids = topk_idx.reshape(-1).to(torch.int64)  # (Ntok*k)
        weights = topk_w.reshape(-1)  # (Ntok*k)
        x_rep = x_lat.unsqueeze(1).expand(Ntok, k, d_lat).reshape(-1, d_lat)  # (Ntok*k, d_lat)

        # Determine destination EP rank for each assignment
        local_experts = self.local_experts
        dest = torch.div(expert_ids, local_experts, rounding_mode="floor")  # (Ntok*k)
        local_e = expert_ids - dest * local_experts  # (Ntok*k)

        ep_size = self.ep_world_size
        if ep_size == 1 or not dist.is_initialized():
            # Local fast-path
            out_rep = torch.empty_like(x_rep)
            for e in range(local_experts):
                m = local_e == e
                if m.any():
                    out_rep[m] = self.experts[e](x_rep[m])
            out_rep = out_rep * weights.unsqueeze(-1)
            y_lat = torch.zeros((Ntok, d_lat), device=device, dtype=x_lat.dtype)
            y_lat.index_add_(0, token_ids, out_rep)
            return y_lat

        # Sort assignments by destination to pack send buffers deterministically
        order = torch.argsort(dest)
        dest = dest[order]
        token_ids = token_ids[order]
        local_e = local_e[order]
        weights = weights[order]
        x_rep = x_rep[order]

        # Compute send_counts per destination rank
        send_counts_t = torch.bincount(dest, minlength=ep_size).to(torch.int64)
        send_counts = send_counts_t.tolist()

        # Exchange counts to determine recv_counts
        gathered = _all_gather_counts_2d(send_counts_t, group=self.ep_group)  # (ep_size, ep_size)
        my_rank = self.ep_rank
        recv_counts_t = gathered[:, my_rank].contiguous()
        recv_counts = recv_counts_t.tolist()

        # Dispatch
        x_recv = _all_to_all_variable(x_rep, send_counts, recv_counts, group=self.ep_group)  # (Nrecv, d_lat)
        tok_recv = _all_to_all_variable(token_ids, send_counts, recv_counts, group=self.ep_group)  # (Nrecv,)
        w_recv = _all_to_all_variable(weights, send_counts, recv_counts, group=self.ep_group)  # (Nrecv,)
        e_recv = _all_to_all_variable(local_e, send_counts, recv_counts, group=self.ep_group)  # (Nrecv,)

        # Local expert compute on received assignments
        out_recv = torch.empty_like(x_recv)
        for e in range(local_experts):
            m = e_recv == e
            if m.any():
                out_recv[m] = self.experts[e](x_recv[m])

        out_recv = out_recv * w_recv.unsqueeze(-1)

        # Return to source ranks (reverse all-to-all)
        out_back = _all_to_all_variable(out_recv, recv_counts, send_counts, group=self.ep_group)  # (Nsend, d_lat)
        tok_back = _all_to_all_variable(tok_recv, recv_counts, send_counts, group=self.ep_group)  # (Nsend,)

        # Aggregate back into token outputs
        y_lat = torch.zeros((Ntok, d_lat), device=device, dtype=x_lat.dtype)
        y_lat.index_add_(0, tok_back, out_back)
        return y_lat


# -----------------------------
# Tiny LM demo (training & inference)
# -----------------------------


class CausalSelfAttention(nn.Module):
    """
    Minimal causal self-attention using scaled_dot_product_attention (PyTorch).
    """

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
        qkv = self.qkv(x)  # (B,T,3D)
        q, k, v = qkv.chunk(3, dim=-1)

        def reshape_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,T,dh)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        # SDPA expects (B,H,T,dh)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """
    Tiny Transformer block: RMSNorm -> Attn -> Residual -> RMSNorm -> LatentMoE -> Residual
    """

    def __init__(self, d_model: int, n_heads: int, moe: LatentMoELayer, dropout: float = 0.0):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, dropout=dropout)
        self.moe = moe
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.moe(self.norm2(x)))
        return x


class TinyMoELM(nn.Module):
    """
    Tiny decoder-only LM that uses LatentMoE as the FFN.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        moe_cfg: LatentMoEConfig,
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
            moe = LatentMoELayer(moe_cfg, mesh=mesh)
            blocks.append(TransformerBlock(d_model, n_heads, moe=moe, dropout=dropout))
        self.blocks = nn.ModuleList(blocks)

        self.norm_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx: (B,T) int64 token ids
        returns logits: (B,T,V)
        """
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
            logits = self(idx_cond)  # (B,T,V)
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.argmax(probs, dim=-1, keepdim=True)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# -----------------------------
# Script entrypoints
# -----------------------------


def _seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def init_distributed_from_env() -> None:
    """
    Initialize torch.distributed from torchrun env vars if needed.
    """
    if dist.is_initialized():
        return
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)


def train_demo(dp_size: int, ep_size: int, steps: int = 50, seed: int = 123, device: Optional[str] = None) -> None:
    """
    Tiny training demo (random data next-token prediction).
    """
    init_distributed_from_env()
    _seed_all(seed + (dist.get_rank() if dist.is_initialized() else 0))

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    mesh = ProcessMesh2D(dp_size=dp_size, mp_size=ep_size)

    vocab = 256
    d_model = 128
    n_heads = 4
    n_layers = 2
    seq_len = 64
    batch = 8

    moe_cfg = LatentMoEConfig(
        d_model=d_model,
        d_latent=32,
        d_hidden=128,
        num_experts=8,
        top_k=2,
        num_shared_experts=1,
        dropout=0.0,
        activation="swiglu",
    )
    model = TinyMoELM(
        vocab_size=vocab,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        moe_cfg=moe_cfg,
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

        # Sync grads across DP replicas
        mesh.all_reduce_grads(model, average=True)

        opt.step()

        if (not dist.is_initialized()) or dist.get_rank() == 0:
            if step % 10 == 0 or step == steps - 1:
                print(f"[train] step={step:04d} loss={loss.item():.4f}")


def infer_demo(dp_size: int, ep_size: int, seed: int = 123, device: Optional[str] = None) -> None:
    """
    Tiny greedy generation demo.
    """
    init_distributed_from_env()
    _seed_all(seed + (dist.get_rank() if dist.is_initialized() else 0))

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    mesh = ProcessMesh2D(dp_size=dp_size, mp_size=ep_size)

    vocab = 64
    d_model = 128
    n_heads = 4
    n_layers = 2
    seq_len = 64

    moe_cfg = LatentMoEConfig(
        d_model=d_model,
        d_latent=32,
        d_hidden=128,
        num_experts=8,
        top_k=2,
        num_shared_experts=1,
        dropout=0.0,
        activation="swiglu",
    )

    model = TinyMoELM(
        vocab_size=vocab,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        moe_cfg=moe_cfg,
        mesh=mesh,
        dropout=0.0,
        max_seq_len=seq_len,
    ).to(device_t)

    # No training here; just show that generation runs end-to-end.
    prompt = torch.zeros((1, 8), dtype=torch.long, device=device_t)
    out = model.generate(prompt, max_new_tokens=16)

    if (not dist.is_initialized()) or dist.get_rank() == 0:
        print("[infer] generated ids:", out.tolist())


def _parse_args():
    import argparse

    p = argparse.ArgumentParser(description="LatentMoE standalone demo")
    p.add_argument("--demo", choices=["train", "infer"], default="train")
    p.add_argument("--dp_size", type=int, default=1)
    p.add_argument("--ep_size", type=int, default=1)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.demo == "train":
        train_demo(dp_size=args.dp_size, ep_size=args.ep_size, steps=args.steps, seed=args.seed, device=args.device)
    else:
        infer_demo(dp_size=args.dp_size, ep_size=args.ep_size, seed=args.seed, device=args.device)

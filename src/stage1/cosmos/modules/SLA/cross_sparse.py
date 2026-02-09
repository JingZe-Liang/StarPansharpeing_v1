import torch
import torch.nn as nn

from .core import _get_feature_map, _linear_attention
from .kernel_cross import _cross_attention
from .utils import mean_pool


def get_block_map_cross(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    topk_ratio: float,
    BLKQ: int = 64,
    BLKK: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    arg_k = k - torch.mean(k, dim=-2, keepdim=True)  # smooth-k technique in SageAttention
    pooled_qblocks = mean_pool(q, BLKQ)
    pooled_kblocks = mean_pool(arg_k, BLKK)
    pooled_score = pooled_qblocks @ pooled_kblocks.transpose(-1, -2)

    n_blocks = pooled_score.shape[-1]
    topk = min(n_blocks, max(1, int(topk_ratio * n_blocks)))
    lut = torch.topk(pooled_score, topk, dim=-1, sorted=False).indices

    sparse_map = torch.zeros_like(pooled_score, dtype=torch.int8)
    sparse_map.scatter_(-1, lut, 1)
    return sparse_map, lut, topk


class SparseLinearCrossAttention(nn.Module):
    def __init__(
        self,
        head_dim: int,
        topk: float,
        *,
        feature_map: str = "softmax",
        BLKQ: int = 64,
        BLKK: int = 64,
        use_bf16: bool = True,
        tie_feature_map_qk: bool = True,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.dtype = torch.bfloat16 if use_bf16 else torch.float16
        self.topk = topk
        self.BLKQ = BLKQ
        self.BLKK = BLKK
        self.eps = eps

        self.feature_map_q, self.feature_map_k = _get_feature_map(feature_map, tie_feature_map_qk=tie_feature_map_qk)
        self.proj_l = nn.Linear(head_dim, head_dim, dtype=torch.float32)

        self.init_weights_()

    def init_weights_(self) -> None:
        with torch.no_grad():
            nn.init.zeros_(self.proj_l.weight)
            nn.init.zeros_(self.proj_l.bias)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        q : torch.Tensor
            Queries of shape (B, H, Lq, D)
        k : torch.Tensor
            Keys of shape (B, H, Lk, D)
        v : torch.Tensor
            Values of shape (B, H, Lk, D)
        """
        if q.shape[:2] != k.shape[:2] or k.shape[:2] != v.shape[:2]:
            raise ValueError(f"Expected matching (B, H), got {q.shape[:2]=}, {k.shape[:2]=}, {v.shape[:2]=}")
        if k.shape[-2] != v.shape[-2]:
            raise ValueError(f"Expected k/v to share the same length, got {k.shape[-2]=} and {v.shape[-2]=}")
        if q.shape[-1] != k.shape[-1] or k.shape[-1] != v.shape[-1]:
            raise ValueError(f"Expected matching head_dim, got {q.shape[-1]=}, {k.shape[-1]=}, {v.shape[-1]=}")

        dtype = q.dtype

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        sparse_map, lut, real_topk = get_block_map_cross(q, k, topk_ratio=self.topk, BLKQ=self.BLKQ, BLKK=self.BLKK)

        q16 = q.to(self.dtype)
        k16 = k.to(self.dtype)
        v16 = v.to(self.dtype)
        o_s = _cross_attention.apply(q16, k16, v16, sparse_map, lut, real_topk, self.BLKQ, self.BLKK)

        qf = self.feature_map_q(q16).contiguous().to(self.dtype)
        kf = self.feature_map_k(k16).contiguous().to(self.dtype)
        o_l = _linear_attention(qf.float(), kf.float(), v16.float(), eps=self.eps).to(self.dtype)

        with torch.amp.autocast("cuda", dtype=self.dtype):
            o_l = self.proj_l(o_l)
        o = (o_s + o_l).to(dtype)
        return o

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch.distributed as dist

from .hadamard import get_hadamard_matrix


# -------------------------
# shape utils: BNC / BCHW
# -------------------------


def is_bchw(x: torch.Tensor) -> bool:
    return x.dim() == 4


def is_bnc(x: torch.Tensor) -> bool:
    return x.dim() == 3


def flatten_samples(x: torch.Tensor) -> torch.Tensor:
    """
    Convert x to (N, C) where samples are rows.
    Accepts:
      - BNC  -> (B*N, C)
      - BCHW -> (B*H*W, C)
    """
    if is_bnc(x):
        B, N, C = x.shape
        return x.reshape(B * N, C)
    elif is_bchw(x):
        B, C, H, W = x.shape
        return x.permute(0, 2, 3, 1).contiguous().reshape(B * H * W, C)
    else:
        raise ValueError(f"Only support BNC or BCHW, got shape={tuple(x.shape)}")


def apply_phi_s(x: torch.Tensor, mean: torch.Tensor, W_ship: torch.Tensor) -> torch.Tensor:
    C = mean.numel()
    if is_bnc(x):
        if x.shape[-1] != C:
            raise ValueError(f"Expected channel={C}, got shape={tuple(x.shape)}")
        return (x - mean.view(1, 1, C)) @ W_ship.t()
    if is_bchw(x):
        if x.shape[1] != C:
            raise ValueError(f"Expected channel={C}, got shape={tuple(x.shape)}")
        x_hw = x.permute(0, 2, 3, 1).contiguous()
        y_hw = (x_hw - mean.view(1, 1, 1, C)) @ W_ship.t()
        return y_hw.permute(0, 3, 1, 2).contiguous()
    raise ValueError(f"Only support BNC or BCHW, got shape={tuple(x.shape)}")


def _dist_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _dist_rank() -> int:
    return dist.get_rank() if _dist_is_initialized() else 0


def _dist_world_size() -> int:
    return dist.get_world_size() if _dist_is_initialized() else 1


def _flatten_to_nxc_fp64(x: torch.Tensor) -> torch.Tensor:
    X = flatten_samples(x)
    if X.dtype != torch.float64:
        X = X.to(dtype=torch.float64)
    return X


@torch.no_grad()
def _mean_cov_from_sums(
    *,
    n: int,
    sum_x: torch.Tensor,
    sum_xxt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if n < 2:
        raise ValueError(f"Need n>=2 to compute covariance, got n={n}")

    mean = sum_x / float(n)
    cov = (sum_xxt - float(n) * (mean[:, None] @ mean[None, :])) / float(n - 1)
    return mean, cov


@torch.no_grad()
def _compute_w_ship_from_cov(
    *,
    mean: torch.Tensor,
    cov: torch.Tensor,
    hadamard_fn: Callable[..., torch.Tensor],
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    e, U = torch.linalg.eigh(cov)
    e = torch.clamp(e, min=0.0)

    H = hadamard_fn(e.shape[0], device=U.device, dtype=U.dtype)
    alpha = torch.rsqrt(e.mean() + eps)
    W_ship = alpha * (H @ U.T)
    return mean, W_ship


def _layer_key(name: str, layer_idx: int) -> str:
    return f"{name}__L{layer_idx}"


def _sanitize_buffer_token(token: str) -> str:
    sanitized = "".join(ch if ch.isalnum() else "_" for ch in token)
    return sanitized or "key"


def _unique_buffer_name(prefix: str, token: str, used: set[str]) -> str:
    base = f"{prefix}_{_sanitize_buffer_token(token)}"
    name = base
    idx = 1
    while name in used:
        name = f"{base}_{idx}"
        idx += 1
    used.add(name)
    return name


def _ensure_feat_dim(name: str, feat: torch.Tensor, dim: int) -> None:
    if is_bnc(feat):
        if int(feat.shape[-1]) != dim:
            raise ValueError(f"{name}: expected channel={dim}, got shape={tuple(feat.shape)}")
    elif is_bchw(feat):
        if int(feat.shape[1]) != dim:
            raise ValueError(f"{name}: expected channel={dim}, got shape={tuple(feat.shape)}")
    else:
        raise ValueError(f"{name}: only supports BNC/BCHW, got shape={tuple(feat.shape)}")


def _ensure_list_len(name: str, feats: list[Tensor], dims: list[int]) -> None:
    if len(feats) != len(dims):
        raise ValueError(f"{name}: expected {len(dims)} layers, got {len(feats)}")


# -------------------------
# PHI-S matrix estimation (matches your repo get_phi_s_matrix)
# -------------------------


@torch.no_grad()
def get_phi_s_matrix_from_features(
    X: torch.Tensor,
    hadamard_fn: Callable[..., torch.Tensor],  # e.g. your get_hadamard_matrix wrapper
    eps: float = 1e-6,
):
    """
    X: BNC or BCHW, channel dim = C
    Returns:
      mean_val: (C,)
      W_ship:   (C,C)  == alpha * (H @ U.T)
    """
    X2 = flatten_samples(X)  # (N, C)
    mean_val = X2.mean(dim=0)  # (C,)
    Xc = X2 - mean_val  # (N, C)

    cov = torch.cov(Xc.T)  # (C, C)
    e, U = torch.linalg.eigh(cov)  # e:(C,), U:(C,C)
    e = torch.clamp(e, min=0.0)

    H = hadamard_fn(e.shape[0], device=U.device, dtype=U.dtype)  # (C,C)
    normalizer = torch.sqrt(e.mean() + eps)
    inv_normalizer = 1.0 / normalizer

    W_ship = inv_normalizer * (H @ U.T)  # (C,C)
    return mean_val, W_ship


# -------------------------
# Full PHI-S distillation module
# -------------------------


class PhiSDistillLoss(nn.Module):
    """
    Multi-teacher PHI-S distillation (student-side heads).

    Usage:
      loss_mod = PhiSDistillLoss(teacher_dims, hadamard_fn)
      # 1) estimate PHI-S stats for each teacher (teacher space):
      loss_mod.reset_phi_stats()
      for batches:
          loss_mod.update_phi_stats(teacher_feats_dict)
      loss_mod.finalize_phi_from_stats()

      # 2) training:
      # student_feats_dict are produced by external teacher-specific heads
      loss = loss_mod(student_feats_dict, teacher_feats_dict)
    """

    def __init__(
        self,
        teacher_dims: dict[str, int],  # {"t1": Ct1, "t2": Ct2, ...}
        hadamard_fn: Callable[..., torch.Tensor],
        loss_type: str = "mse",  # "mse" | "smoothl1"
        eps: float = 1e-6,
        teacher_weights: dict | None = None,
    ):
        super().__init__()
        self.teacher_dims = dict(teacher_dims)
        self.hadamard_fn = hadamard_fn
        self.loss_type = loss_type
        self.eps = eps
        self.teacher_weights = teacher_weights or {k: 1.0 for k in self.teacher_dims.keys()}

        # buffers for mean and W_ship per teacher (teacher space)
        self._ready = False
        self._mean_buf_names: dict[str, str] = {}
        self._w_buf_names: dict[str, str] = {}
        used_buffer_names: set[str] = set()
        for k, dim in self.teacher_dims.items():
            mean_name = _unique_buffer_name("phi_mean", k, used_buffer_names)
            w_name = _unique_buffer_name("phi_w_ship", k, used_buffer_names)
            self.register_buffer(mean_name, torch.zeros(dim), persistent=True)
            self.register_buffer(w_name, torch.eye(dim), persistent=True)
            self._mean_buf_names[k] = mean_name
            self._w_buf_names[k] = w_name

        # streaming stats (official-like): sum and sum(x^T x) in teacher space
        self._stat_n: dict[str, int] = {k: 0 for k in self.teacher_dims.keys()}
        self._stat_sum: dict[str, torch.Tensor] = {}
        self._stat_xxt: dict[str, torch.Tensor] = {}

    def _mean_buf(self, key: str) -> torch.Tensor:
        return self.get_buffer(self._mean_buf_names[key])

    def _w_buf(self, key: str) -> torch.Tensor:
        return self.get_buffer(self._w_buf_names[key])

    @torch.no_grad()
    def reset_phi(self):
        self._ready = False
        self.reset_phi_stats()

    @torch.no_grad()
    def reset_phi_stats(self) -> None:
        self._ready = False
        self._stat_n = {k: 0 for k in self.teacher_dims.keys()}
        self._stat_sum = {}
        self._stat_xxt = {}

    @torch.no_grad()
    def update_phi(self, teacher_feats: dict[str, torch.Tensor]) -> None:
        self.update_phi_stats(teacher_feats)

    @torch.no_grad()
    def update_phi_stats(self, teacher_feats: dict[str, torch.Tensor]) -> None:
        for k, feat in teacher_feats.items():
            if k not in self.teacher_dims:
                raise KeyError(f"Unknown teacher key: {k}")
            _ensure_feat_dim(f"Teacher={k}", feat, self.teacher_dims[k])

            X = _flatten_to_nxc_fp64(feat.detach())

            n = int(X.shape[0])
            if n == 0:
                continue

            sum_x = X.sum(dim=0)
            sum_xxt = X.T @ X

            self._stat_n[k] = self._stat_n.get(k, 0) + n
            if k not in self._stat_sum:
                self._stat_sum[k] = sum_x
                self._stat_xxt[k] = sum_xxt
            else:
                self._stat_sum[k] = self._stat_sum[k] + sum_x
                self._stat_xxt[k] = self._stat_xxt[k] + sum_xxt

    @torch.no_grad()
    def finalize_phi(self, *, distributed: bool = True) -> None:
        self.finalize_phi_from_stats(distributed=distributed)

    @torch.no_grad()
    def finalize_phi_from_stats(self, *, distributed: bool = True) -> None:
        use_dist = bool(distributed) and _dist_is_initialized() and _dist_world_size() > 1

        for k in self.teacher_dims.keys():
            if self._stat_n.get(k, 0) == 0 or k not in self._stat_sum or k not in self._stat_xxt:
                raise RuntimeError(f"No stats for teacher={k}. Call reset_phi_stats()->update_phi_stats() first.")

            n = int(self._stat_n[k])
            sum_x = self._stat_sum[k]
            sum_xxt = self._stat_xxt[k]

            if use_dist:
                n_t = torch.tensor(n, device=sum_x.device, dtype=torch.long)
                dist.all_reduce(n_t, op=dist.ReduceOp.SUM)
                dist.all_reduce(sum_x, op=dist.ReduceOp.SUM)
                dist.all_reduce(sum_xxt, op=dist.ReduceOp.SUM)
                n = int(n_t.item())

            mean64, cov64 = _mean_cov_from_sums(n=n, sum_x=sum_x, sum_xxt=sum_xxt)
            mean64, W_ship64 = _compute_w_ship_from_cov(
                mean=mean64, cov=cov64, hadamard_fn=self.hadamard_fn, eps=self.eps
            )

            mean_buf = self._mean_buf(k)
            w_buf = self._w_buf(k)
            mean_buf.copy_(mean64.to(dtype=mean_buf.dtype))
            w_buf.copy_(W_ship64.to(dtype=w_buf.dtype))

        self._ready = True

    def requires_loader(self, cache_path: str | Path) -> bool:
        return (not self._ready) and (not Path(cache_path).exists())

    def phi_state_dict(self) -> dict[str, Any]:
        return {
            "teacher_dims": dict(self.teacher_dims),
            "eps": float(self.eps),
            "mean": {k: self._mean_buf(k).detach().cpu() for k in self.teacher_dims.keys()},
            "W_ship": {k: self._w_buf(k).detach().cpu() for k in self.teacher_dims.keys()},
        }

    @torch.no_grad()
    def load_phi_from_cache(self, cache_path: str | Path, *, broadcast: bool = True) -> None:
        cache_path = Path(cache_path)
        use_dist = bool(broadcast) and _dist_is_initialized() and _dist_world_size() > 1

        if not use_dist:
            sd = torch.load(cache_path, map_location="cpu")
            self._load_phi_state_dict(sd)
            return

        sd: dict[str, Any] | None = None
        if _dist_rank() == 0:
            sd = torch.load(cache_path, map_location="cpu")

        for k in self.teacher_dims.keys():
            mean = self._mean_buf(k).detach().clone()
            W_ship = self._w_buf(k).detach().clone()

            if _dist_rank() == 0:
                assert sd is not None
                mean.copy_(sd["mean"][k].to(device=mean.device, dtype=mean.dtype))
                W_ship.copy_(sd["W_ship"][k].to(device=W_ship.device, dtype=W_ship.dtype))

            dist.broadcast(mean, src=0)
            dist.broadcast(W_ship, src=0)

            self._mean_buf(k).copy_(mean)
            self._w_buf(k).copy_(W_ship)

        self._ready = True

    def save_phi_to_cache(self, cache_path: str | Path) -> None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.phi_state_dict(), cache_path)

    @torch.no_grad()
    def initialize_phi(
        self,
        *,
        cache_path: str | Path,
        teacher_feats_iter: Any | None = None,
        num_batches: int | None = None,
        distributed_stats: bool = True,
        broadcast_on_load: bool = True,
    ) -> None:
        cache_path = Path(cache_path)

        if cache_path.exists():
            self.load_phi_from_cache(cache_path, broadcast=broadcast_on_load)
            return

        if teacher_feats_iter is None:
            raise ValueError("cache_path does not exist; teacher_feats_iter must be provided to estimate PHI-S stats.")

        self.reset_phi_stats()
        for i, teacher_feats in enumerate(teacher_feats_iter):
            if num_batches is not None and i >= num_batches:
                break
            self.update_phi_stats(teacher_feats)
        self.finalize_phi_from_stats(distributed=distributed_stats)

        if not _dist_is_initialized() or _dist_rank() == 0:
            self.save_phi_to_cache(cache_path)

    def _load_phi_state_dict(self, sd: dict[str, Any]) -> None:
        teacher_dims = sd.get("teacher_dims", {})
        if dict(teacher_dims) != dict(self.teacher_dims):
            raise ValueError(f"teacher_dims mismatch: cache={teacher_dims} vs module={self.teacher_dims}")

        mean_map: dict[str, torch.Tensor] = sd["mean"]
        w_map: dict[str, torch.Tensor] = sd["W_ship"]

        for k in self.teacher_dims.keys():
            mean_buf = self._mean_buf(k)
            w_buf = self._w_buf(k)
            mean_buf.copy_(mean_map[k].to(device=mean_buf.device, dtype=mean_buf.dtype))
            w_buf.copy_(w_map[k].to(device=w_buf.device, dtype=w_buf.dtype))

        self._ready = True

    @classmethod
    @torch.no_grad()
    def build_and_initialize(
        cls,
        *,
        teacher_dims: dict[str, int],
        hadamard_fn: Callable[..., torch.Tensor],
        cache_path: str | Path,
        teacher_feats_iter: Any | None = None,
        num_batches: int | None = None,
        loss_type: str = "mse",
        eps: float = 1e-6,
        teacher_weights: dict[str, float] | None = None,
        distributed_stats: bool = True,
        broadcast_on_load: bool = True,
    ) -> "PhiSDistillLoss":
        mod = cls(
            teacher_dims=teacher_dims,
            hadamard_fn=hadamard_fn,
            loss_type=loss_type,
            eps=eps,
            teacher_weights=teacher_weights,
        )
        mod.initialize_phi(
            cache_path=cache_path,
            teacher_feats_iter=teacher_feats_iter,
            num_batches=num_batches,
            distributed_stats=distributed_stats,
            broadcast_on_load=broadcast_on_load,
        )
        return mod

    def forward(self, student_feats: dict[str, torch.Tensor], teacher_feats: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        student_feats: dict[teacher_name] -> student-side projected feat (same shape as teacher feat)
        teacher_feats: dict[teacher_name] -> teacher feat (BNC or BCHW)
        """
        if not self._ready:
            raise RuntimeError(
                "PHI-S not ready. Run reset_phi_stats()->update_phi_stats()->finalize_phi_from_stats() first."
            )

        if not student_feats:
            raise ValueError("student_feats is empty")

        total = next(iter(student_feats.values())).new_zeros(())
        wsum = 0.0

        for k, tfeat in teacher_feats.items():
            if k not in student_feats:
                raise KeyError(f"Missing student feat for teacher={k}")

            w = float(self.teacher_weights.get(k, 1.0))
            wsum += w

            # PHI-S transform teacher target into standardized space
            mean = self._mean_buf(k)
            W_ship = self._w_buf(k)
            tphi = apply_phi_s(tfeat, mean, W_ship)

            sfeat = student_feats[k]
            if sfeat.shape != tphi.shape:
                raise ValueError(
                    f"Shape mismatch for teacher={k}: student={tuple(sfeat.shape)} vs tphi={tuple(tphi.shape)}"
                )

            if self.loss_type == "mse":
                loss = F.mse_loss(sfeat, tphi)
            elif self.loss_type == "smoothl1":
                loss = F.smooth_l1_loss(sfeat, tphi)
            else:
                raise ValueError("loss_type must be 'mse' or 'smoothl1'")

            total = total + w * loss

        return total / max(wsum, 1e-12)


class MultiLayersPhiSDistillLoss(nn.Module):
    """
    Multi-layer PHI-S distillation (student-side heads).

    student_feats/teacher_feats: dict[teacher_name] -> list[Tensor] (layer-aligned).
    """

    def __init__(
        self,
        teacher_dims: dict[str, list[int]],
        hadamard_fn: Callable[..., torch.Tensor],
        loss_type: str = "mse",
        eps: float = 1e-6,
        teacher_weights: dict[str, float] | None = None,
        layer_weights: dict[str, list[float]] | None = None,
    ) -> None:
        super().__init__()
        self.teacher_dims = {k: list(v) for k, v in teacher_dims.items()}
        self.hadamard_fn = hadamard_fn
        self.loss_type = loss_type
        self.eps = eps
        self.teacher_weights = teacher_weights or {k: 1.0 for k in self.teacher_dims.keys()}
        self.layer_weights = layer_weights

        self._ready = False
        self._mean_buf_names: dict[str, str] = {}
        self._w_buf_names: dict[str, str] = {}
        used_buffer_names: set[str] = set()
        for k, dims in self.teacher_dims.items():
            if not dims:
                raise ValueError(f"{k}: teacher_dims list is empty")
            for i, dim in enumerate(dims):
                logical_key = _layer_key(k, i)
                mean_name = _unique_buffer_name("phi_mean", logical_key, used_buffer_names)
                w_name = _unique_buffer_name("phi_w_ship", logical_key, used_buffer_names)
                self.register_buffer(mean_name, torch.zeros(dim), persistent=True)
                self.register_buffer(w_name, torch.eye(dim), persistent=True)
                self._mean_buf_names[logical_key] = mean_name
                self._w_buf_names[logical_key] = w_name

        self._stat_n: dict[str, list[int]] = {k: [0 for _ in v] for k, v in self.teacher_dims.items()}
        self._stat_sum: dict[str, list[torch.Tensor]] = {}
        self._stat_xxt: dict[str, list[torch.Tensor]] = {}

    def _mean_buf(self, logical_key: str) -> torch.Tensor:
        return self.get_buffer(self._mean_buf_names[logical_key])

    def _w_buf(self, logical_key: str) -> torch.Tensor:
        return self.get_buffer(self._w_buf_names[logical_key])

    @torch.no_grad()
    def reset_phi(self) -> None:
        self.reset_phi_stats()

    @torch.no_grad()
    def reset_phi_stats(self) -> None:
        self._ready = False
        self._stat_n = {k: [0 for _ in v] for k, v in self.teacher_dims.items()}
        self._stat_sum = {}
        self._stat_xxt = {}

    @torch.no_grad()
    def update_phi(self, teacher_feats: dict[str, list[Tensor]]) -> None:
        self.update_phi_stats(teacher_feats)

    @torch.no_grad()
    def update_phi_stats(self, teacher_feats: dict[str, list[Tensor]]) -> None:
        for k, feats in teacher_feats.items():
            if k not in self.teacher_dims:
                raise KeyError(f"Unknown teacher key: {k}")
            dims = self.teacher_dims[k]
            _ensure_list_len(k, feats, dims)

            for i, feat in enumerate(feats):
                _ensure_feat_dim(f"{k}[{i}]", feat, dims[i])
                X = _flatten_to_nxc_fp64(feat.detach())

                n = int(X.shape[0])
                if n == 0:
                    continue

                sum_x = X.sum(dim=0)
                sum_xxt = X.T @ X

                if k not in self._stat_sum:
                    self._stat_sum[k] = []
                    self._stat_xxt[k] = []

                if len(self._stat_sum[k]) <= i:
                    self._stat_sum[k].append(sum_x)
                    self._stat_xxt[k].append(sum_xxt)
                else:
                    self._stat_sum[k][i] = self._stat_sum[k][i] + sum_x
                    self._stat_xxt[k][i] = self._stat_xxt[k][i] + sum_xxt

                self._stat_n[k][i] += n

    @torch.no_grad()
    def finalize_phi(self, *, distributed: bool = True) -> None:
        self.finalize_phi_from_stats(distributed=distributed)

    @torch.no_grad()
    def finalize_phi_from_stats(self, *, distributed: bool = True) -> None:
        use_dist = bool(distributed) and _dist_is_initialized() and _dist_world_size() > 1

        for k, dims in self.teacher_dims.items():
            for i in range(len(dims)):
                if k not in self._stat_sum or k not in self._stat_xxt:
                    raise RuntimeError(f"No stats for teacher={k}. Call reset_phi_stats()->update_phi_stats() first.")
                if i >= len(self._stat_sum[k]) or i >= len(self._stat_xxt[k]):
                    raise RuntimeError(f"No stats for teacher={k} layer={i}.")

                n = int(self._stat_n[k][i])
                if n == 0:
                    raise RuntimeError(f"No stats for teacher={k} layer={i}.")

                sum_x = self._stat_sum[k][i]
                sum_xxt = self._stat_xxt[k][i]

                if use_dist:
                    n_t = torch.tensor(n, device=sum_x.device, dtype=torch.long)
                    dist.all_reduce(n_t, op=dist.ReduceOp.SUM)
                    dist.all_reduce(sum_x, op=dist.ReduceOp.SUM)
                    dist.all_reduce(sum_xxt, op=dist.ReduceOp.SUM)
                    n = int(n_t.item())

                mean64, cov64 = _mean_cov_from_sums(n=n, sum_x=sum_x, sum_xxt=sum_xxt)
                mean64, W_ship64 = _compute_w_ship_from_cov(
                    mean=mean64, cov=cov64, hadamard_fn=self.hadamard_fn, eps=self.eps
                )

                mean_key = _layer_key(k, i)
                mean_buf = self._mean_buf(mean_key)
                w_buf = self._w_buf(mean_key)
                mean_buf.copy_(mean64.to(dtype=mean_buf.dtype))
                w_buf.copy_(W_ship64.to(dtype=w_buf.dtype))

        self._ready = True

    def requires_loader(self, cache_path: str | Path) -> bool:
        return (not self._ready) and (not Path(cache_path).exists())

    def phi_state_dict(self) -> dict[str, Any]:
        return {
            "teacher_dims": {k: list(v) for k, v in self.teacher_dims.items()},
            "eps": float(self.eps),
            "mean": {
                k: [self._mean_buf(_layer_key(k, i)).detach().cpu() for i in range(len(v))]
                for k, v in self.teacher_dims.items()
            },
            "W_ship": {
                k: [self._w_buf(_layer_key(k, i)).detach().cpu() for i in range(len(v))]
                for k, v in self.teacher_dims.items()
            },
        }

    @torch.no_grad()
    def load_phi_from_cache(self, cache_path: str | Path, *, broadcast: bool = True) -> None:
        cache_path = Path(cache_path)
        use_dist = bool(broadcast) and _dist_is_initialized() and _dist_world_size() > 1

        if not use_dist:
            sd = torch.load(cache_path, map_location="cpu")
            self._load_phi_state_dict(sd)
            return

        sd: dict[str, Any] | None = None
        if _dist_rank() == 0:
            sd = torch.load(cache_path, map_location="cpu")

        for k, dims in self.teacher_dims.items():
            for i in range(len(dims)):
                mean_key = _layer_key(k, i)
                w_key = _layer_key(k, i)
                mean = self._mean_buf(mean_key).detach().clone()
                W_ship = self._w_buf(w_key).detach().clone()

                if _dist_rank() == 0:
                    assert sd is not None
                    mean.copy_(sd["mean"][k][i].to(device=mean.device, dtype=mean.dtype))
                    W_ship.copy_(sd["W_ship"][k][i].to(device=W_ship.device, dtype=W_ship.dtype))

                dist.broadcast(mean, src=0)
                dist.broadcast(W_ship, src=0)

                self._mean_buf(mean_key).copy_(mean)
                self._w_buf(w_key).copy_(W_ship)

        self._ready = True

    def save_phi_to_cache(self, cache_path: str | Path) -> None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.phi_state_dict(), cache_path)

    @torch.no_grad()
    def initialize_phi(
        self,
        *,
        cache_path: str | Path,
        teacher_feats_iter: Any | None = None,
        num_batches: int | None = None,
        distributed_stats: bool = True,
        broadcast_on_load: bool = True,
    ) -> None:
        cache_path = Path(cache_path)

        if cache_path.exists():
            self.load_phi_from_cache(cache_path, broadcast=broadcast_on_load)
            return

        if teacher_feats_iter is None:
            raise ValueError("cache_path does not exist; teacher_feats_iter must be provided to estimate PHI-S stats.")

        self.reset_phi_stats()
        for i, teacher_feats in enumerate(teacher_feats_iter):
            if num_batches is not None and i >= num_batches:
                break
            self.update_phi_stats(teacher_feats)
        self.finalize_phi_from_stats(distributed=distributed_stats)

        if not _dist_is_initialized() or _dist_rank() == 0:
            self.save_phi_to_cache(cache_path)

    def _load_phi_state_dict(self, sd: dict[str, Any]) -> None:
        teacher_dims = sd.get("teacher_dims", {})
        if {k: list(v) for k, v in teacher_dims.items()} != {k: list(v) for k, v in self.teacher_dims.items()}:
            raise ValueError(f"teacher_dims mismatch: cache={teacher_dims} vs module={self.teacher_dims}")

        mean_map: dict[str, list[torch.Tensor]] = sd["mean"]
        w_map: dict[str, list[torch.Tensor]] = sd["W_ship"]

        for k, dims in self.teacher_dims.items():
            _ensure_list_len(k, mean_map[k], dims)
            _ensure_list_len(k, w_map[k], dims)
            for i in range(len(dims)):
                mean_key = _layer_key(k, i)
                w_key = _layer_key(k, i)
                mean_buf = self._mean_buf(mean_key)
                w_buf = self._w_buf(w_key)
                mean_buf.copy_(mean_map[k][i].to(device=mean_buf.device, dtype=mean_buf.dtype))
                w_buf.copy_(w_map[k][i].to(device=w_buf.device, dtype=w_buf.dtype))

        self._ready = True

    @classmethod
    @torch.no_grad()
    def build_and_initialize(
        cls,
        *,
        teacher_dims: dict[str, list[int]],
        hadamard_fn: Callable[..., torch.Tensor],
        cache_path: str | Path,
        teacher_feats_iter: Any | None = None,
        num_batches: int | None = None,
        loss_type: str = "mse",
        eps: float = 1e-6,
        teacher_weights: dict[str, float] | None = None,
        layer_weights: dict[str, list[float]] | None = None,
        distributed_stats: bool = True,
        broadcast_on_load: bool = True,
    ) -> "MultiLayersPhiSDistillLoss":
        mod = cls(
            teacher_dims=teacher_dims,
            hadamard_fn=hadamard_fn,
            loss_type=loss_type,
            eps=eps,
            teacher_weights=teacher_weights,
            layer_weights=layer_weights,
        )
        mod.initialize_phi(
            cache_path=cache_path,
            teacher_feats_iter=teacher_feats_iter,
            num_batches=num_batches,
            distributed_stats=distributed_stats,
            broadcast_on_load=broadcast_on_load,
        )
        return mod

    def forward(
        self,
        student_feats: dict[str, list[Tensor]],
        teacher_feats: dict[str, list[Tensor]],
    ) -> torch.Tensor:
        if not self._ready:
            raise RuntimeError(
                "PHI-S not ready. Run reset_phi_stats()->update_phi_stats()->finalize_phi_from_stats() first."
            )

        if not student_feats:
            raise ValueError("student_feats is empty")

        first_key = next(iter(student_feats.keys()))
        if not student_feats[first_key]:
            raise ValueError("student_feats contains empty layer list")
        total = student_feats[first_key][0].new_zeros(())
        wsum = 0.0

        for k, tfeats in teacher_feats.items():
            if k not in student_feats:
                raise KeyError(f"Missing student feat for teacher={k}")
            if k not in self.teacher_dims:
                raise KeyError(f"Unknown teacher key: {k}")

            dims = self.teacher_dims[k]
            _ensure_list_len(k, tfeats, dims)
            _ensure_list_len(k, student_feats[k], dims)

            lw = None
            if self.layer_weights is not None:
                lw = self.layer_weights.get(k)
                if lw is None:
                    raise KeyError(f"Missing layer_weights for teacher={k}")
                if len(lw) != len(dims):
                    raise ValueError(f"{k}: layer_weights length mismatch (expected {len(dims)}, got {len(lw)})")

            for i, tfeat in enumerate(tfeats):
                sfeat = student_feats[k][i]
                _ensure_feat_dim(f"{k}[{i}]", tfeat, dims[i])

                mean = self._mean_buf(_layer_key(k, i))
                W_ship = self._w_buf(_layer_key(k, i))
                tphi = apply_phi_s(tfeat, mean, W_ship)

                if sfeat.shape != tphi.shape:
                    raise ValueError(
                        f"Shape mismatch for teacher={k}[{i}]: student={tuple(sfeat.shape)} vs tphi={tuple(tphi.shape)}"
                    )

                if self.loss_type == "mse":
                    loss = F.mse_loss(sfeat, tphi)
                elif self.loss_type == "smoothl1":
                    loss = F.smooth_l1_loss(sfeat, tphi)
                else:
                    raise ValueError("loss_type must be 'mse' or 'smoothl1'")

                weight = float(self.teacher_weights.get(k, 1.0))
                layer_w = float(lw[i]) if lw is not None else 1.0
                total = total + weight * layer_w * loss
                wsum += weight * layer_w

        return total / max(wsum, 1e-12)


class PhiSLossCached(nn.Module):
    """
    在 forward 里计算 loss；
    但 mean/W_ship 不每步重算，只按 update_every 间隔用当前 batch 更新一次缓存。
    """

    step: torch.Tensor
    mean_buf: torch.Tensor
    W_buf: torch.Tensor

    def __init__(
        self,
        dim: int,
        hadamard_fn: Callable[..., torch.Tensor],  # 你的 get_hadamard_matrix 包一层
        update_every: int = 100,  # 每 100 step 更新一次
        eps: float = 1e-6,
        loss_type: str = "mse",  # "mse"|"smoothl1"
    ):
        super().__init__()
        self.dim = dim
        self.hadamard_fn = hadamard_fn
        self.update_every = update_every
        self.eps = eps
        self.loss_type = loss_type

        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        self.register_buffer("mean_buf", torch.zeros(dim))
        self.register_buffer("W_buf", torch.eye(dim))

    @torch.no_grad()
    def _recompute_from_batch(self, teacher_proj: torch.Tensor) -> None:
        X = flatten_samples(teacher_proj)  # (N,C)
        mean_val = X.mean(dim=0)
        Xc = X - mean_val
        cov = torch.cov(Xc.T)
        e, U = torch.linalg.eigh(cov)
        e = torch.clamp(e, min=0.0)

        H = self.hadamard_fn(self.dim, device=U.device, dtype=U.dtype)
        normalizer = torch.sqrt(e.mean() + self.eps)
        W_ship = (1.0 / normalizer) * (H @ U.t())

        self.mean_buf.copy_(mean_val)
        self.W_buf.copy_(W_ship)

    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        """
        输入约定：
          - student_feat/teacher_feat 都在同一个通道维空间 dim 上（BNC 或 BCHW）
          - PHI-S 只基于 teacher_feat 估计/更新 mean、W_ship，再做标准化作为 target。
        """
        # 更新 step 计数
        self.step += 1

        # 间隔更新统计量（用 teacher_proj_feat 当前 batch）
        if (self.step.item() % self.update_every) == 1:  # 第 1、101、201...步更新
            self._recompute_from_batch(teacher_feat.detach())

        teacher_phi = apply_phi_s(teacher_feat, self.mean_buf, self.W_buf)

        if student_feat.shape != teacher_phi.shape:
            raise ValueError(f"shape mismatch: student {student_feat.shape} vs teacher_phi {teacher_phi.shape}")

        if self.loss_type == "mse":
            return F.mse_loss(student_feat, teacher_phi)
        elif self.loss_type == "smoothl1":
            return F.smooth_l1_loss(student_feat, teacher_phi)
        else:
            raise ValueError("loss_type must be 'mse' or 'smoothl1'")


if __name__ == "__main__":
    hadamard_fn = lambda n, device=None, dtype=None: get_hadamard_matrix(n, allow_approx=True).to(
        device=device, dtype=dtype
    )

    loss_mod = PhiSDistillLoss(teacher_dims={"sam": 1024, "clip": 768}, hadamard_fn=hadamard_fn, loss_type="mse")

    # 1) Dist Est Steps: 只用 teacher 特征估计 PHI-S（cache 不依赖 student head）
    loss_mod.reset_phi_stats()
    B, N = 2, 8
    for _ in range(10):
        teacher_feats_dict: dict[str, torch.Tensor] = {"sam": torch.randn(B, N, 1024), "clip": torch.randn(B, N, 768)}
        loss_mod.update_phi_stats(teacher_feats_dict)
    loss_mod.finalize_phi_from_stats(distributed=False)

    # 2) Training
    teacher_feats_dict = {"sam": torch.randn(B, N, 1024), "clip": torch.randn(B, N, 768)}
    student_feats_dict = {"sam": torch.randn(B, N, 1024), "clip": torch.randn(B, N, 768)}
    loss = loss_mod(student_feats_dict, teacher_feats_dict)
    print(loss.item())

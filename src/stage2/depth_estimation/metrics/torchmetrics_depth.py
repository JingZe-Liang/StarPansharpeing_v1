from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor
from torchmetrics.metric import Metric


AlignMode = Literal["scale_shift", "scale_only"]


class DepthEstimationMetrics(Metric):
    """
    Masked depth regression metrics with optional per-image alignment.

    Parameters
    ----------
    eps : float
        Small positive threshold for metrics that require positive depth (AbsRel/LogRMSE/δ).
    align_mode : {"scale_shift", "scale_only"} | None
        - None: compute ONLY raw metrics (directly comparing pred vs target).
        - "scale_shift": per-image least-squares fit (s, t) on valid pixels, then compute aligned metrics too.
        - "scale_only": per-image least-squares fit s (t=0) on valid pixels, then compute aligned metrics too.

    Notes
    -----
    - All metrics are computed only on `valid_mask`.
    - Metrics requiring positive depth (AbsRel/LogRMSE/δ) additionally filter target>eps and pred>eps
      (for aligned metrics, uses aligned_pred>eps).
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, *, eps: float = 1e-6, align_mode: AlignMode | None = None) -> None:
        super().__init__(dist_sync_on_step=False)
        self.eps = float(eps)
        self.align_mode = align_mode

        # ---- raw states ----
        self.add_state("l1_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("se_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_valid", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("absrel_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("logse_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("delta1", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("delta2", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("delta3", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_pos", default=torch.tensor(0.0), dist_reduce_fx="sum")

        # ---- aligned states (only if requested) ----
        if self.align_mode is not None:
            self.add_state("al_l1_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("al_se_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("al_n_valid", default=torch.tensor(0.0), dist_reduce_fx="sum")

            self.add_state("al_absrel_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("al_logse_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("al_delta1", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("al_delta2", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("al_delta3", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("al_n_pos", default=torch.tensor(0.0), dist_reduce_fx="sum")

    @staticmethod
    def _as_bool_mask(mask: Tensor, like: Tensor) -> Tensor:
        if mask.dtype == torch.bool:
            # return mask
            pass
        mask = mask.to(dtype=torch.bool, device=like.device)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)  # [B, 1, H, W]
        return mask

    @staticmethod
    def _ensure_bchw(x: Tensor) -> Tensor:
        """
        Normalize shapes to [B, C, H, W] for per-image looping.
        Accepts [H, W], [C, H, W], or [B, C, H, W].
        """
        if x.ndim == 2:  # [H, W]
            return x.unsqueeze(0).unsqueeze(0)
        if x.ndim == 3:  # [C, H, W]
            return x.unsqueeze(0)
        if x.ndim == 4:  # [B, C, H, W]
            return x
        raise ValueError(f"Unsupported tensor shape {tuple(x.shape)}; expected 2D/3D/4D.")

    @staticmethod
    def _fit_scale_shift(p: Tensor, g: Tensor, eps_det: float = 1e-12) -> tuple[Tensor, Tensor]:
        """
        Solve min_{s,t} || s*p + t - g ||^2 on 1D vectors p,g.
        Closed-form least squares.
        """
        n = p.numel()
        Sp = p.sum()
        Sg = g.sum()
        Spp = (p * p).sum()
        Spg = (p * g).sum()

        det = n * Spp - Sp * Sp
        if det.abs() < eps_det:
            # Degenerate: prediction nearly constant on valid pixels
            s = p.new_tensor(1.0)
            t = (g.median() - p.median()) if n > 0 else p.new_tensor(0.0)
            return s, t

        s = (n * Spg - Sp * Sg) / det
        t = (Spp * Sg - Sp * Spg) / det
        return s, t

    @staticmethod
    def _fit_scale_only(p: Tensor, g: Tensor, eps_den: float = 1e-12) -> Tensor:
        """
        Solve min_{s} || s*p - g ||^2 => s = (p·g)/(p·p)
        """
        den = (p * p).sum()
        if den.abs() < eps_den:
            return p.new_tensor(1.0)
        return (p * g).sum() / den

    def _accumulate_raw(self, pred_v: Tensor, target_v: Tensor) -> None:
        # MAE/RMSE on all valid
        self.l1_sum = self.l1_sum + (pred_v - target_v).abs().sum()
        self.se_sum = self.se_sum + ((pred_v - target_v) ** 2).sum()
        self.n_valid = self.n_valid + torch.tensor(float(pred_v.numel()), device=self.n_valid.device)

        # Positive-only metrics
        pos = (pred_v > self.eps) & (target_v > self.eps)
        if not torch.any(pos):
            return
        pred_p = pred_v[pos]
        target_p = target_v[pos]

        self.absrel_sum = self.absrel_sum + ((pred_p - target_p).abs() / target_p).sum()
        self.logse_sum = self.logse_sum + ((torch.log(pred_p) - torch.log(target_p)) ** 2).sum()

        ratio = torch.maximum(pred_p / target_p, target_p / pred_p)
        self.delta1 = self.delta1 + (ratio < 1.25).sum()
        self.delta2 = self.delta2 + (ratio < 1.25**2).sum()
        self.delta3 = self.delta3 + (ratio < 1.25**3).sum()
        self.n_pos = self.n_pos + torch.tensor(float(pred_p.numel()), device=self.n_pos.device)

    def _accumulate_aligned(self, pred_v: Tensor, target_v: Tensor) -> None:
        # MAE/RMSE on all valid
        self.al_l1_sum = self.al_l1_sum + (pred_v - target_v).abs().sum()
        self.al_se_sum = self.al_se_sum + ((pred_v - target_v) ** 2).sum()
        self.al_n_valid = self.al_n_valid + torch.tensor(float(pred_v.numel()), device=self.al_n_valid.device)

        # Positive-only metrics
        pos = (pred_v > self.eps) & (target_v > self.eps)
        if not torch.any(pos):
            return
        pred_p = pred_v[pos]
        target_p = target_v[pos]

        self.al_absrel_sum = self.al_absrel_sum + ((pred_p - target_p).abs() / target_p).sum()
        self.al_logse_sum = self.al_logse_sum + ((torch.log(pred_p) - torch.log(target_p)) ** 2).sum()

        ratio = torch.maximum(pred_p / target_p, target_p / pred_p)
        self.al_delta1 = self.al_delta1 + (ratio < 1.25).sum()
        self.al_delta2 = self.al_delta2 + (ratio < 1.25**2).sum()
        self.al_delta3 = self.al_delta3 + (ratio < 1.25**3).sum()
        self.al_n_pos = self.al_n_pos + torch.tensor(float(pred_p.numel()), device=self.al_n_pos.device)

    def update(self, pred: Tensor, target: Tensor, valid_mask: Tensor) -> None:  # type: ignore[override]
        pred_f = pred.float()
        target_f = target.float()
        if pred_f.shape != target_f.shape:
            raise ValueError(f"pred/target shape mismatch: {pred_f.shape} vs {target_f.shape}")

        valid = self._as_bool_mask(valid_mask, pred_f)
        if valid.shape != pred_f.shape:
            raise ValueError(f"valid_mask shape mismatch: {valid.shape} vs {pred_f.shape}")

        if not torch.any(valid):
            return

        # ---- raw metrics (flatten all valid pixels across batch) ----
        pred_v = pred_f[valid]
        target_v = target_f[valid]
        self._accumulate_raw(pred_v, target_v)

        # ---- aligned metrics (per-image alignment) ----
        if self.align_mode is None:
            return

        # Normalize shapes to [B, C, H, W]
        pred_bchw = self._ensure_bchw(pred_f)
        target_bchw = self._ensure_bchw(target_f)
        valid_bchw = self._ensure_bchw(valid)

        B = pred_bchw.shape[0]
        for b in range(B):
            vb = valid_bchw[b]
            if not torch.any(vb):
                continue

            p = pred_bchw[b][vb]  # 1D
            g = target_bchw[b][vb]  # 1D

            if self.align_mode == "scale_shift":
                s, t = self._fit_scale_shift(p, g)
                p_al = s * p + t
            elif self.align_mode == "scale_only":
                s = self._fit_scale_only(p, g)
                p_al = s * p
            else:
                raise ValueError(f"Unknown align_mode: {self.align_mode}")

            self._accumulate_aligned(p_al, g)

    def compute(self) -> dict[str, Tensor]:  # type: ignore[override]
        # ---- raw ----
        n_valid = self.n_valid.clamp_min(1.0)
        n_pos = self.n_pos.clamp_min(1.0)

        mae = self.l1_sum / n_valid
        rmse = torch.sqrt(self.se_sum / n_valid)
        absrel = self.absrel_sum / n_pos
        logrmse = torch.sqrt(self.logse_sum / n_pos)
        d1 = self.delta1 / n_pos
        d2 = self.delta2 / n_pos
        d3 = self.delta3 / n_pos

        out: dict[str, Tensor] = {
            "mae": mae,
            "rmse": rmse,
            "absrel": absrel,
            "logrmse": logrmse,
            "delta1": d1,
            "delta2": d2,
            "delta3": d3,
            "n_valid": self.n_valid,
            "n_pos": self.n_pos,
        }

        # ---- aligned (optional) ----
        if self.align_mode is not None:
            al_n_valid = self.al_n_valid.clamp_min(1.0)
            al_n_pos = self.al_n_pos.clamp_min(1.0)

            out.update(
                {
                    "aligned_mae": self.al_l1_sum / al_n_valid,
                    "aligned_rmse": torch.sqrt(self.al_se_sum / al_n_valid),
                    "aligned_absrel": self.al_absrel_sum / al_n_pos,
                    "aligned_logrmse": torch.sqrt(self.al_logse_sum / al_n_pos),
                    "aligned_delta1": self.al_delta1 / al_n_pos,
                    "aligned_delta2": self.al_delta2 / al_n_pos,
                    "aligned_delta3": self.al_delta3 / al_n_pos,
                    "aligned_n_valid": self.al_n_valid,
                    "aligned_n_pos": self.al_n_pos,
                }
            )

        return out

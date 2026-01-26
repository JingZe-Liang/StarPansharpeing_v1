from __future__ import annotations

import torch
from torch import Tensor


def summarize_depth_targets(
    target: Tensor,
    valid_mask: Tensor,
    *,
    eps: float = 1e-6,
    near_thresholds: tuple[float, ...] = (0.1, 1.0),
    quantiles: tuple[float, ...] = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0),
) -> str:
    """Create a compact string summary of target depth distribution.

    This is intended for logging/debugging to diagnose metric blow-ups (e.g., AbsRel)
    when targets have many values close to zero.
    """

    target_v = target[valid_mask].detach().float()
    if target_v.numel() == 0:
        return "Depth target stats: no valid pixels"

    device = target_v.device
    q_levels = torch.tensor(quantiles, device=device)
    q_vals_t = torch.quantile(target_v, q_levels)
    q_vals = {f"q{int(p * 100):02d}": float(v) for p, v in zip(quantiles, q_vals_t.tolist(), strict=True)}

    pos = target_v > float(eps)
    pos_ratio = float(pos.float().mean().item())

    near_parts: list[str] = []
    for thr in near_thresholds:
        near_ratio = float(((target_v > float(eps)) & (target_v < float(thr))).float().mean().item())
        near_parts.append(f"(0,{thr:g})={near_ratio:.4f}")

    return (
        "Depth target stats: "
        f"n_valid={target_v.numel()} "
        f"eps={float(eps):g} pos_ratio={pos_ratio:.4f} "
        + " ".join(near_parts)
        + " "
        + " ".join([f"{k}={v:.4f}" for k, v in q_vals.items()])
    )

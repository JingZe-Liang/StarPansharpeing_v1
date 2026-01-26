from __future__ import annotations

import torch
from torch import Tensor


def make_valid_mask(depth: Tensor, *, invalid_threshold: float) -> Tensor:
    return depth > float(invalid_threshold)


def apply_clamp_and_scale(
    depth: Tensor,
    valid_mask: Tensor,
    *,
    clamp_range: tuple[float | None, float | None] | None,
    scale: float | None,
) -> Tensor:
    out = depth
    if clamp_range is not None:
        min_v, max_v = clamp_range
        out = out.clone()
        out[valid_mask] = out[valid_mask].clamp(min=min_v, max=max_v)
    if scale is not None:
        if out is depth:
            out = out.clone()
        out[valid_mask] = out[valid_mask] / float(scale)
    return out


def fill_invalid(
    depth: Tensor,
    valid_mask: Tensor,
    *,
    fill_value: float = 0.0,
) -> Tensor:
    out = depth
    if out is depth:
        out = out.clone()
    out[~valid_mask] = float(fill_value)
    return out

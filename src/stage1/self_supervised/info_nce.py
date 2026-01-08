from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn.functional as F


def _validate_inputs(embeddings: torch.Tensor, temperature: float) -> tuple[int, int, int]:
    if embeddings.ndim != 3:
        raise ValueError(f"`embeddings` 需要是 [V, B, D] 的 3 维张量，但得到 shape={tuple(embeddings.shape)}")
    if not torch.is_floating_point(embeddings):
        raise TypeError(f"`embeddings` 需要是浮点张量，但得到 dtype={embeddings.dtype}")

    v, b, d = embeddings.shape
    if v < 2:
        raise ValueError(f"`V` 需要 >= 2 才能构造正样本对，但得到 V={v}")
    if b < 1:
        raise ValueError(f"`B` 需要 >= 1，但得到 B={b}")
    if d < 1:
        raise ValueError(f"`D` 需要 >= 1，但得到 D={d}")
    if not (temperature > 0.0):
        raise ValueError(f"`temperature` 需要 > 0，但得到 temperature={temperature}")

    return v, b, d


def _build_positive_and_eye_masks(v: int, b: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    n = v * b
    labels = torch.arange(b, device=device).repeat(v)  # [0..B-1] 重复 V 次
    label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # [VB, VB]
    eye_mask = torch.eye(n, device=device, dtype=torch.bool)
    positive_mask = label_mask & ~eye_mask
    return positive_mask, eye_mask


def _build_anchor_mask(
    *,
    v: int,
    b: int,
    device: torch.device,
    anchor_views: Sequence[int] | torch.Tensor | None,
) -> torch.Tensor | None:
    if anchor_views is None:
        return None

    if isinstance(anchor_views, torch.Tensor):
        if anchor_views.ndim != 1:
            raise ValueError(f"`anchor_views` 需要是一维张量，但得到 shape={tuple(anchor_views.shape)}")
        anchor_view_ids = anchor_views.to(device=device, dtype=torch.long)
    else:
        if len(anchor_views) == 0:
            raise ValueError("`anchor_views` 不能为空；传 None 表示使用所有 view 作为 anchor。")
        anchor_view_ids = torch.tensor(list(anchor_views), device=device, dtype=torch.long)

    if (anchor_view_ids < 0).any() or (anchor_view_ids >= v).any():
        raise ValueError(f"`anchor_views` 里的 view id 必须在 [0, {v - 1}]，但得到：{anchor_view_ids.tolist()}")

    n = v * b
    view_ids = torch.arange(n, device=device, dtype=torch.long) // b
    return (view_ids.unsqueeze(1) == anchor_view_ids.unsqueeze(0)).any(dim=1)


def _logits_from_embeddings(embeddings: torch.Tensor, temperature: float) -> torch.Tensor:
    v, b, d = embeddings.shape
    n = v * b

    features = embeddings.reshape(n, d)
    features = F.normalize(features, dim=1)
    logits = (features @ features.T) / temperature

    row_max = logits.max(dim=1, keepdim=True).values.detach()
    return logits - row_max


def multiview_info_nce_loss(
    embeddings: torch.Tensor,  # shape: [V, B, D]
    temperature: float = 0.1,
    objective: Literal["supcon", "multi_positive_infonce"] = "supcon",
    anchor_views: Sequence[int] | torch.Tensor | None = None,
) -> torch.Tensor:
    """
    多视角对比学习损失（支持 SupCon 与多正样本 InfoNCE）。

    参数
    ----
    embeddings:
        形状为 `[V, B, D]` 的特征，其中 V 是 view 数，B 是 batch size，D 是特征维度。
    temperature:
        温度系数，必须 > 0。
    objective:
        - `"supcon"`: 对每个 anchor 的所有正样本 log-prob 做平均（Supervised Contrastive 风格）。
        - `"multi_positive_infonce"`: 分子对所有正样本做 `logsumexp`（多正样本 InfoNCE）。
    anchor_views:
        选择哪些 view 作为 anchor 来做平均聚合（对比集合仍然是 `embeddings` 的全部 view）。
        - `None`: 使用所有 view 作为 anchor（默认）。
        - `Sequence[int]` / `Tensor`: 指定 view id（范围 `[0, V-1]`）。
    """
    v, b, _ = _validate_inputs(embeddings, temperature)
    logits = _logits_from_embeddings(embeddings, temperature)
    positive_mask, eye_mask = _build_positive_and_eye_masks(v, b, embeddings.device)
    anchor_mask = _build_anchor_mask(v=v, b=b, device=embeddings.device, anchor_views=anchor_views)

    neg_inf = torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype)
    log_denom = torch.logsumexp(logits.masked_fill(eye_mask, neg_inf), dim=1)  # [VB]

    pos_count = positive_mask.sum(dim=1)
    if (pos_count == 0).any():
        raise RuntimeError("存在 anchor 没有任何正样本（通常是 V<2 或 mask 构造错误）。")

    if objective == "supcon":
        log_prob = logits - log_denom.unsqueeze(1)
        positive_weight = positive_mask.to(dtype=log_prob.dtype)
        mean_log_prob_pos = (positive_weight * log_prob).sum(dim=1) / pos_count.to(dtype=log_prob.dtype)
        per_anchor = -mean_log_prob_pos
        return per_anchor.mean() if anchor_mask is None else per_anchor[anchor_mask].mean()

    if objective == "multi_positive_infonce":
        log_num = torch.logsumexp(logits.masked_fill(~positive_mask, neg_inf), dim=1)  # [VB]
        per_anchor = -(log_num - log_denom)
        return per_anchor.mean() if anchor_mask is None else per_anchor[anchor_mask].mean()

    raise ValueError(f"未知 objective={objective!r}，可选值为 'supcon' 或 'multi_positive_infonce'")

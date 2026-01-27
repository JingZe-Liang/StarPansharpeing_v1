from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


def _as_bool_mask(mask: Tensor | None, like: Tensor) -> Tensor:
    if mask is None:
        return torch.ones_like(like, dtype=torch.bool)
    if mask.dtype == torch.bool:
        return mask.to(device=like.device)
    return mask.to(dtype=torch.bool, device=like.device)


def _as_float_mask(mask: Tensor | None, like: Tensor) -> Tensor:
    if mask is None:
        return torch.ones_like(like, dtype=like.dtype)
    if mask.dtype == torch.bool:
        return mask.to(dtype=like.dtype, device=like.device)
    return mask.to(dtype=like.dtype, device=like.device)


def _masked_mean(value: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return value.mean()
    mask_f = _as_float_mask(mask, value)
    denom = mask_f.sum().clamp_min(1.0)
    return (value * mask_f).sum() / denom


def _diff_x(value: Tensor) -> Tensor:
    return value[..., :, 1:] - value[..., :, :-1]


def _diff_y(value: Tensor) -> Tensor:
    return value[..., 1:, :] - value[..., :-1, :]


def _channel_mean_abs(value: Tensor) -> Tensor:
    if value.dim() >= 4:
        return value.abs().mean(dim=-3, keepdim=True)
    return value.abs()


def l1_loss(pred: Tensor, target: Tensor, valid_mask: Tensor | None = None) -> Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {pred.shape} vs {target.shape}")
    return _masked_mean((pred - target).abs(), valid_mask)


def l2_loss(pred: Tensor, target: Tensor, valid_mask: Tensor | None = None) -> Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {pred.shape} vs {target.shape}")
    return _masked_mean((pred - target) ** 2, valid_mask)


def huber_loss(
    pred: Tensor,
    target: Tensor,
    valid_mask: Tensor | None = None,
    *,
    delta: float = 1.0,
) -> Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {pred.shape} vs {target.shape}")
    err = pred - target
    abs_err = err.abs()
    delta_t = torch.as_tensor(delta, device=pred.device, dtype=pred.dtype)
    loss = torch.where(
        abs_err <= delta_t,
        0.5 * (err**2) / delta_t,
        abs_err - 0.5 * delta_t,
    )
    return _masked_mean(loss, valid_mask)


def berhu_loss(
    pred: Tensor,
    target: Tensor,
    valid_mask: Tensor | None = None,
    *,
    c: float | None = None,
) -> Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {pred.shape} vs {target.shape}")

    abs_err = (pred - target).abs()
    if valid_mask is not None:
        mask = _as_bool_mask(valid_mask, abs_err)
        if not torch.any(mask):
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        abs_err_valid = abs_err[mask]
    else:
        mask = None
        abs_err_valid = abs_err

    if c is None:
        c_val = 0.2 * abs_err_valid.max()
    else:
        c_val = float(c)

    c_t = torch.as_tensor(c_val, device=pred.device, dtype=pred.dtype).clamp_min(1e-6)
    loss = torch.where(abs_err <= c_t, abs_err, (abs_err**2 + c_t**2) / (2.0 * c_t))
    return _masked_mean(loss, mask)


def silog_loss(
    pred: Tensor,
    target: Tensor,
    valid_mask: Tensor | None = None,
    *,
    lambda_weight: float = 0.5,
    eps: float = 1e-6,
) -> Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {pred.shape} vs {target.shape}")

    pos_mask = (pred > eps) & (target > eps)
    mask = _as_bool_mask(valid_mask, pred) & pos_mask
    if not torch.any(mask):
        return torch.zeros((), device=pred.device, dtype=pred.dtype)

    log_pred = torch.log(pred.clamp_min(eps))
    log_target = torch.log(target.clamp_min(eps))
    diff = log_pred - log_target
    mean_diff = _masked_mean(diff, mask)
    mean_sq = _masked_mean(diff**2, mask)
    return mean_sq - float(lambda_weight) * (mean_diff**2)


def _flatten_batch(value: Tensor) -> Tensor:
    if value.dim() == 2:
        value = value.unsqueeze(0)
    return value.reshape(value.shape[0], -1)


def _solve_scale_shift(
    pred: Tensor,
    target: Tensor,
    valid_mask: Tensor | None,
    *,
    mode: str = "scale_shift",
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    pred_f = _flatten_batch(pred)
    target_f = _flatten_batch(target)
    if pred_f.shape != target_f.shape:
        raise ValueError(f"pred/target shape mismatch: {pred.shape} vs {target.shape}")

    if valid_mask is None:
        mask_f = torch.ones_like(pred_f)
    else:
        mask_full = _as_float_mask(valid_mask, pred)
        mask_f = _flatten_batch(mask_full)

    sum_p = (mask_f * pred_f).sum(dim=1)
    sum_t = (mask_f * target_f).sum(dim=1)
    sum_pp = (mask_f * pred_f * pred_f).sum(dim=1)
    sum_pt = (mask_f * pred_f * target_f).sum(dim=1)

    if mode == "scale_only":
        denom = sum_pp
        invalid = denom.abs() < eps
        scale = torch.ones_like(denom)
        shift = torch.zeros_like(denom)
        valid = ~invalid
        if torch.any(valid):
            scale[valid] = sum_pt[valid] / denom[valid]
        return scale, shift

    if mode != "scale_shift":
        raise ValueError(f"Unsupported mode: {mode}")

    n = mask_f.sum(dim=1)
    det = sum_pp * n - sum_p * sum_p
    invalid = det.abs() < eps
    scale = torch.ones_like(det)
    shift = torch.zeros_like(det)
    valid = ~invalid
    if torch.any(valid):
        scale_num = sum_pt[valid] * n[valid] - sum_t[valid] * sum_p[valid]
        shift_num = sum_pp[valid] * sum_t[valid] - sum_p[valid] * sum_pt[valid]
        scale[valid] = scale_num / det[valid]
        shift[valid] = shift_num / det[valid]

    return scale, shift


def _prepare_work_tensors(
    pred: Tensor,
    target: Tensor,
    valid_mask: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor | None, bool]:
    added_batch = pred.dim() == 2
    pred_work = pred.unsqueeze(0) if added_batch else pred
    target_work = target.unsqueeze(0) if added_batch else target
    mask_work = valid_mask.unsqueeze(0) if added_batch and valid_mask is not None else valid_mask
    return pred_work, target_work, mask_work, added_batch


def align_scale_shift(
    pred: Tensor,
    target: Tensor,
    valid_mask: Tensor | None = None,
    *,
    mode: str = "scale_shift",
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    pred_work, target_work, mask_work, added_batch = _prepare_work_tensors(pred, target, valid_mask)

    scale, shift = _solve_scale_shift(pred_work, target_work, mask_work, mode=mode, eps=eps)
    view_shape = (scale.shape[0],) + (1,) * (pred_work.dim() - 1)
    scale_view = scale.view(view_shape)
    shift_view = shift.view(view_shape)
    aligned = scale_view * pred_work + shift_view

    if added_batch:
        aligned = aligned.squeeze(0)
    return aligned, scale, shift


def scale_shift_invariant_loss(
    pred: Tensor,
    target: Tensor,
    valid_mask: Tensor | None = None,
    *,
    reduction: str = "l1",
    mode: str = "scale_shift",
    eps: float = 1e-6,
) -> Tensor:
    pred_work, target_work, mask_work, _ = _prepare_work_tensors(pred, target, valid_mask)
    scale, shift = _solve_scale_shift(pred_work, target_work, mask_work, mode=mode, eps=eps)
    view_shape = (scale.shape[0],) + (1,) * (pred_work.dim() - 1)
    aligned = scale.view(view_shape) * pred_work + shift.view(view_shape)
    if reduction == "l1":
        return _masked_mean((aligned - target_work).abs(), mask_work)
    if reduction == "l2":
        return _masked_mean((aligned - target_work) ** 2, mask_work)
    raise ValueError(f"Unsupported reduction: {reduction}")


def gradient_loss(
    pred: Tensor,
    target: Tensor,
    valid_mask: Tensor | None = None,
) -> Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {pred.shape} vs {target.shape}")

    diff_x = (_diff_x(pred) - _diff_x(target)).abs()
    diff_y = (_diff_y(pred) - _diff_y(target)).abs()

    if valid_mask is None:
        return diff_x.mean() + diff_y.mean()

    mask = _as_bool_mask(valid_mask, pred)
    mask_x = mask[..., :, 1:] & mask[..., :, :-1]
    mask_y = mask[..., 1:, :] & mask[..., :-1, :]
    return _masked_mean(diff_x, mask_x) + _masked_mean(diff_y, mask_y)


def edge_aware_smoothness_loss(
    pred: Tensor,
    image: Tensor,
    valid_mask: Tensor | None = None,
) -> Tensor:
    pred_dx = _diff_x(pred)
    pred_dy = _diff_y(pred)
    img_dx = _diff_x(image)
    img_dy = _diff_y(image)

    weight_x = torch.exp(-_channel_mean_abs(img_dx))
    weight_y = torch.exp(-_channel_mean_abs(img_dy))

    smooth_x = pred_dx.abs() * weight_x
    smooth_y = pred_dy.abs() * weight_y

    if valid_mask is None:
        return smooth_x.mean() + smooth_y.mean()

    mask = _as_bool_mask(valid_mask, pred)
    mask_x = mask[..., :, 1:] & mask[..., :, :-1]
    mask_y = mask[..., 1:, :] & mask[..., :-1, :]
    return _masked_mean(smooth_x, mask_x) + _masked_mean(smooth_y, mask_y)


def laplace_nll_loss(
    pred: Tensor,
    target: Tensor,
    scale: Tensor,
    valid_mask: Tensor | None = None,
    *,
    eps: float = 1e-6,
) -> Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {pred.shape} vs {target.shape}")
    if scale.shape != target.shape:
        raise ValueError(f"scale/target shape mismatch: {scale.shape} vs {target.shape}")

    scale_safe = scale.clamp_min(eps)
    loss = (pred - target).abs() / scale_safe + torch.log(scale_safe) + math.log(2.0)
    return _masked_mean(loss, valid_mask)


class DepthEstimationLoss(nn.Module):
    """
    Composite loss for depth regression with optional regularizers.

    Notes
    -----
    - data_term: primary regression loss.
        - "l1": robust to outliers, preserves edges.
        - "l2": penalizes large errors more, smoother but sensitive to outliers.
        - "huber": l2 near zero, l1 for large errors; good default.
        - "berhu": l1 for small errors, l2 for large errors; emphasizes big mistakes.
    - data_transform:
        - "log1p": compresses large depth values, reduces far-range dominance.
    - silog: scale-invariant log loss, stabilizes relative error, common in monocular depth.
    - ssi: scale/shift invariant loss, useful when absolute scale is ambiguous.
    - grad: gradient matching, encourages sharp depth boundaries.
    - smooth: edge-aware smoothness, suppresses noise in textureless regions (requires image).
    - laplace: heteroscedastic NLL, models per-pixel uncertainty (requires scale).
    """

    def __init__(
        self,
        *,
        data_term: str = "berhu",
        data_transform: str | None = None,
        data_weight: float = 1.0,
        grad_weight: float = 0.0,
        smooth_weight: float = 0.0,
        silog_weight: float = 0.0,
        ssi_weight: float = 0.0,
        berhu_weight: float = 0.0,
        laplace_weight: float = 0.0,
        huber_delta: float = 1.0,
        berhu_c: float = 0.2,
        silog_lambda: float = 0.5,
        silog_eps: float = 1e-6,
        ssi_reduction: str = "l1",
        ssi_mode: str = "scale_shift",
        ssi_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.data_term = data_term
        self.data_transform = data_transform
        self.data_weight = float(data_weight)
        self.grad_weight = float(grad_weight)
        self.smooth_weight = float(smooth_weight)
        self.silog_weight = float(silog_weight)
        self.ssi_weight = float(ssi_weight)
        self.berhu_weight = float(berhu_weight)
        self.laplace_weight = float(laplace_weight)
        self.huber_delta = float(huber_delta)
        self.berhu_c = berhu_c
        self.silog_lambda = float(silog_lambda)
        self.silog_eps = float(silog_eps)
        self.ssi_reduction = ssi_reduction
        self.ssi_mode = ssi_mode
        self.ssi_eps = float(ssi_eps)

    def _apply_data_transform(self, pred: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        if self.data_transform is None:
            return pred, target
        if self.data_transform == "log1p":
            pred_t = torch.log1p(pred.clamp_min(0.0))
            target_t = torch.log1p(target.clamp_min(0.0))
            return pred_t, target_t
        raise ValueError(f"Unsupported data_transform: {self.data_transform}")

    def _data_loss(self, pred: Tensor, target: Tensor, valid_mask: Tensor | None) -> Tensor:
        if self.data_term == "l1":
            return l1_loss(pred, target, valid_mask)
        if self.data_term == "l2":
            return l2_loss(pred, target, valid_mask)
        if self.data_term == "huber":
            return huber_loss(pred, target, valid_mask, delta=self.huber_delta)
        if self.data_term == "berhu":
            return berhu_loss(pred, target, valid_mask, c=self.berhu_c)
        raise ValueError(f"Unsupported data_term: {self.data_term}")

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        valid_mask: Tensor | None = None,
        *,
        image: Tensor | None = None,
        scale: Tensor | None = None,
        return_logs: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        loss_terms: dict[str, Tensor] = {}
        total = torch.zeros((), device=pred.device, dtype=pred.dtype)

        data_pred, data_target = self._apply_data_transform(pred, target)
        if self.data_weight > 0.0:
            data_loss = self._data_loss(data_pred, data_target, valid_mask)
            total = total + data_loss * self.data_weight
            loss_terms["data"] = data_loss

        if self.berhu_weight > 0.0 and self.data_term != "berhu":
            berhu = berhu_loss(data_pred, data_target, valid_mask, c=self.berhu_c)
            total = total + berhu * self.berhu_weight
            loss_terms["berhu"] = berhu

        if self.silog_weight > 0.0:
            silog = silog_loss(pred, target, valid_mask, lambda_weight=self.silog_lambda, eps=self.silog_eps)
            total = total + silog * self.silog_weight
            loss_terms["silog"] = silog

        if self.ssi_weight > 0.0:
            ssi = scale_shift_invariant_loss(
                pred,
                target,
                valid_mask,
                reduction=self.ssi_reduction,
                mode=self.ssi_mode,
                eps=self.ssi_eps,
            )
            total = total + ssi * self.ssi_weight
            loss_terms["ssi"] = ssi

        if self.grad_weight > 0.0:
            grad = gradient_loss(pred, target, valid_mask)
            total = total + grad * self.grad_weight
            loss_terms["grad"] = grad

        if self.smooth_weight > 0.0:
            if image is None:
                raise ValueError("edge-aware smoothness requires `image`.")
            smooth = edge_aware_smoothness_loss(pred, image, valid_mask)
            total = total + smooth * self.smooth_weight
            loss_terms["smooth"] = smooth

        if self.laplace_weight > 0.0:
            if scale is None:
                raise ValueError("Laplace NLL loss requires `scale`.")
            laplace = laplace_nll_loss(pred, target, scale, valid_mask)
            total = total + laplace * self.laplace_weight
            loss_terms["laplace"] = laplace

        if return_logs:
            loss_terms["total"] = total.detach()
            return total, loss_terms
        return total

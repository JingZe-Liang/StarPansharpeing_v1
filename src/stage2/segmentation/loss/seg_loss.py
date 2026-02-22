import segmentation_models_pytorch as smp
import torch
from easydict import EasyDict as edict
from einx import get_at
from jaxtyping import Float, Int
from loguru import logger
from torch import Tensor


def loss_apply_weights(
    losses: list[Tensor],
    weights: Tensor | tuple | list | None = None,
    ret_loss_parts: bool = False,
) -> Tensor | tuple[Tensor, tuple[Tensor, ...]]:
    loss_stk = torch.stack(losses)
    if weights is None:
        return loss_stk.mean()

    if isinstance(weights, (tuple, list)):
        weights = torch.as_tensor(weights).to(loss_stk)
    weights = weights.to(loss_stk.device)

    weighted_loss = loss_stk * weights
    loss_out = weighted_loss.sum()

    if ret_loss_parts:
        loss_parts = loss_out.unbind()
        return loss_out, loss_parts
    else:
        return loss_out


class HyperSegmentationLoss(torch.nn.Module):
    loss_names = ["dice_loss", "ce_loss", "lovasz_loss", "focal_loss"]

    def __init__(
        self,
        mode: str = "multiclass",
        num_classes: int = 24,
        dice_cal_classes: list[int] | None = None,
        ce_weight: list[float] | Tensor | None = None,
        ignore_index: int = 255,
        loss_weights: list[float] | None = None,
        dynamic_mode: str | None = None,  # None, "batch", "ema"
        ema_decay: float = 0.99,
        focal_alpha: float | None = None,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.dynamic_mode = dynamic_mode
        self.ema_decay = ema_decay
        ce_weight = torch.as_tensor(ce_weight, device="cuda").float() if ce_weight is not None else None
        self.cross_entropy = torch.nn.CrossEntropyLoss(weight=ce_weight, ignore_index=ignore_index)

        self.dice_loss = smp.losses.DiceLoss(
            mode=mode,
            classes=dice_cal_classes,
            ignore_index=ignore_index,
        )

        self.lovasz_loss = smp.losses.LovaszLoss(
            mode=mode,
            per_image=False,
            ignore_index=ignore_index,
            from_logits=True,
        )

        self.focal_loss = smp.losses.FocalLoss(
            mode=mode,
            alpha=focal_alpha,
            gamma=focal_gamma,
            ignore_index=ignore_index,
            normalized=False,
        )

        self.loss_weights = (1.0, 1.0, 0.75, 0.5) if loss_weights is None else loss_weights
        self.loss_weights = torch.as_tensor(self.loss_weights)

        # Register buffer for EMA class counts
        if dynamic_mode == "ema":
            self.register_buffer("class_counts", torch.ones(num_classes))

    def _update_weights(self, gt: Tensor) -> None:
        if self.dynamic_mode is None:
            return

        valid_mask = gt != self.ignore_index
        valid_gt = gt[valid_mask]
        if valid_gt.numel() == 0:
            return

        current_counts = torch.bincount(valid_gt.flatten(), minlength=self.num_classes).float()

        if self.dynamic_mode == "batch":
            counts = current_counts
        elif self.dynamic_mode == "ema":
            self.class_counts = self.class_counts.to(gt.device)
            self.class_counts = self.ema_decay * self.class_counts + (1 - self.ema_decay) * current_counts
            counts = self.class_counts
        else:
            raise ValueError(f"Unknown mode={self.dynamic_mode}")

        freq = counts / (counts.sum() + 1e-6)
        weights = 1.0 / (torch.sqrt(freq) + 0.1)
        weights = weights / weights.sum() * self.num_classes

        self.cross_entropy.weight = weights.to(gt.device)

    def _forward_loss(self, pred: Tensor, gt: Tensor, mask: Tensor | None = None):
        # Dynamically update CE weights if enabled
        self._update_weights(gt)

        if pred.shape[-2:] != gt.shape[-2:]:
            gt = torch.nn.functional.interpolate(gt[:, None], size=pred.shape[-2:], mode="nearest").squeeze(1)

        if mask is not None:
            assert pred.shape[0] == 1, f"Support only single-image predictions, got {pred.shape[0]} images."
            if mask.shape[-2:] != pred.shape[-2:]:
                mask = torch.nn.functional.interpolate(mask, size=pred.shape[-2:], mode="nearest")
            mask_nonzero = torch.nonzero(mask, as_tuple=False)  # [n, 3]
            pred = get_at("[b h w] c, n [3] -> n 3", pred.permute(0, -2, -1, 1), mask_nonzero)  # 1d image
            gt = get_at("[b h w], n [3] -> n 3", gt, mask_nonzero)  # 1d gt map

        # Keep tensors contiguous because some downstream losses (e.g., SMP DiceLoss)
        # use `view`, which requires contiguous memory.
        pred = pred.contiguous()
        gt = gt.contiguous()
        if mask is not None:
            mask = mask.contiguous()

        # Only calculate losses with non-zero weights
        losses = []
        loss_dict = {}
        loss_names = self.loss_names
        weights = self.loss_weights
        loss_fns = [self.dice_loss, self.cross_entropy, self.lovasz_loss, self.focal_loss]

        for i, (name, weight, fn) in enumerate(zip(loss_names, weights, loss_fns)):
            if weight > 0:
                val = fn(pred, gt)
                losses.append(val)
                loss_dict[name] = val.detach()
            else:
                loss_dict[name] = torch.tensor(0.0, device=pred.device)

        used_weights = [w for w in weights if w > 0]
        if losses:
            loss = loss_apply_weights(losses, used_weights)
        else:
            loss = torch.tensor(0.0, device=pred.device)

        return loss, edict(loss_dict)

    def forward(
        self, pred: Float[Tensor, "b c h w"] | list[Tensor], gt: Int[Tensor, "b h w"], mask: Tensor | None = None
    ):
        # Multiple outputs (deep supervision)
        if isinstance(pred, (tuple, list)):
            total_loss = 0.0
            loss_dict = {name: torch.tensor(0.0, device=pred[0].device) for name in self.loss_names}

            for p in pred:
                loss, l_dict = self._forward_loss(p, gt, mask)
                total_loss += loss

                for name in self.loss_names:
                    loss_dict[name] += l_dict[name]

            total_loss = total_loss / len(pred)

            for name in self.loss_names:
                loss_dict[name] /= len(pred)

            return total_loss, edict(loss_dict)
        # Single output
        else:
            return self._forward_loss(pred, gt, mask)


def boost_strap_update_label(
    model: torch.nn.Module,
    img: Tensor,
    label: Tensor,
    ratio: float = 0.2,
    ignore_index: int = 255,
    confidence_threshold: float = 0.9,
    balance_mode: str = "per_class",  # "per_class" or "global"
    min_pixels_per_class: int = 10,
) -> Tensor:
    """
    Dynamically generate pseudo-labels for unlabeled data with class-balanced sampling.

    Args:
        model: The segmentation model.
        img: Input image tensor [B, C, H, W].
        label: Original label tensor (used for shape/device reference).
        ratio: Ratio of pixels to select as pseudo-labels (0.0 to 1.0).
        ignore_index: Value to fill for unselected pixels (default: 255).
        confidence_threshold: Minimum confidence score to select a pixel (0.0 to 1.0).
        balance_mode: Sampling strategy:
            - "per_class": Select top-k pixels per class (ensures class balance)
            - "global": Select top-k pixels globally (original behavior)
        min_pixels_per_class: Minimum number of pixels to select per class (only for per_class mode).

    Returns:
        new_label: Tensor [B, H, W] with pseudo-labels at selected indices
                   and ignore_index elsewhere.
    """
    # 1. Ensure probabilistic/deterministic behavior for inference
    training_mode = model.training
    model.eval()

    with torch.no_grad():
        # 2. Forward pass
        logits = model(img)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        # 3. Calculate probabilities and confidence
        probs = torch.softmax(logits, dim=1)  # [B, C, H, W]
        b, num_classes, h, w = probs.shape

        # Get predicted class and confidence for each pixel
        max_probs, pred_labels = torch.max(probs, dim=1)  # [B, H, W]

        # Create new label tensor filled with ignore_index
        new_label = torch.full_like(label, ignore_index)

        if balance_mode == "global":
            # Original global Top-K approach (with confidence threshold)
            num_pixels = h * w
            k = int(num_pixels * ratio)
            k = max(1, k)

            for batch_idx in range(b):
                confidence = max_probs[batch_idx].view(-1)  # [H*W]
                pred_flat = pred_labels[batch_idx].view(-1)  # [H*W]

                # Apply confidence threshold
                valid_mask = confidence >= confidence_threshold
                if valid_mask.sum() == 0:
                    continue

                valid_confidence = confidence[valid_mask]
                valid_pred = pred_flat[valid_mask]
                valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)

                # Select top-k from valid pixels
                k_actual = min(k, valid_confidence.numel())
                if k_actual == 0:
                    continue

                _, topk_idx = torch.topk(valid_confidence, k_actual, largest=True)
                selected_indices = valid_indices[topk_idx]
                selected_preds = valid_pred[topk_idx]

                # Scatter into new_label
                new_label_flat = new_label[batch_idx].view(-1)
                new_label_flat.scatter_(0, selected_indices, selected_preds)

        elif balance_mode == "per_class":
            # Per-class balanced sampling
            num_pixels = h * w
            total_budget = int(num_pixels * ratio)

            for batch_idx in range(b):
                confidence = max_probs[batch_idx].view(-1)  # [H*W]
                pred_flat = pred_labels[batch_idx].view(-1)  # [H*W]

                # Apply confidence threshold
                valid_mask = confidence >= confidence_threshold
                if valid_mask.sum() == 0:
                    continue

                valid_confidence = confidence[valid_mask]
                valid_pred = pred_flat[valid_mask]
                valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)

                # Count predictions per class
                unique_classes = torch.unique(valid_pred)
                num_present_classes = unique_classes.numel()

                if num_present_classes == 0:
                    continue

                # Allocate budget per class (proportional to predicted pixels, but with minimum)
                class_pixel_counts = torch.bincount(valid_pred, minlength=num_classes)
                present_class_counts = class_pixel_counts[unique_classes]

                # Calculate per-class budget (ensure minimum + proportional distribution)
                base_budget = min_pixels_per_class * num_present_classes
                remaining_budget = max(0, total_budget - base_budget)

                # Proportional allocation for remaining budget
                if present_class_counts.sum() > 0:
                    class_ratios = present_class_counts.float() / present_class_counts.sum()
                    class_budgets = (class_ratios * remaining_budget).long()
                    class_budgets += min_pixels_per_class
                else:
                    class_budgets = torch.full_like(present_class_counts, min_pixels_per_class)

                # Select pixels per class
                selected_indices_list = []
                selected_preds_list = []

                for cls_idx, cls in enumerate(unique_classes):
                    # Find pixels belonging to this class
                    cls_mask = valid_pred == cls
                    cls_confidence = valid_confidence[cls_mask]
                    cls_indices = valid_indices[cls_mask]

                    if cls_confidence.numel() == 0:
                        continue

                    # Select top-k for this class
                    k_cls = min(class_budgets[cls_idx].item(), cls_confidence.numel())
                    if k_cls > 0:
                        _, topk_idx = torch.topk(cls_confidence, k_cls, largest=True)
                        selected_indices_list.append(cls_indices[topk_idx])
                        selected_preds_list.append(torch.full((k_cls,), cls, device=cls_indices.device))

                if len(selected_indices_list) > 0:
                    all_selected_indices = torch.cat(selected_indices_list)
                    all_selected_preds = torch.cat(selected_preds_list)

                    # Scatter into new_label
                    new_label_flat = new_label[batch_idx].view(-1)
                    new_label_flat.scatter_(0, all_selected_indices, all_selected_preds)
        else:
            raise ValueError(f"Unknown balance_mode: {balance_mode}")

    # Restore model mode
    model.train(training_mode)

    return new_label


# * --- Test --- #


def test_hyper_seg_loss():
    pred = torch.randn(2, 5, 32, 32)
    gt = torch.randint(0, 5, (2, 32, 32))
    loss_fn = HyperSegmentationLoss()
    loss, loss_dict = loss_fn(pred, gt)
    print(f"Loss: {loss}")

    # Test bootstrap
    model = torch.nn.Conv2d(3, 5, 3, padding=1)
    img = torch.randn(2, 3, 32, 32)
    label = torch.zeros(2, 32, 32, dtype=torch.long)
    new_label = boost_strap_update_label(model, img, label, ratio=0.1)
    print(f"Bootstrap Label Shape: {new_label.shape}")
    print(f"Non-ignore pixels: {(new_label != 255).sum()}")
    print(f"Total pixels: {new_label.numel()}")
    print(f"Ratio: {(new_label != 255).sum() / new_label.numel()}")


if __name__ == "__main__":
    test_hyper_seg_loss()

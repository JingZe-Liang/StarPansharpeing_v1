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
    loss_names = ["dice_loss", "ce_loss", "lovasz_loss"]

    def __init__(
        self,
        mode: str = "multiclass",
        dice_cal_classes: list[int] | None = None,
        ce_weight: list[float] | Tensor | None = None,
        ignore_index: int = 255,
        loss_weights: list[float] | None = None,
    ):
        super().__init__()
        ce_weight = torch.as_tensor(ce_weight).cuda() if ce_weight is not None else None

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

        self.loss_weights = loss_weights or (1.0, 1.0, 0.75)
        self.loss_weights = torch.as_tensor(self.loss_weights)

    def _forward_loss(self, pred: Tensor, gt: Tensor, mask: Tensor | None = None):
        if pred.shape[-2:] != gt.shape[-2:]:
            gt = torch.nn.functional.interpolate(gt[:, None], size=pred.shape[-2:], mode="nearest").squeeze(1)

        if mask is not None:
            assert pred.shape[0] == 1, f"Support only single-image predictions, got {pred.shape[0]} images."
            if mask.shape[-2:] != pred.shape[-2:]:
                mask = torch.nn.functional.interpolate(mask, size=pred.shape[-2:], mode="nearest")
            mask_nonzero = torch.nonzero(mask, as_tuple=False)  # [n, 3]
            pred = get_at("[b h w] c, n [3] -> n 3", pred.permute(0, -2, -1, 1), mask_nonzero)  # 1d image
            gt = get_at("[b h w], n [3] -> n 3", gt, mask_nonzero)  # 1d gt map

        # Only calculate losses with non-zero weights
        losses = []
        loss_dict = {}
        loss_names = self.loss_names
        weights = self.loss_weights
        loss_fns = [self.dice_loss, self.cross_entropy, self.lovasz_loss]

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
            loss_dict = {name: torch.tensor(0.0, device=pred.device) for name in self.loss_names}

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
) -> Tensor:
    """
    Dynamically generate pseudo-labels for unlabeled data (Bootstrap / Self-Training).

    Args:
        model: The segmentation model.
        img: Input image tensor [B, C, H, W].
        label: Original label tensor (used for shape/device reference).
        ratio: Ratio of pixels to select as pseudo-labels (0.0 to 1.0).
            Pixels with the lowest entropy (highest confidence) are selected.
        ignore_index: Value to fill for unselected pixels (default: 255).

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

        # 3. Calculate Entropy
        # probs: [B, C, H, W]
        probs = torch.softmax(logits, dim=1)
        b, c, h, w = probs.shape

        # entropy: [B, H, W] = -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)

        # 4. Select top-k pixels (lowest entropy)
        num_pixels = h * w
        k = int(num_pixels * ratio)
        k = max(1, k)  # Ensure at least 1 pixel is selected if ratio > 0

        # Flatten for topk
        entropy_flat = entropy.view(b, -1)

        # Find indices of the k smallest entropy values (highest confidence)
        _, indices = torch.topk(entropy_flat, k, dim=1, largest=False)

        # 5. Generate Pseudo Labels
        pred_labels = torch.argmax(logits, dim=1)  # [B, H, W]
        pred_flat = pred_labels.view(b, -1)

        # Create new label tensor filled with ignore_index
        new_label = torch.full_like(label, ignore_index)
        new_label_flat = new_label.view(b, -1)

        # Gather selected predictions
        selected_preds = torch.gather(pred_flat, 1, indices)

        # Scatter predictions into the new label tensor
        new_label_flat.scatter_(1, indices, selected_preds)

        # Reshape back to [B, H, W]
        new_label = new_label_flat.view(b, h, w)

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

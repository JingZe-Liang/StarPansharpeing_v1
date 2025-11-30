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
            gt = torch.nn.functional.interpolate(gt, size=pred.shape[-2:], mode="nearest", align_corners=False)

        if mask is not None:
            assert pred.shape[0] == 1, f"Support only single-image predictions, got {pred.shape[0]} images."
            if mask.shape[-2:] != pred.shape[-2:]:
                mask = torch.nn.functional.interpolate(mask, size=pred.shape[-2:], mode="nearest", align_corners=False)
            mask_nonzero = torch.nonzero(mask, as_tuple=False)  # [n, 3]
            pred = get_at("[b h w] c, n [3] -> n 3", pred.permute(0, -2, -1, 1), mask_nonzero)  # 1d image
            gt = get_at("[b h w], n [3] -> n 3", gt, mask_nonzero)  # 1d gt map

        # Loss
        dice_loss = self.dice_loss(pred, gt)
        ce_loss = self.cross_entropy(pred, gt)
        lovasz_loss = self.lovasz_loss(pred, gt)
        loss = loss_apply_weights([dice_loss, ce_loss, lovasz_loss], self.loss_weights)
        loss_dict = edict(
            {
                "dice_loss": dice_loss.detach(),
                "ce_loss": ce_loss.detach(),
                "lovasz_loss": lovasz_loss.detach(),
            }
        )
        return loss, loss_dict

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


# * --- Test --- #


def test_hyper_seg_loss():
    pred = torch.randn(2, 5, 32, 32)
    gt = torch.randint(0, 5, (2, 32, 32))
    loss_fn = HyperSegmentationLoss()
    loss, loss_dict = loss_fn(pred, gt)

    print(loss)


if __name__ == "__main__":
    test_hyper_seg_loss()

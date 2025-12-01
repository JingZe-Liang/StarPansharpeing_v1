import segmentation_models_pytorch as smp
import torch
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

    def forward(self, pred: Float[Tensor, "b c h w"], gt: Int[Tensor, "b h w"]):
        dice_loss = self.dice_loss(pred, gt)
        ce_loss = self.cross_entropy(pred, gt)
        lovasz_loss = self.lovasz_loss(pred, gt)
        loss = loss_apply_weights([dice_loss, ce_loss, lovasz_loss], self.loss_weights)

        return loss, {
            "dice_loss": dice_loss,
            "ce_loss": ce_loss,
            "lovasz_loss": lovasz_loss,
        }


# * --- Test --- #


def test_hyper_seg_loss():
    pred = torch.randn(2, 5, 32, 32)
    gt = torch.randint(0, 5, (2, 32, 32))
    loss_fn = HyperSegmentationLoss()
    loss, loss_dict = loss_fn(pred, gt)

    print(loss)


if __name__ == "__main__":
    test_hyper_seg_loss()

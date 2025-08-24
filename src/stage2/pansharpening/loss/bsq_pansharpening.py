import einops
import torch.nn as nn


class BinaryQuantizedSRLoss(nn.Module):
    def __init__(self, class_dim_take_out: bool = True):
        super().__init__()
        self.class_dim_take_out = class_dim_take_out

    def forward(self, pred, label):
        if self.class_dim_take_out:
            pred = einops.rearrange(pred, "b (c k) h w -> b k c h w", k=2)
        else:
            assert pred.ndim == 5, "pred must be 5 dims (bs, 2, c, h, w)"

        b, _, c, h, w = pred.shape
        b_t, c_t, h_t, w_t = label.shape
        assert b == b_t and c == c_t and h == h_t and w == w_t, (
            "pred and label shape must be same"
        )

        loss = nn.functional.cross_entropy(
            pred,
            label,
            reduction="mean",
            ignore_index=-1,
        )

        loss_dict = {"loss": loss}

        return loss, loss_dict

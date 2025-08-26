import torch
from beartype import beartype
from jaxtyping import Array, Float
from torch import Tensor

type Image = Float[Tensor, "b c h w"]
type EndMember = Float[Tensor, "c_out c_in"]
type Abunds = Float[Tensor, "b c_in h w"]


def SAD_loss(y_true: Image, y_pred: Image):
    y_true = torch.nn.functional.normalize(y_true, dim=1, p=2)
    y_pred = torch.nn.functional.normalize(y_pred, dim=1, p=2)

    A = torch.mul(y_true, y_pred)
    A = torch.sum(A, dim=1)
    sad = torch.acos(A)
    loss = torch.mean(sad)
    return loss


def abunds_loss(abunds: Abunds):
    loss = torch.sqrt(abunds).mean()
    return loss


def endmember_tv_loss(end_members: EndMember):
    return torch.abs(end_members[:, 1:] - end_members[:, :-1]).sum()


@beartype
class UnmixingLoss(torch.nn.Module):
    def __init__(self, weights: list[float] | tuple[float, ...]):
        super().__init__()
        self.weights = weights

    def forward(
        self,
        hyper_in: Image,
        hyper_recon: Image,
        abunds: Abunds,
        end_members: EndMember,
    ):
        # loss1 = torch.nn.functional.mse_loss(hyper_recon, hyper_in)
        sad_loss = SAD_loss(hyper_recon, hyper_in)
        abds_loss = abunds_loss(abunds)
        endmember_loss = endmember_tv_loss(end_members)

        total_loss = (
            self.weights[0] * sad_loss
            + self.weights[1] * abds_loss
            + self.weights[2] * endmember_loss
        )

        return total_loss, dict(
            sad_loss=sad_loss, abunds_loss=abds_loss, endmember_loss=endmember_loss
        )


# * --- Test --- * #


def test_loss():
    input = torch.randn(2, 128, 64, 64)
    recon = torch.randn(2, 128, 64, 64)
    abunds = torch.randn(2, 4, 64, 64)
    endmembers = torch.randn(128, 4)

    loss_fn = UnmixingLoss([1.0, 1.0, 1.0])
    total_loss, loss_dict = loss_fn(
        hyper_in=input,
        hyper_recon=recon,
        abunds=abunds,
        end_members=endmembers,
    )
    print("Total Loss:", total_loss.item())
    print("Loss Dictionary:", loss_dict)

    assert isinstance(total_loss, torch.Tensor)
    assert isinstance(loss_dict, dict)

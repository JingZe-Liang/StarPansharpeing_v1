import torch
from beartype import beartype
from jaxtyping import Array, Float
from kornia.filters import laplacian
from torch import Tensor

type Image = Float[Tensor, "b bands h w"]
type EndMember = Float[Tensor, "n_endmember bands"]
type Abunds = Float[Tensor, "b bands h w"]


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


def mini_volumn_loss(end_members: EndMember, delta: float = 1.0):
    bands = end_members.shape[1]
    n_endmember = end_members.shape[0]
    edm_mean = end_members.mean(dim=1, keepdim=True)
    loss = delta * ((end_members - edm_mean) ** 2).sum() / bands / n_endmember
    return loss


def sparse_loss(abunds: Abunds, alpha: float = 1.0):
    loss = abunds.abs().sqrt().sum(dim=1).mean()
    return alpha * loss


def laplacian_loss(abunds1: Abunds, abunds2: Abunds):
    lap1 = laplacian(abunds1, 3)
    lap2 = laplacian(abunds2, 3)
    loss = torch.nn.functional.mse_loss(lap1, lap2)
    return loss


_loss_registry = {
    "sad": SAD_loss,
    "abunds": abunds_loss,
    "endmember_tv": endmember_tv_loss,
    "mini_volumn": mini_volumn_loss,
    "sparse": sparse_loss,
    "laplacian": laplacian_loss,
}


@beartype
class UnmixingLoss(torch.nn.Module):
    def __init__(self, weights: list[float] | tuple[float, ...] | None = None):
        super().__init__()
        self.weights = weights if weights is not None else [1.0, 0.35, 0.1]

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

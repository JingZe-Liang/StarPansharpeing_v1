import torch
from beartype import beartype
from jaxtyping import Array, Float
from kornia.filters import laplacian
from torch import Tensor

type Image = Float[Tensor, "b bands h w"]
type EndMember = Float[Tensor, "n_endmember bands"]
type Abunds = Float[Tensor, "b bands h w"]


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

    weighted_loss = loss_stk * weights
    loss_out = weighted_loss.sum()

    if ret_loss_parts:
        loss_parts = weighted_loss.unbind()
        return loss_out, loss_parts
    else:
        return loss_out


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
    return (end_members[:, 1:] - end_members[:, :-1]).abs().mean()


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


_unmixing_loss_registry = {
    "sad": SAD_loss,
    "abunds": abunds_loss,
    "endmember_tv": endmember_tv_loss,
    "mini_volumn": mini_volumn_loss,
    "sparse": sparse_loss,
    "laplacian": laplacian_loss,
}


@beartype
class UnmixingLoss(torch.nn.Module):
    loss_names = ["sad_loss", "abunds_loss", "endmember_loss"]

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
        sad_loss = SAD_loss(hyper_recon, hyper_in)
        abds_loss = abunds_loss(abunds)
        endmember_loss = endmember_tv_loss(end_members)

        total_loss, (sad_loss, abds_loss, endmember_loss) = loss_apply_weights(
            [sad_loss, abds_loss, endmember_loss],
            weights=self.weights,
            ret_loss_parts=True,
        )

        return total_loss, dict(
            sad_loss=sad_loss, abunds_loss=abds_loss, endmember_loss=endmember_loss
        )


# * --- Test --- * #


def test_loss():
    input = torch.randn(2, 128, 64, 64)
    recon = torch.randn(2, 128, 64, 64)
    abunds = torch.randn(2, 4, 64, 64).relu_()
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


if __name__ == "__main__":
    test_loss()

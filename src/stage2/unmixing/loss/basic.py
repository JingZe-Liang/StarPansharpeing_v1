from typing import TypeAlias

import torch
from beartype import beartype
from jaxtyping import Float
from kornia.filters import laplacian
from torch import Tensor

from src.utilities.config_utils import function_config_to_basic_types
from src.utilities.train_utils.state import StepsCounter

Image: TypeAlias = Float[Tensor, "b bands h w"]
EndMember: TypeAlias = Float[Tensor, "n_endmember bands"]
Abunds: TypeAlias = Float[Tensor, "b bands h w"]


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


def reconstruction_loss(hyper_in: Image, hyper_recon: Image, recon_type: str = "mse"):
    if recon_type == "mse":
        loss = torch.nn.functional.mse_loss(hyper_recon, hyper_in)
    elif recon_type == "l1":
        loss = torch.nn.functional.l1_loss(hyper_recon, hyper_in)
    elif recon_type == "sad":
        hyper_in = torch.nn.functional.normalize(hyper_in, dim=1, p=2)
        hyper_recon = torch.nn.functional.normalize(hyper_recon, dim=1, p=2)
        A = torch.mul(hyper_in, hyper_recon)
        A = A.sum(dim=1)
        loss = torch.acos(A)
        loss = loss.mean()
    elif recon_type == "sad_cos":
        bs, band, h, w = hyper_recon.shape
        hyper_recon = hyper_recon.reshape(bs, band, h * w)
        hyper_in = hyper_in.reshape(bs, band, h * w)
        loss = torch.acos(torch.cosine_similarity(hyper_in, hyper_recon, dim=1))
        loss = loss.mean()
    else:
        raise ValueError(f"recon_type {recon_type} not supported")

    return loss


def endmember_tv_loss(end_members: EndMember):
    # shape: [em, c]
    return (end_members[1:] - end_members[:-1]).abs().mean()


def endmember_volumn_loss(endmembers: EndMember):
    # Compute mean endmember
    em_mean = endmembers.mean(dim=1, keepdim=True)  # [n_endmember, 1]
    # Minimum volume loss (distance from mean)
    loss = ((endmembers - em_mean) ** 2).mean()
    return loss


def endmember_spectral_loss(endmembers: EndMember):
    # Compute mean endmember spectrum
    em_mean = endmembers.mean(dim=0)  # [bands]

    # Vectorized computation of spectral angles
    # Normalize endmembers and mean spectrum
    em_norms = torch.norm(endmembers, dim=1)  # [n_endmember]
    mean_norm = torch.norm(em_mean)  # scalar

    # Compute dot products
    dot_products = torch.matmul(endmembers, em_mean)  # [n_endmember]

    # Compute cosine similarities
    cos_sims = dot_products / (em_norms * mean_norm + 1e-6)
    cos_sims = torch.clamp(cos_sims, -1.0, 1.0)

    # Compute spectral angles and average
    sad_values = torch.acos(cos_sims)
    loss = sad_values.mean()
    return loss


def abunds_sparse_loss(abunds: Abunds):
    loss_sparse = torch.tensor(0.0, device=abunds.device)
    # loss_sparse = abunds.abs().sqrt().mean()  # close to 0

    # diff with others
    for i in range(abunds.shape[1]):
        for j in range(i + 1, abunds.shape[1]):
            ab_i = abunds[:, i]
            ab_j = abunds[:, j]
            loss_ij = torch.nn.functional.mse_loss(ab_i, ab_j)
            loss_sparse += 1.0 / (loss_ij + 1e-6)

    # but not all zero
    loss_ab_zero = torch.tensor(0.0, device=abunds.device)
    for i in range(abunds.shape[1]):
        ab_i = abunds[:, i].abs().mean()
        loss_ab_i = 1.0 / ab_i  # far from 0.
        loss_ab_zero += loss_ab_i
    loss_ab_zero = loss_ab_zero / abunds.shape[1]
    return loss_sparse + loss_ab_zero * 0.1


def abunds_physical_loss(abunds: Abunds):
    # no negative values
    neg_loss = torch.relu(-abunds).mean()
    # sum to one
    sum_loss = ((abunds.sum(dim=1) - 1) ** 2).mean()
    return neg_loss, sum_loss


def abunds_laplacian_loss(abunds1: Abunds, abunds2: Abunds):
    lap1 = laplacian(abunds1, 3)
    lap2 = laplacian(abunds2, 3)
    loss = torch.nn.functional.mse_loss(lap1, lap2)
    return loss


LossAlias = {
    "em_tv": endmember_tv_loss,
    "em_vol": endmember_volumn_loss,
    "em_spec": endmember_spectral_loss,
    "ab_sparse": abunds_sparse_loss,
    "ab_phys": abunds_physical_loss,
}


class UnmixingLoss(torch.nn.Module):
    loss_names = ["sad_loss", "abunds_loss", "endmember_loss"]

    # FIXME: bug somewhere
    """
    This is wield that the model will output NaN using this loss.
    """

    def __init__(self, weights: list[float] | tuple[float, ...] | None = None):
        super().__init__()
        self.weights = weights if weights is not None else [1.0, 0.35, 0.1]

    def forward(
        self,
        hyper_in: Image,
        hyper_recon: Image,
        abunds_pred: Abunds,
        endmembers: EndMember,
    ):
        sad_loss = SAD_loss(hyper_in, hyper_recon)
        abds_loss = abunds_loss(abunds_pred)
        endmember_loss = endmember_tv_loss(endmembers)

        total_loss, (sad_loss, abds_loss, endmember_loss) = loss_apply_weights(
            [sad_loss, abds_loss, endmember_loss],
            weights=self.weights,
            ret_loss_parts=True,
        )

        return total_loss, dict(sad_loss=sad_loss, abunds_loss=abds_loss, endmember_loss=endmember_loss)


class StagedUnmixingLoss(torch.nn.Module):
    """
    Simplified Staged Unmixing Loss based on SSAF-Net training strategy

    This loss function implements a two-stage training approach for simple networks:
    Stage 1 (steps < switch_step): Focus on reconstruction and abundance constraints
    Stage 2 (steps >= switch_step): Add endmember volume and spectral constraints

    Network outputs expected: abunds, recon, em (no complex VAE components)

    Based on SSAF-Net's training strategy with separate sub-loss functions
    """

    loss_names = [
        "rec_loss",
        "ab_phys_neg",
        "ab_phys_sum",
        "ab_sparse",
        "em_tv",
        "em_vol",
        "em_spec",
        "total_loss",
    ]

    @function_config_to_basic_types
    def __init__(
        self,
        switch_step: int = 6000,
        weights: dict = {},
        recon_type="sad",
    ):
        super().__init__()
        self.switch_step = switch_step
        self.steps_counter = StepsCounter()  # may raise error
        self.recon_type = recon_type
        self.weights = weights

    def _reconstruction_loss(self, hyper_in: Image, hyper_recon: Image) -> Tensor:
        loss = reconstruction_loss(hyper_in, hyper_recon, recon_type=self.recon_type)
        return loss

    def _abundance_constraint_loss(self, abunds: Abunds, abunds_fcls: Abunds | None = None) -> Tensor:
        if abunds_fcls is not None:
            # Compare with FCLS results (SSAF-Net's stage 1 approach)
            loss = torch.nn.functional.mse_loss(abunds, abunds_fcls)
        else:
            # Use sparsity constraint as fallback
            loss = abunds.abs().sqrt().mean()
        return loss

    def _endmember_losses(self, endmembers: EndMember):
        endmember_losses = torch.tensor(0.0, device=endmembers.device)
        endmember_losses_dict = {}
        endmember_loss_types = ["em_tv", "em_vol", "em_spec"]
        for loss_type in endmember_loss_types:
            if (w := self.weights.get(loss_type, 0.0)) > 0.0:
                loss = LossAlias[loss_type](endmembers) * w
                endmember_losses += loss
                endmember_losses_dict[loss_type] = loss.detach()
        return endmember_losses, endmember_losses_dict

    def _abunds_losses(self, abunds: Abunds):
        abunds_losses = torch.tensor(0.0, device=abunds.device)
        abunds_losses_dict = {}
        abunds_loss_types = ["ab_phys", "ab_sparse"]
        for loss_type in abunds_loss_types:
            if (w := self.weights.get(loss_type, 0.0)) > 0.0:
                if loss_type == "ab_phys":
                    neg_loss, sum_loss = LossAlias[loss_type](abunds)
                    loss = (neg_loss + sum_loss) * w
                    abunds_losses += loss
                    abunds_losses_dict["ab_phys_neg"] = neg_loss.detach()
                    abunds_losses_dict["ab_phys_sum"] = sum_loss.detach()
                else:
                    loss = LossAlias[loss_type](abunds) * w
                    abunds_losses += loss
                    abunds_losses_dict[loss_type] = loss.detach()
        return abunds_losses, abunds_losses_dict

    def _stage1_loss(
        self,
        hyper_in: Image,
        abunds_pred: Abunds,
        hyper_recon: Image,
        abunds_fcls: Abunds | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        # Reconstruction loss
        rec_loss = self._reconstruction_loss(hyper_in, hyper_recon) * self.weights.get("rec_loss", 1.0)

        # Abundance constraint loss
        if (w := self.weights.get("ab_sparse", 0.0)) > 0.0 and abunds_fcls is not None:
            ab_pre_loss = torch.nn.functional.mse_loss(abunds_pred, abunds_fcls) * w
            rec_loss += ab_pre_loss
        loss_dict = {"ab_pre": ab_pre_loss, "rec_loss": rec_loss}

        # leave the endmember unchanged

        return rec_loss, loss_dict

    def _stage2_loss(
        self,
        hyper_in: Image,
        abunds_pred: Abunds,
        endmembers: EndMember,
        hyper_recon: Image,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        # Reconstruction loss
        rec_loss = self._reconstruction_loss(hyper_in, hyper_recon) * self.weights.get("rec_loss", 1.0)
        loss_dict = {"rec_loss": rec_loss.detach().clone()}

        # Endmember losses
        em_total_loss, em_loss_dict = self._endmember_losses(endmembers)
        rec_loss += em_total_loss

        # Abundance losses
        ab_total_loss, ab_loss_dict = self._abunds_losses(abunds_pred)
        rec_loss += ab_total_loss

        loss_dict.update(
            {
                **em_loss_dict,
                **ab_loss_dict,
                "total_loss": rec_loss,
            }
        )

        return rec_loss, loss_dict

    def forward(
        self,
        hyper_in: Image,
        abunds_pred: Abunds,
        endmembers: EndMember,
        hyper_recon: Image,
        abunds_fcls: Abunds | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Compute staged unmixing loss for simple network outputs

        Parameters
        ----------
        hyper_in : Image
            Input hyperspectral image [b bands h w]
        abunds_pred : Abunds
            Predicted abundances [b n_endmember h w]
        endmembers : EndMember
            Predicted endmembers [n_endmember bands]
        hyper_recon : Image
            Reconstructed image [b bands h w]
        abunds_fcls : Abunds | None
            FCLS reference abundances (optional) [b n_endmember h w]

        Returns
        -------
        tuple[Tensor, dict[str, Tensor]]
            Total loss and loss dictionary
        """
        current_step = self.steps_counter["train"]

        # Stage 1: Reconstruction and abundance constraints
        if current_step < self.switch_step:
            total_loss, loss_dict = self._stage1_loss(hyper_in, abunds_pred, hyper_recon, abunds_fcls)
        # Stage 2: Add endmember property constraints
        else:
            total_loss, loss_dict = self._stage2_loss(hyper_in, abunds_pred, endmembers, hyper_recon)

        return total_loss, loss_dict


# * --- Test --- * #


def test_loss():
    input = torch.randn(2, 128, 64, 64)
    recon = torch.randn(2, 128, 64, 64)
    abunds = torch.randn(2, 4, 64, 64).relu_()
    endmembers = torch.randn(4, 128)  # Note: shape should be [n_endmembers, bands]

    loss_fn = UnmixingLoss([1.0, 1.0, 1.0])
    total_loss, loss_dict = loss_fn(
        hyper_in=input,
        hyper_recon=recon,
        abunds=abunds,
        endmembers=endmembers,
    )
    print("Total Loss:", total_loss.item())
    print("Loss Dictionary:", loss_dict)

    assert isinstance(total_loss, torch.Tensor)
    assert isinstance(loss_dict, dict)


def test_staged_loss():
    """
    Test function for StagedUnmixingLoss
    """
    print("Testing StagedUnmixingLoss...")

    # Initialize StepsCounter first
    from src.utilities.train_utils.state import StepsCounter

    StepsCounter(step_names=["train"])

    # Create test data
    batch_size, bands, height, width = 2, 128, 64, 64
    n_endmembers = 4

    # Input tensors
    hyper_in = torch.randn(batch_size, bands, height, width)
    abunds_pred = torch.rand(batch_size, n_endmembers, height, width).relu_()
    endmembers = torch.rand(n_endmembers, bands)
    hyper_recon = torch.randn(batch_size, bands, height, width)
    abunds_fcls = torch.rand(batch_size, n_endmembers, height, width).relu_()

    # Test Stage 1 loss
    print("\n--- Testing Stage 1 Loss ---")
    loss_fn = StagedUnmixingLoss(switch_step=1000)  # Will use stage 1

    # Mock the steps counter to simulate stage 1
    loss_fn.steps_counter.update("train", 500)

    total_loss, loss_dict = loss_fn(
        hyper_in=hyper_in,
        abunds_pred=abunds_pred,
        endmembers=endmembers,
        hyper_recon=hyper_recon,
        abunds_fcls=abunds_fcls,
    )

    print(f"Stage 1 Total Loss: {total_loss.item():.6f}")
    print("Stage 1 Loss Dictionary:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.6f}")

    # Test Stage 2 loss
    print("\n--- Testing Stage 2 Loss ---")
    # Mock the steps counter to simulate stage 2
    loss_fn.steps_counter.update("train", 1500)

    total_loss, loss_dict = loss_fn(
        hyper_in=hyper_in,
        abunds_pred=abunds_pred,
        endmembers=endmembers,
        hyper_recon=hyper_recon,
        abunds_fcls=abunds_fcls,
    )

    print(f"Stage 2 Total Loss: {total_loss.item():.6f}")
    print("Stage 2 Loss Dictionary:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.6f}")

    # Test without FCLS abundances
    print("\n--- Testing without FCLS abundances ---")
    # loss_fn.steps_counter.update('train', 500)  # Back to stage 1
    loss_fn.steps_counter["train"] = 500  # Directly set to stage 1

    total_loss, loss_dict = loss_fn(
        hyper_in=hyper_in,
        abunds_pred=abunds_pred,
        endmembers=endmembers,
        hyper_recon=hyper_recon,
        abunds_fcls=None,
    )

    print(f"Stage 1 (no FCLS) Total Loss: {total_loss.item():.6f}")
    print("Stage 1 (no FCLS) Loss Dictionary:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.6f}")

    # Validate outputs
    assert isinstance(total_loss, torch.Tensor)
    assert isinstance(loss_dict, dict)
    assert all(isinstance(v, torch.Tensor) for v in loss_dict.values())
    assert len(loss_dict) == 4  # Should have 4 loss components

    print("\nStagedUnmixingLoss test passed!")


if __name__ == "__main__":
    test_loss()
    print("\n" + "=" * 50)
    test_staged_loss()

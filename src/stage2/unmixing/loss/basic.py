from typing import TypeAlias

import einx
import torch
from beartype import beartype
from jaxtyping import Array, Float
from kornia.filters import laplacian
from torch import Tensor

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


def SAD_loss(gt: Image, pred: Image):
    """Spectral Angle Distance Loss"""

    gt = einx.rearrange("b c h w -> b c (h w)", gt)
    pred = einx.rearrange("b c h w -> b c (h w)", pred)
    sad_loss = torch.acos(torch.cosine_similarity(gt, pred, dim=1))
    sad_loss = sad_loss.mean()  # mean across batch and pixels
    return sad_loss


def abunds_loss(abunds: Abunds):
    loss = torch.sqrt(torch.abs(abunds)).mean()
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

        return total_loss, dict(
            sad_loss=sad_loss, abunds_loss=abds_loss, endmember_loss=endmember_loss
        )


@beartype
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
        "reconstruction",
        "abundance_constraint",
        "endmember_volume",
        "endmember_spectral",
    ]

    def __init__(
        self,
        switch_step: int = 6000,
        lambda_pre: float = 10.0,
        lambda_vol: float = 10.0,
        lambda_sad: float = 5.0,
        lambda_abunds_physical: float = 1.0,
        recon_type="sad",
    ):
        """
        Initialize staged unmixing loss with SSAF-Net hyperparameters

        Parameters
        ----------
        switch_step : int
            Step threshold to switch from stage 1 to stage 2 training
        lambda_y2 : float
            Weight for secondary reconstruction loss
        lambda_pre : float
            Weight for abundance constraint loss
        lambda_vol : float
            Weight for endmember volume loss
        lambda_sad : float
            Weight for endmember spectral angle loss
        """
        super().__init__()
        self.switch_step = switch_step
        self.steps_counter = StepsCounter()

        # SSAF-Net inspired loss weights
        self.lambda_pre = lambda_pre
        self.lambda_vol = lambda_vol
        self.lambda_sad = lambda_sad
        self.lambda_abunds_physical = lambda_abunds_physical
        self.recon_type = recon_type

    def _reconstruction_loss(self, hyper_in: Image, hyper_recon: Image) -> Tensor:
        """
        Compute reconstruction loss between input and reconstructed image

        Based on SSAF-Net's dual reconstruction loss strategy

        Parameters
        ----------
        hyper_in : Image
            Input hyperspectral image [b bands h w]
        hyper_recon : Image
            Reconstructed image [b bands h w]

        Returns
        -------
        Tensor
            Reconstruction loss
        """
        # Primary reconstruction loss (MSE)
        if self.recon_type == "mse":
            loss = torch.nn.functional.mse_loss(hyper_recon, hyper_in)
        elif self.recon_type == "l1":
            loss = torch.nn.functional.l1_loss(hyper_recon, hyper_in)
        elif self.recon_type == "sad":
            bs, band, h, w = hyper_recon.shape
            hyper_recon = hyper_recon.reshape(bs, band, h * w)
            hyper_in = hyper_in.reshape(bs, band, h * w)
            loss = torch.acos(torch.cosine_similarity(hyper_in, hyper_recon, dim=1))
            loss = loss.mean()

        return loss

    def _abundance_constraint_loss(
        self, abunds: Abunds, abunds_fcls: Abunds | None = None
    ) -> Tensor:
        """
        Compute abundance constraint loss

        Based on SSAF-Net's abundance consistency loss with FCLS reference

        Parameters
        ----------
        abunds : Abunds
            Predicted abundances [b n_endmember h w]
        abunds_fcls : Abunds | None
            FCLS reference abundances (optional) [b n_endmember h w]

        Returns
        -------
        Tensor
            Abundance constraint loss
        """
        if abunds_fcls is not None:
            # Compare with FCLS results (SSAF-Net's stage 1 approach)
            loss = torch.nn.functional.mse_loss(abunds, abunds_fcls)
        else:
            # Use sparsity constraint as fallback
            loss = abunds.abs().sqrt().mean()
        return loss

    def _endmember_volume_loss(self, endmembers: EndMember) -> Tensor:
        """
        Compute minimum volume loss for endmembers

        Based on SSAF-Net's endmember minimum volume constraint:
        loss_minvol = ((em_tensor - em_bar) ** 2).sum() / pixels / endmember_number / band_number

        Parameters
        ----------
        endmembers : EndMember
            Endmember tensors [n_endmember bands]

        Returns
        -------
        Tensor
            Minimum volume loss
        """
        # Compute mean endmember
        em_mean = endmembers.mean(dim=1, keepdim=True)  # [n_endmember, 1]

        # Minimum volume loss (distance from mean)
        n_endmembers, n_bands = endmembers.shape
        loss = ((endmembers - em_mean) ** 2).sum() / n_endmembers / n_bands

        return loss

    def _endmember_spectral_loss(self, endmembers: EndMember) -> Tensor:
        """
        Compute spectral angle loss for endmembers

        Based on SSAF-Net's spectral angle divergence loss:
        em_bar = em_tensor.mean(dim=0)
        aa = (em_tensor * em_bar).sum(dim=2)
        sad = torch.acos(aa / (em_bar_norm + 1e-6) / (em_tensor_norm + 1e-6))

        Parameters
        ----------
        endmembers : EndMember
            Endmember tensors [n_endmember bands]

        Returns
        -------
        Tensor
            Spectral angle loss
        """
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

    def _abunds_physical_loss(self, abunds: Abunds):
        # no negative values
        neg_loss = torch.relu(-abunds).mean()
        # sum to one
        sum_loss = ((abunds.sum(dim=1) - 1).sum() ** 2).mean()
        return neg_loss, sum_loss

    def _stage1_loss(
        self,
        hyper_in: Image,
        abunds_pred: Abunds,
        hyper_recon: Image,
        abunds_fcls: Abunds | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Stage 1 loss: Reconstruction and abundance constraints

        Based on SSAF-Net's first stage training (epoch < epochs // 2):
        loss = loss_rec + lambda_kl * loss_kl + lambda_pre * loss_a + 0.1 * loss_a1_a2

        Parameters
        ----------
        hyper_in : Image
            Input hyperspectral image [b bands h w]
        abunds_pred : Abunds
            Predicted abundances [b n_endmember h w]
        hyper_recon : Image
            Reconstructed image [b bands h w]
        abunds_fcls : Abunds | None
            FCLS reference abundances (optional) [b n_endmember h w]

        Returns
        -------
        tuple[Tensor, dict[str, Tensor]]
            Total loss and loss dictionary
        """
        # Reconstruction loss
        rec_loss = self._reconstruction_loss(hyper_in, hyper_recon)
        total_loss = rec_loss

        # Abundance constraint loss
        abundance_loss = 0.0
        if self.lambda_pre > 0.0:
            abundance_loss = self._abundance_constraint_loss(abunds_pred, abunds_fcls)
            total_loss += self.lambda_pre * abundance_loss

        loss_dict = {
            "reconstruction": rec_loss,
            "abundance_constraint": abundance_loss,
            "abundance_physical": torch.tensor(0.0, device=rec_loss.device),
            "endmember_volume": torch.tensor(0.0, device=rec_loss.device),
            "endmember_spectral": torch.tensor(0.0, device=rec_loss.device),
        }

        return total_loss, loss_dict

    def _stage2_loss(
        self,
        hyper_in: Image,
        abunds_pred: Abunds,
        endmembers: EndMember,
        hyper_recon: Image,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Stage 2 loss: Add endmember property constraints

        Based on SSAF-Net's second stage training (epoch >= epochs // 2):
        loss = loss_rec + lambda_vol * loss_minvol + lambda_sad * loss_sad

        Parameters
        ----------
        hyper_in : Image
            Input hyperspectral image [b bands h w]
        abunds_pred : Abunds
            Predicted abundances [b n_endmember h w] (not used in stage 2)
        endmembers : EndMember
            Predicted endmembers [n_endmember bands]
        hyper_recon : Image
            Reconstructed image [b bands h w]

        Returns
        -------
        tuple[Tensor, dict[str, Tensor]]
            Total loss and loss dictionary
        """
        # abunds_pred is kept for interface consistency but not used in stage 2

        # Reconstruction loss
        rec_loss = self._reconstruction_loss(hyper_in, hyper_recon)
        total_loss = rec_loss

        # Endmember volume loss
        volume_loss = 0.0
        if self.lambda_vol > 0.0:
            volume_loss = self._endmember_volume_loss(endmembers)
            total_loss += self.lambda_vol * volume_loss

        # Endmember spectral loss
        spectral_loss = 0.0
        if self.lambda_sad > 0.0:
            spectral_loss = self._endmember_spectral_loss(endmembers)
            total_loss += self.lambda_sad * spectral_loss

        # Abundance physical constraints
        abunds_phy_loss = 0.0
        if self.lambda_abunds_physical > 0.0:
            abu_neg_loss, abu_sum_loss = self._abunds_physical_loss(abunds_pred)
            abunds_phy_loss = abu_neg_loss + abu_sum_loss
            total_loss += self.lambda_abunds_physical * abunds_phy_loss

        loss_dict = {
            "reconstruction": rec_loss,
            "abundance_constraint": torch.tensor(0.0, device=rec_loss.device),
            "abundance_physical": abunds_phy_loss,
            "endmember_volume": volume_loss,
            "endmember_spectral": spectral_loss,
        }

        return total_loss, loss_dict

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
            total_loss, loss_dict = self._stage1_loss(
                hyper_in, abunds_pred, hyper_recon, abunds_fcls
            )
        # Stage 2: Add endmember property constraints
        else:
            total_loss, loss_dict = self._stage2_loss(
                hyper_in, abunds_pred, endmembers, hyper_recon
            )

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

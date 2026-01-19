import pytest
import torch

from src.stage1.utilities.losses.latent_reg import LatentSparsityLoss


def test_latent_sparsity_loss_mean_term_non_negative() -> None:
    torch.manual_seed(0)
    z = torch.randn(10, 4)

    loss_fn = LatentSparsityLoss(
        dim_z=4,
        lambda_l1=0.0,
        lambda_l2=0.0,
        lambda_v=0.0,
        lambda_c=0.0,
        lambda_m=1.0,
    )
    loss, _ = loss_fn(z)

    expected = z.mean(dim=0).pow(2).mean()
    torch.testing.assert_close(loss, expected)
    assert loss.item() >= 0.0


def test_latent_sparsity_loss_cov_term_matches_vicreg_scaling() -> None:
    torch.manual_seed(0)
    z = torch.randn(10, 4)

    loss_fn = LatentSparsityLoss(
        dim_z=4,
        lambda_l1=0.0,
        lambda_l2=0.0,
        lambda_v=0.0,
        lambda_c=1.0,
        lambda_m=0.0,
    )
    loss, logs = loss_fn(z)

    n, d = z.shape
    z_centered = z - z.mean(dim=0)
    cov = (z_centered.T @ z_centered) / (n - 1)
    expected = (cov[~torch.eye(d, dtype=torch.bool)] ** 2).sum() / d

    torch.testing.assert_close(loss, expected)
    assert logs["cov"] == pytest.approx(float(expected.detach().cpu().item()))

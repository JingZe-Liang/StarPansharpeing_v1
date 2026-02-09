import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.stage1.self_supervised.lejepa_aug import lejepa_loss as lejepa_loss_fn
from src.stage1.self_supervised.lejepa_objectives import LeJEPALoss, RectifiedLpJEPALoss


def test_lejepa_loss_wrapper_matches_function_with_fixed_seed() -> None:
    torch.manual_seed(0)
    global_emb = torch.randn(2, 4, 16)
    local_emb = torch.randn(3, 4, 16)

    torch.manual_seed(123)
    ref_loss, _ = lejepa_loss_fn(global_emb, local_emb, lam=0.02, infonce_weight=None)

    torch.manual_seed(123)
    loss_fn = LeJEPALoss(lam=0.02, infonce_weight=None)
    loss, metrics = loss_fn(global_emb, local_emb)

    assert loss.ndim == 0
    assert torch.allclose(loss, ref_loss)
    assert "inv_loss" in metrics
    assert "sigreg_loss" in metrics


def test_rectified_lpjepa_loss_runs_on_cpu() -> None:
    torch.manual_seed(0)
    z1 = torch.randn(8, 32)
    z2 = torch.randn(8, 32)

    loss_fn = RectifiedLpJEPALoss(
        invariance_loss_weight=25.0,
        rdm_reg_loss_weight=125.0,
        target_distribution="rectified_lp_distribution",
        projection_vectors_type="random",
        num_projections=64,
        mean_shift_value=0.0,
        lp_norm_parameter=1.0,
        sigma_mode="sigma_GN",
        sync_ddp_gather=False,
    )
    loss, metrics = loss_fn(z1, z2)

    assert loss.ndim == 0
    assert metrics.total_loss.ndim == 0
    assert metrics.invariance_loss.ndim == 0
    assert metrics.rdmreg_loss.ndim == 0

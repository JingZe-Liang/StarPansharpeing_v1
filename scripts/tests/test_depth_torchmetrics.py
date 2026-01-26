import torch

from src.stage2.depth_estimation.metrics import DepthEstimationMetrics


def test_depth_metrics_masked_values() -> None:
    m = DepthEstimationMetrics(eps=1e-6)

    pred = torch.tensor([[[[2.0, 4.0], [0.0, 10.0]]]])
    target = torch.tensor([[[[1.0, 4.0], [-999.0, 5.0]]]])
    valid = target > -500.0

    m.update(pred=pred, target=target, valid_mask=valid)
    out = m.compute()

    # Valid pixels: (2 vs 1), (4 vs 4), (10 vs 5)
    # MAE: (1 + 0 + 5) / 3 = 2
    assert torch.isclose(out["mae"], torch.tensor(2.0))
    # RMSE: sqrt((1^2 + 0^2 + 5^2) / 3) = sqrt(26/3)
    assert torch.isclose(out["rmse"], torch.sqrt(torch.tensor(26.0 / 3.0)))

    # Positive pixels for AbsRel/LogRMSE/δ: all 3 (pred and target > eps)
    # AbsRel: (|2-1|/1 + |4-4|/4 + |10-5|/5) / 3 = (1 + 0 + 1) / 3 = 2/3
    assert torch.isclose(out["absrel"], torch.tensor(2.0 / 3.0))
    # δ1: ratios are 2, 1, 2 -> only middle is <1.25 => 1/3
    assert torch.isclose(out["delta1"], torch.tensor(1.0 / 3.0))

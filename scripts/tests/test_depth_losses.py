import math

import torch

from src.stage2.depth_estimation.loss.basic import (
    berhu_loss,
    edge_aware_smoothness_loss,
    gradient_loss,
    huber_loss,
    l1_loss,
    l2_loss,
    laplace_nll_loss,
    scale_shift_invariant_loss,
    silog_loss,
)


def test_l1_l2_masked_losses() -> None:
    pred = torch.tensor([[[[2.0, 4.0], [0.0, 10.0]]]])
    target = torch.tensor([[[[1.0, 4.0], [-999.0, 5.0]]]])
    valid = target > -500.0

    l1 = l1_loss(pred, target, valid)
    l2 = l2_loss(pred, target, valid)

    assert torch.isclose(l1, torch.tensor(2.0))
    assert torch.isclose(l2, torch.tensor(26.0 / 3.0))


def test_huber_loss_delta() -> None:
    pred = torch.tensor([[[[0.5, 2.0]]]])
    target = torch.zeros_like(pred)

    loss = huber_loss(pred, target, delta=1.0)
    expected = torch.tensor((0.125 + 1.5) / 2.0)
    assert torch.isclose(loss, expected)


def test_berhu_loss_with_c() -> None:
    pred = torch.tensor([[[[1.0, 4.0]]]])
    target = torch.zeros_like(pred)

    loss = berhu_loss(pred, target, c=2.0)
    expected = torch.tensor((1.0 + 5.0) / 2.0)
    assert torch.isclose(loss, expected)


def test_silog_loss_constant_ratio() -> None:
    pred = torch.tensor([[[[2.0, 4.0]]]])
    target = torch.tensor([[[[1.0, 2.0]]]])

    loss = silog_loss(pred, target, lambda_weight=0.5)
    expected = 0.5 * (math.log(2.0) ** 2)
    assert torch.isclose(loss, torch.tensor(expected))


def test_scale_shift_invariant_loss_zero_when_aligned() -> None:
    pred = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    target = pred * 2.0 + 1.0

    loss = scale_shift_invariant_loss(pred, target, reduction="l1")
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_gradient_and_smoothness_zero_for_constant_pred() -> None:
    pred = torch.zeros((1, 1, 3, 3))
    target = torch.zeros_like(pred)
    image = torch.rand((1, 3, 3, 3))

    grad = gradient_loss(pred, target)
    smooth = edge_aware_smoothness_loss(pred, image)

    assert torch.isclose(grad, torch.tensor(0.0))
    assert torch.isclose(smooth, torch.tensor(0.0))


def test_laplace_nll_zero_when_exact() -> None:
    pred = torch.tensor([[[[1.0, 2.0]]]])
    target = pred.clone()
    scale = torch.ones_like(pred)

    loss = laplace_nll_loss(pred, target, scale)
    assert torch.isclose(loss, torch.tensor(0.0))

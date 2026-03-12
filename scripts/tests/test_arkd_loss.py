from pathlib import Path
import sys

import pytest
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.stage1.utilities.losses.distill.ARKD import ARKD, ARKDLoss


def test_arkd_zero_when_teacher_and_student_match() -> None:
    loss_mod = ARKDLoss(gather_distributed=False)
    summary = torch.tensor(
        [
            [0.0],
            [1.0],
            [10.0],
        ]
    )

    loss = loss_mod(summary, summary)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_arkd_penalizes_close_pair_becoming_farther() -> None:
    loss_mod = ARKDLoss(gather_distributed=False)
    teacher = torch.tensor(
        [
            [0.0],
            [1.0],
            [10.0],
        ]
    )
    student = torch.tensor(
        [
            [0.0],
            [5.0],
            [10.0],
        ]
    )

    loss = loss_mod(student, teacher)
    assert loss > 0


def test_arkd_does_not_penalize_close_pair_becoming_closer() -> None:
    loss_mod = ARKDLoss(gather_distributed=False)
    teacher = torch.tensor(
        [
            [0.0],
            [1.0],
            [10.0],
        ]
    )
    student = torch.tensor(
        [
            [0.0],
            [0.5],
            [10.0],
        ]
    )

    loss = loss_mod(student, teacher)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_arkd_penalizes_far_pair_becoming_too_close() -> None:
    loss_mod = ARKDLoss(gather_distributed=False)
    teacher = torch.tensor(
        [
            [0.0],
            [1.0],
            [10.0],
        ]
    )
    student = torch.tensor(
        [
            [0.0],
            [1.0],
            [2.0],
        ]
    )

    loss = loss_mod(student, teacher)
    assert loss > 0


def test_arkd_does_not_penalize_far_pair_becoming_farther() -> None:
    loss_mod = ARKDLoss(gather_distributed=False)
    teacher = torch.tensor(
        [
            [0.0],
            [1.0],
            [10.0],
        ]
    )
    student = torch.tensor(
        [
            [0.0],
            [1.0],
            [20.0],
        ]
    )

    loss = loss_mod(student, teacher)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_arkd_returns_zero_for_single_sample() -> None:
    loss_mod = ARKD(gather_distributed=False)
    teacher = torch.tensor([[1.0, 2.0]])
    student = torch.tensor([[3.0, 4.0]])

    loss = loss_mod(student, teacher)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_arkd_rejects_non_2d_input() -> None:
    loss_mod = ARKDLoss(gather_distributed=False)
    teacher = torch.zeros(2, 4, 1)
    student = torch.zeros(2, 4, 1)

    with pytest.raises(ValueError):
        loss_mod(student, teacher)


def test_arkd_rejects_shape_mismatch() -> None:
    loss_mod = ARKDLoss(gather_distributed=False)
    teacher = torch.zeros(2, 4)
    student = torch.zeros(3, 4)

    with pytest.raises(ValueError):
        loss_mod(student, teacher)


def test_arkd_is_stable_when_teacher_distances_are_zero() -> None:
    loss_mod = ARKDLoss(gather_distributed=False)
    teacher = torch.zeros(3, 4)
    student = torch.randn(3, 4)

    loss = loss_mod(student, teacher)
    assert torch.isfinite(loss)
    assert torch.isclose(loss, torch.tensor(0.0))

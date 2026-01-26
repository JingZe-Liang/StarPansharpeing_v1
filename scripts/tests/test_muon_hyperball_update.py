from __future__ import annotations

import torch

from src.utilities.optim.dion.dion.muon import muon_update_post_orthogonalize


def _clone_list(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    return [t.clone() for t in tensors]


def test_muon_hyperball_preserves_norm_when_no_wd() -> None:
    torch.manual_seed(0)

    x = [torch.randn(8, 5), torch.randn(3, 7)]
    u = [torch.randn_like(x[0]), torch.randn_like(x[1])]

    base_lr = torch.tensor(0.01)
    adjusted_lr = torch.tensor(0.02)
    weight_decay = torch.tensor(0.0)
    eps = 1e-8

    x_before = _clone_list(x)
    muon_update_post_orthogonalize(
        X=x,
        U=u,
        base_lr=base_lr,
        adjusted_lr=adjusted_lr,
        weight_decay=weight_decay,
        epsilon=eps,
        cautious_wd=False,
        hyperball=True,
    )

    for xb, xa in zip(x_before, x, strict=True):
        torch.testing.assert_close(xa.norm(), xb.norm(), rtol=1e-5, atol=1e-6)


def test_muon_hyperball_preserves_post_decay_norm_when_wd_nonzero() -> None:
    torch.manual_seed(0)

    x = [torch.randn(8, 5), torch.randn(3, 7)]
    u = [torch.randn_like(x[0]), torch.randn_like(x[1])]

    base_lr = torch.tensor(0.01)
    adjusted_lr = torch.tensor(0.02)
    weight_decay = torch.tensor(0.1)
    eps = 1e-8

    expected_decay = 1.0 - float(base_lr) * float(weight_decay)
    norms_after_decay = [t.norm() * expected_decay for t in _clone_list(x)]

    muon_update_post_orthogonalize(
        X=x,
        U=u,
        base_lr=base_lr,
        adjusted_lr=adjusted_lr,
        weight_decay=weight_decay,
        epsilon=eps,
        cautious_wd=False,
        hyperball=True,
    )

    for expected_norm, xa in zip(norms_after_decay, x, strict=True):
        torch.testing.assert_close(xa.norm(), expected_norm, rtol=1e-5, atol=1e-6)


def test_muon_sgd_matches_reference_update() -> None:
    torch.manual_seed(0)

    x = [torch.randn(8, 5), torch.randn(3, 7)]
    u = [torch.randn_like(x[0]), torch.randn_like(x[1])]

    base_lr = torch.tensor(0.01)
    adjusted_lr = torch.tensor(0.02)
    weight_decay = torch.tensor(0.1)
    eps = 1e-8

    x_before = _clone_list(x)
    muon_update_post_orthogonalize(
        X=x,
        U=u,
        base_lr=base_lr,
        adjusted_lr=adjusted_lr,
        weight_decay=weight_decay,
        epsilon=eps,
        cautious_wd=False,
        hyperball=False,
    )

    decay = 1.0 - float(base_lr) * float(weight_decay)
    expected = [xb * decay - float(adjusted_lr) * uu for xb, uu in zip(x_before, u, strict=True)]

    for xa, exp in zip(x, expected, strict=True):
        torch.testing.assert_close(xa, exp, rtol=1e-5, atol=1e-6)

import torch

from src.stage1.discretization.collections.finite_scalar_quantization import FSQ


def test_fsq_mode_defaults_to_ifsq() -> None:
    fsq = FSQ(levels=[3, 3])
    assert fsq.bound_alpha == 1.6


def test_fsq_mode_fsq_uses_tanh_equivalent() -> None:
    fsq = FSQ(levels=[3, 3], fsq_mode="fsq")
    assert fsq.bound_alpha == 2.0


def test_fsq_bound_alpha_overrides_mode() -> None:
    fsq = FSQ(levels=[3, 3], fsq_mode="fsq", bound_alpha=1.25)
    assert fsq.bound_alpha == 1.25


def test_fsq_quantize_runs() -> None:
    fsq = FSQ(levels=[3, 3], dim=2, num_codebooks=1, fsq_mode="ifsq")
    x = torch.randn(2, 4, 2)
    out, indices = fsq(x)
    assert out.shape == x.shape
    assert indices is not None

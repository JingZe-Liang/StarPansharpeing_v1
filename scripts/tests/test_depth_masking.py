import torch

from src.stage2.depth_estimation.utils.masking import apply_clamp_and_scale, make_valid_mask


def test_make_valid_mask_threshold() -> None:
    depth = torch.tensor([[-999.0, -0.5, 10.0]])
    mask = make_valid_mask(depth, invalid_threshold=-500.0)
    assert mask.tolist() == [[False, True, True]]


def test_apply_clamp_and_scale_keeps_invalid_values() -> None:
    depth = torch.tensor([[-999.0, -1.0, 10.0]])
    mask = make_valid_mask(depth, invalid_threshold=-500.0)

    out = apply_clamp_and_scale(depth, mask, clamp_range=(0.0, 5.0), scale=10.0)
    assert torch.isclose(out[0, 0], torch.tensor(-999.0))
    assert torch.isclose(out[0, 1], torch.tensor(0.0))  # clamped -1 -> 0, then scaled -> 0
    assert torch.isclose(out[0, 2], torch.tensor(0.5))  # clamped 10 -> 5, then scaled -> 0.5


def test_fill_invalid_replaces_invalid_values() -> None:
    depth = torch.tensor([[-999.0, 2.0, 3.0]])
    mask = make_valid_mask(depth, invalid_threshold=-500.0)

    from src.stage2.depth_estimation.utils.masking import fill_invalid

    out = fill_invalid(depth, mask, fill_value=0.0)
    assert torch.isclose(out[0, 0], torch.tensor(0.0))
    assert torch.isclose(out[0, 1], torch.tensor(2.0))
    assert torch.isclose(out[0, 2], torch.tensor(3.0))


def test_fill_invalid_does_not_modify_input_in_place() -> None:
    depth = torch.tensor([[-999.0, 2.0, 3.0]])
    mask = make_valid_mask(depth, invalid_threshold=-500.0)

    from src.stage2.depth_estimation.utils.masking import fill_invalid

    out = fill_invalid(depth, mask, fill_value=0.0)
    assert torch.isclose(depth[0, 0], torch.tensor(-999.0))
    assert torch.isclose(out[0, 0], torch.tensor(0.0))

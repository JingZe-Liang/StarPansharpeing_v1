import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from src.stage2.cloud_removal.model.image_condition_unet import ImageConditionUNet


def _build_tiny_image_condition_unet(*, attention_stages: tuple[bool, ...] | None = None) -> "ImageConditionUNet":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.stage2.cloud_removal.model.image_condition_unet import ImageConditionUNet

    unet = ImageConditionUNet(
        in_channels=3,
        cond_channels=5,
        out_channels=3,
        base_channels=16,
        channel_mults=(1, 2),
        num_res_blocks=1,
        attention_stages=attention_stages,
    )
    return unet


def test_image_condition_unet_forward():
    model = _build_tiny_image_condition_unet()
    model.eval()

    b = 2
    x = torch.randn(b, 3, 8, 8)
    t = torch.randint(0, 1000, (b,))
    cond = torch.randn(b, 5, 8, 8)

    out, zs, cls_out = model(x, t, conditions=cond)
    assert out.shape == (b, 3, 8, 8)
    assert zs is None
    assert cls_out is None


def test_image_condition_unet_uncond_ignores_condition():
    model = _build_tiny_image_condition_unet()
    model.eval()

    b = 2
    x = torch.randn(b, 3, 8, 8)
    t = torch.randint(0, 1000, (b,))
    cond = torch.randn(b, 5, 8, 8)

    out0, _, _ = model(x, t, conditions=None)
    out1, _, _ = model(x, t, conditions=cond, uncond=True)
    torch.testing.assert_close(out0, out1, rtol=0, atol=0)


def test_image_condition_unet_attention_stage_forward():
    model = _build_tiny_image_condition_unet(attention_stages=(True, False))
    model.eval()

    b = 2
    x = torch.randn(b, 3, 8, 8)
    t = torch.randint(0, 1000, (b,))
    cond = torch.randn(b, 5, 8, 8)

    out, zs, cls_out = model(x, t, conditions=cond)
    assert out.shape == (b, 3, 8, 8)
    assert zs is None
    assert cls_out is None

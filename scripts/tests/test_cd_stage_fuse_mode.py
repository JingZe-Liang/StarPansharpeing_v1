import pytest
import torch

from src.stage2.change_detection.models.dinov3_adapted import (
    MultiscaleMBConvSkipsStage,
    MultiscaleMBConvStageConfig,
)


@pytest.mark.parametrize("fuse_mode", ["sub", "abs_sub", "cat", "cat_sub", "cat_abs"])
def test_multiscale_mbconv_skips_stage_fuse_modes_forward(fuse_mode: str) -> None:
    cfg = MultiscaleMBConvStageConfig(
        channels=[8, 16, 32, 64],
        depth=1,
        block_type="conv",
        kernel_size=3,
        stride=1,
        fuse_mode=fuse_mode,
    )
    stage = MultiscaleMBConvSkipsStage(cfg)

    b = 2
    skips1 = [
        torch.randn(b, 8, 64, 64),
        torch.randn(b, 16, 32, 32),
        torch.randn(b, 32, 16, 16),
        torch.randn(b, 64, 8, 8),
    ]
    skips2 = [
        torch.randn(b, 8, 64, 64),
        torch.randn(b, 16, 32, 32),
        torch.randn(b, 32, 16, 16),
        torch.randn(b, 64, 8, 8),
    ]

    outs = stage(skips1, skips2)

    assert len(outs) == 4
    assert outs[0].shape == (b, 8, 64, 64)
    assert outs[1].shape == (b, 16, 32, 32)
    assert outs[2].shape == (b, 32, 16, 16)
    assert outs[3].shape == (b, 64, 8, 8)


def test_multiscale_mbconv_skips_stage_unknown_fuse_mode_raises() -> None:
    cfg = MultiscaleMBConvStageConfig(channels=[8, 16, 32, 64], fuse_mode="unknown")
    with pytest.raises(ValueError, match="Unknown fuse mode"):
        MultiscaleMBConvSkipsStage(cfg)

import torch

from src.stage1.cosmos.modules.swin import (
    SwinBackbone2D,
    SwinBackboneCfg,
    SwinBlock2D,
    SwinBlockCfg,
    SwinStage2D,
    SwinStageCfg,
)


def test_swin_block2d_keeps_shape_on_non_multiple_window() -> None:
    x = torch.randn(2, 64, 17, 19)
    cfg = SwinBlockCfg(embed_dim=64, num_heads=8, window_size=7)
    block = SwinBlock2D(cfg, shift=True)

    y = block(x)

    assert y.shape == x.shape


def test_swin_stage2d_shapes_with_and_without_downsample() -> None:
    x = torch.randn(2, 64, 17, 19)

    no_down_cfg = SwinStageCfg(embed_dim=64, num_heads=8, depth=2, window_size=7, downsample=False)
    stage_no_down = SwinStage2D(no_down_cfg)
    y_no_down = stage_no_down(x)
    assert isinstance(y_no_down, torch.Tensor)
    assert y_no_down.shape == x.shape

    down_cfg = SwinStageCfg(embed_dim=64, num_heads=8, depth=2, window_size=7, downsample=True, out_dim=128)
    stage_down = SwinStage2D(down_cfg)
    y_down = stage_down(x)
    assert isinstance(y_down, torch.Tensor)
    assert y_down.shape == (2, 128, 9, 10)


def test_swin_backbone2d_out_indices_and_backward() -> None:
    cfg = SwinBackboneCfg(
        in_chans=64,
        embed_dim=64,
        patch_size=1,
        depths=[2, 2],
        num_heads=[4, 8],
        window_size=7,
        out_indices=[0, 1],
        patch_norm=False,
    )
    model = SwinBackbone2D(cfg)

    x = torch.randn(2, 64, 17, 19, requires_grad=True)
    outputs = model(x, return_features=True)

    assert isinstance(outputs, list)
    assert len(outputs) == 2
    assert outputs[0].shape == (2, 64, 17, 19)
    assert outputs[1].shape == (2, 128, 9, 10)

    loss = sum(out.mean() for out in outputs)
    loss.backward()
    assert x.grad is not None

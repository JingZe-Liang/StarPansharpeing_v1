from __future__ import annotations

import torch

from src.stage1.cosmos.modules.swin import SwinBackbone2D, SwinBackboneCfg, SwinBlock2D, SwinBlockCfg
from src.stage1.cosmos.modules.swin_op.swin_transformer import SwinTransformer
from src.stage1.cosmos.modules.variants.mlp import SwiGLU


def test_swin_block2d_supports_swiglu_shape() -> None:
    x = torch.randn(2, 64, 17, 19)
    cfg = SwinBlockCfg(
        embed_dim=64,
        num_heads=8,
        window_size=7,
        mlp_cls=SwiGLU,
        mlp_kwargs={"is_fused": None, "use_conv": False},
    )
    block = SwinBlock2D(cfg, shift=True)
    y = block(x)
    assert y.shape == x.shape


def test_swin_backbone2d_supports_swiglu_backward() -> None:
    cfg = SwinBackboneCfg(
        in_chans=64,
        embed_dim=64,
        patch_size=1,
        depths=[1, 1],
        num_heads=[4, 8],
        window_size=7,
        out_indices=[0, 1],
        patch_norm=False,
        mlp_cls=SwiGLU,
        mlp_kwargs={"is_fused": None, "use_conv": False},
    )
    model = SwinBackbone2D(cfg)

    x = torch.randn(2, 64, 17, 19, requires_grad=True)
    outputs = model(x, return_features=True)

    assert isinstance(outputs, list)
    assert len(outputs) == 2
    loss = sum(out.mean() for out in outputs)
    loss.backward()
    assert x.grad is not None


def test_swin_transformer_supports_swiglu_forward_backward() -> None:
    model = SwinTransformer(
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=0,
        embed_dim=32,
        depths=[1, 1],
        num_heads=[2, 4],
        window_size=4,
        is_flash=False,
        attn_backend="py",
        window_backend="py",
        merge_backend="py",
        mlp_cls=SwiGLU,
        mlp_kwargs={"is_fused": None, "use_conv": False},
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        patch_norm=True,
        use_checkpoint=False,
    )

    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    y = model(x)
    assert y.shape == (2, model.num_features)

    y.mean().backward()
    assert x.grad is not None

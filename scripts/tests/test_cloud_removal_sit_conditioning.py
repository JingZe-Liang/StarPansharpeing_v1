import sys
from pathlib import Path

import torch


def _build_tiny_sit(*, cond_drop_prob: float = 0.0):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.stage2.cloud_removal.model.sit import SiT

    model = SiT(
        input_size=8,
        patch_size=2,
        in_channels=3,
        hidden_size=64,
        encoder_depth=2,
        depth=6,
        num_heads=4,
        mlp_ratio=2.0,
        z_dims=[64],
        projector_dim=128,
        cond_drop_prob=cond_drop_prob,
        qk_norm=False,
    )
    model.path_drop_prob = 0.0
    model.sprint_drop_ratio = 0.0
    return model


def test_sit_accepts_condition_tensor():
    model = _build_tiny_sit()
    model.eval()

    b = 2
    x = torch.randn(b, 3, 8, 8)
    t = torch.randint(0, 1000, (b,))
    cond = torch.randn(b, 5, 8, 8)

    out, zs, cls_out = model(x, t, conditions=cond)
    assert out.shape == (b, 3, 8, 8)
    assert isinstance(zs, list) and zs[0].shape[0] == b
    assert cls_out is None


def test_sit_accepts_condition_with_sar_channels():
    model = _build_tiny_sit()
    model.eval()

    b = 2
    x = torch.randn(b, 3, 8, 8)
    t = torch.randint(0, 1000, (b,))
    opt = torch.randn(b, 13, 8, 8)
    sar = torch.randn(b, 2, 8, 8)
    conditions = torch.cat([opt, sar], dim=1)

    out, zs, cls_out = model(x, t, conditions=conditions)
    assert out.shape == (b, 3, 8, 8)
    assert isinstance(zs, list) and zs[0].shape[0] == b
    assert cls_out is None


def test_sit_condition_drop_matches_no_condition():
    model = _build_tiny_sit(cond_drop_prob=1.0)
    model.train()

    b = 2
    x = torch.randn(b, 3, 8, 8)
    t = torch.randint(0, 1000, (b,))
    cond = torch.randn(b, 5, 8, 8)

    out0, _, _ = model(x, t, conditions=None)
    out1, _, _ = model(x, t, conditions=cond)
    torch.testing.assert_close(out0, out1, rtol=0, atol=0)

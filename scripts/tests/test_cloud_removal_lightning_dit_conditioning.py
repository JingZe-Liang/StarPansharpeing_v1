import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from src.stage2.cloud_removal.model.lightning_dit import LightningDiT


def _build_tiny_lightning_dit(*, cond_drop_prob: float = 0.0, qk_norm: bool = False) -> "LightningDiT":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.stage2.cloud_removal.model.lightning_dit import LightningDiT

    return LightningDiT(
        input_size=8,
        patch_size=2,
        in_channels=3,
        hidden_size=64,
        depth=6,
        num_heads=4,
        mlp_ratio=2.0,
        cond_drop_prob=cond_drop_prob,
        qk_norm=qk_norm,
        use_rope=False,
    )


def test_lightning_dit_accepts_condition_tensor():
    model = _build_tiny_lightning_dit(qk_norm=True)
    model.eval()

    assert not isinstance(model.blocks[0].attn.q_norm, torch.nn.Identity)
    assert not isinstance(model.blocks[0].attn.k_norm, torch.nn.Identity)

    b = 2
    x = torch.randn(b, 3, 8, 8)
    t = torch.randint(0, 1000, (b,))
    cond = torch.randn(b, 5, 8, 8)

    out, zs, cls_out = model(x, t, conditions=cond)
    assert out.shape == (b, 3, 8, 8)
    assert zs is None
    assert cls_out is None


def test_lightning_dit_condition_drop_matches_no_condition():
    model = _build_tiny_lightning_dit(cond_drop_prob=1.0)
    model.train()

    b = 2
    x = torch.randn(b, 3, 8, 8)
    t = torch.randint(0, 1000, (b,))
    cond = torch.randn(b, 5, 8, 8)

    out0, _, _ = model(x, t, conditions=None)
    out1, _, _ = model(x, t, conditions=cond)
    torch.testing.assert_close(out0, out1, rtol=0, atol=0)

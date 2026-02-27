from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.stage1.utilities.losses.distill.teachers.dino_adapter import DinoTeacherAdapter


class _DummyDinoEncoder(nn.Module):
    def __init__(self, *, embed_dim: int = 8, image_size: int = 32, patch_size: int = 16) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=1)

    def get_intermediate_layers(
        self, x: torch.Tensor, n: int | list[int], reshape: bool, norm: bool
    ) -> list[torch.Tensor]:
        _ = reshape, norm
        feat = self.proj(x)
        feat = torch.nn.functional.avg_pool2d(feat, kernel_size=self.patch_size, stride=self.patch_size)
        if isinstance(n, int):
            return [feat for _ in range(n)]
        return [feat + float(i) for i, _ in enumerate(n)]


def test_dino_adapter_single_and_multi_layer_output() -> None:
    encoder = _DummyDinoEncoder()
    adapter = DinoTeacherAdapter(
        repa_encoder=encoder,
        dino_type="torch",
        repa_model_name="dinov3_vitl16",
        c_dim_first=True,
        img_is_neg1_1=True,
        rgb_channels=[0, 1, 2],
        img_resize="dino",
        pca_fn=None,
        dino_pretrained_on="satellite",
    )

    img = torch.randn(2, 5, 30, 34)

    out_single = adapter.encode(
        img,
        get_interm_feats=False,
        use_linstretch=True,
        detach=True,
        repa_fixed_bs=None,
    )
    assert isinstance(out_single, list)
    assert len(out_single) == 1
    assert out_single[0].shape == (2, 8, 2, 2)

    out_multi = adapter.encode(
        img,
        get_interm_feats=True,
        use_linstretch=True,
        detach=True,
        repa_fixed_bs=None,
    )
    assert isinstance(out_multi, list)
    assert len(out_multi) == 4
    for feat in out_multi:
        assert feat.shape == (2, 8, 2, 2)

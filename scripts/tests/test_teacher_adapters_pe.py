from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.stage1.utilities.losses.distill.teachers.pe_adapter import PETeacherAdapter


class _DummyPEEncoder(nn.Module):
    def __init__(self, *, width: int = 10, image_size: int = 32, patch_size: int = 8, layers: int = 12) -> None:
        super().__init__()
        self.width = width
        self.image_size = image_size
        self.patch_size = patch_size
        self.layers = layers
        self.proj = nn.Conv2d(3, width, kernel_size=1)

    def forward_features(
        self,
        x: torch.Tensor,
        *,
        layer_idx: int | list[int] = -1,
        strip_cls_token: bool = True,
        norm: bool = True,
    ) -> torch.Tensor | list[torch.Tensor]:
        _ = strip_cls_token, norm
        b, _, h, w = x.shape
        feat2d = self.proj(x)
        feat2d = torch.nn.functional.avg_pool2d(feat2d, kernel_size=self.patch_size, stride=self.patch_size)
        tokens = feat2d.permute(0, 2, 3, 1).reshape(b, (h // self.patch_size) * (w // self.patch_size), self.width)
        if isinstance(layer_idx, list):
            return [tokens + float(i) for i, _ in enumerate(layer_idx)]
        return tokens


def test_pe_adapter_multilayer_and_token_to_2d() -> None:
    adapter = PETeacherAdapter(
        repa_encoder=_DummyPEEncoder(),
        repa_model_name="PE-Core-B16-224",
        img_is_neg1_1=True,
        rgb_channels=[0, 1, 2],
        img_resize="dino",
        pca_fn=None,
    )

    img = torch.randn(2, 5, 31, 29)

    single = adapter.encode(
        img,
        get_interm_feats=False,
        use_linstretch=True,
        detach=True,
        repa_fixed_bs=None,
    )
    assert len(single) == 1
    assert single[0].shape == (2, 10, 4, 4)

    multi = adapter.encode(
        img,
        get_interm_feats=True,
        use_linstretch=True,
        detach=True,
        repa_fixed_bs=None,
    )
    assert isinstance(multi, list)
    assert len(multi) == 4
    for feat in multi:
        assert feat.shape == (2, 10, 4, 4)

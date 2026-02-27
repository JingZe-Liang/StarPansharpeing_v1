from __future__ import annotations

from types import SimpleNamespace
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.stage1.utilities.losses.distill.teachers.siglip_adapter import SiglipTeacherAdapter


class _DummyImageProcessor:
    def __init__(self) -> None:
        self.size: dict[str, int] = {"height": 224, "width": 224}
        self.do_resize = True
        self.do_rescale = False


class _DummySiglipProcessor:
    def __init__(self) -> None:
        self.image_processor = _DummyImageProcessor()

    def __call__(
        self, *, images: torch.Tensor, return_tensors: str = "pt", **kwargs: object
    ) -> dict[str, torch.Tensor]:
        _ = kwargs, return_tensors
        b, _, h, w = images.shape
        n = h * w
        return {
            "pixel_values": images,
            "pixel_attention_mask": torch.ones(b, n, dtype=torch.long),
        }


class _DummySiglipEncoder(nn.Module):
    def __init__(self, *, c: int = 6) -> None:
        super().__init__()
        self.c = c
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self, pixel_values: torch.Tensor, output_hidden_states: bool = False, **kwargs: object
    ) -> SimpleNamespace:
        _ = kwargs
        b, _, h, w = pixel_values.shape
        n = h * w
        base = pixel_values.mean(dim=1).reshape(b, n, 1).repeat(1, 1, self.c) * self.scale
        if output_hidden_states:
            h1 = base
            h2 = base + 1
            h3 = base + 2
            return SimpleNamespace(last_hidden_state=h3, hidden_states=(h1, h2, h3))
        return SimpleNamespace(last_hidden_state=base, hidden_states=None)


def test_siglip_adapter_regular_and_naflex_paths() -> None:
    adapter = SiglipTeacherAdapter(
        repa_encoder=_DummySiglipEncoder(),
        processor=_DummySiglipProcessor(),
        repa_img_size=32,
        img_is_neg1_1=True,
        rgb_channels=[0, 1, 2],
        pca_fn=None,
    )

    img = torch.randn(2, 5, 4, 4)
    out = adapter.encode(
        img,
        get_interm_feats=True,
        use_linstretch=True,
        detach=True,
        repa_fixed_bs=None,
    )
    assert isinstance(out, list)
    assert len(out) == 3
    for feat in out:
        assert feat.shape == (2, 6, 4, 4)

    naflex_inputs = {
        "pixel_values": torch.randn(2, 3, 4, 4),
        "spatial_shapes": torch.tensor([[4, 4], [4, 4]], dtype=torch.long),
    }
    naflex_out = adapter.forward_features(naflex_inputs, get_interm_feats=False, detach=True)
    assert len(naflex_out) == 1
    assert naflex_out[0].shape == (2, 6, 4, 4)

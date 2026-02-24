from __future__ import annotations

import hydra
import torch
import torch.nn as nn

from src.stage2.change_detection.models.tokenizer_bit_sdpa_cd import (
    BITSDPAHead,
    RawInjectionNeck,
    SDPAAttention,
    TokenizerHybridBITCDModel,
    _create_default_cfg,
)


class _FakeEncoder(nn.Module):
    def __init__(self, channels_per_stage: list[int]) -> None:
        super().__init__()
        self.channels_per_stage = channels_per_stage

    def forward(self, x: torch.Tensor):
        b, _, h, w = x.shape
        skips = [
            torch.randn(b, self.channels_per_stage[0], h, w, device=x.device, dtype=x.dtype),
            torch.randn(b, self.channels_per_stage[1], h // 2, w // 2, device=x.device, dtype=x.dtype),
            torch.randn(b, self.channels_per_stage[2], h // 4, w // 4, device=x.device, dtype=x.dtype),
            torch.randn(b, self.channels_per_stage[3], h // 8, w // 8, device=x.device, dtype=x.dtype),
        ]
        return skips, None


def test_sdpa_attention_forward_shape() -> None:
    attn = SDPAAttention(in_dims=32, embed_dims=64, num_heads=8, drop_rate=0.0)
    x = torch.randn(2, 49, 32)
    ref = torch.randn(2, 16, 32)

    out = attn(x, ref)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_raw_injection_neck_forward() -> None:
    neck = RawInjectionNeck(backbone_channels=64, raw_in_channels=3, out_channels=64, raw_embed_channels=16)

    feat = torch.randn(2, 64, 32, 32)
    raw = torch.randn(2, 3, 128, 128)
    out = neck(feat, raw)

    assert out.shape == (2, 64, 32, 32)


def test_bit_head_forward_shape() -> None:
    head = BITSDPAHead(
        in_channels=64,
        num_classes=2,
        channels=32,
        embed_dims=64,
        enc_depth=1,
        dec_depth=2,
        num_heads=8,
        token_len=4,
        upsample_scale=4,
    )

    x1 = torch.randn(2, 64, 32, 32)
    x2 = torch.randn(2, 64, 32, 32)
    out = head(x1, x2)

    assert out.shape == (2, 2, 128, 128)


def test_model_forward_with_fake_encoder() -> None:
    cfg = _create_default_cfg()
    cfg.input_channels = 3
    cfg.num_classes = 2
    cfg.freeze_tokenizer = True
    cfg._debug = True

    cfg.tokenizer_feature.features_per_stage = [64, 64, 64, 64]
    cfg.neck.stage_index = 2
    cfg.neck.raw_embed_channels = 16
    cfg.neck.out_channels = 64

    cfg.bit_head.channels = 32
    cfg.bit_head.embed_dims = 64
    cfg.bit_head.enc_depth = 1
    cfg.bit_head.dec_depth = 1
    cfg.bit_head.num_heads = 8
    cfg.bit_head.token_len = 4
    cfg.bit_head.upsample_scale = 4

    model = TokenizerHybridBITCDModel(cfg=cfg, encoder=_FakeEncoder([64, 64, 64, 64]))

    img1 = torch.randn(2, 3, 128, 128)
    img2 = torch.randn(2, 3, 128, 128)
    out = model([img1, img2])

    assert out.shape == (2, 2, 128, 128)


def test_model_hydra_instantiate_levir_cfg() -> None:
    with hydra.initialize(version_base=None, config_path="../configs/change_detection"):
        cfg = hydra.compose(
            config_name="tokenizer_hybrid_bit_sdpa_levir_cd",
            overrides=[
                "segment_model.input_channels=3",
                "segment_model.num_classes=2",
                "segment_model._debug=true",
                "segment_model.tokenizer_pretrained_path=null",
                "segment_model.tokenizer_feature.features_per_stage=[64,64,64,64]",
                "segment_model.neck.out_channels=64",
                "segment_model.neck.raw_embed_channels=16",
                "segment_model.bit_head.channels=32",
                "segment_model.bit_head.embed_dims=64",
                "segment_model.bit_head.dec_depth=1",
            ],
        )

    model = hydra.utils.instantiate(cfg.segment_model, encoder=_FakeEncoder([64, 64, 64, 64]))
    assert isinstance(model, TokenizerHybridBITCDModel)

    img1 = torch.randn(1, 3, 64, 64)
    img2 = torch.randn(1, 3, 64, 64)
    out = model([img1, img2])
    assert out.shape == (1, 2, 64, 64)

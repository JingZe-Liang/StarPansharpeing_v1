from __future__ import annotations

import types

import torch

from src.stage1.cosmos.cosmos_tokenizer import ContinuousImageTokenizer
from src.stage1.cosmos.modules.layers2d import Decoder, Encoder, make_block_fn
from src.stage1.cosmos.modules.swin_op import SwinTransformerBlock


def _build_encoder(**kwargs: object) -> Encoder:
    defaults: dict[str, object] = dict(
        in_channels=3,
        channels=16,
        channels_mult=[1, 2],
        num_res_blocks=1,
        attn_resolutions=[],
        dropout=0.0,
        resolution=16,
        z_channels=8,
        spatial_compression=2,
        patch_size=1,
        block_name="res_block",
        attn_type="none",
        norm_groups=8,
        swin_window_backend="py",
        swin_attn_backend="py",
    )
    defaults.update(kwargs)
    return Encoder(**defaults)  # ty: ignore error[invalid-argument-type]


def _build_decoder(**kwargs: object) -> Decoder:
    defaults: dict[str, object] = dict(
        out_channels=3,
        channels=16,
        channels_mult=[1, 2],
        num_res_blocks=1,
        attn_resolutions=[],
        dropout=0.0,
        resolution=16,
        z_channels=8,
        spatial_compression=2,
        patch_size=1,
        block_name="res_block",
        attn_type="none",
        norm_groups=8,
        swin_window_backend="py",
        swin_attn_backend="py",
    )
    defaults.update(kwargs)
    return Decoder(**defaults)  # ty: ignore error[invalid-argument-type]


def test_encoder_default_no_swin_regression() -> None:
    encoder = _build_encoder()
    x = torch.randn(2, 3, 16, 16)
    out = encoder(x)
    assert out.shape == (2, 8, 8, 8)
    assert not isinstance(encoder.down[0].block[0], SwinTransformerBlock)  # ty: ignore error[not-subscriptable]


def test_encoder_levelwise_swin_replace() -> None:
    encoder = _build_encoder(swin_replace_levels=[0], swin_shift_size=2, swin_window_size=4)
    x = torch.randn(2, 3, 16, 16)
    out = encoder(x)
    assert out.shape == (2, 8, 8, 8)
    assert isinstance(encoder.down[0].block[0], SwinTransformerBlock)  # ty: ignore error[not-subscriptable]
    assert not isinstance(encoder.down[1].block[0], SwinTransformerBlock)  # ty: ignore error[not-subscriptable]


def test_decoder_levelwise_swin_replace() -> None:
    decoder = _build_decoder(swin_replace_levels=[1], swin_shift_size=2, swin_window_size=4)
    z = torch.randn(2, 8, 8, 8)
    out = decoder(z, out_channels=3)
    assert out.shape == (2, 3, 16, 16)
    assert isinstance(decoder.up[1].block[0], SwinTransformerBlock)  # ty: ignore error[not-subscriptable]


def test_mid_swin_replace_switch() -> None:
    decoder = _build_decoder(swin_replace_mid=True, swin_shift_size=2, swin_window_size=4)
    assert isinstance(decoder.mid.block_1, SwinTransformerBlock)
    assert isinstance(decoder.mid.block_2, SwinTransformerBlock)


def test_swin_no_extra_attn_overlap() -> None:
    encoder = _build_encoder(
        attn_resolutions=[16],
        swin_replace_levels=[0],
        swin_disable_extra_attn=True,
        swin_shift_size=2,
        swin_window_size=4,
    )
    assert len(encoder.down[0].attn) == 0  # ty: ignore error[invalid-argument-type]


def test_swin_block_accepts_nchw_and_nlc() -> None:
    block = SwinTransformerBlock(
        dim=16,
        out_dim=16,
        input_resolution=(8, 8),
        num_heads=4,
        window_size=4,
        shift_size=2,
        attn_backend="py",
        window_backend="py",
    )
    x_nchw = torch.randn(2, 16, 8, 8)
    y_nchw = block(x_nchw)
    assert y_nchw.shape == (2, 16, 8, 8)

    x_nlc = torch.randn(2, 64, 16)
    y_nlc = block(x_nlc)
    assert y_nlc.shape == (2, 64, 16)


def test_get_last_enc_layer_swin_safe() -> None:
    tokenizer = ContinuousImageTokenizer.__new__(ContinuousImageTokenizer)
    tokenizer._vf_on_z_or_module = "module"
    tokenizer.encoder = types.SimpleNamespace(
        encoder=types.SimpleNamespace(block_name="swin_block"),
        quant_conv=torch.nn.Conv2d(1, 1, kernel_size=1),
    )
    assert tokenizer.get_last_enc_layer() is None


def test_swin_backend_fallback_to_py_without_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    block_fn = make_block_fn(
        block_name="swin_block",
        swin_attn_backend="triton_v3",
        swin_window_backend="triton",
    )
    block = block_fn(16, 16, 0.0, 8)
    assert isinstance(block, SwinTransformerBlock)
    assert block.window_backend == "py"
    assert block.attn.attn_backend == "py"

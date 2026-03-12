from typing import Any

import pytest
import torch

from src.stage1.cosmos.modules.blocks import AdaptiveInputConvLayer, AdaptiveOutputConvLayer
from src.stage1.cosmos.modules.layers2d import Decoder, Encoder


def _assert_has_grad(module: torch.nn.Module) -> None:
    grads = [param.grad for param in module.parameters() if param.requires_grad]
    assert any(grad is not None and torch.isfinite(grad).all() for grad in grads)


@pytest.mark.parametrize(
    ("mode", "extra_kwargs", "expected_channels"),
    [
        ("slice", {}, 4),
        ("interp", {}, 4),
        ("interp_proj", {"k_hidden": 6}, 4),
        ("mix", {"router_hidden_dim": 0, "always_use_router": True}, 4),
        ("sitok", {"sitok_reduce": "none"}, 20),
        ("sitok", {"sitok_reduce": "mean"}, 4),
        ("sitok", {"sitok_reduce": "pointwise"}, 4),
        ("cross_attn", {"cross_attn_pool_size": 2, "cross_attn_embed_dim": 16}, 4),
    ],
)
def test_adaptive_input_conv_all_modes_forward_and_backward(
    mode: Any,
    extra_kwargs: dict[str, Any],
    expected_channels: int,
) -> None:
    layer = AdaptiveInputConvLayer(
        in_channels=8,
        out_channels=4,
        kernel_size=3,
        stride=1,
        padding=1,
        use_bias=False,
        mode=mode,
        **extra_kwargs,
    )
    x = torch.randn(2, 5, 8, 8, requires_grad=True)
    y = layer(x)
    assert y.shape == (2, expected_channels, 8, 8)
    y.mean().backward()
    _assert_has_grad(layer)


@pytest.mark.parametrize(
    ("mode", "extra_kwargs", "runtime_out_channels", "supports_expand"),
    [
        ("slice", {}, 5, False),
        ("interp", {}, 5, True),
        ("interp_proj", {"k_hidden": 6}, 5, True),
        ("mix", {"router_hidden_dim": 0}, 5, True),
        ("sitok_film", {"sitok_embed_dim": 8, "sitok_hidden_dim": 0}, 5, True),
        (
            "sitok_pointwise",
            {"sitok_embed_dim": 8, "sitok_hidden_dim": 0, "sitok_basis_dim": 4},
            5,
            True,
        ),
        ("cross_attn", {"cross_attn_pool_size": 2, "cross_attn_embed_dim": 16}, 5, True),
    ],
)
def test_adaptive_output_conv_all_modes_forward_and_backward(
    mode: Any,
    extra_kwargs: dict[str, Any],
    runtime_out_channels: int,
    supports_expand: bool,
) -> None:
    layer = AdaptiveOutputConvLayer(
        in_channels=4,
        out_channels=8,
        kernel_size=3,
        stride=1,
        padding=1,
        use_bias=True,
        mode=mode,
        **extra_kwargs,
    )
    x = torch.randn(2, 4, 8, 8, requires_grad=True)
    y = layer(x, out_channels=runtime_out_channels)
    assert y.shape == (2, runtime_out_channels, 8, 8)
    y.mean().backward()
    _assert_has_grad(layer)

    if supports_expand:
        y_expand = layer(x.detach(), out_channels=10)
        assert y_expand.shape == (2, 10, 8, 8)


def test_adaptive_input_conv_cross_attn_is_content_conditioned() -> None:
    layer = AdaptiveInputConvLayer(
        in_channels=8,
        out_channels=4,
        kernel_size=3,
        stride=1,
        padding=1,
        use_bias=False,
        mode="cross_attn",
        cross_attn_pool_size=2,
        cross_attn_embed_dim=16,
    )

    x0 = torch.zeros(2, 5, 8, 8)
    x1 = torch.randn(2, 5, 8, 8)
    y0 = layer(x0)
    y1 = layer(x1)
    assert y0.shape == (2, 4, 8, 8)
    assert y1.shape == (2, 4, 8, 8)
    assert not torch.allclose(y0, y1)


def test_adaptive_output_conv_cross_attn_is_content_conditioned() -> None:
    layer = AdaptiveOutputConvLayer(
        in_channels=4,
        out_channels=8,
        kernel_size=3,
        stride=1,
        padding=1,
        use_bias=False,
        mode="cross_attn",
        cross_attn_pool_size=2,
        cross_attn_embed_dim=16,
    )

    x0 = torch.zeros(2, 4, 8, 8)
    x1 = torch.randn(2, 4, 8, 8)
    y0 = layer(x0, out_channels=5)
    y1 = layer(x1, out_channels=5)
    assert y0.shape == (2, 5, 8, 8)
    assert y1.shape == (2, 5, 8, 8)
    assert not torch.allclose(y0, y1)


def _build_encoder(**kwargs: object) -> Encoder:
    defaults: dict[str, object] = dict(
        in_channels=5,
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
    )
    defaults.update(kwargs)
    return Encoder(**defaults)  # type: ignore[arg-type]


def _build_decoder(**kwargs: object) -> Decoder:
    defaults: dict[str, object] = dict(
        out_channels=5,
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
    )
    defaults.update(kwargs)
    return Decoder(**defaults)  # type: ignore[arg-type]


def test_encoder_decoder_cross_attn_override_smoke() -> None:
    encoder = _build_encoder(
        adaptive_mode="interp",
        adaptive_input_mode="cross_attn",
        adaptive_input_conv_kwargs={"cross_attn_pool_size": 2, "cross_attn_embed_dim": 16},
    )
    decoder = _build_decoder(
        adaptive_mode="interp",
        adaptive_output_mode="cross_attn",
        adaptive_output_conv_kwargs={"cross_attn_pool_size": 2, "cross_attn_embed_dim": 16},
    )

    x = torch.randn(2, 5, 16, 16)
    z = encoder(x)
    assert z.shape == (2, 8, 8, 8)

    y = decoder(z, out_channels=5)
    assert y.shape == (2, 5, 16, 16)


def test_encoder_decoder_interp_path_still_works_with_cross_attn_overrides_present() -> None:
    encoder = _build_encoder(
        adaptive_mode="interp",
        adaptive_input_mode=None,
        adaptive_output_mode="cross_attn",
        adaptive_output_conv_kwargs={"cross_attn_pool_size": 2, "cross_attn_embed_dim": 16},
    )
    x = torch.randn(2, 5, 16, 16)
    z = encoder(x)
    assert z.shape == (2, 8, 8, 8)

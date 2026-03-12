import torch
from torch import nn
from torch.export import export
import pytest

from src.stage1.cosmos.modules.efficience.qat import (
    _PreparedTokenizerEncoderForPT2EQAT,
    _TokenizerEncoderBodyForPT2EQAT,
    _TokenizerEncoderForPT2EQAT,
    _TokenizerEncoderInputAdapterForPT2EQAT,
    _prepare_tokenizer_qat_export_inputs,
)
from src.stage1.cosmos.modules.blocks import AdaptiveInputConvLayer


class _ToyEncoder(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sin()


class _ToyNestedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patcher = nn.Identity()
        self.conv_in = AdaptiveInputConvLayer(in_channels=512, out_channels=16, mode="interp")

    def forward_from_conv_in(self, h: torch.Tensor) -> torch.Tensor:
        return h.sin()


class _ToySpatialBody(nn.Module):
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.avg_pool2d(h, kernel_size=8, stride=8)


def test_prepare_tokenizer_qat_export_inputs_keeps_batch_dynamic():
    example_inputs = (torch.randn(1, 3, 8, 8),)

    adjusted_inputs, dynamic_shapes = _prepare_tokenizer_qat_export_inputs(example_inputs)

    assert adjusted_inputs[0].shape == (2, 3, 8, 8)

    exported = export(
        _TokenizerEncoderForPT2EQAT(_ToyEncoder()).eval(),
        adjusted_inputs,
        dynamic_shapes=dynamic_shapes,
    )
    graph_module = exported.module()

    out = graph_module(torch.randn(4, 3, 8, 8))

    assert out.shape == (4, 3, 8, 8)


def test_prepare_tokenizer_qat_export_inputs_keeps_spatial_multiple_dynamic():
    example_inputs = (torch.randn(1, 128, 1024, 1024),)

    adjusted_inputs, dynamic_shapes = _prepare_tokenizer_qat_export_inputs(example_inputs, spatial_multiple=8)

    exported = export(
        _ToySpatialBody().eval(),
        adjusted_inputs,
        dynamic_shapes=dynamic_shapes,
    )
    graph_module = exported.module()

    out = graph_module(torch.randn(2, 128, 128, 128))

    assert out.shape == (2, 128, 16, 16)


def test_prepared_tokenizer_encoder_keeps_nested_conv_in_float():
    encoder = _ToyNestedEncoder().eval()
    input_adapter = _TokenizerEncoderInputAdapterForPT2EQAT(encoder).eval()
    body_inputs = (input_adapter(torch.randn(1, 512, 8, 8)),)
    body_inputs, dynamic_shapes = _prepare_tokenizer_qat_export_inputs(body_inputs)

    exported = export(
        _TokenizerEncoderBodyForPT2EQAT(encoder).eval(),
        body_inputs,
        dynamic_shapes=dynamic_shapes,
    )
    prepared = _PreparedTokenizerEncoderForPT2EQAT(
        exported.module(),
        stage="prepare",
        ignore_layer_names=(),
        float_encoder_fallback=encoder,
        float_input_adapter=input_adapter,
    )

    out_512 = prepared(torch.randn(2, 512, 8, 8))
    out_256 = prepared(torch.randn(3, 256, 8, 8))
    out_128 = prepared(torch.randn(4, 128, 8, 8))

    assert out_512.shape == (2, 16, 8, 8)
    assert out_256.shape == (3, 16, 8, 8)
    assert out_128.shape == (4, 16, 8, 8)


def test_nested_input_adapter_handles_bf16_inputs():
    if not torch.cuda.is_available():
        return

    encoder = _ToyNestedEncoder().cuda().eval()
    input_adapter = _TokenizerEncoderInputAdapterForPT2EQAT(encoder).cuda().eval()
    x = torch.randn(2, 256, 8, 8, device="cuda", dtype=torch.bfloat16)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = input_adapter(x)

    assert out.dtype == torch.bfloat16
    assert out.shape == (2, 16, 8, 8)

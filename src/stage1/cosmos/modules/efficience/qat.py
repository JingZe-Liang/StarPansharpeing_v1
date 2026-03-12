import math
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger
from torch import nn

if TYPE_CHECKING:
    from src.stage1.cosmos.cosmos_tokenizer import ContinuousImageTokenizer


class _TokenizerEncoderForPT2EQAT(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class _TokenizerEncoderBodyForPT2EQAT(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        forward_from_conv_in: Any = getattr(self.encoder, "forward_from_conv_in", None)
        if not callable(forward_from_conv_in):
            raise ValueError("PT2E QAT encoder body export requires encoder.forward_from_conv_in.")
        return forward_from_conv_in(h)


class _TokenizerEncoderInputAdapterForPT2EQAT(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patcher: Any = getattr(self.encoder, "patcher", None)
        conv_in: Any = getattr(self.encoder, "conv_in", None)
        if not callable(patcher) or not callable(conv_in):
            raise ValueError("PT2E QAT encoder input adapter requires encoder.patcher and encoder.conv_in.")
        return conv_in(patcher(x))


class _PreparedTokenizerEncoderForPT2EQAT(nn.Module):
    def __init__(
        self,
        graph_module: nn.Module,
        *,
        stage: str,
        ignore_layer_names: tuple[str, ...],
        example_shape: tuple[torch.Size, ...] | None = None,
        float_encoder_fallback: nn.Module | None = None,
        float_input_adapter: nn.Module | None = None,
    ):
        super().__init__()
        self.graph_module = graph_module
        self._pt2e_qat_stage = stage
        self._pt2e_qat_ignore_layer_names = ignore_layer_names
        self._pt2e_qat_example_shape = example_shape
        self._warned_ret_interm_feats_fallback = False
        self.__dict__["_float_encoder_fallback"] = float_encoder_fallback
        self.__dict__["_float_input_adapter"] = float_input_adapter

    def forward(
        self,
        x: torch.Tensor,
        ret_interm_feats: bool | list[int] | tuple[int, ...] = False,
    ):
        if ret_interm_feats not in (False, None):
            float_encoder_fallback = self.__dict__.get("_float_encoder_fallback")
            if float_encoder_fallback is None:
                raise ValueError("PT2E QAT encoder does not support ret_interm_feats without a float fallback encoder.")
            if not self._warned_ret_interm_feats_fallback:
                logger.warning(
                    "PT2E QAT encoder falls back to the original float encoder when ret_interm_feats is requested."
                )
                self._warned_ret_interm_feats_fallback = True
            return float_encoder_fallback(x, ret_interm_feats=ret_interm_feats)
        float_input_adapter = self.__dict__.get("_float_input_adapter")
        if float_input_adapter is not None:
            x = float_input_adapter(x)
        return self.graph_module(x)


def _count_fake_quant_or_observer_modules(model: nn.Module) -> int:
    count = 0
    for _, module in model.named_modules():
        module_name = type(module).__name__
        if "FakeQuantize" in module_name or "Observer" in module_name:
            count += 1
    return count


def _infer_tokenizer_qat_example_inputs(model: "ContinuousImageTokenizer") -> tuple[torch.Tensor, ...]:
    resolution = model.model_cfg.resolution
    if not isinstance(resolution, int):
        raise ValueError(f"PT2E QAT only supports int resolution, got {type(resolution).__name__}: {resolution}")

    in_channels = model.in_channels_after_patcher
    if not isinstance(in_channels, int):
        raise ValueError(f"PT2E QAT only supports int in_channels, got {type(in_channels).__name__}: {in_channels}")

    float_param = next((param for param in model.parameters() if param.is_floating_point()), None)
    device = float_param.device if float_param is not None else torch.device("cpu")
    dtype = float_param.dtype if float_param is not None else torch.float32
    return (torch.randn(1, in_channels, resolution, resolution, device=device, dtype=dtype),)


def _prepare_tokenizer_qat_export_inputs(
    example_inputs: tuple[torch.Tensor, ...],
    *,
    spatial_multiple: int | None = None,
) -> tuple[tuple[torch.Tensor, ...], tuple[dict[int, Any] | None, ...]]:
    from torch.export import Dim

    if len(example_inputs) == 0:
        raise ValueError("PT2E QAT export requires at least one example input.")

    export_inputs = list(example_inputs)
    first_input = export_inputs[0]
    if first_input.ndim == 0:
        raise ValueError("PT2E QAT export expects the first example input to have a batch dimension.")

    if first_input.shape[0] == 1:
        repeat_factors = [1] * first_input.ndim
        repeat_factors[0] = 2
        export_inputs[0] = first_input.repeat(*repeat_factors)

    first_dynamic_shape: dict[int, Any] = {0: Dim("batch", min=1)}
    if spatial_multiple is not None:
        if spatial_multiple <= 0:
            raise ValueError(f"spatial_multiple must be positive, got {spatial_multiple}.")
        if first_input.ndim < 4:
            raise ValueError("spatial_multiple requires the first example input to be at least 4D.")
        first_dynamic_shape[2] = spatial_multiple * Dim("height_divisible", min=2)
        first_dynamic_shape[3] = spatial_multiple * Dim("width_divisible", min=2)

    dynamic_shapes: list[dict[int, Any] | None] = [first_dynamic_shape]
    dynamic_shapes.extend(None for _ in export_inputs[1:])
    return tuple(export_inputs), tuple(dynamic_shapes)


def _prepare_tokenizer_qat_body_example_inputs(
    example_inputs: tuple[torch.Tensor, ...],
    input_adapter: nn.Module,
) -> tuple[torch.Tensor, ...]:
    if len(example_inputs) == 0:
        raise ValueError("PT2E QAT body export requires at least one example input.")

    with torch.no_grad():
        return (input_adapter(example_inputs[0]),)


def _infer_tokenizer_qat_body_spatial_multiple(encoder: nn.Module) -> int | None:
    num_downsamples = getattr(encoder, "num_downsamples", None)
    if not isinstance(num_downsamples, int):
        return None
    if num_downsamples < 0:
        raise ValueError(f"num_downsamples must be non-negative, got {num_downsamples}.")
    return int(math.pow(2, num_downsamples))


def _normalize_module_stack_path(path: str) -> str:
    prefix = "L['self']."
    return path[len(prefix) :] if path.startswith(prefix) else path


def _node_module_paths(node: Any) -> list[str]:
    nn_module_stack = node.meta.get("nn_module_stack", {})
    paths: list[str] = []
    for stack_entry in nn_module_stack.values():
        if isinstance(stack_entry, tuple) and len(stack_entry) >= 1:
            paths.append(_normalize_module_stack_path(stack_entry[0]))
    return paths


def _build_qat_ignore_filter(ignore_layer_names: list[str]):
    def _filter(node: Any) -> bool:
        if len(ignore_layer_names) == 0:
            return True
        module_paths = _node_module_paths(node)
        return not any(ignore_name in module_path for ignore_name in ignore_layer_names for module_path in module_paths)

    return _filter


def _normalize_pt2e_qat_mode(quantize_type: str, model: nn.Module) -> str:
    normalized = quantize_type.strip().lower()
    prepare_aliases = {"pt2e_qat", "pt2e_qat_prepare", "qat_prepare", "prepare", "int8", "pt2e_int8"}
    convert_aliases = {"pt2e_qat_convert", "qat_convert", "convert", "int8_convert", "pt2e_int8_convert"}

    if normalized in convert_aliases:
        return "convert"
    if normalized in prepare_aliases:
        if normalized in {"int8", "pt2e_int8"} and (
            getattr(model, "_pt2e_qat_stage", None) == "prepare" or _count_fake_quant_or_observer_modules(model) > 0
        ):
            return "convert"
        return "prepare"
    raise ValueError(
        f"Unsupported PT2E QAT quantize_type={quantize_type!r}. "
        f"Use one of prepare={sorted(prepare_aliases)} or convert={sorted(convert_aliases)}."
    )


def _extract_tokenizer_encoder_graph_module(model: nn.Module) -> nn.Module:
    if isinstance(model, _PreparedTokenizerEncoderForPT2EQAT):
        return model.graph_module
    return model


def apply_tokenizer_pt2e_qat(
    model: "ContinuousImageTokenizer | nn.Module",
    quantize_type: str,
    ignore_layer_names: list[str],
    example_inputs: tuple[torch.Tensor, ...] | None = None,
):
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )
    from torch.export import export
    from torchao.quantization.pt2e import allow_exported_model_train_eval, move_exported_model_to_eval
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_qat_pt2e

    from src.stage1.cosmos.cosmos_tokenizer import ContinuousImageTokenizer

    qat_mode = _normalize_pt2e_qat_mode(quantize_type, model)
    logger.info(
        f"<cyan>PT2E QAT stage={qat_mode}, quantize_type={quantize_type}, ignore_layer_names={ignore_layer_names}</cyan>"
    )

    if qat_mode == "convert":
        if isinstance(model, ContinuousImageTokenizer):
            prepared_encoder = model.encoder.encoder
            prepared_graph = _extract_tokenizer_encoder_graph_module(prepared_encoder)
            move_exported_model_to_eval(prepared_graph)  # type: ignore[arg-type]
            int8_graph = convert_pt2e(prepared_graph)  # type: ignore[arg-type]
            model.encoder.encoder = _PreparedTokenizerEncoderForPT2EQAT(
                int8_graph,
                stage="convert",
                ignore_layer_names=tuple(ignore_layer_names),
                example_shape=getattr(prepared_encoder, "_pt2e_qat_example_shape", None),
                float_encoder_fallback=getattr(prepared_encoder, "_float_encoder_fallback", None),
                float_input_adapter=getattr(prepared_encoder, "_float_input_adapter", None),
            )
            setattr(model, "_pt2e_qat_stage", "convert")
            setattr(model, "_pt2e_qat_ignore_layer_names", tuple(ignore_layer_names))
            logger.info(
                "<cyan>Converted tokenizer.encoder.encoder from prepared PT2E QAT graph to int8 inference graph.</cyan>"
            )
            return model

        prepared_graph = _extract_tokenizer_encoder_graph_module(model)
        move_exported_model_to_eval(prepared_graph)  # type: ignore[arg-type]
        int8_model = convert_pt2e(prepared_graph)  # type: ignore[arg-type]
        if isinstance(model, _PreparedTokenizerEncoderForPT2EQAT):
            return _PreparedTokenizerEncoderForPT2EQAT(
                int8_model,
                stage="convert",
                ignore_layer_names=tuple(ignore_layer_names),
                example_shape=getattr(model, "_pt2e_qat_example_shape", None),
                float_encoder_fallback=getattr(model, "_float_encoder_fallback", None),
                float_input_adapter=getattr(model, "_float_input_adapter", None),
            )
        setattr(int8_model, "_pt2e_qat_stage", "convert")
        setattr(int8_model, "_pt2e_qat_ignore_layer_names", tuple(ignore_layer_names))
        logger.info("<cyan>Converted prepared PT2E QAT model to int8 inference graph.</cyan>")
        return int8_model

    if not isinstance(model, ContinuousImageTokenizer):
        raise ValueError(
            "PT2E QAT prepare currently only supports ContinuousImageTokenizer. "
            "For convert stage, pass the prepared model returned by this function."
        )

    float_encoder = model.encoder.encoder
    input_adapter = _TokenizerEncoderInputAdapterForPT2EQAT(float_encoder).eval()
    raw_qat_inputs = example_inputs if example_inputs is not None else _infer_tokenizer_qat_example_inputs(model)
    qat_inputs = _prepare_tokenizer_qat_body_example_inputs(raw_qat_inputs, input_adapter)
    qat_inputs, dynamic_shapes = _prepare_tokenizer_qat_export_inputs(
        qat_inputs,
        spatial_multiple=_infer_tokenizer_qat_body_spatial_multiple(float_encoder),
    )
    wrapped_model = _TokenizerEncoderBodyForPT2EQAT(float_encoder).eval()
    exported = export(wrapped_model, qat_inputs, dynamic_shapes=dynamic_shapes)

    quantizer = XNNPACKQuantizer().set_module_name(
        "encoder",
        get_symmetric_quantization_config(is_per_channel=True, is_qat=True),
    )
    quantizer.set_filter_function(_build_qat_ignore_filter(ignore_layer_names))

    qat_graph = prepare_qat_pt2e(exported.module(), quantizer)
    allow_exported_model_train_eval(qat_graph)
    qat_graph.train()

    model.encoder.encoder = _PreparedTokenizerEncoderForPT2EQAT(
        qat_graph,
        stage="prepare",
        ignore_layer_names=tuple(ignore_layer_names),
        example_shape=tuple(t.shape for t in qat_inputs),
        float_encoder_fallback=float_encoder,
        float_input_adapter=input_adapter,
    )
    setattr(model, "_pt2e_qat_stage", "prepare")
    setattr(model, "_pt2e_qat_ignore_layer_names", tuple(ignore_layer_names))
    setattr(model, "_pt2e_qat_example_shape", tuple(t.shape for t in qat_inputs))

    logger.info(
        "<cyan>Prepared PT2E fake-quant training model for tokenizer.encoder.encoder and replaced the encoder in-place.</cyan>"
    )
    return model

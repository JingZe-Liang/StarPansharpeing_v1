from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from loguru import logger

from .transformer import Attention, GatedAttention

TE_PATCH_ENABLED = False
te: Any = None
try:
    import transformer_engine.pytorch as te

    TE_PATCH_ENABLED = True
except ImportError:
    TE_PATCH_ENABLED = False

_NEUTRAL_CONTAINERS = (nn.Sequential, nn.ModuleList, nn.ModuleDict)
_COLUMN_PARALLEL_KEYS = ("q_proj", "k_proj", "v_proj", "qkv", "w1", "w2", "w12", "fc1", "up_proj", "gate_proj")
_ROW_PARALLEL_KEYS = ("proj", "out_proj", "o_proj", "w3", "fc2", "down_proj", "head")


def _is_repo_module(module: nn.Module) -> bool:
    return type(module).__module__.startswith("src.")


def _should_recurse_into(module: nn.Module, is_root: bool = False) -> bool:
    return is_root or isinstance(module, _NEUTRAL_CONTAINERS) or _is_repo_module(module)


def _infer_parallel_mode_from_name(full_name: str) -> str | None:
    lowered_name = full_name.lower()
    if any(key in lowered_name for key in _COLUMN_PARALLEL_KEYS):
        return "column"
    if any(key in lowered_name for key in _ROW_PARALLEL_KEYS):
        return "row"
    return None


def _shard_tensor_for_tp(tensor: torch.Tensor, *, dim: int, rank: int, tp_size: int) -> torch.Tensor:
    chunks = tensor.chunk(tp_size, dim=dim)
    if len(chunks) != tp_size:
        raise ValueError(f"Failed to shard tensor of shape {tuple(tensor.shape)} across tp_size={tp_size}")
    return chunks[rank].contiguous()


def _copy_linear_weights(
    te_linear: Any,
    src_linear: nn.Linear,
    *,
    parallel_mode: str | None,
    tp_group=None,
    tp_size: int = 1,
) -> None:
    rank = 0
    should_shard = parallel_mode is not None and tp_group is not None and tp_size > 1
    if should_shard:
        rank = dist.get_rank(tp_group)

    with torch.no_grad():
        if parallel_mode == "column" and should_shard:
            te_linear.weight.copy_(_shard_tensor_for_tp(src_linear.weight, dim=0, rank=rank, tp_size=tp_size))
            if te_linear.bias is not None and src_linear.bias is not None:
                te_linear.bias.copy_(_shard_tensor_for_tp(src_linear.bias, dim=0, rank=rank, tp_size=tp_size))
        elif parallel_mode == "row" and should_shard:
            te_linear.weight.copy_(_shard_tensor_for_tp(src_linear.weight, dim=1, rank=rank, tp_size=tp_size))
            if te_linear.bias is not None and src_linear.bias is not None:
                te_linear.bias.copy_(src_linear.bias)
        else:
            te_linear.weight.copy_(src_linear.weight)
            if te_linear.bias is not None and src_linear.bias is not None:
                te_linear.bias.copy_(src_linear.bias)


def _patch_te_linears_in_module(
    module: nn.Module,
    *,
    prefix: str = "",
    tp_group=None,
    tp_size: int = 1,
    sequence_parallel: bool = False,
    is_root: bool = False,
) -> None:
    if not _should_recurse_into(module, is_root=is_root):
        return

    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Linear):
            parallel_mode = _infer_parallel_mode_from_name(full_name)
            te_linear = te.Linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                sequence_parallel=sequence_parallel,
                tp_group=tp_group,
                tp_size=tp_size,
                params_dtype=child.weight.dtype,
                parallel_mode=parallel_mode,
                device=child.weight.device,
                name=full_name,
            )
            _copy_linear_weights(
                te_linear,
                child,
                parallel_mode=parallel_mode,
                tp_group=tp_group,
                tp_size=tp_size,
            )
            te_linear.train(child.training)
            setattr(module, child_name, te_linear)
            continue

        _patch_te_linears_in_module(
            child,
            prefix=full_name,
            tp_group=tp_group,
            tp_size=tp_size,
            sequence_parallel=sequence_parallel,
            is_root=False,
        )


def _enable_te_attention_in_module(
    module: nn.Module,
    *,
    prefix: str = "",
    te_qkv_format: str = "bshd",
    tp_group=None,
    tp_size: int = 1,
    sequence_parallel: bool = False,
    cp_group=None,
    cp_global_ranks: list[int] | None = None,
    cp_stream=None,
    cp_comm_type: str = "p2p",
    is_root: bool = False,
) -> None:
    if not _should_recurse_into(module, is_root=is_root):
        return

    if isinstance(module, (Attention, GatedAttention)):
        module.enable_te_dpa(True, te_qkv_format=te_qkv_format)
        module.set_te_parallel_groups(
            tp_group=tp_group,
            tp_size=tp_size,
            sequence_parallel=sequence_parallel,
            cp_group=cp_group,
            cp_global_ranks=cp_global_ranks,
            cp_stream=cp_stream,
            cp_comm_type=cp_comm_type,
        )

    for child_name, child in module.named_children():
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        _enable_te_attention_in_module(
            child,
            prefix=full_name,
            te_qkv_format=te_qkv_format,
            tp_group=tp_group,
            tp_size=tp_size,
            sequence_parallel=sequence_parallel,
            cp_group=cp_group,
            cp_global_ranks=cp_global_ranks,
            cp_stream=cp_stream,
            cp_comm_type=cp_comm_type,
            is_root=False,
        )


def enable_te_attention(
    model: nn.Module,
    *,
    te_qkv_format: str = "bshd",
    tp_group=None,
    tp_size: int = 1,
    sequence_parallel: bool | None = None,
    cp_group=None,
    cp_global_ranks: list[int] | None = None,
    cp_stream=None,
    cp_comm_type: str = "p2p",
) -> nn.Module:
    if not TE_PATCH_ENABLED:
        logger.warning("TransformerEngine is not available, skipping TE attention patch.")
        return model
    resolved_sequence_parallel = (
        sequence_parallel if sequence_parallel is not None else (tp_group is not None and tp_size > 1)
    )
    _enable_te_attention_in_module(
        model,
        te_qkv_format=te_qkv_format,
        tp_group=tp_group,
        tp_size=tp_size,
        sequence_parallel=resolved_sequence_parallel,
        cp_group=cp_group,
        cp_global_ranks=cp_global_ranks,
        cp_stream=cp_stream,
        cp_comm_type=cp_comm_type,
        is_root=True,
    )
    return model


def patch_te_linears(
    model: nn.Module,
    *,
    tp_group=None,
    tp_size: int = 1,
    sequence_parallel: bool | None = None,
) -> nn.Module:
    if not TE_PATCH_ENABLED:
        logger.warning("TransformerEngine is not available, skipping TE linear patch.")
        return model
    resolved_sequence_parallel = (
        sequence_parallel if sequence_parallel is not None else (tp_group is not None and tp_size > 1)
    )
    _patch_te_linears_in_module(
        model,
        tp_group=tp_group,
        tp_size=tp_size,
        sequence_parallel=resolved_sequence_parallel,
        is_root=True,
    )
    return model


def apply_te_patches(
    model: nn.Module,
    *,
    te_qkv_format: str = "bshd",
    tp_group=None,
    tp_size: int = 1,
    sequence_parallel: bool | None = None,
    cp_group=None,
    cp_global_ranks: list[int] | None = None,
    cp_stream=None,
    cp_comm_type: str = "p2p",
) -> nn.Module:
    model = enable_te_attention(
        model,
        te_qkv_format=te_qkv_format,
        tp_group=tp_group,
        tp_size=tp_size,
        sequence_parallel=sequence_parallel,
        cp_group=cp_group,
        cp_global_ranks=cp_global_ranks,
        cp_stream=cp_stream,
        cp_comm_type=cp_comm_type,
    )
    model = patch_te_linears(
        model,
        tp_group=tp_group,
        tp_size=tp_size,
        sequence_parallel=sequence_parallel,
    )
    return model


__all__ = [
    "apply_te_patches",
    "enable_te_attention",
    "patch_te_linears",
]

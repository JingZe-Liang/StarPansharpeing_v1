import argparse
import csv
import inspect
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.data.litdata_hyperloader import create_hyper_image_litdata_flatten_paths_loader, get_dataset_len

umap_module: Any = None
try:
    import umap as _umap_module

    umap_module = _umap_module
except ImportError:  # pragma: no cover - runtime dependency
    pass


GENERIC_LITDATA_DIR_NAMES = {
    "litdata_hyper_images",
    "litdata_hyper_images_3bands",
    "litdata_hyper_images_4bands",
    "litdata_hyper_images_8bands",
    "litdata_images",
    "litdata_images_train",
    "litdata_images_val",
    "litdata_conditions",
    "litdata",
}

STATE_DICT_PREFIXES = (
    "_orig_mod.",
    "module.",
    "model.",
    "tokenizer.",
    "ema.",
    "ema_model.",
)


@dataclass(slots=True)
class DatasetSource:
    label: str
    input_paths: list[str]
    stream_kwargs: dict[str, Any]
    group_name: str


@dataclass(slots=True)
class EmbeddingRecord:
    dataset_label: str
    dataset_group: str
    dataset_path: str
    sample_key: str
    embedding: np.ndarray
    latent_shape: str


CACHE_FIELD_TO_ATTR = {
    "embedding": "embedding",
    "dataset_label": "dataset_label",
    "dataset_group": "dataset_group",
    "dataset_path": "dataset_path",
    "sample_key": "sample_key",
    "latent_shape": "latent_shape",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize tokenizer latent embeddings with UMAP.")
    parser.add_argument(
        "--tokenizer-config",
        type=Path,
        default="scripts/configs/tokenizer_gan/tokenizer/cosmos_hybrid_one_big_transformer.yaml",
        help="Tokenizer yaml config path.",
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default="scripts/configs/tokenizer_gan/dataset/litdata_one_loader.yaml",
        help="Dataset yaml config path.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default="runs/stage1_cosmos_hybrid/2025-12-21_23-52-12_hybrid_cosmos_f16c64_ijepa_pretrained_sem_no_lejepa/ema/tokenizer/model.safetensors",
        help="Optional tokenizer checkpoint path.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default="tmp/latent_umap_cache/", help="Directory for plots and cached embeddings."
    )
    parser.add_argument("--split", type=str, choices=("train", "val"), default="train", help="Dataset split.")
    parser.add_argument(
        "--group-by",
        type=str,
        choices=("path", "group"),
        default="path",
        help="Use each source path or each yaml group as one dataset label.",
    )
    parser.add_argument(
        "--samples-per-dataset",
        type=int,
        default=40,
        help="Upper bound of sampled items per dataset.",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=None,
        help="Optional ratio for each dataset. If set together with samples-per-dataset, the smaller target is used.",
    )
    parser.add_argument(
        "--sample-strategy",
        type=str,
        choices=("sequential", "random"),
        default="sequential",
        help="How to select samples from each dataset.",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for sampling.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override dataloader batch size.")
    parser.add_argument("--num-workers", type=int, default=2, help="Loader workers per dataset.")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device.")
    parser.add_argument(
        "--amp-dtype",
        type=str,
        choices=("fp32", "fp16", "bf16"),
        default="bf16",
        help="Autocast dtype on CUDA.",
    )
    parser.add_argument(
        "--latent-key",
        type=str,
        default="auto",
        help="Which encode output to use. Common values: auto, latent, to_dec, sem_z, low_lvl_z.",
    )
    parser.add_argument(
        "--pool",
        type=str,
        choices=("avg", "max", "flatten"),
        default="avg",
        help="How to convert latent tensor into one embedding vector per sample.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default="tmp/latent_umap_cache/",
        help="Directory for per-dataset npz caches.",
    )
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Skip tokenizer inference and reuse cache-path if it already exists.",
    )
    parser.add_argument(
        "--cache-fields",
        type=str,
        default="embedding,dataset_label,sample_key",
        help="Comma-separated fields to store in npz cache.",
    )
    parser.add_argument("--umap-neighbors", type=int, default=15, help="UMAP n_neighbors.")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP min_dist.")
    parser.add_argument("--umap-metric", type=str, default="euclidean", help="UMAP metric.")
    parser.add_argument("--point-size", type=float, default=10.0, help="Scatter point size.")
    parser.add_argument("--point-alpha", type=float, default=0.75, help="Scatter point alpha.")
    parser.add_argument("--title", type=str, default=None, help="Custom plot title.")
    return parser.parse_args()


def _normalize_loader_kwargs(
    loader_kwargs_cfg: DictConfig | dict[str, Any] | None, batch_size: int | None
) -> dict[str, Any]:
    loader_kwargs: dict[str, Any] = {}
    if loader_kwargs_cfg is not None:
        container = OmegaConf.to_container(loader_kwargs_cfg, resolve=True)
        if isinstance(container, dict):
            loader_kwargs = dict(container)
        else:
            raise TypeError("loader_kwargs must resolve to a dict.")

    if batch_size is not None:
        loader_kwargs["batch_size"] = batch_size

    loader_kwargs["shuffle"] = False
    loader_kwargs.setdefault("batch_size", 4)
    if int(loader_kwargs.get("num_workers", 0)) <= 0:
        loader_kwargs.pop("prefetch_factor", None)
    # This script creates many short-lived loaders and often breaks early after a few batches.
    # Persistent workers make shutdown noisier and slower in this usage pattern.
    loader_kwargs["persistent_workers"] = False
    return loader_kwargs


def _flatten_path_groups(paths_cfg: Any) -> dict[str, tuple[list[str], dict[str, Any]]]:
    if isinstance(paths_cfg, DictConfig):
        raw_paths = OmegaConf.to_container(paths_cfg, resolve=True)
    else:
        raw_paths = paths_cfg

    if not isinstance(raw_paths, dict):
        raise TypeError("Dataset paths config must resolve to a dict.")

    flattened: dict[str, tuple[list[str], dict[str, Any]]] = {}

    def _walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            if isinstance(value, (list, tuple)) and len(value) == 2 and isinstance(value[1], dict):
                ds_paths = value[0]
                if isinstance(ds_paths, str):
                    ds_paths_list = [ds_paths]
                elif isinstance(ds_paths, list) and all(isinstance(item, str) for item in ds_paths):
                    ds_paths_list = ds_paths
                else:
                    raise TypeError(f"Invalid dataset paths for group {key}: {type(ds_paths)}")
                flattened[str(key)] = (ds_paths_list, dict(value[1]))
                continue
            if isinstance(value, dict):
                _walk(value)

    _walk(raw_paths)
    return flattened


def _derive_dataset_label(path: str) -> str:
    path_obj = Path(path)
    parts = list(path_obj.parts)
    if parts and parts[0] in {"data", "data2"}:
        parts = parts[1:]
    if not parts:
        return path_obj.name

    last_part = parts[-1]
    if last_part.lower() in GENERIC_LITDATA_DIR_NAMES and len(parts) >= 2:
        return "/".join(parts[-2:])
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return parts[-1]


def _build_dataset_sources(dataset_cfg: DictConfig, split: str, group_by: str) -> list[DatasetSource]:
    loader_key = f"{split}_loader"
    loader_cfg = dataset_cfg.get(loader_key)
    if loader_cfg is None:
        raise KeyError(f"Missing {loader_key} in dataset config.")

    paths_cfg = loader_cfg.get("paths")
    if paths_cfg is None:
        raise KeyError(f"Missing {loader_key}.paths in dataset config.")

    path_groups = _flatten_path_groups(paths_cfg)
    sources: list[DatasetSource] = []
    for group_name, (input_paths, stream_kwargs) in path_groups.items():
        normalized_stream_kwargs = dict(stream_kwargs)
        normalized_stream_kwargs["transform_prob"] = 0.0
        normalized_stream_kwargs["shuffle"] = False
        normalized_stream_kwargs["is_cycled"] = False
        if group_by == "group":
            sources.append(
                DatasetSource(
                    label=group_name,
                    input_paths=input_paths,
                    stream_kwargs=normalized_stream_kwargs,
                    group_name=group_name,
                )
            )
            continue

        for input_path in input_paths:
            sources.append(
                DatasetSource(
                    label=_derive_dataset_label(input_path),
                    input_paths=[input_path],
                    stream_kwargs=normalized_stream_kwargs,
                    group_name=group_name,
                )
            )
    return sources


def _load_dict_config(config_path: Path, *, config_name: str) -> DictConfig:
    cfg = OmegaConf.load(config_path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"{config_name} must resolve to DictConfig, got {type(cfg)}")
    return cfg


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable, fallback to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def _resolve_amp_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    return torch.float32


def _parse_cache_fields(cache_fields: str) -> list[str]:
    fields = [field.strip() for field in cache_fields.split(",") if field.strip()]
    if len(fields) == 0:
        raise ValueError("cache_fields must contain at least one field.")
    if "embedding" not in fields:
        raise ValueError("cache_fields must include embedding.")
    invalid_fields = [field for field in fields if field not in CACHE_FIELD_TO_ATTR]
    if invalid_fields:
        raise ValueError(f"Unsupported cache fields: {invalid_fields}")
    return fields


def _sanitize_cache_stem(name: str) -> str:
    sanitized_chars = [char if char.isalnum() or char in ("-", "_", ".") else "_" for char in name]
    sanitized = "".join(sanitized_chars).strip("._")
    return sanitized or "dataset"


def _resolve_cache_dir(cache_path: Path) -> Path:
    if cache_path.suffix == ".npz":
        return cache_path.parent / cache_path.stem
    return cache_path


def _dataset_cache_path(cache_dir: Path, source: DatasetSource, source_index: int) -> Path:
    stem = _sanitize_cache_stem(source.label)
    return cache_dir / f"{source_index:03d}_{stem}.npz"


def _is_tensor_state_dict(candidate: Any) -> bool:
    return (
        isinstance(candidate, dict)
        and len(candidate) > 0
        and all(torch.is_tensor(value) for value in candidate.values())
    )


def _iter_state_dict_candidates(candidate: Any) -> list[dict[str, torch.Tensor]]:
    if _is_tensor_state_dict(candidate):
        return [candidate]
    if not isinstance(candidate, dict):
        return []

    results: list[dict[str, torch.Tensor]] = []
    for value in candidate.values():
        results.extend(_iter_state_dict_candidates(value))
    return results


def _strip_prefixes_once(state_dict: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    stripped_versions = [state_dict]
    current = state_dict
    while True:
        next_version = None
        for prefix in STATE_DICT_PREFIXES:
            if all(key.startswith(prefix) for key in current):
                next_version = {key[len(prefix) :]: value for key, value in current.items()}
                break
        if next_version is None:
            break
        stripped_versions.append(next_version)
        current = next_version
    return stripped_versions


def _choose_best_state_dict(model: nn.Module, checkpoint_obj: Any) -> dict[str, torch.Tensor]:
    candidates = _iter_state_dict_candidates(checkpoint_obj)
    if not candidates:
        raise ValueError("No tensor state_dict was found in checkpoint.")

    model_keys = set(model.state_dict().keys())
    best_state_dict = candidates[0]
    best_overlap = -1
    for candidate in candidates:
        for version in _strip_prefixes_once(candidate):
            overlap = len(model_keys.intersection(version.keys()))
            if overlap > best_overlap:
                best_overlap = overlap
                best_state_dict = version
    return best_state_dict


def _load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if checkpoint_path.is_dir():
        candidate_files = sorted(
            [
                *checkpoint_path.glob("*.safetensors"),
                *checkpoint_path.glob("*.bin"),
                *checkpoint_path.glob("*.pt"),
                *checkpoint_path.glob("*.pth"),
            ]
        )
        if len(candidate_files) == 0:
            raise ValueError(f"No checkpoint file found under directory: {checkpoint_path}")
        checkpoint_path = candidate_files[0]

    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        checkpoint_obj: Any = load_file(str(checkpoint_path), device=str(device))
    else:
        checkpoint_obj = torch.load(checkpoint_path, map_location=device)

    state_dict = _choose_best_state_dict(model, checkpoint_obj)
    model_state = model.state_dict()
    filtered_state_dict = {
        key: value for key, value in state_dict.items() if key in model_state and model_state[key].shape == value.shape
    }
    load_result = model.load_state_dict(filtered_state_dict, strict=False)
    logger.info(
        "Loaded checkpoint {} with {} missing keys and {} unexpected keys.",
        checkpoint_path,
        len(load_result.missing_keys),
        len(load_result.unexpected_keys),
    )


def _instantiate_tokenizer(
    tokenizer_config_path: Path, checkpoint_path: Path | None, device: torch.device
) -> nn.Module:
    tokenizer_cfg = OmegaConf.load(tokenizer_config_path)
    tokenizer_obj = hydra.utils.instantiate(tokenizer_cfg)
    if isinstance(tokenizer_obj, tuple):
        tokenizer = tokenizer_obj[-1]
    else:
        tokenizer = tokenizer_obj

    if not isinstance(tokenizer, nn.Module):
        raise TypeError(f"Tokenizer config must instantiate nn.Module, got {type(tokenizer)}")
    if not hasattr(tokenizer, "encode"):
        raise AttributeError(f"{tokenizer.__class__.__name__} does not implement encode().")

    tokenizer = tokenizer.to(device)
    tokenizer.eval()
    tokenizer.requires_grad_(False)

    if checkpoint_path is not None:
        _load_checkpoint(tokenizer, checkpoint_path, device)

    return tokenizer


def _resolve_target_sample_count(
    dataset_len: int,
    samples_per_dataset: int | None,
    sample_ratio: float | None,
) -> int:
    if dataset_len <= 0:
        return 0

    targets = [dataset_len]
    if samples_per_dataset is not None and samples_per_dataset > 0:
        targets.append(samples_per_dataset)
    if sample_ratio is not None:
        if sample_ratio <= 0:
            return 0
        targets.append(max(1, math.ceil(dataset_len * sample_ratio)))
    return max(0, min(targets))


def _choose_sample_indices(dataset_len: int, sample_count: int, seed: int) -> np.ndarray:
    if dataset_len <= 0 or sample_count <= 0:
        return np.zeros((0,), dtype=np.int64)
    if sample_count >= dataset_len:
        return np.arange(dataset_len, dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected = rng.choice(dataset_len, size=sample_count, replace=False)
    selected.sort()
    return selected.astype(np.int64, copy=False)


def _tensor_from_mapping(mapping: Any, key: str) -> Any:
    if isinstance(mapping, dict):
        return mapping.get(key)
    if hasattr(mapping, key):
        return getattr(mapping, key)
    return None


def _extract_latent_tensor(encode_out: Any, latent_key: str) -> torch.Tensor | list[torch.Tensor]:
    if torch.is_tensor(encode_out):
        return encode_out

    if latent_key != "auto":
        candidate = _tensor_from_mapping(encode_out, latent_key)
        if candidate is None:
            raise KeyError(f"latent_key={latent_key} not found in encode output.")
        if torch.is_tensor(candidate) or isinstance(candidate, list):
            return candidate
        raise TypeError(f"Unsupported latent type for key={latent_key}: {type(candidate)}")

    preferred_keys = ("latent", "to_dec", "sem_z", "low_lvl_z", "latent_before_quantizer")
    for key in preferred_keys:
        candidate = _tensor_from_mapping(encode_out, key)
        if torch.is_tensor(candidate) or isinstance(candidate, list):
            return candidate

    if isinstance(encode_out, tuple):
        for item in encode_out:
            if torch.is_tensor(item):
                return item
            if isinstance(item, list) and item and all(torch.is_tensor(sub_item) for sub_item in item):
                return item

    raise TypeError(f"Cannot extract latent tensor from encode output type {type(encode_out)}")


def _encode_with_optional_intermediate_features(
    tokenizer: nn.Module,
    batch_imgs: torch.Tensor,
    latent_key: str,
) -> Any:
    encode_fn = getattr(tokenizer, "encode")
    encode_kwargs: dict[str, Any] = {}
    if latent_key in {"sem_z", "low_lvl_z"}:
        try:
            signature = inspect.signature(encode_fn)
        except (TypeError, ValueError):
            signature = None
        if signature is not None and "get_intermediate_features" in signature.parameters:
            encode_kwargs["get_intermediate_features"] = True
    return encode_fn(batch_imgs, **encode_kwargs)


def _pool_latent_tensor(latent: torch.Tensor | list[torch.Tensor], pool: str) -> torch.Tensor:
    if isinstance(latent, list):
        if len(latent) == 0:
            raise ValueError("Latent tensor list is empty.")
        latent = latent[-1]

    if not torch.is_tensor(latent):
        raise TypeError(f"Latent must be tensor, got {type(latent)}")

    if pool == "flatten":
        return latent.reshape(latent.shape[0], -1)

    if latent.ndim == 4:
        if pool == "avg":
            return latent.mean(dim=(-1, -2))
        return latent.amax(dim=(-1, -2))

    if latent.ndim == 3:
        if pool == "avg":
            return latent.mean(dim=1)
        return latent.amax(dim=1)

    if latent.ndim == 2:
        return latent

    flattened = latent.reshape(latent.shape[0], -1)
    if pool == "avg":
        return flattened
    return flattened


def _build_loader_for_source(
    dataset_cfg: DictConfig,
    source: DatasetSource,
    split: str,
    batch_size: int | None,
    num_workers: int,
) -> tuple[Any, Any]:
    loader_key = f"{split}_loader"
    loader_cfg = dataset_cfg.get(loader_key)
    if loader_cfg is None:
        raise KeyError(f"Missing {loader_key} in dataset config.")

    loader_kwargs = _normalize_loader_kwargs(loader_cfg.get("loader_kwargs"), batch_size)
    loader_kwargs["num_workers"] = num_workers
    loader_kwargs["persistent_workers"] = False
    if num_workers <= 0:
        loader_kwargs.pop("prefetch_factor", None)

    macro_sampled_batch_size_cfg = loader_cfg.get("macro_sampled_batch_size")
    if macro_sampled_batch_size_cfg is not None:
        macro_sampled_batch_size_container = OmegaConf.to_container(macro_sampled_batch_size_cfg, resolve=True)
        if not isinstance(macro_sampled_batch_size_container, dict):
            raise TypeError("macro_sampled_batch_size must resolve to a dict.")
        macro_sampled_batch_size = dict(macro_sampled_batch_size_container)
    else:
        macro_sampled_batch_size = {
            128: loader_kwargs["batch_size"],
            256: loader_kwargs["batch_size"],
            512: loader_kwargs["batch_size"],
        }

    datasets, dataloader = create_hyper_image_litdata_flatten_paths_loader(
        paths={source.label: (source.input_paths, source.stream_kwargs)},
        weights=None,
        loader_kwargs=loader_kwargs,
        macro_sampled_batch_size=macro_sampled_batch_size,
        use_itered_cycle=True,
    )
    return datasets[0], dataloader


def _encode_selected_batch(
    tokenizer: nn.Module,
    batch_imgs: torch.Tensor,
    device: torch.device,
    amp_dtype: torch.dtype,
    latent_key: str,
    pool: str,
) -> tuple[np.ndarray, str]:
    batch_imgs = batch_imgs.to(device, non_blocking=True)
    autocast_context = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if device.type == "cuda" and amp_dtype != torch.float32
        else nullcontext()
    )
    with torch.inference_mode():
        with autocast_context:
            encode_out = _encode_with_optional_intermediate_features(
                tokenizer=tokenizer,
                batch_imgs=batch_imgs,
                latent_key=latent_key,
            )
            latent = _extract_latent_tensor(encode_out, latent_key=latent_key)
            pooled = _pool_latent_tensor(latent, pool=pool)
    latent_tensor = latent[-1] if isinstance(latent, list) else latent
    latent_shape = "x".join(str(dim) for dim in latent_tensor.shape[1:])
    return pooled.detach().cpu().float().numpy(), latent_shape


def _collect_embeddings_for_source(
    tokenizer: nn.Module,
    dataset_cfg: DictConfig,
    source: DatasetSource,
    source_index: int,
    args: argparse.Namespace,
    device: torch.device,
) -> list[EmbeddingRecord]:
    amp_dtype = _resolve_amp_dtype(args.amp_dtype)
    records: list[EmbeddingRecord] = []
    dataset, dataloader = _build_loader_for_source(
        dataset_cfg=dataset_cfg,
        source=source,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dataset_len = int(get_dataset_len(dataset))
    target_count = _resolve_target_sample_count(
        dataset_len=dataset_len,
        samples_per_dataset=args.samples_per_dataset,
        sample_ratio=args.sample_ratio,
    )
    if target_count <= 0:
        logger.warning("Skip dataset {} because no samples are selected.", source.label)
        return records

    logger.info(
        "Sampling {} / {} items from dataset {} ({}) with strategy={}",
        target_count,
        dataset_len,
        source.label,
        ",".join(source.input_paths),
        args.sample_strategy,
    )

    progress = tqdm(
        total=target_count,
        desc=source.label,
        unit="sample",
        leave=False,
    )
    dataloader_iter = iter(dataloader)
    try:
        for batch in dataloader_iter:
            imgs = batch.get("img")
            if not torch.is_tensor(imgs):
                raise TypeError("Batch must contain tensor img.")

            remaining = target_count - len(records)
            if remaining <= 0:
                break

            if args.sample_strategy == "random":
                if remaining >= int(imgs.shape[0]):
                    rel_positions = list(range(int(imgs.shape[0])))
                else:
                    batch_indices = _choose_sample_indices(
                        dataset_len=int(imgs.shape[0]),
                        sample_count=remaining,
                        seed=args.seed + source_index + len(records),
                    )
                    rel_positions = batch_indices.tolist()
            else:
                rel_positions = list(range(min(remaining, int(imgs.shape[0]))))

            if not rel_positions:
                continue

            selected_imgs = imgs[rel_positions]
            embedding_np, latent_shape = _encode_selected_batch(
                tokenizer=tokenizer,
                batch_imgs=selected_imgs,
                device=device,
                amp_dtype=amp_dtype,
                latent_key=args.latent_key,
                pool=args.pool,
            )

            batch_keys = batch.get("__key__")
            if isinstance(batch_keys, (list, tuple)):
                sample_keys = [str(batch_keys[pos]) for pos in rel_positions]
            else:
                sample_keys = [f"{source.label}_{len(records) + index}" for index in range(len(rel_positions))]

            dataset_path = ",".join(source.input_paths)
            for sample_key, embedding in zip(sample_keys, embedding_np, strict=True):
                records.append(
                    EmbeddingRecord(
                        dataset_label=source.label,
                        dataset_group=source.group_name,
                        dataset_path=dataset_path,
                        sample_key=sample_key,
                        embedding=np.asarray(embedding, dtype=np.float32),
                        latent_shape=latent_shape,
                    )
                )
            progress.update(len(rel_positions))

            if len(records) >= target_count:
                break
    finally:
        shutdown_workers = getattr(dataloader_iter, "_shutdown_workers", None)
        if callable(shutdown_workers):
            shutdown_workers()
        del dataloader_iter
        progress.close()

    if len(records) < target_count:
        raise RuntimeError(
            f"Dataset {source.label} ended early: collected {len(records)} / {target_count} selected samples."
        )

    return records


def _stack_embeddings(records: list[EmbeddingRecord]) -> np.ndarray:
    if len(records) == 0:
        raise ValueError("No embedding records were provided.")
    embedding_dims = {int(record.embedding.shape[0]) for record in records}
    if len(embedding_dims) != 1:
        raise ValueError(
            "Embedding dimensions are inconsistent across records. "
            "Use pool=avg or pool=max for mixed-resolution datasets."
        )
    return np.stack([record.embedding for record in records], axis=0)


def _save_cache(cache_path: Path, records: list[EmbeddingRecord]) -> None:
    _save_cache_with_fields(
        cache_path=cache_path,
        records=records,
        cache_fields=["embedding", "dataset_label", "sample_key"],
    )


def _save_cache_with_fields(cache_path: Path, records: list[EmbeddingRecord], cache_fields: list[str]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dict: dict[str, np.ndarray] = {}
    for field in cache_fields:
        attr_name = CACHE_FIELD_TO_ATTR[field]
        if field == "embedding":
            cache_dict[field] = _stack_embeddings(records)
            continue
        cache_dict[field] = np.asarray([getattr(record, attr_name) for record in records], dtype=object)
    np.savez_compressed(file=cache_path, **cache_dict)  # type: ignore[arg-type]


def _load_cache(cache_path: Path) -> list[EmbeddingRecord]:
    cache = np.load(cache_path, allow_pickle=True)
    embeddings = np.asarray(cache["embedding"], dtype=np.float32)
    dataset_labels = cache["dataset_label"].tolist() if "dataset_label" in cache else [""] * len(embeddings)
    dataset_groups = cache["dataset_group"].tolist() if "dataset_group" in cache else [""] * len(embeddings)
    dataset_paths = cache["dataset_path"].tolist() if "dataset_path" in cache else [""] * len(embeddings)
    sample_keys = cache["sample_key"].tolist() if "sample_key" in cache else [""] * len(embeddings)
    latent_shapes = cache["latent_shape"].tolist() if "latent_shape" in cache else [""] * len(embeddings)

    records: list[EmbeddingRecord] = []
    for index in range(len(embeddings)):
        records.append(
            EmbeddingRecord(
                dataset_label=str(dataset_labels[index]),
                dataset_group=str(dataset_groups[index]),
                dataset_path=str(dataset_paths[index]),
                sample_key=str(sample_keys[index]),
                embedding=np.asarray(embeddings[index], dtype=np.float32),
                latent_shape=str(latent_shapes[index]),
            )
        )
    return records


def _load_records_from_cache_dir(cache_dir: Path, sources: list[DatasetSource]) -> list[EmbeddingRecord]:
    records: list[EmbeddingRecord] = []
    for source_index, source in enumerate(sources):
        cache_file = _dataset_cache_path(cache_dir, source, source_index)
        if not cache_file.exists():
            logger.warning("Cache file is missing for dataset {}: {}", source.label, cache_file)
            continue
        records.extend(_load_cache(cache_file))
    return records


def _run_umap(records: list[EmbeddingRecord], args: argparse.Namespace) -> np.ndarray:
    if umap_module is None:
        raise ImportError("umap-learn is required. Please install it before running this script.")
    embeddings = _stack_embeddings(records)
    reducer = umap_module.UMAP(
        n_components=2,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=args.seed,
    )
    coords = reducer.fit_transform(embeddings)
    return np.asarray(coords, dtype=np.float32)


def _save_csv(output_path: Path, records: list[EmbeddingRecord], coords: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["dataset_label", "dataset_group", "dataset_path", "sample_key", "latent_shape", "umap_x", "umap_y"]
        )
        for record, coord in zip(records, coords, strict=True):
            writer.writerow(
                [
                    record.dataset_label,
                    record.dataset_group,
                    record.dataset_path,
                    record.sample_key,
                    record.latent_shape,
                    float(coord[0]),
                    float(coord[1]),
                ]
            )


def _save_plot(output_path: Path, records: list[EmbeddingRecord], coords: np.ndarray, args: argparse.Namespace) -> None:
    labels = [record.dataset_label for record in records]
    unique_labels = list(dict.fromkeys(labels))
    cmap = plt.get_cmap("tab20", max(1, len(unique_labels)))
    label_to_color = {label: cmap(index) for index, label in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(12, 9), dpi=200)
    for label in unique_labels:
        indices = [index for index, current_label in enumerate(labels) if current_label == label]
        subset = coords[indices]
        ax.scatter(
            subset[:, 0],
            subset[:, 1],
            s=args.point_size,
            alpha=args.point_alpha,
            c=[label_to_color[label]],
            label=label,
            edgecolors="none",
        )

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(args.title or f"Latent UMAP ({args.latent_key}, pool={args.pool}, split={args.split})")
    if len(unique_labels) <= 24:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = _resolve_cache_dir(args.cache_path or args.output_dir / "latent_embeddings_cache")
    cache_fields = _parse_cache_fields(args.cache_fields)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset_cfg = _load_dict_config(args.dataset_config, config_name="dataset_config")
    sources = _build_dataset_sources(dataset_cfg=dataset_cfg, split=args.split, group_by=args.group_by)
    logger.info("Prepared {} dataset sources for split={}.", len(sources), args.split)

    tokenizer: nn.Module | None = None
    for source_index, source in enumerate(sources):
        dataset_cache_path = _dataset_cache_path(cache_dir, source, source_index)
        if args.reuse_cache and dataset_cache_path.exists():
            logger.info("Reuse per-dataset cache for {}: {}", source.label, dataset_cache_path)
            continue

        if tokenizer is None:
            tokenizer = _instantiate_tokenizer(
                tokenizer_config_path=args.tokenizer_config,
                checkpoint_path=args.checkpoint,
                device=device,
            )

        dataset_records = _collect_embeddings_for_source(
            tokenizer=tokenizer,
            dataset_cfg=dataset_cfg,
            source=source,
            source_index=source_index,
            args=args,
            device=device,
        )
        if len(dataset_records) == 0:
            continue
        _save_cache_with_fields(dataset_cache_path, dataset_records, cache_fields=cache_fields)
        logger.info("Saved per-dataset cache for {} to {}", source.label, dataset_cache_path)

    records = _load_records_from_cache_dir(cache_dir, sources)
    if len(records) == 0:
        raise RuntimeError("No embeddings were collected or cached.")

    coords = _run_umap(records, args)
    coord_cache_path = args.output_dir / "latent_umap_points.csv"
    plot_path = args.output_dir / "latent_umap.png"
    _save_csv(coord_cache_path, records, coords)
    _save_plot(plot_path, records, coords, args)
    logger.info("Saved UMAP csv to {}", coord_cache_path)
    logger.info("Saved UMAP figure to {}", plot_path)


if __name__ == "__main__":
    main()

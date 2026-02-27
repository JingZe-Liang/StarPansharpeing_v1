from __future__ import annotations

from collections.abc import Callable
from typing import Any

import hydra
from loguru import logger
from mmengine.evaluator import BaseMetric, Evaluator
from mmengine.runner import Runner
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.stage2.object_detection.metrics.map_metric import DetectionMAPMetric
from src.stage2.object_detection.utils.large_image_inferencer import LargeImageTiledInferencer


def _as_bool_cfg(cfg: DictConfig | None, key: str, default: bool) -> bool:
    if cfg is None or key not in cfg:
        return default
    return bool(cfg[key])


def _as_int_cfg(cfg: DictConfig | None, key: str, default: int) -> int:
    if cfg is None or key not in cfg:
        return default
    return int(cfg[key])


def _as_float_cfg(cfg: DictConfig | None, key: str, default: float) -> float:
    if cfg is None or key not in cfg:
        return default
    return float(cfg[key])


def _as_str_cfg(cfg: DictConfig | None, key: str, default: str) -> str:
    if cfg is None or key not in cfg:
        return default
    return str(cfg[key])


def _build_val_evaluator(cfg: DictConfig) -> Evaluator | dict[Any, Any] | list[Any]:
    evaluator_cfg = cfg.get("val_evaluator")
    if evaluator_cfg is None:
        raise ValueError("val_evaluator is required for DIOR inference.")

    evaluator_obj = hydra.utils.instantiate(evaluator_cfg)
    if isinstance(evaluator_obj, Evaluator):
        return evaluator_obj
    if isinstance(evaluator_obj, BaseMetric):
        return Evaluator(metrics=[evaluator_obj])

    evaluator_dict = OmegaConf.to_container(evaluator_cfg, resolve=True)
    if isinstance(evaluator_dict, dict):
        return evaluator_dict
    if isinstance(evaluator_dict, list):
        return evaluator_dict
    raise TypeError("val_evaluator must resolve to Evaluator, BaseMetric, dict, or list.")


def _build_no_resize_transform(to_neg_1_1: bool) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def _transform(sample: dict[str, Any]) -> dict[str, Any]:
        image = sample.get("image")
        if image is None:
            return sample

        if hasattr(image, "detach"):
            image_t = image.float()
        else:
            image_t = torch.as_tensor(image, dtype=torch.float32)

        if image_t.ndim == 2:
            image_t = image_t.unsqueeze(0)
        if image_t.ndim == 3 and image_t.shape[0] not in (1, 3) and image_t.shape[-1] in (1, 3):
            image_t = image_t.permute(2, 0, 1)

        image_t = image_t / 255.0
        if to_neg_1_1:
            image_t = image_t * 2.0 - 1.0

        sample["image"] = image_t
        if "bbox_xyxy" in sample and sample["bbox_xyxy"] is not None:
            sample["bbox_xyxy"] = torch.as_tensor(sample["bbox_xyxy"], dtype=torch.float32)
        if "labels" in sample and sample["labels"] is not None:
            sample["labels"] = torch.as_tensor(sample["labels"], dtype=torch.long)
        return sample

    return _transform


def _resolve_autocast_dtype(dtype_name: str) -> torch.dtype:
    name = dtype_name.lower().strip()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    return torch.float32


def _to_uint8_hwc(image_chw: torch.Tensor, to_neg_1_1: bool) -> np.ndarray:
    image = image_chw.detach().cpu().float()
    if to_neg_1_1:
        image = (image + 1.0) * 0.5
    image = image.clamp(0.0, 1.0)
    image_hwc = image.permute(1, 2, 0).numpy()
    return np.clip(np.round(image_hwc * 255.0), 0, 255).astype(np.uint8)


def _build_tiled_predictor(
    model: Any,
    *,
    to_neg_1_1: bool,
    amp_enabled: bool,
    amp_dtype_name: str,
) -> Callable[[list[np.ndarray]], list[Any]]:
    try:
        from mmdet.structures import DetDataSample  # type: ignore[import-not-found]
    except Exception as exc:
        raise ImportError("Building tiled predictor requires mmdet.") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    amp_dtype = _resolve_autocast_dtype(amp_dtype_name)

    def _predict(images: list[np.ndarray]) -> list[Any]:
        if not images:
            return []
        batch_tensors: list[torch.Tensor] = []
        data_samples: list[Any] = []
        for image in images:
            if image.ndim != 3:
                raise ValueError(f"Expected image shape HWC, got {image.shape}")
            tensor = torch.from_numpy(image)
            if tensor.shape[-1] in (1, 3):
                tensor = tensor.permute(2, 0, 1)
            tensor = tensor.float() / 255.0
            if to_neg_1_1:
                tensor = tensor * 2.0 - 1.0
            batch_tensors.append(tensor)

            h = int(tensor.shape[-2])
            w = int(tensor.shape[-1])
            c = int(tensor.shape[-3])
            sample = DetDataSample()
            sample.set_metainfo(
                {
                    "img_shape": (h, w, c),
                    "ori_shape": (h, w, c),
                    "scale_factor": (1.0, 1.0),
                }
            )
            data_samples.append(sample)

        batch = torch.stack(batch_tensors, dim=0).to(device, non_blocking=True)
        with torch.inference_mode():
            if amp_enabled and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    outputs = model.forward(batch, data_samples=data_samples, mode="predict")
            else:
                outputs = model.forward(batch, data_samples=data_samples, mode="predict")
        if isinstance(outputs, list):
            return outputs
        return [outputs]

    return _predict


def _limit_dataloader_samples(val_loader: DataLoader[Any], max_samples: int) -> DataLoader[Any]:
    if max_samples <= 0:
        return val_loader

    dataset = val_loader.dataset
    dataset_len = len(dataset)  # type: ignore[arg-type]
    if dataset_len <= max_samples:
        return val_loader

    subset = Subset(dataset, list(range(max_samples)))
    prefetch_factor = getattr(val_loader, "prefetch_factor", None)
    loader_kwargs: dict[str, Any] = {
        "batch_size": val_loader.batch_size,
        "shuffle": False,
        "num_workers": val_loader.num_workers,
        "collate_fn": val_loader.collate_fn,
        "pin_memory": val_loader.pin_memory,
        "drop_last": False,
        "timeout": val_loader.timeout,
        "worker_init_fn": val_loader.worker_init_fn,
        "persistent_workers": val_loader.persistent_workers if val_loader.num_workers > 0 else False,
        "pin_memory_device": val_loader.pin_memory_device,
    }
    if val_loader.num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    logger.info(f"Limit val samples: {dataset_len} -> {max_samples}")
    return DataLoader(subset, **loader_kwargs)


def _instantiate_val_loader(
    cfg: DictConfig,
    *,
    keep_large_image: bool,
    force_bs1: bool,
    max_samples: int,
) -> Any:
    val_loader_cfg = OmegaConf.create(OmegaConf.to_container(cfg.dataloader.val, resolve=True))
    if not isinstance(val_loader_cfg, DictConfig):
        raise TypeError("dataloader.val must resolve to DictConfig.")

    if force_bs1:
        val_loader_cfg.batch_size = 1

    val_loader = hydra.utils.instantiate(val_loader_cfg)
    if keep_large_image and hasattr(val_loader, "dataset"):
        dataset = val_loader.dataset
        to_neg_1_1 = bool(getattr(getattr(val_loader_cfg, "dataset", None), "to_neg_1_1", False))
        dataset.transforms = _build_no_resize_transform(to_neg_1_1=to_neg_1_1)
    if isinstance(val_loader, DataLoader):
        val_loader = _limit_dataloader_samples(val_loader, max_samples=max_samples)
    return val_loader


def _eval_with_tiled_infer(
    cfg: DictConfig,
    model: Any,
    val_loader: Any,
) -> dict[str, float]:
    infer_cfg = cfg.get("infer")
    patch_size = _as_int_cfg(infer_cfg, "patch_size", 1024)
    overlap_ratio = _as_float_cfg(infer_cfg, "patch_overlap_ratio", 0.25)
    patch_batch_size = _as_int_cfg(infer_cfg, "patch_batch_size", 1)
    merge_iou_thr = _as_float_cfg(infer_cfg, "merge_iou_thr", 0.5)
    merge_nms_type = _as_str_cfg(infer_cfg, "merge_nms_type", "nms")
    max_samples = _as_int_cfg(infer_cfg, "max_samples", -1)

    amp_cfg = cfg.get("amp")
    amp_enabled = bool(amp_cfg.enabled) if amp_cfg is not None and "enabled" in amp_cfg else False
    amp_dtype_name = str(amp_cfg.dtype) if amp_cfg is not None and "dtype" in amp_cfg else "float32"
    to_neg_1_1 = bool(getattr(getattr(cfg.dataloader.val, "dataset", None), "to_neg_1_1", False))

    predictor = _build_tiled_predictor(
        model,
        to_neg_1_1=to_neg_1_1,
        amp_enabled=amp_enabled,
        amp_dtype_name=amp_dtype_name,
    )
    inferencer = LargeImageTiledInferencer(
        predictor=predictor,
        patch_size=patch_size,
        patch_overlap_ratio=overlap_ratio,
        batch_size=patch_batch_size,
        merge_iou_thr=merge_iou_thr,
        merge_nms_type=merge_nms_type,  # type: ignore[arg-type]
    )

    metric = DetectionMAPMetric()
    processed = 0
    for batch in val_loader:
        if isinstance(batch, list):
            if not batch:
                continue
            batch_item = batch[0]
        else:
            batch_item = batch
        if not isinstance(batch_item, dict):
            raise TypeError(f"Unexpected batch item type: {type(batch_item)}")

        inputs = batch_item.get("inputs")
        gt_sample = batch_item.get("data_samples")
        if inputs is None or gt_sample is None:
            continue

        if hasattr(inputs, "detach"):
            input_tensor = inputs
        else:
            input_tensor = torch.as_tensor(inputs, dtype=torch.float32)
        if input_tensor.ndim == 4:
            input_tensor = input_tensor[0]
        image = _to_uint8_hwc(input_tensor, to_neg_1_1=to_neg_1_1)
        pred_sample = inferencer.infer(image)
        metric.process(data_batch={"data_samples": [gt_sample]}, data_samples=[pred_sample])

        processed += 1
        if max_samples > 0 and processed >= max_samples:
            break
        if processed % 10 == 0:
            logger.info(f"Tiled infer progress: {processed} samples")

    if processed == 0:
        return {"map": 0.0, "map50": 0.0}
    return metric.compute_metrics(metric.results)


def _build_val_runner(
    cfg: DictConfig,
    model: Any,
    val_loader: Any,
    val_evaluator: Evaluator | dict[Any, Any] | list[Any],
) -> Runner:
    amp_cfg = cfg.get("amp")
    amp_enabled = bool(amp_cfg.enabled) if amp_cfg is not None and "enabled" in amp_cfg else False
    val_cfg = {"type": "ValLoop", "fp16": amp_enabled}
    default_hooks = {
        "timer": {"type": "IterTimerHook"},
        "logger": {"type": "LoggerHook", "interval": int(cfg.runner.log_interval)},
    }

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise TypeError("Runner cfg must resolve to dict.")

    checkpoint = str(cfg.runner.load_from) if "load_from" in cfg.runner and cfg.runner.load_from else None
    if checkpoint is None:
        raise ValueError("runner.load_from is required for inference.")

    return Runner(
        model=model,
        work_dir=str(cfg.runner.work_dir),
        val_dataloader=val_loader,
        val_cfg=val_cfg,
        val_evaluator=val_evaluator,
        default_hooks=default_hooks,
        custom_hooks=[],
        load_from=checkpoint,
        resume=False,
        launcher=str(cfg.runner.launcher),
        randomness={"seed": int(cfg.runner.seed)},
        default_scope="mmdet",
        cfg=cfg_dict,
    )


def _load_model_checkpoint(model: Any, cfg: DictConfig) -> None:
    checkpoint = str(cfg.runner.load_from) if "load_from" in cfg.runner and cfg.runner.load_from else None
    if checkpoint is None:
        raise ValueError("runner.load_from is required for inference.")
    from mmengine.runner.checkpoint import load_checkpoint

    load_checkpoint(model, checkpoint, map_location="cpu")
    logger.info(f"Loaded checkpoint: {checkpoint}")


def _pick_metric(metrics: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in metrics:
            value = metrics[key]
            return float(value.item()) if hasattr(value, "item") else float(value)
    return None


@hydra.main(
    config_path="../configs/object_detection",
    config_name="mmdet_dior_trainer",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    model = hydra.utils.instantiate(cfg.model)
    infer_cfg = cfg.get("infer")
    use_tiled = _as_bool_cfg(infer_cfg, "use_tiled", False)
    keep_large_image = _as_bool_cfg(infer_cfg, "keep_large_image", use_tiled)
    force_bs1 = _as_bool_cfg(infer_cfg, "force_bs1", use_tiled)
    max_samples = _as_int_cfg(infer_cfg, "max_samples", -1)

    val_loader = _instantiate_val_loader(
        cfg,
        keep_large_image=keep_large_image,
        force_bs1=force_bs1,
        max_samples=max_samples,
    )
    if use_tiled:
        logger.info("Use tiled inference mode.")
        _load_model_checkpoint(model, cfg)
        metrics = _eval_with_tiled_infer(cfg=cfg, model=model, val_loader=val_loader)
    else:
        logger.info("Use normal inference mode.")
        val_evaluator = _build_val_evaluator(cfg)
        runner = _build_val_runner(cfg=cfg, model=model, val_loader=val_loader, val_evaluator=val_evaluator)
        metrics = runner.val()
        if not isinstance(metrics, dict):
            raise TypeError("Runner.val() should return a dict of metrics.")

    map_value = _pick_metric(metrics, ("det/map", "map"))
    map50_value = _pick_metric(metrics, ("det/map50", "map50", "det/map_50", "map_50"))
    logger.info(f"Validation metrics: {metrics}")
    if map_value is not None:
        logger.info(f"det/map={map_value:.6f}")
    if map50_value is not None:
        logger.info(f"det/map50={map50_value:.6f}")


if __name__ == "__main__":
    """
    infer:

    - normally
    python scripts/infer/mmdet_dior_map_infer.py \
    runner.load_from=runs/object_detection/epoch_50.pth \
    runner.work_dir=outputs/object_detection_infer \
    +infer.max_samples=-1

    - with large image sliding window

    python scripts/infer/mmdet_dior_map_infer.py \
    runner.load_from=runs/object_detection/epoch_38.pth \
    runner.work_dir=outputs/object_detection_infer \
    +infer.use_tiled=true \
    +infer.keep_large_image=true \
    +infer.force_bs1=true \
    +infer.patch_size=1024 \
    +infer.patch_overlap_ratio=0.25 \
    +infer.patch_batch_size=1 \
    +infer.merge_iou_thr=0.5 \
    +infer.merge_nms_type=nms \
    +infer.max_samples=50
    """
    main()

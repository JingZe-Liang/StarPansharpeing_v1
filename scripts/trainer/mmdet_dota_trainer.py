from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from loguru import logger
from mmengine.dataset import pseudo_collate
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.structures import InstanceData
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from torchgeo.datasets import DOTA

from src.stage2.object_detection.data.DOTA import build_default_transforms, dota_poly_to_obb_le90
from src.stage2.object_detection.model.mmdetect.hybrid_fcos import build_hybrid_fcos_model
from src.stage2.object_detection.model.mmdetect.hybrid_fcos_obb import build_hybrid_fcos_obb_model
from src.stage2.object_detection.model.mmdetect.hybrid_rcnn import build_hybrid_rcnn_model
from src.stage2.object_detection.model.mmdetect.hybrid_rcnn_obb import build_hybrid_rcnn_obb_model

TrainSplit = Literal["train", "val"]


@dataclass
class TrainerPaths:
    cfg_path: Path | None


class DotaBatchHook(Hook):
    def __init__(self) -> None:
        self._logged = False

    def before_train_iter(self, runner: Runner, batch_idx: int, data_batch: dict | None = None) -> None:
        if self._logged or data_batch is None:
            return
        inputs = data_batch.get("inputs", [])
        data_samples = data_batch.get("data_samples", [])
        if inputs:
            runner.logger.info(f"[DOTA] First batch inputs: {inputs[0].shape}")
        if data_samples:
            num_boxes = getattr(data_samples[0].gt_instances, "bboxes", None)
            if num_boxes is not None and hasattr(num_boxes, "shape"):
                runner.logger.info(f"[DOTA] First batch boxes: {num_boxes.shape[0]}")
        self._logged = True


class DotaMMDetDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: TrainSplit,
        version: Literal["1.0", "1.5", "2.0"],
        bbox_orientation: Literal["horizontal", "oriented"],
        target_size: int,
        use_obb: bool,
    ) -> None:
        self.use_obb = use_obb
        self.bbox_orientation = bbox_orientation
        self.transforms = build_default_transforms(target_size=target_size)
        self.dataset = DOTA(
            root=root,
            split=split,
            version=version,
            bbox_orientation=bbox_orientation,
            transforms=self.transforms,
            download=False,
            checksum=False,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def _build_bboxes(self, sample: dict[str, Any]) -> torch.Tensor:
        if self.use_obb:
            polys = sample.get("bbox")
            if polys is None:
                return torch.zeros((0, 5), dtype=torch.float32)
            if hasattr(polys, "tolist"):
                polys = polys.tolist()
            obb = dota_poly_to_obb_le90(polys)
            if not obb:
                return torch.zeros((0, 5), dtype=torch.float32)
            return torch.as_tensor(obb, dtype=torch.float32)

        boxes = sample.get("bbox_xyxy")
        if boxes is None:
            return torch.zeros((0, 4), dtype=torch.float32)
        if hasattr(boxes, "detach"):
            return boxes.float()
        return torch.as_tensor(boxes, dtype=torch.float32)

    def _build_data_sample(self, image: torch.Tensor, labels: Any, bboxes: torch.Tensor) -> Any:
        from mmdet.structures import DetDataSample  # type: ignore[import-not-found]
        from mmdet.structures.bbox import HorizontalBoxes  # type: ignore[import-not-found]

        if self.use_obb:
            try:
                from mmrotate.structures.bbox import RotatedBoxes  # type: ignore[import-not-found]
            except Exception as exc:  # pragma: no cover - optional dependency
                raise ImportError("mmrotate is required for OBB training.") from exc
            box_type = RotatedBoxes
        else:
            box_type = HorizontalBoxes

        if hasattr(labels, "detach"):
            labels_t = labels.long()
        else:
            labels_t = torch.as_tensor(labels, dtype=torch.long)

        gt_instances = InstanceData()
        gt_instances.bboxes = box_type(bboxes)
        gt_instances.labels = labels_t

        data_sample = DetDataSample()
        height = int(image.shape[-2])
        width = int(image.shape[-1])
        channels = int(image.shape[-3]) if image.ndim >= 3 else 1
        data_sample.set_metainfo(
            {
                "img_shape": (height, width, channels),
                "ori_shape": (height, width, channels),
                "scale_factor": (1.0, 1.0),
            }
        )
        data_sample.gt_instances = gt_instances
        return data_sample

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.dataset[idx]
        image = sample["image"]
        if not hasattr(image, "detach"):
            image = torch.as_tensor(image, dtype=torch.float32)

        bboxes = self._build_bboxes(sample)
        labels = sample.get("labels", torch.zeros((0,), dtype=torch.long))
        data_sample = self._build_data_sample(image, labels, bboxes)
        return {"inputs": image, "data_samples": data_sample}


def _load_cfg(cfg_path: Path | None) -> DictConfig:
    if cfg_path is None:
        cfg = OmegaConf.create(
            {
                "model": {
                    "type": "fcos",
                    "cfg_path": "src/stage2/object_detection/model/mmdetect/hybrid_mmdet_fcos.yaml",
                    "overrides": {"num_classes": 18, "input_channels": 3, "tokenizer_feature": {"in_channels": 3}},
                },
                "dataset": {
                    "root": "data/Downstreams/DOTA",
                    "version": "2.0",
                    "bbox_orientation": "horizontal",
                    "target_size": 512,
                    "batch_size": 2,
                    "num_workers": 2,
                    "train_split": "train",
                    "val_split": "val",
                },
                "train": {
                    "work_dir": "output/mmdet_dota",
                    "max_epochs": 12,
                    "val_interval": 1,
                    "log_interval": 50,
                    "checkpoint_interval": 1,
                    "launcher": "none",
                    "seed": 2025,
                    "do_val": True,
                },
                "optimizer": {"type": "AdamW", "lr": 1e-4, "weight_decay": 0.05},
            }
        )
        return cfg
    loaded = OmegaConf.load(str(cfg_path))
    if not isinstance(loaded, DictConfig):
        raise TypeError("Trainer config must be a mapping (DictConfig).")
    return loaded


def _build_model(cfg: DictConfig) -> torch.nn.Module:
    model_cfg = OmegaConf.load(str(cfg.model.cfg_path))
    if not isinstance(model_cfg, DictConfig):
        raise TypeError("Model config must be a mapping (DictConfig).")
    overrides = cfg.model.get("overrides", None)
    if overrides:
        model_cfg = OmegaConf.merge(model_cfg, overrides)
    if not isinstance(model_cfg, DictConfig):
        raise TypeError("Model config must be a mapping (DictConfig).")

    model_type = str(cfg.model.type).lower()
    if model_type == "fcos":
        return build_hybrid_fcos_model(model_cfg)
    if model_type == "rcnn":
        return build_hybrid_rcnn_model(model_cfg)
    if model_type == "fcos_obb":
        return build_hybrid_fcos_obb_model(model_cfg)
    if model_type == "rcnn_obb":
        return build_hybrid_rcnn_obb_model(model_cfg)
    raise ValueError(f"Unknown model.type: {cfg.model.type}")


def _build_dataloader(cfg: DictConfig, split: TrainSplit) -> DataLoader:
    use_obb = cfg.model.type in ("fcos_obb", "rcnn_obb")
    dataset = DotaMMDetDataset(
        root=str(cfg.dataset.root),
        split=split,
        version=cfg.dataset.version,
        bbox_orientation=cfg.dataset.bbox_orientation,
        target_size=int(cfg.dataset.target_size),
        use_obb=bool(use_obb),
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg.dataset.batch_size),
        num_workers=int(cfg.dataset.num_workers),
        shuffle=split == "train",
        collate_fn=pseudo_collate,
    )


def _build_runner(
    cfg: DictConfig,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
) -> Runner:
    optim_wrapper = {
        "type": "OptimWrapper",
        "optimizer": dict(cfg.optimizer),
    }
    train_cfg = {
        "type": "EpochBasedTrainLoop",
        "max_epochs": int(cfg.train.max_epochs),
        "val_interval": int(cfg.train.val_interval),
    }
    val_cfg = {"type": "ValLoop"} if val_loader is not None else None

    default_hooks = {
        "timer": {"type": "IterTimerHook"},
        "logger": {"type": "LoggerHook", "interval": int(cfg.train.log_interval)},
        "param_scheduler": {"type": "ParamSchedulerHook"},
        "checkpoint": {
            "type": "CheckpointHook",
            "interval": int(cfg.train.checkpoint_interval),
            "max_keep_ckpts": 3,
        },
        "sampler_seed": {"type": "DistSamplerSeedHook"},
    }

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise TypeError("Runner cfg must be a dict.")

    runner = Runner(
        model=model,
        work_dir=str(cfg.train.work_dir),
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        train_cfg=train_cfg,
        val_cfg=val_cfg,
        optim_wrapper=optim_wrapper,
        default_hooks=default_hooks,
        custom_hooks=[DotaBatchHook()],
        launcher=str(cfg.train.launcher),
        randomness={"seed": int(cfg.train.seed)},
        cfg=cfg_dict,
    )
    return runner


def main() -> None:
    parser = argparse.ArgumentParser(description="MMDetection trainer for DOTA dataset.")
    parser.add_argument("--config", type=str, default=None, help="Path to trainer YAML config.")
    args = parser.parse_args()

    cfg = _load_cfg(Path(args.config) if args.config else None)
    model = _build_model(cfg)

    train_loader = _build_dataloader(cfg, split=cfg.dataset.train_split)
    val_loader = None
    if bool(cfg.train.do_val):
        val_loader = _build_dataloader(cfg, split=cfg.dataset.val_split)

    runner = _build_runner(cfg, model, train_loader, val_loader)
    logger.info("Start training with MMDetection runner.")
    runner.train()


if __name__ == "__main__":
    main()

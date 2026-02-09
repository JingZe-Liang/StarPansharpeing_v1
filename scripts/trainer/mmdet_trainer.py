from __future__ import annotations

from typing import Any

import hydra
from loguru import logger
from mmengine.hooks import Hook
from mmengine.optim import OptimWrapper
from mmengine.runner import Runner
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader


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


def _build_runner(
    cfg: DictConfig,
    model: Any,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
) -> Runner:
    optim_wrapper = _build_optim_wrapper(cfg, model)
    train_cfg = {
        "type": "EpochBasedTrainLoop",
        "max_epochs": int(cfg.runner.max_epochs),
        "val_interval": int(cfg.runner.val_interval),
    }
    val_cfg = {"type": "ValLoop"} if val_loader is not None else None

    default_hooks = {
        "timer": {"type": "IterTimerHook"},
        "logger": {"type": "LoggerHook", "interval": int(cfg.runner.log_interval)},
        "param_scheduler": {"type": "ParamSchedulerHook"},
        "checkpoint": {
            "type": "CheckpointHook",
            "interval": int(cfg.runner.checkpoint_interval),
            "max_keep_ckpts": 3,
        },
        "sampler_seed": {"type": "DistSamplerSeedHook"},
    }

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise TypeError("Runner cfg must be a dict.")

    runner = Runner(
        model=model,
        work_dir=str(cfg.runner.work_dir),
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        train_cfg=train_cfg,
        val_cfg=val_cfg,
        optim_wrapper=optim_wrapper,
        default_hooks=default_hooks,
        custom_hooks=[DotaBatchHook()],
        launcher=str(cfg.runner.launcher),
        randomness={"seed": int(cfg.runner.seed)},
        cfg=cfg_dict,
    )
    return runner


def _build_optim_wrapper(cfg: DictConfig, model: Any) -> OptimWrapper | dict[str, Any]:
    optimizer_cfg = cfg.optimizer
    if isinstance(optimizer_cfg, DictConfig) and "_target_" in optimizer_cfg:
        optimizer = hydra.utils.instantiate(optimizer_cfg, params=model.parameters())
        return OptimWrapper(optimizer=optimizer)
    return {
        "type": "OptimWrapper",
        "optimizer": dict(optimizer_cfg),
    }


@hydra.main(
    config_path="../configs/object_detection",
    config_name="mmdet_dota_trainer",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    model = hydra.utils.instantiate(cfg.model)
    train_loader = hydra.utils.instantiate(cfg.dataloader.train)
    val_loader = None
    if bool(cfg.runner.do_val):
        val_loader = hydra.utils.instantiate(cfg.dataloader.val)

    runner = _build_runner(cfg, model, train_loader, val_loader)
    logger.info("Start training with MMDetection runner.")
    runner.train()


if __name__ == "__main__":
    main()

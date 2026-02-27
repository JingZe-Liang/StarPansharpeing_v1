from __future__ import annotations

from typing import Any

import hydra
from loguru import logger
from mmengine.evaluator import BaseMetric, Evaluator
from mmengine.optim import OptimWrapper
from mmengine.runner import Runner
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader


def _build_custom_hooks(cfg: DictConfig) -> list[Any]:
    custom_hooks_cfg = cfg.get("custom_hooks")
    if custom_hooks_cfg is None:
        return []
    hooks: list[Any] = []
    for hook_cfg in custom_hooks_cfg:
        hook = hydra.utils.instantiate(hook_cfg)
        hooks.append(hook)
    return hooks


def _build_val_evaluator(cfg: DictConfig) -> Evaluator | dict[Any, Any] | list[Any] | None:
    evaluator_cfg = cfg.get("val_evaluator")
    if evaluator_cfg is None:
        return None

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


def _build_runner(
    cfg: DictConfig,
    model: Any,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    val_evaluator: Any | None,
) -> Runner:
    optim_wrapper = _build_optim_wrapper(cfg, model)
    amp_cfg = cfg.get("amp")
    amp_enabled = bool(amp_cfg.enabled) if amp_cfg is not None and "enabled" in amp_cfg else False
    train_cfg = {
        "type": "EpochBasedTrainLoop",
        "max_epochs": int(cfg.runner.max_epochs),
        "val_interval": int(cfg.runner.val_interval),
    }
    val_cfg = (
        {
            "type": "ValLoop",
            "fp16": amp_enabled,
        }
        if (val_loader is not None and val_evaluator is not None)
        else None
    )

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

    param_scheduler_cfg = cfg.get("param_scheduler")
    param_scheduler: dict[str, Any] | list[Any] | None
    if param_scheduler_cfg is None:
        param_scheduler = None
    else:
        param_scheduler_resolved = OmegaConf.to_container(param_scheduler_cfg, resolve=True)
        if isinstance(param_scheduler_resolved, dict):
            param_scheduler = {str(k): v for k, v in param_scheduler_resolved.items()}
        elif isinstance(param_scheduler_resolved, list):
            param_scheduler = param_scheduler_resolved
        else:
            raise TypeError("param_scheduler must resolve to dict or list.")

    runner = Runner(
        model=model,
        work_dir=str(cfg.runner.work_dir),
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        train_cfg=train_cfg,
        val_cfg=val_cfg,
        val_evaluator=val_evaluator,
        optim_wrapper=optim_wrapper,
        param_scheduler=param_scheduler,
        default_hooks=default_hooks,
        custom_hooks=_build_custom_hooks(cfg),
        load_from=(str(cfg.runner.load_from) if "load_from" in cfg.runner and cfg.runner.load_from else None),
        resume=(bool(cfg.runner.resume) if "resume" in cfg.runner else False),
        launcher=str(cfg.runner.launcher),
        randomness={"seed": int(cfg.runner.seed)},
        cfg=cfg_dict,
    )
    return runner


def _build_optim_wrapper(cfg: DictConfig, model: Any) -> OptimWrapper | dict[str, Any]:
    optimizer_cfg = cfg.optimizer
    paramwise_cfg = cfg.get("paramwise_cfg")
    amp_cfg = cfg.get("amp")
    amp_enabled = bool(amp_cfg.enabled) if amp_cfg is not None and "enabled" in amp_cfg else False
    wrapper_type = "AmpOptimWrapper" if amp_enabled else "OptimWrapper"

    if isinstance(optimizer_cfg, DictConfig) and "_target_" in optimizer_cfg:
        optimizer_cfg_dict = OmegaConf.to_container(optimizer_cfg, resolve=True)
        if not isinstance(optimizer_cfg_dict, dict):
            raise TypeError("optimizer config must resolve to a dict.")

        ignored_keys_for_muon = tuple(optimizer_cfg_dict.pop("ignored_keys_for_muon", []))
        oned_param_algo = str(optimizer_cfg_dict.pop("oned_param_algo", "lion"))
        auto_split_muon_params = bool(optimizer_cfg_dict.pop("auto_split_muon_params", True))

        target = optimizer_cfg_dict.get("_target_")
        params_for_optimizer: Any = model.parameters()
        if isinstance(target, str) and auto_split_muon_params:
            optimizer_cls = hydra.utils.get_class(target)
            clear_fn = getattr(optimizer_cls, "clear_muon_adamw_params", None)
            if callable(clear_fn):
                muon_group, oned_group = clear_fn(
                    named_params=model.named_parameters(),
                    ignored_keys_for_muon=ignored_keys_for_muon,
                    oned_param_algo=oned_param_algo,
                )
                params_for_optimizer = [dict(muon_group), dict(oned_group)]

        if not isinstance(target, str):
            raise TypeError("optimizer._target_ must be a class path string.")
        optimizer_cls = hydra.utils.get_class(target)
        optimizer_kwargs: dict[str, Any] = {}
        for key, value in optimizer_cfg_dict.items():
            if key == "_target_":
                continue
            optimizer_kwargs[str(key)] = value
        optimizer = optimizer_cls(params=params_for_optimizer, **optimizer_kwargs)
        if amp_enabled:
            amp_kwargs: dict[str, Any] = {}
            if "dtype" in amp_cfg:
                amp_kwargs["dtype"] = str(amp_cfg.dtype)
            if "loss_scale" in amp_cfg:
                amp_kwargs["loss_scale"] = amp_cfg.loss_scale
            if "use_fsdp" in amp_cfg:
                amp_kwargs["use_fsdp"] = bool(amp_cfg.use_fsdp)
            return {
                "type": wrapper_type,
                "optimizer": optimizer,
                **amp_kwargs,
            }
        return OptimWrapper(optimizer=optimizer)

    wrapper_cfg: dict[str, Any] = {
        "type": wrapper_type,
        "optimizer": dict(optimizer_cfg),
    }
    if paramwise_cfg is not None:
        paramwise_cfg_resolved = OmegaConf.to_container(paramwise_cfg, resolve=True)
        if not isinstance(paramwise_cfg_resolved, dict):
            raise TypeError("paramwise_cfg must resolve to dict.")
        wrapper_cfg["paramwise_cfg"] = {str(k): v for k, v in paramwise_cfg_resolved.items()}
    if amp_enabled:
        if "dtype" in amp_cfg:
            wrapper_cfg["dtype"] = str(amp_cfg.dtype)
        if "loss_scale" in amp_cfg:
            wrapper_cfg["loss_scale"] = amp_cfg.loss_scale
        if "use_fsdp" in amp_cfg:
            wrapper_cfg["use_fsdp"] = bool(amp_cfg.use_fsdp)
    return wrapper_cfg


@hydra.main(
    config_path="../configs/object_detection",
    config_name="mmdet_dior_trainer",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    model = hydra.utils.instantiate(cfg.model)
    train_loader = hydra.utils.instantiate(cfg.dataloader.train)
    val_loader = None
    val_evaluator: Evaluator | dict[Any, Any] | list[Any] | None = None
    if bool(cfg.runner.do_val):
        val_evaluator = _build_val_evaluator(cfg)
        if val_evaluator is None:
            logger.warning("runner.do_val=true but val_evaluator is missing. Disable validation loop.")
        else:
            val_loader = hydra.utils.instantiate(cfg.dataloader.val)

    runner = _build_runner(cfg, model, train_loader, val_loader, val_evaluator)
    logger.info("Start training with MMDetection runner.")
    runner.train()


if __name__ == "__main__":
    main()

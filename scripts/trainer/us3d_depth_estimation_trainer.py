from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import accelerate
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.state import PartialState
from accelerate.tracking import TensorBoardTracker
from accelerate.utils.deepspeed import DummyOptim, DummyScheduler
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import save_file
from torch import Tensor
from tqdm import tqdm, trange

from src.stage2.depth_estimation.metrics import DepthEstimationMetrics
from src.stage2.depth_estimation.utils.diagnostics import summarize_depth_targets
from src.utilities.train_utils.state import LossMetricTracker


@dataclass
class ScalarAverageMeter:
    sum: float = 0.0
    n: int = 0

    def reset(self) -> None:
        self.sum = 0.0
        self.n = 0

    def update(self, value: Tensor) -> None:
        self.sum += float(value.detach().float().item())
        self.n += 1

    def compute(self) -> float:
        if self.n == 0:
            return 0.0
        return self.sum / self.n


@dataclass
class DepthEstimationOutput:
    loss: Tensor
    pred_depth: Tensor
    log_losses: dict[str, Tensor] | None = None

    def __post_init__(self) -> None:
        if self.log_losses is None:
            self.log_losses = {"loss": self.loss.detach()}


def masked_mean(value: Tensor, valid_mask: Tensor) -> Tensor:
    valid = valid_mask.to(dtype=value.dtype)
    denom = valid.sum().clamp_min(1.0)
    return (value * valid).sum() / denom


def masked_l1(pred: Tensor, target: Tensor, valid_mask: Tensor) -> Tensor:
    return masked_mean((pred - target).abs(), valid_mask)


def masked_rmse(pred: Tensor, target: Tensor, valid_mask: Tensor) -> Tensor:
    mse = masked_mean((pred - target) ** 2, valid_mask)
    return torch.sqrt(mse.clamp_min(0.0))


def _normalize_rgb(img: Tensor) -> np.ndarray:
    img_np = img.detach().float().cpu().numpy()
    if img_np.shape[0] > 3:
        img_np = img_np[:3]
    img_np = img_np.transpose(1, 2, 0)
    min_val = np.nanmin(img_np)
    max_val = np.nanmax(img_np)
    denom = max(max_val - min_val, 1e-6)
    img_np = (img_np - min_val) / denom
    return np.clip(img_np, 0.0, 1.0)


def save_agl_visualization(
    pred: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    image: Tensor,
    out_path: Path,
    *,
    align_to_target: bool = False,
) -> None:
    if align_to_target:
        from src.stage2.depth_estimation.loss.basic import align_scale_shift

        pred, _, _ = align_scale_shift(pred, target, valid_mask)

    pred_np = pred.detach().float().cpu().squeeze(0).squeeze(0).numpy()
    target_np = target.detach().float().cpu().squeeze(0).squeeze(0).numpy()
    valid_np = valid_mask.detach().cpu().squeeze(0).squeeze(0).numpy().astype(bool)
    image_np = _normalize_rgb(image.detach().float().cpu().squeeze(0))

    pred_vis = np.where(valid_np, pred_np, np.nan)
    target_vis = np.where(valid_np, target_np, np.nan)
    error_vis = np.where(valid_np, np.abs(pred_np - target_np), np.nan)

    vmin = np.nanmin(target_vis) if np.isfinite(target_vis).any() else 0.0
    vmax = np.nanmax(target_vis) if np.isfinite(target_vis).any() else 1.0

    err_min = np.nanmin(error_vis) if np.isfinite(error_vis).any() else 0.0
    err_max = np.nanmax(error_vis) if np.isfinite(error_vis).any() else 1.0

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image_np)
    axes[0].set_title("RGB")
    axes[0].axis("off")

    axes[1].imshow(target_vis, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("AGL GT")
    axes[1].axis("off")

    axes[2].imshow(pred_vis, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[2].set_title("AGL Pred (Aligned)" if align_to_target else "AGL Pred")
    axes[2].axis("off")

    axes[3].imshow(error_vis, cmap="magma", vmin=err_min, vmax=err_max)
    axes[3].set_title("Abs Error (Aligned)" if align_to_target else "Abs Error")
    axes[3].axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


class US3DDepthEstimationTrainer:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.train_cfg = cfg.train
        self.dataset_cfg = cfg.dataset
        self.val_cfg = cfg.val

        dataset_scale = getattr(self.dataset_cfg, "scale", None)
        self.depth_scale: float | None = None if dataset_scale is None else float(dataset_scale)

        self.accelerator: Accelerator = hydra.utils.instantiate(cfg.accelerator)
        accelerate.utils.set_seed(int(getattr(self.train_cfg, "seed", 2025)))

        self.device = self.accelerator.device
        torch.cuda.set_device(self.accelerator.local_process_index)

        self.dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "no": torch.float32,
        }[self.accelerator.mixed_precision]

        self.proj_dir = self._configure_logger()

        self.train_dataset, self.train_dataloader = hydra.utils.instantiate(self.dataset_cfg.train)
        self.val_dataset, self.val_dataloader = hydra.utils.instantiate(self.dataset_cfg.val)

        self.model = hydra.utils.instantiate(cfg.depth_model)
        self.optim, self.sched = self._create_optimizer_and_scheduler()

        # activation checkpointing
        if hasattr(self.model, "set_grad_checkpointing") and self.train_cfg.act_checkpoint:
            self.model.set_grad_checkpointing(True)
        # model info
        if self.accelerator.is_main_process:
            try:
                from fvcore.nn import parameter_count_table

                print(parameter_count_table(self.model))
            except Exception as exc:
                self._log(f"Failed to print params table: {exc}", level="WARNING")

        self.model, self.optim, self.train_dataloader, self.val_dataloader, self.sched = self.accelerator.prepare(
            self.model,
            self.optim,
            self.train_dataloader,
            self.val_dataloader,
            self.sched,
        )

        self.loss_tracker = LossMetricTracker(loss_metrics_tracked={"total": []})
        self.val_metrics = DepthEstimationMetrics(eps=float(getattr(self.val_cfg, "eps", 1e-6))).to(self.device)

        if hasattr(self.train_cfg, "loss"):
            self.loss_fn = hydra.utils.instantiate(self.train_cfg.loss)
            self._log(f"use depth loss: {self.loss_fn.__class__.__name__}")
        else:
            self.loss_fn = None
            self._log("use depth loss: masked_l1")

        self.global_step = 0

    def _log(self, msg: str, *, level: str = "INFO") -> None:
        if self.accelerator.is_main_process:
            getattr(logger, level.lower())(msg)

    def _configure_logger(self) -> Path:
        log_dir = Path(self.train_cfg.proj_dir)
        if self.train_cfg.log.log_with_time:
            log_dir = log_dir / time.strftime("%Y-%m-%d_%H-%M-%S")
        if self.train_cfg.log.run_comment is not None:
            log_dir = Path(log_dir.as_posix() + "_" + self.train_cfg.log.run_comment)

        log_file = log_dir / "log.log"

        if self.accelerator.use_distributed:
            if self.accelerator.is_main_process:
                input_list: list[Path | None] = [log_file] * self.accelerator.num_processes
            else:
                input_list = [None] * self.accelerator.num_processes
            output_list: list[Path | None] = [None]
            torch.distributed.scatter_object_list(output_list, input_list, src=0)
            log_file = cast(Path, output_list[0])

        logger.remove()
        logger.add(
            sys.stdout,
            format="{time:HH:mm:ss} - {level.icon} <level>[{level}:{file.name}:{line}]</level>- <level>{message}</level>",
            level="DEBUG",
            backtrace=True,
            colorize=True,
        )

        if not self.train_cfg.debug:
            log_dir.mkdir(parents=True, exist_ok=True)
            logger.add(
                log_file,
                format="<green>[{time:MM-DD HH:mm:ss}]</green> - <level>[{level}]</level> - <cyan>{file}:{line}</cyan> - <level>{message}</level>",
                level="INFO",
                rotation="10 MB",
                enqueue=True,
                backtrace=True,
                colorize=False,
            )

            yaml_cfg = OmegaConf.to_yaml(self.cfg, resolve=True)
            cfg_cp_path = log_dir / "config" / "config_total.yaml"
            cfg_cp_path.parent.mkdir(parents=True, exist_ok=True)
            cfg_cp_path.write_text(yaml_cfg)

            self.accelerator.project_configuration.project_dir = str(log_dir)
            tenb_dir = log_dir / "tensorboard"
            self.accelerator.project_configuration.logging_dir = str(tenb_dir)

            if self.accelerator.is_main_process:
                self.accelerator.init_trackers("us3d_depth_estimation")
                if "tensorboard" in self.train_cfg.log.log_with:
                    _tb: TensorBoardTracker = self.accelerator.get_tracker("tensorboard")  # type: ignore[assignment]
                    logger.log("NOTE", "will log with tensorboard")

        self._log(f"Log file: {log_file}")
        self._log(f"Project dir: {log_dir}")
        return log_dir

    def _create_optimizer_and_scheduler(self) -> tuple[torch.optim.Optimizer | DummyOptim, Any]:
        _dpsp_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        if _dpsp_plugin is not None:
            optim = DummyOptim(self.model.parameters())
            sched = DummyScheduler(optim, total_num_steps=int(self.train_cfg.max_steps))
            return optim, sched

        optim_factory = hydra.utils.instantiate(self.train_cfg.optim)
        if "muon" in self.train_cfg.optim._target_:
            optim = optim_factory(self.model.named_parameters())
        else:
            optim = optim_factory(self.model.parameters())

        sched_factory = hydra.utils.instantiate(self.train_cfg.sched)
        sched = sched_factory(optim)

        return optim, sched

    def _forward(self, img: Tensor) -> Tensor:
        out = self.model(img)
        if isinstance(out, list | tuple):
            out = out[0]
        return cast(Tensor, out)

    def train_step(self, batch: dict[str, Tensor]) -> DepthEstimationOutput:
        img = batch["img"].to(self.device, dtype=self.dtype)
        depth_gt = batch["depth"].to(self.device, dtype=self.dtype)
        valid_mask = batch["valid_mask"].to(self.device)

        # Check for NaN in input data
        # if torch.isnan(img).any():
        #     self._log("WARNING: NaN detected in input img!", level="WARNING")
        # if torch.isnan(depth_gt).any():
        #     self._log("WARNING: NaN detected in depth_gt!", level="WARNING")
        #     self._log(
        #         f"depth_gt stats: min={depth_gt.min()}, max={depth_gt.max()}, mean={depth_gt.mean()}", level="WARNING"
        #     )

        with self.accelerator.autocast():
            pred = self._forward(img)

            # Check for NaN in model output
            # if torch.isnan(pred).any():
            #     self._log("WARNING: NaN detected in model prediction!", level="WARNING")
            #     self._log(f"pred stats: min={pred.min()}, max={pred.max()}, mean={pred.mean()}", level="WARNING")

            if pred.shape[-2:] != depth_gt.shape[-2:]:
                logger.warning(
                    f"Iterpolated prediction to match depth_gt shape: {pred.shape[-2:]} -> {depth_gt.shape[-2:]}"
                )
                pred = F.interpolate(pred, size=depth_gt.shape[-2:], mode="bilinear", align_corners=False)

                # Check after interpolation
                # if torch.isnan(pred).any():
                #     self._log("WARNING: NaN detected after interpolation!", level="WARNING")

            if self.accelerator.is_main_process and self.global_step % 500 == 0:
                vis_path = self.proj_dir / "vis" / "train" / f"train_step_{self.global_step:09d}.webp"
                scale = self.depth_scale if self.depth_scale is not None else 1.0
                align_for_vis = bool(getattr(self.loss_fn, "ssi_weight", 0.0) > 0.0) if self.loss_fn else False
                save_agl_visualization(
                    pred[:1] * scale,
                    depth_gt[:1] * scale,
                    valid_mask[:1],
                    img[:1],
                    vis_path,
                    align_to_target=align_for_vis,
                )

            if self.loss_fn is None:
                loss = masked_l1(pred, depth_gt, valid_mask)
                log_losses = {"l1": loss.detach()}
            else:
                loss_out = self.loss_fn(
                    pred,
                    depth_gt,
                    valid_mask=valid_mask,
                    image=img,
                    return_logs=True,
                )
                if isinstance(loss_out, tuple):
                    loss, log_losses = loss_out
                else:
                    loss = loss_out
                    log_losses = {"loss": loss.detach()}

            # Check for NaN in loss
            # if torch.isnan(loss):
            #     self._log("ERROR: NaN detected in loss!", level="ERROR")
            #     self._log(f"Loss components: {log_losses}", level="ERROR")
            #     self._log(f"pred range: [{pred.min()}, {pred.max()}]", level="ERROR")
            #     self._log(f"depth_gt range: [{depth_gt.min()}, {depth_gt.max()}]", level="ERROR")
            #     self._log(f"valid_mask sum: {valid_mask.sum()}", level="ERROR")

        self.accelerator.backward(loss)
        if float(self.train_cfg.max_grad_norm) > 0:
            self.accelerator.clip_grad_norm_(self.model.parameters(), float(self.train_cfg.max_grad_norm))
        self.optim.step()
        self.sched.step()
        self.optim.zero_grad(set_to_none=True)

        return DepthEstimationOutput(
            loss=loss,
            pred_depth=pred.detach(),
            log_losses=log_losses,
        )

    def _update_loss_tracker(self, loss: Tensor, log_losses: dict[str, Tensor] | None) -> None:
        self.loss_tracker.add_tracked("total", float(loss.detach().item()))
        if log_losses is None:
            return
        for name, value in log_losses.items():
            self.loss_tracker.add_tracked(name, float(value.detach().item()))

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        self.model.eval()
        self.val_metrics.reset()

        iter_fn = self.finite_val_loader()

        for i, batch in enumerate(iter_fn):
            img = batch["img"].to(self.device, dtype=self.dtype)
            depth_gt = batch["depth"].to(self.device, dtype=self.dtype)
            valid_mask = batch["valid_mask"].to(self.device)

            with self.accelerator.autocast():
                pred = self._forward(img)
                if pred.shape[-2:] != depth_gt.shape[-2:]:
                    pred = F.interpolate(pred, size=depth_gt.shape[-2:], mode="bilinear", align_corners=False)

                if self.depth_scale is None:
                    pred_eval = pred
                    target_eval = depth_gt
                else:
                    pred_eval = pred * self.depth_scale
                    target_eval = depth_gt * self.depth_scale

                # ty has trouble with torchmetrics' dynamic method signatures under `Metric`.
                self.val_metrics.update(pred=pred_eval, target=target_eval, valid_mask=valid_mask)  # type: ignore[misc]

            if i == 0 and self.accelerator.is_main_process:
                eps = float(getattr(self.val_cfg, "eps", 1e-6))
                self._log(summarize_depth_targets(target_eval, valid_mask, eps=eps), level="INFO")
                vis_path = self.proj_dir / "vis" / "val" / f"val_step_{self.global_step:09d}.webp"
                align_for_vis = bool(getattr(self.loss_fn, "ssi_weight", 0.0) > 0.0) if self.loss_fn else False
                save_agl_visualization(
                    pred_eval[:1],
                    target_eval[:1],
                    valid_mask[:1],
                    img[:1],
                    vis_path,
                    align_to_target=align_for_vis,
                )

        metrics_t = self.val_metrics.compute()  # type: ignore[misc]
        metrics = {
            "val/mae": float(metrics_t["mae"].item()),
            "val/rmse": float(metrics_t["rmse"].item()),
            "val/absrel": float(metrics_t["absrel"].item()),
            "val/logrmse": float(metrics_t["logrmse"].item()),
            "val/delta1": float(metrics_t["delta1"].item()),
            "val/delta2": float(metrics_t["delta2"].item()),
            "val/delta3": float(metrics_t["delta3"].item()),
            "val/n_pos": float(metrics_t["n_pos"].item()),
            "val/n_valid": float(metrics_t["n_valid"].item()),
        }

        self.model.train()
        return metrics

    def save_checkpoint(self, step: int) -> None:
        if not self.accelerator.is_main_process:
            return

        save_dir = self.proj_dir / "checkpoints"
        save_dir.mkdir(parents=True, exist_ok=True)

        unwrapped = self.accelerator.unwrap_model(self.model)
        state_dict = unwrapped.state_dict()
        self.accelerator.save_model(unwrapped, save_directory=save_dir.as_posix())

        meta = {
            "global_step": step,
            "cfg": OmegaConf.to_container(self.cfg, resolve=True),
        }
        (save_dir / "meta.yaml").write_text(OmegaConf.to_yaml(meta, resolve=True))
        logger.success(f"Saved checkpoint to {save_dir}")  # may overwrite previous

    def infinite_train_loader(self):
        while True:
            if hasattr(self, "_train_loader_iter"):
                try:
                    yield next(self._train_loader_iter)
                except StopIteration:
                    self._train_loader_iter = iter(self.train_dataloader)
                    yield next(self._train_loader_iter)
            else:
                self._train_loader_iter = iter(self.train_dataloader)
                yield next(self._train_loader_iter)

    def finite_val_loader(self):
        max_iters = int(getattr(self.val_cfg, "max_val_iters", -1))

        if max_iters < 0:
            # itering all val loader
            for batch in tqdm(self.val_dataloader, desc="Validation ...", disable=not self.accelerator.is_main_process):
                yield batch
        else:
            if not hasattr(self, "_val_loader_iter"):
                # continue to iterate (the rest of the val loader)
                self._val_loader_iter = iter(self.val_dataloader)
            for i in trange(max_iters, desc="Validation ...", disable=not self.accelerator.is_main_process):
                try:
                    yield next(self._val_loader_iter)
                except StopIteration:
                    self._val_loader_iter = iter(self.val_dataloader)
                    yield next(self._val_loader_iter)

    def run(self) -> None:
        self.model.train()
        max_steps = int(self.train_cfg.max_steps)
        log_every = int(self.train_cfg.log.log_every)
        save_every = int(self.train_cfg.save_every)
        val_every = int(getattr(self.val_cfg, "every", 1000))
        step = 0
        train_loader = self.infinite_train_loader()

        for step in range(1, max_steps + 1):
            self.global_step = step
            batch = next(train_loader)
            out = self.train_step(batch)

            self._update_loss_tracker(out.loss, out.log_losses)
            if step % log_every == 0:
                lr = float("nan")
                if hasattr(self.optim, "param_groups") and len(self.optim.param_groups) > 0:  # type: ignore[attr-defined]
                    lr = float(self.optim.param_groups[0]["lr"])  # type: ignore[index]
                tracked = self.loss_tracker.get_tracked_values_op(
                    name=None,
                    track_value_op="mean",
                    round_decimals=6,
                )
                metrics = {f"train/{k}": float(v) for k, v in tracked.items() if v is not None}
                metrics["train/lr"] = lr
                if self.accelerator.is_main_process:
                    self.accelerator.log(metrics, step=step)
                    self._log(f"step={step} " + " ".join([f"{k}={v:.6f}" for k, v in metrics.items()]))
                    self._log(
                        f"gt min/max={batch['depth'].min().item():.6f}/{batch['depth'].max().item():.6f} - "
                        f"pred min/max={out.pred_depth.min().item():.6f}/{out.pred_depth.max().item():.6f}\n"
                    )
                self.loss_tracker.clear_tracked(clear_name=False)

            if val_every > 0 and step % val_every == 0:
                val_metrics = self.validate()
                if self.accelerator.is_main_process:
                    self.accelerator.log(val_metrics, step=step)
                    self._log(" ".join([f"{k}={v:.4f}" for k, v in val_metrics.items()]))

            if save_every > 0 and step % save_every == 0:
                self.save_checkpoint(step)


_configs_dict = {
    "us3d_depth_estimation": "us3d_depth_estimation",
}
_key = "us3d_depth_estimation"


if __name__ == "__main__":
    from src.utilities.train_utils.cli import argsparse_cli_args, print_colored_banner

    cli_default_dict = {
        "config_name": _key,
        "only_rank_zero_catch": True,
    }
    chosen_cfg, _cli_args = argsparse_cli_args(_configs_dict, cli_default_dict)
    if PartialState().is_main_process:
        print_colored_banner("Depth Estimation")

    @hydra.main(
        config_path="../configs/depth_estimation",
        config_name=chosen_cfg,
        version_base=None,
    )
    def main(cfg: DictConfig) -> None:
        trainer = US3DDepthEstimationTrainer(cfg)
        trainer.run()

    main()

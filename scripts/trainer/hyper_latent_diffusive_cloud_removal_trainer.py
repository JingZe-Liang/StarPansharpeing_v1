from __future__ import annotations

import os
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast, overload

import accelerate
import hydra
import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.state import PartialState
from accelerate.tracking import TensorBoardTracker
from accelerate.utils.deepspeed import DummyOptim, DummyScheduler
from ema_pytorch import EMA
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from tqdm import tqdm, trange

from src.stage2.cloud_removal.data.sen12_vis import s1_to_gray_for_display
from src.stage2.cloud_removal.diffusion.loss import (
    CloudRemovalSILoss,
    apply_time_shift,
    estimate_x0_from_v,
    pack_conditions,
)
from src.stage2.cloud_removal.diffusion.vae import CosmosRSVAE
from src.utilities.logging import dict_round_to_list_str
from src.utilities.train_utils.state import StepsCounter, dict_tensor_sync, object_scatter
from src.utilities.train_utils.visualization import visualize_hyperspectral_image


@dataclass(frozen=True)
class TrainCloudRemovalStepOutput:
    loss: Tensor
    log_losses: dict[str, Tensor]


class HyperLatentDiffusiveCloudRemovalTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.train_cfg = cfg.train
        self.dataset_cfg = cfg.dataset
        self.ema_cfg = cfg.ema
        self.val_cfg = cfg.val

        self.accelerator: Accelerator = hydra.utils.instantiate(cfg.accelerator)
        accelerate.utils.set_seed(2025)

        log_file = self.configure_logger()

        self.device = self.accelerator.device
        torch.cuda.set_device(self.accelerator.local_process_index)
        self.dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "no": torch.float32,
        }[self.accelerator.mixed_precision]
        self.log_msg(f"Log file is saved at: {log_file}")
        self.log_msg(f"Weights will be saved at: {self.proj_dir}")

        _dpsp_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        _fsdp_plugin: accelerate.utils.FullyShardedDataParallelPlugin | None = getattr(
            self.accelerator.state, "fsdp_plugin", None
        )
        self.no_ema = False
        self._is_ds = _dpsp_plugin is not None
        if self._is_ds:
            self.log_msg("[Deepspeed]: using deepspeed plugin")
            self.no_ema = _dpsp_plugin.deepspeed_config["zero_optimization"]["stage"] in [2, 3]  # type: ignore[index]
        self._is_fsdp = _fsdp_plugin is not None
        if self._is_fsdp:
            self.log_msg("[FSDP]: using Fully Sharded Data Parallel plugin")
            self.no_ema = True

        used_dataset = getattr(getattr(self.dataset_cfg, "cfgs", None), "used", "unknown")
        self.log_msg(f"[Data]: using dataset {used_dataset}")
        self.train_dataset, self.train_dataloader = hydra.utils.instantiate(self.dataset_cfg.train_loader)
        self.val_dataset, self.val_dataloader = hydra.utils.instantiate(self.dataset_cfg.val_loader)
        if _dpsp_plugin is not None:
            micro_batch_size = getattr(self.dataset_cfg, "batch_size_train", None)
            if micro_batch_size is None:
                micro_batch_size = getattr(self.train_dataloader, "batch_size", None)
            if micro_batch_size is None:
                raise ValueError("Cannot infer train micro batch size for deepspeed; set dataset.batch_size_train.")
            self.accelerator.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (  # type: ignore[index]
                micro_batch_size
            )

        self.setup_cloud_removal_model()
        self.optim, self.sched = self.get_optimizer_lr_scheduler()

        self.vae: CosmosRSVAE | None = self._build_vae()
        self.prepare_for_training()
        self.prepare_ema_models()

        self.loss_fn: CloudRemovalSILoss = self._build_loss_fn()

        self.train_state = StepsCounter(["train", "val"])

    def _build_vae(self) -> CosmosRSVAE | None:
        if not hasattr(self.cfg, "vae") or self.cfg.vae is None:
            self.log_msg("[VAE]: disabled (cfg.vae not set)")
            return None

        vae = hydra.utils.instantiate(self.cfg.vae)
        if not isinstance(vae, CosmosRSVAE):
            raise TypeError(f"cfg.vae must be CosmosRSVAE for now, got {type(vae)}")
        vae = vae.to(device=self.device)
        vae.eval().requires_grad_(False)
        self.log_msg(f"[VAE]: loaded {vae.__class__.__name__}")
        return vae

    def _build_loss_fn(self) -> CloudRemovalSILoss:
        if hasattr(self.cfg, "loss"):
            loss = hydra.utils.instantiate(self.cfg.loss)
            if not isinstance(loss, CloudRemovalSILoss):
                raise TypeError(f"cfg.loss must be CloudRemovalSILoss, got {type(loss)}")
            return loss

        return CloudRemovalSILoss(
            path_type=getattr(self.train_cfg, "path_type", "linear"),
            weighting=getattr(self.train_cfg, "time_weighting", "uniform"),
            apply_time_shift=bool(getattr(self.train_cfg, "time_shifting", False)),
            shift_base=int(getattr(self.train_cfg, "shift_base", 4096)),
        )

    def setup_cloud_removal_model(self) -> None:
        model_cfg: Any | None = None
        for key in ("cloud_removal_model", "diffusion_model", "model"):
            if hasattr(self.cfg, key):
                model_cfg = getattr(self.cfg, key)
                if key in {"model", "diffusion_model"} and hasattr(model_cfg, "cloud_removal_model"):
                    model_cfg = model_cfg.cloud_removal_model
                break
        if model_cfg is None:
            raise ValueError("Config must provide one of: cfg.cloud_removal_model / cfg.diffusion_model / cfg.model")
        self.model = hydra.utils.instantiate(model_cfg)

        model_name = getattr(self.train_cfg, "cloud_removal_name", None) or self.model.__class__.__name__
        self.log_msg(f"[Model]: use cloud removal diffusion model: {model_name}")

        if getattr(self.train_cfg, "compile_model", False):
            self.model = torch.compile(self.model)
            self.log_msg(f"[Model]: compiled {model_name}")

    def prepare_ema_models(self) -> None:
        if self.no_ema:
            return
        self.ema_model: EMA = hydra.utils.instantiate(self.ema_cfg)(self.model)
        self.ema_model.to(self.device)
        self.log_msg("[EMA]: create EMA model")

    def configure_logger(self) -> Path:
        self.logger = logger

        log_file = Path(self.train_cfg.proj_dir)
        if self.train_cfg.log.log_with_time:
            str_time = time.strftime("%Y-%m-%d_%H-%M-%S")
            log_file = log_file / str_time
        if self.train_cfg.log.run_comment is not None:
            log_file = Path(log_file.as_posix() + "_" + self.train_cfg.log.run_comment)
        log_file = log_file / "log.log"

        log_file = object_scatter(log_file)
        if not isinstance(log_file, Path):
            raise TypeError("log_file type should be Path")

        self.logger.remove()
        log_format_in_file = (
            "<green>[{time:MM-DD HH:mm:ss}]</green> "
            "- <level>[{level}]</level> "
            "- <cyan>{file}:{line}</cyan> - <level>{message}</level>"
        )
        log_format_in_cmd = (
            "{time:HH:mm:ss} - {level.icon} <level>[{level}:{file.name}:{line}]</level>- <level>{message}</level>"
        )
        if not self.train_cfg.debug:
            self.logger.add(
                log_file,
                format=log_format_in_file,
                level="INFO",
                rotation="10 MB",
                enqueue=True,
                backtrace=True,
                colorize=False,
            )
        self.logger.add(
            sys.stderr,
            format=log_format_in_cmd,
            level=os.getenv("SHELL_LOG_LEVEL", "DEBUG"),
            backtrace=True,
            colorize=True,
        )

        log_dir = log_file.parent
        if not self.train_cfg.debug:
            log_dir.mkdir(parents=True, exist_ok=True)

        if not self.train_cfg.debug:
            yaml_cfg = OmegaConf.to_yaml(self.cfg, resolve=True)
            cfg_cp_path = log_dir / "config" / "config_total.yaml"
            cfg_cp_path.parent.mkdir(parents=True, exist_ok=True)
            cfg_cp_path.write_text(yaml_cfg)
            self.logger.info(f"[Cfg]: configuration saved to {cfg_cp_path}")

        self.proj_dir = log_dir
        self.accelerator.project_configuration.project_dir = str(self.proj_dir)

        if not self.train_cfg.debug:
            tenb_dir = log_dir / "tensorboard"
            self.accelerator.project_configuration.logging_dir = tenb_dir
            if self.accelerator.is_main_process:
                self.logger.info(f"[Tensorboard]: tensorboard saved to {tenb_dir}")
                self.accelerator.init_trackers("train")
                self.tb_logger = self.accelerator.get_tracker("tensorboard")  # type: ignore[assignment]

        return log_file

    def tenb_log_any(
        self,
        log_type: Literal["metric", "image"],
        logs: dict[str, Any],
        step: int,
    ) -> None:
        if not hasattr(self, "tb_logger"):
            return
        if log_type == "metric":
            self.tb_logger.log(logs, step=step)  # type: ignore[union-attr]
            return
        if log_type == "image":
            self.tb_logger.log_images(logs, step=step)  # type: ignore[union-attr]
            return
        raise ValueError(f"Unknown {log_type=}")

    def log_msg(self, *msgs: object, only_rank_zero: bool = True, level: str = "INFO", sep: str = ",") -> None:
        if level.lower() not in {"info", "warning", "error", "debug", "critical"}:
            raise ValueError(f"Unknown {level=}")

        def str_msg(*msg: object) -> str:
            return sep.join([str(m) for m in msg])

        log_fn = getattr(self.logger, level.lower())
        if only_rank_zero:
            if self.accelerator.is_main_process:
                log_fn(str_msg(*msgs), stack_level=3)
        else:
            with self.accelerator.main_process_first():
                msg_string = f"rank-{self.accelerator.process_index} | {str_msg(*msgs)}"
                log_fn(msg_string, stack_level=3)

    def get_optimizer_lr_scheduler(
        self,
    ) -> tuple[
        torch.optim.Optimizer | DummyOptim,
        torch.optim.lr_scheduler.LRScheduler | DummyScheduler,
    ]:
        opt_cfg = getattr(self.train_cfg, "cloud_removal_optim", None) or getattr(
            self.train_cfg, "diffusion_optim", None
        )
        sched_cfg = getattr(self.train_cfg, "cloud_removal_sched", None) or getattr(
            self.train_cfg, "diffusion_sched", None
        )
        if opt_cfg is None or sched_cfg is None:
            raise ValueError("train config must define optimizer/scheduler (cloud_removal_optim/cloud_removal_sched)")

        if (
            self.accelerator.state.deepspeed_plugin is None
            or "optimizer" not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):

            def _optimizer_creater(optimizer_cfg: Any):
                if "get_muon_optimizer" in optimizer_cfg._target_:
                    self.log_msg("[Optimizer]: using muon optimizer")
                    return hydra.utils.instantiate(optimizer_cfg)(named_parameters=self.model.named_parameters())
                self.log_msg(f"[Optimizer]: using optimizer: {optimizer_cfg._target_}")
                return hydra.utils.instantiate(optimizer_cfg)(self.model.parameters())

            optim = _optimizer_creater(opt_cfg)
        else:
            optim = DummyOptim([{"params": list(self.model.parameters())}])

        if (
            self.accelerator.state.deepspeed_plugin is None
            or "scheduler" not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            sched = hydra.utils.instantiate(sched_cfg)(optimizer=optim)
        else:
            sched = DummyScheduler(optim)

        is_heavyball_opt = lambda opt: opt.__class__.__module__.startswith("heavyball")
        if is_heavyball_opt(optim):
            import heavyball

            heavyball.utils.compile_mode = None  # type: ignore[invalid-assignment]
            self.log_msg("[Optimizer]: disable heavyball compile_mode for this trainer")

        return optim, sched

    def prepare_for_training(self) -> None:
        self.train_dataloader, self.val_dataloader = self.accelerator.prepare(
            self.train_dataloader, self.val_dataloader
        )
        self.model, self.optim, self.sched = self.accelerator.prepare(self.model, self.optim, self.sched)

    @property
    def global_step(self) -> int:
        return int(self.train_state["train"])

    def step_train_state(self, mode: Literal["train", "val"] = "train") -> None:
        self.train_state.update(mode)

    def ema_update(self) -> None:
        if self.no_ema:
            return
        self.ema_model.update()

    def to_device_dtype(self, x: Any) -> Any:
        if torch.is_tensor(x):
            return x.to(device=self.device, dtype=self.dtype)
        if isinstance(x, dict):
            return {k: self.to_device_dtype(v) for k, v in x.items()}
        if isinstance(x, list):
            return [self.to_device_dtype(v) for v in x]
        if isinstance(x, tuple):
            return tuple(self.to_device_dtype(v) for v in x)
        return x

    def _get_eval_model(self) -> nn.Module:
        if self.no_ema or not hasattr(self, "ema_model"):
            return self.model
        return cast(nn.Module, self.ema_model.ema_model)

    def _extract_conditions(self, batch: dict[str, Any]) -> Tensor | list[Tensor] | None:
        cond_source = getattr(self.train_cfg, "condition_source", "auto")
        if self.vae is not None and cond_source in {"auto", "img"}:
            return batch.get("img", None)
        if "conditions" in batch:
            return batch["conditions"]
        return batch.get("img", None)

    @overload
    def _encode_if_needed(self, x: Tensor) -> Tensor: ...
    @overload
    def _encode_if_needed(self, x: list[Tensor]) -> list[Tensor]: ...
    @overload
    def _encode_if_needed(self, x: tuple[Tensor, ...]) -> tuple[Tensor, ...]: ...
    @overload
    def _encode_if_needed(self, x: None) -> None: ...

    def _encode_if_needed(
        self, x: Tensor | list[Tensor] | tuple[Tensor, ...] | None
    ) -> Tensor | list[Tensor] | tuple[Tensor, ...] | None:
        if x is None:
            return None
        if isinstance(x, list):
            return [self._encode_if_needed(xx) for xx in x]  # type: ignore[return-value]
        if isinstance(x, tuple):
            return tuple(self._encode_if_needed(xx) for xx in x)  # type: ignore[return-value]
        if self.vae is None:
            return x
        return self.vae.encode(x)

    def _decode_if_needed(self, z: Tensor, *, input_shape: torch.Size | int) -> Tensor:
        if self.vae is None:
            return z
        return self.vae.decode(z, input_shape=input_shape)

    def train_cloud_removal_step(self, batch: dict[str, Any]) -> TrainCloudRemovalStepOutput:
        gt_px = batch["gt"]
        cond_px = self._extract_conditions(batch)

        gt = self._encode_if_needed(gt_px)
        conditions = self._encode_if_needed(cond_px) if cond_px is not None else None

        with self.accelerator.autocast():
            loss_vec, t, _, _ = self.loss_fn(self.model, gt, conditions=conditions)
            loss = loss_vec.mean()

        if self.accelerator.sync_gradients:
            self.optim.zero_grad()
            self.accelerator.backward(loss)
            if getattr(self.train_cfg, "max_grad_norm", None) is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.train_cfg.max_grad_norm)
            self.optim.step()
            self.sched.step()
            self.ema_update()

        log_losses = {
            "denoise_loss": loss.detach(),
            "t_mean": t.mean().detach(),
        }
        return TrainCloudRemovalStepOutput(loss=loss, log_losses=log_losses)

    def to_rgb(self, x: Tensor) -> Tensor:
        if getattr(self.train_cfg, "is_neg_1_1", False):
            return ((x + 1.0) / 2.0).clamp(0.0, 1.0).float()
        return x.clamp(0.0, 1.0).float()

    def visualize_triplet(
        self,
        *,
        cloudy: Tensor,
        pred: Tensor,
        gt: Tensor,
        sar: Tensor | None = None,
        img_name: str,
        add_step: bool,
        only_vis_n: int = 8,
    ) -> None:
        rgb_channels = getattr(self.train_cfg, "visualize_rgb_channels", None)

        def to_uint8_grid(x: Tensor, text: str = "") -> np.ndarray:
            # Special handling for 2-channel SAR
            if x.shape[1] == 2:
                # Process SAR image using the specialized function to get grayscale [0, 1] HW
                # We do this for each image in the batch and stack them
                x_vis = []
                for i in range(min(len(x), only_vis_n)):
                    # s1_to_gray_for_display expects CHW and returns HW
                    # We default to reasonable visualization params similar to sen12_vis defaults
                    gray_hw = s1_to_gray_for_display(
                        x[i],
                        channel="mean",
                        domain="db10",
                        smooth="median",
                        kernel_size=3,
                        linear_compress="log1p",
                    )
                    # Expand to CHW (1, H, W) -> RGB (3, H, W)
                    rgb_chw = gray_hw.unsqueeze(0).repeat(3, 1, 1)
                    x_vis.append(rgb_chw)

                # Stack back to (B, C, H, W)
                x_disp = torch.stack(x_vis)

                # Now visualize using the standard pipeline, treating it as RGB
                grid = visualize_hyperspectral_image(
                    x_disp.detach().cpu(),
                    rgb_channels=[0, 1, 2],  # It's already RGB
                    to_uint8=True,
                    to_grid=True,
                    nrows=4,
                )
                return grid if isinstance(grid, np.ndarray) else np.asarray(grid)

            grid = visualize_hyperspectral_image(
                self.to_rgb(x[:only_vis_n].detach().cpu()),
                rgb_channels=rgb_channels,
                to_uint8=True,
                to_grid=True,
                nrows=4,
            )
            return grid if isinstance(grid, np.ndarray) else np.asarray(grid)

        imgs_list = [to_uint8_grid(cloudy)]
        if sar is not None:
            imgs_list.append(to_uint8_grid(sar))
        imgs_list.extend([to_uint8_grid(pred), to_uint8_grid(gt)])

        img = np.concatenate(imgs_list, axis=1)
        save_dir = Path(self.proj_dir) / "vis"
        if self.accelerator.is_main_process:
            save_dir.mkdir(parents=True, exist_ok=True)
        if add_step:
            img_name = f"{img_name}_step_{str(self.global_step).zfill(6)}.webp"
        else:
            img_name = f"{img_name}.webp"
        save_path = save_dir / img_name
        if self.accelerator.is_main_process:
            from PIL import Image

            Image.fromarray(img).save(save_path, quality=95)
            self.log_msg(f"[Visualize]: save visualization at {save_path}")

    def simple_euler_sampler(
        self,
        model: nn.Module,
        shape: torch.Size,
        conditions: Tensor | list[Tensor] | tuple[Tensor, ...] | None,
        num_steps: int = 25,
    ) -> Tensor:
        # Initial noise
        x = torch.randn(shape, device=self.device, dtype=self.dtype)

        # Prepare conditions
        cond_tensor = pack_conditions(conditions)

        dt = 1.0 / num_steps

        for i in range(num_steps):
            t_curr = 1.0 - i * dt
            t_vec = torch.full((x.shape[0],), t_curr, device=x.device, dtype=x.dtype)

            # Apply time shift if usually applied
            if getattr(self.train_cfg, "time_shifting", False):
                shift_base = int(getattr(self.train_cfg, "shift_base", 4096))
                shift_dim = x.shape[1] * x.shape[2] * x.shape[3]
                t_vec = apply_time_shift(t_vec, shift_dim=shift_dim, shift_base=shift_base)

            # Model prediction (v)
            v_pred = model(x, t_vec, conditions=cond_tensor)[0]

            # Euler step: x_{t-dt} = x_t - v_t * dt
            x = x - v_pred * dt

        return x

    def train_step(self, batch: dict[str, Any]) -> None:
        batch = self.to_device_dtype(batch)
        with self.accelerator.accumulate(self.model):
            out = self.train_cloud_removal_step(batch)

        self.step_train_state("train")

        if self.global_step % self.train_cfg.log.log_every == 0:
            log_losses = dict_tensor_sync(out.log_losses)
            self.log_msg(
                f"[Train]: lr {self.optim.param_groups[0]['lr']:1.4e} | step {self.global_step}/{self.train_cfg.max_steps}"
            )
            self.log_msg(f"[Train]: {dict_round_to_list_str(log_losses, n_round=4)}")
            self.tenb_log_any("metric", log_losses, step=self.global_step)

        if self.global_step % getattr(self.train_cfg, "visualize_every", 500) == 0:
            gt_px = batch["gt"]
            cloudy_px = batch.get("img", gt_px)
            cond_px = self._extract_conditions(batch)
            model_for_vis = self._get_eval_model()

            with torch.no_grad(), self.accelerator.autocast():
                gt = self._encode_if_needed(gt_px)
                conditions = self._encode_if_needed(cond_px) if cond_px is not None else None

                _, t, x_t, v_t = self.loss_fn(model_for_vis, gt, conditions=conditions)
                x0_hat = estimate_x0_from_v(
                    x_t,
                    v_t,
                    t,
                    path_type=getattr(self.train_cfg, "path_type", "linear"),
                )

                pred_px = self._decode_if_needed(x0_hat, input_shape=gt_px.shape)

            self.visualize_triplet(
                cloudy=cloudy_px,
                pred=pred_px,
                gt=gt_px,
                sar=batch.get("s1", None),
                img_name="train/cloud_removal",
                add_step=True,
            )

    def infinity_train_loader(self):
        while True:
            for batch in self.train_dataloader:
                yield batch

    def save_state(self) -> None:
        self.accelerator.save_state()
        self.log_msg("[State]: save states")

    def save_ema(self) -> None:
        if self.no_ema:
            self.log_msg("[EMA]: deepspeed/FSDP enabled, skip saving EMA")
            return
        ema_path = Path(self.proj_dir) / "ema"
        if self.accelerator.is_main_process:
            ema_path.mkdir(parents=True, exist_ok=True)
        self.accelerator.save_model(
            cast(nn.Module, self.ema_model.ema_model),
            ema_path / "cloud_removal_ema_model",
        )
        accelerate.utils.save(self.train_state.state_dict(), ema_path / "train_state.pth")
        self.log_msg(f"[EMA]: save ema at {ema_path}")

    def load_from_ema(self, ema_path: str | Path, *, strict: bool = True) -> None:
        ema_path = Path(ema_path)
        accelerate.load_checkpoint_in_model(self.model, ema_path / "cloud_removal_ema_model", strict=strict)
        self.prepare_ema_models()
        self.log_msg("[Load EMA]: loaded EMA weights")

    def resume(self, path: str) -> None:
        self.log_msg("[Resume]: resume training")
        self.accelerator.load_state(path)
        self.accelerator.wait_for_everyone()

    def get_val_loader_iter(self):
        if self.val_cfg.max_val_iters > 0:
            if not hasattr(self, "_val_loader_iter"):
                self._val_loader_iter = iter(self.val_dataloader)
            for _ in trange(
                self.val_cfg.max_val_iters,
                desc="validating ...",
                leave=False,
                disable=not self.accelerator.is_main_process,
            ):
                try:
                    batch = next(self._val_loader_iter)
                except StopIteration:
                    self._val_loader_iter = iter(self.val_dataloader)
                    batch = next(self._val_loader_iter)
                yield batch
        else:
            for batch in tqdm(self.val_dataloader, desc="validating ...", disable=not self.accelerator.is_main_process):
                yield batch

    @torch.no_grad()
    def val_loop(self) -> None:
        model_for_eval = self._get_eval_model()
        model_for_eval.eval()

        loss_sum = torch.tensor(0.0, device=self.device)
        loss_count = 0
        last_batch: dict[str, Any] | None = None
        for batch in self.get_val_loader_iter():
            batch = self.to_device_dtype(batch)
            gt_px = batch["gt"]
            cond_px = self._extract_conditions(batch)
            gt = self._encode_if_needed(gt_px)
            conditions = self._encode_if_needed(cond_px) if cond_px is not None else None

            with self.accelerator.autocast():
                loss_vec, _, _, _ = self.loss_fn(model_for_eval, gt, conditions=conditions)
                loss_batch = loss_vec.mean().detach()
                loss_batch = self.accelerator.gather(loss_batch)
                loss_sum += loss_batch.sum()
                loss_count += int(loss_batch.numel())
            self.step_train_state("val")
            last_batch = batch

        loss_val = loss_sum / max(1, loss_count)
        if self.accelerator.is_main_process:
            self.log_msg(f"[Val]: denoise_loss={loss_val.item():.4e}")
            self.tenb_log_any("metric", {"val/denoise_loss": loss_val.item()}, step=self.global_step)

        if last_batch is not None and self.accelerator.is_main_process:
            gt_px = last_batch["gt"]
            cloudy_px = last_batch.get("img", gt_px)
            cond_px = self._extract_conditions(last_batch)
            with self.accelerator.autocast():
                gt = self._encode_if_needed(gt_px)
                conditions = self._encode_if_needed(cond_px) if cond_px is not None else None

                pred_latent = self.simple_euler_sampler(model_for_eval, gt.shape, conditions=conditions)
                pred_px = self._decode_if_needed(pred_latent, input_shape=gt_px.shape)
            self.visualize_triplet(
                cloudy=cloudy_px,
                pred=pred_px,
                gt=gt_px,
                sar=last_batch.get("s1", None),
                img_name="val/cloud_removal",
                add_step=True,
            )

        model_for_eval.train()

    def train_loop(self) -> None:
        stop_and_save = False
        self.accelerator.wait_for_everyone()
        self.log_msg("[Train]: start training", only_rank_zero=False)

        for batch in self.infinity_train_loader():
            self.train_step(batch)

            if self.global_step % self.val_cfg.val_duration == 0:
                self.val_loop()

            if self.global_step >= self.train_cfg.max_steps:
                stop_and_save = True

            if self.global_step % self.train_cfg.save_every == 0 or stop_and_save:
                self.save_state()
                self.save_ema()

            if stop_and_save:
                self.log_msg("[Train]: reached max_steps, stop")
                break

    def run(self) -> None:
        if self.train_cfg.resume_path is not None:
            self.resume(self.train_cfg.resume_path)
        elif self.train_cfg.ema_load_path is not None:
            self.load_from_ema(self.train_cfg.ema_load_path)

        self.train_loop()


_key = "cloud_removal_sen12"
_configs_dict = {
    "cloud_removal_sen12": "cloud_removal_sen12",
}[_key]


if __name__ == "__main__":
    from src.utilities.train_utils.cli import argsparse_cli_args, print_colored_banner

    print_colored_banner("CloudRemoval")

    @hydra.main(
        config_path="../configs/cloud_removal",
        config_name=_configs_dict,
        version_base=None,
    )
    def main(cfg: DictConfig):
        catcher = logger.catch if PartialState().is_main_process else nullcontext
        with catcher():
            trainer = HyperLatentDiffusiveCloudRemovalTrainer(cfg)
            trainer.run()

    main()

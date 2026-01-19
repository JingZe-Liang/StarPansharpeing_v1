from __future__ import annotations

import math
import os
import sys
import time
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from collections.abc import Callable
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
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.functional.image import peak_signal_noise_ratio
from tqdm import tqdm, trange

from src.utilities.logging import configure_logger, log
from src.stage2.cloud_removal.data.sen12_vis import s1_to_gray_for_display
from src.stage2.cloud_removal.diffusion.loss import apply_time_shift, pack_conditions
from src.stage2.cloud_removal.diffusion.vae import CosmosRSVAE
from src.utilities.logging import dict_round_to_list_str
from src.utilities.transport.flow_matching import Sampler, Transport

from src.utilities.train_utils.state import StepsCounter, dict_tensor_sync, object_scatter
from src.utilities.train_utils.visualization import visualize_hyperspectral_image


def compute_gradient_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Compute gradient (edge) loss using Sobel operators.

    Args:
        pred: Predicted tensor [B, C, H, W]
        target: Target tensor [B, C, H, W]

    Returns:
        Gradient loss value
    """
    # Sobel filters
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred.dtype, device=pred.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=pred.dtype, device=pred.device)

    sobel_x = sobel_x.view(1, 1, 3, 3).repeat(pred.shape[1], 1, 1, 1)
    sobel_y = sobel_y.view(1, 1, 3, 3).repeat(pred.shape[1], 1, 1, 1)

    # Compute gradients
    pred_grad_x = torch.nn.functional.conv2d(pred, sobel_x, padding=1, groups=pred.shape[1])
    pred_grad_y = torch.nn.functional.conv2d(pred, sobel_y, padding=1, groups=pred.shape[1])

    target_grad_x = torch.nn.functional.conv2d(target, sobel_x, padding=1, groups=target.shape[1])
    target_grad_y = torch.nn.functional.conv2d(target, sobel_y, padding=1, groups=target.shape[1])

    # L1 loss on gradients
    loss_x = torch.abs(pred_grad_x - target_grad_x).mean()
    loss_y = torch.abs(pred_grad_y - target_grad_y).mean()

    return (loss_x + loss_y) / 2.0


def compute_fft_high_freq_loss(pred: Tensor, target: Tensor, high_freq_ratio: float = 0.3) -> Tensor:
    """Compute FFT high-frequency loss.

    Args:
        pred: Predicted tensor [B, C, H, W]
        target: Target tensor [B, C, H, W]
        high_freq_ratio: Ratio of high frequency components to penalize (default 0.3 = top 30%)

    Returns:
        High-frequency loss value
    """
    # FFT on spatial dimensions
    pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
    target_fft = torch.fft.fft2(target, dim=(-2, -1))

    # Shift zero frequency to center
    pred_fft = torch.fft.fftshift(pred_fft, dim=(-2, -1))
    target_fft = torch.fft.fftshift(target_fft, dim=(-2, -1))

    # Create high-pass mask (keep outer regions, zero center)
    B, C, H, W = pred.shape
    center_h, center_w = H // 2, W // 2
    mask_h = int(H * (1 - high_freq_ratio) / 2)
    mask_w = int(W * (1 - high_freq_ratio) / 2)

    mask = torch.ones_like(pred_fft.real)
    mask[..., center_h - mask_h : center_h + mask_h, center_w - mask_w : center_w + mask_w] = 0

    # Apply mask to keep only high frequencies
    pred_fft_high = pred_fft * mask
    target_fft_high = target_fft * mask

    # L1 loss on magnitude of high-frequency components
    pred_mag = torch.abs(pred_fft_high)
    target_mag = torch.abs(target_fft_high)

    return torch.abs(pred_mag - target_mag).mean()


class ModelOutputWrapper(nn.Module):
    """Wrapper that extracts first element from model tuple output for transport.training_losses."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor, t: Tensor, **kwargs) -> Tensor:
        output = self.model(x, t, **kwargs)
        if isinstance(output, tuple):
            return output[0]
        return output


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
        self._initialize_lazy_modules()
        self.prepare_ema_models()

        self.fm_transport: Transport = self._build_flow_matching_transport()
        self.fm_sampler: Sampler = self._build_flow_matching_sampler(self.fm_transport)

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

    def _build_flow_matching_transport(self) -> Transport:
        return Transport(
            model_type=getattr(self.train_cfg, "model_type", "x1"),
            path_type=getattr(self.train_cfg, "path_type", "linear"),
            loss_type=getattr(self.train_cfg, "loss_type", "none"),
            train_eps=float(getattr(self.train_cfg, "train_eps", 1e-4)),
            sample_eps=float(getattr(self.train_cfg, "sample_eps", 1e-4)),
            time_sample_type=str(getattr(self.train_cfg, "time_weighting", "uniform")),
        )

    def _build_flow_matching_sampler(self, transport: Transport) -> Sampler:
        time_type = str(getattr(getattr(self.cfg, "sampler", None), "sampling_time_type", "uniform"))
        return Sampler(transport, time_type=time_type)

    def _maybe_apply_time_shift(self, t: Tensor, *, x: Tensor) -> Tensor:
        if not bool(getattr(self.train_cfg, "time_shifting", False)):
            return t
        shift_base = int(getattr(self.train_cfg, "shift_base", 4096))
        shift_dim = int(math.prod(x.shape[1:]))
        return apply_time_shift(t, shift_dim=shift_dim, shift_base=shift_base)

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

        # Wrap model for transport.training_losses (which expects tensor output)
        self.model_for_fm = ModelOutputWrapper(self.model)

    def _initialize_lazy_modules(self) -> None:
        """Initialize any LazyLinear/LazyConv2d modules with a dummy forward pass.

        This is necessary when using lazy_init_ema=True, because EMA will deepcopy
        the model on first update(), and lazy modules can't be deepcopied until initialized.
        """
        self.log_msg("[Model]: initializing lazy modules with dummy forward pass")

        # Get model's expected input channels and size
        model_unwrapped = self.accelerator.unwrap_model(self.model)
        in_channels = getattr(model_unwrapped, "in_channels", 16)
        input_size = getattr(model_unwrapped, "input_size", 64)

        # Create dummy inputs
        dummy_x = torch.randn(1, in_channels, input_size, input_size, device=self.device, dtype=self.dtype)
        dummy_t = torch.tensor([0.5], device=self.device, dtype=self.dtype)
        dummy_cond = torch.randn(1, in_channels, input_size, input_size, device=self.device, dtype=self.dtype)

        # Run a forward pass to initialize lazy modules
        with torch.no_grad():
            try:
                _ = model_unwrapped(dummy_x, dummy_t, conditions=dummy_cond)
                self.log_msg("[Model]: lazy modules initialized successfully")
            except Exception as e:
                self.log_msg(f"[Model]: failed to initialize lazy modules: {e}", level="WARNING")

    def prepare_ema_models(self) -> None:
        if self.no_ema:
            return
        # Pass the unwrapped model to EMA, because accelerator-wrapped models
        # may have hooks/buffers that can't be deepcopied
        model_unwrapped = self.accelerator.unwrap_model(self.model)
        self.ema_model: EMA = hydra.utils.instantiate(self.ema_cfg)(model_unwrapped)
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
                self.tb_logger: Unknown = self.accelerator.get_tracker("tensorboard")  # type: ignore[assignment]

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
                if "muon" in optimizer_cfg._target_.lower():
                    self.log_msg("[Optimizer]: using muon optimizer")
                    return hydra.utils.instantiate(optimizer_cfg)(named_parameters=list(self.model.named_parameters()))
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
        with torch.no_grad():
            if getattr(self, "_check_vae_recon", True):
                h_ = self.vae.encode(x)
                recon_ = self.vae.decode(h_, input_shape=x.shape[1])
                psnr_value = peak_signal_noise_ratio(self.to_rgb(recon_), self.to_rgb(x), data_range=1.0).item()
                logger.debug(
                    f"[Check VAE correctness]: PSNR {psnr_value} - latent range {(h_.min().item(), h_.max().item())}"
                )
                self._check_vae_recon = False
                return h_
            else:
                return self.vae.encode(x)

    def _decode_if_needed(self, z: Tensor, *, input_shape: torch.Size | int) -> Tensor:
        if self.vae is None:
            return z
        with torch.no_grad():
            return self.vae.decode(z, input_shape=input_shape)

    def train_cloud_removal_step(self, batch: dict[str, Any]) -> TrainCloudRemovalStepOutput:
        gt_px = batch["gt"]
        cond_px = self._extract_conditions(batch)

        gt = self._encode_if_needed(gt_px)
        conditions = self._encode_if_needed(cond_px) if cond_px is not None else None
        cond_tensor = pack_conditions(conditions)

        with self.accelerator.autocast():
            t, _, _ = self.fm_transport.sample(gt)
            t = self._maybe_apply_time_shift(t, x=gt)

            terms = self.fm_transport.training_losses(
                self.model_for_fm,
                gt,
                model_kwargs={"conditions": cond_tensor},
                t_forced=t,
            )
            fm_loss = cast(Tensor, terms["loss"]).mean()

            # Add high-frequency losses if enabled
            grad_loss_weight = float(getattr(self.train_cfg, "grad_loss_weight", 0.0))
            fft_loss_weight = float(getattr(self.train_cfg, "fft_loss_weight", 0.0))

            loss = fm_loss
            grad_loss_val = torch.tensor(0.0, device=fm_loss.device)
            fft_loss_val = torch.tensor(0.0, device=fm_loss.device)

            if grad_loss_weight > 0 or fft_loss_weight > 0:
                # Get predicted x1 from terms
                pred_x1 = cast(Tensor, terms.get("pred_x_clean", terms["pred"]))

                if grad_loss_weight > 0:
                    grad_loss_val = compute_gradient_loss(pred_x1, gt)
                    loss = loss + grad_loss_weight * grad_loss_val

                if fft_loss_weight > 0:
                    high_freq_ratio = float(getattr(self.train_cfg, "fft_high_freq_ratio", 0.3))
                    fft_loss_val = compute_fft_high_freq_loss(pred_x1, gt, high_freq_ratio=high_freq_ratio)
                    loss = loss + fft_loss_weight * fft_loss_val

        if self.accelerator.sync_gradients:
            self.optim.zero_grad()
            self.accelerator.backward(loss)
            if getattr(self.train_cfg, "max_grad_norm", None) is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.train_cfg.max_grad_norm)
            self.optim.step()
            self.sched.step()
            self.ema_update()

        log_losses = {
            "loss/total": loss.detach(),
            "loss/fm": fm_loss.detach(),
            "loss/grad": grad_loss_val.detach(),
            "loss/fft": fft_loss_val.detach(),
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
        # Convert OmegaConf ListConfig to native Python list for beartype compatibility
        if rgb_channels is not None and not isinstance(rgb_channels, (list, str)):
            rgb_channels = list(rgb_channels)

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
            # Ensure parent directory exists (img_name may contain subdirs like "val/...")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            from PIL import Image

            Image.fromarray(img).save(save_path, quality=95)
            self.log_msg(f"[Visualize]: save visualization at {save_path}")

    def general_sampler(
        self,
        model: nn.Module,
        shape: torch.Size,
        conditions: Tensor | list[Tensor] | tuple[Tensor, ...] | None,
        sampling_type: str | None = None,
    ) -> Tensor:
        sampler_cfg = getattr(self.cfg, "sampler", None)
        if sampling_type is None:
            if sampler_cfg is None:
                sampling_type = "ode"
            else:
                sampling_type = str(getattr(sampler_cfg, "type", "ode"))

        if sampler_cfg is None:
            # Fallback to defaults if not present
            num_steps = 25
            cfg_scale = 1.0
            time_shifting = bool(getattr(self.train_cfg, "time_shifting", False))
            shift_base = int(getattr(self.train_cfg, "shift_base", 4096))
            sampling_time_type = "uniform"
            progress = False
            clip_for_x1_pred = False
        else:
            num_steps = sampler_cfg.num_steps
            cfg_scale = sampler_cfg.cfg_scale
            time_shifting = bool(getattr(sampler_cfg, "time_shifting", False))
            shift_base = int(getattr(sampler_cfg, "shift_base", 4096))
            sampling_time_type = str(getattr(sampler_cfg, "sampling_time_type", "uniform"))
            progress = bool(getattr(sampler_cfg, "progress", False))
            clip_for_x1_pred = bool(getattr(sampler_cfg, "clip_for_x1_pred", False))

        x0 = torch.randn(shape, device=self.device, dtype=self.dtype)
        cond_tensor = pack_conditions(conditions)
        do_cfg = cfg_scale > 1.0

        # Wrap model for sampling
        model_wrapped = ModelOutputWrapper(model)

        def _model_forward(x_in: Tensor, t_vec: Tensor, cond: Tensor | None, uncond: bool) -> Tensor:
            if time_shifting:
                shift_dim = int(math.prod(shape[1:]))
                t_vec = apply_time_shift(t_vec, shift_dim=shift_dim, shift_base=shift_base)
            return model_wrapped(x_in, t_vec, conditions=cond, uncond=uncond)

        def _model_fn(x_t: Tensor, t_vec: Tensor, **model_kwargs: Any) -> Tensor:
            conditions = cast(Tensor | None, model_kwargs.get("conditions"))
            if do_cfg:
                x1_cond = _model_forward(x_in=x_t, t_vec=t_vec, cond=conditions, uncond=False)
                x1_uncond = _model_forward(x_in=x_t, t_vec=t_vec, cond=conditions, uncond=True)
                return x1_uncond + cfg_scale * (x1_cond - x1_uncond)
            return _model_forward(x_in=x_t, t_vec=t_vec, cond=conditions, uncond=False)

        if sampling_type == "ode":
            sample_fn = self.fm_sampler.sample_ode(
                num_steps=num_steps,
                sampling_time_type=sampling_time_type,
                progress=progress,
                clip_for_x1_pred=clip_for_x1_pred,
            )
        elif sampling_type == "sde":
            sde_kwargs = {}
            if sampler_cfg is not None:
                sde_kwargs = {
                    "sampling_method": str(getattr(sampler_cfg, "sde_sampling_method", "Euler")),
                    "diffusion_form": str(getattr(sampler_cfg, "sde_diffusion_form", "SBDM")),
                    "diffusion_norm": float(getattr(sampler_cfg, "sde_diffusion_norm", 1.0)),
                    "last_step": str(getattr(sampler_cfg, "sde_last_step", "Mean")),
                    "last_step_size": float(getattr(sampler_cfg, "sde_last_step_size", 0.04)),
                    "temperature": float(getattr(sampler_cfg, "temperature", 1.0)),
                }

            sample_fn = self.fm_sampler.sample_sde(num_steps=num_steps, **sde_kwargs)
        else:
            raise ValueError(f"Unknown sampling type: {sampling_type}")

        samples = sample_fn(x0, _model_fn, conditions=cond_tensor)
        return cast(Tensor, samples[-1])

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
                gt = cast(Tensor, self._encode_if_needed(gt_px))
                conditions = self._encode_if_needed(cond_px) if cond_px is not None else None
                cond_tensor = pack_conditions(conditions)
                t, _, _ = self.fm_transport.sample(gt)
                t = self._maybe_apply_time_shift(t, x=gt)

                # Wrap eval model for transport.training_losses
                model_for_fm_eval = ModelOutputWrapper(model_for_vis)
                terms = self.fm_transport.training_losses(
                    model_for_fm_eval,
                    gt,
                    model_kwargs={"conditions": cond_tensor},
                    t_forced=t,
                )
                pred_latent = cast(Tensor, terms.get("pred_x_clean", terms["pred"]))
                pred_px = self._decode_if_needed(pred_latent, input_shape=gt_px.shape)

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

        psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

        last_batch: dict[str, Any] | None = None
        last_pred_px: Tensor | None = None
        for batch in self.get_val_loader_iter():
            batch = self.to_device_dtype(batch)
            gt_px = batch["gt"]
            cond_px = self._extract_conditions(batch)

            with self.accelerator.autocast():
                gt_latent = self._encode_if_needed(gt_px)
                conditions = self._encode_if_needed(cond_px) if cond_px is not None else None
                pred_latent = self.general_sampler(model_for_eval, gt_latent.shape, conditions=conditions)
                pred_px = self._decode_if_needed(pred_latent, input_shape=gt_px.shape)

            # torchmetrics expects float tensors; we normalize/clamp into [0, 1]
            pred_metric = self.to_rgb(pred_px).float()
            gt_metric = self.to_rgb(gt_px).float()
            psnr_metric.update(pred_metric, gt_metric)  # type: ignore
            ssim_metric.update(pred_metric, gt_metric)  # type: ignore

            self.step_train_state("val")
            last_batch = batch
            last_pred_px = pred_px

        psnr_val = psnr_metric.compute().detach()  # type: ignore
        ssim_val = ssim_metric.compute().detach()  # type: ignore
        psnr_val = self.accelerator.gather(psnr_val).mean()
        ssim_val = self.accelerator.gather(ssim_val).mean()

        if self.accelerator.is_main_process:
            self.log_msg(f"[Val]: psnr={psnr_val.item():.4f} | ssim={ssim_val.item():.4f}")
            self.tenb_log_any(
                "metric",
                {
                    "val/psnr": float(psnr_val.item()),
                    "val/ssim": float(ssim_val.item()),
                },
                step=self.global_step,
            )

        if last_batch is not None and last_pred_px is not None and self.accelerator.is_main_process:
            gt_px = last_batch["gt"]
            cloudy_px = last_batch.get("img", gt_px)
            self.visualize_triplet(
                cloudy=cloudy_px,
                pred=last_pred_px,
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


_key = "cuhk_cr1"
_configs_dict = {
    "cuhk_cr1": "cloud_removal_cuhk_cr1",
    "sen12": "cloud_removal_sen12",
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

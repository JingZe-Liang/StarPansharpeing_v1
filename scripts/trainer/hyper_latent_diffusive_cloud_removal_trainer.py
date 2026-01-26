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
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
from tqdm import tqdm, trange

from src.utilities.logging import configure_logger, log
from src.stage2.cloud_removal.data.sen12_vis import s1_to_gray_for_display
from src.stage2.cloud_removal.diffusion.loss import apply_time_shift, pack_conditions
from src.stage2.cloud_removal.diffusion.vae import CosmosRSVAE, FluxVAE
from src.stage2.cloud_removal.metrics.basic import CRMetrics
from src.utilities.logging import dict_round_to_list_str
from src.utilities.transport.SDB.plan import DiffusionTarget, SDBContinuousPlan, SDBContinuousSampler
from src.utilities.transport.flow_matching import Sampler, Transport
from src.utilities.transport.I2SB.diffusion import Diffusion as I2SBDiffusion

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


def compute_ssim_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Compute SSIM loss (1 - SSIM).

    Args:
        pred: Predicted tensor [B, C, H, W]
        target: Target tensor [B, C, H, W]

    Returns:
        SSIM loss value (1 - SSIM, so lower is better)
    """
    ssim_val = structural_similarity_index_measure(pred, target, data_range=1.0)
    if isinstance(ssim_val, tuple):
        ssim_val = ssim_val[0]
    return 1.0 - ssim_val


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


class SDBModelAdapter(nn.Module):
    """Adapter to ensure SDB timesteps are compatible with models expecting (B,) timesteps."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor, t: Tensor, **kwargs) -> Tensor | tuple[Tensor, ...]:
        if t.ndim != 1:
            t = t.reshape(t.shape[0], -1)[:, 0]
        return self.model(x, t, **kwargs)


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
        self.train_mode = self._get_train_mode()
        self.transport_backend = self._get_transport_backend() if self.train_mode == "diffusion" else None

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
        self.log_msg(f"[Train]: mode={self.train_mode} transport={self.transport_backend}")

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
        logger.info(f"Train dataset length: {len(self.train_dataset)}")
        logger.info(f"Val dataset length: {len(self.val_dataset)}")

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

        self.fm_transport: Transport | None = None
        self.fm_sampler: Sampler | None = None
        self.sdb_plan: SDBContinuousPlan | None = None
        self.sdb_sampler: SDBContinuousSampler | None = None
        self.sdb_model: nn.Module | None = None
        self.i2sb_diffusion: I2SBDiffusion | None = None
        if self.train_mode == "diffusion":
            if self.transport_backend == "flow_matching":
                self.fm_transport = self._build_flow_matching_transport()
                self.fm_sampler = self._build_flow_matching_sampler(self.fm_transport)
            elif self.transport_backend == "sdb":
                self.sdb_plan = self._build_sdb_plan()
                self.sdb_sampler = self._build_sdb_sampler(self.sdb_plan)
                self.sdb_model = self._build_sdb_precond(self.model)
            else:
                self.i2sb_diffusion = self._build_i2sb_diffusion()

        interp_to = getattr(self.val_cfg, "metrics_interp_to", None)
        self.cr_metric = CRMetrics(lpips_device=self.device, interp_to=interp_to).to(self.device)

        self.train_state = StepsCounter(["train", "val"])

    def _build_vae(self) -> CosmosRSVAE | None:
        if not hasattr(self.cfg, "vae") or self.cfg.vae is None:
            self.log_msg("[VAE]: disabled (cfg.vae not set)")
            return None

        vae = hydra.utils.instantiate(self.cfg.vae)
        if not isinstance(vae, (CosmosRSVAE, FluxVAE)):
            raise TypeError(f"cfg.vae must be CosmosRSVAE or FluxVAEfor now, got {type(vae)}")
        vae = vae.to(device=self.device)
        vae.eval().requires_grad_(False)
        self.log_msg(f"[VAE]: loaded {vae.__class__.__name__}")

        self._vae_only_support_rgb = False
        if isinstance(vae, FluxVAE):
            self._vae_only_support_rgb = True
            logger.warning(f"Using {vae.__class__.__name__}, only supports RGB image.")

        return vae

    def _build_flow_matching_transport(self) -> Transport:
        return Transport(
            model_type=getattr(self.train_cfg, "model_type", "x1"),
            path_type=getattr(self.train_cfg, "path_type", "linear"),
            loss_type=getattr(self.train_cfg, "loss_type", "none"),
            train_eps=getattr(self.train_cfg, "train_eps", None),
            sample_eps=getattr(self.train_cfg, "sample_eps", None),
            time_sample_type=str(getattr(self.train_cfg, "time_weighting", "uniform")),
        )

    def _build_flow_matching_sampler(self, transport: Transport) -> Sampler:
        time_type = str(getattr(getattr(self.cfg, "sampler", None), "sampling_time_type", "uniform"))
        return Sampler(transport, time_type=time_type)

    def _get_transport_backend(self) -> Literal["flow_matching", "sdb", "i2sb"]:
        raw_backend = str(getattr(self.train_cfg, "transport_backend", "flow_matching")).lower()
        if raw_backend in {"flow_matching", "flow-matching", "fm"}:
            return "flow_matching"
        if raw_backend in {"sdb", "sdb-transport", "sdb_transport"}:
            return "sdb"
        if raw_backend in {"i2sb"}:
            return "i2sb"
        raise ValueError(f"Unknown transport_backend: {raw_backend}")

    def _build_sdb_plan(self) -> SDBContinuousPlan:
        sdb_cfg = getattr(self.train_cfg, "sdb", None)
        plan_tgt = DiffusionTarget.x_0
        gamma_max = 0.5
        eps_eta = 1.0
        alpha_beta_type = "linear"
        diffusion_type = "bridge"
        t_train_type = "edm"
        t_sample_type = "edm"
        t_train_kwargs: dict[str, Any] = {"device": self.device, "clip_t_min_max": (1e-4, 1 - 1e-4)}

        if sdb_cfg is not None:
            plan_tgt = DiffusionTarget(str(getattr(sdb_cfg, "plan_tgt", plan_tgt.value)))
            gamma_max = float(getattr(sdb_cfg, "gamma_max", gamma_max))
            eps_eta = float(getattr(sdb_cfg, "eps_eta", eps_eta))
            alpha_beta_type = str(getattr(sdb_cfg, "alpha_beta_type", alpha_beta_type))
            diffusion_type = str(getattr(sdb_cfg, "diffusion_type", diffusion_type))
            t_train_type = str(getattr(sdb_cfg, "t_train_type", t_train_type))
            t_sample_type = str(getattr(sdb_cfg, "t_sample_type", t_sample_type))
            t_train_kwargs_raw = getattr(sdb_cfg, "t_train_kwargs", None)
            if t_train_kwargs_raw is not None:
                t_train_kwargs = OmegaConf.to_container(t_train_kwargs_raw, resolve=True)  # type: ignore[assignment]
                if not isinstance(t_train_kwargs, dict):
                    raise TypeError("train.sdb.t_train_kwargs must be a mapping")

        clip_range = t_train_kwargs.get("clip_t_min_max")
        if clip_range is not None:
            t_train_kwargs["clip_t_min_max"] = (float(clip_range[0]), float(clip_range[1]))
        t_train_kwargs["device"] = self.device

        return SDBContinuousPlan(
            plan_tgt=plan_tgt,
            gamma_max=gamma_max,
            eps_eta=eps_eta,
            alpha_beta_type=alpha_beta_type,
            diffusion_type=diffusion_type,
            t_train_type=t_train_type,
            t_train_kwargs=t_train_kwargs,
            t_sample_type=t_sample_type,
        )

    def _build_sdb_sampler(self, plan: SDBContinuousPlan) -> SDBContinuousSampler:
        sdb_cfg = getattr(self.train_cfg, "sdb", None)
        sample_noisy_x1_b = float(getattr(sdb_cfg, "sample_noisy_x1_b", 0.0)) if sdb_cfg is not None else 0.0
        return SDBContinuousSampler(plan, sample_noisy_x1_b=sample_noisy_x1_b)

    def _build_i2sb_diffusion(self) -> I2SBDiffusion:
        i2sb_cfg = getattr(self.train_cfg, "i2sb", None)
        if i2sb_cfg is None:
            raise ValueError("I2SB training requires i2sb config")

        n_timestep = int(getattr(i2sb_cfg, "n_timestep", 1000))
        linear_start = float(getattr(i2sb_cfg, "linear_start", 1e-4))
        linear_end = getattr(i2sb_cfg, "linear_end", None)
        beta_max = getattr(i2sb_cfg, "beta_max", 0.3)
        model_pred = str(getattr(i2sb_cfg, "model_pred", "x0"))
        ot_ode = bool(getattr(i2sb_cfg, "ot_ode", False))

        diffusion = I2SBDiffusion(
            n_timestep=n_timestep,
            linear_start=linear_start,
            linear_end=linear_end,
            beta_max=beta_max,
            model_pred=model_pred,
            ot_ode=ot_ode,
        )
        return diffusion.to(self.device)

    def _build_sdb_precond(self, model: nn.Module) -> nn.Module:
        """Build SDB preconditioner wrapper around model."""
        from src.utilities.transport.SDB.precond import EDMPrecond, EDMPrecondRawT

        if self.sdb_plan is None:
            raise RuntimeError("SDB plan must be initialized before building preconditioner")

        sdb_cfg = getattr(self.train_cfg, "sdb", None)
        precond_type = str(getattr(sdb_cfg, "precond", "none")).lower() if sdb_cfg is not None else "none"

        # null is treated as none
        if precond_type == "null":
            precond_type = "none"

        if precond_type == "none":
            # No preconditioning, just wrap in SDBModelAdapter
            return SDBModelAdapter(model)
        elif precond_type == "edm":
            # EDM preconditioning
            # First wrap model in SDBModelAdapter to handle time dimensions
            adapted_model = SDBModelAdapter(model)
            # Then wrap in EDMPrecond
            return EDMPrecond(model=adapted_model, plan=self.sdb_plan)
        elif precond_type in {"edm_raw_t", "edm_raw", "edm_t"}:
            adapted_model = SDBModelAdapter(model)
            return EDMPrecondRawT(model=adapted_model, plan=self.sdb_plan)
        else:
            raise ValueError(f"Unknown precond type: {precond_type}. Expected 'none', 'edm', or 'edm_raw_t'.")

    def _build_sdb_time_grid(self, num_steps: int, *, device: torch.device) -> Tensor:
        if self.sdb_plan is None:
            raise RuntimeError("SDB plan is not initialized.")
        sdb_cfg = getattr(self.train_cfg, "sdb", None)
        sample_kwargs: dict[str, Any] = {}
        if sdb_cfg is not None:
            sample_kwargs_raw = getattr(sdb_cfg, "t_sample_kwargs", None)
            if sample_kwargs_raw is not None:
                sample_kwargs = OmegaConf.to_container(sample_kwargs_raw, resolve=True)  # type: ignore[assignment]
                if not isinstance(sample_kwargs, dict):
                    raise TypeError("train.sdb.t_sample_kwargs must be a mapping")

        sample_kwargs["n_timesteps"] = num_steps
        sample_kwargs.setdefault("t_min", self.sdb_plan.t_min)
        sample_kwargs.setdefault("t_max", self.sdb_plan.t_max)
        if self.sdb_plan.t_sample_type == "edm":
            sample_kwargs.setdefault("rho", 0.3)
        if self.sdb_plan.t_sample_type == "sigmoid":
            sample_kwargs.setdefault("k", 7.0)

        time_grid = self.sdb_plan.sample_continous_t(**sample_kwargs)
        return time_grid.to(device=device, dtype=torch.float32)

    def _get_sdb_x1(self, batch: dict[str, Any], *, like: Tensor | None = None) -> Tensor:
        x1_source = str(getattr(self.train_cfg, "sdb_x1_source", "img")).lower()
        if x1_source in {"img", "cloud"}:
            x1_px = batch.get("img", None)
        elif x1_source == "conditions":
            x1_px = batch.get("conditions", None)
        else:
            raise ValueError(f"Unknown sdb_x1_source: {x1_source}")

        if x1_px is None or not torch.is_tensor(x1_px):
            raise KeyError(f"SDB x1_source='{x1_source}' expects a tensor in the batch.")

        x1_latent = self._encode_if_needed(x1_px)
        if not torch.is_tensor(x1_latent):
            raise TypeError("Encoded SDB x1 must be a tensor.")
        if like is not None:
            x1_latent = x1_latent.to(like)
        return x1_latent

    def _get_i2sb_x1(self, batch: dict[str, Any], *, like: Tensor | None = None) -> Tensor:
        x1_source = str(getattr(self.train_cfg, "i2sb_x1_source", "img")).lower()
        if x1_source in {"img", "cloud"}:
            x1_px = batch.get("img", None)
        elif x1_source == "conditions":
            x1_px = batch.get("conditions", None)
        else:
            raise ValueError(f"Unknown i2sb_x1_source: {x1_source}")

        if x1_px is None or not torch.is_tensor(x1_px):
            raise KeyError(f"I2SB x1_source='{x1_source}' expects a tensor in the batch.")

        x1_latent = self._encode_if_needed(x1_px)
        if not torch.is_tensor(x1_latent):
            raise TypeError("Encoded I2SB x1 must be a tensor.")
        if like is not None:
            x1_latent = x1_latent.to(like)
        return x1_latent

    def _get_sdb_model(self, model: nn.Module) -> nn.Module:
        from src.utilities.transport.SDB.precond import EDMPrecond, EDMPrecondRawT

        if isinstance(model, (SDBModelAdapter, EDMPrecond, EDMPrecondRawT)):
            return model
        if self.sdb_plan is None:
            return SDBModelAdapter(model)

        sdb_cfg = getattr(self.train_cfg, "sdb", None)
        precond_type = str(getattr(sdb_cfg, "precond", "none")).lower() if sdb_cfg is not None else "none"
        if precond_type in {"none", "null"}:
            return SDBModelAdapter(model)

        if model is self.model and self.sdb_model is not None:
            return self.sdb_model
        return self._build_sdb_precond(model)

    def _get_i2sb_model(self, model: nn.Module) -> nn.Module:
        if isinstance(model, ModelOutputWrapper):
            return model
        return ModelOutputWrapper(model)

    def _sdb_model_forward(
        self,
        *,
        model: nn.Module,
        x_t: Tensor,
        t: Tensor,
        conditions: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """
        Forward pass for SDB model with optional preconditioning.

        Returns:
            tuple: (pred_tgt, weight) where weight is 1.0 for no precond, or computed weight for EDM precond
        """
        model_wrapped = self._get_sdb_model(model)
        x_t = self._to_model_device_dtype(x_t, model_wrapped)
        t = self._to_model_device_dtype(t, model_wrapped)
        cond_tensor = self._to_model_device_dtype(conditions, model_wrapped) if conditions is not None else None
        model_out = model_wrapped(x_t, t, conditions=cond_tensor)

        # Handle different output formats
        if isinstance(model_out, tuple):
            if len(model_out) == 3:
                # EDMPrecond returns (pred_x_0, aux_loss, weight)
                pred_tgt, aux_loss, weight = model_out
                return pred_tgt, weight
            elif len(model_out) == 2:
                # Regular model might return (output, aux_loss)
                pred_tgt, aux_loss = model_out
                weight = torch.ones_like(pred_tgt[..., 0:1, 0:1, 0:1])  # Shape: (B, 1, 1, 1)
                return pred_tgt, weight
            else:
                raise ValueError(f"Unexpected model output tuple length: {len(model_out)}")
        else:
            # Single tensor output
            weight = torch.tensor(1.0).to(x_t)  # Shape: (B, 1, 1, 1)
            return model_out, weight

    def _sdb_pred_to_x0(self, pred: Tensor, *, t: Tensor, x_t: Tensor, x_1: Tensor) -> Tensor:
        if self.sdb_plan is None:
            raise RuntimeError("SDB plan is not initialized.")
        if self.sdb_plan.plan_tgt == DiffusionTarget.x_0:
            return pred
        if self.sdb_plan.plan_tgt == DiffusionTarget.score:
            return self.sdb_plan.get_x0_from_score(pred, t, x_t, x_1)
        if self.sdb_plan.plan_tgt == DiffusionTarget.noise:
            score = self.sdb_plan.get_score_from_noise(pred, t)
            return self.sdb_plan.get_x0_from_score(score, t, x_t, x_1)
        raise ValueError(f"Unsupported SDB plan_tgt: {self.sdb_plan.plan_tgt}")

    def _compute_sdb_loss(self, pred: Tensor, target: Tensor, weight: Tensor | None = None) -> Tensor:
        loss_type = str(getattr(self.train_cfg, "sdb_loss_type", "mse")).lower()
        if loss_type in {"mse", "l2"}:
            loss_val = torch.nn.functional.mse_loss(pred, target, reduction="none")
        elif loss_type in {"l1", "mae"}:
            loss_val = torch.nn.functional.l1_loss(pred, target, reduction="none")
        else:
            raise ValueError(f"Unknown sdb_loss_type: {loss_type}")

        # Apply weight if provided (from EDM precond)
        if weight is not None:
            loss_val = loss_val * weight

        # Reduce to scalar
        loss_val = loss_val.mean()

        loss_weight = float(getattr(self.train_cfg, "sdb_loss_weight", 1.0))
        return loss_val * loss_weight

    def _sdb_sample_latent(
        self,
        *,
        model: nn.Module,
        x_1: Tensor,
        conditions: Tensor | list[Tensor] | tuple[Tensor, ...] | None,
        sampling_type: str | None = None,
    ) -> Tensor:
        if self.sdb_plan is None or self.sdb_sampler is None:
            raise RuntimeError("SDB sampler is not initialized.")

        sampler_cfg = getattr(self.cfg, "sampler", None)
        if sampling_type is None:
            if sampler_cfg is None:
                sampling_type = "ode"
            else:
                sampling_type = str(getattr(sampler_cfg, "type", "ode"))

        if sampler_cfg is None:
            num_steps = 25
            progress = False
            clip_value = False
        else:
            num_steps = sampler_cfg.num_steps
            progress = bool(getattr(sampler_cfg, "progress", False))
            clip_value = bool(getattr(sampler_cfg, "clip_for_x1_pred", False))

        # When sampling in latent space (VAE enabled), clipping to [-1, 1] often collapses
        # latents into a near-constant decode (e.g., all black). Only clip in pixel space.
        if self.vae is not None:
            clip_value = False

        time_grid = self._build_sdb_time_grid(num_steps, device=x_1.device)
        cond_tensor = pack_conditions(conditions)
        model_kwargs: dict[str, Any] = {}
        if cond_tensor is not None:
            model_kwargs["conditions"] = cond_tensor

        model_wrapped = self._get_sdb_model(model)
        if sampling_type == "sde":
            samples, _, _ = self.sdb_sampler.sample_sde_euler(
                model=model_wrapped,
                x_1=x_1,
                time_grid=time_grid,
                model_kwargs=model_kwargs,
                clip_value=clip_value,
                progress=progress,
            )
            return samples
        if sampling_type == "ode":
            samples, _, _ = self.sdb_sampler.sample_ode_euler(
                model=model_wrapped,
                x_1=x_1,
                time_grid=time_grid,
                model_kwargs=model_kwargs,
                clip_value=clip_value,
                progress=progress,
            )
            return samples
        raise ValueError(f"Unknown sampling type: {sampling_type}")

    def _i2sb_sample_latent(
        self,
        *,
        model: nn.Module,
        x_1: Tensor,
        conditions: Tensor | list[Tensor] | tuple[Tensor, ...] | None,
        sampling_type: str | None = None,
    ) -> Tensor:
        if self.i2sb_diffusion is None:
            raise RuntimeError("I2SB diffusion is not initialized.")

        sampler_cfg = getattr(self.cfg, "sampler", None)
        if sampling_type is None:
            sampling_type = "ddpm"
        if sampling_type not in {"ddpm", "ode", "sde"}:
            raise ValueError(f"Unknown I2SB sampling type: {sampling_type}")

        if sampler_cfg is None:
            num_steps = 25
            progress = False
            clip_value = False
        else:
            num_steps = sampler_cfg.num_steps
            progress = bool(getattr(sampler_cfg, "progress", False))
            clip_value = bool(getattr(sampler_cfg, "clip_for_x1_pred", False))

        if num_steps >= self.i2sb_diffusion.n_timestep:
            raise ValueError("I2SB num_steps must be smaller than i2sb.n_timestep.")

        model_wrapped = self._get_i2sb_model(model)
        x_1 = self._to_model_device_dtype(x_1, model_wrapped)
        cond_tensor = pack_conditions(conditions)
        cond_tensor = self._to_model_device_dtype(cond_tensor, model_wrapped)

        sample_out = self.i2sb_diffusion.sample(
            model_wrapped,
            x_1,
            clip_denoise=clip_value,
            nfe=int(num_steps),
            model_kwargs={"conditions": cond_tensor},
            ot_ode=False,
            log_count=1,
            verbose=progress,
        )
        sampled = cast(Tensor, sample_out["sampled"])
        return sampled[:, -1]

    def _get_train_mode(self) -> Literal["diffusion", "regression"]:
        raw_mode = str(getattr(self.train_cfg, "train_mode", "diffusion")).lower()
        if raw_mode in {"diffusion", "flow_matching", "flow-matching", "fm", "sdb"}:
            return "diffusion"
        if raw_mode in {"regression", "direct"}:
            return "regression"
        raise ValueError(f"Unknown train_mode: {raw_mode}")

    def _maybe_apply_time_shift(self, t: Tensor, *, x: Tensor) -> Tensor:
        if not bool(getattr(self.train_cfg, "time_shifting", False)):
            return t
        shift_base = int(getattr(self.train_cfg, "shift_base", 4096))
        shift_dim = int(math.prod(x.shape[1:]))
        return apply_time_shift(t, shift_dim=shift_dim, shift_base=shift_base)

    def _get_regression_input_px(self, batch: dict[str, Any]) -> Tensor:
        regression_px = batch.get("img")
        if regression_px is None or not torch.is_tensor(regression_px):
            raise KeyError("Regression mode expects batch['img'] as a tensor input.")
        return regression_px

    def _forward_regression_latent(
        self,
        *,
        model: nn.Module,
        regression_input: Tensor,
        conditions: Tensor | list[Tensor] | tuple[Tensor, ...] | None,
    ) -> Tensor:
        cond_tensor = pack_conditions(conditions)
        model_wrapped = model if isinstance(model, ModelOutputWrapper) else ModelOutputWrapper(model)
        regression_input = self._to_model_device_dtype(regression_input, model_wrapped)
        cond_tensor = self._to_model_device_dtype(cond_tensor, model_wrapped)
        if regression_input is None:
            raise TypeError("Regression input must be a tensor.")
        return model_wrapped(regression_input, None, conditions=cond_tensor)

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

        # print infos
        import fvcore.nn as fnn

        print(fnn.parameter_count_table(self.model))

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

    def _get_model_device_dtype(self, model: nn.Module) -> tuple[torch.device, torch.dtype]:
        for param in model.parameters():
            return param.device, param.dtype
        for buffer in model.buffers():
            return buffer.device, buffer.dtype
        return self.device, self.dtype

    @overload
    def _to_model_device_dtype(self, x: Tensor, model: nn.Module) -> Tensor: ...

    @overload
    def _to_model_device_dtype(self, x: None, model: nn.Module) -> None: ...

    def _to_model_device_dtype(self, x: Tensor | None, model: nn.Module) -> Tensor | None:
        if x is None:
            return None
        device, dtype = self._get_model_device_dtype(model)
        return x.to(device=device, dtype=dtype)

    def _get_eval_model(self) -> nn.Module:
        if self.no_ema or not hasattr(self, "ema_model"):
            return self.model
        return cast(nn.Module, self.ema_model.ema_model)

    def _to_condition_list(self, conditions: Tensor | list[Tensor] | tuple[Tensor, ...]) -> list[Tensor]:
        if torch.is_tensor(conditions):
            return [conditions]
        return list(conditions)

    def _normalize_conditions(
        self, conditions: Tensor | list[Tensor] | tuple[Tensor, ...] | None
    ) -> Tensor | list[Tensor] | None:
        if conditions is None or torch.is_tensor(conditions):
            return conditions
        return list(conditions)

    def _merge_conditions(
        self,
        primary: Tensor | list[Tensor] | tuple[Tensor, ...] | None,
        extra: Tensor | list[Tensor] | tuple[Tensor, ...] | None,
    ) -> Tensor | list[Tensor] | None:
        if primary is None:
            return self._normalize_conditions(extra)
        if extra is None:
            return self._normalize_conditions(primary)
        primary_list = self._to_condition_list(primary)
        extra_list = self._to_condition_list(extra)
        return primary_list + extra_list

    def _extract_conditions(self, batch: dict[str, Any]) -> Tensor | list[Tensor] | None:
        cond_source = str(getattr(self.train_cfg, "condition_source", "auto")).lower()
        if cond_source in {"auto", "img"}:
            if self.vae is not None:
                return batch.get("img", None)
            if "conditions" in batch:
                return batch["conditions"]
            return batch.get("img", None)
        if cond_source == "conditions":
            return batch.get("conditions", batch.get("img", None))
        if cond_source in {"conditions_with_img", "img_with_conditions"}:
            return self._merge_conditions(batch.get("img", None), batch.get("conditions", None))
        raise ValueError(f"Unknown condition_source: {cond_source}")

    def _get_flow_matching_x0(self, batch: dict[str, Any]) -> Tensor | None:
        x0_source = str(getattr(self.train_cfg, "x0_source", "noise")).lower()
        if x0_source in {"noise", "random", "gaussian"}:
            return None
        if x0_source in {"cloud", "img"}:
            cloud_px = batch.get("img", None)
            if cloud_px is None or not torch.is_tensor(cloud_px):
                raise KeyError("Flow matching x0_source='cloud' expects batch['img'] as a tensor.")
            x0_latent = self._encode_if_needed(cloud_px)
            if not torch.is_tensor(x0_latent):
                raise TypeError("Encoded x0 must be a tensor.")
            return x0_latent
        raise ValueError(f"Unknown x0_source: {x0_source}")

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

        # If VAE only supports RGB (e.g., Flux VAE), crop to first 3 channels
        x_vae = x
        if getattr(self, "_vae_only_support_rgb", False) and x.shape[1] > 3:
            x_vae = x[:, :3]  # only use RGB channels
            # logger.debug(f"[VAE Encode]: cropped {x.shape[1]} channels to RGB (3 channels) for RGB-only VAE")

        with torch.no_grad():
            if getattr(self, "_check_vae_recon", True):
                h_ = self.vae.encode(x_vae)
                recon_ = self.vae.decode(h_, input_shape=x_vae.shape[1]).to(x_vae)
                psnr_value = peak_signal_noise_ratio(self.to_rgb(recon_), self.to_rgb(x_vae), data_range=1.0).item()
                logger.debug(
                    f"[Check VAE correctness]: PSNR {psnr_value} - latent range {(h_.min().item(), h_.max().item())}"
                )
                self._check_vae_recon = False
                return h_
            else:
                return self.vae.encode(x_vae).to(dtype=self.dtype)

    def _decode_if_needed(self, z: Tensor, *, input_shape: torch.Size | int) -> Tensor:
        if self.vae is None:
            return z
        with torch.no_grad():
            return self.vae.decode(z, input_shape=input_shape).to(dtype=self.dtype)

    def _get_preview_batch(self) -> dict[str, Any] | None:
        try:
            batch = next(iter(self.train_dataloader))
        except StopIteration:
            return None
        return self.to_device_dtype(batch)

    def _save_vae_recon_preview(self, *, only_vis_n: int = 8) -> None:
        if self.vae is None:
            self.log_msg("[VAE]: skip recon preview (vae disabled)")
            return
        if not self.accelerator.is_main_process:
            return

        batch = self._get_preview_batch()
        if batch is None:
            self.log_msg("[VAE]: skip recon preview (empty train dataloader)", level="WARNING")
            return

        img_px = batch.get("img")
        gt_px = batch.get("gt")
        if not torch.is_tensor(img_px) or not torch.is_tensor(gt_px):
            self.log_msg("[VAE]: skip recon preview (img/gt missing or not tensor)", level="WARNING")
            return

        if img_px.ndim != 4 or gt_px.ndim != 4:
            self.log_msg("[VAE]: skip recon preview (img/gt must be 4D tensors)", level="WARNING")
            return

        with torch.no_grad(), self.accelerator.autocast():
            # Crop to RGB if VAE only supports RGB
            img_vae = img_px[:, :3] if getattr(self, "_vae_only_support_rgb", False) and img_px.shape[1] > 3 else img_px
            gt_vae = gt_px[:, :3] if getattr(self, "_vae_only_support_rgb", False) and gt_px.shape[1] > 3 else gt_px

            img_latent = self.vae.encode(img_vae).to(dtype=self.dtype)
            gt_latent = self.vae.encode(gt_vae).to(dtype=self.dtype)
            recon_img = self.vae.decode(img_latent, input_shape=img_vae.shape[1]).to(dtype=self.dtype)
            recon_gt = self.vae.decode(gt_latent, input_shape=gt_vae.shape[1]).to(dtype=self.dtype)

        rgb_channels = getattr(self.train_cfg, "visualize_rgb_channels", None)
        if rgb_channels is not None and not isinstance(rgb_channels, (list, str)):
            rgb_channels = list(rgb_channels)

        def to_uint8_grid(x: Tensor) -> np.ndarray:
            if x.shape[1] == 2:
                x_vis = []
                for i in range(min(len(x), only_vis_n)):
                    gray_hw = s1_to_gray_for_display(
                        x[i],
                        channel="mean",
                        domain="db10",
                        smooth="median",
                        kernel_size=3,
                        linear_compress="log1p",
                    )
                    rgb_chw = gray_hw.unsqueeze(0).repeat(3, 1, 1)
                    x_vis.append(rgb_chw)
                x_disp = torch.stack(x_vis)
                grid = visualize_hyperspectral_image(
                    x_disp.detach().cpu(),
                    rgb_channels=[0, 1, 2],
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

        img = np.concatenate(
            [
                to_uint8_grid(img_px),
                to_uint8_grid(recon_img),
                to_uint8_grid(gt_px),
                to_uint8_grid(recon_gt),
            ],
            axis=1,
        )

        save_dir = Path(self.proj_dir) / "vis" / "vae_check"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "train_start.webp"
        from PIL import Image

        Image.fromarray(img).save(save_path, quality=95)
        self.log_msg(f"[VAE]: save recon preview at {save_path}")

    def train_cloud_removal_step(self, batch: dict[str, Any]) -> TrainCloudRemovalStepOutput:
        gt_px = batch["gt"]
        cond_px = self._extract_conditions(batch)

        gt = self._encode_if_needed(gt_px)
        if not torch.is_tensor(gt):
            raise TypeError("Encoded ground-truth must be a tensor.")
        conditions = self._encode_if_needed(cond_px) if cond_px is not None else None

        with self.accelerator.autocast():
            if self.train_mode == "diffusion":
                cond_tensor = pack_conditions(conditions)
                if self.transport_backend == "flow_matching":
                    if self.fm_transport is None:
                        raise RuntimeError("Flow matching transport is not initialized.")
                    x0_forced = self._get_flow_matching_x0(batch)
                    t, _, _ = self.fm_transport.sample(gt)
                    t = self._maybe_apply_time_shift(t, x=gt)

                    terms = self.fm_transport.training_losses(
                        self.model_for_fm,
                        gt,
                        model_kwargs={"conditions": cond_tensor},
                        t_forced=t,
                        x0_forced=x0_forced,
                    )
                    fm_loss = cast(Tensor, terms["loss"]).mean()
                    sdb_loss_val = torch.tensor(0.0, device=fm_loss.device)
                    i2sb_loss_val = torch.tensor(0.0, device=fm_loss.device)
                    regression_loss_val = torch.tensor(0.0, device=fm_loss.device)
                    pred_x1 = cast(Tensor, terms.get("pred_x_clean", terms["pred"]))
                    loss = fm_loss
                elif self.transport_backend == "sdb":
                    if self.sdb_plan is None:
                        raise RuntimeError("SDB plan is not initialized.")
                    x1 = self._get_sdb_x1(batch, like=gt)
                    t = self.sdb_plan.train_continous_t(gt.shape[0]).to(device=gt.device, dtype=gt.dtype)
                    x_t, target = self.sdb_plan.get_x_t_with_target(t, gt, x1)
                    pred_tgt, weight = self._sdb_model_forward(
                        model=self.model,
                        x_t=x_t,
                        t=t,
                        conditions=cond_tensor,
                    )
                    sdb_loss_val = self._compute_sdb_loss(pred_tgt, target, weight=weight)
                    pred_x0 = self._sdb_pred_to_x0(pred_tgt, t=t, x_t=x_t, x_1=x1)
                    fm_loss = sdb_loss_val
                    i2sb_loss_val = torch.tensor(0.0, device=sdb_loss_val.device)
                    regression_loss_val = torch.tensor(0.0, device=sdb_loss_val.device)
                    pred_x1 = pred_x0
                    loss = sdb_loss_val
                else:
                    if self.i2sb_diffusion is None:
                        raise RuntimeError("I2SB diffusion is not initialized.")
                    x1 = self._get_i2sb_x1(batch, like=gt)
                    model_wrapped = self._get_i2sb_model(self.model)
                    cond_tensor = self._to_model_device_dtype(cond_tensor, model_wrapped)
                    terms = self.i2sb_diffusion.training_loss(
                        model_wrapped,
                        gt,
                        x1,
                        model_kwargs={"conditions": cond_tensor},
                    )
                    i2sb_loss_val = cast(Tensor, terms["loss"]).mean()
                    pred_x1 = cast(Tensor, terms["pred_x0"])
                    fm_loss = torch.tensor(0.0, device=i2sb_loss_val.device)
                    sdb_loss_val = torch.tensor(0.0, device=i2sb_loss_val.device)
                    regression_loss_val = torch.tensor(0.0, device=i2sb_loss_val.device)
                    t = torch.zeros(gt.shape[0], device=gt.device, dtype=gt.dtype)
                    loss = i2sb_loss_val
            else:
                regression_px = self._get_regression_input_px(batch)
                regression_input = self._encode_if_needed(regression_px)
                if not torch.is_tensor(regression_input):
                    raise TypeError("Encoded regression input must be a tensor.")
                pred_x1 = self._forward_regression_latent(
                    model=self.model_for_fm,
                    regression_input=regression_input,
                    conditions=conditions,
                )
                regression_loss_val = torch.nn.functional.l1_loss(pred_x1, gt)
                fm_loss = torch.tensor(0.0, device=regression_loss_val.device)
                sdb_loss_val = torch.tensor(0.0, device=regression_loss_val.device)
                i2sb_loss_val = torch.tensor(0.0, device=regression_loss_val.device)
                t = torch.zeros(
                    regression_input.shape[0],
                    device=regression_input.device,
                    dtype=regression_input.dtype,
                )
                loss = regression_loss_val

            pixel_loss_weight = float(getattr(self.train_cfg, "pixel_loss_weight", 0.0))

            grad_loss_weight = float(getattr(self.train_cfg, "grad_loss_weight", 0.0))
            fft_loss_weight = float(getattr(self.train_cfg, "fft_loss_weight", 0.0))
            ssim_loss_weight = float(getattr(self.train_cfg, "ssim_loss_weight", 0.0))

            pixel_loss_val = torch.tensor(0.0, device=loss.device)
            grad_loss_val = torch.tensor(0.0, device=loss.device)
            fft_loss_val = torch.tensor(0.0, device=loss.device)
            ssim_loss_val = torch.tensor(0.0, device=loss.device)

            if pixel_loss_weight > 0 and self.vae is not None:
                pred_px = self.vae.decode_with_grad(pred_x1, input_shape=gt_px.shape[1])
                pixel_loss_val = torch.nn.functional.l1_loss(self.to_rgb(pred_px), self.to_rgb(gt_px))
                loss = loss + pixel_loss_weight * pixel_loss_val

            if grad_loss_weight > 0:
                grad_loss_val = compute_gradient_loss(pred_x1, gt)
                loss = loss + grad_loss_weight * grad_loss_val

            if fft_loss_weight > 0:
                high_freq_ratio = float(getattr(self.train_cfg, "fft_high_freq_ratio", 0.3))
                fft_loss_val = compute_fft_high_freq_loss(pred_x1, gt, high_freq_ratio=high_freq_ratio)
                loss = loss + fft_loss_weight * fft_loss_val

            if ssim_loss_weight > 0:
                ssim_loss_val = compute_ssim_loss(pred_x1, gt)
                loss = loss + ssim_loss_weight * ssim_loss_val

        self.accelerator.backward(loss)
        if getattr(self.train_cfg, "max_grad_norm", None) is not None and self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.train_cfg.max_grad_norm)
        self.optim.step()
        self.sched.step()
        self.optim.zero_grad(set_to_none=True)
        self.ema_update()

        log_losses = {
            "loss/total": loss.detach(),
            "loss/fm": fm_loss.detach(),
            "loss/sdb": sdb_loss_val.detach(),
            "loss/i2sb": i2sb_loss_val.detach(),
            "loss/regression": regression_loss_val.detach(),
            "loss/pixel": pixel_loss_val.detach(),
            "loss/grad": grad_loss_val.detach(),
            "loss/fft": fft_loss_val.detach(),
            "loss/ssim": ssim_loss_val.detach(),
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
        *,
        x0: Tensor | None = None,
        x1: Tensor | None = None,
    ) -> Tensor:
        # Narrow cases by asserting
        assert conditions is not None, "Diffusion samplers require conditions for cloud removal."

        if self.transport_backend == "i2sb":
            if x1 is None:
                raise ValueError("I2SB sampling requires x1 (cloud) tensor.")
            if x1.shape != shape:
                raise ValueError(f"x1 shape {x1.shape} must match target shape {shape}.")
            return self._i2sb_sample_latent(
                model=model,
                x_1=x1,
                conditions=conditions,
                sampling_type=sampling_type,
            )

        if self.transport_backend == "sdb":
            if x1 is None:
                raise ValueError("SDB sampling requires x1 (cloud) tensor.")
            if x1.shape != shape:
                raise ValueError(f"x1 shape {x1.shape} must match target shape {shape}.")
            return self._sdb_sample_latent(
                model=model,
                x_1=x1,
                conditions=conditions,
                sampling_type=sampling_type,
            )

        if self.fm_sampler is None:
            raise RuntimeError("Flow matching sampler is not initialized.")
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

        if x0 is None:
            x0 = torch.randn(shape, device=self.device, dtype=self.dtype)
        else:
            if not torch.is_tensor(x0):
                raise TypeError("x0 must be a tensor when provided.")
            if x0.shape != shape:
                raise ValueError(f"x0 shape {x0.shape} must match target shape {shape}.")
            x0 = x0.to(device=self.device, dtype=self.dtype)
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
                if self.train_mode == "diffusion":
                    if self.transport_backend == "flow_matching":
                        if self.fm_transport is None:
                            raise RuntimeError("Flow matching transport is not initialized.")
                        cond_tensor = pack_conditions(conditions)
                        x0_forced = self._get_flow_matching_x0(batch)
                        t, _, _ = self.fm_transport.sample(gt)
                        t = self._maybe_apply_time_shift(t, x=gt)

                        model_for_fm_eval = ModelOutputWrapper(model_for_vis)
                        terms = self.fm_transport.training_losses(
                            model_for_fm_eval,
                            gt,
                            model_kwargs={"conditions": cond_tensor},
                            t_forced=t,
                            x0_forced=x0_forced,
                        )
                        pred_latent = cast(Tensor, terms.get("pred_x_clean", terms["pred"]))
                    elif self.transport_backend == "sdb":
                        if self.sdb_plan is None:
                            raise RuntimeError("SDB plan is not initialized.")
                        cond_tensor = pack_conditions(conditions)
                        x1 = self._get_sdb_x1(batch, like=gt)
                        t = self.sdb_plan.train_continous_t(gt.shape[0]).to(device=gt.device, dtype=gt.dtype)
                        x_t, _ = self.sdb_plan.get_x_t_with_target(t, gt, x1)
                        pred_tgt, _ = self._sdb_model_forward(
                            model=model_for_vis,
                            x_t=x_t,
                            t=t,
                            conditions=cond_tensor,
                        )
                        pred_latent = self._sdb_pred_to_x0(pred_tgt, t=t, x_t=x_t, x_1=x1)
                    else:
                        if self.i2sb_diffusion is None:
                            raise RuntimeError("I2SB diffusion is not initialized.")
                        cond_tensor = pack_conditions(conditions)
                        x1 = self._get_i2sb_x1(batch, like=gt)
                        model_wrapped = self._get_i2sb_model(model_for_vis)
                        cond_tensor = self._to_model_device_dtype(cond_tensor, model_wrapped)
                        terms = self.i2sb_diffusion.training_loss(
                            model_wrapped,
                            gt,
                            x1,
                            model_kwargs={"conditions": cond_tensor},
                        )
                        pred_latent = cast(Tensor, terms["pred_x0"])
                else:
                    regression_px = self._get_regression_input_px(batch)
                    regression_input = self._encode_if_needed(regression_px)
                    if not torch.is_tensor(regression_input):
                        raise TypeError("Encoded regression input must be a tensor.")
                    pred_latent = self._forward_regression_latent(
                        model=model_for_vis,
                        regression_input=regression_input,
                        conditions=conditions,
                    )
                pred_px = self._decode_if_needed(pred_latent, input_shape=gt_px.shape)

            self.visualize_triplet(
                cloudy=cloudy_px,
                pred=pred_px,
                gt=gt_px,
                sar=batch.get("s1", None),
                img_name="train/cloud_removal",
                add_step=True,
            )

    def _convert_rgb_vae_cause(self, batch: dict):
        if self._vae_only_support_rgb:
            for n in batch.keys():
                v = batch[n]
                if torch.is_tensor(v) and v.ndim == 4:
                    # is image
                    batch[n] = v[:, :3]  # assume first 3 channels is RGB.
        return batch

    def infinity_train_loader(self):
        while True:
            for batch in self.train_dataloader:
                batch = self._convert_rgb_vae_cause(batch)
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
        logger.success("[Load EMA]: loaded EMA weights")

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
                batch = self._convert_rgb_vae_cause(batch)
                yield batch
        else:
            for batch in tqdm(self.val_dataloader, desc="validating ...", disable=not self.accelerator.is_main_process):
                batch = self._convert_rgb_vae_cause(batch)
                yield batch

    @torch.no_grad()
    def val_loop(self) -> None:
        model_for_eval = self._get_eval_model()
        model_for_eval.eval()

        self.cr_metric.reset()

        last_batch: dict[str, Any] | None = None
        last_pred_px: Tensor | None = None
        for batch in self.get_val_loader_iter():
            batch = self.to_device_dtype(batch)
            gt_px = batch["gt"]
            cond_px = self._extract_conditions(batch)

            with self.accelerator.autocast():
                gt_latent = self._encode_if_needed(gt_px)
                conditions = self._encode_if_needed(cond_px) if cond_px is not None else None
                if self.train_mode == "diffusion":
                    if self.transport_backend == "flow_matching":
                        x0_forced = self._get_flow_matching_x0(batch)
                        pred_latent = self.general_sampler(
                            model_for_eval,
                            gt_latent.shape,
                            conditions=conditions,
                            x0=x0_forced,
                        )
                    elif self.transport_backend == "sdb":
                        if not torch.is_tensor(gt_latent):
                            raise TypeError("Encoded ground-truth must be a tensor for SDB sampling.")
                        x1 = self._get_sdb_x1(batch, like=gt_latent)
                        pred_latent = self.general_sampler(
                            model_for_eval,
                            gt_latent.shape,
                            conditions=conditions,
                            x1=x1,
                        )
                    else:
                        if not torch.is_tensor(gt_latent):
                            raise TypeError("Encoded ground-truth must be a tensor for I2SB sampling.")
                        x1 = self._get_i2sb_x1(batch, like=gt_latent)
                        pred_latent = self.general_sampler(
                            model_for_eval,
                            gt_latent.shape,
                            conditions=conditions,
                            x1=x1,
                        )
                else:
                    regression_px = self._get_regression_input_px(batch)
                    regression_input = self._encode_if_needed(regression_px)
                    if not torch.is_tensor(regression_input):
                        raise TypeError("Encoded regression input must be a tensor.")
                    pred_latent = self._forward_regression_latent(
                        model=model_for_eval,
                        regression_input=regression_input,
                        conditions=conditions,
                    )
                pred_px = self._decode_if_needed(pred_latent, input_shape=gt_px.shape)

            # torchmetrics expects float tensors; we normalize/clamp into [0, 1]
            pred_metric = self.to_rgb(pred_px).float()
            gt_metric = self.to_rgb(gt_px).float()
            self.cr_metric.update(pred_metric, gt_metric)  # type: ignore[call-arg]

            self.step_train_state("val")
            last_batch = batch
            last_pred_px = pred_px

        cr_vals = self.cr_metric.compute()  # type: ignore[call-arg]
        psnr_val = cr_vals["PSNR"].detach()
        ssim_val = cr_vals["SSIM"].detach()
        lpips_val = cr_vals["LPIPS"].detach()
        rmse_val = cr_vals["RMSE"].detach()
        psnr_val = self.accelerator.gather(psnr_val).mean()
        ssim_val = self.accelerator.gather(ssim_val).mean()
        lpips_val = self.accelerator.gather(lpips_val).mean()
        rmse_val = self.accelerator.gather(rmse_val).mean()

        if self.accelerator.is_main_process:
            self.log_msg(
                f"[Val]: psnr={psnr_val.item():.4f} | ssim={ssim_val.item():.4f} | "
                f"lpips={lpips_val.item():.4f} | rmse={rmse_val.item():.4f}"
            )
            self.tenb_log_any(
                "metric",
                {
                    "val/psnr": float(psnr_val.item()),
                    "val/ssim": float(ssim_val.item()),
                    "val/lpips": float(lpips_val.item()),
                    "val/rmse": float(rmse_val.item()),
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

        # self._save_vae_recon_preview()
        self.train_loop()


_key = "cuhk_cr1_regression_unet"
_config = {
    # regression in latent space
    "cuhk_cr1_regression_unet": "cloud_removal_cuhk_cr1_regression_unet",
    "cuhk_cr1_regression_flux": "cloud_removal_cuhk_cr1_regression_unet_flux_vae",
    # genrative
    "cuhk_cr1_fm_cloud_x0": "cloud_removal_cuhk_cr1_fm_x0_cloud",
    "cuhk_cr1_sdb_cloud_x0_unet": "cloud_removal_cuhk_cr1_sdb_x0_cloud_unet",
    "cuhk_cr1_i2sb_cloud_x1_unet": "cloud_removal_cuhk_cr1_i2sb_x0_cloud_unet",
    # pixel space
    "cuhk_cr1_regression_unet_no_vae_ps2": "cloud_removal_cuhk_cr1_regression_unet_no_vae_ps2",
    # "sen12": "cloud_removal_sen12",
}[_key]


if __name__ == "__main__":
    from src.utilities.train_utils.cli import argsparse_cli_args, print_colored_banner

    print_colored_banner("Cloud Removal")

    @hydra.main(
        config_path="../configs/cloud_removal",
        config_name=_config,
        version_base=None,
    )
    def main(cfg: DictConfig):
        catcher = logger.catch if PartialState().is_main_process else nullcontext
        with catcher():
            trainer = HyperLatentDiffusiveCloudRemovalTrainer(cfg)
            trainer.run()

    main()

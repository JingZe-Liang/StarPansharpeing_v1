import math
import os
import sys
import time
from collections import namedtuple
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Callable, Literal, Sequence, cast

import accelerate
import hydra
import lazy_loader
import numpy as np
import PIL.Image as Image
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.state import PartialState
from accelerate.tracking import TensorBoardTracker
from accelerate.utils.deepspeed import DummyOptim, DummyScheduler
from einops import rearrange
from ema_pytorch import EMA
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._fully_shard import FSDPModule
from torch.distributed.tensor import DTensor
from torchmetrics.aggregation import MeanMetric
from tqdm import tqdm, trange

from src.data.window_slider import WindowSlider, model_predict_patcher
from src.stage1.cosmos.inference.utils import load_jit_model_shape_matched
from src.stage2.change_detection.data.label_centric_patcher import (
    label_centrical_patcher,
)
from src.stage2.segmentation.data.data_split import (
    SingleImageHyperspectralSegmentationDataset,
)
from src.stage2.segmentation.metrics import HyperSegmentationScore
from src.stage2.utilities.loss import HyperSegmentationLoss
from src.utilities.config_utils import (
    to_object as to_cont,  # register new resolvers at the same time
)
from src.utilities.logging import dict_round_to_list_str, log
from src.utilities.logging.print import log_print
from src.utilities.network_utils import load_peft_model_checkpoint
from src.utilities.network_utils.Dtensor import safe_dtensor_operation
from src.utilities.train_utils.state import (
    StepsCounter,
    dict_tensor_sync,
    metrics_sync,
    object_scatter,
)
from src.utilities.train_utils.visualization import (
    get_rgb_image,
    visualize_segmentation_map,
)

heavyball = lazy_loader.load("heavyball")

CDModelStepOutput = namedtuple(
    "CDModelStepOutput",
    ["pred_pixel", "loss", "log_losses"],
)


class HyperCDTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        # Tokenizer configuration is now handled by wrapper
        self.train_cfg = cfg.train
        self.dataset_cfg = cfg.dataset
        self.ema_cfg = cfg.ema
        self.val_cfg = cfg.val
        self.metric_cfg = cfg.metric

        # accelerator
        self.accelerator: Accelerator = hydra.utils.instantiate(cfg.accelerator)
        accelerate.utils.set_seed(2025)

        # logger
        log_file = self.configure_logger()

        # attributes
        self.device = self.accelerator.device
        torch.cuda.set_device(self.accelerator.local_process_index)
        self.dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "no": torch.float32,
        }[self.accelerator.mixed_precision]
        self.log_msg("Log file is saved at: {}".format(log_file))
        self.log_msg("Weights will be saved at: {}".format(self.proj_dir))
        self.log_msg("Training is configured and ready to start.")

        # Initialize indexing mode
        self._use_single_img_indexing = getattr(
            self.train_cfg, "use_single_img_indexing", False
        )
        if self._use_single_img_indexing:
            self.log_msg("Using HyperSIGMA-style indexing for loss computation")
            # Initialize accumulation buffers for HyperSIGMA mode
            self._accumulated_preds = []
            self._accumulated_gts = []
            self._accumulated_indices = []
            self._current_image_id = 0
            self._patches_per_image = getattr(self.train_cfg, "patches_per_image", None)
            self.macro_batch_size = getattr(self.train_cfg, "macro_batch_size", 32)
            if self._patches_per_image is None:
                self.log_msg(
                    "Warning: patches_per_image not specified, will auto-detect",
                    level="WARNING",
                )
        else:
            self.log_msg("Using standard segmentation loss computation")

        # is zero 2 or 3, not EMA
        _dpsp_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        _fsdp_plugin: accelerate.utils.FullyShardedDataParallelPlugin = getattr(  # type: ignore
            self.accelerator.state, "fsdp_plugin", None
        )
        self.no_ema = False
        self._is_ds = _dpsp_plugin is not None
        if self._is_ds:
            self.log_msg("[Deepspeed]: using deepspeed plugin")
            self.no_ema = _dpsp_plugin.deepspeed_config["zero_optimization"][  # type: ignore
                "stage"
            ] in [2, 3]

        self._is_fsdp = _fsdp_plugin is not None
        if self._is_fsdp:
            self.log_msg("[FSDP]: using Fully Sharded Data Parallel plugin")
            self.no_ema = True

        # dataloader
        self.train_dataset, self.train_dataloader = hydra.utils.instantiate(
            self.dataset_cfg.train
        )
        self.val_dataset, self.val_dataloader = hydra.utils.instantiate(
            self.dataset_cfg.val
        )
        if _dpsp_plugin is not None:
            self.accelerator.deepspeed_plugin.deepspeed_config[  # type: ignore
                "train_micro_batch_size_per_gpu"
            ] = self.dataset_cfg.batch_size_train

        # setup the segmentation model
        self.setup_segmentation_model()  # setup the segmentation model

        # optimizers and lr schedulers
        self.optim, self.sched = self.get_optimizer_lr_scheduler()

        # EMA models and accelerator prepare
        self.prepare_for_training()
        self.prepare_ema_models()

        # loss
        self.segment_loss: HyperSegmentationLoss | Callable
        if hasattr(self.cfg, "segment_loss"):
            self.segment_loss = hydra.utils.instantiate(self.cfg.segment_loss)
        else:
            # default loss
            self.segment_loss = nn.CrossEntropyLoss()
        self.log_msg(f"use segmentation loss: {self.segment_loss.__class__.__name__}")

        # training state counter
        self.train_state = StepsCounter(["train", "val"])

        # clear GPU memory
        torch.cuda.empty_cache()

    def setup_segmentation_model(self):
        self.model = hydra.utils.instantiate(self.cfg.segment_model)
        # self.model.to(device=self.device, dtype=self.dtype)

        if self.train_cfg.act_checkpoint and hasattr(
            self.model, "set_grad_checkpointing"
        ):
            self.log_msg("Using activation checkpointing to save memory")
            self.model.set_grad_checkpointing(True)

        segment_name = (
            getattr(self.train_cfg, "segment_name", None)
            or self.model.__class__.__name__
        )
        self.log_msg(f"use change detection model: {segment_name}")

    def prepare_ema_models(self):
        if self.no_ema:
            return

        buffer_name = [name for name, _ in self.model.named_buffers()]
        self.log_msg(f"EMA ignore buffers: {buffer_name}")
        self.ema_model: EMA = hydra.utils.instantiate(self.cfg.ema)(
            model=self.model, ignore_names=set(buffer_name)
        ).to(self.device)
        self.log_msg(f"create EMA model for change detection")

    def configure_logger(self):
        self.logger = logger

        log_file = Path(self.train_cfg.proj_dir)
        if self.train_cfg.log.log_with_time:
            str_time = time.strftime("%Y-%m-%d_%H-%M-%S")
            log_file = log_file / str_time
        if self.train_cfg.log.run_comment is not None:
            log_file = Path(log_file.as_posix() + "_" + self.train_cfg.log.run_comment)
        log_file = log_file / "log.log"

        # when distributed, there should be the same log_file
        if self.accelerator.use_distributed:
            log_file = object_scatter(log_file)
            assert isinstance(log_file, Path), "log_file type should be Path"

        # logger
        self.logger.remove()
        log_format_in_file = (
            "<green>[{time:MM-DD HH:mm:ss}]</green> "
            "- <level>[{level}]</level> "
            "- <cyan>{file}:{line}</cyan> - <level>{message}</level>"
        )
        log_format_in_cmd = (
            "{time:HH:mm:ss} "
            "- {level.icon} <level>[{level}] {file.name}:{line}</level>"
            "- <level>{message}</level>"
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
            # including trace and debug
            self.logger.add(
                log_file.parent / "debug.log",
                format=log_format_in_file,
                level="DEBUG",
                filter=lambda record: record["level"].no <= 10,
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

        # make log dir
        log_dir = log_file.parent
        if not self.train_cfg.debug:
            log_dir.mkdir(parents=True, exist_ok=True)

        # copy cfg
        if not self.train_cfg.debug:
            yaml_cfg = OmegaConf.to_yaml(self.cfg, resolve=True)
            cfg_cp_path = log_file.parent / "config" / "config_total.yaml"
            cfg_cp_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cfg_cp_path, "w") as f:
                f.write(yaml_cfg)
            self.logger.info(f"[Cfg]: configuration saved to {cfg_cp_path}")

        # accelerate project configuration
        self.proj_dir = log_dir
        self.accelerator.project_configuration.project_dir = str(self.proj_dir)

        # tensorboard logger
        if not self.train_cfg.debug:
            tenb_dir = log_dir / "tensorboard"
            self.accelerator.project_configuration.logging_dir = tenb_dir
            if self.accelerator.is_main_process:
                self.logger.info(f"[Tensorboard]: tensorboard saved to {tenb_dir}")
                self.accelerator.init_trackers("train")
                self.tb_logger: TensorBoardTracker = self.accelerator.get_tracker(  # type: ignore
                    "tensorboard"
                )

        return log_file

    def tenb_log_any(
        self,
        log_type: Literal["metric", "image", "grad_norm_per_param", "grad_norm_sum"],
        logs: dict,
        step: int,
        **kwargs,
    ):
        assert log_type in [
            "metric",
            "image",
            "grad_norm_per_param",
            "grad_norm_sum",
        ], "log_type must be one of [metric, image, grad_norm_per_param, grad_norm_sum]"

        if log_type == "metric":
            if hasattr(self, "tb_logger"):
                self.tb_logger.log(logs, step=step)
        elif log_type == "image":
            if hasattr(self, "tb_logger"):
                self.tb_logger.log_images(logs, step=step)
        elif log_type in ("grad_norm_per_param", "grad_norm_sum"):
            assert "model" in logs, "model name must be in logs"
            model = logs.pop("model")
            # take out the grad of norms
            model_cls_n = model.__class__.__name__
            norms = {}
            if log_type == "grad_norm_sum":
                norms[f"{model_cls_n}_grad_norm"] = 0
                _n_params_sumed = 0
            for n, p in model.named_parameters():
                if p.grad is not None:
                    # must sync grad here, `is_main_process` would cause the ranks do not sync
                    if isinstance(p.grad, DTensor):
                        _grad = p.grad._local_tensor
                        if p.grad._local_tensor.device == torch.device("cpu"):
                            self.log_msg(
                                "p.grad is on cpu, this should not happen",
                                level="WARNING",
                            )
                            # ensure the corss rank does not involve cpu bankend
                            _grad = _grad.cuda()
                        _p_grad = safe_dtensor_operation(p.grad)
                        _grad_norm = (_p_grad.data**2).sum() ** 0.5
                    else:
                        _grad_norm = p.grad.data.norm()

                    if log_type == "grad_norm_per_param":
                        norms[f"{model_cls_n}/{n}"] = _grad_norm
                    else:
                        norms[f"{model_cls_n}_grad_norm"] += _grad_norm
                        _n_params_sumed += 1
            # log
            if log_type == "grad_norm_sum":
                norms[f"{model_cls_n}_grad_norm"] /= _n_params_sumed
            if hasattr(self, "tb_logger"):
                self.tb_logger.log(norms, step=step)
        else:
            raise NotImplementedError(f"Unknown log_type {log_type}")

    def log_msg(
        self,
        *msgs,
        only_rank_zero=True,
        level="INFO",
        sep=",",
        warn_once=False,
        **kwargs,
    ):
        if warn_once:
            level = "warning"

        assert level.lower() in [
            "info",
            "warning",
            "error",
            "debug",
            "critical",
        ], f"Unknown level {level}"

        def str_msg(*msg):
            return sep.join([str(m) for m in msg])

        _log_fn = partial(
            log_print,
            level=level.lower(),
            warn_once=warn_once,
            only_rank_zero=only_rank_zero,
            stack_level=3,
            **kwargs,
        )

        def log_it(*msg, **kwargs):
            msg_string = str_msg(*msg)
            _log_fn(msg_string, **kwargs)

        log_it(*msgs, **kwargs)

    def get_optimizer_lr_scheduler(
        self,
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        # optimizers
        if (
            self.accelerator.state.deepspeed_plugin is None
            or "optimizer"
            not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):

            def _optimizer_creater(optimizer_cfg):
                if "get_muon_optimizer" in optimizer_cfg._target_:
                    self.log_msg("[Optimizer]: using muon optimizer")
                    # is muon optimizer function
                    named_params = {
                        k: v
                        for k, v in self.model.named_parameters()
                        if v.requires_grad
                    }
                    return hydra.utils.instantiate(optimizer_cfg)(
                        named_parameters=named_params
                    )
                else:
                    self.log_msg(
                        f"[Optimizer]: using optimizer: {optimizer_cfg._target_}"
                    )
                    params = [p for p in self.model.parameters() if p.requires_grad]
                    return hydra.utils.instantiate(optimizer_cfg)(params)

            model_opt = _optimizer_creater(self.train_cfg.segment_optim)
        else:
            model_opt = DummyOptim([{"params": list(self.model.parameters())}])

        # schedulers
        if (
            self.accelerator.state.deepspeed_plugin is None
            or "scheduler"
            not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            model_sched = hydra.utils.instantiate(self.train_cfg.segment_sched)(
                optimizer=model_opt
            )
        else:
            model_sched = DummyScheduler(model_opt)

        # set the heavyball optimizer without torch compiling
        is_heavyball_opt = lambda opt: opt.__class__.__module__.startswith("heavyball")
        if is_heavyball_opt(model_opt):
            heavyball.utils.compile_mode = None
            self.log_msg(
                "use heavyball optimizer, it will compile the optimizer, "
                "for efficience testing the scripts, disable the compilation."
            )

        return model_opt, model_sched

    def set_fsdp_cpu_local_tensor_to_each_rank(self, model: nn.Module):
        if not self._is_fsdp:
            return model

        self.log_msg(
            "FSDP module seems do not move the original parameter (_local_tensor) on the"
            "correct rank, we need to manually move them on cuda while using `to_local` or `redistributed` methods",
            level="WARNING",
        )
        _cpu_device = torch.device("cpu")
        for name, param in model.named_parameters():
            if isinstance(param, DTensor) and param.device == _cpu_device:
                param._local_tensor = param._local_tensor.to(self.device)
                self.log_msg(f"set {name} local_tensor on cuda", level="DEBUG")

        return model

    def set_hypersigma_indexing_mode(self, enable: bool = True):
        """Enable or disable HyperSIGMA-style indexing for loss computation.

        Args:
            enable: Whether to use HyperSIGMA-style indexing
        """
        self._use_single_img_indexing = enable
        mode_str = "enabled" if enable else "disabled"
        self.log_msg(f"HyperSIGMA indexing mode {mode_str}")

        # Initialize or clear accumulation buffers
        if enable:
            self._accumulated_preds = []
            self._accumulated_gts = []
            self._accumulated_indices = []
            self._current_image_id = 0
        else:
            # Clear buffers when disabling
            if hasattr(self, "_accumulated_preds"):
                self._accumulated_preds.clear()
                self._accumulated_gts.clear()
                self._accumulated_indices.clear()

    def get_current_indexing_mode(self) -> str:
        """Get current indexing mode.

        Returns:
            str: Either 'hypersigma' or 'standard'
        """
        return "hypersigma" if self._use_single_img_indexing else "standard"

    def set_hypersigma_config(self, patches_per_image: int | None = None):
        """Set HyperSIGMA configuration parameters.

        Args:
            patches_per_image: Number of patches per complete image
        """
        if patches_per_image is not None:
            self._patches_per_image = patches_per_image
            self.log_msg(f"HyperSIGMA patches per image set to {patches_per_image}")

    def get_hypersigma_config(self) -> dict:
        """Get current HyperSIGMA configuration.

        Returns:
            dict: Configuration parameters
        """
        return {
            "enabled": self._use_single_img_indexing,
            "patches_per_image": getattr(self, "_patches_per_image", None),
            "accumulated_patches": len(getattr(self, "_accumulated_preds", [])),
        }

    def prepare_for_training(self) -> None:
        # FIXME: FSDP2 seems do not support the sync_bn, find a way to fix it.
        if self._is_fsdp or self.accelerator.distributed_type in (
            accelerate.utils.DistributedType.MULTI_GPU,
            accelerate.utils.DistributedType.FSDP,
        ):  # seems that FSDP does not support synchronized batchnorm
            # discriminator may have batch norm layer
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.log_msg("[Model] convert discriminator to sync batch norm")

        # if use FSDP2
        if self._is_fsdp and self.accelerator.is_fsdp2:
            # set models with property dtype
            self.model.dtype = torch.float

        # cd model
        self.model, self.optim = self.accelerator.prepare(self.model, self.optim)

        # train and val dataloader
        self.train_dataloader, self.val_dataloader = self.accelerator.prepare(
            self.train_dataloader, self.val_dataloader
        )

    def step_train_state(self, mode="train"):
        self.train_state.update(mode)

    def ema_update(self, mode="segment"):
        assert mode == "segment"
        if self.no_ema:
            # not support ema when is deepspeed zero2 or zero3
            return

        self.ema_model.update()

    def get_global_step(self, mode: str = "train"):
        # TODO: add val state
        assert mode in (
            "train",
            "val",
        ), "Only train and val modes are supported for now."

        return self.train_state[mode]

    @property
    def global_step(self):
        return self.get_global_step("train")

    def may_freeze(self, model: nn.Module, freeze=True):
        # for p in model.parameters():
        #     p.requires_grad = not freeze
        model.requires_grad_(not freeze)

    def gradient_check(self, model: nn.Module):
        # check nan gradient
        if self.accelerator.sync_gradients and getattr(
            self.train_cfg, "grad_check", True
        ):
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if param.grad is None:
                        self.log_msg(
                            f"step {self.global_step} - {name} has None gradient, shaped as {param.shape}",
                            only_rank_zero=False,
                            level="WARNING",
                        )
                    elif torch.isnan(param.grad).any():
                        self.log_msg(
                            f"step {self.global_step} - {name} has nan gradient, shaped as {param.shape}",
                            only_rank_zero=False,
                            level="WARNING",
                        )
                        torch.nan_to_num(
                            param.grad, nan=0.0, posinf=1e5, neginf=-1e5, out=param.grad
                        )

        # clip gradient by norm
        _max_grad_norm = self.train_cfg.max_grad_norm
        if _max_grad_norm is not None and _max_grad_norm > 0:
            if self.dtype != torch.float16 and not self.accelerator.is_fsdp2:
                self.accelerator.clip_grad_norm_(model.parameters(), _max_grad_norm)
            elif (
                self.accelerator.distributed_type
                == accelerate.utils.DistributedType.FSDP
                or self.accelerator.is_fsdp2
            ) and isinstance(model, FSDP):
                FSDP.clip_grad_norm_(model.parameters(), max_norm=_max_grad_norm)

    def compute_segmentation_loss(self, out, gt):
        # loss, ensure gt is long and ndim == 3
        gt = gt.type(torch.long).squeeze_(1)
        assert gt.ndim == 3, f"gt ndim should be 3, got {gt.ndim} of shape {gt.shape}"
        gt = gt.contiguous()
        loss = self.segment_loss(out, gt)

        if isinstance(loss, torch.Tensor):
            log_losses = {"seg_loss": loss.detach()}
        elif isinstance(loss, (tuple, list)):
            loss, log_losses = loss
        else:
            raise NotImplementedError(f"Unknown type {type(loss)}")

        return loss, log_losses

    def forward_segment_model(self, img1, img2, gt):
        with self.accelerator.autocast():
            # pixel data -> wrapper (handles tokenization + change detection)
            out = self.model([img1, img2])
            # loss
            loss, log_losses = self.compute_segmentation_loss(out, gt.to(out.device))

        return out, loss, log_losses

    def _check_image_completion(self, batch: dict) -> bool:
        """Check if we have processed all patches for a complete image."""

        # Method 1: Check batch metadata
        if "is_last_patch" in batch and batch["is_last_patch"]:
            return True

        # Method 2: Check accumulated patches count
        if self._patches_per_image is not None:
            if len(self._accumulated_preds) >= self._patches_per_image:
                return True

        # Method 3: Check if we've reached a reasonable limit (safety check)
        if len(self._accumulated_preds) >= 1000:  # Prevent infinite accumulation
            self.log_msg(
                "Warning: Accumulated too many patches, processing as complete image",
                level="WARNING",
            )
            return True

        return False

    def _process_complete_hypersigma_image(self):
        """Process a complete hyperspectral image and compute loss."""

        if not self._accumulated_preds:
            raise ValueError("No accumulated predictions to process.")

        # Reconstruct complete image from patches
        pred_2d, pred_indexed_1d, gt_indexed_1d = self._reconstruct_hypersigma_image()
        loss, log_losses = self.compute_segmentation_loss(
            pred_indexed_1d, gt_indexed_1d
        )
        self._accumulated_preds.clear()

        return CDModelStepOutput(pred_2d, loss, log_losses)

    def _reconstruct_hypersigma_image(
        self, accumulated_preds: list[Tensor] | None = None, mode="train"
    ):
        """Reconstruct complete hyperspectral image from accumulated patches."""

        # Stack all predictions
        # (total_patches, C, H, W)
        if accumulated_preds is None:
            accumulated_preds = self._accumulated_preds
        all_preds = torch.cat(accumulated_preds, dim=0)

        ds: SingleImageHyperspectralSegmentationDataset = (
            self.train_dataset if mode == "train" else self.val_dataset
        )
        index = ds._sampled_index
        gt_indexed_1d = ds.gt_for_loss  # (indices,)
        n_row, n_cols = ds.n_rows, ds.n_cols
        hp, wp = all_preds[0].shape[-2:]
        assert ds.total_patches == all_preds.shape[0], (
            f"{all_preds.shape[0]=} should equal to {ds.total_patches=}"
        )

        # Revert back to a full image
        pred_1d = rearrange(
            all_preds,
            "(nh nw) n_class hp wp -> (nh hp nw wp) n_class",
            nh=n_row,
            nw=n_cols,
        )
        pred_2d = rearrange(
            pred_1d,
            "(h w) n_class -> n n_class h w",
            h=n_row * hp,
            w=n_cols * wp,
            n=1,
        )
        pred_indexed_1d = pred_1d[index]
        gt_indexed_1d = gt_indexed_1d.to(pred_1d.device)

        return {
            "pred_2d": pred_2d,
            "pred_indexed_1d": pred_indexed_1d,
            "gt_indexed_1d": gt_indexed_1d,
        }

    def _optimize_step(self, loss: Tensor):
        if self.accelerator.sync_gradients:
            # backward
            self.optim.zero_grad()
            self.accelerator.backward(loss)
            self.gradient_check(self.model)
            self.optim.step()
            self.sched.step()
            # ema update
            self.ema_update()

    def _macro_forward_seg_model(self, batch: dict, macro_batch_size: int = 32):
        # For change detection, we need to handle both img1 and img2
        total_img1_patches: Tensor = batch["img1"]
        total_img2_patches: Tensor = batch["img2"]
        img1_macro_batches: tuple[Tensor, ...] = total_img1_patches.chunk(
            macro_batch_size, dim=0
        )
        img2_macro_batches: tuple[Tensor, ...] = total_img2_patches.chunk(
            macro_batch_size, dim=0
        )
        macro_outputs: list[Tensor] = []
        with self.accelerator.autocast():
            for img1_batch, img2_batch in zip(img1_macro_batches, img2_macro_batches):
                macro_pred = self.model([img1_batch, img2_batch])
                macro_outputs.append(macro_pred)
        return macro_outputs

    def train_segment_step(self, img1, img2, gt) -> CDModelStepOutput:
        pred, loss, log_losses = self.forward_segment_model(img1, img2, gt)
        self._optimize_step(loss)
        return CDModelStepOutput(pred, loss, log_losses)

    def train_single_img_accum_step(
        self,
        batch: dict,
        macro_batch_size: int = 32,
    ):
        """HyperSIGMA-style training step: accumulate predictions and compute loss after full image."""

        # Forward pass without computing loss yet
        macro_outputs = self._macro_forward_seg_model(batch, macro_batch_size)
        # Store predictions and metadata for later loss computation
        self._accumulated_preds = macro_outputs

        # Store sample indices if available
        if "sample_index" in batch:
            self._accumulated_indices.append(batch["sample_index"].detach())

        # Check if we have completed a full image
        # is_image_complete = self._check_image_completion(batch)
        # anyway, I temp set to True
        is_image_complete = True

        if is_image_complete:
            # Process the complete image and compute loss
            ret_dict = self._process_complete_hypersigma_image()
            self._optimize_step(ret_dict.loss)
            return ret_dict
        else:
            raise NotImplementedError("Not implemented yet")
            return None

    def _cast_to_dtype(self, x: torch.Tensor | list | dict[str, torch.Tensor]):
        if isinstance(x, torch.Tensor):
            return x.to(dtype=self.dtype, device=self.device)
        elif isinstance(x, list):
            return [self._cast_to_dtype(xx) for xx in x]
        elif isinstance(x, dict):
            for k, v in x.items():
                if "gt" != k:
                    x[k] = self._cast_to_dtype(v)
                else:
                    x[k] = v.to(device=self.device)
            return x

    def _get_metric_fn(self, clear=False):
        if not hasattr(self, "_seg_metrics") or clear:
            self._seg_metrics: HyperSegmentationScore = hydra.utils.instantiate(
                self.metric_cfg
            )
        # Return the segmentation metrics object
        self._seg_metrics.to(self.device)
        return self._seg_metrics

    def update_metrics(self, pred, gt, clear=False, only_update=True):
        seg_metrics = self._get_metric_fn(clear)
        assert seg_metrics is not None
        # Update segmentation metrics with prediction and ground truth
        seg_metrics.update(pred, gt)

    def get_metrics(self):
        seg_metrics = self._get_metric_fn()
        assert seg_metrics is not None
        metrics = seg_metrics.compute()

        metrics_item = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                metrics_item[k] = v.item()
            elif isinstance(v, (int, float)):
                metrics_item[k] = v
        return metrics_item

    def train_step(self, batch: dict):
        # NOTE: in HyperSIGMA mode: the batch is a single image patches, to input
        # the segmented model, we use a 'macro-batch' to control the input-model-batch-size
        batch = self._cast_to_dtype(batch)

        # Check if we should use HyperSIGMA-style indexing
        use_batch_accum = hasattr(batch, "sample_index") or "sample_index" in batch
        if use_batch_accum:
            self._current_sample_index = batch.get("sample_index", None)

        if self._use_single_img_indexing:
            # HyperSIGMA mode: accumulate predictions
            with self.accelerator.accumulate(self.model):  # use gradient accmulation ?
                train_out = self.train_single_img_accum_step(
                    batch, self.macro_batch_size
                )
        else:
            # Standard mode: process immediately
            with self.accelerator.accumulate(self.model):
                # train segmentation model with pixel data
                train_out = self.train_segment_step(
                    batch["img1"], batch["img2"], batch["gt"]
                )

        # update training state
        self.step_train_state()
        # log losses
        if self.global_step % self.train_cfg.log.log_every == 0:
            _log_losses = self.format_log(train_out.log_losses)
            self.log_msg(
                f"[Train State]: lr {self.optim.param_groups[0]['lr']:1.4e} | "
                f"[Step]: {self.global_step}/{self.train_cfg.max_steps}"
            )
            self.log_msg(f"[Train Tok]: {_log_losses}")

            # tensorboard log
            self.tenb_log_any("metric", train_out.log_losses, self.global_step)

    def format_log(self, log_loss: dict, sync=False) -> str:
        if sync:
            # log_loss = metrics_sync(log_loss, output_tensor_dict=True)
            log_loss = dict_tensor_sync(log_loss)
        strings = dict_round_to_list_str(log_loss, select=list(log_loss.keys()))
        return " - ".join(strings)

    def _postprocess_batch_label_centric(self, batch: dict):
        img1, img2, label = batch["img1"], batch["img2"], batch["gt"]
        for img1_i, img2_i, label_i in label_centrical_patcher(
            img1,
            img2,
            label,
            micro_batch_size=self.dataset_cfg.micro_batch_size,
            patch_size=self.dataset_cfg.patch_size // 2,
            unchanged_label=self.dataset_cfg.consts.unchanged_label,
            changed_label=self.dataset_cfg.consts.changed_label,
            changed_ratio=0.8,
            label_mode="seg",
        ):
            ret = {"img1": img1_i, "img2": img2_i, "gt": label_i}
            yield ret

    def _post_process_batch_slide_window(self, batch: dict):
        slider = WindowSlider(
            ["img1", "img2", "gt"],
            window_size=self.dataset_cfg.patch_size,
            stride=self.dataset_cfg.patch_size // 2,
        )
        m_bs = self.dataset_cfg.micro_batch_size
        for batch in slider.slide_windows(batch):
            n_bs = batch["img1"].shape[0]
            ...

    def infinity_train_loader(self):
        while True:
            for batch in self.train_dataloader:
                if self.dataset_cfg.data_type == "label_centrical":
                    for micro_batch in self._postprocess_batch_label_centric(batch):
                        yield micro_batch
                elif self.dataset_cfg.data_type == "slide_window":
                    for micro_batch in self._post_process_batch_slide_window(batch):
                        yield micro_batch
                else:
                    yield batch

    def train_loop(self):
        _stop_train_and_save = False
        self.accelerator.wait_for_everyone()
        self.log_msg("[Train]: start training", only_rank_zero=False)

        for batch in self.infinity_train_loader():
            # train step
            try:
                # breakpoint()
                self.train_step(batch)
            except Exception as e:
                print(f"Error during training step: {e}")
                for k, v in batch.items():
                    print(f"{k}: {v.shape if torch.is_tensor(v) else v}")
                raise RuntimeError from e

            if self.global_step % self.val_cfg.val_duration == 0:
                torch.cuda.empty_cache()
                self.val_loop()
                torch.cuda.empty_cache()
                # breakpoint()

            if self.global_step >= self.train_cfg.max_steps:
                _stop_train_and_save = True

            if (
                self.global_step % self.train_cfg.save_every == 0
                or _stop_train_and_save
            ):
                self.save_state()
                self.save_ema()

            if _stop_train_and_save:
                self.log_msg(
                    "[Train]: max training step budget reached, stop training and save"
                )
                break

    def get_val_loader_iter(self):
        if self.val_cfg.max_val_iters > 0:
            # create a new iterator for the validation loader
            # state in the loader generator
            if not hasattr(self, "_val_loader_iter"):
                self._val_loader_iter = iter(self.val_dataloader)

            def _tbar_loader():
                tbar = trange(
                    self.val_cfg.max_val_iters,
                    desc="validating ...",
                    leave=False,
                    disable=not self.accelerator.is_main_process,
                )
                for _ in tbar:
                    try:
                        yield next(self._val_loader_iter)
                    except StopIteration:
                        # re-create the iterator
                        self._val_loader_iter = iter(self.val_dataloader)
                        yield next(self._val_loader_iter)

            self.log_msg(
                f"[Val]: start validating with only {self.val_cfg.max_val_iters} batches",
                only_rank_zero=False,
            )
        else:
            iterable_ = iter(self.val_dataloader)

            def _tbar_loader():
                for batch in tqdm(
                    iterable_,
                    desc="validating ...",
                    leave=False,
                    disable=not self.accelerator.is_main_process,
                ):
                    yield batch

            self.log_msg(
                f"[Val]: start validating with the whole val set", only_rank_zero=False
            )

        tbar = _tbar_loader()
        return tbar

    @torch.no_grad()
    def val_step(self, batch: dict):
        if self._use_single_img_indexing:
            use_batch_accum = hasattr(batch, "sample_index") or "sample_index" in batch
            if use_batch_accum:
                self._current_sample_index = batch.get("sample_index", None)
            macro_outputs = self._macro_forward_seg_model(batch, self.macro_batch_size)
            # revert back to a full image
            pred_seg, *_ = (
                self._reconstruct_hypersigma_image(macro_outputs, mode="val")
            ).values()
        else:

            def _val_model_closure(batch):
                # forward the segmentation network with pixel data
                pred_seg, *_ = self.forward_segment_model(
                    batch["img1"], batch["img2"], batch["gt"]
                )
                return {"pred_logits": pred_seg}

            # slide windows
            if self.train_cfg.val_slide_window:
                model_outputs = model_predict_patcher(
                    _val_model_closure,
                    batch,
                    patch_keys=["img1", "img2", "gt"],
                    patch_size=128,
                    stride=64,
                    merge_keys=["pred_logits"],
                )
                pred_seg = model_outputs["pred_logits"].argmax(1)
            else:
                pred_seg = _val_model_closure(batch)["pred_logits"].argmax(1)

        return pred_seg

    @torch.inference_mode()
    def val_loop(self):
        self.model.eval()

        # loss_metrics = MeanMetric().to(device=self.device)
        val_iter = self.get_val_loader_iter()
        for batch_or_idx in val_iter:  # type: ignore
            if self.val_cfg.max_val_iters > 0:
                batch = next(self._val_loader_iter)
            else:
                batch = batch_or_idx

            batch = self._cast_to_dtype(batch)
            batch = cast(dict[str, torch.Tensor], batch)

            gt = batch.get("gt", batch.get("gt_full", None))
            assert gt is not None, "gt or gt_full not found in the val batch"

            pred_seg = self.val_step(batch)
            # metrics - use segmentation predictions and ground truth
            self.update_metrics(pred_seg, gt)
            self.step_train_state("val")

        metrics = self.get_metrics()
        # loss_val = loss_metrics.compute()
        loss_val = 0.0

        self.model.train()
        self.optim.zero_grad()

        if self.accelerator.is_main_process:
            _metric_str = ""
            for k, v in metrics.items():
                _metric_str += f"{k}: {v:.4f} - "
            self.log_msg(f"[Val]: {_metric_str} loss: {loss_val:.3e}")
            self.tenb_log_any("metric", metrics, step=self.global_step)

            # visualize the last val batch
            self.visualize_segmentation(
                batch["img1"],
                batch["img2"],
                pred_seg,  # prediction
                gt,  # gt
                add_step=True,
                img_name="val/segmentation",
            )

    def save_state(self):
        self.accelerator.save_state()
        self.log_msg("[State]: save states")

    def save_ema(self):
        if self.no_ema:
            self.log_msg(f"use deepspeed or FSDP, do have EMA model to save")
            return

        ema_path = self.proj_dir / "ema"
        if self.accelerator.is_main_process:
            ema_path.parent.mkdir(parents=True, exist_ok=True)

        self.accelerator.save_model(self.ema_model.ema_model, ema_path / "cd_ema_model")
        # train state
        _ema_path_state_train = ema_path / "train_state.pth"
        _ema_path_state_train.parent.mkdir(parents=True, exist_ok=True)
        accelerate.utils.save(self.train_state.state_dict(), _ema_path_state_train)
        self.log_msg(f"[ckpt]: save ema at {ema_path}")

    def load_from_ema(self, ema_path: str | Path, strict: bool = True):
        ema_path = Path(ema_path)

        accelerate.load_checkpoint_in_model(
            self.model, ema_path / "model", strict=strict
        )

        # Prepare models
        self.prepare_ema_models()  # This will update EMA models with online models' weights

        # clear the accelerator model registration
        self.log_msg(
            f"[Load EMA]: clear the accelerator registrations and re-prepare training"
        )

    def resume(self, path: str):
        self.log_msg("[Resume]: resume training")
        self.accelerator.load_state(path)
        self.accelerator.wait_for_everyone()

    def to_rgb(self, x):
        # Default to [0, 1] range for visualization
        return x.clamp(0, 1).float()

    def visualize_segmentation(
        self,
        img1,
        img2,
        pred_map: torch.Tensor,
        gt_map: torch.Tensor,
        img_name: str = "val/segmentation",
        add_step: bool = False,
        only_vis_n: int | None = None,
        n_class: int = 20,
        use_coco_colors: bool = True,
    ):
        """Visualize predicted and ground truth segmentation maps side by side.

        Args:
            pred_map: Predicted segmentation map.
            gt_map: Ground truth segmentation map.
            img_name: Name for the saved image. Defaults to "val/segmentation".
            add_step: Whether to add step number to the image name. Defaults to False.
            only_vis_n: Number of images to visualize. Defaults to None (all).
            n_class: Number of classes. Defaults to 20.
            use_coco_colors: Whether to use COCO dataset colors. Defaults to True.
        """

        _only_n = only_vis_n or 1

        # Process prediction and ground truth maps
        if pred_map.dim() == 4:  # (B, C, H, W)
            pred_map = pred_map.squeeze(1)  # (B, H, W)

        if gt_map.dim() == 4:  # (B, C, H, W)
            gt_map = gt_map.squeeze(1)  # (B, H, W)

        # Only visualize the first few samples
        assert pred_map.ndim == 3 and gt_map.ndim == 3, (
            f"{pred_map.shape=}, {gt_map.shape=}"
        )
        assert pred_map.shape[-2:] == gt_map.shape[-2:], (
            f"{pred_map.shape=}, {gt_map.shape=}"
        )
        assert img1.shape == img2.shape, f"{img1.shape=}, {img2.shape=}"

        pred_map = pred_map[:_only_n]
        gt_map = gt_map[:_only_n]
        img1 = img1[:_only_n]
        img2 = img2[:_only_n]

        # Visualize input images
        img1_rgb = get_rgb_image(
            self.to_rgb(img1), rgb_channels=[2, 1, 0], use_linstretch=True
        )
        img2_rgb = get_rgb_image(
            self.to_rgb(img2), rgb_channels=[2, 1, 0], use_linstretch=True
        )

        # Convert RGB images to numpy arrays and ensure correct format
        def convert_to_hwc_format(img_tensor):
            """Convert tensor to (H, W, C) numpy array format."""
            if isinstance(img_tensor, Image.Image):
                return np.array(img_tensor)

            # Handle tensor format
            if torch.is_tensor(img_tensor):
                img_tensor = img_tensor.detach().cpu()
                if img_tensor.dim() == 4:  # (B, C, H, W) -> (H, W, C)
                    img_tensor = img_tensor[0]  # Take first sample for now
                if (
                    img_tensor.dim() == 3 and img_tensor.shape[0] == 3
                ):  # (C, H, W) -> (H, W, C)
                    img_tensor = img_tensor.permute(1, 2, 0)
            elif (
                isinstance(img_tensor, np.ndarray) and img_tensor.shape[0] == 3
            ):  # (C, H, W) -> (H, W, C)
                img_tensor = np.transpose(img_tensor, (1, 2, 0))

            return img_tensor.numpy() if torch.is_tensor(img_tensor) else img_tensor

        # Convert batch images to correct format
        img1_rgb = convert_to_hwc_format(img1_rgb)
        img2_rgb = convert_to_hwc_format(img2_rgb)

        # Visualize segmentation maps for entire batch
        pred_vis = visualize_segmentation_map(
            pred_map,
            n_class=n_class,
            use_coco_colors=use_coco_colors,
            to_pil=False,
            bg_black=True,
        )

        gt_vis = visualize_segmentation_map(
            gt_map,
            n_class=n_class,
            use_coco_colors=use_coco_colors,
            to_pil=False,
            bg_black=True,
        )

        # Ensure segmentation maps are numpy arrays
        if isinstance(pred_vis, Image.Image):
            pred_vis = np.array(pred_vis)
        elif isinstance(pred_vis, list):
            pred_vis = np.stack([np.array(p) for p in pred_vis], axis=0)

        if isinstance(gt_vis, Image.Image):
            gt_vis = np.array(gt_vis)
        elif isinstance(gt_vis, list):
            gt_vis = np.stack([np.array(g) for g in gt_vis], axis=0)

        # Resize input images to match segmentation map size
        h, w = pred_vis.shape[1:3] if pred_vis.ndim == 4 else pred_vis.shape[:2]

        if img1_rgb.shape[:2] != (h, w):
            img1_rgb = np.array(
                Image.fromarray(img1_rgb).resize((w, h), Image.Resampling.BILINEAR)
            )
        if img2_rgb.shape[:2] != (h, w):
            img2_rgb = np.array(
                Image.fromarray(img2_rgb).resize((w, h), Image.Resampling.BILINEAR)
            )

        # Handle batch dimension and concatenate images
        if pred_vis.ndim == 4:  # Batch processing
            # Stack all images horizontally for each sample
            batch_size = pred_vis.shape[0]
            vis_images = []

            for i in range(batch_size):
                # Get individual samples
                img1_sample = img1_rgb[i] if img1_rgb.ndim == 4 else img1_rgb
                img2_sample = img2_rgb[i] if img2_rgb.ndim == 4 else img2_rgb
                pred_sample = pred_vis[i]
                gt_sample = gt_vis[i]

                # Concatenate horizontally: [img1, img2, pred_map, gt_map]
                combined_vis = np.concatenate(
                    [img1_sample, img2_sample, pred_sample, gt_sample], axis=1
                )
                vis_images.append(combined_vis)
        else:  # Single sample
            # Concatenate horizontally: [img1, img2, pred_map, gt_map]
            vis_images = [
                np.concatenate([img1_rgb, img2_rgb, pred_vis, gt_vis], axis=1)
            ]

        # Stack all visualizations vertically by batch
        if len(vis_images) > 1:
            # Concatenate all samples vertically along the height axis
            img = np.concatenate(vis_images, axis=0)
        else:
            img = vis_images[0]

        # Convert to PIL Image and save
        img = (img * 255.0).astype(np.uint8)
        img_to_save = Image.fromarray(img)

        if add_step:
            img_name = f"{img_name}_step_{str(self.global_step).zfill(6)}.jpg"
        else:
            img_name = f"{img_name}.jpg"

        save_path = Path(self.proj_dir) / "vis" / img_name
        if self.accelerator.is_main_process:
            save_path.parent.mkdir(parents=True, exist_ok=True)
        if self.accelerator.is_main_process:
            img_to_save.save(save_path, quality=95)

        self.log_msg(
            "[Visualize]: save segmentation visualization at {}".format(save_path)
        )

    def run(self):
        if self.train_cfg.resume_path is not None:
            self.resume(self.train_cfg.resume_path)
        elif self.train_cfg.ema_load_path is not None:
            self.load_from_ema(self.train_cfg.ema_load_path)

        # train !
        self.train_loop()


_key = "tokenizer_dinov3_adaptor"  # vq, bsq, fsq, kl
_configs_dict = {
    "tokenizer_dinov3_adaptor": "tokenizer_dinov3_adaptor",
}


if __name__ == "__main__":
    from src.utilities.train_utils.cli import argsparse_cli_args, print_colored_banner

    # change the config name in cli
    cli_default_dict = {
        "config_name": _key,
        "only_rank_zero_catch": True,
    }
    chosen_cfg, cli_args = argsparse_cli_args(_configs_dict, cli_default_dict)
    if is_rank_zero := PartialState().is_main_process:
        print_colored_banner("Change Detection")
    log(
        "<green>\n"
        + "=" * 60
        + "\n"
        + f"[Config]: {chosen_cfg}\n"
        + "\n"
        + "Start Running , Good Luck!\n"
        + "=" * 60
        + "</>",
        only_rank_zero=False,
    )

    # Main function
    @hydra.main(
        config_path="../configs/change_detection",
        config_name=chosen_cfg,
        version_base=None,
    )
    def main(cfg):
        catcher = logger.catch if PartialState().is_main_process else nullcontext

        with catcher():
            trainer = HyperCDTrainer(cfg)
            trainer.run()

    main()

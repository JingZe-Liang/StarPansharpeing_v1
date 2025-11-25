import os
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Generator, Iterable, Literal, TypedDict, cast

import accelerate
import accelerate.utils
import ema_pytorch
import hydra
import numpy as np
import PIL.Image as Image
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.state import PartialState
from accelerate.tracking import TensorBoardTracker
from accelerate.utils import DummyOptim, DummyScheduler  # ty: ignore
from kornia.contrib import combine_tensor_patches, extract_tensor_patches
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch import Tensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._fully_shard import FSDPModule
from torch.distributed.tensor import DTensor
from torchmetrics.aggregation import MeanMetric
from tqdm import tqdm, trange

from src.stage1.cosmos.inference.utils import load_jit_model_shape_matched
from src.stage2.pansharpening.loss import AmotizedPixelLoss
from src.stage2.pansharpening.metrics import AnalysisPanAcc, PansharpeningMetrics
from src.utilities.config_utils import to_object as to_cont
from src.utilities.func.dict_utils import keys_in_dict
from src.utilities.logging import log_print
from src.utilities.network_utils import (
    load_peft_model_checkpoint,
    load_weights_with_shape_check,
)
from src.utilities.network_utils.Dtensor import safe_dtensor_operation
from src.utilities.train_utils import metrics_sync, object_all_gather, object_scatter
from src.utilities.train_utils.state import StepsCounter


class BatchInput(TypedDict):
    lrms: Tensor
    pan: Tensor
    hrms: Tensor
    # NotRequired fields
    lrms_latent: Tensor
    pan_latent: Tensor
    hrms_latent: Tensor


@dataclass
class PansharpeningOutput:
    pred_sr: Tensor
    pred_sr_from_latent: Tensor = None
    pred_latent: Tensor = None
    sr_loss: Tensor = None
    sr_log_losses: dict[str, Tensor] = None


class PansharpeningTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        # Tokenizer configuration is now handled by wrapper
        self.train_cfg = cfg.train
        self.dataset_cfg = cfg.dataset
        self.ema_cfg = cfg.ema
        self.val_cfg = cfg.val

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
        used_dataset = self.dataset_cfg.cfgs.used
        self.log_msg(f"[Data]: using dataset {used_dataset}")
        self.train_dataset, self.train_dataloader = hydra.utils.instantiate(self.dataset_cfg.train)
        self.val_dataset, self.val_dataloader = hydra.utils.instantiate(self.dataset_cfg.val)
        self.val_full_dataset, self.val_full_dataloader = hydra.utils.instantiate(self.dataset_cfg.val_full)
        if _dpsp_plugin is not None:
            self.accelerator.deepspeed_plugin.deepspeed_config[  # type: ignore
                "train_micro_batch_size_per_gpu"
            ] = self.dataset_cfg.batch_size_train

        # setup the pansharpening model
        self.setup_pansharpening_model()  # setup the pansharpening model

        # optimizers and lr schedulers
        self.pansp_optim, self.pansp_sched = self.get_optimizer_lr_scheduler()

        # EMA models and accelerator prepare
        self.prepare_for_training()
        self.prepare_ema_models()

        # loss
        self.pansp_loss: AmotizedPixelLoss | nn.L1Loss | Callable
        if hasattr(self.train_cfg, "pansharpening_loss"):
            self.pansp_loss = hydra.utils.instantiate(self.train_cfg.pansharpening_loss)
        else:
            # default loss
            self.pansp_loss = nn.L1Loss()
            assert not self.pansp_amotizing_pixels, "pansharpening loss must be set in the config"
        self.log_msg(f"use pansharpening loss: {self.pansp_loss.__class__.__name__}")

        # training state counter
        self.train_state = StepsCounter(["train"])

        # clear GPU memory
        torch.cuda.empty_cache()

    def setup_pansharpening_model(self):
        self.pansp_model = hydra.utils.instantiate(self.cfg.pansharp_model)
        # cast to dtype
        self.pansp_model = self.pansp_model.to(dtype=self.dtype)

        # cast to dtype
        # self.pansp_model = self.pansp_model.to(dtype=self.dtype)

        # cast to dtype
        # self.pansp_model = self.pansp_model.to(dtype=self.dtype)

        self.pansp_amotizing_pixels = self.accelerator.unwrap_model(self.pansp_model.downstream_model).amotizing_pixels

        # add processors
        if self.train_cfg.add_pan_processor:
            pan_processor = lambda ms, pan: (
                ms,
                pan.repeat_interleave(ms.shape[1], dim=1),
            )
            self.pansp_model.tokenizer_img_processor = pan_processor

        # checkpoint
        _unwrapped_model = self.accelerator.unwrap_model(self.pansp_model)
        use_checkpoint = hasattr(_unwrapped_model, "set_checkpoint_mode") and self.train_cfg.model_act_checkpoint
        if use_checkpoint:
            _unwrapped_model.set_grad_checkpointing(True)  # type: ignore

        if self.pansp_amotizing_pixels:
            # wrapper will handle tokenization for amotizing pixels
            self.backward_detokenizer = True
        else:
            self.backward_detokenizer = self.train_cfg.backward_detokenizer

        pansp_name = getattr(self.train_cfg, "pansharpening_name", None) or self.pansp_model.__class__.__name__
        self.log_msg(f"use pansharpening model: {pansp_name}, amotizing pixels: {self.pansp_amotizing_pixels}")

    def prepare_ema_models(self):
        if self.no_ema:
            return

        buffer_name = [name for name, _ in self.pansp_model.named_buffers()]
        self.log_msg(f"EMA ignore buffers: {buffer_name}")
        self.ema_pansp_model: ema_pytorch.EMA = hydra.utils.instantiate(self.cfg.ema)(
            model=self.pansp_model, ignore_names=set(buffer_name)
        ).to(self.device)
        self.log_msg(f"create EMA model for pansharpening")

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
            "{time:HH:mm:ss} - {level.icon} <level>[{level}:{file.name}:{line}]</level>- <level>{message}</level>"
        )
        logger.disable("ema_pytorch")
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
                filter=lambda record: record["level"].no <= 20,
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
            self.accelerator.project_configuration.logging_dir = tenb_dir.as_posix()
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
            _n_params_sumed = 0
            if log_type == "grad_norm_sum":
                norms[f"{model_cls_n}_grad_norm"] = 0
            for n, p in model.named_parameters():
                if p.grad is not None:
                    # must sync grad here, `is_main_process` would cause the ranks do not sync
                    if isinstance(p.grad, DTensor):
                        _grad: torch.Tensor = p.grad._local_tensor
                        if p.grad._local_tensor.device == torch.device("cpu"):
                            self.log_msg(
                                "p.grad is on cpu, this should not happen",
                                level="WARNING",
                            )
                            _grad = _grad.cuda()
                        _p_grad = safe_dtensor_operation(p.grad)
                        _grad_norm = (_p_grad.data**2).sum() ** 0.5
                    else:
                        # Handle regular tensor case
                        _grad_norm = (p.grad.data**2).sum() ** 0.5

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
            stack_level=2,
            **kwargs,
        )

        def log_it(*msg, **kwargs):
            msg_string = str_msg(*msg)
            _log_fn(msg_string, **kwargs)

        log_it(*msgs, **kwargs)

    def get_optimizer_lr_scheduler(self):
        # optimizers
        if (
            self.accelerator.state.deepspeed_plugin is None
            or "optimizer" not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):

            def _optimizer_creater(optimizer_cfg, params_getter):
                if "muon" in optimizer_cfg._target_:
                    self.log_msg("[Optimizer]: using muon optimizer")
                    # is muon optimizer function
                    named_params = params_getter(with_name=True)
                    return hydra.utils.instantiate(optimizer_cfg)(named_parameters=named_params)
                else:
                    self.log_msg(f"[Optimizer]: using optimizer: {optimizer_cfg._target_}")
                    params = params_getter(with_name=False)
                    return hydra.utils.instantiate(optimizer_cfg)(params)

            _get_panshap_model_params = (
                lambda with_name: {k: v for k, v in self.pansp_model.named_parameters() if v.requires_grad}
                if with_name
                else [p for p in self.pansp_model.parameters() if p.requires_grad]
            )
            pansp_opt = _optimizer_creater(self.train_cfg.pansharp_optim, _get_panshap_model_params)
        else:
            pansp_opt = DummyOptim([{"params": list(self.pansp_model.parameters())}])

        # schedulers
        if (
            self.accelerator.state.deepspeed_plugin is None
            or "scheduler" not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            pansp_sched = hydra.utils.instantiate(self.train_cfg.pansharp_sched)(optimizer=pansp_opt)
        else:
            pansp_sched = DummyScheduler(pansp_opt)

        # set the heavyball optimizer without torch compiling
        is_heavyball_opt = lambda opt: opt.__class__.__module__.startswith("heavyball")
        if is_heavyball_opt(pansp_opt):
            import heavyball

            heavyball.utils.compile_mode = None

            self.log_msg(
                "use heavyball optimizer, it will compile the optimizer, "
                "for efficience testing the scripts, disable the compilation."
            )

        return pansp_opt, pansp_sched

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

    def prepare_for_training(self):
        # FIXME: FSDP2 seems do not support the sync_bn, find a way to fix it.
        if self._is_fsdp or self.accelerator.distributed_type in (
            accelerate.utils.DistributedType.MULTI_GPU,
            accelerate.utils.DistributedType.FSDP,
        ):  # seems that FSDP does not support synchronized batchnorm
            # discriminator may have batch norm layer
            self.pansp_model = nn.SyncBatchNorm.convert_sync_batchnorm(self.pansp_model)
            # self.log_msg("[Model] convert discriminator to sync batch norm")

        # cast the model to dtype?
        # self.pansp_model = self.pansp_model.to(dtype=self.dtype)

        # if use FSDP2
        if self._is_fsdp and self.accelerator.is_fsdp2:
            # set models with property dtype
            self.pansp_model.dtype = torch.float

        # prepare the model, optimizer, dataloader
        self.pansp_model, self.pansp_optim = self.accelerator.prepare(self.pansp_model, self.pansp_optim)
        self.train_dataloader, self.val_dataloader = self.accelerator.prepare(
            self.train_dataloader, self.val_dataloader
        )
        if hasattr(self, "val_full_dataloader"):
            self.val_full_dataloader = self.accelerator.prepare(self.val_full_dataloader)

    def step_train_state(self):
        self.train_state.update("train")

    def ema_update(self, mode="pansharpening"):
        assert mode == "pansharpening"
        if self.no_ema:
            # not support ema when is deepspeed zero2 or zero3
            return
        self.ema_pansp_model.update()

    def get_global_step(self, mode="train"):
        # TODO: add val state
        assert mode in ("train",), "Only train mode is supported for now."
        return self.train_state[mode]

    @property
    def global_step(self):
        return self.get_global_step("train")

    def may_freeze(self, model, freeze=True):
        model.requires_grad_(not freeze)

    def gradient_check(self, model: nn.Module):
        # check nan gradient
        if self.accelerator.sync_gradients and getattr(self.train_cfg, "grad_check", True):
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
                        torch.nan_to_num(param.grad, nan=0.0, posinf=1e5, neginf=-1e5, out=param.grad)

        # clip gradient by norm
        _max_grad_norm = self.train_cfg.max_grad_norm
        if _max_grad_norm is not None and _max_grad_norm > 0:
            if self.dtype != torch.float16 and not self.accelerator.is_fsdp2:
                self.accelerator.clip_grad_norm_(model.parameters(), _max_grad_norm)
            elif (
                self.accelerator.distributed_type == accelerate.utils.DistributedType.FSDP or self.accelerator.is_fsdp2
            ) and isinstance(model, FSDP):
                FSDP.clip_grad_norm_(model.parameters(), max_norm=_max_grad_norm)

    def forward_pansp_model(
        self,
        lrms,
        pan,
        sr=None,
        lrms_latent=None,
        pan_latent=None,
        sr_latent=None,
        ema=False,
    ):
        pansp_model: nn.Module = (
            self.pansp_model if not ema else self.ema_pansp_model.ema_model  # type: ignore
        )
        with self.accelerator.autocast():
            if self.pansp_amotizing_pixels:
                pixel_in = (lrms, pan)
                latent_in = None
                if lrms_latent is not None and pan_latent is not None:
                    latent_in = (lrms_latent, pan_latent)
                out = pansp_model(pixel_in, latent_in)
                pred_latent = out["latent_out"]
                pred_sr = out["pixel_out"]
                pred_sr_from_latent = out.get("pixel_from_latent", None)

                # print("-- lrms value:", lrms.min().item(), lrms.max().item())
                # print("-- pan value:", pan.min().item(), pan.max().item())
                # print("-- max value of pred_sr:", pred_sr.abs().max().item())

            else:
                # Works at latent space only
                pred_latent = pansp_model(lrms_latent, pan_latent)
                # Use wrapper decoder for pixel output
                if hasattr(pansp_model, "decode"):
                    pred_sr = pansp_model.decode(pred_latent, chans=lrms.shape[1])
                else:
                    # If no decode method, return None for pixel output
                    pred_sr = None
                pred_sr_from_latent = None

            # loss
            sr_loss, sr_log_losses = None, {}
            if sr is not None or sr_latent is not None:
                sr_loss = self.pansp_loss(pred_latent, sr_latent, pred_sr, sr, pred_sr_from_latent, sr)

                if isinstance(sr_loss, torch.Tensor):
                    sr_log_losses = {"pansharp_loss": sr_loss.detach()}
                elif isinstance(sr_loss, (tuple, list)):
                    sr_loss, sr_log_losses = sr_loss
                else:
                    raise NotImplementedError(f"Unknown type {type(sr_loss)}")
            sr_loss = cast(torch.Tensor, sr_loss)

        return PansharpeningOutput(
            pred_latent=pred_latent,
            pred_sr=pred_sr,
            pred_sr_from_latent=pred_sr_from_latent,
            sr_loss=sr_loss,
            sr_log_losses=sr_log_losses,
        )

    def train_pansp_step(
        self,
        # pixels
        lrms,
        pan,
        sr,
        lrms_latent: torch.Tensor | None = None,
        pan_latent: torch.Tensor | None = None,
        sr_latent: torch.Tensor | None = None,
    ):
        out = self.forward_pansp_model(lrms, pan, sr, lrms_latent, pan_latent, sr_latent)

        if self.accelerator.sync_gradients:
            # backward
            self.pansp_optim.zero_grad()
            self.accelerator.backward(out.sr_loss)
            self.gradient_check(self.pansp_model)
            self.pansp_optim.step()
            self.pansp_sched.step()

            # ema update
            self.ema_update(mode="pansharpening")

        return out

    def check_quality(
        self,
        check_fn: Callable[[Tensor, Tensor | None], None],
        pred_sr,
        lms=None,
        pan=None,
        gt=None,
    ):
        pred_sr = self.to_rgb(pred_sr)
        to_rgb_fn = lambda x: self.to_rgb(x) if x is not None else None
        lms, pan, gt = map(to_rgb_fn, (lms, pan, gt))
        check_fn(pred_sr, gt)

    def _cast_to_dtype(self, x: torch.Tensor | list | dict[str, torch.Tensor]):
        if isinstance(x, torch.Tensor):
            return x.to(dtype=self.dtype, device=self.device)
        elif isinstance(x, list):
            return [self._cast_to_dtype(xx) for xx in x]
        elif isinstance(x, dict):
            return {k: self._cast_to_dtype(v) for k, v in x.items()}

    def train_step(self, batch: BatchInput):
        lrms_latent = pan_latent = hrms_latent = None
        batch = self._cast_to_dtype(batch)

        # Get latents from batch - wrapper handles tokenization
        if keys_in_dict(["lrms_latent", "pan_latent", "hrms_latent"], batch):  # type: ignore
            lrms_latent = batch["lrms_latent"].to(self.device, self.dtype)
            pan_latent = batch["pan_latent"].to(self.device, self.dtype)
            hrms_latent = batch["hrms_latent"].to(self.device, self.dtype)
        else:
            # Wrapper expects pixel inputs directly
            lrms, pan, hrms = batch["lrms"], batch["pan"], batch["hrms"]
            # Convert to tokenizer range if needed
            # lrms, pan, hrms = map(self.to_tokenizer_range, (lrms, pan, hrms))
            pan = pan.repeat_interleave(lrms.shape[1], dim=1)

            # Wrapper will handle encoding internally
            # Just pass the pixel data to the model
            lrms_latent = None
            pan_latent = None
            hrms_latent = self.pansp_model.encode(hrms)

        quality_track_n = self.train_cfg.track_metrics_duration
        # start track after n steps
        quality_track_after = self.train_cfg.track_metrics_after
        if quality_track_n >= 0:
            ratio = self.train_cfg.pansp_ratio
            self.pan_acc_reduced = PansharpeningMetrics(ratio=ratio, ref=True, ergas_ratio=ratio)
            self.pan_acc_reduced_latent = PansharpeningMetrics(ratio=ratio, ref=True, ergas_ratio=ratio)

        with self.accelerator.accumulate(self.pansp_model):
            # train pansharpening model
            train_out = self.train_pansp_step(
                batch["lrms"],
                batch["pan"],
                batch["hrms"],
                lrms_latent=lrms_latent,
                pan_latent=pan_latent,
                sr_latent=hrms_latent,
            )
            pred_img = train_out.pred_sr
            logger.trace(f"Step: {self.global_step} - sr_loss: {train_out.sr_loss}")

            # track reconstruction quality
            if (
                quality_track_n >= 0
                and self.global_step % quality_track_n != 0
                and self.global_step >= quality_track_after
            ):
                self.check_quality(self.pan_acc_reduced, pred_sr=pred_img, gt=batch["hrms"])

        self.step_train_state()

        # log losses
        if self.global_step % self.train_cfg.log.log_every == 0:
            _log_tok_losses = self.format_log(train_out.sr_log_losses)

            self.log_msg(
                f"[Train State]: lr {self.pansp_optim.param_groups[0]['lr']:1.4e} | "
                f"[Step]: {self.global_step}/{self.train_cfg.max_steps}"
            )
            self.log_msg(f"[Train Tok]: {_log_tok_losses}")

            # tensorboard log
            self.tenb_log_any("metric", train_out.sr_log_losses, self.global_step)

        if quality_track_n >= 0 and self.global_step % quality_track_n == 0 and self.global_step >= quality_track_after:
            self.log_msg(f"[Real GT Metrics]: {self.pan_acc_reduced.print_str()}")
            self.log_msg(f"[Latent Metrics]: {self.pan_acc_reduced_latent.print_str()}")

        # visualize train
        if self.global_step % self.train_cfg.visualize_every == 0:
            hrms = batch["hrms"]
            self.visualize_reconstruction(
                [hrms, pred_img],
                img_name="train/train_vis",
                add_step=True,
            )

    def format_log(self, log_sr_loss: dict) -> str:
        def dict_round_to_list_str(d: dict, n_round: int = 4, select: list[str] | None = None):
            strings = []
            for k, v in d.items():
                if select is not None and k not in select:
                    continue

                if isinstance(v, (float, torch.Tensor)):
                    if torch.is_tensor(v):
                        if v.numel() > 1:
                            self.log_msg(
                                f'logs has non-scalar tensor "{k}", skip it',
                                level="WARNING",
                            )
                            continue
                        v = v.item()
                    strings.append(f"{k}: {v:.{n_round}f}")
                else:
                    strings.append(f"{k}: {v}")
            return strings

        strings = dict_round_to_list_str(log_sr_loss, select=list(log_sr_loss.keys()))

        return " - ".join(strings)

    def infinity_train_loader(self):
        while True:
            for batch in self.train_dataloader:
                yield batch

    def train_loop(self):
        _stop_train_and_save = False
        self.accelerator.wait_for_everyone()

        self.log_msg("[Train]: start training", only_rank_zero=False)
        for batch in self.infinity_train_loader():
            ### Check the tokenizer is loaded
            # from torchmetrics.functional import peak_signal_noise_ratio as psnr_fn

            # img = batch["hrms"]
            # with torch.no_grad():
            #     latent = self.pansp_model.encode(img)
            #     recon = self.pansp_model.decode(latent, chans=img.shape[1])

            #     recon = (recon + 1) / 2
            #     psnr = psnr_fn(recon, img, data_range=1.0)
            #     print(f"PSNR: {psnr.item():.4f}")
            # continue

            ### train step
            # try:
            self.train_step(batch)
            # except Exception as e:
            #     self.log_msg(
            #         f"Training failed, batch keys are {batch.keys()}. {e}",
            #         level="critical",
            #     )
            #     for k, v in batch.items():
            #         self.log_msg(
            #             f"{k}: {v.shape if hasattr(v, 'shape') else (v.__class__.__name__, v)}, "
            #         )
            #     raise e

            if self.global_step % self.val_cfg.val_duration == 0:
                # torch.cuda.empty_cache()
                self.val_loop()
                # self.val_full_loop()
                # torch.cuda.empty_cache()

            if self.global_step >= self.train_cfg.max_steps:
                _stop_train_and_save = True

            if self.global_step % self.train_cfg.save_every == 0 or _stop_train_and_save:
                self.save_state()
                self.save_ema()

            if _stop_train_and_save:
                self.log_msg("[Train]: max training step budget reached, stop training and save")
                break

    def _ensure_paired_shapes(self, lrms, pan, hrms=None):
        # lrms: is upsampled MS or only MS
        if hrms is not None:
            H, W = hrms.shape[-2:]
        else:
            H, W = pan.shape[-2:]

        ms_shape = H // self.train_cfg.pansp_ratio, W // self.train_cfg.pansp_ratio
        if lrms.shape[-2:] == tuple(ms_shape):
            if getattr(self.train_cfg, "upsample_lrms", True):
                lrms = torch.nn.functional.interpolate(lrms, size=(H, W), mode="bilinear")
            else:
                raise ValueError(f"MS upsampled shape {lrms.shape} does not match MS shape {ms_shape}")
        else:
            assert lrms.shape[-2:] == torch.Size((H, W)), (
                f"lrms shape {lrms.shape} does not match pan/hrms shape {pan.shape}/{hrms.shape}"
            )

        return lrms

    @torch.no_grad()
    def val_step_patches_legacy(self, batch: dict):
        from src.stage2.utilities.patches.patcher_legacy import PatchMergeModule

        batch = self._cast_to_dtype(batch)
        lrms, pan, hrms = batch["lrms"], batch["pan"], batch.get("hrms", None)

        def _model_step_fn(lrms, pan):
            # lrms, pan = map(self.to_tokenizer_range, (lrms, pan))
            pred_sr = self.forward_pansp_model(lrms, pan, None, None, None, None, ema=True).pred_sr
            return pred_sr

        patcher = PatchMergeModule(
            patch_merge_step=_model_step_fn,
            crop_batch_size=self.val_cfg.crop_batch_size,
            patch_size_list=[64, 64],
            scale=1,
            device=self.device,
        )
        # batch size should be 1
        pred_sr = []
        for i in range(lrms.shape[0]):
            lrms_i = lrms[i : i + 1]
            pan_i = pan[i : i + 1]
            pred_sr_i = patcher.forward_chop(lrms_i, pan_i)[0]
            pred_sr.append(pred_sr_i)
        pred_sr = torch.cat(pred_sr, dim=0)
        return PansharpeningOutput(pred_sr=pred_sr)

    @torch.no_grad()
    def val_step_patches(self, batch: BatchInput):
        batch = self._cast_to_dtype(batch)
        lrms, pan, hrms = (
            batch["lrms"],
            batch["pan"],
            batch.get("hrms", None),
        )
        assert self.train_cfg.online_tokenize, "only support online tokenize for val full"
        patcher_kwargs = dict(window_size=64, stride=32)
        lrms_patches, pan_patches = map(partial(extract_tensor_patches, **patcher_kwargs), [lrms, pan])
        assert lrms_patches.shape[1] == pan_patches.shape[1]
        n = lrms_patches.shape[1]
        sr_s = []
        for patch_idx in range(n):
            lrms_i, pan_i = lrms_patches[:, patch_idx], pan_patches[:, patch_idx]
            # now normal val step
            batch_in = {"lrms": lrms_i, "pan": pan_i}
            val_out = self.val_step(batch_in)
            pred_sr_i = val_out.pred_sr
            sr_s.append(pred_sr_i)
        pred_sr = torch.stack(sr_s, dim=1)
        # combine patches
        pred_sr = combine_tensor_patches(pred_sr, original_size=tuple(lrms.shape[-2:]), **patcher_kwargs)
        return PansharpeningOutput(pred_sr)

    @torch.no_grad()
    def val_step(self, batch: BatchInput):
        batch = self._cast_to_dtype(batch)
        lrms, pan, hrms = (
            batch["lrms"],
            batch["pan"],
            batch.get("hrms", None),
        )
        lrms_latent, pan_latent, gt_latent = None, None, None
        if hrms is not None:
            gt_latent = self.pansp_model.encode(hrms)

        # forward the fusion network
        out = self.forward_pansp_model(lrms, pan, hrms, lrms_latent, pan_latent, gt_latent, ema=True)

        return out

    def _get_val_tbar_iter(self, mode="reduced") -> tqdm | Generator[BatchInput, None, None]:
        max_iters = getattr(self.val_cfg, f"max_val_{mode}_iters")
        val_loader = self.val_dataloader if mode == "reduced" else self.val_full_dataloader
        if max_iters > 0:
            # Create a generator that limits the number of iterations
            def _val_loader():
                val_loader_iter = iter(val_loader)
                tbar = trange(
                    max_iters,
                    desc="validating ...",
                    leave=False,
                    disable=not self.accelerator.is_main_process,
                )

                for _ in tbar:
                    try:
                        yield next(val_loader_iter)
                    except StopIteration:
                        # Recreate the iterator when exhausted
                        val_loader_iter = iter(val_loader)
                        yield next(val_loader_iter)

            self.log_msg(
                f"[Val]: start validating with only {max_iters} batches",
                only_rank_zero=False,
            )
            return _val_loader()
        else:
            # Use the full validation set
            self.log_msg(f"[Val]: start validating with the whole val set", only_rank_zero=False)
            return tqdm(
                val_loader,
                desc="validating ...",
                leave=False,
                disable=not self.accelerator.is_main_process,
            )

    @torch.inference_mode()
    def val_loop(self):
        self.pansp_model.eval()

        tbar = self._get_val_tbar_iter(mode="reduced")
        # track psnr and ssim
        pan_acc_fn = AnalysisPanAcc(ratio=self.train_cfg.pansp_ratio, ref=True)
        # pan_acc_fn = PansharpeningMetrics(ratio=self.train_cfg.pansp_ratio, ref=True)
        loss_metrics = MeanMetric().to(device=self.device)

        self._val_reduced_loader_iter: Iterable
        for batch in tbar:
            batch = cast(BatchInput, batch)
            gt = batch["hrms"].to(self.device)
            if self.val_cfg.val_use_patches:
                self.log_msg("use patches for val step", level="DEBUG")
                # val_out = self.val_step_patches(batch)
                val_out = self.val_step_patches_legacy(batch)
            else:
                val_out = self.val_step(batch)

            # l1 loss
            pred_sr = val_out.pred_sr
            loss_of_latent = nn.functional.l1_loss(pred_sr, gt.to(pred_sr))  # latent to img and then loss
            loss_metrics.update(loss_of_latent)

            # metrics
            pan_acc_fn(self.to_rgb(pred_sr), self.to_rgb(gt))

        # pan_acc = pan_acc_fn._get_acc_ave()
        pan_acc = pan_acc_fn.acc_ave
        loss_val = loss_metrics.compute()

        if self.accelerator.is_main_process:
            _metric_str = ""
            for k, v in pan_acc.items():
                _metric_str += f"{k}: {v:.4f} - "
            self.log_msg(f"[Val]: {_metric_str} loss: {loss_val:.4f}")
            self.tenb_log_any("metric", pan_acc, step=self.global_step)

            # visualize the last val batch
            self.visualize_reconstruction(
                [gt, val_out.pred_sr],
                add_step=True,
                img_name="val/pansharpened_reduced",
            )

    def val_full_loop(self):
        self.pansp_model.eval()

        tbar = self._get_val_tbar_iter(mode="full")
        # track psnr and ssim
        pan_acc_fn = PansharpeningMetrics(
            ratio=self.train_cfg.pansp_ratio,
            sensor=self.dataset_cfg.cfgs.used,
            ref=False,
        )
        loss_metrics = MeanMetric().to(device=self.device)

        self._val_full_loader_iter: Iterable
        for batch in tbar:
            batch = cast(BatchInput, batch)
            if self.val_cfg.val_use_patches:
                # val_out = self.val_step_patches(batch)
                val_out = self.val_step_patches_legacy(batch)
            else:
                val_out = self.val_step(batch)

            acc_fn_inputs = map(
                self.to_rgb,
                [
                    val_out.pred_sr,
                    batch["lrms"].to(self.device),
                    batch["pan"].to(self.device),
                ],
            )
            pan_acc_fn(*list(acc_fn_inputs))

        pan_acc = pan_acc_fn._get_acc_ave()
        loss_val = loss_metrics.compute()

        if self.accelerator.is_main_process:
            _metric_str = ""
            for k, v in pan_acc.items():
                _metric_str += f"{k}: {v:.4f} - "
            self.log_msg(f"[Val]: {_metric_str} loss: {loss_val:.4f}")
            self.tenb_log_any("metric", pan_acc, step=self.global_step)

            # visualize the last val batch with pan, lrms, pred_sr, and gt for comparison
            self.visualize_reconstruction(
                [batch["pan"], batch["lrms"], val_out.pred_sr],
                add_step=True,
                img_name="val/full",
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
            ema_path.mkdir(parents=True, exist_ok=True)
            (ema_path / "pansharp_model").mkdir(parents=True, exist_ok=True)

        # for accelerate dir loading
        self.accelerator.save_model(self.ema_pansp_model.ema_model, ema_path / "pansharp_model")
        # train state
        _ema_path_state_train = ema_path / "train_state.pth"
        _ema_path_state_train.parent.mkdir(parents=True, exist_ok=True)
        accelerate.utils.save(self.train_state.state_dict(), _ema_path_state_train)
        self.log_msg(f"[Ckpt]: save ema at {ema_path}")

    def load_from_ema(self, ema_path: str | Path, strict: bool = False):
        ema_path = Path(ema_path)

        try:
            accelerate.load_checkpoint_in_model(self.pansp_model, ema_path / "pansharp_model", strict=strict)
        except Exception as e:
            logger.warning(f"[Load EMA]: {e}")
            from safetensors.torch import load_file

            load_weights_with_shape_check(
                self.pansp_model,
                load_file(str(ema_path / "pansharp_model" / "model.safetensors")),
            )

        # Prepare models
        self.prepare_ema_models()  # This will update EMA models with online models' weights

        # clear the accelerator model registration
        self.log_msg(f"[Load EMA]: clear the accelerator registrations and re-prepare training")

    def resume(self, path: str):
        self.log_msg("[Resume]: resume training")
        self.accelerator.load_state(path)
        self.accelerator.wait_for_everyone()

    def to_rgb(self, x):
        if self.train_cfg.to_neg_1_1:  # [-1, 1] -> [0, 1]
            return ((x + 1) / 2).clamp(0, 1).float()
        return x.float()

    def to_tokenizer_range(self, x):
        if not self.train_cfg.to_neg_1_1:
            # to (-1, 1)
            return x * 2 - 1
        return x

    def visualize_reconstruction(
        self,
        tensors: torch.Tensor | list[torch.Tensor],
        img_name: str = "pansharpening_reduced",
        add_step: bool = False,
        only_vis_n: int | None = None,
        no_to_rgb: bool = False,
        use_linstretch: bool = True,
        linstretch_tol: list[float] | None = None,
    ):
        """Visualize reconstruction results using the enhanced batch comparison function.

        This function now uses the optimized visualize_batch_comparisons_imgs function
        which supports linear stretching for better contrast enhancement.

        Parameters
        ----------
        tensors : torch.Tensor | list[torch.Tensor]
            Input tensors to visualize.
        img_name : str, optional
            Base name for the saved image. Defaults to "pansharpening_reduced".
        add_step : bool, optional
            Whether to add training step to filename. Defaults to False.
        only_vis_n : int | None, optional
            Number of images to visualize. Defaults to None (uses 8).
        no_to_rgb : bool, optional
            Whether to skip RGB conversion. Defaults to False.
        use_linstretch : bool, optional
            Whether to apply linear stretching for contrast enhancement. Defaults to True.
        linstretch_tol : list[float] | None, optional
            Tolerance values for linear stretching. Defaults to None.
        """
        # Import the enhanced visualization function
        from src.utilities.train_utils.visualization import (
            visualize_batch_comparisons_imgs,
        )

        # Convert single tensor to list for uniform processing
        if isinstance(tensors, torch.Tensor):
            tensors = [tensors]

        # Apply RGB conversion if needed
        processed_tensors = []
        for tensor in tensors:
            if not no_to_rgb:
                tensor = self.to_rgb(tensor)
            processed_tensors.append(tensor)

        # Limit number of images to visualize
        _only_n = only_vis_n or 8
        limited_tensors = [tensor[:_only_n] for tensor in processed_tensors]

        # Get RGB channels configuration for hyperspectral data
        rgb_channels = None
        if hasattr(self.dataset_cfg, "consts") and hasattr(self.dataset_cfg.consts, "rgb_channels"):
            rgb_channels = to_cont(self.dataset_cfg.consts.rgb_channels)
            # Ensure rgb_channels is the correct type
            if isinstance(rgb_channels, list):
                pass  # Already correct format
            elif isinstance(rgb_channels, tuple):
                rgb_channels = list(rgb_channels)  # Convert tuple to list
            elif isinstance(rgb_channels, str):
                pass  # String format is also acceptable
            else:
                rgb_channels = None  # Fallback to automatic selection

        # Use the enhanced visualization function with linear stretching
        img = visualize_batch_comparisons_imgs(
            *limited_tensors,
            rgb_channels=rgb_channels,
            norm=False,  # Don't apply additional normalization since we already applied to_rgb
            spacing=5,  # 5 pixels spacing between images
            to_uint8=True,
            to_pil=True,  # We'll handle PIL conversion ourselves
            to_grid=True,  # We want horizontal concatenation
            use_linstretch=use_linstretch,
            linstretch_tol=linstretch_tol,
        )

        # Save
        if isinstance(img, np.ndarray):
            if img.dtype == np.uint8:
                img_to_save = Image.fromarray(img)
            else:
                img_to_save = Image.fromarray((img * 255.0).astype(np.uint8))
        elif isinstance(img, Image.Image):
            img_to_save = img
        elif torch.is_tensor(img):
            # Handle tensor case
            img_array = img.cpu().numpy()
            if img_array.dtype != np.uint8:
                img_array = (img_array * 255.0).astype(np.uint8)
            img_to_save = Image.fromarray(img_array)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

        if add_step:
            img_name = f"{img_name}_step_{str(self.global_step).zfill(6)}.jpg"
        else:
            img_name = f"{img_name}.jpg"

        save_path = Path(self.proj_dir) / "vis" / img_name
        if self.accelerator.is_main_process:
            save_path.parent.mkdir(parents=True, exist_ok=True)
        if self.accelerator.is_main_process:
            img_to_save.save(save_path, quality=95)

        self.log_msg("[Visualize]: save visualization at {}".format(save_path))

    def run(self):
        if self.train_cfg.resume_path is not None:
            self.resume(self.train_cfg.resume_path)
        elif self.train_cfg.ema_load_path is not None:
            self.load_from_ema(self.train_cfg.ema_load_path)

        # train !
        self.train_loop()


_key = "tokenizer_lora_nafnet"
_configs = {"tokenizer_lora_nafnet": "pansharp_wrapper_nafnet_cosmos"}

if __name__ == "__main__":
    from src.utilities.train_utils.cli import argsparse_cli_args, print_colored_banner

    # change the config name in cli
    cli_default_dict = {
        "config_name": _key,
        "only_rank_zero_catch": True,
    }

    chosen_cfg, cli_args = argsparse_cli_args(_configs, cli_default_dict)

    log_print(f"[Train]: using config {chosen_cfg} with CLI args {cli_args}")
    log_print(f"Hydra configs are {sys.argv[1:]}")
    if is_rank_zero := PartialState().is_main_process:
        print_colored_banner("HyperSharpening")

    @hydra.main(
        config_path="../configs/pansharpening",
        config_name=chosen_cfg,
        version_base=None,
    )
    def main(cfg):
        catcher = logger.catch if is_rank_zero else nullcontext

        with catcher():
            trainer = PansharpeningTrainer(cfg)
            trainer.run()

    main()

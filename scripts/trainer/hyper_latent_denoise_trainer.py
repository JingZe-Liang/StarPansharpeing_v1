import sys
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Callable, Literal, NamedTuple, Sequence, cast

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
from accelerate.utils import DummyOptim, DummyScheduler
from ema_pytorch import EMA
from kornia.utils.image import make_grid, tensor_to_image
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._fully_shard import FSDPModule
from torch.distributed.tensor import DTensor
from torchmetrics.aggregation import MeanMetric
from tqdm import trange

from src.stage2.denoise.metrics import DenoisingMetrics
from src.stage2.pansharpening.loss import AmotizedPixelLoss
from src.stage2.pansharpening.metrics import AnalysisPanAcc
from src.utilities.config_utils import (
    to_object as to_cont,  # register new resolvers at the same time
)
from src.utilities.logging import dict_round_to_list_str
from src.utilities.network_utils import load_peft_model_checkpoint
from src.utilities.network_utils.Dtensor import safe_dtensor_operation
from src.utilities.train_utils.state import StepsCounter, dict_tensor_sync, metrics_sync
from src.utilities.train_utils.visualization import visualize_hyperspectral_image

heavyball = lazy_loader.load("heavyball")


class TrainDenoisingStepOutput(NamedTuple):
    pred_latent: Tensor
    pred_pixel: Tensor
    denoise_loss: Tensor
    log_losses: dict[str, Tensor]


class DenoisingTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
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

        # setup the denoising model
        self.setup_denoising_model()  # setup the denoising model

        # optimizers and lr schedulers
        self.optim, self.sched = self.get_optimizer_lr_scheduler()

        # EMA models and accelerator prepare
        self.prepare_for_training()
        self.prepare_ema_models()

        # loss
        self.denoising_loss: AmotizedPixelLoss | nn.L1Loss | Callable
        if hasattr(self.train_cfg, "denoising_loss"):
            self.denoising_loss = hydra.utils.instantiate(self.train_cfg.denoising_loss)
        else:
            # default loss
            self.denoising_loss = nn.L1Loss()
            assert not self.denoising_amotizing_pixels, (
                "denoising loss must be set in the config"
            )
        self.log_msg(f"use denoising loss: {self.denoising_loss.__class__.__name__}")

        # training state counter
        self.train_state = StepsCounter(["train", "val"])

    def setup_denoising_model(self):
        self.model = hydra.utils.instantiate(self.cfg.denoising_model)
        self.denoising_amotizing_pixels = self.accelerator.unwrap_model(
            self.model
        ).amotizing_pixels
        if self.denoising_amotizing_pixels:
            # Wrapper will handle tokenization for amotizing pixels
            self.backward_detokenizer = True
        else:
            self.backward_detokenizer = self.train_cfg.backward_detokenizer

        denoising_name = (
            getattr(self.train_cfg, "denoising_name", None)
            or self.model.__class__.__name__
        )
        self.log_msg(
            f"use denoising model: {denoising_name}, amotizing pixels: {self.denoising_amotizing_pixels}"
        )

    def prepare_ema_models(self):
        if self.no_ema:
            return

        self.ema_model = EMA(
            self.model,
            beta=self.ema_cfg.beta,
            update_every=self.ema_cfg.update_every,
        ).to(self.device)
        self.log_msg(f"create EMA model for denoising")

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
            if self.accelerator.is_main_process:
                input_lst = [log_file] * self.accelerator.num_processes
            else:
                input_lst = [None] * self.accelerator.num_processes
            output_lst = [None]
            torch.distributed.scatter_object_list(output_lst, input_lst, src=0)
            log_file: Path = output_lst[0]
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
            "- {level.icon} <level>[{level}:{file.name}:{line}]</level>"
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
        self.logger.add(
            sys.stdout,
            format=log_format_in_cmd,
            level="DEBUG",
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

    def log_msg(self, *msgs, only_rank_zero=True, level="INFO", sep=",", **kwargs):
        assert level.lower() in [
            "info",
            "warning",
            "error",
            "debug",
            "critical",
        ], f"Unknown level {level}"

        def str_msg(*msg):
            return sep.join([str(m) for m in msg])

        log_fn = getattr(self.logger, level.lower())

        if only_rank_zero:
            if self.accelerator.is_main_process:
                log_fn(str_msg(*msgs), **kwargs)
        else:  # not only rank zero
            with self.accelerator.main_process_first():
                msg_string = str_msg(*msgs)
                # prefix rank info
                msg_string = f"rank-{self.accelerator.process_index} | {msg_string}"
                log_fn(msg_string, **kwargs)

    def get_optimizer_lr_scheduler(
        self,
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        # optimizers
        if (
            self.accelerator.state.deepspeed_plugin is None
            or "optimizer"
            not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):

            def _optimizer_creater(optimizer_cfg, params_getter):
                if "get_muon_optimizer" in optimizer_cfg._target_:
                    self.log_msg("[Optimizer]: using muon optimizer")
                    # is muon optimizer function
                    named_params = params_getter(with_name=True)
                    return hydra.utils.instantiate(optimizer_cfg)(
                        named_parameters=named_params
                    )
                else:
                    self.log_msg(
                        f"[Optimizer]: using optimizer: {optimizer_cfg._target_}"
                    )
                    params = params_getter(with_name=False)
                    return hydra.utils.instantiate(optimizer_cfg)(params)

            _get_model_params = (
                lambda with_name: self.model.named_parameters()
                if with_name
                else self.model.parameters()
            )
            model_opt = _optimizer_creater(
                self.train_cfg.denoise_optim, _get_model_params
            )
        else:
            model_opt = DummyOptim([{"params": list(self.model.parameters())}])

        # schedulers
        if (
            self.accelerator.state.deepspeed_plugin is None
            or "scheduler"
            not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            model_sched = hydra.utils.instantiate(self.train_cfg.denoise_sched)(
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

    def prepare_for_training(self):
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

        self.train_dataloader, self.val_dataloader = self.accelerator.prepare(
            self.train_dataloader, self.val_dataloader
        )

    def step_train_state(self, mode="train"):
        self.train_state.update(mode)

    def ema_update(self, mode="denoising"):
        assert mode == "denoising"
        if self.no_ema:
            # not support ema when is deepspeed zero2 or zero3
            return

        self.ema_model.update()

    def get_training_sample_channels(self):
        bands: int = getattr(self, "_processed_bands", self.dataset_cfg.consts.bands)
        assert bands is not None and bands.is_integer() and bands > 0, (
            f"channel num: {bands}"
        )
        return bands

    def forward_tokenizer(
        self, x: torch.Tensor, mode: str = "encode", no_grad=True
    ) -> dict:
        """
        Forward through tokenizer using wrapper functionality.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (image for encode, latent for decode)
        mode : str
            Either 'encode' or 'decode'
        no_grad : bool
            Whether to use no_grad context

        Returns
        -------
        dict
            Dictionary containing 'latent' (for encode) or 'recon' (for decode)
        """
        grad_ctx = torch.no_grad() if no_grad else nullcontext()

        with grad_ctx and self.accelerator.autocast():
            if mode == "encode":
                # Use wrapper encode method
                latent = self.model.encode(x)
                return dict(latent=latent, recon=None)
            elif mode == "decode":
                # Use wrapper decode method
                bands = self.get_training_sample_channels()
                recon = self.model.decode(x, bands)
                if isinstance(recon, tuple):
                    recon = recon[0]  # construction is the first output
                return dict(latent=None, recon=recon)
            else:
                raise ValueError(f"Unsupported mode: {mode}")

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

    def may_freeze(self, model, freeze=True):
        for p in model.parameters():
            p.requires_grad = not freeze

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

    def forward_denoising_model(
        self,
        noisy: torch.Tensor,
        gt: torch.Tensor,
        noisy_latent: torch.Tensor | None,
        gt_latent: torch.Tensor | None,
    ) -> TrainDenoisingStepOutput:
        """
        Forward pass through the denoising model.

        Parameters
        ----------
        noisy : torch.Tensor
            Input noisy image
        gt : torch.Tensor
            Ground truth clean image
        noisy_latent : torch.Tensor | None
            Latent representation of noisy image
        gt_latent : torch.Tensor | None
            Latent representation of ground truth image

        Returns
        -------
        TrainDenoisingStepOutput
            Model output containing predictions and losses
        """
        with self.accelerator.autocast():
            if self.denoising_amotizing_pixels:
                out = self.model(noisy, noisy_latent)
                pred_latent = out["latent_out"]
                pred = out["pixel_out"]
                pred_from_latent = out.get("pixel_from_latent", None)
            else:
                pred_latent = self.model(noisy_latent)
                # Use wrapper decode method
                bands = self.get_training_sample_channels()
                pred = self.model.decode(pred_latent, bands)
                if isinstance(pred, tuple):
                    pred = pred[0]  # construction is the first output
                pred_from_latent = None

            # loss
            loss, log_losses = None, {}
            if gt is not None or gt_latent is not None:
                loss = self.denoising_loss(
                    pred_latent, gt_latent, pred, gt, pred_from_latent, gt
                )
                if isinstance(loss, torch.Tensor):
                    log_losses = {"denoise_loss": loss.detach()}
                elif isinstance(loss, (tuple, list)):
                    loss, log_losses = loss
                else:
                    raise NotImplementedError(f"Unknown type {type(loss)}")

        out = TrainDenoisingStepOutput(pred_latent, pred, loss, log_losses)
        return out

    def train_denoise_step(
        self,
        # pixels
        noisy,
        gt,
        # optional for *tok_out
        noisy_tok_out: dict | None = None,
        gt_tok_out: dict | None = None,
        # latents
        noisy_latent: torch.Tensor | None = None,
        gt_latent: torch.Tensor | None = None,
    ):
        if noisy_tok_out is not None and gt_tok_out is not None:
            noisy_latent = noisy_tok_out["latent"].detach()
            gt_latent = gt_tok_out["latent"].detach()

        out = self.forward_denoising_model(noisy, gt, noisy_latent, gt_latent)

        if self.accelerator.sync_gradients:
            # backward
            self.optim.zero_grad()
            self.accelerator.backward(out.denoise_loss)
            self.gradient_check(self.model)
            self.optim.step()
            self.sched.step()

            # ema update
            self.ema_update(mode="denoising")

        return out

    def _get_metric_fn(self, clear=False):
        if not hasattr(self, "_denoising_metrics"):
            self._denoising_metrics = DenoisingMetrics().to(self.device)
            if clear:
                self._denoising_metrics.reset()
        return self._denoising_metrics

    def update_metrics(self, noisy, gt, clear=False, only_update=True):
        metrics_fn = self._get_metric_fn(clear)
        metrics_fn(noisy, gt)

    def get_metrics(self):
        assert hasattr(self, "_denoising_metrics")
        return self._denoising_metrics.compute()

    def train_step(self, batch: dict):
        """
        Execute one training step for denoising model.

        Parameters
        ----------
        batch : dict
            Batch input containing noisy and clean images, and optionally latents

        Note
        ----
        The wrapper handles tokenization internally. If latents are not provided in the batch,
        the wrapper will encode the images to latents automatically.
        """
        noisy_latent = gt_latent = None

        # Get latents from batch - wrapper handles tokenization
        if "noisy_latent" in batch:
            noisy_latent = batch["noisy_latent"].to(self.device, self.dtype)
            gt_latent = batch["gt_latent"].to(self.device, self.dtype)

        with self.accelerator.accumulate(self.model):
            # train denoising model
            train_out = self.train_denoise_step(
                batch["noisy"],
                batch["gt"],
                # latents (None if not provided, wrapper will handle encoding)
                None,
                None,
                # offline latents if available
                noisy_latent,
                gt_latent,
            )

        self.step_train_state()

        # log losses
        if self.global_step % self.train_cfg.log.log_every == 0:
            _log_losses = self.format_log(train_out.log_losses)

            self.log_msg(
                f"[Train State]: lr {self.optim.param_groups[0]['lr']:1.4e} | "
                f"[Step]: {self.global_step}/{self.train_cfg.max_steps}"
            )
            self.log_msg(f"[Train Denoising]: {_log_losses}")

            # tensorboard log
            self.tenb_log_any("metric", train_out.log_losses, self.global_step)

    def format_log(self, log_loss: dict, sync=False) -> str:
        if sync:
            log_loss = dict_tensor_sync(log_loss)
        strings = dict_round_to_list_str(log_loss, select=list(log_loss.keys()))
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
            # train step
            self.train_step(batch)

            if self.global_step % self.val_cfg.val_duration == 0:
                self.val_loop()

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

    def _finite_val_loader(self):
        if self.val_dataloader is None:
            raise ValueError("No validation dataloader found")

        for batch in self.val_dataloader:
            yield batch

    def get_val_loader_iter(self):
        if self.val_cfg.max_val_iters > 0:
            # create a new iterator for the validation loader
            # state in the loader generator
            if not hasattr(self, "_val_loader_iter"):
                self._val_loader_iter = iter(self._finite_val_loader())

            iterable_ = trange(
                self.val_cfg.max_val_iters,
                desc="validating ...",
                leave=False,
                disable=not self.accelerator.is_main_process,
            )
            self.log_msg(
                f"[Val]: start validating with only {self.val_cfg.max_val_iters} batches",
                only_rank_zero=False,
            )
        else:
            iterable_ = self._finite_val_loader()
            self.log_msg(
                f"[Val]: start validating with the whole val set", only_rank_zero=False
            )

        return iterable_

    @torch.no_grad()
    def val_step(self, batch: dict):
        """
        Execute one validation step for denoising model.

        Parameters
        ----------
        batch : dict
            Batch input containing noisy and clean images, and optionally latents

        Returns
        -------
        dict
            Dictionary containing predictions

        Note
        ----
        The wrapper handles tokenization internally. If latents are not provided in the batch,
        the wrapper will encode the images to latents automatically.
        """
        noisy_latent = gt_latent = None

        # Get latents from batch - wrapper handles tokenization
        if "noisy_latent" in batch:
            noisy_latent = batch["noisy_latent"].to(self.device, self.dtype)
            gt_latent = batch["gt_latent"].to(self.device, self.dtype)

        # forward the fusion network
        pred_latent, pred_img, *_ = self.forward_denoising_model(
            batch["noisy"], batch["gt"], noisy_latent, gt_latent
        )

        return {"pred_img": pred_img, "pred_latent": pred_latent}

    def val_loop(self):
        self.model.eval()
        # torch.cuda.empty_cache()

        loss_metrics = MeanMetric().to(device=self.device)
        val_iter = self.get_val_loader_iter()
        for batch_or_idx in val_iter:  # type: ignore
            if self.val_cfg.max_val_iters > 0:
                batch = next(self._val_loader_iter)
            else:
                batch = batch_or_idx

            batch = cast(dict[str, torch.Tensor], batch)
            gt = batch["gt"].to(self.device)
            val_out = self.val_step(batch)

            pred_img_rgb = self.to_rgb(val_out["pred_img"])
            batch_img_rgb = self.to_rgb(gt)

            # l1 loss
            loss_of_latent = nn.functional.mse_loss(pred_img_rgb, gt.to(pred_img_rgb))
            loss_metrics.update(loss_of_latent)

            # metrics
            self.update_metrics(batch_img_rgb, pred_img_rgb)
            self.step_train_state("val")

        metrics = self.get_metrics()
        loss_val = loss_metrics.compute()

        if self.accelerator.is_main_process:
            _metric_str = ""
            for k, v in metrics.items():
                _metric_str += f"{k}: {v:.4f} - "
            self.log_msg(f"[Val]: {_metric_str} loss: {loss_val:.3e}")
            self.tenb_log_any("metric", metrics, step=self.global_step)

            # visualize the last val batch
            self.visualize_reconstruction(
                batch_img_rgb,  # gt
                pred_img_rgb,  # prediction
                add_step=True,
                img_name="val/denoising",
                no_to_rgb=True,
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

        self.accelerator.save_model(
            self.ema_model.ema_model, ema_path / "denoise_ema_model"
        )
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
        if self.train_cfg.is_neg_1_1:
            return ((x + 1) / 2).clamp(0, 1).float()
        return x.clamp(0, 1).float()

    def visualize_reconstruction(
        self,
        x: torch.Tensor,
        recon: torch.Tensor | None = None,
        img_name: str = "train_original_recon",
        add_step: bool = False,
        only_vis_n: int | None = None,
        no_to_rgb: bool = False,
    ):
        _only_n = only_vis_n or 16

        def hyperspectral_to_rgb_fn(x):
            x_np = visualize_hyperspectral_image(
                x[_only_n],
                rgb_channels=self.train_cfg.visualize_rgb_channels,
                to_uint8=True,
                to_grid=True,
                nrows=4,
            )
            x_np = cast(np.ndarray, x_np)
            return x_np

        if not no_to_rgb:
            x = self.to_rgb(x)
        if recon is not None and not no_to_rgb:
            recon = self.to_rgb(recon)

        x_np = hyperspectral_to_rgb_fn(x)
        # cat original and reconstructed images
        if recon is not None:
            recon_np = hyperspectral_to_rgb_fn(recon)
            img = np.concatenate([x_np, recon_np], axis=1)
        else:
            img = x_np

        # save
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

        self.log_msg("[Visualize]: save visualization at {}".format(save_path))

    def run(self):
        if self.train_cfg.resume_path is not None:
            self.resume(self.train_cfg.resume_path)
        elif self.train_cfg.ema_load_path is not None:
            self.load_from_ema(self.train_cfg.ema_load_path)

        # train !
        self.train_loop()


_key = "cosmos_f8c16p4_fusionnet"  # vq, bsq, fsq, kl
_configs_dict = {
    # cosmos tokenizer, simple denoising
    "cosmos_f8c16p4_fusionnet": "cosmos_f8c16p4_fusionnet",
}[_key]


@hydra.main(
    config_path="configs/denoising",
    config_name=_configs_dict,
    version_base=None,
)
def main(cfg):
    catcher = logger.catch if PartialState().is_main_process else nullcontext

    with catcher():
        trainer = DenoisingTrainer(cfg)
        trainer.run()


if __name__ == "__main__":
    main()

import math
import sys
import time
from collections import namedtuple
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Literal, Sequence, cast

import accelerate
import hydra
import lazy_loader
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.state import PartialState
from accelerate.tracking import TensorBoardTracker
from accelerate.utils import DummyOptim, DummyScheduler
from einops import rearrange
from ema_pytorch import EMA
from jaxtyping import Float, Int
from kornia.utils.image import make_grid, tensor_to_image
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._fully_shard import FSDPModule
from torch.distributed.tensor import DTensor
from torchmetrics.aggregation import MeanMetric
from tqdm import trange

from src.stage1.cosmos.inference.utils import load_jit_model_shape_matched
from src.stage2.detections.data.HAD100 import HADDataset as HAD100Dataset
from src.stage2.detections.metrics import HADDetectionMetrics
from src.stage2.detections.metrics.visualizer import AnomalyDetectionVisualizer
from src.utilities.config_utils import (
    to_object as to_cont,  # register new resolvers at the same time
)
from src.utilities.logging import dict_round_to_list_str, log
from src.utilities.network_utils import load_peft_model_checkpoint
from src.utilities.network_utils.Dtensor import safe_dtensor_operation
from src.utilities.train_utils.state import StepsCounter, dict_tensor_sync, metrics_sync
from stage2.detections.loss.msgms_loss import MSGMSLoss


@dataclass
class HADModelStepOutput:
    recon: Tensor
    loss: Tensor
    log_losses: dict


class HyperHADTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.tokenizer_cfg = cfg.tokenizer
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
        self._use_single_img_indexing = False
        self.log_msg("Using standard HAD loss computation")

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

        # setup the tokenizer
        self.online_tokenize = self.train_cfg.online_tokenize
        self.setup_tokenizer()  # must setup the tokenizer to decode the image
        self.setup_detection_model()  # setup the detection model

        # optimizers and lr schedulers
        self.optim, self.sched = self.get_optimizer_lr_scheduler()

        # EMA models and accelerator prepare
        self.prepare_for_training()
        self.prepare_ema_models()

        # loss
        self.had_loss: nn.Module | MSGMSLoss
        if hasattr(self.train_cfg, "had_loss"):
            self.had_loss = hydra.utils.instantiate(self.train_cfg.had_loss)
        else:
            # default loss for anomaly detection
            self.had_loss = nn.MSELoss()
        self.log_msg(f"use HAD loss: {self.had_loss.__class__.__name__}")

        # training state counter
        self.train_state = StepsCounter(["train", "val"])

        # clear GPU memory
        torch.cuda.empty_cache()

    def setup_detection_model(self):
        self.model = hydra.utils.instantiate(self.cfg.detect_model)

        detection_name = (
            getattr(self.train_cfg, "detection_name", None)
            or self.model.__class__.__name__
        )
        self.log_msg(f"use detection model: {detection_name}")

    def setup_tokenizer(self):
        tokenizer_name = self.train_cfg.tokenizer_name
        self.sep_enc_dec = self.train_cfg.seperate_enc_dec
        self.quantizer_type: str | None = self.train_cfg.quantizer_type
        self.log_msg(f"[Tokenizer] tokenizer name: {tokenizer_name}]")
        self.log_msg(f"[Train Tokenizer Setter]: quantizer_type={self.quantizer_type}")

        if self.train_cfg.seperate_enc_dec:
            self.log_msg(
                "[Tokenizer]: use pretrained cosmos tokenizer with seperate encoder and decoder"
            )
            tokenizer_config = to_cont(self.tokenizer_cfg.config)
            self.tokenizer_encoder, self._enc_model_mody_keys = (
                load_jit_model_shape_matched(
                    self.cfg.tokenizer.enc_path,
                    tokenizer_config,
                    device=self.device,
                    part="encoder",
                )
            )
            self.tokenizer_decoder, self._dec_model_mody_keys = (
                load_jit_model_shape_matched(
                    self.cfg.tokenizer.dec_path,
                    tokenizer_config,
                    device=self.device,
                    part="decoder",
                )
            )
            self.tokenizer_encoder: nn.Module
            self.tokenizer_decoder: nn.Module

            # quantizer
            if self.cfg.quantizer.quant is not None:
                self.quantizer = hydra.utils.instantiate(self.cfg.quantizer.quant).to(
                    self.device
                )
            elif hasattr(self.tokenizer, "quantizer"):
                self.quantizer = self.tokenizer.quantizer
            else:
                self.quantizer = None

            self.use_quantizer = self.quantizer is not None

            if (
                self.use_quantizer
                and isinstance(self.quantizer, nn.Module)
                and len(list(self.quantizer.parameters())) > 0
            ):
                self.log_msg("[Quantizer]: quantizer has parameters")

        else:
            self.log_msg(
                "[Tokenizer]: Use encoder, decoder, and quantizer in one class"
            )
            self.norm_z = False  # in the model, not in trainer
            self.tokenizer = hydra.utils.instantiate(self.cfg.tokenizer)

            if self.train_cfg.peft_pretrained_path is not None:
                self.log_msg(
                    f"[Tokenizer]: load peft model from {self.train_cfg.peft_pretrained_path}"
                )
                self.peft_cfg, self.tokenizer = load_peft_model_checkpoint(
                    base_model=self.tokenizer,
                    base_model_pretrained_path=getattr(
                        self.train_cfg, "base_model_pretrained_path", None
                    ),
                    peft_pretrained_path=self.train_cfg.peft_pretrained_path,
                    merge_and_unload=True,
                )

            # quantizer in the tokenizer, not handled by this trainer
            # vq, bsq, fsq, kl
            self.use_quantizer = hasattr(self.tokenizer, "quantizer")
            self.quantizer = None
            self.log_msg(
                f"[Tokenizer]: init tokenizer {self.tokenizer.__class__.__name__}"
            )
            if self.use_quantizer:
                self.log_msg(
                    f"[Tokenizer]: has quantizer {self.tokenizer.quantizer.__class__}"
                )

        # set to eval
        if self.sep_enc_dec:
            self.tokenizer_encoder.eval()
            self.tokenizer_decoder.eval()
        else:
            self.tokenizer.eval()

        if self.use_quantizer and isinstance(self.quantizer, nn.Module):
            self.quantizer.eval()

        self.log_msg("freeze the tokenizer and quantizer (if used).")

    def prepare_ema_models(self):
        if self.no_ema:
            return

        self.ema_model = EMA(
            self.model,
            beta=self.ema_cfg.beta,
            update_every=self.ema_cfg.update_every,
        ).to(self.device)
        self.log_msg(f"create EMA model for detection")

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
                self.train_cfg.model_optim, _get_model_params
            )
        else:
            model_opt = DummyOptim([{"params": list(self.model.parameters())}])

        # schedulers
        if (
            self.accelerator.state.deepspeed_plugin is None
            or "scheduler"
            not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            model_sched = hydra.utils.instantiate(self.train_cfg.detection_sched)(
                optimizer=model_opt
            )
        else:
            model_sched = DummyScheduler(model_opt)

        # set the heavyball optimizer without torch compiling
        is_heavyball_opt = lambda opt: opt.__class__.__module__.startswith("heavyball")
        if is_heavyball_opt(model_opt):
            import heavyball.utils

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
            _get_model_dtype = lambda model: next(model.parameters()).dtype
            if self.sep_enc_dec:
                self.tokenizer_encoder.dtype = torch.float  # self.dtype
                self.tokenizer_decoder.dtype = torch.float  # self.dtype
            else:
                self.tokenizer.dtype = torch.float  # self.dtype
            self.model.dtype = torch.float

        # tokenizer
        if self.train_cfg.prepare_tokenizer_in_accelerator:
            if self.sep_enc_dec:
                # FIXME: FSDP2 missing mapping for a parameter in the optmizer
                self.tokenizer_encoder = self.accelerator.prepare(
                    self.tokenizer_encoder
                )
                self.accelerator._models.pop(-1)
                self.tokenizer_decoder = self.accelerator.prepare(
                    self.tokenizer_decoder
                )
                self.accelerator._models.pop(-1)
            else:
                self.tokenizer = self.accelerator.prepare(self.tokenizer)
                self.accelerator._models.pop(-1)

        # quantizer
        if self.quantizer is not None and self.use_quantizer:
            self.quantizer = self.accelerator.prepare(self.quantizer)
            self.accelerator._models.pop(-1)

        self.train_dataloader, self.val_dataloader = self.accelerator.prepare(
            self.train_dataloader, self.val_dataloader
        )

    def step_train_state(self, mode="train"):
        self.train_state.update(mode)

    def ema_update(self, mode="detection"):
        assert mode == "detection"
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
        assert hasattr(self, "tokenizer"), "Tokenizer not found"
        grad_ctx = torch.no_grad() if no_grad else nullcontext()

        latent_q, recon = None, None
        with grad_ctx and self.accelerator.autocast():
            if self.sep_enc_dec:
                to_enc = lambda x: self.tokenizer_encoder(x)[0]
                to_dec = lambda x: self.tokenizer_decoder(x)

                if mode == "encode":
                    latent = to_enc(x)  # x is the image
                    if self.quantizer is not None:
                        latent_q, q_loss, q_info = self.quantizer(latent)
                    else:
                        latent_q = latent
                elif mode == "decode":
                    recon = to_dec(x)  # x is the latent
            else:
                to_enc = lambda img: self.tokenizer.encode(img)
                to_dec = lambda latent, sz: self.tokenizer.decode(latent, sz)

                if mode == "encode":
                    latent = to_enc(x)
                else:
                    bands = self.get_training_sample_channels()
                    bs_chan = [x.shape[0], bands]
                    recon = to_dec(x, bs_chan)
                    if isinstance(recon, tuple):
                        recon = recon[0]  # construction is the first output

        return dict(latent=latent_q, recon=recon)

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

    def get_tokenizer_encoded(self, batch: dict) -> Tensor:
        img_latent = None
        # offline latents
        if "img_latent" in batch:
            img_latent = batch["img_latent"].to(self.device, self.dtype)

        # tokenizer online
        else:
            assert self.online_tokenize and (
                hasattr(self, "tokenizer")
                or (
                    hasattr(self, "tokenizer_encoder")
                    and hasattr(self, "tokenizer_decoder")
                )
            ), "tokenizer not found for online tokenize"
            img_latent = self.forward_tokenizer(batch["img"])["latent"]

        assert img_latent is not None, "img_latent is None"
        return img_latent

    def compute_had_loss(self, recon, gt):
        loss = self.had_loss(recon, gt)

        if isinstance(loss, torch.Tensor):
            log_losses = {"had_loss": loss.detach()}
        elif isinstance(loss, (tuple, list)):
            loss, log_losses = loss
        else:
            raise NotImplementedError(f"Unknown type {type(loss)}")

        return loss, log_losses

    def forward_detection_model(self, img, gt, img_latent):
        with self.accelerator.autocast():
            # img -> model (anomaly detection model)
            recon = self.model(img, img_latent)
            # loss
            loss, log_losses = self.compute_had_loss(recon, gt)

        return recon, loss, log_losses

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

    def train_had_step(self, img, gt, img_latent):
        recon, loss, log_losses = self.forward_detection_model(img, gt, img_latent)
        self._optimize_step(loss)
        return HADModelStepOutput(recon, loss, log_losses)

    def _get_metric_fn(self, clear=False):
        if not hasattr(self, "_had_metrics") or clear:
            self._had_metrics: HADDetectionMetrics = HADDetectionMetrics()

        # Return the HAD metrics object
        return self._had_metrics

    def update_metrics(self, anomaly_scores, gt, clear=False, only_update=True):
        had_metrics = self._get_metric_fn(clear)
        assert had_metrics is not None
        # Update HAD metrics with anomaly scores and ground truth
        had_metrics.update(anomaly_scores, gt)

    def get_metrics(self):
        had_metrics = self._get_metric_fn()
        assert had_metrics is not None
        metrics = had_metrics.compute()
        return metrics

    def train_step(self, batch: dict):
        # Standard HAD training step
        with self.accelerator.accumulate(self.model):
            img_latent = self.get_tokenizer_encoded(batch)
            # train HAD model
            train_out = self.train_had_step(batch["img"], batch["gt"], img_latent)

        # update training state
        self.step_train_state()
        # log losses
        if self.global_step % self.train_cfg.log.log_every == 0:
            _log_losses = self.format_log(train_out.log_losses)
            self.log_msg(
                f"[Train State]: lr {self.optim.param_groups[0]['lr']:1.4e} | "
                f"[Step]: {self.global_step}/{self.train_cfg.max_steps}"
            )
            self.log_msg(f"[Train HAD]: {_log_losses}")

            # tensorboard log
            self.tenb_log_any("metric", train_out.log_losses, self.global_step)

    def format_log(self, log_loss: dict, sync=False) -> str:
        if sync:
            # log_loss = metrics_sync(log_loss, output_tensor_dict=True)
            log_loss = dict_tensor_sync(log_loss)
        strings = dict_round_to_list_str(log_loss, select=list(log_loss.keys()))
        return " - ".join(strings)

    def finite_train_loader(self):
        """Finite train loader that handles complete images properly."""
        for batch in self.train_dataloader:
            yield batch

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

    def get_val_loader_iter(self):
        if self.val_cfg.max_val_iters > 0:
            # create a new iterator for the validation loader
            # state in the loader generator
            if not hasattr(self, "_val_loader_iter"):
                self._val_loader_iter = iter(self.val_dataloader)

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
            for _ in range(self.val_cfg.max_val_iters):
                try:
                    yield next(self._val_loader_iter)
                except StopIteration:
                    self._val_loader_iter = iter(self.val_dataloader)
                    yield next(self._val_loader_iter)
        else:
            self.log_msg(
                f"[Val]: start validating with the whole val set", only_rank_zero=False
            )
            for batch in self.val_dataloader:
                yield batch

    def val_step(self, batch: dict):
        img_latent = self.get_tokenizer_encoded(batch)
        # forward the HAD network
        with torch.no_grad():
            anomaly_scores, *_ = self.forward_detection_model(
                batch["img"], batch["gt"], img_latent
            )
        return anomaly_scores

    def val_loop(self):
        self.model.eval()
        loss_metrics = MeanMetric().to(device=self.device)
        val_iter = self.get_val_loader_iter()
        anomaly_scores = gt = None
        for batch in val_iter:  # type: ignore
            batch = cast(dict[str, torch.Tensor], batch)
            gt = batch.get("gt", batch.get("gt_full", None))
            assert gt is not None, "gt or gt_full not found in the val batch"
            anomaly_scores = self.val_step(batch)
            # metrics - use HAD predictions and ground truth
            self.update_metrics(anomaly_scores, gt)
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
            assert anomaly_scores is not None and gt is not None, (
                "anomaly_scores or gt is None"
            )
            self.visualize_anomaly_detection(
                anomaly_scores,  # anomaly scores
                gt,  # gt
                add_step=True,
                img_name="val/anomaly_detection",
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
            self.ema_model.ema_model, ema_path / "had_ema_model"
        )
        # train state
        _ema_path_state_train = ema_path / "had_train_state.pth"
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
        else:
            return x

    def visualize_anomaly_detection(
        self,
        anomaly_scores: torch.Tensor,
        gt_map: torch.Tensor,
        img_name: str = "val/anomaly_detection",
        add_step: bool = False,
        only_vis_n: int | None = None,
        threshold: float = 0.5,
    ):
        """Visualize anomaly detection results side by side.

        Args:
            anomaly_scores: Anomaly scores from the model.
            gt_map: Ground truth anomaly map.
            img_name: Name for the saved image. Defaults to "val/anomaly_detection".
            add_step: Whether to add step number to the image name. Defaults to False.
            only_vis_n: Number of images to visualize. Defaults to None (all).
            threshold: Threshold for binary anomaly detection. Defaults to 0.5.
        """

        _only_n = only_vis_n or 1

        # Process anomaly scores and ground truth maps
        if anomaly_scores.dim() == 4:  # (B, 1, H, W) or (B, C, H, W)
            anomaly_scores = anomaly_scores.squeeze(1)  # (B, H, W)

        if gt_map.dim() == 4:  # (B, C, H, W)
            gt_map = gt_map.squeeze(1)  # (B, H, W)

        # Only visualize the first few samples
        anomaly_scores = anomaly_scores[:_only_n]
        gt_map = gt_map[:_only_n]

        # Visualize each sample and concatenate
        vis_images = []
        for i in range(anomaly_scores.shape[0]):
            # Normalize anomaly scores to [0, 1]
            anomaly_norm = (anomaly_scores[i] - anomaly_scores[i].min()) / (
                anomaly_scores[i].max() - anomaly_scores[i].min() + 1e-8
            )

            # Create binary prediction
            binary_pred = (anomaly_norm > threshold).float()

            # Create RGB visualization
            # Anomaly scores as heatmap
            anomaly_heatmap = plt.cm.hot(anomaly_norm.cpu().numpy())[
                :, :, :3
            ]  # Remove alpha

            # Binary prediction
            binary_vis = np.zeros((*binary_pred.shape, 3))
            binary_vis[binary_pred.cpu().numpy() == 1] = [1, 0, 0]  # Red for anomalies

            # Ground truth
            gt_vis = np.zeros((*gt_map[i].shape, 3))
            gt_vis[gt_map[i].cpu().numpy() == 1] = [0, 1, 0]  # Green for ground truth

            # Concatenate the three visualizations
            combined_vis = np.concatenate([anomaly_heatmap, binary_vis, gt_vis], axis=1)
            vis_images.append(combined_vis)

        # Stack all visualizations
        if len(vis_images) > 1:
            # Create a grid of images
            rows = int(np.ceil(len(vis_images) / 2))
            cols = 2 if len(vis_images) > 1 else 1

            # Pad with empty images if necessary
            while len(vis_images) < rows * cols:
                vis_images.append(np.zeros_like(vis_images[0]))

            # Create grid
            grid_images = []
            for row in range(rows):
                row_images = vis_images[row * cols : (row + 1) * cols]
                grid_images.append(np.concatenate(row_images, axis=1))
            img = np.concatenate(grid_images, axis=0)
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
            "[Visualize]: save anomaly detection visualization at {}".format(save_path)
        )

    def run(self):
        if self.train_cfg.resume_path is not None:
            self.resume(self.train_cfg.resume_path)
        elif self.train_cfg.ema_load_path is not None:
            self.load_from_ema(self.train_cfg.ema_load_path)

        # train !
        self.train_loop()


_key = "cosmos_f8c16p4_had"  # vq, bsq, fsq, kl
_configs_dict = {
    "cosmos_f8c16p4_had": "cosmos_f8c16p4_had",
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
        print_colored_banner("HAD")
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
        config_path="configs/detections",
        config_name=chosen_cfg,
        version_base=None,
    )
    def main(cfg):
        catcher = logger.catch if PartialState().is_main_process else nullcontext

        with catcher():
            trainer = HyperHADTrainer(cfg)
            trainer.run()

    main()

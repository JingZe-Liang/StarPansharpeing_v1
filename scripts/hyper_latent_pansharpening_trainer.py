import os
import sys
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Literal

import accelerate
import colored_traceback
import hydra
import numpy as np
import peft
import PIL.Image as Image
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.state import PartialState
from accelerate.tracking import TensorBoardTracker
from accelerate.utils import DummyOptim, DummyScheduler, FullyShardedDataParallelPlugin
from ema_pytorch import EMA
from kornia.utils.image import make_grid, tensor_to_image
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    inject_adapter_in_model,
    load_peft_weights,
    set_peft_model_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._fully_shard import FSDPModule
from torch.distributed.tensor import DTensor
from torchmetrics.aggregation import MeanMetric

colored_traceback.add_hook()

sys.path.insert(0, __file__[: __file__.find("scripts")])
from src.data.hyperspectral_loader import get_hyperspectral_dataloaders
from src.stage1.cosmos.cosmos_tokenizer import (
    ContinuousImageTokenizer as CosmosTokenizer,
)

# from src.stage1.LeanVAE.LeanVAE.models.autoencoder import LeanVAE2D
from src.stage1.cosmos.inference.utils import load_jit_model_shape_matched
from src.stage1.sana_dcae.models.efficientvit.dc_ae import DCAE
from src.stage1.two_d_vit_tokenizer.tokenizer import VITBSQModel, VITVQModel
from src.stage1.utilities.losses.gan_loss import VQLPIPSWithDiscriminator
from src.stage2.pansharpening.metrics import AnalysisPanAcc
from src.utilities.network_utils import load_fsdp_model, remap_peft_model_state_dict
from src.utilities.train_utils.state import StepsCounter

to_cont = partial(OmegaConf.to_container, resolve=True)
# omegaconf resolver
OmegaConf.register_new_resolver("eval", lambda x: eval(x))
OmegaConf.register_new_resolver("function", lambda x: hydra.utils.get_method(x))


## TODO:
# 1. add diffusion, rectify flow, or inductive moumentumn matching to the decoder
# 2. add BSQ or LFQ to discretize
# 3. add rectify flow for continuous latents; maskbit or maskgit for discrete latents


class PansharpeningTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.tokenizer_cfg = cfg.tokenizer
        self.pansp_cfg = cfg.pansharpening
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
        _fsdp_plugin: accelerate.utils.FullyShardedDataParallelPlugin = getattr(
            self.accelerator.state, "fsdp_plugin", None
        )
        self.no_ema = False
        self._is_ds = _dpsp_plugin is not None
        if self._is_ds:
            self.log_msg("[Deepspeed]: using deepspeed plugin")
            self.no_ema = _dpsp_plugin.deepspeed_config["zero_optimization"][
                "stage"
            ] in [2, 3]

        self._is_fsdp = _fsdp_plugin is not None
        if self._is_fsdp:
            self.log_msg("[FSDP]: using Fully Sharded Data Parallel plugin")
            self.no_ema = True

        # dataloader
        used_dataset = self.dataset_cfg.used
        self.log_msg(f"[Data]: using dataset {used_dataset}")
        self.train_dataset, self.train_dataloader = get_hyperspectral_dataloaders(
            wds_paths=self.dataset_cfg.wds_path_train,
            batch_size=self.dataset_cfg.batch_size_train,
            num_workers=self.dataset_cfg.num_workers,
            shuffle_size=self.dataset_cfg.shuffle_size,
            hyper_transforms_lst=self.dataset_cfg.hyper_transforms_lst,
            transform_prob=self.dataset_cfg.transform_prob,
            random_apply=to_cont(self.dataset_cfg.random_apply),
            to_neg_1_1=True,
        )
        self.val_dataset, self.val_dataloader = get_hyperspectral_dataloaders(
            wds_paths=self.dataset_cfg.wds_path_val,
            batch_size=self.dataset_cfg.batch_size_val,
            num_workers=self.dataset_cfg.num_workers,
            shuffle_size=self.dataset_cfg.shuffle_size,
            hyper_transforms_lst=None,
            transform_prob=0.0,
            to_neg_1_1=True,
        )

        if _dpsp_plugin is not None:
            self.accelerator.deepspeed_plugin.deepspeed_config[
                "train_micro_batch_size_per_gpu"
            ] = self.dataset_cfg.batch_size_train

        # setup the tokenizer
        self.online_tokenize = self.train_cfg.online_tokenize
        if self.train_cfg.online_tokenize:
            self.setup_tokenizer()

        # optimizers and lr schedulers
        self.pansp_optim, self.pansp_sched = self.get_optimizer_lr_scheduler()
        # EMA models and accelerator prepare
        self.prepare_for_training()
        self.prepare_ema_models()

        # loss
        if hasattr(self.train_cfg, "pansharpeing_loss"):
            self.pansp_loss = hydra.utils.instantiate(self.train_cfg.pansharpeing_loss)
        else:
            self.pansp_loss = nn.L1Loss()
        self.log_msg(f"use pansharpening loss: {self.pansp_loss.__class__.__name__}")

        # traing state counter
        self.train_state = StepsCounter(["train"])

        # clear GPU memory
        torch.cuda.empty_cache()

    def setup_pansharpening_model(self):
        self.pansp_model = hydra.utils.instantiate(self.train_cfg.pansharpeing_model)
        pansp_name = (
            getattr(self.train_cfg, "pansharpeing_name", None)
            or self.pansp_model.__class__.__name__
        )
        self.log_msg(f"use pansharpening model: {pansp_name}")

    def setup_tokenizer(self):
        tokenizer_name = self.train_cfg.tokenizer_name
        self.sep_enc_dec = self.train_cfg.seperate_enc_dec
        self.quantizer_type: str | None = self.cfg.vq_loss.quantizer_type
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
            # self.tokenizer: (
            #     VITVQModel | VITBSQModel | CosmosTokenizer | DCAE | LeanVAE2D
            # )
            # quantizer in the tokenizer, not handled by this trainer
            self.use_quantizer = hasattr(
                self.tokenizer, "quantizer"
            )  # vq, bsq, fsq, kl
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

        self.ema_pansp_model = EMA(
            self.pansp_model,
            beta=self.ema_cfg.beta,
            update_every=self.ema_cfg.update_every,
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
        log_format_in_cmd = "<level>[{level}]</level> - <level>{message}</level>"
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
        self.accelerator.project_configuration.project_dir = self.proj_dir

        # tensorboard logger
        if not self.train_cfg.debug:
            tenb_dir = log_dir / "tensorboard"
            self.accelerator.project_configuration.logging_dir = tenb_dir
            if self.accelerator.is_main_process:
                self.logger.info(f"[Tensorboard]: tensorboard saved to {tenb_dir}")
                self.accelerator.init_trackers("train")
                self.tb_logger: TensorBoardTracker = self.accelerator.get_tracker(
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
                        _p_grad = _grad.full_tensor()  # across all ranks
                    _grad_norm = (_p_grad.data**2).sum() ** 0.5
                    if log_type == "grad_norm_per_param":
                        norms[f"{model_cls_n}/{n}"] = _grad_norm
                    else:
                        norms[f"{model_cls_n}_grad_norm"] += _grad_norm
                        _n_params_sumed += 1
            # log
            if log_type == "grad_norm_sum":
                norms[f"{model_cls_n}_grad_norm"] /= _n_params_sumed
            if hasattr(self, "tb_logger"):
                self.tb_logger.log(
                    norms,
                    step=step,
                )
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

    def get_optimizer_lr_scheduler(self):
        # optimizers
        if (
            self.accelerator.state.deepspeed_plugin is None
            or "optimizer"
            not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            pansp_opt = hydra.utils.instantiate(self.train_cfg.tokenizer_optimizer)(
                self._get_tokenizer_params()
            )
        else:
            pansp_opt = DummyOptim([{"params": list(self.pansp_model.parameters())}])

        # schedulers
        if (
            self.accelerator.state.deepspeed_plugin is None
            or "scheduler"
            not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            pansp_sched = hydra.utils.instantiate(self.train_cfg.pansp_scheduler)(
                optimizer=pansp_opt
            )
        else:
            tokenizer_sched = DummyScheduler(pansp_opt)

        # set the heavyball optimizer without torch compiling
        is_heavyball_opt = lambda opt: opt.__class__.__module__.startswith("heavyball")
        if is_heavyball_opt(pansp_opt):
            import heavyball

            heavyball.utils.compile_mode = None

            self.log_msg(
                f"use heavyball optimizer, it will compile the optimizer, "
                "for efficience testing the scripts, disable the compilation."
            )

        return pansp_opt, pansp_sched

    def set_fsdp_cpu_local_tensor_to_each_rank(self, model: nn.Module | FSDPModule):
        if not self._is_fsdp:
            return model

        self.log_msg(
            "FSDP module seems do not move the original parameter (_local_tensor) on the"
            "correct rank, we need to manully move them on cuda while using `to_local` or `redistributed` methods",
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
            self.pansp_model.dtype = torch.float

        # tokenizer
        if self.train_cfg.prepare_tokenizer_in_accelerator:
            if self.sep_enc_dec:
                # FIXME: FSDP2 missing mapping for a parameter in the optmizer
                self.tokenizer_encoder, self.tokenizer_optim = self.accelerator.prepare(
                    self.tokenizer_encoder, self.tokenizer_optim
                )
                self.accelerator._models.pop(-1)
                self.accelerator._optimizers.pop(-1)
                self.tokenizer_decoder, self.tokenizer_optim = self.accelerator.prepare(
                    self.tokenizer_decoder, self.tokenizer_optim
                )
                self.accelerator._models.pop(-1)
                self.accelerator._optimizers.pop(-1)
            else:
                self.tokenizer, self.tokenizer_optim = self.accelerator.prepare(
                    self.tokenizer, self.tokenizer_optim
                )
                self.accelerator._models.pop(-1)
                self.accelerator._optimizers.pop(-1)

        # quantizer
        if self.quantizer is not None and self.use_quantizer:
            self.quantizer = self.accelerator.prepare(self.quantizer)
            self.accelerator._models.pop(-1)

        self.train_dataloader, self.val_dataloader = self.accelerator.prepare(
            self.train_dataloader, self.val_dataloader
        )

        self.tokenizer_sched, self.disc_sched = self.accelerator.prepare(
            self.tokenizer_sched, self.disc_sched
        )

    def step_train_state(self):
        self.train_state.update("train")

    def ema_update(self, mode="pansharpening"):
        assert mode == "pansharpening"
        if self.no_ema:
            # not support ema when is deepspeed zero2 or zero3
            return

        self.ema_pansp_model.update()

    @torch.no_grad()
    def forward_tokenizer(self, x: torch.Tensor, mode: str = "encode"):
        if not self.online_tokenize:
            return None

        latent_q, recon = None
        with self.accelerator.autocast():
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
                to_dec = lambda latent: self.tokenizer.decode(latent)

                if mode == "encode":
                    latent = to_enc(x)
                    if self.use_quantizer:
                        recon, q_loss, q_info = self.tokenizer(x)
                elif mode == "decode":
                    recon = to_dec(x)
                    if isinstance(recon, tuple):
                        recon = recon[0]

        return dict(latent=latent_q, recon=recon)

    def get_global_step(self, mode="train"):
        # TODO: add val state
        assert mode in ("train",), "Only train mode is supported for now."

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

    def train_pansp_step(
        self,
        lrms_tok_out: dict | None = None,
        pan_tok_out: dict | None = None,
        sr_toke_out: dict | None = None,
        lrms_latent: torch.Tensor | None = None,
        pan_latent: torch.Tensor | None = None,
        sr_latent: torch.Tensor | None = None,
    ):
        if lrms_tok_out is not None and pan_tok_out is not None:
            lrms_latent = lrms_tok_out["latent"].detach()
            pan_latent = pan_tok_out["latent"].detach()
            sr_latent = sr_toke_out["latent"].detach()

        with self.accelerator.autocast():
            pred_latent = self.pansp_model(lrms_latent, pan_latent)
            sr_loss, sr_log_losses = self.pansp_loss(pred_latent, sr_latent)

        if self.accelerator.sync_gradients:
            # backward
            self.pansp_optim.zero_grad()
            self.accelerator.backward(sr_loss)
            self.gradient_check(self.pansp_model)
            self.pansp_optim.step()

            # ema update
            self.ema_update(mode="pansharpening")

        return dict(
            pred_latent=pred_latent, sr_loss=sr_loss, sr_log_losses=sr_log_losses
        )

    def train_step(
        self,
        batch: dict,
    ):
        lrms_latent = pan_latent = hrms_latent = None
        if "lrms_latent" in batch:
            lrms_latent = batch["lrms_latent"].to(self.device, self.dtype)
            pan_latent = batch["pan_latent"].to(self.device, self.dtype)
            hrms_latent = batch["hrms_latent"].to(self.device, self.dtype)
        # tokenizer online
        else:
            assert self.online_tokenize and (
                hasattr(self, "tokenizer")
                or (
                    hasattr(self, "tokenizer_encoder")
                    and hasattr(self, "tokenizer_decoder")
                )
            ), "tokenizer not found for online tokenize"
            lrms_tok_out = self.forward_tokenizer(batch["lms"])
            pan_tok_out = self.forward_tokenizer(batch["pan"])
            sr_tok_out = self.forward_tokenizer(batch["gt"])

        quality_track_n = self.train_cfg.track_metrics_duration
        quality_track_after = self.train_cfg.track_metrics_after
        if quality_track_n >= 0:
            self.pan_acc_reduced = AnalysisPanAcc(ratio=4, ref=True, ergas_ratio=4)
            self.pan_acc_reduced_latent = AnalysisPanAcc(
                ratio=4, ref=True, ergas_ratio=4
            )

            def check_quality(check_fn, pred_sr, lms=None, pan=None, gt=None):
                pred_sr = self.to_rgb(pred_sr)
                if lms is not None:
                    lms = self.to_rgb(lms)
                if pan is not None:
                    pan = self.to_rgb(pan)
                if gt is not None:
                    gt = self.to_rgb(gt)

                check_fn(pred_sr, gt)

        with self.accelerator.accumulate(self.pansp_model):
            # train pansharpening model
            train_out = self.train_pansp_step(
                lrms_tok_out=lrms_tok_out,
                pan_tok_out=pan_tok_out,
                sr_toke_out=sr_tok_out,
                lrms_latent=lrms_latent,
                pan_latent=pan_latent,
                sr_latent=hrms_latent,
            )
            pred_img = self.forward_tokenizer(train_out["pred_latent"], mode="decode")[
                "recon"
            ]

            # track reconstruction quality
            if (
                quality_track_n >= 0
                and self.global_step % quality_track_n != 0
                and self.global_step >= quality_track_after
            ):
                check_quality(self.pan_acc_reduced, pred_sr=pred_img, gt_sr=batch["gt"])
                check_quality(
                    self.pan_acc_reduced_latent,
                    pred_sr=pred_img,
                    gt_sr=self.forward_tokenizer(sr_tok_out, mode="decode"),
                )

        self.step_train_state()

        # log losses
        if self.global_step % self.train_cfg.log.log_every == 0:
            _log_tok_losses = self.format_log(log_token_loss=train_out["sr_log_losses"])

            self.log_msg(
                f"[Train State]: lr {self.tokenizer_optim.param_groups[0]['lr']:.4e} | "
                f"[Step]: {self.global_step}/{self.train_cfg.max_steps}"
            )
            self.log_msg(f"[Train Tok]: {_log_tok_losses}")

            # tensorboard log
            self.tenb_log_any("metric", train_out["sr_log_losses"], self.global_step)

        if (
            quality_track_n >= 0
            and self.global_step % quality_track_n == 0
            and self.global_step >= quality_track_after
        ):
            self.log_msg(f"[Real GT Metrics]: {self.pan_acc_reduced.print_str()}")
            self.log_msg(f"[Latent Metrics]: {self.pan_acc_reduced_latent.print_str()}")

    def format_log(self, log_sr_loss: dict) -> str:
        def dict_round_to_list_str(
            d: dict, n_round: int = 3, select: list[str] | None = None
        ):
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

        n_round = 4
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

    def finite_val_loader(self):
        if self.val_dataloader is None:
            raise ValueError("No validation dataloader found")

        for batch in self.val_dataloader:
            yield batch

    def val_step(self, batch: dict):
        if not self.online_tokenize:
            lrms_tok_out = self.forward_tokenizer(batch["lms"])["latent"]
            pan_tok_out = self.forward_tokenizer(batch["pan"])["latent"]
            sr_tok_out = self.forward_tokenizer(batch["gt"])["latent"]
        else:
            lrms_latent = batch["lrms_latent"].to(self.device, self.dtype)
            pan_latent = batch["pan_latent"].to(self.device, self.dtype)
            hrms_latent = batch["hrms_latent"].to(self.device, self.dtype)

        # forward the fusion network
        pred_latent = self.pansp_model(lrms_latent, pan_latent)
        pred_img = self.forward_tokenizer(pred_latent, mode="decode")

        return {"pred_img": pred_img, "pred_latent": pred_latent}

    def val_loop(self):
        self.pansp_model.eval()

        if not hasattr(self, "_val_loader_iter"):
            # state in the loader generator
            self._val_loader_iter = iter(self.finite_val_loader())

        # track psnr and ssim
        pan_acc_fn = AnalysisPanAcc(ratio=4, ref=True)
        loss_metrics = MeanMetric().to(device=self.device)

        for val_iter in range(self.val_cfg.max_val_iters):
            batch = next(self._val_loader_iter)
            gt = batch["gt"].to(self.device)
            val_out = self.val_step(batch)

            pred_img_rgb = self.to_rgb(val_out["pred_img"])
            batch_img_rgb = self.to_rgb(gt)

            # l1 loss
            loss = nn.functional.l1_loss(pred_img_rgb, gt.to(pred_img_rgb))
            loss_metrics.update(loss)

            # metrics
            pan_acc_fn(batch_img_rgb, pred_img_rgb)

        pan_acc = pan_acc_fn.acc_ave
        loss_val = loss_metrics.compute()

        # gather
        if self.accelerator.use_distributed:
            pan_accs = self.accelerator.gather_for_metrics(pan_acc)
            pan_acc_mean = {}
            for k in pan_accs[0].keys():
                for i in range(len(pan_accs)):
                    pan_acc_mean[k] = pan_acc_mean.get(k, 0) + pan_accs[i][k]
                pan_acc_mean[k] /= len(pan_accs)

        if self.accelerator.is_main_process:
            _metric_str = ""
            for k, v in pan_acc_mean.items():
                _metric_str += f"{k}: {v:.4f} - "
            self.log_msg(f"[Val]: {_metric_str} loss: {loss_val:.4f}")
            self.tenb_log_any(
                "metric",
                pan_acc_mean,
                step=self.global_step,
            )

            # visualize the last val batch
            # self.visualize_reconstruction(
            #     batch["img"], recon, add_step=True, img_name="val_sampled/sampled"
            # )

    def save_state(self):
        self.accelerator.save_state()
        self.log_msg("[State]: save states")

        if self._is_peft_tuning and self.train_cfg.save_peft_ckpts:
            # save peft model
            assert self.tokenizer_peft_wrapped is not None, "peft model not wrapped"
            if self._is_fsdp:
                accelerate.utils.save_fsdp_model(
                    output_dir=self.proj_dir / "peft_ckpt",
                    fsdp_plugin=self.accelerator.state.fsdp_plugin,
                    accelerator=self.accelerator,
                    model=self.tokenizer_peft_wrapped,
                    model_index=0,
                    adapter_only=True,  # for possible loading, we save the whole model
                )
            else:
                # is ddp
                self.tokenizer_peft_wrapped.save_pretrained(
                    self.proj_dir / "peft_ckpt",
                    is_main_process=self.accelerator.is_main_process,
                )

            self.log_msg(f"[State]: save peft (only lora layers) model")

    def save_ema(self):
        if self.no_ema:
            self.log_msg(f"use deepspeed or FSDP, do have EMA model to save")
            return

        ema_path = self.proj_dir / "ema"
        if self.accelerator.is_main_process:
            ema_path.parent.mkdir(parents=True, exist_ok=True)

        if self.sep_enc_dec:
            self.accelerator.save_model(
                self.ema_encoder.ema_model,
                ema_path / "encoder",
            )
            self.accelerator.save_model(
                self.ema_decoder.ema_model,
                ema_path / "decoder",
            )
        else:
            self.accelerator.save_model(
                self.ema_tokenizer.ema_model,
                ema_path / "tokenizer",
            )

        self.accelerator.save_model(
            self.ema_vq_disc.ema_model,
            ema_path / "discriminator",
        )

        # train state
        _ema_path_state_train = ema_path / "train_state.pth"
        _ema_path_state_train.parent.mkdir(parents=True, exist_ok=True)
        accelerate.utils.save(self.train_state.state_dict(), _ema_path_state_train)

        if (
            self.use_quantizer
            and self.quantizer is not None
            and isinstance(self.quantizer, nn.Module)
        ):
            self.accelerator.save_model(
                self.quantizer,
                ema_path / "quantizer",
            )

        self.log_msg(f"[Ckpt]: save ema at {ema_path}")

    def load_from_ema_or_lora(self, ema_path: str, strict: bool = True):
        ##! FIXME: if is loaded after FSDP2 shard, it won't work

        ema_path = Path(ema_path)

        if self.sep_enc_dec:
            if self._is_fsdp:
                raise NotImplementedError(
                    "FSDP2 loading for seperated encoder and decoder is not implemented yet"
                )

            # Load encoder to online model
            accelerate.load_checkpoint_in_model(
                self.tokenizer_encoder, ema_path / "encoder"
            )
            # Load decoder to online model
            accelerate.utils.load_checkpoint_in_model(
                self.tokenizer_decoder, ema_path / "decoder"
            )
        else:
            assert (
                self.accelerator.distributed_type
                != accelerate.utils.DistributedType.DEEPSPEED
            ), "Deepspeed does not support PEFT yet."

            _assume_path = ema_path / "tokenizer"
            if not _assume_path.exists():
                if self._is_peft_tuning:
                    _assume_path = ema_path / "model"
                    # we are working on loading the peft model (only lora layers)
                    if self._is_fsdp:
                        if not getattr(self.train_cfg, "load_weight_shard", True):
                            _not_shard_weights = (
                                "model" in _assume_path.as_posix()
                            )  # depend on if there is a full state dict
                        else:
                            _not_shard_weights = False  # shard loaded
                    else:
                        _not_shard_weights = True

                    self.log_msg(
                        "loading peft checkpoint into model", only_rank_zero=False
                    )
                    if not _not_shard_weights:  # is shard weights
                        if self._is_fsdp:
                            loaded_res = load_fsdp_model(
                                self.accelerator.state.fsdp_plugin,
                                self.accelerator,
                                self.tokenizer_peft_wrapped,
                                ema_path.as_posix(),
                                model_index=0,
                                adapter_only=True,
                            )
                        else:  # is ddp or one gpu
                            load_res = set_peft_model_state_dict(
                                self.tokenizer_peft_wrapped,
                                accelerate.utils.load_state_dict(ema_path.as_posix()),
                                adapter_name="default",
                                ignore_mismatched_sizes=strict,
                                low_cpu_mem_usage=False,
                            )

                    else:  # not shard weights
                        peft_path = _assume_path / "pytorch_model_fsdp.bin"
                        assert peft_path.exists(), "peft checkpoint dir not found, use accelerate-merge-weights CLI first"

                        # ensure there is model/pytorch_model_fsdp.bin file
                        from copy import deepcopy

                        from torch.distributed.fsdp import StateDictType

                        _fsdp_plugin_cp = deepcopy(self.accelerator.state.fsdp_plugin)
                        _fsdp_plugin_cp.state_dict_type = StateDictType.FULL_STATE_DICT
                        loaded_res = load_fsdp_model(
                            _fsdp_plugin_cp,
                            self.accelerator,
                            self.tokenizer_peft_wrapped,
                            ema_path.as_posix(),
                            model_index=0,
                            adapter_only=True,  # Warn: loading from shard only-lora-layer weights does not work.
                        )

                    self.log_msg(
                        f"[Warning]: {loaded_res} keys are incompatible with the model",
                        level="warning",
                    )

                    # is refer
                    # NOTE: forcing to get the base model, not sure if this is referencing the same model
                    self.tokenizer = self.tokenizer_peft_wrapped.get_base_model()

                # TODO: test it, this will not work
                elif self._is_fsdp:
                    _assume_path = ema_path / "pytorch_model_fsdp_0"  # model_idx=0
                    if _assume_path.exists():
                        assert _assume_path.exists(), "FSDP checkpoint dir not found"
                        self.log_msg(
                            "loading FSDP checkpoint into model", only_rank_zero=False
                        )
                        incomp_keys = accelerate.utils.load_fsdp_model(
                            self.accelerator.state.fsdp_plugin,
                            self.accelerator,
                            self.tokenizer,
                            ema_path.as_posix(),
                            model_index=0,
                        )
                        if incomp_keys:
                            self.log_msg(
                                f"[Warning]: {incomp_keys} keys are incompatible with the model",
                                level="warning",
                            )
                else:
                    raise ValueError("load FSDP or LoRA weights failed")
            else:
                # Load combined model
                self.log_msg(f"loading bin or safetensors checkpoint into model ...")
                accelerate.utils.load_checkpoint_in_model(
                    self.accelerator.unwrap_model(self.tokenizer),
                    _assume_path,
                    strict=strict,
                )

        # Load discriminator to online model
        if (
            self.train_cfg.finetune_strategy
            not in [
                "dcae_refine_decoder_head",
                "dcae_adapt_latent",
                "peft",
            ]
            and not self.train_cfg.only_load_tokenizer
        ):
            accelerate.utils.load_checkpoint_in_model(
                self.accelerator.unwrap_model(self.vq_loss_fn.discriminator),
                ema_path / "discriminator",
                strict=strict,
            )
            # self.train_state.load_state_dict(torch.load(ema_path / "train_state.pth"))

        # Load quantizer if exists
        if (
            self.use_quantizer
            and self.quantizer is not None
            and isinstance(self.quantizer, nn.Module)
        ):
            if (ema_path / "quantizer").exists():
                accelerate.utils.load_checkpoint_in_model(
                    self.accelerator.unwrap_model(self.quantizer),
                    ema_path / "quantizer",
                    strict=strict,
                )
            else:
                raise RuntimeError(
                    "Quantizer not found in the checkpoint, please check your checkpoint."
                )

        # Prepare models
        self.prepare_ema_models()  # This will update EMA models with online models' weights

        # clear the accelerator model registration
        self.log_msg(
            f"[Load EMA]: clear the accelerator registrations and re-prepare training"
        )

        # Remove this

        # self.accelerator._models = []
        # self.accelerator._optimizers = []
        # self.accelerator._schedulers = []
        # self.prepare_for_training()

        # self.log_msg(
        #     f"[EMA]: load ema weights to online models from {ema_path}, {strict=}"
        # )

    def resume(self, path: str):
        self.log_msg("[Resume]: resume training")
        self.accelerator.load_state(path)
        self.accelerator.wait_for_everyone()

    def to_rgb(self, x):
        return ((x + 1) / 2).clamp(0, 1).float()

    def visualize_reconstruction(
        self,
        x: torch.Tensor,
        recon: torch.Tensor | None = None,
        img_name: str = "train_original_recon",
        add_step: bool = False,
        only_vis_n: int | None = None,
    ):
        x = self.to_rgb(x)
        if recon is not None:
            recon = self.to_rgb(recon)

        _only_n = only_vis_n or 16
        to_img = lambda x: tensor_to_image(make_grid(x[:_only_n], n_row=4, padding=2))
        c = x.shape[1]

        def hyperspectral_to_rgb(x):
            # is rgb or gray images
            if c in (1, 3):
                x_np = to_img(x)
            else:
                rgb_channels = to_cont(
                    self.dataset_cfg.rgb_channels
                )  # _prefixed_rgb_channels[c]
                x_np = to_img(x[:, rgb_channels])

            return x_np

        x_np = hyperspectral_to_rgb(x)

        # cat original and reconstructed images
        if recon is not None:
            recon_np = hyperspectral_to_rgb(recon)
            img = np.concatenate([x_np, recon_np], axis=1)
        else:
            img = x_np

        # save
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

        self.log_msg("[Visualize]: save visualization at {}".format(save_path))

    def run(self):
        if self.train_cfg.finetune_strategy in [
            "dcae_refine_decoder_head",
            "dcae_adapt_latent",
        ]:
            assert (
                self.train_cfg.ema_load_path is not None
            ), f"ema_load_path must be specified when finetune_strategy is {self.train_cfg.finetune_strategy}"

        # test

        # save_path = self.accelerator.save_state()
        # self.log_msg(f"save state into {save_path}", only_rank_zero=False)
        # self.accelerator.wait_for_everyone()

        # # try to load
        # self.accelerator.load_state(input_dir=save_path)
        # self.log_msg(f"loaded state from {save_path}", only_rank_zero=False)

        # exit(0)

        if self.train_cfg.resume_path is not None:
            self.resume(self.train_cfg.resume_path)
        elif self.train_cfg.ema_load_path is not None:
            self.load_from_ema_or_lora(self.train_cfg.ema_load_path)

        # train !
        self.train_loop()


_key = "unicosmos_f8c16p4_repa_kl"
_configs = {
    # use pretrained cosmos world tokenizer (continous image configuration)
    "cosmos_sep_f8c16p4": "cosmos_post_train_f8c16p4",
    "cosmos_sep_f8c32p1": "cosmos_post_train_f8c32p1",
    "cosmos_sep_f16c16p4": "cosmos_post_train_f16c16p4",
    # unify encoder and decoder int one model
    "unicosmos_f8c16p4": "unicosmos_tokenizer_f8c16p4",
    "unicosmos_f16c16p1": "unicosmos_tokenizer_f16c16p1",
    "unicosmos_f16c16p2": "unicosmos_tokenizer_f16c16p2",
    "unicosmos_f8c16p4_repa_kl": "unicosmos_tokenizer_kl_repa_f8c16p4",
    # sana CDAE
    "sana_f8c16p1": "cdae_f8c16p1",
    "sana_f32c32p1_pretrained": "cdae_f32c32p1_pretrained",
    # leanvae
    "lean_vae_f8c16p4": "lean_vae_f8c16p4",
    # lora finetuning
    "unicosmos_lora_f8c16p4": "unicosmos_lora_finetune_f8c16p4",
}[_key]


@hydra.main(
    config_path="configs/tokenizer_gan",
    config_name=_configs,
    version_base=None,
)
def main(cfg):
    catcher = logger.catch if PartialState().is_main_process else nullcontext

    with catcher():
        trainer = PansharpeningTrainer(cfg)
        trainer.run()


if __name__ == "__main__":
    main()

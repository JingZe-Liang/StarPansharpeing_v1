import dataclasses
import os
import sys
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Callable, Iterable, Literal, TypedDict, cast

import accelerate
import accelerate.utils
import hydra
import numpy as np
import PIL.Image as Image
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.state import PartialState
from accelerate.tracking import TensorBoardTracker
from accelerate.utils import DummyOptim, DummyScheduler  # ty: ignore
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
from src.stage2.pansharpening.loss import AmotizedPixelLoss
from src.stage2.pansharpening.metrics import AnalysisPanAcc, PansharpeningMetrics
from src.utilities.config_utils import (
    to_object as to_cont,  # register new resolvers at the same time
)
from src.utilities.func.dict_utils import keys_in_dict
from src.utilities.logging import log_print
from src.utilities.network_utils import load_peft_model_checkpoint
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


@dataclasses.dataclass
class PansharpeningOutput:
    pred_latent: Tensor
    pred_sr: Tensor
    pred_sr_from_latent: Tensor | None
    sr_loss: Tensor
    sr_log_losses: dict[str, Tensor]


class PansharpeningTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.tokenizer_cfg = cfg.tokenizer
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

        # setup the tokenizer
        self.online_tokenize = self.train_cfg.online_tokenize
        self.setup_tokenizer()  # must setup the tokenizer to decode the image
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
            assert not self.pansp_amotizing_pixels, (
                "pansharpening loss must be set in the config"
            )
        self.log_msg(f"use pansharpening loss: {self.pansp_loss.__class__.__name__}")

        # training state counter
        self.train_state = StepsCounter(["train"])

        # clear GPU memory
        torch.cuda.empty_cache()

    def setup_pansharpening_model(self):
        self.pansp_model = hydra.utils.instantiate(self.cfg.pansharp_model)

        # is partial
        if isinstance(self.pansp_model, partial):
            # add the decode fn
            decoder_fn = self.tokenizer.decode
            orignal_func = self.pansp_model.func
            known_args, known_kwargs = self.pansp_model.args, self.pansp_model.keywords
            # assert class must be init
            self.pansp_model = orignal_func(
                *known_args, decoder_fn=decoder_fn, **known_kwargs
            )
            self.log_msg(
                f"[Pansharpening Model]: use partial model {orignal_func.__name__} with add 'decode_fn'={decoder_fn.__name__}"
            )
        self.pansp_amotizing_pixels = self.accelerator.unwrap_model(
            self.pansp_model
        ).amotizing_pixels

        # checkpoint
        if (
            hasattr(
                self.accelerator.unwrap_model(self.pansp_model), "set_checkpoint_mode"
            )
            and self.train_cfg.model_act_checkpoint
        ):
            self.accelerator.unwrap_model(self.pansp_model).set_checkpoint_mode(True)

        if self.pansp_amotizing_pixels:
            assert hasattr(self, "tokenizer"), (
                "pansharpening model must be used with tokenizer for amotizing pixels"
            )
            self.backward_detokenizer = True
        else:
            self.backward_detokenizer = self.train_cfg.backward_detokenizer

        pansp_name = (
            getattr(self.train_cfg, "pansharpening_name", None)
            or self.pansp_model.__class__.__name__
        )
        self.log_msg(
            f"use pansharpening model: {pansp_name}, amotizing pixels: {self.pansp_amotizing_pixels}"
        )

    def setup_tokenizer(self):
        tokenizer_name = self.train_cfg.tokenizer_name
        self.sep_enc_dec = self.train_cfg.seperate_enc_dec
        self.quantizer_type: str | None = self.train_cfg.quantizer_type
        self.tokenizer_not_req_grad = getattr(
            self.train_cfg, "tokenizer_not_req_grad", True
        )
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
            if self.tokenizer_not_req_grad:
                self.tokenizer_encoder.requires_grad_(False)
                self.tokenizer_decoder.requires_grad_(False)

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
            self.tokenizer: nn.Module = hydra.utils.instantiate(self.cfg.tokenizer)

            # not lora mixin, handled by this trainer
            if not self.train_cfg.is_lora_mixin:
                if self.tokenizer_not_req_grad:
                    self.tokenizer.requires_grad_(False)

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
            self.quantizer = getattr(self.tokenizer, "quantizer", None)
            self.use_quantizer = self.quantizer is not None
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
            self.tokenizer.to(self.device, self.dtype)
            self.tokenizer.to(self.device, self.dtype)
        else:
            self.tokenizer.eval()
            self.tokenizer.to(self.device, self.dtype)

        if self.use_quantizer and isinstance(self.quantizer, nn.Module):
            self.quantizer.eval()

        self.log_msg("freeze the tokenizer and quantizer (if used).")

    def prepare_ema_models(self):
        if self.no_ema:
            return

        self.ema_pansp_model = hydra.utils.instantiate(self.cfg.ema)(
            self.pansp_model
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

    def get_optimizer_lr_scheduler(
        self,
    ):
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

            _get_panshap_model_params = (
                lambda with_name: self.pansp_model.named_parameters()
                if with_name
                else self.pansp_model.parameters()
            )
            pansp_opt = _optimizer_creater(
                self.train_cfg.pansharp_optim, _get_panshap_model_params
            )
        else:
            pansp_opt = DummyOptim([{"params": list(self.pansp_model.parameters())}])

        # schedulers
        if (
            self.accelerator.state.deepspeed_plugin is None
            or "scheduler"
            not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            pansp_sched = hydra.utils.instantiate(self.train_cfg.pansharp_sched)(
                optimizer=pansp_opt
            )
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
                self.tokenizer_encoder = self.accelerator.prepare(
                    self.tokenizer_encoder
                )
                self.accelerator._models.pop(-1)
                self.tokenizer_decoder = self.accelerator.prepare(
                    self.tokenizer_decoder
                )
                self.accelerator._models.pop(-1)
            else:
                # FIXME: may return the base model. Why?
                self.tokenizer = self.accelerator.prepare(self.tokenizer)
                self.accelerator._models.pop(-1)

        # quantizer
        if self.quantizer is not None and self.use_quantizer:
            self.quantizer = self.accelerator.prepare(self.quantizer)
            self.accelerator._models.pop(-1)

        self.train_dataloader, self.val_dataloader = self.accelerator.prepare(
            self.train_dataloader, self.val_dataloader
        )

    def step_train_state(self):
        self.train_state.update("train")

    def ema_update(self, mode="pansharpening"):
        assert mode == "pansharpening"
        if self.no_ema:
            # not support ema when is deepspeed zero2 or zero3
            return
        self.ema_pansp_model.update()

    def forward_tokenizer(
        self, x: torch.Tensor, mode: str = "encode", no_grad=True
    ) -> dict:
        assert hasattr(self, "tokenizer"), "Tokenizer not found"
        grad_ctx = torch.no_grad() if no_grad else nullcontext()
        x = self.to_tokenizer_range(x)

        latent_q, recon = None, None
        with self.accelerator.autocast():
            if self.sep_enc_dec:
                to_enc = lambda x: self.tokenizer_encoder(x)[0]
                to_dec = lambda x: self.tokenizer_decoder(x)

                with grad_ctx:
                    if mode == "encode":
                        latent = to_enc(x)  # x is the image
                        if self.quantizer is not None:
                            latent_q, q_loss, q_info = self.quantizer(latent)
                        else:
                            latent_q = latent
                    elif mode == "decode":
                        recon = to_dec(x)  # x is the latent
            else:
                with grad_ctx:
                    if mode == "encode":
                        latent_q = self.tokenizer.encode(x)
                        if isinstance(latent_q, tuple):
                            latent_q = latent_q[0]
                    elif mode == "decode":
                        _bc = [x.shape[0], self.dataset_cfg.consts.bands]
                        recon = self.tokenizer.decode(x, _bc)
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

    def forward_pansp_model(self, lrms, pan, sr, lrms_latent, pan_latent, sr_latent):
        with self.accelerator.autocast():
            if self.pansp_amotizing_pixels:
                out = self.pansp_model((lrms, pan), (lrms_latent, pan_latent))
                pred_latent = out["latent_out"]
                pred_sr = out["pixel_out"]
                pred_sr_from_latent = out.get("pixel_from_latent", None)
            else:
                pred_latent = self.pansp_model(lrms_latent, pan_latent)
                pred_sr = self.forward_tokenizer(
                    pred_latent, mode="decode", no_grad=True
                )["recon"]
                pred_sr_from_latent = None

            # loss
            sr_loss, sr_log_losses = None, {}
            if sr is not None or sr_latent is not None:
                sr_loss = self.pansp_loss(
                    pred_latent, sr_latent, pred_sr, sr, pred_sr_from_latent, sr
                )

                if isinstance(sr_loss, torch.Tensor):
                    sr_log_losses = {"pansharp_loss": sr_loss.detach()}
                elif isinstance(sr_loss, (tuple, list)):
                    sr_loss, sr_log_losses = sr_loss
                else:
                    raise NotImplementedError(f"Unknown type {type(sr_loss)}")
            sr_loss = cast(torch.Tensor, sr_loss)

        return PansharpeningOutput(
            pred_latent, pred_sr, pred_sr_from_latent, sr_loss, sr_log_losses
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
        out = self.forward_pansp_model(
            lrms, pan, sr, lrms_latent, pan_latent, sr_latent
        )

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

    def _to_tokenizer_range(self, x: torch.Tensor):
        if not self.train_cfg.to_neg_1_1:  # dataset gives data range [0, 1]
            return x * 2 - 1
        return x

    def train_step(self, batch: BatchInput):
        lrms_latent = pan_latent = hrms_latent = None
        batch = self._cast_to_dtype(batch)

        # offline latents
        if keys_in_dict(["lrms_latent", "pan_latent", "hrms_latent"], batch):  # type: ignore
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
            lrms, pan, hrms = batch["lrms"], batch["pan"], batch["hrms"]
            lrms, pan, hrms = map(self._to_tokenizer_range, (lrms, pan, hrms))
            pan = pan.repeat_interleave(lrms.shape[1], dim=1)
            lrms_latent = self.forward_tokenizer(lrms)["latent"]
            pan_latent = self.forward_tokenizer(pan)["latent"]
            hrms_latent = self.forward_tokenizer(hrms)["latent"]

        quality_track_n = self.train_cfg.track_metrics_duration
        # start track after n steps
        quality_track_after = self.train_cfg.track_metrics_after
        if quality_track_n >= 0:
            ratio = self.train_cfg.pansp_ratio
            self.pan_acc_reduced = PansharpeningMetrics(
                ratio=ratio, ref=True, ergas_ratio=ratio
            )
            self.pan_acc_reduced_latent = PansharpeningMetrics(
                ratio=ratio, ref=True, ergas_ratio=ratio
            )

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

            # track reconstruction quality
            if (
                quality_track_n >= 0
                and self.global_step % quality_track_n != 0
                and self.global_step >= quality_track_after
            ):
                self.check_quality(
                    self.pan_acc_reduced, pred_sr=pred_img, gt=batch["hrms"]
                )
                self.check_quality(
                    self.pan_acc_reduced_latent,
                    pred_sr=pred_img,
                    gt=self.forward_tokenizer(hrms_latent, mode="decode"),
                )

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

        if (
            quality_track_n >= 0
            and self.global_step % quality_track_n == 0
            and self.global_step >= quality_track_after
        ):
            self.log_msg(f"[Real GT Metrics]: {self.pan_acc_reduced.print_str()}")
            self.log_msg(f"[Latent Metrics]: {self.pan_acc_reduced_latent.print_str()}")

    def format_log(self, log_sr_loss: dict) -> str:
        def dict_round_to_list_str(
            d: dict, n_round: int = 4, select: list[str] | None = None
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
            try:
                self.train_step(batch)
            except Exception as e:
                self.log_msg(
                    f"Training failed, batch keys are {batch.keys()}. {e}",
                    level="critical",
                )
                for k, v in batch.items():
                    self.log_msg(
                        f"{k}: {v.shape if hasattr(v, 'shape') else (v.__class__.__name__, v)}, "
                    )
                raise e

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

    def finite_val_full_loader(self):
        if self.val_full_dataloader is None:
            raise ValueError("No validation dataloader found")

        for batch in self.val_full_dataloader:
            return batch

    @torch.no_grad()
    def val_step(self, batch: BatchInput):
        batch = self._cast_to_dtype(batch)
        if self.online_tokenize:
            lrms, pan, hrms = batch["lrms"], batch["pan"], batch["hrms"]
            lrms, pan, hrms = map(self._to_tokenizer_range, (lrms, pan, hrms))
            pan = pan.repeat_interleave(lrms.shape[1], dim=1)
            lrms_latent = self.forward_tokenizer(lrms)["latent"]
            pan_latent = self.forward_tokenizer(pan)["latent"]
            gt_latent = self.forward_tokenizer(hrms)["latent"]
        else:
            lrms_latent = batch["lrms_latent"].to(self.device, self.dtype)
            pan_latent = batch["pan_latent"].to(self.device, self.dtype)
            gt_latent = batch["hrms_latent"].to(self.device, self.dtype)

        # forward the fusion network
        out = self.forward_pansp_model(
            batch["lrms"],
            batch["pan"],
            batch["hrms"],
            lrms_latent,
            pan_latent,
            gt_latent,
        )

        return out

    def _get_val_tbar_iter(self, val_loader, mode="reduced"):
        if self.val_cfg.max_val_iters > 0:
            # create a new iterator for the validation loader
            # state in the loader generator
            if not hasattr(self, "_val_loader_iter"):
                val_loader_iter = iter(val_loader)
                name = f"_val_{mode}_loader_iter"
                setattr(self, name, val_loader_iter)

            tbar = trange(
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
            tbar = self.finite_val_loader()
            self.log_msg(
                f"[Val]: start validating with the whole val set", only_rank_zero=False
            )
        return tbar

    def val_loop(self):
        self.pansp_model.eval()
        torch.cuda.empty_cache()

        tbar = self._get_val_tbar_iter(self.finite_val_loader())
        # track psnr and ssim
        pan_acc_fn = AnalysisPanAcc(ratio=self.train_cfg.pansp_ratio, ref=True)
        # pan_acc_fn = PansharpeningMetrics(ratio=self.train_cfg.pansp_ratio, ref=True)
        loss_metrics = MeanMetric().to(device=self.device)

        self._val_reduced_loader_iter: Iterable
        for batch_or_idx in tbar:
            if self.val_cfg.max_val_iters > 0:
                batch = next(self._val_reduced_loader_iter)
            else:
                batch = batch_or_idx

            batch = cast(BatchInput, batch)
            gt = batch["hrms"].to(self.device)
            val_out = self.val_step(batch)

            # l1 loss
            pred_sr = val_out.pred_sr
            loss_of_latent = nn.functional.l1_loss(
                pred_sr, gt.to(pred_sr)
            )  # latent to img and then loss
            loss_metrics.update(loss_of_latent)

            # metrics
            pred_sr = self.to_rgb(pred_sr)
            gt = self.to_rgb(gt)
            pan_acc_fn(pred_sr, gt)

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
                gt,  # gt
                pred_sr,  # prediction
                add_step=True,
                img_name="val/pansharpened_reduced",
                no_to_rgb=True,
            )

    def val_full_loop(self):
        self.pansp_model.eval()
        torch.cuda.empty_cache()

        tbar = self._get_val_tbar_iter(self.finite_val_full_loader(), mode="full")
        # track psnr and ssim
        pan_acc_fn = PansharpeningMetrics(ratio=self.train_cfg.pansp_ratio, ref=False)
        loss_metrics = MeanMetric().to(device=self.device)

        self._val_full_loader_iter: Iterable
        for batch_or_idx in tbar:
            if self.val_cfg.max_val_iters > 0:
                batch = next(self._val_full_loader_iter)
            else:
                batch = batch_or_idx

            batch = cast(BatchInput, batch)
            val_out = self.val_step(batch)

            pred_sr, lrms, pan = map(
                self.to_rgb,
                [
                    val_out.pred_sr,
                    batch["lrms"].to(self.device),
                    batch["pan"].to(self.device),
                ],
            )
            pan_acc_fn(pred_sr, lrms, pan)

        pan_acc = pan_acc_fn._get_acc_ave()
        loss_val = loss_metrics.compute()

        if self.accelerator.is_main_process:
            _metric_str = ""
            for k, v in pan_acc.items():
                _metric_str += f"{k}: {v:.4f} - "
            self.log_msg(f"[Val]: {_metric_str} loss: {loss_val:.4f}")
            self.tenb_log_any("metric", pan_acc, step=self.global_step)

            # visualize the last val batch
            self.visualize_reconstruction(
                pred_sr,  # gt
                add_step=True,
                img_name="val/pansharpened_full",
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
            self.ema_pansp_model.ema_model, ema_path / "pansharp_model"
        )
        # train state
        _ema_path_state_train = ema_path / "train_state.pth"
        _ema_path_state_train.parent.mkdir(parents=True, exist_ok=True)
        accelerate.utils.save(self.train_state.state_dict(), _ema_path_state_train)
        self.log_msg(f"[Ckpt]: save ema at {ema_path}")

    def load_from_ema(self, ema_path: str | Path, strict: bool = True):
        ema_path = Path(ema_path)

        accelerate.load_checkpoint_in_model(
            self.pansp_model, ema_path / "pansharp_model", strict=strict
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
        x: torch.Tensor,
        recon: torch.Tensor | None = None,
        img_name: str = "pansharpening_reduced",
        add_step: bool = False,
        only_vis_n: int | None = None,
        no_to_rgb: bool = False,
    ):
        if not no_to_rgb:
            x = self.to_rgb(x)
        if recon is not None and not no_to_rgb:
            recon = self.to_rgb(recon)

        _only_n = only_vis_n or 16
        to_img = lambda x: tensor_to_image(
            make_grid(x[:_only_n].float(), n_row=4, padding=2)
        )
        c = x.shape[1]

        def hyperspectral_to_rgb(x):
            # is rgb or gray images
            if c in (1, 3):
                x_np = to_img(x)
            else:
                rgb_channels = to_cont(
                    self.dataset_cfg.consts.rgb_channels
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
        if self.train_cfg.resume_path is not None:
            self.resume(self.train_cfg.resume_path)
        elif self.train_cfg.ema_load_path is not None:
            self.load_from_ema(self.train_cfg.ema_load_path)

        # train !
        self.train_loop()


_key = "tokenizer_lora_vitamin"
_configs = {
    "tokenizer_lora_vitamin": "tokenizer_lora_vitamin",
}[_key]


@hydra.main(
    config_path="../configs/pansharpening",
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

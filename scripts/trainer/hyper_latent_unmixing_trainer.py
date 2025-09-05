import sys
import time
from collections import namedtuple
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Callable, Literal, Sequence, TypedDict, cast

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
from ema_pytorch import EMA
from kornia.utils.image import make_grid, tensor_to_image
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from torch import Tensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._fully_shard import FSDPModule
from torch.distributed.tensor import DTensor
from torchmetrics.aggregation import MeanMetric
from tqdm import trange

from src.stage1.cosmos.inference.utils import load_jit_model_shape_matched
from src.stage1.cosmos.lora_mixin import TokenizerLoRAMixin
from src.stage2.unmixing.loss import UnmixingLoss
from src.stage2.unmixing.metrics import UnmixingMetrics, endmembers_visualize
from src.utilities.config_utils import (
    to_object as to_cont,  # register new resolvers at the same time
)
from src.utilities.logging import dict_round_to_list_str
from src.utilities.network_utils import load_peft_model_checkpoint
from src.utilities.network_utils.Dtensor import safe_dtensor_operation
from src.utilities.train_utils.state import StepsCounter, dict_tensor_sync, metrics_sync
from stage2.unmixing.models.model import LatentUnmixingModel

heavyball = lazy_loader.load("heavyball")

UnmixingModelOutput = namedtuple(
    "UnmixingModelOutput",
    [
        "pred_latent",
        "pred_pixel",
        "abunds",
        "endmembers",
        "unmixing_loss",
        "log_losses",
    ],
)
BatchInput = TypedDict(
    "BatchInput",
    {
        "img": Tensor,
        "endmembers": Tensor,
        "init_vca_indices": Tensor,
        "init_vca_endmembers": Tensor,
        "abunds": Tensor,
    },
)


class UnmixingTrainer:
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
        self.setup_unmixing_model()  # setup the denoising model

        # optimizers and lr schedulers
        self.optim, self.sched = self.get_optimizer_lr_scheduler()

        # EMA models and accelerator prepare
        self.prepare_for_training()
        self.prepare_ema_models()

        # loss
        self.unmixing_loss: UnmixingLoss
        if hasattr(self.train_cfg, "unmixing_loss"):
            self.unmixing_loss = hydra.utils.instantiate(self.train_cfg.unmixing_loss)
        else:
            # default loss - use UnmixingLoss with default weights
            self.unmixing_loss = UnmixingLoss()
        self.log_msg(f"use denoising loss: {self.unmixing_loss.__class__.__name__}")

        # training state counter
        self.train_state = StepsCounter(["train", "val"])

        # initialize unmixing metrics
        self.unmixing_metrics = UnmixingMetrics()

        # clear GPU memory
        torch.cuda.empty_cache()

    def setup_unmixing_model(self):
        self.model = hydra.utils.instantiate(self.cfg.unmixing_model)
        self.unmixing_amotizing_pixels = self.accelerator.unwrap_model(
            self.model
        ).amotizing_pixels
        if self.unmixing_amotizing_pixels:
            assert hasattr(self, "tokenizer"), (
                "unmixing model must be used with tokenizer for amotizing pixels"
            )
            self.backward_detokenizer = True
        else:
            self.backward_detokenizer = self.train_cfg.backward_detokenizer

        pansp_name = (
            getattr(self.train_cfg, "unmixing_name", None)
            or self.model.__class__.__name__
        )
        self.log_msg(
            f"use unmixing model: {pansp_name}, amotizing pixels: {self.unmixing_amotizing_pixels}"
        )

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
            self.tokenizer: TokenizerLoRAMixin | PeftModel | nn.Module = (
                hydra.utils.instantiate(self.cfg.tokenizer)
            )

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
        self._model_inited_endmembers = False

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
                self.train_cfg.unmixing_optim, _get_model_params
            )
        else:
            model_opt = DummyOptim([{"params": list(self.model.parameters())}])

        # schedulers
        if (
            self.accelerator.state.deepspeed_plugin is None
            or "scheduler"
            not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            model_sched = hydra.utils.instantiate(self.train_cfg.unmixing_sched)(
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
                "for efficiently testing the scripts, disable the compilation."
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

    def ema_update(self, mode="unmixing"):
        assert mode == "unmixing"
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

    def _init_model_endmembers(self, endmembers_init: torch.Tensor):
        # init the endmembers of the unmixing model
        if not self._model_inited_endmembers:
            model = self.accelerator.unwrap_model(self.model)
            model = cast(LatentUnmixingModel, model)
            model.init_endmembers(endmembers_init)
            self._model_inited_endmembers = True
            self.log_msg("initialize the endmembers of the unmixing model")

    def forward_unmixing_model(
        self, img, img_latent, abunds_gt=None, endmember_gt=None
    ):
        with self.accelerator.autocast():
            # Forward pass through the model to get unmixing results
            model_output = self.model(img, img_latent)

        pred_latent = model_output.get("amotized_model_out", None)
        recon = model_output["recon"]
        abunds = model_output["abunds"]
        endmembers = model_output["endmembers"]

        # loss
        # Use UnmixingLoss with the required inputs
        loss, loss_dict = self.unmixing_loss(
            hyper_in=img,
            hyper_recon=recon,
            abunds=abunds,
            endmembers=endmembers,
        )

        # Create log_losses dictionary
        log_losses = loss_dict
        log_losses["total_loss"] = loss.detach()

        return UnmixingModelOutput(
            pred_latent=pred_latent,
            pred_pixel=recon,
            abunds=abunds,
            endmembers=endmembers,
            unmixing_loss=loss,
            log_losses=log_losses,
        )

    def train_unmixing_step(
        self,
        img,
        img_latent: torch.Tensor | None = None,
        abunds_gt: torch.Tensor | None = None,
        endmember_gt: torch.Tensor | None = None,
    ):
        out = self.forward_unmixing_model(img, img_latent, abunds_gt, endmember_gt)

        if self.accelerator.sync_gradients:
            # backward
            self.optim.zero_grad()
            self.accelerator.backward(out.unmixing_loss)
            self.gradient_check(self.model)
            self.optim.step()
            self.sched.step()

            # ema update
            self.ema_update(mode="unmixing")

        return out

    def update_metrics(
        self,
        endmembers_pred,
        abunds_pred,
        endmembers_gt,
        abunds_gt=None,
        plot=False,
        clear=False,
    ):
        if clear:
            self.unmixing_metrics.reset()

        return self.unmixing_metrics(
            endmembers_pred,
            endmembers_gt,
            abunds_pred,
            abunds_gt,
            plot=plot,
        )

    def get_metrics(self):
        return self.unmixing_metrics._compute_metrics()

    def train_step(self, batch: BatchInput):
        # < Latent obtaining
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

        # Init model endmembers
        self._init_model_endmembers(batch["init_vca_endmembers"].to(self.device))

        # < Unmixing training step
        with self.accelerator.accumulate(self.model):
            train_out = self.train_unmixing_step(
                batch["img"], img_latent, batch["abunds"], batch["endmembers"]
            )

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
    def val_step(self, batch: BatchInput):
        if not self.online_tokenize:
            img_latent = self.forward_tokenizer(batch["img"])["latent"]
        else:
            img_latent = batch.get("img_latent", None)
            assert img_latent is not None, "img_latent must be provided in the batch"
        img_latent = img_latent.to(self.device, self.dtype)

        # forward the unmixing network
        out = self.forward_unmixing_model(
            batch["img"], img_latent, batch["abunds"], batch["endmembers"]
        )

        return out

    def val_loop(self):
        self.model.eval()

        val_iter = self.get_val_loader_iter()
        # Init loss dict metrics
        loss_metrics = {k: MeanMetric() for k in self.unmixing_loss.loss_names}
        loss_metrics_update = lambda log_losses: {
            k: v.update(log_losses[k]) for k, v in loss_metrics.items()
        }

        for batch_or_idx in val_iter:  # type: ignore
            if self.val_cfg.max_val_iters > 0:
                batch = next(self._val_loader_iter)
            else:
                batch = batch_or_idx
            batch = cast(BatchInput, batch)

            endmembers_gt = batch["endmembers"].to(self.device)
            abundances_gt = batch["abunds"].to(self.device)

            val_out = self.val_step(batch)

            log_losses = val_out.log_losses
            loss_metrics_update(log_losses)

            # metrics
            self.update_metrics(
                abunds_gt=abundances_gt,
                endmembers_gt=endmembers_gt,
                abunds_pred=val_out.abunds,
                endmembers_pred=val_out.endmembers,
            )
            self.step_train_state("val")

        metrics = self.get_metrics()
        loss_val = metrics_sync(loss_metrics, output_tensor_dict=True)

        if self.accelerator.is_main_process:
            # visualize the last val batch
            self.visualize_reconstruction(
                batch_img_rgb,  # gt
                pred_img_rgb,  # prediction
                add_step=True,
                img_name="val/denoising",
                no_to_rgb=True,
            )

            # visualize unmixing results if available
            if (
                val_out.get("abunds") is not None
                and val_out.get("end_member") is not None
            ):
                self.visualize_unmixing_results(
                    val_out["abunds"],
                    val_out["end_member"],
                    add_step=True,
                    img_name="val/unmixing",
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
        return ((x + 1) / 2).clamp(0, 1).float()

    def visualize_reconstruction(
        self,
        x: torch.Tensor,
        recon: torch.Tensor | None = None,
        img_name: str = "train_original_recon",
        add_step: bool = False,
        only_vis_n: int | None = None,
        no_to_rgb: bool = False,
    ):
        if not no_to_rgb:
            x = self.to_rgb(x)
        if recon is not None and not no_to_rgb:
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

    def visualize_unmixing_results(
        self,
        abunds: torch.Tensor,
        end_members: torch.Tensor,
        img_name: str = "unmixing_results",
        add_step: bool = False,
        only_vis_n: int | None = None,
    ):
        """
        Visualize unmixing results including abundance maps and endmember spectra

        Args:
            abunds: Abundance maps [batch, num_endmembers, height, width]
            end_members: Endmember spectra [num_endmembers, num_bands]
            img_name: Base name for the visualization
            add_step: Whether to add step number to the filename
            only_vis_n: Number of samples to visualize
        """
        if abunds is None or end_members is None:
            self.log_msg("[Visualize]: Skip unmixing visualization due to missing data")
            return

        _only_n = only_vis_n or min(16, abunds.shape[0])
        abunds = abunds[:_only_n]

        # Move to CPU for visualization
        abunds_cpu = abunds.detach().cpu()
        end_members_cpu = end_members.detach().cpu()

        # Create abundance maps visualization
        num_endmembers = abunds_cpu.shape[1]

        # Create a grid of abundance maps
        abundance_images = []
        for i in range(num_endmembers):
            # Take first sample in batch for abundance visualization
            abundance_map = abunds_cpu[0, i]  # [height, width]
            abundance_images.append(abundance_map)

        # Create endmember spectra plot
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Plot endmember spectra
        for i in range(num_endmembers):
            axes[0].plot(end_members_cpu[i].numpy(), label=f"Endmember {i + 1}")
        axes[0].set_xlabel("Band Index")
        axes[0].set_ylabel("Reflectance")
        axes[0].set_title("Endmember Spectra")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot abundance maps as grid
        n_cols = min(4, num_endmembers)
        n_rows = (num_endmembers + n_cols - 1) // n_cols

        # Clear the second subplot and create subplots for abundance maps
        axes[1].remove()

        # Create abundance map subplots
        abundance_axes = []
        for i in range(num_endmembers):
            row = i // n_cols
            col = i % n_cols
            if i == 0:
                ax = fig.add_subplot(2, n_cols, n_cols + 1)
            else:
                ax = fig.add_subplot(2, n_cols, n_cols + 1 + i)

            im = ax.imshow(abundance_images[i], cmap="viridis", aspect="auto")
            ax.set_title(f"Abundance {i + 1}")
            ax.set_xticks([])
            ax.set_yticks([])
            abundance_axes.append(ax)

            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()

        # Save the figure
        if add_step:
            img_name = f"{img_name}_step_{str(self.global_step).zfill(6)}.png"
        else:
            img_name = f"{img_name}.png"

        save_path = Path(self.proj_dir) / "vis" / img_name
        if self.accelerator.is_main_process:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        self.log_msg(f"[Visualize]: save unmixing visualization at {save_path}")

        # Also save abundance maps as individual images
        for i in range(num_endmembers):
            abundance_img = tensor_to_image(
                abundance_images[i].unsqueeze(0).unsqueeze(0)
            )
            abundance_img = (abundance_img * 255.0).astype(np.uint8)

            if add_step:
                abundance_img_name = f"{img_name}_abundance_{i + 1}_step_{str(self.global_step).zfill(6)}.jpg"
            else:
                abundance_img_name = f"{img_name}_abundance_{i + 1}.jpg"

            abundance_save_path = Path(self.proj_dir) / "vis" / abundance_img_name
            if self.accelerator.is_main_process:
                abundance_img_to_save = Image.fromarray(abundance_img)
                abundance_img_to_save.save(abundance_save_path, quality=95)

            self.log_msg(f"[Visualize]: save abundance map at {abundance_save_path}")

    def run(self):
        if self.train_cfg.resume_path is not None:
            self.resume(self.train_cfg.resume_path)
        elif self.train_cfg.ema_load_path is not None:
            self.load_from_ema(self.train_cfg.ema_load_path)

        # train !
        self.train_loop()


_key = "cosmos_f8c16p4_dino_unmixing"  # vq, bsq, fsq, kl
_configs = {
    # cosmos tokenizer, simple denoising
    "cosmos_f8c16p4_dino_unmixing": "cosmos_f8c16p4_dino_unmixing",
}[_key]


@hydra.main(
    config_path="configs/unmixing",
    config_name=_configs,
    version_base=None,
)
def main(cfg):
    catcher = logger.catch if PartialState().is_main_process else nullcontext

    with catcher():
        trainer = UnmixingTrainer(cfg)
        trainer.run()


if __name__ == "__main__":
    main()

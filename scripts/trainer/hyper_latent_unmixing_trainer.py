import math
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
from torch import Tensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor
from torchmetrics.aggregation import MeanMetric
from tqdm import trange
from typing_extensions import Generator

from src.data.window_slider import WindowSlider
from src.stage2.layers.wrapper.tokenizer_wrapper import DownstreamModelTokenizerWrapper
from src.stage2.unmixing.loss import UnmixingLoss
from src.stage2.unmixing.metrics import (
    UnmixingMetrics,
    abunds_visualize,
    endmembers_visualize,
)
from src.utilities.config_utils import (
    to_object as to_cont,  # register new resolvers at the same time
)
from src.utilities.logging import dict_round_to_list_str
from src.utilities.logging.print import log_print
from src.utilities.network_utils.Dtensor import safe_dtensor_operation
from src.utilities.train_utils.state import StepsCounter, dict_tensor_sync, metrics_sync
from src.utilities.train_utils.visualization import get_rgb_image

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
        "img_latent": Tensor | None,
        "endmembers": Tensor,
        "init_vca_indices": Tensor,
        "init_vca_endmembers": Tensor,
        "init_vca_abunds": Tensor | None,
        "abunds": Tensor,
    },
)


class UnmixingTrainer:
    def __init__(self, cfg: DictConfig):
        """Initialize the unmixing trainer.

        Parameters
        ----------
        cfg : DictConfig
            Configuration dictionary containing all settings
        """
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
            self.dataset_cfg.train_loader
        )
        self.val_dataset, self.val_dataloader = hydra.utils.instantiate(
            self.dataset_cfg.val_loader
        )
        if _dpsp_plugin is not None:
            self.accelerator.deepspeed_plugin.deepspeed_config[  # type: ignore
                "train_micro_batch_size_per_gpu"
            ] = self.dataset_cfg.batch_size_train

        # setup the unmixing model
        self._model_em_init = self.setup_unmixing_model()  # setup the unmixing model

        # optimizers and lr schedulers
        self.optim, self.sched = self.get_optimizer_lr_scheduler()

        # EMA models and accelerator prepare
        self.prepare_for_training()
        self.prepare_ema_models()

        # training state counter
        self.train_state = StepsCounter(["train", "val"])

        # loss
        self.unmixing_loss: UnmixingLoss
        if hasattr(self.train_cfg, "unmixing_loss"):
            self.unmixing_loss = hydra.utils.instantiate(self.train_cfg.unmixing_loss)
        else:
            # default loss - use UnmixingLoss with default weights
            self.unmixing_loss = UnmixingLoss()
        self.log_msg(f"use denoising loss: {self.unmixing_loss.__class__.__name__}")

        # initialize unmixing metrics
        self.unmixing_metrics = UnmixingMetrics()

        # clear GPU memory
        torch.cuda.empty_cache()

    def setup_unmixing_model(self) -> Callable[..., None]:
        self.model: DownstreamModelTokenizerWrapper = hydra.utils.instantiate(
            self.cfg.unmixing_model
        ).to(self.device, self.dtype)
        self.unmixing_amotizing_pixels = self.model.downstream_model.amotizing_pixels

        pansp_name = (
            getattr(self.train_cfg, "unmixing_name", None)
            or self.model.__class__.__name__
        )
        self.log_msg(
            f"use unmixing model: {pansp_name}, amotizing pixels: {self.unmixing_amotizing_pixels}"
        )

        # Set gradient checkpointing
        if self.train_cfg.set_grad_checkpoint and hasattr(
            self.model, "set_grad_checkpointing"
        ):
            self.model.set_grad_checkpoint(True)

        # Initialize the model endmember using VCA
        self._inited_endmember = False
        if self.train_cfg.init_endmembers:
            assert hasattr(self.model.downstream_model, "init_endmembers"), (
                "The model does not support endmember initialization"
            )

            # Return the a function that init the endmembers of the unmixing model
            # and then for other steps we skip the initialization
            def _model_init_EM_delayed(endmembers: Tensor):
                endmembers.squeeze_(0)  # remove batch dim
                assert endmembers.ndim == 2, "endmembers must be 2D"
                if endmembers.shape[0] < endmembers.shape[1]:  # [c, d]
                    endmembers = endmembers.T
                self.model.downstream_model.init_endmembers(endmembers)  # type: ignore
                self._inited_endmember = True
                self.log_msg(
                    f"[Unmixing Model]: endmembers initialized with shape {endmembers.shape}"
                )
        else:
            _model_init_EM_delayed = lambda endmembers: None
        return _model_init_EM_delayed

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
            "- {level.icon} <level>{level} - [{file.name}:{line}]</level>"
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

        _n_params_sumed = 0
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
            norms: dict[str, float] = {}
            if log_type == "grad_norm_sum":
                norms[f"{model_cls_n}_grad_norm"] = 0
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
                assert _n_params_sumed > 0
                norms[f"{model_cls_n}_grad_norm"] /= float(_n_params_sumed)
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

        log_fn = partial(
            log_print,
            level=level.lower(),
            only_rank_zero=only_rank_zero,
            stack_level=3,
            **kwargs,
        )

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
            self.model.dtype = torch.float

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

    def forward_unmixing_model(
        self,
        img: torch.Tensor,
        img_latent: torch.Tensor | None,
        abunds_gt: torch.Tensor | None = None,
        abunds_fcls: torch.Tensor | None = None,  # initial abundances
        endmember_gt: torch.Tensor | None = None,
        ema: bool = False,
        grad=True,
    ) -> UnmixingModelOutput:
        """
        Forward pass through the unmixing model.

        Parameters
        ----------
        img : torch.Tensor
            Input hyperspectral image
        img_latent : torch.Tensor | None
            Latent representation, if None wrapper will encode img
        abunds_gt : torch.Tensor | None
            Ground truth abundances
        endmember_gt : torch.Tensor | None
            Ground truth endmembers

        Returns
        -------
        UnmixingModelOutput
            Model output containing predictions and losses
        """
        model = self.model if not ema else self.ema_model
        grad_ctx = torch.no_grad() if not grad else nullcontext()
        with self.accelerator.autocast():
            # Forward pass through the model to get unmixing results
            with grad_ctx:
                model_output = model(img, img_latent)

            pred_latent = model_output.get("amotized_model_out", None)
            recon = model_output["recon"]
            abunds = model_output["abunds"].to(recon.dtype)
            endmembers = model_output["endmembers"].to(recon.dtype)

            # loss
            # Use UnmixingLoss with the required inputs
            loss, loss_dict = self.unmixing_loss(
                hyper_in=img,
                hyper_recon=recon,
                abunds_pred=abunds,
                endmembers=endmembers,
                abunds_fcls=abunds_fcls,
            )

            # if self.get_global_step() == 0:
            #     self.log_msg(f"Init the model and check the correctness")
            #     self.visualize_unmixing_results(
            #         abunds,
            #         endmembers,
            #         endmember_gt,
            #         abunds_gt,
            #         img_name="tmp/unmixing_init_model_out",
            #     )
            #     self.visualize_unmixing_results(
            #         abunds_fcls,
            #         endmembers,
            #         endmember_gt,
            #         abunds_gt,
            #         img_name="tmp/unmixing_init_fcls",
            #     )

            #     import time

            #     time.sleep(4)

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
        abunds_fcls: torch.Tensor | None = None,
        endmember_gt: torch.Tensor | None = None,
    ):
        out = self.forward_unmixing_model(
            img, img_latent, abunds_gt, abunds_fcls, endmember_gt
        )

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
            endmembers_gt.squeeze(0),
            abunds_pred.squeeze(0),
            abunds_gt.squeeze(0) if abunds_gt is not None else None,
            plot=plot,
        )

    def get_metrics(self):
        return self.unmixing_metrics._compute_metrics()

    def train_step(self, batch: BatchInput):
        """
        Execute one training step for unmixing model.

        Parameters
        ----------
        batch : BatchInput
            Batch input containing image, abundances, endmembers, and optionally latents

        Note
        ----
        The wrapper handles tokenization internally. If latents are not provided in the batch,
        the wrapper will encode the image to latents automatically.
        """
        # Get latents from batch - wrapper handles tokenization
        img_latent = None
        if "img_latent" in batch and batch["img_latent"] is not None:
            img_latent = batch["img_latent"].to(self.device, self.dtype)

        # Init model endmembers
        if not self._inited_endmember and self.train_cfg.init_endmembers:
            self._model_em_init(batch["init_vca_endmembers"].to(self.device))

        # < Unmixing training step
        with self.accelerator.accumulate(self.model):
            train_out = self.train_unmixing_step(
                batch["img"],
                img_latent,
                batch["abunds"],
                batch.get("init_vca_abunds", None),
                batch["endmembers"],
            )

        self.step_train_state()

        # log losses
        if self.global_step % self.train_cfg.log.log_every == 0:
            _log_losses = self.format_log(train_out.log_losses)

            self.log_msg(
                f"[Train State]: lr {self.optim.param_groups[0]['lr']:1.4e} | "
                f"[Step]: {self.global_step}/{self.train_cfg.max_steps}"
            )
            self.log_msg(f"[Train Unmixing]: {_log_losses}")

            # tensorboard log
            self.tenb_log_any("metric", train_out.log_losses, self.global_step)

    def format_log(self, log_loss: dict, sync=False) -> str:
        if sync:
            log_loss = dict_tensor_sync(log_loss)
        strings = dict_round_to_list_str(
            log_loss, select=list(log_loss.keys()), n_round=5
        )
        return " - ".join(strings)

    def _cast_dtype(self, x: dict | Tensor):
        if isinstance(x, dict):
            return {
                k: v.to(self.device, self.dtype) if torch.is_tensor(v) else v
                for k, v in x.items()
            }
        elif isinstance(x, Tensor):
            return x.to(self.device, self.dtype)
        else:
            raise ValueError("Input must be a dict or Tensor")

    def infinity_train_loader(self):
        while True:
            w_size = self.dataset_cfg.cfgs.window_size
            stride = self.dataset_cfg.cfgs.stride
            if w_size > 0:
                slider = WindowSlider(
                    slide_keys=["img", "abunds"], window_size=w_size, stride=stride
                )
                for batch in slider.create_window_generator(self.train_dataloader):
                    yield self._cast_dtype(batch)
            else:
                for batch in self.train_dataloader:
                    yield self._cast_dtype(batch)

    def train_loop(self):
        _stop_train_and_save = False
        self.accelerator.wait_for_everyone()

        self.model.train()
        self.log_msg("[Train]: start training", only_rank_zero=False)
        for batch in self.infinity_train_loader():
            # train step
            self.train_step(batch)

            if self.global_step % self.val_cfg.val_duration == 0:
                # self.model.eval()
                # self.ema_model.ema_model.eval()
                __ps = {n: p.clone() for n, p in self.model.named_parameters()}
                self.optim.zero_grad()
                self.val_loop()
                # self.model.train()
                for (name, _p), (_, _p_val) in zip(
                    __ps.items(), self.model.named_parameters()
                ):
                    if not torch.allclose(_p, _p_val, atol=1e-7):
                        self.log_msg(
                            f"{name} changed after val loop, max diff: {(_p - _p_val).abs().max()}",
                            level="WARNING",
                        )

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

        val_w_size = getattr(self.val_cfg, "window_size", 0)
        val_stride = getattr(self.val_cfg, "stride", 0)
        self._is_val_sliding_window = False

        if val_w_size > 0:
            val_slider = WindowSlider(
                slide_keys=["img", "abunds"], window_size=val_w_size, stride=val_stride
            )
            self._is_val_sliding_window = True
            self._val_slider = val_slider
            for batch in val_slider.create_window_generator(self.val_dataloader):
                yield self._cast_dtype(batch)

        else:
            for batch in self.val_dataloader:
                yield self._cast_dtype(batch)

        # reload the val_loader if is a generator
        if isinstance(self.val_dataloader, Generator):
            self.log_msg("[Val]: reload the val_dataloader")
            self.val_dataloader = self.val_dataset()
            assert isinstance(self.val_dataloader, Generator), (
                "val_dataloader must be a generator"
            )

    def val_step(self, batch: BatchInput):
        """
        Execute one validation step for unmixing model.

        Parameters
        ----------
        batch : BatchInput
            Batch input containing image, abundances, endmembers, and optionally latents

        Returns
        -------
        UnmixingModelOutput
            Model output containing predictions and losses

        Note
        ----
        The wrapper handles tokenization internally. If latents are not provided in the batch,
        the wrapper will encode the image to latents automatically.
        """
        # Get latents from batch - wrapper handles tokenization
        img_latent = batch.get("img_latent", None)
        if img_latent is not None:
            img_latent = img_latent.to(self.device, self.dtype)

        # forward the unmixing network
        out = self.forward_unmixing_model(
            batch["img"],
            img_latent,
            batch["abunds"],
            batch.get("init_vca_abunds", None),
            batch["endmembers"],
            ema=False,
            grad=False,
        )

        return out

    @torch.inference_mode()
    def val_loop(self):
        val_iter = self._finite_val_loader()
        # Init loss dict metrics
        loss_metrics = {
            k: MeanMetric().to(self.device) for k in self.unmixing_loss.loss_names
        }
        loss_metrics_update = lambda log_losses: {
            k: v.update(log_losses[k]) for k, v in loss_metrics.items()
        }

        val_out = None
        for batch in val_iter:  # type: ignore
            batch = cast(BatchInput, batch)

            if "endmembers" not in batch:
                self.log_msg(
                    f"[Val]: this dataset does not provide the GT endmembers",
                    level="WARNING",
                    once=True,
                )
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

        # Print out metrics
        _log_losses = self.format_log(loss_val)
        _log_metrics = self.format_log(metrics)
        self.log_msg(f"[Val Unmixing Loss]: {_log_losses}")
        self.log_msg(f"[Val Unmixing Metrics]: {_log_metrics}")

        assert val_out is not None
        if self.accelerator.is_main_process:
            # visualize reconstruction if available
            if val_out.pred_pixel is not None:
                self.visualize_reconstruction(
                    batch["img"],  # original image
                    val_out.pred_pixel,  # reconstructed image
                    add_step=True,
                    img_name="val/reconstruction",
                )

            # visualize unmixing results if available
            if val_out.abunds is not None and val_out.endmembers is not None:
                self.visualize_unmixing_results(
                    val_out.abunds,
                    val_out.endmembers,
                    end_members_gt=endmembers_gt,
                    abunds_gt=abundances_gt,
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
            ema_path.mkdir(parents=True, exist_ok=True)
        self.accelerator.save_model(
            self.ema_model.ema_model, ema_path / "unmixing_ema_model"
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
        # return ((x + 1) / 2).clamp(0, 1).float()
        return x  # not [-1, 1]

    def visualize_reconstruction(
        self,
        x: torch.Tensor,
        recon: torch.Tensor | None = None,
        img_name: str = "train_original_recon",
        add_step: bool = False,
        only_vis_n: int | None = None,
    ):
        """
        Visualize reconstruction results for hyperspectral images.

        This function converts hyperspectral data to RGB for visualization
        and saves the results as image files.

        Parameters
        ----------
        x : torch.Tensor
            Input hyperspectral tensor
        recon : torch.Tensor | None, optional
            Reconstructed tensor, by default None
        img_name : str, optional
            Base name for the saved image, by default "train_original_recon"
        add_step : bool, optional
            Whether to add training step to filename, by default False
        only_vis_n : int | None, optional
            Number of images to visualize, by default None (uses 16)

        Notes
        -----
        This function handles both RGB and hyperspectral data by converting
        hyperspectral bands to RGB using predefined channel indices.
        """
        x = self.to_rgb(x)
        if recon is not None:
            recon = self.to_rgb(recon)

        _only_n = only_vis_n or 16
        _n_row = min(4, int(math.sqrt(_only_n)))
        to_img = lambda x: tensor_to_image(
            make_grid(x[:_only_n].float(), n_row=_n_row, padding=2)
        )
        vis_fn = lambda x: to_img(
            get_rgb_image(x, self.dataset_cfg.consts.rgb_channels, use_linstretch=False)
        )

        x_np = vis_fn(x)

        # concatenate original and reconstructed images
        if recon is not None:
            recon_np = vis_fn(recon)
            img = np.concatenate([x_np, recon_np], axis=1)
        else:
            img = x_np

        # save image
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

    def endmembers_visualize(
        self,
        abunds: torch.Tensor,
        end_members: torch.Tensor,
        end_members_gt: torch.Tensor | None = None,
        img_name: str = "unmixing_results",
        add_step: bool = False,
        only_vis_n: int | None = None,
    ):
        """
        Visualize unmixing results using the endmembers_visualize function from metrics module.

        This function creates comprehensive visualizations including endmember spectral
        comparison plots and abundance maps using the existing endmembers_visualize
        function from the metrics module.

        Parameters
        ----------
        abunds : torch.Tensor
            Predicted abundance maps with shape [batch, num_endmembers, height, width]
        end_members : torch.Tensor
            Predicted endmember spectra with shape [num_endmembers, num_bands]
        end_members_gt : torch.Tensor | None
            Ground truth endmember spectra with shape [num_endmembers, num_bands].
            If None, will use predicted endmembers for visualization purposes
        img_name : str
            Base name for the visualization files
        add_step : bool
            Whether to add step number to the filename
        only_vis_n : int | None
            Number of samples to visualize, if None uses min(16, batch_size)

        Notes
        -----
        This function creates two types of visualizations:
        1. Endmember spectral comparison plots with SAD values
        2. Individual abundance map images for each endmember

        The function leverages the existing endmembers_visualize function which
        handles endmember matching, SAD computation, and professional plotting.
        """
        if abunds is None or end_members is None:
            self.log_msg("[Visualize]: Skip unmixing visualization due to missing data")
            return

        # Limit the number of samples to visualize
        _only_n = only_vis_n or min(16, abunds.shape[0])
        abunds = abunds[:_only_n]

        # Use provided ground truth endmembers if available, otherwise use predicted
        # endmembers as fallback for visualization purposes
        if end_members_gt is None:
            end_members_gt = end_members.clone()
        end_members_gt = end_members_gt.squeeze(0)

        # Get the first sample's abundances for visualization
        # endmembers_visualize expects [num_endmembers, H, W]
        if abunds.dim() == 4:  # [batch, num_endmembers, H, W]
            abunds_vis = abunds[0]  # Take first sample
        else:
            abunds_vis = abunds  # Already [num_endmembers, H, W]

        # Use the endmembers_visualize function to create the plot
        # This returns: sad_values, ordered_endmembers, ordered_abundances, fig, axes
        sad_values, _, ordered_abundances, fig, _ = endmembers_visualize(
            end_members, end_members_gt, abunds_vis
        )

        # Save the endmember comparison plot
        if add_step:
            endmember_plot_name = (
                f"{img_name}_endmembers_step_{str(self.global_step).zfill(6)}.png"
            )
        else:
            endmember_plot_name = f"{img_name}_endmembers.png"

        save_path = Path(self.proj_dir) / "vis" / endmember_plot_name
        if self.accelerator.is_main_process:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        self.log_msg(f"[Visualize]: save endmember comparison at {save_path}")

        # Log average SAD value
        if len(sad_values) > 0:
            avg_sad = sad_values[-1].item()  # Last value is the average
            self.log_msg(f"[Visualize]: average SAD: {avg_sad:.4f} rad")

    def visualize_unmixing_results(
        self,
        abunds: torch.Tensor,
        end_members: torch.Tensor,
        end_members_gt: torch.Tensor | None = None,
        abunds_gt: torch.Tensor | None = None,
        img_name: str = "unmixing_results",
        add_step: bool = False,
        only_vis_n: int | None = None,
    ):
        """
        Visualize unmixing results including endmembers and abundance maps.

        This function creates comprehensive visualizations using the metrics module functions:
        1. Endmember spectral comparison plots with SAD values
        2. Abundance maps comparison with ground truth if available

        Parameters
        ----------
        abunds : torch.Tensor
            Predicted abundance maps with shape [batch, num_endmembers, height, width]
        end_members : torch.Tensor
            Predicted endmember spectra with shape [num_endmembers, num_bands]
        end_members_gt : torch.Tensor | None
            Ground truth endmember spectra with shape [num_endmembers, num_bands]
        abunds_gt : torch.Tensor | None
            Ground truth abundance maps with shape [batch, num_endmembers, height, width]
        img_name : str
            Base name for the visualization files
        add_step : bool
            Whether to add step number to the filename
        only_vis_n : int | None
            Number of samples to visualize, if None uses min(16, batch_size)

        Notes
        -----
        This function leverages the existing visualization functions from the metrics module:
        - endmembers_visualize: for endmember spectral comparison
        - abunds_visualize: for abundance map comparison with ground truth
        """
        if abunds is None or end_members is None:
            self.log_msg("[Visualize]: Skip unmixing visualization due to missing data")
            return

        # all to float
        to_float = lambda x: x.to(torch.float32)
        abunds, end_members = map(to_float, [abunds, end_members])
        if end_members_gt is not None:
            end_members_gt = to_float(end_members_gt)
        if abunds_gt is not None:
            abunds_gt = to_float(abunds_gt)

        # Limit the number of samples to visualize
        _only_n = only_vis_n or min(16, abunds.shape[0])
        abunds = abunds[:_only_n]

        # Visualize endmembers
        self.endmembers_visualize(
            abunds, end_members, end_members_gt, f"{img_name}_endmembers", add_step
        )

        # Visualize abundance maps if ground truth is available
        if abunds_gt is not None:
            # Limit ground truth to same number of samples
            abunds_gt = abunds_gt[:_only_n]

            # Get the first sample for abundance visualization
            if abunds.dim() == 4:  # [batch, num_endmembers, H, W]
                abunds_vis = abunds[0]  # Take first sample
                abunds_gt_vis = abunds_gt[0]  # Take first sample
            else:
                abunds_vis = abunds  # Already [num_endmembers, H, W]
                abunds_gt_vis = abunds_gt  # Already [num_endmembers, H, W]

            # For abundance visualization, we need to create dummy ground truth endmembers
            # since abunds_visualize expects both predicted and ground truth endmembers
            dummy_endmembers_gt = end_members.clone()

            # Use the abunds_visualize function to create abundance comparison plots
            fig, axes = abunds_visualize(
                end_members, dummy_endmembers_gt, abunds_vis, abunds_gt_vis
            )

            # Save the abundance comparison plot
            if add_step:
                abundance_plot_name = (
                    f"{img_name}_abunds_step_{str(self.global_step).zfill(6)}.png"
                )
            else:
                abundance_plot_name = f"{img_name}_abunds.png"

            save_path = Path(self.proj_dir) / "vis" / abundance_plot_name
            if self.accelerator.is_main_process:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close(fig)

            self.log_msg(f"[Visualize]: save abundance comparison at {save_path}")
        else:
            self.log_msg(
                "[Visualize]: No ground truth abundances provided, skipping abundance comparison"
            )

    def run(self):
        if self.train_cfg.resume_path is not None:
            self.resume(self.train_cfg.resume_path)
        elif self.train_cfg.ema_load_path is not None:
            self.load_from_ema(self.train_cfg.ema_load_path)

        # train !
        self.train_loop()


_key = "resnet"
_configs = {
    "vitamin": "cosmos_f8c16p1_vitamin_unmixing",
    "resnet": "cosmos_f8c16p1_resnet_unmixing",
}


if __name__ == "__main__":
    from src.utilities.train_utils.cli import argsparse_cli_args, print_colored_banner

    # change the config name in cli
    cli_default_dict = {
        "config_name": _key,
        "only_rank_zero_catch": True,
    }
    chosen_cfg, cli_args = argsparse_cli_args(_configs, cli_default_dict)

    if is_rank_zero := PartialState().is_main_process:
        print_colored_banner("HyperUnmixing")
    logger.info(
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

    @hydra.main(
        config_path="../configs/unmixing",
        config_name=chosen_cfg,
        version_base=None,
    )
    def main(cfg):
        if cli_args.only_rank_zero_catch:
            catcher = logger.catch if is_rank_zero else nullcontext
        else:
            catcher = logger.catch

        with catcher():
            trainer = UnmixingTrainer(cfg)
            trainer.run()

    main()

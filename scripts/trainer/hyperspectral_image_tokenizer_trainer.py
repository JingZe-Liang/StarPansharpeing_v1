import os
import random
import sys
import time
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Callable, Literal, Sequence, cast, no_type_check

import accelerate
import accelerate.utils
import hydra
import numpy as np
import PIL.Image as Image
import torch
import torch._functorch.config
import torch.nn as nn
from accelerate import Accelerator
from accelerate.state import PartialState
from accelerate.tracking import TensorBoardTracker
from accelerate.utils.deepspeed import DummyOptim, DummyScheduler
from ema_pytorch import EMA
from fvcore.nn import parameter_count_table
from kornia.utils.image import make_grid, tensor_to_image
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp._fully_shard import FSDPModule
from torch.distributed.tensor import DTensor
from torchmetrics.aggregation import MeanMetric
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from src.data.hyperspectral_loader import (
    get_hyperspectral_img_loaders_with_different_backends,
)
from src.stage1.cosmos.inference.utils import load_jit_model_shape_matched
from src.stage1.utilities.losses.gan_loss import VQLPIPSWithDiscriminator
from src.stage1.utilities.train.network import (
    get_model_learnable_params,
    get_parameters_encoder_frozen,
)
from src.utilities.config_utils import to_object as to_cont
from src.utilities.logging import log_print
from src.utilities.network_utils import load_fsdp_model, safe_dtensor_operation
from src.utilities.network_utils.network_loading import load_weights_with_shape_check
from src.utilities.train_utils.state import StepsCounter


class CosmosHyperspectralTokenizerTrainer:
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
        _fsdp_plugin: accelerate.utils.FullyShardedDataParallelPlugin | None = getattr(
            self.accelerator.state, "fsdp_plugin", None
        )

        if _dpsp_plugin is not None:
            self.accelerator.deepspeed_plugin.deepspeed_config[  # type: ignore
                "train_micro_batch_size_per_gpu"
            ] = self.dataset_cfg.batch_size_train

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
        used_dataset = self.dataset_cfg.used
        self.log_msg(f"[Data]: using dataset {used_dataset}")

        if hasattr(self.dataset_cfg, "train_loader") and hasattr(
            self.dataset_cfg, "val_loader"
        ):
            self.log_msg("[Data]: init dataloaders by hydra instantiate")
            self.train_dataset, self.train_dataloader = hydra.utils.instantiate(
                self.dataset_cfg.train_loader
            )
            self.val_dataset, self.val_dataloader = hydra.utils.instantiate(
                self.dataset_cfg.val_loader
            )
        else:
            self.log_msg("[Data]: init dataloaders manually")
            self.train_dataset, self.train_dataloader = (
                get_hyperspectral_img_loaders_with_different_backends(
                    paths=self.dataset_cfg.wds_path_train,
                    batch_size=self.dataset_cfg.batch_size_train,
                    num_workers=self.dataset_cfg.num_workers,
                    shuffle_size=self.dataset_cfg.shuffle_size,
                    hyper_transforms_lst=self.dataset_cfg.hyper_transforms_lst,
                    transform_prob=self.dataset_cfg.transform_prob,
                    random_apply=to_cont(self.dataset_cfg.random_apply),
                    prefetch_factor=self.dataset_cfg.prefetch_factor,
                    to_neg_1_1=True,
                    loader_type=self.dataset_cfg.loader_type,
                    channels=self.dataset_cfg.channels,
                    check_channels=True,
                    shuffle_within_workers=self.dataset_cfg.shuffle_within_workers,
                )
            )
            self.val_dataset, self.val_dataloader = (
                get_hyperspectral_img_loaders_with_different_backends(
                    paths=self.dataset_cfg.wds_path_val,
                    batch_size=self.dataset_cfg.batch_size_val,
                    num_workers=self.dataset_cfg.num_workers,
                    shuffle_size=self.dataset_cfg.shuffle_size,
                    prefetch_factor=self.dataset_cfg.prefetch_factor,
                    hyper_transforms_lst=None,
                    transform_prob=0.0,
                    to_neg_1_1=True,
                    channels=self.dataset_cfg.channels,
                    check_channels=True,
                    shuffle_within_workers=self.dataset_cfg.shuffle_within_workers,
                )
            )

        # setup the tokenizer
        self.setup_tokenizer()

        # pretrained tokenizer or peft tuning
        self._is_peft_tuning = False
        self.tokenizer_peft_wrapped = None
        if self.train_cfg.finetune_strategy == "peft":
            self.log_msg("[PEFT]: using peft tuning, wrapping the tokenizer")
            self._is_peft_tuning = True
            self._wrap_peft_tokenizer()

        # GAN, perceptual losses
        self.vq_loss_fn: VQLPIPSWithDiscriminator = hydra.utils.instantiate(cfg.vq_loss)
        self.vq_loss_fn.discriminator = self.vq_loss_fn.discriminator.to(self.dtype)

        # Augmentation pipelines and anti-degradation network / losses
        self.setup_aug_pipe_and_anti_degradation_network()

        # optimizers and lr schedulers
        self.tokenizer_optim, self.tokenizer_sched, self.disc_optim, self.disc_sched = (
            self.get_optimizer_lr_scheduler()
        )
        # The last layer weight must require grad
        self._ensure_last_layer_requires_grad()

        # EMA models and accelerator prepare
        self.prepare_for_training()
        self.prepare_ema_models()

        # if compile the model
        if self.train_cfg.compile_model:
            self.tokenizer = torch.compile(self.tokenizer)
            self.log_msg(f"Compiled tokenizer {self.tokenizer.__class__.__name__}")
            self.vq_loss_fn.discriminator = torch.compile(self.vq_loss_fn.discriminator)
            self.log_msg(
                f"Compiled discriminator {self.vq_loss_fn.discriminator.__class__.__name__}"
            )

            # no donated buffers
            if self.cfg.vq_loss.gen_loss_weight is None:
                torch._functorch.config.donated_buffer = False
                self.log_msg(
                    "donated_buffer is set to False, since the model is compiled, and use adaptive gen/disc loss,"
                    "this will somehow affect the total running efficiency",
                    level="WARNING",
                )

        self.vq_loss_fn.discriminator = self.vq_loss_fn.discriminator.to(self.dtype)
        # self.set_fsdp_cpu_local_tensor_to_each_rank(self.tokenizer)
        # self.set_fsdp_cpu_local_tensor_to_each_rank(self.vq_loss_fn.discriminator)

        # traing state counter
        self.train_state = StepsCounter(["train"])

        # clear GPU memory
        torch.cuda.empty_cache()

    def _ensure_last_layer_requires_grad(self):
        # Call after the optimizer is initialized
        if self._is_peft_tuning:
            last_layer = self.tokenizer.decoder.decoder.conv_out
            for w in last_layer.parameters():
                w.requires_grad_(True)

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
            self.norm_z = self.cfg.quantizer.norm_z
            if not self.sep_enc_dec:
                assert not self.norm_z, (
                    "norm_z is not supported when sep_enc_dec is False"
                )
            if not self.use_quantizer:
                assert not self.norm_z, (
                    "norm_z can not be set when quantizer is not used"
                )
            if self.norm_z:
                self.log_msg(
                    "norm_z is set to True in the trainer, which is not recommanded",
                    level="WARNING",
                )

            self.log_msg(
                f"[Tokenizer Encoder]: tokenizer parameter table:\n{parameter_count_table(self.tokenizer_encoder)}"
            )
            self.log_msg(
                f"[Tokenizer Decoder]: tokenizer parameter table:\n{parameter_count_table(self.tokenizer_decoder)}"
            )
            if (
                self.use_quantizer
                and isinstance(self.quantizer, nn.Module)
                and len(list(self.quantizer.parameters())) > 0
            ):
                self.log_msg("[Quantizer]: quantizer has parameters")
                self.log_msg(
                    "[Quantizer]: quantizer parameter table:\n{}".format(
                        parameter_count_table(self.quantizer)
                    )
                )

        # < the encoder and decoder is one class or lora mixin
        else:
            self.log_msg(
                "[Tokenizer]: Use encoder, decoder, and quantizer in one class"
            )
            self.norm_z = False  # in the model, not in trainer

            # Init tokenizer model
            self.tokenizer: nn.Module = hydra.utils.instantiate(self.tokenizer_cfg)

            # quantizer in the tokenizer, not handled by this trainer
            self.use_quantizer = (
                getattr(self.tokenizer, "quantizer", None) is not None
            )  # vq, bsq, fsq, kl
            self.quantizer = None
            self.log_msg(
                f"[Tokenizer]: init tokenizer {self.tokenizer.__class__.__name__}"
            )
            if self.use_quantizer:
                self.log_msg(
                    f"[Tokenizer]: has quantizer {self.tokenizer.quantizer.__class__}"
                )

            # Gradient checkpointing
            if self.train_cfg.grad_checkpoint:
                self.tokenizer.set_grad_checkpointing()
                self.log_msg("Set tokenizer gradient checkpointing enabled")

            # the params
            self.log_msg(
                f"[Tokenizer]: tokenizer parameter table:\n{parameter_count_table(self.tokenizer)}"
            )

    def setup_aug_pipe_and_anti_degradation_network(self):
        self.use_training_aug = False
        self.aug_pipe = getattr(self.train_cfg, "aug_pipeline", None)
        self.antideg_net = getattr(self.train_cfg, "anti_degradation_network", None)
        self.aug_pipeline_train_obj = getattr(
            self.train_cfg, "aug_pipeline_train_obj", None
        )

        if self.aug_pipe is not None:
            self.log_msg("[Tokenizer]: using augmentation pipeline")
            assert self.aug_pipeline_train_obj in [
                "decoder_clean",
                "decoder_deg",
                "anti_deg_network",
            ], "Augmentation pipeline is specified but no train object is provided"

            self.aug_pipe = hydra.utils.instantiate(self.train_cfg.aug_pipeline)
            if (
                self.aug_pipeline_train_obj == "anti_deg_network"
                and self.antideg_net is not None
            ):
                self.antideg_net = hydra.utils.instantiate(self.antideg_net)
                self.antideg_net_optim = hydra.utils.instantiate(
                    self.train_cfg.antideg_net_optim
                )
                self.log_msg(
                    f"Using anti-degradation network: {self.antideg_net.__class__.__name__}"
                )
            self.use_training_aug = True

    def setup_invariant_pipeline(self):
        self.invariant_pipe = None

    def prepare_ema_models(self):
        if self.no_ema:
            return

        if self.sep_enc_dec:
            self.ema_encoder = EMA(
                self.tokenizer_encoder,
                beta=self.ema_cfg.beta,
                update_every=self.ema_cfg.update_every,
            ).to(self.device)
            self.ema_decoder = EMA(
                self.tokenizer_decoder,
                beta=self.ema_cfg.beta,
                update_every=self.ema_cfg.update_every,
            ).to(self.device)
        else:
            self.ema_tokenizer = EMA(
                self.tokenizer,
                beta=self.ema_cfg.beta,
                update_every=self.ema_cfg.update_every,
            ).to(self.device)

        self.ema_vq_disc = EMA(
            self.vq_loss_fn.discriminator,
            beta=self.ema_cfg.beta,
            update_every=self.ema_cfg.update_every,
        ).to(self.device)

    def configure_logger(self):
        self.logger = logger

        log_file = Path(self.train_cfg.proj_dir)
        if self.train_cfg.log.log_with_time:
            str_time = time.strftime("%Y-%m-%d_%H-%M-%S")
            log_file = log_file / str_time
        if self.train_cfg.log.run_comment is not None:
            log_file = Path(log_file.as_posix() + "_" + self.train_cfg.log.run_comment)
        log_file = log_file / "log.log"
        # log_file.parent.mkdir(parents=True, exist_ok=True)

        # when distributed, there should be the same log_file
        if self.accelerator.use_distributed:
            if self.accelerator.is_main_process:
                input_lst = [log_file] * self.accelerator.num_processes
            else:
                input_lst = [None] * self.accelerator.num_processes
            output_lst = [None]
            torch.distributed.scatter_object_list(output_lst, input_lst, src=0)
            output_lst: list[Path]
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
            colorize=bool(int(os.getenv("COLOR_LOG", "1"))),
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
            self.logger.info(f"[cfg]: configuration saved to {cfg_cp_path}")

        # accelerate project configuration
        self.proj_dir = log_dir
        self.accelerator.project_configuration.project_dir = str(self.proj_dir)

        # tensorboard logger
        if not self.train_cfg.debug:
            tenb_dir = log_dir / "tensorboard"
            self.accelerator.project_configuration.logging_dir = str(tenb_dir)
            if self.accelerator.is_main_process:
                self.logger.info(f"[Tensorboard]: tensorboard saved to {tenb_dir}")
                self.accelerator.init_trackers("train")
                self.tb_logger: TensorBoardTracker = self.accelerator.get_tracker(
                    "tensorboard"
                )  # type: ignore

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
            model: torch.nn.Module = logs.pop("model")
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
                        _grad = p.grad._local_tensor
                        if p.grad._local_tensor.device == torch.device("cpu"):
                            self.log_msg(
                                "p.grad is on cpu, this should not happen",
                                level="WARNING",
                            )
                            # ensure the corss rank does not involve cpu bankend
                            _grad = _grad.cuda()
                        # _p_grad = p.grad.full_tensor()  # across all ranks
                        _grad = safe_dtensor_operation(_grad)
                    _grad_norm = (_grad.data**2).sum() ** 0.5
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

    def _wrap_peft_tokenizer(self):
        assert "peft" in self.cfg, "peft_cfg not in the config"
        assert not self.sep_enc_dec, "peft_cfg not supported for sep enc dec"
        assert (
            self.accelerator.distributed_type
            != accelerate.utils.DistributedType.DEEPSPEED
        ), "Deepspeed PEFT tuning supports not implemented yet"

        peft_cfg: LoraConfig = hydra.utils.instantiate(self.cfg.peft)

        # LoRA configured peft modules and additional modules
        peft_cfg.target_modules = (
            list(peft_cfg.target_modules) if peft_cfg.target_modules is not None else []
        )
        peft_cfg.modules_to_save = (
            list(peft_cfg.modules_to_save)
            if peft_cfg.modules_to_save is not None
            else []
        )

        # Add conv in/conv out modules, and additional lora target modules
        if hasattr(self.tokenizer, "peft_fully_finetune_modules"):
            peft_cfg.modules_to_save += self.tokenizer.peft_fully_finetune_modules(
                add_norms=self.train_cfg.add_norms,
                conv_stem_reinit=self.train_cfg.conv_in_out_reinit,
            )  # type: ignore
            self.log_msg(
                "[PEFT]: use tokenizer defined input and output convs for tuning on different input/output channels"
                "when dealing with different hyperspectral dataset"
            )
        if hasattr(self.tokenizer, "peft_lora_modules"):
            tgt_lora_modules = self.tokenizer.peft_lora_modules(
                conv_stem_reinit=self.train_cfg.conv_in_out_reinit,
                conv_stem_chan=self.dataset_cfg.dataset_channel,
            )  # type: ignore
            peft_cfg.target_modules += list(tgt_lora_modules)

        self.log_msg(
            f"[PEFT]: use tokenizer defined lora target modules: {peft_cfg.target_modules} for tuning"
        )
        if not peft_cfg.modules_to_save:
            self.log_msg(
                f"[PEFT]: fully finetuning modules (except lora layers) are {peft_cfg.modules_to_save}"
            )

        self.log_msg(f"[PEFT]: peft_cfg is {peft_cfg}, wrapping the tokenizer... \n\n")
        adapter_name = getattr(self.dataset_cfg, "used", "default")
        self.tokenizer_peft_wrapped = get_peft_model(
            self.tokenizer,
            peft_config=peft_cfg,
            adapter_name=adapter_name,  # dataset name ?
            low_cpu_mem_usage=False,  # do not use meta device to load, since the tokenizer is not huge.
        )
        self.log_msg(self.tokenizer_peft_wrapped.print_trainable_parameters())

        # base model to train
        self.tokenizer = self.tokenizer_peft_wrapped.get_base_model()  # type: ignore

    def _get_tokenizer_params(self, for_optimizer=False, with_name: bool = False):
        # key to params
        def get_tokenizer_params_from_keys(keys: list[str]):
            return [
                p
                for name, p in self.tokenizer_encoder.named_parameters()
                if name in keys
            ]

        def get_tokenizer_params_not_from_keys(keys: list[str]):
            return [
                p
                for name, p in self.tokenizer_encoder.named_parameters()
                if name not in keys
            ]

        # quantizer
        if self.use_quantizer and self.sep_enc_dec:
            quant_params = list(self.quantizer.parameters())
            self.log_msg("[Optim]: add quantizer params into optimizer")
            for quant_p in quant_params:
                self.log_msg(f"[Optim]: quantizer param - {quant_p.shape}")
        else:
            quant_params = []

        if not for_optimizer:
            if self.sep_enc_dec:
                assert not with_name, "with_name is not supported for sep enc dec"

                not_pretrained_keys = [
                    *self._enc_model_mody_keys["not_pretrained_keys"],
                    *self._dec_model_mody_keys["not_pretrained_keys"],
                ]

                # * finetune all, different layers, first conv ===================
                if self.train_cfg.finetune_strategy == "finetune_all":
                    params = [
                        *list(self.tokenizer_encoder.parameters()),
                        *list(self.tokenizer_decoder.parameters()),
                    ]
                elif self.train_cfg.finetune_strategy == "hier_finetune":
                    params = [
                        {
                            "lr": self.train_cfg.tokenizer_optimizer.hier_base_lr,
                            "params": get_tokenizer_params_from_keys(
                                not_pretrained_keys
                            ),
                            "weight_decay": self.train_cfg.tokenizer_optimizer.weight_decay,
                        },
                        {
                            "lr": self.train_cfg.tokenizer_optimizer.hier_small_lr,
                            "params": get_tokenizer_params_not_from_keys(
                                not_pretrained_keys
                            ),
                            "weight_decay": self.train_cfg.tokenizer_optimizer.weight_decay,
                        },
                    ]
                elif self.train_cfg.finetune_strategy == "finetune_first_conv":
                    # performs poor, and still consume GPU mem.
                    params = get_tokenizer_params_from_keys(not_pretrained_keys)
                    # gradient not required
                    for p in self.tokenizer.parameters():
                        if p not in not_pretrained_keys:
                            p.requires_grad = False

                # * finetune decoder output head, or encoder output head + decoder input head =======
                # * from DCAE training phases 2 and 3
                elif self.train_cfg.finetune_strategy == "dcae_refine_decoder_head":
                    # decoder head is only refined for low-resolution images
                    # use l1, perceptual, gan losses

                    n_layer_ft = (
                        self.train_cfg.finetune_cfg.refine_decoder_head_n_layers
                    )
                    if isinstance(n_layer_ft, str):
                        assert n_layer_ft in ["all"], 'n_layer_ft must be "all"'
                        self.log_msg(
                            f"[Finetune Strategy]: {self.train_cfg.finetune_strategy}, "
                            "refine the whole decoder"
                        )
                    else:
                        assert n_layer_ft >= 0, (
                            "n_layer_ft must be equal or bigger than 0"
                        )
                        self.log_msg(
                            f"[Finetune Strategy]: {self.train_cfg.finetune_strategy}, "
                            "use DCAE refine decoder head (phase 3) stragety for low-resolution images. "
                            f"finetune the decoder last {n_layer_ft} layers and last output convs"
                        )

                    # fix the codebook if any is learnable
                    quant_params = []
                    if self.use_quantizer and isinstance(self.quantizer, nn.Module):
                        self.quantizer.eval()

                    # get decoder head params
                    params = []
                    match self.train_cfg.tokenizer_name:
                        case "cosmos_sep":
                            # let encoder fixed
                            self.tokenizer_encoder.eval()

                            if n_layer_ft == "all":
                                return self.tokenizer_decoder.parameters()

                            # add the conv_out layer and unpatcher layer
                            params.extend(
                                list(self.tokenizer_decoder[1].norm_out.parameters())
                                + list(self.tokenizer_decoder[1].unpatcher.parameters())
                                + list(self.tokenizer_decoder[1].conv_out.parameters())
                            )

                            # last n_layers decoder layers
                            if n_layer_ft > 0:
                                all_layers = list(self.tokenizer_decoder[1].up)
                                selected_layers = all_layers[-n_layer_ft:]
                                params.extend(
                                    [
                                        p
                                        for layer in selected_layers
                                        for p in layer.parameters()
                                    ]
                                )
                        case "cosmos_uni":
                            self.tokenizer.encoder.eval()

                            if n_layer_ft == "all":
                                return self.tokenizer.decoder.parameters()

                            params.extend(
                                list(self.tokenizer.decoder.norm_out.parameters())
                                + list(self.tokenizer.decoder.unpatcher.parameters())
                                + list(self.tokenizer.decoder.conv_out.parameters())
                            )

                            # last n_layers decoder layers
                            if n_layer_ft > 0:
                                all_layers = list(self.tokenizer.decoder.up)
                                selected_layers = all_layers[-n_layer_ft:]
                                params.extend(
                                    [
                                        p
                                        for layer in selected_layers
                                        for p in layer.parameters()
                                    ]
                                )
                        case "dcae":
                            self.tokenizer.encoder.eval()

                            if n_layer_ft == "all":
                                return self.tokenizer.decoder.parameters()

                            params.extend(
                                # proj out layer
                                list(self.tokenizer.decoder.project_out)
                            )

                            if n_layer_ft > 0:
                                stages = self.tokenizer.decoder.stages
                                selected_layers = stages[-n_layer_ft:]
                                for layer in selected_layers:
                                    params.extend(list(layer.parameters()))
                        case _:
                            raise ValueError(
                                f"Unknown tokenizer name: {self.train_cfg.tokenizer_name}"
                            )

                elif self.train_cfg.finetune_strategy == "dcae_adapt_latent":
                    # adapt latents
                    # by finetuning encoder head and decoder for high-resolution images
                    # use l1, perceptual losses (w/o gan loss)

                    raise NotImplementedError("not implemented yet")

                else:
                    raise ValueError(
                        f"Unknown finetune strategy: {self.train_cfg.finetune_strategy}"
                    )

                self.log_msg(
                    f"[Optimizer]: finetune strategy: {self.train_cfg.finetune_strategy}"
                )
            else:
                if self.train_cfg.finetune_strategy == "decoder_only":
                    self.log_msg("Only decoder parameters are tunable.")
                    params = get_parameters_encoder_frozen(
                        model=self.tokenizer,
                        with_name=with_name,
                    )
                # peft will handle the learnable parameters
                elif self.train_cfg.finetune_strategy in ("finetune_all", "peft"):
                    self.log_msg("All parameters are trainable")
                    params = get_model_learnable_params(
                        self.tokenizer, with_name=with_name
                    )
                else:
                    raise ValueError(
                        f"Unknown training/finetuning strategy {self.train_cfg.finetune_strategy}"
                    )

            # add with quantizer params
            if self.sep_enc_dec:
                assert isinstance(params, list)
                params += quant_params
        else:
            raise NotImplementedError("not implemented")

        return params

    def _get_disc_params(self, for_optimizer=False, with_name: bool = False):
        if not for_optimizer:
            if with_name:
                return dict(self.vq_loss_fn.discriminator.named_parameters())
            else:
                return list(self.vq_loss_fn.discriminator.parameters())
        else:
            return self.vq_loss_fn.discriminator.state_dict()

    def get_optimizer_lr_scheduler(self):
        # optimizers
        if (
            self.accelerator.state.deepspeed_plugin is None
            or "optimizer"
            not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):

            def _optimizer_creater(optimizer_cfg, params_getter: Callable):
                if "muon" in optimizer_cfg._target_:
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

            tokenizer_optim = _optimizer_creater(
                self.train_cfg.tokenizer_optimizer, self._get_tokenizer_params
            )
            disc_optim = _optimizer_creater(
                self.train_cfg.disc_optimizer, self._get_disc_params
            )
        else:
            tokenizer_optim = DummyOptim([{"params": self._get_tokenizer_params()}])
            disc_optim = DummyOptim([{"params": self._get_disc_params()}])

        # schedulers
        if (
            self.accelerator.state.deepspeed_plugin is None
            or "scheduler"
            not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            tokenizer_sched = hydra.utils.instantiate(self.train_cfg.tokenizer_sched)(
                optimizer=tokenizer_optim
            )
            disc_sched = hydra.utils.instantiate(self.train_cfg.disc_sched)(
                optimizer=disc_optim
            )
        else:
            tokenizer_sched = DummyScheduler(tokenizer_optim)
            disc_sched = DummyScheduler(disc_optim)

        # set the heavyball optimizer without torch compiling
        is_heavyball_opt = lambda opt: opt.__class__.__module__.startswith("heavyball")
        if is_heavyball_opt(tokenizer_optim) or is_heavyball_opt(disc_optim):
            self.log_msg(
                "use heavyball optimizer, it will compile the optimizer, "
                "for efficience testing the scripts, disable the compilation.",
                level="WARNING",
            )

        return tokenizer_optim, tokenizer_sched, disc_optim, disc_sched

    def set_fsdp_cpu_local_tensor_to_each_rank(self, model: nn.Module | FSDPModule):
        fsdp_plugin = self.accelerator.state.fsdp_plugin
        cpu_offload_enabled = (
            hasattr(fsdp_plugin, "cpu_offload")
            and fsdp_plugin.cpu_offload is not None
            and fsdp_plugin.cpu_offload.offload_params
        )
        # cpu_offload_enabled = False
        if not self._is_fsdp and cpu_offload_enabled:
            return model

        self.log_msg(
            "FSDP module seems do not move the original parameter (_local_tensor) on the"
            "correct rank, we need to manully move them on cuda while using `to_local` or `redistributed` methods",
            level="WARNING",
            warn_once=True,
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
            self.vq_loss_fn.discriminator = nn.SyncBatchNorm.convert_sync_batchnorm(
                self.vq_loss_fn.discriminator
            )
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
            self.vq_loss_fn.discriminator.dtype = self.dtype

        # tokenizer
        if self.sep_enc_dec:
            # FIXME: FSDP2 missing mapping for a parameter in the optmizer
            self.tokenizer_encoder, self.tokenizer_optim = self.accelerator.prepare(
                self.tokenizer_encoder, self.tokenizer_optim
            )
            self.tokenizer_decoder, self.tokenizer_optim = self.accelerator.prepare(
                self.tokenizer_decoder, self.tokenizer_optim
            )
            # self.tokenizer_encoder = self.set_fsdp_cpu_local_tensor_to_each_rank(
            #     self.tokenizer_encoder
            # )
            # self.tokenizer_decoder = self.set_fsdp_cpu_local_tensor_to_each_rank(
            #     self.tokenizer_decoder
            # )
        else:
            self.tokenizer, self.tokenizer_optim = self.accelerator.prepare(
                self.tokenizer, self.tokenizer_optim
            )
            # self.tokenizer = self.set_fsdp_cpu_local_tensor_to_each_rank(self.tokenizer)

        # quantizer
        if self.quantizer is not None:
            self.quantizer = self.accelerator.prepare(self.quantizer)
            # self.quantizer = self.set_fsdp_cpu_local_tensor_to_each_rank(self.quantizer)

        # discriminator
        (self.vq_loss_fn.discriminator, self.disc_optim) = self.accelerator.prepare(
            self.vq_loss_fn.discriminator, self.disc_optim
        )
        # self.vq_loss_fn.discriminator = self.set_fsdp_cpu_local_tensor_to_each_rank(
        #     self.vq_loss_fn.discriminator
        # )

        # augmentation network
        if self.use_training_aug and self.antideg_net is not None:
            self.antideg_net.dtype = self.dtype
            self.accelerator.prepare(self.antideg_net, self.antideg_net_optim)

        # dataloaders
        # NOTE: if is litdata loader, it will automatically handle the world_size dispatch and sampler
        # do not prepare loaders
        # self.train_dataloader, self.val_dataloader = self.accelerator.prepare(
        #     self.train_dataloader, self.val_dataloader
        # )
        (self.tokenizer_sched, self.disc_sched) = self.accelerator.prepare(
            self.tokenizer_sched, self.disc_sched
        )

        def _fake_prepare(model, no_split_modules, dtype=torch.float32):
            model._no_split_modules = no_split_modules
            model.dtype = dtype
            # dummy optimizer for accelerate, will not train dino encoder
            # since accelerate need to prepare model and optimizer for FSDP2 at the same time
            model_prepared, _ = self.accelerator.prepare(
                model, torch.optim.AdamW(model.parameters())
            )
            for p in model.parameters():
                if isinstance(p, DTensor):
                    p._local_tensor = p._local_tensor.to(self.device)
            # pop out the repa encoder
            self.accelerator._models.pop(-1)
            self.accelerator._optimizers.pop(-1)
            return model_prepared

        if self.accelerator.is_fsdp2:
            # repa loss
            if self.vq_loss_fn.use_repa:
                self.log_msg("prepare repa encoder for FSDP2", level="WARNING")
                self.vq_loss_fn.repa_loss.repa_encoder = _fake_prepare(
                    self.vq_loss_fn.repa_loss.repa_encoder, ["NestedTensorBlock"]
                )

            # vf loss
            if self.vq_loss_fn.use_vf:
                self.log_msg("prepare vf encoder for FSDP2", level="WARNING")
                self.vq_loss_fn.vf_loss.repa_encoder = _fake_prepare(
                    self.vq_loss_fn.vf_loss.repa_encoder, ["NestedTensorBlock"]
                )

            # gram loss
            if self.vq_loss_fn.use_gram:
                self.log_msg("prepare gram encoder for FSDP2", level="WARNING")
                self.vq_loss_fn.gram_loss.repa_encoder = _fake_prepare(
                    self.vq_loss_fn.gram_loss.repa_encoder, ["NestedTensorBlock"]
                )

    def step_train_state(self):
        self.train_state.update("train")

    def ema_update(self, mode="tokenizer"):
        if self.no_ema:
            # not support ema when is deepspeed zero2 or zero3
            return

        if mode == "tokenizer":
            if self.sep_enc_dec:
                self.ema_encoder.update()
                self.ema_decoder.update()
            else:
                self.ema_tokenizer.update()

        elif mode == "disc":
            self.ema_vq_disc.update()
        else:
            raise ValueError(f"Unknown mode {mode}")

    def forward_tokenizer(self, x, ema: bool = False, is_testing: bool = False) -> dict:
        out_d = {}
        with self.accelerator.autocast():
            if self.sep_enc_dec:
                if not ema:
                    if is_testing:
                        self.tokenizer_encoder.eval()
                        self.tokenizer_decoder.eval()
                    to_enc = lambda x: self.tokenizer_encoder(x)[0]
                    to_dec = lambda x: self.tokenizer_decoder(x)
                else:
                    if is_testing:
                        self.ema_encoder.eval()
                        self.ema_decoder.eval()
                    to_enc = lambda x: self.ema_encoder(x)[0]
                    to_dec = lambda x: self.ema_decoder(x)

                latent = to_enc(x)

                # TODO: move the total seperated encoder and decoder into a unified class
                if self.norm_z:  # only norm for seperated encoder and decoder
                    latent = torch.nn.functional.normalize(latent, dim=1)

                if self.quantizer is not None:
                    latent_q, q_loss, q_info = self.quantizer(latent)
                else:
                    latent_q = latent
                recon = to_dec(latent_q)
            else:
                if is_testing:
                    self.tokenizer.eval()
                latent = None
                # TODO: add ema tokenizer

                if self.use_quantizer:
                    recon, q_loss, q_info = self.tokenizer(x)
                else:
                    recon = self.tokenizer(x)

        # basic out
        out_d.update({"latent": latent, "recon": recon})

        # quantizer output dict
        if self.use_quantizer:
            _q_dict = dict(q_loss=q_loss, q_info=q_info, latent_q=latent)
        else:
            _q_dict = dict(q_loss=None, q_info=None, latent_q=None)
        out_d.update(_q_dict)

        # repa or vf feature
        _unwrap_tok = self.accelerator.unwrap_model(self.tokenizer)
        if (
            hasattr(_unwrap_tok, "get_repa_feature")
            and getattr(_unwrap_tok, "_use_repa_loss", False)
            and _unwrap_tok.training
        ):
            repa_feature = _unwrap_tok.get_repa_feature()  # type: ignore
            assert repa_feature is not None, "repa_feature is None"
            if torch.is_tensor(repa_feature):
                out_d["repa_feature"] = repa_feature
            elif isinstance(repa_feature, (tuple, list)):
                assert len(repa_feature) == 2, (
                    "only support two repa features (low level and semantic features distillation)."
                )
                out_d["repa_feature"] = repa_feature[0]
                out_d["semantic_feature"] = repa_feature[1]
            else:
                raise ValueError(
                    f"Unknown repa_feature type {type(repa_feature)}, only support tensor, tuple or list."
                )
        elif hasattr(_unwrap_tok, "get_vf_feature") and getattr(
            _unwrap_tok, "_use_vf_loss", False
        ):
            vf_feature = _unwrap_tok.get_vf_feature()  # type: ignore
            assert vf_feature is not None, "vf_feature is None"
            out_d["vf_feature"] = vf_feature

        return out_d

    def forward_discriminator(
        self,
        x,
        out_d: dict,
        train_tokenizer: bool = True,
        split: str = "train",
        ema: bool = False,
    ):
        if ema:
            _non_ema_disc = self.vq_loss_fn.discriminator
            self.vq_loss_fn.discriminator = self.ema_vq_disc.ema_model

        optim_idx = 0 if train_tokenizer else 1  # tokenizer -> 0, discriminator -> 1

        repa_low_lvl_feat = semantic_feat = None
        # repa or vf feature (low-level feature alignment)
        if (repa_low_lvl_feat := out_d.get("repa_feature", None)) is None:
            repa_low_lvl_feat = out_d.get("vf_feature", None)
        # semantic feature (high-level feature alignment)
        if "semantic_feature" in out_d:
            semantic_feat = out_d.get("semantic_feature", None)

        # loss
        with self.accelerator.autocast():
            # self.vq_loss_fn.forward
            disc_train_loss_d, log_disc = self.vq_loss_fn(
                inputs=x,
                reconstructions=out_d["recon"],
                q_loss_total=out_d.get("q_loss", None),
                q_loss_breakdown=out_d.get("q_info", None),
                tokenizer_feat=repa_low_lvl_feat,
                tokenizer_feat2=semantic_feat,
                last_layer=self.get_last_layer(mode="dec"),
                enc_last_layer=self.get_last_layer(mode="enc"),
                global_step=self.global_step,
                optimizer_idx=optim_idx,
                add_prefix=False,
                split=split,
            )
            if train_tokenizer:
                loss = disc_train_loss_d["gen_loss"] + disc_train_loss_d["q_loss"]
            else:
                loss = disc_train_loss_d["disc_loss"]

        # back
        if ema:
            self.vq_loss_fn.discriminator = _non_ema_disc

        return loss, log_disc

    def get_global_step(self, mode="train"):
        # TODO: add val state
        assert mode in ("train",), "Only train mode is supported for now."

        return self.train_state[mode]

    @property
    def global_step(self):
        return self.get_global_step("train")

    def may_freeze(self, model, freeze=True):
        # for p in model.parameters():
        #     p.requires_grad = not freeze
        model.requires_grad_(not freeze)

    @no_type_check
    def get_last_layer(self, use_ema: bool = False, mode="dec"):
        if mode == "dec":
            if self.sep_enc_dec:
                w = self.accelerator.unwrap_model(
                    self.tokenizer_decoder
                ).decoder.get_last_layer()
            else:
                w = self.accelerator.unwrap_model(self.tokenizer).get_last_layer()
        else:  # encoder last conv out weight
            if not self.vq_loss_fn.use_gram and not self.vq_loss_fn.use_vf:
                return None
            if self.sep_enc_dec:
                w = self.accelerator.unwrap_model(
                    self.tokenizer_encoder
                ).encoder.conv_out.weight
            else:
                w = self.accelerator.unwrap_model(self.tokenizer).get_last_enc_layer()

        w = safe_dtensor_operation(w, prefer_full=True)
        if not w.requires_grad:
            # assert self._is_peft_tuning, f'the last layer weight may not requires_grad only if seted by PEFT'
            raise ValueError("The last layer weight must be enabled requires_grad")

        return w

    def gradient_check(self, model: nn.Module):
        # check nan gradient
        if self.accelerator.sync_gradients and getattr(
            self.train_cfg, "grad_check", False
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

    def train_tokenizer_step(self, x: torch.Tensor, tok_dict: dict):
        # freeze discriminator
        self.may_freeze(self.vq_loss_fn.discriminator, True)

        # quantizer loss sent to discriminator
        gen_loss, log_losses = self.forward_discriminator(
            x,
            tok_dict,
            train_tokenizer=True,
            split="train",
        )

        if self.use_training_aug and self.aug_pipeline_train_obj == "anti_deg_network":
            assert "aug_x" in tok_dict, "aug_x not in tokenizer output dict"
            assert self.antideg_net is not None, "antideg_net not defined"
            deg_x = tok_dict["aug_x"]
            gt = x
            recovery = self.antideg_net(deg_x)
            # Additional recovery loss ensuring the latent space suitable for restoration
            recovery_loss = (
                torch.nn.functional.mse_loss(recovery, gt)
                * self.train_cfg.antideg_loss_weight
            )
            gen_loss = gen_loss + recovery_loss
            log_losses["recovery_loss"] = recovery_loss.item()

        if self.accelerator.sync_gradients:
            # backward
            self.tokenizer_optim.zero_grad()
            self.accelerator.backward(gen_loss)
            if self.sep_enc_dec:
                self.gradient_check(self.tokenizer_encoder)
                self.gradient_check(self.tokenizer_decoder)
            else:
                self.gradient_check(self.tokenizer)
            self.tokenizer_optim.step()
            self.tokenizer_sched.step()

            if self.antideg_net is not None:
                self.antideg_net_optim.step()
                # self.gradient_check(self.antideg_net)

            # ema update
            self.ema_update(mode="tokenizer")

        return gen_loss, log_losses

    def train_disc_step(self, x: torch.Tensor, tokenizer_out: dict):
        self.may_freeze(self.vq_loss_fn.discriminator, False)

        disc_loss, log_disc = self.forward_discriminator(
            x, tokenizer_out, train_tokenizer=False, split="train"
        )

        if self.accelerator.sync_gradients:
            # backward
            self.disc_optim.zero_grad()
            self.accelerator.backward(disc_loss)
            self.gradient_check(self.vq_loss_fn.discriminator)
            self.disc_optim.step()
            self.disc_sched.step()

            # ema update
            self.ema_update(mode="disc")

        return disc_loss, log_disc

    def train_step(self, batch: dict):
        # torch.autograd.set_detect_anomaly(True)
        x = batch["img"].to(self.device, self.dtype)  # [-1, 1]

        quality_track_n = self.train_cfg.track_metrics_duration
        quality_track_after = self.train_cfg.track_metrics_after
        check_quality = None
        if quality_track_n >= 0:
            self._psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(
                self.device, self.dtype
            )
            self._ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(
                self.device, self.dtype
            )

            def check_quality(x, recon):
                x_q = self.to_rgb(x)
                recon_q = self.to_rgb(recon)
                self._psnr_fn.update(x_q, recon_q)
                self._ssim_fn.update(x_q, recon_q)

        _accum_models = (
            [self.tokenizer_encoder, self.tokenizer_decoder]
            if self.sep_enc_dec
            else [self.tokenizer]
        )
        _accum_models.append(self.vq_loss_fn.discriminator)

        with self.accelerator.accumulate(*_accum_models):
            # with torch.autograd.set_detect_anomaly(True):
            # self.log_msg(f"input shape: {x.shape}", level="debug")

            if self.use_training_aug:
                # NOTE: augmentation pipeline
                # is decoder_clean, x_clean -> aug -> x_deg
                # 1. -> encoder -> latent_deg -> decoder -> decoder_clean -> disc (tokenizer)
                #    -> x_clean -> disc (discriminator)
                # 2. -> encoder -> latent_deg -> decoder -> decoder_deg -> disc (tokenizer)
                #    -> x_deg -> disc (discriminator)
                # * version 2 is just the degrading the clean x into a degraded one, does not
                # * require the decoder to recover from the degraded latent <----- augmentation
                # ! version 1 requires the decoder to recover from the degraded latent <---- decoding the restoration
                # (may some generation ability).

                self.aug_pipe = cast(Callable, self.aug_pipe)
                x_deg = self.aug_pipe(x.float()).to(x.device, x.dtype)

                # forward the tokenizer using degraded images
                out_d = self.forward_tokenizer(x_deg)

                if self.aug_pipeline_train_obj == "decoder_clean":
                    # x: clean image
                    tokenizer_loss, log_token_loss = self.train_tokenizer_step(x, out_d)
                    # train discriminator on clean image
                    disc_loss, log_disc_loss = self.train_disc_step(x, out_d)

                elif self.aug_pipeline_train_obj == "decoder_deg":
                    tokenizer_loss, log_token_loss = self.train_tokenizer_step(
                        x_deg, out_d
                    )
                    # train discriminator on degraded image
                    disc_loss, log_disc_loss = self.train_disc_step(x_deg, out_d)

                else:
                    # The case for "anti_deg_network" is handled inside train_tokenizer_step
                    out_d["aug_x"] = x_deg
                    tokenizer_loss, log_token_loss = self.train_tokenizer_step(x, out_d)
                    disc_loss, log_disc_loss = self.train_disc_step(x, out_d)

            else:  # normal AE training pipeline
                out_d = self.forward_tokenizer(x)  # no augmentation
                # train tokenizer and discriminator
                tokenizer_loss, log_token_loss = self.train_tokenizer_step(x, out_d)
                disc_loss, log_disc_loss = self.train_disc_step(x, out_d)

            # track reconstruction quality
            if check_quality is not None:
                check_quality(x, out_d["recon"])

        self.step_train_state()

        # log losses
        if self.global_step % self.train_cfg.log.log_every == 0:
            _log_tok_losses = self.format_log(log_token_loss=log_token_loss)
            _log_disc_losses = self.format_log(log_disc_loss=log_disc_loss)

            self.log_msg(
                f"[Train State]: lr {self.tokenizer_optim.param_groups[0]['lr']:.4e} | "
                f"[Step]: {self.global_step}/{self.train_cfg.max_steps}"
            )
            self.log_msg(f"[Train Tok]: {_log_tok_losses}")
            self.log_msg(f"[Train Disc]: {_log_disc_losses}")

            # tensorboard log
            self.tenb_log_any("metric", log_token_loss, self.global_step)
            self.tenb_log_any("metric", log_disc_loss, self.global_step)

        if (
            quality_track_n >= 0
            and self.global_step % quality_track_n == 0
            and self.global_step >= quality_track_after
        ):
            self.log_msg(
                f"[Train Metrics]: PSNR: {self._psnr_fn.compute():.3f}, "
                f"SSIM: {self._ssim_fn.compute():.3f}"
            )

        if self.global_step % self.train_cfg.log.visualize_every == 0:
            self.visualize_reconstruction(
                x, out_d["recon"], add_step=True, img_name="recon/train_recon"
            )

    def format_log(
        self, log_token_loss: dict | None = None, log_disc_loss: dict | None = None
    ) -> str:
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
                                'logs has non-scalar tensor "{k}", skip it'.format(k=k),
                                warn_once=True,
                            )
                            continue
                        v = v.item()
                    strings.append(f"<cyan>{k}</>: {v:.{n_round}f}")  # colorize the key
                else:
                    strings.append(f"<cyan>{k}</>: {v}")
            return strings

        n_round = 4
        strings = []

        # * tokenzier losses
        if log_token_loss is not None:
            _selects = ["reconstruct_loss", "ssim_loss", "g_loss", "d_weight"]
            if self.vq_loss_fn.use_perceptual_loss:
                _selects.extend(["perceptual_loss", "gram_loss"])
            if self.vq_loss_fn.use_repa:
                _selects.extend(["repa_loss"])
            if self.vq_loss_fn.use_vf:
                _selects.extend(["vf_loss"])
            if self.vq_loss_fn.use_sem_distill:
                _selects.extend(["sem_dist_loss"])

            _log_token = dict_round_to_list_str(
                log_token_loss,
                n_round=n_round,
                select=_selects,
            )
            strings.extend(_log_token)

            if self.use_quantizer:
                _quant_logs_out_select = {
                    "bsq": [
                        "q_loss",
                        "entropy",
                        "avg_prob",
                        "commit_loss",
                    ],
                    "lfq": [
                        "q_loss",
                        "commit_loss",
                        "batch_entropy",
                        "per_sample_entropy",
                    ],
                    "vq_advance": [
                        "q_loss",
                        "commitment",
                        "code_diversity",
                        "orthogonal_reg",
                        "learn_code_opt_loss",
                    ],  # none is all to be selected
                    "kl": ["kl_loss"],
                    "fsq": ["fsq_loss"],  # is zero for FSQ
                    "psd": ["kl_loss"],
                }

                _log_q = dict_round_to_list_str(
                    log_token_loss,
                    n_round,
                    select=_quant_logs_out_select.get(self.quantizer_type or "", None),
                )
                strings.extend(_log_q)
        # * discriminator losses
        elif log_disc_loss is not None:
            _selects = ["disc_loss", "logits_real", "logits_fake", "lecam_loss"]
            if log_disc_loss.get("r1_scale", 0.0) != 0.0:
                _selects.append("r1_scale")
            if log_disc_loss.get("r2_scale", 0.0) != 0.0:
                _selects.append("r2_scale")

            _disc_reg_logs = dict_round_to_list_str(
                log_disc_loss,
                n_round,
                _selects,
            )
            strings.extend(_disc_reg_logs)

        else:
            raise ValueError("At least one of the logs should be provided")

        return " - ".join(strings)

    def _randomly_batch_sample_key(self, batch):
        if "img" not in batch:
            keys_not_dunder = [k for k in batch.keys() if not k.startswith("__")]
            # randomly choose one
            k = random.choice(keys_not_dunder)
            batch["img"] = batch[k]

        return batch

    def infinity_train_loader(self):
        while True:
            for batch in self.train_dataloader:
                batch = self._randomly_batch_sample_key(batch)
                if batch is None or batch.get("img", None) is None:
                    continue
                yield batch

    def train_loop(self):
        _stop_train_and_save = False
        self.accelerator.wait_for_everyone()

        self.log_msg("[Train]: start training", only_rank_zero=False)
        for batch in self.infinity_train_loader():
            # train step
            self.train_step(batch)

            if (
                self.global_step % self.val_cfg.val_duration == 0
            ):  # and self.accelerator.sync_gradients:
                self.log_msg("[Train]: start validation ...")
                self.val_loop()
                torch.cuda.empty_cache()

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
            batch = self._randomly_batch_sample_key(batch)
            if batch is None or batch.get("img", None) is None:
                continue
            yield batch

    def val_step(self, batch: dict) -> torch.Tensor:
        img = batch["img"].to(self.device, self.dtype)
        with torch.no_grad():
            recon = self.forward_tokenizer(img, ema=True, is_testing=True)["recon"]

        return recon

    def val_loop(self):
        if not hasattr(self, "_val_loader_iter"):
            # state in the loader generator
            self._val_loader_iter = iter(self.finite_val_loader())

        if hasattr(self.tokenizer_optim, "eval"):
            self.log_msg("set optimizer to eval mode (support for splus optimizer)")
            self.tokenizer_optim.eval()
            self.disc_optim.eval()

        # set all mode
        def _set_all_model_modes(train=False):
            if not train:  # eval
                if self.sep_enc_dec:
                    self.ema_encoder.eval()
                    self.ema_decoder.eval()
                    self.tokenizer_encoder.eval()
                    self.tokenizer_decoder.eval()
                else:
                    self.tokenizer.eval()

            else:  # train
                if self.sep_enc_dec:
                    self.tokenizer_encoder.train()
                    self.tokenizer_decoder.train()
                else:
                    self.tokenizer.train()

        # track psnr and ssim
        if self.train_cfg.track_metrics:
            psnr_fn = PeakSignalNoiseRatio(1.0).to(device=self.device, dtype=self.dtype)
            ssim_fn = StructuralSimilarityIndexMeasure().to(
                device=self.device, dtype=self.dtype
            )
        loss_metrics = MeanMetric().to(device=self.device)

        _set_all_model_modes(train=False)

        for val_iter in range(self.val_cfg.max_val_iters):
            batch = next(self._val_loader_iter)
            batch = self._randomly_batch_sample_key(batch)
            try:
                recon = self.val_step(batch)
            except Exception as e:
                # debug here
                self.log_msg(
                    f"Validation failed, batch keys are {batch.keys()}", level="debug"
                )
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        self.log_msg(
                            f"{k} shape: {v.shape}", only_rank_zero=False, level="debug"
                        )
                    else:
                        self.log_msg(f"{k}: {v}", only_rank_zero=False, level="debug")
                raise e

            recon_for_metrics = self.to_rgb(recon)
            batch_img_rgb = self.to_rgb(batch["img"].to(self.device))

            if self.train_cfg.track_metrics:
                psnr_fn.update(batch_img_rgb, recon_for_metrics)
                ssim_fn.update(batch_img_rgb, recon_for_metrics)

            # recon loss
            loss = nn.functional.l1_loss(recon, batch["img"].to(recon))
            loss_metrics.update(loss)

        if self.train_cfg.track_metrics:
            psnr_val = psnr_fn.compute()
            ssim_val = ssim_fn.compute()
        else:
            psnr_val = torch.tensor(0.0).to(self.device)
            ssim_val = torch.tensor(0.0).to(self.device)
        loss_val = loss_metrics.compute()

        # gather
        if self.accelerator.use_distributed:
            psnr_val = self.accelerator.gather(psnr_val).mean().item()
            ssim_val = self.accelerator.gather(ssim_val).mean().item()
            loss_val = self.accelerator.gather(loss_val).mean().item()

        if self.accelerator.is_main_process:
            self.log_msg(
                f"[Val]: PSNR: {psnr_val:.4f}, SSIM: {ssim_val:.4f} | loss: {loss_val:.4f}"
            )
            self.tenb_log_any(
                "metric",
                {"psnr": psnr_val, "ssim": ssim_val, "loss_val": loss_val},
                step=self.global_step,
            )

            # visualize the last val batch
            self.visualize_reconstruction(
                batch["img"], recon, add_step=True, img_name="val_sampled/sampled"
            )

        _set_all_model_modes(train=True)
        self.tokenizer_optim.zero_grad()
        self.disc_optim.zero_grad()

        if hasattr(self.tokenizer_optim, "train"):
            self.log_msg("set optimizer to train mode (support for splus optimizer)")
            self.tokenizer_optim.train()
            self.disc_optim.train()

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
                    (self.proj_dir / "peft_ckpt").as_posix(),
                    is_main_process=self.accelerator.is_main_process,
                )

            self.log_msg("[State]: save peft (only lora layers) model")

    def save_ema(self):
        if self.no_ema:
            self.log_msg("use deepspeed or FSDP, do have EMA model to save")
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

        self.log_msg(f"[ckpt]: save ema at {ema_path}")

    def load_from_ema_or_lora(self, ema_path: str | Path, strict: bool = False):
        ema_path = Path(ema_path)
        if self.sep_enc_dec:
            if self._is_fsdp:
                raise NotImplementedError(
                    "FSDP2 loading for separated encoder and decoder is not implemented yet"
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
                        # depend on if there is a full state dict
                        if not getattr(self.train_cfg, "load_weight_shard", True):
                            _not_shard_weights = "model" in _assume_path.as_posix()
                        else:
                            _not_shard_weights = False  # shard loaded
                    else:
                        _not_shard_weights = True

                    self.log_msg(
                        "loading peft checkpoint into model", only_rank_zero=False
                    )
                    if _not_shard_weights:  # is shard weights
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
                            loaded_res = set_peft_model_state_dict(
                                self.tokenizer_peft_wrapped,
                                accelerate.utils.load_state_dict(ema_path.as_posix()),
                                adapter_name=getattr(
                                    self.dataset_cfg, "used", "default"
                                ),
                                ignore_mismatched_sizes=not strict,
                                low_cpu_mem_usage=False,
                            )

                    else:  # not shard weights
                        peft_path = _assume_path / "pytorch_model_fsdp.bin"
                        assert peft_path.exists(), (
                            "peft checkpoint dir not found, use accelerate-merge-weights CLI first"
                        )

                        # ensure there is model/pytorch_model_fsdp.bin file
                        _fsdp_plugin_cp = deepcopy(self.accelerator.state.fsdp_plugin)
                        _fsdp_plugin_cp.state_dict_type = StateDictType.FULL_STATE_DICT
                        loaded_res = load_fsdp_model(
                            _fsdp_plugin_cp,
                            self.accelerator,
                            self.tokenizer_peft_wrapped,
                            ema_path.as_posix(),
                            model_index=0,
                            adapter_only=True,  # warn: loading from shard only-lora-layer weights does not work.
                        )

                    self.log_msg(
                        f"[Warning]: {loaded_res} keys are incompatible with the model",
                        level="warning",
                    )

                    # is refer
                    # NOTE: forcing to get the base model, not sure if this is referencing the same model
                    assert self.tokenizer_peft_wrapped is not None, (
                        "tokenizer_peft_wrapped is None"
                    )
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
                    raise RuntimeError("load FSDP or LoRA weights failed")
            else:
                # Load combined model
                self.log_msg("loading bin or safetensors checkpoint into model ...")
                try:
                    accelerate.utils.load_checkpoint_in_model(
                        self.accelerator.unwrap_model(self.tokenizer),
                        _assume_path,
                        strict=strict,
                    )
                except Exception as e:
                    self.log_msg(
                        f"loading bin or safetensors checkpoint into tokenizer failed: {e}, "
                        f"trying to load partial of the weights ..."
                    )
                    load_weights_with_shape_check(
                        self.accelerator.unwrap_model(self.tokenizer),
                        accelerate.utils.load_state_dict(
                            (ema_path / "tokenizer" / "model.safetensors").as_posix()
                        ),
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
            try:
                accelerate.utils.load_checkpoint_in_model(
                    self.accelerator.unwrap_model(self.vq_loss_fn.discriminator),
                    ema_path / "discriminator",
                    strict=strict,
                )
                # self.train_state.load_state_dict(torch.load(ema_path / "train_state.pth")
            except Exception as e:
                self.log_msg(
                    f"loading discriminator checkpoint into model failed: {e}, "
                    f"trying to load partial of the weights ..."
                )
                load_weights_with_shape_check(
                    self.accelerator.unwrap_model(self.vq_loss_fn.discriminator),
                    accelerate.utils.load_state_dict(
                        (ema_path / "discriminator" / "model.safetensors").as_posix()
                    ),
                )

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
            "[Load EMA]: clear the accelerator registrations and re-prepare training"
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
    ):
        x = self.to_rgb(x)
        if recon is not None:
            recon = self.to_rgb(recon)

        _only_n = only_vis_n or 16
        to_img = lambda x: tensor_to_image(make_grid(x[:_only_n], n_row=4, padding=2))
        c = x.shape[1]

        # * --- hyperspectral image to rgb images --- #
        def hyperspectral_to_rgb(x):
            # is rgb or gray images
            if c in (1, 3):
                x_np = to_img(x)
            elif isinstance(self.dataset_cfg.rgb_channels, Sequence):
                rgb_channels = to_cont(self.dataset_cfg.rgb_channels)
                x_np = to_img(x[:, rgb_channels])
            elif callable(self.dataset_cfg.rgb_channels):
                x_np = to_img(self.dataset_cfg.rgb_channels(x))
            else:
                raise ValueError(
                    f"Unknown rgb_channels {self.dataset_cfg.rgb_channels},"
                    f"typed {type(self.dataset_cfg.rgb_channels)}"
                )

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
            assert self.train_cfg.ema_load_path is not None, (
                f"ema_load_path must be specified when finetune_strategy is {self.train_cfg.finetune_strategy}"
            )

        if self.train_cfg.resume_path is not None:
            self.resume(self.train_cfg.resume_path)
        elif self.train_cfg.ema_load_path is not None:
            self.load_from_ema_or_lora(self.train_cfg.ema_load_path)

        # train !
        self.train_loop()


_key = "hybrid_cosmos_f16c64p1"
_configs_dict = {
    # use pretrained cosmos world tokenizer (continous image configuration)
    "cosmos_sep_f8c16p4": "cosmos_post_train_f8c16p4",
    "cosmos_sep_f8c32p1": "cosmos_post_train_f8c32p1",
    "cosmos_sep_f16c16p4": "cosmos_post_train_f16c16p4",
    # unify encoder and decoder int one model
    "unicosmos_f8c16p2": "unicosmos_tokenizer_f8c16p2",
    "unicosmos_f8c16p4": "unicosmos_tokenizer_f8c16p4",
    "unicosmos_large_f8c16p4": "unicosmos_tokenizer_large_f8c16p4",
    "unicosmos_f16c16p1": "unicosmos_tokenizer_f16c16p1",
    "unicosmos_f16c16p4": "unicosmos_tokenizer_f16c16p4",
    "unicosmos_f8c16p4_repa_kl": "unicosmos_tokenizer_kl_repa_f8c16p4",
    # psd kl vae
    "unicosmos_psd_f8c16p1": "unicosmos_tokenizer_psd_f8c16p1",
    # TODO: flow decoder or flow head decoder AE
    # hybrid ae
    "hybrid_cosmos_f16c32p1": "hybrid_cosmos_tokenizer_f16c32p1",
    "hybrid_cosmos_f16c64p1": "hybrid_cosmos_tokenizer_f16c64p1",
    # \sigma-vae decoder
    "unicosmos_gen_f8c16p1": "unicosmos_gen_tokenizer_f8c16p1",
    # bsq quantized
    "unicosmos_bsq_f8c36p4": "unicosmos_tokenizer_bsq_repa_f8c36p4",
    # sana CDAE
    "sana_f8c16p1_lita": "dcae_f8c16p1_attn",
    "sana_f8c16p1_conv": "dcae_f8c16p1_conv",
    "sana_f8c32p1_bsq": "dcae_f8c32p1_bsq",
    "sana_f16c16p1_lita": "dcae_f16c16p1_attn",
    "sana_f16c16p1_conv": "dcae_f16c16p1_conv",
    "sana_f32c32p1_pretrained": "cdae_f32c32p1_pretrained",
    # leanvae
    "lean_vae_f8c16p4": "lean_vae_f8c16p4",
    # ldm vae
    "ldm_vae_f8c16p1": "ldm_vae_f8c16p1",
    # lora finetuning
    "unicosmos_lora_f8c16p4": "unicosmos_lora_finetune_f8c16p4",
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
        print_colored_banner("HyperTokenizer")
    logger.info(
        "<green>\n"
        + "=" * 60
        + "\n"
        + f"[Config]: {chosen_cfg}\n"
        + "\n"
        + "Start Running , Good Luck!\n"
        + "=" * 60
        + "</>",
        not_rank0_print=True,
    )

    # Main function
    @hydra.main(
        config_path="../configs/tokenizer_gan",
        config_name=chosen_cfg,
        version_base=None,
    )
    def main(cfg):
        if cli_args.only_rank_zero_catch:
            catcher = (
                partial(logger.catch, reraise=True) if is_rank_zero else nullcontext
            )
        else:
            catcher = partial(logger.catch, reraise=True)

        with catcher():
            trainer = CosmosHyperspectralTokenizerTrainer(cfg)
            trainer.run()

    main()

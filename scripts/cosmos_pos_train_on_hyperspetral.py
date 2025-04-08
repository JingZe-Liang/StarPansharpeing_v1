import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Literal

import accelerate
import hydra
import numpy as np
import PIL.Image as Image
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.tracking import TensorBoardTracker
from accelerate.utils import DummyOptim, DummyScheduler
from ema_pytorch import EMA
from kornia.utils.image import make_grid, tensor_to_image
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torchmetrics.aggregation import MeanMetric
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

os.environ["HYDRA_FULL_ERROR"] = "1"
sys.path.insert(0, "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer")
sys.path.insert(0, "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/src")
import ast

from src.data.hyperspectral_loader import get_hyperspectral_dataloaders
from src.stage1.cosmos.cosmos_tokenizer import (
    ContinuousImageTokenizer as CosmosTokenizer,
)
from src.stage1.cosmos.inference.utils import load_jit_model_shape_matched
from src.stage1.cosmos.networks.configs import continuous_image
from src.stage1.sana_dcae.models.efficientvit.dc_ae import DCAE
from src.stage1.two_d_vit_tokenizer.tokenizer import VITBSQModel, VITVQModel
from src.stage1.utilities.losses.gan_loss import VQLPIPSWithDiscriminator
from src.utilities.train_utils.state import StepsCounter

to_cont = partial(OmegaConf.to_container, resolve=True)
# omegaconf resolver
OmegaConf.register_new_resolver("eval", lambda x: ast.literal_eval(x))
OmegaConf.register_new_resolver("function", lambda x: hydra.utils.get_method(x))


## TODO:
# 1. add diffusion, rectify flow, or inductive moumentumn matching to the decoder
# 2. add BSQ or LFQ to discretize
# 3. add rectify flow for continuous latents; maskbit or maskgit for discrete latents


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
        self.dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "no": torch.float32,
        }[self.accelerator.mixed_precision]
        self.log_msg("Log file is saved at: {}".format(log_file))
        self.log_msg("Weights will be saved at: {}".format(self.proj_dir))
        self.log_msg("Training is configured and ready to start.")

        # is zero 2 or 3, not EMA
        _dpsp_plugin = self.accelerator.state.deepspeed_plugin
        self.is_zero_2_3 = False
        if _dpsp_plugin is not None:
            self.log_msg("[Deepspeed]: using deepspeed plugin")
            self.is_zero_2_3 = _dpsp_plugin.deepspeed_config["zero_optimization"][
                "stage"
            ] in [2, 3]

        # pretrained tokenizer

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

        # setup tokenizer
        self.setup_tokenizer()

        # GAN loss
        self.vq_loss_fn: VQLPIPSWithDiscriminator = hydra.utils.instantiate(cfg.vq_loss)

        # optimizers and lr schedulers
        self.tokenizer_optim, self.tokenizer_sched, self.disc_optim, self.disc_sched = (
            self.get_optimizer_lr_scheduler()
        )
        # EMA models and accelerator prepare
        self.prepare_ema_models()
        self.prepare_for_training()

        # traing state counter
        self.train_state = StepsCounter(["train"])

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
            self.tokenizer_encoder.train()
            self.tokenizer_decoder.train()
            self.tokenizer_encoder: nn.Module
            self.tokenizer_decoder: nn.Module

            # quantizer
            if self.cfg.quantizer.quant is not None:
                self.quantizer = hydra.utils.instantiate(self.cfg.quantizer.quant).to(
                    self.device
                )
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
            self.norm_z = False  # in the class, not in trainer
            self.tokenizer = hydra.utils.instantiate(self.cfg.tokenizer)
            self.tokenizer: VITVQModel | VITBSQModel | CosmosTokenizer | DCAE
            # quantizer in the tokenizer, not handled by this trainer
            self.use_quantizer = hasattr(self, "quantizer")
            self.quantizer = None
            self.log_msg(
                f"[Tokenizer]: init tokenizer {self.tokenizer.__class__.__name__}"
            )
            if self.use_quantizer:
                self.log_msg(
                    f"[Tokenizer]: has quantizer {self.tokenizer.quantizer.__class__}"
                )

    def prepare_ema_models(self):
        if self.is_zero_2_3:
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

        # when distributed, there should be the same log_file
        if self.accelerator.use_distributed:
            input_lst = [log_file] * self.accelerator.num_processes
            output_lst = [None] * self.accelerator.num_processes
            torch.distributed.scatter_object_list(
                output_lst,
                input_lst,
            )
            log_file: Path = output_lst[self.accelerator.process_index]
            assert isinstance(log_file, Path), "log_file type should be Path"

        # logger
        self.logger.remove()
        log_format = (
            "<green>[{time:MM-DD HH:mm:ss}]</green>"
            " - <level>[{level}]</level> - <level>{message}</level>"
        )
        if not self.train_cfg.debug:
            self.logger.add(
                log_file,
                format=log_format,
                level="INFO",
                rotation="10 MB",
                enqueue=True,
                backtrace=True,
                colorize=False,
            )
        self.logger.add(
            sys.stdout,
            format=log_format,
            level="INFO",
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
        self, log_type: Literal["metric", "image"], logs: dict, step: int, **kwargs
    ):
        if not hasattr(self, "tb_logger") or not self.accelerator.is_main_process:
            return
        assert log_type in [
            "metric",
            "image",
        ], "log_type must be one of [metric, image]"

        if log_type == "metric":
            self.tb_logger.log(logs, step=step)
        elif log_type == "image":
            self.tb_logger.log_images(logs, step=step)

    def log_msg(self, *msgs, only_rank_one=True, level="INFO", sep=",", **kwargs):
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

        if only_rank_one:
            if self.accelerator.is_main_process:
                log_fn(str_msg(*msgs), **kwargs)
        else:  # not only rank zero
            with self.accelerator.main_process_first():
                msg_string = str_msg(*msgs)
                # prefix rank info
                msg_string = f"rank-{self.accelerator.process_index} | {msg_string}"
                log_fn(msg_string, **kwargs)

    def _get_tokenizer_params(self, for_optimizer=False):
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
            self.log_msg(f"[Optim]: add quantizer params into optimizer")
            for quant_p in quant_params:
                self.log_msg(f"[Optim]: quantizer param - {quant_p.shape}")
        else:
            quant_params = []

        if not for_optimizer:
            if self.sep_enc_dec:
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
                params = list(self.tokenizer.parameters())

            # * add with quantizer params
            params += quant_params
        else:
            raise NotImplementedError(f"not implemented")

        return params

    def _get_disc_params(self, for_optimizer=False):
        if not for_optimizer:
            return list(self.vq_loss_fn.discriminator.parameters())
        else:
            return self.vq_loss_fn.discriminator.state_dict()

    def get_optimizer_lr_scheduler(self):
        if (
            self.accelerator.state.deepspeed_plugin is None
            or "optimizer"
            not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            tokenizer_optim = hydra.utils.instantiate(
                self.train_cfg.tokenizer_optimizer
            )(self._get_tokenizer_params())
            disc_optim = hydra.utils.instantiate(self.train_cfg.disc_optimizer)(
                self._get_disc_params()
            )
        else:
            tokenizer_optim = DummyOptim([{"params": self._get_tokenizer_params()}])
            disc_optim = DummyOptim([{"params": self._get_disc_params()}])

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

        return tokenizer_optim, tokenizer_sched, disc_optim, disc_sched

    def prepare_for_training(self):
        # discriminator may have batch norm layer
        self.vq_loss_fn.discriminator = nn.SyncBatchNorm.convert_sync_batchnorm(
            self.vq_loss_fn.discriminator
        )
        self.log_msg("[Model] convert discriminator to sync batch norm")

        (
            self.vq_loss_fn.discriminator,
            self.tokenizer_optim,
            self.tokenizer_sched,
            self.disc_optim,
            self.disc_sched,
            self.train_dataloader,
        ) = self.accelerator.prepare(
            self.vq_loss_fn.discriminator,
            self.tokenizer_optim,
            self.tokenizer_sched,
            self.disc_optim,
            self.disc_sched,
            self.train_dataloader,
        )
        if self.sep_enc_dec:
            self.tokenizer_encoder, self.tokenizer_decoder = self.accelerator.prepare(
                self.tokenizer_encoder, self.tokenizer_decoder
            )
        else:
            self.tokenizer = self.accelerator.prepare(self.tokenizer)
        if self.quantizer is not None:
            self.quantizer = self.accelerator.prepare(self.quantizer)

    def step_train_state(self):
        self.train_state.update("train")

    def ema_update(self, mode="tokenizer"):
        if self.is_zero_2_3:
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

    def forward_tokenizer(self, x, ema: bool = False, is_testing: bool = False):
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
                if self.use_quantizer:
                    recon, q_loss, q_info = self.tokenizer(x)
                else:
                    recon = self.tokenizer(x)

        # basic out
        out_d = dict(latent=latent, recon=recon)

        # quantizer output dict
        if self.use_quantizer:
            _q_dict = dict(q_loss=q_loss, q_info=q_info, latent_q=latent)
        else:
            _q_dict = dict(q_loss=None, q_info=None, latent_q=None)
        out_d.update(_q_dict)

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

        # loss
        with self.accelerator.autocast():
            disc_train_loss_d, log_disc = self.vq_loss_fn.forward(
                inputs=x,
                reconstructions=out_d["recon"],
                optimizer_idx=optim_idx,
                global_step=self.global_step,
                last_layer=self.get_last_layer(),
                split=split,
                q_loss_total=out_d["q_loss"],
                q_loss_breakdown=out_d["q_info"],
                add_prefix=False,
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
        for p in model.parameters():
            p.requires_grad = not freeze

    def get_last_layer(self, use_ema: bool = False):
        if use_ema:
            if self.sep_enc_dec:
                return self.accelerator.unwrap_model(
                    self.ema_decoder
                ).ema_model.get_last_layer()
            else:
                return self.accelerator.unwrap_model(
                    self.ema_tokenizer
                ).ema_model.get_last_layer()
        else:
            if self.sep_enc_dec:
                return self.accelerator.unwrap_model(
                    self.tokenizer_decoder
                ).decoder.get_last_layer()
            else:
                return self.accelerator.unwrap_model(self.tokenizer).get_last_layer()

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
                            only_rank_one=False,
                            level="WARNING",
                        )
                    elif torch.isnan(param.grad).any():
                        self.log_msg(
                            f"step {self.global_step} - {name} has nan gradient, shaped as {param.shape}",
                            only_rank_one=False,
                            level="WARNING",
                        )
                        torch.nan_to_num(
                            param.grad, nan=0.0, posinf=1e5, neginf=-1e5, out=param.grad
                        )

        # clip gradient by norm
        if self.dtype != torch.float16:
            self.accelerator.clip_grad_norm_(
                model.parameters(), self.train_cfg.max_grad_norm
            )

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

    def train_step(
        self,
        batch: dict,
    ):
        x = batch["img"].to(self.device, self.dtype)  # [-1, 1]

        quality_track_n = self.train_cfg.track_metrics_duration
        quality_track_after = self.train_cfg.track_metrics_after
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
            else [
                self.tokenizer,
            ]
        )
        _accum_models.append(self.vq_loss_fn.discriminator)

        with self.accelerator.accumulate(*_accum_models):
            out_d = self.forward_tokenizer(x)

            # train tokenizer and discriminator
            tokenizer_loss, log_token_loss = self.train_tokenizer_step(x, out_d)
            disc_loss, log_disc_loss = self.train_disc_step(x, out_d)

            # track reconstruction quality
            if (
                quality_track_n >= 0
                and self.global_step % quality_track_n != 0
                and self.global_step >= quality_track_after
            ):
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
        assert log_token_loss is not None or log_disc_loss is not None, (
            "At least one of the logs should be provided"
        )

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

        strings = []
        if log_token_loss is not None:
            _log_token = dict_round_to_list_str(
                log_token_loss,
                n_round=n_round,
                select=[
                    "reconstruct_loss",
                    "ssim_loss",
                    "g_loss",
                ],
            )
            strings.extend(_log_token)

            if self.vq_loss_fn.use_perceptual_loss:
                _log_token = dict_round_to_list_str(
                    log_token_loss,
                    n_round=n_round,
                    select=["perceptual_loss", "gram_loss"],
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
                }

                _log_q = dict_round_to_list_str(
                    log_token_loss,
                    n_round,
                    select=_quant_logs_out_select.get(self.quantizer_type, None),
                )
                strings.extend(_log_q)

        else:
            _log_disc = dict_round_to_list_str(
                log_disc_loss,
                n_round,
                select=["disc_loss", "logits_real", "logits_fake", "lecam_loss"],
            )
            strings.extend(_log_disc)
            if log_disc_loss.get("r1_scale", 0) != 0:
                _disc_reg_logs = dict_round_to_list_str(
                    log_disc_loss,
                    n_round,
                    ["r1_loss", "r1_scale"],
                )
                strings.extend(_disc_reg_logs)

        return " - ".join(strings)

    def infinity_train_loader(self):
        while True:
            for batch in self.train_dataloader:
                yield batch

    def train_loop(self):
        _stop_train_and_save = False
        self.accelerator.wait_for_everyone()

        self.log_msg("[Train]: start training", only_rank_one=False)
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
        img = batch["img"].to(self.device, self.dtype)
        with torch.no_grad():
            recon = self.forward_tokenizer(img, ema=True, is_testing=True)["recon"]

        return recon

    def val_loop(self):
        if not hasattr(self, "_val_loader_iter"):
            # state in the loader generator
            self._val_loader_iter = iter(self.finite_val_loader())

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
        psnr_fn = PeakSignalNoiseRatio().to(device=self.device, dtype=self.dtype)
        ssim_fn = StructuralSimilarityIndexMeasure().to(
            device=self.device, dtype=self.dtype
        )
        loss_metrics = MeanMetric().to(device=self.device)

        _set_all_model_modes(train=False)

        for val_iter in range(self.val_cfg.max_val_iters):
            batch = next(self._val_loader_iter)
            recon = self.val_step(batch)

            recon_for_metrics = self.to_rgb(recon)
            batch_img_rgb = self.to_rgb(batch["img"].to(self.device))

            psnr_fn.update(batch_img_rgb, recon_for_metrics)
            ssim_fn.update(batch_img_rgb, recon_for_metrics)

            # recon loss
            loss = nn.functional.l1_loss(recon, batch["img"].to(recon))
            loss_metrics.update(loss)

        psnr_val = psnr_fn.compute()
        ssim_val = ssim_fn.compute()
        loss_val = loss_metrics.compute()

        # gather
        if self.accelerator.use_distributed:
            psnr_val = torch.distributed.reduce(
                psnr_val, op=torch.distributed.ReduceOp.AVG
            )
            ssim_val = torch.distributed.reduce(
                ssim_val, op=torch.distributed.ReduceOp.AVG
            )
            loss_val = torch.distributed.reduce(
                loss_val, op=torch.distributed.ReduceOp.AVG
            )

        if self.accelerator.is_main_process:
            psnr_val = psnr_val.item()
            ssim_val = ssim_val.item()
            loss_val = loss_val.item()

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

    def save_state(self):
        self.accelerator.save_state()
        self.log_msg("[State]: save states")

    def save_ema(self):
        self.accelerator.wait_for_everyone()
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
        self.accelerator.wait_for_everyone()

    def load_from_ema(self, ema_path: str, strict: bool = True):
        ema_path = Path(ema_path)

        if self.sep_enc_dec:
            # Load encoder to online model
            accelerate.load_checkpoint_in_model(
                self.tokenizer_encoder, ema_path / "encoder"
            )

            # Load decoder to online model
            accelerate.utils.load_checkpoint_in_model(
                self.tokenizer_decoder, ema_path / "decoder"
            )

        else:
            # Load combined model
            self.accelerator.utils.load_checkpoint_in_model(
                self.ema_tokenizer, ema_path / "tokenizer"
            )

        # Load discriminator to online model
        if self.train_cfg.finetune_strategy not in [
            "dcae_refine_decoder_head",
            "dcae_adapt_latent",
        ]:
            accelerate.utils.load_checkpoint_in_model(
                self.vq_loss_fn.discriminator, ema_path / "discriminator"
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
                    self.quantizer, ema_path / "quantizer"
                )
            else:
                raise RuntimeError(
                    "Quantizer not found in the checkpoint, please check your checkpoint."
                )

        # Prepare models
        self.prepare_ema_models()  # This will update EMA models with online models' weights
        self.prepare_for_training()

        self.log_msg(
            f"[EMA]: load ema weights to online models from {ema_path}, {strict=}"
        )

    def resume(self, path: str):
        self.log_msg("[Resume]: resume training")
        self.accelerator.load_state(path)
        self.accelerator.wait_for_everyone()

    def distributed_mean_dict(self, d: dict):
        assert isinstance(d, dict), f"Input should be a dict, but {type(d)} is given"

        if self.accelerator.num_processes == 1:
            return d

        dist_d_lst = [None for _ in range(self.accelerator.num_processes)]
        self.accelerator.wait_for_everyone()
        torch.distributed.all_gather_object(dist_d_lst, d)

        mean_d = {}
        for k in d.keys():
            mean_d[k] = sum(
                [
                    di[k].item() if isinstance(di[k], torch.Tensor) else di[k]
                    for di in dist_d_lst
                ]
            ) / len(dist_d_lst)

        return mean_d

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
            img_to_save.save(save_path)

        self.log_msg("[Visualize]: save visualization at {}".format(save_path))
        self.accelerator.wait_for_everyone()

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
            self.load_from_ema(self.train_cfg.ema_load_path)
        self.train_loop()


_key = "cosmos_sep_f8c32p1"
_configs = {
    # use pretrained cosmos world tokenizer (continous image configuration)
    "cosmos_sep_f8c16p4": "cosmos_post_train_f8c16p4",
    "cosmos_sep_f8c32p1": "cosmos_post_train_f8c32p1",
    "cosmos_sep_f16c16p4": "cosmos_post_train_f16c16p4",
    # unify encoder and decoder int one model
    "unicosmos_f16c16p1": "unicosmos_tokenizer_f16c16p1",
    "unicosmos_f16c16p2": "unicosmos_tokenizer_f16c16p2",
    # sana CDAE
    "sana_f8c16p1": "cdae_f8c16p1",
}[_key]


@hydra.main(
    config_path="configs/tokenizer_gan",
    config_name=_configs,
    version_base=None,
)
def main(cfg):
    with logger.catch():
        trainer = CosmosHyperspectralTokenizerTrainer(cfg)
        trainer.run()


if __name__ == "__main__":
    main()

import sys
import time
from functools import partial
from pathlib import Path
from typing import Literal

import accelerate
import hydra
import ipdb
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
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

sys.path.insert(0, "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer")
sys.path.insert(0, "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/src")
from src.data.hyperspectral_loader import get_hyperspectral_dataloaders
from src.stage1.cosmos.two_dim_diffusion_slots import TwoDimDiffusionSlots
from src.stage1.one_d_tokenizer.semanticist.diffuse_slot import DiffuseSlot
from src.stage1.two_d_vit_tokenizer.tokenizer import VITBSQModel, VITVQModel
from src.stage1.utilities.losses.gan_loss import VQLPIPSWithDiscriminator
from src.utilities.train_utils.state import StepsCounter

to_cont = partial(OmegaConf.to_container, resolve=True)
# omegaconf resolver
OmegaConf.register_new_resolver("eval", lambda x: eval(x))
OmegaConf.register_new_resolver("function", lambda x: hydra.utils.get_method(x))


## TODO:
# 1. add diffusion, rectify flow, or inductive momentum matching to the decoder
# 2. add BSQ or LFQ to discretize
# 3. add rectify flow for continuous latents; maskbit or maskgit for discrete latents

# 1. diffusion or rectified flow latents (slots) |-----> w/ or w/o kl (continuous)  --> diffusion/flow/imm/mar/autoregess+diffusion loss
#                                                |-----> bsq (discrete)             --> maskbit/maskgit/maskgen/
#                                                |-----> vq (discrete)              --> autoregressive/maskgit


class DiffusiveHyperspectralTokenizerTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.tokenizer_cfg = cfg.tokenizer
        self.loss_cfg = cfg.loss
        self.train_cfg = cfg.train
        self.val_cfg = cfg.val
        self.dataset_cfg = cfg.dataset
        self.ema_cfg = cfg.ema
        self.use_disc = cfg.train.use_disc

        # accelerator
        self.accelerator: Accelerator = hydra.utils.instantiate(cfg.accelerator)
        accelerate.utils.set_seed(2025, device_specific=True, deterministic=False)

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
        self.tokenizer = hydra.utils.instantiate(cfg.tokenizer)
        self.tokenizer: DiffuseSlot | TwoDimDiffusionSlots | VITBSQModel | VITVQModel
        self.quantizer_type = self.tokenizer.quantizer_type
        self.quantizer_kwargs = self.tokenizer.quantizer_kwargs

        # dataloader
        used_dataset = self.dataset_cfg.used
        self.log_msg(f"[Data]: using dataset {used_dataset}")
        self.train_dataset, self.train_dataloader = get_hyperspectral_dataloaders(
            wds_paths=self.dataset_cfg.wds_path_train,
            batch_size=self.dataset_cfg.batch_size_train,
            num_workers=self.dataset_cfg.num_workers,
            shuffle_size=self.dataset_cfg.shuffle_size,
            to_neg_1_1=True,
        )
        self.val_dataset, self.val_dataloader = get_hyperspectral_dataloaders(
            wds_paths=self.dataset_cfg.wds_path_val,
            batch_size=self.dataset_cfg.batch_size_val,
            num_workers=self.dataset_cfg.num_workers,
            shuffle_size=self.dataset_cfg.shuffle_size,
            to_neg_1_1=True,
        )

        # GAN loss
        if self.use_disc:
            self.vq_loss_fn: VQLPIPSWithDiscriminator = hydra.utils.instantiate(
                cfg.vq_loss
            )

        # optimizers and lr schedulers
        self.tokenizer_optim, self.tokenizer_sched, self.disc_optim, self.disc_sched = (
            self.get_optimizer_lr_scheduler()
        )
        # EMA models and accelerator prepare
        self.prepare_ema_models()
        self.prepare_for_training()

        # training state counter
        self.train_state = StepsCounter(["train"])

    def prepare_ema_models(self):
        if self.is_zero_2_3:
            return

        self.ema_tokenizer = EMA(
            self.tokenizer,
            beta=self.ema_cfg.beta,
            update_every=self.ema_cfg.update_every,
        )
        if self.use_disc:
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
            log_file = log_file / self.train_cfg.log.run_comment
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
            yaml_cfg = OmegaConf.to_yaml(self.cfg, resolve=False)
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
        log_type: Literal["metric", "image"],
        logs: dict,
        step: int | None = None,
        **kwargs,
    ):
        if (
            not hasattr(self, "tb_logger")
            or not self.accelerator.is_main_process
            or self.train_cfg.debug
        ):
            return

        if step is None:
            step = self.global_step

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
        else:
            msg_string = str_msg(*msgs)
            # prefix rank info
            msg_string = f"rank-{self.accelerator.process_index} | {msg_string}"
            log_fn(msg_string, **kwargs)

    def _get_tokenizer_params(self, for_optimizer=False):
        if not for_optimizer:
            params = self.tokenizer.parameters()
        else:
            params = self.tokenizer.state_dict()

        return params

    def _get_disc_params(self, for_optimizer=False):
        if not self.use_disc:
            return None

        if not for_optimizer:
            return list(self.vq_loss_fn.discriminator.parameters())
        else:
            return self.vq_loss_fn.discriminator.state_dict()

    def get_optimizer_lr_scheduler(self):
        tokenizer_optim, tokenizer_sched = None, None
        disc_optim, disc_sched = None, None

        if (
            self.accelerator.state.deepspeed_plugin is None
            or "optimizer"
            not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            tokenizer_optim = hydra.utils.instantiate(
                self.train_cfg.tokenizer_optimizer
            )(self._get_tokenizer_params())
            if self.use_disc:
                disc_optim = hydra.utils.instantiate(self.train_cfg.disc_optimizer)(
                    self._get_disc_params()
                )
        else:
            tokenizer_optim = DummyOptim([{"params": self._get_tokenizer_params()}])
            if self.use_disc:
                disc_optim = DummyOptim([{"params": self._get_disc_params()}])

        if (
            self.accelerator.state.deepspeed_plugin is None
            or "scheduler"
            not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            tokenizer_sched = hydra.utils.instantiate(self.train_cfg.tokenizer_sched)(
                optimizer=tokenizer_optim
            )
            if self.use_disc:
                disc_sched = hydra.utils.instantiate(self.train_cfg.disc_sched)(
                    optimizer=disc_optim
                )
        else:
            tokenizer_sched = DummyScheduler(tokenizer_optim)
            if self.use_disc:
                disc_sched = DummyScheduler(disc_optim)

        return tokenizer_optim, tokenizer_sched, disc_optim, disc_sched

    def prepare_for_training(self):
        # discriminator may have batch norm layer
        if self.use_disc:
            self.vq_loss_fn.discriminator = nn.SyncBatchNorm.convert_sync_batchnorm(
                self.vq_loss_fn.discriminator
            )
            self.log_msg("[Model] convert discriminator to sync batch norm")

        if self.use_disc:
            (
                self.tokenizer,
                self.ema_tokenizer.ema_model,
                self.vq_loss_fn.discriminator,
                self.ema_vq_disc,
                self.tokenizer_optim,
                self.tokenizer_sched,
                self.disc_optim,
                self.disc_sched,
                self.train_dataloader,
            ) = self.accelerator.prepare(
                self.tokenizer,
                self.ema_tokenizer.ema_model,
                self.vq_loss_fn.discriminator,
                self.ema_vq_disc,
                self.tokenizer_optim,
                self.tokenizer_sched,
                self.disc_optim,
                self.disc_sched,
                self.train_dataloader,
            )
        else:
            (
                self.tokenizer,
                self.ema_tokenizer.ema_model,
                self.tokenizer_optim,
                self.tokenizer_sched,
                self.train_dataloader,
            ) = self.accelerator.prepare(
                self.tokenizer,
                self.ema_tokenizer.ema_model,
                self.tokenizer_optim,
                self.tokenizer_sched,
                self.train_dataloader,
            )

    def step_train_state(self):
        self.train_state.update("train")

    def ema_update(self, mode="tokenizer"):
        if self.is_zero_2_3:
            return

        if mode == "tokenizer":
            self.ema_tokenizer.update()
        elif mode == "disc" and self.use_disc:
            self.ema_vq_disc.update()
        else:
            raise ValueError(f"Unknown mode {mode}")

    def forward_tokenizer(
        self,
        x,
        ema: bool = False,
        sample: bool = False,
        cfg: float = 1.0,
    ):
        with self.accelerator.autocast():
            # if not sample: losses has keys
            # pred_x_clean, diff_loss, quantizer_loss
            if not ema:
                if sample:
                    self.accelerator.unwrap_model(self.tokenizer).eval()
                losses, terms = self.tokenizer.forward(
                    x,
                    sample=sample,
                    epoch=self.global_step,
                    get_pred_x_clean=True,
                    cfg=cfg,
                )
            else:
                self.accelerator.unwrap_model(self.ema_tokenizer.ema_model).eval()
                losses, terms = self.ema_tokenizer.ema_model.forward(
                    x,
                    sample=sample,
                    epoch=self.global_step,
                    get_pred_x_clean=True,
                    cfg=cfg,
                )

        if sample:
            if ema:
                self.accelerator.unwrap_model(self.ema_tokenizer.ema_model).train()
            else:
                self.accelerator.unwrap_model(self.tokenizer).train()

            assert torch.is_tensor(losses), "if sampled, must output a sampled image"

        return losses, terms

    def forward_discriminator(
        self,
        x,
        recon,
        train_tokenizer: bool = True,
        split: str = "train",
        ema: bool = False,
        q_loss=None,
        q_loss_break=None,
    ):
        if ema:
            _non_ema_disc = self.vq_loss_fn.discriminator
            self.vq_loss_fn.discriminator = self.ema_vq_disc.ema_model

        optim_idx = 0 if train_tokenizer else 1  # tokenizer -> 0, discriminator -> 1

        # loss
        with self.accelerator.autocast():
            disc_train_loss_d, log_disc = self.vq_loss_fn.forward(
                x,
                recon,
                optim_idx,
                self.global_step,
                q_loss_total=q_loss,
                q_loss_breakdown=q_loss_break,
                last_layer=self.get_last_layer(),
                split=split,
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

    def freeze(self, mode="tokenizer"):
        def _freeze_model(model: nn.Module, freeze=True):
            for p in model.parameters():
                p.requires_grad = not freeze

        if mode == "tokenizer":
            _freeze_model(self.tokenizer, False)
            _freeze_model(self.vq_loss_fn.discriminator, True)
        elif mode == "disc" and self.use_disc:
            _freeze_model(self.ema_encoder, True)
            _freeze_model(self.ema_decoder, True)
            _freeze_model(self.vq_loss_fn.discriminator, False)
        else:
            raise ValueError(f"Unknown mode {mode}")

    def get_last_layer(self, use_ema: bool = False, which_model: str = "tokenizer"):
        if which_model == "tokenizer":
            if use_ema:
                return self.accelerator.unwrap_model(
                    self.ema_tokenizer
                ).ema_model.decoder.get_last_layer()
            else:
                return self.accelerator.unwrap_model(
                    self.tokenizer
                ).decoder.get_last_layer()
        else:
            if use_ema:
                return self.accelerator.unwrap_model(
                    self.ema_decoder
                ).ema_model.decoder.get_last_layer()
            else:
                return self.accelerator.unwrap_model(
                    self.tokenizer_decoder
                ).decoder.conv_out.weight

    def gradient_check(self, model: nn.Module | None):
        # safe check
        if model is None:
            return

        # check nan gradient
        if self.accelerator.sync_gradients:
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

            if self.dtype != torch.float16:
                self.accelerator.clip_grad_norm_(
                    model.parameters(), self.train_cfg.max_grad_norm
                )

    def train_tokenizer_step(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        losses: dict,
        terms: dict,
    ):
        log_losses_from_diffusion = {}

        # losses
        repa_loss_weighted = losses["repa_loss"] * self.loss_cfg.repa_loss_weight
        if not self.use_disc:
            # process in this if not use discriminator
            quant_loss_weighted = (
                losses["quantizer_loss"] * self.loss_cfg.quantizer_loss_weight
            )
            log_losses_from_diffusion["q_loss"] = quant_loss_weighted
        else:
            # process in discriminator class not in this statement
            quant_loss_weighted = 0.0
        diffusion_loss = losses["diff_loss"] + repa_loss_weighted + quant_loss_weighted

        # log
        log_losses_from_diffusion.update(
            {
                "total_loss": diffusion_loss,
                "diff_loss": losses["diff_loss"],
                "repa_loss": repa_loss_weighted,
            }
        )

        if self.use_disc:
            gen_loss, log_losses = self.forward_discriminator(
                x,
                recon,
                train_tokenizer=True,
                split="train",
                q_loss=losses["q_loss"],
                q_loss_break=terms["quan_info"],
            )  # l2, ssim, generator loss
            gen_loss += diffusion_loss
            log_losses.update(log_losses_from_diffusion)
        else:
            gen_loss = diffusion_loss
            log_losses = log_losses_from_diffusion

        if self.accelerator.sync_gradients:
            # backward
            self.accelerator.backward(gen_loss)
            self.gradient_check(self.tokenizer)
            self.tokenizer_optim.step()
            self.tokenizer_optim.zero_grad()
            self.tokenizer_sched.step()

            # ema update
            self.ema_update(mode="tokenizer")

        return gen_loss, log_losses

    def train_disc_step(self, x: torch.Tensor, recon: torch.Tensor):
        if not self.use_disc:
            return torch.tensor(0.0).to(self.device), {}

        disc_loss, log_disc = self.forward_discriminator(
            x,
            recon,
            train_tokenizer=False,
            split="train",
        )

        if self.accelerator.sync_gradients:
            # backward
            self.accelerator.backward(disc_loss)
            self.gradient_check(self.vq_loss_fn.discriminator)
            self.disc_optim.step()
            self.disc_optim.zero_grad()
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
            psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(self.device, self.dtype)
            ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(
                self.device, self.dtype
            )

            def check_quality(x, recon):
                x_q = self.to_rgb(x)
                recon_q = self.to_rgb(recon)
                psnr_fn.update(x_q, recon_q)
                ssim_fn.update(x_q, recon_q)

        # acummulate models
        _accum_models = [self.tokenizer]
        if self.use_disc:
            _accum_models.append(self.vq_loss_fn.discriminator)

        # train
        with self.accelerator.accumulate(
            _accum_models
        ):  # Warn: accumualte may cause some layers' params without gradient
            losses, terms = self.forward_tokenizer(x)
            recon = terms["pred_x_clean"]
            quan_info = terms["quan_info"]

            # loss
            tokenizer_loss, log_tokenizer_loss = self.train_tokenizer_step(
                x, recon, losses, terms
            )
            if self.use_disc:
                disc_loss, log_disc_loss = self.train_disc_step(x, recon)

            # track reconstruction quality
            if (
                quality_track_n >= 0
                and self.global_step % quality_track_n != 0
                and self.global_step >= quality_track_after
            ):
                check_quality(x, recon)

        self.step_train_state()

        # log losses
        if self.global_step % self.train_cfg.log.log_every == 0:
            _log_tok_losses = self.format_log(log_token_loss=log_tokenizer_loss)
            if self.use_disc:
                _log_disc_losses = self.format_log(log_disc_loss=log_disc_loss)

            self.log_msg(
                f"[Train State]: {self.tokenizer_optim.param_groups[0]["lr"]:.4e} | "
                f"[Step]: {self.global_step}/{self.train_cfg.max_steps}"
            )
            self.log_msg(f"[Train Tok]: {_log_tok_losses}")
            if self.use_disc:
                self.log_msg(f"[Train Disc]: {_log_disc_losses}")

            # tensorboard log
            self.tenb_log_any("metric", log_tokenizer_loss, self.global_step)
            if self.use_disc:
                self.tenb_log_any("metric", log_disc_loss, self.global_step)

        if (
            quality_track_n >= 0
            and self.global_step % quality_track_n == 0
            and self.global_step >= quality_track_after
        ):
            self.log_msg(
                f"[Train Metrics]: PSNR: {psnr_fn.compute():.3f}, SSIM: {ssim_fn.compute():.3f}"
            )

        if self.global_step % self.train_cfg.log.visualize_every == 0:
            self.visualize_reconstruction(
                x, recon, add_step=True, img_name="recon/train_recon"
            )

    def val_step(self, batch):
        with torch.no_grad():
            sampled, _ = self.forward_tokenizer(
                batch["img"].to(self.device).to(self.dtype),
                sample=True,
                ema=self.val_cfg.sample_use_ema,
                cfg=self.val_cfg.sample_cfg,
            )

        return sampled

    def format_log(
        self, log_token_loss: dict | None = None, log_disc_loss: dict | None = None
    ) -> str:
        assert (
            log_token_loss is not None or log_disc_loss is not None
        ), "At least one of the logs should be provided"

        n_round = 3

        strings = []
        if log_token_loss is not None:
            # perceptual loss is not supported for hyperspectral images
            _log_token = [
                f'diff_loss: {log_token_loss["diff_loss"]:.{n_round}f}',
                f'repa_loss: {log_token_loss["repa_loss"]:.{n_round}f}',
            ]
            if self.use_disc:
                _log_token_from_disc = [
                    f'recon_loss: {log_token_loss["reconstruct_loss"]:.{n_round}f}',
                    f'ssim_loss: {log_token_loss["ssim_loss"]:.{n_round}f}',
                    f'g_loss: {log_token_loss["g_loss"]:.{n_round}f}',
                ]
                _log_token.extend(_log_token_from_disc)

            if self.vq_loss_fn.use_perceptual_loss:
                _log_percep_loss = [
                    f"p_loss: {log_token_loss['perceptual_loss']:.{n_round}f}"
                ]
                _log_token.extend(_log_percep_loss)

            if self.quantizer_type is not None:
                _log_q = [
                    f'quantizer_loss: {log_token_loss["q_loss"]:.{n_round}f}',
                ]
                _log_token.extend(_log_q)

            strings.extend(_log_token)

        else:
            _log_disc = [
                f'disc_loss: {log_disc_loss["disc_loss"]:.{n_round}f}',
                f'logits_real: {log_disc_loss["logits_real"]:.{n_round}f}',
                f'logits_fake: {log_disc_loss["logits_fake"]:.{n_round}f}',
                f'lecam_loss: {log_disc_loss["lecam_loss"]:.{n_round}f}',
            ]
            strings.extend(_log_disc)
            if "r1_scale" in log_disc_loss:
                _disc_reg_logs = [
                    f'r1_scale: {log_disc_loss["r1_scale"]:.{n_round}f}',
                    f'r1_loss: {log_disc_loss["r1_loss"]:.{n_round}f}',
                ]
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

            # save
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

    def val_loop(self):
        if not hasattr(self, "_val_loader_iter"):
            # state in the loader generator
            self._val_loader_iter = iter(self.finite_val_loader())

        # track psnr and ssim
        psnr_fn = PeakSignalNoiseRatio().to(device=self.device, dtype=self.dtype)
        ssim_fn = StructuralSimilarityIndexMeasure().to(
            device=self.device, dtype=self.dtype
        )

        for val_iter in range(self.val_cfg.max_val_iters):
            batch = next(self._val_loader_iter)
            sampled = self.val_step(batch)

            sampled_for_metrics = self.to_rgb(sampled)
            batch_img_rgb = self.to_rgb(batch["img"].to(self.device))

            psnr_fn.update(batch_img_rgb, sampled_for_metrics)
            ssim_fn.update(batch_img_rgb, sampled_for_metrics)

        psnr_val = psnr_fn.compute()
        ssim_val = ssim_fn.compute()

        # gather
        if self.accelerator.use_distributed:
            psnr_val = torch.distributed.reduce(
                psnr_val, op=torch.distributed.ReduceOp.AVG
            )
            ssim_val = torch.distributed.reduce(
                ssim_val, op=torch.distributed.ReduceOp.AVG
            )

        if self.accelerator.is_main_process:
            psnr_val = psnr_val.item()
            ssim_val = ssim_val.item()

            self.log_msg(f"[Val]: PSNR: {psnr_val:.4f}, SSIM: {ssim_val:.4f}")
            self.tenb_log_any(
                "metric",
                {"psnr": psnr_val, "ssim": ssim_val},
                step=self.global_step,
            )

            # visualize the last val batch
            self.visualize_reconstruction(
                batch["img"], sampled, add_step=True, img_name="val_sampled/sampled"
            )

    def save_state(self):
        self.accelerator.save_state()
        self.log_msg("[State]: save states")

    def save_ema(self):
        self.accelerator.wait_for_everyone()
        ema_path = (
            self.proj_dir / "ema" / "ema.pt"
        )  # can not load by safetensors, if is a nested dict
        assert ema_path.suffix == ".pt", "Only .pt file is supported"

        if self.accelerator.is_main_process:
            ema_path.parent.mkdir(parents=True, exist_ok=True)

        # * is compiled model ==============
        if self.tokenizer_cfg.compile_model:
            # must unwarp the model by encoder, decoder, quantizer one by one
            # or it still contains `_orig_module` prefix key

            get_state_dict = lambda model: self.accelerator.unwrap_model(
                model, keep_torch_compile=False
            ).state_dict()
            ema_model: DiffuseSlot | TwoDimDiffusionSlots = (
                self.accelerator.unwrap_model(
                    self.ema_tokenizer.ema_model, keep_torch_compile=False
                )
            )
            encoder_params = get_state_dict(ema_model.encoder)
            decoder_params = get_state_dict(ema_model.decoder)

            # tokenizer total params
            tokenizer_params = {
                "encoder": encoder_params,
                "decoder": decoder_params,
            }
            if hasattr(ema_model, "encoder2slot"):
                encoder2slot_params = get_state_dict(ema_model.encoder2slot)
                tokenizer_params["encoder2slot"] = encoder2slot_params
            if ema_model.quantizer is not None:
                quantizer_params = get_state_dict(ema_model.quantizer)
                tokenizer_params["quantizer"] = quantizer_params

        # * if not compiled ==============
        else:
            tokenizer_params = self.accelerator.unwrap_model(
                self.ema_tokenizer.ema_model, keep_torch_compile=False
            ).state_dict()

        # * train state
        counter_params = self.train_state.state_dict()

        # * if has discriminator
        ema_ckpt = dict(
            tokenizer=tokenizer_params,
            counter=counter_params,
        )
        if self.use_disc:
            disc_params = self.accelerator.unwrap_model(
                self.ema_vq_disc.ema_model, keep_torch_compile=False
            ).state_dict()
            ema_ckpt["discriminator"] = disc_params

        # * save
        if self.accelerator.is_main_process:
            self.accelerator.save(ema_ckpt, ema_path)

        self.log_msg(f"[Ckpt]: save ema at {ema_path}")
        self.accelerator.wait_for_everyone()

    def load_from_ema(self, ema_path: str, strict: bool = True):
        # because safetensors does not support nested dict
        # loading a nested ckpt with extension .safetensors.pt will raise an HeaderTooLarge Error
        assert ema_path.endswith("pt"), "Only .pt file is supported"

        # load state dict
        ckpt = accelerate.utils.load_state_dict(ema_path)
        tokenizer_params = ckpt["tokenizer"]
        counter_params = ckpt["counter"]
        if self.use_disc:
            disc_params = ckpt["discriminator"]

        # load into models
        tokenizer = self.accelerator.unwrap_model(self.tokenizer)
        tokenizer.encoder.load_state_dict(tokenizer_params["encoder"], strict=strict)
        tokenizer.decoder.load_state_dict(tokenizer_params["decoder"], strict=strict)
        if hasattr(tokenizer, "encoder2slot"):
            tokenizer.encoder2slot.load_state_dict(
                tokenizer_params["encoder2slot"], strict
            )
        if tokenizer.quantizer is not None:
            tokenizer.quantizer.load_state_dict(
                tokenizer_params["quantizer"], strict=strict
            )
        if self.use_disc:
            disc = self.accelerator.unwrap_model(self.vq_loss_fn.discriminator)
            disc.load_state_dict(disc_params, strict=strict)
        self.train_state.load_state_dict(counter_params, strict=strict)

        # to prepare
        self.tokenizer = tokenizer
        if self.use_disc:
            self.vq_loss_fn.discriminator = disc
        self.prepare_ema_models()
        self.prepare_for_training()

        self.log_msg(f"[EMA]: load ema from {ema_path}, {strict=}")

    def resume(self, path: str):
        self.log_msg("[Resume]: resume training")
        self.accelerator.load_state(path, weights_only=False)
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
    ):
        x = self.to_rgb(x)
        if recon is not None:
            recon = self.to_rgb(recon)

        _only_n = 16
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
        if self.train_cfg.resume_path is not None:
            self.resume(self.train_cfg.resume_path)
        elif self.train_cfg.ema_load_path is not None:
            self.load_from_ema(self.train_cfg.ema_load_path)

        self.train_loop()


if __name__ == "__main__":
    # load config
    hydra.initialize("configs/2d_cosmos_diff", version_base=None)
    cfg = hydra.compose(config_name="2d_cosmos_no_quantizer")

    with logger.catch():
        trainer = DiffusiveHyperspectralTokenizerTrainer(cfg)
        trainer.run()

import io
import sys
import time
from functools import partial
from pathlib import Path

import accelerate
import hydra
import numpy as np
import PIL.Image as Image
import tifffile
import torch
import torch.nn as nn
import webdataset as wds
from accelerate import Accelerator
from accelerate.state import PartialState
from accelerate.utils import DummyOptim, DummyScheduler
from ema_pytorch import EMA
from kornia.utils.image import make_grid, tensor_to_image
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from src.stage1.cosmos.inference.utils import load_jit_model_shape_matched
from src.stage1.cosmos.losses.gan_loss import VQLPIPSWithDiscriminator
from src.stage1.cosmos.networks.configs import continuous_image

to_dict = partial(OmegaConf.to_container, resolve=True)
# omegaconf resolver
OmegaConf.registerregister_new_resolver_resolver("eval", lambda x: eval(x))
OmegaConf.register_new_resolver("function", lambda x: hydra.utils.get_method(x))


## TODO:
# 1. add diffusion, rectify flow, or inductive moumentumn matching to the decoder
# 2. add BSQ or LFQ to discretize
# 3. add rectify flow for continuous latents; maskbit or maskgit for discrete latents


def tiff_decoder(key, x):
    if key.endswith(".tiff"):
        return tifffile.imread(io.BytesIO(x))
    else:
        return x


def get_dict_tensor_mapper(to_neg_1_1=True):
    def wds_to_dict_tensor_mapper(sample):
        img = torch.as_tensor(sample["img.tiff"]).float()
        img = img / img.max()
        if to_neg_1_1:
            img = img * 2 - 1
        img = img.permute(-1, 0, 1)

        return {"img": img}

    return wds_to_dict_tensor_mapper


def get_hyperspectral_dataloaders(
    wds_paths: str | list[str],
    batch_size: int,
    num_workers: int,
    shuffle_size: int = 100,
    to_neg_1_1: bool = True,
):
    dict_mapper = get_dict_tensor_mapper(to_neg_1_1)

    part_state = PartialState()
    is_ddp = part_state.use_distributed

    dataset = wds.WebDataset(
        wds_paths,
        resampled=True,
        shardshuffle=True if is_ddp else False,
        cache_size=shuffle_size,
        nodesplitter=wds.shardlists.split_by_node
        if is_ddp
        else wds.shardlists.single_node_only,
        seed=2025,
        verbose=True,
        detshuffle=True if shuffle_size > 0 else False,
    )
    dataset = dataset.decode(tiff_decoder)
    dataset = dataset.map(dict_mapper)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=6,
        drop_last=False,
    )

    return dataset, dataloader


class StepsCounter:
    def __init__(self, step_names: list[str]):
        # all set to 0
        self.step_names = step_names
        for name in step_names:
            setattr(self, f"n_{name}_steps", 0)

    def __repr__(self):
        return f"StepsCounter({self.state_dict()})"

    def state_dict(self):
        return {
            f"n_{name}_steps": getattr(self, f"n_{name}_steps")
            for name in self.step_names
        }

    def load_state_dict(self, state_dict):
        for name in self.step_names:
            assert name in state_dict, f"Key {name} missing in state_dict"
            setattr(self, f"n_{name}_steps", state_dict[f"n_{name}_steps"])

    def update(self, name: str, update_n: int = 1):
        n_step = self.get(name)
        setattr(self, f"n_{name}_steps", n_step + update_n)

    def get(self, name: str):
        step_name = f"n_{name}_steps"
        assert hasattr(self, step_name), f"Key {step_name} missing in state_dict"

        return getattr(self, step_name)

    def __getitem__(self, name: str):
        return self.get(name)

    def __setitem__(self, name: str, value: int):
        name_set = f"n_{name}_steps"
        assert hasattr(self, name_set), f"Key {name_set} missing in state_dict"
        setattr(self, name_set, value)


class CosmosHyperspectralTokenizerTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.tokenizer_cfg = cfg.tokenizer
        self.train_cfg = cfg.train
        self.dataset_cfg = cfg.dataset
        self.ema_cfg = cfg.ema

        # accelerator
        self.accelerator: Accelerator = hydra.utils.instantiate(cfg.accelerator)
        accelerate.utils.set_seed(42, device_specific=True, deterministic=False)

        # logger
        log_file = self.configure_logger()
        log_dir = log_file.parent
        if not self.train_cfg.debug:
            log_dir.mkdir(parents=True, exist_ok=True)

        # attributes
        self.proj_dir = log_dir
        self.accelerator.project_configuration.project_dir = self.proj_dir
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
        tokenizer_config = continuous_image  # to_dict(self.tokenizer_cfg.config)
        self.tokenizer_encoder, self._enc_model_mody_keys = (
            load_jit_model_shape_matched(
                cfg.tokenizer.enc_path,
                tokenizer_config,
                device=self.device,
                part="encoder",
            )
        )
        self.tokenizer_decoder, self._dec_model_mody_keys = (
            load_jit_model_shape_matched(
                cfg.tokenizer.dec_path,
                tokenizer_config,
                device=self.device,
                part="decoder",
            )
        )
        self.tokenizer_encoder.train()
        self.tokenizer_decoder.train()
        self.tokenizer_encoder: nn.Module
        self.tokenizer_decoder: nn.Module

        # dataloader
        used_dataset = self.dataset_cfg.used
        self.log_msg(f"[Data]: using dataset {used_dataset}")
        self.train_dataset, self.train_dataloader = get_hyperspectral_dataloaders(
            wds_paths=self.dataset_cfg.wds_path,
            batch_size=self.dataset_cfg.batch_size,
            num_workers=self.dataset_cfg.num_workers,
            shuffle_size=self.dataset_cfg.shuffle_size,
            to_neg_1_1=True,
        )

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

    def prepare_ema_models(self):
        if self.is_zero_2_3:
            return

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

        self.logger.remove()
        log_format = "<green>[{time:MM-DD HH:mm:ss}]</green> - <level>[{level}]</level> - <level>{message}</level>"
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

        return log_file

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

        not_pretrained_keys = [
            *self._enc_model_mody_keys["not_pretrained_keys"],
            *self._dec_model_mody_keys["not_pretrained_keys"],
        ]

        if not for_optimizer:
            if self.train_cfg.finetune_strategy == "finetune_all":
                params = [
                    *list(self.tokenizer_encoder.parameters()),
                    *list(self.tokenizer_decoder.parameters()),
                ]
            elif self.train_cfg.finetune_strategy == "hier_finetune":
                params = [
                    {
                        "lr": self.train_cfg.tokenizer_optimizer.hier_base_lr,
                        "params": get_tokenizer_params_from_keys(not_pretrained_keys),
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

            else:
                raise ValueError(
                    f"Unknown finetune strategy: {self.train_cfg.finetune_strategy}"
                )

            self.log_msg(
                f"[Optimizer]: finetune strategy: {self.train_cfg.finetune_strategy}"
            )

        else:
            params = self.tokenizer_decoder.state_dict()

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
            self.tokenizer_encoder,
            self.tokenizer_decoder,
            self.vq_loss_fn.discriminator,
            self.tokenizer_optim,
            self.tokenizer_sched,
            self.disc_optim,
            self.disc_sched,
            self.train_dataloader,
        ) = self.accelerator.prepare(
            self.tokenizer_encoder,
            self.tokenizer_decoder,
            self.vq_loss_fn.discriminator,
            self.tokenizer_optim,
            self.tokenizer_sched,
            self.disc_optim,
            self.disc_sched,
            self.train_dataloader,
        )

    def step_train_state(self):
        self.train_state.update("train")

    def ema_update(self, mode="tokenizer"):
        if self.is_zero_2_3:
            return

        if mode == "tokenizer":
            self.ema_encoder.update()
            self.ema_decoder.update()
        elif mode == "disc":
            self.ema_vq_disc.update()
        else:
            raise ValueError(f"Unknown mode {mode}")

    def forward_tokenizer(self, x, ema: bool = False):
        with self.accelerator.autocast():
            if not ema:
                latent = self.tokenizer_encoder(x)[0]
                recon = self.tokenizer_decoder(latent)
            else:
                latent = self.ema_encoder(x)[0]
                recon = self.ema_decoder(latent)

        return latent, recon

    def forward_discriminator(
        self,
        x,
        recon,
        last_layer,
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
            disc_loss, log_disc = self.vq_loss_fn(
                x,
                recon,
                optim_idx,
                self.global_step,
                last_layer=self.get_last_layer(),
                split=split,
            )

        # back
        if ema:
            self.vq_loss_fn.discriminator = _non_ema_disc

        return disc_loss, log_disc

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
            _freeze_model(self.tokenizer_encoder, False)
            _freeze_model(self.tokenizer_decoder, False)
            _freeze_model(self.vq_loss_fn.discriminator, True)
        elif mode == "disc":
            _freeze_model(self.ema_encoder, True)
            _freeze_model(self.ema_decoder, True)
            _freeze_model(self.vq_loss_fn.discriminator, False)
        else:
            raise ValueError(f"Unknown mode {mode}")

    def get_last_layer(self, use_ema: bool = False):
        if use_ema:
            return self.accelerator.unwrap_model(
                self.ema_decoder
            ).ema_model.conv_out.weight
        else:
            return self.accelerator.unwrap_model(
                self.tokenizer_decoder
            ).decoder.conv_out.weight

    def gradient_check(self, model: nn.Module):
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

    def train_tokenizer_step(self, x: torch.Tensor, recon: torch.Tensor):
        gen_loss, log_losses = self.forward_discriminator(
            x,
            recon,
            last_layer=self.get_last_layer(),
            train_tokenizer=True,
            split="train",
        )

        if self.accelerator.sync_gradients:
            # backward
            self.accelerator.backward(gen_loss)
            self.gradient_check(self.tokenizer_encoder)
            self.gradient_check(self.tokenizer_decoder)
            self.tokenizer_optim.step()
            self.tokenizer_optim.zero_grad()
            self.tokenizer_sched.step()

            # ema update
            self.ema_update(mode="tokenizer")

        return gen_loss, log_losses

    def train_disc_step(self, x: torch.Tensor, recon: torch.Tensor):
        disc_loss, log_disc = self.forward_discriminator(
            x,
            recon,
            last_layer=self.get_last_layer(),
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

        with (
            self.accelerator.accumulate()
        ):  # Warning: accumualte may cause some layers' params without gradient
            _, recon = self.forward_tokenizer(x)

            # loss
            tokenizer_loss, log_token_loss = self.train_tokenizer_step(x, recon)
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
            _log_tok_losses = self.format_log(log_token_loss=log_token_loss)
            _log_disc_losses = self.format_log(log_disc_loss=log_disc_loss)

            self.log_msg(
                f'[Train State]: {self.tokenizer_optim.param_groups[0]["lr"]:.4e} | [Step]: {self.global_step}/{self.train_cfg.max_steps}'
            )
            self.log_msg(f"[Train Tok]: {_log_tok_losses}")
            self.log_msg(f"[Train Disc]: {_log_disc_losses}")

        if (
            quality_track_n >= 0
            and self.global_step % quality_track_n == 0
            and self.global_step >= quality_track_after
        ):
            self.log_msg(
                f"[Train Metrics]: PSNR: {psnr_fn.compute():.3f}, SSIM: {ssim_fn.compute():.3f}"
            )

        if self.global_step % self.train_cfg.log.visualize_every == 0:
            self.visualize_reconstruction(x, recon, add_step=False)

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
                f'recon_loss: {log_token_loss["train/reconstruct_loss"]:.{n_round}f}',
                f'ssim_loss: {log_token_loss["train/ssim_loss"]:.{n_round}f}',
                f'g_loss: {log_token_loss["train/g_loss"]:.{n_round}f}',
            ]
            strings.extend(_log_token)
        else:
            _log_disc = [
                f'disc_loss: {log_disc_loss["train/disc_loss"]:.{n_round}f}',
                f'logits_real: {log_disc_loss["train/logits_real"]:.{n_round}f}',
                f'logits_fake: {log_disc_loss["train/logits_fake"]:.{n_round}f}',
                f'lecam_loss: {log_disc_loss["train/lecam_loss"]:.{n_round}f}',
                f'non_saturate_d_loss: {log_disc_loss["train/non_saturated_d_loss"]:.{n_round}f}',
            ]
            strings.extend(_log_disc)

        return " - ".join(strings)

    def infinity_train_loader(self):
        while True:
            for batch in self.train_dataloader:
                yield batch

    def train_loop(self):
        self.log_msg("[Train]: start training")

        _stop_train_and_save = False
        for batch in self.infinity_train_loader():
            # train step
            self.train_step(batch)

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

    def save_state(self):
        self.accelerator.save_state()
        self.log_msg("[State]: save states")

    def save_ema(self):
        self.accelerator.wait_for_everyone()
        ema_path = self.proj_dir / "ema" / "ema.safetensors"
        ema_path.parent.mkdir(parents=True, exist_ok=True)

        enc_params = self.accelerator.unwrap_model(
            self.ema_encoder.ema_model
        ).state_dict()
        dec_params = self.accelerator.unwrap_model(
            self.ema_decoder.ema_model
        ).state_dict()
        disc_params = self.accelerator.unwrap_model(
            self.ema_vq_disc.ema_model
        ).state_dict()
        counter_params = self.train_state.state_dict()

        ema_ckpt = dict(
            encoder=enc_params,
            decoder=dec_params,
            discriminator=disc_params,
            counter=counter_params,
        )
        self.accelerator.save(ema_ckpt, ema_path)
        self.log_msg(f"[Ckpt]: save ema at {ema_path}")
        self.accelerator.wait_for_everyone()

    def load_from_ema(self, ema_path: str, strict: bool = True):
        ckpt = accelerate.utils.load_state_dict(ema_path)
        enc_params = ckpt["encoder"]
        dec_params = ckpt["decoder"]
        disc_params = ckpt["discriminator"]
        counter_params = ckpt["counter"]

        # load
        tokenizer_encoder, tokenizer_decoder, disc = list(
            map(
                self.accelerator.unwrap_model,
                [self.ema_encoder, self.ema_decoder, self.ema_vq_disc],
            )
        )

        tokenizer_encoder.load_state_dict(enc_params, strict=strict)
        tokenizer_decoder.load_state_dict(dec_params, strict=strict)
        disc.load_state_dict(disc_params, strict=strict)
        self.train_state.load_state_dict(counter_params, strict=strict)

        # to prepare
        self.tokenizer_encoder = tokenizer_encoder
        self.tokenizer_decoder = tokenizer_decoder
        self.vq_loss_fn.discriminator = disc
        self.prepare_ema_models()
        self.prepare_for_training()

        self.log_msg(f"[EMA]: load ema from {ema_path}, {strict=}")

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
        recon: torch.Tensor,
        img_name: str = "train_original_recon.jpg",
        add_step: bool = False,
    ):
        x = self.to_rgb(x)
        recon = self.to_rgb(recon)
        _only_n = 16
        to_img = lambda x: tensor_to_image(make_grid(x[:_only_n], n_row=4, padding=2))
        c = recon.shape[1]
        # is rgb or gray images
        if c in (1, 3):
            x_np = to_img(x)
            recon_np = to_img(recon)
        else:
            # prefix some channels for RGB channels
            _prefixed_rgb_channels = {
                12: [4, 3, 2],
                13: [4, 3, 2],
                8: [4, 2, 0],
                4: [3, 2, 1],
                224: [128, 68, 32],
                50: [25, 15, 5],
            }
            rgb_channels = _prefixed_rgb_channels[c]
            x_np = to_img(x[:, rgb_channels])
            recon_np = to_img(recon[:, rgb_channels])

        # cat original and reconstructed images
        img = np.concatenate([x_np, recon_np], axis=1)

        # save
        img = (img * 255.0).astype(np.uint8)
        img_to_save = Image.fromarray(img)
        if add_step:
            img_name = f"{self.global_step}_{img_name}"

        save_path = Path(self.proj_dir) / "train_vis" / img_name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if self.accelerator.is_main_process:
            img_to_save.save(save_path)

        self.log_msg("[Visualize]: save visualization at {}".format(save_path))
        self.accelerator.wait_for_everyone()

    def run(self):
        with self.logger.catch():
            self.train_loop()


if __name__ == "__main__":
    # load config
    hydra.initialize("configs", version_base=None)
    cfg = hydra.compose(config_name="pos_train")

    trainer = CosmosHyperspectralTokenizerTrainer(cfg)
    trainer.run()


"""
    'attn_resolutions' =
    [32]
    'channels' =
    128
    'channels_mult' =
    [2, 4, 4]
    'dropout' =
    0.0
    'in_channels' =
    12
    'spatial_compression' =
    8
    'num_res_blocks' =
    2
    'out_channels' =
    12
    'resolution' =
    1024
    'patch_size' =
    4
    'patch_method' =
    'haar'
    'latent_channels' =
    16
    'z_channels' =
    16
    'z_factor' =
    1
    'name' =
    'CI'
    'formulation' =
    'AE'
    'encoder' =
    'Default'
    'decoder' =g
    'Default'
    'act_checkpoint' =
    True
"""
"""
float32:
latent: tensor([-0.6327,  0.1611,  0.1389,  0.1916,  0.1961,  0.1893,  0.1862,  0.1863,
         0.1862,  0.1862], device='cuda:0', grad_fn=<SliceBackward0>)
        
recon: tensor([ 0.0542,  0.0098, -0.0485,  0.0064,  0.0462,  0.0799, -0.0232,  0.0005,
         0.0055,  0.0414], device='cuda:0', grad_fn=<SliceBackward0>)
"""

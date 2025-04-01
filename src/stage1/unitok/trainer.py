import sys
import math
import torch
from copy import deepcopy
from pprint import pformat
from typing import Callable, Optional, Tuple
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import config, nan, dist
from utils.optimizer import AmpOptimizer
from utils.diffaug import DiffAug
from utils.loss import hinge_loss, linear_loss, softplus_loss
from utils.lpips import LPIPS
from utils.misc import unwrap_model
from utils.logger import MetricLogger, wandb_log
from open_clip.loss import ClipLoss


class Trainer(object):
    def __init__(
        self,
        args: config.Args,
        unitok: DDP,
        disc: DDP,
        unitok_optim: AmpOptimizer,
        disc_optim: AmpOptimizer,
        lpips_loss: LPIPS,
        clip_loss: ClipLoss,
    ):
        super(Trainer, self).__init__()
        self.unitok = unitok
        self.disc = disc
        self.unitok_optim = unitok_optim
        self.disc_optim = disc_optim

        self.dcrit = args.dcrit
        self.d_criterion = {
            "hg": hinge_loss,
            "hinge": hinge_loss,
            "sp": softplus_loss,
            "softplus": softplus_loss,
            "ln": linear_loss,
            "lin": linear_loss,
            "linear": linear_loss,
        }[self.dcrit]
        self.daug = DiffAug(prob=args.disc_aug_prob, cutout=0.2)

        self.wei_l1 = args.l1
        self.wei_l2 = args.l2
        self.wei_entropy = args.le
        self.wei_lpips = args.lp
        self.wei_disc = args.ld
        self.wei_quant = args.lq

        self.lpips_loss = lpips_loss
        self.lp_reso = args.lpr
        self.adapt_wei_disc = args.ld > 0
        self.adapt_type = args.gada

        self.bcr = args.bcr
        if self.bcr > 0:
            self.bcr_strong_aug = DiffAug(prob=1, cutout=args.bcr_cut)

        self.clip_loss = clip_loss
        self.wei_clip = args.lc

        self.grad_ckpt = args.grad_ckpt

        self.vae_bs = args.vae_local_bs

        self.dbg_nan = args.dbg_nan
        self.dbg_unused = args.dbg_unused_param
        if self.dbg_nan:
            print("[dbg_nan mode on]")
            nan.debug_nan_hook(unitok)
            nan.debug_nan_hook(disc)

    def train_step(
        self,
        img,
        text,
        global_iter: int,
        stepping: bool,
        metric_logger: MetricLogger,
        warmup_disc_schedule: float,
        fade_blur_schedule: float,
        report_wandb: bool = False,
    ) -> Tuple[torch.Tensor, Optional[float], Optional[torch.Tensor], Optional[float]]:
        if warmup_disc_schedule < 1e-6:
            warmup_disc_schedule = 0
        if fade_blur_schedule < 1e-6:
            fade_blur_schedule = 0
        loggable = global_iter % 50 == 0

        # vae loss
        vae_img = img[: self.vae_bs]

        with self.unitok_optim.amp_ctx:
            output = self.unitok(img, self.vae_bs, text, ret_usages=loggable)
            rec_img = output["img_rec"]
            Lq, Le = output["vq_loss"], output["entropy_loss"]

        L1 = F.l1_loss(rec_img, vae_img)
        L2 = F.mse_loss(rec_img, vae_img)
        Lrec = L1 * self.wei_l1 + L2 * self.wei_l2

        Lpip = 0.0
        if vae_img.shape[-2] >= self.lp_reso and self.wei_lpips > 0:
            self.lpips_loss.forward
            Lpip = self.lpips_loss(vae_img, rec_img)
        Lnll = Lrec + self.wei_lpips * Lpip

        # clip loss

        with self.unitok_optim.amp_ctx:
            clip_losses = self.clip_loss(
                image_features=output["clip_image_features"],
                text_features=output["clip_text_features"],
                logit_scale=output["logit_scale"],
                output_dict=True,
            )
            Lc = sum(clip_losses.values())

        if warmup_disc_schedule > 0:
            for d in self.disc.parameters():
                d.requires_grad = False
            self.disc.eval()
            with self.disc_optim.amp_ctx:
                Lg = -self.disc(
                    self.daug.aug(rec_img, fade_blur_schedule), grad_ckpt=False
                ).mean()
            self.disc.train()

            wei_g = warmup_disc_schedule * self.wei_disc
            if self.adapt_wei_disc:
                last_layer = unwrap_model(self.unitok).decoder.get_last_param()
                w = torch.autograd.grad(Lnll, last_layer, retain_graph=True)[
                    0
                ].data.norm() / torch.autograd.grad(Lg, last_layer, retain_graph=True)[
                    0
                ].data.norm().add_(1e-6)
                if self.adapt_type % 10 == 0:
                    w.clamp_(0.0, 1e4)
                elif self.adapt_type % 10 == 1:
                    w.clamp_(0.015, 1e4)
                elif self.adapt_type % 10 == 2:
                    w.clamp_(0.1, 10)
                    w = min(max(w, 0.1), 10)
                elif self.adapt_type % 10 == 3:
                    w.clamp_(0.0, 1e4).sqrt_()
                elif self.adapt_type % 10 == 4:
                    w.clamp_(0.015, 1.5)

                wei_g = wei_g * w

            Lv = (
                Lnll
                + self.wei_quant * Lq
                + self.wei_entropy * Le
                + wei_g * Lg
                + self.wei_clip * Lc
            )
        else:
            Lv = Lnll + self.wei_quant * Lq + self.wei_entropy * Le + self.wei_clip * Lc
            Lg = wei_g = 0.0

        grad_norm_g, scale_log2_g = self.unitok_optim.backward_clip_step(
            stepping=stepping, loss=Lv
        )

        # [discriminator loss]
        if warmup_disc_schedule > 0:
            rec_img = rec_img.data
            for d in self.disc.parameters():
                d.requires_grad = True
            with self.disc_optim.amp_ctx:
                logits_real = self.disc(
                    self.daug.aug(vae_img, fade_blur_schedule), grad_ckpt=self.grad_ckpt
                ).float()
                logits_fake = self.disc(
                    self.daug.aug(rec_img, fade_blur_schedule), grad_ckpt=self.grad_ckpt
                ).float()
            acc_real = (logits_real.data > 0).float().mean().mul_(100)
            acc_fake = (logits_fake.data < 0).float().mean().mul_(100)
            Ld = self.d_criterion(logits_real) + self.d_criterion(-logits_fake)

            if self.bcr:
                with self.disc_optim.amp_ctx:
                    Lbcr = (
                        F.mse_loss(
                            self.disc(
                                self.bcr_strong_aug.aug(vae_img, 0.0),
                                grad_ckpt=self.grad_ckpt,
                            ).float(),
                            logits_real,
                        )
                        + F.mse_loss(
                            self.disc(
                                self.bcr_strong_aug.aug(rec_img, 0.0),
                                grad_ckpt=self.grad_ckpt,
                            ).float(),
                            logits_fake,
                        )
                    ).mul_(self.bcr)
                Ld += Lbcr
            else:
                Lbcr = 0.0
            grad_norm_d, scale_log2_d = self.disc_optim.backward_clip_step(
                stepping=stepping, loss=Ld
            )
            Ld = Ld.data.clone()
        else:
            Ld = Lbcr = acc_real = acc_fake = grad_norm_d = 0.0
            scale_log2_d = None

        if not math.isfinite(Lnll + Ld + wei_g):
            for n, v in zip(["Lrec", "Lpip", "Ld", "wei_g"], [Lrec, Lpip, Ld, wei_g]):
                if not math.isfinite(v):
                    print(
                        f"[rk{dist.get_rank():02d}] {n} is {v}, stopping training!",
                        force=True,
                        flush=True,
                    )
            sys.exit(666)

        with torch.no_grad():
            unwrap_model(self.unitok).logit_scale.clamp_(0, math.log(100))

        # [zero_grad]
        if stepping:
            if self.dbg_nan:
                nan.debug_nan_grad(self.unitok), nan.debug_nan_grad(self.disc)
                nan.debug_nan_param(self.unitok), nan.debug_nan_param(self.disc)
            if self.dbg_unused:
                ls = []
                for n, p in self.unitok.named_parameters():
                    # or tuple(p.grad.shape) == (512, 512, 1, 1):
                    if p.grad is None and n not in {"quantize.embedding.weight"}:
                        ls.append(n)
                for n, p in self.disc.named_parameters():
                    if p.grad is None:  # or tuple(p.grad.shape) == (512, 512, 1, 1):
                        ls.append(n)
                if len(ls):
                    print(f"unused param: {ls}", flush=True, file=sys.stderr)

            self.unitok_optim.optimizer.zero_grad(set_to_none=True)
            self.disc_optim.optimizer.zero_grad(set_to_none=True)

        # logging
        if loggable:
            metric_logger.update(
                L1=L1,
                Lnll=Lnll,
                Ld=Ld,
                Lc=Lc,
                Wg=wei_g,
                acc_real=acc_real,
                acc_fake=acc_fake,
                gnm=grad_norm_g,
                dnm=grad_norm_d,
            )

        if report_wandb:
            log_ferq = 50
            wandb_log({"L1": L1}, step=global_iter, log_ferq=log_ferq)
            # wandb_log({'Lrec': Lrec}, step=global_iter, log_ferq=log_ferq)
            wandb_log({"Lnll": Lnll}, step=global_iter, log_ferq=log_ferq)
            wandb_log({"Lq": Lq}, step=global_iter, log_ferq=log_ferq)
            wandb_log({"Lc": Lc}, step=global_iter, log_ferq=log_ferq)
            wandb_log(
                {"Codebook_usage": output["codebook_usages"]},
                step=global_iter,
                log_ferq=log_ferq,
            )
            wandb_log({"Le": Le}, step=global_iter, log_ferq=log_ferq)
            if self.wei_lpips > 0:
                wandb_log({"Lpip": Lpip}, step=global_iter, log_ferq=log_ferq)
            if warmup_disc_schedule > 0:
                wandb_log({"Ldisc": Ld - Lbcr}, step=global_iter, log_ferq=log_ferq)
                wandb_log({"Lbcr": Lbcr}, step=global_iter, log_ferq=log_ferq)
                wandb_log({"Lg": Lg}, step=global_iter, log_ferq=log_ferq)
                wandb_log({"Wei_g": wei_g}, step=global_iter, log_ferq=log_ferq)
                wandb_log(
                    {"Disc_accu_real": acc_real}, step=global_iter, log_ferq=log_ferq
                )
                wandb_log(
                    {"Disc_accu_fake": acc_fake}, step=global_iter, log_ferq=log_ferq
                )
                wandb_log(
                    {"Disc_accu_avg": (acc_real + acc_fake) * 0.5},
                    step=global_iter,
                    log_ferq=log_ferq,
                )
            if scale_log2_g is not None:
                wandb_log(
                    {"Scaler_g": scale_log2_g}, step=global_iter, log_ferq=log_ferq
                )
            if scale_log2_d is not None:
                wandb_log(
                    {"Scaler_d": scale_log2_d}, step=global_iter, log_ferq=log_ferq
                )
            wandb_log({"Gradnorm_g": grad_norm_g}, step=global_iter, log_ferq=log_ferq)
            wandb_log({"Gradnorm_d": grad_norm_d}, step=global_iter, log_ferq=log_ferq)
            wandb_log(
                {"Disc_warmup_schedule": warmup_disc_schedule},
                step=global_iter,
                log_ferq=log_ferq,
            )
            wandb_log(
                {"Disc_fade_blur_schedule": fade_blur_schedule},
                step=global_iter,
                log_ferq=log_ferq,
            )
            wandb_log(
                {"Logit_scale": output["logit_scale"]},
                step=global_iter,
                log_ferq=log_ferq,
            )
        return

    def __repr__(self):
        return (
            f'\n'
            f'[{type(self).__name__}.config]: {pformat(self.get_config(), indent=2, width=250)}\n'
            f'[{type(self).__name__}.structure]: {super(Trainer, self).__repr__().replace(Trainer.__name__, "")}'
        )

    def get_config(self):
        return {
            "dcrit": self.dcrit,
            "wei_l1": self.wei_l1,
            "wei_l2": self.wei_l2,
            "wei_lpips": self.wei_lpips,
            "wei_disc": self.wei_disc,
            "wei_clip": self.wei_clip,
            "bcr": self.bcr,
        }

    def state_dict(self):
        state = {"config": self.get_config()}
        for k in ("unitok", "disc", "unitok_optim", "disc_optim"):
            m = getattr(self, k)
            if m is None:
                continue
            if isinstance(m, DDP):
                m = unwrap_model(m)
            if hasattr(m, "_orig_mod"):
                m = m._orig_mod
            state[k] = m.state_dict()
        return state

    def load_state_dict(self, state, strict=True):
        for k in ("unitok", "disc", "unitok_optim", "disc_optim"):
            m = getattr(self, k)
            if m is not None:
                if isinstance(m, DDP):
                    m = unwrap_model(m)
                if hasattr(m, "_orig_mod"):
                    m = m._orig_mod
                ret = m.load_state_dict(state[k], strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f"[VAETr.load_state_dict] {k} missing:  {missing}")
                    print(f"[VAETr.load_state_dict] {k} unexpected:  {unexpected}")
        config: dict = state.pop("config", None)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f"[VAETr.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})"
                    if strict:
                        raise AttributeError(err)
                    else:
                        print(err)

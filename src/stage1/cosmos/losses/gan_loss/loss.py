from collections import namedtuple
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.state import AcceleratorState, PartialState
from einops import rearrange
from kornia.losses import SSIMLoss
from loguru import logger
from lpips import LPIPS

from .patchgan_discriminator import NLayerDiscriminator, weights_init
from .stylegan import StyleGANDiscriminator
from .stylegan3d import StyleGAN3DDiscriminator
from .stylegan_utils.ops.conv2d_gradfix import no_weight_gradients


class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


def dict_to_namedtuple(dictionary: dict, name="NamedTuple"):
    """Convert a dictionary to a namedtuple dynamically"""
    if not dictionary:
        return None

    # Create namedtuple type with dictionary keys as fields
    NamedTupleClass = namedtuple(name, dictionary.keys())

    # Convert dict values to tuple (handles nested dicts recursively)
    values = tuple(
        dict_to_namedtuple(v, name) if isinstance(v, dict) else v
        for v in dictionary.values()
    )

    # Instantiate namedtuple
    return NamedTupleClass(*values)


def dict_add_prefix(d: dict, prefix: str = "train"):
    return {f"{prefix}/{k}": v for k, v in d.items()}


def maybe_in_dict_update(
    q_info_dict: dict | namedtuple, may_used_keys: list[str], saved_dict: dict
):
    if isinstance(q_info_dict, namedtuple):
        q_info_dict = dict(q_info_dict._asdict())

    for key in may_used_keys:
        if key in q_info_dict:
            value = q_info_dict[key]
            if torch.is_tensor(value):
                value = value.detach()
            saved_dict[key] = value

    return saved_dict


# *==============================================================
# * Discriminator Losses
# *==============================================================


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real))
        + torch.mean(torch.nn.functional.softplus(logits_fake))
    )
    return d_loss


def _sigmoid_cross_entropy_with_logits(labels, logits):
    """
    non-saturating loss
    """
    zeros = torch.zeros_like(logits, dtype=logits.dtype)
    condition = logits >= zeros
    relu_logits = torch.where(condition, logits, zeros)
    neg_abs_logits = torch.where(condition, -logits, logits)
    return relu_logits - logits * labels + torch.log1p(torch.exp(neg_abs_logits))


def non_saturate_gen_loss(logits_fake):
    """
    logits_fake: [B 1 H W]
    """
    B, _, _, _ = logits_fake.shape
    logits_fake = logits_fake.reshape(B, -1)
    logits_fake = torch.mean(logits_fake, dim=-1)
    gen_loss = torch.mean(
        _sigmoid_cross_entropy_with_logits(
            labels=torch.ones_like(logits_fake), logits=logits_fake
        )
    )

    return gen_loss


def non_saturate_discriminator_loss(logits_real, logits_fake):
    B, _, _, _ = logits_fake.shape
    logits_real = logits_real.reshape(B, -1)
    logits_fake = logits_fake.reshape(B, -1)
    logits_fake = logits_fake.mean(dim=-1)
    logits_real = logits_real.mean(dim=-1)

    real_loss = _sigmoid_cross_entropy_with_logits(
        labels=torch.ones_like(logits_real), logits=logits_real
    )

    fake_loss = _sigmoid_cross_entropy_with_logits(
        labels=torch.zeros_like(logits_fake), logits=logits_fake
    )

    discr_loss = real_loss.mean() + fake_loss.mean()
    return discr_loss


class LeCAM_EMA(object):
    def __init__(self, init=0.0, decay=0.999):
        self.logits_real_ema = init
        self.logits_fake_ema = init
        self.decay = decay

    def update(self, logits_real, logits_fake):
        self.logits_real_ema = self.logits_real_ema * self.decay + torch.mean(
            logits_real
        ).item() * (1 - self.decay)
        self.logits_fake_ema = self.logits_fake_ema * self.decay + torch.mean(
            logits_fake
        ).item() * (1 - self.decay)


def lecam_reg(real_pred, fake_pred, lecam_ema):
    reg = torch.mean(F.relu(real_pred - lecam_ema.logits_fake_ema).pow(2)) + torch.mean(
        F.relu(lecam_ema.logits_real_ema - fake_pred).pow(2)
    )
    return reg


def d_r1_loss(logits_real, img_real):
    with no_weight_gradients():
        (grad_real,) = torch.autograd.grad(
            outputs=logits_real.sum(), inputs=img_real, allow_unused=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


# *==============================================================
# * Generator losses
# *==============================================================


def hinge_g_loss(logits_fake):
    return -torch.mean(logits_fake)


def vanilla_g_loss(logits_fake):
    return torch.mean(F.softplus(-logits_fake))


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(
        self,
        # discriminator
        disc_start: int = 0,
        disc_factor: float = 1.0,
        disc_weight: float = 1.0,
        disc_reg_freq: int = 0,
        disc_reg_r1: float = 10,
        # disc network cfg
        disc_network_type: str = "patchgan",
        disc_input_size: int = 256,
        disc_in_channels: int = 3,
        disc_num_layers: int = 3,
        use_actnorm: bool = False,
        disc_conditional: bool = False,
        disc_ndf: int = 64,
        disc_loss: str = "hinge",
        # codebook losses
        quantizer_options: dict | None = None,
        # perceptual loss
        perceptual_weight: float = 1.0,
        perceptual_type: str = "vgg",
        # generator loss
        pixelloss_weight: float = 1.0,
        gen_loss_weight: float | None = None,
        # quantizer losses
        quantizer_type: str = "bsq",  # [bsq, vq]
        # other losses
        lecam_loss_weight: float | None = None,
        loss_ssim: bool = False,
        ssim_weight: float = 0.1,
        # if is video
        num_frames: int = 1,
    ):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla", "non_saturate"]
        assert quantizer_type in ["bsq", "vq"]
        self.pixel_weight = pixelloss_weight
        self.quantizer_type = quantizer_type
        self.disc_reg_freq = disc_reg_freq
        self.disc_reg_r1 = disc_reg_r1
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_network_type = disc_network_type
        self.disc_conditional = disc_conditional
        self.with_ssim = loss_ssim
        self.num_frames = num_frames

        # * quantizer options
        if quantizer_type == "vq":
            default_quant_opts = dict(
                commit_weight=0.25,
                codebook_weight=1.0,
                codebook_enlarge_ratio=3,
                codebook_enlarge_steps=2000,
            )
        elif quantizer_type == "bsq":
            default_quant_opts = dict(
                codebook_weight=1.0,
                codebook_rampup_multiplier=3.0,
                coderampup_steps=2000,
            )
        self.quantizer_options = quantizer_options or default_quant_opts

        # * perceptual loss
        if perceptual_weight > 0:
            if perceptual_type == "vgg":
                self.perceptual_loss = LPIPS(net="vgg").cuda().eval()
            else:
                raise ValueError(f"Unknown perceptual loss '{perceptual_type}'.")
        self.perceptual_weight = perceptual_weight

        # * LeCAM ema loss
        self.gen_loss_weight = gen_loss_weight
        self.lecam_loss_weight = lecam_loss_weight
        if self.lecam_loss_weight is not None:
            self.lecam_ema = LeCAM_EMA()

        # * discriminator
        if disc_network_type.lower() == "patchgan":
            self.discriminator = NLayerDiscriminator(
                input_nc=disc_in_channels,
                n_layers=disc_num_layers,
                use_actnorm=use_actnorm,
                ndf=disc_ndf,
            ).apply(weights_init)
        elif disc_network_type.lower() == "stylegan":
            self.discriminator = StyleGANDiscriminator(
                0,
                disc_input_size,
                disc_in_channels,
                num_fp16_res=8
                if AcceleratorState().mixed_precision == "bf16"
                else 0,  # 8 is sufficiently large to cover all res
                epilogue_kwargs={"mbstd_group_size": 4},
            )
        elif disc_network_type.lower() == "stylegan3d":
            self.discriminator = StyleGAN3DDiscriminator(
                num_frames,
                disc_input_size,
                video_channels=disc_in_channels,
            )
        else:
            raise ValueError(f"Unsupported discriminator type: {disc_network_type}")

        # * disc loss
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        elif disc_loss == "non_saturate":
            self.disc_loss = non_saturate_discriminator_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")

        # * SSIM loss
        if loss_ssim:
            self.ssim_loss = SSIMLoss(window_size=11)
            print("SSIM Loss is used in VAE losses")

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(
                nll_loss, self.last_layer[0], retain_graph=True
            )[0]
            g_grads = torch.autograd.grad(
                g_loss, self.last_layer[0], retain_graph=True
            )[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        optimizer_idx: int,
        global_step: int,
        q_loss_breakdown: namedtuple | dict | None = None,
        last_layer: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
        split: str = "train",
        add_prefix: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | float]]:
        """
        Forward pass of the model.

        outputs:
            1. dict of loss: when optimizer_idx=0, generator part
                outdict has key `gen_loss` and `q_loss`;
                when optimizer_idx=1, discriminator part
                outdict has key `disc_loss`
            2. logs: dict[str, Tensor], logs for logging the generator and discriminator
        """
        # input shapes
        if inputs.ndim == 5:
            assert (
                self.num_frames == inputs.shape[2]
            ), f"Number of frames does not match input "
            inputs = rearrange(inputs, "n c t h w -> (n t) c h w")
            reconstructions = rearrange(reconstructions, "n c t h w -> (n t) c h w")

        # * ==========================================================
        # * GAN loss

        # now the GAN part
        if optimizer_idx == 0:
            return self.gen_loss(
                inputs=inputs,
                last_layer=last_layer,
                global_step=global_step,
                q_loss_breakdown=q_loss_breakdown,
                split=split,
                add_prefix=add_prefix,
                cond=cond,
            )

        # * ==========================================================
        # * discriminator losses

        elif optimizer_idx == 1:
            self.disc_loss(
                inputs=inputs,
                reconstructions=reconstructions,
                global_step=global_step,
                cond=cond,
                split=split,
            )

        else:
            raise ValueError(f"{optimizer_idx=} is invalid")

    def gen_loss_weight(self, nll_loss, g_loss, last_layer):
        if self.gen_loss_weight is None:
            try:
                d_weight = self.calculate_adaptive_weight(
                    nll_loss, g_loss, last_layer=last_layer
                )
            except RuntimeError:
                assert not self.training
                logger.warning("d_weight is set to 0")
                d_weight = torch.tensor(0.0)
        else:
            d_weight = torch.tensor(self.gen_loss_weight)

        return d_weight

    def q_loss(
        self,
        q_loss_breakdown: namedtuple | dict,
        global_step: int,
    ) -> tuple[torch.Tensor, tuple[str, torch.Tensor]]:
        if isinstance(q_loss_breakdown, dict):
            q_loss_breakdown = dict_to_namedtuple(q_loss_breakdown)

        if q_loss_breakdown is None:
            return torch.tensor(0.0).cuda(), dict()  # None dict info

        q_loss = torch.tensor(0.0).cuda()
        logs = {}

        if self.quantizer_type == "vq":
            codebook_loss = q_loss_breakdown.codebook_loss

            scale_codebook_loss = self.codebook_weight * codebook_loss  # entropy_loss
            if self.codebook_enlarge_ratio > 0:
                scale_codebook_loss = (
                    self.codebook_enlarge_ratio
                    * (max(0, 1 - global_step / self.codebook_enlarge_steps))
                    * scale_codebook_loss
                    + scale_codebook_loss
                )
            q_loss = scale_codebook_loss
            logs = {
                "codebook_loss": scale_codebook_loss,
                "commit_loss": q_loss_breakdown.commitment.detach(),
            }

            maybe_in_dict_update(q_loss_breakdown, ["H"], logs)

        elif self.quantizer_type == "bsq":
            q_loss = q_loss_breakdown.total_loss
            logs = {
                "commit_loss": q_loss_breakdown.commit_loss,
                "entropy": q_loss_breakdown.H,
                "avg_prob": q_loss_breakdown.avg_prob,
            }

        return q_loss, logs

    def train_disc_log_form(
        self,
        split: str,
        disc_factor: float,
        # losses =======
        disc_loss: torch.Tensor,
        lecam_loss: torch.Tensor,
        non_saturate_d_loss: torch.Tensor,
        # logits =======
        logits_real: torch.Tensor,
        logits_fake: torch.Tensor,
        # r1 loss ===
        r1_scale: torch.Tensor | None = None,
        r1_loss: torch.Tensor | None = None,
    ):
        if disc_factor == 0:
            logs = {
                "disc_loss": torch.tensor(0.0),
                "logits_real": torch.tensor(0.0),
                "logits_fake": torch.tensor(0.0),
                "disc_factor": torch.tensor(disc_factor),
                "lecam_loss": lecam_loss.detach(),
                "non_saturated_d_loss": non_saturate_d_loss.detach(),
            }
        else:
            logs = {
                "disc_loss": disc_loss.clone().detach().mean(),
                "logits_real": logits_real.detach().mean(),
                "logits_fake": logits_fake.detach().mean(),
                "disc_factor": torch.tensor(disc_factor),
                "lecam_loss": lecam_loss.detach(),
                "non_saturated_d_loss": non_saturate_d_loss.detach(),
            }

        # r1 regularization
        if r1_loss is not None and r1_scale is not None:
            logs.update(
                {
                    "disc_r1_loss": r1_loss.detach().mean(),
                    "disc_r1_loss_scale": r1_scale.detach().mean(),
                }
            )

        logs = dict_add_prefix(logs, split)

        return logs

    def train_generator_log_form(
        self,
        # infos =======
        disc_factor: float,
        split: str,
        # losses =======
        total_loss: torch.Tensor,
        nll_loss: torch.Tensor,
        reconstruction_loss: torch.Tensor,
        gen_loss: torch.Tensor,
        ssim_loss: torch.Tensor | None = None,
        percep_loss: torch.Tensor | None = None,
        real_g_loss: torch.Tensor | None = None,
        # weights ======
        disc_weight: torch.Tensor | None = None,
        # other =======
        # from quantizer
        quantizer_logs: dict | None = None,
        add_prefix: bool = False,
    ):
        if disc_factor == 0:
            log = {
                "total_loss": total_loss.clone().detach(),
                # "per_sample_entropy": loss_break.per_sample_entropy.detach(),
                # "codebook_entropy": loss_break.batch_entropy.detach(),
                # "commit_loss": loss_break.commitment.detach(),
                "nll_loss": nll_loss.detach(),
                "reconstruct_loss": reconstruction_loss.detach().mean(),
                "ssim_loss": ssim_loss.detach().mean(),
                "perceptual_loss": percep_loss.detach().mean(),
                "d_weight": torch.tensor(0.0),
                "disc_factor": torch.tensor(0.0),
                "g_loss": torch.tensor(0.0),
            }
        else:
            if self.training:
                log = {
                    "total_loss": total_loss.clone().detach(),
                    # losses of LFQ ===========================================
                    # "per_sample_entropy": loss_break.per_sample_entropy.detach(),
                    # "codebook_entropy": loss_break.batch_entropy.detach(),
                    # "commit_loss": loss_break.commitment.detach(),
                    # "entropy_loss": codebook_loss.detach(),
                    # image losses ===========================================
                    "nll_loss": nll_loss.detach(),
                    "reconstruct_loss": reconstruction_loss.detach().mean(),
                    "perceptual_loss": percep_loss.detach().mean(),
                    "ssim_loss": ssim_loss.detach().mean(),
                    # discriminator loss ===========================================
                    "d_weight": disc_weight,
                    "disc_factor": torch.tensor(disc_factor),
                    # generator loss ===========================================
                    "g_loss": gen_loss.detach(),
                }
            else:
                # validation only monitor the reconstruct_loss and p_loss
                assert real_g_loss is not None, "real_g_loss should not be None"

                log = {
                    "reconstruct_loss": reconstruction_loss.detach().mean(),
                    "perceptual_loss": percep_loss.detach().mean(),
                    "g_loss": real_g_loss.detach(),
                }
        if quantizer_logs is not None:
            log.update(quantizer_logs)

        log = add_prefix(log, split)

        return log

    def gen_loss(
        self,
        inputs,
        last_layer,
        global_step,
        q_loss_breakdown,
        split: str,
        add_prefix,
        cond=None,
    ):
        # generator update
        if self.disc_network_type == "stylegan3d":
            reconstructions = rearrange(
                reconstructions, "(n t) c h w -> n c t h w", t=self.num_frames
            )

        # * generator update
        rec_loss = torch.abs(input=inputs.contiguous() - reconstructions.contiguous())

        # * ssim loss
        if self.with_ssim:
            ssim_loss = (
                self.ssim_loss((inputs + 1) / 2, (reconstructions + 1) / 2)
                * self.ssim_weight
            )
        else:
            ssim_loss = torch.tensor(0.0).to(inputs)

        # * perceptual loss
        nll_loss = rec_loss  # .clone()
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs, reconstructions)
            nll_loss = nll_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])
        nll_loss = torch.mean(nll_loss)

        # * (un)conditional gan loss
        if cond is None:
            assert not self.disc_conditional
            logits_fake = self.discriminator(reconstructions.contiguous())
        else:
            assert self.disc_conditional
            logits_fake = self.discriminator(
                torch.cat((reconstructions.contiguous(), cond), dim=1)
            )
        g_loss = non_saturate_gen_loss(logits_fake)

        # * g loss weight
        d_weight = self.gen_loss_weight(nll_loss, g_loss, last_layer)

        disc_factor = adopt_weight(
            self.disc_factor, global_step, threshold=self.discriminator_iter_start
        )
        if not self.training:
            real_g_loss = disc_factor * g_loss

        g_loss = d_weight * disc_factor * g_loss

        # * nll_loss (L1 + LPIPS) + codebook_loss (entropy_loss) + commitment_loss + g_loss (GAN Loss)
        # basic losses
        loss = (
            nll_loss + ssim_loss + g_loss
        )  # + scale_codebook_loss + loss_break.commitment * self.commit_weight

        # * quantization losses
        q_loss, q_loss_logs = self.q_loss(q_loss_breakdown, global_step)

        # * form logs
        log = self.train_generator_log_form(
            disc_factor=disc_factor,
            split=split,
            total_loss=loss,
            nll_loss=nll_loss,
            reconstruction_loss=rec_loss,
            gen_loss=g_loss,
            ssim_loss=ssim_loss,
            percep_loss=p_loss,
            real_g_loss=real_g_loss if not self.training else None,
            disc_weight=d_weight,
            quantizer_logs=q_loss_logs,
            add_prefix=add_prefix,
        )

        gen_loss_for_bkwd = dict(
            gen_loss=loss,
            q_loss=q_loss,
        )
        return gen_loss_for_bkwd, log

    def disc_loss(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        global_step: int,
        cond: torch.Tensor | None = None,
        split: str = "train",
    ):
        # * discrimator loss
        if self.disc_network_type == "stylegan3d":
            inputs = rearrange(inputs, "(n t) c h w -> n c t h w", t=self.num_frames)
            reconstructions = rearrange(
                reconstructions, "(n t) c h w -> n c t h w", t=self.num_frames
            )

        # second pass for discriminator update
        if cond is None:
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())
        else:
            logits_real = self.discriminator(
                torch.cat((inputs.contiguous().detach(), cond), dim=1)
            )
            logits_fake = self.discriminator(
                torch.cat((reconstructions.contiguous().detach(), cond), dim=1)
            )

        disc_factor = adopt_weight(
            self.disc_factor, global_step, threshold=self.discriminator_iter_start
        )

        # ---------------------------------------------------------------------------------------
        # Non-Saturate Loss is the Format of GAN Training, for D Loss, We still adopt Hinge Loss
        # ---------------------------------------------------------------------------------------
        if self.lecam_loss_weight is not None:
            self.lecam_ema.update(logits_real, logits_fake)
            lecam_loss = lecam_reg(logits_real, logits_fake, self.lecam_ema)
            non_saturate_d_loss = self.disc_loss(logits_real, logits_fake)  # hinge loss
            d_loss = disc_factor * (
                lecam_loss * self.lecam_loss_weight + non_saturate_d_loss
            )
        else:
            lecam_loss = torch.tensor(0.0)
            non_saturate_d_loss = self.disc_loss(logits_real, logits_fake)
            d_loss = disc_factor * non_saturate_d_loss

        # r1 regularization loss
        # for stablized training
        if self.disc_reg_freq > 0 and (global_step + 1) % self.disc_reg_freq == 0:
            inputs.requires_grad = True
            logits_real = self.discriminator(inputs.contiguous())
            r1_loss = d_r1_loss(logits_real, inputs)
            r1_loss_scale = self.disc_reg_r1 / 2 * r1_loss * self.disc_reg_freq
            d_loss = d_loss + r1_loss_scale  # changed d_loss
        else:
            r1_loss = torch.tensor(0.0, device=self.device)
            r1_loss_scale = torch.tensor(0.0, device=self.device)

        log = self.train_disc_log_form(
            split=split,
            disc_factor=disc_factor,
            non_saturate_d_loss=non_saturate_d_loss,
            lecam_loss=lecam_loss,
            disc_loss=d_loss,
            logits_real=logits_real,
            logits_fake=logits_fake,
            r1_scale=r1_loss_scale,
            r1_loss=r1_loss,
        )
        disc_loss = dict(disc_loss=d_loss)

        return disc_loss, log

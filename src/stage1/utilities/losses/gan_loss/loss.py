import warnings
from collections import namedtuple
from typing import Dict, NamedTuple

import omegaconf
import torch
import torch.distributed.tensor as dtensor
import torch.nn as nn
import torch.nn.functional as F
from accelerate.state import AcceleratorState, PartialState
from einops import rearrange
from kornia.losses import SSIMLoss
from loguru import logger
from pytorch_wavelets import DWTForward

from src.utilities.config_utils import to_object

from ..model import (
    DinoDiscV2,
    NLayerDiscriminator,
    NLayerDiscriminatorv2,
    StyleGAN3DDiscriminator,
    StyleGANDiscriminator,
    no_weight_gradients,
)
from ..repa import REPALoss, VFLoss
from .hyperspectral_percep_loss import LIPIPSHyperpspectral


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
    q_info_dict: Dict | NamedTuple,
    may_used_keys: list[str],
    saved_dict: dict,
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
    B = logits_fake.shape[0]
    logits_fake = logits_fake.reshape(B, -1)
    logits_fake = torch.mean(logits_fake, dim=-1)

    gen_loss = torch.mean(
        _sigmoid_cross_entropy_with_logits(
            labels=torch.ones_like(logits_fake), logits=logits_fake
        )
    )

    return gen_loss


def non_saturate_discriminator_loss(logits_real, logits_fake):
    B = logits_fake.shape[0]
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
        disc_start_for_g: int = 0,
        disc_start_for_d: int = 0,
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
        disc_norm_type: str = "bn2d",
        disc_spectral_norm: bool = False,
        disc_conditional: bool = False,
        disc_ndf: int = 64,
        disc_loss: str = "hinge",
        force_disc_loss_hinge: bool = False,  # adopted from maskbit paper
        # codebook losses
        quantizer_options: dict | None = None,
        # perceptual loss
        perceptual_weight: float = 1.0,
        perceptual_type: str | None = "resnet",
        perceptual_groups_to_select: int | float | None = None,  # group on all channels
        perceptual_loss_on_logits: bool = False,
        gram_model: str | None = "vgg",
        gram_loss_weight: float = 1.0,
        img_is_neg1_to_1: bool = True,
        perceptual_options: dict = {},
        # generator loss
        reconstruction_loss_type: str | None = "l1",
        reconstruction_weight: float = 1.0,
        gen_loss_weight: float | None = None,
        # quantizer losses
        quantizer_type: str | None = None,
        # repa loss
        repa_loss_weight: float | None = None,
        repa_loss_options: dict = {},
        # vf loss
        vf_loss_weight: float | None = None,
        vf_loss_options: dict = {},
        # other losses
        lecam_loss_weight: float | None = None,
        ssim_weight: float = 0.1,
        # if is video
        num_frames: int = 1,
        # not reconstruction loss if using diffusion slots
        force_not_use_recon_loss: bool = False,
    ):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla", "non_saturate"]
        assert quantizer_type in ["lfq", "bsq", "vq", "vq_advance", "kl", None]
        assert disc_network_type in [
            "patchgan",
            "patchgan_v2",
            "stylegan",
            "stylegan3d",
        ]
        if force_not_use_recon_loss:
            assert reconstruction_loss_type in ["l1", "mse", "dwt"]

        # state
        self.device = PartialState().device
        self.dtype = {
            "fp16": torch.float16,
            "no": torch.float32,
            "bf16": torch.bfloat16,
        }[AcceleratorState().mixed_precision]

        self.reconstruction_weight = reconstruction_weight
        self.quantizer_type = quantizer_type
        self.disc_reg_freq = disc_reg_freq
        self.disc_reg_r1 = disc_reg_r1
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_network_type = disc_network_type
        self.disc_conditional = disc_conditional
        self.use_ssim = ssim_weight > 0.0
        self.num_frames = num_frames
        self.reconstruction_loss_type = reconstruction_loss_type
        self.force_not_use_recon_loss = force_not_use_recon_loss
        if force_not_use_recon_loss:
            logger.warning(
                "[VQ fn loss]: not use reconstruction loss, "
                "make sure you will compute this main loss elsewhere"
            )

        # * if is dwt reocn loss
        if self.reconstruction_loss_type == "dwt":
            self.dwt = DWTForward(J=1, mode="zero", wave="haar").to(
                self.device, self.dtype
            )

        # * quantizer options
        if quantizer_type == "vq":
            default_quant_opts = dict(
                commit_weight=0.25,
                codebook_weight=1.0,
                codebook_enlarge_ratio=3,
                codebook_enlarge_steps=2000,
            )
        elif quantizer_type == "vq_advance":
            default_quant_opts = dict(quantizer_loss_weight=0.1)
        # TODO: add ibq here
        elif quantizer_type == "ibq":
            raise NotImplementedError("ibq loss is not implemented")
        elif quantizer_type in ("bsq", "lfq"):
            default_quant_opts = dict(
                quantizer_loss_weight=0.1,
                codebook_enlarge_ratio=3,
                codebook_enlarge_steps=2000,
            )
        elif quantizer_type == "kl":
            default_quant_opts = dict(kl_weight=1e-6)
        else:
            default_quant_opts = None
        logger.info(f"[VQ fn loss]: use quantizer type={quantizer_type}")
        self.quantizer_options = quantizer_options or default_quant_opts
        self.vq_options_check()

        # * perceptual loss
        self.use_perceptual_loss = False
        if perceptual_weight > 0 and perceptual_type is not None:
            self.use_perceptual_loss = True
            self.perceptual_loss = LIPIPSHyperpspectral(
                perceptual_type,
                group_size=3,
                num_groups_to_select=perceptual_groups_to_select,
                padding_mode="repeat",
                compute_on_logits=perceptual_loss_on_logits,
                img_is_neg1_to_1=img_is_neg1_to_1,
                gram_loss_weight=gram_loss_weight,
                use_gram_model=gram_model,
                **perceptual_options,
            ).cuda()
        self.perceptual_weight = perceptual_weight

        # * repa loss
        self.repa_loss_weight = repa_loss_weight
        self.use_repa = False
        if repa_loss_weight is not None and repa_loss_weight > 0:
            self.use_repa = True
            self.repa_loss = REPALoss(**to_object(repa_loss_options)).cuda()
            logger.info(f"[repa loss]: {self.repa_loss}")
            logger.info(f"[vq loss]: repa loss used, weighted {self.repa_loss_weight}")

        # * visual foudation loss
        self.vf_loss_weight = vf_loss_weight
        self.use_vf = False
        if vf_loss_weight is not None and vf_loss_weight > 0:
            self.use_vf = True
            assert not self.use_repa, (
                "repa loss and vf loss can not be used at the same time"
            )
            self.vf_loss = VFLoss(**to_object(vf_loss_options)).cuda()
            logger.info(f"[vq loss]: vf loss used, weighted {self.vf_loss_weight}")

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
            )
            # spectral norm
            if disc_spectral_norm:
                for layer in self.discriminator.modules():
                    if isinstance(layer, (nn.Conv2d, nn.Linear)):
                        nn.utils.spectral_norm(layer)
        elif disc_network_type.lower() == "patchgan_v2":  # from maskbit paper
            self.discriminator = NLayerDiscriminatorv2(
                num_channels=disc_in_channels,
                hidden_channels=disc_ndf,
                num_stages=disc_num_layers,
                # Zihan Note: gn underperforms than bn, and cause the adversarial
                # training unstable
                norm_type=disc_norm_type,
                # as suggested in the original paper
                blur_kernel_size=4,
                blur_resample=True,
            )
        elif disc_network_type.lower() == "stylegan":
            self.discriminator = StyleGANDiscriminator(
                0,
                disc_input_size,
                disc_in_channels,
                num_fp16_res=8
                if AcceleratorState().mixed_precision == "bf16"
                else 0,  # 8 is sufficiently large to cover all res
                epilogue_kwargs={"mbstd_group_size": 3},
            )
        elif disc_network_type.lower() == "stylegan3d":
            self.discriminator = StyleGAN3DDiscriminator(
                num_frames,
                disc_input_size,
                video_channels=disc_in_channels,
            )
        else:
            raise ValueError(f"Unsupported discriminator type: {disc_network_type}")

        # * disc lossdisc_start_for_g
        self.disc_iter_start_for_g = disc_start_for_g
        self.disc_iter_start_for_d = disc_start_for_d
        if disc_loss == "hinge":
            self.discriminator_loss = hinge_d_loss
            self.generator_loss = hinge_g_loss
        elif disc_loss == "vanilla":
            self.discriminator_loss = (
                vanilla_d_loss if not force_disc_loss_hinge else hinge_d_loss
            )
            self.generator_loss = vanilla_g_loss
        elif disc_loss == "non_saturate":
            self.discriminator_loss = non_saturate_discriminator_loss
            self.generator_loss = non_saturate_gen_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        logger.info(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")

        # * SSIM loss
        self.ssim_weight = ssim_weight
        if ssim_weight > 0:
            self.ssim_loss = SSIMLoss(window_size=11)
            logger.info("SSIM Loss is used in VAE losses")

        # assertions
        self.can_not_comp_adp_loss = (
            self.force_not_use_recon_loss and self.perceptual_weight < 0.0
        )
        if self.can_not_comp_adp_loss and self.gen_loss_weight is None:
            raise ValueError(
                "can not compute adaptive loss weight when ",
                "force_not_use_recon_loss=True and perceptual_weight<0.0",
            )

        # zero buffer
        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            # TODO: add fsdp2 support
            if torch.is_tensor(last_layer) and not isinstance(
                last_layer, dtensor.DTensor
            ):
                with torch.autocast("cuda", self.dtype):
                    nll_grads = torch.autograd.grad(
                        nll_loss,
                        last_layer,
                        retain_graph=True,
                    )[0]
                    g_grads = torch.autograd.grad(
                        g_loss,  # .to(last_layer.dtype),
                        last_layer,
                        retain_graph=True,
                    )[0]
            elif isinstance(last_layer, dtensor.DTensor):
                # is fsdp2 DTensor
                # get the last sharded layer parameters
                last_layer_full = last_layer.full_tensor()
                with torch.autocast("cuda", self.dtype):
                    nll_grads = torch.autograd.grad(
                        nll_loss, last_layer_full, retain_graph=True
                    )[0]
                    g_grads = torch.autograd.grad(
                        g_loss, last_layer_full, retain_graph=True
                    )[0]
            else:
                raise ValueError(
                    f"Adaptive weighting can not be calculated for the last layer {last_layer}"
                )
        else:
            raise ValueError("last_layer is not defined.")

            # !!! can not reach this code !!!
            nll_grads = torch.autograd.grad(
                nll_loss, self.last_layer[0], retain_graph=True
            )[0]
            g_grads = torch.autograd.grad(
                g_loss, self.last_layer[0], retain_graph=True
            )[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight

    def forward(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        optimizer_idx: int,
        global_step: int,
        q_loss_breakdown: NamedTuple | Dict | None = None,
        q_loss_total: torch.Tensor | None = None,
        outer_recon_loss: torch.Tensor | None = None,
        last_layer: nn.Parameter | torch.Tensor | None = None,
        enc_last_layer: nn.Parameter | None = None,
        cond: torch.Tensor | None = None,
        tokenizer_feat: torch.Tensor | None = None,
        split: str = "train",  # TODO: remove this
        add_prefix: bool = False,
    ):
        """
        Forward pass of the model.

        outputs:
            1. dict of loss: when optimizer_idx=0, generator part
                outdict has key `gen_loss` and `q_loss`;
                when optimizer_idx=1, discriminator part
                outdict has key `disc_loss`
            2. logs: dict[str, Tensor], logs for logging the generator and discriminator
        """
        if split != "train":
            # not use `split`
            warnings.warn(
                "split is not used, we will remove this argument", DeprecationWarning
            )

        # input shapes
        if inputs.ndim == 5:
            assert self.num_frames == inputs.shape[2], (
                f"Number of frames does not match input "
            )
            inputs = rearrange(inputs, "n c t h w -> (n t) c h w")
            reconstructions = rearrange(reconstructions, "n c t h w -> (n t) c h w")

        assert (q_loss_total is None and q_loss_breakdown is None) or (
            q_loss_total is not None and q_loss_breakdown is not None
        ), "q_loss_total and q_loss_breakdown must be both None or both not None"

        # * ==========================================================
        # * GAN loss

        # now the GAN part
        if optimizer_idx == 0:
            return self.gen_loss(
                inputs=inputs,
                reconstructions=reconstructions,
                last_layer=last_layer,
                global_step=global_step,
                q_loss_breakdown=q_loss_breakdown,
                split=split,
                add_prefix=add_prefix,
                cond=cond,
                q_loss_total=q_loss_total,
                outer_recon_loss=outer_recon_loss,
                tokenizer_feat=tokenizer_feat,
                enc_last_layer=enc_last_layer,
            )

        # * ==========================================================
        # * discriminator losses

        elif optimizer_idx == 1:
            return self.disc_loss(
                inputs=inputs,
                reconstructions=reconstructions,
                global_step=global_step,
                cond=cond,
                split=split,
                add_prefix=add_prefix,
            )

        else:
            raise ValueError(f"{optimizer_idx=} is invalid")

    def gen_loss_weight_fn(self, nll_loss, g_loss, last_layer):
        if self.gen_loss_weight is None:
            # try:
            #     d_weight = self.calculate_adaptive_weight(
            #         nll_loss, g_loss, last_layer=last_layer
            #     )
            # except RuntimeError as e:
            #     logger.error(f"try to calculate adaptive weight, but met error: {e}")
            #     assert not self.training
            #     logger.warning("d_weight is set to 0")
            #     d_weight = self.zero.to(nll_loss.device)

            d_weight = self.calculate_adaptive_weight(
                nll_loss, g_loss, last_layer=last_layer
            )
        else:
            d_weight = torch.tensor(self.gen_loss_weight).to(nll_loss.device)

        return d_weight

    def q_loss(
        self,
        q_loss_total: torch.Tensor,
        q_loss_breakdown: NamedTuple | Dict,
        global_step: int,
    ) -> tuple[torch.Tensor, tuple[str, torch.Tensor]]:
        if isinstance(q_loss_breakdown, dict):
            q_loss_breakdown = dict_to_namedtuple(q_loss_breakdown)

        if q_loss_breakdown is None or q_loss_total is None:
            return self.zero, dict()  # None dict info

        q_loss = self.zero
        logs = {}

        # q loss enlarge
        def _enlarge_codebook_loss_fn(codebook_loss):
            cb_enlarge_r = self.quantizer_options["codebook_enlarge_ratio"]
            codebook_enlarge_steps = self.quantizer_options["codebook_enlarge_steps"]
            cb_enlarge_on_loss = cb_enlarge_r * (
                max(0, 1 - global_step / codebook_enlarge_steps)
            )

            scale_codebook_loss = (
                self.quantizer_options["codebook_weight"] * codebook_loss
            )  # entropy_loss
            if cb_enlarge_r > 0:
                scale_codebook_loss = (
                    cb_enlarge_on_loss * scale_codebook_loss + scale_codebook_loss
                )

            return scale_codebook_loss

        # * vector quantization ===============
        if self.quantizer_type == "vq":
            codebook_loss = q_loss_breakdown.codebook_loss
            scale_codebook_loss = _enlarge_codebook_loss_fn(codebook_loss)

            # for logs
            q_commit_loss_weighted = (
                q_loss_breakdown.commitment * self.quantizer_options["commit_weight"]
            )
            q_loss = scale_codebook_loss + q_commit_loss_weighted
            logs = {
                "q_loss": q_loss,
                "codebook_loss": scale_codebook_loss,
                "commit_loss": q_commit_loss_weighted,
            }

            maybe_in_dict_update(q_loss_breakdown, ["H"], logs)

        # * vq with codebook diversity loss, orthogonal reg loss, codebook optimization loss ====
        elif self.quantizer_type == "vq_advance":
            q_loss = q_loss_total * self.quantizer_options["quantizer_loss_weight"]
            logs = {
                "q_loss": q_loss,
                "commitment": q_loss_breakdown.commitment,
                "code_diversity": q_loss_breakdown.codebook_diversity,
                "orthogonal_reg": q_loss_breakdown.orthogonal_reg,
                "learn_code_opt_loss": q_loss_breakdown.inplace_optimize,
            }

        # * lfq or bsq ===============
        elif self.quantizer_type in ("lfq", "bsq"):
            q_loss = q_loss_total * self.quantizer_options["quantizer_loss_weight"]
            cb_enlarge_r = self.quantizer_options["codebook_enlarge_ratio"]
            codebook_enlarge_steps = self.quantizer_options["codebook_enlarge_steps"]
            if codebook_enlarge_steps > 0:
                cb_enlarge_on_loss = cb_enlarge_r * (
                    max(0, 1 - global_step / codebook_enlarge_steps)
                )
            else:
                cb_enlarge_on_loss = 0.0

            if cb_enlarge_r > 0:
                q_loss = cb_enlarge_on_loss * q_loss + q_loss

            # logs
            if self.quantizer_type == "bsq":
                logs = {
                    "q_loss": q_loss,
                    "commit_loss": q_loss_breakdown.commit_loss,
                    "entropy": q_loss_breakdown.H,
                    "avg_prob": q_loss_breakdown.avg_prob,
                }
            elif self.quantizer_type == "lfq":
                logs = {
                    "q_loss": q_loss,
                    "commit_loss": q_loss_breakdown.commitment,
                    "batch_entropy": q_loss_breakdown.batch_entropy,
                    "per_sample_entropy": q_loss_breakdown.per_sample_entropy,
                }

        # * kl =============
        elif self.quantizer_type == "kl":
            q_loss = q_loss_total * self.quantizer_options["kl_weight"]
            logs = {
                "kl_loss": q_loss,
            }

        return q_loss, logs

    def vq_options_check(self):
        def _assert_in_dict(names: list[str], d: dict):
            for name in names:
                assert name in d, f"{name} not in {d}"

        if self.quantizer_type == "vq":
            _assert_in_dict(
                [
                    "commit_weight",
                    "codebook_enlarge_steps",
                    "codebook_enlarge_ratio",
                ],
                self.quantizer_options,
            )
        elif self.quantizer_type == "bsq":
            _assert_in_dict(
                [
                    "quantizer_loss_weight",
                    "codebook_enlarge_steps",
                    "codebook_enlarge_ratio",
                ],
                self.quantizer_options,
            )
        elif self.quantizer_type == "vq_adavance":
            _assert_in_dict(
                ["quantizer_loss_weight"],
                self.quantizer_options,
            )
        elif self.quantizer_type == "kl":
            _assert_in_dict(
                ["kl_weight"],
                self.quantizer_options,
            )

        logger.info(
            f"[VQ fn loss]: quantizer options: {self.quantizer_options}"
            f"for quantizer type: {self.quantizer_type}"
        )

    def train_disc_log_form(
        self,
        split: str,
        disc_factor: float,
        # losses =======
        disc_loss: torch.Tensor,
        lecam_loss: torch.Tensor,
        # logits =======
        logits_real: torch.Tensor,
        logits_fake: torch.Tensor,
        # r1 loss ===
        r1_scale: torch.Tensor | None = None,
        r1_loss: torch.Tensor | None = None,
        add_prefix: bool = False,
    ):
        if disc_factor == 0:
            logs = {
                "disc_loss": self.zero,
                "logits_real": self.zero,
                "logits_fake": self.zero,
                "disc_factor": torch.tensor(disc_factor),
                "lecam_loss": lecam_loss.detach(),
            }
        else:
            logs = {
                "disc_loss": disc_loss.clone().detach().mean(),
                "logits_real": logits_real.detach().mean(),
                "logits_fake": logits_fake.detach().mean(),
                "disc_factor": torch.tensor(disc_factor),
                "lecam_loss": lecam_loss.detach(),
            }

        # r1 regularization
        if r1_loss is not None and r1_scale is not None:
            logs.update(
                {
                    "r1_loss": r1_loss.detach().mean(),
                    "r1_scale": r1_scale.detach().mean(),
                }
            )
        if add_prefix:
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
        gram_loss: torch.Tensor | None = None,
        real_g_loss: torch.Tensor | None = None,
        repa_loss: torch.Tensor | None = None,
        vf_loss: torch.Tensor | None = None,
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
                "nll_loss": nll_loss.detach(),
                "reconstruct_loss": reconstruction_loss.detach().mean(),
                "ssim_loss": ssim_loss.detach().mean(),
                "perceptual_loss": percep_loss.detach().mean(),
                "repa_loss": repa_loss.detach().mean(),
                "vf_loss": vf_loss.detach().mean(),
                "gram_loss": gram_loss.detach().mean(),
                "d_weight": self.zero,
                "disc_factor": self.zero,
                "g_loss": self.zero,
            }
        else:
            if self.training:
                log = {
                    "total_loss": total_loss.clone().detach(),
                    # image losses
                    "nll_loss": nll_loss.detach(),
                    "reconstruct_loss": reconstruction_loss.detach().mean(),
                    "perceptual_loss": percep_loss.detach().mean(),
                    "gram_loss": gram_loss.detach().mean(),
                    "ssim_loss": ssim_loss.detach().mean(),
                    "repa_loss": repa_loss.detach().mean(),
                    "vf_loss": vf_loss.detach().mean(),
                    # discriminator loss
                    "d_weight": disc_weight,
                    "disc_factor": torch.tensor(disc_factor),
                    # generator loss
                    "g_loss": gen_loss.detach(),
                }
            else:
                # validation only monitor the reconstruct_loss and p_loss
                assert real_g_loss is not None, "real_g_loss should not be None"

                log = {
                    "reconstruct_loss": reconstruction_loss.detach().mean(),
                    "perceptual_loss": percep_loss.detach().mean(),
                    "gram_loss": gram_loss.detach().mean(),
                    "g_loss": real_g_loss.detach(),
                }

        # * qunatizer loss logs ==========
        if quantizer_logs is not None:
            log.update(quantizer_logs)

        if add_prefix:
            log = dict_add_prefix(log, split)

        return log

    def reconstruction_loss(self, inputs: torch.Tensor, targets: torch.Tensor):
        # recon loss
        if self.reconstruction_loss_type == "mse":
            recon_loss = F.mse_loss(inputs, targets)
        elif self.reconstruction_loss_type == "l1":
            recon_loss = F.l1_loss(inputs, targets)
        elif self.reconstruction_loss_type == "dwt":
            # Haar dwt compuation
            assert hasattr(self, "dwt"), "dwt mode not found"
            with torch.autocast(device_type="cuda"):
                inp_l, inp_h = self.dwt(inputs)
                tgt_l, tgt_h = self.dwt(targets)

            # cat along channels
            # inp_dwt = torch.cat([inp_l, *(inp_h[0].unbind(dim=2))], dim=1)
            # tgt_dwt = torch.cat([tgt_l, *(tgt_h[0].unbind(dim=2))], dim=1)

            # loss
            # recon_loss = (inp_dwt - tgt_dwt).pow(2).mean()
            # recon_loss = F.l1_loss(inp_dwt, tgt_dwt)
            # * assign different weights on low and high-freq components
            low_dwt_loss = F.l1_loss(inp_l, tgt_l) * 0.25
            _to_tensor = lambda tensor_lst: torch.stack(tensor_lst, dim=0)
            high_dwt_loss = F.l1_loss(_to_tensor(inp_h), _to_tensor(tgt_h)) * 0.75
            recon_loss = low_dwt_loss + high_dwt_loss
        else:
            raise NotImplementedError(
                f"Reconstruction loss type {self.recon_loss_type} not implemented."
            )

        if self.ssim_weight > 0.0 and self.use_ssim:
            ssim_loss = self.ssim_loss(inputs, targets)
        else:
            ssim_loss = self.zero

        return dict(recon_loss=recon_loss, ssim_loss=ssim_loss)

    def gen_loss(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        last_layer: nn.Parameter | None,
        global_step: int,
        q_loss_breakdown: NamedTuple | Dict | None,
        split: str,
        add_prefix: bool,
        cond: torch.Tensor | None = None,
        q_loss_total: torch.Tensor | None = None,
        outer_recon_loss: torch.Tensor | None = None,
        tokenizer_feat: torch.Tensor
        | None = None,  # repa projected or z (latent) vf projected
        enc_last_layer: nn.Parameter | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | float]]:
        # generator update
        if self.disc_network_type == "stylegan3d":
            reconstructions = rearrange(
                reconstructions, "(n t) c h w -> n c t h w", t=self.num_frames
            )

        # * construction loss
        if self.force_not_use_recon_loss:
            assert outer_recon_loss is not None, "outer_recon_loss is None"
            recon_loss = outer_recon_loss
            ssim_loss = self.zero
        else:
            recon_loss_d = self.reconstruction_loss(inputs, reconstructions)
            recon_loss = recon_loss_d["recon_loss"] * self.reconstruction_weight
            ssim_loss = recon_loss_d["ssim_loss"] * self.ssim_weight

        # * perceptual loss
        nll_loss = recon_loss + ssim_loss
        p_loss = self.zero
        gram_loss = self.zero
        if self.use_perceptual_loss:
            p_loss_dict = self.perceptual_loss(inputs, reconstructions)
            percep_loss_ = p_loss_dict["perceptual_loss"]
            gram_loss_ = p_loss_dict["gram_loss"]
            p_loss = percep_loss_ * self.perceptual_weight
            gram_loss = gram_loss_ * self.perceptual_weight
            nll_loss = nll_loss + p_loss + gram_loss
        nll_loss = torch.mean(nll_loss)

        # * repa loss
        repa_loss = self.zero
        if hasattr(self, "repa_loss"):
            assert self.repa_loss_weight is not None
            repa_loss = self.repa_loss(inputs, tokenizer_feat)
            repa_loss = repa_loss * self.repa_loss_weight

        # * vf loss
        vf_loss = self.zero
        if self.use_vf:
            vf_loss = self.vf_loss(tokenizer_feat, inputs, nll_loss, enc_last_layer)
            vf_loss = vf_loss * self.vf_loss_weight  # 0.1 by default

        # * (un)conditional gan loss
        disc_factor = adopt_weight(
            self.disc_factor, global_step, threshold=self.disc_iter_start_for_g
        )
        d_weight = 1.0
        g_loss = self.zero
        if disc_factor > 0:
            with torch.autocast(device_type="cuda", dtype=inputs.dtype):
                if cond is None:
                    assert not self.disc_conditional
                    logits_fake = self.discriminator(reconstructions.contiguous())
                else:
                    assert self.disc_conditional
                    logits_fake = self.discriminator(
                        torch.cat((reconstructions.contiguous(), cond), dim=1)
                    )

                # g loss
                g_loss = self.generator_loss(logits_fake)

            d_weight *= self.gen_loss_weight_fn(nll_loss, g_loss, last_layer)
        d_weight *= self.discriminator_weight
        if not self.training:
            real_g_loss = disc_factor * g_loss
        g_loss = d_weight * disc_factor * g_loss

        # * quantization losses
        q_loss, q_loss_logs = self.q_loss(q_loss_total, q_loss_breakdown, global_step)

        # * basic losses
        loss = nll_loss + g_loss + repa_loss + vf_loss

        # * form logs
        log = self.train_generator_log_form(
            disc_factor=disc_factor,
            split=split,
            total_loss=loss,
            nll_loss=nll_loss,
            reconstruction_loss=recon_loss,
            gen_loss=g_loss,
            repa_loss=repa_loss,
            vf_loss=vf_loss,
            ssim_loss=ssim_loss,
            percep_loss=p_loss,
            gram_loss=gram_loss,
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
        add_prefix: bool = False,
    ):
        # * discrimator loss
        if self.disc_network_type == "stylegan3d":
            inputs = rearrange(inputs, "(n t) c h w -> n c t h w", t=self.num_frames)
            reconstructions = rearrange(
                reconstructions, "(n t) c h w -> n c t h w", t=self.num_frames
            )

        # second pass for discriminator update
        with torch.autocast(device_type="cuda", dtype=inputs.dtype):
            if cond is None:
                # detach that only gradients on discriminator
                logits_real = self.discriminator(inputs.contiguous())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(
                    torch.cat((inputs.contiguous(), cond), dim=1)
                )
                logits_fake = self.discriminator(
                    torch.cat((reconstructions.contiguous().detach(), cond), dim=1)
                )

        disc_factor = adopt_weight(
            self.disc_factor, global_step, threshold=self.disc_iter_start_for_d
        )
        disc_loss_out = self.zero
        if self.lecam_loss_weight is not None:
            self.lecam_ema.update(logits_real, logits_fake)
            lecam_loss = lecam_reg(logits_real, logits_fake, self.lecam_ema)
            lecam_loss = lecam_loss * self.lecam_loss_weight
            if disc_factor > 0.0:
                disc_loss_out = self.discriminator_loss(logits_real, logits_fake)
            d_loss = disc_factor * disc_loss_out + lecam_loss
        else:
            lecam_loss = self.zero
            if disc_factor > 0.0:
                disc_loss_out = self.discriminator_loss(logits_real, logits_fake)
            d_loss = disc_factor * disc_loss_out

        # r1 regularization loss from stylegan 2
        # for stablized training
        """
        non-sature loss: real logits: +5, fake logits: -5
        hinge loss: logits_real >> logits_fake; logits_real: (1., 2.), logits_fake: (-1., -2.)
        """
        if self.disc_reg_freq > 0 and (global_step + 1) % self.disc_reg_freq == 0:
            raise RuntimeError("use r1 reg, debugging ...")
            inputs.requires_grad_(True)
            logits_real = self.discriminator(inputs.contiguous())
            r1_loss = d_r1_loss(logits_real, inputs)
            r1_loss_scale = self.disc_reg_r1 / 2 * r1_loss * self.disc_reg_freq
            d_loss = d_loss + r1_loss_scale  # changed d_loss
        else:
            r1_loss = self.zero
            r1_loss_scale = self.zero

        log = self.train_disc_log_form(
            split=split,
            disc_factor=disc_factor,
            lecam_loss=lecam_loss,
            disc_loss=d_loss,
            logits_real=logits_real,
            logits_fake=logits_fake,
            r1_scale=r1_loss_scale,
            r1_loss=r1_loss,
            add_prefix=add_prefix,
        )
        disc_loss = dict(disc_loss=d_loss)

        return disc_loss, log

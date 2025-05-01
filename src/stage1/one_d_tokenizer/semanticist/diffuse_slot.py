import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
from loguru import logger
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.vision_transformer import PatchEmbed

sys.path.insert(0, __file__[: __file__.find("src")])
from src.stage1.discretization.collections import (
    BinarySphericalQuantizer,
    DiagonalGaussianDistribution,
    LogitLaplaceLoss,
)
from src.stage1.one_d_tokenizer.semanticist import vision_transformer
from src.stage1.one_d_tokenizer.semanticist.diffusion_transfomer import DiT
from src.stage1.one_d_tokenizer.semanticist.vision_transformer import VisionTransformer
from src.utilities.transport.diffusion import create_diffusion
from src.utilities.transport.flow_matching import create_transport
from src.utilities.transport.flow_matching.transport import Sampler


class DiTOnlyAttn(DiT):
    def __init__(
        self,
        *args,
        num_autoenc=32,  # n_slots
        autoenc_dim=4,  # slot dim
        use_repa=False,
        z_dim=768,
        encoder_depth=8,
        projector_dim=2048,
        learn_x_tokens: str = "full_size",
        **kwargs,
    ):
        super().__init__(*args, use_adaln=False, **kwargs)
        self.autoenc_dim = autoenc_dim
        self.null_cond = nn.Parameter(torch.zeros(1, num_autoenc, autoenc_dim))
        torch.nn.init.normal_(self.null_cond, std=0.02)
        self.hidden_size = kwargs["hidden_size"]
        self.autoenc_cond_embedder = nn.Linear(autoenc_dim, self.hidden_size)
        self.x_embedder = PatchEmbed(
            self.input_size,
            self.patch_size,
            autoenc_dim,
            self.hidden_size,
            strict_img_size=False,
            bias=True,
        )

        # no need for class embeddings and timestep embeddings
        self.y_embedder = nn.Identity()
        self.t_embedder = nn.Identity()

        self.cond_drop_prob = 0.1
        self.learn_x_tokens = learn_x_tokens
        if learn_x_tokens == "full_size":
            self.x_tokens = nn.Parameter(
                torch.randn(1, self.autoenc_dim, self.input_size, self.input_size)
            )
        elif learn_x_tokens == "only_channel":
            self.x_tokens = nn.Parameter(torch.randn(1, self.autoenc_dim))
        else:  # no learnt, set all zeros
            self.register_buffer("x_tokens", torch.zeros(1, self.autoenc_dim, 1))

        self.use_repa = use_repa
        self._repa_hook = None
        self.encoder_depth = encoder_depth
        if use_repa:
            self.projector = build_mlp(self.hidden_size, projector_dim, z_dim)

    def embed_cond(self, autoenc_cond, drop_mask=None):
        # autoenc_cond: (N, K, D)
        # drop_ids: (N)
        # self.null_cond: (1, K, D)
        batch_size = autoenc_cond.shape[0]

        # nested sampled condition
        if drop_mask is None:
            # randomly drop all conditions, for classifier-free guidance
            if self.training:
                drop_ids = (
                    torch.rand(batch_size, 1, 1, device=autoenc_cond.device)
                    < self.cond_drop_prob
                )  # [bs, 1, 1]
                # null_cond: [n_latents, D], [bs, n_latents, D]
                autoenc_cond_drop = torch.where(drop_ids, self.null_cond, autoenc_cond)
            else:
                autoenc_cond_drop = autoenc_cond
        else:
            # randomly drop some conditions according to the drop_mask (N, K)
            # True means keep
            autoenc_cond_drop = torch.where(
                drop_mask[:, :, None], autoenc_cond, self.null_cond
            )  # [bs, l, 1]
        return self.autoenc_cond_embedder(autoenc_cond_drop)

    def forward(self, autoenc_cond, drop_mask=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        autoenc_cond: (N, K, D) tensor of autoencoder conditions (slots)
        """
        # handle the x_tokens
        if self.learn_x_tokens in ("full_size", "zeros"):
            # repeat batchsize
            x = self.x_tokens.repeat(autoenc_cond.shape[0], 1, 1, 1)
        elif self.learn_x_tokens == "only_channel":
            # repeat bs and hw
            x = self.x_tokens[..., None, None].repeat_interleave(autoenc_cond.shape[0])

        x = (
            self.x_embedder(x) + self.pos_embed
        )  # (N, T, D), where T = H * W / patch_size ** 2
        autoenc = self.embed_cond(autoenc_cond, drop_mask)

        num_tokens = x.shape[1]
        x = torch.cat((x, autoenc), dim=1)  # noise and condition

        for i, block in enumerate(self.blocks):
            x = block(
                x, None
            )  # (N, T, D), None means no condition, only rely on encoded tokens
            if (i + 1) == self.encoder_depth and self.use_repa:
                projected = self.projector(x)
                self._repa_hook = projected[:, :num_tokens]

        # slot --> eps
        x = x[:, :num_tokens]
        x = self.final_layer(x)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)

        return x


class DiT_with_autoenc_cond(DiT):
    def __init__(
        self,
        *args,
        num_autoenc=32,  # n_slots
        autoenc_dim=4,  # slot dim
        use_repa=False,
        z_dim=768,
        encoder_depth=8,
        projector_dim=2048,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.autoenc_dim = autoenc_dim
        self.hidden_size = kwargs["hidden_size"]
        self.null_cond = nn.Parameter(torch.zeros(1, num_autoenc, autoenc_dim))
        torch.nn.init.normal_(self.null_cond, std=0.02)
        self.autoenc_cond_embedder = nn.Linear(autoenc_dim, self.hidden_size)
        self.y_embedder = nn.Identity()
        self.cond_drop_prob = 0.1

        self.use_repa = use_repa
        self._repa_hook = None
        self.encoder_depth = encoder_depth
        if use_repa:
            self.projector = build_mlp(self.hidden_size, projector_dim, z_dim)

    def embed_cond(self, autoenc_cond, drop_mask=None):
        # autoenc_cond: (N, K, D)
        # drop_ids: (N)
        # self.null_cond: (1, K, D)
        autoenc_cond.shape[0]

        # nested sampled condition
        # if drop_mask is None:
        #     # randomly drop all conditions, for classifier-free guidance
        #     if self.training:
        #         drop_ids = (
        #             torch.rand(batch_size, 1, 1, device=autoenc_cond.device)
        #             < self.cond_drop_prob
        #         )  # [bs, 1, 1]
        #         # null_cond: [n_latents, D], [bs, n_latents, D]
        #         autoenc_cond_drop = torch.where(drop_ids, self.null_cond, autoenc_cond)
        #     else:
        #         autoenc_cond_drop = autoenc_cond
        # else:
        if drop_mask is not None:
            # randomly drop some conditions according to the drop_mask (N, K)
            # True means keep
            autoenc_cond_drop = torch.where(
                drop_mask[:, :, None], autoenc_cond, self.null_cond
            )  # [bs, l, 1]
        return self.autoenc_cond_embedder(autoenc_cond_drop)

    def forward(self, x, t, autoenc_cond, drop_mask=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        autoenc_cond: (N, K, D) tensor of autoencoder conditions (slots)
        """
        x = (
            self.x_embedder(x) + self.pos_embed
        )  # (N, T, D), where T = H * W / patch_size ** 2
        c = self.t_embedder(t)  # (N, D)
        autoenc = self.embed_cond(autoenc_cond, drop_mask)

        num_tokens = x.shape[1]
        x = torch.cat((x, autoenc), dim=1)  # noise and condition

        for i, block in enumerate(self.blocks):
            x = block(x, c)  # (N, T, D)
            if (i + 1) == self.encoder_depth and self.use_repa:
                projected = self.projector(x)
                self._repa_hook = projected[:, :num_tokens]

        # slot --> eps
        x = x[:, :num_tokens]
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)

        return x

    def forward_with_cfg(self, x, t, autoenc_cond, drop_mask, y=None, cfg_scale=1.0):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, autoenc_cond, drop_mask)
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


# 256 x256 -> 256 / 512 = 256 / 128
# --> vq gan: 64/128 (kl, vq)


#################################################################################
#                                   DiT Configs                                  #
#################################################################################


dit_configs = {
    "DiT-XL-2": dict(depth=28, hidden_size=1152, patch_size=2, num_heads=16),
    "DiT-XL-4": dict(depth=28, hidden_size=1152, patch_size=4, num_heads=16),
    "DiT-XL-8": dict(depth=28, hidden_size=1152, patch_size=8, num_heads=16),
    "DiT-L-2": dict(depth=24, hidden_size=1024, patch_size=2, num_heads=16),
    "DiT-L-4": dict(depth=24, hidden_size=1024, patch_size=4, num_heads=16),
    "DiT-L-8": dict(depth=24, hidden_size=1024, patch_size=8, num_heads=16),
    "DiT-B-2": dict(depth=12, hidden_size=768, patch_size=2, num_heads=12),
    "DiT-B-4": dict(depth=12, hidden_size=768, patch_size=4, num_heads=12),
    "DiT-B-8": dict(depth=12, hidden_size=768, patch_size=8, num_heads=12),
    "DiT-S-2": dict(depth=12, hidden_size=384, patch_size=2, num_heads=6),
    "DiT-S-4": dict(depth=12, hidden_size=384, patch_size=4, num_heads=6),
    "DiT-S-8": dict(depth=12, hidden_size=384, patch_size=8, num_heads=6),
}


def create_dit_model(name: str, use_diffusion: bool):
    base_model_cls = DiT_with_autoenc_cond if use_diffusion else DiTOnlyAttn
    base_model_cfg = dit_configs[name]

    def _create_model(**kwargs):
        base_model_cfg.update(kwargs)
        return base_model_cls(**base_model_cfg)

    return _create_model


# *==============================================================
# * Token nested sampler
# *==============================================================


class NestedSampler(nn.Module):
    def __init__(
        self,
        num_slots,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.register_buffer("arange", torch.arange(num_slots))

    def uniform_sample(self, num):
        return torch.randint(1, self.num_slots + 1, (num,))

    def sample(self, num):
        samples = self.uniform_sample(num)
        return samples

    def forward(self, batch_size, device, inference_with_n_slots=-1):
        if self.training:
            b = self.sample(batch_size).to(device)  # [bs,]
        else:
            if inference_with_n_slots != -1:
                b = torch.full((batch_size,), inference_with_n_slots, device=device)
            else:
                b = torch.full((batch_size,), self.num_slots, device=device)
        b = torch.clamp(b, max=self.num_slots)

        # [1, n_latents] < [bs, 1] --> [bs, n_latents]
        slot_mask = self.arange[None, :] < b[:, None]  # (batch_size, num_slots)
        return slot_mask


# *==============================================================
# * No Diffusion Wrapper
# *==============================================================


class NoneDiffusionWrapper:
    def __init__(self):
        pass

    def training_losses(
        self,
        decoder: nn.Module,
        x: torch.Tensor,
        autoenc_cond: torch.Tensor,
        drop_mask: torch.Tensor,
    ):
        recon = decoder(autoenc_cond, drop_mask)
        assert recon.shape == x.shape

        # loss
        loss_out = {"loss": F.l1_loss(recon, x), "pred_x_clean": recon}

        return loss_out

    def sample(
        self,
        decoder: nn.Module,
        autoenc_cond: torch.Tensor,
        drop_mask: torch.Tensor | None = None,
    ):
        return decoder(autoenc_cond, drop_mask)


class DiffuseSlot(nn.Module):
    def __init__(
        self,
        enable_nest: bool = False,
        enable_nest_after: int = -1,
        vae: str | None = "stabilityai/sd-vae-ft-ema",
        # diffusion or fm kwargs
        diffusion_type: str | None = "diffusion",
        num_sampling_steps: str = "ddim25",
        fm_sample_type: str | None = "ode",
        diffusion_options: dict | None = None,
        fm_options: dict | None = None,
        # repa
        use_repa: bool = False,
        repa_encoder_depth: int = 8,
        repa_loss_weight: float = 1.0,
        # models
        encoder: str = "vit_base_patch16",
        dit_model: str = "DiT-B-4",
        img_channels: int = 8,  # multispectral or hyperspectral images
        enc_img_size: int = 256,
        enc_causal: bool = True,
        num_slots: int = 256,
        slot_dim: int = 16,
        norm_slots: bool = False,
        drop_path_rate: float = 0.1,
        encoder_block_checkpoint: bool = False,
        decoder_block_checkpoint: bool = False,
        compile_model: bool = False,
        # quantier
        quantizer_type: str | None = None,
        quantizer_kwargs: dict | None = None,
    ):
        super().__init__()
        self.img_channels = img_channels
        self.enc_img_size = enc_img_size

        # repa
        self.use_repa = use_repa
        self.repa_loss_weight = repa_loss_weight
        if use_repa:
            self.repa_encoder = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14"
            )
            self.repa_encoder.image_size = 224
            for param in self.repa_encoder.parameters():
                param.requires_grad = False
            self.repa_encoder.eval()

        # train and generate diffusion
        self.diffusion_type = diffusion_type
        self.fm_sample_type = fm_sample_type
        if diffusion_type == "fm":
            assert fm_sample_type in [
                "sde",
                "ode",
            ], 'fm_sample_type must be "sde" or "ode"'

        if diffusion_type == "diffusion":
            default_diffusion_options = {
                "timestep_respacing": "",
                "learn_sigma": False,
                "predict_xstart": False,
            }

            _train_diff_opts = default_diffusion_options.copy()
            if diffusion_options is not None:
                _train_diff_opts.update(diffusion_options)

            _generate_diff_opts = _train_diff_opts.copy()
            _generate_diff_opts["timestep_respacing"] = num_sampling_steps
            self.learn_sigma = learn_sigma = _train_diff_opts.get("learn_sigma", False)

            self.diffusion = create_diffusion(**_train_diff_opts)
            self.gen_diffusion = create_diffusion(**_generate_diff_opts)

        elif diffusion_type == "fm":
            self.learn_sigma = learn_sigma = False
            fm_default_options = {
                "path_type": "Linear",
                "prediction": "velocity",
                "train_eps": 1e-4,
                "sample_eps": 1e-4,
            }

            _options = fm_default_options.copy()
            if fm_options is not None:
                _options.update(fm_options)

            self.diffusion = create_transport(**_options)
            self.gen_diffusion = Sampler(self.diffusion)

        else:
            self.diffusion = NoneDiffusionWrapper()
            self.gen_diffusion = NoneDiffusionWrapper()

        # * dit decoder ==============
        self.distill_from_pretrained_vae = vae is not None
        if self.distill_from_pretrained_vae:
            self.encoder_patch_sz = 8 if "mar" not in vae else 16
            self.dit_input_size = enc_img_size // self.encoder_patch_sz
            self.dit_in_channels = 4 if "mar" not in vae else 16
        else:
            self.encoder_patch_sz = 16  # fixed 16 patch size
            self.dit_input_size = enc_img_size
            self.dit_in_channels = img_channels
            decoder_patch_size = int(dit_model[-1])
            if decoder_patch_size < 8:
                logger.warning(
                    f"[Patch Size]: patch size={decoder_patch_size} is too small, may use lots of GPU mem."
                )

        # double channel
        if self.diffusion_type in ("diffusion", "fm"):
            assert quantizer_type != "kl", (
                "when using kl quantizer, diffusion_type should can not be any diffusion"
            )

        self.use_diffusion = True
        if self.diffusion_type == "fm":
            double_out_channel = False
            if learn_sigma:
                logger.warning(
                    f"[Diffusion]: diffusion_type={self.diffusion_type} can not use learn_sigma=True"
                )
        elif self.diffusion_type == "diffusion":
            double_out_channel = learn_sigma
        else:  # no diffusion
            self.use_diffusion = False
            double_out_channel = False

        z_dim = 768
        self.decoder = create_dit_model(dit_model, self.use_diffusion)(  # DiT-L-2
            input_size=self.dit_input_size,
            in_channels=self.dit_in_channels,
            # slots
            num_autoenc=num_slots,
            autoenc_dim=slot_dim,
            # repa loss
            use_repa=use_repa,
            encoder_depth=repa_encoder_depth,
            # zd dim
            z_dim=z_dim,
            block_checkpoint=decoder_block_checkpoint,
            learn_sigma=double_out_channel,
        )
        self.decoder: DiT_with_autoenc_cond | DiTOnlyAttn

        # * pretrained vae ===============
        if self.distill_from_pretrained_vae:
            self.vae = AutoencoderKL.from_pretrained(vae)
            self.scaling_factor = self.vae.config.scaling_factor
            self.vae.eval().requires_grad_(False)

        # * vit encoder ===================
        # patch size = 16, 512 / 16 = 32
        # 32 * 32 = 1024 tokens
        self.enc_causal = enc_causal
        self.enc_img_size = enc_img_size
        encoder_fn = vision_transformer.__dict__[encoder]

        self.encoder: VisionTransformer = encoder_fn(  # vit_base_patch16
            img_size=[enc_img_size],
            num_slots=num_slots,
            drop_path_rate=drop_path_rate,
            in_chans=img_channels,
            block_act_checkpoint=encoder_block_checkpoint,
        )
        self.num_slots = num_slots
        self.norm_slots = norm_slots
        self.num_channels = self.encoder.num_features

        # * compile models =================
        logger.info(f"[Compile]: compile models ...") if compile_model else None
        self.decoder = (
            torch.compile(self.decoder, mode="reduce-overhead")
            if compile_model
            else self.decoder
        )
        self.encoder = (
            torch.compile(self.encoder, mode="reduce-overhead")
            if compile_model
            else self.encoder
        )

        # * proj ================
        # vitpathch16 model from hidden dim to latent dim (e.g., 16)
        self.encoder2slot = nn.Linear(self.num_channels, slot_dim)

        # * quantizer ==========
        if not self.distill_from_pretrained_vae:
            # attributes
            self.quantizer_type = quantizer_type
            if quantizer_kwargs is None:
                quantizer_kwargs = {}

            if self.norm_slots:
                self.norm_slots = quantizer_kwargs.get("pre_quant_norm", False)
                logger.warning(
                    f"[Quantizer]: norm_slots is True, set to False for any quantizer"
                )

            # init quantizer
            if quantizer_type == "bsq":
                _default_kwargs = dict(
                    embed_dim=slot_dim,
                    l2_norm=True,
                    persample_entropy_compute="analytical",
                    beta=0.0,  # commitment loss
                    gamma=1.0,
                    gamma0=1.0,
                    zeta=1.0,
                    inv_temperature=1.0,
                    cb_entropy_compute="group",
                    input_format="blc",
                    group_size=slot_dim // 2,
                )
                kwargs = (
                    _default_kwargs if len(quantizer_kwargs) == 0 else quantizer_kwargs
                )
                self.quantizer_kwargs = kwargs
                self.bsq_logit_laplace = self.quantizer_kwargs.pop(
                    "logit_laplace", False
                )
                if self.bsq_logit_laplace:  # no used
                    self.bsq_logt_lap_loss = LogitLaplaceLoss(logit_laplace_eps=0.1)
                    logger.warning(
                        f"[Quantizer]: Logit Laplace Loss is not implemented yet,"
                        "it will not take any effect"
                    )
                self.quantizer = BinarySphericalQuantizer(**kwargs)
            elif quantizer_type == "kl":
                self.kl_weight = quantizer_kwargs.get("kl_weight", 1e-6)
                logvar_init = quantizer_kwargs.get("logvar_init", 0.0)
                self.logvar = nn.Parameter(
                    torch.ones(size=()) * logvar_init, requires_grad=False
                )
                self.quantizer_kwargs = {}
                self.quantizer = DiagonalGaussianDistribution
            elif quantizer_type is None:
                self.quantizer = None
                self.quantizer_kwargs = {}
                logger.info("Do not use quantizer")
            else:
                raise ValueError("quantizer_type must be bsq, kl, or None")
        else:
            if self.quantizer_type is not None:
                logger.warning(
                    f"[Quantizer]: distill from pretrained vae, ignore quantizer_type={self.quantizer_type}"
                )
            self.quantizer = None

        # * token sampler ==========
        self.nested_sampler = NestedSampler(num_slots)
        self.enable_nest = enable_nest
        self.enable_nest_after = enable_nest_after

        # * summary ================
        logger.info(
            f"[Diffusion Tokenizer]:\n"
            f"diffusion type={self.diffusion_type} \n"
            f"decoder_model={dit_model}, tokenizer factor={self.encoder_patch_sz} \n"
            f"decoder image (latent) size={self.dit_input_size} \n"
            f"decoder patch size={self.decoder.patch_size}\n"
            f"tokenizer dim={self.dit_in_channels}, num latents={self.num_slots} \n"
            f"encoder causal={self.enc_causal}, encoder={self.encoder.__class__.__name__} \n"
            f"use pretrained vae={self.distill_from_pretrained_vae}\n"
            f"use repa={self.use_repa}, repa_loss_weight={self.repa_loss_weight}\n"
            f"norm slots={self.norm_slots}\n"
        )

    # * ==========================================================
    # * distill from vae
    @torch.no_grad()
    def vae_encode(self, x):
        x = x * 2 - 1
        x = self.vae.encode(x)
        if hasattr(x, "latent_dist"):
            x = x.latent_dist
        return x.sample().mul_(self.scaling_factor)

    @torch.no_grad()
    def vae_decode(self, z):
        z = self.vae.decode(z / self.scaling_factor)
        if hasattr(z, "sample"):
            z = z.sample
        return (z + 1) / 2

    @torch.no_grad()
    def repa_encode(self, x):
        mean = (
            torch.Tensor(IMAGENET_DEFAULT_MEAN)
            .to(x.device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        std = (
            torch.Tensor(IMAGENET_DEFAULT_STD)
            .to(x.device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        x = (x - mean) / std

        # interpolate to image size
        if self.repa_encoder.image_size != self.enc_img_size:
            x = torch.nn.functional.interpolate(
                x, self.repa_encoder.image_size, mode="bicubic"
            )

        # get dino features
        x = self.repa_encoder.forward_features(x)["x_norm_patchtokens"]

        return x

    def encode_slots(self, x):
        # encode and projection
        slots = self.encoder(x, is_causal=self.enc_causal)
        slots = self.encoder2slot(slots)

        # quantize
        if self.quantizer is not None:
            if self.quantizer_type == "bsq":
                slots, quantizer_loss, quan_info = self.quantizer(slots)
            elif self.quantizer_type == "kl":
                posteriors = self.quantizer(slots, mean_std_split_dim=-1)
                slots = posteriors.sample()
                kl = posteriors.kl()
                kl_loss = kl.sum() / kl.shape[0]
                quan_info = dict(
                    recon_div_var=torch.exp(self.logvar),  # recon_loss / recon_div_var
                    kl_loss=kl_loss * self.kl_weight,
                )
                quantizer_loss = quan_info["kl_loss"]
        else:
            quantizer_loss = torch.tensor(0.0).to(x)
            quan_info = {}

        if self.norm_slots:
            # norm the feature, continuous
            if self.quantizer is None:
                slots_std = torch.std(slots, dim=-1, keepdim=True)
                slots_mean = torch.mean(slots, dim=-1, keepdim=True)
                slots = (slots - slots_mean) / slots_std
            # has quantizer, discrete
            else:
                slots = F.normalize(slots, dim=-1)

        return slots, quantizer_loss, quan_info

    def forward_with_latents(
        self,
        x_vae,
        slots,
        z,
        sample=False,
        epoch=None,
        inference_with_n_slots=-1,
        cfg=1.0,
        get_pred_x_clean: bool = False,
    ):
        losses = {}
        batch_size = x_vae.shape[0]
        device = x_vae.device

        if (
            epoch is not None
            and epoch >= self.enable_nest_after
            and self.enable_nest_after != -1
        ):
            self.enable_nest = True
        # nest sample
        if self.enable_nest or inference_with_n_slots != -1:
            drop_mask = self.nested_sampler(
                batch_size,
                device,
                inference_with_n_slots=inference_with_n_slots,
            )
        else:
            drop_mask = None

        # sample
        if sample:
            return self.sample(slots, drop_mask=drop_mask, cfg=cfg)

        # prepare for diffusion
        # vae latents
        model_kwargs = dict(autoenc_cond=slots, drop_mask=drop_mask)
        if self.diffusion_type == "diffusion":
            t = torch.randint(0, 1000, (x_vae.shape[0],), device=device)
            train_losses_inp = dict(
                model=self.decoder,
                x_start=x_vae,
                t=t,
                model_kwargs=model_kwargs,
                get_pred_x_clean=get_pred_x_clean,
            )
        elif self.diffusion_type == "fm":
            train_losses_inp = dict(
                model=self.decoder,
                x1=x_vae,
                model_kwargs=model_kwargs,
                get_pred_x_clean=get_pred_x_clean,
            )
        else:
            train_losses_inp = dict(
                decoder=self.decoder,
                x=x_vae,
                autoenc_cond=slots,
                drop_mask=drop_mask,
            )

        # diffusion with dit
        loss_dict = self.diffusion.training_losses(**train_losses_inp)
        diff_loss = loss_dict["loss"].mean()
        losses["diff_loss"] = diff_loss
        if get_pred_x_clean:
            losses["pred_x_clean"] = loss_dict["pred_x_clean"]

        # repa loss: dino feature and dit feature
        if self.use_repa:
            assert self.decoder._repa_hook is not None and z is not None
            z_from_enc = self.decoder._repa_hook  # dit features

            if z_from_enc.shape[1] != z.shape[1]:
                z_from_enc = interpolate_features(z_from_enc, z.shape[1])

            # norm
            z_from_enc = F.normalize(z_from_enc, dim=-1)
            z = F.normalize(z, dim=-1)

            # loss
            repa_loss = -torch.sum(z_from_enc * z, dim=-1)
            losses["repa_loss"] = repa_loss.mean() * self.repa_loss_weight
        else:
            losses["repa_loss"] = torch.tensor(0.0).to(x_vae)

        return losses

    def forward(
        self,
        x,
        sample=False,
        epoch=None,
        inference_with_n_slots=-1,
        cfg=1.0,
        get_pred_x_clean: bool = False,
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, dict[str, torch.Tensor] | torch.Tensor],
    ]:
        """
        img -----> convolutional encoder -> 2D latents -- maybe quantize -> 2D quantized latents
             |---> DINO -> repa features (z)
        2D latent --- as conditions --> dit --> diffusion loss
        z --- with dit repa hook --> repa loss

        """
        others = {}

        # vae encode
        if self.distill_from_pretrained_vae:
            x_vae = self.vae_encode(x)
        else:
            x_vae = x  # keep at same

        # dino feature
        z = self.repa_encode(x) if self.use_repa else None

        # encode x to slots
        # * encoder -> linear -> latent
        slots, quantizer_loss, quan_info = self.encode_slots(x)
        others["quan_info"] = quan_info

        # losses
        # * decoder and loss
        # if sample, `losses` is sampled Tensor
        # else: `losses` is a dict
        losses: dict | torch.Tensor = self.forward_with_latents(
            x_vae,
            slots,
            z,
            sample,
            epoch,
            inference_with_n_slots,
            cfg,
            get_pred_x_clean,
        )
        if not sample:
            # get diffusion one-step predicted x start
            if get_pred_x_clean:
                others["pred_x_clean"] = losses.pop("pred_x_clean")
            # update losses
            losses["q_loss"] = quantizer_loss

        return losses, others

    @torch.no_grad()
    def sample(self, slots, drop_mask=None, cfg=1.0):
        batch_size = slots.shape[0]
        device = slots.device

        # * ==========================================================
        # * use diffusion

        if self.use_diffusion:
            if self.distill_from_pretrained_vae:
                # works on pretrained vae latent space
                z = torch.randn(
                    batch_size,
                    self.dit_in_channels,
                    self.dit_input_size,
                    self.dit_input_size,
                    device=device,
                )
            else:
                # works on pixel space
                z = torch.randn(
                    batch_size,
                    self.img_channels,
                    self.enc_img_size,
                    self.enc_img_size,
                    device=device,
                )

            # sample

            if cfg != 1.0:
                z = torch.cat([z, z], 0)
                null_slots = self.decoder.null_cond.expand(batch_size, -1, -1)
                slots = torch.cat([slots, null_slots], 0)

                if drop_mask is not None:
                    null_cond_mask = torch.ones_like(drop_mask)  # keep all
                    drop_mask = torch.cat([drop_mask, null_cond_mask], 0)

                model_kwargs = dict(
                    autoenc_cond=slots, drop_mask=drop_mask, cfg_scale=cfg
                )
                sample_fn = self.decoder.forward_with_cfg
            else:
                model_kwargs = dict(autoenc_cond=slots, drop_mask=drop_mask)
                sample_fn = self.decoder.forward

            # * diffusion sampling
            if self.diffusion_type == "diffusion":
                samples = self.gen_diffusion.p_sample_loop(
                    sample_fn,
                    z.shape,
                    z,
                    clip_denoised=not self.distill_from_pretrained_vae,
                    model_kwargs=model_kwargs,
                    progress=True,
                    device=device,
                )
            else:
                if self.fm_sample_type == "sde":
                    diffusion_sample_fn = self.gen_diffusion.sample_sde(
                        sampling_method="Euler",
                        diffusion_norm=1.0,
                        diffusion_form="SBDM",
                        num_steps=25,
                        last_step="Mean",
                    )
                    samples = diffusion_sample_fn(z, sample_fn, **model_kwargs)[-1]
                elif self.fm_sample_type == "ode":
                    diffusion_sample_fn = self.gen_diffusion.sample_ode(
                        sampling_method="dopri5",
                        num_steps=25,
                        atol=1e-4,
                        rtol=1e-4,
                        progress=True,
                    )
                    samples = diffusion_sample_fn(z, sample_fn, **model_kwargs)[-1]

            if cfg != 1.0:
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        # * ==========================================================
        # * not use diffusion

        else:
            samples = self.gen_diffusion.sample(self.decoder, slots, drop_mask)

        # decoder
        if self.distill_from_pretrained_vae:
            samples = self.vae_decode(samples)

        return samples

    def train(self, mode=True):
        """Override train() to keep certain components in eval mode"""
        super().train(mode)
        if self.distill_from_pretrained_vae:
            self.vae.eval()
            if hasattr(self, "repa_encoder"):
                if isinstance(self.repa_encoder, nn.Module):
                    self.repa_encoder.eval()
        return self

    def eval(self):
        self.encoder.eval()
        self.encoder2slot.eval()
        self.decoder.eval()
        if isinstance(self.quantizer, nn.Module):
            self.quantizer.eval()

    def get_last_layer(self):
        return self.decoder.get_last_layer()


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )


def interpolate_features(x, target_len):
    """Interpolate features to match target sequence length.
    Args:
        x: tensor of shape (B, T1, D)
        target_len: desired sequence length T2
    Returns:
        tensor of shape (B, T2, D)
    """
    B, T1, D = x.shape
    H1 = W1 = int(math.sqrt(T1))
    H2 = W2 = int(math.sqrt(target_len))

    # Reshape to 2D spatial dimensions and move channels to second dimension
    x = x.reshape(B, H1, W1, D).permute(0, 3, 1, 2)

    # Interpolate
    x = F.interpolate(x, size=(H2, W2), mode="bicubic", align_corners=False)

    # Reshape back to sequence
    return x.permute(0, 2, 3, 1).reshape(B, target_len, D)


if __name__ == "__main__":
    """
    params:
        encoder: "vit_base_patch16"
        enc_img_size: 256
        enc_causal: True
        enc_use_mlp: False
        num_slots: 256
        slot_dim: 16
        norm_slots: True
        dit_model: "DiT-XL-2"
        vae: "xwen99/mar-vae-kl16"
        enable_nest: False
        enable_nest_after: 50
        use_repa: True
        eval_fid: True
        fid_stats: "fid_stats/adm_in256_stats.npz"
        num_sampling_steps: "250"
        ckpt_path: None
    """

    torch.cuda.set_device(1)
    tokenizer = DiffuseSlot(
        encoder="vit_base_patch16",
        dit_model="DiT-B-8",
        img_channels=8,
        enc_img_size=512,
        enc_causal=True,
        use_repa=False,
        num_slots=256,
        slot_dim=16,
        vae=None,
        encoder_block_checkpoint=True,
        decoder_block_checkpoint=True,
        diffusion_type=None,
        fm_options={
            "path_type": "Linear",
            "prediction": "velocity",
            "train_eps": 1e-4,
            "sample_eps": 1e-4,
        },
        compile_model=False,
        quantizer_type=None,
        fm_sample_type="ode",
    ).cuda()
    tokenizer.encoder = tokenizer.encoder.to(torch.bfloat16)
    tokenizer.encoder2slot = tokenizer.encoder2slot.to(torch.bfloat16)
    tokenizer.decoder = tokenizer.decoder.to(torch.bfloat16)
    x = torch.randn(7, 8, 512, 512).cuda().to(torch.bfloat16)
    opt = torch.optim.Adam(tokenizer.parameters(), lr=1e-4)

    import time

    from tqdm import trange

    def func_mem_wrapper(func):
        def wrapper(*args, **kwargs):
            # 记录初始显存占用
            torch.cuda.reset_peak_memory_stats()  # reset the peak memory stats
            initial_memory = torch.cuda.memory_allocated()

            ret = func(*args, **kwargs)

            # 执行 tokenizer 并记录显存占用
            allocated_memory = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()

            # 计算显存增量
            memory_usage = allocated_memory - initial_memory

            # 打印显存占用信息
            print(f"Initial memory allocated: {initial_memory / 1024**2:.2f} MB")
            print(
                f"Memory allocated after forward pass: {allocated_memory / 1024**2:.2f} MB"
            )
            print(f"Peak memory allocated: {peak_memory / 1024**2:.2f} MB")
            print(f"Memory usage: {memory_usage / 1024**2:.2f} MB")

            return ret

        return wrapper

    def func_speed_wrapper(test_num=100):
        def inner_func_wrapper(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()

                for _ in trange(test_num):
                    ret = func(*args, **kwargs)

                end_time = time.time()
                total_time = end_time - start_time
                average_time = total_time / test_num

                print(f"Function {func.__name__} executed {test_num} times.")
                print(f"Total time: {total_time:.4f} seconds")
                print(f"Average time per execution: {average_time:.4f} seconds")

                return ret

            return wrapper

        return inner_func_wrapper

    # @func_mem_wrapper
    @func_speed_wrapper(test_num=20)
    def run_forward_backward():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = tokenizer(x, get_pred_x_clean=True)
            loss = loss[0]
            opt.zero_grad()
            loss["diff_loss"].backward()
            opt.step()

    @func_mem_wrapper
    @torch.no_grad()
    def run_sampling():
        tokenizer.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = tokenizer(
                x, sample=True, epoch=None, inference_with_n_slots=-1, cfg=2.0
            )[0]
        print(y.shape)

    run_forward_backward()
    # run_sampling()

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from src.stage1.cosmos.inference.utils import load_jit_model_shape_matched
from src.stage1.cosmos.modules.layers2d import DecoderDiff, Encoder
from src.stage1.discretization.collections import (
    BinarySphericalQuantizer,
    DiagonalGaussianDistribution,
    LogitLaplaceLoss,
)
from src.utilities.transport.diffusion import create_diffusion
from src.utilities.transport.flow_matching import create_transport
from src.utilities.transport.flow_matching.transport import Sampler


def build_pretrained_cosmos_tokenizer(
    enc_path: str,
    dec_path: str,
    tokenizer_cfg: dict,
):
    device = torch.cuda.current_device()
    tok_enc, enc_mod_keys = load_jit_model_shape_matched(
        enc_path,
        tokenizer_config=tokenizer_cfg,
        device=device,
        part="encoder",
    )
    tok_dec, dec_mod_keys = load_jit_model_shape_matched(
        dec_path,
        tokenizer_config=tokenizer_cfg,
        device=device,
        part="decoder",
    )

    return tok_enc, enc_mod_keys, tok_dec, dec_mod_keys


class TwoDimDiffusionSlots(nn.Module):
    def __init__(
        self,
        # tokenizer (encoder and decoder)
        tokenizer_cfg: DictConfig | dict | None = None,
        encoder_cfg: DictConfig | dict | None = None,
        decoder_cfg: DictConfig | dict | None = None,
        enc_path: str | None = None,
        dec_path: str | None = None,
        img_channels: int = 8,  # multispectral or hyperspectral images
        # repa loss
        use_repa: bool = False,
        enc_img_size: int = 256,
        repa_loss_weight: float = 0.1,
        # model compilation
        compile_model: bool = False,
        # diffusions
        diffusion_type: Literal["diffusion", "fm"] = "diffusion",
        fm_options: dict | None = None,
        diffusion_options: dict | None = None,
        num_sampling_steps: str = "ddim25",
        fm_sample_type: str | None = "ode",
        # quantizer
        norm_z: bool = True,
        quantizer_type: str | None = None,
        quantizer_kwargs: dict | None = None,
    ):
        super().__init__()

        def to_dict_any(cfg: DictConfig):
            if isinstance(cfg, DictConfig):
                cfg = OmegaConf.to_container(tokenizer_cfg)
            assert isinstance(cfg, dict), "cfg must be a DictConfig or a dict"
            return cfg

        if tokenizer_cfg is not None:
            encoder_cfg = decoder_cfg = to_dict_any(tokenizer_cfg)
        else:
            assert encoder_cfg is not None and decoder_cfg is not None
            encoder_cfg = to_dict_any(encoder_cfg)
            decoder_cfg = to_dict_any(decoder_cfg)

        self.img_channels = img_channels
        self.enc_img_size = enc_img_size

        # train and generate diffusion
        self.diffusion_type = diffusion_type
        self.fm_sample_type = fm_sample_type
        if diffusion_type == "fm":
            assert fm_sample_type in [
                "sde",
                "ode",
            ], 'fm_sample_type must be "sde" or "ode"'
        if diffusion_type == "diffusion":
            _train_diff_opts = (
                dict(timestep_respacing="", learn_sigma=False) if diffusion_options is None else diffusion_options
            )
            self.learn_sigma = learn_sigma = _train_diff_opts.get("learn_sigma", False)
            _generate_diff_opts = _train_diff_opts.copy()
            _generate_diff_opts["timestep_respacing"] = num_sampling_steps
            self.diffusion = create_diffusion(**_train_diff_opts)
            self.gen_diffusion = create_diffusion(**_generate_diff_opts)
        elif diffusion_type == "fm":
            self.learn_sigma = False
            fm_default_options = dict(
                path_type="Linear",
                prediction="velocity",
            )
            _options = fm_default_options if fm_options is None else fm_options
            self.diffusion = create_transport(**_options)
            self.gen_diffusion = Sampler(self.diffusion)
        else:
            raise ValueError("diffusion_type must be diffusion or fm")

        # double channels
        if self.diffusion_type in ("diffusion", "fm"):
            assert quantizer_type != "kl", "when using kl quantizer, diffusion_type should can not be any diffusion"
        if self.diffusion_type == "fm":
            double_out_channel = False
            if self.learn_sigma:
                logger.warning(
                    f"[Diffusion]: diffusion_type={self.diffusion_type} can not use learn_sigma=True, set it to False"
                )
        elif self.diffusion_type == "diffusion":
            double_out_channel = learn_sigma

        # * ==========================================================
        # * tokenizer

        tokenizer_cfg["learn_sigma"] = double_out_channel

        if enc_path is not None and dec_path is not None:
            logger.info(f"[Tokenizer]: fintuning model from pretrained Cosmos tokenizer")
            self.encoder, self.enc_parsed_keys, self.decoder, self.dec_parsed_keys = build_pretrained_cosmos_tokenizer(
                enc_path=enc_path,
                dec_path=dec_path,
                tokenizer_cfg=tokenizer_cfg,
            )
        else:
            logger.info(f"[Tokenizer]: building a new Cosmos tokenizer")
            self.encoder = Encoder(**encoder_cfg)
            self.decoder = DecoderDiff(**decoder_cfg)

        # * compile models =================

        logger.info(f"[Compile]: compile models ...") if compile_model else None
        self.decoder = torch.compile(self.decoder, mode="reduce-overhead") if compile_model else self.decoder
        self.encoder = torch.compile(self.encoder, mode="reduce-overhead") if compile_model else self.encoder

        # * ==========================================================
        # * quantizer

        # attributes
        self.quantizer_type = quantizer_type
        if quantizer_kwargs is None:
            quantizer_kwargs = {}
        self.norm_z = quantizer_kwargs.pop("pre_quant_norm", False) or norm_z
        z_channels = decoder_cfg["z_channels"]

        # init quantizer
        if quantizer_type == "bsq":
            _default_kwargs = dict(
                embed_dim=z_channels,
                l2_norm=True,
                persample_entropy_compute="analytical",
                beta=0.0,
                gamma=1.0,
                gamma0=1.0,
                zeta=1.0,
                inv_temperature=1.0,
                cb_entropy_compute="group",
                input_format="bchw",
                group_size=z_channels // 2,
            )
            kwargs = _default_kwargs if len(quantizer_kwargs) == 0 else quantizer_kwargs
            self.quantizer_kwargs = kwargs
            self.bsq_logit_laplace = self.quantizer_kwargs.pop("logit_laplace", False)
            if self.bsq_logit_laplace:
                self.bsq_logt_lap_loss = LogitLaplaceLoss(logit_laplace_eps=0.1)
                logger.warning(f"[Quantizer]: Logit Laplace Loss is not implemented yet,it will not take any effect")
            self.quantizer = BinarySphericalQuantizer(**kwargs)  # type: ignore
        elif quantizer_type == "kl":
            self.kl_weight = quantizer_kwargs.get("kl_weight", 1e-6)
            logvar_init = quantizer_kwargs.get("logvar_init", 0.0)
            self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init, requires_grad=False)
            self.quantizer_kwargs = {}
            self.quantizer = DiagonalGaussianDistribution
        elif quantizer_type is None:
            self.quantizer = None
            self.quantizer_kwargs = {}
            logger.info("[Tokenizer]: Do not use quantizer")
        else:
            raise ValueError("quantizer_type must be bsq, kl, or None")

        if quantizer_type is not None:
            logger.info(f"[Quantizer]: {quantizer_type} quantizer is used")

        # * ==========================================================
        # * repa loss

        self.use_repa = use_repa
        self.repa_loss_weight = repa_loss_weight
        if use_repa:
            self.repa_encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            self.repa_encoder.image_size = 224
            for param in self.repa_encoder.parameters():
                param.requires_grad = False
            self.repa_encoder.eval()

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

        # dino feature
        repa_feat = self.repa_encode(x) if self.use_repa else None

        # encode x to slots
        # * encoder -> linear -> latent
        enc, quantizer_loss, quan_info = self.encode(x)
        others["quan_info"] = quan_info

        # losses
        # * decoder and loss
        losses: dict | torch.Tensor = self.forward_with_latents(
            x,
            enc,
            repa_feat,
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

    def encode(self, x):
        # encode and projection
        enc = self.encoder(x)
        if self.norm_z:
            enc = F.normalize(enc, dim=1)  # [b,c,h,w]

        # quantize
        if self.quantizer is not None:
            if self.quantizer_type == "bsq":
                enc, quantizer_loss, quan_info = self.quantizer(enc)
            elif self.quantizer_type == "kl":
                posteriors = self.quantizer(enc, mean_std_split_dim=-1)
                enc = posteriors.sample()
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

        return enc, quantizer_loss, quan_info

    @torch.no_grad()
    def repa_encode(self, x):
        mean = torch.Tensor(IMAGENET_DEFAULT_MEAN).to(x.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        std = torch.Tensor(IMAGENET_DEFAULT_STD).to(x.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        x = (x - mean) / std

        # interpolate to image size
        if self.repa_encoder.image_size != self.enc_img_size:
            x = torch.nn.functional.interpolate(x, self.repa_encoder.image_size, mode="bicubic")

        # get dino features
        x = self.repa_encoder.forward_features(x)["x_norm_patchtokens"]

        return x

    def forward_with_latents(
        self,
        x,
        enc,
        repa_feat,
        sample=False,
        epoch=None,
        inference_with_n_slots=-1,
        cfg=1.0,
        get_pred_x_clean: bool = False,
    ):
        losses = {}
        x.shape[0]
        device = x.device

        # if (
        #     epoch is not None
        #     and epoch >= self.enable_nest_after
        #     and self.enable_nest_after != -1
        # ):
        #     self.enable_nest = True
        # # nest sample
        # if self.enable_nest or inference_with_n_slots != -1:
        #     drop_mask = self.nested_sampler(
        #         batch_size,
        #         device,
        #         inference_with_n_slots=inference_with_n_slots,
        #     )
        # else:
        #     drop_mask = None

        # sample
        if sample:
            return self.sample(enc, cfg=cfg)

        # prepare for diffusion
        # vae latents
        model_kwargs = dict(z=enc)
        if self.diffusion_type == "diffusion":
            t = torch.randint(0, 1000, (x.shape[0],), device=device)
            train_losses_inp = dict(
                model=self.decoder,
                x_start=x,
                t=t,
                model_kwargs=model_kwargs,
                get_pred_x_clean=get_pred_x_clean,
            )
        else:
            train_losses_inp = dict(
                model=self.decoder,
                x1=x,
                model_kwargs=model_kwargs,
                get_pred_x_clean=get_pred_x_clean,
            )

        # diffusion with decoder
        loss_dict = self.diffusion.training_losses(**train_losses_inp)
        diff_loss = loss_dict["loss"].mean()
        losses["diff_loss"] = diff_loss
        if get_pred_x_clean:
            losses["pred_x_clean"] = loss_dict["pred_x_clean"]

        # repa loss: dino feature and dit feature
        if self.use_repa:
            assert self.decoder._repa_hook is not None and repa_feat is not None
            z_tilde = self.decoder._repa_hook  # dit features

            if z_tilde.shape[1] != repa_feat.shape[1]:
                z_tilde = interpolate_features(z_tilde, repa_feat.shape[1])

            # norm
            z_tilde = F.normalize(z_tilde, dim=-1)
            repa_feat = F.normalize(repa_feat, dim=-1)

            # loss
            repa_loss = -torch.sum(z_tilde * repa_feat, dim=-1)
            losses["repa_loss"] = repa_loss.mean() * self.repa_loss_weight
        else:
            losses["repa_loss"] = torch.tensor(0.0).to(x)

        return losses

    @torch.no_grad()
    def sample(self, enc, cfg=1.0):
        batch_size = enc.shape[0]
        device = enc.device

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
            null_slots = self.decoder.null_cond.expand(batch_size, -1, -1, -1)
            enc = torch.cat([enc, null_slots], 0)

            # if drop_mask is not None:
            #     null_cond_mask = torch.ones_like(drop_mask)  # keep all
            #     drop_mask = torch.cat([drop_mask, null_cond_mask], 0)

            model_kwargs = dict(z=enc, cfg_scale=cfg)
            sample_fn = self.decoder.forward_with_cfg
        else:
            model_kwargs = dict(z=enc)
            sample_fn = self.decoder.forward

        # * diffusion sampling
        if self.diffusion_type == "diffusion":
            samples = self.gen_diffusion.p_sample_loop(
                sample_fn,
                z.shape,
                z,
                clip_denoised=True,
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

        return samples

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()
        if isinstance(self.quantizer, nn.Module):
            self.quantizer.eval()


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
    from src.stage1.cosmos.networks.configs import continuous_image_diff

    torch.cuda.set_device(1)
    tokenizer = TwoDimDiffusionSlots(
        tokenizer_cfg=continuous_image_diff,
        img_channels=12,
        enc_img_size=512,
        use_repa=False,
        diffusion_type="fm",
        compile_model=False,
        fm_options=dict(
            path_type="Linear",
            prediction="velocity",
            train_eps=1e-4,
            sample_eps=1e-4,
        ),
        diffusion_options=dict(timestep_respacing="", noise_schedule="linear", learn_sigma=False),
        quantizer_type=None,
        fm_sample_type="ode",
    ).cuda()
    tokenizer.encoder = tokenizer.encoder.to(torch.bfloat16)
    tokenizer.decoder = tokenizer.decoder.to(torch.bfloat16)
    x = torch.randn(1, 12, 512, 512).cuda().to(torch.bfloat16)
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
            print(f"Memory allocated after forward pass: {allocated_memory / 1024**2:.2f} MB")
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
            y = tokenizer(x, sample=True, epoch=None, inference_with_n_slots=-1, cfg=2.0)[0]
        print(y.shape)

    # run_forward_backward()
    run_sampling()

# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from typing import cast
from loguru import logger
from peft import PeftModel
from diffusers import AutoencoderDC, AutoencoderKL
from mmengine.registry import Registry
from termcolor import colored
from transformers import AutoModelForCausalLM, AutoTokenizer, T5EncoderModel, T5Tokenizer
from transformers import logging as transformers_logging

from src.stage1.cosmos.cosmos_tokenizer import ContinuousImageTokenizer
from src.stage2.generative.Sana.diffusion.model.dc_ae.efficientvit.ae_model_zoo import DCAE_HF
from src.stage2.generative.Sana.diffusion.model.utils import set_fp32_attention, set_grad_checkpoint


MODELS = Registry("models")

transformers_logging.set_verbosity_error()


def _match_dim(x: Tensor) -> Tensor:
    if x.numel() > 1 and x.ndim == 1:
        x = x[None].view(-1, -1, 1, 1)
    return x


class CosmosRSVAE(nn.Module):
    """
    Cosmos_RS VAE wrapper (encode/decode) consistent with
    `src/stage2/generative/Sana/diffusion/model/builder.py` cosmos_RS branch.
    """

    latent_channels: int = 16
    spatial_compression: int = 8

    def __init__(
        self,
        *,
        model_path: str,
        lora_path: str | None = None,
        device: str | torch.device = "cuda",
        dtype: str = "bf16",
        tokenizer_overrides: dict[str, Any] | None = None,
        scale_shift_type: str | None = None,
        scaling_factor: list[float] | float | None = None,
        shift_factor: list[float] | float | None = None,
    ) -> None:
        super().__init__()
        if dtype not in {"bf16", "fp32"}:
            raise ValueError(f"Unsupported {dtype=}, expected 'bf16' or 'fp32'.")
        self._dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
        self._device = torch.device(device)

        model_cfg = dict(
            attn_resolutions=[32],
            channels=128,
            channels_mult=[2, 4, 4],
            dropout=0.0,
            in_channels=512,
            spatial_compression=self.spatial_compression,
            num_res_blocks=2,
            out_channels=512,
            resolution=1024,
            z_channels=256,
            latent_channels=self.latent_channels,
            act_checkpoint=False,
            norm_type="rmsnorm2d",
            act_type="silu",
            block_name="res_block",
            use_residual_factor=False,
            patch_method="haar",
            patch_size=1,
            attn_type="none",
            padding_mode="reflect",
            adaptive_mode="interp",
        )
        if tokenizer_overrides:
            model_cfg.update(tokenizer_overrides)

        self.ae = ContinuousImageTokenizer.create_model(
            model=dict(**model_cfg),
            uni_path=model_path,
            loading_type="pretrained",
            qunatizer_type=None,
            hook_for_repa=False,
            use_repa_loss=False,
            use_vf_loss=False,
            vf_on_z_or_module="z",
            z_factor=1,
        )
        logger.success("Load base VAE from {}".format(model_path))

        # lora load
        if lora_path is not None:
            peft_model = PeftModel.from_pretrained(self.ae, model_id=lora_path, ignore_mismatched_sizes=True)
            if hasattr(peft_model, "merge_and_unload"):
                self.ae = peft_model.merge_and_unload()
            else:
                peft_model.merge_adapter()
                self.ae = peft_model
            self.ae.eval()
            logger.success("Load LoRA from {} and fused into base model".format(lora_path))
            logger.info(f"LoRA fused model type: {type(self.ae).__name__}")
        self.ae = self.ae.to(device=self._device, dtype=self._dtype).eval()

        if scaling_factor is None or shift_factor is None:
            sf, sh = self._get_scale_shift(ae_type=scale_shift_type)
        else:
            sf = list(scaling_factor) if isinstance(scaling_factor, Sequence) else scaling_factor
            sh = list(shift_factor) if isinstance(shift_factor, Sequence) else shift_factor
        sf = torch.tensor(sf, device=self._device, dtype=self._dtype)
        sh = torch.tensor(sh, device=self._device, dtype=self._dtype)
        if sf.numel() > 1:
            sf = sf.view(1, self.latent_channels, 1, 1)
        if sh.numel() > 1:
            sh = sh.view(1, self.latent_channels, 1, 1)

        self.register_buffer("scaling_factor", sf, persistent=False)
        self.register_buffer("shift_factor", sh, persistent=False)

        logger.info(f"CosmosRSVAE device={self._device}, dtype={self._dtype}")
        logger.info(f"Use scaling factor: {sf}")
        logger.info(f"Use shift factor: {sh}")

    @torch.no_grad()
    def encode(self, images: Tensor, no_std: bool = False) -> Tensor:
        scaling_factor_buf = cast(Tensor, self.scaling_factor)
        shift_factor_buf = cast(Tensor, self.shift_factor)

        images = images.to(dtype=self._dtype, device=scaling_factor_buf.device)
        scaling_factor = _match_dim(scaling_factor_buf.to(images))
        shift_factor = _match_dim(shift_factor_buf.to(images))

        with torch.autocast(
            device_type=images.device.type,
            dtype=torch.bfloat16 if self._dtype == torch.bfloat16 else torch.float32,
        ):
            z = self.ae.encode(images)

        if isinstance(z, tuple):
            z = z[0]
        elif isinstance(z, dict):
            z = z["latent"]
        elif torch.is_tensor(z):
            pass
        else:
            raise TypeError(f"Unexpected encode output type: {type(z)}")
        if not no_std:
            z = z.sub(shift_factor.to(z)).div(scaling_factor.to(z))
        return z

    @torch.no_grad()
    def decode(self, latent: Tensor, *, inp_shape: torch.Size | int, no_std: bool = False) -> Tensor:
        return self._decode_impl(latent, inp_shape=inp_shape, no_std=no_std)

    def decode_with_grad(self, latent: Tensor, *, inp_shape: torch.Size | int, no_std: bool = False) -> Tensor:
        """Decode latent with gradient support.

        This is used for pixel-space supervision: gradients should flow to the latent
        (and thus to the cloud-removal model), while VAE parameters remain frozen.
        """
        return self._decode_impl(latent, inp_shape=inp_shape, no_std=no_std)

    def _decode_impl(self, latent: Tensor, *, inp_shape: torch.Size | int, no_std: bool) -> Tensor:
        scaling_factor_buf = cast(Tensor, self.scaling_factor)
        shift_factor_buf = cast(Tensor, self.shift_factor)

        latent = latent.to(dtype=self._dtype, device=scaling_factor_buf.device)
        scaling_factor = _match_dim(scaling_factor_buf.to(latent))
        shift_factor = _match_dim(shift_factor_buf.to(latent))

        if not no_std:
            latent = latent.mul(scaling_factor.to(latent)).add(shift_factor.to(latent))
        with torch.autocast(
            device_type=latent.device.type,
            dtype=torch.bfloat16 if self._dtype == torch.bfloat16 else torch.float32,
        ):
            decoded = self.ae.decode(latent, inp_shape=inp_shape)

        if isinstance(decoded, tuple):
            decoded = decoded[0]
        elif isinstance(decoded, dict):
            decoded = decoded["recon"]
        elif torch.is_tensor(decoded):
            pass
        else:
            raise TypeError(f"Unexpected decode output type: {type(decoded)}")

        return decoded

    def _get_scale_shift(self, ae_type: str | None = "ae"):
        if ae_type is None:
            ae_type = "lcr"

        # ae and lcr reg: two version of pretrained vae
        if ae_type == "lcr":
            shift = [
                -1.4291600526869297,
                -0.4082975222915411,
                -1.0497894948720932,
                -1.4701983284950257,
                -1.7765559741854668,
                -0.8735650272667408,
                1.1257692495174707,
                0.039055027384310964,
                -0.7113604667782784,
                -0.13758214071393013,
                0.6148364602774382,
                -0.6186087974905968,
                -0.7121491965651512,
                1.2216752705536782,
                -1.3880603381455876,
                -0.45283570379018784,
            ]
            scale = [
                0.42588172074344655,
                0.8176063984445395,
                0.61324094397192,
                0.4221144967798025,
                0.7483219732102587,
                0.458599659957132,
                0.7212014900244281,
                0.5361430005310517,
                0.45571632195680367,
                0.7655523709791279,
                0.4172092222154172,
                0.5363149351470319,
                0.7398311229856277,
                0.691116324846224,
                0.7972184947063319,
                0.8416582425637736,
            ]
        elif ae_type == "ae":
            shift = [
                -0.3395568211376667,
                -0.22520461447536946,
                0.0372724512219429,
                -0.10425473003648222,
                -0.12993480741977692,
                0.15024892359972,
                0.6248297291994095,
                0.1264730566367507,
                -0.24228530898690223,
                0.09681867036037148,
                0.11547808751463891,
                -0.15777894280850888,
                -0.09696531526744366,
                0.25315030217170714,
                -0.21809950307011605,
                -0.16787929370999335,
            ]
            scale = [
                0.11568219267030617,
                0.19795570742383053,
                0.1653055727663197,
                0.10820484436353721,
                0.1673998085487709,
                0.07892267036023079,
                0.23162721283100093,
                0.1229123996528739,
                0.10866767695586152,
                0.18754874772861008,
                0.0867098794162515,
                0.17479499629405454,
                0.1599094335227425,
                0.15633937637654485,
                0.20044135354362982,
                0.15534570714645543,
            ]
        else:
            raise ValueError(f"Unsupported ae_type: {ae_type}. Expected 'lcr' or 'ae'.")

        return scale, shift


def build_model(cfg, use_grad_checkpoint=False, use_fp32_attention=False, gc_step=1, **kwargs):
    if isinstance(cfg, str):
        cfg = dict(type=cfg)
    # Import model modules to ensure they are registered into MODELS.
    import src.stage2.generative.Sana.diffusion.model.nets  # noqa: F401

    model = MODELS.build(cfg, default_args=kwargs)

    if use_grad_checkpoint:
        set_grad_checkpoint(model, gc_step=gc_step)
    if use_fp32_attention:
        set_fp32_attention(model)
    return model


def get_tokenizer_and_text_encoder(
    name: str = "T5",
    device: str = "cuda",
    *,
    cache_dir: str | None = None,
    local_files_only: bool = False,
    pretrained_path: str | None = None,
):
    text_encoder_dict = {
        "T5": "DeepFloyd/t5-v1_1-xxl",
        "T5-small": "google/t5-v1_1-small",
        "T5-base": "google/t5-v1_1-base",
        "T5-large": "google/t5-v1_1-large",
        "T5-xl": "google/t5-v1_1-xl",
        "T5-xxl": "google/t5-v1_1-xxl",
        "gemma-2b": "google/gemma-2b",
        "gemma-2b-it": "google/gemma-2b-it",
        "gemma-2-2b": "google/gemma-2-2b",
        "gemma-2-2b-it": "Efficient-Large-Model/gemma-2-2b-it",
        "gemma-2-9b": "google/gemma-2-9b",
        "gemma-2-9b-it": "google/gemma-2-9b-it",
        "Qwen2-0.5B-Instruct": "Qwen/Qwen2-0.5B-Instruct",
        "Qwen2-1.5B-Instruct": "Qwen/Qwen2-1.5B-Instruct",
    }
    assert name in list(text_encoder_dict.keys()), f"not support this text encoder: {name}"
    model_id = pretrained_path or text_encoder_dict[name]
    if "T5" in name:
        tokenizer = T5Tokenizer.from_pretrained(model_id, cache_dir=cache_dir, local_files_only=local_files_only)
        text_encoder = T5EncoderModel.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            torch_dtype=torch.float16,
        ).to(device)
    elif "gemma" in name or "Qwen" in name:
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, local_files_only=local_files_only)
        tokenizer.padding_side = "right"
        text_encoder = (
            AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                torch_dtype=torch.bfloat16,
            )
            .get_decoder()
            .to(device)
        )
    else:
        print("error load text encoder")
        exit()

    return tokenizer, text_encoder


def get_vae(name, model_path, device="cuda", **kwargs):
    logger.info(f"Loading VAE: {name} with {kwargs=}")

    if name == "sdxl" or name == "sd3":
        model_name = "stabilityai/stable-diffusion-xl-base-1.0"
        vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(device).to(torch.float16)
        if name == "sdxl":
            vae.config.shift_factor = 0
        return vae
    elif name == "sd35":
        model_name = "stabilityai/stable-diffusion-3.5-large-turbo"
        vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(device).to(torch.bfloat16)
        return vae
    elif name == "flux":
        model_name = "black-forest-labs/FLUX.1-dev"
        vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(device).to(torch.bfloat16)
        return vae
    elif "dc-ae" in name:
        print(colored(f"[DC-AE] Loading model from {model_path}", attrs=["bold"]))
        model_name = "mit-han-lab/dc-ae-f32c32-sana-1.0"
        dc_ae = DCAE_HF.from_pretrained(model_name).to(device).eval()
        return dc_ae
    elif "AutoencoderDC" in name:
        print(colored(f"[AutoencoderDC] Loading model from {model_path}", attrs=["bold"]))
        model_name = "Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers"
        dc_ae = AutoencoderDC.from_pretrained(model_name, subfolder="vae").to(device).eval()
        return dc_ae

    # remote sensing cosmos AE
    elif "cosmos_RS" == name:
        print(colored(f"[Cosmos RS] Loading model from {model_path}", attrs=["bold"]))
        from src.stage1.cosmos.cosmos_tokenizer import ContinuousImageTokenizer
        from src.stage1.cosmos.lora_mixin import TokenizerLoRAMixin

        cosmos_ae = ContinuousImageTokenizer.create_model(
            model=dict(
                attn_resolutions=[32],
                channels=128,
                channels_mult=[2, 4, 4],
                dropout=0.0,
                in_channels=512,
                spatial_compression=8,
                num_res_blocks=2,
                out_channels=512,
                resolution=1024,
                z_channels=256,
                latent_channels=16,
                act_checkpoint=False,
                norm_type="rmsnorm2d",
                act_type="silu",
                block_name="res_block",
                use_residual_factor=False,
                patch_method="haar",
                patch_size=1,
                attn_type="none",
                padding_mode="reflect",
                adaptive_mode="interp",
            ),
            uni_path=model_path,
            loading_type="pretrained",
            qunatizer_type=None,
            hook_for_repa=False,
            use_repa_loss=False,
            use_vf_loss=False,
            vf_on_z_or_module="z",
            z_factor=1,
        )

        # deprecated ----
        # cosmos_ae_lora = TokenizerLoRAMixin(cosmos_ae, **kwargs)
        # cosmos_ae_lora.requires_grad_(False)
        # return cosmos_ae.to(device=device, dtype=torch.bfloat16).eval()
        # -----

        cosmos_ae = cosmos_ae.to(device=device, dtype=torch.bfloat16).eval()
        # fmt: off
        cosmos_ae.scaling_factor = torch.tensor(
            [0.470703125, 0.95703125, 0.63671875, 0.455078125, 0.74609375, 0.53515625, 0.8359375, 0.671875, 0.62890625, 0.375, 0.51171875, 0.69921875, 0.447265625, 0.66015625, 0.65234375, 0.53515625],
            device=device,
        ).view(1, 16, 1, 1)
        cosmos_ae.shift_factor = torch.tensor(
            [-1.2734375, 0.193359375, -1.1171875, -1.0859375, -1.78125, -0.52734375, 0.3984375, -0.5, -0.482421875, -0.09375, 0.1689453125, -0.38671875, -0.8046875, 0.49609375, -0.62109375, -0.2578125],
            device=device,
        ).view(1, 16, 1, 1)
        # fmt: on
        return cosmos_ae.eval()

    elif "cosmos_RS_lora" == name:
        vae = CosmosRSVAE(
            model_path=model_path,
            lora_path=kwargs.get("lora_path"),
            device=device,
            dtype=kwargs.get("dtype", "bf16"),
            scale_shift_type=kwargs.get("scale_shift_type"),
            scaling_factor=kwargs.get("scaling_factor"),
            shift_factor=kwargs.get("shift_factor"),
        )
        return vae.eval()

    else:
        _msg = "error load vae"
        logger.critical(_msg)
        raise ValueError(_msg)


def match_dim(x: torch.Tensor):
    if x.numel() > 1 and x.ndim == 1:
        # channel scaling
        x = x[None].view(-1, -1, 1, 1)
    return x


def vae_encode(name, vae, images, sample_posterior, device: torch.device):
    if name == "sdxl" or name == "sd3" or name == "sd35" or name == "flux":
        posterior = vae.encode(images.to(device)).latent_dist
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        z = (z - vae.config.shift_factor) * vae.config.scaling_factor
    elif "dc-ae" in name:
        ae = vae
        scaling_factor = ae.cfg.scaling_factor if ae.cfg.scaling_factor else 0.41407
        z = ae.encode(images.to(device))
        z = z * scaling_factor
    elif "AutoencoderDC" in name:
        ae = vae
        scaling_factor = ae.config.scaling_factor if ae.config.scaling_factor else 0.41407
        z = ae.encode(images.to(device))[0]
        z = z * scaling_factor

    # --------- remote sensing cosmos AE --------
    elif name in {"cosmos_RS", "cosmos_RS_lora"}:
        ae = vae
        images = images.to(device, torch.bfloat16)
        scaling_factor = torch.as_tensor(ae.scaling_factor).to(images)
        shift_factor = torch.as_tensor(ae.shift_factor).to(images)

        scaling_factor, shift_factor = map(match_dim, (scaling_factor, shift_factor))
        assert scaling_factor is not None and shift_factor is not None, (
            "scaling_factor and shift_factor must be set for cosmos_RS, please check the class attribution"
        )

        with torch.autocast(device_type=images.device.type, dtype=torch.bfloat16):
            z = ae.encode(images)
        if isinstance(z, tuple):
            z = z[0]
        elif isinstance(z, dict):
            z = z["latent"]
        elif isinstance(z, Tensor):
            pass
        else:
            raise ValueError("z must be a tuple or dict")
        z.sub_(shift_factor.to(z)).div_(scaling_factor.to(z))
    else:
        print("error load vae")
        exit()
    return z


def vae_decode(name, vae, latent, **kwargs):
    if name == "sdxl" or name == "sd3" or name == "sd35" or name == "flux":
        latent = (latent.detach() / vae.config.scaling_factor) + vae.config.shift_factor
        samples = vae.decode(latent).sample
    elif "dc-ae" in name:
        ae = vae
        vae_scale_factor = (
            2 ** (len(ae.config.encoder_block_out_channels) - 1)
            if hasattr(ae, "config") and ae.config is not None
            else 32
        )
        scaling_factor = ae.cfg.scaling_factor if ae.cfg.scaling_factor else 0.41407
        if latent.shape[-1] * vae_scale_factor > 4000 or latent.shape[-2] * vae_scale_factor > 4000:
            from patch_conv import convert_model

            ae = convert_model(ae, splits=4)
        samples = ae.decode(latent.detach() / scaling_factor)
    elif "AutoencoderDC" in name:
        ae = vae
        scaling_factor = ae.config.scaling_factor if ae.config.scaling_factor else 0.41407
        try:
            samples = ae.decode(latent / scaling_factor, return_dict=False)[0]
        except torch.cuda.OutOfMemoryError as e:
            print("Warning: Ran out of memory when regular VAE decoding, retrying with tiled VAE decoding.")
            ae.enable_tiling(tile_sample_min_height=1024, tile_sample_min_width=1024)
            samples = ae.decode(latent / scaling_factor, return_dict=False)[0]

    elif name in {"cosmos_RS", "cosmos_RS_lora"}:
        from src.stage1.cosmos.cosmos_tokenizer import ContinuousImageTokenizer

        scaling_factor = torch.as_tensor(vae.scaling_factor).to(latent)
        shift_factor = torch.as_tensor(vae.shift_factor).to(latent)
        vae = cast(CosmosRSVAE | ContinuousImageTokenizer, vae)

        scaling_factor, shift_factor = map(match_dim, (scaling_factor, shift_factor))
        assert scaling_factor is not None and shift_factor is not None, (
            "scaling_factor and shift_factor must be set for cosmos_RS, please check the class attribution"
        )

        latent = latent.mul(scaling_factor.to(latent)).add(shift_factor.to(latent))
        with torch.autocast(device_type=latent.device.type, dtype=torch.bfloat16):
            shape = kwargs.get("input_shape", 3)  # [b,c,h,w]
            samples = vae.decode(latent, inp_shape=shape)

        if isinstance(samples, tuple):
            samples = samples[0]
        elif isinstance(samples, dict):
            samples = samples["recon"]
        elif torch.is_tensor(samples):
            pass
        else:
            raise ValueError("samples must be a tensor or a dict or a tuple, please check the class attribution")
    else:
        print("error load vae")
        exit()
    return samples


if __name__ == "__main__":
    from torchmetrics.image import PeakSignalNoiseRatio

    from src.data.litdata_hyperloader import get_fast_test_hyper_litdata_load

    # vae = get_vae("cosmos_RS", model_path="runs/pretrained/VAEInterp-f8c16.safetensors")
    vae_name = "AutoencoderDC"
    print(f"Load VAE {vae_name} ...")
    vae = get_vae(vae_name, "")
    vae = vae.cuda()
    print("Load VAE done.")

    _, dl = get_fast_test_hyper_litdata_load("SAM270k", 1)

    # sample = next(iter(dl))
    metric = PeakSignalNoiseRatio(data_range=1.0).cuda()
    for i, sample in enumerate(dl):
        images = sample["img"].cuda()
        print("Get data done.", images.shape)
        with torch.no_grad():
            with torch.autocast("cuda", torch.bfloat16):
                z = vae_encode(vae_name, vae=vae, images=images, sample_posterior=None, device="cuda")
                print(f"Encoding z info: min: {z.min()}, max: {z.max()}, std: {z.std()}")
                images_rec = vae_decode(vae_name, vae=vae, latent=z)
        images_rec = images_rec.to(torch.float32)
        if i >= 200:
            break

        # PSNR
        images.add_(1).div_(2)
        images_rec.add_(1).div_(2)
        psnr = metric(images_rec, images)
        # print("PSNR:", psnr.item())
    print("Average PSNR:", metric.compute())

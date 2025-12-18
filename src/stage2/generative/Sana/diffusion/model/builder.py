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

import torch
from diffusers import AutoencoderDC
from diffusers.models import AutoencoderKL
from mmengine.registry import Registry
from termcolor import colored
from transformers import AutoModelForCausalLM, AutoTokenizer, T5EncoderModel, T5Tokenizer
from transformers import logging as transformers_logging

from src.stage2.generative.Sana.diffusion.model.dc_ae.efficientvit.ae_model_zoo import DCAE_HF
from src.stage2.generative.Sana.diffusion.model.utils import set_fp32_attention, set_grad_checkpoint

MODELS = Registry("models")

transformers_logging.set_verbosity_error()


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
    elif "cosmos_RS" in name:
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
                norm_type="gn",
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
        # cosmos_ae_lora = TokenizerLoRAMixin(cosmos_ae, **kwargs)
        # cosmos_ae_lora.requires_grad_(False)
        # return cosmos_ae.to(device=device, dtype=torch.bfloat16).eval()
        cosmos_ae = cosmos_ae.to(device=device, dtype=torch.bfloat16).eval()
        cosmos_ae.scaling_factor = torch.tensor(
            [
                3.796875,
                3.0,
                3.640625,
                3.390625,
                3.640625,
                2.921875,
                4.15625,
                3.984375,
                2.65625,
                2.9375,
                4.375,
                5.125,
                7.03125,
                2.734375,
                2.703125,
                4.96875,
            ],
            device=device,
        ).view(1, 16, 1, 1)
        cosmos_ae.shift_factor = torch.tensor(
            [
                -1.0546875,
                -1.1328125,
                -2.171875,
                0.07666015625,
                -0.7109375,
                0.3984375,
                -1.6171875,
                -0.453125,
                0.65625,
                2.828125,
                0.09619140625,
                -1.15625,
                -1.4453125,
                0.8984375,
                3.421875,
                -1.078125,
            ],
            device=device,
        ).view(1, 16, 1, 1)
        return cosmos_ae
    else:
        print("error load vae")
        exit()


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

    ######## remote sensing cosmos AE
    elif "cosmos_RS" in name:
        ae = vae
        images = images.to(device, torch.bfloat16)
        scaling_factor = torch.as_tensor(ae.scaling_factor).to(images)
        shift_factor = torch.as_tensor(ae.shift_factor).to(images)

        scaling_factor, shift_factor = map(match_dim, (scaling_factor, shift_factor))
        assert scaling_factor is not None and shift_factor is not None, (
            "scaling_factor and shift_factor must be set for cosmos_RS, please check the class attribution"
        )

        with torch.autocast(device_type=str(device), dtype=torch.bfloat16):
            z = ae.encode(images)
        if isinstance(z, tuple):
            z = z[0]
        elif isinstance(z, dict):
            z = z["latent"]
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

    elif "cosmos_RS" in name:
        scaling_factor = torch.as_tensor(vae.scaling_factor).to(latent)
        shift_factor = torch.as_tensor(vae.shift_factor).to(latent)

        scaling_factor, shift_factor = map(match_dim, (scaling_factor, shift_factor))
        assert scaling_factor is not None and shift_factor is not None, (
            "scaling_factor and shift_factor must be set for cosmos_RS, please check the class attribution"
        )

        latent.mul_(scaling_factor.to(latent)).add_(shift_factor.to(latent))
        with torch.autocast(device_type=str(latent.device), dtype=torch.bfloat16):
            shape = kwargs.get("input_shape", 3)  # [b,c,h,w]
            samples = vae.decode(latent, shape)

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

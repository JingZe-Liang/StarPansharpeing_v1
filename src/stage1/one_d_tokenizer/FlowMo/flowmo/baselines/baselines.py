"""
In order to minimize outside dependencies, many imports here don't happen at
the top level.

You may find you need to install additional deps (i.e. diffusers) to run the
baselines here.
"""

import torch


def load_cosmos(arch):
    import sys

    sys.path.insert(-1, "path_to_cosmos/Cosmos-Tokenizer")
    from cosmos_tokenizer.image_lib import ImageTokenizer

    if arch == "cosmos_8x8":
        model_name = "Cosmos-0.1-Tokenizer-DI8x8"
    elif arch == "cosmos_16x16":
        model_name = "Cosmos-0.1-Tokenizer-DI16x16"
    else:
        raise NotImplementedError

    encoder_ckpt = f"pretrained_ckpts/{model_name}/encoder.jit"
    decoder_ckpt = f"pretrained_ckpts/{model_name}/decoder.jit"

    tokenizer = ImageTokenizer(
        checkpoint_enc=encoder_ckpt,
        checkpoint_dec=decoder_ckpt,
        device="cuda",
        dtype="bfloat16",
    )

    @torch.no_grad()
    def reconstruct_fn(images):
        # images in (-1, 1) bchw
        images = (
            ((images / 2 + 0.5) * 255)
            .clip(0, 255)
            .to(torch.uint8)
            .permute((0, 2, 3, 1))
            .cpu()
            .numpy()
        )
        batched_output_image = tokenizer(images)

        batched_output_image = (
            torch.from_numpy(batched_output_image)
            .cuda()
            .permute((0, 3, 1, 2))
            .to(torch.float32)
            / 255
        )
        batched_output_image = batched_output_image * 2 - 1
        return batched_output_image.clip(-1, 1)

    return tokenizer, reconstruct_fn


def load_llamagen(arch):
    from transformer_enc.baselines import llamagen

    if arch == "llamagen32":
        llamagen_tokenizer = llamagen.VQ_8()
        llamagen_tokenizer.load_state_dict(torch.load("vq_ds8_c2i.pt")["model"])
    elif arch == "llamagen16":
        llamagen_tokenizer = llamagen.VQ_16()
        llamagen_tokenizer.load_state_dict(torch.load("vq_ds16_c2i.pt")["model"])
    else:
        raise NotImplementedError

    llamagen_tokenizer = llamagen_tokenizer.cuda().eval()

    @torch.no_grad()
    def reconstruct_fn(images, **kwargs):
        images_rec_llamagen, _ = llamagen_tokenizer(images.cuda())
        return images_rec_llamagen

    return llamagen_tokenizer, reconstruct_fn


def load_flux():
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
        tokenizer=None,
        tokenizer_2=None,
        text_encoder=None,
        transformer=None,
    )
    vae = pipe.vae.cuda()

    @torch.no_grad()
    def reconstruct_fn(images, **kwargs):
        # images are bchw in [-1, 1]
        encoded = vae.encode(images.cuda().to(torch.bfloat16))
        encoded = encoded.latent_dist.sample()
        decoded = vae.decode(encoded).sample
        images_rec_vae = decoded.clip(-1, 1)
        return images_rec_vae.to(torch.float32)

    return vae, reconstruct_fn


def load_sdxl():
    from diffusers.models import AutoencoderKL

    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").cuda()

    def reconstruct_fn(images, **kwargs):
        # images are bchw in [-1, 1]
        encoded = vae.encode(images)
        encoded = encoded.latent_dist.sample()
        decoded = vae.decode(encoded).sample
        images_rec_vae = decoded.clip(-1, 1)
        return images_rec_vae

    return vae, reconstruct_fn


def load_titok(arch="l32"):
    from transformer_enc.baselines import titok

    full_arch = next(
        iter(
            full_arch
            for full_arch in [
                "tokenizer_titok_l32_imagenet",
                "tokenizer_titok_b64_imagenet",
                "tokenizer_titok_s128_imagenet",
            ]
            if arch in full_arch
        )
    )
    print(full_arch)

    titok_tokenizer = titok.TiTok.from_pretrained(f"yucornetto/{full_arch}")
    titok_tokenizer = titok_tokenizer.eval()
    titok_tokenizer = titok_tokenizer.requires_grad_(False).cuda()

    @torch.no_grad()
    def reconstruct_fn(images, **kwargs):
        encoded_tokens = titok_tokenizer.encode(images.to("cuda") / 2 + 0.5)[1][
            "min_encoding_indices"
        ]
        decoded = titok_tokenizer.decode_tokens(encoded_tokens) * 2 - 1
        return decoded

    return titok_tokenizer, reconstruct_fn

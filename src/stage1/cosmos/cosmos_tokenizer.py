import sys
from collections import OrderedDict, namedtuple
from types import SimpleNamespace
from typing import Literal

import torch
from loguru import logger as logging
from torch import nn

sys.path.insert(0, __file__[: __file__.find("src")])
from src.stage1.cosmos.inference.utils import load_jit_model_shape_matched
from src.stage1.cosmos.modules.layers2d import Decoder, Encoder, RMSNorm2d


class ContinuousImageTokenizer(nn.Module):
    def __init__(
        self,
        z_channels: int,
        z_factor: int = 1,
        latent_channels: int = 8,
        loading_type: Literal["pretrained", "nvidia"] | None = "pretrained",
        **kwargs,
    ) -> None:
        super().__init__()
        tokenizer_cfg = dict(
            z_channels=z_channels,
            z_factor=z_factor,
            latent_channels=latent_channels,
            **kwargs,
        )
        self.loading_type = loading_type
        self.name = kwargs.get("name", "ContinuousImageTokenizer")
        self.latent_channels = latent_channels
        enc_path = kwargs.pop("enc_path", "")
        dec_path = kwargs.pop("dec_path", "")

        # pretrained encoder and decoder
        if loading_type == "nvidia":
            assert enc_path.endswith(".jit") and dec_path.endswith(".jit")
            # pretrained model from NVIDIA cosmos tokenizer
            assert not kwargs.get(
                "norm_in_quant_conv", False
            ), "norm_in_quant_conv is not supported for nvidia pretrained model settings, trian it from scratch"

            logging.debug(
                f"start from the pretrained model, cosmos tokenizer cfg is {tokenizer_cfg}"
            )
            enc_jit, dec_jit = self.load_pretrained(enc_path, dec_path, tokenizer_cfg)

            # split the encoder and decoder
            self.encoder = enc_jit[0]
            self.quant_conv = enc_jit[1]

            self.decoder = dec_jit[1]
            self.post_quant_conv = dec_jit[0]
        else:
            # encoder and decoder
            # not combile the encoder, for FSDP wrap
            encoder = Encoder(z_channels=z_factor * z_channels, **kwargs)
            decoder = Decoder(z_channels=z_channels, **kwargs)
            if kwargs.get("norm_in_quant_conv", False):
                quant_conv = nn.Sequential(
                    RMSNorm2d(z_factor * z_channels),
                    torch.nn.Conv2d(
                        z_factor * z_channels, z_factor * latent_channels, 1
                    ),
                )
            else:
                quant_conv = torch.nn.Conv2d(
                    z_factor * z_channels, z_factor * latent_channels, 1
                )
            post_quant_conv = torch.nn.Conv2d(latent_channels, z_channels, 1)
            self.encoder = self.encoder_jit(encoder, quant_conv)
            self.decoder = self.decoder_jit(decoder, post_quant_conv)

            if kwargs.get("norm_in_quant_conv", False):
                assert (
                    enc_path == "" and dec_path == ""
                ), "norm_in_quant_conv is not supported for pretrained settings, trian it from scratch"
            self.load_pretrained(enc_path, dec_path)

        num_parameters = sum(param.numel() for param in self.parameters())
        logging.info(f"model={self.name}, num_parameters={num_parameters:,}")
        logging.info(
            f"z_channels={z_channels}, latent_channels={self.latent_channels}."
        )

    def encoder_jit(self, encoder, quant_conv):
        return nn.Sequential(
            OrderedDict(
                [
                    ("encoder", encoder),
                    ("quant_conv", quant_conv),
                    # ("distribution", self.distribution),
                ]
            )
        )

    def decoder_jit(self, decoder, post_quant_conv):
        return nn.Sequential(
            OrderedDict(
                [
                    ("post_quant_conv", post_quant_conv),
                    ("decoder", decoder),
                ]
            )
        )

    def get_last_layer(self):
        if not self.decoder._wrap_fsdp_last_layer:
            return self.decoder.conv_out.weight
        else:
            return self.decoder.conv_out.wrap_mod.weight

    def encode(self, x):
        h = self.encoder(x)
        # moments = self.quant_conv(h)

        return h

    def decode(self, z):
        # z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input):
        # latent, posteriors = self.encode(input)
        latent = self.encode(input)
        dec = self.decode(latent)

        return dec

    def load_pretrained(self, enc_path: str, dec_path: str, tokenizer_cfg=None):
        if enc_path == "" or dec_path == "":
            return

        if self.loading_type == "nvidia":
            assert (
                tokenizer_cfg is not None
            ), "tokenizer_cfg is required when loading the nvidia pretrained tokenizer"
            logging.info(
                f"Loading pretrained encoder from {enc_path} for NVIDIA pretrained model"
            )
            encoder, _enc_model_mody_keys = load_jit_model_shape_matched(
                jit_model_path=enc_path,
                device="cuda",
                tokenizer_config=tokenizer_cfg,
                part="encoder",
            )
            logging.info(
                f"Loading pretrained decoder from {dec_path} for NVIDIA pretrained model"
            )
            decoder, _dec_model_mody_keys = load_jit_model_shape_matched(
                jit_model_path=dec_path,
                device="cuda",
                tokenizer_config=tokenizer_cfg,
                part="decoder",
            )

            logging.warning(
                f"not compatible for pretraine models: \n",
                f"encoder: {_enc_model_mody_keys}\n",
                f"decoder: {_dec_model_mody_keys}\n",
            )
            return encoder, decoder
        else:
            import accelerate

            assert enc_path.endswith("safetensors") and dec_path.endswith(
                "safetensors"
            ), "only support safetensors for now"
            logging.info(
                "pretrained model is pretrained on hyperspectral images, "
                "for now is used to finetune on the other dataset"
            )
            enc_sd = accelerate.utils.load_state_dict(enc_path)
            dec_sd = accelerate.utils.load_state_dict(dec_path)
            _enc_missing, _enc_unexp = self.encoder.load_state_dict(
                enc_sd, strict=False
            )
            _dec_missing, _dec_unexp = self.decoder.load_state_dict(
                dec_sd, strict=False
            )

            # if is peft loading, make sure the encoder is perfectly loaded
            # if self.loading_type == "peft":
            #     if len(_enc_missing) > 0:
            #         assert _enc_missing == "conv_in.weight"
            #     assert len(_enc_unexp) == 0

            logging.warning(
                f"load pretrained model done. \n"
                f"encoder: missing keys {_enc_missing}, unexpected keys {_enc_unexp}\n"
                f"decoder: missing keys {_dec_missing}, unexpected keys {_dec_unexp}\n"
            )

    @property
    def _no_split_modules(self):
        return ["wrap_fsdp_last_layer"]

    def peft_first_last_convs_moduel_names(self):
        return [
            "encoder.encoder.conv_in",
            "decoder.decoder.conv_out"
            if not self.decoder._wrap_fsdp_last_layer
            else "decoder.decoder.conv_out.wrap_mod",
        ]


if __name__ == "__main__":
    config = {
        "attn_resolutions": [32],
        "channels": 128,
        "channels_mult": [2, 4, 4],
        "dropout": 0.0,
        "in_channels": 8,
        "spatial_compression": 8,
        "num_res_blocks": 2,
        "out_channels": 8,
        "resolution": 1024,
        "patch_size": 4,
        "patch_method": "haar",
        "latent_channels": 16,
        "z_channels": 16,
        "z_factor": 1,
        "name": "CI",
        "formulation": "AE",
        "encoder": "Default",
        "decoder": "Default",
        "act_checkpoint": False,
        # "enc_path": "/Data4/cao/ZiHanCao/exps/Cosmos/checkpoints/Cosmos-0.1-Tokenizer-CI8x8/encoder.jit",
        # "dec_path": "/Data4/cao/ZiHanCao/exps/Cosmos/checkpoints/Cosmos-0.1-Tokenizer-CI8x8/decoder.jit",
        "enc_path": "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/runs/stage1_cosmos/2025-04-08_03-14-32_cosmos_pretrained_f8c16p4_percep_remote_clip_RN50/ema/encoder/model.safetensors",
        "dec_path": "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/runs/stage1_cosmos/2025-04-08_03-14-32_cosmos_pretrained_f8c16p4_percep_remote_clip_RN50/ema/decoder/model.safetensors",
    }

    tokenizer = ContinuousImageTokenizer(**config)

    x = torch.randn(1, 8, 256, 256).to("cuda", torch.bfloat16)
    with torch.autocast("cuda", torch.bfloat16):
        y = tokenizer(x)
        logging.debug(y.shape)

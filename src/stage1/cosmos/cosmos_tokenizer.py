from collections import OrderedDict, namedtuple
from types import SimpleNamespace

import torch
from loguru import logger as logging
from torch import nn

from src.stage1.cosmos.inference.utils import load_jit_model_shape_matched
from src.stage1.cosmos.modules.layers2d import Decoder, Encoder, RMSNorm2d


class ContinuousImageTokenizer(nn.Module):
    def __init__(
        self,
        z_channels: int,
        z_factor: int = 1,
        latent_channels: int = 8,
        **kwargs,
    ) -> None:
        super().__init__()
        tokenizer_cfg = dict(
            z_channels=z_channels,
            z_factor=z_factor,
            latent_channels=latent_channels,
            **kwargs,
        )
        self.name = kwargs.get("name", "ContinuousImageTokenizer")
        self.latent_channels = latent_channels
        enc_path = kwargs.pop("enc_path", "")
        dec_path = kwargs.pop("dec_path", "")

        # pretrained encoder and decoder
        if enc_path.endswith(".jit"):
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
            self.encoder = Encoder(z_channels=z_factor * z_channels, **kwargs)
            self.decoder = Decoder(z_channels=z_channels, **kwargs)
            self.quant_conv = nn.Sequential(
                RMSNorm2d(z_factor * z_channels)
                if kwargs.get("norm_in_quant_conv", False)
                else nn.Identity(),
                torch.nn.Conv2d(z_factor * z_channels, z_factor * latent_channels, 1),
            )
            self.post_quant_conv = torch.nn.Conv2d(latent_channels, z_channels, 1)

            # formulation_name = kwargs.get("formulation", ContinuousFormulation.AE.name)
            # self.distribution = ContinuousFormulation[formulation_name].value()
            # logging.info(
            #     f"{self.name} based on {formulation_name} formulation, with {kwargs}."
            # )

        num_parameters = sum(param.numel() for param in self.parameters())
        logging.info(f"model={self.name}, num_parameters={num_parameters:,}")
        logging.info(
            f"z_channels={z_channels}, latent_channels={self.latent_channels}."
        )

    def encoder_jit(self):
        return nn.Sequential(
            OrderedDict(
                [
                    ("encoder", self.encoder),
                    ("quant_conv", self.quant_conv),
                    # ("distribution", self.distribution),
                ]
            )
        )

    def decoder_jit(self):
        return nn.Sequential(
            OrderedDict(
                [
                    ("post_quant_conv", self.post_quant_conv),
                    ("decoder", self.decoder),
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
        moments = self.quant_conv(h)

        # return self.distribution(moments)

        return moments

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input):
        # latent, posteriors = self.encode(input)
        latent = self.encode(input)
        dec = self.decode(latent)

        return dec

    def load_pretrained(self, enc_path: str, dec_path: str, tokenizer_cfg):
        if enc_path.endswith(".jit"):
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
        else:
            raise RuntimeError(
                "`load_pretrained` function only used to load pretrained from nvidia jit model "
                "for training startup"
            )

        logging.warning(
            f"not compatible for pretraine models: \n",
            f"encoder: {_enc_model_mody_keys}\n",
            f"decoder: {_dec_model_mody_keys}\n",
        )

        return encoder, decoder

    @property
    def _no_split_modules(self):
        return ["wrap_fsdp_last_layer"]


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
        "enc_path": "/Data4/cao/ZiHanCao/exps/Cosmos/checkpoints/Cosmos-0.1-Tokenizer-CI8x8/encoder.jit",
        "dec_path": "/Data4/cao/ZiHanCao/exps/Cosmos/checkpoints/Cosmos-0.1-Tokenizer-CI8x8/decoder.jit",
    }

    tokenizer = ContinuousImageTokenizer(**config)

    x = torch.randn(1, 8, 256, 256).to("cuda", torch.bfloat16)
    with torch.autocast("cuda", torch.bfloat16):
        y = tokenizer(x)
        logging.debug(y.shape)

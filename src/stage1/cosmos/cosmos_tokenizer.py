from collections import OrderedDict, namedtuple

import torch
from loguru import logger as logging
from torch import nn

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
        self.name = kwargs.get("name", "ContinuousImageTokenizer")
        self.latent_channels = latent_channels

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
        return self.decoder.conv_out.weight

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

        # if self.training:
        #     return dict(reconstructions=dec, posteriors=posteriors, latent=latent)
        # return NetworkEval(reconstructions=dec, posteriors=posteriors, latent=latent)

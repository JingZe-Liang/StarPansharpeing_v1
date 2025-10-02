from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..discretization.collections.psd import PowerSphericalDistribution, l2_norm
from .modules.hybrid.vae import ViTDecoder, ViTEncoder
from .modules.proj import build_mlp


@dataclass
class VAEConfig:
    latent_dim: int = 16
    image_size: int = 256
    patch_size: int = 16
    z_channels: int = 512
    cnn_chs: list[int] = field(default_factory=lambda: [64, 64, 128, 256, 512])
    encoder_vit_layers: int = 6
    decoder_vit_layers: int = 12

    use_repa_loss: bool = False
    dino_feature_dim: int = 1024

    scaling_factor: Optional[torch.Tensor] = None
    shift_factor: Optional[torch.Tensor] = None


class VAE(nn.Module):
    _no_split_modules = ["ResDownBlock", "TransformerBlock"]

    _use_repa_loss = False
    _dino_feature_dim = 1024

    # state
    _hook_feature: torch.Tensor | None = None
    z: torch.Tensor | None = None  # the latent z

    def __init__(
        self,
        latent_dim=16,
        image_size=512,
        patch_size=16,
        z_channels=512,
        cnn_chs=[64, 64, 128, 256, 512],
        encoder_vit_layers=6,
        decoder_vit_layers=12,
    ):
        super().__init__()
        self.z_channels = z_channels
        n_head = z_channels // 64
        self.encoder = ViTEncoder(
            n_layers=encoder_vit_layers,
            d_model=z_channels,
            n_heads=n_head,
            cnn_chs=cnn_chs,
            image_size=image_size,
            patch_size=patch_size,
        )
        self.decoder = ViTDecoder(
            n_layers=decoder_vit_layers,
            d_model=z_channels,
            n_heads=n_head,
            cnn_chs=cnn_chs[::-1],
            image_size=image_size,
            patch_size=patch_size,
        )
        self.latent_dim = latent_dim
        self.quant_proj = nn.Linear(z_channels, latent_dim + 1, bias=True)
        self.post_quant_proj = nn.Linear(latent_dim, z_channels, bias=False)

    def initialize_weights(self):
        self.quant_proj.reset_parameters()
        self.post_quant_proj.reset_parameters()
        self.encoder.output.reset_parameters()

    def normalize(self, x):
        x = l2_norm(x)
        x = x * (self.latent_dim**0.5)
        return x

    def encode(self, x):
        x = self.encoder(x)
        x = self.quant_proj(x)
        mu = x[..., :-1]
        kappa = x[..., -1]
        mu = l2_norm(mu)
        kappa = F.softplus(kappa) + 1.0
        qz = PowerSphericalDistribution(mu, kappa)
        loss = qz.kl_to_uniform()
        x = qz.rsample()
        x = x * (self.latent_dim**0.5)
        return x, loss.mean()

    def decode(self, x):
        x = self.post_quant_proj(x)
        dec = self.decoder(x)
        return dec

    def _build_repa_mlp(self):
        self._repa_proj = build_mlp(
            self.z_channels,
            self._dino_feature_dim,
            self._dino_feature_dim,
        )

    # @torch.autocast("cuda", dtype=torch.float16)
    # def get_repa_feature(self):

    # def forward(self, x):
    #     z, qloss = self.encode(x)
    #     x_rec = self.decode(z)
    #     self.z = z
    #     return x_rec, qloss


if __name__ == "__main__":
    """
    python -m src.stage1.cosmos.hybrid_vae
    """
    vae = VAE(patch_size=8)  # 75M params
    from fvcore.nn import parameter_count_table

    print(parameter_count_table(vae))

    vae = vae.to("cuda", torch.bfloat16)
    x = torch.randn(1, 3, 512, 512).to("cuda", torch.bfloat16)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        z, qloss = vae.encode(x)
        print(z.shape, qloss)
        x_recon = vae.decode(z)
        print(x_recon.shape)
        assert x_recon.shape == x.shape

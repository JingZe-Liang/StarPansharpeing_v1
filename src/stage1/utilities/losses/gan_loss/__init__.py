from .hyperspectral_percep_loss import LPIPSHyperpspectralLoss
from .logit_laplace_loss import LogitLaplaceLoss
from .loss import (
    DinoDiscV2,
    NLayerDiscriminator,
    StyleGAN3DDiscriminator,
    StyleGANDiscriminator,
    # tokenizer adversarial loss, perceptual loss
    VQLPIPSWithDiscriminator,
)

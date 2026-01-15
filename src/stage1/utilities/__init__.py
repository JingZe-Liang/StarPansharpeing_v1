from .losses.gan_loss.logit_laplace_loss import LogitLaplaceLoss
from .losses.gan_loss.loss import VQLPIPSWithDiscriminator
from .losses.latent_reg import (
    ChannelDropConfig,
    NestChannelDrop,
    lcr_loss,
    LatentMaskConfig,
    lmr_apply,
    LatentSparsityLoss,
    ls_loss,
)

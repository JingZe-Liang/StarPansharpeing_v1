from .dinov2_discrim import DinoDiscV2
from .patchgan_disc_maskbit import NLayerDiscriminatorv2
from .patchgan_discriminator import NLayerDiscriminator
from .stylegan import StyleGANDiscriminator
from .stylegan3d import StyleGAN3DDiscriminator
from .stylegan_utils.ops.conv2d_gradfix import no_weight_gradients

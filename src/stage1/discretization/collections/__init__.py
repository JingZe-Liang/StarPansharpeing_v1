from .bsq import BinarySphericalQuantizer, LogitLaplaceLoss
from .finite_scalar_quantization import FSQ
from .kl_continuous import DiagonalGaussianDistribution
from .latent_quantization import LatentQuantize
from .lookup_free_quantization import LFQ
from .random_projection_quantizer import RandomProjectionQuantizer
from .residual_fsq import GroupedResidualFSQ, ResidualFSQ
from .residual_lfq import GroupedResidualLFQ, ResidualLFQ
from .residual_sim_vq import ResidualSimVQ
from .residual_vq import GroupedResidualVQ, ResidualVQ
from .sim_vq import SimVQ
from .utils import Sequential
from .vector_quantize_pytorch import VectorQuantize

# ruff:noqa

# Components
from .norm_act import Qwen3NextRMSNorm, RMSNorm
from .layerscale import LayerScale
from .patcher import create_patcher, create_unpatcher
from .rescale import RescaleOutput
from .rope import (
    AxialPositionalEmbedding,
    AxialPositionalEmbedding2D,
    ContinuousAxialPositionalEmbedding,
    RopePosEmbed,
    RotaryPositionEmbeddingPytorchV2,
    get_2d_sincos_pos_embed,
)

# Blocks
from .conv import (
    ConvNeXtBlock,
    MbConvLNBlock,
    MBStem,
    ConvLayer,
    MBConv,
    FusedMBConv,
    GLUMBConv,
    GLUResBlock,
    ChannelAttentionResBlock,
    ResBlock,
    DSConv,
)
from .attention import (
    Attention,
    LiteLA,
    LiteMLA,
    NatAttention1d,
    NatAttention2d,
    Qwen3SdpaAttention,
    ReLULinearAttention,
)
from .mlp import SwiGLU, SwiGLUAct
from .naf import NAFBlock, NAFBlockConditional, NAFCrossAttentionConditional
from .blocks import (
    AttentionBlock,
    LiteLA_GLUMB_Block,
    CrossAttentionBlock,
    MbConvStages,
    Spatial2DNATBlock,
    Spatial2DNATBlockConditional,
    build_block,
    ConditionalBlock,
)
from .resample import create_downsample_layer, create_upsample_layer

# Adapters
from .dinov3_adapter import DINOv3_Adapter, DINOv3_Adapter_MS_Down

# Stages
from .stages import (
    MbConvSequentialCond,
    MbConvStagesCond,
    Spatial2DNatStage,
    ResBlockStage,
)

from .utils import pack_one, unpack_one

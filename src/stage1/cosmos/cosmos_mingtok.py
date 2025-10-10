from typing import List, Optional

import torch
import torch.nn as nn
from timm.layers import (
    create_norm_act_layer,
    create_norm_layer,
    get_act_layer,
    get_norm_act_layer,
    get_norm_layer,
)

from .modules import Attention, AttentionBlock, TransformerTokenizer

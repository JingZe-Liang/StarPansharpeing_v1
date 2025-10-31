import sys

sys.path.append("src/stage1/utilities/losses/dinov3")  # load dinov3 self-holded adapter

from .dinov3_adapted import DinoUNet
from .tokenizer_backbone_adapted import TokenizerHybridUNet

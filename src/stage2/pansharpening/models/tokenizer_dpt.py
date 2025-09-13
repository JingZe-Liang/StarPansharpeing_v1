from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn

from src.stage1.cosmos.cosmos_tokenizer import (
    ContinuousImageTokenizer,
    ContinuousTokenizerConfig,
)
from src.stage2.layers.dpt import DPTHead, EncoderDecoder, make_head


@dataclass
class DPTHeadConfig:
    in_channels = (1024, 1024, 1024, 1024)
    channels = 256
    post_process_channels = [128, 256, 512, 1024]
    readout_type = "project"
    expand_channels = False
    n_output_channels = 256
    use_batchnorm = False
    inputs_has_cls_token = True


@dataclass
class CosmosTokenizerDPTConfig:
    tokenizer_cfg: ContinuousTokenizerConfig = field(
        default_factory=lambda: ContinuousTokenizerConfig()
    )
    dpt_head_cfg: DPTHeadConfig = field(default_factory=lambda: DPTHeadConfig())
    encoder_dtype: str = "bfloat16"
    decoder_dtype: str = "bfloat16"


class CosmosTokenizerDPT(nn.Module):
    def __init__(self, cfg: CosmosTokenizerDPTConfig):
        super().__init__()
        tokenizer = ContinuousImageTokenizer(cfg.tokenizer_cfg)
        decoder = DPTHead(**asdict(cfg.dpt_head_cfg))
        self.model = EncoderDecoder(
            tokenizer,
            decoder,
            encoder_dtype=getattr(torch, cfg.encoder_dtype),
            decoder_dtype=getattr(torch, cfg.decoder_dtype),
        )

    def forward(self, x):
        output = self.model(x)
        return output

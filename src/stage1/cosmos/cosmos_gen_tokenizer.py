from dataclasses import asdict, dataclass, field

from src.stage1.cosmos.cosmos_tokenizer import (
    ContinuousImageTokenizer,
    ContinuousTokenizerConfig,
    EncoderDecoderConfig,
    test_tokenizer_forward_backward,
)
from src.stage1.cosmos.modules.layers2d import Encoder, GenerativeDecoder
from src.utilities.config_utils import dataclass_from_dict


@dataclass
class EncoderDecoderGenerativeConfig(EncoderDecoderConfig):
    per_layer_noise: bool = False


@dataclass
class CosmosGenerativeConfig(ContinuousTokenizerConfig):
    model: EncoderDecoderGenerativeConfig = field(default_factory=lambda: EncoderDecoderGenerativeConfig())


class CosmosGenerativeTokenizer(ContinuousImageTokenizer):
    def _build_encoder_decoder(  # type: ignore
        self,
        model_cfg: EncoderDecoderGenerativeConfig,
    ):
        encoder = Encoder(**asdict(model_cfg))
        decoder = GenerativeDecoder(**asdict(model_cfg))
        return encoder, decoder

    @classmethod
    def create_model(cls, **kwargs):
        cfg = dataclass_from_dict(CosmosGenerativeConfig, kwargs, strict=False)
        return cls(cfg)


# * --- Test --- #
if __name__ == "__main__":
    test_tokenizer_forward_backward(
        model_cls=CosmosGenerativeTokenizer,
        base_model_ckpt="",
        other_model_kwargs={"model": {"per_layer_noise": True}},
        use_optim=True,
        show_mem_usage=True,
        real_data=None,
        fake_img_shape=(5, 12, 512, 512),
    )

from typing import Any, Union

import attrs
from omegaconf import OmegaConf


@attrs.define
class Encoder:
    block_chans: int = 128


@attrs.define(slots=False)
class ConfigExample:
    learning_rate: float = 0.001
    batch_size: int = 32
    model_name: str = "resnet50"
    use_augmentation: bool = True
    optimizer: str = attrs.field(default="adam", validator=attrs.validators.in_(("adam", "sgd")))
    levels: list[int] = attrs.field(default=[1, 2, 3])
    encoder_cfg: Encoder = attrs.field(default=Encoder())


# cfg = ConfigExample(learning_rate=0.01, batch_size=64)
# cfg.use_augmentation = False
# print(cfg)


base = OmegaConf.structured(ConfigExample)
OmegaConf.set_struct(base, False)
print(OmegaConf.to_yaml(base))

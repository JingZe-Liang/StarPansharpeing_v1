import pydoc
import sys
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.utils.flop_counter
from jaxtyping import Float
from torch import Tensor
from torch.nn.modules.conv import _ConvNd

sys.path.append("src/stage1/utilities/losses/dinov3")  # load dinov3 self-holded adapter

from src.stage1.utilities.losses.repa import (
    load_repa_dino_v3_model as load_dino_v3_model,
)
from src.stage1.utilities.losses.repa.feature_pca import (
    feature_pca_torch as hyper_to_rgb_pca,
)
from src.stage2.classification.models.adapter import (
    DINOv3_Adapter,
    DINOv3EncoderAdapter,
    UNetDecoder,
)
from src.utilities.config_utils import (
    dataclass_from_dict,
    function_config_to_basic_types,
)
from src.utilities.logging import log

# DINOv3_MODEL_FACTORIES = {
#     "dinounet_s": dinov3_vits16,
#     "dinounet_b": dinov3_vitb16,
#     "dinounet_l": dinov3_vitl16,
#     "dinounet_7b": dinov3_vit7b16,
# }

DINOv3_INTERACTION_INDEXES = {
    "dinov3_vits16": [2, 5, 8, 11],
    "dinov3_vitb16": [2, 5, 8, 11],
    "dinov3_vitl16": [4, 11, 17, 23],
    "dinov3_vit7b16": [9, 19, 29, 39],
}


# DINOv3_MODEL_INFO = {
#     "dinounet_s": {"embed_dim": 384, "depth": 12, "num_heads": 6, "params": "~22M"},
#     "dinounet_b": {"embed_dim": 768, "depth": 12, "num_heads": 12, "params": "~86M"},
#     "dinounet_l": {"embed_dim": 1024, "depth": 24, "num_heads": 16, "params": "~300M"},
#     "dinounet_7b": {"embed_dim": 4096, "depth": 40, "num_heads": 32, "params": "~7B"},
# }
# * --- Configurations --- #


@dataclass
class DinoConfig:
    features_per_stage: Any = (256, 256, 256, 256)
    pretrained_path: Any = field(default=None)  # path to pretrained weights
    model_name: str = "dinov3_vits16"
    pretrained_on: str = "web"


@dataclass
class AdapterConfig:
    adapter_type: str = "default"
    latent_width: Any = None
    n_conv_per_stage: int = 1
    depth_per_stage: int = 1
    norm: str = "layernorm2d"
    act: str = "gelu"
    drop: float = 0.0
    act_first: bool = False
    conv_bias: bool = False


@dataclass
class DinoUnetConfig:
    dino: DinoConfig = field(default_factory=lambda: DinoConfig())
    adapter: AdapterConfig = field(default_factory=lambda: AdapterConfig())
    input_channels: int = 3
    num_classes: int = 7
    deep_supervision: bool = False
    n_stages: int = 4
    use_latent: bool = False
    ensure_rgb_type: str = "pca"
    _debug: bool = False


class DinoUNet(nn.Module):
    """
    U-Net with DINOv3_Adapter as encoder, compatible with PlainConvUNet interface
    """

    def __init__(self, cfg: DinoUnetConfig):
        super().__init__()
        self.cfg = cfg
        self.dino_cfg = cfg.dino
        self.adapter_cfg = cfg.adapter

        self.use_latent = cfg.use_latent
        self._debug = cfg._debug

        # Validate parameters
        n_conv_per_stage = self.adapter_cfg.n_conv_per_stage
        n_stages = cfg.n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        # if isinstance(n_conv_per_stage_decoder, int):
        #     n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        # Ensure we have 4 stages to match DINOv3_Adapter output
        if cfg.n_stages != 4:
            print(
                f"Warning: DINOv3_Adapter outputs 4 scales, but n_stages={n_stages}. Adjusting to 4."
            )
            n_stages = 4
            if isinstance(self.dino_cfg.features_per_stage, int):
                self.cfg.dino.features_per_stage = [
                    self.dino_cfg.features_per_stage * (2**i) for i in range(4)
                ]
            elif len(self.dino_cfg.features_per_stage) != 4:
                # Adjust features_per_stage to 4 stages
                base_features = (
                    self.dino_cfg.features_per_stage[0]
                    if self.dino_cfg.features_per_stage
                    else 32
                )
                self.cfg.dino.features_per_stage = [
                    base_features * (2**i) for i in range(4)
                ]

        # Create DINOv3 encoder
        self.encoder = self._create_dinov3_encoder(self.cfg)

        # Create decoder
        self.decoder = UNetDecoder(
            self.encoder,
            cfg.num_classes,
            cfg.adapter.latent_width,
            cfg.adapter.n_conv_per_stage,
            cfg.adapter.depth_per_stage,
            nonlin_first=cfg.adapter.act_first,
            deep_supervision=cfg.deep_supervision,
        )

        # Init weights
        self.apply(self.initialize)

    def _create_dinov3_encoder(self, cfg: DinoUnetConfig):
        """Create DINOv3 encoder"""

        # Get model information
        model_name = cfg.dino.model_name
        # if model_name not in DINOv3_MODEL_INFO:
        #     raise ValueError(f"Unknown model: {model_name}")

        # model_info = DINOv3_MODEL_INFO[model_name]
        interaction_indexes = DINOv3_INTERACTION_INDEXES[model_name]

        log(f"Creating DINOv3 encoder: {model_name}")
        # log(f"   Embedding dimension: {model_info['embed_dim']}")
        # log(f"   Model depth: {model_info['depth']}")
        # log(f"   Number of attention heads: {model_info['num_heads']}")
        # log(f"   Parameter count: {model_info['params']}")
        # log(f"   Interaction layer indices: {interaction_indexes}")

        # Load DINOv3 backbone
        dinov3_backbone = load_dino_v3_model(
            cfg.dino.pretrained_path, model_name, pretrained_on=cfg.dino.pretrained_on
        )

        # Create DINOv3_Adapter using correct interaction layer indices
        dinov3_adapter = DINOv3_Adapter(
            backbone=dinov3_backbone,
            interaction_indexes=interaction_indexes,
            pretrain_size=512,
            conv_inplane=64,
            n_points=4,
            deform_num_heads=16,
            drop_path_rate=0.3,
            init_values=0.0,
            with_cffn=True,
            cffn_ratio=0.25,
            deform_ratio=0.5,
            add_vit_feature=True,
            use_extra_extractor=True,
            with_cp=True,
        )

        encoder_adapter = DINOv3EncoderAdapter(
            dinov3_adapter=dinov3_adapter,
            target_channels=cfg.dino.features_per_stage,
            conv_op=nn.Conv2d,
            norm_op=cfg.adapter.norm,
            dropout_op=cfg.adapter.drop,
            nonlin=cfg.adapter.act,
            conv_bias=cfg.adapter.conv_bias,
        )

        return encoder_adapter

    def _ensure_rgb_input(
        self, x: Float[Tensor, "b c h w"], larger_then_3_op: str | list[int] = "pca"
    ):
        C = x.size(1)
        if C == 1:
            x = x.repeat(1, 3, 1, 1)
        elif C != 3:
            if C < 3:
                x = x.repeat(1, 3 // C + (1 if 3 % C != 0 else 0), 1, 1)[:, :3, :, :]
            elif larger_then_3_op == "first_3":
                x = x[:, :3, :, :]
            elif larger_then_3_op == "mean":
                channels_per_group = C // 3
                remainder = C % 3
                groups = []
                start = 0
                for i in range(3):
                    group_size = channels_per_group + (1 if i < remainder else 0)
                    end = start + group_size
                    groups.append(x[:, start:end].mean(dim=1, keepdim=True))
                    start = end
                x = torch.cat(groups, dim=1)
            elif larger_then_3_op == "pca":
                x = hyper_to_rgb_pca(x, pca_k=3)
            elif isinstance(larger_then_3_op, (list, tuple)):
                x = x[:, larger_then_3_op]
            else:
                raise ValueError(
                    f"Unknown operation for C > 3 ({C=}): {larger_then_3_op}"
                )
        elif C == 3:
            pass
        else:
            raise ValueError(f"Unexpected number of channels: {C}")

        return x

    def forward(self, x: Float[Tensor, "b c h w"], cond=None):
        x = self._ensure_rgb_input(x)
        skips = self.encoder(x)
        output = self.decoder(skips, cond)
        return output

    def initialize(self, module) -> None:
        if isinstance(module, _ConvNd):
            if module.weight.requires_grad:
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @classmethod
    @function_config_to_basic_types
    def from_config(cls, overrides: dict | None = None):
        """
        Create DinoUNet instance from network configuration dictionary
        """
        cfg = dataclass_from_dict(
            DinoUnetConfig, {} if overrides is None else overrides
        )

        return cls(cfg)


# * --- Test --- #


def test_cfg():
    from src.utilities.config_utils.to_dataclass import (
        dataclass_from_dict,
        dataclass_to_dict,
    )

    d = DinoUnetConfig()
    dd = dataclass_to_dict(d)
    print(dd)

    # to dataclass
    d2 = dataclass_from_dict(DinoUnetConfig, dd)
    print(d2)


def test_model():
    from fvcore.nn import parameter_count_table

    model = DinoUNet.from_config({"adapter": {"latent_width": 16}})
    # print(model)
    print(parameter_count_table(model))

    x = torch.randn(1, 3, 256, 256)
    cond = torch.randn(1, 16, 32, 32)
    output = model(x, cond)
    print(output.shape)

    from torch.utils.flop_counter import FlopCounterMode

    flop_counter = FlopCounterMode(display=True)
    with flop_counter:
        output = model(x, cond)
    print(flop_counter.get_total_flops() / 1024 / 1024 / 1024, "GFlops")


if __name__ == "__main__":
    test_model()

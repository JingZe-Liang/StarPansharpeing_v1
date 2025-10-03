from dataclasses import asdict, dataclass, field
from typing import Any, override

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor, nn

from src.utilities.config_utils import dataclass_from_dict, dataclass_to_dict
from src.utilities.logging import log

from ...utilities.amotized import AmotizedModelMixin
from .to_endmember import (
    EndMemberBase,
    ToEndMemberConfig,
    ToEndMemberConv,
    ToEndMemberParameter,
)
from .transformer import Transformer, TransformerConfig
from .vitamin_conv import ConvCfg, VitaminCfg, VitaminModel


@dataclass
class UnmixingConfig:
    transformer: TransformerConfig
    vitamin: VitaminCfg
    to_endmember: ToEndMemberConfig

    amotize_type: str = "latent_to_pixel_fusion"
    learn_decoder: bool = False
    backward_decoder: bool = False

    set_grad_checkpoint: bool = False


class LatentUnmixingModel(AmotizedModelMixin):
    def __init__(
        self,
        pixel_model: nn.Module,
        amotized_model: nn.Module,
        end_member_model: EndMemberBase,
        decoder_fn,
        amotize_type: str,
        backward_decoder: bool = False,
        learn_decoder: bool = False,
        set_grad_checkpoint: bool = False,
        endmember_init_value: np.ndarray | Tensor | None = None,
    ):
        assert decoder_fn is not None
        assert hasattr(end_member_model, "get_endmember")

        super().__init__(
            pixel_model,
            amotized_model,
            decoder_fn,
            amotize_type,
            backward_decoder,
            learn_decoder,
        )
        self.end_member_model = end_member_model
        self._has_init_endmembers = False

        if set_grad_checkpoint:
            self.set_grad_checkpoint(mode=True)

        if endmember_init_value is not None:
            endmember_init_value = torch.as_tensor(endmember_init_value)
            self.init_endmembers(endmember_init_value)

    def init_endmembers(self, init_value: Float[torch.Tensor, "n_endmember channels"]):
        if self._has_init_endmembers:
            log(f"Endmembers already initialized.", level="warning")
            return

        self.end_member_model.init_endmembers(init_value)
        self._has_init_endmembers = True
        log(f"Endmembers initialized with shape {init_value.shape}", once=True)

    @override
    def latent_to_pixel_fusion_forward(self, pixel_in: tuple, latent_in: tuple):
        transfomer_out, _, abunds = (
            super().latent_to_pixel_fusion_forward(pixel_in, latent_in).values()
        )
        recon = self.end_member_model(abunds).clip(0, 1)
        endmembers = self.end_member_model.get_endmember()  # type: ignore

        return {
            "amotized_model_out": transfomer_out,
            "abunds": abunds,
            "recon": recon,
            "endmembers": endmembers,
        }

    @classmethod
    def create_model(cls, detokenizer=None, **overrides: dict):
        config: UnmixingConfig = dataclass_from_dict(UnmixingConfig, overrides)
        amotized_model = Transformer(config.transformer)
        pixel_model = VitaminModel(config.vitamin)
        end_member_model = (
            ToEndMemberConv(**asdict(config.to_endmember))
            if config.to_endmember.module_type == "conv"
            else ToEndMemberParameter(**asdict(config.to_endmember))
        )

        assert config.amotize_type == "latent_to_pixel_fusion", (
            "other types of amotize fn does not implemented yet"
        )

        return cls(
            pixel_model=pixel_model,
            amotized_model=amotized_model,
            end_member_model=end_member_model,
            decoder_fn=detokenizer if detokenizer is not None else lambda x: x,
            amotize_type=config.amotize_type,
            set_grad_checkpoint=config.set_grad_checkpoint,
        )

    def set_grad_checkpoint(self, mode=True):
        for name, module in self.named_modules():
            if hasattr(module, "grad_checkpointing"):
                module.grad_checkpointing = mode  # type: ignore[attr-defined]
                log(f"Set grad_checkpointing={mode} for {name} ({module})")


def test_model():
    import torch

    # Example usage of UnmixingConfig dataclass
    unmixing_cfg = UnmixingConfig(
        transformer=TransformerConfig(
            in_dim=16,
            out_channels=256,
            num_heads=8,
            mlp_ratio=2.0,
        ),
        vitamin=VitaminCfg(
            stem_width=64,
            embed_dim=[192, 192, 192],
            depths=[2, 2, 2],
            input_channel=170,  # for Houston dataset
            condition_channel=256,
            use_residual=False,
            conv_cfg=ConvCfg(
                expand_ratio=2.0,
                kernel_size=3,
                act_layer="gelu",
                norm_layer="layernorm2d",
            ),
            output_channel=8,
        ),
        to_endmember=ToEndMemberConfig(
            apply_relu=True,
            channels=170,
            init_value=None,
            num_endmember=8,
        ),
        amotize_type="latent_to_pixel_fusion",
        learn_decoder=False,
        backward_decoder=False,
        set_grad_checkpoint=False,
    )
    print("UnmixingConfig example:", unmixing_cfg)

    # Create sample inputs
    batch_size, height, width = 2, 256, 256
    img = torch.randn(batch_size, unmixing_cfg.vitamin.input_channel, height, width)
    latent = torch.randn(batch_size, 16, height // 8, width // 8)  # latent code

    # Create model
    model = LatentUnmixingModel.create_model(**asdict(unmixing_cfg))

    # Forward pass
    with torch.no_grad():
        output = model(img, latent)
        print("Output keys:", output.keys())

    for k, v in output.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {v.shape}")


if __name__ == "__main__":
    """
        python -m src.stage2.unmixing.models.model
    """

    test_model()

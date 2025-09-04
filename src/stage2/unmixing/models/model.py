from dataclasses import asdict, dataclass, field
from typing import Any, override

import torch
from jaxtyping import Float
from torch import nn

from src.utilities.config_utils import dataclass_from_dict, dataclass_to_dict
from src.utilities.logging import log
from src.utilities.network_utils import register_network_init

from ...utilities.amotized import AmotizedModelMixin
from .to_endmember import ToEndMemberConfig, ToEndMemberConv, ToEndMemberParameter
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
        end_member_model: nn.Module,
        decoder_fn,
        amotize_type: str,
        backward_decoder: bool = False,
        learn_decoder: bool = False,
        set_grad_checkpoint: bool = False,
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

        if set_grad_checkpoint:
            self.set_grad_checkpoint(mode=True)

    @override
    def latent_to_pixel_fusion_forward(self, pixel_in: tuple, latent_in: tuple):
        transfomer_out, _, abunds = (
            super().latent_to_pixel_fusion_forward(pixel_in, latent_in).values()
        )
        recon = self.end_member_model(abunds)
        end_member = self.end_member_model.get_endmember()  # type: ignore

        return {
            "amotized_model_out": transfomer_out,
            "abunds": abunds,
            "recon": recon,
            "end_member": end_member,
        }

    @classmethod
    def from_config(cls, overrides: dict, detokenizer=None):
        config: UnmixingConfig = dataclass_from_dict(UnmixingConfig, overrides)
        amotized_model = Transformer(**asdict(config.transformer))
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
                module.grad_checkpointing = mode
                log(f"Set grad_checkpointing={mode} for {name} ({module})")

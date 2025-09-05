from abc import ABC, abstractmethod
from dataclasses import MISSING, dataclass, field
from typing import Any

import torch
import torch.nn as nn
from jaxtyping import Float


@dataclass
class ToEndMemberConfig:
    num_endmember: int
    channels: int
    init_value: Any
    kernel: int = 1
    module_type: str = "conv"
    apply_relu: bool = True


class EndMemberBase(nn.Module, ABC):
    @abstractmethod
    def init_endmembers(self, init_value: torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    def get_endmember(self):
        raise NotImplementedError


class ToEndMemberConv(EndMemberBase):
    def __init__(
        self,
        num_endmember: int,
        channels: int,
        kernel: int,
        init_value=None,
        apply_relu=True,
        **kwargs,
    ):
        super(ToEndMemberConv, self).__init__()
        padding = kernel // 2
        self.decoder = nn.Conv2d(
            in_channels=channels,
            out_channels=num_endmember,
            kernel_size=kernel,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.apply_relu = apply_relu
        if init_value is not None:
            assert kernel == 1
            init_value.squeeze_(0)
            assert init_value.ndim == 2, "init_value must be 2D"
            self.decoder.weight.data = init_value[..., None, None]

    def forward(self, code):
        code = self.decoder(code)  # [bs, c_in, h, w] -> [bs, c_out, h, w]
        if self.apply_relu:
            code = torch.nn.functional.relu(code)
        return code

    def init_endmembers(
        self, init_value: Float[torch.Tensor, "num_endmember channels"]
    ):
        """Initialize endmembers with given values.

        Args:
            init_value: Tensor containing endmember initialization values.
                       Should have shape (num_endmember, channels).
        """
        init_value.squeeze_(0)

        assert self.decoder.kernel_size[0] == 1, (
            "Kernel size must be 1 for endmember initialization"
        )
        assert init_value.ndim == 2, "init_value must be 2D"

        self.decoder.weight.data.copy_(init_value[..., None, None])

    def get_endmember(self):
        # (num_endmember, channels)
        endmember = self.decoder.weight.data.clamp_(min=0.0)
        return endmember


class ToEndMemberParameter(EndMemberBase):
    def __init__(
        self,
        num_endmember: int,
        channels: int,
        init_value=None,
        apply_relu=True,
        **kwargs,
    ):
        super(ToEndMemberParameter, self).__init__()
        self.endmember = nn.Parameter(torch.randn(num_endmember, channels))
        self.apply_relu = apply_relu

        # init
        if init_value is not None:
            init_value.squeeze_(0)
            assert init_value.ndim == 2, "init_value must be 2D"
            assert self.endmember.data.shape == init_value.shape, (
                f"init_value shape must be {self.endmember.data.shape}"
            )
            self.endmember.data = init_value
        else:
            nn.init.trunc_normal_(self.endmember.data, mean=0.0, std=1.0)
            self.endmember.data.clamp_(min=0.0)

    def forward(self, code):
        code = torch.einsum("bchw,dc->bdhw", code, self.endmember)
        if self.apply_relu:
            # if self.apply_relu == False,
            # the negative values will cause the abunds_loss to nan
            code = torch.relu(code)

        return code

    def init_endmembers(
        self, init_value: Float[torch.Tensor, "num_endmember channels"]
    ):
        """Initialize endmembers with given values.

        Args:
            init_value: Tensor containing endmember initialization values.
                       Should have shape (channels, num_endmember).
        """
        init_value.squeeze_(0)
        assert init_value.ndim == 2, "init_value must be 2D"
        assert self.endmember.data.shape == init_value.shape, (
            f"init_value shape must be {self.endmember.data.shape}"
        )

        self.endmember.data.copy_(init_value)

    def get_endmember(self):
        return self.endmember.data.clamp_(min=0.0)

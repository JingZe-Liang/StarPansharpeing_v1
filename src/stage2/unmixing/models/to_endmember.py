from abc import ABC, abstractmethod
from dataclasses import MISSING, dataclass, field
from typing import Any

import torch
import torch.nn as nn
from jaxtyping import Float
from loguru import logger


@dataclass
class ToEndMemberConfig:
    num_endmember: int
    channels: int
    init_value: Any
    kernel: int = 1
    module_type: str = "conv"
    apply_relu: bool = True


class EndMemberBase(nn.Module, ABC):
    _is_inited: bool = False

    def init_endmembers(
        self, init_value: Float[torch.Tensor, "channels num_endmember"]
    ):
        if self._is_inited:
            logger.error(
                "[Endmember Base]: Endmembers have already been initialized, skipping."
            )
            return
        self._is_inited = True
        self.init_endmembers_fn(init_value)
        logger.info("[Endmember Base]: Endmembers initialized.")

    @abstractmethod
    def init_endmembers_fn(self, init_value: torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    def get_endmember(self):
        raise NotImplementedError


class ToEndMemberConv(EndMemberBase):
    def __init__(
        self,
        num_endmember: int,
        channels: int,
        kernel: int = 1,
        init_value=None,
        apply_relu=True,
        **kwargs,
    ):
        super(ToEndMemberConv, self).__init__()
        padding = kernel // 2
        # Remove redundant is_inited flag as it's already handled by parent class
        self.decoder = nn.Conv2d(
            in_channels=num_endmember,
            out_channels=channels,
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

    def init_endmembers_fn(
        self, init_value: Float[torch.Tensor, "channels num_endmember"]
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

        # channels, num_endmember, 1, 1
        self.decoder.weight.data.copy_(init_value[..., None, None])

    def get_endmember(self):
        # (num_endmember, channels)
        # Use non-in-place clamp to avoid modifying the original weights
        endmember = torch.clamp(self.decoder.weight.data, min=0.0)  # [c, em]
        endmember = endmember.squeeze(-2, -1).T  # [em, c]
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
        self.endmember = nn.Parameter(
            torch.empty(channels, num_endmember), requires_grad=True
        )

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
        code = torch.einsum("bdhw,cd->bchw", code, self.endmember)

        # c_p = code.permute(0, 2, 3, 1)
        # code = self.endmember(c_p)  # [bs, h, w, em]
        # code = code.permute(0, 3, 1, 2)  # [bs, em, h, w]

        if self.apply_relu:
            # if self.apply_relu == False,
            # the negative values will cause the abunds_loss to nan
            code = torch.relu(code)

        return code

    def init_endmembers_fn(
        self, init_value: Float[torch.Tensor, "channels num_endmember"]
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
        # Use non-in-place clamp to avoid modifying the original parameters
        return torch.clamp(self.endmember.data, min=0.0).T

        # w1 = self.endmember[0].weight  # [c1, em]
        # w2 = self.endmember[1].weight  # [c, c1]
        # em = w2 @ w1  # [c, em]
        # return em.clamp(min=0.0).T  # [em, c]


class ToEndMemberMutliLinear(EndMemberBase):
    def __init__(
        self,
        num_endmember: int,
        channels: int,
        init_value=None,
        apply_relu=True,
        n_layers=3,
        hidden_channel: int = 256,
        **kwargs,
    ):
        super(ToEndMemberMutliLinear, self).__init__()
        assert n_layers >= 2
        self.n_layers = n_layers
        self.hidden_channel = hidden_channel
        self.num_endmember = num_endmember
        self.channels = channels

        st_in = [nn.Linear(num_endmember, hidden_channel, bias=False)]
        for _ in range(n_layers - 2):
            st_in.append(nn.Linear(hidden_channel, hidden_channel, bias=False))
        st_in.append(nn.Linear(hidden_channel, channels, bias=False))
        self.decoder = nn.Sequential(*st_in)

        self.apply_relu = apply_relu

        self.init_weights()

    def init_weights(self):
        for i in range(self.n_layers):
            nn.init.zeros_(self.decoder[i].weight)
            self.decoder[i].weight.data.fill_diagonal_(1.0)

    def forward(self, em):
        h, w = em.shape[-2:]
        em = em.flatten(2).permute(0, 2, 1)  # [bs, h*w, em]
        recon = self.decoder(em)
        recon = recon.permute(0, 2, 1).reshape(-1, self.channels, h, w)  # [bs, c, h, w]
        if self.apply_relu:
            recon = torch.relu(recon)
        return recon

    def init_endmembers_fn(
        self, init_value: Float[torch.Tensor, "channels num_endmember"]
    ):
        # [e, c1] @ [c1, c1] @ [c1, c] -> [e, c1] @ [c1, c] -> [e, c]
        # [e, c] (inited) @ [c, c1] (eyes) -> [e, c1] (inited, first layer) @ [c1, c] (eyes, seq layers) -> [e, c]
        device = self.decoder[0].weight.device

        eyes = (
            torch.zeros(self.hidden_channel, self.channels, device=device)
            .fill_diagonal_(1.0)
            .type_as(init_value)
        )  # [c1, c]
        first_inited_w = eyes @ init_value.to(device)  # [c1, c] @ [c, em] -> [c1, em]
        self.decoder[0].weight.data.copy_(first_inited_w)

    def get_endmember(self):
        with torch.no_grad():
            _w = self.decoder[0].weight
            eye = torch.eye(self.num_endmember, self.num_endmember).to(_w)  # [em, em]
            em = self.decoder(eye)  # [em, em] @ [em, c1] @ ... @ [c1, c] -> [em, c]
            return torch.clamp(em, min=0.0)  # [em, c]


if __name__ == "__main__":
    model = ToEndMemberMutliLinear(
        3, 5, n_layers=5, hidden_channel=256, apply_relu=False
    )

    # x = torch.randn(1, 3)
    # print(x)
    # y = model(x)
    # print(y)

    init_v = torch.rand(5, 3)
    model.init_endmembers(init_v)

    em = model.get_endmember()
    assert em.shape == (3, 5)

    print(model.get_endmember() - init_v.T)

    # em = torch.randn(3, 100)

    # eye100 = torch.eye(100)
    # eye256 = torch.eye(256)
    # zero_left = torch.zeros(100, 156)
    # ones1 = torch.zeros(100, 256).fill_diagonal_(1)
    # ones2 = eye256
    # ones3 = torch.zeros(256, 100).fill_diagonal_(1)

    # em2 = em @ ones1 @ ones2 @ ones3
    # print(em2.shape)

    # print(em2 - em)

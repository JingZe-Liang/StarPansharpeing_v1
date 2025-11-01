import functools
from collections.abc import Sequence
from functools import partial, wraps
from typing import Annotated, Literal, Union

import torch
import torch.nn as nn
from einops import rearrange
from loguru import logger
from timm.layers import create_conv2d
from timm.layers.create_norm import create_norm_layer
from timm.layers.create_norm_act import create_norm_act_layer
from torch import Tensor
import torch.nn.functional as F

from src.utilities.network_utils import null_decorator_no_any_kwgs

compile_forward_fn = False
if compile_forward_fn:
    _compile_decorator = torch.compile
else:
    _compile_decorator = null_decorator_no_any_kwgs


def _kernel_norm(
    w: Annotated[Tensor, "c_out c_in k k"],
    kernel_norm: str | None,
    dim: Literal["c_in", "c_out"] = "c_in",
) -> Annotated[Tensor, "c_out c_in k k"]:
    if kernel_norm is None:
        return w

    if kernel_norm == "softmax":
        # Apply softmax on input channel dimension
        dim_ = 1 if dim == "c_in" else 0
        w = F.softmax(w, dim=dim_)
        return w

    elif kernel_norm == "layernorm":
        w_shape = w.shape
        if dim == "c_in":
            w = w.reshape(*w.shape[:2], -1).permute(0, 2, 1)  # c_out, (k*k), c_in
            w = F.layer_norm(w, [w.shape[-1]])  # normalize on c_in dimension
            return w.permute(0, 2, 1).reshape(w_shape)  # c_out, c_in, k, k
        else:
            w = w.reshape(*w.shape[2:], -1).permute(1, 2, 0)  # c_in, (k*k) c_out
            w = F.layer_norm(w, [w.shape[-1]])  # normalize on c_out dimension
            return w.permute(2, 1, 0).reshape(w_shape)  # c_out, c_in, k, k

    elif kernel_norm == "weight_std":
        dim_ = 1 if dim == "c_in" else 0
        # Weight standardization: normalize to zero mean and unit std
        mean = w.mean(dim=dim_, keepdim=True)  # mean over c_in dimension
        std = w.std(dim=dim_, keepdim=True) + 1e-5  # std over c_in dimension
        return (w - mean) / std

    else:
        raise ValueError(f"Unknown kernel_norm type: {kernel_norm}")


class ActNorm(nn.Module):
    def __init__(
        self, num_features, logdet=False, affine=True, allow_reverse_init=False
    ):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data = -mean.to(input.device)
            self.scale.data = 1 / (std + 1e-6).to(input.device)

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized = torch.tensor(1, dtype=torch.uint8).to(input.device)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height * width * torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h


class DiscriminatorLayer(torch.nn.Sequential):
    def __init__(
        self,
        ndf,
        nf_mult_prev,
        nf_mult,
        kw,
        padw,
        use_bias,
        norm_layer,
    ):
        super().__init__()
        self.extend(
            [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                    padding_mode="reflect",
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2),
            ]
        )


class DiffBandsInputConvIn(nn.Module):
    def __init__(
        self,
        band_lst: list[int],
        hidden_dim: int,
        basic_module: str = "conv_norm_act",
        use_timm=False,
    ):
        super().__init__()

        self.band_lst = band_lst
        self.hidden_dim = hidden_dim
        if basic_module == "conv":
            basic_module_fn = nn.Conv2d
        elif basic_module == "conv_norm_act":

            def basic_module_fn(
                in_channels, out_channels, kernel_size, stride, padding
            ):
                if use_timm:
                    norm_act = create_norm_act_layer(
                        "layernorm2d",
                        out_channels,
                        act_layer=nn.LeakyReLU(negative_slope=0.2),
                    )
                else:
                    norm_act = (
                        nn.GroupNorm(32, out_channels, eps=1e-6),
                        nn.LeakyReLU(negative_slope=0.2),
                    )

                layer = [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    )
                ]
                layer += norm_act if isinstance(norm_act, Sequence) else [norm_act]
                return nn.Sequential(*layer)
        else:
            raise ValueError(f"basic_module {basic_module} is not supported")

        kw = 4
        padw = 1

        self.in_modules = nn.ModuleDict()
        for c in band_lst:
            self.in_modules["conv_in_{}".format(c)] = basic_module_fn(  # type: ignore
                in_channels=c,
                out_channels=hidden_dim,
                kernel_size=kw,
                stride=2,
                padding=padw,
            )

            logger.info(
                f"[Disc] set conv to hidden module and buffer for channel {c}", "debug"
            )
        # logger.info(f"[Disc] diffbands input convs: {self.in_modules}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c_ = x.shape[1]
        module = getattr(self.in_modules, "conv_in_{}".format(c_))
        if module is None:
            raise ValueError(
                f"[Disc] no module for channel {c_}, please check the channel list"
            )
        h = module(x)

        if self.training:
            for c in self.band_lst:
                if c != c_:
                    m = self.in_modules["conv_in_{}".format(c)]
                    dummy_loss = sum(p.sum() * 0.0 for p in m.parameters())
                    h = h + dummy_loss

        return h


class AdaptiveInputConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        padding: int | None = None,
        use_bias: bool = False,
        mode: Literal["slice", "interp", "interp_proj"] = "slice",
        k_hidden: int | None = None,
        kernel_norm: str | None = None,
    ):
        super().__init__()
        conv_kwargs = dict(
            stride=stride,
            groups=groups,
            dilation=dilation,
            bias=use_bias,
            padding_mode="reflect",
        )
        if padding is not None:
            # if padding not set, the create_conv2d will use same padding
            conv_kwargs["padding"] = padding

        self.conv = create_conv2d(in_channels, out_channels, kernel_size, **conv_kwargs)
        self.mode = mode
        self.kernel_norm = kernel_norm

        if mode == "interp_proj":
            # (bs, c, k1, k2) img -> (c_out, c_in, k1, k2) kernel
            # kernel -> (c_out, k1*k2, c_in) -> Linear(in_channels, k_hidden) -> k_hidden -> c_in -> (c_out, k1*k2, c_in)
            k_hidden = k_hidden or in_channels
            self.kernel_proj = nn.ModuleList(
                [
                    nn.Linear(in_channels, k_hidden, bias=use_bias),
                    nn.Linear(k_hidden, in_channels),
                ]
            )

        self.forward_mappings = dict(
            slice=self._slice_forward,
            interp=self._interp_forward,
            interp_proj=self._interp_proj_forward,
        )

    def _forward_conv_with_wb(self, x, w, b):
        x = nn.functional.conv2d(  # type: ignore
            x,
            w,
            b,
            self.conv.stride,
            self.conv.padding,
            self.conv.dilation,
            self.conv.groups,
        )
        return x

    def _slice_forward(self, x):
        w = self.conv.weight[:, : x.shape[1]]
        w = _kernel_norm(w, self.kernel_norm, "c_in")
        x = self._forward_conv_with_wb(x, w, self.conv.bias)
        return x

    def _interp_forward(self, x, w: Tensor | None = None):
        in_channels = x.shape[1]
        if w is None:
            w: Tensor = self.conv.weight

        c_out, c_in, k1, k2 = w.shape
        # c_in -> in_channels
        w = rearrange(w, "c_out c_in k1 k2 -> k1 k2 c_out c_in")
        w = torch.nn.functional.interpolate(
            w, size=(c_out, in_channels), mode="bicubic", align_corners=False
        )
        w = rearrange(w, "k1 k2 c_out c_in -> c_out c_in k1 k2")
        # Conv
        w = _kernel_norm(w, self.kernel_norm, "c_in")
        x = self._forward_conv_with_wb(x, w, self.conv.bias)
        return x

    def _interp_proj_forward(self, x):
        in_channels = x.shape[1]
        lin1, lin2 = self.kernel_proj

        # weights
        w: Tensor = self.conv.weight
        c_out, c_in, k1, k2 = w.shape

        # weight projection
        w = rearrange(w, "c_out c_in k1 k2 -> c_out (k1 k2) c_in")
        w = lin1(w)  # c_out, (k1*k2), k_hidden
        w = lin2(w)  # c_out, (k1*k2), c_in
        w = rearrange(w, "c_out (k1 k2) c_in -> c_out c_in k1 k2", k1=k1, k2=k2)

        # Interpolation
        return self._interp_forward(x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_channels = x.shape[1]

        # Native case
        if in_channels == self.conv.weight.shape[1]:
            w = _kernel_norm(self.conv.weight, self.kernel_norm, "c_in")
            return self._forward_conv_with_wb(x, w, self.conv.bias)

        # Adaptive cases
        return self.forward_mappings[self.mode](x)

    @property
    def weight(self):
        return self.conv.weight


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
    --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    _no_split_modules = ["DiscriminatorLayer"]

    def __init__(
        self,
        input_nc: int | list[int] = 3,
        ndf=64,
        n_layers=3,
        use_actnorm=False,
        # use_bn: bool = True,
        norm_type="bn2d",
    ):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if not use_actnorm:
            # Zihan Note: GroupNorm underperforms than BatchNorm
            if norm_type != "gn":
                _mapping = {
                    "bn2d": "batchnorm2d",
                    "ln2d": "layernorm2d",
                    "rmsnorm2d": "rmsnorm2d",
                }
                norm_type = _mapping.get(norm_type, norm_type)
                norm_layer = lambda c: create_norm_layer(
                    layer_name=norm_type, num_features=c
                )
            else:
                norm_layer = lambda c: nn.GroupNorm(32, c)
        else:
            norm_layer = ActNorm

        if isinstance(
            norm_layer, functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        logger.info(
            "[NLayerDisc] config: "
            f"input_nc={input_nc}, "
            f"ndf={ndf}, "
            f"n_layers={n_layers}, "
            f"use_actnorm={use_actnorm}, "
            f"norm_type={norm_type}, "
            f"use_bias={use_bias}"
        )

        kw = 4
        padw = 1

        if isinstance(input_nc, int):
            # AdaptiveInputConvLayer(
            #     input_nc, ndf, kernel_size=kw, stride=2, padding=padw
            # )
            conv_in = AdaptiveInputConvLayer(
                input_nc,
                ndf,
                kernel_size=kw,
                stride=2,
                padding=padw,
                mode="interp",
            )
            logger.info("[NLayerDisc]: use interp conv kernel for input conv")
        else:
            conv_in = DiffBandsInputConvIn(
                band_lst=input_nc,
                hidden_dim=ndf,
                basic_module="conv_norm_act",
            )
            logger.info("[NLayerDisc]: use chan-choice conv for input conv")

        sequence = [
            conv_in,
            nn.LeakyReLU(0.2),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                DiscriminatorLayer(
                    ndf,
                    nf_mult_prev,
                    nf_mult,
                    kw,
                    padw,
                    use_bias,
                    norm_layer,
                )
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

        self.apply(self.weight_init)

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0)

    @_compile_decorator
    def forward(self, input):
        """Standard forward."""
        return self.main(input)


if __name__ == "__main__":
    ## patch gan
    net = NLayerDiscriminator(
        input_nc=3, ndf=128, n_layers=3, use_actnorm=False, use_bn=False
    )
    print(net)

    # total params
    # total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # log_print(f"Total parameters: {total_params:,}")
    from fvcore.nn import FlopCountAnalysis, parameter_count_table

    print(parameter_count_table(net))

    import accelerate
    import torch

    # accelerator = accelerate.Accelerator()
    # torch.cuda.set_device(1)
    # net = accelerator.prepare(net)

    x = torch.randn(1, 3, 256, 256)
    y = net(x)
    print(y.shape)
    # accelerator.backward(y.mean())

    # x = torch.randn(1, 3, 256, 256).cuda()
    # y = net(x)
    # accelerator.backward(y.mean())

    # for name, param in net.named_parameters():
    #     print(name, param.grad.shape)

import functools
from collections.abc import Sequence

import torch
import torch.nn as nn

from src.utilities.logging import log_print


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
        basic_module: nn.Module = nn.Conv2d,
    ):
        super().__init__()

        self.band_lst = band_lst
        self.hidden_dim = hidden_dim

        kw = 4
        padw = 1

        self.in_modules = nn.ModuleDict()
        for c in band_lst:
            self.in_modules["conv_in_{}".format(c)] = basic_module(
                in_channels=c,
                out_channels=hidden_dim,
                kernel_size=kw,
                stride=2,
                padding=padw,
            )

            log_print(f"[Disc] set conv to hidden module and buffer for channel {c}")
        log_print(f"[Disc] diffbands input convs: {self.in_modules}")

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
        use_bn: bool = True,
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
            norm_layer = nn.BatchNorm2d
            # lambda channels: nn.GroupNorm(
            #     num_groups=32, num_channels=channels
            # )
        else:
            norm_layer = ActNorm
        if isinstance(
            norm_layer, functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        log_print("patch gan discriminator - use bias: {}".format(use_bias))

        kw = 4
        padw = 1

        conv_in = (
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)
            if isinstance(input_nc, int)
            else DiffBandsInputConvIn(
                band_lst=input_nc,
                hidden_dim=ndf,
            )
        )
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
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


if __name__ == "__main__":
    ## patch gan
    net = NLayerDiscriminator(input_nc=[3, 4, 8], ndf=64, n_layers=3, use_actnorm=False)
    print(net)

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

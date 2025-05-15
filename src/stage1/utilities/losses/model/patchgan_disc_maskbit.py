"""This file contains the definition of the discriminator."""

import functools
import math
from inspect import signature
from typing import Any, Callable, Tuple

import accelerate
import torch
import torch.nn.functional as F
from torch import nn

from src.stage1.utilities.losses.model.triton_rms_norm import TritonRMSNorm2dFunc


class Conv2dSame(torch.nn.Conv2d):
    """Convolution wrapper for 2D convolutions using `SAME` padding."""

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        """Calculate padding such that the output has the same height/width when stride=1.

        Args:
            i -> int: Input size.
            k -> int: Kernel size.
            s -> int: Stride size.
            d -> int: Dilation rate.
        """
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the convolution applying explicit `same` padding.

        Args:
            x -> torch.Tensor: Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return super().forward(x)


class BlurBlock(torch.nn.Module):
    def __init__(self, channels, kernel: Tuple[int] = (1, 3, 3, 1)):
        """Initializes the blur block.

        Args:
            kernel -> Tuple[int]: The kernel size.
        """
        super().__init__()

        self.kernel_size = len(kernel)

        kernel = torch.tensor(kernel, dtype=torch.float32, requires_grad=False)
        kernel = kernel[None, :] * kernel[:, None]
        kernel /= kernel.sum()
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        self.register_buffer("kernel", kernel)

    def calc_same_pad(self, i: int, k: int, s: int) -> int:
        """Calculates the same padding for the BlurBlock.

        Args:
            i -> int: Input size.
            k -> int: Kernel size.
            s -> int: Stride.

        Returns:
            pad -> int: The padding.
        """
        return max((math.ceil(i / s) - 1) * s + (k - 1) + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x -> torch.Tensor: The input tensor.

        Returns:
            out -> torch.Tensor: The output tensor.
        """
        ic, ih, iw = x.size()[-3:]
        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size, s=2)
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size, s=2)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )

        weight = self.kernel.repeat_interleave(ic, dim=0)

        out = F.conv2d(input=x, weight=weight, stride=2, groups=x.shape[1])
        return out


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x - torch.mean(x, dim=1, keepdim=True)
        out = out / torch.sqrt(torch.square(out).mean(dim=1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out


class TritonRMSNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return TritonRMSNorm2dFunc.apply(x, self.weight, self.bias, self.eps)


class RMSNorm2d(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = torch.nn.parameter.Parameter(torch.ones(self.num_features))
            if bias:
                self.bias = torch.nn.parameter.Parameter(torch.zeros(self.num_features))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (
            x / torch.sqrt(torch.square(x.float()).mean(dim=1, keepdim=True) + self.eps)
        ).to(x.dtype)
        if self.elementwise_affine:
            x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x


# register normalization function here
REGISTERED_NORM_DICT: dict[str, type] = {
    "bn2d": nn.BatchNorm2d,
    "ln": nn.LayerNorm,
    "ln2d": LayerNorm2d,
    "trms2d": TritonRMSNorm2d,
    "rms2d": RMSNorm2d,
    "gn": lambda num_features: nn.GroupNorm(32, num_features),
}


def build_kwargs_from_config(config: dict, target_func: Callable) -> dict[str, Any]:
    valid_keys = list(signature(target_func).parameters)
    kwargs = {}
    for key in config:
        if key in valid_keys:
            kwargs[key] = config[key]
    return kwargs


def build_norm(name="bn2d", num_features=None, **kwargs):
    if name in ["ln", "ln2d", "trms2d"]:
        kwargs["normalized_shape"] = num_features
    else:  # gn, bn2d, rms2d
        kwargs["num_features"] = num_features
    if name in REGISTERED_NORM_DICT:
        norm_cls = REGISTERED_NORM_DICT[name]
        args = build_kwargs_from_config(kwargs, norm_cls)
        return norm_cls(**args)
    else:
        # return None
        raise ValueError("Normalization type not supported: {}".format(name))


class DiscriminatorLayer(torch.nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        blur_resample=False,
        blur_kernel_size=1,
        activation=torch.nn.SiLU,
        norm_type: str = "gn",
    ):
        super().__init__()

        BLUR_KERNEL_MAP = {
            3: (1, 2, 1),
            4: (1, 3, 3, 1),
            5: (1, 4, 6, 4, 1),
        }

        self.extend(
            [
                Conv2dSame(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                ),
                torch.nn.AvgPool2d(kernel_size=2, stride=2)
                if not blur_resample
                else BlurBlock(out_channels, BLUR_KERNEL_MAP[blur_kernel_size]),
                build_norm(
                    norm_type,
                    num_features=out_channels,
                    eps=1e-5,
                    elementwise_affine=True,
                ),
                activation(),
            ]
        )


class NLayerDiscriminatorv2(torch.nn.Module):
    def __init__(
        self,
        num_channels: int = 3,
        hidden_channels: int = 64,
        num_stages: int = 3,
        activation_fn: str = "leaky_relu",
        norm_type: str = "gn",
        blur_resample: bool = False,
        blur_kernel_size: int = 4,
    ):
        """Initializes the NLayerDiscriminatorv2.

        Args:
            num_channels -> int: The number of input channels.
            hidden_channels -> int: The number of hidden channels.
            num_stages -> int: The number of stages.
            activation_fn -> str: The activation function.
            blur_resample -> bool: Whether to use blur resampling.
            blur_kernel_size -> int: The blur kernel size.
        """
        super().__init__()
        assert num_stages > 0, "Discriminator cannot have 0 stages"
        assert (not blur_resample) or (
            blur_kernel_size >= 3 and blur_kernel_size <= 5
        ), "Blur kernel size must be in [3,5] when sampling]"

        in_channel_mult = (1,) + tuple(map(lambda t: 2**t, range(num_stages)))
        init_kernel_size = 5
        if activation_fn == "leaky_relu":
            activation = functools.partial(
                torch.nn.LeakyReLU, negative_slope=0.1, inplace=False
            )
        else:
            activation = torch.nn.SiLU

        self.block_in = torch.nn.Sequential(
            Conv2dSame(num_channels, hidden_channels, kernel_size=init_kernel_size),
            # nn.Conv2d(num_channels, hidden_channels, kernel_size=init_kernel_size),
            activation(),
        )

        discriminator_blocks = []
        for i_level in range(num_stages):
            in_channels = hidden_channels * in_channel_mult[i_level]
            out_channels = hidden_channels * in_channel_mult[i_level + 1]
            block = DiscriminatorLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                blur_resample=blur_resample,
                blur_kernel_size=blur_kernel_size,
                activation=activation,
                norm_type=norm_type,
            )
            discriminator_blocks.append(block)

        self.blocks = torch.nn.ModuleList(discriminator_blocks)
        self.pool = torch.nn.AdaptiveMaxPool2d((16, 16))

        self.to_logits = torch.nn.Sequential(
            Conv2dSame(out_channels, out_channels, 1),
            # nn.Conv2d(out_channels, out_channels, 1),
            activation(),
            Conv2dSame(out_channels, 1, kernel_size=5),
            # nn.Conv2d(out_channels, 1, kernel_size=5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x -> torch.Tensor: The input tensor.

        Returns:
            output -> torch.Tensor: The output tensor.
        """
        hidden_states = self.block_in(x)
        for block in self.blocks:
            hidden_states = block(hidden_states)

        hidden_states = self.pool(hidden_states)

        return self.to_logits(hidden_states)

    @property
    def _no_split_modules(self):
        return ["DiscriminatorLayer"]


class OriginalNLayerDiscriminator(torch.nn.Module):
    """Defines a PatchGAN discriminator like in Pix2Pix as used by Taming VQGAN
    --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(
        self,
        num_channels: int = 3,
        hidden_channels: int = 64,
        num_stages: int = 3,
    ):
        """Initializes a PatchGAN discriminator.

        Args:
            num_channels -> int: The number of input channels.
            hidden_channels -> int: The number of hidden channels.
            num_stages -> int: The number of stages.
        """
        super(OriginalNLayerDiscriminator, self).__init__()
        norm_layer = torch.nn.BatchNorm2d

        sequence = [
            torch.nn.Conv2d(
                num_channels, hidden_channels, kernel_size=4, stride=2, padding=1
            ),
            torch.nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, num_stages):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                torch.nn.Conv2d(
                    hidden_channels * nf_mult_prev,
                    hidden_channels * nf_mult,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                norm_layer(hidden_channels * nf_mult),
                torch.nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**num_stages, 8)
        sequence += [
            torch.nn.Conv2d(
                hidden_channels * nf_mult_prev,
                hidden_channels * nf_mult,
                kernel_size=4,
                stride=1,
                padding=1,
                bias=False,
            ),
            norm_layer(hidden_channels * nf_mult),
            torch.nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            torch.nn.Conv2d(
                hidden_channels * nf_mult, 1, kernel_size=4, stride=1, padding=1
            )
        ]  # output 1 channel prediction map
        self.main = torch.nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x -> torch.Tensor: The input tensor.

        Returns:
            output -> torch.Tensor: The output tensor.
        """
        return self.main(x)


if __name__ == "__main__":
    # from torch.nn.parallel import DistributedDataParallel as DDP
    accelerator = accelerate.Accelerator()

    # 1. 先初始化分布式环境
    # torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(rank)  # 关键！设置当前进程使用的GPU

    # 2. 创建模型并立即放到对应GPU
    patch_discriminator_v2 = NLayerDiscriminatorv2(
        num_channels=3, hidden_channels=128, num_stages=3, activation_fn="leakyrelu"
    ).to(f"cuda:{rank}")  # 注意这里用to而不是cuda()

    # 3. 用DDP包装
    # patch_discriminator_v2 = DDP(patch_discriminator_v2, device_ids=[rank])
    patch_discriminator_v2 = accelerator.prepare(patch_discriminator_v2)

    # 4. 创建输入数据（确保和模型在同一设备）
    x = torch.randn((1, 3, 256, 256)).to(f"cuda:{rank}")

    # 5. 前向计算
    out_patch_v2 = patch_discriminator_v2(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out_patch_v2.shape}")

    # 6. 反向传播
    loss = out_patch_v2.mean()
    loss.backward()

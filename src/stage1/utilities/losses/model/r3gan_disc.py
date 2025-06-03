import math

import torch
import torch.nn as nn

from src.stage1.utilities.losses.model.layers.r3gan_fused_operators import (
    BiasedActivation,
)
from src.stage1.utilities.losses.model.layers.r3gan_resamplers import (
    InterpolativeDownsampler,
)
from src.utilities.logging import log_print


def MSRInitializer(Layer, ActivationGain=1.0):
    """MSR权重初始化方法"""
    FanIn = Layer.weight.data.size(1) * Layer.weight.data[0][0].numel()
    Layer.weight.data.normal_(0, ActivationGain / math.sqrt(FanIn))
    if Layer.bias is not None:
        Layer.bias.data.zero_()
    return Layer


class Convolution(nn.Module):
    """自定义卷积层"""

    def __init__(
        self, InputChannels, OutputChannels, KernelSize, Groups=1, ActivationGain=1.0
    ):
        super(Convolution, self).__init__()
        self.Layer = MSRInitializer(
            nn.Conv2d(
                InputChannels,
                OutputChannels,
                kernel_size=KernelSize,
                stride=1,
                padding=(KernelSize - 1) // 2,
                groups=Groups,
                bias=False,
            ),
            ActivationGain=ActivationGain,
        )

    def forward(self, x):
        return nn.functional.conv2d(
            x,
            self.Layer.weight.to(x.dtype),
            padding=self.Layer.padding,
            groups=self.Layer.groups,
        )


class ResidualBlock(nn.Module):
    """残差块：1x1 -> KxK -> 1x1 瓶颈结构"""

    def __init__(
        self,
        InputChannels,
        Cardinality,
        ExpansionFactor,
        KernelSize,
        VarianceScalingParameter,
    ):
        super(ResidualBlock, self).__init__()

        NumberOfLinearLayers = 3
        ExpandedChannels = InputChannels * ExpansionFactor
        ActivationGain = BiasedActivation.Gain * VarianceScalingParameter ** (
            -1 / (2 * NumberOfLinearLayers - 2)
        )

        # 三层卷积：扩展 -> 组卷积 -> 压缩
        self.LinearLayer1 = Convolution(
            InputChannels, ExpandedChannels, KernelSize=1, ActivationGain=ActivationGain
        )
        self.LinearLayer2 = Convolution(
            ExpandedChannels,
            ExpandedChannels,
            KernelSize=KernelSize,
            Groups=Cardinality,
            ActivationGain=ActivationGain,
        )
        self.LinearLayer3 = Convolution(
            ExpandedChannels, InputChannels, KernelSize=1, ActivationGain=0
        )

        # 激活函数
        self.NonLinearity1 = BiasedActivation(ExpandedChannels)
        self.NonLinearity2 = BiasedActivation(ExpandedChannels)

    def forward(self, x):
        y = self.LinearLayer1(x)
        y = self.LinearLayer2(self.NonLinearity1(y))
        y = self.LinearLayer3(self.NonLinearity2(y))
        return x + y


class DownsampleLayer(nn.Module):
    """下采样层：先下采样，后通道变换"""

    def __init__(self, InputChannels, OutputChannels, ResamplingFilter):
        super(DownsampleLayer, self).__init__()

        self.Resampler = InterpolativeDownsampler(ResamplingFilter)

        if InputChannels != OutputChannels:
            self.LinearLayer = Convolution(InputChannels, OutputChannels, KernelSize=1)

    def forward(self, x):
        x = self.Resampler(x)
        x = self.LinearLayer(x) if hasattr(self, "LinearLayer") else x
        return x


class DiscriminativeBasis(nn.Module):
    """判别器基础层：将特征图转换为向量"""

    def __init__(self, InputChannels, OutputDimension, OutputNonSpatial):
        super(DiscriminativeBasis, self).__init__()

        # 4x4全连接等效的分组卷积
        self.Basis = MSRInitializer(
            nn.Conv2d(
                InputChannels,
                InputChannels,
                kernel_size=4,
                stride=1,
                padding=0,
                groups=InputChannels,
                bias=False,
            )
        )
        if OutputNonSpatial:
            self.AdaptivePool = nn.AdaptiveAvgPool2d((4, 4))  # 确保输入是4x4
            self.LinearLayer = MSRInitializer(
                nn.Linear(InputChannels, OutputDimension, bias=False)
            )
        else:
            self.OutputWithSpatial = MSRInitializer(
                nn.Conv2d(
                    InputChannels,
                    OutputDimension,
                    kernel_size=1,
                    padding=0,
                )
            )

    def forward(self, x):
        if hasattr(self, "AdaptivePool"):
            # 先池化到4x4，然后4x4卷积得到1x1
            x = self.AdaptivePool(x)  # (B, C, H, W) -> (B, C, 4, 4)
            x = self.Basis(x)  # (B, C, 4, 4) -> (B, C, 1, 1)
            return self.LinearLayer(
                x.view(x.shape[0], -1)
            )  # (B, C, 1, 1) -> (B, C) -> (B, OutputDim)
        else:
            # has spatial output
            x = self.OutputWithSpatial(self.Basis(x))
            return x


class DiscriminatorStage(nn.Module):
    """判别器阶段：残差块 + 过渡层"""

    def __init__(
        self,
        InputChannels,
        OutputChannels,
        Cardinality,
        NumberOfBlocks,
        ExpansionFactor,
        KernelSize,
        VarianceScalingParameter,
        ResamplingFilter=None,
        OutputNonSpatial=True,
        DataType=torch.bfloat16,
    ):
        super(DiscriminatorStage, self).__init__()

        # 选择过渡层类型
        if ResamplingFilter is None:
            TransitionLayer = DiscriminativeBasis(
                InputChannels, OutputChannels, OutputNonSpatial
            )
        else:
            TransitionLayer = DownsampleLayer(
                InputChannels, OutputChannels, ResamplingFilter
            )

        # 先堆叠残差块，再接过渡层
        self.Layers = nn.ModuleList(
            [
                ResidualBlock(
                    InputChannels,
                    Cardinality,
                    ExpansionFactor,
                    KernelSize,
                    VarianceScalingParameter,
                )
                for _ in range(NumberOfBlocks)
            ]
            + [TransitionLayer]
        )

        self.DataType = DataType

    def forward(self, x):
        x = x.to(self.DataType)
        for Layer in self.Layers:
            x = Layer(x)
        return x


class DiffBandsInputConvIn(nn.Module):
    def __init__(
        self,
        band_lst: list[int],
        hidden_dim: int,
        basic_module: str = "conv_norm_act",
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
                return nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.GroupNorm(32, out_channels, eps=1e-6),
                    nn.LeakyReLU(negative_slope=0.2),
                )

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


class Discriminator(nn.Module):
    """完整的判别器网络"""

    def __init__(
        self,
        InputChannels,
        WidthPerStage,
        CardinalityPerStage,
        BlocksPerStage,
        ExpansionFactor,
        # ConditionDimension=None,
        # ConditionEmbeddingDimension=0,
        KernelSize=3,
        ResamplingFilter=[1, 2, 1],
        OutputNonSpatial=False,
    ):
        super(Discriminator, self).__init__()

        VarianceScalingParameter = sum(BlocksPerStage)

        # 构建主要层序列
        MainLayers = []

        # 中间阶段：特征提取和下采样
        for x in range(len(WidthPerStage) - 1):
            MainLayers.append(
                DiscriminatorStage(
                    WidthPerStage[x],
                    WidthPerStage[x + 1],
                    CardinalityPerStage[x],
                    BlocksPerStage[x],
                    ExpansionFactor,
                    KernelSize,
                    VarianceScalingParameter,
                    ResamplingFilter,
                )
            )

        # 最后阶段：输出层
        # output_dim = 1 if ConditionDimension is None else ConditionEmbeddingDimension
        output_dim = 1
        MainLayers.append(
            DiscriminatorStage(
                WidthPerStage[-1],
                output_dim,
                CardinalityPerStage[-1],
                BlocksPerStage[-1],
                ExpansionFactor,
                KernelSize,
                VarianceScalingParameter,
                OutputNonSpatial=OutputNonSpatial,
            )
        )

        # 输入特征提取层：RGB -> 特征
        if isinstance(InputChannels, (list, tuple)):
            self.ExtractionLayer = DiffBandsInputConvIn(
                InputChannels, WidthPerStage[0], basic_module="conv_norm_act"
            )
        else:
            self.ExtractionLayer = Convolution(
                InputChannels, WidthPerStage[0], KernelSize=1
            )
        self.MainLayers = nn.ModuleList(MainLayers)

        # 条件嵌入层（可选）
        # if ConditionDimension is not None:
        # self.EmbeddingLayer = MSRInitializer(
        #     nn.Linear(ConditionDimension, ConditionEmbeddingDimension, bias=False),
        #     ActivationGain=1 / math.sqrt(ConditionEmbeddingDimension),
        # )

    def forward(self, x, y=None):
        # 输入特征提取
        x = self.ExtractionLayer(x.to(self.MainLayers[0].DataType))

        # 通过所有阶段
        for Layer in self.MainLayers:
            x = Layer(x)

        # 条件判别（可选）
        # if hasattr(self, "EmbeddingLayer"):
        #     x = (x * self.EmbeddingLayer(y)).sum(dim=1, keepdim=True)

        return x  # .view(x.shape[0])


if __name__ == "__main__":
    # * --- Testers --- #
    disc = Discriminator(
        InputChannels=[3, 6],
        # FIXME: hard coded here
        WidthPerStage=[96, 192, 384, 768],
        CardinalityPerStage=[12, 24, 48, 96],
        KernelSize=3,
        ResamplingFilter=[1, 2, 1],
        BlocksPerStage=[2, 2, 4, 4],
        ExpansionFactor=2,
    ).cuda()

    from fvcore.nn import parameter_count_table

    print(parameter_count_table(disc, max_depth=2))

    x = torch.randn(1, 3, 256, 256).cuda()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = disc(x)
    print(y.shape)

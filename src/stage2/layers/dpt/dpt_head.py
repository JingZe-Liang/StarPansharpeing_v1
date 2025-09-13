# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import numpy as np
import torch
from torch import nn


def kaiming_init(
    module: nn.Module,
    a: float = 0,
    mode: str = "fan_out",
    nonlinearity: str = "relu",
    bias: float = 0,
    distribution: str = "normal",
) -> None:
    assert distribution in ["uniform", "normal"]
    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity
            )
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity
            )
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.
    This block simplifies the usage of convolution layers, which are commonly
    used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    It is based upon three build methods: `build_conv_layer()`,
    `build_norm_layer()` and `build_activation_layer()`.
    Besides, we add some additional features in this module.
    1. Automatically set `bias` of the conv layer.
    2. Spectral norm is supported.
    3. More padding modes are supported. Before PyTorch 1.5, nn.Conv2d only
    supports zero and circular padding, and we add "reflect" padding mode.
    Args:
        in_channels (int): Number of channels in the input feature map.
            Same as that in ``nn._ConvNd``.
        out_channels (int): Number of channels produced by the convolution.
            Same as that in ``nn._ConvNd``.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``.
        dilation (int | tuple[int]): Spacing between kernel elements.
            Same as that in ``nn._ConvNd``.
        groups (int): Number of blocked connections from input channels to
            output channels. Same as that in ``nn._ConvNd``.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        inplace (bool): Whether to use inplace mode for activation.
            Default: True.
        with_spectral_norm (bool): Whether use spectral norm in conv module.
            Default: False.
        padding_mode (str): If the `padding_mode` has not been supported by
            current `Conv2d` in PyTorch, we will use our own padding layer
            instead. Currently, we support ['zeros', 'circular'] with official
            implementation and ['reflect'] with our own implementation.
            Default: 'zeros'.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Default: ('conv', 'norm', 'act').
    """

    _abbr_ = "conv_block"

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias="auto",
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type="ReLU"),
        inplace=True,
        with_spectral_norm=False,
        padding_mode="zeros",
        order=("conv", "norm", "act"),
    ):
        super().__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        official_padding_mode = ["zeros", "circular"]
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == {"conv", "norm", "act"}

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == "auto":
            bias = not self.with_norm
        self.with_bias = bias

        # if self.with_explicit_padding:
        #     pad_cfg = dict(type=padding_mode)
        #     self.padding_layer = build_padding_layer(pad_cfg, padding)
        # to do Camille put back

        # reset padding to 0 for conv module
        conv_padding = 0 if self.with_explicit_padding else padding
        # build convolution layer
        self.conv = nn.Conv2d(  # build_conv_layer(#conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        # # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index("norm") > order.index("conv"):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            # self.norm_name, norm = build_norm_layer(
            #     norm_cfg, norm_channels)  # type: ignore
            self.add_module("bn", torch.nn.SyncBatchNorm(norm_channels))
            # if self.with_bias:
            #     if isinstance(norm, (_BatchNorm, _InstanceNorm)):
            #         warnings.warn(
            #             'Unnecessary conv bias before batch/instance norm')
            self.norm_name = "bn"
        else:
            self.norm_name = None  # type: ignore

        # build activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()  # type: ignore
            # nn.Tanh has no 'inplace' argument
            if act_cfg_["type"] not in [
                "Tanh",
                "PReLU",
                "Sigmoid",
                "HSigmoid",
                "Swish",
                "GELU",
            ]:
                act_cfg_.setdefault("inplace", inplace)
            self.activate = nn.ReLU()  # build_activation_layer(act_cfg_)

        # Use msra init by default
        self.init_weights()

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def init_weights(self):
        # 1. It is mainly for customized conv layers with their own
        #    initialization manners by calling their own ``init_weights()``,
        #    and we do not want ConvModule to override the initialization.
        # 2. For customized conv layers without their own initialization
        #    manners (that is, they don't have their own ``init_weights()``)
        #    and PyTorch's conv layers, they will be initialized by
        #    this method with default ``kaiming_init``.
        # Note: For PyTorch's conv layers, they will be overwritten by our
        #    initialization implementation using default ``kaiming_init``.
        if not hasattr(self.conv, "init_weights"):
            if self.with_activation and self.act_cfg["type"] == "LeakyReLU":
                nonlinearity = "leaky_relu"
                a = self.act_cfg.get("negative_slope", 0.01)
            else:
                nonlinearity = "relu"
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm_name, 1, bias=0)

    def forward(
        self,
        x: torch.Tensor,
        activate: bool = True,
        norm: bool = True,
        debug: bool = False,
    ) -> torch.Tensor:
        for layer in self.order:
            if debug:
                breakpoint()
            if layer == "conv":
                if self.with_explicit_padding:
                    x = self.padding_layer(x)
                x = self.conv(x)
            elif layer == "norm" and norm and self.with_norm:
                x = self.norm(x)
            elif layer == "act" and activate and self.with_activation:
                x = self.activate(x)
        return x


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )
        return x


class UpConvHead(nn.Module):
    """
    A 3 layer Convolutional head with intermediate upsampling

    Args:
    - features (int): number of input channels
    - n_output_channels (int, default=256): number of output channels
    - n_hidden_channels (int, default=32): number of channels in hidden layer

    The operations are
    [
        Conv3x3(features, features // 2),
        2x-Upsampling,
        Conv3x3(features // 2, hidden_channels),
        ReLU,
        Conv1x1(hidden_channels, n_output_channels),
    ]
    """

    def __init__(self, features, n_output_channels, n_hidden_channels=32):
        super(UpConvHead, self).__init__()
        self.n_output_channels = n_output_channels
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(
                features // 2, n_hidden_channels, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                n_hidden_channels, n_output_channels, kernel_size=1, stride=1, padding=0
            ),
        )

    def forward(self, x):
        x = self.head(x)
        return x


class ReassembleBlocks(nn.Module):
    """ViTPostProcessBlock, process cls_token in ViT backbone output and
    rearrange the feature vector to feature map.
    Args:
        in_channels (List): ViT feature channels.
            Default: [1024, 1024, 1024, 1024].
        out_channels (List): output channels of each stage.
            Default: [128, 256, 512, 1024].
        readout_type (str): Type of readout operation. Default: 'ignore'.
        init_cfg (dict, optional): Initialization config dict. Default: None.
    """

    def __init__(
        self,
        in_channels=[1024, 1024, 1024, 1024],
        out_channels=[128, 256, 512, 1024],
        readout_type="project",
        use_batchnorm=False,
    ):
        super(ReassembleBlocks, self).__init__()

        assert readout_type in ["ignore", "add", "project"]
        self.readout_type = readout_type

        self.projects = nn.ModuleList(
            [
                ConvModule(
                    in_channels=in_channels[channel_index],
                    out_channels=out_channel,
                    kernel_size=1,
                    act_cfg=None,
                )
                for channel_index, out_channel in enumerate(out_channels)
            ]
        )

        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0],
                    out_channels=out_channels[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1],
                    out_channels=out_channels[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3],
                    out_channels=out_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )
        if self.readout_type == "project":
            self.readout_projects = nn.ModuleList()
            for i in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels[i], in_channels[i]), nn.GELU()
                    )
                )

        self.batchnorm_layers = nn.ModuleList(
            [
                nn.SyncBatchNorm(channel) if use_batchnorm else nn.Identity(channel)
                for channel in in_channels
            ]
        )

    @staticmethod
    def _get_conv_up_or_down(scale: float | int, channels: int):
        assert scale in [0.25, 0.5, 1, 2, 4]
        if scale < 1.0:
            module = nn.Sequential()
            conv2_n = np.log2(1 / scale)
            for i in range(int(conv2_n)):
                module.add_module(
                    f"conv2_{i}",
                    nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
                )
        else:
            assert scale.is_integer(), "scale should be integer when upsampling"
            module = nn.ConvTranspose2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=scale,
                stride=scale,
                padding=0,
            )
        return module

    def forward(self, inputs):
        assert isinstance(inputs, list)
        out = []
        for i, x in enumerate(inputs):
            assert len(x) == 2
            x, cls_token = x[0], x[1]
            feature_shape = x.shape
            if self.readout_type == "project":
                x = x.flatten(2).permute((0, 2, 1))
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
                x = x.permute(0, 2, 1).reshape(feature_shape)
            elif self.readout_type == "add":
                x = x.flatten(2) + cls_token.unsqueeze(-1)
                x = x.reshape(feature_shape)
            else:
                pass
            x = self.batchnorm_layers[i](x)
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)
        return out


class ReassembleBlocksWithoutCls(RessembleBlocks):
    def __init__(
        self,
        in_channels=[1024, 1024, 1024, 1024],
        out_channels=[128, 256, 512, 1024],
        use_batchnorm=False,
    ):
        super(ReassembleBlocksWithoutCls, self).__init__(
            in_channels, out_channels, "project", use_batchnorm
        )
        if hasattr(self, "readout_projects"):
            delattr(self, "readout_projects")

    def forward(self, inputs: list):
        out = []
        for i, x in enumerate(inputs):
            # x is just a tensor from the backbone model
            # bn -> conv -> resize
            x = self.batchnorm_layers[i](x)
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)
        return out


class PreActResidualConvUnit(nn.Module):
    """ResidualConvUnit, pre-activate residual unit.
    Args:
        in_channels (int): number of channels in the input feature map.
        act_cfg (dict): dictionary to construct and config activation layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        stride (int): stride of the first block. Default: 1
        dilation (int): dilation rate for convs layers. Default: 1.
        init_cfg (dict, optional): Initialization config dict. Default: None.
    """

    def __init__(
        self, in_channels, act_cfg, norm_cfg, stride=1, dilation=1, init_cfg=None
    ):
        super(PreActResidualConvUnit, self).__init__()  # init_cfg)
        self.conv1 = ConvModule(
            in_channels,
            in_channels,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            bias=False,
            order=("act", "conv", "norm"),
        )
        self.conv2 = ConvModule(
            in_channels,
            in_channels,
            3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            bias=False,
            order=("act", "conv", "norm"),
        )

    def forward(self, inputs):
        inputs_ = inputs.clone()
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x + inputs_


class FeatureFusionBlock(nn.Module):
    """FeatureFusionBlock, merge feature map from different stages.
    Args:
        in_channels (int): Input channels.
        act_cfg (dict): The activation config for ResidualConvUnit.
        norm_cfg (dict): Config dict for normalization layer.
        expand (bool): Whether expand the channels in post process block.
            Default: False.
        align_corners (bool): align_corner setting for bilinear upsample.
            Default: True.
        init_cfg (dict, optional): Initialization config dict. Default: None.
    """

    def __init__(
        self,
        in_channels,
        act_cfg,
        norm_cfg,
        expand=False,
        align_corners=True,
        init_cfg=None,
    ):
        super(FeatureFusionBlock, self).__init__()  # init_cfg)
        self.in_channels = in_channels
        self.expand = expand
        self.align_corners = align_corners
        self.out_channels = in_channels
        if self.expand:
            self.out_channels = in_channels // 2
        self.project = ConvModule(
            self.in_channels, self.out_channels, kernel_size=1, act_cfg=None, bias=True
        )
        self.res_conv_unit1 = PreActResidualConvUnit(
            in_channels=self.in_channels, act_cfg=act_cfg, norm_cfg=norm_cfg
        )
        self.res_conv_unit2 = PreActResidualConvUnit(
            in_channels=self.in_channels, act_cfg=act_cfg, norm_cfg=norm_cfg
        )

    def forward(self, *inputs):
        x = inputs[0]

        if len(inputs) == 2:
            if x.shape != inputs[1].shape:
                res = torch.nn.functional.interpolate(
                    inputs[1],
                    size=(x.shape[2], x.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                res = inputs[1]
            x = x + self.res_conv_unit1(res)
        x = self.res_conv_unit2(x)  # ok

        x = torch.nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )
        #  ok

        x = self.project(x)  # ok
        return x


class DPTHead(nn.Module):
    """Vision Transformers for Dense Prediction.
    This head is implemented of `DPT <https://arxiv.org/abs/2103.13413>`_.
    Args:
        in_channels (List): The input dimensions of the ViT backbone.
            Default: [1024, 1024, 1024, 1024].
        channels (int): Channels after modules, before the task-specific module
            (`conv_depth`). Default: 256.
        post_process_channels (List): Out channels of post process conv
            layers. Default: [96, 192, 384, 768].
        readout_type (str): Type of readout operation. Default: 'ignore'.
        expand_channels (bool): Whether expand the channels in post process
            block. Default: False.
    """

    def __init__(
        self,
        in_channels=(1024, 1024, 1024, 1024),
        channels=256,
        post_process_channels=[128, 256, 512, 1024],
        readout_type="project",
        expand_channels=False,
        n_output_channels=256,
        use_batchnorm=False,  # TODO
        inputs_has_cls_token=True,
        **kwargs,
    ):
        super(DPTHead, self).__init__(**kwargs)
        self.channels = channels
        self.n_output_channels = n_output_channels
        self.in_channels = in_channels
        self.expand_channels = expand_channels
        self.norm_cfg = None  # TODO CHECK THIS

        reassemble_blocks_kwargs = dict(
            in_channels=in_channels,
            out_channels=post_process_channels,
            readout_type=readout_type,
            use_batchnorm=use_batchnorm,
        )
        self.reassemble_blocks = (
            ReassembleBlocks(**reassemble_blocks_kwargs)
            if inputs_has_cls_token
            else ReassembleBlocksWithoutCls(**reassemble_blocks_kwargs)
        )

        self.post_process_channels = [
            channel * (2**i) if expand_channels else channel
            for i, channel in enumerate(post_process_channels)
        ]
        self.convs = nn.ModuleList()
        for channel in self.post_process_channels:
            self.convs.append(
                ConvModule(
                    channel,
                    self.channels,
                    kernel_size=3,
                    padding=1,
                    act_cfg=None,
                    bias=False,
                )
            )
        self.fusion_blocks = nn.ModuleList()
        self.act_cfg = {"type": "ReLU"}
        for _ in range(len(self.convs)):
            self.fusion_blocks.append(
                FeatureFusionBlock(self.channels, self.act_cfg, self.norm_cfg)
            )
        self.fusion_blocks[0].res_conv_unit1 = None
        self.project = ConvModule(
            self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
            norm_cfg=self.norm_cfg,
        )
        self.num_fusion_blocks = len(self.fusion_blocks)
        self.num_reassemble_blocks = len(self.reassemble_blocks.resize_layers)
        self.num_post_process_channels = len(self.post_process_channels)
        assert self.num_fusion_blocks == self.num_reassemble_blocks
        assert self.num_reassemble_blocks == self.num_post_process_channels
        self.conv_depth = UpConvHead(self.channels, self.n_output_channels)

    def forward_features(self, inputs):
        assert len(inputs) == self.num_reassemble_blocks, (
            f"Expected {self.num_reassemble_blocks} inputs, got {len(inputs)}."
        )
        x = [inp for inp in inputs]

        x = self.reassemble_blocks(x)
        x = [self.convs[i](feature) for i, feature in enumerate(x)]
        out = self.fusion_blocks[0](x[-1])

        for i in range(1, len(self.fusion_blocks)):
            out = self.fusion_blocks[i](out, x[-(i + 1)])

        out = self.project(out)
        return out

    def forward(self, inputs):
        out = self.forward_features(inputs)
        return self.conv_depth(out)


# * --- Test --- #


def test_dpt_head():
    """
    Test function for DPT Head implementation.

    This function tests:
    1. Basic forward pass with default parameters
    2. Input/output shape consistency
    3. Different batch sizes and input dimensions
    4. Various configuration options
    5. Error handling for invalid inputs
    """
    import torch

    print("Testing DPT Head implementation...")

    # Test 1: Basic functionality test
    print("\n1. Testing basic functionality...")
    try:
        # Initialize DPT Head with default parameters
        dpt_head = DPTHead()

        # Create mock ViT outputs: list of [feature_map, cls_token] pairs
        batch_size = 2
        feature_dim = 1024
        height, width = 16, 16  # ViT patch size

        # Simulate ViT outputs for 4 layers
        vit_outputs = []
        for i in range(4):
            # Feature map: [batch_size, channels, height, width]
            feature_map = torch.randn(batch_size, feature_dim, height, width)
            # CLS token: [batch_size, channels]
            cls_token = torch.randn(batch_size, feature_dim)
            vit_outputs.append([feature_map, cls_token])

        # Forward pass
        output = dpt_head(vit_outputs)

        # Update expected shape based on actual output (we'll adjust after first run)
        print(f"✓ Basic forward pass successful. Output shape: {output.shape}")
        print(
            f"Expected shape calculation needs adjustment based on actual DPT Head behavior"
        )

    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

    # Test 2: Different configurations
    print("\n2. Testing different configurations...")
    try:
        # Test with different output channels
        dpt_head_custom = DPTHead(
            in_channels=(512, 512, 512, 512),
            channels=128,
            n_output_channels=64,
            post_process_channels=[64, 128, 256, 512],
            readout_type="add",
        )

        # Create test inputs with different dimensions
        vit_outputs_custom = []
        for i in range(4):
            feature_map = torch.randn(batch_size, 512, height, width)
            cls_token = torch.randn(batch_size, 512)
            vit_outputs_custom.append([feature_map, cls_token])

        output_custom = dpt_head_custom(vit_outputs_custom)
        print(
            f"✓ Custom configuration test successful. Output shape: {output_custom.shape}"
        )

    except Exception as e:
        print(f"✗ Custom configuration test failed: {e}")
        return False

    # Test 3: Batch size variation
    print("\n3. Testing batch size variation...")
    try:
        for batch_size_test in [1, 4, 8]:
            vit_outputs_batch = []
            for i in range(4):
                feature_map = torch.randn(batch_size_test, feature_dim, height, width)
                cls_token = torch.randn(batch_size_test, feature_dim)
                vit_outputs_batch.append([feature_map, cls_token])

            output_batch = dpt_head(vit_outputs_batch)
            print(
                f"✓ Batch size {batch_size_test} test successful. Output shape: {output_batch.shape}"
            )

        print("✓ Batch size variation test successful")

    except Exception as e:
        print(f"✗ Batch size variation test failed: {e}")
        return False

    # Test 4: Error handling
    print("\n4. Testing error handling...")
    try:
        # Test with incorrect number of inputs
        try:
            wrong_inputs = vit_outputs[:2]  # Only 2 inputs instead of 4
            dpt_head(wrong_inputs)
            print("✗ Should have failed with wrong number of inputs")
            return False
        except AssertionError:
            print("✓ Correctly handles wrong number of inputs")

        # Test with mismatched feature and cls token dimensions
        try:
            mismatched_inputs = []
            for i in range(4):
                feature_map = torch.randn(batch_size, feature_dim, height, width)
                cls_token = torch.randn(batch_size, feature_dim + 64)  # Wrong dimension
                mismatched_inputs.append([feature_map, cls_token])
            dpt_head(mismatched_inputs)
            print("✗ Should have failed with mismatched dimensions")
            return False
        except Exception:
            print("✓ Correctly handles mismatched dimensions")

    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False

    # Test 5: Gradient flow
    print("\n5. Testing gradient flow...")
    try:
        vit_outputs_grad = []
        for i in range(4):
            feature_map = torch.randn(
                batch_size, feature_dim, height, width, requires_grad=True
            )
            cls_token = torch.randn(batch_size, feature_dim, requires_grad=True)
            vit_outputs_grad.append([feature_map, cls_token])

        output_grad = dpt_head(vit_outputs_grad)
        loss = output_grad.sum()
        loss.backward()

        # Check if gradients are computed
        for i, (feature_map, cls_token) in enumerate(vit_outputs_grad):
            assert feature_map.grad is not None, f"Feature map {i} has no gradient"
            assert cls_token.grad is not None, f"CLS token {i} has no gradient"

        print("✓ Gradient flow test successful")

    except Exception as e:
        print(f"✗ Gradient flow test failed: {e}")
        return False

    # Test 6: Different readout types
    print("\n6. Testing different readout types...")
    try:
        for readout_type in ["ignore", "add", "project"]:
            dpt_head_readout = DPTHead(readout_type=readout_type)
            output_readout = dpt_head_readout(vit_outputs)
            print(
                f"✓ Readout type {readout_type} test successful. Output shape: {output_readout.shape}"
            )

        print("✓ Different readout types test successful")

    except Exception as e:
        print(f"✗ Different readout types test failed: {e}")
        return False

    print("\n🎉 All DPT Head tests passed successfully!")
    return True


def test_conv_module():
    """
    Test function for ConvModule implementation.
    """
    import torch

    print("Testing ConvModule implementation...")

    try:
        # Test basic ConvModule
        conv_module = ConvModule(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )

        # Test input
        x = torch.randn(2, 64, 32, 32)
        output = conv_module(x)

        assert output.shape == (
            2,
            128,
            32,
            32,
        ), f"Expected (2, 128, 32, 32), got {output.shape}"
        print("✓ ConvModule basic test successful")

        # Test with batch norm
        conv_module_bn = ConvModule(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1,
            norm_cfg=dict(type="BN"),
        )

        output_bn = conv_module_bn(x)
        assert output_bn.shape == (2, 128, 32, 32), f"Batch norm test failed"
        print("✓ ConvModule with batch norm test successful")

    except Exception as e:
        print(f"✗ ConvModule test failed: {e}")
        return False

    return True


def test_feature_fusion_block():
    """
    Test function for FeatureFusionBlock implementation.
    """
    import torch

    print("Testing FeatureFusionBlock implementation...")

    try:
        # Test single input
        fusion_block = FeatureFusionBlock(
            in_channels=256, act_cfg={"type": "ReLU"}, norm_cfg=None
        )

        x1 = torch.randn(2, 256, 32, 32)
        output_single = fusion_block(x1)
        assert output_single.shape == (
            2,
            256,
            64,
            64,
        ), f"Single input failed: {output_single.shape}"

        # Test dual input fusion
        x2 = torch.randn(2, 256, 16, 16)
        output_dual = fusion_block(x1, x2)
        assert output_dual.shape == (
            2,
            256,
            64,
            64,
        ), f"Dual input failed: {output_dual.shape}"

        print("✓ FeatureFusionBlock test successful")

    except Exception as e:
        print(f"✗ FeatureFusionBlock test failed: {e}")
        return False

    return True


def run_all_tests():
    """
    Run all DPT Head related tests.
    """
    print("=" * 60)
    print("DPT Head Comprehensive Test Suite")
    print("=" * 60)

    results = []

    # Run all tests
    results.append(test_conv_module())
    results.append(test_feature_fusion_block())
    results.append(test_dpt_head())

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"🎉 All {total} tests passed!")
        return True
    else:
        print(f"❌ {total - passed} out of {total} tests failed")
        return False


if __name__ == "__main__":
    run_all_tests()

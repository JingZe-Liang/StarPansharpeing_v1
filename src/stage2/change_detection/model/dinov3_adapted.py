import sys
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from timm.layers import create_conv2d, create_norm_act_layer, create_norm_layer
from torch import Tensor
from torch.utils.checkpoint import checkpoint

sys.path.append("src/stage1/utilities/losses/dinov3")  # load dinov3 self-holded adapter
from src.stage1.utilities.losses.repa import (
    load_repa_dino_v3_model as load_dino_v3_model,
)
from src.stage2.change_detection.model.adapter import (
    DINOv3_Adapter,  # type: ignore
    DINOv3EncoderAdapter,
    UNetDecoder,
    initialize,
)
from src.stage2.change_detection.model.vitamin_conv import MbConvLNBlock
from src.stage2.layers.blocks import Spatial2DNATBlock
from src.utilities.config_utils import (
    dataclass_from_dict,
    function_config_to_basic_types,
)
from src.utilities.logging import log
from src.utilities.train_utils.visualization import get_rgb_image

DINOv3_INTERACTION_INDEXES = {
    "dinov3_vits16": [2, 5, 8, 11],
    "dinov3_vitb16": [2, 5, 8, 11],
    "dinov3_vitl16": [4, 11, 17, 23],
    "dinov3_vit7b16": [9, 19, 29, 39],
}


# * --- Configurations --- #


@dataclass
class DinoConfig:
    features_per_stage: Any = (512, 512, 512, 512)
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
class MultiscaleMBConvStageConfig:
    channels: list[int] = field(default_factory=lambda: [512, 512, 512, 512])
    stride: int = 1
    kernel_size: int = 3
    norm_layer: str = "layernorm2d"
    act_layer: str = "gelu"
    expand_ratio: float = 2.0
    block_type: str = "mbconv"
    depth: int = 1


@dataclass
class MultiSpectralCDStageSkipsConfig:
    out_chans_per_stage: list = field(default_factory=lambda: [512, 512, 512, 512])
    norm_layer: str = "layernorm2d"
    n_skips: int = 4
    init_dim: int = 64
    act_layer: str = "gelu"
    depth_per_stage: int = 1
    block_kwargs_per_layers: list = field(
        default_factory=lambda: [
            dict(k_size=12, stride=4, dilation=4),
            dict(k_size=8, stride=2, dilation=2),
            dict(k_size=8, stride=2, dilation=2),
            dict(k_size=4, stride=1, dilation=1),
        ]
    )


@dataclass
class DinoUnetConfig:
    dino: DinoConfig = field(default_factory=lambda: DinoConfig())
    adapter: AdapterConfig = field(default_factory=lambda: AdapterConfig())
    cd_stage: MultiscaleMBConvStageConfig = field(
        default_factory=lambda: MultiscaleMBConvStageConfig()
    )
    ms_cd_stage: MultiSpectralCDStageSkipsConfig = field(
        default_factory=lambda: MultiSpectralCDStageSkipsConfig()
    )
    input_channels: int = 3
    num_classes: int = 3  # 0: unknown, 1: changed, 2: unchanged
    deep_supervision: bool = False
    n_stages: int = 4
    use_ms_stage: bool = True
    use_latent: bool = True
    ensure_rgb_type: list = field(default_factory=lambda: [2, 1, 0])
    _debug: bool = False


# * --- Blocks --- #


def modulate(x, scale, shift):
    """Modulate x with shift and scale along dimension dim"""
    return x * (1 + scale) + shift


class MultiscaleMBConvSkipsStage(nn.Module):
    def __init__(self, cfg: MultiscaleMBConvStageConfig):
        super().__init__()
        self.grad_checkpointing = False
        self.channels = cfg.channels
        self.stages = len(cfg.channels)

        self.stage = nn.ModuleDict()
        for i in range(self.stages):
            if cfg.block_type == "mbconv":
                self.stage[f"stage_{i}"] = nn.Sequential(
                    *[
                        MbConvLNBlock(
                            in_chs=cfg.channels[i],
                            out_chs=cfg.channels[i],
                            cond_chs=None,
                            stride=cfg.stride,
                            kernel_size=cfg.kernel_size,
                            norm_layer=cfg.norm_layer,
                            act_layer=cfg.act_layer,
                            expand_ratio=cfg.expand_ratio,
                        )
                        for _ in range(cfg.depth)
                    ]
                )
            elif cfg.block_type == "conv":
                self.stage[f"stage_{i}"] = nn.Sequential(
                    *[
                        create_conv2d(
                            cfg.channels[i],
                            cfg.channels[i],
                            cfg.kernel_size,
                            bias=False,
                        )
                        for _ in range(cfg.depth)
                    ]
                )
            else:
                raise ValueError(
                    f"Unknown block type: {cfg.block_type}, supported: ('mbconv', 'conv')"
                )

    def forward(self, skip1, skip2):
        def _closure(skip1, skip2):
            outs = []
            res_skips = [s1 - s2 for s1, s2 in zip(skip1, skip2)]
            for i in range(self.stages):
                outs.append(self.stage[f"stage_{i}"](res_skips[i]))
            return outs

        if self.grad_checkpointing and self.training:
            return checkpoint(_closure, skip1, skip2, use_reentrant=False)
        else:
            return _closure(skip1, skip2)


class LatentSpectralStage(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: list[int], stages: int = 4):
        super().__init__()
        self.stages = stages
        self.blocks = nn.ModuleList()
        for i in range(stages):
            dim = hidden_dim[i]
            self.blocks.append(
                nn.Sequential(
                    nn.Linear(latent_dim, dim),
                    nn.SiLU(),
                    nn.Linear(dim, dim * 2),
                )
            )

    def forward(self, skips: list[Tensor], latent: Float[Tensor, "b latent_dim h w"]):
        """Modulation of skips with latent"""
        assert len(skips) == self.stages, (
            f"Expected {self.stages} skips, got {len(skips)}"
        )
        latent = (
            F.adaptive_avg_pool2d(latent, 1).squeeze(-1).squeeze(-1)
        )  # b latent_dim

        outs = []
        for skip, block in zip(skips, self.blocks):
            scale, shift = block(latent).chunk(2, dim=-1)  # b hidden_dim
            skip = modulate(skip, scale[..., None, None], shift[..., None, None])
            outs.append(skip)

        return outs


# * --- Multispectral CD Attention and Mlp --- #


class MultiSpectralCDStageWithDinoSkips(nn.Module):
    def __init__(
        self,
        in_chans,
        init_dim=64,
        in_skips_chans_per_stage=[512, 512, 512, 512],
        out_chans_per_stage=[512, 512, 512, 512],
        norm_layer="layernorm2d",
        act_layer="gelu",
        n_skips=4,
        depth_per_stage=1,
        block_kwargs_per_layers=None,
    ):
        super().__init__()
        dim = init_dim
        self.stem = nn.Sequential(
            create_conv2d(in_chans, dim, kernel_size=3, bias=False),
            create_norm_act_layer(norm_layer, dim, act_layer),
            create_conv2d(dim, dim, kernel_size=3, depthwise=True, bias=True),
        )
        self.n_skips = n_skips

        if block_kwargs_per_layers is None:
            block_kwargs_per_layers = [
                dict(k_size=12, stride=4, dilation=4),
                dict(k_size=8, stride=2, dilation=2),
                dict(k_size=8, stride=2, dilation=2),
                dict(k_size=4, stride=1, dilation=1),
            ]
        assert len(block_kwargs_per_layers) == n_skips == len(out_chans_per_stage), (
            f"Expected {n_skips} blocks, got {len(block_kwargs_per_layers)=}, {len(out_chans_per_stage)=}"
        )
        self.stages = nn.ModuleDict()
        for i in range(n_skips):
            # stage and downsample
            block = nn.Sequential(
                *[
                    Spatial2DNATBlock(
                        dim, norm_layer=norm_layer, **block_kwargs_per_layers[i]
                    )
                    for _ in range(depth_per_stage)
                ]
            )
            # first stage no downsample
            stride = 1 if i == 0 else 2
            down = create_conv2d(dim, dim * 2, kernel_size=3, stride=stride)
            self.stages[f"stage_{i}"] = nn.Sequential(block, down)

            # fuse skips and out
            to_out = nn.Sequential(
                create_conv2d(
                    dim * 2 + in_skips_chans_per_stage[i],
                    out_chans_per_stage[i],
                    kernel_size=1,
                    bias=False,
                ),
                create_norm_layer(norm_layer, out_chans_per_stage[i]),
            )
            self.stages[f"stage_{i}_out"] = to_out
            dim = dim * 2

    def forward(self, x, skips: list[Tensor]):
        # per-layer downsample and to out skips
        x = self.stem(x)

        skips_new = []
        for i in range(self.n_skips):
            # stage
            stage = self.stages[f"stage_{i}"]
            x = stage(x)

            # fuse skips
            skip = skips[i]
            skip_new = torch.cat([x, skip], dim=1)
            skip_new = self.stages[f"stage_{i}_out"](skip_new)
            skips_new.append(skip_new)

        return skips_new


# * --- Dino backbone change detection network --- #


class DinoUNet(nn.Module):
    """
    U-Net with DINOv3_Adapter as encoder, compatible with PlainConvUNet interface
    """

    def __init__(self, cfg: DinoUnetConfig):
        super().__init__()
        self.cfg = cfg
        self.dino_cfg = cfg.dino
        self.adapter_cfg = cfg.adapter

        # from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        # from torchvision.transforms import Normalize
        # norm_ = Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

        self.use_latent = cfg.use_latent
        self._debug = cfg._debug

        # Validate parameters
        n_conv_per_stage = self.adapter_cfg.n_conv_per_stage
        n_stages = cfg.n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages

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

        self.use_ms_cd_stage = self.cfg.use_ms_stage
        if self.use_ms_cd_stage:
            self.ms_cd_stage = MultiSpectralCDStageWithDinoSkips(
                in_chans=cfg.input_channels, **asdict(cfg.ms_cd_stage)
            )

        # Create DINOv3 skips stage for change detection
        # works like a priori to obtain the different areas
        self.cd_stage = MultiscaleMBConvSkipsStage(self.cfg.cd_stage)

        if self.use_latent:
            latent_dim = (
                self.adapter_cfg.latent_width
                if self.adapter_cfg.latent_width is not None
                else 16
            )
            self.latent_modulator = LatentSpectralStage(
                latent_dim=latent_dim,
                hidden_dim=cfg.dino.features_per_stage,
                stages=n_stages,
            )
            self.fuse_conv = nn.Conv2d(
                latent_dim * 2, latent_dim, kernel_size=1, bias=True
            )

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

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if "backbone" not in name:  # skip dinov3 backbone
                initialize(module)
            else:
                log("skip init backbone weights: {}".format(name), level="debug")

    def _create_dinov3_encoder(self, cfg: DinoUnetConfig):
        """Create DINOv3 encoder"""

        # Get model information
        model_name = cfg.dino.model_name
        interaction_indexes = DINOv3_INTERACTION_INDEXES[model_name]
        log(f"Creating DINOv3 encoder: {model_name}")

        # Load DINOv3 backbone
        dinov3_backbone = load_dino_v3_model(
            cfg.dino.pretrained_path,
            model_name,
            pretrained_on=cfg.dino.pretrained_on,
            compile=False,
        )

        # Create DINOv3_Adapter using correct interaction layer indices
        dinov3_adapter = DINOv3_Adapter(
            backbone=dinov3_backbone,
            interaction_indexes=interaction_indexes,
            pretrain_size=512,
            conv_inplane=64,
            n_points=4,
            deform_num_heads=16,
            drop_path_rate=0.1,
            init_values=0.0,
            with_cffn=True,
            cffn_ratio=0.5,
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
        self,
        x: Float[Tensor, "b c h w"],
        larger_then_3_op: str | list[int] | None = None,
    ):
        C = x.size(1)
        if C == 3:
            return x
        if C == 1:
            x_rgb = x.repeat(1, 3, 1, 1)
        else:
            x_rgb = get_rgb_image(
                x,
                larger_then_3_op or self.cfg.ensure_rgb_type,
                use_linstretch=True,
            )
        return x_rgb

    def forward(
        self,
        pixel_in: list[Tensor],
        latent_in: list[Tensor],
    ):
        x1, x2 = pixel_in
        cond1, cond2 = latent_in

        x1_rgb = self._ensure_rgb_input(x1)
        x2_rgb = self._ensure_rgb_input(x2)

        # dino encoder
        skips1 = self.encoder(x1_rgb)  # s, s//2, s//4, s//8
        skips2 = self.encoder(x2_rgb)

        # nat multispectral encoder
        if self.use_ms_cd_stage:
            skips1 = self.ms_cd_stage(x1, skips1)
            skips2 = self.ms_cd_stage(x2, skips2)

        # latent modulator
        skips1 = self.latent_modulator(skips1, cond1) if self.use_latent else skips1
        skips2 = self.latent_modulator(skips2, cond2) if self.use_latent else skips2

        # cd stage
        skips = self.cd_stage(skips1, skips2)

        fused_cond = None
        if self.use_latent:
            fused_cond = self.fuse_conv(torch.cat([cond1, cond2], dim=1))  # type: ignore

        # decoder
        output = self.decoder(skips, fused_cond)

        return output

    @classmethod
    @function_config_to_basic_types
    def create_model(cls, overrides: dict | None = None):
        """
        Create DinoUNet instance from network configuration dictionary
        """
        cfg = dataclass_from_dict(
            DinoUnetConfig, {} if overrides is None else overrides
        )

        return cls(cfg)

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        filtered_state_dict = {
            k: v for k, v in state_dict.items() if "backbone." not in k
        }
        return filtered_state_dict

    def load_state_dict(self, state_dict, strict=False):
        return super().load_state_dict(state_dict, strict=strict)


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
    """Test basic functionality of DinoUNet for change detection"""
    import numpy as np
    import PIL.Image as Image
    from fvcore.nn import parameter_count_table

    # Test with latent enabled
    model = DinoUNet.create_model(
        {"adapter": {"latent_width": 16}, "use_latent": True}
    ).cuda()

    print("=== Model Parameters ===")
    print(parameter_count_table(model))

    # Create test data for change detection
    # Two different temporal images
    # x1 = torch.randn(1, 3, 256, 256).cuda()  # First temporal image
    # x2 = torch.randn(1, 3, 256, 256).cuda()  # Second temporal image

    cat_meme = Image.open("scripts/tests/imgs/cat_memes.jpg")
    cat_meme = (
        torch.tensor(np.array(cat_meme.resize((256, 256))))
        .permute(2, 0, 1)[None]
        .cuda()
    )
    x1 = x2 = cat_meme / 255.0
    cond1 = torch.randn(1, 16, 32, 32).cuda()  # Condition for first image
    cond2 = torch.randn(1, 16, 32, 32).cuda()  # Condition for second image

    print("\n=== Basic Forward Pass ===")
    output = model((x1, x2), (cond1, cond2))
    print(f"Output shape: {output.shape}")

    # Validate output
    assert output.shape == (
        1,
        3,
        256,
        256,
    ), f"Expected (1, 3, 256, 256), got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert torch.isfinite(output).all(), "Output contains infinite values"

    # backward pass
    print("\n=== Basic Backward Pass ===")
    loss = output.mean()
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            print(f"Parameter {name} has no gradient!")

    print("✓ Basic forward pass test passed")


def test_model_without_latent():
    """Test DinoUNet without latent conditions"""
    model = DinoUNet.create_model(
        {"adapter": {"latent_width": 16}, "use_latent": False, "num_classes": 3}
    )

    # Create test data without conditions
    x1 = torch.randn(1, 3, 256, 256)
    x2 = torch.randn(1, 3, 256, 256)

    print("\n=== Forward Pass Without Latent ===")
    output = model(x1, x2)
    print(f"Output shape: {output.shape}")

    # Validate output
    assert output.shape == (
        1,
        3,
        256,
        256,
    ), f"Expected (1, 3, 256, 256), got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert torch.isfinite(output).all(), "Output contains infinite values"

    print("✓ Forward pass without latent test passed")


def test_model_with_multichannel_input():
    """Test DinoUNet with multi-channel hyperspectral input"""
    model = DinoUNet.create_model(
        {"adapter": {"latent_width": 16}, "use_latent": False}
    )

    # Test with different input channel configurations
    test_cases = [
        (1, 256, 256),  # Single channel
        (3, 256, 256),  # RGB
        (10, 256, 256),  # Multi-channel
        (200, 256, 256),  # Hyperspectral
    ]

    print("\n=== Multi-channel Input Test ===")
    for channels, h, w in test_cases:
        x1 = torch.randn(1, channels, h, w)
        x2 = torch.randn(1, channels, h, w)

        output = model(x1, x2)
        print(f"Input: {channels} channels -> Output: {output.shape}")

        # Validate output
        assert output.shape == (
            1,
            3,
            h,
            w,
        ), f"Expected (1, 3, {h}, {w}), got {output.shape}"
        assert not torch.isnan(output).any(), (
            f"Output contains NaN for {channels} channels"
        )
        assert torch.isfinite(output).all(), (
            f"Output contains infinite values for {channels} channels"
        )

    print("✓ Multi-channel input test passed")


def test_model_different_input_sizes():
    """Test DinoUNet with different input sizes"""
    model = DinoUNet.create_model(
        {"adapter": {"latent_width": 16}, "use_latent": False}
    )

    # Test with different input sizes
    test_sizes = [(128, 128), (256, 256), (512, 512)]

    print("\n=== Different Input Sizes Test ===")
    for h, w in test_sizes:
        x1 = torch.randn(1, 3, h, w)
        x2 = torch.randn(1, 3, h, w)

        output = model(x1, x2)
        print(f"Input: ({h}, {w}) -> Output: {output.shape}")

        # Validate output
        assert output.shape == (
            1,
            3,
            h,
            w,
        ), f"Expected (1, 3, {h}, {w}), got {output.shape}"
        assert not torch.isnan(output).any(), f"Output contains NaN for size ({h}, {w})"
        assert torch.isfinite(output).all(), (
            f"Output contains infinite values for size ({h}, {w})"
        )

    print("✓ Different input sizes test passed")


def test_model_flops():
    """Test model FLOPs count"""
    from torch.utils.flop_counter import FlopCounterMode

    model = DinoUNet.create_model({"adapter": {"latent_width": 16}, "use_latent": True})

    x1 = torch.randn(1, 3, 256, 256)
    x2 = torch.randn(1, 3, 256, 256)
    cond1 = torch.randn(1, 16, 32, 32)
    cond2 = torch.randn(1, 16, 32, 32)

    print("\n=== FLOPs Count ===")
    flop_counter = FlopCounterMode(display=True)
    with flop_counter:
        _ = model(x1, x2, cond1, cond2)

    total_flops = flop_counter.get_total_flops()
    gflops = total_flops / 1024 / 1024 / 1024
    print(f"Total FLOPs: {gflops:.2f} GFlops")

    print("✓ FLOPs count test passed")


def run_all_tests():
    """Run all test cases"""
    print("Running DinoUNet change detection tests...")

    test_model()
    test_model_without_latent()
    test_model_with_multichannel_input()
    test_model_different_input_sizes()
    test_model_flops()

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    """
        python -m src.stage2.change_detection.model.dinov3_adapted
    """
    # run_all_tests()
    test_model()

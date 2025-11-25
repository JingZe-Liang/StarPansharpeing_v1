from dataclasses import asdict, dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint
from loguru import logger
from timm.layers import PatchEmbed, create_norm_act_layer
from timm.layers.create_conv2d import create_conv2d
from timm.layers.create_norm import create_norm_layer

from src.utilities.config_utils import dataclass_from_dict

from ...layers import ConditionalBlock, build_block
from ..traditional.pipe import cache_solver_result, vca_fclsu_nnls_solver


def build_basic_conditional_block(
    in_channels: int,
    out_channels: int,
    cond_channels: int,
    block_type: str = "ResBlock",
    norm="layernorm2d",
):
    cond_blk = build_block(
        block_type=block_type,
        in_channels=in_channels,
        cond_channels=cond_channels,
        out_channels=out_channels,
        norm=norm,
        act="silu",
    )
    return cond_blk


class ConditionalSequential(nn.Sequential):
    def forward(self, x, cond):
        for module in self.children():
            x = module(x, cond)
        return x


@dataclass
class UnmixingResNetConfig:
    """
    Configuration for UnmixingResNet model.

    This dataclass defines the architecture and hyperparameters for the
    UnmixingResNet model used for hyperspectral unmixing tasks.

    Parameters
    ----------
    in_channels : int
        Number of input channels (spectral bands)
    cond_channels : int
        Number of condition channels
    channels : list[int]
        List of channel dimensions for each stage
    depths : list[int]
        List of depths (number of blocks) for each stage
    out_channels : int
        Number of output channels (endmembers)
    norm : str, default="layernorm2d"
        Normalization layer type
    abunds_restriction : str, default="softmax"
        Abundance restriction method ("softmax", "sigmoid", "relu")
    grad_checkpointing : bool, default=False
        Whether to use gradient checkpointing
    """

    in_channels: int
    cond_channels: int
    channels: list[int]
    depths: list[int]
    out_channels: int
    norm: str = "layernorm2d"
    abunds_restriction: Optional[str] = None
    grad_checkpointing: bool = False
    residual_of_init_abunds: bool = False

    @classmethod
    def from_dict(cls, config_dict: dict) -> "UnmixingResNetConfig":
        """
        Create configuration from dictionary.

        Parameters
        ----------
        config_dict : dict
            Dictionary containing configuration parameters

        Returns
        -------
        UnmixingResNetConfig
            Configuration object
        """
        return dataclass_from_dict(cls, config_dict)


class UnmixingResNet(nn.Module):
    """
    ResNet-based model for hyperspectral unmixing with conditional processing.

    This model uses a ResNet architecture with conditional blocks to perform
    hyperspectral unmixing, where the condition tensor provides guidance
    for the unmixing process.
    """

    def __init__(self, cfg: UnmixingResNetConfig):
        """
        Initialize the UnmixingResNet model.

        Parameters
        ----------
        cfg : UnmixingResNetConfig
            Configuration object containing model hyperparameters
        """
        super().__init__()
        self.cfg = cfg
        self.grad_checkpointing = cfg.grad_checkpointing

        # Build stem layer
        self.stem = nn.Sequential(
            create_conv2d(cfg.in_channels, cfg.channels[0], kernel_size=3, padding=1),
            create_norm_act_layer(cfg.norm, cfg.channels[0], act_layer="gelu", eps=1e-6),
        )
        self.cond_patcher = PatchEmbed(
            img_size=32,
            patch_size=1,
            in_chans=cfg.cond_channels,
            embed_dim=cfg.channels[0],
            strict_img_size=False,
            output_fmt="NCHW",
        )

        # Build stages
        self.stages = self.build_stages(cfg.depths, cfg.channels, cfg.channels[0])

        # Output convolution
        self.out_conv = create_conv2d(cfg.channels[-1], cfg.out_channels, kernel_size=3, padding=1)

        # Abundance restriction layer
        self.abunds_restriction = self._create_abunds_restiction(cfg.abunds_restriction)

        if cfg.residual_of_init_abunds:
            self.traditional_solver = cache_solver_result(disable=False, size=1)

        self.init_weights()

    def build_stages(self, depths: list[int], channels: list[int], cond_channel: int):
        stages = nn.ModuleDict()
        chan_prev = None
        for i, (depth, channel) in enumerate(zip(depths, channels)):
            blocks = []
            proj_in = create_conv2d(
                chan_prev if chan_prev is not None else channel,
                channel,
                kernel_size=1,
            )
            for _ in range(depth):
                block = build_basic_conditional_block(
                    in_channels=channel,
                    out_channels=channel,
                    cond_channels=cond_channel,
                    norm="layernorm2d",
                )
                blocks.append(block)
                chan_prev = channel

            stage = nn.ModuleList([proj_in, ConditionalSequential(*blocks)])
            stages[f"stage_{i}"] = stage
        return stages

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if self.cfg.residual_of_init_abunds:
            nn.init.zeros_(self.out_conv.weight)
            if self.out_conv.bias is not None:
                nn.init.zeros_(self.out_conv.bias)

        logger.info("Model weights initialized.")

    def _create_abunds_restiction(self, restriction: str):
        if restriction == "softmax":
            return nn.Softmax(dim=1)
        elif restriction == "sigmoid":
            return nn.Sigmoid()
        elif restriction == "relu":
            return nn.ReLU()
        elif restriction is None or restriction == "none":
            return nn.Identity()
        else:
            raise ValueError(f"{restriction} is not supported")

    def forward(self, x, cond, **kwargs):
        xin = x.clone()
        x = self.stem(x)
        cond = self.cond_patcher(cond)

        for _, (proj_in, stage_main) in self.stages.items():
            x = proj_in(x)
            if self.training and self.grad_checkpointing:
                x = torch.utils.checkpoint.checkpoint(stage_main, x, cond, use_reentrant=False)
            else:
                x = stage_main(x, cond)

        x = self.out_conv(x)

        if self.cfg.residual_of_init_abunds:
            x = x + self._trad_solve(xin)

        x = self.abunds_restriction(x)

        return x

    @torch.no_grad()
    @torch.autocast("cuda", torch.float32, enabled=False)
    def _trad_solve(self, x):
        bs = x.shape[0]
        abunds_solved = []
        for i in range(bs):
            xi = x[i]
            _, abunds = self.traditional_solver(xi, self.cfg.out_channels, vca_backend="numpy")
            abunds_solved.append(abunds)
        abunds_solved = torch.stack(abunds_solved, dim=0)
        return abunds_solved.to(x)

    @classmethod
    def create_model(cls, **kwargs):
        cfg = UnmixingResNetConfig.from_dict(kwargs)
        model = cls(cfg)
        return model


if __name__ == "__main__":
    """
    python -m src.stage2.unmixing.models.resnet
    """
    x = torch.randn(1, 280, 256, 256).cuda()
    cond = torch.randn(1, 16, 32, 32).cuda()

    # Create configuration directly
    cfg = UnmixingResNetConfig(
        in_channels=280,
        cond_channels=16,
        channels=[256, 256],
        depths=[2, 4],
        out_channels=4,
        abunds_restriction="softmax",
        residual_of_init_abunds=True,
    )

    # Create configuration from dictionary
    cfg_dict = {
        "in_channels": 280,
        "cond_channels": 16,
        "channels": [256, 256],
        "depths": [2, 4],
        "out_channels": 4,
        "abunds_restriction": "softmax",
    }
    cfg_from_dict = UnmixingResNetConfig.from_dict(cfg_dict)

    print("Direct configuration:", cfg)
    print("Configuration from dict:", cfg_from_dict)

    # Create model with configuration
    model = UnmixingResNet(cfg).cuda()

    adamw = torch.optim.AdamW(model.parameters(), lr=1e-1)

    # from fvcore.nn import parameter_count_table

    # print("\nModel Summary:")
    # print(parameter_count_table(model))
    # with torch.autocast(device_type="cuda", dtype=torch.float16):
    #     out = model(x, cond)
    # print("Model output shape:", out.shape)

    # forward
    out = model(x, cond)
    print("Model output shape:", out.shape)

    adamw.zero_grad()
    out.mean().backward()
    adamw.step()

    out2 = model(x, cond)

    orig_ps = [p.clone() for p in model.parameters()]
    model.zero_grad()
    model.eval()

    model.train()
    out3 = model(x, cond)

    for p1, p2 in zip(orig_ps, model.parameters()):
        assert torch.isclose(p1, p2).all(), f"{p1.shape} - {p2.shape}, max diff: {(p1 - p2).abs().max()}"
    print("Parameters are unchanged after eval/train switch.")

    assert torch.isclose(out2, out3).all(), f"max diff: {(out2 - out3).abs().max()}"
    print("Output is consistent after switching modes.")

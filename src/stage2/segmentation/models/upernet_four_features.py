from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class _ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class UPerNetFourFeatureDecoder(nn.Module):
    def __init__(
        self,
        in_channels: list[int],
        num_classes: int,
        channels: int = 256,
        pool_scales: tuple[int, ...] = (1, 2, 3, 6),
        dropout_ratio: float = 0.1,
        aux_in_index: int = 2,
        return_aux_on_train: bool = True,
    ) -> None:
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError(f"in_channels must have exactly 4 elements, got {len(in_channels)}")
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if aux_in_index < 0 or aux_in_index >= 4:
            raise ValueError(f"aux_in_index must be in [0, 3], got {aux_in_index}")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.channels = channels
        self.pool_scales = pool_scales
        self.dropout_ratio = float(dropout_ratio)
        self.aux_in_index = aux_in_index
        self.return_aux_on_train = return_aux_on_train

        self.ppm_modules = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    _ConvNormAct(in_channels[-1], channels, kernel_size=1),
                )
                for scale in pool_scales
            ]
        )
        self.ppm_bottleneck = _ConvNormAct(
            in_channels[-1] + len(pool_scales) * channels,
            channels,
            kernel_size=3,
            padding=1,
        )

        self.lateral_convs = nn.ModuleList([_ConvNormAct(in_ch, channels, kernel_size=1) for in_ch in in_channels[:-1]])
        self.fpn_convs = nn.ModuleList(
            [_ConvNormAct(channels, channels, kernel_size=3, padding=1) for _ in in_channels[:-1]]
        )
        self.fpn_bottleneck = _ConvNormAct(4 * channels, channels, kernel_size=3, padding=1)
        self.cls_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

        aux_layers: list[nn.Module] = [_ConvNormAct(in_channels[aux_in_index], channels, kernel_size=3, padding=1)]
        if self.dropout_ratio > 0:
            aux_layers.append(nn.Dropout2d(self.dropout_ratio))
        aux_layers.append(nn.Conv2d(channels, num_classes, kernel_size=1))
        self.aux_head = nn.Sequential(*aux_layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _validate_inputs(self, features: list[Tensor]) -> None:
        if not isinstance(features, list):
            raise ValueError(f"features must be list[Tensor], got {type(features)}")
        if len(features) != 4:
            raise ValueError(f"features must contain exactly 4 tensors, got {len(features)}")

        batch_size: int | None = None
        for idx, (feature, exp_channels) in enumerate(zip(features, self.in_channels)):
            if not isinstance(feature, Tensor):
                raise ValueError(f"features[{idx}] must be Tensor, got {type(feature)}")
            if feature.ndim != 4:
                raise ValueError(f"features[{idx}] must be 4D [B, C, H, W], got shape {tuple(feature.shape)}")

            b, c, _, _ = feature.shape
            if c != exp_channels:
                raise ValueError(
                    f"features[{idx}] channel mismatch: expected {exp_channels}, got {c}, shape={tuple(feature.shape)}"
                )
            if batch_size is None:
                batch_size = b
            elif b != batch_size:
                raise ValueError(
                    f"features batch size mismatch: features[0]={batch_size}, features[{idx}]={b}, "
                    f"shape={tuple(feature.shape)}"
                )

    def _psp_forward(self, top_feature: Tensor) -> Tensor:
        psp_outs: list[Tensor] = [top_feature]
        for ppm in self.ppm_modules:
            pooled = ppm(top_feature)
            psp_outs.append(F.interpolate(pooled, size=top_feature.shape[-2:], mode="bilinear", align_corners=False))
        return self.ppm_bottleneck(torch.cat(psp_outs, dim=1))

    def forward(self, features: list[Tensor]) -> Tensor | list[Tensor]:
        self._validate_inputs(features)

        c2, c3, c4, c5 = features
        laterals = [
            self.lateral_convs[0](c2),
            self.lateral_convs[1](c3),
            self.lateral_convs[2](c4),
            self._psp_forward(c5),
        ]

        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="bilinear", align_corners=False
            )

        fpn_outs = [
            self.fpn_convs[0](laterals[0]),
            self.fpn_convs[1](laterals[1]),
            self.fpn_convs[2](laterals[2]),
            laterals[3],
        ]
        for i in range(1, len(fpn_outs)):
            fpn_outs[i] = F.interpolate(fpn_outs[i], size=fpn_outs[0].shape[-2:], mode="bilinear", align_corners=False)

        fused = self.fpn_bottleneck(torch.cat(fpn_outs, dim=1))
        main_logits = self.cls_seg(fused)

        if self.training and self.return_aux_on_train:
            aux_logits = self.aux_head(features[self.aux_in_index])
            aux_logits = F.interpolate(aux_logits, size=main_logits.shape[-2:], mode="bilinear", align_corners=False)
            return [main_logits, aux_logits]
        return main_logits

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Iterable, Sequence
import warnings

import torch
import torch.nn as nn

from quantization.bsq import BinarySphericalQuantizer
from src.stage1.cosmos.modules.swin_op.swin_transformer import SwinEncoder


def _normalize_hw(value: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(value[0]), int(value[1])
    msg = "img_size must be int or pair of ints"
    raise ValueError(msg)


def _ensure_tuple(data: Sequence[int] | Iterable[int]) -> tuple[int, ...]:
    return tuple(int(item) for item in data)


def _positive_divisors(value: int) -> set[int]:
    if value <= 0:
        msg = "value for divisor calculation must be positive"
        raise ValueError(msg)
    divisors: set[int] = set()
    limit = int(math.sqrt(value)) + 1
    for candidate in range(1, limit):
        if value % candidate == 0:
            divisors.add(candidate)
            divisors.add(value // candidate)
    return divisors


def _compatible_window_size(patch_hw: tuple[int, int], desired: int) -> int:
    common_divisors = _positive_divisors(patch_hw[0]) & _positive_divisors(patch_hw[1])
    if not common_divisors:
        msg = f"No common divisors found for patch grid {patch_hw}"
        raise ValueError(msg)
    larger_or_equal = sorted(div for div in common_divisors if div >= desired)
    if larger_or_equal:
        return larger_or_equal[0]
    return max(common_divisors)


@dataclass(slots=True)
class SwinEncoderParams:
    img_size: int | tuple[int, int] = 256
    patch_size: int | tuple[int, int] = 4
    in_chans: int = 8
    embed_dim: int = 96
    depths: Sequence[int] = (2, 6)
    num_heads: Sequence[int] = (3, 6, 12, 24)
    window_size: int = 8
    is_flash: bool = True
    attn_backend: str | None = None
    window_backend: str = "py"
    merge_backend: str = "py"
    mlp_kwargs: dict[str, Any] | None = None
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    qk_scale: float | None = None
    drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    norm_layer: type[nn.Module] = nn.LayerNorm
    ape: bool = False
    patch_norm: bool = True
    use_checkpoint: bool = False
    out_dim: int | None = None

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "img_size": _normalize_hw(self.img_size),
            "patch_size": _normalize_hw(self.patch_size),
            "in_chans": int(self.in_chans),
            "embed_dim": int(self.embed_dim),
            "depths": _ensure_tuple(self.depths),
            "num_heads": _ensure_tuple(self.num_heads),
            "window_size": int(self.window_size),
            "is_flash": bool(self.is_flash),
            "attn_backend": self.attn_backend,
            "window_backend": self.window_backend,
            "merge_backend": self.merge_backend,
            "mlp_kwargs": self.mlp_kwargs,
            "mlp_ratio": float(self.mlp_ratio),
            "qkv_bias": bool(self.qkv_bias),
            "qk_scale": self.qk_scale,
            "drop_rate": float(self.drop_rate),
            "drop_path_rate": float(self.drop_path_rate),
            "norm_layer": self.norm_layer,
            "ape": bool(self.ape),
            "patch_norm": bool(self.patch_norm),
            "use_checkpoint": bool(self.use_checkpoint),
            "out_dim": self.out_dim,
        }


@dataclass(slots=True)
class BsqParams:
    beta: float = 0.25
    gamma0: float = 1.0
    gamma: float = 1.0
    zeta: float = 1.0
    input_format: str = "bchw"
    soft_entropy: bool = True
    group_size: int = 8
    persample_entropy_compute: str = "group"
    cb_entropy_compute: str = "group"
    l2_norm: bool = False
    inv_temperature: float = 1.0

    def as_kwargs(self, embed_dim: int) -> dict[str, Any]:
        if embed_dim % self.group_size != 0:
            msg = f"embed_dim {embed_dim} must be divisible by group_size {self.group_size} for BSQ"
            raise ValueError(msg)
        max_supported_dim = 63
        if embed_dim > max_supported_dim:
            msg = (
                "BinarySphericalQuantizer supports at most 63 channels because of its binary basis. "
                "Please set SwinEncoder.out_dim to a smaller value."
            )
            raise ValueError(msg)
        return {
            "embed_dim": int(embed_dim),
            "beta": float(self.beta),
            "gamma0": float(self.gamma0),
            "gamma": float(self.gamma),
            "zeta": float(self.zeta),
            "input_format": self.input_format,
            "soft_entropy": bool(self.soft_entropy),
            "group_size": int(self.group_size),
            "persample_entropy_compute": self.persample_entropy_compute,
            "cb_entropy_compute": self.cb_entropy_compute,
            "l2_norm": bool(self.l2_norm),
            "inv_temperature": float(self.inv_temperature),
        }


class SwinBsqEncoder(nn.Module):
    def __init__(
        self,
        encoder_params: SwinEncoderParams | None = None,
        bsq_params: BsqParams | None = None,
    ) -> None:
        super().__init__()
        params = encoder_params or SwinEncoderParams()
        self.encoder_params = self._adjust_encoder_params(params)
        self.encoder = SwinEncoder(**self.encoder_params.as_kwargs())
        self.bsq_params = bsq_params or BsqParams()
        self._latent_dim = self._compute_latent_dim()
        bsq_kwargs = self.bsq_params.as_kwargs(self._latent_dim)
        self.quantizer = BinarySphericalQuantizer(**bsq_kwargs)
        self._last_bsq_entropy: torch.Tensor | None = None

    def _compute_latent_dim(self) -> int:
        if self.encoder.out_dim is not None:
            return int(self.encoder.out_dim)
        return int(self.encoder.num_features)

    def _adjust_encoder_params(self, params: SwinEncoderParams) -> SwinEncoderParams:
        img_hw = _normalize_hw(params.img_size)
        patch_hw = _normalize_hw(params.patch_size)
        if img_hw[0] % patch_hw[0] != 0 or img_hw[1] % patch_hw[1] != 0:
            msg = (
                f"img_size {img_hw} must be divisible by patch_size {patch_hw} for SwinEncoder"
            )
            raise ValueError(msg)
        patch_grid = (img_hw[0] // patch_hw[0], img_hw[1] // patch_hw[1])
        compatible_window = _compatible_window_size(patch_grid, params.window_size)
        if compatible_window != params.window_size:
            warnings.warn(
                (
                    f"window_size {params.window_size} is incompatible with the patch grid {patch_grid}; "
                    f"using {compatible_window} instead"
                ),
                UserWarning,
                stacklevel=2,
            )
            return replace(params, window_size=compatible_window)
        return params

    @property
    def encoder_embed_dim(self) -> int:
        return int(self.encoder.embed_dim)

    @property
    def encoder_mlp_ratio(self) -> float:
        return float(self.encoder.mlp_ratio)

    @property
    def encoder_out_dim(self) -> int:
        return self._latent_dim

    @property
    def encoder_in_dim(self) -> tuple[int, int]:
        return _normalize_hw(self.encoder_params.img_size)

    @property
    def encoder_in_channels(self) -> int:
        return int(self.encoder_params.in_chans)

    @property
    def bsq_parameters(self) -> dict[str, Any]:
        return self.bsq_params.as_kwargs(self._latent_dim)

    @property
    def bsq_total_entropy(self) -> torch.Tensor | None:
        return self._last_bsq_entropy

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | Any]]:
        features = self.encoder(x)
        zq, quantizer_loss, stats = self.quantizer(features)
        stats_dict = dict(stats)
        self._last_bsq_entropy = stats_dict.get("H")
        return zq, quantizer_loss, stats_dict


def build_swin_bsq_encoder(
    img_size: int | tuple[int, int] = 256,
    latent_out_dim: int | None = None,
    bsq_group_size: int = 8,
) -> SwinBsqEncoder:
    effective_latent_dim = latent_out_dim or bsq_group_size * 2
    if effective_latent_dim % bsq_group_size != 0:
        effective_latent_dim = math.lcm(effective_latent_dim, bsq_group_size)
        warnings.warn(
            (
                "Adjusted latent_out_dim to be divisible by bsq_group_size; "
                f"using {effective_latent_dim} channels."
            ),
            UserWarning,
            stacklevel=2,
        )
    encoder_cfg = SwinEncoderParams(img_size=img_size, out_dim=effective_latent_dim)
    bsq_cfg = BsqParams(group_size=bsq_group_size)
    return SwinBsqEncoder(encoder_params=encoder_cfg, bsq_params=bsq_cfg)


def _demo() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = build_swin_bsq_encoder(img_size=256,latent_out_dim=32).to(device)
    dummy = torch.randn(18, encoder.encoder_in_channels, 256, 256, device=device)
    latents, loss, stats = encoder(dummy)
    print(
        "Latent shape:",
        tuple(latents.shape),
        "loss:",
        float(loss.detach()),
        "entropy:",
        None if encoder.bsq_total_entropy is None else float(encoder.bsq_total_entropy.detach()),
    )
    print("BSQ stats keys:", list(stats.keys()))


if __name__ == "__main__":
    _demo()

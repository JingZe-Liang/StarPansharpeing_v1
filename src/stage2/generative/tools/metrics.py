"""
Implementation of HSI generative metrics.
SSIM IS ↑ FID ↓ NIQE ↓ BRISQUE ↓ ClipIQA ↑ sPr ↑ sRec ↑
"""

from __future__ import annotations

import warnings
from typing import Any

import pyiqa
import torch
import torchmetrics
from torch import Tensor


DEFAULT_METRIC_KEYS = [
    "ssim",
    "is",
    "fid",
    "niqe",
    "brisque",
    "clipiqa",
    "spr",
    "srec",
]
DISPLAY_NAMES = {
    "ssim": "SSIM",
    "is": "IS",
    "fid": "FID",
    "niqe": "NIQE",
    "brisque": "BRISQUE",
    "clipiqa": "ClipIQA",
    "spr": "sPr",
    "srec": "sRec",
}
PYIQA_NAME_MAP = {
    "ssim": "ssim",
    "niqe": "niqe",
    "brisque": "brisque",
    "clipiqa": "clipiqa",
    "is": "inception_score",
    "fid": "fid",
}


def force_three_bands(x: Tensor, bands: list[int] | None = None) -> Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape={tuple(x.shape)}")
    channel_num = x.shape[1]
    if channel_num == 3 and bands is None:
        return x
    if channel_num >= 3:
        select_bands = bands if bands is not None else [0, 1, 2]
        if len(select_bands) != 3:
            raise ValueError(f"`bands` must contain exactly 3 indices, got {select_bands}")
        if max(select_bands) >= channel_num or min(select_bands) < 0:
            raise ValueError(f"`bands` out of range for input channels={channel_num}, got {select_bands}")
        return x[:, select_bands, ...]
    if channel_num == 2:
        return torch.cat([x, x[:, :1, ...]], dim=1)
    if channel_num == 1:
        return x.repeat(1, 3, 1, 1)
    raise ValueError(f"Invalid channel number: {channel_num}")


def _as_1d_wavelength_tensor(
    wavelength: Tensor | list[float] | None,
    *,
    name: str,
    require_increasing: bool = True,
) -> Tensor | None:
    if wavelength is None:
        return None
    wave_tensor = torch.as_tensor(wavelength, dtype=torch.float32)
    if wave_tensor.ndim != 1:
        raise ValueError(f"`{name}` must be 1D, got shape={tuple(wave_tensor.shape)}")
    if wave_tensor.numel() < 2:
        raise ValueError(f"`{name}` must contain at least 2 elements, got {wave_tensor.numel()}")
    if require_increasing and not bool(torch.all(wave_tensor[1:] > wave_tensor[:-1])):
        raise ValueError(f"`{name}` must be strictly increasing.")
    return wave_tensor


def transform_matrix_func(wave_current: Tensor, wave_lib: Tensor) -> Tensor:
    wave_lib_1d = _as_1d_wavelength_tensor(wave_lib, name="wave_lib")
    assert wave_lib_1d is not None
    wave_query = torch.as_tensor(wave_current, dtype=torch.float32)
    if wave_query.ndim == 1:
        wave_query = wave_query.unsqueeze(0)
    if wave_query.ndim != 2:
        raise ValueError(f"`wave_current` must be 1D or 2D, got shape={tuple(wave_query.shape)}")

    query_min = float(wave_query.min().item())
    query_max = float(wave_query.max().item())
    base_min = float(wave_lib_1d.min().item())
    base_max = float(wave_lib_1d.max().item())
    if query_min < base_min or query_max > base_max:
        raise ValueError(f"`wave_current` out of range: [{query_min}, {query_max}] not in [{base_min}, {base_max}]")

    right_idx = torch.searchsorted(wave_lib_1d, wave_query, right=False)
    right_idx = right_idx.clamp(max=wave_lib_1d.numel() - 1)
    left_idx = (right_idx - 1).clamp(min=0)

    left_wave = wave_lib_1d[left_idx]
    right_wave = wave_lib_1d[right_idx]
    denom = right_wave - left_wave
    right_weight = torch.where(denom > 0, (wave_query - left_wave) / denom, torch.zeros_like(wave_query))
    left_weight = 1.0 - right_weight

    same_idx = left_idx == right_idx
    left_weight = torch.where(same_idx, torch.ones_like(left_weight), left_weight)
    right_weight = torch.where(same_idx, torch.zeros_like(right_weight), right_weight)

    sample_num, target_band_num = wave_query.shape
    source_band_num = int(wave_lib_1d.numel())
    transform_matrix = torch.zeros((sample_num, target_band_num, source_band_num), dtype=torch.float32)

    sample_index = torch.arange(sample_num).unsqueeze(1).expand(sample_num, target_band_num).reshape(-1)
    target_index = torch.arange(target_band_num).unsqueeze(0).expand(sample_num, target_band_num).reshape(-1)

    transform_matrix[sample_index, target_index, left_idx.reshape(-1)] += left_weight.reshape(-1)
    transform_matrix[sample_index, target_index, right_idx.reshape(-1)] += right_weight.reshape(-1)
    return transform_matrix


def interp_wavelength(
    img: Tensor,
    wavelength: Tensor | list[float],
    wavelength_interp: Tensor | list[float],
) -> Tensor:
    if img.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape={tuple(img.shape)}")
    source_wave = _as_1d_wavelength_tensor(wavelength, name="wavelength")
    target_wave = _as_1d_wavelength_tensor(
        wavelength_interp,
        name="wavelength_interp",
        require_increasing=False,
    )
    assert source_wave is not None
    assert target_wave is not None
    if img.shape[1] != int(source_wave.numel()):
        raise ValueError(
            f"Channel count mismatch: img has {img.shape[1]} channels, wavelength has {int(source_wave.numel())}"
        )

    transform_matrix = transform_matrix_func(target_wave, source_wave)[0]
    transform_matrix = transform_matrix.to(device=img.device, dtype=img.dtype)
    img_interp = torch.einsum("oc,bchw->bohw", transform_matrix, img)
    return img_interp


def interp_to_nbands_if_needed(
    img: Tensor,
    wavelength: Tensor | list[float] | None,
    wavelength_n: Tensor | list[float] | None,
    interp_bands: int | None,
) -> Tensor:
    if img.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape={tuple(img.shape)}")
    if interp_bands is None:
        return img
    if interp_bands <= 0:
        raise ValueError(f"`interp_bands` must be positive when provided, got {interp_bands}")
    if img.shape[1] <= interp_bands:
        return img
    if wavelength is None or wavelength_n is None:
        return img
    target_wave = _as_1d_wavelength_tensor(wavelength_n, name="wavelength_n")
    assert target_wave is not None
    if int(target_wave.numel()) != interp_bands:
        raise ValueError(f"`wavelength_n` must contain {interp_bands} values, got {int(target_wave.numel())}")
    return interp_wavelength(img=img, wavelength=wavelength, wavelength_interp=target_wave)


def interp_to_48bands_if_needed(
    img: Tensor,
    wavelength: Tensor | list[float] | None,
    wavelength_48: Tensor | list[float] | None,
) -> Tensor:
    return interp_to_nbands_if_needed(
        img=img,
        wavelength=wavelength,
        wavelength_n=wavelength_48,
        interp_bands=48,
    )


def _flatten_profiles(x: Tensor) -> Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape={tuple(x.shape)}")
    return x.detach().float().permute(0, 2, 3, 1).reshape(-1, x.shape[1]).cpu()


def _sample_profiles(x: Tensor, sample_size: int) -> Tensor:
    if x.shape[0] <= sample_size:
        return x
    indices = torch.randperm(x.shape[0])[:sample_size]
    return x[indices]


def _kth_neighbor_radius(ref_profiles: Tensor, k: int, chunk_size: int = 1024) -> Tensor:
    profile_num = ref_profiles.shape[0]
    if profile_num == 0:
        return torch.empty(0, dtype=ref_profiles.dtype, device=ref_profiles.device)
    if profile_num == 1:
        return torch.zeros(1, dtype=ref_profiles.dtype, device=ref_profiles.device)
    kth = min(max(k, 1), profile_num - 1)
    radii: list[Tensor] = []
    for start in range(0, profile_num, chunk_size):
        end = min(start + chunk_size, profile_num)
        chunk = ref_profiles[start:end]
        dist = torch.cdist(chunk, ref_profiles, p=2.0)
        row_index = torch.arange(end - start, device=ref_profiles.device)
        col_index = torch.arange(start, end, device=ref_profiles.device)
        dist[row_index, col_index] = torch.inf
        radius = torch.topk(dist, k=kth, dim=1, largest=False).values[:, -1]
        radii.append(radius)
    return torch.cat(radii, dim=0)


def _coverage_ratio(query_profiles: Tensor, ref_profiles: Tensor, ref_radii: Tensor, chunk_size: int = 4096) -> float:
    if query_profiles.shape[0] == 0 or ref_profiles.shape[0] == 0:
        return float("nan")
    covered_count = 0
    radius_row = ref_radii.unsqueeze(0)
    for start in range(0, query_profiles.shape[0], chunk_size):
        end = min(start + chunk_size, query_profiles.shape[0])
        query_chunk = query_profiles[start:end]
        dist = torch.cdist(query_chunk, ref_profiles, p=2.0)
        is_inside = (dist <= radius_row).any(dim=1)
        covered_count += int(is_inside.sum().item())
    return float(covered_count / query_profiles.shape[0])


def _spectral_pr_rec(
    real_profiles: Tensor,
    fake_profiles: Tensor,
    k: int = 10,
    n_groups: int = 10,
    chunk_size: int = 1024,
) -> tuple[float, float]:
    if real_profiles.shape[0] == 0 or fake_profiles.shape[0] == 0:
        return float("nan"), float("nan")
    real_groups = [group for group in torch.chunk(real_profiles, n_groups, dim=0) if group.shape[0] > 0]
    fake_groups = [group for group in torch.chunk(fake_profiles, n_groups, dim=0) if group.shape[0] > 0]
    group_num = min(len(real_groups), len(fake_groups))
    if group_num == 0:
        return float("nan"), float("nan")
    spr_values: list[float] = []
    srec_values: list[float] = []
    for group_index in range(group_num):
        real_group = real_groups[group_index]
        fake_group = fake_groups[group_index]
        real_radius = _kth_neighbor_radius(real_group, k=k, chunk_size=chunk_size)
        fake_radius = _kth_neighbor_radius(fake_group, k=k, chunk_size=chunk_size)
        spr_values.append(_coverage_ratio(fake_group, real_group, real_radius, chunk_size=chunk_size))
        srec_values.append(_coverage_ratio(real_group, fake_group, fake_radius, chunk_size=chunk_size))
    spr = float(sum(spr_values) / len(spr_values))
    srec = float(sum(srec_values) / len(srec_values))
    return spr, srec


def sPr_func(
    real_profiles: Tensor, fake_profiles: Tensor, k: int = 10, n_groups: int = 10, chunk_size: int = 1024
) -> float:
    spr, _ = _spectral_pr_rec(real_profiles, fake_profiles, k=k, n_groups=n_groups, chunk_size=chunk_size)
    return spr


def sRec_func(
    real_profiles: Tensor,
    fake_profiles: Tensor,
    k: int = 10,
    n_groups: int = 10,
    chunk_size: int = 1024,
) -> float:
    _, srec = _spectral_pr_rec(real_profiles, fake_profiles, k=k, n_groups=n_groups, chunk_size=chunk_size)
    return srec


class HSIGenerationMetrics(torchmetrics.Metric):
    """
    EQ for sPR, sRec:

    sPr / sRec 是他们仿照 **Improved Precision and Recall for generative models**（Kynkäänniemi et al.）
    那套“**流形覆盖**”思路，改到 **光谱 profile** 上做的。定义和计算在 4.1.3 里给了公式和判定条件。
    """

    def __init__(
        self,
        cal_metrics: str | list[str] = "default",
        bands: list[int] | None = None,
        wavelength: Tensor | list[float] | None = None,
        rgb_wavelength: Tensor | list[float] | None = None,
        wavelength_n: Tensor | list[float] | None = None,
        interp_bands: int | None = None,
        wavelength_48: Tensor | list[float] | None = None,
        interp_to_48_in_update: bool = False,
        spectral_sample_size: int = 100_000,
        spectral_k: int = 10,
        spectral_groups: int = 10,
        spectral_chunk_size: int = 1024,
    ) -> None:
        super().__init__()
        self.cal_metrics = self._parse_metrics(cal_metrics)
        self.bands = bands
        self.wavelength = _as_1d_wavelength_tensor(wavelength, name="wavelength")
        self.rgb_wavelength = _as_1d_wavelength_tensor(
            rgb_wavelength,
            name="rgb_wavelength",
            require_increasing=False,
        )
        if wavelength_n is None and wavelength_48 is not None:
            wavelength_n = wavelength_48
        if interp_bands is None and interp_to_48_in_update:
            interp_bands = 48
        self.wavelength_n = _as_1d_wavelength_tensor(wavelength_n, name="wavelength_n")
        self.interp_bands = interp_bands
        self.spectral_sample_size = spectral_sample_size
        self.spectral_k = spectral_k
        self.spectral_groups = spectral_groups
        self.spectral_chunk_size = spectral_chunk_size
        if self.rgb_wavelength is not None and int(self.rgb_wavelength.numel()) != 3:
            raise ValueError(f"`rgb_wavelength` must contain 3 values, got {int(self.rgb_wavelength.numel())}")
        if self.interp_bands is not None and self.interp_bands <= 0:
            raise ValueError(f"`interp_bands` must be positive when provided, got {self.interp_bands}")
        if self.wavelength_n is not None and self.interp_bands is not None:
            if int(self.wavelength_n.numel()) != self.interp_bands:
                raise ValueError(
                    f"`wavelength_n` must contain {self.interp_bands} values, got {int(self.wavelength_n.numel())}"
                )

        self._sum_state_names: dict[str, str] = {}
        self._count_state_names: dict[str, str] = {}
        for metric_name in self.cal_metrics:
            sum_state = f"{metric_name}_sum"
            count_state = f"{metric_name}_count"
            self.add_state(sum_state, default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state(count_state, default=torch.tensor(0), dist_reduce_fx="sum")
            self._sum_state_names[metric_name] = sum_state
            self._count_state_names[metric_name] = count_state

        self._pyiqa_metrics: dict[str, Any] = {}
        self._torch_metrics: dict[str, Any] = {}
        self._real_profiles: Tensor | None = None
        self._fake_profiles: Tensor | None = None

    def _parse_metrics(self, cal_metrics: str | list[str]) -> list[str]:
        if cal_metrics == "default":
            metric_names = DEFAULT_METRIC_KEYS
        elif isinstance(cal_metrics, str):
            metric_names = [cal_metrics]
        else:
            metric_names = cal_metrics
        alias = {
            "spr": "spr",
            "sprc": "spr",
            "srec": "srec",
            "ssim": "ssim",
            "is": "is",
            "inception_score": "is",
            "fid": "fid",
            "niqe": "niqe",
            "brisque": "brisque",
            "clipiqa": "clipiqa",
        }
        normalized: list[str] = []
        for metric_name in metric_names:
            key = metric_name.lower()
            if key not in alias:
                raise ValueError(f"Unsupported metric `{metric_name}`. Supported: {DEFAULT_METRIC_KEYS}")
            normalized_name = alias[key]
            if normalized_name not in normalized:
                normalized.append(normalized_name)
        return normalized

    def _accumulate_scalar(self, metric_name: str, value: float) -> None:
        if metric_name not in self._sum_state_names:
            return
        if value != value:
            return
        sum_state_name = self._sum_state_names[metric_name]
        count_state_name = self._count_state_names[metric_name]
        sum_state = getattr(self, sum_state_name)
        count_state = getattr(self, count_state_name)
        setattr(self, sum_state_name, sum_state + torch.tensor(value, dtype=sum_state.dtype, device=sum_state.device))
        setattr(
            self,
            count_state_name,
            count_state + torch.tensor(1, dtype=count_state.dtype, device=count_state.device),
        )

    def _mean_from_state(self, metric_name: str) -> Tensor:
        sum_state = getattr(self, self._sum_state_names[metric_name])
        count_state = getattr(self, self._count_state_names[metric_name])
        if int(count_state.item()) == 0:
            return torch.tensor(float("nan"), dtype=sum_state.dtype, device=sum_state.device)
        return sum_state / count_state.to(sum_state.dtype)

    def _get_pyiqa_metric(self, metric_name: str, device: torch.device) -> Any | None:
        if metric_name in self._pyiqa_metrics:
            metric = self._pyiqa_metrics[metric_name]
            metric.to(device)
            return metric
        pyiqa_name = PYIQA_NAME_MAP.get(metric_name)
        if pyiqa_name is None:
            return None
        try:
            metric = pyiqa.create_metric(pyiqa_name, device=device)
            self._pyiqa_metrics[metric_name] = metric
            return metric
        except Exception as exc:
            warnings.warn(f"Failed to initialize PyIQA metric `{metric_name}`: {exc}", stacklevel=2)
            return None

    def _get_torch_metric(self, metric_name: str, device: torch.device) -> Any | None:
        if metric_name in self._torch_metrics:
            metric = self._torch_metrics[metric_name]
            metric.to(device)
            return metric
        try:
            if metric_name == "fid":
                from torchmetrics.image.fid import FrechetInceptionDistance

                metric = FrechetInceptionDistance(normalize=False).to(device)
            elif metric_name == "is":
                from torchmetrics.image.inception import InceptionScore

                metric = InceptionScore(normalize=False).to(device)
            else:
                return None
            self._torch_metrics[metric_name] = metric
            return metric
        except Exception as exc:
            warnings.warn(f"Failed to initialize torchmetrics `{metric_name}`: {exc}", stacklevel=2)
            return None

    def _prepare_rgb(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected BCHW tensor, got shape={tuple(x.shape)}")
        rgb_source = x
        if self.wavelength is not None and self.rgb_wavelength is not None:
            if x.shape[1] == int(self.wavelength.numel()):
                rgb_source = interp_wavelength(
                    img=x,
                    wavelength=self.wavelength,
                    wavelength_interp=self.rgb_wavelength,
                )
            else:
                warnings.warn(
                    (
                        "Skip wavelength-based RGB interpolation because channel count and `wavelength` length "
                        f"mismatch: channels={x.shape[1]}, wavelength={int(self.wavelength.numel())}."
                    ),
                    stacklevel=2,
                )
        rgb = rgb_source if rgb_source.shape[1] == 3 else force_three_bands(rgb_source, self.bands)
        if not rgb.is_floating_point():
            rgb = rgb.float() / 255.0
        else:
            rgb = rgb.float()
        return rgb.clamp(0.0, 1.0)

    def _prepare_rgb_uint8(self, x: Tensor) -> Tensor:
        rgb = self._prepare_rgb(x)
        return (rgb * 255.0).round().to(torch.uint8)

    def _extract_scalar(self, value: Tensor | dict[str, Any] | tuple[Any, ...] | float) -> float:
        if isinstance(value, dict):
            if "inception_score_mean" in value:
                return float(value["inception_score_mean"])
            first_value = next(iter(value.values()))
            return float(first_value)
        if isinstance(value, tuple):
            return float(value[0])
        if torch.is_tensor(value):
            return float(value.detach().reshape(-1)[0].item())
        return float(value)

    def _merge_pool(self, existing_pool: Tensor | None, new_profiles: Tensor) -> Tensor:
        if existing_pool is None:
            merged = new_profiles
        else:
            merged = torch.cat([existing_pool, new_profiles], dim=0)
        if merged.shape[0] > self.spectral_sample_size:
            keep_index = torch.randperm(merged.shape[0])[: self.spectral_sample_size]
            merged = merged[keep_index]
        return merged

    def _update_pyiqa_metrics(self, img_rgb: Tensor, label_rgb: Tensor | None) -> None:
        with torch.no_grad():
            if "ssim" in self.cal_metrics:
                if label_rgb is None:
                    warnings.warn("SSIM requires label tensor, but label is None.", stacklevel=2)
                else:
                    metric = self._get_pyiqa_metric("ssim", img_rgb.device)
                    if metric is not None:
                        self._accumulate_scalar("ssim", self._extract_scalar(metric(img_rgb, label_rgb)))

            for metric_name in ["niqe", "brisque", "clipiqa"]:
                if metric_name not in self.cal_metrics:
                    continue
                metric = self._get_pyiqa_metric(metric_name, img_rgb.device)
                if metric is not None:
                    self._accumulate_scalar(metric_name, self._extract_scalar(metric(img_rgb)))

    def _update_tensor_is_fid(self, img: Tensor, label: Tensor | None) -> None:
        if "is" in self.cal_metrics:
            metric = self._get_torch_metric("is", img.device)
            if metric is not None:
                metric.update(self._prepare_rgb_uint8(img))
        if "fid" in self.cal_metrics:
            if label is None:
                warnings.warn("FID requires label tensor when tensor input is used.", stacklevel=2)
            else:
                metric = self._get_torch_metric("fid", img.device)
                if metric is not None:
                    metric.update(self._prepare_rgb_uint8(label), real=True)
                    metric.update(self._prepare_rgb_uint8(img), real=False)

    def _update_path_is_fid(self, img_path: str, label_path: str | None) -> None:
        device = torch.device("cpu")
        with torch.no_grad():
            if "is" in self.cal_metrics:
                metric = self._get_pyiqa_metric("is", device)
                if metric is not None:
                    score = metric(img_path)
                    self._accumulate_scalar("is", self._extract_scalar(score))
            if "fid" in self.cal_metrics:
                if label_path is None:
                    warnings.warn("FID requires label path when path input is used.", stacklevel=2)
                else:
                    metric = self._get_pyiqa_metric("fid", device)
                    if metric is not None:
                        score = metric(img_path, label_path)
                        self._accumulate_scalar("fid", self._extract_scalar(score))

    def _update_spectral_pools(self, img: Tensor, label: Tensor) -> None:
        fake_profiles = _sample_profiles(_flatten_profiles(img), self.spectral_sample_size)
        real_profiles = _sample_profiles(_flatten_profiles(label), self.spectral_sample_size)
        self._fake_profiles = self._merge_pool(self._fake_profiles, fake_profiles)
        self._real_profiles = self._merge_pool(self._real_profiles, real_profiles)

    def _maybe_interp_to_nbands(self, x: Tensor) -> Tensor:
        if self.interp_bands is None:
            return x
        if x.shape[1] <= self.interp_bands:
            return x
        if self.wavelength is None or self.wavelength_n is None:
            warnings.warn(
                "Skip spectral bands `>N -> N` interpolation because `wavelength` or `wavelength_n` is not provided.",
                stacklevel=2,
            )
            return x
        if x.shape[1] != int(self.wavelength.numel()):
            warnings.warn(
                (
                    "Skip spectral bands `>N -> N` interpolation because channel count and `wavelength` length mismatch: "
                    f"channels={x.shape[1]}, wavelength={int(self.wavelength.numel())}."
                ),
                stacklevel=2,
            )
            return x
        return interp_to_nbands_if_needed(
            x,
            wavelength=self.wavelength,
            wavelength_n=self.wavelength_n,
            interp_bands=self.interp_bands,
        )

    def update(self, img: Tensor | str, label: Tensor | str | None = None) -> None:
        if isinstance(img, str):
            label_path = label if isinstance(label, str) else None
            self._update_path_is_fid(img_path=img, label_path=label_path)
            return

        if not torch.is_tensor(img):
            raise TypeError(f"`img` must be Tensor or str, got {type(img)}")
        img_tensor = img
        label_tensor = label if torch.is_tensor(label) else None
        img_tensor_n = self._maybe_interp_to_nbands(img_tensor)
        label_tensor_n = self._maybe_interp_to_nbands(label_tensor) if label_tensor is not None else None

        img_rgb = self._prepare_rgb(img_tensor)
        label_rgb = self._prepare_rgb(label_tensor) if label_tensor is not None else None
        self._update_pyiqa_metrics(img_rgb=img_rgb, label_rgb=label_rgb)
        self._update_tensor_is_fid(img=img_tensor, label=label_tensor)

        if ("spr" in self.cal_metrics or "srec" in self.cal_metrics) and label_tensor_n is not None:
            self._update_spectral_pools(img=img_tensor_n, label=label_tensor_n)
        elif "spr" in self.cal_metrics or "srec" in self.cal_metrics:
            warnings.warn("sPr/sRec require label tensor, but label is None.", stacklevel=2)

    def compute(self) -> dict[str, Tensor]:
        results: dict[str, Tensor] = {}
        for metric_name in self.cal_metrics:
            if metric_name in {"spr", "srec"}:
                continue
            results[DISPLAY_NAMES[metric_name]] = self._mean_from_state(metric_name)

        if "fid" in self.cal_metrics and "fid" in self._torch_metrics:
            try:
                fid_value = self._torch_metrics["fid"].compute()
                results["FID"] = torch.as_tensor(float(fid_value), dtype=torch.float32)
            except Exception:
                pass

        if "is" in self.cal_metrics and "is" in self._torch_metrics:
            try:
                is_value = self._torch_metrics["is"].compute()
                if isinstance(is_value, tuple):
                    is_value = is_value[0]
                results["IS"] = torch.as_tensor(float(is_value), dtype=torch.float32)
            except Exception:
                pass

        if "spr" in self.cal_metrics or "srec" in self.cal_metrics:
            if self._real_profiles is None or self._fake_profiles is None:
                spr = float("nan")
                srec = float("nan")
            else:
                spr, srec = _spectral_pr_rec(
                    real_profiles=self._real_profiles,
                    fake_profiles=self._fake_profiles,
                    k=self.spectral_k,
                    n_groups=self.spectral_groups,
                    chunk_size=self.spectral_chunk_size,
                )
            if "spr" in self.cal_metrics:
                results["sPr"] = torch.tensor(spr, dtype=torch.float32)
            if "srec" in self.cal_metrics:
                results["sRec"] = torch.tensor(srec, dtype=torch.float32)

        return results

    def reset(self) -> None:
        super().reset()
        self._real_profiles = None
        self._fake_profiles = None
        for metric in self._torch_metrics.values():
            metric.reset()

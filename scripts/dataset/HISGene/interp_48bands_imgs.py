from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
import tifffile
import torch
from tqdm import tqdm
from tqdm import tqdm

from src.data.litdata_hyperloader import ImageStreamingDataset


def _as_1d_wavelength(wavelength: np.ndarray, *, name: str, require_increasing: bool = True) -> np.ndarray:
    wave = np.asarray(wavelength, dtype=np.float64)
    if wave.ndim != 1:
        raise ValueError(f"`{name}` must be 1D, got shape={wave.shape}")
    if wave.size < 2:
        raise ValueError(f"`{name}` must contain at least 2 values, got {wave.size}")
    if require_increasing and not np.all(wave[1:] > wave[:-1]):
        raise ValueError(f"`{name}` must be strictly increasing.")
    return wave


def sample_uniform_wavelength_from_lib(
    wave_lib: np.ndarray,
    left_wavelength: float,
    right_wavelength: float,
    bands: int,
) -> np.ndarray:
    lib = _as_1d_wavelength(wave_lib, name="wave_lib", require_increasing=True)
    if bands <= 0:
        raise ValueError(f"`bands` must be positive, got {bands}")
    if left_wavelength >= right_wavelength:
        raise ValueError(
            f"`left_wavelength` must be smaller than `right_wavelength`, got {left_wavelength} >= {right_wavelength}"
        )

    lib_min = float(lib[0])
    lib_max = float(lib[-1])
    if left_wavelength < lib_min or right_wavelength > lib_max:
        raise ValueError(
            f"Sampling range [{left_wavelength}, {right_wavelength}] must be inside wave_lib range [{lib_min}, {lib_max}]"
        )

    return np.linspace(left_wavelength, right_wavelength, bands, dtype=np.float64)


def transform_matrix_func(wave_current: np.ndarray, wave_lib: np.ndarray) -> np.ndarray:
    wave_lib_1d = _as_1d_wavelength(wave_lib, name="wave_lib", require_increasing=True)
    wave_current_arr = np.asarray(wave_current, dtype=np.float64)
    if wave_current_arr.ndim == 1:
        wave_current_arr = wave_current_arr[None, :]
    if wave_current_arr.ndim != 2:
        raise ValueError(f"`wave_current` must be 1D or 2D, got shape={wave_current_arr.shape}")

    query_min = float(np.min(wave_current_arr))
    query_max = float(np.max(wave_current_arr))
    base_min = float(np.min(wave_lib_1d))
    base_max = float(np.max(wave_lib_1d))
    if query_min < base_min or query_max > base_max:
        raise ValueError(f"`wave_current` out of range: [{query_min}, {query_max}] not in [{base_min}, {base_max}]")

    right_idx = np.searchsorted(wave_lib_1d, wave_current_arr, side="left")
    right_idx = np.clip(right_idx, 0, wave_lib_1d.size - 1)
    left_idx = np.clip(right_idx - 1, 0, wave_lib_1d.size - 1)

    left_wave = wave_lib_1d[left_idx]
    right_wave = wave_lib_1d[right_idx]
    denom = right_wave - left_wave
    right_weight = np.divide(
        wave_current_arr - left_wave,
        denom,
        out=np.zeros_like(wave_current_arr, dtype=np.float64),
        where=denom > 0,
    )
    left_weight = 1.0 - right_weight

    same_idx = left_idx == right_idx
    left_weight = np.where(same_idx, 1.0, left_weight)
    right_weight = np.where(same_idx, 0.0, right_weight)

    set_num, target_band_num = wave_current_arr.shape
    source_band_num = wave_lib_1d.size
    transform_matrix = np.zeros((set_num, target_band_num, source_band_num), dtype=np.float32)
    band_idx = np.arange(target_band_num, dtype=np.int64)

    for set_id in range(set_num):
        transform_matrix[set_id, band_idx, left_idx[set_id]] += left_weight[set_id].astype(np.float32)
        transform_matrix[set_id, band_idx, right_idx[set_id]] += right_weight[set_id].astype(np.float32)
    return transform_matrix


def interp_wavelength_hwc(
    img: np.ndarray,
    wavelength: np.ndarray,
    wavelength_interp: np.ndarray,
) -> np.ndarray:
    img_arr = np.asarray(img)
    if img_arr.ndim != 3:
        raise ValueError(f"`img` must be HWC, got shape={img_arr.shape}")

    wave_src = _as_1d_wavelength(wavelength, name="wavelength", require_increasing=True)
    wave_dst = _as_1d_wavelength(wavelength_interp, name="wavelength_interp", require_increasing=False)

    if img_arr.shape[2] != wave_src.size:
        raise ValueError(
            f"Channel count mismatch: img has {img_arr.shape[2]} channels, wavelength has {wave_src.size} values."
        )

    transform_matrix = transform_matrix_func(wave_dst, wave_src)[0]
    img_chw = np.transpose(img_arr, (2, 0, 1))
    img_interp = np.einsum("sn,nwh->swh", transform_matrix, img_chw, optimize=True)
    out = np.transpose(img_interp, (1, 2, 0))
    if np.issubdtype(img_arr.dtype, np.floating):
        return out.astype(img_arr.dtype, copy=False)
    return out.astype(np.float32, copy=False)


def interp_to_nbands_if_needed_hwc(
    img: np.ndarray,
    wavelength: np.ndarray | None,
    wavelength_n: np.ndarray | None,
    interp_bands: int | None,
) -> np.ndarray:
    img_arr = np.asarray(img)
    if img_arr.ndim != 3:
        raise ValueError(f"`img` must be HWC, got shape={img_arr.shape}")
    if interp_bands is None:
        return img_arr
    if interp_bands <= 0:
        raise ValueError(f"`interp_bands` must be positive when provided, got {interp_bands}")
    if img_arr.shape[2] <= interp_bands:
        return img_arr
    if wavelength is None or wavelength_n is None:
        raise ValueError("`wavelength` and `wavelength_n` are required when `interp_bands` is set.")

    wave_dst = _as_1d_wavelength(wavelength_n, name="wavelength_n", require_increasing=True)
    if wave_dst.size != interp_bands:
        raise ValueError(f"`wavelength_n` length must be {interp_bands}, got {wave_dst.size}")
    return interp_wavelength_hwc(img_arr, wavelength=np.asarray(wavelength), wavelength_interp=wave_dst)


def interp_to_48bands_if_needed_hwc(
    img: np.ndarray,
    wavelength: np.ndarray | None,
    wavelength_48: np.ndarray | None,
) -> np.ndarray:
    return interp_to_nbands_if_needed_hwc(
        img=img,
        wavelength=wavelength,
        wavelength_n=wavelength_48,
        interp_bands=48,
    )


def _as_1d_wavelength_tensor(
    wavelength: torch.Tensor | np.ndarray | list[float],
    *,
    name: str,
    require_increasing: bool = True,
) -> torch.Tensor:
    wave = torch.as_tensor(wavelength, dtype=torch.float32)
    if wave.ndim != 1:
        raise ValueError(f"`{name}` must be 1D, got shape={tuple(wave.shape)}")
    if wave.numel() < 2:
        raise ValueError(f"`{name}` must contain at least 2 values, got {wave.numel()}")
    if require_increasing and not bool(torch.all(wave[1:] > wave[:-1])):
        raise ValueError(f"`{name}` must be strictly increasing.")
    return wave


def sample_uniform_wavelength_from_lib_tensor(
    wave_lib: torch.Tensor | np.ndarray | list[float],
    left_wavelength: float,
    right_wavelength: float,
    bands: int,
) -> torch.Tensor:
    lib = _as_1d_wavelength_tensor(wave_lib, name="wave_lib", require_increasing=True)
    if bands <= 0:
        raise ValueError(f"`bands` must be positive, got {bands}")
    if left_wavelength >= right_wavelength:
        raise ValueError(
            f"`left_wavelength` must be smaller than `right_wavelength`, got {left_wavelength} >= {right_wavelength}"
        )
    lib_min = float(lib[0].item())
    lib_max = float(lib[-1].item())
    if left_wavelength < lib_min or right_wavelength > lib_max:
        raise ValueError(
            f"Sampling range [{left_wavelength}, {right_wavelength}] must be inside wave_lib range [{lib_min}, {lib_max}]"
        )
    return torch.linspace(left_wavelength, right_wavelength, bands, dtype=torch.float32)


def transform_matrix_func_tensor(wave_current: torch.Tensor, wave_lib: torch.Tensor) -> torch.Tensor:
    wave_lib_1d = _as_1d_wavelength_tensor(wave_lib, name="wave_lib", require_increasing=True)
    wave_current_arr = torch.as_tensor(wave_current, dtype=torch.float32)
    if wave_current_arr.ndim == 1:
        wave_current_arr = wave_current_arr.unsqueeze(0)
    if wave_current_arr.ndim != 2:
        raise ValueError(f"`wave_current` must be 1D or 2D, got shape={tuple(wave_current_arr.shape)}")

    query_min = float(wave_current_arr.min().item())
    query_max = float(wave_current_arr.max().item())
    base_min = float(wave_lib_1d.min().item())
    base_max = float(wave_lib_1d.max().item())
    if query_min < base_min or query_max > base_max:
        raise ValueError(f"`wave_current` out of range: [{query_min}, {query_max}] not in [{base_min}, {base_max}]")

    right_idx = torch.searchsorted(wave_lib_1d, wave_current_arr, right=False)
    right_idx = right_idx.clamp(max=wave_lib_1d.numel() - 1)
    left_idx = (right_idx - 1).clamp(min=0)

    left_wave = wave_lib_1d[left_idx]
    right_wave = wave_lib_1d[right_idx]
    denom = right_wave - left_wave
    right_weight = torch.where(denom > 0, (wave_current_arr - left_wave) / denom, torch.zeros_like(wave_current_arr))
    left_weight = 1.0 - right_weight

    same_idx = left_idx == right_idx
    left_weight = torch.where(same_idx, torch.ones_like(left_weight), left_weight)
    right_weight = torch.where(same_idx, torch.zeros_like(right_weight), right_weight)

    set_num, target_band_num = wave_current_arr.shape
    source_band_num = int(wave_lib_1d.numel())
    transform_matrix = torch.zeros((set_num, target_band_num, source_band_num), dtype=torch.float32)
    set_idx = torch.arange(set_num).unsqueeze(1).expand(set_num, target_band_num).reshape(-1)
    band_idx = torch.arange(target_band_num).unsqueeze(0).expand(set_num, target_band_num).reshape(-1)

    transform_matrix[set_idx, band_idx, left_idx.reshape(-1)] += left_weight.reshape(-1)
    transform_matrix[set_idx, band_idx, right_idx.reshape(-1)] += right_weight.reshape(-1)
    return transform_matrix


def interp_wavelength_chw_tensor(
    img: torch.Tensor,
    wavelength: torch.Tensor | np.ndarray | list[float],
    wavelength_interp: torch.Tensor | np.ndarray | list[float],
) -> torch.Tensor:
    if img.ndim != 3:
        raise ValueError(f"`img` must be CHW tensor, got shape={tuple(img.shape)}")
    wave_src = _as_1d_wavelength_tensor(wavelength, name="wavelength", require_increasing=True)
    wave_dst = _as_1d_wavelength_tensor(wavelength_interp, name="wavelength_interp", require_increasing=False)
    if img.shape[0] != int(wave_src.numel()):
        raise ValueError(f"Channel count mismatch: img has {img.shape[0]} channels, wavelength has {wave_src.numel()}")

    transform_matrix = transform_matrix_func_tensor(wave_dst, wave_src)[0].to(device=img.device, dtype=img.dtype)
    return torch.einsum("sn,nwh->swh", transform_matrix, img)


def _to_chw_from_sample(
    img: torch.Tensor,
    wavelength: torch.Tensor | np.ndarray | list[float],
    input_layout: Literal["auto", "chw", "hwc"] = "auto",
) -> torch.Tensor:
    wave_src = _as_1d_wavelength_tensor(wavelength, name="wavelength", require_increasing=True)
    channel_num = int(wave_src.numel())
    if img.ndim != 3:
        raise ValueError(f"`img` from dataset must be 3D tensor, got shape={tuple(img.shape)}")
    if input_layout == "chw":
        if img.shape[0] != channel_num:
            raise ValueError(
                f"`input_layout='chw'` requires first dim == channels. Got shape={tuple(img.shape)}, channels={channel_num}"
            )
        return img
    if input_layout == "hwc":
        if img.shape[-1] != channel_num:
            raise ValueError(
                f"`input_layout='hwc'` requires last dim == channels. Got shape={tuple(img.shape)}, channels={channel_num}"
            )
        return img.permute(2, 0, 1)
    if img.shape[0] == channel_num and img.shape[-1] != channel_num:
        return img
    if img.shape[-1] == channel_num and img.shape[0] != channel_num:
        return img.permute(2, 0, 1)
    if img.shape[0] == channel_num and img.shape[-1] == channel_num:
        raise ValueError(
            f"Ambiguous layout for shape={tuple(img.shape)}. Please set explicit `input_layout='chw'` or `'hwc'`."
        )
    raise ValueError(f"Cannot infer CHW/HWC layout for image shape={tuple(img.shape)} and channels={channel_num}")


def _resolve_output_tif_name(sample: dict[str, Any], sample_idx: int, output_name: str | None) -> str:
    if output_name is not None:
        base = output_name
    else:
        key = str(sample.get("__key__", f"sample_{sample_idx:06d}"))
        base = re.sub(r"[^\w\-.]+", "_", key).strip("_")
        if not base:
            base = f"sample_{sample_idx:06d}"
    if not base.lower().endswith((".tif", ".tiff")):
        base = f"{base}.tif"
    return base


def save_litdata_img_interp_tif(
    input_dir: str,
    output_dir: str | Path,
    wavelength: torch.Tensor | np.ndarray | list[float],
    wavelength_interp: torch.Tensor | np.ndarray | list[float] | None = None,
    *,
    left_wavelength: float | None = None,
    right_wavelength: float | None = None,
    bands: int = 48,
    dataset_kwargs: dict[str, Any] | None = None,
    output_name: str | None = None,
    compression: str = "jpeg2000",
    start_idx: int = 0,
    end_idx: int | None = None,
) -> list[Path]:
    ds_kwargs: dict[str, Any] = {
        "to_neg_1_1": False,
        "hyper_transforms_lst": None,
        "hyper_degradation_lst": None,
        "transform_prob": 0.0,
        "degradation_prob": 0.0,
        "is_hwc": True,
        "disable_norm": True,
    }
    if dataset_kwargs is not None:
        ds_kwargs.update(dataset_kwargs)

    ds = ImageStreamingDataset.create_dataset(input_dir=input_dir, is_cycled=False, **ds_kwargs)
    if wavelength_interp is None:
        if left_wavelength is None or right_wavelength is None:
            raise ValueError("Provide `wavelength_interp`, or provide both `left_wavelength` and `right_wavelength`.")
        wavelength_interp = sample_uniform_wavelength_from_lib_tensor(
            wave_lib=wavelength,
            left_wavelength=left_wavelength,
            right_wavelength=right_wavelength,
            bands=bands,
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_len = len(ds)
    if start_idx < 0 or start_idx >= ds_len:
        raise ValueError(f"`start_idx` out of range: {start_idx}, dataset length={ds_len}")
    stop_idx = ds_len if end_idx is None else min(end_idx, ds_len)
    if stop_idx <= start_idx:
        raise ValueError(f"`end_idx` must be greater than `start_idx`, got start_idx={start_idx}, end_idx={end_idx}")

    saved_paths: list[Path] = []
    for sample_idx in tqdm(range(start_idx, stop_idx)):
        sample = ds[sample_idx]
        if sample is None:
            continue
        if not isinstance(sample, dict):
            raise TypeError(f"Expected dict-like sample from dataset, got {type(sample)}")
        if "img" not in sample:
            raise KeyError(f"`img` key not found in sample keys={list(sample.keys())}")

        img_raw = sample["img"]
        if not torch.is_tensor(img_raw):
            img_raw = torch.as_tensor(img_raw)
        img_chw = _to_chw_from_sample(img_raw, wavelength=wavelength, input_layout="chw").float()
        img_interp_chw = interp_wavelength_chw_tensor(
            img=img_chw,
            wavelength=wavelength,
            wavelength_interp=wavelength_interp,
        )
        img_interp_hwc = img_interp_chw.permute(1, 2, 0).cpu().numpy().astype(np.uint16, copy=False)

        if output_name is None:
            file_name = _resolve_output_tif_name(sample=sample, sample_idx=sample_idx, output_name=None)
        else:
            stem = Path(output_name).stem
            suffix = Path(output_name).suffix if Path(output_name).suffix else ".tif"
            file_name = f"{stem}_{sample_idx:06d}{suffix}"
        out_path = out_dir / file_name
        tifffile.imwrite(out_path, img_interp_hwc, compression=compression, compressionargs={"reversible": True})
        saved_paths.append(out_path)
    return saved_paths


def interp_tif_dir_to_nbands(
    input_dir: str | Path,
    output_dir: str | Path,
    wavelength: torch.Tensor | np.ndarray | list[float],
    wavelength_interp: torch.Tensor | np.ndarray | list[float] | None = None,
    *,
    left_wavelength: float | None = None,
    right_wavelength: float | None = None,
    bands: int = 48,
    compression: str = "jpeg2000",
    recursive: bool = True,
) -> list[Path]:
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    if not in_dir.exists() or not in_dir.is_dir():
        raise ValueError(f"`input_dir` must be an existing directory, got {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern_iter = in_dir.rglob("*") if recursive else in_dir.glob("*")
    tif_paths = sorted(p for p in pattern_iter if p.is_file() and p.suffix.lower() in {".tif", ".tiff"})
    if not tif_paths:
        return []

    if wavelength_interp is None:
        if left_wavelength is None or right_wavelength is None:
            raise ValueError("Provide `wavelength_interp`, or provide both `left_wavelength` and `right_wavelength`.")
        wavelength_interp = sample_uniform_wavelength_from_lib_tensor(
            wave_lib=wavelength,
            left_wavelength=left_wavelength,
            right_wavelength=right_wavelength,
            bands=bands,
        )

    saved_paths: list[Path] = []
    for tif_path in tqdm(tif_paths):
        img_np = tifffile.imread(tif_path)
        img_tensor = torch.as_tensor(img_np)
        if img_tensor.ndim == 2:
            img_tensor = img_tensor.unsqueeze(-1)
        img_chw = _to_chw_from_sample(img_tensor, wavelength=wavelength, input_layout="hwc").float()
        img_interp_chw = interp_wavelength_chw_tensor(
            img=img_chw,
            wavelength=wavelength,
            wavelength_interp=wavelength_interp,
        )
        img_interp_hwc = img_interp_chw.permute(1, 2, 0).cpu().numpy().astype(np.uint16, copy=False)

        rel = tif_path.relative_to(in_dir)
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(out_path, img_interp_hwc, compression=compression, compressionargs={"reversible": True})
        saved_paths.append(out_path)

    return saved_paths


if __name__ == "__main__":
    ...
    # path = "data2/RemoteSAM270k/LitData_hyper_images"
    # out = save_litdata_img_interp_tif(
    #     input_dir=path,
    #     output_dir="tmp/interp_tifs",
    #     sample_idx=0,
    #     wavelength=np.linspace(380, 2500, 224),
    #     left_wavelength=400.0,
    #     right_wavelength=1000.0,
    #     bands=48,
    # )

    # Xiongan
    # xiongan_wave_lens = [
    #     391.298300,
    #     393.696700,
    #     396.095000,
    #     398.493300,
    #     400.891700,
    #     403.290000,
    #     405.688300,
    #     408.086700,
    #     410.485000,
    #     412.883300,
    #     415.281700,
    #     417.680000,
    #     420.078300,
    #     422.476700,
    #     424.875000,
    #     427.273300,
    #     429.671700,
    #     432.070000,
    #     434.468300,
    #     436.866700,
    #     439.265000,
    #     441.663300,
    #     444.061700,
    #     446.460000,
    #     448.858300,
    #     451.256700,
    #     453.655000,
    #     456.053300,
    #     458.451700,
    #     460.850000,
    #     463.248300,
    #     465.646700,
    #     468.045000,
    #     470.443300,
    #     472.841700,
    #     475.240000,
    #     477.638300,
    #     480.036700,
    #     482.435000,
    #     484.833300,
    #     487.231700,
    #     489.630000,
    #     492.028300,
    #     494.426700,
    #     496.825000,
    #     499.223300,
    #     501.621700,
    #     504.020000,
    #     506.418300,
    #     508.816700,
    #     511.215000,
    #     513.613300,
    #     516.011700,
    #     518.410000,
    #     520.808300,
    #     523.206700,
    #     525.605000,
    #     528.003300,
    #     530.401700,
    #     532.800000,
    #     535.198300,
    #     537.596700,
    #     539.995000,
    #     542.393300,
    #     544.791700,
    #     547.190000,
    #     549.588300,
    #     551.986700,
    #     554.385000,
    #     556.783300,
    #     559.181700,
    #     561.580000,
    #     563.978300,
    #     566.376700,
    #     568.775000,
    #     571.173300,
    #     573.571700,
    #     575.970000,
    #     578.368300,
    #     580.766700,
    #     583.165000,
    #     585.563300,
    #     587.961700,
    #     590.360000,
    #     592.758300,
    #     595.156700,
    #     597.555000,
    #     599.953300,
    #     602.351700,
    #     604.750000,
    #     607.148300,
    #     609.546700,
    #     611.945000,
    #     614.343300,
    #     616.741700,
    #     619.140000,
    #     621.538300,
    #     623.936700,
    #     626.335000,
    #     628.733300,
    #     631.131700,
    #     633.530000,
    #     635.928300,
    #     638.326700,
    #     640.725000,
    #     643.123300,
    #     645.521700,
    #     647.920000,
    #     650.318300,
    #     652.716700,
    #     655.115000,
    #     657.513300,
    #     659.911700,
    #     662.310000,
    #     664.708300,
    #     667.106700,
    #     669.505000,
    #     671.903300,
    #     674.301700,
    #     676.700000,
    #     679.098300,
    #     681.496700,
    #     683.895000,
    #     686.293300,
    #     688.691700,
    #     691.090000,
    #     693.488300,
    #     695.886700,
    #     698.285000,
    #     700.683300,
    #     703.081700,
    #     705.480000,
    #     707.878300,
    #     710.276700,
    #     712.675000,
    #     715.073300,
    #     717.471700,
    #     719.870000,
    #     722.268300,
    #     724.666700,
    #     727.065000,
    #     729.463300,
    #     731.861700,
    #     734.260000,
    #     736.658300,
    #     739.056700,
    #     741.455000,
    #     743.853300,
    #     746.251700,
    #     748.650000,
    #     751.048300,
    #     753.446700,
    #     755.845000,
    #     758.243300,
    #     760.641700,
    #     763.040000,
    #     765.438300,
    #     767.836700,
    #     770.235000,
    #     772.633300,
    #     775.031700,
    #     777.430000,
    #     779.828300,
    #     782.226700,
    #     784.625000,
    #     787.023300,
    #     789.421700,
    #     791.820000,
    #     794.218300,
    #     796.616700,
    #     799.015000,
    #     801.413300,
    #     803.811700,
    #     806.210000,
    #     808.608300,
    #     811.006700,
    #     813.405000,
    #     815.803300,
    #     818.201700,
    #     820.600000,
    #     822.998300,
    #     825.396700,
    #     827.795000,
    #     830.193300,
    #     832.591700,
    #     834.990000,
    #     837.388300,
    #     839.786700,
    #     842.185000,
    #     844.583300,
    #     846.981700,
    #     849.380000,
    #     851.778300,
    #     854.176700,
    #     856.575000,
    #     858.973300,
    #     861.371700,
    #     863.770000,
    #     866.168300,
    #     868.566700,
    #     870.965000,
    #     873.363300,
    #     875.761700,
    #     878.160000,
    #     880.558300,
    #     882.956700,
    #     885.355000,
    #     887.753300,
    #     890.151700,
    #     892.550000,
    #     894.948300,
    #     897.346700,
    #     899.745000,
    #     902.143300,
    #     904.541700,
    #     906.940000,
    #     909.338300,
    #     911.736700,
    #     914.135000,
    #     916.533300,
    #     918.931700,
    #     921.330000,
    #     923.728300,
    #     926.126700,
    #     928.525000,
    #     930.923300,
    #     933.321700,
    #     935.720000,
    #     938.118300,
    #     940.516700,
    #     942.915000,
    #     945.313300,
    #     947.711700,
    #     950.110000,
    #     952.508300,
    #     954.906700,
    #     957.305000,
    #     959.703300,
    #     962.101700,
    #     964.500000,
    #     966.898300,
    #     969.296700,
    #     971.695000,
    #     974.093300,
    #     976.491700,
    #     978.890000,
    #     981.288300,
    #     983.686700,
    #     986.085000,
    #     988.483300,
    #     990.881700,
    #     993.280000,
    #     995.678300,
    #     998.076700,
    #     1000.475000,
    #     1002.873000,
    # ]
    # out = interp_tif_dir_to_nbands(
    #     input_dir="data/Downstreams/ClassificationCollection/hyper_images/xiongan",
    #     output_dir="data2/HSIGene_dataset/xiongan",
    #     wavelength=xiongan_wave_lens,
    #     left_wavelength=400.0,
    #     right_wavelength=1000.0,
    #     bands=48,
    # )

    # chikusei
    # path = '/Data/ZiHanCao/datasets/HSIs/Chikusei/tmp'
    # out = interp_tif_dir_to_nbands(
    #     input_dir=path,
    #     output_dir='data2/HSIGene_dataset/chikusei',
    #     wavelength=np.linspace(343,1018,128),
    #     left_wavelength=400.0,
    #     right_wavelength=1000.0,
    #     bands=48,
    #     compression='jpeg2000'
    # )

    # Houston DFC2013
    # path = "data/Houston/LitData_hyper_images"
    # save_litdata_img_interp_tif(
    #     path,
    #     output_dir="data2/HSIGene_dataset/houston",
    #     wavelength=np.linspace(380, 1050, 50),
    #     left_wavelength=400.0,
    #     right_wavelength=1000.0,
    #     bands=48,
    # )

    # MDAS-HySpex
    # path = "data/MDAS-HySpex/hyper_images/tmp"
    # interp_tif_dir_to_nbands(
    #     path,
    #     output_dir="data2/HSIGene_dataset/mdas_hyspex",
    #     wavelength=np.linspace(400, 1000, 368),
    #     left_wavelength=400.0,
    #     right_wavelength=1000.0,
    #     bands=48,
    #     compression='jpeg2000'
    # )

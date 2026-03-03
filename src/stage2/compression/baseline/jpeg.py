from pathlib import Path
from typing import Literal

import numpy as np
import tifffile
from loguru import logger

CompressionBackend = Literal["tiff_jpeg2000", "gdal_jp2"]


class HyperspectralJPEG2000Compressor:
    def __init__(self, quality: float = 50.0, compression_ratio: float | None = None, reversible: bool = False):
        self.quality = quality
        self.compression_ratio = compression_ratio
        self.reversible = reversible

    def compress_hyperspectral(
        self,
        hyperspectral_data: np.ndarray,
        output_dir: str | Path,
        prefix: str = "band",
        normalize: bool = False,
        input_layout: str = "HWC",
        backend: CompressionBackend = "tiff_jpeg2000",
        geotransform: tuple[float, float, float, float, float, float] | None = None,
        projection: str | None = None,
        creation_options: list[str] | None = None,
    ) -> tuple[list[str], dict[str, float | int]]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        processed = self._prepare_array(hyperspectral_data, normalize=normalize, input_layout=input_layout)
        height, width, num_bands = processed.shape
        logger.info(f"Processing hyperspectral data: {height}x{width}x{num_bands}, dtype={processed.dtype}")

        if backend == "tiff_jpeg2000":
            output_file = output_dir / f"{prefix}.tif"
            self._write_tiff_jpeg2000(processed, output_file)
        elif backend == "gdal_jp2":
            output_file = output_dir / f"{prefix}.jp2"
            self._write_jp2_with_gdal(
                data_hwc=processed,
                output_file=output_file,
                geotransform=geotransform,
                projection=projection,
                creation_options=creation_options,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        original_size = processed.nbytes
        compressed_size = output_file.stat().st_size
        compression_stats: dict[str, float | int] = {
            "original_size": original_size,
            "in_memory_size_bytes": original_size,
            "compressed_size": compressed_size,
            "compression_ratio": original_size / compressed_size if compressed_size > 0 else 0.0,
            "num_bands": num_bands,
        }
        logger.info(f"Saved compressed file: {output_file.name} (orig {original_size}, comp {compressed_size})")
        return [str(output_file)], compression_stats

    def decompress_hyperspectral(
        self,
        compressed_files: list[str],
        backend: CompressionBackend = "tiff_jpeg2000",
        output_shape: tuple[int, int, int] | None = None,
    ) -> np.ndarray:
        if not compressed_files:
            raise ValueError("No compressed files provided")

        file_path = compressed_files[0]
        if backend == "tiff_jpeg2000":
            data = self._read_tiff(file_path)
        elif backend == "gdal_jp2":
            data = self._read_raster_with_gdal(file_path)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        if output_shape is not None and data.shape != output_shape:
            raise AssertionError(f"Shape mismatch: expected {output_shape}, got {data.shape}")
        return data

    def compress_hyperspectral_file(
        self,
        input_file: str | Path,
        output_dir: str | Path,
        prefix: str | None = None,
        normalize: bool = False,
        input_layout: str = "HWC",
        backend: CompressionBackend = "tiff_jpeg2000",
        creation_options: list[str] | None = None,
    ) -> tuple[list[str], dict[str, float | int]]:
        input_file = Path(input_file)
        input_file_size_bytes = input_file.stat().st_size
        if backend == "tiff_jpeg2000":
            src_data = self._read_tiff(str(input_file))
            geotransform = None
            projection = None
        else:
            src_data, geotransform, projection = self._read_raster_file_with_meta(input_file)
        output_prefix = prefix or input_file.stem
        files, stats = self.compress_hyperspectral(
            hyperspectral_data=src_data,
            output_dir=output_dir,
            prefix=output_prefix,
            normalize=normalize,
            input_layout=input_layout,
            backend=backend,
            geotransform=geotransform,
            projection=projection,
            creation_options=creation_options,
        )
        stats["input_file_size_bytes"] = input_file_size_bytes
        return files, stats

    def evaluate_psnr_after_compression(
        self,
        hyperspectral_data: np.ndarray,
        output_dir: str | Path,
        prefix: str = "band",
        normalize: bool = False,
        input_layout: str = "HWC",
        backend: CompressionBackend = "tiff_jpeg2000",
        creation_options: list[str] | None = None,
    ) -> dict[str, float | int]:
        files, stats = self.compress_hyperspectral(
            hyperspectral_data=hyperspectral_data,
            output_dir=output_dir,
            prefix=prefix,
            normalize=normalize,
            input_layout=input_layout,
            backend=backend,
            creation_options=creation_options,
        )
        encoded = self.decompress_hyperspectral(files, backend=backend)
        original = self._prepare_array(hyperspectral_data, normalize=normalize, input_layout=input_layout)
        metrics = self._compute_metrics_with_torchmetrics(original, encoded)
        stats.update(metrics)
        return stats

    def evaluate_psnr_after_compression_file(
        self,
        input_file: str | Path,
        output_dir: str | Path,
        prefix: str | None = None,
        normalize: bool = False,
        input_layout: str = "HWC",
        backend: CompressionBackend = "tiff_jpeg2000",
        creation_options: list[str] | None = None,
    ) -> dict[str, float | int]:
        input_file = Path(input_file)
        if backend == "tiff_jpeg2000":
            original = self._read_tiff(str(input_file))
        else:
            original, _, _ = self._read_raster_file_with_meta(input_file)
        original_prepared = self._prepare_array(original, normalize=normalize, input_layout=input_layout)
        files, stats = self.compress_hyperspectral_file(
            input_file=input_file,
            output_dir=output_dir,
            prefix=prefix,
            normalize=normalize,
            input_layout=input_layout,
            backend=backend,
            creation_options=creation_options,
        )
        encoded = self.decompress_hyperspectral(files, backend=backend)
        stats.update(self._compute_metrics_with_torchmetrics(original_prepared, encoded))
        return stats

    def _prepare_array(self, data: np.ndarray, normalize: bool, input_layout: str) -> np.ndarray:
        data_hwc = self._to_hwc(data, input_layout=input_layout)
        if normalize:
            return self._normalize_per_band_to_uint16(data_hwc)
        if data_hwc.dtype in (np.uint8, np.uint16, np.int16):
            return data_hwc
        if np.issubdtype(data_hwc.dtype, np.floating):
            return self._normalize_per_band_to_uint16(data_hwc)
        return data_hwc.astype(np.uint16)

    def _to_hwc(self, data: np.ndarray, input_layout: str) -> np.ndarray:
        if data.ndim != 3:
            raise ValueError(f"Expected 3D array, got shape={data.shape}")
        layout = input_layout.upper()
        if layout == "HWC":
            return data
        if layout == "CHW":
            return np.transpose(data, (1, 2, 0))
        raise ValueError(f"Unsupported input_layout={input_layout}. Use 'HWC' or 'CHW'.")

    def _normalize_per_band_to_uint16(self, data_hwc: np.ndarray) -> np.ndarray:
        height, width, num_bands = data_hwc.shape
        out = np.zeros((height, width, num_bands), dtype=np.uint16)
        for band_idx in range(num_bands):
            band = data_hwc[:, :, band_idx].astype(np.float64)
            min_val = float(np.min(band))
            max_val = float(np.max(band))
            if max_val > min_val:
                scaled = (band - min_val) / (max_val - min_val)
                out[:, :, band_idx] = np.round(scaled * 65535.0).astype(np.uint16)
            else:
                out[:, :, band_idx] = np.zeros_like(band, dtype=np.uint16)
        return out

    def _write_tiff_jpeg2000(self, data_hwc: np.ndarray, output_file: Path) -> None:
        compression_args: dict[str, bool | int] = {"reversible": self.reversible}
        if self.compression_ratio is not None:
            if self.compression_ratio <= 0:
                raise ValueError("compression_ratio must be > 0")
            level = int(np.clip(round(100.0 / self.compression_ratio), 1, 100))
            compression_args["level"] = level
        elif not self.reversible:
            compression_args["level"] = int(np.clip(round(self.quality), 1, 100))
        tifffile.imwrite(
            output_file,
            data_hwc,
            compression="jpeg2000",
            compressionargs=compression_args,
        )

    def _read_tiff(self, file_path: str) -> np.ndarray:
        data = tifffile.imread(file_path)
        if data.ndim == 2:
            return data[:, :, np.newaxis]
        if data.ndim != 3:
            raise ValueError(f"Unexpected TIFF shape: {data.shape}")
        return data

    def _compute_metrics_with_torchmetrics(self, reference: np.ndarray, test: np.ndarray) -> dict[str, float]:
        if reference.shape != test.shape:
            raise ValueError(f"Shape mismatch for PSNR: {reference.shape} vs {test.shape}")
        import torch
        from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure

        ref = torch.from_numpy(reference.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        pred = torch.from_numpy(test.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        data_range = float(torch.max(ref).item() - torch.min(ref).item())
        if data_range <= 0.0:
            data_range = 1.0
        psnr = float(peak_signal_noise_ratio(pred, ref, data_range=data_range).item())
        ssim = float(structural_similarity_index_measure(pred, ref, data_range=data_range).item())
        return {"psnr": psnr, "ssim": ssim}

    def _build_creation_options(self, extra_options: list[str] | None) -> list[str]:
        options: list[str] = [f"REVERSIBLE={'YES' if self.reversible else 'NO'}"]
        if self.compression_ratio is not None:
            if self.compression_ratio <= 0:
                raise ValueError("compression_ratio must be > 0")
            options.append(f"RATE={self.compression_ratio}")
        elif not self.reversible:
            q = int(round(self.quality))
            q = min(100, max(1, q))
            options.append(f"QUALITY={q}")
        if extra_options:
            options.extend(extra_options)
        return options

    def _write_jp2_with_gdal(
        self,
        data_hwc: np.ndarray,
        output_file: Path,
        geotransform: tuple[float, float, float, float, float, float] | None,
        projection: str | None,
        creation_options: list[str] | None,
    ) -> None:
        gdal, gdal_array = _require_gdal()
        height, width, num_bands = data_hwc.shape
        gdal_dtype = gdal_array.NumericTypeCodeToGDALTypeCode(data_hwc.dtype)
        if gdal_dtype is None:
            raise TypeError(f"Unsupported dtype for GDAL JP2 writing: {data_hwc.dtype}")
        driver = gdal.GetDriverByName("JP2OpenJPEG")
        if driver is None:
            raise RuntimeError("GDAL JP2OpenJPEG driver not found.")
        dataset = driver.Create(
            str(output_file),
            width,
            height,
            num_bands,
            gdal_dtype,
            options=self._build_creation_options(creation_options),
        )
        if dataset is None:
            raise RuntimeError(f"Failed to create JP2 file: {output_file}")
        if geotransform is not None:
            dataset.SetGeoTransform(geotransform)
        if projection:
            dataset.SetProjection(projection)
        for band_idx in range(num_bands):
            dataset.GetRasterBand(band_idx + 1).WriteArray(data_hwc[:, :, band_idx])
        dataset.FlushCache()
        dataset = None

    def _read_raster_file_with_meta(
        self, input_file: Path
    ) -> tuple[np.ndarray, tuple[float, float, float, float, float, float] | None, str | None]:
        gdal, _ = _require_gdal()
        dataset = gdal.Open(str(input_file), gdal.GA_ReadOnly)
        if dataset is None:
            raise RuntimeError(f"Failed to open raster file: {input_file}")
        raw = dataset.ReadAsArray()
        if raw is None:
            raise RuntimeError(f"Failed to read raster data from: {input_file}")
        if raw.ndim == 2:
            data_hwc = raw[:, :, np.newaxis]
        elif raw.ndim == 3:
            data_hwc = np.transpose(raw, (1, 2, 0))
        else:
            raise ValueError(f"Unexpected raster data shape: {raw.shape}")
        geotransform = dataset.GetGeoTransform(can_return_null=True)
        projection = dataset.GetProjectionRef() or None
        dataset = None
        return data_hwc, geotransform, projection

    def _read_raster_with_gdal(self, file_path: str) -> np.ndarray:
        gdal, _ = _require_gdal()
        dataset = gdal.Open(file_path, gdal.GA_ReadOnly)
        if dataset is None:
            raise RuntimeError(f"Failed to open compressed file: {file_path}")
        raw = dataset.ReadAsArray()
        if raw is None:
            raise RuntimeError(f"Failed to read compressed file: {file_path}")
        data = raw[:, :, np.newaxis] if raw.ndim == 2 else np.transpose(raw, (1, 2, 0))
        dataset = None
        return data


def compress_hyperspectral_file(
    input_file: str | Path,
    output_dir: str | Path,
    quality: float = 50.0,
    compression_ratio: float | None = None,
    normalize: bool = False,
    input_layout: str = "HWC",
    reversible: bool = False,
    backend: CompressionBackend = "tiff_jpeg2000",
    creation_options: list[str] | None = None,
) -> dict[str, float | int]:
    compressor = HyperspectralJPEG2000Compressor(
        quality=quality,
        compression_ratio=compression_ratio,
        reversible=reversible,
    )
    _, stats = compressor.compress_hyperspectral_file(
        input_file=input_file,
        output_dir=output_dir,
        normalize=normalize,
        input_layout=input_layout,
        backend=backend,
        creation_options=creation_options,
    )
    return stats


def benchmark_single_tif_to_tmp(
    input_file: str | Path,
    quality: float = 95.0,
    reversible: bool = False,
    backend: CompressionBackend = "tiff_jpeg2000",
    output_dir: str | Path = "tmp/jpeg2000_benchmark",
    input_layout: str = "HWC",
) -> dict[str, float | int | str]:
    input_path = Path(input_file)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    compressor = HyperspectralJPEG2000Compressor(
        quality=quality,
        compression_ratio=None,
        reversible=reversible,
    )
    stats = compressor.evaluate_psnr_after_compression_file(
        input_file=input_path,
        output_dir=out_dir,
        prefix=f"{input_path.stem}_q{int(round(quality))}",
        normalize=False,
        input_layout=input_layout,
        backend=backend,
    )
    output_suffix = ".tif" if backend == "tiff_jpeg2000" else ".jp2"
    output_file = out_dir / f"{input_path.stem}_q{int(round(quality))}{output_suffix}"
    stats_with_path: dict[str, float | int | str] = {
        "input_file": str(input_path),
        "output_file": str(output_file),
        "output_file_size_bytes": output_file.stat().st_size,
        **stats,
    }
    return stats_with_path


def _require_gdal():
    try:
        from osgeo import gdal, gdal_array
    except ImportError as exc:
        raise ImportError("GDAL Python bindings are required for backend='gdal_jp2'.") from exc
    gdal.UseExceptions()
    return gdal, gdal_array

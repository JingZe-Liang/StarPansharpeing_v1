from __future__ import annotations

import argparse
import importlib
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import tifffile
import torch
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure

QuantizerMode = Literal["lossless", "absolute", "relative", "absrel"]
InputLayout = Literal["HWC", "CHW"]


def _load_verification_ccsds_class() -> type[Any]:
    verification_root = Path(__file__).resolve().parents[1] / "ccsds123_verification"
    verification_root_str = str(verification_root)
    if verification_root_str not in sys.path:
        sys.path.append(verification_root_str)
    module = importlib.import_module("ccsds123_i2_hlm.ccsds123")
    return getattr(module, "CCSDS123")


def _load_verification_header_api() -> dict[str, Any]:
    module = importlib.import_module("ccsds123_i2_hlm.header")
    return {
        "Header": getattr(module, "Header"),
        "QuantizerFidelityControlMethod": getattr(module, "QuantizerFidelityControlMethod"),
    }


class CCSDS123VerificationDemo:
    def __init__(
        self,
        output_dir: str | Path = "tmp/ccsds123_verification_demo",
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._ccsds_class = _load_verification_ccsds_class()
        self._header_api = _load_verification_header_api()
        self._verification_root = Path(__file__).resolve().parents[1] / "ccsds123_verification"
        self._verification_output = self._verification_root / "output"

    def benchmark_single_tif(
        self,
        input_file: str | Path,
        max_bands: int | None = None,
        input_layout: InputLayout = "HWC",
        mode: QuantizerMode = "lossless",
        absolute_error_limit: int = 0,
        relative_error_limit: int = 0,
        keep_raw: bool = False,
        keep_header: bool = False,
    ) -> dict[str, float | int | str]:
        input_path = Path(input_file)
        image_hwc = tifffile.imread(input_path)
        image_hwc = self._to_hwc(image_hwc, input_layout)
        if max_bands is not None and max_bands > 0:
            image_hwc = image_hwc[:, :, :max_bands]
        image_hwc = self._to_uint16_hwc(image_hwc)

        h, w, c = image_hwc.shape
        tag = f"{mode}_a{absolute_error_limit}_r{relative_error_limit}"
        raw_name = f"{input_path.stem}-{tag}-u16be-{c}x{h}x{w}.raw"
        raw_path = self.output_dir / raw_name
        self._write_bsq_u16be_raw(image_hwc, raw_path)

        header_path = self._create_header_file(raw_path, mode, absolute_error_limit, relative_error_limit)

        start = time.perf_counter()
        compressor = self._ccsds_class(str(raw_path))
        compressor.set_header_file(str(header_path))
        compressor.compress_image()
        elapsed_s = time.perf_counter() - start

        bitstream_src = self._verification_output / "z-output-bitstream.bin"
        if not bitstream_src.exists():
            raise FileNotFoundError(f"Expected output bitstream not found: {bitstream_src}")
        bitstream_dst = self.output_dir / f"{input_path.stem}_{tag}_ccsds123_verification.flex"
        shutil.copy2(bitstream_src, bitstream_dst)

        raw_size_bytes = raw_path.stat().st_size
        compressed_size_bytes = bitstream_dst.stat().st_size

        reconstructed_hwc = self._extract_reconstructed_hwc(compressor, image_hwc)
        psnr, ssim = self._compute_metrics(image_hwc, reconstructed_hwc)

        results: dict[str, float | int | str] = {
            "input_file": str(input_path),
            "raw_file": str(raw_path),
            "header_file": str(header_path),
            "bitstream_file": str(bitstream_dst),
            "input_tiff_file_size_bytes": input_path.stat().st_size,
            "raw_size_bytes": raw_size_bytes,
            "compressed_size_bytes": compressed_size_bytes,
            "compression_ratio_raw_over_compressed": (
                raw_size_bytes / compressed_size_bytes if compressed_size_bytes > 0 else 0.0
            ),
            "height": h,
            "width": w,
            "num_bands": c,
            "mode": mode,
            "absolute_error_limit": absolute_error_limit,
            "relative_error_limit": relative_error_limit,
            "psnr": psnr,
            "ssim": ssim,
            "elapsed_seconds": elapsed_s,
        }

        if not keep_raw and raw_path.exists():
            raw_path.unlink()
            results["raw_file"] = "(deleted)"
        if not keep_header and header_path.exists():
            header_path.unlink()
            results["header_file"] = "(deleted)"
        return results

    def _to_hwc(self, image: np.ndarray, input_layout: InputLayout) -> np.ndarray:
        if image.ndim == 2:
            return image[:, :, np.newaxis]
        if image.ndim != 3:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        if input_layout == "HWC":
            return image
        if input_layout == "CHW":
            return np.transpose(image, (1, 2, 0))
        raise ValueError(f"Unsupported input_layout={input_layout}. Use 'HWC' or 'CHW'.")

    def _to_uint16_hwc(self, image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint16:
            return image
        if image.dtype == np.uint8:
            return image.astype(np.uint16) << 8
        if np.issubdtype(image.dtype, np.integer):
            image_i64 = image.astype(np.int64)
            min_val = int(image_i64.min())
            if min_val < 0:
                image_i64 = image_i64 - min_val
            return np.clip(image_i64, 0, 65535).astype(np.uint16)
        image_f64 = image.astype(np.float64)
        out = np.zeros_like(image_f64, dtype=np.uint16)
        for band_idx in range(image.shape[2]):
            band = image_f64[:, :, band_idx]
            band_min = float(band.min())
            band_max = float(band.max())
            if band_max > band_min:
                out[:, :, band_idx] = np.round((band - band_min) / (band_max - band_min) * 65535.0).astype(np.uint16)
        return out

    def _write_bsq_u16be_raw(self, image_hwc_u16: np.ndarray, raw_path: Path) -> None:
        image_zyx = np.transpose(image_hwc_u16, (2, 0, 1)).astype(">u2", copy=False)
        image_zyx.tofile(raw_path)

    def _create_header_file(
        self,
        raw_path: Path,
        mode: QuantizerMode,
        absolute_error_limit: int,
        relative_error_limit: int,
    ) -> Path:
        header_cls = self._header_api["Header"]
        qfcm = self._header_api["QuantizerFidelityControlMethod"]
        header = header_cls(str(raw_path))

        if mode == "lossless":
            header.set_quantizer_fidelity_control_method(qfcm.LOSSLESS)
            header.set_fixed_offset_value(0)
            header.set_absolute_error_limit_value(0)
            header.set_relative_error_limit_value(0)
        elif mode == "absolute":
            if absolute_error_limit < 0:
                raise ValueError("absolute_error_limit must be >= 0")
            header.set_quantizer_fidelity_control_method(qfcm.ABSOLUTE_ONLY)
            header.set_absolute_error_limit_bit_depth(self._required_bit_depth(absolute_error_limit))
            header.set_absolute_error_limit_value(absolute_error_limit)
        elif mode == "relative":
            if relative_error_limit < 0:
                raise ValueError("relative_error_limit must be >= 0")
            header.set_quantizer_fidelity_control_method(qfcm.RELATIVE_ONLY)
            header.set_relative_error_limit_bit_depth(self._required_bit_depth(relative_error_limit))
            header.set_relative_error_limit_value(relative_error_limit)
        else:
            if absolute_error_limit < 0 or relative_error_limit < 0:
                raise ValueError("absolute_error_limit and relative_error_limit must be >= 0")
            header.set_quantizer_fidelity_control_method(qfcm.ABSOLUTE_AND_RELATIVE)
            header.set_absolute_error_limit_bit_depth(self._required_bit_depth(absolute_error_limit))
            header.set_relative_error_limit_bit_depth(self._required_bit_depth(relative_error_limit))
            header.set_absolute_error_limit_value(absolute_error_limit)
            header.set_relative_error_limit_value(relative_error_limit)

        header_path = self.output_dir / f"{raw_path.stem}.hdr.bin"
        header.save_header_binary(str(header_path))
        return header_path

    def _required_bit_depth(self, value: int) -> int:
        return max(1, int(value).bit_length())

    def _extract_reconstructed_hwc(self, compressor: Any, reference_hwc_u16: np.ndarray) -> np.ndarray:
        reconstructed = getattr(compressor.predictor, "clipped_quantizer_bin_center")
        reconstructed_i64 = np.asarray(reconstructed, dtype=np.int64)
        negative_mask = reconstructed_i64 < 0
        if np.any(negative_mask):
            reconstructed_i64 = reconstructed_i64.copy()
            reconstructed_i64[negative_mask] = reference_hwc_u16.astype(np.int64)[negative_mask]
        reconstructed_i64 = np.clip(reconstructed_i64, 0, 65535)
        return reconstructed_i64.astype(np.uint16)

    def _compute_metrics(self, reference_hwc_u16: np.ndarray, test_hwc_u16: np.ndarray) -> tuple[float, float]:
        ref = torch.from_numpy(reference_hwc_u16.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        pred = torch.from_numpy(test_hwc_u16.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        data_range = float(ref.max().item() - ref.min().item())
        if data_range <= 0.0:
            data_range = 1.0
        psnr = float(peak_signal_noise_ratio(pred, ref, data_range=data_range).item())
        ssim = float(structural_similarity_index_measure(pred, ref, data_range=data_range).item())
        return psnr, ssim


def benchmark_gf5_ccsds123_verification(
    input_file: str
    | Path = "/home/user/zihancao/Project/hyperspectral-1d-tokenizer/data/HyperGlobal/tmp/GF5-part1-1.img.tiff",
    output_dir: str | Path = "tmp/ccsds123_verification_demo",
    max_bands: int | None = 16,
    input_layout: InputLayout = "HWC",
    mode: QuantizerMode = "lossless",
    absolute_error_limit: int = 0,
    relative_error_limit: int = 0,
    keep_raw: bool = False,
    keep_header: bool = False,
) -> dict[str, float | int | str]:
    demo = CCSDS123VerificationDemo(output_dir=output_dir)
    return demo.benchmark_single_tif(
        input_file=input_file,
        max_bands=max_bands,
        input_layout=input_layout,
        mode=mode,
        absolute_error_limit=absolute_error_limit,
        relative_error_limit=relative_error_limit,
        keep_raw=keep_raw,
        keep_header=keep_header,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TIFF->RAW->CCSDS123 verification-model benchmark.")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path(
            "/home/user/zihancao/Project/hyperspectral-1d-tokenizer/data/HyperGlobal/tmp/GF5-part1-1.img.tiff"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/ccsds123_verification_demo2"))
    parser.add_argument(
        "--max-bands",
        type=int,
        default=16,
        help="Number of leading bands to use. <=0 means use all bands.",
    )
    parser.add_argument(
        "--input-layout",
        type=str,
        choices=["HWC", "CHW"],
        default="HWC",
        help="Input hyperspectral layout for TIFF arrays.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="lossless",
        choices=["lossless", "absolute", "relative", "absrel"],
        help="Quantizer mode for CCSDS123.",
    )
    parser.add_argument("--abs-error", type=int, default=0, help="Absolute error limit (for absolute/absrel modes).")
    parser.add_argument("--rel-error", type=int, default=0, help="Relative error limit (for relative/absrel modes).")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--keep-header", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = _parse_args()
    benchmark = benchmark_gf5_ccsds123_verification(
        input_file=cli_args.input_file,
        output_dir=cli_args.output_dir,
        max_bands=cli_args.max_bands,
        input_layout=cli_args.input_layout,
        mode=cli_args.mode,
        absolute_error_limit=cli_args.abs_error,
        relative_error_limit=cli_args.rel_error,
        keep_raw=cli_args.keep_raw,
        keep_header=cli_args.keep_header,
    )
    for key, value in benchmark.items():
        print(f"{key}: {value}")

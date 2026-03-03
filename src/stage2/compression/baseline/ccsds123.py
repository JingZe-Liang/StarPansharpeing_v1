from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import tifffile
import torch
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure

CCSDSMode = Literal["lossless", "near_lossless"]
InputLayout = Literal["HWC", "CHW"]


class CCSDS123HSICompressor:
    def __init__(
        self,
        mode: CCSDSMode = "lossless",
        dynamic_range: int = 16,
        absolute_error_limit: float = 2.0,
        entropy_coder_type: str = "hybrid",
        use_optimized: bool = False,
        device: str = "cpu",
        optimization_mode: str = "full",
    ):
        self.mode = mode
        self.dynamic_range = dynamic_range
        self.absolute_error_limit = absolute_error_limit
        self.entropy_coder_type = entropy_coder_type
        self.use_optimized = use_optimized
        self.device = device
        self.optimization_mode = optimization_mode
        self._ccsds_api = _load_ccsds_api()
        self._optimized_api = _load_optimized_api() if use_optimized else None

    def encode(self, data_hwc: np.ndarray) -> dict[str, Any]:
        image_zyx = self._prepare_to_zyx(data_hwc)
        num_bands = image_zyx.shape[0]
        image_t = torch.from_numpy(image_zyx.astype(np.float32))

        if self.use_optimized:
            if self._optimized_api is None:
                raise RuntimeError("Optimized CCSDS API is not available.")
            return self._encode_optimized(image_t, num_bands)

        if self.mode == "lossless":
            compressor = self._ccsds_api["create_lossless_compressor"](
                num_bands=num_bands,
                dynamic_range=self.dynamic_range,
                entropy_coder_type=self.entropy_coder_type,
            )
        else:
            abs_limits = torch.ones(num_bands, dtype=torch.float32) * float(self.absolute_error_limit)
            compressor = self._ccsds_api["create_near_lossless_compressor"](
                num_bands=num_bands,
                absolute_error_limits=abs_limits,
                dynamic_range=self.dynamic_range,
                entropy_coder_type=self.entropy_coder_type,
            )
        compressed = compressor.compress(image_t)
        return compressed

    def _encode_optimized(self, image_t: torch.Tensor, num_bands: int) -> dict[str, Any]:
        assert self._optimized_api is not None
        if self.mode == "lossless":
            compressor = self._optimized_api["create_optimized_lossless_compressor"](
                num_bands=num_bands,
                dynamic_range=self.dynamic_range,
                optimization_mode=self.optimization_mode,
                device=self.device,
                use_mixed_precision=self.device.startswith("cuda"),
                gpu_batch_size=num_bands,
            )
        else:
            abs_limits = torch.ones(num_bands, dtype=torch.float32) * float(self.absolute_error_limit)
            compressor = self._optimized_api["create_optimized_near_lossless_compressor"](
                num_bands=num_bands,
                absolute_error_limits=abs_limits,
                dynamic_range=self.dynamic_range,
                optimization_mode=self.optimization_mode,
                device=self.device,
                use_mixed_precision=self.device.startswith("cuda"),
                gpu_batch_size=num_bands,
            )
        forward_results = compressor(image_t)
        compressed_bitstream = compressor.compress(image_t)
        return {
            "compressed_bitstream": compressed_bitstream,
            "compression_metadata": {
                "optimized": True,
                "device": str(compressor.device),
            },
            "intermediate_data": {
                "reconstructed_samples": forward_results["reconstructed_samples"].detach().cpu(),
            },
        }

    def decode(self, compressed_data: dict[str, Any]) -> np.ndarray:
        if "intermediate_data" in compressed_data and "reconstructed_samples" in compressed_data["intermediate_data"]:
            reconstruct_t = compressed_data["intermediate_data"]["reconstructed_samples"]
        else:
            reconstruct_t = self._ccsds_api["decompress"](compressed_data)
        reconstruct_np = reconstruct_t.detach().cpu().numpy()
        if reconstruct_np.ndim != 3:
            raise ValueError(f"Unexpected reconstructed shape: {reconstruct_np.shape}")
        return np.transpose(reconstruct_np, (1, 2, 0))

    def evaluate(self, data_hwc: np.ndarray) -> dict[str, float | int | str]:
        prepared = self._prepare_hwc_uint16(data_hwc)
        compressed = self.encode(prepared)
        decoded = self.decode(compressed)
        metrics = self._compute_metrics(prepared, decoded)

        compressed_bitstream = compressed["compressed_bitstream"]
        compressed_size_bytes = len(compressed_bitstream)
        original_size_bytes = int(prepared.nbytes)
        return {
            "original_size": original_size_bytes,
            "in_memory_size_bytes": original_size_bytes,
            "compressed_size": compressed_size_bytes,
            "compression_ratio": original_size_bytes / compressed_size_bytes if compressed_size_bytes > 0 else 0.0,
            "num_bands": prepared.shape[2],
            "height": prepared.shape[0],
            "width": prepared.shape[1],
            "device": self.device if self.use_optimized else "cpu",
            **metrics,
        }

    def benchmark_single_tif_to_tmp(
        self,
        input_file: str | Path,
        output_dir: str | Path = "tmp/ccsds123_benchmark",
        prefix: str | None = None,
        max_bands: int | None = None,
        input_layout: InputLayout = "HWC",
    ) -> dict[str, float | int | str]:
        input_path = Path(input_file)
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        output_prefix = prefix or input_path.stem

        image = tifffile.imread(input_path)
        image = self._to_hwc(image, input_layout)
        if max_bands is not None and max_bands > 0:
            image = image[:, :, :max_bands]

        prepared = self._prepare_hwc_uint16(image)
        compressed = self.encode(prepared)
        decoded = self.decode(compressed)
        metrics = self._compute_metrics(prepared, decoded)

        bitstream_path = output_dir_path / f"{output_prefix}_ccsds123_{self.mode}.bin"
        bitstream_path.write_bytes(compressed["compressed_bitstream"])
        compressed_size_bytes = bitstream_path.stat().st_size
        original_size_bytes = int(prepared.nbytes)

        return {
            "input_file": str(input_path),
            "output_file": str(bitstream_path),
            "input_file_size_bytes": input_path.stat().st_size,
            "output_file_size_bytes": compressed_size_bytes,
            "original_size": original_size_bytes,
            "in_memory_size_bytes": original_size_bytes,
            "compressed_size": compressed_size_bytes,
            "compression_ratio": original_size_bytes / compressed_size_bytes if compressed_size_bytes > 0 else 0.0,
            "num_bands": prepared.shape[2],
            "height": prepared.shape[0],
            "width": prepared.shape[1],
            "device": self.device if self.use_optimized else "cpu",
            **metrics,
        }

    def _prepare_to_zyx(self, data_hwc: np.ndarray) -> np.ndarray:
        prepared = self._prepare_hwc_uint16(data_hwc)
        return np.transpose(prepared, (2, 0, 1))

    def _to_hwc(self, data: np.ndarray, input_layout: InputLayout) -> np.ndarray:
        if data.ndim == 2:
            return data[:, :, np.newaxis]
        if data.ndim != 3:
            raise ValueError(f"Unexpected TIFF shape: {data.shape}")
        if input_layout == "HWC":
            return data
        if input_layout == "CHW":
            return np.transpose(data, (1, 2, 0))
        raise ValueError(f"Unsupported input_layout={input_layout}. Use 'HWC' or 'CHW'.")

    def _prepare_hwc_uint16(self, data_hwc: np.ndarray) -> np.ndarray:
        if data_hwc.dtype == np.uint16:
            return data_hwc
        if data_hwc.dtype == np.uint8:
            return data_hwc.astype(np.uint16) << 8
        if data_hwc.dtype == np.int16:
            data_i32 = data_hwc.astype(np.int32)
            min_val = int(data_i32.min())
            if min_val < 0:
                data_i32 = data_i32 - min_val
            data_i32 = np.clip(data_i32, 0, 65535)
            return data_i32.astype(np.uint16)
        if np.issubdtype(data_hwc.dtype, np.floating):
            return self._normalize_float_to_uint16(data_hwc)
        return np.clip(data_hwc.astype(np.int64), 0, 65535).astype(np.uint16)

    def _normalize_float_to_uint16(self, data_hwc: np.ndarray) -> np.ndarray:
        h, w, c = data_hwc.shape
        out = np.zeros((h, w, c), dtype=np.uint16)
        for idx in range(c):
            band = data_hwc[:, :, idx].astype(np.float64)
            min_val = float(np.min(band))
            max_val = float(np.max(band))
            if max_val > min_val:
                out[:, :, idx] = np.round(((band - min_val) / (max_val - min_val)) * 65535.0).astype(np.uint16)
        return out

    def _compute_metrics(self, reference_hwc: np.ndarray, test_hwc: np.ndarray) -> dict[str, float]:
        if reference_hwc.shape != test_hwc.shape:
            raise ValueError(f"Shape mismatch for metrics: {reference_hwc.shape} vs {test_hwc.shape}")
        ref = torch.from_numpy(reference_hwc.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        pred = torch.from_numpy(test_hwc.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        data_range = float(torch.max(ref).item() - torch.min(ref).item())
        if data_range <= 0.0:
            data_range = 1.0
        psnr = float(peak_signal_noise_ratio(pred, ref, data_range=data_range).item())
        ssim = float(structural_similarity_index_measure(pred, ref, data_range=data_range).item())
        return {"psnr": psnr, "ssim": ssim}


def _load_ccsds_api() -> dict[str, Any]:
    ccsds_src = Path(__file__).resolve().parents[1] / "CCSDS_HSI" / "src"
    ccsds_src_str = str(ccsds_src)
    if ccsds_src_str not in sys.path:
        sys.path.append(ccsds_src_str)
    mod = importlib.import_module("ccsds")
    return {
        "create_lossless_compressor": getattr(mod, "create_lossless_compressor"),
        "create_near_lossless_compressor": getattr(mod, "create_near_lossless_compressor"),
        "decompress": getattr(mod, "decompress"),
    }


def _load_optimized_api() -> dict[str, Any]:
    ccsds_src = Path(__file__).resolve().parents[1] / "CCSDS_HSI" / "src"
    ccsds_src_str = str(ccsds_src)
    if ccsds_src_str not in sys.path:
        sys.path.append(ccsds_src_str)
    mod = importlib.import_module("optimized.optimized_compressor")
    return {
        "create_optimized_lossless_compressor": getattr(mod, "create_optimized_lossless_compressor"),
        "create_optimized_near_lossless_compressor": getattr(mod, "create_optimized_near_lossless_compressor"),
    }


def benchmark_gf5_ccsds123(
    input_file: str
    | Path = "/home/user/zihancao/Project/hyperspectral-1d-tokenizer/data/HyperGlobal/tmp/GF5-part1-1.img.tiff",
    mode: CCSDSMode = "lossless",
    output_dir: str | Path = "tmp/ccsds123_benchmark",
    absolute_error_limit: float = 2.0,
    max_bands: int | None = None,
    input_layout: InputLayout = "HWC",
    use_optimized: bool = False,
    device: str = "cpu",
    optimization_mode: str = "full",
) -> dict[str, float | int | str]:
    compressor = CCSDS123HSICompressor(
        mode=mode,
        dynamic_range=16,
        absolute_error_limit=absolute_error_limit,
        entropy_coder_type="hybrid",
        use_optimized=use_optimized,
        device=device,
        optimization_mode=optimization_mode,
    )
    return compressor.benchmark_single_tif_to_tmp(
        input_file=input_file,
        output_dir=output_dir,
        max_bands=max_bands,
        input_layout=input_layout,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CCSDS-123 benchmark for hyperspectral TIFF.")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path(
            "/home/user/zihancao/Project/hyperspectral-1d-tokenizer/data/HyperGlobal/tmp/GF5-part1-1.img.tiff"
        ),
    )
    parser.add_argument("--mode", choices=["lossless", "near_lossless"], default="lossless")
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/ccsds123_benchmark"))
    parser.add_argument("--absolute-error-limit", type=float, default=2.0)
    parser.add_argument("--max-bands", type=int, default=None)
    parser.add_argument("--input-layout", choices=["HWC", "CHW"], default="HWC")
    parser.add_argument("--use-optimized", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--optimization-mode", choices=["full", "causal", "streaming"], default="full")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    results = benchmark_gf5_ccsds123(
        input_file=args.input_file,
        mode=args.mode,
        output_dir=args.output_dir,
        absolute_error_limit=args.absolute_error_limit,
        max_bands=args.max_bands,
        input_layout=args.input_layout,
        use_optimized=args.use_optimized,
        device=args.device,
        optimization_mode=args.optimization_mode,
    )
    for k, v in results.items():
        print(f"{k}: {v}")

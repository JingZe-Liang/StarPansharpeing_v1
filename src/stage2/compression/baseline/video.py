from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
import tifffile
import torch
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure

VideoCodec = Literal["h265", "h266"]
InputLayout = Literal["HWC", "CHW"]


class HyperspectralVideoCompressor:
    def __init__(
        self,
        codec: VideoCodec = "h265",
        quality: int = 95,
        bit_depth: Literal[8, 10] = 10,
        ffmpeg_bin: str = "ffmpeg",
        preset: str = "medium",
    ):
        self.codec = codec
        self.quality = quality
        self.bit_depth = bit_depth
        self.ffmpeg_bin = ffmpeg_bin
        self.preset = preset

    def compress_hyperspectral_file(
        self,
        input_file: str | Path,
        output_dir: str | Path = "tmp/video_codec_benchmark",
        prefix: str | None = None,
        input_layout: InputLayout = "HWC",
    ) -> tuple[list[str], dict[str, float | int]]:
        input_path = Path(input_file)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        data_hwc = self._read_tiff_hwc(input_path, input_layout=input_layout)
        output_prefix = prefix or input_path.stem
        compressed_files, stats = self.compress_hyperspectral_array(data_hwc, output_dir, output_prefix)
        stats["input_file_size_bytes"] = input_path.stat().st_size
        return compressed_files, stats

    def compress_hyperspectral_array(
        self,
        hyperspectral_data_hwc: np.ndarray,
        output_dir: str | Path,
        prefix: str,
    ) -> tuple[list[str], dict[str, float | int]]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        prepared = self._prepare_array(hyperspectral_data_hwc)
        frames_chw = self._quantize_to_frames(prepared, bit_depth=self.bit_depth)
        height, width, num_bands = prepared.shape

        ext = ".mkv" if self.codec == "h266" else ".mp4"
        output_file = output_dir / f"{prefix}_{self.codec}_q{self.quality}_b{self.bit_depth}{ext}"
        self._encode_frames(frames_chw, width, height, output_file)

        original_size = prepared.nbytes
        compressed_size = output_file.stat().st_size
        stats: dict[str, float | int] = {
            "original_size": original_size,
            "in_memory_size_bytes": original_size,
            "compressed_size": compressed_size,
            "compression_ratio": original_size / compressed_size if compressed_size > 0 else 0.0,
            "num_bands": num_bands,
            "height": height,
            "width": width,
        }
        return [str(output_file)], stats

    def evaluate_file(
        self,
        input_file: str | Path,
        output_dir: str | Path = "tmp/video_codec_benchmark",
        prefix: str | None = None,
        input_layout: InputLayout = "HWC",
    ) -> dict[str, float | int | str]:
        input_path = Path(input_file)
        original = self._prepare_array(self._read_tiff_hwc(input_path, input_layout=input_layout))
        output_prefix = prefix or input_path.stem
        compressed_files, stats = self.compress_hyperspectral_array(original, output_dir, output_prefix)
        decoded = self.decompress_hyperspectral_file(
            compressed_file=compressed_files[0],
            shape_hwc=original.shape,
        )
        metrics = self._compute_metrics(reference=original, test=decoded)
        output_file = Path(compressed_files[0])
        result: dict[str, float | int | str] = {
            "input_file": str(input_path),
            "output_file": str(output_file),
            "output_file_size_bytes": output_file.stat().st_size,
            "input_file_size_bytes": input_path.stat().st_size,
            **stats,
            **metrics,
        }
        return result

    def decompress_hyperspectral_file(self, compressed_file: str | Path, shape_hwc: tuple[int, int, int]) -> np.ndarray:
        compressed_path = Path(compressed_file)
        height, width, num_bands = shape_hwc
        frames_chw = self._decode_frames(
            compressed_file=compressed_path,
            width=width,
            height=height,
            expected_frames=num_bands,
        )
        return self._dequantize_frames(frames_chw, bit_depth=self.bit_depth)

    def _read_tiff_hwc(self, input_file: Path, input_layout: InputLayout = "HWC") -> np.ndarray:
        data = tifffile.imread(input_file)
        if data.ndim == 2:
            return data[:, :, np.newaxis]
        if data.ndim != 3:
            raise ValueError(f"Unexpected TIFF shape: {data.shape}")
        if input_layout == "HWC":
            return data
        if input_layout == "CHW":
            return np.transpose(data, (1, 2, 0))
        raise ValueError(f"Unsupported input_layout={input_layout}. Use 'HWC' or 'CHW'.")

    def _prepare_array(self, data_hwc: np.ndarray) -> np.ndarray:
        if data_hwc.dtype in (np.uint8, np.uint16):
            return data_hwc
        if data_hwc.dtype == np.int16:
            shifted = data_hwc.astype(np.int32) - np.iinfo(np.int16).min
            return shifted.astype(np.uint16)
        if np.issubdtype(data_hwc.dtype, np.floating):
            return self._normalize_float_to_uint16(data_hwc)
        return data_hwc.astype(np.uint16)

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

    def _quantize_to_frames(self, data_hwc_uint16: np.ndarray, bit_depth: Literal[8, 10]) -> np.ndarray:
        if bit_depth == 8:
            frames = (data_hwc_uint16 >> 8).astype(np.uint8)
        else:
            frames = (data_hwc_uint16 >> 6).astype(np.uint16)
        return np.transpose(frames, (2, 0, 1))

    def _dequantize_frames(self, frames_chw: np.ndarray, bit_depth: Literal[8, 10]) -> np.ndarray:
        if bit_depth == 8:
            restored = (frames_chw.astype(np.uint16) << 8).astype(np.uint16)
        else:
            restored = (frames_chw.astype(np.uint16) << 6).astype(np.uint16)
        return np.transpose(restored, (1, 2, 0))

    def _encode_frames(self, frames_chw: np.ndarray, width: int, height: int, output_file: Path) -> None:
        pix_fmt = "gray10le" if self.bit_depth == 10 else "gray"
        encoder, params = self._build_encoder_params()
        with tempfile.NamedTemporaryFile(suffix=".raw", delete=True) as raw_file:
            frames_chw.tofile(raw_file.name)
            cmd = [
                self.ffmpeg_bin,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                pix_fmt,
                "-s:v",
                f"{width}x{height}",
                "-r",
                "1",
                "-i",
                raw_file.name,
                "-an",
                "-c:v",
                encoder,
                "-preset",
                self.preset,
                *params,
                str(output_file),
            ]
            self._run_cmd(cmd)

    def _decode_frames(self, compressed_file: Path, width: int, height: int, expected_frames: int) -> np.ndarray:
        pix_fmt = "gray10le" if self.bit_depth == 10 else "gray"
        dtype = np.uint16 if self.bit_depth == 10 else np.uint8
        frame_pixels = width * height
        with tempfile.NamedTemporaryFile(suffix=".raw", delete=True) as raw_dec:
            cmd = [
                self.ffmpeg_bin,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(compressed_file),
                "-f",
                "rawvideo",
                "-pix_fmt",
                pix_fmt,
                raw_dec.name,
            ]
            self._run_cmd(cmd)
            decoded = np.fromfile(raw_dec.name, dtype=dtype)
        expected_values = expected_frames * frame_pixels
        if decoded.size < expected_values:
            raise RuntimeError(
                f"Decoded data is too short: expected {expected_values} values, got {decoded.size} for {compressed_file}."
            )
        decoded = decoded[:expected_values]
        return decoded.reshape(expected_frames, height, width)

    def _build_encoder_params(self) -> tuple[str, list[str]]:
        if self.codec == "h265":
            crf = int(np.clip(round(100 - self.quality), 0, 51))
            return "libx265", ["-x265-params", "keyint=1:min-keyint=1:scenecut=0", "-crf", str(crf)]

        # H.266 / VVC path, requires ffmpeg with libvvenc.
        if self.codec == "h266":
            qp = int(np.clip(round(100 - self.quality), 0, 63))
            return "libvvenc", ["-qp", str(qp)]

        raise ValueError(f"Unsupported codec: {self.codec}")

    def _compute_metrics(self, reference: np.ndarray, test: np.ndarray) -> dict[str, float]:
        if reference.shape != test.shape:
            raise ValueError(f"Shape mismatch for metrics: {reference.shape} vs {test.shape}")
        ref_t = torch.from_numpy(reference.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        test_t = torch.from_numpy(test.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        data_range = float(torch.max(ref_t).item() - torch.min(ref_t).item())
        if data_range <= 0.0:
            data_range = 1.0
        psnr = float(peak_signal_noise_ratio(test_t, ref_t, data_range=data_range).item())
        ssim = float(structural_similarity_index_measure(test_t, ref_t, data_range=data_range).item())
        return {"psnr": psnr, "ssim": ssim}

    def _run_cmd(self, cmd: list[str]) -> None:
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            if self.codec == "h266" and "Unknown encoder" in stderr:
                raise RuntimeError(
                    "FFmpeg does not have H.266/VVC encoder (libvvenc). "
                    "Please install FFmpeg with libvvenc support or switch codec='h265'."
                ) from exc
            raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{stderr}") from exc


def benchmark_single_tif_video_to_tmp(
    input_file: str | Path,
    codec: VideoCodec = "h265",
    quality: int = 95,
    bit_depth: Literal[8, 10] = 10,
    output_dir: str | Path = "tmp/video_codec_benchmark",
    input_layout: InputLayout = "HWC",
) -> dict[str, float | int | str]:
    compressor = HyperspectralVideoCompressor(
        codec=codec,
        quality=quality,
        bit_depth=bit_depth,
    )
    return compressor.evaluate_file(input_file=input_file, output_dir=output_dir, input_layout=input_layout)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark hyperspectral compression using video codecs.")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path(
            "/home/user/zihancao/Project/hyperspectral-1d-tokenizer/data/HyperGlobal/tmp/GF5-part1-1.img.tiff"
        ),
        help="Input hyperspectral TIFF file path.",
    )
    parser.add_argument(
        "--codec",
        type=str,
        choices=["h265", "h266"],
        default="h265",
        help="Video codec backend.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="Codec quality setting (higher is better quality).",
    )
    parser.add_argument(
        "--bit-depth",
        type=int,
        choices=[8, 10],
        default=10,
        help="Intermediate video bit depth.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/video_codec_benchmark"),
        help="Output directory for compressed files.",
    )
    parser.add_argument("--input-layout", choices=["HWC", "CHW"], default="HWC", help="Input TIFF layout.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = benchmark_single_tif_video_to_tmp(
        input_file=args.input_file,
        codec=args.codec,
        quality=args.quality,
        bit_depth=args.bit_depth,
        output_dir=args.output_dir,
        input_layout=args.input_layout,
    )
    for key, value in result.items():
        print(f"{key}: {value}")

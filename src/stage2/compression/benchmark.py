from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from stage2.compression.baseline.ccsds123_verification_demo import CCSDS123VerificationDemo
from stage2.compression.baseline.jpeg import HyperspectralJPEG2000Compressor
from stage2.compression.baseline.video import HyperspectralVideoCompressor


@dataclass
class MethodResult:
    file: str
    method: str
    psnr: float
    ssim: float
    compression_ratio: float
    compressed_size_bytes: int
    original_size_bytes: int
    elapsed_seconds: float
    status: str
    error: str


def _list_tif_files(input_dir: Path, recursive: bool) -> list[Path]:
    patterns = ("*.tif", "*.tiff", "*.TIF", "*.TIFF")
    files: list[Path] = []
    if recursive:
        for pattern in patterns:
            files.extend(input_dir.rglob(pattern))
    else:
        for pattern in patterns:
            files.extend(input_dir.glob(pattern))
    return sorted(path for path in files if path.is_file())


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _run_jp2k(
    input_file: Path,
    output_dir: Path,
    input_layout: str,
    quality: float,
    backend: str,
    reversible: bool,
) -> dict[str, Any]:
    compressor = HyperspectralJPEG2000Compressor(quality=quality, reversible=reversible)
    return compressor.evaluate_psnr_after_compression_file(
        input_file=input_file,
        output_dir=output_dir,
        normalize=False,
        input_layout=input_layout,
        backend=backend,  # type: ignore[arg-type]
    )


def _run_video(
    input_file: Path,
    output_dir: Path,
    input_layout: str,
    codec: str,
    quality: int,
    bit_depth: int,
) -> dict[str, Any]:
    compressor = HyperspectralVideoCompressor(codec=codec, quality=quality, bit_depth=bit_depth)  # type: ignore[arg-type]
    return compressor.evaluate_file(
        input_file=input_file,
        output_dir=output_dir,
        input_layout=input_layout,  # type: ignore[arg-type]
    )


def _run_ccsds(
    input_file: Path,
    output_dir: Path,
    input_layout: str,
    mode: str,
    abs_error: int,
    keep_raw: bool,
) -> dict[str, Any]:
    demo = CCSDS123VerificationDemo(output_dir=output_dir)
    return demo.benchmark_single_tif(
        input_file=input_file,
        max_bands=-1,
        input_layout=input_layout,  # type: ignore[arg-type]
        mode=mode,  # type: ignore[arg-type]
        absolute_error_limit=abs_error,
        relative_error_limit=0,
        keep_raw=keep_raw,
        keep_header=False,
    )


def _format_float(x: float, digits: int = 4) -> str:
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf"
    return f"{x:.{digits}f}"


def _print_summary_table(results: list[MethodResult]) -> None:
    methods = sorted({r.method for r in results})
    header = [
        "method",
        "ok/total",
        "mean_psnr",
        "mean_ssim",
        "mean_ratio",
        "mean_time_s",
    ]

    rows: list[list[str]] = []
    for method in methods:
        items = [r for r in results if r.method == method]
        ok_items = [r for r in items if r.status == "ok"]
        psnr_vals = np.array([r.psnr for r in ok_items], dtype=np.float64)
        ssim_vals = np.array([r.ssim for r in ok_items], dtype=np.float64)
        ratio_vals = np.array([r.compression_ratio for r in ok_items], dtype=np.float64)
        time_vals = np.array([r.elapsed_seconds for r in ok_items], dtype=np.float64)

        rows.append(
            [
                method,
                f"{len(ok_items)}/{len(items)}",
                _format_float(float(np.nanmean(psnr_vals)) if len(psnr_vals) > 0 else float("nan"), 3),
                _format_float(float(np.nanmean(ssim_vals)) if len(ssim_vals) > 0 else float("nan"), 4),
                _format_float(float(np.nanmean(ratio_vals)) if len(ratio_vals) > 0 else float("nan"), 3),
                _format_float(float(np.nanmean(time_vals)) if len(time_vals) > 0 else float("nan"), 3),
            ]
        )

    col_widths = [len(h) for h in header]
    for row in rows:
        for idx, val in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(val))

    def _line(values: list[str]) -> str:
        return " | ".join(v.ljust(col_widths[i]) for i, v in enumerate(values))

    print("\nSummary (PSNR/SSIM):")
    print(_line(header))
    print("-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(_line(row))


def _save_csv(results: list[MethodResult], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_csv = output_dir / "benchmark_detail.csv"
    summary_csv = output_dir / "benchmark_summary.csv"

    with detail_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "file",
                "method",
                "psnr",
                "ssim",
                "compression_ratio",
                "compressed_size_bytes",
                "original_size_bytes",
                "elapsed_seconds",
                "status",
                "error",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.file,
                    r.method,
                    r.psnr,
                    r.ssim,
                    r.compression_ratio,
                    r.compressed_size_bytes,
                    r.original_size_bytes,
                    r.elapsed_seconds,
                    r.status,
                    r.error,
                ]
            )

    methods = sorted({r.method for r in results})
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "ok", "total", "mean_psnr", "mean_ssim", "mean_ratio", "mean_time_s"])
        for method in methods:
            items = [r for r in results if r.method == method]
            ok_items = [r for r in items if r.status == "ok"]
            psnr_vals = np.array([r.psnr for r in ok_items], dtype=np.float64)
            ssim_vals = np.array([r.ssim for r in ok_items], dtype=np.float64)
            ratio_vals = np.array([r.compression_ratio for r in ok_items], dtype=np.float64)
            time_vals = np.array([r.elapsed_seconds for r in ok_items], dtype=np.float64)

            writer.writerow(
                [
                    method,
                    len(ok_items),
                    len(items),
                    float(np.nanmean(psnr_vals)) if len(psnr_vals) > 0 else float("nan"),
                    float(np.nanmean(ssim_vals)) if len(ssim_vals) > 0 else float("nan"),
                    float(np.nanmean(ratio_vals)) if len(ratio_vals) > 0 else float("nan"),
                    float(np.nanmean(time_vals)) if len(time_vals) > 0 else float("nan"),
                ]
            )

    return detail_csv, summary_csv


def run_benchmark(args: argparse.Namespace) -> list[MethodResult]:
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    files = _list_tif_files(input_dir=input_dir, recursive=args.recursive)
    if args.max_files is not None and args.max_files > 0:
        files = files[: args.max_files]
    if len(files) == 0:
        raise FileNotFoundError(f"No tif/tiff files found in: {input_dir}")

    methods: list[str] = ["jp2k", "video", "ccsds_lossless", "ccsds_lossy"]
    results: list[MethodResult] = []

    total_steps = len(files) * len(methods)
    pbar = tqdm(total=total_steps, desc="Benchmark", unit="task")

    try:
        for file_path in files:
            for method in methods:
                method_out = output_dir / method
                method_out.mkdir(parents=True, exist_ok=True)
                t0 = time.perf_counter()
                try:
                    if method == "jp2k":
                        out = _run_jp2k(
                            input_file=file_path,
                            output_dir=method_out,
                            input_layout=args.input_layout,
                            quality=args.jp2k_quality,
                            backend=args.jp2k_backend,
                            reversible=args.jp2k_reversible,
                        )
                    elif method == "video":
                        out = _run_video(
                            input_file=file_path,
                            output_dir=method_out,
                            input_layout=args.input_layout,
                            codec=args.video_codec,
                            quality=args.video_quality,
                            bit_depth=args.video_bit_depth,
                        )
                    elif method == "ccsds_lossless":
                        out = _run_ccsds(
                            input_file=file_path,
                            output_dir=method_out,
                            input_layout=args.input_layout,
                            mode="lossless",
                            abs_error=0,
                            keep_raw=args.keep_ccsds_raw,
                        )
                    elif method == "ccsds_lossy":
                        out = _run_ccsds(
                            input_file=file_path,
                            output_dir=method_out,
                            input_layout=args.input_layout,
                            mode="absolute",
                            abs_error=args.ccsds_abs_error,
                            keep_raw=args.keep_ccsds_raw,
                        )
                    else:
                        raise ValueError(f"Unsupported method: {method}")

                    elapsed = time.perf_counter() - t0
                    result = MethodResult(
                        file=str(file_path),
                        method=method,
                        psnr=_safe_float(out.get("psnr")),
                        ssim=_safe_float(out.get("ssim")),
                        compression_ratio=_safe_float(
                            out.get("compression_ratio", out.get("compression_ratio_raw_over_compressed"))
                        ),
                        compressed_size_bytes=_safe_int(out.get("compressed_size", out.get("compressed_size_bytes"))),
                        original_size_bytes=_safe_int(out.get("original_size", out.get("raw_size_bytes"))),
                        elapsed_seconds=elapsed,
                        status="ok",
                        error="",
                    )
                except Exception as exc:
                    elapsed = time.perf_counter() - t0
                    result = MethodResult(
                        file=str(file_path),
                        method=method,
                        psnr=float("nan"),
                        ssim=float("nan"),
                        compression_ratio=float("nan"),
                        compressed_size_bytes=-1,
                        original_size_bytes=-1,
                        elapsed_seconds=elapsed,
                        status="error",
                        error=str(exc),
                    )
                results.append(result)
                pbar.update(1)
                pbar.set_postfix(file=file_path.name, method=method, status=result.status)
    finally:
        pbar.close()

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch benchmark for hyperspectral compression baselines.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing tif/tiff files.")
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/compression_benchmark"), help="Output directory.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan subdirectories.")
    parser.add_argument("--max-files", type=int, default=None, help="Benchmark only first N files.")
    parser.add_argument("--input-layout", choices=["HWC", "CHW"], default="HWC")

    parser.add_argument("--jp2k-quality", type=float, default=75.0)
    parser.add_argument("--jp2k-backend", choices=["tiff_jpeg2000", "gdal_jp2"], default="tiff_jpeg2000")
    parser.add_argument("--jp2k-reversible", action="store_true")

    parser.add_argument("--video-codec", choices=["h265", "h266"], default="h265")
    parser.add_argument("--video-quality", type=int, default=75)
    parser.add_argument("--video-bit-depth", choices=[8, 10], type=int, default=10)

    parser.add_argument("--ccsds-abs-error", type=int, default=20, help="Absolute error for ccsds_lossy.")
    parser.add_argument("--keep-ccsds-raw", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results = run_benchmark(args)
    _print_summary_table(results)
    detail_csv, summary_csv = _save_csv(results, args.output_dir)
    print(f"\nDetail CSV: {detail_csv}")
    print(f"Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()

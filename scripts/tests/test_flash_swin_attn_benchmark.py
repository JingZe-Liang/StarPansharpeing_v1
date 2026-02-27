from __future__ import annotations

import argparse
import csv
import gc
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.benchmark import Timer

from src.stage1.cosmos.third_party.FlashWindowAttention.flash_swin_attn.func_flash_swin import flash_swin_attn_func
from src.stage1.cosmos.third_party.FlashWindowAttention.flash_swin_attn.func_flash_swin_v2 import (
    flash_swin_attn_func_v2,
)
from src.stage1.cosmos.third_party.FlashWindowAttention.flash_swin_attn.func_swin import mha_core

try:
    import psutil
except ImportError as exc:
    msg = "psutil is required for RAM benchmarking. Please install psutil."
    raise RuntimeError(msg) from exc


@dataclass(frozen=True)
class BenchCase:
    batch: int
    seq: int
    head_dim: int

    @property
    def label(self) -> str:
        return f"B{self.batch}-L{self.seq}-D{self.head_dim}"


def _parse_cases(raw: str) -> list[BenchCase]:
    cases: list[BenchCase] = []
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        b_str, l_str, d_str = chunk.split("x")
        case = BenchCase(batch=int(b_str), seq=int(l_str), head_dim=int(d_str))
        if case.head_dim % 16 != 0:
            msg = f"head_dim must be divisible by 16, got {case.head_dim}"
            raise ValueError(msg)
        cases.append(case)
    if not cases:
        raise ValueError("No benchmark case was provided")
    return cases


def _make_shift_mask(num_windows: int, seq: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((num_windows, seq, seq), dtype=dtype, device=device)
    block = max(1, seq // 4)
    split = min(seq, block * 3)
    mask[:, :block, block:split] = -100.0
    mask[:, block:split, :block] = -100.0
    mask[:, -block:, :-block] = -100.0
    return mask


def _expand_window_mask(mask_nw: torch.Tensor, batch: int) -> torch.Tensor:
    n_w, seq, _ = mask_nw.shape
    return mask_nw.unsqueeze(0).expand(batch // n_w, n_w, seq, seq).reshape(batch, seq, seq).contiguous()


def _run_backend(
    backend: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    mask_nw: torch.Tensor,
    scale_qk: float,
) -> torch.Tensor:
    if backend == "eager":
        return mha_core(q, k, v, bias, mask_nw, scale_qk)

    if backend == "sdpa":
        attn_bias = bias.unsqueeze(0)
        attn_bias = attn_bias + _expand_window_mask(mask_nw, q.size(0)).unsqueeze(1)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=0.0, scale=scale_qk)

    if backend == "triton_v1":
        return flash_swin_attn_func(q, k, v, bias, _expand_window_mask(mask_nw, q.size(0)), scale_qk)

    if backend == "triton_v2":
        return flash_swin_attn_func_v2(q, k, v, bias, _expand_window_mask(mask_nw, q.size(0)), scale_qk)

    raise ValueError(f"Unsupported backend: {backend}")


def _one_step(
    backend: str,
    q_base: torch.Tensor,
    k_base: torch.Tensor,
    v_base: torch.Tensor,
    bias_base: torch.Tensor,
    mask_nw: torch.Tensor,
    scale_qk: float,
) -> None:
    q = q_base.detach().clone().requires_grad_(True)
    k = k_base.detach().clone().requires_grad_(True)
    v = v_base.detach().clone().requires_grad_(True)
    b = bias_base.detach().clone().requires_grad_(True)

    out = _run_backend(backend, q, k, v, b, mask_nw, scale_qk)
    loss = out.float().square().mean() + 0.1 * out.float().abs().mean()
    loss.backward()


@dataclass
class BenchResult:
    backend: str
    case: str
    batch: int
    seq: int
    head_dim: int
    heads: int
    dtype: str
    time_ms: float
    time_iqr_ms: float
    gpu_peak_mb: float
    cpu_delta_mb: float


def _measure_memory(
    backend: str,
    q_base: torch.Tensor,
    k_base: torch.Tensor,
    v_base: torch.Tensor,
    bias_base: torch.Tensor,
    mask_nw: torch.Tensor,
    scale_qk: float,
    proc: psutil.Process,
) -> tuple[float, float]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    rss_before = proc.memory_info().rss
    _one_step(backend, q_base, k_base, v_base, bias_base, mask_nw, scale_qk)
    torch.cuda.synchronize()
    rss_after = proc.memory_info().rss

    gpu_peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    cpu_delta_mb = max(0.0, (rss_after - rss_before) / (1024**2))
    return gpu_peak_mb, cpu_delta_mb


def _benchmark_case(
    case: BenchCase,
    backend: str,
    heads: int,
    dtype: torch.dtype,
    min_run_time: float,
    warmup_steps: int,
    proc: psutil.Process,
) -> BenchResult:
    device = torch.device("cuda")
    scale_qk = case.head_dim**-0.5

    q_base = torch.randn((case.batch, heads, case.seq, case.head_dim), device=device, dtype=dtype)
    k_base = torch.randn((case.batch, heads, case.seq, case.head_dim), device=device, dtype=dtype)
    v_base = torch.randn((case.batch, heads, case.seq, case.head_dim), device=device, dtype=dtype)
    bias_base = torch.randn((heads, case.seq, case.seq), device=device, dtype=dtype)
    mask_nw = _make_shift_mask(num_windows=2, seq=case.seq, dtype=dtype, device=device)

    for _ in range(warmup_steps):
        _one_step(backend, q_base, k_base, v_base, bias_base, mask_nw, scale_qk)
    torch.cuda.synchronize()

    timer = Timer(
        stmt="run_once()",
        globals={
            "run_once": lambda: _one_step(backend, q_base, k_base, v_base, bias_base, mask_nw, scale_qk),
        },
    )
    measurement = timer.blocked_autorange(min_run_time=min_run_time)

    gpu_peak_mb, cpu_delta_mb = _measure_memory(backend, q_base, k_base, v_base, bias_base, mask_nw, scale_qk, proc)

    return BenchResult(
        backend=backend,
        case=case.label,
        batch=case.batch,
        seq=case.seq,
        head_dim=case.head_dim,
        heads=heads,
        dtype=str(dtype).replace("torch.", ""),
        time_ms=measurement.median * 1e3,
        time_iqr_ms=measurement.iqr * 1e3,
        gpu_peak_mb=gpu_peak_mb,
        cpu_delta_mb=cpu_delta_mb,
    )


def _save_csv(results: list[BenchResult], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "backend",
        "case",
        "batch",
        "seq",
        "head_dim",
        "heads",
        "dtype",
        "time_ms",
        "time_iqr_ms",
        "gpu_peak_mb",
        "cpu_delta_mb",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: getattr(row, k) for k in fieldnames})


def _format_markdown_table(results: list[BenchResult]) -> str:
    header = "| backend | case | time_ms | time_iqr_ms | gpu_peak_mb | cpu_delta_mb |\n|---|---:|---:|---:|---:|---:|"
    lines = [header]
    for r in results:
        lines.append(
            f"| {r.backend} | {r.case} | {r.time_ms:.3f} | {r.time_iqr_ms:.3f} | {r.gpu_peak_mb:.1f} | {r.cpu_delta_mb:.1f} |"
        )
    return "\n".join(lines)


def _save_plot(results: list[BenchResult], out_png: Path) -> None:
    backends = ["eager", "sdpa", "triton_v1", "triton_v2"]
    cases = sorted({r.case for r in results})

    metrics = [
        ("time_ms", "Time (ms)", "time_ms"),
        ("gpu_peak_mb", "GPU Peak Memory (MB)", "gpu_peak_mb"),
        ("cpu_delta_mb", "CPU RSS Delta (MB)", "cpu_delta_mb"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(15, 11), constrained_layout=True)
    x = list(range(len(cases)))
    width = 0.18

    for ax, (field, title, y_label) in zip(axes, metrics, strict=True):
        for i, backend in enumerate(backends):
            ys: list[float] = []
            for case in cases:
                val = next(getattr(r, field) for r in results if r.backend == backend and r.case == case)
                ys.append(val)
            shift = (i - 1.5) * width
            ax.bar([v + shift for v in x], ys, width=width, label=backend)

        ax.set_title(title)
        ax.set_ylabel(y_label)
        ax.set_xticks(x)
        ax.set_xticklabels(cases, rotation=20)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    axes[0].legend(ncols=4, loc="upper center")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark eager/sdpa/triton(v1/v2) for Flash Swin attention")
    parser.add_argument(
        "--cases",
        type=str,
        default="4x49x32;8x49x32;4x64x32;4x98x32;4x49x64;8x64x64",
        help="Semicolon separated cases with format BxLxD",
    )
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--min-run-time", type=float, default=0.6)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/benchmarks/flash_swin_attn"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    cases = _parse_cases(args.cases)
    backends = ["eager", "sdpa", "triton_v1", "triton_v2"]

    proc = psutil.Process(os.getpid())
    results: list[BenchResult] = []

    for case in cases:
        for backend in backends:
            result = _benchmark_case(
                case=case,
                backend=backend,
                heads=args.heads,
                dtype=dtype,
                min_run_time=args.min_run_time,
                warmup_steps=args.warmup_steps,
                proc=proc,
            )
            results.append(result)
            print(
                f"[bench] {backend:9s} {case.label:12s} "
                f"time={result.time_ms:.3f}ms "
                f"gpu={result.gpu_peak_mb:.1f}MB "
                f"cpu={result.cpu_delta_mb:.1f}MB"
            )

    out_csv = args.out_dir / "flash_swin_attn_benchmark.csv"
    out_md = args.out_dir / "flash_swin_attn_benchmark.md"
    out_png = args.out_dir / "flash_swin_attn_benchmark_bar.png"

    _save_csv(results, out_csv)
    _save_plot(results, out_png)

    md_table = _format_markdown_table(results)
    out_md.write_text(md_table + "\n", encoding="utf-8")

    print("\nSaved:")
    print(f"- CSV: {out_csv}")
    print(f"- Markdown table: {out_md}")
    print(f"- Bar chart: {out_png}")
    print("\n" + md_table)


if __name__ == "__main__":
    main()

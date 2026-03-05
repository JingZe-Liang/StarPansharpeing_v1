from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.models import resnet101  # type: ignore[unresolved-import]

from src.utilities.optim.flashoptim.flashoptim.optimizers import FlashAdamW, cast_model

_MB = 1024**2
_JSON_PREFIX = "__BENCH_JSON__"
_CASE_NAMES = ("torch_adamw_foreach", "flash_adamw_noquant", "flash_adamw_quant")


@dataclass(slots=True)
class BenchConfig:
    batch_size: int = 16
    image_size: int = 224
    warmup_steps: int = 3
    measure_steps: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    seed: int = 3407


@dataclass(slots=True)
class BenchResult:
    name: str
    avg_step_ms: float
    peak_alloc_mb: float
    peak_reserved_mb: float
    final_loss: float


def _sync_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _train_one_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(x)
    loss = F.cross_entropy(logits.float(), y)
    loss.backward()
    optimizer.step()
    return float(loss.detach().float().cpu())


def _build_optimizer(case_name: str, model: torch.nn.Module, cfg: BenchConfig) -> torch.optim.Optimizer:
    if case_name == "torch_adamw_foreach":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
            foreach=True,
        )

    if case_name == "flash_adamw_noquant":
        return FlashAdamW(
            model.parameters(),
            lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
            quantize=False,
            master_weight_bits=None,
            fused=True,
        )

    if case_name == "flash_adamw_quant":
        cast_model(model, dtype=torch.bfloat16, selective=True)
        return FlashAdamW(
            model.parameters(),
            lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
            quantize=True,
            master_weight_bits=24,
            fused=True,
        )

    raise ValueError(f"Unsupported benchmark case: {case_name}")


def _run_single_case(case_name: str, cfg: BenchConfig) -> BenchResult:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    device = torch.device("cuda")
    torch.manual_seed(cfg.seed)

    model = resnet101(num_classes=1000).to(device)
    optimizer = _build_optimizer(case_name, model, cfg)

    torch.manual_seed(cfg.seed + 1)
    x = torch.randn(
        cfg.batch_size,
        3,
        cfg.image_size,
        cfg.image_size,
        device=device,
        dtype=torch.bfloat16,
    )
    y = torch.randint(0, 1000, (cfg.batch_size,), device=device)

    model.train()
    for _ in range(cfg.warmup_steps):
        _train_one_step(model, optimizer, x, y)

    _sync_if_needed()
    torch.cuda.reset_peak_memory_stats(device)

    total_time_ms = 0.0
    final_loss = 0.0
    for _ in range(cfg.measure_steps):
        t0 = time.perf_counter()
        final_loss = _train_one_step(model, optimizer, x, y)
        _sync_if_needed()
        total_time_ms += (time.perf_counter() - t0) * 1000.0

    return BenchResult(
        name=case_name,
        avg_step_ms=total_time_ms / cfg.measure_steps,
        peak_alloc_mb=torch.cuda.max_memory_allocated(device) / _MB,
        peak_reserved_mb=torch.cuda.max_memory_reserved(device) / _MB,
        final_loss=final_loss,
    )


def _format_table(results: list[BenchResult]) -> str:
    headers = ["Optimizer", "Avg Step (ms)", "Peak Alloc (MB)", "Peak Reserved (MB)", "Final Loss"]
    rows = [
        [
            r.name,
            f"{r.avg_step_ms:.3f}",
            f"{r.peak_alloc_mb:.2f}",
            f"{r.peak_reserved_mb:.2f}",
            f"{r.final_loss:.6f}",
        ]
        for r in results
    ]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    def fmt_row(values: list[str]) -> str:
        return "| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(values)) + " |"

    sep = "|-" + "-|-".join("-" * w for w in widths) + "-|"
    lines = [fmt_row(headers), sep]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)


def _cfg_to_cli(cfg: BenchConfig) -> list[str]:
    return [
        "--batch-size",
        str(cfg.batch_size),
        "--image-size",
        str(cfg.image_size),
        "--warmup-steps",
        str(cfg.warmup_steps),
        "--measure-steps",
        str(cfg.measure_steps),
        "--seed",
        str(cfg.seed),
    ]


def _parse_json_result(output: str) -> BenchResult:
    for line in reversed(output.splitlines()):
        if line.startswith(_JSON_PREFIX):
            payload = json.loads(line[len(_JSON_PREFIX) :])
            return BenchResult(**payload)
    raise RuntimeError(f"Missing benchmark JSON marker in subprocess output:\n{output}")


def run_benchmark(cfg: BenchConfig) -> list[BenchResult]:
    script_path = Path(__file__).resolve()
    results: list[BenchResult] = []

    print("---- Start testing ... ----")
    for case_name in _CASE_NAMES:
        print(f"Running {case_name}")
        cmd = [
            sys.executable,
            str(script_path),
            "--single-case",
            case_name,
            *_cfg_to_cli(cfg),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Subprocess failed for {case_name} (exit={proc.returncode})\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        result = _parse_json_result(proc.stdout)
        results.append(result)
        print(f"Finished {case_name}\n")

    return results


def _parse_args() -> tuple[BenchConfig, str | None]:
    parser = argparse.ArgumentParser(description="Benchmark FlashAdamW vs torch AdamW (time + CUDA memory)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--single-case", type=str, choices=list(_CASE_NAMES), default=None)
    args = parser.parse_args()

    cfg = BenchConfig(
        batch_size=args.batch_size,
        image_size=args.image_size,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        seed=args.seed,
    )
    return cfg, args.single_case


def main() -> None:
    cfg, single_case = _parse_args()
    if single_case is not None:
        result = _run_single_case(single_case, cfg)
        print(f"{_JSON_PREFIX}{json.dumps(asdict(result))}")
        return

    results = run_benchmark(cfg)
    print(_format_table(results))


if __name__ == "__main__":
    main()

"""
Generate and save histograms for different latent noise t sampling distributions.

The script samples interpolation factors t using `_sample_t_distributional` for several
`latent_noise_type` configuration strings and plots their empirical distributions.

Usage
-----
python test_t_sampling.py

Output
------
Saves histogram figure to `tmp/t_sampling_hist.png` and prints summary stats.

Note
----
No try/except blocks (project convention). Ensure `matplotlib` is installed.
Install if needed:
    pip install matplotlib
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.stage1.cosmos.cosmos_tokenizer import _sample_t_distributional


@dataclass
class SamplingConfig:
    name: str
    noise_type_cfg: str


def _upper_bound_from_cfg(cfg: str) -> float:
    """Infer an expected upper bound for t for axis scaling.

    Parameters
    ----------
    cfg : str
        The latent noise configuration string.
    """
    if cfg.startswith("uniform_max_"):
        return float(cfg.split("_")[-1])
    if cfg.startswith("exp_max_"):
        parts = cfg.split("_")
        return float(parts[2])
    # beta and default normal are clamped to [0,1]
    return 1.0


def _sample(cfg: SamplingConfig, sample_size: int, device: str) -> torch.Tensor:
    """Sample t values for a single configuration.

    Parameters
    ----------
    cfg : SamplingConfig
        Sampling configuration wrapper.
    sample_size : int
        Number of samples.
    device : str
        Device string ("cuda" or "cpu").
    """
    t = _sample_t_distributional(bs=sample_size, device=device, noise_type_cfg=cfg.noise_type_cfg)
    return t.flatten()


def _plot_hist(ax, data: torch.Tensor, title: str, upper: float) -> None:
    """Plot histogram on provided axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.
    data : torch.Tensor
        1D tensor of samples.
    title : str
        Title for subplot.
    upper : float
        Upper bound for x-axis.
    """
    bins = 100
    ax.hist(
        data.cpu().numpy(),
        bins=bins,
        range=(0.0, upper),
        density=True,
        color="#1f77b4",
        alpha=0.75,
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0.0, upper)
    ax.grid(alpha=0.25)


def _describe(data: torch.Tensor) -> dict[str, float]:
    """Compute summary statistics for sampled t."""
    q25, q50, q75 = torch.quantile(data, torch.tensor([0.25, 0.5, 0.75], device=data.device)).tolist()
    return {
        "mean": float(data.mean()),
        "std": float(data.std(unbiased=False)),
        "min": float(data.min()),
        "max": float(data.max()),
        "q25": q25,
        "q50": q50,
        "q75": q75,
    }


def generate_and_plot(
    noise_cfgs: list[str],
    sample_size: int = 100_000,
    out_path: str = "tmp/t_sampling_hist.png",
) -> None:
    """Generate histograms for multiple latent noise type configurations.

    Parameters
    ----------
    noise_cfgs : list[str]
        List of latent noise configuration strings.
    sample_size : int, default 100_000
        Number of samples per distribution.
    out_path : str, default "tmp/t_sampling_hist.png"
        Output figure path.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfgs = [SamplingConfig(name=c, noise_type_cfg=c) for c in noise_cfgs]

    n = len(cfgs)
    cols = 3
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 2.8), dpi=140)
    axes = axes.flatten()

    stats_table: list[tuple[str, dict[str, float]]] = []
    for i, cfg in enumerate(cfgs):
        data = _sample(cfg, sample_size=sample_size, device=device)
        upper = _upper_bound_from_cfg(cfg.noise_type_cfg)
        _plot_hist(axes[i], data, cfg.name, upper)
        stats = _describe(data)
        stats_table.append((cfg.name, stats))

    # Remove unused axes
    for j in range(len(cfgs), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Latent t Sampling Distributions", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p)

    print(f"Saved histogram figure to: {out_p}")
    print("Summary stats (mean,std,min,max,q25,q50,q75):")
    for name, stats in stats_table:
        print(
            f"{name:>25}: "
            f"mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
            f"min={stats['min']:.4f}, max={stats['max']:.4f}, "
            f"q25={stats['q25']:.4f}, q50={stats['q50']:.4f}, q75={stats['q75']:.4f}"
        )


if __name__ == "__main__":
    noise_cfgs = [
        "uniform_max_0.3",
        "uniform_max_0.6",
        "exp_0.2",  # backward compatibility mapping
        "exp_max_0.3_lambda_3",
        "exp_max_0.3_lambda_8",
        "beta_1_8",
        "beta_2_5",
        "beta_1_20",
        "normal",  # default fallback
    ]
    generate_and_plot(noise_cfgs=noise_cfgs, sample_size=100_000, out_path="tmp/t_sampling_hist.png")

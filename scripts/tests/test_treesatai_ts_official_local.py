from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf
import pytest

from src.stage2.classification.data.TreeSatAI_ts_official import DEFAULT_CLASSES, TreeSAT, collate_fn


def _identity_transform(sample: dict[str, object]) -> dict[str, object]:
    return sample


def _modalities_from_yaml_sensors(sensors: list[str]) -> list[str]:
    return [sensor.replace("_", "-") for sensor in sensors]


def _load_local_dataset(monkeypatch: pytest.MonkeyPatch | None = None) -> TreeSAT:
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = repo_root / "scripts/configs/classification/dataset/treesatai_ts.yaml"
    cfg = OmegaConf.load(cfg_path)

    root = repo_root / str(cfg.root)
    assert root.exists(), f"Local dataset root not found: {root}"

    modalities = _modalities_from_yaml_sensors(list(cfg.sensors))

    if monkeypatch is not None:

        def _fail_if_torch_load_called(*args, **kwargs):
            raise AssertionError("torch.load should not be used in __getitem__ for S1 processing")

        monkeypatch.setattr(torch, "load", _fail_if_torch_load_called)

    return TreeSAT(
        path=root,
        modalities=modalities,
        transform=_identity_transform,
        split="train",
        classes=DEFAULT_CLASSES,
        partition=0.999,
    )


def _to_display_rgb(image_chw: torch.Tensor, rgb_indices: tuple[int, int, int] | None = None) -> torch.Tensor:
    if image_chw.ndim != 3:
        raise ValueError(f"Expected CHW image, got {tuple(image_chw.shape)}")

    channels = int(image_chw.shape[0])
    if channels == 1:
        rgb = image_chw.repeat(3, 1, 1)
    elif channels >= 3:
        if rgb_indices is not None and max(rgb_indices) < channels:
            rgb = image_chw[list(rgb_indices)]
        else:
            rgb = image_chw[:3]
    else:
        pad = image_chw.new_zeros((3 - channels, *image_chw.shape[1:]))
        rgb = torch.cat([image_chw, pad], dim=0)

    rgb = rgb.to(torch.float32)
    min_v = rgb.amin()
    max_v = rgb.amax()
    rgb = (rgb - min_v) / (max_v - min_v).clamp_min(1e-6)
    return rgb.permute(1, 2, 0).cpu()


def _plot_treesat_sample(
    sample: dict[str, object],
    save_path: Path,
    max_s1_frames: int = 2,
    max_s2_frames: int = 2,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: list[tuple[str, list[torch.Tensor]]] = []

    aerial = sample.get("aerial")
    if isinstance(aerial, torch.Tensor) and aerial.ndim == 3:
        rows.append(("aerial", [aerial]))

    s1_asc = sample.get("s1-asc")
    if isinstance(s1_asc, torch.Tensor) and s1_asc.ndim == 4:
        n_s1 = min(max_s1_frames, int(s1_asc.shape[0]))
        rows.append(("s1-asc", [s1_asc[idx] for idx in range(n_s1)]))

    s2 = sample.get("s2")
    if isinstance(s2, torch.Tensor) and s2.ndim == 4:
        n_s2 = min(max_s2_frames, int(s2.shape[0]))
        rows.append(("s2", [s2[idx] for idx in range(n_s2)]))

    if not rows:
        raise ValueError("No plottable modality found in sample")

    cols = max(len(frames) for _, frames in rows)
    fig, axes = plt.subplots(len(rows), cols, figsize=(2.4 * cols, 2.4 * len(rows)), squeeze=False)

    for row_idx, (modality, frames) in enumerate(rows):
        for col_idx in range(cols):
            ax = axes[row_idx][col_idx]
            if col_idx >= len(frames):
                ax.axis("off")
                continue

            frame = frames[col_idx]
            if modality == "s2":
                rgb = _to_display_rgb(frame, rgb_indices=(3, 2, 1))
            else:
                rgb = _to_display_rgb(frame)
            ax.imshow(rgb.numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_title(modality, fontsize=9)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def test_treesatai_ts_official_getitem_uses_h5_not_pth(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _load_local_dataset(monkeypatch=monkeypatch)

    assert len(dataset) > 0

    sample = dataset[0]
    assert "aerial" in sample
    assert "s1-asc" in sample
    assert "s2" in sample

    s1_asc = sample["s1-asc"]
    assert isinstance(s1_asc, torch.Tensor)
    assert s1_asc.ndim == 4
    assert s1_asc.shape[1] == 2
    assert s1_asc.shape[0] <= 50
    assert not torch.isnan(s1_asc).any()

    sample_2 = dataset[min(1, len(dataset) - 1)]
    batch = collate_fn([sample, sample_2])
    assert "s1-asc" in batch
    assert isinstance(batch["s1-asc"], torch.Tensor)
    assert batch["s1-asc"].ndim == 5


def test_plot_treesat_ts() -> None:
    dataset = _load_local_dataset()
    sample = dataset[0]

    out_path = Path("treesat_ts_sample.png")
    _plot_treesat_sample(sample, out_path, max_s1_frames=2, max_s2_frames=2)

    assert out_path.exists()
    assert out_path.stat().st_size > 0

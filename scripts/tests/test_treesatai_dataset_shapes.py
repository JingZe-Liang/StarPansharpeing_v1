from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.stage2.classification.data.TreeSatAI_torchgeo import TreeSatAIMultiLabel


def _parse_sensors(text: str) -> tuple[str, ...]:
    sensors = tuple(part.strip() for part in text.split(",") if part.strip())
    if not sensors:
        raise ValueError("sensors must not be empty")
    return sensors


def _tensor_summary(tensor: Tensor) -> str:
    if tensor.numel() == 0:
        return f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, empty"
    min_value = float(tensor.min())
    max_value = float(tensor.max())
    return f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, min={min_value:.4f}, max={max_value:.4f}"


def _print_mapping(name: str, data: dict[str, Any]) -> None:
    print(f"\n{name}:")
    for key in sorted(data.keys()):
        value = data[key]
        if isinstance(value, Tensor):
            print(f"  {key}: {_tensor_summary(value)}")
        else:
            print(f"  {key}: type={type(value).__name__}")


def _split_time_frames(image: Tensor) -> list[Tensor]:
    if image.ndim == 3:
        return [image]
    if image.ndim == 4:
        return [image[index] for index in range(image.shape[0])]
    raise ValueError(f"Unsupported image ndim: {image.ndim}, expected 3 or 4.")


def _sensor_rgb_indices(sensor: str, channels: int) -> list[int]:
    if sensor == "aerial" and channels >= 4:
        return [3, 1, 2]
    if sensor == "s2" and channels >= 3:
        return [2, 1, 0]
    if channels >= 3:
        return [0, 1, 2]
    if channels == 2:
        return [0, 1, 1]
    if channels == 1:
        return [0, 0, 0]
    raise ValueError(f"Invalid channel size: {channels}")


def _to_display_image(image: Tensor, sensor: str) -> Tensor:
    if image.ndim != 3:
        raise ValueError(f"Expected (C,H,W), got shape={tuple(image.shape)}")
    image = image.detach().float().cpu()
    rgb_indices = _sensor_rgb_indices(sensor=sensor, channels=int(image.shape[0]))
    rgb = image[rgb_indices]
    if rgb.numel() == 0:
        return torch.zeros((1, 1, 3), dtype=torch.float32)

    q_low = torch.quantile(rgb.reshape(-1), 0.02)
    q_high = torch.quantile(rgb.reshape(-1), 0.98)
    rgb = (rgb - q_low) / (q_high - q_low).clamp_min(1e-6)
    rgb = rgb.clamp(0.0, 1.0)
    return rgb.permute(1, 2, 0)


def _save_paired_plots_by_time(
    *,
    sample: dict[str, Any],
    sensors: Sequence[str],
    out_dir: Path,
    sample_index: int,
    max_time_steps: int,
) -> list[Path]:
    sensor_frames: dict[str, list[Tensor]] = {}
    for sensor in sensors:
        key = f"image_{sensor}"
        value = sample.get(key)
        if not isinstance(value, Tensor):
            continue
        sensor_frames[sensor] = _split_time_frames(value)

    if not sensor_frames:
        return []

    total_time = max(len(frames) for frames in sensor_frames.values())
    total_time = min(total_time, max_time_steps) if max_time_steps > 0 else total_time

    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for time_idx in range(total_time):
        fig, axes = plt.subplots(1, len(sensors), figsize=(5 * len(sensors), 5), squeeze=False)
        for col_idx, sensor in enumerate(sensors):
            axis = axes[0, col_idx]
            frames = sensor_frames.get(sensor, [])
            if time_idx >= len(frames):
                axis.axis("off")
                axis.set_title(f"{sensor} t={time_idx} (missing)")
                continue
            display = _to_display_image(frames[time_idx], sensor=sensor)
            axis.imshow(display.numpy())
            axis.axis("off")
            axis.set_title(f"{sensor} t={time_idx}")

        fig.tight_layout()
        save_path = out_dir / f"treesatai_sample_{sample_index:05d}_t{time_idx:03d}.png"
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(save_path)
    return saved_paths


def inspect_treesatai(
    *,
    root: str,
    split: str,
    sensors: Sequence[str],
    image_size: int,
    batch_size: int,
    num_workers: int,
    download: bool,
    checksum: bool,
    normalize: bool,
    to_neg_1_1: bool,
    val_ratio: float,
    val_seed: int,
    sample_index: int,
    save_pairs: bool,
    out_dir: str,
    max_time_steps: int,
) -> None:
    dataset = TreeSatAIMultiLabel(
        root=root,
        split=split,
        sensors=tuple(sensors),
        image_size=image_size,
        normalize=normalize,
        to_neg_1_1=to_neg_1_1,
        val_ratio=val_ratio,
        val_seed=val_seed,
        download=download,
        checksum=checksum,
    )

    print("TreeSatAI dataset ready")
    print(f"  split: {split}")
    print(f"  root: {root}")
    print(f"  sensors: {tuple(sensors)}")
    print(f"  dataset_len: {len(dataset)}")
    print(f"  num_classes: {len(dataset.classes)}")

    sample = dataset[sample_index]
    _print_mapping("Single sample", sample)

    if save_pairs:
        saved_paths = _save_paired_plots_by_time(
            sample=sample,
            sensors=sensors,
            out_dir=Path(out_dir),
            sample_index=sample_index,
            max_time_steps=max_time_steps,
        )
        if saved_paths:
            print(f"\nSaved {len(saved_paths)} paired time-step figure(s):")
            for path in saved_paths:
                print(f"  {path}")
        else:
            print("\nNo image tensor found for pair plotting.")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    batch = next(iter(dataloader))
    if isinstance(batch, dict):
        _print_mapping("One batch", batch)
    else:
        print(f"\nOne batch type: {type(batch).__name__}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print TreeSatAI dataset keys and shapes.")
    parser.add_argument("--root", type=str, default="data/Downstreams/TreeSatAI")
    parser.add_argument("--split", type=str, default="train", choices=("train", "val", "test"))
    parser.add_argument("--sensors", type=str, default="aerial,s1,s2")
    parser.add_argument("--image-size", type=int, default=304)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--checksum", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--to-neg-1-1", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--val-seed", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--save-pairs", action="store_true")
    parser.add_argument("--out-dir", type=str, default="tmp/treesatai_pairs")
    parser.add_argument("--max-time-steps", type=int, default=0)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    sensors = _parse_sensors(args.sensors)
    normalize = args.normalize
    if not args.normalize and not args.to_neg_1_1:
        normalize = True

    inspect_treesatai(
        root=args.root,
        split=args.split,
        sensors=sensors,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
        checksum=args.checksum,
        normalize=normalize,
        to_neg_1_1=args.to_neg_1_1,
        val_ratio=args.val_ratio,
        val_seed=args.val_seed,
        sample_index=args.sample_index,
        save_pairs=args.save_pairs,
        out_dir=args.out_dir,
        max_time_steps=args.max_time_steps,
    )


if __name__ == "__main__":
    main()

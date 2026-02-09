from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Sample:
    image_rel: str
    mask_rel: str
    image_id: str
    class_counts: np.ndarray
    valid_pixels: int


def _load_val_images(val_list_path: Path) -> set[str]:
    if not val_list_path.exists():
        return set()
    images: set[str] = set()
    for raw in val_list_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 1:
            images.add(parts[0])
    return images


def _image_rel_from_mask_name(mask_name: str) -> str:
    if "_lab_" not in mask_name:
        raise ValueError(f"Unexpected mask name: {mask_name}")
    image_name = mask_name.replace("_lab_", "_").removesuffix(".png") + ".jpg"
    return f"Images/{image_name}"


def _mask_rel_from_name(mask_name: str) -> str:
    return f"Semantic_mask/{mask_name}"


def _class_counts(mask_path: Path, *, num_classes: int) -> np.ndarray:
    mask = np.asarray(Image.open(mask_path).convert("L"))
    counts = np.bincount(mask.reshape(-1), minlength=256)[:num_classes]
    return counts.astype(np.int64, copy=False)


def build_samples(
    *,
    root: Path,
    val_list: Path,
    num_classes: int,
) -> list[Sample]:
    val_images = _load_val_images(val_list)
    mask_dir = root / "Semantic_mask"
    if not mask_dir.exists():
        raise FileNotFoundError(f"Missing folder: {mask_dir}")

    out: list[Sample] = []
    for mask_path in sorted(mask_dir.glob("*.png")):
        if "_lab_" not in mask_path.name:
            continue
        image_rel = _image_rel_from_mask_name(mask_path.name)
        if image_rel in val_images:
            continue
        if not (root / image_rel).exists():
            continue
        counts = _class_counts(mask_path, num_classes=num_classes)
        valid_pixels = int(counts.sum())
        if valid_pixels <= 0:
            continue
        out.append(
            Sample(
                image_rel=image_rel,
                mask_rel=_mask_rel_from_name(mask_path.name),
                image_id=Path(image_rel).stem.split("_", maxsplit=1)[0],
                class_counts=counts,
                valid_pixels=valid_pixels,
            )
        )
    return out


def _select_greedy(
    *,
    candidates: list[Sample],
    class_id: int,
    k: int,
    unique_image_id: bool,
) -> list[Sample]:
    ranked = sorted(
        candidates,
        key=lambda s: (
            int(s.class_counts[class_id]),
            s.valid_pixels,
            s.image_rel,
        ),
        reverse=True,
    )
    selected: list[Sample] = []
    used_ids: set[str] = set()
    for sample in ranked:
        if int(sample.class_counts[class_id]) <= 0:
            continue
        if unique_image_id and sample.image_id in used_ids:
            continue
        selected.append(sample)
        used_ids.add(sample.image_id)
        if len(selected) == k:
            break
    return selected


def _weighted_pick_index(rng: np.random.Generator, weights: np.ndarray) -> int:
    total = float(weights.sum())
    probs = weights.astype(np.float64) / total
    return int(rng.choice(np.arange(weights.size), p=probs))


def _select_random_weighted(
    *,
    candidates: list[Sample],
    class_id: int,
    k: int,
    unique_image_id: bool,
    seed: int,
) -> list[Sample]:
    positive = [s for s in candidates if int(s.class_counts[class_id]) > 0]
    rng = np.random.default_rng(seed + class_id)
    selected: list[Sample] = []
    used_ids: set[str] = set()
    remaining = positive[:]

    while len(selected) < k and remaining:
        pool = [s for s in remaining if (not unique_image_id or s.image_id not in used_ids)]
        if not pool:
            break
        weights = np.array([int(s.class_counts[class_id]) for s in pool], dtype=np.float64)
        pick = pool[_weighted_pick_index(rng, weights)]
        selected.append(pick)
        used_ids.add(pick.image_id)
        remaining = [s for s in remaining if s.image_rel != pick.image_rel]

    return selected


def select_for_class(
    *,
    candidates: list[Sample],
    class_id: int,
    k: int,
    strategy: str,
    unique_image_id: bool,
    seed: int,
) -> list[Sample]:
    if strategy == "greedy":
        selected = _select_greedy(
            candidates=candidates,
            class_id=class_id,
            k=k,
            unique_image_id=unique_image_id,
        )
    else:
        selected = _select_random_weighted(
            candidates=candidates,
            class_id=class_id,
            k=k,
            unique_image_id=unique_image_id,
            seed=seed,
        )

    if len(selected) < k:
        raise RuntimeError(
            f"class={class_id}: only selected {len(selected)} samples, expected {k}. Relax constraints or reduce k."
        )
    return selected


def write_support_file(*, path: Path, samples: list[Sample]) -> None:
    lines = [f"{s.image_rel} {s.mask_rel}" for s in samples]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    # fmt: off
    p = argparse.ArgumentParser(description="Select per-class support sets for Flood-3i.")
    p.add_argument("--root", type=Path, default=Path("data/Downstreams/Flood-3i"))
    p.add_argument("--val-list", type=Path, default=Path("data/Downstreams/Flood-3i/val.txt"))
    p.add_argument("--out-dir", type=Path, default=Path("."))
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--k", type=int, default=5, help="Support shots per class, e.g. 1 or 5.")
    p.add_argument("--strategy", choices=["greedy", "random"], default="greedy")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--classes", type=int, nargs="*", default=None, help="Optional class ids. Default: all [0..num_classes-1].")
    p.add_argument("--unique-image-id", action=argparse.BooleanOptionalAction, default=True)
    # fmt: on
    return p.parse_args()


def main() -> None:
    args = parse_args()
    classes = args.classes if args.classes is not None else list(range(args.num_classes))
    if any(c < 0 or c >= args.num_classes for c in classes):
        raise ValueError(f"Class ids must be in [0, {args.num_classes - 1}]")

    samples = build_samples(
        root=args.root,
        val_list=args.val_list,
        num_classes=args.num_classes,
    )
    if not samples:
        raise RuntimeError("No usable samples found.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"strategy: {args.strategy} | k: {args.k} | candidates: {len(samples)}")
    for class_id in classes:
        selected = select_for_class(
            candidates=samples,
            class_id=class_id,
            k=args.k,
            strategy=args.strategy,
            unique_image_id=args.unique_image_id,
            seed=args.seed,
        )
        out_file = args.out_dir / f"support_class_{class_id}_{args.k}.txt"
        write_support_file(path=out_file, samples=selected)
        top_pixels = int(selected[0].class_counts[class_id])
        print(f"class={class_id} -> {out_file} ({len(selected)} lines, top_pixels={top_pixels})")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Candidate:
    image_rel: str
    mask_rel: str
    image_id: str
    present_class_ids: frozenset[int]
    present_classes: int
    effective_classes: float
    entropy: float
    min_present_fraction: float
    base_score: float


def _load_val_images(val_list_path: Path) -> set[str]:
    images: set[str] = set()
    for raw in val_list_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 1:
            continue
        images.add(parts[0])
    return images


def _image_rel_from_mask_name(mask_name: str) -> str:
    # Flood-3i mask naming convention: <id>_lab_<row>_<col>.png -> <id>_<row>_<col>.jpg
    if "_lab_" not in mask_name:
        raise ValueError(f"Unexpected mask name: {mask_name}")
    image_name = mask_name.replace("_lab_", "_").removesuffix(".png") + ".jpg"
    return f"Images/{image_name}"


def _mask_rel(mask_name: str) -> str:
    return f"Semantic_mask/{mask_name}"


def _class_histogram(mask_path: Path, *, num_classes: int, ignore_index: int) -> tuple[np.ndarray, int]:
    mask = np.asarray(Image.open(mask_path).convert("L"))
    bincount = np.bincount(mask.reshape(-1), minlength=256)
    class_counts = bincount[:num_classes].astype(np.int64, copy=False)
    ignore = int(bincount[ignore_index]) if 0 <= ignore_index < bincount.size else 0
    return class_counts, ignore


def _entropy_from_counts(counts: np.ndarray) -> float:
    total = int(counts.sum())
    if total <= 0:
        return 0.0
    p = counts.astype(np.float64) / float(total)
    p = p[p > 0]
    # Use natural log; effective_classes = exp(entropy).
    return float(-(p * np.log(p)).sum())


def _candidate_from_mask(
    mask_path: Path,
    *,
    root: Path,
    num_classes: int,
    ignore_index: int,
    min_class_fraction: float,
) -> Candidate | None:
    mask_name = mask_path.name
    image_rel = _image_rel_from_mask_name(mask_name)
    mask_rel = _mask_rel(mask_name)
    image_id = Path(image_rel).stem.split("_", maxsplit=1)[0]

    class_counts, ignore = _class_histogram(mask_path, num_classes=num_classes, ignore_index=ignore_index)
    total = int(class_counts.sum())
    if total <= 0:
        return None

    present = class_counts > 0
    present_class_ids = frozenset(int(i) for i in np.nonzero(present)[0].tolist())
    present_classes = int(present.sum())
    entropy = _entropy_from_counts(class_counts)
    effective = float(math.exp(entropy))

    fractions = class_counts[present].astype(np.float64) / float(total)
    min_present_fraction = float(fractions.min()) if fractions.size else 0.0
    if min_present_fraction < min_class_fraction:
        return None

    # Base score favors many classes with balanced pixels (effective_classes already captures both).
    base_score = effective + 0.05 * float(present_classes)

    # Ensure pair exists (image is only used for loader; keep validation strict here).
    if not (root / image_rel).exists():
        return None

    _ = ignore  # Keep for potential future extensions; no-op now.
    return Candidate(
        image_rel=image_rel,
        mask_rel=mask_rel,
        image_id=image_id,
        present_class_ids=present_class_ids,
        present_classes=present_classes,
        effective_classes=effective,
        entropy=entropy,
        min_present_fraction=min_present_fraction,
        base_score=base_score,
    )


def build_candidates(
    *,
    root: Path,
    val_list_path: Path,
    num_classes: int,
    ignore_index: int,
    min_class_fraction: float,
) -> list[Candidate]:
    val_images = _load_val_images(val_list_path)
    mask_dir = root / "Semantic_mask"
    if not mask_dir.exists():
        raise FileNotFoundError(f"Missing folder: {mask_dir}")

    candidates: list[Candidate] = []
    for mask_path in sorted(mask_dir.glob("*.png")):
        if "_lab_" not in mask_path.name:
            continue
        image_rel = _image_rel_from_mask_name(mask_path.name)
        if image_rel in val_images:
            continue
        cand = _candidate_from_mask(
            mask_path,
            root=root,
            num_classes=num_classes,
            ignore_index=ignore_index,
            min_class_fraction=min_class_fraction,
        )
        if cand is not None:
            candidates.append(cand)
    return candidates


def greedy_select(
    candidates: list[Candidate],
    *,
    k: int,
    coverage_weight: float,
    unique_image_id: bool,
) -> list[Candidate]:
    # Greedy set selection: base_score + coverage gain (new classes added into the set).
    selected: list[Candidate] = []
    covered: set[int] = set()
    used_image_ids: set[str] = set()
    remaining = candidates[:]

    while len(selected) < k and remaining:
        best_idx = 0
        best_score = -1.0
        for i, cand in enumerate(remaining):
            if unique_image_id and cand.image_id in used_image_ids:
                continue
            new_classes = cand.present_class_ids.difference(covered)
            score = cand.base_score + coverage_weight * float(len(new_classes))
            if score > best_score:
                best_idx = i
                best_score = score

        if best_score < 0:
            raise RuntimeError("Cannot satisfy uniqueness constraint; relax --unique-image-id.")

        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        covered.update(chosen.present_class_ids)
        used_image_ids.add(chosen.image_id)

    return selected


def write_list(path: Path, selected: list[Candidate]) -> None:
    lines = [f"{c.image_rel} {c.mask_rel}" for c in selected]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    # fmt: off
    p = argparse.ArgumentParser(description="Select n-shot samples for Flood-3i segmentation.")
    p.add_argument("--root", type=Path, default=Path("data/Downstreams/Flood-3i"))
    p.add_argument("--val-list", type=Path, default=Path("data/Downstreams/Flood-3i/val.txt"))
    p.add_argument("--out", type=Path, default=Path("train_5shots.txt"))
    p.add_argument("--num-shots", type=int, default=5)
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--ignore-index", type=int, default=255)
    p.add_argument("--min-class-fraction", type=float, default=0.01, help="Minimum fraction among present classes in one image (0-1).")
    p.add_argument("--coverage-weight", type=float, default=0.2, help="Weight for greedy class-coverage gain (heuristic).")
    p.add_argument("--unique-image-id", action=argparse.BooleanOptionalAction, default=True, help="If true, do not pick multiple tiles from the same image id prefix.")
    # fmt: on
    return p.parse_args()


def main() -> None:
    args = parse_args()
    candidates = build_candidates(
        root=args.root,
        val_list_path=args.val_list,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        min_class_fraction=args.min_class_fraction,
    )
    if not candidates:
        raise RuntimeError("No candidates found. Check paths and thresholds.")

    selected = greedy_select(
        candidates,
        k=args.num_shots,
        coverage_weight=args.coverage_weight,
        unique_image_id=args.unique_image_id,
    )
    if len(selected) < args.num_shots:
        raise RuntimeError(f"Only selected {len(selected)} samples, expected {args.num_shots}.")

    write_list(args.out, selected)

    # Print a compact report for reproducibility.
    print(f"candidates: {len(candidates)}")
    print(f"selected: {len(selected)} -> {args.out}")
    for i, c in enumerate(selected):
        print(
            f"[{i}] {c.image_rel} | present={c.present_classes} "
            f"effective={c.effective_classes:.2f} min_frac={c.min_present_fraction:.3f}"
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import math
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def start_points(size: int, split_size: int, overlap: float = 0.0) -> list[int]:
    if size <= split_size:
        return [0]
    points = [0]
    stride = max(1, int(split_size * (1 - overlap)))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        points.append(pt)
        counter += 1
    return points


def list_sites(input_root: Path) -> list[Path]:
    return sorted([p for p in input_root.iterdir() if p.is_dir()])


def load_label(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def is_positive_patch(label_patch: np.ndarray, min_pos_ratio: float) -> bool:
    nonzero = np.count_nonzero(label_patch)
    if nonzero == 0:
        return False
    ratio = nonzero / label_patch.size
    return ratio >= min_pos_ratio


def count_pos_neg_for_site(
    site: Path,
    patch_size: int,
    overlap: float,
    min_pos_ratio: float,
) -> tuple[int, int]:
    pos_count = 0
    neg_count = 0
    label = load_label(site / "ref.png")
    height, width = label.shape[:2]
    y_points = start_points(height, patch_size, overlap)
    x_points = start_points(width, patch_size, overlap)
    for y in y_points:
        for x in x_points:
            label_patch = label[y : y + patch_size, x : x + patch_size]
            if is_positive_patch(label_patch, min_pos_ratio):
                pos_count += 1
            else:
                neg_count += 1
    return pos_count, neg_count


def count_pos_neg(
    sites: list[Path],
    patch_size: int,
    overlap: float,
    min_pos_ratio: float,
) -> tuple[int, int]:
    pos_total = 0
    neg_total = 0
    for site in tqdm(sites, desc="count patches", unit="site"):
        pos_count, neg_count = count_pos_neg_for_site(site, patch_size, overlap, min_pos_ratio)
        pos_total += pos_count
        neg_total += neg_count
    return pos_total, neg_total


def clear_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_file():
            item.unlink()


def save_patch(
    out_a: Path,
    out_b: Path,
    out_label: Path,
    index: int,
    img1: np.ndarray,
    img2: np.ndarray,
    label: np.ndarray,
    img_ext: str,
    label_ext: str,
    jpg_quality: int,
) -> None:
    img1_path = out_a / f"{index}.{img_ext}"
    img2_path = out_b / f"{index}.{img_ext}"
    label_path = out_label / f"{index}.{label_ext}"
    Image.fromarray(img1).save(img1_path, quality=jpg_quality)
    Image.fromarray(img2).save(img2_path, quality=jpg_quality)
    Image.fromarray(label).save(label_path)


def allocate_neg_targets(site_neg_counts: list[int], target_neg: int) -> list[int]:
    if target_neg <= 0:
        return [0 for _ in site_neg_counts]
    total_neg = sum(site_neg_counts)
    if total_neg == 0:
        return [0 for _ in site_neg_counts]
    raw_targets = [target_neg * (count / total_neg) for count in site_neg_counts]
    base_targets = [int(math.floor(value)) for value in raw_targets]
    remainder = target_neg - sum(base_targets)
    fractions = [value - base for value, base in zip(raw_targets, base_targets, strict=False)]
    order = sorted(range(len(fractions)), key=lambda i: fractions[i], reverse=True)
    for i in range(remainder):
        base_targets[order[i]] += 1
    return base_targets


def save_site_patches(
    site: Path,
    out_a: Path,
    out_b: Path,
    out_label: Path,
    patch_size: int,
    overlap: float,
    min_pos_ratio: float,
    target_neg: int,
    seed: int,
    img_ext: str,
    label_ext: str,
    jpg_quality: int,
    neg_total: int,
    base_index: int,
) -> tuple[int, int, int]:
    img1 = np.asarray(Image.open(site / "im1.png"))
    img2 = np.asarray(Image.open(site / "im2.png"))
    label = load_label(site / "ref.png")
    height, width = label.shape[:2]
    y_points = start_points(height, patch_size, overlap)
    x_points = start_points(width, patch_size, overlap)
    rng = random.Random(seed)

    neg_seen = 0
    neg_kept = 0
    saved = 0
    for y in y_points:
        for x in x_points:
            label_patch = label[y : y + patch_size, x : x + patch_size]
            if is_positive_patch(label_patch, min_pos_ratio):
                save_patch(
                    out_a,
                    out_b,
                    out_label,
                    base_index + saved,
                    img1[y : y + patch_size, x : x + patch_size],
                    img2[y : y + patch_size, x : x + patch_size],
                    label_patch,
                    img_ext,
                    label_ext,
                    jpg_quality,
                )
                saved += 1
                continue

            if target_neg == 0:
                neg_seen += 1
                continue

            remaining = neg_total - neg_seen
            need = target_neg - neg_kept
            keep_prob = need / remaining if remaining > 0 else 0.0
            if rng.random() < keep_prob:
                save_patch(
                    out_a,
                    out_b,
                    out_label,
                    base_index + saved,
                    img1[y : y + patch_size, x : x + patch_size],
                    img2[y : y + patch_size, x : x + patch_size],
                    label_patch,
                    img_ext,
                    label_ext,
                    jpg_quality,
                )
                saved += 1
                neg_kept += 1
            neg_seen += 1
    return saved, neg_kept, neg_seen


def sample_and_save(
    sites: list[Path],
    output_root: Path,
    patch_size: int,
    overlap: float,
    min_pos_ratio: float,
    neg_to_pos: float,
    seed: int,
    img_ext: str,
    label_ext: str,
    jpg_quality: int,
    clear_output: bool,
    workers: int,
) -> None:
    out_a = output_root / "A"
    out_b = output_root / "B"
    out_label = output_root / "label"
    if clear_output:
        clear_dir(out_a)
        clear_dir(out_b)
        clear_dir(out_label)
    else:
        out_a.mkdir(parents=True, exist_ok=True)
        out_b.mkdir(parents=True, exist_ok=True)
        out_label.mkdir(parents=True, exist_ok=True)

    site_counts: list[tuple[Path, int, int]] = []
    pos_total = 0
    neg_total = 0
    for site in tqdm(sites, desc="count patches", unit="site"):
        pos_count, neg_count = count_pos_neg_for_site(site, patch_size, overlap, min_pos_ratio)
        site_counts.append((site, pos_count, neg_count))
        pos_total += pos_count
        neg_total += neg_count

    target_neg = min(neg_total, int(round(pos_total * neg_to_pos)))
    target_neg_per_site = allocate_neg_targets([item[2] for item in site_counts], target_neg)
    base_offsets: list[int] = []
    running = 0
    for (_, pos_count, _), neg_target in zip(site_counts, target_neg_per_site, strict=False):
        base_offsets.append(running)
        running += pos_count + neg_target

    saved = 0
    neg_kept = 0
    neg_seen = 0
    if workers <= 1:
        for (site, _, neg_count), neg_target, base_index in tqdm(
            zip(site_counts, target_neg_per_site, base_offsets, strict=False),
            desc="save patches",
            unit="site",
            total=len(site_counts),
        ):
            site_saved, site_neg_kept, site_neg_seen = save_site_patches(
                site=site,
                out_a=out_a,
                out_b=out_b,
                out_label=out_label,
                patch_size=patch_size,
                overlap=overlap,
                min_pos_ratio=min_pos_ratio,
                target_neg=neg_target,
                seed=seed + base_index,
                img_ext=img_ext,
                label_ext=label_ext,
                jpg_quality=jpg_quality,
                neg_total=neg_count,
                base_index=base_index,
            )
            saved += site_saved
            neg_kept += site_neg_kept
            neg_seen += site_neg_seen
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = []
            for (site, _, neg_count), neg_target, base_index in zip(
                site_counts, target_neg_per_site, base_offsets, strict=False
            ):
                futures.append(
                    executor.submit(
                        save_site_patches,
                        site,
                        out_a,
                        out_b,
                        out_label,
                        patch_size,
                        overlap,
                        min_pos_ratio,
                        neg_target,
                        seed + base_index,
                        img_ext,
                        label_ext,
                        jpg_quality,
                        neg_count,
                        base_index,
                    )
                )
            for future in tqdm(futures, desc="save patches", unit="site"):
                site_saved, site_neg_kept, site_neg_seen = future.result()
                saved += site_saved
                neg_kept += site_neg_kept
                neg_seen += site_neg_seen

    print(
        "Done. pos_total={}, neg_total={}, target_neg={}, neg_kept={}, saved={}".format(
            pos_total,
            neg_total,
            target_neg,
            neg_kept,
            saved,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clip GVLM patches with a mixed strategy (all positive + sampled negatives)."
    )
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--min-pos-ratio", type=float, default=0.0)
    parser.add_argument("--neg-to-pos", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-ext", type=str, default="jpg")
    parser.add_argument("--label-ext", type=str, default="png")
    parser.add_argument("--jpg-quality", type=int, default=95)
    parser.add_argument("--clear-output", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not (0.0 <= args.overlap < 1.0):
        raise ValueError("overlap must be in [0.0, 1.0).")
    if args.patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    if args.min_pos_ratio < 0.0 or args.min_pos_ratio > 1.0:
        raise ValueError("min_pos_ratio must be in [0.0, 1.0].")
    if args.neg_to_pos < 0.0:
        raise ValueError("neg_to_pos must be >= 0.")
    if args.jpg_quality < 1 or args.jpg_quality > 100:
        raise ValueError("jpg_quality must be in [1, 100].")
    if args.workers < 1:
        raise ValueError("workers must be >= 1.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    sites = list_sites(input_root)
    if not sites:
        raise RuntimeError(f"No site folders found under: {input_root}")

    sample_and_save(
        sites=sites,
        output_root=output_root,
        patch_size=args.patch_size,
        overlap=args.overlap,
        min_pos_ratio=args.min_pos_ratio,
        neg_to_pos=args.neg_to_pos,
        seed=args.seed,
        img_ext=args.img_ext,
        label_ext=args.label_ext,
        jpg_quality=args.jpg_quality,
        clear_output=args.clear_output,
        workers=args.workers,
    )


if __name__ == "__main__":
    """
    python scripts/dataset/GVLM/mixed_patch_sampler.py \
    --input-root data/Downstreams/滑坡检测-GVLM/GVLM_CD \
    --output-root data/Downstreams/滑坡检测-GVLM/GVLM_CD256_0.3neg \
    --patch-size 256 \
    --overlap 0.0 \
    --min-pos-ratio 0.05 \
    --neg-to-pos 0.3 \
    --seed 42 \
    --img-ext png \
    --label-ext png \
    --clear-output \
    --workers 4

    """
    main()

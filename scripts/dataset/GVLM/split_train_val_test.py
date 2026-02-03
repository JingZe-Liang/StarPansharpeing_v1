from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from tqdm import tqdm


def list_images(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_file()])


def build_name_map(paths: list[Path]) -> dict[str, Path]:
    return {p.stem: p for p in paths}


def validate_pairs(a_dir: Path, b_dir: Path, label_dir: Path) -> list[str]:
    a_files = list_images(a_dir)
    b_files = list_images(b_dir)
    l_files = list_images(label_dir)
    a_map = build_name_map(a_files)
    b_map = build_name_map(b_files)
    l_map = build_name_map(l_files)
    stems = sorted(set(a_map) & set(b_map) & set(l_map))
    if not stems:
        raise RuntimeError("No matched stems found between A/B/label.")
    if len(stems) != len(a_map) or len(stems) != len(b_map) or len(stems) != len(l_map):
        missing_a = (set(b_map) | set(l_map)) - set(a_map)
        missing_b = (set(a_map) | set(l_map)) - set(b_map)
        missing_l = (set(a_map) | set(b_map)) - set(l_map)
        raise RuntimeError(
            "Unmatched files detected. "
            f"missing_in_A={sorted(missing_a)[:10]}, "
            f"missing_in_B={sorted(missing_b)[:10]}, "
            f"missing_in_label={sorted(missing_l)[:10]}"
        )
    return stems


def split_indices(n: int, train_ratio: float, val_ratio: float, seed: int) -> tuple[list[int], list[int], list[int]]:
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return train_idx, val_idx, test_idx


def write_list(paths: list[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in tqdm(paths, desc=f"write {out_path.name}", unit="patch"):
            f.write(f"{p.name}\n")


def copy_split(
    stems: list[str],
    a_dir: Path,
    b_dir: Path,
    label_dir: Path,
    out_root: Path,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
) -> None:
    split_map = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }
    for split_name, idxs in split_map.items():
        (out_root / split_name / "im1").mkdir(parents=True, exist_ok=True)
        (out_root / split_name / "im2").mkdir(parents=True, exist_ok=True)
        (out_root / split_name / "label").mkdir(parents=True, exist_ok=True)
        for i in tqdm(idxs, desc=f"copy {split_name}", unit="patch"):
            stem = stems[i]
            a_path = next(a_dir.glob(f"{stem}.*"))
            b_path = next(b_dir.glob(f"{stem}.*"))
            l_path = next(label_dir.glob(f"{stem}.*"))
            shutil.copy2(a_path, out_root / split_name / "im1" / a_path.name)
            shutil.copy2(b_path, out_root / split_name / "im2" / b_path.name)
            shutil.copy2(l_path, out_root / split_name / "label" / l_path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split GVLM patches into train/val/test.")
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, choices=["list", "folders"], default="list")
    parser.add_argument("--list-dir", type=str, default="list")
    parser.add_argument("--output-root", type=str, default="")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not (0.0 <= args.train_ratio <= 1.0):
        raise ValueError("train_ratio must be in [0.0, 1.0].")
    if not (0.0 <= args.val_ratio <= 1.0):
        raise ValueError("val_ratio must be in [0.0, 1.0].")
    if not (0.0 <= args.test_ratio <= 1.0):
        raise ValueError("test_ratio must be in [0.0, 1.0].")
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    input_root = Path(args.input_root)
    a_dir = input_root / "A"
    b_dir = input_root / "B"
    label_dir = input_root / "label"
    a_files = list_images(a_dir)
    b_files = list_images(b_dir)
    l_files = list_images(label_dir)
    a_map = build_name_map(a_files)
    b_map = build_name_map(b_files)
    l_map = build_name_map(l_files)
    stems = validate_pairs(a_dir, b_dir, label_dir)

    train_idx, val_idx, test_idx = split_indices(
        len(stems),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    if args.mode == "list":
        list_dir = input_root / args.list_dir
        write_list([a_map[stems[i]] for i in train_idx], list_dir / "train.txt")
        write_list([a_map[stems[i]] for i in val_idx], list_dir / "val.txt")
        write_list([a_map[stems[i]] for i in test_idx], list_dir / "test.txt")
    else:
        out_root = Path(args.output_root) if args.output_root else input_root.parent / f"{input_root.name}-splitted"
        copy_split(stems, a_dir, b_dir, label_dir, out_root, train_idx, val_idx, test_idx)

    print(
        "Done. total={}, train={}, val={}, test={}".format(
            len(stems),
            len(train_idx),
            len(val_idx),
            len(test_idx),
        )
    )


if __name__ == "__main__":
    """
    python scripts/dataset/GVLM/split_train_val_test.py \
    --input-root data/Downstreams/滑坡检测-GVLM/GVLM_CD256_0.3neg \
    --train-ratio 0.7 \
    --val-ratio 0.1 \
    --test-ratio 0.2 \
    --seed 42 \
    --mode list \
    --list-dir list \
    """
    main()

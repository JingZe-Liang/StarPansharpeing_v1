from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8)).save(path)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _mk_mask(h: int, w: int, class_pixels: dict[int, int]) -> np.ndarray:
    arr = np.zeros(h * w, dtype=np.uint8)
    idx = 0
    for cls, cnt in class_pixels.items():
        arr[idx : idx + cnt] = np.uint8(cls)
        idx += cnt
    return arr.reshape(h, w)


def _build_toy_dataset(root: Path) -> Path:
    (root / "Images").mkdir(parents=True, exist_ok=True)
    (root / "Semantic_mask").mkdir(parents=True, exist_ok=True)

    # class 0: 70 (A), 35 (B), 10 (C)
    # class 1: 80 (D), 55 (B), 20 (A)
    # class 2: 90 (C), 20 (B), 5  (D)
    data = [
        ("10000_0_0", {0: 70, 1: 20, 2: 10}),
        ("10001_0_0", {0: 35, 1: 55, 2: 10}),
        ("10002_0_0", {0: 10, 1: 0, 2: 90}),
        ("10003_0_0", {0: 15, 1: 80, 2: 5}),
    ]
    for stem, counts in data:
        _touch(root / "Images" / f"{stem}.jpg")
        mask_name = f"{stem.split('_')[0]}_lab_{'_'.join(stem.split('_')[1:])}.png"
        _write_mask(root / "Semantic_mask" / mask_name, _mk_mask(10, 10, counts))

    val_list = root / "val.txt"
    val_list.write_text("", encoding="utf-8")
    return val_list


def test_support_greedy_top1_per_class(tmp_path: Path) -> None:
    root = tmp_path / "Flood-3i"
    val_list = _build_toy_dataset(root)
    out_dir = tmp_path / "out"

    script = Path("scripts/dataset/flood3i/select_support_by_class.py")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(root),
            "--val-list",
            str(val_list),
            "--out-dir",
            str(out_dir),
            "--num-classes",
            "3",
            "--k",
            "1",
            "--strategy",
            "greedy",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    c0 = (out_dir / "support_class_0_1.txt").read_text(encoding="utf-8").strip()
    c1 = (out_dir / "support_class_1_1.txt").read_text(encoding="utf-8").strip()
    c2 = (out_dir / "support_class_2_1.txt").read_text(encoding="utf-8").strip()
    assert c0.startswith("Images/10000_0_0.jpg")
    assert c1.startswith("Images/10003_0_0.jpg")
    assert c2.startswith("Images/10002_0_0.jpg")


def test_support_random_k2_writes_files(tmp_path: Path) -> None:
    root = tmp_path / "Flood-3i"
    val_list = _build_toy_dataset(root)
    out_dir = tmp_path / "out_random"

    script = Path("scripts/dataset/flood3i/select_support_by_class.py")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(root),
            "--val-list",
            str(val_list),
            "--out-dir",
            str(out_dir),
            "--num-classes",
            "3",
            "--k",
            "2",
            "--strategy",
            "random",
            "--seed",
            "7",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    for c in [0, 1, 2]:
        lines = [
            ln.strip()
            for ln in (out_dir / f"support_class_{c}_2.txt").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        assert len(lines) == 2
        assert all(len(ln.split()) == 2 for ln in lines)
        assert len(set(lines)) == 2

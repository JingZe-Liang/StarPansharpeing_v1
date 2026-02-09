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


def _balanced_mask(h: int, w: int, classes: list[int]) -> np.ndarray:
    # Repeat class ids in a grid to get near-uniform counts.
    arr = np.zeros((h, w), dtype=np.uint8)
    flat = arr.reshape(-1)
    for i in range(flat.size):
        flat[i] = classes[i % len(classes)]
    return arr


def test_select_nshots_excludes_val_and_writes_list(tmp_path: Path) -> None:
    root = tmp_path / "Flood-3i"
    (root / "Images").mkdir(parents=True, exist_ok=True)
    (root / "Semantic_mask").mkdir(parents=True, exist_ok=True)

    # Create 6 candidates, with one held out as val.
    samples = [
        ("10000_0_0", list(range(10))),  # best: 10 classes balanced
        ("10001_0_0", list(range(8))),  # 8 classes balanced
        ("10002_0_0", list(range(6))),  # 6 classes balanced
        ("10003_0_0", list(range(5))),  # 5 classes balanced
        ("10004_0_0", [0, 1]),  # 2 classes balanced
        ("10005_0_0", [0]),  # 1 class
    ]

    for stem, cls in samples:
        img_rel = root / "Images" / f"{stem}.jpg"
        mask_rel = root / "Semantic_mask" / f"{stem.split('_')[0]}_lab_{'_'.join(stem.split('_')[1:])}.png"
        _touch(img_rel)
        _write_mask(mask_rel, _balanced_mask(10, 10, cls))

    # Put the best one into val to ensure it is excluded.
    val_list = root / "val.txt"
    val_list.write_text("Images/10000_0_0.jpg Semantic_mask/10000_lab_0_0.png\n", encoding="utf-8")

    out = tmp_path / "train_5shots.txt"
    script = Path("scripts/dataset/flood3i/select_nshots.py")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(root),
            "--val-list",
            str(val_list),
            "--out",
            str(out),
            "--num-shots",
            "5",
            "--num-classes",
            "10",
            "--ignore-index",
            "255",
            "--min-class-fraction",
            "0.0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    lines = [ln.strip() for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 5
    assert all(len(ln.split()) == 2 for ln in lines)

    # Excludes val sample.
    assert "Images/10000_0_0.jpg Semantic_mask/10000_lab_0_0.png" not in set(lines)

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.stage2.segmentation.data.sos_oil_leakage import SOSOilLeakageDataset


def _write_sample(image_path: Path, mask_path: Path, *, image_value: int, mask_foreground: bool) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    image = np.full((4, 4, 3), fill_value=image_value, dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)
    if mask_foreground:
        mask[1:3, 1:3] = 255

    Image.fromarray(image).save(image_path)
    Image.fromarray(mask).save(mask_path)


def _build_fake_sos(root: Path) -> None:
    _write_sample(
        root / "train" / "sentinel" / "20001_sat.jpg",
        root / "train" / "sentinel" / "20001_mask.png",
        image_value=120,
        mask_foreground=True,
    )
    _write_sample(
        root / "train" / "palsar" / "10001_sat.jpg",
        root / "train" / "palsar" / "10001_mask.png",
        image_value=80,
        mask_foreground=False,
    )
    _write_sample(
        root / "test" / "sentinel" / "sat" / "20002_sat.jpg",
        root / "test" / "sentinel" / "gt" / "20002_mask.png",
        image_value=140,
        mask_foreground=True,
    )
    _write_sample(
        root / "test" / "palsar" / "sat" / "10002_sat.jpg",
        root / "test" / "palsar" / "gt" / "10002_mask.png",
        image_value=90,
        mask_foreground=False,
    )


def test_train_sentinel_sample_loading(tmp_path: Path) -> None:
    _build_fake_sos(tmp_path)
    ds = SOSOilLeakageDataset(root=tmp_path, split="train", modality="sentinel", return_name=True, return_modality=True)
    assert len(ds) == 1

    sample = ds[0]
    assert sample["image"].shape == (3, 4, 4)
    assert sample["mask"].shape == (1, 4, 4)
    assert sample["name"] == "20001"
    assert sample["modality"] == "sentinel"
    assert set(sample["mask"].unique().tolist()) == {0, 1}


def test_test_split_nested_dirs_loading(tmp_path: Path) -> None:
    _build_fake_sos(tmp_path)
    ds = SOSOilLeakageDataset(root=tmp_path, split="test", modality="palsar", return_name=True)
    assert len(ds) == 1

    sample = ds[0]
    assert sample["name"] == "10002"
    assert sample["image"].shape == (3, 4, 4)
    assert sample["mask"].shape == (1, 4, 4)


def test_both_modalities_concat(tmp_path: Path) -> None:
    _build_fake_sos(tmp_path)
    ds = SOSOilLeakageDataset(root=tmp_path, split="train", modality="both", return_modality=True)
    assert len(ds) == 2
    modalities = {ds[i]["modality"] for i in range(len(ds))}
    assert modalities == {"sentinel", "palsar"}

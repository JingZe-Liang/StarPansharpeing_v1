from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest


def _read_label_mat(label_path: Path) -> np.ndarray:
    with h5py.File(label_path, "r") as f:
        label = np.array(f["label"])
    if label.ndim == 3:
        label = np.squeeze(label)
    return label


@pytest.mark.parametrize(
    ("city", "label_file"),
    [
        ("beijing", "beijing_label.mat"),
        ("wuhan", "wuhan_label.mat"),
    ],
)
def test_cross_city_label_unique_values(city: str, label_file: str) -> None:
    data_root = Path("data/Downstreams/CrossCitySegmentation/data2")
    label_path = data_root / label_file
    assert label_path.exists(), f"Label file not found: {label_path}"

    label = _read_label_mat(label_path)
    uniques = np.unique(label)
    print(f"[{city}] label shape={label.shape}, dtype={label.dtype}, unique={uniques.tolist()}")

    assert label.ndim == 2
    assert uniques.size > 0

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.stage2.change_detection.data.cd_basic import LEVIRRaw1024Dataset, create_levir_raw1024_test_dataloader

RAW_ROOT = Path("data/Downstreams/ChangeDetection/LEVIR-CD/raw")


def _has_raw_data() -> bool:
    required = [
        RAW_ROOT / "test" / "A",
        RAW_ROOT / "test" / "B",
        RAW_ROOT / "test" / "label",
    ]
    return all(path.exists() for path in required)


@pytest.mark.skipif(not _has_raw_data(), reason="LEVIR raw1024 test data not found.")
def test_raw1024_dataset_getitem_shape() -> None:
    ds = LEVIRRaw1024Dataset(
        data_root=RAW_ROOT,
        split="test",
        transform=None,
        normalize=True,
        img_to_neg1_1=False,
        resize_to=None,
        random_crop_resize_to=None,
        return_name=True,
    )

    sample = ds[0]
    assert isinstance(sample["img1"], torch.Tensor)
    assert isinstance(sample["img2"], torch.Tensor)
    assert isinstance(sample["gt"], torch.Tensor)
    assert sample["img1"].shape == (3, 1024, 1024)
    assert sample["img2"].shape == (3, 1024, 1024)
    assert sample["gt"].shape == (1, 1024, 1024)
    assert isinstance(sample["name"], str)
    assert sample["name"].startswith("test/")


@pytest.mark.skipif(not _has_raw_data(), reason="LEVIR raw1024 test data not found.")
def test_raw1024_dataloader_batch_shape() -> None:
    dataset, dataloader = create_levir_raw1024_test_dataloader(
        data_root=RAW_ROOT,
        split="test",
        batch_size=1,
        shuffle=False,
        num_workers=0,
        transform=None,
        normalize=True,
        img_to_neg1_1=False,
        resize_to=None,
        random_crop_resize_to=None,
        return_name=True,
    )

    assert isinstance(dataset, LEVIRRaw1024Dataset)
    batch = next(iter(dataloader))
    assert batch["img1"].shape == (1, 3, 1024, 1024)
    assert batch["img2"].shape == (1, 3, 1024, 1024)
    assert batch["gt"].shape == (1, 1, 1024, 1024)
    assert "name" in batch

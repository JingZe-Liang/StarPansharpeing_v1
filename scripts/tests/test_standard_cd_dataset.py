from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from torch.utils.data import ConcatDataset, RandomSampler, SequentialSampler

from src.stage2.change_detection.data.levir_cd import (
    LEVIRCDDataset,
    S2LookingCDDataset,
    build_standard_cd_sample_paths,
    create_concat_cd_train_dataloader,
    create_s2looking_cd_dataloader,
)


def _save_rgb(path: Path, fill_value: int) -> None:
    arr = np.full((16, 16, 3), fill_value, dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _save_label(path: Path, fill_value: int) -> None:
    arr = np.full((16, 16), fill_value, dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _make_list_style_root(root: Path) -> Path:
    for folder in ["A", "B", "label", "list"]:
        (root / folder).mkdir(parents=True, exist_ok=True)

    names = ["sample_0.png", "sample_1.png"]
    (root / "list" / "train.txt").write_text("\n".join(names), encoding="utf-8")
    (root / "list" / "val.txt").write_text("\n".join(names[:1]), encoding="utf-8")
    (root / "list" / "test.txt").write_text("\n".join(names[:1]), encoding="utf-8")
    (root / "list" / "trainval.txt").write_text("\n".join(names), encoding="utf-8")

    _save_rgb(root / "A" / names[0], 100)
    _save_rgb(root / "B" / names[0], 120)
    _save_label(root / "label" / names[0], 255)

    _save_rgb(root / "A" / names[1], 80)
    _save_rgb(root / "B" / names[1], 60)
    _save_label(root / "label" / names[1], 0)
    return root


def _make_split_style_root(root: Path, img1_dir: str, img2_dir: str, label_dir: str) -> Path:
    for split in ["train", "val", "test"]:
        for folder in [img1_dir, img2_dir, label_dir]:
            (root / split / folder).mkdir(parents=True, exist_ok=True)

        names = [f"{split}_0.png", f"{split}_1.png"]
        _save_rgb(root / split / img1_dir / names[0], 100)
        _save_rgb(root / split / img2_dir / names[0], 120)
        _save_label(root / split / label_dir / names[0], 255)

        _save_rgb(root / split / img1_dir / names[1], 80)
        _save_rgb(root / split / img2_dir / names[1], 60)
        _save_label(root / split / label_dir / names[1], 0)

    return root


@pytest.fixture
def levir_like_root(tmp_path: Path) -> Path:
    return _make_list_style_root(tmp_path / "levir_like")


@pytest.fixture
def s2looking_like_root(tmp_path: Path) -> Path:
    return _make_split_style_root(tmp_path / "s2looking_like", img1_dir="Image1", img2_dir="Image2", label_dir="label")


@pytest.fixture
def svcd_like_root(tmp_path: Path) -> Path:
    return _make_split_style_root(tmp_path / "svcd_like", img1_dir="A", img2_dir="B", label_dir="OUT")


@pytest.fixture
def jl1_like_root(tmp_path: Path) -> Path:
    return _make_split_style_root(tmp_path / "jl1_like", img1_dir="A", img2_dir="B", label_dir="label")


def test_build_standard_cd_sample_paths_from_list(levir_like_root: Path) -> None:
    samples = build_standard_cd_sample_paths(data_root=levir_like_root, list_dir="list", split="train_val")
    assert len(samples) == 2
    assert samples[0].img1.exists()
    assert samples[0].img2.exists()
    assert samples[0].label.exists()


def test_build_standard_cd_sample_paths_from_split_subdir(s2looking_like_root: Path) -> None:
    samples = build_standard_cd_sample_paths(
        data_root=s2looking_like_root,
        list_dir="list",
        split="train",
        img1_dir="Image1",
        img2_dir="Image2",
        label_dir="label",
        use_split_subdir=True,
    )
    assert len(samples) == 2
    assert samples[0].name.startswith("train/")


def test_s2looking_random_crop_resize_output_shape(s2looking_like_root: Path) -> None:
    dataset = S2LookingCDDataset(
        data_root=s2looking_like_root,
        split="train",
        transform=None,
        normalize=True,
        resize_to=None,
        random_crop_resize_to=8,
    )
    sample = dataset[0]
    assert sample["img1"].shape[-2:] == (8, 8)
    assert sample["img2"].shape[-2:] == (8, 8)
    assert sample["gt"].shape[-2:] == (8, 8)


def test_create_s2looking_cd_dataloader(s2looking_like_root: Path) -> None:
    dataset, dataloader = create_s2looking_cd_dataloader(
        data_root=s2looking_like_root,
        split="val",
        batch_size=1,
        num_workers=0,
        transform=None,
        random_crop_resize_to=None,
        resize_to=8,
        shuffle=None,
    )

    assert isinstance(dataset, S2LookingCDDataset)
    assert isinstance(dataloader.sampler, SequentialSampler)

    _, train_loader = create_s2looking_cd_dataloader(
        data_root=s2looking_like_root,
        split="train",
        batch_size=1,
        num_workers=0,
        transform=None,
        random_crop_resize_to=None,
        resize_to=8,
        shuffle=None,
    )
    assert isinstance(train_loader.sampler, RandomSampler)


def test_create_concat_cd_train_dataloader(
    levir_like_root: Path,
    s2looking_like_root: Path,
    svcd_like_root: Path,
    jl1_like_root: Path,
) -> None:
    concat_dataset, dataloader = create_concat_cd_train_dataloader(
        batch_size=2,
        num_workers=0,
        dataset_names=("levir", "s2looking", "svcd", "jl1"),
        output_size=8,
        shuffle=False,
        transform=None,
        normalize=True,
        img_to_neg1_1=False,
        return_name=True,
        levir_data_root=levir_like_root,
        s2looking_data_root=s2looking_like_root,
        svcd_data_root=svcd_like_root,
        jl1_data_root=jl1_like_root,
        levir_split="train",
        s2looking_split="train",
        svcd_split="train",
        jl1_split="train",
        s2looking_use_random_crop_resize=True,
        jl1_use_random_crop_resize=False,
    )

    assert isinstance(concat_dataset, ConcatDataset)
    assert len(concat_dataset) == 8

    batch = next(iter(dataloader))
    assert batch["img1"].shape[-2:] == (8, 8)
    assert batch["img2"].shape[-2:] == (8, 8)
    assert batch["gt"].shape[-2:] == (8, 8)
    assert "name" in batch


def test_levir_and_s2looking_share_base(levir_like_root: Path, s2looking_like_root: Path) -> None:
    levir_ds = LEVIRCDDataset(
        data_root=levir_like_root, split="train", transform=None, normalize=True, return_name=True
    )
    s2_ds = S2LookingCDDataset(
        data_root=s2looking_like_root,
        split="train",
        transform=None,
        normalize=True,
        resize_to=16,
        random_crop_resize_to=None,
        return_name=True,
    )

    levir_sample = levir_ds[0]
    s2_sample = s2_ds[0]

    assert levir_sample["img1"].shape == s2_sample["img1"].shape
    assert levir_sample["img2"].shape == s2_sample["img2"].shape
    assert levir_sample["gt"].shape == s2_sample["gt"].shape

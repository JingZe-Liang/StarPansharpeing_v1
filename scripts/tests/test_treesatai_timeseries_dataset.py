from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import tifffile
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.stage2.classification.data.TreeSatAI_timeseries import (
    TreeSatAITimeSeriesDataset,
    treesatai_timeseries_collate_fn,
)


def _write_h5(path: Path, t1: int, t2: int, t3: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as file:
        file.create_dataset("sen-1-asc-data", data=np.random.randn(t1, 2, 6, 6).astype(np.float32))
        file.create_dataset("sen-1-des-data", data=np.random.randn(t2, 2, 6, 6).astype(np.float32))
        file.create_dataset("sen-2-data", data=np.random.randn(t3, 10, 6, 6).astype(np.float32))
        file.create_dataset("sen-2-masks", data=np.random.rand(t3, 2, 6, 6).astype(np.float32))
        file.create_dataset(
            "sen-1-asc-products",
            data=np.array([f"S1_ASC_201901{idx + 1:02d}" for idx in range(t1)], dtype=object),
            dtype=text_dtype,
        )
        file.create_dataset(
            "sen-1-des-products",
            data=np.array([f"S1_DES_201902{idx + 1:02d}" for idx in range(t2)], dtype=object),
            dtype=text_dtype,
        )
        file.create_dataset(
            "sen-2-products",
            data=np.array([f"S2_201903{idx + 1:02d}" for idx in range(t3)], dtype=object),
            dtype=text_dtype,
        )


def _prepare_fake_treesatai_ts(root: Path) -> None:
    (root / "split").mkdir(parents=True, exist_ok=True)
    (root / "split" / "train_filenames.lst").write_text("Acer_p_1_x.tif\nAbies_a_2_y.tif\n")
    (root / "split" / "val_filenames.lst").write_text("Acer_p_1_x.tif\n")
    (root / "split" / "test_filenames.lst").write_text("Abies_a_2_y.tif\n")

    labels = {
        "Acer_p_1_x.tif": [["Acer", 0.7], ["Quercus", 0.3]],
        "Abies_a_2_y.tif": [["Abies", 1.0]],
    }
    (root / "TreeSatBA_v9_60m_multi_labels.json").write_text(json.dumps(labels))

    _write_h5(root / "sentinel-ts" / "Acer_p_1_x_2019.h5", t1=3, t2=4, t3=2)
    _write_h5(root / "sentinel-ts" / "Abies_a_2_y_2020.h5", t1=5, t2=2, t3=3)
    (root / "aerial").mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(root / "aerial" / "Acer_p_1_x.tif", np.random.randint(0, 255, (304, 304, 4), dtype=np.uint8))
    tifffile.imwrite(root / "aerial" / "Abies_a_2_y.tif", np.random.randint(0, 255, (304, 304, 4), dtype=np.uint8))


def test_treesatai_timeseries_dataset_basic(tmp_path: Path) -> None:
    _prepare_fake_treesatai_ts(tmp_path)
    dataset = TreeSatAITimeSeriesDataset(
        root=str(tmp_path),
        split="train",
        label_mode="both",
        scale_mode="none",
    )

    assert len(dataset) == 2
    sample = dataset[0]
    assert sample["image_s1_asc"].shape == (3, 2, 6, 6)
    assert sample["image_s1_des"].shape == (4, 2, 6, 6)
    assert sample["image_s2"].shape == (2, 10, 6, 6)
    assert sample["mask_s2"].shape == (2, 2, 6, 6)
    assert sample["label"].shape == (15,)
    assert sample["label_proportion"].shape == (15,)
    assert sample["label"].dtype == torch.float32
    assert sample["doy_s1_asc"].dtype == torch.long
    assert isinstance(sample["products_s2"], list)
    assert sample["image_aerial"].shape == (4, 304, 304)


def test_treesatai_timeseries_collate_padding(tmp_path: Path) -> None:
    _prepare_fake_treesatai_ts(tmp_path)
    dataset = TreeSatAITimeSeriesDataset(root=str(tmp_path), split="train", label_mode="multi_hot")
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=treesatai_timeseries_collate_fn,
    )
    batch = next(iter(loader))

    assert batch["image_s1_asc"].shape == (2, 5, 2, 6, 6)
    assert batch["image_s1_des"].shape == (2, 4, 2, 6, 6)
    assert batch["image_s2"].shape == (2, 3, 10, 6, 6)
    assert batch["image_s1_asc_valid"].shape == (2, 5)
    assert batch["image_s2_valid"].shape == (2, 3)
    assert batch["label"].shape == (2, 15)
    assert len(batch["sample_id"]) == 2


def test_treesatai_timeseries_spatial_interp(tmp_path: Path) -> None:
    _prepare_fake_treesatai_ts(tmp_path)
    dataset = TreeSatAITimeSeriesDataset(
        root=str(tmp_path),
        split="train",
        label_mode="multi_hot",
        s1_hw=(64, 64),
        s2_hw=(96, 96),
        image_interp_mode="bilinear",
        s2_mask_interp_mode="nearest",
    )
    sample = dataset[0]
    assert sample["image_s1_asc"].shape == (3, 2, 64, 64)
    assert sample["image_s1_des"].shape == (4, 2, 64, 64)
    assert sample["image_s2"].shape == (2, 10, 96, 96)
    assert sample["mask_s2"].shape == (2, 2, 96, 96)


def test_treesatai_timeseries_s2_to_neg_1_1_range(tmp_path: Path) -> None:
    _prepare_fake_treesatai_ts(tmp_path)
    dataset = TreeSatAITimeSeriesDataset(
        root=str(tmp_path),
        split="train",
        label_mode="multi_hot",
        s2_to_neg_1_1=True,
    )
    sample = dataset[0]
    s2 = sample["image_s2"]
    assert float(s2.min()) >= -1.0
    assert float(s2.max()) <= 1.0


def test_treesatai_timeseries_max_t_limit(tmp_path: Path) -> None:
    _prepare_fake_treesatai_ts(tmp_path)
    dataset = TreeSatAITimeSeriesDataset(
        root=str(tmp_path),
        split="train",
        label_mode="multi_hot",
        max_t_s1=2,
        max_t_s2=1,
    )
    sample = dataset[0]
    assert sample["image_s1_asc"].shape[0] <= 2
    assert sample["image_s1_des"].shape[0] <= 2
    assert sample["image_s2"].shape[0] <= 1
    assert sample["mask_s2"].shape[0] <= 1
    assert len(sample["products_s1_asc"]) <= 2
    assert len(sample["products_s1_des"]) <= 2
    assert len(sample["products_s2"]) <= 1


def test_treesatai_timeseries_s1_nan_replaced_with_zero(tmp_path: Path) -> None:
    _prepare_fake_treesatai_ts(tmp_path)
    h5_path = tmp_path / "sentinel-ts" / "Acer_p_1_x_2019.h5"
    with h5py.File(h5_path, "r+") as file:
        s1_asc = file["sen-1-asc-data"][:]
        s1_des = file["sen-1-des-data"][:]
        s1_asc[0, 0, 0, 0] = np.nan
        s1_des[0, 1, 1, 1] = np.nan
        file["sen-1-asc-data"][...] = s1_asc
        file["sen-1-des-data"][...] = s1_des

    dataset = TreeSatAITimeSeriesDataset(
        root=str(tmp_path),
        split="train",
        sensors=("s1_asc", "s1_des"),
        label_mode="multi_hot",
        scale_mode="none",
    )
    sample = dataset[0]
    assert torch.isfinite(sample["image_s1_asc"]).all()
    assert torch.isfinite(sample["image_s1_des"]).all()
    assert sample["image_s1_asc"][0, 0, 0, 0].item() == 0.0
    assert sample["image_s1_des"][0, 1, 1, 1].item() == 0.0


def _pick_time_indices(num_frames: int, num_slots: int) -> list[int]:
    if num_frames <= 0:
        return [0] * num_slots
    if num_frames == 1:
        return [0] * num_slots
    picked = np.linspace(0, num_frames - 1, num=num_slots, dtype=int)
    return [int(x) for x in picked]


def _to_display_rgb(frame_chw: torch.Tensor, modality: str) -> np.ndarray:
    if frame_chw.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape={tuple(frame_chw.shape)}")
    arr = frame_chw.detach().float().cpu()

    if modality == "s2":
        rgb = arr[[2, 1, 0], :, :]
    elif modality == "aerial":
        rgb = arr[[3, 1, 2], :, :]
    elif modality in {"s1_asc", "s1_des"}:
        ch0 = arr[0:1]
        ch1 = arr[1:2]
        rgb = torch.cat([ch0, ch1, ch1], dim=0)
    else:
        rgb = arr[:3]

    low = torch.quantile(rgb.reshape(-1), 0.02)
    high = torch.quantile(rgb.reshape(-1), 0.98)
    rgb = (rgb - low) / (high - low).clamp_min(1e-6)
    rgb = rgb.clamp(0.0, 1.0)
    return rgb.permute(1, 2, 0).numpy()


def visualize_real_treesatai_sample(
    *,
    root: str = "data/Downstreams/TreeSatAI_TimeSeries",
    split: str = "train",
    sample_index: int = 0,
    num_time_slots: int = 4,
    out_path: str = "outputs/treesatai_ts_real_sample.png",
    s1_hw: tuple[int, int] = (96, 96),
    s2_hw: tuple[int, int] = (96, 96),
) -> Path:
    dataset = TreeSatAITimeSeriesDataset(
        root=root,
        split=split,  # type: ignore[arg-type]
        sensors=("aerial", "s1_asc", "s1_des", "s2"),
        aerial_hw=s1_hw,
        label_mode="both",
        s1_hw=s1_hw,
        s2_hw=s2_hw,
        image_interp_mode="bilinear",
        s2_mask_interp_mode="nearest",
        s2_to_neg_1_1=True,
    )
    sample = dataset[sample_index]

    modality_map: dict[str, torch.Tensor] = {
        "aerial": sample["image_aerial"].unsqueeze(0),
        "s1_asc": sample["image_s1_asc"],
        "s1_des": sample["image_s1_des"],
        "s2": sample["image_s2"],
    }

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, num_time_slots, figsize=(4 * num_time_slots, 13), squeeze=False)
    row_order = ["aerial", "s1_asc", "s1_des", "s2"]
    for row_idx, modality in enumerate(row_order):
        seq = modality_map[modality]
        indices = _pick_time_indices(int(seq.shape[0]), num_time_slots)
        for col_idx, time_idx in enumerate(indices):
            axis = axes[row_idx, col_idx]
            image = _to_display_rgb(seq[time_idx], modality)
            axis.imshow(image)
            axis.set_title(f"{modality} t={time_idx}")
            axis.axis("off")

    fig.suptitle(f"TreeSatAI-TS sample={sample['sample_id']}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_file


if __name__ == "__main__":
    visualize_real_treesatai_sample(
        root="data/Downstreams/TreeSatAI_TimeSeries",
        split="train",
        sample_index=0,
        num_time_slots=6,
        out_path="outputs/treesatai_ts_real_sample_t6.png",
        s1_hw=(96, 96),
        s2_hw=(96, 96),
    )

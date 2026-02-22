from io import StringIO
from pathlib import Path
from typing import Callable, Literal, cast

import numpy as np
import rasterio
import torch as th
from easydict import EasyDict as edict
from kornia.augmentation import AugmentationSequential, RandomHorizontalFlip, RandomRotation, RandomVerticalFlip, Resize
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.data.utils import normalize_image_ as norm
from src.utilities.config_utils import function_config_to_easy_dict
from src.utilities.io import read_image

from .robust_sampler import RobustHyperspectralSampler, sample_img_with_gt_indices


def label_background_convert(label: Tensor | np.ndarray, convert_bg: bool = True):
    where_fn = th.where if th.is_tensor(label) else np.where
    if convert_bg:
        # Convert background (0) to 255, but preserve existing 255 (padding)
        # This ensures padding values remain 255 and don't become 254
        fixed_padded_label = where_fn(label == 255, 255, label - 1)
        result = where_fn(label == 0, 255, fixed_padded_label)  # type: ignore[no-matching-overload]
    else:
        result = label
    return result


def label_background_recover(label: Tensor | np.ndarray, convert_bg: bool = True):
    where_fn = th.where if th.is_tensor(label) else np.where
    result = where_fn(label == 255, 0, label + 1) if convert_bg else label
    return result


def get_default_transform(p=0.5):
    """Get default augmentation transform for hyperspectral images."""
    transform = AugmentationSequential(
        RandomHorizontalFlip(p=p),
        RandomVerticalFlip(p=p),
        RandomRotation(degrees=90, p=p, align_corners=False),
        data_keys=["input", "mask"],
        same_on_batch=False,
        keepdim=True,
    )
    return transform


_HOUSTON13_RAW_REQUIRED_FILES = (
    "2013_IEEE_GRSS_DF_Contest_CASI.tif",
    "2013_IEEE_GRSS_DF_Contest_Samples_TR.txt",
    "2013_IEEE_GRSS_DF_Contest_Samples_VA.txt",
)


def _resolve_houston13_raw_dir(data_root: Path, raw_dir: str | None = None) -> Path:
    candidates: list[Path] = []
    if raw_dir is not None:
        candidates.append(data_root / raw_dir)
    candidates.extend([data_root / "Houston2013" / "2013_DFTC", data_root / "2013_DFTC", data_root])
    for candidate in candidates:
        if all((candidate / filename).exists() for filename in _HOUSTON13_RAW_REQUIRED_FILES):
            return candidate
    raise FileNotFoundError(
        "Cannot find Houston13 raw files. "
        f"Expected files {_HOUSTON13_RAW_REQUIRED_FILES} under one of: {[c.as_posix() for c in candidates]}"
    )


def _apply_houston_roi_block(label: np.ndarray, roi_block: str, class_id: int) -> int:
    if roi_block.strip() == "":
        return class_id
    try:
        data = np.loadtxt(StringIO(roi_block), usecols=(2, 1), comments=";", dtype=np.int64)
    except ValueError:
        return class_id
    if data.size == 0:
        return class_id
    if data.ndim == 1:
        data = data[None, :]

    rows = data[:, 0]
    cols = data[:, 1]
    valid = (rows >= 0) & (rows < label.shape[0]) & (cols >= 0) & (cols < label.shape[1])
    rows = rows[valid]
    cols = cols[valid]
    if rows.size == 0:
        return class_id

    current = label[rows, cols]
    conflict_mask = (current != 0) & (current != class_id)
    if np.any(conflict_mask):
        label[rows[conflict_mask], cols[conflict_mask]] = 0

    assign_mask = current == 0
    label[rows[assign_mask], cols[assign_mask]] = class_id
    return class_id + 1


def _read_houston_roi_txt(path: Path, shape: tuple[int, int]) -> np.ndarray:
    label = np.zeros(shape, dtype=np.int32)
    block_lines: list[str] = []
    class_id = 1

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith(";") or line.isspace():
                continue
            if line.lstrip().startswith("1 "):
                class_id = _apply_houston_roi_block(label, "".join(block_lines), class_id)
                block_lines.clear()
            else:
                block_lines.append(line)

    class_id = _apply_houston_roi_block(label, "".join(block_lines), class_id)
    _ = class_id
    return label


def _ensure_numpy_array(value: object, value_name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"Failed to load {value_name}: got None")
    if isinstance(value, dict):
        if len(value) != 1:
            raise ValueError(f"Failed to load {value_name}: dict has {len(value)} entries, expected exactly 1")
        value = next(iter(value.values()))
    if isinstance(value, list):
        if len(value) != 1 or not isinstance(value[0], np.ndarray):
            raise ValueError(f"Failed to load {value_name}: list shape is unsupported")
        value = value[0]
    if not isinstance(value, np.ndarray):
        raise ValueError(f"Failed to load {value_name}: unsupported type {type(value)!r}")
    return value


def houston13_load_from_dir(
    data_dir: str,
    raw_dir: str | None = None,
):
    data_root = Path(data_dir)
    dataset_dir = _resolve_houston13_raw_dir(data_root=data_root, raw_dir=raw_dir)
    casi_path = dataset_dir / "2013_IEEE_GRSS_DF_Contest_CASI.tif"
    train_label_path = dataset_dir / "2013_IEEE_GRSS_DF_Contest_Samples_TR.txt"
    test_label_path = dataset_dir / "2013_IEEE_GRSS_DF_Contest_Samples_VA.txt"

    with rasterio.open(casi_path) as f:
        casi_chw = f.read()
    img_hwc = np.transpose(casi_chw, (1, 2, 0))

    shape_hw = (img_hwc.shape[0], img_hwc.shape[1])
    train_label = _read_houston_roi_txt(train_label_path, shape=shape_hw)
    test_label = _read_houston_roi_txt(test_label_path, shape=shape_hw)

    all_label = np.zeros(shape_hw, dtype=np.int32)
    train_mask = train_label != 0
    test_mask = test_label != 0
    all_label[train_mask] = train_label[train_mask]
    all_label[test_mask] = test_label[test_mask]

    return img_hwc, all_label


class SingleMatDataset(Dataset):
    """
    Hyperspectral Collection Dataset,
    loading one single mat file to train an classification model.
    """

    task: str = "classification"
    paths: edict = edict(
        {
            "Houston13": {"img": "mat/Houston13_hwc.mat", "gt": "cls_GT/Houston13_7gt.mat"},
            "Houston13_Raw": {"raw_dir": "Houston2013/2013_DFTC"},
            "Houston18": {"img": "mat/Houston18_hwc.mat", "gt": "cls_GT/Houston18_7gt.mat"},
            "Indian_pines": {"img": "mat/Indian_pines_corrected.mat", "gt": "cls_GT/Indian_pines_gt.mat"},
            "Pavia": {"img": "mat/Pavia.mat", "gt": "cls_GT/Pavia_gt.mat"},
            "PaviaU": {"img": "mat/PaviaU.mat", "gt": "cls_GT/PaviaU_gt.mat"},
            "WHU_Hi_HanChuan": {"img": "mat/WHU_Hi_HanChuan.mat", "gt": "cls_GT/WHU_Hi_HanChuan_gt.mat"},
            "WHU_Hi_HongHu": {"img": "mat/WHU_Hi_HongHu.mat", "gt": "cls_GT/WHU_Hi_HongHu_gt.mat"},
            "WHU_Hi_LongKou": {"img": "mat/WHU_Hi_LongKou.mat", "gt": "cls_GT/WHU_Hi_LongKou_gt.mat"},
            "XiongAn": {"img": "mat/XiongAn.img", "gt": "cls_GT/farm_roi.img"},
        }
    )

    def __init__(
        self,
        data_dir: str,
        dataset_name: str,
        train_ratio: float = 0.2,
        samples_per_cls: int = 5,
        patch_size: int = 32,
        full_img: bool = False,
        skip_bg: bool = True,
        to_neg_1_1: bool = True,
        transform: Callable | None | Literal["default"] = None,
        online_slice: bool = False,
    ):
        self.train_ratio = train_ratio
        self.dataset_name = dataset_name
        assert dataset_name in list(self.paths.keys()), (
            f"Dataset {dataset_name} not in supported datasets: {list(self.paths.keys())}"
        )
        path = self.paths[dataset_name]

        if dataset_name == "Houston13_Raw":
            self.img, self.gt = houston13_load_from_dir(data_dir=data_dir, raw_dir=path.get("raw_dir"))
        else:
            img_path = Path(data_dir, path.img)
            gt_path = Path(data_dir, path.gt)
            self.img = _ensure_numpy_array(read_image(img_path), "img")
            self.gt = _ensure_numpy_array(read_image(gt_path), "gt")
        if self.gt.ndim == 3 and self.gt.shape[-1] == 1:
            self.gt = self.gt[..., 0]
        assert self.gt.ndim == 2, f"GT must be 2D, got shape {self.gt.shape}"

        self.full_img = full_img
        self.skip_bg = skip_bg
        self.n_classes = np.unique(self.gt).shape[0] - (1 if skip_bg else 0)

        logger.info(f"Load image name: {dataset_name}, shaped {self.img.shape}")
        logger.info(f"Load gt, labels class has number: {np.unique(self.gt)}")

        # Normalize to [0, 1]
        self.to_neg_1_1 = to_neg_1_1
        self.img = (self.img - self.img.min()) / (self.img.max() - self.img.min())
        if self.to_neg_1_1:
            self.img = self.img * 2.0 - 1.0

        # Transform
        if transform == "default":
            transform = get_default_transform()
        self.transform = transform
        self.resize_fn = AugmentationSequential(
            Resize(patch_size, keepdim=True), data_keys=["input", "mask"], keepdim=True
        )
        self.online_slice = online_slice
        self.patch_size = patch_size

        if not full_img:
            self.sampler = RobustHyperspectralSampler(
                self.gt,
                balance_strategy="equal_size",
                skip_bg=0 if skip_bg else None,  # skip_bg is int type, 0 is background class
                random_seed=2025,
                verbose=True,
            )

            # Samples
            self.sample_indices_train = self.sampler.sample(samples_per_class=samples_per_cls)
            # If online slice is requested, we don't pre-extract patches into memory;
            # instead, we keep the center coordinates and will extract patches in __getitem__.
            if self.online_slice:
                # Build list of valid coordinates and unsampled area mask
                margin = patch_size // 2
                self.patches_coords = []
                unsampled_area = np.zeros_like(self.gt)
                height, width = self.gt.shape

                # Cache padded image and gt ONCE to avoid repeated allocation
                self.padded_img = np.pad(
                    self.img, ((margin, margin), (margin, margin), (0, 0)), mode="constant", constant_values=0
                )
                self.padded_gt = np.pad(
                    self.gt, ((margin, margin), (margin, margin)), mode="constant", constant_values=255
                )

                for class_id, coords in self.sample_indices_train.items():
                    for coord in coords:
                        c = np.asarray(coord)
                        cx, cy = int(c[0]), int(c[1])
                        # ensure valid center
                        if cx < margin or cx >= height - margin or cy < margin or cy >= width - margin:
                            continue
                        self.patches_coords.append((int(cx), int(cy)))
                        # mark unsampled area
                        x0, x1 = cx - margin, cx + margin + 1
                        y0, y1 = cy - margin, cy + margin + 1
                        unsampled_area[x0:x1, y0:y1] = 1
                # unsampled_area means True is unsampled
                self.unsampled_area = np.logical_not(unsampled_area.astype(np.bool_))
                # No pre-extracted patches
                self.patches_indices, self.label_indices = [], []
            else:
                patches_indices, label_indices, self.unsampled_area, patches_coords = sample_img_with_gt_indices(
                    self.img, self.gt, cast(dict[int, np.ndarray], self.sample_indices_train), patch_size=patch_size
                )
                # Flatten as before
                self.patches_indices, self.label_indices = [], []
                self.patches_coords = []
                for img_p in patches_indices.values():
                    self.patches_indices += img_p
                for gt_p in label_indices.values():
                    self.label_indices += gt_p
                for coords_p in patches_coords.values():
                    self.patches_coords += coords_p
            logger.info(f"Dataset sampled done.")
            self.unsampled_area = th.as_tensor(self.unsampled_area)

            # We already created / flattened patches or coords above
            if self.online_slice:
                logger.info(f"{self.dataset_name} train patches (online coords): {len(self.patches_coords)}")
            else:
                logger.info(f"{self.dataset_name} train patches: {len(self.patches_indices)}")
                assert len(self.patches_indices) == len(self.label_indices), (
                    f"{self.dataset_name} train patches and labels are not equal"
                )

    def __len__(self) -> int:
        return len(self.patches_coords) if self.online_slice else len(self.patches_indices)

    def _extract_patch_at_coord(self, coord: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """Extract a single image patch and label patch centered at `coord`, using cached padding."""
        cx, cy = coord
        patch_size = self.patch_size
        padded_x = int(cx)
        padded_y = int(cy)
        slices = slice(padded_x, padded_x + patch_size), slice(padded_y, padded_y + patch_size)
        patch = self.padded_img[slices[0], slices[1], :]
        label = self.padded_gt[slices[0], slices[1]]
        return patch, label

    def _label_post_process(self, label: np.ndarray | Tensor) -> np.ndarray | Tensor:
        """Convert 0 background to 255"""
        return label_background_convert(label, self.skip_bg)

    def _get_full_img_tensor(self):
        img = th.as_tensor(self.img).permute(-1, 0, 1)
        label = th.as_tensor(self._label_post_process(self.gt)).type(th.long)
        return edict({"img": img, "gt": label})

    def _get_patch_img_tensor(self, idx: int):
        if self.online_slice:
            center = self.patches_coords[idx]
            patch_arr, label_arr = self._extract_patch_at_coord(center)
            patch = th.as_tensor(patch_arr.transpose(-1, 0, 1))
            label = th.as_tensor(label_arr).type(th.long)
        else:
            patch = th.as_tensor(self.patches_indices[idx].transpose(-1, 0, 1))
            label = th.as_tensor(self.label_indices[idx]).type(th.long)
        label = self._label_post_process(label)

        # Resize
        # patch, label = self.resize_fn(patch, label)
        # print(patch.shape, label.shape)

        # Transform
        if self.transform is not None:
            patch, label = self.transform(patch, label)

        # Optionally include the center coordinate when using online slicing
        if self.online_slice:
            return edict({"img": patch, "gt": label, "coord": self.patches_coords[idx]})
        return edict({"img": patch, "gt": label})

    def __getitem__(self, idx: int):
        if self.full_img:
            return self._get_full_img_tensor()
        else:
            return self._get_patch_img_tensor(idx)

    def get_unsampled_area(self):
        return self.unsampled_area.unsqueeze(0) if not self.full_img else None

    @classmethod
    @function_config_to_easy_dict
    def from_config(cls, cfg):
        ds = cls(**cfg.dataset_kwargs)
        dl_cfg = cfg.dataloader_kwargs
        dl = DataLoader(
            ds,
            batch_size=dl_cfg.batch_size,
            num_workers=dl_cfg.num_workers,
            shuffle=dl_cfg.get("shuffle", False),
            pin_memory=dl_cfg.get("pin_memory", True),
            drop_last=dl_cfg.get("drop_last", False),
        )
        return ds, dl


def _test_single_mat_loader():
    from src.utilities.train_utils.visualization import get_rgb_image, visualize_segmentation_map

    img_dir = "data/Downstreams/ClassificationCollection/cls"

    ds = SingleMatDataset(img_dir, "XiongAn", train_ratio=0.4, samples_per_cls=10, patch_size=128, online_slice=True)
    print("Dataset length:", len(ds))

    # full_img = th.as_tensor((ds.img + 1) / 2)[None].permute(0, -1, 1, 2)
    # full_gt = ds.gt
    # full_img_vis = get_rgb_image(full_img, "mean", True)
    # full_gt_vis = visualize_segmentation_map(full_gt, n_class=ds.n_classes, to_pil=False)

    train_loader = DataLoader(ds, batch_size=4, shuffle=True)
    for batch in train_loader:
        img_vis = get_rgb_image((batch.img + 1) / 2, "mean", True)
        gt = label_background_recover(batch.gt)
        gt_vis = th.as_tensor(visualize_segmentation_map(gt, n_class=ds.n_classes, to_pil=False)).permute(0, -1, 1, 2)

        print(batch["img"].shape, "min: {}, max: {}".format(batch["img"].min(), batch["img"].max()))
        print(batch["gt"].shape, batch["gt"].unique())


def _test_single_mat_houston_13_loader():
    img_dir = "data/Downstreams/ClassificationCollection/cls"

    ds = SingleMatDataset(
        data_dir=img_dir,
        dataset_name="Houston13_Raw",
        samples_per_cls=10,
        patch_size=128,
        online_slice=True,
        to_neg_1_1=True,
        skip_bg=True,
    )
    print("Dataset length:", len(ds))
    print("Image shape:", ds.img.shape, "GT shape:", ds.gt.shape)
    print("GT unique labels:", np.unique(ds.gt))

    train_loader = DataLoader(ds, batch_size=4, shuffle=True)
    for batch in train_loader:
        print(batch["img"].shape, "min:", float(batch["img"].min()), "max:", float(batch["img"].max()))
        print(batch["gt"].shape, batch["gt"].unique())
        break


if __name__ == "__main__":
    """
    python -m src.stage2.segmentation.data.single_mat_loader
    """
    # _test_single_mat_loader()
    _test_single_mat_houston_13_loader()

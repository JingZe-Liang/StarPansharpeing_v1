from __future__ import annotations

import os
import importlib
import urllib.request
from collections.abc import Sequence
from typing import Any

import kornia.augmentation as K
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchgeo.datasets import TreeSatAI as TorchGeoTreeSatAI
from torchgeo.datasets.utils import check_integrity, extract_archive

from src.stage2.classification.data.builder import create_dataloader as build_dataloader

TREESATAI_MEAN: dict[str, tuple[float, ...]] = {
    "aerial": (
        151.26809261440323,
        93.1159469148246,
        85.05016794624635,
        81.0471576353153,
    ),
    "s1": (
        -6.933713050794077,
        -12.628564056094067,
        0.47448312147709354,
    ),
    "s2": (
        231.43385024546893,
        376.94788434611434,
        241.03688288984037,
        2809.8421354087955,
        616.5578221193639,
        2104.3826773960823,
        2695.083864757169,
        2969.868417923599,
        1306.0814241837832,
        587.0608264363341,
        249.1888624097736,
        2950.2294375352285,
    ),
}
TREESATAI_STD: dict[str, tuple[float, ...]] = {
    "aerial": (
        48.70879149145466,
        33.59622314610158,
        28.000497087051126,
        33.683983599997724,
    ),
    "s1": (
        87.8762246957811,
        47.03070478433704,
        1.297291303623673,
    ),
    "s2": (
        123.16515044781909,
        139.78991338362886,
        140.6154081184225,
        786.4508872594147,
        202.51268536579394,
        530.7255451201194,
        710.2650071967689,
        777.4421400779165,
        424.30312334282684,
        247.21468849049668,
        122.80062680549261,
        702.7404237034002,
    ),
}


def _extract_treesatai_file(root: str, file_name: str) -> None:
    if not file_name.endswith(".zip"):
        return
    to_path = root
    if file_name.startswith("aerial"):
        to_path = os.path.join(root, "aerial", "60m")
    extract_archive(os.path.join(root, file_name), to_path)


def _download_with_progress(url: str, output_path: str) -> None:
    progress_factory: Any | None = None
    try:
        tqdm_module = importlib.import_module("tqdm")
        progress_factory = getattr(tqdm_module, "tqdm", None)
    except Exception:
        progress_factory = None

    with urllib.request.urlopen(url) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        part_path = f"{output_path}.part"
        progress_bar: Any | None = None
        if progress_factory is not None:
            progress_bar = progress_factory(
                total=total_size if total_size > 0 else None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=os.path.basename(output_path),
                leave=True,
            )
        try:
            with open(part_path, "wb") as file:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    file.write(chunk)
                    if progress_bar is not None:
                        progress_bar.update(len(chunk))
            os.replace(part_path, output_path)
        finally:
            if progress_bar is not None:
                progress_bar.close()
            if os.path.exists(part_path):
                os.remove(part_path)


def prepare_treesatai_download(root: str, checksum: bool = False) -> None:
    root_dir = os.path.expanduser(root)
    os.makedirs(root_dir, exist_ok=True)
    sensor_dirs_ready = all(os.path.isdir(os.path.join(root_dir, sensor)) for sensor in TorchGeoTreeSatAI.all_sensors)
    if sensor_dirs_ready:
        return

    for file_name, md5 in TorchGeoTreeSatAI.md5s.items():
        print(f"Check {file_name=} and {md5=}")
        file_path = os.path.join(root_dir, file_name)
        is_valid = check_integrity(file_path, md5 if checksum else None)
        if not is_valid:
            url = f"{TorchGeoTreeSatAI.url}{file_name}"
            _download_with_progress(url, file_path)
            if not check_integrity(file_path, md5 if checksum else None):
                raise RuntimeError(f"Downloaded file '{file_path}' is corrupted.")
        print(f"start extracting {file_name=}")
        _extract_treesatai_file(root_dir, file_name)


class TreeSatAIMultiLabel(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        sensors: Sequence[str] = ("aerial", "s1", "s2"),
        image_size: int | tuple[int, int] = 304,
        normalize: bool = True,
        to_neg_1_1: bool = False,
        make_fused_image: bool = True,
        train_with_aug: bool = True,
        val_ratio: float = 0.1,
        val_seed: int = 0,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}. Expected one of train/val/test.")
        if normalize and to_neg_1_1:
            raise ValueError("normalize and to_neg_1_1 are mutually exclusive.")

        sensor_tuple = tuple(sensors)
        if not sensor_tuple:
            raise ValueError("sensors must not be empty.")
        invalid_sensors = set(sensor_tuple) - set(TorchGeoTreeSatAI.all_sensors)
        if invalid_sensors:
            raise ValueError(f"Unsupported sensors: {sorted(invalid_sensors)}.")

        self.split = split
        self.sensors = sensor_tuple
        self.normalize = normalize
        self.to_neg_1_1 = to_neg_1_1
        self.make_fused_image = make_fused_image
        self.train_with_aug = train_with_aug

        if download:
            prepare_treesatai_download(root=root, checksum=checksum)
            download = False

        source_split = "test" if split == "test" else "train"
        self.base_dataset = TorchGeoTreeSatAI(
            root=root,
            split=source_split,
            sensors=self.sensors,
            transforms=None,
            download=download,
            checksum=checksum,
        )
        self.indices = self._build_indices(
            dataset_size=len(self.base_dataset),
            split=split,
            val_ratio=val_ratio,
            val_seed=val_seed,
        )
        self.classes = tuple(self.base_dataset.classes)

        size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.train_aug = K.AugmentationSequential(
            K.RandomVerticalFlip(p=0.5),
            K.RandomHorizontalFlip(p=0.5),
            K.Resize(size=size),
            data_keys=["input"],
            keepdim=True,
        )
        self.eval_aug = K.AugmentationSequential(
            K.Resize(size=size),
            data_keys=["input"],
            keepdim=True,
        )
        self.norm_layers = {
            sensor: K.Normalize(mean=list(TREESATAI_MEAN[sensor]), std=list(TREESATAI_STD[sensor]), keepdim=True)
            for sensor in self.sensors
        }

    @staticmethod
    def _build_indices(
        dataset_size: int,
        split: str,
        val_ratio: float,
        val_seed: int,
    ) -> list[int]:
        if split == "test":
            return list(range(dataset_size))
        if not (0.0 <= val_ratio < 1.0):
            raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}.")

        generator = torch.Generator().manual_seed(val_seed)
        indices = torch.randperm(dataset_size, generator=generator).tolist()
        val_size = int(dataset_size * val_ratio)
        if val_ratio > 0.0 and val_size == 0 and dataset_size > 1:
            val_size = 1

        train_size = dataset_size - val_size
        if split == "train":
            return indices[:train_size]
        if split == "val":
            if val_size == 0:
                raise ValueError("val split is empty. Increase val_ratio.")
            return indices[train_size:]
        raise ValueError(f"Unsupported split: {split}.")

    def __len__(self) -> int:
        return len(self.indices)

    def _apply_geometric_transform(
        self,
        image: Tensor,
        params: Any | None = None,
    ) -> tuple[Tensor, Any]:
        image = image.unsqueeze(0)
        aug = self.train_aug if self.split == "train" and self.train_with_aug else self.eval_aug
        if params is None:
            transformed = aug(image)
            return transformed.squeeze(0), aug._params
        transformed = aug(image, params=params)
        return transformed.squeeze(0), params

    @staticmethod
    def _normalize_to_neg_1_1(image: Tensor) -> Tensor:
        min_value = image.amin()
        max_value = image.amax()
        image = (image - min_value) / (max_value - min_value).clamp_min(1e-6)
        return image * 2.0 - 1.0

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        base_sample = self.base_dataset[self.indices[index]]
        output: dict[str, Tensor] = {"label": base_sample["label"].float()}

        aug_params: Any | None = None
        fused_parts: list[Tensor] = []
        for sensor in self.sensors:
            key = f"image_{sensor}"
            image = base_sample[key].float()
            image, aug_params = self._apply_geometric_transform(image, params=aug_params)

            if self.normalize:
                image = self.norm_layers[sensor](image.unsqueeze(0)).squeeze(0)
            elif self.to_neg_1_1:
                image = self._normalize_to_neg_1_1(image)

            output[key] = image
            fused_parts.append(image)

        if self.make_fused_image:
            output["image"] = torch.cat(fused_parts, dim=0)
        return output

    @classmethod
    def create_dataloader(
        cls,
        dataset_kwargs: dict[str, Any] | None = None,
        loader_kwargs: dict[str, Any] | None = None,
    ) -> tuple["TreeSatAIMultiLabel", DataLoader[Any]]:
        dataset_kwargs = dataset_kwargs or {}
        loader_kwargs = loader_kwargs or {}
        dataset = cls(**dataset_kwargs)
        _, dataloader = build_dataloader(dataset=dataset, loader_kwargs=loader_kwargs)
        return dataset, dataloader

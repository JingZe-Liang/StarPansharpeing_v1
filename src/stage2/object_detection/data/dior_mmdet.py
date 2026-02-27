from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
import xml.etree.ElementTree as ET

import numpy as np
import torch
from mmengine.dataset import pseudo_collate
from mmengine.structures import InstanceData
from torch.utils.data import DataLoader, Dataset

from src.stage2.object_detection.data.DOTA import build_default_transforms, dota_poly_to_obb_le90

Split = Literal["train", "val", "test"]
BBoxOrientation = Literal["horizontal", "oriented"]

DIOR_CLASSES: tuple[str, ...] = (
    "airplane",
    "airport",
    "baseballfield",
    "basketballcourt",
    "bridge",
    "chimney",
    "dam",
    "expresswayservicearea",
    "expresswaytollstation",
    "golffield",
    "groundtrackfield",
    "harbor",
    "overpass",
    "ship",
    "stadium",
    "storagetank",
    "tenniscourt",
    "trainstation",
    "vehicle",
    "windmill",
)

LABEL_ALIASES: dict[str, str] = {
    "Expressway-Service-area": "expresswayservicearea",
    "Expressway-toll-station": "expresswaytollstation",
}


@dataclass(slots=True, frozen=True)
class DIORSampleRecord:
    image_id: str
    image_path: Path
    annotation_path: Path | None


def _normalize_label_name(label: str) -> str:
    label = label.strip()
    if label in LABEL_ALIASES:
        return LABEL_ALIASES[label]
    return label.lower().replace("-", "").replace("_", "").replace(" ", "")


def _read_split_ids(root: Path, split: Split) -> list[str]:
    split_path = root / "Main" / f"{split}.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    with split_path.open("r", encoding="utf-8") as file:
        sample_ids = [line.strip() for line in file if line.strip()]
    if not sample_ids:
        raise ValueError(f"No ids found in split file: {split_path}")
    return sample_ids


def _resolve_image_path(image_dir: Path, image_id: str) -> Path:
    jpg_path = image_dir / f"{image_id}.jpg"
    if jpg_path.exists():
        return jpg_path
    png_path = image_dir / f"{image_id}.png"
    if png_path.exists():
        return png_path
    raise FileNotFoundError(f"Image not found for id={image_id} under {image_dir}")


def _build_records(root: Path, split: Split, bbox_orientation: BBoxOrientation) -> list[DIORSampleRecord]:
    split_ids = _read_split_ids(root=root, split=split)
    image_dir_name = "JPEGImages-trainval" if split in ("train", "val") else "JPEGImages-test"
    image_dir = root / image_dir_name
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    if bbox_orientation == "horizontal":
        annotation_dir = root / "Annotations" / "Horizontal Bounding Boxes"
    else:
        annotation_dir = root / "Annotations" / "Oriented Bounding Boxes"

    records: list[DIORSampleRecord] = []
    for sample_id in split_ids:
        image_path = _resolve_image_path(image_dir=image_dir, image_id=sample_id)
        annotation_path = annotation_dir / f"{sample_id}.xml"
        if not annotation_path.exists():
            if split in ("train", "val"):
                raise FileNotFoundError(f"Annotation not found: {annotation_path}")
            annotation_path = None
        records.append(DIORSampleRecord(image_id=sample_id, image_path=image_path, annotation_path=annotation_path))
    return records


def _safe_float(text: str | None) -> float:
    if text is None:
        return 0.0
    return float(text)


def _extract_horizontal_objects(
    annotation_path: Path, class_to_idx: dict[str, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    root = ET.parse(annotation_path).getroot()
    boxes: list[list[float]] = []
    labels: list[int] = []

    for obj in root.findall("object"):
        label_name = obj.findtext("name")
        if label_name is None:
            continue
        normalized = _normalize_label_name(label_name)
        if normalized not in class_to_idx:
            raise KeyError(f"Unknown DIOR class name: {label_name}")
        bbox_node = obj.find("bndbox")
        if bbox_node is None:
            continue
        x1 = _safe_float(bbox_node.findtext("xmin"))
        y1 = _safe_float(bbox_node.findtext("ymin"))
        x2 = _safe_float(bbox_node.findtext("xmax"))
        y2 = _safe_float(bbox_node.findtext("ymax"))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2, y2])
        labels.append(class_to_idx[normalized])

    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)
    return torch.as_tensor(boxes, dtype=torch.float32), torch.as_tensor(labels, dtype=torch.long)


def _extract_oriented_objects(
    annotation_path: Path, class_to_idx: dict[str, int]
) -> tuple[torch.Tensor, list[list[float]], torch.Tensor]:
    root = ET.parse(annotation_path).getroot()
    boxes_xyxy: list[list[float]] = []
    polys: list[list[float]] = []
    labels: list[int] = []

    for obj in root.findall("object"):
        label_name = obj.findtext("name")
        if label_name is None:
            continue
        normalized = _normalize_label_name(label_name)
        if normalized not in class_to_idx:
            raise KeyError(f"Unknown DIOR class name: {label_name}")

        obb_node = obj.find("robndbox")
        if obb_node is None:
            continue
        x1 = _safe_float(obb_node.findtext("x_left_top"))
        y1 = _safe_float(obb_node.findtext("y_left_top"))
        x2 = _safe_float(obb_node.findtext("x_right_top"))
        y2 = _safe_float(obb_node.findtext("y_right_top"))
        x3 = _safe_float(obb_node.findtext("x_right_bottom"))
        y3 = _safe_float(obb_node.findtext("y_right_bottom"))
        x4 = _safe_float(obb_node.findtext("x_left_bottom"))
        y4 = _safe_float(obb_node.findtext("y_left_bottom"))

        poly = [x1, y1, x2, y2, x3, y3, x4, y4]
        xs = [x1, x2, x3, x4]
        ys = [y1, y2, y3, y4]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        if max_x <= min_x or max_y <= min_y:
            continue

        polys.append(poly)
        boxes_xyxy.append([min_x, min_y, max_x, max_y])
        labels.append(class_to_idx[normalized])

    if not polys:
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            [],
            torch.zeros((0,), dtype=torch.long),
        )
    return (
        torch.as_tensor(boxes_xyxy, dtype=torch.float32),
        polys,
        torch.as_tensor(labels, dtype=torch.long),
    )


class DiorMMDetDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: Split,
        target_size: int,
        bbox_orientation: BBoxOrientation = "horizontal",
        use_obb: bool = False,
        download: bool = False,
        checksum: bool = False,
        to_neg_1_1: bool = False,
        is_train: bool = True,
    ) -> None:
        del download, checksum
        if use_obb and bbox_orientation != "oriented":
            raise ValueError("use_obb=True requires bbox_orientation='oriented'.")

        self.root = Path(root)
        self.split = split
        self.bbox_orientation = bbox_orientation
        self.use_obb = use_obb
        self.class_names = DIOR_CLASSES
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.records = _build_records(root=self.root, split=split, bbox_orientation=bbox_orientation)
        self.transforms = build_default_transforms(target_size=target_size, to_neg_1_1=to_neg_1_1, is_train=is_train)

    def __len__(self) -> int:
        return len(self.records)

    def _load_image_tensor(self, image_path: Path) -> torch.Tensor:
        try:
            from PIL import Image
        except Exception as exc:  # pragma: no cover
            raise ImportError("Loading DIOR images requires Pillow.") from exc

        image = Image.open(image_path).convert("RGB")
        image_np = np.asarray(image)
        return torch.from_numpy(image_np.copy())

    def _build_bboxes(self, sample: dict[str, Any]) -> torch.Tensor:
        if self.use_obb:
            polys = sample.get("bbox")
            if polys is None:
                return torch.zeros((0, 5), dtype=torch.float32)
            if hasattr(polys, "tolist"):
                polys = polys.tolist()
            obbs = dota_poly_to_obb_le90(polys)
            if not obbs:
                return torch.zeros((0, 5), dtype=torch.float32)
            return torch.as_tensor(obbs, dtype=torch.float32)

        boxes = sample.get("bbox_xyxy")
        if boxes is None:
            return torch.zeros((0, 4), dtype=torch.float32)
        if hasattr(boxes, "detach"):
            boxes_t = boxes.float()
        else:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
        if boxes_t.ndim == 1:
            boxes_t = boxes_t.unsqueeze(0)
        return boxes_t

    def _build_labels(self, sample: dict[str, Any], num_boxes: int) -> torch.Tensor:
        labels = sample.get("labels")
        if labels is None:
            return torch.zeros((num_boxes,), dtype=torch.long)
        if hasattr(labels, "detach"):
            labels_t = labels.long()
        else:
            labels_t = torch.as_tensor(labels, dtype=torch.long)
        if labels_t.ndim == 0:
            labels_t = labels_t.unsqueeze(0)
        if labels_t.numel() == num_boxes:
            return labels_t
        if labels_t.numel() > num_boxes:
            return labels_t[:num_boxes]
        pad = torch.zeros((num_boxes - labels_t.numel(),), dtype=torch.long)
        return torch.cat([labels_t, pad], dim=0)

    def _build_data_sample(self, image: torch.Tensor, bboxes: torch.Tensor, labels: torch.Tensor) -> Any:
        from mmdet.structures import DetDataSample  # type: ignore[import-not-found]
        from mmdet.structures.bbox import HorizontalBoxes  # type: ignore[import-not-found]

        gt_instances = InstanceData()
        if self.use_obb:
            try:
                from mmrotate.structures.bbox import RotatedBoxes  # type: ignore[import-not-found]

                gt_instances.bboxes = RotatedBoxes(bboxes)
            except Exception:
                gt_instances.bboxes = bboxes
        else:
            gt_instances.bboxes = HorizontalBoxes(bboxes)
        gt_instances.labels = labels

        data_sample = DetDataSample()
        height = int(image.shape[-2])
        width = int(image.shape[-1])
        channels = int(image.shape[-3]) if image.ndim >= 3 else 1
        data_sample.set_metainfo(
            {
                "img_shape": (height, width, channels),
                "ori_shape": (height, width, channels),
                "scale_factor": (1.0, 1.0),
            }
        )
        data_sample.gt_instances = gt_instances
        return data_sample

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        image = self._load_image_tensor(record.image_path)
        sample: dict[str, Any] = {"image": image}

        if record.annotation_path is not None:
            if self.bbox_orientation == "horizontal":
                bbox_xyxy, labels = _extract_horizontal_objects(record.annotation_path, self.class_to_idx)
                sample["bbox_xyxy"] = bbox_xyxy
                sample["labels"] = labels
            else:
                bbox_xyxy, polys, labels = _extract_oriented_objects(record.annotation_path, self.class_to_idx)
                sample["bbox_xyxy"] = bbox_xyxy
                sample["bbox"] = polys
                sample["labels"] = labels

        sample = self.transforms(sample)
        transformed_image = sample["image"]
        if not hasattr(transformed_image, "detach"):
            transformed_image = torch.as_tensor(transformed_image, dtype=torch.float32)
        else:
            transformed_image = transformed_image.float()

        bboxes = self._build_bboxes(sample)
        labels = self._build_labels(sample, num_boxes=int(bboxes.shape[0]))
        data_sample = self._build_data_sample(transformed_image, bboxes=bboxes, labels=labels)
        return {"inputs": transformed_image, "data_samples": data_sample}


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=pseudo_collate,
    )

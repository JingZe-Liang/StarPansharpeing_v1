from collections.abc import Callable
from typing import Any, Literal

from torch.utils.data import DataLoader
from torchgeo.datasets import DOTA
import torch
from kornia import augmentation as K
from kornia.constants import DataKey


def _extract_polygon_lists(bboxes: Any) -> list[list[float]]:
    if bboxes is None:
        return []
    if hasattr(bboxes, "tolist"):
        bboxes = bboxes.tolist()
    polys: list[list[float]] = []
    for poly in bboxes:
        if isinstance(poly, (list, tuple)) and len(poly) == 8:
            polys.append([float(v) for v in poly])
    return polys


def detection_collate(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collate detection samples and add OBB in a new field for DOTA oriented boxes."""
    for sample in batch:
        polys = _extract_polygon_lists(sample.get("bbox"))
        if polys:
            sample["bbox_obb_le90"] = dota_poly_to_obb_le90(polys)
    return batch


def dota_poly_to_obb_le90(
    polys: list[list[float]] | list[tuple[float, ...]],
) -> list[tuple[float, float, float, float, float]]:
    """Convert DOTA 8-point polygons to (cx, cy, w, h, angle) with le90 convention.

    Args:
        polys: List of polygons, each polygon is [x1, y1, x2, y2, x3, y3, x4, y4].

    Returns:
        List of tuples (cx, cy, w, h, angle_deg).
    """
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("dota_poly_to_obb_le90 requires opencv-python (cv2).") from exc

    obbs: list[tuple[float, float, float, float, float]] = []
    for poly in polys:
        if len(poly) != 8:
            raise ValueError(f"Expected 8 values per polygon, got {len(poly)}")
        pts = np.array(poly, dtype=np.float32).reshape(4, 2)
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
        if w < h:
            w, h = h, w
            angle = angle - 90.0
        if angle >= 0:
            angle = angle - 180.0
        obbs.append((float(cx), float(cy), float(w), float(h), float(angle)))
    return obbs


def _clip_xyxy(boxes: Any, height: int, width: int) -> Any:
    if hasattr(boxes, "clone"):
        boxes = boxes.clone()
        boxes[..., 0] = boxes[..., 0].clamp(0, width - 1)
        boxes[..., 2] = boxes[..., 2].clamp(0, width - 1)
        boxes[..., 1] = boxes[..., 1].clamp(0, height - 1)
        boxes[..., 3] = boxes[..., 3].clamp(0, height - 1)
        return boxes
    clipped: list[list[float]] = []
    for box in boxes or []:
        if len(box) != 4:
            raise ValueError(f"Expected 4 values per box, got {len(box)}")
        x1, y1, x2, y2 = box
        clipped.append(
            [
                float(max(0, min(width - 1, x1))),
                float(max(0, min(height - 1, y1))),
                float(max(0, min(width - 1, x2))),
                float(max(0, min(height - 1, y2))),
            ]
        )
    return clipped


def _polys_to_keypoints(polys: list[list[float]]) -> tuple[torch.Tensor, int]:
    points: list[list[float]] = []
    for poly in polys:
        if len(poly) != 8:
            raise ValueError(f"Expected 8 values per polygon, got {len(poly)}")
        points.extend(
            [
                [float(poly[0]), float(poly[1])],
                [float(poly[2]), float(poly[3])],
                [float(poly[4]), float(poly[5])],
                [float(poly[6]), float(poly[7])],
            ]
        )
    keypoints = torch.as_tensor(points, dtype=torch.float32).unsqueeze(0)
    return keypoints, len(polys)


def _clip_keypoints(keypoints: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if keypoints.ndim == 2:
        keypoints = keypoints.unsqueeze(0)
    keypoints = keypoints.clone()
    keypoints[..., 0] = keypoints[..., 0].clamp(0, width - 1)
    keypoints[..., 1] = keypoints[..., 1].clamp(0, height - 1)
    return keypoints


def _keypoints_to_polys(keypoints: torch.Tensor, num_polys: int) -> list[list[float]]:
    if keypoints.ndim == 3:
        keypoints = keypoints[0]
    if keypoints.shape[0] < num_polys * 4:
        raise ValueError("Not enough keypoints to rebuild polygons.")
    polys: list[list[float]] = []
    for idx in range(num_polys):
        pts = keypoints[idx * 4 : (idx + 1) * 4]
        coords: list[float] = []
        for x, y in pts.tolist():
            coords.append(float(x))
            coords.append(float(y))
        polys.append(coords)
    return polys


def _poly_areas_from_keypoints(keypoints: torch.Tensor, num_polys: int) -> torch.Tensor:
    if num_polys <= 0:
        return torch.empty((0,), dtype=torch.float32)
    if keypoints.ndim == 3:
        keypoints = keypoints[0]
    pts = keypoints.view(num_polys, 4, 2)
    x = pts[:, :, 0]
    y = pts[:, :, 1]
    x_next = torch.roll(x, shifts=-1, dims=1)
    y_next = torch.roll(y, shifts=-1, dims=1)
    area = 0.5 * torch.abs(torch.sum(x * y_next - y * x_next, dim=1))
    return area


def _filter_labels(labels: Any, mask: torch.Tensor) -> Any:
    if labels is None:
        return labels
    if hasattr(labels, "shape"):
        return labels[mask]
    labels_list = list(labels)
    return [labels_list[i] for i, keep in enumerate(mask.tolist()) if keep]


def build_default_transforms(
    target_size: int = 512, to_neg_1_1: bool = False
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build a default transformer that uses square RandomResizedCrop on image and boxes."""

    def _build_aug(keys: list[DataKey]) -> K.AugmentationSequential:
        return K.AugmentationSequential(
            K.RandomResizedCrop(
                (target_size, target_size),
                scale=(0.9, 1.0),
                ratio=(1.0, 1.0),
                p=1.0,
            ),
            data_keys=keys,
            same_on_batch=True,
        )

    aug_img = _build_aug([DataKey.IMAGE])
    aug_img_box = _build_aug([DataKey.IMAGE, DataKey.BBOX_XYXY])
    aug_img_kp = _build_aug([DataKey.IMAGE, DataKey.KEYPOINTS])
    aug_img_box_kp = _build_aug([DataKey.IMAGE, DataKey.BBOX_XYXY, DataKey.KEYPOINTS])

    def _transform(sample: dict[str, Any]) -> dict[str, Any]:
        image = sample.get("image")

        if image is None:
            return sample

        image = image / 255.0
        if to_neg_1_1:
            image = image * 2.0 - 1.0

        if hasattr(image, "detach"):
            image_t = image.float()
        else:
            image_t = torch.as_tensor(image, dtype=torch.float32)

        if image_t.ndim == 2:
            image_t = image_t.unsqueeze(0)
        if image_t.ndim == 3:
            if image_t.shape[0] not in (1, 3) and image_t.shape[-1] in (1, 3):
                image_t = image_t.permute(2, 0, 1)
            image_b = image_t.unsqueeze(0)
        else:
            image_b = image_t

        bbox_xyxy = sample.get("bbox_xyxy")
        bbox_t: torch.Tensor | None = None
        if bbox_xyxy is not None:
            bbox_t = bbox_xyxy
            if hasattr(bbox_t, "detach"):
                bbox_t = bbox_t.float()
            else:
                bbox_t = torch.as_tensor(bbox_t, dtype=torch.float32)
            if bbox_t.ndim == 2:
                bbox_t = bbox_t.unsqueeze(0)

        polys = _extract_polygon_lists(sample.get("bbox"))
        keypoints_t: torch.Tensor | None = None
        num_polys = 0
        if polys:
            keypoints_t, num_polys = _polys_to_keypoints(polys)

        if bbox_t is not None and keypoints_t is not None:
            image_out, bbox_out, keypoints_out = aug_img_box_kp(image_b, bbox_t, keypoints_t)
            image_out = image_out[0]
            bbox_out = bbox_out[0]
            keypoints_out = keypoints_out[0]
            bbox_out = _clip_xyxy(bbox_out, target_size, target_size)
            keypoints_out = _clip_keypoints(keypoints_out, target_size, target_size)
            valid_xyxy = (bbox_out[:, 2] - bbox_out[:, 0] > 1.0) & (bbox_out[:, 3] - bbox_out[:, 1] > 1.0)
            poly_areas = _poly_areas_from_keypoints(keypoints_out, num_polys)
            valid_poly = poly_areas > 1.0
            keep = valid_xyxy & valid_poly
            sample["bbox_xyxy"] = bbox_out[keep]
            sample["bbox"] = _keypoints_to_polys(keypoints_out, num_polys)
            if keep.numel() != num_polys:
                keep = keep[:num_polys]
            sample["bbox"] = [sample["bbox"][i] for i, k in enumerate(keep.tolist()) if k]
            sample["labels"] = _filter_labels(sample.get("labels"), keep)
        elif bbox_t is not None:
            image_out, bbox_out = aug_img_box(image_b, bbox_t)
            image_out = image_out[0]
            bbox_out = bbox_out[0]
            bbox_out = _clip_xyxy(bbox_out, target_size, target_size)
            keep = (bbox_out[:, 2] - bbox_out[:, 0] > 1.0) & (bbox_out[:, 3] - bbox_out[:, 1] > 1.0)
            sample["bbox_xyxy"] = bbox_out[keep]
            sample["labels"] = _filter_labels(sample.get("labels"), keep)
        elif keypoints_t is not None:
            image_out, keypoints_out = aug_img_kp(image_b, keypoints_t)
            image_out = image_out[0]
            keypoints_out = keypoints_out[0]
            keypoints_out = _clip_keypoints(keypoints_out, target_size, target_size)
            poly_areas = _poly_areas_from_keypoints(keypoints_out, num_polys)
            keep = poly_areas > 1.0
            sample["bbox"] = _keypoints_to_polys(keypoints_out, num_polys)
            if keep.numel() != num_polys:
                keep = keep[:num_polys]
            sample["bbox"] = [sample["bbox"][i] for i, k in enumerate(keep.tolist()) if k]
            sample["labels"] = _filter_labels(sample.get("labels"), keep)
        else:
            image_out = aug_img(image_b)[0]

        sample["image"] = image_out
        return sample

    return _transform


def get_dataloader(
    batch_size: int,
    num_workers: int,
    split: Literal["train", "val"] = "train",
    root: str = "data/Downstreams/DOTA",
    version: Literal["1.0", "1.5", "2.0"] = "2.0",
    bbox_orientation: Literal["horizontal", "oriented"] = "oriented",
    transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    target_size: int = 512,
    download: bool = False,
    checksum: bool = False,
    collate_fn: Callable[[list[dict[str, Any]]], Any] | None = None,
    to_neg_1_1: bool = False,
    **loader_kwargs: Any,
) -> tuple[DOTA, DataLoader]:
    """Get DOTA dataloader for object detection.

    Args:
        batch_size: Batch size for training.
        num_workers: Number of worker processes for data loading.
        split: One of 'train' or 'val'.
        root: Root directory where dataset is stored.
        version: One of '1.0', '1.5', or '2.0'.
        bbox_orientation: One of 'horizontal' or 'oriented'.
        transforms: Optional transform function applied to samples.
        target_size: Target size for the default transformer (square).
        download: If True, download dataset if not found.
        checksum: If True, verify dataset checksums.
        collate_fn: Optional collate function for DataLoader.
        **loader_kwargs: Additional kwargs passed to DataLoader.

    Returns:
        Tuple of (dataset, dataloader).

    Note:
        Sample keys:
        - image: Tensor of shape (3, H, W)
        - bbox: Tensor of shape (N, 8) for oriented boxes
        - bbox_xyxy: Tensor of shape (N, 4) for horizontal boxes
        - labels: Tensor of shape (N,)
    """
    if transforms is None:
        transforms = build_default_transforms(target_size=target_size, to_neg_1_1=to_neg_1_1)

    dataset = DOTA(
        root=root,
        split=split,
        version=version,
        bbox_orientation=bbox_orientation,
        transforms=transforms,
        download=download,
        checksum=checksum,
    )

    if collate_fn is None:
        collate_fn = detection_collate

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=(split == "train"),
        collate_fn=collate_fn,
        **loader_kwargs,
    )

    return dataset, dataloader

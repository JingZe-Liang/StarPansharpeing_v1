from __future__ import annotations

from typing import Any, Literal

import torch
from mmengine.dataset import pseudo_collate
from mmengine.structures import InstanceData
from torch.utils.data import DataLoader, Dataset
from torchgeo.datasets import DOTA

from src.stage2.object_detection.data.DOTA import build_default_transforms, dota_poly_to_obb_le90

Split = Literal["train", "val"]


class DotaMMDetDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: Split,
        version: Literal["1.0", "1.5", "2.0"],
        bbox_orientation: Literal["horizontal", "oriented"],
        target_size: int,
        use_obb: bool,
    ) -> None:
        self.use_obb = use_obb
        self.transforms = build_default_transforms(target_size=target_size)
        self.dataset = DOTA(
            root=root,
            split=split,
            version=version,
            bbox_orientation=bbox_orientation,
            transforms=self.transforms,
            download=False,
            checksum=False,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def _build_bboxes(self, sample: dict[str, Any]) -> torch.Tensor:
        if self.use_obb:
            polys = sample.get("bbox")
            if polys is None:
                return torch.zeros((0, 5), dtype=torch.float32)
            if hasattr(polys, "tolist"):
                polys = polys.tolist()
            obb = dota_poly_to_obb_le90(polys)
            if not obb:
                return torch.zeros((0, 5), dtype=torch.float32)
            return torch.as_tensor(obb, dtype=torch.float32)

        boxes = sample.get("bbox_xyxy")
        if boxes is None:
            return torch.zeros((0, 4), dtype=torch.float32)
        if hasattr(boxes, "detach"):
            return boxes.float()
        return torch.as_tensor(boxes, dtype=torch.float32)

    def _build_data_sample(self, image: torch.Tensor, labels: Any, bboxes: torch.Tensor) -> Any:
        from mmdet.structures import DetDataSample  # type: ignore[import-not-found]
        from mmdet.structures.bbox import HorizontalBoxes  # type: ignore[import-not-found]

        if self.use_obb:
            try:
                from mmrotate.structures.bbox import RotatedBoxes  # type: ignore[import-not-found]
            except Exception as exc:  # pragma: no cover - optional dependency
                raise ImportError("mmrotate is required for OBB training.") from exc
            box_type = RotatedBoxes
        else:
            box_type = HorizontalBoxes

        if hasattr(labels, "detach"):
            labels_t = labels.long()
        else:
            labels_t = torch.as_tensor(labels, dtype=torch.long)

        gt_instances = InstanceData()
        gt_instances.bboxes = box_type(bboxes)
        gt_instances.labels = labels_t

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
        sample = self.dataset[idx]
        image = sample["image"]
        if not hasattr(image, "detach"):
            image = torch.as_tensor(image, dtype=torch.float32)

        bboxes = self._build_bboxes(sample)
        labels = sample.get("labels", torch.zeros((0,), dtype=torch.long))
        data_sample = self._build_data_sample(image, labels, bboxes)
        return {"inputs": image, "data_samples": data_sample}


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

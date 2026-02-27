from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .vis import draw_detections

MergeNmsType = Literal["nms", "soft_nms"]


class LargeImageTiledInferencer:
    def __init__(
        self,
        predictor: Any,
        patch_size: int = 1024,
        patch_overlap_ratio: float = 0.25,
        batch_size: int = 1,
        merge_iou_thr: float = 0.5,
        merge_nms_type: MergeNmsType = "nms",
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if not (0.0 <= patch_overlap_ratio < 1.0):
            raise ValueError(f"patch_overlap_ratio must be in [0, 1), got {patch_overlap_ratio}")
        self.predictor = predictor
        self.patch_size = patch_size
        self.patch_overlap_ratio = patch_overlap_ratio
        self.batch_size = batch_size
        self.merge_iou_thr = merge_iou_thr
        self.merge_nms_type = merge_nms_type

    def _load_image(self, image: str | Path | np.ndarray) -> np.ndarray:
        if isinstance(image, (str, Path)):
            try:
                from PIL import Image
            except Exception as exc:  # pragma: no cover
                raise ImportError("Reading image from path requires Pillow.") from exc
            pil_image = Image.open(str(image)).convert("RGB")
            return np.asarray(pil_image)
        if not isinstance(image, np.ndarray):
            raise TypeError(f"image must be path or ndarray, got {type(image)}")
        return image

    def _call_predictor(self, images: list[np.ndarray]) -> list[Any]:
        if callable(self.predictor):
            output = self.predictor(images)
        else:
            try:
                from mmdet.apis import inference_detector  # type: ignore[import-not-found]
            except Exception as exc:  # pragma: no cover
                raise TypeError("predictor must be callable or a valid mmdet model instance.") from exc
            output = inference_detector(self.predictor, images)

        if isinstance(output, list):
            return output
        return [output]

    def _as_numpy_patch(self, patch: Any) -> np.ndarray:
        if isinstance(patch, np.ndarray):
            return patch
        if hasattr(patch, "convert"):
            return np.asarray(patch)
        raise TypeError(f"Unsupported patch type from sahi: {type(patch)}")

    def infer(self, image: str | Path | np.ndarray) -> Any:
        try:
            from PIL import Image
            from sahi.slicing import slice_image
        except Exception as exc:  # pragma: no cover
            raise ImportError('Please install sahi: "pip install -U sahi".') from exc
        try:
            from mmdet.utils.large_image import merge_results_by_nms  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover
            raise ImportError("merge_results_by_nms requires mmdet.") from exc

        img = self._load_image(image)
        height, width = img.shape[:2]
        slice_input = Image.fromarray(img)
        sliced = slice_image(
            slice_input,
            slice_height=self.patch_size,
            slice_width=self.patch_size,
            auto_slice_resolution=False,
            overlap_height_ratio=self.patch_overlap_ratio,
            overlap_width_ratio=self.patch_overlap_ratio,
        )

        slice_results: list[Any] = []
        start = 0
        while True:
            end = min(start + self.batch_size, len(sliced))
            batch_images = [self._as_numpy_patch(patch) for patch in sliced.images[start:end]]
            slice_results.extend(self._call_predictor(batch_images))
            if end >= len(sliced):
                break
            start += self.batch_size

        merged = merge_results_by_nms(
            slice_results,
            sliced.starting_pixels,
            src_image_shape=(height, width),
            nms_cfg={"type": self.merge_nms_type, "iou_threshold": self.merge_iou_thr},
        )
        return merged

    def _to_drawable_image(self, image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        out = image.astype(np.float32)
        max_val = float(out.max()) if out.size > 0 else 1.0
        if max_val > 1.0:
            out = out / 255.0
        return out

    def visualize(
        self,
        image: str | Path | np.ndarray,
        result: Any,
        class_names: list[str] | None = None,
        score_thr: float = 0.0,
        line_width: int = 4,
        out_file: str | Path | None = None,
    ) -> Any:
        img = self._load_image(image)
        draw_img = self._to_drawable_image(img)

        pred_instances = result.pred_instances
        boxes = pred_instances.bboxes
        if hasattr(boxes, "tensor"):
            boxes = boxes.tensor
        labels = pred_instances.labels
        scores = getattr(pred_instances, "scores", None)

        if scores is not None and score_thr > 0.0:
            keep = scores >= score_thr
            boxes = boxes[keep]
            labels = labels[keep]

        if hasattr(boxes, "shape") and int(boxes.shape[-1]) == 5:
            vis_img = draw_detections(
                draw_img,
                boxes_obb=boxes,
                labels_obb=[int(v) for v in labels.tolist()],
                class_names=class_names,
                line_width=line_width,
            )
        else:
            vis_img = draw_detections(
                draw_img,
                boxes_xyxy=boxes,
                labels_xyxy=[int(v) for v in labels.tolist()],
                class_names=class_names,
                line_width=line_width,
            )

        if out_file is not None:
            out_path = Path(out_file)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            vis_img.save(out_path)
        return vis_img

    def infer_and_visualize(
        self,
        image: str | Path | np.ndarray,
        class_names: list[str] | None = None,
        score_thr: float = 0.0,
        line_width: int = 4,
        out_file: str | Path | None = None,
    ) -> tuple[Any, Any]:
        merged_result = self.infer(image)
        vis_img = self.visualize(
            image=image,
            result=merged_result,
            class_names=class_names,
            score_thr=score_thr,
            line_width=line_width,
            out_file=out_file,
        )
        return merged_result, vis_img

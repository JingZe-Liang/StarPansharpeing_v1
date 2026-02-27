from __future__ import annotations

from typing import Any

import math


def _lazy_import_pil() -> tuple[Any, Any, Any]:
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("vis requires Pillow. Install pillow to enable drawing.") from exc
    return Image, ImageDraw, ImageFont


def _to_pil_image(image: Any) -> Any:
    Image, _, _ = _lazy_import_pil()
    if isinstance(image, Image.Image):
        return image
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise ImportError("vis requires numpy when input is not a PIL image.") from exc
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    image_min = float(image.min()) if hasattr(image, "min") else 0.0
    image_max = float(image.max()) if hasattr(image, "max") else 1.0
    if image_min < -0.01:
        image = (image + 1.0) / 2.0
    image = image * 255.0
    if image.ndim == 3 and image.shape[0] in {1, 3}:
        image = image.transpose(1, 2, 0)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return Image.fromarray(image)


def _ensure_list(data: Any, expected_len: int | None = None) -> list[list[float]]:
    if data is None:
        return []
    if hasattr(data, "tolist"):
        data = data.tolist()
    out: list[list[float]] = []
    for item in data:
        if isinstance(item, (list, tuple)):
            out.append([float(v) for v in item])
    if expected_len is not None:
        for item in out:
            if len(item) != expected_len:
                raise ValueError(f"Expected {expected_len} values, got {len(item)}")
    return out


def _get_palette() -> list[tuple[int, int, int]]:
    return [
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (250, 190, 212),
        (0, 128, 128),
        (220, 190, 255),
        (170, 110, 40),
        (255, 250, 200),
        (128, 0, 0),
        (170, 255, 195),
        (128, 128, 0),
        (255, 215, 180),
        (0, 0, 128),
        (128, 128, 128),
    ]


def _label_to_color(label: int, colors: list[tuple[int, int, int]] | None) -> tuple[int, int, int]:
    palette = colors or _get_palette()
    return palette[int(label) % len(palette)]


def _format_label(label: int, class_names: list[str] | None) -> str:
    if class_names is None:
        return str(label)
    if 0 <= label < len(class_names):
        return class_names[label]
    return str(label)


def _obb_to_polygon(cx: float, cy: float, w: float, h: float, angle_deg: float) -> list[tuple[float, float]]:
    angle = math.radians(angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    dx = w / 2.0
    dy = h / 2.0
    corners = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]
    pts: list[tuple[float, float]] = []
    for x, y in corners:
        rx = x * cos_a - y * sin_a + cx
        ry = x * sin_a + y * cos_a + cy
        pts.append((rx, ry))
    return pts


def draw_detections(
    image: Any,
    boxes_xyxy: Any | None = None,
    labels_xyxy: list[int] | None = None,
    boxes_obb: Any | None = None,
    labels_obb: list[int] | None = None,
    class_names: list[str] | None = None,
    colors: list[tuple[int, int, int]] | None = None,
    line_width: int = 4,
) -> Any:
    """Draw horizontal and oriented detections with labels.

    Args:
        image: Input image as Tensor/ndarray/PIL. Expected value range is [0, 1].
            Supported shapes: (H, W), (C, H, W), or (H, W, C). Single-channel is
            converted to RGB for visualization.
        boxes_xyxy: Horizontal boxes in [x1, y1, x2, y2]. Tensor/ndarray/list.
        labels_xyxy: Class indices for horizontal boxes.
        boxes_obb: Oriented boxes in (cx, cy, w, h, angle_deg). Tensor/ndarray/list.
            Angle is in degrees (le90 by default in this project).
        labels_obb: Class indices for oriented boxes.
        class_names: Optional class name list, used to render label text.
        colors: Optional custom color palette as list of RGB tuples.
        line_width: Line width in pixels for box outlines.

    Returns:
        PIL Image with rendered boxes and labels.
    """
    Image, ImageDraw, ImageFont = _lazy_import_pil()
    img = _to_pil_image(image)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    xyxy = _ensure_list(boxes_xyxy, expected_len=4)
    obb = _ensure_list(boxes_obb, expected_len=5)

    if labels_xyxy is None:
        labels_xyxy = [0] * len(xyxy)
    if labels_obb is None:
        labels_obb = [0] * len(obb)

    for box, label in zip(xyxy, labels_xyxy, strict=False):
        color = _label_to_color(label, colors)
        draw.rectangle(box, outline=color, width=line_width)
        text = _format_label(label, class_names)
        draw.text((box[0] + 2, box[1] + 2), text, fill=color, font=font)

    for box, label in zip(obb, labels_obb, strict=False):
        cx, cy, w, h, angle = box
        color = _label_to_color(label, colors)
        poly = _obb_to_polygon(cx, cy, w, h, angle)
        draw.line(poly + [poly[0]], fill=color, width=line_width)
        text = _format_label(label, class_names)
        draw.text((poly[0][0] + 2, poly[0][1] + 2), text, fill=color, font=font)

    return img

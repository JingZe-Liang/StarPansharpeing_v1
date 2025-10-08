import argparse
import json
import math
import os
import sys
from typing import Callable, Literal, cast, no_type_check

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.stage2.generative.tools.conditions.caption.qwen25VL import get_qwen25vl_model
from src.stage2.generative.tools.conditions.hed import HEDdetector
from src.stage2.generative.tools.conditions.mlsd import MLSDdetector
from src.stage2.generative.tools.conditions.sketch import SketchDetector
from src.stage2.generative.tools.conditions.uniformer import SAMDetector
from src.utilities.config_utils import function_config_to_basic_types
from src.utilities.logging import log_print
from src.utilities.train_utils.visualization import get_rgb_image


def rsshow(I, scale=0.005):
    low, high = np.quantile(I, [scale, 1 - scale])
    I[I > high] = high
    I[I < low] = low
    I = (I - low) / (high - low)
    return I


class UnifiedAnnotator:
    """A unified wrapper for various image condition annotators."""

    def __init__(self, annotation: str, device=None):
        """
        Initializes all the required annotators.

        Args:
            annotation (str): The type of annotation to use. Supported values:
                              'hed', 'segmentation', 'sketch', 'mlsd', 'caption', 'content'.
            device (str, optional): The torch device to use, e.g., 'cuda:0' or 'cpu'.
                                    If None, it will be set automatically. Defaults to None.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # Set device for torch operations if a specific GPU is requested
        if "cuda" in str(device):
            gpu_id = device.split(":")[-1]
            if gpu_id.isdigit():
                device = f"cuda:{gpu_id}"
        self.device = device if device is not None else "cpu"

        self.init_annotator(annotation, self.device)

    def init_annotator(self, annotation: str, device: str | None = None):
        self.annotation = annotation.lower()
        match self.annotation:
            case "hed":
                self.annotator = HEDdetector()
            case "segmentation":
                self.annotator = SAMDetector()
            case "sketch":
                self.annotator = SketchDetector()
            case "mlsd":
                self.annotator = MLSDdetector()
            case "caption":
                self._ann_inner_model, self.annotator = get_qwen25vl_model(encode=True)
                self.annotator = cast(Callable[[np.ndarray | str], str], self.annotator)
            case "content":
                pass

        if hasattr(self, "annotator") and isinstance(self.annotator, torch.nn.Module):
            self.annotator = self.annotator.cuda()
        self._del = False

    def release_gpu(self, device=None):
        self._del = True
        if device is None:
            # del the annotators
            self.annotator = None
            if hasattr(self, "_ann_inner_model"):
                self._ann_inner_model = None
        elif isinstance(device, (str, torch.device)):
            if isinstance(self.annotator, torch.nn.Module):
                self.annotator = self.annotator.to(device)
            if hasattr(self, "_ann_inner_model") and isinstance(
                self._ann_inner_model, torch.nn.Module
            ):
                self._ann_inner_model = self._ann_inner_model.to(device)
        else:
            raise ValueError(
                f"Invalid device type: {type(device)}. Expected str or torch.device."
            )

    def reallocate_gpu(self, device=None):
        if self._del:
            self.init_annotator(self.annotation, device)

    @no_type_check
    def __call__(self, image: np.ndarray | str) -> np.ndarray | str:
        """
        Processes an image to generate a condition map.

        Args:
            image (np.ndarray): Input image as a NumPy array (RGB, HWC).
            condition_name (str): The name of the condition to generate.
                                  Supported: 'hed', 'segmentation', 'sketch', 'mlsd', 'content'.

        Raises:
            ValueError: If the condition_name is not supported.

        Returns:
            np.ndarray: The condition image as a NumPy array.
        """
        img_path: str | None = image if isinstance(image, str) else None
        if img_path is not None:
            # Load image from path
            image = self.read_img(img_path)

        condition_name = self.annotation
        if condition_name == "hed":
            return self.annotator(image)
        elif condition_name == "segmentation":
            _, seg_labels = self.annotator(image)
            image_segment_with_color = np.zeros_like(image)
            seg_labels = seg_labels[0].astype(np.uint8)
            max_label = int(seg_labels.max())
            for i in range(1, max_label + 1):
                mask = seg_labels == i
                if np.any(mask):
                    image_segment_with_color[mask] = image[mask].mean(0)
                else:
                    image_segment_with_color[mask] = 0
            return image_segment_with_color
        elif condition_name == "sketch":
            return self.annotator(image)
        elif condition_name == "mlsd":
            # Using default values from the original script
            return self.annotator(image, 0.05, 20)
        elif condition_name == "caption":
            return self.annotator(img_path if img_path is not None else image)
        elif condition_name == "content":
            return image
        else:
            raise ValueError(f"Condition {condition_name} is not supported.")

    def read_img(self, path: str):
        assert os.path.exists(path), "Image path does not exist."
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = rsshow(np.array(image), 0)
        image = (image * 255).astype(np.uint8)

        return path


annotators: dict[str, Callable] = {}


@function_config_to_basic_types
def prepare_condition_from_webdataset(
    ds,
    conditions: str | list[str] = "all",
    rgb_channels: list[int] | Literal["mean"] | None = None,
    device="cuda",
    to_pil: bool = True,
    resume_from: str | None = None,
    use_linstretch: bool = False,
):
    if conditions == "all":
        conditions = ["hed", "segmentation", "sketch", "mlsd", "caption"]

    global annotators
    for cond in conditions:
        if cond not in annotators:
            annotators[cond] = UnifiedAnnotator(cond, device)
            log_print("annotator loaded: {}".format(cond))
        else:
            log_print("annotator already loaded: {}".format(cond))

    _resumed_flag = resume_from is None  # 如果 resume_from 为 None，直接全部处理
    for sample in ds:
        assert len(sample["__key__"]) == 1, (
            "WebDataset sample key must be a single string."
        )

        key = sample["__key__"][0]
        if resume_from is not None and not _resumed_flag:
            if key != resume_from:
                # log_print(f"Skipping {key} until resume point {resume_from}.", "debug")
                yield sample, None
                continue
            else:
                _resumed_flag = True
                log_print(f"Resuming from {key}.", "info")

        img = sample["img"]

        # if the image is too large, we need to resize it
        if math.prod(tuple(_orig_size := img.shape[-2:])) > 1024 * 1024:
            _orig_size = np.array(tuple(_orig_size))
            l_max_i = np.argmax(_orig_size)
            l_max = _orig_size[l_max_i]
            ratio = 1024 / l_max
            log_print(
                f"Image is too large {tuple(_orig_size)}, resizing to {tuple(_orig_size * ratio)}...",
                "debug",
            )
            img = torch.nn.functional.interpolate(
                img,
                antialias=True,
                scale_factor=ratio,
                mode="bilinear",
                # size=(1024, 1024),
            )

        # img is batched
        assert img.shape[0] == 1, "Only single image input is supported."
        img = get_rgb_image(img, rgb_channels, use_linstretch=use_linstretch)
        img = img.squeeze(0).cpu().numpy()  # remove batch dim
        img = img.transpose(1, 2, 0)  # Convert to HWC format
        # not to_neg_1_1
        img = (img * 255.0).astype(np.uint8)
        assert img.ndim == 3 and img.shape[2] == 3, "Image must be RGB format."

        ret: dict[str, np.ndarray | Image.Image | str] = {}
        for cond in conditions:
            with torch.inference_mode():
                out = annotators[cond](img)

            # is numpy array
            if hasattr(out, "shape") and tuple(out.shape[:2]) != tuple(_orig_size):
                out = cv2.resize(
                    out,
                    (_orig_size[1], _orig_size[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

            ret[cond] = (
                Image.fromarray(out) if to_pil and isinstance(out, np.ndarray) else out
            )

        yield sample, ret


if __name__ == "__main__":
    # An example
    torch.cuda.set_device(1)
    hed_annotator = HEDdetector()
    sam_annotator = SAMDetector()
    sketch_annotator = SketchDetector()
    mlsd_annotator = MLSDdetector()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datadir",
        type=str,
        default="data/YuZhongDataset/LoveDA/Train/Rural/images_png",
    )
    parser.add_argument(
        "--savedir",
        type=str,
        default="data/YuZhongDataset/LoveDA/Train/Rural/conditions",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default="all",
        choices=["all", "hed", "segmentation", "sketch", "mlsd", "content"],
    )
    opts = parser.parse_args()

    data_dir = opts.datadir
    fns = [t for t in os.listdir(data_dir) if t.endswith(".png") or t.endswith(".jpg")]

    save_dir = opts.savedir
    os.makedirs(save_dir, exist_ok=True)

    if opts.conditions == "all":
        conditions_names = ["hed", "segmentation", "sketch", "mlsd", "content"]
    else:
        conditions_names = [opts.conditions]

    for fn in tqdm(fns):
        fn_path = os.path.join(data_dir, fn)
        image = cv2.imread(fn_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = rsshow(np.array(image), 0)
        image = (image * 255).astype(np.uint8)

        save_path = os.path.join(save_dir, fn.split(".")[0])
        os.makedirs(save_path, exist_ok=True)

        for condition_name in conditions_names:
            if condition_name == "hed":
                condition = hed_annotator(image)
            elif condition_name == "segmentation":
                image_seg, seg_labels = sam_annotator(image)
                image_segment_with_color = np.zeros_like(image)
                seg_labels = seg_labels[0].astype(np.uint8)
                for i in range(1, seg_labels.max() + 1):
                    mask = seg_labels == i
                    image_segment_with_color[mask] = image[mask].mean(0)
                condition = image_segment_with_color
            elif condition_name == "sketch":
                condition = sketch_annotator(image)
            elif condition_name == "mlsd":
                condition = mlsd_annotator(image, 0.05, 20)
            elif condition_name == "content":
                condition = image
            else:
                raise ValueError("Invalid Condition")

            Image.fromarray(condition).save(
                os.path.join(save_path, condition_name + ".png")
            )

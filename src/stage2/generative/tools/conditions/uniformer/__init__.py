import os

import cv2
import numpy as np

from ..condition_utils import annotator_ckpts_path
from ..download_util import load_file_from_url
from .mmseg.apis import init_segmentor
from .mmseg.core.evaluation import get_palette


def convert_color_factory(src, dst):
    code = getattr(cv2, f"COLOR_{src.upper()}2{dst.upper()}")

    def convert_color(img):
        out_img = cv2.cvtColor(img, code)
        return out_img

    convert_color.__doc__ = f"""Convert a {src.upper()} image to {dst.upper()}
        image.

    Args:
        img (ndarray or str): The input image.

    Returns:
        ndarray: The converted {dst.upper()} image.
    """

    return convert_color


bgr2rgb = convert_color_factory("bgr", "rgb")


def show_result_pyplot(
    model,
    img,
    result,
    palette=None,
    fig_size=(15, 10),
    opacity=0.5,
    title="",
    block=True,
):
    """Visualize the segmentation results on the image.

    Args:
        model (nn.Module): The loaded segmentor.
        img (str or np.ndarray): Image filename or loaded image.
        result (list): The segmentation result.
        palette (list[list[int]]] | None): The palette of segmentation
            map. If None is given, random palette will be generated.
            Default: None
        fig_size (tuple): Figure size of the pyplot figure.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
        title (str): The title of pyplot figure.
            Default is ''.
        block (bool): Whether to block the pyplot figure.
            Default is True.
    """
    if hasattr(model, "module"):
        model = model.module
    img = model.show_result(img, result, palette=palette, show=False, opacity=opacity)

    return bgr2rgb(img)


class SAMDetector:
    def __init__(self):
        try:
            from segment_anything import (
                SamAutomaticMaskGenerator,
                SamPredictor,
                sam_model_registry,
            )
        except ImportError:
            raise ImportError("Please install the segment-anything package")
        remote_model_path = "https://huggingface.co/ybelkada/segment-anything/tree/main/checkpoints/sam_vit_h_4b8939.pth"
        modelpath = os.path.join(annotator_ckpts_path, "sam_vit_h_4b8939.pth")

        if not os.path.exists(modelpath):
            load_file_from_url(remote_model_path, model_dir=annotator_ckpts_path)

        sam = sam_model_registry["default"](checkpoint=modelpath)
        sam.to(device="cuda")
        mask_generator = SamAutomaticMaskGenerator(sam)
        self.mask_generator = mask_generator

        config_file = os.path.join(
            os.path.dirname(annotator_ckpts_path),
            "uniformer",
            "exp",
            "upernet_global_small",
            "config.py",
        )
        modelpath = os.path.join(annotator_ckpts_path, "upernet_global_small.pth")
        self.model = init_segmentor(config_file, modelpath).cuda()

    def __call__(self, img):
        masks = self.mask_generator.generate(img)
        result = np.zeros((1, img.shape[0], img.shape[1]))
        for i, mask in enumerate(masks):
            mask = mask["segmentation"]
            result[0, mask] = i + 1
        res_img = show_result_pyplot(
            self.model, img, result, get_palette("ade"), opacity=1
        )
        return res_img, result

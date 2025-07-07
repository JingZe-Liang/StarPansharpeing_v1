import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ..condition_utils import annotator_ckpts_path
from ..hed import HEDdetector
from .model import module

remote_model_path = "https://github.com/aidreamwin/sketch_simplification_pytorch/releases/download/model/model_gan.pth"


class SketchDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model_path = os.path.join(annotator_ckpts_path, "model_gan.pth")
        self.immean, self.imstd = 0.9664114577640158, 0.0858381272736797
        self.model = module.Net()
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.hub.load_state_dict_from_url(
                remote_model_path, model_dir=os.path.dirname(model_path), progress=True
            )
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.hed_func = HEDdetector()

    def forward_img(self, pre_img):
        img = 255 - self.hed_func(pre_img)
        assert img.ndim == 2
        img = Image.fromarray(img).convert("L")
        w, h = img.size[0], img.size[1]
        pw = 8 - (w % 8) if w % 8 != 0 else 0
        ph = 8 - (h % 8) if h % 8 != 0 else 0
        data = ((transforms.ToTensor()(img) - self.immean) / self.imstd).unsqueeze(0)
        if pw != 0 or ph != 0:
            data = torch.nn.ReplicationPad2d((0, pw, 0, ph))(data).data
        data = data.float().cuda()
        with torch.no_grad():
            pred = self.model.cuda().forward(data).float()[0][0]
            pred = pred.detach().cpu().numpy()
            pred = cv2.resize(pred, (w, h)) * 255
            pred = pred.astype(np.uint8)
        return pred

    def forward_batch(self, input_images: torch.Tensor):
        # [bs, c, h, w], 0 .. 1
        assert input_images.ndim == 4 and input_images.shape[1] == 3

        img = 1 - self.hed_func(input_images, "tensor")
        # to gray image
        img = img.mean(1, keepdim=True)  # [bs, 1, h, w]
        h, w = img.shape[-2:]
        pw = 8 - (w % 8) if w % 8 != 0 else 0
        ph = 8 - (h % 8) if h % 8 != 0 else 0

        data = img.cuda().float()
        data = (img - self.immean) / self.imstd  # [bs, 1, h, w]
        if pw != 0 or ph != 0:
            data = torch.nn.ReplicationPad2d((0, pw, 0, ph))(data).data

        pred = self.model(data).float()  # [bs, 1, h, w]
        # resize back
        pred = torch.nn.functional.interpolate(
            input=pred,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        return pred  # [bs, 1, h, w] with values in [0, 1]

    def forward(self, input_image, tt="array"):
        if tt == "array":
            return self.forward_img(input_image)
        elif tt == "tensor":
            return self.forward_batch(input_image)
        else:
            raise ValueError(f"Unsupported type: {tt}")

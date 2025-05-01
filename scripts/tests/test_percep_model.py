import sys

sys.path.append(__file__[: __file__.find("scripts")])
import math

import einops
import torch
from PIL import Image

import src.stage1.perception_models.core.vision_encoder.pe as pe
import src.stage1.perception_models.core.vision_encoder.transforms as transforms
from src.stage1.utilities.losses.repa.feature_pca import (
    feature_pca_cuml,
)

model_name = "PE-Spatial-G14-448"
model = pe.VisionTransformer.from_config(
    model_name,
    pretrained=True,
    local_dir="/HardDisk/ZiHanCao/checkpoints/perception_models",
)  # Loads from HF
preprocess = transforms.get_image_transform(577)
image = preprocess(Image.open("scripts/tests/imgs/rs_dcf2019_demo.png")).unsqueeze(0)
is_giga = "G14" in model_name

torch.cuda.set_device(1)

with torch.no_grad():
    # regional features
    out = model.forward_features(image)
    print(out.shape)  # 1, 577, 1024. 577=24*24+1(cls); 336/14=24

    if not is_giga:
        h = w = int(math.sqrt(out.shape[1] - 1))
        out_img = out[:, 1:]
    else:
        h = w = int(math.sqrt(out.shape[1]))
        out_img = out
    out_img = einops.rearrange(out_img, "b (h w) c -> b c h w", h=h, w=w)
    print(out_img.shape)  # 1, 1024, 24, 24

    # plot images
    out_img_rgb = feature_pca_cuml(out_img.cuda(), pca_k=3)
    out_img_rgb = out_img_rgb.permute(0, 2, 3, 1).cpu().numpy()[0]
    out_img_rgb = (out_img_rgb - out_img_rgb.min()) / (
        out_img_rgb.max() - out_img_rgb.min()
    )  # normalize to [0,1]
    out_img_rgb = (out_img_rgb * 255).astype("uint8")
    out_img_rgb = Image.fromarray(out_img_rgb)
    out_img_rgb.resize((512, 512), resample=Image.Resampling.LANCZOS).save(
        "cat_memes_pca.jpg"
    )

    # image features (1)
    out = model(image)
    print(out.shape)  # 1, 1024

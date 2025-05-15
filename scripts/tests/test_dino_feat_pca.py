import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import torch
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from stage1.utilities.losses.repa.feature_pca import feature_pca_cuml

# read image
img = Image.open(
    "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/scripts/tests/imgs/rs_mmseg_demo.jpg"
)
_next_pow_of_x = lambda x, y: int(np.ceil(x / y)) * y
_sz = _next_pow_of_x(512, 14)
img = img.resize((_sz, _sz))
img = img.convert("RGB")
img = torch.as_tensor(np.array(img))

# to batch
img = img.permute(-1, 0, 1)[None].cuda()  # [1, 3, 224, 224]

# normalize
img = img / 255.0
_mean = torch.tensor(IMAGENET_DEFAULT_MEAN)[None, :, None, None].cuda()
_std = torch.tensor(IMAGENET_DEFAULT_STD)[None, :, None, None].cuda()
img = (img - _mean) / _std

# dino
_model_kwargs = {"interpolate_antialias": True}
repa_encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").cuda()
repa_encoder.eval()
repa_encoder.image_size = 224

# encode
with torch.no_grad():
    repa_feat = repa_encoder.get_intermediate_layers(img, 1, reshape=True, norm=True)[0]
print(repa_feat.shape)

# pca
pca = feature_pca_cuml(repa_feat, pca_k=3)
print(pca.shape)


# plot

pca_vis = pca.detach().cpu().numpy().squeeze(0).transpose(1, 2, 0)

# norm
pca_vis = (pca_vis - np.min(pca_vis)) / (np.max(pca_vis) - np.min(pca_vis))
pca_vis = (pca_vis * 255).astype(np.uint8)

plt.imshow(pca_vis)
plt.savefig("pca_vis.png")

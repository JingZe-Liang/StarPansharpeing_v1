from copy import deepcopy

import matplotlib.pyplot as plt
import open_clip
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models.feature_extraction import create_feature_extractor

model_name = "RN50"  # 'RN50' or 'ViT-B-32' or 'ViT-L-14'
model, _, preprocess = open_clip.create_model_and_transforms(model_name)
tokenizer = open_clip.get_tokenizer(model_name)

torch.cuda.set_device(1)

ckpt = torch.load(
    f"/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/src/stage1/utilities/losses/remote_clip_v2/checkpoint/RemoteCLIP-RN50.pt",
    map_location="cpu",
)
message = model.load_state_dict(ckpt)
print(message)

img = Image.open(
    "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/src/stage1/utilities/losses/remote_clip_v2/demo/imgs/demo_img1.png"
)
img = preprocess(img).unsqueeze(0).cuda()

model = model.cuda().eval()
# (0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
visual = deepcopy(model.visual)
del model
print(visual.__class__)


class Norm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return nn.functional.normalize(x, dim=-1)


# wrap visual model
class WrappedClipVisual(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.body = model
        self.final_norm = Norm()

    def forward(self, x):
        x = self.body(x)
        x = self.final_norm(x)
        return x


visual = WrappedClipVisual(visual)


visual = create_feature_extractor(
    visual,
    return_nodes={
        "body.layer1": "features_1",
        "body.layer2": "features_2",
        "body.layer4": "features_4",
        # "final_norm": "logits",
    },
)

features = visual(img)

# 为每个layer创建单独的图
for name, feat in features.items():
    # 获取前32个通道的特征
    feat_channels = feat[0, :32].detach().cpu().numpy()  # [32, H, W]

    # 计算子图布局的行列数
    n_rows = 4
    n_cols = 8

    # 创建新的图
    plt.figure(figsize=(20, 10))

    # 绘制32个通道的特征图
    for i in range(32):
        plt.subplot(n_rows, n_cols, i + 1)
        im = plt.imshow(feat_channels[i])
        plt.colorbar(im)
        plt.title(f"Channel {i}")
        plt.axis("off")

    plt.suptitle(f"Feature Maps for {name}")
    plt.tight_layout()
    plt.savefig(
        f"/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/src/stage1/utilities/losses/remote_clip_v2/demo/feats/demo_1_feature_visualization_{name}.png"
    )
    plt.close()

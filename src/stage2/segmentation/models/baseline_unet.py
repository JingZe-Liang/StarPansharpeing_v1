# taken from https://github.com/x-ytong/DPA/blob/main/unet
"""Parts of the U-Net model"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


def test_unet_forward() -> None:
    """Test UNet forward pass with random input.

    This function tests the UNet model by:
    1. Creating a model instance with specified parameters
    2. Generating random input tensor
    3. Performing forward pass
    4. Printing model and output information

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    # Model configuration
    n_channels: int = 4
    n_classes: int = 25
    bilinear: bool = True

    # Create model
    net = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)
    state_dict = torch.load(
        "data/Downstreams/5Billion-ChinaCity-Segmentation/pretrained_unet/unet_chengdu.pth.tar", weights_only=False
    )["state_dict"]
    imcomp_keys = net.load_state_dict(state_dict)
    print(f"Loaded done. can not loaded keys: {imcomp_keys}")
    net = net.cuda()

    print(f"Model created successfully")
    print(f"Number of parameters: {sum(p.numel() for p in net.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad):,}")

    # Test with different input sizes
    import tifffile
    from src.data.window_slider import WindowSlider, model_predict_patcher
    from torchvision import transforms
    import PIL
    import numpy as np

    x = PIL.Image.open(
        "data/Downstreams/5Billion-ChinaCity-Segmentation/TestSet/Image__8bit_NirRGB/20190713_032217_1003_3B_AnalyticMS_SR.tif"
    ).convert("CMYK")
    x = np.array(x)

    net.eval()

    # 将16-bit数据归一化到[0, 1]，然后标准化
    mean = torch.tensor([0.577, 0.362, 0.410, 0.372])
    std = torch.tensor([0.235, 0.245, 0.231, 0.230])
    x = torch.as_tensor(x).permute(2, 0, 1)[None].cuda()  # HWC -> NCHW
    # x = x / 1024.0  # 16-bit: 0-65535

    x = x / 255

    # x = normalize(x)
    # transforms.Normalize
    x = (x - mean.view(1, -1, 1, 1).to(x)) / std.view(1, -1, 1, 1).cuda()
    x = x.bfloat16()

    def forward_closure(batch):
        x = batch["img"]
        # Forward pass
        with torch.autocast("cuda", torch.bfloat16):
            with torch.no_grad():
                output = net(x)
        return {"pred_logits": output}

    outputs = model_predict_patcher(
        forward_closure,
        {"img": x},
        patch_keys=["img"],
        merge_keys=["pred_logits"],
        patch_size=512,
        stride=256,
    )
    pred = outputs["pred_logits"][:, 1:]

    import matplotlib.pyplot as plt
    from src.utilities.train_utils.visualization import visualize_segmentation_map

    out_map = pred.argmax(1)
    out_rgb = visualize_segmentation_map(out_map, n_class=24, use_coco_colors=True, to_pil=True)
    out_rgb.save("chengdu_tmp.png")

    # Print output information
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out_map.shape}")
    print("\n✓ All tests passed successfully!")


if __name__ == "__main__":
    test_unet_forward()

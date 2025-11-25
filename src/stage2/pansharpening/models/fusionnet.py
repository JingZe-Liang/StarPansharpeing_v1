# GPL License
# Copyright (C) UESTC
# All Rights Reserved
# @Author  : Xiao Wu
# @reference:


import torch
import torch.nn as nn

from src.utilities.logging import log_print


class loss_with_l2_regularization(nn.Module):
    def __init__(self):
        super(loss_with_l2_regularization, self).__init__()

    def forward(self, criterion, model, weight_decay=1e-5, flag=False):
        regularizations = []
        for k, v in model.named_parameters():
            if "conv" in k and "weight" in k:
                # print(k)
                penality = weight_decay * ((v.data**2).sum() / 2)
                regularizations.append(penality)
                if flag:
                    print("{} : {}".format(k, penality))
        # r = torch.sum(regularizations)

        loss = criterion + sum(regularizations)
        return loss


# -------------Initialization----------------------------------------
def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):  ## initialization for Conv2d
                print("initial nn.Conv2d with var_scale_new: ", m)
                # try:
                #     import tensorflow as tf
                #     tensor = tf.get_variable(shape=m.weight.shape, initializer=tf.variance_scaling_initializer(seed=1))
                #     m.weight.data = tensor.eval()
                # except:
                #     print("try error, run variance_scaling_initializer")
                # variance_scaling_initializer(m.weight)
                # variance_scaling_initializer(m.weight)  # method 1: initialization
                # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  # method 2: initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):  ## initialization for BN
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):  ## initialization for nn.Linear
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


# -------------ResNet Block (One)----------------------------------------
class Resblock(nn.Module):
    def __init__(self):
        super(Resblock, self).__init__()

        channel = 32
        self.conv20 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.conv21 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # x= hp of ms; y = hp of pan
        rs1 = self.relu(self.conv20(x))  # Bsx32x64x64
        rs1 = self.conv21(rs1)  # Bsx32x64x64
        rs = torch.add(x, rs1)  # Bsx32x64x64
        return rs


class FusionNet(nn.Module):
    def __init__(self, spectral_num, channel=32, is_classifier: bool = False):
        super(FusionNet, self).__init__()
        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        self.spectral_num = spectral_num

        self.conv1 = nn.Conv2d(
            in_channels=spectral_num,
            out_channels=channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.res1 = Resblock()
        self.res2 = Resblock()
        self.res3 = Resblock()
        self.res4 = Resblock()

        self.relu = nn.ReLU(inplace=True)

        self.backbone = nn.Sequential(  # method 2: 4 resnet repeated blocks
            self.res1, self.res2, self.res3, self.res4
        )

        self.is_classifier = is_classifier
        if is_classifier:
            self.classifier = nn.Conv2d(channel, spectral_num * 2, kernel_size=3, stride=1, padding=1, bias=True)

            log_print("[Fusionet]: construct the classifier head")

        else:
            self.head = nn.Conv2d(
                in_channels=channel,
                out_channels=spectral_num,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            )

            log_print("[Fusionet]: construct the regression head")

    def forward(self, x, y):  # x= lms; y = pan
        # for the image space input
        # pan_concat = y.repeat(1, self.spectral_num, 1, 1)
        # input = torch.sub(pan_concat, x)

        # for the latent space input
        pan = y
        input = torch.sub(pan, x)

        rs = self.relu(self.conv1(input))
        rs = self.backbone(rs)  # ResNet's backbone!
        output = self.head(rs)

        if self.is_classifier:  # (bs, 2, c, h, w)
            # output = einops.rearrange(
            #     output, "b c h w -> b (k c) h w", k=2
            # )  # must be bsq
            return output  # cross_entropy(output, label)
        else:  # (bs, c, h, w)
            # regression head, add the input
            return output + x


if __name__ == "__main__":
    # Test the FusionNet class
    model = FusionNet(spectral_num=16, channel=32, is_classifier=False)

    x = torch.randn(1, 16, 64, 64)  # Example input latent
    y = torch.randn(1, 16, 64, 64)  # Example input pan

    fused = model(x, y)
    print(fused.shape)  # Should be (1, 16, 64, 64)

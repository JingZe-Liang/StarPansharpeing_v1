from segmentation_models_pytorch import DeepLabV3, DPT, UnetPlusPlus
import segmentation_models_pytorch as smp
import torch

model = smp.Unet(
    encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
    in_channels=4,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=3,  # model output channels (number of classes in your dataset)
)

# model = DeepLabV3(classes=24,in_channels=3)
# print(model)

x = torch.randn(2, 4, 512, 512)
print(model(x).shape)

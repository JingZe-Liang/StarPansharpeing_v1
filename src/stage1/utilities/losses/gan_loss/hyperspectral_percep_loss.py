import copy
from typing import Literal, Sequence, cast

import einx
import numpy as np
import open_clip
import torch
import torch as th
import torch.nn as nn
from jaxtyping import Float
from loguru import logger
from lpips import LPIPS
from numpy.typing import NDArray
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import Normalize

from src.stage1.utilities.losses.repa.repa_feature_loss import (
    DINOV3_TO_NUM_LAYERS,
    load_repa_dino_v3_model,
)

from .utils import get_rgb_channels_for_model, linstretch_torch


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


# Perceptual model creation


def create_peceptual_model(model_type: str, compute_on_logits, use_lpips_vgg, **model_kwargs):
    if model_type in ("vgg", "lpips-vgg"):
        assert not compute_on_logits, "LPIPS-VGG does not support computing on logits"
        if use_lpips_vgg:
            model = LPIPS(net="vgg").eval()
        else:
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            return_nodes = {"features.20": "features", "classifier.6": "logits"}
    elif model_type == "resnet":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        return_nodes = {
            "layer1": "features_1",
            "layer2": "features_2",
            "layer4": "features_4",
            "fc": "logits",
        }
    elif model_type == "convnext_s":
        model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
        return_nodes = {
            "features.1": "features_1",
            "features.3": "features_2",
            "features.4": "features_4",
            "classifier": "logits",
        }
    elif model_type == "remote_clip_RN50":
        assert "clip_model_ckpt_path" in model_kwargs, "clip_model_ckpt_path must be specified for remote_clip_RN50"

        # Zihan note: this is a bit nasty, because it load the
        # whole clip model (with ununsed text model) but anyway, it is simple and works fine

        with th.device("cpu"):
            clip, *_ = open_clip.create_model_and_transforms("RN50")
            clip.load_state_dict(th.load(model_kwargs["clip_model_ckpt_path"]))
            visual = copy.deepcopy(clip.visual).eval()
            del clip

        model = WrappedClipVisual(visual)

        return_nodes = {
            "body.layer1": "features_1",
            "body.layer2": "features_2",
            "body.layer4": "features_3",
            "final_norm": "logits",
        }
    elif model_type[:6] == "dinov3":
        model = load_repa_dino_v3_model(model_name=model_type, pretrained_on="satellite")
    else:
        raise NotImplementedError(
            f"Unsupported model type: {model_type}. "
            "Currently supported types are: vgg, lpips-vgg, resnet, "
            "convnext_s, remote_clip_RN50"
        )

    if not compute_on_logits and not use_lpips_vgg and model_type[:6] != "dinov3":
        model = create_feature_extractor(model, return_nodes=return_nodes)

    return model


class HyperspectralFeatureLoss:
    def __init__(
        self,
        use_group: bool = False,
        # rgb options
        rgb_channels: list | Literal["random", "mean", "largest"] | str | None = None,
        # group options
        group_size: int = 3,
        num_groups_to_select: int | float | None = None,
        padding_mode: Literal["zero", "repeat"] = "zero",
        # image options
        img_is_neg1_1=True,
        img_size: int = 224,
    ):
        super().__init__()
        self.use_group = use_group
        self.rgb_channels = rgb_channels
        self.group_size = group_size
        self.num_groups_to_select = num_groups_to_select
        self.padding_mode = padding_mode
        self.img_is_neg1_1 = img_is_neg1_1
        self.img_size = img_size

        # Normalization
        self.normalize = Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

        # Check on grouping
        assert self.padding_mode in ["zero", "repeat"]
        if isinstance(self.num_groups_to_select, float):
            assert 0 < self.num_groups_to_select <= 1.0, "num_groups_to_select must be between 0 and 1"
        elif isinstance(self.num_groups_to_select, int):
            assert self.num_groups_to_select > 0, "num_groups_to_select must be greater than 0"
        else:
            assert self.num_groups_to_select is None, (
                f"num_groups_to_select must be either an integer, a float or None, but got {type(self.num_groups_to_select)}"
            )

        # Check on rgb channels
        if self.rgb_channels is not None:
            if isinstance(self.rgb_channels, str):
                assert self.rgb_channels in (
                    "random",
                    "mean",
                    "largest",
                ), "rgb_channels must be randomly selected"
            elif isinstance(self.rgb_channels, (list, tuple)):
                assert len(self.rgb_channels) == 3, "rgb_channels must be 3 channels"
            else:
                raise TypeError(f"rgb_channels must be list or tuple or str, but got {type(rgb_channels)}")

    def _pad_channels(self, x: th.Tensor) -> th.Tensor:
        """Pad input tensor channels to make them divisible by group_size.

        Args:
            x (th.Tensor): Input tensor of shape [B, C, H, W]

        Returns:
            th.Tensor: Padded tensor with channels divisible by group_size
        """
        b, c, h, w = x.shape
        remainder = c % self.group_size
        if remainder == 0:
            return x

        pad_size = self.group_size - remainder
        if self.padding_mode == "zero":
            return th.nn.functional.pad(x, (0, 0, 0, 0, 0, pad_size))
        elif self.padding_mode == "repeat":
            last_slice = x[:, -1:, :, :]
            return th.cat([x, last_slice.repeat(1, pad_size, 1, 1)], dim=1)
        else:
            raise ValueError(f"Invalid padding mode: {self.padding_mode}")

    def _group_select(self, x: th.Tensor) -> th.Tensor:
        """Group channels and optionally select random groups.

        Args:
            x (th.Tensor): Input tensor of shape [B, C, H, W]

        Returns:
            th.Tensor: Grouped tensor of shape [B, num_groups, group_size, H, W]
        """
        b, c, h, w = x.shape

        # Group channels [B, C, H, W] => [B, num_groups, group_size, H, W]
        x_grouped = x.view(b, -1, self.group_size, h, w)

        # Randomly select groups
        num_groups = x_grouped.size(1)
        if isinstance(self.num_groups_to_select, float):
            num_groups_to_select = int(self.num_groups_to_select * num_groups)
        else:
            # is int type
            num_groups_to_select = self.num_groups_to_select

        # Select groups
        if num_groups_to_select and num_groups > num_groups_to_select:
            indices = th.randperm(num_groups)[:num_groups_to_select]
            x_selected = x_grouped.index_select(1, indices.to(x.device))
        else:
            x_selected = x_grouped

        # linestretch each group
        x_selected = x_selected.view(-1, self.group_size, h, w)
        x_selected = linstretch_torch(x_selected)
        x_selected = einx.rearrange("(b g) c h w -> b g c h w", x_selected, b=b, c=self.group_size)

        return x_selected

    def _rgb_select(self, img: th.Tensor, use_linstretch: bool = False):
        """Select RGB channels from hyperspectral image."""
        img = get_rgb_channels_for_model(self.rgb_channels, img, use_linstretch)
        return img

    def get_img_rgb(self, x: th.Tensor):
        if self.img_is_neg1_1:
            x = (x + 1) / 2

        # Interpolation
        x = th.nn.functional.interpolate(x, size=self.img_size, mode="bilinear")

        if self.use_group:
            # Group channels
            x = self._pad_channels(x)
            x = self._group_select(x)  # (bs, n_grps, grp_size, h, w)
        else:
            x = self._rgb_select(x).unsqueeze(1)  # (bs, 1, 3, h, w)

        return x


type PerceptionModelChoices = Literal["vgg", "lpips-vgg", "resnet", "convnext_s", "remote_clip_RN50"]


class LPIPSHyperpspectralLoss(nn.Module, HyperspectralFeatureLoss):
    """A perceptual loss module for hyperspectral images that groups channels and computes LPIPS loss.

    This loss function extends traditional perceptual loss to handle hyperspectral images by:
    1. Grouping spectral channels into groups of fixed size
    2. Randomly selecting a subset of groups (optional)
    3. Computing perceptual loss for each group independently
    4. Averaging the losses across groups

    Args:
        model_type (Literal["vgg", "lpips-vgg", "resnet", "convnext_s"]):
            Backbone model type for feature extraction.
        group_size (int): Number of channels per group. Default: 3.
        num_groups_to_select (int, optional): Number of groups to randomly select.
            If None, uses all groups. Can also be float (0-1) for percentage.
        padding_mode (Literal["zero", "repeat"]): How to pad channels to make
            divisible by group_size. Default: "zero".
        compute_on_logits (bool): Whether to compute loss on final logits or
            intermediate features. Default: True.
        img_is_neg1_to_1 (bool): Whether input images are in [-1,1] range
            (True) or [0,1] range (False). Default: False.
    """

    def __init__(
        self,
        percep_model: PerceptionModelChoices | str | nn.Module,
        use_group=False,
        rgb_channels: list | Literal["random", "mean", "largest"] | str | None = None,
        group_size: int = 3,
        num_groups_to_select: int | float | None = None,
        padding_mode: Literal["zero", "repeat"] = "repeat",
        compute_on_logits: bool = True,
        img_is_neg1_to_1: bool = True,
        img_size: int = 224,
        **kwargs,
    ):
        super().__init__()
        HyperspectralFeatureLoss.__init__(
            self,
            use_group,
            rgb_channels,
            group_size,
            num_groups_to_select,
            padding_mode,
            img_is_neg1_to_1,
            img_size=img_size,
        )

        self.compute_on_logits = compute_on_logits
        self.use_lpips_vgg = percep_model[:5] == "lpips" if isinstance(percep_model, str) else False

        # Initialize backbone model
        if isinstance(percep_model, nn.Module):
            self.percep_model_name = percep_model.__class__.__name__
            self.percep_model = percep_model
        else:
            self.percep_model_name = percep_model
            self.percep_model = create_peceptual_model(
                percep_model,
                self.compute_on_logits,
                self.use_lpips_vgg,
                **kwargs,
            )

        logger.info(
            f"[LPIPS Hyperspectral]: using {self.percep_model.__class__.__name__} model "
            f"for extracting features for LPIPS loss"
        )

        self.assertions()

        # Freeze model parameters
        self.percep_model.requires_grad_(False)

    def _compute_gram_matrix(self, feature_maps: th.Tensor) -> th.Tensor:
        b, c, h, w = feature_maps.size()
        features = feature_maps.view(b, c, h * w)
        # Atoken: it can stabilize the training, even eliminating the need for GAN training.
        gram = th.bmm(features, features.transpose(1, 2))  # [b, c, l] x [b, l, c] -> [b, c, c]
        return gram / (h * w)

    def _compute_percep_loss(self, input: th.Tensor, target: th.Tensor) -> th.Tensor:
        """Compute perceptual loss for a single group of channels.

        Args:
            input (th.Tensor): Input image group [B, group_size, H, W]
            target (th.Tensor): Target image group [B, group_size, H, W]

        Returns:
            th.Tensor: Scalar loss value
        """

        # Directly use LPIPS vgg network to compute loss
        if self.use_lpips_vgg:
            loss = self.percep_model(input, target)
            return loss

        # Forward pass
        features_input = self.percep_model(input)
        with th.no_grad():
            features_target = self.percep_model(target)

        # Compute loss
        if self.compute_on_logits:
            # Only on the last logits
            return th.nn.functional.mse_loss(features_input, features_target)
        else:
            loss_layers = th.zeros(1).to(input.device)
            for (_, inp_feat), (_, tgt_feat) in zip(features_input.items(), features_target.items()):
                loss_layers += th.nn.functional.mse_loss(inp_feat, tgt_feat)
            return loss_layers / len(features_input)

    def forward_perceptual_model(self, x) -> dict[str, th.Tensor]:
        # 4 layers as default
        model_name = self.percep_model_name
        # is dino model
        if model_name[:6] == "dinov3" or model_name == "DinoVisionTransformer":
            n_layers_take = 4
            total_layers = DINOV3_TO_NUM_LAYERS.get(model_name, None)
            if total_layers is None:
                n = -1
            else:
                n = list(np.linspace(0, total_layers, num=n_layers_take))
            if hasattr(self.percep_model, "get_intermediate_layers"):
                feats = self.percep_model.get_intermediate_layers(  # type: ignore
                    x, n=n, reshape=True, norm=True
                )
                feats = {str(idx): feat for idx, feat in zip(range(len(feats)), feats)}
            else:
                raise RuntimeError(f"Dino v3 model must has method get_intermediate_layers to take features")
            return feats
        # vgg-like model that has feature hook
        else:
            return self.percep_model(x)

    def forward(self, input: th.Tensor, target: th.Tensor):
        """Compute the hyperspectral perceptual loss between input and target.

        Args:
            input (th.Tensor): Input hyperspectral image [B, C, H, W]
            target (th.Tensor): Target hyperspectral image [B, C, H, W]

        Returns:
            th.Tensor: Average perceptual loss across all selected groups

        Raises:
            AssertionError: If input and target shapes don't match or aren't 4D
        """
        # Check dimensions [B, C, H, W]
        assert input.dim() == 4 and target.dim() == 4
        assert input.shape == target.shape

        # To RGB or groups of RGB images
        input_groups = self.get_img_rgb(input)
        target_groups = self.get_img_rgb(target)

        # Compute loss for each group
        percep_loss = 0.0
        num_groups = input_groups.size(1)

        for i in range(num_groups):
            # Prepare for model
            input_, target_ = input_groups[:, i], target_groups[:, i]
            input_, target_ = map(self.normalize, (input_, target_))
            percep_loss_ = self._compute_percep_loss(input_, target_)
            percep_loss += percep_loss_

        return percep_loss / num_groups

    def assertions(self):
        """Validate configuration parameters.

        Raises:
            AssertionError: If any configuration is invalid
            TypeError: If num_groups_to_select has invalid type
        """
        if self.use_lpips_vgg:
            assert self.model_type in (
                "vgg",
                "lpips-vgg",
            ), "LPIPS VGG is only supported for VGG model type"

    def __repr__(self):
        return (
            f"\n{self.__class__.__name__}(\n"
            f"    model_type={self.percep_model_name},\n"
            f"    rgb_channels={self.rgb_channels},\n"
            f"    group_size={self.group_size},\n"
            f"    num_groups_to_select={self.num_groups_to_select},\n"
            f"    padding_mode={self.padding_mode},\n"
            f"    compute_on_logits={self.compute_on_logits},\n"
            f"    use_lpips_vgg={self.use_lpips_vgg},\n"
            f"    img_is_neg1_to_1={self.img_is_neg1_1}\n"
            ")"
        )


# * --- test --- * #
def test_hyperspectral_perceptual_loss():
    # 基本测试配置
    base_test_cases = [
        # {"model_type": "vgg"},
        # {"model_type": "lpips-vgg"},
        # {"model_type": "resnet"},
        # {"model_type": "convnext_s"},
    ]

    # 扩展测试配置
    extended_test_cases = [
        # 测试不同的compute_on_logits设置
        # {"model_type": "vgg", "compute_on_logits": False, },
        {
            "model_type": "resnet",
            "compute_on_logits": False,
            "use_gram_model": None,
            "num_groups_to_select": 0.5,
        },
        {"model_type": "convnext_s", "compute_on_logits": False},
        # 测试不同的padding模式
        {"model_type": "vgg", "padding_mode": "zero"},
        {"model_type": "resnet", "padding_mode": "zero"},
        # 测试不同的group_size
        {"model_type": "vgg", "group_size": 4},
        {"model_type": "resnet", "group_size": 5},
        # 测试不同的num_groups_to_select类型
        {"model_type": "vgg", "num_groups_to_select": 0.5},  # 比例选择
        {"model_type": "resnet", "num_groups_to_select": 1},  # 最小选择数
    ]

    # 边界情况测试
    edge_cases = [
        {"model_type": "vgg", "img_is_neg1_to_1": False},  # 测试[0,1]范围输入
        {"model_type": "vgg", "group_size": 1},  # 最小group_size
        {"model_type": "resnet", "num_groups_to_select": None},  # 选择所有组
    ]

    # 新增Gram Loss测试配置
    gram_test_cases = [
        {"use_gram_model": "vgg", "gram_loss_weight": 1.0},
        {"use_gram_model": "vgg", "gram_loss_weight": 0.5},
        {"use_gram_model": None},  # 测试不启用Gram Loss
    ]

    # 合并所有测试用例
    all_test_cases = base_test_cases + extended_test_cases + edge_cases + gram_test_cases

    for config in all_test_cases:
        print(f"\nTesting config: {config}")

        try:
            # 初始化损失函数 (添加Gram相关参数)
            loss_fn = LIPIPSHyperpspectral(
                percep_model=config.get("model_type", "vgg"),
                group_size=config.get("group_size", 3),
                num_groups_to_select=config.get("num_groups_to_select", 2),
                padding_mode=config.get("padding_mode", "repeat"),
                compute_on_logits=config.get("compute_on_logits", True),
                img_is_neg1_to_1=config.get("img_is_neg1_to_1", True),
                use_gram_model=config.get("use_gram_model"),  # 新增
                gram_loss_weight=config.get("gram_loss_weight", 1.0),  # 新增
            ).cuda()

            # 创建测试数据
            batch_size = 2
            channels = 3  # Gram Loss需要3通道输入
            height, width = 256, 256
            input = th.rand(batch_size, channels, height, width) * 2 - 1
            target = th.rand(batch_size, channels, height, width) * 2 - 1
            input = input.cuda()
            target = target.cuda()

            # 计算损失 (现在返回total_loss和loss_dict)
            loss_dict = loss_fn(input, target)

            # 验证输出
            assert "perceptual_loss" in loss_dict
            assert "gram_loss" in loss_dict

            # Gram Loss特定验证
            if config.get("use_gram_model"):
                assert loss_dict["gram_loss"].item() > 0
                print(f"Gram Loss active: {loss_dict['gram_loss'].item():.4f}")
            else:
                assert loss_dict["gram_loss"].item() == 0
                print("Gram Loss inactive as expected")

            print(f"Test passed! Total loss: {(loss_dict['perceptual_loss'] + loss_dict['gram_loss']).item():.4f}")

        except Exception as e:
            print(f"Test failed for config {config}: {str(e)}")
            continue

    # 特殊测试：空输入检查
    print("\nTesting empty input handling...")
    try:
        loss_fn = LIPIPSHyperpspectral(percep_model="vgg").cuda()
        empty_input = th.zeros(0, 3, 256, 256).cuda()
        loss_fn(empty_input, empty_input)
        print("Empty input test passed (should have raised an assertion)")
    except AssertionError:
        print("Empty input test correctly raised an assertion")
    except Exception as e:
        print(f"Unexpected error with empty input: {str(e)}")


if __name__ == "__main__":
    test_hyperspectral_perceptual_loss()

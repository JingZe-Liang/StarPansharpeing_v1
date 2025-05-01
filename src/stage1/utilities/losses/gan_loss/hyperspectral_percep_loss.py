import copy
from typing import Literal

import open_clip
import torch as th
import torch.nn as nn
from loguru import logger
from lpips import LPIPS
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor


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


class LIPIPSHyperpspectral(nn.Module):
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
        model_type: Literal[
            "vgg", "lpips-vgg", "resnet", "convnext_s", "remote_clip_RN50"
        ],
        group_size: int = 3,
        num_groups_to_select: int | None = None,
        padding_mode: Literal["zero", "repeat"] = "zero",
        compute_on_logits: bool = True,
        img_is_neg1_to_1: bool = True,
        use_gram_model: str | None = "vgg",
        gram_loss_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.model_type = model_type
        self.group_size = group_size
        self.num_groups_to_select = num_groups_to_select
        self.padding_mode = padding_mode
        self.compute_on_logits = compute_on_logits
        self.use_lpips_vgg = model_type[:5] == "lpips"
        self.img_is_neg1_to_1 = img_is_neg1_to_1
        self.use_gram_loss = use_gram_model is not None
        self.use_gram_model = use_gram_model
        self.gram_loss_weight = gram_loss_weight
        logger.info(
            f"[LPIPS Hyperspectral]: using {model_type} model for extracting features for LPIPS loss, "
            f"{self.use_gram_model} for Gram loss",
        )

        # Initialize backbone model
        self.model = self._make_perceptual_model(
            model_type,
            self.compute_on_logits,
            self.use_lpips_vgg,
            **kwargs,
        )

        # Gram loss
        if self.use_gram_loss:
            if (
                self.use_gram_model == model_type
            ):  # gram model is same as perceptual model
                self.gram_model = self.model

            else:
                self.gram_model = self._make_perceptual_model(
                    self.use_gram_model, False, False, **kwargs
                )

        self.assertions()

        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Normalization parameters
        self.register_buffer(
            "mean", th.Tensor([0.485, 0.456, 0.406])[None, :, None, None]
        )
        self.register_buffer(
            "std", th.Tensor([0.229, 0.224, 0.225])[None, :, None, None]
        )
        self.register_buffer("zero", th.tensor(0.0), persistent=False)

    def _make_perceptual_model(
        self,
        model_type: str,
        compute_on_logits: bool,
        use_lpips_vgg: bool,
        **kwargs,
    ):
        if model_type in ("vgg", "lpips-vgg"):
            assert not compute_on_logits, (
                "LPIPS-VGG does not support computing on logits"
            )
            if self.use_lpips_vgg:
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
            model = models.convnext_small(
                weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1
            )
            return_nodes = {
                "features.1": "features_1",
                "features.3": "features_2",
                "features.4": "features_4",
                "classifier": "logits",
            }
        elif model_type == "remote_clip_RN50":
            assert "clip_model_ckpt_path" in kwargs, (
                "clip_model_ckpt_path must be specified for remote_clip_RN50"
            )

            # Zihan note: this is a bit nasty, because it load the
            # whole clip model (with ununsed text model) but anyway, it is simple and works fine

            clip, *_ = open_clip.create_model_and_transforms("RN50")
            clip.load_state_dict(th.load(kwargs["clip_model_ckpt_path"]))
            visual = copy.deepcopy(clip.visual).eval()
            del clip

            model = WrappedClipVisual(visual)

            return_nodes = {
                "body.layer1": "features_1",
                "body.layer2": "features_2",
                "body.layer4": "features_3",
                "final_norm": "logits",
            }
        else:
            raise NotImplementedError(
                f"Unsupported model type: {model_type}. "
                "Currently supported types are: vgg, lpips-vgg, resnet, "
                "convnext_s, remote_clip_RN50"
            )

        if not compute_on_logits and not use_lpips_vgg:
            model = create_feature_extractor(model, return_nodes=return_nodes)

        return model

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

        return x_selected

    def _compute_gram_matrix(self, feature_maps: th.Tensor) -> th.Tensor:
        b, c, h, w = feature_maps.size()
        features = feature_maps.view(b, c, h * w)
        gram = th.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)

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
            loss = self.model(input, target)
            return loss

        # Forward pass
        features_input = self.model(input)
        with th.no_grad():
            features_target = self.model(target)

        # Compute loss
        if self.compute_on_logits:
            # Only on the last logits
            return th.nn.functional.mse_loss(features_input, features_target)
        else:
            loss_layers = 0.0
            for (name, inp_feat), (_, tgt_feat) in zip(
                features_input.items(), features_target.items()
            ):
                loss_layers += th.nn.functional.mse_loss(inp_feat, tgt_feat)
                # loss_layers += th.nn.functional.mse_loss(
                #     features_input["features"], features_target["features"]
                # )
                # loss += th.nn.functional.mse_loss(
                #     features_input["logits"], features_target["logits"]
                # )
            return loss_layers / len(features_input)

    def _compute_gram_loss(self, input: th.Tensor, target: th.Tensor) -> th.Tensor:
        """Compute Gram matrix loss between input and target feature maps.

        Args:
            input: Input image tensor [B, C, H, W]
            target: Target image tensor [B, C, H, W]

        Returns:
            Combined Gram matrix loss from all selected layers
        """
        if not self.use_gram_loss:
            return self.zero

        # Get features from both images
        input_features = self.gram_model(input)
        with th.no_grad():
            target_features = self.gram_model(target)

        gram_loss = 0.0
        # Compute Gram loss for each feature layer
        for layer_name in input_features.keys():
            if layer_name == "logits":
                continue

            # Compute Gram matrices
            input_gram = self._compute_gram_matrix(input_features[layer_name])
            target_gram = self._compute_gram_matrix(target_features[layer_name])

            # Add MSE between Gram matrices
            gram_loss += th.nn.functional.mse_loss(input_gram, target_gram)

        return gram_loss / len(input_features)  # Average over layers

    def _preprare_for_percep_model(self, input, target):
        # Resize to 224x224 for pretrained models
        input = th.nn.functional.interpolate(
            input, size=224, mode="bilinear", align_corners=True
        )
        target = th.nn.functional.interpolate(
            target, size=224, mode="bilinear", align_corners=True
        )

        # Normalize
        if self.img_is_neg1_to_1:
            # to [0, 1]
            input = (input + 1) / 2
            target = (target + 1) / 2
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std

        return input, target

    def forward(self, input: th.Tensor, target: th.Tensor) -> th.Tensor:
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

        # Pad channels if needed
        input_padded = self._pad_channels(input)
        target_padded = self._pad_channels(target)

        # Group and select channels
        input_groups = self._group_select(input_padded)
        target_groups = self._group_select(target_padded)

        # Compute loss for each group
        percep_loss = 0.0
        gram_loss = 0.0
        num_groups = input_groups.size(1)

        for i in range(num_groups):
            # Prepare for model
            input_, target_ = self._preprare_for_percep_model(
                input_groups[:, i], target_groups[:, i]
            )

            percep_loss_ = self._compute_percep_loss(input_, target_)
            gram_loss_ = (
                self._compute_gram_loss(input_, target_) * self.gram_loss_weight
            )

            percep_loss += percep_loss_
            gram_loss += gram_loss_

        return {
            "perceptual_loss": percep_loss / num_groups,
            "gram_loss": gram_loss / num_groups,
        }

    def assertions(self):
        """Validate configuration parameters.

        Raises:
            AssertionError: If any configuration is invalid
            TypeError: If num_groups_to_select has invalid type
        """
        assert self.padding_mode in ["zero", "repeat"]
        if self.use_lpips_vgg:
            assert self.model_type in (
                "vgg",
                "lpips-vgg",
            ), "LPIPS VGG is only supported for VGG model type"

        if isinstance(self.num_groups_to_select, float):
            assert 0 < self.num_groups_to_select <= 1.0, (
                "num_groups_to_select must be between 0 and 1"
            )
        elif isinstance(self.num_groups_to_select, int):
            assert self.num_groups_to_select > 0, (
                "num_groups_to_select must be greater than 0"
            )
        else:
            assert self.num_groups_to_select is None, (
                f"num_groups_to_select must be either an integer, a float or None, but got {type(self.num_groups_to_select)}"
            )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"model_type={self.model_type}, "
            f"group_size={self.group_size}, "
            f"num_groups_to_select={self.num_groups_to_select}, "
            f"padding_mode={self.padding_mode}, "
            f"compute_on_logits={self.compute_on_logits}, "
            f"use_lpips_vgg={self.use_lpips_vgg}, "
            f"img_is_neg1_to_1={self.img_is_neg1_to_1}"
            f"grad_model={self.use_gram_model}"
            f")"
        )


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
    all_test_cases = (
        base_test_cases + extended_test_cases + edge_cases + gram_test_cases
    )

    for config in all_test_cases:
        print(f"\nTesting config: {config}")

        try:
            # 初始化损失函数 (添加Gram相关参数)
            loss_fn = LIPIPSHyperpspectral(
                model_type=config.get("model_type", "vgg"),
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

            print(
                f"Test passed! Total loss: {(loss_dict['perceptual_loss'] + loss_dict['gram_loss']).item():.4f}"
            )

        except Exception as e:
            print(f"Test failed for config {config}: {str(e)}")
            continue

    # 特殊测试：空输入检查
    print("\nTesting empty input handling...")
    try:
        loss_fn = LIPIPSHyperpspectral(model_type="vgg").cuda()
        empty_input = th.zeros(0, 3, 256, 256).cuda()
        loss_fn(empty_input, empty_input)
        print("Empty input test passed (should have raised an assertion)")
    except AssertionError:
        print("Empty input test correctly raised an assertion")
    except Exception as e:
        print(f"Unexpected error with empty input: {str(e)}")


if __name__ == "__main__":
    test_hyperspectral_perceptual_loss()

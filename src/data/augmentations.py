from typing import Sequence

from kornia.augmentation import (
    AugmentationSequential,
    CenterCrop,
    RandomBoxBlur,
    RandomChannelShuffle,
    RandomCutMixV2,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomRotation,
    RandomSharpness,
    RandomVerticalFlip,
)
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D

from .utils import to_n_tuple


class HyperRandomGrayScale(IntensityAugmentationBase2D):
    """Randomly convert hyperspectral images to grayscale.

    This augmentation randomly converts hyperspectral images to grayscale by computing
    the mean across all channels and repeating it for all channels. This preserves
    the original tensor shape while simulating grayscale appearance.

    Args:
        p (float): Probability of applying the grayscale transformation. Defaults to 0.5.
    """

    def __init__(self, p=0.5):
        super().__init__(p)

    def apply_transform(self, input, params, flags, transform=None):
        """Apply grayscale transformation to input tensor.

        Args:
            input (Tensor): Input tensor of shape (B, C, H, W) where B is batch size,
                C is number of channels, H is height, and W is width.
            params: Parameters for the transformation (unused in this implementation).
            flags: Flags for the transformation (unused in this implementation).
            transform: Additional transform parameters (unused in this implementation).

        Returns:
            Tensor: Grayscale version of input tensor with the same shape (B, C, H, W).
        """
        assert input.ndim == 4
        c = input.shape[1]
        gray = input.mean(dim=1, keepdim=True).repeat_interleave(c, dim=1)
        return gray


def hyper_transform(
    op_list: tuple[str, ...],
    probs: tuple[float, ...] | float = 0.5,
    random_apply: int | tuple[int, int] = 2,
    default_img_size: int = 256,
):
    """Create a hyper-spectral image augmentation transform function.

    This function creates a configurable data augmentation pipeline for hyperspectral images
    using various image transformations. The transformations are applied sequentially with
    specified probabilities.

    Args:
        op_list (tuple[str, ...]): List of augmentation operations to apply.
            Supported operations: 'grayscale', 'channel_shuffle', 'sharpness',
            'rotation', 'horizontal_flip', 'vertical_flip', 'cutmix', 'blur',
            'center_crop', 'resized_crop'.
        probs (tuple[float, ...] | float): Probability of applying each operation.
            If float, same probability is used for all operations. Defaults to 0.5.
        random_apply (int | tuple[int]): Number of operations to randomly apply.
            If tuple, specifies the range. Defaults to 2.
        default_img_size (int): Default image size for crop operations. Defaults to 256.

    Returns:
        function: A transform function that applies the specified augmentations to input data.
    """
    if isinstance(probs, float):
        probs: tuple[float, ...] = tuple([probs] * len(op_list))
    assert len(probs) == len(op_list), (  # type: ignore
        "Number of probabilities must match number of operations."
    )

    _default_size: tuple[int, int] = to_n_tuple(default_img_size, 2)

    _op_list_cls = dict(
        grayscale=lambda p: HyperRandomGrayScale(p=p),
        channel_shuffle=lambda p: RandomChannelShuffle(p=p),
        sharpness=lambda p: RandomSharpness(p=p, sharpness=(0.5, 1.0)),
        rotation=lambda p: RandomRotation((-30, 30), p=p),
        horizontal_flip=lambda p: RandomHorizontalFlip(p=p),
        vertical_flip=lambda p: RandomVerticalFlip(p=p),
        cutmix=lambda p: RandomCutMixV2(num_mix=1, p=p, cut_size=(0.4, 0.6)),
        blur=lambda p: RandomBoxBlur((3, 3), p=p),
        center_crop=lambda p: CenterCrop(_default_size, p=p),
        resized_crop=lambda p: RandomResizedCrop(_default_size, scale=(0.5, 1.0), ratio=(0.75, 1.333), p=p),
    )

    ops = []
    for op_str, prob in zip(op_list, probs):
        op = _op_list_cls[op_str]
        ops.append(op(prob))

    op_seq = AugmentationSequential(
        *ops,
        data_keys=["input"],
        random_apply=(  # type: ignore
            tuple(random_apply) if isinstance(random_apply, Sequence) else random_apply
        ),
        same_on_batch=False,
        keepdim=True,
    )

    def dict_mapper(x):
        x = op_seq(x)
        return x

    return dict_mapper

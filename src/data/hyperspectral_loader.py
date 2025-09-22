import os
import warnings
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, Sequence, cast

import accelerate
import braceexpand
import numpy as np
import scipy.io
import tifffile
import torch
import torch.distributed
import webdataset as wds
import wids
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
from natsort import natsorted
from safetensors.torch import load_file
from timm.layers.helpers import to_2tuple
from torch.utils.data import DataLoader, Dataset
from typing_extensions import deprecated

from src.data.codecs import (
    img_decode_io,
    json_decode_io,
    safetensors_decode_io,
    string_decode_io,
    tiff_decode_io,
    wids_image_decode,
)
from src.data.curriculums import get_curriculum_fn
from src.data.utils import (
    chained_dataloaders,
    check_img_channel,
    de_structure_tar,
    expand_paths_and_correct_loader_kwargs,
    extract_modality_names,
    filter_undecoded,
    flatten_nested_list,
    flatten_sub_dict,
    generate_wds_config_modify_only_some_kwgs,
    get_wids_index_json_info,
    large_image_resizer_clipper,
    loop_apply_filters,
    norm_img,
    remove_dot_and_extensions,
    remove_extension,
    remove_keys,
    rename_keys,
    search_one_key_not_dunder,
    size_filtering,
    to_n_tuple,
    wids_filter_img_size,
    wids_img_size_filter_by_parquet_index,
)
from src.data.utils import remove_meta_data as remove_meta_data_fn
from src.data.wids_samplers import IndexFilteredDistributedSampler, IndexFilteredSampler
from src.stage2.denoise.utils.noise_adder import (
    UniHSINoiseAdderKornia,
    get_tokenizer_trainer_noise_adder,
)
from src.utilities.config_utils import function_config_to_basic_types
from src.utilities.logging import log_print
from src.utilities.train_utils.state import StepsCounter

__compile_collate_fn = False
if __compile_collate_fn:
    torch._dynamo.config.recompile_limit = 20

type WebdatasetPath = str | list[str] | list[list[str]]
type SupportedLoaderType = Literal["webdataset", "folder", "wids"]

import hashlib

loader_seed = int(
    hashlib.sha256("uestc_ZihanCao_add_my_wechat:iamzihan123".encode()).hexdigest(), 16
) % (2**31)
loader_generator = torch.Generator().manual_seed(loader_seed)


class FunctionDeprecatedWarning(UserWarning):
    """
    Custom warning for deprecated functions.
    """

    pass


@deprecated(
    "get_dict_tensor_mapper is deprecated",
    category=FunctionDeprecatedWarning,
)
def get_dict_tensor_mapper(to_neg_1_1=True):
    """
    Returns a mapper function that converts a raw sample into a dictionary of tensors.

    This is typically used in data pipelines to map raw data (e.g., from WebDataset)
    into processed PyTorch tensors. The image is normalized and optionally scaled
    to the range [-1, 1].

    Args:
        to_neg_1_1 (bool): If True, scales the image from [0, 1] to [-1, 1]. Default is True.

    Returns:
        function: A function that takes a sample and returns a dictionary containing:
            - 'img' (Tensor): The processed image tensor of shape (C, H, W).
            - 'img_max' (float): The maximum pixel value used for normalization.
    """

    def wds_to_dict_tensor_mapper(sample):
        img = torch.as_tensor(sample["img.tiff"]).float()
        img_max = img.max()
        img = img / img_max
        if to_neg_1_1:
            img = img * 2 - 1
        img = img.permute(-1, 0, 1)

        return {"img": img, "img_max": img_max}

    return wds_to_dict_tensor_mapper


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
    random_apply: int | tuple[int] = 2,
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
        resized_crop=lambda p: RandomResizedCrop(
            _default_size, scale=(0.5, 1.0), ratio=(0.75, 1.333), p=p
        ),
    )

    ops = []
    for op_str, prob in zip(op_list, probs):
        op = _op_list_cls[op_str]
        ops.append(op(prob))

    op_seq = AugmentationSequential(
        *ops,
        data_keys=["input"],
        random_apply=(
            tuple(random_apply) if isinstance(random_apply, Sequence) else random_apply  # type: ignore
        ),
        same_on_batch=False,
        keepdim=True,
    )

    def dict_mapper(x):
        x = op_seq(x)
        return x

    return dict_mapper


@torch.no_grad()
def collate_fn_for_dict(
    sample: list,
    augmentations: Callable | None = None,
    norm: bool = False,
    to_negative_1_1: bool = False,
):
    """Collate function for dictionary samples.
    If you norm and make the image to [-1, 1] per-sample, you need to set
    `norm=False` and `to_negative_1_1=False` in the batch collate function.

    Args:
        sample (list): List of samples to collate.
        augmentations (Callable, optional): Augmentation function to apply. Defaults to None.
        norm (bool, optional): Whether to normalize the image. Defaults to False.
        to_negative_1_1 (bool, optional): Whether to convert the image to [-1, 1]. Defaults to False.
    Returns:
        dict: Collated dictionary of samples.
    """
    results = {}
    assert len(sample) > 0, "sample is empty"
    keys = sample[0].keys()
    _stack = False

    for k in keys:
        sample_lst = []
        for s in sample:
            v = s[k]
            _stack = True
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
                sample_lst.append(v)
            elif isinstance(v, torch.Tensor):
                sample_lst.append(v)
            else:
                _stack = False
                sample_lst.append(v)

        if _stack:
            sample_stacked = torch.stack(sample_lst, dim=0)
            if norm:
                sample_stacked = sample_stacked / sample_stacked.max()
            if to_negative_1_1:
                sample_stacked = sample_stacked * 2 - 1
            if augmentations is not None:
                sample_stacked = augmentations(sample_stacked)

            results[k] = sample_stacked
            del sample_stacked
        else:
            results[k] = sample_lst

    return results


# * --- webdataset loader --- #


def get_hyperspectral_dataloaders(
    wds_paths: str | list[str],
    batch_size: int,
    num_workers: int,
    shuffle_size: int = 100,
    to_neg_1_1: bool = True,
    permute: bool = True,
    check_nan: bool = False,
    per_channel_norm: bool = False,
    # image key in the sample dictionary
    img_key: str | list[str] = "auto",
    tgt_key: str | list[str] | None = None,
    keys_to_remove: str | list[str] | None = None,
    random_one_key: bool = False,
    quantile_img_clip: Annotated[float, "0.0<f<1.0"] = 1.0,
    mannual_img_min_max: tuple[float, float] | None = None,
    undecoded_filtered: bool = True,
    # int for minimal_size; tuple for (min, max)
    constraint_size: int | tuple[int, int] | None = None,
    # resize to the same size before transformation and batching
    resize_before_transform: int | None = None,
    hyper_transforms_lst: tuple[str, ...] | None = (
        "grayscale",
        "channel_shuffle",
        "rotation",
        "cutmix",
        "horizontal_flip",
        "vertical_flip",
    ),
    hyper_degradation_lst: str | list[str] | None = None,
    transform_prob: float = 0.0,
    degradation_prob: float = 0.0,
    random_apply: int | tuple[int, int] = 1,
    resample: bool = True,
    prefetch_factor: int | None = 6,
    pin_memory: bool = False,
    shuffle_within_workers: bool = False,
    check_channels_n: int | None = None,
    is_structured_tar: bool = False,
    remove_meta_data: bool = False,
    repeat_n: int = 1,
    timeout: int = 300,
    epoch_len: int = 0,
    persistent_workers: bool = True,
) -> tuple[wds.WebDataset, wds.WebLoader]:
    """Get dataloaders for hyperspectral image data

    Args:
        wds_paths (str | list[str]): Path or list of paths to WebDataset tar files
        batch_size (int): Number of samples per batch
        num_workers (int): Number of worker processes for data loading
        shuffle_size (int, optional): Buffer size for shuffling data. Defaults to 100.
        to_neg_1_1 (bool, optional): Whether to normalize data to [-1,1] range. Defaults to True.
        permute (bool, optional): Whether to permute tensor dimensions. Defaults to True.
        check_nan (bool, optional): Whether to check for NaN values in data. Defaults to False.
        img_key (str | list[str], optional): Image key(s) in the sample dictionary. Defaults to "auto".
        tgt_key (str | list[str] | None, optional): Target key(s) for image data. Defaults to None.
        keys_to_remove (str | list[str] | None, optional): Keys to remove from samples. Defaults to None.
        random_one_key (bool, optional): Whether to randomly select one key when img_key="auto". Defaults to False.
        undecoded_filtered (bool, optional): Whether to filter undecoded images. Defaults to True.
        constraint_size (int | tuple[int, int] | None, optional): Size constraints for images (min or (min, max)). Defaults to None.
        resize_before_transform (int | None, optional): Resize images to this size before transformation. Defaults to None.
        hyper_transforms_lst (tuple[str, ...] | None, optional): List of data augmentation operations.
            Defaults to ("grayscale", "channel_shuffle", "rotation", "cutmix", "horizontal_flip", "vertical_flip").
        transform_prob (float, optional): Probability for each augmentation operation. Defaults to 0.0.
        random_apply (int | tuple[int, int], optional): Number of random augmentations to apply. Defaults to 1.
        resample (bool, optional): Whether to allow data resampling. Defaults to True.
        prefetch_factor (int | None, optional): Number of batches prefetched by each worker. Defaults to 6.
        pin_memory (bool, optional): Whether to use pinned memory for data loading. Defaults to False.
        shuffle_within_workers (bool, optional): Whether to shuffle data within workers. Defaults to False.
        check_channels_n (int | None, optional): Expected number of channels to check. Defaults to None.
        is_structured_tar (bool, optional): Whether the tar files are structured. Defaults to False.
        remove_meta_data (bool, optional): Whether to remove metadata from samples. Defaults to False.
        repeat_n (int, optional): Number of times to repeat the dataset. Defaults to 1.
        timeout (int, optional): Timeout for data loading operations in seconds. Defaults to 300.
        epoch_len (int, optional): Length of an epoch. Defaults to 0.
        persistent_workers (bool, optional): Whether to keep workers alive between epochs. Defaults to True.

    Returns:
        tuple[wds.WebDataset, wds.WebLoader]: Returns a tuple containing the WebDataset object and its corresponding dataloader
    """
    use_transf = (
        hyper_transforms_lst is not None
        and len(hyper_transforms_lst) > 0
        and transform_prob > 0
    )
    if use_transf:
        transform = hyper_transform(hyper_transforms_lst, transform_prob, random_apply)  # type: ignore
        log_print(
            f"[HyperWebdataset]: use augmentations {hyper_transforms_lst} with prob {transform_prob}"
        )
    use_deg = hyper_degradation_lst is not None and degradation_prob > 0

    dataset = wds.WebDataset(
        wds_paths,
        resampled=resample,  # no need `iter(dataloader)` for `next` function
        shardshuffle=False,
        nodesplitter=wds.shardlists.single_node_only
        if not accelerate.state.PartialState().use_distributed
        else wds.shardlists.split_by_node,  # split_by_node if is multi-node training
        workersplitter=wds.shardlists.split_by_worker,
        seed=2025,
        verbose=True,
        detshuffle=True if shuffle_size > 0 else False,
        empty_check=False,
        handler=wds.warn_and_continue,
    )

    # de-structure tar files
    if is_structured_tar:
        warnings.warn(
            "The 'is_structured_tar' parameter is deprecated and will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        # says we only need the images, so we just the flatten the tar file
        log_print(f"[HyperWebdataset]: the tar file is structured tar, de-structure it")
        dataset = dataset.map(de_structure_tar)

    if keys_to_remove is not None:
        dataset = dataset.map(remove_keys(keys_to_remove))

    # decode
    default_decoder = [
        wds.handle_extension("tif tiff", tiff_decode_io),
        # webdataset read safetensors in tar, can not suit the memmap for fast loading
        wds.handle_extension("safetensors", safetensors_decode_io),
        wds.handle_extension("jpg png img_content", img_decode_io),
        wds.handle_extension("txt", string_decode_io),
        wds.handle_extension("json jsonl", json_decode_io),
        "torch",  # handle npy file
    ]
    dataset = dataset.decode(*default_decoder, handler=wds.warn_and_continue)

    # extension
    dataset = dataset.map(remove_extension)

    # meta data
    if remove_meta_data:
        dataset = dataset.map(remove_meta_data_fn)

    # rename key
    if img_key == "auto":
        if tgt_key is None:
            tgt_key = "img"
        assert isinstance(tgt_key, str), (
            f'tgt_key must be a string if img_key is "auto", but got {type(tgt_key)}'
        )
        # only search one key as the tgt_key, it will remove other keys if you have multiple image keys
        dataset = dataset.map(
            partial(
                search_one_key_not_dunder, out_key=tgt_key, random_one=random_one_key
            )
        )
    elif isinstance(img_key, (str, tuple, list)) and (
        isinstance(tgt_key, (str, tuple, list))
    ):
        dataset = dataset.map(
            partial(rename_keys, tgt_keys=tgt_key, source_keys=img_key)
        )
    elif tgt_key is None:
        tgt_key = img_key
    else:
        raise ValueError(
            f"[Webdataset Dataset]: img_key and tgt_key must be either a string or a list of strings, "
            f"but got {type(img_key)} and {type(tgt_key)}"
        )

    if isinstance(tgt_key, str):
        tgt_key = [tgt_key]

    # catch the undecoded images
    if undecoded_filtered:
        _filter_undecoded = partial(filter_undecoded, key=tgt_key)
        dataset = dataset.map(_filter_undecoded)

    # pass the minimal size or range of the image
    if constraint_size is not None:
        _sz_filter_fn = partial(
            size_filtering, tgt_key=tgt_key, constraint_size=constraint_size
        )
        dataset = dataset.map(_sz_filter_fn)

    # norm
    dataset = dataset.map(
        partial(
            norm_img,
            to_neg_1_1=to_neg_1_1,
            norm_keys=tgt_key,
            permute=permute,
            check_nan=check_nan,
            per_channel=per_channel_norm,
            quantile_clip=quantile_img_clip,
            mannual_img_min_max=mannual_img_min_max,
        )
    )

    # channel check
    if check_channels_n is not None:
        check_fn = partial(
            check_img_channel, expect_channels=check_channels_n, img_key=tgt_key
        )
        dataset = dataset.map(check_fn)

    # resize
    if resize_before_transform is not None:
        dataset = dataset.map(
            large_image_resizer_clipper(
                img_key=tgt_key,
                tgt_size=resize_before_transform,
                op_for_large="clip",
            )
        )

    # re-batched
    if shuffle_within_workers:
        dataset = dataset.batched(
            batch_size, collation_fn=collate_fn_for_dict
        )  # stacked into batch

    # augmentations
    if use_transf:
        dataset = dataset.map_dict(**{k: transform for k in tgt_key})  # type: ignore

    # Dataloader
    dataloader = wds.WebLoader(
        dataset,
        batch_size=batch_size if not shuffle_within_workers else None,
        collate_fn=None if shuffle_within_workers else collate_fn_for_dict,
        num_workers=num_workers,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=pin_memory,
        drop_last=False,
        timeout=timeout if num_workers > 0 else 0,
        generator=loader_generator,
        shuffle=False,
    )

    # Data augmentation on batch
    # 1. Hyperspectral denoising task
    # we can also init the hyper-nosing pipeline in trainer, check 'self.aug_pipe' in 'CosmosHyperspectralTokenizerTrainer'
    if use_deg:
        deg_pipe = lambda x: x
        if hyper_degradation_lst == "all":
            deg_pipe = get_tokenizer_trainer_noise_adder(p=degradation_prob)
        elif hyper_degradation_lst is not None:
            deg_pipe = UniHSINoiseAdderKornia(
                noise_type=hyper_degradation_lst,
                p=degradation_prob,
                same_on_batch=True,
                is_neg_1_1=True,
            )
        dataloader.map_dict(**{k: deg_pipe for k in tgt_key})

    if epoch_len > 0:
        _world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        dataloader = dataloader.with_epoch(epoch_len // _world_size)
        log_print(f"dataloader with epoch length: {epoch_len // _world_size}")

    # unbatch, shuffle, and rebatch within different workers
    if shuffle_within_workers:
        dataloader = dataloader.unbatched()
        if shuffle_size > 0:
            dataloader = dataloader.shuffle(shuffle_size)
        dataloader = dataloader.batched(batch_size, collation_fn=collate_fn_for_dict)

    if repeat_n != 1:
        log_print(f"repeat dataloader <red>{repeat_n}</> times")
        dataloader = dataloader.repeat(nepochs=repeat_n)

    log_print(
        f"[HyperDataset]: batch size: {batch_size}, num workers: {num_workers}, "
        f"use transformations: {hyper_transforms_lst if hyper_transforms_lst is None or transform_prob > 0.0 else 'None'} "
    )

    return dataset, dataloader


def get_fast_test_hyperspectral_data(
    data_type: Literal[
        "DCF",
        "MMSeg",
        "Houston",
        "OHS",
        "WV3",
        "QB",
        "WV2",
        "WV4",
        "IKONOS",
        "RS5M",
        "BigEarthNetS2",
        "Xiongan",
        "fmow_RGB",
        "fmow_MS",
    ] = "DCF",
    batch_size: int = 1,
):
    """
    get a test data for model/module/function testing.
    """
    wds_paths = {
        "DCF": "data/DCF_2019/hyper_images/DCF_2019_Track_2-8_bands-px_512-MSI-0000.tar",
        "MMSeg": "data/MMSeg_YREB/hyper_images/MMSeg_YREB_train_part-12_bands-MSI-0000.tar",
        "Houston": "data/Houston/hyper_images/Houston-50_bands-px_512-MSI-0000.tar",
        "OHS": "data/OHS/hyper_images/OHS-32_bands-px_512-MSI-0000.tar",
        "WV3": "data/WorldView3/hyper_images/WorldView3-8_bands-px_256-MSI-0000.tar",
        "QB": "data/QuickBird/hyper_images/QuickBird-4_bands-px_256-MSI-0000.tar",
        "WV2": "data/WorldView2/hyper_images/WorldView2-8_bands-px_256-MSI-0000.tar",
        "WV4": "data/WorldView4/hyper_images/WorldView4-4_bands-px_256-MSI-0000.tar",
        "IKONOS": "data/IKONOS/hyper_images/IKONOS-4_bands-px_256-MSI-0000.tar",
        "RS5M": "/HardDisk/ZiHanCao/datasets/RS5M/train/pub11-train-0010.tar",
        "BigEarthNetS2": "data/BigEarthNet_S2/hyper_images/BigEarthNet_data_0000.tar",
        "Xiongan": "data/Downstreams/ClassificationCollection/hyper_images/Xiongan-256bands-px_256-0000.tar",
        "fmow_MS": "data/Multispectral-FMow-full/hyper_images_8bands/FMoW-8_bands-px_1024-RGB-jp2k-80-{0069..0000}.tar",
        "fmow_RGB": "data/Fmow_rgb/hyper_images/FMoW-3_bands-RGB-{0064..0000}.tar",
    }[data_type]

    if "BigEarthNetS2" != data_type:
        permute = True
    else:
        permute = False  # BigEarthNet images are already permuted

    other_kwargs = {}
    if data_type == "RS5M":
        other_kwargs.update(
            {
                "img_key": "auto",
                "tgt_key": "img",
                "constraint_size": 65536,
                "resize_before_transform": 512,
            }
        )
    elif data_type in ["QB", "WV2", "WV4", "IKONOS", "WV3"]:
        other_kwargs.update({"mannual_img_min_max": (0, 2047)})

    _, dataloader = get_hyperspectral_dataloaders(
        wds_paths,
        batch_size=batch_size,
        num_workers=0,
        shuffle_size=-1,
        to_neg_1_1=True,
        transform_prob=0.0,
        shuffle_within_workers=False,
        remove_meta_data=False,
        prefetch_factor=None,
        permute=permute,
        resample=False,
        quantile_img_clip=1,
        **other_kwargs,
    )

    return dataloader


# * --- folder loader --- #


def ms_pan_dir_paired_loader(
    path: str,
    ms_dir_name: str = "ms",
    pan_dir_name: str = "pan",
    batch_size: int = 1,
    num_workers: int = 1,
    to_neg_1_1: bool = True,
):
    """
    Read MS (Multispectral) and PAN (Panchromatic) images from a directory, and return a dataloader.

    Args:
        path (str): Root directory path containing MS and PAN image folders
        ms_dir_name (str, optional): Name of the multispectral images directory. Defaults to "ms".
        pan_dir_name (str, optional): Name of the panchromatic images directory. Defaults to "pan".
        batch_size (int, optional): Batch size for the dataloader. Defaults to 1.
        num_workers (int, optional): Number of workers for data loading. Defaults to 1.
        to_neg_1_1 (bool, optional): Whether to normalize images to [-1,1] range. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - dataset (MSPANPairedDataset): The dataset object
            - dataloader (DataLoader): PyTorch DataLoader for the dataset

    The loader supports both .tiff and .mat file formats. Images are automatically paired
    based on filename order using natural sorting. The images are normalized to [0,1] range
    and optionally to [-1,1] range if to_neg_1_1 is True.
    """

    class MSPANPairedDataset(Dataset):
        def __init__(
            self,
            path: str,
            ms_dir_name: str = "ms",
            pan_dir_name: str = "pan",
            to_neg_1_1: bool = True,
        ):
            self.path = Path(path)
            self.ms_dir_name = ms_dir_name
            self.pan_dir_name = pan_dir_name
            self.to_neg_1_1 = to_neg_1_1

            self.ms_paths = natsorted(
                list((self.path / self.ms_dir_name).glob("*.tiff"))
                + list((self.path / self.ms_dir_name).glob("*.mat"))
            )
            self.pan_paths = natsorted(
                list((self.path / self.pan_dir_name).glob("*.tiff"))
                + list((self.path / self.pan_dir_name).glob("*.mat"))
            )
            assert len(self.ms_paths) == len(self.pan_paths), (
                f"MS and PAN images are not paired, MS is {len(self.ms_paths)}, PAN is {len(self.pan_paths)}"
            )

        def __len__(self):
            return len(self.ms_paths)

        @staticmethod
        def read_file(path: Path):
            if path.suffix == ".tiff":
                return tifffile.imread(path)
            elif path.suffix == ".mat":
                # default to read the last key
                mat_file = scipy.io.loadmat(path)
                _key = list(mat_file.keys())[-1]
                return mat_file[_key]
            else:
                raise ValueError(f"Unsupported file type: {path.suffix}")

        def get_item(self, idx):
            ms_path = self.ms_paths[idx]
            pan_path = self.pan_paths[idx]

            key = ms_path.stem

            ms_img = self.read_file(ms_path)
            pan_img = self.read_file(pan_path)

            # to 0 to 1
            ms_img = ms_img / ms_img.max()
            pan_img = pan_img / pan_img.max()
            if self.to_neg_1_1:
                ms_img = ms_img * 2 - 1
                pan_img = pan_img * 2 - 1

            return key, ms_img, pan_img

        def __getitem__(self, idx):
            key, ms_img, pan_img = self.get_item(idx)
            return {
                "__key__": key,  # same as key in webdataset
                "MS": ms_img,
                "PAN": pan_img,
            }

    dataset = MSPANPairedDataset(
        path,
        ms_dir_name=ms_dir_name,
        pan_dir_name=pan_dir_name,
        to_neg_1_1=to_neg_1_1,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
    )

    return dataset, dataloader


class HyperImageDataset(Dataset):
    """
    Webdataset sometimes read tar streams slow. We need to untar these tar files into a dir, and
    read the safetensors files in the dir with a normal torch Dataset.
    This will be faster than reading tar stream with webdataset.

    However, in NEVM maybe not a big issue, since NVEM is faster enough, and the webdataset tar files
    are more organized with multi-modalities (refer to `.utils.merge_modalities`).

    """

    def __init__(self, dir: str, return_key: bool = False, extension="safetensors"):
        self.path = Path(dir)
        self.return_key = return_key
        self.img_paths = natsorted(list(self.path.glob(f"*.{extension}")))
        self.extension = extension
        assert extension in ("safetensors", "tiff", "tif"), (
            f"Unsupported extension: {extension}. Supported extensions are 'safetensors', 'tiff', 'tif'."
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        key = img_path.stem

        # mmap load faster not as in tar data stream in webdataset
        if self.extension == "safetensors":
            img = load_file(img_path, device="cpu")["img"]
        elif self.extension in ("tiff", "tif"):
            img = tifffile.imread(self.img_paths[idx])
            img = torch.as_tensor(img).float()

        ret = {}
        if self.return_key:
            ret["__key__"] = key
        ret.update({"img": img.cpu()})

        return ret


def only_hyperspectral_img_folder_dataloader(
    dir: str,
    batch_size: int = 1,
    num_workers: int = 1,
    to_neg_1_1: bool = True,
    shuffle: bool = False,
    pin_memory: bool = False,
    hyper_transforms_lst: tuple[str, ...] | None = (
        "grayscale",
        "channel_shuffle",
        "rotation",
        "cutmix",
        "horizontal_flip",
        "vertical_flip",
    ),
    random_apply: int | tuple[int, int] = 1,
    transform_prob: float = 0.2,
    ret_key: bool = False,
    **__discard_kwargs,
):
    """
    Read hyperspectral images from a directory, and return a dataloader.
    Args:
        dir (str): Directory path containing hyperspectral images
        batch_size (int, optional): Batch size for the dataloader. Defaults to 1.
        num_workers (int, optional): Number of workers for data loading. Defaults to 1.
        to_neg_1_1 (bool, optional): Whether to normalize images to [-1,1] range. Defaults to True.
    Returns:
        tuple: A tuple containing:
            - dataset (Dataset): The dataset object
            - dataloader (DataLoader): PyTorch DataLoader for the dataset
    """

    assert os.path.exists(dir), f"Directory {dir} does not exist"
    log_print(f"discarded kwargs: {__discard_kwargs}", "debug")

    # transforms
    use_transf = (
        hyper_transforms_lst is not None
        and len(hyper_transforms_lst) > 0
        and transform_prob > 0
    )
    if use_transf:
        transform = hyper_transform(hyper_transforms_lst, transform_prob, random_apply)
        log_print(
            f"[HyperWebdataset]: use augmentations {hyper_transforms_lst} with prob {transform_prob}"
        )

    dataset = HyperImageDataset(dir, return_key=ret_key)
    if use_transf:
        collate_fn = partial(
            collate_fn_for_dict,
            augmentations=transform,  # type: ignore
            norm=True,
            to_negative_1_1=to_neg_1_1,
        )
    else:
        collate_fn = collate_fn_for_dict

    _collate_fn = torch.compile(collate_fn) if __compile_collate_fn else collate_fn
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=_collate_fn,
        generator=loader_generator,
        pin_memory=pin_memory,
        multiprocessing_context="spawn"
        if (num_workers > 0 and __compile_collate_fn)
        else None,
        timeout=30 if num_workers > 0 else 0,
    )

    return dataset, dataloader


# * --- wids loaders --- #


def get_wids_sampler(
    dataset: wids.ShardListDataset,
    shuffle_size: int,
    img_index: list[int] | None = None,
):
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        if img_index is None:
            sampler = wids.DistributedChunkedSampler(
                dataset,
                shuffle=shuffle_size > 0,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                seed=loader_seed,
            )
        else:
            sampler = IndexFilteredDistributedSampler(
                dataset,
                valid_indices=img_index,
                shuffle=shuffle_size > 0,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                seed=loader_seed,
            )
    elif shuffle_size > 0:
        if img_index is None:
            sampler = wids.ChunkedSampler(
                dataset,
                chunksize=shuffle_size,
                shuffle=True,
                seed=loader_seed,
            )
        else:
            sampler = IndexFilteredSampler(
                dataset,
                valid_indices=img_index,
                chunksize=shuffle_size,
                shuffle=True,
                seed=loader_seed,
            )
    else:
        sampler = None
    return sampler


def get_hyperspectral_wids_dataloaders(
    index_file: str | list[str],
    batch_size: int,
    num_workers: int,
    shuffle_size: int = 100,
    to_neg_1_1: bool = True,
    permute: bool = True,
    img_size_filter_file: str | None = None,
    constraint_size: int | tuple[int, int] | None = None,
    resize_before_transform: int | tuple[int, int] | None = 512,
    tgt_key: str | None = "img",
    quantile_img_clip: Annotated[float, "0.0<f<1.0"] = 0.99,
    manual_img_max: float | None = None,
    hyper_transforms_lst: tuple[str, ...] | None = (
        "grayscale",
        "channel_shuffle",
        "rotation",
        "cutmix",
        "horizontal_flip",
        "vertical_flip",
    ),
    transform_prob: float = 0.2,
    random_apply: int | tuple[int, int] = 1,
    prefetch_factor: int | None = 6,
    pin_memory: bool = False,
    remove_meta_data: bool = False,
    persistent_workers: bool = False,
    wids_local_name_prefix: str | None = None,
    filter_out_none: bool = True,
    **__discard_kwargs,  # discard other kwargs
):
    if isinstance(index_file, (list, tuple)):
        index_infos: list[dict] = []
        for p in index_file:
            assert os.path.exists(p), f"Index file {p} does not exist"
            index_info, _ = get_wids_index_json_info(p)
            index_infos.extend(index_info)
        index_file = index_infos

    decodes = [
        partial(  # type: ignore
            wids_image_decode,
            to_neg_1_1=to_neg_1_1,
            permute=permute,
            resize=None,
            process_img_keys="ALL",
            quantile_clip=quantile_img_clip,
            manual_img_max=manual_img_max,
        )
    ]
    if constraint_size is not None:
        decodes.append(wids_filter_img_size(constraint_size=constraint_size))
    dataset = wids.ShardListDataset(
        index_file,
        localname=lambda x: f"{wids_local_name_prefix}/{x}"
        if wids_local_name_prefix is not None
        else x,
        transformations=decodes,
    )

    # image size filtering
    img_index = None
    if constraint_size is not None and img_size_filter_file is not None:
        img_index = wids_img_size_filter_by_parquet_index(
            img_size_filter_file,
            min_size=constraint_size
            if isinstance(constraint_size, int)
            else constraint_size[0],
            max_size=None if isinstance(constraint_size, int) else constraint_size[1],  # type: ignore
        )

    # dataloader
    sampler = get_wids_sampler(dataset, shuffle_size, img_index)
    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        drop_last=False,
        sampler=sampler,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        generator=loader_generator,
    )

    # rename
    dataloader = dataloader.map(partial(remove_dot_and_extensions, tgt_key=tgt_key))

    # filter out None
    def filter_out_none_fn(sample):
        for k, v in sample.items():
            if v is None and not k.startswith("__"):
                return None

        return sample

    if filter_out_none:
        dataloader = dataloader.map(filter_out_none_fn, handler=wds.warn_and_continue)

    # resize
    if resize_before_transform is not None:
        if isinstance(resize_before_transform, int):
            resize_before_transform = to_2tuple(resize_before_transform, 2)
        resizer = large_image_resizer_clipper(
            tgt_size=resize_before_transform, op_for_large="clip"
        )
        dataloader = dataloader.map(resizer, handler=wds.warn_and_continue)
    else:
        log_print(
            "[Wids dataset]: no resize function, the images are used as the original size "
            "that may cause batching dimension mismatch",
            "warning",
        )

    # batched
    dataloader = dataloader.batched(batch_size)

    # remove meta
    if remove_meta_data:
        dataloader = dataloader.map(remove_meta_data_fn)

    # augmentations
    use_transf = (
        hyper_transforms_lst is not None
        and len(hyper_transforms_lst) > 0
        and transform_prob > 0
    )
    if use_transf:
        transform = hyper_transform(hyper_transforms_lst, transform_prob, random_apply)  # type: ignore
        log_print(
            f"[Wids dataset]: use augmentations {hyper_transforms_lst} with prob {transform_prob}"
        )
        # dataloader = dataloader.map_dict(img=transform)
        # img keys with one same transform?
        assert tgt_key is not None, "tgt_key must be specified"
        tgt_key = [tgt_key] if isinstance(tgt_key, str) else list(tgt_key.values())  # type: ignore
        dataloader = dataloader.map_dict(**{k: transform for k in tgt_key})

    return dataset, dataloader


@function_config_to_basic_types
def get_hyperspectral_img_loaders_with_different_backends(
    paths: WebdatasetPath,
    loader_type: SupportedLoaderType = "webdataset",
    # curriculums
    curriculum_type: str | None = None,
    curriculum_kwargs: dict | None = None,
    # * there are three choices to provide the loaders' configuration:
    # 1. provide a basic config and change some of cfg by different loaders
    basic_kwargs: dict | None = None,
    changed_kwargs_by_loader: list[dict] | None = None,
    # 2. every loader has its own kwargs
    rep_loader_kwargs: list[dict] | None = None,
    chain_loader_infinit: bool = True,
    shuffle_loaders: bool = True,
    # 3. simple for all loaders
    **loader_kwargs,
):
    """
    Get hyperspectral image dataloaders with flexible configuration options.

    This function supports loading hyperspectral image data using different backends
    (WebDataset for tar files or folder-based loaders) with configurable options
    for each data source.

    Args:
        paths (WebdatasetPath): Path(s) to data sources. Can be:
            - A single string path to a tar file or directory
            - A list of string paths for multiple tar files
            - A list of lists of strings for grouped data sources
        loader_type (str, optional): The data loading backend to use:
            - "webdataset": For WebDataset tar files (default)
            - "folder": For directory of safetensors image files
        basic_kwargs (dict, optional): Base configuration applied to all loaders.
        changed_kwargs_by_loader (list[dict], optional): List of dicts with parameters
            to override in basic_kwargs for each data source group.
        rep_loader_kwargs (list[dict], optional): Complete configuration for each data
            source group. If provided, each dict configures a separate loader.
        chain_loader_infinit (bool, optional): If True, the chained loader will repeat
            infinitely. If False, it will stop after one complete cycle. Defaults to True.
        **loader_kwargs: Default configuration parameters applied to all loaders if neither
            basic_kwargs/changed_kwargs_by_loader nor rep_loader_kwargs are provided.

    Returns:
        tuple:
            - For "webdataset" with multiple groups: (list of datasets, chained dataloader)
            - For "webdataset" with single group: (dataset, dataloader)
            - For "folder": (dataset, dataloader) for directory images

    Raises:
        ValueError: If an unsupported loader_type is provided or input paths are invalid
        AssertionError: If configuration parameters don't match expected formats
    """

    # fluid webdataset/datapipline
    if loader_type == "webdataset":
        log_print("Using WebDataset loader")

        def expand_paths_and_correct_loader_kwargs(
            paths: str | list[str], loader_kwargs: dict
        ):
            # Ensure that the number of workers is not greater than the number of shards
            if isinstance(paths, str):
                paths = list(braceexpand.braceexpand(paths))
                _len = len(paths)
            elif isinstance(paths, (list, tuple)):
                path_exp = []
                for p in paths:
                    assert isinstance(p, str), (
                        "'paths' should be a list of strings or a string that can be expanded"
                        f"list of {type(p)} is not allowed."
                    )
                    lst_p = list(braceexpand.braceexpand(p))
                    path_exp.extend(lst_p)
                _len = len(path_exp)
                paths = path_exp
            else:
                raise ValueError(
                    f"paths should be a string or a list of strings, but got {type(paths)}"
                )

            assert _len > 0, f"paths should not be empty, but got {paths}"

            for p in paths:
                # assert tar file exists
                assert os.path.exists(p), f"tar file {p} does not exist"

            # n_worker should less than the number of shards
            n_workers = loader_kwargs.get("num_workers", 0)
            if _len < n_workers and n_workers > 0:
                # set n_workers to the number of shards
                loader_kwargs["num_workers"] = _len
                log_print(
                    f"n_workers={n_workers} is larger than the number of shards {_len}, set n_workers={_len}",
                    level="debug",
                )

            return paths, loader_kwargs

        # Groups of dataloaders
        if isinstance(paths, list) and isinstance(paths[0], Sequence):
            log_print(
                "input paths contains list of lists, we will chain the dataloader with each loader"
            )
            if rep_loader_kwargs is not None:
                log_print(
                    f"rep_loader_kwargs is provided: {rep_loader_kwargs}", "debug"
                )
                assert isinstance(rep_loader_kwargs, list), (
                    f"rep_loader_kwargs should be a list, but got {type(rep_loader_kwargs)}"
                )
                assert len(rep_loader_kwargs) == len(paths), (
                    f"rep_loader_kwargs should be the same length as paths, "
                    f"but got {len(rep_loader_kwargs)} and {len(paths)}"
                )
            elif basic_kwargs is not None:
                log_print("basic_kwargs is provided", "debug")
                if changed_kwargs_by_loader is None:
                    changed_kwargs_by_loader = [{}] * len(paths)

                # assertions
                assert isinstance(basic_kwargs, dict), (
                    f"basic_kwargs should be a dict, but got {type(basic_kwargs)}"
                )
                assert isinstance(changed_kwargs_by_loader, list), (
                    f"changed_kwargs_by_loader should be a list, but got {type(changed_kwargs_by_loader)}"
                )
                assert len(changed_kwargs_by_loader) == len(paths), (
                    f"changed_kwargs_by_loader should be the same length as paths, "
                    f"but got {len(changed_kwargs_by_loader)} and {len(paths)}"
                )

                rep_loader_kwargs = generate_wds_config_modify_only_some_kwgs(
                    basic_kwargs, changed_kwargs_by_loader
                )
            else:
                log_print("rep_loader_kwargs is None, use the loader_kwargs", "debug")
                rep_loader_kwargs = [loader_kwargs] * len(paths)

            # * --- build datasets and dataloaders --- #
            datasets = []
            dataloaders = []
            for i, (p_lst, loader_kwargs) in enumerate(zip(paths, rep_loader_kwargs)):
                # assertions
                assert isinstance(p_lst, (list, tuple)), (
                    f"paths {p_lst} should be a list of lists, but got {type(p_lst)}"
                )
                assert len(p_lst) > 0, f"paths should not be empty, but got {p_lst}"

                if len(p_lst) == 1:
                    p_lst = p_lst[0]
                    assert isinstance(p_lst, str), (
                        f"paths should be a list of strings, but got {type(p_lst)} for paths {p_lst}"
                    )
                else:
                    p_lst = list(flatten_nested_list(p_lst))
                    for p in p_lst:
                        assert isinstance(p, str), (
                            f"paths should be a list of strings, but got {type(p)} for paths {p_lst}"
                        )

                # resample must be false
                if not shuffle_loaders:
                    loader_kwargs["resample"] = False
                loader_kwargs["epoch_len"] = -1

                p_lst, loader_kwargs = expand_paths_and_correct_loader_kwargs(
                    p_lst, loader_kwargs
                )
                log_print(
                    f"dataset group {i} gets paths: \n<cyan>[{'\n'.join(p_lst)}]</>\n"
                    + "-" * 30
                )

                dataset, dataloader = get_hyperspectral_dataloaders(
                    p_lst, **loader_kwargs
                )
                datasets.append(dataset)
                dataloaders.append(dataloader)

            # curriculum
            if curriculum_type is not None:
                assert curriculum_kwargs is not None, (
                    f"curriculum_kwargs must be provided if {curriculum_type=}."
                )
                curriculum_fn = get_curriculum_fn(  # type: ignore
                    c_type=curriculum_type,
                    **curriculum_kwargs,
                )
                log_print(
                    f"use {curriculum_type} curriculum learning with kwargs: {curriculum_kwargs}",
                    "debug",
                )
            else:
                curriculum_fn = None

            # make the chainable dataloader
            if not chain_loader_infinit:
                log_print(
                    "chain loader generator is finite, the loader can not be iter and next. "
                    "Please set <cyan>chain_loader_infinit=True</> if you are know what you are doing.",
                    "warning",
                )
            dataloader = chained_dataloaders(
                dataloaders, chain_loader_infinit, shuffle_loaders, curriculum_fn
            )
            return datasets, dataloader
        else:
            paths, loader_kwargs = expand_paths_and_correct_loader_kwargs(
                paths, loader_kwargs
            )
            log_print(
                f"dataset gets paths: \n<green>[{'\n'.join(paths)}</>\n" + "-" * 30
            )
            return get_hyperspectral_dataloaders(paths, **loader_kwargs)

    # torch dataset
    elif loader_type == "folder":
        log_print("Using folder loader")

        assert rep_loader_kwargs is None, (
            "rep_loader_kwargs is not supported for folder loader"
        )
        assert isinstance(paths, str), (
            f"paths should be a string, but got {type(paths)}"
        )

        return only_hyperspectral_img_folder_dataloader(paths, **loader_kwargs)
    else:
        raise ValueError(f"Unsupported loader type: {loader_type}")


@function_config_to_basic_types
def get_hyperspectral_img_loaders_with_different_backends_v2(
    paths: WebdatasetPath,
    loader_type: SupportedLoaderType | None = None,
    # curriculums
    curriculum_type: str | None = None,
    curriculum_kwargs: dict | None = None,
    # * there are three choices to provide the loaders' configuration:
    # 1. provide a basic config and change some of cfg by different loaders
    basic_kwargs: dict | None = None,
    changed_kwargs_by_loader: list[dict] | None = None,
    # 2. every loader has its own kwargs
    rep_loader_kwargs: list[dict] | None = None,
    chain_loader_infinit: bool = True,
    shuffle_loaders: bool = True,
    # 3. simple for all loaders
    **loader_kwargs,
):
    """
    Get hyperspectral image dataloaders with flexible configuration options.

    This function supports loading hyperspectral image data using different backends
    (WebDataset for tar files or folder-based loaders) with configurable options
    for each data source.

    Args:
        paths (WebdatasetPath): Path(s) to data sources. Can be:
            - A single string path to a tar file or directory
            - A list of string paths for multiple tar files
            - A list of lists of strings for grouped data sources
        loader_type (str, optional): The data loading backend to use:
            - "webdataset": For WebDataset tar files (default)
            - "folder": For directory of safetensors image files
        basic_kwargs (dict, optional): Base configuration applied to all loaders.
        changed_kwargs_by_loader (list[dict], optional): List of dicts with parameters
            to override in basic_kwargs for each data source group.
        rep_loader_kwargs (list[dict], optional): Complete configuration for each data
            source group. If provided, each dict configures a separate loader.
        chain_loader_infinit (bool, optional): If True, the chained loader will repeat
            infinitely. If False, it will stop after one complete cycle. Defaults to True.
        **loader_kwargs: Default configuration parameters applied to all loaders if neither
            basic_kwargs/changed_kwargs_by_loader nor rep_loader_kwargs are provided.

    Returns:
        tuple:
            - For "webdataset" with multiple groups: (list of datasets, chained dataloader)
            - For "webdataset" with single group: (dataset, dataloader)
            - For "folder": (dataset, dataloader) for directory images

    Raises:
        ValueError: If an unsupported loader_type is provided or input paths are invalid
        AssertionError: If configuration parameters don't match expected formats
    """
    loader_types: list[SupportedLoaderType] = []
    if loader_type is not None:
        log_print(
            f"loader_type is provided: {loader_type}, "
            f"input all paths should be {loader_type} loader",
            "debug",
        )

    # clear the kwargs
    # 1. paths is a list of list of string
    if isinstance(paths, list) and isinstance(paths[0], Sequence):
        log_print(
            "input paths contains list of lists, we will chain the dataloader with each loader"
        )
        if rep_loader_kwargs is not None:  # every loader kwargs
            log_print(f"rep_loader_kwargs is provided: {rep_loader_kwargs}", "debug")
            assert isinstance(rep_loader_kwargs, list), (
                f"rep_loader_kwargs should be a list, but got {type(rep_loader_kwargs)}"
            )
            assert len(rep_loader_kwargs) == len(paths), (
                f"rep_loader_kwargs should be the same length as paths, "
                f"but got {len(rep_loader_kwargs)} and {len(paths)}"
            )
        elif basic_kwargs is not None:
            log_print("basic_kwargs is provided", "debug")
            if changed_kwargs_by_loader is None:
                changed_kwargs_by_loader = [{}] * len(paths)

            assert isinstance(basic_kwargs, dict), (
                f"basic_kwargs should be a dict, but got {type(basic_kwargs)}"
            )
            assert isinstance(changed_kwargs_by_loader, list), (
                f"changed_kwargs_by_loader should be a list, but got {type(changed_kwargs_by_loader)}"
            )
            assert len(changed_kwargs_by_loader) == len(paths), (
                f"changed_kwargs_by_loader should be the same length as paths, "
                f"but got {len(changed_kwargs_by_loader)} and {len(paths)}"
            )

            rep_loader_kwargs = generate_wds_config_modify_only_some_kwgs(
                basic_kwargs, changed_kwargs_by_loader
            )
        else:
            log_print("rep_loader_kwargs is None, use the loader_kwargs", "debug")
            rep_loader_kwargs = [loader_kwargs] * len(paths)

        for i in range(len(rep_loader_kwargs)):
            kwg_ldt = rep_loader_kwargs[i].pop("loader_type", "webdataset")
            loader_types.append(kwg_ldt if kwg_ldt is not None else "webdataset")
    # 2. paths is a list of strings
    elif isinstance(paths, (list, tuple)) and any(isinstance(p, str) for p in paths):
        raise NotImplementedError(
            "paths is a list of strings, please use a list of lists of strings instead. For example: "
            "paths=[data1/{0000..0002}.tar, [data2/{0000..0002}.tar, data2/0003.tar]] should be "
            "paths=[[data1/{0000..0002}.tar], [data2/{0000..0002}.tar, data2/0003.tar]]"
        )
    # 3. paths is a string
    else:
        log_print("input paths is a string, use the loader_kwargs", "debug")
        if len(loader_kwargs) != 0:
            assert basic_kwargs is None and changed_kwargs_by_loader is None, (
                "loader_kwargs should be used only when basic_kwargs and changed_kwargs_by_loader are None"
            )
            rep_loader_kwargs = [loader_kwargs]
        elif basic_kwargs is not None:
            assert len(loader_kwargs) == 0 and changed_kwargs_by_loader is None, (
                "basic_kwargs should be used only when loader_kwargs is empty and changed_kwargs_by_loader is None"
            )
            rep_loader_kwargs = [basic_kwargs]
        else:
            raise ValueError(
                "when paths is a string, loader_kwargs should not be provided or "
                "should be empty when basic_kwargs is provided, changed_kwargs_by_loader should always be None"
            )
        kwg_ldt = rep_loader_kwargs[0].pop("loader_type", "webdataset")
        loader_types.append(kwg_ldt if kwg_ldt is not None else "webdataset")

    # * --- build datasets and dataloaders --- #
    datasets = []
    dataloaders = []

    # for-loop over the paths and rep_loader_kwargs
    for i, (p_lst, loader_kwargs, loader_type) in enumerate(
        zip(paths, rep_loader_kwargs, loader_types)
    ):
        # one group of datasets that have the same channel
        p_lst: list[str] | str

        # > webdataset or wids loader
        if loader_type in ("webdataset", "wids"):
            # assertions
            assert isinstance(p_lst, (list, tuple)), (
                f"paths should be a list of lists, but got {type(p_lst)}"
            )
            assert len(p_lst) > 0, f"paths should not be empty, but got {p_lst}"

            p_lst = list(flatten_nested_list(p_lst))  # type: ignore
            for p in p_lst:
                assert isinstance(p, str), (
                    f"paths should be a list of strings, but got {type(p)} for paths {p_lst}"
                )

            # resample must be false
            if not shuffle_loaders:
                loader_kwargs["resample"] = False
            loader_kwargs["epoch_len"] = -1

            p_lst, loader_kwargs = expand_paths_and_correct_loader_kwargs(
                loader_type, p_lst, loader_kwargs
            )
            log_print(
                f"dataset group {i} gets paths: \n<green>[{'\n'.join(p_lst)}]</>\n"
                + "-" * 30
            )

            # construct webdataset/wids datasets and dataloaders
            if loader_type == "webdataset":
                log_print("Using webdataset loader")
                dataset, dataloader = get_hyperspectral_dataloaders(
                    p_lst, **loader_kwargs
                )
            elif loader_type == "wids":
                log_print("Using wids loader")
                if isinstance(p_lst, list) and len(p_lst) == 1:
                    p_lst = p_lst[0]

                dataset, dataloader = get_hyperspectral_wids_dataloaders(
                    index_file=p_lst,
                    **loader_kwargs,
                )
            else:
                raise ValueError(f"loader_type {loader_type} is not supported")

            datasets.append(dataset)
            dataloaders.append(dataloader)

        # > folder loader
        elif loader_type == "folder":
            log_print("Using folder loader")
            assert isinstance(paths, str), f"paths should be a string, but got paths"
            return only_hyperspectral_img_folder_dataloader(paths, **loader_kwargs)
        else:
            raise ValueError(f"Unsupported loader type: {loader_type}")

    # > prepare for curriculum
    if curriculum_type is not None:
        assert curriculum_kwargs is not None, (
            f"curriculum_kwargs must be provided if {curriculum_type=}."
        )
        curriculum_fn = get_curriculum_fn(  # type: ignore
            c_type=curriculum_type,
            **curriculum_kwargs,
        )
    else:
        curriculum_fn = None

    # make the chainable dataloader
    if not chain_loader_infinit:
        log_print(
            "chain loader generator is finite, the loader can not be iter and next. "
            "Please set <cyan>chain_loader_infinit=True</> if you are know what you are doing.",
            "warning",
        )

    # > prepare chained unified dataloader
    dataloader = chained_dataloaders(
        dataloaders, chain_loader_infinit, shuffle_loaders, curriculum_fn
    )

    return datasets, dataloader


if __name__ == "__main__":
    import glob

    from src.utilities.logging import set_logger_file

    set_logger_file(
        file="error.log",
        level="warning",
        add_time=False,
    )

    # Test config
    test_wds_path = [
        # ["data/MMSeg_YREB/hyper_images/MMSeg_YREB_train_part-12_bands-MSI-0003.tar"],
        # ["data/DCF_2019/hyper_images/DCF_2019_Track_2-8_bands-px_512-MSI-0017.tar"],
        # ["data/DCF_2020/hyper_images/DFC_2020_public-13_bands-px_256-MSI-0002.tar"],
        # ["data/Houston/hyper_images/Houston-50_bands-px_512-MSI-0000.tar"],
        # ["data/GID-GF2/hyper_images/GID-GF2-test-3_bands-px_512-MSI-0001.tar"],
        # ["data/WorldView3/hyper_images/WorldView3-8_bands-px_256-MSI-0000.tar"],
        ["data/DryadHyper/hyper_images/DryadHyper-224_bands-px_128-MSI-0002.tar"],
        # ["data/OHS/hyper_images/OHS-32_bands-px_512-MSI-0015.tar"],
        # [
        #     "data_local/GID-GF2/hyper_images/GID-GF2-train-3_bands-px_512-MSI-{0000..0003}.tar",
        #     "data_local/GID-GF2/hyper_images/GID-GF2-val-3_bands-px_512-MSI-0000.tar",
        #     "data_local/GID-GF2/hyper_images/GID-GF2-test-3_bands-px_512-MSI-{0000..0001}.tar",
        # ],
        # ["data/WorldView3/hyper_images/WorldView3-PAN-1_bands-px_512-MSI-0000.tar"],
        # ["data/MDAS-Optical/MDAS-Optical-4_bands-px_512-MSI-0000.tar"],
        # ["data/BigEarthNet_S2/hyper_images/BigEarthNet_data_{0000..0101}.tar"],
        # ["data/MDAS-HySpex/MDAS-HySpex-368_bands-px_256-MSI-{0000..0003}.tar"],
        # ["data/TUM_128/hyper_images/TUM_128_data_{0000..0006}.tar"],
        # [p.as_posix() for p in Path("data/RS5M").glob("**/*.tar")]
        # ["/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/shardindex.json"]
        # [
        #     "/HardDisk/ZiHanCao/datasets/RS5M/train/pub11-train-{0000..0031}.tar",
        #     # "/HardDisk/ZiHanCao/datasets/RS5M/train/rs3-train-{0000..0007}.tar",
        #     # "/HardDisk/ZiHanCao/datasets/RS5M/val/pub11-val-{0000..0031}.tar",
        #     # "/HardDisk/ZiHanCao/datasets/RS5M/val/rs3-val-{0000..0031}.tar",
        # ],
        # ["data/miniFrance/miniFrance_labeled_unlabeled_3_bands-px_1024-MSI-0000.tar"],
        # ["data/MUSLI/hyper_images/MUSLI_MSI-438_bands-px_512-MSI-jp2k-80-0000.tar"]
        # [
        #     "data/DOTA_v1/hyper_images/DOTA_v1-3_bands-px_1024-{0000..0001}.tar",
        #     "data/DIOR_RSVG_Dataset/hyper_images/DIOR_RSVG_3_bands-px_800-RGB-jp2k-80-0000.tar",
        #     "data/InriaAerialLabelingDataset/hyper_images/InriaAerialLabelingDataset-3_bands-px_512-RGB-jp2k-80-0000.tar",
        #     "data/LoveDA/hyper_images/LoveDA-3_bands-px_1024-0000.tar",
        # "data/OpenEarthMap/hyper_images/OpenEarthMap-3_bands-px_1024-0000.tar",
        #     "data/RefSegRS/hyper_images/RefSegRS_3_bands-px_512-RGB-jp2k-80-0000.tar",
        #     "data/RSCaptions/hyper_images/RSCaptionCollection-hyper_images-0000.tar",
        #     "data/RSCaptions/hyper_images/RSCaptionCollection-RSICD-0000.tar",
        #     "data/RSCaptions/hyper_images/RSCaptionCollection-RSITMD-0000.tar",
        #     "data/RSCaptions/hyper_images/RSCaptionCollection-Sydney_caption-0000.tar",
        #     "data/RSCaptions/hyper_images/RSCaptionCollection-UCM-0000.tar",
        #     "data/UDD/hyper_images/UDD-3_bands-px_512-RGB-jp2k-95-0000.tar",
        #     "data/VDD/hyper_images/VDD_3_bands-px_2k-RGB-0000.tar",
        #     "data/xView2/hyper_images/xView2-3_bands-px_1024-RGB-jp2k-80-0000.tar",
        #     "data/miniFrance/miniFrance_labeled_unlabeled_3_bands-px_1024-MSI-{0000..0010}.tar",
        #     "data/miniFrance/miniFrance_test_3_bands-px_1024-MSI-{0000..0017}.tar",
        #   "data/ERA_UAV_Video_Dataset/hyper_images/ERA_UAV_Video_Dataset_key_frames_3_bands-px_512-RGB-jp2k-80-0000.tar",
        # ],
        # [
        #     "data/BigEarthNet_S2/hyper_images_compressed/BigEarthNet_data_{0000..0006}.tar"
        # ]
        # ["data/HyperGlobal/hyper_images/HyperGlobal450k-150_bands-px_128_0018.tar"],
        # [
        #     "data/hyspecnet11k/hyper_images/hyspecnet11k_202_bands_px_128_0000.tar"
        # ],  # [39, 32, 16]
        # ["data/TEOChatlas/train/GeoChat_Instruct_images2.tar"]
        # ["data/TEOChatlas/train/TEOChatlas_images.tar"],
        # ["data/CityBench-CityData/hyper_images/CityBench-CityData-0000.tar"],
        # [p.as_posix() for p in Path("data/MMOT/hyper_images").glob("*.tar")],
        # [
        # [
        #     *glob.glob("data/MMOT/conditions/**/*.tar", recursive=True),
        #     *glob.glob("data/RSCaptions/conditions/**/*.tar", recursive=True),
        #     "data/DCF_2020/conditions/DFC_2020_public-13_bands-px_256-MSI-{0000..0002}.tar",
        #     "data/DCF_2019/conditions/DCF_2019_Track_2-8_bands-px_512-MSI-{0000..0015}.tar",
        #     "data/BigEarthNet_S2/conditions/BigEarthNet_data_{0004..0006}.tar",
        #     "data/Houston/conditions/Houston-50_bands-px_512-MSI-0000.tar",
        #     "data/LoveDA/conditions/LoveDA-3_bands-px_1024-0000.tar",
        #     "data/MDAS-HySpex/conditions/MDAS-HySpex-368_bands-px_256-MSI-{0000..0003}.tar",
        #     "data/MMSeg_YREB/conditions/MMSeg_YREB_train_part-12_bands-MSI-{0000..0003}.tar",
        #     "data/RefSegRS/conditions/RefSegRS_3_bands-px_512-RGB-jp2k-80-0000.tar",
        #     "data/QuickBird/conditions/QuickBird-4_bands-px_256-MSI-0000.tar",
        #     "data/WorldView2/conditions/WorldView2-8_bands-px_256-MSI-0000.tar",
        #     "data/WorldView3/conditions/WorldView3-8_bands-px_256-MSI-0000.tar",
        # ]
        # ["data/EarthView/hyper_images/satellogic/shard1.tar"]
        # ["data/EarthView/hyper_images/neon/neon-{0000..0013}.tar"],
        # ["data/Multispectral-Spacenet-series/05_SN6_buildings_PS-RGB.tar"]
        # ["data/Multispectral-Spacenet-series/00_SN1_buildings_3band.tar"],
        # ["data/Multispectral-Spacenet-series/02_SN7_buildings_images.tar"]
        # ["data/RemoteSAM270k/RemoteSAM-270K/RemoteSAM270K.tar"]
        # ["data/DCF_2019/conditions/DCF_2019_Track_2-8_bands-px_512-MSI-0010.tar"],
        # [
        #     ["data/RS5M/train/pub11-train-0006.tar"],
        #     ["data/Fmow_rgb/hyper_images/FMoW-3_bands-RGB-0021.tar"],
        # ]
        # ["data/HyperGlobal/hyper_images/HyperGlobal-GF5-bands-px_64_0002.tar"]
        # ["data/Multispectral-FMow-full/hyper_images_8bands/shardindex.json"]
        # ["data/WDC/hyper_images/Washington_DC_mall-191_bands-px_160-0000.tar"]
        # ["data/Downstreams/UrbanUnmixing/Urban_188_em4_init.tar"]
        # ["data/Downstreams/ChangeDetection/OSCD/OSCD_13bands_train.tar"]
        # ["data/BigEarthNet_S2/conditions/BigEarthNet_data_{0000..0006}.tar"]
        # ["data/EarthView/hyper_images/neon/neon-{0000..0013}.tar"]
        # ["data/MUSLI/hyper_images/shardindex.json"]
    ]
    test_batch_size = 2
    test_num_workers = 0
    test_shuffle_size = -1

    loader_kwargs = dict(
        loader_type="webdataset",
        batch_size=test_batch_size,
        num_workers=test_num_workers,
        shuffle_size=test_shuffle_size,
        to_neg_1_1=True,
        transform_prob=0.0,
        random_apply=(1, 2),
        pin_memory=False,
        prefetch_factor=2,
        remove_meta_data=False,
        resize_before_transform=128,
        shuffle_within_workers=False,
        resample=True,
    )
    changed_kwargs = [
        # {"img_key": ["hsi", "rgb"]},
        # {"img_key": ["npy"], "tgt_key": ["img"]},
        # {
        #     "img_key": ["mlsd", "sketch", "hed", "segmentation"],
        #     "tgt_key": None,
        #     "keys_to_remove": ["rgb.png"],
        #     "resize_before_transform": 64,
        #     "resample": False,
        # },
        # {"loader_type": "wids", "img_key": "auto", "tgt_key": "img"},
        {"img_key": "auto"},
        # {"img_key": ["img"], "keys_to_remove": ["metadata"]}
        # {"img_key": ["rgb"], "tgt_key": ["img"], "keys_to_remove": ["hsi"]},
        # {"img_key": "npy", "tgt_key": "img"},
        # {
        #     # "img_key": ["img1", "img2"],
        #     "tgt_key": "img",
        #     "random_one_key": True,
        #     "manual_img_max": 3800.0,
        #     "keys_to_remove": ["gt"],
        # }
        # {"permute": False},
        # {
        #     "loader_type": "wids",
        #     "is_structured_tar": False,
        #     "img_key": "auto",
        #     "constraint_size": 128 * 128,
        # },
        # {
        #     "loader_type": "wids",
        #     "tgt_key": None,
        # },
        # {
        #     "img_key": "auto",
        #     "random_one_key": True,
        #     "img_key": None,
        #     "tgt_key": None,
        #     "undecoded_filtered": False,
        #     "transform_prob": 0.0,
        # },
        # {"img_key": "npy"},
        # {"check_nan": True},
        # {
        #     "img_key": "img_content",
        #     "tgt_key": "img",
        #     "constraint_size": 128 * 128,
        #     "resize_before_transform": 512,
        # },
        # {
        #     "img_key": "img_content",
        #     "constraint_size": 256 * 256,
        #     "resize_before_transform": 512,
        # },
    ]
    curriculum_kwargs = {
        "start_prob": [0.3, 2.0],
        "end_prob": [0.5, 0.5],
        "total_steps": 600,
    }
    # channels = [12, 8, 13, 50, 4, 8, 224]
    accelerator = accelerate.Accelerator()

    step_counter = StepsCounter(["train"])
    _, test_loader = get_hyperspectral_img_loaders_with_different_backends_v2(
        test_wds_path,
        loader_type="webdataset",
        basic_kwargs=loader_kwargs,
        changed_kwargs_by_loader=changed_kwargs,
        chain_loader_infinit=False,
        shuffle_loaders=False,
    )

    from tqdm import tqdm

    indices = set()

    sizes = {}
    # for i, sample in enumerate((tbar := tqdm(test_loader))):

    test_loader = iter(test_loader)
    tbar = tqdm()
    import matplotlib.pyplot as plt
    import PIL.Image

    i = 0
    while True:
        sample = next(test_loader)

        # print(sample.keys(), sample["segmentation"].shape)
        # print(sample.keys())

        # img = sample["img"]
        # img = img[0].permute(1, 2, 0)[..., [32, 18, 9]]
        # img = (img * 255.0).to(torch.uint8)
        # img = img.cpu().numpy()
        # PIL.Image.fromarray(img).save(f"data/Downstreams/UrbanUnmixing/patch_{i}.jpg")

        i += 1

        tbar.update(1)
        # continue

        # img = sample["segmentation"]
        # img =sample['img']
        # assert "img" not in sample and "rgb" not in sample, f"{sample.keys()}"

        # update sizes of the count
        # s = np.prod(img.shape[-2:])
        # if s not in sizes:
        #     sizes[s] = img.shape[0]
        # else:
        #     sizes[s] += img.shape[0]

        # index = [int(url.split("_")[-1].split(".")[0]) for url in sample["__url__"]]
        # index = set(index)
        # indices.update(index)

        # tbar.set_description_str(
        #     "rank: {}, img shape: {}, loader idx: {}, url: {}, exists indices: {}".format(
        #         accelerator.process_index,
        #         img.shape,
        #         sample["__loader_idx__"],
        #         sample["__url__"][0],
        #         indices,
        #         # (img.min().item(), img.max().item()),
        #     )
        # )

        # if i == 2000:
        #     break

        step_counter.update("train")

        # plot a batch
        # import torchvision.utils as vutils

        # img_s = (
        #     vutils.make_grid(
        #         (img[:8,] + 1) / 2,
        #         # img / 2 + 0.5,
        #         nrow=4,
        #         padding=2,
        #         normalize=True,
        #     )
        #     .permute(1, 2, 0)
        #     .cpu()
        #     .numpy()
        #     * 255.0
        # ).astype(np.uint8)

        # import PIL.Image as Image

        # Image.fromarray(img_s).save(f"S2_compress_{i}.png")
        # if i == 6:
        #     exit(0)

    # * plot the image size bar image
    # import matplotlib.pyplot as plt

    # # group the sizes into n groups
    # ngroups = 50
    # uniq_sizes = sorted(sizes.keys())
    # # group the sizes into n groups
    # grouped_sizes = {k: 0 for k in uniq_sizes[:: len(uniq_sizes) // ngroups]}
    # for size, count in sizes.items():
    #     # find the closest key in grouped_sizes
    #     closest_key = min(grouped_sizes.keys(), key=lambda x: abs(x - size))
    #     grouped_sizes[closest_key] += count
    # sizes = grouped_sizes
    # # plot the sizes
    # plt.figure(figsize=(10, 6))

    # plt.bar(
    #     [np.sqrt(k) for k in sizes.keys()],
    #     list(sizes.values()),
    #     width=20,
    #     align="center",
    #     # color="blue",
    #     # alpha=0.7,
    # )
    # plt.xlabel("Image Size (sqrt of pixels)")
    # plt.ylabel("Number of Samples")
    # plt.title("Distribution of Image Sizes in Hyperspectral Dataset")
    # plt.xticks(rotation=45)
    # plt.grid(axis="y")
    # plt.tight_layout()
    # plt.savefig("hyperspectral_image_sizes.png", dpi=300)

    # if i % 200 == 0:
    #     # the top-10 sizes
    #     sz_lst = sorted(sizes.items(), key=lambda x: x[1], reverse=True)[:10]
    #     sz_lst = [f"{np.round(np.sqrt(k), 1).item()}: {v}" for k, v in sz_lst]
    #     print(f"top-10 sizes: {sz_lst}, total samples: {sum(sizes.values())}")

    # assert img.shape[-2] == img.shape[-1]
    # assert torch.isnan(img).sum() == 0, "Image contains NaN values"
    # assert img.shape[1] < img.shape[-1]

    # step_counter.update("train")
    # if img.shape[1] == 3:
    #     print(sample["__url__"])
    #     break
    # if img.shape[1] not in channels:
    #     print(sample["__url__"])
    #     raise ValueError(f"Unknown image channel: {img.shape[1]}")

    #     if img.shape[-1] == 256:
    #         total_samples_256 += img.shape[0]
    #     elif img.shape[-1] == 512:
    #         total_samples_512 += img.shape[0]
    #     else:
    #         raise ValueError(f"Unknown image size: {img.shape[-1]}")

    # print("pixel 256 size total samples: ", total_samples_256)
    # print("pixel 512 size total samples: ", total_samples_512)

    # * for loop the images
    # for batch in test_loader:
    #     # print(batch['img'].shape, batch.keys())
    #     tar_paths = batch["__url__"]

    #     # assert all tar paths are the same
    # print(tar_paths)
    # assert all(
    #     tar_paths[0] == tar_path for tar_path in tar_paths
    # ), "tar paths are not the same"

    # img_tensor = batch["img"]  # shape: [N, C, H, W]
    # print(f"{img_tensor.shape}")

    # * plot the grid of images
    # # Get a batch of data
    # batch = next(iter(test_loader))
    # img_tensor = batch["img"]  # shape: [N, C, H, W]

    # # Select [4,2,0] channels for RGB visualization for all images in the batch
    # rgb_imgs = img_tensor[:, [4, 2, 0], :, :]

    # # Normalize from [-1, 1] to [0, 1]
    # rgb_imgs = (rgb_imgs + 1) / 2

    # # Create image grid using make_grid
    # grid = torchvision.utils.make_grid(rgb_imgs, nrow=8, padding=2, normalize=False)

    # # Convert to numpy and adjust dimension order for plotting
    # grid_img = grid.permute(1, 2, 0).numpy()

    # # Display and save the grid
    # plt.figure(figsize=(15, 15))
    # plt.imshow(grid_img)
    # plt.axis("off")
    # plt.savefig("multispectral_grid.png", bbox_inches="tight", pad_inches=0, dpi=300)
    # plt.close()

    # print(f"Batch shape: {img_tensor.shape}")
    # print(f"Range: min={img_tensor.min():.2f}, max={img_tensor.max():.2f}")
    # print(f"Data type: {img_tensor.dtype}")

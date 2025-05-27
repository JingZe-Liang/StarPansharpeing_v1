import os
import random
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import accelerate
import braceexpand
import numpy as np
import scipy.io
import tifffile
import torch
import torch.distributed
import webdataset as wds
from kornia.augmentation import (
    AugmentationSequential,
    RandomBoxBlur,
    RandomChannelShuffle,
    RandomCutMixV2,
    RandomHorizontalFlip,
    RandomRotation,
    RandomSharpness,
    RandomVerticalFlip,
)
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from natsort import natsorted
from safetensors import safe_open
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from typing_extensions import deprecated

from src.data.codecs import safetensors_decode_io, tiff_decode_io
from src.data.utils import (
    extract_modality_names,
    flatten_sub_dict,
    may_repeat_channels,
    merge_modalities,
    norm_img,
    remove_extension,
    remove_meta_data,
)
from src.utilities.config_utils import kwargs_to_basic_types
from src.utilities.logging import log_print

__compile_collate_fn = False
if __compile_collate_fn:
    torch._dynamo.config.recompile_limit = 20

type WebdatasetPath = str | list[str] | list[list[str]]


@deprecated(
    "get_dict_tensor_mapper is deprecated",
    category=UserWarning,
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
    def __init__(self, p=0.5):
        super().__init__(p)

    def apply_transform(self, input, params, flags, transform=None):
        assert input.ndim == 4
        c = input.shape[1]
        gray = input.mean(dim=1, keepdim=True).repeat_interleave(c, dim=1)
        return gray


def hyper_transform(
    op_list: tuple[str],
    probs: tuple[float, float] | float = 0.5,
    random_apply: int | tuple[int] = 2,
):
    if isinstance(probs, float):
        probs = [probs] * len(op_list)
    assert len(probs) == len(op_list)

    _op_list_cls = dict(
        grayscale=lambda p: HyperRandomGrayScale(p=p),
        channel_shuffle=lambda p: RandomChannelShuffle(p=p),
        sharpness=lambda p: RandomSharpness(p=p, sharpness=[0.5, 1.0]),
        rotation=lambda p: RandomRotation((-30, 30), p=p),
        horizontal_flip=lambda p: RandomHorizontalFlip(p=p),
        vertical_flip=lambda p: RandomVerticalFlip(p=p),
        cutmix=lambda p: RandomCutMixV2(num_mix=1, p=p, cut_size=(0.4, 0.6)),
        blur=lambda p: RandomBoxBlur((3, 3), p=p),
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

    def dict_mapper(sample):
        sample = op_seq(sample)
        return sample

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
    keys = sample[0].keys()
    _stack = False
    assert len(sample) > 0, "sample is empty"

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


def get_hyperspectral_dataloaders(
    wds_paths: str | list[str],
    batch_size: int,
    num_workers: int,
    shuffle_size: int = 100,
    to_neg_1_1: bool = True,
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
    resample: bool = True,
    prefetch_factor: int | None = 6,
    pin_memory: bool = False,
    shuffle_within_workers: bool = True,
    check_channels: bool = False,
    channels: int | None = None,
    remove_meta_data: bool = False,
    repeat_n: int = 1,
    timeout: int = 300,
    epoch_len: int = 0,
    persistent_workers: bool = False,
) -> tuple[wds.WebDataset, wds.WebLoader]:
    """Get dataloaders for hyperspectral image data

    Args:
        wds_paths (str | list[str]): Path or list of paths to WebDataset tar files
        batch_size (int): Number of samples per batch
        num_workers (int): Number of worker processes for data loading
        shuffle_size (int, optional): Buffer size for shuffling data. Defaults to 100.
        to_neg_1_1 (bool, optional): Whether to normalize data to [-1,1] range. Defaults to True.
        hyper_transforms_lst (tuple[str] | None, optional): List of data augmentation operations. Defaults to ("grayscale","channel_shuffle","rotation","cutmix","horizontal_flip","vertical_flip").
        transform_prob (tuple[float] | float, optional): Probability for each augmentation operation. Defaults to 0.2.
        random_apply (int | tuple[int], optional): Number of random augmentations to apply. Defaults to 1.
        resample (bool, optional): Whether to allow data resampling. Defaults to True.

    Returns:
        tuple[wds.WebDataset, wds.WebLoader]: Returns a tuple containing the WebDataset object and its corresponding dataloader
    """
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

    dataset = wds.WebDataset(
        wds_paths,
        resampled=resample,  # no need `iter(dataloader)` for `next` function
        shardshuffle=False,
        nodesplitter=wds.shardlists.single_node_only
        if not accelerate.state.PartialState().use_distributed
        else wds.shardlists.split_by_node,  # split_by_node if is multi-node training
        workersplitter=wds.split_by_worker,
        seed=2025,
        verbose=True,
        detshuffle=True if shuffle_size > 0 else False,
        empty_check=False,
    )
    dataset = dataset.decode(
        wds.handle_extension("tif tiff", tiff_decode_io),
        wds.handle_extension("safetensors", safetensors_decode_io),
    )
    dataset = dataset.map(remove_extension)
    if remove_meta_data:
        dataset = dataset.map(remove_meta_data)
    dataset = dataset.map(partial(norm_img, to_neg_1_1=to_neg_1_1))
    if check_channels:
        assert channels is not None, (
            "channels must be specified if check_channels is True"
        )
        dataset = dataset.map_dict(
            img=partial(may_repeat_channels, rep_channels=channels)
        )

    dataset = dataset.batched(batch_size, collation_fn=collate_fn_for_dict)
    if use_transf:
        dataset = dataset.map_dict(img=transform)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=pin_memory,
        drop_last=False,
        timeout=timeout if num_workers > 0 else 0,
        generator=torch.Generator().manual_seed(2025),
        shuffle=False,
    )

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
        f"[HyperDataset]: batch size: {batch_size}, num workers: {num_workers}, use transformations: {hyper_transforms_lst}"
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
    }[data_type]

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
        resample=False,
    )

    return dataloader


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

    def __init__(self, dir: str, return_key: bool = False):
        self.path = Path(dir)
        self.return_key = return_key
        self.img_paths = natsorted(list(self.path.glob("*.safetensors")))

    def __len__(self):
        return len(self.img_paths)

    @torch.no_grad()
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        key = img_path.stem

        img = load_file(img_path, device="cpu")["img"]
        # img = img.to(torch.float32).permute(2, 0, 1)

        # # to 0 to 1
        # img = img / img.max()
        # if self.to_neg_1_1:
        #     img = img * 2 - 1

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
            augmentations=transform,
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
        generator=torch.Generator().manual_seed(2025),
        pin_memory=pin_memory,
        multiprocessing_context="spawn"
        if (num_workers > 0 and __compile_collate_fn)
        else None,
        timeout=30 if num_workers > 0 else 0,
    )

    return dataset, dataloader


def chained_dataloaders(dataloaders, infinit=True, shuffle_loaders=True):
    """
    Chains multiple dataloaders into a single generator that yields samples from
    the provided dataloaders. Supports both finite and infinite iteration modes.

    Args:
        dataloaders (list): A list of dataloaders to chain together. Each dataloader
            should be an iterable that yields data samples.
        infinit (bool, optional): If True, the chained dataloader will reset and
            continue iterating indefinitely. If False, the iteration will stop
            once all dataloaders are exhausted. Defaults to True.
        shuffle_loaders (bool, optional): If True, the order of dataloaders will
            be shuffled when selecting which dataloader to sample from. Defaults to True.

    Yields:
        dict: A sample from one of the dataloaders, with an additional key
            "__loader_idx__" indicating the index of the dataloader from which
            the sample was drawn.

    Notes:
        - In infinite mode (`infinit=True`), the dataloaders will reset once
          exhausted, allowing continuous iteration.
        - In finite mode (`infinit=False`), the iteration stops when all dataloaders
          are exhausted.
        - If all dataloaders fail to yield data unexpectedly in infinite mode,
          a warning will be logged.
        - The function logs critical and debug messages when all dataloaders
          are exhausted or fail to yield data.

    Example:
        >>> dataloader1 = (
        ...     DataLoader(
        ...         dataset1
        ...     )
        ... )
        >>> dataloader2 = (
        ...     DataLoader(
        ...         dataset2
        ...     )
        ... )
        >>> chained_loader = chained_dataloaders(
        ...     [
        ...         dataloader1,
        ...         dataloader2,
        ...     ],
        ...     infinit=False,
        ... )
        >>> for sample in chained_loader:
        >>>     process(sample)
    """

    def _chain_dataloaders(dataloaders, infinit=True, shuffle_loaders=True):
        n = len(dataloaders)
        iters = [iter(dl) for dl in dataloaders]
        is_exhausted = [False] * n

        def get_active_idx():
            active_indices = [i for i in range(n) if not is_exhausted[i]]
            if shuffle_loaders:
                random.shuffle(active_indices)
            return active_indices

        active_indices = get_active_idx()

        while True:
            if all(is_exhausted):
                if infinit:
                    log_print(
                        "All loaders are exhausted, this should not happen in infinite mode",
                        "critical",
                    )
                log_print("All loaders exhausted, stop iteration", "debug")
                break

            # Reinitialize active indices if empty
            if not active_indices:
                active_indices = get_active_idx()
                if not active_indices:  # No active loaders left
                    break

            # Attempt to get next sample with retry logic
            max_attempts = len(active_indices) * 2  # Reasonable upper bound
            attempt = 0
            sample = None

            while attempt < max_attempts and active_indices:
                idx = active_indices.pop()
                attempt += 1

                try:
                    sample = next(iters[idx])
                    sample["__loader_idx__"] = idx
                    break  # Successfully got sample
                except StopIteration:
                    if infinit:
                        iters[idx] = iter(dataloaders[idx])  # Reset iterator
                    else:
                        is_exhausted[idx] = True

                    # Refresh active indices if needed
                    if not active_indices:
                        active_indices = get_active_idx()
                        if not active_indices:
                            break  # No more loaders to try

            if sample is not None:
                yield sample
            else:
                # All attempts failed
                if infinit:
                    log_print(
                        "All loaders failed to yield data unexpectedly", "warning"
                    )
                else:
                    break  # In finite mode, exit when we can't get any samples

            if not infinit and all(is_exhausted):
                break

    if not infinit:
        log_print(
            "Chained dataloader is finite and will stop once all data is exhausted. "
            "Set it to True if you want infinite iterations.",
            "warning",
        )

    log_print(f"Chaining {len(dataloaders)} dataloaders, {shuffle_loaders=}...")
    chained_loader = _chain_dataloaders(dataloaders, infinit, shuffle_loaders)

    return chained_loader


def get_hyperspectral_img_loaders_with_different_backends(
    paths: WebdatasetPath,
    loader_type: Literal["webdataset", "folder"] = "webdataset",
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

    # clear the kwargs
    (
        paths,
        loader_kwargs,
        rep_loader_kwargs,
        basic_kwargs,
        changed_kwargs_by_loader,
    ) = map(
        kwargs_to_basic_types,
        (
            paths,
            loader_kwargs,
            rep_loader_kwargs,
            basic_kwargs,
            changed_kwargs_by_loader,
        ),
    )  # type: ignore

    if loader_type == "webdataset":
        log_print("Using WebDataset loader")

        def ensure_non_empty_worker_shard(paths: str | list[str], loader_kwargs: dict):
            # Ensure that the number of workers is not greater than the number of shards
            if isinstance(paths, str):
                paths = list(braceexpand.braceexpand(paths))
                _len = len(paths)

            assert isinstance(paths, (list, tuple)), (
                f"paths should be a list or string, but got {type(paths)}"
            )

            _len = len(paths)
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
                    level="warning",
                )

            return loader_kwargs

        # Groups of dataloaders
        if isinstance(paths, list) and isinstance(paths[0], Sequence):
            log_print(
                "input paths contains list of lists, we will chain the dataloader with each loader"
            )
            if rep_loader_kwargs is not None:
                log_print("rep_loader_kwargs is provided", "debug")
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

            datasets = []
            dataloaders = []
            for p_lst, loader_kwargs in zip(paths, rep_loader_kwargs):
                # assertions
                assert isinstance(p_lst, (list, tuple)), (
                    f"paths should be a list of lists, but got {type(p_lst)}"
                )
                assert len(p_lst) > 0, f"paths should not be empty, but got {p_lst}"

                if len(p_lst) == 1:
                    p_lst = p_lst[0]
                    assert isinstance(p_lst, str), (
                        f"paths should be a list of strings, but got {type(p_lst)} for paths {p_lst}"
                    )
                else:
                    for p in p_lst:
                        assert isinstance(p, str), (
                            f"paths should be a list of strings, but got {type(p)} for paths {p_lst}"
                        )

                # resample must be false
                if not shuffle_loaders:
                    loader_kwargs["resample"] = False
                loader_kwargs["epoch_len"] = -1

                loader_kwargs = ensure_non_empty_worker_shard(p_lst, loader_kwargs)

                dataset, dataloader = get_hyperspectral_dataloaders(
                    p_lst, **loader_kwargs
                )
                datasets.append(dataset)
                dataloaders.append(dataloader)

            # make the chainable dataloader
            if not chain_loader_infinit:
                log_print(
                    "chain loader generator is finite, the loader can not be iter and next. "
                    "Please set <cyan>chain_loader_infinit=True</> if you are know what you are doing.",
                    "warning",
                )
            dataloader = chained_dataloaders(
                dataloaders, chain_loader_infinit, shuffle_loaders
            )
            return datasets, dataloader
        else:
            loader_kwargs = ensure_non_empty_worker_shard(paths, loader_kwargs)
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


def generate_wds_config_modify_only_some_kwgs(
    basic_cfg: dict, changed_kwargs: list[dict[str, Any]]
) -> list[dict]:
    """
    Generate a list of modified configurations based on a basic configuration and a list of changed keyword arguments.

    Args:
        basic_cfg (dict): The basic configuration dictionary.
        changed_kwargs (list[dict[str, Any]]): A list of dictionaries containing the changed keyword arguments.

    Returns:
        list[dict]: A list of modified configurations.
    """
    assert isinstance(basic_cfg, dict), (
        f"basic_cfg should be a dict, but got {type(basic_cfg)}"
    )
    assert isinstance(changed_kwargs, list), (
        f"changed_kwargs should be a list, but got {type(changed_kwargs)}"
    )

    cfgs = []
    for kwg in changed_kwargs:
        cfg = basic_cfg.copy()  # or deepcopy
        cfg.update(kwg)
        cfgs.append(cfg)

    return cfgs


if __name__ == "__main__":
    # Test config
    # test_wds_path = [
    #     "data/MMSeg_YREB/hyper_images/MMSeg_YREB_train_part-12_bands-MSI-0000.tar",
    #     "data/MMSeg_YREB/hyper_images/MMSeg_YREB_train_part-12_bands-MSI-0001.tar",
    #     "data/MMSeg_YREB/hyper_images/MMSeg_YREB_train_part-12_bands-MSI-0002.tar",
    #     "data/MMSeg_YREB/hyper_images/MMSeg_YREB_train_part-12_bands-MSI-0003.tar",
    # # ]
    test_wds_path = [
        # ["data/MMSeg_YREB/hyper_images/MMSeg_YREB_train_part-12_bands-MSI-0003.tar"],
        # ["data/DCF_2019/hyper_images/DCF_2019_Track_2-8_bands-px_512-MSI-0017.tar"],
        # ["data/DCF_2020/hyper_images/DFC_2020_public-13_bands-px_256-MSI-0002.tar"],
        # ["data/Houston/hyper_images/Houston-50_bands-px_512-MSI-0000.tar"],
        # ["data/GID-GF2/hyper_images/GID-GF2-test-3_bands-px_512-MSI-0001.tar"],
        # ["data/WorldView3/hyper_images/WorldView3-8_bands-px_256-MSI-0000.tar"],
        # ["data/DryadHyper/hyper_images/DryadHyper-224_bands-px_128-MSI-0002.tar"],
        # ["data/OHS/hyper_images/OHS-32_bands-px_512-MSI-0015.tar"],
        ["data/WorldView3/hyper_images/WorldView3-8_bands-px_256-MSI-0000.tar"],
        ["data/WorldView3/hyper_images/WorldView3-PAN-1_bands-px_512-MSI-0000.tar"],
    ]
    test_batch_size = 16
    test_num_workers = 3
    test_shuffle_size = -1

    loader_kwargs = dict(
        batch_size=test_batch_size,
        num_workers=test_num_workers,
        shuffle_size=test_shuffle_size,
        to_neg_1_1=True,
        transform_prob=0.0,
        random_apply=(1, 2),
        pin_memory=False,
        prefetch_factor=1,
        shuffle_within_workers=False,
        remove_meta_data=False,
        resample=True,
        check_channels=False,
    )
    changed_kwargs = [
        {
            "batch_size": 8,
        },
        {
            "batch_size": 4,
            "channels": 8,
            "check_channels": True,
        },
    ]  # * len(test_wds_path)
    channels = [12, 8, 13, 50, 4, 8, 224]
    accelerator = accelerate.Accelerator()

    _, test_loader = get_hyperspectral_img_loaders_with_different_backends(
        test_wds_path,
        loader_type="webdataset",
        basic_kwargs=loader_kwargs,
        changed_kwargs_by_loader=changed_kwargs,
        chain_loader_infinit=True,
    )

    # # for sample in iter(test_dataset):
    # #     print(sample.keys())

    from tqdm import tqdm

    total_samples_256 = 0
    total_samples_512 = 0

    for sample in (tbar := tqdm(test_loader)):
        img = sample["img"]

        tbar.set_description_str(
            "rank: {}, img shape: {}, loader idx: {}".format(
                accelerator.process_index, img.shape, sample["__loader_idx__"]
            )
        )
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

    # * get the dir safetensors dataloader
    # path = "data/MUSLI_safetensors/safetensors"
    # dataset, dataloader = only_hyperspectral_img_folder_dataloader(
    #     path,
    #     batch_size=8,
    #     num_workers=0,
    #     to_neg_1_1=True,
    #     transform_prob=0.0,
    #     # pin_memory=False,
    #     hyper_transforms_lst=None,
    # )

    # from tqdm import tqdm

    # for sample in tqdm(dataloader):
    #     print(sample.keys())
    #     img = sample["img"].cuda()
    #     img = img.permute(0, -1, 1, 2).float()
    #     img = img / img.max()
    #     img = img * 2 - 1

    #     print(img.shape)

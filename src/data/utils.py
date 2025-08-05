import json
import os
import random
import re
import types
from itertools import chain, product
from pathlib import Path
from typing import Any, Callable, Iterable, TypeAlias, cast

import braceexpand
import numpy as np
import pandas as pd
import scipy.special
import torch
import webdataset as wds
from kornia.augmentation import (
    RandomResizedCrop,
    Resize,
)

from src.utilities.train_utils.state import StepsCounter

from ..utilities.logging.print import log_print

LoadedData: TypeAlias = torch.Tensor | np.ndarray | str
SampleType: TypeAlias = dict[str, dict[str, LoadedData] | LoadedData] | tuple


def flatten_nested_list(lst):
    for item in lst:
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            yield from flatten_nested_list(item)
        else:
            yield item


def key_list_to_dict(keys: list[str] | set[str], use_ordered_dict=True):
    """
    find complexity O(1) time, but dict is C-implemented, it's faster than ordered dict.
    """
    if use_ordered_dict:
        from collections import OrderedDict

        return OrderedDict((k, None) for k in keys)
    else:
        return {k: None for k in keys}


def not_dunder_keys(sample: dict) -> list[str]:
    return [k for k in sample.keys() if not k.startswith("__")]


def extract_modality_names(s, square_bracket=False):
    # Regular expression pattern to match anything enclosed in '{' and '}', and comma separated
    pattern = r"\{([^}]*)\}" if not square_bracket else r"\[([^\]]*)\]"
    match = re.search(pattern, s)
    return match.group(1).split(",") if match else []


def is_path_multimodal(path: str) -> bool:
    """
    Check for either [] or {}, with no slashes inside.
    """
    return bool(re.search(r"(?:\[[^/]*\]|\{[^/]*\})", path))


def remove_extension(sample: dict[str, LoadedData]) -> dict[str, LoadedData]:
    # 'img.tiff' -> 'img'
    # '.img' -> 'img'
    # 'img' -> 'img'
    sample_dict = {}
    for k, v in sample.items():
        k_wo_ext = k
        _has_ext = k.rfind(".", 1) != -1
        if _has_ext:
            k_wo_ext = k[: k.find(".")]
        sample_dict[k_wo_ext] = v

    return sample_dict


def remove_meta_data(sample: dict[str, LoadedData]) -> dict[str, LoadedData]:
    for k in sample.keys():
        if k.startswith("__"):
            del sample[k]

    return sample


def check_img_channel(
    sample: dict[str, torch.Tensor],
    expect_channels: int = 3,
    img_key: str | list[str] = "img",
):
    # img: (1, H, W), (H, W), (C, H, W)

    def inner(sample, k):
        img = sample[k]
        if img.ndim == 2:
            img = img[None].repeat(expect_channels, 1, 1)
        elif img.ndim == 3:
            if img.shape[0] == 1:
                img = img.repeat(expect_channels, 1, 1)
            elif img.shape[0] > expect_channels:
                img = img[:expect_channels]
            elif img.shape[0] == expect_channels:
                pass  # <- expected channels, do nothing
            else:
                log_print(
                    f"sample image key {k} got shape {tuple(img.shape)}, "
                    f"expect channels {expect_channels}, skip this sample.",
                    "warning",
                )
                img = None
        else:
            raise ValueError(
                f"Unsupported number of dimensions for img: {img.ndim}. Expected 2 or 3."
            )

        return img

    img_key = [img_key] if isinstance(img_key, str) else img_key
    for k in img_key:
        img = inner(sample, k)
        if img is None:
            return None
        sample[k] = img
    return sample


def _check_dots(s: str):
    if ".gz" in s:
        return s.count(".") == 2
    return s.count(".") == 1


def de_structure_tar(sample: dict[str, LoadedData]) -> dict[str, LoadedData]:
    result = {}
    for k, val in sample.items():
        if "/" in k:
            file_name = k.split("/")[-1]  # Get the last part of the path
            result[file_name] = val
        else:
            result[k] = val
    return result


def remove_keys(keys_to_remove: str | list[str]):
    if isinstance(keys_to_remove, str):
        keys_to_remove = [keys_to_remove]

    p: list[re.Pattern] = []
    for k in keys_to_remove:
        p.append(re.compile(k))

    def inner(sample: dict):
        keys = list(sample.keys())
        for k in keys:
            for p_re in p:
                if p_re.match(k):
                    del sample[k]
                    break  # del the key and then jump to the next re pattern

        return sample

    return inner


def search_one_key_not_dunder(
    sample: dict[str, LoadedData],
    out_key: str | types.SimpleNamespace,
    random_one=False,
):
    _not_dunder_keys: list[str] = []
    # most dataset has 'img' key and 'img_content' for RS5M
    _default_img_search_keys = ("img", "img_content", "rgb")
    new_sample = {}

    for k in sample.keys():
        if not k.startswith("__"):
            _not_dunder_keys.append(k)
        else:
            new_sample[k] = sample[k]

    if not random_one:
        # if the key is in the default keys
        _searched_flag = False
        for dk in _default_img_search_keys:
            if dk in _not_dunder_keys:
                _searched_flag = True
                ch_key = dk

        # else not in default keys, then assert only one key not starting with "__"
        if not _searched_flag:
            assert len(_not_dunder_keys) == 1, (
                'Expected exactly one key not starting with "__", but found: '
                + str(_not_dunder_keys)
            )
            ch_key = _not_dunder_keys[0]
    else:
        if len(_not_dunder_keys) < 1:
            raise ValueError(
                'Expected at least one key not starting with "__" when random choosing one key, '
                "but found: " + str(_not_dunder_keys)
            )
        ch_key = random.choice(_not_dunder_keys)

    new_sample[out_key] = sample[ch_key]

    return new_sample


def filter_undecoded(sample, key: str | list[str] | None):
    if key is None:
        # check all keys
        for k, v in sample.items():
            if not k.startswith("__") and isinstance(v, bytes):
                log_print(f"{k} is undecoded", "warning")
                return None
    else:
        if isinstance(key, str):
            key = [key]

        for k in key:
            if k not in sample:
                log_print(f"{k} not in sample", "error")
                raise ValueError(
                    f"{k} not in sample, sample keys: {sample.keys()} at url {sample['__url__']} with name {sample['__key__']}"
                )
            if isinstance(sample[k], bytes):
                log_print(f"{k} is undecoded", "warning")
                return None

        return sample


def rename_keys(
    sample: dict,
    tgt_keys: str | list[str],
    source_keys: str | list[str],
    remove_others: bool = False,
):
    if isinstance(tgt_keys, str):
        tgt_keys = [tgt_keys]
    if isinstance(source_keys, str):
        source_keys = [source_keys]

    assert len(tgt_keys) == len(source_keys), (
        "tgt_keys and source_keys must have the same length"
    )

    for sk, tk in zip(source_keys, tgt_keys):
        if sk in sample and tk is not None:
            sample[tk] = sample.pop(sk, None)
            if sample[tk] is None:
                log_print(
                    '"Key {sk} not found in sample, setting {tk} to None"', "warning"
                )

    if remove_others:
        for k in sample.keys():
            if k not in source_keys and not k.startswith("__"):
                del sample[k]

    return sample


def remove_dot_and_extensions(sample, tgt_key: str | dict[str, str] | None = None):
    _keys = list(sample.keys())
    for k in _keys:
        if not k.startswith("__"):
            v = sample.pop(k)
            if tgt_key is None:
                tgt_key = Path(k).stem.rsplit(".", 1)[-1]
            elif isinstance(tgt_key, dict):
                tgt_key = tgt_key[k]  # may raise KeyError
            elif isinstance(tgt_key, str):
                sample[tgt_key] = v
                break
            else:
                raise ValueError(
                    f"Unsupported type for tgt_key: {type(tgt_key)}. Expected str or dict."
                )

            sample[tgt_key] = v

    return sample


# @torch.compile
def norm_img(
    sample: SampleType,
    norm_keys: str | list[str] | None = ["img"],
    to_neg_1_1: bool = True,
    permute: bool = True,
    check_nan: bool = False,
    on_device: bool = False,
    clip_zero: bool = True,
):
    """
    Normalize the image to [0, 1] and optionally to [-1, 1].
    Args:
        to_neg_1_1 (bool): Whether to normalize to [-1, 1].
        norm_keys (list[str]): List of keys to normalize.
        permute (bool): Whether to permute the image dimensions.

    Returns:
        function: A function that takes a sample and normalizes the image.
    """

    if norm_keys is None:
        norm_keys = not_dunder_keys(sample)
    elif isinstance(norm_keys, str):
        norm_keys = [norm_keys]

    for key in norm_keys:
        if key not in sample:
            log_print(f"{key} not in sample", level="warning", warn_once=True)
            continue

        if isinstance(sample[key], dict):
            _sample_dict: dict = sample[key]  # type: ignore
            _img = _sample_dict.get("img")
        elif isinstance(sample[key], (np.ndarray, torch.Tensor)):
            _img = sample[key]
        else:
            raise ValueError(
                f"Unsupported type for {key}: {type(sample[key])}. Expected dict, np.ndarray, or torch.Tensor."
            )

        img = torch.as_tensor(_img, dtype=torch.float32)
        if on_device:
            img = img.to(torch.device("cuda"), non_blocking=True)
        if check_nan:
            img = torch.nan_to_num(img, nan=0.0, posinf=1, neginf=0.0)
        if clip_zero:
            img = img.clip(min=0.0, max=None)  # clip to [0, inf)
        _img_max = img.max()
        if _img_max < 1e-4:
            img = torch.zeros_like(img) if not to_neg_1_1 else torch.ones_like(img) / 2
        else:
            img = img / (_img_max + 1e-6)
        if to_neg_1_1:
            img = (img * 2 - 1).clip(-1.0, 1.0)
        if permute:
            if img.ndim == 3:
                # (H, W, C) -> (C, H, W)
                img = img.permute(-1, 0, 1)
            elif img.ndim == 4:
                #  # (B, C, H, W) -> (B, C, H, W)
                img = img.permute(0, -1, 1, 2)
            else:
                log_print(
                    f"found img dim with {img.ndim} dimensions, expected 3 or 4",
                    level="warning",
                    warn_once=True,
                )
                return None  # None for webdataset means drop this sample

        sample[key] = img

    return sample


def size_filtering(
    sample, tgt_key: str | list[str] | None, constraint_size: int | tuple
):
    if tgt_key is None:
        tgt_key = [k for k in sample.keys() if not k.startswith("__")]
    elif isinstance(tgt_key, str):
        tgt_key = [tgt_key]

    for k in tgt_key:
        img = sample[k]
        img_size = np.prod(img.shape[:2]).item()
        if isinstance(constraint_size, int):
            if img_size < constraint_size:  # larger than, keep
                return None
        elif isinstance(constraint_size, tuple):
            if not (
                constraint_size[0] <= img_size <= constraint_size[1]
            ):  # in the range, keep
                return None

    return sample


def loop_apply_filters(
    filtering: Callable,
    loop_kwargs: dict[str, list | tuple],
    const_kwargs: dict[str, Any] | None = None,
):
    if const_kwargs is None:
        const_kwargs = {}

    for v in loop_kwargs.values():
        if not isinstance(v, (list, tuple)):
            raise ValueError(
                f"Expected list or tuple for loop_kwargs values, but got {type(v)}"
            )

    # compose looped kwargs
    keys = loop_kwargs.keys()
    composited_kwargs = [dict(zip(keys, v)) for v in product(*loop_kwargs.values())]

    def inner(sample):
        for kwargs in composited_kwargs:
            # merge constant kwargs
            _kwargs = {**const_kwargs, **kwargs}
            # apply filtering
            res = filtering(sample, **_kwargs)

            if res is not None:
                return res
            else:
                log_print(
                    "Filtering failed with kwargs: {kwargs} at filter: {filtering.__name__}",
                    "warning",
                )
                continue
        return None

    return inner


# * --- Webdataset reading utilities --- #


def merge_modalities(
    source: Iterable[dict],
    modality_name_map: dict | None = None,
    handler=wds.warn_and_stop,
):
    for src in source:
        multi_tar_urls = src["url"].translate(str.maketrans("[]", "{}"))
        modality_names = extract_modality_names(multi_tar_urls)

        if len(modality_names) == 0:
            # Case where multi-modal braceexpand is not used, e.g. shard_dir/shard00000.tar
            modality_names = [None]
            multi_tar_urls = [multi_tar_urls]
        elif len(modality_names) == 1:
            # Brace expand doesn't work with a single entry, e.g. shard_dir/[foo]/shard00000.tar
            multi_tar_urls = [multi_tar_urls.replace("{", "", 1).replace("}", "", 1)]
            multi_tar_urls = [
                list(braceexpand.braceexpand(tar_url)) for tar_url in multi_tar_urls
            ]
            multi_tar_urls = sum(multi_tar_urls, [])  # Flatten the list
        else:
            # Remaining cases where multiple modalities are specified, e.g. shard_dir/[foo,bar]/shard00000.tar
            multi_tar_urls = list(braceexpand.braceexpand(multi_tar_urls))

        if len(multi_tar_urls) != len(modality_names):
            # shard_dir/[foo,bar]/shard{0000..0002}.tar, list of tars
            assert len(multi_tar_urls) % len(modality_names) == 0, (
                "Number of shards is not divisible by number of modalities"
                f"but got {len(multi_tar_urls)} shards and {len(modality_names)} modalities"
                f"multi_tar_files: {multi_tar_urls}, modality_names: {modality_names}"
            )
            tar_files = [
                wds.tarfile_samples([{"url": tar_url}]) for tar_url in multi_tar_urls
            ]
            chained_per_modality = [
                chain.from_iterable(
                    tar_files[i : i + len(multi_tar_urls) // len(modality_names)]
                )
                for i in range(
                    0, len(multi_tar_urls), len(multi_tar_urls) // len(modality_names)
                )
            ]
            multi_tar_loop = zip(*chained_per_modality)
        else:
            # Create tar iterators for shards of all modalities
            tar_iters = [
                wds.tarfile_samples([{"url": tar_url}]) for tar_url in multi_tar_urls
            ]
            multi_tar_loop = zip(*tar_iters)

        # * --- Try to loop the multi-modality tar sources --- #

        try:
            # Loop over these iterators in parallel and combine the tar files from different modalities
            for multi_tar_files in multi_tar_loop:
                merged_dict = {}
                merged_dict["__key__"] = multi_tar_files[0]["__key__"]
                merged_dict["__url__"] = src["url"]

                for modality_name, modality_dict in zip(
                    modality_names, multi_tar_files
                ):
                    _key = modality_dict.pop("__key__")
                    _url = modality_dict.pop("__url__")

                    if _key != merged_dict["__key__"]:
                        raise ValueError(
                            f"Divergence detected! Trying to merge keys {_key} of {modality_name} "
                            f"and {merged_dict['__key__']} of merged_dict with modalities {merged_dict.keys()}."
                        )

                    tar_is_multimodal = len(modality_dict) > 1
                    for k, v in modality_dict.items():
                        if tar_is_multimodal or _check_dots(k) or modality_name is None:
                            # We don't change the keys in the following cases:
                            # 1. The shard contains multiple modalities. Then they *have* to follow the idx.modality_id.ext convention
                            # 2. If any key contains a dot, this means it already has the idx.modality_id.ext format (idx. is already removed at this stage)
                            # 3. If the modality name is None, no modality folder was specified (see beginning of function)
                            merged_dict[k] = v
                        else:
                            mapped_name = (
                                modality_name
                                if modality_name_map is None
                                else modality_name_map.get(modality_name, modality_name)
                            )
                            merged_dict[f"{mapped_name}.{k}"] = v

                yield merged_dict

        except Exception as e:
            print(e)
            print(f"Exception occurred while processing {src['url']}.")
            if handler(e):
                print("Skipping shard...")
                continue
            else:
                break


# * --- filters --- #


def wids_img_size_filter_by_parquet_index(
    parquet_file: str,
    min_size: int = 256,
    max_size: int | None = None,
) -> set[int]:
    """Filter function to check if the image size is within the specified range."""
    shapes = pd.read_parquet(parquet_file)  # TODO: page reading?
    shapes = shapes[shapes["height"] >= min_size and shapes["width"] >= min_size]
    if max_size is not None:
        shapes = shapes[(shapes["height"] <= max_size) & (shapes["width"] <= max_size)]
    indexes = shapes["index"].tolist()
    log_print(
        "[img_size_filter] Found {len(indexes)} samples that fit the size constraint."
    )
    return set(indexes)


def flatten_sub_dict(regardless_of_any_collisions=True):
    _has_checked = False

    def key_is_in(key, sample):
        nonlocal _has_checked
        if _has_checked:
            return

        if key in sample:
            raise ValueError(f"Key {key} already exists in {sample}.")

    def _inner(sample):
        sample_flatten = {}
        for k, v in sample.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if regardless_of_any_collisions:
                        key_is_in(k, sample_flatten)
                        sample_flatten[kk] = vv
                    else:
                        sample_flatten[f"{k}_{kk}"] = vv
            else:
                key_is_in(k, sample_flatten)
                sample_flatten[k] = v

        nonlocal _has_checked
        _has_checked = True

        return sample_flatten

    return _inner


def to_n_tuple(val: int | float | tuple, n: int) -> tuple:
    if isinstance(val, tuple):
        assert len(val) == n, f"Expected tuple of length {n}, got {len(val)}"
        return val

    assert isinstance(val, (int, float)), f"Expected int or float, got {type(val)}"

    return tuple([val] * n)


def wids_filter_img_size(constraint_size: int | tuple):
    def size_filter_fn(sample):
        img = sample["img"]
        img_size = np.prod(img.shape[-2:]).item()
        if isinstance(constraint_size, int):
            if img_size >= constraint_size:
                return sample
        elif isinstance(constraint_size, tuple):
            if img_size >= constraint_size[0] and img_size <= constraint_size[1]:
                return sample
        # return a null dict
        log_print("is None", "debug")
        return None

    return size_filter_fn


def large_image_resizer_clipper(
    tgt_size: int | tuple[int, int],
    img_key: str | list[str] | None = "img",
    op_for_large: str = "clip",
):
    """
    Create a function to process large images by either random cropping or resizing.

    Args:
        tgt_size (int | tuple[int, int]): Target image size. If int, will be converted to (h, w).
        op_for_large (str): Operation for large images. "clip" for random crop, "resize" for resizing.

    Returns:
        dict_mapper (function): A function that takes an image tensor (torch.Tensor) as input and returns the processed image tensor.

    Example:
        process = large_image_resizer_clipper(224, op_for_large="resize")
        result = process(img_tensor)
    """
    if isinstance(tgt_size, int):
        tgt_size = to_n_tuple(tgt_size, 2)
    if isinstance(img_key, str):
        img_key = [img_key]

    if op_for_large == "clip":
        clipper = RandomResizedCrop(
            tgt_size,
            scale=(0.6, 1.0),
            ratio=(0.75, 1.333),
            p=1.0,
            keepdim=True,
            cropping_mode="resample",
        )
    elif op_for_large == "resize":
        clipper = Resize(tgt_size, antialias=True, p=1, keepdim=True)
    else:
        raise ValueError(f"op_for_large {op_for_large} not supported")

    def sample_mapper(sample: dict, img_key=img_key):
        if img_key is None:
            img_key = not_dunder_keys(sample)

        try:
            _keys = sample.keys()
            for k in _keys:
                if k in img_key:
                    sample[k] = clipper(sample[k])
        except Exception as e:
            log_print(f"worker pid <u>{os.getpid()}</u> clip img error: {e}", "warning")
            return None
        return sample

    # def dict_mapper(img: torch.Tensor):
    #     return clipper(img)

    return sample_mapper


def wids_filter_out_none_map_dict_iterator(self: wds.pipeline.DataPipeline):
    """Create an iterator through the entire dataset, using the given number of repetitions.

    Yields:
        Samples from the dataset.
    """
    for _ in range(self.repetitions):
        count = 0
        for sample in self.iterator1():
            if sample is None:
                continue
            yield sample
            count += 1
        if count == 0:
            # if the dataset is empty, don't keep looping
            break


# * --- Chainable loaders --- #


def chained_dataloaders(
    dataloaders,
    infinit=True,
    shuffle_loaders=True,
    curriculum_fn: Callable[[int], list[float]] | None = None,
    other_sample_fn: list[Callable[[SampleType], SampleType]] | None = None,
):
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
        curriculum_fn (Callable[[int], list[float]], optional): A function that
            takes the current training step as input and returns a list of probabilities
            for each dataloader. This is used for curriculum learning to select dataloaders
            based on their probabilities. If None, a standard round-robin selection is used.
        other_sample_fn (list[Callable[[SampleType], SampleType]], optional): A list of
            functions that take a sample and return a modified sample. These functions
            are applied to each sample yielded by the chained dataloader.

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

    if curriculum_fn is not None:
        shuffle_loaders = False  # curriculum learning requires fixed order
        assert infinit, "using curriculum learning requires infinit=True"
        # assert StepsCounter._initialized, "StepsCounter has not initialized"

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
                # * --- CURRICULUM LEARNING SELECTION STRATEGY --- #
                if curriculum_fn:
                    train_step: int = StepsCounter()["train"]  # sigleton class instance
                    # log step sometimes
                    _log_cur = train_step % 200 == 0
                    if _log_cur:
                        log_print(f"[Curriculum]: train step {train_step}", "debug")

                    # Get probability distribution for current step
                    raw_probs = curriculum_fn(train_step)
                    # log_print(f"probs {raw_probs}", "debug")

                    # Filter only active loaders and their probabilities
                    valid_indices = []
                    valid_probs = []
                    for idx in active_indices:
                        # if not is_exhausted[idx]:
                        valid_indices.append(idx)
                        valid_probs.append(raw_probs[idx])

                    # Normalize probabilities for active loaders
                    total_prob = sum(valid_probs)
                    if total_prob > 0:
                        # softmax the valid probabilities
                        normalized_probs = scipy.special.softmax(
                            valid_probs, axis=0
                        ).tolist()
                        # normalized_probs = [p / total_prob for p in valid_probs]
                        # Select loader based on curriculum probabilities
                        idx = random.choices(
                            valid_indices, weights=normalized_probs, k=1
                        )[0]
                        if _log_cur:
                            log_print(
                                f"[Curriculum]: selected loader {idx} with prob: {normalized_probs}",
                                only_rank_zero=False,
                            )
                    else:
                        log_print(
                            "All curriculum probabilities are less than 0", "warning"
                        )

                        # Fallback to uniform selection if all probs zero
                        idx = random.choice(valid_indices)

                # * --- STANDARD SELECTION STRATEGY --- #
                else:
                    if shuffle_loaders:
                        random.shuffle(active_indices)
                    idx = active_indices[0]  # Select first active index
                    # round-robin strategy
                    active_indices.pop(0)
                    active_indices.append(idx)

                attempt += 1

                try:
                    sample = next(iters[idx])
                    if isinstance(sample, dict):
                        sample["__loader_idx__"] = idx
                    break  # Successfully got sample
                except StopIteration:
                    if infinit:
                        iters[idx] = iter(dataloaders[idx])  # Reset iterator
                    else:
                        is_exhausted[idx] = True

                    # Remove the index from active_indices only if it failed
                    if (
                        idx in active_indices
                    ):  # Check if it's still there (it should be)
                        active_indices.remove(idx)

                    # Refresh active indices if needed
                    if not active_indices:
                        active_indices = get_active_idx()
                        if not active_indices:
                            break  # No more loaders to try

            if sample is not None:
                # additional sample functions
                if other_sample_fn is not None:
                    for fn in other_sample_fn:
                        sample = fn(sample)

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


def get_wids_index_json_info(path: str) -> tuple[list[dict], int]:
    assert path.endswith(".json"), (
        f"wids json file should end with .json, but got {path}"
    )
    assert os.path.exists(path), (
        f"wids json file {path} does not exist, please check the path"
    )

    with open(path, "r") as f:
        index_d = json.load(f)
    assert "shardlist" in index_d, (
        f"wids json file {path} should contain 'shardlist' key, but got {index_d.keys()}"
    )

    index_info = index_d["shardlist"]
    n = len(index_info)

    return index_info, n


def expand_paths_and_correct_loader_kwargs(
    load_type: str, paths: str | list, loader_kwargs: dict
):
    # Ensure that the number of workers is not greater than the number of shards
    if load_type == "wids":
        if isinstance(paths, str):
            paths = [paths]

        if isinstance(paths, (tuple, list)):
            _len = 0
            for p in paths:
                assert isinstance(p, str) and p.endswith(".json"), (
                    "'paths' should be a list of strings that are wids json files, "
                    f"but got {type(p)} with value {p}"
                )
                _, len_i = get_wids_index_json_info(p)
                _len += len_i
            path_exp = paths
        else:
            raise ValueError(
                f"'paths' should be a string or list of stirng that is json files but got {paths}"
            )

    else:  # webdataset
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
                assert not p.endswith(".json"), (
                    "list of paths contains multiple wids json files or "
                    "combination of tar file path and wids json file path, "
                    "please use a list of lists of strings instead. "
                    "If you want to use wids json dataset, "
                    "please use a single json file as index_file."
                )
                lst_p = list(braceexpand.braceexpand(p))
                path_exp.extend(lst_p)
        else:
            raise ValueError(
                f"paths should be a string or a list of strings, but got {type(paths)}"
            )
        _len = len(path_exp)

    assert _len > 0, f"paths should not be empty, but got {paths}"

    for p in path_exp:
        # assert tar file exists
        assert os.path.exists(p), f"tar file or wids json file {p} does not exist"

    # n_worker should less than the number of shards
    n_workers = loader_kwargs.get("num_workers", 0)
    if _len < n_workers and n_workers > 0:
        # set n_workers to the number of shards
        loader_kwargs["num_workers"] = _len
        log_print(
            f"n_workers={n_workers} is larger than the number of shards {_len}, set n_workers={_len}",
            level="debug",
        )

    return path_exp, loader_kwargs


# multimodal version of expand_paths_and_correct_loader_kwargs
def expand_paths_and_correct_loader_kwargs_mm(
    paths: str | list[str] | dict[str, str], loader_kwargs: dict
):
    # Ensure that the number of workers is not greater than the number of shards
    if isinstance(paths, str):
        # Ensure is multimodal
        n_modalities = len(extract_modality_names(paths, square_bracket=True))
        assert n_modalities > 1, (
            f"n_modalities must be greater than 1, when the path is {paths}"
        )

        mm_paths = paths.translate(str.maketrans("[]", "{}"))
        mm_paths = list(braceexpand.braceexpand(mm_paths))
        _len = len(mm_paths) // n_modalities
    elif isinstance(paths, (list, tuple)):
        mm_paths = []
        n_modalities = 1
        for p in paths:
            assert isinstance(p, str), (
                "'paths' should be a list of strings or a string that can be expanded"
                f"list of {type(p)} is not allowed."
            )
            n_modalities = len(extract_modality_names(p, square_bracket=True))
            assert n_modalities > 1, (
                f"n_modalities must be greater than 1, when the path is {p}"
            )
            mm_p = p.translate(str.maketrans("[]", "{}"))
            lst_p = list(braceexpand.braceexpand(mm_p))
            mm_paths.extend(lst_p)
        _len = len(mm_paths) // n_modalities
    elif isinstance(paths, dict):
        mm_paths = list(paths.values())
        n_modalities = len(mm_paths)
        # assert all files have equal number of shards
        wids_files: list[dict] = [json.load(open(p, "r")) for p in mm_paths]
        get_n_shards = lambda json_file: len(json_file["shardlist"])
        total_shards = set(get_n_shards(wf) for wf in wids_files)
        assert len(total_shards) == 1, "all files must have equal number of shards"
        _len = total_shards.pop()
    else:
        raise ValueError(
            f"paths should be a string or a list of strings, but got {type(paths)}"
        )

    assert _len > 0, f"paths should not be empty, but got {paths}"

    # assert tar files exist
    for p in mm_paths:
        # assert tar file exists
        assert os.path.exists(p), f"tar file or wids json file {p} does not exist"

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


# * --- Collate functions --- #


__default_wids_keys = [
    "__key__",
    "__index__",
    "__shard__",
    "__shardindex__",
]


# TODO: complete this collate function
def multimodal_wids_collate_fn(batch: list[dict]):
    """
    Custom collate function for multimodal WIDS dataset.
    This function handles the different modalities in the batch.
    """

    collated_batch = {
        "__key__": {},
        "__index__": {},
        "__shard__": {},
        "__shardindex__": {},
    }
    for d in batch:
        if not isinstance(d, dict):
            raise TypeError(f"Expected dict, got {type(d)}")
        for name, mm_dict in d.items():
            for key, value in mm_dict.items():
                if key in __default_wids_keys:
                    # collated_batch
                    ...

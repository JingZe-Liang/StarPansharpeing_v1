import json
import os
import random
import re
from itertools import chain
from typing import Any, Callable, Iterable, TypeAlias

import braceexpand
import numpy as np
import pandas as pd
import scipy.special
import torch
import webdataset as wds
from librosa import ex

from src.utilities.train_utils.state import StepsCounter

from ..utilities.logging.print import log_print

LoadedData: TypeAlias = torch.Tensor | np.ndarray | str
SampleType: TypeAlias = dict[str, dict[str, LoadedData] | LoadedData]


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
    sample_dict = {}
    for k, v in sample.items():
        k_wo_ext = k
        _has_ext = k.find(".") != -1
        if _has_ext:
            k_wo_ext = k[: k.find(".")]
        sample_dict[k_wo_ext] = v

    return sample_dict


def remove_meta_data(sample: dict[str, LoadedData]) -> dict[str, LoadedData]:
    for k in sample.keys():
        if k.startswith("__"):
            del sample[k]

    return sample


def may_repeat_channels(img: torch.Tensor, rep_channels: int = 3) -> torch.Tensor:
    # img: (1, H, W), (H, W), (C, H, W)
    if img.ndim == 2:
        img = img[None].repeat(rep_channels, 1, 1)
    elif img.ndim == 3:
        if img.shape[0] == 1:
            img = img.repeat(rep_channels, 1, 1)
        elif img.shape[0] != rep_channels:
            raise ValueError(
                f"Expected img to have {rep_channels} channels, but got {img.shape[0]} channels."
            )
    else:
        raise ValueError(
            f"Unsupported number of dimensions for img: {img.ndim}. Expected 2 or 3."
        )
    return img


def _check_dots(s: str):
    if ".gz" in s:
        return s.count(".") == 2
    return s.count(".") == 1


# @torch.compile
def norm_img(
    sample: SampleType,
    norm_keys: list[str] = ["img"],
    to_neg_1_1: bool = True,
    permute: bool = True,
    check_nan: bool = False,
    on_device: bool = False,
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

    for key in norm_keys:
        if key not in sample:
            log_print(f"{key} not in sample", level="warning", warn_once=True)
            continue

        if isinstance(sample[key], dict):
            _sample_dict: dict = sample[key]
            _img = _sample_dict.get("img")
        elif isinstance(sample[key], (np.ndarray, torch.Tensor)):
            _img = sample[key]
        else:
            raise ValueError(
                f"Unsupported type for {key}: {type(sample[key])}. Expected dict, np.ndarray, or torch.Tensor."
            )

        img = torch.as_tensor(
            _img,
            dtype=torch.float32,
        )
        if on_device:
            img = img.to(torch.device("cuda"), non_blocking=True)
        if check_nan:
            img = torch.nan_to_num(img, nan=0.0, posinf=1, neginf=0.0)
        _img_max = img.max()
        if _img_max < 1e-4:
            img = torch.zeros_like(img) if not to_neg_1_1 else torch.ones_like(img) / 2
        else:
            img = img / (_img_max + 1e-6)
        if to_neg_1_1:
            img = (img * 2 - 1).clip(-1.0, 1.0)
        if permute:
            if img.ndim == 3:
                img = img.permute(-1, 0, 1)
            elif img.ndim == 4:
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


def img_size_filter(
    parquet_file: str,
    min_size: int = 256,
    max_size: int | None = None,
) -> set[int]:
    """Filter function to check if the image size is within the specified range."""
    shapes = pd.read_parquet(parquet_file)
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


def to_n_tuple(val: int | float | tuple, n: int) -> tuple[int | float, ...]:
    if isinstance(val, tuple):
        assert len(val) == n, f"Expected tuple of length {n}, got {len(val)}"
        return val

    assert isinstance(val, (int, float)), f"Expected int or float, got {type(val)}"

    return tuple([val] * n)


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


def get_wids_index_json_info(path: str):
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


def expand_paths_and_correct_loader_kwargs(paths: str | list[str], loader_kwargs: dict):
    # Ensure that the number of workers is not greater than the number of shards
    if isinstance(paths, str):
        if paths.endswith(".json"):  # is indexed wids json file
            _, _len = get_wids_index_json_info(paths)
            paths = [paths]
        else:
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
        _len = len(path_exp)
        paths = path_exp
    else:
        raise ValueError(
            f"paths should be a string or a list of strings, but got {type(paths)}"
        )

    assert _len > 0, f"paths should not be empty, but got {paths}"

    for p in paths:
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


# multimodal version of expand_paths_and_correct_loader_kwargs
def expand_paths_and_correct_loader_kwargs_mm(
    paths: str | list[str], loader_kwargs: dict
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
    else:
        raise ValueError(
            f"paths should be a string or a list of strings, but got {type(paths)}"
        )

    assert _len > 0, f"paths should not be empty, but got {paths}"

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

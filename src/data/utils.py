import re
from itertools import chain
from typing import Dict

import braceexpand
import numpy as np
import torch
import webdataset as wds

from ..utilities.logging.print import log_print

type LoadedData = torch.Tensor | np.ndarray | str
type SampleType = dict[str, dict[str, LoadedData] | LoadedData]


def extract_modality_names(s):
    # Regular expression pattern to match anything enclosed in '{' and '}', and comma separated
    pattern = r"\{([^}]*)\}"
    match = re.search(pattern, s)
    return match.group(1).split(",") if match else []


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
        img = img / img.max()
        if to_neg_1_1:
            img = img * 2 - 1
        if permute:
            if img.ndim == 3:
                img = img.permute(-1, 0, 1)
            elif img.ndim == 4:
                img = img.permute(0, -1, 1, 2)
            else:
                raise ValueError(
                    f"Unsupported number of dimensions for {key}: {img.ndim}. Expected 3 or 4."
                )

        sample[key] = img

    return sample


def merge_modalities(
    source, modality_name_map: dict | None = None, handler=wds.warn_and_stop
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

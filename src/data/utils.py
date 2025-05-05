import re

import braceexpand
import torch
import webdataset as wds

from utilities.logging.print import log_print


def extract_modality_names(s):
    # Regular expression pattern to match anything enclosed in '{' and '}', and comma separated
    pattern = r"\{([^}]*)\}"
    match = re.search(pattern, s)
    return match.group(1).split(",") if match else []


def remove_extension(sample):
    sample_dict = {}
    for k, v in sample.items():
        k_wo_ext = k
        _has_ext = k.find(".") != -1
        if _has_ext:
            k_wo_ext = k[: k.find(".")]
        sample_dict[k_wo_ext] = v

    return sample_dict


def remove_meta_data(sample):
    for k, v in sample.items():
        if k.startswith("__"):
            del sample[k]

    return sample


def _check_dots(s):
    if ".gz" in s:
        return s.count(".") == 2
    return s.count(".") == 1


def norm_img(
    to_neg_1_1: bool = True, norm_keys: list[str] = ["img"], permute: bool = True
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

    def _inner(sample):
        for key in norm_keys:
            if key not in sample:
                log_print(f"{key} not in sample", level="warning", warn_once=True)
                continue

            img = torch.as_tensor(sample[key], dtype=torch.float32)
            img = img / img.max()
            if to_neg_1_1:
                img = img * 2 - 1
            if permute:
                img = img.permute(-1, 0, 1)

            sample[key] = img

        return sample

    return _inner


def merge_modalities(
    source, modality_name_map: dict | None = None, handler=wds.warn_and_continue
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
            multi_tar_urls = [multi_tar_urls.replace("{", "").replace("}", "")]
        else:
            # Remaining cases where multiple modalities are specified, e.g. shard_dir/[foo,bar]/shard00000.tar
            multi_tar_urls = list(braceexpand.braceexpand(multi_tar_urls))

        # Create tar iterators for shards of all modalities
        tar_iters = [
            wds.tarfile_samples([{"url": tar_url}]) for tar_url in multi_tar_urls
        ]

        try:
            # Loop over these iterators in parallel and combine the tar files from different modalities
            for multi_tar_files in zip(*tar_iters):
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

        _has_checked = True

        return sample_flatten

    return _inner

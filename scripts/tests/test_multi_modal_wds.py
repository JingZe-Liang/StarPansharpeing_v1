import sys

import torch
import webdataset as wds
import wids

sys.path.insert(0, __file__[: __file__.find("scripts")])
import re
from functools import partial

import braceexpand

from src.data.codecs import npz_decode_io, safetensors_decode_io, tiff_decode_io


def extract_modality_names(s):
    # Regular expression pattern to match anything enclosed in '{' and '}', and comma separated
    pattern = r"\{([^}]*)\}"
    match = re.search(pattern, s)
    return match.group(1).split(",") if match else []


def _map_without_extsion(sample):
    sample_dict = {}
    for k, v in sample.items():
        k_wo_ext = k
        _has_ext = k.find(".") != -1
        if _has_ext:
            k_wo_ext = k[: k.find(".")]
        sample_dict[k_wo_ext] = v

    return sample_dict


def _check_dots(s):
    if ".gz" in s:
        return s.count(".") == 2
    return s.count(".") == 1


def _merge_latent(
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


def _flatten_sub_dict(regardless_of_any_collisions=True):
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


def make_loader():
    ds_img = wds.DataPipeline(
        wds.ResampledShards(
            "data/MMSeg_YREB/[hyper_images,latent]/MMSeg_YREB_train_part-12_bands-MSI-{0000..0003}.tar"
        ),
        _merge_latent,
        wds.decode(
            wds.handle_extension("tif tiff", tiff_decode_io),
            wds.handle_extension("npz", npz_decode_io),
            wds.handle_extension("safetensors", safetensors_decode_io),
            "torch",
        ),
        wds.map(_map_without_extsion),
        wds.map(_flatten_sub_dict(regardless_of_any_collisions=True)),
        wds.shuffle(100),
    )

    return ds_img


ds_img = make_loader()
loader = wds.WebLoader(
    ds_img,
    batch_size=4,
    num_workers=0,
    pin_memory=True,
    drop_last=False,
    timeout=0,
)

for sample in loader:
    print(sample["__key__"], sample["img"].shape, sample["hrms_latent"].shape)

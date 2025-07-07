import sys
from functools import partial
from typing import Literal, Sequence

import numpy as np
import torch
import webdataset as wds

from src.data.codecs import (
    img_decode_io,
    mat_decode_io,
    npz_decode_io,
    safetensors_decode_io,
    tiff_decode_io,
)
from src.data.utils import (
    chained_dataloaders,
    expand_paths_and_correct_loader_kwargs_mm,
    flatten_sub_dict,
    generate_wds_config_modify_only_some_kwgs,
    merge_modalities,
    norm_img,
    remove_extension,
    remove_meta_data,
)
from src.utilities.config_utils import function_config_to_basic_types
from src.utilities.logging import log_print

# generative modeling dataloader needs HRMS images, bands informations or loader index (?)
# and even image captions (for RS5M captions can be accessed).

# img -> hyper_image/
# caption -> hyper_image/.caption
# bands -> hyper_image -> images.shape[1]


def get_image_bands_info(sample: dict):
    assert "hrms" in sample, "Sample must contain 'hrms' key"
    bs = sample["hrms"].shape[0]
    sample["bands"] = [sample["hrms"].shape[1]] * bs

    return sample


def extract_keys(samples: dict, keys: list[str]):
    return {key: samples[key] for key in keys}


# * --- WebDataset --- #


def get_generative_dataloaders(
    wds_paths: str | list[str],
    batch_size: int,
    num_workers: int,
    shuffle_size: int = 100,
    to_neg_1_1: bool = True,
    resample: bool = True,
    prefetch_factor: int = 6,
    pin_memory: bool = True,
    remove_meta: bool = False,
    normed_keys: list[str] = ["img"],
    **__discarded_kwargs,
) -> tuple[wds.DataPipeline, wds.WebLoader]:
    dataset = wds.DataPipeline(
        wds.ResampledShards(wds_paths) if resample else wds.SimpleShardList(wds_paths),
        merge_modalities,
        wds.decode(
            wds.handle_extension("tif tiff", tiff_decode_io),
            wds.handle_extension("jpg png jpeg image_content", img_decode_io),
            wds.handle_extension("npz", npz_decode_io),
            wds.handle_extension("safetensors", safetensors_decode_io),
            "torch",
        ),
        wds.map(remove_extension),
        wds.map(flatten_sub_dict(regardless_of_any_collisions=True)),
        wds.map(
            partial(
                norm_img,
                to_neg_1_1=to_neg_1_1,
                norm_keys=normed_keys,
                permute=False,
            )
        ),
        wds.shuffle(shuffle_size),
        wds.map(partial(extract_keys, keys=["hrms", "hrms_latent"])),
    )
    if remove_meta:
        dataset.append(wds.map(remove_meta_data))

    dataloader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=False,
        pin_memory=pin_memory,
    )
    dataloader = dataloader.map(get_image_bands_info)
    # dataloader = dataloader.with_length(10_000)  # 10k iterations of the dataloader

    log_print(f"[Pansharpening Dataset]: constructed the dataloader")

    return dataset, dataloader


# * --- Wids dataset --- #

from src.data.panshap_loader import MultimodalityDataloader


class GenerativeMMDataloader(MultimodalityDataloader): ...


# * --- Entry function --- #


@function_config_to_basic_types
def get_hyperspectral_img_loaders_with_different_backends_v2(
    paths: str | list[str] | list[list[str]],
    loader_type: str | None = None,
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
    loader_types: list[str] = []
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

        for i in range(len(rep_loader_kwargs)):
            kwg_ldt = rep_loader_kwargs[i].pop("loader_type", "webdataset")
            loader_types.append(kwg_ldt if loader_type is None else loader_type)
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
        loader_types.append(kwg_ldt if loader_type is None else loader_type)

    # * --- build datasets and dataloaders --- #
    datasets = []
    dataloaders = []

    # for-loop over the paths and rep_loader_kwargs
    for i, (p_lst, loader_kwargs, loader_type) in enumerate(
        zip(paths, rep_loader_kwargs, loader_types)
    ):
        p_lst: list[str] | str
        # webdataset or wids loader
        if loader_type in ("webdataset", "wids"):
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

            p_lst, loader_kwargs = expand_paths_and_correct_loader_kwargs_mm(
                p_lst, loader_kwargs
            )
            log_print(
                f"dataset group {i} gets paths: \n<cyan>[{p_lst}]</>\n" + "-" * 30
            )

            if loader_type == "webdataset":
                log_print("Using webdataset loader")
                dataset, dataloader = get_generative_dataloaders(p_lst, **loader_kwargs)
            # elif loader_type == "wids":
            #     log_print("Using wids loader")
            #     assert (
            #         isinstance(p_lst, list) and len(p_lst) == 1
            #     ), "must contain only one json file"
            #     p_lst = p_lst[0]
            #     assert (
            #         isinstance(p_lst, str) and p_lst.endswith(".json")
            #     ), f"wids loader expects a single json file as index_file, but got {p_lst}"
            #     dataset, dataloader = get_hyperspectral_wids_dataloaders(
            #         index_file=p_lst,
            #         **loader_kwargs,
            #     )
            else:
                raise ValueError(f"loader_type {loader_type} is not supported")
            datasets.append(dataset)
            dataloaders.append(dataloader)
        # folder loader
        # elif loader_type == "folder":
        #     log_print("Using folder loader")
        #     assert isinstance(paths, str), f"paths should be a string, but got paths"
        #     return only_hyperspectral_img_folder_dataloader(paths, **loader_kwargs)
        # else:
        #     raise ValueError(f"Unsupported loader type: {loader_type}")

    # prepare for curriculum
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

    # prepare chained unified dataloader
    dataloader = chained_dataloaders(
        dataloaders, chain_loader_infinit, shuffle_loaders, curriculum_fn
    )

    return datasets, dataloader


if __name__ == "__main__":
    # path = "data/MMSeg_YREB/[latents,pansharpening_pairs]/MMSeg_YREB_train_part-12_bands-MSI-0000.tar"

    # dataset, dataloader = get_generative_dataloaders(path, batch_size=8, num_workers=0)

    # for sample in dataloader:
    #     print(sample.keys())

    main_kwargs = {
        "batch_size": 8,
        "num_workers": 0,
    }
    loader_kwargs = [{}, {}]
    paths = [
        [
            "data/MMSeg_YREB/[latents,pansharpening_pairs]/MMSeg_YREB_train_part-12_bands-MSI-0000.tar",
        ],
        [
            "data/MMSeg_YREB/[latents,pansharpening_pairs]/MMSeg_YREB_train_part-12_bands-MSI-0001.tar",
        ],
    ]
    _, loader = get_hyperspectral_img_loaders_with_different_backends_v2(
        paths,
        loader_type="webdataset",
        basic_kwargs=main_kwargs,
        changed_kwargs_by_loader=loader_kwargs,
    )

    for sample in loader:
        print(sample.keys())

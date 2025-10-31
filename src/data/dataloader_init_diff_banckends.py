from typing import Any, Callable

import torch
from litdata import StreamingDataLoader
from loguru import logger
from torch.utils.data import DataLoader
from typing_extensions import List, Literal, Sequence

from src.utilities.config_utils import function_config_to_basic_types
from src.utilities.train_utils.state import StepsCounter

from .curriculums import get_curriculum_fn
from .hyperspectral_loader import (
    get_hyperspectral_dataloaders,
    get_hyperspectral_wids_dataloaders,
    only_hyperspectral_img_folder_dataloader,
)
from .litdata_hyperloader import (
    CaptionStreamingDataset,
    ConditionsStreamingDataset,
    GenerativeStreamingDataset,
    ImageStreamingDataset,
    IndexedCombinedStreamingDataset,
    SingleCycleStreamingDataset,
    SizeBasedBatchsizeStreamingDataloader,
)
from .utils import (
    chained_dataloaders,
    expand_paths_and_correct_loader_kwargs,
    generate_wds_config_modify_only_some_kwgs,
)

type ImageLoaderPath = str | list[str] | list[list[str]]
type SupportedLoaderType = Literal["webdataset", "folder", "wids", "mds"]


@function_config_to_basic_types
def get_hyperspectral_img_loaders_with_different_backends_v3(
    paths: ImageLoaderPath,
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
            - "mds": MDS streaming dataset loader
            - "wids": For WIDs streaming dataset loader
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
        logger.debug(
            f"loader_type is provided: {loader_type}, "
            f"input all paths should be {loader_type} loader",
        )

    # clear the kwargs
    # 1. paths is a list of list of string
    if isinstance(paths, list) and isinstance(paths[0], Sequence):
        logger.info(
            "input paths contains list of lists, we will chain the dataloader with each loader"
        )
        if rep_loader_kwargs is not None:  # every loader kwargs
            logger.debug(f"rep_loader_kwargs is provided: {rep_loader_kwargs}")
            assert isinstance(rep_loader_kwargs, list), (
                f"rep_loader_kwargs should be a list, but got {type(rep_loader_kwargs)}"
            )
            assert len(rep_loader_kwargs) == len(paths), (
                f"rep_loader_kwargs should be the same length as paths, "
                f"but got {len(rep_loader_kwargs)} and {len(paths)}"
            )
        elif basic_kwargs is not None:
            logger.debug("basic_kwargs is provided")
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
            logger.debug("rep_loader_kwargs is None, use the loader_kwargs")
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
        logger.info("input paths is a string, use the loader_kwargs", "debug")
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
            logger.debug(
                f"dataset group {i} gets paths: \n<green>[{'\n'.join(p_lst)}]</>\n"
                + "-" * 30
            )

            # construct webdataset/wids datasets and dataloaders
            if loader_type == "webdataset":
                logger.info("Using webdataset loader")
                dataset, dataloader = get_hyperspectral_dataloaders(
                    p_lst, **loader_kwargs
                )
            elif loader_type == "wids":
                logger.info("Using wids loader")
                if isinstance(p_lst, list) and len(p_lst) == 1:
                    p_lst = p_lst[0]

                dataset, dataloader = get_hyperspectral_wids_dataloaders(
                    index_file=p_lst, **loader_kwargs
                )
            elif loader_type == "mds":
                logger.info("use mds loader")
                ...
            else:
                raise ValueError(f"loader_type {loader_type} is not supported")

            datasets.append(dataset)
            dataloaders.append(dataloader)

        # > folder loader
        elif loader_type == "folder":
            logger.info("Using folder loader")
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
        logger.info(
            "chain loader generator is finite, the loader can not be iter and next. "
            "Please set <cyan>chain_loader_infinit=True</> if you are know what you are doing.",
            "warning",
        )

    # > prepare chained unified dataloader
    dataloader = chained_dataloaders(
        dataloaders, chain_loader_infinit, shuffle_loaders, curriculum_fn
    )

    return datasets, dataloader


class ChainedDataLoader:
    _datasets = []

    def __init__(
        self,
        dataloaders: list[StreamingDataLoader | DataLoader],
        infinite: bool = True,
        shuffle_loaders: bool = True,
        curriculum_type: str | None = None,
        curriculum_kwargs: dict = {},
        post_process_fn: Callable | None = None,
        *,
        loader_type: str = "litdata",
    ):
        self.dataloaders = dataloaders
        self.infinite = infinite
        self.shuffle_loaders = shuffle_loaders
        self.post_process_fn = post_process_fn
        self.loader_type = loader_type

        self.curriculum_type = curriculum_type
        self.curriculum_kwargs = curriculum_kwargs
        curriculum_fn = None
        if curriculum_type is not None:
            curriculum_fn = get_curriculum_fn(curriculum_type, **curriculum_kwargs)

        self._iterator_chained = chained_dataloaders(
            dataloaders=self.dataloaders,
            infinit=infinite,
            shuffle_loaders=shuffle_loaders,
            curriculum_fn=curriculum_fn,
            other_sample_fn=self.post_process_fn,
        )

    def next(self):
        return next(self._iterator_chained)

    def __iter__(self):
        yield from self._iterator_chained

    def state_dict(self):
        if self.loader_type != "litdata":
            return None
        return {
            "dl_state_dict": [
                {f"dl_{i}": dl.state_dict()} for i, dl in enumerate(self.dataloaders)
            ],
            # class attrs
            "infinite": self.infinite,
            "shuffle_loaders": self.shuffle_loaders,
            "curriculum_type": self.curriculum_type,
            "curriculum_kwargs": self.curriculum_kwargs,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.infinite = state_dict["infinite"]
        self.shuffle_loaders = state_dict["shuffle_loaders"]
        self.curriculum_type = state_dict["curriculum_type"]
        self.curriculum_kwargs = state_dict["curriculum_kwargs"]
        for i, dl_state_dict in enumerate(state_dict["dl_state_dict"]):
            self.dataloaders[i].load_state_dict(dl_state_dict[f"dl_{i}"])

    @classmethod
    def _create_litdata_one_img_dataloader(
        cls,
        input_dir: str | list[str],
        stream_ds_kwargs: dict = {},
        combined_kwargs: dict = {"batching_method": "per_stream"},
        loader_kwargs: dict = {},
    ):
        ds, dl = ImageStreamingDataset.create_dataloader(
            input_dir,
            stream_ds_kwargs=stream_ds_kwargs,
            combined_kwargs=combined_kwargs,
            loader_kwargs=loader_kwargs,
        )
        cls._datasets.append(ds)
        return ds, dl

    @classmethod
    def _create_litdata_one_gen_dataloader(
        cls,
        img_input_dir: str | list[str],
        condition_input_dir: str | list[str],
        caption_input_dir: str | list[str],
        img_kwargs: dict = {},
        cond_kwargs: dict = {},
        caption_kwargs: dict = {},
        gen_kwargs: dict = {},
        loader_kwargs: dict = {},
    ):
        ds, dl = GenerativeStreamingDataset.create_dataloader(
            img_input_dir,
            condition_input_dir,
            caption_input_dir,
            img_kwargs,
            cond_kwargs,
            caption_kwargs,
            gen_kwargs,
            loader_kwargs,
        )
        cls._datasets.append(ds)
        return ds, dl

    @classmethod
    def create_litdata_img_chained_loader(
        cls,
        input_dirs: list[dict[str, str | list[str]]],
        stream_ds_kwargs: list[dict[str, Any]] | dict[str, Any] = {},
        combined_kwargs: list[dict[str, Any]] | dict[str, Any] = {
            "batching_method": "per_stream"
        },
        loader_kwargs: list[dict[str, Any]] | dict[str, Any] = {},
        **chain_kwargs,
    ):
        n_inputs = len(input_dirs)
        if not is_list(stream_ds_kwargs):
            stream_ds_kwargs = [stream_ds_kwargs] * n_inputs
        if not is_list(combined_kwargs):
            combined_kwargs = [combined_kwargs] * n_inputs
        if not is_list(loader_kwargs):
            loader_kwargs = [loader_kwargs] * n_inputs

        assert (
            n_inputs
            == len(stream_ds_kwargs)
            == len(combined_kwargs)
            == len(loader_kwargs)
        ), (
            "The number of inputs must be equal to the number of stream_ds_kwargs, "
            "combined_kwargs, and loader_kwargs"
        )

        ld_dataloaders = []
        for stream, sk, ck, lk in zip(
            input_dirs, stream_ds_kwargs, combined_kwargs, loader_kwargs
        ):
            _, dl = cls._create_litdata_one_img_dataloader(stream, sk, ck, lk)
            ld_dataloaders.append(dl)

        return cls(ld_dataloaders, **chain_kwargs)


def is_list(x):
    return isinstance(x, list)


def is_dict(x):
    return isinstance(x, dict)


def create_litdata_img_hyper_loader(
    paths: dict[str, dict],
    size_based_batch_sizes: dict[int, int],
    loader_kwargs: list[dict],
    sub_combined_kwargs: list[dict],
    chained_loader_kwargs: dict,
):
    assert len(paths) == len(loader_kwargs), (
        "The number of paths must be equal to the number of loader_kwargs"
    )
    assert len(paths) == len(sub_combined_kwargs), (
        "The number of paths must be equal to the number of combined_kwargs"
    )

    datasets = []
    dataloaders = []
    for (path_sub_name, path_dict), load_kwgs, combined_kwgs in zip(
        paths.items(), loader_kwargs, sub_combined_kwargs
    ):
        datasets_ = []
        for subsub_name, path_lst in path_dict.items():
            path, ds_kwargs = path_lst
            ds = ImageStreamingDataset.create_dataset(input_dir=path, **ds_kwargs)
            datasets_.append(ds)

        if len(datasets_) == 1:
            sub_combined_ds = datasets_[0]
        else:
            sub_combined_ds = IndexedCombinedStreamingDataset(
                datasets=datasets_,
                batching_method="per_stream",
                iterate_over_all=False,
                combined_is_cycled=True,
                # weights=[1.0] * len(datasets_),
                **combined_kwgs,
            )
        datasets.append(sub_combined_ds)

        dataloader = StreamingDataLoader(
            sub_combined_ds,
            size_based_batch_sizes=size_based_batch_sizes,
            **load_kwgs,
        )
        dataloaders.append(dataloader)

    # Chain dataloaders
    chained_dataloader = ChainedDataLoader(dataloaders, **chained_loader_kwargs)
    return datasets, chained_dataloader


def test_litdata_hyper_dataloaders():
    from tqdm import tqdm

    from .path_consts import HYPERSPECTRAL_PATHS, MULTISPECTRAL_PATHS, RGB_PATHS

    stream_ds_kwargs = {
        "transform_prob": 0.0,
        "resize_before_transform": 256,
        "is_cycled": True,
    }
    loader_kwargs = {
        "batch_size": 16,
        "num_workers": 12,
        "persistent_workers": False,
        "prefetch_factor": None,
        # "collate_fn": collate_fn_skip_none,
        "shuffle": False,
    }
    size_based_batch_sizes = {128: 16, 256: 12, 512: 6}

    # RGB dataset
    stream_ds_kwargs_ = stream_ds_kwargs.copy()
    rgb_paths, kwargs = list(RGB_PATHS.values())[0]
    stream_ds_kwargs_.update(kwargs)
    rgb_ds = ImageStreamingDataset.create_dataset(
        input_dir=rgb_paths,
        combined_kwargs={"batching_method": "stratified"},
        **stream_ds_kwargs_,
    )
    rgb_dl = StreamingDataLoader(
        rgb_ds, batch_size=6, num_workers=8, persistent_workers=True
    )

    # Multispectral dataset
    multi_ds = []
    for name, (paths, kwargs) in MULTISPECTRAL_PATHS.items():
        stream_ds_kwargs_ = stream_ds_kwargs.copy()
        stream_ds_kwargs_.update(kwargs)
        ds = ImageStreamingDataset.create_dataset(
            input_dir=paths,
            combined_kwargs={"batching_method": "per_stream"},
            **stream_ds_kwargs_,
        )
        multi_ds.append(ds)
    multi_ds = IndexedCombinedStreamingDataset(
        combined_is_cycled=True,
        datasets=multi_ds,
        batching_method="per_stream",
        weights=[1.0] * len(multi_ds),
        iterate_over_all=False,
    )
    multi_dl = SizeBasedBatchsizeStreamingDataloader(
        dataset=multi_ds, size_based_batch_sizes=size_based_batch_sizes, **loader_kwargs
    )

    # Hyperspectral dataset
    hyper_ds = []
    for name, (paths, kwargs) in HYPERSPECTRAL_PATHS.items():
        stream_ds_kwargs_ = stream_ds_kwargs.copy()
        stream_ds_kwargs_.update(kwargs)
        ds = ImageStreamingDataset.create_dataset(
            input_dir=paths,
            combined_kwargs={"batching_method": "per_stream"},
            **stream_ds_kwargs_,
        )
        hyper_ds.append(ds)
    hyper_ds = IndexedCombinedStreamingDataset(
        combined_is_cycled=True,
        datasets=hyper_ds,
        batching_method="per_stream",
        weights=[1.0] * len(hyper_ds),
        iterate_over_all=False,
    )
    hyper_dl = SizeBasedBatchsizeStreamingDataloader(
        dataset=hyper_ds, size_based_batch_sizes=size_based_batch_sizes, **loader_kwargs
    )

    # Chained dataset
    chained_dl = ChainedDataLoader(
        [rgb_dl, multi_dl, hyper_dl],
        infinite=True,
        shuffle_loaders=False,
    )

    # For-loop it
    for i, batch in enumerate(tqdm(chained_dl, total=1000)):
        print(f"Batch {i}: {batch['img'].shape}")


if __name__ == "__main__":
    """
        python -m src.data.dataloader_init_diff_banckends
    """
    test_litdata_hyper_dataloaders()

import hashlib
import math
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, cast

import torch
import webdataset as wds
import wids

from src.data.codecs import (
    img_decode_io,
    npz_decode_io,
    safetensors_decode_io,
    tiff_decode_io,
    wids_caption_embed_decode,
    wids_image_decode,
    wids_latent_decode,
    wids_remove_none_keys,
)
from src.data.curriculums import get_curriculum_fn
from src.data.utils import (
    chained_dataloaders,
    flatten_sub_dict,
    merge_modalities,
    norm_img,
    remove_extension,
    remove_meta_data,
)
from src.data.wids_samplers import (
    IndexFilteredDistributedSampler,
    IndexFilteredSampler,
)
from src.utilities.config_utils import function_config_to_basic_types
from src.utilities.logging import log_print

type ModalityPrefixedName = Literal[
    "pixel_image",
    "condition_image",
    "pansharpening_image",
    "denoising_image",
    "image_latent",
    "condition_latent",
    "pansharpening_latent",
    "denoising_latent",
    "captions",
]
type ModalityName = str | ModalityPrefixedName
type IndexFilePath = str | Path

loader_seed = int(hashlib.sha256("uestc_ZihanCao_add_my_wechat:iamzihan123".encode()).hexdigest(), 16) % (2**31)
loader_generator = torch.Generator().manual_seed(loader_seed)


def local_name_fn(name, prefix: str | None = None):
    """Helper function to create local names with a prefix."""

    if prefix is None:
        return name
    else:
        return f"{prefix}/{name}"


# * --- multimodal wids dataloader --- #


class GetitemType(str, Enum):
    FLAT = "flat"
    EXTRACTED = "extracted"
    STRUCTURED = "structured"


class MultimodalityDataloader:
    registed_decoders: dict[str, Callable] = {}

    @function_config_to_basic_types
    def __init__(
        self,
        wds_paths: dict[ModalityName, IndexFilePath],
        codecs: dict[ModalityName, Callable] | None = None,
        to_neg_1_1: bool = True,
        permute: bool = True,
        wids_local_name_prefix: str | None = None,
        # 'structured' | 'flat' | 'extracted'
        getitem_type: GetitemType = GetitemType.FLAT,
        # num_of_modalities[extracted_keys[(in_key, out_key)]]
        extracted_keys: list[list[tuple[str, str]]] | None = None,
        getitem_remove_meta: bool = False,
        **kwargs,
    ):
        # get wids datasets
        self.wds_paths = wds_paths
        self.datasets = {}
        self.mm_names = []
        self.getitem_type = getitem_type
        self.extracted_keys = extracted_keys
        self.getitem_remove_meta = getitem_remove_meta

        # default codecs
        self.supported_codecs_mapping = self._get_supported_decodes(to_neg_1_1=to_neg_1_1, permute=permute)
        if codecs is not None:
            self.supported_codecs_mapping.update(codecs)

        # loader kwargs
        self.batch_size = None
        self.num_workers = None
        self.shuffle_size = None
        self.drop_last = None
        self.other_kwargs = kwargs

        # transformations
        self._named_transformations = {}

        # hook before init datasets
        self.before_init_datasets()

        self.total_len = -1
        if extracted_keys is not None:
            assert len(extracted_keys) == len(wds_paths), "extracted_keys must have the same length as wds_paths"

        for name, p in self.wds_paths.items():
            # decode functions
            decode_fn = self.supported_codecs_mapping.get(name, None)
            if isinstance(decode_fn, (list, tuple)):
                transformations = [*decode_fn, wids_remove_none_keys]
            elif decode_fn is not None:
                transformations = [decode_fn, wids_remove_none_keys]
            else:
                raise ValueError(f"No codec found for {name}")

            self._named_transformations[name] = transformations
            log_print(f"[wids dataset]: {name}")

            # add wids dataset
            self.mm_names.append(name)
            ds = wids.ShardListDataset(
                p,
                localname=partial(local_name_fn, prefix=wids_local_name_prefix),
                transformations=transformations,  # type: ignore
            )

            # check length of multimodal datasets
            if self.total_len == -1:
                self.total_len = len(ds)
            else:
                assert self.total_len == len(ds), (
                    f"All datasets must have the same length, "
                    f"but got {self.total_len} of previous dataset and {len(ds)} for modality {name} dataset"
                )
            self.datasets[name] = ds

        # hook after init datasets
        self.datasets = self.after_init_datasets(self.datasets)

    @classmethod
    def register_decoder(cls, name: ModalityName, fn: Callable):
        assert callable(fn), "Decoder must be a callable function"
        cls.registed_decoders[name] = fn

    def _get_supported_decodes(self, **kwargs):
        img_fn_ = partial(
            wids_image_decode,
            to_neg_1_1=kwargs.get("to_neg_1_1", True),
            permute=kwargs.get("permute", True),
            resize=None,
            process_img_keys="ALL",
        )
        default_decodes = {
            # hyperspectral images or pairs
            "pixel_image": img_fn_,
            "condition_image": img_fn_,
            "pansharpening_image": img_fn_,
            "denoising_image": img_fn_,
            # images, conditions or downstream latents
            "image_latent": wids_latent_decode,
            "condition_latent": wids_latent_decode,
            "pansharpening_latent": partial(wids_latent_decode, return_dict=True),
            "denoising_latent": partial(wids_latent_decode, return_dict=True),  # if gt is in, return_dict=True
            # language captions and its embeddings
            "captions": wids_caption_embed_decode,
        }
        return default_decodes | self.registed_decoders

    def __len__(self):
        len_f = self.total_len / getattr(self, "batch_size", 1)
        return math.ceil(len_f) if getattr(self, "drop_last", False) else math.floor(len_f)

    def __getitem__(self, index):
        modality_sample: dict[ModalityName | str, Any] = {}
        # modality_sample = getattr(self, "before_getitem", lambda init_sample: init_sample)(modality_sample)

        check_sample_key = None
        for i, (name, ds) in enumerate(self.datasets.items()):
            sample = ds[index]
            # sample key
            if check_sample_key is None:
                check_sample_key = sample["__key__"]
            else:
                assert check_sample_key == sample["__key__"], (
                    f"All samples must have the same key, "
                    f"but got {check_sample_key} and {sample['__key__']} for modality {name}"
                )

            # convert the sample into a flat dict
            if isinstance(sample, dict):
                if self.getitem_type == GetitemType.FLAT:
                    modality_sample.update(self._extract_key_for_non_nested_dict(sample, name))
                elif self.getitem_type == GetitemType.EXTRACTED and self.extracted_keys is not None:
                    ext_key = self.extracted_keys[i]
                    modality_sample.update(self._extract_keys(sample, ext_key, None))
                elif self.getitem_type == GetitemType.EXTRACTED:
                    modality_sample[name] = self._extract_without_extension(sample)
                elif self.getitem_type == GetitemType.STRUCTURED:
                    modality_sample[name] = self._extract_without_extension(sample)
                else:
                    raise ValueError(
                        f"Invalid getitem_type {self.getitem_type}, must be 'flat', 'extracted' or 'structured'"
                    )
            else:
                modality_sample[name] = {"default": sample}

        # hook after getitem
        modality_sample = self.after_getitem(modality_sample)

        if self.getitem_remove_meta:
            modality_sample = self._remove_all_meta_info(modality_sample)

        return modality_sample

    # * --- Getitem utilities --- #

    def _remove_all_meta_info(self, modal_sample: dict) -> dict[str, Any]:
        return {k: v for k, v in modal_sample.items() if not k.startswith("__")}

    def _extract_key_for_non_nested_dict(
        self, modal_sample: dict, modality_name: str, without_extension=True
    ) -> dict[str, Any]:
        meta_info_keys = []
        content_not_dunder = []
        for x in modal_sample.keys():
            if not x.startswith("__"):
                content_not_dunder.append(x)
            else:
                meta_info_keys.append(x)

        if len(content_not_dunder) == 1:  # modality_name: Tensor
            modal_sample_new = {modality_name: modal_sample[content_not_dunder[0]]}
        else:  # modality_name: {"img": Tensor, 'img2': Tensor, ...}
            modal_sample_new = {}
            for name in content_not_dunder:
                new_name = Path(name).stem.rsplit(".", 1)[-1] if without_extension else name
                modal_sample_new[new_name] = modal_sample[name]

        # Add meta infos
        # add modality name prefix to avoid collision
        keys_ = None
        shard_ = None
        for meta_key in meta_info_keys:
            if meta_key == "__key__":
                keys_ = modal_sample[meta_key]
            if meta_key == "__shard__":
                shard_ = modal_sample[meta_key]

            meta_key_new = f"__{modality_name}_{meta_key[2:]}"
            modal_sample_new[meta_key_new] = modal_sample[meta_key]

        if keys_ is not None:
            modal_sample_new["__key__"] = keys_
        if shard_ is not None:
            modal_sample_new["__shard__"] = shard_

        return modal_sample_new

    def _extract_without_extension(self, samples: dict):
        _keys = list(samples.keys())
        for k in _keys:
            if not str(k).startswith("__"):
                # .image.tiff -> image
                # img.png -> img
                k_e = Path(k).stem.rsplit(".", 1)[-1]
                samples[k_e] = samples.pop(k)

        return samples

    def _extract_keys(
        self,
        samples: dict,
        keys: list[tuple],
        modality_name: str | None = None,
    ):
        # flatten the nested dict into a flat dict

        def extract_fn(x: dict, key: str):
            parts = key.split(".")
            for p in parts:
                x = x.get(p, None)  # type: ignore
                if x is None:
                    return None
            return x

        extracted = {}
        for in_k, out_k in keys:
            extract_value = extract_fn(samples, in_k)
            if extract_value is not None:
                if modality_name is not None:
                    out_k = modality_name + "_" + out_k
                extracted[out_k] = extract_value

        return extracted

    @staticmethod
    def get_loader_sampler(dataset, shuffle_size: int = -1):
        img_index = None
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            log_print("[wids mm dataset]: use distributed sampler")
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
            log_print("[wids mm dataset]: use single process sampler")
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

    # * --- Create loader interface --- #
    @classmethod
    def create_loader(
        cls,
        wds_paths: dict[ModalityName, IndexFilePath],
        to_neg_1_1: bool = True,
        permute: bool = True,
        wids_local_name_prefix: str | None = None,
        return_classed_dict: bool | None = None,
        extracted_keys: list[list[tuple[str, str]]] | None = None,
        batch_size: int = 32,
        num_workers: int = 0,
        shuffle_size: int = 100,
        drop_last: bool = False,
        **kwargs,
    ):
        dataset = cls(
            wds_paths,
            to_neg_1_1=to_neg_1_1,
            permute=permute,
            wids_local_name_prefix=wids_local_name_prefix,
            extracted_keys=extracted_keys,
            return_classed_dict=return_classed_dict,
            **kwargs,
        )

        # dataloader
        sampler = cls.get_loader_sampler(
            dataset.datasets[dataset.mm_names[0]],  # nested solution
            shuffle_size=shuffle_size,
        )

        setattr(dataset, "batch_size", batch_size)
        setattr(dataset, "num_workers", num_workers)
        setattr(dataset, "shuffle_size", shuffle_size)
        setattr(dataset, "drop_last", drop_last)

        dataloader = wds.WebLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=None if num_workers == 0 else 6,
            drop_last=drop_last,
            sampler=sampler,
            generator=loader_generator,
            # collate_fn=multimodal_wids_collate_fn,
        )
        # if shuffle_size > 0:
        #     dataloader = dataloader.shuffle(shuffle_size)

        # dataloader = dataloader.with_length(10_000)

        return dataset, dataloader

    # * --- Hooks --- #
    def before_init_datasets(self):
        """Hook before initializing datasets."""
        pass

    def after_init_datasets(self, datasets: dict[ModalityName, wids.ShardListDataset]):
        """Hook after initializing datasets."""
        return self.datasets

    def before_getitem(self, init_sample: dict) -> Any:
        """Hook before getting an item from the dataset."""
        return init_sample

    def after_getitem(self, sample: dict) -> Any:
        """Hook after getting an item from the dataset."""
        return sample


# * --- Multi-modal Webdataset --- #


@function_config_to_basic_types
def get_multimodal_dataloaders(
    wds_paths: str | list[str],
    batch_size: int,
    num_workers: int,
    shuffle_size: int = 100,
    to_neg_1_1: bool = True,
    resample: bool = True,
    prefetch_factor: int = 6,
    pin_memory: bool = True,
    remove_meta: bool = False,
    normed_keys: list[str] = ["img", "hrms", "lrms", "pan"],
) -> tuple[wds.DataPipeline, wds.WebLoader]:
    dataset = wds.DataPipeline(
        wds.ResampledShards(wds_paths) if resample else wds.SimpleShardList(wds_paths),
        merge_modalities,
        wds.decode(
            wds.handle_extension("tif tiff", tiff_decode_io),
            wds.handle_extension("jpg png jpeg", img_decode_io),
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
        # wds.batched(batch_size, partial=True),
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
    dataloader = dataloader.with_length(10_000)  # 10k pairs of the dataloader
    # dataloader = dataloader.unbatched()

    # if shuffle_size > 0:
    #     dataloader = dataloader.shuffle(shuffle_size)
    # dataloader = dataloader.batched(batch_size)

    log_print(f"[Pansharpening Dataset]: constructed the dataloader")

    return dataset, dataloader


@function_config_to_basic_types
def get_mm_chained_loaders(
    paths: list[dict[ModalityName, IndexFilePath]],
    basic_kwargs: dict | None = None,
    changed_kwargs_by_loader: list[dict] | None = None,
    shuffle_loaders: bool = True,
    curriculum_type: str | None = None,
    curriculum_kwargs: dict | None = None,
):
    # < 0. assertions
    assert all(isinstance(p, dict) for p in paths), (
        "paths must be a list of dictionaries, "
        "where each dictionary contains modality names as keys and index file paths as values."
    )
    if changed_kwargs_by_loader is None:
        changed_kwargs_by_loader = [{} for _ in paths]
    assert len(paths) == len(changed_kwargs_by_loader), "changed_kwargs_by_loader must have the same length as paths"

    # < 1. loop index file and create loaders
    datasets = []
    loaders = []
    curriculum_fn = None
    for i, (path_dict, per_loader_options) in enumerate(zip(paths, changed_kwargs_by_loader)):
        if basic_kwargs is None:
            loader_kwargs = per_loader_options
        else:
            loader_kwargs = basic_kwargs.copy()
            loader_kwargs.update(per_loader_options)

        log_print(
            "create loader for path {path_dict} with loader_kwargs {loader_kwargs}",
            "debug",
        )

        dataset, loader = MultimodalityDataloader.create_loader(
            wds_paths=path_dict,
            **loader_kwargs,
        )
        datasets.append(dataset)
        loaders.append(loader)

        # < 1.1 curriculums fn
        if curriculum_type is not None:
            assert curriculum_kwargs is not None, f"curriculum_kwargs must be provided if {curriculum_type=}."
            curriculum_fn = get_curriculum_fn(  # type: ignore
                c_type=curriculum_type,
                **curriculum_kwargs,
            )
            log_print(
                f"use {curriculum_type} curriculum learning with kwargs: {curriculum_kwargs} for loader {path_dict}",
                "debug",
            )

    # < 2. chain dataloaders
    assert len(datasets) > 0 and len(loaders) > 0, "At least one dataset and one loader must be provided."
    dataloader = chained_dataloaders(loaders, shuffle_loaders=shuffle_loaders, curriculum_fn=curriculum_fn)

    return datasets, dataloader


# * --- testers --- #


def __test_wids_mm_loaders(*args):
    if 0 < len(args) < 2:
        torch.distributed.init_process_group(
            backend="nccl",  # or "nccl" if you have GPUs
            init_method="tcp://localhost:12346",
            world_size=args[1],
            rank=args[0],  # this will be set by mp.spawn
        )

    datasets, loaders = MultimodalityDataloader.create_loader(
        # {
        #     "hyper_images": "data/DCF_2020/hyper_images/shardindex.json",
        #     # "latents": "data/MMSeg_YREB/latents/shardindex.json",
        #     "conditions" "data/DCF_2020/conditions/shardindex.json",
        # },
        {
            "captions": "data/DCF_2020/condition_captions/indexshard.json",
            "condition_image": "data/DCF_2020/conditions/shardindex.json",
            "pixel_image": "data/DCF_2020/hyper_images/shardindex.json",
        },
        batch_size=2,
        num_workers=0,
        return_classed_dict=True,
        # extracted_keys=[
        #     # [("img", "img")],
        #     # [("latents.hrms_latent", "hrms_latent")],
        #     [(".img.tiff", "img")],
        #     [
        #         (".mlsd.png", "mlds"),
        #         (".hed.png", "hed"),
        #         (".segmentation.png", "segmentation"),
        #         (".sketch.png", "sketch"),
        #     ],
        # ],
    )

    # if torch.distributed.is_initialized():
    #     rank = torch.distributed.get_rank()
    # else:
    #     rank = 0

    rank = args[0] if len(args) > 0 else 0

    for i, sample in enumerate(loaders):
        # print(f"Rank {rank} - Batch {i}: index {sample['latents']['__index__']}")
        print(sample.keys())


if __name__ == "__main__":
    __test_wids_mm_loaders()

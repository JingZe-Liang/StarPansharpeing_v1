import math
import sys
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
import webdataset as wds
import wids

from src.data.codecs import (
    img_decode_io,
    mat_decode_io,
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
    expand_paths_and_correct_loader_kwargs,
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
from src.utilities.logging import log_print

type ModalityName = str
type IndexFilePath = str | Path
import hashlib

loader_seed = int(
    hashlib.sha256("uestc_ZihanCao_add_my_wechat:iamzihan123".encode()).hexdigest(), 16
) % (2**31)
loader_generator = torch.Generator().manual_seed(loader_seed)


def local_name_fn(name, prefix: str | None = None):
    """Helper function to create local names with a prefix."""

    if prefix is None:
        return name
    else:
        return f"{prefix}/{name}"


# * --- webdataset dataloader --- #


def get_panshap_dataloaders(
    wds_paths: str | list[str],
    batch_size: int,
    num_workers: int,
    shuffle_size: int = 100,
    to_neg_1_1: bool = True,
    resample: bool = True,
    latent_ext: Literal["safetensors", "npy", "npz"] = "safetensors",
    shardshuffle: bool = True,
):
    if isinstance(wds_paths, str):
        wds_paths = [wds_paths]

    dataset = wds.WebDataset(
        wds_paths,
        resampled=resample,
        shardshuffle=shardshuffle,
        handler=wds.warn_and_continue,
        nodesplitter=wds.shardlists.split_by_node,
        workersplitter=wds.split_by_worker,
        seed=2025,
        verbose=True,
        detshuffle=True if shuffle_size > 0 else False,
    )

    dataset = dataset.decode(
        wds.handle_extension("tif tiff", tiff_decode_io),
        wds.handle_extension("jpg png jpeg", img_decode_io),
        wds.handle_extension("mat", mat_decode_io),
        wds.handle_extension("safetensors", safetensors_decode_io),
        wds.handle_extension("npz", npz_decode_io),
        "torch",
    )

    # * --- dict mapper of images and latents ---

    def img_dict_mapper_with_ext(sample: dict):
        # keys: ['hrms', 'lrms', 'pan', 'ms', 'hrms_latent', 'lrms_latent', 'pan_latent', 'ms_latent']
        hrms = torch.as_tensor(sample["hrms.tiff"]).float()
        lrms = torch.as_tensor(sample["lrms.tiff"]).float()
        pan = torch.as_tensor(sample["pan.tiff"]).float()

        # normalize
        hrms = hrms / hrms.max()
        lrms = lrms / lrms.max()
        pan = pan / pan.max()

        if to_neg_1_1:
            hrms = hrms * 2 - 1
            lrms = lrms * 2 - 1
            pan = pan * 2 - 1

        # if has latents
        if f"hrms_latents.{latent_ext}" in sample or f"hrms_latent.npy" in sample:
            if latent_ext == "npz":
                sample_dict = sample["latenets.npz"]
                hrms_latent = torch.as_tensor(
                    sample_dict["hrms_latent"], dtype=torch.float32
                )
                lrms_latent = torch.as_tensor(
                    sample_dict["lrms_latent"], dtype=torch.float32
                )
                pan_latent = torch.as_tensor(
                    sample_dict["pan_latent"], dtype=torch.float32
                )
            elif latent_ext == "safetensors":
                sample_dict = sample["latents.safetensors"]
                hrms_latent = sample_dict["hrms_latent"].float()
                lrms_latent = sample_dict["lrms_latent"].float()
                pan_latent = sample_dict["pan_latent"].float()
            else:
                sample_dict = sample["latents.npz"]
                hrms_latent = torch.as_tensor(
                    sample["hrms_latent"], dtype=torch.float32
                )
                lrms_latent = torch.as_tensor(
                    sample["lrms_latent"], dtype=torch.float32
                )
                pan_latent = torch.as_tensor(sample["pan_latent"], dtype=torch.float32)

            sample = {
                "hrms": hrms,
                "lrms": lrms,
                "pan": pan,
                "hrms_latent": hrms_latent,
                "lrms_latent": lrms_latent,
                "pan_latent": pan_latent,
            }

        else:
            sample = {
                "hrms": hrms,
                "lrms": lrms,
                "pan": pan,
            }
        return sample

    dataset = dataset.map(img_dict_mapper_with_ext)
    dataset: wds.WebDataset

    # since we do not use any transforms, the batch and unbatch is useless.

    dataloader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=None if num_workers == 0 else 6,
        drop_last=False,
    )

    dataloader = dataloader.with_length(10_000)  # 10k pairs of the dataloader

    log_print(f"[Pansharpening Dataset]: constructed the dataloader")

    return dataset, dataloader


# * --- multimodal wids dataloader --- #


class MultimodalityDataloader:
    def __init__(
        self,
        wds_paths: dict[ModalityName, IndexFilePath],
        to_neg_1_1: bool = True,
        permute: bool = True,
        wids_local_name_prefix: str | None = None,
        return_nested_dict: bool | None = None,
        extracted_keys: list[list[tuple[str, str]]]
        | None = None,  # num_of_modalities[extracted_keys[(in_key, out_key)]]
    ):
        # get wids datasets
        self.wds_paths = wds_paths
        self.datasets = {}
        self.mm_names = []
        self.return_nested_dict = return_nested_dict  # remove the '__any_key' keys
        self.extracted_keys = extracted_keys

        # loader kwargs
        self.batch_size = None
        self.num_workers = None
        self.shuffle_size = None
        self.drop_last = None
        self.total_len = -1

        if extracted_keys is not None:
            assert len(extracted_keys) == len(wds_paths), (
                "extracted_keys must have the same length as wds_paths"
            )

        for name, p in self.wds_paths.items():
            self.mm_names.append(name)
            ds = wids.ShardListDataset(
                p,
                localname=partial(local_name_fn, prefix=wids_local_name_prefix),
                transformations=[
                    # image decoder ===
                    partial(
                        wids_image_decode,
                        to_neg_1_1=to_neg_1_1,
                        permute=permute,
                        resize=None,
                        process_img_keys="ALL",
                    ),  # type: ignore
                    # latents, caption decoders ===
                    wids_latent_decode,
                    wids_caption_embed_decode,
                    wids_remove_none_keys,
                ],
            )
            if self.total_len == -1:
                self.total_len = len(ds)
            else:
                assert self.total_len == len(ds), (
                    f"All datasets must have the same length, "
                    f"but got {self.total_len} of previous dataset and {len(ds)} for modality {name} dataset"
                )
            self.datasets[name] = ds

    def __len__(self):
        len_f = self.total_len / getattr(self, "batch_size", 1)
        return (
            math.ceil(len_f) if getattr(self, "drop_last", False) else math.floor(len_f)
        )

    def __getitem__(self, index):
        modality_sample: dict[ModalityName | str, Any] = {}
        for i, (name, ds) in enumerate(self.datasets.items()):
            sample = ds[index]
            if isinstance(sample, dict):  # * <-- default in this case
                if self.return_nested_dict is not None and not self.return_nested_dict:
                    modality_sample[name] = self.extract_key_for_non_nested_dict(sample)
                elif self.extracted_keys is not None:
                    ext_key = self.extracted_keys[i]
                    modality_sample.update(self.extract_keys(sample, ext_key, None))
                else:
                    modality_sample.update(self.extract_without_extension(sample))
            else:
                modality_sample[name] = {"default": sample}

        return modality_sample

    def extract_key_for_non_nested_dict(self, subsample: dict):
        content_not_dunder = [x for x in subsample.keys() if not x.startswith("__")]

        if len(content_not_dunder) == 1:  # modality_name: Tensor
            return subsample[content_not_dunder[0]]
        else:  # modality_name: {"img": Tensor, 'img2': Tensor, ...}
            subsample_f = {}
            for name in content_not_dunder:
                subsample_f[name] = subsample[name]

        return subsample_f

    def extract_without_extension(self, samples: dict):
        _keys = list(samples.keys())
        for k in _keys:
            if not str(k).startswith("__"):
                # .image.tiff -> image
                # img.png -> img
                k_e = Path(k).stem.rsplit(".", 1)[-1]
                samples[k_e] = samples.pop(k)

        return samples

    def extract_keys(
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
        if (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
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

    @classmethod
    def create_loader(
        cls,
        wds_paths: dict[ModalityName, IndexFilePath],
        to_neg_1_1: bool = True,
        permute: bool = True,
        wids_local_name_prefix: str | None = None,
        return_nested_dict: bool | None = None,
        extracted_keys: list[list[tuple[str, str]]] | None = None,
        batch_size: int = 32,
        num_workers: int = 1,
        shuffle_size: int = 100,
        drop_last: bool = False,
        resample: bool = True,
        **__discarded_kwargs,
    ):
        dataset = cls(
            wds_paths,
            to_neg_1_1=to_neg_1_1,
            permute=permute,
            wids_local_name_prefix=wids_local_name_prefix,
            extracted_keys=extracted_keys,
            return_nested_dict=return_nested_dict,
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


# * --- Multi-modal Webdataset --- #


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


def get_mm_chained_loaders(
    paths: list[dict[ModalityName, IndexFilePath]],
    basic_kwargs: dict | None = None,
    changed_kwargs_by_loader: list[dict] | None = None,
    shuffle_loaders: bool = True,
    curriculum_type: str | None = None,
    curriculum_kwargs: dict | None = None,
):
    # > assertions
    assert all(isinstance(p, dict) for p in paths), (
        "paths must be a list of dictionaries, "
        "where each dictionary contains modality names as keys and index file paths as values."
    )
    if changed_kwargs_by_loader is None:
        changed_kwargs_by_loader = [{} for _ in paths]
    assert len(paths) == len(changed_kwargs_by_loader), (
        "changed_kwargs_by_loader must have the same length as paths"
    )

    # > loop index file and create loaders
    datasets = []
    loaders = []
    for i, (path_dict, per_loader_options) in enumerate(
        zip(paths, changed_kwargs_by_loader)
    ):
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

        # > curriculums fn
        if curriculum_type is not None:
            assert curriculum_kwargs is not None, (
                f"curriculum_kwargs must be provided if {curriculum_type=}."
            )
            curriculum_fn = get_curriculum_fn(  # type: ignore
                c_type=curriculum_type,
                **curriculum_kwargs,
            )
            log_print(
                f"use {curriculum_type} curriculum learning with kwargs: {curriculum_kwargs} for loader {path_dict}",
                "debug",
            )
        else:
            curriculum_fn = None

    # > chain dataloaders
    assert len(datasets) > 0 and len(loaders) > 0, (
        "At least one dataset and one loader must be provided."
    )
    dataloader = chained_dataloaders(
        loaders, shuffle_loaders=shuffle_loaders, curriculum_fn=curriculum_fn
    )

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
        #     "conditions": "data/DCF_2020/conditions/shardindex.json",
        # },
        {
            "conditions": "data/DCF_2020/conditions/shardindex.json",
            "img": "data/DCF_2020/hyper_images/shardindex.json",
        },
        batch_size=2,
        num_workers=0,
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
    # _, loader = get_panshap_dataloaders(
    #     wds_paths=[
    #         "data/pansharpening/MMSeg_YREB/dataset_0000.tar",
    #         # "data/pansharpening/MMSeg_YREB/dataset_0001.tar",
    #     ],
    #     batch_size=4,
    #     num_workers=1,
    #     latent_ext="npz",
    # )
    # for i, sample in enumerate(loader):
    #     # print(sample)
    #     # break
    #     print(i)

    # _, loader = get_multimodal_dataloaders(
    #     wds_paths=[
    #         "data/MMSeg_YREB/[hyper_images,pansharpening_pairs,latents]/MMSeg_YREB_train_part-12_bands-MSI-{0000..0003}.tar"
    #     ],
    #     batch_size=32,
    #     num_workers=3,
    #     resample=True,
    #     prefetch_factor=6,
    #     pin_memory=False,
    #     shuffle_size=-1,
    # )

    # from tqdm import tqdm

    # for sample in tqdm(loader):
    #     print(sample.keys())
    #     # pass

    # for name, data in sample.items():
    #     print(f"  {name}: {data['img'].shape if 'img' in data else data.shape}")
    # if i >= 5:
    #     break

    __test_wids_mm_loaders()

    # # mp
    # import torch.multiprocessing as mp

    # mp.set_start_method("spawn")

    # # start with 2 processes
    # world_size = 2
    # # torch distributed init

    # mp.spawn(
    #     test_wids_mm_loaders,
    #     args=(world_size,),
    #     nprocs=world_size,
    #     join=True,
    # )

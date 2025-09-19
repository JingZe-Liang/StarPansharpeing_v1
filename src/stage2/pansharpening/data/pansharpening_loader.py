from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Literal

import beartype
import h5py
import torch
import webdataset as wds
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm

from src.data.codecs import (
    img_decode_io,
    mat_decode_io,
    npz_decode_io,
    safetensors_decode_io,
    tiff_decode_io,
    wids_image_decode,
)
from src.data.multimodal_loader import MultimodalityDataloader
from src.data.utils import norm_img_ as norm_img_fn
from src.data.utils import permute_img_to_chw, remove_extension, to_tensor
from src.utilities.config_utils import function_config_to_basic_types
from src.utilities.logging import log_print

SUPPORTED_SATELLITES = {"WV2", "WV3", "WV4", "QB", "IKONOS"}


# * --- Webdataset dataloader --- #


def img_dict_mapper_with_ext(
    sample: dict, to_neg_1_1: bool = True, latent_ext="safetensors"
):
    # keys: ['hrms', 'lrms', 'pan', 'ms', 'hrms_latent', 'lrms_latent', 'pan_latent', 'ms_latent']
    hrms = permute_img_to_chw(to_tensor(sample["hrms"]))
    lrms = permute_img_to_chw(to_tensor(sample["lrms"]))
    pan = permute_img_to_chw(to_tensor(sample["pan"]))  # h, w
    if pan.ndim == 2:
        pan = pan[None]  # add channel dim

    # normalize
    norm_fn_partial = partial(
        norm_img_fn,
        to_neg_1_1=to_neg_1_1,
        per_channel=False,
    )
    hrms, lrms, pan = map(norm_fn_partial, [hrms, lrms, pan])

    # if has latents
    if "latents" in sample or "hrms_latent" in sample:
        if latent_ext == "npz":
            sample_dict = sample["latents"]
            hrms_latent, lrms_latent, pan_latent = map(
                to_tensor,
                [
                    sample_dict["hrms_latent"],
                    sample_dict["lrms_latent"],
                    sample_dict["pan_latent"],
                ],
            )
        elif latent_ext == "safetensors":
            sample_dict = sample["latents"]
            hrms_latent = sample_dict["hrms_latent"].float()
            lrms_latent = sample_dict["lrms_latent"].float()
            pan_latent = sample_dict["pan_latent"].float()
        else:
            sample_dict = sample["latents"]
            hrms_latent, lrms_latent, pan_latent = map(
                to_tensor,
                [
                    sample["hrms_latent"],
                    sample["lrms_latent"],
                    sample["pan_latent"],
                ],
            )

        sample_new = {
            "hrms": hrms,
            "lrms": lrms,
            "pan": pan,
            "hrms_latent": hrms_latent,
            "lrms_latent": lrms_latent,
            "pan_latent": pan_latent,
        }

    else:
        sample_new = {"hrms": hrms, "lrms": lrms, "pan": pan}
    sample.update(sample_new)

    return sample


def satellite_name_add(sample: dict):
    url = sample["__url__"]
    file_stem = Path(url).stem
    sat_name = file_stem.split("_")[-2]  # e.g., Pansharpening_WV3_train
    assert sat_name in SUPPORTED_SATELLITES, (
        f"Satellite name {sat_name} not in supported list {SUPPORTED_SATELLITES}"
    )
    sample["satellite"] = sat_name
    return sample


def satellite_name_add_collated(sample: dict):
    satellite_names = []
    for url in sample.get("__url__", sample.get("__lrms_shard__", [])):
        file_stem = Path(url).stem
        sat_name = file_stem.split("_")[-2]  # e.g., Pansharpening_WV3_train
        assert sat_name in SUPPORTED_SATELLITES, (
            f"Satellite name {sat_name} not in supported list {SUPPORTED_SATELLITES}"
        )
        satellite_names.append(sat_name)

    # assertions
    _lrms_for_shape = sample.get("lrms", sample.get("img", None))
    assert _lrms_for_shape is not None, "No lrms or img in the sample"
    if len(satellite_names) != _lrms_for_shape.shape[0]:
        raise ValueError(
            f"Batch size {sample['lrms'].shape[0]} does not match satellite names "
            f"length {len(satellite_names)}"
        )

    sample["satellite"] = satellite_names
    return sample


@function_config_to_basic_types
def get_pansharp_lantent_dataloader(
    wds_paths: str | list[str],
    batch_size: int,
    num_workers: int,
    shuffle_size: int = 100,
    to_neg_1_1: bool = True,
    resample: bool = True,
    prefetch_factor: int = 6,
    latent_ext: Literal["safetensors", "npy", "npz"] = "safetensors",
    shardshuffle: bool = False,
    add_satellite_name=False,
):
    """
    Compact with script 'scripts/data_prepare/pansharpening_data_simulation.py'

    When using as the pansharpening trainer dataloader, assume that we aleady normalize the images.
    """

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
        empty_check=False,
    )

    dataset = dataset.decode(
        wds.handle_extension("tif tiff", tiff_decode_io),
        wds.handle_extension("mat", mat_decode_io),
        wds.handle_extension("jpg png jpeg", img_decode_io),
        wds.handle_extension("safetensors", safetensors_decode_io),
        wds.handle_extension("npz", npz_decode_io),
        "torch",
    )

    # * --- dict mapper of images and latents ---

    dataset = dataset.map(remove_extension)
    dataset = dataset.map(
        partial(img_dict_mapper_with_ext, to_neg_1_1=to_neg_1_1, latent_ext=latent_ext)
    )
    if add_satellite_name:
        dataset = dataset.map(satellite_name_add)

    # since we do not use any transforms, the batch and unbatch is useless.

    dataloader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=None if num_workers == 0 else prefetch_factor,
        drop_last=False,
    )

    # dataloader = dataloader.with_length(10_000)  # 10k pairs of the dataloader

    log_print(f"[Pansharpening Dataset]: constructed the dataloader")

    return dataset, dataloader


class CachedDataset(Dataset):
    def __init__(self, dataloader):
        self._dl = dataloader
        self.cached_samples: dict[str, Any] = self.get_loader_cached(dataloader)

    def get_loader_cached(self, outter_dl, ignore_meta=True):
        # if isinstance(outter_dl, torch.utils.data.DataLoader):
        #     assert outter_dl.batch_size == 1, "Only support batch_size=1 for caching"
        # elif isinstance(outter_dl, wds.WebLoader):
        #     assert (
        #         outter_dl.pipeline[0].batch_size == 1
        #     ), "Only support batch_size=1 for caching"

        cached_samples = defaultdict(list)
        for sample in tqdm(outter_dl, desc="Caching samples..."):
            for k, v in sample.items():
                if ignore_meta and k.startswith("__"):
                    continue

                for vi in v:
                    cached_samples[k].append(vi)

        return cached_samples

    def __len__(self):
        return len(self.cached_samples[self.cached_keys[0]])

    @property
    def cached_keys(self):
        return list(self.cached_samples.keys())

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.cached_samples.items()}


class H5Dataset(Dataset):
    def __init__(
        self, h5file: h5py.File, group_names: list[str] | None = None, transform=None
    ):
        self.h5file = h5file
        self.grp_names = group_names if group_names is not None else list(h5file.keys())
        if transform is not None:
            self.transform = transform
        else:
            self.transform = Compose([lambda x: torch.as_tensor(x)])
        self._h5dataset_check()

        log_print(
            f"[H5Dataset] Subgroups in the H5 file: {self.grp_names}, has n_sample={len(self)}"
        )

    def _h5dataset_check(self):
        sub_grp_keys = self.h5file[self.grp_names[0]].keys()
        for k in self.grp_names[1:]:
            assert self.h5file[k].keys() == sub_grp_keys, (
                "Inconsistent sub-groups in H5 file"
            )

        self.grp_keys = list(sub_grp_keys)

    def __len__(self):
        return len(self.grp_keys)

    def __getitem__(self, idx):
        grp_key = self.grp_keys[idx]
        data = {}
        for grp_name in self.grp_names:
            d = self.h5file[grp_name][grp_key][:]
            if self.transform is not None:
                d = self.transform(d)
            data[grp_name] = d
        return data


# * --- Wids dataloader --- #


@function_config_to_basic_types
def get_wids_mat_full_resolution_dataloder(
    wds_paths: dict[str, str],
    batch_size: int,
    num_workers: int = 0,
    shuffle_size: int = 0,
    to_neg_1_1: bool = False,
    add_satellite_name: bool = True,
    **__discarded_kwargs,
):
    """
    Left tokenizer encoding in the trainer (on-the-fly encoding).
    full-resolution pansharpening only.
    """
    img_decode_fn = partial(
        wids_image_decode,
        to_neg_1_1=to_neg_1_1,
        mat_is_single_img=True,
        process_img_keys="ALL",
    )
    pan_decode_fn = partial(
        wids_image_decode,
        to_neg_1_1=to_neg_1_1,
        mat_is_single_img=True,
        process_img_keys="ALL",
        permute=False,
    )
    codecs = {"lrms": img_decode_fn, "pan": pan_decode_fn}

    # compactibility with pansharpening simulator
    codecs["img"] = img_decode_fn

    dataset, dataloader = MultimodalityDataloader.create_loader(
        wds_paths,  # type: ignore
        to_neg_1_1=to_neg_1_1,
        permute=True,
        return_classed_dict=True,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle_size=shuffle_size,
        codecs=codecs,
        # getitem_type="extracted",
    )
    if add_satellite_name:
        dataloader = dataloader.map(satellite_name_add_collated)

    return dataset, dataloader


# * --- Test --- #


def test_wds_reduced_resolution_loading():
    ds, dl = get_pansharp_lantent_dataloader(
        "data/WorldView3/pansharpening_reduced/Pansharping_WV3_train.tar",
        batch_size=2,
        shuffle_size=0,
        num_workers=0,
        add_satellite_name=True,
    )
    for sample in dl:
        print(f"Satellite: {sample['satellite']}: {sample.keys()}")
        break


def test_mat_full_resolution_loading():
    wds_paths = {
        "lrms": "data/WorldView4/pansharpening_full/MS_shardindex.json",
        "pan": "data/WorldView4/pansharpening_full/PAN_shardindex.json",
    }
    dataset, dataloader = get_wids_mat_full_resolution_dataloder(
        wds_paths, 2, 0, to_neg_1_1=True
    )
    for sample in dataloader:
        print(sample.keys())


def test_cache_dataset():
    ds, dl = get_pansharp_lantent_dataloader(
        "data_local/Pansharpening_WV2_train.tar",
        batch_size=8,
        num_workers=1,
        shuffle_size=0,
        add_satellite_name=True,
        resample=False,
    )
    c_ds = CachedDataset(dl)
    c_dl = torch.utils.data.DataLoader(c_ds, batch_size=2, num_workers=0)
    for sample in tqdm(c_dl):
        print(sample.keys())
        # break


def test_h5_dataset():
    h5_path = "data/WorldView3/pansharpening_reduced/tmp.h5"
    h5file = h5py.File(h5_path, "r")
    ds = H5Dataset(h5file, group_names=["lrms", "pan"])
    print(f"Length of H5Dataset: {len(ds)}")
    for i in range(len(ds)):
        sample = ds[i]
        print(sample.keys())
        print(sample["lrms"].shape, sample["pan"].shape)


if __name__ == "__main__":
    # test_mat_full_resolution_loading()
    # test_wds_reduced_resolution_loading()
    # test_cache_dataset()
    test_h5_dataset()

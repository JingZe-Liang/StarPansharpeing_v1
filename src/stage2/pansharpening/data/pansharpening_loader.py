from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Literal, Mapping

import accelerate
import h5py
import torch
import webdataset as wds
from kornia.augmentation import (
    AugmentationSequential,
    RandomAffine,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomVerticalFlip,
)
from timm.layers.helpers import to_2tuple
from torch.utils.data import Dataset
from torchvision.transforms import Compose
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
from src.data.utils import (
    flatten_sample_subdict,
    permute_img_to_chw,
    remove_extension,
    to_tensor,
)
from src.data.utils import norm_img_ as norm_img_fn
from src.utilities.config_utils import function_config_to_basic_types
from src.utilities.logging import log_print

SUPPORTED_SATELLITES = {"WV2", "WV3", "WV4", "QB", "IKONOS"}


__all__ = [
    "get_pansharp_wds_dataloader",
    "get_wids_mat_full_resolution_dataloder",
    "CachedDataset",
    "H5Dataset",
]

# * --- Webdataset dataloader --- #


def get_default_pan_pair_transform(p=0.5, img_size=256):
    transform = AugmentationSequential(
        RandomHorizontalFlip(p=p),
        RandomVerticalFlip(p=p),
        RandomAffine(degrees=30, translate=(0.0, 0.1), scale=(0.8, 1.2), p=p),
        RandomResizedCrop(to_2tuple(img_size), scale=(0.8, 1.2), ratio=(0.6, 1.0), p=p),
        keepdim=True,
        same_on_batch=False,
        data_keys=["image"],
        random_apply=1,
    )

    def _forward_pipe(sample: dict):
        if p <= 0:
            return sample

        lrms, pan = sample["lrms"], sample["pan"]
        lrms = transform(lrms)
        _params = transform._params
        pan = transform(pan, params=_params)
        if "hrms" in sample:
            hrms = sample["hrms"]
            hrms = transform(hrms, params=_params)
            sample["hrms"] = hrms

        # apply changes
        sample.update({"lrms": lrms, "pan": pan})
        return sample

    return _forward_pipe


def norm_to_neg_1_1(sample: dict) -> dict:
    keys = ["hrms", "lrms", "pan"]
    for k in keys:
        if k in sample:
            sample[k] = sample[k] * 2 - 1
    return sample


def img_dict_mapper_with_ext(
    sample: dict,
    to_neg_1_1: bool = True,
    latent_ext="safetensors",
    permute: bool | float = False,
    norm: bool | float = True,
    upsample_lrms: bool = True,
    norm_info_add_in_sample: bool = True,
):
    # keys: ['hrms', 'lrms', 'pan', 'ms', 'hrms_latent', 'lrms_latent', 'pan_latent', 'ms_latent']
    permutatio_fn = permute_img_to_chw if permute else lambda x: x
    hrms = permutatio_fn(to_tensor(sample["hrms"])) if "hrms" in sample else None
    lrms = permutatio_fn(to_tensor(sample["lrms"]))
    pan = permutatio_fn(to_tensor(sample["pan"]))  # h, w
    if pan.ndim == 2:
        pan = pan[None]  # add channel dim

    # normalize
    if norm is True:
        norm_fn_partial = partial(
            norm_img_fn,
            to_neg_1_1=False,
            per_channel=False,
        )
        if hrms is not None:
            hrms, hrms_min, hrms_max = norm_fn_partial(hrms)
        (lrms, lrms_min, lrms_max), (pan, pan_min, pan_max) = map(
            norm_fn_partial, [lrms, pan]
        )
        if norm_info_add_in_sample:
            if hrms is not None:
                sample["hrms_min"] = hrms_min
                sample["hrms_max"] = hrms_max
            sample["lrms_min"] = lrms_min
            sample["lrms_max"] = lrms_max
            sample["pan_min"] = pan_min
            sample["pan_max"] = pan_max
    elif isinstance(norm, float):
        # Aligned with Pan-Collection behavior
        # WV3: 2047; WV2: 2047; QB: 2047; IKONOS: 1023
        if hrms is not None:
            hrms = (hrms.float() / norm).clip(0, 1)
        lrms = (lrms.float() / norm).clip(0, 1)
        pan = (pan.float() / norm).clip(0, 1)
        if norm_info_add_in_sample:
            if hrms is not None:
                sample["hrms_min"] = 0.0
                sample["hrms_max"] = float(norm)
            sample["lrms_min"] = 0.0
            sample["lrms_max"] = float(norm)
            sample["pan_min"] = 0.0
            sample["pan_max"] = float(norm)

    # upsample lrms to pan size
    if lrms.shape[-2:] != pan.shape[-2:] and upsample_lrms:
        # for full-res dataset, give lrms is [h, w] and pan is [H, W],
        # where H = h * ratio
        if lrms.ndim == 3:
            lrms.unsqueeze_(0)
        lrms = torch.nn.functional.interpolate(
            lrms, size=pan.shape[-2:], mode="bilinear", align_corners=False
        )
        lrms.squeeze_(0)

    # normalize to [-1, 1]
    if to_neg_1_1:
        hrms = hrms * 2 - 1 if hrms is not None else None
        lrms = lrms * 2 - 1
        pan = pan * 2 - 1

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
        sample_new = (
            {"hrms": hrms, "lrms": lrms, "pan": pan}
            if hrms is not None
            else {"lrms": lrms, "pan": pan}
        )
    sample.update(sample_new)

    return sample


def satellite_name_add(sample: dict):
    url = sample["__url__"]
    file_stem = Path(url).stem
    sat_name = file_stem.split("_")[-2]  # e.g., Pansharpening_WV3_train
    assert sat_name in SUPPORTED_SATELLITES, (
        f"Satellite name {sat_name} not in supported list {SUPPORTED_SATELLITES}, "
        f"but got from {url}"
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


def get_pansharp_wds_dataloader(
    wds_paths: str | list[str],
    batch_size: int,
    num_workers: int,
    shuffle_size: int = 100,
    to_neg_1_1: bool = True,
    resample: bool = True,
    prefetch_factor: int = 6,
    norm: bool | float = False,
    latent_ext: Literal["safetensors", "npy", "npz"] = "safetensors",
    shardshuffle: bool = False,
    drop_last: bool = False,
    add_satellite_name=False,
    transform_prob: float = 0.0,
    img_size=256,
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

    dataset = dataset.map(flatten_sample_subdict)
    dataset = dataset.map(remove_extension)
    dataset = dataset.map(
        partial(
            img_dict_mapper_with_ext,
            to_neg_1_1=to_neg_1_1,
            latent_ext=latent_ext,
            norm=norm if not isinstance(norm, bool) else float(norm),
        )
    )
    dataset = dataset.map(
        get_default_pan_pair_transform(p=transform_prob, img_size=img_size)
    )
    if add_satellite_name:
        dataset = dataset.map(satellite_name_add)
    if shuffle_size > 0:
        dataset = dataset.shuffle(shuffle_size)

    # since we do not use any transforms, the batch and unbatch is useless.

    dataloader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=None if num_workers == 0 else prefetch_factor,
        drop_last=drop_last,
        pin_memory=True,
    )

    log_print(f"[Pansharpening Dataset]: constructed the dataloader")

    return dataset, dataloader


class CachedDataset(Dataset):
    def __init__(self, dataloader):
        self._dl = dataloader
        self.cached_samples: dict[str, Any] = self.get_loader_cached(dataloader)

    def get_loader_cached(self, outter_dl, ignore_meta=True):
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
    """
    Dataset class for loading hyperspectral pansharpening data from HDF5 files.

    This dataset supports three different HDF5 storage formats:
    1. Per-sample storage: Each sample is stored as a separate subgroup
    2. Stacked storage: All samples are stacked in a single array, compatible with old PanCollection format.
    3. Multiple files: List of HDF5 files with identical group names, stacked along batch dimension.

    Parameters
    ----------
    h5file : h5py.File | str | list[str]
        HDF5 file object, file path, or list of file paths to HDF5 files containing the dataset.
        If list[str] is provided, datasets from all files will be stacked along batch dimension.
    group_names : list[str] | None, optional
        List of group names to load from the HDF5 file(s). If None, loads all groups from first file.
    transform : callable, optional
        Transform function to apply to the loaded data. If None, default normalization
        to float32 tensor divided by norm value is applied.
    norm : float, default=2047.0
        Normalization factor for converting data to float32 tensors.
    mapping_names : Mapping[str, str], default={"lms": "lrms", "gt": "hrms"}
        Mapping of group names to output keys for the returned samples.

    Attributes
    ----------
    h5file : h5py.File | list[h5py.File]
        The opened HDF5 file object(s).
    grp_names : list[str]
        List of group names to load data from.
    mapping_names : Mapping[str, str]
        Mapping from group names to output keys.
    transform : callable
        Transform function applied to loaded data.
    _is_persample_h5 : bool
        Flag indicating if data is stored per-sample or stacked.
    _is_multi_file : bool
        Flag indicating if multiple files are being used.
    grp_keys : list[str]
        List of sample keys for per-sample storage format.
    _file_lengths : list[int]
        List of sample counts for each file in multi-file mode.

    Raises
    ------
    AssertionError
        If inconsistent shapes or keys are found across different groups or files.

    Note
    ----
    The dataset automatically detects the storage format during initialization
    and handles per-sample, stacked, and multi-file loading efficiently.
    When using multiple files, all files must have identical group names and shapes.
    """

    def __init__(
        self,
        h5file: h5py.File | str | list[str],
        group_names: list[str] | None = None,
        transform=None,
        norm: float = 2047.0,
        mapping_names: Mapping[str, str] = {"lms": "lrms", "gt": "hrms"},
        to_neg_1_1: bool = True,
    ):
        # Handle different input types
        if isinstance(h5file, h5py.File):
            self.h5file = h5file
            self._is_multi_file = False
        elif isinstance(h5file, list):
            self.h5file = [h5py.File(f) for f in h5file]
            self._is_multi_file = True
        else:
            self.h5file = h5py.File(h5file)
            self._is_multi_file = False

        # Set group names based on input type
        if self._is_multi_file:
            if group_names is None:
                self.grp_names = list(self.h5file[0].keys())
            else:
                self.grp_names = group_names
        else:
            if group_names is None:
                self.grp_names = list(self.h5file.keys())
            else:
                self.grp_names = group_names
        self.mapping_names = mapping_names
        if transform is not None:
            self.transform = transform
        else:
            self.transform = Compose(
                [lambda x: torch.as_tensor(x, dtype=torch.float32).clip_(min=0) / norm]
            )
            if to_neg_1_1:
                self.transform.transforms.append(lambda x: x.mul_(2).sub_(1))
        self._h5dataset_check()

        log_print(
            f"[H5Dataset] Subgroups in the H5 file: <green>{self.grp_names}</>, "
            f"has <green>n_sample={len(self)}</>"
        )

    def _h5dataset_check(self):
        if self._is_multi_file:
            # For multiple files, always treat as stacked array dataset
            self._is_persample_h5 = False
            # Validate that all files have the same group names and shapes
            first_file = self.h5file[0]
            for file_idx, h5file in enumerate(self.h5file[1:], 1):
                assert set(h5file.keys()) == set(first_file.keys()), (
                    f"File {file_idx} has different group names than first file"
                )

                for grp_name in self.grp_names:
                    shape_first = first_file[grp_name].shape
                    shape_current = h5file[grp_name].shape
                    assert shape_current[1:] == shape_first[1:], (
                        f"File {file_idx}, group {grp_name} has inconsistent shape: {shape_current} vs {shape_first}"
                    )

            # Store file lengths for indexing
            self._file_lengths = [
                h5file[self.grp_names[0]].shape[0] for h5file in self.h5file
            ]
        else:
            # Single file handling
            sub_grp_ds = self.h5file[self.grp_names[0]]
            if hasattr(sub_grp_ds, "keys"):
                sub_grp_keys = (
                    sub_grp_ds.keys()
                )  # the samples are stored in sub-groups, not in one array
                for k in self.grp_names[1:]:
                    assert self.h5file[k].keys() == sub_grp_keys, (
                        "Inconsistent sub-groups in H5 file"
                    )
                self.grp_keys = list(sub_grp_keys)
                self._is_persample_h5 = True
            else:
                assert hasattr(sub_grp_ds, "shape"), (
                    "H5 group is not a dataset or group of datasets"
                )
                ds_shape = sub_grp_ds.shape
                for k in self.grp_names[1:]:
                    shape_k = self.h5file[k].shape
                    assert shape_k[0] == ds_shape[0], (
                        f"Inconsistent shape of sub-groups in H5 file, should be {ds_shape} but got {shape_k}"
                    )
                self._is_persample_h5 = False  # the samples are all stacked in one array, as in PanCollection

    def __len__(self):
        if self._is_multi_file:
            return sum(self._file_lengths)
        elif self._is_persample_h5:
            return len(self.grp_keys)
        else:
            return self.h5file[self.grp_names[0]].shape[0]

    def _per_sample_get_item(self, idx):
        grp_key = self.grp_keys[idx]
        data = {}
        for grp_name in self.grp_names:
            d = self.h5file[grp_name][grp_key][:]
            if self.transform is not None:
                d = self.transform(d)
            data[grp_name] = d
        return data

    def _stacked_data_get_item(self, idx):
        data = {}
        for grp_name in self.grp_names:
            d = self.h5file[grp_name][idx][:]
            if self.transform is not None:
                d = self.transform(d)
            grp_name = self.mapping_names.get(grp_name, grp_name)
            data[grp_name] = d
        return data

    def __getitem__(self, index) -> Any:
        if self._is_multi_file:
            return self._multi_file_get_item(index)
        elif self._is_persample_h5:
            return self._per_sample_get_item(index)
        else:
            return self._stacked_data_get_item(index)

    def _multi_file_get_item(self, index):
        """Get item from multiple files by finding the correct file and local index."""
        # Find which file contains this index
        cumulative_length = 0
        for file_idx, file_length in enumerate(self._file_lengths):
            if index < cumulative_length + file_length:
                local_index = index - cumulative_length
                h5file = self.h5file[file_idx]
                break
            cumulative_length += file_length
        else:
            raise IndexError(
                f"Index {index} out of range for total length {cumulative_length}"
            )

        # Load data from the specific file
        data = {}
        for grp_name in self.grp_names:
            d = h5file[grp_name][local_index][:]
            if self.transform is not None:
                d = self.transform(d)
            output_name = self.mapping_names.get(grp_name, grp_name)
            data[output_name] = d
        return data

    def close(self):
        """Close all HDF5 file handles."""
        if self._is_multi_file and isinstance(self.h5file, list):
            for h5file in self.h5file:
                h5file.close()
        elif hasattr(self.h5file, "close"):
            self.h5file.close()

    def __del__(self):
        """Ensure files are closed when object is destroyed."""
        if hasattr(self, "h5file"):
            self.close()


# * --- Wids dataloader --- #


def get_wids_mat_full_resolution_dataloder(
    wds_paths: dict[str, str],
    batch_size: int,
    num_workers: int = 0,
    shuffle_size: int = 0,
    to_neg_1_1: bool = False,
    add_satellite_name: bool = False,
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


# *==============================================================
# * Interface
# *==============================================================


@function_config_to_basic_types
def get_pansharpening_dataloader_interface(
    ds_type: Literal["wds", "wids_full", "h5"],
    func_kwargs: dict,
    loader_kwargs: dict | None = None,
):
    if ds_type == "wds":
        return get_pansharp_wds_dataloader(**func_kwargs)
    elif ds_type == "wids_full":
        return get_wids_mat_full_resolution_dataloder(**func_kwargs)
    elif ds_type == "h5":
        loader_kwargs = loader_kwargs or {}
        ds = H5Dataset(**func_kwargs)
        dl = torch.utils.data.DataLoader(ds, **loader_kwargs)
        return ds, dl
    else:
        raise ValueError(f"Unknown dataset type: {ds_type}")


# * --- Test --- #


def test_wds_reduced_resolution_loading():
    # IKONOS: 1193.00
    # QB: 634.00
    # WV3: 1293.00
    # WV2: 1208.00
    # WV4: 1516.00

    ds, dl = get_pansharp_wds_dataloader(
        "data/Downstreams/PanCollectionV2/WV3/pansharpening_reduced/Pansharpening_WV3.tar",
        batch_size=1,
        shuffle_size=0,
        num_workers=0,
        add_satellite_name=False,
        to_neg_1_1=False,
        resample=False,
        norm=1426.0,
        transform_prob=1.0,
        img_size=256,
    )
    max_v = 0.0
    for sample in dl:
        lrms, pan, hrms = sample["lrms"], sample["pan"], sample.get("hrms", None)
        print(sample.keys())
        print("lrms value range:", sample["lrms"].min(), sample["lrms"].max())
        m = sample["lrms"].max().item()
        max_v = max(m, max_v)
        print("max value now: ", max_v)
        print(
            "lrms mean: ",
            sample["lrms"].mean().item(),
            "hrms mean: ",
            sample["hrms"].mean().item(),
            "pan mean: ",
            sample["pan"].mean().item(),
        )
        print("------------")


def test_full_resolution_wds_loading():
    path = "data/Downstreams/PanCollectionV2/WV3/pansharpening_full/Pansharpening_FullResolution_WV3_val.npz.tar"
    ds, dl = get_pansharp_wds_dataloader(
        path,
        batch_size=4,
        num_workers=1,
        shuffle_size=0,
        add_satellite_name=True,
        resample=False,
        to_neg_1_1=False,
        norm=False,
    )
    for sample in tqdm(dl, total=100):
        print(f"Satellite: {sample['satellite']}: {sample.keys()}")
        print(sample["lrms"].shape, sample["pan"].shape)
        # range values
        print("lrms value range: ", sample["lrms"].min(), sample["lrms"].max())
        print("pan value range: ", sample["pan"].min(), sample["pan"].max())


def test_mat_full_resolution_loading():
    wds_paths = {
        "lrms": "data/WorldView4/pansharpening_full/MS_shardindex.json",
        "pan": "data/WorldView4/pansharpening_full/PAN_shardindex.json",
    }
    dataset, dataloader = get_wids_mat_full_resolution_dataloder(
        wds_paths, 2, 0, to_neg_1_1=True
    )
    for sample in dataloader:
        print(
            sample.keys(),
            "lrms value range:",
            sample["lrms"].min(),
            sample["lrms"].max(),
        )


def test_cache_dataset():
    ds, dl = get_pansharp_wds_dataloader(
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
    # h5_path = "data/WorldView3/pansharpening_reduced/tmp.h5"
    h5_path = (
        "data/Downstreams/PanCollectionV1/qb/reduced_examples/test_qb_multiExm1.h5"
    )
    ds = H5Dataset(h5_path)
    print(f"Length of H5Dataset: {len(ds)}")
    dl = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=0)
    from accelerate import Accelerator

    accelerator = Accelerator()
    dl = accelerator.prepare(dl)
    print(type(dl))
    for sample in dl:
        print(sample.keys())
        print(sample["lrms"].shape, sample["pan"].shape, sample["hrms"].shape)


def test_multi_file_h5_dataset():
    """
    Test the multi-file H5Dataset functionality with WV3 training and validation files.

    This test verifies that multiple HDF5 files can be loaded and accessed as a single
    continuous dataset with proper indexing and data stacking.
    """
    import h5py

    # Define file paths
    train_path = "data/Downstreams/PanCollectionV1/wv3/training_wv3/train_wv3.h5"
    valid_path = "data/Downstreams/PanCollectionV1/wv3/training_wv3/valid_wv3.h5"
    file_list = [train_path, valid_path]

    print("Testing multi-file H5Dataset...")
    print(f"Files: {file_list}")

    # Test individual files first
    print("\n--- Individual file info ---")
    for i, file_path in enumerate(file_list):
        with h5py.File(file_path, "r") as h5file:
            print(f"File {i}: {file_path}")
            print(f"  Groups: {list(h5file.keys())}")
            if "lms" in h5file:
                print(f"  LMS shape: {h5file['lms'].shape}")
            if "pan" in h5file:
                print(f"  PAN shape: {h5file['pan'].shape}")
            if "hrms" in h5file:
                print(f"  HRMS shape: {h5file['hrms'].shape}")

    # Test multi-file dataset
    print("\n--- Multi-file dataset test ---")
    try:
        multi_ds = H5Dataset(file_list)
        print(f"Multi-file dataset length: {len(multi_ds)}")
        print(f"Is multi-file mode: {multi_ds._is_multi_file}")
        print(f"File lengths: {multi_ds._file_lengths}")
        print(f"Group names: {multi_ds.grp_names}")

        # Test accessing samples across file boundaries
        total_length = len(multi_ds)
        test_indices = [
            0,
            total_length // 4,
            total_length // 2,
            3 * total_length // 4,
            total_length - 1,
        ]

        print(f"\nTesting {len(test_indices)} samples across file boundaries...")
        for i, idx in enumerate(test_indices):
            sample = multi_ds[idx]
            print(f"Sample {i} (index {idx}):")
            print(f"  Keys: {list(sample.keys())}")
            for key, value in sample.items():
                print(
                    f"  {key}: {value.shape}, dtype: {value.dtype}, range: [{value.min():.3f}, {value.max():.3f}]"
                )

        # Verify continuity by checking if we can access all indices
        print(
            f"\nTesting access to all {total_length} samples (this may take a while)..."
        )
        error_count = 0
        for idx in tqdm(
            range(min(total_length, 100))
        ):  # Test first 100 samples to avoid long runtime
            try:
                sample = multi_ds[idx]
                # Basic validation
                for key, value in sample.items():
                    if (
                        value.ndim == 0
                        or torch.isnan(value).any()
                        or torch.isinf(value).any()
                    ):
                        print(f"Warning: Invalid data at index {idx}, key {key}")
                        error_count += 1
            except Exception as e:
                print(f"Error accessing index {idx}: {e}")
                error_count += 1

        if error_count == 0:
            print("✓ All test samples accessed successfully!")
        else:
            print(f"✗ Found {error_count} errors during testing")

        # Close the dataset
        multi_ds.close()
        print("✓ Multi-file dataset test completed successfully!")

    except Exception as e:
        print(f"✗ Error in multi-file test: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # test_mat_full_resolution_loading()
    # test_wds_reduced_resolution_loading()
    # test_cache_dataset()
    test_h5_dataset()
    # test_multi_file_h5_dataset()
    # test_full_resolution_wds_loading()

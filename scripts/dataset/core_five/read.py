import multiprocessing as mp
import pickle
from functools import partial
from pathlib import Path

import datatree as dt
import litdata as ld
import numpy as np
import pandas as pd
import xarray as xr  # datatree 依赖于 xarray
from easydict import EasyDict as edict
from loguru import logger
from natsort import natsorted
from tqdm import tqdm

from src.data.codecs import img_decode_io, npy_codec_io, npy_decode_io, rgb_codec_io, tiff_codec_io, tiff_decode_io
from src.utilities.logging import set_logger_file

set_logger_file("data2/core-five/log.log", add_time=False, mode="w")


def read_nc_file(file: str | Path) -> dt.DataTree | None:
    data_tree: dt.DataTree | None = None
    try:
        data_tree = dt.open_datatree(file, engine="netcdf4")

    except Exception as e:
        logger.error(f"使用 datatree 读取文件时发生错误: {e}")

    return data_tree


def extract_data_from_datatree(data_tree: dt.DataTree, group_name: str):
    try:
        # 获取指定组中的数据
        group_data = data_tree[group_name].to_dataset()
        return group_data
    except KeyError:
        logger.error(f"组 '{group_name}' 不存在于 DataTree 中。")
        return None


def extract_time_and_pos_data(dataset, dataset_name: str) -> tuple:
    """
    Extract time and position data from dataset with proper error handling.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset containing time and position data
    dataset_name : str
        Name of the dataset for logging purposes

    Returns
    -------
    tuple
        A tuple containing (time_data, pos_data) where time_data is properly
        formatted datetime data or pd.NaT, and pos_data is position data or pd.NaT
    """
    try:
        # Extract time data and ensure it's datetime
        if hasattr(dataset, "time"):
            time_data = pd.to_datetime(dataset.time)
        else:
            logger.warning(f"No time data found in {dataset_name}")
            time_data = pd.NaT

    except Exception as e:
        logger.warning(f"Error extracting time data from {dataset_name}: {e}")
        time_data = pd.NaT

    try:
        # Extract position data
        if hasattr(dataset, "x") and hasattr(dataset, "y"):
            pos_data = [dataset.x.data, dataset.y.data]
        else:
            logger.warning(f"No position data found in {dataset_name}")
            pos_data = pd.NaT

    except Exception as e:
        logger.warning(f"Error extracting position data from {dataset_name}: {e}")
        pos_data = pd.NaT

    return time_data, pos_data


def extract_core_five_groups(data_tree, path, mod: str = "all"):
    groups_to_extract = ["s2", "s1", "modis", "landsat", "hr"]
    s2_data = s1_data = modis_data = landsat_data = hr_data = None

    if mod == "all" or mod == "s2":
        # logger.debug("reading s2 data")
        s2_bands_list = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
        ds_s2 = data_tree.s2
        s2_data_arrays = [ds_s2[b] for b in s2_bands_list if b in ds_s2]
        s2_data_arrays = np.stack(s2_data_arrays, axis=1)  # T,C,H,W
        s2_data_arrays = np.nan_to_num(s2_data_arrays, nan=0.0).clip(min=0, max=65535).astype(np.uint16)
        s2_times, s2_pos = extract_time_and_pos_data(ds_s2, "s2")
        # data = data / 10000
        s2_data = dict(img=s2_data_arrays, time=s2_times, pos=s2_pos)

    if mod == "all" or mod == "s1":
        # logger.debug("Reading s1 data")
        ds_s1 = data_tree.s1
        vv, vh = ds_s1.vv.values, ds_s1.vh.values
        s1_data_arrays = np.stack([vv, vh], axis=1)
        s1_data_arrays = np.nan_to_num(s1_data_arrays, nan=0.0).astype(np.uint16)
        # data = 10 * log10(data + 1e-6) # 转换为 dB
        s1_times, s1_pos = extract_time_and_pos_data(ds_s1, "s1")
        s1_data = dict(img=s1_data_arrays, time=s1_times, pos=s1_pos)

    # Modis
    if mod == "all" or mod == "modis":
        # logger.debug("Reading modis data")
        modis_bands = [
            "sur_refl_b01",
            "sur_refl_b02",
            "sur_refl_b03",
            "sur_refl_b04",
            "sur_refl_b05",
            "sur_refl_b06",
            "sur_refl_b07",
        ]
        ds_modis = data_tree.modis.fillna(0).clip(0, 65535).astype(np.uint16)
        modis_data_arrays = [ds_modis[b].values for b in modis_bands if b in ds_modis]
        modis_data_arrays = np.stack(modis_data_arrays, axis=1)  # T,C,H,W
        modis_times, modis_pos = extract_time_and_pos_data(ds_modis, "modis")
        modis_data = dict(img=modis_data_arrays, time=modis_times, pos=modis_pos)

    # Landsat
    if mod == "all" or mod == "landsat":
        # logger.debug("Reading landsat data")
        landsat_bands = ["blue", "green", "red", "nir08", "swir16", "swir22", "lwir11"]
        ds_landsat = data_tree.landsat.fillna(0).clip(0, 65535).astype(np.uint16)
        # data x 0.0000275 - 0.2
        landsat_data_arrays = [ds_landsat[b].values for b in landsat_bands]
        landsat_data_arrays = np.stack(landsat_data_arrays, axis=1)  # T,C,H,W
        landsat_times, landsat_pos = extract_time_and_pos_data(ds_landsat, "landsat")
        landsat_data = dict(img=landsat_data_arrays, time=landsat_times, pos=landsat_pos)

    # HR
    if mod == "all" or mod == "hr":
        # logger.debug("Reading hr data")
        hr_ds = data_tree["hr/data"].to_dataset().load()  # load in mem.
        ds_hr = hr_ds.data.data.astype(np.uint8)
        _, hr_pos = extract_time_and_pos_data(hr_ds, "hr")
        hr_data = dict(img=ds_hr, pos=hr_pos, meta_data=data_tree["hr/data"].metadata)

    return edict(
        name=Path(path).stem,  # data_tree.attrs["s2id"],
        s2=s2_data,
        s1=s1_data,
        modis=modis_data,
        landsat=landsat_data,
        hr=hr_data,
    )


def dict_data_to_bytes(data: edict):
    s2, s1, modis, landsat, hr = data.s2, data.s1, data.modis, data.landsat, data.hr
    data_bytes = edict()

    tiff_codec_fn = partial(
        tiff_codec_io,
        compression="jpeg2000",
        compression_args={
            "reversible": False,
            "level": 87,
        },
    )
    if s2 is not None:
        s2_bytes_list = [tiff_codec_fn(s2.img[i].transpose(1, 2, 0)) for i in range(s2.img.shape[0])]
        s2_lst_bytes = pickle.dumps(s2_bytes_list, protocol=5)
        data_bytes.s2_img = s2_lst_bytes
        data_bytes.s2_time = s2.time
        data_bytes.s2_pos = s2.pos

    if s1 is not None:
        s1_bytes_list = [tiff_codec_fn(s1.img[i].transpose(1, 2, 0)) for i in range(s1.img.shape[0])]
        s1_lst_bytes = pickle.dumps(s1_bytes_list, protocol=5)
        data_bytes.s1_img = s1_lst_bytes
        data_bytes.s1_time = s1.time
        data_bytes.s1_pos = s1.pos

    if modis is not None:
        modis_bytes_list = [tiff_codec_io(modis.img[i].transpose(1, 2, 0)) for i in range(modis.img.shape[0])]
        modis_lst_bytes = pickle.dumps(modis_bytes_list, protocol=5)
        data_bytes.modis_img = modis_lst_bytes
        data_bytes.modis_time = modis.time
        data_bytes.modis_pos = modis.pos

    if landsat is not None:
        landsat_bytes_list = [tiff_codec_fn(landsat.img[i].transpose(1, 2, 0)) for i in range(landsat.img.shape[0])]
        landsat_lst_bytes = pickle.dumps(landsat_bytes_list, protocol=5)
        data_bytes.landsat_img = landsat_lst_bytes
        data_bytes.landsat_time = landsat.time
        data_bytes.landsat_pos = landsat.pos

    if hr is not None:
        hr_img = hr.img.transpose(1, 2, 0)  # H,W,C
        hr_bytes = rgb_codec_io(hr_img, quality=83)
        data_bytes.hr_img = hr_bytes
        data_bytes.hr_meta_data = hr.meta_data
        data_bytes.hr_pos = hr.pos

    return data_bytes


def litdata_optimize_with_q(
    q,
    base_dir: str = "data2/core-five/src/datatree",
    parquet_path: str = "data2/core-five/meta_infos_processed.parquet",
    mod: str = "all",
):
    def _as_cell_list(value: object) -> list[object]:
        if value is None or value is pd.NaT:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, np.ndarray):
            return value.tolist()  # type: ignore[no-any-return]
        if isinstance(value, xr.DataArray):
            return value.values.tolist()  # type: ignore[no-any-return]
        if isinstance(value, (pd.Index, pd.Series)):
            return value.to_list()
        return [value]

    def _to_timestamp(obj: object) -> pd.Timestamp | None:
        ts = pd.to_datetime(obj, errors="coerce")
        if ts is pd.NaT:
            return None
        return ts  # type: ignore[return-value]

    def _as_time_cell_list(value: object) -> list[str]:
        raw_items = _as_cell_list(value)
        items: list[str] = []
        for item in raw_items:
            ts = _to_timestamp(item)
            if ts is None:
                continue
            items.append(ts.isoformat())
        return items

    def _normalize_meta_info_for_parquet(meta_info: pd.DataFrame) -> pd.DataFrame:
        time_columns = ["s2_time", "s1_time", "modis_time", "landsat_time"]
        pos_columns = ["s2_pos", "s1_pos", "modis_pos", "landsat_pos", "hr_pos"]

        for col in [*time_columns, *pos_columns, "hr_meta_data"]:
            if col in meta_info.columns:
                meta_info[col] = meta_info[col].astype("object")

        for col in time_columns:
            if col in meta_info.columns:
                meta_info[col] = meta_info[col].map(_as_time_cell_list)

        for col in pos_columns:
            if col in meta_info.columns:
                meta_info[col] = meta_info[col].map(_as_cell_list)

        return meta_info

    nc_files = natsorted(list(Path(base_dir).rglob("*.nc")))
    print(f"{len(nc_files)} nc files found in {base_dir}")

    if Path(parquet_path).exists():
        meta_info = pd.read_parquet(parquet_path)
        meta_info = _normalize_meta_info_for_parquet(meta_info)
        processed_name_set = set(meta_info["name"].unique().tolist())
        print(f"{len(processed_name_set)} nc files already processed.")
    else:
        meta_info = pd.DataFrame(
            columns=[  # type: ignore
                "name",
                "s2_time",
                "s2_pos",
                "s1_time",
                "s1_pos",
                "modis_time",
                "modis_pos",
                "landsat_time",
                "landsat_pos",
                "hr_meta_data",
                "hr_pos",
            ]
        )

        meta_info = _normalize_meta_info_for_parquet(meta_info)
        print("Creating new meta info dataframe.")
        processed_name_set = set()

    for i, nc_file in tqdm(enumerate(nc_files), desc="Processing nc files"):
        logger.info(f"Processing {nc_file}")
        if "/".join(Path(nc_file).parts[-2:]) in processed_name_set:
            logger.warning(f"{Path(nc_file).name} already processed, skipping...")
            continue

        data_tree = read_nc_file(nc_file)
        if data_tree is None:
            logger.error(f"Skipping unreadable nc file: {nc_file}")
            continue
        data_grps = extract_core_five_groups(data_tree, nc_file, mod=mod)
        data = dict_data_to_bytes(data_grps)

        # Save litdata
        if not isinstance(q, dict):
            q.put({"__key__": "/".join(Path(nc_file).parts[-2:]), "img": data})
        else:
            key = "/".join(Path(nc_file).parts[-2:])
            q["s1"].put({"__key__": key, "img": data.s1_img})
            q["s2"].put({"__key__": key, "img": data.s2_img})
            q["landsat"].put({"__key__": key, "img": data.landsat_img})
            q["modis"].put({"__key__": key, "img": data.modis_img})
            q["hr"].put({"__key__": key, "img": data.hr_img})

        # Save meta info
        if mod == "all":
            index = len(meta_info)
            meta_info.at[index, "name"] = data.name
            meta_info.at[index, "s2_time"] = _as_time_cell_list(data.s2_time)
            meta_info.at[index, "s2_pos"] = _as_cell_list(data.s2_pos)
            meta_info.at[index, "s1_time"] = _as_time_cell_list(data.s1_time)
            meta_info.at[index, "s1_pos"] = _as_cell_list(data.s1_pos)
            meta_info.at[index, "modis_time"] = _as_time_cell_list(data.modis_time)
            meta_info.at[index, "modis_pos"] = _as_cell_list(data.modis_pos)
            meta_info.at[index, "landsat_time"] = _as_time_cell_list(data.landsat_time)
            meta_info.at[index, "landsat_pos"] = _as_cell_list(data.landsat_pos)
            meta_info.at[index, "hr_meta_data"] = data.hr_meta_data
            meta_info.at[index, "hr_pos"] = _as_cell_list(data.hr_pos)
        else:
            to_time_name = {
                "s1": "s1_time",
                "s2": "s2_time",
                "modis": "modis_time",
                "landsat": "landsat_time",
                "hr": "hr_meta_data",
            }
            col = to_time_name[mod]
            index = len(meta_info)
            meta_info.at[index, "name"] = "/".join(Path(nc_file).parts[-2:])
            if col.endswith("_time"):
                meta_info.at[index, col] = _as_time_cell_list(data[col])
            else:
                meta_info.at[index, col] = data[col]
            meta_info.at[index, f"{mod}_pos"] = _as_cell_list(data[f"{mod}_pos"])

        if i % 30 == 0:
            logger.info("Writing meta info to parquet...")
            meta_info.to_parquet(parquet_path)

    # Close the queue by sending all done
    if isinstance(q, dict):
        for q_i in q.values():
            q_i.put("ALL_DONE")
    else:
        q.put("ALL_DONE")
    logger.success(f"All nc files processed for modality={mod}")


def litdata_optimize(mod="s1"):
    base_dir = "data2/core-five/src/datatree"
    save_path = "data2/core-five/src/litdata"
    parquet_path = "data2/core-five/meta_infos_processed.parquet"
    Path(save_path).mkdir(parents=True, exist_ok=True)

    if mod == "all":
        mod_names = ["s2", "s1", "modis", "landsat", "hr"]
        qs = {name: mp.Queue(maxsize=10) for name in mod_names}
        p = mp.Process(target=litdata_optimize_with_q, args=(qs,))
        p.start()
        for mod in mod_names:
            ld.optimize(
                fn=lambda x: x,
                queue=qs[mod],
                output_dir=f"/Data2/ZihanCao/dataset/core-five/src/litdata/{mod}",
                chunk_bytes="512Mb",
                num_workers=0,
                mode="append",
                start_method="spawn",
            )
        p.join()
    else:
        q = mp.Queue(maxsize=10)
        p = mp.Process(target=litdata_optimize_with_q, args=(q, base_dir, parquet_path, mod))
        p.start()
        print("Starting litdata optimize...")
        ld.optimize(
            fn=lambda x: x,
            queue=q,
            output_dir=f"{save_path}/{mod}",
            chunk_bytes="512Mb",
            num_workers=0,
            mode="append",
            start_method="fork",
        )
        p.join()

    print("Successfully optimized images.")


def load_list_img_bytes(img_bytes_list: bytes, decode_type: str = "tiff"):
    # list of bytes of images
    # pickle load into a list of image bytes
    img_lst = pickle.loads(img_bytes_list)
    ret_img_lst = []
    for img_b in img_lst:
        if decode_type == "tiff":
            img = tiff_decode_io(img_b)
        elif decode_type == "img":
            img = img_decode_io(img_b)
        else:
            raise ValueError(f"Unsupported decode type: {decode_type}")
        ret_img_lst.append(img)
    return ret_img_lst


def load_dict_data(data_bytes: dict):
    s2_seq_bytes = data_bytes["s2_img"]
    s1_seq_bytes = data_bytes["s1_img"]
    modis_seq_bytes = data_bytes["modis_img"]
    landsat_seq_bytes = data_bytes["landsat_img"]
    hr_img = data_bytes["hr_img"]

    s2_seq = load_list_img_bytes(s2_seq_bytes, decode_type="tiff")
    s1_seq = load_list_img_bytes(s1_seq_bytes, decode_type="tiff")
    modis_seq = load_list_img_bytes(modis_seq_bytes, decode_type="tiff")
    landsat_seq = load_list_img_bytes(landsat_seq_bytes, decode_type="tiff")
    hr_img = img_decode_io(hr_img)

    return edict(
        s2=s2_seq,
        s1=s1_seq,
        modis=modis_seq,
        landsat=landsat_seq,
        hr=hr_img,
    )


if __name__ == "__main__":
    # file_path = "/Data2/ZihanCao/dataset/core-five/src/datatree/0c6061/0c605df4.nc"
    # data_tree = read_nc_file(file_path)
    # core_five_groups = extract_core_five_groups(data_tree)
    # print(core_five_groups)

    litdata_optimize("s2")

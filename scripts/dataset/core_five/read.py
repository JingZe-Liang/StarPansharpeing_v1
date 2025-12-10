import pickle
from functools import partial

import datatree as dt
import numpy as np
import xarray as xr  # datatree 依赖于 xarray
from easydict import EasyDict as edict

from src.data.codecs import img_decode_io, npy_codec_io, npy_decode_io, rgb_codec_io, tiff_codec_io, tiff_decode_io


def read_nc_file(file):
    try:
        data_tree = dt.open_datatree(file)

    except Exception as e:
        print(f"使用 datatree 读取文件时发生错误: {e}")

    return data_tree


def extract_data_from_datatree(data_tree: dt.DataTree, group_name: str):
    try:
        # 获取指定组中的数据
        group_data = data_tree[group_name].to_dataset()
        return group_data
    except KeyError:
        print(f"组 '{group_name}' 不存在于 DataTree 中。")
        return None


def extract_core_five_groups(data_tree):
    groups_to_extract = ["s2", "s1", "modis", "landsat", "hr"]

    print("reading s2 data")
    s2_bands_list = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    ds_s2 = data_tree.s2
    s2_data_arrays = [ds_s2[b] for b in s2_bands_list if b in ds_s2]
    s2_data_arrays = np.stack(s2_data_arrays, axis=1)  # T,C,H,W
    s2_data_arrays = np.nan_to_num(s2_data_arrays, nan=0.0).clip(min=0, max=65535).astype(np.uint16)
    s2_times = ds_s2.time.values
    # data = data / 10000
    s2_data = dict(img=s2_data_arrays, time=s2_times)

    print("Reading s1 data")
    ds_s1 = data_tree.s1
    vv, vh = ds_s1.vv.values, ds_s1.vh.values
    s1_data_arrays = np.stack([vv, vh], axis=1)
    s1_data_arrays = np.nan_to_num(s1_data_arrays, nan=0.0).astype(np.uint16)
    # data = 10 * log10(data + 1e-6) # 转换为 dB
    s1_times = ds_s1.time.values
    s1_data = dict(img=s1_data_arrays, time=s1_times)

    # Modis
    print("Reading modis data")
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
    modis_times = ds_modis.time.values
    modis_data = dict(img=modis_data_arrays, time=modis_times)

    # Landsat
    print("Reading landsat data")
    landsat_bands = ["blue", "green", "red", "nir08", "swir16", "swir22", "lwir11"]
    ds_landsat = data_tree.landsat.fillna(0).clip(0, 65535).astype(np.uint16)
    # data x 0.0000275 - 0.2
    landsat_data_arrays = [ds_landsat[b].values for b in landsat_bands]
    landsat_data_arrays = np.stack(landsat_data_arrays, axis=1)  # T,C,H,W
    landsat_times = ds_landsat.time.values
    landsat_data = dict(img=landsat_data_arrays, time=landsat_times)

    # HR
    print("Reading hr data")
    hr_ds = data_tree["hr/data"].to_dataset().load()  # load in mem.
    ds_hr = hr_ds.data.data.astype(np.uint8)
    print(ds_hr.shape)
    hr_data = dict(img=ds_hr, meta_data=data_tree["hr/data"].metadata)

    return edict(
        name=data_tree.attrs["s2id"],
        s2=s2_data,
        s1=s1_data,
        modis=modis_data,
        landsat=landsat_data,
        hr=hr_data,
    )


def dict_data_to_bytes(data: edict):
    s2, s1, modis, landsat, hr = data.s2, data.s1, data.modis, data.landsat, data.hr

    tiff_codec_fn = partial(
        tiff_codec_io,
        compression="jpeg2000",
        compression_args={
            "reversible": False,
            "quality": 90,
        },
    )
    s2_bytes_list = [tiff_codec_fn(s2.img[i].transpose(1, 2, 0)) for i in range(s2.img.shape[0])]
    s1_bytes_list = [tiff_codec_fn(s1.img[i].transpose(1, 2, 0)) for i in range(s1.img.shape[0])]
    modis_bytes_list = [tiff_codec_io(modis.img[i].transpose(1, 2, 0)) for i in range(modis.img.shape[0])]
    landsat_bytes_list = [tiff_codec_fn(landsat.img[i].transpose(1, 2, 0)) for i in range(landsat.img.shape[0])]

    hr_img = hr.img.transpose(1, 2, 0)  # H,W,C
    hr_bytes = rgb_codec_io(hr_img, quality=85)

    # pickle all seq data
    s2_lst_bytes = pickle.dumps(s2_bytes_list, protocol=5)
    s1_lst_bytes = pickle.dumps(s1_bytes_list, protocol=5)
    modis_lst_bytes = pickle.dumps(modis_bytes_list, protocol=5)
    landsat_lst_bytes = pickle.dumps(landsat_bytes_list, protocol=5)

    data_bytes = dict(
        s2_img=s2_lst_bytes,
        s2_time=s2.time,
        s1_img=s1_lst_bytes,
        s1_time=s1.time,
        modis_img=modis_lst_bytes,
        modis_time=modis.time,
        landsat_img=landsat_lst_bytes,
        landsat_time=landsat.time,
        hr_img=hr_bytes,
        hr_meta_data=hr.meta_data,
    )
    return data_bytes


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
    file_path = "/Data2/ZihanCao/dataset/core-five/src/datatree/0c6061/0c605df4.nc"
    data_tree = read_nc_file(file_path)
    core_five_groups = extract_core_five_groups(data_tree)
    print(core_five_groups)

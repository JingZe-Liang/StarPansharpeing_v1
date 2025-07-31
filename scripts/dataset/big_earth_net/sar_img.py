import numpy as np
import scipy

# from scripts.dataset.big_earth_net.sar_proc import sar
# from scripts.dataset.big_earth_net.sar_proc.image import (
#     Image,
#     MergeToStack,
#     SaveImage,
#     SelectBands,
# )
# from scripts.dataset.big_earth_net.sar_proc.pipesegment import LoadSegment


def amplitude_transform(data):
    """计算复数数据的幅度"""
    return np.absolute(data)


def decibels_transform(data, flag="min"):
    """将数据转换为分贝单位"""
    if isinstance(flag, str) and flag.lower() == "min":
        flagval = 10.0 * np.log10((data)[data > 0].min())
    elif isinstance(flag, str) and flag.lower() == "nan":
        flagval = np.nan
    else:
        flagval = flag / 10.0

    result = np.full(np.shape(data), flagval).astype(data.dtype)
    result[data > 0] = 10.0 * np.log10(data[data > 0])
    return result


def multilook_filter(data, kernel_size=3, method="avg"):
    """多视滤波降噪"""
    if method == "avg":
        filter_func = scipy.ndimage.uniform_filter
    elif method == "med":
        filter_func = scipy.ndimage.median_filter
    elif method == "max":
        filter_func = scipy.ndimage.maximum_filter
    else:
        raise Exception("! Invalid method in Multilook.")

    result = np.zeros(data.shape, dtype=data.dtype)
    for i in range(data.shape[0]):
        result[i, :, :] = filter_func(data[i, :, :], size=kernel_size, mode="reflect")
    return result


def select_bands(data, bands=[0]):
    """选择指定波段"""
    if not hasattr(bands, "__iter__"):
        bands = [bands]
    return data[bands, :, :]


def merge_to_stack(images):
    """将多个图像合并为一个多波段图像"""
    return np.concatenate(images, axis=0)


def process_sentinel1_vh_vv(data):
    """
    处理哨兵1号VH/VV双极化数据 (height, width, 2) 格式

    Args:
        data: 形状为 (height, width, 2) 的numpy数组，其中:
              data[:, :, 0] = VH极化
              data[:, :, 1] = VV极化
        name: 图像名称
    Returns:
        处理后的RGB图像用于可视化
    """
    # 转换数据维度为 (2, height, width)
    image_data = np.transpose(data, (2, 0, 1))
    metadata = {}  # 可以添加元数据

    # 1. 转换为幅度图像
    amplitude_data = amplitude_transform(image_data)

    # 2. 分别提取VH和VV波段并转换为分贝
    vh_band = select_bands(amplitude_data, 0)
    vv_band = select_bands(amplitude_data, 1)

    vh_db = decibels_transform(vh_band)
    vv_db = decibels_transform(vv_band)

    # 3. 应用多视滤波降噪
    vh_smooth = multilook_filter(vh_db, kernel_size=3)
    vv_smooth = multilook_filter(vv_db, kernel_size=3)

    # 4. 创建RGB图像用于可视化 (VH=Red, VV=Green, Blue=0)
    # 手动创建三波段图像 (VH=Red, VV=Green, Blue=0)
    blue_band = np.zeros_like(vh_smooth)
    rgb_data = merge_to_stack([vh_smooth, vv_smooth, blue_band])

    # 返回处理后的图像数据
    return rgb_data


def norm(rgb_result):
    """
    归一化RGB图像数据到0-1范围
    """
    rgb_data = rgb_result
    rgb_data = (rgb_data - np.min(rgb_data)) / (np.max(rgb_data) - np.min(rgb_data))
    return rgb_data


def gamma(img, gamma_value: float = 2.0):
    """
    应用Gamma校正到图像数据
    """
    # 确保数据为正数并归一化到[0,1]
    data = img
    data = np.clip(data, 0, None)  # 将负数设为0

    # 归一化到[0,1]
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max > data_min:
        data = (data - data_min) / (data_max - data_min)

    # 应用Gamma校正
    data = np.power(data, 1 / gamma_value)

    # 恢复原始范围
    data = data * (data_max - data_min) + data_min
    return data


# 使用示例
if __name__ == "__main__":
    # 示例用法:
    # 1. 加载哨兵1号数据
    # sentinel1_data = np.load("sentinel1_data.npy")  # 形状: (height, width, 2)
    import tifffile

    data = tifffile.imread(
        "data/Multispectral-Spacenet-series/SN6_buildings/test_public/AOI_11_Rotterdam/SAR-Intensity/SN6_Test_Public_AOI_11_Rotterdam_SAR-Intensity_20190804113009_20190804113242_tile_4823.tif"
    )
    print(data.shape)
    rgb = data[..., [0, -1, 1]]
    rgb = gamma(rgb, gamma_value=2.0)

    sentinel1_data = tifffile.imread("data/ISASeg/Sentinel1/Tile_86_52.tif")
    print(f"Loaded Sentinel-1 data with shape: {sentinel1_data.shape}")

    # 2. 处理数据
    img = process_sentinel1_vh_vv(sentinel1_data)
    print("Processed RGB image shape:", img.shape)
    img = norm(img)
    # img = gamma(img, gamma_value=2.0)
    pass
    # vh_result, vv_result = process_separately(sentinel1_data, "my_sentinel1_image")

    # 3. 保存结果
    # save_processed_images(sentinel1_data, "sentinel1_sample", "./output/")

    # print("SAR图像处理模块已准备好使用")

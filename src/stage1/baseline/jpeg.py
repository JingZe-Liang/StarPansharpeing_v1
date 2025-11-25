import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import imageio
import numpy as np
import tifffile
from PIL import Image
from loguru import logger


class HyperspectralJPEG2000Compressor:
    """
    高光谱图像JPEG2000压缩器
    """

    def __init__(self, quality: float = 50.0, compression_ratio: Optional[float] = None):
        """
        初始化压缩器

        Args:
            quality: JPEG2000压缩质量 (0-100, 100为无损)
            compression_ratio: 压缩比，如果指定则忽略quality参数
        """
        self.quality = quality
        self.compression_ratio = compression_ratio

    def compress_hyperspectral(
        self,
        hyperspectral_data: np.ndarray,
        output_dir: Union[str, Path],
        prefix: str = "band",
        normalize: bool = False,
    ) -> Tuple[list, dict]:
        """
        压缩高光谱图像数据为单个多波段 TIFF 文件

        Args:
            hyperspectral_data: 高光谱数据 (H, W, C) 或 (C, H, W)
            output_dir: 输出目录
            prefix: 文件名前缀
            normalize: 是否归一化数据到0-255

        Returns:
            compressed_files: 压缩文件路径列表
            compression_stats: 压缩统计信息
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        height, width, num_bands = hyperspectral_data.shape
        logger.info(f"Processing hyperspectral data: {height}x{width}x{num_bands}")

        # 数据预处理（可选归一化）
        if normalize:
            processed = np.zeros((height, width, num_bands), dtype=np.uint8)
            for i in range(num_bands):
                processed[:, :, i] = self._preprocess_band(hyperspectral_data[:, :, i], True)
        else:
            processed = (
                hyperspectral_data
                if hyperspectral_data.dtype in (np.uint8, np.uint16)
                else hyperspectral_data.astype(np.uint16)
            )

        # 选择压缩方式
        if self.compression_ratio is not None:
            if self.compression_ratio > 10:
                compression = "zlib"
                compressionargs = {"level": 9}
            else:
                compression = "lzw"
                compressionargs = {}
        else:
            if self.quality < 30:
                compression = "zlib"
                compressionargs = {"level": 9}
            elif self.quality < 70:
                compression = "lzw"
                compressionargs = {}
            else:
                compression = "deflate"
                compressionargs = {}

        # 输出为单文件
        output_file = output_dir / f"{prefix}.tif"
        tifffile.imwrite(
            str(output_file),
            processed,
            compression=compression,
            compressionargs=compressionargs,
        )

        # 统计信息
        original_size = hyperspectral_data.nbytes
        compressed_size = output_file.stat().st_size
        compression_stats = {
            "original_size": original_size,
            "compressed_size": compressed_size,
            "compression_ratio": original_size / compressed_size if compressed_size > 0 else 0,
            "num_bands": num_bands,
        }
        logger.info(f"Saved multi-band TIFF: {output_file.name} (orig {original_size}, comp {compressed_size})")
        return [str(output_file)], compression_stats

    def _preprocess_band(self, band_data: np.ndarray, normalize: bool) -> np.ndarray:
        """
        预处理单个波段数据

        Args:
            band_data: 单波段数据
            normalize: 是否归一化

        Returns:
            处理后的波段数据
        """
        if normalize:
            # 归一化到0-255范围
            min_val = np.min(band_data)
            max_val = np.max(band_data)
            if max_val > min_val:
                normalized = ((band_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                normalized = np.zeros_like(band_data, dtype=np.uint8)
            return normalized
        else:
            # 确保数据类型为uint8或uint16
            if band_data.dtype != np.uint8 and band_data.dtype != np.uint16:
                return band_data.astype(np.uint16)
            return band_data

    def decompress_hyperspectral(
        self,
        compressed_files: list,
        output_shape: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """
        解压缩高光谱图像

        Args:
            compressed_files: 压缩文件路径列表
            output_shape: 期望的输出形状 (H, W, C)，如果为None则自动推断

        Returns:
            解压缩的高光谱数据
        """
        if not compressed_files:
            raise ValueError("No compressed files provided")

        # 读取多波段 TIFF 文件
        data = tifffile.imread(compressed_files[0])
        if output_shape is not None and data.shape != output_shape:
            raise AssertionError(f"Shape mismatch: expected {output_shape}, got {data.shape}")
        logger.info(f"Decompressed multi-band data: {data.shape}")
        return data


def compress_hyperspectral_file(
    input_file: Union[str, Path],
    output_dir: Union[str, Path],
    quality: float = 50.0,
    compression_ratio: Optional[float] = None,
    normalize: bool = True,
) -> dict:
    """
    压缩高光谱文件的便捷函数

    Args:
        input_file: 输入高光谱文件 (.npy, .mat等)
        output_dir: 输出目录
        quality: 压缩质量
        compression_ratio: 压缩比
        normalize: 是否归一化

    Returns:
        压缩统计信息
    """
    input_file = Path(input_file)

    # 加载高光谱数据
    if input_file.suffix == ".npy":
        hyperspectral_data = np.load(input_file)
    elif input_file.suffix == ".mat":
        from scipy.io import loadmat

        mat_data = loadmat(input_file)
        # 假设数据在名为'data'或第一个非元数据键中
        hyperspectral_data = None
        for key in mat_data.keys():
            if not key.startswith("__"):
                hyperspectral_data = mat_data[key]
                break
        if hyperspectral_data is None:
            raise ValueError(f"No valid data found in {input_file}")
    else:
        raise ValueError(f"Unsupported file format: {input_file.suffix}")

    # 创建压缩器并执行压缩
    compressor = HyperspectralJPEG2000Compressor(quality, compression_ratio)
    compressed_files, stats = compressor.compress_hyperspectral(
        hyperspectral_data, output_dir, input_file.stem, normalize
    )

    return stats


if __name__ == "__main__":
    # 示例用法

    # 创建示例高光谱数据 (100x100x200波段)
    # sample_data = np.random.rand(100, 100, 200) * 1000
    # sample_data = sample_data.astype(np.uint16)

    sample_data = tifffile.imread("/HardDisk/ZiHanCao/datasets/Multispectral-DFC2020/s1_0/ROIs0000_test_s1_0_p140.tif")

    # 压缩
    quality = 50.0
    compressor = HyperspectralJPEG2000Compressor(quality=quality)
    compressed_files, stats = compressor.compress_hyperspectral(
        sample_data, "compressed_output", prefix=f"hyperspectral_band_{quality}"
    )

    print(f"压缩完成！")
    print(f"原始大小: {stats['original_size'] / (1024 * 1024):.2f} MB")
    print(f"压缩大小: {stats['compressed_size'] / (1024 * 1024):.2f} MB")
    print(f"压缩比: {stats['compression_ratio']:.2f}")

    # 解压缩
    decompressed_data = compressor.decompress_hyperspectral(compressed_files)
    print(f"解压缩形状: {decompressed_data.shape}")

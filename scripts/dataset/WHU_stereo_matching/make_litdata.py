"""
WHU立体匹配数据集litdata转换脚本

将WHU Stereo Matching数据集转换为litdata格式。
数据集结构:
- train/val/test
  - left: 左视图影像
  - right: 右视图影像
  - disp: 视差图ground truth
"""

from pathlib import Path

import numpy as np
import tifffile
from litdata import optimize

from src.data.codecs import tiff_codec_io


def process_stereo_pair(
    paths: tuple[str, str, str],
) -> dict[str, str | bytes]:
    """处理单个立体匹配样本

    Args:
        paths: (left_path, right_path, disp_path) 的元组

    Returns:
        包含序列化数据的字典
    """
    left_path, right_path, disp_path = paths

    # 提取文件名作为key
    # 例如: KM_left_100.tiff -> KM_100
    left_stem = Path(left_path).stem
    # 移除 "left" 或 "right" 或 "disparity" 前缀
    if "_left_" in left_stem:
        key = left_stem.replace("_left_", "_")
    else:
        key = left_stem

    # 读取图像数据
    left_img = tifffile.imread(left_path)
    right_img = tifffile.imread(right_path)
    disp_img = tifffile.imread(disp_path)

    # 编码为字节流
    left_bytes = tiff_codec_io(
        left_img[..., None], compression="jpeg2000", compression_args={"reversible": False, "level": 95}
    )
    right_bytes = tiff_codec_io(
        right_img[..., None], compression="jpeg2000", compression_args={"reversible": False, "level": 95}
    )
    disp_bytes = tiff_codec_io(disp_img, compression="zlib", compression_args={"level": 9})

    return {"__key__": key, "left": left_bytes, "right": right_bytes, "disp": disp_bytes}


def process_split(base_dir: Path, split: str, output_dir: Path) -> None:
    """处理单个数据划分 (train/val/test)

    Args:d
        base_dir: 数据集根目录
        split: 数据划分名称 (train/val/test)
        output_dir: 输出目录
    """
    split_dir = base_dir / split
    left_dir = split_dir / "left"
    right_dir = split_dir / "right"
    disp_dir = split_dir / "disp"

    # 检查目录是否存在
    if not split_dir.exists():
        print(f"警告: {split_dir} 不存在，跳过")
        return

    # 获取所有left图像文件
    left_files = sorted(list(left_dir.glob("*.tiff")) + list(left_dir.glob("*.tif")))

    if len(left_files) == 0:
        print(f"警告: {split_dir} 中没有找到图像文件，跳过")
        return

    # 构建配对列表
    pairs = []
    for left_path in left_files:
        # 根据left文件名推断right和disp文件名
        # 例如: KM_left_100.tiff -> KM_right_100.tiff, KM_disparity_100.tiff
        left_stem = left_path.stem

        # 提取编号部分
        if "_left_" in left_stem:
            prefix = left_stem.split("_left_")[0]
            suffix = left_stem.split("_left_")[1]
            right_name = f"{prefix}_right_{suffix}{left_path.suffix}"
            disp_name = f"{prefix}_disparity_{suffix}{left_path.suffix}"
        else:
            # 如果命名格式不同，尝试简单替换
            right_name = left_stem.replace("left", "right") + left_path.suffix
            disp_name = left_stem.replace("left", "disparity") + left_path.suffix

        right_path = right_dir / right_name
        disp_path = disp_dir / disp_name

        # 检查配对文件是否存在
        if not right_path.exists():
            print(f"警告: 找不到配对的right图像: {right_path}")
            continue
        if not disp_path.exists():
            print(f"警告: 找不到配对的disp图像: {disp_path}")
            continue

        pairs.append((str(left_path), str(right_path), str(disp_path)))

    print(f"\n处理 {split} 数据集:")
    print(f"  找到 {len(pairs)} 对立体影像")

    # 创建输出目录
    split_output_dir = output_dir / split
    split_output_dir.mkdir(parents=True, exist_ok=True)

    # 使用litdata优化
    optimize(
        process_stereo_pair,
        pairs,
        str(split_output_dir),
        num_workers=0,
        # start_method="fork",
        chunk_bytes="512Mb",
    )

    print(f"  完成! 输出到: {split_output_dir}")


def main():
    """主函数"""
    # 数据集路径
    base_dir = Path("data/Downstreams/WHU_Stereo_Matching/experimental data/with ground truth")
    output_dir = Path("data/Downstreams/WHU_Stereo_Matching/litdata")

    # 处理每个数据划分
    for split in ["train", "val", "test"]:
        process_split(base_dir, split, output_dir)

    print("\n所有数据集转换完成!")


if __name__ == "__main__":
    main()

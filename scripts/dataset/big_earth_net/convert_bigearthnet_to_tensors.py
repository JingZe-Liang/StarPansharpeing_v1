#!/usr/bin/env python3
# ruff disable
# flake8: noqa

"""
BigEarthNet-S2 数据集转换脚本
将每个补丁文件夹中的多分辨率 TIF 文件转换为统一尺寸的张量并保存
"""

import os
import numpy as np
import tifffile
import torch
from scipy import ndimage
from pathlib import Path
import argparse
from tqdm import tqdm
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import multiprocessing as mp


# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s %(funcName)s] - %(message)s",
)
logger = logging.getLogger(__name__)


class BigEarthNetProcessor:
    def __init__(self, target_size=120):
        """
        初始化处理器

        Args:
            target_size: 目标分辨率尺寸，默认120x120（对应10m分辨率）
        """
        self.target_size = target_size

        # Sentinel-2 波段信息：波段名 -> (原始尺寸, 分辨率)
        self.band_info = {
            "B01": (20, 60),  # 60m 分辨率 -> 20x20 像素
            "B02": (120, 10),  # 10m 分辨率 -> 120x120 像素
            "B03": (120, 10),  # 10m 分辨率 -> 120x120 像素
            "B04": (120, 10),  # 10m 分辨率 -> 120x120 像素
            "B05": (60, 20),  # 20m 分辨率 -> 60x60 像素
            "B06": (60, 20),  # 20m 分辨率 -> 60x60 像素
            "B07": (60, 20),  # 20m 分辨率 -> 60x60 像素
            "B08": (120, 10),  # 10m 分辨率 -> 120x120 像素
            "B8A": (60, 20),  # 20m 分辨率 -> 60x60 像素
            "B09": (20, 60),  # 60m 分辨率 -> 20x20 像素
            "B11": (60, 20),  # 20m 分辨率 -> 60x60 像素
            "B12": (60, 20),  # 20m 分辨率 -> 60x60 像素
        }

        # 波段顺序（按光谱顺序排列）
        self.band_order = [
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B09",
            "B11",
            "B12",
        ]

    def load_and_resample_band(self, tif_path, target_size):
        """
        加载并重采样单个波段

        Args:
            tif_path: TIF 文件路径
            target_size: 目标尺寸

        Returns:
            重采样后的波段数据 (numpy array)
        """
        try:
            # 使用 tifffile 读取数据
            band_data = tifffile.imread(tif_path)
            dtype = band_data.dtype
            band_data = band_data.astype(np.float32)

            # 如果尺寸不匹配，进行重采样
            if band_data.shape[0] != target_size or band_data.shape[1] != target_size:
                zoom_factor = target_size / band_data.shape[0]  # 假设图像是方形的
                band_data = ndimage.zoom(band_data, zoom_factor, order=1)  # 双线性插值

            return band_data.astype(dtype), dtype

        except Exception as e:
            logger.error(f"Error loading {tif_path}: {e}")
            return None, None

    def process_patch(self, patch_dir):
        """
        处理单个补丁文件夹

        Args:
            patch_dir: 补丁文件夹路径

        Returns:
            numpy array: shape (12, target_size, target_size)
        """
        patch_name = os.path.basename(patch_dir)
        bands_data = []

        for band_name in self.band_order:
            # 构建 TIF 文件路径
            tif_path = os.path.join(patch_dir, f"{patch_name}_{band_name}.tif")

            if not os.path.exists(tif_path):
                logger.warning(f"Missing file: {tif_path}")
                # 创建空白数组作为占位符
                band_data = np.zeros(
                    (self.target_size, self.target_size),
                    dtype=np.int16,  # assume the data type is int16
                )
            else:
                band_data, dtype = self.load_and_resample_band(
                    tif_path, self.target_size
                )

                if band_data is None:
                    # 如果加载失败，创建空白数组
                    band_data = np.zeros(
                        (self.target_size, self.target_size), dtype=dtype
                    )

            bands_data.append(band_data)

        # 堆叠所有波段
        tensor_data = np.stack(bands_data, axis=0)  # (12, target_size, target_size)

        return tensor_data

    def save_tensor(self, tensor_data, output_path):
        """
        保存张量数据

        Args:
            tensor_data: numpy array
            output_path: 输出文件路径
        """
        try:
            # 转换为 PyTorch 张量并保存
            # tensor = torch.from_numpy(tensor_data)
            # torch.save(tensor, output_path)
            # logger.info(f"Saved tensor to {output_path}, shape: {tensor.shape}")

            # save in tif files
            tifffile.imwrite(output_path, tensor_data)
            # logger.info(f"Saved tensor to {output_path}, shape: {tensor_data.shape}")

        except Exception as e:
            logger.error(f"Error saving tensor to {output_path}: {e}")

    def process_dataset(
        self, input_dir, output_dir, tile_dirs=None, process_n: int = 1
    ):
        """
        处理整个数据集

        Args:
            input_dir: 输入目录（BigEarthNet-S2 根目录）
            output_dir: 输出目录
            tile_dirs: 要处理的瓦片目录列表，None表示处理所有
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 获取所有瓦片目录
        if tile_dirs is None:
            tile_dirs = [d for d in input_path.iterdir() if d.is_dir()]
        else:
            tile_dirs = [input_path / tile_dir for tile_dir in tile_dirs]

        total_processed = 0

        for tile_dir in tqdm(tile_dirs, desc="Processing tiles"):
            if not tile_dir.is_dir():
                continue

            logger.info(f"Processing tile: {tile_dir.name}")

            # 获取该瓦片下的所有补丁目录
            patch_dirs = [d for d in tile_dir.iterdir() if d.is_dir()]

            for patch_dir in tqdm(
                patch_dirs, desc=f"Processing patches in {tile_dir.name}", leave=False
            ):
                patch_name = patch_dir.name

                # 处理补丁
                tensor_data = self.process_patch(patch_dir)

                # 保存张量
                output_file = output_path / f"{patch_name}.tif"
                self.save_tensor(tensor_data, output_file)

                total_processed += 1

        logger.info(f"Processing completed! Total patches processed: {total_processed}")

    def process_single_patch(self, patch_info):
        """
        处理单个补丁的静态方法（用于多进程）

        Args:
            patch_info: tuple (patch_dir, output_dir, target_size)

        Returns:
            tuple: (patch_name, success, error_msg)
        """
        patch_dir, output_dir, target_size = patch_info
        patch_name = os.path.basename(patch_dir)

        try:
            # 创建临时处理器实例
            temp_processor = BigEarthNetProcessor(target_size)

            # 处理补丁
            tensor_data = temp_processor.process_patch(patch_dir)

            # 保存张量
            output_file = Path(output_dir) / f"{patch_name}.tif"
            temp_processor.save_tensor(tensor_data, output_file)

            return (patch_name, True, None)

        except Exception as e:
            return (patch_name, False, str(e))

    def process_dataset_multiprocess(
        self, input_dir, output_dir, tile_dirs=None, max_workers=None, batch_size=100
    ):
        """
        多进程处理整个数据集

        Args:
            input_dir: 输入目录（BigEarthNet-S2 根目录）
            output_dir: 输出目录
            tile_dirs: 要处理的瓦片目录列表，None表示处理所有
            max_workers: 最大进程数，None表示使用CPU核心数
            batch_size: 每批处理的补丁数量
        """
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 如果没有指定进程数，使用CPU核心数
        if max_workers is None:
            max_workers = mp.cpu_count()

        logger.info(f"Using {max_workers} processes for parallel processing")

        # 获取所有瓦片目录
        if tile_dirs is None:
            tile_dirs = [d for d in input_path.iterdir() if d.is_dir()]
        else:
            tile_dirs = [input_path / tile_dir for tile_dir in tile_dirs]

        # 收集所有补丁目录
        all_patch_dirs = []
        for tile_dir in tile_dirs:
            if tile_dir.is_dir():
                patch_dirs = [d for d in tile_dir.iterdir() if d.is_dir()]
                all_patch_dirs.extend(patch_dirs)

        total_patches = len(all_patch_dirs)
        logger.info(f"Found {total_patches} patches to process")

        # 准备任务参数
        patch_tasks = [
            (patch_dir, output_dir, self.target_size) for patch_dir in all_patch_dirs
        ]

        # 多进程处理
        processed_count = 0
        failed_count = 0

        # 计算总批次数（确保包含最后一个不完整的批次）
        total_batches = (len(patch_tasks) + batch_size - 1) // batch_size

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 分批提交任务
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(patch_tasks))
                batch = patch_tasks[start_idx:end_idx]

                logger.info(
                    f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch)} patches)"
                )

                # 提交批次任务
                future_to_patch = {
                    executor.submit(process_patch_worker, task): task[0].name
                    for task in batch
                }

                # 处理完成的任务
                for future in tqdm(
                    as_completed(future_to_patch),
                    total=len(batch),
                    desc=f"Batch {batch_idx + 1}/{total_batches}",
                ):
                    patch_name = future_to_patch[future]
                    try:
                        result_patch_name, success, error_msg = future.result()
                        if success:
                            processed_count += 1
                        else:
                            failed_count += 1
                            logger.error(
                                f"Failed to process {result_patch_name}: {error_msg}"
                            )
                    except Exception as e:
                        failed_count += 1
                        logger.error(f"Exception processing {patch_name}: {e}")

        logger.info(
            f"Processing completed! Processed: {processed_count}, Failed: {failed_count}"
        )
        logger.info(f"Total patches expected: {total_patches}")

    def process_dataset_multithread(
        self, input_dir, output_dir, tile_dirs=None, max_workers=None
    ):
        """
        多线程处理整个数据集（适用于I/O密集型任务）

        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            tile_dirs: 要处理的瓦片目录列表
            max_workers: 最大线程数
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if max_workers is None:
            max_workers = min(32, (mp.cpu_count() or 1) + 4)

        logger.info(f"Using {max_workers} threads for parallel processing")

        # 获取所有补丁目录
        if tile_dirs is None:
            tile_dirs = [d for d in input_path.iterdir() if d.is_dir()]
        else:
            tile_dirs = [input_path / tile_dir for tile_dir in tile_dirs]

        all_patch_dirs = []
        for tile_dir in tqdm(tile_dirs, desc="Collecting patch directories"):
            if tile_dir.is_dir():
                patch_dirs = [d for d in tile_dir.iterdir() if d.is_dir()]
                all_patch_dirs.extend(patch_dirs)

        total_patches = len(all_patch_dirs)
        logger.info(f"Found {total_patches} patches to process")

        processed_count = 0
        failed_count = 0
        lock = threading.Lock()

        def process_patch_thread(patch_dir):
            nonlocal processed_count, failed_count

            try:
                patch_name = patch_dir.name
                tensor_data = self.process_patch(patch_dir)
                output_file = output_path / f"{patch_name}.tif"
                self.save_tensor(tensor_data, output_file)

                with lock:
                    processed_count += 1

            except Exception as e:
                with lock:
                    failed_count += 1
                logger.error(f"Failed to process {patch_dir.name}: {e}")

        # 使用线程池
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = [
                executor.submit(process_patch_thread, patch_dir)
                for patch_dir in all_patch_dirs
            ]

            # 等待所有任务完成
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing patches"
            ):
                future.result()  # 获取结果，如果有异常会抛出

        logger.info(
            f"Processing completed! Processed: {processed_count}, Failed: {failed_count}"
        )

    def process_dataset_multithread_online(
        self,
        input_dir,
        output_dir,
        tile_dirs=None,
        max_workers=None,
        submit_batch_size=100,
    ):
        """
        在线多线程处理整个数据集（边发现边提交任务）

        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            tile_dirs: 要处理的瓦片目录列表
            max_workers: 最大线程数
            submit_batch_size: 每次提交的任务批次大小
        """

        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if max_workers is None:
            max_workers = min(32, (mp.cpu_count() or 1) + 4)

        logger.info(f"Using {max_workers} threads for parallel processing")

        # 获取瓦片目录
        if tile_dirs is None:
            tile_dirs = [d for d in input_path.iterdir() if d.is_dir()]
        else:
            tile_dirs = [input_path / tile_dir for tile_dir in tile_dirs]

        processed_count = 0
        failed_count = 0
        submitted_count = 0
        lock = threading.Lock()

        def process_patch_thread(patch_dir):
            nonlocal processed_count, failed_count

            try:
                patch_name = patch_dir.name
                tensor_data = self.process_patch(patch_dir)
                output_file = output_path / f"{patch_name}.tif"
                self.save_tensor(tensor_data, output_file)

                with lock:
                    processed_count += 1

            except Exception as e:
                with lock:
                    failed_count += 1
                logger.error(f"Failed to process {patch_dir.name}: {e}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 使用双端队列管理待处理的futures
            pending_futures = deque()

            # 在线发现和提交任务
            for tile_dir in tqdm(tile_dirs, desc="Processing tiles"):
                if not tile_dir.is_dir():
                    continue

                logger.info(f"Processing tile: {tile_dir.name}")

                # 遍历该瓦片下的补丁目录
                patch_dirs = [d for d in tile_dir.iterdir() if d.is_dir()]

                for i, patch_dir in enumerate(patch_dirs):
                    # 提交任务
                    future = executor.submit(process_patch_thread, patch_dir)
                    pending_futures.append(future)
                    submitted_count += 1

                    # 每处理一定数量的任务或达到批次大小时，等待部分任务完成
                    if (
                        len(pending_futures) >= submit_batch_size
                        or i == len(patch_dirs) - 1
                    ):
                        # 等待一些任务完成，避免队列过长
                        completed_count = 0
                        target_complete = (
                            len(pending_futures) // 2
                            if len(pending_futures) > submit_batch_size
                            else len(pending_futures)
                        )

                        while completed_count < target_complete and pending_futures:
                            # 检查已完成的任务
                            for _ in range(len(pending_futures)):
                                future = pending_futures.popleft()
                                if future.done():
                                    try:
                                        future.result()  # 获取结果，处理异常
                                        completed_count += 1
                                    except Exception as e:
                                        logger.error(f"Future execution error: {e}")
                                else:
                                    pending_futures.append(future)  # 放回队列末尾

                            # 如果没有任务完成，等待一小段时间
                            if completed_count == 0:
                                import time

                                time.sleep(0.1)

                # 更新进度
                with lock:
                    logger.info(
                        f"Tile {tile_dir.name} completed. Processed: {processed_count}, Failed: {failed_count}"
                    )

            # 等待所有剩余任务完成
            logger.info("Waiting for remaining tasks to complete...")
            while pending_futures:
                future = pending_futures.popleft()
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Final future execution error: {e}")

        logger.info(
            f"Processing completed! Submitted: {submitted_count}, Processed: {processed_count}, Failed: {failed_count}"
        )

    def process_dataset_multiprocess_online(
        self,
        input_dir,
        output_dir,
        tile_dirs=None,
        max_workers=None,
        submit_batch_size=100,
    ):
        """
        在线多进程处理整个数据集
        """
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from collections import deque

        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if max_workers is None:
            max_workers = mp.cpu_count()

        logger.info(f"Using {max_workers} processes for parallel processing")

        # 获取瓦片目录
        if tile_dirs is None:
            tile_dirs = [d for d in input_path.iterdir() if d.is_dir()]
        else:
            tile_dirs = [input_path / tile_dir for tile_dir in tile_dirs]

        processed_count = 0
        failed_count = 0
        submitted_count = 0

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            pending_futures = deque()

            for tile_dir in (tbar := tqdm(tile_dirs)):
                if not tile_dir.is_dir():
                    continue

                # logger.info(f"Processing tile: {tile_dir.name}")
                tbar.set_description(f"Processing tile: {tile_dir.name}")

                # 遍历该瓦片下的补丁目录
                patch_dirs = [d for d in tile_dir.iterdir() if d.is_dir()]

                for i, patch_dir in enumerate(patch_dirs):
                    # 准备任务参数
                    task = (patch_dir, output_dir, self.target_size)

                    # 提交任务
                    future = executor.submit(process_patch_worker, task)
                    pending_futures.append((future, patch_dir.name))
                    submitted_count += 1

                    # 管理队列大小
                    if len(pending_futures) >= submit_batch_size:
                        # 处理一批已完成的任务
                        batch_processed = 0
                        target_process = submit_batch_size // 2

                        while batch_processed < target_process and pending_futures:
                            # 查找已完成的任务
                            for j in range(len(pending_futures)):
                                future, patch_name = pending_futures.popleft()

                                if future.done():
                                    try:
                                        result_patch_name, success, error_msg = (
                                            future.result()
                                        )
                                        if success:
                                            processed_count += 1
                                        else:
                                            failed_count += 1
                                            logger.error(
                                                f"Failed to process {result_patch_name}: {error_msg}"
                                            )
                                        batch_processed += 1
                                    except Exception as e:
                                        failed_count += 1
                                        logger.error(
                                            f"Exception processing {patch_name}: {e}"
                                        )
                                        batch_processed += 1
                                else:
                                    pending_futures.append((future, patch_name))

                            if batch_processed == 0:
                                import time

                                time.sleep(0.1)

                logger.info(
                    f"Tile {tile_dir.name} submitted. Current stats - Processed: {processed_count}, Failed: {failed_count}"
                )

            # 处理所有剩余任务
            logger.info("Processing remaining tasks...")
            for future, patch_name in tqdm(
                pending_futures, desc="Finishing remaining tasks"
            ):
                try:
                    result_patch_name, success, error_msg = future.result()
                    if success:
                        processed_count += 1
                    else:
                        failed_count += 1
                        logger.error(
                            f"Failed to process {result_patch_name}: {error_msg}"
                        )
                except Exception as e:
                    failed_count += 1
                    logger.error(f"Exception processing {patch_name}: {e}")

        logger.info(
            f"Processing completed! Submitted: {submitted_count}, Processed: {processed_count}, Failed: {failed_count}"
        )


# 全局函数用于多进程（必须在模块级别定义）
def process_patch_worker(patch_info):
    """
    多进程工作函数
    """
    patch_dir, output_dir, target_size = patch_info
    patch_name = os.path.basename(patch_dir)

    try:
        # 创建处理器实例
        processor = BigEarthNetProcessor(target_size)

        # 处理补丁
        tensor_data = processor.process_patch(patch_dir)

        # 保存张量
        output_file = Path(output_dir) / f"{patch_name}.tif"
        processor.save_tensor(tensor_data, output_file)

        return (patch_name, True, None)

    except Exception as e:
        return (patch_name, False, str(e))


def main():
    parser = argparse.ArgumentParser(
        description="Convert BigEarthNet-S2 patches to tensors"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/HardDisk/ZiHanCao/datasets/Multispectral-BigEarthNet/S2/BigEarthNet-S2",
        help="Input directory containing BigEarthNet-S2 data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/HardDisk/ZiHanCao/datasets/Multispectral-BigEarthNet/S2/S2_tiff",
        help="Output directory for tensor files",
    )
    parser.add_argument(
        "--target_size", type=int, default=128, help="Target size for resampling"
    )
    parser.add_argument(
        "--tile_dirs",
        nargs="+",
        default=None,
        help="Specific tile directories to process (default: all)",
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Process only first 5 patches for testing",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Maximum number of workers (default: CPU count)",
    )
    parser.add_argument(
        "--use_multiprocess",
        action="store_true",
        help="Use multiprocessing instead of multithreading",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for multiprocessing (default: 100)",
    )
    parser.add_argument(
        "--online_mode",
        action="store_true",
        help="Use online processing (discover and submit tasks on-the-fly)",
    )
    parser.add_argument(
        "--submit_batch_size",
        type=int,
        default=100,
        help="Batch size for online submission",
    )

    args = parser.parse_args()

    args.online_mode = True
    args.use_multiprocess = True  # 强制使用多进程

    # 创建处理器
    processor = BigEarthNetProcessor(target_size=args.target_size)
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"make dir {output_path.as_posix()}")

    if args.test_mode:
        logger.info("Running in test mode - processing only first 5 patches")
        # 测试模式保持单进程

        patch_count = 0
        for tile_dir in input_path.iterdir():
            if not tile_dir.is_dir():
                continue
            for patch_dir in tile_dir.iterdir():
                if not patch_dir.is_dir() or patch_count >= 5:
                    break

                logger.info(f"Processing test patch: {patch_dir.name}")
                tensor_data = processor.process_patch(patch_dir)
                output_file = output_path / f"{patch_dir.name}.tif"
                processor.save_tensor(tensor_data, output_file)
                patch_count += 1

            if patch_count >= 5:
                break
    else:
        # 选择处理方式
        if args.online_mode:
            if args.use_multiprocess:
                logger.info("Using online multiprocessing for parallel execution")
                processor.process_dataset_multiprocess_online(
                    args.input_dir,
                    args.output_dir,
                    args.tile_dirs,
                    max_workers=args.max_workers,
                    submit_batch_size=args.submit_batch_size,
                )
            else:
                logger.info("Using online multithreading for parallel execution")
                processor.process_dataset_multithread_online(
                    args.input_dir,
                    args.output_dir,
                    args.tile_dirs,
                    max_workers=args.max_workers,
                    submit_batch_size=args.submit_batch_size,
                )
        else:
            if args.use_multiprocess:
                logger.info("Using multiprocessing for parallel execution")
                processor.process_dataset_multiprocess(
                    args.input_dir,
                    args.output_dir,
                    args.tile_dirs,
                    max_workers=args.max_workers,
                    batch_size=args.batch_size,
                )
            else:
                logger.info("Using multithreading for parallel execution")
                processor.process_dataset_multithread(
                    args.input_dir,
                    args.output_dir,
                    args.tile_dirs,
                    max_workers=args.max_workers,
                )


if __name__ == "__main__":
    main()

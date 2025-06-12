import math
import random
import warnings
from typing import List, Optional, Set

import torch.distributed as dist
import wids


class IndexFilteredSampler(wids.ChunkedSampler):
    """A sampler that only yields specific indices."""

    def __init__(
        self,
        dataset,
        *,
        valid_indices: Set[int] | List[int],
        dslength_per_replica=-1,
        chunksize=2000,
        seed=0,
        shuffle=True,
        shufflefirst=False,
    ):
        self.chunksize = chunksize

        # 将有效索引转换为set以提高查找效率
        self.valid_indices = (
            set(valid_indices) if isinstance(valid_indices, list) else valid_indices
        )

        # 计算实际长度
        actual_length = len(self.valid_indices)

        # assertion
        assert max(self.valid_indices) <= len(dataset), (
            f"Max index in valid_indices ({max(self.valid_indices)}) "
            f"exceeds dataset length ({len(dataset)})."
        )

        effective_dslength = dslength_per_replica
        if dslength_per_replica == -1:
            effective_dslength = actual_length
        elif dslength_per_replica > actual_length:
            warnings.warn(
                f"dslength_per_replica ({dslength_per_replica}) is greater than "
                f"the number of valid_indices ({actual_length}). "
                f"Using {actual_length} as the effective length."
            )
            effective_dslength = actual_length

        # Ensure effective_dslength is not negative if actual_length is 0
        if actual_length == 0:
            effective_dslength = 0
            raise ValueError(
                "No valid indices provided. Please check the valid_indices parameter."
            )

        super().__init__(
            dataset,
            dslength_per_replica=effective_dslength,
            num_samples=(0, len(dataset)),  # 全范围，但会过滤
            chunksize=chunksize,
            seed=seed,
            shuffle=shuffle,
            shufflefirst=shufflefirst,
        )
        # self.chunksize should be set by the parent class wids.ChunkedSampler
        # If not, ensure it's available, e.g. self.user_chunksize = chunksize

    def __iter__(self):
        self.rng = random.Random(self.seed + 1289738273 * self.epoch)
        valid_indices_list = list(self.valid_indices)

        if self.shuffle:
            self.rng.shuffle(valid_indices_list)

        # We need to yield exactly self.length items.
        # Since self.length is now guaranteed to be <= len(valid_indices_list)
        # (unless valid_indices_list is empty and self.length > 0, which __init__ should prevent),
        # we take a slice of valid_indices_list.
        indices_to_yield = valid_indices_list[: len(self)]

        # Use the chunksize passed to this sampler, assuming parent stores it as self.chunksize
        # If wids.ChunkedSampler doesn't store it as self.chunksize,
        # you might need to retrieve it from self.init_kwargs or store it manually in __init__.
        # For this example, we assume self.chunksize is available and is the intended one.
        iter_chunksize = getattr(
            self, "chunksize", 2000
        )  # Fallback if not set by parent

        for i in range(0, len(indices_to_yield), iter_chunksize):
            chunk = indices_to_yield[i : i + iter_chunksize]
            if self.shuffle:  # Shuffle within the chunk again
                self.rng.shuffle(chunk)
            yield from chunk

        self.epoch += 1


class IndexFilteredDistributedSampler(IndexFilteredSampler):
    """分布式版本的索引过滤采样器，与 WIDS DistributedChunkedSampler 保持一致"""

    def __init__(
        self,
        dataset,
        *,
        valid_indices: Set[int] | List[int],
        num_replicas: Optional[int] = None,
        num_samples: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        shufflefirst: bool = False,
        seed: int = 0,
        drop_last: bool | None = None,
        chunksize: int = 1000000,
        sorted: bool = False,
    ):
        if drop_last is not None:
            warnings.warn("IndexFilteredDistributedSampler does not support drop_last")

        if not dist.is_initialized():
            num_replicas = 1
            rank = 0
        else:
            num_replicas = num_replicas or dist.get_world_size()
            rank = rank or dist.get_rank()

        # 将有效索引分配到各个replica
        if sorted:
            valid_indices = (
                list(valid_indices) if isinstance(valid_indices, set) else valid_indices
            )
            valid_indices.sort()  # 确保顺序一致
        else:
            valid_indices = list(valid_indices)

        # 计算每个replica的索引范围
        num_valid_samples = len(valid_indices)
        dslength_per_replica = (
            math.ceil(num_valid_samples / num_replicas)
            if num_replicas > 1
            else num_valid_samples
        )

        worker_chunk = (num_valid_samples + num_replicas - 1) // num_replicas
        worker_start = rank * worker_chunk
        worker_end = min(worker_start + worker_chunk, num_valid_samples)

        # 当前replica的有效索引
        replica_indices = valid_indices[worker_start:worker_end]

        super().__init__(
            dataset,
            valid_indices=replica_indices,
            dslength_per_replica=dslength_per_replica,
            chunksize=chunksize,
            seed=seed,
            shuffle=shuffle,
            shufflefirst=shufflefirst,
        )


def DistributedIndexFilteredSampler(
    dataset,
    *,
    valid_indices: Set[int] | List[int],
    num_replicas: Optional[int] = None,
    num_samples: Optional[int] = None,
    rank: Optional[int] = None,
    shuffle: bool = True,
    shufflefirst: bool = False,
    seed: int = 0,
    drop_last: bool | None = None,
    chunksize: int = 1000000,
) -> IndexFilteredDistributedSampler:
    """工厂函数，与 WIDS 的 DistributedChunkedSampler 保持一致的接口"""
    return IndexFilteredDistributedSampler(
        dataset,
        valid_indices=valid_indices,
        num_replicas=num_replicas,
        num_samples=num_samples,
        rank=rank,
        shuffle=shuffle,
        shufflefirst=shufflefirst,
        seed=seed,
        drop_last=drop_last,
        chunksize=chunksize,
    )


if __name__ == "__main__":
    import os
    import tempfile

    import torch.multiprocessing as mp

    def _run_distributed_test(rank, world_size):
        """每个rank运行的测试函数"""
        dist.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
            init_method="tcp://localhost:12345",  # 使用本地TCP初始化
        )

        # 测试数据 (所有rank相同)
        dataset_size = 100
        valid_indices = list(range(50))  # 假设有效索引为0-49

        # 创建sampler
        sampler = IndexFilteredDistributedSampler(
            dataset=list(range(dataset_size)),
            valid_indices=valid_indices,
            num_replicas=world_size,
            rank=rank,
            seed=42,
            shuffle=True,
        )

        # 收集当前rank的索引
        local_indices = list(sampler)

        print(f"Rank {rank} got {len(local_indices)} indices: {local_indices[:5]}...")

        # 简单验证
        assert len(local_indices) == len(sampler), "Sampler length mismatch"
        assert all(0 <= idx < dataset_size for idx in local_indices), "Invalid indices"

        dist.destroy_process_group()

    def test_distributed_sampler():
        """启动多进程测试"""
        world_size = 4  # 模拟2个GPU
        # temp_file = tempfile.mktemp()  # 用于进程间通信

        mp.spawn(
            _run_distributed_test,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )

    # os.unlink(temp_file)  # 清理临时文件

    # print("=== Testing Single Process ===")
    # dataset = list(range(100))
    # valid_indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12301}
    # sampler = IndexFilteredSampler(
    #     dataset, valid_indices=valid_indices, dslength_per_replica=-1, seed=42
    # )
    # print(f"sampler length: {len(sampler)}")
    # print("sampled indices:", list(sampler))

    print("\n=== Testing Distributed ===")
    test_distributed_sampler()

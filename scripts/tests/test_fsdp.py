import os

import torch
import torch.distributed as dist
import torch.distributed.tensor as dtensor
from torch.distributed.tensor import DeviceMesh, DTensor, Replicate, Shard


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # 确保每个进程使用不同的GPU
    torch.cuda.set_device(rank)

    # 1. 初始化分布式
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # 2. 创建DeviceMesh
    device_mesh = DeviceMesh("cuda", torch.arange(world_size))

    # 3. 创建分片DTensor
    shard_dtensor = dtensor.ones(
        (4, 4),
        dtype=torch.bfloat16,
        device_mesh=device_mesh,
        placements=[Shard(0)],
    )
    print(f"Rank {rank} shard tensor:", shard_dtensor)

    local_tensor = shard_dtensor.to_local()
    print(f"Rank {rank} local tensor shape:", local_tensor.shape)

    # 4. 尝试复制到所有rank
    try:
        all_rank_rep_tensor = shard_dtensor.redistribute(
            device_mesh, placements=[Replicate()]
        )
        print(f"Rank {rank} replicated tensor:", all_rank_rep_tensor)
    except Exception as e:
        print(f"Rank {rank} redistribution failed:", str(e))

    # 5. 清理
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

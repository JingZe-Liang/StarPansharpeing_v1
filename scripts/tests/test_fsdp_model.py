from functools import partial

import accelerate
import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate.utils import FullyShardedDataParallelPlugin, ProjectConfiguration
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, wrap
from torch.distributed.tensor import DTensor, Replicate, Shard


class Conv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3)

    def forward(self, x):
        return torch.relu(self.conv(x))


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv(3, 16)
        self.conv2 = Conv(16, 32)
        self.fc = nn.Linear(32 * 6 * 6, 10)  # 假设输入图像大小为32x32
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.max_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def get_last_layer(self):
        return self.fc.weight


def main():
    accelerator = accelerate.Accelerator(
        fsdp_plugin=FullyShardedDataParallelPlugin(
            fsdp_version=2,
            reshard_after_forward=False,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=partial(
                wrap.transformer_auto_wrap_policy,
                transformer_layer_cls={Conv},
            ),
        )
    )
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    # 1. 初始化模型并封装为FSDP
    model = SimpleCNN().cuda()
    model.dtype = torch.float32
    print(f"rank {rank} - init model to FSDP")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    fsdp_model, optimizer = accelerator.prepare(model, optimizer)
    fsdp_model: FSDP

    # all parameters
    # for name, param in fsdp_model.named_parameters():
    #     param: DTensor
    #     print(
    #         f"rank {rank} - param {name} typed as {type(param)}, placement {param.placements}, shaped as {param.shape}"
    #     )
    #     if name == "fc.weight":
    #         assert param.placements[0] == Shard(0)
    #         _device_mesh = param.device_mesh
    #         print(
    #             f"fc.weight device mesh is {_device_mesh}, device type {_device_mesh.device_type}"
    #         )
    #         local_fc_weight = param.redistribute(
    #             placements=[Replicate()],
    #         )
    #         print(f"local fc.weight shape is {local_fc_weight.shape}")

    # print(type(accelerator.unwrap_model(fsdp_model).get_last_layer()))

    print("===========")

    # 2. 模拟输入数据
    # inputs = torch.randn(2, 3, 32, 32).cuda()  # batch_size=2
    # targets = torch.randint(0, 10, (2,)).cuda()

    # # 3. 前向传播
    # outputs = fsdp_model(inputs)
    # loss = nn.CrossEntropyLoss()(outputs, targets)

    # 4. 获取最后一层权重并计算梯度
    last_layer_weight = accelerator.unwrap_model(fsdp_model).get_last_layer()
    print(
        f"rank {rank} - get last layer weight in FSDP typed as {type(last_layer_weight)}, {last_layer_weight}"
    )
    last_layer_weight.redistribute(
        placements=[Replicate()],
    )

    # with FSDP.summon_full_params(fsdp_model):
    #     last_layer_weight = accelerator.unwrap_model(
    #         fsdp_model
    #     ).fc.weight  # 获取完整权重
    #     print(
    #         f"rank {rank} - get last layer weight typed as {type(last_layer_weight)}, shaped as {last_layer_weight.shape}"
    #     )

    #     # 计算loss对最后一层权重的梯度
    #     grads = torch.autograd.grad(
    #         loss,
    #         last_layer_weight,
    #         retain_graph=True,
    #         create_graph=True,
    #         allow_unused=True,
    #     )[0]

    #     print(f"Rank {rank}: Last layer gradients norm:", grads.norm().item())

    dist.destroy_process_group()


if __name__ == "__main__":
    import os

    main()

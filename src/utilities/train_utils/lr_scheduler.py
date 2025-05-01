import math

import torch
from torch.optim import Optimizer


def get_cosine_schedule_reduced_restart_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    reduced_factor: float = 2,
    last_epoch: int = -1,
    min_lr: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Cosine annealing learning rate scheduler with warmup.
    The maximum learning rate will be reduced by reduced_factor after each cycle.

    Args:
        optimizer: Optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        num_cycles: Number of cycles
        reduced_factor: Factor by which maximum learning rate decreases after each cycle
        last_epoch: The index of last epoch
        min_lr: Minimum learning rate to decay to
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cycle_num = int(progress * num_cycles)
        cycle_progress = (progress * num_cycles) % 1.0

        max_lr = 1.0 / (reduced_factor**cycle_num)
        # Modified to decay to min_lr
        return min_lr + (max_lr - min_lr) * (
            0.5 * (1.0 + math.cos(math.pi * cycle_progress))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_reduced_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    reduced_factor: float = 2,
    last_epoch: int = -1,
    min_lr: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Cosine annealing learning rate scheduler with warmup.
    The learning rate slowly increases and decreases within each cycle,
    and the maximum learning rate is reduced by reduced_factor times after each cycle.

    Args:
        optimizer: Optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        num_cycles: Number of cycles
        reduced_factor: Factor by which maximum learning rate decreases after each cycle
        last_epoch: The index of last epoch
        min_lr: Minimum learning rate to decay to
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cycle_num = int(progress * num_cycles)
        cycle_progress = (progress * num_cycles) % 1.0

        max_lr = 1.0 / (reduced_factor**cycle_num)

        if cycle_num == 0:
            # 第一个周期直接从max_lr开始余弦下降
            return min_lr + (max_lr - min_lr) * (
                0.5 * (1.0 + math.cos(math.pi * cycle_progress))
            )
        else:
            # 后续周期从max_lr/2缓慢上升到max_lr再下降
            if cycle_progress < 0.5:
                return min_lr + (max_lr - min_lr) * (
                    0.5 + 0.5 * math.cos(math.pi * (1 - 2 * cycle_progress))
                )
            else:
                return min_lr + (max_lr - min_lr) * (
                    0.5 * (1 + math.cos(math.pi * (2 * cycle_progress - 1)))
                )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


import matplotlib.pyplot as plt


def __plot_lr_schedule():
    # 模拟参数
    num_warmup_steps = 500
    num_training_steps = 5000
    num_cycles = 3
    reduced_factor = 2

    # 创建虚拟优化器
    dummy_optimizer = torch.optim.SGD([torch.nn.Parameter(torch.randn(1))], lr=1.0)

    # 创建调度器
    scheduler = get_cosine_schedule_reduced_with_warmup(
        dummy_optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        reduced_factor=reduced_factor,
        min_lr=0.1,
    )

    # 计算每个step的学习率
    lrs = []
    for step in range(num_training_steps):
        lrs.append(scheduler.get_last_lr()[0])
        dummy_optimizer.step()
        scheduler.step()

    # 绘制曲线
    plt.figure(figsize=(10, 6))
    plt.plot(
        lrs, label=f"Cosine with Warmup (cycles={num_cycles}, factor={reduced_factor})"
    )
    plt.xlabel("Training Steps")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.legend()
    plt.grid(True)

    # 标记warmup结束位置
    plt.axvline(x=num_warmup_steps, color="r", linestyle="--", alpha=0.3)
    plt.text(num_warmup_steps, max(lrs) * 0.8, "Warmup End", rotation=90, alpha=0.7)

    # 标记周期边界
    cycle_steps = int((num_training_steps - num_warmup_steps) / num_cycles)
    for i in range(1, num_cycles + 1):
        x_pos = num_warmup_steps + i * cycle_steps
        plt.axvline(x=x_pos, color="g", linestyle=":", alpha=0.3)
        plt.text(x_pos, max(lrs) * 0.6, f"Cycle {i}", rotation=90, alpha=0.7)

    # plt.show()
    plt.savefig("lr_schedule.png")


# 调用函数绘制曲线
__plot_lr_schedule()

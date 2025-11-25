from itertools import zip_longest

import torch
import torchmetrics
from loguru import logger
from torch import Tensor
from torchmetrics.aggregation import BaseAggregator


def dim_zero_stack(x: Tensor | list[Tensor]) -> Tensor:
    """Stack along the zero dimension."""
    if isinstance(x, torch.Tensor):
        return x

    xs = []
    _prev_shape = None
    for xi in x:
        if xi.numel() == 1 and xi.ndim == 0:
            xi = xi.unsqueeze(0)
        elif xi.ndim != 0:  # per-class scores
            if _prev_shape is None:
                _prev_shape = xi.shape
            else:
                assert _prev_shape == xi.shape, (
                    "Expected all tensors to have the same shape, but got different shapes, "
                    f"got {_prev_shape} and {xi.shape}. Total gathered tensors shaped as {[xi.shape for xi in x]}."
                )
            xi = xi.unsqueeze(0)
        else:
            logger.error(f"Expected each tensor to be a scalar or 1D tensor, got {xi.shape}. ")
            raise ValueError("Expected each tensor to be a scalar or 1D tensor.")
        xs.append(xi)

    return torch.cat(xs, dim=0)


class StackMetrics(BaseAggregator):
    stack_value: list

    def __init__(self, nan_strategy: str = "warn", **kwargs):
        super().__init__(
            fn=dim_zero_stack,
            default_value=[],
            nan_strategy=nan_strategy,
            state_name="stack_value",
            **kwargs,
        )

    def update(self, value: float | Tensor) -> None:
        value, _ = self._cast_and_nan_check_input(value)
        if value.numel():
            self.stack_value.append(value)

    def compute(self) -> Tensor | list:
        if isinstance(self.stack_value, list) and self.stack_value:
            return torch.stack(self.stack_value, dim=0)
        return self.stack_value


class StackMeanMetrics(BaseAggregator):
    mean_value: Tensor
    weight: Tensor

    def __init__(self, nan_strategy: str = "warn", **kwargs):
        super().__init__(
            fn="mean",
            default_value=torch.tensor(0.0, dtype=torch.get_default_dtype()),
            nan_strategy=nan_strategy,
            state_name="mean_value",
            **kwargs,
        )
        self.add_state(
            "weight",
            default=torch.tensor(0.0, dtype=torch.get_default_dtype()),
            dist_reduce_fx="sum",
        )
        self._is_inited = True

    def update(
        self,
        value: float | Tensor | list[float | Tensor],
        weight: float | Tensor | list[float | Tensor] | None = None,
    ) -> None:
        if not isinstance(value, list):
            value = [value]
        if weight is not None and not isinstance(weight, list):
            weight = [weight]
        elif weight is None:
            weight = []
        if len(weight) != 0 and len(value) != len(weight):
            raise ValueError(
                f"If weight is provided, it must have the same length as value. Got {len(value)} and {len(weight)}."
            )

        for v, w in zip_longest(value, weight):
            self._per_sample_update(v, w)

    def _per_sample_update(self, value: float | Tensor, weight: float | Tensor | None = None):
        # broadcast weight to value shape
        if not isinstance(value, Tensor):
            value = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        if weight is None:
            weight = torch.ones_like(value)
        elif not isinstance(weight, Tensor):
            weight = torch.as_tensor(weight, dtype=self.dtype, device=self.device)
        weight = torch.broadcast_to(weight, value.shape)
        value, weight = self._cast_and_nan_check_input(value, weight)
        if value.numel() == 0:
            return

        if self._is_inited:
            self.mean_value = self.mean_value.broadcast_to(value.shape)
            self.weight = self.weight.broadcast_to(weight.shape)
            self._is_inited = False
        else:
            assert self.mean_value.shape == value.shape, (
                f"Expected value shape {self.mean_value.shape}, got {value.shape}."
            )

        self.mean_value = self.mean_value + weight * value  # no sum up, take per-update is one sample
        self.weight = self.weight + weight

    def compute(self) -> Tensor:
        if isinstance(self.mean_value, list) and self.mean_value:
            mean_val = torch.stack(self.mean_value, dim=0)
            mean_val = mean_val / self.weight.expand_as(mean_val)
            return mean_val
        return self.mean_value / self.weight


# * --- Test --- #


def test_stack_metrics(rank: int) -> None:
    """Test StackMetrics with distributed data."""
    metric = StackMetrics()
    # Create test data: each rank gets different values
    for i in range(5):
        # Each update adds a tensor with 2 elements
        tensor_values = torch.tensor([i + rank * 10, i + rank * 10 + 1])
        metric.update(tensor_values)

    result = metric.compute()
    print(f"Rank {rank}: StackMetrics result = {result}")
    print(f"Rank {rank}: StackMetrics shape = {result.shape}")


def test_stack_mean_metrics(rank: int) -> None:
    """Test StackMeanMetrics with distributed data."""
    metric = StackMeanMetrics()
    # Create test data: each rank gets different values
    for i in range(5):
        # Each update adds a tensor with 2 elements
        tensor_values = torch.tensor([i + rank * 10, i + rank * 10 + 1])
        metric.update(tensor_values)

    result = metric.compute()
    print(f"Rank {rank}: StackMeanMetrics result = {result}")
    print(f"Rank {rank}: StackMeanMetrics shape = {result.shape}")


def test_weighted_stack_mean_metrics(rank: int) -> None:
    """Test StackMeanMetrics with weighted updates."""
    metric = StackMeanMetrics()
    # Create test data with different weights
    values = [
        torch.tensor([1.0 + rank, 2.0 + rank]),
        torch.tensor([3.0 + rank, 4.0 + rank]),
        torch.tensor([5.0 + rank, 6.0 + rank]),
    ]
    weights = [
        torch.tensor([0.5, 1.0]),
        torch.tensor([1.5, 2.0]),
        torch.tensor([2.5, 3.0]),
    ]

    for val, weight in zip(values, weights):
        metric.update(val, weight)

    result = metric.compute()
    print(f"Rank {rank}: Weighted StackMeanMetrics result = {result}")
    print(f"Rank {rank}: Weighted StackMeanMetrics shape = {result.shape}")


def setup_process(rank: int, world_size: int) -> None:
    """Setup distributed environment and run tests."""
    import os

    import torch.distributed as dist

    # Set up distributed environment variables
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # Initialize process group
    dist.init_process_group(backend="gloo", init_method="env://")

    print(f"\n=== Rank {rank} started ===")

    # Run different tests
    test_stack_metrics(rank)
    test_stack_mean_metrics(rank)
    test_weighted_stack_mean_metrics(rank)

    print(f"=== Rank {rank} finished ===\n")

    # Clean up
    dist.destroy_process_group()


def run_single_process_test() -> None:
    """Run tests in a single process (non-distributed)."""
    print("=== Single Process Test ===")

    # Test StackMetrics
    print("\n--- Testing StackMetrics ---")
    metric = StackMetrics()
    for i in range(5):
        metric.update(torch.tensor([i, i + 1]))
    result = metric.compute()
    print(f"Single process StackMetrics result: {result}")
    print(f"Single process StackMetrics shape: {result.shape}")

    # Test StackMeanMetrics
    print("\n--- Testing StackMeanMetrics ---")
    metric = StackMeanMetrics()
    for i in range(5):
        metric.update(torch.tensor([i, i + 1]))
    result = metric.compute()
    print(f"Single process StackMeanMetrics result: {result}")
    print(f"Single process StackMeanMetrics shape: {result.shape}")

    print("\n=== Single Process Test Complete ===")


if __name__ == "__main__":
    import torch.multiprocessing as mp

    # First run single process test
    run_single_process_test()

    print("\n" + "=" * 50)
    print("Starting distributed test...")
    print("=" * 50 + "\n")

    # Run distributed test
    world_size = 2
    mp.spawn(setup_process, args=(world_size,), nprocs=world_size)

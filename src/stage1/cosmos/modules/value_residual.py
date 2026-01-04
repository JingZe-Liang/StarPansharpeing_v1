import torch
from torch import Tensor


def mix_value_residual(v: Tensor, v1: Tensor, lamb1: Tensor | float, lamb2: Tensor | float) -> Tensor:
    return lamb1 * v + lamb2 * v1.view_as(v)


class ValueResidualState:
    def __init__(self):
        self.v1: Tensor | None = None

    def reset(self) -> None:
        self.v1 = None

    def mix(self, v: Tensor, lamb1: Tensor | float, lamb2: Tensor | float) -> Tensor:
        if self.v1 is None:
            self.v1 = v
        assert self.v1 is not None
        return mix_value_residual(v, self.v1, lamb1, lamb2)

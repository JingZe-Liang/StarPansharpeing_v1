import torch
import torch.nn as nn
from torch import Tensor


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float | Tensor = 1e-5,
        inplace: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(torch.empty(dim, device=device))
        self.init_values = float(init_values) if not isinstance(init_values, Tensor) else init_values
        self.reset_parameters()  # 确保初始化被调用

    def reset_parameters(self):
        init_val = float(self.init_values) if not isinstance(self.init_values, Tensor) else self.init_values
        nn.init.constant_(self.gamma, init_val)

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

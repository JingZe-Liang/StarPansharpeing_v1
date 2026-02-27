from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch.nn as nn
from torch import Tensor


class TeacherAdapter(ABC):
    def __init__(self, encoder: nn.Module, *, processor: Callable[..., object] | None = None) -> None:
        self.encoder = encoder
        self.processor = processor

    @abstractmethod
    def encode(
        self,
        img: Tensor,
        *,
        get_interm_feats: bool,
        use_linstretch: bool,
        detach: bool,
        repa_fixed_bs: int | None,
    ) -> list[Tensor]:
        raise NotImplementedError

    @abstractmethod
    def forward_features(self, x: Tensor | dict, *, get_interm_feats: bool, detach: bool) -> list[Tensor]:
        raise NotImplementedError

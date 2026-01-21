from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .blocks import build_spatial_block


class Spatial2DNatStage(nn.Module):
    def __init__(
        self,
        in_chans: int,
        embed_dim: list[int],
        depths: list[int],
        cond_width: int | None,
        out_chans: int | None,
        block_type: str = "natblock",
        resblock_cfg: dict[str, Any] | None = None,
        nat_cfg: dict[str, Any] | None = None,
        **block_kwargs: Any,
    ):
        super().__init__()
        self.grad_checkpointing = False
        self._use_condition = cond_width is not None
        resblock_cfg = resblock_cfg or {}
        nat_cfg = nat_cfg or {}
        if block_kwargs:
            if block_type.lower() in {"resblock", "res"}:
                resblock_cfg = {**block_kwargs, **resblock_cfg}
            else:
                nat_cfg = {**block_kwargs, **nat_cfg}

        layers = nn.ModuleList()
        for s, dim in enumerate(embed_dim):
            stage_in_chs = embed_dim[s - 1] if s > 0 else in_chans
            blocks = [
                build_spatial_block(
                    block_type=block_type,
                    in_channels=stage_in_chs if i == 0 else dim,
                    out_channels=dim,
                    cond_channels=cond_width,
                    use_time_block=False,
                    time_embed_dim=None,
                    dropout=0.0,
                    num_groups=32,
                    resblock_cfg=resblock_cfg,
                    nat_cfg=nat_cfg,
                )
                for i in range(depths[s])
            ]
            layers.append(nn.ModuleList(blocks))
        self.layers = layers

        # Output projection
        if out_chans is not None:
            self.out_conv = nn.Conv2d(embed_dim[-1], out_chans, kernel_size=1)
        else:
            self.out_conv = nn.Identity()

    def forward(self, x, cond=None):
        # interpolate the condition
        if cond is not None:
            cond = nn.functional.interpolate(cond, size=x.shape[2:], mode="bilinear", align_corners=False)

        def _apply_block(block: nn.Module, x_in: torch.Tensor, cond_in: torch.Tensor | None) -> torch.Tensor:
            if self.grad_checkpointing and self.training:
                if cond_in is None:
                    return checkpoint(lambda x_: block(x_, None, None), x_in, use_reentrant=False)
                return checkpoint(lambda x_, cond_: block(x_, None, cond_), x_in, cond_in, use_reentrant=False)
            return block(x_in, None, cond_in)

        for stage in self.layers:
            for block in stage:  # type: ignore[not-iterable]
                x = _apply_block(block, x, cond)

        x = self.out_conv(x)
        return x

    def set_grad_checkpointing(self, enable: bool = True):
        self.grad_checkpointing = enable

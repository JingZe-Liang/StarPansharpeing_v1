from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import create_act_layer
from torch import Tensor
import numpy as np

# * --- Functional blocks --- #


def list_sum(x: list) -> Any:
    return sum(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        main: nn.Module,
        shortcut: Optional[nn.Module],
        ##### keep the norm/act and pre/post block distinct
        post_act: Optional[Union[nn.Module, str]] = None,
        post_block: Optional[nn.Module] = None,
        pre_block: Optional[nn.Module] = None,
        pre_norm: Optional[nn.Module] = None,
        #### Main input/output index
        in_main_idx: int | None = None,
        out_main_idx: int | None = None,
        main_out_tuple: bool = False,
    ):
        super(ResidualBlock, self).__init__()

        self.pre_norm = pre_norm
        self.pre_block = pre_block
        self.main = main
        self.shortcut = shortcut
        self.post_act = create_act_layer(post_act) if isinstance(post_act, str) else post_act
        self.post_block = post_block
        self.in_main_idx = in_main_idx if in_main_idx is not None else 0
        self.out_main_idx = out_main_idx if out_main_idx is not None else 0
        self.main_out_tuple = main_out_tuple

    def _forward_pre_layers(self, x: Tensor) -> Tensor:
        """Take the input x and pass through pre-norm and pre-block layers."""
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        if self.pre_block is not None:
            x = self.pre_block(x)
        return x

    def _forward_post_layers(self, x: Tensor) -> Tensor:
        """Take the input x and pass through post-norm and post-block layers."""
        if self.post_act is not None:
            x = self.post_act(x)
        if self.post_block is not None:
            x = self.post_block(x)
        return x

    def forward_main(self, *args):
        """Forward pass through the pre-norm/block, main block, post-act/block"""
        args = list(args)

        # Pre-norm/block
        x = args[self.in_main_idx]
        x = self._forward_pre_layers(x)
        args[self.in_main_idx] = x

        # Main block
        out = self.main(*args)
        return out

    def _take_out_main_x(self, out: Tensor | tuple[Tensor, ...]):
        if self.main_out_tuple:
            assert isinstance(out, (tuple, list)), f"Expected tuple/list output from main block, but got {type(out)}"
            return out[self.out_main_idx]
        else:
            return out

    def _replace_main_x(self, out: Tensor | tuple[Tensor, ...], x: Tensor):
        if self.main_out_tuple:
            assert isinstance(out, (tuple, list)), f"Expected tuple/list output from main block, but got {type(out)}"
            out = list(out)
            out[self.out_main_idx] = x
            return tuple(out)
        else:
            return x

    def forward(self, *args) -> Tensor | tuple[Tensor, ...]:
        # Main + Shortcut
        if self.main is None:
            assert self.shortcut is None, "Identity block do not support shortcut"
            assert not self.main_out_tuple, "Identity block do not support multiple outputs, set it to False"
            out = args[self.in_main_idx]
            return out
        elif self.shortcut is None:
            assert self.main is not None, "Main block cannot be None if no shortcut is provided."
            out = self.forward_main(*args)
            x = self._take_out_main_x(out)
        else:
            x_inp = args[self.in_main_idx]
            out = self.forward_main(*args)
            x = self._take_out_main_x(out)
            # Shortcut connection
            shortcut = self.shortcut(x_inp)
            x = x + shortcut

        # Post-act/block
        x = self._forward_post_layers(x)
        out = self._replace_main_x(out, x)

        return out


class ConditionalBlock(nn.Module):
    """
    Maybe work with EfficientViTBlock with seperate context and local modules.
    """

    def __init__(
        self,
        main: nn.Module | None = None,
        condition_module: nn.Module | None = None,
        premain_module: nn.Module | None = None,
        n_conditions: int = 1,
        condition_types: str = "add",
        process_cond_before: Callable[[Tensor, Tensor], Tensor] | str | None = None,
        dim=1,
    ):
        super().__init__()
        self.main = main if main is not None else nn.Identity()
        self.n_conditions = n_conditions
        self.condition_types = condition_types
        assert condition_types in [
            "add",
            "modulate_3",
            "modulate_2",
            # alias
            "adaln3",
            "adaln2",
        ], f"condition_types must be one of [add, modulate_3, modulate_2, adaln3, adaln2] but got {condition_types}"
        self.dim = dim

        self.premain_module = premain_module if premain_module is not None else nn.Identity()
        self.condition_module = condition_module if condition_module is not None else nn.Identity()
        self.process_cond_before = process_cond_before
        if isinstance(process_cond_before, str):
            assert process_cond_before in ["interpolate_as_x"]

    def _modulate_3(self, x, cond):
        scale, shift, gate = cond.chunk(3, dim=self.dim)
        if self.premain_module is not None:
            x = self.premain_module(x)

        # scale and shift
        x = x * (1 + scale) + shift
        return self.main(x) * gate

    def _modulate_2(self, x, cond):
        scale, shift = cond.chunk(2, dim=self.dim)
        if self.premain_module is not None:
            x = self.premain_module(x)

        # scale and shift
        x = x * (1 + scale) + shift
        return self.main(x)

    def _add(self, x, cond):
        if self.premain_module is not None:
            x = self.premain_module(x)

        x = self.main(x) + cond
        return x

    def _in_module_cond(self, x, cond):
        if self.premain_module is not None:
            x = self.premain_module(x)
        x = self.main(x)
        x = self.condition_module(x, cond)
        return x

    def _forward_condition(self, x, cond: Tensor):
        if self.process_cond_before == "interpolate_as_x":
            cond = torch.nn.functional.interpolate(cond, size=x.shape[2:], mode="bilinear", align_corners=False)
        elif callable(self.process_cond_before):
            cond = self.process_cond_before(x, cond)

        if self.condition_types != "in_module":
            return self.condition_module(cond)
        else:
            return cond

    def forward(self, x: torch.Tensor, condition: torch.Tensor):
        cond = self._forward_condition(x, condition)

        if self.condition_types == "add":
            return self._add(x, cond)
        elif self.condition_types in ("modulate_3", "adaln3"):
            return self._modulate_3(x, cond)
        elif self.condition_types in ("modulate_2", "adaln2"):
            return self._modulate_2(x, cond)
        elif self.condition_types == "in_module":
            return self._in_module_cond(x, cond)
        else:
            raise ValueError(f"Unknown condition_types: {self.condition_types}")


class DAGBlock(nn.Module):
    def __init__(
        self,
        inputs: dict[str, nn.Module],
        merge: str,
        post_input: Optional[nn.Module],
        middle: nn.Module,
        outputs: dict[str, nn.Module],
    ):
        super(DAGBlock, self).__init__()

        self.input_keys = list(inputs.keys())
        self.input_ops = nn.ModuleList(list(inputs.values()))
        self.merge = merge
        self.post_input = post_input

        self.middle = middle

        self.output_keys = list(outputs.keys())
        self.output_ops = nn.ModuleList(list(outputs.values()))

    def forward(self, feature_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        feat = [op(feature_dict[key]) for key, op in zip(self.input_keys, self.input_ops)]
        if self.merge == "add":
            feat = list_sum(feat)
        elif self.merge == "cat":
            feat = torch.concat(feat, dim=1)
        else:
            raise NotImplementedError
        if self.post_input is not None:
            feat = self.post_input(feat)
        feat = self.middle(feat)
        for key, op in zip(self.output_keys, self.output_ops):
            feature_dict[key] = op(feat)
        return feature_dict


class OpSequential(nn.Module):
    def __init__(self, op_list: list[Optional[nn.Module]]):
        super(OpSequential, self).__init__()
        valid_op_list = []
        for op in op_list:
            if op is not None:
                valid_op_list.append(op)
        self.op_list = nn.ModuleList(valid_op_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.op_list:
            x = op(x)
        return x

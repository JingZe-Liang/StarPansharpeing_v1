from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import create_act_layer
from torch import Tensor

# * --- Functional blocks --- #


def list_sum(x: list) -> Any:
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])


class ResidualBlock(nn.Module):
    def __init__(
        self,
        main: Optional[nn.Module],
        shortcut: Optional[nn.Module],
        post_act=None,
        pre_norm: Optional[nn.Module] = None,
        in_main_idx: int = 0,
        out_main_idx: int = 0,
    ):
        super(ResidualBlock, self).__init__()

        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = create_act_layer(post_act)
        self.in_main_idx = in_main_idx
        self.out_main_idx = out_main_idx

    def forward_main(self, *args) -> torch.Tensor:
        if self.pre_norm is None:
            return self.main(*args)
        else:
            x = args[self.in_main_idx]
            args = list(args)
            args[self.in_main_idx] = self.pre_norm(x)
            return self.main(*args)

    def forward(self, *args) -> torch.Tensor:
        # take out the x
        x = args[self.in_main_idx]

        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(*args)
        else:
            res = self.forward_main(*args) + self.shortcut(x)
            if self.post_act:
                if isinstance(res, (list, tuple)):
                    x = res[self.out_main_idx]
                else:
                    x = res
                res = self.post_act(x)
        return res


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
        ], (
            f"condition_types must be one of [add, modulate_3, modulate_2] but got {condition_types}"
        )
        self.dim = dim

        self.premain_module = (
            premain_module if premain_module is not None else nn.Identity()
        )
        self.condition_module = (
            condition_module if condition_module is not None else nn.Identity()
        )
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
            cond = torch.nn.functional.interpolate(
                cond, size=x.shape[2:], mode="bilinear", align_corners=False
            )
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
        elif self.condition_types == "modulate_3":
            return self._modulate_3(x, cond)
        elif self.condition_types == "modulate_2":
            return self._modulate_2(x, cond)
        elif self.condition_types == "in_module":
            return self._in_module_cond(x, cond)
        else:
            raise NotImplementedError(
                'condition_types should be "add", "modulate_2" or "modulate_3"'
            )


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
        feat = [
            op(feature_dict[key]) for key, op in zip(self.input_keys, self.input_ops)
        ]
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

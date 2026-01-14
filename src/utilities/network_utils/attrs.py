import inspect
from typing import TypeVar

import torch.nn as nn
import torch

from ..config_utils import dataclass_from_dict, function_config_to_basic_types
from ..logging import log

DataclassT = TypeVar("DataclassT")


def register_network_init(
    model: type[nn.Module],
    cfg_dataclass: type[DataclassT],
    dataclass_in_name: str = "cfg",
):
    # model __init__ has cfg arg
    sig = inspect.signature(model.__init__)
    if dataclass_in_name not in sig.parameters:
        raise ValueError(f"Model {model} __init__ has no argument named {dataclass_in_name}")

    @function_config_to_basic_types
    def register_init_fn(cls, cfg_overrides: dict | None = None, init_kwargs: dict | None = None):
        cfg = dataclass_from_dict(cfg_dataclass, {} if cfg_overrides is None else cfg_overrides)
        # bind signature
        bindings = {dataclass_in_name: cfg}
        bindings.update({} if init_kwargs is None else init_kwargs)
        bound_args = sig.bind(**bindings)
        bound_args.apply_defaults()
        for name, p in sig.parameters.items():
            if name == dataclass_in_name:
                continue
            if name not in bound_args.arguments and p.default is inspect.Parameter.empty:
                raise ValueError(f"Missing required parameter: {name}")
        return cls(**bound_args.arguments)

    # Set the register_init_fn as model's from_config class method
    setattr(model, "from_config", classmethod(register_init_fn))
    log(f'Set classmethod for Class {model.__name__}: "from_config"')


def may_dynamo_module_hasattr(module: torch._dynamo.OptimizedModule | nn.Module, attr_name: str):
    if isinstance(module, torch._dynamo.OptimizedModule):
        return hasattr(module._orig_mod, attr_name), module._orig_mod
    return hasattr(module, attr_name), module

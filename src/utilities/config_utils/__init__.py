from pathlib import Path

import hydra
from omegaconf import ListConfig, OmegaConf

from ..logging import log, once
from .dict_config import (
    dump_config,
    flatten_dict,
    load_config_file,
    set_struct_recursively,
)
from .to_container import (
    function_config_to_basic_types,
    kwargs_to_basic_types,
    to_object,
    to_object_recursive,
)
from .to_dataclass import dataclass_from_dict, dataclass_to_dict, kwargs_to_dataclass

__all__ = [
    "function_config_to_basic_types",
    "kwargs_to_basic_types",
    "to_object",
    "to_object_recursive",
]

# * --- register new resolvers --- * #


@once
def register_new_resolvers():
    new_solvers = {
        "eval": lambda x: eval(x),
        "glob": lambda x: ListConfig([str(p) for p in Path().rglob(x)]),
        "function": lambda x: hydra.utils.get_method(x),
        "class": lambda x: hydra.utils.get_class(x),
        "list": lambda x: list(x),
        "tuple": lambda x: tuple(x),
    }

    for name, resolver in new_solvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver, replace=True)


register_new_resolvers()

import hydra
from omegaconf import OmegaConf

from ..logging import log_print, once
from .to_container import (
    function_config_to_basic_types,
    kwargs_to_basic_types,
    to_object,
    to_object_recursive,
)
from pathlib import Path

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
        "function": lambda x: hydra.utils.get_method(x),
        "class": lambda x: hydra.utils.get_class(x),
        "list": lambda x: list(x),
        "tuple": lambda x: tuple(x),
    }

    for name, resolver in new_solvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver)

    log_print(
        "[Omegaconf]: Registered new resolvers: eval, function, class, list, tuple",
        "info",
    )


register_new_resolvers()

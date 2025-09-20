from pathlib import Path

import hydra
from omegaconf import ListConfig, OmegaConf
from types import CodeType

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
    "is_compiled_code",
]

# * --- register new resolvers --- * #


def is_compiled_code(obj):
    """Check if object is a compiled code object.

    Parameters
    ----------
    obj : object
        Object to check

    Returns
    -------
    bool
        True if object is a compiled code object, False otherwise
    """
    return isinstance(obj, CodeType)


def compile_code_to_call(exec_string: str):
    code = compile(exec_string, "<inject>", "exec")

    def call_code(globals=None, locals=None, closure=None):
        exec(code, globals, locals, closure=closure)

    return call_code


@once
def register_new_resolvers():
    # fmt: off
    new_solvers = {
        "eval":     lambda x: eval(x),
        "glob":     lambda x: ListConfig([str(p) for p in Path().rglob(x)]),
        "function": lambda x: hydra.utils.get_method(x),
        "class":    lambda x: hydra.utils.get_class(x),
        "list":     lambda x: list(x),
        "tuple":    lambda x: tuple(x),
        "compile":  lambda x: compile(x, "<inject>", "exec"),
    }
    # fmt: on

    for name, resolver in new_solvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver, replace=True)


register_new_resolvers()

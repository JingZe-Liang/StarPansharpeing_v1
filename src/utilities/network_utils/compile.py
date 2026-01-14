from functools import wraps

import torch._inductor.config
import torch._functorch.config
import torch._dynamo
import os
from collections.abc import Callable
from typing import Any, Literal, ParamSpec, TypeVar, overload
from loguru import logger

__all__ = [
    "null_decorator",
    "compile_decorator",
]


P = ParamSpec("P")
R = TypeVar("R")


@overload
def null_decorator(func: Callable[P, R], /) -> Callable[P, R]: ...


@overload
def null_decorator(func: None = None, /, **any_kwargs: Any) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def null_decorator(
    func: Callable[P, R] | None = None,
    /,
    **any_kwargs: Any,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """
    No-op decorator: usable as a decorator or as a plain function.

    - As a decorator (no args): `@null_decorator`
    - As a decorator (args ignored): `@null_decorator(x=1)`
    - As a function: `null_decorator(func)` returns `func`
    """
    if func is not None:
        return func

    def _inner_decorator(inner_func: Callable[P, R]) -> Callable[P, R]:
        return inner_func

    return _inner_decorator


# Compilation decorator

model_compiled_flag = bool(int(os.getenv("MODEL_COMPILED", "1")))
# options
compile_mode: Literal["default", "reduce-overhead", "max-autotune", None] = "default"
compile_full_graph = bool(int(os.getenv("MODEL_COMPILED_FULL_GRAPH", "0")))
epilogue_fusion = True
shape_padding = True

# set tf32 matmul
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def compile_model_or_func(obj: Any | None = None, /, **kwargs: Any) -> Any:
    if obj is None:

        def wrapper(inner_obj: Any) -> Any:
            return compile_model_or_func(inner_obj, **kwargs)

        return wrapper

    global compile_full_graph, shape_padding, epilogue_fusion

    # Default configuration
    compile_kwargs = dict(
        fullgraph=compile_full_graph,
        options={
            "max_autotune": False,
            "triton.cudagraphs": False,
            "shape_padding": shape_padding,
            "epilogue_fusion": epilogue_fusion,
        },
    )

    # Apply global compile_mode if not default
    if compile_mode != "default":
        compile_kwargs["mode"] = compile_mode

    # Override with user specified kwargs
    compile_kwargs.update(kwargs)

    # Resolve conflict: Torch compile forbids both 'mode' and 'options'
    if "mode" in compile_kwargs and compile_kwargs["mode"] is not None:
        compile_kwargs.pop("options", None)

    if isinstance(obj, torch.nn.Module):
        obj.compile(**compile_kwargs)
        return obj
    elif isinstance(obj, Callable):
        return torch.compile(obj, **compile_kwargs)
    else:
        raise ValueError(f"Unsupported type: {type(obj)}")


if model_compiled_flag:
    compile_decorator = compile_model_or_func
    logger.debug("will compile the forward function and disable donated buffer")
    torch._inductor.config.compile_threads = 20
    torch._functorch.config.donated_buffer = False  # for adaptive weighting
    torch._dynamo.config.cache_size_limit = 200  # larger cache size
else:
    compile_decorator = null_decorator
    logger.debug("not compile the forward function")

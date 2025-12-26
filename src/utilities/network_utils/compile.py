from functools import wraps

import torch._functorch.config
import torch._dynamo
import os
from typing import Literal
from loguru import logger

__all__ = [
    "null_decorator",
    "null_decorator_no_any_kwgs",
    "compile_forward_fn",
    "compile_decorator",
]


def null_decorator(**any_kwargs):
    def _inner_decorator(func):
        return func

    return _inner_decorator


def null_decorator_no_any_kwgs(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


# Compilation decorator

model_compiled_flag = bool(int(os.getenv("MODEL_COMPILED", "1")))
# options
compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default"
compile_full_graph = True
epilogue_fusion = True
shape_padding = True

if model_compiled_flag:
    compile_decorator = torch.compile(
        mode=compile_mode,
        fullgraph=compile_full_graph,
        # options={
        #     "max_autotune": False,
        #     "triton.cudagraphs": True,
        #     "shape_padding": shape_padding,
        #     "epilogue_fusion": epilogue_fusion,
        # },
    )
    logger.debug("will compile the forward function and disable donated buffer")
    torch._functorch.config.donated_buffer = False  # for adaptive weighting
    torch._dynamo.config.cache_size_limit = 1_000  # larger cache size
else:
    compile_decorator = null_decorator_no_any_kwgs
    logger.debug("not compile the forward function")

import inspect
from functools import wraps
from typing import Any, Callable

from loguru import logger


def once(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to ensure a function is only executed once.
    """
    has_run = False

    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal has_run
        is_auto = kwargs.pop("auto", True)
        if not has_run or not is_auto:
            has_run = True
            return func(*args, **kwargs)

    return wrapper


def default(x, val):
    return x if x is not None else val


def print_shape_if_raise(func):
    def _print_shape(name, x, var_type: str):
        if hasattr(x, "shape"):
            logger.error(f"{var_type} {name} typed: {type(x)}, shaped: {x.shape}")
        elif isinstance(x, (list, tuple)):
            for i, xi in enumerate(x):
                _print_shape(f"{name}[{i}]", xi, var_type)
        elif name in ("cls", "self"):
            pass
        else:
            logger.error(f"{var_type} {name} typed: {type(x)}, var is {x}")

    def inner(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # bind the args and kwargs to get the var name
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            inspect_args = bound.arguments
            inspect_kwargs = bound.kwargs

            for name, var in inspect_args.items():
                _print_shape(name, var, "Arg")

            for name, var in inspect_kwargs.items():
                _print_shape(name, var, "Kwarg")

            raise RuntimeError from e

    return inner


# * --- test --- * #


def test_print_with_raise():
    import torch as th

    @print_shape_if_raise
    def _test_func(a: th.Tensor, b: tuple[th.Tensor, ...], c: int):
        if a.shape[0] != 1:
            raise ValueError

        if len(b) != 2:
            raise ValueError

        if c != 3:
            raise ValueError

    a = th.randn(1, 3)
    b = (th.randn(2, 3), [th.randn(3, 3), 3])
    c = 3

    _test_func(a, b, c)
    print("case 1 passed")

    _test_func(a, b, 4)  # should raise

    # _test_func(th.randn(3, 3), b, 3)


if __name__ == "__main__":
    test_print_with_raise()

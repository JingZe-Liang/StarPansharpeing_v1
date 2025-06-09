from functools import wraps
from typing import Any, Callable


def once(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to ensure a function is only executed once.
    """
    has_run = False

    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal has_run
        if not has_run:
            has_run = True
            return func(*args, **kwargs)

    return wrapper


def default(x, val):
    return x if x is not None else val

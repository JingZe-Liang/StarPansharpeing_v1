from functools import wraps
from typing import Any, Callable

import torch
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


def dict_round_to_list_str(d: dict, n_round: int = 3, select: list[str] | None = None):
    strings = []
    for k, v in d.items():
        if select is not None and k not in select:
            continue

        if isinstance(v, (float, torch.Tensor)):
            if torch.is_tensor(v):
                if v.numel() > 1:
                    logger.warning(f'logs has non-scalar tensor "{k}", skip it')
                    continue
                v = v.item()
            strings.append(f"{k}: {v:.{n_round}f}")
        else:
            strings.append(f"{k}: {v}")
    return strings

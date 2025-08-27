import inspect
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


# * --- test --- * #


def test_print_with_raise():
    import torch as th

    class A:
        cls_var = "a"

        def __init__(self) -> None:
            self.a = [1, 2]
            self.t = torch.tensor([1, 3])

        @print_info_if_raise
        def may_raise_fn(self, b):
            if b != 3:
                raise ValueError

    @print_info_if_raise
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

    aa = A()
    aa.may_raise_fn(3)
    aa.may_raise_fn(4)  # should raise

    _test_func(a, b, 4)  # should raise

    # _test_func(th.randn(3, 3), b, 3)


if __name__ == "__main__":
    test_print_with_raise()

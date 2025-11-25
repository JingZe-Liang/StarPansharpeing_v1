import dowhen
import lovely_tensors as lt
import torch
from loguru import logger

"""
Function that raises an exception is called.
Lovely-Tensors hook is enabled.
2025-10-19 18:38:29.315 | ERROR    | __main__:<module>:24 - An error has been caught in function '<module>', process 'MainProcess' (2528598), thread 'MainThread' (138072585852736):
Traceback (most recent call last):

> File "/home/user/zihancao/Project/hyperspectral-1d-tokenizer/scripts/tests/test_logger_catch.py", line 25, in <module>
    func_that_raised()
    └ <function func_that_raised at 0x7d91ba1ef1a0>

  File "/home/user/zihancao/Project/hyperspectral-1d-tokenizer/scripts/tests/test_logger_catch.py", line 17, in func_that_raised
    y = 1 / 0 + x  # that will raise an exception
                └ tensor[1, 3, 256] n=768 (3Kb) x∈[-3.326, 3.120] μ=-0.056 σ=0.958

ZeroDivisionError: division by zero
"""


def lt_hook():
    lt.monkey_patch()
    logger.warning("Lovely-Tensors hook is enabled. Make sure only call this function in trackback!")


def func_that_raised():
    print("Function that raises an exception is called.")
    x = torch.randn(1, 3, 256)
    y = 1 / 0 + x  # that will raise an exception
    return y


# patch in logger
dowhen.do(lt_hook).when(logger.catch, "from_decorator = self._from_decorator")

with logger.catch():
    func_that_raised()

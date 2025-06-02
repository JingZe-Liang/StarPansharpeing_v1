import sys
from contextlib import ContextDecorator
from functools import wraps
from typing import Literal

import torch.distributed as dist
from beartype import beartype
from loguru import logger

# Define a type hint for allowed log levels
LogLevel = Literal["debug", "info", "warning", "error", "critical"]
__warn_once_set = set()

# Set the level
logger.level("DEBUG", icon="🔍", color="<blue>")
logger.level("INFO", icon="➡️", color="<light-black>")
logger.level("WARNING", icon="⚠️", color="<yellow><bold>")
logger.level("ERROR", icon="❌", color="<red><bold>")
logger.level("CRITICAL", icon="💥", color="<red><bold>")


# Configure logger
__re_config_logger = True
if __re_config_logger:
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG",
        colorize=True,
        format="{time:HH:mm:ss} - {level.icon} <level>[{level}:{file.name}:{line}]</level> - <level>{message}</level>",
    )
    logger.info("Logger reconfigured", level="info")
    __re_config_logger = False


def is_rank_zero() -> bool:
    """
    Check if the current process is the main process (rank 0).
    This is useful for distributed training scenarios.

    Returns:
        bool: True if the current process is rank 0, False otherwise.
    """
    # Assuming rank 0 is the main process
    if dist.is_initialized():
        return dist.get_rank() == 0
    else:
        return True


@beartype
def log_print(
    msg: str,
    level: LogLevel = "info",
    only_rank_zero: bool = True,
    warn_once: bool = False,
    **kwargs,
) -> None:
    """
    Logs a message using loguru, optionally only on rank 0,
    and ensures the correct caller information (file, line, function) is logged.

    Args:
        msg (str): The message to log.
        level (LogLevel): The logging level ('debug', 'info', 'warning', 'error', 'critical').
                          Defaults to "info".
        only_rank_zero (bool): If True, only log on the rank 0 process in distributed settings.
                               Defaults to True.
        warn_once (bool): If True, only log the message once (as a warning) and skip duplicates.
                          Defaults to False.
    """
    if only_rank_zero and not is_rank_zero():
        return

    if warn_once and msg in __warn_once_set:
        return
    elif warn_once:
        __warn_once_set.add(msg)
        level = "warning"

    logger_with_correct_depth = logger.opt(depth=2, colors=True)
    log_fn = getattr(logger_with_correct_depth, level)

    if only_rank_zero:
        log_fn(msg, **kwargs)
    else:
        rank: int = dist.get_rank() if dist.is_initialized() else 0
        log_fn(f"[Rank {rank}] | {msg}", **kwargs)


class catch_any(ContextDecorator):
    """
    A context manager and decorator that catches any exception raised within its scope or the decorated function,
    logs the exception using the configured logger, and suppresses the exception to prevent it from propagating.

    Usage as a context manager:
        with catch_any():
            # code that may raise exceptions

    Usage as a decorator:
        @catch_any()
        def my_function():
            # code that may raise exceptions

    Exceptions are logged with traceback details, and execution continues after the block or function.
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.opt(exception=(exc_type, exc_val, exc_tb)).error(
                "Exception occurred"
            )
        return True  # Suppress the exception

    def __call__(self, func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            with self:
                return logger.catch(func)(*args, **kwargs)

        return wrapped


if __name__ == "__main__":
    logger.info("this is a log")

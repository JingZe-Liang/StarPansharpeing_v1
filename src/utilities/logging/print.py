import torch.distributed as dist
from loguru import logger
from typing import Literal
from beartype import beartype

# Define a type hint for allowed log levels
LogLevel = Literal["debug", "info", "warning", "error", "critical"]
__warn_once_set = set()


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

    logger_with_correct_depth = logger.opt(depth=2)
    log_fn = getattr(logger_with_correct_depth, level)

    if only_rank_zero:
        log_fn(msg)
    else:
        rank: int = dist.get_rank() if dist.is_initialized() else 0
        log_fn(f"[Rank {rank}] | {msg}")

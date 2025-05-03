import torch.distributed as dist
from loguru import logger
from typing import Literal
from beartype import beartype

# Define a type hint for allowed log levels
LogLevel = Literal["debug", "info", "warning", "error", "critical"]


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
def log_print(msg: str, level: LogLevel = "info", only_rank_zero: bool = True):
    """
    Logs a message using loguru, optionally only on rank 0,
    and ensures the correct caller information (file, line, function) is logged.

    Args:
        msg (str): The message to log.
        level (LogLevel): The logging level ('debug', 'info', 'warning', 'error', 'critical').
                          Defaults to "info".
        only_rank_zero (bool): If True, only log on the rank 0 process in distributed settings.
                               Defaults to True.
    """
    if only_rank_zero and not is_rank_zero():
        return

    # Validate level (already done by type hint, but good for runtime check if needed)
    # level = level.lower() # Type hint ensures lowercase
    # assert level in ["debug", "info", "warning", "error", "critical"], (
    #     f"Invalid log level: {level}. "
    #     f"Choose from 'debug', 'info', 'warning', 'error', or 'critical'."
    # )

    # Use opt(depth=1) to tell Loguru to look 1 frame up the stack
    # for the correct file/line/function information.
    logger_with_correct_depth = logger.opt(depth=2)
    log_fn = getattr(logger_with_correct_depth, level)

    if only_rank_zero:
        log_fn(msg)
    else:
        # Add rank info if logging on all ranks
        rank: int = dist.get_rank() if dist.is_initialized() else 0
        log_fn(f"[Rank {rank}] | {msg}")

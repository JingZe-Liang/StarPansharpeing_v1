import torch.distributed as dist
from loguru import logger


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


def log_print(msg: str, level: str = "info", only_rank_zero: bool = True):
    if only_rank_zero and not is_rank_zero():
        return

    level = level.lower()
    assert level in ["debug", "info", "warning", "error", "critical"], (
        f"Invalid log level: {level}. "
        f"Choose from 'debug', 'info', 'warning', 'error', or 'critical'."
    )
    log_fn = getattr(logger, level)
    if only_rank_zero:
        log_fn(msg)
    else:
        # add rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        log_fn(f"[Rank {rank}] | {msg}")

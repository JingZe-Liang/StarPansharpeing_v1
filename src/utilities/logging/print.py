import re
import sys
from contextlib import ContextDecorator
from functools import wraps
from pathlib import Path
from typing import Literal

import torch.distributed as dist
from beartype import beartype
from loguru import logger
from rich.console import Console

from .functions import once

# Define a type hint for allowed log levels
LogLevel = Literal["debug", "info", "warning", "error", "critical"]
__warn_once_set = set()
__warn_once_pattern_set = set()

__setup_console = (
    False  # rich console, when use this, the markup shold be [ ] not loguru < >.
)
_console = None

__re_config_logger = True

# Set the level
logger.level("DEBUG", icon="🔍", color="<blue>")
logger.level("INFO", icon="⭐", color="<light-black>")
logger.level("WARNING", icon="⚠️", color="<yellow><bold>")
logger.level("ERROR", icon="❌", color="<red><bold>")
logger.level("CRITICAL", icon="💥", color="<red><bold>")


# Setup console
@once
def setup_console():
    global __setup_console, _console

    _console = Console()
    __setup_console = True


if __setup_console:
    setup_console()


# Configure logger
def print_custom_markup(text: str):
    processed_text = text.replace("</", "[/")
    processed_text = processed_text.replace("<", "[")
    processed_text = processed_text.replace(">", "]")

    global _console
    assert _console is not None, (
        "Console is not initialized. Call setup_console() first."
    )
    _console.print(processed_text, markup=True, highlight=False, end="")


@once
def configure_logger(sink=None, level="debug", auto=True):
    global __re_config_logger, _console

    __re_config_logger = False

    if _console is not None:
        # sink = lambda msg: _console.print(msg, markup=True, highlight=False, end="")
        sink = print_custom_markup
    elif sink is None:
        sink = sys.stderr

    logger.remove()
    logger.add(
        sink,
        level=level.upper(),
        enqueue=True,
        colorize=True,
        format=(
            "{time:HH:mm:ss} "
            "- {level.icon} <level>[{level}:{file.name}:{line}]</level>"
            "- <level>{message}</level>"
        ),
    )
    logger.debug("Logger reconfigured")


if __re_config_logger:
    configure_logger()


def set_logger_file(
    file: str | Path | None = None,
    level: LogLevel = "debug",
    add_time: bool = True,
    mode="w",
) -> None:
    log_format_in_file = (
        "<green>[{time:MM-DD HH:mm:ss}]</green> "
        "- <level>[{level}]</level> "
        "- <cyan>{file}:{line}</cyan> - <level>{message}</level>"
    )

    import time

    t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    if file is None:
        Path("tmp/logs").mkdir(parents=True, exist_ok=True)
        file = f"tmp/logs/{t}.log"
    else:
        file = Path(file)
        if file.is_dir():
            if add_time:
                file = file / f"{t}.log"
            else:
                file = file / "log.log"
        else:
            assert file.suffix == ".log", "File must be a .log file"
            if add_time:
                file = file.parent / f"{t}_{file.name}"

        file = file.as_posix()  # Convert to string path

    Path(file).parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        file,
        format=log_format_in_file,
        level=level.upper(),
        enqueue=True,
        rotation="10 MB",
        backtrace=True,
        colorize=False,
        mode=mode,
    )
    log_print(f"Set logger to log to file: {file} with level {level}")


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


def get_dist_rank() -> tuple[bool, int]:
    return (
        dist.is_initialized(),
        dist.get_rank() if dist.is_initialized() else 0,
    )


@beartype
def log_print(
    msg: str,
    level: LogLevel = "info",
    only_rank_zero: bool = True,
    warn_once: bool = False,  # deprecated
    warn_once_pattern: str | None = None,
    once: bool = False,
    stack_level: int = 1,
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
        stack_level (int): The stack level to adjust the depth of the log message.
                           Defaults to 1, which means it will log the caller's information.
    """
    if only_rank_zero and not is_rank_zero():
        return

    log_once = once or warn_once

    if log_once or warn_once_pattern is not None:
        level = "warning" if warn_once else level

        if warn_once_pattern is not None:
            if any(
                re.search(stored_pattern, msg)
                for stored_pattern in __warn_once_pattern_set
            ):
                return
            __warn_once_pattern_set.add(warn_once_pattern)
        else:
            if msg in __warn_once_set:
                return
            __warn_once_set.add(msg)

    logger_with_correct_depth = logger.opt(depth=stack_level + 1, colors=True)
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
    from rich.progress import track

    for i in track(range(10), description="Processing...", console=_console):
        log_print(
            f"This is a warning once message at step {i}",
            warn_once_pattern=r".* at step \d+",
            level="warning",
        )

import inspect
import re
import sys
from contextlib import ContextDecorator
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Literal

import torch.distributed as dist
from beartype import beartype
from loguru import logger
from rich.console import Console

from .functions import default, once

# Define a type hint for allowed log levels
LogLevel = Literal["debug", "info", "warning", "error", "critical"]
__warn_once_set = set()
__warn_once_pattern_set = set()

# Rich console, when use this, the markup shold be [ ] not loguru < >.
__setup_console = False
_console = None

# Configure the logger
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
def configure_logger(
    sink=None,
    level="debug",
    filter=None,
    removed=True,
    auto=True,  # reserved for once decorator
):
    global __re_config_logger, _console

    __re_config_logger = False

    if _console is not None:
        # sink = lambda msg: _console.print(msg, markup=True, highlight=False, end="")
        sink = print_custom_markup
    elif sink is None:
        sink = sys.stderr

    if removed:
        logger.remove()
    handler = logger.add(
        sink,
        level=level.upper(),
        enqueue=False,
        colorize=True,
        format=(
            "{time:HH:mm:ss} "
            "- {level.icon} <level>[{level}] {file.name}:{line}</level>"
            "<green>{extra}</green> "
            "- <level>{message}</level>"
        ),
        filter=filter,
    )
    # logger.debug("Logger reconfigured", re_config=True)

    return handler


if __re_config_logger:
    configure_logger()


def set_logger_file(
    file: str | Path | None = None,
    level: LogLevel = "debug",
    add_time: bool = True,
    mode="w",
    filter=None,
):
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

    handler = logger.add(
        file,
        format=log_format_in_file,
        level=level.upper(),
        enqueue=True,
        rotation="10 MB",
        backtrace=True,
        colorize=False,
        mode=mode,
        filter=filter,
    )
    log_print(
        f"Set logger to log to file: {file} with level {level}, handler id: {handler}",
        level="info",
    )

    return handler


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


def format_extra(record):
    if len(record["extra"]) == 0:
        record["extra"] = ""
        return record

    extras = " ".join(f"{k}={v}" for k, v in record["extra"].items())
    record["extra"] = f" [{extras}]"
    return record


@beartype
def log_print(
    msg: str | Any,
    level: LogLevel = "info",
    only_rank_zero: bool = True,
    warn_once: bool = False,  # deprecated
    warn_once_pattern: str | None = None,
    once: bool = False,
    stack_level: int = 1,
    opt_record: bool = False,
    opt_lazy: bool = False,
    opt_raw: bool = False,
    patch_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    context: dict[str, Any] | None = None,
    **other_context,
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
    if not isinstance(msg, str):
        try:
            msg = str(msg)
        except Exception as e:
            raise RuntimeError(f"Can not cast log message into a string: {e}") from e

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

    if patch_fn is None:
        # e.g., patch_fn = lambda record: r.update({"extra": {"user": "user_id"}})
        # patch_fn = lambda r: r.update({"extra": ""})
        patch_fn = format_extra

    logger_patched = logger.patch(patch_fn)

    logger_with_correct_depth = logger_patched.opt(
        depth=stack_level + 1,
        colors=True,
        record=opt_record,
        lazy=opt_lazy,
        raw=opt_raw,
    )
    log_fn = getattr(logger_with_correct_depth, level)

    if context is None:
        context = {}
    context.update(other_context)

    if only_rank_zero:
        log_fn(msg, **context)
    else:
        rank: int = dist.get_rank() if dist.is_initialized() else 0
        log_fn(msg, rank=rank, **context)


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


def print_info_if_raise(ret_all_stacks_info=False):
    def _print_info(name, x, var_type: str):
        if hasattr(x, "shape"):
            log_print(
                f"[{var_type}] <yellow>{name}</> typed: <green>{type(x)}</>, shaped: <blue>{x.shape}</>"
                + f", device: {x.device}"
                if hasattr(x, "device")
                else "",
            )
        elif isinstance(x, (list, tuple)):
            for i, xi in enumerate(x):
                _print_info(f"{name}[{i}]", xi, var_type)
        elif isinstance(x, dict):
            for n, xi in x.items():
                _print_info(f"{name}['{n}']", xi, var_type)
        elif name in ("cls", "self"):
            vars_from_class = vars(x)
            for k, v in vars_from_class.items():
                _print_info(f"{name}.{k}", v, var_type)
        else:
            log_print(
                f"[{var_type}] <yellow>{name}</> typed: <green>{type(x)}</>, var is <blue>{x}</>",
            )

    def inner(func):
        def inner_(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if not is_rank_zero():
                    return

                _code_ctx = "            return func(*args, **kwargs)\n"
                for stack_i, trace in enumerate(reversed(inspect.trace())):
                    frame = trace[0]

                    # Trace to the current frame
                    if trace.code_context[0] == _code_ctx or (
                        not ret_all_stacks_info and stack_i > 0
                    ):
                        break

                    # Print all variables
                    local_vars = frame.f_locals
                    use_vars = {
                        k: v for k, v in local_vars.items() if not k.startswith("__")
                    }

                    log_print("=" * 30 + f" [stack index: {stack_i} ] " + "=" * 30)
                    log_print(
                        f"file: {frame.f_code.co_filename}:{frame.f_lineno}",
                    )
                    log_print(f"Error occurred in {func.__name__}, local vars:")
                    for name, var in use_vars.items():
                        _print_info(name, var, "local var")
                    log_print("=" * 60 + "\n")

                raise RuntimeError from e

        return inner_

    return inner


if __name__ == "__main__":
    with logger.contextualize(user="test_user"):
        log_print("Hello, World!")

    log_print("This is a debug message", level="debug")

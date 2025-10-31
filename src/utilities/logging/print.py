"""
Print and log functions using loguru.

env:
    COLOR_LOG: 0/1, whether to colorize the log output to shell (default: 1).
    SHELL_LOG_LEVEL: str, the logger's level. (default: "DEBUG").
"""

import inspect
import os
import re
import sys
import time
from contextlib import ContextDecorator
from functools import lru_cache, partial, wraps
from pathlib import Path
from typing import Any, Callable, Dict, Literal, LiteralString, Optional, Union

import torch.distributed as dist
from beartype import beartype
from loguru import logger
from rich.console import Console

from .functions import default, once

type Record = Dict[str, Any]

# Define a type hint for allowed log levels
LogLevel = Literal["trace", "debug", "info", "warning", "error", "critical"] | str
PreservedKeys: list[LiteralString] = [
    "tqdm",
    "once",
    "log_once",
    "warn_once",
    "once_pattern",
    "not_rank0_print",
]
__warn_once_set = set()
__warn_once_pattern_set = set()

# Rich console, when use this, the markup shold be [ ] not loguru < >.
__setup_console = False
_console = None

# Configure the logger
__re_config_logger = True

# Set the level
logger.level("DEBUG", icon="🔍", color="<blue>")
logger.level("INFO", icon="ℹ️ ", color="<light-black>")
logger.level("WARNING", icon="⚠️", color="<yellow><bold>")
logger.level("ERROR", icon="❌", color="<red><bold>")
logger.level("CRITICAL", icon="💥", color="<red><bold>")


def is_true(x):
    return x in (True, 1, "true", "True")


def is_false(x):
    return not is_true(x)


def is_none(x):
    return x in (None, "none", "None")


# Setup console
@once
def setup_console(_auto_=True):
    global __setup_console, _console

    _console = Console()
    __setup_console = True


if __setup_console:
    setup_console()


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


@lru_cache(maxsize=1)
def get_dist_rank() -> tuple[bool, int]:
    return (
        dist.is_initialized(),
        dist.get_rank() if dist.is_initialized() else 0,
    )


def print_custom_markup(text: str):
    processed_text = text.replace("</", "[/")
    processed_text = processed_text.replace("<", "[")
    processed_text = processed_text.replace(">", "]")

    global _console
    assert _console is not None, (
        "Console is not initialized. Call setup_console() first."
    )
    _console.print(processed_text, markup=True, highlight=False, end="")


def rank0_filter(record: Record) -> bool:
    if record["extra"] is None:
        return True
    proc_id = get_dist_rank()[1]
    mannual_proc_print = record["extra"].get("not_rank0_print", False)

    if proc_id == 0 or is_true(mannual_proc_print):
        if mannual_proc_print:
            record["extra"]["proc"] = proc_id
        return True
    return False


def process_id_patcher(record: Record):
    if record["extra"] is None:
        record["extra"] = {}

    record["extra"]["proc"] = get_dist_rank()[1]
    return record


def print_once_filter(record: Record):
    if record["extra"] is None:
        return True

    # flags
    is_once_ = record["extra"].get("once", False)
    is_log_once_ = record["extra"].get("log_once", False)
    is_warn_once_ = record["extra"].get("warn_once", False)
    log_once = is_once_ or is_log_once_ or is_warn_once_
    # pattern
    once_pattern = record["extra"].get("once_pattern", None)

    if log_once or once_pattern is not None:
        record["level"] = "warning" if is_warn_once_ else record["level"]
        msg = record["message"]

        global __warn_once_pattern_set, __warn_once_set

        if once_pattern is not None:
            if any(
                re.search(stored_pattern, msg)
                for stored_pattern in __warn_once_pattern_set
            ):
                # matched in stored patterns or in __once_pattern_set
                # filter out
                return False
            __warn_once_pattern_set.add(once_pattern)
        else:
            if msg in __warn_once_set:
                return False
            __warn_once_set.add(msg)

    # not filter out
    return True


def filter_cat(filters: list[Callable[[Record], bool] | None]):
    def cat_filters(record):
        for f in filters:
            if f is None:
                continue

            if not f(record):
                return False
        return True

    return cat_filters


def log_level_range_filters(
    level_range: tuple[LogLevel, LogLevel],
) -> Callable[[Record], bool]:
    level_order = {
        "TRACE": 10,
        "DEBUG": 20,
        "INFO": 30,
        "WARNING": 40,
        "ERROR": 50,
        "CRITICAL": 60,
    }

    min_level_value = level_order.get(level_range[0].upper(), 0)
    max_level_value = level_order.get(level_range[1].upper(), 60)

    def level_filter(record: Record) -> bool:
        level_name = (
            record["level"].name
            if hasattr(record["level"], "name")
            else record["level"]
        )
        record_level_value = level_order.get(level_name.upper(), 0)
        return min_level_value <= record_level_value <= max_level_value

    return level_filter


def tqdm_undisrupt_print_filter(record: Record):
    if record["extra"] is None:
        return True

    # let added tqdm logger to print, disable the main logger
    # bind tqdm=True, return False
    # is not set, tqdm=False, return True
    return is_false(record["extra"].get("tqdm", False))


def format_extra_patcher(record: Record, preserve_keys=PreservedKeys):
    if record["extra"] is None:
        return record

    if len(record["extra"]) == 0:
        return {}

    if preserve_keys is not None:
        extras_to_msg = {}
        extras_preserve = {}
        for k, v in record["extra"].items():
            if k in preserve_keys:
                extras_preserve[k] = v
            else:
                extras_to_msg[k] = v
    else:
        extras_to_msg = record["extra"]
        extras_preserve = {}
    # Set to record
    record["extra"] = extras_preserve

    if len(extras_to_msg) > 0:
        # Dict extras to msg
        extras_msg = " ".join(f"{k}={v}" for k, v in extras_to_msg.items())
        # Add extra dict string to msg
        record["message"] = f"[{extras_msg}] {record['message']}"

    return record


def add_logger_filtes_(
    only_rank_one: bool,
    main_log_lvl_range: tuple | None,
    add_tqdm_filter: bool,
    add_print_once_filter: bool,
    filters: list[Callable] | Callable | None = None,
):
    # Filter concate
    main_ft: list[Callable[[Record], bool] | None] = []

    # Add level range filter if specified
    if main_log_lvl_range is not None:
        lvl_rng_ft = log_level_range_filters(main_log_lvl_range)
        main_ft.append(lvl_rng_ft)

    # Rank 0 filter
    if only_rank_one:
        main_ft.append(rank0_filter)

    # Add tqdm filter if specified
    if add_tqdm_filter:
        main_ft.append(tqdm_undisrupt_print_filter)

    if add_print_once_filter:
        main_ft.append(print_once_filter)

    # Add custom filter if provided
    if filters is not None:
        if callable(filters):
            main_ft.append(filters)
        else:
            main_ft.extend(filters)

    # Combine all filters
    if len(main_ft) != 0:
        main_log_filter = filter_cat(main_ft)
    else:
        main_log_filter = None

    return main_log_filter


@once
def configure_logger(
    sink=None,
    level: str = os.getenv("SHELL_LOG_LEVEL", "debug"),
    filters: list[Callable] | Callable | None = None,
    removed=True,
    main_log_lvl_range: tuple | None = None,
    only_rank_one=False,
    print_rank_info=False,
    add_print_once_filter=True,
    add_tqdm_filter=False,
    *,
    _auto_=True,  # reserved for once decorator
):
    global __re_config_logger, _console, logger

    __re_config_logger = False

    # * --- Sinks --- #

    if _console is not None:
        # sink = lambda msg: _console.print(msg, markup=True, highlight=False, end="")
        sink = print_custom_markup
    elif sink is None:
        sink = sys.stderr

    is_file = False
    if isinstance(sink, (str, Path)):
        is_file = True

    if removed:
        logger.remove()

    colorize = True
    if "COLOR_LOG" in os.environ:
        colorize = is_true(os.getenv("COLOR_LOG", "1").lower())
    elif is_file:
        colorize = False

    # * --- Filters --- #

    main_log_filter = add_logger_filtes_(
        only_rank_one=only_rank_one,
        main_log_lvl_range=main_log_lvl_range,
        add_tqdm_filter=add_tqdm_filter,
        add_print_once_filter=add_print_once_filter,
        filters=filters,
    )

    # * --- Formatter and logger configuration --- #

    # Main logger
    fmt = (
        "<green>{time:HH:mm:ss}</green> "
        "- {level.icon} <level>[{level:^6}] {file.name}:{line}</level> "
        "- <level>{message}</level>"
    )
    handler = logger.add(
        sink,
        level=level.upper(),
        enqueue=False,
        colorize=colorize,
        format=fmt,
        filter=main_log_filter,
    )

    # * --- Patchers --- #

    # add not preserved extra patcher
    if print_rank_info:
        logger = logger.patch(process_id_patcher)
    logger = logger.patch(format_extra_patcher)

    # Tqdm logger
    if add_tqdm_filter:
        from tqdm import tqdm

        tqdm_logger_filter = add_logger_filtes_(
            only_rank_one=only_rank_one,
            main_log_lvl_range=main_log_lvl_range,
            add_tqdm_filter=False,  # disable the tqdm fileter
            add_print_once_filter=add_print_once_filter,
            filters=[lambda record: "tqdm" in record["extra"]],
        )

        # Add a handler for tqdm-specific logs that uses tqdm.write to avoid conflicts
        logger.add(
            sink=lambda msg: tqdm.write(msg, end=""),
            level=level.upper(),
            filter=tqdm_logger_filter,
            format=fmt,
            colorize=colorize,
            enqueue=False,
        )
        # Only log this message if it doesn't have tqdm binding to avoid infinite recursion
        logger.info("Add tqdm write logger.")

    # Tqdm logger
    if add_tqdm_filter:
        from tqdm import tqdm

        # Add a handler for tqdm-specific logs that uses tqdm.write to avoid conflicts
        logger.add(
            sink=lambda msg: tqdm.write(msg, end=""),
            level=level.upper(),
            filter=lambda record: "tqdm" in record["extra"],
            format=fmt,
            colorize=colorize,
            enqueue=False,
        )
        # Only log this message if it doesn't have tqdm binding to avoid infinite recursion
        logger.info("Add tqdm write logger.")

    return handler


if __re_config_logger:
    configure_logger()


def set_logger_file(
    file: Optional[Union[str, Path]] = None,
    level: LogLevel = os.getenv("FILE_LOG_LEVEL", "debug"),
    add_time: bool = True,
    mode="w",
    filter=None,
    main_log_lvl_range=None,
    add_print_once_filter=True,
):
    global logger

    log_format_in_file = (
        "<green>[{time:MM-DD HH:mm:ss}]</green> "
        "- <level>[{level:^6}]</level> "
        "- <cyan>{file}:{line}</cyan> "
        "- <level>{message}</level>"
    )

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

    file_filter = add_logger_filtes_(
        only_rank_one=False,
        main_log_lvl_range=main_log_lvl_range,
        add_tqdm_filter=False,
        add_print_once_filter=add_print_once_filter,
        filters=filter,
    )

    handler = logger.add(
        file,
        format=log_format_in_file,
        level=level.upper(),
        enqueue=False,
        rotation="10 MB",
        backtrace=True,
        colorize=False,
        mode=mode,
        filter=file_filter,
    )
    logger = logger.patch(format_extra_patcher)

    log_print(
        f"Set logger to log to file: {file} with level {level}, handler id: {handler}",
        level="info",
    )

    return handler


# *==============================================================
# * function calling
# *==============================================================

"""
Usages:
    calling the log_print function or it's alias log function

Or directly calling logger._log or trace, debug, info, warning, error, critical methods,
with record['extra'] values:
    extras:
        - tqdm: used when in a tqdm context, prevent loguru from breaking the progress bar
        - not_rank0_print: used as filter indicator, print all processes even not rank 0
        - once, warn_once, log_once: used to indicate only log once
        - once_pattern: used to indicate only log once with pattern matching
"""


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
    patch_fn: Callable[[Record], Record] | None = None,
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

    if context is None:
        context = {}
    context.update(other_context)

    logger_with_correct_depth = logger.opt(
        depth=stack_level + 1,
        colors=True,
        record=opt_record,
        lazy=opt_lazy,
        raw=opt_raw,
    )
    log_fn = getattr(logger_with_correct_depth, level)

    if only_rank_zero:
        log_fn(msg, **context)
    else:
        rank: int = dist.get_rank() if dist.is_initialized() else 0
        log_fn(msg, rank=rank, **context)


def log(*msg, sep="", **kwargs):
    msg_str = ""
    for i, m in enumerate(msg):
        msg_str = msg_str + m
        if i != len(msg) - 1:
            msg_str = msg_str + sep

    kwargs.setdefault("stack_level", 2)

    log_print(msg_str, **kwargs)


class catch_any(ContextDecorator):
    """
    A context manager and decorator that catches any exception raised within its scope or the decorated function,
    logs the exception using the configured logger, and suppresses the exception to prevent it from propagating.

    This utility is particularly useful for:
    - Handling unexpected errors in non-critical code paths
    - Preventing exceptions from interrupting program flow
    - Debugging by logging exceptions without crashing
    - Graceful error handling in user-facing applications

    Usage as a context manager:
        with catch_any():
            # code that may raise exceptions
            result = risky_operation()
            # continues execution even if risky_operation() fails

    Usage as a decorator:
        @catch_any()
        def my_function():
            # code that may raise exceptions
            return some_value
        # returns None if exception occurs, otherwise returns function result

    Examples:
        # Context manager with file operations
        with catch_any():
            with open('nonexistent_file.txt', 'r') as f:
                content = f.read()

        # Decorator for function that might fail
        @catch_any()
        def fetch_data(url):
            response = requests.get(url)
            return response.json()

        # Nested usage
        @catch_any()
        def process_data():
            with catch_any():
                data = load_external_data()
                return transform_data(data)

    Args:
        func (callable, optional): Function to be decorated when used as a decorator.
                                 When used as a context manager, this should be None.

    Returns:
        catch_any: When used as a decorator, returns the wrapped function.
                  When used as a context manager, returns self for __enter__.

    Notes:
        - All exceptions are caught and logged with full traceback information
        - The exception is suppressed and does not propagate up the call stack
        - When used as a decorator, the decorated function returns None if an exception occurs
        - When used as a context manager, execution continues after the 'with' block
        - Exceptions are logged using the configured loguru logger with ERROR level
        - Use this sparingly as it can hide important errors that should be handled explicitly

    Warning:
        Overuse of this utility can make debugging difficult by masking exceptions.
        Only use it for non-critical operations where failure is acceptable and expected.
    """

    def __enter__(self, func=None):
        self.func = func
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.opt(exception=(exc_type, exc_val, exc_tb)).error(
                "Exception occurred"
            )
        return True  # Suppress the exception

    def __call__(self, func=None):
        if func is None and self.func is None:
            return self
        elif func is None:
            return self

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


# * --- Test --- #


def _test_print(rank, is_mp=False):
    if is_mp:
        dist.init_process_group(
            backend="gloo",
            init_method="tcp://localhost:23456",
            rank=rank,
            world_size=2,
        )
        print("Initialized process group for rank", rank)
        import accelerate

        accelerator = accelerate.Accelerator()
    # Reconfigure logger to avoid conflicts when run as module
    configure_logger(
        level="debug",
        add_tqdm_filter=True,
        removed=True,
        only_rank_one=True,
        print_rank_info=False,
        _auto_=False,
    )

    print("Configured logger")
    # print(f"I am process {rank}")
    logger.info(f"i am process {rank}", not_rank0_print=True)

    # Test tqdm
    from tqdm import tqdm

    for i in tqdm(range(10)):
        time.sleep(0.3)
        logger.bind(tqdm=True).info(f"Processing item {i} in rank {rank}")

    if is_mp:
        accelerator.wait_for_everyone()

    # Print other levels
    logger.debug(f"Debug message from rank {rank}")
    logger.warning(f"Warning message from rank {rank}", once=True)
    logger.warning(f"Warning message from rank {rank}", once=True)  # not print
    logger.error(f"Error message from rank {rank}")

    # Extras
    logger.bind(user="zihan").info("iam zihan")
    logger.info("this is a normal info log", time="2025-10-23")

    if is_mp:
        dist.destroy_process_group()


if __name__ == "__main__":
    """
        python -m src.utilities.logging.print
    """

    import torch.multiprocessing as mp

    mp.spawn(partial(_test_print, is_mp=True), nprocs=2)
    # _test_print(0, False)

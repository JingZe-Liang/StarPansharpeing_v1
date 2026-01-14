import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Literal, ParamSpec, TypeVar

import torch
from rich.console import Console
from rich.table import Table

P = ParamSpec("P")
R = TypeVar("R")


@dataclass(slots=True)
class _TimerStats:
    count: int = 0
    cpu_total_ms: float = 0.0
    cuda_total_ms: float = 0.0

    def update(self, cpu_ms: float, cuda_ms: float) -> None:
        self.count += 1
        self.cpu_total_ms += cpu_ms
        self.cuda_total_ms += cuda_ms


class _NoopContext:
    def __enter__(self) -> "_NoopContext":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False

    def __call__(self, func: Callable[P, R]) -> Callable[P, R]:
        return func


class _TimerContext:
    def __init__(self, recorder: "TimeRecorder", name: str) -> None:
        self._recorder = recorder
        self._name = name
        self._cpu_start: float | None = None
        self._cuda_start: torch.cuda.Event | None = None
        self._cuda_end: torch.cuda.Event | None = None

    def __enter__(self) -> "_TimerContext":
        self._cpu_start = time.perf_counter()
        if self._recorder.use_cuda:
            self._cuda_start = torch.cuda.Event(enable_timing=True)
            self._cuda_end = torch.cuda.Event(enable_timing=True)
            self._cuda_start.record()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        cpu_end = time.perf_counter()
        cpu_ms = 0.0
        if self._cpu_start is not None:
            cpu_ms = (cpu_end - self._cpu_start) * 1000.0

        cuda_ms = 0.0
        if self._cuda_start is not None and self._cuda_end is not None:
            self._cuda_end.record()
            self._cuda_end.synchronize()
            cuda_ms = float(self._cuda_start.elapsed_time(self._cuda_end))

        self._recorder._update(self._name, cpu_ms, cuda_ms)
        return False

    def __call__(self, func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with _TimerContext(self._recorder, self._name):
                return func(*args, **kwargs)

        return wrapper


class TimeRecorder:
    _instance: "TimeRecorder | None" = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs) -> "TimeRecorder":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        *,
        enabled: bool = True,
        use_cuda: bool | None = None,
        console: Console | None = None,
    ) -> None:
        if self.__class__._initialized:
            return
        self.__class__._initialized = True
        self._enabled = enabled
        cuda_available = torch.cuda.is_available()
        if use_cuda is None:
            self._use_cuda = cuda_available
        else:
            self._use_cuda = use_cuda and cuda_available
        self._stats: dict[str, _TimerStats] = {}
        self._console = console or Console()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def use_cuda(self) -> bool:
        return self._enabled and self._use_cuda

    def configure(
        self,
        *,
        enabled: bool | None = None,
        use_cuda: bool | None = None,
        console: Console | None = None,
    ) -> None:
        if enabled is not None:
            self._enabled = enabled
        if use_cuda is not None:
            self._use_cuda = use_cuda and torch.cuda.is_available()
        if console is not None:
            self._console = console

    def record(self, name: str) -> _NoopContext | _TimerContext:
        if not self._enabled:
            return _NoopContext()
        return _TimerContext(self, name)

    def reset(self) -> None:
        self._stats.clear()

    def _update(self, name: str, cpu_ms: float, cuda_ms: float) -> None:
        stats = self._stats.get(name)
        if stats is None:
            stats = _TimerStats()
            self._stats[name] = stats
        stats.update(cpu_ms, cuda_ms)

    def _sort_key(self, stats: _TimerStats, sort_by: Literal["cpu", "cuda", "max"]) -> float:
        if sort_by == "cpu":
            return stats.cpu_total_ms
        if sort_by == "cuda":
            return stats.cuda_total_ms
        return max(stats.cpu_total_ms, stats.cuda_total_ms)

    def print_table(self, *, sort_by: Literal["cpu", "cuda", "max"] = "max", limit: int | None = None) -> None:
        if not self._stats:
            self._console.print("[TimeRecorder] No records.")
            return

        items = sorted(self._stats.items(), key=lambda item: self._sort_key(item[1], sort_by), reverse=True)
        if limit is not None:
            items = items[: max(0, limit)]

        table = Table(title="Time Recorder", show_lines=False)
        table.add_column("Name", justify="left")
        table.add_column("Calls", justify="right")
        table.add_column("CPU total (ms)", justify="right")
        table.add_column("CUDA total (ms)", justify="right")
        table.add_column("CPU avg (ms)", justify="right")
        table.add_column("CUDA avg (ms)", justify="right")

        for name, stats in items:
            calls = max(stats.count, 1)
            cpu_avg = stats.cpu_total_ms / calls
            cuda_avg = stats.cuda_total_ms / calls
            table.add_row(
                name,
                str(stats.count),
                f"{stats.cpu_total_ms:,.3f}",
                f"{stats.cuda_total_ms:,.3f}",
                f"{cpu_avg:,.3f}",
                f"{cuda_avg:,.3f}",
            )

        self._console.print(table)


time_recorder = TimeRecorder()  # global var


if __name__ == "__main__":

    @time_recorder.record("func1")
    def some_func():
        time.sleep(1)

    @time_recorder.record("func2")
    def some_func2():
        time.sleep(2)

    some_func()
    some_func2()
    some_func()

    time_recorder.print_table()

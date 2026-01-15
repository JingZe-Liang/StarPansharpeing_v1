import torch
import torch.nn.functional as F
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Sequence

from .lejepa_aug import AugmentationBase
from src.utilities.train_utils import time_recorder, TimeRecorder


@dataclass(slots=True)
class _ProxyAugPrefetchResult:
    global_views: list[torch.Tensor]
    local_views: list[torch.Tensor]
    done_event: Any | None
    image_size: int


class ProxyAugFuture:
    def __init__(
        self,
        *,
        device: torch.device,
        image_size: int,
        future: Future[_ProxyAugPrefetchResult] | None = None,
        result: _ProxyAugPrefetchResult | None = None,
    ) -> None:
        if (future is None) == (result is None):
            raise ValueError("Exactly one of `future` or `result` must be provided.")

        self._device = device
        self.image_size = image_size
        self._future = future
        self._result = result
        self._event_waited = False

    @classmethod
    def from_future(
        cls,
        future: Future[_ProxyAugPrefetchResult],
        *,
        device: torch.device,
        image_size: int,
    ) -> "ProxyAugFuture":
        return cls(device=device, image_size=image_size, future=future)

    @classmethod
    def from_result(
        cls,
        result: _ProxyAugPrefetchResult,
        *,
        device: torch.device,
        image_size: int,
    ) -> "ProxyAugFuture":
        return cls(device=device, image_size=image_size, result=result)

    def wait(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if self._result is None:
            assert self._future is not None
            self._result = self._future.result()

        if (
            not self._event_waited
            and self._result.done_event is not None
            and self._device.type == "cuda"
            and torch.cuda.is_available()
        ):
            torch.cuda.current_stream(self._device).wait_event(self._result.done_event)
            self._event_waited = True

        return self._result.global_views, self._result.local_views

    def __del__(self) -> None:  # noqa: PLW1641
        try:
            if self._device.type != "cuda" or self._event_waited or not torch.cuda.is_available():
                return

            if self._result is None:
                if self._future is None or not self._future.done():
                    return
                self._result = self._future.result()

            if self._result.done_event is not None:
                torch.cuda.current_stream(self._device).wait_event(self._result.done_event)
                self._event_waited = True
        except Exception:
            return


class ProxyAugManager:
    def __init__(
        self,
        *,
        proxy_aug_async=True,
        proxy_aug_async_cpu=False,
        proxy_aug_pipeline: AugmentationBase | None = None,
    ) -> None:
        self._proxy_aug_async = proxy_aug_async
        self._proxy_aug_async_cpu = proxy_aug_async_cpu
        self._proxy_aug_async_workers = 1
        self._proxy_aug_pipeline = proxy_aug_pipeline
        self._executor: ThreadPoolExecutor | None = None
        self._stream: Any | None = None
        self._time_recorder = time_recorder

    def set_pipeline(self, proxy_aug_pipeline: AugmentationBase | None) -> None:
        self._proxy_aug_pipeline = proxy_aug_pipeline

    def set_time_recorder(self, time_recorder: TimeRecorder | None) -> None:
        self._time_recorder = time_recorder

    @staticmethod
    def _to_view_list(views: torch.Tensor | list[torch.Tensor]) -> list[torch.Tensor]:
        if isinstance(views, torch.Tensor):
            return [views]
        return views

    def _need_aug(self, proxy_tasks: Sequence[str]) -> bool:
        if self._proxy_aug_pipeline is None:
            return False
        return any(t in proxy_tasks for t in ("ibot", "lejepa", "lejepa_latent", "dino_cls"))

    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            max_workers = self._proxy_aug_async_workers
            max_workers = max(1, max_workers)
            self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="proxy_aug")
        return self._executor

    def _get_stream(self, device: torch.device) -> Any:
        if self._stream is None:
            self._stream = torch.cuda.Stream(device=device)
        return self._stream

    def _record(self, name: str):
        if self._time_recorder is None:
            return nullcontext()
        return self._time_recorder.record(name)

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        self._stream = None

    def _run_sync(self, x: torch.Tensor, *, img_size: int) -> _ProxyAugPrefetchResult:
        assert self._proxy_aug_pipeline is not None, "proxy_aug_pipeline is not initialized"
        x_resized = F.interpolate(x, size=img_size, mode="bilinear")
        global_views, local_views = self._proxy_aug_pipeline(x_resized)
        return _ProxyAugPrefetchResult(
            global_views=self._to_view_list(global_views),
            local_views=self._to_view_list(local_views),
            done_event=None,
            image_size=img_size,
        )

    def _run_async(self, x: torch.Tensor, *, img_size: int) -> _ProxyAugPrefetchResult:
        if x.device.type != "cuda" or not torch.cuda.is_available():
            return self._run_sync(x, img_size=img_size)

        assert self._proxy_aug_pipeline is not None, "proxy_aug_pipeline is not initialized"

        device_index = x.device.index
        if device_index is not None:
            torch.cuda.set_device(device_index)

        stream = self._get_stream(x.device)
        done_event: Any = torch.cuda.Event()
        with torch.cuda.stream(stream):
            x_resized = F.interpolate(x, size=img_size, mode="bilinear")
            global_views, local_views = self._proxy_aug_pipeline(x_resized)
            global_views_list = self._to_view_list(global_views)
            local_views_list = self._to_view_list(local_views)
            done_event.record(stream)

        return _ProxyAugPrefetchResult(
            global_views=global_views_list,
            local_views=local_views_list,
            done_event=done_event,
            image_size=img_size,
        )

    def maybe_prefetch(
        self,
        x: torch.Tensor,
        *,
        proxy_tasks: Sequence[str],
        img_size: int = 224,
    ) -> ProxyAugFuture | None:
        if not self._need_aug(proxy_tasks):
            return None

        async_enabled = self._proxy_aug_async
        cpu_async_enabled = self._proxy_aug_async_cpu

        if x.device.type != "cuda" and not (async_enabled and cpu_async_enabled):
            return ProxyAugFuture.from_result(
                self._run_sync(x, img_size=img_size), device=x.device, image_size=img_size
            )

        if not async_enabled:
            return ProxyAugFuture.from_result(
                self._run_sync(x, img_size=img_size), device=x.device, image_size=img_size
            )

        executor = self._get_executor()
        future = executor.submit(self._run_async, x, img_size=img_size)
        return ProxyAugFuture.from_future(future, device=x.device, image_size=img_size)

    def get_views(
        self,
        x_resized: torch.Tensor,
        *,
        img_size: int,
        proxy_tasks: Sequence[str],
        proxy_aug_future: ProxyAugFuture | None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if not self._need_aug(proxy_tasks):
            raise ValueError(f"Proxy tasks do not need aug, but `get_views` is called: {list(proxy_tasks)!r}")

        if proxy_aug_future is not None and proxy_aug_future.image_size == img_size:
            return proxy_aug_future.wait()

        assert self._proxy_aug_pipeline is not None, "proxy_aug_pipeline is not initialized"
        global_views, local_views = self._proxy_aug_pipeline(x_resized)
        return self._to_view_list(global_views), self._to_view_list(local_views)

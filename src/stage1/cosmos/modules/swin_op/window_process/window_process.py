import importlib
from types import ModuleType

import torch

_window_process_impl: ModuleType | None = None
WINDOW_PROCESS_BACKEND = "uninitialized"


def _load_impl(backend: str) -> tuple[ModuleType, str]:
    if backend == "triton":
        from .kernel import window_process_triton as triton_impl

        return triton_impl, "triton"
    if backend == "cuda_ext":
        cuda_impl = importlib.import_module("swin_window_process")
        return cuda_impl, "cuda_ext"
    if backend == "auto":
        try:
            cuda_impl = importlib.import_module("swin_window_process")
            return cuda_impl, "cuda_ext"
        except Exception:
            from .kernel import window_process_triton as triton_impl

            return triton_impl, "triton"
    raise ValueError(f"Unsupported window process backend: {backend}")


def set_window_process_backend(backend: str) -> str:
    global _window_process_impl, WINDOW_PROCESS_BACKEND
    impl, resolved = _load_impl(backend)
    _window_process_impl = impl
    WINDOW_PROCESS_BACKEND = resolved
    return WINDOW_PROCESS_BACKEND


def get_window_process_backend() -> str:
    return WINDOW_PROCESS_BACKEND


set_window_process_backend("auto")


class WindowProcess(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, B, H, W, C, shift_size, window_size):
        output = _window_process_impl.roll_and_window_partition_forward(input, B, H, W, C, shift_size, window_size)

        ctx.B = B
        ctx.H = H
        ctx.W = W
        ctx.C = C
        ctx.shift_size = shift_size
        ctx.window_size = window_size
        return output

    @staticmethod
    def backward(ctx, grad_in):
        B = ctx.B
        H = ctx.H
        W = ctx.W
        C = ctx.C
        shift_size = ctx.shift_size
        window_size = ctx.window_size

        grad_out = _window_process_impl.roll_and_window_partition_backward(grad_in, B, H, W, C, shift_size, window_size)
        return grad_out, None, None, None, None, None, None, None


class WindowProcessReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, B, H, W, C, shift_size, window_size):
        output = _window_process_impl.window_merge_and_roll_forward(input, B, H, W, C, shift_size, window_size)

        ctx.B = B
        ctx.H = H
        ctx.W = W
        ctx.C = C
        ctx.shift_size = shift_size
        ctx.window_size = window_size

        return output

    @staticmethod
    def backward(ctx, grad_in):
        B = ctx.B
        H = ctx.H
        W = ctx.W
        C = ctx.C
        shift_size = ctx.shift_size
        window_size = ctx.window_size

        # grad_out = ctx.saved_tensors[0]
        # grad_out = torch.zeros((B, H, W, C), dtype=dtype).cuda()
        grad_out = _window_process_impl.window_merge_and_roll_backward(grad_in, B, H, W, C, shift_size, window_size)
        return grad_out, None, None, None, None, None, None, None

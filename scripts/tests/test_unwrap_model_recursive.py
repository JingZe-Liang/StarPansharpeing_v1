import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utilities.network_utils.network_loading import unwrap_model_recursive


class _Inner(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.compiled_linear = torch.compile(torch.nn.Linear(4, 4), backend="eager")


class _Outer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.inner = _Inner()


def test_unwrap_model_recursive_unwraps_compiled_submodules() -> None:
    model = _Outer()
    assert isinstance(model.inner.compiled_linear, torch._dynamo.OptimizedModule)
    orig = model.inner.compiled_linear._orig_mod

    unwrapped = unwrap_model_recursive(model, keep_submodule_compiled=False)
    assert isinstance(unwrapped, torch.nn.Module)
    assert unwrapped.inner.compiled_linear is orig
    assert not isinstance(unwrapped.inner.compiled_linear, torch._dynamo.OptimizedModule)


def test_unwrap_model_recursive_unwraps_top_level_compiled_module() -> None:
    model = torch.compile(_Outer(), backend="eager")
    assert isinstance(model, torch._dynamo.OptimizedModule)

    unwrapped = unwrap_model_recursive(model, keep_submodule_compiled=False)
    assert isinstance(unwrapped, torch.nn.Module)
    assert not isinstance(unwrapped, torch._dynamo.OptimizedModule)

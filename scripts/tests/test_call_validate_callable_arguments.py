import sys
from pathlib import Path

import pytest


sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utilities.func.call import _validate_callable_arguments


def test_validate_filters_extra_kwargs() -> None:
    def func(a: int, b: int = 0) -> int:
        return a + b

    is_valid, filtered = _validate_callable_arguments(
        func,
        args=(1,),
        kwargs={"b": 2, "extra": 3},
        func_name="func",
        print_warning=False,
    )

    assert is_valid is True
    assert filtered == {"b": 2}


def test_validate_missing_required_keyword_only_is_invalid() -> None:
    def func(a: int, *, k: int) -> int:
        return a + k

    is_valid, filtered = _validate_callable_arguments(
        func,
        args=(1,),
        kwargs={},
        func_name="func",
        print_warning=False,
    )

    assert is_valid is False
    assert filtered == {}


def test_validate_positional_only_cannot_be_passed_by_kw() -> None:
    def func(a: int, /, b: int) -> int:
        return a + b

    is_valid, filtered = _validate_callable_arguments(
        func,
        args=(),
        kwargs={"a": 1, "b": 2},
        func_name="func",
        print_warning=False,
    )

    assert is_valid is False
    assert filtered == {"b": 2}


def test_validate_accepts_var_keyword_keeps_all_kwargs() -> None:
    def func(a: int, **kwargs: int) -> tuple[int, dict[str, int]]:
        return a, kwargs

    is_valid, filtered = _validate_callable_arguments(
        func,
        args=(1,),
        kwargs={"x": 1, "y": 2},
        func_name="func",
        print_warning=False,
    )

    assert is_valid is True
    assert filtered == {"x": 1, "y": 2}


def test_validate_too_many_positional_args_is_invalid() -> None:
    def func(a: int) -> int:
        return a

    is_valid, filtered = _validate_callable_arguments(
        func,
        args=(1, 2),
        kwargs={},
        func_name="func",
        print_warning=False,
    )

    assert is_valid is False
    assert filtered == {}

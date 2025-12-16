from collections.abc import Iterable
from functools import wraps
from typing import Any

from beartype import beartype
from easydict import EasyDict
from omegaconf import DictConfig, ListConfig, OmegaConf


def to_object(config: Any):
    """
    Convert a DictConfig or ListConfig to a Python object.
    """
    if isinstance(config, (DictConfig, ListConfig)):
        return OmegaConf.to_object(config)
    else:
        return config


def to_object_recursive(config: Iterable):
    """
    Recursively convert a DictConfig or ListConfig to a Python object.
    """
    # if is iterable, check if is a DictConfig or ListConfig
    if isinstance(config, (dict, DictConfig)):
        return {k: to_object_recursive(v) for k, v in config.items()}
    elif isinstance(config, (list, ListConfig)):
        return [to_object_recursive(item) for item in config]

    return config


def to_easydict_recursive(config: Any):
    """
    Recursively convert a DictConfig or ListConfig to a EasyDict object.

    For dictionaries with non-string keys, keeps them as regular dictionaries
    since EasyDict only supports string keys.
    """
    # if is iterable, check if is a DictConfig or ListConfig
    if isinstance(config, (dict, DictConfig)):
        # Check if all keys are strings - if so, use EasyDict, otherwise use regular dict
        if all(isinstance(k, str) for k in config.keys()):
            # Create empty EasyDict and populate it properly
            result = EasyDict()
            # Clear any default content
            result.clear()
            for k, v in config.items():
                # Process the value recursively
                processed_value = to_easydict_recursive(v)
                # Use __setitem__ directly to avoid EasyDict's __setattr__ conversion
                super(EasyDict, result).__setitem__(k, processed_value)
                # Also set in __dict__ for attribute access
                result.__dict__[k] = processed_value
            return result
        else:
            # For non-string keys, keep as regular dict but still recurse on values
            return {k: to_easydict_recursive(v) for k, v in config.items()}
    elif isinstance(config, (list, ListConfig)):
        return [to_easydict_recursive(item) for item in config]

    return config


def kwargs_to_basic_types(
    kwargs: DictConfig | ListConfig | dict | list | None,
    easydict_type: bool = False,
):
    """
    Convert a dictionary or list of dictionaries to basic types.
    """
    if easydict_type:
        return to_easydict_recursive(kwargs) if kwargs is not None else None
    else:
        return to_object_recursive(kwargs) if kwargs is not None else None


def function_config_to_basic_types(func):
    """
    handle the hydra or omegaconf config objects that are passed to a function
    and convert them to basic types (dict, list, str, int, float, etc.).
    This is useful for functions that expect basic types as arguments
    and may receive DictConfig or ListConfig objects from Hydra or OmegaConf.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Convert args
        new_args = [kwargs_to_basic_types(arg) for arg in args]

        # Convert kwargs
        new_kwargs = kwargs_to_basic_types(kwargs)
        return func(*new_args, **new_kwargs)

    return wrapper


def function_config_to_easy_dict(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        new_args = [kwargs_to_basic_types(arg, easydict_type=True) for arg in args]
        new_kwargs = kwargs_to_basic_types(kwargs, easydict_type=True)

        return func(*new_args, **new_kwargs)

    return wrapper


def function_config_to_basic_types_hint_check(func):
    """
    Decorator that converts Hydra/OmegaConf config objects to basic Python types
    and applies beartype runtime type checking to the decorated function.

    This decorator performs two operations:
    1. Converts DictConfig and ListConfig objects in function arguments to
       standard Python dict and list objects recursively
    2. Applies beartype runtime type checking to ensure arguments match
       the function's type annotations

    Args:
        func: The function to be decorated. Should have proper type annotations
              for beartype checking to work effectively.

    Returns:
        A wrapper function that automatically converts config objects and
        performs type checking before calling the original function.

    Raises:
        beartype.roar.BeartypeCallHintViolation: If type annotations are violated
        after config conversion.

    Example:
        >>> from omegaconf import (
        ...     DictConfig,
        ...     ListConfig,
        ... )
        >>> @function_config_to_basic_types_hint_check
        ... def process_data(
        ...     config: dict,
        ...     items: list[
        ...         int
        ...     ],
        ... ) -> str:
        ...     return f"Config keys: {list(config.keys())}, Items: {len(items)}"
        >>> config = DictConfig(
        ...     {"key": "value"}
        ... )
        >>> items = ListConfig(
        ...     [1, 2, 3]
        ... )
        >>> result = process_data(
        ...     config, items
        ... )  # Automatically converted and type-checked

    Note:
        This decorator combines the functionality of config conversion and
        beartype checking. The function must have proper type annotations
        for the beartype checking to be effective.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Convert args
        new_args = [kwargs_to_basic_types(arg) for arg in args]

        # Convert kwargs
        new_kwargs = kwargs_to_basic_types(kwargs)

        # Create a beartype-checked version of the function
        checked_func = beartype(func)

        return checked_func(*new_args, **new_kwargs)

    return wrapper


if __name__ == "__main__":
    # Example usage
    config = DictConfig({"key": "value", "list": [1, 2, 3], "anydict": {"a": 1, "b": 2}, "int_dict": {1: 2, 3: 4}})
    # print(type(to_object(config)["list"]))  # <class 'list'>
    # print(to_object_recursive(config))  # {'key': 'value', 'list': [1, 2, 3]}

    config = to_easydict_recursive(config)

    print(type(config))
    print(type(config.list))
    print(type(config.int_dict))

    # @function_config_to_basic_types
    # @function_config_to_easy_dict
    # def func(**kwargs):
    #     for k, v in kwargs.items():
    #         print(k, type(v))

    # func(key=config.key, lst=config.list)
    # func(anydict=config.anydict)

    # @function_config_to_basic_types_hint_check
    # def func(key: str, lst: tuple | list):
    #     print(key, type(key))
    #     print(lst, type(lst))

    # func(config.key, lst=config.list)

    # config = ListConfig([1, 2, 3])
    # print(to_object(config))  # [1, 2, 3]
    # print(to_object_recursive(config))  # [1, 2, 3]

    # config = [1, 2, 3, ListConfig([1, 2, 3])]
    # print(
    #     type(to_object(config)[-1])
    # )  # [1, 2, 3, [1, 2, 3]], but last element is a ListConfig
    # print(type(to_object_recursive(config)[-1]))  # [1, 2, 3, [1, 2, 3]]

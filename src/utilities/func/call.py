from loguru import logger
from collections.abc import Callable
from typing import Any

import inspect


def _validate_callable_arguments(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    func_name: str,
    print_warning: bool = True,
) -> tuple[bool, dict[str, Any]]:
    """
    Validate arguments against a callable's signature.

    This function checks if the provided positional and keyword arguments
    are compatible with the callable's parameter signature.

    Parameters
    ----------
    func : Callable
        The function/method to validate arguments against
    args : tuple
        Positional arguments to validate
    kwargs : dict
        Keyword arguments to validate
    func_name : str
        Name of the function for warning messages
    print_warning : bool, default=True
        Whether to print warning messages for invalid arguments

    Returns
    -------
    Tuple[bool, dict]
        A tuple containing:
        - bool: True if arguments are valid, False otherwise
        - dict: Filtered kwargs with only accepted parameters

    Notes
    -----
    This function provides centralized argument validation for the
    maybe_call utility, avoiding code duplication.
    """
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError) as e:
        if print_warning:
            logger.warning(f"Cannot get signature for {func_name}: {e}")
        return False, {}

    parameters = sig.parameters

    # If the callable accepts **kwargs, keep all keyword arguments.
    accepts_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    if accepts_var_keyword:
        filtered_kwargs = dict(kwargs)
    else:
        # Only keep keyword-passable parameters; drop POSITIONAL_ONLY even if name matches.
        allowed_keyword_names = {
            p.name
            for p in parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keyword_names}

    try:
        sig.bind(*args, **filtered_kwargs)
    except TypeError as e:
        if print_warning:
            logger.warning(f"{func_name} argument validation failed: {e}")
        return False, filtered_kwargs

    return True, filtered_kwargs


def maybe_call(
    cls: object = None,
    func: str | Callable | None = None,
    print_warning: bool = True,
    *args,
    **kwargs,
) -> Any:
    """
    Safely call a class or function with optional method name.

    This function provides a flexible interface for calling classes,
    functions, or class methods with proper argument validation.

    Parameters
    ----------
    cls : object, optional
        The class object to call methods from. If None, only function calling is used.
    func : str | Callable | None, optional
        The function to call. Can be:
        - A class: will be instantiated with provided args
        - A callable: will be called directly with provided args
        - A string: will be used as method name on cls
        - None: no function to call
    print_warning : bool, default=True
        Whether to print warning messages for invalid operations
    *args : tuple
        Positional arguments to pass to the callable
    **kwargs : dict
        Keyword arguments to pass to the callable

    Returns
    -------
    Any
        Result of the function call, or None if call failed

    Raises
    ------
    TypeError
        If class is not callable when only class is provided
    AttributeError
        If method name doesn't exist on class when both are provided

    Notes
    -----
    This function follows the project's error handling approach
    by avoiding try-catch blocks and using explicit validation.
    """
    # Case 1: Only class is provided, no func
    if cls is not None and func is None:
        if not callable(cls):
            if print_warning:
                logger.warning(f"Class {cls} is not callable")
            return None
        return cls(*args, **kwargs)

    # Case 2: Only function is provided, no class
    elif cls is None and func is not None:
        # If func is a class, treat it as case 1
        if inspect.isclass(func):
            return func(*args, **kwargs)

        # If func is a callable (function/method)
        elif callable(func):
            # Validate arguments using the helper function
            func_name = getattr(func, "__name__", "anonymous_function")
            is_valid, filtered_kwargs = _validate_callable_arguments(
                func, args, kwargs, f"Function {func_name}", print_warning
            )

            if not is_valid:
                return None

            return func(*args, **filtered_kwargs)

        else:
            if print_warning:
                logger.warning(f"Function {func} is not callable")
            return None

    # Case 3: Both class and function are provided
    elif cls is not None and func is not None:
        # func must be a string (method name)
        if not isinstance(func, str):
            if print_warning:
                logger.warning(
                    f"When both cls and func are provided, func must be a string method name, got {type(func)}"
                )
            return None

        # Check if the method exists on the class
        if not hasattr(cls, func):
            if print_warning:
                cls_name = getattr(cls, "__name__", str(cls))
                logger.warning(f"Class {cls_name} does not have method '{func}'")
            return None

        # Get the method from the class
        method = getattr(cls, func)

        # Ensure the method is callable
        if not callable(method):
            if print_warning:
                cls_name = getattr(cls, "__name__", str(cls))
                logger.warning(f"Attribute '{func}' of {cls_name} is not callable")
            return None

        # Validate arguments using the helper function
        cls_name = getattr(cls, "__name__", str(cls))
        is_valid, filtered_kwargs = _validate_callable_arguments(
            method, args, kwargs, f"Method {cls_name}.{func}", print_warning
        )

        if not is_valid:
            return None

        return method(*args, **filtered_kwargs)

    # Case 4: Neither class nor function is provided
    else:
        if print_warning:
            logger.warning("Both cls and func are None, nothing to call")
        return None


def extract_needed_kwargs(kwargs: dict, cls: Callable | type, include_default: bool = False) -> dict:
    """
    Extracts the subset of `kwargs` that match the parameters of a given class's __init__ method or a function.

    If a parameter is not provided in `kwargs` but has a default value, the default is used.
    Missing required parameters will raise a ValueError.

    Args:
        kwargs (dict): A dictionary of keyword arguments to filter.
        cls (type or function): A class or function whose signature is used to extract the needed kwargs.

    Returns:
        dict: A dictionary containing only the relevant keyword arguments.

    Raises:
        AssertionError: If `cls` is a class without an __init__ method.
        ValueError: If a required argument is missing or an unsupported type is passed.
    """
    needed_kwargs = {}
    if inspect.isclass(cls):
        assert hasattr(cls, "__init__"), f"{cls} does not have an __init__ method."
        sig = inspect.signature(cls.__init__)
    elif inspect.isfunction(cls):
        sig = inspect.signature(cls)
    else:
        raise ValueError(f"Expected a class or function, got {type(cls)}.")

    for param in sig.parameters.values():
        if param.name == "self":
            continue
        if param.name in kwargs:
            needed_kwargs[param.name] = kwargs[param.name]
        elif include_default and param.default is not inspect.Parameter.empty:
            needed_kwargs[param.name] = param.default
        # else:
        #     raise ValueError(
        #         f"Missing required argument '{param.name}' for {cls.__name__}."
        #     )
    return needed_kwargs

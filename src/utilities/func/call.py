from loguru import logger
from typing import Callable, Any, Tuple

import inspect


def _validate_callable_arguments(
    func: Callable,
    args: tuple,
    kwargs: dict,
    func_name: str,
    print_warning: bool = True,
) -> Tuple[bool, dict]:
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

    func_params = sig.parameters

    # Filter kwargs to only include parameters that the function accepts
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in func_params.keys()}

    # Check if positional arguments are acceptable
    param_count = len(func_params)
    args_count = len(args)

    # Get the number of required positional parameters
    required_params = sum(
        1
        for param in func_params.values()
        if param.default == inspect.Parameter.empty
        and param.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    )

    # Validate positional argument count
    if args_count < required_params:
        if print_warning:
            logger.warning(
                f"{func_name} requires at least {required_params} positional arguments, but got {args_count}"
            )
        return False, filtered_kwargs

    if args_count > param_count:
        # Check if function accepts *args
        has_var_args = any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in func_params.values())
        if not has_var_args:
            if print_warning:
                logger.warning(f"{func_name} accepts at most {param_count} positional arguments, but got {args_count}")
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

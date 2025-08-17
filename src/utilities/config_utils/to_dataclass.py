import inspect
from typing import TypeVar


T = TypeVar("T")


def kwargs_to_dataclass(dc: type[T], **kwargs) -> T:
    """Convert keyword arguments to a dataclass instance.

    Args:
        dc (type[T]): The target dataclass type.
        **kwargs: The keyword arguments to set as attributes.
            Only arguments that match the dataclass fields will be used.
            Extra arguments will be ignored.
            Supports fields with default values and default_factory.

    Returns:
        T: An instance of the target dataclass with the specified attributes.
            Fields not provided in kwargs will use their default values if available.
    """
    # Get the signature of the dataclass
    sig = inspect.signature(dc)

    # Filter kwargs to only include parameters that exist in the dataclass
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

    # Bind the filtered arguments to the signature
    bound_args = sig.bind(**filtered_kwargs)
    bound_args.apply_defaults()

    # Create and return the dataclass instance
    return dc(**bound_args.arguments)

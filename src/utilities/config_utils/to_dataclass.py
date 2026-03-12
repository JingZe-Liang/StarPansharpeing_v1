import inspect
from typing import Any, Type, TypeVar

from omegaconf import DictConfig, OmegaConf
from typing_extensions import deprecated
from dataclasses import asdict

T = TypeVar("T")


def dataclass_from_dict(cls: type[T], data: dict | DictConfig, strict: bool = True) -> T:
    """
    Converts a dictionary to a dataclass instance, recursively for nested structures.
    """
    base = OmegaConf.structured(cls)
    OmegaConf.set_struct(base, strict)
    override = OmegaConf.create(data)
    return OmegaConf.to_object(OmegaConf.merge(base, override))  # type: ignore


def dataclass_from_dict_config(cls: type[T], cfg: DictConfig, strict: bool = True) -> T:
    """
    Converts a DictConfig to a dataclass instance, recursively for nested structures.
    """
    base = OmegaConf.structured(cls)
    OmegaConf.set_struct(base, strict)
    return OmegaConf.to_object(OmegaConf.merge(base, cfg))  # type: ignore


def dataclass_to_dict(dataclass_instance: T) -> dict[str, Any]:
    """
    Converts a dataclass instance to a dictionary, recursively for nested structures.
    """
    if isinstance(dataclass_instance, dict):
        return dataclass_instance

    return OmegaConf.to_container(  # type: ignore
        OmegaConf.structured(dataclass_instance), resolve=True
    )


@deprecated("Use `dataclass_from_dict` instead.")
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

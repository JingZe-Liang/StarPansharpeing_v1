from collections.abc import Iterable
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf


def to_object(config: Any) -> dict | list:
    """
    Convert a DictConfig or ListConfig to a Python object.
    """
    if isinstance(config, (DictConfig, ListConfig)):
        return OmegaConf.to_object(config)
    else:
        return config
    # elif isinstance(config, (list, dict)):
    #     return config
    # else:
    #     raise ValueError(f"Unknown config type: {type(config)}")


def to_object_recursive(config: Iterable) -> dict | list:
    """
    Recursively convert a DictConfig or ListConfig to a Python object.
    """
    # if is iterable, check if is a DictConfig or ListConfig
    if isinstance(config, (dict, DictConfig)):
        return {k: to_object_recursive(v) for k, v in config.items()}
    elif isinstance(config, (list, ListConfig)):
        return [to_object_recursive(item) for item in config]

    return config


def kwargs_to_basic_types(
    kwargs: DictConfig | ListConfig | dict | list | None,
) -> dict | list | None:
    """
    Convert a dictionary or list of dictionaries to basic types.
    """

    return to_object_recursive(kwargs) if kwargs is not None else None


if __name__ == "__main__":
    # Example usage
    config = DictConfig({"key": "value", "list": [1, 2, 3]})
    print(type(to_object(config)["list"]))  # <class 'list'>
    print(to_object_recursive(config))  # {'key': 'value', 'list': [1, 2, 3]}

    config = ListConfig([1, 2, 3])
    print(to_object(config))  # [1, 2, 3]
    print(to_object_recursive(config))  # [1, 2, 3]

    config = [1, 2, 3, ListConfig([1, 2, 3])]
    print(
        type(to_object(config)[-1])
    )  # [1, 2, 3, [1, 2, 3]], but last element is a ListConfig
    print(type(to_object_recursive(config)[-1]))  # [1, 2, 3, [1, 2, 3]]

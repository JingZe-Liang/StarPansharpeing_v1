from omegaconf import DictConfig, ListConfig, OmegaConf
from typing import Any


def to_object(config: Any) -> dict | list:
    """
    Convert a DictConfig to a Python object.
    """
    if isinstance(config, (DictConfig, ListConfig)):
        return OmegaConf.to_object(config)
    elif isinstance(config, (list, dict)):
        return config
    else:
        raise ValueError(f"Unknown config type: {type(config)}")

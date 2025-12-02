from typing import Any

import yaml
from easydict import EasyDict as edict
from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf

from .to_dataclass import dataclass_from_dict


def set_struct_recursively(cfg, strict: bool = True):
    # Set struct mode for the current level
    OmegaConf.set_struct(cfg, strict)

    # Traverse through nested dictionaries and lists
    if isinstance(cfg, DictConfig):
        for key, value in cfg.items():
            if isinstance(value, (DictConfig, ListConfig)):
                set_struct_recursively(value, strict)
    elif isinstance(cfg, ListConfig):
        for item in cfg:
            if isinstance(item, (DictConfig, ListConfig)):
                set_struct_recursively(item, strict)


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_config_file[T](config_file, dataclass_cls: type[T] | None = None) -> T | edict | DictConfig | object:
    """Load a config file and return a dataclass, Omegaconf DictConfig, or EasyDict."""
    try:
        config = OmegaConf.to_container(OmegaConf.load(config_file), resolve=True)
        if dataclass_cls is None:
            return config
        return dataclass_from_dict(dataclass_cls, config)  # type: ignore
    except Exception as e:
        logger.warning(
            f"Error loading config file {config_file}: {e}. Try with PyYaml UnsafeLoader, ignore the dataclass"
        )

        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)

        return edict(config)


def dump_config(config, path, log_config=True):
    yaml_dump = OmegaConf.to_yaml(OmegaConf.structured(config))
    with open(path, "w") as f:
        if log_config:
            logger.info("Using the following config for this run:")
            logger.info(yaml_dump)
        f.write(yaml_dump)


def set_defaults(cfg: dict | None, defaults: dict[str, Any], use_edict=True) -> dict | edict:
    """set defaults for dict/edict config"""
    if cfg is None:
        cfg = {}
    for k, v in defaults.items():
        cfg.setdefault(k, v)
    return edict(cfg) if use_edict else cfg

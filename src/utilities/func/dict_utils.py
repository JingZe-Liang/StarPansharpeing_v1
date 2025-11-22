from collections import defaultdict
from typing import Any, Literal

from easydict import EasyDict as edict


def keys_in_dict(
    keys: str | list[str],
    d: dict[str, Any],
    mode: Literal["any", "all"] = "any",
) -> bool:
    """
    Check if keys are present in dictionary.

    Args:
        keys: Single key or list of keys to check
        d: Dictionary to check against
        mode: "any" - return True if any key exists (default)
              "all" - return True if all keys exist

    Returns:
        bool: True if keys exist according to mode, False otherwise
    """
    keys = [keys] if isinstance(keys, str) else keys

    if not keys:
        return True

    if mode == "any":
        # Return True if at least one key exists
        return any(k in d for k in keys)
    elif mode == "all":
        # Return True if all keys exist
        return all(k in d for k in keys)
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'any' or 'all'")


def prefix_dict_keys(d: dict[str, Any], prefix: str, sep=".") -> dict[str, Any]:
    """
    Add a prefix to all keys in a dictionary.

    Args:
        d: Input dictionary
        prefix: Prefix string to add to each key
        sep: Separator between prefix and original key (default: ".")

    Returns:
        dict: New dictionary with prefixed keys
    """
    return {f"{prefix}{sep}{k}": v for k, v in d.items()}


def parse_dict_keys(d: dict, sep=".") -> dict[str, dict[str, Any]]:
    """
    Parse dictionary keys with a separator into a nested dictionary.
    Args:
        d: Input dictionary with keys containing separators
        sep: Separator string used in keys (default: ".")
    Returns:
        dict: Nested dictionary with top-level keys as prefixes and values as sub-dictionaries

    """

    out = defaultdict(dict)
    prefixes = []
    for k, v in d.items():
        prefix, orig_k = k.split(sep, 1)
        if prefix not in prefixes:
            prefixes.append(prefix)
            out = defaultdict(dict)
        out[prefix][orig_k] = v
    return out


def set_defaults(cfg: dict | None, defaults: dict[str, Any], use_edict=True):
    """set defaults for dict/edict config"""
    if cfg is None:
        cfg = edict() if use_edict else {}
    for k, v in defaults.items():
        cfg.setdefault(k, v)
    return edict(cfg) if use_edict else dict(cfg)

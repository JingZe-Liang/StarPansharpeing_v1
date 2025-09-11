from typing import Any, Literal


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

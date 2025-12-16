import zipfile
from functools import wraps
from itertools import chain
from pathlib import Path
from typing import Any, Callable, cast
import torch
from loguru import logger


def once(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to ensure a function is only executed once.
    """
    has_run = False

    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal has_run
        is_auto = kwargs.pop("_auto_", True)
        if not has_run or not is_auto:
            has_run = True
            return func(*args, **kwargs)

    return wrapper


def default(x, val):
    return x if x is not None else val


def dict_round_to_list_str(d: dict, n_round: int = 3, select: list[str] | None = None):
    strings = []
    for k, v in d.items():
        if select is not None and k not in select:
            continue

        if isinstance(v, (float, torch.Tensor)):
            if torch.is_tensor(v):
                if v.numel() > 1:
                    logger.warning(f'logs has non-scalar tensor "{k}", skip it')
                    continue
                v = v.item()
            strings.append(f"{k}: {v:.{n_round}f}")
        else:
            strings.append(f"{k}: {v}")
    return strings


def zip_code_into_dir(
    save_dir: str | Path,
    code_dir: list[str | Path] | str | Path,
    include_patterns: list[str] | None = ["*.py", "*.yaml", "*.yml", "json"],
):
    """
    Zip code files from specified directories into a zip file.

    Parameters
    ----------
    save_dir : str
        Directory where the zip file will be saved
    code_dir : list[str] | str
        Directory or directories to search for code files
    include_patterns : list[str] | None
        File patterns to include (default: ["*.py", "*.yaml", "*.yml", "json"])
    """
    include_patterns = include_patterns if include_patterns is not None else ["*.py", "*.yaml", "*.yml", "json"]

    # Convert to list if single directory
    if isinstance(code_dir, (str, Path)):
        code_dir = [code_dir]
    code_dir = cast(list[str | Path], code_dir)

    # Collect all code files
    all_code_files: list[Path] = []
    for code_f in code_dir:
        for pattern in include_patterns:
            all_code_files.extend(Path(code_f).rglob(pattern))

    logger.info(f"Logging code files into an zip file, found {len(all_code_files)} files")

    save_dir = Path(save_dir)
    assert save_dir.exists(), f"Error: {save_dir} does not exist"
    zip_path = save_dir / "code.zip"
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for code_file in all_code_files:
            # Use relative path to avoid deep directory structure in zip
            arcname = str(code_file)
            zipf.write(code_file, arcname=arcname)

    logger.info(f"Code files zipped into {zip_path}")


def get_python_pkg_env(file: str | None = None) -> str:
    """
    Get the current Python package environment as a string.
    """
    import sys
    import pkg_resources

    packages = sorted(
        [(dist.project_name, dist.version) for dist in pkg_resources.working_set],
        key=lambda x: x[0].lower(),
    )
    env_str = "\n".join([f"{name}=={version}" for name, version in packages])
    if file is not None:
        with open(file, "w") as f:
            f.write(env_str)
    return env_str


if __name__ == "__main__":
    # zip_code_into_dir("tmp/", ["./src/", "./configs/"])
    print(get_python_pkg_env())

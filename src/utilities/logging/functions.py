import zipfile
from functools import wraps
from itertools import chain
from pathlib import Path
from typing import Any, Callable, cast, Literal, cast
import torch
from loguru import logger
import wandb
from tqdm import tqdm
from torch.distributed.tensor import DTensor


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


def log_any_into_writter(
    log_type: Literal["metric", "image", "grad_norm_per_param", "grad_norm_sum"],
    writter: dict[str, Any],
    logs: dict[str, Any],
    step: int | None,
    **kwargs,
):
    assert log_type in [
        "metric",
        "image",
        "grad_norm_per_param",
        "grad_norm_sum",
    ], "log_type must be one of [metric, image, grad_norm_per_param, grad_norm_sum]"
    if step is None:
        from ..train_utils import StepsCounter

        try:
            # StepsCounter is a singleton, it should be initialized in the trainer
            step = StepsCounter()["train"]
        except (ValueError, KeyError, AttributeError):
            # fallback if not initialized or "train" not found
            step = 0
    step = cast(int, step)

    # Early exit if no writers and no collective operations expected
    # Collective operations (like all-gather for DTensor) are expected for grad_norm if distributed
    is_distributed = torch.distributed.is_initialized()
    if not writter and not (is_distributed and log_type in ("grad_norm_per_param", "grad_norm_sum")):
        return

    def _any_writter_log(names: list[str], logs: dict[str, Any]):
        for name in names:
            if name not in writter:
                continue

            if log_type == "image":
                if name == "tensorboard":
                    writter["tensorboard"].log_images(logs, step=step, dataformats="HWC")
                elif name == "wandb":
                    writter["wandb"].log({k: wandb.Image(v) for k, v in logs.items()}, step=step)
                elif name == "swanlab":
                    import swanlab

                    writter["swanlab"].log(
                        {k: swanlab.Image(v, file_type="jpg") for k, v in logs.items()},
                        step=step,
                    )
            else:
                writter[name].log(logs, step=step)

    def _watch_model():
        assert "model" in logs, "model name must be in logs"
        model: torch.nn.Module = logs.pop("model")
        # take out the grad of norms
        model_cls_n = model.__class__.__name__
        norms = {}
        _n_params_sumed = 0
        if log_type == "grad_norm_sum":
            norms[f"{model_cls_n}_grad_norm"] = 0.0

        is_main = not is_distributed or torch.distributed.get_rank() == 0
        for n, p in tqdm(
            model.named_parameters(),
            desc="logging grad norms",
            leave=False,
            disable=not is_main,
        ):
            if p.grad is not None:
                from ..network_utils import safe_dtensor_operation

                _grad = safe_dtensor_operation(p.grad)
                _grad_norm = (_grad.data**2).sum() ** 0.5

                if log_type == "grad_norm_per_param":
                    norms[f"{model_cls_n}/{n}"] = _grad_norm
                elif log_type == "grad_norm_sum":
                    norms[f"{model_cls_n}_grad_norm"] += _grad_norm
                    _n_params_sumed += 1
                else:
                    raise ValueError(f"Unknown log_type {log_type}")

        # Mean the gradient norm
        if log_type == "grad_norm_sum" and _n_params_sumed > 0:
            norms[f"{model_cls_n}_grad_norm"] /= _n_params_sumed

        return norms

    if log_type in ("metric", "image"):
        _any_writter_log(list(writter.keys()), logs)

    elif log_type in ("grad_norm_per_param", "grad_norm_sum"):
        model_stats = _watch_model()
        _any_writter_log(list(writter.keys()), model_stats)

    else:
        raise NotImplementedError(f"Unknown log_type {log_type}")


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

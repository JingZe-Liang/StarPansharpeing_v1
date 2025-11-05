import re

import torch as th
from loguru import logger


def get_parameters_module_frozen(
    model: th.nn.Module,
    frozen_module: list[str] | str,
    with_name=True,
    no_grad_required=True,
):
    """Get trainable parameters from a model excluding specified frozen modules.

    This function extracts trainable parameters from a PyTorch model while excluding
    parameters from specified frozen modules. It supports regular expressions for
    flexible module name matching and can optionally disable gradients for frozen
    parameters.

    Args:
        model (th.nn.Module): The PyTorch model to extract parameters from.
        frozen_module (list[str] | str): Module name(s) or pattern(s) to exclude from parameter extraction.
            Supports regular expressions for flexible pattern matching.
            Parameters from modules matching these patterns will not be included in the result.
        with_name (bool, optional): Whether to return parameters with their names.
            If True, returns a dictionary with parameter names as keys.
            If False, returns a list of parameter values only.
            Defaults to True.
        no_grad_required (bool, optional): Whether to set requires_grad=False for frozen parameters.
            If True, disables gradients for parameters in frozen modules.
            If False, frozen parameters are still returned in the model but not in the result.
            Defaults to True.

    Returns:
        dict[str, torch.Tensor] | list[torch.Tensor]: Trainable parameters.
            If with_name is True, returns a dictionary with parameter names as keys
            and parameter tensors as values.
            If with_name is False, returns a list of parameter tensors.
            Only includes parameters that are not in frozen modules and have requires_grad=True.

    Note:
        - Parameters in frozen modules are excluded from the result regardless of their requires_grad status
        - If no_grad_required is True, frozen parameters will have their requires_grad set to False
        - The function uses regex matching, so special regex characters in module names should be escaped
    """
    trainable_ps: dict = {}
    if isinstance(frozen_module, str):
        frozen_module = [frozen_module]

    for n, p in model.named_parameters():
        if (
            any(re.match(pattern, n) for pattern in frozen_module)
            or not p.requires_grad
        ):
            logger.debug(f"[Params]: skip the param: <u>{n}</u>, {no_grad_required=}")
            if no_grad_required:
                p.requires_grad_(False)
            continue
        else:
            trainable_ps[n] = p

    if with_name:
        return trainable_ps
    else:
        return list(trainable_ps.values())


def get_model_learnable_params(model: th.nn.Module, with_name=True):
    trainable_ps = {}

    for n, p in model.named_parameters():
        if p.requires_grad:
            trainable_ps[n] = p
            # logger.debug(f"Parameter {n} is learnable")

    if with_name:
        return trainable_ps
    else:
        return list(trainable_ps.values())


# * --- partials --- #

import functools

get_parameters_encoder_frozen = functools.partial(
    get_parameters_module_frozen, frozen_module="encoder"
)

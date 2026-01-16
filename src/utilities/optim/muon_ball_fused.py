from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
import re
from typing import Any, cast

import torch
from torch.optim.optimizer import Optimizer, ParamsT

from ..config_utils import function_config_to_basic_types
from .emerging_optimizers.orthogonalized_optimizers.muon_ball import compute_muon_ball_update
from .emerging_optimizers.orthogonalized_optimizers.spectral_ball_utils import (
    compute_target_radius,
    get_spectral_ball_scale_factor,
)


@dataclass(frozen=True, slots=True)
class MuonBallFusedParamGroups:
    muonball: dict[str, Any]
    oned: dict[str, Any]


def _iter_named_parameters(
    named_params: Iterable[tuple[str, torch.nn.Parameter]] | dict[str, torch.nn.Parameter],
) -> Iterable[tuple[str, torch.nn.Parameter]]:
    if isinstance(named_params, dict):
        yield from named_params.items()
        return
    yield from named_params


def _compile_patterns(patterns: Iterable[str]) -> list[re.Pattern[str]]:
    return [re.compile(pat) for pat in patterns]


def _should_ignore_param(name: str, ignore_patterns: list[re.Pattern[str]]) -> bool:
    return any(pat.search(name) is not None for pat in ignore_patterns)


def _apply_weight_decay_inplace(
    p: torch.Tensor,
    grad: torch.Tensor,
    *,
    lr: float,
    weight_decay: float,
    weight_decay_method: str,
) -> None:
    if weight_decay == 0.0:
        return

    if weight_decay_method == "decoupled":
        p.add_(p, alpha=-weight_decay * lr)
    elif weight_decay_method == "independent":
        p.add_(p, alpha=-weight_decay)
    elif weight_decay_method == "l2":
        grad.add_(p, alpha=weight_decay)
    else:
        raise ValueError(f"Invalid weight_decay_method: {weight_decay_method}")


@function_config_to_basic_types
def split_muonball_oned_params(
    named_params: Iterable[tuple[str, torch.nn.Parameter]] | dict[str, torch.nn.Parameter],
    ignored_keys_for_muonball: tuple[str, ...] | list[str] = (),
    oned_param_algo: str = "adamw",
) -> MuonBallFusedParamGroups:
    if oned_param_algo not in ("adamw", "lion"):
        raise ValueError(f"oned_param_algo must be one of: adamw, lion. Got: {oned_param_algo}")

    muonball_params: list[torch.nn.Parameter] = []
    oned_params: list[torch.nn.Parameter] = []

    ignore_patterns = _compile_patterns(ignored_keys_for_muonball)
    for name, p in _iter_named_parameters(named_params):
        if not p.requires_grad:
            continue

        if p.ndim == 2 and not _should_ignore_param(name, ignore_patterns):
            muonball_params.append(p)
        else:
            oned_params.append(p)

    muonball_group: dict[str, Any] = {"params": muonball_params, "algorithm": "muonball"}
    oned_group: dict[str, Any] = {"params": oned_params, "algorithm": oned_param_algo}
    return MuonBallFusedParamGroups(muonball=muonball_group, oned=oned_group)


class MuonBallFused(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        *,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        # MuonBall defaults (2D)
        momentum_beta: float = 0.9,
        use_nesterov: bool = True,
        weight_decay_method: str = "decoupled",
        fp32_matmul_prec: str = "medium",
        power_iteration_steps: int = 10,
        msign_steps: int = 5,
        radius_mode: str = "spectral_mup",
        scale_mode: str = "align_adamw_rms",
        muon_type: str = "small",
        retract_mode: str = "hard",
        retract_alpha: float = 0.05,
        # AdamW/Lion defaults (1D or ignored)
        betas: tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-8,
    ) -> None:
        if power_iteration_steps < 1:
            raise ValueError(f"power_iteration_steps must be at least 1, got {power_iteration_steps}")
        if msign_steps < 1:
            raise ValueError(f"msign_steps must be at least 1, got {msign_steps}")
        if radius_mode not in ("spectral_mup", "identity"):
            raise ValueError(f"Invalid radius_mode: {radius_mode}, must be one of: spectral_mup, identity")
        if retract_mode not in ("hard", "dynamic"):
            raise ValueError(f"Invalid retract_mode: {retract_mode}, must be one of: hard, dynamic")
        if weight_decay_method not in ("decoupled", "independent", "l2"):
            raise ValueError(
                f"Invalid weight_decay_method: {weight_decay_method}, must be one of: decoupled, independent, l2"
            )

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "algorithm": "muonball",
            # MuonBall
            "momentum_beta": momentum_beta,
            "use_nesterov": use_nesterov,
            "weight_decay_method": weight_decay_method,
            "fp32_matmul_prec": fp32_matmul_prec,
            "power_iteration_steps": power_iteration_steps,
            "msign_steps": msign_steps,
            "muon_type": muon_type,
            "radius_mode": radius_mode,
            "scale_mode": scale_mode,
            "retract_mode": retract_mode,
            "retract_alpha": retract_alpha,
            # AdamW/Lion
            "betas": betas,
            "eps": eps,
            "step": 0,
        }
        super().__init__(params, defaults)

    @classmethod
    @function_config_to_basic_types
    def create_muonball_optimizer(
        cls,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]] | dict[str, torch.nn.Parameter],
        ignored_keys_for_muonball: tuple[str, ...] | list[str] = (),
        *,
        oned_param_algo: str = "adamw",
        muonball_params_defaults: dict[str, Any] | None = None,
        oned_params_defaults: dict[str, Any] | None = None,
        **defaults: Any,
    ) -> MuonBallFused:
        groups = split_muonball_oned_params(
            named_parameters,
            ignored_keys_for_muonball=ignored_keys_for_muonball,
            oned_param_algo=oned_param_algo,
        )

        param_groups: list[dict[str, Any]] = []
        if groups.muonball["params"]:
            g = dict(groups.muonball)
            if muonball_params_defaults:
                g.update(muonball_params_defaults)
            param_groups.append(g)
        if groups.oned["params"]:
            g = dict(groups.oned)
            if oned_params_defaults:
                g.update(oned_params_defaults)
            param_groups.append(g)

        return cls(param_groups, **defaults)

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = cast(float, closure())

        for group in self.param_groups:
            group["step"] += 1
            algo = group.get("algorithm", "muonball")
            if algo == "muonball":
                self._step_muonball_group(group)
            elif algo == "adamw":
                self._step_adamw_group(group)
            elif algo == "lion":
                self._step_lion_group(group)
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

        return loss

    def _step_muonball_group(self, group: dict[str, Any]) -> None:
        lr = float(group["lr"])
        momentum_beta = float(group["momentum_beta"])
        use_nesterov = bool(group["use_nesterov"])
        power_iteration_steps = int(group["power_iteration_steps"])
        msign_steps = int(group["msign_steps"])
        radius_mode = cast(str, group["radius_mode"])
        scale_mode = cast(str, group["scale_mode"])
        retract_mode = cast(str, group["retract_mode"])
        retract_alpha = float(group["retract_alpha"])
        weight_decay_method = cast(str, group.get("weight_decay_method", "decoupled"))
        muon_type = cast(str, group.get("muon_type", "small"))

        wd_mult = float(group.get("wd_mult", 1.0))
        weight_decay = float(group.get("weight_decay", 0.0)) * wd_mult

        for p in group["params"]:
            if p.grad is None:
                continue
            if p.ndim != 2:
                raise ValueError("MuonBallFused muonball group only supports 2D parameters.")

            grad = p.grad
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(grad)

            exp_avg = state["momentum_buffer"]

            _apply_weight_decay_inplace(
                p,
                grad,
                lr=lr,
                weight_decay=weight_decay,
                weight_decay_method=weight_decay_method,
            )

            exp_avg.lerp_(grad, 1.0 - momentum_beta)
            if use_nesterov:
                grad_eff = grad.lerp(exp_avg, momentum_beta)
            else:
                grad_eff = exp_avg

            target_radius = compute_target_radius(shape=tuple(p.shape), radius_mode=radius_mode)

            update, _, _ = compute_muon_ball_update(
                W=p.data,
                M=grad_eff,
                target_radius=target_radius,
                power_iteration_steps=power_iteration_steps,
                msign_steps=msign_steps,
                retract_mode=retract_mode,
                retract_alpha=retract_alpha,
                current_lr=lr,
                muon_type=muon_type,
            )

            scale_factor = get_spectral_ball_scale_factor(p.shape[0], p.shape[1], mode=scale_mode)
            p.add_(update, alpha=-lr * scale_factor)

    def _step_adamw_group(self, group: dict[str, Any]) -> None:
        lr = float(group["lr"])
        beta1, beta2 = cast(tuple[float, float], group.get("betas", (0.9, 0.99)))
        eps = float(group.get("eps", 1e-8))
        wd_mult = float(group.get("wd_mult", 1.0))
        weight_decay = float(group.get("weight_decay", 0.0)) * wd_mult

        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)

            state["step"] += 1
            step = int(state["step"])
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            if weight_decay != 0.0:
                p.add_(p, alpha=-lr * weight_decay)

            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            step_size = lr * math.sqrt(bias_correction2) / bias_correction1

            denom = exp_avg_sq.sqrt().add_(eps)
            p.addcdiv_(exp_avg, denom, value=-step_size)

    def _step_lion_group(self, group: dict[str, Any]) -> None:
        lr = float(group["lr"])
        beta1, beta2 = cast(tuple[float, float], group.get("betas", (0.9, 0.99)))
        wd_mult = float(group.get("wd_mult", 1.0))
        weight_decay = float(group.get("weight_decay", 0.0)) * wd_mult

        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state["exp_avg"] = torch.zeros_like(p)

            exp_avg = state["exp_avg"]

            if weight_decay != 0.0:
                p.mul_(1.0 - lr * weight_decay)

            update = exp_avg.clone().lerp_(grad, 1.0 - beta1).sign_()
            p.add_(update, alpha=-lr)
            exp_avg.lerp_(grad, 1.0 - beta2)

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
import importlib
import math
from typing import Any, Literal

import torch
from torch.optim.optimizer import Optimizer

from ..config_utils import function_config_to_basic_types


MaskMode = Literal["magma", "skipupdate"]


@dataclass(slots=True)
class MagmaMaskConfig:
    mode: MaskMode = "magma"
    survival_prob: float = 0.5
    tau: float = 2.0
    ema_beta: float = 0.9
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("magma", "skipupdate"):
            raise ValueError(f"Invalid mode: {self.mode}. Expected one of: magma, skipupdate.")
        if not 0.0 < self.survival_prob <= 1.0:
            raise ValueError(f"survival_prob must be in (0, 1], got: {self.survival_prob}")
        if self.tau <= 0.0:
            raise ValueError(f"tau must be > 0, got: {self.tau}")
        if not 0.0 <= self.ema_beta < 1.0:
            raise ValueError(f"ema_beta must be in [0, 1), got: {self.ema_beta}")


@dataclass(slots=True)
class _MaskState:
    s_t: float | None = None
    grad_ema: torch.Tensor | None = None


@dataclass(slots=True)
class _ForeachBucket:
    params: list[torch.Tensor]
    before: list[torch.Tensor]
    alpha: list[float]


class MagmaSkipUpdateWrapper(Optimizer):
    def __init__(self, base_optimizer: Optimizer, mask_config: MagmaMaskConfig | None = None) -> None:
        self.base_optimizer = base_optimizer
        self.mask_config = MagmaMaskConfig() if mask_config is None else mask_config
        self._rng = torch.Generator(device="cpu")
        if self.mask_config.seed is not None:
            self._rng.manual_seed(self.mask_config.seed)
        self._mask_state_by_param_id: dict[int, _MaskState] = {}
        self._initializing_wrapper = True

        super().__init__(self.base_optimizer.param_groups, {})
        self._initializing_wrapper = False
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = self.base_optimizer.defaults

    def _iter_params(self) -> list[torch.nn.Parameter]:
        params: list[torch.nn.Parameter] = []
        for group in self.base_optimizer.param_groups:
            for param in group["params"]:
                params.append(param)
        return params

    def _active_params_with_grad(self) -> list[torch.nn.Parameter]:
        params: list[torch.nn.Parameter] = []
        for group in self.base_optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    params.append(param)
        return params

    def _get_mask_state(self, param: torch.nn.Parameter) -> _MaskState:
        param_id = id(param)
        state = self._mask_state_by_param_id.get(param_id)
        if state is None:
            state = _MaskState()
            self._mask_state_by_param_id[param_id] = state
        return state

    @staticmethod
    def _cosine_similarity(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
        x32 = x.detach().float().reshape(-1)
        y32 = y.detach().float().reshape(-1)
        x_norm = float(torch.norm(x32))
        y_norm = float(torch.norm(y32))
        if x_norm <= eps or y_norm <= eps:
            return 0.0
        value = float(torch.dot(x32, y32) / (x_norm * y_norm + eps))
        return max(min(value, 1.0), -1.0)

    def _bernoulli_mask(self) -> float:
        if self.mask_config.survival_prob >= 1.0:
            return 1.0
        sampled = float(torch.rand((), generator=self._rng).item() < self.mask_config.survival_prob)
        return sampled

    def _get_momentum_like_tensor(self, param: torch.nn.Parameter, grad: torch.Tensor) -> torch.Tensor:
        base_state = self.base_optimizer.state.get(param, {})
        for key in ("exp_avg", "momentum_buffer", "momentum"):
            value = base_state.get(key)
            if torch.is_tensor(value):
                return value.detach()

        state = self._get_mask_state(param)
        if state.grad_ema is None:
            return grad.detach()
        return state.grad_ema.detach()

    def _update_fallback_grad_ema(self, param: torch.nn.Parameter, grad: torch.Tensor) -> None:
        state = self._get_mask_state(param)
        grad_fp32 = grad.detach().float()
        if state.grad_ema is None:
            state.grad_ema = grad_fp32.clone()
            return
        state.grad_ema.mul_(self.mask_config.ema_beta).add_(grad_fp32, alpha=1.0 - self.mask_config.ema_beta)

    def _compute_scale(self, param: torch.nn.Parameter, grad: torch.Tensor) -> float:
        mask = self._bernoulli_mask()
        if self.mask_config.mode == "skipupdate":
            return mask / self.mask_config.survival_prob

        momentum_like = self._get_momentum_like_tensor(param, grad)
        cossim = self._cosine_similarity(momentum_like, grad)
        s_tilde = 1.0 / (1.0 + math.exp(-cossim / self.mask_config.tau))

        state = self._get_mask_state(param)
        if state.s_t is None:
            state.s_t = s_tilde
        else:
            state.s_t = self.mask_config.ema_beta * state.s_t + (1.0 - self.mask_config.ema_beta) * s_tilde
        return state.s_t * mask

    @staticmethod
    def _can_use_foreach(param: torch.nn.Parameter, before: torch.Tensor) -> bool:
        if param.layout != torch.strided or before.layout != torch.strided:
            return False
        if param.is_sparse or before.is_sparse:
            return False
        if param.dtype != before.dtype or param.device != before.device:
            return False
        return True

    @torch.no_grad()
    def step(self, *args: Any, **kwargs: Any) -> Any:
        active_params = self._active_params_with_grad()
        params_before = {id(param): param.detach().clone() for param in active_params}
        grads_before = {id(param): param.grad.detach().clone() for param in active_params}

        loss = self.base_optimizer.step(*args, **kwargs)

        foreach_buckets: dict[tuple[torch.device, torch.dtype], _ForeachBucket] = {}
        for param in active_params:
            param_id = id(param)
            grad = grads_before[param_id]
            grad_for_scale = grad.to_dense() if grad.is_sparse else grad
            scale = self._compute_scale(param, grad_for_scale)
            self._update_fallback_grad_ema(param, grad_for_scale)

            if abs(scale - 1.0) < 1e-20:
                continue

            before = params_before[param_id]
            if self._can_use_foreach(param, before):
                bucket_key = (param.device, param.dtype)
                bucket = foreach_buckets.setdefault(bucket_key, _ForeachBucket(params=[], before=[], alpha=[]))
                bucket.params.append(param)
                bucket.before.append(before)
                bucket.alpha.append(scale - 1.0)
                continue

            delta = param.detach() - before
            param.copy_(before + delta * scale)

        for bucket in foreach_buckets.values():
            if not bucket.params:
                continue
            try:
                deltas = torch._foreach_sub(bucket.params, bucket.before)
                scaled_deltas = torch._foreach_mul(deltas, bucket.alpha)
                torch._foreach_add_(bucket.params, scaled_deltas)
            except RuntimeError:
                for p, p_before, alpha in zip(bucket.params, bucket.before, bucket.alpha, strict=True):
                    delta = p.detach() - p_before
                    p.copy_(p_before + delta * (alpha + 1.0))

        return loss

    def zero_grad(self, *args: Any, **kwargs: Any) -> None:
        self.base_optimizer.zero_grad(*args, **kwargs)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        if self._initializing_wrapper:
            Optimizer.add_param_group(self, param_group)
            return
        self.base_optimizer.add_param_group(param_group)
        self.param_groups = self.base_optimizer.param_groups

    def state_dict(self) -> dict[str, Any]:
        params = self._iter_params()
        wrapper_state_entries: list[dict[str, Any]] = []
        for param in params:
            state = self._mask_state_by_param_id.get(id(param))
            if state is None:
                wrapper_state_entries.append({"s_t": None, "grad_ema": None})
                continue
            grad_ema = None if state.grad_ema is None else state.grad_ema.detach().clone()
            wrapper_state_entries.append({"s_t": state.s_t, "grad_ema": grad_ema})

        return {
            "base_optimizer": self.base_optimizer.state_dict(),
            "mask_config": asdict(self.mask_config),
            "wrapper_state": wrapper_state_entries,
            "rng_state": self._rng.get_state(),
            "version": 1,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = self.base_optimizer.defaults

        if "mask_config" in state_dict:
            self.mask_config = MagmaMaskConfig(**state_dict["mask_config"])

        if "rng_state" in state_dict:
            rng_state = state_dict["rng_state"]
            if not torch.is_tensor(rng_state):
                raise TypeError("rng_state must be a torch.Tensor.")
            # Some checkpoints may store optimizer state tensors on CUDA or non-byte dtype.
            rng_state = rng_state.detach().to(device="cpu", dtype=torch.uint8)
            self._rng.set_state(rng_state)

        self._mask_state_by_param_id.clear()
        params = self._iter_params()
        wrapper_state_entries = state_dict.get("wrapper_state", [])
        if len(wrapper_state_entries) != len(params):
            raise ValueError(
                "Mismatched wrapper_state length while loading MagmaSkipUpdateWrapper state_dict: "
                f"{len(wrapper_state_entries)} vs {len(params)}"
            )

        for param, entry in zip(params, wrapper_state_entries, strict=True):
            grad_ema = entry.get("grad_ema")
            loaded_grad_ema = None
            if torch.is_tensor(grad_ema):
                loaded_grad_ema = grad_ema.detach().to(device=param.device, dtype=torch.float32)
            self._mask_state_by_param_id[id(param)] = _MaskState(
                s_t=entry.get("s_t"),
                grad_ema=loaded_grad_ema,
            )

    @property
    def base_state(self) -> dict[Any, Any]:
        return self.base_optimizer.state

    def get_param_mask_state(self, param: torch.nn.Parameter) -> dict[str, Any]:
        state = self._get_mask_state(param)
        grad_ema = None if state.grad_ema is None else state.grad_ema.detach().clone()
        return {"s_t": state.s_t, "grad_ema": grad_ema}


@function_config_to_basic_types
def wrap_optimizer_with_magma(
    base_optimizer: Optimizer,
    *,
    mode: MaskMode = "magma",
    survival_prob: float = 0.5,
    tau: float = 2.0,
    ema_beta: float = 0.9,
    seed: int | None = None,
) -> MagmaSkipUpdateWrapper:
    config = MagmaMaskConfig(
        mode=mode,
        survival_prob=survival_prob,
        tau=tau,
        ema_beta=ema_beta,
        seed=seed,
    )
    return MagmaSkipUpdateWrapper(base_optimizer=base_optimizer, mask_config=config)


def _normalize_named_parameters(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]] | dict[str, torch.nn.Parameter],
) -> list[tuple[str, torch.nn.Parameter]]:
    if isinstance(named_parameters, dict):
        items = list(named_parameters.items())
    else:
        items = list(named_parameters)
    normalized: list[tuple[str, torch.nn.Parameter]] = []
    for name, param in items:
        if not isinstance(name, str) or not isinstance(param, torch.nn.Parameter):
            raise TypeError("named_parameters must contain (str, torch.nn.Parameter).")
        normalized.append((name, param))
    return normalized


def _resolve_optimizer_cls(optimizer_cls: str | type[Optimizer]) -> type[Optimizer]:
    if isinstance(optimizer_cls, str):
        module_name, _, class_name = optimizer_cls.rpartition(".")
        if module_name == "":
            raise ValueError(f"Invalid optimizer_cls path: {optimizer_cls}")
        module = importlib.import_module(module_name)
        resolved = getattr(module, class_name)
    else:
        resolved = optimizer_cls

    if not isinstance(resolved, type) or not issubclass(resolved, Optimizer):
        raise TypeError(f"optimizer_cls must resolve to torch.optim.Optimizer subclass, got: {resolved}")
    return resolved


@function_config_to_basic_types
def create_torch_magma_optimizer(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]] | dict[str, torch.nn.Parameter],
    *,
    optimizer_cls: str | type[Optimizer],
    optimizer_kwargs: dict[str, Any] | None = None,
    mode: MaskMode = "magma",
    survival_prob: float = 0.5,
    tau: float = 2.0,
    ema_beta: float = 0.9,
    seed: int | None = None,
) -> MagmaSkipUpdateWrapper:
    normalized = _normalize_named_parameters(named_parameters)
    params = [param for _, param in normalized if param.requires_grad]
    optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
    resolved_cls = _resolve_optimizer_cls(optimizer_cls)
    base_optimizer = resolved_cls(params, **optimizer_kwargs)
    return wrap_optimizer_with_magma(
        base_optimizer=base_optimizer,
        mode=mode,
        survival_prob=survival_prob,
        tau=tau,
        ema_beta=ema_beta,
        seed=seed,
    )

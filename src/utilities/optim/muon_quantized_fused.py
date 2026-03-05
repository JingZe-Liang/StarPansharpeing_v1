from __future__ import annotations

from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass
from typing import Any, Literal

import torch

from ..config_utils import function_config_to_basic_types
from .flashoptim.flashoptim.optimizers import FlashAdamW, FlashLion, dequantize, quantize
from .muon_fused import MuonFSDP, zeropower_via_newtonschulz6_diff_abc


@dataclass(slots=True)
class _PackedMomentum:
    quantized: torch.Tensor
    scales: torch.Tensor
    group_size: int
    softsign: bool


class QuantizedMuonFSDP(MuonFSDP):
    """Muon optimizer with quantized Muon momentum and FlashOptim 1D fallback."""

    _STATE_DICT_VERSION = 1

    def __init__(
        self,
        params: Any,
        distributed_mesh: Any = None,
        lr: float = 0.01,
        mu: float = 0.95,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        cautious_wd: bool = False,
        epsilon: float = 1e-8,
        nesterov: bool = False,
        adjust_lr: str | None = "rms_norm",
        flatten: bool = True,
        use_triton: bool = False,
        use_preconditioned: bool = False,
        newton_schulz_func: Callable | None = zeropower_via_newtonschulz6_diff_abc,
        muon_steps: int = 5,
        enable_qk_clip: bool = False,
        qk_clip_threshold: float = 100.0,
        qk_clip_alpha: float = 0.5,
        *,
        muon_params_defaults: dict[str, Any] | None = None,
        oned_params_defaults: dict[str, Any] | None = None,
        qk_clip_options: dict[str, Any] | None = None,
        magma_kwargs: dict[str, Any] | None = None,
        muon_quantize_momentum: bool = True,
        muon_quant_group_size: int = 32,
        muon_quant_softsign: bool = True,
        oned_flash_quantize: bool = True,
        oned_flash_compress_state_dict: bool = False,
        oned_flash_master_weight_bits: Literal[24, 32] | None = 24,
        oned_flash_check_numerics: bool = False,
        oned_flash_fused: bool = True,
    ) -> None:
        super().__init__(
            params=params,
            distributed_mesh=distributed_mesh,
            lr=lr,
            mu=mu,
            betas=betas,
            weight_decay=weight_decay,
            cautious_wd=cautious_wd,
            epsilon=epsilon,
            nesterov=nesterov,
            adjust_lr=adjust_lr,
            flatten=flatten,
            use_triton=use_triton,
            use_preconditioned=use_preconditioned,
            newton_schulz_func=newton_schulz_func,
            muon_steps=muon_steps,
            enable_qk_clip=enable_qk_clip,
            qk_clip_threshold=qk_clip_threshold,
            qk_clip_alpha=qk_clip_alpha,
            muon_params_defaults=muon_params_defaults,
            oned_params_defaults=oned_params_defaults,
            qk_clip_options=qk_clip_options,
            magma_kwargs=magma_kwargs,
        )

        self._muon_qstate: dict[int, _PackedMomentum] = {}
        self._flash_adamw_optimizer: FlashAdamW | None = None
        self._flash_lion_optimizer: FlashLion | None = None
        self._adamw_source_groups: list[dict[str, Any]] = []
        self._lion_source_groups: list[dict[str, Any]] = []

        self._oned_flash_quantize = bool(oned_flash_quantize)
        self._oned_flash_compress_state_dict = bool(oned_flash_compress_state_dict)
        self._oned_flash_master_weight_bits = oned_flash_master_weight_bits
        self._oned_flash_check_numerics = bool(oned_flash_check_numerics)
        self._oned_flash_fused = bool(oned_flash_fused)

        if oned_flash_master_weight_bits not in (None, 24, 32):
            raise ValueError("oned_flash_master_weight_bits must be one of: None, 24, 32.")

        for group in self.param_groups:
            if group.get("algorithm") != "muon":
                continue
            group.setdefault("muon_quantize_momentum", muon_quantize_momentum)
            group.setdefault("muon_quant_group_size", muon_quant_group_size)
            group.setdefault("muon_quant_softsign", muon_quant_softsign)
            self._validate_muon_quant_group(group)

        self._build_oned_flash_optimizers()

    @classmethod
    @function_config_to_basic_types
    def clear_muon_adamw_params(
        cls,
        named_params: Iterable[tuple[str, torch.nn.Parameter]] | dict[str, torch.nn.Parameter],
        ignored_keys_for_muon: tuple[str, ...] | list[str] = (),
        oned_param_algo: str = "lion",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return MuonFSDP.clear_muon_adamw_params(
            named_params=named_params,
            ignored_keys_for_muon=ignored_keys_for_muon,
            oned_param_algo=oned_param_algo,
        )

    @classmethod
    @function_config_to_basic_types
    def create_muon_optimizer(
        cls,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]] | dict[str, torch.nn.Parameter],
        ignored_keys_for_muon: tuple[str, ...] | list[str] = (),
        oned_param_algo: str = "lion",
        magma_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "QuantizedMuonFSDP":
        named_parameters_items = cls._normalize_named_parameters(named_parameters)
        muon_params_defaults = dict(kwargs.pop("muon_params_defaults", {}) or {})
        oned_params_defaults = dict(kwargs.pop("oned_params_defaults", {}) or {})
        normalized_magma_kwargs = cls._normalize_create_magma_kwargs(magma_kwargs)
        if "reg_mask_mode" in kwargs or "magma_wrap_kwargs" in kwargs:
            raise ValueError(
                "Deprecated args `reg_mask_mode` / `magma_wrap_kwargs` are not supported. "
                "Please use `magma_kwargs={'mode': ..., ...}`."
            )

        kwargs["muon_params_defaults"] = muon_params_defaults
        kwargs["oned_params_defaults"] = oned_params_defaults

        muon_params_g, oned_params_g = cls.clear_muon_adamw_params(
            named_params=named_parameters_items,
            ignored_keys_for_muon=ignored_keys_for_muon,
            oned_param_algo=oned_param_algo,
        )

        optimizer = cls(
            params=(muon_params_g, oned_params_g),
            magma_kwargs=normalized_magma_kwargs,
            **kwargs,
        )
        optimizer.register_attention_params({name: param for name, param in named_parameters_items})
        return optimizer

    def _build_oned_flash_optimizers(self) -> None:
        adamw_groups: list[dict[str, Any]] = []
        lion_groups: list[dict[str, Any]] = []

        self._adamw_source_groups = []
        self._lion_source_groups = []

        for group in self.param_groups:
            algo = group.get("algorithm")
            params = group.get("params", [])
            if not params:
                continue
            if algo == "adamw":
                self._adamw_source_groups.append(group)
                adamw_groups.append(self._flash_group_from_source(group, algo="adamw"))
            elif algo == "lion":
                self._lion_source_groups.append(group)
                lion_groups.append(self._flash_group_from_source(group, algo="lion"))

        use_cuda = torch.cuda.is_available()
        flash_quantize = self._oned_flash_quantize and use_cuda
        flash_fused = self._oned_flash_fused and use_cuda
        flash_master_weight_bits = self._resolve_flash_master_weight_bits(
            adamw_groups=adamw_groups,
            lion_groups=lion_groups,
            use_cuda=use_cuda,
        )

        if adamw_groups:
            self._flash_adamw_optimizer = FlashAdamW(
                params=adamw_groups,  # type: ignore[arg-type]
                lr=float(adamw_groups[0]["lr"]),
                betas=adamw_groups[0]["betas"],
                eps=float(adamw_groups[0]["eps"]),
                weight_decay=float(adamw_groups[0]["weight_decay"]),
                quantize=flash_quantize,
                compress_state_dict=self._oned_flash_compress_state_dict,
                master_weight_bits=flash_master_weight_bits,
                check_numerics=self._oned_flash_check_numerics,
                fused=flash_fused,
            )

        if lion_groups:
            self._flash_lion_optimizer = FlashLion(
                params=lion_groups,  # type: ignore[arg-type]
                lr=float(lion_groups[0]["lr"]),
                betas=lion_groups[0]["betas"],
                weight_decay=float(lion_groups[0]["weight_decay"]),
                quantize=flash_quantize,
                compress_state_dict=self._oned_flash_compress_state_dict,
                master_weight_bits=flash_master_weight_bits,
                check_numerics=self._oned_flash_check_numerics,
                fused=flash_fused,
            )

    @staticmethod
    def _is_all_params_fp32(groups: list[dict[str, Any]]) -> bool:
        params = [p for group in groups for p in group.get("params", [])]
        if not params:
            return False
        return all(p.dtype == torch.float32 for p in params)

    def _resolve_flash_master_weight_bits(
        self,
        adamw_groups: list[dict[str, Any]],
        lion_groups: list[dict[str, Any]],
        use_cuda: bool,
    ) -> Literal[24, 32] | None:
        if not use_cuda:
            return None

        requested_bits = self._oned_flash_master_weight_bits
        if requested_bits is None:
            return None

        all_groups = [*adamw_groups, *lion_groups]
        if self._is_all_params_fp32(all_groups):
            return None

        return requested_bits

    @staticmethod
    def _to_local(tensor: torch.Tensor) -> torch.Tensor:
        maybe_to_local = getattr(tensor, "to_local", None)
        if callable(maybe_to_local):
            return maybe_to_local()
        return tensor

    def _flash_group_from_source(self, source_group: dict[str, Any], algo: str) -> dict[str, Any]:
        beta1 = float(source_group.get("beta1", self.defaults.get("beta1", 0.9)))
        beta2 = float(source_group.get("beta2", self.defaults.get("beta2", 0.95)))

        base_group = {
            "params": source_group["params"],
            "lr": float(source_group.get("lr", self.defaults["lr"])),
            "weight_decay": float(source_group.get("weight_decay", self.defaults["weight_decay"])),
            "betas": (beta1, beta2),
        }
        if algo == "adamw":
            base_group["eps"] = float(source_group.get("epsilon", self.defaults.get("epsilon", 1e-8)))
        return base_group

    def _sync_oned_flash_hparams(self) -> None:
        if self._flash_adamw_optimizer is not None:
            for source_group, flash_group in zip(
                self._adamw_source_groups,
                self._flash_adamw_optimizer.param_groups,
                strict=True,
            ):
                flash_group["lr"] = float(source_group.get("lr", flash_group["lr"]))
                flash_group["weight_decay"] = float(source_group.get("weight_decay", flash_group["weight_decay"]))
                flash_group["betas"] = (
                    float(source_group.get("beta1", flash_group["betas"][0])),
                    float(source_group.get("beta2", flash_group["betas"][1])),
                )
                flash_group["eps"] = float(source_group.get("epsilon", flash_group.get("eps", 1e-8)))

        if self._flash_lion_optimizer is not None:
            for source_group, flash_group in zip(
                self._lion_source_groups,
                self._flash_lion_optimizer.param_groups,
                strict=True,
            ):
                flash_group["lr"] = float(source_group.get("lr", flash_group["lr"]))
                flash_group["weight_decay"] = float(source_group.get("weight_decay", flash_group["weight_decay"]))
                flash_group["betas"] = (
                    float(source_group.get("beta1", flash_group["betas"][0])),
                    float(source_group.get("beta2", flash_group["betas"][1])),
                )

    def _iter_muon_groups_and_params(self) -> Generator[tuple[dict[str, Any], torch.Tensor], None, None]:
        for group in self.param_groups:
            if group.get("algorithm") != "muon":
                continue
            for param in group["params"]:
                yield group, param

    def _iter_muon_params_in_order(self) -> list[torch.Tensor]:
        return [param for _, param in self._iter_muon_groups_and_params()]

    @staticmethod
    def _validate_muon_quant_group(group: dict[str, Any]) -> None:
        group_size = int(group.get("muon_quant_group_size", 32))
        if group_size <= 0:
            raise ValueError("muon_quant_group_size must be > 0.")
        if group_size % 8 != 0:
            raise ValueError("muon_quant_group_size must be divisible by 8.")

    @staticmethod
    def _muon_quant_config(group: dict[str, Any]) -> tuple[bool, int, bool]:
        return (
            bool(group.get("muon_quantize_momentum", True)),
            int(group.get("muon_quant_group_size", 32)),
            bool(group.get("muon_quant_softsign", True)),
        )

    def _materialize_muon_momentum_states(self) -> None:
        for group, param in self._iter_muon_groups_and_params():
            state = self.state[param]
            momentum = state.get("momentum")
            if isinstance(momentum, torch.Tensor):
                continue

            packed = self._muon_qstate.get(id(param))
            if packed is not None:
                self._validate_muon_quant_group(group)
                local_param = self._to_local(param)
                restored = dequantize(
                    packed.quantized,
                    packed.scales,
                    signed=True,
                    group_size=packed.group_size,
                    sqrt=False,
                    softsign=packed.softsign,
                )
                state["momentum"] = restored.to(dtype=local_param.dtype)
                continue

            local_param = self._to_local(param)
            state["momentum"] = torch.zeros_like(local_param)

    def _quantize_muon_momentum_states(self) -> None:
        for group, param in self._iter_muon_groups_and_params():
            state = self.state.get(param)
            if not state:
                continue
            momentum = state.get("momentum")
            if not isinstance(momentum, torch.Tensor):
                continue

            quantize_enabled, group_size, softsign = self._muon_quant_config(group)
            self._validate_muon_quant_group(group)

            momentum_local = self._to_local(momentum)
            if quantize_enabled and momentum_local.is_cuda:
                quantized, scales = quantize(
                    momentum_local,
                    signed=True,
                    group_size=group_size,
                    sqrt=False,
                    softsign=softsign,
                )
                self._muon_qstate[id(param)] = _PackedMomentum(
                    quantized=quantized.detach(),
                    scales=scales.detach(),
                    group_size=group_size,
                    softsign=softsign,
                )
                state["momentum"] = None
            else:
                self._muon_qstate.pop(id(param), None)

    def _param_index_map(self) -> dict[int, int]:
        mapping: dict[int, int] = {}
        idx = 0
        for group in self.param_groups:
            for param in group["params"]:
                mapping[id(param)] = idx
                idx += 1
        return mapping

    def _step_oned_flash_optimizers(self) -> None:
        self._sync_oned_flash_hparams()
        if self._flash_adamw_optimizer is not None:
            self._flash_adamw_optimizer.step()
        if self._flash_lion_optimizer is not None:
            self._flash_lion_optimizer.step()

    def _create_lion_tasks(
        self,
        param_groups: list[dict[str, Any]],
        algo_name: str = "lion",
    ) -> Generator[Any, None, None]:
        del param_groups, algo_name
        yield from ()

    def _create_adamw_tasks(
        self,
        param_groups: list[dict[str, Any]],
        algo_name: str = "adamw",
    ) -> Generator[Any, None, None]:
        del param_groups, algo_name
        yield from ()

    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], torch.Tensor] | None = None,
        attention_max_logits: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        self._materialize_muon_momentum_states()
        loss = super().step(closure=closure, attention_max_logits=attention_max_logits)
        self._step_oned_flash_optimizers()
        self._quantize_muon_momentum_states()
        return loss

    def state_dict(self) -> dict[str, Any]:
        self._quantize_muon_momentum_states()
        base_state = super().state_dict()

        param_to_index = self._param_index_map()
        muon_qstate_entries: list[dict[str, Any] | None] = []
        for param in self._iter_muon_params_in_order():
            packed = self._muon_qstate.get(id(param))
            if packed is None:
                muon_qstate_entries.append(None)
                continue

            muon_qstate_entries.append(
                {
                    "q": packed.quantized,
                    "scales": packed.scales,
                    "group_size": packed.group_size,
                    "softsign": packed.softsign,
                }
            )

            state_idx = param_to_index[id(param)]
            state_entry = base_state["state"].get(state_idx)
            if isinstance(state_entry, dict):
                state_entry["momentum"] = None

        oned_flash_state: dict[str, Any] = {}
        if self._flash_adamw_optimizer is not None:
            oned_flash_state["adamw"] = self._flash_adamw_optimizer.state_dict()
        if self._flash_lion_optimizer is not None:
            oned_flash_state["lion"] = self._flash_lion_optimizer.state_dict()

        return {
            "base_state": base_state,
            "oned_flash_state": oned_flash_state,
            "muon_qstate": muon_qstate_entries,
            "meta": {"version": self._STATE_DICT_VERSION},
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if "base_state" not in state_dict:
            raise ValueError("Expected QuantizedMuonFSDP state_dict with key 'base_state'.")
        if "muon_qstate" not in state_dict:
            raise ValueError("Expected QuantizedMuonFSDP state_dict with key 'muon_qstate'.")

        super().load_state_dict(state_dict["base_state"])

        oned_flash_state = state_dict.get("oned_flash_state", {})
        if "adamw" in oned_flash_state:
            if self._flash_adamw_optimizer is None:
                raise ValueError("State dict contains adamw flash state but optimizer has no AdamW 1D groups.")
            self._flash_adamw_optimizer.load_state_dict(oned_flash_state["adamw"])
        if "lion" in oned_flash_state:
            if self._flash_lion_optimizer is None:
                raise ValueError("State dict contains lion flash state but optimizer has no Lion 1D groups.")
            self._flash_lion_optimizer.load_state_dict(oned_flash_state["lion"])

        muon_qstate_entries = state_dict["muon_qstate"]
        if not isinstance(muon_qstate_entries, list):
            raise ValueError("Expected `muon_qstate` to be a list indexed by muon parameter order.")

        muon_params = self._iter_muon_params_in_order()
        if len(muon_qstate_entries) != len(muon_params):
            raise ValueError(
                "Mismatch between saved muon_qstate entries and current Muon parameter count. "
                f"Expected {len(muon_params)}, got {len(muon_qstate_entries)}."
            )

        self._muon_qstate.clear()
        for param, entry in zip(muon_params, muon_qstate_entries, strict=True):
            if entry is None:
                continue
            q = entry["q"].to(dtype=torch.int8)
            scales = entry["scales"].to(dtype=torch.float16)
            group_size = int(entry["group_size"])
            softsign = bool(entry.get("softsign", True))
            self._muon_qstate[id(param)] = _PackedMomentum(
                quantized=q,
                scales=scales,
                group_size=group_size,
                softsign=softsign,
            )
            self.state[param]["momentum"] = None

        self._sync_oned_flash_hparams()

    def __repr__(self) -> str:
        return (
            "QuantizedMuonFSDP(\n"
            f"    lr={self.defaults['lr']}, mu={self.defaults['mu']}, weight_decay={self.defaults['weight_decay']},\n"
            f"    oned_flash_quantize={self._oned_flash_quantize}, oned_flash_fused={self._oned_flash_fused},\n"
            f"    oned_flash_master_weight_bits={self._oned_flash_master_weight_bits}\n"
            ")"
        )

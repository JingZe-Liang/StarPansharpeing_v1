from dataclasses import dataclass
from collections.abc import Callable, Iterable
from typing import Any, TypedDict
import re
import inspect

import torch
from loguru import logger
from torch.distributed.distributed_c10d import ProcessGroup
from torch.distributed.tensor import DeviceMesh
from torch.optim.optimizer import ParamsT

from .dion.dion.muon import Muon
from .dion.dion.newton_schulz_triton import ns_line_1, ns_line_2
from ..config_utils import function_config_to_basic_types

######### Newton-Schulz ABCs ##########


@dataclass
class MuonConfig:
    name: str = "su"


QKClipSelector = Callable[[torch.Tensor], torch.Tensor]


class QKClipOption(TypedDict):
    p_name: str | tuple[str, ...]
    p_func: QKClipSelector


def gen_muon_consts(cfg: MuonConfig) -> list[tuple]:
    if cfg.name == "su":
        abc_s = [
            (8.205160841660083, -22.901935266050053, 16.46072465627276),
            (4.066395089804725, -2.8611542266844125, 0.5183995687182832),
            (3.909594978763042, -2.823351854742293, 0.5250370015984781),
            (3.2855640853802908, -2.4153019551207473, 0.4852940692752496),
            (2.2778733156966533, -1.6198217682227882, 0.39848078792166197),
            (1.8725756548189942, -1.230704294480799, 0.3585161692646899),
            (1.8564356610934158, -1.2132377145299476, 0.35679964557408256),
        ]
    elif cfg.name == "turbo_muon":
        abc_s = [
            (4.0848, -6.8946, 2.9270),
            (3.9505, -6.3029, 2.6377),
            (3.7418, -5.5913, 2.3037),
            (2.8769, -3.1427, 1.2046),
            (2.8366, -3.0525, 1.2012),
        ]
    else:
        raise ValueError(f"Unknown MuonConfig name: {cfg.name}")

    return abc_s


@torch.compile(dynamic=False, fullgraph=True)
def zeropower_via_newtonschulz6_diff_abc(
    G,
    steps: int = 6,
    norm=False,
    ns_dtype=torch.bfloat16,
    use_triton=False,
    preconditioned=False,
    epsilon=1e-8,
    muon_abcs: list[tuple] = gen_muon_consts(MuonConfig()),
):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """

    assert (
        G.ndim >= 2
    )  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng

    X = G.to(dtype=ns_dtype)
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1 (standard Muon).
    # TurboMuon-style AOL 预条件化通常不做这一步，而是依赖首轮 AOL rescaling 来稳定迭代。
    if not preconditioned:
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + epsilon)
    else:
        # TurboMuon reference implementation uses 4 iterations for the preconditioned variant.
        steps = min(int(steps), 4)

    # Coefficients: TurboMuon reference uses `ns_consts[-iter:]` (iter=4 => last 4).
    # 为了对齐该行为，这里取末尾 `steps` 个系数；当 steps 超过可用系数时，用最后一个补齐。
    consts = muon_abcs[-steps:] + max(steps - len(muon_abcs), 0) * muon_abcs[-1:]

    if use_triton:
        X = X.contiguous()
        A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
        B = torch.empty_like(A)
        C = torch.empty_like(X)
        ns_line_3 = torch.baddbmm if X.ndim > 2 else torch.addmm

        # triton code
        ns_line_1(X, out=A)  # A = X @ X.mT

        a, b, c = consts[0]
        if preconditioned:
            # see https://github.com/thib-s/flash-newton-schulz/blob/145260f4b49c81b9200c61e0f95751b43bf672d5/newton_schulz_triton.py#L587
            s = torch.rsqrt(torch.clamp_min(A.abs().sum(dim=-1, keepdim=False), min=epsilon))  # AOL rescaling vector
            X = X * s.unsqueeze(-1)  # rescale X using s making it closer to orthogonal
            # first NS iteration with reuse of A
            A = A * s.unsqueeze(-1) * s.unsqueeze(-2)
        ns_line_2(A, alpha=c, beta=b, out=B)  # B = b * A + c * A @ A
        ns_line_3(X, B, X, beta=a, out=C)  # C = a * X + B @ X
        X, C = C, X  # Swap references to avoid unnecessary copies

        # Perform the NS iterations
        for a, b, c in consts[1:]:
            ns_line_1(X, out=A)  # A = X @ X.mT
            ns_line_2(A, alpha=c, beta=b, out=B)  # B = b * A + c * A @ A
            ns_line_3(X, B, X, beta=a, out=C)  # C = a * X + B @ X
            X, C = C, X  # Swap references to avoid unnecessary copies

    else:
        # naive python code: compiled with torch.compile
        for i, (a, b, c) in enumerate(consts):
            A = X @ X.mT
            A2 = A @ A
            if i == 0:
                if norm:
                    # Comes from su's blog
                    n = ((A2**2).sum(dim=(-2, -1), keepdim=True) + epsilon) ** 0.125
                    X, A, A2 = X / n, A / n**2, A2 / n**4
                if preconditioned:
                    # Comes from the TurboMuon paper
                    s = torch.rsqrt(torch.clamp_min(A.abs().sum(dim=-1, keepdim=False), min=epsilon))
                    X = X * s.unsqueeze(-1)
                    A = A * s.unsqueeze(-1) * s.unsqueeze(-2)

            # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
            B = b * A + c * A2
            X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class MuonFSDP(Muon):
    def __init__(
        self,
        params: ParamsT,
        distributed_mesh: DeviceMesh | ProcessGroup | None = None,
        # defaults
        lr: float = 0.01,
        mu: float = 0.95,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        cautious_wd: bool = False,
        epsilon: float = 1e-8,
        nesterov: bool = False,
        adjust_lr: str | None = "rms_norm",
        flatten: bool = True,  # NOTE: set to True by default if using 4D tensors, e.g., Conv2D weights
        use_triton: bool = False,
        use_preconditioned: bool = False,
        newton_schulz_func: Callable | None = zeropower_via_newtonschulz6_diff_abc,
        muon_steps: int = 5,
        enable_qk_clip: bool = False,
        qk_clip_threshold: float = 100.0,
        qk_clip_alpha: float = 0.5,
        *,
        # muon and adamw/lion param group defaults
        muon_params_defaults: dict[str, Any] | None = None,
        oned_params_defaults: dict[str, Any] | None = None,
        qk_clip_options: dict[str, QKClipOption] | None = None,
        magma_kwargs: dict | None = None,
    ):
        """
        Create a Muon optimizer with optional Kimi-style QK-Clip.

        Args:
            params: Parameter groups passed to torch optimizer.
            distributed_mesh: Device mesh or process group used by distributed Muon.
            lr: Base learning rate.
            mu: Momentum factor for Muon updates.
            betas: Betas for one-dimensional fallback algorithms (AdamW/Lion groups).
            weight_decay: Weight decay coefficient.
            cautious_wd: Whether to apply cautious weight decay.
            epsilon: Numerical epsilon used by Muon updates.
            nesterov: Whether to enable Nesterov momentum for Muon.
            adjust_lr: Muon LR adjustment mode. One of {"rms_norm", "spectral_norm", None}.
            flatten: Whether to flatten 3D+ tensors for Muon matrix updates.
            use_triton: Whether to use Triton kernels in Newton-Schulz.
            use_preconditioned: Whether to use the preconditioned Newton-Schulz path.
            newton_schulz_func: Custom Newton-Schulz function. If provided, this wrapper injects
                the configured `muon_steps`, `use_triton`, and `use_preconditioned`.
            muon_steps: Number of Newton-Schulz iterations.
            enable_qk_clip: Enable QK-Clip post optimizer step.
            qk_clip_threshold: QK-Clip threshold `tau`.
            qk_clip_alpha: Split exponent between query/key context branch. `Wq_c` uses `gamma**alpha`,
                `Wk_c` uses `gamma**(1-alpha)`.
            reg_mask_mode: Optional inline masking mode for all param groups.
                One of {None, "magma", "skipupdate"}.
            reg_mask_survival_prob: Bernoulli survival probability for masking.
            reg_mask_tau: Temperature used by Magma's sigmoid(cos/tau).
            reg_mask_ema_beta: EMA coefficient for Magma alignment state.
            reg_mask_seed: Optional RNG seed for reproducible masking.
            muon_params_defaults: Extra defaults for Muon param groups.
            oned_params_defaults: Extra defaults for one-dimensional param groups.
            qk_clip_options: Mapping that defines how parameter names and tensor slices map to
                semantic branches {"Wq_c", "Wk_c", "Wq_r", optional "Wk_r"}.
                - `p_name`: str or tuple[str, ...], used to match parameter names.
                - `p_func`: callable that receives the full parameter tensor and returns the
                  tensor/view to be scaled.

                DeepSeek-MLA style example:
                ```python
                qk_nope = config.qk_nope_head_dim
                qk_rope = config.qk_rope_head_dim
                v_dim = config.v_head_dim
                q_head_dim = qk_nope + qk_rope
                kv_out_head_dim = qk_nope + v_dim

                qk_clip_options = {
                    # q_proj or q_b_proj output layout per head: [q_nope, q_rope]
                    "Wq_c": {"p_name": ("q_proj.weight", "q_b_proj.weight"), "p_func": lambda w: w},
                    "Wq_r": {"p_name": ("q_proj.weight", "q_b_proj.weight"), "p_func": lambda w: w},
                    # kv_b_proj output layout per head: [k_nope, value]
                    "Wk_c": {
                        "p_name": "kv_b_proj.weight",
                        "p_func": lambda w: w.view(-1, kv_out_head_dim, w.shape[1])[:, :qk_nope, :].reshape(-1, w.shape[1]),
                    },
                    # shared key-rope branch from kv_a_proj_with_mqa, usually skipped in clipping
                    "Wk_r": {
                        "p_name": "kv_a_proj_with_mqa.weight",
                        "p_func": lambda w: w[-qk_rope:, :],
                    },
                }
                ```

                Note:
                For models where query/key are fully RoPE (no NOPE split), you can map
                `q_proj` to `Wq_r` and `k_proj` to `Wk_r` directly without sub-slicing.
        """
        if qk_clip_threshold <= 0.0:
            raise ValueError(f"Invalid qk_clip_threshold: {qk_clip_threshold}. Must be > 0.")
        if not 0.0 <= qk_clip_alpha <= 1.0:
            raise ValueError(f"Invalid qk_clip_alpha: {qk_clip_alpha}. Must be in [0, 1].")

        muon_params_defaults = {} if muon_params_defaults is None else muon_params_defaults
        oned_params_defaults = {} if oned_params_defaults is None else oned_params_defaults

        muon_abcs = gen_muon_consts(MuonConfig(name="su"))
        if newton_schulz_func is not None:
            newton_schulz_func = lambda G, epsilon: zeropower_via_newtonschulz6_diff_abc(
                G=G,
                steps=muon_steps,
                use_triton=use_triton,
                preconditioned=use_preconditioned,
                epsilon=epsilon,
                muon_abcs=muon_abcs,
            )
            logger.log(
                "NOTE",
                "[Muon]: using self-defined triton Newton-Schulz function, "
                f"use_triton={use_triton}, use_preconditioned={use_preconditioned}",
            )

        # check if support magma kwargs
        support_kwargs = inspect.signature(super().__init__).parameters.keys()
        dd = {}
        if "magma_kwargs" in support_kwargs:
            dd |= {"magma_kwargs": magma_kwargs}
        elif magma_kwargs is not None:
            logger.warning("[MuonFused]: magma_kwargs is not supported by the current version of Dion verson.")

        super().__init__(
            params,
            distributed_mesh,
            lr,
            mu,
            betas,
            weight_decay,
            cautious_wd,
            epsilon,
            nesterov,
            adjust_lr,
            flatten,
            use_triton,
            newton_schulz_func,
            **dd,
        )

        self.defaults["enable_qk_clip"] = enable_qk_clip
        self.defaults["qk_clip_threshold"] = qk_clip_threshold
        self.defaults["qk_clip_alpha"] = qk_clip_alpha
        self._param_to_name: dict[int, str] = {}
        self.qk_clip_options = qk_clip_options

        self._init_param_groups_defaults(muon_params_defaults, oned_params_defaults)

    def _init_param_groups_defaults(self, muon_params_defaults: dict, oned_params_defaults: dict):
        for group in self.param_groups:
            if group["algorithm"] == "muon":
                for k, v in self.defaults.items():
                    group.setdefault(k, v)
                group.update(muon_params_defaults)

                weight_update_method = group.get("weight_update_method", "sgd")
                if weight_update_method == "hyperball" and float(group.get("weight_decay", 0.0)) > 0.0:
                    logger.error(
                        "[MuonFSDP] weight_update_method='hyperball' is incompatible with weight_decay>0. "
                        "Forcing group['weight_decay']=0.0."
                    )
                    group["weight_decay"] = 0.0
            elif group["algorithm"] in ("adamw", "lion"):
                for k, v in self.defaults.items():
                    group.setdefault(k, v)
                group.update(oned_params_defaults)
            else:
                raise ValueError(f"Unknown algorithm: {group['algorithm']}")

    @staticmethod
    def _normalize_named_parameters(
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]] | dict[str, torch.nn.Parameter],
    ) -> list[tuple[str, torch.nn.Parameter]]:
        normalized_items: list[tuple[str, torch.nn.Parameter]] = []
        if isinstance(named_parameters, dict):
            for name, param in named_parameters.items():
                if not isinstance(name, str) or not isinstance(param, torch.nn.Parameter):
                    raise TypeError("`named_parameters` dict must map `str` to `torch.nn.Parameter`.")
                normalized_items.append((name, param))
            return normalized_items

        for item in named_parameters:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("`named_parameters` iterable must contain `(name, parameter)` tuples.")
            name, param = item
            if not isinstance(name, str) or not isinstance(param, torch.nn.Parameter):
                raise TypeError("`named_parameters` iterable must contain `(str, torch.nn.Parameter)` tuples.")
            normalized_items.append((name, param))
        return normalized_items

    @classmethod
    @function_config_to_basic_types
    def clear_muon_adamw_params(
        cls,
        named_params: Iterable[tuple[str, torch.nn.Parameter]] | dict[str, torch.nn.Parameter],
        ignored_keys_for_muon: tuple[str, ...] | list[str] = (),  # for re to match
        oned_param_algo: str = "lion",
    ):
        muon_params = []
        oned_params = []

        re_ignore_pats = [re.compile(ik) for ik in ignored_keys_for_muon]
        iter_named_params = cls._normalize_named_parameters(named_params)

        for name, p in iter_named_params:
            if p.requires_grad:
                # conv, linear weights, and other 2D+ parameters
                if p.ndim >= 2:
                    should_ignore = False
                    for re_ignore_pat in re_ignore_pats:
                        if re_ignore_pat.search(name):
                            should_ignore = True
                            logger.debug(
                                f"[MuonFSDP] Ignored param for Muon: {name} at pattern {re_ignore_pat.pattern}"
                            )
                            break

                    if should_ignore:
                        oned_params.append(p)
                    else:
                        muon_params.append(p)
                # bias, norm weights, embeddings, lm heads (for nlp tasks), and other 1D parameters
                else:
                    oned_params.append(p)

        muon_params_g = {"params": muon_params, "algorithm": "muon"}
        oned_params_g = {"params": oned_params, "algorithm": oned_param_algo}

        return muon_params_g, oned_params_g

    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], torch.Tensor] | None = None,
        attention_max_logits: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        loss = super().step(closure)

        # apply qk-clip from kimi-k2 paper
        if self.defaults["enable_qk_clip"] and attention_max_logits is not None:
            self._apply_qk_clip(attention_max_logits)

        return loss

    # ------------------ QK-clip ----------------- #
    @torch.no_grad()
    def _apply_qk_clip(self, attention_max_logits: dict[str, torch.Tensor]) -> None:
        tau = float(self.defaults["qk_clip_threshold"])
        alpha = float(self.defaults["qk_clip_alpha"])
        assert self.qk_clip_options is not None

        def _name_filter(candidates: str | tuple[str, ...], param_name: str) -> bool:
            if isinstance(candidates, str):
                return candidates in param_name
            for c in candidates:
                if c in param_name:
                    return True
            return False

        def _parse_option(name: str) -> tuple[str | tuple[str, ...], QKClipSelector]:
            option = self.qk_clip_options.get(name)
            if option is None:
                raise ValueError(f"Missing qk_clip_options[{name!r}].")
            p_name = option["p_name"]
            p_func = option["p_func"]
            return p_name, p_func

        for group in self.param_groups:
            for p in group["params"]:
                param_name = self._get_param_name(p)
                if param_name == "" or param_name not in attention_max_logits:
                    continue

                max_logits = attention_max_logits[param_name]
                if max_logits.numel() == 0:
                    continue

                max_logits = max_logits.to(device=p.device, dtype=torch.float32)
                needs_clip = max_logits > tau
                if not bool(needs_clip.any().item()):
                    continue

                gamma = torch.clamp(tau / max_logits, max=1.0)

                wq_c_names, wq_c_func = _parse_option("Wq_c")
                wk_c_names, wk_c_func = _parse_option("Wk_c")
                wq_r_names, wq_r_func = _parse_option("Wq_r")

                # query/key context and query-rope can coexist in the same packed weight
                if _name_filter(wq_c_names, param_name):
                    self._scale_attention_heads(wq_c_func(p), torch.pow(gamma, alpha), needs_clip)
                if _name_filter(wk_c_names, param_name):
                    self._scale_attention_heads(wk_c_func(p), torch.pow(gamma, 1.0 - alpha), needs_clip)
                if _name_filter(wq_r_names, param_name):
                    self._scale_attention_heads(wq_r_func(p), gamma, needs_clip)

                # key-rope branch is usually shared and kept untouched
                wk_r_option = self.qk_clip_options.get("Wk_r")
                if wk_r_option is not None:
                    wk_r_names = wk_r_option["p_name"]
                    if _name_filter(wk_r_names, param_name):
                        continue

    @staticmethod
    def _scale_attention_heads(param: torch.Tensor, scale_factors: torch.Tensor, mask: torch.Tensor) -> None:
        num_heads = int(scale_factors.shape[0])
        if num_heads == 0:
            return

        scale_factors = scale_factors.to(device=param.device, dtype=param.dtype)
        mask = mask.to(device=param.device, dtype=torch.bool)

        if param.dim() == 2:
            if param.shape[0] % num_heads != 0:
                logger.warning(
                    f"[MuonFSDP] Skip QK-Clip: param first dim {param.shape[0]} not divisible by num_heads {num_heads}."
                )
                return
            head_dim = param.shape[0] // num_heads
            for h in range(num_heads):
                if not bool(mask[h].item()):
                    continue
                start_idx = h * head_dim
                end_idx = (h + 1) * head_dim
                param[start_idx:end_idx].mul_(scale_factors[h])
        elif param.dim() == 3:
            if param.shape[0] != num_heads:
                logger.warning(
                    f"[MuonFSDP] Skip QK-Clip: param head dim {param.shape[0]} mismatches num_heads {num_heads}."
                )
                return
            for h in range(num_heads):
                if bool(mask[h].item()):
                    param[h].mul_(scale_factors[h])

    # ---------------- Utils ---------------- #
    def _get_param_name(self, param: torch.Tensor) -> str:
        return self._param_to_name.get(id(param), "")

    def register_attention_params(self, param_name_mapping: dict[str, torch.nn.Parameter]) -> None:
        for name, param in param_name_mapping.items():
            self._param_to_name[id(param)] = name

    @staticmethod
    def _normalize_create_magma_kwargs(magma_kwargs: dict[str, Any] | None) -> dict[str, Any] | None:
        if magma_kwargs is None:
            return None

        config = dict(magma_kwargs)
        supported_keys = {"mode", "survival_prob", "tau", "ema_beta", "seed"}
        unknown_keys = set(config).difference(supported_keys)
        if unknown_keys:
            unknown_str = ", ".join(sorted(unknown_keys))
            raise ValueError(f"Unsupported magma_kwargs keys: {unknown_str}.")

        reg_mask_mode = config.get("mode")
        if reg_mask_mode not in (None, "magma", "skipupdate"):
            raise ValueError(f"Invalid magma mode: {reg_mask_mode}. Expected one of: None, magma, skipupdate.")
        if reg_mask_mode is None and any(key in config for key in ("survival_prob", "tau", "ema_beta", "seed")):
            raise ValueError("magma_kwargs['mode'] is None, but mask hyperparameters are provided.")

        normalized = {"reg_mask_mode": reg_mask_mode}
        if "survival_prob" in config:
            normalized["reg_mask_survival_prob"] = float(config["survival_prob"])
        if "tau" in config:
            normalized["reg_mask_tau"] = float(config["tau"])
        if "ema_beta" in config:
            normalized["reg_mask_ema_beta"] = float(config["ema_beta"])
        if "seed" in config:
            seed = config["seed"]
            normalized["reg_mask_seed"] = None if seed is None else int(seed)
        return normalized

    # ------------------- Instantiation ------------------- #
    @classmethod
    @function_config_to_basic_types
    def create_muon_optimizer(
        cls,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]] | dict[str, torch.nn.Parameter],
        ignored_keys_for_muon: tuple[str, ...] | list[str] = (),
        oned_param_algo: str = "lion",
        magma_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> "MuonFSDP":
        """
        Build a MuonFSDP optimizer from named parameters, with optional inline Magma/SkipUpdate masking.

        This helper splits parameters into:
        - Muon group: trainable parameters with ndim >= 2 (except ignored patterns).
        - 1D fallback group: trainable parameters with ndim < 2, or ignored by regex.

        Parameters
        ----------
        named_parameters : Iterable[tuple[str, torch.nn.Parameter]] | dict[str, torch.nn.Parameter]
            Model named parameters from ``model.named_parameters()`` or an equivalent dict.
        ignored_keys_for_muon : tuple[str, ...] | list[str], default=()
            Regex patterns. Matched 2D+ parameters are moved to the 1D fallback group.
        oned_param_algo : str, default="lion"
            Algorithm for the fallback group. Expected values are ``"lion"`` or ``"adamw"``.
        magma_kwargs : dict[str, Any] | None, default=None
            Optional inline masking configuration.
            Supported keys:
            - ``mode``: one of ``None``, ``"magma"``, ``"skipupdate"``.
            - ``survival_prob`` -> ``reg_mask_survival_prob``
            - ``tau`` -> ``reg_mask_tau``
            - ``ema_beta`` -> ``reg_mask_ema_beta``
            - ``seed`` -> ``reg_mask_seed``
            When ``mode`` is ``None``, the other keys must not be provided.
        **kwargs
            Remaining kwargs forwarded to ``MuonFSDP(...)`` (for example ``lr``, ``muon_steps``,
            ``muon_params_defaults``, ``oned_params_defaults``, ``enable_qk_clip``, etc.).

        Returns
        -------
        MuonFSDP
            Always returns a ``MuonFSDP`` instance.

        Raises
        ------
        ValueError
            If ``mode`` is invalid, or if ``magma_kwargs`` contains unsupported keys.

        Examples
        --------
        Plain MuonFSDP:
        ```python
        optimizer = MuonFSDP.create_muon_optimizer(
            model.named_parameters(),
            ignored_keys_for_muon=("head",),
            lr=1e-3,
            muon_steps=5,
        )
        ```

        MuonFSDP with inline Magma masking:
        ```python
        optimizer = MuonFSDP.create_muon_optimizer(
            model.named_parameters(),
            magma_kwargs={
                "mode": "magma",
                "survival_prob": 0.5,
                "tau": 2.0,
                "ema_beta": 0.9,
                "seed": 0,
            },
            lr=1e-3,
        )
        ```

        MuonFSDP with inline SkipUpdate masking:
        ```python
        optimizer = MuonFSDP.create_muon_optimizer(
            model.named_parameters(),
            magma_kwargs={"mode": "skipupdate", "survival_prob": 0.5, "seed": 123},
            lr=1e-3,
        )
        ```
        """
        named_parameters_items = cls._normalize_named_parameters(named_parameters)
        muon_params_defaults = dict(kwargs.pop("muon_params_defaults", {}) or {})
        oned_params_defaults = dict(kwargs.pop("oned_params_defaults", {}) or {})
        normalized_magma_kwargs = cls._normalize_create_magma_kwargs(magma_kwargs)
        if "reg_mask_mode" in kwargs or "magma_wrap_kwargs" in kwargs:
            raise ValueError(
                "Deprecated args `reg_mask_mode` / `magma_wrap_kwargs` are not supported. "
                "Please use `magma_kwargs={'mode': ..., ...}`."
            )

        # Muon 2d and 1d params
        kwargs["muon_params_defaults"] = muon_params_defaults
        kwargs["oned_params_defaults"] = oned_params_defaults

        # Clear out the 2d and 1d params
        muon_params_g, oned_params_g = cls.clear_muon_adamw_params(
            named_parameters_items, ignored_keys_for_muon, oned_param_algo
        )

        # Instantiate the optimizer
        optimizer = cls(params=(muon_params_g, oned_params_g), magma_kwargs=normalized_magma_kwargs, **kwargs)
        optimizer.register_attention_params({name: param for name, param in named_parameters_items})

        return optimizer

    def __repr__(self) -> str:
        return (
            f"MuonFSDP(\n"
            f"    lr={self.defaults['lr']}, \n"
            f"    mu={self.defaults['mu']}, betas={self.defaults['betas']}, \n"
            f"    weight_decay={self.defaults['weight_decay']}, epsilon={self.defaults['epsilon']}, \n"
            f"    nesterov={self.defaults['nesterov']}, adjust_lr={self.defaults['adjust_lr']}, \n"
            f"    flatten={self.defaults['flatten']}, use_triton={self.defaults['use_triton']}\n"
            ")"
        )


def __test_muon_fsdp() -> None:
    import torch.nn as nn
    from timm.models.resnet import resnet50

    network = resnet50(num_classes=1000).cuda()
    optimizer = MuonFSDP.create_muon_optimizer(
        network.named_parameters(),
        ignored_keys_for_muon=("fc",),
        muon_params_defaults={"lr": 1e-3, "weight_decay": 1e-4},
        oned_params_defaults={"lr": 1e-4, "weight_decay": 1e-5},
        betas=(0.95, 0.99),
        lr=1e-2,
    )

    x = torch.randn(2, 3, 224, 224).cuda()
    y = network(x)
    criterion = nn.CrossEntropyLoss()
    target = torch.randint(0, 1000, (2,)).cuda()
    print("Forwarded the model ...")
    loss = criterion(y, target)
    loss.backward()
    print("Step the optimizer...")
    optimizer.step()

    print("MuonFSDP test passed.")


if __name__ == "__main__":
    """python -m src.utilities.optim.muon_fused"""
    __test_muon_fsdp()

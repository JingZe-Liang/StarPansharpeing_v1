from dataclasses import dataclass
from collections.abc import Callable, Iterable
from typing import Any
import re

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


def gen_muon_consts(cfg: MuonConfig) -> list[tuple]:
    if cfg.name == "su":
        abc_s = torch.tensor(
            [
                (8.287212018145622, -23.59588651909882, 17.300387312530923),
                (4.107059111542197, -2.9478499167379084, 0.54484310829266),
                (3.9486908534822938, -2.908902115962947, 0.5518191394370131),
                (3.3184196573706055, -2.488488024314878, 0.5100489401237208),
                (2.3006520199548186, -1.6689039845747518, 0.4188073119525678),
                (1.8913014077874002, -1.2679958271945908, 0.37680408948524996),
                (1.875, -1.25, 0.375),
            ]
        ).to(torch.float64)
        denorm = torch.tensor([1.01, 1.01**3, 1.01**5])
        abc_s = abc_s / denorm[None]
        abc_s = abc_s.detach().cpu().numpy().tolist()
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
        *,
        # muon and adamw/lion param group defaults
        muon_params_defaults: dict[str, Any] | None = None,
        oned_params_defaults: dict[str, Any] | None = None,
    ):
        muon_params_defaults = {} if muon_params_defaults is None else muon_params_defaults
        oned_params_defaults = {} if oned_params_defaults is None else oned_params_defaults
        muon_abcs = gen_muon_consts(
            MuonConfig(name="turbo_muon" if use_preconditioned else "su"),
        )
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
        )

        self._init_param_groups_defaults(muon_params_defaults, oned_params_defaults)

    def _init_param_groups_defaults(self, muon_params_defaults: dict, oned_params_defaults: dict):
        for group in self.param_groups:
            if group["algorithm"] == "muon":
                for k, v in self.defaults.items():
                    group.setdefault(k, v)
                group.update(muon_params_defaults)
            elif group["algorithm"] in ("adamw", "lion"):
                for k, v in self.defaults.items():
                    group.setdefault(k, v)
                group.update(oned_params_defaults)
            else:
                raise ValueError(f"Unknown algorithm: {group['algorithm']}")

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
        if isinstance(named_params, dict):
            named_params = named_params.items()

        for name, p in named_params:
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

    @classmethod
    @function_config_to_basic_types
    def create_muon_optimizer(
        cls,
        named_parameters: ParamsT,
        ignored_keys_for_muon: tuple | list = (),
        oned_param_algo: str = "lion",
        **kwargs,
    ):
        muon_params_g, oned_params_g = cls.clear_muon_adamw_params(
            named_parameters, ignored_keys_for_muon, oned_param_algo
        )
        optimizer = cls(params=(muon_params_g, oned_params_g), **kwargs)
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

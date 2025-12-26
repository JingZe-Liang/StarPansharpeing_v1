from typing import Any, Callable, Iterable, Optional, Tuple, Union

import torch
from loguru import logger
from torch.distributed.distributed_c10d import ProcessGroup
from torch.distributed.tensor import DeviceMesh
from torch.optim.optimizer import ParamsT

from ..config_utils import function_config_to_basic_types
from .dion.dion.normuon import NorMuon
from .muon_fused import zeropower_via_newtonschulz6_diff_abc


class NorMuonFSDP(NorMuon):
    def __init__(
        self,
        params: ParamsT,
        distributed_mesh: Optional[Union[DeviceMesh, ProcessGroup]] = None,
        # defaults
        lr: float = 0.01,
        mu: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.95),
        muon_beta2: float = 0.95,
        weight_decay: float = 0.01,
        cautious_wd: bool = False,
        epsilon: float = 1e-8,
        nesterov: bool = False,
        adjust_lr: Optional[str] = "rms_norm",
        flatten: bool = True,  # NOTE: set to True by default if using 4D tensors, e.g., Conv2D weights
        use_triton: bool = False,
        use_preconditioned: bool = False,
        newton_schulz_func: Optional[Callable] = zeropower_via_newtonschulz6_diff_abc,
        muon_steps: int = 5,
        *,
        # normuon and adamw/lion param group defaults
        muon_params_defaults: dict[str, Any] = {},
        oned_params_defaults: dict[str, Any] = {},
    ):
        if newton_schulz_func is not None:
            newton_schulz_func = lambda G, epsilon: zeropower_via_newtonschulz6_diff_abc(
                G=G,
                steps=muon_steps,
                use_triton=use_triton,
                preconditioned=use_preconditioned,
                epsilon=epsilon,
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
            muon_beta2,
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
            if group["algorithm"] == "normuon":
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
        named_params: Iterable | dict[str, torch.nn.Parameter],
        ignored_keys_for_muon: tuple | list = (),  # for re to match
        oned_param_algo: str = "lion",
    ):
        import re

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

        muon_params_g = {"params": muon_params, "algorithm": "normuon"}
        oned_params_g = {"params": oned_params, "algorithm": oned_param_algo}

        return muon_params_g, oned_params_g

    @classmethod
    @function_config_to_basic_types
    def create_normuon_optimizer(
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
            f"NorMuonFSDP(\n"
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
    optimizer = NormMuonFSDP.create_normuon_optimizer(
        network.named_parameters(),
        ignored_keys_for_muon=("fc",),
        muon_params_defaults={"lr": 1e-3, "weight_decay": 1e-4},
        oned_params_defaults={"lr": 1e-4, "weight_decay": 1e-5},
        betas=(0.95, 0.99),
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
    """python -m src.utilities.optim.normuon_fused"""
    __test_muon_fsdp()

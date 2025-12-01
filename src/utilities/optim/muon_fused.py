from functools import partial
from typing import Any, Callable, Iterable, Optional, Tuple, Union

import torch
from loguru import logger
from omegaconf import OmegaConf
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.optim.optimizer import Optimizer, ParamsT

from ..config_utils import function_config_to_basic_types
from .dion.dion.muon import Muon
from .dion.dion.newton_schulz_triton import ns_line_1, ns_line_2

######### Newton-Schulz ABCs ##########

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


@torch.compile(dynamic=False, fullgraph=True)
def zeropower_via_newtonschulz6_diff_abc(
    G,
    steps: int = 6,
    norm=False,
    ns_dtype=torch.bfloat16,
    use_triton=False,
    epsilon=1e-8,
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
    global abc_s

    assert (
        G.ndim >= 2
    )  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng

    X = G.to(dtype=ns_dtype)
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + epsilon)  # epsilon: 1e-7
    iters = abc_s[:steps] + max(steps - len(abc_s), 0) * abc_s[-1:]

    if use_triton:
        X = X.contiguous()
        A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
        B = torch.empty_like(A)
        C = torch.empty_like(X)
        ns_line_3 = torch.baddbmm if X.ndim > 2 else torch.addmm

    for i, (a, b, c) in enumerate(iters):
        if use_triton:
            # triton code
            ns_line_1(X, out=A)  # A = X @ X.mT
            ns_line_2(A, alpha=c, beta=b, out=B)  # B = b * A + c * A @ A
            ns_line_3(X, B, X, beta=a, out=C)  # C = a * X + B @ X
            X, C = C, X  # Swap references to avoid unnecessary copies
        else:
            # naive python code
            A = X @ X.mT
            A2 = A @ A
            if i == 0 and norm:
                n = ((A2**2).sum(dim=(-2, -1), keepdim=True) + epsilon) ** 0.125
                X, A, A2 = X / n, A / n**2, A2 / n**4
            B = (
                b * A + c * A2
            )  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
            X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class MuonFSDP(Muon):
    def __init__(
        self,
        params: ParamsT,
        distributed_mesh: Optional[Union[DeviceMesh, ProcessGroup]] = None,
        # defaults
        lr: float = 0.01,
        mu: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        epsilon: float = 1e-8,
        nesterov: bool = False,
        adjust_lr: Optional[str] = "spectral_norm",
        flatten: bool = True,  # NOTE: set to True by default
        use_triton: bool = False,
        newton_schulz_func: Optional[Callable] = zeropower_via_newtonschulz6_diff_abc,
        force_my_triton: bool = True,
        *,
        # muon and adamw/lion param group defaults
        muon_params_defaults: dict[str, Any] = {},
        oned_params_defaults: dict[str, Any] = {},
    ):
        if use_triton:
            if force_my_triton:
                newton_schulz_func = partial(zeropower_via_newtonschulz6_diff_abc, use_triton=True)
            else:
                newton_schulz_func = None
        if newton_schulz_func is not None:
            logger.info(f"[Muon]: will self-defined Newton-Schulz function")

        super().__init__(
            params,
            distributed_mesh,
            lr,
            mu,
            betas,
            weight_decay,
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
                group.update(muon_params_defaults)
            elif group["algorithm"] in ("adamw", "lion"):
                group.update(oned_params_defaults)
            else:
                raise ValueError(f"Unknown algorithm: {group['algorithm']}")

    @classmethod
    @function_config_to_basic_types
    def clear_muon_adamw_params(
        cls,
        named_params: Iterable | dict[str, torch.nn.Parameter],
        ignored_keys_for_muon: tuple | list = (),
        oned_param_algo: str = "lion",
    ):
        muon_params = []
        oned_params = []

        if isinstance(named_params, dict):
            named_params = named_params.items()

        for name, p in named_params:
            if p.requires_grad:
                # conv, linear weights, and other 2D+ parameters
                if p.ndim >= 2 and name not in ignored_keys_for_muon:
                    muon_params.append(p)
                    # logger.debug(f"Muon params: {name} - shaped: {tuple(p.shape)}")
                # bias, norm weights, embeddings, lm heads (for nlp tasks), and other 1D parameters
                else:
                    oned_params.append(p)
                    # logger.debug(
                    #     f"{oned_param_algo} params: {name} - shaped: {tuple(p.shape)}"
                    # )
            # else:
            #     logger.debug(f"{name} is not requires_grad")

        muon_params_g = {"params": muon_params, "algorithm": "muon"}
        oned_params_g = {"params": oned_params, "algorithm": oned_param_algo}

        # # Update with additional group params if provided
        # muon_params_g.update(muon_group_params)
        # oned_params_g.update(oned_group_params)

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
    optimizer = MuonFSDP.create_optimizer(
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
    """python -m src.utilities.optim.muon_fused"""
    __test_muon_fsdp()

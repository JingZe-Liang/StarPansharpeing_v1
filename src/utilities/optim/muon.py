"""
Copied from https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py

distributed version from MagtronLLM:
https://github.com/NVIDIA/Megatron-LM/pull/1428/files/f432fbe45c169aeb5a0805ff6f41e13f989c6730#diff-8fe91f4096ff232fc6f97b17e60e619eda92b6dffc80b4573a23e06aa56d2559
"""

import math
from typing import Iterable, cast

import torch
from loguru import logger
from torch.distributed.tensor import DTensor

from .utils import to_dist, to_local

try:
    from flash_muon_cuda import matmul_transpose_assign

    _flash_muon_cuda_available = True
except ImportError:
    __url = "https://github.com/nil0x9/flash-muon"
    _flash_muon_cuda_available = False
    print(
        "Flash-Muon-CUDA not installed, "
        f"please install it from {__url} if you want to use the cuda kernel ns optimization"
    )

_cuda_dim_in_min = 8
__abc_s_type = "su"

if __abc_s_type == "su":  # JianlinSu's abc_s
    abc_s = [
        (3.86230469, -8.11132812, 4.89062500),
        (3.64746094, -6.52441406, 3.38183594),
        (3.70996094, -6.34667969, 3.13574219),
        (3.92480469, -6.23535156, 2.83789062),
        (2.61425781, -2.95800781, 1.13476562),
        (2.12109375, -1.79003906, 0.66601562),
    ]
elif __abc_s_type == "su2":  # JianlinSu's abc_s version2
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
    abc_s = abc_s.detach().cpu().numpy().tolist()  # convert to list for compatibility
elif __abc_s_type == "you":  # YouJiaChen's abc_s
    # Only for 5 NS steps
    abc_s = [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]
else:
    raise ValueError(f"Unknown abc_s type: {__abc_s_type}")
abc_s = cast(list[list], abc_s)


# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
@torch.compile
def zeropower_via_newtonschulz5(G, steps: int):
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
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    eps = 1e-8  # 1e-7 as eps by default
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = (
            b * A + c * A @ A
        )  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


@torch.compile
def zeropower_via_newtonschulz6_diff_abc(
    G, steps: int = 6, norm=False, ns_dtype=torch.bfloat16
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

    # Different steps use different a, b, c: https://kexue.fm/archives/10922
    # abc_s = torch.tensor([
    #     (3955, -8306, 5008),
    #     (3735, -6681, 3463),
    #     (3799, -6499, 3211),
    #     (4019, -6385, 2906),
    #     (2677, -3029, 1162),
    #     (2172, -1833, 682),
    # ]) / 1024

    global abc_s, __abc_s_type

    X = G.to(dtype=ns_dtype)
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    eps = 1e-8  # 1e-7 as eps by default
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    # Perform the NS iterations
    # if __abc_s_type == 'su':
    #     iters=abc_s
    # elif __abc_s_type == 'su2':
    #     iters = abc_s[: steps] + max(steps - 7, 0) * abc_s[-1:]
    abc_s = cast(list[list], abc_s)
    iters = abc_s[:steps] + max(steps - len(abc_s), 0) * abc_s[-1:]

    for i, (a, b, c) in enumerate(iters):
        A = X @ X.mT
        A2 = A @ A
        if i == 0 and norm:
            n = ((A2**2).sum(dim=(-2, -1), keepdim=True) + eps) ** 0.125
            X, A, A2 = X / n, A / n**2, A2 / n**4
        B = (
            b * A + c * A2
        )  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def fast_newtonschulz(G: torch.Tensor, steps: int) -> torch.Tensor:
    """
    adapted from https://github.com/KellerJordan/Muon/blob/master/muon.py
    Arguments:
        G: The gradient or momentum matrix to be orthogonalized.
        steps: Number of Newton-Schulz iterations.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    buf1 = torch.empty(X.size(0), X.size(0), dtype=X.dtype, device=X.device)
    buf2 = torch.empty(X.size(0), X.size(0), dtype=X.dtype, device=X.device)

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        matmul_transpose_assign(X, buf1)
        matmul_transpose_assign(buf1, buf2)
        B = b * buf1 + c * buf2
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(
        self,
        lr=1e-3,
        weight_decay=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        ns_norm=False,
        ns_diff_abc=False,
        adamw_params=None,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        use_cuda_kernel: bool = False,
    ):
        defaults = dict(
            lr=lr,
            wd=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps if not ns_diff_abc else 6,  # ns_diff_abc use 6 ns steps
            ns_norm=ns_norm,
            ns_diff_abc=ns_diff_abc,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            use_cuda_kernel=use_cuda_kernel and _flash_muon_cuda_available,
        )
        logger.debug(f"[Muon optimizer]: defaults: {defaults}")

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            # assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape):
        if len(param_shape) >= 2:
            A = param_shape[0]
            B = math.prod(list(param_shape[1:]))
        else:
            raise ValueError(
                f"Muon only supports 2D+ parameters, but got param shaped as {param_shape}"
            )

        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            ############################
            #           Muon           #
            ############################

            # TODO: add DDP support !!
            # check the original implementation: https://github.com/KellerJordan/Muon/blob/master/muon.py

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            # import pdb; pdb.set_trace()
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            # generate weight updates in distributed fashion
            for p in params:
                # sanity check
                g = p.grad
                _orig_g_shape = g.shape
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                # calc update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                # to local
                meta = None
                if isinstance(g, DTensor):
                    g, meta = to_local(g, keep_sharded=False)

                if (
                    self.defaults["use_cuda_kernel"]
                    and min(list(g.shape)) % _cuda_dim_in_min == 0
                ):  # cuda kernel layout constraint
                    u = fast_newtonschulz(g, steps=group["ns_steps"])
                elif self.defaults["ns_diff_abc"]:
                    u = zeropower_via_newtonschulz6_diff_abc(
                        g, steps=6, norm=group["ns_norm"]
                    )
                else:
                    u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                # back to shard
                if meta is not None:
                    g = to_dist(g, **meta)

                # scale update
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                # apply weight decay
                p.data.mul_(1 - lr * wd)

                # apply update
                p.data.add_(u.view(_orig_g_shape), alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group["lr"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss

    @classmethod
    def clear_muon_adamw_params(
        cls,
        named_params: Iterable | dict,
        ignored_keys_for_muon: tuple | list = ("embed_tokens", "lm_head"),
    ):
        # # ndim <= 2 in muon optimization
        # muon_params = [
        #     p
        #     for name, p in named_params
        #     if p.ndim >= 2 and name not in ignored_keys_for_muon
        # ]

        # # ndim > 2 or embed_tokens or lm_head in adamw optimization
        # adamw_params = [
        #     p
        #     for name, p in named_params
        #     if not (p.ndim < 2 and "embed_tokens" not in name and "lm_head" not in name)
        # ]

        muon_params = []
        adamw_params = []

        if isinstance(named_params, dict):
            named_params = named_params.items()

        for name, p in named_params:
            if p.requires_grad:
                if p.ndim >= 2 and name not in ignored_keys_for_muon:
                    muon_params.append(p)
                    logger.debug(f"Muon params: {name} - shaped: {p.shape}")
                else:
                    adamw_params.append(p)
                    logger.debug(f"AdamW params: {name} - shaped: {p.shape}")
            else:
                logger.debug(f"{name} is not requires_grad")

        return muon_params, adamw_params


if __name__ == "__main__":
    net = torch.nn.Linear(32, 32, bias=False).cuda()
    optimizer = Muon(muon_params=list(net.parameters()), lr=0.01, use_cuda_kernel=True)

    from tqdm import trange

    for _ in trange(100):
        optimizer.zero_grad()
        loss = net(torch.randn(32, 32).cuda()).sum()
        loss.backward()
        optimizer.step()

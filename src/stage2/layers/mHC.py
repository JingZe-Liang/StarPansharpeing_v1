from typing import Callable

from functools import partial
from random import randrange

import torch
from torch import nn
from torch.nn import Module
from torch.utils._pytree import tree_flatten, tree_unflatten

from einops import rearrange, einsum, repeat
from einops.layers.torch import Rearrange, Reduce


"""
ein notation:
b - batch
d - feature dimension
s - residual streams
t - residual streams + num branch inputs
f - number of fractions (division of feature dimension space)
v - number of views for branch input
"""

# helper functions


def exists(v):
    return v is not None


def divisible_by(num, den):
    return (num % den) == 0


def default(v, d):
    return v if exists(v) else d


def add(x, y):
    return x + y


def sinkhorn_log(logits, num_iters=10, tau=0.05):
    n = logits.shape[-1]
    Z = logits / tau
    log_marginal = torch.zeros((n,), device=logits.device, dtype=logits.dtype)

    u = torch.zeros(logits.shape[:-1], device=Z.device, dtype=Z.dtype)
    v = torch.zeros_like(u)

    for _ in range(num_iters):
        u = log_marginal - torch.logsumexp(Z + v.unsqueeze(-2), dim=-1)
        v = log_marginal - torch.logsumexp(Z + u.unsqueeze(-1), dim=-2)

    return torch.exp(Z + u.unsqueeze(-1) + v.unsqueeze(-2))


# residual base class


class Residual(Module):
    def __init__(
        self,
        *args,
        branch: Module | None = None,
        residual_transform: Module | None = None,
        **kwargs,
    ):
        super().__init__()
        self.branch = branch
        self.residual_transform = default(residual_transform, nn.Identity())

    def width_connection(self, residuals):
        return residuals, residuals, dict()

    def depth_connection(
        self,
        branch_output,
        residuals,
    ):
        return branch_output + self.residual_transform(residuals)

    def decorate_branch(self, branch: Callable):
        assert not exists(self.branch), "branch was already wrapped on init"

        def forward_and_add_residual(residual, *args, **kwargs):
            branch_input, add_residual = self.forward(residual)

            branch_output = branch(branch_input, *args, **kwargs)

            residual = add_residual(branch_output)

            return residual

        return forward_and_add_residual

    def forward(self, residuals, *branch_args, **branch_kwargs):
        branch_input, residuals, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_out):
            (branch_out, *rest), tree_spec = tree_flatten(branch_out)

            branch_out = self.depth_connection(branch_out, residuals, **residual_kwargs)

            return tree_unflatten((branch_out, *rest), tree_spec)

        if not exists(self.branch):
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)

        return add_residual_fn(branch_output)


# stream embed


class StreamEmbed(Module):
    def __init__(self, num_streams, dim, channel_first=False, expand_to_streams=False):
        super().__init__()
        self.channel_first = channel_first
        self.num_streams = num_streams

        self.expand_to_streams = expand_to_streams
        self.stream_embed = nn.Parameter(torch.zeros(num_streams, dim))

    def forward(self, residuals):
        if self.expand_to_streams:
            residuals = repeat(residuals, "b ... -> (b s) ...", s=self.num_streams)

        if self.channel_first:
            residuals = rearrange(residuals, "(b s) d ... -> b ... s d", s=self.num_streams)
        else:
            residuals = rearrange(residuals, "(b s) ... d -> b ... s d", s=self.num_streams)

        residuals = residuals + self.stream_embed

        if self.channel_first:
            residuals = rearrange(residuals, "b ... s d -> (b s) d ...", s=self.num_streams)
        else:
            residuals = rearrange(residuals, "b ... s d -> (b s) ... d", s=self.num_streams)

        return residuals


# main functions


def get_expand_reduce_stream_functions(num_streams, add_stream_embed=False, dim=None, disable=False):
    if num_streams == 1 or disable:
        return (nn.Identity(), nn.Identity())

    if add_stream_embed:
        assert exists(dim), (
            "`dim` must be passed into get_init_and_expand_reduce_stream_functions for returning an expansion function with stream embeddings added"
        )

        expand_fn = StreamEmbed(num_streams, dim, expand_to_streams=True)
    else:
        expand_fn = Reduce(pattern="b ... -> (b s) ...", reduction="repeat", s=num_streams)

    reduce_fn = Reduce(pattern="(b s) ... -> b ...", reduction="sum", s=num_streams)

    return expand_fn, reduce_fn


def get_init_and_expand_reduce_stream_functions(
    num_streams,
    num_fracs=1,
    dim=None,
    add_stream_embed=False,
    disable=None,
):
    disable = default(disable, num_streams == 1 and num_fracs == 1)

    hyper_conn_klass = HyperConnections if not disable else Residual

    init_hyper_conn_fn = partial(hyper_conn_klass, num_streams, num_fracs=num_fracs)
    expand_reduce_fns = get_expand_reduce_stream_functions(
        num_streams, add_stream_embed=add_stream_embed, dim=dim, disable=disable
    )

    if exists(dim):
        init_hyper_conn_fn = partial(init_hyper_conn_fn, dim=dim)

    return (init_hyper_conn_fn, *expand_reduce_fns)


# main classes


class HyperConnections(Module):
    def __init__(
        self,
        num_residual_streams,
        *,
        dim,
        branch: Module | None = None,
        layer_index=None,
        channel_first=False,
        dropout=0.0,
        residual_transform: Module
        | None = None,  # to support resnet blocks where dimension in not equal to dimension out - usually a residual conv
        add_branch_out_to_residual=True,  # will disable depth connections (weighted residual sum with beta) if set False
        num_input_views=1,  # allow for the branch module to receive multiple input views, dimension placed on the very left (before batch)
        depth_residual_fn=add,
        num_fracs=1,  # https://arxiv.org/abs/2503.14125
        mhc_num_iters=10,
        mhc_tau=0.05,
    ):
        """
        Appendix J, Algorithm2 in - https://arxiv.org/abs/2409.19606
        """
        super().__init__()

        self.branch = branch
        self.mhc_num_iters = mhc_num_iters
        self.mhc_tau = mhc_tau

        # frac-connections paper - num_fracs > 1 will be the `m` in their paper https://arxiv.org/abs/2503.14125

        assert num_fracs >= 1
        assert num_fracs == 1, "`num_fracs` must be 1 for mHC"

        self.num_fracs = num_fracs
        self.has_fracs = num_fracs > 1

        self.split_fracs = Rearrange("b ... (f d) -> b ... f d", f=num_fracs)
        self.merge_fracs = Rearrange("b ... f d -> b ... (f d)")

        assert divisible_by(dim, num_fracs), (
            f"feature dimension ({dim}) must be divisible by the `num_fracs` ({num_fracs})"
        )

        dim //= num_fracs  # effective dim handled in dimension is feature dimension divided by num fractions

        assert num_residual_streams > 0, "`num_residual_streams` must be greater than 0"

        self.num_residual_streams = num_residual_streams
        init_residual_index = (
            default(layer_index, randrange(num_residual_streams)) % num_residual_streams
        )  # just choose one random residual stream if layer index not given

        # width num residual streams

        assert num_input_views >= 1
        assert num_input_views == 1, "`num_input_views` must be 1 for mHC"
        self.num_input_views = num_input_views

        self.add_branch_out_to_residual = add_branch_out_to_residual

        # width connection

        init_h_res = torch.full((num_residual_streams, num_residual_streams), -8.0)
        init_h_res.fill_diagonal_(0.0)
        self.H_res_logits = nn.Parameter(init_h_res)

        init_h_pre = torch.full((num_input_views, num_residual_streams), -8.0)
        init_h_pre[:, init_residual_index] = 0.0
        self.H_pre_logits = nn.Parameter(init_h_pre)

        if add_branch_out_to_residual:
            self.H_post_logits = nn.Parameter(torch.zeros(num_input_views, num_residual_streams))

        # dropouts

        self.dropout = nn.Dropout(dropout)

        # channel first option

        self.channel_first = channel_first

        # maybe residual transform

        self.residual_transform = default(residual_transform, nn.Identity())

        # maybe custom depth connection residual function
        # this is to prepare for gating the addition of the branch outputs to the residual streams
        # needed for memory lanes a la RMT / LMM

        self.depth_residual_fn = depth_residual_fn

    def width_connection(self, residuals):
        streams = self.num_residual_streams

        maybe_transformed_residuals = self.residual_transform(residuals)

        # width connection

        if self.channel_first:
            residuals = rearrange(residuals, "b d ... -> b ... d")
            maybe_transformed_residuals = rearrange(maybe_transformed_residuals, "b d ... -> b ... d")

        residuals = self.split_fracs(residuals)
        maybe_transformed_residuals = self.split_fracs(maybe_transformed_residuals)

        residuals = rearrange(residuals, "(b s) ... d -> b ... s d", s=streams)
        maybe_transformed_residuals = rearrange(maybe_transformed_residuals, "(b s) ... d -> b ... s d", s=streams)

        h_res = sinkhorn_log(self.H_res_logits, num_iters=self.mhc_num_iters, tau=self.mhc_tau)
        residuals_out = einsum(h_res, maybe_transformed_residuals, "s t, ... s d -> ... t d")

        h_pre = self.H_pre_logits.softmax(dim=-1)
        branch_input = einsum(h_pre, residuals, "v s, ... s d -> ... v d")

        h_post = None
        if self.add_branch_out_to_residual:
            h_post = self.H_post_logits.softmax(dim=-1)

        if getattr(self, "collect_stats", False):
            with torch.no_grad():
                stats = dict(
                    h_res_min=h_res.min(),
                    h_res_row_sum=h_res.sum(dim=-1).mean(),
                    h_res_col_sum=h_res.sum(dim=-2).mean(),
                    h_pre_min=h_pre.min(),
                )
                if h_post is not None:
                    stats["h_post_min"] = h_post.min()
                self.last_stats = {k: v.detach() for k, v in stats.items()}

        if self.num_input_views == 1:
            branch_input = branch_input[..., 0, :]
        else:
            branch_input = rearrange(branch_input, "b ... v d -> v b ... d")

        if self.channel_first:
            branch_input = rearrange(branch_input, "b ... d -> b d ...")

        branch_input = self.merge_fracs(branch_input)

        residuals_out = rearrange(residuals_out, "b ... s d -> (b s) ... d")
        residuals_out = self.merge_fracs(residuals_out)

        if self.channel_first:
            residuals_out = rearrange(residuals_out, "b ... d -> b d ...")

        return branch_input, residuals_out, dict(beta=h_post)

    def depth_connection(self, branch_output, residuals, *, beta):
        assert self.add_branch_out_to_residual
        assert beta is not None

        branch_output = self.split_fracs(branch_output)

        if self.channel_first:
            branch_output = rearrange(branch_output, "b d ... -> b ... d")

        if beta.ndim == 2:
            beta = beta[0]

        output = einsum(branch_output, beta, "b ... d, s -> b ... s d")
        output = rearrange(output, "b ... s d -> (b s) ... d")

        output = self.merge_fracs(output)

        if self.channel_first:
            output = rearrange(output, "b ... d -> b d ...")

        residuals = self.depth_residual_fn(output, residuals)

        return self.dropout(residuals)

    def decorate_branch(self, branch: Callable):
        assert not exists(self.branch), "branch was already wrapped on init"

        def forward_and_add_residual(residual, *args, **kwargs):
            branch_input, add_residual = self.forward(residual)

            branch_output = branch(branch_input, *args, **kwargs)

            residual = add_residual(branch_output)

            return residual

        return forward_and_add_residual

    def forward(self, residuals, *branch_args, **branch_kwargs):
        branch_input, residuals, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_out):
            if not self.add_branch_out_to_residual:
                return branch_out

            (branch_out, *rest), tree_spec = tree_flatten(branch_out)

            branch_out = self.depth_connection(branch_out, residuals, **residual_kwargs)

            return tree_unflatten((branch_out, *rest), tree_spec)

        if not exists(self.branch):
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)

        return add_residual_fn(branch_output)


HyperConnections.get_expand_reduce_stream_functions = staticmethod(get_expand_reduce_stream_functions)
HyperConnections.get_init_and_expand_reduce_stream_functions = staticmethod(get_init_and_expand_reduce_stream_functions)


def _require_mhc_ops():
    try:
        from mhc import ops as mhc_ops
    except ImportError as exc:
        raise ImportError(
            "Cannot find `mhc` CUDA extension (mHC.cu). Please install the extension first before using `HyperConnectionsCUDA`."
        ) from exc
    return mhc_ops


def get_init_and_expand_reduce_stream_functions_cuda(
    num_streams: int,
    num_fracs: int = 1,
    dim: int | None = None,
    add_stream_embed: bool = False,
    disable: bool | None = None,
):
    disable = default(disable, num_streams == 1 and num_fracs == 1)

    hyper_conn_klass = HyperConnectionsCUDA if not disable else Residual

    init_hyper_conn_fn = partial(hyper_conn_klass, num_streams, num_fracs=num_fracs)
    expand_reduce_fns = get_expand_reduce_stream_functions(
        num_streams, add_stream_embed=add_stream_embed, dim=dim, disable=disable
    )

    if exists(dim):
        init_hyper_conn_fn = partial(init_hyper_conn_fn, dim=dim)

    return (init_hyper_conn_fn, *expand_reduce_fns)


class HyperConnectionsCUDA(Module):
    """
    CUDA version of mHC wrapper: maintains multi-stream residuals throughout (streams folded into batch dimension),
    and allows arbitrary branch (attention or MLP) to be inserted on the aggregated single-stream view,
    then injects branch output back to multi-stream residuals.

    Expects input residuals shape of `[(B * s), T, C]`, output has the same shape.
    """

    def __init__(
        self,
        num_residual_streams: int,
        *,
        dim: int,
        branch: Module | None = None,
        layer_index: int | None = None,
        dropout: float = 0.0,
        add_branch_out_to_residual: bool = True,
        num_fracs: int = 1,
        sinkhorn_iters: int = 20,
        eps: float = 1e-5,
        alpha_init: float = 0.01,
        **_,
    ):
        super().__init__()

        assert num_fracs == 1, "`num_fracs` must be 1 for CUDA mHC wrapper"
        assert num_residual_streams > 0, "`num_residual_streams` must be greater than 0"

        self.branch = branch
        self.num_residual_streams = num_residual_streams
        self.add_branch_out_to_residual = add_branch_out_to_residual

        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps

        self.dropout = nn.Dropout(dropout)
        self.rmsnorm_weight = nn.Parameter(torch.ones(dim, dtype=torch.bfloat16))

        init_residual_index = default(layer_index, randrange(num_residual_streams)) % num_residual_streams

        init_h_pre = torch.full((num_residual_streams,), -8.0, dtype=torch.float32)
        init_h_pre[init_residual_index] = 0.0
        self.H_pre_logits = nn.Parameter(init_h_pre)

        self.H_post_logits = nn.Parameter(torch.zeros(num_residual_streams, dtype=torch.float32))

        init_h_res = torch.full((num_residual_streams, num_residual_streams), -8.0, dtype=torch.float32)
        init_h_res.fill_diagonal_(0.0)
        if alpha_init != 0.0:
            init_h_res = init_h_res + alpha_init * torch.randn_like(init_h_res)
        self.H_res = nn.Parameter(init_h_res)

    def _h_pre(self):
        return torch.sigmoid(self.H_pre_logits)

    def _h_post(self):
        return 2.0 * torch.sigmoid(self.H_post_logits)

    def _mix_matrix(self):
        mhc_ops = _require_mhc_ops()
        return mhc_ops.sinkhorn_knopp(self.H_res.exp(), num_iters=self.sinkhorn_iters, eps=self.eps)

    def width_connection(self, residuals):
        streams = self.num_residual_streams

        if residuals.ndim != 3:
            raise ValueError(f"`residuals` must be a 3D tensor [(B*s), T, C], but got shape={tuple(residuals.shape)}")
        if residuals.shape[0] % streams != 0:
            raise ValueError(
                f"`residuals.shape[0]` must be divisible by streams, but got {residuals.shape[0]=}, {streams=}"
            )

        mhc_ops = _require_mhc_ops()

        batch = residuals.shape[0] // streams
        seq_len = residuals.shape[1]

        x = rearrange(residuals, "(b s) t d -> (b t) s d", s=streams)
        x_agg = mhc_ops.stream_aggregate(x, self._h_pre())
        x_norm = mhc_ops.rmsnorm(x_agg, self.rmsnorm_weight, eps=self.eps)
        branch_dtype = residuals.dtype
        if self.branch is not None:
            branch_param = next(self.branch.parameters(), None)
            if branch_param is not None:
                branch_dtype = branch_param.dtype

        branch_input = x_norm.to(dtype=branch_dtype).view(batch, seq_len, -1)

        return branch_input, residuals, dict(batch=batch, seq_len=seq_len)

    def depth_connection(self, branch_output, residuals, *, batch, seq_len):
        if not self.add_branch_out_to_residual:
            return branch_output

        streams = self.num_residual_streams
        mhc_ops = _require_mhc_ops()

        if branch_output.shape[:2] != (batch, seq_len):
            raise ValueError(
                f"branch output must be [B, T, C], but got shape={tuple(branch_output.shape)}, expected B={batch}, T={seq_len}"
            )

        x = rearrange(residuals, "(b s) t d -> (b t) s d", s=streams)
        y = branch_output.reshape(batch * seq_len, -1).to(torch.float32).contiguous()
        y_dist = mhc_ops.stream_distribute(y, self._h_post())
        mixed = mhc_ops.stream_mix(y_dist, self._mix_matrix())

        out = x.to(torch.float32) + mixed
        out = rearrange(out, "(b t) s d -> (b s) t d", b=batch, t=seq_len, s=streams)
        return self.dropout(out.to(dtype=residuals.dtype))

    def decorate_branch(self, branch: Callable):
        assert not exists(self.branch), "branch was already wrapped on init"

        def forward_and_add_residual(residual, *args, **kwargs):
            branch_input, add_residual = self.forward(residual)

            branch_output = branch(branch_input, *args, **kwargs)

            residual = add_residual(branch_output)

            return residual

        return forward_and_add_residual

    def forward(self, residuals, *branch_args, **branch_kwargs):
        branch_input, residuals, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_out):
            if not self.add_branch_out_to_residual:
                return branch_out

            (branch_out, *rest), tree_spec = tree_flatten(branch_out)

            branch_out = self.depth_connection(branch_out, residuals, **residual_kwargs)

            return tree_unflatten((branch_out, *rest), tree_spec)

        if not exists(self.branch):
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)

        return add_residual_fn(branch_output)


__all__ = [
    "HyperConnections",
    "Residual",
    "StreamEmbed",
    "get_expand_reduce_stream_functions",
    "get_init_and_expand_reduce_stream_functions",
    "HyperConnectionsCUDA",
    "get_init_and_expand_reduce_stream_functions_cuda",
]

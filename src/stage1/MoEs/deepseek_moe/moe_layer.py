"""
Deepseek MoE implemtation

Author: Deepseek team, Zihan Cao
Date: 2025-01-26
Email: iamzihan666@gmail.com
License: MIT Linsence

---------------------------------------------------------

Copyright (c) Deepseek, 2025
Copyright (c) ZihanCao, University of Electronic Science and Technology of China (UESTC), Mathematical School
"""

import math
import sys
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from src.utilities.logging import log_print


def moe_balance_info(num_experts: int):
    balance_info = {}

    # @torch.profiler.record_function("moe_balance_info")
    def get_balance_info(top_k_index: torch.Tensor):
        # top_k_index: [n, group]
        top_k_index = top_k_index.view(-1)
        total_tokens = top_k_index.shape[0]

        # count the index number in this tensor (chosed experts)
        # 统计每个专家被选中的次数
        unique, counts = torch.unique(top_k_index, return_counts=True)
        counts_dict = dict(zip(unique.cpu().numpy(), counts.cpu().numpy()))

        # 初始化所有专家的计数为0
        for expert_idx in range(num_experts):
            if expert_idx not in counts_dict:
                counts_dict[expert_idx] = 0

        # 更新balance_info
        balance_info["expert_counts"] = counts_dict
        balance_info["total_tokens"] = total_tokens
        balance_info["expert_percent"] = {
            expert_idx: counts_dict[expert_idx] / total_tokens
            for expert_idx in range(num_experts)
        }

        return balance_info

    return get_balance_info


# *==============================================================
# * distributed utils
# *==============================================================


class All2All(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        output_splits: List[int],
        input_splits: List[int],
        group=None,
    ):
        ctx.output_splits = output_splits
        ctx.input_splits = input_splits
        ctx.group = group
        output = (
            input.new_empty(sum(output_splits), *input.shape[1:])
            if output_splits
            else torch.empty_like(input)
        )
        dist.all_to_all_single(output, input, output_splits, input_splits, group)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        output_splits = ctx.output_splits
        input_splits = ctx.input_splits
        group = ctx.group
        grad_input = (
            grad_output.new_empty(sum(input_splits), *grad_output.shape[1:])
            if input_splits
            else torch.empty_like(grad_output)
        )
        dist.all_to_all_single(
            grad_input, grad_output, input_splits, output_splits, group
        )
        return grad_input, None, None, None


# *==============================================================
# * MoE gate, MLP, and load balance loss
# *==============================================================


class DeepseekV2MLP(nn.Module):
    def __init__(self, hidden_size=128, intermediate_size=256, hidden_act="gelu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    @torch.compile
    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class MoEGate(nn.Module):
    def __init__(
        self,
        num_experts_per_tok=2,
        n_routed_experts=4,
        routed_scaling_factor=1.0,
        scoring_func="softmax",
        aux_loss_alpha=0.01,
        seq_aux=False,
        topk_method="gready",
        n_group=2,
        topk_group=1,
        norm_topk_prob=True,
        hidden_size=128,
    ):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.scoring_func = scoring_func
        self.alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group

        # topk selection algorithm
        self.norm_topk_prob = norm_topk_prob
        self.gating_dim = hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.reshape(-1, h)
        logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32), None
        )  # [n, e]
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1, dtype=torch.float32)  # [n, e]
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        ### select top-k experts
        if self.topk_method == "gready":
            topk_weight, topk_idx = torch.topk(
                scores, k=self.top_k, dim=-1, sorted=False
            )  # [n, top_k]
        elif self.topk_method == "group_limited_greedy":
            group_scores = (
                scores.view(bsz * seq_len, self.n_group, -1).max(dim=-1).values
            )  # [n, n_group]
            group_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False
            )[1]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, n_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group
                )
                .reshape(bsz * seq_len, -1)
            )  # [n, e]
            tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
            topk_weight, topk_idx = torch.topk(
                tmp_scores, k=self.top_k, dim=-1, sorted=False
            )

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        else:
            topk_weight = topk_weight * self.routed_scaling_factor
        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)  # [n, l * top_k]
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)  # [n, l, e]
                ce = torch.zeros(
                    bsz, self.n_routed_experts, device=hidden_states.device
                )  # [n, e]
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,  # [n, l * top_k]
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                    dim=1
                ).mean() * self.alpha
            else:
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )  # [n * l * top_k, e]
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss


class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss,
    which includes the gradient of the aux loss during backpropagation.
    """

    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


# *==============================================================
# * MoE interface class
# *==============================================================


class DeepseekV2MoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(
        self,
        num_experts_per_tok: int,
        n_routed_experts: int = 8,
        moe_intermediate_size: int = 512,
        n_shared_experts: int = 1,
        hidden_size: int = 512,
        activation: str = "gelu",
        ep_size: int | None = 1,
        compute_moe_info: bool = False,
        is_first_moe_layer: bool = False,
        **moe_kwargs,
    ):
        super().__init__()
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.get_moe_info = moe_balance_info(n_routed_experts)
        self.compute_moe_info = compute_moe_info
        if ep_size is None:
            ep_size = dist.get_world_size() if dist.is_initialized() else 1
        self.use_ep = ep_size > 1
        self.ep_size = ep_size
        self.is_first_moe_layer = is_first_moe_layer and self.use_ep

        if ep_size > 1 and self.use_ep:
            self.ep_group = None
            self.experts_per_rank = n_routed_experts // ep_size
            self.ep_rank = dist.get_rank() % self.ep_size
            self.experts = nn.ModuleList(
                [
                    (
                        DeepseekV2MLP(
                            intermediate_size=moe_intermediate_size,
                            hidden_size=hidden_size,
                            hidden_act=activation,
                        )
                        if i
                        >= self.ep_rank
                        * self.experts_per_rank  # ep_rank 0, i >= 0, i < 2; ep_rank 1, i >= 2, i < 4
                        and i < (self.ep_rank + 1) * self.experts_per_rank
                        else None
                    )
                    for i in range(n_routed_experts)
                ]  # type: ignore
            )
        else:
            self.ep_size = 1
            self.experts_per_rank = n_routed_experts
            self.ep_rank = 0
            self.experts = nn.ModuleList(
                [
                    DeepseekV2MLP(
                        intermediate_size=moe_intermediate_size,
                        hidden_size=hidden_size,
                        hidden_act=activation,
                    )
                    for i in range(n_routed_experts)
                ]
            )
        # print expert info
        log_print(
            f"[MoE] rank {dist.get_rank() if dist.is_initialized() else 0} "
            f"ep_size {ep_size} "
            f"has valid experts {sum([int(exp is not None) for exp in self.experts])}",
            "debug",
        )

        self.gate = MoEGate(
            num_experts_per_tok=num_experts_per_tok,
            n_routed_experts=n_routed_experts,
            hidden_size=hidden_size,
            **moe_kwargs,
        )
        if n_shared_experts is not None:
            intermediate_size = moe_intermediate_size * n_shared_experts
            self.shared_experts = DeepseekV2MLP(
                intermediate_size=intermediate_size,
                hidden_size=hidden_size,
                hidden_act=activation,
            )

    def _forward_implm(self, hidden_states: torch.Tensor):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])  # [n * l, d]
        flat_topk_idx = topk_idx.reshape(-1)  # [n * l * top_k]
        if self.compute_moe_info:
            moe_info = self.get_moe_info(topk_idx)

        # training
        if self.training:
            # training with EP or not
            if self.ep_size == 1:
                hidden_states = hidden_states.repeat_interleave(
                    self.num_experts_per_tok, dim=0
                )  # [n * l * top_k, d]
                y = torch.empty_like(hidden_states, dtype=hidden_states.dtype)
                for i, expert in enumerate(self.experts):
                    assert expert is not None
                    y[flat_topk_idx == i] = expert(
                        hidden_states[flat_topk_idx == i]
                    ).to(hidden_states.dtype)
            else:
                y = self.moe_ep(hidden_states, topk_idx)

            # reshape and apply auxiliary loss
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.to(hidden_states.dtype).view(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)

        # inference
        else:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)

        # shared experts
        if self.n_shared_experts is not None:
            y = y + self.shared_experts(identity)

        if self.compute_moe_info:
            return y, moe_info
        else:
            return y

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))  # [n * l, e]
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)  # [n * l, e] -> [e]
        idxs = topk_ids.view(-1).argsort()  # [n * l * top_k]
        sorted_tokens = x[idxs // topk_ids.shape[1]]  # [n * l * top_k, d]
        sorted_tokens_shape = sorted_tokens.shape
        if self.ep_size > 1:
            tokens_per_ep_rank = tokens_per_expert.view(self.ep_size, -1).sum(dim=1)
            tokens_per_expert_group = tokens_per_expert.new_empty(
                tokens_per_expert.shape[0]
            )
            dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert)
            output_splits = (
                tokens_per_expert_group.view(self.ep_size, -1)
                .sum(1)
                .cpu()
                .numpy()
                .tolist()
            )
            gathered_tokens = sorted_tokens.new_empty(
                tokens_per_expert_group.sum(dim=0).cpu().item(), sorted_tokens.shape[1]
            )
            input_split_sizes = tokens_per_ep_rank.cpu().numpy().tolist()
            dist.all_to_all(
                list(gathered_tokens.split(output_splits)),
                list(sorted_tokens.split(input_split_sizes)),
            )
            tokens_per_expert_post_gather = tokens_per_expert_group.view(
                self.ep_size, self.experts_per_rank
            ).sum(dim=0)
            gatherd_idxs = np.zeros(shape=(gathered_tokens.shape[0],), dtype=np.int32)
            s = 0
            for i, k in enumerate(tokens_per_expert_group.cpu().numpy()):
                gatherd_idxs[s : s + k] = i % self.experts_per_rank
                s += k
            gatherd_idxs = gatherd_idxs.argsort()
            sorted_tokens = gathered_tokens[gatherd_idxs]
            tokens_per_expert = tokens_per_expert_post_gather
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        if self.ep_size > 1:
            new_x = torch.empty_like(outs)
            new_x[gatherd_idxs] = outs
            gathered_tokens = new_x.new_empty(*sorted_tokens_shape)
            dist.all_to_all(
                list(gathered_tokens.split(input_split_sizes)),
                list(new_x.split(output_splits)),
            )
            outs = gathered_tokens

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out

    def forward(self, hidden_states):
        """
        wrap accelerator backward for EP

        >>> edp_size = (
        ...     world_size
        ...     // ep_size
        ... )
        >>> def custom_backward(self, loss, **kwargs):
        >>>     backward(loss, **kwargs)
        >>>     if not self.sync_gradients or edp_size == 1:
        >>>         return
        >>>     for p in expert_params:
        >>>         g = p.grad if p.grad is not None else torch.zeros_like(p)
        >>>         dist.all_reduce(g, op=dist.ReduceOp.AVG, group=edp_group)
        >>>         if p.grad is not g:
        >>>             p.grad = g
        >>> accelerator.backward = MethodType(
        ...     custom_backward,
        ...     accelerator,
        ... )

        """

        # for EP
        if (
            dist.is_initialized()
            and dist.get_world_size() > 1
            and self.is_first_moe_layer
            and self.use_ep
        ):
            _req_grad = torch.is_grad_enabled()
            hidden_states = hidden_states.requires_grad_(_req_grad)
            return self._forward_implm(hidden_states)
        else:
            return self._forward_implm(hidden_states)

    def moe_ep(self, x, topk_ids):
        dtype_x = x.dtype
        cnts = topk_ids.new_zeros((topk_ids.shape[0], self.n_routed_experts))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // self.num_experts_per_tok]
        if self.ep_size > 1:
            tokens_per_expert_group = torch.empty_like(tokens_per_expert)
            dist.all_to_all_single(
                tokens_per_expert_group, tokens_per_expert, group=self.ep_group
            )
            output_splits = (
                tokens_per_expert_group.view(self.ep_size, -1).sum(dim=1).cpu().tolist()
            )
            input_splits = (
                tokens_per_expert.view(self.ep_size, -1).sum(dim=1).cpu().tolist()
            )
            gathered_tokens = All2All.apply(
                sorted_tokens, output_splits, input_splits, self.ep_group
            )
            gatherd_idxs = idxs.new_empty(gathered_tokens.shape[0], device="cpu")
            s = 0
            for i, k in enumerate(tokens_per_expert_group.cpu()):
                gatherd_idxs[s : s + k] = i % self.experts_per_rank
                s += k
            gatherd_idxs = gatherd_idxs.to(idxs.device).argsort()
            sorted_tokens = gathered_tokens[gatherd_idxs]
            tokens_per_expert = tokens_per_expert_group.view(self.ep_size, -1).sum(
                dim=0
            )
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            if num_tokens == 0:
                continue
            end_idx = start_idx + num_tokens
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            outputs.append(expert(sorted_tokens[start_idx:end_idx]).to(dtype_x))
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        if self.ep_size > 1:
            sorted_tokens = torch.empty_like(outs)
            sorted_tokens[gatherd_idxs] = outs
            gathered_tokens = All2All.apply(
                sorted_tokens, input_splits, output_splits, self.ep_group
            )
            outs = gathered_tokens

        y = torch.empty_like(outs)
        y[idxs] = outs
        return y


# *==============================================================
# * Loss-free load balance
# *==============================================================

# TODO: add loss-free load balance


# *==============================================================
# * Try with this simple MoE version without expert parrellel
# *==============================================================


class DeepseekV1MoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(
        self,
        num_experts_per_tok: int,
        n_routed_experts: int = 8,
        moe_intermediate_size: int = 512,
        n_shared_experts: int = 1,
        hidden_size: int = 512,
        activation: str = "gelu",
        **moe_kwargs,
    ):
        super().__init__()
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.get_moe_info = moe_balance_info(n_routed_experts)
        self.compute_moe_info = moe_kwargs.pop("compute_moe_info", False)

        self.experts = nn.ModuleList(
            [
                DeepseekV2MLP(hidden_size, moe_intermediate_size)
                for i in range(n_routed_experts)
            ]
        )
        self.gate = MoEGate(
            num_experts_per_tok=num_experts_per_tok,
            n_routed_experts=n_routed_experts,
            hidden_size=hidden_size,
            **moe_kwargs,
        )
        if n_shared_experts is not None:
            intermediate_size = moe_intermediate_size * n_shared_experts
            self.shared_experts = DeepseekV2MLP(
                intermediate_size=intermediate_size,
                hidden_size=hidden_size,
                hidden_act=activation,
            )

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            hidden_states = hidden_states.repeat_interleave(
                self.num_experts_per_tok, dim=0
            )
            y = torch.empty_like(hidden_states)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(
                hidden_states, flat_topk_idx, topk_weight.view(-1, 1)
            ).view(*orig_shape)
        if self.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_reduce_(
                0,
                exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]),
                expert_out,
                reduce="sum",
            )
        return expert_cache


# *==============================================================
# * Token choice MoE (EC-DiT)
# *==============================================================


class ECGate(nn.Module):
    """
    Expert Choice (EC) Gating module that implements EC-DIT's routing layer.
    Each expert selects its preferred tokens rather than each token selecting experts.
    """

    def __init__(
        self,
        expert_capacity_per_batch=128,  # 每个专家最多处理多少个令牌
        n_routed_experts=4,
        routed_scaling_factor=1.0,
        scoring_func="softmax",
        hidden_size=128,
    ):
        super().__init__()
        self.expert_capacity = expert_capacity_per_batch
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.scoring_func = scoring_func

        # 路由参数
        self.gating_dim = hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.gating_dim, self.n_routed_experts))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape

        # 1. 计算令牌-专家亲和度分数
        # 使用einsum按照伪代码实现，保持维度正确
        # x_p: [bsz, seq_len, h], W_r: [h, n_experts]
        # logits: [bsz, seq_len, n_experts]
        logits = torch.einsum(
            "bsd,de->bse",
            hidden_states.type(torch.float32),
            self.weight.type(torch.float32),
        )

        if self.scoring_func == "softmax":
            affinity = F.softmax(logits, dim=-1)  # [bsz, seq_len, n_experts]
        else:
            raise NotImplementedError(f"不支持的评分函数: {self.scoring_func}")

        # 转置以获得 [bsz, n_experts, seq_len] 形状，与伪代码一致
        affinity = affinity.permute(0, 2, 1)  # [bsz, n_experts, seq_len]

        # 2. 为每个专家选择top-k个令牌
        expert_capacity = min(seq_len, self.expert_capacity)  # 不能超过序列长度

        # 对于每个专家选择具有最高亲和度分数的令牌
        topk_scores, topk_indices = torch.topk(
            affinity, k=expert_capacity, dim=-1, sorted=True
        )

        # 应用路由缩放因子
        topk_scores = topk_scores * self.routed_scaling_factor

        # 创建分发掩码，与伪代码中的dispatch一致
        # [bsz, n_experts, capacity, seq_len]
        dispatch_mask = torch.zeros(
            bsz,
            self.n_routed_experts,
            expert_capacity,
            seq_len,
            device=hidden_states.device,
        )

        # 为每个专家的选定令牌设置分发掩码
        batch_indices = (
            torch.arange(bsz, device=hidden_states.device)
            .view(-1, 1, 1)
            .expand(bsz, self.n_routed_experts, expert_capacity)
        )
        expert_indices = (
            torch.arange(self.n_routed_experts, device=hidden_states.device)
            .view(1, -1, 1)
            .expand(bsz, self.n_routed_experts, expert_capacity)
        )

        # 设置one-hot编码
        dispatch_mask[
            batch_indices,
            expert_indices,
            torch.arange(expert_capacity, device=hidden_states.device)
            .view(1, 1, -1)
            .expand(bsz, self.n_routed_experts, expert_capacity),
            topk_indices,
        ] = 1.0

        return topk_indices, topk_scores, dispatch_mask


class DeepseekECMoE(nn.Module):
    """
    DeepSeek Mixture of Experts implementation with Expert Choice Routing.
    """

    def __init__(
        self,
        expert_capacity_per_batch=128,
        n_routed_experts: int = 8,
        moe_intermediate_size: int = 512,
        n_shared_experts: int = 1,
        hidden_size: int = 512,
        activation: str = "gelu",
        compute_moe_info: bool = False,
        **moe_kwargs,
    ):
        super().__init__()
        self.expert_capacity = expert_capacity_per_batch
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.get_moe_info = moe_balance_info(n_routed_experts)
        self.compute_moe_info = compute_moe_info
        self.hidden_size = hidden_size

        # 初始化专家
        self.experts = nn.ModuleList(
            [
                DeepseekV2MLP(
                    intermediate_size=moe_intermediate_size,
                    hidden_size=hidden_size,
                    hidden_act=activation,
                )
                for _ in range(n_routed_experts)
            ]
        )

        # 初始化EC路由门控
        self.gate = ECGate(
            expert_capacity_per_batch=expert_capacity_per_batch,
            n_routed_experts=n_routed_experts,
            hidden_size=hidden_size,
            **moe_kwargs,
        )

        # 共享专家 (如果需要)
        if n_shared_experts is not None:
            intermediate_size = moe_intermediate_size * n_shared_experts
            self.shared_experts = DeepseekV2MLP(
                intermediate_size=intermediate_size,
                hidden_size=hidden_size,
                hidden_act=activation,
            )

    def forward(self, hidden_states):
        identity = hidden_states
        bsz, seq_len, h = hidden_states.shape

        # 获取路由信息
        topk_indices, topk_scores, dispatch_mask = self.gate(hidden_states)

        # 计算MOE信息 (如果需要)
        if self.compute_moe_info:
            moe_info = {}
            # 统计每个专家选择的令牌数
            expert_counts = {}
            total_tokens = bsz * seq_len
            expert_percent = {}

            for expert_idx in range(self.n_routed_experts):
                count = (dispatch_mask[:, expert_idx].sum()).item()
                expert_counts[expert_idx] = count
                expert_percent[expert_idx] = count / total_tokens

            moe_info["expert_counts"] = expert_counts
            moe_info["total_tokens"] = total_tokens
            moe_info["expert_percent"] = expert_percent

        # 对每个专家处理选定的令牌
        outputs = torch.zeros_like(hidden_states)
        expert_outputs = []

        for expert_idx in range(self.n_routed_experts):
            # 提取此专家要处理的令牌
            # [bsz, capacity, seq_len] * [bsz, seq_len, h] -> [bsz, capacity, h]
            tokens_for_expert = torch.bmm(
                dispatch_mask[:, expert_idx], hidden_states
            )  # [bsz, capacity, h]

            # 将所有batch的令牌拼接起来一次性处理
            flat_tokens = tokens_for_expert.reshape(-1, h)
            expert = self.experts[expert_idx]
            flat_expert_output = expert(flat_tokens)
            expert_output = flat_expert_output.reshape(
                bsz, self.gate.expert_capacity, h
            )
            expert_outputs.append(expert_output)

        # 组合所有专家的输出
        for expert_idx in range(self.n_routed_experts):
            # 应用专家的权重
            weighted_output = expert_outputs[expert_idx] * topk_scores[
                :, expert_idx
            ].unsqueeze(-1)

            # 将输出分发回原始位置
            # [bsz, capacity, h] * [bsz, capacity, seq_len] -> [bsz, seq_len, h]
            outputs += torch.bmm(
                weighted_output.transpose(1, 2), dispatch_mask[:, expert_idx]
            )

        # 应用共享专家 (如果有)
        if self.n_shared_experts is not None:
            outputs = outputs + self.shared_experts(identity)

        if self.compute_moe_info:
            return outputs, moe_info
        else:
            return outputs


# *==============================================================
# * Testers
# *==============================================================


def __test_expert_routing():
    # 获取当前进程的rank
    # rank = int(os.getenv('RANK', 0))

    # 设置当前进程使用的GPU设备
    # torch.cuda.set_device(rank % torch.cuda.device_count())

    # dist.init_process_group(
    #     backend='nccl',
    #     init_method='env://',
    #     rank=rank,
    #     world_size=2,
    # )
    # dist.barrier()

    # 初始化模型
    moe = DeepseekV2MoE(
        num_experts_per_tok=2,
        n_routed_experts=4,
        moe_intermediate_size=512,
        n_shared_experts=1,
        hidden_size=128,
        activation="gelu",
        topk_method="group_limited_greedy",
        n_group=2,  # 8
        topk_group=1,  # 3
        norm_topk_prob=False,
        compute_moe_info=True,
        is_first_moe_layer=True,
        ep_size=1,
    ).cuda()  # 将模型移动到GPU

    # 测试输入
    batch_size = 2
    seq_len = 128
    hidden_size = 128
    test_input = torch.randn(
        batch_size, seq_len, hidden_size
    ).cuda()  # 将输入数据移动到GPU
    out = moe(test_input)
    print(out[0].shape)
    print(out[1])

    # EP backward
    out[0].mean().backward()
    epd_size = dist.get_world_size() // moe.ep_size if dist.is_initialized() else -1
    if epd_size != 1 and moe.ep_size != 1:
        for name, p in moe.named_parameters():
            if p.requires_grad and ".expert" in name:  # if is expert parameter
                g = p.grad if p.grad is not None else torch.zeros_like(p)
                dist.all_reduce(g, op=dist.ReduceOp.AVG, group=moe.ep_group)
                if p.grad is not g:
                    p.grad = g

    # dist.barrier()

    # check grad
    for n, p in moe.named_parameters():
        if p.requires_grad and p.grad is None:
            print(f"{n} grad is None")

    # dist.destroy_process_group()

    # # 简化后的profiler配置
    # with torch.profiler.profile(
    #     activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],  # 只监控CPU
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True
    # ) as prof:
    #     for _ in range(5):  # 运行多次以获得更准确的结果
    #         # 前向传播
    #         output, moe_info = moe(test_input)

    # # 获取moe_balance_info的运行时间
    # profile_results = prof.key_averages()
    # for item in profile_results:
    #     if 'moe_balance_info' in item.key:
    #         print(f"moe_balance_info 运行时间: CPU总时间={item.cpu_time_total}微秒, 调用次数={item.count}")

    # # 打印最耗时的前10个操作
    # print(profile_results.table(sort_by="cpu_time_total", row_limit=10))

    # # 验证输出形状
    # assert output.shape == (batch_size, seq_len, hidden_size), "输出形状不正确"

    # print("MOE 测试通过！")


def __test_token_routing():
    ec_moe = DeepseekECMoE(
        expert_capacity_per_batch=32,  # 每个专家处理的最大令牌数
        n_routed_experts=4,
        moe_intermediate_size=512,
        n_shared_experts=1,
        hidden_size=128,
        activation="gelu",
        compute_moe_info=True,
    ).cuda()

    batch_size = 2
    seq_len = 128
    hidden_size = 128
    test_input = torch.randn(batch_size, seq_len, hidden_size).cuda()

    out = ec_moe(test_input)

    if isinstance(out, tuple):
        output, moe_info = out
        print(f"output shape: {output.shape}")
        print(f"MOE info: {moe_info}")
    else:
        print(f"output shape: {out.shape}")

    # 反向传播测试
    if isinstance(out, tuple):
        out[0].mean().backward()
    else:
        out.mean().backward()

    print("EC-MoE test pass!")


if __name__ == "__main__":
    # __test_expert_routing()
    __test_token_routing()

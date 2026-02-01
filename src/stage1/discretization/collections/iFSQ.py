import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import einsum
from einops import rearrange

import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange, pack, unpack
from typing import List, Optional
from torch import Tensor, int32, int64
from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor, int32, int64

from einops import rearrange, pack, unpack


def exists(v):
    return v is not None


def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


# tensor helpers


def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


act_fun_mapping = {
    "scale_sigmoid_16": lambda x: 2.0 * torch.sigmoid(1.6 * x) - 1.0,
    "scale_sigmoid_20": lambda x: 2.0 * torch.sigmoid(2.0 * x) - 1.0,  # 2.0 is tanh
}


class GroupFSQ(nn.Module):
    def __init__(
        self,
        levels: List[int] = [8, 8, 8, 8],
        dim: Optional[int] = None,
        num_codebooks: int = 1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
        factorized_bits: Optional[List[int]] = None,  # 新增参数
        do_simple_bound: bool = False,
        act_fun: str = "tanh",
        **kwargs,
    ):
        super().__init__()
        self.act_fun = act_fun_mapping[act_fun]
        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent=False)

        # 原始全局 basis（仍保留作为方便/兼容）
        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int64)
        self.register_buffer("_basis", _basis, persistent=False)

        self.scale = scale
        self.do_simple_bound = do_simple_bound

        # 基本维度
        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim) if has_projections else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim) if has_projections else nn.Identity()
        self.has_projections = has_projections

        self.codebook_size = int(self._levels.prod().item())

        # ---------- factorized bits 处理 ----------
        # factorized_bits 指每个 group 包含的 codebook 维度数，例如 [4,4] 表示把 codebook_dim=8 拆成两组，每组 4 维
        if factorized_bits is None:
            # 默认单组行为：整个 codebook_dim 作为一组（与旧逻辑一致）
            self.factorized_bits = [self.codebook_dim]
        else:
            assert sum(factorized_bits) == self.codebook_dim, "sum(factorized_bits) must equal codebook_dim"
            self.factorized_bits = factorized_bits

        # 为每个 group 计算 group_levels（levels 的 slice）和 group_basis
        self.group_levels = []
        self.group_bases = nn.ModuleList()  # 只是用来存放，下面实际以 buffer 方式注册
        start = 0
        self._group_basis_tensors = []  # 暂存以便注册
        self.group_codebook_sizes = []
        for i, g in enumerate(self.factorized_bits):
            gl = levels[start : start + g]
            start += g
            self.group_levels.append(torch.tensor(gl, dtype=int32))
            # group basis: cumprod([1] + gl[:-1])
            gb = torch.cumprod(torch.tensor([1] + gl[:-1], dtype=int64), dim=0)
            # 注册为 buffer（命名唯一）
            self.register_buffer(f"_basis_group_{i}", gb, persistent=False)
            self._group_basis_tensors.append(gb)
            # group codebook size
            self.group_codebook_sizes.append(int(torch.tensor(gl, dtype=int64).prod().item()))

        self.num_groups = len(self.factorized_bits)

    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def simple_bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        half_l = (self._levels - 1) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        assert torch.sum(offset) == 0
        return self.act_fun(z) * half_l - offset

    def quantize(self, z: Tensor) -> Tensor:
        quantized = round_ste(self.simple_bound(z) if self.do_simple_bound else self.bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift_group(self, zhat_normalized: Tensor, group_levels: torch.Tensor) -> Tensor:
        # zhat_normalized: (..., g) 其中 g = len(group_levels)
        half_width = group_levels.to(zhat_normalized.device, non_blocking=True) // 2
        # half_width 是 tensor，需要和 zhat_normalized 广播
        return (zhat_normalized * half_width) + half_width

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """
        输入 zhat (..., codebook_dim) 或者 (..., c, codebook_dim)
        返回 per-group indices，shape (..., num_groups) 或 (..., c, num_groups)
        """
        assert zhat.shape[-1] == self.codebook_dim, f"expected last dim {self.codebook_dim} but got {zhat.shape[-1]}"

        # 支持前导 dims 任意
        lead_shape = zhat.shape[:-1]
        # 把最后一维按 group 切分
        splits = list(self.factorized_bits)
        z_splits = list(torch.split(zhat, splits, dim=-1))  # 每项 shape (..., g)

        idxs = []
        for i, zgroup in enumerate(z_splits):
            # 获取对应的 basis buffer
            gb = getattr(self, f"_basis_group_{i}")  # shape (g,)
            # 先做 scale&shift（和原逻辑一致，但用 group_levels）
            gl = self.group_levels[i]
            zgroup_scaled = self._scale_and_shift_group(zgroup, gl)
            # zgroup_scaled 的最后一维与 gb 长度相同，按 mixed-radix 乘 basis 求和
            # 需要把 gb 的形状广播到 zgroup_scaled 的前导 dims
            idx_group = (zgroup_scaled * gb.to(zgroup_scaled.device)).sum(dim=-1).to(int64)
            idxs.append(idx_group)

        # stack 每组 index，最后一维是 num_groups
        return torch.stack(idxs, dim=-1)  # shape (..., num_groups)

    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def indices_to_codes(self, indices: Tensor, project_out=False) -> Tensor:
        """Inverse of `codes_to_indices`.

        支持输入形式：
          - (..., num_groups)            # e.g. (b, n, g) 或 (b, h, w, g)
          - (..., num_codebooks, num_groups)  # e.g. (b, n, c, g)
        输出 codes 语义与原实现一致：最终会返回 codes（可能是 (..., c, d) 或在 keep_num_codebooks_dim 时合并成 (..., c*d)）。
        """
        # 判断是否为 image/video 风格（简单判定：至少 3 维且最后一维为 num_groups）
        if indices.ndim == 2:
            indices = indices.unsqueeze(-1)
        is_img_or_video = indices.ndim >= 3 and indices.shape[-1] == self.num_groups

        # 统一到 (..., c, g) 形式
        if indices.shape[-1] != self.num_groups:
            raise ValueError(f"Expected last dim == num_groups ({self.num_groups}), got {indices.shape[-1]}")

        # 现在 idx 形状为 (..., c, num_groups)
        # 我们要把每个 group 的单个索引展开回该组内部的多个量化维
        group_codes_list = []
        device = indices.device
        for gi in range(self.num_groups):
            # 取出 group 的 basis 和 levels（basis 已在 __init__ 注册为 buffer）
            gb = getattr(self, f"_basis_group_{gi}").to(device)  # shape (g_len,)
            gl = self.group_levels[gi].to(device)  # shape (g_len,)
            idx_g = indices[..., gi]
            idx_g = rearrange(idx_g, "... -> ... 1")
            # 计算组内各个 digit： (idx // basis) % level
            digits = (idx_g // gb) % gl  # (..., c, g_len)
            group_codes_list.append(digits)

        codes_non_centered = torch.cat(group_codes_list, dim=-1)
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, "... c d -> ... (c d)")

        if project_out:
            codes = self.project_out(codes)
        # print(codes.shape, "codes.shape")
        if is_img_or_video:
            codes = rearrange(codes, "b ... d -> b d ...")

        return codes

    def group_index_to_global(self, indices):
        group_sizes = torch.tensor(self.group_codebook_sizes, device=indices.device, dtype=torch.long)  # [s0, s1, ...]
        mults = torch.cumprod(
            torch.cat([torch.tensor([1], device=indices.device), group_sizes[:-1]]),
            dim=0,
        )  # [1, s0, s0*s1, ...]
        # mults shape: (num_groups,)
        # 合成：每个 group index * mults，然后 sum -> per-codebook global index
        codebook_global_idx = (indices * mults).sum(dim=-1)  # -> (b, n, c)
        return codebook_global_idx

    def forward(self, z: Tensor):
        """
        z: 可能是 image-like (B, D, H, W) 或者 (B, N, D)
        返回 out, None, (None, None, indices_flattened)
        indices_flattened 的结构会是扁平化的 int tensor（按原流程保持兼容）
        但内部我们有 (..., c, num_groups) 的 group-wise indices
        """
        is_img_or_video = z.ndim >= 4

        if is_img_or_video:
            z = rearrange(z, "b d ... -> b ... d")
            z, ps = pack_one(z, "b * d")

        assert z.shape[-1] == self.dim, f"expected dimension of {self.dim} but found dimension of {z.shape[-1]}"

        z = self.project_in(z)

        # 变成 (b, n, c, d) 其中 d == codebook_dim
        z = rearrange(z, "b n (c d) -> b n c d", c=self.num_codebooks)

        codes = self.quantize(z)  # (b, n, c, d)
        indices = self.codes_to_indices(codes)  # (b, n, c, num_groups)

        # 还原 codes 用于重构输出
        codes = rearrange(codes, "b n c d -> b n (c d)")
        out = self.project_out(codes)

        # reconstitute image or video dimensions
        if is_img_or_video:
            out = unpack_one(out, ps, "b * d")
            out = rearrange(out, "b ... d -> b d ...")

            # 注意 unpack indices 模式：末尾有 num_groups
            indices = unpack_one(indices, ps, f"b * c g")

        # 如果不保留 num_codebooks dim，则去掉 c 轴（当 num_codebooks==1 且 keep_num_codebooks_dim False）
        if not self.keep_num_codebooks_dim and self.num_codebooks == 1:
            # indices 形状现在是 (..., 1, num_groups) -> 去掉中间那个 1
            indices = rearrange(indices, "... 1 g -> ... g")

        # 最后返回 indices 的扁平表示（与原代码风格一致）
        return out, None, (None, None, indices)


class RunningFSQ(GroupFSQ):
    def __init__(
        self,
        eps=1e-6,
        momentum=0.1,
        **kwargs,
    ):
        super(RunningFSQ, self).__init__(**kwargs)
        self.eps = eps
        self.momentum = momentum
        self.num_channels = num_channels = kwargs["dim"]

        # running stats
        self.register_buffer("running_mean", torch.zeros(1, num_channels, 1, 1))
        self.register_buffer("running_var", torch.ones(1, num_channels, 1, 1))

    def forward(self, x):
        if self.training:
            batch_mean = x.mean(dim=(0, 2, 3), keepdim=True)
            batch_var = x.var(
                dim=(0, 2, 3), keepdim=True, unbiased=False
            )  # 用 batch 统计值做归一化（这部分保留梯度，类似 BatchNorm 的训练模式）

            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * batch_mean)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * batch_var)

        x_fsq, _, (_, _, indices) = super().forward(x)

        return x_fsq, indices

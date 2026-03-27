from __future__ import annotations

import inspect
import math
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import einops
from timm.models.layers import DropPath, to_2tuple, trunc_normal_, get_norm_layer
from loguru import logger
# try:
#     import transformer_engine as te
# except ImportError:
#     te = None

from ..variants.mlp import SwiGLU
from .patch_merge.patch_merge_triton import patch_merge_blc
from .window_process.window_process import (
    WindowProcess,
    WindowProcessReverse,
    set_window_process_backend,
)
from .attn.func_flash_swin import flash_swin_attn_func
from .attn.func_flash_swin_hybrid import hybrid_sdpa_fwd_flash_swin_v3_bwd
from .attn.func_flash_swin_v2 import flash_swin_attn_func_v2
from .attn.func_flash_swin_v3 import flash_swin_attn_func_v3
from .attn.func_swin import mha_core, window_partition, window_reverse

from src.utilities.network_utils import compile_decorator

logger = logger.bind(_name_="swin")


@lru_cache(maxsize=128)
def _cached_shift_attn_mask(height: int, width: int, window_size: int, shift_size: int) -> torch.Tensor:
    img_mask = torch.zeros((1, height, width, 1))
    h_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    w_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    return attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))


def _build_mlp(
    mlp_cls: type[nn.Module],
    dim: int,
    out_dim: int,
    hidden_dim: int,
    drop: float,
    act_layer: type[nn.Module],
    mlp_kwargs: dict[str, object] | None = None,
) -> nn.Module:
    kwargs = {} if mlp_kwargs is None else dict(mlp_kwargs)
    if mlp_cls is SwiGLU:
        kwargs.setdefault("is_fused", None)
        kwargs.setdefault("use_conv", False)

    default_kwargs: dict[str, object] = {
        "in_features": dim,
        "hidden_features": hidden_dim,
        "out_features": out_dim,
        "drop": drop,
        "act_layer": act_layer,
    }
    init_sig = inspect.signature(mlp_cls.__init__)
    params = init_sig.parameters
    accepts_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
    valid_keys = {name for name in params if name != "self"}

    call_kwargs: dict[str, object] = {}
    for key, value in default_kwargs.items():
        if (accepts_var_kwargs or key in valid_keys) and key not in kwargs:
            call_kwargs[key] = value
    for key, value in kwargs.items():
        if accepts_var_kwargs or key in valid_keys:
            call_kwargs[key] = value

    return mlp_cls(**call_kwargs)


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        is_flash (bool): If True, use FlashSwinAttn, else use MHA.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        is_flash=True,
        qkv_bias=True,
        qk_scale=None,
        proj_drop=0.0,
        attn_backend: str | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        if attn_backend is None:
            attn_backend = "triton_v1" if is_flash else "py"
        valid_backends = {"py", "sdpa", "triton_v1", "triton_v2", "triton_v3", "hybrid_v3"}
        if attn_backend not in valid_backends:
            raise ValueError(f"Unsupported attn backend: {attn_backend}, valid={sorted(valid_backends)}")
        self.attn_backend = attn_backend
        if self.attn_backend.startswith("triton") and (dim // num_heads) % 16 != 0:
            raise ValueError("head_dim should be divisible by 16 for triton backends")
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)  # type: ignore[call-non-callable]
        ].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        window_mask = None
        if mask is not None:
            nW = mask.shape[0]
            window_mask = mask.unsqueeze(0).expand(B_ // nW, nW, N, N).reshape(B_, N, N).contiguous()
            window_mask = window_mask.to(device=q.device, dtype=q.dtype)

        if self.attn_backend == "py":
            if mask is not None:
                mask = mask.to(device=q.device, dtype=q.dtype)
            attn = mha_core(q, k, v, relative_position_bias, mask, self.scale)
        elif self.attn_backend == "sdpa":
            attn_bias = relative_position_bias.unsqueeze(0).to(device=q.device, dtype=q.dtype)
            if window_mask is not None:
                attn_bias = attn_bias + window_mask.unsqueeze(1)
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=0.0, scale=self.scale)
        elif self.attn_backend == "triton_v1":
            attn = flash_swin_attn_func(q, k, v, relative_position_bias, window_mask, self.scale)
        elif self.attn_backend == "triton_v2":
            attn = flash_swin_attn_func_v2(q, k, v, relative_position_bias, window_mask, self.scale)
        elif self.attn_backend == "triton_v3":
            attn = flash_swin_attn_func_v3(q, k, v, relative_position_bias, window_mask, self.scale)
        elif self.attn_backend == "hybrid_v3":
            attn = hybrid_sdpa_fwd_flash_swin_v3_bwd(q, k, v, relative_position_bias, window_mask, self.scale)
        else:
            raise RuntimeError(f"Unexpected attn backend: {self.attn_backend}")

        x = einops.rearrange(attn, "b head n c -> b n (head c)")
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, window_size={self.window_size}, "
            f"num_heads={self.num_heads}, attn_backend={self.attn_backend}"
        )

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, actx_layer=nn.SiLU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc12 = nn.Linear(in_features, hidden_features * 2)
        self.act = actx_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x_gate, x_feat = self.fc12(x).chunk(2, dim=-1)
        x = self.act(x_gate) * x_feat
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        is_flash (bool): If True, use FlashSwinAttn, else use MHA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    Example:
        >>> block = SwinTransformerBlock(
        ...     dim=96,
        ...     input_resolution=(56, 56),
        ...     num_heads=3,
        ...     mlp_cls=SwiGLU,
        ...     mlp_kwargs={"is_fused": None, "use_conv": False},
        ... )
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        out_dim=None,
        window_size=7,
        shift_size=0,
        is_flash=True,
        attn_backend: str | None = None,
        window_backend: str = "py",
        mlp_cls: type[nn.Module] = Mlp,
        mlp_kwargs: dict[str, object] | None = None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.SiLU,
        norm_layer=None,
        output_2d=False,
        act_checkpoint=False,
    ):
        super().__init__()
        self.act_checkpoint = act_checkpoint
        # print(f"[Swin]: {act_checkpoint}")
        self.dim = dim
        self.out_dim = dim if out_dim is None else out_dim
        self.input_resolution = self._normalize_input_resolution(input_resolution)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.is_flash = is_flash
        self.window_backend = window_backend
        self.mlp_ratio = mlp_ratio
        self.output_2d = output_2d
        if self.window_backend not in {"py", "triton"}:
            raise ValueError(f"Unsupported window backend: {self.window_backend}")
        if self.window_backend == "triton":
            set_window_process_backend("triton")
        if self.input_resolution is not None and min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        if norm_layer is None:
            norm_layer = get_norm_layer("rmsnorm")
        assert callable(norm_layer), "norm_layer should be a valid nn.Module"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            is_flash=is_flash,
            proj_drop=drop,
            attn_backend=attn_backend,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = _build_mlp(
            mlp_cls=mlp_cls,
            dim=dim,
            out_dim=self.out_dim,
            hidden_dim=mlp_hidden_dim,
            drop=drop,
            act_layer=act_layer,
            mlp_kwargs=mlp_kwargs,
        )
        self.ffn_residual_proj = nn.Identity() if self.out_dim == dim else nn.Linear(dim, self.out_dim)

        attn_mask = None
        if self.shift_size > 0 and self.input_resolution is not None:
            # calculate attention mask for SW-MSA
            attn_mask = self._prepare_shift_attn_mask(self.input_resolution)

        self.register_buffer("attn_mask", attn_mask, persistent=False)

    @staticmethod
    def _normalize_input_resolution(
        input_resolution: tuple[int, int] | list[int] | int | None,
    ) -> tuple[int, int] | None:
        if input_resolution is None:
            return None
        if isinstance(input_resolution, int):
            return (input_resolution, input_resolution)
        if isinstance(input_resolution, (tuple, list)) and len(input_resolution) == 2:
            return (int(input_resolution[0]), int(input_resolution[1]))
        raise ValueError(f"Invalid input_resolution: {input_resolution}")

    def _prepare_shift_attn_mask(self, input_resolution: tuple[int, int]):
        assert isinstance(input_resolution, (tuple, list)), "Input resolution must be provided"
        H, W = input_resolution
        return _cached_shift_attn_mask(H, W, self.window_size, self.shift_size)

    def _infer_hw_from_x(self, x):
        L = x.shape[1]
        # assume x is a square image
        H, W = int(math.sqrt(L)), int(math.sqrt(L))
        return H, W

    def _resolve_runtime_shift_size(self, height: int, width: int) -> int:
        if self.shift_size <= 0:
            return 0
        if self.shift_size >= height or self.shift_size >= width:
            return 0
        return self.shift_size

    def forward_fn(self, x):
        is_nchw_input = x.ndim == 4
        if is_nchw_input:
            B, C, H_2d, W_2d = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B, H_2d * W_2d, C).contiguous()
            H, W = H_2d, W_2d
        elif x.ndim == 3:
            H, W = self.input_resolution if self.input_resolution is not None else self._infer_hw_from_x(x)
        else:
            raise ValueError(f"SwinTransformerBlock expects 3D or 4D input, got ndim={x.ndim}")

        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert C == self.dim, f"input channel mismatch, expected {self.dim}, got {C}"
        runtime_shift_size = self._resolve_runtime_shift_size(H, W)

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.window_backend == "triton":
            x_windows = WindowProcess.apply(
                x.contiguous(),
                B,
                H,
                W,
                C,
                -runtime_shift_size,
                self.window_size,
            )
        else:
            if runtime_shift_size > 0:
                shifted_x = torch.roll(x, shifts=(-runtime_shift_size, -runtime_shift_size), dims=(1, 2))
                x_windows = window_partition(shifted_x, self.window_size)
            else:
                shifted_x = x
                x_windows = window_partition(shifted_x, self.window_size)

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        if runtime_shift_size > 0 and self.input_resolution == (H, W) and self.attn_mask is not None:
            attn_mask = self.attn_mask
        elif runtime_shift_size > 0:
            attn_mask = self._prepare_shift_attn_mask((H, W))
        else:
            attn_mask = None
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        if self.window_backend == "triton":
            x = WindowProcessReverse.apply(
                attn_windows.contiguous(),
                B,
                H,
                W,
                C,
                runtime_shift_size,
                self.window_size,
            )
        else:
            if runtime_shift_size > 0:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)
                x = torch.roll(shifted_x, shifts=(runtime_shift_size, runtime_shift_size), dims=(1, 2))
            else:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)
                x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        x = self.ffn_residual_proj(x) + self.drop_path(self.mlp(self.norm2(x)))
        if is_nchw_input or self.output_2d:
            x = x.view(B, H, W, self.out_dim).permute(0, 3, 1, 2).contiguous()

        return x

    def forward(self, x):
        if self.act_checkpoint:
            return checkpoint.checkpoint(self.forward_fn, x, use_reentrant=False)
        else:
            return self.forward_fn(x)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}, "
            f"window_backend={self.window_backend}, out_dim={self.out_dim}, output_2d={self.output_2d}"
        )

    def flops(self):
        if self.input_resolution is None:
            raise ValueError("input_resolution is required to compute flops")
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        flops += H * W * self.dim * (2 * mlp_hidden_dim)
        flops += H * W * mlp_hidden_dim * self.out_dim
        if self.out_dim != self.dim:
            flops += H * W * self.dim * self.out_dim
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, merge_backend: str = "py"):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.merge_backend = merge_backend
        if self.merge_backend not in {"py", "triton"}:
            raise ValueError(f"Unsupported patch merge backend: {self.merge_backend}")
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x, _ = patch_merge_blc(x, (H, W), use_triton=self.merge_backend == "triton")

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        is_flash (bool): If True, use FlashSwinAttn, else use MHA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        is_flash=True,
        attn_backend: str | None = None,
        window_backend: str = "py",
        merge_backend: str = "py",
        mlp_cls: type[nn.Module] = Mlp,
        mlp_kwargs: dict[str, object] | None = None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        **_kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        if len(_kwargs) != 0:
            logger.debug(f"Got unexpected kwargs: {_kwargs}")

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    is_flash=is_flash,
                    attn_backend=attn_backend,
                    window_backend=window_backend,
                    mlp_cls=mlp_cls,
                    mlp_kwargs=mlp_kwargs,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            if downsample is PatchMerging:
                self.downsample = downsample(
                    input_resolution,
                    dim=dim,
                    norm_layer=norm_layer,
                    merge_backend=merge_backend,
                )
            else:
                self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()  # type: ignore[call-non-callable]
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r"""Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        )
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformer(nn.Module):
    r"""Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        is_flash (bool): If True, use FlashSwinAttn, else use MHA. Default: True
        mlp_cls (type[nn.Module]): FFN module class. Default: Mlp
        mlp_kwargs (dict[str, object] | None): Optional FFN kwargs, filtered by class signature.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        is_flash=True,
        attn_backend: str | None = None,
        window_backend: str = "py",
        merge_backend: str = "py",
        mlp_cls: type[nn.Module] = Mlp,
        mlp_kwargs: dict[str, object] | None = None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.is_flash = is_flash
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(patches_resolution[0] // (2**i_layer), patches_resolution[1] // (2**i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                is_flash=is_flash,
                attn_backend=attn_backend,
                window_backend=window_backend,
                merge_backend=merge_backend,
                mlp_cls=mlp_cls,
                mlp_kwargs=mlp_kwargs,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2**self.num_layers)
        flops += self.num_features * self.num_classes
        return flops

if __name__ == "__main__":
    import sys
    import traceback

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"SwinTransformer smoke test, device={device}")

    try:
        # 使用最小配置，避免依赖 triton/flash 等加速后端
        model = SwinTransformer(
            img_size=32,
            patch_size=4,
            in_chans=3,
            num_classes=10,
            embed_dim=32,
            depths=[1, 1],
            num_heads=[2, 2],
            window_size=4,
            is_flash=True,
            attn_backend="py",
            window_backend="py",
            merge_backend="py",
        )
        model.to(device)
        model.eval()

        x = torch.randn(1, 3, 32, 32, device=device)
        with torch.no_grad():
            out = model(x)
        print("Forward OK, output shape:", getattr(out, "shape", out))
    except Exception:
        print("Error during SwinTransformer smoke test:", file=sys.stderr)
        traceback.print_exc()

        



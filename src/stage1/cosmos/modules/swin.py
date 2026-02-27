from __future__ import annotations

import inspect
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_op.patch_merge.patch_merge_triton import patch_merge_blc
from .swin_op import SwinTransformerBlock, BasicLayer, PatchMerging
from .variants.mlp import SwiGLU


@dataclass
class SwinBlockCfg:
    embed_dim: int
    num_heads: int
    mlp_ratio: float = 4.0
    mlp_cls: type[nn.Module] | None = None
    mlp_kwargs: dict[str, object] | None = None
    window_size: int = 7
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be > 0")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if self.window_size <= 0:
            raise ValueError("window_size must be > 0")


@dataclass
class SwinStageCfg:
    embed_dim: int
    num_heads: int
    depth: int
    mlp_ratio: float = 4.0
    mlp_cls: type[nn.Module] | None = None
    mlp_kwargs: dict[str, object] | None = None
    window_size: int = 7
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float | list[float] = 0.0
    downsample: bool = False
    out_dim: int | None = None
    norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.depth <= 0:
            raise ValueError("depth must be > 0")
        if isinstance(self.drop_path_rate, list) and len(self.drop_path_rate) != self.depth:
            raise ValueError("drop_path_rate list length must equal depth")


@dataclass
class SwinBackboneCfg:
    in_chans: int
    embed_dim: int = 96
    patch_size: int = 4
    depths: list[int] = field(default_factory=lambda: [2, 2, 6, 2])
    num_heads: list[int] = field(default_factory=lambda: [3, 6, 12, 24])
    window_size: int = 7
    mlp_ratio: float = 4.0
    mlp_cls: type[nn.Module] | None = None
    mlp_kwargs: dict[str, object] | None = None
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    out_indices: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    patch_norm: bool = True
    norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.in_chans <= 0:
            raise ValueError("in_chans must be > 0")
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be > 0")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        if len(self.depths) == 0:
            raise ValueError("depths must not be empty")
        if len(self.depths) != len(self.num_heads):
            raise ValueError("depths and num_heads must have the same length")
        if any(d <= 0 for d in self.depths):
            raise ValueError("all depths must be > 0")
        n_stages = len(self.depths)
        for idx in self.out_indices:
            if idx < 0 or idx >= n_stages:
                raise ValueError(f"out_indices contains invalid stage index: {idx}")


def _to_tokens(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    if x.ndim != 4:
        raise ValueError(f"Expected NCHW tensor, got ndim={x.ndim}")
    b, c, h, w = x.shape
    tokens = x.permute(0, 2, 3, 1).reshape(b, h * w, c).contiguous()
    return tokens, (h, w)


def _to_nchw(x: torch.Tensor, hw_shape: tuple[int, int]) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"Expected token tensor [B, L, C], got ndim={x.ndim}")
    b, l, c = x.shape
    h, w = hw_shape
    if l != h * w:
        raise ValueError(f"Token length mismatch: L={l}, H*W={h * w}")
    return x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()


def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(-1, window_size * window_size, c)


def _window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(b, h, w, -1)


def _drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path(x, self.drop_prob, self.training)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(drop_rate)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.trunc_normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def _build_mlp(
    mlp_cls: type[nn.Module] | None,
    dim: int,
    hidden_dim: int,
    drop: float,
    act_layer: type[nn.Module],
    mlp_kwargs: dict[str, object] | None = None,
) -> nn.Module:
    if mlp_cls is None:
        return FeedForward(dim=dim, hidden_dim=hidden_dim, drop_rate=drop)

    kwargs = {} if mlp_kwargs is None else dict(mlp_kwargs)
    if mlp_cls is SwiGLU:
        kwargs.setdefault("is_fused", None)
        kwargs.setdefault("use_conv", False)

    default_kwargs: dict[str, object] = {
        "in_features": dim,
        "hidden_features": hidden_dim,
        "out_features": dim,
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


class WindowMSA(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int,
        qkv_bias: bool = True,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        use_flash: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = embed_dim // num_heads
        self.scale = head_dim**-0.5

        n_rel = (2 * window_size - 1) * (2 * window_size - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(n_rel, num_heads))
        self.register_buffer(
            "relative_position_index", self._build_relative_position_index(window_size), persistent=False
        )

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self._init_weights()

    @staticmethod
    def _build_relative_position_index(window_size: int) -> torch.Tensor:
        coords = torch.stack(torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        nn.init.trunc_normal_(self.qkv.weight, std=0.02)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        rel = self.relative_position_bias_table[self.relative_position_index.view(-1)]  # type: ignore[call-non-callable]
        rel = rel.view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        rel = rel.permute(2, 0, 1).contiguous()
        attn = attn + rel.unsqueeze(0)

        if mask is not None:
            n_windows = mask.shape[0]
            attn = attn.view(b // n_windows, n_windows, self.num_heads, n, n)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ShiftWindowMSA(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int = 0,
        qkv_bias: bool = True,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if shift_size < 0 or shift_size >= window_size:
            raise ValueError("shift_size must satisfy 0 <= shift_size < window_size")

        self.window_size = window_size
        self.shift_size = shift_size
        self.w_msa = WindowMSA(
            embed_dim=embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
        )
        self._attn_mask_cache: dict[tuple[int, int, torch.device], torch.Tensor] = {}

    def _get_attn_mask(self, h_pad: int, w_pad: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if self.shift_size == 0:
            return None

        key = (h_pad, w_pad, device)
        if key in self._attn_mask_cache:
            return self._attn_mask_cache[key].to(dtype=dtype)

        img_mask = torch.zeros((1, h_pad, w_pad, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )

        count = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = count
                count += 1

        mask_windows = _window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        self._attn_mask_cache[key] = attn_mask
        return attn_mask.to(dtype=dtype)

    def forward(self, query: torch.Tensor, hw_shape: tuple[int, int]) -> torch.Tensor:
        b, l, c = query.shape
        h, w = hw_shape
        if l != h * w:
            raise ValueError(f"input feature has wrong size, got L={l}, expected H*W={h * w}")

        x = query.view(b, h, w, c)

        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        h_pad, w_pad = x.shape[1], x.shape[2]

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = self._get_attn_mask(h_pad, w_pad, x.device, x.dtype)
        else:
            shifted_x = x
            attn_mask = None

        x_windows = _window_partition(shifted_x, self.window_size)
        attn_windows = self.w_msa(x_windows, mask=attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = _window_reverse(attn_windows, self.window_size, h_pad, w_pad)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()

        x = x.view(b, h * w, c)
        return x


class SwinBlock(nn.Module):
    def __init__(self, cfg: SwinBlockCfg, shift: bool = False) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.embed_dim, eps=cfg.norm_eps)
        self.attn = ShiftWindowMSA(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            window_size=cfg.window_size,
            shift_size=cfg.window_size // 2 if shift else 0,
            qkv_bias=cfg.qkv_bias,
            attn_drop_rate=cfg.attn_drop_rate,
            proj_drop_rate=cfg.drop_rate,
        )
        self.drop_path = DropPath(cfg.drop_path_rate)
        self.norm2 = nn.LayerNorm(cfg.embed_dim, eps=cfg.norm_eps)
        hidden_dim = int(cfg.embed_dim * cfg.mlp_ratio)
        self.mlp = _build_mlp(
            mlp_cls=cfg.mlp_cls,
            dim=cfg.embed_dim,
            hidden_dim=hidden_dim,
            drop=cfg.drop_rate,
            act_layer=nn.GELU,
            mlp_kwargs=cfg.mlp_kwargs,
        )

    def forward(self, x: torch.Tensor, hw_shape: tuple[int, int]) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), hw_shape))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, norm_eps: float = 1e-6, use_triton_merge: bool = True) -> None:
        super().__init__()
        self.use_triton_merge = use_triton_merge
        self.norm = nn.LayerNorm(4 * in_dim, eps=norm_eps)
        self.reduction = nn.Linear(4 * in_dim, out_dim, bias=False)
        nn.init.trunc_normal_(self.reduction.weight, std=0.02)

    def forward(self, x: torch.Tensor, hw_shape: tuple[int, int]) -> tuple[torch.Tensor, tuple[int, int]]:
        b, l, c = x.shape
        h, w = hw_shape
        if l != h * w:
            raise ValueError(f"input feature has wrong size, got L={l}, expected H*W={h * w}")
        if c * 4 != self.norm.normalized_shape[0]:
            raise ValueError(
                f"Channel mismatch for PatchMerging: got C={c}, expected {self.norm.normalized_shape[0] // 4}"
            )

        x, (h2, w2) = patch_merge_blc(x, (h, w), use_triton=self.use_triton_merge)

        x = self.norm(x)
        x = self.reduction(x)
        return x, (h2, w2)


class SwinStage(nn.Module):
    def __init__(self, cfg: SwinStageCfg) -> None:
        super().__init__()
        if isinstance(cfg.drop_path_rate, list):
            drop_path_rates = cfg.drop_path_rate
        else:
            drop_path_rates = [cfg.drop_path_rate for _ in range(cfg.depth)]

        self.blocks = nn.ModuleList()
        for i in range(cfg.depth):
            block_cfg = SwinBlockCfg(
                embed_dim=cfg.embed_dim,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                mlp_cls=cfg.mlp_cls,
                mlp_kwargs=cfg.mlp_kwargs,
                window_size=cfg.window_size,
                qkv_bias=cfg.qkv_bias,
                drop_rate=cfg.drop_rate,
                attn_drop_rate=cfg.attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                norm_eps=cfg.norm_eps,
            )
            self.blocks.append(SwinBlock(block_cfg, shift=(i % 2 == 1)))

        self.downsample: PatchMerging | None = None
        if cfg.downsample:
            out_dim = cfg.out_dim if cfg.out_dim is not None else cfg.embed_dim * 2
            self.downsample = PatchMerging(cfg.embed_dim, out_dim=out_dim, norm_eps=cfg.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        hw_shape: tuple[int, int],
    ) -> tuple[torch.Tensor, tuple[int, int], torch.Tensor, tuple[int, int]]:
        for block in self.blocks:
            x = block(x, hw_shape)

        stage_out = x
        stage_hw = hw_shape

        if self.downsample is not None:
            x, hw_shape = self.downsample(x, hw_shape)

        return x, hw_shape, stage_out, stage_hw


def _build_stage_cfgs(cfg: SwinBackboneCfg) -> tuple[list[SwinStageCfg], list[int]]:
    n_stages = len(cfg.depths)
    stage_dims = [cfg.embed_dim * (2**i) for i in range(n_stages)]
    total_depth = sum(cfg.depths)
    drop_path_rates = torch.linspace(0, cfg.drop_path_rate, total_depth).tolist() if total_depth > 0 else []

    stage_cfgs: list[SwinStageCfg] = []
    start = 0
    for i, depth in enumerate(cfg.depths):
        end = start + depth
        stage_drop = drop_path_rates[start:end]
        start = end

        has_downsample = i < n_stages - 1
        out_dim = stage_dims[i + 1] if has_downsample else None
        stage_cfgs.append(
            SwinStageCfg(
                embed_dim=stage_dims[i],
                num_heads=cfg.num_heads[i],
                depth=depth,
                mlp_ratio=cfg.mlp_ratio,
                mlp_cls=cfg.mlp_cls,
                mlp_kwargs=cfg.mlp_kwargs,
                window_size=cfg.window_size,
                qkv_bias=cfg.qkv_bias,
                drop_rate=cfg.drop_rate,
                attn_drop_rate=cfg.attn_drop_rate,
                drop_path_rate=stage_drop,
                downsample=has_downsample,
                out_dim=out_dim,
                norm_eps=cfg.norm_eps,
            )
        )

    return stage_cfgs, stage_dims


class SwinBackbone(nn.Module):
    def __init__(self, cfg: SwinBackboneCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.out_indices = sorted(set(cfg.out_indices))

        self.patch_embed = nn.Conv2d(
            cfg.in_chans,
            cfg.embed_dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
            padding=0,
        )
        nn.init.kaiming_normal_(self.patch_embed.weight, mode="fan_out")
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

        self.patch_norm = nn.LayerNorm(cfg.embed_dim, eps=cfg.norm_eps) if cfg.patch_norm else None

        stage_cfgs, stage_dims = _build_stage_cfgs(cfg)
        self.stages = nn.ModuleList([SwinStage(stage_cfg) for stage_cfg in stage_cfgs])

        self.out_norms = nn.ModuleDict(
            {str(i): nn.LayerNorm(stage_dims[i], eps=cfg.norm_eps) for i in self.out_indices}
        )

    def _forward_tokens(self, tokens: torch.Tensor, hw_shape: tuple[int, int]) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        x = tokens
        hw = hw_shape

        for idx, stage in enumerate(self.stages):
            x, hw, stage_out, stage_hw = stage(x, hw)
            if idx in self.out_indices:
                out = self.out_norms[str(idx)](stage_out)
                outputs.append(_to_nchw(out, stage_hw))

        return outputs

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor | list[torch.Tensor]:
        x = self.patch_embed(x)
        tokens, hw_shape = _to_tokens(x)
        if self.patch_norm is not None:
            tokens = self.patch_norm(tokens)

        outputs = self._forward_tokens(tokens, hw_shape)

        if return_features:
            return outputs
        if len(outputs) == 0:
            raise ValueError("No output features selected, please set out_indices")
        return outputs[-1]


class SwinBlock2D(nn.Module):
    def __init__(self, cfg: SwinBlockCfg, shift: bool = False) -> None:
        super().__init__()
        self.block = SwinBlock(cfg, shift=shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens, hw_shape = _to_tokens(x)
        tokens = self.block(tokens, hw_shape)
        return _to_nchw(tokens, hw_shape)


class SwinStage2D(nn.Module):
    def __init__(self, cfg: SwinStageCfg) -> None:
        super().__init__()
        self.stage = SwinStage(cfg)

    def forward(
        self, x: torch.Tensor, return_stage_output: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        tokens, hw_shape = _to_tokens(x)
        tokens_next, hw_next, stage_out, stage_hw = self.stage(tokens, hw_shape)
        x_next = _to_nchw(tokens_next, hw_next)
        if not return_stage_output:
            return x_next
        return x_next, _to_nchw(stage_out, stage_hw)


class SwinBackbone2D(nn.Module):
    def __init__(self, cfg: SwinBackboneCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.out_indices = sorted(set(cfg.out_indices))

        self.input_proj: nn.Module
        if cfg.in_chans != cfg.embed_dim:
            self.input_proj = nn.Conv2d(cfg.in_chans, cfg.embed_dim, kernel_size=1)
            nn.init.kaiming_normal_(self.input_proj.weight, mode="fan_out")
            assert isinstance(self.input_proj, nn.Conv2d)
            if self.input_proj.bias is not None:
                nn.init.zeros_(self.input_proj.bias)
        else:
            self.input_proj = nn.Identity()

        stage_cfgs, stage_dims = _build_stage_cfgs(cfg)
        self.stages = nn.ModuleList([SwinStage(stage_cfg) for stage_cfg in stage_cfgs])
        self.out_norms = nn.ModuleDict(
            {str(i): nn.LayerNorm(stage_dims[i], eps=cfg.norm_eps) for i in self.out_indices}
        )

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor | list[torch.Tensor]:
        x = self.input_proj(x)
        tokens, hw_shape = _to_tokens(x)

        outputs: list[torch.Tensor] = []
        hw = hw_shape
        for idx, stage in enumerate(self.stages):
            tokens, hw, stage_out, stage_hw = stage(tokens, hw)
            if idx in self.out_indices:
                out = self.out_norms[str(idx)](stage_out)
                outputs.append(_to_nchw(out, stage_hw))

        if return_features:
            return outputs
        if len(outputs) == 0:
            raise ValueError("No output features selected, please set out_indices")
        return outputs[-1]


__all__ = [
    "PatchMerging",
    "ShiftWindowMSA",
    "SwinBackbone",
    "SwinBackbone2D",
    "SwinBackboneCfg",
    "SwinBlock",
    "SwinBlock2D",
    "SwinBlockCfg",
    "SwinStage",
    "SwinStage2D",
    "SwinStageCfg",
    "WindowMSA",
]

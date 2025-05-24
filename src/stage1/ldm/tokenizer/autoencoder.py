import math
from functools import partial
from inspect import isclass
from typing import Any, Callable, Literal, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange
from transformers.activations import ACT2FN

from src.utilities.logging import log_print

compile_forward_fn = True
if compile_forward_fn:
    _compile_decorator = torch.compile
    log_print("will compile the forward function", "debug")
else:

    def _null_decorator(**any_kwargs) -> Callable[..., Any]:
        def _inner_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return _inner_decorator

    def _null_decorator_no_any_kwgs(func: Callable[..., Any]) -> Callable[..., Any]:
        return func

    _compile_decorator = _null_decorator_no_any_kwgs
    log_print("not compile the forward function", "debug")


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(
            out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w
        )
        return self.to_out(out)


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=0,
        act_checkpoint=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

        self.act_checkpoint = act_checkpoint

    def forward_fn(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h

    def forward(self, x, temb=None):
        if self.act_checkpoint:
            h = torch.utils.checkpoint.checkpoint(self.forward_fn, x, temb)
        else:
            h = self.forward_fn(x, temb)
        return h


class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""

    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class AttnBlock(nn.Module):
    def __init__(self, in_channels, act_checkpoint=False):
        super().__init__()
        self.in_channels = in_channels
        self.act_checkpoint = act_checkpoint

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    @_compile_decorator
    def forward_fn(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_

    def forward(self, x):
        if self.act_checkpoint:
            h = torch.utils.checkpoint.checkpoint(self.forward_fn, x)
        else:
            h = self.forward_fn(x)
        return h


# * --- Dico Block --- #


class DiCoCompactChannelAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        conv_cls: nn.Module = nn.Conv2d,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.body = conv_cls(in_channels, in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.global_avg(x)
        h = self.body(h)
        h = self.sigmoid(h) * x
        return h


class GLUFeedForward(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        hidden_act="gelu_fast",  # gelu_fast, gelu
        conv_cls=nn.Conv2d,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = conv_cls(self.hidden_size, self.intermediate_size, 1, 1, 0)
        self.up_proj = conv_cls(self.hidden_size, self.intermediate_size, 1, 1, 0)
        self.down_proj = conv_cls(self.intermediate_size, self.hidden_size, 1, 1, 0)
        self.act_fn = ACT2FN[hidden_act]

    def weight_init(self):
        torch.nn.init.trunc_normal_(self.gate_proj.weight, std=0.02)
        torch.nn.init.trunc_normal_(self.up_proj.weight, std=0.02)
        torch.nn.init.trunc_normal_(self.down_proj.weight, std=0.02)

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = (bias, bias)
        drop_probs = (drop, drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1)

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer() if isclass(act_layer) else act_layer
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class DiCoBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int | None = None,
        padding_mode: str = "zeros",
        act_checkpoint: bool = False,
        use_ffn: bool = True,
        conv_type: str = "conv",
    ):
        super().__init__()
        self.act_checkpoint = act_checkpoint
        self.use_ffn = use_ffn
        if out_channels is None:
            out_channels = in_channels

        log_print(
            f"[Dico block]: in: {in_channels} "
            f"out: {out_channels} "
            f"hidden: {hidden_channels} "
            f"conv type: {conv_type} ",
            "debug",
        )

        self.norm = Normalize(in_channels)
        conv_cls = nn.Conv2d

        self.body = nn.Sequential(
            # point conv
            conv_cls(in_channels, out_channels, kernel_size=1, stride=1),
            # depth conv
            conv_cls(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode=padding_mode,
                groups=out_channels,
            ),
            ACT2FN["gelu_fast"],
        )
        self.body_out = conv_cls(out_channels, out_channels, kernel_size=1, stride=1)

        # cca
        self.cca = DiCoCompactChannelAttention(out_channels, conv_cls=conv_cls)

        # ffn
        if self.use_ffn:
            self.ffn = Mlp(
                in_features=out_channels,
                hidden_features=hidden_channels,
                out_features=out_channels,
                act_layer=ACT2FN["gelu_fast"],
                bias=True,
                drop=0.0,
            )
            self.norm_ffn = Normalize(out_channels)

        self.nin_shortcut = (
            conv_cls(in_channels, out_channels, kernel_size=1, stride=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    @_compile_decorator
    def forward_fn(self, x: torch.Tensor) -> torch.Tensor:
        h = x

        # token mixer
        h = self.body(self.norm(h))
        h = self.cca(h)
        h = self.body_out(h)
        h = h + self.nin_shortcut(x)

        # ffn
        if self.use_ffn:
            h = self.ffn(self.norm_ffn(h)) + h

        return h

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        if self.act_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self.forward_fn, x, use_reentrant=True
            )

        return self.forward_fn(x)


def make_attn(in_channels, attn_type="vanilla", act_checkpoint=False):
    assert attn_type in ["vanilla", "linear", "none"], f"attn_type {attn_type} unknown"
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels, act_checkpoint=act_checkpoint)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)


def make_block_fn(
    block_name: Literal["resblock", "dico_block"] = "resblock",
    act_checkpoint=False,
    hidden_factor=2,
    padding_mode: str = "zeros",
    norm_type: str = "gn",
    **kwargs,
):
    if block_name == "dico_block":

        def block_fn(block_in, block_out, dropout):
            return DiCoBlock(
                in_channels=block_in,
                hidden_channels=block_out * hidden_factor,
                out_channels=block_out,
                padding_mode=padding_mode,
                act_checkpoint=act_checkpoint,
                use_ffn=True,
            )

    elif block_name == "res_block":

        def block_fn(block_in, block_out, dropout):
            return ResnetBlock(
                in_channels=block_in,
                out_channels=block_out,
                dropout=dropout,
                act_checkpoint=act_checkpoint,
                temb_channels=0,
                **kwargs,
            )

    else:
        raise ValueError(
            f"block_name {block_name} is not supported. Supported: 'res_block', 'dico_block'"
        )

    return block_fn


# * --- diffbands input conv --- #


class DiffBandsInputConvIn(nn.Module):
    def __init__(
        self,
        band_lst: list[int],
        hidden_dim: int,
        basic_module: Literal["conv", "resnet", "inv_bottleneck", "moe"] = "conv",
        padding_mode: str = "zeros",
    ):
        super().__init__()

        self.band_lst = band_lst
        self.hidden_dim = hidden_dim

        self.in_modules = nn.ModuleDict()
        for c in band_lst:
            if basic_module == "conv":
                module = nn.Conv2d(
                    in_channels=c,
                    out_channels=hidden_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="zeros",
                )
            elif basic_module == "mobile":
                module = nn.Sequential(
                    nn.Conv2d(c, hidden_dim, 1, 1, 0),
                    nn.Conv2d(
                        hidden_dim,
                        hidden_dim,
                        3,
                        1,
                        1,
                        padding_mode="zeros",
                        groups=hidden_dim,
                    ),
                )
            elif basic_module == "resnet":
                module = ResnetBlock(
                    in_channels=c,
                    out_channels=hidden_dim,
                    num_groups=1,
                    dropout=0.0,
                    act_checkpoint=False,
                    padding_mode=padding_mode,
                    norm_type="gn",
                )
            else:
                raise ValueError(
                    f"[DiffBandsInputConvIn] Unknown basic_module={basic_module}"
                )

            self.in_modules["conv_in_{}".format(c)] = module
            log_print(
                f"[DiffBandsInputConvIn] set conv to hidden module and buffer for channel {c}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c_ = x.shape[1]
        module = getattr(self.in_modules, "conv_in_{}".format(c_))
        if module is None:
            raise ValueError(
                f"[DiffBandsInputConvIn] no module for channel {c_}, please check the channel list"
            )
        h = module(x)

        if self.training:
            for c in self.band_lst:
                if c != c_:
                    m = self.in_modules["conv_in_{}".format(c)]
                    dummy_loss = sum(p.sum() * 0.0 for p in m.parameters())
                    h = h + dummy_loss

        return h


class DiffBandsInputConvOut(nn.Module):
    def __init__(
        self,
        band_lst: list[int],
        hidden_dim: int,
        basic_module: str = "conv",
    ):
        super().__init__()

        self.band_lst = band_lst
        self.hidden_dim = hidden_dim
        self.basic_module = basic_module

        self.in_modules = nn.ModuleDict()
        for c in band_lst:
            if basic_module == "conv":
                module = nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=c,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="zeros",
                )
            elif basic_module == "mobile":
                module = nn.Sequential(
                    nn.Conv2d(
                        hidden_dim,
                        hidden_dim,
                        3,
                        1,
                        1,
                        padding_mode="zeros",
                        groups=hidden_dim,
                    ),
                    nn.Conv2d(hidden_dim, c, 1, 1, 0),
                )
            elif basic_module == "resnet":
                module = ResnetBlock(
                    in_channels=hidden_dim,
                    out_channels=c,
                    dropout=0.0,
                    act_checkpoint=False,
                )
            else:
                raise ValueError(
                    f"[DiffBandsInputConvIn] Unknown basic_module={basic_module}"
                )

            self.in_modules["conv_out_{}".format(c)] = module

            print(f"[DiffBandsInputConvOut] set conv to hidden module for channel {c}")

        self.out_channel = None

    def forward(self, x: torch.Tensor, out_channel: int) -> torch.Tensor:
        self.out_channel = out_channel

        module = getattr(self.in_modules, f"conv_out_{out_channel}", None)
        if module is None:
            raise ValueError(
                f"[DiffBandsInputConvOut] No module for out_channel={out_channel}. Available: {list(self.in_modules.keys())}",
            )
        h = module(x)

        if self.training:
            for c in self.band_lst:
                if c != out_channel:
                    m = self.in_modules["conv_out_{}".format(c)]
                    dummy_loss = sum(p.sum() * 0.0 for p in m.parameters())
                    h = h + dummy_loss

        return h

    @property
    def weight(self):
        # used to get the weight of the conv_out module for GAN loss
        assert self.out_channel is not None, (
            "out_channel is not set, please call forward first"
        )
        module = getattr(self.in_modules, f"conv_out_{self.out_channel}", None)
        if module is None:
            raise ValueError(
                f"[DiffBandsInputConvOut] No module for out_channel={self.out_channel}. Available: {list(self.in_modules.keys())}",
            )

        match self.basic_module:
            case "conv":
                return module.weight
            case "resnet":
                return module.conv2.weight
            case "mobile":
                return module[-1].weight
            case _:
                raise ValueError(
                    f"[DiffBandsInputConvOut] Unknown basic_module={self.basic_module}. Available: conv, resnet, moe, inv_bottleneck"
                )


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        double_z=True,
        use_linear_attn=False,
        attn_type="vanilla",
        act_checkpoint=False,
        block_type="res_block",
        hidden_factor=2,
        **ignore_kwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        log_print(f"[Encoder] - {block_type=}, {attn_type=}", "debug")
        block_fn = make_block_fn(block_type, act_checkpoint, hidden_factor)

        # downsampling
        if isinstance(in_channels, int):
            self.conv_in = torch.nn.Conv2d(
                in_channels, self.ch, kernel_size=3, stride=1, padding=1
            )
        else:
            assert isinstance(in_channels, Sequence), (
                "in_channels must be int or Sequence"
            )
            self.conv_in = DiffBandsInputConvIn(
                band_lst=in_channels,
                hidden_dim=ch,
                basic_module="conv",
                padding_mode="zeros",
            )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(block_fn(block_in, block_out, dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(
                        make_attn(
                            block_in, attn_type=attn_type, act_checkpoint=act_checkpoint
                        )
                    )
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = block_fn(block_in, block_in, dropout)
        self.mid.attn_1 = make_attn(
            block_in, attn_type=attn_type, act_checkpoint=act_checkpoint
        )
        self.mid.block_2 = block_fn(block_in, block_in, dropout)
        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                # print(f"[encoder] - h norm {h.norm()}, h max {h.abs().max()}")
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        tanh_out=False,
        use_linear_attn=False,
        attn_type="vanilla",
        block_type="res_block",
        act_checkpoint=False,
        hidden_factor=2,
        **ignorekwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print(
            "Working with z of shape {} = {} dimensions.".format(
                self.z_shape, np.prod(self.z_shape)
            )
        )

        # z to block_in
        self.conv_in = torch.nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        log_print(f"[Decoder] - {block_type=}, {attn_type=}", "debug")
        block_fn = make_block_fn(
            block_type, act_checkpoint=act_checkpoint, hidden_factor=hidden_factor
        )
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = block_fn(block_in, block_in, dropout)
        self.mid.attn_1 = make_attn(
            block_in, attn_type=attn_type, act_checkpoint=act_checkpoint
        )
        self.mid.block_2 = block_fn(block_in, block_in, dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(block_fn(block_in, block_out, dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(
                        make_attn(
                            block_in, attn_type=attn_type, act_checkpoint=act_checkpoint
                        )
                    )
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        if isinstance(out_ch, int):
            self.conv_out = torch.nn.Conv2d(
                block_in, out_ch, kernel_size=3, stride=1, padding=1
            )
            self.use_diffbands = False
        else:
            assert isinstance(out_ch, Sequence), "out_ch must be Sequence"
            self.conv_out = DiffBandsInputConvOut(
                band_lst=out_ch,
                hidden_dim=block_in,
                basic_module="conv",
            )
            self.use_diffbands = True

    def forward(self, z, c):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                # print(f"[decoder] - h norm {h.norm()}, h max {h.abs().max()}")
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        if self.use_diffbands:
            h = self.conv_out(h, c)
        else:
            h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device
            )

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.parameters.device
        )
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean


class AutoencoderKL(nn.Module):
    def __init__(self, embed_dim: int, ddconfig: dict):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        self.use_variation = ddconfig["double_z"]
        ch_mul = 2 if self.use_variation else 1

        self.quant_conv = torch.nn.Conv2d(
            ch_mul * ddconfig["z_channels"], ch_mul * embed_dim, 1
        )
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

    def encode(self, x):
        z = self.encoder(x)
        z = self.quant_conv(z)

        if self.use_variation:
            post = DiagonalGaussianDistribution(z)
            z = post.sample()
            return (
                z,
                post.kl(),
                {
                    "posterior": post,
                    "mean": post.mean,
                    "logvar": post.logvar,
                },
            )

        return z

    def decode(self, z, c):
        if isinstance(z, Sequence) or self.use_variation:
            parts = z[1:]
            z = z[0]

            z = self.post_quant_conv(z)
            recon = self.decoder(z, c)

            return recon, parts
        else:
            z = self.post_quant_conv(z)
            recon = self.decoder(z, c)
            return recon

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z, x.shape[1])
        return recon

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def register_feature_hook(self): ...

    def get_repa_feature(self): ...

    def peft_first_last_convs_module_names(self):
        return ["encoder.conv_in", "decoder.conv_out"]

    def register_layer_output_hooks(self): ...


if __name__ == "__main__":
    dd_config = {
        "double_z": False,
        "z_channels": 32,
        "resolution": 256,
        "in_channels": [3, 4, 8],
        "out_ch": [3, 4, 8],
        "ch": 128,
        "ch_mult": [1, 2, 2, 4],
        "num_res_blocks": 2,
        "attn_resolutions": [16],
        "dropout": 0.0,
        "block_type": "dico_block",
        "attn_type": "none",
    }
    model = AutoencoderKL(embed_dim=32, ddconfig=dd_config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Simple training loop
    num_steps = 100
    batch_size = 2

    for step in range(num_steps):
        # Create random input
        x = torch.randn(batch_size, 3, 256, 256).cuda()

        # Forward pass
        optimizer.zero_grad()
        z = model.encode(x)
        x_recon = model.decode(z, x.shape[1])

        if isinstance(x_recon, tuple):
            x_recon = x_recon[0]

        # Compute loss (simple MSE)
        loss = torch.nn.functional.mse_loss(x_recon, x)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        # Print stats
        print(f"Step {step}, Loss: {loss.item():.6f}")
        print(f"Z shape: {z[0].shape if isinstance(z, tuple) else z.shape}")
        print(f"Reconstruction shape: {x_recon.shape}")
        # print(
        #     f"Gradient norm: {sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None):.6f}"
        # )
        print("-" * 40)

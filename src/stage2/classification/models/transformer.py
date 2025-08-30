import functools
from typing import Callable

import natten
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange, unpack
from jaxtyping import Array, Float
from timm.layers import create_act, create_norm, get_act_layer, get_norm_layer

# Attention, MLP
from timm.layers.attention import Attention as Attention_
from timm.layers.create_conv2d import create_conv2d
from timm.layers.drop import DropPath
from timm.layers.helpers import to_2tuple
from timm.layers.mlp import SwiGLU as SwiGLUMLP_
from timm.layers.patch_embed import PatchEmbed
from torch import Tensor
from transformers.activations import ACT2CLS, ACT2FN, ClassInstantier

from src.stage2.pansharpening.models.rope import (
    RotaryPositionEmbeddingPytorchV2,
    get_2d_sincos_pos_embed,
)


def pack_one(x, pattern):
    x, ps = pack([x], pattern)
    return x, ps


def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


# * --- Activations --- * #


class PolyNorm(torch.nn.Module):
    """
    A trainable activation function introduced in https://arxiv.org/html/2411.03884v1.
    The code is copied from https://github.com/BryceZhuo/PolyCom?tab=readme-ov-file/README.md
    taken from https://huggingface.co/Motif-Technologies/Motif-2.6B/blob/main/modeling_motif.py#L26
    """

    def __init__(self, eps=1e-6):
        super(PolyNorm, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(3) / 3)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.eps = eps

    def _norm(self, x):
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (
            self.weight[0] * self._norm(x**3)
            + self.weight[1] * self._norm(x**2)
            + self.weight[2] * self._norm(x)
            + self.bias
        )


class SwiGLUAct(nn.Module):
    def __init__(self, alpha: float = 1.702, limit: float = 7.0):
        super().__init__()
        self.alpha = alpha
        self.limit = limit

    def forward(self, x_glu, x_linear):
        alpha, limit = self.alpha, self.limit
        # x_glu, x_linear = x[..., ::2], x[..., 1::2]
        # Clamp the input values
        x_glu = x_glu.clamp(min=None, max=limit)
        x_linear = x_linear.clamp(min=-limit, max=limit)
        out_glu = x_glu * torch.sigmoid(alpha * x_glu)
        # Note we add an extra bias of 1 to the linear layer
        return out_glu * (x_linear + 1)


# Norm registration

# ACT2CLS["polynorm"] = PolyNorm
# ACT2FN = ClassInstantier(ACT2CLS)

create_act._ACT_FN_DEFAULT["poly_norm"] = PolyNorm
create_act._ACT_FN_ME["poly_norm"] = PolyNorm


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float | Tensor = 1e-5,
        inplace: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(torch.empty(dim, device=device))
        self.init_values = init_values

    def reset_parameters(self):
        nn.init.constant_(self.gamma, self.init_values)

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class NatAttention(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size=8,
        stride=2,
        dilation=2,
        num_heads=8,
        qkv_bias=True,
        qk_norm: nn.Module | None = None,
        norm_layer=None,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        scale_norm: bool = False,
        torch_compile=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.torch_compile = torch_compile
        self.is_causal = False

        self.qkv = nn.Conv2d(dim, dim * 3, 3, 1, 1, groups=dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(dim) if scale_norm else nn.Identity()
        self.proj = nn.Conv2d(dim, dim, 1, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        if qk_norm:
            self.q_norm = norm_layer(self.head_dim)
            self.k_norm = norm_layer(self.head_dim)
        else:
            self.q_norm = self.k_norm = nn.Identity()

    def _apply_rope(self, q: Float[Tensor, "b h w nh hd"], k, rope):
        if rope:
            h, w = q.shape[1:3]
            q = rearrange(q, "b h w nh hd -> b (h w) nh hd")
            k = rearrange(k, "b h w nh hd -> b (h w) nh hd")

            q, k = rope(q, k)

            q = rearrange(q, "b (h w) nh hd -> b h w nh hd", h=h, w=w)
            k = rearrange(k, "b (h w) nh hd -> b h w nh hd", h=h, w=w)

        return q, k

    def forward(self, x: Float[Tensor, "b c h w"], mask=None, rope=None):
        B, C, H, W = x.shape

        qkv = self.qkv(x)
        qkv = rearrange(qkv, "b (qkv c) h w -> b qkv c h w", qkv=3)
        q, k, v = qkv.unbind(1)
        dtype = q.dtype

        q = self.q_norm(q)  # (bs, c, h, w)
        k = self.k_norm(k)

        q = rearrange(q, "b (nh hd) h w -> b h w nh hd", nh=self.num_heads)
        k = rearrange(k, "b (nh hd) h w -> b h w nh hd", nh=self.num_heads)
        v = rearrange(v, "b (nh hd) h w -> b h w nh hd", nh=self.num_heads)

        q, k = self._apply_rope(q, k, rope)

        x = natten.na2d(  # (bs, h, w, nh, hd)
            q,
            k,
            v,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            is_causal=False,
            torch_compile=self.torch_compile,
        )

        x = rearrange(x, "b h w nh hd -> b (nh hd) h w")
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)

        if torch.get_autocast_dtype("cuda") == torch.float16:
            x = x.clip(-65504, 65504)

        return x


class Attention(Attention_):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        qk_norm: nn.Module | None = None,
        **block_kwargs,
    ):
        super().__init__(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            **block_kwargs,
        )

        if qk_norm:
            self.q_norm = qk_norm(dim)
            self.k_norm = qk_norm(dim)
        else:
            self.q_norm = self.k_norm = nn.Identity()

    def forward(self, x, mask=None, rope: Callable | None = None):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, C)
        q, k, v = qkv.unbind(2)
        dtype = q.dtype

        breakpoint()
        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.reshape(B, N, self.num_heads, C // self.num_heads).to(dtype)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).to(dtype)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).to(dtype)

        # RoPE
        if rope is not None:
            q, k = rope(q, k)

        use_fp32_attention = getattr(
            self, "fp32_attention", False
        )  # necessary for NAN loss
        if use_fp32_attention:
            q, k, v = q.float(), k.float(), v.float()

        # sdpa
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None and mask.ndim == 2:
            mask = (1 - mask.to(x.dtype)) * -10000.0
            mask = mask[:, None, None].repeat(1, self.num_heads, 1, 1)

        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
        )
        x = x.transpose(1, 2)

        x = x.view(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if torch.get_autocast_dtype("cuda") == torch.float16:
            x = x.clip(-65504, 65504)

        return x


class SwiGLU(nn.Module):
    """SwiGLU
    NOTE: GluMLP above can implement SwiGLU, but this impl has split fc1 and
    better matches some other common impl which makes mapping checkpoints simpler.
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.SiLU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        self.use_conv = use_conv

        linear_layer = (
            functools.partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        )
        self.fc1_g = linear_layer(in_features, hidden_features, bias=bias[0])
        self.fc1_x = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        if self.fc1_g.bias is not None:
            nn.init.ones_(self.fc1_g.bias)
        nn.init.normal_(self.fc1_g.weight, std=1e-6)

    def forward(self, x):
        x_gate = self.fc1_g(x)
        x = self.fc1_x(x)
        x = self.act(x_gate) * x
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class ClipSwiGLUMlp(SwiGLU):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=SwiGLUAct,
        norm_layer=None,
        bias=True,
        drop=0.0,
        mlp_bias=True,
        use_conv=False,
    ):
        super().__init__(
            in_features,
            hidden_features,
            out_features,
            act_layer,
            norm_layer,
            bias,
            drop,
            use_conv,
        )
        self.mlp_bias = (
            nn.Parameter(torch.zeros(self.fc2.weight.shape[0])) if mlp_bias else None
        )

    def forward(self, x):
        x_gate = self.fc1_g(x)
        x = self.fc1_x(x)
        if isinstance(self.act, SwiGLUAct):
            x = self.act(x_gate, x)
        else:
            x = self.act(x_gate) * x
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        if self.mlp_bias is not None:
            if not self.use_conv:
                bias = self.mlp_bias
            else:
                bias = self.mlp_bias[..., None, None]
            x += bias
        x = self.drop2(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        num_heads=8,
        attn_type="1d",
        qkv_bias=True,
        kernel_size=8,
        stride=2,
        dilation=2,
        qk_norm: nn.Module | None = None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        mlp_norm_layer=nn.LayerNorm,
        act_layer=nn.SiLU,
        layer_scale_value=1e-3,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if attn_type == "1d":
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                norm_layer=norm_layer,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        else:
            self.attn = NatAttention(
                dim,
                kernel_size,
                stride,
                dilation,
                num_heads,
                qkv_bias,
                qk_norm,
                norm_layer,
                proj_bias=True,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        self.norm2 = norm_layer(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = ClipSwiGLUMlp(
            in_features=dim,
            hidden_features=hidden_dim,
            act_layer=act_layer,
            drop=drop,
            norm_layer=mlp_norm_layer,
            use_conv=attn_type != "1d",
        )
        self.ls1 = LayerScale(dim, layer_scale_value)
        self.ls2 = LayerScale(dim, layer_scale_value)

        self.drop_path = DropPath(drop_path)

    def forward(self, x, mask=None, pe=None):
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x), mask=mask, rope=pe)))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        dim,
        depth,
        num_heads,
        with_raw_img=False,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        input_size: int = 32,
        patch_size=2,
        out_channels=16,
        raw_img_size=None,
        raw_img_chans=None,
        pos_embed_type="sincos",
        norm_layer=functools.partial(nn.LayerNorm, eps=1e-6),
        mlp_norm_layer=functools.partial(nn.LayerNorm, eps=1e-6),
        act_layer=SwiGLU,
        feature_layer_ids: list[int] | None = None,
    ):
        super().__init__()

        # patch embedding
        self.patch_size = patch_size
        self.input_size = input_size
        self.num_patches = (input_size // patch_size) ** 2
        self._n_modalities = 1
        self.with_raw_img = with_raw_img
        self.raw_img_size = raw_img_size
        self.raw_img_chans = raw_img_chans
        self.raw_patch_size = 16
        self.patch_embed = PatchEmbed(
            img_size=input_size,
            patch_size=patch_size,
            in_chans=in_dim,
            embed_dim=dim,
            bias=True,
            strict_img_size=False,
        )
        if self.with_raw_img:
            assert raw_img_size is not None
            assert raw_img_chans is not None
            assert self.pos_embed_type == "sincos"
            self.raw_img_patcher = PatchEmbed(
                img_size=raw_img_size,
                patch_size=self.raw_patch_size,
                in_chans=raw_img_chans,
                embed_dim=dim,
                bias=True,
                strict_img_size=False,
            )
        # self.fuse_stem = nn.Linear(dim // self._n_modalities * self._n_modalities, dim)
        self.base_size = input_size // self.patch_size
        self.pe_interpolation = 1.0
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.feature_layer_ids = feature_layer_ids
        if feature_layer_ids:
            assert max(feature_layer_ids) < depth, (
                "max feature_layer_id must be less than depth"
            )

        # layers
        layers = []
        drop_path = [
            x.item() for x in torch.linspace(0, drop_path, depth)
        ]  # stochastic depth decay rule
        norm_layer = get_norm_layer(norm_layer)
        act_layer = get_act_layer(act_layer)
        for i in range(depth):
            layers.append(
                AttentionBlock(
                    dim=dim,
                    mlp_ratio=mlp_ratio,
                    num_heads=num_heads,
                    qkv_bias=True,
                    qk_norm=norm_layer,
                    drop=drop,
                    attn_drop=drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=mlp_norm_layer,
                    act_layer=act_layer,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.head = nn.Sequential(
            norm_layer(dim),
            nn.Linear(dim, out_channels * patch_size**2, bias=True),
        )

        # positional embedding
        self.pos_embed_type = pos_embed_type
        self.setup_pe(dim)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def setup_pe(self, dim, rope_options: dict | None = None):
        seq_len = self.num_patches
        if self.with_raw_img:
            raw_img_patches = self.raw_img_patcher.num_patches

        if self.pos_embed_type == "sincos":
            self.pos_embed_latent: nn.Buffer
            self.register_buffer(
                "pos_embed_latent", torch.zeros(1, self.num_patches, dim)
            )
            # sincos
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed_latent.shape[-1],
                int(seq_len**0.5),
                pe_interpolation=self.pe_interpolation,
                base_size=self.base_size,
            )
            self.pos_embed_latent.data.copy_(
                torch.from_numpy(pos_embed).float().unsqueeze(0)
            )
            if self.with_raw_img:
                pos_embed_raw = get_2d_sincos_pos_embed(
                    dim,
                    int(raw_img_patches**0.5),
                    pe_interpolation=self.pe_interpolation,
                    base_size=self.raw_img_size // self.patch_size,
                )
                self.pos_embed_raw: nn.Buffer
                self.register_buffer(
                    "pos_embed_raw", torch.as_tensor(pos_embed_raw).float().unsqueeze(0)
                )

        elif self.pos_embed_type == "rope":
            if rope_options is None:
                self.rope_options = {
                    "dim": dim // self.num_heads,
                    "rope_dim": "2D",
                    "beta_fast": 4,
                    "beta_slow": 1,
                    "rope_theta": 10000,
                    "apply_yarn": True,
                    "scale": 1.0,
                    "original_latent_shape": (self.base_size, self.base_size),
                }
            self.rope = RotaryPositionEmbeddingPytorchV2(
                seq_len=seq_len,
                latent_shape=(self.base_size, self.base_size),
                # does not changed
                **self.rope_options,
            )
        else:
            raise ValueError(
                f"Unsupported pos_embed_type: {self.pos_embed_type}. "
                "Supported types are 'sincos' and 'rope'."
            )

    def get_pe(self, hw: tuple | torch.Size, img_type=None):
        h, w = hw
        if self.pos_embed_type == "sincos":
            if img_type == "raw":
                assert self.with_raw_img
                pe = self.pos_embed_raw
                name = "pos_embed_raw"
                base_size = self.raw_img_size // self.raw_patch_size
                ps = self.raw_patch_size
            else:
                pe = self.pos_embed_latent
                name = "pos_embed_latent"
                base_size = self.base_size
                ps = self.patch_size

            if pe.shape[1] != h * w:
                # re-init the pos_embed
                pe = get_2d_sincos_pos_embed(
                    pe.shape[-1],
                    (h // ps, w // ps),
                    pe_interpolation=self.pe_interpolation,
                    base_size=base_size,
                )
                self.register_buffer(
                    name, torch.from_numpy(pe).float().unsqueeze(0), persistent=False
                )
            return pe

        elif self.pos_embed_type == "rope":
            # TODO: add multi-modal-rope
            # (modalities -> ids -> online RoPE class -> positional embedding -> kv rope fn)

            ph, pw = h // self.patch_size, w // self.patch_size
            seq_len_x = ph * pw
            pre_rope_seq_len = self.rope.cos_cached.shape[1]
            if seq_len_x > pre_rope_seq_len:
                # re-init the rope
                self.rope_options["latent_shape"] = (ph, pw)
                self.rope.__init__(seq_len_x, **self.rope_options)
            return self.rope

        else:
            raise ValueError(
                f"Unsupported pos_embed_type: {self.pos_embed_type}. "
                "Supported types are 'sincos' and 'rope'."
            )

    def unpatchify(self, x: torch.Tensor):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = rearrange(
            x, "bs (h w) (p1 p2 c) -> bs c (h p1) (w p2)", h=h, w=w, p1=p, p2=p, c=c
        )
        return x

    def forward(
        self,
        noisy_latent: Float[Tensor, "b c_latent h w"],
        hyper_img: Float[Tensor, "b c_hyper H W"] | None = None,
        *,
        feature_layer_ids: list[int] | None = None,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        # patch embedding
        y = self.patch_embed(noisy_latent)

        # positional embedding
        pe = self.get_pe(noisy_latent.shape[-2:])
        if self.pos_embed_type == "sincos":
            assert torch.is_tensor(pe), "Positional embedding must be a tensor."
            y = y + pe.to(y)
            pe = None

        if self.with_raw_img:
            assert hyper_img is not None
            y2 = self.raw_img_patcher(hyper_img)

            if self.pos_embed_type == "sincos":
                pe2 = self.get_pe(hyper_img.shape[-2:], img_type="raw")
                assert torch.is_tensor(pe2), "Positional embedding must be a tensor."
                y2 = y2 + pe2.to(y2)

            y = torch.cat([y, y2], dim=1)  # [bs, s1 + s2, c]

        # blocks
        feature_layer_ids_ = self.feature_layer_ids or feature_layer_ids
        features = []
        for i, layer in enumerate(self.layers):
            y = layer(y, mask=None, pe=pe)
            if feature_layer_ids_ is not None and i in feature_layer_ids_:
                features.append(y)

        # head
        y = y[:, : y.size(1)]
        y = self.head(y)

        # unpatchify
        out = self.unpatchify(y)

        if len(features) == 0:
            return out
        else:
            return out, features

    def init_weights(self):
        # Initialize transformer layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # patch embedding
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))


if __name__ == "__main__":
    device = "cuda:1"
    torch.cuda.set_device(device)
    # x = torch.randn(1, 768, 128).to(device)

    # rmsnorm = RMSNorm(dim=128)
    # print(rmsnorm(x).shape)

    # attn = Attention(128, 8, qk_norm=RMSNormFlash).to(device)
    # print(attn(x).shape)

    # mlp = ClipMlp(
    #     in_features=128,
    #     hidden_features=256,
    #     out_features=128,
    #     norm_layer=RMSNormFlash,
    # ).to(device)
    # print(mlp(x).shape)

    # model = Transformer(
    #     16, 128, 4, 8, pos_embed_type="rope", norm_layer=nn.LayerNorm, input_size=32
    # ).to(device)
    # x = torch.randn(1, 16, 64, 64).to(device)
    # out = model(x)
    # print(out.shape)  # Expected shape: (1, 16, 32, 32)

    block = NatAttention(128).cuda()
    x = torch.randn(1, 128, 32, 32).cuda()
    print(block(x).shape)

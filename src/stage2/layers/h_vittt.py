# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# ViT^3: Unlocking Test-Time Training in Vision
# Modified by Dongchen Han
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, create_rope_embed, to_2tuple, trunc_normal_


class TTT(nn.Module):
    r"""Test-Time Training block for ViT^3 model.
        - https://arxiv.org/abs/2512.01643

    This block implements test-time inner training of two parallel sub-modules:
        1. Simplified SwiGLU inner module, i.e., SwiGLU with identity output layer
        2. 3x3 depth-wise convolution (3x3dwc) inner module

    Note:
        The TTT inner loss is a per-head / per-sample vector-valued loss (shape [B, num_heads]).
        The torch.autograd.backward only supports scalar losses, so here we implement a hand-derived
        backward (closed-form gradient expressions) that directly computes parameter gradients.
        Alternative efficient implementations are welcome and appreciated.

    Args:
        dim (int): Number of input channels
        num_heads (int): Number of attention heads
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, num_heads, qkv_bias=True, **kwargs):
        super().__init__()
        head_dim = dim // num_heads
        self.dim = dim
        self.num_heads = num_heads

        self.qkv = nn.Linear(dim, dim * 3 + head_dim * 3, bias=qkv_bias)
        self.w1 = nn.Parameter(torch.zeros(1, self.num_heads, head_dim, head_dim))
        self.w2 = nn.Parameter(torch.zeros(1, self.num_heads, head_dim, head_dim))
        self.w3 = nn.Parameter(torch.zeros(head_dim, 1, 3, 3))
        trunc_normal_(self.w1, std=0.02)
        trunc_normal_(self.w2, std=0.02)
        trunc_normal_(self.w3, std=0.02)
        self.proj = nn.Linear(dim + head_dim, dim)

        equivalent_head_dim = 9
        self.scale = equivalent_head_dim**-0.5
        # The equivalent head_dim of 3x3dwc branch is 1x(3x3)=9 (1 channel, 3x3 kernel)
        # We used this equivalent_head_dim to compute self.scale in our earlier experiments
        # Using self.scale=head_dim**-0.5 (head_dim of simplified SwiGLU branch) leads to similar performance

    def inner_train_simplified_swiglu(self, k, v, w1, w2, lr=1.0):
        """
        Args:
            k (torch.Tensor): Key tensor of shape [B, num_heads, N, head_dim]
            v (torch.Tensor): Value tensor of shape [B, num_heads, N, head_dim]
            w1 (torch.Tensor): First weight matrix of shape [1, num_heads, head_dim, head_dim]
            w2 (torch.Tensor): Second weight matrix of shape [1, num_heads, head_dim, head_dim]
            lr (float, optional): Learning rate for inner-loop update. Default: 1.0

        Returns:
            tuple: Updated w1 and w2
        """
        # --- Forward ---
        z1 = k @ w1
        z2 = k @ w2
        sig = F.sigmoid(z2)
        a = z2 * sig
        # v_hat = a
        # l = (v_hat * v).sum(dim=3).mean(dim=2) * self.scale
        # Notably, v_hat and l are not computed here because
        # they are unnecessary for deriving the gradient expression below.
        # We directly compute e = dl/dv_hat for the backward pass.

        # --- Backward ---
        e = -v / float(v.shape[2]) * self.scale
        g1 = k.transpose(-2, -1) @ (e * a)
        g2 = k.transpose(-2, -1) @ (e * z1 * (sig * (1.0 + z2 * (1.0 - sig))))

        # --- Clip gradient (for stability) ---
        g1 = g1 / (g1.norm(dim=-2, keepdim=True) + 1.0)
        g2 = g2 / (g2.norm(dim=-2, keepdim=True) + 1.0)

        # --- Step ---
        w1, w2 = w1 - lr * g1, w2 - lr * g2
        return w1, w2

    def inner_train_3x3dwc(self, k, v, w, lr=1.0, implementation="prod"):
        """
        Args:
            k (torch.Tensor): Spatial key tensor of shape [B, C, H, W]
            v (torch.Tensor): Spatial value tensor of shape [B, C, H, W]
            w (torch.Tensor): 3x3 convolution weights of shape [C, 1, 3, 3]
            lr (float, optional): Learning rate for inner-loop update. Default: 1.0
            implementation (str, optional): Implementation method, 'conv' or 'prod'. Default: 'prod'

        Returns:
            torch.Tensor: Updated convolution weights
        """
        # --- Forward ---
        # v_hat = F.conv2d(k, w, padding=1, groups=C)
        # l = - (v_hat * v).mean(dim=[-2, -1]) * self.scale
        # Notably, v_hat and l are not computed here because
        # they are unnecessary for deriving the gradient expression below.
        # We directly compute e = dl/dv_hat for the backward pass.

        # --- Backward ---
        # Two equivalent implementations. The 'prod' implementation appears to be slightly faster
        B, C, H, W = k.shape
        e = -v / float(v.shape[2] * v.shape[3]) * self.scale
        if implementation == "conv":
            g = F.conv2d(k.reshape(1, B * C, H, W), e.reshape(B * C, 1, H, W), padding=1, groups=B * C)
            g = g.transpose(0, 1)
        elif implementation == "prod":
            k = F.pad(k, (1, 1, 1, 1))
            outs = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ys = 1 + dy
                    xs = 1 + dx
                    dot = (k[:, :, ys : ys + H, xs : xs + W] * e).sum(dim=(-2, -1))
                    outs.append(dot)
            g = torch.stack(outs, dim=-1).reshape(B * C, 1, 3, 3)
        else:
            raise NotImplementedError

        # --- Clip gradient (for stability) ---
        g = g / (g.norm(dim=[-2, -1], keepdim=True) + 1.0)

        # --- Step ---
        w = w.repeat(B, 1, 1, 1) - lr * g
        return w

    def forward(self, x, h, w, rope=None):
        """
        Args:
            x (torch.Tensor): Input features with shape of (B, N, C)
            h (int): Feature map height
            w (int): Feature map width
            rope (nn.Module, optional): Rotary Position Embedding
        """
        b, n, c = x.shape
        d = c // self.num_heads

        # Prepare q/k/v
        q1, k1, v1, q2, k2, v2 = torch.split(self.qkv(x), [c, c, c, d, d, d], dim=-1)
        if rope is not None:
            q1 = rope(q1.reshape(b, h, w, c)).reshape(b, n, self.num_heads, d).transpose(1, 2)
            k1 = rope(k1.reshape(b, h, w, c)).reshape(b, n, self.num_heads, d).transpose(1, 2)
        else:
            q1 = q1.reshape(b, n, self.num_heads, d).transpose(1, 2)  # [B, N, H, D]
            k1 = k1.reshape(b, n, self.num_heads, d).transpose(1, 2)
        v1 = v1.reshape(b, n, self.num_heads, d).transpose(1, 2)

        q2 = q2.reshape(b, h, w, d).permute(0, 3, 1, 2)  # [B, D, H, W]
        k2 = k2.reshape(b, h, w, d).permute(0, 3, 1, 2)
        v2 = v2.reshape(b, h, w, d).permute(0, 3, 1, 2)

        # Inner training using (k, v)
        w1, w2 = self.inner_train_simplified_swiglu(k1, v1, self.w1, self.w2)
        w3 = self.inner_train_3x3dwc(k2, v2, self.w3, implementation="prod")

        # Apply updated inner module to q
        # [B, N, H, D] @ [1, H, D, D] * silu([B, N, H, D] @ [1, H, D, D]) = [B, N, H, D]
        x1 = (q1 @ w1) * F.silu(q1 @ w2)
        x1 = x1.transpose(1, 2).reshape(b, n, c)
        x2 = F.conv2d(q2.reshape(1, b * d, h, w), w3, padding=1, groups=b * d)
        x2 = x2.reshape(b, d, n).transpose(1, 2)

        # Output proj
        x = torch.cat([x1, x2], dim=-1)
        x = self.proj(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}"


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwc = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, h, w):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = x + self.dwc(x.reshape(x.shape[0], h, w, x.shape[-1]).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        dropout=0,
        norm=nn.BatchNorm2d,
        act_func=nn.ReLU,
    ):
        super(ConvLayer, self).__init__()
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=bias,
        )
        self.norm = norm(num_features=out_channels) if norm else None
        self.act = act_func() if act_func else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class RoPE(torch.nn.Module):
    r"""Rotary Positional Embedding."""

    def __init__(self, shape, base=10000):
        super(RoPE, self).__init__()

        channel_dims, feature_dim = shape[:-1], shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))

        assert feature_dim % k_max == 0

        # angles
        theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))
        angles = torch.cat(
            [
                t.unsqueeze(-1) * theta_ks
                for t in torch.meshgrid([torch.arange(d) for d in channel_dims], indexing="ij")
            ],
            dim=-1,
        )

        # rotation
        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
        self.register_buffer("rotations", rotations)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = torch.view_as_complex(self.rotations) * x
        return torch.view_as_real(pe_x).flatten(-2)


class TTTBlock(nn.Module):
    r"""TTT Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.cpe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.rope = RoPE(shape=(input_resolution[0], input_resolution[1], dim))
        self.attn = TTT(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x + self.cpe(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)

        # Attention
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, self.rope))

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, mlp_ratio={self.mlp_ratio}"


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        dim_out (int): Number of output channels.
    """

    def __init__(self, input_resolution, dim, dim_out, ratio=4.0):
        super().__init__()
        self.input_resolution = input_resolution
        in_channels = dim
        out_channels = dim_out
        self.conv = nn.Sequential(
            ConvLayer(in_channels, int(out_channels * ratio), kernel_size=1, norm=None),
            ConvLayer(
                int(out_channels * ratio),
                int(out_channels * ratio),
                kernel_size=3,
                stride=2,
                padding=1,
                groups=int(out_channels * ratio),
                norm=None,
            ),
            ConvLayer(int(out_channels * ratio), out_channels, kernel_size=1, act_func=None),
        )

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = self.conv(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        return x


class BasicLayer(nn.Module):
    """A basic TTT layer for one stage.

    Args:
        dim (int): Number of input channels.
        dim_out (int): Number of output channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        dim_out,
        input_resolution,
        depth,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList(
            [
                TTTBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, dim_out=dim_out)
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


class Stem(nn.Module):
    r"""Stem

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.patches_resolution = patches_resolution

        self.conv = nn.Sequential(
            ConvLayer(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            ConvLayer(embed_dim // 2, embed_dim * 4, kernel_size=3, stride=2, padding=1, bias=False),
            ConvLayer(embed_dim * 4, embed_dim, kernel_size=1, bias=False, act_func=None),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ViTTT(nn.Module):
    r"""ViTTT
        A PyTorch impl of : `ViT^3: Unlocking Test-Time Training in Vision`

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        dim (tuple(int)): Patch embedding dimension of each TTT layer.
        depths (tuple(int)): Depth of each TTT layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        dim=[96, 192, 384, 768],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = dim[0]
        self.mlp_ratio = mlp_ratio

        self.patch_embed = Stem(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=dim[0])
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=dim[i_layer],
                dim_out=dim[i_layer + 1] if i_layer < self.num_layers - 1 else None,
                input_resolution=(patches_resolution[0] // (2**i_layer), patches_resolution[1] // (2**i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.norm = nn.BatchNorm1d(dim[-1])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(dim[-1], num_classes) if num_classes > 0 else nn.Identity()

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

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x.transpose(1, 2))
        x = self.avgpool(x)  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def h_vittt_tiny(**kwargs):
    model = ViTTT(dim=[64, 128, 320, 512], depths=[1, 3, 9, 4], num_heads=[2, 4, 10, 16], **kwargs)
    return model


def h_vittt_small(**kwargs):
    model = ViTTT(dim=[64, 128, 320, 512], depths=[2, 6, 18, 8], num_heads=[2, 4, 10, 16], **kwargs)
    return model


def h_vittt_base(**kwargs):
    model = ViTTT(dim=[96, 192, 448, 640], depths=[2, 6, 18, 8], num_heads=[3, 6, 14, 20], **kwargs)
    return model

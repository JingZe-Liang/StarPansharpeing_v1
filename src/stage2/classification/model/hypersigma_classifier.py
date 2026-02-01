"""HyperSIGMA backbone for remote sensing scene classification.

Adapted from HyperSIGMA: https://github.com/laprf/HyperSIGMA
This implementation removes external dependencies (mmengine) and provides
a consistent interface for scene classification tasks.
"""

import math
from functools import partial
from typing import Literal, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import resample_abs_pos_embed
from timm.models.layers import drop_path, to_2tuple, trunc_normal_


def get_reference_points(spatial_shapes: tuple[int, int], device: torch.device) -> torch.Tensor:
    """Generate normalized reference points for deformable attention."""
    H_, W_ = spatial_shapes[0], spatial_shapes[1]
    ref_y, ref_x = torch.meshgrid(
        torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
        torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
        indexing="ij",
    )
    ref_y = ref_y.reshape(-1)[None] / H_
    ref_x = ref_x.reshape(-1)[None] / W_
    ref = torch.stack((ref_x, ref_y), -1)
    return ref


def deform_inputs_func(x: torch.Tensor, patch_size: int) -> list:
    """Prepare inputs for deformable attention."""
    B, c, h, w = x.shape
    spatial_shapes = torch.as_tensor([h // patch_size, w // patch_size], dtype=torch.long, device=x.device)
    reference_points = get_reference_points((h // patch_size, w // patch_size), x.device)
    deform_inputs = [reference_points, spatial_shapes]
    return deform_inputs


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"


class Mlp(nn.Module):
    """MLP with GELU activation."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SampleAttention(nn.Module):
    """Deformable sampling attention mechanism."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        window_size: tuple[int, int] | None = None,
        attn_head_dim: int | None = None,
        n_points: int = 4,
    ) -> None:
        super().__init__()
        self.n_points = n_points
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)
        self.sampling_offsets = nn.Linear(all_head_dim, self.num_heads * n_points * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, H: int, W: int, deform_inputs: list) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, -1).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        reference_points, input_spatial_shapes = deform_inputs

        sampling_offsets = self.sampling_offsets(q).reshape(B, N, self.num_heads, self.n_points, 2).transpose(1, 2)

        _, _, L = q.shape
        q = q.reshape(B, N, self.num_heads, L // self.num_heads).transpose(1, 2)

        offset_normalizer = torch.stack([input_spatial_shapes[1], input_spatial_shapes[0]])

        sampling_locations = (
            reference_points[:, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, None, :]
        )
        sampling_locations = 2 * sampling_locations - 1  # [0, 1] -> [-1, 1]

        k = k.reshape(B, N, self.num_heads, L // self.num_heads).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, L // self.num_heads).transpose(1, 2)

        # B*H, c, H, W
        k = (
            k.flatten(0, 1)
            .transpose(1, 2)
            .reshape(
                B * self.num_heads,
                L // self.num_heads,
                input_spatial_shapes[0],
                input_spatial_shapes[1],
            )
        )
        v = (
            v.flatten(0, 1)
            .transpose(1, 2)
            .reshape(
                B * self.num_heads,
                L // self.num_heads,
                input_spatial_shapes[0],
                input_spatial_shapes[1],
            )
        )

        # B*H, N, P, 2
        sampling_locations = sampling_locations.flatten(0, 1).reshape(B * self.num_heads, N, self.n_points, 2)

        q = q[:, :, :, None, :]  # B, H, N, 1, C

        # B*H, c, N, P
        sampled_k = (
            F.grid_sample(k, sampling_locations, mode="bilinear", padding_mode="zeros", align_corners=False)
            .reshape(B, self.num_heads, L // self.num_heads, N, self.n_points)
            .permute(0, 1, 3, 4, 2)
        )

        sampled_v = (
            F.grid_sample(v, sampling_locations, mode="bilinear", padding_mode="zeros", align_corners=False)
            .reshape(B, self.num_heads, L // self.num_heads, N, self.n_points)
            .permute(0, 1, 3, 4, 2)
        )

        attn = (q * sampled_k).sum(-1) * self.scale  # B, H, N, P

        attn = attn.softmax(dim=-1)[:, :, :, :, None]  # B, H, N, P, 1

        x = (attn * sampled_v).sum(-2).transpose(1, 2).reshape(B, N, -1)  # B, H, N, c

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        window_size: tuple[int, int] | None = None,
        attn_head_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, H: int, W: int, rel_pos_bias: torch.Tensor | None = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # B,H,N,N

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Transformer block with attention and MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: float | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        window_size: tuple[int, int] | None = None,
        attn_head_dim: int | None = None,
        sample: bool = False,
        restart_regression: bool = True,
        n_points: int | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.sample = sample

        if not sample:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                window_size=window_size,
                attn_head_dim=attn_head_dim,
            )
        else:
            self.attn = SampleAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                window_size=window_size,
                attn_head_dim=attn_head_dim,
                n_points=n_points or 4,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x: torch.Tensor, H: int, W: int, deform_inputs: list) -> torch.Tensor:
        if self.gamma_1 is None:
            if not self.sample:
                x = x + self.drop_path(self.attn(self.norm1(x), H, W))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
            else:
                x = x + self.drop_path(self.attn(self.norm1(x), H, W, deform_inputs))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            if not self.sample:
                x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), H, W))
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            else:
                x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), H, W, deform_inputs))
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768) -> None:
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor, **kwargs) -> tuple[torch.Tensor, tuple[int, int]]:
        B, C, H, W = x.shape
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)


class Norm2d(nn.Module):
    """2D LayerNorm."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class SpatViT(nn.Module):
    """Spatial Vision Transformer backbone."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 80,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        init_values: float | None = None,
        use_checkpoint: bool = False,
        use_abs_pos_emb: bool = False,
        out_indices: list[int] | None = None,
        interval: int = 3,
        n_points: int = 4,
        use_ssa: bool = True,
    ) -> None:
        super().__init__()
        if out_indices is None:
            out_indices = [11]
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.out_indices = out_indices

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(in_features=embed_dim * 4, out_features=128),
            nn.Linear(in_features=128, out_features=64),
            nn.Linear(in_features=64, out_features=num_classes),
        )

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            self.pos_embed = None

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                    sample=(use_ssa and ((i + 1) % interval != 0)),
                    n_points=n_points,
                )
                for i in range(depth)
            ]
        )

        self.interval = interval

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        self.norm = norm_layer(embed_dim)

        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self) -> None:
        """Rescale weights for better initialization."""

        def rescale(param: torch.Tensor, layer_id: int) -> None:
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: torch.Tensor, patch_size: int) -> list[torch.Tensor]:
        """Extract features from intermediate layers."""
        img = [x]
        deform_inputs = deform_inputs_func(x, patch_size)

        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)

        if self.pos_embed is not None:
            pos_embed = self._resize_pos_embed(self.pos_embed, (Hp, Wp))
            x = x + pos_embed
        x = self.pos_drop(x)

        features = []
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, Hp, Wp, deform_inputs)
            else:
                x = blk(x, Hp, Wp, deform_inputs)

            if i in self.out_indices:
                features.append(x)

        features = [feat.permute(0, 2, 1).reshape(B, -1, Hp, Wp) for feat in features]
        return img + features

    def _resize_pos_embed(self, pos_embed: torch.Tensor, new_size: tuple[int, int]) -> torch.Tensor:
        num_patches = pos_embed.shape[1]
        old_size = int(num_patches**0.5)
        if old_size * old_size != num_patches:
            return pos_embed
        if (old_size, old_size) == new_size:
            return pos_embed
        return resample_abs_pos_embed(pos_embed, new_size=new_size, old_size=(old_size, old_size), num_prefix_tokens=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        features = self.forward_features(x, self.patch_size)
        feature1 = features[4]
        feature2 = features[3]
        feature3 = features[2]
        feature4 = features[1]

        # Global average pooling
        y1 = F.avg_pool2d(feature1, feature1.size()[2:])
        y2 = F.avg_pool2d(feature2, feature2.size()[2:])
        y3 = F.avg_pool2d(feature3, feature3.size()[2:])
        y4 = F.avg_pool2d(feature4, feature4.size()[2:])

        y1 = y1.view(feature1.size(0), -1)
        y2 = y2.view(feature2.size(0), -1)
        y3 = y3.view(feature3.size(0), -1)
        y4 = y4.view(feature4.size(0), -1)

        output = torch.concat((y1, y2, y3, y4), 1)
        output = self.classifier(output)
        return output


class HyperSIGMAClassifier(nn.Module):
    """HyperSIGMA backbone with classifier for scene classification.

    Args:
        backbone_type: Type of backbone - "spat_vit_b", "spat_vit_l", "spat_vit_h"
        weights_path: Path to pretrained weights
        pretrained: Whether to load pretrained weights
        freeze_backbone: Whether to freeze backbone during training
        num_classes: Number of output classes
        img_size: Input image size
        patch_size: Patch size for SpatViT backbone.
        use_abs_pos_emb: Whether to enable absolute positional embedding.
        ssa_interval: Interval for SSA sampling blocks.
        use_ssa: Whether to enable SSA sampling attention.
        expected_in_chans: Expected input channels for HyperSIGMA pretrain (default: 100).
        enable_input_proj: If True, create a 1x1 conv when input channels != expected_in_chans.
        img_is_neg_1_1: Whether input is in [-1, 1] range
        input_scale: Optional scale factor applied after range mapping, before normalization.
        input_mean: Per-channel mean for input normalization (applied after range mapping).
        input_std: Per-channel std for input normalization (applied after range mapping).
    """

    SUPPORTED_BACKBONES: dict[str, dict[str, int | list[int]]] = {
        "spat_vit_b": {
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "out_indices": [3, 5, 7, 11],
            "interval": 3,
        },
        "spat_vit_l": {
            "embed_dim": 1024,
            "depth": 24,
            "num_heads": 16,
            "out_indices": [7, 11, 15, 23],
            "interval": 6,
        },
        "spat_vit_h": {
            "embed_dim": 1280,
            "depth": 32,
            "num_heads": 16,
            "out_indices": [10, 15, 20, 31],
            "interval": 8,
        },
    }

    def __init__(
        self,
        backbone_type: Literal["spat_vit_b", "spat_vit_l", "spat_vit_h"] = "spat_vit_b",
        weights_path: str | None = None,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        num_classes: int = 21,
        img_size: int = 224,
        patch_size: int = 16,
        use_abs_pos_emb: bool = False,
        ssa_interval: int = 6,
        use_ssa: bool = True,
        expected_in_chans: int = 100,
        enable_input_proj: bool = True,
        img_is_neg_1_1: bool = True,
        input_scale: float | None = None,
        input_mean: list[float] | None = None,
        input_std: list[float] | None = None,
    ) -> None:
        super().__init__()

        if backbone_type not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone: {backbone_type}. Choose from {list(self.SUPPORTED_BACKBONES.keys())}"
            )

        self.backbone_type = backbone_type
        self.freeze_backbone = freeze_backbone
        self.img_is_neg_1_1 = img_is_neg_1_1
        self.expected_in_chans = expected_in_chans
        self.enable_input_proj = enable_input_proj
        self.input_scale = self._validate_input_scale(input_scale)
        self._set_input_norm(input_mean, input_std)

        # Build backbone
        config = self.SUPPORTED_BACKBONES[backbone_type]
        self.backbone = SpatViT(
            img_size=img_size,
            in_chans=expected_in_chans,
            patch_size=patch_size,
            drop_path_rate=0.1,
            out_indices=config["out_indices"],  # type: ignore[arg-type]
            embed_dim=config["embed_dim"],  # type: ignore[arg-type]
            depth=config["depth"],  # type: ignore[arg-type]
            num_heads=config["num_heads"],  # type: ignore[arg-type]
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            use_checkpoint=False,
            use_abs_pos_emb=use_abs_pos_emb,
            interval=ssa_interval if use_ssa else 1,
            num_classes=num_classes,
            use_ssa=use_ssa,
        )

        self.input_proj = self._build_input_proj()

        if pretrained and weights_path:
            self.load_pretrained_weights(weights_path)

        if freeze_backbone:
            # Freeze only Transformer blocks, keep patch_embed and classifier trainable
            # This allows the model to adapt to RGB inputs (3 channels)
            for name, param in self.backbone.named_parameters():
                # Freeze Transformer blocks and position embeddings
                if name.startswith("blocks.") or name.startswith("norm.") or name.startswith("pos_"):
                    param.requires_grad_(False)
                # Keep patch_embed trainable (needs to learn RGB -> features mapping)
                elif name.startswith("patch_embed."):
                    param.requires_grad_(True)
                # Keep classifier trainable
                elif name.startswith("classifier."):
                    param.requires_grad_(True)

            # Set Transformer blocks to eval mode, keep patch_embed in train mode
            for name, module in self.backbone.named_modules():
                if name.startswith("blocks.") or name == "norm":
                    module.eval()

    def load_pretrained_weights(self, weights_path: str) -> None:
        """Load pretrained weights from checkpoint.

        For RGB inputs (3 channels), we skip patch_embed layers as they are
        trained on hyperspectral images (~100 channels). Only Transformer
        blocks are loaded from pretrained weights.
        """
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # Strip prefix if present
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        # Remove encoder prefix if present
        if sorted(list(state_dict.keys()))[0].startswith("encoder"):
            state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")}

        model_state = self.backbone.state_dict()

        # Filter out incompatible layers for RGB inputs or mismatched configs
        # HyperSIGMA pretrained weights are for hyperspectral images (~100 channels)
        keys_to_remove = []
        for k in list(state_dict.keys()):
            # Remove patch embedding layers (channel mismatch: 100 -> 3)
            if "patch_embed" in k:
                if k not in model_state or model_state[k].shape != state_dict[k].shape:
                    keys_to_remove.append(k)
            # Remove MAE decoder layers (not needed for classification)
            elif "decoder" in k or "mask_token" in k:
                keys_to_remove.append(k)
            # Remove classifier/head layers (task-specific)
            elif "classifier" in k or "DR" in k or "cls" in k or "fpn" in k:
                keys_to_remove.append(k)

        print(f"Loading HyperSIGMA pretrained weights from:")
        print(f"   {weights_path}")
        print(f"   Loading {len(state_dict)} compatible Transformer parameters")
        print(f"   Skipping {len(keys_to_remove)} incompatible layers:")
        for k in keys_to_remove:
            print(f"      - {k}")
            if k in state_dict:
                del state_dict[k]

        if "pos_embed" in state_dict and self.backbone.pos_embed is not None:
            ckpt_pos = state_dict["pos_embed"]
            model_pos = self.backbone.pos_embed
            if ckpt_pos.shape != model_pos.shape:
                num_patches = ckpt_pos.shape[1]
                old_size = int(num_patches**0.5)
                if old_size * old_size == num_patches:
                    state_dict["pos_embed"] = resample_abs_pos_embed(
                        ckpt_pos,
                        new_size=self.backbone.patch_embed.patch_shape,
                        old_size=(old_size, old_size),
                        num_prefix_tokens=0,
                    )

        msg = self.backbone.load_state_dict(state_dict, strict=False)
        print(f"   Missing: {len(msg.missing_keys)} keys (patch_embed, classifier, etc.)")
        for k in msg.missing_keys:
            print(f"      - {k}")
        print(f"   Unexpected: {len(msg.unexpected_keys)} keys")
        for k in msg.unexpected_keys:
            print(f"      - {k}")

    def train(self, mode: bool = True) -> "HyperSIGMAClassifier":
        super().train(mode)
        if self.freeze_backbone:
            # Keep Transformer blocks in eval mode
            for name, module in self.backbone.named_modules():
                if name.startswith("blocks.") or name == "norm":
                    module.eval()
            # patch_embed and classifier remain in train mode
        return self

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Dictionary with "logits" key
        """
        x = self._preprocess_input(x)
        x = self._maybe_project_input(x)
        logits = self.backbone(x)
        return {"logits": logits}

    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess input: convert [-1,1] to [0,1] if needed."""
        if self.img_is_neg_1_1:
            x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        x = self._scale_input(x)
        return self._normalize_input(x)

    @staticmethod
    def _validate_input_scale(input_scale: float | None) -> float | None:
        if input_scale is None:
            return None
        if input_scale <= 0:
            raise ValueError("input_scale must be a positive number.")
        return float(input_scale)

    def _set_input_norm(
        self,
        input_mean: list[float] | None,
        input_std: list[float] | None,
    ) -> None:
        if (input_mean is None) != (input_std is None):
            raise ValueError("input_mean and input_std must both be set or both be None.")
        if input_mean is None:
            return
        mean = torch.tensor(input_mean).view(1, -1, 1, 1)
        std = torch.tensor(input_std).view(1, -1, 1, 1)
        self.register_buffer("_input_mean", mean, persistent=False)
        self.register_buffer("_input_std", std, persistent=False)

    def _scale_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_scale is None:
            return x
        return x * self.input_scale

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "_input_mean") or not hasattr(self, "_input_std"):
            return x
        mean = cast(torch.Tensor, getattr(self, "_input_mean")).to(device=x.device, dtype=x.dtype)
        std = cast(torch.Tensor, getattr(self, "_input_std")).to(device=x.device, dtype=x.dtype)
        if mean.numel() not in (1, x.shape[1]):
            raise ValueError(f"input_mean channels {mean.numel()} do not match input channels {x.shape[1]}.")
        if std.numel() not in (1, x.shape[1]):
            raise ValueError(f"input_std channels {std.numel()} do not match input channels {x.shape[1]}.")
        return (x - mean) / std

    def _maybe_project_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == self.expected_in_chans:
            return x
        if not self.enable_input_proj:
            raise ValueError(
                f"Input channels {x.shape[1]} != expected {self.expected_in_chans}, and enable_input_proj is False."
            )
        return self.input_proj(x)

    def _build_input_proj(self) -> nn.Module:
        if not self.enable_input_proj:
            return nn.Identity()
        return nn.LazyConv2d(self.expected_in_chans, kernel_size=1, bias=False)

    def state_dict(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Return only classifier weights for checkpoint."""
        full_state = super().state_dict(*args, **kwargs)
        return {k: v for k, v in full_state.items() if (not k.startswith("backbone.") or "classifier" in k)}

    def load_state_dict(
        self, state_dict: dict[str, torch.Tensor], strict: bool = True
    ) -> torch.nn.modules.module._IncompatibleKeys:
        """Load classifier weights only."""
        filtered_state = {k: v for k, v in state_dict.items() if (not k.startswith("backbone.") or "classifier" in k)}
        if not filtered_state:
            # If state_dict doesn't have backbone.classifier prefix, add it
            filtered_state = {f"backbone.classifier.{k}": v for k, v in state_dict.items()}
        return super().load_state_dict(filtered_state, strict=False)

    def parameters(self, recurse: bool = True):
        """Return only trainable parameters if freeze_backbone is True."""
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        """Yield only trainable parameters if freeze_backbone is True."""
        for name, param in super().named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate):
            if self.freeze_backbone:
                if param.requires_grad:
                    yield name, param
            else:
                yield name, param


def test_hypersigma_classifier() -> None:
    """Test HyperSIGMAClassifier with different backbones."""
    print("Testing HyperSIGMA Classifier...")

    for backbone_type in ["spat_vit_b", "spat_vit_l", "spat_vit_h"]:
        print(f"\n{'=' * 60}")
        print(f"Testing {backbone_type}")
        print(f"{'=' * 60}")

        model = HyperSIGMAClassifier(
            backbone_type=backbone_type,  # type: ignore[arg-type]
            weights_path=None,
            pretrained=False,
            freeze_backbone=True,
            num_classes=21,
            img_size=224,
        )

        # Test input
        batch_size = 2
        x = torch.randn(batch_size, 3, 224, 224)

        print(f"Input shape: {x.shape}")
        print(f"Backbone embed_dim: {model.backbone.embed_dim}")

        model.eval()
        with torch.no_grad():
            output = model(x)

        logits = output["logits"]
        print(f"Output logits shape: {logits.shape}")
        assert logits.shape == (batch_size, 21), f"Expected (2, 21), got {logits.shape}"
        print(f"✅ {backbone_type} test passed!")


def test_hypersigma_checkpoint_match() -> None:
    """Test pretrained checkpoint key matching for HyperSIGMA."""
    from pathlib import Path

    ckpt_path = Path("/Data2/ZihanCao/Checkpoints/Hypersigma/spat-vit-large-ultra-checkpoint-1599.pth")
    if not ckpt_path.exists():
        print(f"checkpoint not found: {ckpt_path}")
        return

    model = HyperSIGMAClassifier(
        backbone_type="spat_vit_l",
        weights_path=None,
        pretrained=False,
        freeze_backbone=True,
        num_classes=21,
        img_size=64,
        patch_size=8,
        use_abs_pos_emb=True,
        ssa_interval=6,
        use_ssa=False,  # Pretrained weights don't have SSA parameters
        expected_in_chans=100,
        enable_input_proj=True,
    )

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    if sorted(list(state_dict.keys()))[0].startswith("encoder"):
        state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")}

    model_state = model.backbone.state_dict()
    matched = 0
    missing: list[str] = []
    unexpected: list[str] = []
    mismatched: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    for key, value in state_dict.items():
        if key not in model_state:
            unexpected.append(key)
            continue
        if model_state[key].shape != value.shape:
            mismatched.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        matched += 1

    for key in model_state.keys():
        if key not in state_dict:
            missing.append(key)

    print(f"matched: {matched}")
    print(f"missing: {len(missing)}")
    print(f"unexpected: {len(unexpected)}")
    print(f"mismatched: {len(mismatched)}")
    if missing:
        print(f"missing sample: {missing[:10]}")
    if unexpected:
        print(f"unexpected sample: {unexpected[:10]}")
    if mismatched:
        print(f"mismatched sample: {mismatched[:10]}")

    x = torch.randn(2, 100, 64, 64)
    model.eval()
    with torch.no_grad():
        output = model(x)
    logits = output["logits"]
    print(f"forward logits shape: {logits.shape}")


if __name__ == "__main__":
    # test_hypersigma_classifier()
    test_hypersigma_checkpoint_match()

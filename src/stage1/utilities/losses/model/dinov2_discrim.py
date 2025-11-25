import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.nn.utils.spectral_norm import SpectralNorm
from torchvision.transforms import RandomCrop

from src.stage1.utilities.losses.dinov2 import dinov2


class ResidualBlock(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.ratio = 1 / np.sqrt(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.fn(x).add(x)).mul_(self.ratio)


class SpectralConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SpectralNorm.apply(self, name="weight", n_power_iterations=1, dim=0, eps=1e-12)


class BatchNormLocal(nn.Module):
    def __init__(
        self,
        num_features: int,
        affine: bool = True,
        virtual_bs: int = 8,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.virtual_bs = virtual_bs
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.float()

        # Reshape batch into groups.
        G = np.ceil(x.size(0) / self.virtual_bs).astype(int)
        x = x.view(G, -1, x.size(-2), x.size(-1))

        # Calculate stats.
        mean = x.mean([1, 3], keepdim=True)
        var = x.var([1, 3], keepdim=True, unbiased=False)
        x = (x - mean) / (torch.sqrt(var + self.eps))

        if self.affine:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]

        return x.view(shape)


def new_local_machine_group():
    if torch.distributed.is_initialized():
        cur_subgroup, subgroups = torch.distributed.new_subgroups()
        return cur_subgroup
    return None


def make_block(channels, kernel_size, norm_type, norm_eps, use_specnorm) -> nn.Module:
    if norm_type == "bn":
        norm = BatchNormLocal(channels, eps=norm_eps)
    elif norm_type == "sbn":
        norm = nn.SyncBatchNorm(channels, eps=norm_eps, process_group=None)
    elif norm_type in {"lbn", "hbn"}:
        norm = nn.SyncBatchNorm(channels, eps=norm_eps, process_group=new_local_machine_group())
    elif norm_type == "gn":
        norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=norm_eps, affine=True)
    else:
        raise NotImplementedError

    conv_class = SpectralConv1d if use_specnorm else nn.Conv1d
    return nn.Sequential(
        conv_class(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode="circular",
        ),
        norm,
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )


def make_dinov2_model(
    *,
    arch_name: str = "vit_small",
    img_size: int = 518,
    patch_size: int = 14,
    init_values: float = 1.0,
    ffn_layer: str = "mlp",
    block_chunks: int = 0,
    num_register_tokens: int = 0,
    interpolate_antialias: bool = False,
    interpolate_offset: float = 0.1,
    **kwargs,
):
    vit_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        init_values=init_values,
        ffn_layer=ffn_layer,
        block_chunks=block_chunks,
        num_register_tokens=num_register_tokens,
        interpolate_antialias=interpolate_antialias,
        interpolate_offset=interpolate_offset,
    )
    vit_kwargs.update(**kwargs)
    model = dinov2.__dict__[arch_name](**vit_kwargs)

    return model


class DinoDiscV2(nn.Module):
    def __init__(
        self,
        ks: int,
        device: str,
        dino_ckpt: str,
        norm_type: str = "bn",
        norm_eps: float = 1e-6,
        use_specnorm: bool = True,
        dino_size: str = "vit_small",
        key_depths: tuple[int, ...] = (2, 5, 8, 11),
        input_channels: int = 3,
        fix_dino: bool = True,
    ):
        super().__init__()
        dino: nn.Module = make_dinov2_model(
            arch_name=dino_size,
            patch_size=14,
            num_register_tokens=4,
            in_chans=3,
        )
        missing_keys, unexpected_keys = dino.load_state_dict(torch.load(dino_ckpt), strict=False)
        if len(missing_keys) > 0:
            logger.warning(f"[DinoDiscV2] Missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            logger.warning(f"[DinoDiscV2] Unexpected keys: {unexpected_keys}")

        _embed_chans = dino.embed_dim
        _patch_size = 14
        dino.patch_embed.proj = nn.Conv2d(
            in_channels=input_channels,
            out_channels=_embed_chans,
            kernel_size=_patch_size,
            stride=_patch_size,
        )

        self.dino: list[nn.Module] = [
            dino.to(device=device),
        ]
        if fix_dino:
            self.dino[0].requires_grad_(False)
            self.dino[0].patch_embed.proj.requires_grad_(True)
            self.dino[0].eval()

        self.is_rgb = input_channels == 3
        if input_channels == 3:
            # Normalize to [-1, 1]
            mean = torch.tensor((0.485, 0.456, 0.406))
            std = torch.tensor((0.229, 0.224, 0.225))
            self.register_buffer("x_scale", (0.5 / std).reshape(1, 3, 1, 1))
            self.register_buffer("x_shift", ((0.5 - mean) / std).reshape(1, 3, 1, 1))

        self.key_depths: tuple[int, ...] = key_depths

        dino_C = self.dino[0].embed_dim
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    make_block(
                        dino_C,
                        kernel_size=1,
                        norm_type=norm_type,
                        norm_eps=norm_eps,
                        use_specnorm=use_specnorm,
                    ),
                    ResidualBlock(
                        make_block(
                            dino_C,
                            kernel_size=ks,
                            norm_type=norm_type,
                            norm_eps=norm_eps,
                            use_specnorm=use_specnorm,
                        )
                    ),
                    (SpectralConv1d if use_specnorm else nn.Conv1d)(dino_C, 1, kernel_size=1, padding=0),
                )
                for _ in range(len(key_depths))  # +1: before all attention blocks
            ]
        )

    def forward(
        self, x: torch.Tensor, grad_ckpt: bool = False
    ) -> torch.Tensor:  # x: image tensor normalized to [-1, 1]
        with torch.cuda.amp.autocast(enabled=False):
            if self.is_rgb:
                x = (self.x_scale * x.float()).add_(self.x_shift)

            H, W = x.shape[-2:]
            near_H = (H // self.dino[0].patch_size) * self.dino[0].patch_size
            near_W = (W // self.dino[0].patch_size) * self.dino[0].patch_size
            if H > near_H and W > near_W:
                if random.random() <= 0.5:
                    x = RandomCrop((near_H, near_W))(x)
                else:
                    x = F.interpolate(x, size=(near_H, near_W), mode="area")
            else:
                x = F.interpolate(x, size=(near_H, near_W), mode="bicubic")

        activations: list[torch.Tensor] = []
        feature_maps = self.dino[0].get_intermediate_layers(x, n=self.key_depths, return_class_token=True, norm=False)
        for patch_tokens, class_token in feature_maps:
            class_token = class_token.unsqueeze(dim=1)
            middle_act = (patch_tokens.float() + class_token.float()).transpose_(1, 2)
            activations.append(middle_act)

        B = x.shape[0]
        predictions: list[torch.Tensor] = []
        for h, act in zip(self.heads, activations):
            if not grad_ckpt:
                predictions.append(h(act).view(B, -1))
            else:
                predictions.append(torch.utils.checkpoint.checkpoint(h, act, use_reentrant=False).view(B, -1))
        return torch.cat(predictions, dim=1)  # cat 5 BL => B, 5L

    def parameters(self):
        ps = []

        for n, p in self.dino[0].named_parameters():
            if p.requires_grad:
                ps.append(p)
            else:
                logger.debug(f"{n} has no grad")

        for p in self.heads:
            ps.extend(p.parameters())

        return ps


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    ks = 9
    norm_type = "bn"
    norm_eps = 1e-6
    dino_C = 384
    key_layers = (2, 5, 8, 11)
    use_specnorm = True
    # heads = nn.ModuleList(
    #     [
    #         nn.Sequential(
    #             make_block(
    #                 dino_C,
    #                 kernel_size=1,
    #                 norm_type=norm_type,
    #                 norm_eps=norm_eps,
    #                 use_specnorm=use_specnorm,
    #             ),
    #             ResidualBlock(
    #                 make_block(
    #                     dino_C,
    #                     kernel_size=ks,
    #                     norm_type=norm_type,
    #                     norm_eps=norm_eps,
    #                     use_specnorm=use_specnorm,
    #                 )
    #             ),
    #             SpectralConv1d(dino_C, 1, kernel_size=1, padding=0),
    #         )
    #         for _ in range(len(key_layers) + 1)
    #     ]
    # )

    ckpt = "src/stage1/utilities/losses/dinov2/ckpt/dinov2_vits14_reg4_pretrain.pth"
    with torch.no_grad():
        DinoDiscV2.forward

        # Print initial CUDA memory usage
        if torch.cuda.is_available():
            print(
                f"Initial CUDA memory: {torch.cuda.memory_allocated() / 1e6:.2f}MB allocated, {torch.cuda.memory_reserved() / 1e6:.2f}MB reserved"
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dd = DinoDiscV2(
            ks=ks,
            device=device,
            dino_ckpt=ckpt,
            norm_type=norm_type,
            norm_eps=norm_eps,
            input_channels=8,
            key_depths=key_layers,
        ).to(device)

        params = dd.parameters()

        # Print CUDA memory after model creation
        if torch.cuda.is_available():
            print(
                f"After model creation: {torch.cuda.memory_allocated() / 1e6:.2f}MB allocated, {torch.cuda.memory_reserved() / 1e6:.2f}MB reserved"
            )

        # dd.eval()
        # dd.heads.load_state_dict(heads.state_dict())
        print(f"{sum(p.numel() for p in dd.parameters() if p.requires_grad) / 1e6:.2f}M parameters")

        inp = torch.linspace(-2, 2, 2 * 8 * 512 * 512).reshape(2, 8, 512, 512).to(device)

        # Print CUDA memory after input creation
        if torch.cuda.is_available():
            print(
                f"After input creation: {torch.cuda.memory_allocated() / 1e6:.2f}MB allocated, {torch.cuda.memory_reserved() / 1e6:.2f}MB reserved"
            )

        cond = torch.rand(2, 64).to(device)

        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        start_time.record()
        mids = dd(inp)
        end_time.record()

        torch.cuda.synchronize()
        print(f"Forward pass time: {start_time.elapsed_time(end_time):.2f}ms")

        # Print CUDA memory after forward pass
        if torch.cuda.is_available():
            print(
                f"After forward pass: {torch.cuda.memory_allocated() / 1e6:.2f}MB allocated, {torch.cuda.memory_reserved() / 1e6:.2f}MB reserved"
            )

        print(mids.shape)

        # mid_ls = dd.dino_proxy[0](inp)
        # means = [round(m.mean().item(), 3) for m in mid_ls]
        # stds = [round(m.std().item(), 3) for m in mid_ls]
        # print(f"mean: {means}")
        # print(f"std: {stds}")

        # o = dd(inp)
        # print(f"o: {o.abs().mean().item():.9f}, {o.abs().std().item():.9f}")

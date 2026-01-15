from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from typing import cast

from src.stage1.cosmos.cosmos_tokenizer import ContinuousImageTokenizer


def _match_dim(x: Tensor) -> Tensor:
    if x.numel() > 1 and x.ndim == 1:
        x = x[None].view(-1, -1, 1, 1)
    return x


@dataclass(frozen=True)
class CosmosRSVAEConfig:
    model_path: str
    dtype: str = "bf16"


class CosmosRSVAE(nn.Module):
    """
    Cosmos_RS VAE wrapper (encode/decode) consistent with
    `src/stage2/generative/Sana/diffusion/model/builder.py` cosmos_RS branch.
    """

    latent_channels: int = 16
    spatial_compression: int = 8

    def __init__(
        self,
        *,
        model_path: str,
        dtype: str = "bf16",
        tokenizer_overrides: dict[str, Any] | None = None,
        scaling_factor: list[float] | None = None,
        shift_factor: list[float] | None = None,
    ) -> None:
        super().__init__()
        if dtype not in {"bf16", "fp32"}:
            raise ValueError(f"Unsupported {dtype=}, expected 'bf16' or 'fp32'.")
        self._dtype = torch.bfloat16 if dtype == "bf16" else torch.float32

        model_cfg = dict(
            attn_resolutions=[32],
            channels=128,
            channels_mult=[2, 4, 4],
            dropout=0.0,
            in_channels=512,
            spatial_compression=self.spatial_compression,
            num_res_blocks=2,
            out_channels=512,
            resolution=1024,
            z_channels=256,
            latent_channels=self.latent_channels,
            act_checkpoint=False,
            norm_type="rmsnorm2d",
            act_type="silu",
            block_name="res_block",
            use_residual_factor=False,
            patch_method="haar",
            patch_size=1,
            attn_type="none",
            padding_mode="reflect",
            adaptive_mode="interp",
        )
        if tokenizer_overrides:
            model_cfg.update(tokenizer_overrides)

        self.ae = ContinuousImageTokenizer.create_model(
            model=dict(**model_cfg),
            uni_path=model_path,
            loading_type="pretrained",
            qunatizer_type=None,
            hook_for_repa=False,
            use_repa_loss=False,
            use_vf_loss=False,
            vf_on_z_or_module="z",
            z_factor=1,
        )

        sf = scaling_factor or [
            0.470703125,
            0.95703125,
            0.63671875,
            0.455078125,
            0.74609375,
            0.53515625,
            0.8359375,
            0.671875,
            0.62890625,
            0.375,
            0.51171875,
            0.69921875,
            0.447265625,
            0.66015625,
            0.65234375,
            0.53515625,
        ]
        sh = shift_factor or [
            -1.2734375,
            0.193359375,
            -1.1171875,
            -1.0859375,
            -1.78125,
            -0.52734375,
            0.3984375,
            -0.5,
            -0.482421875,
            -0.09375,
            0.1689453125,
            -0.38671875,
            -0.8046875,
            0.49609375,
            -0.62109375,
            -0.2578125,
        ]

        self.register_buffer("scaling_factor", torch.tensor(sf).view(1, self.latent_channels, 1, 1), persistent=False)
        self.register_buffer("shift_factor", torch.tensor(sh).view(1, self.latent_channels, 1, 1), persistent=False)

    @torch.no_grad()
    def encode(self, images: Tensor) -> Tensor:
        scaling_factor_buf = cast(Tensor, self.scaling_factor)
        shift_factor_buf = cast(Tensor, self.shift_factor)

        images = images.to(dtype=self._dtype, device=scaling_factor_buf.device)
        scaling_factor = _match_dim(scaling_factor_buf.to(images))
        shift_factor = _match_dim(shift_factor_buf.to(images))

        with torch.autocast(
            device_type=str(images.device), dtype=torch.bfloat16 if self._dtype == torch.bfloat16 else torch.float32
        ):
            z = self.ae.encode(images)

        if isinstance(z, tuple):
            z = z[0]
        elif isinstance(z, dict):
            z = z["latent"]
        else:
            raise TypeError(f"Unexpected encode output type: {type(z)}")

        z = z.sub(shift_factor.to(z)).div(scaling_factor.to(z))
        return z

    @torch.no_grad()
    def decode(self, latent: Tensor, *, input_shape: torch.Size | int) -> Tensor:
        scaling_factor_buf = cast(Tensor, self.scaling_factor)
        shift_factor_buf = cast(Tensor, self.shift_factor)

        latent = latent.to(dtype=self._dtype, device=scaling_factor_buf.device)
        scaling_factor = _match_dim(scaling_factor_buf.to(latent))
        shift_factor = _match_dim(shift_factor_buf.to(latent))

        latent = latent.mul(scaling_factor.to(latent)).add(shift_factor.to(latent))
        with torch.autocast(
            device_type=str(latent.device), dtype=torch.bfloat16 if self._dtype == torch.bfloat16 else torch.float32
        ):
            decoded = self.ae.decode(latent, input_shape)

        if isinstance(decoded, tuple):
            decoded = decoded[0]
        elif isinstance(decoded, dict):
            decoded = decoded["recon"]
        elif torch.is_tensor(decoded):
            pass
        else:
            raise TypeError(f"Unexpected decode output type: {type(decoded)}")

        return decoded

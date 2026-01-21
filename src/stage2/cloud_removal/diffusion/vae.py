from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from typing import cast
from diffusers import AutoencoderKL
from loguru import logger
from peft import PeftModel

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
        lora_path: str | None = None,
        dtype: str = "bf16",
        tokenizer_overrides: dict[str, Any] | None = None,
        scaling_factor: list[float] | float | None = None,
        shift_factor: list[float] | float | None = None,
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
        logger.success("Load base VAE from {}".format(model_path))

        # lora load
        if lora_path is not None:
            peft_model = PeftModel.from_pretrained(self.ae, model_id=lora_path, ignore_mismatched_sizes=True)
            peft_model.merge_adapter()
            self.ae = peft_model
            self.ae.eval()
            logger.success("Load LoRA from {} and fused into base model".format(lora_path))

        if scaling_factor is None or shift_factor is None:
            sf, sh = self._get_scale_shift("ae")
        else:
            sf = scaling_factor
            sh = shift_factor
        sf = torch.tensor(sf)
        sh = torch.tensor(sh)
        if sf.numel() > 1:
            sf = sf.view(1, self.latent_channels, 1, 1)
        if sh.numel() > 1:
            sh = sh.view(1, self.latent_channels, 1, 1)

        self.register_buffer("scaling_factor", sf, persistent=False)
        self.register_buffer("shift_factor", sh, persistent=False)

        logger.info(f"Use scaling factor: {sf}")
        logger.info(f"Use shift factor: {sh}")

    @torch.no_grad()
    def encode(self, images: Tensor, no_std: bool = False) -> Tensor:
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
        if not no_std:
            z = z.sub(shift_factor.to(z)).div(scaling_factor.to(z))
        return z

    @torch.no_grad()
    def decode(self, latent: Tensor, *, input_shape: torch.Size | int, no_std: bool = False) -> Tensor:
        return self._decode_impl(latent, input_shape=input_shape, no_std=no_std)

    def decode_with_grad(self, latent: Tensor, *, input_shape: torch.Size | int, no_std: bool = False) -> Tensor:
        """Decode latent with gradient support.

        This is used for pixel-space supervision: gradients should flow to the latent
        (and thus to the cloud-removal model), while VAE parameters remain frozen.
        """
        return self._decode_impl(latent, input_shape=input_shape, no_std=no_std)

    def _decode_impl(self, latent: Tensor, *, input_shape: torch.Size | int, no_std: bool) -> Tensor:
        scaling_factor_buf = cast(Tensor, self.scaling_factor)
        shift_factor_buf = cast(Tensor, self.shift_factor)

        latent = latent.to(dtype=self._dtype, device=scaling_factor_buf.device)
        scaling_factor = _match_dim(scaling_factor_buf.to(latent))
        shift_factor = _match_dim(shift_factor_buf.to(latent))

        if not no_std:
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

    def _get_scale_shift(self, ae_type: str = "ae"):
        # ae and lcr reg
        if ae_type == "lcr":
            shift = [
                -1.4291600526869297,
                -0.4082975222915411,
                -1.0497894948720932,
                -1.4701983284950257,
                -1.7765559741854668,
                -0.8735650272667408,
                1.1257692495174707,
                0.039055027384310964,
                -0.7113604667782784,
                -0.13758214071393013,
                0.6148364602774382,
                -0.6186087974905968,
                -0.7121491965651512,
                1.2216752705536782,
                -1.3880603381455876,
                -0.45283570379018784,
            ]
            scale = [
                0.42588172074344655,
                0.8176063984445395,
                0.61324094397192,
                0.4221144967798025,
                0.7483219732102587,
                0.458599659957132,
                0.7212014900244281,
                0.5361430005310517,
                0.45571632195680367,
                0.7655523709791279,
                0.4172092222154172,
                0.5363149351470319,
                0.7398311229856277,
                0.691116324846224,
                0.7972184947063319,
                0.8416582425637736,
            ]
        elif ae_type == "ae":
            shift = [
                -0.3395568211376667,
                -0.22520461447536946,
                0.0372724512219429,
                -0.10425473003648222,
                -0.12993480741977692,
                0.15024892359972,
                0.6248297291994095,
                0.1264730566367507,
                -0.24228530898690223,
                0.09681867036037148,
                0.11547808751463891,
                -0.15777894280850888,
                -0.09696531526744366,
                0.25315030217170714,
                -0.21809950307011605,
                -0.16787929370999335,
            ]
            scale = [
                0.11568219267030617,
                0.19795570742383053,
                0.1653055727663197,
                0.10820484436353721,
                0.1673998085487709,
                0.07892267036023079,
                0.23162721283100093,
                0.1229123996528739,
                0.10866767695586152,
                0.18754874772861008,
                0.0867098794162515,
                0.17479499629405454,
                0.1599094335227425,
                0.15633937637654485,
                0.20044135354362982,
                0.15534570714645543,
            ]
        else:
            raise

        return scale, shift


@dataclass(frozen=True)
class FluxVAEConfig:
    model_path: str = "black-forest-labs/FLUX.1-dev"
    dtype: str = "bf16"


class FluxVAE(nn.Module):
    """
    Flux VAE wrapper with the same encode/decode API as CosmosRSVAE.
    """

    def __init__(self, *, model_path: str, dtype: str = "bf16") -> None:
        super().__init__()
        if dtype not in {"bf16", "fp32"}:
            raise ValueError(f"Unsupported {dtype=}, expected 'bf16' or 'fp32'.")
        self._dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
        self.vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae").to(self._dtype).eval()
        logger.success("Load Flux VAE from {}".format(model_path))

    def _get_factors(self, device: torch.device) -> tuple[Tensor, Tensor]:
        scaling = getattr(self.vae.config, "scaling_factor", 1.0)
        shift = getattr(self.vae.config, "shift_factor", 0.0)
        scaling_t = torch.as_tensor(scaling, device=device, dtype=self._dtype)
        shift_t = torch.as_tensor(shift, device=device, dtype=self._dtype)
        return _match_dim(scaling_t), _match_dim(shift_t)

    @torch.no_grad()
    def encode(self, images: Tensor, no_std: bool = False) -> Tensor:
        images = images.to(dtype=self._dtype, device=self.vae.device)
        posterior = self.vae.encode(images).latent_dist
        z = posterior.mode()
        if no_std:
            return z
        scaling, shift = self._get_factors(z.device)
        return z.sub(shift.to(z)).div(scaling.to(z))

    @torch.no_grad()
    def decode(self, latent: Tensor, *, input_shape: torch.Size | int, no_std: bool = False) -> Tensor:
        _ = input_shape
        latent = latent.to(dtype=self._dtype, device=self.vae.device)
        if not no_std:
            scaling, shift = self._get_factors(latent.device)
            latent = latent.mul(scaling.to(latent)).add(shift.to(latent))
        decoded = self.vae.decode(latent)
        if hasattr(decoded, "sample"):
            return decoded.sample
        if torch.is_tensor(decoded):
            return decoded
        if isinstance(decoded, tuple):
            return decoded[0]
        raise TypeError(f"Unexpected decode output type: {type(decoded)}")


TEST = True

if TEST:

    def _init_channel_stats(channel: int, *, device: torch.device) -> dict[str, Tensor]:
        return {
            "sum": torch.zeros(channel, dtype=torch.float64, device=device),
            "sumsq": torch.zeros(channel, dtype=torch.float64, device=device),
            "count": torch.zeros(1, dtype=torch.float64, device=device),
        }

    def _update_channel_stats(stats: dict[str, Tensor], data: Tensor) -> None:
        data = data.to(dtype=torch.float64)
        stats["sum"] += data.sum(dim=(0, 2, 3))
        stats["sumsq"] += (data**2).sum(dim=(0, 2, 3))
        stats["count"] += data.shape[0] * data.shape[2] * data.shape[3]

    def _finalize_channel_stats(stats: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        count = stats["count"].item()
        mean = stats["sum"] / count
        var = stats["sumsq"] / count - mean**2
        std = torch.sqrt(torch.clamp(var, min=0.0))
        return mean, std

    def _collect_batch_channel_stats(
        loader: Iterable[dict[str, Tensor]],
        *,
        keys: list[str],
        max_batches: int,
    ) -> dict[str, tuple[Tensor, Tensor]]:
        stats_map: dict[str, dict[str, Tensor]] = {}
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            for key in keys:
                if key not in batch:
                    raise KeyError(f"Missing key {key} in batch.")
                data = batch[key]
                if not torch.is_tensor(data):
                    raise TypeError(f"Batch key {key} must be a tensor, got {type(data)}.")
                if data.ndim != 4:
                    raise ValueError(f"Batch key {key} must be 4D tensor, got {data.shape}.")
                if key not in stats_map:
                    stats_map[key] = _init_channel_stats(data.shape[1], device=data.device)
                _update_channel_stats(stats_map[key], data)
        return {key: _finalize_channel_stats(stats) for key, stats in stats_map.items()}

    def _collect_latent_channel_stats(
        vae: CosmosRSVAE, loader: Iterable[dict[str, Tensor]], *, max_batches: int, device="cuda"
    ) -> dict[str, tuple[Tensor, Tensor]]:
        from tqdm import tqdm

        stats_map: dict[str, dict[str, Tensor]] = {}
        keys = ["img_latent", "gt_latent"]

        for batch_idx, batch in (tbar := tqdm(enumerate(loader), total=max_batches)):
            if batch_idx >= max_batches:
                break
            if "img" not in batch or "gt" not in batch:
                raise KeyError("Missing img/gt in batch.")
            img_latent = vae.encode(batch["img"].to(device))
            gt_latent = vae.encode(batch["gt"].to(device))

            if batch_idx == 0:
                from torchmetrics.functional.image import peak_signal_noise_ratio

                gt_recon = vae.decode(gt_latent, input_shape=batch["gt"].shape)
                psnr = peak_signal_noise_ratio((batch["gt"].to(device) + 1) / 2, (gt_recon + 1) / 2, data_range=1.0)
                print(f"PSNR: {psnr}")

            for key, data in {"img_latent": img_latent, "gt_latent": gt_latent}.items():
                if data.ndim != 4:
                    raise ValueError(f"Latent {key} must be 4D tensor, got {data.shape}.")
                if key not in stats_map:
                    stats_map[key] = _init_channel_stats(data.shape[1], device=data.device)
                _update_channel_stats(stats_map[key], data)
        return {key: _finalize_channel_stats(stats) for key, stats in stats_map.items()}

    def test_cosmos_cuhk_cr_mean_std(
        *,
        vae_config_path: str = "scripts/configs/cloud_removal/vae/cosmos_rs.yaml",
        max_batches: int = 100,
        batch_size: int = 4,
    ):
        import litdata as ld
        from omegaconf import OmegaConf

        from src.stage2.cloud_removal.data.CUHK_CR import CUHK_CR_StreamingDataset

        vae_cfg = OmegaConf.load(vae_config_path)
        cfg_container = OmegaConf.to_container(vae_cfg, resolve=True)
        if not isinstance(cfg_container, dict):
            raise TypeError("VAE config must be a mapping.")
        vae = CosmosRSVAE(
            model_path=str(cfg_container["model_path"]),
            lora_path=str(cfg_container["lora_path"]) if "lora_path" in cfg_container else None,
            dtype=str(cfg_container.get("dtype", "bf16")),
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vae = vae.to(device)

        input_dir = "data/Downstreams/CUHK-CR/litdata_out"

        for name in ["cr1", "cr2"]:
            ds = CUHK_CR_StreamingDataset(
                input_dir=input_dir,
                name=name,  # type: ignore[arg-type]
                split="train",
                rgb_nir_aug_p=0.0,
                to_neg_1_1=True,
            )
            dl = ld.StreamingDataLoader(ds, batch_size=batch_size, num_workers=0)  # type: ignore[invalid-argument-type]
            stats = _collect_latent_channel_stats(vae, dl, max_batches=max_batches, device=device)
            for key, (mean, std) in stats.items():
                print(f"{name}/{key} mean: {mean.tolist()}")
                print(f"{name}/{key} std: {std.tolist()}")

    def test_vae_recon_quality():
        import litdata as ld
        from omegaconf import OmegaConf
        from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure

        from src.stage2.cloud_removal.data.CUHK_CR import CUHK_CR_StreamingDataset

        vae_config_path: str = "scripts/configs/cloud_removal/vae/cosmos_rs.yaml"

        vae_cfg = OmegaConf.load(vae_config_path)
        cfg_container = OmegaConf.to_container(vae_cfg, resolve=True)
        if not isinstance(cfg_container, dict):
            raise TypeError("VAE config must be a mapping.")
        vae = CosmosRSVAE(
            model_path=str(cfg_container["model_path"]),
            lora_path=str(cfg_container["lora_path"]) if "lora_path" in cfg_container else None,
            dtype=str(cfg_container.get("dtype", "bf16")),
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vae = vae.to(device)

        input_dir = "data/Downstreams/CUHK-CR/litdata_out"

        for name in ["cr1", "cr2"]:
            ds = CUHK_CR_StreamingDataset(
                input_dir=input_dir,
                name=name,  # type: ignore[arg-type]
                split="train",
                rgb_nir_aug_p=0.0,
                to_neg_1_1=True,
            )
            dl = ld.StreamingDataLoader(ds, batch_size=1, num_workers=0)  # type: ignore[invalid-argument-type]

            for i, sample in enumerate(dl):
                img, gt = sample["img"].to(device), sample["gt"].to(device)
                img_latent = vae.encode(img)
                gt_latent = vae.encode(gt)

                img_recon = vae.decode(img_latent, input_shape=img.shape)
                gt_recon = vae.decode(gt_latent, input_shape=gt.shape)

                # psnr and ssim
                psnr = peak_signal_noise_ratio((img.to(device) + 1) / 2, (img_recon + 1) / 2, data_range=1.0)
                ssim = structural_similarity_index_measure(
                    (img.to(device) + 1) / 2, (img_recon + 1) / 2, data_range=1.0
                )

                print(f"img PSNR: {psnr}, SSIM: {ssim} | min: {img_latent.min()}, max: {img_latent.max()}")

                psnr = peak_signal_noise_ratio((gt.to(device) + 1) / 2, (gt_recon + 1) / 2, data_range=1.0)
                ssim = structural_similarity_index_measure((gt.to(device) + 1) / 2, (gt_recon + 1) / 2, data_range=1.0)

                print(f"gt PSNR: {psnr}, SSIM: {ssim} | min: {img_latent.min()}, max: {img_latent.max()}")
                print("-" * 20)

                # plot
                name = "tmp/recon_quality_test_{i}.webp".format(i=i)

                import torchvision.utils as vutils

                img, gt, img_recon, gt_recon = map(lambda x: x[:, :3], [img, gt, img_recon, gt_recon])
                vutils.save_image(
                    torch.cat([img, img_recon, gt, gt_recon], dim=0),
                    name,
                    nrow=2,
                    padding=2,
                    normalize=False,
                    scale_each=True,
                    value_range=(-1, 1),
                )


if __name__ == "__main__":
    if TEST:
        # test_cosmos_cuhk_cr_mean_std()
        test_vae_recon_quality()

from pathlib import Path
from typing import Any

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.stage1.utilities.losses.repa import PhiSMultipleTeacherDistillLoss
from src.utilities.config_utils import register_new_resolvers


def _default_cfg() -> DictConfig:
    cfg_str = """
loader:
  _target_: src.data.litdata_hyperloader.create_hyper_image_litdata_flatten_paths_loader
  paths:
    rgb_path:
      3bands_512:
        - ["data2/RemoteSAM270k/LitData_hyper_images"]
        - resize_before_transform: 512
          force_to_rgb: true
    multispectral_paths:
      4bands_512:
        - [
            "data2/Multispectral-FMow-full/LitData_hyper_images_4bands",
            "data/Gaofen1/LitData_hyper_images",
            "data/MDAS-Optical/LitData_hyper_images",
          ]
        - resize_before_transform: 512
      8bands_512:
        - [
            "data2/Multispectral-FMow-full/LitData_hyper_images_8bands",
            "data/Multispectral-Spacenet-series/LitData_hyper_images_8bands",
          ]
        - resize_before_transform: 512
    hyperspectral_paths:
      50bands_512:
        - ["data/Houston/LitData_hyper_images"]
        - resize_before_transform: 512
      202bands_512:
        - [
            "data/hyspecnet11k/LitData_hyper_images",
            "data2/HyperspectralEarth/LitData_hyper_images",
          ]
        - resize_before_transform: 512
      368bands_512:
        - ["data/MDAS-HySpex/LitData_hyper_images"]
        - resize_before_transform: 512
  macro_sampled_batch_size: {512: 6}
  loader_kwargs:
    num_workers: 8
    pin_memory: true
    drop_last: true
    persistent_workers: true
    batch_size: 6
    prefetch_factor: null
    shuffle: false
  weights: [0.18, 0.20, 0.20, 0.12, 0.18, 0.12]

phis_loss_options:
  c_dim_first: true
  rgb_channels: mean
  img_resize: 512
  repa_img_size: 512
  feature_resample_type: match_student
  phi_loss_type: mse
  phi_cache_path: runs/phi_s_cache/phi_s_buffer.pt
  phi_cache_required: false
  phi_cache_broadcast: false
  phi_cache_load_on_init: false
  teacher_configs:
    dinov3:
      repa_model_type: dinov3
      repa_model_name: dinov3_vitl16
      dino_pretrained_on: satellite
      dino_repo_path: src/stage1/utilities/losses/dinov3
      rgb_channels: mean
      img_resize: 512
      repa_img_size: 512
      repa_model_load_path: src/stage1/utilities/losses/dinov3/weights
    siglip2:
      repa_model_type: siglip2
      repa_model_name: google/siglip2-large-patch16-512
      rgb_channels: mean
      img_resize: 512
      repa_img_size: 512
    pe:
      repa_model_type: pe
      repa_model_name: PE-Spatial-L14-448
      repa_model_load_path: /Data/ZiHanCao/checkpoints/perception_models/PE-Spatial-L14-448.pt
      rgb_channels: mean
      img_resize: 512
      repa_img_size: 512
  teacher_dims:
    dinov3: [1024, 1024, 1024, 1024]
    siglip2: [1024, 1024, 1024, 1024]
    pe: [1024, 1024, 1024, 1024]
  teacher_weights:
    dinov3: 0.34
    siglip2: 0.33
    pe: 0.33

runtime:
  seed: 2025
  device: cuda
  max_batches: 2000
  log_every: 10
  output_path: runs/phi_s_cache/phi_s_buffer.pt
"""
    cfg = OmegaConf.create(cfg_str)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected DictConfig from OmegaConf.create, got {type(cfg)}")
    return cfg


def _get_img_from_batch(batch: Any) -> torch.Tensor | None:
    if batch is None:
        return None
    if isinstance(batch, dict):
        img = batch.get("img")
        if torch.is_tensor(img):
            return img
    return None


def _build_loader(cfg: DictConfig) -> Any:
    loader_or_pair = hydra.utils.instantiate(cfg.loader)
    if isinstance(loader_or_pair, tuple):
        if len(loader_or_pair) != 2:
            raise ValueError(f"Expected (dataset, loader) tuple, got len={len(loader_or_pair)}")
        _, loader = loader_or_pair
        return loader
    return loader_or_pair


def _get_phis_options(cfg: DictConfig) -> dict[str, Any]:
    phis_options = OmegaConf.to_container(cfg.phis_loss_options, resolve=True)
    if not isinstance(phis_options, dict):
        raise TypeError(f"phis_loss_options must resolve to dict, got {type(phis_options)}")
    normalized_options = {str(k): v for k, v in phis_options.items()}
    # Initialization script should never require pre-existing phi cache.
    normalized_options["phi_cache_load_on_init"] = False
    normalized_options["phi_cache_required"] = False
    return normalized_options


def _resolve_output_path(cfg: DictConfig) -> Path:
    output = cfg.runtime.output_path
    if output is None or str(output).strip() == "":
        output = cfg.phis_loss_options.phi_cache_path
    if output is None or str(output).strip() == "":
        raise ValueError("Missing output path: set `runtime.output_path` or `phis_loss_options.phi_cache_path`.")
    return Path(str(output)).expanduser().resolve()


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        logger.warning("[PhiS init]: CUDA not available, fallback to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


@logger.catch()
def main() -> None:
    register_new_resolvers()
    cfg = _default_cfg()

    seed = int(cfg.runtime.seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = _resolve_device(str(cfg.runtime.device))
    max_batches = int(cfg.runtime.max_batches)
    log_every = max(int(cfg.runtime.log_every), 1)
    output_path = _resolve_output_path(cfg)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    loader = _build_loader(cfg)
    phis_options = _get_phis_options(cfg)
    phis_loss = PhiSMultipleTeacherDistillLoss(**phis_options).to(device)
    phis_loss.move_teachers_to(device)
    phis_loss.reset_phi_stats()

    used_batches = 0
    used_samples = 0
    for batch in loader:
        if used_batches >= max_batches:
            break
        img = _get_img_from_batch(batch)
        if img is None:
            continue
        img = img.to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.no_grad() and torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            phis_loss.update_phi_stats_from_image(img)
        used_batches += 1
        used_samples += int(img.shape[0])
        if used_batches % log_every == 0:
            logger.info(f"[PhiS init]: processed batches={used_batches}, samples={used_samples}")

    if used_batches == 0:
        raise RuntimeError("No valid image batches were processed. Check loader output and image key.")

    phis_loss.finalize_phi_from_stats(distributed=False)
    phis_loss.save_phi_to_cache(output_path)
    logger.success(
        f"[PhiS init]: saved Phi-S buffer to {output_path}, total batches={used_batches}, samples={used_samples}"
    )


if __name__ == "__main__":
    main()

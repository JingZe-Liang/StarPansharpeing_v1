from dataclasses import dataclass
from pathlib import Path
from typing import Any
from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import umap

from scripts.infer.model_embedding_umap_vis import (
    _encode_with_optional_intermediate_features,
    _extract_latent_tensor,
    _instantiate_tokenizer,
    _pool_latent_tensor,
    _resolve_amp_dtype,
    _resolve_device,
)


CFG_YAML = """
dataset:
  root: data/Downstreams/SceneClassification/RESISC45/NWPU-RESISC45
  samples_per_class: 100
  image_size: 256
  to_neg_1_1: true
  seed: 2026
  extensions: [".jpg", ".jpeg", ".png", ".bmp"]

tokenizer:
  config_path: scripts/configs/tokenizer_gan/tokenizer/cosmos_hybrid_one_big_transformer.yaml
  checkpoint_path: runs/stage1_cosmos_hybrid/2025-12-21_23-52-12_hybrid_cosmos_f16c64_ijepa_pretrained_sem_no_lejepa/ema/tokenizer/model.safetensors
  latent_key: sem_z
  pool: avg

loader:
  batch_size: 16
  num_workers: 4
  pin_memory: true

runtime:
  device: cuda
  amp_dtype: bf16

umap:
  n_neighbors: 50
  min_dist: 0.3
  metric: cosine
  random_state: 2026

plot:
  output_path: tmp/resisc45_class_umap.png
  point_size: 10.0
  point_alpha: 0.8
  figsize: [14, 11]
  dpi: 220
"""


@dataclass(slots=True)
class Resisc45Sample:
    image_path: Path
    class_name: str
    class_index: int


class Resisc45ImageDataset(Dataset[dict[str, Any]]):
    def __init__(self, samples: list[Resisc45Sample], image_size: int, to_neg_1_1: bool) -> None:
        self.samples = samples
        self.image_size = image_size
        self.to_neg_1_1 = to_neg_1_1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = _load_rgb_image(sample.image_path, image_size=self.image_size, to_neg_1_1=self.to_neg_1_1)
        return {
            "image": image,
            "class_index": sample.class_index,
            "class_name": sample.class_name,
            "image_path": str(sample.image_path),
        }


def _load_cfg() -> DictConfig:
    cfg = OmegaConf.create(CFG_YAML)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"CFG_YAML must resolve to DictConfig, got {type(cfg)}")
    return cfg


def _list_class_dirs(root: Path) -> list[Path]:
    class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if len(class_dirs) == 0:
        raise FileNotFoundError(f"No class directories found under {root}")
    return class_dirs


def _collect_class_image_paths(root: Path, extensions: set[str]) -> dict[str, list[Path]]:
    class_to_paths: dict[str, list[Path]] = {}
    for class_dir in _list_class_dirs(root):
        image_paths = sorted(
            path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in extensions
        )
        if len(image_paths) == 0:
            continue
        class_to_paths[class_dir.name] = image_paths

    if len(class_to_paths) == 0:
        raise FileNotFoundError(f"No images with extensions {sorted(extensions)} found under {root}")
    return class_to_paths


def _sample_images_per_class(
    class_to_paths: dict[str, list[Path]],
    samples_per_class: int,
    seed: int,
) -> list[Resisc45Sample]:
    rng = np.random.default_rng(seed)
    samples: list[Resisc45Sample] = []
    class_names = sorted(class_to_paths)

    for class_index, class_name in enumerate(tqdm(class_names, desc="Sampling classes", unit="class", leave=False)):
        image_paths = class_to_paths[class_name]
        if len(image_paths) < samples_per_class:
            raise ValueError(
                f"Class {class_name} has only {len(image_paths)} images, but {samples_per_class} are required."
            )

        selected_indices = rng.choice(len(image_paths), size=samples_per_class, replace=False)
        selected_indices.sort()
        for selected_index in selected_indices.tolist():
            samples.append(
                Resisc45Sample(
                    image_path=image_paths[selected_index],
                    class_name=class_name,
                    class_index=class_index,
                )
            )
    return samples


def _load_rgb_image(image_path: Path, image_size: int, to_neg_1_1: bool) -> torch.Tensor:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        if image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
        image_np = np.asarray(image, dtype=np.float32) / 255.0

    image_chw = torch.from_numpy(image_np).permute(2, 0, 1)
    if to_neg_1_1:
        image_chw = image_chw * 2.0 - 1.0
    return image_chw


def _collect_embeddings(
    cfg: DictConfig,
    samples: list[Resisc45Sample],
) -> tuple[np.ndarray, list[str]]:
    device = _resolve_device(str(cfg.runtime.device))
    amp_dtype = _resolve_amp_dtype(str(cfg.runtime.amp_dtype))
    tokenizer = _instantiate_tokenizer(
        tokenizer_config_path=Path(str(cfg.tokenizer.config_path)),
        checkpoint_path=Path(str(cfg.tokenizer.checkpoint_path)),
        device=device,
    )

    dataset = Resisc45ImageDataset(
        samples=samples,
        image_size=int(cfg.dataset.image_size),
        to_neg_1_1=bool(cfg.dataset.to_neg_1_1),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.loader.batch_size),
        num_workers=int(cfg.loader.num_workers),
        pin_memory=bool(cfg.loader.pin_memory),
        shuffle=False,
        persistent_workers=False,
    )

    embedding_chunks: list[np.ndarray] = []
    labels: list[str] = []
    autocast_context = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if device.type == "cuda" and amp_dtype != torch.float32
        else nullcontext()
    )

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Embedding batches", unit="batch"):
            images = batch["image"].to(device, non_blocking=True)
            class_names = [str(name) for name in batch["class_name"]]
            if device.type == "cuda" and amp_dtype != torch.float32:
                with autocast_context:
                    encode_out = _encode_with_optional_intermediate_features(
                        tokenizer=tokenizer,
                        batch_imgs=images,
                        latent_key=str(cfg.tokenizer.latent_key),
                    )
            else:
                encode_out = _encode_with_optional_intermediate_features(
                    tokenizer=tokenizer,
                    batch_imgs=images,
                    latent_key=str(cfg.tokenizer.latent_key),
                )

            latent = _extract_latent_tensor(encode_out, latent_key=str(cfg.tokenizer.latent_key))
            pooled = _pool_latent_tensor(latent, pool=str(cfg.tokenizer.pool))
            embedding_chunks.append(pooled.detach().cpu().float().numpy())
            labels.extend(class_names)

    embeddings = np.concatenate(embedding_chunks, axis=0)
    return embeddings, labels


def _run_umap(cfg: DictConfig, embeddings: np.ndarray) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=int(cfg.umap.n_neighbors),
        min_dist=float(cfg.umap.min_dist),
        metric=str(cfg.umap.metric),
        random_state=int(cfg.umap.random_state),
    )
    return np.asarray(reducer.fit_transform(embeddings), dtype=np.float32)


def _save_plot(cfg: DictConfig, coords: np.ndarray, labels: list[str]) -> None:
    class_names = sorted(set(labels))
    cmap = plt.get_cmap("nipy_spectral", max(1, len(class_names)))
    label_to_color = {label: cmap(index) for index, label in enumerate(class_names)}

    figsize_cfg = cfg.plot.figsize
    fig, ax = plt.subplots(
        figsize=(float(figsize_cfg[0]), float(figsize_cfg[1])),
        dpi=int(cfg.plot.dpi),
    )

    for class_name in tqdm(class_names, desc="Plotting classes", unit="class", leave=False):
        indices = [index for index, label in enumerate(labels) if label == class_name]
        subset = coords[indices]
        ax.scatter(
            subset[:, 0],
            subset[:, 1],
            s=float(cfg.plot.point_size),
            alpha=float(cfg.plot.point_alpha),
            c=[label_to_color[class_name]],
            edgecolors="none",
            label=class_name,
        )
        center = subset.mean(axis=0)
        ax.text(center[0], center[1], class_name, fontsize=6, color=label_to_color[class_name], alpha=0.9)

    ax.set_title("RESISC45 Class UMAP")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(alpha=0.15)
    fig.tight_layout()

    output_path = Path(str(cfg.plot.output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cfg = _load_cfg()
    root = Path(str(cfg.dataset.root))
    extensions = {str(ext).lower() for ext in cfg.dataset.extensions}
    class_to_paths = _collect_class_image_paths(root=root, extensions=extensions)
    samples = _sample_images_per_class(
        class_to_paths=class_to_paths,
        samples_per_class=int(cfg.dataset.samples_per_class),
        seed=int(cfg.dataset.seed),
    )
    embeddings, labels = _collect_embeddings(cfg=cfg, samples=samples)
    coords = _run_umap(cfg=cfg, embeddings=embeddings)
    _save_plot(cfg=cfg, coords=coords, labels=labels)


if __name__ == "__main__":
    main()

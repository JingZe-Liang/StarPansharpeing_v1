from contextlib import nullcontext
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from tqdm import tqdm
import umap

from scripts.infer.model_embedding_umap_vis import _pool_latent_tensor
from scripts.infer.resisc45_class_umap_vis import (
    Resisc45ImageDataset,
    Resisc45Sample,
    _collect_class_image_paths,
    _sample_images_per_class,
)
from src.stage1.utilities.losses.distill.teachers.dino_adapter import DinoTeacherAdapter, load_repa_dino_v3_model


CFG_YAML = """
dataset:
  root: data/Downstreams/SceneClassification/RESISC45/NWPU-RESISC45
  samples_per_class: 100
  image_size: 256
  to_neg_1_1: true
  seed: 2026
  extensions: [".jpg", ".jpeg", ".png", ".bmp"]

dino:
  weight_path: null
  model_name: dinov3_vitl16
  pretrained_on: satellite
  compile: false
  repo_path: null
  get_interm_feats: false
  feature_index: -1
  c_dim_first: true
  img_is_neg1_1: true
  rgb_channels: [0, 1, 2]
  img_resize: dino
  use_linstretch: false
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
  output_path: tmp/resisc45_dinov3_umap.png
  point_size: 10.0
  point_alpha: 0.8
  figsize: [14, 11]
  dpi: 220
"""


def _load_cfg() -> DictConfig:
    cfg = OmegaConf.create(CFG_YAML)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"CFG_YAML must resolve to DictConfig, got {type(cfg)}")
    return cfg


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def _resolve_amp_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    return torch.float32


def _resolve_pretrained_on(value: str) -> str:
    if value not in {"satellite", "web"}:
        raise ValueError(f"Unsupported pretrained_on={value}")
    return value


def _select_adapter_feature(features: list[torch.Tensor], feature_index: int) -> torch.Tensor:
    if len(features) == 0:
        raise ValueError("DINO teacher adapter returned no features.")
    return features[feature_index]


def _collect_embeddings(
    cfg: DictConfig,
    samples: list[Resisc45Sample],
) -> tuple[np.ndarray, list[str]]:
    device = _resolve_device(str(cfg.runtime.device))
    amp_dtype = _resolve_amp_dtype(str(cfg.runtime.amp_dtype))
    pretrained_on = _resolve_pretrained_on(str(cfg.dino.pretrained_on))
    dino_model = load_repa_dino_v3_model(
        weight_path=cfg.dino.get("weight_path"),
        model_name=str(cfg.dino.model_name),
        pretrained_on=pretrained_on,  # type: ignore[arg-type]
        compile=bool(cfg.dino.compile),
        repo_path=cfg.dino.get("repo_path"),
    )
    dino_model = dino_model.to(device)
    dino_model.eval()
    dino_model.requires_grad_(False)
    dino_adapter = DinoTeacherAdapter(
        repa_encoder=dino_model,
        dino_type="torch",
        repa_model_name=str(cfg.dino.model_name),
        c_dim_first=bool(cfg.dino.c_dim_first),
        img_is_neg1_1=bool(cfg.dino.img_is_neg1_1),
        rgb_channels=list(cfg.dino.rgb_channels),
        img_resize=cfg.dino.get("img_resize"),
        pca_fn=None,
        dino_pretrained_on=pretrained_on,
    )

    dataset = Resisc45ImageDataset(
        samples=samples,
        image_size=int(cfg.dataset.image_size),
        to_neg_1_1=bool(cfg.dataset.to_neg_1_1),
    )
    loader = torch.utils.data.DataLoader(
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
        for batch in tqdm(loader, desc="DINOv3 batches", unit="batch"):
            images = batch["image"].to(device, non_blocking=True)
            class_names = [str(name) for name in batch["class_name"]]

            with autocast_context:
                features = dino_adapter.encode(
                    images,
                    get_interm_feats=bool(cfg.dino.get_interm_feats),
                    use_linstretch=bool(cfg.dino.use_linstretch),
                    detach=True,
                    repa_fixed_bs=None,
                )
                selected_feature = _select_adapter_feature(features, feature_index=int(cfg.dino.feature_index))
                embeddings = _pool_latent_tensor(selected_feature, pool=str(cfg.dino.pool))

            embedding_chunks.append(embeddings.detach().cpu().float().numpy())
            labels.extend(class_names)

    return np.concatenate(embedding_chunks, axis=0), labels


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

    title = (
        f"RESISC45 DINOv3 UMAP "
        f"(interm={bool(cfg.dino.get_interm_feats)}, idx={int(cfg.dino.feature_index)}, pool={cfg.dino.pool})"
    )
    ax.set_title(title)
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

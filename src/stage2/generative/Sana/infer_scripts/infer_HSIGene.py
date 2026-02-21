from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

SANA_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(SANA_ROOT) not in sys.path:
    sys.path.insert(0, str(SANA_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["DISABLE_XFORMERS"] = "1"

from src.stage2.generative.Sana.diffusion.data.builder import build_dataset
from src.stage2.generative.Sana.infer_scripts.infer_SAM270k import (
    build_components,
    build_control_signal,
    choose_device,
    decode_to_image,
    encode_prompt,
    load_config,
    normalize_pair_key,
    parse_cfg_ratios,
    parse_sampler_list,
    resolve_path,
    safe_name,
    sample_latent,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sana ControlNet inference for HSIGene test set")
    parser.add_argument(
        "--config",
        type=str,
        default="src/stage2/generative/Sana/configs/sana_controlnet_config/Sana_600M_HSIGene_controlnet.yaml",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="src/stage2/generative/Sana/output_HSI/20260216_221238_Sana_RS_tokenizer_ControlNet/checkpoints/latest.pth",
        help="Override checkpoint path. Default uses this HSIGene checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="src/stage2/generative/Sana/outputs_hsigene_infer",
        help="Directory to save generated images",
    )
    parser.add_argument("--metadata-jsonl", type=str, default="", help="Default: <output-dir>/metadata.jsonl")
    parser.add_argument(
        "--test-caption-dir",
        type=str,
        default="data2/HSIGene_dataset/LitData_image_captions/test",
    )
    parser.add_argument("--test-image-dir", type=str, default="")
    parser.add_argument("--test-condition-dir", type=str, default="")
    parser.add_argument("--num-images", type=int, default=-1, help="-1 means all samples from start-index")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--default-prompt", type=str, default="")
    parser.add_argument("--negative-prompt", type=str, default="")
    parser.add_argument("--samplers", type=str, default="flow_euler")
    parser.add_argument("--cfg-ratios", type=str, default="4.5")
    parser.add_argument("--flow-euler-steps", type=int, default=50)
    parser.add_argument("--flow-dpm-steps", type=int, default=20)
    parser.add_argument("--dpm-steps", type=int, default=14)
    parser.add_argument("--save-control-preview", action="store_true")
    parser.add_argument("--save-hsi-tif", action="store_true", default=True)
    parser.add_argument("--no-save-hsi-tif", action="store_false", dest="save_hsi_tif")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dataset-resolution", type=int, default=-1, help="-1 uses config.data.image_size")
    return parser.parse_args()


def _swap_dataset_folder(source: Path, from_name: str, to_name: str) -> Path:
    parts = list(source.parts)
    if from_name in parts:
        idx = parts.index(from_name)
        parts[idx] = to_name
        return Path(*parts)
    return source.parent.parent / to_name / source.name


def infer_test_data_dirs(
    *,
    caption_dir: str,
    image_dir: str,
    condition_dir: str,
) -> tuple[str, str, str]:
    caption_path = Path(resolve_path(caption_dir))
    image_path = (
        Path(resolve_path(image_dir))
        if image_dir
        else _swap_dataset_folder(caption_path, "LitData_image_captions", "LitData_hyper_images")
    )
    condition_path = (
        Path(resolve_path(condition_dir))
        if condition_dir
        else _swap_dataset_folder(caption_path, "LitData_image_captions", "LitData_conditions")
    )
    return str(image_path), str(condition_path), str(caption_path)


def ensure_dataset_dirs(paths: tuple[str, str, str]) -> None:
    labels = ("image_dir", "condition_dir", "caption_dir")
    for label, path_str in zip(labels, paths, strict=True):
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"{label} is not a directory: {path}")


def save_control_preview(control_image: torch.Tensor, save_path: Path) -> None:
    preview = torch.clamp((control_image + 1.0) / 2.0, 0.0, 1.0)
    preview_np = (preview.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    Image.fromarray(preview_np).save(save_path)


def save_hsi_tif(decoded: torch.Tensor, save_path: Path) -> None:
    import tifffile

    hsi = decoded[0].detach().cpu().to(torch.float32).numpy()
    tifffile.imwrite(save_path, hsi)


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)

    if args.dataset_resolution > 0:
        config = replace(
            config,
            data=replace(config.data, image_size=args.dataset_resolution),
            model=replace(config.model, image_size=args.dataset_resolution),
        )

    test_dirs = infer_test_data_dirs(
        caption_dir=args.test_caption_dir,
        image_dir=args.test_image_dir,
        condition_dir=args.test_condition_dir,
    )
    ensure_dataset_dirs(test_dirs)

    device = choose_device(args.device)
    output_dir = Path(resolve_path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = Path(resolve_path(args.metadata_jsonl)) if args.metadata_jsonl else output_dir / "metadata.jsonl"
    control_preview_dir = output_dir / "control_preview"
    hsi_tif_dir = output_dir / "hsi_tif"

    dataset_cfg = asdict(config.data)
    dataset_cfg["data_dir"] = list(test_dirs)
    dataset = build_dataset(dataset_cfg, resolution=config.data.image_size, config=config)
    dataset_len = len(dataset)

    start_index = max(0, args.start_index)
    remaining = max(0, dataset_len - start_index)
    target_images = remaining if args.num_images <= 0 else min(args.num_images, remaining)

    samplers = parse_sampler_list(args.samplers)
    cfg_ratios = parse_cfg_ratios(args.cfg_ratios)
    combos = [(sampler, cfg_ratio) for sampler in samplers for cfg_ratio in cfg_ratios]

    model, vae, tokenizer, text_encoder, weight_dtype, vae_dtype, latent_size = build_components(
        config=config,
        model_path=resolve_path(args.model_path),
        device=device,
    )
    null_prompt = args.negative_prompt if args.negative_prompt else ""
    null_y, _ = encode_prompt(config, null_prompt, tokenizer, text_encoder, device, weight_dtype)

    print(
        f"HSIGene test dirs={test_dirs}, dataset_len={dataset_len}, target={target_images}, "
        f"samplers={samplers}, cfg_ratios={cfg_ratios}, image_size={config.data.image_size}, latent_size={latent_size}"
    )

    generated = 0
    cursor = start_index

    with metadata_path.open("a", encoding="utf-8") as metadata_fp:
        with tqdm(total=target_images, desc="HSIGene Inference", unit="img") as pbar:
            while generated < target_images and cursor < dataset_len:
                sample = dataset[cursor]
                cursor += 1
                if sample is None:
                    continue

                _, caption, _, data_info, _, _, dataindex_info, _ = sample
                prompt = str(caption).strip() if isinstance(caption, str) else ""
                if not prompt:
                    prompt = args.default_prompt

                control_images = data_info.get("control_image") if isinstance(data_info, dict) else None
                control_types = data_info.get("control_types") if isinstance(data_info, dict) else None
                if not isinstance(control_images, list) or len(control_images) == 0:
                    continue
                if not all(torch.is_tensor(image) for image in control_images):
                    continue

                sampler, cfg_ratio = combos[generated % len(combos)]
                output_name_key = (
                    normalize_pair_key(dataindex_info.get("key", cursor - 1))
                    if isinstance(dataindex_info, dict)
                    else str(cursor - 1)
                )
                output_name_key = safe_name(output_name_key)
                cfg_text = str(cfg_ratio).replace(".", "p")
                output_name = f"{generated:06d}_{output_name_key}_{sampler}_cfg{cfg_text}.png"
                output_path = output_dir / output_name

                if args.skip_existing and output_path.exists():
                    generated += 1
                    pbar.update(1)
                    continue

                seed = args.seed + generated
                generator = torch.Generator(device=device).manual_seed(seed)
                z = torch.randn(
                    1,
                    config.vae.vae_latent_dim,
                    latent_size,
                    latent_size,
                    device=device,
                    generator=generator,
                )

                caption_embs, emb_masks = encode_prompt(config, prompt, tokenizer, text_encoder, device, weight_dtype)
                control_signal = build_control_signal(control_images, config, vae, device, latent_size, z.dtype)

                with torch.no_grad():
                    latent = sample_latent(
                        sampler=sampler,
                        z=z,
                        model=model,
                        caption_embs=caption_embs,
                        null_y=null_y,
                        emb_masks=emb_masks,
                        control_signal=control_signal,
                        cfg_ratio=cfg_ratio,
                        flow_euler_steps=args.flow_euler_steps,
                        flow_dpm_steps=args.flow_dpm_steps,
                        dpm_steps=args.dpm_steps,
                        flow_shift=config.scheduler.flow_shift,
                        image_size=config.model.image_size,
                    )
                    image, decoded = decode_to_image(latent, config, vae, vae_dtype)

                image.save(output_path)

                if args.save_control_preview:
                    control_preview_dir.mkdir(parents=True, exist_ok=True)
                    save_control_preview(control_images[0], control_preview_dir / f"{output_path.stem}_control0.png")

                if args.save_hsi_tif and decoded.shape[1] > 3:
                    hsi_tif_dir.mkdir(parents=True, exist_ok=True)
                    save_hsi_tif(decoded, hsi_tif_dir / f"{output_path.stem}.tif")

                metadata: dict[str, Any] = {
                    "output": str(output_path),
                    "dataset_index": cursor - 1,
                    "dataset_key": output_name_key,
                    "prompt": prompt,
                    "seed": seed,
                    "sampler": sampler,
                    "cfg_ratio": cfg_ratio,
                    "control_types": control_types,
                }
                metadata_fp.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                metadata_fp.flush()

                generated += 1
                pbar.update(1)

    if generated < target_images:
        print(f"Finished with {generated} images (target {target_images}); dataset exhausted at index {cursor}.")
    else:
        print(f"Done. Generated {generated} images to {output_dir}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    """
    python src/stage2/generative/Sana/infer_scripts/infer_HSIGene.py --device cuda:0 --num-images -1
    """
    main()

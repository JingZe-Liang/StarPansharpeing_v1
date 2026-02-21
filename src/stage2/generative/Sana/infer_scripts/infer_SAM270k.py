from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import re
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import kornia
import pyrallis
import torch
from PIL import Image
from tqdm import tqdm

SANA_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(SANA_ROOT) not in sys.path:
    sys.path.insert(0, str(SANA_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep inference path aligned with training setup when xFormers is disabled.
os.environ["DISABLE_XFORMERS"] = "1"

from src.stage2.generative.Sana.diffusion import DPMS, FlowEuler

from src.stage2.generative.Sana.diffusion.data.builder import build_dataset
from src.stage2.generative.Sana.diffusion.model.builder import (
    build_model,
    get_tokenizer_and_text_encoder,
    get_vae,
    vae_decode,
    vae_encode,
)
from src.stage2.generative.Sana.diffusion.model.utils import get_weight_dtype
from src.stage2.generative.Sana.diffusion.utils.config import SanaConfig, model_init_config
from src.stage2.generative.Sana.tools.download import find_model
from src.utilities.train_utils import get_rgb_image


VALID_SAMPLERS = ("flow_euler", "flow_dpm-solver", "dpm-solver")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sana ControlNet inference for Sam270k litdata")
    parser.add_argument(
        "--config",
        type=str,
        default="src/stage2/generative/Sana/configs/sana_controlnet_config/Sana_600M_RS_tokenizer_controlnet.yaml",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="src/stage2/generative/Sana/output/20260217_005439_Sana_RS_tokenizer_ControlNet/checkpoints/latest.pth",
        help="Override checkpoint path. Default uses config.model.load_from",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="src/stage2/generative/Sana/outputs_sam270k_infer",
        help="Directory to save generated images",
    )
    parser.add_argument("--metadata-jsonl", type=str, default="", help="Default: <output-dir>/metadata.jsonl")
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--default-prompt", type=str, default="")
    parser.add_argument("--negative-prompt", type=str, default="")
    parser.add_argument(
        "--samplers", type=str, default="flow_euler", help="Comma list, e.g. flow_euler,flow_dpm-solver"
    )
    parser.add_argument("--cfg-ratios", type=str, default="4.5", help="Comma list, e.g. 3.0,4.5,6.0")
    parser.add_argument("--flow-euler-steps", type=int, default=50)
    parser.add_argument("--flow-dpm-steps", type=int, default=20)
    parser.add_argument("--dpm-steps", type=int, default=14)
    parser.add_argument("--save-control-preview", action="store_true")
    parser.add_argument("--save-hsi-tif", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dataset-resolution", type=int, default=-1, help="-1 uses config.data.image_size")
    return parser.parse_args()


def resolve_path(path_str: str) -> str:
    candidate = Path(path_str)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)

    roots = [Path.cwd(), PROJECT_ROOT, SANA_ROOT]
    for root in roots:
        merged = (root / path_str).resolve()
        if merged.exists():
            return str(merged)
    return path_str


def parse_sampler_list(raw: str) -> list[str]:
    samplers = [item.strip() for item in raw.split(",") if item.strip()]
    if not samplers:
        raise ValueError("`--samplers` is empty")
    normalized: list[str] = []
    alias = {
        "flow_euler": "flow_euler",
        "flow-dpm-solver": "flow_dpm-solver",
        "flow_dpm-solver": "flow_dpm-solver",
        "dpm-solver": "dpm-solver",
        "dpm_solver": "dpm-solver",
    }
    for sampler in samplers:
        key = alias.get(sampler, sampler)
        if key not in VALID_SAMPLERS:
            raise ValueError(f"Unsupported sampler `{sampler}`. Supported: {VALID_SAMPLERS}")
        normalized.append(key)
    return normalized


def parse_cfg_ratios(raw: str) -> list[float]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("`--cfg-ratios` is empty")
    ratios: list[float] = []
    for value in values:
        ratios.append(float(value))
    return ratios


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", name).strip("._")
    return cleaned if cleaned else "sample"


def normalize_pair_key(key: Any) -> str:
    if isinstance(key, (list, tuple)):
        return str(key[0]) if key else ""
    return str(key)


def choose_device(device_str: str) -> torch.device:
    desired = torch.device(device_str)
    if desired.type == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable, fallback to CPU")
        return torch.device("cpu")
    return desired


def load_config(config_path: str) -> SanaConfig:
    with open(config_path, encoding="utf-8") as fp:
        config = pyrallis.load(SanaConfig, fp)
    return config


def build_components(
    config: SanaConfig,
    model_path: str,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, Any, Any, torch.dtype, torch.dtype, int]:
    latent_size = config.model.image_size // config.vae.vae_downsample_rate
    model_kwargs = model_init_config(config, latent_size=latent_size)
    model = build_model(
        config.model.model,
        use_fp32_attention=config.model.get("fp32_attention", False) and config.model.mixed_precision != "bf16",
        **model_kwargs,
    )

    checkpoint = find_model(model_path)
    state_dict = checkpoint.get("state_dict", checkpoint)
    for key in ["pos_embed", "base_model.pos_embed", "model.pos_embed"]:
        if key in state_dict:
            del state_dict[key]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: missing={len(missing)}, unexpected={len(unexpected)}")

    weight_dtype = get_weight_dtype(config.model.mixed_precision)
    vae_dtype = get_weight_dtype(getattr(config.vae, "weight_dtype", "float32"))

    model = model.to(device=device, dtype=weight_dtype).eval()
    model = model.requires_grad_(False)

    vae_extra = config.vae.extra if isinstance(config.vae.extra, dict) else {}
    vae = get_vae(config.vae.vae_type, config.vae.vae_pretrained, device, **vae_extra)
    vae = vae.to(device=device, dtype=vae_dtype).eval()

    text_extra = config.text_encoder.extra if isinstance(config.text_encoder.extra, dict) else {}
    tokenizer, text_encoder = get_tokenizer_and_text_encoder(
        name=config.text_encoder.text_encoder_name,
        device=str(device),
        **text_extra,
    )
    text_encoder = text_encoder.to(dtype=weight_dtype).eval()

    return model, vae, tokenizer, text_encoder, weight_dtype, vae_dtype, latent_size


def encode_prompt(
    config: SanaConfig,
    prompt: str,
    tokenizer: Any,
    text_encoder: Any,
    device: torch.device,
    weight_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if "T5" in config.text_encoder.text_encoder_name:
        tokens = tokenizer(
            prompt,
            max_length=config.text_encoder.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)
        embeds = text_encoder(tokens.input_ids, attention_mask=tokens.attention_mask)[0][:, None]
        mask = tokens.attention_mask[:, None, None]
        return embeds.to(dtype=weight_dtype), mask

    if "gemma" in config.text_encoder.text_encoder_name or "Qwen" in config.text_encoder.text_encoder_name:
        if not config.text_encoder.chi_prompt:
            prompt_all = [prompt]
            max_length_all = config.text_encoder.model_max_length
        else:
            chi_prompt_parts = [part for part in config.text_encoder.chi_prompt if isinstance(part, str)]
            chi_prompt = "\n".join(chi_prompt_parts)
            prompt_all = [chi_prompt + prompt]
            num_sys_prompt_tokens = len(tokenizer.encode(chi_prompt))
            max_length_all = num_sys_prompt_tokens + config.text_encoder.model_max_length - 2

        tokens = tokenizer(
            prompt_all,
            max_length=max_length_all,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)
        select_index = [0] + list(range(-config.text_encoder.model_max_length + 1, 0))
        embeds = text_encoder(tokens.input_ids, attention_mask=tokens.attention_mask)[0][:, None][:, :, select_index]
        mask = tokens.attention_mask[:, select_index]
        return embeds.to(dtype=weight_dtype), mask

    raise ValueError(f"Unsupported text encoder: {config.text_encoder.text_encoder_name}")


@torch.no_grad()
def random_resized_crop_control_image(
    control_image: torch.Tensor,
    cropper: kornia.augmentation.RandomResizedCrop,
    params: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], tuple[int, int]]:
    if control_image.ndim == 3:
        x = control_image.unsqueeze(0)
    elif control_image.ndim == 4:
        x = control_image
    else:
        raise ValueError(f"control_image ndim must be 3 or 4, got {control_image.ndim}")

    input_hw = (int(x.shape[-2]), int(x.shape[-1]))
    if params is None:
        x = cropper(x)
        params = cropper._params
    else:
        x = cropper(x, params=params)
    return x, params, input_hw


@torch.no_grad()
def build_control_signal(
    control_images: list[torch.Tensor],
    config: SanaConfig,
    vae: torch.nn.Module,
    device: torch.device,
    latent_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    cropper = kornia.augmentation.RandomResizedCrop(
        size=(512, 512),
        scale=(0.9, 1.0),
        ratio=(0.9, 1.2),
        resample="BILINEAR",
        same_on_batch=True,
        align_corners=False,
        p=1.0,
    )

    control_latents: list[torch.Tensor] = []
    shared_params: dict[str, torch.Tensor] | None = None
    shared_input_hw: tuple[int, int] | None = None
    for control_image in control_images:
        if shared_params is None:
            x, shared_params, shared_input_hw = random_resized_crop_control_image(control_image, cropper=cropper)
        else:
            current_hw = (
                int(control_image.shape[-2]) if control_image.ndim >= 3 else -1,
                int(control_image.shape[-1]) if control_image.ndim >= 3 else -1,
            )
            if current_hw == shared_input_hw:
                x, _, _ = random_resized_crop_control_image(control_image, cropper=cropper, params=shared_params)
            else:
                x, _, _ = random_resized_crop_control_image(control_image, cropper=cropper)

        x = x.to(device=device)
        latent = vae_encode(config.vae.vae_type, vae, x, config.vae.sample_posterior, device)
        control_latents.append(latent)

    control_signal = torch.cat(control_latents, dim=1) if len(control_latents) > 1 else control_latents[0]
    if control_signal.shape[-2:] != (latent_size, latent_size):
        print(f"[warning]: control signal encoded {control_signal.shape[-2:]} != {latent_size, latent_size}")
        control_signal = torch.nn.functional.interpolate(
            control_signal, size=(latent_size, latent_size), mode="nearest"
        )
    return control_signal.to(dtype=dtype)


@torch.no_grad()
def sample_latent(
    *,
    sampler: str,
    z: torch.Tensor,
    model: torch.nn.Module,
    caption_embs: torch.Tensor,
    null_y: torch.Tensor,
    emb_masks: torch.Tensor,
    control_signal: torch.Tensor,
    cfg_ratio: float,
    flow_euler_steps: int,
    flow_dpm_steps: int,
    dpm_steps: int,
    flow_shift: float,
    image_size: int,
) -> torch.Tensor:
    data_info = {
        "img_hw": torch.tensor([[float(image_size), float(image_size)]], dtype=torch.float32, device=z.device),
        "aspect_ratio": torch.tensor([[1.0]], dtype=torch.float32, device=z.device),
        "control_signal": control_signal,
    }
    model_kwargs = {"data_info": data_info, "mask": emb_masks}

    if sampler == "flow_euler":
        flow_solver = FlowEuler(
            model,
            condition=caption_embs,
            uncondition=null_y,
            cfg_scale=cfg_ratio,
            model_kwargs=model_kwargs,
        )
        return flow_solver.sample(z, steps=flow_euler_steps)

    if sampler == "flow_dpm-solver":
        dpm_solver = DPMS(
            model.forward_with_dpmsolver,
            condition=caption_embs,
            uncondition=null_y,
            cfg_scale=cfg_ratio,
            model_type="flow",
            model_kwargs=model_kwargs,
            schedule="FLOW",
        )
        return dpm_solver.sample(
            z,
            steps=flow_dpm_steps,
            order=2,
            skip_type="time_uniform_flow",
            method="multistep",
            flow_shift=flow_shift,
        )

    if sampler == "dpm-solver":
        dpm_solver = DPMS(
            model.forward_with_dpmsolver,
            condition=caption_embs,
            uncondition=null_y,
            cfg_scale=cfg_ratio,
            model_kwargs=model_kwargs,
        )
        return dpm_solver.sample(
            z,
            steps=dpm_steps,
            order=2,
            skip_type="time_uniform",
            method="multistep",
        )

    raise ValueError(f"Unsupported sampler: {sampler}")


def decode_to_image(
    latent: torch.Tensor,
    config: SanaConfig,
    vae: torch.nn.Module,
    vae_dtype: torch.dtype,
) -> tuple[Image.Image, torch.Tensor]:
    latent = latent.to(vae_dtype)
    samples = vae_decode(
        config.vae.vae_type,
        vae,
        latent,
        input_shape=getattr(config.vae, "input_shape", 3),
    )
    img_type = getattr(config.vae, "img_type", "rgb")
    if img_type == "rgb":
        image_np = (
            torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
        )
        return Image.fromarray(image_np), samples

    samples_01 = torch.clamp((samples + 1.0) / 2.0, 0.0, 1.0)
    vis_rgb_channels = getattr(config.train, "vis_rgb_channels", "mean")
    rgb = get_rgb_image(samples_01, vis_rgb_channels, use_linstretch=True)
    rgb = torch.clamp(rgb, 0, 1)
    rgb_np = (255.0 * rgb).to("cpu", dtype=torch.uint8).numpy()[0].transpose(1, 2, 0)
    return Image.fromarray(rgb_np), samples


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

    device = choose_device(args.device)
    output_dir = Path(resolve_path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = Path(resolve_path(args.metadata_jsonl)) if args.metadata_jsonl else output_dir / "metadata.jsonl"
    control_preview_dir = output_dir / "control_preview"
    hsi_tif_dir = output_dir / "hsi_tif"

    dataset_cfg = asdict(config.data)
    dataset = build_dataset(dataset_cfg, resolution=config.data.image_size, config=config)
    dataset_len = len(dataset)

    samplers = parse_sampler_list(args.samplers)
    cfg_ratios = parse_cfg_ratios(args.cfg_ratios)
    combos = [(sampler, cfg_ratio) for sampler in samplers for cfg_ratio in cfg_ratios]

    model_path = resolve_path(args.model_path if args.model_path else str(config.model.load_from))
    model, vae, tokenizer, text_encoder, weight_dtype, vae_dtype, latent_size = build_components(
        config, model_path, device
    )

    null_prompt = args.negative_prompt if args.negative_prompt else ""
    null_y, _ = encode_prompt(config, null_prompt, tokenizer, text_encoder, device, weight_dtype)

    print(
        f"Dataset length={dataset_len}, target={args.num_images}, samplers={samplers}, cfg_ratios={cfg_ratios}, "
        f"image_size={config.data.image_size}, latent_size={latent_size}"
    )

    generated = 0
    cursor = max(0, args.start_index)

    with metadata_path.open("a", encoding="utf-8") as metadata_fp:
        with tqdm(total=args.num_images, desc="Sana Inference", unit="img") as pbar:
            while generated < args.num_images and cursor < dataset_len:
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
                if not all(torch.is_tensor(img) for img in control_images):
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
                    preview = torch.clamp((control_images[0] + 1.0) / 2.0, 0.0, 1.0)
                    preview_np = (preview.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
                    Image.fromarray(preview_np).save(control_preview_dir / f"{output_path.stem}_control0.png")

                if args.save_hsi_tif and decoded.shape[1] > 3:
                    import tifffile

                    hsi_tif_dir.mkdir(parents=True, exist_ok=True)
                    hsi = decoded[0].detach().cpu().to(torch.float32).numpy()
                    tifffile.imwrite(hsi_tif_dir / f"{output_path.stem}.tif", hsi)

                metadata = {
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

    if generated < args.num_images:
        print(f"Finished with {generated} images (requested {args.num_images}); dataset exhausted at index {cursor}.")
    else:
        print(f"Done. Generated {generated} images to {output_dir}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()

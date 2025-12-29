import argparse
import os
import sys
import io
from tqdm import tqdm
import torch
import numpy as np
from easydict import EasyDict as edict
from litdata import StreamingDataLoader, StreamingDataset, optimize
import time
from functools import partial
import multiprocessing

# Add Sana root and Project root to path
current_file_path = os.path.abspath(__file__)
sana_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(sana_root)))

sys.path.append(sana_root)
sys.path.append(project_root)

from diffusion.model.builder import get_tokenizer_and_text_encoder

# Now we can import from src
from src.data.litdata_hyperloader import CaptionStreamingDataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_dir",
        type=str,
        default="data2/RemoteSAM270k/LitData_image_captions",
        help="Input litdata directory containing captions",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data2/RemoteSAM270k/LitData_image_captions_gemma2it_encoded",
        help="Output litdata directory",
    )
    parser.add_argument(
        "--model_path", type=str, default="/Data2/ZihanCao/Checkpoints/gemma2-2b-it", help="Path to Gemma model"
    )
    parser.add_argument("--max_length", type=int, default=300, help="Max length for tokenization")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for encoding")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for optimize")
    parser.add_argument("--compress", action="store_true", help="Use zlib compression for embeddings")
    parser.add_argument("--chi_prompt", action="store_true", help="Use CHI prompt for encoding")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    return parser.parse_args()


def encode_batch(tokenizer, text_encoder, texts, max_length, device, chi_prompt=False):
    if chi_prompt:
        chi_prompts = [
            'Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:',
            "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
            "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
            "Here are examples of how to transform or refine prompts:",
            "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
            "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
            "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
            "User Prompt: ",
        ]
        chi_prompt_str = "\n".join(chi_prompts)
        texts_input = [chi_prompt_str + t for t in texts]
        num_sys_prompt_tokens = len(tokenizer.encode(chi_prompt_str))
        max_length_all = num_sys_prompt_tokens + max_length - 2
    else:
        texts_input = texts
        max_length_all = max_length

    with torch.no_grad():
        txt_tokens = tokenizer(
            texts_input, padding="max_length", max_length=max_length_all, truncation=True, return_tensors="pt"
        ).to(device)

        select_index = [0] + list(range(-max_length + 1, 0))

        y = text_encoder(
            txt_tokens.input_ids,
            attention_mask=txt_tokens.attention_mask,
        )[0]  # [bs, L, D]

        y = y[:, select_index]  # [bs, selected_L, D]
        y_mask = txt_tokens.attention_mask[:, select_index]

        y = y[:, None]  # [bs, 1, select_L, D]

    return y, y_mask


def process_batch(batch, tokenizer, text_encoder, args):
    # batch is a list of samples (dicts) because we use optimize with batch_size
    # Sample structure from CaptionStreamingDataset: {'caption': str, ...}

    if not batch:
        return

    captions = []
    # Check integrity and extract captions
    for item in batch:
        if isinstance(item, dict):
            cap = item.get("caption")
            if isinstance(cap, dict):
                cap = cap.get("caption")  # Handle potential nesting if any
            captions.append(cap)
        else:
            # Should not happen with CaptionStreamingDataset
            captions.append(str(item))

    # Filter out invalid captions if necessary, but keep batch alignment
    # We assume dataset returns valid strings or we fail/handle empty

    # Encode
    y, y_mask = encode_batch(tokenizer, text_encoder, captions, args.max_length, args.device, args.chi_prompt)

    y = y.float().cpu().numpy()  # [bs, 1, L, D]
    y_mask = y_mask.cpu().numpy().astype(np.int16)  # [bs, L]

    bs = len(batch)
    for j in range(bs):
        sample = batch[j]
        # Ensure sample is a dict to add keys
        if not isinstance(sample, dict):
            sample = {"caption": captions[j]}

        # Add embeddings
        caption_feature = y[j]  # [1, L, D]

        if args.compress:
            compressed_buffer = io.BytesIO()
            np.savez_compressed(compressed_buffer, caption_feature=caption_feature, caption_mask=y_mask[j])
            sample["caption_feature"] = compressed_buffer.getvalue()
            sample["is_compressed"] = True
        else:
            sample["caption_feature"] = caption_feature
            sample["caption_mask"] = y_mask[j]
            sample["is_compressed"] = False

        yield sample


def _identity_fn(x):
    return x


def process_and_put_q(q, dataset, args):
    print(f"Loading Gemma model from {args.model_path}...")
    tokenizer, text_encoder = get_tokenizer_and_text_encoder(
        name="gemma-2-2b-it", device=args.device, pretrained_path=args.model_path, local_files_only=True
    )

    for d in dataset:
        # d is a dict
        d2 = [d]
        for res in process_batch(d2, tokenizer, text_encoder, args):
            q.put(res)
    q.put("ALL_DONE")  # Stop for optimize fn


def main():
    args = get_args()

    # Init output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f"Loading Gemma model from {args.model_path}...")
    tokenizer, text_encoder = get_tokenizer_and_text_encoder(
        name="gemma-2-2b-it", device=args.device, pretrained_path=args.model_path, local_files_only=True
    )

    print(f"Loading Caption Dataset from {args.src_dir}...")
    # Use CaptionStreamingDataset
    dataset = CaptionStreamingDataset(input_dir=args.src_dir, shuffle=False)

    print("Start processing...")

    print(f"Optimizing/Saving to {args.output_dir}...")

    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue(10)
    p = ctx.Process(target=process_and_put_q, args=(q, dataset, args))
    p.start()

    optimize(
        fn=_identity_fn,
        queue=q,
        output_dir=args.output_dir,
        num_workers=0,
        chunk_bytes="512Mb",
        start_method="spawn",
    )
    p.join()

    print("Done!")


if __name__ == "__main__":
    main()

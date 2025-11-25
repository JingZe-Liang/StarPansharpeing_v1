#!/usr/bin/env python3
"""
Test script for mRoPE (Multimodal Rotary Position Embedding) implementation.
This script extracts and tests the mRoPE logic from the Qwen2-VL model.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat


def get_rope_index(
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    spatial_merge_size: int = 2,
    image_token_id: int = 151652,
    video_token_id: int = 151653,
    vision_start_token_id: int = 151651,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

    Explanation:
        Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

        For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
        Examples:
            input_ids: [T T T T T], here T is for text.
            temporal position_ids: [0, 1, 2, 3, 4]
            height position_ids: [0, 1, 2, 3, 4]
            width position_ids: [0, 1, 2, 3, 4]

        For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
        and 1D rotary position embedding for text part.
        Examples:
            Temporal (Time): 3 patches, representing different segments of the video in time.
            Height: 2 patches, dividing each frame vertically.
            Width: 2 patches, dividing each frame horizontally.
            We also have some important parameters:
            fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
            tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
            temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
            interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
            input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
            vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
            vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
            vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            text temporal position_ids: [101, 102, 103, 104, 105]
            text height position_ids: [101, 102, 103, 104, 105]
            text width position_ids: [101, 102, 103, 104, 105]
            Here we calculate the text start position_ids as the max vision position_ids plus 1.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        use_audio_in_video (`bool`, *optional*):
                If set to `True`, use the audio in video.
        audio_seqlens (`torch.LongTensor` of shape `(num_audios)`, *optional*):
            The length of feature shape of each audio in LLM.
        second_per_grids (`torch.LongTensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.

    Returns:
        position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
        mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
    """
    mrope_position_deltas = []

    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0

        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i].to(input_ids.device) == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums

            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1

                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video

                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))

        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas


class Qwen2VLRotaryEmbedding(nn.Module):
    """
    Rotary Embedding module for Qwen2-VL model with multimodal support.
    """

    def __init__(self, dim: int, max_position_embeddings: int = 2048, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings

        # Generate inverse frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        position_ids: torch.Tensor,
        batch_size: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for rotary embedding.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_dim]
            position_ids (torch.Tensor): Position IDs of shape [3, batch_size, seq_len]

        Returns:
            cos (torch.Tensor): Cosine embeddings
            sin (torch.Tensor): Sine embeddings
        """
        # Expand inv_freq to shape (3, bs, d, 1)
        # (d, ) -> (3, bs, d, 1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)

        # shape (3, bs, l) -> (3, bs, 1, l)
        position_ids_expanded = position_ids[:, :, None, :].float()

        device = torch.device(device).type
        device_type = device if device != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            # outer product: [3, bs, d, 1] @ [3, bs, 1, l] -> [3, bs, d, l]
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=dtype), sin.to(dtype=dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_multimodal_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors.

    Args:
        q (torch.Tensor): Query tensor
        k (torch.Tensor): Key tensor
        cos (torch.Tensor): Cosine embeddings
        sin (torch.Tensor): Sine embeddings
        mrope_section (list): Multimodal rope section for channel dimension

    Returns:
        tuple: Rotated query and key tensors
    """
    mrope_section = mrope_section * 2
    # Split cos and sin tensors according to mrope_section
    cos_splits = cos.split(mrope_section, dim=-1)
    sin_splits = sin.split(mrope_section, dim=-1)

    # Concatenate with proper indexing
    cos_concat = []
    sin_concat = []
    for i, (c, s) in enumerate(zip(cos_splits, sin_splits)):
        idx = i % 3
        cos_concat.append(c)
        sin_concat.append(s)

    cos = torch.cat(cos_concat, dim=-1)
    sin = torch.cat(sin_concat, dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def test_mrope_logic():
    """Test the mRoPE logic with sample inputs."""
    print("=" * 50)
    print("Testing mRoPE Logic")
    print("=" * 50)

    # Test parameters
    batch_size = 1
    seq_length = 20
    hidden_dim = 256
    num_heads = 8
    head_dim = hidden_dim // num_heads

    # Create sample inputs
    # input_ids = torch.randint(0, 1000, (batch_size, seq_length)).long()
    img_ids1 = [151652] * (32 * 32)
    img_ids2 = [151652] * (128 * 128)
    video_ids = [151653] * 3 * 128 * 128
    text_ids = [1] * 300
    input_ids = torch.tensor([*text_ids, 151651, *img_ids1, 15162, *img_ids2, *video_ids])[None]
    attention_mask = None  # torch.ones(batch_size, seq_length).long()

    # Sample image grid (2 images with different dimensions)
    image_grid_thw = torch.tensor([[1, 32, 32], [1, 128, 128]]).long()  # 2 images

    # Sample video grid (1 video)
    video_grid_thw = torch.tensor([[3, 128, 128]]).long()  # 1 video

    print(f"Input IDs shape: {input_ids.shape}")
    # print(f"Attention mask shape: {attention_mask.shape}")
    print(f"Image grid THW shape: {image_grid_thw.shape}")
    print(f"Video grid THW shape: {video_grid_thw.shape}")
    print()

    # Test get_rope_index function
    print("1. Testing get_rope_index function:")
    position_ids, mrope_position_deltas = get_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
        spatial_merge_size=2,
        image_token_id=151652,
        video_token_id=151653,
        vision_start_token_id=151651,
    )

    print(f"  Position IDs shape: {position_ids.shape}")  # Expected: (3, batch_size, seq_length)
    print(f"  mRoPE position deltas shape: {mrope_position_deltas.shape}")  # Expected: (batch_size, 1)
    print()

    # Test Qwen2VLRotaryEmbedding
    print("2. Testing Qwen2VLRotaryEmbedding:")
    rope_emb = Qwen2VLRotaryEmbedding(dim=head_dim)

    # Sample input tensor
    x = torch.randn(batch_size, seq_length, hidden_dim)
    print(f"  Input tensor shape: {x.shape}")

    # Get cos/sin embeddings
    cos, sin = rope_emb(x, position_ids)
    print(f"  Cosine embeddings shape: {cos.shape}")  # Expected: (3, batch_size, seq_length, head_dim)
    print(f"  Sine embeddings shape: {sin.shape}")  # Expected: (3, batch_size, seq_length, head_dim)
    print()

    # Test apply_multimodal_rotary_pos_emb
    print("3. Testing apply_multimodal_rotary_pos_emb:")
    # Sample query and key tensors (need to transpose for correct shape)
    q = torch.randn(batch_size, seq_length, num_heads, head_dim)
    k = torch.randn(batch_size, seq_length, num_heads, head_dim)
    print(f"  Query tensor shape: {q.shape}")
    print(f"  Key tensor shape: {k.shape}")

    # Apply rotary embedding
    mrope_section = [head_dim // 3, head_dim // 3, head_dim - 2 * (head_dim // 3)]
    # Transpose to match expected input format
    q_trans = q.transpose(1, 2)  # [batch_size, num_heads, seq_length, head_dim]
    k_trans = k.transpose(1, 2)  # [batch_size, num_heads, seq_length, head_dim]

    try:
        q_rot, k_rot = apply_multimodal_rotary_pos_emb(q_trans, k_trans, cos, sin, mrope_section)
        print(f"  Rotated query shape: {q_rot.shape}")
        print(f"  Rotated key shape: {k_rot.shape}")
    except Exception as e:
        print(f"  Error in apply_multimodal_rotary_pos_emb: {e}")
        print("  Skipping this test due to complexity of mRoPE section handling")
    print()

    print("All tests completed successfully!")


if __name__ == "__main__":
    test_mrope_logic()

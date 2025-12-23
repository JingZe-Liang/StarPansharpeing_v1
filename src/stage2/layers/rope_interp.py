import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


def get_resize_crop_region_for_grid(src, tgt_size):
    th = tw = tgt_size
    h, w = src

    r = h / w

    # resize
    if r > 1:
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


def apply_rdfc_correction(
    freqs: torch.Tensor, train_res: int, target_res: int, recursive_iter: int = 2
) -> torch.Tensor:
    """
    Algorithm 1: Recursive Dominant Frequency Correction (RDFC) from UltraImage paper.
    修正主频以防止内容重复。
    """
    # 避免原地修改
    freqs_new = freqs.clone()

    # 1. 计算每个频率分量的周期 T = 2 * pi / theta
    # 注意：这里的 freqs 输入通常是 1 / (theta^(...))，即论文中的 theta_i
    # Period T_i = 2 * pi / freq_val
    periods = 2 * torch.pi / freqs_new

    # 初始化当前的重复周期 (初始认为就是训练分辨率)
    N_current = train_res

    # 论文中提到可能存在多个接近的主频 (尤其是在 Qwen-Image 中)，所以递归修正
    for _ in range(recursive_iter):
        if N_current >= target_res:
            break

        # 找到周期最接近当前 N_current 的频率分量下标 k
        # Eq (6): k = argmin |T_i - N|
        diffs = torch.abs(periods - N_current)
        k = torch.argmin(diffs)

        # 获取该主频的周期 T_k
        T_k = periods[k]

        # 检查是否满足非重复条件 Eq (7)
        # 如果目标分辨率超过了主频周期，需要拉伸该频率
        if target_res > T_k:
            # Eq (8): theta'_k = 2 * pi / H
            # 将该频率强制设为对应目标分辨率的波长
            new_freq = 2 * torch.pi / target_res
            freqs_new[k] = new_freq
            periods[k] = target_res  # 更新周期表以便后续迭代

            # 更新 N_current (模拟消除了当前层级的重复后，寻找下一个潜在重复)
            # 论文中这是一个启发式过程，通常一次修正(针对 mid-band)就足够，但为了稳健可以多次
            # 这里简单起见，我们主要修正这一个最显著的。如果需要完全匹配算法1，需更新 N_current
            # 在实践中，修正最接近 train_res 的那个频率是最关键的。

    return freqs_new


def get_1d_rotary_pos_embed_ultra(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=torch.float32,
    # === UltraImage 新增参数 ===
    train_seq_len: Optional[int] = None,  # 训练时的序列长度 (分辨率)
    infer_seq_len: Optional[int] = None,  # 当前推理的序列长度 (分辨率)
    rdfc_iter: int = 2,
):
    """
    基于 UltraImage 改进的 1D RoPE 生成函数。
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)

    theta = theta * ntk_factor

    # 生成基础频率: theta_i
    freqs = (
        1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim)) / linear_factor
    )  # [D/2]

    # === UltraImage RDFC Start ===
    if train_seq_len is not None and infer_seq_len is not None:
        # 仅当处于外推模式(infer > train)且指定了训练分辨率时启用
        if infer_seq_len > train_seq_len:
            freqs = apply_rdfc_correction(freqs, train_seq_len, infer_seq_len, recursive_iter=rdfc_iter)
    # === UltraImage RDFC End ===

    # 生成 outer product [S, D/2]
    freqs = torch.outer(pos, freqs)

    is_npu = freqs.device.type == "npu"
    if is_npu:
        freqs = freqs.float()

    if use_real and repeat_interleave_real:
        # Flux, Hunyuan, CogVideoX style: [cos, cos, sin, sin] interleaved
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()
        return freqs_cos, freqs_sin
    elif use_real:
        # Stable Audio style: [cos, ..., sin, ...] concatenated
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()
        return freqs_cos, freqs_sin
    else:
        # Lumina style: complex64
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis


def get_2d_rotary_pos_embed_ultra_from_grid(
    embed_dim,
    grid,
    use_real=False,
    train_res_hw: Optional[Tuple[int, int]] = None,  # (train_h, train_w)
):
    """
    2D RoPE Wrapper for UltraImage.
    """
    assert embed_dim % 4 == 0

    # Grid shape is typically [2, H, W] or [2, 1, H, W] depending on input
    # Flatten grid for 1D embedding generation
    h_pos = grid[0].reshape(-1)  # [H*W]
    w_pos = grid[1].reshape(-1)  # [H*W]

    # 获取当前推理的分辨率 (通过 grid 的最大值估算，或者通过 grid 形状)
    # 注意：grid 可能是归一化的或者像素坐标。如果是像素坐标：
    cur_h = int(h_pos.max().item()) + 1
    cur_w = int(w_pos.max().item()) + 1

    train_h, train_w = (None, None)
    if train_res_hw is not None:
        train_h, train_w = train_res_hw

    # H 维度 Embedding (带 RDFC)
    res_h = get_1d_rotary_pos_embed_ultra(
        embed_dim // 2, h_pos, use_real=use_real, train_seq_len=train_h, infer_seq_len=cur_h
    )

    # W 维度 Embedding (带 RDFC)
    res_w = get_1d_rotary_pos_embed_ultra(
        embed_dim // 2, w_pos, use_real=use_real, train_seq_len=train_w, infer_seq_len=cur_w
    )

    if use_real:
        cos_h, sin_h = res_h
        cos_w, sin_w = res_w
        # Concatenate H and W embeddings
        cos = torch.cat([cos_h, cos_w], dim=1)  # [H*W, D]
        sin = torch.cat([sin_h, sin_w], dim=1)  # [H*W, D]
        return cos, sin
    else:
        # Complex case
        emb = torch.cat([res_h, res_w], dim=1)
        return emb


class UltraImageAttentionProcessor:
    """
    实现了 UltraImage 的 Entropy-guided Adaptive Attention Concentration.
    """

    def __init__(self, lambda_min=1.0, lambda_max=1.3, p=2.0):
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.p = p

        # 缓存每个 layer 的 lambda 因子
        # Key: layer_name (str), Value: tensor [1, Num_Heads, 1, 1]
        self.cached_lambdas = {}

        # 状态标记
        self.is_calibration_step = False  # 是否是采样的第一步(去噪步)

    def reset(self):
        self.cached_lambdas = {}
        self.is_calibration_step = True  # 重置后，下一次调用应视为校准步

    def __call__(
        self,
        attn,  # Attention layer object (nn.Module)
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        layer_name=None,  # 需要传入唯一标识符
        scale=1.0,
    ):
        """
        替代标准的 Attention forward。
        """
        batch_size, sequence_length, _ = hidden_states.shape

        # 1. 准备 Q, K, V
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # [Batch * Heads, SeqLen, HeadDim] -> [Batch, Heads, SeqLen, HeadDim]
        # 为了方便计算 Head-wise 熵，我们需要把 Batch 和 Heads 分开
        # 假设 head_to_batch_dim 只是做了 reshape/transpose
        head_dim = query.shape[-1]
        num_heads = attn.heads

        # Reshape for entropy calculation: [B, H, N, D]
        query_view = query.view(batch_size, num_heads, -1, head_dim)
        key_view = key.view(batch_size, num_heads, -1, head_dim)

        # 2. 计算 Attention Logits
        # S = Q @ K.T / sqrt(d)
        attention_scores = torch.matmul(query_view, key_view.transpose(-1, -2)) * scale

        # 3. UltraImage Adaptive Logic
        if self.is_calibration_step:
            # === Calibration (First Step) ===
            # 计算标准概率用于熵计算
            attn_probs_temp = F.softmax(attention_scores, dim=-1)

            # 计算熵 H_alpha (Eq 10)
            # Entropy = - sum(p * log(p))
            # [B, H, N] (average over target tokens later)
            eps = 1e-10
            entropy = -torch.sum(attn_probs_temp * torch.log(attn_probs_temp + eps), dim=-1)

            # 对 Query 维度取平均得到每个 Head 的熵
            entropy = torch.mean(entropy, dim=-1)  # [B, H]
            # 对 Batch 取平均
            entropy = torch.mean(entropy, dim=0)  # [H]

            # 计算 Focus Factor (Eq 11)
            H_min = entropy.min()
            H_max = entropy.max()

            if H_max - H_min < 1e-6:
                scaling_factor = torch.zeros_like(entropy)
            else:
                scaling_factor = (H_max - entropy) / (H_max - H_min)

            # lambda = min + (max - min) * factor^p
            lambdas = self.lambda_min + (self.lambda_max - self.lambda_min) * (scaling_factor**self.p)

            # Cache it: [1, H, 1, 1] for broadcasting
            # 存入字典
            if layer_name is None:
                layer_name = f"layer_{len(self.cached_lambdas)}"

            lambda_tensor = lambdas.view(1, num_heads, 1, 1).to(query.device)
            self.cached_lambdas[layer_name] = lambda_tensor

            # 应用 lambda
            attention_scores = attention_scores * lambda_tensor

        else:
            # === Inference (Subsequent Steps) ===
            if layer_name in self.cached_lambdas:
                lambda_tensor = self.cached_lambdas[layer_name]
                attention_scores = attention_scores * lambda_tensor

        # 4. Standard Softmax & Output
        attention_probs = F.softmax(attention_scores, dim=-1)

        # Dropout if needed (usually 0 for inference)
        # attention_probs = attn.attn_drop(attention_probs)

        # Reshape back to [B*H, N, N] for matmul if needed, or keep 4D
        # View V: [B, H, N, D]
        value_view = value.view(batch_size, num_heads, -1, head_dim)

        hidden_states = torch.matmul(attention_probs, value_view)

        # Reshape back to [B, N, H*D]
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, num_heads * head_dim)

        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

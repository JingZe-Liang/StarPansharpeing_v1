import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

# ==========================================
# Part 1: Recursive Dominant Frequency Correction (RDFC) for RoPE
# ==========================================


def apply_rdfc_to_freqs(
    freqs: torch.Tensor, train_res: int, target_res: int, theta: float = 10000.0, recursive_iter: int = 1
):
    """
    实现论文 Algorithm 1: Recursive Dominant Frequency Correction.
    自动识别接近训练分辨率的主频(Dominant Frequency)，并将其周期拉伸至目标分辨率。

    Args:
        freqs: 原始频率张量 [D/2]
        train_res: 训练时的分辨率 (height 或 width)
        target_res: 推理时的目标分辨率 (height 或 width)
        theta: RoPE base
        recursive_iter: 递归查找的次数 (论文中 Qwen-Image 可能有多个主频需要修正)
    """
    # 复制一份频率，以免修改原始数据
    freqs_new = freqs.clone()

    # 计算每个频率分量对应的周期 T_i = 2 * pi / theta_i
    # theta_i = 1 / (theta ** (2i/d)) -> freqs input usually is theta_i
    periods = 2 * torch.pi / freqs_new

    # 我们需要找到周期最接近 train_res 的那个频率分量
    # 论文公式 (6): k_h = argmin |T^h_i - h|

    # 为了支持递归(处理多个接近的频率)，我们将最接近的几个都找出来
    # 在实践中，通常修正最接近的一个即可，但 Qwen 可能需要修正 k=8 和 k=9

    current_periods = periods.clone()

    # 简单的递归模拟：找到最接近的，修改它，然后如果还需要，找下一个
    # 注意：这里简化为直接找 Top-K 个最接近 train_res 的频率

    diffs = torch.abs(current_periods - train_res)

    # 获取差异最小的几个索引
    # 对于 Flux 通常 k=1 个 (k=9), Qwen 可能 k=2 个
    sorted_indices = torch.argsort(diffs)

    for i in range(recursive_iter):
        idx = sorted_indices[i]

        # 检查是否满足非重复条件 (Eq 7)，如果当前周期小于目标分辨率，则需要修正
        # 实际上论文是强制修正 dominant frequency

        dominant_period = periods[idx]
        print(f"RDFC: Detected dominant period {dominant_period:.2f} at index {idx} (Target: {train_res})")

        # 修正频率: theta'_k = 2 * pi / H (Eq 8)
        # 将该频率的周期强制设为目标分辨率长度 (或稍大以防边缘效应)
        new_freq = 2 * torch.pi / target_res

        # 应用修正
        freqs_new[idx] = new_freq

    return freqs_new


def get_2d_rotary_pos_embed_ultraimage(
    dim: int, h: int, w: int, train_h: int, train_w: int, theta: float = 10000.0, recursive_iter: int = 1
):
    """
    生成适用于 2D 图像的 RoPE，并应用 UltraImage 的 RDFC 修正。

    Args:
        dim: Head dimension
        h, w: 当前推理的目标高度和宽度
        train_h, train_w: 模型训练时的分辨率 (如 1024, 1024)
    """
    assert dim % 2 == 0
    d_half = dim // 2

    # 生成基础频率
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))  # [D/2]

    # 分别对 H 和 W 维度应用 RDFC 修正
    # 注意：通常图像 RoPE 是将 dim 分为两半，一半给 H，一半给 W；或者是两个独立的 RoPE 拼接
    # 这里假设是 Flux/Qwen 的方式：两个独立的 1D RoPE 拼接 (Axial RoPE)

    # 1. 计算 Height 方向的频率
    freqs_h = apply_rdfc_to_freqs(freqs, train_h, h, theta, recursive_iter)

    # 2. 计算 Width 方向的频率
    freqs_w = apply_rdfc_to_freqs(freqs, train_w, w, theta, recursive_iter)

    # 生成 Grid
    # grid_h: [H], grid_w: [W]
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)

    # Meshgrid
    # y: [H, W], x: [H, W]
    y, x = torch.meshgrid(grid_h, grid_w, indexing="ij")

    # Flatten: [L] where L = H*W
    y = y.flatten()
    x = x.flatten()

    # Outer product to get embeddings
    # args_h: [L, D/2], args_w: [L, D/2]
    args_h = torch.outer(y, freqs_h)
    args_w = torch.outer(x, freqs_w)

    # 组合成 Complex Cis
    # [L, D/2] (complex64)
    freqs_cis_h = torch.polar(torch.ones_like(args_h), args_h)
    freqs_cis_w = torch.polar(torch.ones_like(args_w), args_w)

    # 将 H 和 W 的编码拼接。这取决于具体模型实现。
    # Flux/Qwen 通常是将 dim 分成两部分，或者在该维度上在此拼接
    # 这里返回两个分量供模型内部拼接使用，或者拼接成 [L, D] (如果 dim 是总维度)
    # 假设返回拼接后的 complex 张量 [L, D/2] (意指两组频率交织或拼接)
    # 为简单起见，这里模仿常见实现，返回拼接后的 cis

    return torch.cat([freqs_cis_h, freqs_cis_w], dim=-1)


# ==========================================
# Part 2: Entropy-guided Adaptive Attention Concentration
# ==========================================


class UltraImageAttentionController:
    """
    控制器类，用于管理 Attention 的熵计算和 Focus Factor 缓存。
    """

    def __init__(self, lambda_min=1.0, lambda_max=1.3, p=2.0):
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.p = p
        self.cached_lambdas = {}  # Key: layer_id, Value: tensor of shape [num_heads, 1, 1]
        self.is_calibration = False

    def reset(self):
        self.cached_lambdas = {}

    def set_calibration(self, mode: bool):
        self.is_calibration = mode

    def compute_focus_factors(self, attn_probs: torch.Tensor, layer_id: str):
        """
        Stage 1: 根据第一步去噪的 Attention Map 计算熵和 lambda。
        attn_probs: [B, Num_Heads, N, N] (Full attention map)

        Note: 论文中提到对于超高分辨率，计算全量 Attention Map 显存开销巨大，
        使用了 Block-wise 的 Triton kernel。这里演示 PyTorch 逻辑。
        """
        B, H, N, _ = attn_probs.shape

        # 1. 计算熵 H_alpha (Eq 10)
        # H_alpha = - (1/N) * sum(P * log(P))
        # 为了数值稳定性，输入最好是 logits，但这里假设输入已经是 softmax 后的 probs
        # 加一个 epsilon 防止 log(0)
        eps = 1e-10
        entropy = -torch.sum(attn_probs * torch.log(attn_probs + eps), dim=-1)  # [B, H, N]
        entropy = torch.mean(entropy, dim=-1)  # Mean over queries -> [B, H]

        # 如果 batch > 1，取平均
        entropy = torch.mean(entropy, dim=0)  # [H]

        H_min = entropy.min()
        H_max = entropy.max()

        # 2. 计算 Focus Factor lambda (Eq 11)
        # 熵越小(Local pattern)，lambda 越大(Sharpen)
        # 熵越大(Global pattern)，lambda 越小(Preserve)

        if H_max - H_min < 1e-6:
            scaling = torch.zeros_like(entropy)
        else:
            scaling = (H_max - entropy) / (H_max - H_min)

        lambdas = self.lambda_min + (self.lambda_max - self.lambda_min) * (scaling**self.p)

        # Cache it: [1, H, 1, 1] for broadcasting
        self.cached_lambdas[layer_id] = lambdas.view(1, -1, 1, 1)
        return self.cached_lambdas[layer_id]


def scaled_dot_product_attention_ultraimage(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    controller: UltraImageAttentionController,
    layer_id: str,
    scale: float = None,
):
    """
    UltraImage 的 Attention 实现。
    """
    B, H, N, D = query.shape
    scale = scale or (1.0 / math.sqrt(D))

    # 1. Compute Logits
    # S = QK^T
    attn_logits = torch.matmul(query, key.transpose(-2, -1)) * scale  # [B, H, N, N]

    # 2. Apply Entropy-guided Concentration

    if controller.is_calibration:
        # Stage 1: Calibration (First Step)
        # 计算标准的 Softmax
        attn_probs = F.softmax(attn_logits, dim=-1)
        # 计算并缓存 Lambda
        focus_factors = controller.compute_focus_factors(attn_probs, layer_id)

        # 重新应用 Lambda 到 logits 并计算最终 output
        # P = softmax(lambda * S)
        attn_logits = attn_logits * focus_factors.to(attn_logits.device)
        attn_probs = F.softmax(attn_logits, dim=-1)

    else:
        # Stage 2: Inference (Subsequent Steps)
        # 直接读取缓存的 Lambda
        if layer_id in controller.cached_lambdas:
            focus_factors = controller.cached_lambdas[layer_id].to(attn_logits.device)
            # Apply scaling
            attn_logits = attn_logits * focus_factors

        attn_probs = F.softmax(attn_logits, dim=-1)

    # 3. Output
    output = torch.matmul(attn_probs, value)
    return output


# ==========================================
# Example Usage
# ==========================================

if __name__ == "__main__":
    # 模拟参数
    dim = 64  # Head dimension
    train_res = 1024
    target_res = 4096  # 4x extrapolation

    print("--- Testing RDFC (RoPE) ---")
    # 1. 获取 UltraImage 的 RoPE Cis
    rope_cis = get_2d_rotary_pos_embed_ultraimage(
        dim=dim,
        h=target_res,
        w=target_res,
        train_h=train_res,
        train_w=train_res,
        recursive_iter=1,  # 自动修正最显著的一个频率
    )
    print(f"RoPE shape: {rope_cis.shape}")  # Should be [H*W, D] (concatenated) or similar

    print("\n--- Testing Adaptive Attention ---")
    # 2. 初始化控制器
    controller = UltraImageAttentionController(lambda_min=1.0, lambda_max=1.3, p=2.0)

    # 模拟输入 [B, Heads, SeqLen, Dim]
    # 假设 SeqLen = 100 (为了跑得快，实际上是 HW)
    q = torch.randn(1, 8, 100, dim)
    k = torch.randn(1, 8, 100, dim)
    v = torch.randn(1, 8, 100, dim)

    # Step 1: Calibration Step (First denoising step)
    controller.set_calibration(True)
    out_step1 = scaled_dot_product_attention_ultraimage(q, k, v, controller, layer_id="layer_0")
    print("Calibration step done. Lambdas cached.")
    print(f"Cached lambda stats for layer_0: \n{controller.cached_lambdas['layer_0'].squeeze()}")

    # Step 2: Normal Step
    controller.set_calibration(False)
    out_step2 = scaled_dot_product_attention_ultraimage(q, k, v, controller, layer_id="layer_0")
    print("Inference step done using cached lambdas.")

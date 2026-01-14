import torch
import torch.nn as nn
from timm.layers.patch_embed import PatchEmbedInterpolator


def test_patch_embed_interpolator():
    print("=== Testing PatchEmbedInterpolator ===")

    # 1. 基础配置
    base_patch_size = (16, 16)
    in_chans = 3
    embed_dim = 32  # 用小一点的维度方便观察

    # 初始化插值器
    interpolator = PatchEmbedInterpolator(
        base_patch_size=base_patch_size, in_chans=in_chans, embed_dim=embed_dim, interpolation="bicubic", antialias=True
    )

    print(f"Initialized with base_patch_size={base_patch_size}, embed_dim={embed_dim}")

    # ==========================================
    # 场景 1: Conv2d 模式 (最常用的 ViT patch embed 方式)
    # ==========================================
    print("\n--- Scenario 1: Conv2d Mode (e.g. standard ViT Stem) ---")

    # 模拟一个标准 ViT 的 Conv2d 权重
    # Shape: [Out, In, kH, kW]
    conv_weight = torch.randn(embed_dim, in_chans, *base_patch_size)
    conv_bias = torch.zeros(embed_dim)

    # 输入图片 (B, C, H, W)
    # 假设输入 224x224，用 base patch size 16 -> 14x14 tokens
    input_img = torch.randn(1, in_chans, 224, 224)

    # Case A: 使用原始 Patch Size (应该等同于普通 Conv2d)
    out_orig = interpolator(
        patches=input_img, proj_weight=conv_weight, proj_bias=conv_bias, patch_size=(16, 16), is_linear=False
    )
    print(f"Original (16x16) output shape: {out_orig.shape}")
    # Expect: (1, 32, 14, 14) -> because 224/16 = 14

    # Case B: 动态改变 Patch Size 为 (32, 32) (FlexiViT 场景)
    # 此时权重会被下采样/重采样以适应更大的 patch (即更粗粒度的 token)
    out_new = interpolator(
        patches=input_img,
        proj_weight=conv_weight,  # 传入原始 16x16 权重
        proj_bias=conv_bias,
        patch_size=(32, 32),  # 请求 32x32
        is_linear=False,
    )
    print(f"Resized  (32x32) output shape: {out_new.shape}")
    # Expect: (1, 32, 7, 7) -> because 224/32 = 7

    # Case C: 动态改变 Patch Size 为 (8, 8) (更细粒度)
    out_fine = interpolator(
        patches=input_img, proj_weight=conv_weight, proj_bias=conv_bias, patch_size=(8, 8), is_linear=False
    )
    print(f"Resized  (8x8)   output shape: {out_fine.shape}")
    # Expect: (1, 32, 28, 28) -> because 224/8 = 28

    # ==========================================
    # 场景 2: Linear 模式 (例如 MLP Mixer 或特殊的 Patchify 预处理)
    # ==========================================
    print("\n--- Scenario 2: Linear Mode ---")

    # Linear 权重通常处理的是 flatten 后的 patch: (Ph * Pw * C) -> Embed
    # Shape: [Embed, Ph * Pw * C]
    linear_input_dim = base_patch_size[0] * base_patch_size[1] * in_chans
    linear_weight = torch.randn(embed_dim, linear_input_dim)
    linear_bias = torch.zeros(embed_dim)

    # 对于 Linear 模式，interpolator 期望的输入比较特殊
    # 如果涉及到 resize，它需要知道 pixel 的空间排列，所以需要输入 5D 张量:
    # [B, Num_Patches, Patch_H, Patch_W, Ops_In_Channels]

    # 假设我们想用 8x8 的 patch size (原始是 16x16)
    target_patch_h, target_patch_w = 8, 8

    # 构造假数据：Batch=1, 100个patches, 每个patch 8x8x3
    patches_input = torch.randn(1, 100, target_patch_h, target_patch_w, in_chans)

    out_linear = interpolator(
        patches=patches_input,  # 已经是切分好的 8x8 patch
        proj_weight=linear_weight,  # 原始 16x16 对应的权重
        proj_bias=linear_bias,
        patch_size=(8, 8),  # 目标尺寸
        is_linear=True,
    )

    print(f"Linear resize input shape: {patches_input.shape}")
    print(f"Linear resize output shape: {out_linear.shape}")
    # Expect: (1, 100, 32) -> [B, N, Embed]


if __name__ == "__main__":
    test_patch_embed_interpolator()

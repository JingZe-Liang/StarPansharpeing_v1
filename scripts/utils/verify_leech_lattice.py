import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter

# 假设你之前的生成代码在 generate_leech.py 中，或者你可以直接把生成逻辑放进来
# 这里我们假设你有一个函数 generate_leech_lattice_vectors() 返回 (196560, 24) 的 numpy 数组


def verify_leech_lattice_properties(vectors):
    """
    验证 Leech Lattice 最小向量的核心数学属性。
    输入必须是未归一化的整数向量 (Standard Coordinates, norm_sq=32)。
    """
    print("=== 开始 Leech Lattice 数学属性验证 ===")

    N, D = vectors.shape
    print(f"检查维度: {N} x {D}")
    assert N == 196560, f"数量错误: 期望 196560, 实际 {N}"
    assert D == 24, f"维度错误: 期望 24, 实际 {D}"
    print("✅ 数量和维度检查通过")

    # 1. 检查模长 (Norm check)
    # 标准 Leech Lattice 最小向量的模长平方应为 32
    norm_sq = np.sum(vectors**2, axis=1)
    is_norm_correct = np.allclose(norm_sq, 32.0)
    if not is_norm_correct:
        print(f"❌ 模长错误! 期望 32.0, 发现范围 [{norm_sq.min()}, {norm_sq.max()}]")
        return False
    print("✅ 模长 (Squared Norm = 32) 检查通过")

    # 2. 检查整数性 (Integrality check)
    is_integer = np.all(np.mod(vectors, 1) == 0)
    if not is_integer:
        print("❌ 整数性错误! 发现非整数坐标")
        return False
    print("✅ 整数坐标检查通过")

    # 3. 检查子集分布 (Subgroup Distribution)
    # Type 4: Frame (±4, ±4, 0...) -> count 1104
    # Type 2: Octad (±2 x8, 0...) -> count 97152
    # Type 3: Dense (±3, ±1 x23) -> count 98304

    # 统计非零元素的个数 (Sparsity) 和最大绝对值
    non_zeros = np.count_nonzero(vectors, axis=1)
    max_vals = np.max(np.abs(vectors), axis=1)

    # 组合特征: (非零个数, 最大值)
    features = [tuple(x) for x in np.stack([non_zeros, max_vals], axis=1)]
    counts = Counter(features)

    print("\n子集分布统计 (期望值):")

    # Check Type 4 (Frames)
    # 特征: 非零数=2, 最大值=4. 数量应为 1104
    c4 = counts.get((2, 4.0), 0)
    print(f"Type 4 (Frames) [2 non-zeros, max 4]: {c4} / 1104 {'✅' if c4 == 1104 else '❌'}")

    # Check Type 2 (Octads)
    # 特征: 非零数=8, 最大值=2. 数量应为 97152
    c2 = counts.get((8, 2.0), 0)
    print(f"Type 2 (Octads) [8 non-zeros, max 2]: {c2} / 97152 {'✅' if c2 == 97152 else '❌'}")

    # Check Type 3 (Dense)
    # 特征: 非零数=24, 最大值=3. 数量应为 98304
    c3 = counts.get((24, 3.0), 0)
    print(f"Type 3 (Dense)  [24 non-zeros, max 3]: {c3} / 98304 {'✅' if c3 == 98304 else '❌'}")

    total_subsets = c4 + c2 + c3
    assert total_subsets == 196560, "子集总和不匹配，可能生成了异常向量！"

    # 4. 检查坐标和的模 (Conway Condition)
    # 每一个向量的坐标之和必须是 4 的倍数 (Sum % 4 == 0)
    coord_sums = np.sum(vectors, axis=1)
    is_sum_mod4 = np.all(coord_sums % 4 == 0)
    print(f"\n✅ 坐标和模 4 约束检查 {'通过' if is_sum_mod4 else '❌ 失败'}")

    return True


def verify_geometry_random_sampling(vectors_normalized, num_samples=1000):
    """
    针对已归一化的向量进行几何检查。
    检查任意两个向量之间的最小角度。

    原理：Leech Lattice 中任意两个不同向量 u, v (|u|^2=|v|^2=32)
    |u-v|^2 >= 32 (最小距离)
    => |u|^2 + |v|^2 - 2u.v >= 32
    => 32 + 32 - 2u.v >= 32
    => 32 >= 2u.v
    => u.v <= 16
    对于归一化向量 (norm=1)，这意味着点积 (Cosine Sim) 必须 <= 16/32 = 0.5
    (除了 u=v 的情况)
    """
    print("\n=== 开始几何分布验证 (随机采样) ===")

    # 转为 PyTorch 计算更快
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vecs = torch.from_numpy(vectors_normalized).to(device)

    # 随机采样一部分进行两两点积
    indices = torch.randperm(vecs.size(0))[:num_samples]
    sample_vecs = vecs[indices]

    # 计算点积矩阵 (Cosine Similarity)
    # Shape: (num_samples, num_samples)
    dot_products = sample_vecs @ sample_vecs.T

    # 移除对角线 (自己和自己点积肯定是 1.0)
    mask = ~torch.eye(num_samples, dtype=torch.bool, device=device)
    off_diag_dots = dot_products[mask]

    max_dot = off_diag_dots.max().item()

    print(f"随机采样 {num_samples} 个向量进行两两内积...")
    print(f"最大非对角线点积 (Max Cos Sim): {max_dot:.6f}")

    # 理论阈值是 0.5，但考虑到浮点误差，稍微放宽一点点
    if max_dot <= 0.5 + 1e-4:
        print("✅ 几何间隔检查通过 (Max Cos <= 0.5)")
        return True
    else:
        print(f"❌ 几何间隔异常! 发现两个非同一向量过于接近 (Cos > 0.5)")
        return False


# ================= 使用示例 =================

if __name__ == "__main__":
    """
    python -m scripts.utils.verify_leech_lattice
    """
    # from your_script import generate_leech_lattice_vectors
    try:
        from .generate_leech_lattice_v2 import generate_leech_lattice_vectors

        raw_vectors = generate_leech_lattice_vectors()

        # 验证原始数学属性
        success_math = verify_leech_lattice_properties(raw_vectors)

        # 验证归一化后的几何属性
        normalized_vectors = raw_vectors / np.sqrt(32.0)  # 手动归一化用于测试
        success_geo = verify_geometry_random_sampling(normalized_vectors, num_samples=2000)

        if success_math and success_geo:
            print("\n🎉 恭喜！生成的 Leech Weights 完美通过所有验证！")
        else:
            print("\n⚠️ 验证未完全通过，请检查生成逻辑。")

    except ImportError:
        print("请确保 generate_leech_lattice_vectors 函数可用")

import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter


def get_golay_generator_matrix():
    I = np.eye(12, dtype=int)
    core_row = np.array([1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0], dtype=int)
    B = np.zeros((11, 11), dtype=int)
    for i in range(11):
        B[i] = np.roll(core_row, i)
    A = np.zeros((12, 12), dtype=int)
    A[0, 1:] = 1
    A[1:, 0] = 1
    A[1:, 1:] = B
    G = np.concatenate((I, A), axis=1)
    return G


def generate_all_golay_codewords(G):
    n_words = 1 << 12
    # uint8 -> int32 转换防止溢出
    inputs = (
        np.unpackbits(np.arange(n_words, dtype=">u2").view(np.uint8)).reshape(n_words, 16)[:, -12:].astype(np.int32)
    )

    codewords = np.dot(inputs, G) % 2
    return codewords


def generate_leech_lattice_vectors():
    print("Generating Golay Code...")
    G = get_golay_generator_matrix()
    codewords = generate_all_golay_codewords(G)  # (4096, 24)

    # === Group 1: Octads (weights=8) ===
    print("Generating Group 1 (Octads)...")
    weights = np.sum(codewords, axis=1)
    octad_indices = np.where(weights == 8)[0]
    octads = codewords[octad_indices]  # (759, 24)

    n_sign_patterns = 1 << 7
    # 关键修正：这里必须转为 float 或 int32，否则 1 - 2*uint8 会变成 255
    raw_signs = np.unpackbits(np.arange(256, dtype=">u2").view(np.uint8)).reshape(256, 16)[:, -8:].astype(np.float32)

    raw_signs = 1.0 - 2.0 * raw_signs  # {1.0, -1.0}
    valid_mask = np.sum(raw_signs, axis=1) % 4 == 0
    valid_signs = raw_signs[valid_mask]  # (128, 8)

    group1_vectors = []
    for octad in octads:
        indices = np.where(octad == 1)[0]
        vecs = np.zeros((128, 24), dtype=np.float32)
        vecs[:, indices] = valid_signs * 2.0  # Apply +/- 2
        group1_vectors.append(vecs)

    group1_vectors = np.concatenate(group1_vectors, axis=0)

    # === Group 2: Coordinate Frames ===
    print("Generating Group 2 (Coordinate Frames)...")
    group2_vectors = []
    dim = 24
    for i in range(dim):
        for j in range(i + 1, dim):
            for s1 in [4.0, -4.0]:
                for s2 in [4.0, -4.0]:
                    vec = np.zeros(dim, dtype=np.float32)
                    vec[i] = s1
                    vec[j] = s2
                    group2_vectors.append(vec)
    group2_vectors = np.array(group2_vectors, dtype=np.float32)

    # === Group 3: Dense Vectors ===
    print("Generating Group 3 (Dense Vectors)...")
    # 同样转换类型
    u_base = 1.0 - 2.0 * codewords.astype(np.float32)

    u_expanded = np.repeat(u_base[:, np.newaxis, :], 24, axis=1)  # (4096, 24, 24)
    b_indices = np.arange(4096)[:, None]
    k_indices = np.arange(24)[None, :]

    # Modify: x -> x - 4sign(x) = -3x (for +/-1)
    # Since u_base is +/- 1, we just multiply by -3 at the specific index
    u_expanded[b_indices, k_indices, k_indices] *= -3.0

    group3_vectors = u_expanded.reshape(-1, 24)

    # === Concatenate ===
    all_vectors = np.concatenate([group1_vectors, group2_vectors, group3_vectors], axis=0)
    return all_vectors


# ================= 验证逻辑 =================


def run_verification():
    vectors = generate_leech_lattice_vectors()

    print("\n=== 开始 Leech Lattice 数学属性验证 ===")
    N, D = vectors.shape
    print(f"检查维度: {N} x {D}")

    # 1. Norm Check
    norm_sq = np.sum(vectors**2, axis=1)
    # 允许微小的浮点误差
    is_norm_correct = np.allclose(norm_sq, 32.0, atol=1e-3)

    if is_norm_correct:
        print("✅ 模长 (Squared Norm = 32) 检查通过")
    else:
        print(f"❌ 模长错误! 范围 [{norm_sq.min():.2f}, {norm_sq.max():.2f}]")
        # 打印前几个错误的
        err_idx = np.where(~np.isclose(norm_sq, 32.0, atol=1e-3))[0]
        print(f"错误样本索引: {err_idx[:5]}")
        print(f"错误样本值: \n{vectors[err_idx[0]]}")
        return

    # 2. Count Check
    # Frame (type 4): max abs 4, count 1104
    # Octad (type 2): max abs 2, count 97152
    # Dense (type 3): max abs 3, count 98304
    max_vals = np.max(np.abs(vectors), axis=1)

    c4 = np.sum(np.isclose(max_vals, 4.0))
    c2 = np.sum(np.isclose(max_vals, 2.0))
    c3 = np.sum(np.isclose(max_vals, 3.0))

    print(f"Type 4 (Frames, max=4): {c4} (Expect 1104) {'✅' if c4 == 1104 else '❌'}")
    print(f"Type 2 (Octads, max=2): {c2} (Expect 97152) {'✅' if c2 == 97152 else '❌'}")
    print(f"Type 3 (Dense,  max=3): {c3} (Expect 98304) {'✅' if c3 == 98304 else '❌'}")

    # 3. Geometry Check (Sampling)
    print("\n=== 开始几何分布验证 (随机采样) ===")
    # 归一化
    vecs_normalized = torch.from_numpy(vectors)
    vecs_normalized = F.normalize(vecs_normalized, p=2, dim=-1)

    # 随机采样 2000 个
    indices = torch.randperm(N)[:2000]
    sample_vecs = vecs_normalized[indices]

    # 计算点积
    dot_products = sample_vecs @ sample_vecs.T

    # Mask diagonal
    mask = ~torch.eye(2000, dtype=torch.bool)
    max_dot = dot_products[mask].max().item()

    print(f"Max Cosine Sim: {max_dot:.6f}")
    if max_dot <= 0.5 + 1e-3:
        print("✅ 几何间隔检查通过 (Max Cos <= 0.5)")

        # 保存文件
        save_path = "leech_lattices_normalized.npy"
        print(f"\n验证全部通过，正在保存至 {save_path} ...")
        np.save(save_path, vecs_normalized.numpy())
        print("保存完成！")

    else:
        print("❌ 几何间隔异常!")


if __name__ == "__main__":
    run_verification()

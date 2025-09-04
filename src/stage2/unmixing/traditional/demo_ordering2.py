#!/usr/bin/env python3
"""
演示VCA端成员顺序问题的更明显例子
"""

import numpy as np
import torch
from VAC import vca, vca_torch, find_best_matching


def create_problematic_example():
    """创建一个更容易出现顺序问题的例子"""
    # 创建5个端成员，其中有些很相似
    endmembers = np.array(
        [
            [0.9, 0.8, 0.7, 0.6, 0.5],  # 物质A
            [0.1, 0.2, 0.3, 0.4, 0.5],  # 物质B
            [0.8, 0.7, 0.6, 0.5, 0.4],  # 物质C (与A相似)
            [0.2, 0.3, 0.4, 0.5, 0.6],  # 物质D (与B相似)
            [0.5, 0.5, 0.5, 0.5, 0.5],  # 物质E (中等值)
        ]
    ).T

    # 创建丰度矩阵
    np.random.seed(123)
    abundances = np.random.rand(5, 2000)
    abundances = abundances / abundances.sum(axis=0, keepdims=True)

    # 生成混合数据
    Y = np.dot(endmembers, abundances)

    # 添加噪声
    Y += 0.02 * np.random.randn(*Y.shape)

    return Y, endmembers


def main():
    print("=== VCA端成员顺序问题演示 (复杂案例) ===\n")

    # 生成测试数据
    Y, true_endmembers = create_problematic_example()
    print(f"真实端成员形状: {true_endmembers.shape}")
    print(f"数据形状: {Y.shape}")

    # 运行numpy VCA
    print("\n--- 运行Numpy VCA ---")
    np.random.seed(42)
    Ae_np, indices_np, _ = vca(Y, 5, verbose=False)
    print(f"Numpy提取的端成员顺序: {indices_np}")

    # 运行torch VCA (使用不同的随机种子)
    print("\n--- 运行Torch VCA ---")
    torch.manual_seed(456)  # 不同的种子
    Y_torch = torch.from_numpy(Y).float()
    Ae_torch, indices_torch, _ = vca_torch(Y_torch, 5, verbose=False)
    print(f"Torch提取的端成员顺序: {indices_torch.cpu().numpy()}")

    # 直接比较 (按位置)
    print("\n--- 直接比较 (按位置) ---")
    direct_corrs = []
    for i in range(5):
        corr = np.corrcoef(Ae_np[:, i], Ae_torch[:, i].cpu().numpy())[0, 1]
        direct_corrs.append(corr)
        print(f"位置{i}: numpy端成员 vs torch端成员, 相关性 = {corr:.4f}")
    print(f"平均直接相关性: {np.mean(direct_corrs):.4f}")

    # 最优匹配比较
    print("\n--- 最优匹配比较 ---")
    row_ind, col_ind, correlations = find_best_matching(Ae_np, Ae_torch.cpu().numpy())

    print("最优匹配结果:")
    for i, (np_idx, torch_idx) in enumerate(zip(row_ind, col_ind)):
        corr = correlations[i]
        print(
            f"匹配{i}: numpy端成员{np_idx} <-> torch端成员{torch_idx}, 相关性 = {corr:.4f}"
        )
    print(f"平均匹配相关性: {np.mean(correlations):.4f}")

    # 分析相关性矩阵
    print("\n--- 相关性矩阵分析 ---")
    corr_matrix = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            corr_matrix[i, j] = np.corrcoef(Ae_np[:, i], Ae_torch[:, j].cpu().numpy())[
                0, 1
            ]

    print("相关性矩阵 (numpy行 vs torch列):")
    for i in range(5):
        print(f"  {i}: {corr_matrix[i, :]}")

    print(f"\n每行的最大相关性: {np.max(corr_matrix, axis=1)}")
    print(f"每列的最大相关性: {np.max(corr_matrix, axis=0)}")

    print("\n=== 结论 ===")
    print("1. 当端成员相似时，VCA更容易出现顺序混乱")
    print("2. 直接按位置比较可能得到很差的结果")
    print("3. 最优匹配能找到真正的对应关系")
    print("4. 实际应用中必须使用匹配算法来评估VCA性能")


if __name__ == "__main__":
    main()

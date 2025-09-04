#!/usr/bin/env python3
"""
演示VCA端成员顺序问题
"""

import numpy as np
import torch
from VAC import vca, vca_torch, find_best_matching


def create_simple_example():
    """创建一个简单的例子来演示端成员顺序问题"""
    # 创建4个端成员光谱 (模拟真实物质)
    endmembers = np.array(
        [
            [0.1, 0.2, 0.8, 0.7, 0.3],  # 植被
            [0.9, 0.8, 0.1, 0.2, 0.4],  # 土壤
            [0.2, 0.3, 0.4, 0.8, 0.9],  # 水体
            [0.7, 0.6, 0.5, 0.4, 0.3],  # 建筑
        ]
    ).T  # [5, 4] - 5个波段，4个端成员

    # 创建丰度矩阵
    np.random.seed(42)
    abundances = np.random.rand(4, 1000)
    abundances = abundances / abundances.sum(axis=0, keepdims=True)

    # 生成混合数据
    Y = np.dot(endmembers, abundances)

    # 添加少量噪声
    Y += 0.01 * np.random.randn(*Y.shape)

    return Y, endmembers


def main():
    print("=== VCA端成员顺序问题演示 ===\n")

    # 生成测试数据
    Y, true_endmembers = create_simple_example()
    print(f"真实端成员形状: {true_endmembers.shape}")
    print(f"数据形状: {Y.shape}")

    # 设置相同的随机种子
    np.random.seed(42)
    torch.manual_seed(42)

    # 运行numpy VCA
    print("\n--- 运行Numpy VCA ---")
    Ae_np, indices_np, _ = vca(Y, 4, verbose=False)
    print(f"Numpy提取的端成员顺序: {indices_np}")

    # 运行torch VCA (使用不同的随机种子来模拟差异)
    print("\n--- 运行Torch VCA ---")
    torch.manual_seed(123)  # 使用不同的种子
    Y_torch = torch.from_numpy(Y).float()
    Ae_torch, indices_torch, _ = vca_torch(Y_torch, 4, verbose=False)
    print(f"Torch提取的端成员顺序: {indices_torch.cpu().numpy()}")

    # 直接比较 (按位置)
    print("\n--- 直接比较 (按位置) ---")
    direct_corrs = []
    for i in range(4):
        corr = np.corrcoef(Ae_np[:, i], Ae_torch[:, i].cpu().numpy())[0, 1]
        direct_corrs.append(corr)
        print(f"位置{i}: numpy端成员 vs torch端成员, 相关性 = {corr:.4f}")
    print(f"平均直接相关性: {np.mean(direct_corrs):.4f}")

    # 最优匹配比较
    print("\n--- 最优匹配比较 ---")
    row_ind, col_ind, correlations = find_best_matching(Ae_np, Ae_torch.cpu().numpy())

    for i, (np_idx, torch_idx) in enumerate(zip(row_ind, col_ind)):
        corr = correlations[i]
        print(
            f"匹配{i}: numpy端成员{np_idx} <-> torch端成员{torch_idx}, 相关性 = {corr:.4f}"
        )
    print(f"平均匹配相关性: {np.mean(correlations):.4f}")

    print("\n=== 结论 ===")
    print("1. VCA算法由于随机性，提取的端成员顺序可能不同")
    print("2. 直接按位置比较会低估算法性能")
    print("3. 使用匈牙利算法进行最优匹配能反映真实性能")
    print("4. 端成员是无序集合，顺序不重要，重要的是能否提取出相同的端成员")


if __name__ == "__main__":
    main()

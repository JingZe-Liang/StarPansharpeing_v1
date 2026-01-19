from torch.utils.data import DataLoader
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as T
import random
import matplotlib.pyplot as plt
import math


def _normalize(data, division, norm_range):
    if not norm_range:
        data = data / division
    else:
        data = data / division * 2.0 - 1.0

    return data.astype(np.float32)


class PanDataset(Dataset):
    def __init__(self, h5_path, aug_prob=0.0, norm_range=False, division=1024.0):
        super().__init__()
        self.h5_path = h5_path
        self.aug_prob = aug_prob
        self.norm_range = norm_range
        self.division = division
        self.h5_file = None  # 文件句柄将在需要时被创建

        # 在初始化时，只读取数据集的长度
        with h5py.File(self.h5_path, "r") as f:
            # 假设所有 dataset 的第一维度(axis 0)都是样本数
            self.length = len(f["gt"])  # 使用 'gt' 作为长度基准

        # 定义数据增强变换
        if self.aug_prob > 0.0:
            self.geo_trans = T.Compose(
                [
                    T.RandomHorizontalFlip(p=1.0),  # p=1.0因为我们会用aug_prob来决定是否应用整个Compose
                    T.RandomVerticalFlip(p=1.0),
                ]
            )

        print(f"Dataset initialized from {h5_path}.")
        print(f"Number of samples: {self.length}")
        print(f"Augmentation probability: {self.aug_prob}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 为了在多进程数据加载中保持健壮性，在每个worker中独立打开文件句柄
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")

        # 1. 按需从硬盘加载一条数据
        # 读取的数据是 NumPy 数组
        gt_np = self.h5_file["gt"][idx]
        lms_np = self.h5_file["lms"][idx]
        pan_np = self.h5_file["pan"][idx]

        # 2. 对这一条数据进行归一化处理
        gt_norm = _normalize(gt_np, self.division, self.norm_range)
        lms_norm = _normalize(lms_np, self.division, self.norm_range)
        pan_norm = _normalize(pan_np, self.division, self.norm_range)

        # 3. 将 NumPy 数组转换为 PyTorch Tensors
        gt_tensor = torch.from_numpy(gt_norm)
        lms_tensor = torch.from_numpy(lms_norm)
        pan_tensor = torch.from_numpy(pan_norm)

        # 4. 如果需要，应用数据增强
        if self.aug_prob > 0 and random.random() < self.aug_prob:
            # 使用相同的随机种子，确保对三张图的几何变换是一致的
            seed = torch.random.seed()
            torch.manual_seed(seed)
            gt_aug = self.geo_trans(gt_tensor)
            torch.manual_seed(seed)
            lms_aug = self.geo_trans(lms_tensor)
            torch.manual_seed(seed)
            pan_aug = self.geo_trans(pan_tensor)
        else:
            gt_aug, lms_aug, pan_aug = gt_tensor, lms_tensor, pan_tensor

        # 5. 构建模型需要的 cond_image 和 target_image
        # target 是真实图像和低分辨率上采样图像的残差
        target_image = gt_aug - lms_aug

        # condition 是低分辨率上采样图像和全色图像的堆叠
        # 注意: pan_aug 可能只有一个通道, 需要扩展维度以匹配
        if pan_aug.ndim == 2:  # (H, W) -> (1, H, W)
            pan_aug = pan_aug.unsqueeze(0)
        cond_image = torch.cat([lms_aug, pan_aug], dim=0)
        # cond_image = pan_aug.repeat(3, 1, 1)  # 将 [1, H, W] 复制为 [3, H, W]

        # 6. 返回最终的字典
        return {
            "cond_image": cond_image,
            "target_image": target_image,
            "lms_image": lms_aug,
            "gt_image": gt_aug,
            # 'recon_image': target_image + lms_aug
        }

    def __repr__(self):
        return f"PanDataset(file='{self.h5_path}', size={self.length})"


if __name__ == "__main__":
    train_h5_path = "D:/work/dataset/pansharpening/gf2/test/reduced_examples/test_gf2_multiExm1.h5"

    try:
        train_dataset = PanDataset(
            h5_path=train_h5_path,
            aug_prob=0.0,
            norm_range=False,
            division=1023.0,  # 根据您的数据调整
        )
    except (FileNotFoundError, OSError) as e:
        print(f"错误: H5 文件未找到或无法读取，请检查路径: {train_h5_path}\n{e}")
        exit()

    # 3. 使用 DataLoader 加载
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,  # 使用大于1的批大小来测试
        shuffle=False,
        num_workers=0,
    )

    print("\n--- Testing DataLoader for PanDataset ---")

    try:
        first_batch = next(iter(train_dataloader))
    except StopIteration:
        print("错误：DataLoader 为空，无法获取数据。请检查H5文件是否包含数据。")
        exit()

    cond_img_batch = first_batch["cond_image"]
    target_img_batch = first_batch["target_image"]
    lms_img_batch = first_batch["lms_image"]
    gt_img_batch = first_batch["gt_image"]

    # recon_batch = first_batch['recon_image']

    print(f"\n--- Batch Information (Batch Size = {gt_img_batch.shape[0]}) ---")
    print(f"{'Key':<15} | {'Shape':<30} | {'DType':<15} | {'Value Range'}")
    print("-" * 80)

    # 辅助函数，用于打印张量信息
    def print_info(name, tensor):
        val_range = f"[{tensor.min():.4f}, {tensor.max():.4f}]"
        print(f"{name:<15} | {str(tuple(tensor.shape)):<30} | {str(str(tensor.dtype)):<15} | {val_range}")

    print_info("cond_image", cond_img_batch)
    print_info("target_image", target_img_batch)
    print_info("lms_image", lms_img_batch)
    print_info("gt_image", gt_img_batch)

    # print_info('recon_image', recon_batch)

    print("-" * 80)

    print("\n--- Visualizing First Sample in the Batch ---")

    def visualize_all_shearlet_channels(tensor, title):
        tensor = tensor.squeeze(0).cpu()
        C, H, W = tensor.shape
        cols = min(9, C)
        rows = math.ceil(C / cols)

        plt.figure(figsize=(3 * cols, 3 * rows))
        for i in range(C):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(tensor[i], cmap="gray")
            plt.title(f"{title}[{i}]", fontsize=8)
            plt.axis("off")
        plt.tight_layout()
        plt.show()

    visualize_all_shearlet_channels(first_batch["cond_image"], "cond_img_batch")
    visualize_all_shearlet_channels(first_batch["target_image"], "target")
    visualize_all_shearlet_channels(first_batch["lms_image"], "LRMS")

    print("\n--- Test Complete ---")

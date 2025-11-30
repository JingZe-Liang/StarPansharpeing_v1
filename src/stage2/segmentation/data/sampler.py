import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from jaxtyping import Int


class BalancedSampler:
    """
    高光谱数据平衡采样器

    提供多种策略来处理高光谱数据中的类别不平衡问题，
    支持分类和分割任务的采样需求。

    Parameters
    ----------
    class_indices : dict[int, list[int]]
        每个类别的像素索引列表
    balance_strategy : str, optional
        平衡策略类型，可选：
        - 'equal_size': 每个类别等数量采样
        - 'proportional': 按类别比例采样
        - 'stratified': 分层采样，生成训练/验证/测试集
        - 'adaptive': 自适应采样，根据类别分布调整
        - 'combined': 组合采样，对少数类过采样，多数类欠采样
        Default: 'equal_size'
    random_seed : int or None, optional
        随机种子，用于结果可重现
        Default: None

    Examples
    --------
    >>> # 按类别组织索引
    >>> class_indices = {
    ...     1: [10, 45, 78, 132],
    ...     2: [15, 56, 89, 134, 167],
    ...     3: [23, 67, 98, 143, 176, 209]
    ... }
    >>> # 创建采样器
    >>> sampler = BalancedSampler(class_indices, balance_strategy='equal_size', random_seed=42)
    >>> # 执行采样
    >>> balanced_indices = sampler.sample(samples_per_class=100)
    >>> # 查看统计信息
    >>> stats = sampler.get_class_stats()
    >>> for class_id, info in stats.items():
    ...     print(f"Class {class_id}: {info['count']} samples ({info['percentage']:.1f}%)")
    """

    def __init__(
        self,
        gt2d: Int[np.ndarray, "h w"],
        class_indices: Dict[int, List[int]],
        balance_strategy: str = "equal_size",
        random_seed: Optional[int] = None,
    ):
        self.class_indices = class_indices
        self.strategy = balance_strategy

        if random_seed is not None:
            np.random.seed(random_seed)

        # 验证策略有效性
        valid_strategies = ["equal_size", "proportional", "stratified", "adaptive", "combined"]
        if balance_strategy not in valid_strategies:
            raise ValueError(f"Unknown strategy: {balance_strategy}. Available: {valid_strategies}")

    def sample(self, **kwargs) -> Union[Dict[int, List[int]], Tuple[Dict, Dict, Dict]]:
        """
        执行采样

        Parameters
        ----------
        **kwargs : dict
            采样参数，根据不同策略有所不同

        Returns
        -------
        dict[int, list[int]] or tuple
            平衡采样后的索引。对于'stratified'策略返回3个字典的元组
            (train_indices, val_indices, test_indices)
        """
        if self.strategy == "equal_size":
            return self._equal_size_sampling(**kwargs)
        elif self.strategy == "proportional":
            return self._proportional_sampling(**kwargs)
        elif self.strategy == "stratified":
            return self._stratified_sampling(**kwargs)
        elif self.strategy == "adaptive":
            return self._adaptive_sampling(**kwargs)
        elif self.strategy == "combined":
            return self._combined_sampling(**kwargs)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def get_class_stats(self) -> Dict[int, Dict[str, Union[int, float]]]:
        """
        获取类别统计信息

        Returns
        -------
        dict[int, dict]
            每个类别的统计信息，包含'count'和'percentage'
        """
        stats = {}
        total_samples = sum(len(indices) for indices in self.class_indices.values())

        for class_id, indices in self.class_indices.items():
            stats[class_id] = {"count": len(indices), "percentage": len(indices) / total_samples * 100.0}
        return stats

    def _equal_size_sampling(
        self, samples_per_class: Optional[int] = None, shuffle: bool = True
    ) -> Dict[int, List[int]]:
        """
        等数量采样：每个类别固定数量

        Parameters
        ----------
        samples_per_class : int or None, optional
            每类采样数量，None表示按最少类自动确定
            Default: None
        shuffle : bool, optional
            是否打乱顺序
            Default: True

        Returns
        -------
        dict[int, list[int]]
            平衡采样后的索引
        """
        balanced_indices = {}

        # 确定每类采样数量
        if samples_per_class is None:
            min_class_size = min(len(indices) for indices in self.class_indices.values())
            samples_per_class = min_class_size

        for class_id, indices in self.class_indices.items():
            available_count = len(indices)

            if available_count < samples_per_class:
                # 如果样本不够，使用所有可用样本（并发出警告）
                selected = indices.copy()
                print(
                    f"Warning: Class {class_id} only has {available_count} samples, "
                    f"less than requested {samples_per_class}"
                )
            else:
                # 随机采样指定数量
                indices_copy = indices.copy()
                if shuffle:
                    np.random.shuffle(indices_copy)
                selected = indices_copy[:samples_per_class]

            balanced_indices[class_id] = selected

        return balanced_indices

    def _proportional_sampling(
        self,
        total_samples: int,
        min_per_class: Optional[int] = None,
        max_per_class: Optional[int] = None,
        shuffle: bool = True,
    ) -> Dict[int, List[int]]:
        """
        按类别大小比例采样

        Parameters
        ----------
        total_samples : int
            总采样数量
        min_per_class : int or None, optional
            每类最少采样数量
            Default: None
        max_per_class : int or None, optional
            每类最多采样数量
            Default: None
        shuffle : bool, optional
            是否打乱顺序
            Default: True

        Returns
        -------
        dict[int, list[int]]
            平衡采样后的索引
        """
        # 计算每个类别的比例
        class_sizes = {k: len(v) for k, v in self.class_indices.items()}
        total_original = sum(class_sizes.values())

        proportions = {k: v / total_original for k, v in class_sizes.items()}

        # 初始化采样结果
        sampled_indices = {}
        used_samples = 0

        # 第一轮：满足最少数量要求
        if min_per_class is not None:
            for class_id, indices in self.class_indices.items():
                available = len(indices)
                actual_min = min(min_per_class, available)

                indices_copy = indices.copy()
                if shuffle:
                    np.random.shuffle(indices_copy)
                sampled_indices[class_id] = indices_copy[:actual_min]

                used_samples += actual_min

            # 剩余可采样数量
            remaining_samples = total_samples - used_samples
            if remaining_samples <= 0:
                return sampled_indices

        # 第二轮：按比例分配剩余数量
        remaining_classes = set(self.class_indices.keys()) - set(sampled_indices.keys())

        if remaining_classes:
            # 重新计算剩余类的比例
            remaining_total = sum(len(self.class_indices[cls]) for cls in remaining_classes)
            remaining_props = {cls: len(self.class_indices[cls]) / remaining_total for cls in remaining_classes}

            for class_id, indices in self.class_indices.items():
                if class_id in sampled_indices:
                    continue

                # 计算该类别的采样数量
                class_samples = int(remaining_samples * remaining_props[class_id])

                # 应用最多数量限制
                if max_per_class is not None:
                    class_samples = min(class_samples, max_per_class)

                # 不能超过可用数量
                available = len(indices)
                class_samples = min(class_samples, available)

                # 随机采样
                indices_copy = indices.copy()
                if shuffle:
                    np.random.shuffle(indices_copy)
                sampled_indices[class_id] = indices_copy[:class_samples]

        return sampled_indices

    def _stratified_sampling(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        min_per_class: int = 1,
        shuffle: bool = True,
    ) -> Tuple[Dict, Dict, Dict]:
        """
        分层采样：同时生成训练、验证、测试集

        Parameters
        ----------
        train_ratio : float, optional
            训练集比例
            Default: 0.7
        val_ratio : float, optional
            验证集比例
            Default: 0.2
        test_ratio : float, optional
            测试集比例
            Default: 0.1
        min_per_class : int, optional
            每个集合的最少样本数
            Default: 1
        shuffle : bool, optional
            是否打乱顺序
            Default: True

        Returns
        -------
        tuple[dict, dict, dict]
            (train_indices, val_indices, test_indices)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

        train_indices = {}
        val_indices = {}
        test_indices = {}

        for class_id, indices in self.class_indices.items():
            available_count = len(indices)

            # 检查是否满足最少数量要求
            min_total = min_per_class * 3  # 每个集合都需要
            if available_count < min_total:
                print(f"Warning: Class {class_id} only has {available_count} samples, less than minimum {min_total}")

            indices_copy = indices.copy()
            if shuffle:
                np.random.shuffle(indices_copy)

            # 计算每个集合的数量
            n_train = max(min_per_class, int(available_count * train_ratio))
            n_val = max(min_per_class, int(available_count * val_ratio))
            n_test = max(min_per_class, int(available_count * test_ratio))

            # 检查总数是否超出
            total_assigned = n_train + n_val + n_test
            if total_assigned > available_count:
                # 按比例缩减
                scale = available_count / total_assigned
                n_train = max(min_per_class, int(n_train * scale))
                n_val = max(min_per_class, int(n_val * scale))
                n_test = max(min_per_class, int(n_test * scale))

            # 分配样本
            start_idx = 0
            train_indices[class_id] = indices_copy[start_idx : start_idx + n_train]
            start_idx += n_train

            val_indices[class_id] = indices_copy[start_idx : start_idx + n_val]
            start_idx += n_val

            test_indices[class_id] = indices_copy[start_idx : start_idx + n_test]

        return train_indices, val_indices, test_indices

    def _adaptive_sampling(
        self, target_size: Optional[int] = None, balance_strategy: str = "inverse"
    ) -> Dict[int, List[int]]:
        """
        自适应采样：根据类别分布自动调整

        Parameters
        ----------
        target_size : int or None, optional
            目标总样本数
            Default: None
        balance_strategy : str, optional
            平衡策略: 'inverse', 'sqrt_inverse', 'log_inverse'
            Default: 'inverse'

        Returns
        -------
        dict[int, list[int]]
            平衡采样后的索引
        """
        # 计算原始类别分布
        class_sizes = {k: len(v) for k, v in self.class_indices.items()}

        # 根据策略计算权重
        weights = {}
        for class_id, size in class_sizes.items():
            if balance_strategy == "inverse":
                weights[class_id] = 1.0 / size
            elif balance_strategy == "sqrt_inverse":
                weights[class_id] = 1.0 / np.sqrt(size)
            elif balance_strategy == "log_inverse":
                weights[class_id] = 1.0 / np.log(size + 1)
            else:
                raise ValueError(f"Unknown balance strategy: {balance_strategy}")

        # 归一化权重
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}

        # 计算每类采样数量
        if target_size is None:
            target_size = sum(class_sizes.values()) // 2  # 默认减少一半

        sampled_indices = {}
        for class_id, weight in weights.items():
            # 按权重分配样本数
            samples_for_class = int(target_size * weight)
            samples_for_class = max(1, samples_for_class)  # 至少1个

            available = len(self.class_indices[class_id])
            actual_samples = min(samples_for_class, available)

            # 随机采样
            indices_copy = self.class_indices[class_id].copy()
            np.random.shuffle(indices_copy)
            sampled_indices[class_id] = indices_copy[:actual_samples]

        return sampled_indices

    def _combined_sampling(
        self, target_size: Optional[int] = None, oversample_threshold: float = 0.5, undersample_threshold: float = 2.0
    ) -> Dict[int, List[int]]:
        """
        组合采样：对少数类过采样，对多数类欠采样

        Parameters
        ----------
        target_size : int or None, optional
            目标总样本数
            Default: None
        oversample_threshold : float, optional
            过采样阈值（相对于平均大小的比例）
            Default: 0.5
        undersample_threshold : float, optional
            欠采样阈值（相对于平均大小的比例）
            Default: 2.0

        Returns
        -------
        dict[int, list[int]]
            平衡采样后的索引
        """
        # 计算平均类别大小
        class_sizes = {k: len(v) for k, v in self.class_indices.items()}
        avg_size = sum(class_sizes.values()) / len(class_sizes)

        # 分类：少数类、中间类、多数类
        minority_classes = []
        majority_classes = []
        balanced_classes = []

        for class_id, size in class_sizes.items():
            ratio = size / avg_size

            if ratio < oversample_threshold:
                minority_classes.append(class_id)
            elif ratio > undersample_threshold:
                majority_classes.append(class_id)
            else:
                balanced_classes.append(class_id)

        # 采样策略
        sampled_indices = {}

        # 少数类：过采样（允许重复）
        for class_id in minority_classes:
            original_indices = self.class_indices[class_id]
            available_count = len(original_indices)

            # 计算目标数量（放大到平均大小）
            target_count = int(avg_size * oversample_threshold * 1.2)

            # 需要重复次数
            repeat_times = target_count // available_count
            remainder = target_count % available_count

            # 重复采样
            sampled = original_indices * repeat_times
            sampled += original_indices[:remainder]

            # 随机打乱
            np.random.shuffle(sampled)
            sampled_indices[class_id] = sampled

        # 中间类：保持不变
        for class_id in balanced_classes:
            sampled_indices[class_id] = self.class_indices[class_id].copy()
            np.random.shuffle(sampled_indices[class_id])

        # 多数类：欠采样
        for class_id in majority_classes:
            original_indices = self.class_indices[class_id]

            # 采样到目标大小
            target_count = int(avg_size * undersample_threshold * 0.8)
            target_count = min(target_count, len(original_indices))

            indices_copy = original_indices.copy()
            np.random.shuffle(indices_copy)
            sampled_indices[class_id] = indices_copy[:target_count]

        return sampled_indices


def balance_hyperspectral_segments(
    gt_2d: np.ndarray,
    balance_strategy: str = "equal_size",
    target_samples_per_class: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """
    专门用于高光谱分割的平衡采样

    Parameters
    ----------
    gt_2d : np.ndarray
        2D ground truth 图像
    balance_strategy : str, optional
        平衡策略类型
        Default: 'equal_size'
    target_samples_per_class : int or None, optional
        目标每类样本数
        Default: None
    random_seed : int or None, optional
        随机种子
        Default: None

    Returns
    -------
    np.ndarray
        平衡采样后的索引

    Examples
    --------
    >>> # 加载高光谱标签
    >>> gt_2d = np.load('indian_pines_gt.npy')  # Shape: (145, 145)
    >>> # 平衡采样
    >>> train_index = balance_hyperspectral_segments(
    ...     gt_2d,
    ...     balance_strategy='equal_size',
    ...     target_samples_per_class=100,
    ...     random_seed=42
    ... )
    >>> print(f"Sampled {len(train_index)} pixels for training")
    """
    # 按类别组织像素索引
    height, width = gt_2d.shape
    class_indices = {}

    for class_id in np.unique(gt_2d):
        if class_id == 0:  # 跳过背景
            continue

        # 获取该类别的所有像素位置
        class_pixels = np.where(gt_2d == class_id)

        # 将(row, col)转换为flatten索引
        indices = [r * width + c for r, c in zip(class_pixels[0], class_pixels[1])]
        class_indices[int(class_id)] = indices

    # 使用平衡采样器
    sampler = BalancedSampler(class_indices, balance_strategy=balance_strategy, random_seed=random_seed)

    # 执行采样
    if balance_strategy == "equal_size":
        balanced_indices = sampler.sample(samples_per_class=target_samples_per_class)
    else:
        if target_samples_per_class is not None:
            target_size = len(class_indices) * target_samples_per_class
        else:
            target_size = None
        balanced_indices = sampler.sample(target_size=target_size)

    # 转换为flatten的索引数组
    train_index = np.concatenate(list(balanced_indices.values()))

    return train_index


def create_hyperspectral_class_indices(gt_2d: np.ndarray) -> Dict[int, List[int]]:
    """
    从高光谱ground truth创建类别索引字典

    Parameters
    ----------
    gt_2d : np.ndarray
        2D ground truth 图像

    Returns
    -------
    dict[int, list[int]]
        每个类别的像素索引列表

    Examples
    --------
    >>> gt_2d = np.array([
    ...     [1, 1, 0, 2, 2],
    ...     [3, 0, 1, 0, 2],
    ...     [1, 1, 3, 3, 0]
    ... ])
    >>> class_indices = create_hyperspectral_class_indices(gt_2d)
    >>> print(class_indices[1])  # 类别1的像素索引
    [0, 1, 7, 12, 13]
    """
    height, width = gt_2d.shape
    class_indices = {}

    for class_id in np.unique(gt_2d):
        if class_id == 0:  # 跳过背景
            continue

        # 获取该类别的所有像素位置
        class_pixels = np.where(gt_2d == class_id)

        # 将(row, col)转换为flatten索引
        indices = [r * width + c for r, c in zip(class_pixels[0], class_pixels[1])]
        class_indices[int(class_id)] = indices

    return class_indices


def analyze_class_distribution(gt_2d: np.ndarray, print_stats: bool = True) -> Dict[int, Dict[str, Union[int, float]]]:
    """
    分析高光谱数据的类别分布

    Parameters
    ----------
    gt_2d : np.ndarray
        2D ground truth 图像
    print_stats : bool, optional
        是否打印统计信息
        Default: True

    Returns
    -------
    dict[int, dict]
        每个类别的统计信息

    Examples
    --------
    >>> gt_2d = np.load('indian_pines_gt.npy')
    >>> stats = analyze_class_distribution(gt_2d)
    >>> print(f"Total classes: {len(stats)}")
    Total classes: 16
    >>> print(f"Total pixels: {sum(info['count'] for info in stats.values())}")
    Total pixels: 21025
    """
    class_indices = create_hyperspectral_class_indices(gt_2d)
    sampler = BalancedSampler(class_indices)
    stats = sampler.get_class_stats()

    if print_stats:
        print(f"高光谱数据类别分布分析:")
        print(f"图像尺寸: {gt_2d.shape}")
        print(f"有效类别数: {len(stats)}")
        print(f"总像素数: {sum(info['count'] for info in stats.values())}")
        print("-" * 50)
        print(f"{'类别ID':<8} {'样本数':<10} {'百分比':<10} {'类别名称'}")
        print("-" * 50)

        # 常见高光谱数据集类别名称映射
        class_names = {
            1: "Alfalfa",
            2: "Corn-notill",
            3: "Corn-mintill",
            4: "Corn",
            5: "Grass-pasture",
            6: "Grass-trees",
            7: "Grass-pasture-mowed",
            8: "Hay-windrowed",
            9: "Oats",
            10: "Soybean-notill",
            11: "Soybean-mintill",
            12: "Soybean-clean",
            13: "Wheat",
            14: "Woods",
            15: "Buildings-Grass-Trees-Drives",
            16: "Stone-Steel-Towers",
        }

        for class_id in sorted(stats.keys()):
            info = stats[class_id]
            class_name = class_names.get(class_id, f"Class_{class_id}")
            print(f"{class_id:<8} {info['count']:<10} {info['percentage']:<10.2f}% {class_name}")

        print("-" * 50)

        # 计算不平衡程度
        counts = [info["count"] for info in stats.values()]
        imbalance_ratio = max(counts) / min(counts)
        print(f"不平衡比例: {imbalance_ratio:.2f} (最多/最少)")

        # 计算基尼系数
        total_samples = sum(counts)
        probs = [c / total_samples for c in counts]
        gini = sum(abs(p_i - p_j) for i, p_i in enumerate(probs) for j, p_j in enumerate(probs) if i != j) / (
            2 * len(probs)
        )
        print(f"基尼系数: {gini:.4f} (0=完全平衡, 1=完全不平衡)")

    return stats


# 示例使用代码
if __name__ == "__main__":
    # 示例：演示不同采样策略的使用
    print("高光谱数据平衡采样示例")
    print("=" * 50)

    # 模拟高光谱数据类别分布（Indian Pines的简化版本）
    class_indices_example = {
        1: list(range(100, 200)),  # 100个样本
        2: list(range(200, 350)),  # 150个样本
        3: list(range(350, 400)),  # 50个样本 (少数类)
        4: list(range(400, 550)),  # 150个样本
        5: list(range(550, 650)),  # 100个样本
        6: list(range(650, 850)),  # 200个样本 (多数类)
        7: list(range(850, 900)),  # 50个样本 (少数类)
        8: list(range(900, 1100)),  # 200个样本 (多数类)
    }

    # 创建采样器
    sampler = BalancedSampler(class_indices_example, random_seed=42)

    # 打印原始分布
    print("原始类别分布:")
    original_stats = sampler.get_class_stats()
    for class_id, info in original_stats.items():
        print(f"  类别 {class_id}: {info['count']} 样本 ({info['percentage']:.1f}%)")
    print()

    # 1. 等数量采样
    print("1. 等数量采样 (每类80个样本):")
    equal_sampled = sampler.sample(samples_per_class=80)
    for class_id, indices in equal_sampled.items():
        print(f"  类别 {class_id}: {len(indices)} 样本")
    print()

    # 2. 比例采样
    print("2. 比例采样 (总计400个样本):")
    sampler.strategy = "proportional"
    prop_sampled = sampler.sample(total_samples=400, min_per_class=20)
    for class_id, indices in prop_sampled.items():
        print(f"  类别 {class_id}: {len(indices)} 样本")
    print()

    # 3. 分层采样
    print("3. 分层采样 (70%训练, 20%验证, 10%测试):")
    sampler.strategy = "stratified"
    train_idx, val_idx, test_idx = sampler.sample(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    print("  训练集:")
    for class_id, indices in train_idx.items():
        print(f"    类别 {class_id}: {len(indices)} 样本")
    print("  验证集:")
    for class_id, indices in val_idx.items():
        print(f"    类别 {class_id}: {len(indices)} 样本")
    print("  测试集:")
    for class_id, indices in test_idx.items():
        print(f"    类别 {class_id}: {len(indices)} 样本")
    print()

    # 4. 自适应采样
    print("4. 自适应采样 (总计300个样本):")
    sampler.strategy = "adaptive"
    adaptive_sampled = sampler.sample(target_size=300, balance_strategy="sqrt_inverse")
    for class_id, indices in adaptive_sampled.items():
        print(f"  类别 {class_id}: {len(indices)} 样本")
    print()

    # 5. 组合采样
    print("5. 组合采样 (过采样少数类, 欠采样多数类):")
    sampler.strategy = "combined"
    combined_sampled = sampler.sample(target_size=500)
    for class_id, indices in combined_sampled.items():
        print(f"  类别 {class_id}: {len(indices)} 样本")
    print()

    print("采样策略演示完成！")
    print("在实际使用中，请根据具体的高光谱数据特点选择合适的策略。")

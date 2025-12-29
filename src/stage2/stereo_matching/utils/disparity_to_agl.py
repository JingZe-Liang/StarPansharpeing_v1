"""
US3D数据集视差到AGL高度转换工具

根据US3D数据集特性，提供视差图到地面以上高度(AGL)的转换功能。

US3D数据集使用WorldView-3卫星影像:
- 全色影像GSD: ~0.30米/像素
- 多光谱影像GSD: ~1.30米/像素

Reference:
- US3D Dataset: https://arxiv.org/abs/1803.05895
- SemStereo: Semantic-Constrained Stereo Matching Network for Remote Sensing
"""

import numpy as np
import torch
from torch import Tensor


class US3DDisparityConverter:
    """US3D数据集视差到AGL的转换器"""

    def __init__(
        self,
        gsd: float = 0.30,
        disparity_scale: float = 1.0,
        offset: float = 0.0,
        use_homography: bool = False,
    ):
        """
        初始化转换器

        Args:
            gsd: Ground Sample Distance (米/像素)
                - 0.30: 全色影像 (默认)
                - 1.30: 多光谱影像
            disparity_scale: 视差缩放因子，用于标定
            offset: 高度偏移量(米)，用于校正
            use_homography: 是否使用homography进行更精确的转换
        """
        self.gsd = gsd
        self.disparity_scale = disparity_scale
        self.offset = offset
        self.use_homography = use_homography

    def disparity_to_agl(
        self,
        disparity: Tensor | np.ndarray,
        metadata: dict | None = None,
    ) -> Tensor | np.ndarray:
        """
        将视差图转换为AGL高度图

        Args:
            disparity: 视差图，单位为像素
                - Shape: (H, W) 或 (B, H, W)
            metadata: 可选的元数据字典，包含homography等信息

        Returns:
            agl: Above Ground Level高度图，单位为米
                - Shape同输入
        """
        is_tensor = isinstance(disparity, Tensor)

        # 基本转换公式
        # AGL = disparity * GSD * scale + offset
        agl = disparity * self.gsd * self.disparity_scale + self.offset

        if self.use_homography and metadata is not None:
            # 使用homography进行更精确的转换
            agl = self._apply_homography_correction(agl, metadata)

        return agl

    def _apply_homography_correction(
        self,
        agl: Tensor | np.ndarray,
        metadata: dict,
    ) -> Tensor | np.ndarray:
        """
        使用homography矩阵进行几何校正

        Args:
            agl: 初步计算的AGL
            metadata: 包含left_homography和right_homography的字典

        Returns:
            校正后的AGL
        """
        # 提取homography矩阵
        H_left = np.array(metadata.get("left_homography", np.eye(3)))
        H_right = np.array(metadata.get("right_homography", np.eye(3)))

        # TODO: 实现基于homography的精确转换
        # 这需要更详细的几何模型分析
        # 暂时返回未校正的结果
        return agl

    @staticmethod
    def estimate_gsd_from_homography(homography: list | np.ndarray) -> float:
        """
        从homography矩阵估计GSD

        Args:
            homography: 3x3 homography矩阵

        Returns:
            估计的GSD (米/像素)
        """
        H = np.array(homography)

        # 提取仿射变换部分的尺度
        scale_x = np.linalg.norm(H[0, :2])
        scale_y = np.linalg.norm(H[1, :2])

        # 平均GSD
        gsd = (scale_x + scale_y) / 2

        return gsd

    @staticmethod
    def meters_to_pixels(
        height_meters: float | Tensor | np.ndarray,
        gsd: float = 0.30,
    ) -> float | Tensor | np.ndarray:
        """
        将米制高度转换为像素视差

        Args:
            height_meters: 高度(米)
            gsd: Ground Sample Distance

        Returns:
            视差(像素)
        """
        return height_meters / gsd


def create_default_converter(image_type: str = "panchromatic") -> US3DDisparityConverter:
    """
    创建默认的US3D转换器

    Args:
        image_type: 影像类型
            - "panchromatic" 或 "pan": 全色影像 (GSD=0.30m)
            - "multispectral" 或 "ms": 多光谱影像 (GSD=1.30m)

    Returns:
        配置好的转换器
    """
    gsd_map = {
        "panchromatic": 0.30,
        "pan": 0.30,
        "multispectral": 1.30,
        "ms": 1.30,
    }

    gsd = gsd_map.get(image_type.lower(), 0.30)

    return US3DDisparityConverter(gsd=gsd)


# 便捷函数
def disparity_to_agl_simple(
    disparity: Tensor | np.ndarray,
    gsd: float = 0.30,
) -> Tensor | np.ndarray:
    """
    简单的视差到AGL转换

    Args:
        disparity: 视差图(像素)
        gsd: Ground Sample Distance (米/像素)

    Returns:
        AGL高度图(米)

    Example:
        >>> disparity = model(left, right)['d_final']  # shape: (B, H, W)
        >>> agl = disparity_to_agl_simple(disparity, gsd=0.30)
    """
    return disparity * gsd


if __name__ == "__main__":
    # 测试代码
    print("US3D Disparity to AGL Converter")
    print("=" * 50)

    # 测试估计GSD
    example_homography = [
        [0.0009832890185712088, 0.013094974216889663, 1.234530662837203],
        [-0.014251613426207276, 0.0005306537584558884, 29.560065415214687],
        [-3.0119858610711844e-10, -1.413325221605649e-08, 0.015281562185087659],
    ]

    estimated_gsd = US3DDisparityConverter.estimate_gsd_from_homography(example_homography)
    print(f"\n从Homography估计的GSD: {estimated_gsd:.4f} 米/像素")

    # 测试转换
    converter = create_default_converter("panchromatic")
    print(f"\n使用默认GSD: {converter.gsd} 米/像素")

    # 模拟视差值
    test_disparities = np.array([0, 10, 50, 100, -50])
    test_agls = converter.disparity_to_agl(test_disparities)

    print("\n视差 -> AGL 转换示例:")
    print(f"{'视差(像素)':<12} {'AGL(米)':<12}")
    print("-" * 24)
    for disp, agl in zip(test_disparities, test_agls):
        print(f"{disp:<12.1f} {agl:<12.2f}")

    print("\n说明:")
    print("- 正视差: 物体高于地面")
    print("- 负视差: 物体低于参考面")
    print("- US3D视差范围通常为 [-128, 128] 像素")
    print(f"- 对应AGL范围: [{-128 * converter.gsd:.1f}, {128 * converter.gsd:.1f}] 米")

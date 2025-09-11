from dataclasses import dataclass, field
from typing import Any

from src.utilities.config_utils.to_dataclass import dataclass_from_dict


@dataclass
class ParentConfig:
    """父类配置"""

    in_channels: Any = 16
    out_channels: Any = 16
    channels: int = 128
    channels_mult: list[int] = field(default_factory=lambda: [2, 4, 4])
    num_res_blocks: int = 2
    attn_resolutions: list[int] = field(default_factory=lambda: [])
    dropout: float = 0.0
    resolution: int = 1024
    z_channels: int = 16
    latent_channels: int = 16
    spatial_compression: int = 8
    act_checkpoint: bool = False
    use_residual_factor: bool = False
    # downsampling
    downsample_type: str = "PadConv"
    downsample_shortcut: Any = None  # str
    # patch size, patcher, and blocks
    patch_size: int = 1
    patch_method: str = "haar"
    conv_in_module: str = "conv"
    block_name: str = "res_block"
    attn_type: str = "none"  # 'attn_vanilla' or 'none'
    # if block_name != 'moe', does not use
    moe_n_experts: int = 4
    moe_n_selected: int = 1
    moe_n_shared_experts: int = 1
    hidden_factor: int = 2
    moe_type: str = "tc"
    moe_token_mixer_type: str = "res_block"
    # padding and norm
    padding_mode: str = "zeros"
    norm_type: str = "gn"
    norm_groups: int = 32
    downsample_manually_pad: bool = True
    resample_norm_keep: bool = False


@dataclass
class ChildConfig(ParentConfig):
    """子类配置 - 添加额外字段"""

    per_layer_noise: bool = False
    # 其他新字段
    new_field: str = "hello"


@dataclass
class GrandChildConfig(ChildConfig):
    """孙类配置 - 再添加字段"""

    another_field: int = 42


def test_dataclass_from_dict():
    """测试 dataclass_from_dict 函数"""
    print("=== 测试1: 正常的 dataclass 继承 ===")
    child_normal = ChildConfig()
    print(f"正常创建的子类实例:")
    print(f"  in_channels: {child_normal.in_channels}")
    print(f"  per_layer_noise: {child_normal.per_layer_noise}")
    print(f"  new_field: {child_normal.new_field}")

    print("\n=== 测试2: 使用 dataclass_from_dict ===")
    test_data = {
        "model": {
            "per_layer_noise": True,
            "channels": 256,  # 修改父类字段
            "new_field": "world",  # 修改子类字段
        }
    }

    # 模拟您的使用方式
    try:
        child_from_dict = dataclass_from_dict(
            ChildConfig, test_data["model"], strict=False
        )
        print(f"从字典创建的子类实例:")
        print(f"  in_channels: {child_from_dict.in_channels}")
        print(f"  channels: {child_from_dict.channels}")
        print(f"  per_layer_noise: {child_from_dict.per_layer_noise}")
        print(f"  new_field: {child_from_dict.new_field}")

        # 检查是否有其他父类字段
        print(f"  resolution: {child_from_dict.resolution}")
        print(f"  z_channels: {child_from_dict.z_channels}")

    except Exception as e:
        print(f"错误: {e}")
        import traceback

        traceback.print_exc()

    print("\n=== 测试3: 检查字段是否存在 ===")
    try:
        child_from_dict = dataclass_from_dict(
            ChildConfig, {"per_layer_noise": True}, strict=False
        )
        print(f"只设置 per_layer_noise=True:")
        print(f"  in_channels: {child_from_dict.in_channels}")
        print(f"  channels: {child_from_dict.channels}")
        print(f"  per_layer_noise: {child_from_dict.per_layer_noise}")
        print(f"  resolution: {child_from_dict.resolution}")

    except Exception as e:
        print(f"错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_dataclass_from_dict()

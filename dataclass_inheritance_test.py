from dataclasses import dataclass, field
from typing import Any


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


def test_inheritance():
    """测试 dataclass 继承"""
    print("=== 父类字段 ===")
    parent = ParentConfig()
    for field_name, field_value in parent.__dict__.items():
        print(f"{field_name}: {field_value}")

    print("\n=== 子类字段 ===")
    child = ChildConfig()
    for field_name, field_value in child.__dict__.items():
        print(f"{field_name}: {field_value}")

    print("\n=== 子类是否有父类字段 ===")
    print(f"子类有 in_channels: {hasattr(child, 'in_channels')}")
    print(f"子类有 channels: {hasattr(child, 'channels')}")
    print(f"子类有 per_layer_noise: {hasattr(child, 'per_layer_noise')}")
    print(f"子类有 new_field: {hasattr(child, 'new_field')}")

    print("\n=== 类型检查 ===")
    print(f"子类实例是父类实例: {isinstance(child, ParentConfig)}")
    print(f"父类实例是子类实例: {isinstance(parent, ChildConfig)}")


if __name__ == "__main__":
    test_inheritance()

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CosmosContinuousConfig:
    in_channels: int
    out_channels: int

    attn_resolutions: List[int] = [32]
    channels: int = 128
    channels_mult: List[int] = [2, 4, 4]
    spatial_compression: int = 8
    dropout: float = 0.0
    num_res_blocks: int = 2
    resolution: int = 1024
    patch_size: int = 4
    patch_method: str = "haar"
    latent_channels: int = 16
    z_channels: int = 16
    z_factor: int = 1
    name: str = "CI"
    formulation: str = "AE"
    encoder: str = "Default"
    decoder: str = "Default"
    act_checkpoint: bool = True
    norm_in_quant_conv: bool = False
    uni_tokenizer_path: str = (
        "runs/stage1_cosmos/2025-05-22_23-21-47_cosmos_pretrained_f8c16p4_MMSeg_YREB/ema/tokenizer/model.safetensors"
    )
    wrap_fsdp_last_layer: bool = False
    quantizer_type: Optional[str] = None
    loading_type: str = "pretrained"
    force_not_attn: bool = True
    block_name: str = "resblock"
    hidden_factor: int = 2

    # feature alignment
    hook_for_repa: bool = False

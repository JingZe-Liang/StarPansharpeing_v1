"""
Tokenizer-based Stereo Matching Model.

使用预训练的CosmosHybridTokenizer作为特征提取器，
构建Cost Volume并通过3D CNN聚合预测视差。
"""

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from einops import rearrange
from jaxtyping import Float
from loguru import logger
from omegaconf import OmegaConf
from timm.layers import get_act_layer, get_norm_layer
from timm.models._manipulate import named_apply
from torch import Tensor
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.module import _IncompatibleKeys

from src.stage1.cosmos.cosmos_hybrid import CosmosHybridTokenizer
from src.utilities.config_utils import function_config_to_basic_types

from src.stage2.segmentation.models.adapter import DINOv3EncoderAdapter
from src.stage2.segmentation.models.tokenizer_backbone_adapted import (
    TOKENIZER_INTERACTION_INDEXES,
    HybridTokenizerEncoderAdapter,
)


# =============================================================================
# Cost Volume
# =============================================================================


class CostVolume(nn.Module):
    """构建左右特征的4D Cost Volume。

    将左图特征在不同视差下与右图特征拼接，形成Cost Volume。
    支持正负视差范围。
    """

    def __init__(self, min_disp: int, max_disp: int):
        """初始化Cost Volume模块。

        Args:
            min_disp: 最小视差值（可以为负数）
            max_disp: 最大视差值
        """
        super().__init__()
        self.min_disp = min_disp
        self.max_disp = max_disp
        self.num_disp = max_disp - min_disp

    def forward(
        self,
        left_feat: Float[Tensor, "b c h w"],
        right_feat: Float[Tensor, "b c h w"],
    ) -> Float[Tensor, "b 2c d h w"]:
        """构建Cost Volume。

        Args:
            left_feat: 左图特征 [B, C, H, W]
            right_feat: 右图特征 [B, C, H, W]

        Returns:
            cost_volume: 4D Cost Volume [B, 2C, D, H, W]
        """
        b, c, h, w = left_feat.shape
        cost_volume = []

        for d in range(self.min_disp, self.max_disp):
            if d < 0:
                # 负视差：右图向左移动
                left_slice = left_feat[:, :, :, :d]
                right_slice = right_feat[:, :, :, -d:]
                concat = torch.cat([left_slice, right_slice], dim=1)
                # 右边padding
                concat = F.pad(concat, [0, -d, 0, 0], mode="constant", value=0)
            elif d > 0:
                # 正视差：右图向右移动
                left_slice = left_feat[:, :, :, d:]
                right_slice = right_feat[:, :, :, :-d]
                concat = torch.cat([left_slice, right_slice], dim=1)
                # 左边padding
                concat = F.pad(concat, [d, 0, 0, 0], mode="constant", value=0)
            else:
                # 零视差
                concat = torch.cat([left_feat, right_feat], dim=1)

            cost_volume.append(concat)

        # Stack: [B, 2C, D, H, W]
        cost_volume = torch.stack(cost_volume, dim=2)
        return cost_volume


# =============================================================================
# 3D CNN Aggregation
# =============================================================================


def conv3d_bn(
    in_ch: int,
    out_ch: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: str = "same",
    use_relu: bool = True,
) -> nn.Module:
    """3D卷积 + BatchNorm + ReLU。"""
    layers: list[nn.Module] = [
        nn.Conv3d(in_ch, out_ch, kernel_size, stride, padding=1 if padding == "same" else 0, bias=False),
        nn.BatchNorm3d(out_ch),
    ]
    if use_relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def deconv3d_bn(
    in_ch: int,
    out_ch: int,
    kernel_size: int = 3,
    stride: int = 2,
    use_relu: bool = True,
) -> nn.Module:
    """3D反卷积 + BatchNorm + ReLU。"""
    layers: list[nn.Module] = [
        nn.ConvTranspose3d(in_ch, out_ch, kernel_size, stride, padding=1, output_padding=1, bias=False),
        nn.BatchNorm3d(out_ch),
    ]
    if use_relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class StackedHourglass3D(nn.Module):
    """Stacked Hourglass 3D CNN for Cost Volume aggregation.

    参考PSMNet的设计，使用多个Hourglass模块进行Cost Volume聚合。
    输出3个中间预测用于多尺度监督。
    """

    def __init__(self, in_channels: int, filters: int = 32):
        """初始化Stacked Hourglass。

        Args:
            in_channels: Cost Volume的通道数 (2C)
            filters: 基础滤波器数量
        """
        super().__init__()

        # Initial convolution
        self.conv0_1 = nn.Sequential(
            conv3d_bn(in_channels, filters // 2, 3, 1, "same", True),
            conv3d_bn(filters // 2, filters // 2, 3, 1, "same", True),
        )
        self.conv0_2 = nn.Sequential(
            conv3d_bn(filters // 2, filters // 2, 3, 1, "same", True),
            conv3d_bn(filters // 2, filters // 2, 3, 1, "same", False),
        )

        # Hourglass 1
        self.conv1 = conv3d_bn(filters // 2, filters, 3, 2, "same", True)
        self.conv2 = conv3d_bn(filters, filters, 3, 1, "same", False)
        self.conv3 = conv3d_bn(filters, filters, 3, 2, "same", True)
        self.conv4 = conv3d_bn(filters, filters, 3, 1, "same", True)
        self.conv5 = deconv3d_bn(filters, filters, 3, 2, False)
        self.conv6 = deconv3d_bn(filters, filters // 2, 3, 2, False)

        # Hourglass 2
        self.conv7 = conv3d_bn(filters // 2, filters, 3, 2, "same", True)
        self.conv8 = conv3d_bn(filters, filters, 3, 1, "same", False)
        self.conv9 = conv3d_bn(filters, filters, 3, 2, "same", True)
        self.conv10 = conv3d_bn(filters, filters, 3, 1, "same", True)
        self.conv11 = deconv3d_bn(filters, filters, 3, 2, False)
        self.conv12 = deconv3d_bn(filters, filters // 2, 3, 2, False)

        # Hourglass 3
        self.conv13 = conv3d_bn(filters // 2, filters, 3, 2, "same", True)
        self.conv14 = conv3d_bn(filters, filters, 3, 1, "same", False)
        self.conv15 = conv3d_bn(filters, filters, 3, 2, "same", True)
        self.conv16 = conv3d_bn(filters, filters, 3, 1, "same", True)
        self.conv17 = deconv3d_bn(filters, filters, 3, 2, False)
        self.conv18 = deconv3d_bn(filters, filters // 2, 3, 2, False)

        # Output heads (predict disparity cost)
        self.out1 = nn.Sequential(
            conv3d_bn(filters // 2, filters // 2, 3, 1, "same", True),
            nn.Conv3d(filters // 2, 1, 3, 1, 1, bias=False),
        )
        self.out2 = nn.Sequential(
            conv3d_bn(filters // 2, filters // 2, 3, 1, "same", True),
            nn.Conv3d(filters // 2, 1, 3, 1, 1, bias=False),
        )
        self.out3 = nn.Sequential(
            conv3d_bn(filters // 2, filters // 2, 3, 1, "same", True),
            nn.Conv3d(filters // 2, 1, 3, 1, 1, bias=False),
        )

    def forward(
        self, cost_volume: Float[Tensor, "b c d h w"]
    ) -> tuple[Float[Tensor, "b d h w"], Float[Tensor, "b d h w"], Float[Tensor, "b d h w"]]:
        """Forward pass.

        Args:
            cost_volume: 4D Cost Volume [B, C, D, H, W]

        Returns:
            tuple of 3 cost maps for multi-scale supervision [B, D, H, W]
        """
        # Initial
        x0 = self.conv0_1(cost_volume)
        x1 = self.conv0_2(x0)
        x1 = x1 + x0

        # Hourglass 1
        x2 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x2)
        x3 = self.conv4(x3)
        x4 = self.conv5(x3)
        x4 = x4 + x2
        x5 = self.conv6(x4)
        x5 = x5 + x1

        # Hourglass 2
        x6 = self.conv7(x5)
        x6 = self.conv8(x6)
        x6 = x6 + x4
        x7 = self.conv9(x6)
        x7 = self.conv10(x7)
        x8 = self.conv11(x7)
        x8 = x8 + x2
        x9 = self.conv12(x8)
        x9 = x9 + x1

        # Hourglass 3
        x10 = self.conv13(x9)
        x10 = self.conv14(x10)
        x10 = x10 + x8
        x11 = self.conv15(x10)
        x11 = self.conv16(x11)
        x12 = self.conv17(x11)
        x12 = x12 + x2
        x13 = self.conv18(x12)
        x13 = x13 + x1

        # Output heads
        out1 = self.out1(x5)
        out2 = self.out2(x9)
        out2 = out2 + out1
        out3 = self.out3(x13)
        out3 = out3 + out2

        # Squeeze channel dim and permute: [B, 1, D, H, W] -> [B, D, H, W]
        out1 = out1.squeeze(1)
        out2 = out2.squeeze(1)
        out3 = out3.squeeze(1)

        return out1, out2, out3


# =============================================================================
# Soft ArgMin
# =============================================================================


class SoftArgMin(nn.Module):
    """Soft ArgMin for disparity estimation.

    将Cost Volume转换为概率分布，然后加权求和得到连续视差值。
    """

    def __init__(self, min_disp: int, max_disp: int):
        """初始化Soft ArgMin。

        Args:
            min_disp: 最小视差值
            max_disp: 最大视差值
        """
        super().__init__()
        self.min_disp = min_disp
        self.max_disp = max_disp

        # 预计算视差候选值
        candidates = torch.linspace(float(min_disp), float(max_disp) - 1.0, max_disp - min_disp)
        self.register_buffer("candidates", candidates)

    def forward(self, cost: Float[Tensor, "b d h w"]) -> Float[Tensor, "b 1 h w"]:
        """计算视差。

        Args:
            cost: Cost map [B, D, H, W]

        Returns:
            disparity: 视差图 [B, 1, H, W]
        """
        # Soft ArgMin: 取负数后softmax（成本越低概率越高）
        prob = F.softmax(-cost, dim=1)  # [B, D, H, W]

        # 加权求和
        disp = torch.sum(prob * self.candidates.view(1, -1, 1, 1), dim=1, keepdim=True)
        return disp


# =============================================================================
# Main Model
# =============================================================================


class TokenizerStereoMatching(nn.Module):
    """基于Tokenizer的立体匹配模型。

    使用预训练的CosmosHybridTokenizer作为特征提取器（冻结），
    构建Cost Volume并通过3D CNN聚合预测视差。
    """

    def __init__(
        self,
        cfg: edict,
        min_disp: int = -128,
        max_disp: int = 64,
        hourglass_filters: int = 32,
        feature_scale: int = 4,
    ):
        """初始化立体匹配模型。

        Args:
            cfg: 配置字典，包含tokenizer和adapter配置
            min_disp: 最小视差值
            max_disp: 最大视差值
            hourglass_filters: Hourglass 3D CNN的基础滤波器数
            feature_scale: 特征提取器的下采样倍数
        """
        super().__init__()
        self.cfg = cfg
        self.min_disp = min_disp
        self.max_disp = max_disp
        self.feature_scale = feature_scale

        # 创建特征提取器（左右共享权重）
        self.encoder = self._create_tok_encoder()
        feature_channels = self.encoder.output_channels[-1]  # 使用最深层特征

        # Cost Volume构建
        scaled_min_disp = min_disp // feature_scale
        scaled_max_disp = max_disp // feature_scale
        self.cost_volume = CostVolume(scaled_min_disp, scaled_max_disp)

        # 3D聚合网络
        self.hourglass = StackedHourglass3D(
            in_channels=2 * feature_channels,
            filters=hourglass_filters,
        )

        # Soft ArgMin
        self.soft_argmin = SoftArgMin(min_disp, max_disp)

        # 初始化权重
        self._init_weights()

    def _create_tok_encoder(self) -> DINOv3EncoderAdapter:
        """创建Tokenizer编码器。"""
        cfg = self.cfg
        f_cfg = cfg.tokenizer_feature
        a_cfg = cfg.adapter
        t_cfg = cfg.tokenizer

        model_name = f_cfg.model_name
        interaction_indexes = TOKENIZER_INTERACTION_INDEXES[model_name]
        logger.info(f"Creating tokenizer encoder: {model_name}")

        # 创建Tokenizer backbone
        tok_backbone = CosmosHybridTokenizer.create_model(
            cnn_cfg=t_cfg.cnn_cfg,
            trans_enc_cfg=t_cfg.trans_enc_cfg,
            trans_dec_cfg=t_cfg.trans_dec_cfg,
            distillation_cfg=t_cfg.distill_cfg,
        )

        if cfg.tokenizer_pretrained_path is not None:
            tok_backbone.load_pretrained(cfg.tokenizer_pretrained_path)
            logger.info(f"Loaded tokenizer backbone from: {cfg.tokenizer_pretrained_path}")
        elif cfg._debug:
            logger.warning("Using debug mode, using random weights for tokenizer backbone")
        else:
            raise ValueError("pretrained_path must be specified for tokenizer backbone")

        # 创建Adapter
        dinov3_adapter = HybridTokenizerEncoderAdapter(
            backbone=tok_backbone,
            in_channels=f_cfg.in_channels,
            interaction_indexes=interaction_indexes,
            pretrain_size=512,
            conv_inplane=f_cfg.conv_inplane,
            n_points=4,
            deform_num_heads=f_cfg.deform_num_heads,
            drop_path_rate=f_cfg.drop_path_rate,
            init_values=0.0,
            with_cffn=f_cfg.with_cffn,
            cffn_ratio=f_cfg.cffn_ratio,
            deform_ratio=f_cfg.deform_ratio,
            add_vit_feature=f_cfg.add_vit_feature,
            use_extra_extractor=f_cfg.use_extra_extractor,
            with_cp=f_cfg.with_cp,
        )

        encoder_adapter = DINOv3EncoderAdapter(
            dinov3_adapter=dinov3_adapter,
            target_channels=f_cfg.features_per_stage,
            conv_op=nn.Conv2d,
            norm_op=get_norm_layer(a_cfg.norm),
            nonlin=get_act_layer(a_cfg.act),
            dropout_op=a_cfg.drop,
            conv_bias=a_cfg.conv_bias,
        )

        logger.info("Created tokenizer encoder adapter for stereo matching.")
        return encoder_adapter

    def _init_weights(self) -> None:
        """初始化非backbone权重。"""

        def _apply(module: nn.Module, name: str) -> None:
            if "backbone" in name:
                return

            if hasattr(module, "init_weights"):
                module.init_weights()
            elif isinstance(module, _ConvNd):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        named_apply(_apply, self)
        logger.info("[TokenizerStereoMatching]: Initialized weights (except backbone).")

    def extract_features(self, img: Float[Tensor, "b c h w"]) -> Float[Tensor, "b c_feat h_feat w_feat"]:
        """提取图像特征。

        Args:
            img: 输入图像 [B, C, H, W]

        Returns:
            features: 提取的特征 [B, C_feat, H/scale, W/scale]
        """
        skips, _ = self.encoder(img)
        # 使用最深层特征（分辨率最低）
        return skips[-1]

    def forward(
        self,
        left_img: Float[Tensor, "b c h w"],
        right_img: Float[Tensor, "b c h w"],
    ) -> Union[Float[Tensor, "b 1 h w"], tuple[Float[Tensor, "b 1 h w"], ...]]:
        """前向传播。

        Args:
            left_img: 左图像 [B, C, H, W]
            right_img: 右图像 [B, C, H, W]

        Returns:
            训练时返回3个视差图（多尺度监督），推理时返回最终视差图
        """
        h, w = left_img.shape[-2:]

        # 提取特征（共享权重）
        left_feat = self.extract_features(left_img)
        right_feat = self.extract_features(right_img)

        # 构建Cost Volume
        cost_volume = self.cost_volume(left_feat, right_feat)

        # 3D聚合
        cost1, cost2, cost3 = self.hourglass(cost_volume)

        # 上采样到原始分辨率
        cost1 = F.interpolate(
            cost1.unsqueeze(1), size=(self.max_disp - self.min_disp, h, w), mode="trilinear", align_corners=False
        ).squeeze(1)
        cost2 = F.interpolate(
            cost2.unsqueeze(1), size=(self.max_disp - self.min_disp, h, w), mode="trilinear", align_corners=False
        ).squeeze(1)
        cost3 = F.interpolate(
            cost3.unsqueeze(1), size=(self.max_disp - self.min_disp, h, w), mode="trilinear", align_corners=False
        ).squeeze(1)

        # Soft ArgMin得到视差
        disp1 = self.soft_argmin(cost1)
        disp2 = self.soft_argmin(cost2)
        disp3 = self.soft_argmin(cost3)

        if self.training:
            return disp1, disp2, disp3
        else:
            return disp3

    def parameters(self, *args, **kwargs):  # type: ignore
        """只返回非backbone参数。"""
        for name, param in self.named_parameters(*args, **kwargs):
            if "backbone" in name:
                param.requires_grad = False
            yield param

    def named_parameters(self, *args, **kwargs):  # type: ignore
        """只返回非backbone参数。"""
        for name, param in super().named_parameters(*args, **kwargs):
            if "backbone" in name:
                param.requires_grad = False
            yield name, param

    def _filter_backbone_params(self, k: str) -> bool:
        return "backbone" in k

    def state_dict(self, *args, **kwargs):  # type: ignore
        """排除backbone参数。"""
        state_dict = super().state_dict(*args, **kwargs)
        backbone_keys = [k for k in state_dict.keys() if self._filter_backbone_params(k)]
        for k in backbone_keys:
            del state_dict[k]
        logger.info(f"Get {len(state_dict)} parameters in state_dict (backbone removed).")
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = False, *args, **kwargs):  # type: ignore
        """加载状态字典，忽略backbone。"""
        backbone_keys = [k for k in state_dict.keys() if self._filter_backbone_params(k)]
        for k in backbone_keys:
            del state_dict[k]
        missing_ks, unexpected_ks = super().load_state_dict(state_dict, strict=strict)

        missing_ks = [k for k in missing_ks if not self._filter_backbone_params(k)]
        unexpected_ks = [k for k in unexpected_ks if not self._filter_backbone_params(k)]

        if len(missing_ks) > 0:
            logger.warning(f"Missing Keys: {missing_ks}")
        if len(unexpected_ks) > 0:
            logger.warning(f"Unexpected Keys: {unexpected_ks}")

        return _IncompatibleKeys(missing_ks, unexpected_ks)


# =============================================================================
# Factory
# =============================================================================


def _create_default_cfg():
    """创建默认配置，参考hybrid_distillation_f16_config。"""
    # CNN模型配置
    cnn_cfg = dict(
        model=dict(
            resolution=512,
            in_channels=512,
            out_channels=512,
            z_channels=768,
            latent_channels=32,  # Must match pretrained model
            channels=128,
            channels_mult=[2, 4, 4],
            num_res_blocks=2,
            attn_resolutions=[],
            dropout=0.0,
            spatial_compression=8,
            patch_size=1,
            block_name="res_block",
            norm_type="rmsnorm2d",
            norm_groups=32,
            adaptive_mode="interp",
            downsample_kwargs=dict(padconv_use_manually_pad=False),
            upsample_kwargs=dict(interp_type="nearest_interp"),
            per_layer_noise=False,
        ),
        quantizer_type=None,
        vf_on_z_or_module="z",
        use_repa_loss=False,
        dino_feature_dim=1024,
        decoder_type="default",
        use_channel_drop=False,
        channel_drop_config=dict(
            drop_type=[16, 32, 48],
            max_channels=64,
            drop_prob=0.5,
        ),
        loading_type="hybrid_pretrained",
        uni_path="",
    )

    # Transformer编码器配置
    trans_enc_cfg = dict(
        embed_dim=1152,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        patch_size=2,
        norm_layer="flarmsnorm",
        pos_embed="learned",
        pos_embed_grid_size=[32, 32],
        rope_type="axial",
        img_size=32,
        in_chans=768,  # Must match z_channels from CNN encoder
        out_chans=768,  # Must match pretrained model
        unpatch_size=2,
        reg_tokens=4,
        compile_model=False,
        attn_type="gated",  # Must match pretrained model config
        pretrained_type=None,  # ["ijepa"],
    )

    trans_dec_cfg = None

    distill_cfg = dict(
        dino_feature_dim=1152,
        semantic_feature_dim=1152,
        cache_layers=dict(low_level=[0, 1, 2, -1], semantic=[2, 5, 8, 11]),
    )

    tokenizer_cfg = OmegaConf.create(
        dict(
            cnn_cfg=cnn_cfg,
            trans_enc_cfg=trans_enc_cfg,
            trans_dec_cfg=trans_dec_cfg,
            distillation_cfg=distill_cfg,
            hybrid_tokenizer_cfg=dict(
                latent_bottleneck_type="before_semantic",
                latent_straight_through_skip=True,
            ),
        )
    )

    tokenizer_feature_cfg = dict(
        pretrained_path=None,
        features_per_stage=[512, 512, 512, 512],
        model_name="hybrid_tokenizer_b16",
        pretrained_size=512,
        in_channels=3,
        conv_inplane=64,
        drop_path_rate=0.3,
        with_cffn=True,
        cffn_ratio=0.25,
        deform_num_heads=16,
        deform_ratio=0.5,
        add_vit_feature=True,
        use_extra_extractor=True,
        with_cp=True,
    )

    adapter_cfg = dict(
        adapter_type="default",
        latent_width=64,
        n_conv_per_stage=1,
        depth_per_stage=1,
        norm="layernorm2d",
        act="gelu",
        drop=0.0,
        act_first=False,
        conv_bias=False,
        block_types=["nat", "nat", "mbconv", "mbconv"],
    )

    stereo_cfg = dict(
        min_disp=-128,
        max_disp=64,
        hourglass_filters=32,
        feature_scale=4,
    )

    # 组合配置
    cfg = OmegaConf.create()
    cfg.tokenizer = tokenizer_cfg
    cfg.tokenizer_feature = OmegaConf.create(tokenizer_feature_cfg)
    cfg.adapter = OmegaConf.create(adapter_cfg)
    cfg.stereo = OmegaConf.create(stereo_cfg)
    cfg.tokenizer_pretrained_path = None
    cfg._debug = False

    return cfg


@function_config_to_basic_types
def create_stereo_model(
    cfg: edict | None = None,
    **overrides,
) -> TokenizerStereoMatching:
    """创建立体匹配模型。

    Args:
        cfg: 配置字典，如果为None则使用默认配置
        **overrides: 配置覆盖

    Returns:
        TokenizerStereoMatching模型实例
    """
    if cfg is None:
        cfg = _create_default_cfg()

    if overrides:
        cfg_omega = OmegaConf.create(cfg)
        cfg_omega.merge_with(OmegaConf.create(overrides))
        cfg = edict(OmegaConf.to_container(cfg_omega, resolve=True))

    stereo_cfg = cfg.stereo

    return TokenizerStereoMatching(
        cfg=cfg,
        min_disp=stereo_cfg.min_disp,
        max_disp=stereo_cfg.max_disp,
        hourglass_filters=stereo_cfg.hourglass_filters,
        feature_scale=stereo_cfg.feature_scale,
    )


# =============================================================================
# Test
# =============================================================================


def __test_model():
    """测试模型前向传播。"""
    cfg = _create_default_cfg()
    cfg._debug = True
    # 在debug模式下使用随机权重
    cfg.tokenizer_pretrained_path = None

    model = TokenizerStereoMatching(
        cfg=cfg,
        min_disp=-128,
        max_disp=64,
    ).cuda()

    model.eval()

    # 随机输入
    left_img = torch.randn(1, 3, 256, 256).cuda()
    right_img = torch.randn(1, 3, 256, 256).cuda()

    with torch.autocast("cuda", torch.bfloat16):
        with torch.no_grad():
            disp = model(left_img, right_img)

    print(f"Output shape: {disp.shape}")  # Expected: [1, 1, 256, 256]
    print(f"Disparity range: [{disp.min():.2f}, {disp.max():.2f}]")


if __name__ == "__main__":
    """
    python -m src.stage2.stereo_matching.tokenizer_hybrid_stero_matching
    """
    with logger.catch():
        __test_model()

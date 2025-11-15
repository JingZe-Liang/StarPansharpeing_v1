import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from loguru import logger
from omegaconf import OmegaConf
from torch import Tensor
from torch.nn.modules.conv import _ConvNd

from src.stage1.cosmos.cosmos_hybrid import CosmosHybridTokenizer
from src.utilities.config_utils import (
    dataclass_from_dict,
    function_config_to_basic_types,
)

from ...layers import DINOv3_Adapter
from .adapter import DINOv3EncoderAdapter, UNetDecoder

TOKENIZER_INTERACTION_INDEXES = {
    "hybrid_tokenizer_b16": [3, 6, 8, 11],
}

# *==============================================================
# * Configurations
# *==============================================================


def _create_default_cfg():
    # cm244c384
    _cnn_model_str = (
        "resolution=256 in_channels=384 z_channels=512 latent_channels=64 channels=128 "
        "channels_mult=[2,4,4] num_res_blocks=2 attn_resolutions=[] "
        "dropout=0.0 spatial_compression=8 patch_size=1 block_name=res_block "
        "norm_type=gn norm_groups=32 adaptive_mode=interp"
    )
    _cnn_model_cfg = OmegaConf.from_dotlist(_cnn_model_str.split(" "))
    _cnn_str = (
        "quantizer_type=null vf_on_z_or_module=z use_repa_loss=false "
        "dino_feature_dim=1024 cache_type=h"
    )
    _cnn_cfg = OmegaConf.from_dotlist(_cnn_str.split(" "))
    tokenizer_cfg = OmegaConf.create({"cnn_cfg": {"model": _cnn_model_cfg, **_cnn_cfg}})

    # h16d12c1024
    _trans_enc_str = (
        "embed_dim=1024 depth=12 num_heads=16 mlp_ratio=4.0 qkv_bias=true "
        "patch_size=2 norm_layer=rmsnorm pos_embed=learned "
        "pos_embed_grid_size=[32,32] img_size=32 in_chans=512 "
        "out_chans=512 unpatch_size=1 reg_tokens=0"
    )
    _trans_enc_cfg = OmegaConf.from_dotlist(_trans_enc_str.split(" "))
    tokenizer_cfg.trans_enc_cfg = _trans_enc_cfg

    # h16d12c1024
    _trans_dec_str = (
        "embed_dim=1024 depth=12 num_heads=16 mlp_ratio=4.0 qkv_bias=true "
        "patch_size=1 norm_layer=rmsnorm pos_embed=learned "
        "pos_embed_grid_size=[32,32] img_size=32 in_chans=512 "
        "out_chans=512 unpatch_size=2 reg_tokens=0"
    )
    _trans_dec_cfg = OmegaConf.from_dotlist(_trans_dec_str.split(" "))
    tokenizer_cfg.trans_dec_cfg = _trans_dec_cfg

    _distill_str = "dino_feature_dim=1024 semantic_feature_dim=1024"
    distill_cfg = OmegaConf.from_dotlist(_distill_str.split(" "))
    tokenizer_cfg.distill_cfg = distill_cfg

    # * ----------------------------------------------

    _tokenizer_feature_str = (
        "pretrained_path=null features_per_stage=[256,256,256,256] "
        "model_name=hybrid_tokenizer_b16 "
        # encoder
        "pretrained_size=512 conv_inplane=64 drop_path_rate=0.3 with_cffn=true "
        "cffn_ratio=0.25 deform_num_heads=16 deform_ratio=0.5 add_vit_feature=true "
        "use_extra_extractor=true with_cp=true "
    )
    tokenizer_feature_cfg = OmegaConf.from_dotlist(_tokenizer_feature_str.split(" "))

    # * ----------------------------------------------

    _adapter_str = (
        "adapter_type=default latent_width=64 n_conv_per_stage=1 "
        "depth_per_stage=1 norm=layernorm2d act=gelu drop=0.0 "
        "act_first=false conv_bias=false"
    )
    adapter_cfg = OmegaConf.from_dotlist(_adapter_str.split(" "))

    # * ----------------------------------------------

    _unet_str = (
        "input_channels=3 num_classes=7 deep_supervision=false n_stages=4 "
        "use_latent=true ensure_rgb_type=null _debug=false"
    )
    unet_cfg = OmegaConf.from_dotlist(_unet_str.split(" "))

    # composite the all configs
    cfg = OmegaConf.create()
    cfg.tokenizer = tokenizer_cfg
    cfg.tokenizer_feature = tokenizer_feature_cfg
    cfg.adapter = adapter_cfg
    cfg.merge_with(unet_cfg)

    return cfg


# *==============================================================
# * Tokenizer Unet
# *==============================================================


class HybridTokenizerEncoderAdapter(DINOv3_Adapter):
    # def __init__(
    #     self,
    #     backbone: CosmosHybridTokenizer,  # Dinov3 backbone original
    #     interaction_indexes=[9, 19, 29, 39],
    #     pretrain_size=512,
    #     conv_inplane=64,
    #     n_points=4,
    #     deform_num_heads=16,
    #     drop_path_rate=0.3,
    #     init_values=0.0,
    #     with_cffn=True,
    #     cffn_ratio=0.25,
    #     deform_ratio=0.5,
    #     add_vit_feature=True,
    #     use_extra_extractor=True,
    #     dw_ratios=[2, 1, 0.5],
    #     with_cp=True,
    #     use_bn=True,
    # ):
    #     ...

    def _setup_backbone(
        self,
        backbone: CosmosHybridTokenizer,
        pretrain_size: int,
        interaction_indexes: list[int],
        add_vit_feature=True,
        freeze_backbone=True,
    ):
        self.backbone = backbone
        if freeze_backbone:
            self.backbone.requires_grad_(False)

        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        self.embed_dim = backbone.semantic_enc_transformer.embed_dim
        self.freeze_backbone = freeze_backbone
        self.patch_size = 16  # TODO: in config

        # fmt: off
        logger.info("[Tokenizer backbone adapted]: embed dim", self.embed_dim)
        logger.info("[Tokenizer backbone adapted]: interaction_indexes", self.interaction_indexes)
        # fmt: on

        return self.embed_dim

    def _forward_backbone_intermediate_features(self, x):
        # NOTE: cls feature is None
        grad_ctx = torch.no_grad if self.freeze_backbone else torch.enable_grad
        with torch.autocast("cuda", torch.bfloat16):
            with grad_ctx():
                final_latent = self.backbone.encode(x)
        all_layers = self.backbone.sem_z  # get from semantic encoder

        # reorg all_layers
        assert all_layers is not None, "all_layers is None"
        assert len(all_layers) == len(self.interaction_indexes), (
            f"{len(all_layers)=} != {len(self.interaction_indexes)=}"
        )

        # None stands for cls feature
        all_layers = [
            [rearrange(all_layers[i], "b c h w -> b (h w) c"), None]
            for i in range(len(self.interaction_indexes))
        ]

        return all_layers, final_latent


class TokenizerHybridUNet(nn.Module):
    """
    U-Net with DINOv3_Adapter as encoder, compatible with PlainConvUNet interface
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_cfg = cfg.tokenizer
        self.tok_f_cfg = cfg.tokenizer_feature
        self.adapter_cfg = cfg.adapter

        self.force_rgb_input = cfg.ensure_rgb_type is not None
        self.use_latent = cfg.use_latent
        self._debug = cfg._debug

        # Validate parameters
        n_conv_per_stage = self.adapter_cfg.n_conv_per_stage
        n_stages = cfg.n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        # if isinstance(n_conv_per_stage_decoder, int):
        #     n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        # Ensure we have 4 stages to match adapter output
        if cfg.n_stages != 4:
            logger.error(
                f"Warning: Adapter outputs 4 scales, but n_stages={n_stages}. Adjusting to 4."
            )
            raise ValueError("n_stages must be 4")
            # n_stages = 4
            # if isinstance(self.tok_cfg.feature_per_stage, int):
            #     self.cfg.tok.features_per_stages = [
            #         self.tok_cfg.features_per_stage * (2**i) for i in range(4)
            #     ]
            # elif len(self.tok_cfg.features_per_stage) != 4:
            #     # Adjust features_per_stage to 4 stages
            #     base_features = (
            #         self.tok_cfg.features_per_stage[0]
            #         if self.tok_cfg.features_per_stage
            #         else 32
            #     )
            #     self.cfg.dino.features_per_stage = [
            #         base_features * (2**i) for i in range(4)
            #     ]

        # Create tokenzier encoder
        self.encoder = self._create_tok_encoder()

        # Create decoder
        self.decoder = UNetDecoder(
            self.encoder,
            cfg.num_classes,  # segmentation classes
            self.adapter_cfg.latent_width,
            self.adapter_cfg.n_conv_per_stage,
            self.adapter_cfg.depth_per_stage,
            nonlin_first=self.adapter_cfg.act_first,
            deep_supervision=cfg.deep_supervision,
        )
        logger.info(f"Created Unet decoder with 4 stages")

        # Init weights
        self.apply(self.initialize)

    def _create_tok_encoder(self):
        """Create DINOv3 encoder"""
        cfg = self.cfg
        f_cfg = self.tok_f_cfg
        a_cfg = self.adapter_cfg
        t_cfg = self.tok_cfg

        # Get model information
        model_name = f_cfg.model_name
        interaction_indexes = TOKENIZER_INTERACTION_INDEXES[model_name]
        logger.info(f"Creating tokenizer encoder: {model_name}")

        # Load DINOv3 backbone
        tok_backbone = CosmosHybridTokenizer.create_model(
            cnn_cfg=t_cfg.cnn_cfg,
            trans_enc_cfg=t_cfg.trans_enc_cfg,
            trans_dec_cfg=t_cfg.trans_dec_cfg,
            distillation_kwargs=t_cfg.distill_cfg,
        )
        if self.tok_f_cfg.pretrained_path is not None:
            tok_backbone = tok_backbone.load_pretrained(self.tok_f_cfg.pretrained_path)
            logger.info(
                f"Loaded tokenizer backbone from: {self.tok_f_cfg.pretrained_path}"
            )
        elif cfg._debug:
            logger.warning(
                f"Using debug mode, using random weights for tokenizer backbone"
            )
        else:
            raise ValueError("pretrained_path must be specified for tokenizer backbone")

        # Create DINOv3_Adapter using correct interaction layer indices
        dinov3_adapter = HybridTokenizerEncoderAdapter(
            backbone=tok_backbone,
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
            norm_op=a_cfg.norm,
            dropout_op=a_cfg.drop,
            nonlin=a_cfg.act,
            conv_bias=a_cfg.conv_bias,
        )
        logger.info("Created tokenizer encoder adapter.")

        return encoder_adapter

    def _ensure_rgb_input(
        self, x: Float[Tensor, "b c h w"], larger_then_3_op: str | list[int] = "mean"
    ):
        if not self.force_rgb_input:
            return x

        C = x.size(1)
        if C == 1:
            x = x.repeat(1, 3, 1, 1)
        elif C != 3:
            if C < 3:
                x = x.repeat(1, 3 // C + (1 if 3 % C != 0 else 0), 1, 1)[:, :3, :, :]
            elif larger_then_3_op == "first_3":
                x = x[:, :3, :, :]
            elif larger_then_3_op == "mean":
                channels_per_group = C // 3
                remainder = C % 3
                groups = []
                start = 0
                for i in range(3):
                    group_size = channels_per_group + (1 if i < remainder else 0)
                    end = start + group_size
                    groups.append(x[:, start:end].mean(dim=1, keepdim=True))
                    start = end
                x = torch.cat(groups, dim=1)
            elif isinstance(larger_then_3_op, (list, tuple)):
                x = x[:, larger_then_3_op]
            else:
                raise ValueError(
                    f"Unknown operation for C > 3 ({C=}): {larger_then_3_op}"
                )
        elif C == 3:
            pass
        else:
            raise ValueError(f"Unexpected number of channels: {C}")

        return x

    def forward(
        self,
        x: Float[Tensor, "b c h w"],
        # cond: Float[Tensor, "b latent_c latent_h latent_w"] | None = None,
    ):
        x = self._ensure_rgb_input(x)
        skips, final_h = self.encoder(x)
        output = self.decoder(skips, final_h)
        return output

    def initialize(self, module) -> None:
        if isinstance(module, _ConvNd):
            if module.weight.requires_grad:
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @classmethod
    def create_model(cls, cfg=_create_default_cfg(), overrides: dict | None = None):
        """
        Create DinoUNet instance from network configuration dictionary
        """
        if overrides is not None:
            cfg = cfg.merge_with(overrides)

        return cls(cfg)


def __test_model():
    cfg = _create_default_cfg()
    cfg._debug = True
    unet = TokenizerHybridUNet(cfg)
    x = torch.randn(2, 3, 256, 256)
    y = unet(x)
    print(y.shape)


if __name__ == "__main__":
    """
    python -m src.stage2.segmentation.models.tokenizer_backbone_adapted
    """
    with logger.catch():
        __test_model()

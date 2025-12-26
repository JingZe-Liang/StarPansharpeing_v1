import torch
import torch.nn as nn
import torch.nn.functional as F
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
    cfg_dict = dict(
        tokenizer=dict(
            cnn_cfg=dict(
                model=dict(
                    resolution=512,
                    in_channels=512,
                    out_channels=512,
                    z_channels=512,
                    latent_channels=32,
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
                ),
                quantizer_type=None,
                vf_on_z_or_module="z",
                use_repa_loss=False,
                dino_feature_dim=1024,
            ),
            trans_enc_cfg=dict(
                embed_dim=1024,
                depth=12,
                num_heads=16,
                mlp_ratio=4.0,
                qkv_bias=True,
                patch_size=2,
                norm_layer="layernorm",
                pos_embed="learned",
                rope_type="axial",
                pos_embed_grid_size=[32, 32],
                img_size=32,
                in_chans=512,
                out_chans=512,
                unpatch_size=1,
                reg_tokens=0,
            ),
            trans_dec_cfg=dict(
                embed_dim=1024,
                depth=12,
                num_heads=16,
                mlp_ratio=4.0,
                qkv_bias=True,
                patch_size=1,
                norm_layer="layernorm",
                pos_embed="learned",
                rope_type="axial",
                pos_embed_grid_size=[32, 32],
                img_size=32,
                in_chans=512,
                out_chans=512,
                unpatch_size=2,
                reg_tokens=0,
            ),
            distill_cfg=dict(
                dino_feature_dim=1024,
                semantic_feature_dim=1024,
            ),
        ),
        tokenizer_feature=dict(
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
        ),
        adapter=dict(
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
        ),
        tokenizer_pretrained_path=None,
        input_channels=155,
        num_classes=2,
        deep_supervision=False,
        n_stages=4,
        use_latent=True,
        ensure_rgb_type=None,
        _debug=False,
    )
    return OmegaConf.create(cfg_dict)


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
        # Load pretrained weights

        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        self.embed_dim = backbone.semantic_enc_transformer.embed_dim
        self.freeze_backbone = freeze_backbone
        self.patch_size = 16  # TODO: in config

        logger.info(f"[Tokenizer backbone adapted]: embed dim={self.embed_dim}")
        logger.info(f"[Tokenizer backbone adapted]: interaction_indexes={self.interaction_indexes}")

        return self.embed_dim

    def _forward_backbone_intermediate_features(self, x):
        # NOTE: cls feature is None
        grad_ctx = torch.no_grad if self.freeze_backbone else torch.enable_grad
        with torch.autocast("cuda", torch.bfloat16):
            with grad_ctx():
                enc_out = self.backbone.encode(x, get_intermediate_features=True)
                all_layers = enc_out.sem_z
                final_latent = enc_out.latent

        # reorganize all_layers
        assert all_layers is not None, "all_layers is None"
        assert len(all_layers) == len(self.interaction_indexes), (
            f"{len(all_layers)=} != {len(self.interaction_indexes)=}"
        )

        # None stands for cls feature
        all_layers = [
            [rearrange(all_layers[i], "b c h w -> b (h w) c"), None] for i in range(len(self.interaction_indexes))
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
            logger.error(f"Warning: Adapter outputs 4 scales, but n_stages={n_stages}. Adjusting to 4.")
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
            block_types=self.adapter_cfg.block_types,
        )
        logger.info(f"Created Unet decoder with 4 stages")

        # Init weights
        self.init_weights()

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
            distillation_cfg=t_cfg.distill_cfg,
        )
        if self.cfg.tokenizer_pretrained_path is not None:
            tok_backbone.load_pretrained(self.cfg.tokenizer_pretrained_path)
            logger.info(f"Loaded tokenizer backbone from: {self.cfg.tokenizer_pretrained_path}")
        elif cfg._debug:
            logger.warning(f"Using debug mode, using random weights for tokenizer backbone")
        else:
            raise ValueError("pretrained_path must be specified for tokenizer backbone")

        # Create DINOv3_Adapter using correct interaction layer indices
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
        logger.info("Created tokenizer encoder adapter.")

        return encoder_adapter

    def forward(self, x: Float[Tensor, "b c h w"]):
        skips, final_h = self.encoder(x)
        output = self.decoder(skips, final_h)
        return output

    def init_weights(self) -> None:
        def _apply(module, name: str):
            if "backbone" not in name:  # skip the pretrained weights
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
                elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                    nn.init.ones_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

            bn_types = (nn.BatchNorm2d, nn.SyncBatchNorm)
            if isinstance(module, bn_types):
                logger.warning(f"Found BN in model: {name}.")

        named_apply(_apply, self)
        logger.info(f"[TokenizerHybridUNet]: Initialized weights (except backbone).")

    @classmethod
    @function_config_to_basic_types
    def create_model(cls, cfg=_create_default_cfg(), **overrides):
        """
        Create DinoUNet instance from network configuration dictionary
        """
        if overrides is not None:
            cfg.merge_with(overrides)
        if cfg.tokenizer_pretrained_path is None:
            logger.warning(f"[TokenizerHybridUNet]: No pretrained weights provided. Using random initialization.")

        return cls(cfg)

    def parameters(self, *args, **kwargs):
        for name, param in self.named_parameters(*args, **kwargs):
            if "backbone" in name:
                param.requires_grad = False
            yield param

    def named_parameters(self, *args, **kwargs):
        for name, param in super().named_parameters(*args, **kwargs):
            if "backbone" in name:
                param.requires_grad = False
            yield name, param

    def _filter_backbone_params(self, k: str):
        return "backbone" in k

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        # Remove backbone parameters from state dict
        backbone_keys = [k for k in state_dict.keys() if self._filter_backbone_params(k)]
        for k in backbone_keys:
            del state_dict[k]
        logger.info(f"Get {len(state_dict)} parameters in state_dict (backbone removed).")
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = False, return_not_loaded_keys=False, *args, **kwargs):
        # Filter out backbone parameters from state dict
        backbone_keys = [k for k in state_dict.keys() if self._filter_backbone_params(k)]
        for k in backbone_keys:
            del state_dict[k]
        missing_ks, unexpected_ks = super().load_state_dict(state_dict, strict=strict)

        # remove the pretrained backbone keys from missing/unexpected keys
        missing_ks = [k for k in missing_ks if not self._filter_backbone_params(k)]
        unexpected_ks = [k for k in unexpected_ks if not self._filter_backbone_params(k)]

        if len(missing_ks) > 0:
            logger.warning(f"Missing Keys: {missing_ks}")
        if len(unexpected_ks) > 0:
            logger.warning(f"Unexpected Keys: {unexpected_ks}")

        # Ensure accelerate state loading
        return _IncompatibleKeys(missing_ks, unexpected_ks) if return_not_loaded_keys else _IncompatibleKeys([], [])


def __test_model():
    from src.data.hyperspectral_loader import get_fast_test_hyperspectral_data

    dl = get_fast_test_hyperspectral_data("fmow_RGB")
    sample = next(iter(dl))
    x = sample["img"].cuda()
    x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

    cfg = _create_default_cfg()
    cfg._debug = True
    cfg.tokenizer_pretrained_path = "runs/pretrained_model_ckpts/NaflexHybridTokenizerLayerNorm.safetensors"
    unet = TokenizerHybridUNet(cfg).cuda()
    unet.eval()
    with torch.autocast("cuda", torch.bfloat16):
        with torch.no_grad():
            y = unet(x)
    print(y.shape)

    #         y_recon = unet.encoder.dinov3_adapter.backbone(x)
    # from torchmetrics import PeakSignalNoiseRatio

    # logger.info(f"Input shape: {x.shape}")
    # logger.info(f"Output shape: {y_recon.shape}")

    # psnr_fn = PeakSignalNoiseRatio(data_range=1.0).cuda()
    # psnr = psnr_fn((y_recon + 1) / 2, (x + 1) / 2)
    # logger.info(f"Reconstruction PSNR: {psnr}")


if __name__ == "__main__":
    """
    python -m src.stage2.segmentation.models.tokenizer_backbone_adapted
    """
    with logger.catch():
        __test_model()

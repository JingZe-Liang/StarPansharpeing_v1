import math
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
from fvcore.nn.parameter_count import parameter_count


from src.stage1.cosmos.cosmos_hybrid import CosmosHybridTokenizer
from src.utilities.config_utils import (
    dataclass_from_dict,
    function_config_to_basic_types,
)

from ...layers import DINOv3_Adapter
from .adapter import DINOv3EncoderAdapter, UNetDecoder

logger = logger.bind(_name_="seg_adapted_decoder")

TOKENIZER_INTERACTION_INDEXES = {
    "hybrid_tokenizer_b16": [3, 6, 8, 11],
}

# *==============================================================
# * Configurations
# *==============================================================


def _create_default_cfg():
    yaml_string = """
    tokenizer:
        cnn_cfg:
            model:
                resolution: 1024
                in_channels: 512
                out_channels: 512
                z_channels: 768
                latent_channels: 32
                channels: 128
                channels_mult: [2, 4, 4]
                num_res_blocks: 2
                attn_resolutions: []
                dropout: 0.0
                spatial_compression: 8
                patch_size: 1
                block_name: res_block
                norm_type: rmsnorm2d
                act_type: silu
                norm_groups: 32
                adaptive_mode: interp
                downsample_kwargs:
                    padconv_use_manually_pad: false
                upsample_kwargs:
                    interp_type: nearest_interp
            quantizer_type: null
            vf_on_z_or_module: z
            use_repa_loss: false
            dino_feature_dim: 1024
        trans_enc_cfg:
            embed_dim: 1152
            depth: 24
            num_heads: 16
            mlp_ratio: 4.0
            qkv_bias: true
            patch_size: 2
            norm_layer: flarmsnorm
            pos_embed: learned
            rope_type: axial
            pos_embed_grid_size: [32, 32]
            img_size: 32
            in_chans: 768
            out_chans: 768
            unpatch_size: 2
            reg_tokens: 4
            attn_type: gated
        trans_dec_cfg: null
        distill_cfg:
            dino_feature_dim: 1024
            semantic_feature_dim: 1024
            cache_layers:
                low_level: [0, 1, 2, -1]
                semantic: [5,11,14,19]  #[5, 11, 17, 23]
        hybrid_tokenizer_cfg:
            latent_bottleneck_type: before_semantic
            latent_straight_through_skip: true
    tokenizer_feature:
        pretrained_path: null
        model_name: hybrid_tokenizer_b16
        pretrained_size: 512
        in_channels: 3
        interp_ratio: null
        interaction_indexes: [1,2,4,5]  # [2,3,8,16]
        layer_in_channels: [512, 512, 1152, 1152]
        features_per_stage: [256,384,512, 512]
        conv_inplane: 64
        drop_path_rate: 0.3
        with_cffn: true
        cffn_ratio: 0.25
        deform_num_heads: 16
        deform_ratio: 0.5
        add_vit_feature: true
        use_extra_extractor: true
        with_cp: true
        select_in_all_layers: true
        extractor_type: deform_attention  # [sla, convnext, deform_attention]
        extractor_kwargs: {}
    adapter:
        latent_width: 32
        n_conv_per_stage: 2
        depth_per_stage: 2
        norm: layernorm2d
        act: gelu
        drop: 0.0
        act_first: false
        conv_bias: false
        block_types: [mbconv, mbconv, mbconv, mbconv]
    tokenizer_pretrained_path: null
    input_channels: 155
    num_classes: 2
    deep_supervision: false
    n_stages: 4
    use_latent: true
    freeze_tokenizer: true
    ensure_rgb_type: null
    _debug: false
    """
    return OmegaConf.create(yaml_string)


# *==============================================================
# * Tokenizer Unet
# *==============================================================


class HybridTokenizerEncoderAdapter(DINOv3_Adapter):
    def __init__(
        self,
        backbone: CosmosHybridTokenizer,  # Dinov3 backbone original
        select_in_all_layers: bool = False,
        interp_ratio: None | list[float] = None,
        layer_in_channels: list[int] | None = None,
        **adapter_kwargs,
    ):
        self.layer_in_channels = layer_in_channels
        self.select_in_all_layers = select_in_all_layers
        self.interp_ratio = interp_ratio
        self.layer_projs: nn.ModuleList | None = None
        super().__init__(backbone=backbone, **adapter_kwargs)

    def _build_layer_projs(self) -> None:
        if self.layer_in_channels is None:
            self.layer_projs = None
            return
        if len(self.layer_in_channels) != len(self.interaction_indexes):
            raise ValueError(
                f"{len(self.layer_in_channels)=} must match {len(self.interaction_indexes)=} for layer projections."
            )
        self.layer_projs = nn.ModuleList(
            [TokenLayerProjector(in_ch, self.embed_dim) for in_ch in self.layer_in_channels]
        )

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
        self.patch_size = 16  # TODO: set in config

        logger.info(f"[Tokenizer backbone adapted]: embed dim={self.embed_dim}")
        logger.info(f"[Tokenizer backbone adapted]: interaction_indexes={self.interaction_indexes}")

        # Ensure eval mode for backbone if frozen to save memory from dropout/norm buffers
        if freeze_backbone:
            self.backbone.eval()

        self._build_layer_projs()

        return self.embed_dim

    def _forward_backbone_intermediate_features(self, x: Float[Tensor, "b c h w"]):
        # NOTE: cls feature is None
        grad_ctx = torch.no_grad if self.freeze_backbone else torch.enable_grad
        select_in_all_layers: bool = self.select_in_all_layers
        interp_ratio = self.interp_ratio

        with torch.autocast("cuda", torch.bfloat16):
            with grad_ctx():
                enc_out = self.backbone.encode(x, get_intermediate_features=True)
                if select_in_all_layers:
                    if isinstance(enc_out, dict):
                        all_layers = []
                        all_layers += enc_out.low_lvl_z
                        all_layers += enc_out.sem_z  # use semantic z not low-level z
                        final_latent = enc_out.latent
                    elif isinstance(enc_out, list | tuple):
                        all_layers = list(enc_out[0]) + list(enc_out[1])
                        final_latent = None
                else:
                    assert isinstance(enc_out, dict)
                    all_layers = enc_out.sem_z  # use semantic z not low-level z
                    final_latent = enc_out.latent

        # Clearing the backbone internal cache to free VRAM
        if hasattr(self.backbone, "z"):
            self.backbone.z = None  # type: ignore[invalid-assignment]
        if hasattr(self.backbone, "sem_z"):
            self.backbone.sem_z = None  # type: ignore[invalid-assignment]

        # reorganize all_layers
        assert all_layers is not None, "all_layers is None"
        if interp_ratio is not None:
            assert len(interp_ratio) == len(self.interaction_indexes), (
                f"{interp_ratio=} length must be equal to {len(self.interaction_indexes)=}"
            )

        max_index = len(all_layers) - 1
        out_of_range = [idx for idx in self.interaction_indexes if idx < 0 or idx > max_index]
        if out_of_range:
            raise IndexError(
                f"interaction_indexes has out-of-range values: {out_of_range}, valid range is [0, {max_index}]"
            )
        selected_layers = [all_layers[idx] for idx in self.interaction_indexes]

        # None stands for cls feature
        for i, token_feat in enumerate(selected_layers):
            feat_hw: tuple[int, int] | None = None
            if token_feat.ndim == 4:
                if interp_ratio is not None:
                    token_feat = self._interp_4d(token_feat, interp_ratio[i])

                # project into
                if self.layer_projs is not None:
                    token_feat = self.layer_projs[i](token_feat)

                feat_hw = (token_feat.shape[2], token_feat.shape[3])
                token_feat = rearrange(token_feat, "b c h w -> b (h w) c")
            else:
                assert token_feat.ndim == 3
                if interp_ratio is not None:
                    h = w = int(math.sqrt(token_feat.shape[1]))
                    if h * w != token_feat.shape[1]:
                        raise ValueError(
                            f"Token length {token_feat.shape[1]} is not a square; cannot infer spatial shape."
                        )
                    new_h = int(h * interp_ratio[i])
                    new_w = int(w * interp_ratio[i])
                    token_feat = rearrange(token_feat, "b (h w) c -> b c h w", h=h, w=w)
                    token_feat = self._interp_4d(token_feat, interp_ratio[i])
                    feat_hw = (token_feat.shape[2], token_feat.shape[3])
                    token_feat = rearrange(token_feat, "b c h w -> b (h w) c")
                else:
                    h = w = int(math.sqrt(token_feat.shape[1]))
                    if h * w != token_feat.shape[1]:
                        raise ValueError(
                            f"Token length {token_feat.shape[1]} is not a square; cannot infer spatial shape."
                        )
                    feat_hw = (h, w)
                # proj
                if self.layer_projs is not None:
                    token_feat = self.layer_projs[i](token_feat)

            # if getattr(self, "norm_backbone_features", False):
            #     norm_layer = self.backbone.semantic_enc_transformer.head[0]
            #     token_feat = norm_layer(token_feat)

            cls_feat = None
            selected_layers[i] = (token_feat, cls_feat, feat_hw)

        assert len(selected_layers) == len(self.interaction_indexes), (
            f"{len(selected_layers)=} != {len(self.interaction_indexes)=}"
        )
        return selected_layers, final_latent

    def _interp_4d(self, x, r: float):
        x = F.interpolate(x, scale_factor=r, mode="bilinear", align_corners=False)
        return x


class TokenLayerProjector(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        if in_ch == out_ch:
            self.proj_2d = nn.Identity()
            self.proj_1d = nn.Identity()
        else:
            self.proj_2d = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            self.proj_1d = nn.Linear(in_ch, out_ch, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 4:
            return self.proj_2d(x)
        if x.ndim == 3:
            return self.proj_1d(x)
        raise ValueError(f"Expected 3D or 4D tensor, got ndim={x.ndim}")


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
        self.freeze_tokenizer = bool(cfg.get("freeze_tokenizer", True))
        self._debug = cfg._debug

        # Validate parameters
        n_conv_per_stage = self.adapter_cfg.n_conv_per_stage
        n_stages = cfg.n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages

        # Ensure we have 4 stages to match adapter output
        if cfg.n_stages != 4:
            logger.error(f"Warning: Adapter outputs 4 scales, but n_stages={n_stages}. Adjusting to 4.")
            raise ValueError("n_stages must be 4")

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
            block_types=self.adapter_cfg.block_types,
            deep_supervision=cfg.deep_supervision,
            output_process=self.adapter_cfg.get("output_process", None),
            additional_head=self.adapter_cfg.get("additional_head", None),
        )
        logger.info(f"Created Unet decoder with 4 stages - with Nparams={parameter_count(self.decoder)['']}")

        # Init weights
        self.init_weights()

    def _create_tok_encoder(self) -> DINOv3EncoderAdapter:
        """Create DINOv3 encoder"""
        cfg = self.cfg
        f_cfg = self.tok_f_cfg
        a_cfg = self.adapter_cfg
        t_cfg = self.tok_cfg

        # Get model information
        model_name = f_cfg.model_name
        interaction_indexes = TOKENIZER_INTERACTION_INDEXES[model_name]
        interaction_indexes = f_cfg.get("interaction_indexes", None) or interaction_indexes
        select_in_all_layers = f_cfg.get("select_in_all_layers", False)

        logger.info(
            f"Creating tokenizer encoder: {model_name}\n"
            f"taken interaction indexes: {interaction_indexes}, select in all layers: {select_in_all_layers}"
        )
        logger.info(f"Tokenizer training mode: {'probe (frozen tokenizer)' if self.freeze_tokenizer else 'finetune'}")

        # Load DINOv3 backbone
        tok_backbone = CosmosHybridTokenizer.create_model(
            cnn_cfg=t_cfg.cnn_cfg,
            trans_enc_cfg=t_cfg.trans_enc_cfg,
            trans_dec_cfg=t_cfg.trans_dec_cfg,
            distillation_cfg=t_cfg.distill_cfg,
            hybrid_tokenizer_cfg=t_cfg.get("hybrid_tokenizer_cfg", None),
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
            init_values=0.0,
            n_points=4,
            pretrain_size=512,
            select_in_all_layers=select_in_all_layers,
            conv_inplane=f_cfg.conv_inplane,
            deform_num_heads=f_cfg.deform_num_heads,
            drop_path_rate=f_cfg.drop_path_rate,
            with_cffn=f_cfg.with_cffn,
            cffn_ratio=f_cfg.cffn_ratio,
            deform_ratio=f_cfg.deform_ratio,
            add_vit_feature=f_cfg.add_vit_feature,
            use_extra_extractor=f_cfg.use_extra_extractor,
            with_cp=f_cfg.with_cp,
            interp_ratio=f_cfg.get("interp_ratio", None),
            layer_in_channels=f_cfg.get("layer_in_channels", None),
            extractor_type=f_cfg.get("extractor_type", "deform_attention"),
            extractor_kwargs=f_cfg.get("extractor_kwargs", {}),
            freeze_backbone=self.freeze_tokenizer,
        )
        encoder_adapter = DINOv3EncoderAdapter(
            dinov3_adapter=dinov3_adapter,
            target_channels=f_cfg.features_per_stage,
            conv_op=nn.Conv2d,
            norm_op=get_norm_layer(a_cfg.norm),
            nonlin=get_act_layer(a_cfg.act),
            dropout_op=a_cfg.drop,
            conv_bias=a_cfg.conv_bias,
            with_cp=f_cfg.with_cp,
        )
        logger.info("Created tokenizer encoder adapter.")

        return encoder_adapter

    def forward(self, x: Float[Tensor, "b c h w"]):
        skips, final_h = self.encoder(x)
        output = self.decoder(skips, final_h)
        return output

    def init_weights(self) -> None:
        from timm.layers.weight_init import lecun_normal_

        def _apply(module, name: str):
            if "backbone" not in name:  # skip the pretrained weights
                if hasattr(module, "init_weights"):
                    module.init_weights()
                elif isinstance(module, _ConvNd):
                    lecun_normal_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Linear):
                    nn.init.trunc_normal_(module.weight, std=0.02)
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
            if self.freeze_tokenizer and "backbone" in name:
                param.requires_grad = False
                continue
            yield param

    def named_parameters(self, *args, **kwargs):
        for name, param in super().named_parameters(*args, **kwargs):
            if self.freeze_tokenizer and "backbone" in name:
                param.requires_grad = False
                continue
            yield name, param

    def _filter_backbone_params(self, k: str):
        return self.freeze_tokenizer and "backbone" in k

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        # Remove backbone parameters from state dict in probe mode
        backbone_keys = [k for k in state_dict.keys() if self._filter_backbone_params(k)]
        for k in backbone_keys:
            del state_dict[k]
        if self.freeze_tokenizer:
            logger.info(f"Get {len(state_dict)} parameters in state_dict (backbone removed).")
        else:
            logger.info(f"Get {len(state_dict)} parameters in state_dict (backbone kept for finetune).")
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

    def set_grad_checkpointing(self, enable: bool = True):
        # encoder set checkpoint in cfg
        self.decoder.set_grad_checkpointing(enable)


def test_seg_decoder_model():
    from fvcore.nn import parameter_count_table
    from src.data.hyperspectral_loader import get_fast_test_hyperspectral_data

    def _bytes_to_gb(num_bytes: int) -> float:
        return num_bytes / (1024**3)

    def _log_cuda_mem(tag: str) -> None:
        torch.cuda.synchronize()
        allocated_gb = _bytes_to_gb(torch.cuda.memory_allocated())
        peak_gb = _bytes_to_gb(torch.cuda.max_memory_allocated())
        logger.info(f"[mem] {tag}: alloc={allocated_gb:.2f}G peak={peak_gb:.2f}G")

    def _forward_decoder_with_mem(
        decoder: UNetDecoder,
        skips: list[Float[Tensor, "b c h w"]],
        cond: Float[Tensor, "b latent_ch h w"] | None,
        log_mem: bool,
    ) -> Tensor | list[Tensor]:
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(decoder.stages)):
            x = decoder.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            x = decoder.stages[s](x, cond)
            if log_mem:
                _log_cuda_mem(f"decoder stage {s} after stage")
            if decoder.deep_supervision:
                seg_outputs.append(decoder.seg_layers[s](x))
            elif s == (len(decoder.stages) - 1):
                seg_outputs.append(decoder.seg_layers[-1](x))
            lres_input = x

        seg_outputs = seg_outputs[::-1]
        if not decoder.deep_supervision:
            r = seg_outputs[0]
            r = decoder.output_processor(r)
        else:
            r = [decoder.output_processor(seg_out) for seg_out in seg_outputs]
        return r

    dl = get_fast_test_hyperspectral_data("fmow_RGB", batch_size=4)
    sample = next(iter(dl))
    x = sample["img"].cuda()
    x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)

    cfg = _create_default_cfg()
    cfg._debug = True
    cfg.tokenizer_pretrained_path = "runs/stage1_cosmos_hybrid/2025-12-21_23-52-12_hybrid_cosmos_f16c64_ijepa_pretrained_sem_no_lejepa/ema/tokenizer/model.safetensors"
    unet = TokenizerHybridUNet(cfg).cuda()

    # model info

    print(parameter_count_table(unet))

    # unet.eval()
    # with torch.autocast("cuda", torch.bfloat16):
    #     with torch.no_grad():
    #         y = unet(x)
    # print(y.shape)
    # assert y.shape[-2:] == x.shape[-2:], "Output shape mismatch"

    #         y_recon = unet.encoder.dinov3_adapter.backbone(x)
    # from torchmetrics import PeakSignalNoiseRatio

    # logger.info(f"Input shape: {x.shape}")
    # logger.info(f"Output shape: {y_recon.shape}")

    # psnr_fn = PeakSignalNoiseRatio(data_range=1.0).cuda()
    # psnr = psnr_fn((y_recon + 1) / 2, (x + 1) / 2)
    # logger.info(f"Reconstruction PSNR: {psnr}")

    optim = torch.optim.AdamW(
        unet.parameters(),
        lr=1e-4,
        weight_decay=0.01,
    )
    unet.set_grad_checkpointing(True)
    unet.train()

    # deform: 69G
    # convnext: 74G
    # sla: 74G
    for i in range(100):
        optim.zero_grad()
        if i == 0:
            torch.cuda.reset_peak_memory_stats()
            _log_cuda_mem("after reset")
        with torch.autocast("cuda", torch.bfloat16):
            encoder_out = unet.encoder(x)
            if isinstance(encoder_out, tuple):
                skips, final_h = encoder_out
            else:
                skips, final_h = encoder_out, None
            if i == 0:
                _log_cuda_mem("after encoder")
            y = _forward_decoder_with_mem(unet.decoder, skips, final_h, log_mem=i == 0)
            if i == 0:
                _log_cuda_mem("after decoder")
            y_for_loss = y[0] if isinstance(y, list) else y
            loss = F.mse_loss(torch.randn_like(y_for_loss), y_for_loss)
            if i == 0:
                _log_cuda_mem("after loss")
        loss.backward()
        if i == 0:
            _log_cuda_mem("after backward")
        optim.step()

        if i % 10 == 0:
            logger.info(f"Loss: {loss.item()}")


if __name__ == "__main__":
    """
    CUDA_VISIBLE_DEVICES=1 LOVELY_TENSORS=1 python -m src.stage2.segmentation.models.tokenizer_backbone_adapted
    """
    with logger.catch():
        test_seg_decoder_model()

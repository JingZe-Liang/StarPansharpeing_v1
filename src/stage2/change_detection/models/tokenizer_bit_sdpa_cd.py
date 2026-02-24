from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from omegaconf import OmegaConf
from timm.layers import get_act_layer, get_norm_layer
from torch import Tensor
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.module import _IncompatibleKeys

from src.stage1.cosmos.cosmos_hybrid import CosmosHybridTokenizer
from src.utilities.config_utils import function_config_to_basic_types

from .adapter import DINOv3EncoderAdapter
from .tokenizer_backbone_adapted import TOKENIZER_INTERACTION_INDEXES, HybridTokenizerEncoderAdapter


def _create_default_cfg():
    yaml_cfg = """
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
          semantic: [5, 11, 17, 23]
      hybrid_tokenizer_cfg:
        latent_bottleneck_type: before_semantic
        latent_straight_through_skip: true

    tokenizer_feature:
      pretrained_path: null
      model_name: hybrid_tokenizer_b16
      pretrained_size: 512
      in_channels: 3
      interp_ratio: null
      interaction_indexes: [1, 2, 4, 5]
      layer_in_channels: [512, 512, 1152, 1152]
      features_per_stage: [384, 384, 512, 512]
      conv_inplane: 64
      drop_path_rate: 0.2
      with_cffn: true
      cffn_ratio: 1.0
      deform_num_heads: 16
      deform_ratio: 0.5
      add_vit_feature: true
      use_extra_extractor: true
      with_cp: true
      select_in_all_layers: true
      extractor_type: deform_attention
      extractor_kwargs: {}

    adapter:
      latent_width: 32
      norm: rmsnorm2dfp32
      act: gelu
      drop: 0.0
      conv_bias: true

    neck:
      stage_index: 2
      raw_embed_channels: 64
      out_channels: 512

    bit_head:
      channels: 128
      embed_dims: 256
      enc_depth: 2
      dec_depth: 4
      num_heads: 8
      drop_rate: 0.0
      use_tokenizer: true
      token_len: 4
      pool_size: 2
      pool_mode: max
      upsample_scale: 4

    tokenizer_pretrained_path: null
    input_channels: 3
    num_classes: 2
    freeze_tokenizer: false
    _debug: false
    """
    return OmegaConf.create(yaml_cfg)


class SDPAAttention(nn.Module):
    def __init__(self, in_dims: int, embed_dims: int, num_heads: int, drop_rate: float = 0.0) -> None:
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims ({embed_dims}) must be divisible by num_heads ({num_heads})")

        self.in_dims = in_dims
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        self.drop_rate = float(drop_rate)

        self.to_q = nn.Linear(in_dims, embed_dims, bias=False)
        self.to_k = nn.Linear(in_dims, embed_dims, bias=False)
        self.to_v = nn.Linear(in_dims, embed_dims, bias=False)
        self.proj = nn.Linear(embed_dims, in_dims, bias=True)
        self.out_drop = nn.Dropout(self.drop_rate)

    def _reshape_to_heads(self, x: Tensor) -> Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: Tensor, ref: Tensor) -> Tensor:
        q = self._reshape_to_heads(self.to_q(x))
        k = self._reshape_to_heads(self.to_k(ref))
        v = self._reshape_to_heads(self.to_v(ref))

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.drop_rate if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.embed_dims)
        out = self.proj(out)
        return self.out_drop(out)


class FeedForward(nn.Sequential):
    def __init__(self, dim: int, hidden_dim: int, drop_rate: float = 0.0) -> None:
        super().__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop_rate),
        )


class TransformerEncoderBlock(nn.Module):
    def __init__(self, in_dims: int, embed_dims: int, num_heads: int, drop_rate: float = 0.0) -> None:
        super().__init__()
        self.attn = SDPAAttention(in_dims, embed_dims, num_heads, drop_rate)
        self.ff = FeedForward(in_dims, embed_dims, drop_rate)
        self.norm1 = nn.LayerNorm(in_dims)
        self.norm2 = nn.LayerNorm(in_dims)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x))
        return x + self.ff(self.norm2(x))


class TransformerDecoderBlock(nn.Module):
    def __init__(self, in_dims: int, embed_dims: int, num_heads: int, drop_rate: float = 0.0) -> None:
        super().__init__()
        self.attn = SDPAAttention(in_dims, embed_dims, num_heads, drop_rate)
        self.ff = FeedForward(in_dims, embed_dims, drop_rate)
        self.norm_q = nn.LayerNorm(in_dims)
        self.norm_kv = nn.LayerNorm(in_dims)
        self.norm2 = nn.LayerNorm(in_dims)

    def forward(self, x: Tensor, ref: Tensor) -> Tensor:
        x = x + self.attn(self.norm_q(x), self.norm_kv(ref))
        return x + self.ff(self.norm2(x))


class RawInjectionNeck(nn.Module):
    def __init__(
        self,
        backbone_channels: int,
        raw_in_channels: int,
        out_channels: int,
        raw_embed_channels: int,
    ) -> None:
        super().__init__()
        self.raw_proj = nn.Sequential(
            nn.Conv2d(raw_in_channels, raw_embed_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(raw_embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(raw_embed_channels, raw_embed_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(raw_embed_channels),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Conv2d(backbone_channels + raw_embed_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, feature: Tensor, raw_img: Tensor) -> Tensor:
        raw_feat = self.raw_proj(raw_img)
        raw_feat = F.interpolate(raw_feat, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([feature, raw_feat], dim=1))


class BITSDPAHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels: int = 128,
        embed_dims: int = 256,
        enc_depth: int = 1,
        dec_depth: int = 4,
        num_heads: int = 8,
        drop_rate: float = 0.0,
        use_tokenizer: bool = True,
        token_len: int = 4,
        pool_size: int = 2,
        pool_mode: str = "max",
        upsample_scale: int = 4,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if token_len <= 0:
            raise ValueError(f"token_len must be positive, got {token_len}")

        self.channels = channels
        self.use_tokenizer = use_tokenizer
        self.token_len = token_len
        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.upsample_scale = upsample_scale

        self.pre_process = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        if self.use_tokenizer:
            self.conv_att = nn.Conv2d(channels, token_len, kernel_size=1, bias=True)

        self.enc_pos_embedding = nn.Parameter(torch.randn(1, token_len * 2, channels))

        self.encoder = nn.ModuleList(
            [TransformerEncoderBlock(channels, embed_dims, num_heads, drop_rate=drop_rate) for _ in range(enc_depth)]
        )
        self.decoder = nn.ModuleList(
            [TransformerDecoderBlock(channels, embed_dims, num_heads, drop_rate=drop_rate) for _ in range(dec_depth)]
        )

        self.cls_seg = nn.Conv2d(channels, num_classes, kernel_size=1, bias=True)

    def _forward_semantic_tokens(self, x: Tensor) -> Tensor:
        b, c = x.shape[:2]
        att_map = self.conv_att(x).reshape(b, self.token_len, 1, -1)
        att_map = F.softmax(att_map, dim=-1)
        x = x.reshape(b, 1, c, -1)
        return (x * att_map).sum(-1)

    def _forward_reshaped_tokens(self, x: Tensor) -> Tensor:
        if self.pool_mode == "max":
            x = F.adaptive_max_pool2d(x, (self.pool_size, self.pool_size))
        elif self.pool_mode == "avg":
            x = F.adaptive_avg_pool2d(x, (self.pool_size, self.pool_size))
        elif self.pool_mode != "identity":
            raise ValueError(f"Unsupported pool_mode: {self.pool_mode}")
        return x.permute(0, 2, 3, 1).flatten(1, 2)

    def _tokenize(self, x: Tensor) -> Tensor:
        if self.use_tokenizer:
            return self._forward_semantic_tokens(x)
        return self._forward_reshaped_tokens(x)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self.pre_process(x1)
        x2 = self.pre_process(x2)

        token1 = self._tokenize(x1)
        token2 = self._tokenize(x2)

        token = torch.cat([token1, token2], dim=1)
        if token.shape[1] == self.enc_pos_embedding.shape[1]:
            token = token + self.enc_pos_embedding

        for encoder in self.encoder:
            token = encoder(token)

        token1, token2 = torch.chunk(token, 2, dim=1)

        for decoder in self.decoder:
            b, c, h, w = x1.shape
            x1_seq = x1.permute(0, 2, 3, 1).flatten(1, 2)
            x2_seq = x2.permute(0, 2, 3, 1).flatten(1, 2)

            x1_seq = decoder(x1_seq, token1)
            x2_seq = decoder(x2_seq, token2)

            x1 = x1_seq.transpose(1, 2).reshape(b, c, h, w)
            x2 = x2_seq.transpose(1, 2).reshape(b, c, h, w)

        y = torch.abs(x1 - x2)
        if self.upsample_scale > 1:
            y = F.interpolate(y, scale_factor=self.upsample_scale, mode="bilinear", align_corners=False)

        return self.cls_seg(y)


class TokenizerHybridBITCDModel(nn.Module):
    def __init__(self, cfg, encoder: nn.Module | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_cfg = cfg.tokenizer
        self.tok_f_cfg = cfg.tokenizer_feature
        self.adapter_cfg = cfg.adapter

        self.freeze_tokenizer = bool(cfg.get("freeze_tokenizer", True))
        self._debug = bool(cfg.get("_debug", False))

        self.encoder = encoder if encoder is not None else self._create_tok_encoder()

        self.stage_index = int(cfg.neck.stage_index)
        if self.stage_index < 0 or self.stage_index >= len(self.tok_f_cfg.features_per_stage):
            raise ValueError(
                f"neck.stage_index must be in [0, {len(self.tok_f_cfg.features_per_stage) - 1}], got {self.stage_index}"
            )
        feature_channels = int(self.tok_f_cfg.features_per_stage[self.stage_index])
        neck_out_channels = int(cfg.neck.out_channels)

        self.neck = RawInjectionNeck(
            backbone_channels=feature_channels,
            raw_in_channels=int(cfg.input_channels),
            out_channels=neck_out_channels,
            raw_embed_channels=int(cfg.neck.raw_embed_channels),
        )

        self.head = BITSDPAHead(
            in_channels=neck_out_channels,
            num_classes=int(cfg.num_classes),
            channels=int(cfg.bit_head.channels),
            embed_dims=int(cfg.bit_head.embed_dims),
            enc_depth=int(cfg.bit_head.enc_depth),
            dec_depth=int(cfg.bit_head.dec_depth),
            num_heads=int(cfg.bit_head.num_heads),
            drop_rate=float(cfg.bit_head.drop_rate),
            use_tokenizer=bool(cfg.bit_head.use_tokenizer),
            token_len=int(cfg.bit_head.token_len),
            pool_size=int(cfg.bit_head.pool_size),
            pool_mode=str(cfg.bit_head.pool_mode),
            upsample_scale=int(cfg.bit_head.upsample_scale),
        )

        self._init_weights()

    def _create_tok_encoder(self) -> DINOv3EncoderAdapter:
        cfg = self.cfg
        f_cfg = self.tok_f_cfg
        a_cfg = self.adapter_cfg
        t_cfg = self.tok_cfg

        model_name = f_cfg.model_name
        interaction_indexes = TOKENIZER_INTERACTION_INDEXES[model_name]
        interaction_indexes = f_cfg.get("interaction_indexes", None) or interaction_indexes
        select_in_all_layers = f_cfg.get("select_in_all_layers", False)

        logger.info(
            f"Creating tokenizer encoder for BIT CD: {model_name}, "
            f"interaction indexes: {interaction_indexes}, "
            f"select in all layers: {select_in_all_layers}"
        )
        logger.info(
            f"Tokenizer training mode: "
            f"<green>{'probe (frozen tokenizer)' if self.freeze_tokenizer else 'finetune'}</green>"
        )

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
        elif not cfg._debug:
            raise ValueError("tokenizer_pretrained_path must be specified for tokenizer backbone")

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
        return DINOv3EncoderAdapter(
            dinov3_adapter=dinov3_adapter,
            target_channels=f_cfg.features_per_stage,
            conv_op=nn.Conv2d,
            norm_op=get_norm_layer(a_cfg.norm),
            nonlin=get_act_layer(a_cfg.act),
            dropout_op=a_cfg.drop,
            conv_bias=a_cfg.conv_bias,
            with_cp=f_cfg.with_cp,
        )

    @staticmethod
    def _validate_cd_inputs(x: tuple[Tensor, Tensor] | list[Tensor]) -> tuple[Tensor, Tensor]:
        if not isinstance(x, (tuple, list)):
            raise TypeError(f"Input must be a tuple/list of two tensors (x1, x2), got {type(x)}")
        if len(x) != 2:
            raise ValueError(f"Input must contain exactly two tensors (x1, x2), got len={len(x)}")

        x1, x2 = x
        if not isinstance(x1, Tensor) or not isinstance(x2, Tensor):
            raise TypeError(f"Both inputs must be Tensor, got {type(x1)} and {type(x2)}")
        if x1.shape != x2.shape:
            raise ValueError(f"Input tensors must share shape, got {tuple(x1.shape)} vs {tuple(x2.shape)}")
        return x1, x2

    @staticmethod
    def _split_encoder_output(
        encoder_out: list[Tensor] | tuple[list[Tensor], Tensor | None],
    ) -> tuple[list[Tensor], Tensor | None]:
        if isinstance(encoder_out, tuple):
            skips, latent = encoder_out
            return skips, latent
        return encoder_out, None

    def forward(self, x: tuple[Tensor, Tensor] | list[Tensor]) -> Tensor:
        x1, x2 = self._validate_cd_inputs(x)

        skips1, _ = self._split_encoder_output(self.encoder(x1))
        skips2, _ = self._split_encoder_output(self.encoder(x2))
        if self.stage_index >= len(skips1) or self.stage_index >= len(skips2):
            raise ValueError(
                f"neck.stage_index={self.stage_index} exceeds encoder output length: "
                f"len(skips1)={len(skips1)}, len(skips2)={len(skips2)}"
            )

        feat1 = skips1[self.stage_index]
        feat2 = skips2[self.stage_index]

        feat1 = self.neck(feat1, x1)
        feat2 = self.neck(feat2, x2)

        logits = self.head(feat1, feat2)
        return F.interpolate(logits, size=x1.shape[-2:], mode="bilinear", align_corners=False)

    def _init_weights(self) -> None:
        for name, module in self.named_modules():
            if "backbone" in name:
                continue
            if isinstance(module, _ConvNd):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @classmethod
    @function_config_to_basic_types
    def create_model(cls, cfg=_create_default_cfg(), encoder: nn.Module | None = None, **overrides):
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        if overrides:
            cfg.merge_with(overrides)
        if cfg.tokenizer_pretrained_path is None and encoder is None and not cfg._debug:
            logger.warning("[TokenizerHybridBITCDModel]: No pretrained weights provided.")
        return cls(cfg, encoder=encoder)

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

    def _filter_backbone_params(self, key: str) -> bool:
        return self.freeze_tokenizer and "backbone" in key

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        backbone_keys = [k for k in sd.keys() if self._filter_backbone_params(k)]
        for key in backbone_keys:
            del sd[key]
        return sd

    def load_state_dict(self, state_dict, strict: bool = False, return_not_loaded_keys: bool = False, *args, **kwargs):
        backbone_keys = [k for k in state_dict.keys() if self._filter_backbone_params(k)]
        for key in backbone_keys:
            del state_dict[key]

        missing_keys, unexpected_keys = super().load_state_dict(state_dict, strict=strict)
        missing_keys = [k for k in missing_keys if not self._filter_backbone_params(k)]
        unexpected_keys = [k for k in unexpected_keys if not self._filter_backbone_params(k)]

        if len(missing_keys) > 0:
            logger.warning(f"Missing Keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            logger.warning(f"Unexpected Keys: {unexpected_keys}")

        if return_not_loaded_keys:
            return _IncompatibleKeys(missing_keys, unexpected_keys)
        return _IncompatibleKeys([], [])


if __name__ == "__main__":
    """
        python -m src.stage2.change_detection.models.tokenizer_bit_sdpa_cd
    """
    cfg = _create_default_cfg()
    cfg._debug = True
    cfg.freeze_tokenizer = True
    device = torch.device("cuda:1")
    model = TokenizerHybridBITCDModel.create_model(cfg=cfg).to(device)
    from fvcore.nn import parameter_count_table

    print(parameter_count_table(model))
    x1 = torch.randn(1, 3, 256, 256, device=device)
    x2 = torch.randn(1, 3, 256, 256, device=device)

    with torch.autocast(device.type, dtype=torch.bfloat16):
        y = model([x1, x2])
    print(y.shape)
    print(f"num params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

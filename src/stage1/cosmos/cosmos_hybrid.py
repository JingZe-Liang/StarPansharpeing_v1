"""
Hyperspectral Transformer Tokenizer with hybrid CNN / Transformer architecture.
RMSNorm + SwiGLU + EVA attention with Rope.
"""

from dataclasses import asdict, dataclass
from typing import Any, List, Optional, Union, cast

import accelerate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from einops import rearrange
from loguru import logger
from omegaconf import OmegaConf
from timm.layers import create_conv2d, create_norm_act_layer
from torch import Tensor
from typing_extensions import Annotated

from src.utilities.config_utils import (
    dataclass_from_dict,
    function_config_to_basic_types,
    function_config_to_easy_dict,
    set_defaults,
)
from src.utilities.network_utils.network_loading import load_weights_with_shape_check

from .cosmos_tokenizer import (
    ContinuousImageTokenizer,
    ContinuousTokenizerConfig,
    EncoderDecoderConfig,
)
from .modules import blocks as cosmos_blocks
from .modules.blocks import AdaptiveOutputConvLayer
from .modules.layers2d import Decoder, Encoder, GenerativeDecoder
from .modules.naflex import IJEPANaFlexViT, NaFlexVitCfg, Transformer
from .modules.proj import build_mlp


class CosmosHybridTokenizer(ContinuousImageTokenizer):
    _no_split_modules = ["EvaBlock", "ResnetBlock", "AttnBlock"]
    _vf_on_z_or_module = "z"  # must be z if using this model
    # latents cached
    z: Tensor | None = None
    sem_z: Tensor | None = None
    supported_cached_hiddens: List[str] = ["z", "sem_z"]
    cache_layers: dict[str, list[int]] = {
        "low_level": [0, 1, 2, -1],  # -1 means middle layer
        "semantic": [2, 5, 8, 11],  # 12 layers of encoder
    }
    low_lvl_repa_proj_chans: list[int] = []
    _semantic_feature_dim: int = 1152

    def __init__(
        self,
        cnn_cfg: ContinuousTokenizerConfig,
        trans_enc_cfg: NaFlexVitCfg,
        trans_dec_cfg: Optional[NaFlexVitCfg] = None,
        *,
        cnn_enc_cfg: EncoderDecoderConfig | None = None,
        cnn_dec_cfg: EncoderDecoderConfig | None = None,
        distillation_cfg: dict[str, Any] = {},
        hybrid_tokenizer_cfg: dict[str, Any] = {},
    ):
        self.cnn_cfg = cnn_cfg
        self.trans_enc_cfg = trans_enc_cfg
        self.trans_dec_cfg = trans_dec_cfg
        self.distillation_cfg = distillation_cfg
        self.grad_checkpointing = self.cnn_cfg.model.act_checkpoint

        self.latent_channels = self.cnn_cfg.model.latent_channels

        ###### Distillation configs
        self.distillation_cfg = set_defaults(
            distillation_cfg,
            {
                "dino_feature_dim": 1024,
                "semantic_feature_dim": 1152,
                "cache_layers": self.cache_layers,
            },
        )
        self._dino_feature_dim = self.distillation_cfg.dino_feature_dim
        self._semantic_feature_dim = self.distillation_cfg.semantic_feature_dim
        self.cache_layers = self.distillation_cfg.cache_layers

        ###### Deep supervision configs
        self.hybrid_tokenizer_cfg = set_defaults(
            hybrid_tokenizer_cfg,
            {
                "deep_supervision_type": None,
                "latent_bottleneck_type": "after_semantic",
                "latent_straight_through_skip": False,
            },
        )
        # deep_supervision_type: [direct_out, sum_previous_out]
        self.deep_supervision_type = self.hybrid_tokenizer_cfg.deep_supervision_type
        self.latent_bottleneck_type = self.hybrid_tokenizer_cfg.latent_bottleneck_type
        self._is_deep_supervision = self.deep_supervision_type is not None
        self._build_deep_supervised_heads(cnn_cfg)

        build_enc_dec_kwargs = {}
        if cnn_enc_cfg is not None or cnn_dec_cfg is not None:
            # If the encoder or decoder config is provided, use it
            build_enc_dec_kwargs = edict(enc_cnn_model_cfg=cnn_enc_cfg, dec_cnn_model_cfg=cnn_dec_cfg)
        super().__init__(self.cnn_cfg, build_enc_dec_kwargs=build_enc_dec_kwargs)

        ##### Transformer models
        self._build_transformers(cnn_cfg, trans_enc_cfg, trans_dec_cfg)
        self._build_straight_through_skip(self.hybrid_tokenizer_cfg, cnn_cfg)

        ### Loading pretrained models
        if self.cnn_cfg.loading_type == "hybrid_pretrained":
            self.load_pretrained(self.cnn_cfg.uni_path)

        if self.trans_enc_cfg.pretrained_type is not None and "lejepa_latent" in self.trans_enc_cfg.pretrained_type:
            from src.stage1.self_supervised.lejepa_aug import create_lejepa_projector

            assert self.latent_bottleneck_type == "before_semantic", (
                "LeJEPA latent loss requires the latent bottleneck before semantic encoder"
            )

            self.lejepa_projector_latent = create_lejepa_projector(self.latent_channels, self.latent_channels)
            logger.info("[LeJEPA loss]: create lejepa loss and projector for the latent")

    def _build_encoder_decoder(  # type: ignore
        self,
        cfg: ContinuousTokenizerConfig,
        model_cfg: EncoderDecoderConfig,
        *,
        enc_cnn_model_cfg: EncoderDecoderConfig | None = None,
        dec_cnn_model_cfg: EncoderDecoderConfig | None = None,
    ):
        self._is_diffbands = isinstance(model_cfg.in_channels, (tuple, list))

        enc_kwargs = dec_kwargs = asdict(model_cfg)
        if enc_cnn_model_cfg is not None:
            enc_kwargs = asdict(enc_cnn_model_cfg)
            logger.info("[Build CNN Encoder]: override cnn encoder config.")
        if dec_cnn_model_cfg is not None:
            dec_kwargs = asdict(dec_cnn_model_cfg)
            logger.info("[Build CNN Decoder]: override cnn decoder config.")

        encoder = Encoder(**enc_kwargs)
        if cfg.decoder_type == "default":
            decoder = Decoder(**dec_kwargs)
        elif cfg.decoder_type == "generative":
            decoder = GenerativeDecoder(**dec_kwargs)
        else:
            raise ValueError(f"Unknown decoder type: {cfg.decoder_type}")

        logger.info(f"[CNN tokenizer]: Build encoder and {cfg.decoder_type} decoder.")
        return encoder, decoder

    def _set_low_level_proj_chans(self):
        cache_index = self.cache_layers["low_level"]
        if not isinstance(cache_index, list):
            self.low_lvl_repa_proj_chans = []
            return

        # low-level repa proj channels
        base_chans = self.cnn_cfg.model.channels
        channels_mult = self.cnn_cfg.model.channels_mult
        in_ch_mult = (1,) + tuple(channels_mult)
        n_res = len(channels_mult)
        for i_level in range(n_res):
            in_chan, out_chan = (
                base_chans * in_ch_mult[i_level],
                base_chans * channels_mult[i_level],
            )
            if i_level in cache_index:
                self.low_lvl_repa_proj_chans.append(out_chan)

        # add mid block
        if cache_index[-1] == -1:
            self.low_lvl_repa_proj_chans.append(out_chan)
            logger.info(f"Low-level repa projection channels: {self.low_lvl_repa_proj_chans}")

    def _build_transformers(self, cnn_cfg, trans_enc_cfg, trans_dec_cfg=None):
        # cnn_cfg is already set as self.cnn_cfg in __init__
        # Transformer Encoder and Decoder
        enc_model_cls = Transformer
        dec_model_cls = Transformer
        if trans_enc_cfg.pretrained_type is not None:  # ("ijepa", "lejepa"):
            enc_model_cls = IJEPANaFlexViT
        if trans_dec_cfg is not None and trans_dec_cfg.pretrained_type is not None:
            dec_model_cls = IJEPANaFlexViT

        self.semantic_enc_transformer = enc_model_cls(self.trans_enc_cfg)
        self.semantic_enc_transformer.set_grad_checkpointing(self.grad_checkpointing)
        logger.info(f"Init semantic transformer encoder.")

        self.semantic_transformer_dec = None
        self.st_skip_sem_decoder = False
        if trans_dec_cfg is not None:
            self.semantic_transformer_dec = dec_model_cls(self.trans_dec_cfg)
            self.semantic_transformer_dec.set_grad_checkpointing(self.grad_checkpointing)
            logger.info(f"Init semantic transformer decoder.")

    def _build_straight_through_skip(self, hybrid_tok_cfg, cnn_cfg):
        # Straight through skip for low-level features
        if hybrid_tok_cfg.latent_straight_through_skip:
            self.st_skip_sem_decoder = True
            # Change the postquant conv
            orig_post_quant_conv = self.decoder.quant_conv
            # When skipping the semantic decoder we concatenate two tensors from z-branch:
            # - the semantic-decoder output 'h' and
            # - the skipped latent 'h_skipped' (both are z_channels)
            # therefore the concatenated channel dim equals 2 * z_channels.
            cat_dim = 2 * cnn_cfg.model.z_channels
            skip_through_cat_conv = nn.Sequential(
                create_norm_act_layer("layernorm2d", cat_dim, "gelu"),
                # post conv to z channels and feed into the cnn decoder
                # fuse the skipped latent (=z_channels) and the semantic decoder output
                create_conv2d(cat_dim, cnn_cfg.model.z_channels, kernel_size=3),
            )
            # replace the post quant conv
            self.decoder.quant_conv = nn.ModuleDict(
                {"post_quant_conv": orig_post_quant_conv, "st_cat_conv": skip_through_cat_conv}
            )
            logger.debug(
                f"Will skip the latent through the cat conv without semantic decoder, "
                "maybe better for reconstruction -- Experimenting"
            )

    def _build_deep_supervised_heads(self, cnn_cfg):
        if not self._is_deep_supervision:
            return

        deep_sup_type = self.deep_supervision_type

        # Per-resolution output channels
        chan_mults = cnn_cfg.model.channels_mult
        n_res = len(chan_mults)
        basic_chan = cnn_cfg.model.channels
        head_out_chan = cnn_cfg.model.out_channels

        # Make the output heads
        self.deep_supervised_heads = nn.ModuleList()
        for i in reversed(range(n_res)):
            to_out = nn.ModuleDict()
            out_chan = basic_chan * chan_mults[i]
            # Create head according to supervision type
            to_out_main = nn.Sequential(
                create_norm_act_layer("layernorm2d", out_chan, "silu"),
                AdaptiveOutputConvLayer(out_chan, head_out_chan, mode="interp"),
            )
            to_out["main"] = to_out_main
            if deep_sup_type == "sum_previous_out":
                # Add previous summed feature
                if i not in (0, n_res - 1):
                    prev_out_chan = basic_chan * chan_mults[i - 1]
                    # Upsample previous feature and add to current
                    # Norm + act + conv + upsample
                    prev_out_proj = nn.Sequential(
                        create_norm_act_layer("layernorm2d", prev_out_chan, "silu"),
                        create_conv2d(prev_out_chan, out_chan, kernel_size=3),
                        nn.Upsample(scale_factor=2, mode="nearest"),
                    )
                    to_out["prev_out_proj"] = prev_out_proj

            self.deep_supervised_heads.append(to_out)
        logger.info(f"Build deep supervised heads with type: {deep_sup_type}")

    def load_pretrained(self, uni_tokenizer_path: str, _reinit_quant_convs=False, **kwargs):  # type: ignore[invalid-method-override]
        """Init the model from the pretrained only CNN weights."""
        if uni_tokenizer_path in (None, ""):
            logger.warning(f"No pretrained weights found at {uni_tokenizer_path}, skip loading ckpt.")
            return

        weights = accelerate.utils.load_state_dict(uni_tokenizer_path)
        missing_ks, unexp_ks = load_weights_with_shape_check(self, weights, load_strategy="search")
        if len(missing_ks) > 0 or len(unexp_ks) > 0:
            logger.warning(f"Missing keys: {missing_ks}, Unexpected keys: {unexp_ks}")

        if _reinit_quant_convs:
            # Init the quant convs for latent min/max value not too large
            nn.init.trunc_normal_(self.encoder.quant_conv.weight, std=0.01)
            nn.init.zeros_(self.encoder.quant_conv.bias)

            # then the quantizer will output the latent_channels h
            nn.init.trunc_normal_(self.decoder.quant_conv.weight, std=0.01)
            nn.init.zeros_(self.decoder.quant_conv.bias)

            logger.warning(
                f"temp code for continue training that initialize the quant convs using trunc_norm_ with std=0.01"
            )

    @staticmethod
    def _interp_max_size_features(feats: list[Tensor]):
        max_size = torch.max(torch.tensor([f.shape[-2:] for f in feats]), dim=0).values
        max_size = tuple(max_size.tolist())
        interp_feats = []
        for f in feats:
            if f.shape[-2:] != max_size:
                f = F.interpolate(f, size=max_size, mode="bilinear", align_corners=False)
            interp_feats.append(f)
        return interp_feats

    def encode_mae(self, x, use_quantizer=None, **_ignored_kwargs):
        from ..self_supervised.mae_masking import mae_random_masking, restore_sequence_by_ids

        mae_type = self.trans_enc_cfg.pretrained_type
        mask_ratio = self.trans_enc_cfg.mae_mask_ratio
        mask_type = self.trans_enc_cfg.mae_mask_type
        mae_pixio_mask_grid = self.trans_enc_cfg.mae_pixio_mask_grid
        assert mae_type in ("latent_mae", "pixel_mae"), (
            f"MAE type supports type=(latent_mae, pixel_mae), but got {mae_type}"
        )
        enc_out = edict(to_dec=None, latent=None, q_loss=None, q_loss_breakdown=None, low_lvl_z=None, sem_z=None)

        # Forward the CNN encoder
        z_low_lvl = self.encoder.encoder(x)

        if self.latent_bottleneck_type == "before_semantic":
            quant_ret = self._apply_quant_conv_quantizer(z_low_lvl, use_quantizer)
            if isinstance(quant_ret, tuple):
                h = quant_ret[0]
            else:
                h = quant_ret
            enc_out.latent = h
            h = self.latent_aug(h)

            ##### Do post-quant conv
            if self.st_skip_sem_decoder:
                h = self.decoder.quant_conv["post_quant_conv"](h)  # sent to semantic encoder
                h_skipped = h.clone()
                enc_out.h_skipped = h_skipped
            else:
                h = self.decoder.quant_conv(h)
        else:
            h = z_low_lvl
            h = self.latent_aug(h)

        # For positional embedding, the original mae first add the pe and
        # then mask, here we left the pe inside the transformer, and pass the int mask
        # indices in the using einx to take the cooresponding pe/rope and then add them.

        # prepare the encoder's masks: L_masked_x = L_x * mask_ratio
        ps = self.semantic_enc_transformer.patch_size
        hw = h.shape[-2:]
        bchw = (h.shape[0], h.shape[1], hw[0] // ps, hw[1] // ps)
        _, mask, ids_keep, ids_restore = mae_random_masking(mask_type, mask_ratio, mae_pixio_mask_grid, bchw=bchw)
        mask, ids_keep, ids_restore = mask.to(h.device), ids_keep.to(h.device), ids_restore.to(h.device)

        _, terms = self.semantic_enc_transformer._forward_pretrained_backbone(  # type: ignore
            z_low_lvl, masks=ids_keep, ids_restore=ids_restore, pretrained_task=[mae_type]
        )
        if mae_type == "latent_mae":
            mae_decode_out = terms.mae_decode_out  # [b,l,c]
            # patch 2d low-level latent
            z_tgt = rearrange(z_low_lvl, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=ps, p2=ps)
            loss = (mae_decode_out - z_tgt) ** 2
            loss = loss.mean(dim=-1)
            loss = (loss * mask).sum() / mask.sum()
            return loss, mae_decode_out
        else:
            # mae_type == "pixel_mae"
            # continue to decode to pixel space
            # 1. Unpatchify the transformer output to 2D latent
            # terms.mae_decode_out_2d is already [B, C, H, W] from naflex.py
            h = terms.mae_decode_out_2d
            enc_out.to_dec = h

            # 2. Decode using the cnn decoder
            decode_out = self.decode(enc_out, inp_shape=x.shape)
            recon = decode_out.recon

            # 3. Prepare the pixel-level mask
            mask_2d = mask.reshape(mask.shape[0], 1, bchw[-2], bchw[-1])
            mask_up = F.interpolate(mask_2d, size=x.shape[-2:], mode="nearest")

            # 4. Calculate loss
            loss = (recon - x) ** 2
            # Normalize by total number of masked elements (pixels * channels)
            loss = (loss * mask_up).sum() / (mask_up.sum() * x.shape[1] + 1e-6)
            return loss, recon

    def encode_ibot(self, x, use_quantizer=None, masks: Tensor | None = None, mask_indices=None, **_ignored_kwargs):
        z_low_lvl = self.encoder.encoder(x)

        _, terms = self.semantic_enc_transformer._forward_pretrained_backbone(  # type: ignore
            z_low_lvl, masks=masks, masks_indices=mask_indices, pretrained_task=["ibot"]
        )
        return terms.ibot_proj

    def encode_ijepa(self, x, use_quantizer=None, jepa_masks=None, **_ignored_kwargs):
        """
        x -> CNN (with full image) -> transformer (with unmasked patches) -> 1d tokens.
        no quantization, no quant_conv, no other projections.
        """
        # Low-level encoder
        z_low_lvl = self.encoder.encoder(x)
        # Semantic encoder
        _, terms = self.semantic_enc_transformer._forward_pretrained_backbone(  # type: ignore
            z_low_lvl, masks=jepa_masks, pretrained_task=["ijepa"]
        )
        # no quant_conv here
        return terms.ijepa_feat

    def encode_lejepa_latent(self, x, **_ignored_kwargs):
        assert self.latent_bottleneck_type == "before_semantic"
        # Low-level encoder
        z_low_lvl = self.encoder.encoder(x)
        # quant conv
        h = self.encoder.quant_conv(z_low_lvl)
        h_pool = h.mean(dim=(2, 3))
        h_proj = self.lejepa_projector_latent(h_pool)
        return h_proj

    def encode_lejepa(self, x, **_ignored_kwargs):
        """
        LeJEPA encoding with augmentated global and local views.
        """
        # Low-level encoder
        z_low_lvl = self.encoder.encoder(x)

        # Semantic encoder
        # z_low_lvl -> transformer backbone -> projector -> z_proj
        _, terms = self.semantic_enc_transformer._forward_pretrained_backbone(  # type: ignore
            z_low_lvl, masks=None, pretrained_task=["lejepa"]
        )

        return terms.lejepa_proj

    def encode_dino_cls(self, x: torch.Tensor) -> Tensor:
        """返回semantic encoder输出的normed CLS token (B, D)，用于DINO cls token loss。"""
        z_low_lvl = self.encoder.encoder(x)

        if self.latent_bottleneck_type == "before_semantic":
            h = self.encoder.quant_conv(z_low_lvl)
            h = self.latent_aug(h)
            if self.st_skip_sem_decoder:
                h = self.decoder.quant_conv["post_quant_conv"](h)
            else:
                h = self.decoder.quant_conv(h)
        else:
            h = z_low_lvl
            h = self.latent_aug(h)

        tokens = self.semantic_enc_transformer.forward_features(h, masks=None)  # type: ignore[arg-type]
        tokens = cast(torch.Tensor, tokens)
        if getattr(self.semantic_enc_transformer, "num_prefix_tokens", 0) <= 0:
            raise ValueError("semantic_enc_transformer未启用cls token（请在配置中设置`class_token: true`）。")
        return tokens[:, 0, :]

    def encode_nepa(self, x, **_ignored_kwargs) -> dict:
        """Encode for NePA (Next Embedding Prediction) pretraining.

        NePA trains the transformer to predict the next position's input embedding
        from the current position's output, using causal attention and cosine
        similarity loss in latent space.

        Args:
            x: Input image tensor [B, C, H, W]

        Returns:
            dict with:
                - nepa_loss: scalar loss value
                - nepa_h_in: [B, T, D] input embeddings (optional)
                - nepa_h_out: [B, T, D] output hidden states (optional)
        """
        # Low-level encoder (CNN)
        z_low_lvl = self.encoder.encoder(x)

        # Apply quantization and preprocessing based on bottleneck type
        if self.latent_bottleneck_type == "before_semantic":
            h = self.encoder.quant_conv(z_low_lvl)
            h = self.latent_aug(h)
            if self.st_skip_sem_decoder:
                h = self.decoder.quant_conv["post_quant_conv"](h)
            else:
                h = self.decoder.quant_conv(h)
        else:
            # after_semantic: use raw low-level features
            h = z_low_lvl
            h = self.latent_aug(h)

        # Forward through semantic encoder with NePA
        _, terms = self.semantic_enc_transformer._forward_pretrained_backbone(  # type: ignore
            h, masks=None, pretrained_task=["nepa"], nepa_input=h
        )

        return {
            "nepa_loss": terms.nepa_loss,
            "nepa_h_in": terms.get("nepa_h_in"),
            "nepa_h_out": terms.get("nepa_h_out"),
        }

    def _forward_low_level_encoder(self, x, get_intermediate_features=False):
        ############ Low-level encoder
        cache_low_lvl = None
        only_cache_h = self.cache_layers["low_level"] == -1
        if self.training or get_intermediate_features:
            low_lvl_out = self.encoder.encoder(x, ret_interm_feats=not only_cache_h)
            if only_cache_h:
                assert torch.is_tensor(low_lvl_out)
                cache_low_lvl = z_low_lvl = low_lvl_out
            elif isinstance(low_lvl_out, (list, tuple)):
                z_low_lvl, cache_low_lvl = low_lvl_out
            else:
                raise NotImplementedError
        else:
            z_low_lvl = self.encoder.encoder(x)

        return z_low_lvl, cache_low_lvl

    def _forward_semantic_encoder(self, x, get_intermediate_features=False):
        ########### Forward semantic transformer encoder
        cache_semantic = None
        if not self.training and not get_intermediate_features:  # is eval
            z_semantic = self.semantic_enc_transformer(x)
        elif self.cache_layers["semantic"] == -1 and get_intermediate_features:
            z_semantic = self.semantic_enc_transformer(x)
            cache_semantic = z_semantic
        else:
            # forward the intermediate features
            _model_kwargs = dict(
                x=x,
                indices=self.cache_layers["semantic"],
                return_prefix_tokens=False,
                norm=True,
                output_fmt="NCHW",
                output_dict=False,
            )
            # forward intermediates does not support jepa masks
            z_semantic, cache_semantic = self.semantic_enc_transformer.forward_intermediates(**_model_kwargs)  # type: ignore

            # Add head forward
            z_semantic = z_semantic[:, self.semantic_enc_transformer.num_prefix_tokens :]  # type: ignore
            z_semantic = self.semantic_enc_transformer._forward_after_backbone(
                z_semantic,
                hw=self.semantic_enc_transformer._get_output_shape(x),
            )
            # Stack the intermidates features: [n_cache_layers, b, c, h, w]
            # cache_semantic = torch.stack(cache_semantic, dim=0)

        return z_semantic, cache_semantic

    def _apply_quant_conv_quantizer(self, z, use_quantizer=None):
        # z token -> quant conv -> latent
        h = self.encoder.quant_conv(z)
        # Quantization
        maybe_q_ret = self.apply_quantizer(h, use_quantizer, cache_type=None)
        return maybe_q_ret

    def _encode_bottleneck_after_sem(
        self,
        x,
        use_quantizer=None,
        get_intermediate_features=False,
    ):
        enc_out = edict(
            to_dec=None,
            latent=None,
            latent_before_quantizer=None,
            q_loss=None,
            q_loss_breakdown=None,
            low_lvl_z=None,
            sem_z=None,
        )

        ######## Low-level encoder
        z_low_lvl, cache_low_lvl = self._forward_low_level_encoder(
            x, get_intermediate_features=get_intermediate_features
        )

        ######### Semantic encoder
        z_semantic, cache_semantic = self._forward_semantic_encoder(
            z_low_lvl, get_intermediate_features=get_intermediate_features
        )

        ######## Apply CNN encoder - semantic transformer encoder - quant conv
        quant_ret = self._apply_quant_conv_quantizer(z_semantic, use_quantizer)

        # Do cache here
        self.z = cache_low_lvl  # [b, c, h, w]
        self.sem_z = cache_semantic
        enc_out.update(low_lvl_z=cache_low_lvl, sem_z=cache_semantic, latent_before_quantizer=z_semantic)

        ##### Do quant return out
        if isinstance(quant_ret, tuple):
            h, q_loss, loss_breakdown = quant_ret
            enc_out.update(
                latent=h,
                q_loss=q_loss,
                q_loss_breakdown=loss_breakdown,
            )

        ##### z augmentions (noise adding or channel dropping).
        h = self.latent_aug(quant_ret)
        enc_out.update(latent=h, to_dec=h)

        return enc_out

    def _encode_bottleneck_before_sem(
        self,
        x,
        use_quantizer=None,
        get_intermediate_features=False,
    ):
        enc_out = edict(
            to_dec=None,
            latent=None,
            latent_before_quantizer=None,
            q_loss=None,
            q_loss_breakdown=None,
            low_lvl_z=None,
            sem_z=None,
        )

        ######## Low-level encoder
        z_low_lvl, cache_low_lvl = self._forward_low_level_encoder(
            x, get_intermediate_features=get_intermediate_features
        )
        enc_out["latent_before_quantizer"] = z_low_lvl

        ######## Apply CNN encoder - quant conv
        quant_ret = self._apply_quant_conv_quantizer(z_low_lvl, use_quantizer)
        if isinstance(quant_ret, tuple):
            h, q_loss, loss_breakdown = quant_ret
            enc_out.update(q_loss=q_loss, q_loss_breakdown=loss_breakdown)
        else:
            assert torch.is_tensor(quant_ret)
            h = quant_ret
        enc_out["latent"] = h  # CNN's output is latent

        ##### z augmentions (noise adding or channel dropping)
        h = self.latent_aug(h)

        ##### Do post-quant conv
        if self.st_skip_sem_decoder:
            h = self.decoder.quant_conv["post_quant_conv"](h)  # sent to semantic encoder
            h_skipped = h.clone()
            enc_out.h_skipped = h_skipped  #  z_dim
        else:
            h = self.decoder.quant_conv(h)

        ######### Semantic encoder
        # served as decoder actually ...
        z_semantic, cache_semantic = self._forward_semantic_encoder(
            h, get_intermediate_features=get_intermediate_features
        )
        enc_out["to_dec"] = z_semantic  # sem_z is sent to 'decoder'

        # Do cache here
        self.z = cache_low_lvl  # [b, c, h, w]
        self.sem_z = cache_semantic
        enc_out.update(low_lvl_z=cache_low_lvl, sem_z=cache_semantic)

        return enc_out

    def encode(self, x, use_quantizer=None, get_intermediate_features=False):
        inputs = dict(
            x=x,
            use_quantizer=use_quantizer,
            get_intermediate_features=get_intermediate_features,
        )
        t = self.latent_bottleneck_type
        if t == "after_semantic":
            return self._encode_bottleneck_after_sem(**inputs)
        elif t == "before_semantic":
            return self._encode_bottleneck_before_sem(**inputs)
        else:
            raise ValueError(f"Unknown latent bottleneck type: {t}")

    def decode(  # type: ignore
        self,
        inp: dict,
        inp_shape: Annotated[Union[torch.Size, int, tuple], "bs,c,h,w or bs,c or c"],
        clamp=False,
    ) -> dict:
        """
        Decode the latent into the corresponding channels image.
        Output the latent, to_dec, q_loss, q_loss_breakdown, recon, deep_supervision_outputs (optional)
        """
        out_d = edict(**inp)  # quantization-related, to_dec, latent
        h = inp["to_dec"]

        # Decoder
        chan = inp_shape[1] if isinstance(inp_shape, (torch.Size, tuple)) else inp_shape

        # Apply post-quant conv - semantic transformer decoder - CNN decoder
        if self.st_skip_sem_decoder:
            if self.latent_bottleneck_type == "after_semantic":
                h = self.decoder.quant_conv["post_quant_conv"](h)
                h_skipped = h.clone()  # z_dim
            else:
                h_skipped = inp.get("h_skipped")  # z_dim
                assert h_skipped is not None
        else:
            h = self.decoder.quant_conv(h)

        # Apply semantic transformer decoder if it exists
        if self.semantic_transformer_dec is not None:
            h = self.semantic_transformer_dec(h)

        # Cat the skipped latent
        if self.st_skip_sem_decoder:
            h = torch.cat([h, h_skipped], dim=1)
            h = self.decoder.quant_conv["st_cat_conv"](h)

        # Decode using the CNN decoder
        dec = self.decoder.decoder(h, chan, ret_all_res_features=self._is_deep_supervision)

        # Deep supervision output to 'dec'
        dec: dict[str, Tensor | list[Tensor]]
        if self._is_deep_supervision:
            recon, interms = dec  # decoder multi-scale feature as 'interms'
            deep_sup_outputs = self._forward_deep_supervised_heads(interms)
            if clamp:
                recon = recon.clamp(-1, 1)
            out_d.update(recon=recon, deep_supervision_outputs=deep_sup_outputs)
        else:
            out_d["recon"] = dec

        if clamp:
            out_d.recon = out_d.recon.clamp(-1, 1)

        return out_d

    def _forward_deep_supervised_heads(self, interms: list[Tensor]):
        """Forward the deep supervised heads given the intermediate features."""
        deep_sup_outputs = []

        for i, head in enumerate(self.deep_supervised_heads):
            interm_feat = interms[i]

            if self.deep_supervision_type == "sum_previous_out" and i > 0:
                prev_interm_feat = interms[i - 1]
                interm_feat = interm_feat + head.prev_out_proj(prev_interm_feat)

            out = head.main(interm_feat)
            deep_sup_outputs.append(out)

        return deep_sup_outputs

    @classmethod
    @function_config_to_basic_types
    def create_model(
        cls,
        cnn_cfg,
        trans_enc_cfg,
        trans_dec_cfg=None,
        distillation_cfg: dict | None = None,
        hybrid_tokenizer_cfg: dict | None = None,
        # overrides for the main cnn_cfg, if None, use the default values
        # in cnn_cfg
        cnn_enc_cfg: dict | None = None,
        cnn_dec_cfg: dict | None = None,
    ):
        cnn_cfg = dataclass_from_dict(ContinuousTokenizerConfig, cnn_cfg)
        trans_enc_cfg = dataclass_from_dict(NaFlexVitCfg, trans_enc_cfg)
        if trans_dec_cfg is not None:
            trans_dec_cfg = dataclass_from_dict(NaFlexVitCfg, trans_dec_cfg)
        if cnn_enc_cfg is not None:
            cnn_enc_cfg = dataclass_from_dict(EncoderDecoderConfig, cnn_enc_cfg)
        if cnn_dec_cfg is not None:
            cnn_dec_cfg = dataclass_from_dict(EncoderDecoderConfig, cnn_dec_cfg)

        return cls(
            cnn_cfg,
            trans_enc_cfg,
            trans_dec_cfg,
            cnn_enc_cfg=cnn_enc_cfg,
            cnn_dec_cfg=cnn_dec_cfg,
            distillation_cfg=edict(distillation_cfg),
            hybrid_tokenizer_cfg=edict(hybrid_tokenizer_cfg),
        )

    @torch.autocast("cuda", dtype=torch.bfloat16)
    def get_repa_feature(
        self,
        force_to: bool = False,
    ) -> tuple[Tensor | list[Tensor], Tensor | list[Tensor]] | None:
        if not force_to:
            if not self._use_repa_loss:
                return None
            elif not self.training:
                # is not training, do not proj since we do not need to train using repa features
                return None

        # z and sem_z
        z, sem_z = self.z, self.sem_z
        assert self._vf_on_z_or_module == "z"
        assert z is not None and sem_z is not None, "No cached z or sem_z"

        ###### Low-level repa projection
        if self.low_lvl_repa_proj_is_multi:
            assert isinstance(z, list)
            low_lvl_z_proj = [
                self._repa_proj["low_lvl_repa_proj"][i](z[i]) for i in range(len(self.low_lvl_repa_proj_chans))
            ]
        else:
            assert torch.is_tensor(z)
            low_lvl_z_proj = self._repa_proj["low_lvl_repa_proj"](z)

        ######## Semantic feature repa projection
        if self.sem_repa_proj_is_multi:
            assert isinstance(sem_z, list)
            sem_z_proj = [self._repa_proj["sem_repa_proj"][i](sem_z[i]) for i in range(len(sem_z))]
        else:
            assert torch.is_tensor(sem_z)
            sem_z_proj = self._repa_proj["sem_repa_proj"](sem_z)

        # return tuple of Tensors or list of Tensors
        return low_lvl_z_proj, sem_z_proj

    def _build_feature_align_mlp(self, proj_type: str = "norm_first_force_conv"):
        logger.info(f"[Repa proj]: build mlp type {proj_type}")

        # Set the low-level cache layers for the feature alignment
        self._set_low_level_proj_chans()

        assert self._vf_on_z_or_module == "z", f"Only support z for _vf_on_z_or_modulebut got {self._vf_on_z_or_module}"

        if self._use_repa_loss:
            # Low-level projection
            self.low_lvl_repa_proj_is_multi = isinstance(self.cache_layers["low_level"], (tuple, list))
            if not self.low_lvl_repa_proj_is_multi:
                low_lvl_z_proj = build_mlp(
                    self.cnn_cfg.model.z_channels, self._dino_feature_dim, self._dino_feature_dim, proj_type=proj_type
                )
            else:
                assert len(self.low_lvl_repa_proj_chans) == len(self.cache_layers["low_level"]), (
                    f"Length of low_lvl_repa_proj_chans {len(self.low_lvl_repa_proj_chans)} "
                    f"should match length of cached low level layers "
                    f"{len(self.cache_layers['low_level'])}"
                )

                low_lvl_z_proj = nn.ModuleList()
                for i in range(len(self.cache_layers["low_level"])):
                    proj_ = build_mlp(
                        self.low_lvl_repa_proj_chans[i],
                        self._dino_feature_dim,
                        self._dino_feature_dim,
                        proj_type=proj_type,
                    )
                    low_lvl_z_proj.append(proj_)

            # Semantic projection
            sem_cache_layers = self.cache_layers["semantic"]
            is_multi_layer_cached = isinstance(sem_cache_layers, (tuple, list))
            self.sem_repa_proj_is_multi = is_multi_layer_cached
            if is_multi_layer_cached:
                sem_z_proj = nn.ModuleList()
                for _ in range(len(sem_cache_layers)):
                    sem_z_proj.append(
                        build_mlp(
                            # since the transformer embedding layer has the same channels
                            self.trans_enc_cfg.embed_dim,
                            self._semantic_feature_dim,
                            self._semantic_feature_dim,
                            proj_type=proj_type,
                        )
                    )
            else:
                sem_z_proj = build_mlp(
                    self.trans_enc_cfg.embed_dim,
                    self._semantic_feature_dim,
                    self._semantic_feature_dim,
                    proj_type=proj_type,
                )
            self._repa_proj = nn.ModuleDict(
                {
                    "low_lvl_repa_proj": low_lvl_z_proj,
                    "sem_repa_proj": sem_z_proj,
                }
            )

    def _set_grad_zero_for_ddp(self, starts_with: list[str] | None = None):
        """Set the gradients to zero for DDP training.
        starts_with: e.g., ['decoder', 'semantic_transformer_dec']
        """
        with torch.no_grad():
            for name, param in self.named_parameters():
                if starts_with is None or any(name.startswith(sw) for sw in starts_with):
                    if param.requires_grad and param.grad is None:
                        param.grad = torch.zeros_like(param)


###### Configs #######


def hybrid_distillation_f16_config(
    in_chans: int = 384,
    latent_chans: int = 64,
    z_chans: int = 512,
    backbone_embed_dim: int = 1024,
    backbone_enc_n_layers: int = 12,
    backbone_dec_n_layers: int = 12,
    norm_layer: str = "layernorm",
    use_repa_loss: bool = True,
    pretrained_path: str = "",
):
    cnn_cfg = dict(
        model=dict(
            resolution=512,
            in_channels=in_chans,
            out_channels=in_chans,
            z_channels=z_chans,
            latent_channels=latent_chans,
            channels=128,
            channels_mult=[2, 4, 4],
            num_res_blocks=2,
            attn_resolutions=[],
            dropout=0.0,
            spatial_compression=8,
            patch_size=1,
            block_name="res_block",
            norm_type="gn",
            norm_groups=32,
            adaptive_mode="interp",
            downsample_kwargs=dict(padconv_use_manually_pad=False),
            upsample_kwargs=dict(interp_type="nearest_interp"),
            per_layer_noise=False,
        ),
        quantizer_type=None,
        vf_on_z_or_module="z",
        use_repa_loss=use_repa_loss,
        dino_feature_dim=1024,
        decoder_type="default",
        use_channel_drop=False,
        channel_drop_config=dict(
            drop_type=[16, 32, 48],
            max_channels=64,
            drop_prob=0.5,
        ),
        loading_type="hybrid_pretrained",
        uni_path=pretrained_path,
    )

    trans_enc_cfg = dict(
        embed_dim=backbone_embed_dim,
        depth=backbone_enc_n_layers,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        patch_size=2,
        norm_layer=norm_layer,
        pos_embed="learned",
        pos_embed_grid_size=[32, 32],
        rope_type="axial",
        img_size=32,
        in_chans=z_chans,
        out_chans=z_chans,
        unpatch_size=1,
        reg_tokens=0,
        compile_model=False,
    )

    trans_dec_cfg = dict(
        embed_dim=backbone_embed_dim,
        depth=backbone_dec_n_layers,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        patch_size=1,
        norm_layer=norm_layer,
        pos_embed="learned",
        pos_embed_grid_size=[32, 32],
        rope_type="axial",
        img_size=32,
        in_chans=512,
        out_chans=512,
        unpatch_size=2,
        reg_tokens=0,
        compile_model=False,
    )

    distillation_cfg = dict(
        dino_feature_dim=1024,
        semantic_feature_dim=1024,
    )

    cfg = OmegaConf.create(
        dict(
            cnn_cfg=cnn_cfg,
            trans_enc_cfg=trans_enc_cfg,
            trans_dec_cfg=trans_dec_cfg,
            hybrid_tokenizer_cfg=None,
            distillation_cfg=distillation_cfg,
        )
    )

    return cfg


def hybrid_ijepa_f8_config(
    in_chans: int = 512,
    latent_chans: int = 64,
    z_chans: int = 768,
    backbone_embed_dim: int = 1024,
    backbone_enc_n_layers: int = 12,
    backbone_dec_n_layers: int = 12,
    norm_layer: str = "rmsnorm",
    use_repa_loss: bool = True,
    pretrained_path: str = "",
):
    """
    Configuration for hybrid IJEPA tokenizer model.

    Returns configuration dictionary with CNN, transformer encoder/decoder,
    and distillation settings.

    Network config of hybrid CNN and transformer (as decoder actually)
    Latent is from the CNN encoder, and transformer decoder and CNN decoder
    are used for semantic feature distillation and image reconstruction.
    """
    # Create the model with correct configuration structure
    cnn_cfg = {
        "model": {
            "resolution": 1024,
            "in_channels": in_chans,
            "out_channels": in_chans,
            "z_channels": z_chans,  # Connect encoder and decoder
            "latent_channels": latent_chans,
            "channels": 128,  # Base channels
            "channels_mult": [2, 4, 4],  # Channel multiplier
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "dropout": 0.0,
            "spatial_compression": 8,  # Default spatial compression
            "patch_size": 1,  # Default patch size
            "block_name": "res_block",  # Default block type
            "norm_type": "rmsnorm2d",  # Default normalization
            "act_type": "silu",
            "norm_groups": 32,  # Default number of groups
            "adaptive_mode": "interp",
            "downsample_kwargs": {"padconv_use_manually_pad": False},
            "upsample_kwargs": {"interp_type": "nearest_interp"},
        },
        "quantizer_type": None,  # No quantizer for basic test
        "vf_on_z_or_module": "z",  # Visual feature on z
        "use_repa_loss": use_repa_loss,
        "dino_feature_dim": 1024,
        "loading_type": "hybrid_pretrained",
        "uni_path": pretrained_path,
    }

    trans_enc_cfg = {
        "embed_dim": 1152,
        "depth": 24,  # vit intern 300m -> embed_dim=1024, depth=24, num_heads=16
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "patch_size": 2,
        "norm_layer": norm_layer,  # Use Flash RMSNorm for compatibility
        "pos_embed": "learned",  # Use learned position embeddings
        "pos_embed_grid_size": (32, 32),  # Grid size for position embedding
        # Additional required fields from merged configuration
        "img_size": 32,  # Expected by NaFlexVitCfg
        "in_chans": z_chans,  # Should match z_channels from transformer
        "out_chans": z_chans,  # Output channels
        "unpatch_size": 2,  # 2->1
        "reg_tokens": 4,
        "rope_type": "axial",
        "attn_type": "gated",
        "pretrained_type": ["ijepa"],
    }

    cfg = OmegaConf.create(
        dict(
            cnn_cfg=cnn_cfg,
            trans_enc_cfg=trans_enc_cfg,
            trans_dec_cfg=None,
            distillation_cfg={
                "dino_feature_dim": 1024,
                "semantic_feature_dim": 1024,
                "cache_layers": {"low_level": [0, 1, 2, -1], "semantic": [5, 11, 17, 23]},
            },
            hybrid_tokenizer_cfg={
                "latent_bottleneck_type": "before_semantic",
                "latent_straight_through_skip": True,
            },
        )
    )

    return cfg


###### Tests ##########


def test_model_forward_backward(
    model_type: str = "hybrid_enc_dec_transformer",
    real_data: str | None = None,
    use_optim: bool = False,
    device: str = "cuda",
    save_img_dir: str | None = None,
    rgb_chans: list[int] = [4, 2, 0],
    dtype: torch.dtype = torch.bfloat16,
    upscale: int = 1,
    fake_img_shape: tuple = (1, 12, 256, 256),
    compute_mean_std: bool = False,
    max_iters: int = 100,
    check_grad: bool = False,
    save_pca_vis: bool = False,
    pca_type: str = "proj",
):
    """Test the forward and backward pass of the CosmosHybridTokenizer model.

    This function creates a model, tests forward pass in both eval and train modes,
    and verifies that gradients are computed correctly during backward pass.

    Parameters
    ----------
    real_data : str | None, optional
        Path to real data file or dataset name, by default None
    use_optim : bool, optional
        Whether to use optimizer for training test, by default False
    device : str, optional
        Device to run on, by default "cuda"
    save_img_dir : str | None, optional
        Directory to save reconstruction images, by default None
    rgb_chans : list[int], optional
        RGB channels for visualization, by default [4, 2, 0]
    dtype : torch.dtype, optional
        Data type for computation, by default torch.bfloat16
    upscale : int, optional
        Upscale factor for input images, by default 1
    fake_img_shape : tuple, optional
        Shape for fake input data, by default (1, 12, 256, 256)
    compute_mean_std : bool, optional
        Whether to compute mean and std of latent, by default False
    max_iters : int, optional
        Maximum iterations for dataset, by default 100
    check_grad : bool, optional
        Whether to check gradients, by default False
    save_pca_vis : bool, optional
        Whether to save PCA visualization, by default False
    pca_type : str, optional
        Type of PCA visualization ('proj' or 'z'), by default "proj"
    """
    from contextlib import nullcontext
    from pathlib import Path

    from fvcore.nn import parameter_count_table
    from PIL import Image
    from torchmetrics.aggregation import MeanMetric
    from torchmetrics.image import PeakSignalNoiseRatio
    from torchvision.utils import make_grid
    from tqdm import tqdm

    from src.data.hyperspectral_loader import get_fast_test_hyperspectral_data
    from src.data.litdata_hyperloader import get_fast_test_hyper_litdata_load

    device = torch.device("cuda")

    cfg = hybrid_ijepa_f8_config(
        # pretrained_path="runs/stage1_cosmos_hybrid/2025-12-09_18-38-25_hybrid_cosmos_f16c64_jepa_pretrained/ema/tokenizer/model.safetensors",
        pretrained_path="runs/stage1_cosmos_hybrid/2025-12-21_05-34-15_hybrid_cosmos_f16c64_ijepa_pretrained_sem/ema/tokenizer/model.safetensors",
        latent_chans=32,
        use_repa_loss=True,
    )
    # cfg = hybrid_distillation_f16_config(
    #     pretrained_path="runs/stage1_cosmos_hybrid/2025-11-15_22-11-24_hybrid_cosmos_f16c64/ema/tokenizer/model.safetensors",
    #     use_repa_loss=True,
    # )
    model: CosmosHybridTokenizer = CosmosHybridTokenizer.create_model(
        cfg.cnn_cfg,
        cfg.trans_enc_cfg,
        trans_dec_cfg=cfg.trans_dec_cfg,
        hybrid_tokenizer_cfg=cfg.hybrid_tokenizer_cfg,
        distillation_cfg=cfg.distillation_cfg,
    )
    model = model.to(device, torch.bfloat16)  # Move model to device (CUDA or CPU)
    if save_pca_vis:
        model.train()
    else:
        model.eval()
    logger.info("Model created successfully!")
    # model.set_grad_checkpointing(enabled=True)

    # Print parameter table using fvcore
    logger.info("Model parameter table:")
    logger.info(parameter_count_table(model))

    # Handle real data or fake data
    is_itered = False
    if real_data is not None:
        if Path(real_data).exists():
            # only support RGB image
            x = Image.open(real_data).convert("RGB")
            x = torch.from_numpy(np.array(x)).permute(2, 0, 1).unsqueeze(0).float().to(device)
            x = x / 255.0
            x = x * 2 - 1  # normalize to [-1, 1]
            iterations = [x]
            is_itered = True
        else:
            # dl = get_fast_test_hyperspectral_data(batch_size=1, data_type=real_data)  # type: ignore
            dl = get_fast_test_hyper_litdata_load(real_data, batch_size=1, stream_ds_kwargs={"shuffle": True})[1]  # type: ignore
            iterations = dl
    else:
        x = torch.randn(*fake_img_shape).to(device, dtype)
        iterations = [x]

    if not is_itered and upscale != 1:
        x = torch.nn.functional.interpolate(x, scale_factor=upscale, align_corners=True, mode="bicubic")

    if use_optim:
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    if compute_mean_std:
        mean_lst = []
        std_lst = []
    ctx = torch.no_grad if not (use_optim or check_grad) else torch.enable_grad

    with torch.autocast(str(device), dtype=torch.bfloat16):
        for index, x in (tbar := tqdm(enumerate(iterations))):
            if index < 10:
                continue

            with ctx():
                if isinstance(x, dict):
                    x = x["img"].to(device, dtype)
                elif x.dtype != dtype:
                    x = x.to(dtype)

                # Forward pass
                if not (use_optim or check_grad):
                    # Eval mode
                    with torch.no_grad():
                        encoded = model.encode(x)
                        encoded_tensor = encoded.to_dec

                        # Decode with proper input shape
                        decoded = model.decode(encoded, x.shape)
                        decoded_tensor = decoded.recon

                        logger.info(
                            f"min: {encoded_tensor.min()}, max: {encoded_tensor.max()}, mean: {encoded_tensor.mean()}, std: {encoded_tensor.std()}",
                            tqdm=True,
                        )
                else:
                    # Training mode
                    encoded = model.encode(x)
                    encoded_tensor, q_loss, loss_breakdown = encoded.to_dec, encoded.q_loss, encoded.q_loss_breakdown
                    decoded_tensor = model.decode(encoded, x.shape).recon

            decoded_tensor.clamp_(-1, 1)

            # Compute mean and std of the latent
            if compute_mean_std:
                h = encoded_tensor
                mean_c, std_c = h.mean((0, -2, -1)), h.std((0, -2, -1))  # per-channel value
                mean_lst.append(mean_c)
                std_lst.append(std_c)

            # save reconstruction
            if save_img_dir is not None:
                Path(save_img_dir).mkdir(parents=True, exist_ok=True)

                def plot_img(img, path, name_suffix=""):
                    y_grid = make_grid(img.float(), nrow=1, padding=2)
                    # Handle case where image has fewer channels than rgb_chans
                    available_chans = min(len(rgb_chans), y_grid.shape[0])
                    selected_chans = [
                        rgb_chans[i] if rgb_chans[i] < y_grid.shape[0] else i for i in range(available_chans)
                    ]
                    y_grid = y_grid[selected_chans].permute(1, 2, 0).detach().cpu().numpy()  # [h, w, 3]
                    y_grid = (y_grid + 1) / 2
                    y_grid = (y_grid * 255.0).astype(np.uint8)
                    Image.fromarray(y_grid).save(path)
                    logger.info(f"save reconstruction image {name_suffix}")

                plot_img(
                    decoded_tensor, Path(save_img_dir) / f"recon_{real_data or 'fake'}_{index}.png", "reconstruction"
                )
                plot_img(x, Path(save_img_dir) / f"gt_{real_data or 'fake'}_{index}.png", "ground truth")

            # psnr
            if real_data:
                psnr_val = metric((x + 1) / 2, (decoded_tensor + 1) / 2)
                logger.info(f"PSNR: {psnr_val:.4f} - shape: {x.shape}", tqdm=True)
                tbar.set_description(f"PSNR: {psnr_val:.4f} - shape: {x.shape}")

            if use_optim:
                opt.zero_grad()
                decoded_tensor.mean().backward()
                opt.step()

            if check_grad:
                for n, p in model.named_parameters():
                    if p.grad is None:
                        logger.warning(f"{n} grad is None")

            if max_iters <= index:
                break

        if real_data:
            result = metric.compute()
            logger.info(f"Average PSNR: {result.item() if hasattr(result, 'item') else result}")

    # print mean and std of the latent
    if compute_mean_std:
        m = torch.stack(mean_lst).mean(dim=0)
        s = torch.stack(std_lst).mean(dim=0)
        logger.info(f"mean of the latent: {m.tolist()}")
        logger.info(f"std of the latent: {s.tolist()}")

    # PCA visualization
    if save_pca_vis:
        Path("tmp").mkdir(exist_ok=True)
        if pca_type == "proj":
            feat = model.get_repa_feature(force_to=True)
            if feat is not None:
                # feat is a tuple of (low_lvl_z_proj, sem_z_proj)
                if isinstance(feat, tuple) and len(feat) >= 2:
                    # Use semantic features for PCA
                    # feat_to_use = feat[1] if feat[1] is not None else feat[0]
                    # if isinstance(feat_to_use, list):
                    #     feat_to_use = feat_to_use[2]
                    feat_to_use = feat
                else:
                    feat_to_use = feat
            else:
                logger.warning("No repa features available for PCA visualization")
                feat_to_use = None
        else:
            # Use latent z features
            with torch.no_grad():
                if not is_itered:
                    # Use the last processed x for PCA
                    model.eval()
                    feat_encoded = model.encode(x)
                    feat_to_use = feat_encoded.to_dec
                    if isinstance(feat_to_use, tuple):
                        feat_to_use = feat_to_use[0]
                else:
                    feat_to_use = encoded.latent
                    # logger.warning("PCA with 'z' type not supported for iterated data in current implementation")

        if isinstance(feat_to_use, tuple):
            # cat all together
            feat_to_use = [*feat_to_use[0], *feat_to_use[1]]

        if feat_to_use is not None:
            assert feat_to_use is not None, "Feature should not be None for PCA"
            # plot
            import matplotlib.pyplot as plt

            if torch.is_tensor(feat_to_use):
                logger.info(f"Got 1 feature to PCA visualization with shape {feat_to_use.shape}")
                feat_pca = _get_pca_vis(feat_to_use)
                n_cols, nrows = 2, 1
                fig, axes = plt.subplots(figsize=(8, 8), ncols=n_cols, nrows=nrows)
                axes[0].imshow(feat_pca[:, :, :3])
                axes[0].set_title("PCA")
                x_vis = (x.float() + 1) / 2
                x_vis = x_vis[0].cpu().numpy().transpose(1, 2, 0)
                axes[1].imshow(x_vis)
                axes[1].set_title("Input Image")
            else:
                logger.info(f"Got {len(feat_to_use)} feature maps for PCA visualization")
                # is a list
                feat_pca = [_get_pca_vis(fea)[0] for fea in feat_to_use]
                n_cols, nrows = 5, 2
                fig, axes = plt.subplots(figsize=(20, 8), ncols=n_cols, nrows=nrows)
                x_vis = (x.float() + 1) / 2
                x_vis = x_vis[0].cpu().numpy().transpose(1, 2, 0)
                all_figs = [*feat_pca, x_vis]
                axes = axes.flatten()
                i = 0
                for f in all_figs:
                    axes[i].imshow(f)
                    i += 1

            fig.savefig(f"tmp/pca_vis.webp")

    # Return for compatibility with original function
    return decoded_tensor.mean().item() if use_optim else 0.0


def _get_pca_vis(feat_list: torch.Tensor | list[torch.Tensor], norm_type: str = "channel"):
    from src.stage1.utilities.losses.repa.feature_pca import feature_pca_sk, pca_list, feature_pca_torch
    from tqdm import tqdm

    if isinstance(feat_list, torch.Tensor):
        feat_list = [feat_list]

    feat_list_2 = []
    for feat in feat_list:
        if norm_type == "channel":
            feat = feat / feat.norm(dim=1, keepdim=True)
            feat = (feat - feat.mean(dim=1, keepdim=True)) / feat.std(dim=1, keepdim=True)
        else:
            hw = feat.shape[-2:]
            feat = feat.reshape(feat.shape[0], feat.shape[1], -1)
            feat = feat - feat.mean(-1, keepdim=True)
            feat = feat / feat.std(-1, keepdim=True)
            feat = feat.reshape(-1, feat.shape[1], hw[0], hw[1])
        feat_list_2.append(feat)

    feat_pca = []
    for i in tqdm(range(len(feat_list_2)), desc="PCA visualization", leave=False):
        f = feat_list_2[i]
        if f.shape[1] <= 3:
            feat_pca.append(f.float().detach().cpu().numpy()[0].transpose(1, 2, 0))
            continue
        # Use torch PCA
        f_pca = feature_pca_torch(f.float(), pca_k=3, norm_pca=False)
        feat_pca.append(f_pca)

    # Normalize
    for i in range(len(feat_pca)):
        f_pca = feat_pca[i]
        f_pca = (f_pca - f_pca.min()) / (f_pca.max() - f_pca.min())
        f_pca = (f_pca * 255.0).to(torch.uint8).detach().cpu().numpy()[0].transpose(1, 2, 0)
        feat_pca[i] = f_pca

    return feat_pca


if __name__ == "__main__":
    """
    MODEL_COMPILED=0 LOVELY_TENSORS=1 python -m src.stage1.cosmos.cosmos_hybrid
    """
    with logger.catch():
        test_model_forward_backward(
            real_data="RS5M", compute_mean_std=False, max_iters=1, save_pca_vis=True, pca_type="proj"
        )
        # test_forward_pca()

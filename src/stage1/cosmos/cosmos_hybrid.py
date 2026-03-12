"""
Hyperspectral Transformer Tokenizer with hybrid CNN / Transformer architecture.
RMSNorm + SwiGLU + EVA attention with Rope.
"""

from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

import accelerate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from einops import rearrange
from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf
from timm.layers import create_conv2d, create_norm_act_layer
from torch import Tensor
from typing_extensions import Annotated

from src.utilities.config_utils import (
    dataclass_to_dict_config,
    function_config_to_basic_types,
    function_config_to_easy_dict,
    set_defaults,
    to_easydict_recursive,
    to_object_recursive,
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

logger = logger.bind(_name_="CosmosHybrid")


class CosmosHybridTokenizer(ContinuousImageTokenizer):
    _no_split_modules = ["EvaBlock", "ResnetBlock", "AttnBlock"]
    _vf_on_z_or_module = "z"  # must be z if using this model
    # latents cached
    z: Tensor | list[Tensor] | None = None
    # repa/phi-s projections
    sem_z: Tensor | list[Tensor] | None = None
    supported_cached_hiddens: list[str] = ["z", "sem_z"]
    cache_layers: dict[str, int | list[int]] = {
        "low_level": [0, 1, 2, -1],  # -1 means middle layer
        "semantic": [2, 5, 8, 11],  # 12 layers of encoder
    }
    low_lvl_repa_proj_chans: list[int] = []
    _semantic_feature_dim: int = 1152
    sem_repa_proj_is_multi_layer_cached: bool = False
    sem_repa_proj_is_multi_layer_cached_by_teacher: dict[str, bool] = {}
    quantize_encode_infer: str | None = None

    def __init__(
        self,
        cnn_cfg: ContinuousTokenizerConfig | DictConfig,
        trans_enc_cfg: NaFlexVitCfg | DictConfig,
        trans_dec_cfg: NaFlexVitCfg | DictConfig | None = None,
        cnn_enc_cfg: EncoderDecoderConfig | DictConfig | None = None,
        cnn_dec_cfg: EncoderDecoderConfig | DictConfig | None = None,
        distillation_cfg: DictConfig | None = None,
        hybrid_tokenizer_cfg: DictConfig | None = None,
    ):
        self.z = None
        self.sem_z = None
        self.supported_cached_hiddens = list(type(self).supported_cached_hiddens)
        self.cache_layers = type(self).cache_layers.copy()
        self.low_lvl_repa_proj_chans = []
        self.sem_repa_proj_is_multi_layer_cached = False
        self.sem_repa_proj_is_multi_layer_cached_by_teacher = {}
        self.quantize_encode_infer = None
        assert self.quantize_encode_infer in ("h_fp8", "h_nf4", "", "h_bf16", None)

        self.cnn_cfg = cnn_cfg
        self.trans_enc_cfg = trans_enc_cfg
        self.trans_dec_cfg = trans_dec_cfg
        self.distillation_cfg = distillation_cfg
        self.grad_checkpointing = self.cnn_cfg.model.act_checkpoint
        self.latent_channels = self.cnn_cfg.model.latent_channels

        ###### Distillation configs
        self.distillation_cfg = set_defaults(
            distillation_cfg or edict(),
            {
                # single teacher
                "dino_feature_dim": 1024,
                "semantic_feature_dim": 1152,
                "cache_layers": self.cache_layers,
                # multiple teachers
                "teacher_proj_dims": {},
                "phis_student_source": "semantic",
            },
        )

        # single teacher
        self._dino_feature_dim = self.distillation_cfg.dino_feature_dim
        self._semantic_feature_dim = self.distillation_cfg.semantic_feature_dim
        self.cache_layers = to_object_recursive(self.distillation_cfg.cache_layers)

        # multiple teachers
        self._teacher_proj_dims: dict[str, dict[str, int]] = {
            str(teacher_name): {
                "low_level_out_dim": int(dims_cfg.get("low_level_out_dim", self._dino_feature_dim)),
                "semantic_out_dim": int(dims_cfg.get("semantic_out_dim", self._semantic_feature_dim)),
            }
            for teacher_name, dims_cfg in dict(getattr(self.distillation_cfg, "teacher_proj_dims", {})).items()
        }
        self._teacher_names: list[str] = list(self._teacher_proj_dims.keys())
        self._use_multi_teacher_proj = len(self._teacher_names) > 0
        self.phis_student_source: Literal["semantic", "low_level"] = cast(
            Literal["semantic", "low_level"],
            getattr(self.distillation_cfg, "phis_student_source", "semantic"),
        )
        if self.phis_student_source not in ("semantic", "low_level"):
            raise ValueError(
                f"Unknown phis_student_source={self.phis_student_source}, expected one of ('semantic', 'low_level')."
            )

        ###### Deep supervision configs
        self.hybrid_tokenizer_cfg = set_defaults(
            hybrid_tokenizer_cfg or edict(),
            {
                "deep_supervision_type": None,
                "latent_bottleneck_type": "after_semantic",
                "latent_straight_through_skip": False,
                "decoder_use_ul_noisy_latent": False,
                "decoder_ul_lambda0": 5.0,  # unified latent paper uses 5.0
            },
        )
        # deep_supervision_type: [direct_out, sum_previous_out]
        self.deep_supervision_type = self.hybrid_tokenizer_cfg.deep_supervision_type
        self.latent_bottleneck_type = self.hybrid_tokenizer_cfg.latent_bottleneck_type
        self.decoder_use_ul_noisy_latent = bool(self.hybrid_tokenizer_cfg.decoder_use_ul_noisy_latent)
        self.decoder_ul_lambda0 = float(self.hybrid_tokenizer_cfg.decoder_ul_lambda0)
        self._is_deep_supervision = self.deep_supervision_type is not None
        self._build_deep_supervised_heads(cnn_cfg)

        build_enc_dec_kwargs = {}
        if cnn_enc_cfg is not None or cnn_dec_cfg is not None:
            # If the encoder or decoder config is provided, use it
            build_enc_dec_kwargs = edict(enc_cnn_model_cfg=cnn_enc_cfg, dec_cnn_model_cfg=cnn_dec_cfg)

        super().__init__(self.cnn_cfg, build_enc_dec_kwargs=build_enc_dec_kwargs)  # type: ignore[arg-type]

        ##### Transformer models
        self._build_transformers(cnn_cfg, trans_enc_cfg, trans_dec_cfg)
        self._build_straight_through_skip(self.hybrid_tokenizer_cfg, cnn_cfg)
        self._validate_ul_noisy_decoder_init_constraints()

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

    def _build_encoder_decoder(
        self,
        cfg,
        model_cfg,
        *,
        enc_cnn_model_cfg=None,
        dec_cnn_model_cfg=None,
    ):
        self._is_diffbands = isinstance(model_cfg.in_channels, (tuple, list))

        enc_kwargs = dec_kwargs = model_cfg
        if enc_cnn_model_cfg is not None:
            enc_kwargs = enc_cnn_model_cfg
            logger.info("[Build CNN Encoder]: override cnn encoder config.")
        if dec_cnn_model_cfg is not None:
            dec_kwargs = dec_cnn_model_cfg
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
        self.low_lvl_repa_proj_chans = []
        if not isinstance(cache_index, list):
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

        self.semantic_enc_transformer = enc_model_cls(self.trans_enc_cfg)  # type: ignore[arg-type]
        self.semantic_enc_transformer.set_grad_checkpointing(self.grad_checkpointing)
        logger.info(f"Init semantic transformer encoder.")

        self.semantic_transformer_dec = None
        self.st_skip_sem_decoder = False
        if trans_dec_cfg is not None:
            self.semantic_transformer_dec = dec_model_cls(trans_dec_cfg)
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
                "maybe better for reconstruction, <yellow>Experimenting</yellow>"
            )

    def _validate_ul_noisy_decoder_init_constraints(self) -> None:
        if not self.decoder_use_ul_noisy_latent:
            return
        if self.st_skip_sem_decoder:
            raise ValueError(
                "UL noisy decoder does not support `latent_straight_through_skip=True`; "
                "the skip branch would leak the clean latent to the decoder."
            )
        if self.dual_latent_branch:
            raise ValueError(
                "UL noisy decoder does not support dual/mixed latents; "
                "the decoder condition is no longer a single compact latent aligned with the prior."
            )

    def _sample_ul_noisy_latent(self, latent_clean: Tensor) -> Tensor:
        alpha0 = torch.sqrt(torch.sigmoid(torch.tensor(self.decoder_ul_lambda0, device=latent_clean.device)))
        sigma0 = torch.sqrt(torch.sigmoid(torch.tensor(-self.decoder_ul_lambda0, device=latent_clean.device)))
        alpha0 = alpha0.to(dtype=latent_clean.dtype)
        sigma0 = sigma0.to(dtype=latent_clean.dtype)
        return alpha0 * latent_clean + sigma0 * torch.randn_like(latent_clean)

    def _prepare_single_decoder_latent_inputs(self, latent: Tensor) -> dict[str, Tensor | bool]:
        decoder_latent = self._post_quantize_latent_for_transmission(latent)
        return {
            "to_dec": decoder_latent,
            "to_dec_clean": decoder_latent,
            "to_dec_is_dual_latent": False,
        }

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

    def load_pretrained(self, uni_tokenizer_path: str | None, _reinit_quant_convs=False, **kwargs):
        """Init the model from the pretrained only CNN weights."""
        if uni_tokenizer_path in (None, ""):
            logger.warning(f"No pretrained weights found at {uni_tokenizer_path}, skip loading ckpt.")
            return

        weights = accelerate.utils.load_state_dict(uni_tokenizer_path)
        missing_ks, unexp_ks = load_weights_with_shape_check(self, weights, load_strategy="search")
        if len(missing_ks) > 0 or len(unexp_ks) > 0:
            logger.warning(f"Missing keys: {missing_ks}, Unexpected keys: {unexp_ks}")

        # Do not use it.
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
        logger.success(f"Load pretrained ckpt at {uni_tokenizer_path}")

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

        # Patch tokens
        h_flat = rearrange(h, "b c h w -> (b h w) c")
        h_proj_patch = self.lejepa_projector_latent(h_flat)

        return {"cls_tokens": h_proj, "patch_tokens": h_proj_patch}

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

        return {"cls_tokens": terms.lejepa_proj, "patch_tokens": terms.lejepa_proj_patch}

    def encode_dino_cls(self, x: torch.Tensor) -> Tensor:
        """return semantic encoder's normed CLS token (B, D), for DINO cls token loss"""
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

        tokens = self.semantic_enc_transformer.forward_features(h)
        tokens = cast(torch.Tensor, tokens)
        if getattr(self.semantic_enc_transformer, "num_prefix_tokens", 0) <= 0:
            raise ValueError("semantic_enc_transformer has no cls token, set `class_token: true` in config")
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
        quant_branch_out = self._encode_latent_branches(z_semantic, use_quantizer=use_quantizer)

        # Do cache here
        self.z = cache_low_lvl  # [b, c, h, w]
        self.sem_z = cache_semantic
        enc_out.update(
            low_lvl_z=cache_low_lvl,
            sem_z=cache_semantic,
            latent_before_quantizer=z_semantic,
            latent=quant_branch_out.latent,
            q_loss=quant_branch_out.q_loss,
            q_loss_breakdown=quant_branch_out.q_loss_breakdown,
            latent_is_dual=quant_branch_out.latent_is_dual,
        )
        for key in ("latent_mean", "latent_logvar", "latent_mode"):
            if hasattr(quant_branch_out, key):
                enc_out[key] = getattr(quant_branch_out, key)

        if quant_branch_out.latent_is_dual:
            enc_out.update(
                latent_quant=quant_branch_out.latent_quant,
                latent_cont=quant_branch_out.latent_cont,
                quant_keep_mask=quant_branch_out.quant_keep_mask,
                to_dec=quant_branch_out.latent,
                to_dec_is_dual_latent=True,
                to_dec_latent_quant=quant_branch_out.latent_quant,
                to_dec_latent_cont=quant_branch_out.latent_cont,
                to_dec_quant_keep_mask=quant_branch_out.quant_keep_mask,
            )
        else:
            enc_out.update(self._prepare_single_decoder_latent_inputs(quant_branch_out.latent))

        # for transmission only in eval
        if bool(enc_out.get("to_dec_is_dual_latent", False)):
            enc_out.to_dec = self._post_quantize_latent_for_transmission(enc_out.to_dec)

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
        quant_branch_out = self._encode_latent_branches(z_low_lvl, use_quantizer=use_quantizer)
        enc_out.update(
            q_loss=quant_branch_out.q_loss,
            q_loss_breakdown=quant_branch_out.q_loss_breakdown,
            latent=quant_branch_out.latent,
            latent_is_dual=quant_branch_out.latent_is_dual,
        )
        for key in ("latent_mean", "latent_logvar", "latent_mode"):
            if hasattr(quant_branch_out, key):
                enc_out[key] = getattr(quant_branch_out, key)

        if quant_branch_out.latent_is_dual:
            enc_out.update(
                latent_quant=quant_branch_out.latent_quant,
                latent_cont=quant_branch_out.latent_cont,
                quant_keep_mask=quant_branch_out.quant_keep_mask,
            )
            h = self._dual_latents_to_z_mixture(
                latent_quant=quant_branch_out.latent_quant,
                latent_cont=quant_branch_out.latent_cont,
                quant_keep_mask=quant_branch_out.quant_keep_mask,
            )
            h = self._post_quantize_latent_for_transmission(h)
            # In dual-branch mode, use mixed latent as the public latent for downstream losses/logging.
            enc_out["latent"] = h
            enc_out["latent_mixed"] = h
            if self.st_skip_sem_decoder:
                h_skipped = h.clone()
                enc_out.h_skipped = h_skipped  # z_dim
        else:
            h = quant_branch_out.latent
            h = self._post_quantize_latent_for_transmission(h)
            if self.decoder_use_ul_noisy_latent:
                latent_clean = h
                latent_noisy = self._sample_ul_noisy_latent(latent_clean)
                enc_out.latent_clean = latent_clean
                enc_out.latent_noisy = latent_noisy
                h = latent_noisy
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

    def encode_with_distill_features(
        self,
        x: Tensor,
        use_quantizer: bool | None = None,
        get_intermediate_features: bool = True,
    ) -> dict[str, Any]:
        """Encode and return both raw cached features and projected distillation features.
        Alias for `get_repa_feature` and arg `force_to`=True.
        """
        assert self._vf_on_z_or_module == "z", f"Only support z cache, got {self._vf_on_z_or_module=}"
        assert hasattr(self, "_repa_proj"), "Only repa projection is supported for distillation features."

        enc_out = self.encode(
            x=x,
            use_quantizer=use_quantizer,
            get_intermediate_features=get_intermediate_features,
        )
        repa_feats = self.get_repa_feature(force_to=True)
        assert repa_feats is not None
        low_lvl_z_proj, sem_z_proj = repa_feats
        proj_dict = self.get_repa_feature_dict(force_to=True)

        return {
            "raw": {
                "low_lvl_z": enc_out.get("low_lvl_z"),
                "sem_z": enc_out.get("sem_z"),
            },
            "proj": {
                "low_lvl_z_proj": low_lvl_z_proj,
                "sem_z_proj": sem_z_proj,
            },
            "proj_dict": proj_dict,
            "encode_out": enc_out,
        }

    def decode(
        self,
        inp: dict,
        inp_shape: Annotated[torch.Size | int | tuple, "bs,c,h,w or bs,c or c"],
        clamp=False,
    ) -> dict:
        """
        Decode the latent into the corresponding channels image.
        Output the latent, to_dec, q_loss, q_loss_breakdown, recon, deep_supervision_outputs (optional)"""
        out_d = edict(**inp)  # quantization-related, to_dec, latent
        h = inp["to_dec"]
        latent_clean = inp.get("to_dec_clean")
        to_dec_is_z = bool(inp.get("to_dec_is_z", False))

        if bool(inp.get("to_dec_is_dual_latent", False)):
            quant_keep_mask = inp.get("to_dec_quant_keep_mask")
            assert torch.is_tensor(quant_keep_mask), "to_dec_quant_keep_mask should be a tensor"
            h = self._dual_latents_to_z_mixture(
                latent_quant=inp.get("to_dec_latent_quant"),
                latent_cont=inp.get("to_dec_latent_cont"),
                quant_keep_mask=quant_keep_mask,
            )
            to_dec_is_z = True
            out_d.to_dec = h
            # Keep `latent` aligned with the mixed decoder input in dual-branch mode.
            out_d.latent = h
            out_d.latent_mixed = h

        if latent_clean is not None and self.decoder_use_ul_noisy_latent:
            assert torch.is_tensor(latent_clean), "`to_dec_clean` should be a tensor when UL noisy decoder is enabled."
            latent_noisy = self._sample_ul_noisy_latent(latent_clean)
            out_d.latent_clean = latent_clean
            out_d.latent_noisy = latent_noisy
            h = latent_noisy

        # Decoder
        chan = inp_shape[1] if isinstance(inp_shape, (torch.Size, tuple)) else inp_shape

        # Apply post-quant conv - semantic transformer decoder - CNN decoder
        if not to_dec_is_z:
            if self.st_skip_sem_decoder:
                if self.latent_bottleneck_type == "after_semantic":
                    h = self.decoder.quant_conv["post_quant_conv"](h)
                    h_skipped = h.clone()  # z_dim
                else:
                    h_skipped = inp.get("h_skipped")  # z_dim
                    assert h_skipped is not None
            else:
                h = self.decoder.quant_conv(h)
        elif self.st_skip_sem_decoder:
            h_skipped = inp.get("h_skipped")
            if h_skipped is None:
                h_skipped = h.clone()

        # Apply semantic transformer decoder if it exists
        if self.semantic_transformer_dec is not None:
            h = self.semantic_transformer_dec(h)

        # Cat the skipped latent
        if self.st_skip_sem_decoder:
            if h_skipped.shape[-2:] != h.shape[-2:]:
                h_skipped = F.interpolate(h_skipped, size=h.shape[-2:], mode="nearest")
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
    def create_model(
        cls,
        cnn_cfg,
        trans_enc_cfg,
        trans_dec_cfg=None,
        distillation_cfg=None,
        hybrid_tokenizer_cfg=None,
        # overrides for the main cnn_cfg, if None, use the default values
        # in cnn_cfg
        cnn_enc_cfg=None,
        cnn_dec_cfg=None,
    ):
        def _merge_cfg(base_cfg_cls: type[Any], override_cfg: dict | DictConfig | None):
            cfg_base = dataclass_to_dict_config(base_cfg_cls)
            if override_cfg is None:
                merged_cfg = OmegaConf.create(cfg_base)
            else:
                merged_cfg = OmegaConf.merge(cfg_base, override_cfg)
            return to_easydict_recursive(OmegaConf.to_container(merged_cfg, resolve=True))

        cont_tok_cfg = _merge_cfg(ContinuousTokenizerConfig, cnn_cfg)
        trans_enc_cfg = _merge_cfg(NaFlexVitCfg, trans_enc_cfg)
        trans_dec_cfg = _merge_cfg(NaFlexVitCfg, trans_dec_cfg) if trans_dec_cfg is not None else None
        cnn_enc_cfg = _merge_cfg(EncoderDecoderConfig, cnn_enc_cfg) if cnn_enc_cfg is not None else None
        cnn_dec_cfg = _merge_cfg(EncoderDecoderConfig, cnn_dec_cfg) if cnn_dec_cfg is not None else None

        return cls(
            cont_tok_cfg,
            trans_enc_cfg,
            trans_dec_cfg,
            cnn_enc_cfg=cnn_enc_cfg,
            cnn_dec_cfg=cnn_dec_cfg,
            distillation_cfg=distillation_cfg,
            hybrid_tokenizer_cfg=hybrid_tokenizer_cfg,
        )

    def _should_project_repa_features(self, force_to: bool = False) -> bool:
        if force_to:
            return True
        if not self._use_repa_loss:
            return False
        if not self.training:
            # Skip distillation projections in eval mode by default.
            return False
        return True

    @staticmethod
    def _project_feature_branch(
        *,
        proj: nn.Module,
        cached: Tensor | list[Tensor],
        is_multi: bool,
        proj_name: str,
    ) -> Tensor | list[Tensor]:
        if is_multi:
            assert isinstance(proj, nn.ModuleList), f"{proj_name} must be ModuleList when is_multi=True"
            assert isinstance(cached, list), f"{proj_name} cached feature must be list when is_multi=True"
            assert len(proj) == len(cached), (
                f"{proj_name} projection count mismatch: got {len(proj)} projections for {len(cached)} cached features."
            )
            return [proj_i(feat_i) for proj_i, feat_i in zip(proj, cached)]

        assert torch.is_tensor(cached), f"{proj_name} cached feature must be Tensor when is_multi=False"
        return proj(cached)

    def _project_low_level_features(
        self,
        low_lvl_proj: nn.Module,
        cached_low_lvl: Tensor | list[Tensor],
    ) -> Tensor | list[Tensor]:
        return self._project_feature_branch(
            proj=low_lvl_proj,
            cached=cached_low_lvl,
            is_multi=self.low_lvl_repa_proj_is_multi,
            proj_name="low_lvl_repa_proj",
        )

    def _project_semantic_features(
        self,
        sem_proj: nn.Module,
        cached_sem: Tensor | list[Tensor],
        *,
        is_multi_cached: bool,
    ) -> Tensor | list[Tensor]:
        return self._project_feature_branch(
            proj=sem_proj,
            cached=cached_sem,
            is_multi=is_multi_cached,
            proj_name="sem_repa_proj",
        )

    def _get_single_teacher_repa_feature(
        self,
        force_to: bool = False,
    ) -> tuple[Tensor | list[Tensor], Tensor | list[Tensor]] | None:
        if not self._should_project_repa_features(force_to):
            return None

        z, sem_z = self.z, self.sem_z
        assert self._vf_on_z_or_module == "z"
        assert z is not None and sem_z is not None, "No cached z or sem_z"
        assert isinstance(self._repa_proj, nn.ModuleDict)

        low_lvl_z_proj = self._project_low_level_features(self._repa_proj["low_lvl_repa_proj"], z)
        sem_z_proj = self._project_semantic_features(
            self._repa_proj["sem_repa_proj"],
            sem_z,
            is_multi_cached=self.sem_repa_proj_is_multi_layer_cached,
        )
        return low_lvl_z_proj, sem_z_proj

    def _get_multi_teacher_repa_feature(
        self,
        force_to: bool = False,
    ) -> dict[str, tuple[Tensor | list[Tensor], Tensor | list[Tensor]]] | None:
        if not self._should_project_repa_features(force_to):
            return None
        if not self._use_multi_teacher_proj:
            raise RuntimeError("_get_multi_teacher_repa_feature is only valid in multi-teacher projection mode.")

        z, sem_z = self.z, self.sem_z
        assert self._vf_on_z_or_module == "z"
        assert z is not None and sem_z is not None, "No cached z or sem_z"
        assert isinstance(self._repa_proj, nn.ModuleDict)

        out: dict[str, tuple[Tensor | list[Tensor], Tensor | list[Tensor]]] = {}
        for teacher_name in self._teacher_names:
            teacher_proj = self._repa_proj[teacher_name]
            assert isinstance(teacher_proj, nn.ModuleDict), (
                f"Expected {teacher_name} projection as ModuleDict, got {type(teacher_proj)}"
            )

            low_lvl_z_proj = self._project_low_level_features(teacher_proj["low_lvl_repa_proj"], z)
            sem_z_proj = self._project_semantic_features(
                teacher_proj["sem_repa_proj"],
                sem_z,
                is_multi_cached=self.sem_repa_proj_is_multi_layer_cached_by_teacher.get(
                    teacher_name, self.sem_repa_proj_is_multi_layer_cached
                ),
            )
            out[teacher_name] = (low_lvl_z_proj, sem_z_proj)
        return out

    @torch.autocast("cuda", dtype=torch.bfloat16)
    def get_repa_feature(
        self,
        force_to: bool = False,
    ) -> tuple[Tensor | list[Tensor], Tensor | list[Tensor]] | None:
        if not self._use_multi_teacher_proj:
            return self._get_single_teacher_repa_feature(force_to=force_to)

        multi_feats = self._get_multi_teacher_repa_feature(force_to=force_to)
        if multi_feats is None:
            return None

        primary_teacher = self._teacher_names[0]
        logger.warning(
            f"[Repa proj]: get_repa_feature() is using primary teacher '{primary_teacher}' in multi-teacher mode. "
            "Please switch to get_repa_feature_dict() for all teachers.",
            warn_once=True,
        )
        return multi_feats[primary_teacher]

    @torch.autocast("cuda", dtype=torch.bfloat16)
    def get_repa_feature_dict(self, force_to: bool = False) -> dict[str, Tensor | list[Tensor]] | None:
        if self.phis_student_source not in ("semantic", "low_level"):
            raise ValueError(
                f"Unknown phis_student_source={self.phis_student_source}, expected one of ('semantic', 'low_level')."
            )
        branch_index = 1 if self.phis_student_source == "semantic" else 0

        if not self._use_multi_teacher_proj:
            single_feats = self._get_single_teacher_repa_feature(force_to=force_to)
            if single_feats is None:
                return None
            teacher_name = self._teacher_names[0] if len(self._teacher_names) > 0 else "default"
            return {teacher_name: single_feats[branch_index]}

        multi_feats = self._get_multi_teacher_repa_feature(force_to=force_to)
        if multi_feats is None:
            return None
        return {teacher_name: feats[branch_index] for teacher_name, feats in multi_feats.items()}

    def _build_feature_align_mlp(self, proj_type: str = "norm_first_force_conv"):
        logger.info(f"[Repa proj]: build mlp type {proj_type}")

        # Set the low-level cache layers for the feature alignment
        self._set_low_level_proj_chans()

        assert self._vf_on_z_or_module == "z", f"Only support z for _vf_on_z_or_modulebut got {self._vf_on_z_or_module}"

        if not self._use_repa_loss:
            return

        # Low-level projection
        low_level_cache_layers = self.cache_layers["low_level"]
        self.low_lvl_repa_proj_is_multi = isinstance(low_level_cache_layers, list)
        if self.low_lvl_repa_proj_is_multi:
            assert isinstance(low_level_cache_layers, list)
            assert len(self.low_lvl_repa_proj_chans) == len(low_level_cache_layers), (
                f"Length of low_lvl_repa_proj_chans {len(self.low_lvl_repa_proj_chans)} "
                f"should match length of cached low level layers "
                f"{len(low_level_cache_layers)}"
            )

        sem_cache_layers = self.cache_layers["semantic"]
        self.sem_repa_proj_is_multi_layer_cached = isinstance(sem_cache_layers, list)
        self.sem_repa_proj_is_multi_layer_cached_by_teacher = {}

        def _build_low_level_proj(out_dim: int) -> nn.Module:
            if not self.low_lvl_repa_proj_is_multi:
                return build_mlp(self.cnn_cfg.model.z_channels, out_dim, out_dim, proj_type=proj_type)

            assert isinstance(low_level_cache_layers, list)
            low_lvl_z_proj = nn.ModuleList()
            for i in range(len(low_level_cache_layers)):
                low_lvl_z_proj.append(
                    build_mlp(
                        self.low_lvl_repa_proj_chans[i],
                        out_dim,
                        out_dim,
                        proj_type=proj_type,
                    )
                )
            return low_lvl_z_proj

        def _build_sem_proj(out_dim: int) -> nn.Module:
            if self.sem_repa_proj_is_multi_layer_cached:
                assert isinstance(sem_cache_layers, list)
                sem_z_proj = nn.ModuleList()
                for _ in range(len(sem_cache_layers)):
                    sem_z_proj.append(
                        build_mlp(
                            self.trans_enc_cfg.embed_dim,
                            out_dim,
                            out_dim,
                            proj_type=proj_type,
                        )
                    )
                return sem_z_proj

            return build_mlp(
                self.trans_enc_cfg.embed_dim,
                out_dim,
                out_dim,
                proj_type=proj_type,
            )

        if self._use_multi_teacher_proj:
            teacher_proj = nn.ModuleDict()
            for teacher_name in self._teacher_names:
                dims = self._teacher_proj_dims[teacher_name]
                teacher_proj[teacher_name] = nn.ModuleDict(
                    {
                        "low_lvl_repa_proj": _build_low_level_proj(dims["low_level_out_dim"]),
                        "sem_repa_proj": _build_sem_proj(dims["semantic_out_dim"]),
                    }
                )
                self.sem_repa_proj_is_multi_layer_cached_by_teacher[teacher_name] = (
                    self.sem_repa_proj_is_multi_layer_cached
                )
            self._repa_proj = teacher_proj
            logger.info(f"[Repa proj]: built multi-teacher projections for {self._teacher_names}.")
            return

        low_lvl_z_proj = _build_low_level_proj(self._dino_feature_dim)
        sem_z_proj = _build_sem_proj(self._semantic_feature_dim)
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

    def _post_quantize_latent_for_transmission(self, latent: Tensor):
        """
        Simulate the transmission NF4 or FP8 downcast and then upcast back to BF16.
        """
        from torchao.dtypes import to_nf4

        if self.training:
            return latent

        if self.quantize_encode_infer == "h_fp8":
            latent = (latent * 10).to(torch.float8_e4m3fn)
            latent = latent.to(torch.bfloat16) / 10
        elif self.quantize_encode_infer == "h_nf4":
            b, c, h, w = latent.shape
            latent = latent * 100
            latent = to_nf4(latent.view(b * c, h * w), block_size=16, scaler_block_size=64)
            latent = latent.to(torch.bfloat16).view(b, c, h, w) / 100
        elif self.quantize_encode_infer in ("", None, "h_bf16"):
            pass
        else:
            raise ValueError(f"Unknown quantize_encode_infer: {self.quantize_encode_infer}")

        return latent


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
    yml = f"""
    cnn_cfg:
      model:
        resolution: 512
        in_channels: {in_chans}
        out_channels: {in_chans}
        z_channels: {z_chans}
        latent_channels: {latent_chans}
        channels: 128
        channels_mult: [2, 4, 4]
        num_res_blocks: 2
        attn_resolutions: []
        dropout: 0.0
        spatial_compression: 8
        patch_size: 1
        block_name: res_block
        norm_type: gn
        norm_groups: 32
        adaptive_mode: interp
        downsample_kwargs:
            padconv_use_manually_pad: false
        upsample_kwargs:
            interp_type: nearest_interp
        per_layer_noise: false
      quantizer_type: null
      vf_on_z_or_module: z
      use_repa_loss: {str(use_repa_loss).lower()}
      dino_feature_dim: 1024
      decoder_type: default
      use_channel_drop: false
      channel_drop_config:
        drop_type: [16, 32, 48]
        max_channels: 64
        drop_prob: 0.0
      loading_type: hybrid_pretrained
      uni_path: {pretrained_path}

    trans_enc_cfg:
      embed_dim: {backbone_embed_dim}
      depth: {backbone_enc_n_layers}
      num_heads: 16
      mlp_ratio: 4.0
      qkv_bias: true
      patch_size: 2
      norm_layer: {norm_layer}
      pos_embed: learned
      pos_embed_grid_size: [32, 32]
      rope_type: axial
      img_size: 32
      in_chans: {z_chans}
      out_chans: {z_chans}
      unpatch_size: 1
      reg_tokens: 0
      compile_model: false

    trans_dec_cfg:
      embed_dim: {backbone_embed_dim}
      depth: {backbone_dec_n_layers}
      num_heads: 16
      mlp_ratio: 4.0
      qkv_bias: true
      patch_size: 1
      norm_layer: {norm_layer}
      pos_embed: learned
      pos_embed_grid_size: [32, 32]
      rope_type: axial
      img_size: 32
      in_chans: 512
      out_chans: 512
      unpatch_size: 2
      reg_tokens: 0
      compile_model: false

    distillation_cfg:
      dino_feature_dim: 1024
      semantic_feature_dim: 1024

    hybrid_tokenizer_cfg: null
    """
    return OmegaConf.create(yml)


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
    """
    yml = f"""
    cnn_cfg:
      model:
        resolution: 1024
        in_channels: {in_chans}
        out_channels: {in_chans}
        z_channels: {z_chans}
        latent_channels: {latent_chans}
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
      use_repa_loss: {str(use_repa_loss).lower()}
      dino_feature_dim: 1024
      loading_type: hybrid_pretrained
      uni_path: {pretrained_path}

    trans_enc_cfg:
      embed_dim: 1152
      depth: 24
      num_heads: 16
      mlp_ratio: 4.0
      qkv_bias: true
      patch_size: 2
      norm_layer: {norm_layer}
      pos_embed: learned
      pos_embed_grid_size: [32, 32]
      img_size: 32
      in_chans: {z_chans}
      out_chans: {z_chans}
      unpatch_size: 2
      reg_tokens: 4
      rope_type: axial
      attn_type: gated
      pretrained_type: [ijepa]

    trans_dec_cfg: null

    distillation_cfg:
      dino_feature_dim: 1024
      semantic_feature_dim: 1024
      cache_layers:
        low_level: [0, 1, 2, -1]
        semantic: [5, 11, 17, 23]

    hybrid_tokenizer_cfg:
      latent_bottleneck_type: before_semantic
      latent_straight_through_skip: true
    """
    return OmegaConf.create(yml)


def hyrbid_lejea_iejepa_f8_config():
    """
    Configuration for hybrid LeJEPA + IJEPA tokenizer model with mHC and value residual.
    """
    yml = f"""
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
        adaptive_mode: interp
        downsample_kwargs:
            padconv_use_manually_pad: false
        upsample_kwargs:
            interp_type: nearest_interp
      quantizer_type: null
      vf_on_z_or_module: z
      use_repa_loss: true
      dino_feature_dim: 1024
      loading_type: hybrid_pretrained
      uni_path: ""

    trans_enc_cfg:
      embed_dim: 1152
      depth: 24
      num_heads: 16
      mlp_ratio: 4.0
      qkv_bias: true
      patch_size: 2
      norm_layer: flarmsnorm
      pos_embed: learned
      pos_embed_grid_size: [32, 32]
      img_size: 32
      in_chans: 768
      out_chans: 768
      unpatch_size: 2
      reg_tokens: 4
      rope_type: axial
      attn_type: gated
      pretrained_type: [ijepa, lejepa]
      hc_streams: 4
      hc_implem: naive
      v_residual: true
      hc_other_kwargs:
        sinkhorn_iters: 10

    trans_dec_cfg:
      embed_dim: 1152
      depth: 6
      num_heads: 16
      mlp_ratio: 4.0
      qkv_bias: true
      patch_size: 1
      norm_layer: flarmsnorm
      pos_embed: learned
      pos_embed_grid_size: [32, 32]
      rope_type: axial
      img_size: 32
      in_chans: 768
      out_chans: 768
      unpatch_size: 1
      reg_tokens: 4
      compile_model: false

    distillation_cfg:
      dino_feature_dim: 1024
      semantic_feature_dim: 1152
      cache_layers:
        low_level: [0, 1, 2, -1]
        semantic: [5, 11, 17, 23]

    hybrid_tokenizer_cfg:
      latent_bottleneck_type: after_semantic
      latent_straight_through_skip: true
    """
    return OmegaConf.create(yml)


def hybrid_dual_latent_distill_f8_config(pretrained_path: str = ""):
    cfg = hybrid_ijepa_f8_config(pretrained_path=pretrained_path)
    cfg.cnn_cfg.dual_latent_branch = True
    cfg.cnn_cfg.continuous_latent_channels = 32
    cfg.cnn_cfg.model.latent_channels = 64  # is discreate channel
    cfg.cnn_cfg.quantizer_type = "bsq"
    return cfg


def hybrid_asymmetrical_enc_dec_f16_config(pretrained_path="") -> DictConfig:
    """
    Reuse the base dual-latent config and add an asymmetric CNN decoder config.
    Encoder keeps x8 compression while decoder is set to x16 upsampling.
    """
    cfg = hybrid_dual_latent_distill_f8_config(pretrained_path=pretrained_path)
    cfg.cnn_cfg.model.norm_type = "trmsnorm2d"
    cfg.trans_enc_cfg.patch_size = 2
    cfg.trans_enc_cfg.unpatch_size = 1
    cfg.cnn_dec_cfg = OmegaConf.create(
        """
        resolution: 1024
        in_channels: 512
        out_channels: 512
        z_channels: 768
        latent_channels: 64
        channels: 128
        channels_mult: [1, 2, 4, 4]
        num_res_blocks: 2
        attn_resolutions: []
        dropout: 0.0
        spatial_compression: 16
        patch_size: 1
        block_name: res_block
        norm_type: trmsnorm2d
        act_type: silu
        adaptive_mode: interp
        downsample_kwargs:
            padconv_use_manually_pad: false
        upsample_kwargs:
            interp_type: nearest_interp
        per_layer_noise: false
        """
    )
    return cfg


###### Tests ##########


def test_model_forward_backward(
    model_type: str = "hybrid_enc_dec_transformer",
    real_data: str | None = None,
    use_optim: bool = False,
    device: str | torch.device = "cuda",
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
    quantize_encode_infer: str | None = None,
    per_sample_info: bool = False,
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
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    from torchmetrics.image.sam import SpectralAngleMapper
    from torchmetrics.regression import MeanSquaredError
    from torchvision.utils import make_grid  # type: ignore[unresolved-import]
    from tqdm import tqdm
    from torchao.quantization import quantize_, Int4WeightOnlyConfig, Int8WeightOnlyConfig, Float8WeightOnlyConfig
    from torchao.dtypes import to_nf4

    from src.data.hyperspectral_loader import get_fast_test_hyperspectral_data
    from src.data.litdata_hyperloader import get_fast_test_hyper_litdata_load

    device = torch.device("cuda")

    cfg = hybrid_ijepa_f8_config(
        # pretrained_path="runs/stage1_cosmos_hybrid/2025-12-09_18-38-25_hybrid_cosmos_f16c64_jepa_pretrained/ema/tokenizer/model.safetensors",
        pretrained_path="runs/stage1_cosmos_hybrid/2025-12-21_23-52-12_hybrid_cosmos_f16c64_ijepa_pretrained_sem_no_lejepa/ema/tokenizer/model.safetensors",
        latent_chans=32,
        use_repa_loss=True,
    )
    # cfg = hybrid_distillation_f16_config(
    #     pretrained_path="runs/stage1_cosmos_hybrid/2025-11-15_22-11-24_hybrid_cosmos_f16c64/ema/tokenizer/model.safetensors",
    #     use_repa_loss=True,
    # )
    # cfg = hyrbid_lejea_iejepa_f8_config()
    # cfg = hybrid_dual_latent_distill_f8_config()
    # cfg = hybrid_asymmetrical_enc_dec_f16_config(
    #     "runs/stage1_cosmos_hybrid_dual_latent/2026-03-05_02-20-55_dual_bsq64_cont32_distill_ijepa_litdata_one_loader/ema/tokenizer/model.safetensors"
    # )
    # breakpoint()
    model: CosmosHybridTokenizer = CosmosHybridTokenizer.create_model(
        cfg.cnn_cfg,
        cfg.trans_enc_cfg,
        trans_dec_cfg=cfg.trans_dec_cfg,
        distillation_cfg=cfg.distillation_cfg,
        hybrid_tokenizer_cfg=cfg.hybrid_tokenizer_cfg,
        cnn_enc_cfg=cfg.get("cnn_enc_cfg"),
        cnn_dec_cfg=cfg.get("cnn_dec_cfg"),
    )
    model = model.to(device, torch.bfloat16)  # Move model to device (CUDA or CPU)
    if quantize_encode_infer in ("h_nf4", "h_fp8"):
        model.quantize_encode_infer = quantize_encode_infer

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
            # dl = get_fast_test_hyperspectral_data(batch_size=1, data_type=real_data)
            dl = get_fast_test_hyper_litdata_load(real_data, batch_size=1, stream_ds_kwargs={"shuffle": True})[1]  # type: ignore
            iterations = dl
    else:
        x = torch.randn(*fake_img_shape).to(device, dtype)
        iterations = [x]

    if not is_itered and upscale != 1:
        x = torch.nn.functional.interpolate(x, scale_factor=upscale, align_corners=True, mode="bicubic")

    if use_optim:
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    def _metric_to_float(metric_value: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | float) -> float:
        if isinstance(metric_value, tuple):
            metric_value = metric_value[0]
        if isinstance(metric_value, torch.Tensor):
            return float(metric_value.detach().mean().item())
        return float(metric_value)

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    mse_metric = MeanSquaredError().to(device)
    sam_metric: SpectralAngleMapper | None = None
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    mse_values: list[float] = []
    sam_values: list[float] = []
    if compute_mean_std:
        mean_lst = []
        std_lst = []

    if quantize_encode_infer == "enc_nf4":
        quantize_(
            model.encoder,
            Int4WeightOnlyConfig(group_size=64),
            filter_fn=lambda mod, name: isinstance(mod, (torch.nn.Linear, torch.nn.Conv2d)),
        )
    elif quantize_encode_infer == "enc_int4":
        quantize_(
            model.encoder,
            Int4WeightOnlyConfig(group_size=64),
            filter_fn=lambda mod, name: isinstance(mod, (torch.nn.Linear, torch.nn.Conv2d)),
        )
    elif quantize_encode_infer == "enc_int8":
        quantize_(
            model.encoder,
            Int8WeightOnlyConfig(),
            filter_fn=lambda mod, name: isinstance(mod, (torch.nn.Linear, torch.nn.Conv2d)),
        )
    elif quantize_encode_infer == "enc_fp8":
        quantize_(
            model.encoder,
            Float8WeightOnlyConfig(),
            filter_fn=lambda mod, name: isinstance(mod, (torch.nn.Linear, torch.nn.Conv2d)),
        )

    for index, x in (tbar := tqdm(enumerate(iterations))):
        if index < 10:
            continue

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

            with torch.autocast(str(device), dtype=torch.bfloat16):
                # Decode with proper input shape
                decoded = model.decode(encoded, x.shape)
                decoded_tensor = decoded.recon

                if per_sample_info:
                    if quantize_encode_infer is None:
                        logger.info(
                            f"min: {encoded_tensor.min()}, max: {encoded_tensor.max()}, mean: {encoded_tensor.mean()}, std: {encoded_tensor.std()} | "
                            f"latent dtype: {encoded_tensor.dtype}",
                            tqdm=True,
                        )
                    else:
                        _dtype = (
                            encoded_tensor.quantized_data.dtype
                            if hasattr(encoded_tensor, "quantized_data")
                            else encoded_tensor.dtype
                        )
                        logger.info(f"latent dtype: {_dtype}", tqdm=True)
        else:
            # Training mode
            encoded = model.encode(x)
            encoded_tensor, q_loss, loss_breakdown = encoded.to_dec, encoded.q_loss, encoded.q_loss_breakdown
            decoded_tensor = model.decode(encoded, x.shape).recon

        decoded_tensor.clamp_(-1, 1)
        if decoded_tensor.shape[-2:] != x.shape[-2:]:
            raise RuntimeError(
                "Decoded tensor spatial size mismatch: "
                f"decoded={decoded_tensor.shape[-2:]}, target={x.shape[-2:]}. "
                "Please fix decoder upsampling ratio in config."
            )

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
                selected_chans = [rgb_chans[i] if rgb_chans[i] < y_grid.shape[0] else i for i in range(available_chans)]
                y_grid = y_grid[selected_chans].permute(1, 2, 0).detach().cpu().numpy()  # [h, w, 3]
                y_grid = (y_grid + 1) / 2
                y_grid = (y_grid * 255.0).astype(np.uint8)
                Image.fromarray(y_grid).save(path)
                logger.info(f"save reconstruction image {name_suffix}")

            plot_img(
                decoded_tensor,
                Path(save_img_dir) / f"recon_{real_data or 'fake'}_{index}.png",
                "reconstruction",
            )
            plot_img(x, Path(save_img_dir) / f"gt_{real_data or 'fake'}_{index}.png", "ground truth")

        # reconstruction metrics
        if real_data:
            target = ((x + 1) / 2).clamp(0, 1)
            pred = ((decoded_tensor + 1) / 2).clamp(0, 1)
            psnr_val = _metric_to_float(psnr_metric(pred, target))
            ssim_val = _metric_to_float(ssim_metric(pred, target))
            mse_val = _metric_to_float(mse_metric(pred, target))
            psnr_values.append(psnr_val)
            ssim_values.append(ssim_val)
            mse_values.append(mse_val)

            metric_msg = f"PSNR: {psnr_val:.4f}, SSIM: {ssim_val:.4f}, MSE: {mse_val:.6f}"
            if x.shape[1] > 3:
                if sam_metric is None:
                    sam_metric = SpectralAngleMapper().to(device)
                sam_val = _metric_to_float(sam_metric(pred, target))
                sam_values.append(sam_val)
                metric_msg += f", SAM: {sam_val:.4f}"

            metric_msg += f" - shape: {x.shape}"
            if per_sample_info:
                logger.info(metric_msg, tqdm=True)
            tbar.set_description(metric_msg)

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

    val_stats = None
    if real_data:
        val_stats = {
            "PSNR": float(np.mean(psnr_values)),
            "SSIM": float(np.mean(ssim_values)),
            "MSE": float(np.mean(mse_values)),
        }
        avg_msg = (
            f"Average PSNR: {val_stats['PSNR']:.4f}, "
            f"Average SSIM: {val_stats['SSIM']:.4f}, "
            f"Average MSE: {val_stats['MSE']:.6f}"
        )
        if sam_values:
            val_stats["SAM"] = float(np.mean(sam_values))
            avg_msg += f", Average SAM: {val_stats['SAM']:.4f}"
        logger.info(avg_msg)

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
    return decoded_tensor.mean().item() if use_optim else 0.0, val_stats


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
    from itertools import product

    data_names = ["SAM270k", "fmow_MS", "hyspectnet11k"]
    quantize_infers = [None, "h_fp8", "h_nf4"]

    all_stats = {}

    with logger.catch():
        for data_name, quantize_infer in product(data_names, quantize_infers):
            print(f"working on {data_name=}, {quantize_infer=}")
            print("-" * 60)
            stats = test_model_forward_backward(
                real_data=data_name,
                compute_mean_std=False,
                max_iters=100,
                save_pca_vis=False,
                pca_type="proj",
                quantize_encode_infer=quantize_infer,
            )
            print("-" * 60)
            # test_forward_pca()
            print(stats)
            all_stats[f"{data_name}/{quantize_infer}"] = stats

    print(all_stats)
    """
    Transmission performance:

    BigEarth S1 SAR
    BFloat16    Average PSNR: 31.24, SSIM: 0.8921, MSE: 0.000512
    Latent_FP8  Average PSNR: 31.12, SSIM: 0.8901, MSE: 0.000516
    Latent_NF4  Average PSNR: 29.57, SSIM: 0.8832, MSE: 0.000621

    SAM270k RGB
    BFloat16    Average PSNR: 34.9851, Average SSIM: 0.9126, Average MSE: 0.000960
    Latent_FP8  Average PSNR: 34.2619, Average SSIM: 0.9134, Average MSE: 0.000958
    Latent_NF4  Average PSNR: 31.6517, Average SSIM: 0.8809, Average MSE: 0.001257

    fmow_MS 12bands
    BFloat16     Average PSNR: 38.5254, Average SSIM: 0.9627, Average MSE: 0.000190, Average SAM: 0.0322
    Latent_FP8   Average PSNR: 38.1510, Average SSIM: 0.9591, Average MSE: 0.000203, Average SAM: 0.032
    Latent_NF4   Average PSNR: 35.5014, Average SSIM: 0.9332, Average MSE: 0.000331, Average SAM: 0.0409

    hyspectnet11k 202bands
    BFloat16     Average PSNR: 35.3989, Average SSIM: 0.9104, Average MSE: 0.000342, Average SAM: 0.0775
    Latent_FP8   Average PSNR: 35.1448, Average SSIM: 0.9058, Average MSE: 0.000363, Average SAM: 0.0777
    Latent_NF4   Average PSNR: 33.9596, Average SSIM: 0.8868, Average MSE: 0.000449, Average SAM: 0.0831
    """

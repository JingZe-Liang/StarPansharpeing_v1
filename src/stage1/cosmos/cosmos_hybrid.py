"""
Hyperspectral Transformer Tokenizer with hybrid CNN / Transformer architecture.
RMSNorm + SwiGLU + EVA attention with Rope.
"""

from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, List, NamedTuple, Optional, Self, Tuple, Union

import accelerate
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from einops.layers.torch import Rearrange
from loguru import logger
from timm.layers import create_conv2d, create_norm_act_layer
from torch import Tensor
from typing_extensions import Annotated

from src.utilities.config_utils import (
    dataclass_from_dict,
    function_config_to_basic_types,
    function_config_to_easy_dict,
    set_defaults,
)

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
        distillation_kwargs: dict[str, Any] = {},
        hybrid_tokenizer_kwargs: dict[str, Any] = {},
    ):
        self.cnn_cfg = cnn_cfg
        self.trans_enc_cfg = trans_enc_cfg
        self.trans_dec_cfg = trans_dec_cfg
        self.distillation_kwargs = distillation_kwargs
        self.grad_checkpointing = self.cnn_cfg.model.act_checkpoint

        ###### Distillation configs
        distillation_kwargs = set_defaults(
            distillation_kwargs,
            {
                "dino_feature_dim": 1024,
                "semantic_feature_dim": 1152,
                "cache_layers": self.cache_layers,
            },
        )
        self._dino_feature_dim = distillation_kwargs.dino_feature_dim
        self._semantic_feature_dim = distillation_kwargs.semantic_feature_dim
        self.cache_layers = distillation_kwargs.cache_layers

        ###### Deep supervision configs
        hybrid_tokenizer_kwargs = set_defaults(
            hybrid_tokenizer_kwargs,
            {"deep_supervision_type": None},
        )
        # deep_supervision_type: [direct_out, sum_previous_out]
        self.deep_supervision_type = hybrid_tokenizer_kwargs.deep_supervision_type
        self._is_deep_supervision = self.deep_supervision_type is not None
        self._build_deep_supervised_heads(cnn_cfg)

        build_enc_dec_kwargs = {}
        if cnn_enc_cfg is not None or cnn_dec_cfg is not None:
            # If the encoder or decoder config is provided, use it
            build_enc_dec_kwargs = edict(
                {
                    "enc_cnn_model_cfg": cnn_enc_cfg,
                    "dec_cnn_model_cfg": cnn_dec_cfg,
                }
            )
        super().__init__(self.cnn_cfg, build_enc_dec_kwargs=build_enc_dec_kwargs)

        ##### Transformer models
        self._build_transformers(cnn_cfg, trans_enc_cfg, trans_dec_cfg)

    def _build_encoder_decoder(
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
        if trans_enc_cfg.pretrained_type in ("ijepa", "lejepa"):
            enc_model_cls = IJEPANaFlexViT
        if trans_dec_cfg is not None and trans_dec_cfg.pretrained_type in (
            "ijepa",
            "lejepa",
        ):
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

            # Straight through skip for low-level features
            if trans_dec_cfg.latent_straight_through_skip:
                self.st_skip_sem_decoder = True
                # Change the postquant conv
                orig_post_quant_conv = self.decoder.quant_conv
                cat_dim = cnn_cfg.model.z_channels + cnn_cfg.model.in_channels
                skip_through_cat_conv = nn.Sequential(
                    create_norm_act_layer("layernorm2d", cat_dim, "gelu"),
                    # post conv to z channels and feed into the cnn decoder
                    # fuse the skipped latent (=z_channels) and the semantic decoder output
                    create_conv2d(
                        cat_dim,
                        cnn_cfg.model.z_channels,
                        kernel_size=3,
                    ),
                )
                # replace the post quant conv
                self.decoder.quant_conv = nn.ModuleDict(
                    {
                        "post_quant_conv": orig_post_quant_conv,
                        "st_cat_conv": skip_through_cat_conv,
                    }
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
            to_out = nn.Module()
            out_chan = basic_chan * chan_mults[i]
            # Create head according to supervision type
            to_out_main = nn.Sequential(
                create_norm_act_layer("layernorm2d", out_chan, "silu"),
                AdaptiveOutputConvLayer(out_chan, head_out_chan, mode="interp"),
            )
            to_out.add_module("main", to_out_main)
            if deep_sup_type == "sum_previous_out":
                # Add previous summed feature
                if i not in (0, n_res - 1):
                    prev_out_chan = basic_chan * chan_mults[i - 1]
                    # Upsample previous feature and add to current
                    prev_out_proj = nn.Sequential(
                        create_norm_act_layer("layernorm2d", prev_out_chan, "silu"),
                        create_conv2d(prev_out_chan, out_chan, kernel_size=3),
                        nn.Upsample(scale_factor=2, mode="nearest"),
                    )
                    to_out.add_module("prev_out_proj", prev_out_proj)

            self.deep_supervised_heads.append(to_out)
        logger.info(f"Build deep supervised heads with type: {deep_sup_type}")

    def load_pretrained(self, uni_tokenizer_path: str, directly_load=True, **kwargs):
        """Init the model from the pretrained only CNN weights."""
        from src.utilities.network_utils.network_loading import (
            load_weights_with_shape_check,
        )

        weights = accelerate.utils.load_state_dict(uni_tokenizer_path)

        # Directly load all weights if specified
        if directly_load:
            missing_ks, unexp_ks = load_weights_with_shape_check(self, weights, load_strategy="search")
            if len(missing_ks) > 0 or len(unexp_ks) > 0:
                logger.warning(f"Directly Loading Missing keys: {missing_ks}, Unexpected keys: {unexp_ks}")
            logger.info(f"Finished directly loading pretrained weights.")
            return

        # Filter out the cnn and transformer's keys
        cnn_enc_ws, cnn_dec_ws, trans_ws = {}, {}, {}
        for k, v in weights.items():
            if k.startswith("encoder"):
                cnn_enc_ws[k[8:]] = v
            if k.startswith("decoder"):
                cnn_dec_ws[k[8:]] = v
            elif k.startswith("semantic_enc_transformer"):
                trans_ws[k] = v

        # Load CNN Cosmos tokenizer weights
        logger.info(f"Loading CNN Cosmos tokenizer weights ...")
        missing_k, unexp_k = load_weights_with_shape_check(self.encoder, cnn_enc_ws)
        if len(missing_k) > 0 or len(unexp_k) > 0:
            logger.warning(f"CNN Encoder Missing keys: {missing_k}, Unexpected keys: {unexp_k}")

        missing_k, unexp_k = load_weights_with_shape_check(self.decoder, cnn_dec_ws)
        if len(missing_k) > 0 or len(unexp_k) > 0:
            logger.warning(f"CNN Decoder Missing keys: {missing_k}, Unexpected keys: {unexp_k}")

        logger.info(f"Loading semantic Transformer weights ...")
        # Load Transformer weights
        if len(trans_ws) != 0:
            missing_k, unexp_k = self.semantic_enc_transformer.load_state_dict(trans_ws, strict=False)
            if len(missing_k) > 0 or len(unexp_k) > 0:
                logger.warning(f"Transformer Missing keys: {missing_k}, Unexpected keys: {unexp_k}")

        logger.info(f"Finished loading pretrained weights.")

    @staticmethod
    def _interp_max_size_features(feats: list[Tensor]):
        """"""
        max_size = torch.max(torch.tensor([f.shape[-2:] for f in feats]), dim=0).values
        max_size = tuple(max_size.tolist())
        interp_feats = []
        for f in feats:
            if f.shape[-2:] != max_size:
                f = F.interpolate(f, size=max_size, mode="bilinear", align_corners=False)
            interp_feats.append(f)
        return interp_feats

    def encode_ijepa(self, x, use_quantizer=None, jepa_masks=None, **_ignored_kwargs):
        """
        x -> CNN (with full image) -> transformer (with unmasked patches) -> 1d tokens.
        no quantization, no quant_conv, no other projections.
        """
        # Low-level encoder
        z_low_lvl = self.encoder.encoder(x)

        # Semantic encoder
        z_semantic, _ = self.semantic_enc_transformer._forward_pretrained_backbone(  # type: ignore
            z_low_lvl, jepa_masks=jepa_masks
        )

        # no quant_conv here
        return z_semantic

    def encode_lejepa(self, x, **_ignored_kwargs):
        """
        LeJEPA encoding with augmentated global and local views.
        """

        # Low-level encoder
        z_low_lvl = self.encoder.encoder(x)

        # Semantic encoder
        # z_low_lvl -> transformer backbone -> projector -> z_proj
        _, others = self.semantic_enc_transformer._forward_pretrained_backbone(  # type: ignore
            z_low_lvl, jepa_masks=None
        )

        return others.lejepa_proj

    def encode(
        self,
        x,
        use_quantizer=None,
        get_intermediate_features=False,
    ):
        """Encode the image into latent.
        Output the latent tensor or latent, quantizer loss and loss breakdowns
        if has a quantizer.

        x: (B, C, H, W) - Image
        use_quantizer:  Whether to use the quantizer. If None, use the default setting.
        get_intermediate_features: Whether to return the intermediate features for distillation.
        """
        ############ Low-level encoder
        cache_low_lvl = None
        last_hidden_cached = self.cache_layers["low_level"] == -1
        if self.training or get_intermediate_features:
            low_lvl_out = self.encoder.encoder(x, ret_interm_feats=not last_hidden_cached)
            if last_hidden_cached:
                z_low_lvl = low_lvl_out
                cache_low_lvl = z_low_lvl
            elif isinstance(low_lvl_out, (list, tuple)):
                z_low_lvl, cache_low_lvl = low_lvl_out
        else:
            z_low_lvl = self.encoder.encoder(x)

        ########### Forward semantic transformer encoder
        cache_semantic = None
        if not self.training and not get_intermediate_features:  # is eval
            z_semantic = self.semantic_enc_transformer(z_low_lvl)
        elif self.cache_layers["semantic"] == -1 and get_intermediate_features:
            z_semantic = self.semantic_enc_transformer(z_low_lvl)
            cache_semantic = z_semantic
        else:
            # forward the intermediate features
            _model_kwargs = dict(
                x=z_low_lvl,
                indices=self.cache_layers["semantic"],
                return_prefix_tokens=False,
                norm=True,
                output_fmt="NCHW",
                output_dict=False,
            )
            # forward intermediates does not support jepa masks
            z_semantic, cache_semantic = self.semantic_enc_transformer.forward_intermediates(**_model_kwargs)

            # Add head forward
            z_semantic = z_semantic[:, self.semantic_enc_transformer.num_prefix_tokens :]
            z_semantic = self.semantic_enc_transformer._forward_after_backbone(
                z_semantic,
                hw=self.semantic_enc_transformer._get_output_shape(z_low_lvl),
            )
            # Stack the intermidates features: [n_cache_layers, b, c, h, w]
            # cache_semantic = torch.stack(cache_semantic, dim=0)

        ######## Apply CNN encoder - semantic transformer encoder - quant conv
        # Semantic token -> quant conv -> latent
        h = self.encoder.quant_conv(z_semantic)

        # Quantization
        maybe_q_ret = self.apply_quantizer(h, z_semantic, use_quantizer, cache_type=None)  # Disable cache z or h

        # Do cache here
        self.z = cache_low_lvl  # [b, c, h, w]
        self.sem_z = cache_semantic

        if isinstance(maybe_q_ret, tuple):
            h, q_loss, loss_breakdown = maybe_q_ret
            # NOTE: if quantizer is used, the aug z is not applied
            return h, q_loss, loss_breakdown

        ##### z augmentions (noise adding or channel dropping).
        h = self.latent_aug(maybe_q_ret)

        return h

    def decode(
        self,
        h: Union[torch.Tensor, tuple],
        inp_shape: Annotated[Union[torch.Size, int, tuple], "bs,c,h,w or bs,c or c"],
        clamp=False,
    ):
        """
        Decode the latent into the corresponding channels image.
        Output the reconstructed image or recon, quantizer loss and loss breakdowns.
        """
        q_loss = loss_breakdown = None
        if self.quantizer_type is not None and isinstance(h, (tuple, list)):
            h, q_loss, loss_breakdown = h
        else:
            assert torch.is_tensor(h), "z should be the (quantized) latent"

        # Decoder
        chan = inp_shape[1] if isinstance(inp_shape, (torch.Size, tuple)) else inp_shape

        # Apply Post-quant conv - semantic transformer decoder - CNN decoder
        if self.st_skip_sem_decoder:
            h = self.decoder.quant_conv["post_quant_conv"](h)
            h_skipped = h.clone()
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
        dec: Tensor | dict[str, Tensor | list[Tensor]]
        if self._is_deep_supervision:
            dec, interms = dec
            deep_sup_outputs = self._forward_deep_supervised_heads(interms)
            dec = edict({"recon": dec, "deep_supervision_outputs": deep_sup_outputs})
            if clamp:
                dec.recon = dec.recon.clamp(-1, 1)
        elif clamp:
            dec = dec.clamp(-1, 1)

        if self.quantizer_type is not None:
            return dec, q_loss, loss_breakdown
        else:
            return dec

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
        distillation_kwargs: dict | None = None,
        hybrid_tokenizer_kwargs: dict | None = None,
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
            distillation_kwargs=edict(distillation_kwargs) or edict(),
            hybrid_tokenizer_kwargs=edict(hybrid_tokenizer_kwargs) or edict(),
        )

    @torch.autocast("cuda", dtype=torch.bfloat16)
    def get_repa_feature(
        self,
    ) -> tuple[Tensor | list[Tensor], Tensor | list[Tensor]] | None:
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

    def _build_feature_align_mlp(self):
        # Set the low-level cache layers for the feature alignment
        self._set_low_level_proj_chans()

        assert self._vf_on_z_or_module == "z", f"Only support z for _vf_on_z_or_modulebut got {self._vf_on_z_or_module}"

        if self._use_repa_loss:
            # Low-level projection
            self.low_lvl_repa_proj_is_multi = isinstance(self.cache_layers["low_level"], (tuple, list))
            if not self.low_lvl_repa_proj_is_multi:
                low_lvl_z_proj = build_mlp(
                    self.cnn_cfg.model.z_channels,
                    self._dino_feature_dim,
                    self._dino_feature_dim,
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
                        )
                    )
            else:
                sem_z_proj = build_mlp(
                    self.trans_enc_cfg.embed_dim,
                    self._semantic_feature_dim,
                    self._semantic_feature_dim,
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


def test_model_forward_backward():
    """Test the forward and backward pass of the CosmosHybridTokenizer model.

    This function creates a model, tests forward pass in both eval and train modes,
    and verifies that gradients are computed correctly during backward pass.
    """
    # Import fvcore for parameter counting
    from fvcore.nn import parameter_count_table

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == "cpu":
        logger.warning("CUDA is not available, running on CPU. This may be slow.")

    # Create the model with correct configuration structure
    cnn_cfg = {
        "model": {
            "resolution": 256,
            "in_channels": 300,
            "z_channels": 512,  # Connect encoder and decoder
            "latent_channels": 16,
            "channels": 128,  # Base channels
            "channels_mult": [2, 4, 4],  # Channel multiplier
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "dropout": 0.0,
            "spatial_compression": 8,  # Default spatial compression
            "patch_size": 1,  # Default patch size
            "block_name": "res_block",  # Default block type
            "norm_type": "gn",  # Default normalization
            "norm_groups": 32,  # Default number of groups
        },
        "quantizer_type": None,  # No quantizer for basic test
        "vf_on_z_or_module": "z",  # Visual feature on z
        "use_repa_loss": True,
        "dino_feature_dim": 1024,
        "cache_type": "h",
    }

    trans_enc_cfg = {
        "embed_dim": 1024,
        "depth": 12,  # vit intern 300m -> embed_dim=1024, depth=24, num_heads=16
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "patch_size": 2,
        "norm_layer": "rmsnorm",  # Use Flash RMSNorm for compatibility
        "pos_embed": "learned",  # Use learned position embeddings
        "pos_embed_grid_size": (32, 32),  # Grid size for position embedding
        # Additional required fields from merged configuration
        "img_size": 32,  # Expected by NaFlexVitCfg
        "in_chans": 512,  # Should match z_channels from transformer
        "out_chans": 512,  # Output channels
        "unpatch_size": 1,  # 2->1
        "reg_tokens": 4,
    }

    trans_dec_cfg = {
        "embed_dim": 1024,
        "depth": 12,
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "patch_size": 1,
        "norm_layer": "rmsnorm",  # Use Flash RMSNorm for compatibility
        "pos_embed": "learned",  # Use learned position embeddings
        "pos_embed_grid_size": (32, 32),  # Grid size for position embedding
        # Additional required fields from merged configuration
        "img_size": 32,  # Expected by NaFlexVitCfg
        "in_chans": 512,  # Latent channels after post quant conv
        "out_chans": 512,  # Output channels: z channels
        "unpatch_size": 2,  # 1->2
        "reg_tokens": 0,
    }

    model: CosmosHybridTokenizer = CosmosHybridTokenizer.create_model(cnn_cfg, trans_enc_cfg, trans_dec_cfg)
    model = model.to(device)  # Move model to device (CUDA or CPU)
    model.eval()
    logger.info("Model created successfully!")
    model.set_grad_checkpointing(enabled=True)

    # Print parameter table using fvcore
    logger.info("Model parameter table:")
    logger.info(parameter_count_table(model))

    # Create dummy input data
    batch_size = 4
    x = torch.randn(batch_size, 10, 128, 128).to(device)  # Move input to device
    logger.info(f"Input shape: {x.shape}")

    # Forward pass in eval mode
    with torch.no_grad():
        model.train()
        encoded = model.encode(x)
        if isinstance(encoded, tuple):
            encoded_tensor, q_loss, loss_breakdown = encoded
            logger.info(f"Encoded shape: {encoded_tensor.shape}")
            logger.info(f"Quantizer loss: {q_loss}")
        else:
            encoded_tensor = encoded
            logger.info(f"Encoded shape: {encoded_tensor.shape}")
            q_loss = None

        # Print cached tensors
        logger.info("Cached tensors:")
        logger.info(f"  z (CNN encoder output): {model.z}")
        logger.info(f"  sem_z (semantic transformer output): {model.sem_z}")

        # Projection z and sem_z
        low_lvl_z_proj, sem_z_proj = model.get_repa_feature()
        logger.info(f"Projected low-level z shape: {low_lvl_z_proj}")
        logger.info(f"Projected semantic z shape: {sem_z_proj}")

        # Decode with proper input shape
        decoded = model.decode(encoded_tensor, x.shape)
        if isinstance(decoded, tuple):
            decoded_tensor, dec_q_loss, dec_loss_breakdown = decoded
            logger.info(f"Decoded shape: {decoded_tensor.shape}")
        else:
            decoded_tensor = decoded
            logger.info(f"Decoded shape: {decoded_tensor.shape}")

    # Test training mode
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Forward pass with gradients
    encoded = model.encode(x)
    if isinstance(encoded, tuple):
        encoded_tensor, q_loss, loss_breakdown = encoded
    else:
        encoded_tensor = encoded
        q_loss = None

    decoded = model.decode(encoded_tensor, x.shape)
    if isinstance(decoded, tuple):
        decoded_tensor, dec_q_loss, dec_loss_breakdown = decoded
    else:
        decoded_tensor = decoded

    # Backward and check the gradients
    target = torch.randn_like(decoded_tensor)
    loss = torch.nn.functional.mse_loss(decoded_tensor, target)

    # Add quantizer loss if present
    if q_loss is not None:
        loss += q_loss

    loss.backward()
    optim.step()
    # optim.zero_grad()
    logger.info(f"Backward pass completed successfully!")
    logger.info(f"Total loss: {loss.item()}")

    # Check if gradients are computed
    gradient_count = 0
    n_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                has_gradients = True
                gradient_count += 1
            else:
                logger.warning(f"Parameter {name} has no gradient!")
            n_params += param.numel()

    logger.info(f"Total parameters with gradients: {gradient_count}")
    logger.info(f"Total parameters: {n_params / 1e6:.2f}M")

    return loss.item()


def test_forward_pca():
    # cfg
    from omegaconf import OmegaConf

    from src.stage1.utilities.losses.repa.feature_pca import (
        feature_pca_cuml,
        feature_pca_sk,
        feature_pca_torch,
    )

    cfg = OmegaConf.load("scripts/configs/tokenizer_gan/tokenizer/comos_hybrid_f16c32.yaml")
    logger.info(str(cfg))

    logger.log("NOTE", f"Rope type: {cfg.trans_enc_cfg.rope_type}")

    model = CosmosHybridTokenizer.create_model(
        cnn_cfg=cfg.cnn_cfg,
        trans_enc_cfg=cfg.trans_enc_cfg,
        trans_dec_cfg=cfg.trans_dec_cfg,
        distillation_kwargs=cfg.distillation_kwargs,
    ).cuda()

    from src.data.hyperspectral_loader import get_fast_test_hyperspectral_data

    dl = get_fast_test_hyperspectral_data("MMSeg")

    # import accelerate

    # accelerate.utils.load_checkpoint_in_model(
    #     model,
    #     "runs/stage1_cosmos_hybrid/2025-11-02_02-02-53_hybrid_cosmos_f16c64/ema/tokenizer/model.safetensors",
    # )

    sample = next(iter(dl))
    img = sample["img"].to("cuda", torch.bfloat16)

    # model.eval()
    model.train()
    model = model.to(torch.bfloat16)

    with torch.no_grad():
        output = model(img)

    print(output)

    # z, sem_z = model.z, model.sem_z
    # logger.info(f"z shape: {z.shape}, sem_z shape: {sem_z.shape}")
    # z_pca = [feature_pca_sk(z_i.cpu().float(), 3) for z_i in z]
    # sem_z_pca = [feature_pca_sk(sem_z_i.cpu().float(), 3) for sem_z_i in sem_z]

    # ... plot


if __name__ == "__main__":
    """
    MODEL_COMPILED=0 LOVELY_TENSORS=1 python -m src.stage1.cosmos.cosmos_hybrid
    """
    import lovely_tensors as lt

    lt.monkey_patch()
    with logger.catch():
        # test_model_forward_backward()
        test_forward_pca()

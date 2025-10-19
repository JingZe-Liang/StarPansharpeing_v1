"""
Hyperspectral Transformer Tokenizer with hybrid CNN / Transformer architecture.
RMSNorm + SwiGLU + EVA attention with Rope.
"""

from typing import Any, List, NamedTuple, Optional, Tuple, Union

import accelerate
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from loguru import logger
from torch import Tensor
from typing_extensions import Annotated

from src.utilities.config_utils import (
    dataclass_from_dict,
    function_config_to_basic_types,
)

from .cosmos_tokenizer import (
    ContinuousImageTokenizer,
    ContinuousTokenizerConfig,
    EncoderDecoderConfig,
)
from .modules import blocks as cosmos_blocks
from .modules.layers2d import Decoder, Encoder
from .modules.naflex import NaFlexVitCfg, Transformer
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
        distillation_kwargs: dict[str, Any] = {},
    ):
        self.cnn_cfg = cnn_cfg
        self.trans_enc_cfg = trans_enc_cfg
        self.trans_dec_cfg = trans_dec_cfg
        self.distillation_kwargs = distillation_kwargs
        self._dino_feature_dim = distillation_kwargs.get("dino_feature_dim", 1024)
        self._semantic_feature_dim = distillation_kwargs.get("semantic_feature_dim", 1152)  # fmt: skip
        self.cache_layers = distillation_kwargs.get("cache_layers", self.cache_layers)

        super().__init__(self.cnn_cfg)
        self.grad_checkpointing = self.cnn_cfg.model.act_checkpoint
        self._build_transformers(cnn_cfg, trans_enc_cfg, trans_dec_cfg)

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
        if -1 == cache_index[-1]:
            self.low_lvl_repa_proj_chans.append(out_chan)
            logger.info(
                f"Low-level repa projection channels: {self.low_lvl_repa_proj_chans}"
            )

    def _build_transformers(self, cnn_cfg, trans_enc_cfg, trans_dec_cfg=None):
        # cnn_cfg is already set as self.cnn_cfg in __init__
        # Transformer Encoder and Decoder
        self.semantic_enc_transformer = Transformer(self.trans_enc_cfg)
        self.semantic_enc_transformer.set_grad_checkpointing(self.grad_checkpointing)
        logger.info(f"Init semantic transformer encoder.")

        self.semantic_transformer_dec = None
        if trans_dec_cfg is not None:
            self.semantic_transformer_dec = Transformer(self.trans_dec_cfg)
            self.semantic_transformer_dec.set_grad_checkpointing(
                self.grad_checkpointing
            )
            logger.info(f"Init semantic transformer decoder.")

    def load_pretrained(self, uni_tokenizer_path: str, directly_load=True, **kwargs):
        """Init the model from the pretrained only CNN weights."""
        from src.utilities.network_utils.network_loading import (
            load_weights_with_shape_check,
        )

        weights = accelerate.utils.load_state_dict(uni_tokenizer_path)

        # Directly load all weights if specified
        if directly_load:
            missing_ks, unexp_ks = load_weights_with_shape_check(
                self, weights, load_strategy="search"
            )
            if len(missing_ks) > 0 or len(unexp_ks) > 0:
                logger.warning(
                    f"Directly Loading Missing keys: {missing_ks}, Unexpected keys: {unexp_ks}"
                )
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
            logger.warning(
                f"CNN Encoder Missing keys: {missing_k}, Unexpected keys: {unexp_k}"
            )

        missing_k, unexp_k = load_weights_with_shape_check(self.decoder, cnn_dec_ws)
        if len(missing_k) > 0 or len(unexp_k) > 0:
            logger.warning(
                f"CNN Decoder Missing keys: {missing_k}, Unexpected keys: {unexp_k}"
            )

        logger.info(f"Loading semantic Transformer weights ...")
        # Load Transformer weights
        if len(trans_ws) != 0:
            missing_k, unexp_k = self.semantic_enc_transformer.load_state_dict(
                trans_ws, strict=False
            )
            if len(missing_k) > 0 or len(unexp_k) > 0:
                logger.warning(
                    f"Transformer Missing keys: {missing_k}, Unexpected keys: {unexp_k}"
                )

        logger.info(f"Finished loading pretrained weights.")

    @staticmethod
    def _interp_max_size_features(feats: list[Tensor]):
        """"""
        max_size = torch.max(torch.tensor([f.shape[-2:] for f in feats]), dim=0).values
        max_size = tuple(max_size.tolist())
        interp_feats = []
        for f in feats:
            if f.shape[-2:] != max_size:
                f = F.interpolate(
                    f, size=max_size, mode="bilinear", align_corners=False
                )
            interp_feats.append(f)
        return interp_feats

    def encode(self, x, use_quantizer=None):
        """Encode the image into latent.
        Output the latent tensor or latent, quantizer loss and loss breakdowns
        if has a quantizer.
        """
        # Low-level encoder
        cache_low_lvl = None
        last_hidden_cached = self.cache_layers["low_level"] == -1
        if self.training:
            low_lvl_out = self.encoder.encoder(
                x, ret_interm_feats=not last_hidden_cached
            )
            if last_hidden_cached:
                z_low_lvl = low_lvl_out
                cache_low_lvl = z_low_lvl
            elif isinstance(low_lvl_out, (list, tuple)):
                z_low_lvl, cache_low_lvl = low_lvl_out
        else:
            z_low_lvl = self.encoder.encoder(x)

        # Forward semantic transformer encoder
        cache_semantic = None
        if not self.training:  # is eval
            z_semantic = self.semantic_enc_transformer(z_low_lvl)
        elif self.cache_layers["semantic"] == -1:
            z_semantic = self.semantic_enc_transformer(z_low_lvl)
            cache_semantic = z_semantic
        else:
            # forward the intermidiates
            z_semantic, cache_semantic = (
                self.semantic_enc_transformer.forward_intermediates(
                    z_low_lvl,
                    indices=self.cache_layers["semantic"],
                    return_prefix_tokens=False,
                    norm=True,
                    output_fmt="NCHW",
                    output_dict=False,
                )
            )
            # Add head forward
            z_semantic = z_semantic[:, self.trans_enc_cfg.reg_tokens :]
            z_semantic = self.semantic_enc_transformer._forward_after_backbone(
                z_semantic,
                hw=self.semantic_enc_transformer._get_output_shape(z_low_lvl),
            )
            # Stack the intermidates features: [n_cache_layers, b, c, h, w]
            # cache_semantic = torch.stack(cache_semantic, dim=0)

        # Apply CNN encoder - semantic transformer encoder - quant conv
        h = self.encoder.quant_conv(z_semantic)

        # Quantization
        maybe_q_ret = self.apply_quantizer(
            h, z_semantic, use_quantizer, cache_type=None
        )  # Disable cache z or h

        # Do cache here
        self.z = cache_low_lvl  # [b, c, h, w]
        self.sem_z = cache_semantic

        if isinstance(maybe_q_ret, tuple):
            h, q_loss, loss_breakdown = maybe_q_ret
            # NOTE: if quantizer is used, the aug z is not applied
            return h, q_loss, loss_breakdown

        # z augmentions
        h = self.latent_aug(maybe_q_ret)
        return h

    def decode(
        self,
        h: Union[torch.Tensor, tuple],
        inp_shape: Annotated[Union[torch.Size, int, tuple], "bs,c,h,w or bs,c or c"],
        clamp=False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Union[dict, Tensor]]]:
        """Decode the latent into the corresponding channels image.
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
        h = self.decoder.quant_conv(h)

        # Apply semantic transformer decoder if it exists
        if self.semantic_transformer_dec is not None:
            h = self.semantic_transformer_dec(h)

        # Decode using the CNN decoder
        dec = self.decoder.decoder(h, chan)  # [b, c, h, w]

        if clamp:
            dec = dec.clamp(-1, 1)
        if self.quantizer_type is not None:
            return dec, q_loss, loss_breakdown
        else:
            return dec

    @classmethod
    @function_config_to_basic_types
    def create_model(
        cls,
        cnn_cfg,
        trans_enc_cfg,
        trans_dec_cfg=None,
        distillation_kwargs: dict | None = None,
    ):
        cnn_cfg = dataclass_from_dict(ContinuousTokenizerConfig, cnn_cfg)
        trans_enc_cfg = dataclass_from_dict(NaFlexVitCfg, trans_enc_cfg)
        if trans_dec_cfg is not None:
            trans_dec_cfg = dataclass_from_dict(NaFlexVitCfg, trans_dec_cfg)

        return cls(
            cnn_cfg,
            trans_enc_cfg,
            trans_dec_cfg,
            distillation_kwargs=distillation_kwargs or {},
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

        if self.low_lvl_repa_proj_is_multi:
            assert isinstance(z, list)
            low_lvl_z_proj = [
                self._repa_proj["low_lvl_repa_proj"][i](z[i])
                for i in range(len(self.low_lvl_repa_proj_chans))
            ]
        else:
            assert torch.is_tensor(z)
            low_lvl_z_proj = self._repa_proj["low_lvl_repa_proj"](z)

        if self.sem_repa_proj_is_multi:
            assert isinstance(sem_z, list)
            sem_z_proj = [
                self._repa_proj["sem_repa_proj"][i](sem_z[i]) for i in range(len(sem_z))
            ]
        else:
            assert torch.is_tensor(sem_z)
            sem_z_proj = self._repa_proj["sem_repa_proj"](sem_z)

        # return tuple of Tensors or list of Tensors
        return low_lvl_z_proj, sem_z_proj

    def _build_feature_align_mlp(self):
        # Set the low-level cache layers for the feature alignment
        self._set_low_level_proj_chans()

        assert self._vf_on_z_or_module == "z", (
            f"Only support z for _vf_on_z_or_modulebut got {self._vf_on_z_or_module}"
        )

        if self._use_repa_loss:
            # Low-level projection
            self.low_lvl_repa_proj_is_multi = isinstance(
                self.cache_layers["low_level"], (tuple, list)
            )
            if not self.low_lvl_repa_proj_is_multi:
                low_lvl_z_proj = build_mlp(
                    self.cnn_cfg.model.z_channels,
                    self._dino_feature_dim,
                    self._dino_feature_dim,
                )
            else:
                assert len(self.low_lvl_repa_proj_chans) == len(
                    self.cache_layers["low_level"]
                ), (
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

    model: CosmosHybridTokenizer = CosmosHybridTokenizer.create_model(
        cnn_cfg, trans_enc_cfg, trans_dec_cfg
    )
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


if __name__ == "__main__":
    """
    LOVELY_TENSORS=1 python -m src.stage1.cosmos.cosmos_hybrid
    """
    import lovely_tensors as lt

    lt.monkey_patch()
    with logger.catch():
        test_model_forward_backward()

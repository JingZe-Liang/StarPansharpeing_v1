import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from loguru import logger
from omegaconf import OmegaConf
from timm.layers import get_act_layer, get_norm_act_layer, get_norm_layer
from timm.models._manipulate import named_apply
from torch import Tensor
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.module import _IncompatibleKeys
from torch.utils.checkpoint import checkpoint

from src.stage1.cosmos.cosmos_hybrid import CosmosHybridTokenizer
from src.utilities.config_utils import (
    dataclass_from_dict,
    function_config_to_basic_types,
)

from ...layers.implicit_block import FusionBlock
from .adapter import DINOv3EncoderAdapter
from .tokenizer_backbone_adapted import HybridTokenizerEncoderAdapter

TOKENIZER_INTERACTION_INDEXES = {
    "hybrid_tokenizer_b16": [3, 6, 8, 11],
}

# *==============================================================
# * Configurations
# *==============================================================


def _create_default_cfg():
    cfg_yaml = """
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
                    padconv_use_manually_pad: False
                upsample_kwargs:
                    interp_type: nearest_interp
            quantizer_type: null
            vf_on_z_or_module: z
            use_repa_loss: False
            dino_feature_dim: 1024
        trans_enc_cfg:
            embed_dim: 1152
            depth: 24
            num_heads: 16
            mlp_ratio: 4.0
            qkv_bias: True
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
            latent_straight_through_skip: True

    tokenizer_feature:
        pretrained_path: null
        features_per_stage: [512, 512, 512, 512]
        model_name: hybrid_tokenizer_b16
        pretrained_size: 512
        in_channels: 3
        conv_inplane: 64
        layer_in_channels: [512, 512, 1152, 1152]
        drop_path_rate: 0.3
        with_cffn: True
        cffn_ratio: 0.25
        deform_num_heads: 8
        deform_ratio: 0.5
        add_vit_feature: True
        use_extra_extractor: True
        with_cp: True
        select_in_all_layers: True
        interaction_indexes: [1, 2, 4, 5]

    adapter:
        adapter_type: default
        latent_width: 32
        n_conv_per_stage: 1
        depth_per_stage: 1
        norm: layernorm2d
        act: gelu
        drop: 0.0
        act_first: False
        conv_bias: False
        block_types: ["nat", "nat", "mbconv", "mbconv"]

    decoder:
        embed_dim: 256
        num_fusion_blocks: 2
        fusion_block_type: mbconv
        fusion_block_kwargs: {}

    tokenizer_pretrained_path: null
    input_channels: 155
    num_classes: 2
    deep_supervision: False
    n_stages: 4
    use_latent: True
    ensure_rgb_type: null
    _debug: False
    """
    return OmegaConf.create(cfg_yaml)


# *==============================================================
# * Tokenizer Unet
# *==============================================================


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

        # Ensure we have 4 stages to match adapter output
        if cfg.n_stages != 4:
            logger.error(f"Warning: Adapter outputs 4 scales, but n_stages={n_stages}. Adjusting to 4.")
            raise ValueError("n_stages must be 4")

        # Create tokenzier encoder
        self.encoder = self._create_tok_encoder()

        # Create implicit progressive decoder
        decoder_cfg = self.cfg.decoder if hasattr(self.cfg, "decoder") else {}
        # Get conv_inplane from tokenizer_feature config which usually corresponds to latent dim or similar
        # Actually checking the schema: tokenizer.cnn_cfg.model.latent_channels seems relevant for 'cond'
        # In encoder(), it returns `final_h` which comes from `enc_out.latent`.
        # Let's find the dimension of `final_h`.
        # In HybridTokenizer, latent is projected to `tokenizer.cnn_cfg.model.latent_channels`? No.
        # It's better to check where final_h comes from.
        # It is `enc_out.latent`. In CosmosHybridTokenizer, typical latent dim is defined in config.
        # Assuming `latent_channels` from cnn_cfg is the one.

        cond_dim = self.tok_cfg.cnn_cfg.model.latent_channels if self.use_latent else None

        self.decoder = ImplicitQueryDecoder(
            feature_dims=self.encoder.output_channels,
            embed_dim=decoder_cfg.get("embed_dim", 256),
            num_classes=self.cfg.num_classes,
            num_fusion_blocks=decoder_cfg.get("num_fusion_blocks", 1),
            fusion_block_type=decoder_cfg.get("fusion_block_type", "mbconv"),
            fusion_block_kwargs=decoder_cfg.get("fusion_block_kwargs", {}),
            deep_supervision=self.cfg.deep_supervision,
            cond_dim=cond_dim,
        )
        logger.info(
            f"Created ImplicitQueryDecoder with {len(self.encoder.output_channels)} scales (cond_dim={cond_dim})"
        )

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
        interaction_indexes = f_cfg.get("interaction_indexes", None) or interaction_indexes
        select_in_all_layers = f_cfg.get("select_in_all_layers", False)
        logger.info(
            f"Creating tokenizer encoder: {model_name}\n"
            f"taken interaction indexes: {interaction_indexes}, select in all layers: {select_in_all_layers}"
        )

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

        # freeze the tokenizer
        tok_backbone.requires_grad_(False)

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
            select_in_all_layers=select_in_all_layers,
            interp_ratio=f_cfg.get("interp_ratio", None),
            layer_in_channels=f_cfg.get("layer_in_channels", None),
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
        output = self.decoder(skips, input_h=x.shape[-2], input_w=x.shape[-1], cond=final_h)
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

    def set_grad_checkpointing(self, enable=True):
        """
        Enable or disable gradient checkpointing for the PyTorch model.

        Args:
            enable (bool): whether to enable gradient checkpointing
        """
        self.decoder.set_grad_checkpointing(enable)


class ImplicitQueryDecoder(nn.Module):
    """
    True Implicit Decoder that queries multi-scale features at arbitrary coordinates.

    Process:
    1. Project all multi-scale features to a shared embedding dimension.
    2. Generate a query coordinate grid for the target resolution (H, W).
    3. Sample all feature maps at these coordinates using grid_sample.
    4. Progressively fuse features from low-res to high-res (or all at once) using FusionBlocks.
    5. Decode the final fused features to the output class logits.

    This aligns with the `ImplicitQueryBlock` logic but adapted for a segmentation U-Net structure.
    """

    def __init__(
        self,
        feature_dims: list[int],
        embed_dim: int,
        num_classes: int,
        num_fusion_blocks: int = 1,
        fusion_block_type: str = "default",
        fusion_block_kwargs: dict | None = None,
        deep_supervision: bool = False,
        cond_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.grad_checkpointing = False
        if fusion_block_kwargs is None:
            fusion_block_kwargs = {}

        # Inject cond_dim into fusion_block_kwargs so FusionBlocks know to create projection
        if cond_dim is not None:
            fusion_block_kwargs["cond_dim"] = cond_dim

        self.feature_dims = feature_dims
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.num_scales = len(feature_dims)

        # 1) Project each scale to embed_dim
        self.projections = nn.ModuleList([nn.Conv2d(c, embed_dim, kernel_size=1) for c in feature_dims])

        # 2) Fusion stacks: Hierarchical fusion
        #    We fuse from Coarse -> Fine.
        #    For N scales, we have N-1 fusion stages.
        self.fusion_stacks = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        FusionBlock(embed_dim, fusion_block_type, **fusion_block_kwargs)
                        for _ in range(num_fusion_blocks)
                    ]
                )
                for _ in range(self.num_scales - 1)
            ]
        )

        # 3) Final prediction head
        #    Input is the final fused feature at target resolution
        norm_layer = get_norm_layer("layernorm2d")
        act_layer = get_act_layer("silu")

        self.final_head = nn.Sequential(
            norm_layer(embed_dim),
            act_layer(embed_dim),
            nn.Conv2d(embed_dim, num_classes, kernel_size=1, bias=True),
        )

        # Deep Supervision heads (optional)
        if self.deep_supervision:
            # We can supervise intermediate fusion results
            self.ds_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        norm_layer(embed_dim),
                        act_layer(embed_dim),
                        nn.Conv2d(embed_dim, num_classes, kernel_size=1, bias=True),
                    )
                    for _ in range(self.num_scales - 1)
                ]
            )

        logger.info(
            f"ImplicitQueryDecoder: scales={self.num_scales}, embed_dim={embed_dim}, "
            f"fusion_type={fusion_block_type}, deep_supervision={deep_supervision}"
        )

    @staticmethod
    def get_query_coords(batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Create normalized query grid in [-1, 1] with shape (B, H, W, 2)."""
        y_range = torch.linspace(-1.0, 1.0, height, device=device)
        x_range = torch.linspace(-1.0, 1.0, width, device=device)
        y, x = torch.meshgrid(y_range, x_range, indexing="ij")
        coords = torch.stack([x, y], dim=-1)
        return coords.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    def forward(
        self,
        skips: list[Float[Tensor, "b c h w"]],
        input_h: int,
        input_w: int,
        cond: Float[Tensor, "b latent_ch h w"] | None = None,
    ):
        """
        Args:
            skips: Multi-scale features. Order matters! Usually [Fine, ..., Coarse] or [Coarse, ..., Fine].
                   The DINOv3 adapter usually returns scales from High-Res to Low-Res (Fine to Coarse).
                   But let's double check. Assuming `skips` is [LowRes, ..., HighRes] for fusion convenience
                   or we just index carefully.
                   Actually, let's assume `skips` corresponds to `self.feature_dims`.
            input_h: Target output height
            input_w: Target output width
            cond: Optional global condition (e.g. from tokenizer latent)

        Returns:
            Logits at (input_h, input_w)
        """
        # Assuming skips are ordered same as feature_dims, typically [stage1, stage2, stage3, stage4]
        # stage1 is high res, stage4 is low res.
        # Usually U-Net decoders consume features from Deep (Low Res) to Shallow (High Res).
        # Let's say skips is [Scale 0 (High), Scale 1, Scale 2, Scale 3 (Low)].

        batch_size = skips[0].shape[0]
        device = skips[0].device
        target_res = (input_h, input_w)

        # A) Project all features to embed_dim
        #    And strictly maintain order: proj_feats[0] corresponds to feature_dims[0]
        proj_feats = [proj(feat) for proj, feat in zip(self.projections, skips, strict=False)]

        # B) Generate Implicit Query Grid
        query_coords = self.get_query_coords(batch_size, input_h, input_w, device)

        # C) Sample ALL projected features at the target resolution immediately
        #    This is the "Infinite Resolution Query" step.
        #    Instead of upsampling step-by-step, we sample everything to (H, W).
        #    Note: This is memory intensive for very large images, but standard for Implicit Decoders.
        sampled_tokens = []
        for feat in proj_feats:
            # grid_sample expects (B, C, Hin, Win) and grid (B, Hout, Wout, 2)
            # Returns (B, C, Hout, Wout)
            sampled = F.grid_sample(feat, query_coords, mode="bilinear", align_corners=False)
            sampled_tokens.append(sampled)  # All are now (B, embed_dim, H, W)

        # D) Hierarchical Fusion
        #    We want to fuse information.
        #    Strategy: Start from the Coarsest (Deepest) feature, fuses with finer features.
        #    If skips is [High, ..., Low], then sampled_tokens[-1] is the Coarsest.

        # Order: Deepest (Low Res) -> Shallowest (High Res)
        # sampled_tokens[-1] is the base, format (B, C, H, W)
        hidden = sampled_tokens[-1]

        # Optional: Condition sampling
        cond_sampled = None
        if cond is not None:
            # Interpolate/Sample condition to target resolution
            # F.interpolate is equivalent to grid_sample on regular grid, usually faster optimized
            cond_sampled = F.interpolate(cond, size=target_res, mode="bilinear", align_corners=False)

        seg_outputs = []

        # Iterate from Second-Deepest to Shallowest
        # range(self.num_scales - 2, -1, -1) -> indices [2, 1, 0] if 4 scales
        fusion_indices = list(range(self.num_scales - 2, -1, -1))

        # We have self.num_scales - 1 fusion stacks.
        # Stack 0 fuses (Deepest, 2nd Deepest).
        # Stack 1 fuses (Result, 3rd Deepest).
        # ...

        for i, scale_idx in enumerate(fusion_indices):
            # next_feat format (B, C, H, W)
            next_feat = sampled_tokens[scale_idx]

            # Use the i-th fusion stack
            # hidden (fused so far) + next_feat (detail from current scale)
            fusion_stack = self.fusion_stacks[i]

            for block in fusion_stack:  # type: ignore[not-iterable]
                # FusionBlock expects inputs in (B, C, H, W) format
                if self.grad_checkpointing:
                    hidden = checkpoint(block, hidden, next_feat, cond_sampled)
                else:
                    hidden = block(hidden, next_feat, cond=cond_sampled)

            if self.deep_supervision:
                # Decode intermediate result
                # hidden is (B, C, H, W)
                curr_out = self.ds_heads[i](hidden)
                seg_outputs.append(curr_out)

        # E) Final Decode
        logits = self.final_head(hidden)

        if not self.deep_supervision:
            return logits

        seg_outputs.append(logits)
        # Return Deep Supervision outputs
        # Current order in seg_outputs: [LowRes+1 Fusion, ..., Final Fusion]
        # Usually we return [Final, ..., Low]
        return seg_outputs[::-1]

    def set_grad_checkpointing(self, enable: bool = True):
        self.grad_checkpointing = enable


def test_inr_decoder():
    from src.data.hyperspectral_loader import get_fast_test_hyperspectral_data
    from fvcore.nn import parameter_count_table

    dl = get_fast_test_hyperspectral_data("fmow_RGB")
    sample = next(iter(dl))
    x = sample["img"].cuda()
    x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

    cfg = _create_default_cfg()
    cfg._debug = True
    cfg.tokenizer_pretrained_path = "runs/stage1_cosmos_hybrid/2025-12-21_23-52-12_hybrid_cosmos_f16c64_ijepa_pretrained_sem_no_lejepa/ema/tokenizer/model.safetensors"
    unet = TokenizerHybridUNet(cfg).cuda()
    unet.eval()

    print("=" * 60)
    print("Model Structure:")
    print("=" * 60)
    print(parameter_count_table(unet))
    print("=" * 60)

    print("=" * 60)
    print("Parameters:")
    print("=" * 60)
    with torch.autocast("cuda", torch.bfloat16):
        with torch.no_grad():
            y = unet(x)

    # Handle Tuple return from Deep Supervision
    if isinstance(y, (list, tuple)):
        print(f"Output shapes: {[out.shape for out in y]}")
    else:
        print(f"Output shape: {y.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in unet.parameters())
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters (frozen): {total_params - trainable_params:,}")

    print("=" * 60)


if __name__ == "__main__":
    """
    LOVELY_TENSORS=1 python -m src.stage2.segmentation.models.tokenize-pr_backbone_implicit
    """
    with logger.catch():
        test_inr_decoder()

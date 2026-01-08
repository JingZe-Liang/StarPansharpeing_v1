"""
Mingtok-like three-stage tokenizer with low-level and semantic-level feature alignment.

1. add t-dependent decoder (flow transformer decoder + flow head)
2. fix flow UViT decoder
    //2.1 fix time t-r conditioning embedder (in utilities)
//3. add alpha-flow
"""

import ema_pytorch
import math
import random
from dataclasses import asdict
from functools import partial
from typing import Any, List, Optional, cast, no_type_check

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf
from timm.layers import create_conv2d, create_norm_act_layer, create_norm_layer
from timm.layers.helpers import to_2tuple
from torch import Tensor
from typing_extensions import (
    Annotated,
    Callable,
    Literal,
    Self,
    Sequence,
    TypeGuard,
    TypeAlias,
    TypedDict,
    Union,
)

from src.utilities.config_utils import to_easydict_recursive, function_config_to_basic_types
from src.utilities.network_utils.network_loading import load_weights_with_shape_check
from src.utilities.transport.flow_matching.meanflow import MeanFlow
from src.utilities.transport.flow_matching.transport import Sampler
from src.utilities.transport.flow_matching.transport import Transport as FM_Transport
from src.utilities.transport.tim.transition import TransitionSchedule
from src.utilities.transport.tim.transports import OT_FM
from src.utilities.transport.tim.transports import Transport as Tim_Transport

from src.stage1.discretization.collections import FSQ
from src.stage1.discretization.collections import BinarySphericalQuantizer as BSQ
from src.stage1.discretization.collections.kl_continuous import (
    DiagonalGaussianDistributionV2 as DiagonalGaussianDistribution,
)
from src.stage1.discretization.collections.multiscale_bsq import MultiScaleBSQ
from src.stage1.discretization.collections.multiscale_leechq import MultiScaleLeechQ
from src.stage1.discretization.collections.psd import PowerSphericalDistribution
from src.stage1.utilities.losses.latent_reg import LatentMaskConfig, lmr_apply
from src.utilities.network_utils import compile_decorator, model_compiled_flag, safe_init_weights

from .modules import TransformerTokenizer
from .modules.flowhead import (
    FlowDecoder,
    TimFlowDecoder,
    build_flow_matching_transport,
    build_tim_scheduler,
)
from .modules.layers2d import Encoder as ResEncoder
from .modules.layers2d import Decoder as ResDecoder
from .modules.proj import build_mlp
from .modules.t_transformer import FlowTransformerConditioned
from .modules.uvit_decoder import UViTDecoder

LossOutput = TypedDict(
    "LossOutput",
    {"flow_loss": torch.Tensor},
)
DecoderOutput: TypeAlias = tuple[Tensor, LossOutput]
QuantizerOutput: TypeAlias = tuple[Tensor, Tensor, dict[str, Any] | None]

# *==============================================================
# * Default Configs
# *==============================================================


def _create_flow_transport_cfg():
    ################## Transport Configs ####################

    ################# Tim flow config
    tim_schedule_kwargs_str = (
        "diffusion_ratio=0.5 consistency_ratio=0.1 weight_time_tangent=true "
        "differential_epsilon=0.005 derivative_type='dde' weight_time_type='sqrt'"
    )
    tim_schedule_cfg = OmegaConf.from_dotlist(tim_schedule_kwargs_str.split(" "))
    tim_transport_str = (
        "P_mean=0.0 P_std=1.0 sigma_d=1.0 enhance_target=false w_gt=1.0 w_cond=0.75 w_start=0.3 w_end=0.8"
    )
    tim_transport_cfg = OmegaConf.from_dotlist(tim_transport_str.split(" "))

    ################## Flow head FM config
    # fm_kwargs_str = "num_sampling_steps=100 train_schedule=fat_lognormal"
    fm_kwargs_str = (
        "model_type=velocity path_type=linear loss_type=velocity train_eps=0 sample_eps=0 "
        "cfm_factor=0.0 time_sample_type=sigmoid"
    )
    fm_kwargs_cfg = OmegaConf.from_dotlist(fm_kwargs_str.split(" "))

    ################## MeanFlow config
    # flow_ratio=0.75 is for the training stability
    mf_kwargs_str = (
        "flow_ratio=0.75 time_dist=['lognorm',-0.4,1.0] cfg_ratio=0.2 cfg_scale=2.0 cfg_uncond='v' jvp_api='autograd'"
    )
    mf_kwargs_cfg = OmegaConf.from_dotlist(mf_kwargs_str.split(" "))

    ################## Sampling Configs ##################

    tim_sample_str = (
        "num_steps=8 stochasticity_ratio=0.1 sample_type='transition' cfg_scale=2.0 cfg_low=0.0 cfg_high=0.7"
    )
    tim_sample_cfg = OmegaConf.from_dotlist(tim_sample_str.split(" "))

    # fm_sample_str = "sample_steps=100 schedule=pow_0.25 cfg_scale=2.0 tbar=true"  # flow head fm mannually sample kwargs
    fm_sample_str = (
        "sampling_method=euler num_steps=8 cfg=1.0 progress=true cfg_interval=[0.0,0.7] "
        "clip_velocity_per_step=false sampling_time_type=pow_2.5"
    )
    fm_sample_cfg = OmegaConf.from_dotlist(fm_sample_str.split(" "))

    return {
        "tim": [tim_schedule_cfg, tim_transport_cfg, tim_sample_cfg],
        "fm": [fm_kwargs_cfg, fm_sample_cfg],
        "mf": [mf_kwargs_cfg],
    }


def _create_flow_decoder(
    flow_type,
    decoder_type,
    *,
    tim_cfgs: list | None = None,
    fm_cfgs: list | None = None,
    mf_cfgs: list | None = None,
):
    # Models

    ################# UViT config ##################
    # TODO: add fm config
    is_fm = lambda ft: ft == "fm"
    uvit_flow_str: str = (
        "in_chan=3 z_dim=1024 channels=128 ch_mult=[1,1,2,4,8] act_fn='silu' "
        "vit_act_fn='silu' layers_per_block=2 num_attention_heads=8 "
        "dropout=0.0 norm_num_groups=32 time_scale_shift=true "
        "mid_nlayers=12 mid_chan=1024 mid_theta=10000.0 eps=1e-6 "
        "ada_norm=true ctx_emb_dim=768 t_emb_mult=4 "
        "learned_pos_embed=false relative_pos_embed=false "
        "init=null use_act_ckpt=true total_resolutions=16 "
        f"time_cond_type={'t' if is_fm(flow_type) else 't-r'} "
        f"img_size=512 flow_type={flow_type} cfg_prob=0.0 "
        f"jvp={True if not is_fm(flow_type) else False}"
    )
    uvit_flow_cfg = OmegaConf.from_dotlist(uvit_flow_str.split(" "))
    # NOTE: supports tim, fm, and mf flows
    if flow_type == "tim":
        assert tim_cfgs is not None, "tim_cfgs must be provided when flow_type='tim'"
        tim_schedule_cfg, tim_transport_cfg, _ = tim_cfgs
        uvit_flow_cfg.transition_schedule = tim_schedule_cfg
        uvit_flow_cfg.transport = tim_transport_cfg
    elif flow_type == "fm":
        assert fm_cfgs is not None, "fm_cfgs must be provided when flow_type='fm'"
        uvit_flow_cfg.transport = fm_cfgs[0]  # transport
        uvit_flow_cfg.sampler = fm_cfgs[1]  # not used
    elif flow_type == "mf":
        assert mf_cfgs is not None, "mf_cfgs must be provided when flow_type='mf'"
        uvit_flow_cfg.transport = mf_cfgs[0]  # meanflow config
    else:
        raise ValueError(f"Unknown flow type: {flow_type}")

    if uvit_flow_cfg.jvp:
        logger.log("NOTE", "use JVP attention implem, this is experimental")

    ################# flow head config ###################
    flow_head_str: str = (
        # Transformer decoder kwargs
        "in_chan=768 embed_dim=768 depth=12 num_heads=12 patch_size=1 mlp_ratio=4.0 "
        "norm_layer='rmsnorm' drop_path=0.1 pe_type=rope rope_kwargs.rope_theta=10000.0 "
        "last_norm='rmsnorm' out_chan=512 decoder_img_size=32 head='linear' "
        "n_reg_tokens=0 mask_train_ratio=0.0 is_causal=false attn_type='sdpa' "
        # Flow decoder kwargs
        "target_channels=512 z_channels=512 flow_depth=6 flow_width=768 "
        "patch_size=16 flow_img_size=512 grad_checkpointing=false use_cfg=false "
        "head_type='progressive' head_kwargs.progressive_dims=[512,384,384] cfg_prob=0.5 total_resolutions=16 "
        f"flow_type={flow_type} "
    )
    flow_head_cfg = OmegaConf.from_dotlist(flow_head_str.split(" "))
    if flow_type == "tim":
        flow_head_cfg.fm_kwargs = tim_transport_cfg
        flow_head_cfg.transition_schedule_kwargs = tim_schedule_cfg
    elif flow_type == "fm":
        assert fm_cfgs is not None, "fm_cfgs must be provided when flow_type='fm'"
        flow_head_cfg.fm_kwargs = fm_cfgs[0]
    elif flow_type == "mf":
        assert mf_cfgs is not None, "mf_cfgs must be provided when flow_type='mf'"
        flow_head_cfg.fm_kwargs = mf_cfgs[0]

    ########## Init the config here
    if decoder_type == "flow_head":
        pixel_decoder_cfg = flow_head_cfg
    else:
        pixel_decoder_cfg = uvit_flow_cfg

    return pixel_decoder_cfg


def create_default_mingtok_cfg(flow_type: str = "fm", decoder_type: str = "uvit"):
    """Create a default configuration for Mingtok-like tokenizer model.
    Returns:
        OmegaConf: Default configuration object.

    total_resolution: N downsampled times of the original image size.
    """
    ########### Encoder and decoder

    ################ Encoder
    # CNN encoder and decoder
    # res-cnn encoder f8 + transformer f2l12h8

    res_enc_str: str = (
        "in_channels=3 out_channels=512 channels=128 attn_resolutions=[32] "
        "channels_mult=[2,4,4] dropout=0.0 spatial_compression=8 "
        "num_res_blocks=2 resolution=1024 z_channels=512 "
        "act_checkpoint=true latent_channels=16 norm_type=gn norm_groups=32 "
        "block_name=res_block moe_token_mixer_type=res_block hidden_factor=4 "
        "use_residual_factor=false patch_method=haar patch_size=1 attn_type='none' padding_mode=zeros"
    )
    res_enc_cfg = OmegaConf.from_dotlist(res_enc_str.split(" "))
    transf_enc_str: str = (
        "in_chan=512 embed_dim=768 depth=8 num_heads=8 patch_size=2 out_patch_size=1 mlp_ratio=4.0 "
        "norm_layer='layernorm' drop_path=0.0 pe_type='rope' rope_kwargs.rope_theta=10000.0 "
        "last_norm='layernorm' out_chan=768 img_size=64 patcher_type='patch_embedder' "
        "additional_pe=false n_reg_tokens=0 attn_type='sdpa'"
    )
    transf_enc_cfg = OmegaConf.from_dotlist(transf_enc_str.split(" "))
    enc_cfg = OmegaConf.create(
        {
            "res_encoder": res_enc_cfg,
            "transformer_encoder": transf_enc_cfg,
            # TODO: to check if cnn encoder is useful for hyperspectral data
            "encoder_type": "hybrid",
            "z_dim": 768,
            "latent_dim": 64,
        }
    )

    ############# Semantic decoder

    dec_str: str = (
        "in_chan=64 embed_dim=1024 out_chan=1024 depth=12 num_heads=16 patch_size=1 mlp_ratio=4.0 "
        "norm_layer='layernorm' drop_path=0.0 pe_type='rope' rope_kwargs.rope_theta=10000.0 "
        "last_norm='layernorm' img_size=32 "
        "n_reg_tokens=0 is_causal=false attn_type='sdpa'"
    )
    dec_cfg = OmegaConf.from_dotlist(dec_str.split(" "))

    #############  two variants of flow decoders
    # UViT and flow head
    flow_dec_cfg_dict = _create_flow_transport_cfg()
    tim_cfgs, fm_cfgs, mf_cfgs = (
        flow_dec_cfg_dict["tim"],
        flow_dec_cfg_dict["fm"],
        flow_dec_cfg_dict["mf"],
        # TODO: add scm, mdm, mdm2, rcm
    )
    pixel_decoder_cfg = _create_flow_decoder(
        flow_type, decoder_type, tim_cfgs=tim_cfgs, fm_cfgs=fm_cfgs, mf_cfgs=mf_cfgs
    )

    ######## repa projection config

    repa_proj_str = (
        "low_lvl_repa_out_chan=1024 sem_repa_out_chan=1024 low_lvl_cache_layers=[3,6,9,11] "
        "sem_cache_layers=[3,6,9,11] low_lvl_repa_proj_chans=[768,768,768,768] "
        "sem_repa_proj_chans=[1024,1024,1024,1024]"
    )
    repa_proj_cfg = OmegaConf.from_dotlist(repa_proj_str.split(" "))

    ######## Tokenizer main config

    main_str = (
        f"decoder_type={decoder_type} compile=false straight_through_latent=false "
        "quantizer_type=null random_quant=0.0 use_latent_mask=false"
    )
    main_cfg = OmegaConf.from_dotlist(main_str.split(" "))
    main_cfg.sampling_options_default = flow_dec_cfg_dict[flow_type][-1]
    main_cfg.latent_mask_config = OmegaConf.structured(LatentMaskConfig())

    tokenizer_cfg_default = OmegaConf.create(
        {
            # encoder, semantic decoder and pixel decoder configs
            "low_level_encoder": enc_cfg,
            "semantic_decoder": dec_cfg,
            "pixel_decoder": pixel_decoder_cfg,
            "tokenizer": main_cfg,
            "repa_proj": repa_proj_cfg,
        }
    )

    return tokenizer_cfg_default


def create_default_mingtok_pretrained_cfg() -> DictConfig:
    cfg = """
    low_level_encoder:
        z_dim: 512
        latent_dim: 16
        encoder_type: cnn_only
        norm_latent_type: null
        clip_latent_value: null
        res_encoder:
            in_channels: 512
            out_channels: 512
            channels: 128
            attn_resolutions: [32]
            channels_mult: [2,4,4]
            spatial_compression: 8
            num_res_blocks: 2
            z_channels: 512
            act_checkpoint: true
            norm_type: rmsnorm2d
            block_name: res_block
            use_residual_factor: false
            patch_method: haar
            patch_size: 1
            attn_type: null
            padding_mode: reflect
            dropout: 0.0
            resolution: 512

    semantic_decoder:
        in_chan: 512
        out_chan: 1024
        embed_dim: 1024
        depth: 24
        num_heads: 16
        patch_size: 2
        with_cls_token: true
        out_patch_size: 1
        mlp_ratio: 4.0
        norm_layer: flarmsnorm
        drop_path: 0.0
        pe_type: rope_dinov3
        rope_kwargs:
            rope_theta: 100.0
        last_norm: flarmsnorm
        img_size: 64
        patcher_type: patch_embedder
        additional_pe: true
        n_reg_tokens: 4
        other_blk_kwargs:
            use_gate: true

    pixel_decoder:
        decoder_type: hybrid
        latent_dim: 16
        z_dim: 512
        total_resolutions: 8
        res_decoder:
            in_channels: 512
            out_channels: 512
            channels: 128
            attn_resolutions: [32]
            channels_mult: [2,4,4]
            spatial_compression: 8
            num_res_blocks: 2
            z_channels: 512
            act_checkpoint: true
            norm_type: rmsnorm2d
            block_name: res_block
            use_residual_factor: false
            patch_method: haar
            patch_size: 1
            attn_type: null
            padding_mode: reflect
            dropout: 0.0
            resolution: 512
        transformer_decoder:
            in_chan: 512  # if taken z is 512 else is latent, in_chan is 16
            out_chan: 512
            embed_dim: 1024
            depth: 8
            num_heads: 16
            patch_size: 2
            out_patch_size: 2
            mlp_ratio: 4.0
            norm_layer: flarmsnorm
            drop_path: 0.0
            pe_type: rope_dinov3
            rope_kwargs:
                rope_theta: 100.0
            last_norm: flarmsnorm
            img_size: 64
            patcher_type: patch_embedder
            additional_pe: true
            n_reg_tokens: 4
            other_blk_kwargs:
                use_gate: true

    repa_proj:
        low_lvl_repa_out_chan: 1024
        sem_repa_out_chan: 1024
        low_lvl_cache_layers: [0,1,2,-1]
        sem_cache_layers: [5,11,17,23]
        low_lvl_repa_proj_chans: [256,512,512,512]
        sem_repa_proj_chans: [1024,1024,1024,1024]

    tokenizer:
        straight_through_latent: false
        sem_pix_decoder_type: seperated
        sem_decoder_take: z
        sampling_options_default: {}
        quantizer_type: null
        random_quant: 0.0
        use_latent_mask: false
        latent_mask_config:
            mask_ratios: [0.0, 0.25, 0.5, 0.75]
            block_sizes: {16: [1, 1], 32: [2, 2]}
            mask_probs: {16: [0.7, 0.1, 0.1, 0.1], 32: [0.6, 0.1, 0.15, 0.15]}
        pretrained_task: ['ijepa','contrastive']

    """
    cfg = OmegaConf.create(cfg)
    return cfg


# *==============================================================
# * Utilities
# *==============================================================


def _to_memformat_channels_last(x):
    if not model_compiled_flag:
        return x
    if (not x.is_cuda) or x.ndim != 4:
        return x
    if x.is_contiguous(memory_format=torch.channels_last):
        return x
    return x.contiguous(memory_format=torch.channels_last)


def is_tuple_list(x):
    return isinstance(x, (tuple, list, ListConfig))


def is_sequence_shape(
    shape: torch.Size | tuple[int, ...] | list[int] | int,
) -> TypeGuard[torch.Size | tuple[int, ...] | list[int]]:
    return isinstance(shape, (torch.Size, list, tuple))


def get_chan_from_shape(shape: torch.Size | tuple[int, ...] | list[int] | int) -> int:
    if isinstance(shape, int):
        return shape
    return int(shape[1])


def latent1d_to_2d(
    feat: Tensor,
    feat_2d_shape: Annotated[torch.Size | tuple, "bs,c,gh,gw"] | None = None,
    inp_shape: Annotated[torch.Size | tuple, "bs,c,h,w"] | None = None,
    total_resolutions: int | None = None,
):
    """Convert 1d latent tokens to 2d feature map."""
    if feat_2d_shape is None:
        assert inp_shape is not None and total_resolutions is not None, (
            "Either feat_2d_shape or inp_shape and total_resolutions must be provided"
        )
        gh, gw = (torch.as_tensor(inp_shape[-2:]) // total_resolutions).tolist()
    elif isinstance(feat_2d_shape, (torch.Size, tuple, list)) and len(feat_2d_shape) == 4:
        gh, gw = feat_2d_shape[2:]
    elif feat.ndim == 4:
        # Assume is square
        gh = gw = int(feat.shape[1] ** 0.5)
    else:
        raise ValueError(f"Invalid input args: {feat.shape=}, {feat_2d_shape=}, {inp_shape=}, {total_resolutions=}")

    if feat.ndim == 4:
        assert feat.shape[-2:] == (gh, gw), f"feat shape {feat.shape} already 2d with shape {gh},{gw}"
        return feat
    elif feat.ndim == 3:
        feat_2d = rearrange(feat, "b (h w) c -> b c h w", h=gh, w=gw)
        return feat_2d
    else:
        raise ValueError(f"Invalid feat shape {feat.shape}")


def latent_2d_to_1d(feat: Tensor):
    if feat.ndim == 3:
        return feat
    elif feat.ndim == 4:
        return rearrange(feat, "b c h w -> b (h w) c")
    else:
        raise ValueError(f"Invalid feat shape {feat.shape}")


# *==============================================================
# * Model Sampling Utilities
# *==============================================================


def create_sampling_step_fn(
    model: nn.Module,
    model_cond: Tensor | None = None,
    cfg_scale: float = 1.0,
    cfg_interval: list[float] | tuple[float, float] | None = None,
    clip_v=False,
    model_type: str = "velocity",
    inp_shape=None,
    get_null_cond: Callable[[Tensor], Tensor] | Tensor | None = None,
    x_init: Tensor | None = None,
):
    if clip_v:
        assert x_init is not None, "x_init must be provided for clip velocity"

    if cfg_scale > 1.0:
        assert model_cond is not None, "model_cond is None for CFG"

    def _get_null_cond_fn():
        ######### Get null conditions
        if get_null_cond is not None:
            if callable(get_null_cond):
                null_cond = get_null_cond(model_cond)
            elif isinstance(get_null_cond, Tensor):
                null_cond = get_null_cond
            else:
                raise ValueError("get_null_cond must be a callable or a Tensor, got {}".format(type(get_null_cond)))
        elif model_cond is None:
            null_cond = None
        else:
            null_cond = torch.zeros_like(model_cond)

        return null_cond

    @torch.no_grad()
    def _model_step_fn(x, t):
        ## Model forward
        v = model(x, t, z=model_cond, inp_shape=inp_shape)

        null_cond = _get_null_cond_fn()
        if (
            cfg_scale > 1.0
            and null_cond is not None
            and cfg_interval is not None
            # t is a tensor
            and t[0].item() >= cfg_interval[0]
            and t[0].item() <= cfg_interval[1]
        ):
            # classifier-free guidance
            assert cfg_scale > 1.0, "cfg_scale must be > 1.0 for CFG"
            assert null_cond is not None, "null_cond is None for CFG"
            v_uncond = model(x, t, z=null_cond, inp_shape=inp_shape)
            v = v_uncond + cfg_scale * (v - v_uncond)
            logger.debug(f"CFG sampling step: {t[0].item()}")

        # clip fake x1 to (-1, 1)
        # v = x0 - x1
        if clip_v:
            assert x_init is not None, "x_init is None for clip_v"
            if model_type == "x1":
                x1_hat = v
            elif model_type == "velocity":
                x1_hat = x_init + v
            else:
                raise ValueError(f"Model type {model_type} is not supported.")

            x1_hat = x1_hat.clamp(-1, 1)
            # back to clip velocity
            v = x1_hat - x_init

        return v

    return _model_step_fn


# *==============================================================
# * Model
# * Encoder, Decoder, DecoderFlowHead, UViTDecoder
# *==============================================================

# TODO: Fix EncoderLowLevel support one transformer (patch size 16) and one decoder progressively
# upsample 16x
# or a time-dependent CNN decoder (like those in uvit decoder does).


class EncoderLowLevel(nn.Module):
    """
    NOTE
        # -----------------------------------------------------------------------------------
        # Three types of this encoder:
        #   1. CNN encoder only
        #   2. Transformer encoder only - seems that does not work for hyperspectral data.
        #   3. CNN + Transformer encoder
        # -----------------------------------------------------------------------------------
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        encoder_type = cfg.get("encoder_type", "hybrid")
        assert encoder_type in (
            "cnn_only",
            "transformer_only",
            "hybrid",
        ), f"Invalid encoder type {encoder_type}"
        self.encoder_type = encoder_type
        self.latent_dim = cfg.latent_dim

        res_cfg = cfg.get("res_encoder", None)
        transf_cfg = cfg.get("transformer_encoder", None)
        if transf_cfg is not None:
            transf_kwargs = dict(
                **transf_cfg,
                projections={"input": None, "output": "us_average"},
                is_causal=False,
                head="linear",
            )
        else:
            transf_kwargs = {}

        if encoder_type == "hybrid":
            assert res_cfg is not None, "res_encoder config must be provided if use_cnn"
            self.res_encoder = ResEncoder(**res_cfg)
            self.transformer_encoder = TransformerTokenizer(**transf_kwargs)
            self.z_to_latent = nn.Conv2d(cfg.z_dim, cfg.latent_dim, kernel_size=1)
        elif encoder_type == "cnn_only":
            assert res_cfg is not None, "res_encoder config must be provided if use_cnn"
            self.res_encoder = ResEncoder(**res_cfg)
            self.z_to_latent = nn.Conv2d(cfg.z_dim, cfg.latent_dim, kernel_size=3, padding=1)
        else:  # transformer only
            assert transf_cfg is not None, "transformer_encoder config must be provided"
            self.transformer_encoder = TransformerTokenizer(**transf_kwargs)
            self.z_to_latent = nn.Conv2d(cfg.z_dim, cfg.latent_dim, kernel_size=1)

        # self.latent_scale = nn.Parameter(torch.ones(cfg.latent_dim))
        # nn.init.trunc_normal_(self.latent_scale, std=0.01, a=-1e-2, b=1e-2)

        # norm the latent
        self.norm_latent = nn.Identity()
        if getattr(cfg, "norm_latent_type", None) is not None:
            if cfg.norm_latent_type == "norm_no_affine":
                self.norm = create_norm_layer("layernorm2d", cfg.latent_dim, affine=False)
            elif cfg.norm_latent_type == "norm":
                self.norm = create_norm_layer("layernorm2d", cfg.latent_dim)
            else:
                raise ValueError(f"Invalid norm_latent_type {cfg.norm_latent_type}")

        self.clip_latent_value = getattr(cfg, "clip_latent_value", None)
        if self.clip_latent_value is not None:
            assert isinstance(self.clip_latent_value, (tuple, list))

    def forward(self, x, get_intermidates: list[int] | None = None):
        """Encode the input image into low-level latent tokens."""
        x = _to_memformat_channels_last(x)
        if self.encoder_type == "hybrid":
            # Hybrid encoder, takes the transformer encoder's intermidates
            res_out = self.res_encoder(x)
            out = self.transformer_encoder(
                res_out,
                ret_2d_tokens=True,  # Return 2d tokens
                ret_all=True,
                get_intermidates=get_intermidates,
            )
            z = out["head_out"]  # z is 2d: (b, c, h, w)
            h = self.z_to_latent(z)  # Conv2d directly on 2d

        elif self.encoder_type == "cnn_only":
            res_out = self.res_encoder(x, ret_interm_feats=get_intermidates)
            if get_intermidates is not None:
                z, intermidates = res_out
                out = {"intermidates": intermidates}
            else:
                z = res_out
                out = {}
            # z is already 2d from cnn: (b, c, h, w)
            h = self.z_to_latent(z)  # Conv2d directly on 2d

        else:  # transformer only
            out = self.transformer_encoder(x, ret_2d_tokens=True, ret_all=True, get_intermidates=get_intermidates)
            z = out["head_out"]  # z is 2d: (b, c, h, w)
            h = self.z_to_latent(z)  # Conv2d directly on 2d

        # h = h * self.latent_scale.view(1, h.shape[1], 1, 1)
        h = self.norm_latent(h)
        if self.clip_latent_value is not None:
            h = h.clip(*self.clip_latent_value)
        h = _to_memformat_channels_last(h)
        return dict(latent=h, z=z, **out)

    def get_last_layer(self):
        # TODO: implem it
        return None

    def init_weights(self):
        if hasattr(self, "transformer_encoder"):
            self.transformer_encoder.init_weights()
        if hasattr(self, "res_encoder"):
            self.res_encoder.init_weights()
        nn.init.trunc_normal_(self.z_to_latent.weight, std=0.01, a=-0.4, b=0.4)
        nn.init.zeros_(self.z_to_latent.bias)
        logger.info("init low-level encoder with smaller std conv to latent.")


class DecoderSemantic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch_size = cfg.patch_size
        self.out_patch_size = cfg.out_patch_size
        self.decoder = TransformerTokenizer(
            patch_embeder_with_norm=True, projections={"input": None, "output": None}, head="linear", **cfg
        )

    def forward(self, x, masks=None, get_intermidates=None):
        """
        Encode the latent low-level tokens and convert into semantic tokens.
        """
        out = self.decoder(x, ret_2d_tokens=False, ret_all=True, get_intermidates=get_intermidates, masks=masks)
        x = out["head_out"]
        assert x.ndim == 3, "1D tokens are needed for pretraining."
        return dict(sem_tokens=x, **out)

    def init_weights(self):
        self.decoder.init_weights()


class DecoderFlowHead(nn.Module):
    """
    Big transformer (time-agnostic) with a small (but wide) flow head.

    This decoder uses a TransformerTokenizer to process semantic tokens and a FlowDecoder
    to generate the final image reconstruction. Unlike DecoderUViT, this decoder does not
    accept time as input to the transformer, but passes (t, z) to the flow head where z
    serves as the conditioning.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch_size = cfg.patch_size
        self.out_patch_size = 1
        self.total_resolutions = getattr(cfg, "total_resolutions", cfg.patch_size)
        self.transformer_t_conditioned = getattr(cfg, "transformer_t_conditioned", True)

        # Transformer tokenizer for processing semantic tokens
        # NEEDTEST: make the transformer time-dependent
        basic_kwargs = dict(
            in_chan=cfg.in_chan,
            embed_dim=cfg.embed_dim,
            out_chan=cfg.out_chan,
            img_size=cfg.decoder_img_size,
            patch_size=cfg.patch_size,
            patcher_type="linear",
            depth=cfg.depth,
            mlp_ratio=cfg.mlp_ratio,
            norm_layer=cfg.norm_layer,
            drop_path=cfg.drop_path,
            projections={"input": None, "output": None},
            pe_type=cfg.pe_type,
            rope_kwargs=cfg.rope_kwargs,
            last_norm=cfg.last_norm,
            mask_train_ratio=0.0,
            is_causal=cfg.is_causal,
            n_reg_tokens=cfg.n_reg_tokens,
            attn_type=cfg.attn_type,
            head=cfg.head,
            jvp=getattr(cfg, "jvp", False),
        )

        if self.transformer_t_conditioned:
            # WARNING: need to test
            self.decoder = FlowTransformerConditioned(
                other_blk_kwargs=dict(
                    time_embed_dim=cfg.embed_dim,
                    cxt_embed_dim=cfg.embed_dim,
                    fuse_t_z=cfg.fuse_t_z,
                ),
                **basic_kwargs,
            )
        else:
            # not time-conditioned, like MAR
            # leave the time-conditioning to flow head.
            self.decoder = TransformerTokenizer(**basic_kwargs)

        # Flow decoder for image generation
        self.flow_type = cfg.flow_type
        self.flow_decoder: FlowDecoder | TimFlowDecoder
        if cfg.flow_type == "fm":
            self.flow_decoder = FlowDecoder(
                target_channels=cfg.target_channels,
                z_channels=cfg.z_channels,
                depth=cfg.flow_depth,
                width=cfg.flow_width,
                grad_checkpointing=cfg.grad_checkpointing,
                use_cfg=cfg.use_cfg,
                cfg_prob=cfg.cfg_prob,
                patch_size=cfg.patch_size,
                img_size=cfg.flow_img_size,
                head_type=cfg.head_type,
                head_kwargs=cfg.head_kwargs,
                # fm config
                num_sampling_steps=cfg.fm_kwargs.num_sampling_steps,
                train_schedule=cfg.fm_kwargs.train_schedule,
                stand_alone=not self.transformer_t_conditioned,
            )
        elif cfg.flow_type == "tim":
            # TODO: fix the standalone args
            self.flow_decoder = TimFlowDecoder(
                target_channels=cfg.target_channels,
                z_channels=cfg.z_channels,
                depth=cfg.flow_depth,
                width=cfg.flow_width,
                grad_checkpointing=cfg.grad_checkpointing,
                use_cfg=cfg.use_cfg,
                cfg_prob=cfg.cfg_prob,
                patch_size=cfg.flow_patch_size,
                img_size=cfg.img_size,
                head_type=cfg.head_type,
                head_kwargs=cfg.head_kwargs,
                time_cond_type=cfg.time_cond_type,
                # tim config
                fm_kwargs=cfg.fm_kwargs,
                transition_schedule_kwargs=cfg.transition_schedule_kwargs,
            )
        else:
            raise ValueError(f"Unknown flow_type: {cfg.flow_type}")

        ####### Build transport and sampler
        self.transport: Optional[FM_Transport | Tim_Transport] = None
        self.sampler: Optional[Sampler] = None
        if self.transformer_t_conditioned:
            if cfg.flow_type == "fm":
                self.transport, self.sampler = build_flow_matching_transport()
            elif self.flow_type == "tim":
                self.transport = build_tim_scheduler()
            else:
                raise ValueError(f"{self.flow_type} is not supported.")

    ############ Traning functions #############

    def flow_train_forward_head_standalone(self, x, h, inp_shape, ema_model, clamp=False):
        assert not self.transformer_t_conditioned
        assert self.training, "flow_train_forward should only be used in training mode"

        # Let the decoder decodes the hiddens first (w/o time-dependency)
        h = self.decoder(h, ret_2d_tokens=False, ret_all=False, out_shape=None)

        # Training mode
        if self.flow_type == "fm":
            flow_output = self.flow_decoder(
                z_blc=h,  # condition from transformer
                x_bchw=x,  # input image for training loss
                inp_shape=inp_shape,
                mode="train",
                sample_kwargs={},
            )
            flow_loss = flow_output["loss"].mean()  # mean out batch dim
            recon = flow_output["pred_x_clean"]  # predicted clean image
        elif self.flow_type == "tim":
            recon, flow_losses = self.flow_decoder.training_loss(  # type: ignore
                z_blc=h,  # condition from transformer
                x_bchw=x,  # input image for training loss
                inp_shape=inp_shape,
                ema_model=ema_model,
                clamp=clamp,
            )
            flow_loss = flow_losses["flow_loss"].mean()
        else:
            raise ValueError(f"Unknown flow_type: {self.flow_type}")

        return recon, flow_loss

    def _forward_non_standalone_all_model(self, xt, t, z, inp_shape):
        transf_out = self.decoder(xt, t, z, inp_shape, ret_all=False)
        head_out = self.flow_decoder(transf_out, xt, t, inp_shape=inp_shape)
        return head_out

    def flow_train_all(self, x, h, inp_shape, ema_model, clamp=False):
        assert self.transformer_t_conditioned

        if self.flow_type == "fm":
            self.transport = cast(FM_Transport, self.transport)
            terms = self.transport.training_losses(self._forward_non_standalone_all_model, x, model_kwargs={"z": h})
            flow_loss = terms["loss"].mean()
            recon = terms["pred_x_clean"]
            return recon, flow_loss

        elif self.flow_type == "tim":
            transport = cast(Tim_Transport, self.transport)
            flow_loss, _, loss_dict, breakdowns = transport(  # type: ignore
                self,
                ema_model if ema_model is not None else None,
                self,
                batch_size=h.shape[0],
                x=x,
                z=h,
                model_kwargs={"inp_shape": inp_shape, "z": h},
                ema_kwargs={"inp_shape": inp_shape, "z": torch.zeros_like(h)},
            )
            return breakdowns["x0_pred"], flow_loss

        else:
            raise ValueError(f"Unknown flow_type: {self.flow_type}")

    ########## Sample functions ###############

    def flow_sample_head_stand_alone(self, h, inp_shape, sample_kwargs={}, clamp=False):
        assert not self.transformer_t_conditioned
        assert not self.training, "flow_sample should only be used in inference mode"
        # Inference mode
        if self.flow_type == "fm":
            recon = self.flow_decoder(
                z_blc=h,  # condition from transformer
                x_bchw=None,  # no input image during inference
                inp_shape=inp_shape,
                mode="sample",
                sample_kwargs=sample_kwargs,
            )
        elif self.flow_type == "tim":
            recon = self.flow_decoder.sample(  # type: ignore
                z_blc=h,  # condition from transformer
                inp_shape=inp_shape,
                clamp=clamp,
                sample_kwargs=sample_kwargs,
                ret_trajectory=False,
            )
        else:
            raise ValueError(f"Unknown flow_type: {self.flow_type}")
        return recon

    def flow_sample_all(self, x, h, inp_shape, sample_kwargs={}, clamp=False, ret_trajectory=False):
        assert self.transformer_t_conditioned
        if self.flow_type == "fm":
            cfg_scale, cfg_interval, clip_v = (
                sample_kwargs.pop("cfg", 1.0),
                sample_kwargs.pop("cfg_interval", (0, 1)),
                sample_kwargs.pop("clip_v", False),
            )
            x_init = x.clone()

            @torch.no_grad()
            def _model_step_fn(x, t):
                nonlocal cfg_scale, cfg_interval, clip_v

                null_cond_h = torch.zeros_like(h)
                v = self._forward_non_standalone_all_model(x, t, h, inp_shape)
                if (
                    cfg_scale > 1.0
                    and null_cond_h is not None
                    and cfg_interval is not None
                    # t is a tensor
                    and t[0].item() >= cfg_interval[0]
                    and t[0].item() <= cfg_interval[1]  # fmt: skip
                ):
                    # classifier-free guidance
                    assert cfg_scale > 1.0, "cfg_scale must be > 1.0 for CFG"
                    assert null_cond_h is not None, "null_cond_h is None for CFG"
                    v_uncond = self._forward_non_standalone_all_model(x, t, null_cond_h, inp_shape=inp_shape)
                    v = v_uncond + cfg_scale * (v - v_uncond)

                # clip fake x1 to (-1, 1)
                # v = x0 - x1
                if clip_v:
                    x1_hat = x_init + v
                    x1_hat = x1_hat.clamp(-1, 1)
                    # back to clip velocity
                    v = x1_hat - x_init

                return v

            sample_fn_ = self.sampler.sample_ode(**sample_kwargs)
            recon = sample_fn_(x)

            if not ret_trajectory:
                recon = recon[-1]

            return recon
        else:
            # tim flow sampling, backbone + head fm.
            self.transport = cast(Tim_Transport, self.transport)
            null_cond_h = torch.zeros_like(h)
            recon = self.transport.sample(
                model=self._forward_non_standalone_all_model,
                z=x,
                y=h,
                y_null=null_cond_h,
                T_max=1.0,
                **sample_kwargs,
            )
            return recon

    def forward(
        self,
        x: torch.Tensor,  # bs, c, h, w - input noise latent
        h: Tensor,  # bs, n, dim or bs, c, gh, gw - semantic tokens condition
        inp_shape: Annotated[Union[torch.Size, tuple], "bs,c,h,w"],
        mode: Literal["train", "sample"] = "train",
        clamp: bool = False,
        ema_model: Optional[Self] = None,
        sample_kwargs: dict = {},
        ret_trajectory: bool = False,
    ):
        # Process semantic tokens condition
        h = latent_2d_to_1d(h)

        # Use flow decoder with processed tokens as condition
        # Flow decoder handles time internally and uses processed tokens as conditioning
        if mode == "train":
            train_fn_ = (
                self.flow_train_all if self.transformer_t_conditioned else self.flow_train_forward_head_standalone
            )
            recon, flow_loss = train_fn_(x, h, inp_shape=inp_shape, ema_model=ema_model, clamp=clamp)
        elif mode == "sample":
            if self.transformer_t_conditioned:
                recon = self.flow_sample_all(
                    x,
                    h,
                    inp_shape=inp_shape,
                    sample_kwargs=sample_kwargs,
                    clamp=clamp,
                    ret_trajectory=ret_trajectory,
                )
            else:
                recon = self.flow_sample_head_stand_alone(
                    h,
                    inp_shape=inp_shape,
                    sample_kwargs=sample_kwargs,
                    clamp=clamp,
                )
            flow_loss = None
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Return results in same format as DecoderUViT
        losses = {"flow_loss": flow_loss}
        return recon, losses

    def set_grad_checkpointing(self, enable: bool = True):
        """Enable or disable gradient checkpointing."""
        self.decoder.grad_checkpointing = enable

    def get_last_layer(self) -> Tensor:
        """Get last layer's weights for GAN training."""
        flow_decoder = (
            self.flow_decoder.flow_decoder if isinstance(self.flow_decoder, TimFlowDecoder) else self.flow_decoder
        )
        if flow_decoder.head.head_type.startswith("once"):
            return flow_decoder.head.unpatcher.weight
        return flow_decoder.head.unpatcher.unpatchers[-1].weight


class DecoderUViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.t_res = cfg.total_resolutions
        img_size = cfg.img_size
        grid_size = img_size // self.t_res
        self.patch_size = cfg.patch_size
        self.out_patch_size = 1

        self.decoder = UViTDecoder(
            # basic
            in_channels=cfg.in_chan,
            z_dim=cfg.z_dim,
            channels=cfg.channels,
            ch_mult=cfg.ch_mult,
            ctx_emb_dim=cfg.ctx_emb_dim,
            t_emb_mult=cfg.t_emb_mult,
            time_scale_shift=cfg.time_scale_shift,
            # transformer
            act_fn=cfg.act_fn,
            vit_act_fn=cfg.vit_act_fn,
            layers_per_block=cfg.layers_per_block,
            num_attention_heads=cfg.num_attention_heads,
            total_resolutions=cfg.total_resolutions,
            dropout=cfg.dropout,
            norm_num_groups=cfg.norm_num_groups,
            mid_nlayers=cfg.mid_nlayers,
            mid_theta=cfg.mid_theta,
            # ada norm
            eps=cfg.eps,
            ada_norm=cfg.ada_norm,
            learned_pos_embed=cfg.learned_pos_embed,
            # pe
            image_size=cfg.img_size,
            relative_pos_embed=cfg.relative_pos_embed,
            time_cond_type=cfg.time_cond_type,
            # train
            init=cfg.init,
            use_act_ckpt=cfg.use_act_ckpt,
        )

        self.cfg_prob = getattr(cfg, "cfg_prob", 0.0)
        # Tim transport and scheduler
        # TODO: add FM support
        self._flow_type = cfg.flow_type
        if cfg.flow_type == "tim":
            transition_schedule_kwargs = cfg.transition_schedule
            transport_kwargs = cfg.transport
            self.transport = OT_FM(**transport_kwargs)
            self.transition_schedule = TransitionSchedule(
                self.transport,
                **transition_schedule_kwargs,
            )
        elif cfg.flow_type == "fm":
            self.transport = FM_Transport(**cfg.transport)
            self.sampler = Sampler(transport=self.transport)
            logger.info(
                f"[Decoder UViT]: FM transport={cfg.transport}FM train time type={self.transport.time_sample_type}"
            )
        elif cfg.flow_type == "mf":  # meanflow
            self.transport = MeanFlow(
                # flow_ratio=0.75, time_dist=['lognorm', -0.4, 1.0],
                # drop conditions ratio, scale for one-step sample
                # cfg_ratio=0.2, cfg_scale=2.0
                **cfg.transport
            )
        else:
            raise ValueError(f"flow_type {cfg.flow_type} not supported")

    def _create_model_sampled(self, inp_shape):
        def _inner_tim(x, t, r, y, inp_shape):
            return self.decoder(x, t, r, y, inp_shape)

        def _inner_fm(x, t, y, inp_shape):
            return self.decoder(x, t, None, y, inp_shape)

        if self._flow_type == "tim":
            return partial(_inner_tim, inp_shape=inp_shape)
        elif self._flow_type == "fm":
            return partial(_inner_fm, inp_shape=inp_shape)
        else:
            raise ValueError(f"flow_type {self._flow_type} not supported")

    def _get_null_h(self, h: Tensor):
        if hasattr(self.decoder, "null_cond_h"):
            null_h = self.decoder.null_cond_h.contiguous()  # [1, dim, 1, 1]
            null_h = null_h.expand(h.shape[0], -1, *h.shape[-2:])
            return null_h
        else:
            return torch.zeros_like(h, memory_format=torch.contiguous_format)

    def _replace_h_as_null(self, h: Tensor, cfg_prob: float = 0.0):
        assert cfg_prob >= 0.0
        if cfg_prob == 0.0:
            # h as original, no cfg
            return h
        else:
            n_cfg = int(h.shape[0] * cfg_prob)
            cfg_mask = torch.rand(h.shape[0], device=h.device) < cfg_prob
            null_h = self._get_null_h(h) + 0.0 * h  # partial gradients flow back to encoder/semantic decoder
            h = torch.where(cfg_mask[:, None, None, None], null_h, h)

        return h

    def _forward_tim(
        self,
        x: torch.Tensor,  # bs, c, h, w
        h: Tensor,  # bs, n, dim or bs, c, h, w
        inp_shape: Annotated[Union[torch.Size, int, tuple], "bs,c,h,w or bs,c or c"],
        mode: Literal["train", "sample"] = "train",
        clamp=False,
        ema_model: Optional[Self] = None,
        sample_kwargs: dict = dict(
            num_steps=8,
            stochasticity_ratio=0.0,
            sample_type="transition",
            cfg_scale=1.0,
            progress_bar=True,
        ),
        ret_trajectory=False,
    ):
        # x is the input noise latent
        h = latent1d_to_2d(
            h,
            inp_shape=inp_shape,
            total_resolutions=self.t_res,
        )

        # Decoder channels
        chan = get_chan_from_shape(inp_shape)
        null_cond_h = self._get_null_h(h)

        # Decode using the CNN decoder
        flow_loss, loss_dict = None, {}
        recon: Tensor
        if mode == "train":
            if self.transition_schedule.transport.enhance_target:
                assert ema_model is not None, "EMA model is required for enhance_target"
                assert hasattr(self, "null_cond_h"), "The decoder must have null_cond_h for CFG"

            # training: h is condition, x is the input
            noise = torch.randn_like(x)
            flow_loss, _, loss_dict, breakdowns = self.transition_schedule(
                self.decoder,
                ema_model.decoder if ema_model is not None else None,
                self.decoder,
                batch_size=h.shape[0],
                x=x,
                z=noise,
                model_kwargs={"inp_shape": chan, "z": h},
                ema_kwargs={"inp_shape": chan, "z": h},
                null_kwargs={"inp_shape": chan, "z": null_cond_h},
                use_dir_loss=True,  # default as in dde paper
            )
            # back to x_0
            recon = breakdowns["x0_pred"]
        elif mode == "sample":
            # eval: loop to generate reconstruction image when h is the condition.
            # x_init = torch.randn_like(x)
            x_init = x
            recon = self.transition_schedule.sample(
                # self.decoder,
                self._create_model_sampled(chan),
                # start sampling noise
                z=x_init,
                # conditions
                y=h,
                y_null=null_cond_h,
                T_max=1.0,
                **sample_kwargs,
            )
            if not ret_trajectory:
                recon = recon[-1]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if clamp:
            recon = recon.clamp(-1, 1)

        # Returns
        losses = {"flow_loss": flow_loss}
        return recon, losses

    def _forward_fm(
        self,
        x: torch.Tensor,  # bs, c, h, w; x is the noise for mode='sample', is the clean img for mode='train'
        h: Tensor,  # bs, n, dim or bs, c, h, w
        inp_shape: Annotated[Union[torch.Size, int, tuple], "bs,c,h,w or bs,c or c"],
        mode: Literal["train", "sample"] = "train",
        clamp=False,
        ema_model: Optional[Self] = None,
        sample_kwargs: dict = dict(
            sampling_method="dopri5",
            num_steps=20,
            cfg=1.0,
            progress=True,
            cfg_interval=None,
            clip_velocity_per_step=False,
        ),
        ret_trajectory=False,
    ):
        self.transport = cast(FM_Transport, self.transport)
        if mode == "train":
            h = self._replace_h_as_null(h, cfg_prob=self.cfg_prob)
            flow_output = self.transport.training_losses(
                self.decoder,
                x,
                model_kwargs={"inp_shape": inp_shape, "z": h},
            )
            flow_loss = flow_output["loss"].mean()
            recon = flow_output["pred_x_clean"]
        else:
            # x_init = torch.randn_like(x)
            x_init = x.clone()
            cfg_scale: float = sample_kwargs.pop("cfg", 1.0)
            cfg_interval: tuple[float, float] | None = sample_kwargs.pop("cfg_interval", None)
            clip_v = sample_kwargs.pop("clip_velocity_per_step", False)
            sample_fn_ = self.sampler.sample_ode(**sample_kwargs)
            model_type = self.transport.model_type

            # takes [x, model, model_kwargs], model takes [x, t, **model_kwargs]
            _model_step_fn = create_sampling_step_fn(
                model=self.decoder,
                model_cond=h,
                cfg_scale=cfg_scale,
                cfg_interval=cfg_interval,
                clip_v=clip_v,
                inp_shape=inp_shape,
                model_type=model_type,
                get_null_cond=self._get_null_h,
                x_init=x_init,
            )

            # Sampling on model_type: x1 or velocity
            recon = sample_fn_(x_init, _model_step_fn)

            if not ret_trajectory:
                recon = recon[-1]
            flow_loss = None

        losses = {"flow_loss": flow_loss}
        return recon, losses

    def _forward_mf(
        self,
        x: torch.Tensor,  # bs, c, h, w; x is the noise for mode='sample', is the clean img for mode='train'
        h: Tensor,  # bs, n, dim or bs, c, h, w
        inp_shape: Annotated[Union[torch.Size, int, tuple], "bs,c,h,w or bs,c or c"],
        mode: Literal["train", "sample"] = "train",
        clamp=False,
        ema_model: Optional[Self] = None,
        sample_kwargs: dict = dict(
            sample_steps=5,
        ),
        ret_trajectory=False,
    ):
        """MeanFlow forward pass - supports both training and sampling modes."""
        # Convert h to 2D feature map if needed
        h = latent1d_to_2d(
            h,
            inp_shape=inp_shape,
            total_resolutions=self.t_res,
        )

        self.transport = cast(MeanFlow, self.transport)
        # Get channel dimension
        chan = get_chan_from_shape(inp_shape)

        # Create model function for MeanFlow - it expects (z, t, r, y=c)
        def model_fn(x: torch.Tensor, t: torch.Tensor, r: torch.Tensor, y: torch.Tensor = h) -> torch.Tensor:
            return self.decoder(x, t, r, y, inp_shape)

        flow_loss = None
        losses = {}

        if mode == "train":
            # Training mode: compute MeanFlow loss
            # MeanFlow expects clean image as input for training
            flow_loss, mse_val, recon = self.transport.loss(model_fn, x, null_c=self._get_null_h(h))
            flow_loss = flow_loss.mean()  # Ensure scalar loss
        elif mode == "sample":
            # Sampling mode: generate reconstruction from noise
            x_init = x  # Use provided noise as starting point

            # Extract sample parameters
            sample_steps = sample_kwargs.get("sample_steps", 2)

            # Perform sampling step by step
            trajectory = []

            t_vals = torch.linspace(1.0, 0.0, sample_steps + 1, device=x.device)

            for i in range(sample_steps):
                t = torch.full((x.size(0),), t_vals[i], device=x.device)
                r = torch.full((x.size(0),), t_vals[i + 1], device=x.device)

                # Get velocity from model
                v = model_fn(x, t, r, h)

                # Update x using Euler method
                t_ = t.view(-1, 1, 1, 1)
                r_ = r.view(-1, 1, 1, 1)
                x = x - (t_ - r_) * v

                if ret_trajectory:
                    trajectory.append(x.clone())
            recon = x
            if ret_trajectory:
                recon = torch.stack(trajectory, dim=0)  # [steps, bs, c, h, w]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if clamp:
            recon = recon.clamp(-1, 1)

        # Return results in consistent format
        losses = {"flow_loss": flow_loss}
        return recon, losses

    def forward(self, x, h, inp_shape, mode="train", *args, **kwargs):
        if self._flow_type == "tim":
            return self._forward_tim(x, h, inp_shape, mode, *args, **kwargs)
        elif self._flow_type == "mf":
            return self._forward_mf(x, h, inp_shape, mode, *args, **kwargs)
        else:
            return self._forward_fm(x, h, inp_shape, mode, *args, **kwargs)

    def init_weights(self) -> None:
        return


class HybridPixelDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        decoder_type = cfg.get("decoder_type", "hybrid")
        assert decoder_type in (
            "cnn_only",
            "transformer_only",
            "hybrid",
        ), f"Invalid decoder type {decoder_type}"

        self.decoder_type = decoder_type
        self.latent_dim = cfg.latent_dim
        res_cfg = cfg.get("res_decoder", None)
        transf_cfg = cfg.get("transformer_decoder", None)
        transf_kwargs = dict(**transf_cfg, projections={"input": None, "output": None}, is_causal=False, head="linear")

        if decoder_type == "hybrid":
            assert res_cfg is not None, "res_decoder config must be provided if use_cnn"
            self.res_decoder = ResDecoder(**res_cfg)
            self.transformer_decoder = TransformerTokenizer(**transf_kwargs)
            self.latent_to_z = nn.Conv2d(cfg.latent_dim, cfg.z_dim, kernel_size=1)
        elif decoder_type == "cnn_only":
            assert res_cfg is not None, "res_decoder config must be provided if use_cnn"
            self.res_decoder = ResDecoder(**res_cfg)
            self.latent_to_z = nn.Conv2d(cfg.latent_dim, cfg.z_dim, kernel_size=3, padding=1)
        else:  # transformer only
            assert transf_cfg is not None, "transformer_decder config must be provided"
            self.transformer_decoder = TransformerTokenizer(**transf_kwargs)
            self.latent_to_z = nn.Conv2d(cfg.latent_dim, cfg.z_dim, kernel_size=1)

    def init_weights(self):
        if hasattr(self, "transformer_decoder"):
            self.transformer_decoder.init_weights()
        if hasattr(self, "res_decoder"):
            self.res_decoder.init_weights()
        nn.init.trunc_normal_(self.latent_to_z.weight, std=0.01, a=-0.4, b=0.4)
        nn.init.zeros_(self.latent_to_z.bias)

    def forward(
        self, latent, inp_shape: torch.Size | tuple, get_intermidates: list[int] | None = None, **_ignored_kwargs
    ):
        """Decode the latent to the image patches"""
        z = self.latent_to_z(latent)
        z = _to_memformat_channels_last(z)
        if self.decoder_type == "hybrid":
            # Hybrid encoder, takes the transformer encoder's intermidates
            out = self.transformer_decoder(
                z,
                ret_2d_tokens=True,  # Return 2d tokens
                ret_all=True,
                get_intermidates=get_intermidates,
            )
            x = out["head_out"]  # z is 2d: (b, c, h, w)
            x = self.res_decoder(x, inp_shape[1])
            x = _to_memformat_channels_last(x)

        elif self.decoder_type == "cnn_only":
            x, intermidates = self.res_decoder(z, inp_shape[1], ret_interm_feats=get_intermidates)
            x = _to_memformat_channels_last(x)
            out = {"intermidates": intermidates}

        else:  # transformer only
            out = self.transformer_decoder(
                z, ret_2d_tokens=True, ret_all=True, get_intermidates=get_intermidates, out_shape=inp_shape
            )
            x = out["head_out"]  # z is 2d: (b, c, h, w)

        return x

    def get_last_layer(self):
        if self.decoder_type in ("hybrid", "cnn_only"):
            return self.res_decoder.get_last_layer()
        elif self.decoder_type == "transformer_only":
            return self.transformer_decoder.get_last_layer()
        else:
            raise ValueError(f"Unknown decoder type: {self.decoder_type}")


class MingtokRSModel(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        # Cfgs
        self.low_cfg = cfg.low_level_encoder
        self.sem_cfg = cfg.semantic_decoder
        self.pixel_cfg = cfg.pixel_decoder
        self.tok_cfg = cfg.tokenizer
        self.sem_pix_decoder_type = self.tok_cfg.sem_pix_decoder_type
        self.sem_decoder_take = self.tok_cfg.sem_decoder_take
        assert self.sem_pix_decoder_type in ("unified", "seperated")
        assert self.sem_decoder_take in ("z", "h")
        logger.info(f"Sem-pix decoder type: <green>{self.sem_pix_decoder_type}</>")
        logger.info(f"Sem-decoder take: <green>{self.sem_decoder_take}</>")

        # Model parts
        logger.info(f"Init low-level encoder.")
        self.low_level_encoder: nn.Module = EncoderLowLevel(self.low_cfg)

        logger.info(f"Init semantic decoder.")
        self.semantic_decoder: nn.Module = DecoderSemantic(self.sem_cfg)
        self.pretrained_task: list[str] = getattr(self.tok_cfg, "pretrained_task")

        logger.info(f"Init pixel decoder of type {self.pixel_cfg.decoder_type}.")
        decoder_cls: type[nn.Module] = {
            "flow_head": DecoderFlowHead,
            "uvit": DecoderUViT,
            "hybrid": HybridPixelDecoder,
        }[self.pixel_cfg.decoder_type]
        self._is_flow_matching_pix_decoder = self.pixel_cfg.decoder_type in ("flow_head", "uvit")
        self.pixel_decoder: nn.Module = decoder_cls(self.pixel_cfg)

        # FIXME: shortcuts for latent, still has bugs.
        self._build_st_cat_lin()

        # Attributions
        self.total_resolutions: int = self.pixel_cfg.total_resolutions
        self.sampling_options_default: dict[str, Any] = getattr(self.tok_cfg, "sampling_options_default", {})
        if self.sampling_options_default:
            logger.info(f"Set the sampling options to {self.sampling_options_default}.")

        # TODO: add quantizer
        self._setup_quantizer()
        self._setup_latent_aug()

        # Proxy task modules
        self._setup_pretrained_proxy_module()

        # Semantic and low-level caches
        self.proj_cfg = cfg.repa_proj
        self.use_repa: bool = cfg.repa_proj is not None

        self.z: Tensor | None = None
        self.sem_z: Tensor | None = None
        self._hw: tuple | list | None = None

        # Low-level and semantic projections for feature distillation
        self.low_lvl_proj_is_multi = False
        self.sem_proj_is_multi = False
        if self.use_repa:
            logger.info(f"Feature distillation with low-level and semantic network features")
            self.low_lvl_cache_layers = [int(x) for x in list(self.proj_cfg.low_lvl_cache_layers)]
            self.sem_cache_layers = [int(x) for x in list(self.proj_cfg.sem_cache_layers)]
            self.low_lvl_proj_is_multi = is_tuple_list(self.low_lvl_cache_layers)
            self.sem_proj_is_multi = is_tuple_list(self.sem_cache_layers)
            self._maybe_fix_repa_proj_chans()
            self._build_repa_projections()

        # trainer compatiblity
        self._use_repa_loss = self.use_repa
        self._use_vf_loss = False

        # Initialize the weights
        self.init_weights()

        # TODO: need test, compile the modules
        self._compile_modules()

    def _setup_pretrained_proxy_module(self):
        if "lejepa" in self.pretrained_task:
            from torchvision.ops import MLP as MLP_TV

            lejepa_proj_dim = 512
            self.lejepa_proj = MLP_TV(self.sem_cfg.out_chan, [2048, 2048, lejepa_proj_dim], norm_layer=nn.BatchNorm1d)
            logger.info("Setup the LeJEPA projector")

    def _maybe_fix_repa_proj_chans(self) -> None:
        """对齐`repa_proj.*_repa_proj_chans`与实际cache feature通道数，避免投影层输入维度不匹配。"""
        if not self.use_repa:
            return

        inferred_low = self._infer_low_lvl_repa_proj_chans()
        cfg_low = [int(x) for x in list(getattr(self.proj_cfg, "low_lvl_repa_proj_chans", []))]
        if len(cfg_low) != len(inferred_low) or any(a != b for a, b in zip(cfg_low, inferred_low)):
            logger.warning(
                f"[Mingtok][REPA]: low_lvl_repa_proj_chans与实际feature不匹配，已自动修正：{cfg_low} -> {inferred_low}"
            )
            self.proj_cfg.low_lvl_repa_proj_chans = inferred_low

    def _infer_low_lvl_repa_proj_chans(self) -> list[int]:
        enc_type = str(getattr(self.low_cfg, "encoder_type", "cnn_only"))
        if enc_type not in ("cnn_only", "hybrid"):
            return [int(x) for x in list(getattr(self.proj_cfg, "low_lvl_repa_proj_chans", []))]

        res_cfg = getattr(self.low_cfg, "res_encoder", None)
        if res_cfg is None:
            return [int(x) for x in list(getattr(self.proj_cfg, "low_lvl_repa_proj_chans", []))]

        base = int(res_cfg.channels)
        mults = [int(x) for x in list(res_cfg.channels_mult)]
        inferred: list[int] = []
        for layer_idx in self.low_lvl_cache_layers:
            if layer_idx == -1:
                inferred.append(base * mults[-1])
            else:
                inferred.append(base * mults[int(layer_idx)])
        return inferred

    def _setup_quantizer(self) -> None:
        latent_channels = int(self.low_cfg.latent_dim)
        self.latent_channels: int = latent_channels

        self.quantizer_type: str | None = getattr(self.tok_cfg, "quantizer_type", None)
        self.random_quant: float = float(getattr(self.tok_cfg, "random_quant", 0.0))

        fsq_num_codebooks = int(getattr(self.tok_cfg, "fsq_num_codebooks", 6))
        fsq_levels = list(getattr(self.tok_cfg, "fsq_levels", [8, 8, 8, 5, 5, 5]))
        mbsq_codebook_size = int(getattr(self.tok_cfg, "mbsq_codebook_size", 1024))
        mbsq_schedule_mode = str(getattr(self.tok_cfg, "mbsq_schedule_mode", "original"))
        mleech_codebook_size = int(getattr(self.tok_cfg, "mleech_codebook_size", 196560))
        mleech_leech_type = str(getattr(self.tok_cfg, "mleech_leech_type", "full"))
        mleech_schedule_mode = str(getattr(self.tok_cfg, "mleech_schedule_mode", "original"))

        q_in_chan = latent_channels
        if self.quantizer_type == "kl":
            q_in_chan = latent_channels * 2
        elif self.quantizer_type == "psd":
            q_in_chan = latent_channels + 1

        if self.quantizer_type is None:
            self.quantizer = None
            logger.info("[Mingtok]: use no quantizer (continuous latent).")
            return

        if self.latent_channels != q_in_chan:
            self.pre_quant_conv = nn.Conv2d(latent_channels, q_in_chan, kernel_size=1)
            nn.init.trunc_normal_(self.pre_quant_conv.weight, std=0.01, a=-0.4, b=0.4)
            nn.init.zeros_(self.pre_quant_conv.bias)
        else:
            self.pre_quant_conv = nn.Identity()

        match self.quantizer_type:
            case "kl":
                self.quantizer = DiagonalGaussianDistribution  # type: ignore[assignment]
            case "bsq":
                if latent_channels % 2 != 0:
                    raise ValueError("BSQ要求latent_channels为偶数。")
                self.quantizer = BSQ(
                    embed_dim=latent_channels,
                    beta=0.0,
                    gamma0=1.0,
                    gamma=1.0,
                    zeta=1.0,
                    inv_temperature=1.0,
                    cb_entropy_compute="group",
                    l2_norm=True,
                    input_format="bchw",
                    persample_entropy_compute="analytical",
                    group_size=1,
                )
            case "fsq":
                self.quantizer = FSQ(
                    levels=fsq_levels,
                    dim=latent_channels,
                    num_codebooks=fsq_num_codebooks,
                    channel_first=True,
                )
            case "multiscale_bsq":
                self.quantizer = MultiScaleBSQ(
                    dim=latent_channels,
                    codebook_size=mbsq_codebook_size,
                    schedule_mode=mbsq_schedule_mode,
                    new_quant=True,
                )
            case "multiscale_leechq":
                self.quantizer = MultiScaleLeechQ(
                    dim=latent_channels,
                    codebook_size=mleech_codebook_size,
                    leech_type=mleech_leech_type,
                    schedule_mode=mleech_schedule_mode,
                )
            case "psd":
                self.quantizer = PowerSphericalDistribution  # type: ignore[assignment]
            case _:
                raise ValueError(f"Unsupported quantizer_type={self.quantizer_type!r}")

        logger.info(f"[Mingtok]: using quantizer={self.quantizer_type}.")

    def _setup_latent_aug(self) -> None:
        self.use_latent_mask: bool = bool(getattr(self.tok_cfg, "use_latent_mask", False))
        latent_mask_cfg_raw = getattr(self.tok_cfg, "latent_mask_config", None)
        latent_mask_cfg = LatentMaskConfig()
        if latent_mask_cfg_raw is not None:
            latent_mask_cfg_dict = dict(latent_mask_cfg_raw)
            latent_mask_cfg = LatentMaskConfig(**latent_mask_cfg_dict)
        self.latent_mask_cfg: dict[str, Any] = asdict(latent_mask_cfg)

        if self.use_latent_mask:
            self.mask_token = nn.Parameter(torch.zeros(1, self.latent_channels, 1, 1))
            nn.init.normal_(self.mask_token, std=0.02)
            logger.info(f"[Mingtok]: enable latent mask augmentation with cfg={self.latent_mask_cfg}.")
        else:
            self.mask_token = None

    def _use_quantizer(self, use_quantizer: bool | None = None) -> bool:
        if self.quantizer_type is None:
            return False
        if self.training and self.random_quant > 0.0:
            return self.random_quant > random.random()
        if use_quantizer is None:
            return True
        return use_quantizer

    def _apply_quantizer(self, h: Tensor) -> QuantizerOutput:
        if self.quantizer_type is None or self.quantizer is None:
            raise RuntimeError("quantizer未初始化，但调用了_apply_quantizer。")

        h_dtype = h.dtype
        h = h.float()

        if self.quantizer_type == "bsq":
            h = nn.functional.normalize(h, dim=1)
            hq, bsq_loss, loss_breakdown = self.quantizer(h)  # type: ignore[misc]
            return hq.to(h_dtype), bsq_loss, cast(dict[str, Any], loss_breakdown)

        if self.quantizer_type == "kl":
            m_, logvar_ = h.chunk(2, dim=1)
            posterior = self.quantizer((m_, logvar_))  # type: ignore[operator]
            kl_loss = posterior.kl()
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            hq = posterior.sample()
            return hq.to(h_dtype), kl_loss, {"posterior": posterior, "mean": m_, "logvar": logvar_}

        if self.quantizer_type == "fsq":
            hq, indices = self.quantizer(h)  # type: ignore[misc]
            fsq_loss = torch.tensor(0.0, device=h.device, dtype=h_dtype)
            return hq.to(h_dtype), fsq_loss, {"indices": indices}

        if self.quantizer_type in ("multiscale_bsq", "multiscale_leechq"):
            (
                hq,
                all_indices,
                all_bit_indices,
                residual_norm_per_scale,
                all_losses,
                var_inputs,
                all_entropies,
            ) = self.quantizer(h)  # type: ignore[misc]
            q_loss = all_losses.sum()
            q_info: dict[str, Any] = {
                "all_indices": all_indices,
                "all_bit_indices": all_bit_indices,
                "residual_norm_per_scale": residual_norm_per_scale,
                "all_losses": all_losses,
                "var_inputs": var_inputs,
                "all_entropies": all_entropies,
            }
            return hq.to(h_dtype), q_loss, q_info

        if self.quantizer_type == "psd":
            mu = h[:, :-1]
            kappa = h[:, -1]
            kappa = nn.functional.softplus(kappa) + 1.0
            dist = self.quantizer(mu, kappa, dim=1)  # type: ignore[operator]
            loss = dist.kl_to_uniform()
            hq = dist.rsample()
            hq = hq * (self.latent_channels**0.5)
            psd_loss = loss.mean()
            return hq.to(h_dtype), psd_loss, {"kl_loss": psd_loss}

        raise RuntimeError(f"Unsupported quantizer_type={self.quantizer_type!r}")

    def apply_quantizer(self, h: Tensor, use_quantizer: bool | None = None) -> Tensor | QuantizerOutput:
        if not self._use_quantizer(use_quantizer):
            return h
        return self._apply_quantizer(h)

    def latent_aug(self, h: Tensor) -> Tensor:
        if (not self.training) or (not self.use_latent_mask):
            return h

        lmr_res = lmr_apply(h, **self.latent_mask_cfg)
        if not isinstance(lmr_res, tuple):
            return lmr_res

        h_masked, mask = lmr_res
        if self.mask_token is None:
            return h_masked

        mask_token = self.mask_token.expand(h.shape[0], -1, h.shape[2], h.shape[3])
        mask = mask.expand_as(h_masked)
        return torch.where(mask, mask_token, h_masked)

    def _compile_modules(self):
        # compile model parts
        self._compile = getattr(self.pixel_cfg, "compile", False)
        if self._compile:
            logger.info("Compiling FlowTokenizer model parts...")
            # Only compile the heavy parts - transformer
            self.low_level_encoder.encoder = torch.compile(  # type: ignore
                self.low_level_encoder.encoder
            )
            self.semantic_decoder.encoder = torch.compile(self.semantic_decoder.encoder)
            self.pixel_decoder.decoder = torch.compile(self.pixel_decoder.decoder)  # type: ignore
            logger.info("Compilation done.")

    def _build_st_cat_lin(self):
        # Straight through latent functionality
        self.st_skip_semantic_decoder = False
        if getattr(self.tok_cfg, "straight_through_latent", False) and self.sem_pix_decoder_type == "unified":
            self.st_skip_semantic_decoder = True
            # Create skip connection conv for latent
            latent_channels = self.low_cfg.latent_dim  # channels from low-level encoder
            semantic_channels = self.sem_cfg.out_chan  # channels from semantic decoder
            cat_dim = latent_channels + semantic_channels
            norm_type = self.sem_cfg.norm_layer

            self.st_cat_lin = nn.Sequential(
                create_norm_act_layer(norm_type, cat_dim, "gelu"),
                nn.Linear(cat_dim, semantic_channels),
            )
            logger.log(
                "NOTE",
                f"Will skip the semantic decoder latent through cat conv, "
                f"latent_channels={latent_channels}, semantic_channels={semantic_channels}",
            )

    def _build_repa_projections(self):
        if self.use_repa:
            logger.info(f"Build the representation alignment projections.")

            # Hier distillation
            if self.low_lvl_proj_is_multi:
                low_lvl_z_proj = nn.ModuleList()
                for i in range(len(self.low_lvl_cache_layers)):
                    proj_ = build_mlp(
                        self.proj_cfg.low_lvl_repa_proj_chans[i],
                        self.proj_cfg.low_lvl_repa_out_chan,
                        self.proj_cfg.low_lvl_repa_out_chan,
                        proj_type="norm_first_force_conv",
                    )
                    low_lvl_z_proj.append(proj_)
            else:
                low_lvl_z_proj = build_mlp(
                    self.proj_cfg.low_lvl_repa_proj_chans,
                    self.proj_cfg.low_lvl_repa_out_chan,
                    self.proj_cfg.low_lvl_repa_out_chan,
                    proj_type="norm_first_force_conv",
                )

            if self.sem_proj_is_multi:
                sem_z_proj = nn.ModuleList()
                for i in range(len(self.sem_cache_layers)):
                    proj_ = build_mlp(
                        self.proj_cfg.sem_repa_proj_chans[i],
                        self.proj_cfg.sem_repa_out_chan,
                        self.proj_cfg.sem_repa_out_chan,
                        proj_type="norm_first_force_conv",
                    )
                    sem_z_proj.append(proj_)
            else:
                sem_z_proj = build_mlp(
                    self.proj_cfg.sem_repa_proj_chans,
                    self.proj_cfg.sem_repa_out_chan,
                    self.proj_cfg.sem_repa_out_chan,
                    proj_type="norm_first_force_conv",
                )

            self._repa_proj = nn.ModuleDict(
                {
                    "low_lvl_repa_proj": low_lvl_z_proj,
                    "sem_repa_proj": sem_z_proj,
                }
            )

    @staticmethod
    def _to_2d(x, hw: List[int]) -> None | Tensor:
        if x is None or x.ndim == 4:
            return x
        else:
            return rearrange(x, "b (h w) ... -> b ... h w", h=hw[0], w=hw[1])

    def _encode_latent(self, x):
        low_lvl_out = self.low_level_encoder(
            x,
            get_intermidates=self.low_lvl_cache_layers if self.training and self.use_repa else None,
        )
        # Cache low level features
        self.z = low_lvl_out.get("intermidates") if self.low_lvl_proj_is_multi else low_lvl_out["latent"]
        return low_lvl_out

    def _sem_decode(self, z, masks=None):
        """Decode the input (pre-down-conv latent or latent) into semantic tokens."""
        sem_out = self.semantic_decoder(
            z,
            masks,
            get_intermidates=self.sem_cache_layers if (self.training and self.use_repa) else None,
        )
        return sem_out

    @no_type_check
    @torch.autocast("cuda", dtype=torch.bfloat16)
    def get_repa_feature(self):
        if not self.training or not self.use_repa:
            return None

        assert self.z is not None and self.sem_z is not None, (
            f"z and sem_z must be set before get_repa_feature but {self.z=} and {self.sem_z=}"
        )

        # Low-level feature distillation
        # norm_first_force_conv projector outputs 2d features directly
        if self.low_lvl_proj_is_multi:
            low_lvl_features = [
                self._repa_proj["low_lvl_repa_proj"][i](feat, hw=self._hw) for i, feat in enumerate(self.z)
            ]
        else:
            low_lvl_features = self._repa_proj["low_lvl_repa_proj"](self.z, hw=self._hw)

        # Semantic feature distillation
        if self.sem_proj_is_multi:
            sem_features = [self._repa_proj["sem_repa_proj"][i](feat, hw=self._hw) for i, feat in enumerate(self.sem_z)]
        else:
            sem_features = self._repa_proj["sem_repa_proj"](self.sem_z, hw=self._hw)

        return low_lvl_features, sem_features

    def _decode_to_sem(
        self,
        latent=None,
        z=None,
        masks: list[Tensor] | Tensor | None = None,
        hw: Sequence | None = None,
        no_cache: bool = False,
    ):
        if latent is None or z is None:
            raise ValueError("Both latent and z must be provided")

        # To semantic tokens
        # Use z for semantic decoder
        if z.ndim == 4:
            # Flatten to 1D sequence if necessary is handled by semantic decoder
            pass

        sem_taken = latent if self.sem_decoder_take == "h" else z
        assert torch.is_tensor(sem_taken), f"Sematic decoder takes only Tensor but got {type(sem_taken)}"
        sem_out = self._sem_decode(sem_taken, masks)
        sem_tokens = sem_out["sem_tokens"]

        # if self.st_skip_semantic_decoder:
        #     assert latent is not None, "latent must be provided for skip connection"
        #     assert latent.shape[1] == sem_tokens.shape[1], f"Mismatched shape: {latent.shape=}, {sem_tokens.shape=}"
        #     lat_sem_tokens = torch.cat([latent, sem_tokens], dim=-1)
        #     sem_tokens = self.st_cat_lin(lat_sem_tokens)  # input into the cnn decoder

        # To 2d if masks is None
        # if sem_tokens.ndim == 3 and masks is None:
        #     assert hw is not None, "hw must be provided"
        #     assert math.prod(hw) == sem_tokens.shape[1], (
        #         f"hw {hw} product does not match latent shape {sem_tokens.shape} and total_resolutions={self.total_resolutions}"
        #     )
        #     # for cache into self.sem_z
        #     sem_tokens = self._to_2d(sem_tokens, hw)

        # Cache semantic features
        if not no_cache:
            self.sem_z = sem_out.get("intermidates") if self.sem_proj_is_multi else sem_tokens

        out = dict(latent=latent, sem_tokens=sem_tokens, z=z)
        return out

    def encode(self, x: Tensor, use_quantizer: bool | None = None) -> dict[str, Any]:
        out = dict(self._encode_latent(x))
        latent = cast(Tensor, out["latent"])

        h_pre_quant = self.pre_quant_conv(latent) if hasattr(self, "pre_quant_conv") else latent
        h_pre_quant = _to_memformat_channels_last(h_pre_quant)
        maybe_q = self.apply_quantizer(h_pre_quant, use_quantizer=use_quantizer)

        q_info: dict[str, Any] | None = None
        if isinstance(maybe_q, tuple):
            latent_q, q_loss, q_info = maybe_q
        else:
            latent_q = maybe_q
            q_loss = torch.tensor(0.0, device=latent.device, dtype=latent.dtype)

        latent_q = self.latent_aug(latent_q)
        out["latent"] = latent_q
        out["q_loss"] = q_loss
        out["q_info"] = q_info
        return out

    def decode(
        self,
        latent: torch.Tensor,
        inp_shape: Annotated[Union[torch.Size, tuple], "bs,c,h,w"],
        x: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
        mode: Literal["train", "sample"] = "train",
        clamp=False,
        ema_model: Optional[Self] = None,
        sample_kwargs: dict | None = None,
        ret_trajectory=False,
    ) -> dict:
        """
        Low level encoded latent is decoded into semantic tokens, and then decoded into the reconstructed image or
        velocity. Output dict of losses, recon, semantic tokens and projected semantic tokens.

        If is pixel decoder is generative diffusion heads, sample_kwargs should be set as a dict:
            # tim sample kwargs
            num_steps=8,
            stochasticity_ratio=0.0,
            sample_type="transition",
            cfg_scale=1.0,
            # fm sample kwargs
            sampling_method="Euler",
            diffusion_form="SBDM",
            diffusion_norm=1.0,
            last_step="Mean",
            last_step_size=0.04,
            num_steps=250,
            temperature=1.0,
            # manually sampling kwargs
            sample_steps=10,
            schedule='linear',
            cfg=2.0,
            cfg_interval=None,
            tbar=True
        """

        # Decode to semantic tokens
        hw = (
            torch.tensor(inp_shape[-2:])  # type: ignore
            // self.total_resolutions
            // self.semantic_decoder.patch_size
            * self.semantic_decoder.out_patch_size
        ).tolist()
        self._hw = hw

        if self.sem_pix_decoder_type == "unified":
            sem_decoder_out = self._decode_to_sem(latent=latent, z=z, hw=hw)
            h = sem_tokens = sem_decoder_out["sem_tokens"]  # as conditions if is diffusion decoder
        else:
            # is seperated sematic decoder, do not involved with decoding.
            if self._use_repa_loss:
                sem_decoder_out = self._decode_to_sem(latent=latent, z=z, hw=hw)
                sem_tokens = sem_decoder_out["sem_tokens"]
            else:
                sem_tokens = None
            h = latent

        if self._is_flow_matching_pix_decoder:
            # To flow UViT or head
            sample_kwargs = self.sampling_options_default | (sample_kwargs or {})
            # HW and EMA
            ema_decoder = ema_model.pixel_decoder if ema_model is not None else None

            if mode == "sample":
                # init the x as the noise
                logger.trace(f"[flow tokenizer]: init the x with gaussian - mode = {mode}")
                x_inp = torch.randn_like(x)
            elif mode == "train":
                x_inp = x
            else:
                raise ValueError(f"{mode=} is invalid")

            # Flow decoder or deterministic decoder
            out = self.pixel_decoder(
                x=x_inp,
                h=h,
                inp_shape=inp_shape,
                mode=mode,
                clamp=clamp,
                ema_model=ema_decoder,
                sample_kwargs=sample_kwargs,
                ret_trajectory=ret_trajectory,
            )

            # Form the output
            recon, loss_dict = out
            output = dict(recon=recon, losses=loss_dict, sem_tokens=sem_tokens)
        else:
            # is deterministic decoder
            recon = self.pixel_decoder(
                x=x,
                latent=h,
                inp_shape=inp_shape,
                mode=mode,
                clamp=clamp,
                ema_model=ema_model,
                sample_kwargs=sample_kwargs,
                ret_trajectory=ret_trajectory,
            )
            output = dict(recon=recon, losses=None, sem_tokens=sem_tokens)
        return output

    def forward(
        self,
        input: torch.Tensor,
        *,
        dec_mode: Literal["train", "sample"] = "train",
        clamp: bool = False,
        ema_model: Optional[Self] = None,
        sample_kwargs: dict = {},
        ret_trajectory: bool = False,
    ):
        if input.shape[1] > 512:
            logger.error(f"input shape {input.shape} is too large.")
            raise ValueError("input shape is too large.")

        enc_out = self.encode(input)
        dec_out = self.decode(
            enc_out["latent"],
            input.shape,
            input,
            enc_out["z"],
            dec_mode,
            clamp,
            ema_model,
            sample_kwargs,
            ret_trajectory,
        )

        # Compatible with trainer
        flow_loss = dec_out["losses"]["flow_loss"] if self._is_flow_matching_pix_decoder else None
        recon = dec_out["recon"]
        return {"recon": recon, "flow_loss": flow_loss, **enc_out}

    def encode_ijepa(self, x, jepa_masks: list[Tensor] | None = None) -> Tensor:
        hw = (
            torch.tensor(x.shape[-2:])  # type:ignore
            // self.total_resolutions
            * self.semantic_decoder.out_patch_size
            // self.semantic_decoder.patch_size
        )
        enc_out = self.encode(x)
        latent = enc_out["latent"]
        z = enc_out["z"]

        sem_decoder_out = self._decode_to_sem(latent=latent, z=z, masks=jepa_masks, hw=hw.tolist(), no_cache=True)
        sem_tokens = sem_decoder_out["sem_tokens"]
        return sem_tokens

    def encode_lejepa(self, x, *, lejepa_on: str = "cls") -> dict[str, Tensor]:
        low_lvl_out = self.low_level_encoder(x, get_intermidates=None)
        z = low_lvl_out["z"]
        sem_out = self.semantic_decoder(z, masks=None, get_intermidates=None)
        cls_tokens = sem_out.get("cls_tokens")  # b,c
        if cls_tokens is None:
            raise ValueError("semantic_decode is initialized without class token.")
        cls_t1d = cls_tokens.squeeze(1)
        patch_t1d = rearrange(sem_out["sem_tokens"], "b l c -> (b l) c")  # for full-patch-tokens sigreg
        # project the tokens
        cls_t1d = self.lejepa_proj(cls_t1d)
        patch_t1d = self.lejepa_proj(patch_t1d)
        return {"cls_tokens": cls_t1d, "patch_tokens": patch_t1d}

    def encode_dino_cls(self, x: torch.Tensor) -> Tensor:
        low_lvl_out = self.low_level_encoder(x, get_intermidates=None)
        z = low_lvl_out["z"]
        sem_out = self.semantic_decoder(z, masks=None, get_intermidates=None)
        cls_tokens = sem_out.get("cls_tokens")
        if cls_tokens is None:
            raise ValueError("semantic_decode is initialized without class token.")
        return cls_tokens.squeeze(1)

    def load_pretrained(self, path: str):
        import accelerate

        sd = accelerate.utils.load_state_dict(path)
        missing_keys, unexpected_keys = load_weights_with_shape_check(self, sd)
        if len(missing_keys) > 0:
            logger.warning(f"Missing keys when loading pretrained model: {missing_keys}")
        if len(unexpected_keys) > 0:
            logger.warning(f"Unexpected keys when loading pretrained model: {unexpected_keys}")
        return missing_keys, unexpected_keys

    def set_grad_checkpointing(self, enable: bool = True):
        for module in [
            self.low_level_encoder,
            self.semantic_decoder,
            self.pixel_decoder,
        ]:
            for m in module.modules():
                if hasattr(m, "set_grad_checkpointing"):
                    m.set_grad_checkpointing(enable)  # type: ignore
                    logger.log(
                        "NOTE",
                        f"Set grad_checkpointing={enable} for {m.__class__.__name__}",
                    )

    @classmethod
    @function_config_to_basic_types
    def create_model(cls, **overrides):
        """Create a FlowTokenizer model from configuration."""
        # Update the defaults
        cfg = create_default_mingtok_pretrained_cfg()
        if overrides is not None:
            cfg = OmegaConf.merge(cfg, overrides)
        model = cls(cfg)
        return model

    def get_last_layer(self) -> Tensor:
        """Get last layer's weights for GAN training."""
        get_last_layer = getattr(self.pixel_decoder, "get_last_layer", None)
        get_last_layer = cast(Callable[[], Tensor], get_last_layer)
        if callable(get_last_layer):
            return get_last_layer()
        return getattr(self.pixel_decoder, "decoder").get_last_layer()

    def get_last_enc_layer(self):
        return self.low_level_encoder.get_last_layer()

    def init_weights(self):
        self.low_level_encoder.init_weights()
        self.semantic_decoder.init_weights()
        self.pixel_decoder.init_weights()
        logger.info("<green>[Mingtok]: init weights done.</green>")


# * --- Test --- #


def test_flow_tokenizer():
    from torchmetrics.image import PeakSignalNoiseRatio

    from src.data.litdata_hyperloader import ImageStreamingDataset, StreamingDataLoader

    tokenizer_cfg_default = create_default_mingtok_cfg("fm", "uvit")
    tokenizer = MingtokRSModel(tokenizer_cfg_default).to("cuda", torch.bfloat16)
    tokenizer.set_grad_checkpointing()

    from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count_table

    # print(flop_count_table(FlopCountAnalysis(tokenizer, x)))
    print(parameter_count_table(tokenizer))

    # path = "runs/stage1_mingtok/2025-11-09_21-19-05_mingtok_600M_continous/ema/tokenizer/model.safetensors"
    # tokenizer.load_pretrained(path)
    # logger.info("Load pretrained done.")

    # x = torch.randn(9, 32, 512, 512).to("cuda", torch.bfloat16)
    dataset = ImageStreamingDataset(
        # input_dir="data/MDAS-HySpex/LitData_hyper_images",
        # resize_before_transform=128,
        # input_dir="data/HyperGlobal/LitData_hyper_images_GF5",
        # resize_before_transform=256,
        input_dir="data2/RemoteSAM270k/LitData_hyper_images",
        resize_before_transform=256,
        force_to_rgb=True,
        to_neg_1_1=True,
        # input_dir="data/Houston/LitData_hyper_images",
        # resize_before_transform=512,
        # is_hwc=False,
    )
    dl = StreamingDataLoader(dataset, batch_size=4, num_workers=16, shuffle=True)

    # x = dataset[1923]["img"]
    # x = x[None].type(torch.bfloat16).to("cuda")
    # x = x[:, :129]

    # x = next(iter(dl))["img"].type(torch.bfloat16).to("cuda")

    # for i in range(10):
    #     x = dataset[i]["img"]
    #     x = x[None].type(torch.bfloat16).to("cuda")

    # print(f"Input shape: {x.shape}")
    # print("Encoding and decoding (train mode)")
    # with torch.autocast("cuda", dtype=torch.bfloat16):
    #     with torch.no_grad():
    #         out = tokenizer(x, dec_mode="sample")

    #     psnr = PeakSignalNoiseRatio(1.0).cuda()
    #     x = (x + 1) / 2
    #     recon = (out[0] + 1) / 2
    #     print(f"PSNR: {psnr(x, recon)}")

    # Sample
    # print("Sampling")
    # with torch.no_grad():
    #     tokenizer.eval()
    #     with torch.autocast("cuda", dtype=torch.bfloat16):
    #         out = tokenizer(
    #             x, dec_mode="sample", clamp=True, ema_model=None, ret_trajectory=False
    #         )
    #     recon = out[0]
    #     print(
    #         f"Sampled recon shape: {recon.shape}, min: {recon.min()}, max: {recon.max()}"
    #     )  # type: ignore

    # Backward
    # optimizer = torch.optim.AdamW(tokenizer.parameters(), lr=4e-3, weight_decay=0.01)
    from src.utilities.optim import MuonFSDP
    from src.utilities.optim.muon import Muon

    # muonp, adamp = Muon.clear_muon_adamw_params(
    #     tokenizer.named_parameters(),
    # )
    # optimizer = Muon(lr=1e-3, muon_params=muonp, adamw_params=adamp)

    optimizer = MuonFSDP.create_muon_optimizer(
        tokenizer.named_parameters(),
        muon_params_defaults={"lr": 1e-3},
        oned_params_defaults={"lr": 3e-4},
        betas=(0.9, 0.95),
        nesterov=True,
        use_triton=False,
    )

    tokenizer.train()
    while True:
        for sample in dl:
            x = sample["img"].type(torch.bfloat16).to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = tokenizer(x)
            # tokenizer.get_repa_feature()  # to set repa features
            loss = out[1]["flow_loss"]
            print(f"Loss: {loss}")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), 1.0)
            optimizer.step()

        # print("Backward successful.")
        # Check the gradients
        # for n, p in tokenizer.named_parameters():
        #     if p.requires_grad and p.grad is None:
        #         print(f"Param {n} has no grad!")


def test_decoder_flow_head():
    """Test the DecoderFlowHead implementation."""
    from fvcore.nn import parameter_count_table

    # Create configuration for DecoderFlowHead
    cfg = create_default_mingtok_cfg().pixel_decoder

    # Initialize DecoderFlowHead
    decoder = DecoderFlowHead(cfg).to("cuda", torch.bfloat16)
    print("DecoderFlowHead initialized successfully")
    print(parameter_count_table(decoder))

    # Test inputs
    batch_size = 2
    img_size = 512
    patch_size = 16
    grid_size = img_size // patch_size

    # Input noise latent
    x = torch.randn(batch_size, 3, img_size, img_size).to("cuda", torch.bfloat16)

    # Semantic tokens condition (3D: bs, n, dim)
    sem_tokens = torch.randn(batch_size, grid_size * grid_size, 512).to("cuda", torch.bfloat16)

    # Input shape
    inp_shape = (batch_size, 202, img_size, img_size)

    print(f"Input shapes: x={x.shape}, sem_tokens={sem_tokens.shape}")

    # Test training mode
    with torch.autocast("cuda", dtype=torch.bfloat16):
        recon, losses = decoder(x=x, h=sem_tokens, inp_shape=inp_shape, mode="train", clamp=False)

    print(f"Training mode - recon shape: {recon.shape}, flow_loss: {losses['flow_loss']}")

    # Test inference mode
    with torch.no_grad():
        decoder.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            recon, losses = decoder(
                x=x,
                h=sem_tokens,
                inp_shape=inp_shape,
                mode="sample",
                clamp=True,
                sample_kwargs={"num_steps": 10},
            )

    print(f"Inference mode - recon shape: {recon.shape}, flow_loss: {losses['flow_loss']}")
    print(f"Reconstruction range: [{recon.min().item():.4f}, {recon.max().item():.4f}]")

    # Test gradient flow
    decoder.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        recon, losses = decoder(x, sem_tokens, inp_shape, mode="train")

    if losses["flow_loss"] is not None:
        losses["flow_loss"].backward()
        print("Backward pass successful")

        # Check for gradients
        grad_params = [
            name for name, param in decoder.named_parameters() if param.requires_grad and param.grad is not None
        ]
        print(f"Parameters with gradients: {len(grad_params)}")
    else:
        print("No loss to backward")

    print("DecoderFlowHead test completed successfully!")


def test_deterministic_mingtokrs_tokenizer():
    """Test the full mingtokrs tokenizer with forward-backward pass."""
    from fvcore.nn import parameter_count_table, parameter_count

    cfg = create_default_mingtok_pretrained_cfg()
    print("\n" + "=" * 80)
    print("Testing MingtokRS Tokenizer")
    print("=" * 80)

    # Initialize full model
    print("\n[1/4] Initializing model...")
    print("-" * 80)

    model = MingtokRSModel(cfg).to("cuda", torch.bfloat16)
    model.train()

    # import re

    # re_p = [
    #     "patch_embed",
    #     "pe",
    #     "norm",
    #     "ls1",
    #     "ls2",
    #     "layer_scale",
    #     "norm1",
    #     "norm2",
    #     "q_norm",
    #     "k_norm",
    # ]
    # re_p = [re.compile(p) for p in re_p]
    # for n, p in model.named_parameters():
    #     if p.requires_grad and any([rp.search(n) for rp in re_p]):
    #         print(n)

    # for n, p in model.named_parameters():
    #     if p.isnan().any() or p.isinf().any():
    #         print(f"{n} has nan/inf values")
    #         raise
    #     else:
    #         print(f"{n} value range: {p.min()} - {p.max()}")

    # exit(0)
    print(f"Total resolutions: {model.total_resolutions}")
    print(f"Sem-pix decoder type: {model.sem_pix_decoder_type}")
    print(f"Use REPA: {model.use_repa}")
    print(f"Pretrained tasks: {model.pretrained_task}")
    print(parameter_count_table(model))
    print("Model initialized successfully!")

    # Test forward pass
    print("\n[2/4] Testing forward pass...")
    print("-" * 80)
    batch_size = 3
    img_size = 512
    x = torch.randn(batch_size, 202, img_size, img_size).to("cuda", torch.bfloat16)

    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out['recon'].shape}")
    print("Forward pass successful!")

    # Test backward pass
    print("\n[3/4] Testing backward pass...")
    print("-" * 80)
    x = torch.randn(batch_size, 202, img_size, img_size).to("cuda", torch.bfloat16)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(x)

    # Get the first loss for backward
    loss = out["recon"].mean()
    print(f"Loss value: {loss.item():.4f}")
    loss.backward()

    # Check gradients
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is None:
            print(f"name {n} has no grad")

    # Test encode_ijepa method
    print("\n[4/4] Testing encode_ijepa method...")
    print("-" * 80)
    from src.stage1.self_supervised import MaskCollator

    # model.eval()
    x = torch.randn(batch_size, 202, img_size, img_size).to("cuda", torch.bfloat16)

    # Test without masks
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sem_tokens = model.encode_ijepa(x)

    print(f"Input shape: {x.shape}")
    print(f"Semantic tokens shape (no masks): {sem_tokens.shape}")
    print(f"Semantic tokens dtype: {sem_tokens.dtype}")
    print(f"Semantic tokens range: [{sem_tokens.min().item():.4f}, {sem_tokens.max().item():.4f}]")

    # Test with masks
    mask_collator = MaskCollator(
        input_size=(img_size, img_size),
        patch_size=16,
        enc_mask_scale=[0.8, 1.0],
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        nenc=1,
        npred=4,
        min_keep=10,
        allow_overlap=False,
    )
    x_masked, masks_enc, masks_pred = mask_collator(x)
    masks_enc = [m.to(x.device, torch.int32) for m in masks_enc]

    print(f"\nMask info:")
    print(f"  Number of encoder masks: {len(masks_enc)}")
    print(f"  Encoder mask shapes: {[m.shape for m in masks_enc]}")
    print(f"  Number of prediction masks: {len(masks_pred)}")

    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sem_tokens_masked = model.encode_ijepa(x, jepa_masks=masks_enc)

    print(f"\nSemantic tokens shape (with masks): {sem_tokens_masked.shape}")
    print(f"Semantic tokens dtype: {sem_tokens_masked.dtype}")
    print(f"Semantic tokens range: [{sem_tokens_masked.min().item():.4f}, {sem_tokens_masked.max().item():.4f}]")
    print("encode_ijepa method test successful!")

    repa_feats = model.get_repa_feature()
    print(repa_feats)
    print("\n" + "=" * 80)
    print("All tests passed successfully!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    """
    MODEL_COMPILED=0 CUDA_VISIBLE_DEVICES=0 LOVELY_TENSORS=1 python -m src.stage1.cosmos.cosmos_mingtok
    """

    with logger.catch():
        # test_decoder_flow_head()
        # test_flow_tokenizer()
        test_deterministic_mingtokrs_tokenizer()

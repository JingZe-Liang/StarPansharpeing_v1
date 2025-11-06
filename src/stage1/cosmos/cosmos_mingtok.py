"""
Mingtok-like three-stage tokenizer with low-level and semantic-level feature alignment.

1. add t-dependent decoder (flow transformer decoder + flow head)
2. fix flow UViT decoder
    //2.1 fix time t-r conditioning embedder (in utilities)
//3. add alpha-flow
"""

import math
import random
from functools import partial
from typing import Any, List, Optional, cast, no_type_check

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float
from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf
from timm.layers import (
    create_norm_act_layer,
    create_norm_layer,
    get_act_layer,
    get_norm_act_layer,
    get_norm_layer,
)
from timm.layers.helpers import to_2tuple
from torch import Tensor
from typing_extensions import (
    Annotated,
    Literal,
    Self,
    Sequence,
    TypeAlias,
    TypedDict,
    Union,
)

# fmt: off
from src.utilities.config_utils import to_easydict_recursive, to_object_recursive
from src.utilities.network_utils.network_loading import load_weights_with_shape_check
from src.utilities.transport.flow_matching.transport import Sampler
from src.utilities.transport.flow_matching.transport import Transport as FM_Transport
from src.utilities.transport.tim.transition import TransitionSchedule
from src.utilities.transport.tim.transports import OT_FM
from src.utilities.transport.tim.transports import Transport as Tim_Transport

# fmt: on
from .modules import Attention, AttentionBlock, TransformerTokenizer
from .modules.flowhead import (
    FlowDecoder,
    TimFlowDecoder,
    build_flow_matching_transport,
    build_tim_scheduler,
)
from .modules.proj import build_mlp
from .modules.t_transformer import FlowTransformerConditioned
from .modules.uvit_decoder import UViTDecoder, UViTDecoderConfig

LossOutput = TypedDict(
    "LossOutput",
    {"flow_loss": torch.Tensor},
)
DecoderOutput: TypeAlias = tuple[Tensor, LossOutput]


# *==============================================================
# * Default Configs
# *==============================================================


def _create_flow_transport_cfg():
    ################# Tim flow config
    tim_schedule_kwargs_str = (
        "diffusion_ratio=0.5 consistency_ratio=0.1 weight_time_tangent=true "
        "differential_epsilon=0.005 derivative_type='dde' weight_time_type='sqrt'"
    )
    tim_schedule_cfg = OmegaConf.from_dotlist(tim_schedule_kwargs_str.split(" "))
    tim_transport_str = (
        "P_mean=0.0 P_std=1.0 sigma_d=1.0 enhance_target=false "
        "w_gt=1.0 w_cond=0.75 w_start=0.3 w_end=0.8"
    )
    tim_transport_cfg = OmegaConf.from_dotlist(tim_transport_str.split(" "))

    ################## Flow head FM config mannually sampling
    fm_kwargs_str = "num_sampling_steps=100 train_schedule=fat_lognormal"
    fm_kwargs_cfg = OmegaConf.from_dotlist(fm_kwargs_str.split(" "))

    ########### Sampling cfgs
    tim_sample_str = (
        "num_steps=8 stochasticity_ratio=0.1 sample_type='transition' cfg_scale=2.0 "
        "cfg_low=0.0 cfg_high=0.7"
    )
    tim_sample_cfg = OmegaConf.from_dotlist(tim_sample_str.split(" "))

    # fm_sample_str = "sample_steps=100 schedule=pow_0.25 cfg_scale=2.0 tbar=true"  # flow head fm mannually sample kwargs
    fm_sample_str = "sampling_method=euler num_steps=50 cfg=1.0 progress=true cfg_interval=[0.0,0.7] clip_velocity_per_step=false"
    fm_sample_cfg = OmegaConf.from_dotlist(fm_sample_str.split(" "))

    return {
        "tim": [tim_schedule_cfg, tim_transport_cfg, tim_sample_cfg],
        "fm": [fm_kwargs_cfg, fm_sample_cfg],
    }


def _create_flow_decoder(
    flow_type,
    decoder_type,
    *,
    tim_cfgs: list | None = None,
    fm_cfgs: list | None = None,
):
    # Models

    ################# UViT config
    # TODO: add fm config
    uvit_flow_str: str = (
        "in_chan=512 z_dim=1024 channels=128 ch_mult=[1,1,2,4,8] act_fn='silu' "
        "vit_act_fn='silu' layers_per_block=2 num_attention_heads=8 "
        "dropout=0.0 norm_num_groups=32 time_scale_shift=true "
        "mid_nlayers=12 mid_theta=100.0 eps=1e-5 ada_norm=true "
        "learned_pos_embed=false relative_pos_embed=false "
        f"time_cond_type={'t' if flow_type == 'fm' else 't-r'} "
        "init=null use_act_ckpt=true total_resolutions=16 "
        f"img_size=512 flow_type={flow_type} cfg_prob=0.3"
    )
    uvit_flow_cfg = OmegaConf.from_dotlist(uvit_flow_str.split(" "))
    # NOTE: only supports tim flow yet.
    if flow_type == "tim":
        tim_schedule_cfg, tim_transport_cfg, _ = tim_cfgs
        uvit_flow_cfg.transition_schedule = tim_schedule_cfg
        uvit_flow_cfg.transport = tim_transport_cfg
    elif flow_type == "fm":
        # raise NotImplementedError("flow type fm not implemented yet.")
        pass

    ################# flow head config
    flow_head_str: str = (
        # Transformer decoder kwargs
        "in_chan=768 embed_dim=768 depth=12 num_heads=12 patch_size=1 mlp_ratio=4.0 "
        "norm_layer='rmsnorm' drop_path=0.1 pe_type=rope rope_kwargs.rope_theta=10000.0 "
        "last_norm='rmsnorm' out_chan=512 decoder_img_size=32 head='linear' "
        "n_reg_tokens=0 mask_ratio_train=0.0 is_causal=false attn_type='sdpa' "
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
    else:
        flow_head_cfg.fm_kwargs = fm_cfgs[0]

    if decoder_type == "flow_head":
        pixel_decoder_cfg = flow_head_cfg
    else:
        pixel_decoder_cfg = uvit_flow_cfg

    return pixel_decoder_cfg


def _create_default_mingtok_cfg(flow_type: str = "tim", decoder_type: str = "uvit"):
    """Create a default configuration for Mingtok-like tokenizer model.
    Returns:
        OmegaConf: Default configuration object.

    total_resolution: N downsampled times of the original image size.
    """
    ########### Encoder and decoder
    # CNN encoder and decoder
    # f16c64

    enc_str: str = (
        "in_chan=512 embed_dim=768 depth=12 num_heads=12 patch_size=16 out_patch_size=1 mlp_ratio=4.0 "
        "norm_layer='rmsnorm' drop_path=0.1 pe_type='rope' rope_kwargs.rope_theta=10000.0 "
        "last_norm='rmsnorm' z_dim=512 latent_dim=32 img_size=512 patcher_type='progressive_patch_embedder' "
        "patch_prog_dims=[386,386,512] additional_pe=true "
        "n_reg_tokens=0 mask_ratio_train=0.0 attn_type='sdpa'"
    )
    enc_cfg = OmegaConf.from_dotlist(enc_str.split(" "))

    dec_str: str = (
        "in_chan=32 embed_dim=1024 out_chan=1024 depth=12 num_heads=16 patch_size=1 mlp_ratio=4.0 "
        "norm_layer='rmsnorm' drop_path=0.1 pe_type='rope' rope_kwargs.rope_theta=10000.0 "
        "last_norm='rmsnorm' img_size=32 head='linear' "
        "n_reg_tokens=0 mask_ratio_train=0.0 is_causal=false attn_type='sdpa'"
    )
    dec_cfg = OmegaConf.from_dotlist(dec_str.split(" "))

    #############  two variants of flow decoders
    # UViT and flow head
    flow_dec_cfg_dict = _create_flow_transport_cfg()
    tim_cfgs, fm_cfgs = flow_dec_cfg_dict["tim"], flow_dec_cfg_dict["fm"]
    pixel_decoder_cfg = _create_flow_decoder(
        flow_type, decoder_type, tim_cfgs=tim_cfgs, fm_cfgs=fm_cfgs
    )

    ######## repa projection config

    repa_proj_str = (
        "low_lvl_repa_out_chan=1024 sem_repa_out_chan=1024 low_lvl_cache_layers=[3,6,9,11] "
        "sem_cache_layers=[3,6,9,11] low_lvl_repa_proj_chans=[768,768,768,768] "
        "sem_repa_proj_chans=[1024,1024,1024,1024]"
    )
    repa_proj_cfg = OmegaConf.from_dotlist(repa_proj_str.split(" "))

    ######## Tokenizer main config

    main_str = f"decoder_type={decoder_type} compile=false"
    main_cfg = OmegaConf.from_dotlist(main_str.split(" "))
    main_cfg.sampling_options_default = flow_dec_cfg_dict[flow_type][-1]

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


# *==============================================================
# * Utilities
# *==============================================================


def is_tuple_list(x):
    return isinstance(x, (tuple, list, ListConfig))


def is_sequence_shape(shape: Any) -> bool:
    return isinstance(shape, (torch.Size, list, tuple))


def get_chan_from_shape(shape: torch.Size | tuple[int, ...] | list[int] | int) -> int:
    if is_sequence_shape(shape):
        return shape[1]
    else:
        return shape


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
    elif (
        isinstance(feat_2d_shape, (torch.Size, tuple, list)) and len(feat_2d_shape) == 4
    ):
        gh, gw = feat_2d_shape[2:]
    elif feat.ndim == 4:
        # Assume is square
        gh = gw = int(feat.shape[1] ** 0.5)
    else:
        raise ValueError(
            f"Invalid input args: {feat.shape=}, {feat_2d_shape=}, {inp_shape=}, {total_resolutions=}"
        )

    if feat.ndim == 4:
        assert feat.shape[-2:] == (gh, gw), (
            f"feat shape {feat.shape} already 2d with shape {gh},{gw}"
        )
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
# * Model
# * Encoder, Decoder, DecoderFlowHead, UViTDecoder
# *==============================================================

# TODO: Fix EncoderLowLevel support one transformer (patch size 16) and one decoder progressively
# upsample 16x
# or a time-dependent CNN decoder (like those in uvit decoder does).


class EncoderLowLevel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = TransformerTokenizer(
            in_chan=cfg.in_chan,
            embed_dim=cfg.embed_dim,
            out_chan=cfg.z_dim,
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            out_patch_size=cfg.out_patch_size,
            depth=cfg.depth,
            mlp_ratio=cfg.mlp_ratio,
            norm_layer=cfg.norm_layer,
            drop_path=cfg.drop_path,
            projections={"input": None, "output": "us_average"},
            pe_type=cfg.pe_type,
            rope_kwargs=cfg.rope_kwargs,
            last_norm=cfg.last_norm,
            mask_train_ratio=cfg.mask_ratio_train,
            is_causal=False,
            n_reg_tokens=cfg.n_reg_tokens,
            attn_type=cfg.attn_type,
            head="linear",
            patcher_type=cfg.patcher_type,
            patch_prog_dims=getattr(cfg, "patch_prog_dims", None),
        )
        self.z_to_latent = nn.Linear(cfg.z_dim, cfg.latent_dim)

    def forward(self, x, get_intermidates=None):
        z, out = self.encoder(
            x, ret_2d_tokens=False, ret_all=True, get_intermidates=get_intermidates
        )
        h = self.z_to_latent(z)
        return dict(latent=h, z=z, **out)


class DecoderSemantic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = TransformerTokenizer(
            in_chan=cfg.in_chan,
            embed_dim=cfg.embed_dim,
            out_chan=cfg.out_chan,
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            patcher_type="linear",
            depth=cfg.depth,
            mlp_ratio=cfg.mlp_ratio,
            norm_layer=cfg.norm_layer,
            drop_path=cfg.drop_path,
            projections={"input": "ds_shortcut", "output": None},
            pe_type=cfg.pe_type,
            rope_kwargs=cfg.rope_kwargs,
            last_norm=cfg.last_norm,
            mask_train_ratio=0.0,
            is_causal=cfg.is_causal,
            n_reg_tokens=cfg.n_reg_tokens,
            attn_type=cfg.attn_type,
            head="linear",
        )

    def forward(self, x, get_intermidates=None):
        """
        Encode the latent low-level tokens and convert into semantic tokens.
        """
        h, out = self.encoder(
            x, ret_2d_tokens=False, ret_all=True, get_intermidates=get_intermidates
        )
        return dict(sem_tokens=h, **out)


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
                time_cond_type="t,t-r",
                img_size=cfg.img_size,
                head_type=cfg.head_type,
                head_kwargs=cfg.head_kwargs,
                # tim config
                fm_kwargs=cfg.fm_kwargs,
                transition_schedule_kwargs=cfg.transition_schedule_kwargs,
            )
        else:
            raise ValueError(f"Unknown flow_type: {cfg.flow_type}")

        ####### Build transport and sampler
        if self.transformer_t_conditioned:
            if cfg.flow_type == "fm":
                self.transport, self.sampler = build_flow_matching_transport()
            elif self.flow_type == "tim":
                self.transport = build_tim_scheduler()
            else:
                raise ValueError(f"{self.flow_type} is not supported.")

    ############ Traning functions #############
    
    def flow_train_forward_head_standalone(
        self, x, h, inp_shape, ema_model, clamp=False
    ):
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
            terms = self.transport.training_losses(
                self._forward_non_standalone_all_model, x, model_kwargs={"z": h}
            )
            flow_loss = terms["loss"].mean()
            recon = terms["pred_x_clean"]
            return recon, flow_loss

        elif self.flow_type == "tim":
            raise NotImplementedError

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

    def flow_sample_all(
        self, x, h, inp_shape, sample_kwargs={}, clamp=False, ret_trajectory=False
    ):
        assert self.transformer_t_conditioned
        if self.flow_type == "fm":
            cfg_scale, cfg_interval, clip_v = (
                sample_kwargs.pop("cfg", 1.0),
                sample_kwargs.pop("cfg_interval", (0, 1)),
                sample_kwargs.pop("clip_v", False),
            )
            x_init = x.copy()

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
                    v_uncond = self._forward_non_standalone_all_model(
                        x, t, null_cond_h, inp_shape=inp_shape
                    )
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
            raise NotImplementedError

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
                self.flow_train_all
                if self.transformer_t_conditioned
                else self.flow_train_forward_head_standalone
            )
            recon, flow_loss = train_fn_(
                x, h, inp_shape=inp_shape, ema_model=ema_model, clamp=clamp
            )
        elif mode == "sample":
            sample_fn_ = (
                self.flow_sample_all
                if self.transformer_t_conditioned
                else self.flow_sample_head_stand_alone
            )
            recon = sample_fn_(
                x,
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


class DecoderUViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.t_res = cfg.total_resolutions
        img_size = cfg.img_size
        grid_size = img_size // self.t_res

        self.decoder = UViTDecoder(
            in_channels=cfg.in_chan,
            z_dim=cfg.z_dim,
            channels=cfg.channels,
            ch_mult=cfg.ch_mult,
            act_fn=cfg.act_fn,
            vit_act_fn=cfg.vit_act_fn,
            layers_per_block=cfg.layers_per_block,
            num_attention_heads=cfg.num_attention_heads,
            total_resolutions=cfg.total_resolutions,
            dropout=cfg.dropout,
            norm_num_groups=cfg.norm_num_groups,
            time_scale_shift=cfg.time_scale_shift,
            mid_nlayers=cfg.mid_nlayers,
            mid_theta=cfg.mid_theta,
            eps=cfg.eps,
            ada_norm=cfg.ada_norm,
            learned_pos_embed=cfg.learned_pos_embed,
            image_size=cfg.img_size,
            relative_pos_embed=cfg.relative_pos_embed,
            time_cond_type=cfg.time_cond_type,
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
            # fm_kwargs = cfg.fm_kwargs
            self.transport = FM_Transport(
                model_type="velocity",
                path_type="linear",
                loss_type="velocity",
                train_eps=0.001,
                sample_eps=0.001,
            )
            self.sampler = Sampler(transport=self.transport)
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
        n_cfg = int(h.shape[0] * cfg_prob)
        assert cfg_prob >= 0.0
        if n_cfg == 0:
            # h as original, no cfg
            return h
        else:
            cfg_mask = torch.rand(h.shape[0], device=h.device) < cfg_prob
            null_h = (
                self._get_null_h(h) + 0.0 * h
            )  # partial gradients flow back to encoder/semantic decoder
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
        if mode == "train":
            if self.transition_schedule.transport.enhance_target:
                assert ema_model is not None, "EMA model is required for enhance_target"
                assert hasattr(self, "null_cond_h"), (
                    "The decoder must have null_cond_h for CFG"
                )

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
            recon: Tensor = breakdowns["x0_pred"]
        elif mode == "sample":
            # eval: loop to generate reconstruction image when h is the condition.
            # x_init = torch.randn_like(x)
            x_init = x
            recon: Tensor = self.transition_schedule.sample(
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
            x_init = x
            cfg_scale: float = sample_kwargs.pop("cfg", 1.0)
            cfg_interval: tuple[int, int] | None = sample_kwargs.pop(
                "cfg_interval", None
            )
            clip_v = sample_kwargs.pop("clip_velocity_per_step", False)
            sample_fn_ = self.sampler.sample_ode(**sample_kwargs)

            # takes [x, model, model_kwargs], model takes [x, t, **model_kwargs]
            @torch.no_grad()
            def _model_step_fn(x, t):
                nonlocal cfg_scale, cfg_interval, clip_v

                null_cond_h = self._get_null_h(h)

                v = self.decoder(x, t, z=h, inp_shape=inp_shape)
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
                    v_uncond = self.decoder(x, t, z=null_cond_h, inp_shape=inp_shape)
                    v = v_uncond + cfg_scale * (v - v_uncond)

                # clip fake x1 to (-1, 1)
                # v = x0 - x1
                if clip_v:
                    x1_hat = x_init + v
                    x1_hat = x1_hat.clamp(-1, 1)
                    # back to clip velocity
                    v = x1_hat - x_init

                return v

            recon = sample_fn_(x_init, _model_step_fn)

            if not ret_trajectory:
                recon = recon[-1]
            flow_loss = None

        losses = {"flow_loss": flow_loss}
        return recon, losses

    def forward(self, x, h, inp_shape, mode="train", *args, **kwargs):
        if self._flow_type == "tim":
            return self._forward_tim(x, h, inp_shape, mode, *args, **kwargs)
        else:
            return self._forward_fm(x, h, inp_shape, mode, *args, **kwargs)


class FlowTokenizer(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        cfg = to_easydict_recursive(cfg)

        # Cfgs
        self.low_cfg = cfg.low_level_encoder
        self.sem_cfg = cfg.semantic_decoder
        self.pix_flow_cfg = cfg.pixel_decoder
        self.tok_cfg = cfg.tokenizer

        # Model parts
        logger.info(f"Init low-level encoder.")
        self.low_level_encoder = EncoderLowLevel(self.low_cfg)

        logger.info(f"Init semantic decoder.")
        self.semantic_decoder = DecoderSemantic(self.sem_cfg)

        logger.info(f"Init pixel decoder of type {self.tok_cfg.decoder_type}.")
        decoder_head_cls = {"flow_head": DecoderFlowHead, "uvit": DecoderUViT}[
            self.tok_cfg.decoder_type
        ]
        self.pixel_decoder = decoder_head_cls(self.pix_flow_cfg)

        self.total_resolutions: int = self.pix_flow_cfg.total_resolutions
        self.sampling_options_default: dict[str, Any] = getattr(
            cfg.tokenizer, "sampling_options_default", {}
        )
        logger.info(f"Set the sampling options to {self.sampling_options_default}.")

        # TODO: add quantizer

        # Semantic and low-level caches
        self.proj_cfg = cfg.repa_proj
        self.use_repa: bool = cfg.repa_proj is not None
        self.z: Tensor | None = None
        self.sem_z: Tensor | None = None
        self._hw: tuple | list | None = None
        self.low_lvl_cache_layers = self.proj_cfg.low_lvl_cache_layers
        self.sem_cache_layers = self.proj_cfg.sem_cache_layers
        self.low_lvl_proj_is_multi = is_tuple_list(self.low_lvl_cache_layers)
        self.sem_proj_is_multi = is_tuple_list(self.sem_cache_layers)

        # trainer compatiblity
        self._use_repa_loss = self.use_repa
        self._use_vf_loss = False

        self._build_repa_projections()

        # compile model parts
        self._compile = getattr(self.tok_cfg, "compile", False)
        if self._compile:
            logger.info("Compiling FlowTokenizer model parts...")
            # Only compile the heavy parts - transformer
            self.low_level_encoder.encoder = torch.compile(self.low_level_encoder.encoder)  # fmt: skip
            self.semantic_decoder.encoder = torch.compile(self.semantic_decoder.encoder)
            self.pixel_decoder.decoder = torch.compile(self.pixel_decoder.decoder)  # type: ignore
            logger.info("Compilation done.")

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
                        is_1d=True,
                    )
                    low_lvl_z_proj.append(proj_)
            else:
                low_lvl_z_proj = build_mlp(
                    self.proj_cfg.low_lvl_repa_proj_chans,
                    self.proj_cfg.low_lvl_repa_out_chan,
                    self.proj_cfg.low_lvl_repa_out_chan,
                    is_1d=True,
                )

            if self.sem_proj_is_multi:
                sem_z_proj = nn.ModuleList()
                for i in range(len(self.sem_cache_layers)):
                    proj_ = build_mlp(
                        self.proj_cfg.sem_repa_proj_chans[i],
                        self.proj_cfg.sem_repa_out_chan,
                        self.proj_cfg.sem_repa_out_chan,
                        is_1d=True,
                    )
                    sem_z_proj.append(proj_)
            else:
                sem_z_proj = build_mlp(
                    self.proj_cfg.sem_repa_proj_chans,
                    self.proj_cfg.sem_repa_out_chan,
                    self.proj_cfg.sem_repa_out_chan,
                    is_1d=True,
                )

            self._repa_proj = nn.ModuleDict(
                {
                    "low_lvl_repa_proj": low_lvl_z_proj,
                    "sem_repa_proj": sem_z_proj,
                }
            )

    @staticmethod
    def _to_2d(x, hw: List[int]) -> None | Tensor:
        if x is None:
            return x
        else:
            return rearrange(x, "b (h w) ... -> b ... h w", h=hw[0], w=hw[1])

    def _encode_latent(self, x):
        low_lvl_out = self.low_level_encoder(
            x, get_intermidates=self.low_lvl_cache_layers if self.training else None
        )
        # Cache low level features
        self.z = (
            low_lvl_out["intermidates"]
            if self.low_lvl_proj_is_multi
            else low_lvl_out["h"]
        )
        return low_lvl_out

    def _sem_decode(self, latent):
        sem_out = self.semantic_decoder(
            latent, get_intermidates=self.sem_cache_layers if self.training else None
        )
        # Cache semantic features
        self.sem_z = sem_out["intermidates"] if self.sem_proj_is_multi else sem_out["z"]
        return sem_out

    @no_type_check
    @torch.autocast("cuda", dtype=torch.bfloat16)
    def get_repa_feature(self):
        if not self.training or not self.use_repa:
            return None
        assert self.z is not None and self.sem_z is not None, (
            f"z and sem_z must be set before get_repa_feature but {self.z=} and {self.sem_z=}"
        )

        to_2d = partial(self._to_2d, hw=self._hw)

        # Low-level feature distillation
        if self.low_lvl_proj_is_multi:
            low_lvl_features = [
                to_2d(self._repa_proj["low_lvl_repa_proj"][i](feat))
                for i, feat in enumerate(self.z)
            ]
        else:
            low_lvl_features = to_2d(self._repa_proj["low_lvl_repa_proj"](self.z))

        # Semantic feature distillation
        if self.sem_proj_is_multi:
            sem_features = [
                to_2d(self._repa_proj["sem_repa_proj"][i](feat))
                for i, feat in enumerate(self.sem_z)
            ]
        else:
            sem_features = to_2d(self._repa_proj["sem_repa_proj"](self.sem_z))

        return low_lvl_features, sem_features

    def _decode_to_sem(self, x=None, latent=None, hw: Sequence | None = None):
        mask = None
        if x is not None:
            hw = torch.as_tensor(x.shape[-2:]) // self.total_resolutions
            hw = hw.tolist()
            low_lvl_out = self._encode_latent(x)
            latent = low_lvl_out["latent"]
            mask = low_lvl_out["mask"]
        elif latent is not None:
            assert latent.ndim == 3, (
                f"latent must be of shape (bs, n, dim), but got {latent.shape=}"
            )
            hw_l = latent.shape[1]
            if hw is None:
                hw = to_2tuple(int(math.sqrt(hw_l)))
        else:
            raise ValueError("Either x or latent must be provided")

        # To semantic tokens
        sem_out = self._sem_decode(latent)
        sem_tokens = sem_out["sem_tokens"]
        # sem_proj_out = sem_out["sem_proj_out"]

        # To 2d
        assert hw is not None, "hw must be provided"
        assert math.prod(hw) == latent.shape[1], (
            f"hw {hw} product does not match latent shape {latent.shape} "
            f"and total_resolutions={self.total_resolutions}"
        )

        # output of 2d shape
        latent, sem_tokens, mask = map(
            partial(self._to_2d, hw=hw), (latent, sem_tokens, mask)
        )
        out = dict(latent=latent, sem_tokens=sem_tokens, mask=mask)
        return out

    def encode(self, x):
        out = self._encode_latent(x)
        return out

    def decode(
        self,
        x: torch.Tensor,
        latent: torch.Tensor,
        inp_shape: Annotated[Union[torch.Size, int, tuple], "bs,c,h,w or bs,c or c"],
        mode: Literal["train", "sample"] = "train",
        clamp=False,
        ema_model: Optional[Self] = None,
        sample_kwargs: dict = dict(
            ## tim sample kwargs
            # num_steps=8,
            # stochasticity_ratio=0.0,
            # sample_type="transition",
            # cfg_scale=1.0,
            ## fm sample kwargs
            # sampling_method="Euler",
            # diffusion_form="SBDM",
            # diffusion_norm=1.0,
            # last_step="Mean",
            # last_step_size=0.04,
            # num_steps=250,
            # temperature=1.0,
            ## manually sampling kwargs
            # sample_steps=10,
            # schedule='linear',
            # cfg=2.0,
            # cfg_interval=None,
            # tbar=True,
        ),
        ret_trajectory=False,
    ):
        """
        Low level encoded latent is decoded into semantic tokens, and then decoded into the reconstructed image or
        velocity. Output dict of losses, recon, semantic tokens and projected semantic tokens.
        """
        # HW and EMA
        ema_decoder = ema_model.pixel_decoder if ema_model is not None else None
        hw = (torch.as_tensor(x.shape[-2:]) // self.total_resolutions).tolist()
        self._hw = hw

        # Decode to semantic tokens
        sem_decoder_out = self._decode_to_sem(latent=latent, hw=hw)
        h_sem = sem_decoder_out["sem_tokens"]  # as conditions

        # To flow UViT or head
        sample_kwargs = self.sampling_options_default | sample_kwargs

        if mode == "sample":
            # init the x as the noise
            logger.trace(f"[flow tokenizer]: init the x with gaussian - mode = {mode}")
            x_inp = torch.randn_like(x)
        elif mode == "train":
            x_inp = x
        else:
            raise ValueError(f"[flow tokenizer]: mode {mode} not supported")

        # Flow decoder
        out = self.pixel_decoder(
            x=x_inp,
            h=h_sem,
            inp_shape=inp_shape,
            mode=mode,
            clamp=clamp,
            ema_model=ema_decoder,
            sample_kwargs=sample_kwargs,
            ret_trajectory=ret_trajectory,
        )

        # Form the output
        recon, loss_dict = out
        output = dict(recon=recon, losses=loss_dict, sem_tokens=h_sem)
        return output

    def forward(
        self,
        input: torch.Tensor,
        dec_mode: Literal["train", "sample"] = "train",
        clamp: bool = False,
        ema_model: Optional[Self] = None,
        sample_kwargs: dict = {},
        ret_trajectory: bool = False,
    ):
        if input.shape[1] > 512:
            logger.error(f"input shape {input.shape} is too large.")
            raise

        enc_out = self.encode(input)
        dec_out = self.decode(
            input,
            enc_out["latent"],
            input.shape,
            dec_mode,
            clamp,
            ema_model,
            sample_kwargs,
            ret_trajectory,
        )

        # Compatible with trainer
        out = (
            dec_out["recon"],
            dict(
                flow_loss=dec_out["losses"]["flow_loss"],
                q_loss=torch.tensor(0.0).to(input),
            ),
            None,
        )
        return out

    def load_pretrained(self, path: str):
        import accelerate

        sd = accelerate.utils.load_state_dict(path)
        missing_keys, unexpected_keys = load_weights_with_shape_check(self, sd)
        if len(missing_keys) > 0:
            logger.warning(
                f"Missing keys when loading pretrained model: {missing_keys}"
            )
        if len(unexpected_keys) > 0:
            logger.warning(
                f"Unexpected keys when loading pretrained model: {unexpected_keys}"
            )
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
    def create_model(cls, cfg, flow_type: str, decoder_type: str):
        """Create a FlowTokenizer model from configuration."""
        # Update the defaults
        tokenizer_cfg_default = _create_default_mingtok_cfg(flow_type, decoder_type)
        if cfg is not None:
            cfg = OmegaConf.merge(tokenizer_cfg_default, cfg)
        else:
            cfg = tokenizer_cfg_default

        model = cls(cfg)
        return model


# * --- Test --- #


def test_flow_tokenizer():
    from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count_table

    from src.data.litdata_hyperloader import ImageStreamingDataset

    tokenizer_cfg_default = _create_default_mingtok_cfg("tim", "uvit")
    tokenizer = FlowTokenizer(tokenizer_cfg_default).to("cuda", torch.bfloat16)
    tokenizer.set_grad_checkpointing()
    # print(flop_count_table(FlopCountAnalysis(tokenizer, x)))
    # print(parameter_count_table(tokenizer))

    # path = "runs/stage1_mingtok/2025-11-05_03-54-45_mingtok_600M_continous/ema/tokenizer/model.safetensors"
    # tokenizer.load_pretrained(path)
    # logger.info("Load pretrained done.")

    x = torch.randn(9, 32, 512, 512).to("cuda", torch.bfloat16)
    # dataset = ImageStreamingDataset(
    #     input_dir="data2/RemoteSAM270k/LitData_hyper_images",
    #     resize_before_transform=512,
    # )
    # x = dataset[201]["img"]
    # x = x[None].type(torch.bfloat16).to("cuda")

    # print(f"Input shape: {x.shape}")
    # print("Encoding and decoding (train mode)")
    # with torch.autocast("cuda", dtype=torch.bfloat16):
    #     with torch.no_grad():
    #         out = tokenizer(x)
    # # for k, v in out.items():
    # #     print("{:>20}: {}".format(k, v))

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
    tokenizer.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = tokenizer(x)
    tokenizer.get_repa_feature()  # to set repa features
    loss = out[1]["flow_loss"]
    print(f"Loss: {loss}")
    loss.backward()
    print("Backward successful.")

    # Check the gradients
    for n, p in tokenizer.named_parameters():
        if p.requires_grad and p.grad is None:
            print(f"Param {n} has no grad!")


def test_decoder_flow_head():
    """Test the DecoderFlowHead implementation."""
    from fvcore.nn import parameter_count_table

    # Create configuration for DecoderFlowHead
    cfg = _create_default_mingtok_cfg().pixel_decoder

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
    sem_tokens = torch.randn(batch_size, grid_size * grid_size, 512).to(
        "cuda", torch.bfloat16
    )

    # Input shape
    inp_shape = (batch_size, 202, img_size, img_size)

    print(f"Input shapes: x={x.shape}, sem_tokens={sem_tokens.shape}")

    # Test training mode
    with torch.autocast("cuda", dtype=torch.bfloat16):
        recon, losses = decoder(
            x=x, h=sem_tokens, inp_shape=inp_shape, mode="train", clamp=False
        )

    print(
        f"Training mode - recon shape: {recon.shape}, flow_loss: {losses['flow_loss']}"
    )

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

    print(
        f"Inference mode - recon shape: {recon.shape}, flow_loss: {losses['flow_loss']}"
    )
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
            name
            for name, param in decoder.named_parameters()
            if param.requires_grad and param.grad is not None
        ]
        print(f"Parameters with gradients: {len(grad_params)}")
    else:
        print("No loss to backward")

    print("DecoderFlowHead test completed successfully!")


if __name__ == "__main__":
    """
    CUDA_VISBILE_DEVICES=1 LOVELY_TENSORS=1 python -m src.stage1.cosmos.cosmos_mingtok
    """
    import sys

    with logger.catch():
        # test_decoder_flow_head()
        test_flow_tokenizer()

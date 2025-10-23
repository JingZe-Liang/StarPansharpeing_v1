import math
from functools import partial
from typing import Any, List, Optional, no_type_check

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float
from loguru import logger
from omegaconf.omegaconf import OmegaConf
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

from src.utilities.config_utils import to_object_recursive
from src.utilities.network_utils.network_loading import load_weights_with_shape_check
from src.utilities.transport.tim.transition import TransitionSchedule
from src.utilities.transport.tim.transports import OT_FM, Transport

from .modules import Attention, AttentionBlock, TransformerTokenizer
from .modules.flowhead import FlowDecoder, TimFlowDecoder
from .modules.proj import build_mlp
from .modules.uvit_decoder import UViTDecoder, UViTDecoderConfig

LossOutput = TypedDict(
    "LossOutput",
    {"flow_loss": torch.Tensor},
)
DecoderOutput: TypeAlias = tuple[Tensor, LossOutput]


# *==============================================================
# * Default Configs
# *==============================================================


enc_str: str = (
    "in_chan=3 embed_dim=512 depth=12 patch_size=16 out_patch_size=1 mlp_ratio=4.0 "
    "norm_layer='rmsnorm' drop_path=0.1 pe_type='learned' rope_kwargs={} "
    "last_norm='rmsnorm' z_dim=256 latent_dim=16 img_size=512 head='linear' "
    "n_reg_tokens=0 mask_ratio_train=0.0 attn_type='sdpa'"
)
enc_cfg = OmegaConf.from_dotlist(enc_str.split(" "))

dec_str: str = (
    "in_chan=16 embed_dim=512 depth=12 patch_size=1 mlp_ratio=4.0 "
    "norm_layer='rmsnorm' drop_path=0.1 pe_type='learned' rope_kwargs={} "
    "last_norm='rmsnorm' out_chan=512 img_size=32 head='linear' "
    "n_reg_tokens=0 mask_ratio_train=0.0 is_causal=False attn_type='sdpa'"
)
dec_cfg = OmegaConf.from_dotlist(dec_str.split(" "))

pix_flow_str: str = (
    # Model kwargs
    "in_chan=3 z_dim=512 channels=128 ch_mult=[1,1,2,2,4] act_fn='silu' "
    "vit_act_fn='geglu' layers_per_block=2 num_attention_heads=8 "
    "dropout=0.0 norm_num_groups=32 time_scale_shift=True "
    "mid_nlayers=12 mid_theta=100.0 eps=1e-5 ada_norm=True "
    "learned_pos_embed=False image_size=null relative_pos_embed=False "
    "time_cond_type='t-r' init=null use_act_ckpt=False total_resolutions=16 "
    "img_size=512 "
    # Schedule kwargs
    "transition_schedule.diffusion_ratio=0.5 "
    "transition_schedule.consistency_ratio=0.1 "
    "transition_schedule.derivative_type='dde' "
    "transition_schedule.differential_epsilon=0.05 "
    "transition_schedule.weight_time_type='sqrt' "
    "transition_schedule.weight_time_tangent=True "
    # Transport kwargs
    "transport.P_mean=0.0 "
    "transport.P_std=1.0 "
    "transport.sigma_d=1.0 "
    "transport.enhance_target=False "
)
pix_flow_cfg = OmegaConf.from_dotlist(pix_flow_str.split(" "))

# Configuration for DecoderFlowHead
flow_head_str: str = (
    # Transformer tokenizer kwargs
    "in_chan=512 embed_dim=768 depth=12 patch_size=1 mlp_ratio=4.0 "
    "norm_layer='rmsnorm' drop_path=0.1 pe_type='learned' rope_kwargs={} "
    "last_norm='rmsnorm' out_chan=512 img_size=32 head='linear' "
    "n_reg_tokens=0 mask_ratio_train=0.0 is_causal=False attn_type='sdpa' "
    # Flow decoder kwargs
    "target_channels=3 z_channels=512 flow_depth=6 flow_width=768 "
    "patch_size=16 img_size=512 grad_checkpointing=False "
    "num_sampling_steps=10 train_schedule='fat_lognormal' use_cfg=False "
    "head_type='once' head_kwargs={} cfg_prob=0.5 total_resolutions=16 flow_type=tim"
)
flow_head_cfg = OmegaConf.from_dotlist(flow_head_str.split(" "))

repa_proj_str = (
    "low_lvl_repa_out_chan=1024 sem_repa_out_chan=1152 low_lvl_cache_layers=[3,6,9,11] "
    "sem_cache_layers=[3,6,9,11] low_lvl_repa_proj_chans=[768,768,768,768] "
    "sem_repa_proj_chans=[768,768,768,768]"
)
repa_proj_cfg = OmegaConf.from_dotlist(repa_proj_str.split(" "))

main_str = "decoder_type=flow_head"
main_cfg = OmegaConf.from_dotlist(main_str.split(" "))

tokenizer_cfg_default = OmegaConf.create(
    {
        # encoder, semantic decoder and pixel decoder configs
        "low_level_encoder": enc_cfg,
        "semantic_decoder": dec_cfg,
        "pixel_decoder": flow_head_cfg,  # flow_head_cfg or pix_flow_cfg
        "tokenizer": main_cfg,
        "repa_proj": repa_proj_cfg,
    }
)


# *==============================================================
# * Utilities
# *==============================================================


def is_tuple_list(x):
    return isinstance(x, (tuple, list))


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
            head="linear",
            patcher_type="patch_embedder",
            attn_type=cfg.attn_type,
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

        # Transformer tokenizer for processing semantic tokens
        self.decoder = TransformerTokenizer(
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
            projections={"input": None, "output": None},
            pe_type=cfg.pe_type,
            rope_kwargs=cfg.rope_kwargs,
            last_norm=cfg.last_norm,
            mask_train_ratio=0.0,
            is_causal=cfg.is_causal,
            n_reg_tokens=cfg.n_reg_tokens,
            attn_type=cfg.attn_type,
            head=cfg.head,
        )

        # Flow decoder for image generation
        self.flow_type = cfg.flow_type
        if cfg.flow_type == "fm":
            self.flow_decoder = FlowDecoder(
                target_channels=cfg.target_channels,
                z_channels=cfg.z_channels,
                depth=cfg.flow_depth,
                width=cfg.flow_width,
                grad_checkpointing=cfg.grad_checkpointing,
                num_sampling_steps=cfg.num_sampling_steps,
                train_schedule=cfg.train_schedule,
                use_cfg=cfg.use_cfg,
                cfg_prob=cfg.cfg_prob,
                patch_size=cfg.patch_size,
                img_size=cfg.img_size,
                head_type=cfg.head_type,
                head_kwargs=cfg.head_kwargs,
            )
        elif cfg.flow_type == "tim":
            self.flow_decoder = TimFlowDecoder(
                target_channels=cfg.target_channels,
                z_channels=cfg.z_channels,
                depth=cfg.flow_depth,
                width=cfg.flow_width,
                grad_checkpointing=cfg.grad_checkpointing,
                use_cfg=cfg.use_cfg,
                cfg_prob=cfg.cfg_prob,
                patch_size=cfg.patch_size,
                time_cond_type="t,t-r",
                img_size=cfg.img_size,
                head_type=cfg.head_type,
                head_kwargs=cfg.head_kwargs,
                fm_kwargs=cfg.fm_kwargs,
                transition_schedule_kwargs=cfg.transition_schedule_kwargs,
            )
        else:
            raise ValueError(f"Unknown flow_type: {cfg.flow_type}")

    def flow_train_forward(self, x, h, inp_shape, ema_model, clamp=False):
        assert self.training, "flow_train_forward should only be used in training mode"
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
            recon, flow_losses = self.flow_decoder.training_loss(
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

    def flow_sample(self, h, inp_shape, sample_kwargs={}, clamp=False):
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

    def forward(
        self,
        x: torch.Tensor,  # bs, c, h, w - input noise latent
        h: Tensor,  # bs, n, dim or bs, c, gh, gw - semantic tokens condition
        inp_shape: Annotated[Union[torch.Size, tuple], "bs,c,h,w"],
        mode: Literal["train", "sample"] = "train",
        clamp: bool = False,
        ema_model: Optional[Self] = None,
        sample_kwargs: dict = dict(
            num_steps=8,
            stochasticity_ratio=0.0,
            cfg=1.0,
            cfg_interval=None,
            sample_steps=None,
        ),
        ret_trajectory: bool = False,
    ):
        # Process semantic tokens condition
        h = latent_2d_to_1d(h)

        # Decode semantic tokens using transformer tokenizer
        processed_tokens = self.decoder(
            h, ret_2d_tokens=False, ret_all=False, out_shape=None
        )

        # Use flow decoder with processed tokens as condition
        # Flow decoder handles time internally and uses processed tokens as conditioning
        if mode == "train":
            recon, flow_loss = self.flow_train_forward(
                x,
                processed_tokens,
                inp_shape=inp_shape,
                ema_model=ema_model,
                clamp=clamp,
            )
        elif mode == "sample":
            recon = self.flow_sample(
                processed_tokens,
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
            image_size=cfg.image_size,
            relative_pos_embed=cfg.relative_pos_embed,
            time_cond_type=cfg.time_cond_type,
            init=cfg.init,
            use_act_ckpt=cfg.use_act_ckpt,
        )

        # FM transport and scheduler
        transition_schedule_kwargs = cfg.transition_schedule
        transport_kwargs = cfg.transport
        self.transport = OT_FM(**transport_kwargs)
        self.transition_schedule = TransitionSchedule(
            self.transport,
            **transition_schedule_kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,  # bs, c, h, w
        h: Tensor,  # bs, n, dim or bs, c, h, w
        inp_shape: Annotated[Union[torch.Size, int, tuple], "bs,c,h,w or bs,c or c"],
        mode: Literal["trian", "sample"] = "train",
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
        null_cond_h = self.decoder.null_cond_h if hasattr(self.decoder, "null_cond_h") else None  # fmt: skip

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
            # no CFG
            x_init = torch.randn_like(x)
            recon: Tensor = self.transition_schedule.sample(
                self.decoder,
                # start sampling noise
                z=x_init,
                # conditions
                y=h,
                y_null=null_cond_h,  # since no CFG, y_null is None
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


class FlowTokenizer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Cfgs
        self.low_cfg = cfg.low_level_encoder
        self.sem_cfg = cfg.semantic_decoder
        self.pix_flow_cfg = cfg.pixel_decoder
        self.tok_cfg = cfg.tokenizer

        # Model parts
        self.low_level_encoder = EncoderLowLevel(self.low_cfg)
        self.semantic_decoder = DecoderSemantic(self.sem_cfg)
        decoder_head_cls = {"flow_head": DecoderFlowHead, "uvit": DecoderUViT}[
            self.tok_cfg.decoder_type
        ]
        self.pixel_decoder = decoder_head_cls(self.pix_flow_cfg)
        self.total_resolutions: int = self.pix_flow_cfg.total_resolutions

        # TODO: add quantizer

        # Semantic and low-level caches
        self.proj_cfg = cfg.repa_proj
        self.use_repa = cfg.repa_proj is not None
        self.z = None
        self.sem_z = None
        self.low_lvl_cache_layers = self.proj_cfg.low_lvl_cache_layers
        self.sem_cache_layers = self.proj_cfg.sem_cache_layers
        self.low_lvl_proj_is_multi = is_tuple_list(self.low_lvl_cache_layers)
        self.sem_proj_is_multi = is_tuple_list(self.sem_cache_layers)

        self._build_repa_projections()

    def _build_repa_projections(self):
        if self.use_repa:
            # Hier distillation
            if self.low_lvl_proj_is_multi:
                low_lvl_z_proj = nn.ModuleList()
                for i in range(len(self.low_lvl_cache_layers)):
                    proj_ = build_mlp(
                        self.proj_cfg.low_lvl_repa_proj_chans[i],
                        self.proj_cfg.low_lvl_repa_out_chan,
                        self.proj_cfg.low_lvl_repa_out_chan,
                    )
                    low_lvl_z_proj.append(proj_)
            else:
                low_lvl_z_proj = build_mlp(
                    self.proj_cfg.low_lvl_repa_proj_chans,
                    self.proj_cfg.low_lvl_repa_out_chan,
                    self.proj_cfg.low_lvl_repa_out_chan,
                )

            if self.sem_proj_is_multi:
                sem_z_proj = nn.ModuleList()
                for i in range(len(self.sem_cache_layers)):
                    proj_ = build_mlp(
                        self.proj_cfg.sem_repa_proj_chans[i],
                        self.proj_cfg.sem_repa_out_chan,
                        self.proj_cfg.sem_repa_out_chan,
                    )
                    sem_z_proj.append(proj_)
            else:
                sem_z_proj = build_mlp(
                    self.proj_cfg.sem_repa_proj_chans,
                    self.proj_cfg.sem_repa_out_chan,
                    self.proj_cfg.sem_repa_out_chan,
                )

            self._repa_proj = nn.ModuleDict(
                {
                    "low_lvl_repa_proj": low_lvl_z_proj,
                    "sem_repa_proj": sem_z_proj,
                }
            )
            logger.info("Build Repa distillation projectors.")

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

        if self.low_lvl_proj_is_multi:
            low_lvl_features = [
                self._repa_proj["low_lvl_repa_proj"][i](feat)
                for i, feat in enumerate(self.z)
            ]
        else:
            low_lvl_features = self._repa_proj["low_lvl_repa_proj"](self.z)

        if self.sem_proj_is_multi:
            sem_features = [
                self._repa_proj["sem_repa_proj"][i](feat)
                for i, feat in enumerate(self.sem_z)
            ]
        else:
            sem_features = self._repa_proj["sem_repa_proj"](self.sem_z)

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
        sem_proj_out = sem_out["sem_proj_out"]

        # To 2d
        assert hw is not None, "hw must be provided"
        assert math.prod(hw) == latent.shape[1], (
            f"hw {hw} product does not match latent shape {latent.shape} "
            f"and total_resolutions={self.total_resolutions}"
        )

        # output of 2d shape
        latent, sem_tokens, sem_proj_out, mask = map(
            partial(self._to_2d, hw=hw),
            (latent, sem_tokens, sem_proj_out, mask),
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
            num_steps=8,
            stochasticity_ratio=0.0,
            sample_type="transition",
            cfg_scale=1.0,
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

        # Decode to semantic tokens
        sem_decoder_out = self._decode_to_sem(latent=latent, hw=hw)
        h_sem = sem_decoder_out["sem_tokens"]

        # To flow UViT or head
        out = self.pixel_decoder(
            x=x,
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
        output = dict(
            recon=recon,
            losses=loss_dict,
            sem_tokens=h_sem,
        )
        return output

    def forward(
        self,
        input: torch.Tensor,
        dec_mode: Literal["train", "sample"] = "train",
        clamp: bool = False,
        ema_model: Optional[Self] = None,
        sample_kwargs: dict = dict(
            num_steps=8,
            stochasticity_ratio=0.0,
            sample_type="transition",
            cfg_scale=1.0,
        ),
        ret_trajectory: bool = False,
    ):
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
        # Form the output
        out = dict(
            latent=enc_out["latent"],
            sem_tokens=dec_out["sem_tokens"],
            recon=dec_out["recon"],
            flow_loss=dec_out["losses"]["flow_loss"],
            mask=enc_out["mask"],
        )
        return out

    def load_pretrained(self, path: str):
        missing_keys, unexpected_keys = load_weights_with_shape_check(self, path)
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
        for module in self.children():
            if hasattr(module, "set_grad_checkpointing"):
                module.set_grad_checkpointing(enable)  # type: ignore
                logger.info(
                    f"Set grad_checkpointing={enable} for {module.__class__.__name__}"
                )

    @classmethod
    def create_model(cls, cfg):
        """Create a FlowTokenizer model from configuration."""
        # Update the defaults
        cfg = OmegaConf.merge(tokenizer_cfg_default, cfg)
        model = cls(cfg)
        return model


# * --- Test --- #


def test_flow_tokenizer():
    from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count_table

    tokenizer = FlowTokenizer(tokenizer_cfg_default).to("cuda", torch.bfloat16)
    # print(flop_count_table(FlopCountAnalysis(tokenizer, x)))
    print(parameter_count_table(tokenizer))

    x = torch.randn(2, 3, 512, 512).to("cuda", torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            out = tokenizer(x)
        for k, v in out.items():
            print("{:>20}: {}".format(k, v))

    # Sample
    with torch.no_grad():
        tokenizer.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = tokenizer(
                x,
                dec_mode="sample",
                clamp=True,
                ema_model=None,
                ret_trajectory=False,
            )
        recon = out["recon"]
        print(
            f"Sampled recon shape: {recon.shape}, min: {recon.min()}, max: {recon.max()}"
        )  # type: ignore

    # Backward
    tokenizer.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = tokenizer(x)
    loss = out["flow_loss"]
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
    cfg = flow_head_cfg

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
    inp_shape = (batch_size, 3, img_size, img_size)

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
    LOVELY_TENSORS=1 python -m src.stage1.cosmos.cosmos_mingtok
    """
    import sys

    with logger.catch():
        # test_decoder_flow_head()
        test_flow_tokenizer()

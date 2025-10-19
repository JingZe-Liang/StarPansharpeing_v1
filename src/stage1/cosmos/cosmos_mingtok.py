import math
from functools import partial
from typing import List, Optional

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
from .modules.uvit_decoder import UViTDecoder, UViTDecoderConfig

LossOutput = TypedDict(
    "LossOutput",
    {"flow_loss": torch.Tensor},
)
DecoderOutput: TypeAlias = tuple[Tensor, LossOutput]
EncoderSemOutput = TypedDict(
    "EncoderSemOutput",
    {
        "latent": Tensor,  # low-level latent tokens
        "low_lvl_proj_out": Tensor,  # projected low-level tokens for repa dino loss
        "sem_tokens": Tensor,  # semantic tokens
        "sem_proj_out": Tensor,  # projected semantic tokens for repa siglip loss
        "mask": Tensor | None,  # the mask used in low-level encoder
    },
)


"""
{
"pretrained_checkpoint": "",
"low_level_encoder": {
    "img_size": 512,
    "patch_size": 32,
    "depth": 12,
    "embed_dim": 768,
    "ffn_layer": "swiglufused",
    "out_dim": 32
},
"semantic_decoder": {
    "in_dim": 32,
    "patch_size": 32,
    "embed_dim": 1024,
    "decoder_depth": 24,
    "ffn_layer":"swiglufused"
},
"pixel_decoder": {
    "patch_size": 16,
    "decoder_depth": 24,
    "norm_pix_loss": true,
    "embed_dim": 1024,
    "loss_type": "L1-plain"
},
"scaling_factor": 8.09449291,
"mean": 1.46817409
}
"""


encoder_cfg_default_str: str = (
    "in_chan=3 embed_dim=512 depth=12 patch_size=16 out_patch_size=1 mlp_ratio=4.0 "
    "norm_layer='rmsnorm' drop_path=0.1 pe_type='learned' rope_kwargs={} "
    "last_norm='rmsnorm' z_dim=256 latent_dim=16 img_size=512 head='linear' "
    "n_reg_tokens=0 mask_ratio_train=0.0 low_level_proj_dim=1024 attn_type='sdpa'"
)
encoder_cfg_default = OmegaConf.from_dotlist(encoder_cfg_default_str.split(" "))

decoder_cfg_default_str: str = (
    "in_chan=16 embed_dim=512 depth=12 patch_size=1 mlp_ratio=4.0 "
    "norm_layer='rmsnorm' drop_path=0.1 pe_type='learned' rope_kwargs={} "
    "last_norm='rmsnorm' out_chan=512 img_size=32 head='linear' "
    "n_reg_tokens=0 mask_ratio_train=0.0 semantic_proj_dim=1024 is_causal=False attn_type='sdpa'"
)
decoder_cfg_default = OmegaConf.from_dotlist(decoder_cfg_default_str.split(" "))

pix_flow_cfg_default_str: str = (
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
    "transport.sigma_d=True "
    "transport.enhance_target=False "
)
pix_flow_cfg_default = OmegaConf.from_dotlist(pix_flow_cfg_default_str.split(" "))

tokenizer_cfg_default = OmegaConf.create(
    {
        # encoder, semantic decoder and pixel decoder configs
        "low_level_encoder": encoder_cfg_default,
        "semantic_decoder": decoder_cfg_default,
        "pixel_decoder": pix_flow_cfg_default,
    }
)

# *==============================================================
# * Model
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
        self.low_level_proj_out = nn.Linear(cfg.latent_dim, cfg.low_level_proj_dim)

    def forward(self, x):
        z, out = self.encoder(x, ret_2d_tokens=False, ret_all=True)
        h = self.z_to_latent(z)
        proj_h = self.low_level_proj_out(h)
        return dict(latent=h, z=z, low_lvl_proj_out=proj_h, **out)


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
        self.semantic_proj_out = nn.Linear(cfg.out_chan, cfg.semantic_proj_dim)

    def forward(self, x):
        """
        Encode the latent low-level tokens and convert into semantic tokens.
        """
        h, out = self.encoder(x, ret_2d_tokens=False, ret_all=True)
        proj_h = self.semantic_proj_out(h)
        return dict(sem_tokens=h, sem_proj_out=proj_h, **out)


class DecoderUViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        t_res = cfg.total_resolutions
        img_size = cfg.img_size
        grid_size = img_size // t_res

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
        mode: Literal["step", "loop"] = "step",
        clamp=False,
        ema_model: Optional[Self] = None,
        sample_kwargs: dict = dict(
            num_steps=8,
            stochasticity_ratio=0.0,
            sample_type="transition",
            cfg_scale=1.0,
        ),
        ret_trajectory=False,
    ):
        # x is the input noise latent
        if h.ndim == 3:  # bs, l, c
            if isinstance(inp_shape, (torch.Size, tuple, list)) and len(inp_shape) == 4:
                gh, gw = inp_shape[2:]
            else:
                # Assume is square
                gh = gw = int(h.shape[1] ** 0.5)
            h = rearrange(h, "b (h w) c -> b c h w", h=gh, w=gw)

        # Decoder channels
        chan = (
            inp_shape[1]
            if isinstance(inp_shape, (torch.Size, tuple, list))
            else inp_shape
        )
        null_cond_h = self.decoder.null_cond_h if hasattr(self.decoder, "null_cond_h") else None  # fmt: skip

        # Decode using the CNN decoder
        flow_loss, loss_dict = None, {}
        if mode == "step":
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
        elif mode == "loop":
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
        self.low_cfg = cfg.low_level_encoder
        self.sem_cfg = cfg.semantic_decoder
        self.pix_flow_cfg = cfg.pixel_decoder
        super().__init__()

        self.low_level_encoder = EncoderLowLevel(self.low_cfg)
        self.semantic_decoder = DecoderSemantic(self.sem_cfg)
        self.pixel_decoder = DecoderUViT(self.pix_flow_cfg)
        self.total_resolutions = self.pix_flow_cfg.total_resolutions

        # TODO: add quantizer

    @staticmethod
    def _to_2d(x, hw: List[int]) -> None | Tensor:
        if x is None:
            return x
        else:
            return rearrange(x, "b (h w) ... -> b ... h w", h=hw[0], w=hw[1])

    def _encode_latent(self, x):
        low_lvl_out = self.low_level_encoder(x)
        return low_lvl_out

    def _sem_decode(self, latent):
        sem_out = self.semantic_decoder(latent)
        return sem_out

    def _encode_to_sem(self, x=None, latent=None, hw: Sequence | None = None):
        mask = None
        if x is not None:
            hw = torch.as_tensor(x.shape[-2:]) // self.total_resolutions
            hw = hw.tolist()
            low_lvl_out = self._encode_latent(x)
            latent = low_lvl_out["latent"]
            low_lvl_proj_out = low_lvl_out["low_lvl_proj_out"]
            mask = low_lvl_out["mask"]
        elif latent is not None:
            assert latent.ndim == 3, "latent must be of shape (bs, n, dim)"
            hw_l = latent.shape[1]
            if hw is None:
                hw = to_2tuple(int(math.sqrt(hw_l)))
            low_lvl_proj_out = self.low_level_encoder.low_level_proj_out(latent)
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
        latent, low_lvl_proj_out, sem_tokens, sem_proj_out, mask = map(
            partial(self._to_2d, hw=hw),
            (latent, low_lvl_proj_out, sem_tokens, sem_proj_out, mask),
        )
        out: EncoderSemOutput = dict(
            latent=latent,
            low_lvl_proj_out=low_lvl_proj_out,
            sem_tokens=sem_tokens,
            sem_proj_out=sem_proj_out,
            mask=mask,
        )
        return out

    def encode(self, x):
        out = self._encode_latent(x)
        return out

    def decode(
        self,
        x: torch.Tensor,
        latent: torch.Tensor,
        inp_shape: Annotated[Union[torch.Size, int, tuple], "bs,c,h,w or bs,c or c"],
        mode: Literal["step", "loop"] = "step",
        clamp=False,
        ema_model: Optional[Self] = None,
        sample_kwargs: dict = dict(
            num_steps=8,
            stochasticity_ratio=0.0,
            sample_type="transition",
            cfg_scale=1.0,
        ),
        ret_trajectory=False,
    ):
        """
        Low level encoded latent is decoded into semantic tokens, and then decoded into the reconstructed image or
        velocity. Output dict of losses, recon, semantic tokens and projected semantic tokens.
        """
        ema_decoder = ema_model.pixel_decoder if ema_model is not None else None
        hw = (torch.as_tensor(x.shape[-2:]) // self.total_resolutions).tolist()
        sem_decoder_out = self._encode_to_sem(latent=latent, hw=hw)
        h_sem = sem_decoder_out["sem_tokens"]
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
            sem_proj_out=sem_decoder_out["sem_proj_out"],
        )
        return output

    def forward(
        self,
        input: torch.Tensor,
        dec_mode: Literal["step", "loop"] = "step",
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
            low_lvl_proj_out=enc_out["low_lvl_proj_out"],
            sem_tokens=dec_out["sem_tokens"],
            sem_proj_out=dec_out["sem_proj_out"],
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

    # # Sample
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = tokenizer(
                x,
                dec_mode="loop",
                clamp=True,
                ema_model=None,
                ret_trajectory=False,
            )
        recon = out["recon"]
        print(
            f"Sampled recon shape: {recon.shape}, min: {recon.min()}, max: {recon.max()}"
        )  # type: ignore

    # Backward
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


if __name__ == "__main__":
    """
    LOVELY_TENSORS=1 python -m src.stage1.cosmos.cosmos_mingtok
    """
    with logger.catch():
        test_flow_tokenizer()

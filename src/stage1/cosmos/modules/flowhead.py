import ast
import math
import random
from typing import Any, Literal, Optional, Self

import torch
import torch.nn as nn
from einops import rearrange, repeat
from timm.layers.weight_init import lecun_normal_
from timm.models.vision_transformer import named_apply
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

# Flow matching and tim transition
from src.utilities.transport.flow_matching.transport import Sampler, Transport
from src.utilities.transport.tim.transition import (
    TransitionSchedule,
    get_delta_time_embed,
)
from src.utilities.transport.tim.transports import OT_FM
from src.utilities.transport.tim.transports import Transport as TimTransport

from .blocks import AdaptiveInputLinearLayer, AdaptiveOutputLinearLayer
from .patching import (
    AdaptiveProgressivePatchEmbedding,
    AdaptiveProgressivePatchUnembedding,
)


def default(x, y):
    if x is not None:
        return x
    else:
        return y


def build_flow_matching_transport(
    flow_kwargs: dict | None = None,
) -> tuple[Transport, Sampler]:
    default_flow_kwargs = dict(
        transport=dict(
            model_type="velocity",
            path_type="linear",
            loss_type="velocity",
            train_eps=0.001,
            sample_eps=0.001,
        ),
        sampler=dict(time_type="linear"),
    )
    flow_kwargs = default(flow_kwargs, default_flow_kwargs)
    transport = Transport(**flow_kwargs.get("transport", {}))
    sampler = Sampler(transport=transport, **flow_kwargs.get("sampler", {}))
    return transport, sampler


def build_tim_scheduler(tim_kwargs: dict | None = None):
    default_tim_kwargs = dict(
        num_steps=8,
        stochasticity_ratio=0.1,
        sample_type="transition",
        cfg_scale=2.0,
        cfg_low=0.0,
        cfg_high=0.7,
    )
    tim_kwargs = default(tim_kwargs, default_tim_kwargs)
    transition_schedule = TransitionSchedule(
        transport=TimTransport(**tim_kwargs.get("transport", {})),
        **tim_kwargs.get("transition_schedule", {}),
    )
    return transition_schedule


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def is_sequence_shape(shape: Any) -> bool:
    return isinstance(shape, (torch.Size, list, tuple))


def get_chan_from_shape(shape: torch.Size | tuple | list | int) -> int:
    if is_sequence_shape(shape):
        return shape[1]
    else:
        return shape


def l2p_transform_tensor(x, patch_size, img_size):
    """
    Transform from latent space to pixel space
    [B, H//patch_size * W//patch_size, C*tubelet_size*patch_size*patch_size] -> [B, C, H, W]
    """
    B = x.shape[0]
    C = x.shape[2] // (patch_size * patch_size)
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    x = rearrange(
        x,
        "b (h w) (c hp wp) -> b c (h hp) (w wp)",
        h=img_size[0] // patch_size,
        hp=patch_size,
        w=img_size[1] // patch_size,
        wp=patch_size,
        c=C,
    )
    return x


class Unpatcher(nn.Module):
    """to 2d head: blc -> bchw"""

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        patch_size=14,
        img_size=224,
        head_type="once",
        module_by_time=False,
        **prog_kwargs,
    ):
        super().__init__()
        self.head_type = head_type
        self.patch_size = patch_size
        self.img_size = img_size

        if head_type == "progressive":
            assert "progressive_dims" in prog_kwargs, f"progressive_dims must be provided for progressive unpatching"
            self.unpatcher = AdaptiveProgressivePatchUnembedding(  # type: ignore
                in_chans=in_chans,
                out_chans=out_chans,
                patch_size=patch_size,
                adaptive_mode="interp",
                **prog_kwargs,
            )
        elif head_type == "once":
            self.unpatcher = nn.Linear(in_chans, out_chans * patch_size * patch_size)
        elif head_type == "once_adaptive":
            self.unpatcher = AdaptiveOutputLinearLayer(in_chans, out_chans, bias=True, mode="interp")
        else:
            raise NotImplementedError(f"head_type {head_type} not implemented")

        # Time modulation
        self.module_by_time = module_by_time
        if module_by_time:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(in_chans, 2 * in_chans, bias=True))

    def init_weights(self):
        # zero out?
        if self.head_type == "once":
            nn.init.zeros_(self.unpatcher.weight)
            if self.unpatcher.bias is not None:
                nn.init.zeros_(self.unpatcher.bias)
        else:
            self.unpatcher.init_weights(True)

    def forward(self, x_blc, c=None, out_shape: torch.Size | tuple | None = None):
        # time modulation
        if c is not None and self.module_by_time:
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
            x_blc = modulate(x_blc, shift, scale)

        if self.head_type[:4] == "once":
            # NOTE:
            # does not support dynamic out channels
            # if is a rgb image decoding, select this in most cases
            if self.head_type == "once_adaptive":
                assert out_shape is not None, "out_shape must be provided for once_adaptive unpatching"
            x_blc = self.unpatcher(x_blc)
            img_size = out_shape[-2:] if out_shape is not None else self.img_size
            return l2p_transform_tensor(
                x_blc,
                patch_size=self.patch_size,
                img_size=img_size,
            )
        else:  # is progressive
            assert out_shape is not None, "out_shape must be provided for progressive unpatching"
            return self.unpatcher(x_blc, out_shape=out_shape)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        # rescale t
        t = t * 1000
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 3 * channels, bias=True))

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """

    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, 2 * model_channels, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        grad_checkpointing=False,
        time_cond_type="t",
        first_lin_type="interp_lin",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        # times
        self.time_embed = TimestepEmbedder(model_channels)
        self.time_cond_type = time_cond_type
        self.use_delta_embed = time_cond_type in [
            "t-r",
            "r",
            "t,t-r",
            "r,t-r",
            "t,r,t-r",
        ]
        self.delta_t_embed = None
        if self.use_delta_embed:
            self.delta_t_embed = TimestepEmbedder(model_channels)

        # projections
        self.cond_embed = nn.Linear(z_channels, model_channels)
        if first_lin_type == "linear":
            self.input_proj = nn.Linear(in_channels, model_channels)
        else:
            self.input_proj = AdaptiveInputLinearLayer(in_channels, model_channels, mode=first_lin_type)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(model_channels))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.init_weights()

    def init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t: Tensor | tuple[Tensor, Tensor], c=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps or (t, r) in tim scheduling.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        x = self.input_proj(x)
        c = self.cond_embed(c)
        t = get_delta_time_embed(
            t,
            time_embedder=self.time_embed,
            delta_t_embedder=self.delta_t_embed,
            time_cond_type=self.time_cond_type,
        )

        y = t + c
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x, y), y


class FlowDecoder(nn.Module):
    """patch-wise pixel flow decoder (rectified flow)"""

    def __init__(
        self,
        target_channels,  # base channel
        z_channels,
        depth,
        width,
        grad_checkpointing=False,
        num_sampling_steps="10",
        time_cond_type="t",
        train_schedule="fat_lognormal",
        use_cfg=False,
        cfg_prob: float = 0.5,
        patch_size=14,
        img_size=224,
        head_type="once",
        head_kwargs: dict = {},  # progressive dims
        stand_alone=True,
        flow_kwargs: dict | None = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size

        # configs
        self.use_cfg = use_cfg
        self.train_schedule = train_schedule
        self.num_sampling_steps = int(num_sampling_steps)
        self.head_type = head_type
        self.tgt_chan_patched = target_channels * patch_size * patch_size
        self.cfg_prob = cfg_prob

        # mlp head (latent to pixel)
        self.in_channels = self.tgt_chan_patched
        self.net = SimpleMLPAdaLN(
            in_channels=self.tgt_chan_patched,
            model_channels=width,
            out_channels=width,  # self.tgt_chan_patched,
            z_channels=z_channels,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing,
            time_cond_type=time_cond_type,
            first_lin_type="interp",
        )
        self.head = Unpatcher(
            in_chans=width,
            out_chans=target_channels,
            patch_size=patch_size,
            img_size=img_size,
            head_type=head_type,
            **head_kwargs,
        )

        # Scheduler
        # is stand_alone, the flow decoder is used alone for training and sampling
        self.stand_alone = stand_alone
        if stand_alone:
            self.transport, self.sampler = build_flow_matching_transport()

    def _forward_fn(self, xt_bchw, t, z_blc):
        chan = xt_bchw.shape[1]
        img_size = xt_bchw.shape[-2:]

        # to bl_c
        z, x, (b, l, _) = self._to_bl_c(z_blc, xt_bchw)  # (b * l, c)

        # expand t
        t = repeat(t, "b -> (b l)", l=l)  # (b * l, )

        # model
        x_hidden_blc, y = self.net(x, t, z)  # time-dependent
        x_bchw = self.head(x_hidden_blc.reshape(b, l, -1), y, out_shape=(chan, *img_size))

        return x_bchw

    def _to_bl_c(self, z, x=None):
        b, l, c_z = z.shape
        z = z.reshape(b * l, c_z)

        if x is not None:
            # x should have the same shape as z but with chan channels
            # [b, c, h, w] -> [b*n, c]
            c, h, w = x.shape[-3:]
            x = rearrange(
                x,
                "b c (h nh) (w nw) -> (b h w) (c nh nw)",
                nh=self.patch_size,
                nw=self.patch_size,
            )
            assert x.shape[0] == b * l, f"Input x shape {x.shape} does not match z shape {z.shape}"
            assert x.ndim == 2, "Input x must be a 2D tensor [b*n, c]"
            # x = x.reshape(b * l, c)
        return z, x, (b, l, c_z)

    def _forward_only_model(
        self,
        z_blc,
        x_bchw=None,
        t=None,
        inp_shape: tuple | list | torch.Size | None = None,
        # mode: str = "train",
        # sample_kwargs={},
        **_kwargs,
    ) -> Tensor:
        ########### not stand alone, takes x, t, and z ###############
        # takes states from previous module

        assert not self.stand_alone, f"FlowDecoder must not be stand_alone to forward as a module"

        return self._forward_fn(x_bchw, t, z_blc)

    def _forward_stand_alone(
        self, z_blc, x_bchw=None, t=None, inp_shape=None, mode="train", sample_kwargs={}
    ) -> dict | Tensor:
        """
        x is 2d image: b, c, h, w
        z is 1d latent: b, l, c from transformer, l = h * w / (p ** 2)
        """

        ############# is stand alone decoder ###############
        # x, z: [b*n, c]
        if mode == "train":
            assert x_bchw is not None, "Input x must be provided in step mode"
            # is training
            zc = z_blc
            if self.use_cfg and random.random() < self.cfg_prob:
                zc = torch.zeros_like(zc)
            terms = self.transport.training_losses(
                self._forward_fn,
                x_bchw,
                model_kwargs={"z_blc": zc},
            )  # loss, pred_x_clean
            return terms
        else:
            # TODO: using sampler to sample
            assert inp_shape is not None, "inp_shape must be provided in sample mode"
            return self._manually_sample_loop(z_blc, inp_shape, **sample_kwargs)

    def forward(
        self,
        z_blc,
        x_bchw=None,
        t=None,
        inp_shape=None,
        mode="train",
        sample_kwargs={},
        **kwargs,
    ):
        """main forward function
        The decoder serves as model or flow decoder.
        """
        if self.stand_alone:
            return self._forward_stand_alone(
                z_blc,
                x_bchw=x_bchw,
                t=t,
                inp_shape=inp_shape,
                mode=mode,
                **kwargs,
            )
        else:
            return self._forward_only_model(
                z_blc,
                x_bchw=x_bchw,
                t=t,
                inp_shape=inp_shape,
                **kwargs,
            )

    @torch.no_grad()
    def _manually_sample_loop(
        self,
        z,
        inp_shape: tuple | list | torch.Size,
        sample_steps=None,
        schedule="linear",
        cfg=1.0,
        cfg_interval=None,
        tbar=False,
        **_kwargs,
    ):
        assert self.stand_alone, f"FlowDecoder must be stand_alone for sampling"

        b, n, c_z = z.shape
        c_x, h, w = inp_shape[-3:]
        assert h * w // (self.patch_size**2) == n

        sample_steps = sample_steps or self.num_sampling_steps

        # get all timesteps ts and intervals Δts
        if schedule == "linear":
            ts = torch.arange(1, sample_steps + 1).flip(0) / sample_steps  # (0 .. 1) -> (1 .. 0)
            dts = torch.ones_like(ts) * (1.0 / sample_steps)
        elif schedule.startswith("pow"):  # "pow_0.25"
            p = float(schedule.split("_")[1])
            ts = torch.arange(0, sample_steps + 1).flip(0) ** (  # type: ignore
                1 / p
            ) / sample_steps ** (1 / p)
            dts = ts[:-1] - ts[1:]
        else:
            raise NotImplementedError
        ts = 1 - ts  # (0 .. 1)

        # cfg interval
        if cfg_interval is None:  # cfg_interval = "(.17,1.02)"
            interval = None
        else:
            raise NotImplementedError(f"cfg_interval {cfg_interval} parsing not implemented")
            cfg_lo, cfg_hi = ast.literal_eval(cfg_interval)
            interval = (
                self._edm_to_flow_convention(cfg_lo),
                self._edm_to_flow_convention(cfg_hi),
            )

        # sampling (sample_steps) steps: noise X0 -> clean X1
        trajs = []
        x = torch.randn(b, c_x, h, w).to(z)  # noise start [b, c, h, w]
        x = x.to(z.dtype)

        null_z = z.clone() * 0.0 if cfg != 1.0 else None
        for i, (t, dt) in tqdm(
            enumerate((zip(ts, dts))),
            disable=not tbar,
            total=sample_steps,
            leave=False,  # type: ignore
        ):
            timesteps = torch.tensor([t] * b).to(z.device)

            # conditional velocity
            xc = x
            vc = self._forward_fn(xc, timesteps, z)  # conditional v

            # classifier free guidance
            if null_z is not None and (interval is None or ((t.item() >= interval[0]) and (t.item() <= interval[1]))):
                xu = x
                vu = self._forward_fn(xu, timesteps, null_z)  # unconditional v
                vc = vu + cfg * (vc - vu)

            # update x
            x = x + dt * vc
            trajs.append(x)

        sampled_image = trajs[-1]

        return sampled_image


class TimFlowDecoder(nn.Module):
    """Tim Flow Decoder with t,r time embeddings"""

    def __init__(
        self,
        target_channels,  # base channel
        z_channels,
        depth,
        width,
        grad_checkpointing=False,
        time_cond_type="t,t-r",
        use_cfg=False,
        cfg_prob: float = 0.5,
        patch_size=14,
        img_size=224,
        head_type="once",
        head_kwargs: dict = {},
        # tim flow matching kwargs
        fm_kwargs={},
        transition_schedule_kwargs={},
        stand_alone=True,
    ):
        super().__init__()
        self.stand_alone = stand_alone
        # decoder - always create as standalone=False for internal use
        self.flow_decoder = FlowDecoder(
            target_channels,
            z_channels,
            depth,
            width,
            grad_checkpointing,
            use_cfg=use_cfg,
            cfg_prob=cfg_prob,
            patch_size=patch_size,
            time_cond_type=time_cond_type,
            img_size=img_size,
            head_type=head_type,
            head_kwargs=head_kwargs,
            stand_alone=False,  # FlowDecoder is always used as module inside TimFlowDecoder
            num_sampling_steps="0",  # No sampling steps needed for internal module
        )
        # No need to delete sampler/transport as they won't be created with stand_alone=False

        # FM transport and scheduler
        if stand_alone:
            self.transport = OT_FM(**fm_kwargs)
            self.transition_schedule = TransitionSchedule(
                self.transport,
                **transition_schedule_kwargs,
            )
            self.null_cond_h = nn.Parameter(torch.zeros(1, 1, z_channels))
        else:
            self.transport = None
            self.transition_schedule = None
            self.null_cond_h = None

    def _forward_only_model(
        self,
        z_blc,
        x_bchw=None,
        t=None,
        r=None,
        inp_shape: tuple | list | torch.Size | None = None,
        **_kwargs,
    ) -> Tensor:
        """Forward as a module without transport (for non-standalone mode)"""
        assert not self.stand_alone, f"TimFlowDecoder must not be stand_alone to forward as a module"

        if x_bchw is None:
            raise ValueError("x_bchw must be provided for module forward")
        if t is None or r is None:
            raise ValueError("t and r must be provided for module forward")

        chan = x_bchw.shape[1]
        img_size = x_bchw.shape[-2:]

        # to bl_c
        z, x, (b, l, _) = self.flow_decoder._to_bl_c(z_blc, x_bchw)  # (b * l, c)

        # expand t and r
        t = repeat(t, "b -> (b l)", l=l)  # (b * l, )
        r = repeat(r, "b -> (b l)", l=l)  # (b * l, )

        x_hidden_blc, zt_cond = self.flow_decoder.net(x, (t, r), z)  # time-dependent
        x_bchw = self.flow_decoder.head(
            x_hidden_blc.reshape(b, l, -1),
            zt_cond if zt_cond is not None else z,
            out_shape=(chan, *img_size),
        )

        return x_bchw

    def forward(
        self,
        z_blc,
        x_bchw=None,
        t=None,
        r=None,
        inp_shape=None,
        mode="module",
        **kwargs,
    ):
        """Main forward function for TimFlowDecoder

        In standalone mode: supports training_loss and sample methods
        In non-standalone mode: acts as a simple module via _forward_only_model
        """
        assert self.stand_alone, (
            "TimFlowDecoder in stand_alone mode should use training_loss() or sample() methods, not forward()"
        )
        # In non-standalone mode, act as a simple module
        return self._forward_only_model(z_blc, x_bchw=x_bchw, t=t, r=r, inp_shape=inp_shape, **kwargs)

    def training_loss(
        self,
        z_blc,  # condition
        x_bchw,  # clean image
        inp_shape: tuple | list | torch.Size | None = None,
        ema_model: Optional[Self] = None,
        clamp: bool = False,
    ):
        """Training loss computation (only available in standalone mode)"""
        assert self.stand_alone, (
            "training_loss should only be called in stand_alone mode. Use forward() method for module inference."
        )
        assert self.transition_schedule is not None
        assert self.null_cond_h is not None

        if self.transition_schedule.transport.enhance_target:
            assert ema_model is not None, "EMA model is required for enhance_target"
            assert hasattr(self, "null_cond_h"), "The decoder must have null_cond_h for CFG"
        assert x_bchw is not None, "Input x must be provided in step mode"

        noise = torch.randn_like(x_bchw)
        bs, l = z_blc.shape[:2]
        z_null_cond = self.null_cond_h.expand(bs, l, -1)
        flow_loss, _, loss_dict, breakdowns = self.transition_schedule(
            self,
            ema_model if ema_model is not None else None,
            self,
            batch_size=z_blc.shape[0],
            x=x_bchw,
            z=noise,
            model_kwargs={"inp_shape": inp_shape, "z_blc": z_blc},
            ema_kwargs={"inp_shape": inp_shape, "z_blc": z_null_cond},
        )
        recon = breakdowns["x0_pred"]
        if clamp:
            recon = recon.clamp(-1, 1)

        losses = {"flow_loss": flow_loss}
        return recon, losses

    @torch.no_grad()
    def sample(
        self,
        z_blc,
        inp_shape: torch.Size | tuple,
        clamp=False,
        sample_kwargs: dict = dict(
            num_steps=8,
            stochasticity_ratio=0.0,
            sample_type="transition",
            cfg_scale=1.0,
            progress_bar=True,
        ),
        ret_trajectory=False,
    ):
        """Sampling (only available in standalone mode)"""
        assert self.stand_alone, (
            "sample should only be called in stand_alone mode. Use forward() method for module inference."
        )

        x_init = torch.randn(inp_shape).to(z_blc)
        null_cond_h = self.null_cond_h.expand(*z_blc.shape[:2], -1)
        recon: Tensor = self.transition_schedule.sample(
            self,
            # start sampling noise
            z=x_init,
            # conditions
            y=z_blc,
            y_null=null_cond_h,  # since no CFG, y_null is None
            T_max=1.0,
            **sample_kwargs,
        )
        if not ret_trajectory:
            recon = recon[-1]

        if clamp:
            recon = recon.clamp(-1, 1)

        return recon


##### Test


def test_flow_head():
    flow_head = FlowDecoder(
        3,
        16,
        depth=2,
        width=768,
        num_sampling_steps="10",
        patch_size=16,
        head_type="progressive",
        head_kwargs={"progressive_dims": [256, 128, 64]},
    )

    # 16 * 16 = 256
    # output shape: [2, 3, 256, 256]
    x = torch.randn(2, 3, 256, 256)
    z = torch.randn(2, 256, 16)
    out = flow_head(z, x, inp_shape=x.shape, mode="step")
    print(out["loss"])

    # sampling
    z = torch.randn(2, 256, 16)
    with torch.no_grad():
        sampled = flow_head(
            z,
            x_bchw=None,
            inp_shape=x.shape,
            mode="sample",
            sample_kwargs={
                "schedule": "linear",
                "cfg": 3.0,
            },
        )
        print(sampled)


def test_tim_flow_head():
    # Test standalone mode (default)
    flow_head_standalone = TimFlowDecoder(
        3,
        16,
        depth=6,
        width=768,
        patch_size=16,
        head_type="progressive",
        head_kwargs={"progressive_dims": [256, 128, 64]},
        stand_alone=True,
    )

    # 16 * 16 = 256
    # output shape: [2, 3, 256, 256]
    x = torch.randn(2, 3, 256, 256)
    z = torch.randn(2, 256, 16)
    recon, loss = flow_head_standalone.training_loss(z, x, x.shape, None, False)
    print("Standalone mode - recon shape:", recon.shape, "loss:", loss)

    # sample in standalone mode
    z = torch.randn(2, 256, 16)
    with torch.no_grad():
        sampled = flow_head_standalone.sample(
            z,
            inp_shape=x.shape,
            clamp=False,
            sample_kwargs={
                "num_steps": 20,
                "stochasticity_ratio": 0.0,
                "sample_type": "transition",
                "cfg_scale": 3.0,
                "progress_bar": True,
            },
            ret_trajectory=False,
        )
        print("Standalone mode - sampled shape:", sampled.shape)

    # Test non-standalone mode (module mode)
    flow_head_module = TimFlowDecoder(
        3,
        16,
        depth=6,
        width=768,
        patch_size=16,
        head_type="progressive",
        head_kwargs={"progressive_dims": [256, 128, 64]},
        stand_alone=False,
    )

    # Test module forward (should work)
    t = torch.randn(2)
    r = torch.randn(2)
    try:
        output = flow_head_module(z, x, t, r)  # Updated interface
        print("Module mode - forward output shape:", output.shape)
    except Exception as e:
        print("Module mode - forward error:", e)

    # Test that forward raises error in standalone mode
    try:
        flow_head_standalone(z, x, t, r)
        print("ERROR: forward should have raised RuntimeError in standalone mode")
    except RuntimeError as e:
        print("Standalone mode - forward correctly raised:", e)

    # Test that training_loss raises error in non-standalone mode
    try:
        flow_head_module.training_loss(z, x, x.shape, None, False)
        print("ERROR: training_loss should have raised RuntimeError in module mode")
    except RuntimeError as e:
        print("Module mode - training_loss correctly raised:", e)

    # Test that sample raises error in non-standalone mode
    try:
        flow_head_module.sample(z, inp_shape=x.shape)
        print("ERROR: sample should have raised RuntimeError in module mode")
    except RuntimeError as e:
        print("Module mode - sample correctly raised:", e)


if __name__ == "__main__":
    """
    LOVELY_TENSORS=1 python -m src.stage1.cosmos.modules.flowhead
    """
    from loguru import logger

    with logger.catch():
        # test_flow_head()
        test_tim_flow_head()

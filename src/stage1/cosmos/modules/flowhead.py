import ast
import math
import random

import torch
import torch.nn as nn
from einops import rearrange, repeat
from timm.layers.weight_init import lecun_normal_
from timm.models.vision_transformer import named_apply
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from src.utilities.transport.flow_matching.transport import Sampler, Transport

from .patching import (
    AdaptiveProgressivePatchEmbedding,
    AdaptiveProgressivePatchUnembedding,
)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


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


class FlowDecoderHead(nn.Module):
    """to 2d head: blc -> bchw"""

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        patch_size=14,
        img_size=224,
        head_type="once",
        **prog_kwargs,
    ):
        super().__init__()
        self.head_type = head_type
        self.patch_size = patch_size
        self.img_size = img_size

        # TODO: add time embedding like FinalLayer
        if head_type == "progressive":
            self.unpatcher = AdaptiveProgressivePatchUnembedding(
                in_chans=in_chans,
                out_chans=out_chans,
                patch_size=patch_size,
                **prog_kwargs,
            )
        else:
            self.unpatcher = nn.Linear(in_chans, out_chans * patch_size * patch_size)
            # self.unpatcher = FinalLayer(in_chans, out_chans * patch_size * patch_size)

    def init_weights(self):
        # zero out?
        if self.head_type == "once":
            nn.init.zeros_(self.unpatcher.weight)
            if self.unpatcher.bias is not None:
                nn.init.zeros_(self.unpatcher.bias)
        else:
            self.unpatcher.init_weights(True)

    def forward(self, x_blc, out_shape: torch.Size | tuple | None = None):
        if self.head_type == "once":
            # NOTE:
            # does not support dynamic out channels
            # if is a rgb image decoding, select this in most cases
            x_blc = self.unpatcher(x_blc)
            img_size = out_shape[-2:] if out_shape is not None else self.img_size
            return l2p_transform_tensor(
                x_blc,
                patch_size=self.patch_size,
                img_size=img_size,
            )
        else:  # is progressive
            assert out_shape is not None, (
                "out_shape must be provided for progressive unpatching"
            )
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
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            self.mlp[0].weight.dtype
        )
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

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(channels, 3 * channels, bias=True)
        )

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
        self.norm_final = nn.LayerNorm(
            model_channels, elementwise_affine=False, eps=1e-6
        )
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

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
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(model_channels))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
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

    def forward(self, x, t, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)

        y = t + c

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x, y)


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
        train_schedule="fat_lognormal",
        use_cfg=False,
        patch_size=14,
        img_size=224,
        head_type="once",
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

        # mlp head (latent to pixel)
        self.in_channels = self.tgt_chan_patched
        self.net = SimpleMLPAdaLN(
            in_channels=self.tgt_chan_patched,
            model_channels=width,
            out_channels=width,  # self.tgt_chan_patched,
            z_channels=z_channels,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing,
        )
        self.head = FlowDecoderHead(
            in_chans=width,
            out_chans=target_channels,
            patch_size=patch_size,
            img_size=img_size,
            head_type=head_type,
        )

        # Scheduler
        self.transport = Transport(
            model_type="velocity",
            path_type="linear",
            loss_type="velocity",
            train_eps=0.001,
            sample_eps=0.001,
        )
        self.sampler = Sampler(transport=self.transport)

    def _forward_fn(self, xt_bchw, t, z_blc):
        chan = xt_bchw.shape[1]
        img_size = xt_bchw.shape[-2:]

        # to bl_c
        z, x, (b, l, _) = self._to_bl_c(z_blc, xt_bchw)  # (b * l, c)
        # expand t
        t = repeat(t, "b -> (b l)", l=l)  # (b * l, )
        x_hidden_blc = self.net(x, 1000 * t, z)  # time-dependent
        x_bchw = self.head(x_hidden_blc.reshape(b, l, -1), out_shape=(chan, *img_size))

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
            assert x.shape[0] == b * l, (
                f"Input x shape {x.shape} does not match z shape {z.shape}"
            )
            assert x.ndim == 2, "Input x must be a 2D tensor [b*n, c]"
            # x = x.reshape(b * l, c)
        return z, x, (b, l, c_z)

    def forward(
        self,
        z_blc,
        x_bchw=None,
        inp_shape: tuple | list | torch.Size | None = None,
        mode: str = "step",
        sample_kwargs={},
    ) -> dict | Tensor:
        # x is 2d image: b, c, h, w
        # z is 1d latent: b, l, c from transformer, l = h * w / (p ** 2)

        # x, z: [b*n, c]
        if mode == "step":
            assert x_bchw is not None, "Input x must be provided in step mode"
            # is training
            zc = z_blc
            if self.use_cfg and random.random() < 0.5:
                zc = torch.zeros_like(zc)
            terms = self.transport.training_losses(
                self._forward_fn,
                x_bchw,
                model_kwargs={"z_blc": zc},
            )  # loss, pred_x_clean
            return terms
        else:
            # self.sampler.sample_ode(
            #     sampling_method="Euler",
            #     num_steps=10,
            #     clip_for_x1_pred=True,
            #     progress=False,
            # )
            assert inp_shape is not None, "inp_shape must be provided in sample mode"
            return self.sample(z_blc, inp_shape, **sample_kwargs)

    @torch.no_grad()
    def sample(
        self,
        z,
        inp_shape: tuple | list | torch.Size,
        schedule="linear",
        cfg=1.0,
        cfg_interval=None,
        tbar=False,
    ):
        b, n, c_z = z.shape
        c_x, h, w = inp_shape[-3:]
        assert h * w // (self.patch_size**2) == n

        sample_steps = self.num_sampling_steps

        # get all timesteps ts and intervals Δts
        if schedule == "linear":
            ts = (
                torch.arange(1, sample_steps + 1).flip(0) / sample_steps
            )  # (0 .. 1) -> (1 .. 0)
            dts = torch.ones_like(ts) * (1.0 / sample_steps)
        elif schedule.startswith("pow"):  # "pow_0.25"
            p = float(schedule.split("_")[1])
            ts = torch.arange(0, sample_steps + 1).flip(0) ** (
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
            raise NotImplementedError(
                f"cfg_interval {cfg_interval} parsing not implemented"
            )
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
            enumerate((zip(ts, dts))), disable=not tbar, total=sample_steps
        ):
            timesteps = torch.tensor([t] * b).to(z.device)

            # conditional velocity
            xc = x
            vc = self._forward_fn(xc, timesteps, z)  # conditional v

            # classifier free guidance
            if null_z is not None and (
                interval is None
                or ((t.item() >= interval[0]) and (t.item() <= interval[1]))
            ):
                xu = x
                vu = self._forward_fn(xu, timesteps, null_z)  # unconditional v
                vc = vu + cfg * (vc - vu)

            # update x
            x = x + dt * vc
            trajs.append(x)

        sampled_image = trajs[-1]

        return sampled_image


def test_flow_head():
    flow_head = FlowDecoder(
        3, 16, depth=2, width=768, num_sampling_steps="10", patch_size=16
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


if __name__ == "__main__":
    """
    LOVELY_TENSORS=1 python -m src.stage1.cosmos.modules.flowhead
    """
    from loguru import logger

    with logger.catch():
        test_flow_head()

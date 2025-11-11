import inspect
import random
import warnings
from collections import OrderedDict, namedtuple
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any, Literal, Optional, no_type_check

import accelerate
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_config,
    get_peft_model,
    set_peft_model_state_dict,
)
from torch import Tensor, nn
from typing_extensions import Annotated

import src.stage1.cosmos.modules.blocks as cosmos_block
from src.stage1.cosmos.inference.utils import load_jit_model_shape_matched
from src.stage1.cosmos.modules.layers2d import Decoder, Encoder, GenerativeDecoder
from src.stage1.cosmos.modules.proj import build_mlp
from src.stage1.cosmos.modules.utils import Normalize

# Quantizers
from src.stage1.discretization.collections import FSQ
from src.stage1.discretization.collections import BinarySphericalQuantizer as BSQ
from src.stage1.discretization.collections.kl_continuous import (
    DiagonalGaussianDistributionV2 as DiagonalGaussianDistribution,
)
from src.stage1.discretization.collections.psd import (
    PowerSphericalDistribution,
    l2_norm,
)
from src.utilities.config_utils import (
    dataclass_from_dict,
    function_config_to_basic_types,
)
from src.utilities.config_utils.to_dataclass import dataclass_from_dict_config
from src.utilities.logging import catch_any, log_print
from src.utilities.network_utils import load_weights_with_shape_check

KLLossBreakDown = namedtuple("KLLossBreakDown", ["posterior", "mean", "logvar"])


# * --- Latent augmentation --- #


class NestChannelDrop(nn.Module):
    def __init__(
        self,
        drop_type: str | list[int] = "uniform_4",
        max_channels: int = 16,
        drop_prob: float = 0.5,
    ):
        super().__init__()
        self.max_channels = max_channels
        self.drop_prob = drop_prob

        if isinstance(drop_type, str):
            drop_type, args = drop_type.lower().split("_")
            self.drop_type = drop_type
            if drop_type == "exp":
                self.sample_kwargs = {"lambda": float(args)}
            elif drop_type == "uniform":
                assert args.isdigit(), "args should be an int"
                self.sample_kwargs = {"low": int(args)}
            else:
                raise ValueError(
                    f"drop_type {drop_type} not supported, only exp and uniform are supported"
                )
        else:  # list
            self.drop_list = drop_type
            assert max(self.drop_list) < self.max_channels, (
                f"max_channels {self.max_channels} should be larger than the max of drop_list {max(self.drop_list)}"
            )
            self.drop_type = "prefixed"
        log_print(
            f"[NestChannelDrop]: drop_type={self.drop_type}, max_channels={self.max_channels}, drop_prob={self.drop_prob}"
        )

        self.channel_arange = nn.Buffer(
            torch.arange(self.max_channels, dtype=torch.int32), persistent=False
        )

    def exponential_sampling(self, lambda_val, size=1):
        u = np.random.uniform(size=size)
        k = -np.log(1 - u) / lambda_val
        return (
            torch.as_tensor(np.floor(k).astype(int))
            .clip_(0, self.max_channels)
            .unsqueeze(-1)
        )

    def uniform_sampling(self, low: int, size: int = 1):
        # (bs, 1)
        k = torch.randint(low=low, high=self.max_channels, size=(size, 1))
        return k

    def prefixed_sampling(self, size: int = 1):
        drop_list = self.drop_list
        leave_channels = np.random.choice(drop_list, size=size, replace=True)
        return torch.as_tensor(leave_channels).unsqueeze(-1)

    def forward(
        self, z, inference_channels: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Apply channel drop to the input tensor.

        Args:
            z: Input tensor of shape (bs, c, h, w)
            inference_channels: Number of channels to keep during inference

        Returns:
            If applying channel drop: tuple of (z, mask) where mask is the channel mask
            If not applying channel drop: z tensor only
        """
        if (self.training and np.random.random() > self.drop_prob) or (
            not self.training and inference_channels is None
        ):
            return z

        if inference_channels is not None:
            assert not self.training
            assert inference_channels <= self.max_channels
            return z[:, :inference_channels]

        assert self.max_channels == z.shape[1]

        bs = z.shape[0]
        if self.drop_type == "exp":
            leave_channels = self.exponential_sampling(size=bs, **self.sample_kwargs)
        elif self.drop_type == "uniform":
            leave_channels = self.uniform_sampling(size=bs, **self.sample_kwargs)  # type: ignore
        elif self.drop_type == "prefixed":
            leave_channels = self.prefixed_sampling(size=bs)
        else:
            raise ValueError(
                f"drop_type {self.drop_type} not supported, only exp and uniform are supported"
            )

        # drop channels

        # 1. expand the cached empty z
        # if self.dropped_x.shape[-2:] != z.shape[-2:]:
        #     if self.learnable:
        #         z_empty = nn.functional.interpolate(
        #             self.dropped_x,
        #             size=z.shape[-2:],
        #             mode="bilinear",
        #             align_corners=False,
        #         )
        #         z_empty = z_empty.expand(bs, -1, -1, -1)
        #     else:
        #         z_empty = torch.zeros_like(z)
        # else:
        #     z_empty = self.dropped_x.expand(bs, -1, -1, -1)
        z_zeros = torch.zeros_like(z)

        # 2. drop channels
        channels = self.channel_arange[None].expand(bs, -1)  # type: ignore
        cond = channels < leave_channels.to(channels)
        mask = cond.unsqueeze(-1).unsqueeze(-1).expand_as(z)  # (bs, c, h, w)
        z = torch.where(mask, z, z_zeros)

        return z, mask


# * --- Network utilities --- #


class MultiInputSequential(nn.Sequential):
    def forward(self, input: tuple[torch.Tensor, ...] | torch.Tensor):
        # if input is a tuple, the first element is changed sequentially by module
        # and the last n-1 elements are unchanged for those modules taken not only one input

        out = input
        for module in self:
            _is_multi_input = self.check_if_multi_inputs(module)
            _inp = out if _is_multi_input else (out, *input[1:])
            out = module(_inp)
        return input

    @staticmethod
    def check_if_multi_inputs(module):
        forward_fn = module.forward
        sig = inspect.signature(forward_fn)
        params = list(sig.parameters.values())
        # Exclude 'self'
        params = [p for p in params if p.name != "self"]
        has_var_positional = any(
            p.kind == inspect.Parameter.VAR_POSITIONAL for p in params
        )

        # If more than one parameter or the first parameter is annotated as a tuple, treat as multi-input
        if len(params) > 1 and has_var_positional:
            return True
        if params and (
            params[0].annotation in (tuple, list)
            or (
                hasattr(params[0].annotation, "__origin__")
                and params[0].annotation.__origin__ is tuple
            )
        ):
            return True
        return False


class DecoderSequential(nn.Module):
    def __init__(self, quant_conv, decoder):
        super().__init__()
        self.quant_conv = quant_conv
        self.decoder = decoder

    def __getitem__(self, item):
        if item == 0:
            return self.quant_conv
        elif item == 1:
            return self.decoder
        else:
            raise IndexError(f"Index {item} out of range")

    def __len__(self):
        return 2

    def forward(self, *input):
        assert input is not None, "input should not be None"

        if len(input) > 1:
            quant_conv_out = self.quant_conv(input[0])
            # the decoder's input is the quant_conv_out and the other inputs
            decoder_out = self.decoder(quant_conv_out, *input[1:])
            return decoder_out
        elif len(input) == 1:
            # the decoder's input is the quant_conv_out
            decoder_out = self.decoder(self.quant_conv(input[0]))
            return decoder_out
        else:
            raise ValueError("input should be a tuple of length larger than 1")


# * --- Tokenizer main --- #


@dataclass
class ChannelDropConfig:
    drop_type: Any = "exp"
    max_channels: int = 16
    drop_prob: float = 0.5


@dataclass
class EncoderDecoderConfig:
    in_channels: Any = 16  # in or list[int]
    out_channels: Any = 16
    channels: int = 128
    channels_mult: list[int] = field(default_factory=lambda: [2, 4, 4])
    num_res_blocks: int = 2
    attn_resolutions: list[int] = field(default_factory=lambda: [])
    dropout: float = 0.0
    resolution: int = 1024
    z_channels: int = 256  # feature before qunatizer
    latent_channels: int = 16
    spatial_compression: int = 8
    act_checkpoint: bool = False
    use_residual_factor: bool = False
    # resamples
    downsample_type: str = "PadConv"
    downsample_shortcut: Any = None  # str
    downsample_kwargs: Any = field(default_factory=lambda: {"padconv_use_manually_pad": True})  # fmt: skip
    # downsample_manually_pad: bool = False  # FIXME: True originally  # old version
    upsample_type: str = "RepeatConv"
    upsample_shortcut: Any = None  # str
    upsample_kwargs: Any = field(default_factory=lambda: {"interp_type": "xy_repeat"})
    # patch size, patcher, and blocks
    patch_size: int = 1
    patch_method: str = "haar"
    conv_in_module: str = "conv"
    block_name: str = "res_block"
    attn_type: str = "none"  # 'attn_vanilla' or 'none'
    # if block_name != 'moe', does not use
    moe_n_experts: int = 4
    moe_n_selected: int = 1
    moe_n_shared_experts: int = 1
    hidden_factor: int = 2
    moe_type: str = "tc"
    moe_token_mixer_type: str = "res_block"
    # padding and norm
    padding_mode: str = "reflect"
    norm_type: str = "gn"
    norm_groups: int = 32
    resample_norm_keep: bool = False
    # adaptive conv
    adaptive_mode: str = "interp"
    # generative decoder specific
    per_layer_noise: bool = False


@dataclass
class ContinuousTokenizerConfig:
    # feauture distillation related
    use_repa_loss: bool = False
    use_vf_loss: bool = False
    hook_module: str = "decoder.decoder.mid.block_2"
    vf_on_z_or_module: str = "module"
    dino_feature_dim: int = 1024
    latent_noise_prob: float = 0.0
    cache_type: str = "h"  # z or h
    # quantizer related
    quantizer_type: Optional[str] = None  # "kl", "bsq", "fsq", None
    random_quant: float = 0.0
    fsq_num_codebooks: int = 6
    fsq_levels: list[int] = field(default_factory=lambda: [8, 8, 8, 5, 5, 5])
    norm_in_quant_conv: bool = False
    # loading related
    enc_path: Optional[str] = ""
    dec_path: Optional[str] = ""
    uni_path: Optional[str] = ""
    loading_type: Optional[str] = None  # "nvidia", "uni", None
    # latent augmented related
    use_channel_drop: bool = False
    channel_drop_config: ChannelDropConfig = field(default_factory=ChannelDropConfig)
    # model related
    name: str = "ContinuousImageTokenizer"
    model: EncoderDecoderConfig = field(default_factory=lambda: EncoderDecoderConfig())
    decoder_type: str = "default"  # default or generative
    z_factor: int = 1


class ContinuousImageTokenizer(nn.Module):
    # FSDP attribution
    _no_split_modules: list[str] = ["ResnetBlock", "AttnBlock"]

    # training for feature distillation
    _use_repa_loss: bool = False
    _use_vf_loss: bool = False

    # vf loss on z or module output
    _vf_on_z_or_module: Literal["z", "module"] = "module"
    _hook_module: str = "decoder.decoder.mid.block_2"  # "decoder.decoder.up.1.block.2"
    _dino_feature_dim: int = 768  # [768, 1024]

    # scaling factor for evaluation
    scaling_factor: torch.Tensor | None = None
    shift_factor: torch.Tensor | None = None

    # state
    _hook_feature: torch.Tensor | None = None
    z: torch.Tensor | None = None  # the latent z
    supported_cached_hiddens: list[str] = ["z"]

    def __init__(self, cfg: ContinuousTokenizerConfig):
        super().__init__()
        self._use_repa_loss = cfg.use_repa_loss
        self._use_vf_loss = cfg.use_vf_loss
        self._hook_module = cfg.hook_module
        self._vf_on_z_or_module = cfg.vf_on_z_or_module
        self._dino_feature_dim = cfg.dino_feature_dim
        self.latent_noise_prob = cfg.latent_noise_prob
        self.use_latent_denoise = self.latent_noise_prob > 0.0

        self.cfg = cfg
        self.model_cfg = model_cfg = cfg.model

        # 1. repa or vf projectors
        assert not (self._use_repa_loss and self._use_vf_loss), (
            "repa and vf losses should not be used at the same time"
        )
        self._build_feature_align_mlp()

        # 2. FSDP wrapper module
        if len(model_cfg.attn_resolutions) == 0:
            self._no_split_modules.remove("AttnBlock")

        # 3. Quantizer
        self.quantizer_type = cfg.quantizer_type
        self.random_quant = cfg.random_quant
        self.quantizer = self._build_quantizer(cfg)

        self.loading_type = cfg.loading_type
        self.name = cfg.name
        self.latent_channels = model_cfg.z_channels
        self.norm_in_quant_conv = cfg.norm_in_quant_conv

        self.in_channels_after_patcher = (
            np.array(model_cfg.in_channels * model_cfg.patch_size**2)
        ).tolist()
        self.out_channels_after_patcher = (
            np.array(model_cfg.out_channels * model_cfg.patch_size**2)
        ).tolist()

        # NOTE: encoder and decoder maybe separated, e.g., NVIDIA pretrained tokenizer, or
        # trained on hyperspectral images before
        # if the uni_tokenizer_path is not empty, then the encoder and decoder are loaded directly.
        enc_path = cfg.enc_path
        dec_path = cfg.dec_path
        uni_tokenizer_path = cfg.uni_path

        self.register_buffer("dummy_param", torch.tensor(0), persistent=False)

        # 4. Encoder and Decoder
        # pretrained encoder and decoder
        if cfg.loading_type == "nvidia":
            assert enc_path.endswith(".jit") and dec_path.endswith(".jit")
            # pretrained model from NVIDIA cosmos tokenizer
            assert not self.norm_in_quant_conv, (
                "norm_in_quant_conv is not supported for nvidia pretrained model settings, trian it from scratch"
            )

            tokenizer_cfg = dict(
                z_channels=model_cfg.z_channels,
                z_factor=cfg.z_factor,
                latent_channels=model_cfg.z_channels,
            )
            tokenizer_cfg.update(asdict(model_cfg))
            log_print(
                f"start from the pretrained model, cosmos tokenizer cfg is {tokenizer_cfg}",
                "debug",
            )
            enc_jit, dec_jit = self.load_pretrained(
                enc_path=enc_path, dec_path=dec_path, tokenizer_cfg=tokenizer_cfg
            )  # type: ignore

            # split the encoder and decoder
            encoder, quant_conv = enc_jit[0], enc_jit[1]
            decoder, post_quant_conv = dec_jit[1], dec_jit[0]

            self.encoder = self.encoder_jit(encoder, quant_conv)
            self.decoder = self.decoder_jit(decoder, post_quant_conv)

        else:
            # encoder and decoder
            # not combine the encoder, for FSDP wrap
            encoder, decoder = self._build_encoder_decoder(cfg, model_cfg)
            quant_conv, post_quant_conv = self._build_pre_post_quant_convs(cfg)

            self.encoder = self.encoder_jit(encoder, quant_conv)
            self.decoder = self.decoder_jit(decoder, post_quant_conv)

            # Load weights
            if cfg.loading_type is not None:
                if cfg.norm_in_quant_conv:
                    assert enc_path == "" and dec_path == "", (
                        "norm_in_quant_conv is not supported for pretrained settings, train it from scratch"
                    )

                # loading may slow, profile it.
                verbose = False
                profile = False
                profiler = (
                    torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                        ],
                        record_shapes=True,
                        profile_memory=True,
                        with_stack=True,
                    )
                    if profile
                    else nullcontext()
                )
                with profiler:
                    self.load_pretrained(
                        enc_path=enc_path,
                        dec_path=dec_path,
                        uni_tokenizer_path=uni_tokenizer_path,
                    )
                if verbose and isinstance(profiler, torch.profiler.profile):
                    log_print(
                        profiler.key_averages().table(
                            sort_by="self_cpu_time_total", row_limit=10
                        )
                    )

        # token channel drop
        self.use_channel_drop = cfg.use_channel_drop
        if self.use_channel_drop:
            self.channel_drop = NestChannelDrop(**asdict(cfg.channel_drop_config))
            log_print(f"use channel drop: {cfg.channel_drop_config}")

        # register repa hook
        if self._vf_on_z_or_module == "module" and (
            self._use_vf_loss or self._use_repa_loss
        ):
            self.register_feature_hook()

        num_parameters = sum(param.numel() for param in self.parameters())
        log_print(f"model={self.name}, num_parameters={num_parameters:,}")
        log_print(
            f"z_channels={model_cfg.z_channels}, latent_channels={self.latent_channels}."
        )

    def _build_encoder_decoder(
        self,
        cfg: ContinuousTokenizerConfig,
        model_cfg: EncoderDecoderConfig,
    ):
        self._is_diffbands = isinstance(model_cfg.in_channels, (tuple, list))

        model_kwargs = asdict(model_cfg)
        encoder = Encoder(**model_kwargs)
        if cfg.decoder_type == "default":
            decoder = Decoder(**model_kwargs)
        elif cfg.decoder_type == "generative":
            decoder = GenerativeDecoder(**model_kwargs)
        else:
            raise ValueError(f"Unknown decoder type: {cfg.decoder_type}")

        log_print(f"[CNN tokenizer]: Build encoder and {cfg.decoder_type} decoder.")
        return encoder, decoder

    def _build_quantizer(self, cfg: ContinuousTokenizerConfig):
        model_cfg = self.model_cfg
        if self.quantizer_type == "kl":
            self.quantizer = DiagonalGaussianDistribution  # not quantizer, compatible with trainer  # type: ignore
        elif self.quantizer_type == "bsq":
            assert model_cfg.z_channels % 2 == 0, (
                "quantizer out channels should be even"
            )
            self.quantizer = BSQ(
                embed_dim=model_cfg.z_channels,  # 18 or 36
                beta=0.0,  # commitment loss
                gamma0=1.0,
                gamma=1.0,
                zeta=1.0,
                inv_temperature=1.0,
                cb_entropy_compute="group",
                l2_norm=True,
                input_format="bchw",
                persample_entropy_compute="analytical",
                group_size=1,  # group_size must affect the GPU mem (compared with LFQ), f8z36g36
            )
        elif self.quantizer_type == "fsq":
            self.quantizer = FSQ(
                levels=cfg.fsq_levels,
                dim=model_cfg.z_channels,
                num_codebooks=cfg.fsq_num_codebooks,
                channel_first=True,
            )
        elif self.quantizer_type == "psd":
            self.quantizer = PowerSphericalDistribution  # not quantizer, compatible with trainer  # type: ignore
        elif self.quantizer_type is None:
            self.quantizer = None
        else:
            raise ValueError("quantizer type should be one of [kl, bsq, fsq, None]")

        if self.quantizer_type is not None:
            log_print(f"Using quantizer: {self.quantizer.__class__.__name__}")
        else:
            log_print(f"use no quantizer or VAE, the tokenizer is only an AutoEncoder")

        return self.quantizer

    def _build_pre_post_quant_convs(self, cfg: ContinuousTokenizerConfig):
        model_cfg = self.model_cfg
        q_conv_chan = model_cfg.latent_channels
        if cfg.quantizer_type == "kl":
            q_conv_chan = q_conv_chan * 2
        elif cfg.quantizer_type == "psd":
            q_conv_chan = q_conv_chan + 1

        # quant_conv and post_quant_conv
        if self.norm_in_quant_conv:
            warnings.warn(
                '"norm_in_quant_conv" is not supported for pretrained settings and not recommended to use'
                "it will be removed",
                DeprecationWarning,
            )
            quant_conv = nn.Sequential(
                Normalize(model_cfg.latent_channels, norm_type="gn"),
                torch.nn.Conv2d(
                    model_cfg.latent_channels,
                    q_conv_chan,
                    1,
                ),
            )
        else:
            quant_conv = torch.nn.Conv2d(model_cfg.z_channels, q_conv_chan, 1)

        # then the quantizer will output the latent_channels h
        post_quant_conv = torch.nn.Conv2d(
            model_cfg.latent_channels, model_cfg.z_channels, 1
        )

        return quant_conv, post_quant_conv

    def _build_feature_align_mlp(self):
        if self._use_repa_loss:
            if self._vf_on_z_or_module == "module":
                self._repa_proj = build_mlp(
                    512,  # rely on the module channels
                    self._dino_feature_dim,
                    self._dino_feature_dim,
                )
            else:
                self._repa_proj = build_mlp(
                    # if is z: rely on the z channels
                    # else is the latent channel proj.
                    self.model_cfg.z_channels
                    if self.cfg.cache_type == "z"
                    else self.model_cfg.latent_channels,
                    self._dino_feature_dim,
                    self._dino_feature_dim,
                )
        if self._use_vf_loss:
            if self._vf_on_z_or_module == "module":
                self._vf_proj = build_mlp(
                    512,
                    self._dino_feature_dim,
                    self._dino_feature_dim,
                )
            else:
                self._vf_proj = build_mlp(
                    self.model_cfg.z_channels
                    if self.cfg.cache_type == "z"
                    else self.model_cfg.latent_channels,
                    self._dino_feature_dim,
                    self._dino_feature_dim,
                )

    def encoder_jit(self, encoder, quant_conv):
        return nn.Sequential(
            OrderedDict([("encoder", encoder), ("quant_conv", quant_conv)])
        )

    def decoder_jit(self, decoder, post_quant_conv):
        return DecoderSequential(post_quant_conv, decoder)

    def register_feature_hook(self):
        def hook(module, input, output):
            self._hook_feature = output

        self.get_submodule(self._hook_module).register_forward_hook(hook)
        log_print(
            f"[Cosmos Tokenizer]: module {self._hook_module} is registered for hook"
        )

    # * --- model feature alignment --- #

    @torch.autocast("cuda", torch.bfloat16)
    def get_repa_feature(self):
        # only one feature
        if hasattr(self, "_repa_proj"):
            if self._vf_on_z_or_module == "z":
                # project on latent
                assert self.z is not None, "z should be set before get_vf_feature"
                return self._repa_proj(self.z)
            elif self._vf_on_z_or_module == "module":
                # proj on block out
                return self._repa_proj(self._hook_feature)
            else:
                raise ValueError(
                    f"vf loss should get feature when vf is computed on {self._vf_on_z_or_module}"
                )

        return None

    @torch.autocast("cuda", torch.bfloat16)
    def get_vf_feature(self):
        if hasattr(self, "_vf_proj"):
            if self._vf_on_z_or_module == "z":
                # project on latent
                assert self.z is not None, "z should be set before get_vf_feature"
                return self._vf_proj(self.z)
            elif self._vf_on_z_or_module == "module":
                # proj on block out
                return self._vf_proj(self._hook_feature)
            else:
                raise ValueError(
                    f"vf loss should get feature when vf is computed on {self._vf_on_z_or_module}"
                )

        return None

    # * --- GAN training loss utils --- #

    def get_last_layer(self):
        # get decoder last layer weight for discriminator loss
        if not self.decoder.decoder._wrap_fsdp_last_layer:
            return self.decoder.decoder.conv_out.weight
        else:
            return self.decoder.decoder.conv_out.wrap_mod.weight

    def get_last_enc_layer(self):
        # get encoder last layer weight for visual foundation loss
        # return self.encoder.encoder.conv_out.weight
        if self._vf_on_z_or_module == "z":
            return self.encoder.quant_conv.weight
        else:  # module
            # decoder.decoder.mid.block_2
            # hard code here, say the block_2 is a resnet block
            block_name = self.encoder.encoder.block_name
            if block_name == "res_block":
                return self.get_submodule(self._hook_module).conv2.weight
            elif block_name == "res_moe":
                # return self.get_submodule(self._hook_module).moe.moe['moe_tc'].shared_experts.down_proj.weight
                # return self.get_submodule(self._hook_module).token_mixer.conv2.weight
                return None
            else:
                raise ValueError(
                    f"block_name {block_name} not supported, only res_block and res_moe are supported"
                )

    # * --- latent structure shaping --- #

    def _latent_noising(self, h: torch.Tensor, mask: torch.Tensor | None = None):
        # mask: 1 means the channel is dropped, 0 means the channel is not dropped

        if random.random() > self.latent_noise_prob:
            return h

        # interpolated noising
        bs = h.size(0)
        t = torch.randn((bs,)).to(h).view(-1, 1, 1, 1)
        noise = torch.randn_like(h)
        # mask out the dropped channels
        if mask is not None:
            assert mask.shape[1] == h.shape[1], (
                "mask and h should have the same channel number"
            )
            noise = torch.where(mask, noise, torch.zeros_like(noise))
        h_noise = t * noise + (1 - t) * h

        return h_noise

    # * --- AE encode and decode --- #

    def _use_quantizer(self, use_quantizer=None):
        if self.quantizer_type is None:
            return False

        if self.training and self.random_quant > 0.0:
            # Random select to use quantizer or not
            return self.random_quant > random.random()
        elif use_quantizer is None:
            return True
        else:
            # Or decided by the input arg
            return use_quantizer

    def _has_quantizer_applied_fn(self, h, z, use_quantizer=None, cache_type="z"):
        h_dtype = h.dtype
        # h_clone = h.clone()
        h = h.float()  # quantizers are in float32

        if self.quantizer_type == "bsq":
            # here must be l2-normed
            h = nn.functional.normalize(h, dim=1)
            # TODO: bsq not supported channel drop
            hq, bsq_loss, loss_breakdown = self.quantizer(h)
            res = hq.to(h_dtype), bsq_loss, loss_breakdown

        elif self.quantizer_type == "kl":
            m_, logvar_ = h.chunk(2, dim=1)
            posterior = self.quantizer((m_, logvar_))
            kl_loss = posterior.kl()
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            h = posterior.sample()
            loss_breakdown = KLLossBreakDown(
                posterior=posterior,
                mean=m_,
                logvar=logvar_,
            )

            if self.use_channel_drop:
                h, _ = self.channel_drop(h)
            res = h.to(h_dtype), kl_loss, loss_breakdown

        elif self.quantizer_type == "fsq":
            # dummy loss
            fsq_loss = torch.tensor(0.0).to(h)
            loss_breakdown = {"fsq_loss": fsq_loss}
            hq, indices = self.quantizer(h)
            res = hq.to(h_dtype), fsq_loss, loss_breakdown

        elif self.quantizer_type == "psd":
            mu = h[:, :-1]
            kappa = h[:, -1]
            # mu = l2_norm(mu, dim=1)
            kappa = nn.functional.softplus(kappa) + 1.0
            hq = self.quantizer(mu, kappa, dim=1)
            loss = hq.kl_to_uniform()
            # reparameterization
            h = hq.rsample()
            h = h * (self.latent_channels**0.5)
            psd_loss = loss.mean()
            res = h.to(h_dtype), psd_loss, {"kl_loss": psd_loss}

        else:
            raise RuntimeError("can not reach here")

        # Cache the latent
        # TODO: fix the discreate quantizer
        if self.cfg.cache_type == "z":
            self._training_latent_cache(z, use_quantizer, cache_type)
        else:
            self._training_latent_cache(res[0], use_quantizer, cache_type)

        return res

    def _no_quantizer_applied_fn(self, h, z, use_quantizer=None, cache_type="z"):
        # Do no quantization, but cache the latent
        cached_ = z if self.cfg.cache_type == "z" else h
        self._training_latent_cache(cached_, use_quantizer, cache_type)
        return h

    def apply_quantizer(self, h, z, use_quantizer=None, cache_type="z"):
        # TODO: Cache z or h; affect the repa mlp projection
        _use_quantizer = self._use_quantizer(use_quantizer)
        # Quantization
        if _use_quantizer:
            return self._has_quantizer_applied_fn(h, z, _use_quantizer, cache_type)
        # Autoencoder
        else:
            return self._no_quantizer_applied_fn(h, z, _use_quantizer, cache_type)

    def latent_aug(self, h):
        if self.training:
            mask = None
            if self.use_channel_drop:
                channel_drop_result = self.channel_drop(h)
                if isinstance(channel_drop_result, tuple):
                    h, mask = channel_drop_result
                else:
                    h = channel_drop_result
            if self.use_latent_denoise:
                # latent noising and then to _vf_proj (vf decoder)
                h = self._latent_noising(h, mask)

        return h

    def _training_latent_cache(
        self,
        cached_tensor: Tensor,
        use_quantizer: bool | None = None,
        cache_type: str = "z",
    ):
        if cache_type is None or cache_type == "none":
            # No cache
            return

        assert cache_type in self.supported_cached_hiddens, (
            f"cache_type {cache_type} not supported, only {self.supported_cached_hiddens} are supported"
        )
        # not use_quantizer, save the unquantized latent
        if self.training:
            # Save latent if is AE
            if hasattr(self, "_vf_proj") or hasattr(self, "_repa_proj"):
                setattr(self, cache_type, cached_tensor)
            else:
                setattr(self, cache_type, None)

    def encode_with_itermediate_features(self, x, use_quantizer=None):
        z, feats = self.encoder.encoder(x, ret_interm_feats=True)
        h = self.encoder.quant_conv(z)
        maybe_q_encoded = self.apply_quantizer(h, use_quantizer)

        q_loss = loss_breakdown = None
        if isinstance(maybe_q_encoded, tuple):
            encoded, q_loss, loss_breakdown = maybe_q_encoded
        else:
            encoded = h
        return dict(
            encoded=encoded,
            itermediate_feats=feats,
            q_loss=q_loss,
            loss_breakdown=loss_breakdown,
        )

    def encode(self, x, use_quantizer=None):
        # Encoder
        z = self.encoder.encoder(x)
        h = self.encoder.quant_conv(z)
        # TODO: add additional cross-attention to convert the 2D latent to 1D latent

        # Quantization
        maybe_q_ret = self.apply_quantizer(h, z, use_quantizer)
        if isinstance(maybe_q_ret, tuple):
            h, q_loss, loss_breakdown = maybe_q_ret
            # NOTE: if quantizer is used, the aug z is not applied
            return h, q_loss, loss_breakdown

        # z augmentions
        h = self.latent_aug(maybe_q_ret)
        return h

    def decode(
        self,
        h: torch.Tensor | tuple,
        inp_shape: Annotated[torch.Size | int, "bs,c,h,w or c"],
        clamp=False,
    ):
        # Break down input z losses
        q_loss = loss_breakdown = None
        if self.quantizer_type is not None and isinstance(h, (tuple, list)):
            h, q_loss, loss_breakdown = h
        else:
            assert torch.is_tensor(h), "z should be the (quantized) latent"

        # Decoder
        chan = inp_shape[1] if isinstance(inp_shape, (torch.Size, tuple)) else inp_shape
        dec = self.decoder(h, chan)  # [b, c, h, w]
        if clamp:
            dec = dec.clamp(-1, 1)

        if self.quantizer_type is not None:
            return dec, q_loss, loss_breakdown
        else:
            return dec

    def forward(self, input: torch.Tensor):
        if cosmos_block.compile_forward_fn and cosmos_block.compile_forward_fn:
            torch.compiler.cudagraph_mark_step_begin()  # ty: ignore
        latent = self.encode(input)
        dec = self.decode(latent, input.shape)

        return dec

    # * --- checkpoint loding --- #
    @no_type_check
    def load_pretrained(
        self,
        enc_path: str | None = None,
        dec_path: str | None = None,
        tokenizer_cfg: dict | None = None,
        uni_tokenizer_path: str | None = None,
        mean_init_conv_in_out: bool = False,
    ) -> tuple[Encoder, Decoder] | None:
        if (enc_path == "" or dec_path == "") and uni_tokenizer_path == "":
            return None

        # * --- load NVIDIA Cosmos separated encoder, decoder checkpoints --- #

        if self.loading_type == "nvidia":
            assert tokenizer_cfg is not None, (
                "tokenizer_cfg is required when loading the nvidia pretrained tokenizer"
            )
            log_print(
                f"Loading pretrained encoder from {enc_path} for NVIDIA pretrained model"
            )
            encoder, _enc_model_mody_keys = load_jit_model_shape_matched(
                jit_model_path=enc_path,
                device="cuda",
                tokenizer_config=tokenizer_cfg,
                part="encoder",
            )
            log_print(
                f"Loading pretrained decoder from {dec_path} for NVIDIA pretrained model"
            )
            decoder, _dec_model_mody_keys = load_jit_model_shape_matched(
                jit_model_path=dec_path,
                device="cuda",
                tokenizer_config=tokenizer_cfg,
                part="decoder",
            )

            log_print(
                f"not compatible for pretraine models: \n"
                f"encoder: {_enc_model_mody_keys}\n"
                f"decoder: {_dec_model_mody_keys}\n",
                "warning",
            )
            return encoder, decoder

        # * --- load pretrained uni-tokenizer or separate encoder and decoder --- #

        else:
            if uni_tokenizer_path != "" or uni_tokenizer_path is not None:
                log_print(
                    f"Loading pretrained encoder from {uni_tokenizer_path} for pretrained model"
                )
                weights = accelerate.utils.load_state_dict(uni_tokenizer_path)
                # load_state_dict will check the shape of the model and the state dict
                _missing_keys, _unexp_keys = load_weights_with_shape_check(
                    self, weights
                )
                log_print(
                    f"tokenizer: missing keys {_missing_keys}, unexpected keys {_unexp_keys}",
                    "warning",
                )

                # if conv_in is nn.Conv2d for only one channel
                # and if the pretrained conv_in's basic module is also conv
                _tgt_conv_w = f"encoder.encoder.conv_in.in_modules.conv_in_{self.in_channels_after_patcher}.weight"
                _tgt_conv_b = f"encoder.encoder.conv_in.in_modules.conv_in_{self.in_channels_after_patcher}.bias"
                if (
                    isinstance(self.encoder.encoder.conv_in, nn.Conv2d)
                    and weights.get(_tgt_conv_w, None) is not None
                ):
                    self.encoder.encoder.conv_in.weight.data.copy_(weights[_tgt_conv_w])
                    self.encoder.encoder.conv_in.bias.data.copy_(
                        weights.get(_tgt_conv_b, None)
                    )
                    log_print(
                        f"[Cosmos Tokenizer]: conv_in is copied from pretrained model from key {_tgt_conv_w}"
                    )

                # if conv_out is nn.Conv2d for only one channel
                # and if the pretrained model conv_out is diff bands module
                _tgt_conv_w = f"decoder.decoder.conv_out.in_modules.conv_out_{self.out_channels_after_patcher}.weight"
                _tgt_conv_b = f"decoder.decoder.conv_out.in_modules.conv_out_{self.out_channels_after_patcher}.bias"
                if (
                    isinstance(self.decoder.decoder.conv_out, nn.Conv2d)
                    and weights.get(_tgt_conv_w, None) is not None
                ):
                    self.decoder.decoder.conv_out.weight.data.copy_(
                        weights[_tgt_conv_w]
                    )
                    self.decoder.decoder.conv_out.bias.data.copy_(
                        weights.get(_tgt_conv_b, None)
                    )
                    log_print(
                        f"[Cosmos Tokenizer]: conv_out is copied from pretrained model from key {_tgt_conv_w}"
                    )

                log_print("load pretrained model done.")

            else:
                assert enc_path.endswith("safetensors") and dec_path.endswith(
                    "safetensors"
                ), "only support safetensors for now"
                log_print(
                    "pretrained model is pretrained on hyperspectral images, "
                    "for now is used to finetune on the other dataset"
                )

                enc_sd = accelerate.utils.load_state_dict(enc_path)
                dec_sd = accelerate.utils.load_state_dict(dec_path)

                # * shaped matched loading ==================
                # load_state_dict will check the shape of the model and the state dict
                # if the shape is not matched, it will not raise an error
                # but the model will not be loaded

                _enc_missing, _enc_unexp = load_weights_with_shape_check(
                    self.encoder, enc_sd
                )
                _dec_missing, _dec_unexp = load_weights_with_shape_check(
                    self.decoder, dec_sd
                )

                # * handle the input and output conv manually ===============
                _conv_in_is_missing = any(
                    ["encoder.conv_in" in _key for _key in _enc_missing]
                )  # only weight in conv_in
                if self.decoder.decoder._wrap_fsdp_last_layer:
                    _decoder_conv_out_name = "decoder.conv_out.wrap_mod"
                else:
                    _decoder_conv_out_name = "decoder.conv_out"
                _conv_out_is_missing = any(
                    ["decoder.conv_out" in _key for _key in _dec_missing]
                )

                if _conv_in_is_missing:
                    if mean_init_conv_in_out:
                        assert isinstance(
                            self.in_channels_after_patcher, int
                        ) and isinstance(self.out_channels_after_patcher, int), (
                            "in_channels_after_patcher and out_channels_after_patcher should be int"
                        )

                        _mean_conv_in: Tensor = enc_sd["encoder.conv_in.weight"].mean(
                            keepdim=True, dim=1
                        )  # (d, inp_c, k, k)
                        _mean_conv_in = _mean_conv_in.repeat_interleave(
                            self.in_channels_after_patcher,
                            dim=1,  # after patcher
                        )
                        self.encoder.encoder.conv_in.weight.data.copy_(_mean_conv_in)  # type: ignore
                        log_print(
                            "conv_in is missing, use the mean of the conv_in weight"
                        )

                    # if conv_in is nn.Conv2d for only one channel
                    # and if the pretrained conv_in's basic module is also conv
                    _tgt_conv_w = f"encoder.conv_in.in_modules.conv_in_{self.in_channels_after_patcher}.weight"
                    _tgt_conv_b = f"encoder.conv_in.in_modules.conv_in_{self.in_channels_after_patcher}.bias"
                    if (
                        isinstance(self.encoder.encoder.conv_in, nn.Conv2d)
                        and enc_sd.get(_tgt_conv_w, None) is not None
                    ):
                        self.encoder.encoder.conv_in.weight.data.copy_(
                            enc_sd[_tgt_conv_w]
                        )
                        self.encoder.encoder.conv_in.bias.data.copy_(
                            enc_sd.get(_tgt_conv_b, None)
                        )
                        log_print(
                            f"[Cosmos Tokenizer]: conv_in is copied from pretrained model from key {_tgt_conv_w}"
                        )

                if _conv_out_is_missing:
                    if mean_init_conv_in_out:
                        assert isinstance(
                            self.in_channels_after_patcher, int
                        ) and isinstance(self.out_channels_after_patcher, int), (
                            "in_channels_after_patcher and out_channels_after_patcher should be int"
                        )

                        _mean_conv_out_w = dec_sd["decoder.conv_out.weight"].mean(
                            keepdim=True, dim=0
                        )  # (out_c, d, k, k)
                        _mean_conv_out_w = _mean_conv_out_w.repeat_interleave(
                            self.out_channels_after_patcher, dim=0
                        )
                        _mean_conv_out_bias = (
                            dec_sd["decoder.conv_out.bias"]
                            .mean(keepdim=True, dim=0)
                            .repeat_interleave(self.out_channels_after_patcher)
                        )  # (out_c,)

                        # copy in
                        conv_out_w = self.decoder.get_submodule(
                            _decoder_conv_out_name
                        ).weight
                        conv_out_b = self.decoder.get_submodule(
                            _decoder_conv_out_name
                        ).bias
                        conv_out_w.data.copy_(_mean_conv_out_w)  # type: ignore
                        conv_out_b.data.copy_(_mean_conv_out_bias)  # type: ignore

                        log_print(
                            "conv_out is missing, use the mean of the conv_out weight"
                        )

                    # if conv_out is nn.Conv2d for only one channel
                    # and if the pretrained model conv_out is diff bands module
                    _tgt_conv_w = f"decoder.conv_out.in_modules.conv_out_{self.out_channels_after_patcher}.weight"
                    _tgt_conv_b = f"decoder.conv_out.in_modules.conv_out_{self.out_channels_after_patcher}.bias"
                    if (
                        isinstance(self.decoder.decoder.conv_out, nn.Conv2d)
                        and dec_sd.get(_tgt_conv_w, None) is not None
                    ):
                        self.decoder.decoder.conv_out.weight.data.copy_(
                            enc_sd[_tgt_conv_w]
                        )
                        self.decoder.decoder.conv_out.bias.data.copy_(
                            enc_sd.get(_tgt_conv_b, None)
                        )

                log_print(
                    f"load pretrained model done. \n"
                    f"encoder: missing keys {_enc_missing}, unexpected keys {_enc_unexp}\n"
                    f"decoder: missing keys {_dec_missing}, unexpected keys {_dec_unexp}",
                    "warning",
                )

    @no_type_check
    def register_layer_output_hooks(self):
        self._per_layer_norms = {}
        self._next_call_norm_flag = False

        def _output_norm_hook(module, input, output):
            if not self._next_call_norm_flag:
                return output
            else:
                self._next_call_norm_flag = False

                _per_layer_dict_name = module._norm_hook_name
                _norm = output.norm()
                self._per_layer_norms[_per_layer_dict_name] = _norm
                return output

        for _m_name, _m in chain(
            self.encoder.encoder.down.block.named_children(),
            self.encoder.encoder.down.attn.named_children(),
            self.encoder.encoder.mid.named_children(),
            [
                ("encoder.conv_in", self.encoder.encoder.conv_in),
                ("encoder.conv_out", self.encoder.encoder.conv_out),
            ],
            self.decoder.decoder.up.block.named_children(),
            self.decoder.decoder.up.attn.named_children(),
            self.decoder.decoder.mid.named_children(),
            [
                ("decoder.conv_in", self.decoder.decoder.conv_in),
            ],
        ):
            log_print(f"register norm hook for {_m_name}")
            setattr(_m, "_norm_hook_name", _m_name)
            _m.register_forward_hook(_output_norm_hook)

    def get_layer_output_norms(self):
        norms = getattr(self, "_per_layer_norms", None)
        if norms is not None:
            self._per_layer_norms = {}

        return norms

    # * --- lora-related methods --- #

    def peft_fully_finetune_modules(
        self, add_norms: bool = False, conv_stem_reinit=False
    ) -> list[str]:
        """
        PEFT fully finetuned modules (no LoRA A and B).
        """
        module_to_save_layers = []

        # conv in and conv out
        # first conv: diff-bands convs or nn.Conv2d
        if conv_stem_reinit or not self._is_diffbands:
            # if reinit, the the conv stem is just an nn.Conv2d
            # FIXME: this will add all conv_in convs to fully finetuned, but we only need one conv to be tuned, which will result the state dict is larger

            # Encoder conv_in
            module_to_save_layers = ["encoder.encoder.conv_in"]

            # Decoder conv_out
            if not self.decoder.decoder._wrap_fsdp_last_layer:
                _conv_out_name = "decoder.decoder.conv_out"
            else:
                _conv_out_name = "decoder.decoder.conv_out.wrap_mod"  # decrepeted

            module_to_save_layers.append(_conv_out_name)
            log_print(
                f"[Cosmos Tokenizer LoRA]: add conv_in and conv_out to fully finetune"
            )

        # backbone normalization layers
        if add_norms:
            module_to_save_layers += ["norm", "norm1", "norm2", "norm_out"]

        # projections for repa or vf losses
        if hasattr(self, "_repa_proj"):
            module_to_save_layers.append("_repa_proj")
        if hasattr(self, "_vf_proj"):
            module_to_save_layers.append("_vf_proj")

        return module_to_save_layers

    def peft_lora_modules(
        self, conv_stem_reinit=False, conv_stem_chan: int | None = None
    ) -> list[str]:
        """
        PEFT LoRA modules (with LoRA A and B)
        """
        add_tgt_modules = []

        # If the Conv stem is not reinit, use the pretrained weights,
        # it should be added with the lora weights
        if not conv_stem_reinit and self._is_diffbands:
            assert conv_stem_chan is not None, (
                f"conv_stem_chan must be specified when conv_stem_reinit is False"
            )

            # Only add the corresponding stem_channel convs when the conv_in and conv_out
            # are multi-bands conv module.
            add_tgt_modules = [f"in_modules.conv_in_{conv_stem_chan}"]
            if not self.decoder.decoder._wrap_fsdp_last_layer:
                _conv_out_name = f"in_modules.conv_out_{conv_stem_chan}"
            else:
                _conv_out_name = f"wrap_mod.conv_out_{conv_stem_chan}"
            add_tgt_modules.append(_conv_out_name)
        else:
            # NOTE: if the conv in and conv out is nn.Conv2d, since the lora weights A and B
            # can not be slice or interp, we need to fully finetune them.
            pass

        # Backbone convs
        add_tgt_modules += [
            "nin_shortcut.1",
            "conv",
            "conv1",
            "conv2",
            "quant_conv",
            "encoder.encoder.conv_out",
        ]
        # TODO: add attention q, k, v, proj_out

        return add_tgt_modules

    # * --- Create model --- #

    @classmethod
    def create_model(cls, config: DictConfig | None = None, **kwargs):
        if config is not None:
            cfg = dataclass_from_dict_config(
                ContinuousTokenizerConfig, config, strict=False
            )
        else:
            cfg = dataclass_from_dict(ContinuousTokenizerConfig, kwargs, strict=False)
        return cls(cfg)

    def set_grad_checkpointing(self, enabled: bool = True):
        for m in self.modules():
            if hasattr(m, "grad_checkpointing"):
                m.grad_checkpointing = enabled
                log_print(
                    f"set grad_checkpointing={enabled} for {m.__class__.__name__}"
                )


# * --- test --- * #


@catch_any()
def test_tokenizer_forward_backward(
    model_cls=ContinuousImageTokenizer,
    is_lora=False,
    count_params=False,
    real_data: str | None = None,
    use_optim=False,
    device="cuda",
    base_model_ckpt: str = "runs/stage1_cosmos/2025-08-20_20-14-19_cosmos_f8c16p1_unified_hyperspectral_latent_noise=0.0_channel_drop=False/ema/tokenizer/model.safetensors",
    lora_ckpt: list[str] | None = None,
    lora_changes_chans: dict[str, int] | None = None,
    active_lora_name: str | None = None,
    check_grad=False,
    show_mem_usage=False,
    save_pca_vis=False,
    pca_type: str = "proj",
    other_model_kwargs: dict | None = None,
    save_img_dir: str | None = None,
    rgb_chans: list[int] = [4, 2, 0],
    dtype=torch.bfloat16,
    upscale: int = 1,
    fake_img_shape: tuple = (1, 12, 256, 256),
    compute_mean_std: bool = False,
    max_iters: int = 100,
):
    from contextlib import nullcontext

    from fvcore.nn import parameter_count_table
    from peft import PeftModel
    from PIL import Image
    from torchmetrics.aggregation import MeanMetric
    from torchmetrics.image import PeakSignalNoiseRatio
    from torchvision.utils import make_grid
    from tqdm import tqdm, trange

    from src.data.hyperspectral_loader import get_fast_test_hyperspectral_data
    from src.stage1.cosmos.lora_mixin import TokenizerLoRAMixin
    from src.stage1.cosmos.modules import blocks
    from src.stage1.utilities.losses.repa.feature_pca import feature_pca_sk
    from src.utilities.logging import print
    from src.utilities.metrics.aggregation import StackMeanMetrics
    from src.utilities.network_utils import load_peft_model_checkpoint, mem_context

    default_multi_chans = (
        512  # [3, 4, 8, 10, 12, 13, 32, 50, 150, 175, 202, 224, 242, 368]
    )
    # default_nested_chans = 500  # max hyperspectral chans in the training datasets
    config = {
        "model": {
            "attn_resolutions": [32],
            "channels": 128,
            "channels_mult": [2, 4, 4],
            "dropout": 0.0,
            "in_channels": 512,
            "spatial_compression": 8,
            "num_res_blocks": 2,
            "out_channels": 512,
            "resolution": 1024,
            "patch_size": 1,
            "patch_method": "haar",
            "z_channels": 256,
            "latent_channels": 16,
            "act_checkpoint": True,
            "block_name": "res_block",  # res_block, res_moe
            "padding_mode": "reflect",
            "norm_type": "gn",
            "norm_groups": 32,
            "attn_type": "none",
            "adaptive_mode": "interp",
            # "upsample_kwargs": {},
            # "downsample_kwargs": {},
        },
        "name": "CI",
        "uni_path": base_model_ckpt,
        "loading_type": "pretrained" if Path(base_model_ckpt).exists() else None,
        "quantizer_type": None,
        # repa
        "hook_for_repa": False,
        "use_repa_loss": True,
        "use_vf_loss": False,
        "vf_on_z_or_module": "z",
        "dino_feature_dim": 1024,
        "z_factor": 1,
    }
    if other_model_kwargs:
        if "model" in other_model_kwargs:
            config["model"].update(other_model_kwargs.pop("model"))
        else:
            config.update(other_model_kwargs)
    tokenizer = model_cls.create_model(**config).cuda()
    tokenizer = tokenizer.eval().to(dtype)

    if is_lora:
        assert lora_ckpt is not None, "lora_ckpt is required for lora test"
        # Use TokenizerLoRAMixin for lazy LoRA loading
        if isinstance(lora_ckpt, str):
            lora_ckpt = [lora_ckpt]

        # Create lora_weights dict for TokenizerLoRAMixin
        lora_weights = {}
        for ckpt_path in lora_ckpt:
            # Use directory stem as adapter name
            lora_name = Path(ckpt_path).stem
            lora_weights[lora_name] = ckpt_path

        # Create TokenizerLoRAMixin instance
        tokenizer_mixin = TokenizerLoRAMixin(
            tokenizer=tokenizer,
            lora_weights=lora_weights,
            lora_hyper_chans=lora_changes_chans,
            active_lora=active_lora_name,
            lora_cfg=None,  # Use configs from directories
        )
        # get base model
        # tokenizer = tokenizer_mixin.base_tokenizer
        tokenizer = tokenizer_mixin

        # Move to device and set to eval mode
        tokenizer = tokenizer.to("cuda", dtype)
        tokenizer.eval()

        # Log available LoRAs
        logger.info(f"Available LoRA adapters: {tokenizer_mixin.get_available_loras()}")

    if count_params:
        logger.info(parameter_count_table(tokenizer))

    is_itered = False
    if real_data is not None:
        if Path(real_data).exists():
            # only support RGB image
            x = Image.open(real_data).convert("RGB")
            x = (
                torch.from_numpy(np.array(x))
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .cuda()
            )
            x = x / 255.0
            x = x * 2 - 1  # normalize to [-1, 1]
            iterations = [x]
            is_itered = True
        else:
            dl = get_fast_test_hyperspectral_data(batch_size=1, data_type=real_data)  # type: ignore
            iterations = dl
    else:
        x = torch.randn(*fake_img_shape).to("cuda", dtype)
        iterations = [x]

    if not is_itered and upscale != 1:
        x = torch.nn.functional.interpolate(
            x, scale_factor=upscale, align_corners=True, mode="bicubic"
        )

    if use_optim:
        opt = torch.optim.Adam(tokenizer.parameters(), lr=1e-4, fused=True)

    metric = MeanMetric().cuda()
    if compute_mean_std:
        mean_fn = StackMeanMetrics().cuda()
        std_fn = StackMeanMetrics().cuda()
    ctx = torch.no_grad if not (use_optim or check_grad) else torch.enable_grad
    mem_ctx = mem_context(device) if show_mem_usage else nullcontext()
    with torch.autocast("cuda", dtype) and mem_ctx:
        for index, x in (tbar := tqdm(enumerate(iterations))):
            with ctx():
                if isinstance(x, dict):
                    x = x["img"].to("cuda", dtype)
                encs = tokenizer.encode(x)
                if isinstance(encs, tuple):
                    h = encs[0]
                else:
                    h = encs
                decs = tokenizer.decode(encs, x.shape)
                if isinstance(decs, tuple):
                    y = decs[0]
                else:
                    y = decs

            y.clamp_(-1, 1)

            # Compute mean and std of the latent
            if compute_mean_std:
                # mean_c, std_c = h.mean((0, -2, -1)), h.std((0, -2, -1))
                mean_c, std_c = h.mean(), h.std()
                mean_fn.update(mean_c)
                std_fn.update(std_c)
                # means.append(mean_c)
                # stds.append(std_c)

            # save reconstruction
            if save_img_dir is not None:
                Path(save_img_dir).mkdir(parents=True, exist_ok=True)

                def plot_img(img, path):
                    y_grid = make_grid(img.float(), nrow=1, padding=2)
                    y_grid = (
                        y_grid[rgb_chans].permute(1, 2, 0).detach().cpu().numpy()
                    )  # [h, w, 3]
                    y_grid = (y_grid + 1) / 2
                    y_grid = (y_grid * 255.0).astype(np.uint8)
                    Image.fromarray(y_grid).save(path)
                    logger.info("save reconstruction image", tqdm=True)

                plot_img(y, Path(save_img_dir) / f"recon_{real_data}.png")
                plot_img(x, Path(save_img_dir) / f"gt_{real_data}.png")

            # psnr
            if real_data:
                psnr = PeakSignalNoiseRatio(data_range=1.0).cuda()
                psnr_val = psnr((x + 1) / 2, (y + 1) / 2)
                # logger.info(f"PSNR: {psnr_val}")
                tbar.set_description(f"PSNR: {psnr_val:.4f} - shape: {x.shape}")
                metric.update(psnr_val)

            if use_optim:
                opt.zero_grad()
                y.mean().backward()
                opt.step()

            if check_grad:
                for n, p in tokenizer.named_parameters():
                    if p.grad is None:
                        print(f"{n} grad is None")

            if max_iters <= index:
                break

        if metric.update_count >= 1:
            logger.info(metric.compute())

    # print mean and std of the latent
    if compute_mean_std:
        # m = torch.mean(torch.stack(means), dim=0)
        # s = torch.mean(torch.stack(stds), dim=0)
        m = mean_fn.compute()
        s = std_fn.compute()
        logger.info(f"mean of the latent: {m}")
        logger.info(f"std of the latent: {s}")

    if save_pca_vis:
        if pca_type == "proj":
            feat = tokenizer.get_repa_feature()
        else:
            with torch.no_grad():
                feat = tokenizer.encode(x)
                if isinstance(feat, tuple):
                    feat = feat[0]
        assert feat is not None, "repa feature should not be None"
        feat_pca = feature_pca_sk(feat.float())
        # norm
        feat_pca = (feat_pca - feat_pca.min()) / (feat_pca.max() - feat_pca.min())
        feat_pca = (feat_pca * 255.0).to(torch.uint8).detach().cpu().numpy()[0]
        feat_pca = feat_pca.transpose(1, 2, 0)
        Image.fromarray(feat_pca).save(f"tmp/repa_feature_pca_{pca_type}.png")
        logger.info(f"Save PCA visualization to tmp/repa_feature_pca_{pca_type}.png")


if __name__ == "__main__":
    """
    MODEL_COMPILED=0 python -m src.stage1.cosmos.cosmos_tokenizer
    """
    import lovely_tensors as lt

    lt.monkey_patch()
    # Test lora
    test_tokenizer_forward_backward(
        base_model_ckpt="runs/stage1_cosmos_nested/2025-10-22_19-23-25_cosmos_f8c16p1_unified_hyperspectral_latent_noise=0.0_channel_drop=False/ema/tokenizer/model.safetensors",
        real_data="RS5M",
        save_pca_vis=False,
        pca_type="z",
        is_lora=False,
        lora_ckpt=[
            "runs/stage1_cosmos_lora/2025-09-14_23-31-37_cosmos_lora=lora_r=32_f8c16p1_WV3/peft_ckpt/WV3",
            "runs/stage1_cosmos_lora/2025-09-14_23-27-18_cosmos_lora=lora_r=32_f8c16p1_QB/peft_ckpt/QB",
            "runs/stage1_cosmos_lora/2025-09-15_23-03-10_cosmos_lora=lora_r=32_f8c16p1_IKONOS/peft_ckpt/IKONOS",
            "runs/stage1_cosmos_lora/2025-09-15_17-37-20_cosmos_lora=lora_r=32_f8c16p1_WDC/peft_ckpt/WDC",
            "runs/stage1_cosmos_lora/2025-09-15_17-23-14_cosmos_lora=lora_r=32_f8c16p1_Xiongan/peft_ckpt/Xiongan",
        ],
        lora_changes_chans={
            "WV3": 8,
            "QB": 4,
            "IKONOS": 4,
            "WDC": 191,
            "Xiongan": 256,
        },
        active_lora_name="QB",
        save_img_dir=None,  # "tmp/vis_pansharpening_loras",
        rgb_chans=[0, 1, 2],  # [49, 39, 29],  # RGB
        dtype=torch.bfloat16,
        upscale=1,
        max_iters=2000,
        compute_mean_std=True,
        use_optim=False,
        check_grad=False,
    )

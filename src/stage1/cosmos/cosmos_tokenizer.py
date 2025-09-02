import inspect
import random
import warnings
from collections import OrderedDict, namedtuple
from contextlib import contextmanager
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Callable, Literal, NamedTuple, Sequence, no_type_check, override

import accelerate
import numpy as np
import torch
from peft import (
    LoraConfig,
    get_peft_config,
    get_peft_model,
    set_peft_model_state_dict,
)
from torch import Tensor, nn

import src.stage1.cosmos.modules.blocks as cosmos_block
from src.stage1.cosmos.inference.utils import load_jit_model_shape_matched
from src.stage1.cosmos.modules.layers2d import Decoder, Encoder
from src.stage1.cosmos.modules.utils import Normalize

# Quantiers
from src.stage1.discretization.collections import FSQ
from src.stage1.discretization.collections import BinarySphericalQuantizer as BSQ
from src.stage1.discretization.collections.kl_continuous import (
    DiagonalGaussianDistributionV2 as DiagonalGaussianDistribution,
)
from src.utilities.config_utils import function_config_to_basic_types
from src.utilities.logging import log_print
from src.utilities.network_utils import load_weights_with_shape_check
from stage2.utilities import loss

KLLossBreakDown = namedtuple("KLLossBreakDown", ["posterior", "mean", "logvar"])


def build_mlp(hidden_size, projector_dim, z_dim, is_1d=False):
    ln_cls = nn.Linear if is_1d else partial(nn.Conv2d, kernel_size=1)
    return nn.Sequential(
        ln_cls(hidden_size, projector_dim),
        nn.SiLU(),
        ln_cls(projector_dim, projector_dim),
        nn.SiLU(),
        ln_cls(projector_dim, z_dim),
    )


def _list_or_num_mult(x: list | int | float, factor: int):
    """
    Multiply each element in the list by the factor.
    """
    if not isinstance(x, list):
        assert isinstance(x, (int, float)), "x should be a number or a list of numbers"
        return x * factor
    return [i * factor for i in x]


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
            leave_channels = self.uniform_sampling(size=bs, **self.sample_kwargs)
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
    @override
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

    @function_config_to_basic_types
    def __init__(
        self,
        z_channels: int,
        z_factor: int = 1,
        latent_channels: int = 8,
        loading_type: Literal["pretrained", "nvidia"] | None = "nvidia",
        **kwargs,
    ) -> None:
        super().__init__()
        self._use_repa_loss = kwargs.pop("use_repa_loss", False)
        self._use_vf_loss = kwargs.pop("use_vf_loss", False)
        self._hook_module = kwargs.pop("hook_module", self._hook_module)
        self._vf_on_z_or_module = kwargs.pop(
            "vf_on_z_or_module", self._vf_on_z_or_module
        )
        self._dino_feature_dim = kwargs.pop("dino_feature_dim", 768)
        self.latent_noise_prob = kwargs.pop("latent_noise_prob", 0.0)
        self.use_latent_denoise = self.latent_noise_prob > 0.0

        # < repa or vf projectors
        assert not (self._use_repa_loss and self._use_vf_loss), (
            "repa and vf losses should not be used at the same time"
        )
        if self._use_repa_loss:
            if self._vf_on_z_or_module == "module":
                self._repa_proj = build_mlp(
                    512,
                    self._dino_feature_dim,
                    self._dino_feature_dim,
                )
            else:
                self._repa_proj = build_mlp(
                    latent_channels, self._dino_feature_dim, self._dino_feature_dim
                )
        if self._use_vf_loss:
            if self._vf_on_z_or_module == "z":
                self._vf_proj = build_mlp(
                    latent_channels, self._dino_feature_dim, self._dino_feature_dim
                )
            else:
                self._vf_proj = build_mlp(
                    512,
                    self._dino_feature_dim,
                    self._dino_feature_dim,
                )

        # < FSDP wrapper module
        if kwargs.get("attn_type") in ("none", None):
            self._no_split_modules.remove("AttnBlock")

        # < quantizer
        self.quantizer_type = kwargs.pop("quantizer_type", None)
        self.random_quant = kwargs.pop("random_quant", 0.0)
        self.quantizer: BSQ | FSQ | None
        if self.quantizer_type == "kl":
            if z_factor != 2:
                log_print(
                    "when use kl, z_factor should be 2, set it to 2 explicitly",
                    "warning",
                )
                z_factor = 2
            self.quantizer = DiagonalGaussianDistribution  # not quantizer, compatible with trainer  # type: ignore
        elif self.quantizer_type == "bsq":
            assert latent_channels % 2 == 0, "quantizer out channels should be even"
            self.quantizer = BSQ(
                embed_dim=latent_channels,  # 18 or 36
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
                levels=kwargs.pop("fsq_levels", [8, 8, 8, 5, 5, 5]),
                dim=latent_channels,
                num_codebooks=kwargs.pop("fsq_num_codebooks", 6),
                channel_first=True,
            )
        elif self.quantizer_type is None:
            self.quantizer = None
        else:
            raise ValueError("quantizer type should be one of [kl, bsq, fsq, None]")

        if self.quantizer_type is not None:
            log_print(f"Using quantizer: {self.quantizer.__class__.__name__}")
        else:
            log_print(f"use no quantizer or VAE, the tokenizer is only an AutoEncoder")

        # < Tokenizer configs
        tokenizer_cfg = dict(
            z_channels=z_channels,
            z_factor=z_factor,
            latent_channels=latent_channels,
            **kwargs,
        )
        self.loading_type = loading_type
        self.name = kwargs.get("name", "ContinuousImageTokenizer")
        self.latent_channels = latent_channels
        self.norm_in_quant_conv = kwargs.get("norm_in_quant_conv", False)

        self.in_channels_after_patcher = (
            np.array(kwargs["in_channels"]) * kwargs["patch_size"] ** 2
        ).tolist()
        self.out_channels_after_patcher = (
            np.array(kwargs["out_channels"] * kwargs["patch_size"] ** 2)
        ).tolist()

        # NOTE: encoder and decoder maybe separated, e.g., NVIDIA pretrained tokenizer, or
        # trained on hyperspectral images before
        # if the uni_tokenizer_path is not empty, then the encoder and decoder are loaded directly.
        enc_path = kwargs.pop("enc_path", "")
        dec_path = kwargs.pop("dec_path", "")
        uni_tokenizer_path = kwargs.pop("uni_tokenizer_path", "")

        self.register_buffer("dummy_param", torch.tensor(0))

        # < Encoder and Decoder
        # pretrained encoder and decoder
        if loading_type == "nvidia":
            assert enc_path.endswith(".jit") and dec_path.endswith(".jit")
            # pretrained model from NVIDIA cosmos tokenizer
            assert not self.norm_in_quant_conv, (
                "norm_in_quant_conv is not supported for nvidia pretrained model settings, trian it from scratch"
            )

            log_print(
                f"start from the pretrained model, cosmos tokenizer cfg is {tokenizer_cfg}",
                "debug",
            )
            enc_jit, dec_jit = self.load_pretrained(enc_path, dec_path, tokenizer_cfg)

            # split the encoder and decoder
            encoder = enc_jit[0]
            quant_conv = enc_jit[1]

            decoder = dec_jit[1]
            post_quant_conv = dec_jit[0]

            self.encoder = self.encoder_jit(encoder, quant_conv)
            self.decoder = self.decoder_jit(decoder, post_quant_conv)

        else:
            # encoder and decoder
            # not combine the encoder, for FSDP wrap
            encoder = Encoder(z_channels=z_factor * z_channels, **kwargs)
            decoder = Decoder(z_channels=z_channels, **kwargs)

            # quant_conv and post_quant_conv
            if self.norm_in_quant_conv:
                warnings.warn(
                    '"norm_in_quant_conv" is not supported for pretrained settings and not recommended to use'
                    "it will be removed"
                )
                quant_conv = nn.Sequential(
                    Normalize(z_factor * z_channels, norm_type="gn"),
                    torch.nn.Conv2d(
                        z_factor * z_channels, z_factor * latent_channels, 1
                    ),
                )
            else:
                quant_conv = torch.nn.Conv2d(
                    z_factor * z_channels, z_factor * latent_channels, 1
                )
            post_quant_conv = torch.nn.Conv2d(latent_channels, z_channels, 1)

            self.encoder = self.encoder_jit(encoder, quant_conv)
            self.decoder = self.decoder_jit(decoder, post_quant_conv)

            # Load weights
            if loading_type is not None:
                if kwargs.get("norm_in_quant_conv", False):
                    assert enc_path == "" and dec_path == "", (
                        "norm_in_quant_conv is not supported for pretrained settings, train it from scratch"
                    )
                self.load_pretrained(
                    enc_path, dec_path, uni_tokenizer_path=uni_tokenizer_path
                )

        # token channel drop
        self.use_channel_drop = kwargs.get("use_channel_drop", False)
        if self.use_channel_drop:
            self.channel_drop = NestChannelDrop(**kwargs["channel_drop_config"])
            log_print(f"use channel drop: {kwargs['channel_drop_config']}")

        # register repa hook
        if self._vf_on_z_or_module == "module" and (
            self._use_vf_loss or self._use_repa_loss
        ):
            self.register_feature_hook()

        num_parameters = sum(param.numel() for param in self.parameters())
        log_print(f"model={self.name}, num_parameters={num_parameters:,}")
        log_print(f"z_channels={z_channels}, latent_channels={self.latent_channels}.")

    def encoder_jit(self, encoder, quant_conv):
        return nn.Sequential(
            OrderedDict(
                [
                    ("encoder", encoder),
                    ("quant_conv", quant_conv),
                    # ("distribution", self.distribution),
                ]
            )
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

    def apply_quantizer(self, h, use_quantizer=None):
        _use_quantizer = self._use_quantizer(use_quantizer)
        self._training_latent_cache(h, _use_quantizer)

        # Quantization
        if _use_quantizer:
            if self.quantizer_type == "bsq":
                # here must be l2-normed
                h = nn.functional.normalize(h, dim=1)
                # TODO: bsq not supported channel drop
                hq, bsq_loss, loss_breakdown = self.quantizer(h)

                return hq, bsq_loss, loss_breakdown

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

                return h, kl_loss, loss_breakdown

            elif self.quantizer_type == "fsq":
                # dummy loss
                fsq_loss = torch.tensor(0.0).to(h)
                loss_breakdown = {"fsq_loss": fsq_loss}
                hq, indices = self.quantizer(h)

                return hq, fsq_loss, loss_breakdown

            else:
                raise RuntimeError("can not reach here")

        # autoencoder
        else:
            return h

    def z_aug(self, h):
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

    def _training_latent_cache(self, h: Tensor, use_quantizer: bool):
        # not use_quantizer, save the unquantized latent
        if self.training:
            # Save latent if is AE
            if hasattr(self, "_vf_proj") or hasattr(self, "_repa_proj"):
                self.z = h  # save latent z for repa or vf loss
            else:
                self.z = None

    def encode(
        self, x, use_quantizer=None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, NamedTuple]:
        # Encoder
        h = self.encoder(x)
        # TODO: add additional cross-attention to convert the 2D latent to 1D latent

        # Quantization
        ret = self.apply_quantizer(h, use_quantizer)
        if isinstance(ret, tuple):
            return ret
        else:
            h = ret

        # z augmentions
        h = self.z_aug(h)

        return h

    def decode(self, z: torch.Tensor | tuple, inp_shape: torch.Size, clamp=False):
        # Break down input z losses
        q_loss = loss_breakdown = None
        if self.quantizer_type is not None and isinstance(z, (tuple, list)):
            z, q_loss, loss_breakdown = z
        else:
            assert torch.is_tensor(z), "z should be the (quantized) latent"

        # Decoder
        dec = self.decoder(z, inp_shape[1])  # [b, c, h, w]
        if clamp:
            dec = dec.clamp(-1, 1)

        if self.quantizer_type is not None:
            return dec, q_loss, loss_breakdown
        else:
            return dec

    def forward(self, input: torch.Tensor):
        if cosmos_block.compile_forward_fn:
            torch.compiler.cudagraph_mark_step_begin()
        latent = self.encode(input)
        dec = self.decode(latent, input.shape)

        return dec

    # * --- checkpoint loding --- #
    @no_type_check
    def load_pretrained(
        self,
        enc_path: str,
        dec_path: str,
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
            if uni_tokenizer_path != "":
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
        if conv_stem_reinit:  # if reinit, the the conv stem is just an nn.Conv2d
            module_to_save_layers = ["encoder.encoder.conv_in"]
            if not self.decoder.decoder._wrap_fsdp_last_layer:
                _conv_out_name = "decoder.decoder.conv_out"
            else:
                _conv_out_name = "decoder.decoder.conv_out.wrap_mod"
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
        if not conv_stem_reinit:
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


class TokenizerLoRAMixin(nn.Module):
    def __init__(
        self,
        tokenizer: nn.Module,
        lora_weights: dict[str, str | Path],
        lora_cfg: dict | LoraConfig,
    ):
        super().__init__()
        self._tokenizer = None

        self.encode: Callable
        self.decode: Callable
        self.forward: Callable

        self.tokenizer = tokenizer

        # lora loading
        self.lora_loading(lora_cfg, lora_weights)

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, model):
        self._tokenizer = model
        # has encode, decode, forward attrs
        attrs = ["encode", "decode", "forward", "to", "cuda", "cpu"]
        for attr in attrs:
            assert hasattr(self.tokenizer, attr), f"Tokenizer must has method {attr}"
            # set method for encode and decode
            setattr(self, attr, getattr(self.tokenizer, attr))

    @property
    def peft_config(self):
        return self.model_peft.peft_config

    def lora_loading(
        self,
        lora_cfg: dict | LoraConfig,
        lora_weights: dict[str, Path | str],
        strict=False,
    ):
        if isinstance(lora_cfg, dict):
            peft_cfg = get_peft_config(lora_cfg)
        else:
            peft_cfg = lora_cfg
        assert isinstance(peft_cfg, LoraConfig), (
            "peft_cfg must be an instance of LoraConfig"
        )

        # lora modules
        peft_cfg.target_modules = (
            list(peft_cfg.target_modules) if peft_cfg.target_modules is not None else []
        )
        peft_cfg.target_modules += getattr(
            self.tokenizer, "additional_peft_target_modules", list
        )()
        # re-trained modules
        peft_cfg.modules_to_save = (
            list(peft_cfg.modules_to_save)
            if peft_cfg.modules_to_save is not None
            else []
        )
        peft_cfg.modules_to_save += getattr(self.tokenizer, "peft_modules", list)()
        log_print(
            f"------------------ LoRA config: -----------------------\n"
            f"LoRA modules:\n {peft_cfg.target_modules}\n"
            f"Retrained modules:\n {peft_cfg.modules_to_save}\n"
            "--------------------------------------------------------",
            "debug",
        )

        self.model_peft = getattr(
            self, "model_peft", get_peft_model(self.tokenizer, peft_config=peft_cfg)
        )
        for adapter_name, sd_path in lora_weights.items():
            sd_path = Path(sd_path)
            if sd_path.with_suffix(".safetensors") or sd_path.with_suffix(".pt"):
                sd = accelerate.utils.load_state_dict(str(sd_path))
                self.model_peft.add_adapter(
                    adapter_name=adapter_name,
                    peft_config=peft_cfg,
                    low_cpu_mem_usage=False,
                )
                loaded_result = set_peft_model_state_dict(
                    self.model_peft,
                    sd,
                    adapter_name=adapter_name,
                    ignore_mismatched_sizes=strict,
                    low_cpu_mem_usage=False,
                )
            elif sd_path.is_dir():
                self.model_peft.load_adapter(
                    model_id=str(sd_path),
                    adapter_name=adapter_name,
                    is_trainable=False,
                )
            else:
                raise ValueError(f"Unsupported LoRA loading path: {sd_path}")
            log_print(
                f"Loaded LoRA adaptor: {adapter_name}, load results: {loaded_result}"
            )

        self._tokenizer = self.model_peft.get_base_model()  # avoid rebind

    def change_lora(
        self, lora_name: str, merge=False, no_change_action: str = "warning"
    ):
        if lora_name in self.peft_config:
            self.model_peft.set_adapter(lora_name)
            log_print(f"Successfully switched to LoRA adapter: {lora_name}")
        else:
            available_adapters = list(self.peft_config.keys())
            string = f"Adapter '{lora_name}' not found. Available adapters: {available_adapters}"
            if no_change_action == "warning":
                log_print(string, "warning")
                return
            elif no_change_action == "fallback":
                self._disable_lora()
                log_print(
                    f"Can not change lora {lora_name}, fall back to no lora base model."
                )
            elif no_change_action in ("error", "raise"):
                raise ValueError(string)
            else:
                raise ValueError(f"Can not handle no_change_action: {no_change_action}")
        if merge and lora_name in self.peft_config:
            self.merge_lora_weights()

    def merge_lora_weights(self):
        self.model_peft = self.model_peft.merge_and_unload()
        self.tokenizer = self.model_peft

    def merge_specific_lora(self, adapter_name: str | None = None):
        if adapter_name:
            self.change_lora(adapter_name)
        self.model_peft = self.model_peft.merge_and_unload()
        log_print(f"Merged LoRA weights: {self.model_peft.active_adapter}")
        self.tokenizer = self.model_peft

    @contextmanager
    def disable_lora(self):
        """Context manager to temporarily disable LoRA adapters."""
        try:
            self._disable_lora()
            yield
        except AttributeError:
            yield
        finally:
            try:
                self._enable_lora()
            except AttributeError:
                pass

    def _disable_lora(self):
        if hasattr(self.model_peft, "disable_adapter"):
            self.model_peft.disable_adapter()
        elif hasattr(self.model_peft.base_model, "disable_adapter_layers"):
            self.model_peft.base_model.disable_adapter_layers()

    def _enable_lora(self):
        if hasattr(self.model_peft.base_model, "enable_adapter_layers"):
            self.model_peft.base_model.enable_adapter_layers()


# * --- test --- * #

from src.utilities.logging import catch_any


@catch_any()
def test_tokenizer_fb(
    is_lora=False,
    count_params=False,
    real_data: str | None = None,
    use_optim=False,
    device="cuda:1",
    base_model_ckpt: str = "runs/stage1_cosmos/2025-08-20_20-14-19_cosmos_f8c16p1_unified_hyperspectral_latent_noise=0.0_channel_drop=False/ema/tokenizer/model.safetensors",
    lora_ckpt: str | None = None,
    check_grad=False,
    show_mem_usage=False,
    save_pca_vis=False,
    pca_type: str = "proj",
    other_model_kwargs: dict | None = None,
    idx_of_dl: int = 1,
    save_img_dir: str | None = None,
    rgb_chans: list[int] = [4, 2, 0],
    dtype=torch.bfloat16,
    upscale: int = 1,
):
    from contextlib import nullcontext

    from fvcore.nn import parameter_count_table
    from PIL import Image
    from torchmetrics.aggregation import MeanMetric
    from torchmetrics.image import PeakSignalNoiseRatio
    from torchvision.utils import make_grid
    from tqdm import tqdm, trange

    from src.data.hyperspectral_loader import get_fast_test_hyperspectral_data
    from src.stage1.utilities.losses.repa.feature_pca import feature_pca_sk
    from src.utilities.network_utils import load_peft_model_checkpoint, mem_context

    torch.cuda.set_device(device)

    config = {
        "attn_resolutions": [32],
        "channels": 128,
        "channels_mult": [2, 4, 4],
        "dropout": 0.0,
        "in_channels": [3, 4, 8, 10, 12, 13, 32, 50, 150, 175, 202, 224, 242, 368],
        "spatial_compression": 8,
        "num_res_blocks": 2,
        "out_channels": [3, 4, 8, 10, 12, 13, 32, 50, 150, 175, 202, 224, 242, 368],
        "resolution": 1024,
        "patch_size": 1,
        "patch_method": "haar",
        "latent_channels": 16,
        "z_channels": 16,
        "z_factor": 1,
        "name": "CI",
        "formulation": "AE",
        "encoder": "Default",
        "decoder": "Default",
        "act_checkpoint": True,
        "uni_tokenizer_path": base_model_ckpt,
        "loading_type": "pretrained" if Path(base_model_ckpt).exists() else None,
        "hook_for_repa": False,
        "block_name": "res_block",  # res_block, res_moe
        "quantizer_type": None,
        "enc_moe": False,
        "dec_moe": False,
        "padding_mode": "reflect",
        "norm_type": "gn",
        "norm_groups": 32,
        "attn_type": "none",
        # repa
        "use_repa_loss": True,
        "use_vf_loss": False,
        "vf_on_z_or_module": "z",
        "dino_feature_dim": 1024,
    }
    if other_model_kwargs:
        config.update(other_model_kwargs)
    tokenizer = ContinuousImageTokenizer(**config).cuda()
    tokenizer = tokenizer.eval().to(dtype)

    if is_lora:
        # load lora layers
        assert lora_ckpt is not None
        peft_cfg, tokenizer = load_peft_model_checkpoint(tokenizer, lora_ckpt)
        tokenizer = tokenizer.to("cuda", dtype)
        tokenizer.eval()
        log_print("Loaded peft model checkpoint. PEFT config: \n" + str(peft_cfg))

        # peft modules
        # log_print(str(tokenizer.peft_fully_finetune_modules()))
        # log_print(str(tokenizer.peft_lora_modules()))
        # log_print(tokenizer.get_last_enc_layer())

    if count_params:
        log_print(parameter_count_table(tokenizer))

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
        else:
            dl = get_fast_test_hyperspectral_data(batch_size=1, data_type=real_data)
            for i, sample in enumerate(tqdm(dl, desc="Get sample ...")):
                if i >= idx_of_dl:
                    break
                x = sample["img"]
    else:
        x = torch.randn(1, 12, 256, 256).to("cuda", dtype)
    x = x.to(dtype).cuda()
    if upscale != 1:
        x = torch.nn.functional.interpolate(
            x, scale_factor=upscale, align_corners=True, mode="bicubic"
        )

    if use_optim:
        opt = torch.optim.Adam(tokenizer.parameters(), lr=1e-4, fused=True)

    metric = MeanMetric().cuda()
    ctx = torch.no_grad if not (use_optim or check_grad) else torch.enable_grad
    mem_ctx = mem_context(device) if show_mem_usage else nullcontext()
    with torch.autocast("cuda", dtype) and ctx():
        with mem_ctx:
            y = tokenizer(x)
            y.clamp_(-1, 1)
            log_print(y.shape)

            # save reconstruction
            if save_img_dir is not None:

                def plot_img(img, path):
                    y_grid = make_grid(img.float(), nrow=1, padding=2)
                    y_grid = (
                        y_grid[rgb_chans].permute(1, 2, 0).detach().cpu().numpy()
                    )  # [h, w, 3]
                    y_grid = (y_grid + 1) / 2
                    y_grid = (y_grid * 255.0).astype(np.uint8)
                    Image.fromarray(y_grid).save(path)
                    log_print("save reconstruction image")

                plot_img(y, Path(save_img_dir) / f"recon_{real_data}.png")
                plot_img(x, Path(save_img_dir) / f"gt_{real_data}.png")

            # psnr
            if real_data:
                psnr = PeakSignalNoiseRatio(data_range=1.0).cuda()
                psnr_val = psnr((x + 1) / 2, (y + 1) / 2)
                log_print(f"PSNR: {psnr_val}")
                metric.update(psnr_val)

            if use_optim:
                opt.zero_grad()
                y.mean().backward()
                opt.step()

            if check_grad:
                for n, p in tokenizer.named_parameters():
                    if p.grad is None:
                        print(f"{n} grad is None")

        if metric.update_count >= 1:
            log_print(metric.compute())

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
        log_print(f"Save PCA visualization to tmp/repa_feature_pca_{pca_type}.png")


if __name__ == "__main__":
    # test_tokenizer_fb(
    #     real_data="RS5M",
    #     save_pca_vis=True,
    #     pca_type="z",
    #     idx_of_dl=12,
    # )

    # Test lora
    test_tokenizer_fb(
        base_model_ckpt="runs/stage1_cosmos/2025-08-20_20-14-19_cosmos_f8c16p1_unified_hyperspectral_latent_noise=0.0_channel_drop=False/ema/tokenizer/model.safetensors",
        real_data="WV3",
        save_pca_vis=True,
        pca_type="z",
        idx_of_dl=16,
        is_lora=True,
        lora_ckpt="runs/stage1_cosmos_lora/2025-08-28_23-17-54_cosmos_lora=lora_r=32_f8c16p1_WV3/peft_ckpt/WV3",
        save_img_dir="tmp",
        rgb_chans=[2, 1, 0],  # RGB
        dtype=torch.float32,
        upscale=4,
    )

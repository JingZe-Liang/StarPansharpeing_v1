import inspect
import warnings
from collections import OrderedDict, namedtuple
from functools import partial
from itertools import chain
from typing import Literal, NamedTuple, Sequence, override

import numpy as np
import torch
from torch import Tensor, nn

from src.stage1.cosmos.inference.utils import load_jit_model_shape_matched
from src.stage1.cosmos.modules.layers2d import Decoder, Encoder
from src.stage1.cosmos.modules.utils import Normalize
from src.stage1.discretization.collections import BinarySphericalQuantizer as BSQ
from src.stage1.discretization.collections.kl_continuous import (
    DiagonalGaussianDistributionV2 as DiagonalGaussianDistribution,
)
from src.utilities.config_utils import function_config_to_basic_types
from src.utilities.logging import log_print
from src.utilities.network_utils import load_weights_with_shape_check

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


def _to_two_tuple(x):
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, tuple):
        assert len(x) == 2, "x should be a tuple of length 2"
        return x
    else:
        raise ValueError("x should be an int or a tuple of length 2")


def _list_or_num_mult(x: list | int | float, factor: int):
    """
    Multiply each element in the list by the factor.
    """
    if not isinstance(x, list):
        assert isinstance(x, (int, float)), "x should be a number or a list of numbers"
        return x * factor
    return [i * factor for i in x]


class NestChannelDrop(nn.Module):
    def __init__(
        self,
        learnable: bool = False,
        drop_type: str = "uniform_4",
        max_channels: int = 12,  # MMSeg dataset
        img_size: tuple[int] | int = 256,
        drop_prob: float = 0.5,
    ):
        super().__init__()
        self.learnable = learnable
        self.max_channels = max_channels
        self.img_size = _to_two_tuple(img_size)
        self.drop_prob = drop_prob

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

        if self.learnable:
            self.dropped_x = nn.Parameter(torch.zeros(1, 1, *self.img_size))
            self.dropped_x.data.normal_(0, 0.2)
        else:
            self.register_buffer(
                "dropped_x", torch.zeros(1, 1, *self.img_size), persistent=False
            )

        self.register_buffer(
            "channel_arange", torch.arange(self.max_channels), persistent=False
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

    def forward(self, z, inference_channels: int | None = None):
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
        else:
            raise ValueError(
                f"drop_type {self.drop_type} not supported, only exp and uniform are supported"
            )

        # drop channels

        # 1. expand the cached empty z
        if self.dropped_x.shape[-2:] != z.shape[-2:]:
            if self.learnable:
                z_empty = nn.functional.interpolate(
                    self.dropped_x,
                    size=z.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                z_empty = z_empty.expand(bs, -1, -1, -1)
            else:
                z_empty = torch.zeros_like(z)
        else:
            z_empty = self.dropped_x.expand(bs, -1, -1, -1)

        # 2. drop channels
        _channels = self.channel_arange[None].expand(bs, -1)  # type: ignore
        _cond = _channels < leave_channels.to(_channels)
        z = torch.where(_cond.unsqueeze(-1).unsqueeze(-1).expand_as(z), z, z_empty)

        return z


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


class ContinuousImageTokenizer(nn.Module):
    _no_split_modules: list[str] = ["ResnetBlock", "AttnBlock"]
    _hook_for_repa: bool = False
    _hook_module: str = "decoder.decoder.mid.block_2"  # "decoder.decoder.up.1.block.2"
    _hook_feature: torch.Tensor | None = None
    _proj_for_vf: bool = False
    _vf_on_z_or_module: Literal["z", "module"] = (
        "module"  # vf loss on z or module output
    )
    _dino_feature_dim: int = 768  # [768, 1024]

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
        self._hook_for_repa = kwargs.pop("hook_for_repa", False)
        self._proj_for_vf = kwargs.pop("proj_for_vf", False)
        self._hook_module = kwargs.pop("hook_module", self._hook_module)
        self._vf_on_z_or_module = kwargs.pop(
            "vf_on_z_or_module", self._vf_on_z_or_module
        )
        assert not (self._hook_for_repa and self._proj_for_vf), (
            "repa and vf losses should not be used at the same time"
        )
        if self._hook_for_repa:
            self._repa_proj = build_mlp(
                512, self._dino_feature_dim, self._dino_feature_dim
            )
        if self._proj_for_vf:
            if self._vf_on_z_or_module == "z":
                self._vf_proj = build_mlp(
                    latent_channels, self._dino_feature_dim, self._dino_feature_dim
                )
            else:
                self._vf_proj = build_mlp(
                    512, self._dino_feature_dim, self._dino_feature_dim
                )
        if kwargs.get("attn_type") in ("none", None):
            self._no_split_modules.remove("AttnBlock")

        self.quantizer_type = kwargs.pop("quantizer_type", None)
        assert self.quantizer_type in [
            "kl",
            "bsq",
            None,
        ], "quantizer_type should be bsq or kl"

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
        if self.quantizer_type is not None:
            log_print(f"Using quantizer: {self.quantizer.__class__.__name__}")
        else:
            log_print(f"use no quantizer or kl, just AE to reconstruct")

        tokenizer_cfg = dict(
            z_channels=z_channels,
            z_factor=z_factor,
            latent_channels=latent_channels,
            **kwargs,
        )
        self.loading_type = loading_type
        self.name = kwargs.get("name", "ContinuousImageTokenizer")
        self.latent_channels = latent_channels

        self.in_channels_after_patcher = _list_or_num_mult(
            kwargs["in_channels"], kwargs["patch_size"] ** 2
        )
        self.out_channels_after_patcher = _list_or_num_mult(
            kwargs["out_channels"], kwargs["patch_size"] ** 2
        )

        # NOTE: encoder and decoder maybe separated, e.g., NVIDIA pretrained tokenizer, or
        # trained on hyperspectral images before
        # if the uni_tokenizer_path is not empty, then the encoder and decoder are loaded directly.
        enc_path = kwargs.pop("enc_path", "")
        dec_path = kwargs.pop("dec_path", "")
        uni_tokenizer_path = kwargs.pop("uni_tokenizer_path", "")

        # 强制参数内存连续性
        self.register_buffer("dummy_param", torch.tensor(0))

        # pretrained encoder and decoder
        if loading_type == "nvidia":
            assert enc_path.endswith(".jit") and dec_path.endswith(".jit")
            # pretrained model from NVIDIA cosmos tokenizer
            assert not kwargs.get("norm_in_quant_conv", False), (
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
            if kwargs.get("norm_in_quant_conv", False):
                warnings.warn(
                    '"norm_in_quant_conv" is not supported for pretrained settings and not recommended to use'
                    "it will be removed"
                )
                quant_conv = nn.Sequential(
                    Normalize(z_factor * z_channels, norm_type="rms_triton"),
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
        if self._hook_for_repa or self._vf_on_z_or_module == "module":
            self.register_feature_hook()

        num_parameters = sum(param.numel() for param in self.parameters())
        log_print(f"model={self.name}, num_parameters={num_parameters:,}")
        log_print(f"z_channels={z_channels}, latent_channels={self.latent_channels}.")

    #     if loading_type is None:
    #         self.apply(self.init_weights)
    #         log_print('<red>initialize</> weights done.')

    # def init_weights(self,m):
    #     if isinstance(m, nn.Conv2d):
    #         nn.init.kaiming_normal_(m.weight, mode='fan_out')
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.Linear):
    #         nn.init.trunc_normal_(m.weight, std=0.02)
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)

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
        # return nn.Sequential(
        #     OrderedDict(
        #         [
        #             ("post_quant_conv", post_quant_conv),
        #             ("decoder", decoder),
        #         ]
        #     )
        # )

        return DecoderSequential(post_quant_conv, decoder)

    def register_feature_hook(self):
        def hook(module, input, output):
            self._hook_feature = output

        self.get_submodule(self._hook_module).register_forward_hook(hook)
        log_print(
            f"[Cosmos Tokenizer]: module {self._hook_module} is registered for hook"
        )

    @torch.autocast("cuda", torch.bfloat16)
    def get_repa_feature(self):
        # only one feature
        if hasattr(self, "_repa_proj"):
            return self._repa_proj(self._hook_feature)

        return None

    @torch.autocast("cuda", torch.bfloat16)
    def get_vf_feature(self):
        if hasattr(self, "_vf_proj"):
            if self._vf_on_z_or_module == "z":
                # project on latent
                assert self.z is not None, "z should be set before get_vf_feature"
                return self._vf_proj(self.z)
            elif self._vf_on_z_or_module == "module":
                return self._vf_proj(self._hook_feature)
                # proj on block out
            else:
                raise ValueError(
                    f"vf loss should get feature when vf is computed on {self._vf_on_z_or_module}"
                )

        return None

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

    def encode(self, x) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, NamedTuple]:
        h = self.encoder(x)

        if self.quantizer_type == "bsq":
            # here must be l2-normed
            h = nn.functional.normalize(h, dim=1)
            self.z = h if hasattr(self, "_vf_proj") else None

            # TODO: bsq not supported channel drop
            self.quantizer: BSQ
            hq, bsq_loss, loss_breakdown = self.quantizer(h)

            return hq, bsq_loss, loss_breakdown

        elif self.quantizer_type == "kl":
            self.z = h if hasattr(self, "_vf_proj") else None
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
                h = self.channel_drop(h)

            return h, kl_loss, loss_breakdown

        if self.use_channel_drop:
            h = self.channel_drop(h)

        self.z = h if hasattr(self, "_vf_proj") else None  # save latent z for vf loss

        return h

    def decode(self, z: torch.Tensor | Sequence, inp_shape: torch.Size):
        q_loss = loss_breakdown = None
        if self.quantizer_type is not None and isinstance(z, Sequence):
            z, q_loss, loss_breakdown = z
        else:
            assert torch.is_tensor(z), "z should be the (quantized) latent"

        dec = self.decoder(z, inp_shape[1])  # [b, c, h, w]

        if self.quantizer_type is not None:
            return dec, q_loss, loss_breakdown
        else:
            return dec

    def forward(self, input: torch.Tensor):
        latent = self.encode(input)
        dec = self.decode(latent, input.shape)

        return dec

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
            import accelerate

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

    def peft_first_last_convs_module_names(self, add_norms: bool = False):
        module_to_save_layers = [
            # convs
            "encoder.encoder.conv_in",
            (
                "decoder.decoder.conv_out"
                if not self.decoder.decoder._wrap_fsdp_last_layer
                else "decoder.decoder.conv_out.wrap_mod"
            ),
            # quant convs need to be fully finetuned
            # "encoder.quant_conv",
            # "decoder.quant_conv",
        ]

        # convolution and normalization layers
        for name, module in self.named_modules():
            if "norm" in name and (not isinstance(module, nn.Identity)) and add_norms:
                module_to_save_layers.append(name)
                log_print(
                    f"[Cosmos Tokenizer]: add norm {name} to fully finetune", "debug"
                )

        # projections for repa or vf losses
        if self._hook_for_repa:
            module_to_save_layers.append("_repa_proj")
        if self._proj_for_vf:
            module_to_save_layers.append("_vf_proj")

        return module_to_save_layers

    def additional_peft_target_modules(self):
        add_tgt_modules = []
        for name, module in self.named_modules():
            if (
                (not name.startswith("_"))
                and isinstance(module, (nn.Conv2d, nn.Linear))
                and name
                not in (
                    "encoder.encoder.conv_in",
                    "decoder.decoder.conv_out",
                    # "encoder.quant_conv",
                    # "decoder.quant_conv",
                )
            ):
                add_tgt_modules.append(name)
                log_print(f"[Cosmos Tokenizer]: add {name} to lora finetune", "debug")

        return add_tgt_modules

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


if __name__ == "__main__":
    config = {
        "attn_resolutions": [32],
        "channels": 128,
        "channels_mult": [2, 4, 4],
        "dropout": 0.0,
        "in_channels": [3, 10, 12, 32, 8, 13, 50, 4],
        "spatial_compression": 8,
        "num_res_blocks": 2,
        "out_channels": [3, 10, 12, 32, 8, 13, 50, 4],
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
        "act_checkpoint": False,
        "uni_tokenizer_path": "runs/stage1_cosmos/2025-06-03_17-30-14_cosmos_f8c16p4_unified_hyperspectral_repa_no_pix_shuffle/ema/tokenizer/model.safetensors",
        "hook_for_repa": False,
        "block_name": "res_block",  # res_block, res_moe
        "quantizer_type": None,
        "loading_type": "pretrained",  # "pretrained",
        "enc_moe": False,
        "dec_moe": False,
        "padding_mode": "reflect",
        "norm_type": "gn",
        "norm_groups": 32,
        "attn_type": "none",
        # "resample_norm_type": "gn",
        # "downsample_type": "ConvPixelUnshuffle",
        # "downsample_shortcut": "averaging",
        # "upsample_type": "ConvPixelShuffle",
        # "upsample_shortcut": "duplicating",
    }
    torch.cuda.set_device(1)
    tokenizer = ContinuousImageTokenizer(**config).cuda()
    # tokenizer = torch.compile(tokenizer)

    # load lora layers
    # from src.utilities.network_utils import load_peft_model_checkpoint

    # peft_cfg, tokenizer = load_peft_model_checkpoint(
    #     tokenizer,
    #     "runs/stage1_cosmos_lora/2025-05-30_01-34-59_lora_fintune_f8c16p4_MMSeg_YREB_uni_pretrained/peft_ckpt",
    # )
    # tokenize = tokenizer.to("cuda", torch.bfloat16)
    # tokenizer.eval()
    # log_print("Loaded peft model checkpoint. PEFT config: \n" + str(peft_cfg))

    # peft modules
    # log_print(str(tokenizer.peft_first_last_convs_module_names()))
    # log_print(str(tokenizer.additional_peft_target_modules()))
    # tokenizer.get_last_enc_layer()

    # from fvcore.nn import parameter_count_table

    # print(parameter_count_table(tokenizer))

    # x = torch.randn(1, 12, 256, 256).to("cuda", torch.bfloat16)
    from torchmetrics.image import PeakSignalNoiseRatio

    from src.data.hyperspectral_loader import get_fast_test_hyperspectral_data

    dl = get_fast_test_hyperspectral_data(batch_size=1, data_type="MMSeg")
    dl_iter = iter(dl)
    tokenizer = tokenizer.eval().to(torch.bfloat16)

    # x = torch.randn(8, 12, 256, 256, dtype=torch.bfloat16).cuda()
    # opt = torch.optim.Adam(tokenizer.parameters(), lr=1e-4, fused=True)

    # from src.utilities.optim import get_muon_optimizer

    # opt = get_muon_optimizer(
    #     tokenizer.named_parameters(),
    #     lr=1e-4,
    #     weight_decay=0.1,
    # )

    from torchmetrics.aggregation import MeanMetric

    from src.utilities.logging.print import catch_any

    metric = MeanMetric().cuda()
    with torch.autocast("cuda", torch.bfloat16) and catch_any() and torch.no_grad():
        for i in range(20):
            x = next(dl_iter)["img"].cuda().to(torch.bfloat16)
            y = tokenizer(x)
            # yy = ((y[0, [2, 1, 0]].permute(1, 2, 0).float() + 1) / 2).cpu().numpy()
            # xx = ((x[0, [2, 1, 0]].permute(1, 2, 0).float() + 1) / 2).cpu().numpy()

            # psnr
            psnr = PeakSignalNoiseRatio(data_range=1.0).cuda()
            psnr_val = psnr((x + 1) / 2, (y + 1) / 2)
            print(f"PSNR: {psnr_val}")
            print(y.shape)

            # print(y.shape)
            # y.mean().backward()

            # grads
            # for n, p in tokenizer.named_parameters():
            #     if p.grad is None:
            #         print(f"{n} grad is None")

            # opt.zero_grad()
            # y.mean().backward()
            # opt.step()
            # metric.update(psnr_val)

        # print(metric.compute())

        # import time

        # time.sleep(20)

        # feat = tokenizer.get_repa_feature()
        # psnr = PeakSignalNoiseRatio(data_range=1.0).cuda()
        # psnr_val = psnr((x + 1) / 2, (y + 1) / 2)
        # logging.debug(y.shape)
        # logging.debug(psnr_val)

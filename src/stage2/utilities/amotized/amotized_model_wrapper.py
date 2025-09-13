import functools
from typing import Any, Callable, Literal, TypedDict, no_type_check

import torch
from torch import Tensor, nn
from torch.nn import Module


def amotizing_call_model(
    detokenizer: Module,
    pansp_model: Module,
    amotize_fn: Callable | str,
):
    def wrapper(lrms_latent, pan_latent, lrms, pan):
        amotize_call = (
            amotize_fn
            if isinstance(amotize_fn, Callable)
            else getattr(pansp_model, amotize_fn)
        )
        assert callable(amotize_call), "amotize_fn must be callable"

        latent_sr, pixel_amot = pansp_model(lrms_latent, pan_latent, lrms, pan)
        sr_detok = detokenizer(latent_sr)
        sr = amotize_call(sr_detok, pixel_amot)

        return latent_sr, sr

    return wrapper


# *==============================================================
# * Amotized Model (two branches)
# * latent branch amotizes the pixel branch
# * to get the final super-resolution image
# *==============================================================


type AmotizedType = (
    Literal[
        "pixel_to_pixel_fusion",
        "latent_to_pixel_fusion",
        "one_model_conditioning",
    ]
    | str
)

ModelOutput = TypedDict(
    "ModelOutput",
    {
        "latent_out": Tensor,
        "pixel_from_latent": Tensor | None,
        "pixel_out": Tensor,
    },
)


class AmotizedModelMixin(nn.Module):
    """
    # Amotized model

    # < version 1: simple merging
    #                                |-------> latent loss
    # latent -> latent branch -> sr latent -> de-tokenizer --- sr (not accurate) -> sr -> pixel loss
    # (lrms, pan) -> pixel branch ---------------------------------↑ (e.g., add)

    # < version 2: pixel to pixel fusion
    # latent -> latent branch -> sr latent -> de-tokenizer --- sr (not accurate)
    #                        ↓ -------------------------------------|
    # (lrms, pan) ---> pixel branch -> sr

    # < version 3: latent to pixel fusion
    # latent -> latent branch -> sr latent -> de-tokenizer --- sr (not accurate)
    #                        ↓ ------|
    # (lrms, pan) ---> pixel branch -> sr -> pixel loss

    # < version 4: pixel to latent fusion
    # latent -------------------------> big transformer -----> latent -> de-tokenizer -> pixel loss
    # pixels -----> CNN ---> patcher -----↑                     ↓----> latent loss (optional)

    """

    def __init__(
        self,
        pixel_model: Module,
        amotized_model: Module,
        decoder_fn: Module | Callable[[Tensor], Tensor],
        amotize_type: AmotizedType,
        backward_decoder: bool = False,
        learn_decoder: bool = False,
    ):
        super().__init__()
        self.pixel_model = pixel_model
        self.amotized_model = amotized_model
        self.amotize_type = amotize_type
        self.decoder = decoder_fn  # not saved in state_dict if not learn_decoder

        self.backward_decoder = backward_decoder
        self.learn_decoder = learn_decoder
        if self.learn_decoder:
            assert isinstance(self.decoder, nn.Module), (
                "decoder_fn must be a nn.Module if learn_decoder is True"
            )

        # amotizing type check
        if not (
            self.amotize_type
            in (
                "pixel_to_pixel_fusion",
                "latent_to_pixel_fusion",
                "one_model_conditioning",
            )
            or self.amotize_type[:11] == "simple_merge"
        ):
            raise ValueError(
                f"amotize_type {self.amotize_type} is not supported for AmotizedModel"
            )

    def _single_tensor_to_tuple(self, x: Tensor | tuple) -> tuple[Tensor | Any, ...]:
        return (x,) if torch.is_tensor(x) else x

    def simple_merge_forward(
        self, pixel_in: tuple, latent_in: tuple, amotize_type: str
    ):
        latent_out = self.amotized_model(*latent_in)
        amotized_pixel_from_latent = self.decoder(
            latent_out
        )  # latent loss ? (if use, the pixel branch is just residual)
        pixel_out = self.pixel_model(*pixel_in)
        if amotize_type == "add":
            out = (
                pixel_out + amotized_pixel_from_latent  # -> pixel loss
            )  # todo: add more complex merging logic
        else:
            raise ValueError(f"Unknown amotize_type: {amotize_type}")

        return {
            "latent_out": latent_out,
            "pixel_from_latent": amotized_pixel_from_latent,
            "pixel_out": out,
        }

    def pixel_to_pixel_fusion_forward(self, pixel_in: tuple, latent_in: tuple):
        latent_out = self.amotized_model(*latent_in)  # -> latent loss
        amotized_pixel_from_latent = self.decoder(latent_out)  # -> pixel loss
        # args: (pixel_in, amotized_pixel_from_latent)
        pixel_ins = pixel_in + self._single_tensor_to_tuple(amotized_pixel_from_latent)
        pixel_out = self.pixel_model(*pixel_ins)  # -> pixel loss

        return {
            "latent_out": latent_out,
            "pixel_from_latent": amotized_pixel_from_latent,
            "pixel_out": pixel_out,
        }

    def latent_to_pixel_fusion_forward(self, pixel_in: tuple, latent_in: tuple):
        latent_out = self.amotized_model(*latent_in)  # -> latent loss
        # args: pixel_in, latent_out
        pixel_model_ins = pixel_in + self._single_tensor_to_tuple(latent_out)
        pixel_out = self.pixel_model(*pixel_model_ins)  # -> pixel loss

        return {
            "latent_out": latent_out,
            "pixel_from_latent": None,  # No pixel from latent in this case, left to decoder out from forward
            "pixel_out": pixel_out,
        }

    def pixel_to_latent_forward(self, pixel_in: tuple, latent_in: tuple):
        pixel_out_as_cond = self.pixel_model(*pixel_in)
        model_conds = latent_in + self._single_tensor_to_tuple(pixel_out_as_cond)
        latent_out = self.amotized_model(*model_conds)  # -> latent loss

        pixel_out = None
        if self.backward_decoder or self.learn_decoder:
            pixel_out = self.decoder(latent_out)  # -> pixel loss

        return {
            "latent_out": latent_out,
            "pixel_from_latent": pixel_out,
            "pixel_out": pixel_out,
        }

    @no_type_check
    def forward(
        self, pixel_in: Tensor | tuple, latent_in: Tensor | tuple
    ) -> ModelOutput:
        pixel_in = self._single_tensor_to_tuple(pixel_in)
        latent_in = self._single_tensor_to_tuple(latent_in)

        if self.amotize_type[:11] == "simple_merge":
            return self.simple_merge_forward(
                pixel_in, latent_in, self.amotize_type.rsplit("_")[-1]
            )
        elif self.amotize_type == "pixel_to_pixel_fusion":
            return self.pixel_to_pixel_fusion_forward(pixel_in, latent_in)
        elif self.amotize_type == "latent_to_pixel_fusion":
            return self.latent_to_pixel_fusion_forward(pixel_in, latent_in)
        elif self.amotize_type == "one_model":
            return self.one_model_conditioning_forward(pixel_in, latent_in)
        else:
            raise ValueError(f"Unknown amotize_type: {self.amotize_type}")


# * --- Instances --- #

from src.stage2.pansharpening.models.transformer import Transformer
from src.stage2.pansharpening.models.vitamin_conv import (
    ConvCfg,
    VitaminCfg,
    VitaminModel,
)


def vitamin_transformer_amotized_in_pixel_small(
    ms_chan: int, pan_chan: int, latent_chan: int, decoder_fn
):
    latent_model = Transformer(
        in_dim=latent_chan,
        dim=256,
        depth=8,
        num_heads=8,
        mlp_ratio=4,
        drop=0.0,
        patch_size=2,
        out_channels=256,
    )

    conv_cfg = ConvCfg(
        expand_ratio=2.0, kernel_size=3, act_layer="gelu", norm_layer="layernorm2d"
    )
    vitamin_cfg = VitaminCfg(
        stem_width=32,
        embed_dim=[64, 192, 192],
        depths=[2, 2, 2],
        ms_channel=ms_chan,
        pan_channel=pan_chan,
        condition_channel=256,
        use_residual=True,
        conv_cfg=conv_cfg,
    )

    pixel_model = VitaminModel(vitamin_cfg)

    amotized_model = AmotizedModelMixin(
        decoder_fn=decoder_fn,
        amotize_type="latent_to_pixel_fusion",
        amotized_model=latent_model,
        backward_decoder=False,
        learn_decoder=False,
        pixel_model=pixel_model,
    )

    return amotized_model


ALL_AMOTIZED_MODELS = {
    "transformer_vitamin_latent_to_pixel_small": vitamin_transformer_amotized_in_pixel_small,
}

# * --- Test --- * #


def test_amotized_pansharpening_model():
    from src.stage2.pansharpening.models.transformer import Transformer
    from src.stage2.pansharpening.models.vitamin_conv import (
        ConvCfg,
        VitaminCfg,
        VitaminModel,
    )

    device = "cuda:1"
    torch.cuda.set_device(device)

    transformer = Transformer(
        in_dim=16,
        dim=256,
        depth=8,
        num_heads=8,
        mlp_ratio=4,
        drop=0.0,
        patch_size=2,
        out_channels=256,
    ).cuda()

    conv_cfg = ConvCfg(
        expand_ratio=2.0, kernel_size=3, act_layer="gelu", norm_layer="layernorm2d"
    )
    vitamin_cfg = VitaminCfg(
        stem_width=32,
        embed_dim=[64, 192, 192],
        depths=[2, 2, 2],
        pan_channel=1,
        ms_channel=8,
        condition_channel=256,
        use_residual=True,
        conv_cfg=conv_cfg,
    )
    vitamin_model = VitaminModel(vitamin_cfg).cuda()

    bs = 2
    ms_latent = torch.randn(bs, 16, 32, 32).cuda()
    pan_latent = torch.randn(bs, 16, 32, 32).cuda()
    ms = torch.randn(bs, 8, 256, 256).cuda()
    pan = torch.randn(bs, 1, 256, 256).cuda()

    amotized_model = AmotizedModelMixin(
        decoder_fn=lambda x: x,  # Dummy decoder
        amotize_type="latent_to_pixel_fusion",
        amotized_model=transformer,
        backward_decoder=False,
        learn_decoder=False,
        pixel_model=vitamin_model,
    ).cuda()

    y = amotized_model(pixel_in=(ms, pan), latent_in=(ms_latent, pan_latent))
    print(y["pixel_out"].shape)

    # Parameters
    from fvcore.nn import FlopCountAnalysis, parameter_count_table

    print(parameter_count_table(amotized_model, max_depth=3))


if __name__ == "__main__":
    test_amotized_model()

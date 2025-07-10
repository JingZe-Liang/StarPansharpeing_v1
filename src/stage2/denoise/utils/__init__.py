# disable ruff
# ruff: noqa

# unifying the classes
from .noise_adder import UniHSINoiseAdder, UniHSINoiseAdderKornia

# numpy classes
from .add_noise import (
    AddNoiseBlind as AddNoiseBlind_CHW_np,
    AddNoiseBlindv1 as AddNoiseBlindv1_CHW_np,
    AddNoiseBlindv2 as AddNoiseBlindv2_CHW_np,
    AddNoiseNoniid as AddNoiseNoniid_CHW_np,
    AddNoiseNoniid_v2 as AddNoiseNoniid_v2_CHW_np,
    AddNoiseImpulse as AddNoiseImpulse_CHW_np,
    AddNoiseDeadline as AddNoiseDeadline_CHW_np,
    AddNoiseStripe as AddNoiseStripe_CHW_np,
    AddNoiseInpainting as AddNoiseInpainting_CHW_np,
    AddNoiseComplex as AddNoiseComplex_CHW_np,
)

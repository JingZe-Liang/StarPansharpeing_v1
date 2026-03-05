import torch
from bitsandbytes.optim import (
    LAMB,
    AdamW8bit,
    # AdEMAMix8bit,
    LAMB8bit,
    PagedAdamW8bit,
    # PagedAdEMAMix8bit,
    RMSprop8bit,
)

# CAME optimizer: SANA
from .came import CAME

# Distributional optimizers from Dino
_dion_imported = False
try:
    from .dion.dion import Dion, DionMixedPrecisionConfig, DionReference, DionSimple
    # from .dion.dion import Muon as MuonAll2All

    _dion_imported = True
except ImportError:
    print("Dion optimizers not available")

# FSDP optimizers: https://github.com/ethansmith2000/fsdp_optimizers/tree/main
from .kron import Kron
from .kron_mars import KronMars

# MARS optimizers
from .mars import MARS

# from https://github.com/samsja/muon_fsdp_2/blob/main/src/zeroband/muon.py
from .muon import Muon
from .muon_fsdp import Muon as MounFSDP_v1
from .muon_fsdp_v2 import Muon as MounFSDP_v2
from .muon_fused import MuonFSDP
from .muon_quantized_fused import QuantizedMuonFSDP
from .magma_skipupdate_wrapper import (
    MagmaMaskConfig,
    MagmaSkipUpdateWrapper,
    create_torch_magma_optimizer,
    wrap_optimizer_with_magma,
)
from .muon_ball_fused import MuonBallFused
from .spectral_ball_fused import SpectralBallFused
from .normuon_fused import NorMuonFSDP
from .muon_triton import Muon as MuonTriton
from .sana_came import CAME8BitWrapper, CAMEWrapper, Lion

# add to torch.serialization.safe_globals
torch.serialization.add_safe_globals(
    [
        LAMB,
        AdamW8bit,
        # AdEMAMix8bit,
        LAMB8bit,
        PagedAdamW8bit,
        # PagedAdEMAMix8bit,
        RMSprop8bit,
        Kron,
        KronMars,
        MARS,
        # Muon
        Muon,
        MuonTriton,
        MuonFSDP,
        QuantizedMuonFSDP,
        MagmaMaskConfig,
        MagmaSkipUpdateWrapper,
        MuonBallFused,
        SpectralBallFused,
        MounFSDP_v1,
        MounFSDP_v2,
        NorMuonFSDP,
        CAME,
        CAME8BitWrapper,
        CAMEWrapper,
        Lion,
    ]
)
if _dion_imported:
    torch.serialization.add_safe_globals(
        [
            Dion,
            DionMixedPrecisionConfig,
            DionReference,
            DionSimple,
            # MuonAll2All,
        ]
    )


# Muon optimizer parameters getter
from typing import Any, Iterable

from ..config_utils import function_config_to_basic_types


@function_config_to_basic_types
def get_muon_optimizer(named_parameters: Iterable, **other_muon_kwargs):
    muon_p, adamw_p = Muon.clear_muon_adamw_params(named_parameters)

    return Muon(muon_params=muon_p, adamw_params=adamw_p, **other_muon_kwargs)


@function_config_to_basic_types
def get_muon_triton_optimizer(
    named_parameters: Iterable,
    general_defaults: dict[str, Any] | None = None,
    muon_defaults: dict[str, Any] | None = None,
    adamw_defaults: dict[str, Any] | None = None,
):
    """
    Create a MuonTriton optimizer with separate configurations for Muon and AdamW parameters.

    Parameters
    ----------
    named_parameters : Iterable
        Iterable of named model parameters to optimize
    general_defaults : dict[str, Any]
        General default configuration for all parameters
    muon_defaults : dict[str, Any] | None, optional
        Specific defaults for Muon optimizer (2D+ parameters)
    adamw_defaults : dict[str, Any] | None, optional
        Specific defaults for AdamW optimizer (1D parameters)

    Returns
    -------
    MuonTriton
        Configured MuonTriton optimizer instance
    """
    gp = MuonTriton.clear_muon_adamw_params(named_parameters)

    # Set general defaults as base configuration
    if general_defaults is None:
        general_defaults = {}

    gp[0].update(general_defaults)
    gp[1].update(general_defaults)

    # Override with muon-specific defaults
    if muon_defaults is not None:
        gp[0].update(muon_defaults)

    # Override with adamw-specific defaults
    if adamw_defaults is not None:
        gp[1].update(adamw_defaults)

    optim = MuonTriton(gp, defaults=general_defaults)
    return optim


def _disable_heavyball_optim_compilation():
    try:
        import heavyball

        heavyball.utils.compile_mode = None  # type: ignore[invalid-assignment]
    except ImportError:
        pass


_disable_heavyball_optim_compilation()

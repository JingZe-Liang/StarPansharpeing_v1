import torch
from bitsandbytes.optim import (
    LAMB,
    AdamW8bit,
    AdEMAMix8bit,
    LAMB8bit,
    PagedAdamW8bit,
    PagedAdEMAMix8bit,
    RMSprop8bit,
)

# CAME optimizer: SANA
from .came import CAME

# FSDP optimizers: https://github.com/ethansmith2000/fsdp_optimizers/tree/main
from .kron import Kron
from .kron_mars import KronMars

# MARS optimizers
from .mars import MARS

# from https://github.com/samsja/muon_fsdp_2/blob/main/src/zeroband/muon.py
from .muon import Muon
from .muon_fsdp_v2 import Muon as MounFSDP_v2
from .muon_fsdp import Muon as MounFSDP_v1
from .sana_came import CAME8BitWrapper, CAMEWrapper, Lion

# add to torch.serialization.safe_globals
torch.serialization.add_safe_globals(
    [
        LAMB,
        AdamW8bit,
        AdEMAMix8bit,
        LAMB8bit,
        PagedAdamW8bit,
        PagedAdEMAMix8bit,
        RMSprop8bit,
        Kron,
        KronMars,
        MARS,
        Muon,
        MounFSDP_v2,
        MounFSDP_v1,
        CAME,
        CAME8BitWrapper,
        CAMEWrapper,
        Lion,
    ]
)


from typing import Iterable


def get_moun_optimizer(named_parameters: Iterable, **other_muon_kwargs):
    muon_p, adamw_p = Muon.clear_muon_adamw_params(named_parameters)

    return Muon(muon_params=muon_p, adamw_params=adamw_p, **other_muon_kwargs)

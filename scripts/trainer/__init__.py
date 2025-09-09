# ruff: noqa
# Change Detection
from scripts.trainer.hyper_latent_change_detection_trainer import (
    HyperCDTrainer,
    _configs_dict as cd_configs,
    _key as cd_key,
)

# Denoising
from scripts.trainer.hyper_latent_denoise_trainer import (
    DenoisingTrainer,
    _configs_dict as denoise_configs,
    _key as denoise_key,
)

# Pansharpening
from scripts.trainer.hyper_latent_pansharpening_trainer import (
    PansharpeningTrainer,
    _configs as pansharp_configs,
    _key as pansharp_key,
)

# Segmentation
from scripts.trainer.hyper_latent_segmentation_trainer import (
    HyperSegmentationTrainer,
    _configs_dict as seg_configs,
    _key as seg_key,
)

# Unmixing
from scripts.trainer.hyper_latent_unmixing_trainer import (
    UnmixingTrainer,
    _configs as unmixing_configs,
    _key as unmixing_key,
)

# Tokenizer
from scripts.trainer.hyperspectral_image_tokenizer_trainer import (
    CosmosHyperspectralTokenizerTrainer,
    _configs_dict as tokenizer_configs,
    _key as tokenizer_key,
)

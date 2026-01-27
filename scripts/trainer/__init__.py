# ruff: noqa
import os
import importlib
import lazy_loader

# Env
os.environ["MODEL_COMPILED"] = "0"  # no compiled model
os.environ["SHELL_LOG_LEVEL"] = "DEBUG"  # set shell log level to DEBUG to reduce logs

# Define aliases for direct mapping: alias_name -> (module_name, original_name)
_aliases = {
    # Classification
    "classification_configs": ("hyper_latent_classification_trainer", "_configs_dict"),
    "classification_key": ("hyper_latent_classification_trainer", "_key"),
    # Change Detection
    "cd_configs": ("hyper_latent_change_detection_trainer", "_configs_dict"),
    "cd_key": ("hyper_latent_change_detection_trainer", "_key"),
    # Denoising
    "denoise_configs": ("hyper_latent_denoise_trainer", "_configs_dict"),
    "denoise_key": ("hyper_latent_denoise_trainer", "_key"),
    # Pansharpening
    "pansharp_configs": ("hyper_latent_pansharpening_trainer", "_configs"),
    "pansharp_key": ("hyper_latent_pansharpening_trainer", "_key"),
    # Segmentation
    "seg_configs": ("hyper_latent_segmentation_trainer", "_configs_dict"),
    "seg_key": ("hyper_latent_segmentation_trainer", "_key"),
    # Unmixing
    "unmixing_configs": ("hyper_latent_unmixing_trainer", "_configs"),
    "unmixing_key": ("hyper_latent_unmixing_trainer", "_key"),
    # Tokenizer
    "tokenizer_configs": ("hyperspectral_image_tokenizer_trainer", "_configs_dict"),
    "tokenizer_key": ("hyperspectral_image_tokenizer_trainer", "_key"),
}

_lazy_getattr, _lazy_dir, _lazy_all = lazy_loader.attach(
    __name__,
    submod_attrs={
        "hyper_latent_classification_trainer": ["HyperClassificationTrainer"],
        "hyper_latent_change_detection_trainer": ["HyperCDTrainer"],
        "hyper_latent_denoise_trainer": ["DenoisingTrainer"],
        "hyper_latent_pansharpening_trainer": ["PansharpeningTrainer"],
        "hyper_latent_segmentation_trainer": ["HyperSegmentationTrainer"],
        "hyper_latent_unmixing_trainer": ["UnmixingTrainer"],
        "hyperspectral_image_tokenizer_trainer": ["CosmosHyperspectralTokenizerTrainer"],
    },
)

__all__ = _lazy_all + list(_aliases.keys())


def __getattr__(name):
    if name in _aliases:
        module_name, attr_name = _aliases[name]
        module = importlib.import_module(f".{module_name}", __name__)
        return getattr(module, attr_name)
    return _lazy_getattr(name)


def __dir__():
    return __all__

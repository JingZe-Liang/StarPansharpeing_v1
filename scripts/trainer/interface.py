import argparse
import os
import sys
import warnings
from contextlib import nullcontext

warnings.filterwarnings(
    "ignore",
    "subpackages can technically be lazily loaded",
    category=RuntimeWarning,
    module="lazy_loader",
)
warnings.filterwarnings(
    "ignore",
    "Importing from timm.models.layers is deprecated",
    category=FutureWarning,
    module="timm.models.layers",
)


# Set os.env
os.environ["SHELL_LOG_LEVEL"] = "INFO"

import hydra
from accelerate.state import PartialState

# Change Detection
from scripts.trainer.hyper_latent_change_detection_trainer import HyperCDTrainer
from scripts.trainer.hyper_latent_change_detection_trainer import (
    _configs_dict as cd_configs,
)
from scripts.trainer.hyper_latent_change_detection_trainer import _key as cd_key
from scripts.trainer.single_hyper_image_change_detection_trainer import (
    SingleHyperImageCDTrainer,
)
from scripts.trainer.single_hyper_image_change_detection_trainer import (
    _configs_dict as cd_single_configs,
)
from scripts.trainer.single_hyper_image_change_detection_trainer import _key as cd_single_key

# Classification
from scripts.trainer.hyper_latent_classification_trainer import HyperClassificationTrainer
from scripts.trainer.hyper_latent_classification_trainer import (
    _configs_dict as classification_configs,
)
from scripts.trainer.hyper_latent_classification_trainer import _key as classification_key

# Denoising
from scripts.trainer.hyper_latent_denoise_trainer import DenoisingTrainer
from scripts.trainer.hyper_latent_denoise_trainer import (
    _configs_dict as denoise_configs,
)
from scripts.trainer.hyper_latent_denoise_trainer import _key as denoise_key

# Pansharpening
from scripts.trainer.hyper_latent_pansharpening_trainer import PansharpeningTrainer
from scripts.trainer.hyper_latent_pansharpening_trainer import (
    _configs as pansharp_configs,
)
from scripts.trainer.hyper_latent_pansharpening_trainer import _key as pansharp_key

# Segmentation
from scripts.trainer.hyper_latent_segmentation_trainer import HyperSegmentationTrainer
from scripts.trainer.hyper_latent_segmentation_trainer import (
    _configs_dict as seg_configs,
)
from scripts.trainer.hyper_latent_segmentation_trainer import _key as seg_key

# Unmixing
from scripts.trainer.hyper_latent_unmixing_trainer import UnmixingTrainer
from scripts.trainer.hyper_latent_unmixing_trainer import _configs as unmixing_configs
from scripts.trainer.hyper_latent_unmixing_trainer import _key as unmixing_key

# Tokenizer
from scripts.trainer.hyperspectral_image_tokenizer_trainer import (
    CosmosHyperspectralTokenizerTrainer,
)
from scripts.trainer.hyperspectral_image_tokenizer_trainer import (
    _configs_dict as tokenizer_configs,
)
from scripts.trainer.hyperspectral_image_tokenizer_trainer import _key as tokenizer_key

# Utilities
from src.utilities.logging import catch_any, log
from src.utilities.train_utils.cli import argsparse_cli_args, print_colored_banner

# Mappings
trainer_mapping = {
    "tokenize": CosmosHyperspectralTokenizerTrainer,
    "denoise": DenoisingTrainer,
    "pansharpening": PansharpeningTrainer,
    "unmixing": UnmixingTrainer,
    "segmentation": HyperSegmentationTrainer,
    "change_detection": HyperCDTrainer,
    "change_detection_single": SingleHyperImageCDTrainer,
    "classification": HyperClassificationTrainer,
}
cfg_mapping = {
    "tokenize": tokenizer_configs,
    "denoise": denoise_configs,
    "pansharpening": pansharp_configs,
    "unmixing": unmixing_configs,
    "segmentation": seg_configs,
    "change_detection": cd_configs,
    "change_detection_single": cd_single_configs,
    "classification": classification_configs,
}
default_cfg_mapping = {
    "tokenize": tokenizer_key,
    "denoise": denoise_key,
    "pansharpening": pansharp_key,
    "unmixing": unmixing_key,
    "segmentation": seg_key,
    "change_detection": cd_key,
    "change_detection_single": cd_single_key,
    "classification": classification_key,
}
config_path_mapping = {
    "tokenize": "../configs/tokenizer_gan",
    "denoise": "../configs/denoising",
    "pansharpening": "../configs/pansharpening",
    "unmixing": "../configs/unmixing",
    "segmentation": "../configs/segmentation",
    "change_detection": "../configs/change_detection",
    "change_detection_single": "../configs/change_detection",
    "classification": "../configs/classification",
}


def task_trainer():
    parser = argparse.ArgumentParser(description="Trainer for different hyperspectral image tasks")
    parser.add_argument("-t", "--task", type=str, choices=list(trainer_mapping.keys()), required=True)
    args, unknown = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown

    log(f"Trainer task: {args.task}")
    print_colored_banner(args.task)
    trainer = trainer_mapping[args.task]
    default_cfg = default_cfg_mapping[args.task]
    cfg_dict = cfg_mapping[args.task]

    cli_default_dict = {
        "config_name": default_cfg,
        "only_rank_zero_catch": True,
    }
    chosen_cfg, cli_args = argsparse_cli_args(cfg_dict, cli_default_dict)  # type: ignore[arg-type]
    config_path = config_path_mapping[args.task]
    breakpoint()

    # Main entrypoint
    @hydra.main(
        config_path=config_path,
        config_name=chosen_cfg,
        version_base=None,
    )
    def main(cfg):
        if is_rank_zero := PartialState().is_main_process:
            if cli_args.only_rank_zero_catch:
                catcher = catch_any if is_rank_zero else nullcontext
            else:
                catcher = catch_any

            with catcher():
                log(f"Using config: {chosen_cfg}")
                log(f"CLI args: {cli_args}")
                trainer_instance = trainer(cfg)
                trainer_instance.run()

    main()


if __name__ == "__main__":
    task_trainer()

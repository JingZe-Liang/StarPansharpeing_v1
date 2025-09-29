import os
import sys
import warnings
from contextlib import nullcontext
import typer

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

# *==============================================================
# * Import Trainers and their config mappings
# *==============================================================

from scripts.trainer import (
    CosmosHyperspectralTokenizerTrainer,
    tokenizer_configs,
    tokenizer_key,

    DenoisingTrainer,
    denoise_configs,
    denoise_key,

    HyperCDTrainer,
    cd_configs,
    cd_key,

    HyperSegmentationTrainer,
    seg_configs,
    seg_key,

    PansharpeningTrainer,
    pansharp_configs,
    pansharp_key,

    UnmixingTrainer,
    unmixing_configs,
    unmixing_key,
)

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
}
cfg_mapping = {
    "tokenize": tokenizer_configs,
    "denoise": denoise_configs,
    "pansharpening": pansharp_configs,
    "unmixing": unmixing_configs,
    "segmentation": seg_configs,
    "change_detection": cd_configs,
}
default_cfg_mapping = {
    "tokenize": tokenizer_key,
    "denoise": denoise_key,
    "pansharpening": pansharp_key,
    "unmixing": unmixing_key,
    "segmentation": seg_key,
    "change_detection": cd_key,
}
config_path_mapping = {
    "tokenize": "scripts/configs/tokenizer_gan",
    "denoise": "scripts/configs/denoising",
    "pansharpening": "scripts/configs/pansharpening",
    "unmixing": "scripts/configs/unmixing",
    "segmentation": "scripts/configs/segmentation",
    "change_detection": "scripts/configs/change_detection",
}

# *==============================================================
# * Trainer Entrypoint
# *==============================================================

app = typer.Typer(
    help="Trainer for different hyperspectral image tasks",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)


@app.command(rich_help_panel="Main Interface")
def task_trainer(
    ctx: typer.Context,
    task: str = typer.Argument(
        ...,
        help="Training task to perform",
        case_sensitive=False,
        callback=lambda val: val in trainer_mapping.keys()
        or typer.BadParameter(
            f"Task must be one of: {', '.join(trainer_mapping.keys())}"
        ),
    ),
):
    unknown = ctx.args
    sys.argv = [sys.argv[0]] + unknown

    # Task / trainer / cfgs mappings
    log(f"Trainer task: {task}")
    print_colored_banner(task)
    trainer = trainer_mapping[task]
    default_cfg = default_cfg_mapping[task]
    cfg_dict = cfg_mapping[task]

    # Choose any config and change Hydra configs
    cli_default_dict = {
        "config_name": default_cfg,
        "only_rank_zero_catch": True,
    }
    chosen_cfg, cli_args = argsparse_cli_args(cfg_dict, cli_default_dict)
    config_path = config_path_mapping[task]

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

    # Run the task
    main()


if __name__ == "__main__":
    typer.run(task_trainer)

import math
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, Sequence, cast

import accelerate
import hydra
import numpy as np
import PIL.Image as Image
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.state import PartialState
from accelerate.tracking import TensorBoardTracker
from accelerate.utils.deepspeed import DummyOptim, DummyScheduler
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file
from torch import Tensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._fully_shard import FSDPModule
from torch.distributed.tensor import DTensor
from torchmetrics.aggregation import MeanMetric
from torchvision.utils import make_grid
from tqdm import tqdm, trange
import wandb

from src.data.window_slider import WindowSlider, model_predict_patcher
from src.stage2.segmentation.data.data_split import SingleImageHyperspectralSegmentationDataset
from src.stage2.segmentation.data.single_mat_loader import SingleMatDataset, label_background_recover
from src.stage2.segmentation.loss.seg_loss import boost_strap_update_label
from src.stage2.segmentation.metrics import HyperSegmentationScore
from src.stage2.utilities.loss import HyperSegmentationLoss
from src.utilities.config_utils import to_easydict_recursive
from src.utilities.config_utils import to_object as to_cont
from src.utilities.logging import dict_round_to_list_str, log, log_any_into_writter, set_logger_file
from src.utilities.network_utils import load_peft_model_checkpoint
from src.utilities.network_utils.Dtensor import safe_dtensor_operation
from src.utilities.network_utils.network_loading import load_weights_with_shape_check
from src.utilities.train_utils.state import StepsCounter, dict_tensor_sync, metrics_sync
from src.utilities.train_utils.visualization import get_rgb_image, visualize_segmentation_map
from loguru import logger


@dataclass
class SegmentationOutput:
    loss: Tensor
    pred_pixel: Tensor
    log_losses = None  # dict[str, Tensor]

    def __post_init__(self):
        assert self.loss is not None, "loss should not be None"
        if isinstance(self.loss, tuple):
            self.loss, self.log_losses = self.loss
        else:
            self.log_losses = {"seg_loss": self.loss.detach()}

    def __getitem__(self, name):
        return getattr(self.__dict__, name, None)


class HyperSegmentationTrainer:
    """
    Hyperspectral image segmentation/classification trainer.
    For usual classification task, the dataloader takes in a single image and a single label.
    Randomly taking some pixels from the image and label to directly input the model, or clip patches
    around the pixels (as center) into the model. If the test set is also taken from the same image,
    it will leak the training infomation into the test set.

    Avoid this by (spatially) predefining the train/test rigorously into patches and do not mixing them
    in training stage.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.train_cfg = cfg.train
        self.dataset_cfg = cfg.dataset
        self.ema_cfg = cfg.ema
        self.val_cfg = cfg.val
        self.metric_cfg = cfg.metric

        # accelerator
        self.accelerator: Accelerator = hydra.utils.instantiate(cfg.accelerator)
        self._trackers_name = self.train_cfg.log.log_with
        accelerate.utils.set_seed(2025)

        # logger
        log_file = self.configure_logger()

        # attributes
        self.device = self.accelerator.device
        torch.cuda.set_device(self.accelerator.local_process_index)
        self.dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "no": torch.float32,
        }[self.accelerator.mixed_precision]
        self.log_msg("Log file is saved at: {}".format(log_file))
        self.log_msg("Weights will be saved at: {}".format(self.proj_dir))
        self.log_msg("Training is configured and ready to start.")

        # Initialize indexing mode
        self._use_single_img_indexing = getattr(self.train_cfg, "use_single_img_indexing", False)
        if self._use_single_img_indexing:
            self.log_msg("Using HyperSIGMA-style indexing for loss computation")
            # Initialize accumulation buffers for HyperSIGMA mode
            self._accumulated_preds: list[torch.Tensor] = []
            self._accumulated_gts: list[torch.Tensor] = []
            self._accumulated_indices: list[int] = []
            self._current_image_id: int = 0
            self._patches_per_image: int | None = getattr(self.train_cfg, "patches_per_image", None)
            self.macro_batch_size: int = getattr(self.train_cfg, "macro_batch_size", 32)
            if self._patches_per_image is None:
                self.log_msg(
                    "Warning: patches_per_image not specified, will auto-detect",
                    level="WARNING",
                )
        else:
            self.log_msg("Using standard segmentation loss computation")

        # is zero 2 or 3, not EMA
        _dpsp_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        _fsdp_plugin: accelerate.utils.FullyShardedDataParallelPlugin = getattr(  # type: ignore
            self.accelerator.state, "fsdp_plugin", None
        )
        self.no_ema = False
        self._is_ds = _dpsp_plugin is not None
        if self._is_ds:
            self.log_msg("[Deepspeed]: using deepspeed plugin")
            self.no_ema = _dpsp_plugin.deepspeed_config["zero_optimization"][  # type: ignore
                "stage"
            ] in [2, 3]

        self._is_fsdp = _fsdp_plugin is not None
        if self._is_fsdp:
            self.log_msg("[FSDP]: using Fully Sharded Data Parallel plugin")
            self.no_ema = True

        # dataloader
        dataset_name = self.dataset_cfg.dataset_name
        self.log_msg(f"[Data]: using dataset {dataset_name}")

        self.train_dataset: SingleMatDataset
        self.val_dataset: SingleMatDataset
        self.train_dataset, self.train_dataloader = hydra.utils.instantiate(self.dataset_cfg.train)
        if hasattr(self.dataset_cfg, "val"):
            logger.info(f"[Data]: get independent validation dataset.")
            self.val_dataset, self.val_dataloader = hydra.utils.instantiate(self.dataset_cfg.val)
        else:
            logger.info(f"[Data]: use single image dataset.")
            self.val_dataset, self.val_dataloader = self.train_dataset, self.train_dataloader

        self.ds_is_single_image = getattr(self.dataset_cfg.consts, "is_single_image", False)
        self._val_unsampled_mask = None  # if is single image training, sampled from one image
        if self.ds_is_single_image:
            self._val_unsampled_mask = self.val_dataset.get_unsampled_area().to(self.device)

        if _dpsp_plugin is not None:
            self.accelerator.deepspeed_plugin.deepspeed_config[  # type: ignore
                "train_micro_batch_size_per_gpu"
            ] = self.dataset_cfg.batch_size_train

        # setup the segmentation model
        self.setup_segmentation_model()  # setup the segmentation model

        # optimizers and lr schedulers
        self.optim, self.sched = self.get_optimizer_lr_scheduler()

        # EMA models and accelerator prepare
        self.prepare_for_training()
        self.prepare_ema_models()

        # loss
        self.segment_loss: HyperSegmentationLoss | Callable
        if hasattr(self.train_cfg, "segment_loss"):
            self.segment_loss = hydra.utils.instantiate(self.train_cfg.segment_loss)
        else:
            # default loss
            self.segment_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.log_msg(f"use segmentation loss: {self.segment_loss.__class__.__name__}")

        # training state counter
        self.train_state = StepsCounter(["train", "val"])

        # clear GPU memory
        torch.cuda.empty_cache()

    def setup_segmentation_model(self):
        self.model = hydra.utils.instantiate(self.cfg.segment_model)

        segment_name = getattr(self.train_cfg, "segment_name", None) or self.model.__class__.__name__
        self.log_msg(f"use segmentation model: {segment_name}")

        # print parameters table
        from fvcore.nn import parameter_count_table

        self.log_msg("\n" + parameter_count_table(self.model))

    def prepare_ema_models(self):
        if self.no_ema:
            return

        self.ema_model = hydra.utils.instantiate(self.ema_cfg)(self.model)
        self.log_msg(f"create EMA model for segmentation")

    def configure_logger(self) -> Path:
        self.logger = logger

        log_file = Path(self.train_cfg.proj_dir)
        if self.train_cfg.log.log_with_time:
            str_time = time.strftime("%Y-%m-%d_%H-%M-%S")
            log_file = log_file / str_time
        if self.train_cfg.log.run_comment is not None:
            log_file = Path(log_file.as_posix() + "_" + self.train_cfg.log.run_comment)
        log_file = log_file / "log.log"

        # when distributed, there should be the same log_file
        if self.accelerator.use_distributed:
            input_list: list[Path | None]
            if self.accelerator.is_main_process:
                input_list = [log_file] * self.accelerator.num_processes
            else:
                input_list = [None] * self.accelerator.num_processes
            output_list: list[Path | None] = [None]
            torch.distributed.scatter_object_list(output_list, input_list, src=0)
            log_file = cast(Path, output_list[0])
            assert isinstance(log_file, Path), "log_file type should be Path"

        # logger
        # self.logger.remove()
        log_format_in_file = (
            "<green>[{time:MM-DD HH:mm:ss}]</green> "
            "- <level>[{level}]</level> "
            "- <cyan>{file}:{line}</cyan> - <level>{message}</level>"
        )
        log_format_in_cmd = (
            "{time:HH:mm:ss} - {level.icon} <level>[{level}:{file.name}:{line}]</level>- <level>{message}</level>"
        )
        if not self.train_cfg.debug:
            set_logger_file(log_file, level="info")
            set_logger_file(
                log_file.parent / "debug.log", level="debug", filter=lambda record: record["level"].no <= 10
            )
        #     self.logger.add(
        #         log_file,
        #         format=log_format_in_file,
        #         level="INFO",
        #         rotation="10 MB",
        #         enqueue=True,
        #         backtrace=True,
        #         colorize=False,
        #     )
        # self.logger.add(
        #     sys.stdout,
        #     format=log_format_in_cmd,
        #     level="DEBUG",
        #     backtrace=True,
        #     colorize=True,
        # )

        # make log dir
        log_dir = log_file.parent
        if not self.train_cfg.debug:
            log_dir.mkdir(parents=True, exist_ok=True)

        # copy cfg
        if not self.train_cfg.debug:
            yaml_cfg = OmegaConf.to_yaml(self.cfg, resolve=True)
            cfg_cp_path = log_file.parent / "config" / "config_total.yaml"
            cfg_cp_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cfg_cp_path, "w") as f:
                f.write(yaml_cfg)
            self.logger.info(f"[Cfg]: configuration saved to {cfg_cp_path}")

        # accelerate project configuration
        self.proj_dir = log_dir
        self.accelerator.project_configuration.project_dir = str(self.proj_dir)

        # tensorboard logger
        if not self.train_cfg.debug:
            tenb_dir = log_dir / "tensorboard"
            self.accelerator.project_configuration.logging_dir = str(tenb_dir)
            if self.accelerator.is_main_process:
                self.logger.info(f"[Tensorboard]: logger files at {tenb_dir}")

                self.accelerator.init_trackers("train")
                if "tensorboard" in self._trackers_name:
                    self.tb_logger: TensorBoardTracker = self.accelerator.get_tracker("tensorboard")  # type: ignore
                    logger.log("NOTE", "will log with tensorboard")

                if "wandb" in self._trackers_name:
                    import wandb

                    self.wandb_logger: wandb.Run = wandb.init(
                        project="RS_Hyperspectral_Segmentation",
                        name=getattr(self.train_cfg.log, "exp_name", None),
                        config=to_cont(self.cfg),
                    )
                    logger.log("NOTE", "will log with wandb")

                if "swanlab" in self._trackers_name:
                    import swanlab

                    self.swanlab_logger = swanlab.init(
                        project="RS_Hyperspectral_Segmentation",
                        workspace="iamzihan",
                        experiment_name=getattr(self.train_cfg.log, "exp_name", None),  # type: ignore
                        logdir=str(Path(log_dir) / "swanlab"),
                        resume="never",
                        config=to_cont(self.cfg),
                    )
                    logger.log("NOTE", "Will log with swanlab")

            #### Log code into dir
            from src.utilities.logging import get_python_pkg_env, zip_code_into_dir

            code_dir = ["src/data", "src/stage2", "scripts"]
            zip_code_into_dir(save_dir=log_dir, code_dir=code_dir)
            get_python_pkg_env(file=str(log_dir / "requirements.txt"))

            logger.info("[Code]: code files are zipped and saved.")
            logger.info("[Env]: python package environment requirements are saved.")

        return log_file

    @property
    def loggers(self) -> dict[str, Any]:
        _loggers = {}
        if hasattr(self, "tb_logger"):
            _loggers["tensorboard"] = self.tb_logger
        if hasattr(self, "wandb_logger"):
            _loggers["wandb"] = self.wandb_logger
        if hasattr(self, "swanlab_logger"):
            _loggers["swanlab"] = self.swanlab_logger
        return _loggers

    def tenb_log_any(
        self,
        log_type: Literal["metric", "image", "grad_norm_per_param", "grad_norm_sum"],
        logs: dict,
        step: int | None = None,
        **kwargs,
    ):
        log_any_into_writter(log_type, self.loggers, logs, step, **kwargs)

    def log_msg(self, *msgs, only_rank_zero=True, level="INFO", sep=",", **kwargs):
        assert level.lower() in [
            "info",
            "warning",
            "error",
            "debug",
            "critical",
        ], f"Unknown level {level}"

        def str_msg(*msg):
            return sep.join([str(m) for m in msg])

        log_fn = getattr(self.logger, level.lower())

        if only_rank_zero:
            if self.accelerator.is_main_process:
                log_fn(str_msg(*msgs), **kwargs)
        else:  # not only rank zero
            with self.accelerator.main_process_first():
                msg_string = str_msg(*msgs)
                # prefix rank info
                msg_string = f"rank-{self.accelerator.process_index} | {msg_string}"
                log_fn(msg_string, **kwargs)

    def get_optimizer_lr_scheduler(
        self,
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        # optimizers
        if (
            self.accelerator.state.deepspeed_plugin is None
            or "optimizer" not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):

            def _optimizer_creater(optimizer_cfg, params_getter):
                if "muon" in optimizer_cfg._target_:
                    self.log_msg("[Optimizer]: using muon optimizer")
                    # is muon optimizer function
                    named_params = params_getter(with_name=True)
                    return hydra.utils.instantiate(optimizer_cfg)(named_parameters=named_params)
                else:
                    self.log_msg(f"[Optimizer]: using optimizer: {optimizer_cfg._target_}")
                    params = params_getter(with_name=False)
                    return hydra.utils.instantiate(optimizer_cfg)(params)

            _get_model_params = (
                lambda with_name: self.model.named_parameters() if with_name else self.model.parameters()
            )
            model_opt = _optimizer_creater(self.train_cfg.segment_optim, _get_model_params)
        else:
            model_opt = DummyOptim([{"params": list(self.model.parameters())}])

        # schedulers
        if (
            self.accelerator.state.deepspeed_plugin is None
            or "scheduler" not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            model_sched = hydra.utils.instantiate(self.train_cfg.segment_sched)(optimizer=model_opt)
        else:
            model_sched = DummyScheduler(model_opt)

        # set the heavyball optimizer without torch compiling
        is_heavyball_opt = lambda opt: opt.__class__.__module__.startswith("heavyball")
        if is_heavyball_opt(model_opt):
            import heavyball

            heavyball.utils.compile_mode = None  # type: ignore
            self.log_msg(
                "use heavyball optimizer, it will compile the optimizer, "
                "for efficience testing the scripts, disable the compilation."
            )

        return model_opt, model_sched  # type: ignore

    def set_fsdp_cpu_local_tensor_to_each_rank(self, model: nn.Module):
        if not self._is_fsdp:
            return model

        self.log_msg(
            "FSDP module seems do not move the original parameter (_local_tensor) on the"
            "correct rank, we need to manually move them on cuda while using `to_local` or `redistributed` methods",
            level="WARNING",
        )
        _cpu_device = torch.device("cpu")
        for name, param in model.named_parameters():
            if isinstance(param, DTensor) and param.device == _cpu_device:
                param._local_tensor = param._local_tensor.to(self.device)
                self.log_msg(f"set {name} local_tensor on cuda", level="DEBUG")

        return model

    def set_hypersigma_indexing_mode(self, enable: bool = True):
        """Enable or disable HyperSIGMA-style indexing for loss computation.

        Args:
            enable: Whether to use HyperSIGMA-style indexing
        """
        self._use_single_img_indexing = enable
        mode_str = "enabled" if enable else "disabled"
        self.log_msg(f"HyperSIGMA indexing mode {mode_str}")

        # Initialize or clear accumulation buffers
        if enable:
            self._accumulated_preds = []
            self._accumulated_gts = []
            self._accumulated_indices = []
            self._current_image_id = 0
        else:
            # Cclear buffers when disabling
            if hasattr(self, "_accumulated_preds"):
                self._accumulated_preds.clear()
                self._accumulated_gts.clear()
                self._accumulated_indices.clear()

    def get_current_indexing_mode(self) -> str:
        """Get current indexing mode.

        Returns:
            str: Either 'hypersigma' or 'standard'
        """
        return "hypersigma" if self._use_single_img_indexing else "standard"

    def set_hypersigma_config(self, patches_per_image: int | None = None):
        """Set HyperSIGMA configuration parameters.

        Args:
            patches_per_image: Number of patches per complete image
        """
        if patches_per_image is not None:
            self._patches_per_image = patches_per_image
            self.log_msg(f"HyperSIGMA patches per image set to {patches_per_image}")

    def get_hypersigma_config(self) -> dict:
        """Get current HyperSIGMA configuration.

        Returns:
            dict: Configuration parameters
        """
        return {
            "enabled": self._use_single_img_indexing,
            "patches_per_image": getattr(self, "_patches_per_image", None),
            "accumulated_patches": len(getattr(self, "_accumulated_preds", [])),
        }

    def prepare_for_training(self) -> None:
        # FIXME: FSDP2 seems do not support the sync_bn, find a way to fix it.
        if self._is_fsdp or self.accelerator.distributed_type in (
            accelerate.utils.DistributedType.MULTI_GPU,
            accelerate.utils.DistributedType.FSDP,
        ):  # seems that FSDP does not support synchronized batchnorm
            # discriminator may have batch norm layer
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.log_msg("[Model] convert model bn to sync batch norm")

        # if use FSDP2
        if self._is_fsdp and self.accelerator.is_fsdp2:
            # set models with property dtype
            self.model.dtype = torch.float

        self.model, self.optim = self.accelerator.prepare(self.model, self.optim)

        self.train_dataloader, self.val_dataloader = self.accelerator.prepare(
            self.train_dataloader, self.val_dataloader
        )

    def step_train_state(self, mode="train"):
        self.train_state.update(mode)

    def ema_update(self, mode="segment"):
        assert mode == "segment"
        if self.no_ema:
            # not support ema when is deepspeed zero2 or zero3
            return

        self.ema_model.update()

    def get_training_sample_channels(self):
        bands: int = getattr(self, "_processed_bands", self.dataset_cfg.consts.bands)
        assert bands is not None and bands.is_integer() and bands > 0, f"channel num: {bands}"
        return bands

    def get_global_step(self, mode: str = "train"):
        # TODO: add val state
        assert mode in ("train", "val"), "Only train and val modes are supported for now."

        return self.train_state[mode]

    @property
    def global_step(self):
        return self.get_global_step("train")

    def may_freeze(self, model: nn.Module, freeze=True):
        # for p in model.parameters():
        #     p.requires_grad = not freeze
        model.requires_grad_(not freeze)

    def gradient_check(self, model: nn.Module):
        # check nan gradient
        if self.accelerator.sync_gradients and getattr(self.train_cfg, "grad_check", True):
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if param.grad is None:
                        self.log_msg(
                            f"step {self.global_step} - {name} has None gradient, shaped as {param.shape}",
                            only_rank_zero=False,
                            level="WARNING",
                        )
                    elif torch.isnan(param.grad).any():
                        self.log_msg(
                            f"step {self.global_step} - {name} has nan gradient, shaped as {param.shape}",
                            only_rank_zero=False,
                            level="WARNING",
                        )
                        torch.nan_to_num(param.grad, nan=0.0, posinf=1e5, neginf=-1e5, out=param.grad)

        # clip gradient by norm
        _max_grad_norm = self.train_cfg.max_grad_norm
        if _max_grad_norm is not None and _max_grad_norm > 0:
            if self.dtype != torch.float16 and not self.accelerator.is_fsdp2:
                self.accelerator.clip_grad_norm_(model.parameters(), _max_grad_norm)
            elif (
                self.accelerator.distributed_type == accelerate.utils.DistributedType.FSDP or self.accelerator.is_fsdp2
            ) and isinstance(model, FSDP):
                FSDP.clip_grad_norm_(model.parameters(), max_norm=_max_grad_norm)  # type: ignore

    def compute_segmentation_loss(self, out, gt):
        loss = self.segment_loss(out, gt)
        return loss

    def forward_segment_model(self, img, gt, is_test=False):
        """
        Forward pass through segmentation model.

        Parameters
        ----------
        img : torch.Tensor
            Input image tensor
        gt : torch.Tensor
            Ground truth segmentation tensor

        Returns
        -------
        SegmentationOutput
            Output containing loss and predictions
        """
        if self.no_ema or not is_test:
            model = self.model
        else:
            model = self.ema_model.ema_model
        with self.accelerator.autocast():
            out = model(img)
            # loss
            loss = torch.zeros(1).to(img)
            if not is_test:
                loss = self.compute_segmentation_loss(out, gt)

        output = SegmentationOutput(loss=loss, pred_pixel=out)
        return output

    def _check_image_completion(self, batch: dict) -> bool:
        """Check if we have processed all patches for a complete image."""

        # Method 1: Check batch metadata
        if "is_last_patch" in batch and batch["is_last_patch"]:
            return True

        # Method 2: Check accumulated patches count
        if self._patches_per_image is not None:
            if len(self._accumulated_preds) >= self._patches_per_image:
                return True

        # Method 3: Check if we've reached a reasonable limit (safety check)
        if len(self._accumulated_preds) >= 1000:  # Prevent infinite accumulation
            self.log_msg(
                "Warning: Accumulated too many patches, processing as complete image",
                level="WARNING",
            )
            return True

        return False

    def _process_complete_hypersigma_image(self):
        """Process a complete hyperspectral image and compute loss."""

        if not self._accumulated_preds:
            raise ValueError("No accumulated predictions to process.")

        # Reconstruct complete image from patches
        pred_2d, pred_indexed_1d, gt_indexed_1d = self._reconstruct_hypersigma_image()

        # NOTE: Bootstrap for HyperSIGMA mode is tricky because we reconstruct
        # from patches. If we want to bootstrap, we ideally need the full image
        # probability map to do entropy selection on the whole image.
        # But `pred_2d` here IS the full image prediction from patches.
        # However, `boost_strap_update_label` expects `model` to run separate inference.
        # Running inference AGAIN on full image might be too heavy or duplicate work.
        # `prior` implementation of boost_strap_update_label runs model(img).
        # Here we already have pred_2d.
        # So we might need to modify boost_strap_update_label or implement similar logic here manually
        # using the existing `pred_2d`.

        # For now, let's skip bootstrap for HyperSIGMA mode or warn it's not supported fully yet
        # unless we pass the full image `img` which isn't available here directly
        # (we only have accumulated preds).

        # Actually, let's ONLY support it for standard `train_segment_step` for now as per user request context.
        # If needed for HyperSIGMA, we need access to the full image to run `boost_strap_update_label`.

        loss = self.compute_segmentation_loss(pred_indexed_1d, gt_indexed_1d)
        self._accumulated_preds.clear()

        return SegmentationOutput(loss=loss, pred_pixel=pred_2d)

    def _reconstruct_hypersigma_image(self, accumulated_preds: list[Tensor] | None = None, mode="train"):
        """Reconstruct complete hyperspectral image from accumulated patches."""

        # Stack all predictions
        # (total_patches, C, H, W)
        if accumulated_preds is None:
            accumulated_preds = self._accumulated_preds
        all_preds = torch.cat(accumulated_preds, dim=0)

        ds: SingleImageHyperspectralSegmentationDataset = self.train_dataset if mode == "train" else self.val_dataset  # type: ignore
        index = ds._sampled_index
        gt_indexed_1d = ds.gt_for_loss  # (indices,)
        n_row, n_cols = ds.n_rows, ds.n_cols
        hp, wp = all_preds[0].shape[-2:]
        assert ds.total_patches == all_preds.shape[0], f"{all_preds.shape[0]=} should equal to {ds.total_patches=}"

        # Revert back to a full image
        pred_1d = rearrange(
            all_preds,
            "(nh nw) n_class hp wp -> (nh hp nw wp) n_class",
            nh=n_row,
            nw=n_cols,
        )
        pred_2d = rearrange(
            pred_1d,
            "(h w) n_class -> n n_class h w",
            h=n_row * hp,
            w=n_cols * wp,
            n=1,
        )
        pred_indexed_1d = pred_1d[index]
        gt_indexed_1d = gt_indexed_1d.to(pred_1d.device)

        return pred_2d, pred_indexed_1d, gt_indexed_1d

    def _optimize_step(self, loss: Tensor):
        if self.accelerator.sync_gradients:
            # backward
            self.optim.zero_grad()
            self.accelerator.backward(loss)
            self.gradient_check(self.model)
            self.optim.step()
            self.sched.step()
            # ema update
            self.ema_update()

    def _macro_forward_seg_model(self, batch: dict, macro_batch_size: int = 32):
        """
        Forward pass with macro batch processing.

        Parameters
        ----------
        batch : dict
            Batch containing image tensor
        macro_batch_size : int
            Size of macro batches for memory management

        Returns
        -------
        list[Tensor]
            List of model outputs for each macro batch
        """
        total_img_patches: Tensor = batch["img"]
        img_macro_batches: tuple[Tensor, ...] = total_img_patches.chunk(macro_batch_size, dim=0)
        macro_outputs: list[Tensor] = []
        with self.accelerator.autocast():
            for macro_img in img_macro_batches:
                macro_pred = self.model(macro_img)
                macro_outputs.append(macro_pred)
        return macro_outputs

    def train_segment_step(self, img, gt):
        """
        Single training step for segmentation.

        Parameters
        ----------
        img : torch.Tensor
            Input image tensor
        gt : torch.Tensor
            Ground truth segmentation tensor

        Returns
        -------
        SegmentationOutput
            Output containing loss and predictions
        """
        # Bootstrap: generate pseudo-labels if configuredbootstrap_start_step
        bootstrap_ratio = getattr(self.train_cfg, "bootstrap_ratio", 0.0)
        if bootstrap_ratio > 0 and self.global_step > getattr(self.train_cfg, "bootstrap_start_step", 0):
            # Use bootstrap to update the label
            # Only apply where GT is ignore_index (if you want to keep original labels)
            # OR just update everything based on confidence if that's the intention.
            # Based on the function docstring: "Dynamically generate pseudo-labels for unlabeled data"
            # It returns a new label with ignore_index filled where not selected.
            # Assuming we want to AUGMENT existing labels or use it on unlabeled data.
            # If 'gt' passed here is actually partial or has ignores, this fills them.
            # logger.info('Starting label boost strapping.', once=True)
            gt = boost_strap_update_label(
                self.model,
                img,
                gt,
                ratio=bootstrap_ratio,
                ignore_index=getattr(self.dataset_cfg, "ignore_index", 255),
            )

        out = self.forward_segment_model(img, gt)
        self._optimize_step(out.loss)
        return out

    def train_single_img_accum_step(
        self,
        batch: dict,
        macro_batch_size: int = 32,
    ):
        """HyperSIGMA-style training step: accumulate predictions and compute loss after full image."""

        # Forward pass without computing loss yet
        macro_outputs = self._macro_forward_seg_model(batch, macro_batch_size)
        # Store predictions and metadata for later loss computation
        self._accumulated_preds = macro_outputs

        # Store sample indices if available
        if "sample_index" in batch:
            self._accumulated_indices.append(batch["sample_index"].detach())

        # Check if we have completed a full image
        # is_image_complete = self._check_image_completion(batch)
        # anyway, I temp set to True
        is_image_complete = True

        if is_image_complete:
            # Process the complete image and compute loss
            out = self._process_complete_hypersigma_image()
            self._optimize_step(out.loss)
            return out
        else:
            raise NotImplementedError("Not implemented yet")
            return None

    def train_step(self, batch: dict):
        # NOTE: in HyperSIGMA mode: the batch is a single image patches, to input
        # the segmented model, we use a 'macro-batch' to control the input-model-batch-size

        # Check if we should use HyperSIGMA-style indexing
        use_batch_accum = hasattr(batch, "sample_index") or "sample_index" in batch
        if use_batch_accum:
            self._current_sample_index = batch.get("sample_index", None)

        if self._use_single_img_indexing:
            # HyperSIGMA mode: accumulate predictions
            with self.accelerator.accumulate(self.model):  # use gradient accmulation ?
                train_out = self.train_single_img_accum_step(batch, self.macro_batch_size)
        else:
            # Standard mode: process immediately
            with self.accelerator.accumulate(self.model):
                # train segmentation model
                train_out = self.train_segment_step(batch["img"], batch["gt"])

        # update training state
        self.step_train_state()
        # log losses
        if self.global_step % self.train_cfg.log.log_every == 0:
            _log_losses = self.format_log(train_out.log_losses)
            self.log_msg(
                f"[Train State]: lr {self.optim.param_groups[0]['lr']:1.4e} | "
                f"[Step]: {self.global_step}/{self.train_cfg.max_steps}"
            )
            self.log_msg(f"[Train Tok]: {_log_losses}")

            # tensorboard log
            self.tenb_log_any("metric", train_out.log_losses, self.global_step)

    def format_log(self, log_loss: dict, sync=False) -> str:
        if sync:
            # log_loss = metrics_sync(log_loss, output_tensor_dict=True)
            log_loss = dict_tensor_sync(log_loss)
        strings = dict_round_to_list_str(log_loss, select=list(log_loss.keys()))
        return " - ".join(strings)

    def finite_train_loader(self):
        """Finite train loader that handles complete images properly."""
        for batch in self.train_dataloader:
            yield batch

    def infinity_train_loader(self):
        while True:
            for batch in self.train_dataloader:
                yield batch

    def train_loop(self):
        _stop_train_and_save = False
        self.accelerator.wait_for_everyone()
        self.log_msg("[Train]: start training", only_rank_zero=False)

        for batch in self.infinity_train_loader():
            batch = self._cast_to_dtype(batch)
            # train step
            self.train_step(batch)

            if self.global_step % self.val_cfg.val_duration == 0:
                self.val_loop()

            if self.global_step >= self.train_cfg.max_steps:
                _stop_train_and_save = True

            if self.global_step % self.train_cfg.save_every == 0 or _stop_train_and_save:
                self.save_state()
                self.save_ema()

            if _stop_train_and_save:
                self.log_msg("[Train]: max training step budget reached, stop training and save")
                break

    def _finite_val_loader(self):
        if self.val_dataloader is None:
            raise ValueError("No validation dataloader found")

        if self.ds_is_single_image:
            # Only one image to validate
            d = self.val_dataset._get_full_img_tensor()
            img, gt = d["img"][None], d["gt"][None]
            yield {
                "img": torch.as_tensor(img, device=self.device),
                "gt": torch.as_tensor(gt, device=self.device),
            }
        else:
            for batch in self.val_dataloader:
                yield batch

    def get_val_loader_iter(self):
        if self.val_cfg.max_val_iters > 0:
            # create a new iterator for the validation loader
            # state in the loader generator
            if not hasattr(self, "_val_loader_iter"):
                self._val_loader_iter = iter(self._finite_val_loader())

            iterable_ = trange(
                self.val_cfg.max_val_iters,
                desc="validating ...",
                leave=False,
                disable=not self.accelerator.is_main_process,
            )
            self.log_msg(
                f"[Val]: start validating with only {self.val_cfg.max_val_iters} batches",
                only_rank_zero=False,
            )

            def _loading():
                for _ in iterable_:
                    try:
                        yield next(self._val_loader_iter)
                    except StopIteration:
                        # re-create the iterator
                        self._val_loader_iter = iter(self.val_dataloader)
                        yield next(self._val_loader_iter)

        else:
            iterable_ = self._finite_val_loader()

            def _loading():
                for batch in tqdm(
                    iterable_,
                    desc="validating ...",
                    leave=False,
                    disable=not self.accelerator.is_main_process,
                ):
                    yield batch

            self.log_msg(f"[Val]: start validating with the whole val set", only_rank_zero=False)

        return _loading()

    @torch.no_grad()
    def _cast_to_dtype(self, x: torch.Tensor | list | dict[str, torch.Tensor]):
        """Cast data to the trainer's dtype and device."""
        if isinstance(x, torch.Tensor):
            return x.to(dtype=self.dtype, device=self.device)
        elif isinstance(x, list):
            return [self._cast_to_dtype(xx) for xx in x]
        elif isinstance(x, dict):
            for k, v in x.items():
                if "gt" != k:
                    x[k] = self._cast_to_dtype(v)
                else:
                    x[k] = v.to(device=self.device)
            return x

    @torch.no_grad()
    def val_step(self, batch: dict):
        """
        Validation step for segmentation model.

        Parameters
        ----------
        batch : dict
            Batch containing image and ground truth

        Returns
        -------
        torch.Tensor
            Predicted segmentation map
        """
        if self._use_single_img_indexing:
            use_batch_accum = "sample_index" in batch
            if use_batch_accum:
                self._current_sample_index = batch.get("sample_index", None)
            macro_outputs = self._macro_forward_seg_model(batch, self.macro_batch_size)
            # revert back to a full image
            pred_seg, *_ = self._reconstruct_hypersigma_image(macro_outputs, mode="val")
        else:

            def _val_model_closure(batch):
                # forward the segmentation network
                pred_pixel = self.forward_segment_model(batch["img"], batch["gt"], True).pred_pixel
                return {"pred_logits": pred_pixel}

            # slide windows
            if getattr(self.train_cfg, "val_slide_window", False):
                logger.info(f"[Val]: using slide window validation")
                model_outputs = model_predict_patcher(
                    _val_model_closure,
                    batch,
                    patch_keys=["img", "gt"],
                    merge_keys=["pred_logits"],
                    **getattr(self.train_cfg, "val_slide_window_kwargs", {}),
                )
                pred_logits = model_outputs["pred_logits"]
            else:
                logger.info(f"[Val]: using full image validation")
                pred_logits = _val_model_closure(batch)["pred_logits"]

            img_h, img_w = batch["img"].shape[-2:]
            if pred_logits.shape[-2:] != (img_h, img_w):
                pred_logits = torch.nn.functional.interpolate(
                    pred_logits, size=(img_h, img_w), mode="bilinear", align_corners=False
                )
            pred_seg = pred_logits.argmax(1)

        return pred_seg

    def _resize_to(self, x, mode="nearest"):
        """
        Resize tensor to target size if specified in dataset config.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to resize
        mode : str
            Interpolation mode

        Returns
        -------
        torch.Tensor
            Resized tensor
        """
        inp_ndim = x.ndim
        if getattr(self.dataset_cfg, "resize_to", None):
            tgt_sz = self.dataset_cfg.consts.shape
            if inp_ndim == 3:
                x = x.unsqueeze(1)
            x = torch.nn.functional.interpolate(x, size=tgt_sz, mode=mode)
            if inp_ndim == 3:
                x = x.squeeze(1)
        return x

    def val_loop(self):
        self.model.eval()

        # Initialize two metric instances:
        # 1. Per-class metrics for distributions (per_class=True)
        seg_metrics_per_class: HyperSegmentationScore = hydra.utils.instantiate(
            self.metric_cfg, per_class=True, cal_metrics=["dice", "miou"]
        ).to(self.device)
        # 2. Overall metrics for OA (per_class=False, reduction="micro")
        seg_metrics_overall: HyperSegmentationScore = hydra.utils.instantiate(
            self.metric_cfg, per_class=False, reduction="micro"
        ).to(self.device)
        loss_metrics = MeanMetric().to(device=self.device)

        val_iter = self.get_val_loader_iter()
        for batch in val_iter:
            batch = self._cast_to_dtype(batch)
            batch = cast(dict[str, torch.Tensor], batch)
            gt = batch.get("gt", batch.get("gt_full", None))
            assert gt is not None, "gt or gt_full not found in the val batch"

            # Define validation model closure for slide window
            def _val_model_closure(val_batch):
                """Model closure for slide window validation."""
                with torch.no_grad():
                    output = self.forward_segment_model(val_batch["img"], gt, True)
                    return {"pred_logits": output.pred_pixel}

            # Use slide window or full image validation
            if getattr(self.train_cfg, "val_slide_window", False):
                self.log_msg("[Val]: using slide window validation")
                model_outputs = model_predict_patcher(
                    _val_model_closure,
                    batch,
                    patch_keys=["img", "gt"],
                    merge_keys=["pred_logits"],
                    **getattr(self.train_cfg, "val_slide_window_kwargs", {}),
                )
                pred_logits = model_outputs["pred_logits"]
            else:
                self.log_msg("[Val]: using full image validation")
                pred_logits = _val_model_closure(batch)["pred_logits"]

            # resize to target size if needed
            gt = self._resize_to(gt, mode="nearest")

            # Upsample logits to match GT size using bilinear interpolation for smoother results
            if pred_logits.shape[-2:] != gt.shape[-2:]:
                pred_logits = torch.nn.functional.interpolate(
                    pred_logits, size=gt.shape[-2:], mode="bilinear", align_corners=False
                )

            pred_seg = pred_logits.argmax(1)

            # Update metrics directly
            assert self._val_unsampled_mask is None or self._val_unsampled_mask.shape == gt.shape, (
                "val unsampled mask shape does not match gt shape"
            )

            # Update both metric instances
            seg_metrics_per_class.update(pred_seg, gt)
            seg_metrics_overall.update(pred_seg, gt)
            self.step_train_state("val")

        # Compute metrics from both instances
        metrics_per_class = seg_metrics_per_class.compute()
        metrics_overall = seg_metrics_overall.compute()
        loss_val: float = loss_metrics.compute().item()  # type: ignore

        if self.accelerator.is_main_process:
            # For logging display: show OA (micro-average) metrics
            _metric_str = ""
            for k, v in metrics_overall.items():
                if isinstance(v, torch.Tensor):
                    v_val = v.item()
                    _metric_str += f"{k}: {v_val:.4f} - "
                else:
                    _metric_str += f"{k}: {v:.4f} - "
            self.log_msg(f"[Val] OA (micro): {_metric_str}loss: {loss_val:.3e}")

            # For wandb/tensorboard: log both OA and per-class distributions
            log_dict = {}

            # Log OA (micro-average) metrics
            for k, v in metrics_overall.items():
                if isinstance(v, torch.Tensor):
                    log_dict[f"val/{k}"] = v.item()
                else:
                    log_dict[f"val/{k}"] = v

            # Log per-class distributions
            for k, v in metrics_per_class.items():
                if isinstance(v, torch.Tensor):
                    if v.numel() > 1:
                        # Per-class tensor: log mean as macro-average and full tensor as distribution
                        log_dict[f"val/{k}_macro"] = np.mean(
                            [vi for vi in v.tolist() if vi != -1]
                        )  # -1 means redudent class
                        log_dict[f"val/{k}_per_class"] = v.detach().cpu().numpy().tolist()
                    else:
                        # Scalar tensor (e.g., kappa doesn't have per-class version)
                        log_dict[f"val/{k}"] = v.item()
                else:
                    log_dict[f"val/{k}"] = v

            self.tenb_log_any("metric", log_dict, step=self.global_step)

            # visualize the last val batch
            self.visualize_segmentation(
                batch["img"],  # input image
                pred_seg,  # prediction
                gt,  # gt
                add_step=True,
                only_vis_n=2,
                img_name="val/segmentation",
            )

    def save_state(self):
        self.accelerator.save_state()
        self.log_msg("[State]: save states")

    def save_ema(self):
        if self.no_ema:
            self.log_msg(f"use deepspeed or FSDP, do have EMA model to save")
            return

        ema_path = self.proj_dir / "ema"
        if self.accelerator.is_main_process:
            ema_path.parent.mkdir(parents=True, exist_ok=True)

        self.accelerator.save_model(self.ema_model.ema_model, ema_path / "seg_ema_model")
        # train state
        _ema_path_state_train = ema_path / "train_state.pth"
        _ema_path_state_train.parent.mkdir(parents=True, exist_ok=True)
        accelerate.utils.save(self.train_state.state_dict(), _ema_path_state_train)
        self.log_msg(f"[ckpt]: save ema at {ema_path}")

    def load_from_ema(self, ema_path: str | Path, strict: bool = False):
        ema_path = Path(ema_path)

        try:
            accelerate.load_checkpoint_in_model(self.model, ema_path / "seg_ema_model", strict=strict)
            self.log_msg(f"[Load EMA]: successfully loaded from {ema_path}")
        except Exception as e:
            self.log_msg(f"[Load EMA]: standard loading failed: {e}", level="WARNING")
            model_path = ema_path / "seg_ema_model" / "model.safetensors"
            checkpoint = load_file(model_path, device="cpu")
            result = load_weights_with_shape_check(self.model, checkpoint)
            self.log_msg(f"[Load EMA]: fallback loading completed")
            self.log_msg(f"[Load EMA]: Missing {result.missing_keys}, Unexpected {result.unexpected_keys}")

        # Prepare models
        self.prepare_ema_models()  # This will update EMA models with online models' weights

        # clear the accelerator model registration
        self.log_msg(f"[Load EMA]: clear the accelerator registrations and re-prepare training")

    def resume(self, path: str):
        self.log_msg("[Resume]: resume training")
        self.accelerator.load_state(path)
        self.accelerator.wait_for_everyone()

    def to_rgb(self, x):
        if self.train_cfg.is_neg_1_1:
            return ((x + 1) / 2).clamp(0, 1).float()
        else:
            return x

    def visualize_segmentation(
        self,
        img: torch.Tensor,
        pred_map: torch.Tensor,
        gt_map: torch.Tensor,
        img_name: str = "val/segmentation",
        add_step: bool = False,
        only_vis_n: int | None = None,
        n_class: int = 20,
        use_coco_colors: bool = False,
        n_row: int = 1,
    ):
        """Visualize predicted and ground truth segmentation maps side by side.

        Args:
            pred_map: Predicted segmentation map.
            gt_map: Ground truth segmentation map.
            img_name: Name for the saved image. Defaults to "val/segmentation".
            add_step: Whether to add step number to the image name. Defaults to False.
            only_vis_n: Number of images to visualize. Defaults to None (all).
            n_class: Number of classes. Defaults to 20.
            use_coco_colors: Whether to use COCO dataset colors. Defaults to True.
        """

        _only_n = only_vis_n or 1

        # Process prediction and ground truth maps
        if pred_map.dim() == 4:  # (B, C, H, W)
            pred_map = pred_map.squeeze(1)  # (B, H, W)

        if gt_map.dim() == 4:  # (B, C, H, W)
            gt_map = gt_map.squeeze(1)  # (B, H, W)

        # Convert the 255 background class to 0
        gt_map = label_background_recover(gt_map)
        pred_map += 1

        # Only visualize the first few samples
        pred_map = pred_map[:_only_n]
        gt_map = gt_map[:_only_n]
        img = img[:_only_n]

        assert pred_map.shape[-2:] == gt_map.shape[-2:] == pred_map.shape[-2:]

        # Visualize input image to RGB
        img_rgb = self.to_rgb(img)
        img_rgb = get_rgb_image(img, "mean", use_linstretch=True)
        img_rgb_grid = make_grid(img_rgb, n_row).permute(1, 2, 0).cpu().numpy()

        # Visualize segmentation maps for entire batch
        pred_vis = visualize_segmentation_map(
            pred_map,
            n_class=n_class,
            use_coco_colors=use_coco_colors,
            to_pil=False,
            bg_black=True,
        )
        gt_vis = visualize_segmentation_map(  # [B, H, W, C]
            gt_map,
            n_class=n_class,
            use_coco_colors=use_coco_colors,
            to_pil=False,
            bg_black=True,
        )

        def _maps_to_grid(m):
            m = m[..., :3] if m.shape[-1] > 3 else m
            if m.ndim == 3:
                return m  # [H, W, C]
            elif m.ndim == 4:
                # [B, H, W, C]
                m = m.transpose(0, -1, 1, 2)
                m = torch.as_tensor(m)
                m = make_grid(m, n_row).permute(1, 2, 0)  # [H, W, C]
            else:
                raise ValueError(f"Unkonwn shape: {m.shape}")
            return m.cpu().numpy()

        pred_vis_grid = _maps_to_grid(pred_vis)
        gt_vis_grid = _maps_to_grid(gt_vis)

        # Concatenate together
        img = np.concatenate([img_rgb_grid, pred_vis_grid, gt_vis_grid], axis=1)  # [H, W*3, 3]

        # Convert to PIL Image and save as PNG
        img = (img * 255.0).astype(np.uint8)
        img_to_save = Image.fromarray(img)

        if add_step:
            img_name = f"{img_name}_step_{str(self.global_step).zfill(6)}.png"
        else:
            img_name = f"{img_name}.png"

        save_path = Path(self.proj_dir) / "vis" / img_name
        if self.accelerator.is_main_process:
            save_path.parent.mkdir(parents=True, exist_ok=True)
        if self.accelerator.is_main_process:
            img_to_save.save(save_path)

    def run(self):
        if self.train_cfg.resume_path is not None:
            self.resume(self.train_cfg.resume_path)
        elif self.train_cfg.ema_load_path is not None:
            self.load_from_ema(self.train_cfg.ema_load_path)

        # train !
        self.train_loop()


_key = "unet_seg"
_configs_dict = {
    "deep_globe_road_unet_seg": "deep_globe_road_unet_seg",
    "hybrid_tokenizer_seg": "hybrid_tokenizer_seg",
    "deeplabv3_seg": "deeplabv3_seg",
    "unet_seg": "unet_seg",
}


if __name__ == "__main__":
    from src.utilities.train_utils.cli import argsparse_cli_args, print_colored_banner

    # change the config name in cli
    cli_default_dict = {
        "config_name": _key,
        "only_rank_zero_catch": True,
    }
    chosen_cfg, cli_args = argsparse_cli_args(_configs_dict, cli_default_dict)
    if is_rank_zero := PartialState().is_main_process:
        print_colored_banner("Segmentation")
    log(
        "<green>\n"
        + "=" * 60
        + "\n"
        + f"[Config]: {chosen_cfg}\n"
        + "\n"
        + "Start Running , Good Luck!\n"
        + "=" * 60
        + "</>",
        only_rank_zero=False,
    )

    # Main function
    @hydra.main(
        config_path="../configs/segmentation",
        config_name=chosen_cfg,
        version_base=None,
    )
    def main(cfg):
        catcher = logger.catch if PartialState().is_main_process else nullcontext

        with catcher():
            trainer = HyperSegmentationTrainer(cfg)
            trainer.run()

    main()

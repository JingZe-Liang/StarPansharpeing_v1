import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Literal, Sequence, cast, Any

import accelerate
import hydra
import numpy as np
import PIL.Image as Image
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.state import PartialState
from accelerate.utils.deepspeed import DummyOptim, DummyScheduler
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file
from torch import Tensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._fully_shard import FSDPModule
from torchmetrics.aggregation import MeanMetric
from torchvision.utils import make_grid
from tqdm import tqdm, trange

from src.data.window_slider import WindowSlider, model_predict_patcher
from src.stage2.segmentation.data.data_split import SingleImageHyperspectralSegmentationDataset
from src.stage2.segmentation.data.single_mat_loader import SingleMatDataset
from src.stage2.stereo_matching.utils.vis import visualize_stereo
from src.utilities.config_utils import to_easydict_recursive
from src.utilities.config_utils import to_object as to_cont
from src.utilities.logging import dict_round_to_list_str, log, log_any_into_writter
from src.utilities.network_utils import load_peft_model_checkpoint
from src.utilities.network_utils.network_loading import load_weights_with_shape_check
from src.utilities.train_utils.state import StepsCounter, dict_tensor_sync, metrics_sync
from src.utilities.train_utils.visualization import get_rgb_image

from loguru import logger
from src.stage2.stereo_matching.metrics import UnifiedStereoSegmentationMetrics


@dataclass
class StereoMatchingOutput:
    """立体匹配输出数据类"""

    loss: Tensor
    disparity_final: Tensor
    disparity_aux: list[Tensor] | None = None
    semantic_left: Tensor | None = None
    semantic_right: Tensor | None = None
    log_losses: dict[str, Tensor] | None = None

    def __post_init__(self):
        assert self.loss is not None, "loss should not be None"
        if self.log_losses is None:
            self.log_losses = {"total_loss": self.loss.detach()}

    def __getitem__(self, name):
        return getattr(self.__dict__, name, None)


class StereoMatchingTrainer:
    """
    立体匹配训练器。

    接受从 dataloader 输出的 dict，包含：
    - left: 左图 RGB 图像
    - right: 右图 RGB 图像
    - dsp: 视差真值
    - agl: 高度真值（可选）
    - cls: 分割 mask（可选，US3D 有但 WHU 没有）
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
        self.val_dataset, self.val_dataloader = hydra.utils.instantiate(self.dataset_cfg.val)

        # setup the stereo model
        self.setup_stereo_model()

        # optimizers and lr schedulers
        self.optim, self.sched = self.get_optimizer_lr_scheduler()

        # EMA models and accelerator prepare
        self.prepare_for_training()
        self.prepare_ema_models()

        # loss
        # Loss is now handled by the model's compute_losses method
        self.log_msg(f"Loss computation integrated in model")

        # training state counter
        self.train_state = StepsCounter(["train", "val"])

        # clear GPU memory
        torch.cuda.empty_cache()

    def setup_stereo_model(self):
        self.model = hydra.utils.instantiate(self.cfg.segment_model)
        segment_name = getattr(self.train_cfg, "segment_name", None) or self.model.__class__.__name__
        self.log_msg(f"use stereo model: {segment_name}")

    def prepare_ema_models(self):
        if self.no_ema:
            return

        self.ema_model = hydra.utils.instantiate(self.ema_cfg)(self.model)
        self.log_msg(f"create EMA model for stereo")

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
            if self.accelerator.is_main_process:
                input_list: list[Path | None] = [log_file] * self.accelerator.num_processes
            else:
                input_list = [None] * self.accelerator.num_processes
            output_list: list[Path | None] = [None]
            torch.distributed.scatter_object_list(output_list, input_list, src=0)
            log_file = cast(Path, output_list[0])
            assert isinstance(log_file, Path), "log_file type should be Path"

        # logger
        self.logger.remove()
        log_format_in_file = (
            "<green>[{time:MM-DD HH:mm:ss}]</green> "
            "- <level>[{level}]</level> "
            "- <cyan>{file}:{line}</cyan> - <level>{message}</level>"
        )
        log_format_in_cmd = (
            "{time:HH:mm:ss} - {level.icon} <level>[{level}:{file.name}:{line}]</level>- <level>{message}</level>"
        )
        if not self.train_cfg.debug:
            self.logger.add(
                log_file,
                format=log_format_in_file,
                level="INFO",
                rotation="10 MB",
                enqueue=True,
                backtrace=True,
                colorize=False,
            )
        self.logger.add(sys.stdout, format=log_format_in_cmd, level="DEBUG", backtrace=True, colorize=True)

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
                        project="RS_Stereo_Matching",
                        name=getattr(self.train_cfg.log, "exp_name", None),
                        config=to_cont(self.cfg),
                    )
                    # self.wandb_logger.watch(self.model, log="gradients", log_freq=200)
                    logger.log("NOTE", "will log with wandb")

                if "swanlab" in self._trackers_name:
                    import swanlab

                    exp_cfg = {
                        "train_cfg": to_cont(self.train_cfg),
                        "model_cfg": to_cont(self.cfg.segment_model),
                        "dataset_cfg": to_cont(self.dataset_cfg),
                    }
                    self.swanlab_logger = swanlab.init(
                        project="RS_Stereo_Matching",
                        workspace="iamzihan",
                        experiment_name=getattr(self.train_cfg.log, "exp_name", None),  # type: ignore
                        logdir=str(Path(log_dir) / "swanlab"),
                        resume="never",
                        config=exp_cfg,
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

            heavyball.utils.compile_mode = None
            self.log_msg(
                "use heavyball optimizer, it will compile the optimizer, "
                "for efficience testing the scripts, disable the compilation."
            )

        return model_opt, model_sched

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
        self._base_model = self.accelerator.unwrap_model(self.model)

        self.train_dataloader, self.val_dataloader = self.accelerator.prepare(
            self.train_dataloader, self.val_dataloader
        )

    def step_train_state(self, mode="train"):
        self.train_state.update(mode)

    def ema_update(self, mode="stereo"):
        assert mode == "stereo"
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
                FSDP.clip_grad_norm_(model.parameters(), max_norm=_max_grad_norm)

    def compute_stereo_loss(self, outputs, batch):
        d_gt = batch["dsp"]
        seg_gt_l = batch.get("cls", None)

        # handle case where module is wrapped (e.g. DistributedDataParallel, FSDP)
        losses = self._base_model.compute_losses(outputs, d_gt, seg_gt_l=seg_gt_l)

        return losses

    def forward_stereo_model(self, batch, is_test=False):
        """
        Forward pass through stereo matching model.

        Parameters
        ----------
        batch : dict
            Batch containing left, right images and optional ground truth
        is_test : bool
            Whether it is test/inference mode

        Returns
        -------
        StereoMatchingOutput
            Output containing loss and predictions
        """
        if self.no_ema or not is_test:
            model = self.model
        else:
            model = self.ema_model.ema_model

        left = batch["left"]
        right = batch["right"]

        with self.accelerator.autocast():
            # Model signature: forward(self, left, right)
            out = model(left, right)

            # loss
            losses = self.compute_stereo_loss(out, batch)
            loss = losses["total"]
            log_losses = {k: v.detach() for k, v in losses.items()}

        output = StereoMatchingOutput(
            loss=loss,
            disparity_final=out.get("d_final"),
            disparity_aux=out.get("d_aux"),
            semantic_left=out.get("P_l"),
            semantic_right=out.get("P_r"),
            log_losses=log_losses,
        )
        return output

    def _optimize_step(self, loss: Tensor):
        if self.accelerator.sync_gradients:
            # backward
            self.optim.zero_grad()
            self.accelerator.backward(loss)

            # Optional: Log gradient norms before clipping
            if self.train_cfg.log.log_every > 0 and self.global_step % self.train_cfg.log.log_every == 0:
                self.tenb_log_any(
                    "grad_norm_sum",
                    {"model": self.model},
                    step=self.global_step,
                )

            self.gradient_check(self.model)
            self.optim.step()
            self.sched.step()
            # ema update
            self.ema_update()

    def train_stereo_step(self, batch):
        """
        Single training step for stereo matching.
        """
        out = self.forward_stereo_model(batch)
        self._optimize_step(out.loss)
        return out

    def train_step(self, batch: dict):
        with self.accelerator.accumulate(self.model):
            # train stereo model
            train_out = self.train_stereo_step(batch)

        # update training state
        self.step_train_state()
        # log losses
        if self.global_step % self.train_cfg.log.log_every == 0:
            _log_losses = self.format_log(train_out.log_losses)
            self.log_msg(
                f"[Train State]: lr {self.optim.param_groups[0]['lr']:1.4e} | "
                f"[Step]: {self.global_step}/{self.train_cfg.max_steps}"
            )

            # Simple check the disp value range
            gt_disp = batch["dsp"]
            mask = gt_disp > -100
            gt_disp = gt_disp[mask]
            disp = train_out.disparity_final
            disp = disp[mask.squeeze(1)]
            self.log_msg(f"[Train state]: {_log_losses}")
            logger.info(
                f"Disp pred min/max {disp.min().item():.4f}/{disp.max().item():.4f} "
                f"DSP gt min/max {gt_disp.min().item():.4f}/{gt_disp.max().item():.4f}"
            )

            # tensorboard log
            self.tenb_log_any("metric", train_out.log_losses, self.global_step)

    def format_log(self, log_loss: dict, sync=False) -> str:
        if sync:
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
            # Don't cast Long/Int/Bool to float16/bfloat16
            if x.dtype in (torch.int64, torch.int32, torch.bool):
                return x.to(device=self.device)
            return x.to(dtype=self.dtype, device=self.device)
        elif isinstance(x, list):
            return [self._cast_to_dtype(xx) for xx in x]
        elif isinstance(x, dict):
            for k, v in x.items():
                x[k] = self._cast_to_dtype(v)
            return x

    @torch.no_grad()
    def val_step(self, batch: dict):
        """
        Validation step for stereo matching model.

        Parameters
        ----------
        batch : dict
            Batch containing image and ground truth

        Returns
        -------
        StereoMatchingOutput
            Output containing predictions
        """
        # Forward pass in test mode
        # Note: we might want loss too, so we use is_test=False but in no_grad context
        # to get metrics computed by the model if possible, or just raw output.
        # But forward_stereo_model with is_test=False returns losses.
        # Let's use is_test=True to get standard inference output, and compute EPE manually.
        out = self.forward_stereo_model(batch, is_test=True)
        return out

    def val_loop(self):
        self.model.eval()

        # Initialize metrics
        loss_metrics = {}

        # Initialize unified metrics
        # Get configuration from dataset and metric configs
        min_disp = getattr(self.dataset_cfg, "min_disp", -64)
        max_disp = getattr(self.dataset_cfg, "max_disp", 64)
        num_classes = getattr(self.dataset_cfg, "num_classes", 6)
        compute_seg = getattr(self.dataset_cfg, "compute_seg", False)
        ignore_index = getattr(self.metric_cfg, "ignore_index", None)

        # Create unified metrics
        unified_metrics = UnifiedStereoSegmentationMetrics(
            min_disp=min_disp,
            max_disp=max_disp,
            num_classes=num_classes,
            compute_stereo=True,
            compute_seg=compute_seg,
            ignore_index=ignore_index,
            stereo_thresholds=[1.0, 2.0, 3.0],
            seg_reduction="macro",
        )

        # Move metrics to device
        unified_metrics = unified_metrics.to(self.device)

        val_iter = self.get_val_loader_iter()
        for i, batch in enumerate(val_iter):
            batch = self._cast_to_dtype(batch)

            with torch.no_grad():
                val_out = self.forward_stereo_model(batch, is_test=True)

            if val_out.log_losses is not None:
                for k, v in val_out.log_losses.items():
                    if k not in loss_metrics:
                        loss_metrics[k] = MeanMetric().to(self.device)
                    loss_metrics[k].update(v)

            # Update unified metrics
            # Stereo matching
            d_gt = batch["dsp"]
            d_pred = val_out.disparity_final

            # Semantic segmentation (if available)
            seg_pred = val_out.semantic_left
            seg_gt = batch.get("cls")  # US3D has 'cls', WHU doesn't

            # Move to float for metric calculation
            d_gt_f = d_gt.float()
            d_pred_f = d_pred.float()

            # Update metrics
            unified_metrics.update(
                disp_pred=d_pred_f,
                disp_target=d_gt_f,
                seg_pred=seg_pred,
                seg_target=seg_gt,
            )

            # Debug logging for the first batch to check disparity range
            if i == 0 and self.accelerator.is_main_process:
                # Check d_gt stats
                valid_mask_gt = d_gt_f > -100  # Simple check
                if valid_mask_gt.any():
                    gt_min = d_gt_f[valid_mask_gt].min().item()
                    gt_max = d_gt_f[valid_mask_gt].max().item()
                    gt_mean = d_gt_f[valid_mask_gt].mean().item()
                    logger.info(f"[Val Debug] GT Disp - Min: {gt_min:.2f}, Max: {gt_max:.2f}, Mean: {gt_mean:.2f}")

                # Check d_pred stats
                pred_min = d_pred_f.min().item()
                pred_max = d_pred_f.max().item()
                pred_mean = d_pred_f.mean().item()
                logger.info(f"[Val Debug] Pred Disp - Min: {pred_min:.2f}, Max: {pred_max:.2f}, Mean: {pred_mean:.2f}")

            self.step_train_state("val")

        # Compute metrics
        metrics = {k: m.compute().item() for k, m in loss_metrics.items()}

        # Add unified metrics
        unified_results = unified_metrics.compute()
        flat_metrics = unified_metrics.flatten_metrics(unified_results)

        # Convert to float for logging
        for k, v in flat_metrics.items():
            metrics[k] = v

        if self.accelerator.is_main_process:
            _metric_str = " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            self.log_msg(f"[Val]: {_metric_str}")
            self.tenb_log_any("metric", metrics, step=self.global_step)

            # visualize the last val batch
            self.visualize_stereo(
                batch,
                val_out,  # Fixed: Pass complete StereoMatchingOutput object
                add_step=True,
                img_name="val/stereo",
            )

        # Restore training mode
        self.model.train()

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

        self.accelerator.save_model(self.ema_model.ema_model, ema_path / "model")
        # train state
        _ema_path_state_train = ema_path / "train_state.pth"
        _ema_path_state_train.parent.mkdir(parents=True, exist_ok=True)
        accelerate.utils.save(self.train_state.state_dict(), _ema_path_state_train)
        self.log_msg(f"[ckpt]: save ema at {ema_path}")

    def load_from_ema(self, ema_path: str | Path, strict: bool = True):
        ema_path = Path(ema_path)

        try:
            accelerate.load_checkpoint_in_model(self.model, ema_path / "model", strict=strict)
            self.log_msg(f"[Load EMA]: successfully loaded from {ema_path}")
        except Exception as e:
            self.log_msg(f"[Load EMA]: standard loading failed: {e}", level="WARNING")
            model_path = ema_path / "model" / "model.safetensors"
            checkpoint = load_file(model_path, device="cpu")
            result = load_weights_with_shape_check(self.model, checkpoint)
            self.log_msg(f"[Load EMA]: fallback loading completed")
            self.log_msg(
                f"[Load EMA]: Loaded {result['loaded_keys']}, Missing {result['missing_keys']}, Unexpected {result['unexpected_keys']}"
            )

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

    def visualize_stereo(
        self,
        batch: dict,
        output: StereoMatchingOutput,
        img_name: str = "val/stereo",
        add_step: bool = False,
        n_samples: int = 1,
    ):
        """使用统一的 visualize_stereo 函数进行可视化"""
        # 准备输入数据（取前 n_samples 个样本）
        left = batch["left"][:n_samples].float()
        right = batch["right"][:n_samples].float()
        left = (left + 1) / 2
        right = (right + 1) / 2

        d_gt = batch["dsp"][:n_samples].float()
        d_pred = output.disparity_final[:n_samples].float()

        # 可选数据
        agl = batch.get("agl", None)
        if agl is not None:
            agl = agl[:n_samples].float()

        left_seg_gt = batch.get("cls", None)
        if left_seg_gt is not None:
            left_seg_gt = left_seg_gt[:n_samples]

        # 获取预测的语义分割（如果有）
        pred_seg_left = output.semantic_left
        if pred_seg_left is not None:
            pred_seg_left = pred_seg_left[:n_samples]
            if pred_seg_left.dim() == 4:  # (B, C, H, W)
                pred_seg_left = pred_seg_left.argmax(dim=1)  # (B, H, W)

        pred_seg_right = output.semantic_right
        if pred_seg_right is not None:
            pred_seg_right = pred_seg_right[:n_samples]
            if pred_seg_right.dim() == 4:  # (B, C, H, W)
                pred_seg_right = pred_seg_right.argmax(dim=1)  # (B, H, W)

        # 构造标题
        title = f"Stereo Matching - Step {self.global_step}"

        # 获取视差范围（从数据集配置）
        dsp_vmin = getattr(self.dataset_cfg, "min_disp", None)
        dsp_vmax = getattr(self.dataset_cfg, "max_disp", None)

        # 调用统一的可视化函数（自动处理batch）
        pil_images = visualize_stereo(
            left_rgb=left,
            right_rgb=right,
            dsp_gt=d_gt,
            dsp_pred=d_pred,
            agl=agl,
            left_seg=left_seg_gt,
            pred_seg_left=pred_seg_left,
            pred_seg_right=pred_seg_right,
            title=title,
            invalid_thres=-500,
            dsp_vmin=dsp_vmin,
            dsp_vmax=dsp_vmax,
        )

        for i, pil_img in enumerate(pil_images):
            if add_step:
                save_name = f"{img_name}_step_{str(self.global_step).zfill(6)}"
            else:
                save_name = img_name

            if len(pil_images) > 1:
                save_name += f"_sample_{i}"
            save_name += ".webp"

            save_path = Path(self.proj_dir) / "vis" / save_name
            if self.accelerator.is_main_process:
                if self._trackers_name:
                    tag = "val/vis" if len(pil_images) == 1 else f"val/vis_{i}"
                    self.tenb_log_any(log_type="image", logs={tag: pil_img}, step=self.global_step)
                else:
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    pil_img.save(save_path)
                    self.log_msg(f"Saved visualization to {save_path}")

    def run(self):
        if self.train_cfg.resume_path is not None:
            self.resume(self.train_cfg.resume_path)
        elif self.train_cfg.ema_load_path is not None:
            self.load_from_ema(self.train_cfg.ema_load_path)

        # train !
        self.train_loop()


_key = "hybrid_stereo_matching"
_configs_dict = {
    "hybrid_stereo_matching": "hybrid_stereo_matching",
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
        print_colored_banner("Stereo Matching")
    logger.info(
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
        config_path="../configs/stereo_matching",
        config_name=chosen_cfg,
        version_base=None,
    )
    def main(cfg):
        catcher = logger.catch if PartialState().is_main_process else nullcontext

        with catcher():
            trainer = StereoMatchingTrainer(cfg)
            trainer.run()

    main()

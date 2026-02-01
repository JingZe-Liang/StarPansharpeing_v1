import itertools
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import accelerate
import hydra
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.state import PartialState
from accelerate.tracking import TensorBoardTracker
from accelerate.utils.deepspeed import DummyOptim, DummyScheduler
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import Accuracy, F1Score

from src.utilities.config_utils import to_object as to_cont
from src.utilities.logging import dict_round_to_list_str, log_any_into_writter, set_logger_file
from src.utilities.train_utils.state import StepsCounter, dict_tensor_sync


@dataclass
class ClassificationOutput:
    loss: Tensor
    logits: Tensor
    log_losses: dict[str, Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.log_losses:
            self.log_losses = {"cls_loss": self.loss.detach()}

    def __getitem__(self, name: str):
        return getattr(self.__dict__, name, None)


class HyperClassificationTrainer:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.train_cfg = cfg.train
        self.dataset_cfg = cfg.dataset
        self.ema_cfg = cfg.ema
        self.val_cfg = cfg.val
        self.metric_cfg = cfg.metric

        self.accelerator = cast(Accelerator, hydra.utils.instantiate(cfg.accelerator))
        self._trackers_name = self.train_cfg.log.log_with
        accelerate.utils.set_seed(2025)

        log_file = self.configure_logger()

        self.device = self.accelerator.device
        torch.cuda.set_device(self.accelerator.local_process_index)
        self.dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "no": torch.float32,
        }[self.accelerator.mixed_precision]
        self.log_msg(f"Log file is saved at: {log_file}")
        self.log_msg(f"Weights will be saved at: {self.proj_dir}")
        self.log_msg("Training is configured and ready to start.")

        _dpsp_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        _fsdp_plugin: accelerate.utils.FullyShardedDataParallelPlugin | None = getattr(
            self.accelerator.state, "fsdp_plugin", None
        )

        self.no_ema = False
        self._is_ds = _dpsp_plugin is not None
        if self._is_ds:
            self.log_msg("[Deepspeed]: using deepspeed plugin")
            self.no_ema = _dpsp_plugin.deepspeed_config["zero_optimization"]["stage"] in [2, 3]  # type: ignore

        self._is_fsdp = _fsdp_plugin is not None
        if self._is_fsdp:
            self.log_msg("[FSDP]: using Fully Sharded Data Parallel plugin")
            self.no_ema = True

        dataset_name = self.dataset_cfg.dataset_name
        self.log_msg(f"[Data]: using dataset {dataset_name}")

        self.train_dataset, self.train_dataloader = hydra.utils.instantiate(self.dataset_cfg.train)
        if hasattr(self.dataset_cfg, "val"):
            self.val_dataset, self.val_dataloader = hydra.utils.instantiate(self.dataset_cfg.val)
        else:
            self.val_dataset, self.val_dataloader = self.train_dataset, self.train_dataloader

        self.setup_classification_model()

        self.optim, self.sched = self.get_optimizer_lr_scheduler()

        self.prepare_for_training()
        self.prepare_ema_models()

        self.loss_fn = self.build_loss_fn()
        self.init_metrics()

        self.train_state = StepsCounter(["train", "val"])
        torch.cuda.empty_cache()

    def setup_classification_model(self) -> None:
        self.model = hydra.utils.instantiate(self.cfg.model)
        model_name = getattr(self.train_cfg, "model_name", None) or self.model.__class__.__name__
        self.log_msg(f"use classification model: {model_name}")

        # Initialize lazy modules with a dummy forward pass
        # This is necessary for optimizer creation when using nn.LazyConv2d
        self.model.to(self.device)
        self.model.eval()
        try:
            batch = next(iter(self.train_dataloader))
            image, _ = self._parse_batch(batch)
            with torch.no_grad():
                self.model(image)
            self.log_msg("Model parameters initialized with a dummy forward pass")
        except Exception as e:
            self.log_msg(f"Dummy forward pass failed: {e}", level="WARNING")
        finally:
            self.model.train()

    def prepare_ema_models(self) -> None:
        if self.no_ema:
            return
        self.ema_model = hydra.utils.instantiate(self.ema_cfg)(self.model)
        self.log_msg("create EMA model for classification")

    def configure_logger(self) -> Path:
        self.logger = logger

        log_file = Path(self.train_cfg.proj_dir)
        if self.train_cfg.log.log_with_time:
            str_time = time.strftime("%Y-%m-%d_%H-%M-%S")
            log_file = log_file / str_time
        if self.train_cfg.log.run_comment is not None:
            log_file = Path(log_file.as_posix() + "_" + self.train_cfg.log.run_comment)
        log_file = log_file / "log.log"

        if self.accelerator.use_distributed:
            if self.accelerator.is_main_process:
                input_list: list[Path | None] = [log_file] * self.accelerator.num_processes
            else:
                input_list = [None] * self.accelerator.num_processes
            output_list: list[Path | None] = [None]
            torch.distributed.scatter_object_list(output_list, input_list, src=0)
            log_file = cast(Path, output_list[0])
            assert isinstance(log_file, Path), "log_file type should be Path"

        if not self.train_cfg.debug:
            set_logger_file(log_file, level="info")
            set_logger_file(
                log_file.parent / "debug.log", level="debug", filter=lambda record: record["level"].no <= 10
            )

        log_dir = log_file.parent
        if not self.train_cfg.debug:
            log_dir.mkdir(parents=True, exist_ok=True)

        if not self.train_cfg.debug:
            yaml_cfg = OmegaConf.to_yaml(self.cfg, resolve=True)
            cfg_cp_path = log_file.parent / "config" / "config_total.yaml"
            cfg_cp_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cfg_cp_path, "w") as f:
                f.write(yaml_cfg)
            self.logger.info(f"[Cfg]: configuration saved to {cfg_cp_path}")

        self.proj_dir = log_dir
        self.accelerator.project_configuration.project_dir = str(self.proj_dir)

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
                        project="RS_Classification",
                        name=getattr(self.train_cfg.log, "exp_name", None),
                        config=to_cont(self.cfg),
                    )
                    logger.log("NOTE", "will log with wandb")

                if "swanlab" in self._trackers_name:
                    import swanlab

                    self.swanlab_logger = swanlab.init(
                        project="RS_Classification",
                        workspace="iamzihan",
                        experiment_name=getattr(self.train_cfg.log, "exp_name", None),  # type: ignore
                        logdir=str(Path(log_dir) / "swanlab"),
                        resume="never",
                        config=to_cont(self.cfg),
                    )
                    logger.log("NOTE", "Will log with swanlab")

            from src.utilities.logging import get_python_pkg_env, zip_code_into_dir

            code_dir = ["src/data", "src/stage2", "scripts"]
            zip_code_into_dir(save_dir=log_dir, code_dir=code_dir)
            get_python_pkg_env(file=str(log_dir / "requirements.txt"))

            logger.info("[Code]: code files are zipped and saved.")
            logger.info("[Env]: python package environment requirements are saved.")

        return log_file

    @property
    def loggers(self) -> dict[str, Any]:
        _loggers: dict[str, Any] = {}
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
    ) -> None:
        log_any_into_writter(log_type, self.loggers, logs, step, **kwargs)

    def log_msg(self, *msgs: Any, only_rank_zero: bool = True, level: str = "INFO", sep: str = ",", **kwargs):
        assert level.lower() in ["info", "warning", "error", "debug", "critical"], f"Unknown level {level}"

        def str_msg(*msg: Any) -> str:
            return sep.join([str(m) for m in msg])

        log_fn = getattr(self.logger, level.lower())

        if only_rank_zero:
            if self.accelerator.is_main_process:
                log_fn(str_msg(*msgs), **kwargs)
        else:
            with self.accelerator.main_process_first():
                msg_string = str_msg(*msgs)
                msg_string = f"rank-{self.accelerator.process_index} | {msg_string}"
                log_fn(msg_string, **kwargs)

    def build_loss_fn(self) -> nn.Module:
        label_smoothing = getattr(self.train_cfg, "label_smoothing", 0.0)
        class_weights = getattr(self.train_cfg, "class_weights", None)
        if class_weights is not None:
            weight = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
        else:
            weight = None
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)

    def init_metrics(self) -> None:
        num_classes = self.metric_cfg.num_classes
        top_k = getattr(self.metric_cfg, "top_k", 1)
        acc_metric = Accuracy(task="multiclass", num_classes=num_classes, top_k=top_k, average="micro")
        f1_metric = F1Score(task="multiclass", num_classes=num_classes, top_k=top_k, average="macro")
        acc_metric_top5 = Accuracy(task="multiclass", num_classes=num_classes, top_k=5, average="micro")
        self.acc_metric: Accuracy = acc_metric.to(self.device)
        self.f1_metric: F1Score = f1_metric.to(self.device)
        self.acc_metric_top5: Accuracy = acc_metric_top5.to(self.device)

    def get_optimizer_lr_scheduler(
        self,
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        if (
            self.accelerator.state.deepspeed_plugin is None
            or "optimizer" not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):

            def _optimizer_creater(optimizer_cfg, params_getter):
                if "muon" in optimizer_cfg._target_:
                    self.log_msg("[Optimizer]: using muon optimizer")
                    named_params = params_getter(with_name=True)
                    return hydra.utils.instantiate(optimizer_cfg)(named_parameters=named_params)
                self.log_msg(f"[Optimizer]: using optimizer: {optimizer_cfg._target_}")
                params = params_getter(with_name=False)
                return hydra.utils.instantiate(optimizer_cfg)(params)

            def _get_model_params(with_name: bool):
                if with_name:
                    return [(name, param) for name, param in self.model.named_parameters() if param.requires_grad]
                return [param for param in self.model.parameters() if param.requires_grad]

            model_opt = _optimizer_creater(self.train_cfg.model_optim, _get_model_params)
        else:
            model_opt = DummyOptim([{"params": [param for param in self.model.parameters() if param.requires_grad]}])

        if (
            self.accelerator.state.deepspeed_plugin is None
            or "scheduler" not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            model_sched = hydra.utils.instantiate(self.train_cfg.model_sched)(optimizer=model_opt)
        else:
            model_sched = DummyScheduler(model_opt)

        is_heavyball_opt = lambda opt: opt.__class__.__module__.startswith("heavyball")
        if is_heavyball_opt(model_opt):
            import heavyball

            heavyball.utils.compile_mode = None  # type: ignore
            self.log_msg(
                "use heavyball optimizer, it will compile the optimizer, "
                "for efficience testing the scripts, disable the compilation."
            )

        return model_opt, model_sched  # type: ignore[return-value]

    def prepare_for_training(self) -> None:
        if self._is_fsdp or self.accelerator.distributed_type in (
            accelerate.utils.DistributedType.MULTI_GPU,
            accelerate.utils.DistributedType.FSDP,
        ):
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.log_msg("[Model] convert model bn to sync batch norm")

        if self._is_fsdp and self.accelerator.is_fsdp2:
            self.model.dtype = torch.float

        self.model, self.optim = self.accelerator.prepare(self.model, self.optim)
        self.train_dataloader, self.val_dataloader = self.accelerator.prepare(
            self.train_dataloader, self.val_dataloader
        )

    def step_train_state(self, mode: str = "train") -> None:
        self.train_state.update(mode)

    def ema_update(self) -> None:
        if self.no_ema:
            return
        self.ema_model.update()

    def get_global_step(self, mode: str = "train") -> int:
        assert mode in ("train", "val"), "Only train and val modes are supported for now."
        return self.train_state[mode]

    @property
    def global_step(self) -> int:
        return self.get_global_step("train")

    def gradient_check(self, model: nn.Module) -> None:
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

        _max_grad_norm = self.train_cfg.max_grad_norm
        if _max_grad_norm is not None and _max_grad_norm > 0:
            if self.dtype != torch.float16 and not self.accelerator.is_fsdp2:
                self.accelerator.clip_grad_norm_(model.parameters(), _max_grad_norm)
            elif (
                self.accelerator.distributed_type == accelerate.utils.DistributedType.FSDP or self.accelerator.is_fsdp2
            ) and isinstance(model, FSDP):
                FSDP.clip_grad_norm_(model.parameters(), max_norm=_max_grad_norm)  # type: ignore

    def _parse_batch(self, batch: Any) -> tuple[Tensor, Tensor]:
        if isinstance(batch, dict):
            image = batch.get("image", None)
            if image is None:
                image = batch.get("img", None)
            label = batch.get("label", None)
            if label is None:
                label = batch.get("gt", None)
            if label is None:
                label = batch.get("target", None)
        elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
            image, label = batch[0], batch[1]
        else:
            raise ValueError("Unsupported batch type for classification")

        if image is None or label is None:
            raise ValueError("Batch missing image/label fields")

        image = image.to(device=self.device, dtype=self.dtype)
        label = label.to(device=self.device)
        if label.dtype != torch.long:
            label = label.long()
        return image, label

    def _forward_model(self, image: Tensor, is_eval: bool = False) -> Tensor:
        if is_eval and not self.no_ema and getattr(self.val_cfg, "use_ema", False):
            model = self.ema_model.ema_model
        else:
            model = self.model

        with self.accelerator.autocast():
            output = model(image)
            if isinstance(output, dict):
                logits = output.get("logits", None)
                if logits is None:
                    raise ValueError("Model output dict must contain 'logits'")
            else:
                logits = output
        return logits

    def _optimize_step(self, loss: Tensor) -> None:
        if self.accelerator.sync_gradients:
            self.optim.zero_grad()
            self.accelerator.backward(loss)
            self.gradient_check(self.model)
            self.optim.step()
            self.sched.step()
            self.ema_update()

    def train_step(self, batch: Any) -> ClassificationOutput:
        with self.accelerator.accumulate(self.model):
            image, label = self._parse_batch(batch)
            logits = self._forward_model(image, is_eval=False)
            loss = self.loss_fn(logits, label)
            self._optimize_step(loss)

        pred = torch.argmax(logits, dim=1)
        acc = (pred == label).float().mean()
        log_losses = {"cls_loss": loss.detach(), "acc": acc.detach()}
        return ClassificationOutput(loss=loss, logits=logits, log_losses=log_losses)

    def format_log(self, log_loss: dict[str, Tensor], sync: bool = False) -> str:
        if sync:
            log_loss = dict_tensor_sync(log_loss)
        strings = dict_round_to_list_str(log_loss, select=list(log_loss.keys()))
        return " - ".join(strings)

    def finite_train_loader(self):
        for batch in self.train_dataloader:
            yield batch

    def infinity_train_loader(self):
        while True:
            for batch in self.train_dataloader:
                yield batch

    def get_val_loader_iter(self):
        if self.val_cfg.max_val_iters > 0:
            self.log_msg(
                f"[Val]: start validating with only {self.val_cfg.max_val_iters} batches",
                only_rank_zero=False,
            )

            return itertools.islice(self.val_dataloader, self.val_cfg.max_val_iters)
        if self.val_cfg.max_val_iters <= 0:
            self.log_msg("[Val]: start validating with the whole val set", only_rank_zero=False)
            return self.finite_val_loader()
        raise ValueError("max_val_iters must be greater than 0 or less than 0")

    def finite_val_loader(self):
        for batch in self.val_dataloader:
            yield batch

    @torch.no_grad()
    def val_step(self, batch: Any) -> tuple[Tensor, Tensor, Tensor]:
        image, label = self._parse_batch(batch)
        logits = self._forward_model(image, is_eval=True)
        loss = self.loss_fn(logits, label)
        return loss, logits, label

    def val_loop(self) -> None:
        self.model.eval()
        loss_metric = MeanMetric().to(device=self.device)
        self.acc_metric.reset()
        self.f1_metric.reset()
        self.acc_metric_top5.reset()

        for batch in self.get_val_loader_iter():
            loss, logits, label = self.val_step(batch)
            gathered_logits = self.accelerator.gather_for_metrics(logits)
            gathered_label = self.accelerator.gather_for_metrics(label)
            self.acc_metric.update(gathered_logits, gathered_label)
            self.f1_metric.update(gathered_logits, gathered_label)
            self.acc_metric_top5.update(gathered_logits, gathered_label)
            loss_metric.update(loss.detach())  # type: ignore[arg-type]
            self.step_train_state("val")

        metrics = {
            "val_acc": self.acc_metric.compute(),  # type: ignore[call-arg]
            "val_f1": self.f1_metric.compute(),  # type: ignore[call-arg]
            "val_acc_top5": self.acc_metric_top5.compute(),  # type: ignore[call-arg]
            "val_loss": loss_metric.compute(),  # type: ignore[call-arg]
        }

        if self.accelerator.is_main_process:
            metric_str = " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            self.log_msg(f"[Val]: {metric_str}")
            self.tenb_log_any("metric", metrics, step=self.global_step)

    def train_loop(self) -> None:
        _stop_train_and_save = False
        self.accelerator.wait_for_everyone()
        self.log_msg("[Train]: start training", only_rank_zero=False)

        for batch in self.infinity_train_loader():
            train_out = self.train_step(batch)
            self.step_train_state()

            if self.global_step % self.train_cfg.log.log_every == 0:
                _log_losses = self.format_log(train_out.log_losses)
                self.log_msg(
                    f"[Train State]: lr {self.optim.param_groups[0]['lr']:1.4e} | "
                    f"[Step]: {self.global_step}/{self.train_cfg.max_steps}"
                )
                self.log_msg(f"[Train CLS]: {_log_losses}")
                self.tenb_log_any("metric", train_out.log_losses, self.global_step)

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

    def save_state(self) -> None:
        self.accelerator.save_state()
        self.log_msg("[State]: save states")

    def save_ema(self) -> None:
        if self.no_ema:
            self.log_msg("use deepspeed or FSDP, do have EMA model to save")
            return

        ema_path = self.proj_dir / "ema"
        if self.accelerator.is_main_process:
            ema_path.parent.mkdir(parents=True, exist_ok=True)

        self.accelerator.save_model(self.ema_model.ema_model, ema_path / "cls_ema_model")
        _ema_path_state_train = ema_path / "cls_train_state.pth"
        _ema_path_state_train.parent.mkdir(parents=True, exist_ok=True)
        accelerate.utils.save(self.train_state.state_dict(), _ema_path_state_train)
        self.log_msg(f"[ckpt]: save ema at {ema_path}")

    def load_from_ema(self, ema_path: str | Path, strict: bool = True) -> None:
        ema_path = Path(ema_path)
        accelerate.load_checkpoint_in_model(self.model, ema_path / "model", strict=strict)
        self.prepare_ema_models()
        self.log_msg("[Load EMA]: clear the accelerator registrations and re-prepare training")

    def resume(self, path: str) -> None:
        self.log_msg("[Resume]: resume training")
        self.accelerator.load_state(path)
        self.accelerator.wait_for_everyone()

    def run(self) -> None:
        if self.train_cfg.resume_path is not None:
            self.resume(self.train_cfg.resume_path)
        elif self.train_cfg.ema_load_path is not None:
            self.load_from_ema(self.train_cfg.ema_load_path)
        self.train_loop()


_key = "spectralgpt_ucmerced"
_configs_dict = {
    "cosmos_ucmerced_linear": "cosmos_ucmerced_linear",
    "cosmos_eurosat_linear": "cosmos_eurosat_linear",
    "dinov3_ucmerced_linear": "dinov3_ucmerced_linear",
    "cosmos_ucmerced_hybrid_linear": "cosmos_ucmerced_hybrid_linear",
    # Other SSL models
    "hypersigma_ucmerced": "hypersigma_ucmerced",
    "mae_ucmerced": "mae_ucmerced",
    "spectralgpt_ucmerced": "spectralgpt_ucmerced",
}


if __name__ == "__main__":
    from src.utilities.train_utils.cli import argsparse_cli_args, print_colored_banner
    from src.utilities.logging import log

    cli_default_dict = {
        "config_name": _key,
        "only_rank_zero_catch": True,
    }
    chosen_cfg, cli_args = argsparse_cli_args(_configs_dict, cli_default_dict)
    if is_rank_zero := PartialState().is_main_process:
        print_colored_banner("Classification")
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

    @hydra.main(
        config_path="../configs/classification",
        config_name=chosen_cfg,
        version_base=None,
    )
    def main(cfg):
        trainer = HyperClassificationTrainer(cfg)
        trainer.run()

    main()

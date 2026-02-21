from __future__ import annotations

from contextlib import nullcontext
from numbers import Number
from typing import Any

import hydra
import torch
from accelerate.state import PartialState
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from scripts.trainer.hyper_latent_segmentation_trainer import HyperSegmentationTrainer
from src.utilities.logging import log


class SingleHyperImageSegmentationTrainer(HyperSegmentationTrainer):
    """Trainer for single-image hyperspectral classification-as-segmentation."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._validate_single_image_constraints()
        self._validate_patch_size_compatibility()

    def _validate_single_image_constraints(self) -> None:
        is_single_image = bool(getattr(self.dataset_cfg.consts, "is_single_image", False))
        if not is_single_image:
            raise ValueError(
                "SingleHyperImageSegmentationTrainer requires dataset.consts.is_single_image=True, "
                f"but got {is_single_image}."
            )

        train_full_img = bool(getattr(self.train_dataset, "full_img", True))
        if train_full_img:
            raise ValueError(
                "SingleHyperImageSegmentationTrainer requires patch-level training with full_img=False on train_dataset."
            )

        if not hasattr(self.train_dataset, "patch_size"):
            raise ValueError("train_dataset.patch_size is required for patch-level training.")

        if not hasattr(self.val_dataset, "_get_full_img_tensor"):
            raise ValueError(
                "val_dataset must provide `_get_full_img_tensor` for full-image validation in single-image mode."
            )

    def _normalize_patch_size_value(self, value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, Number):
            value_int = int(value)
            return value_int if value_int > 0 else None
        if isinstance(value, (list, tuple)):
            parsed: list[int] = []
            for item in value:
                parsed_item = self._normalize_patch_size_value(item)
                if parsed_item is not None:
                    parsed.append(parsed_item)
            if parsed:
                return min(parsed)
        return None

    def _collect_patch_sizes_from_cfg(self, cfg_node: Any) -> list[int]:
        patch_sizes: list[int] = []
        if isinstance(cfg_node, dict):
            for key, value in cfg_node.items():
                key_lower = str(key).lower()
                if "patch" in key_lower and "size" in key_lower:
                    parsed = self._normalize_patch_size_value(value)
                    if parsed is not None:
                        patch_sizes.append(parsed)
                patch_sizes.extend(self._collect_patch_sizes_from_cfg(value))
        elif isinstance(cfg_node, list):
            for value in cfg_node:
                patch_sizes.extend(self._collect_patch_sizes_from_cfg(value))
        return patch_sizes

    def _collect_patch_sizes_from_model(self) -> list[int]:
        patch_sizes: list[int] = []
        candidate_objs = [self.model, getattr(self.model, "backbone", None), getattr(self.model, "encoder", None)]
        candidate_keys = ("patch_size", "token_patch_size", "vit_patch_size", "img_patch_size")
        for obj in candidate_objs:
            if obj is None:
                continue
            for key in candidate_keys:
                parsed = self._normalize_patch_size_value(getattr(obj, key, None))
                if parsed is not None:
                    patch_sizes.append(parsed)
        return patch_sizes

    def _resolve_train_patch_size(self) -> int:
        dataset_patch_size = self._normalize_patch_size_value(getattr(self.train_dataset, "patch_size", None))
        if dataset_patch_size is not None:
            return dataset_patch_size

        cfg_patch_size = self._normalize_patch_size_value(getattr(self.dataset_cfg.consts, "patch_size", None))
        if cfg_patch_size is not None:
            return cfg_patch_size

        raise ValueError(
            "Unable to resolve training patch size from train_dataset.patch_size or dataset.consts.patch_size."
        )

    def _resolve_model_patch_size_candidates(self) -> list[int]:
        patch_sizes: list[int] = []
        seg_model_cfg = OmegaConf.to_container(self.cfg.segment_model, resolve=False)
        patch_sizes.extend(self._collect_patch_sizes_from_cfg(seg_model_cfg))
        patch_sizes.extend(self._collect_patch_sizes_from_model())
        return sorted(set(patch_sizes))

    def _validate_patch_size_compatibility(self) -> None:
        train_patch_size = self._resolve_train_patch_size()
        model_patch_sizes = self._resolve_model_patch_size_candidates()
        if not model_patch_sizes:
            self.log_msg(
                "Cannot infer model patch size from segment_model config/model attrs; skip strict patch-size check.",
                level="WARNING",
            )
            return

        min_model_patch_size = min(model_patch_sizes)
        if train_patch_size < min_model_patch_size:
            raise ValueError(
                f"train patch_size ({train_patch_size}) is smaller than model patch_size ({min_model_patch_size}). "
                "This setup is unsafe for patch-token models."
            )

    def _finite_val_loader(self):
        if self.val_dataloader is None:
            raise ValueError("No validation dataloader found")
        if not self.ds_is_single_image:
            raise ValueError("SingleHyperImageSegmentationTrainer only supports single-image validation mode.")

        full_sample = self.val_dataset._get_full_img_tensor()
        img = torch.as_tensor(full_sample["img"], device=self.device)
        gt = torch.as_tensor(full_sample["gt"], device=self.device)
        if img.ndim == 3:
            img = img.unsqueeze(0)
        if gt.ndim == 2:
            gt = gt.unsqueeze(0)
        yield {"img": img, "gt": gt}

    def get_val_loader_iter(self):
        self.log_msg("[Val]: start single-image full-map validation", only_rank_zero=False)

        def _loading():
            for batch in self._finite_val_loader():
                yield batch

        return _loading()

    def train_step(self, batch: dict):
        with self.accelerator.accumulate(self.model):
            train_out = self.train_segment_step(batch["img"], batch["gt"])

        self.step_train_state()
        if self.global_step % self.train_cfg.log.log_every == 0:
            _log_losses = self.format_log(train_out.log_losses)
            self.log_msg(
                f"[Train State]: lr {self.optim.param_groups[0]['lr']:1.4e} | "
                f"[Step]: {self.global_step}/{self.train_cfg.max_steps}"
            )
            self.log_msg(f"[Train Tok]: {_log_losses}")
            self.tenb_log_any("metric", train_out.log_losses, self.global_step)


# add cfg here
_key = "single_hyper_image_indian_pines_hybrid_decoder_seg"
_configs_dict = {
    "single_hyper_image_indian_pines_hybrid_decoder_seg": "single_hyper_image_indian_pines_hybrid_decoder_seg",
    "single_hyper_image_indian_pines_unet_seg": "single_hyper_image_indian_pines_unet_seg",
}


if __name__ == "__main__":
    from src.utilities.train_utils.cli import argsparse_cli_args, print_colored_banner

    cli_default_dict = {
        "config_name": _key,
        "only_rank_zero_catch": True,
    }
    chosen_cfg, cli_args = argsparse_cli_args(_configs_dict, cli_default_dict)
    if is_rank_zero := PartialState().is_main_process:
        print_colored_banner("SingleHyperImageSegmentation")

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
        config_path="../configs/segmentation",
        config_name=chosen_cfg,
        version_base=None,
    )
    def main(cfg):
        catcher = logger.catch if PartialState().is_main_process else nullcontext

        with catcher():
            trainer = SingleHyperImageSegmentationTrainer(cfg)
            trainer.run()

    main()

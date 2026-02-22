from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import hydra
import numpy as np
import PIL.Image as Image
import torch
from accelerate.state import PartialState
from loguru import logger
from omegaconf import DictConfig
from torch import Tensor

from scripts.trainer.hyper_latent_change_detection_trainer import CDModelStepOutput, HyperCDTrainer
from src.data.window_slider import model_predict_patcher
from src.utilities.logging import log
from src.utilities.train_utils.visualization import get_rgb_image


class SingleHyperImageCDTrainer(HyperCDTrainer):
    """Single-image hyperspectral CD trainer with HyperSIGMA-style patch sampling."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._use_single_img_indexing = True
        self.macro_batch_size = int(getattr(self.train_cfg, "macro_batch_size", 32))
        self.log_msg("[Single CD]: force-enable single-image training mode")

    def _extract_center_logits_and_labels(self, pred_logits: Tensor, gt: Tensor) -> tuple[Tensor, Tensor]:
        if gt.ndim == 4 and gt.shape[1] == 1:
            gt = gt.squeeze(1)
        if gt.ndim != 3:
            raise ValueError(f"Expected gt with shape [B,H,W] or [B,1,H,W], got {tuple(gt.shape)}")

        center_h = pred_logits.shape[-2] // 2
        center_w = pred_logits.shape[-1] // 2
        gt_center_h = gt.shape[-2] // 2
        gt_center_w = gt.shape[-1] // 2

        pred_center = pred_logits[:, :, center_h, center_w]  # [B, C]
        gt_center = gt[:, gt_center_h, gt_center_w]  # [B]
        return pred_center, gt_center

    def train_single_img_accum_step(self, batch: dict, macro_batch_size: int = 32) -> CDModelStepOutput:
        macro_outputs = self._macro_forward_seg_model(batch, macro_batch_size)
        pred_logits = torch.cat(macro_outputs, dim=0)

        pred_center, gt_center = self._extract_center_logits_and_labels(pred_logits, batch["gt"])
        pred_for_loss = pred_center[:, :, None, None].contiguous()
        # Keep gt as [B, 1, 1, 1] so base trainer squeeze_(1) -> [B, 1, 1]
        gt_for_loss = gt_center[:, None, None, None].to(pred_for_loss.device)
        loss, log_losses = self.compute_segmentation_loss(pred_for_loss, gt_for_loss)
        self._optimize_step(loss)
        return CDModelStepOutput(pred_logits, loss, log_losses)

    def train_step(self, batch: dict):
        batch = self._cast_to_dtype(batch)
        with self.accelerator.accumulate(self.model):
            train_out = self.train_single_img_accum_step(batch, self.macro_batch_size)
        self.step_train_state()

        if self.global_step % self.train_cfg.log.log_every == 0:
            _log_losses = self.format_log(train_out.log_losses)
            self.log_msg(
                f"[Train State]: lr {self.optim.param_groups[0]['lr']:1.4e} | "
                f"[Step]: {self.global_step}/{self.train_cfg.max_steps}"
            )
            self.log_msg(f"[Train Tok]: {_log_losses}")
            self.tenb_log_any("metric", train_out.log_losses, self.global_step)

    @torch.no_grad()
    def val_step(self, batch: dict):
        def _val_model_closure(batch_inputs: dict[str, Tensor]):
            model = self.ema_model.ema_model if not self.no_ema else self.model
            if model is None:
                raise ValueError("Validation model is None.")
            with self.accelerator.autocast():
                pred_logits = model([batch_inputs["img1"], batch_inputs["img2"]])
            return {"pred_logits": pred_logits}

        if self.train_cfg.val_slide_window:
            model_outputs = model_predict_patcher(
                _val_model_closure,
                batch,
                patch_keys=["img1", "img2", "gt"],
                merge_keys=["pred_logits"],
                **self.train_cfg.val_slide_window_kwargs,
            )
            pred_seg = model_outputs["pred_logits"].argmax(1)
        else:
            pred_seg = _val_model_closure(batch)["pred_logits"].argmax(1)

        return pred_seg

    def _to_hwc_batch(self, img_tensor: Tensor) -> np.ndarray:
        img_tensor = img_tensor.detach().cpu()
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)
        if img_tensor.ndim != 4 or img_tensor.shape[1] != 3:
            raise ValueError(f"Expected image tensor shaped [B,3,H,W], got {tuple(img_tensor.shape)}")
        return img_tensor.permute(0, 2, 3, 1).numpy()

    def _labels_to_bw_gray_rgb(self, label_map: Tensor) -> np.ndarray:
        label_np = label_map.detach().cpu().numpy()
        vis = np.full((*label_np.shape, 3), 0.5, dtype=np.float32)  # unknown -> gray
        vis[label_np == 0] = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # unchanged -> black
        vis[label_np == 1] = np.array([1.0, 1.0, 1.0], dtype=np.float32)  # changed -> white
        return vis

    def _tp_tn_fn_fp_rgb(self, pred_map: Tensor, gt_map: Tensor) -> np.ndarray:
        pred_np = pred_map.detach().cpu().numpy()
        gt_np = gt_map.detach().cpu().numpy()

        vis = np.full((*gt_np.shape, 3), 0.5, dtype=np.float32)  # unknown -> gray
        valid_mask = gt_np != 255

        tn = valid_mask & (gt_np == 0) & (pred_np == 0)
        tp = valid_mask & (gt_np == 1) & (pred_np == 1)
        fn = valid_mask & (gt_np == 1) & (pred_np == 0)
        fp = valid_mask & (gt_np == 0) & (pred_np == 1)

        vis[tn] = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # TN black
        vis[tp] = np.array([1.0, 1.0, 1.0], dtype=np.float32)  # TP white
        vis[fn] = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # FN green
        vis[fp] = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # FP red
        return vis

    def visualize_segmentation(
        self,
        img1: Tensor,
        img2: Tensor,
        pred_map: Tensor,
        gt_map: Tensor,
        img_name: str = "val/segmentation",
        add_step: bool = False,
        only_vis_n: int | None = None,
        n_class: int = 2,
        use_coco_colors: bool = False,
    ):
        del n_class, use_coco_colors
        only_n = only_vis_n or 1

        if pred_map.ndim == 4 and pred_map.shape[1] == 1:
            pred_map = pred_map.squeeze(1)
        if gt_map.ndim == 4 and gt_map.shape[1] == 1:
            gt_map = gt_map.squeeze(1)

        if pred_map.ndim != 3 or gt_map.ndim != 3:
            raise ValueError(
                f"Expected pred/gt with [B,H,W], got pred={tuple(pred_map.shape)}, gt={tuple(gt_map.shape)}"
            )
        if pred_map.shape[-2:] != gt_map.shape[-2:]:
            raise ValueError(f"Pred/gt shape mismatch: pred={tuple(pred_map.shape)}, gt={tuple(gt_map.shape)}")
        if img1.shape != img2.shape:
            raise ValueError(f"Input image mismatch: img1={tuple(img1.shape)}, img2={tuple(img2.shape)}")

        pred_map = pred_map[:only_n].long()
        gt_map = gt_map[:only_n].long()
        img1 = img1[:only_n]
        img2 = img2[:only_n]

        img1_rgb = get_rgb_image(self.to_rgb(img1), rgb_channels=[2, 1, 0], use_linstretch=True)
        img2_rgb = get_rgb_image(self.to_rgb(img2), rgb_channels=[2, 1, 0], use_linstretch=True)
        img1_rgb_np = self._to_hwc_batch(img1_rgb)
        img2_rgb_np = self._to_hwc_batch(img2_rgb)

        pred_masked = pred_map.clone()
        pred_masked[gt_map == 255] = 255

        pred_vis = self._labels_to_bw_gray_rgb(pred_masked)
        gt_vis = self._labels_to_bw_gray_rgb(gt_map)
        err_vis = self._tp_tn_fn_fp_rgb(pred_map, gt_map)

        vis_images: list[np.ndarray] = []
        for i in range(pred_vis.shape[0]):
            img1_sample = img1_rgb_np[i]
            img2_sample = img2_rgb_np[i]
            pred_sample = pred_vis[i]
            gt_sample = gt_vis[i]
            err_sample = err_vis[i]

            h, w = pred_sample.shape[:2]
            if img1_sample.shape[:2] != (h, w):
                img1_sample = (
                    np.array(
                        Image.fromarray((img1_sample * 255.0).astype(np.uint8)).resize(
                            (w, h), Image.Resampling.BILINEAR
                        )
                    ).astype(np.float32)
                    / 255.0
                )
            if img2_sample.shape[:2] != (h, w):
                img2_sample = (
                    np.array(
                        Image.fromarray((img2_sample * 255.0).astype(np.uint8)).resize(
                            (w, h), Image.Resampling.BILINEAR
                        )
                    ).astype(np.float32)
                    / 255.0
                )

            top_row = np.concatenate([img1_sample, img2_sample, pred_sample, gt_sample], axis=1)
            bottom_row = np.zeros((h, top_row.shape[1], 3), dtype=np.float32)
            bottom_row[:, -w:, :] = err_sample

            combined_vis = np.concatenate([top_row, bottom_row], axis=0)
            vis_images.append(combined_vis)

        img = np.concatenate(vis_images, axis=0) if len(vis_images) > 1 else vis_images[0]
        img_uint8 = (img * 255.0).clip(0, 255).astype(np.uint8)
        img_to_save = Image.fromarray(img_uint8)

        out_name = f"{img_name}.png"
        if add_step:
            out_name = f"{img_name}_step_{str(self.global_step).zfill(6)}.png"

        save_path = Path(self.proj_dir) / "vis" / out_name
        if self.accelerator.is_main_process:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            img_to_save.save(save_path)
        self.log_msg(f"[Visualize]: save segmentation visualization at {save_path}")


_key = "tokenizer_hybrid_adaptor_single_mat"
_configs_dict = {
    "tokenizer_hybrid_adaptor_single_mat": "tokenizer_hybrid_adaptor_single_mat",
}


if __name__ == "__main__":
    from src.utilities.train_utils.cli import argsparse_cli_args, print_colored_banner

    cli_default_dict = {
        "config_name": _key,
        "only_rank_zero_catch": True,
    }
    chosen_cfg, cli_args = argsparse_cli_args(_configs_dict, cli_default_dict)
    if PartialState().is_main_process:
        print_colored_banner("Change Detection Single Image")

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
        config_path="../configs/change_detection",
        config_name=chosen_cfg,
        version_base=None,
    )
    def main(cfg):
        catcher = logger.catch if PartialState().is_main_process else nullcontext
        with catcher():
            trainer = SingleHyperImageCDTrainer(cfg)
            trainer.run()

    main()

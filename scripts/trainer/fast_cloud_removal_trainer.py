from collections.abc import Iterable
from typing import Any

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from src.stage2.cloud_removal.diffusion.vae import CosmosRSVAE, FluxVAE
from src.stage2.cloud_removal.metrics.basic import CRMetrics


@logger.catch()
def train(
    model: torch.nn.Module,
    vae: CosmosRSVAE | FluxVAE,
    train_dataloader: Iterable[dict[str, torch.Tensor]],
    val_dataloader: Iterable[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    metric_fn: CRMetrics,
    num_iters: int = 100_000,
    lr: float = 1e-4,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    val_every: int = 500,
    is_neg_1_1: bool = True,
) -> None:
    model.train()
    vae.eval().requires_grad_(False)
    scaler = torch.amp.GradScaler()

    step = 0
    for _epoch_idx in range(num_iters):
        for batch in train_dataloader:
            optimizer.zero_grad()

            img = batch["img"].to(device)
            gt = batch.get("gt", img).to(device)

            if isinstance(vae, FluxVAE):
                # flux vae only uses RGB channels
                img = img[:, :3]
                gt = gt[:, :3]

            with torch.no_grad():
                cond_latent = vae.encode(img)
                gt_latent = vae.encode(gt)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                pred = model(cond_latent, None, conditions=cond_latent)
                if isinstance(pred, tuple):
                    pred = pred[0]
                loss = torch.nn.functional.mse_loss(pred, gt_latent)  # MSE in latent space

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if lr_scheduler is not None:
                lr_scheduler.step()

            step += 1
            if step % 10 == 0:
                logger.info(
                    f"Step {step}/{num_iters}, lr: {optimizer.param_groups[0]['lr']:.4e}, Loss: {loss.item():.6f}"
                )
            if step % val_every == 0:
                logger.info(f"Running validation at step {step}/{num_iters}...")
                _run_validation(
                    model=model,
                    vae=vae,
                    val_dataloader=val_dataloader,
                    device=device,
                    metric_fn=metric_fn,
                    is_neg_1_1=is_neg_1_1,
                )
            if step >= num_iters:
                break
        if step >= num_iters:
            break


def _to_rgb(x: torch.Tensor, *, is_neg_1_1: bool) -> torch.Tensor:
    """Convert tensor to [0, 1] range for metrics calculation."""
    if is_neg_1_1:
        return ((x + 1.0) / 2.0).clamp(0.0, 1.0).float()
    return x.clamp(0.0, 1.0).float()


def _run_validation(
    *,
    model: torch.nn.Module,
    vae: CosmosRSVAE | FluxVAE,
    val_dataloader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    metric_fn: CRMetrics,
    is_neg_1_1: bool,
) -> None:
    model.eval()
    metric_fn.reset()
    total_loss = 0.0
    total_batches = 0
    vae_metric_fn = CRMetrics(lpips_device=device, interp_to=metric_fn.interp_to)

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Running validation ...", leave=False):
            img = batch["img"].to(device)
            gt = batch.get("gt", img).to(device)

            if isinstance(vae, FluxVAE):
                img = img[:, :3]
                gt = gt[:, :3]

            cond_latent = vae.encode(img)
            gt_latent = vae.encode(gt)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                pred = model(cond_latent, None, conditions=cond_latent)
                if isinstance(pred, tuple):
                    pred = pred[0]
                total_loss += torch.nn.functional.mse_loss(pred, gt_latent).item()
            total_batches += 1

            pred_px = vae.decode(pred, input_shape=gt.shape[1])
            gt_px = vae.decode(gt_latent, input_shape=gt.shape[1])
            metric_fn.update(
                _to_rgb(pred_px, is_neg_1_1=is_neg_1_1),  # type: ignore[arg-type]
                _to_rgb(gt_px, is_neg_1_1=is_neg_1_1),  # type: ignore[arg-type]
            )

            # VAE reconstruction metrics: compare VAE(GT) with original GT
            vae_metric_fn.update(
                _to_rgb(gt_px, is_neg_1_1=is_neg_1_1),  # type: ignore[arg-type]
                _to_rgb(gt, is_neg_1_1=is_neg_1_1),  # type: ignore[arg-type]
            )

    avg_loss = total_loss / max(total_batches, 1)
    metrics = metric_fn.compute()  # type: ignore[call-arg]
    vae_metrics = vae_metric_fn.compute()  # type: ignore[call-arg]
    logger.info(
        "Val: loss={:.6f}, PSNR={:.4f}, SSIM={:.4f}, LPIPS={:.4f}, RMSE={:.4f} | "
        "VAE Recon GT: PSNR={:.4f}, SSIM={:.4f}, LPIPS={:.4f}, RMSE={:.4f}".format(
            avg_loss,
            metrics["PSNR"].item(),
            metrics["SSIM"].item(),
            metrics["LPIPS"].item(),
            metrics["RMSE"].item(),
            vae_metrics["PSNR"].item(),
            vae_metrics["SSIM"].item(),
            vae_metrics["LPIPS"].item(),
            vae_metrics["RMSE"].item(),
        )
    )
    model.train()


def _get_model_cfg(cfg: DictConfig) -> Any:
    model_cfg = cfg.model
    if hasattr(model_cfg, "cloud_removal_model"):
        model_cfg = model_cfg.cloud_removal_model
    return model_cfg


def _build_optimizer(cfg: DictConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    opt_cfg = cfg.train.cloud_removal_optim
    optimizer_factory = hydra.utils.instantiate(opt_cfg)
    if "muon" in str(opt_cfg._target_).lower():
        return optimizer_factory(named_parameters=list(model.named_parameters()))
    return optimizer_factory(model.parameters())


def _build_scheduler(cfg: DictConfig, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
    sched_cfg = cfg.train.cloud_removal_sched
    scheduler_factory = hydra.utils.instantiate(sched_cfg)
    return scheduler_factory(optimizer=optimizer)


def _build_vae(cfg: DictConfig, device: torch.device) -> CosmosRSVAE | FluxVAE:
    if not hasattr(cfg, "vae") or cfg.vae is None:
        raise ValueError("Fast trainer requires cfg.vae to be set.")
    vae = hydra.utils.instantiate(cfg.vae).to(device)
    if not isinstance(vae, (CosmosRSVAE, FluxVAE)):
        raise TypeError(f"cfg.vae must be CosmosRSVAE or FluxVAE, got {type(vae)}")
    return vae


def _get_dataloaders(
    cfg: DictConfig,
) -> tuple[Iterable[dict[str, torch.Tensor]], Iterable[dict[str, torch.Tensor]]]:
    _, train_loader = hydra.utils.instantiate(cfg.dataset.train_loader)
    _, val_loader = hydra.utils.instantiate(cfg.dataset.val_loader)
    return train_loader, val_loader


if __name__ == "__main__":
    _config_name = "cloud_removal_cuhk_cr1_regression_unet"

    @hydra.main(
        config_path="../configs/cloud_removal",
        config_name=_config_name,
        version_base=None,
    )
    def main(cfg: DictConfig) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_cfg = _get_model_cfg(cfg)
        model = hydra.utils.instantiate(model_cfg).to(device)
        vae = _build_vae(cfg, device)
        train_dataloader, val_dataloader = _get_dataloaders(cfg)
        metric_fn = CRMetrics(lpips_device=device, interp_to=getattr(cfg.val, "metrics_interp_to", None))

        optimizer = _build_optimizer(cfg, model)
        lr_scheduler = _build_scheduler(cfg, optimizer)
        num_iters = int(getattr(cfg.train, "max_steps", 100_000))
        val_every = int(getattr(cfg.val, "val_duration", 500))
        is_neg_1_1 = bool(getattr(cfg.train, "is_neg_1_1", True))

        # debug
        # val_every = 10

        train(
            model=model,
            vae=vae,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            device=device,
            metric_fn=metric_fn,
            num_iters=num_iters,
            lr_scheduler=lr_scheduler,
            val_every=val_every,
            is_neg_1_1=is_neg_1_1,
        )

    main()

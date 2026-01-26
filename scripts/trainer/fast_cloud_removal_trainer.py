from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.stage2.cloud_removal.diffusion.vae import CosmosRSVAE, FluxVAE
from src.stage2.cloud_removal.metrics.basic import CRMetrics
from src.utilities.transport.SDB.plan import DiffusionTarget, SDBContinuousPlan, SDBContinuousSampler


def train_regression(
    model: torch.nn.Module,
    vae: CosmosRSVAE | FluxVAE,
    train_dataloader: Iterable[dict[str, torch.Tensor]],
    val_dataloader: Iterable[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    metric_fn: CRMetrics,
    num_iters: int = 100_000,
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
                _run_validation_regression(
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


def _run_validation_regression(
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


def _build_sdb_plan(cfg: DictConfig, *, device: torch.device) -> SDBContinuousPlan:
    sdb_cfg = getattr(cfg.train, "sdb", None)
    if sdb_cfg is None:
        raise ValueError("SDB training requires cfg.train.sdb to be set.")

    plan_tgt = DiffusionTarget(str(getattr(sdb_cfg, "plan_tgt", "x_0")))
    gamma_max = float(getattr(sdb_cfg, "gamma_max", 0.2))
    eps_eta = float(getattr(sdb_cfg, "eps_eta", 1.0))
    alpha_beta_type = str(getattr(sdb_cfg, "alpha_beta_type", "linear"))
    diffusion_type = str(getattr(sdb_cfg, "diffusion_type", "bridge"))
    t_train_type = str(getattr(sdb_cfg, "t_train_type", "edm"))
    t_sample_type = str(getattr(sdb_cfg, "t_sample_type", "edm"))

    t_train_kwargs_raw = getattr(sdb_cfg, "t_train_kwargs", None)
    t_train_kwargs: dict[str, Any] = {"device": device, "clip_t_min_max": (1e-4, 1 - 1e-4)}
    if t_train_kwargs_raw is not None:
        t_train_kwargs = OmegaConf.to_container(t_train_kwargs_raw, resolve=True)  # type: ignore[assignment]
        if not isinstance(t_train_kwargs, dict):
            raise TypeError("cfg.train.sdb.t_train_kwargs must be a mapping")

    clip_range = t_train_kwargs.get("clip_t_min_max")
    if clip_range is not None:
        t_train_kwargs["clip_t_min_max"] = (float(clip_range[0]), float(clip_range[1]))
    t_train_kwargs["device"] = device

    return SDBContinuousPlan(
        plan_tgt=plan_tgt,
        gamma_max=gamma_max,
        eps_eta=eps_eta,
        alpha_beta_type=alpha_beta_type,
        diffusion_type=diffusion_type,
        t_train_type=t_train_type,
        t_train_kwargs=t_train_kwargs,
        t_sample_type=t_sample_type,
    )


def _build_sdb_sampler(cfg: DictConfig, plan: SDBContinuousPlan) -> SDBContinuousSampler:
    sample_noisy_x1_b = float(getattr(getattr(cfg.train, "sdb", None), "sample_noisy_x1_b", 0.0))
    return SDBContinuousSampler(plan, sample_noisy_x1_b=sample_noisy_x1_b)


def _build_sdb_time_grid(
    cfg: DictConfig, plan: SDBContinuousPlan, *, device: torch.device, num_steps: int
) -> torch.Tensor:
    sdb_cfg = getattr(cfg.train, "sdb", None)
    sample_kwargs: dict[str, Any] = {}
    if sdb_cfg is not None:
        sample_kwargs_raw = getattr(sdb_cfg, "t_sample_kwargs", None)
        if sample_kwargs_raw is not None:
            sample_kwargs = OmegaConf.to_container(sample_kwargs_raw, resolve=True)  # type: ignore[assignment]
            if not isinstance(sample_kwargs, dict):
                raise TypeError("cfg.train.sdb.t_sample_kwargs must be a mapping")

    sample_kwargs["n_timesteps"] = int(num_steps)
    sample_kwargs.setdefault("t_min", plan.t_min)
    sample_kwargs.setdefault("t_max", plan.t_max)
    if plan.t_sample_type == "edm":
        sample_kwargs.setdefault("rho", 0.3)
    if plan.t_sample_type == "sigmoid":
        sample_kwargs.setdefault("k", 7.0)

    return plan.sample_continous_t(**sample_kwargs).to(device=device)


def train_sdb(
    *,
    cfg: DictConfig,
    model: torch.nn.Module,
    vae: CosmosRSVAE | FluxVAE,
    plan: SDBContinuousPlan,
    sampler: SDBContinuousSampler,
    train_dataloader: Iterable[dict[str, torch.Tensor]],
    val_dataloader: Iterable[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    metric_fn: CRMetrics,
    num_iters: int,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    val_every: int,
    is_neg_1_1: bool,
) -> None:
    model.train()
    vae.eval().requires_grad_(False)
    scaler = torch.amp.GradScaler()

    loss_type = str(getattr(cfg.train, "sdb_loss_type", "mse")).lower()
    loss_weight = float(getattr(cfg.train, "sdb_loss_weight", 1.0))

    step = 0
    for _epoch_idx in range(num_iters):
        for batch in train_dataloader:
            optimizer.zero_grad(set_to_none=True)

            img = batch["img"].to(device)
            gt = batch.get("gt", img).to(device)

            if isinstance(vae, FluxVAE):
                img = img[:, :3]
                gt = gt[:, :3]

            with torch.no_grad():
                x_1 = vae.encode(img)
                x_0 = vae.encode(gt)

            t_vec = plan.train_continous_t(x_0.shape[0]).to(device=device, dtype=x_0.dtype)
            x_t, target = plan.get_x_t_with_target(t_vec, x_0, x_1)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                pred = model(x_t, t_vec, conditions=x_1)
                if isinstance(pred, tuple):
                    pred = pred[0]

                if loss_type in {"mse", "l2"}:
                    loss_val = torch.nn.functional.mse_loss(pred, target)
                elif loss_type in {"l1", "mae"}:
                    loss_val = torch.nn.functional.l1_loss(pred, target)
                else:
                    raise ValueError(f"Unknown sdb_loss_type: {loss_type}")

                loss = loss_val * loss_weight

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
                _run_validation_sdb(
                    cfg=cfg,
                    model=model,
                    vae=vae,
                    plan=plan,
                    sampler=sampler,
                    val_dataloader=val_dataloader,
                    device=device,
                    metric_fn=metric_fn,
                    is_neg_1_1=is_neg_1_1,
                )

            if step >= num_iters:
                break
        if step >= num_iters:
            break


def _run_validation_sdb(
    *,
    cfg: DictConfig,
    model: torch.nn.Module,
    vae: CosmosRSVAE | FluxVAE,
    plan: SDBContinuousPlan,
    sampler: SDBContinuousSampler,
    val_dataloader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    metric_fn: CRMetrics,
    is_neg_1_1: bool,
) -> None:
    model.eval()
    metric_fn.reset()
    vae_metric_fn = CRMetrics(lpips_device=device, interp_to=metric_fn.interp_to)

    sampler_cfg = getattr(cfg, "sampler", None)
    sampling_type = str(getattr(sampler_cfg, "type", "ode")) if sampler_cfg is not None else "ode"
    num_steps = int(getattr(sampler_cfg, "num_steps", 25)) if sampler_cfg is not None else 25
    progress = bool(getattr(sampler_cfg, "progress", False)) if sampler_cfg is not None else False

    # Latent space should not be clipped into [-1, 1]; it often collapses into a constant decode (black).
    clip_value = False

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Running validation ...", leave=False):
            img = batch["img"].to(device)
            gt = batch.get("gt", img).to(device)

            if isinstance(vae, FluxVAE):
                img = img[:, :3]
                gt = gt[:, :3]

            x_1 = vae.encode(img)
            x_0 = vae.encode(gt)

            time_grid = _build_sdb_time_grid(cfg, plan, device=device, num_steps=num_steps)
            model_kwargs: dict[str, Any] = {"conditions": x_1}

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                if sampling_type == "sde":
                    pred_x0, _, _ = sampler.sample_sde_euler(
                        model=model,
                        x_1=x_1,
                        time_grid=time_grid,
                        model_kwargs=model_kwargs,
                        clip_value=clip_value,
                        progress=progress,
                    )
                elif sampling_type == "ode":
                    pred_x0, _, _ = sampler.sample_ode_euler(
                        model=model,
                        x_1=x_1,
                        time_grid=time_grid,
                        model_kwargs=model_kwargs,
                        clip_value=clip_value,
                        progress=progress,
                    )
                else:
                    raise ValueError(f"Unknown sampling type: {sampling_type}")

            pred_px = vae.decode(pred_x0, input_shape=gt.shape[1])
            gt_px = vae.decode(x_0, input_shape=gt.shape[1])

            metric_fn.update(
                _to_rgb(pred_px, is_neg_1_1=is_neg_1_1),  # type: ignore[arg-type]
                _to_rgb(gt_px, is_neg_1_1=is_neg_1_1),  # type: ignore[arg-type]
            )
            vae_metric_fn.update(
                _to_rgb(gt_px, is_neg_1_1=is_neg_1_1),  # type: ignore[arg-type]
                _to_rgb(gt, is_neg_1_1=is_neg_1_1),  # type: ignore[arg-type]
            )

    metrics = metric_fn.compute()  # type: ignore[call-arg]
    vae_metrics = vae_metric_fn.compute()  # type: ignore[call-arg]
    logger.info(
        "Val: PSNR={:.4f}, SSIM={:.4f}, LPIPS={:.4f}, RMSE={:.4f} | "
        "VAE Recon GT: PSNR={:.4f}, SSIM={:.4f}, LPIPS={:.4f}, RMSE={:.4f}".format(
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
    _config_name = os.environ.get("FAST_CLOUD_REMOVAL_CONFIG", "cloud_removal_cuhk_cr1_regression_unet")

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

        train_mode = str(getattr(cfg.train, "train_mode", "regression")).lower()
        backend = str(getattr(cfg.train, "transport_backend", "flow_matching")).lower()

        if train_mode == "diffusion" and backend in {"sdb", "sdb-transport", "sdb_transport"}:
            plan = _build_sdb_plan(cfg, device=device)
            sampler = _build_sdb_sampler(cfg, plan)
            train_sdb(
                cfg=cfg,
                model=model,
                vae=vae,
                plan=plan,
                sampler=sampler,
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
        else:
            train_regression(
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

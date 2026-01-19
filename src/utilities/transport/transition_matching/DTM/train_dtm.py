import argparse
import copy
from copy import deepcopy
import logging
import os
from pathlib import Path
from collections import OrderedDict
import json
import torch
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from models.dtm import DTM
from models.backbone_model import Backbone_models
from models.head_model import HeadMLP

from utils.metric import AnalysisPanAcc
from dataset.pan_dataset import PanDataset
from utils.patch_util import split_into_patches, combine_patches

try:
    import wandb
except ImportError:
    wandb = None
import math
from torchvision.utils import make_grid

logger = get_logger(__name__)


def array2grid(x):
    x_rgb = x[:, :3, :, :]
    nrow = round(math.sqrt(x_rgb.shape[0]))
    x_rgb = make_grid(x_rgb.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x_rgb = x_rgb.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return x_rgb


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")],
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    log_with = args.report_to if args.report_to != "none" else None

    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=log_with,
        project_config=accelerator_project_config,
        # kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, "w") as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    patch_dim = args.patch_size * args.patch_size * args.image_channels
    backbone = Backbone_models[args.backbone_model](
        image_size=args.resolution,
        in_channels=args.image_channels,
        patch_size=args.patch_size,
        cond_channels=args.cond_channels,
        image_cond_drop_prob=args.image_cond_drop_prob,
    ).to(device)

    head = HeadMLP(
        backbone_h_dim=backbone.pos_embed.shape[-1],
        patch_dim=patch_dim,
        mlp_hidden_dim=args.head_mlp_dim,
        num_layers=args.head_num_layers,
    ).to(device)

    dtm_handler = DTM(backbone_disc_T=args.dtm_backbone_T, fm_head_N=args.dtm_head_N)

    ema_backbone = deepcopy(backbone).to(device)
    ema_head = deepcopy(head).to(device)
    requires_grad(ema_backbone, False)
    requires_grad(ema_head, False)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if accelerator.is_main_process:
        num_params_backbone = sum(p.numel() for p in backbone.parameters())
        num_params_head = sum(p.numel() for p in head.parameters())
        logger.info(f"Backbone Parameters: {num_params_backbone:,}")
        logger.info(f"Head Parameters: {num_params_head:,}")
        logger.info(f"Total Parameters: {num_params_backbone + num_params_head:,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(head.parameters()),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.adam_weight_decay,
        eps=1e-8,
    )

    train_dataset = PanDataset(args.train_data_path, division=1024.0)
    valid_dataset = PanDataset(args.valid_data_path, division=1024.0)

    local_batch_size = int(args.batch_size // accelerator.num_processes)
    steps_per_epoch = len(train_dataset) // (args.batch_size * args.gradient_accumulation_steps)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=args.valid_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    if accelerator.is_main_process:
        logger.info(f"Training dataset: {len(train_dataset)} images ({args.train_data_path})")
        logger.info(f"Validation dataset: {len(valid_dataset)} images ({args.valid_data_path})")

    backbone, head, optimizer, train_dataloader = accelerator.prepare(backbone, head, optimizer, train_dataloader)

    if args.epochs > 0:
        max_train_steps = min(args.max_train_steps, steps_per_epoch * args.epochs)
    else:
        max_train_steps = args.max_train_steps

    ema_backbone = ema_backbone.to(device)
    ema_head = ema_head.to(device)
    update_ema(ema_backbone, backbone, decay=0)
    update_ema(ema_head, head, decay=0)

    global_step = 0

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="DTM_Pansharpening", config=vars(args), init_kwargs={"wandb": {"name": f"{args.exp_name}"}}
        )

    progress_bar = tqdm(
        range(max_train_steps), initial=global_step, desc="Steps", disable=not accelerator.is_local_main_process
    )

    if accelerator.is_main_process:
        fixed_batch = next(iter(valid_dataloader))
        fixed_condition_images = fixed_batch["cond_image"][: args.valid_batch_size].to(device)
        fixed_lms_images = fixed_batch["lms_image"][: args.valid_batch_size].to(device)
        fixed_gt_images = fixed_batch["gt_image"][: args.valid_batch_size].to(device)
        fixed_X0 = torch.rand_like(fixed_gt_images)

    for epoch in range(args.epochs):
        backbone.train()
        head.train()
        for batch in train_dataloader:
            gt_images = batch["gt_image"].to(device)
            cond_images = batch["cond_image"].to(device)

            with accelerator.accumulate(backbone, head):
                loss_dict = dtm_handler.loss(
                    backbone, head, patch_size=args.patch_size, X_T=gt_images, conditions=cond_images, to_1d=True
                )
                loss = loss_dict["loss"]

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = list(backbone.parameters()) + list(head.parameters())
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                unwrapped_backbone = accelerator.unwrap_model(backbone)
                unwrapped_head = accelerator.unwrap_model(head)
                update_ema(ema_backbone, unwrapped_backbone)
                update_ema(ema_head, unwrapped_head)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        checkpoint = {
                            "backbone": accelerator.unwrap_model(backbone).state_dict(),
                            "head": accelerator.unwrap_model(head).state_dict(),
                            "ema_backbone": ema_backbone.state_dict(),
                            "ema_head": ema_head.state_dict(),
                            "opt": optimizer.state_dict(),
                            "args": args,
                            "steps": global_step,
                        }
                        checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")

                if global_step % args.sampling_steps == 0 or global_step == 1:
                    if accelerator.is_main_process:
                        analysis_metric = AnalysisPanAcc()
                        with torch.no_grad():
                            B, _, H, W = fixed_gt_images.shape

                            validation_chunk_size = args.resolution  # 64
                            cond_patches = split_into_patches(fixed_condition_images, validation_chunk_size)
                            X0_patches = split_into_patches(fixed_X0, validation_chunk_size)

                            predicted_patches = dtm_handler.sample_loop(
                                ema_backbone,
                                ema_head,
                                X_0=X0_patches,
                                patch_size=args.patch_size,
                                conditions=cond_patches,
                            )

                            reconstructed_images = combine_patches(predicted_patches, B, args.image_channels, H, W, 64)
                            # 分块大小

                            reconstructed_images = reconstructed_images + fixed_lms_images

                            reconstructed_images = reconstructed_images.clamp(0, 1)

                            analysis_metric(fixed_gt_images, reconstructed_images)

                            metric_results = analysis_metric.acc_ave

                        results_txt_path = os.path.join(accelerator.project_dir, "validation_metrics.txt")
                        with open(results_txt_path, "a") as f:
                            f.write(f"--- Global Step: {global_step} ---\n")
                            for metric, value in metric_results.items():
                                f.write(f"{metric}: {value:.4f}\n")
                            f.write("\n")

                        wandb_metrics = {f"metrics/{k}": v for k, v in metric_results.items()}

                        use_wandb = ("wandb" in [str(x) for x in accelerator.log_with]) and wandb is not None

                        if use_wandb:
                            recon_grid = array2grid(reconstructed_images)
                            lms_grid = array2grid(fixed_lms_images)
                            gt_grid = array2grid(fixed_gt_images)

                            wandb.log(
                                {
                                    "samples/reconstructed": wandb.Image(recon_grid),
                                    "samples/lms": wandb.Image(lms_grid),
                                    "samples/ground_truth": wandb.Image(gt_grid),
                                    **wandb_metrics,
                                },
                                step=global_step,
                            )

                logs = {
                    "loss": accelerator.gather(loss).mean().detach().item(),
                    "grad_norm": accelerator.gather(grad_norm).mean().detach().item(),
                }

                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    break
            if global_step >= args.max_train_steps:
                break

    accelerator.end_training()


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="DTM Training")

    # 日志和保存
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--checkpointing-steps", type=int, default=1000)

    parser.add_argument("--train-data-path", type=str, default="/home/liu/CPY/gf2/train/train_gf2.h5")
    parser.add_argument(
        "--valid-data-path", type=str, default="/home/liu/CPY/gf2/test/reduced_examples/test_gf2_multiExm1.h5"
    )

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--valid-batch-size", type=int, default=20)

    parser.add_argument("--backbone_model", type=str, default="Backbone-Pansharpening")
    parser.add_argument("--patch_size", type=int, default=4, help="Patch size for image tokenization.")
    parser.add_argument("--image_channels", type=int, default=4, help="Number of channels in the target image.")
    parser.add_argument("--cond_channels", type=int, default=5, help="Number of channels in the conditioning image.")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--image_cond_drop_prob", type=float, default=0.0, help="CFG dropout probability for backbone.")

    # Head MLP 参数
    parser.add_argument("--head_mlp_dim", type=int, default=512, help="Hidden dimension of the Head MLP.")
    parser.add_argument("--head_num_layers", type=int, default=6, help="Number of layers in the Head MLP.")

    # DTM 框架参数
    parser.add_argument("--dtm_backbone_T", type=int, default=10, help="Number of discrete steps for the backbone.")
    parser.add_argument(
        "--dtm_head_N",
        type=int,
        default=50,
        help="Number of integration steps for the head's ODE solver during sampling.",
    )

    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-train-steps", type=int, default=1_000_000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-weight-decay", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", default=1.0, type=float)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)

    args = parser.parse_args(input_args)

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

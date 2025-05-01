"""Continuous evaluation script for FlowMo.

Code is (partially) from https://github.com/TencentARC/SEED-Voken. Thanks!
"""

import functools
import os
import time

import lpips
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from PIL import Image
from scipy import linalg
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from flowmo import train_utils
from flowmo.baselines import baselines
from flowmo.inception import InceptionV3


def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1.0, 1.0)
    x = (x + 1.0) / 2.0
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, (
        "Training and test mean vectors have different lengths"
    )
    assert sigma1.shape == sigma2.shape, (
        "Training and test covariances have different dimensions"
    )

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def get_subsampled_dataset(dataset, subsample_rate=1):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    indices = [
        idx
        for idx in np.arange(len(dataset))
        if (idx % (world_size * subsample_rate)) == rank
    ]
    return torch.utils.data.Subset(dataset, indices)


def _gather_concat_np(pred_np):
    pred = torch.from_numpy(pred_np).cuda().contiguous()
    out = [torch.zeros_like(pred) for _ in range(dist.get_world_size())]
    dist.all_gather(out, pred)
    return torch.concatenate(out).cpu().numpy()


def _calculate_fid(pred_xs, pred_recs):
    pred_xs = np.concatenate(pred_xs, axis=0)
    pred_recs = np.concatenate(pred_recs, axis=0)

    pred_xs = _gather_concat_np(pred_xs)
    pred_recs = _gather_concat_np(pred_recs)
    print(f"Compute FID from {len(pred_xs), len(pred_recs)} samples for pred,recs")

    mu_x = np.mean(pred_xs, axis=0)
    sigma_x = np.cov(pred_xs, rowvar=False)

    mu_rec = np.mean(pred_recs, axis=0)
    sigma_rec = np.cov(pred_recs, rowvar=False)

    fid_value = calculate_frechet_distance(mu_x, sigma_x, mu_rec, sigma_rec)
    return fid_value


def _reduce_scalar(scalar):
    scalar = torch.from_numpy(np.array([scalar])).cuda()
    dist.all_reduce(scalar, op=dist.ReduceOp.AVG)
    return scalar.cpu().numpy().item()


def eval_imagenet(model, config):
    config = train_utils.restore_config(config)
    model = model.eval()

    bs = config.data.eval_batch_size
    config.data.batch_size = bs

    val_dataloader = train_utils.load_dataset(config, split="val")

    # FID score related
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    inception_model = InceptionV3([block_idx]).cuda()
    inception_model.eval()

    pred_xs = []
    pred_recs = []

    loss_fn_alex = lpips.LPIPS(net="alex").cuda()  # best forward scores
    loss_fn_vgg = lpips.LPIPS(net="vgg").to(
        "cuda"
    )  # closer to "traditional" perceptual loss, when used for optimization
    lpips_alex = 0.0
    lpips_vgg = 0.0
    ssim_value = 0.0
    psnr_value = 0.0
    pairwise_inception_value = 0.0

    num_images = 0
    num_iter = 0

    dataset = val_dataloader.dataset
    num_examples = len(dataset)
    print("Total number of examples: ", num_examples)
    global_batch_size = bs * dist.get_world_size() * config.eval.subsample_rate
    assert num_examples % global_batch_size == 0, (
        num_examples,
        bs,
        dist.get_world_size(),
        config.eval.subsample_rate,
    )

    dataset = get_subsampled_dataset(dataset, subsample_rate=config.eval.subsample_rate)
    num_examples = len(dataset)
    print("Total number of examples for this task:", num_examples)

    assert num_examples % bs == 0, (num_examples, bs)

    eval_batches = num_examples // bs

    # rebuild dataloader after subsampling
    dataloader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    assert len(dataloader) == eval_batches, (len(dataloader), eval_batches)

    if config.eval.eval_baseline == "sdxl":
        _, reconstruct_fn = baselines.load_sdxl()
    elif "titok" in config.eval.eval_baseline:
        _, reconstruct_fn = baselines.load_titok(config.eval.eval_baseline)
    elif config.eval.eval_baseline in ["llamagen16", "llamagen32"]:
        _, reconstruct_fn = baselines.load_llamagen(config.eval.eval_baseline)
    elif config.eval.eval_baseline in ["cosmos_8x8", "cosmos_16x16"]:
        _, reconstruct_fn = baselines.load_cosmos(config.eval.eval_baseline)
    elif config.eval.eval_baseline == "flux":
        _, reconstruct_fn = baselines.load_flux()
    elif config.eval.eval_baseline == "magvitv2_tokenizer":
        _, reconstruct_fn = baselines.load_magvitv2_tokenizer_in16x16()
    elif config.eval.eval_baseline == "":
        # run normal evaluation
        dtype = torch.bfloat16 if train_utils.bfloat16_is_available() else torch.float32
        print(dtype)
        reconstruct_fn = functools.partial(model.reconstruct, dtype=dtype)
    else:
        raise NotImplementedError

    generated_batches = []

    with torch.no_grad():
        for idx, batch in tqdm(
            enumerate(
                dataloader,
            ),
            total=len(dataloader),
        ):
            print("processing batch idx: ", idx)
            images = batch["image"].permute(0, 3, 1, 2).cuda()
            batch["image"] = images
            num_images += images.shape[0]

            reconstructed_images = reconstruct_fn(images)
            reconstructed_images = reconstructed_images.clamp(-1, 1)

            images = (images + 1) / 2
            reconstructed_images = (reconstructed_images + 1) / 2

            generated_batches.append(
                np.clip(
                    reconstructed_images.permute((0, 2, 3, 1)).cpu().numpy() * 255,
                    0,
                    255,
                ).astype(np.uint8)
            )

            if config.eval.reconstruction:
                # calculate lpips
                lpips_alex += loss_fn_alex(
                    images * 2 - 1, reconstructed_images * 2 - 1
                ).sum()
                lpips_vgg += loss_fn_vgg(
                    images * 2 - 1, reconstructed_images * 2 - 1
                ).sum()

                pred_x = inception_model(images)[0]
                pred_x = pred_x.squeeze(3).squeeze(2).cpu().numpy()
                pred_rec = inception_model(reconstructed_images)[0]
                pred_rec = pred_rec.squeeze(3).squeeze(2).cpu().numpy()

                pred_xs.append(pred_x)
                pred_recs.append(pred_rec)

                # calculate PSNR and SSIM
                rgb_restored = (
                    (reconstructed_images * 255.0)
                    .permute(0, 2, 3, 1)
                    .to("cpu", dtype=torch.uint8)
                    .numpy()
                )
                rgb_gt = (
                    (images * 255.0)
                    .permute(0, 2, 3, 1)
                    .to("cpu", dtype=torch.uint8)
                    .numpy()
                )
                rgb_restored = rgb_restored.astype(np.float32) / 255.0
                rgb_gt = rgb_gt.astype(np.float32) / 255.0
                ssim_temp = 0
                psnr_temp = 0
                B, _, _, _ = rgb_restored.shape
                for i in range(B):
                    rgb_restored_s, rgb_gt_s = rgb_restored[i], rgb_gt[i]
                    ssim_temp += ssim_loss(
                        rgb_restored_s,
                        rgb_gt_s,
                        data_range=1.0,
                        channel_axis=-1,
                    )
                    psnr_temp += psnr_loss(rgb_gt, rgb_restored)
                ssim_value += ssim_temp / B
                psnr_value += psnr_temp / B

                pairwise_inception_value += ((pred_x - pred_rec) ** 2).sum().item() / B

            num_iter += 1

            # Need this to prevent replicas from getting too out of sync w each other
            dist.barrier()

    generated_batches = np.concatenate(generated_batches)
    generated_batches = _gather_concat_np(generated_batches)
    print(generated_batches.shape)

    fid_value = _calculate_fid(pred_xs, pred_recs)

    if config.eval.reconstruction:
        lpips_alex_value = _reduce_scalar((lpips_alex / num_images).item())
        lpips_vgg_value = _reduce_scalar((lpips_vgg / num_images).item())
        ssim_value = _reduce_scalar(ssim_value / num_iter)
        psnr_value = _reduce_scalar(psnr_value / num_iter)
        pairwise_inception_value = _reduce_scalar(pairwise_inception_value / num_iter)

        reconstruction_metrics = {
            "LPIPS_ALEX: ": lpips_alex_value,
            "LPIPS_VGG: ": lpips_vgg_value,
            "PSNR: ": psnr_value,
            "SSIM: ": ssim_value,
            "PAIRWISE_INCEPTION": pairwise_inception_value,
        }

    metrics = {"FID: ": fid_value, **reconstruction_metrics}
    return generated_batches, metrics


def _safe_load(checkpoint_path):
    while True:
        try:
            return torch.load(checkpoint_path)
        except Exception as e:
            print(e)
            time.sleep(30)


def main(args, config):
    config = train_utils.restore_config(config)

    torch.multiprocessing.set_start_method("forkserver", force=True)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    os.environ["TORCHELASTIC_ERROR_FILE"] = os.path.join(
        config.eval.eval_dir, "error_v0.json"
    )

    train_utils.soft_init()

    evaluated_checkpoints = set()
    eval_dir = config.eval.eval_dir
    assert eval_dir, eval_dir
    writer = SummaryWriter(eval_dir)

    seed = config.global_seed * dist.get_world_size() + dist.get_rank()
    torch.manual_seed(seed)

    print(OmegaConf.to_yaml(config))

    rank = dist.get_rank()
    print(dist.get_world_size())
    print(dist.get_rank())
    print(torch.cuda.device_count())

    device = rank % torch.cuda.device_count()
    print(device, torch.cuda.device_count())
    torch.cuda.set_device(device)

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(
        eval_dir, f"torchinductor_ksarge_{args.experiment_name}_{str(rank)}"
    )
    main_tag = "eval/"

    while True:
        if config.eval.force_ckpt_path:
            checkpoint_path = config.eval.force_ckpt_path
        else:
            checkpoint_path = train_utils.get_last_checkpoint(config, eval_dir)

        if checkpoint_path is None or checkpoint_path in evaluated_checkpoints:
            print("No checkpoints to evaluate! Sleeping...")
            time.sleep(30)
            continue

        print("Evaluating checkpoint!", checkpoint_path)
        model = train_utils.build_model(config)
        state_dict = train_utils.load_state_dict(checkpoint_path)
        total_steps = state_dict["total_steps"]
        model.load_state_dict(state_dict[config.eval.state_dict_key])

        images, metrics = eval_imagenet(model, config)
        print(metrics)
        torch.distributed.barrier()

        if dist.get_rank() == 0 and not config.eval.eval_baseline:
            for metric in metrics:
                writer.add_scalar(
                    main_tag + metric,
                    metrics[metric],
                    global_step=total_steps,
                )
        evaluated_checkpoints.add(checkpoint_path)

        del state_dict, model, images
        if not config.eval.continuous:
            return


if __name__ == "__main__":
    try:
        args, config = train_utils.get_args_and_config()
        main(args, config)
    finally:
        torch.distributed.destroy_process_group()

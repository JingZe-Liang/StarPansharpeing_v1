import torch
import webdataset as wds

from src.data.codecs import mat_decode_io, safetensors_decode_io, tiff_decode_io
from src.utilities.logging import log_print


def get_panshap_dataloaders(
    wds_paths: str | list[str],
    batch_size: int,
    num_workers: int,
    shuffle_size: int = 100,
    to_neg_1_1: bool = True,
    resample: bool = True,
    latent_ext: str = "safetensors",
):
    if isinstance(wds_paths, str):
        wds_paths = [wds_paths]
    is_ddp = (
        torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
    )
    dataset = wds.WebDataset(
        wds_paths,
        resampled=resample,
        shardshuffle=shuffle_size if is_ddp else False,
        handler=wds.warn_and_continue,
        nodesplitter=wds.shardlists.split_by_node
        if is_ddp
        else wds.shardlists.single_node_only,
        seed=2025,
        verbose=True,
        detshuffle=True if shuffle_size > 0 else False,
    )

    dataset = dataset.decode(
        wds.handle_extension("tif tiff", tiff_decode_io),
        wds.handle_extension("mat", mat_decode_io),
        wds.handle_extension("safetensors", safetensors_decode_io),
        "torch",
    )

    # * --- dict mapper of images and latents ---

    def img_dict_mapper_with_ext(sample: dict):
        # keys: ['hrms', 'lrms', 'pan', 'ms', 'hrms_latent', 'lrms_latent', 'pan_latent', 'ms_latent']
        hrms = sample["hrms.tiff"].float()
        lrms = sample["lrms.tiff"].float()
        pan = sample["pan.tiff"].float()

        # normalize
        hrms = hrms / hrms.max()
        lrms = lrms / lrms.max()
        pan = pan / pan.max()

        if to_neg_1_1:
            hrms = hrms * 2 - 1
            lrms = lrms * 2 - 1
            pan = pan * 2 - 1

        # if has latents
        if "hrms_latent" in sample:
            hrms_latent = sample[f"hrms_latent.{latent_ext}"].float()
            lrms_latent = sample[f"lrms_latent.{latent_ext}"].float()
            pan_latent = sample[f"pan_latent.{latent_ext}"].float()
            sample = {
                "hrms": hrms,
                "lrms": lrms,
                "pan": pan,
                "hrms_latent": hrms_latent,
                "lrms_latent": lrms_latent,
                "pan_latent": pan_latent,
            }

        else:
            sample = {
                "hrms": hrms,
                "lrms": lrms,
                "pan": pan,
            }
        return sample

    dataset = dataset.map(img_dict_mapper_with_ext)

    # since we do not use any transforms, the batch and unbatch is useless.

    dataloader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=6,
        droplast=False,
    )

    dataloader = dataloader.with_length(10_000)  # 10k pairs of the dataloader

    log_print(f"[Pansharpening Dataset]: constructed the dataloader")

    return dataset, dataloader

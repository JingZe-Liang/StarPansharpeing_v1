import io
from pathlib import Path
from typing import Sequence

import accelerate
import scipy.io
import tifffile
import torch
import torch.distributed
import webdataset as wds
from accelerate.state import PartialState
from kornia.augmentation import (
    AugmentationSequential,
    RandomBoxBlur,
    RandomChannelShuffle,
    RandomCutMixV2,
    RandomHorizontalFlip,
    RandomRotation,
    RandomSharpness,
    RandomVerticalFlip,
)
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from loguru import logger
from natsort import natsorted
from torch.utils.data import DataLoader, Dataset


def tiff_decoder(x):
    return tifffile.imread(io.BytesIO(x))


def get_dict_tensor_mapper(to_neg_1_1=True):
    def wds_to_dict_tensor_mapper(sample):
        img = torch.as_tensor(sample["img.tiff"]).float()
        img_max = img.max()
        img = img / img_max
        if to_neg_1_1:
            img = img * 2 - 1
        img = img.permute(-1, 0, 1)

        return {"img": img, "img_max": img_max}

    return wds_to_dict_tensor_mapper


class HyperRandomGrayScale(IntensityAugmentationBase2D):
    def __init__(self, p=0.5):
        super().__init__(p)

    def apply_transform(self, input, params, flags, transform=None):
        assert input.ndim == 4
        c = input.shape[1]
        gray = input.mean(dim=1, keepdim=True).repeat_interleave(c, dim=1)
        return gray


def hyper_transform(
    op_list: tuple[str],
    probs: tuple[float] | float = 0.5,
    random_apply: int | tuple[int] = 2,
):
    if isinstance(probs, float):
        probs = [probs] * len(op_list)
    assert len(probs) == len(op_list)

    _op_list_cls = dict(
        grayscale=lambda p: HyperRandomGrayScale(p=p),
        channel_shuffle=lambda p: RandomChannelShuffle(p=p),
        sharpness=lambda p: RandomSharpness(p=p, sharpness=[0.5, 1.0]),
        rotation=lambda p: RandomRotation((-30, 30), p=p),
        horizontal_flip=lambda p: RandomHorizontalFlip(p=p),
        vertical_flip=lambda p: RandomVerticalFlip(p=p),
        cutmix=lambda p: RandomCutMixV2(num_mix=1, p=p, cut_size=(0.4, 0.6)),
        blur=lambda p: RandomBoxBlur((3, 3), p=p),
    )

    ops = []
    for op_str, prob in zip(op_list, probs):
        op = _op_list_cls[op_str]
        ops.append(op(prob))

    op_seq = AugmentationSequential(
        *ops,
        data_keys=["input"],
        random_apply=tuple(random_apply)
        if isinstance(random_apply, Sequence)
        else random_apply,
        same_on_batch=False,
        keepdim=True,
    )

    def dict_mapper(sample):
        sample = op_seq(sample)
        return sample

    return dict_mapper


def get_hyperspectral_dataloaders(
    wds_paths: str | list[str],
    batch_size: int,
    num_workers: int,
    shuffle_size: int = 100,
    to_neg_1_1: bool = True,
    hyper_transforms_lst: tuple[str] | None = (
        "grayscale",
        "channel_shuffle",
        "rotation",
        "cutmix",
        "horizontal_flip",
        "vertical_flip",
    ),
    transform_prob: tuple[float] | float = 0.2,
    random_apply: int | tuple[int] = 1,
    resample: bool = True,
) -> tuple[wds.WebDataset, wds.WebLoader]:
    """Get dataloaders for hyperspectral image data

    Args:
        wds_paths (str | list[str]): Path or list of paths to WebDataset tar files
        batch_size (int): Number of samples per batch
        num_workers (int): Number of worker processes for data loading
        shuffle_size (int, optional): Buffer size for shuffling data. Defaults to 100.
        to_neg_1_1 (bool, optional): Whether to normalize data to [-1,1] range. Defaults to True.
        hyper_transforms_lst (tuple[str] | None, optional): List of data augmentation operations. Defaults to ("grayscale","channel_shuffle","rotation","cutmix","horizontal_flip","vertical_flip").
        transform_prob (tuple[float] | float, optional): Probability for each augmentation operation. Defaults to 0.2.
        random_apply (int | tuple[int], optional): Number of random augmentations to apply. Defaults to 1.
        resample (bool, optional): Whether to allow data resampling. Defaults to True.

    Returns:
        tuple[wds.WebDataset, wds.WebLoader]: Returns a tuple containing the WebDataset object and its corresponding dataloader
    """
    dict_mapper = get_dict_tensor_mapper(to_neg_1_1)
    use_transf = (
        hyper_transforms_lst is not None
        and len(hyper_transforms_lst) > 0
        and transform_prob > 0
    )
    if use_transf:
        transform = hyper_transform(hyper_transforms_lst, transform_prob, random_apply)
        logger.info(
            f"[HyperWebdataset]: use augmentations {hyper_transforms_lst} with prob {transform_prob}"
        )

    part_state = PartialState()
    is_ddp = part_state.use_distributed
    if is_ddp:
        logger.info("[HyperWebdataset]: using DDP, split the datset by node")

    dataset = wds.WebDataset(
        wds_paths,
        resampled=resample,  # no need `iter(dataloader)` for `next` function
        shardshuffle=shuffle_size if is_ddp else False,
        nodesplitter=wds.shardlists.split_by_node
        if is_ddp
        else wds.shardlists.single_node_only,
        seed=2025,
        verbose=True,
        detshuffle=True if shuffle_size > 0 else False,
    )
    dataset = dataset.decode(wds.handle_extension("tif tiff", tiff_decoder))
    dataset = dataset.map(dict_mapper)
    dataset = dataset.batched(batch_size)
    if use_transf:
        dataset = dataset.map_dict(img=transform)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=6,
        drop_last=False,
    )

    # unbatch, shuffle, and rebatch within different workers
    dataloader = dataloader.unbatched()
    if shuffle_size > 0:
        dataloader = dataloader.shuffle(shuffle_size)
    dataloader = dataloader.batched(batch_size)

    logger.info(
        f"[HyperDataset]: batch size: {batch_size}, num workers: {num_workers}, use transformations: {hyper_transforms_lst}"
    )

    return dataset, dataloader


def get_fast_test_hyperspectral_data(
    data_type: str = "DCF",
    batch_size: int = 1,
):
    """
    get a test data for model/module/function testing.
    """
    wds_paths = {
        "DCF": "data/DCF_2019_Track_2-8_bands-px_512-MSI-0000.tar",
        "MMSeg": "data/MMSeg_YREB_train_part-12_bands-MSI-0000.tar",
    }[data_type]

    _, dataloader = get_hyperspectral_dataloaders(
        wds_paths,
        batch_size=batch_size,
        num_workers=1,
        shuffle_size=100,
        to_neg_1_1=True,
    )

    return dataloader


def ms_pan_dir_paired_loader(
    path: str,
    ms_dir_name: str = "ms",
    pan_dir_name: str = "pan",
    batch_size: int = 1,
    num_workers: int = 1,
    to_neg_1_1: bool = True,
):
    """
    Read MS (Multispectral) and PAN (Panchromatic) images from a directory, and return a dataloader.

    Args:
        path (str): Root directory path containing MS and PAN image folders
        ms_dir_name (str, optional): Name of the multispectral images directory. Defaults to "ms".
        pan_dir_name (str, optional): Name of the panchromatic images directory. Defaults to "pan".
        batch_size (int, optional): Batch size for the dataloader. Defaults to 1.
        num_workers (int, optional): Number of workers for data loading. Defaults to 1.
        to_neg_1_1 (bool, optional): Whether to normalize images to [-1,1] range. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - dataset (MSPANPairedDataset): The dataset object
            - dataloader (DataLoader): PyTorch DataLoader for the dataset

    The loader supports both .tiff and .mat file formats. Images are automatically paired
    based on filename order using natural sorting. The images are normalized to [0,1] range
    and optionally to [-1,1] range if to_neg_1_1 is True.
    """

    class MSPANPairedDataset(Dataset):
        def __init__(
            self,
            path: str,
            ms_dir_name: str = "ms",
            pan_dir_name: str = "pan",
            to_neg_1_1: bool = True,
        ):
            self.path = Path(path)
            self.ms_dir_name = ms_dir_name
            self.pan_dir_name = pan_dir_name
            self.to_neg_1_1 = to_neg_1_1

            self.ms_paths = natsorted(
                list((self.path / self.ms_dir_name).glob("*.tiff"))
                + list((self.path) / self.ms_dir_name.glob("*.mat"))
            )
            self.pan_paths = natsorted(
                list((self.path / self.pan_dir_name).glob("*.tiff"))
                + list((self.path) / self.pan_dir_name.glob("*.mat"))
            )
            assert len(self.ms_paths) == len(self.pan_paths), (
                f"MS and PAN images are not paired, MS is {len(self.ms_paths)}, PAN is {len(self.pan_paths)}"
            )

        def __len__(self):
            return len(self.ms_paths)

        @staticmethod
        def read_file(path: Path):
            if path.suffix == ".tiff":
                return tifffile.imread(path)
            elif path.suffix == ".mat":
                # default to read the last key
                mat_file = scipy.io.loadmat(path)
                _key = list(mat_file.keys())[-1]
                return mat_file[_key]
            else:
                raise ValueError(f"Unsupported file type: {path.suffix}")

        def get_item(self, idx):
            ms_path = self.ms_paths[idx]
            pan_path = self.pan_paths[idx]

            key = ms_path.stem

            ms_img = self.read_file(ms_path)
            pan_img = self.read_file(pan_path)

            # to 0 to 1
            ms_img = ms_img / ms_img.max()
            pan_img = pan_img / pan_img.max()
            if self.to_neg_1_1:
                ms_img = ms_img * 2 - 1
                pan_img = pan_img * 2 - 1

            return key, ms_img, pan_img

        def __getitem__(self, idx):
            key, ms_img, pan_img = self.get_item(idx)
            return {
                "__key__": key,  # same as key in webdataset
                "MS": ms_img,
                "PAN": pan_img,
            }

    dataset = MSPANPairedDataset(
        path,
        ms_dir_name=ms_dir_name,
        pan_dir_name=pan_dir_name,
        to_neg_1_1=to_neg_1_1,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
    )

    return dataset, dataloader


if __name__ == "__main__":
    # Test config
    test_wds_path = [
        "data/DCF_2019_Track_2-8_bands-px_512-MSI-0000.tar",
        "data/DCF_2019_Track_2-8_bands-px_512-MSI-0001.tar",
        "data/DCF_2019_Track_2-8_bands-px_512-MSI-0002.tar",
        "data/DCF_2019_Track_2-8_bands-px_512-MSI-0003.tar",
        "data/DCF_2019_Track_2-8_bands-px_512-MSI-0004.tar",
        "data/DCF_2019_Track_2-8_bands-px_512-MSI-0005.tar",
        "data/DCF_2019_Track_2-8_bands-px_512-MSI-0006.tar",
    ]
    test_batch_size = 32
    test_num_workers = 2
    test_shuffle_size = 300
    accelerator = accelerate.Accelerator()

    # Get test dataloader
    test_dataset, test_loader = get_hyperspectral_dataloaders(
        wds_paths=test_wds_path,
        batch_size=test_batch_size,
        num_workers=test_num_workers,
        shuffle_size=test_shuffle_size,
        to_neg_1_1=True,
        transform_prob=1.0,
        random_apply=(1, 2),
    )

    # * for loop the images
    for batch in test_loader:
        img_tensor = batch["img"]  # shape: [N, C, H, W]
        print(f"proc={torch.distributed.get_rank()} - {img_tensor.shape}")

    # * plot the grid of images
    # # Get a batch of data
    # batch = next(iter(test_loader))
    # img_tensor = batch["img"]  # shape: [N, C, H, W]

    # # Select [4,2,0] channels for RGB visualization for all images in the batch
    # rgb_imgs = img_tensor[:, [4, 2, 0], :, :]

    # # Normalize from [-1, 1] to [0, 1]
    # rgb_imgs = (rgb_imgs + 1) / 2

    # # Create image grid using make_grid
    # grid = torchvision.utils.make_grid(rgb_imgs, nrow=8, padding=2, normalize=False)

    # # Convert to numpy and adjust dimension order for plotting
    # grid_img = grid.permute(1, 2, 0).numpy()

    # # Display and save the grid
    # plt.figure(figsize=(15, 15))
    # plt.imshow(grid_img)
    # plt.axis("off")
    # plt.savefig("multispectral_grid.png", bbox_inches="tight", pad_inches=0, dpi=300)
    # plt.close()

    # print(f"Batch shape: {img_tensor.shape}")
    # print(f"Range: min={img_tensor.min():.2f}, max={img_tensor.max():.2f}")
    # print(f"Data type: {img_tensor.dtype}")

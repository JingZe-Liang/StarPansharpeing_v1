import io

import tifffile
import torch
import webdataset as wds
from accelerate.state import PartialState


def tiff_decoder(key, x):
    if key.endswith(".tiff"):
        return tifffile.imread(io.BytesIO(x))
    else:
        return x


def get_dict_tensor_mapper(to_neg_1_1=True):
    def wds_to_dict_tensor_mapper(sample):
        img = torch.as_tensor(sample["img.tiff"]).float()
        img = img / img.max()
        if to_neg_1_1:
            img = img * 2 - 1
        img = img.permute(-1, 0, 1)

        return {"img": img}

    return wds_to_dict_tensor_mapper


def get_hyperspectral_dataloaders(
    wds_paths: str | list[str],
    batch_size: int,
    num_workers: int,
    shuffle_size: int = 100,
    to_neg_1_1: bool = True,
):
    dict_mapper = get_dict_tensor_mapper(to_neg_1_1)

    part_state = PartialState()
    is_ddp = part_state.use_distributed

    dataset = wds.WebDataset(
        wds_paths,
        resampled=True,
        shardshuffle=shuffle_size if is_ddp else False,
        nodesplitter=wds.shardlists.split_by_node
        if is_ddp
        else wds.shardlists.single_node_only,
        seed=2025,
        verbose=True,
        detshuffle=True if shuffle_size > 0 else False,
    )
    dataset = dataset.decode(tiff_decoder)
    dataset = dataset.map(dict_mapper)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=6,
        drop_last=False,
    )

    return dataset, dataloader


if __name__ == "__main__":
    # Test config
    test_wds_path = [
        "/HardDisk/ZiHanCao/datasets/Multispectral_webdatasets/MMSeg_YREB_train_part-12_bands-MSI-0000.tar",
        "/HardDisk/ZiHanCao/datasets/Multispectral_webdatasets/MMSeg_YREB_train_part-12_bands-MSI-0001.tar",
        "/HardDisk/ZiHanCao/datasets/Multispectral_webdatasets/MMSeg_YREB_train_part-12_bands-MSI-0002.tar",
        "/HardDisk/ZiHanCao/datasets/Multispectral_webdatasets/MMSeg_YREB_train_part-12_bands-MSI-0003.tar",
    ]
    test_batch_size = 32
    test_num_workers = 2
    test_shuffle_size = 300

    # Get test dataloader
    test_dataset, test_loader = get_hyperspectral_dataloaders(
        wds_paths=test_wds_path,
        batch_size=test_batch_size,
        num_workers=test_num_workers,
        shuffle_size=test_shuffle_size,
        to_neg_1_1=True,
    )

    # Test multiple batches
    num_batches_to_test = 5
    for i, batch in enumerate(test_loader):
        if i >= num_batches_to_test:
            break
        img_tensor = batch["img"]
        print(f"\nBatch {i+1}:")
        print(f"Shape: {img_tensor.shape}")
        print(f"Value range: min={img_tensor.min():.2f}, max={img_tensor.max():.2f}")
        print(f"Data type: {img_tensor.dtype}")

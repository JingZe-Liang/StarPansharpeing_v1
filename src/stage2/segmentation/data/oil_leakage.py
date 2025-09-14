from functools import partial

from src.data.multimodal_loader import MultimodalityDataloader, wids_image_decode

# * --- wids loader --- #


def get_oil_leakage_dataloaders(
    wds_paths,
    batch_size: int,
    num_workers: int,
    shuffle_size: int = 0,
    to_neg_1_1: bool = True,
    per_channel_norm: bool = True,
):
    img_fn_ = partial(
        wids_image_decode,
        to_neg_1_1=to_neg_1_1,
        permute=True,
        resize=None,
        process_img_keys="ALL",
        per_channel=per_channel_norm,
    )
    gt_img_fn_ = partial(
        wids_image_decode,
        to_neg_1_1=False,
        permute=True,
        resize=None,
        interp_mode="nearest",
        process_img_keys="ALL",
    )
    codecs = {"img": img_fn_, "gt": gt_img_fn_}
    ds, dl = MultimodalityDataloader.create_loader(
        wds_paths,
        codecs=codecs,
        to_neg_1_1=to_neg_1_1,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle_size=shuffle_size,
    )
    return ds, dl


def test_loading():
    import os

    import matplotlib.pyplot as plt

    from src.utilities.train_utils.visualization import visualize_hyperspectral_image

    os.makedirs("tmp/oil_leakage", exist_ok=True)

    img_path = "data/Downstreams/OilLeackage/tif/image_index.json"
    gt_path = "data/Downstreams/OilLeackage/tif/gt_index.json"

    ds, dl = get_oil_leakage_dataloaders(
        {"img": img_path, "gt": gt_path},
        batch_size=1,
        num_workers=0,
        shuffle_size=100,
        to_neg_1_1=False,
        per_channel_norm=True,
    )
    for i, sample in enumerate(dl):
        img, gt = sample["img"], sample["gt"]
        print(img.shape, gt.shape)
        # visualize
        # img_rgb = img[0, [59, 49, 39]].permute(1, 2, 0).numpy()
        img_rgb = visualize_hyperspectral_image(img, rgb_channels="mean", to_grid=True)
        gt_mask = gt[0, 0].numpy()
        plt.subplot(1, 2, 1)
        plt.imshow(img_rgb)
        plt.title("RGB")
        plt.subplot(1, 2, 2)
        plt.imshow(gt_mask)
        plt.title("GT")
        plt.savefig(f"tmp/oil_leakage/fig_{i}.jpg", dpi=200, bbox_inches="tight")
        plt.clf()


if __name__ == "__main__":
    test_loading()

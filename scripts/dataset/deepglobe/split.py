from pathlib import Path
from litdata.streaming.cache import Cache
from natsort import natsorted
from tqdm import tqdm


def main(save_dir: str):
    path = Path("/Data2/ZihanCao/dataset/RoadExtraction-DeepGlobe/train")
    imgs = list(path.glob("*_sat.jpg"))
    masks = list(path.glob("*_mask.png"))

    assert len(imgs) == len(masks)

    sort_fn = lambda x: x.as_posix()
    imgs = natsorted(imgs, key=sort_fn)
    masks = natsorted(masks, key=sort_fn)

    print(f"Found paired img/mask {len(imgs)}")

    train_ratio = 0.8
    train_len = int(len(imgs) * train_ratio)

    train_imgs = imgs[:train_len]
    train_masks = masks[:train_len]

    val_imgs = imgs[train_len:]
    val_masks = masks[train_len:]

    print(f"Train: {len(train_imgs)}, Val: {len(val_imgs)}")
    # ======
    print(f"Will save train pairs {len(train_imgs)} and val pairs {len(val_imgs)}")

    # train
    dd = {"train": (train_imgs, train_masks), "val": (val_imgs, val_masks)}
    for mode, (imgs_path, masks_path) in dd.items():
        save_path = Path(save_dir, mode)
        save_path.mkdir(parents=True, exist_ok=True)

        writer = Cache(save_path.as_posix(), chunk_bytes="256Mb")

        idx = 0
        for img, mask in tqdm(
            zip(imgs_path, masks_path),
            total=len(imgs_path),
            desc=f"mode={mode} data ...",
            leave=False,
        ):
            assert img.stem.split("_")[0] == mask.stem.split("_")[0], f"Mismatch name {img} != {mask}"
            img_bytes, mask_bytes = img.read_bytes(), mask.read_bytes()
            name = img.stem
            d = {"img": img_bytes, "mask": mask_bytes, "__key__": name}
            writer[idx] = d
            idx += 1
        writer.done()
        print(f"write litdata of {mode=}")

    print("done.")


main("data/Downstreams/RoadExtraction/DeepGlobe_Litdata/new_splits")

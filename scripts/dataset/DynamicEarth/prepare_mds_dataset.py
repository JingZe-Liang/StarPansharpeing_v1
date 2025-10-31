import os
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed, wait
from io import BytesIO
from multiprocessing import Process, Queue
from pathlib import Path

import numpy as np
import PIL.Image as Image
import tifffile
import torch
import torchvision.transforms as transforms
from loguru import logger
from streaming import MDSWriter, Stream, StreamingDataLoader, StreamingDataset
from streaming.base.format.mds import encodings
from streaming.base.format.mds.encodings import Encoding
from streaming.base.util import merge_index
from tqdm import tqdm

import src.utilities.logging.print
from src.data.utils import norm_img


class TiffEncoding(Encoding):
    def encode(self, img):
        if isinstance(img, bytes):
            return img
        with BytesIO() as b:
            tifffile.imwrite(b, img)
            return b.getvalue()

    def decode(self, byte_data):
        with BytesIO(byte_data) as b:
            img = tifffile.imread(b)
            return img


encodings._encodings["tiff"] = TiffEncoding


def _is_tiff(name: str):
    return name.lower().endswith((".tif", ".tiff"))


def _is_rgb(name: str):
    return name.lower().endswith((".jpg", ".jpeg", ".png", ".img_content"))


def tar_pure_image_read_fn(tar_file: str, has_caption=False, decode=False):
    tar = tarfile.TarFile(tar_file, "r")

    def inner(q: Queue):
        # only extract the image files
        _prev_name = None
        data = {}
        while True:
            member = None
            try:
                # member = tar_next_with_timeout(tar, timeout_seconds=10)  # 10s timeout
                member = tar.next()
                if member is None:
                    logger.info("Tar reading completed or timed out, exiting loop")
                    break
                elif member.isfile():
                    # .caption or .img_content
                    rgb_type = _is_rgb(member.name)
                    tiff_type = _is_tiff(member.name)
                    is_img = rgb_type or tiff_type
                    is_caption = member.name.lower().endswith((".txt", ".caption"))

                    # not caption or img, skip
                    if not (is_caption or is_img):
                        continue

                    with tar.extractfile(member) as f:
                        d = f.read()
                        name = Path(member.name).stem.split(".")[0]

                        data["name"] = name
                        if is_img:
                            data["decode"] = "pil" if rgb_type else "tiff"
                        if has_caption:
                            if _prev_name is None:
                                _prev_name = name
                            else:
                                # pairs of caption and image
                                if _prev_name != name:
                                    logger.warning(
                                        f"Caption and image name mismatch: {_prev_name} vs {name}"
                                    )
                                    # clear the data since is not paired
                                    data.clear()
                                    _prev_name = None
                                    continue
                                else:
                                    # set to None and wait for the next pair
                                    _prev_name = None

                        if is_caption:
                            data["caption"] = d
                        elif is_img:
                            if decode:
                                with BytesIO(d) as b:
                                    data["img"] = tifffile.imread(b)
                            else:
                                data["img"] = d

                        if has_caption and "caption" in data and "img" in data:
                            q.put(data.copy())
                            data.clear()
                            _prev_name = None
                        elif has_caption:
                            # do nothing, wait for the img or caption
                            pass
                        elif len(data) != 0:
                            # only image
                            q.put(data.copy())
                            data.clear()
                        else:
                            logger.warning(f"{data=}, code should not reach here")
                            data = {}
                            continue
            except tarfile.ReadError:
                member_name = member.name if member else "unknown"
                logger.warning(
                    f"Failed to read {member_name} with tarfile, trying with extract_member_from_tar"
                )
            except tarfile.TarError as e:
                logger.info(f"Tar read done: {e}")
            except Exception as e:
                member_name = member.name if member else "unknown"
                logger.warning(f"Failed to read {member_name} with tarfile: {e}")
                # pass
        tar.close()
        logger.info("Closed tar file")
        q.put("DONE")

    return inner


def tar_conditions_read_fn(
    tar_file: str,
):
    tar = tarfile.TarFile(tar_file, "r")

    condition_types = ["hed", "segmentation", "sketch", "mlsd"]

    def inner(q: Queue):
        round_i = 0
        data = {}
        while True:
            member = None
            try:
                # member = tar_next_with_timeout(tar, timeout_seconds=10)  # 10s timeout
                member = tar.next()
                if member is None:
                    logger.warning("None member")
                    break

                if member.isfile():
                    with tar.extractfile(member) as f:
                        d = f.read()
                        name, cond_type = Path(member.name).stem.split(".")

                        data["name"] = name
                        data[cond_type] = d

                        round_i = round_i + 1

                        if round_i == len(condition_types):  # name in
                            _failed = False
                            round_i = 0
                            for c in condition_types:
                                if c not in data:
                                    logger.warning(
                                        f"Missing condition {c} for {name}, {data.keys()=}, skipping"
                                    )
                                    data = {}
                                    round_i = 0
                                    _failed = True
                            if _failed:
                                continue

                            q.put(data.copy())
                            data.clear()
            except tarfile.ReadError:
                member_name = member.name if member else "unknown"
                logger.warning(
                    f"Failed to read {member_name} with tarfile, trying with extract_member_from_tar"
                )
            except tarfile.TarError as e:
                logger.info(f"Tar read done: {e}")
            except Exception as e:
                member_name = member.name if member else "unknown"
                logger.warning(f"Failed to read {member_name} with tarfile: {e}")
                # pass
        tar.close()
        logger.info("Closed tar file")
        q.put("DONE")

    return inner


def mds_file_writter(
    tar_file: str,
    out_root_dir: str | Path,
    has_caption=False,
    use_tqdm=False,
    decode=False,
    convert_type: str = "image",
):
    logger_ = logger.bind(pid=os.getpid())
    out_root_dir = Path(out_root_dir)
    out_root_dir.mkdir(parents=True, exist_ok=True)
    logger_.info(f"Ready to rewrite tar file {tar_file} into MDS file")

    if convert_type == "condition":
        read_fn = tar_conditions_read_fn(tar_file)
    elif convert_type == "image":
        read_fn = tar_pure_image_read_fn(
            tar_file, has_caption=has_caption, decode=decode
        )
    else:
        raise ValueError(f"Unknown convert_type {convert_type}")

    q = Queue(maxsize=100)
    producer = Process(target=read_fn, args=(q,))
    producer.start()

    # writter
    if convert_type == "condition":
        columns = {
            "hed": "png",
            "segmentation": "png",
            "sketch": "png",
            "mlsd": "png",
            "name": "str",
        }
    elif convert_type == "image":
        columns = {
            "img": "jpeg",  # tiff bytes
            # "caption": "str",
            "name": "str",
            "decode": "str",
        }

    writter = MDSWriter(
        columns=columns,
        out=str(out_root_dir),
        compression=None,
        size_limit="256mb",
    )

    # get sample from queue
    tbar = tqdm() if use_tqdm else None
    while True:
        item = q.get()
        if item == "DONE":
            logger_.info("All items processed, exiting writer")
            break

        writter.write(item)
        if use_tqdm:
            tbar.update(1)
            tbar.set_description("Written {:<40}".format(item["name"]))
    producer.join()
    writter.finish()
    logger_.info(f"MDS written to {out_root_dir}")


def parallel_mds_file_writter(
    orig_tar_files: list[str],
    save_dir_name: str = "MDS_hyper_images",
    sub_mds_name: str = "image",
    convert_type: str = "image",
    *,
    max_workers=6,
):
    def init_worker():
        logger.info("Worker initialized, pid: {}".format(os.getpid()))

    with ProcessPoolExecutor(
        initializer=init_worker, max_workers=max_workers
    ) as executor:
        futures = []
        for i, tar_file in enumerate(orig_tar_files):
            out_dir = (
                Path(tar_file).parents[1] / save_dir_name / f"{sub_mds_name}_{i:02d}"
            )
            future = executor.submit(
                mds_file_writter,
                tar_file,
                out_dir,
                has_caption=False,
                use_tqdm=True if i % max_workers == 0 else False,
                convert_type=convert_type,
            )
            futures.append(future)

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error in processing: {e}")


class ImageStreamDataset(StreamingDataset):
    def __init__(
        self,
        norm=True,
        permute=False,
        *args,
        **kwargs,
    ):
        self.norm = norm
        self.permute = permute
        super().__init__(*args, **kwargs)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # let norm_img do normalization
            ]
        )
        self.augmentation = None

    def __getitem__(self, idx):
        d = super().__getitem__(idx)

        # Image transformation and augmentation
        d["img"] = self.transform(d["img"])
        if self.norm:
            d = norm_img(d, permute=self.permute)

        return d


class ConditionsDataset(StreamingDataset):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.transform = transforms.Compose(
            [
                lambda img: img.convert("RGB") if isinstance(img, Image.Image) else img,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __getitem__(self, idx):
        d = super().__getitem__(idx)
        # process conditions if needed

        # to neg 1 to 1
        condition_types = ["hed", "segmentation", "sketch", "mlsd"]
        for c in condition_types:
            d[c] = self.transform(d[c])

        return d


class GenerativeDataset(torch.utils.data.Dataset):
    def __init__(self, img_ds: StreamingDataset, condition_ds: StreamingDataset):
        super().__init__()
        self.img_ds = img_ds
        self.condition_ds = condition_ds

        assert img_ds.size == condition_ds.size, (
            "Image and condition dataset size mismatch"
        )
        assert img_ds.batch_size == condition_ds.batch_size, (
            "Image and condition dataset batch size mismatch"
        )

        self.batch_size = img_ds.batch_size
        self.epoch_size = img_ds.epoch_size
        self.size = img_ds.size

    def __len__(self):
        return self.img_ds.size

    def __getitem__(self, idx):
        img_data = self.img_ds[idx]
        condition_data = self.condition_ds[idx]
        assert img_data["name"].split(".")[0] == condition_data["name"].split(".")[0], (
            f"Name mismatch: {img_data['name']} vs {condition_data['name']}"
        )

        data = {}
        data = data | img_data | condition_data
        return data

    # TODO: implem the stateful dataset


def test_img_stream_dataset():
    path = "data2/RemoteSAM270k/MDS_hyper_images/image_00"
    img_ds = ImageStreamDataset(True, True, local=path, batch_size=16, shuffle=False)
    print(len(img_ds))
    dl = StreamingDataLoader(
        img_ds, batch_size=16, num_workers=4, persistent_workers=True
    )
    for i, sample in enumerate(dl):
        print(f"Batch {i}: keys={sample.keys()}, img_shape={sample['img'].shape}")


def test_stream_dataset():
    cond_stream = Stream(local="data/BigEarthNet_S2/MDS_hyper_conditions")
    img_stream = Stream(local="data/BigEarthNet_S2/MDS_hyper_images")

    img_ds = ImageStreamDataset(streams=[img_stream], batch_size=16, shuffle=False)
    cond_ds = ConditionsDataset(streams=[cond_stream], batch_size=16, shuffle=False)
    generative_ds = GenerativeDataset(img_ds, cond_ds)

    dl = StreamingDataLoader(
        generative_ds, batch_size=16, num_workers=12, persistent_workers=True
    )
    logger.info(f"Streaming dataset prepared. Length {len(generative_ds)}")

    # for sample in ds:
    #     print(sample["name"], sample["img"].shape)

    file_list = []

    for sample in tqdm(iter(dl), total=len(generative_ds) // generative_ds.batch_size):
        # print(sample["name"], sample["img"].shape)
        pass
        # file_list.extend(sample["name"])
        # if len(file_list) % 200 == 0:
        #     np.savetxt(
        #         "data/BigEarthNet_S2/MDS_hyper_images/file_list.txt",
        #         file_list,
        #         fmt="%s",
        #     )
        #     logger.info(f"Saved {len(file_list)} file names so far.")


if __name__ == "__main__":
    from braceexpand import braceexpand

    # * BigEarthNet S2 MDS preparation
    # orig_tar_files = (
    #     "data/BigEarthNet_S2/hyper_images/BigEarthNet_data_{0000..0006}.tar"
    # )
    # orig_tar_files = list(braceexpand(orig_tar_files))
    # root_mds_dir = "data/BigEarthNet_S2/MDS_hyper_images"

    # orig_tar_files = list(
    #     braceexpand("data/BigEarthNet_S2/conditions/BigEarthNet_data_{0000..0006}.tar")
    # )
    # root_mds_dir = "data/BigEarthNet_S2/MDS_conditions"
    # parallel_mds_file_writter(
    #     orig_tar_files,
    #     save_dir_name="MDS_hyper_conditions",
    #     sub_mds_name="condition",
    #     convert_type="condition",
    #     max_workers=6,
    # )
    # merge_index(root_mds_dir, root_mds_dir)

    # * RemoteSAM270k
    # tar_files = ["data2/RemoteSAM270k/RemoteSAM-270K/RemoteSAM270K.tar"]
    # parallel_mds_file_writter(
    #     tar_files,
    #     save_dir_name="MDS_hyper_images",
    #     sub_mds_name="image",
    #     convert_type="image",
    #     max_workers=1,
    # )

    # # debug
    # tar_file = orig_tar_files[0]
    # out_dir = Path(root_mds_dir) / f"image_00_ndarray"
    # mds_file_writter(
    #     tar_file,
    #     out_dir,
    #     has_caption=False,
    #     use_tqdm=True,
    #     decode=True,
    # )

    # *   test streaming dataset
    # test_stream_dataset()
    test_img_stream_dataset()
    # exit(0)

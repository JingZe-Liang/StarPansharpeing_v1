import functools
import queue
import signal
import tarfile
import threading
from io import BytesIO
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any, cast

import litdata as ld
import numpy as np
import tifffile
import torch
from braceexpand import braceexpand
from litdata import (
    CombinedStreamingDataset,
    ParallelStreamingDataset,
    StreamingDataLoader,
    StreamingDataset,
    StreamingRawDataset,
    optimize,
)
from litdata.processing.data_processor import ALL_DONE
from litdata.raw.indexer import FileMetadata
from litdata.streaming.serializers import NumpySerializer
from litdata.streaming.writer import BinaryWriter
from loguru import logger
from natsort import natsorted
from rasterio.io import MemoryFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing_extensions import Generator

from src.data.augmentations import hyper_transform
from src.data.codecs import img_decode_io, tiff_codec_io, tiff_decode_io
from src.data.tar_utils import extract_member_from_tar, read_tar_filenames_safe
from src.data.utils import large_image_resizer_clipper, norm_img
from src.utilities.logging import configure_logger, set_logger_file

# Global variable to track the current producer process
_current_producer: Process | None = None


def signal_handler(signum: int, frame) -> None:
    """Handle SIGINT (Ctrl+C) by terminating the producer process."""
    global _current_producer
    logger.info("Received interrupt signal, terminating producer process...")
    if _current_producer is not None and _current_producer.is_alive():
        logger.info(f"Terminating producer process {_current_producer.pid}")
        _current_producer.terminate()
        _current_producer.join(timeout=5)  # Wait up to 5 seconds for graceful shutdown
        if _current_producer.is_alive():
            logger.warning("Producer process did not terminate gracefully, forcing kill")
            _current_producer.kill()
            _current_producer.join()
    logger.info("Producer process terminated. Exiting...")
    exit(0)


def tar_next_with_timeout(tar_obj: tarfile.TarFile, timeout_seconds: int = 30):
    result_queue = queue.Queue()

    def worker():
        try:
            result = tar_obj.next()
            result_queue.put(("success", result))
        except Exception as e:
            result_queue.put(("error", e))

    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()

    try:
        status, result = result_queue.get(timeout=timeout_seconds)
        if status == "success":
            return result
        else:
            logger.warning(f"tar.next() failed with error: {result}")
            return None
    except queue.Empty:
        logger.warning(f"tar.next() timed out after {timeout_seconds} seconds, assuming end of file")
        return None


def read_files():
    paths = list(Path("/home/user/zihancao/Project/hyperspectral-1d-tokenizer/data/BigEarthNet_S2/tmp").glob("*"))
    output_dir = "/home/user/zihancao/Project/hyperspectral-1d-tokenizer/data/BigEarthNet_S2/litdata_hyper_images_2"
    print(f"Found {len(paths)} TiFF files")
    return paths, output_dir


def read_tiff(path):
    return tifffile.imread(path)


def _is_tiff(name: str):
    return name.lower().endswith((".tif", ".tiff"))


def _is_rgb(name: str):
    return name.lower().endswith((".jpg", ".jpeg", ".png", ".img_content", ".npy"))


def save_data(sink: Any, data: dict) -> None:
    # Use duck typing to support both Queue-like and BinaryWriter-like sinks
    if hasattr(sink, "put"):
        sink.put(data.copy())
        data.clear()
        return
    if hasattr(sink, "add_item"):
        sink.add_item(data.copy())
        data.clear()
        return
    raise ValueError(f"Unsupported sink type: {type(sink)}")


def tar_pure_image_read_fn(
    tar_file: str,
    has_caption=False,
    discard_caption=False,
    tiff_re_encode=False,
):
    tar = tarfile.TarFile(tar_file, "r")

    np_serilizer = NumpySerializer()

    def _npy_to_tiff(bin: bytes):
        # arr = np.load(bin)
        # arr = np_serilizer.deserialize(bin)
        with BytesIO(bin) as f:
            arr = np.load(f)
        tiff_bytes = tiff_codec_io(
            arr,
            compression="jpeg2000",
            compression_args={
                "reversible": False,
                "level": 90,
            },
        )
        return tiff_bytes

    def _img_caption_in(data):
        return "caption" in data and "img" in data

    def _keep_seq(data):
        data_ = {}
        match len(list(data.keys())):
            case 4:
                for k in ["__key__", "caption", "img", "rgb"]:
                    data_[k] = data[k]
            case 3:
                for k in ["__key__", "caption", "img"]:
                    data_[k] = data[k]
            case 2:
                for k in ["__key__", "img"]:
                    data_[k] = data[k]
            case _:
                raise ValueError(f"Unsupported data keys: {data.keys()}")
        return data_

    def inner(q: "Queue | BinaryWriter"):
        # only extract the image files
        nonlocal has_caption, discard_caption

        not_paired_data = []
        data = {}
        _name_set = set()
        is_queue = hasattr(q, "put")

        _re_encoded_fn = _npy_to_tiff if tiff_re_encode else lambda x: x
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
                    is_caption = member.name.lower().endswith((".jsonl", ".txt", ".caption"))

                    # not caption or img, skip
                    if not (is_caption or is_img):
                        continue

                    extracted_file = tar.extractfile(member)
                    if extracted_file is not None:
                        with extracted_file as f:
                            d = f.read()
                            name, mod_type = Path(member.name).stem.split(".")

                        # tmp code
                        # if name.startswith("label"):
                        #     logger.debug(f"Skip label non-multispectral image: {name}")
                        #     continue

                        if name in _name_set:
                            logger.warning(f"Duplicated name: {name}, skip this one.")
                            continue
                        else:
                            _name_set.add(name)

                        data["__key__"] = name
                        if is_caption:
                            data["caption"] = d
                        else:
                            data["img"] = _re_encoded_fn(d)
                        # elif mod_type == "hsi":
                        #     data["img"] = d
                        # elif mod_type == "rgb":
                        #     data["img_rgb"] = d

                        if has_caption and not discard_caption:
                            if len(not_paired_data) == 0:
                                not_paired_data.append(data)
                            else:
                                # pairs of caption and image
                                if len(not_paired_data) > 1:
                                    logger.warning(
                                        f"There are more than one not paired data: n={len(not_paired_data)}."
                                    )

                                for i in range(len(not_paired_data)):
                                    data_ = not_paired_data[i]
                                    if data_["__key__"] == name:
                                        # is paired
                                        data_.update(data)
                                        if not _img_caption_in(data_):
                                            logger.warning(
                                                f"Paired data incomplete: name={data_['name']}, keys={data_.keys()}"
                                            )
                                        else:
                                            data = _keep_seq(data)
                                            # q.put(data_.copy())
                                            save_data(q, data)
                                            not_paired_data.pop(i)
                                            break

                                        # 2.
                                        # ks = ["__key__", "img", "img_rgb", "caption"]
                                        # is_full = all(k in data for k in ks)
                                        # if is_full:
                                        #     data = {k: data[k] for k in ks}
                                        #     save_data(q, data)
                                        #     not_paired_data.pop(i)
                                        #     break

                                    else:
                                        # is not paired
                                        logger.warning(
                                            f"Unpaired data found: want to matched name={data_['__key__']}, current name={name}"
                                        )

                                        # keep still in list
                                        not_paired_data.append(data)
                                        logger.warning(f"Now has not paired data len: {len(not_paired_data)}")

                                    if len(not_paired_data) > 5:
                                        logger.warning(f"Too many unpaired data, only keep the last 5 items.")
                                        not_paired_data = not_paired_data[-5:]
                        elif has_caption and discard_caption:
                            if is_caption:
                                pass
                            else:
                                # img
                                data["img"] = _re_encoded_fn(d)
                                data = _keep_seq(data)
                                save_data(q, data)
                        else:
                            # only has image
                            if is_img:
                                data["img"] = _re_encoded_fn(d)
                                data = _keep_seq(data)
                                save_data(q, data)
            except tarfile.ReadError as e:
                member_name = member.name if member else "unknown"
                logger.warning(
                    f"Failed to read {member_name} with tarfile, trying with extract_member_from_tar. Error: {e}"
                )
                break
            except tarfile.TarError as e:
                logger.info(f"Tar read done: {e}")
                break
            except Exception as e:
                member_name = member.name if member else "unknown"
                logger.warning(f"Failed to read {member_name} with tarfile: {e}")
                break

        tar.close()
        logger.info("Closed tar file")

        if is_queue:
            cast(Queue, q).put(ALL_DONE)

    return inner


def tar_conditions_read_fn(tar_file: str):
    tar = tarfile.TarFile(tar_file, "r")

    condition_types = ["hed", "segmentation", "sketch", "mlsd"]

    def inner(q: Queue):
        round_i = 0
        data = {}
        while True:
            member = None
            _prev_name = None
            try:
                # member = tar_next_with_timeout(tar, timeout_seconds=10)  # 10s timeout
                member = tar.next()
                if member is None:
                    logger.warning("None member")
                    break

                if member.isfile():
                    extracted_file = tar.extractfile(member)
                    if extracted_file is not None:
                        with extracted_file as f:
                            d = f.read()
                            name, cond_type = Path(member.name).stem.split(".")

                        if cond_type == "rgb":
                            # skip this
                            continue

                        # name
                        data["__key__"] = name
                        if _prev_name is None:
                            _prev_name = name
                        elif _prev_name != name:
                            logger.warning(f"Name mismatch: {_prev_name} != {name}, {data.keys()=}, skipping")
                            data = {}
                            round_i = 0
                            _prev_name = name

                        round_i = round_i + 1
                        # save to dict data
                        data[cond_type] = d

                        if round_i == len(condition_types):  # name in
                            _failed = False
                            round_i = 0
                            for c in condition_types:
                                if c not in data:
                                    logger.warning(f"Missing condition {c} for {name}, {data.keys()=}, skipping")
                                    data = {}
                                    round_i = 0
                                    _prev_name = None
                                    _failed = True
                            if _failed:
                                continue
                            # keep the same order
                            data = {k: data[k] for k in (["__key__"] + condition_types)}
                            save_data(q, data)
                            _prev_name = None
            except tarfile.ReadError:
                member_name = member.name if member else "unknown"
                logger.warning(f"Failed to read {member_name} with tarfile, trying with extract_member_from_tar")
                break
            except tarfile.TarError as e:
                logger.info(f"Tar read done: {e}")
                break
            except Exception as e:
                member_name = member.name if member else "unknown"
                logger.warning(f"Failed to read {member_name} with tarfile: {e}")
                break
                # pass
        tar.close()
        logger.info("Closed tar file")

        if hasattr(q, "put"):
            q.put(ALL_DONE)

    return inner


def tar_captions_embeddings_read_fn(tar_file: str, has_embed=True, not_keep_embed=True):
    # remove embeddings
    cond_types = ["caption", "features"]
    tar = tarfile.TarFile(tar_file, "r")

    def _keep_seq(data):
        data_ = {}
        data_["__key__"] = data["__key__"]
        for k in cond_types:
            if k in data:
                data_[k] = data[k]
        return data_

    def inner(q: Queue):
        data = {}
        while True:
            member = None
            try:
                member = tar.next()
                if member is None:
                    logger.info("Tar reading completed, exiting loop")
                    break
                elif member.isfile():
                    extracted_file = tar.extractfile(member)
                    if extracted_file is not None:
                        with extracted_file as f:
                            d = f.read()
                            name, cond_type = Path(member.name).stem.split(".")

                        data["__key__"] = name
                        if cond_type == "caption":
                            data["caption"] = d
                        elif cond_type == "features" and has_embed:
                            if not_keep_embed:
                                continue
                            else:
                                data["features"] = d
                        else:
                            logger.warning(f"Found unknown cond_type: {cond_type}, tar member name {member.name}, skip")

                        if "caption" in data:
                            data = _keep_seq(data)
                            save_data(q, data)

            except tarfile.TarError as e:
                logger.info(f"Tar read done: {e}")
                break
            except Exception as e:
                member_name = member.name if member else "unknown"
                logger.warning(f"Failed to read {member_name} with tarfile: {e}")
                break
        tar.close()
        logger.info("Closed tar file")
        if hasattr(q, "put"):  # is queue
            q.put(ALL_DONE)

    return inner


def path_read_fn(index: int, paths: list[str]):
    failed_ = False
    tiff_data: bytes | None = None

    path = paths[index]
    name = Path(path).name
    try:
        # tiff_data = read_tiff_tifffile(path)
        if Path(path).suffix.lower() not in (".tif", ".tiff"):
            raise ValueError(f"Unsupported file format: {Path(path).suffix}")
        with open(path, "rb") as f:
            tiff_data = f.read()

    except Exception as e:
        logger.warning(f"Failed to read {path} with tifffile: {e}")
        failed_ = True

    if not failed_ and tiff_data is not None:
        data = {"__key__": name, "img": tiff_data}
        yield data
    else:
        pass


def _data_producer_from_paths(paths: list[str]):
    for path in paths:
        failed_ = False
        tiff_data: bytes | None = None
        name = Path(path).name
        try:
            # tiff_data = read_tiff_tifffile(path)
            if Path(path).suffix.lower() not in (".tif", ".tiff"):
                raise ValueError(f"Unsupported file format: {Path(path).suffix}")
            with open(path, "rb") as f:
                tiff_data = f.read()

        except Exception as e:
            logger.warning(f"Failed to read {path} with tifffile: {e}")
            failed_ = True

        if not failed_ and tiff_data is not None:
            data = {"__key__": name, "img": tiff_data}
            yield data
        else:
            pass


def queue_data_producer(producer_fn: Generator):
    def inner(q: Queue):
        for data in producer_fn:
            q.put(data)
        q.put(ALL_DONE)

    return inner


def demo_identity(x):
    return x


# --- Process/Queue shutdown utilities ---
def _shutdown_producer(producer: Process, q: "Queue | None" = None, timeout: float = 10.0) -> None:
    """Finalize a producer Process and optionally release its Queue.

    Always attempt to join to avoid zombie processes. If the process doesn't
    exit in time, terminate then kill as a last resort. Finally, close and
    join the queue thread to release resources.
    """
    if producer is None:
        return

    # First, try to join gracefully
    if producer.is_alive():
        logger.info(f"Main process waiting for producer {producer.pid} to finish...")
        producer.join(timeout=timeout)

    # Escalate if still alive
    if producer.is_alive():
        logger.warning(f"Producer {producer.pid} did not finish gracefully, terminating...")
        producer.terminate()
        producer.join(timeout=max(0.0, timeout / 2))

    if producer.is_alive():
        logger.warning(f"Producer {producer.pid} still alive, killing...")
        producer.kill()
        producer.join()

    # Ensure the process is fully reaped
    if not producer.is_alive():
        # join() returns immediately if already finished
        producer.join(timeout=0.1)
        logger.info(f"Producer process {producer.pid} finalized.")

    # Close the Queue from the main process side to cleanup resources
    if q is not None:
        # Close and join the queue thread to avoid hanging on interpreter exit
        q.close()
        # join_thread is a no-op if no feeder thread was started in this process
        q.join_thread()


# Raw Dataset


class ImageDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.files = list(self.data_dir.glob("*.tif"))
        print(f"Found {len(self.files)} TIFF files in {data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        if idx >= len(self.files):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.files)}")

        path = self.files[idx]

        # Read the TIFF data
        data = read_tiff(path)

        if data is None:
            raise ValueError(f"Failed to read data from {path}")

        name = path.name
        data_tensor = torch.as_tensor(data)

        return {"__key__": name, "img": data_tensor}


def test_streaming_raw_dataset():
    """Test the image dataset with DataLoader."""
    ds = ImageDataset("data/BigEarthNet_S2/LitData_hyper_images")

    print(f"Dataset length: {len(ds)}")

    # Test individual items first
    print("\nTesting individual items:")
    for i in range(min(3, len(ds))):
        try:
            item = ds[i]
            print(f"Item {i}: keys={item.keys()}, img_shape={item['img'].shape}")
        except Exception as e:
            print(f"Error getting item {i}: {e}")

    # Simple collate function
    def custom_collate(batch):
        """Collate function for the dataset."""
        names = [item["__key__"] for item in batch]
        imgs = torch.stack([item["img"] for item in batch])
        return {"__key__": names, "img": imgs}

    print("\nTesting DataLoader:")
    dl = DataLoader(
        ds,
        batch_size=8,  # Smaller batch size for testing
        num_workers=0,
        shuffle=False,
        collate_fn=custom_collate,
    )

    for i, sample in enumerate(dl):
        print(f"Batch {i}: keys={sample.keys()}, img_shape={sample['img'].shape}")


def prepare_from_tar(
    tar_path: str | list[str],
    output_path: str,
    dataset_type: str = "image",
    has_caption=False,
    tiff_re_encode=False,
    use_queue=True,
) -> None:
    """Prepare data from tar files with proper signal handling for graceful shutdown."""
    global _current_producer

    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    if isinstance(tar_path, str) and Path(tar_path).is_dir():
        tar_paths = natsorted([str(p) for p in Path(tar_path).rglob("*.tar")])
    elif isinstance(tar_path, str):
        tar_paths = [tar_path]
    else:
        tar_paths = tar_path

    for single_tar_path in tar_paths:
        logger.info(f"Processing tar file: {single_tar_path}")

        chunk_kwargs = dict(chunk_bytes="512MB")
        if dataset_type == "image":
            q_data_producer_ = tar_pure_image_read_fn(
                single_tar_path,
                has_caption=has_caption,
                tiff_re_encode=tiff_re_encode,
            )
        elif dataset_type == "condition":
            q_data_producer_ = tar_conditions_read_fn(single_tar_path)
        elif dataset_type == "caption":
            q_data_producer_ = tar_captions_embeddings_read_fn(single_tar_path)
            chunk_kwargs = dict(chunk_size=1024)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        if not use_queue:
            # main process binary writer
            sink = BinaryWriter(
                output_path,
            )
            # TODO: add mode='append' logic
            raise NotImplementedError("Non-queue mode is not implemented yet.")
        else:
            q = Queue(maxsize=60)
            producer = Process(target=q_data_producer_, args=(q,))
            _current_producer = producer  # Track the current producer process
            producer.start()

            try:
                optimize(
                    fn=demo_identity,
                    queue=q,
                    output_dir=output_path,
                    num_workers=1,
                    mode="append",
                    keep_data_ordered=True,
                    start_method="fork",
                    **chunk_kwargs,
                )
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received, cleaning up...")
                _shutdown_producer(producer, q, timeout=10.0)
                logger.info("Cleanup completed. Exiting...")
                raise
            finally:
                _current_producer = None  # Clear the reference
            # Normal path: ensure child and queue are fully finalized
            _shutdown_producer(producer, q, timeout=10.0)


if __name__ == "__main__":
    # path = "data2/Fmow_rgb/hyper_images"
    # output_path = 'data2/Fmow_rgb/LitData_hyper_images'

    # path = 'data2/Multispectral-FMow-full/hyper_images_4bands'
    # output_path = 'data2/Multispectral-FMow-full/LitData_hyper_images_4bands'

    # path = "/home/user/zihancao/Project/hyperspectral-1d-tokenizer/data/TEOChatlas/train/GeoChat_Instruct_images1.tar"
    # output_path = "data/TEOChatlas/LitData_images"

    set_logger_file("tmp/convert_to_litdata_RGB_tar_Files.log")

    paths_dict: list[dict[str, object]] = [
        # {
        #     "tar_path": "data/RS5M/train",
        #     "output_path": "data/RS5M/LitData_images_train",
        #     "has_caption": True,
        # },
        # {
        #     "tar_path": "data/RS5M/val",
        #     "output_path": "data/RS5M/LitData_images_val",
        #     "has_caption": True,
        # },
        # {
        #     "tar_path": "data/miniFrance/hyper_images",
        #     "output_path": "data/miniFrance/LitData_hyper_images",
        # },
        # {
        #     "tar_path": "data/Disaterm3/hyper_images",
        #     "output_path": "data/Disaterm3/LitData_hyper_images",
        # },
        # {
        #     "tar_path": "data/BigEarthNet_S2/hyper_images",
        #     "output_path": "data/BigEarthNet_S2/LitData_hyper_images",
        # },
        # {
        #     "tar_path": "data/Fmow_rgb/hyper_images",
        #     "output_path": "data/Fmow_rgb/LitData_hyper_images",
        # },
        # {
        #     "tar_path": "data2/Multispectral-FMow-full/hyper_images_4bands",
        #     "output_path": "data2/Multispectral-FMow-full/LitData_hyper_images_4bands",
        # },
        # {
        #     "tar_path": "data2/Multispectral-FMow-full/hyper_images_8bands",
        #     "output_path": "data2/Multispectral-FMow-full/LitData_hyper_images_8bands",
        # },
        # {
        #     "tar_path": "data/GID-GF2/hyper_images",
        #     "output_path": "data/GID-GF2/LitData_hyper_images",
        # },
        # {
        #     "tar_path": "data/InriaAerialLabelingDataset/hyper_images",
        #     "output_path": "data/InriaAerialLabelingDataset/LitData_hyper_images",
        # },
        # {
        #     "tar_path": "data/TEOChatlas/train/GeoChat_Instruct_images1.tar",
        #     "output_path": "data/TEOChatlas/LitData_images_train",
        # },
        # {
        #     "tar_path": "data/TEOChatlas/train/GeoChat_Instruct_images2.tar",
        #     "output_path": "data/TEOChatlas/LitData_images_train",
        # },
        # {
        #     "tar_path": "data/TEOChatlas/train/TEOChatlas_images.tar",
        #     "output_path": "data/TEOChatlas/LitData_images_train",
        # },
        # {
        #     "tar_path": "data/TEOChatlas/eval/External_images.tar",
        #     "output_path": "data/TEOChatlas/LitData_images_eval",
        # },
        # {
        #     "tar_path": "data/TEOChatlas/eval/TEOChatlas_images.tar",
        #     "output_path": "data/TEOChatlas/LitData_images_eval",
        # },
        # {
        #     "tar_path": "data/SkyDiffusion/0000.tar",
        #     "output_path": "data/SkyDiffusion/LitData_images",
        # },
        # {
        #     "tar_path": [
        #         "data/RSCaptions/hyper_images/RSCaptionCollection-hyper_images-0000.tar",
        #         "data/RSCaptions/hyper_images/RSCaptionCollection-RSICD-0000.tar",
        #         "data/RSCaptions/hyper_images/RSCaptionCollection-RSITMD-0000.tar",
        #         "data/RSCaptions/hyper_images/RSCaptionCollection-Sydney_caption-0000.tar",
        #         "data/RSCaptions/hyper_images/RSCaptionCollection-UCM-0000.tar",
        #     ],
        #     "output_path": "data/RSCaptions/LitData_hyper_images",
        # },
        # {
        #     "tar_path": "data2/DCF_2020/hyper_images",
        #     "output_path": "data2/DCF_2020/LitData_hyper_images",
        # },
        # {
        #     "tar_path": "data2/DCF_2019/hyper_images",
        #     "output_path": "data2/DCF_2019/LitData_hyper_images",
        # },
        # {
        #     "tar_path": list(
        #         braceexpand(
        #             "data/HyperGlobal/hyper_images/HyperGlobal-EO1-bands-px_64_{0000..0011}.tar"
        #         )
        #     ),
        #     "output_path": "data/HyperGlobal/LitData_hyper_images_EO1",
        # },
        # {
        #     "tar_path": list(
        #         braceexpand(
        #             "data/HyperGlobal/hyper_images/HyperGlobal-GF5-bands-px_64_{0000..0004}.tar"
        #         )
        #     ),
        #     "output_path": "data/HyperGlobal/LitData_hyper_images_GF5",
        # },
        # {
        #     # 8 bands
        #     "tar_path": [
        #         "data/Multispectral-Spacenet-series/01_SN1_buildings_8band.tar",
        #         "data/Multispectral-Spacenet-series/00_SN1_buildings_8band.tar",
        #         "data/Multispectral-Spacenet-series/02_SN2_buildings_PS-MS.tar",
        #         "data/Multispectral-Spacenet-series/03_SN3_roads_PS-MS.tar",
        #         "data/Multispectral-Spacenet-series/05_SN5_roads_PS-MS.tar",
        #     ],
        #     "output_path": "data/Multispectral-Spacenet-series/LitData_hyper_images_8bands",
        # },
        # {
        #     # 3bands
        #     "tar_path": [
        #         "data/Multispectral-Spacenet-series/00_SN1_buildings_3band.tar",
        #         "data/Multispectral-Spacenet-series/03_SN2_buildings_PS-RGB.tar",
        #         "data/Multispectral-Spacenet-series/04_SN5_roads_PS-RGB.tar",
        #         "data/Multispectral-Spacenet-series/05_SN6_buildings_PS-RGB.tar",
        #     ],
        #     "output_path": "data/Multispectral-Spacenet-series/LitData_hyper_images_3bands",
        # },
        # {
        #     "tar_path": "data/RSCaptions/hyper_images",
        #     "output_path": "data/RSCaptions/LitData_hyper_images",
        # },
        # {
        #     "tar_path": "data/AerialVG/hyper_images/AerialVG-3_bands-RGB-0000.tar",
        #     "output_path": "data/AerialVG/LitData_hyper_images",
        # },
        # {
        #     "tar_path": "data/RefSegRS/hyper_images/RefSegRS_3_bands-px_512-RGB-jp2k-80-0000.tar",
        #     "output_path": "data/RefSegRS/LitData_hyper_images",
        # },
        # {
        #     "tar_path": "data/UDD/hyper_images/UDD-3_bands-px_512-RGB-jp2k-95-0000.tar",
        #     "output_path": "data/UDD/LitData_hyper_images",
        # },
        # {
        #     "tar_path": "data/VDD/hyper_images/VDD_3_bands-px_2k-RGB-0000.tar",
        #     "output_path": "data/VDD/LitData_hyper_images",
        # },
        # {
        #     "tar_path": "data/xView2/hyper_images/xView2-3_bands-px_1024-RGB-jp2k-80-0000.tar",
        #     "output_path": "data/xView2/LitData_hyper_images",
        # },
        # {
        #     # 202 bands
        #     "tar_path": "data/hyspecnet11k/hyper_images",
        #     "output_path": "data/hyspecnet11k/LitData_hyper_images",
        # },
        # {
        #     # 224 bands
        #     "tar_path": "data/DryadHyper/hyper_images",
        #     "output_path": "data/DryadHyper/LitData_hyper_images",
        # },
        # {
        #     # 368 bands
        #     "tar_path": "data/MDAS-HySpex/hyper_images",
        #     "output_path": "data/MDAS-HySpex/LitData_hyper_images",
        # },
        # {
        #     # 242 bands
        #     "tar_path": "data/MDAS-EeteS/hyper_images",
        #     "output_path": "data/MDAS-EeteS/LitData_hyper_images",
        # },
        # {
        #     # 369 channels
        #     "tar_path": "data/EarthView/hyper_images/neon",
        #     "output_path": "data/EarthView/LitData_neon",
        #     "has_caption": True,
        # },
        # {
        #     # Satellogic
        #     "tar_path": "data/EarthView/hyper_images/satellogic",
        #     "output_path": "data/EarthView/LitData_satellogic",
        #     "has_caption": True,
        # },
        # {
        #     'tar_path': 'data/ERA_UAV_Video_Dataset/hyper_images/ERA_UAV_Video_Dataset_key_frames_3_bands-px_512-RGB-jp2k-80-0000.tar',
        #     'output_path': 'data/ERA_UAV_Video_Dataset/LitData_hyper_images',
        # },
        # {
        #     # 32 bands
        #     "tar_path": "data/OHS/hyper_images",
        #     "output_path": "data/OHS/LitData_hyper_images",
        # },
        # {
        #     "tar_path": "data/CityBench-CityData/hyper_images",
        #     "output_path": "data/CityBench-CityData/LitData_hyper_images",
        # },
        # {
        #     "tar_path": "data/MMSeg_YREB/hyper_images",
        #     "output_path": "data/MMSeg_YREB/LitData_hyper_images",
        # },
        # {
        #     "tar_path": "data/LoveDA/hyper_images",
        #     "output_path": "data/LoveDA/LitData_hyper_images",
        # },
        # {
        #     "tar_path": "data/WorldView4/hyper_images/WorldView4-4_bands-px_256-MSI-0000.tar",
        #     "output_path": "data/WorldView4/LitData_hyper_images",
        # },
        # {
        #     "tar_path": "data/WorldView3/hyper_images/WorldView3-8_bands-px_256-MSI-0000.tar",
        #     "output_path": "data/WorldView3/LitData_hyper_images",
        # },
        # {
        #     "tar_path": "data/WorldView2/hyper_images/WorldView2-8_bands-px_256-MSI-0000.tar",
        #     "output_path": "data/WorldView2/LitData_hyper_images",
        # },
        # {
        #     "tar_path": "data/QuickBird/hyper_images/QuickBird-4_bands-px_256-MSI-0000.tar",
        #     "output_path": "data/QuickBird/LitData_hyper_images",
        # },
        # {
        #     "tar_path": "data/IKONOS/hyper_images/IKONOS-4_bands-px_256-MSI-0000.tar",
        #     "output_path": "data/IKONOS/LitData_hyper_images",
        # },
        # {
        #     "tar_path": "data/Gaofen1/hyper_images/Gaofen1-4_bands-px_256-MSI-0000.tar",
        #     "output_path": "data/Gaofen1/LitData_hyper_images",
        # },
        # {
        #     "tar_path": "data/uavid/hyper_images",
        #     "output_path": "data/uavid/LitData_hyper_images",
        # },
        # {
        #     "tar_path": "data2/TUM_128/hyper_images/",
        #     "output_path": "data2/TUM_128/LitData_hyper_images",
        # },
        # {
        #     "tar_path": "data/OpenEarthMap/hyper_images",
        #     "output_path": "data/OpenEarthMap/LitData_hyper_images",
        # },
        # {
        #     "tar_path": "data/MUSLI/hyper_images",
        #     "output_path": "data/MUSLI/LitData_hyper_images",
        # },
        # {
        #     "tar_path": "data/MMOT/hyper_images",
        #     "output_path": "data/MMOT/LitData_hyper_images",
        #     "has_caption": True,
        # },
        # {
        #     "tar_path": "data/Houston/hyper_images",
        #     "output_path": "data/Houston/LitData_hyper_images",
        # },
        # {
        #     "tar_path": "data/MDAS-Optical/hyper_images",
        #     "output_path": "data/MDAS-Optical/LitData_hyper_images",
        # },
        ## ************* Conditions **************** ##
        # {
        #     "tar_path": "data/LoveDA/conditions",
        #     "output_path": "data/LoveDA/LitData_conditions/",
        #     "dataset_type": "condition",
        # },
        # {
        #     "tar_path": "data/MMSeg_YREB/conditions",
        #     "output_path": "data/MMSeg_YREB/LitData_conditions/",
        #     "dataset_type": "condition",
        # },
        # {
        #     "tar_path": "data/MMOT/conditions",
        #     "output_path": "data/MMOT/LitData_conditions/",
        #     "dataset_type": "condition",
        # },
        # {
        #     "tar_path": "data/IKONOS/conditions",
        #     "output_path": "data/IKONOS/LitData_conditions/",
        #     "dataset_type": "condition",
        # },
        # {
        #     "tar_path": "data/Houston/conditions",
        #     "output_path": "data/Houston/LitData_conditions/",
        #     "dataset_type": "condition",
        # },
        # {
        #     "tar_path": "data/Gaofen1/conditions",
        #     "output_path": "data/Gaofen1/LitData_conditions/",
        #     "dataset_type": "condition",
        # },
        # {
        #     "tar_path": "data/WorldView4/conditions",
        #     "output_path": "data/WorldView4/LitData_conditions/",
        #     "dataset_type": "condition",
        # },
        # {
        #     "tar_path": "data/WorldView3/conditions",
        #     "output_path": "data/WorldView3/LitData_conditions/",
        #     "dataset_type": "condition",
        # },
        # {
        #     "tar_path": "data/WorldView2/conditions",
        #     "output_path": "data/WorldView2/LitData_conditions/",
        #     "dataset_type": "condition",
        # },
        # {
        #     "tar_path": "data/QuickBird/conditions",
        #     "output_path": "data/QuickBird/LitData_conditions/",
        #     "dataset_type": "condition",
        # },
        # {
        #     "tar_path": "data/OpenEarthMap/conditions",
        #     "output_path": "data/OpenEarthMap/LitData_conditions/",
        #     "dataset_type": "condition",
        # },
        # {
        #     "tar_path": "data2/DCF_2019/conditions",
        #     "output_path": "data2/DCF_2019/LitData_conditions/",
        #     "dataset_type": "condition",
        # },
        # {
        #     "tar_path": "data2/DCF_2020/conditions",
        #     "output_path": "data2/DCF_2020/LitData_conditions/",
        #     "dataset_type": "condition",
        # },
        # {
        #     "tar_path": "data/RefSegRS/conditions",
        #     "output_path": "data/RefSegRS/LitData_conditions/",
        #     "dataset_type": "condition",
        # },
        # {
        #     "tar_path": "data/MDAS-HySpex/conditions",
        #     "output_path": "data/MDAS-HySpex/LitData_conditions/",
        #     "dataset_type": "condition",
        # },
        # {
        #     "tar_path": "data/MDAS-EeteS/conditions",
        #     "output_path": "data/MDAS-EeteS/LitData_conditions/",
        #     "dataset_type": "condition",
        # },
        # {
        #     "tar_path": "data/GID-GF2/conditions",
        #     "output_path": "data/GID-GF2/LitData_conditions/",
        #     "dataset_type": "condition",
        # },
        # {
        #     "tar_path": "data/MDAS-Optical/condition",
        #     "output_path": "data/MDAS-Optical/LitData_conditions/",
        #     "dataset_type": "condition",
        # },
        # {
        #     "tar_path": [
        #         "data/RemoteSAM270k/RemoteSAM-270K/conditions/RemoteSAM270K.tar",
        #         "data/RemoteSAM270k/RemoteSAM-270K/conditions_2_resumed/RemoteSAM270K.tar",
        #     ],
        #     "dataset_type": "condition",
        # },
        # {
        #     "tar_path": "data/RSCaptions/conditions",
        #     "output_path": "data/RSCaptions/LitData_conditions/",
        #     "dataset_type": "condition",
        # },
        ## ************ Captions ************* ##
        # {
        #     "tar_path": "data/LoveDA/condition_captions",
        #     "output_path": "data/LoveDA/LitData_captions/",
        #     "dataset_type": "caption",
        # },
        # {
        #     "tar_path": "data/WorldView4/condition_captions",
        #     "output_path": "data/WorldView4/LitData_captions/",
        #     "dataset_type": "caption",
        # },
        # {
        #     "tar_path": "data/WorldView3/condition_captions",
        #     "output_path": "data/WorldView3/LitData_captions/",
        #     "dataset_type": "caption",
        # },
        # {
        #     "tar_path": "data/WorldView2/condition_captions",
        #     "output_path": "data/WorldView2/LitData_captions/",
        #     "dataset_type": "caption",
        # },
        # {
        #     "tar_path": "data/QuickBird/condition_captions",
        #     "output_path": "data/QuickBird/LitData_captions/",
        #     "dataset_type": "caption",
        # },
        # {
        #     "tar_path": "data/OpenEarthMap/condition_captions",
        #     "output_path": "data/OpenEarthMap/LitData_captions/",
        #     "dataset_type": "caption",
        # },
        # {
        #     "tar_path": "data/MUSLI/condition_captions",
        #     "output_path": "data/MUSLI/LitData_captions/",
        #     "dataset_type": "caption",
        # },
        # {
        #     "tar_path": "data/MMOT/condition_captions",
        #     "output_path": "data/MMOT/LitData_captions/",
        #     "dataset_type": "caption",
        # },
        # {
        #     "tar_path": "data/IKONOS/condition_captions",
        #     "output_path": "data/IKONOS/LitData_captions/",
        #     "dataset_type": "caption",
        # },
        # {
        #     "tar_path": "data/Houston/condition_captions",
        #     "output_path": "data/Houston/LitData_captions/",
        #     "dataset_type": "caption",
        # },
        # {
        #     "tar_path": "data/Gaofen1/condition_captions",
        #     "output_path": "data/Gaofen1/LitData_captions/",
        #     "dataset_type": "caption",
        # },
        # {
        #     "tar_path": "data2/DCF_2020/condition_captions",
        #     "output_path": "data2/DCF_2020/LitData_captions/",
        #     "dataset_type": "caption",
        # },
        # {
        #     "tar_path": "data/MDAS-HySpex/condition_captions",
        #     "output_path": "data/MDAS-HySpex/LitData_captions/",
        #     "dataset_type": "caption",
        # },
        # {
        #     "tar_path": "data/MDAS-EeteS/condition_captions",
        #     "output_path": "data/MDAS-EeteS/LitData_captions/",
        #     "dataset_type": "caption",
        # },
        # {
        #     "tar_path": "data/GID-GF2/condition_captions",
        #     "output_path": "data/GID-GF2/LitData_captions/",
        #     "dataset_type": "caption",
        # },
        # {
        #     "tar_path": "data/MDAS-Optical/condition_captions",
        #     "output_path": "data/MDAS-Optical/LitData_captions/",
        #     "dataset_type": "caption",
        # },
        # {
        #     "tar_path": "data/DryadHyper/condition_captions",
        #     "output_path": "data/DryadHyper/LitData_captions",
        #     "dataset_type": "caption",
        # },
        # {
        #     "tar_path": "data2/TUM_128/condition_captions",
        #     "output_path": "data2/TUM_128/LitData_captions",
        #     "dataset_type": "caption",
        # },
    ]
    for kwargs in paths_dict:
        try:
            logger.info(f"input preparing function args: {kwargs}")
            tar_path: str | list[str] = cast(str | list[str], kwargs["tar_path"])  # required
            output_path: str = cast(str, kwargs["output_path"])  # required
            dataset_type: str = cast(str, kwargs.get("dataset_type", "image"))
            has_caption: bool = cast(bool, kwargs.get("has_caption", False))
            use_queue: bool = cast(bool, kwargs.get("use_queue", True))

            prepare_from_tar(
                tar_path=tar_path,
                output_path=output_path,
                dataset_type=dataset_type,
                has_caption=has_caption,
                use_queue=use_queue,
                tiff_re_encode=False,
            )
            # break
        except Exception as e:
            logger.opt(depth=2).error(
                f"Error processing {kwargs['tar_path']}: {e}",
            )
            continue

from io import BytesIO
from multiprocessing import Process, Queue
from pathlib import Path

import tifffile
import torch
from litdata import StreamingDataLoader, StreamingDataset, optimize
from litdata.processing.data_processor import ALL_DONE
from rasterio.io import MemoryFile
from webdataset import WebDataset, WebLoader

from src.data.hyperspectral_loader import get_hyperspectral_dataloaders

paths = "data/BigEarthNet_S2/hyper_images/BigEarthNet_data_{0000..0006}.tar"
ds = get_hyperspectral_dataloaders(
    paths,
    batch_size=32,
    num_workers=2,
    to_neg_1_1=False,
    shuffle_size=-1,
    permute=False,
    force_no_norm=True,
)
dl = WebLoader(ds, batch_size=1, num_workers=0)
print("DataLoader prepared.")


def process_fn(dl):
    for data in dl:
        name = data["__url__"][0]

        # name_b = name.encode("utf-8")
        b = BytesIO()
        b.write(name.encode("utf-8"))
        b.seek(0)
        name_b = b.read()

        data_saved = {
            "img": data["img"][0],
            "name": name_b,
        }
        yield data_saved


def put_in_queue(q: Queue):
    for item in process_fn(dl):
        q.put(item)

    q.put(ALL_DONE)


def fn(index):
    return index


if __name__ == "__main__":
    q = Queue(maxsize=100)
    producer = Process(target=put_in_queue, args=(q,))
    producer.start()

    optimize(
        fn=fn,
        queue=q,
        output_dir="data/BigEarthNet_S2/litdata_hyperspectral",
        chunk_size=100,
        num_workers=1,
        compression="zstd",
        start_method="spawn",
    )

    producer.join()

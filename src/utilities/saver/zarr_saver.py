import zarr
import zarr.storage
import zarr.codecs
import numpy as np

from typing_extensions import Callable, Generator


def zarr_saver(
    arr_generator: Generator,
    save_path: str,
    compression: bool = False,
):
    compressors = zarr.codecs.BloscCodec(
        cname="zstd", clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle
    )
    store = zarr.storage.LocalStore(save_path)
    root = zarr.group(store=store).create_group("hyperspectral_images")
    for name, arr in arr_generator:
        assert isinstance(name, str), "name must be a string"
        group = root.create_group(name)
        z = group.create_array(
            "img",
            shape=arr.shape,
            chunks=arr.shape if not compression else (1000, 1000),
            dtype=arr.dtype,
            compressors=compressors if compression else None,
        )
        z[:] = arr
        print(f"save {name} to zarr file, shaped as {arr.shape}, dtype as {arr.dtype}")


def __example_arr_generator():
    for i in range(10):
        yield f"arr_{i}", np.random.rand(1000, 1000).astype(np.float32)


if __name__ == "__main__":
    zarr_saver(__example_arr_generator(), "./test.zarr")

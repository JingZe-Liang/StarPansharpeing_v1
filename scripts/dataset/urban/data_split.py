import io
import os
from typing import Any

import numpy as np
import scipy.io as sio
import tifffile
import webdataset as wds


def tiff_codec_io(
    img: np.ndarray,
    planarconfig: str | tifffile.PLANARCONFIG | None = None,
    photometric: str | tifffile.PHOTOMETRIC | None = None,
    compression: str = "zlib",
    compression_args: dict[str, Any] | None = None,
) -> bytes:
    """Encodes a NumPy array into TIFF formatted bytes.

    Args:
        img (np.ndarray): Input image array to be encoded. Can be 2D (grayscale) or 3D (color) array.
        planarconfig (str | tifffile.PLANARCONFIG | None, optional): TIFF planar configuration. Defaults to None.
            Can be 'CONTIG' (contiguous, interleaved channels) or 'SEPARATE' (planar, separate planes).
        photometric (str | tifffile.PHOTOMETRIC | None, optional): TIFF photometric interpretation. Defaults to None.
            Common values include 'MINISBLACK' (grayscale), 'RGB', 'PALETTE', 'MASK', etc.
        compression (str, optional): Compression method to use. Defaults to "zlib".
            Other options include 'lzw', 'jpeg', 'packbits', 'deflate', 'none', etc.
        compression_args (dict[str, Any] | None, optional): Additional arguments for the compression method.
            For example, {'level': 5} for zlib compression level.

    Returns:
        bytes: TIFF formatted bytes of the encoded image.

    Note:
        The function uses tifffile library to encode the image. The planarconfig and photometric
        parameters should match the image data structure to ensure correct encoding.
        For example, for an RGB image, photometric should be 'RGB' and planarconfig can be 'CONTIG'.
    """
    with io.BytesIO() as buffer:
        tifffile.imwrite(
            buffer,
            img,
            planarconfig=planarconfig,
            photometric=photometric,
            compression=compression,
            compressionargs=compression_args,
        )

        return buffer.getvalue()


def load_data(file):
    print("data file:", file)

    dataset = sio.loadmat(file)
    data, GT, abundance = dataset["img_3d"], dataset["endmember"], dataset["abundance"]
    init_em = dataset["init_em"]
    # (n_cols, n_rows, n_bands))
    data = data.transpose([1, 2, 0])
    # n_rows, n_cols, n_bands = data.shape
    # abundance = np.reshape(abundance, [abundance.shape[0], n_rows, n_cols])
    # GT = GT.transpose([1, 0])
    print("data.shape:", data.shape)
    print("init endmember.shape:", init_em.shape)  # channel,num_em
    print("endmember.shape:", GT.shape)
    print("abundance.shape:", abundance.shape)
    return data, GT, abundance, init_em


def whole2_train_and_test_data(img_size, img):
    H0, W0, C = img.shape
    if H0 < img_size:
        gap = img_size - H0
        mirror_img = img[(H0 - gap) : H0, :, :]
        img = np.concatenate([img, mirror_img], axis=0)
    if W0 < img_size:
        gap = img_size - W0
        mirror_img = img[:, (W0 - gap) : W0, :]
        img = np.concatenate([img, mirror_img], axis=1)
    H, W, C = img.shape

    num_H = H // img_size
    num_W = W // img_size
    sub_H = H % img_size
    sub_W = W % img_size
    if sub_H != 0:
        gap = (num_H + 1) * img_size - H
        mirror_img = img[(H - gap) : H, :, :]
        img = np.concatenate([img, mirror_img], axis=0)

    if sub_W != 0:
        gap = (num_W + 1) * img_size - W
        mirror_img = img[:, (W - gap) : W, :]
        img = np.concatenate([img, mirror_img], axis=1)
    H, W, C = img.shape
    print("padding img:", img.shape)

    num_H = H // img_size
    num_W = W // img_size

    sub_imgs = []
    for i in range(num_H):
        for j in range(num_W):
            z = img[i * img_size : (i + 1) * img_size, j * img_size : (j + 1) * img_size, :]
            sub_imgs.append(z)
    sub_imgs = np.array(sub_imgs)
    return sub_imgs, num_H, num_W


def to_webdataset():
    path = "data/UrbanUnmixing/Urban_188_em4_init.mat"
    data, endmember, abundance, init_em = load_data(path)

    writter = wds.TarWriter("data/UrbanUnmixing/Urban_188_em4_init.tar")
    sample = {
        "__key__": "Urban_188_em4_init",
        "img.npy": data.astype(np.float32),
        "endmember.npy": endmember.astype(np.float32),
        "abundance.npy": abundance.astype(np.float32),
        "init_em.npy": init_em.astype(np.float32),
    }
    writter.write(sample)
    writter.close()

    print("to webdataset done")


# * --- Test --- * #


def test_loading():
    path = "data/UrbanUnmixing/Urban_188_em4_init.mat"
    load_data(path)


def test_split_data():
    path = "data/UrbanUnmixing/Urban_188_em4_init.mat"
    data, GT, abundance, init_em = load_data(path)

    # Split train / test
    sub_imgs, num_H, num_W = whole2_train_and_test_data(img_size=64, img=data)
    print("split images:", sub_imgs.shape)
    print(f"{num_H=}, {num_W=}")


if __name__ == "__main__":
    # test_loading()
    # test_split_data()
    to_webdataset()

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile


def read_tiff_to_RGB(path: str | Path):
    path = Path(path)
    assert path.exists(), f"File {path} does not exist."
    # read
    img = tifffile.imread(path)
    img_rgb = img[..., [5, 4, 3]].astype(np.float32)
    img_rgb = np.clip(img_rgb, 0, None)
    # min-max normalization
    img_rgb = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min())
    img_rgb = (img_rgb * 255).astype(np.uint8)
    return img_rgb


if __name__ == "__main__":
    path = "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/data/HLS/tiff/HLS.S30.T49RGH.2025018T025949.v2.0.tif"
    img_rgb = read_tiff_to_RGB(path)
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()
    # save the image
    plt.imsave("test_HLS_image.png", img_rgb)

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import tifffile

names = [
    "10000.mat",
    "10001.mat",
    "10002.mat",
    "10003.mat",
]
rgb_bands = [32, 16, 4]
base_path = "/HardDisk/ZiHanCao/datasets/HyperGlobal-450k/tmp/GF5-part1"
for i, name in enumerate(names):
    path = f"{base_path}/{name}"
    mat_file = sio.loadmat(path)
    img = mat_file["img"]
    img_rgb = img[rgb_bands, :, :].transpose(1, 2, 0)
    img_rgb = img_rgb.astype("float32")
    img_rgb = np.clip(img_rgb, 0, None) / img_rgb.max()
    img_rgb = (img_rgb * 255.0).astype("uint8")
    # print(img_rgb.shape)

    ii = i // 2
    jj = i % 2
    plt.subplot(2, 2, i + 1)
    plt.imshow(img_rgb)

    # save img
    tifffile.imwrite(
        f"{name}.tif",
        img,
        compression="jpeg2000",
        compressionargs={"level": 80, "codecformat": "jp2", "reversible": False},
    )


plt.savefig(f"show.png", dpi=300)

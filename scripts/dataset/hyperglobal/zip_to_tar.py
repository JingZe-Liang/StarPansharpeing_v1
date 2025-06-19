# unzip mat file into tiff file into tar file
import io
import os
import zipfile

import webdataset as wds
from scipy.io import loadmat
from tqdm import tqdm

from src.data.codecs import tiff_codec_io


def zipfile_to_tarfile(zip_file_path, tar_file_writer: wds.ShardWriter, tbar=True):
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        if tbar:
            tbar = tqdm(zip_ref.infolist())
        else:
            tbar = zip_ref.infolist()
        for info in tbar:
            file = info.filename
            string = f"Processing {file}, "
            if file.endswith(".mat"):
                mat_file_bytes = zip_ref.read(file)
                try:
                    mat_data = loadmat(io.BytesIO(mat_file_bytes))
                except ImportError:
                    print(f"Raw bytes of {info.filename}: {len(mat_file_bytes)} bytes")
                    continue

                assert "img" in mat_data, f"Key 'img' not found in {file}"
                img = mat_data["img"]
                string += str(img.shape)

                if tbar:
                    tbar.set_description(string)
                else:
                    print(string)

                # save into io buffer
                # img: [c, h, w]
                img = img.transpose(1, 2, 0)
                # print(img.shape)
                tar_file_writer.write(
                    {
                        "__key__": "-".join(os.path.split(file)).replace(".mat", ""),
                        "img.tiff": tiff_codec_io(  # compression using jpeg2000
                            img,  # [h, w, c]
                            compression="jpeg2000",
                            compression_args={"reversible": False, "level": 80},
                        ),
                    }
                )


if __name__ == "__main__":
    from pathlib import Path

    from braceexpand import braceexpand

    shard_pattern = (
        "data/HyperGlobal/hyper_images/4/HyperGlobal450k-xx_bands-px_128_%04d.tar"
    )

    Path(shard_pattern).parent.mkdir(parents=True, exist_ok=True)
    sink = wds.ShardWriter(
        shard_pattern,
        maxsize=4 * 1024 * 1024 * 1024,  # 4G
    )
    # _zipfiles = [
    #     "/HardDisk/ZiHanCao/datasets/HyperGlobal-450k/EO1-part{4..5}.zip",
    #     "/HardDisk/ZiHanCao/datasets/HyperGlobal-450k/GF5-part{1..5}.zip",
    # ]
    # zipfiles = []
    # for zipf in _zipfiles:
    #     zipfiles.extend(braceexpand(zipf))

    zipfiles = ["/HardDisk/ZiHanCao/datasets/HyperGlobal-450k/EO1-part4.zip"]

    for zip_file_path in zipfiles:
        print(f"Processing {zip_file_path}")
        zipfile_to_tarfile(zip_file_path, sink)

    sink.close()

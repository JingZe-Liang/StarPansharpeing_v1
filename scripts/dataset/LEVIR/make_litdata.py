from pathlib import Path

import numpy as np
import PIL.Image as Image
from litdata import optimize

from src.data.codecs import rgb_codec_io


def process(
    paths: tuple[str, str, str],
) -> dict[str, str | bytes]:
    img1, img2, label = paths
    name = Path(img1).stem
    img1 = np.array(Image.open(str(img1)))
    img2 = np.array(Image.open(str(img2)))
    label = np.array(Image.open(str(label)).convert("L"))

    img1_bytes = rgb_codec_io(img1, quality=95)
    img2_bytes = rgb_codec_io(img2, quality=95)
    label_bytes = rgb_codec_io(label, "png")

    return {
        "__key__": name,
        "img1": img1_bytes,
        "img2": img2_bytes,
        "label": label_bytes,
    }


# all files
base_dir = Path("data/Downstreams/ChangeDetection/LEVIR-CD256")
As = list(Path(base_dir / "A").glob("*"))
Bs = list(Path(base_dir / "B").glob("*"))
labels = list(Path(base_dir / "label").glob("*"))
assert len(As) == len(Bs) == len(labels)
# sorted
As = sorted(As, key=lambda x: x.stem)
Bs = sorted(Bs, key=lambda x: x.stem)
labels = sorted(labels, key=lambda x: x.stem)

As_stem = [Path(x).stem for x in As]
Bs_stem = [Path(x).stem for x in Bs]
labels_stem = [Path(x).stem for x in labels]
assert set(As_stem) == set(Bs_stem) == set(labels_stem)
print("sorted done.")

print("Summary: \n   As: {}, Bs: {}, labels: {}".format(len(As), len(Bs), len(labels)))


# Save
save_dir = "data/Downstreams/ChangeDetection/LEVIR-CD256/LitData_CDImages"
Path(save_dir).mkdir(parents=True, exist_ok=True)
pair_zip = list(zip(As, Bs, labels))
optimize(
    process,
    pair_zip,
    save_dir,
    num_workers=0,
    start_method="fork",
    chunk_bytes="256Mb",
)

from safetensors import safe_open

import os
from pathlib import Path
from tqdm import tqdm


dataset_path = (
    "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/data/MUSLI_safetensors/safetensors"
)
paths = Path(dataset_path) / "*.safetensors"
out_one_safetensor = Path(dataset_path) / "MUSLI.safetensors"
save_f = safe_open(out_one_safetensor, "w")

for path in tqdm(list(paths.glob("*.safetensors"))):
    print(path)
    name = path.stem
    with safe_open(path, framework="pt", device="cpu") as f:
        tensor = f.get_tensor("img")
        save_f.set_tensor(name, tensor)

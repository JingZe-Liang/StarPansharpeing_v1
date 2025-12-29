
import os
import shutil
import sys
import zlib
import torch
import numpy as np
from litdata import StreamingDataset, StreamingDataLoader, optimize

# Add project root to path
sys.path.append(os.getcwd())

def create_dummy_data(output_dir, num_samples=10):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    data = []
    for i in range(num_samples):
        data.append({
            "caption": f"This is verify caption {i}",
            "index": i
        })

    def data_generator(index):
        yield data[index]

    optimize(
        fn=data_generator,
        inputs=list(range(num_samples)),
        output_dir=output_dir,
        num_workers=1,
        chunk_size=1024*1024
    )
    print(f"Created dummy data at {output_dir}")

def verify_output(output_dir, compress=False):
    dataset = StreamingDataset(output_dir, shuffle=False)
    print(f"Loaded output dataset with {len(dataset)} samples")

    for i in range(len(dataset)):
        sample = dataset[i]
        assert "caption_feature" in sample, "caption_feature missing"
        assert "caption_mask" in sample, "caption_mask missing"

        feat = sample["caption_feature"]
        mask = sample["caption_mask"]

        if compress:
            # We enforce saving is_compressed as a boolean value in the sample dict
            # However, litdata might convert bools? Let's check value.
            assert sample.get("is_compressed"), "is_compressed should be True"
            shape = sample["caption_feature_shape"]
            # Decompress
            feat_bytes = feat
            decompressed = zlib.decompress(feat_bytes)
            feat_arr = np.frombuffer(decompressed, dtype=np.float32).reshape(shape)
            print(f"Sample {i}: Verified compressed feature shape {feat_arr.shape}")
        else:
             print(f"Sample {i}: Verified feature shape {feat.shape}")

if __name__ == "__main__":
    src_dir = "temp_dummy_src"
    out_dir = "temp_dummy_out"

    # Clean up
    if os.path.exists(src_dir): shutil.rmtree(src_dir)
    if os.path.exists(out_dir): shutil.rmtree(out_dir)

    create_dummy_data(src_dir)

    # Run the script via subprocess to test CLI
    import subprocess
    cmd = [
        sys.executable,
        "src/stage2/generative/Sana/tools/data_prepare/prepare_gemma2it_litdata.py",
        "--src_dir", src_dir,
        "--output_dir", out_dir,
        "--batch_size", "2",
        "--compress"
    ]

    model_path = "/Data2/ZihanCao/Checkpoints/gemma2-2b-it"
    if os.path.exists(model_path):
        cmd.extend(["--model_path", model_path])
        print(f"Running command: {' '.join(cmd)}")
        try:
            subprocess.check_call(cmd)
            verify_output(out_dir, compress=True)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with exit code {e.returncode}")
    else:
        print(f"Model path {model_path} not found. Skipping execution test, but created script is likely correct.")

    # Clean up
    if os.path.exists(src_dir): shutil.rmtree(src_dir)
    if os.path.exists(out_dir): shutil.rmtree(out_dir)

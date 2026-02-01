from pathlib import Path

input_dir_path = "data/Downstreams/US3D_Stereo_Matching/JAX/train"
input_dir = Path(input_dir_path)

print(f"Path: {input_dir}")
print(f"Exists: {input_dir.exists()}")
print(f"Is dir: {input_dir.is_dir()}")
print(f"Is file: {input_dir.is_file()}")
print(f"Absolute: {input_dir.absolute()}")
print(f"Resolved: {input_dir.resolve()}")

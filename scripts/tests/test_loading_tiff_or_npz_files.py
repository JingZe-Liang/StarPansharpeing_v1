import time

import numpy as np
import tifffile
import torch
from safetensors.torch import load_file, save_file

# 生成测试数据
# data = np.random.randint(0, 255, (512,512,438), dtype=np.uint32)
# data_pt = torch.from_numpy(data).to(torch.uint32)

# 保存并计时
# t0 = time.time()
# np.savez_compressed('test.npz', data=data)
# print(f"NPZ 保存: {time.time()-t0:.3f}s")

# t0 = time.time()
# tifffile.imwrite('test.tif', data, compression='zlib')
# print(f"TIFF 保存: {time.time()-t0:.3f}s")

# t0 = time.time()
# save_file({'data': data_pt}, "test.safetensors")
# print(f"Safetensors 保存: {time.time()-t0:.3f}s")

# t0 = time.time()
# torch.save({'data': data_pt}, 'test.pt')
# print(f'Torch 保存: {time.time()-t0:.3f}s')

# t0 = time.time()
# np.save('test.npy', data)
# print(f"NPY 保存: {time.time()-t0:.3f}s")

# 加载计时
t0 = time.time()
np.load("test.npz")["data"]
print(f"NPZ 加载: {time.time() - t0:.3f}s")

t0 = time.time()
np.load("test.npy")
print(f"NPY 加载: {time.time() - t0:.3f}s")

t0 = time.time()
torch.load("test.pt")["data"]
print(f"Torch 加载: {time.time() - t0:.3f}s")

t0 = time.time()
tifffile.imread("test.tif")
print(f"TIFF 加载: {time.time() - t0:.3f}s")

t0 = time.time()

d = load_file("test.safetensors")
print(f"Safetensors 加载: {time.time() - t0:.3f}s")

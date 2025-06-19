import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import imagecodecs
import tifffile
from nvidia import nvimgcodec

path = "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/data/DCF_2019/hyper_images/tmp/JAX_156_025_018_LEFT_MSI_patch-3.img.tiff"
# path = "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/encoded_multispectral_img.tif"
img = tifffile.imread(path)
print(f"tifffile read image shape: {img.shape}")


# compress

# tifffile.imwrite(
#     "encoded_multispectral_img-lzw.tif",
#     img,
#     compression="lzw",
#     # compression='jpeg2000',
#     # compressionargs={
#     #     'reversible': False,
#     #     'level': 90,
#     #     'numthreads': 4,
#     #     "codecformat": imagecodecs.JPEG2K.CODEC.JP2,
#     #     "colorspace": imagecodecs.JPEG2K.CLRSPC.UNSPECIFIED,
#     # }
# )
# print('tifffile write image done.')

# read jp2
# path = "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/encoded_multispectral_img.jp2"
# from skimage import io

# img = io.imread(path)
# print(f'skimage read image shape: {img.shape}')


# to jp2


# encoded = imagecodecs.jpeg2k_encode(
#     img,
#     level=90,
#     colorspace=imagecodecs.JPEG2K.CLRSPC.UNSPECIFIED,
#     codecformat='jp2',
#     reversible=False,
# )
# print(f'encoded image done.')
# with open('encoded_multispectral_img.jp2', 'wb') as f:
#     f.write(encoded)


# jpg image
# path = "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/encoded_multispectral_img.jp2"
# decoder = nvimgcodec.Decoder()
# img_nv = decoder.decode(path)
# print(f'nvimgcodec read image shape: {img_nv.shape}')

# nvimagecodec read image

# decoder = nvimgcodec.Decoder()
# img_nv = decoder.decode(path)
# print(f'nvimgcodec read image shape: {img_nv.shape}')


# encoder to save
gpu_encoder = nvimgcodec.Encoder(
    backends=[
        nvimgcodec.Backend(nvimgcodec.GPU_ONLY, load_hint=0.5),
        nvimgcodec.Backend(nvimgcodec.HYBRID_CPU_GPU),
    ]
)
# cpu_encoder = nvimgcodec.Encoder(backends=[nvimgcodec.Backend(nvimgcodec.CPU_ONLY)])
jp2k_params = nvimgcodec.Jpeg2kEncodeParams(reversible=False)
# jp2k_params.num_resolutions = 2
# jp2k_params.code_block_size = (32, 32)
jp2k_params.bitstream_type = nvimgcodec.Jpeg2kBitstreamType.JP2

gpu_encoder.write(
    "test_img_nvimgcodec.tiff",
    img,
    params=nvimgcodec.EncodeParams(jpeg2k_encode_params=jp2k_params),
)
print("Image saved with nvimgcodec encoder.")

import torch
from torchmetrics.image import PeakSignalNoiseRatio


import torchvision

from src.data.hyperspectral_loader import get_hyperspectral_dataloaders
from src.stage1.cosmos.inference.utils import load_decoder_model, load_encoder_model

base_path = "/Data4/cao/ZiHanCao/exps/Cosmos/checkpoints/Cosmos-0.1-Tokenizer-CI8x8"
enc_path = f"{base_path}/encoder.jit"
dec_path = f"{base_path}/decoder.jit"
dataset_path = "/HardDisk/ZiHanCao/datasets/Multispectral_webdatasets/DCF_2019_Track_2-8_bands-px_512-MSI-0017.tar"
save_base_dir = "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/scripts/tests/test_cosmos_imgs"

device = "cuda:1"

enc = load_encoder_model(enc_path, device=device)
dec = load_decoder_model(dec_path, device=device)
print("loaded cosmos pretrained tokenizer")

test_dataset, test_loader = get_hyperspectral_dataloaders(
    wds_paths=dataset_path,
    batch_size=2,
    num_workers=1,
    shuffle_size=100,
    to_neg_1_1=True,
    transform_prob=0.0,
)

for i, batch in enumerate(test_loader):
    img_tensor = batch["img"][:, [4, 2, 1]]  # shape: [N, C, H, W]
    img_tensor = img_tensor.to(device)

    # to cosmos tokenizer
    with torch.no_grad():
        latent = enc(img_tensor)[0]
        print(latent.shape)
        recon = dec(latent)

    # psnr
    to_rgb = lambda x: (x + 1) / 2
    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
    psnr = psnr_fn(to_rgb(recon), to_rgb(img_tensor)).item()

    print(psnr)

    # plot the image
    # 将张量转换为PIL图像并显示
    img_grid = torchvision.utils.make_grid(to_rgb(img_tensor), nrow=2, padding=2)
    recon_grid = torchvision.utils.make_grid(to_rgb(recon), nrow=2, padding=2)

    # 保存图像到本地
    torchvision.utils.save_image(img_grid, f"{save_base_dir}/original_{i}.png")
    torchvision.utils.save_image(recon_grid, f"{save_base_dir}/reconstructed_{i}.png")

    if i > 5:
        break

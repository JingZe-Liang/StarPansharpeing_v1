import hydra
import torch
import torch.nn as nn
import yaml
from easydict import EasyDict
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel, get_peft_model

from src.stage1.cosmos.cosmos_tokenizer import ContinuousImageTokenizer

# Load model
config = yaml.load(
    open(
        "runs/stage1_cosmos_nested/2025-10-22_19-23-25_cosmos_f8c16p1_unified_hyperspectral_latent_noise=0.0_channel_drop=False/config/config_total.yaml"
    ),
    Loader=yaml.UnsafeLoader,
)
config = EasyDict(config)
tokenizer_cfg = config.tokenizer

# Init the model
model = hydra.utils.instantiate(tokenizer_cfg)
model.load_pretrained(
    uni_tokenizer_path="runs/stage1_cosmos_nested/2025-10-22_19-23-25_cosmos_f8c16p1_unified_hyperspectral_latent_noise=0.0_channel_drop=False/ema/tokenizer/model.safetensors"
)
print("Model loaded")

# Wrap with LoRA
peft_model_path = "runs/stage1_cosmo_nested_lora/2025-11-10_21-17-21_cosmos_lora=lora_r=32_f8c16p1_WDC/peft_ckpt/WDC"
peft_model = PeftModel.from_pretrained(
    model,
    model_id=peft_model_path,
    adapter_name="WDC",
    ignore_mismatched_sizes=True,
)
print("LoRA loaded")


# Add another LoRA
peft_model.load_adapter(
    "runs/stage1_cosmo_nested_lora/2025-11-11_14-46-46_cosmos_lora=lora_r=32_f8c16p1_Houston/peft_ckpt/Houston",
    adapter_name="Houston",
    is_trainable=False,
    allow_missing_keys=True,
)
print("Second LoRA loaded")

peft_model.set_adapter("WDC")
print("Active adapter set to WDC")

# Merge LoRA
peft_model = peft_model.base_model.merge_and_unload()
peft_model = peft_model.cuda()
print("LoRA merged")


# Unmerge LoRA
# unmerged_model = peft_model.base_model.unmerge_adapter()
# print("LoRA unmerged")

# Verify that the unmerged model produces the same output as the original LoRA model
from src.data.hyperspectral_loader import get_hyperspectral_dataloaders

_, dl = get_hyperspectral_dataloaders(
    "data/Downstreams/WDCDenoise/hyper_images/Washington_DC_mall-191_bands-px_192-0000.tar",
    batch_size=1,
    num_workers=0,
    shuffle_size=-1,
)
x = next(iter(dl))["img"]
x = x.cuda()
print(f"range: {x.min().item()} to {x.max().item()}, shape: {x.shape}")

with torch.no_grad():
    out_lora = peft_model(x)
    if isinstance(out_lora, tuple):
        out_lora = out_lora[0]

# psnr
from torchmetrics import PeakSignalNoiseRatio

psnr = PeakSignalNoiseRatio().cuda()
psnr_val = psnr((x + 1) / 2, (out_lora + 1) / 2)
print(f"PSNR after LoRA merge: {psnr_val.item():.2f} dB")

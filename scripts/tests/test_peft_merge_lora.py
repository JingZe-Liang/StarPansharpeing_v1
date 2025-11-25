import json

import torch
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

from src.stage1.cosmos.cosmos_tokenizer import ContinuousImageTokenizer
from src.utilities.network_utils.network_loading import load_peft_model_checkpoint

tokenizer = ContinuousImageTokenizer(
    attn_resolutions=[32],
    channels=128,
    channels_mult=[2, 4, 4],
    dropout=0.0,
    in_channels=50,
    spatial_compression=8,
    num_res_blocks=2,
    out_channels=50,
    resolution=1024,
    patch_size=4,
    patch_method="haar",
    latent_channels=16,
    z_channels=16,
    z_factor=1,
    name="CI",
    formulation="AE",
    encoder="Default",
    decoder="Default",
    act_checkpoint=True,
    norm_in_quant_conv=False,
    enc_path="/Data4/cao/ZiHanCao/exps/Cosmos/checkpoints/Cosmos-0.1-Tokenizer-CI8x8/encoder.jit",
    dec_path="/Data4/cao/ZiHanCao/exps/Cosmos/checkpoints/Cosmos-0.1-Tokenizer-CI8x8/decoder.jit",
    wrap_fsdp_last_layer=False,
    uni_tokenizer_path="runs/stage1_cosmos/cosmos_f8c16p4_psnr_39/ema/tokenizer/model.safetensors",
    loading_type="pretrained",
    force_not_attn=True,
    hook_for_repa=True,
).cuda()

peft_cfg = json.load(
    open("runs/stage1_cosmos_lora/2025-05-08_04-55-03_lora_fintune_f8c16p4_Houston/peft_ckpt/adapter_config.json")
)
print(peft_cfg)

LOAD_TYPE = 0  # 0: load from load_from_pretrained method; 1: manually load from checkpoint
if LOAD_TYPE == 0:
    print("load from load_from_pretrained method")
    peft_cfg, peft_model = load_peft_model_checkpoint(
        base_model=tokenizer,
        base_model_pretrained_path=None,
        peft_pretrained_path="runs/stage1_cosmos_lora/2025-05-08_04-55-03_lora_fintune_f8c16p4_Houston/peft_ckpt",
    )

elif LOAD_TYPE == 1:
    print("load from manually load from checkpoint")
    peft_model = get_peft_model(tokenizer, LoraConfig(**peft_cfg), adapter_name="default")
    peft_model.print_trainable_parameters()
    peft_model.load_adapter(
        model_id="runs/stage1_cosmos_lora/2025-05-08_04-55-03_lora_fintune_f8c16p4_Houston/peft_ckpt",
        adapter_name="default",
        peft_config=peft_cfg,
    )
    print("load adapter done")
    unloaded_model = peft_model.merge_and_unload(progressbar=True, adapter_names=["default"])
    print("merge and unload done")
    print("unloaded model", type(unloaded_model))

from src.data.hyperspectral_loader import get_fast_test_hyperspectral_data

loader = get_fast_test_hyperspectral_data("Houston", batch_size=1)

img = next(iter(loader))["img"]
img = img.to("cuda")
peft_model.eval()

print("\n\n")
print("-" * 30, "start evaluating", "-" * 30)
with torch.no_grad():
    print("img shape", img.shape)
    print("img dtype", img.dtype)
    output = peft_model(img)
    print("output shape", output.shape)
    print("output dtype", output.dtype)

    # compute the psnr

    from torchmetrics.image import PeakSignalNoiseRatio

    psnr = PeakSignalNoiseRatio(data_range=1.0).to("cuda")

    output = output * 0.5 + 0.5
    img = img * 0.5 + 0.5
    output = output.clamp(0, 1)
    img = img.clamp(0, 1)
    psnr_value = psnr(output, img)

    print(f"PSNR: {psnr_value.item()}")

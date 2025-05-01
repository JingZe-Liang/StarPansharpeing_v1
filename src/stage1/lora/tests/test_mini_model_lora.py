import sys

import accelerate
import torch
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
from torch.distributed.tensor import DTensor

sys.path.insert(0, __file__[: __file__.find("src")])
from src.stage1.cosmos.cosmos_tokenizer import ContinuousImageTokenizer


def is_main_process():
    return torch.distributed.is_initialized() and torch.distributed.get_rank() == 0


def main():
    accelerator = accelerate.Accelerator()
    peft_cfg = LoraConfig(
        peft_type="lokr",
        r=16,
        lora_alpha=8,
        use_dora=True,
        target_modules=["conv1", "conv2", "q", "k", "v", "proj_out"],
        lora_dropout=0.1,
        modules_to_save=["encoder.encoder.conv_in", "decoder.decoder.conv_out"],
    )
    model_cfg = {
        "attn_resolutions": [32],
        "channels": 128,
        "channels_mult": [2, 4, 4],
        "dropout": 0.0,
        "in_channels": 3,
        "spatial_compression": 8,
        "num_res_blocks": 2,
        "out_channels": 3,
        "resolution": 1024,
        "patch_size": 4,
        "patch_method": "haar",
        "latent_channels": 16,
        "z_channels": 16,
        "z_factor": 1,
        "name": "CI",
        "formulation": "AE",
        "encoder": "Default",
        "decoder": "Default",
        "act_checkpoint": False,
    }

    model = ContinuousImageTokenizer(**model_cfg)
    peft_model = get_peft_model(
        model=model,
        peft_config=peft_cfg,
        adapter_name="default",
        low_cpu_mem_usage=False,
    )
    print(peft_model.get_base_model().__class__)

    # model = inject_adapter_in_model(peft_cfg, model)
    model = peft_model.cuda().to(torch.bfloat16)

    print(peft_model.print_trainable_parameters())

    # * fsdp
    # optimization
    optimizer = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad],
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.01,
        amsgrad=False,
    )

    model = peft_model.get_base_model()
    peft_model.dtype = torch.float32
    peft_model, optimizer = accelerator.prepare(peft_model, optimizer)
    print(model.__class__)

    # for k, v in model.named_parameters():
    #     if torch.distributed.get_rank() == 0:
    #         print(k)

    for k, v in model.get_submodule("encoder.encoder.down.0.block").named_children():
        if is_main_process():
            print(k, v.__class__)

    # * ckpt in fsdp
    sd = get_peft_model_state_dict(peft_model)
    if is_main_process():
        for k, v in sd.items():
            print(f"global size: {k} {v.size()}")
            if isinstance(v, DTensor):
                print(f"dtensor size: {k} {v.to_local().size()}")

    # get the fsdp shard ckpt
    accelerate.utils.save_fsdp_model(
        accelerator.state.fsdp_plugin,
        accelerator,
        peft_model,
        output_dir="tmp_ckpt/shard_ckpts/",
        model_index=0,
        adapter_only=True,
    )
    print("save shard ckpt done")

    # * params
    # for k, v in model.named_parameters():
    #     if "lora" in k:
    #         print(f"found lora module: {k}")

    # * save model
    # model.save_pretrained(
    #     save_directory="./tmp_lora",
    # )
    # state_dict = get_peft_model_state_dict(model.base_model)
    # torch.save(state_dict, "./tmp_lora/pytorch_model.pt")
    # print("saved peft adapters")

    # * load model
    # outcome = set_peft_model_state_dict(
    #     model.base_model, torch.load("./tmp_lora/pytorch_model.pt")
    # )
    # print(f"missing_keys= {outcome.missing_keys}")
    # print("-" * 30)
    # print(f"unexpected_keys= {outcome.unexpected_keys}")

    # x = torch.randn(1, 3, 1024, 1024).cuda().to(torch.bfloat16)
    # y = model(x)
    # optimizer.zero_grad()
    # y.mean().backward()
    # optimizer.step()

    # import time

    # time.sleep(20)


if __name__ == "__main__":
    main()

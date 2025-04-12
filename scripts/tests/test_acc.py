import accelerate

# import safetensors
# import torch
# from accelerate.utils import load_state_dict
# from safetensors.torch import load_file, save_file
# # file_path = "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/runs/stage1_cosmos/2025-03-28_02-19-46/cosmos_tokenizer_pos_training/ema/ema.pt"
# # ckpt = load_state_dict(file_path)
# # print(ckpt)
# # pass
# # d = {
# #     # "model_1": {"a": torch.randn(1, 2, 3), "b": torch.randn(1, 2, 3)},
# #     # "model_2": {"c": torch.randn(1, 2, 3), "d": torch.randn(1, 2, 3)},
# #     "a": torch.randn(1, 2, 3), "b": torch.randn(1, 2, 3)
# # }
# # save_file(d, "test.safetensors")
# accelerator = accelerate.Accelerator()
# net = torch.nn.Conv2d(
#     3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
# ).cuda()
# net_c = torch.compile(net)
# print(net_c.state_dict().keys())
# print(accelerator.unwrap_model(net_c, keep_torch_compile=False).state_dict().keys())
# import accelerate
# import torch
# accelerator = accelerate.Accelerator(
#     project_config=accelerate.utils.ProjectConfiguration(
#         project_dir="/Data2/ZiHanCao/exps/hyperspectral-1d-tokenizer/runs/stage1_1d_tok/ckpts",
#         automatic_checkpoint_naming=True,
#         total_limit=2,
#         save_on_each_node=False,
#     )
# )
# net = torch.nn.Conv2d(
#     3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
# ).cuda()
# net = accelerator.prepare(net)
# for i in range(10):
#     accelerator.save_state()
#     print("save_state done")
import torch

accelerator = accelerate.Accelerator()

a = torch.tensor([accelerator.process_index], dtype=torch.float32).cuda()
a_mean = accelerator.gather(a).mean()
print(a_mean)

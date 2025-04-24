import torch
from accelerate import Accelerator


def main():
    accelerator = Accelerator()
    model = torch.nn.Conv2d(3, 3, 3, 1, 1).to(accelerator.device)
    model._no_split_modules = []
    optimizer = torch.optim.Adam(model.parameters())
    x = torch.randn(32, 3, 32, 32).to(accelerator.device)
    model.dtype = torch.float32
    model, optimizer = accelerator.prepare(model, optimizer)

    with accelerator.accumulate(model):
        output = model(x)
        loss = output.mean()
        accelerator.backward(loss)  # 测试是否能正常反向传播


if __name__ == "__main__":
    main()
    print("Test passed!")

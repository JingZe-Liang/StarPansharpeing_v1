import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 100, 3, 1, 1)

    def forward(self, x):
        return self.conv(x)


if __name__ == "__main__":
    model = Conv()
    print(model)

    print("---------------------------")
    model.compile()
    print(model)
    print(model.forward)

import torch
import torch.nn as nn


class ToEndMemberConv(nn.Module):
    def __init__(self, num_endmember: int, channels: int, kernel: int):
        super(ToEndMemberConv, self).__init__()
        padding = kernel // 2
        self.decoder = nn.Conv2d(
            in_channels=num_endmember,
            out_channels=channels,
            kernel_size=kernel,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.relu = nn.ReLU()

    def forward(self, code):
        code = self.relu(self.decoder(code))  # [bs, c_in, h, w] -> [bs, c_out, h, w]
        return code

    def get_endmember(self):
        endmember = self.decoder.weight.data.clamp_(min=0.0)
        return endmember


class ToEndMemberParameter(nn.Module):
    def __init__(self, num_endmember: int, channels: int):
        super(ToEndMemberParameter, self).__init__()
        self.endmember = nn.Parameter(torch.randn(channels, num_endmember))

        # init
        nn.init.trunc_normal_(self.endmember.data, mean=0.0, std=1.0)
        self.endmember.data.clamp_(min=0.0)

    def forward(self, code):
        code = torch.einsum("bchw,dc->bdhw", code, self.endmember)
        return code

    def get_endmember(self):
        return self.endmember.data.clamp_(min=0.0)

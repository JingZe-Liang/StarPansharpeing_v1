import torch
import torch.nn as nn


class ToEndMemberConv(nn.Module):
    def __init__(self, num_endmember: int, channels: int, kernel: int, init_value=None):
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
        if init_value is not None:
            assert kernel == 1
            init_value.squeeze_(0)
            assert init_value.ndim == 2, "init_value must be 2D"
            self.decoder.weight.data = init_value[..., None, None]

    def forward(self, code):
        code = self.relu(self.decoder(code))  # [bs, c_in, h, w] -> [bs, c_out, h, w]
        return code

    def get_endmember(self):
        endmember = self.decoder.weight.data.clamp_(min=0.0)
        return endmember


class ToEndMemberParameter(nn.Module):
    def __init__(self, num_endmember: int, channels: int, init_value=None):
        super(ToEndMemberParameter, self).__init__()
        self.endmember = nn.Parameter(torch.randn(channels, num_endmember))

        # init
        if init_value is not None:
            init_value.squeeze_(0)
            assert init_value.ndim == 2, "init_value must be 2D"
            assert self.endmember.data.shape == init_value.shape, (
                f"init_value shape must be {self.endmember.data.shape}"
            )
            self.endmember.data = init_value
        else:
            nn.init.trunc_normal_(self.endmember.data, mean=0.0, std=1.0)
            self.endmember.data.clamp_(min=0.0)

    def forward(self, code):
        code = torch.einsum("bchw,dc->bdhw", code, self.endmember)
        return code

    def get_endmember(self):
        return self.endmember.data.clamp_(min=0.0)

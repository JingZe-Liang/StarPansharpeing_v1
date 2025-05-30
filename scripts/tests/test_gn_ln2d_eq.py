import torch
import torch.nn as nn

from src.stage1.cosmos.modules.utils import RMSNorm2d

# gn = torch.nn.GroupNorm(num_groups=1, num_channels=3, eps=1e-6, affine=True)


# class LN2D(torch.nn.Module):
#     def __init__(self, shape: int | tuple, num_groups=1):
#         super().__init__()
#         self.norm = torch.nn.LayerNorm(
#             normalized_shape=shape,
#             eps=1e-6,
#             elementwise_affine=True,
#         )

#     def forward(self, x):
#         if len(self.norm.normalized_shape) == 1:
#             return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
#         else:
#             return self.norm(x)


# ln2d = LN2D((3, 32, 32), 1)

# x = torch.arange(3 * 32 * 32).float()
# x = x.reshape(3, 32, 32)[None]
# x_gn = gn(x)
# x_ln = ln2d(x)


# # assert close
# close = torch.allclose(x_gn, x_ln, atol=1e-6)
# print(f"GroupNorm and LayerNorm2D are close: {close}")  # True


# ln2d_only_channel = LN2D(3, 1)
# x_ln_c = ln2d_only_channel(x)
# close = torch.allclose(x_gn, x_ln_c, atol=1e-6)
# print(f"GroupNorm and LayerNorm2D only channel are close: {close}")  # False


class RMSNorm2dTorch(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = torch.nn.RMSNorm(
            normalized_shape=c,
            eps=1e-6,
            elementwise_affine=True,
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # Change to (N, H, W, C)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


norm2dtorch = RMSNorm2dTorch(3).cuda()
x = torch.arange(3 * 32 * 32).float()
x = x.reshape(3, 32, 32)[None].cuda()
x_norm2dtorch = norm2dtorch(x)

print(x_norm2dtorch.shape)  # Should be (1, 3, 32, 32)

# Test RMSNorm2dTriton

rms2dtriton = RMSNorm2d(3).cuda()
x_rms2dtriton = rms2dtriton(x)
print(x_rms2dtriton.shape)  # Should be (1, 3, 32, 32)


# close
closed = torch.allclose(x_norm2dtorch, x_rms2dtriton, atol=1e-6)
print(f"RMSNorm2dTorch and RMSNorm2dTriton are close: {closed}")  # True

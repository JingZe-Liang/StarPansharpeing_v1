import torch
from flash_attn.ops.rms_norm import RMSNorm

from src.stage1.sana_dcae.models.nn.norm import TritonRMSNorm2d


class RMSNorm2dNative(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = torch.nn.parameter.Parameter(torch.ones(self.num_features))
            if bias:
                self.bias = torch.nn.parameter.Parameter(torch.zeros(self.num_features))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x / torch.sqrt(torch.square(x.float()).mean(dim=1, keepdim=True) + self.eps)).to(x.dtype)
        if self.elementwise_affine:
            x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x


class RMSNorm2dFlash(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        self.rms_norm = RMSNorm(num_features, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        return self.rms_norm(x).permute(0, 3, 1, 2)


rms_2d_native = RMSNorm2dNative(32, 1e-6).to("cuda", torch.bfloat16)
rms_2d_flash = RMSNorm2dFlash(32, 1e-6).to("cuda", torch.bfloat16)
rms_2d_triton = TritonRMSNorm2d(32, 1e-6).to("cuda", torch.bfloat16)

x = torch.randn(64, 32, 128, 128).to("cuda", torch.bfloat16)

# Warm-up
with torch.autocast("cuda", dtype=torch.bfloat16):
    for _ in range(30):  # Run a few iterations for warm-up
        _ = rms_2d_native(x)
        _ = rms_2d_flash(x)
        _ = rms_2d_triton(x)
torch.cuda.synchronize()  # Ensure warm-up is complete


t1 = torch.cuda.Event(enable_timing=True)  # Add enable_timing=True
t2 = torch.cuda.Event(enable_timing=True)  # Add enable_timing=True
t1.record(torch.cuda.current_stream())
with torch.autocast("cuda", dtype=torch.bfloat16) and torch.no_grad():
    for _ in range(2000):
        native_x = rms_2d_native(x)
t1.synchronize()
t2.record(torch.cuda.current_stream())
t2.synchronize()
print(f"RMSNorm2dNative time: {t1.elapsed_time(t2)} ms")


t1 = torch.cuda.Event(enable_timing=True)  # Add enable_timing=True
t2 = torch.cuda.Event(enable_timing=True)  # Add enable_timing=True
t1.record(torch.cuda.current_stream())
with torch.autocast("cuda", dtype=torch.bfloat16) and torch.no_grad():
    for _ in range(2000):
        flash_x = rms_2d_flash(x)
t1.synchronize()
t2.record(torch.cuda.current_stream())
t2.synchronize()
print(f"RMSNorm2dFlash time: {t1.elapsed_time(t2)} ms")

t1 = torch.cuda.Event(enable_timing=True)  # Add enable_timing=True
t2 = torch.cuda.Event(enable_timing=True)  # Add enable_timing=True
t1.record(torch.cuda.current_stream())
with torch.autocast("cuda", dtype=torch.bfloat16) and torch.no_grad():
    for _ in range(2000):
        triton_x = rms_2d_triton(x)
t1.synchronize()
t2.record(torch.cuda.current_stream())
t2.synchronize()
print(f"RMSNorm2dTriton time: {t1.elapsed_time(t2)} ms")

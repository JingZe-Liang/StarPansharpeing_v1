import os
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm

# Adjust paths to ensure we can import SDB modules
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file)))))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import SDB modules
from src.utilities.transport.SDB.plan import SDBContinuousPlan, SDBContinuousSampler, DiffusionTarget
from src.utilities.optim.muon_fused import MuonFSDP

# Assuming 'precond.py' is in the same folder as 'plan.py' and contains EDMPrecond
try:
    from src.utilities.transport.SDB.precond import EDMPrecond
except ImportError:
    # Local fallback
    sys.path.append(os.path.dirname(current_file))
    from precond import EDMPrecond

# --- Model Definitions (SimpleUNet with Time Conditioning) ---


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int):
        super().__init__()
        num_groups = 32 if in_channels % 32 == 0 else 1
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        num_groups_out = 32 if out_channels % 32 == 0 else 1
        self.norm2 = nn.GroupNorm(num_groups_out, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if time_channels > 0:
            self.time_emb = nn.Linear(time_channels, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor | None = None) -> torch.Tensor:
        h = self.act1(self.norm1(x))
        h = self.conv1(h)
        if t_emb is not None and hasattr(self, "time_emb"):
            # Broadcast t_emb: [B, C] -> [B, C, 1, 1]
            h = h + self.time_emb(t_emb)[:, :, None, None]
        h = self.act2(self.norm2(h))
        h = self.conv2(h)
        return h + self.shortcut(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels: int = 7, out_channels: int = 3, base_channels: int = 128):
        super().__init__()
        self.time_dim = base_channels * 4

        # Time embedding
        self.sinusoid = SinusoidalPosEmb(base_channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        # Encoder
        # Level 1: 32x32
        self.inc_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.inc_res = nn.ModuleList([ResBlock(base_channels, base_channels, self.time_dim) for _ in range(2)])

        # Level 2: 16x16
        self.down1_conv = nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1)
        self.down1_res = nn.ModuleList(
            [
                ResBlock(base_channels, base_channels * 2, self.time_dim),
                ResBlock(base_channels * 2, base_channels * 2, self.time_dim),
            ]
        )

        # Level 3: 8x8
        self.down2_conv = nn.Conv2d(base_channels * 2, base_channels * 2, 3, stride=2, padding=1)
        self.down2_res = nn.ModuleList(
            [
                ResBlock(base_channels * 2, base_channels * 4, self.time_dim),
                ResBlock(base_channels * 4, base_channels * 4, self.time_dim),
            ]
        )

        # Level 4: 4x4
        self.down3_conv = nn.Conv2d(base_channels * 4, base_channels * 4, 3, stride=2, padding=1)
        self.down3_res = nn.ModuleList(
            [
                ResBlock(base_channels * 4, base_channels * 4, self.time_dim),
                ResBlock(base_channels * 4, base_channels * 4, self.time_dim),
            ]
        )

        # Bottleneck
        self.mid = nn.ModuleList(
            [
                ResBlock(base_channels * 4, base_channels * 4, self.time_dim),
                ResBlock(base_channels * 4, base_channels * 4, self.time_dim),
            ]
        )

        # Decoder
        # Level 4 -> 3 (4x4 -> 8x8)
        self.up3_upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.up3_conv = nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1)
        self.up3_res = nn.ModuleList(
            [
                ResBlock(base_channels * 8, base_channels * 4, self.time_dim),
                ResBlock(base_channels * 4, base_channels * 4, self.time_dim),
            ]
        )

        # Level 3 -> 2 (8x8 -> 16x16)
        self.up2_upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.up2_conv = nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1)
        self.up2_res = nn.ModuleList(
            [
                ResBlock(base_channels * 4, base_channels * 2, self.time_dim),
                ResBlock(base_channels * 2, base_channels * 2, self.time_dim),
            ]
        )

        # Level 2 -> 1 (16x16 -> 32x32)
        self.up1_upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.up1_conv = nn.Conv2d(base_channels * 2, base_channels, 3, padding=1)
        self.up1_res = nn.ModuleList(
            [
                ResBlock(base_channels * 2, base_channels, self.time_dim),
                ResBlock(base_channels, base_channels, self.time_dim),
            ]
        )

        self.outc = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(
        self,
        img: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        x_1: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if img is None:
            img = kwargs.pop("x", None)
        if img is None:
            raise TypeError("SimpleUNet.forward expects `img` or `x`")
        if timesteps is None:
            raise TypeError("SimpleUNet.forward expects `timesteps`")
        if x_1 is None:
            raise TypeError("SimpleUNet.forward expects `x_1`")

        t = timesteps
        if t.ndim > 1:
            t = t.view(-1)

        t_emb = self.sinusoid(t)
        t_emb = self.time_mlp(t_emb)

        # Encode
        inputs = [img, x_1]
        if mask is not None:
            inputs.append(mask)
        x = torch.cat(inputs, dim=1)

        # Level 1: 32x32
        x1 = self.inc_conv(x)
        for block in self.inc_res:
            x1 = block(x1, t_emb)

        # Level 2: 16x16
        x2 = self.down1_conv(x1)
        for block in self.down1_res:
            x2 = block(x2, t_emb)

        # Level 3: 8x8
        x3 = self.down2_conv(x2)
        for block in self.down2_res:
            x3 = block(x3, t_emb)

        # Level 4: 4x4
        x4 = self.down3_conv(x3)
        for block in self.down3_res:
            x4 = block(x4, t_emb)

        # Mid
        h = x4
        for block in self.mid:
            h = block(h, t_emb)

        # Decode
        # Level 4 -> 3 (4x4 -> 8x8)
        h = self.up3_upsample(h)
        h = self.up3_conv(h)
        h = torch.cat([h, x3], dim=1)
        for block in self.up3_res:
            h = block(h, t_emb)

        # Level 3 -> 2 (8x8 -> 16x16)
        h = self.up2_upsample(h)
        h = self.up2_conv(h)
        h = torch.cat([h, x2], dim=1)
        for block in self.up2_res:
            h = block(h, t_emb)

        # Level 2 -> 1 (16x16 -> 32x32)
        h = self.up1_upsample(h)
        h = self.up1_conv(h)
        h = torch.cat([h, x1], dim=1)
        for block in self.up1_res:
            h = block(h, t_emb)

        return self.outc(h)


# --- Data Loading ---


def _make_center_mask(*, height: int, width: int, hole_ratio: float, device: torch.device) -> torch.Tensor:
    if not (0.0 < hole_ratio < 1.0):
        raise ValueError("hole_ratio must be in (0, 1)")

    hole_h = max(1, int(round(height * hole_ratio)))
    hole_w = max(1, int(round(width * hole_ratio)))
    top = (height - hole_h) // 2
    left = (width - hole_w) // 2

    mask = torch.ones((1, height, width), device=device, dtype=torch.float32)
    mask[:, top : top + hole_h, left : left + hole_w] = 0.0
    return mask


def _apply_inpainting_mask(x0: torch.Tensor, *, mask: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
    fill = torch.full_like(x0, fill_value)
    return x0 * mask + fill * (1.0 - mask)


def get_dataloader(batch_size: int = 64, hole_ratio: float = 0.5):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Use CIFAR10
    try:
        ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    except:
        print("CIFAR10 download failed, using fake data")
        ds = torch.utils.data.TensorDataset(torch.randn(100, 3, 32, 32), torch.zeros(100))

    class InpaintingDataset(torch.utils.data.Dataset):
        def __init__(self, ds, hole_ratio: float):
            self.ds = ds
            self.hole_ratio = hole_ratio

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            clean, _ = self.ds[idx]
            _, height, width = clean.shape

            # mask=1 means known pixels, mask=0 means hole (to be inpainted)
            mask = _make_center_mask(height=height, width=width, hole_ratio=self.hole_ratio, device=clean.device)
            masked = _apply_inpainting_mask(clean, mask=mask, fill_value=0.0)
            return clean, masked, mask

    inpaint_ds = InpaintingDataset(ds, hole_ratio)
    return torch.utils.data.DataLoader(inpaint_ds, batch_size=batch_size, shuffle=True, num_workers=2)


# --- Main Training/Sampling ---


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Setup Plan
    plan = SDBContinuousPlan(
        plan_tgt=DiffusionTarget.x_0,
        gamma_max=0.3,
        diffusion_type="bridge",
        t_train_kwargs={"device": device.type, "clip_t_min_max": (1e-4, 1 - 1e-4)},
    )

    # 2. Setup Model
    # For inpainting, pass condition image and mask to the model:
    # input = concat([x_t, x_1(masked), mask]) => 3 + 3 + 1 = 7 channels
    inner_model = SimpleUNet(in_channels=7, out_channels=3).to(device)

    # 3. Setup Preconditioner
    model = EDMPrecond(inner_model, plan).to(device)
    # model = inner_model.to(device)

    # optimizer = optim.AdamW(model.parameters(), lr=1e-4, betas=(0.95, 0.999), weight_decay=1e-4)
    optimizer = MuonFSDP.create_muon_optimizer(
        model.named_parameters(),
        lr=3e-4,
        cautious_wd=True,
        nesterov=True,
    )
    loader = get_dataloader(batch_size=64, hole_ratio=0.5)

    # 4. Train Loop
    epochs = 6  # Reducing to 1 for quick verification
    print("Starting SDB Training for Inpainting (x0 target) with EDMPrecond...")

    for epoch in range(epochs):
        loop = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for clean, masked, mask in loop:
            clean = clean.to(device)  # x_0
            masked = masked.to(device)  # x_1 (condition / endpoint)
            mask = mask.to(device)  # 1=known, 0=hole

            B = clean.shape[0]
            t = plan.train_continous_t(B).to(device)

            # x_t, target (=x0 for DiffusionTarget.x_0)
            x_t, target = plan.get_x_t_with_target(t, clean, masked)

            # EDMPrecond returns (pred_x0, aux, weight)
            # weight = 1.0
            pred, _, weight = model(x_t, t, x_1=masked, mask=mask)

            # Only supervise the missing region (hole)
            # hole = (1.0 - mask).to(dtype=pred.dtype)
            # denom = hole.sum().clamp_min(1.0)
            # loss = (((pred - target).pow(2) * weight) * hole).sum() / denom
            loss = ((pred - target).pow(2) * weight).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

    # 5. Sampling
    print("Sampling...")
    model.eval()

    # Define Sampler with pre_noise_x_1 implementation (missing in base)
    # class MySampler(SDBContinuousSampler):
    #     def pre_noise_x_1(self, x_1):
    #         if self.sample_noisy_x1_b > 0:
    #             # Simple Gaussian noise addition if needed
    #             return x_1 + torch.randn_like(x_1) * self.sample_noisy_x1_b
    #         return x_1

    sampler = SDBContinuousSampler(plan)

    # Get a batch
    clean, masked, mask = next(iter(loader))
    clean = clean.to(device)
    masked = masked.to(device)
    mask = mask.to(device)

    # Reverse Sampling
    # Start: x_1 (noisy). Condition: x_sample_noisy_x1_b=0 (clean condition is just x_1 itself)
    # The sampler expects x_1.
    # It will use x_1 as starting point x_T (actually pre-noised if sample_noisy_x1_b > 0).
    # And x_1 as condition x_cond.

    with torch.no_grad():
        # out, _, _ = sampler.sample_ode_euler(
        #     model, x_1=masked, sample_n_steps=25, clip_value=True, model_kwargs={"mask": mask}
        # )
        out, _, _ = sampler.sample_sde_euler(
            model,
            x_1=masked,
            sample_n_steps=20,
            last_n_step_only_mean=5,
            clip_value=True,
            model_kwargs={"mask": mask},
        )
        out = out * (1.0 - mask) + masked * mask

    mse = (out - clean).pow(2).mean().item()
    print(f"SDB Reconstruction MSE: {mse:.6f}")

    # Visualize
    def unnorm(img):
        return (img * 0.5 + 0.5).clamp(0, 1).cpu()

    grid = torch.cat([unnorm(clean[:8]), unnorm(masked[:8]), unnorm(out[:8])], dim=0)
    save_image(grid, "sdb_cifar_demo.png", nrow=8)
    print("Saved sdb_cifar_demo.png")


if __name__ == "__main__":
    main()

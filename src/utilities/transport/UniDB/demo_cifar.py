import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import math

# Add project root to sys.path to allow imports
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file)))))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from plan_continuous import UniDBContinuous, UniDBLoss, UniDBEpsilonSampler
    from plan_disc import UniDB as UniDBDiscrete
except ImportError:
    from src.utilities.transport.UniDB.plan_disc import UniDB as UniDBDiscrete
    # Assuming plan_continuous was already imported

    from plan_disc import UniDB as UniDBDiscrete
except ImportError:
    from src.utilities.transport.UniDB.plan_disc import UniDB as UniDBDiscrete
    # Assuming plan_continuous was already imported


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
    def __init__(self, in_channels, out_channels, time_channels, dropout=0.1):
        super().__init__()
        num_groups = 8
        if in_channels % num_groups != 0:
            num_groups = 1  # Fallback to LayerNorm style (approx) or InstanceNorm if groups=channels
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        num_groups_out = 8
        if out_channels % num_groups_out != 0:
            num_groups_out = 1
        self.norm2 = nn.GroupNorm(num_groups_out, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if time_channels > 0:
            self.time_emb = nn.Linear(time_channels, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t=None):
        h = self.act1(self.norm1(x))
        h = self.conv1(h)

        if t is not None and hasattr(self, "time_emb"):
            h = h + self.time_emb(t)[:, :, None, None]

        h = self.act2(self.norm2(h))
        h = self.conv2(h)

        return h + self.shortcut(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()

        # Time Embedding
        self.time_dim = base_channels * 4
        self.sinusoid = SinusoidalPosEmb(base_channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        # Encoder
        # Level 1
        self.inc = ResBlock(in_channels, base_channels, self.time_dim)

        # Down 1
        self.down1_conv = nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1)
        self.down1_res = ResBlock(base_channels, base_channels * 2, self.time_dim)

        # Down 2
        self.down2_conv = nn.Conv2d(base_channels * 2, base_channels * 2, 3, stride=2, padding=1)
        self.down2_res = ResBlock(base_channels * 2, base_channels * 4, self.time_dim)

        # Bottleneck
        self.mid_res1 = ResBlock(base_channels * 4, base_channels * 4, self.time_dim)
        self.mid_res2 = ResBlock(base_channels * 4, base_channels * 4, self.time_dim)

        # Decoder
        # Up 2
        self.up2_tconv = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.up2_res = ResBlock(base_channels * 2 + base_channels * 2, base_channels * 2, self.time_dim)

        # Up 1
        self.up1_tconv = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.up1_res = ResBlock(base_channels + base_channels, base_channels, self.time_dim)

        self.outc = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x, t, mu=None):
        if t.ndim > 1:
            t = t.view(-1)

        t_emb = self.sinusoid(t)
        t_emb = self.time_mlp(t_emb)

        # Encoder
        x1 = self.inc(x, t_emb)  # [B, 64, 32, 32]

        x2 = self.down1_conv(x1)  # [B, 64, 16, 16]
        x2 = self.down1_res(x2, t_emb)  # [B, 128, 16, 16]

        x3 = self.down2_conv(x2)  # [B, 128, 8, 8]
        x3 = self.down2_res(x3, t_emb)  # [B, 256, 8, 8]

        # Bottleneck
        x3 = self.mid_res1(x3, t_emb)
        x3 = self.mid_res2(x3, t_emb)

        # Decoder
        x_up2 = self.up2_tconv(x3)  # [B, 128, 16, 16]
        x_up2 = torch.cat([x_up2, x2], dim=1)  # [B, 256, 16, 16]
        x_up2 = self.up2_res(x_up2, t_emb)  # [B, 128, 16, 16]

        x_up1 = self.up1_tconv(x_up2)  # [B, 64, 32, 32]
        x_up1 = torch.cat([x_up1, x1], dim=1)  # [B, 128, 32, 32]
        x_up1 = self.up1_res(x_up1, t_emb)  # [B, 64, 32, 32]

        return self.outc(x_up1)


def get_dataloader(batch_size=128, noise_std=0.2):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1] as typical in diffusion
        ]
    )

    # Use CIFAR10 train set
    try:
        dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    except Exception as e:
        print(f"Failed to download/load CIFAR10: {e}")
        print("Using random data for verification purposes.")
        dataset = TensorDataset(torch.randn(100, 3, 32, 32), torch.zeros(100))

    class NoisyDataset(torch.utils.data.Dataset):
        def __init__(self, ds, noise_std):
            self.ds = ds
            self.noise_std = noise_std

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            clean_img, label = self.ds[idx]  # clean_img is [-1, 1]
            noise = torch.randn_like(clean_img) * self.noise_std
            noisy_img = clean_img + noise
            return clean_img, noisy_img, label

    noisy_ds = NoisyDataset(dataset, noise_std)
    return DataLoader(noisy_ds, batch_size=batch_size, shuffle=True, num_workers=2)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Setup Data
    loader = get_dataloader(batch_size=64, noise_std=0.2)
    print("Dataloader ready.")

    # 2. Setup UniDB (Discrete)
    print("Initializing Discrete UniDB...")
    sde = UniDBDiscrete(lambda_square=30.0, gamma=1e8, T=100, schedule="cosine", eps=0.01, device=device)

    # 3. Model
    model = SimpleUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # Model Wrapper for discrete sde
    # plan_disc expects model(x, mu, t)
    # SimpleUNet expects model(x, t, mu=None) (and handles time emb)
    class ModelWrapper:
        def __init__(self, model):
            self.model = model

        def __call__(self, x, mu, t, **kwargs):
            # t in plan_disc is [B] or scalar tensor, or just tensor.
            # OR it is an int (during reverse sampling).

            if isinstance(t, int):
                t = torch.tensor([t], device=x.device, dtype=torch.long)
            else:
                t = t.to(x.device)

            # SimpleUNet needs [B].
            if t.ndim == 0:
                t = t.unsqueeze(0)
            if t.ndim == 1 and t.shape[0] == 1:
                # If it's a scalar tensor or shape [1], expand to [B]
                t = t.expand(x.shape[0])
            elif t.shape[0] != x.shape[0]:
                # Fallback for unexpected shapes, try to expand.
                t = t.expand(x.shape[0])

            # SimpleUNet signature: forward(self, x, t, mu=None)
            # mu is optional in SimpleUNet, but let's pass it if needed (not used for unconditional, but 'mu' is condition here).
            # The 'mu' in UniDBDiscrete calls is basically 'condition'.
            # SimpleUNet doesn't actually use 'mu' in forward?
            # Let's check SimpleUNet definition in Step 153.
            # def forward(self, x, t, mu=None): ...
            # It doesn't seem to use mu in current implementation.
            # But the task is restoration where mu IS the input/condition.
            # The SimpleUNet currently is just Denoising the xt.
            # But if it's conditional restoration, we should probably feed mu.
            # The user code in previous step didn't seem to pass mu to model?
            # Continuous: loss_fn -> model(xt, t01)
            # Discrete: sde.noise_fn -> model(x, mu, t)

            return self.model(x, t, mu)

    model_wrapper = ModelWrapper(model)
    sde.set_model(model_wrapper)

    # Loss function (Epsilon Matching)
    # Mean matching can be unstable due to small denominators in r_mean_1.
    # Epsilon matching is generally more stable and standard for diffusion.
    def compute_loss(sde, x0, mu):
        # x0=Clean, mu=Noisy
        sde.set_mu(mu)
        timesteps, states = sde.generate_random_states(x0=x0, mu=mu)
        # states is xt

        # Get predicted noise from model
        # sde.noise_fn calls model_wrapper -> model
        noise_pred = sde.noise_fn(states, timesteps.squeeze())

        # Get ground truth noise
        noise_gt = sde.get_real_noise(states, x0, timesteps)

        # L1 or MSE Loss
        return (noise_pred - noise_gt).abs().mean()
        # return (noise_pred - noise_gt).pow(2).mean()

    # 4. Train Loop
    print("Starting training...")
    model.train()

    max_steps = float("inf")
    step = 0

    for epoch in (tbar := tqdm(range(1), position=1, desc="Epoch")):
        for clean, noisy, _ in tqdm(loader, position=2, desc="Batch", leave=False):
            clean = clean.to(device)
            noisy = noisy.to(device)

            # Forward + Loss
            loss = compute_loss(sde, x0=clean, mu=noisy)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % 10 == 0:
                tbar.set_postfix(loss=loss.item())

            if step >= max_steps:
                break
        if step >= max_steps:
            break

    print("Training finished.")

    # 5. Verification: Sampling
    print("Verifying correctness by sampling...")
    model.eval()

    # Take a batch
    clean, noisy, _ = next(iter(loader))
    clean = clean.to(device)
    noisy = noisy.to(device)

    # 5. Verification: Sampling of Discrete SDE
    print("Sampling with Discrete SDE...")
    sde.set_mu(noisy)
    with torch.no_grad():
        sampled_clean = sde.reverse_sde(xt=noisy, save_states=False)

    # Compute error
    mse = (sampled_clean - clean).pow(2).mean().item()
    print(f"Reconstruction MSE: {mse:.6f}")

    # 6. Visualization
    def unnorm(img):
        img = img / 2 + 0.5
        return img.clamp(0, 1).cpu().permute(1, 2, 0).numpy()

    idx = 0
    img_clean = unnorm(clean[idx])
    img_noisy = unnorm(noisy[idx])
    img_rec = unnorm(sampled_clean[idx])

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img_clean)
    axs[0].set_title("Clean")
    axs[0].axis("off")

    axs[1].imshow(img_noisy)
    axs[1].set_title("Noisy (Input)")
    axs[1].axis("off")

    axs[2].imshow(img_rec)
    axs[2].set_title("Denoised (Output)")
    axs[2].axis("off")

    save_path = "cifar_denoising_demo.png"
    plt.savefig(save_path)
    print(f"Visualization saved to {os.path.abspath(save_path)}")


if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---------------------------------------------------------
# 1. Transport Solver (数学核心)
# ---------------------------------------------------------
class LinearSchedule:
    def alpha_in(self, t):
        return t

    def gamma_in(self, t):
        return 1 - t

    def alpha_to(self, t):
        return 1

    def gamma_to(self, t):
        return -1


class TransportSolver(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.sched = LinearSchedule()

    def get_params(self, t):
        t = t.view(-1, 1, 1, 1)
        return (self.sched.alpha_in(t), self.sched.gamma_in(t), self.sched.alpha_to(t), self.sched.gamma_to(t))

    def sample_location(self, x1, x0, t):
        a_in, g_in, _, _ = self.get_params(t)
        return g_in * x0 + a_in * x1

    def get_target_velocity(self, x1, x0, t):
        _, _, a_to, g_to = self.get_params(t)
        return a_to * x1 + g_to * x0

    def predict_x0_from_velocity(self, x_t, t, F_t):
        """
        从 x_t 和 velocity F_t 解算出 x0 (x_hat) 和 z (z_hat)
        """
        a_in, g_in, a_to, g_to = self.get_params(t)
        dent = a_in * g_to - g_in * a_to

        # 避免除以0的数值保护
        dent = torch.where(dent.abs() < 1e-5, -1.0 * torch.ones_like(dent), dent)

        x_hat = (F_t * a_in - x_t * a_to) / dent
        z_hat = (x_t * g_to - F_t * g_in) / dent
        return x_hat, z_hat


# ---------------------------------------------------------
# 2. Model with Class Conditioning (CFG Ready)
# ---------------------------------------------------------
class ConditionalUNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )

        # Class embedding (num_classes + 1 for null token)
        self.label_emb = nn.Embedding(num_classes + 1, 64)

        # Encoder
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=2, padding=1)

        self.bot1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bot2 = nn.Conv2d(128, 64, 3, padding=1)

        self.up1 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.up2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.out = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x, t, y):
        # Time Emb
        t = t.view(-1, 1).float()
        t_vec = self.time_mlp(t)

        # Class Emb
        y_vec = self.label_emb(y)  # (B, 64)

        # Combine (Simple addition)
        cond = t_vec + y_vec
        cond = cond.unsqueeze(-1).unsqueeze(-1)  # (B, 64, 1, 1)

        x1 = torch.relu(self.conv1(x))
        x2 = torch.relu(self.conv2(x1))
        x3 = torch.relu(self.conv3(x2))

        # Inject condition at bottleneck
        x3 = x3 + cond

        b = torch.relu(self.bot1(x3))
        b = torch.relu(self.bot2(b))

        x_up1 = torch.relu(self.up1(b + x3))
        x_up2 = torch.relu(self.up2(x_up1 + _upsample_like(x2, x_up1)))

        return self.out(x_up2)


def _upsample_like(src, target):
    return torch.nn.functional.interpolate(src, size=target.shape[2:], mode="bilinear", align_corners=False)


def get_mnist_loader(batch_size=128):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


# ---------------------------------------------------------
# 3. Training Utilities (Dropout & Loss)
# ---------------------------------------------------------
def apply_cfg_dropout(y, drop_prob=0.1, null_val=10):
    if drop_prob > 0:
        mask = torch.rand(y.shape[0], device=y.device) < drop_prob
        y_drop = y.clone()
        y_drop[mask] = null_val
        return y_drop
    return y


# ---------------------------------------------------------
# STAGE 1: Pre-training (With Labels)
# ---------------------------------------------------------
def train_pretrain(model, loader, device, epochs=3):
    print(f"\n>>> Starting Stage 1: Pre-training (Flow Matching w/ CFG support)...")
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    transport = TransportSolver(device)
    model.train()
    model.load_state_dict(torch.load("mnist_cfg_pretrained.pth", map_location=device))

    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # Label Dropout for CFG
            y_in = apply_cfg_dropout(y, drop_prob=0.1)

            z = torch.randn_like(x).to(device)
            t = torch.rand(x.shape[0], device=device)

            x_t = transport.sample_location(z, x, t)
            target_v = transport.get_target_velocity(z, x, t)

            # Forward with labels
            pred_v = model(x_t, t, y_in)
            loss = torch.mean((pred_v - target_v) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "mnist_cfg_pretrained.pth")


# ---------------------------------------------------------
# STAGE 2: TwinFlow Distillation (With CFG)
# ---------------------------------------------------------
def train_distill_unified(model, loader, device, epochs=3, lambda_val=1.0 / 3.0):
    print(f"\n>>> Starting Stage 2: TwinFlow Distillation (Conditional)...")

    model.load_state_dict(torch.load("mnist_cfg_pretrained.pth"))
    optimizer = optim.AdamW(model.parameters(), lr=2e-4)
    transport = TransportSolver(device)
    model.train()

    null_token = 10
    iter_idx = 0
    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            iter_idx += 1
            optimizer.zero_grad()
            batch_size = x.shape[0]
            x, y = x.to(device), y.to(device)

            # CFG Dropout for training
            y_in = apply_cfg_dropout(y, drop_prob=0.1, null_val=null_token)

            # --- Part A: Base Loss (Standard FM) ---
            # Paper uses N=2 RCGM, here simplified to N=0 (Standard FM)
            z = torch.randn_like(x).to(device)
            t_base = torch.rand(batch_size, device=device)
            x_t_base = transport.sample_location(z, x, t_base)
            target_v_base = transport.get_target_velocity(z, x, t_base)
            pred_v_base = model(x_t_base, t_base, y_in)
            loss_base = torch.mean((pred_v_base - target_v_base) ** 2)

            # --- Part B: TwinFlow Objectives ---

            # 1. Generate Fake Data (x_fake)
            # Use t=1 (pure noise). Note: We condition on y_in to generate class-specific fake data
            t_one = torch.ones(batch_size, device=device)
            z_input = torch.randn_like(x).to(device)

            with torch.no_grad():
                # Predict Velocity at t=1
                v_at_1 = model(z_input, t_one, y_in)
                # Solve for x_hat (x_fake)
                x_fake, _ = transport.predict_x0_from_velocity(z_input, t_one, v_at_1)

            # 2. Fake Trajectory (Negative Time)
            z_fake_noise = torch.randn_like(x).to(device)
            t_prime = torch.rand(batch_size, device=device)

            # Sample on Fake Trajectory: from z_fake_noise to x_fake
            x_t_fake = transport.sample_location(z_fake_noise, x_fake, t_prime)

            # 3. Self-Adversarial Loss (L_adv)
            # Input: negative time (-t_prime), conditioned on y_in
            # Target: Velocity pointing to x_fake
            target_v_fake = transport.get_target_velocity(z_fake_noise, x_fake, t_prime)
            pred_v_fake_neg = model(x_t_fake, -t_prime, y_in)
            loss_adv = torch.mean((pred_v_fake_neg - target_v_fake) ** 2)

            # 4. Rectification Loss (L_rectify)
            # Input: positive time (t_prime), conditioned on y_in
            # Target: The prediction from the negative branch (detached)
            # This aligns the "Real" interpretation with the "Fake" self-generated path
            pred_v_fake_pos = model(x_t_fake, t_prime, y_in)

            # Eq 9: Minimize distance between V_pos and sg[V_neg]
            # Note: The paper formulation uses sg[Delta_v + F_theta], which simplifies
            # to matching the endpoints or velocities if linear.
            # Matching velocities directly is the standard implementation for "straightening".
            loss_rectify = torch.mean((pred_v_fake_pos - pred_v_fake_neg.detach()) ** 2)

            # Total Loss
            loss_twin = loss_adv + loss_rectify
            loss = loss_base + lambda_val * loss_twin

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if iter_idx % 200 == 0:
                print(
                    f"Iter {iter_idx} | Loss base: {loss_base.item():.4f} | loss adv: {loss_adv.item():.4f} | loss rectify: {loss_rectify.item():.4f}"
                )

        print(f"Epoch {epoch + 1} | Loss: {total_loss / len(loader):.4f} ")

        # Save model
        torch.save(model.state_dict(), "mnist_cfg_distilled.pth")


# ---------------------------------------------------------
# 4. CFG Sampler (Unified Logic)
# ---------------------------------------------------------
@torch.no_grad()
def sample_cfg_unified(model, device, labels, steps=1, cfg_scale=4.0):
    """
    labels: tensor of shape (N,) with values 0-9
    """
    model.eval()
    transport = TransportSolver(device)
    num_samples = labels.shape[0]
    z = torch.randn(num_samples, 1, 28, 28).to(device)

    # Create null labels for unconditional prediction
    null_token = 10
    y_null = torch.ones_like(labels) * null_token

    t_steps = torch.linspace(1, 0, steps + 1).to(device)
    x_curr = z

    for i in range(steps):
        t_curr = t_steps[i].repeat(num_samples)
        t_next = t_steps[i + 1].repeat(num_samples)

        # --- CFG Prediction ---
        # 1. Conditional Prediction
        v_cond = model(x_curr, t_curr, labels)
        # 2. Unconditional Prediction
        v_uncond = model(x_curr, t_curr, y_null)

        # 3. Guidance
        v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)

        # --- Unified Sampler Step ---
        # 1. Solve for x_hat using Guided Velocity
        x_hat, z_hat = transport.predict_x0_from_velocity(x_curr, t_curr, v_pred)

        # 2. Mix to next step
        a_next, g_next, _, _ = transport.get_params(t_next)
        x_next = g_next * x_hat + a_next * z_hat

        x_curr = x_next

    return x_curr.cpu()


def show_results_conditional(real, pre_1step, pre_ode, distilled_1step):
    # Helper to visualize grid
    fig, axs = plt.subplots(4, 10, figsize=(10, 5))
    titles = ["Real", "Pre (1-step)", "Pre (20-step)", "TwinFlow (1-step)"]
    img_lists = [real, pre_1step, pre_ode, distilled_1step]

    for i, imgs in enumerate(img_lists):
        for j in range(10):
            if j < len(imgs):
                axs[i, j].imshow(imgs[j].squeeze(), cmap="gray")
                axs[i, j].axis("off")
            if j == 0:
                axs[i, j].text(-5, 14, titles[i], fontsize=8, ha="right")
    plt.tight_layout()
    # plt.show()
    plt.savefig("ImagesOutput.png")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Setup
    loader = get_mnist_loader(batch_size=128)
    model = ConditionalUNet(num_classes=10).to(device)  # 0-9 labels + 1 null

    # 1. Pretrain
    train_pretrain(model, loader, device, epochs=3)

    # Prepare Labels for visualization (0 to 9)
    vis_labels = torch.arange(10).long().to(device)
    real_batch = next(iter(loader))[0][:10]  # Just random reals for shape

    # Baseline Sampling
    print("Sampling Baselines...")
    pre_1step = sample_cfg_unified(model, device, vis_labels, steps=1, cfg_scale=4.0)
    pre_ode = sample_cfg_unified(model, device, vis_labels, steps=20, cfg_scale=4.0)

    # 2. Distill
    train_distill_unified(model, loader, device, epochs=100, lambda_val=1.0)

    # Distilled Sampling
    print("Sampling Distilled...")
    distilled_1step = sample_cfg_unified(model, device, vis_labels, steps=4, cfg_scale=4.5)

    show_results_conditional(real_batch, pre_1step, pre_ode, distilled_1step)

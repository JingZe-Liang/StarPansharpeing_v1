import math
from einops import rearrange, pack
import torch
from torch import Tensor, nn
from torch.nn import functional as F
import torchvision
import torchvision.datasets.mnist as mnist
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from .plan import SDBContinuousPlan, SDBContinuousSampler, DiffusionTarget
from .precond import EDMPrecond


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, y: Tensor | None = None):
        identity = x
        x = self.relu(self.bn(self.conv1(x)))
        x = self.conv2(x)
        if y is not None:
            x = x + y
        x = self.bn2(x) + identity
        x = self.relu(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(t.dtype))
        return t_emb


# Define simple CNN model
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual blocks
        self.res1 = self._make_res_block(64, 64, 2)
        self.res2 = self._make_res_block(64, 64, 2)
        self.res3 = self._make_res_block(64, 64, 2)
        self.timestep_embedding = TimestepEmbedder(64)
        self.class_embedding = nn.Embedding(10, 64)

        # Final convolution
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def _make_res_block(self, in_channels, out_channels, depth):
        stage = []
        for _ in range(depth):
            stage.append(ResBlock(in_channels, out_channels))
        return nn.Sequential(*stage)

    def forward(self, img, timesteps, **kwargs):
        # Input shape: (batch, 1, 28, 28)
        x = img
        t = timesteps

        t = self.timestep_embedding(t.squeeze())
        c = (
            self.class_embedding(kwargs["y"].squeeze())
            if "y" in kwargs
            else torch.zeros((x.shape[0], 64), device=x.device).type_as(img)
        )

        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))
        x = x + rearrange(t, "b d -> b d 1 1")
        c = rearrange(c, "b d -> b d 1 1")

        # Residual blocks
        stages = [self.res1, self.res2, self.res3]
        for stage in stages:
            for resblock in stage:
                x = resblock(x, c)

        # Final conv
        x = self.conv2(x)

        return x


# Load MNIST dataset
def load_mnist(batch_size=128, stage="train"):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            lambda x: x * 2 - 1,
        ]
    )

    # Load full datasets
    dataset = mnist.MNIST(
        root="/Data4/cao/ZiHanCao/exps/panformer/model/flux/SDB_diffusion/data",
        train=True if stage == "train" else False,
        download=True,
        transform=transform,
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def load_mnist_transfer(batch_size=128, stage="train"):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            lambda x: x * 2 - 1,
        ]
    )

    # Load full datasets
    dataset = mnist.MNIST(
        root="/Data4/cao/ZiHanCao/exps/panformer/model/flux/SDB_diffusion/data",
        train=True if stage == "train" else False,
        download=True,
        transform=transform,
    )

    train_idx_0 = dataset.targets == 8
    train_idx_1 = dataset.targets == 1

    train_dataset_0 = torch.utils.data.Subset(dataset, train_idx_0.nonzero().squeeze())
    train_dataset_1 = torch.utils.data.Subset(dataset, train_idx_1.nonzero().squeeze())

    len_train = min(len(train_dataset_0), len(train_dataset_1))

    class TransferDataset(torch.utils.data.Dataset):
        def __init__(self, d0, d1):
            self.d0 = d0
            self.d1 = d1

        def __getitem__(self, idx):
            return self.d0[idx][0], self.d1[idx][0]

        def __len__(self):
            return len_train

    return DataLoader(TransferDataset(train_dataset_0, train_dataset_1), batch_size=batch_size, shuffle=True)


def main_transfer():
    # Hyperparameters
    batch_size = 128
    num_epochs = 200
    learning_rate = 4e-4

    # Load data
    train_loader = load_mnist_transfer(batch_size, stage="train")
    test_loader = load_mnist_transfer(batch_size, stage="test")

    # Initialize model and components
    model = ResNet().cuda()
    plan = SDBContinuousPlan(
        plan_tgt=DiffusionTarget.x_0,
        t_train_type="uniform",
        t_sample_type="uniform",
        alpha_beta_type="linear",
        diffusion_type="bridge",
        gamma_max=1.0,
        eps_eta=1.0,
    )
    sampler = SDBContinuousSampler(plan, 0)
    precond_model = EDMPrecond(model, plan)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()

        # Sampling
        if epoch % 3 == 0 and epoch != 0:
            model.eval()
            with torch.no_grad():
                # Generate samples
                x_0, x_1 = next(iter(test_loader))  # 8, 1
                x_1 = x_1.cuda()

                n_steps = 80
                sample_t_dict = {
                    "uniform": dict(n_timesteps=n_steps),
                    "edm": dict(rho=0.3, n_timesteps=n_steps, t_min=1e-4, t_max=1 - 1e-4),
                    "sigmoid": dict(k=7, n_timesteps=n_steps, t_min=1e-4, t_max=1 - 1e-4),
                }[plan.t_sample_type]
                time_grid = plan.sample_continous_t(**sample_t_dict).to(x_1)
                samples, saved_x0_traj, saved_xt_traj = sampler.sample_sde_euler(
                    model=precond_model,
                    x_1=x_1,
                    time_grid=time_grid,
                    clip_value=False,
                    traj_saved_n=10,
                )

                # Save samples
                samples = (samples + 1) / 2
                samples.clip_(0, 1)
                save_image(samples, "samples.png")

                # Save x0 traj
                def save_grid(traj, name):
                    n_traj = len(traj)
                    traj = torch.stack(traj, dim=0)
                    traj = rearrange(traj[:, :8], "traj bs ... -> (bs traj) ...")
                    traj_grid = make_grid(traj, nrow=n_traj)
                    save_image(traj_grid.add_(1).mul_(0.5).clip_(0, 1), name)

                save_grid(saved_x0_traj, "x0_traj.png")
                save_grid(saved_xt_traj, "xt_traj.png")

        for batch_idx, (x_0, x_1) in enumerate(train_loader):
            x_0 = x_0.cuda()  # 8
            x_1 = x_1.cuda()  # 1

            optimizer.zero_grad()

            # Sample random timesteps
            t = plan.train_continous_t(x_0.shape[0])
            t = plan.expand_t_as(t, x_0)
            x_t, target = plan.get_x_t_with_target(t, x_0, x_1)

            # Forward pass through preconditioned model
            pred_x0, _, weight = precond_model(x_t, t)

            # Compute loss
            loss = ((pred_x0 - target).pow(2) * weight).mean()

            # Backward pass
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(x_0)}/{len(train_loader.dataset)}]"
                    f"\tLoss: {loss.item():.6f}"
                )


def main_generate():
    # Hyperparameters
    batch_size = 128
    num_epochs = 100
    learning_rate = 1e-4

    # Load data
    train_loader = load_mnist(batch_size, stage="train")
    test_loader = load_mnist(batch_size, stage="test")

    # Initialize model and components
    model = ResNet().cuda()
    plan = SDBContinuousPlan(plan_tgt=DiffusionTarget.x_0, t_train_type="uniform", gamma_max=1.0, eps_eta=0.8)
    sampler = SDBContinuousSampler(plan)
    precond_model = EDMPrecond(model, plan)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()

        # Sampling
        if epoch % 1 == 0 and epoch != 0:
            model.eval()
            with torch.no_grad():
                # Generate samples
                # x_0, _ = next(iter(test_loader))  # 1, 9
                x_1 = torch.randn(5 * 10, 1, 28, 28).cuda()
                y = torch.arange(0, 10, device=x_1.device).repeat(5).cuda()

                time_grid = plan.sample_continous_t(rho=7, n_timesteps=100, t_min=0.001, t_max=1 - 1e-4)
                samples, _ = sampler.sample_sde_euler(
                    model=precond_model,
                    x_1=x_1,
                    time_grid=time_grid,
                    model_kwargs={"y": y},
                )

                # Save samples
                samples = (samples + 1) / 2
                torchvision.utils.save_image(samples, "samples.png", nrow=10)

        for batch_idx, (x_0, y) in enumerate(train_loader):
            x_0 = x_0.cuda()
            y = y.cuda()

            optimizer.zero_grad()

            # Sample random timesteps
            t = plan.train_continous_t(x_0.shape[0])
            t = plan.expand_t_as(t, x_0)
            x_t, target = plan.get_x_t_with_target(t, x_0)

            # Forward pass through preconditioned model
            pred_x0, weight = precond_model(x_t, t, y=y)

            # Compute loss
            loss = ((pred_x0 - target).pow(2) * weight).mean()

            # Backward pass
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(x_0)}/{len(train_loader.dataset)}]"
                    f"\tLoss: {loss.item():.6f}"
                )


if __name__ == "__main__":
    torch.cuda.set_device(0)
    main_type = "transfer"  # ['transfer', 'generate']
    if main_type == "transfer":
        main_transfer()
    elif main_type == "generate":
        main_generate()
    else:
        raise ValueError(f"Invalid main type: {main_type}")

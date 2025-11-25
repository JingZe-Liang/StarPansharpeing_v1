import torch
import torch.nn as nn

from src.utilities.logging import print_info_if_raise
from src.utilities.optim import Dion, MuonAll2All


@print_info_if_raise(True)
def test_dion_optimizer():
    # Create a simple model and optimizer
    model = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.Flatten(1), nn.Linear(32 * 8 * 8, 10))
    x = torch.randn(1, 3, 16, 16)
    y = model(x)

    param_group = [
        {
            "params": [p for p in model.parameters() if p.ndim >= 2],
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "algorithm": "dion",
        },
        {
            "params": [p for p in model[1].parameters() if p.ndim < 2],
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "algorithm": "lion",
        },
    ]

    optimizer = Dion(param_group, lr=0.01)

    out = model(x)
    out.mean().backward()

    # Test optimizer step
    optimizer.step()

    # Test optimizer zero_grad
    optimizer.zero_grad()


def test_muon_optimizer():
    # Create a simple model and optimizer
    model = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.Flatten(1), nn.Linear(32 * 8 * 8, 10))
    x = torch.randn(1, 3, 16, 16)

    param_group = [
        {
            "params": [p for p in model.parameters() if p.ndim >= 2],
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "algorithm": "muon",
        },
        {
            "params": [p for p in model[1].parameters() if p.ndim < 2],
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "algorithm": "lion",
        },
    ]

    optimizer = MuonAll2All(param_group, lr=0.02, adjust_lr="spectral_norm", flatten=True)

    out = model(x)
    out.mean().backward()

    # Test optimizer step
    print("optimizer step")
    optimizer.step()

    # Test optimizer zero_grad
    optimizer.zero_grad()
    print("Muon optimization done")


if __name__ == "__main__":
    test_dion_optimizer()
    # test_muon_optimizer()

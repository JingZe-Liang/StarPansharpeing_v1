import numpy as np
import torch
import torch.nn as nn


def _to_two_tuple(x):
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, tuple):
        assert len(x) == 2, "x should be a tuple of length 2"
        return x
    else:
        raise ValueError("x should be an int or a tuple of length 2")


class NestChannelDrop(nn.Module):
    def __init__(
        self,
        drop_prob: float,
        drop_dim: int = 1,
        learnable: bool = False,
        drop_type: str = "exp_1",
        max_channels: int = 12,  # MMSeg dataset
        img_size: tuple[int] | int = 256,
    ):
        super().__init__()
        self.drop_prob = drop_prob
        self.drop_dim = drop_dim
        self.learnable = learnable
        self.max_channels = max_channels
        self.img_size = _to_two_tuple(img_size)

        drop_type, args = drop_type.lower().split("_")
        self.drop_type = drop_type
        if drop_type == "exp":
            self.sample_kwargs = {"lambda": float(args)}
        elif drop_type == "uniform":
            assert args.isdigit(), "args should be an int"
            self.sample_kwargs = {"low": int(args)}
        else:
            raise ValueError(
                f"drop_type {drop_type} not supported, only exp and uniform are supported"
            )

        if self.learnable:
            self.dropped_x = nn.Parameter(torch.zeros(1, 1, *self.img_size))
            self.dropped_x.data.normal_(0, 0.2)
        else:
            self.register_buffer("dropped_x", torch.zeros(1, 1, *self.img_size))

        self.register_buffer("channel_arange", torch.arange(self.max_channels))

    def exponential_sampling(self, lambda_val, size=1):
        u = np.random.uniform(size=size)
        k = -np.log(1 - u) / lambda_val
        return (
            torch.as_tensor(np.floor(k).astype(int))
            .clip_(0, self.max_channels)
            .unsqueeze(-1)
        )

    def uniform_sampling(self, low: int, size=1):
        # (bs, 1)
        k = torch.randint(low=low, high=self.max_channels, size=(size, 1))

        return k

    def forward(self, z, inference_channels: int | None = None):
        if inference_channels is not None:
            assert not self.training
            assert inference_channels <= self.max_channels
            return z[:, :inference_channels]

        assert self.max_channels == z.shape[1]

        bs = z.shape[0]
        if self.drop_type == "exp":
            leave_channels = self.exponential_sampling(size=bs, **self.sample_kwargs)
        elif self.drop_type == "uniform":
            leave_channels = self.uniform_sampling(size=bs, **self.sample_kwargs)

        # drop channels

        # 1. expand the cached empty z
        z_empty = self.dropped_x.expand(bs, -1, -1, -1)

        _channels = self.channel_arange[None].expand(bs, -1)
        _cond = _channels < leave_channels.to(_channels)
        z = torch.where(_cond.unsqueeze(-1).unsqueeze(-1).expand_as(z), z, z_empty)

        return z


if __name__ == "__main__":
    z = torch.randn(2, 12, 256, 256)
    drop = NestChannelDrop(
        0.5, drop_dim=1, learnable=False, drop_type="uniform_3", max_channels=12
    )
    drop.train()

    z = drop(z)
    print(z.shape)

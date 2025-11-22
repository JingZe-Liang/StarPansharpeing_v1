"""
Taken from https://github.com/rbalestr-lab/lejepa

LeJEPA augmentations:
    - Global views with local views
    - Compatible with Sketched Isotropic Gaussian Regularization (SIGReg)

Augmentations include:
    For global views:
        RandomResizedCrop
            - Resolution: 224x224
            - Scale: (0.3, 1.0)
            - Covers 30-100% of the image
        RandomHorizontalFlip (p=0.5)
        ColorJitter (p=0.8)
            - Brightness: 0.4
            - Contrast: 0.4
            - Saturation: 0.2
            - Hue: 0.1
        RandomGrayscale (p=0.2)
        GaussianBlur (p=0.5)
        RandomSolarize (p=0.2, threshold=128)
        Normalization (mean, std)

    For local views:
        RandomResizedCrop
            - Resolution: 98x98
            - Scale: (0.05, 0.3)
            - Covers 5-30% of the image
        RandomHorizontalFlip (p=0.5)
        ColorJitter (p=0.8)
            - Brightness: 0.4
            - Contrast: 0.4
            - Saturation: 0.2
            - Hue: 0.1
        RandomGrayscale (p=0.2)
        GaussianBlur (p=0.5)
        RandomSolarize (p=0.2, threshold=128)
        Normalization (mean, std)

Each training image is augmented to produce 2 global views and 6 local views with
different spatial scales but the same set of color and geometric transformations
"""

import kornia.augmentation as K
import torch
from easydict import EasyDict as edict
from jaxtyping import Float
from torch import Tensor
from torch import distributed as dist
from torch.distributed.nn import ReduceOp
from torch.distributed.nn import all_reduce as functional_all_reducef

# TODO: add some hyperspectral-data-specific augmentations.


def create_global_view_augmentations():
    return K.AugmentationSequential(
        K.RandomResizedCrop(size=(224, 224), scale=(0.3, 1.0), p=1.0),
        K.RandomHorizontalFlip(p=0.5),
        K.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
        K.RandomGrayscale(p=0.2),
        K.RandomGaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0), p=0.5),
        K.RandomSolarize(thresholds=0.1, p=0.2),
        # K.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        data_keys=["input"],
    )


def create_local_view_augmentations():
    return K.AugmentationSequential(
        K.RandomResizedCrop(size=(98, 98), scale=(0.05, 0.3), p=1.0),
        K.RandomHorizontalFlip(p=0.5),
        K.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.2,
        ),
        K.RandomGrayscale(p=0.2),
        K.RandomGaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0), p=0.5),
        K.RandomSolarize(thresholds=0.1, p=0.2),
        # K.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        data_keys=["input"],
    )


def forward_augmentation_pipelines(
    x: torch.Tensor,
    gv_pipe: K.AugmentationSequential,
    lv_pipe: K.AugmentationSequential,
    n_locals: int = 6,
    n_globals: int = 2,
):
    global_views = []
    local_views = []

    for _ in range(n_globals):
        global_views.append(gv_pipe(x))

    for _ in range(n_locals):
        local_views.append(lv_pipe(x))

    return global_views, local_views


class LeJEPAAugmentation:
    def __init__(
        self,
        n_locals: int = 6,
        n_globals: int = 2,
        stack: bool = False,
        global_view_pipe=None,
        local_view_pipe=None,
    ):
        self.n_locals = n_locals
        self.n_globals = n_globals
        self.stack = stack

        self.global_view_pipe = (
            global_view_pipe
            if global_view_pipe is not None
            else create_global_view_augmentations()
        )
        self.local_view_pipe = (
            local_view_pipe
            if local_view_pipe is not None
            else create_local_view_augmentations()
        )

    def __call__(self, x: torch.Tensor):
        global_views, local_views = forward_augmentation_pipelines(
            x,
            self.global_view_pipe,
            self.local_view_pipe,
            self.n_locals,
            self.n_globals,
        )

        if self.stack:
            global_views = torch.stack(global_views)
            local_views = torch.stack(local_views)

        return global_views, local_views


def _all_reduce(x: torch.Tensor, op: str = "AVG") -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        reduce_op = ReduceOp.__dict__[op.upper()]
        return functional_all_reduce(x, reduce_op)
    return x


class SIGReg(torch.nn.Module):
    def __init__(self, knots=17, rnd_proj_dim: int = 256):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.rnd_proj_dim = rnd_proj_dim
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: [V, N, C]
        """
        N = proj.size(-2)

        # Random projection
        if self.rnd_proj_dim is not None:
            A = torch.randn(
                proj.size(-1), self.rnd_proj_dim, device=proj.device
            )  # [D, projD]
            A = A.div_(A.norm(p=2, dim=0))
            x_t = (proj @ A).unsqueeze(-1) * self.t
        # No projection
        else:
            x_t = proj.unsqueeze(-1) * self.t

        # [V, N, D] @ [D, projD]; [V, N, projD, 1] * [K,] -> [V, N, 256, K]
        # mean across batch dimension then DDP all_reduce
        cos_mean = x_t.cos().mean(-3)
        sin_mean = x_t.sin().mean(-3)

        cos_mean = _all_reduce(cos_mean, op="AVG")
        sin_mean = _all_reduce(sin_mean, op="AVG")

        err = (cos_mean - self.phi).square() + sin_mean.square()
        statistic = (err @ self.weights) * proj.size(-2)

        return statistic.mean()


class _MeanOutHW(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        if x.ndim == 4:
            return x.mean(dim=(-2, -1))
        elif x.ndim == 3:
            # b, l, c
            return x.mean(dim=-2)


def create_lejepa_projector(input_dim: int, proj_dim: int = 128, mean_out_hw=False):
    from torchvision.ops import MLP

    mlp_projector = MLP(
        input_dim, [2048, 2048, proj_dim], norm_layer=torch.nn.BatchNorm1d
    )
    if mean_out_hw:
        return torch.nn.Sequential(_MeanOutHW(), mlp_projector)
    return mlp_projector


def lejepa_loss(
    global_emb: Float[Tensor, "V B D"],
    local_emb: Float[Tensor, "V B D"] | None = None,
    sigreg: torch.nn.Module | SIGReg | None = None,
    lam: float = 0.02,
) -> tuple[Tensor, dict]:
    """
    from the minimal lejepa implementation.

    embeddings: Tensor of shape [n_views, batch_size, feature_dim]
    """
    ng_views, batch_size, feature_dim = global_emb.shape
    if local_emb is not None:
        nl_views, *_ = local_emb.shape
        embeddings = torch.cat([global_emb, local_emb], dim=0)
    else:
        embeddings = global_emb

    if sigreg is not None:
        sigreg = SIGReg().to(global_emb.device)

    # Regularization loss for lejepa
    sigreg_loss = sigreg(embeddings)  # proj: (V, B, D)

    # Center of the views
    centers = global_emb.mean(0, keepdim=True)

    # Losses
    inv_loss = (centers - embeddings).square().mean()

    # Total loss and loss breakdowns
    loss = inv_loss * (1 - lam) + lam * sigreg_loss
    breakdowns = edict(
        {
            "inv_loss": inv_loss,
            "sigreg_loss": sigreg_loss,
            "lejepa_loss": loss,
        }
    )

    return loss, breakdowns


def __test_aug_pipe():
    from einops import rearrange, reduce

    from .lejepa.lejepa.univariate.epps_pulley import EppsPulley

    aug = LeJEPAAugmentation()

    # network = torch.nn.Conv2d(3, 384, 3, 1, 1)

    x = torch.randn(8, 128, 224, 224)  # Example input tensor
    global_views, local_views = aug(x)

    # print(
    #     f"Generated {len(global_views)} global views and {len(local_views)} local views."
    # )

    # # Forward the networks
    # bs, c, h, w = x.shape
    # spatial_size = h * w
    # global_emb = network(torch.cat(global_views, dim=0))  # (views*bs, c, h, w)
    # local_emb = network(torch.cat(local_views, dim=0))

    # # Fake model output, assume the model mean out the hw dimensions
    # global_emb = reduce(global_emb, "(v b) c h w -> v b c", "mean", v=len(global_views))
    # local_emb = reduce(local_emb, "(v b) c h w -> v b c", "mean", v=len(local_views))

    # all_emb = torch.cat([global_emb, local_emb], dim=0)  # [views, bs, c]

    # center = global_emb.mean(0, True)  # [1, bs, c]
    # sim = (center - all_emb).square().mean()

    # sigreg = EppsPulley(t_range=(-5, 5), n_points=512)
    # sigreg_loss = torch.stack([sigreg(emb) for emb in all_emb]).mean()

    # print(f"Similarity loss: {sim:.4f}")
    # print(f"SIGReg loss: {sigreg_loss:.4f}")


if __name__ == "__main__":
    """
    python -m src.stage1.self_supervised.lejepa_aug
    """
    __test_aug_pipe()

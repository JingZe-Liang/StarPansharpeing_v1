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

from .lejepa.lejepa.univariate.epps_pulley import EppsPulley

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
        global_view_pipe=None,
        local_view_pipe=None,
    ):
        self.n_locals = n_locals
        self.n_globals = n_globals

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

    def __call__(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        return forward_augmentation_pipelines(
            x,
            self.global_view_pipe,
            self.local_view_pipe,
            self.n_locals,
            self.n_globals,
        )


def __test_aug_pipe():
    from einops import rearrange, reduce

    aug = LeJEPAAugmentation()

    network = torch.nn.Conv2d(3, 384, 3, 1, 1)

    x = torch.randn(8, 3, 224, 224)  # Example input tensor
    global_views, local_views = aug(x)

    print(
        f"Generated {len(global_views)} global views and {len(local_views)} local views."
    )

    # Forward the networks
    bs, c, h, w = x.shape
    spatial_size = h * w
    global_emb = network(torch.cat(global_views, dim=0))  # (views*bs, c, h, w)
    local_emb = network(torch.cat(local_views, dim=0))

    global_emb = reduce(global_emb, "(v b) c h w -> v b c", "mean", v=len(global_views))
    local_emb = reduce(local_emb, "(v b) c h w -> v b c", "mean", v=len(local_views))

    all_emb = torch.cat([global_emb, local_emb], dim=0)  # [views, bs, c]

    center = global_emb.mean(0, True)  # [1, bs, c]
    sim = (center - all_emb).square().mean()

    sigreg = EppsPulley(t_range=(-5, 5), n_points=512)
    sigreg_loss = torch.stack([sigreg(emb) for emb in all_emb]).mean()

    print(f"Similarity loss: {sim:.4f}")
    print(f"SIGReg loss: {sigreg_loss:.4f}")


if __name__ == "__main__":
    """
    python -m src.stage1.self_supervised.lejepa_aug
    """
    __test_aug_pipe()

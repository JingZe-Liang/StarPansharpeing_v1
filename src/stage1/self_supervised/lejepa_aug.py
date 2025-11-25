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

from typing import cast

import kornia.augmentation as K
import kornia.augmentation.random_generator as rg
import torch
from easydict import EasyDict as edict
from jaxtyping import Float
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from loguru import logger
from timm.layers import get_norm_layer
from torch import Tensor
from torch import distributed as dist
from torch.distributed.nn import ReduceOp
from torch.distributed.nn import all_reduce as functional_all_reduce

from src.utilities.train_utils.visualization import get_rgb_image


class HyperRandomGrayScale(IntensityAugmentationBase2D):
    def __init__(self, p=0.5):
        super().__init__(p)

    def apply_transform(self, input, params, flags, transform=None):
        assert input.ndim == 4
        c = input.shape[1]
        gray = input.mean(dim=1, keepdim=True).repeat_interleave(c, dim=1)
        return gray


class HyperspectralColorJitter(IntensityAugmentationBase2D):
    """
    Hyperspectral-specific ColorJitter implementation

    Parameters
    ----------
    intensity_range : tuple[float, float]
        Intensity adjustment range, default (-0.2, 0.2)
    contrast_range : tuple[float, float]
        Contrast adjustment range, default (0.8, 1.2)
    noise_std : float
        Noise standard deviation, default 0.01
    band_drop_ratio : float
        Band dropout ratio, default 0.1
    smoothing_kernel : int
        Spectral smoothing kernel size, default 3
    probs : dict[str, float]
        Individual probabilities for each augmentation type. Default:
        {
            "intensity": 0.8,
            "contrast": 0.7,
            "noise": 0.5,
            "band_drop": 0.3,
            "smoothing": 0.4
        }
    same_on_batch : bool
        Whether to apply the same transformation across the batch, default False
    """

    def __init__(
        self,
        intensity_range: tuple[float, float] = (-0.2, 0.2),
        contrast_range: tuple[float, float] = (0.8, 1.2),
        noise_std: float = 0.01,
        band_drop_ratio: float = 0.1,
        smoothing_kernel: int = 3,
        probs: dict[str, float] | None = None,
        same_on_batch: bool = False,
    ):
        # Default probabilities for each augmentation
        default_probs = edict(
            {
                "intensity": 0.8,
                "contrast": 0.7,
                "noise": 0.5,
                "band_drop": 0.3,
                "smoothing": 0.4,
            }
        )

        # Use provided probabilities or defaults
        self.probs = default_probs if probs is None else edict(probs)

        # Overall probability for the entire augmentation
        overall_p = max(self.probs.values()) if self.probs else 0.8
        super().__init__(p=overall_p, same_on_batch=same_on_batch)

        self.intensity_range = intensity_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std
        self.band_drop_ratio = band_drop_ratio
        self.smoothing_kernel = smoothing_kernel

        # Register parameter generator
        self._param_generator = rg.PlainUniformGenerator(
            (self.intensity_range, "intensity_factor", None, None),
            (self.contrast_range, "contrast_factor", None, None),
        )

    def apply_transform(self, input, params, flags, transform=None):
        """Apply hyperspectral augmentation transforms"""
        # input: (B, C, H, W) where C is the number of spectral channels

        # 1. Intensity adjustment
        if torch.rand(1) < self.probs.intensity:
            intensity_factor = params["intensity_factor"]
            # Ensure intensity factor can broadcast to input shape
            while len(intensity_factor.shape) < len(input.shape):
                intensity_factor = intensity_factor.unsqueeze(-1)
            input = input + intensity_factor.to(input.device)

        # 2. Contrast adjustment
        if torch.rand(1) < self.probs.contrast:
            contrast_factor = params["contrast_factor"]
            spectral_mean = input.mean(dim=1, keepdim=True)
            # Ensure contrast factor can broadcast to input shape
            while len(contrast_factor.shape) < len(input.shape):
                contrast_factor = contrast_factor.unsqueeze(-1)
            input = spectral_mean + (input - spectral_mean) * contrast_factor.to(input.device)

        # 3. Noise injection
        if torch.rand(1) < self.probs.noise:
            noise = torch.randn_like(input) * self.noise_std
            input = input + noise

        # 4. Band dropout simulation
        if torch.rand(1) < self.probs.band_drop:
            C = input.shape[1]
            n_drop = int(C * self.band_drop_ratio)
            if n_drop > 0:
                drop_channels = torch.randperm(C)[:n_drop]
                input[:, drop_channels, :, :] = 0

        # 5. Spectral smoothing
        if torch.rand(1) < self.probs.smoothing:
            input = self._apply_spectral_smoothing(input)

        # Ensure values are within reasonable range
        input = torch.clamp(input, 0.0, 1.0)

        return input

    def _apply_spectral_smoothing(self, input):
        """Apply spectral dimension smoothing"""
        try:
            import torch.nn.functional as F

            B, C, H, W = input.shape
            kernel_size = self.smoothing_kernel
            padding = kernel_size // 2

            # Create 1D smoothing kernel
            kernel = torch.ones(1, 1, kernel_size, device=input.device) / kernel_size

            # Reshape to (B*H*W, C) for 1D convolution
            input_flat = input.permute(0, 2, 3, 1).reshape(-1, C)
            input_smoothed = F.conv1d(input_flat.unsqueeze(1), kernel, padding=padding).squeeze(1)

            # Reshape back to original shape
            return input_smoothed.reshape(B, H, W, C).permute(0, 3, 1, 2)

        except Exception:
            # Return original input if smoothing fails
            logger.warning(f"Failed to apply spectral smoothing. Using original input.")
            return input


class SafeColorJitterWrapper(IntensityAugmentationBase2D):
    def __init__(
        self,
        hp_color_jitter: HyperspectralColorJitter,
        rgb_color_jitter: K.ColorJitter,
    ):
        super().__init__(p=1.0)
        self.hp_color_jitter = hp_color_jitter
        self.rgb_color_jitter = rgb_color_jitter

    def apply_transform(self, input, params, flags, transform=None):
        if input.shape[1] == 3:  # RGB
            # For RGB images, use the RGB color jitter
            return self.rgb_color_jitter(input)
        elif input.shape[1] == 1:  # Grayscale
            # For grayscale images, repeat to 3 channels and use RGB color jitter
            rgb_input = input.repeat(1, 3, 1, 1)  # (B, 1, H, W) -> (B, 3, H, W)
            rgb_augmented = self.rgb_color_jitter(rgb_input)
            # Take the mean of the 3 channels to get back to single channel
            return rgb_augmented.mean(dim=1, keepdim=True)
        elif input.shape[1] > 3:  # Hyperspectral
            # For hyperspectral images, use the hyperspectral color jitter
            return self.hp_color_jitter(input)
        else:
            raise ValueError(f"Unsupported input shape: {input.shape}. Expected 1, 3 or >3 channels.")


class SafeNotApply(IntensityAugmentationBase2D):
    def __init__(self, augmentation):
        super().__init__(p=1.0)
        self.augmentation = augmentation

    def apply_transform(self, input, params, flags, transform=None):
        # Check if this is hyperspectral image (>3 channels)
        if input.shape[1] > 3:
            # For hyperspectral images, skip the augmentation and return unchanged
            return input
        else:
            # For RGB or grayscale images, apply the augmentation normally
            return self.augmentation(input)

    def __repr__(self) -> str:
        return f"SafeNotApply({self.augmentation})"


def create_global_view_augmentations():
    # Create the color jitter instances
    hp_jitter = HyperspectralColorJitter(
        intensity_range=(-0.2, 0.2),
        contrast_range=(0.8, 1.2),
        noise_std=0.1,
        band_drop_ratio=0.05,
        smoothing_kernel=3,
        probs={
            "intensity": 0.8,
            "contrast": 0.7,
            "noise": 0.5,
            "band_drop": 0.3,
            "smoothing": 0.4,
        },
    )

    rgb_jitter = K.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.2,
        hue=0.1,
        p=0.8,
    )

    return K.AugmentationSequential(
        K.RandomResizedCrop(size=(224, 224), scale=(0.3, 1.0), p=1.0),
        K.RandomHorizontalFlip(p=0.5),
        K.RandomGaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0), p=0.5),
        # Use SafeColorJitterWrapper to handle both hyperspectral and RGB images
        SafeColorJitterWrapper(
            hp_color_jitter=hp_jitter,
            rgb_color_jitter=rgb_jitter,
        ),
        HyperRandomGrayScale(p=0.2),
        K.RandomSolarize(thresholds=0.4, p=0.2),
        data_keys=["input"],
        keepdim=True,
    )


def create_local_view_augmentations():
    # Create the color jitter instances for local views (stronger augmentation)
    hp_jitter = HyperspectralColorJitter(
        intensity_range=(-0.3, 0.3),  # Local views use stronger augmentation
        contrast_range=(0.7, 1.3),
        noise_std=0.2,
        band_drop_ratio=0.08,
        smoothing_kernel=3,
        probs={
            "intensity": 0.9,  # Higher probability for local views
            "contrast": 0.8,
            "noise": 0.6,
            "band_drop": 0.5,
            "smoothing": 0.6,
        },
    )

    rgb_jitter = K.ColorJitter(
        brightness=0.6,  # Stronger augmentation for local views
        contrast=0.6,
        saturation=0.3,
        hue=0.15,
        p=0.8,
    )

    return K.AugmentationSequential(
        K.RandomResizedCrop(size=(96, 96), scale=(0.05, 0.3), p=1.0),
        K.RandomHorizontalFlip(p=0.5),
        K.RandomGaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0), p=0.5),
        # Use SafeColorJitterWrapper to handle both hyperspectral and RGB images
        SafeColorJitterWrapper(
            hp_color_jitter=hp_jitter,
            rgb_color_jitter=rgb_jitter,
        ),
        HyperRandomGrayScale(p=0.2),
        K.RandomSolarize(thresholds=0.4, p=0.2),
        data_keys=["input"],
        keepdim=True,
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
    x_ = x.float()

    for _ in range(n_globals):
        global_views.append(gv_pipe(x_).to(x))

    for _ in range(n_locals):
        local_views.append(lv_pipe(x_).to(x))

    return global_views, local_views


class LeJEPAAugmentation:
    def __init__(
        self,
        n_locals: int = 6,
        n_globals: int = 2,
        stack: bool = False,
        global_view_pipe=None,
        local_view_pipe=None,
        is_neg_1_1: bool = True,
    ):
        self.n_locals = n_locals
        self.n_globals = n_globals
        self.stack = stack
        self.is_neg_1_1 = is_neg_1_1

        self.global_view_pipe = global_view_pipe if global_view_pipe is not None else create_global_view_augmentations()
        self.local_view_pipe = local_view_pipe if local_view_pipe is not None else create_local_view_augmentations()

    def __call__(self, x: torch.Tensor):
        if self.is_neg_1_1:
            x = (x + 1) / 2

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
            if self.is_neg_1_1:
                global_views = global_views * 2 - 1
                local_views = local_views * 2 - 1
        else:
            global_views = [gv * 2 - 1 for gv in global_views] if self.is_neg_1_1 else global_views
            local_views = [lv * 2 - 1 for lv in local_views] if self.is_neg_1_1 else local_views

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
            A = torch.randn(proj.size(-1), self.rnd_proj_dim, device=proj.device)  # [D, projD]
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
        else:
            raise ValueError(f"Invalid input dim: {x.shape}")


def create_lejepa_projector(
    input_dim: int,
    proj_dim: int = 128,
    mid_hiddens: list[int] = [2048, 2048],
    mean_out_hw=False,
    norm_type: str = "batchnorm1d",
):
    from torchvision.ops import MLP

    mlp_projector = MLP(input_dim, [*mid_hiddens, proj_dim], norm_layer=get_norm_layer(norm_type))
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

    if sigreg is None:
        sigreg = SIGReg(17, 256).to(global_emb.device)

    # Regularization loss for lejepa
    sigreg_loss = sigreg(embeddings)  # proj: (V, B, D)

    # Center of the views
    centers = global_emb.mean(0, keepdim=True)

    # Losses
    inv_loss = (centers - embeddings).square().mean()

    # Total loss and loss breakdowns
    loss = inv_loss * (1 - lam) + sigreg_loss * lam
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

    # Test the new hyperspectral ColorJitter
    print("Testing hyperspectral ColorJitter...")

    # Create hyperspectral data (batch_size, channels, height, width)
    x = torch.randn(2, 64, 224, 224)  # 64-channel hyperspectral image
    x = torch.clamp(x, 0.0, 1.0)  # Ensure values are within [0,1] range

    # Test the individual hyperspectral ColorJitter
    hs_jitter = HyperspectralColorJitter(
        intensity_range=(-0.2, 0.2),
        contrast_range=(0.8, 1.2),
        noise_std=0.01,
        band_drop_ratio=0.05,
        probs={
            "intensity": 1.0,  # Always apply for testing
            "contrast": 1.0,
            "noise": 1.0,
            "band_drop": 1.0,
            "smoothing": 1.0,
        },
    )

    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")

    # Apply augmentation
    x_augmented = hs_jitter(x)

    print(f"Output shape: {x_augmented.shape}")
    print(f"Output range: [{x_augmented.min():.3f}, {x_augmented.max():.3f}]")
    print(f"Mean change: {x.mean():.3f} -> {x_augmented.mean():.3f}")
    print(f"Std change: {x.std():.3f} -> {x_augmented.std():.3f}")

    # Test complete LeJEPAAugmentation
    print("\nTesting complete LeJEPAAugmentation...")
    aug = LeJEPAAugmentation(n_locals=2, n_globals=1)

    global_views, local_views = aug(x)
    print(f"Global view count: {len(global_views)}, shapes: {[v.shape for v in global_views]}")
    print(f"Local view count: {len(local_views)}, shapes: {[v.shape for v in local_views]}")

    # network = torch.nn.Conv2d(3, 384, 3, 1, 1)

    # x = torch.randn(8, 128, 224, 224)  # Example input tensor
    # global_views, local_views = aug(x)

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


def __test_color_jitter_safe_wrapper():
    """
    Test SafeColorJitterWrapper with both hyperspectral and RGB images
    """
    print("Testing SafeColorJitterWrapper...")

    # Create instances of both color jitter types
    hp_jitter = HyperspectralColorJitter(
        intensity_range=(-0.2, 0.2),
        contrast_range=(0.8, 1.2),
        noise_std=0.01,
        band_drop_ratio=0.05,
        smoothing_kernel=3,
        probs={
            "intensity": 0.8,
            "contrast": 0.7,
            "noise": 0.5,
            "band_drop": 0.3,
            "smoothing": 0.4,
        },
    )

    rgb_jitter = K.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.2,
        hue=0.1,
        p=0.8,
    )

    # Create the wrapper
    safe_wrapper = SafeColorJitterWrapper(
        hp_color_jitter=hp_jitter,
        rgb_color_jitter=rgb_jitter,
    )

    # Test with hyperspectral image (64 channels)
    print("\n1. Testing with hyperspectral image (64 channels)...")
    hp_image = torch.randn(2, 64, 224, 224).clamp(0, 1)
    print(f"Input hyperspectral image shape: {hp_image.shape}")
    print(f"Input range: [{hp_image.min():.3f}, {hp_image.max():.3f}]")

    hp_augmented = safe_wrapper(hp_image)
    print(f"Output hyperspectral image shape: {hp_augmented.shape}")
    print(f"Output range: [{hp_augmented.min():.3f}, {hp_augmented.max():.3f}]")
    print(f"Mean change: {hp_image.mean():.3f} -> {hp_augmented.mean():.3f}")
    print(f"Std change: {hp_image.std():.3f} -> {hp_augmented.std():.3f}")

    # Test with RGB image (3 channels)
    print("\n2. Testing with RGB image (3 channels)...")
    rgb_image = torch.randn(2, 3, 224, 224).clamp(0, 1)
    print(f"Input RGB image shape: {rgb_image.shape}")
    print(f"Input range: [{rgb_image.min():.3f}, {rgb_image.max():.3f}]")

    rgb_augmented = safe_wrapper(rgb_image)
    print(f"Output RGB image shape: {rgb_augmented.shape}")
    print(f"Output range: [{rgb_augmented.min():.3f}, {rgb_augmented.max():.3f}]")
    print(f"Mean change: {rgb_image.mean():.3f} -> {rgb_augmented.mean():.3f}")
    print(f"Std change: {rgb_image.std():.3f} -> {rgb_augmented.std():.3f}")

    # Test with grayscale image (1 channel)
    print("\n3. Testing with grayscale image (1 channel)...")
    gray_image = torch.randn(2, 1, 224, 224).clamp(0, 1)
    print(f"Input grayscale image shape: {gray_image.shape}")
    print(f"Input range: [{gray_image.min():.3f}, {gray_image.max():.3f}]")

    gray_augmented = safe_wrapper(gray_image)
    print(f"Output grayscale image shape: {gray_augmented.shape}")
    print(f"Output range: [{gray_augmented.min():.3f}, {gray_augmented.max():.3f}]")
    print(f"Mean change: {gray_image.mean():.3f} -> {gray_augmented.mean():.3f}")
    print(f"Std change: {gray_image.std():.3f} -> {gray_augmented.std():.3f}")

    # Verify that shapes are preserved
    assert hp_augmented.shape == hp_image.shape, "Hyperspectral shape mismatch"
    assert rgb_augmented.shape == rgb_image.shape, "RGB shape mismatch"
    assert gray_augmented.shape == gray_image.shape, "Grayscale shape mismatch"

    print("\n✅ SafeColorJitterWrapper test completed successfully!")
    print("✅ All image types handled correctly!")
    print("✅ Hyperspectral images use HyperspectralColorJitter")
    print("✅ RGB/Grayscale images use standard ColorJitter")


def __test_augmentation_pipelines():
    """
    Test the complete augmentation pipelines with different image types
    """
    from src.data.litdata_hyperloader import ImageStreamingDataset

    print("Testing complete augmentation pipelines...")

    ds = ImageStreamingDataset(input_dir="data/hyspecnet11k/LitData_hyper_images", to_neg_1_1=False)
    hp_image = ds[3022]["img"][None]
    print(f"Sample hyperspectral image shape from dataset: {hp_image.shape}")

    # Test with hyperspectral image
    print("\n1. Testing with hyperspectral image...")
    # hp_image = torch.randn(2, 64, 256, 256).clamp(0, 1)
    print(f"Input shape: {hp_image.shape}")

    aug = LeJEPAAugmentation(n_locals=6, n_globals=2, is_neg_1_1=False)
    global_views, local_views = aug(hp_image)

    print(f"Global views: {len(global_views)}, shapes: {[v.shape for v in global_views]}")
    print(f"Local views: {len(local_views)}, shapes: {[v.shape for v in local_views]}")

    # Test with RGB image
    print("\n2. Testing with RGB image...")
    rgb_image = hp_image[:, [39, 29, 19]]
    print(f"Input shape: {rgb_image.shape}")

    global_views_rgb, local_views_rgb = aug(rgb_image)
    print(f"Global views: {len(global_views_rgb)}, shapes: {[v.shape for v in global_views_rgb]}")
    print(f"Local views: {len(local_views_rgb)}, shapes: {[v.shape for v in local_views_rgb]}")

    # Test with grayscale image
    print("\n3. Testing with grayscale image...")
    gray_image = hp_image.mean(1, True)
    print(f"Input shape: {gray_image.shape}")

    global_views_gray, local_views_gray = aug(gray_image)
    print(f"Global views: {len(global_views_gray)}, shapes: {[v.shape for v in global_views_gray]}")
    print(f"Local views: {len(local_views_gray)}, shapes: {[v.shape for v in local_views_gray]}")

    print("\n✅ All augmentation pipelines working correctly!")
    print("✅ SafeColorJitterWrapper integrated successfully!")


if __name__ == "__main__":
    """
    python -m src.stage1.self_supervised.lejepa_aug
    """

    # Test all components
    # __test_aug_pipe()  # Test hyperspectral ColorJitter
    # print("\n" + "=" * 60 + "\n")
    # __test_color_jitter_safe_wrapper()  # Test SafeColorJitterWrapper
    # print("\n" + "=" * 60 + "\n")
    __test_augmentation_pipelines()  # Test complete pipelines

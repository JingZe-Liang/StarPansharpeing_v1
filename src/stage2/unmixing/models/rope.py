import math
from itertools import repeat as repeat_iter
from math import ceil, floor
from functools import reduce
from operator import mul
from typing import (
    Callable,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np
import torch
import torch.nn as nn
from einops import pack, rearrange, repeat, unpack
from torch import Size, Tensor, tensor
from torch.nn import ModuleList

__all__ = [
    "RopePosEmbed",
    "ContinuousAxialPositionalEmbedding",
    "get_1d_rotary_pos_embed",
    "apply_rotary_emb",
    "RotaryPositionEmbedding",
    "RotaryPositionEmbeddingPytorchV2",
    "get_2d_sincos_pos_embed",
    "get_2d_sincos_pos_embed_from_grid",
    "get_1d_sincos_pos_embed_from_grid",
    "LearnablePosAxisEmbedding",
    "AxialPositionalEmbedding",
    "AxialPositionalEmbedding2D",
    "LearnablePosAxisEmbedding2D",
]

# * --- RoPE from Sana and Flux1.dev --- #


class RopePosEmbed(nn.Module):
    # modified from https://github.com/black-forest-labs/flux/blob/c00d7c60b085fce8058b9df845e036090873f2ce/src/flux/modules/layers.py#L11
    def __init__(self, theta: int, axes_dim: List[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor):
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        is_mps = ids.device.type == "mps"
        is_npu = ids.device.type == "npu"
        freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64
        for i in range(n_axes):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[:, i],
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width)[None, :]
        )

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = (
            latent_image_ids.shape
        )

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)


def get_1d_rotary_pos_embed(
    dim: int,
    pos: np.ndarray | int | torch.Tensor,
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=torch.float32,  #  torch.float32, torch.float64 (flux)
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        linear_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the context extrapolation. Defaults to 1.0.
        ntk_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the NTK-Aware RoPE. Defaults to 1.0.
        repeat_interleave_real (`bool`, *optional*, defaults to `True`):
            If `True` and `use_real`, real part and imaginary part are each interleaved with themselves to reach `dim`.
            Otherwise, they are concateanted with themselves.
        freqs_dtype (`torch.float32` or `torch.float64`, *optional*, defaults to `torch.float32`):
            the dtype of the frequency tensor.
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0
    # print(f"{dim=}")

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # type: ignore  # [S]
    assert torch.is_tensor(pos), f"pos must be a tensor, got {type(pos)}"

    theta = theta * ntk_factor
    freqs = (
        1.0
        / (
            theta
            ** (
                torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device)[
                    : (dim // 2)
                ]
                / dim
            )
        )
        / linear_factor
    )  # [D/2]
    freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]
    if use_real and repeat_interleave_real:
        # flux, hunyuan-dit, cogvideox
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()  # [S, D]
        return freqs_cos, freqs_sin
    elif use_real:
        # stable audio, allegro
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()  # [S, D]
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        # lumina
        freqs_cis = torch.polar(
            torch.ones_like(freqs), freqs
        )  # complex64     # [S, D/2]
        return freqs_cis


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(
                -1
            )  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Sana
            cos = cos.transpose(-1, -2)
            sin = sin.transpose(-1, -2)
            x_real, x_imag = x.reshape(*x.shape[:-2], -1, 2, x.shape[-1]).unbind(
                -2
            )  # [B, H, D//2, S]
            x_rotated = torch.stack([-x_imag, x_real], dim=-2).flatten(2, 3)
        else:
            raise ValueError(
                f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2."
            )

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        # used for lumina
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


# * --- RoPE from Cosmos Predictor2 --- #


def _rotate_half_te(x: torch.Tensor) -> torch.Tensor:
    """
    change sign so the last dimension becomes [-odd, +even].
    Adopted from TransformerEngine.
    Source: https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/attention.py
    """
    x = x.view(x.shape[:-1] + torch.Size((2, x.shape[-1] // 2)))
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


@torch.autocast(enabled=False, device_type="cuda")
def _apply_rotary_pos_emb_te(
    t: torch.Tensor,
    cos_freqs: torch.Tensor,
    sin_freqs: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary positional embedding tensor to the input tensor.
    Adopted from TransformerEngine.
    Source: https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/attention.py

    Parameters
    ----------
    t: torch.Tensor
        Input tensor of shape `[b, s, h, d]`, on which
        rotary positional embedding will be applied.
    cos_freqs: torch.Tensor
        Cosine component of rotary positional embedding tensor of shape `[s, 1, 1, d]` and dtype 'float',
    sin_freqs: torch.Tensor
        Sine component of rotary positional embedding tensor of shape `[s, 1, 1, d]` and dtype 'float',
    """
    rot_dim = cos_freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    t = (t * cos_freqs) + (_rotate_half_te(t) * sin_freqs)
    output = torch.cat((t, t_pass), dim=-1)
    return output


class RotaryPositionEmbedding(torch.nn.Module):
    """
    Rotary Position Embedding module as described in the paper:
    https://arxiv.org/abs/2104.09864

    This module implements rotary positional embeddings, which are used to
    enhance the performance of transformer models.

    Args:
        dim (int): Dimensionality of the input tensor.
        max_position_embeddings (Optional[int]): Maximum position embeddings.
        original_max_position_embeddings (Optional[int]): Original maximum position embeddings.
        rope_theta (Optional[float]): Base for the frequency calculation.
        apply_yarn (Optional[bool]): Whether to apply YaRN (Yet another Rotary).
        scale (Optional[int]): Scaling factor for the frequency calculation.
        extrapolation_factor (Optional[int]): Extrapolation factor for the frequency extension.
        attn_factor (Optional[int]): Attention factor for the frequency calculation.
        beta_fast (Optional[int]): Fast beta value for the YaRN frequency calculation.
        beta_slow (Optional[int]): Slow beta value for the YaRN frequency calculation.
        rope_dim (Optional[str]): Dimensionality of the RoPE. Choices: "1D", "2D", "3D".
        latent_shape (Optional[List[int]]): Shape of the latent tensor for video or image inputs.
        original_latent_shape (Optional[List[int]]): Original shape of the latent tensor for video or image inputs.
        pad_to_multiple_of (Optional[int]): Pad the position embedding to a multiple of this value.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int | None = None,
        original_max_position_embeddings: int | None = None,
        rope_theta: float = 10000.0,
        apply_yarn: bool = False,
        scale: int | None = None,
        extrapolation_factor: int = 1,
        attn_factor: int = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        rope_dim: str = "1D",
        latent_shape: list[int] | None = None,
        original_latent_shape: list[int] | None = None,
        pad_to_multiple_of: int | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.rope_theta = rope_theta
        self.apply_yarn = apply_yarn
        self.scale = scale
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = 1.0
        self.rope_dim = rope_dim
        self.latent_shape = latent_shape
        self.original_latent_shape = original_latent_shape
        self.pad_to_multiple_of = pad_to_multiple_of
        self.device = (
            torch.device(torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.get_inv_freq(self.device)

    def get_mscale(self, scale: float = 1.0) -> float:
        """Get the magnitude scaling factor for YaRN."""
        if scale <= 1:
            return 1.0
        return 0.1 * math.log(scale) + 1.0

    def forward(self, seq_len: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass for the rotary position embedding.

        Args:
            seq_len (Optional[int]): Length of the sequence.

        Returns:
            torch.Tensor: The computed frequencies for positional embedding.
        """

        if self.apply_yarn and seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
        self.freqs = self.compute_freqs()

        return self.freqs

    def compute_freqs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the spatial frequencies for the latent tensor."""
        self.seq = torch.arange(self.max_seq_len_cached, dtype=torch.float).to(
            self.device
        )
        if self.rope_dim == "1D":
            emb = torch.einsum("i,j->ij", self.seq, self.inv_freq)

        elif self.rope_dim == "2D":
            H, W = self.latent_shape
            half_emb_h = torch.outer(self.seq[:H], self.spatial_inv_freq)
            half_emb_w = torch.outer(self.seq[:W], self.spatial_inv_freq)
            emb = torch.cat(
                [
                    repeat(half_emb_h, "h d -> h w d", w=W),
                    repeat(half_emb_w, "w d -> h w d", h=H),
                ]
                * 2,
                dim=-1,
            )
            emb = rearrange(emb, "h w d -> (h w) 1 1 d").float()

        elif self.rope_dim == "3D":
            T, H, W = self.latent_shape
            half_emb_t = torch.outer(self.seq[:T], self.temporal_inv_freq)
            half_emb_h = torch.outer(self.seq[:H], self.spatial_inv_freq)
            half_emb_w = torch.outer(self.seq[:W], self.spatial_inv_freq)
            emb = torch.cat(
                [
                    repeat(half_emb_t, "t d -> t h w d", h=H, w=W),  # d // 3
                    repeat(half_emb_h, "h d -> t h w d", t=T, w=W),  # d - (d // 3 * 2)
                    repeat(half_emb_w, "w d -> t h w d", t=T, h=H),
                ]
                * 2,
                dim=-1,
            )
            emb = rearrange(emb, "t h w d -> (t h w) 1 1 d").float()
        else:
            raise ValueError(f"Invalid RoPE dimensionality: {self.rope_dim}")
        return emb

    def get_scale_factors(
        self, inv_freq: torch.Tensor, original_seq_len: int
    ) -> torch.Tensor:
        """Get the scale factors for YaRN."""
        # Calculate the high and low frequency cutoffs for YaRN. Note: `beta_fast` and `beta_slow` are called
        # `high_freq_factor` and `low_freq_factor` in the Llama 3.1 RoPE scaling code.
        high_freq_cutoff = 2 * math.pi * self.beta_fast / original_seq_len
        low_freq_cutoff = 2 * math.pi * self.beta_slow / original_seq_len
        # Obtain a smooth mask that has a value of 0 for low frequencies and 1 for high frequencies, with linear
        # interpolation in between.
        smooth_mask = torch.clamp(
            (inv_freq - low_freq_cutoff) / (high_freq_cutoff - low_freq_cutoff),
            min=0,
            max=1,
        )
        # For low frequencies, we scale the frequency by 1/self.scale. For high frequencies, we keep the frequency.
        scale_factors = (1 - smooth_mask) / self.scale + smooth_mask
        return scale_factors

    @torch.autocast(enabled=False, device_type="cuda")
    def get_inv_freq(self, device: torch.device) -> None:
        """Get the inverse frequency."""
        if self.rope_dim == "1D":
            assert self.max_position_embeddings is not None, (
                "Max position embeddings required."
            )
            inv_freq = 1.0 / (
                self.rope_theta
                ** (
                    torch.arange(0, self.dim, 2, dtype=torch.float32, device=device)
                    / self.dim
                )
            )
            if self.apply_yarn:
                assert self.original_max_position_embeddings is not None, (
                    "Original max position embeddings required."
                )
                assert self.beta_slow is not None, "Beta slow value required."
                assert self.beta_fast is not None, "Beta fast value required."

                scale_factors = self.get_scale_factors(
                    inv_freq, self.original_max_position_embeddings
                )
                # Apply the scaling factors to inv_freq.
                inv_freq = inv_freq * scale_factors
                # Set the magnitude scaling factor.
                self.mscale = float(self.get_mscale(self.scale) * self.attn_factor)
            self.max_seq_len_cached = self.max_position_embeddings
            self.inv_freq = inv_freq

        elif self.rope_dim == "2D":
            assert self.latent_shape is not None, "Latent shape required."
            dim_h = self.dim // 2
            spatial_inv_freq = 1.0 / (
                self.rope_theta
                ** torch.arange(0, dim_h, 2, dtype=torch.float32, device=device)
                / dim_h
            )
            if self.apply_yarn:
                assert self.original_latent_shape is not None, (
                    "Original latent shape required."
                )
                assert self.beta_slow is not None, "Beta slow value required."
                assert self.beta_fast is not None, "Beta fast value required."

                scale_factors = self.get_scale_factors(
                    spatial_inv_freq, self.original_latent_shape[0]
                )
                spatial_inv_freq = spatial_inv_freq * scale_factors
                self.mscale = float(self.get_mscale(self.scale) * self.attn_factor)
            self.spatial_inv_freq = spatial_inv_freq
            self.max_seq_len_cached = max(self.latent_shape)

        elif self.rope_dim == "3D":
            assert self.latent_shape is not None, "Latent shape required."
            dim_h = self.dim // 6 * 2
            dim_t = self.dim - 2 * dim_h
            self.dim_spatial_range = (
                torch.arange(0, dim_h, 2)[: (dim_h // 2)].float().to(device) / dim_h
            )
            spatial_inv_freq = 1.0 / (self.rope_theta**self.dim_spatial_range)
            self.dim_temporal_range = (
                torch.arange(0, dim_t, 2)[: (dim_t // 2)].float().to(device) / dim_t
            )
            temporal_inv_freq = 1.0 / (self.rope_theta**self.dim_temporal_range)
            if self.apply_yarn:
                assert self.original_latent_shape is not None, (
                    "Original latent shape required."
                )
                assert self.beta_slow is not None, "Beta slow value required."
                assert self.beta_fast is not None, "Beta fast value required."
                scale_factors_spatial = self.get_scale_factors(
                    spatial_inv_freq, self.original_latent_shape[1]
                )
                spatial_inv_freq = spatial_inv_freq * scale_factors_spatial
                scale_factors_temporal = self.get_scale_factors(
                    temporal_inv_freq, self.original_latent_shape[0]
                )
                temporal_inv_freq = temporal_inv_freq * scale_factors_temporal
                self.mscale = float(self.get_mscale(self.scale) * self.attn_factor)
            self.spatial_inv_freq = spatial_inv_freq
            self.temporal_inv_freq = temporal_inv_freq
            self.max_seq_len_cached = max(self.latent_shape)
        else:
            raise ValueError(f"Invalid RoPE dimensionality: {self.rope_dim}")

        self.freqs = self.compute_freqs()


class RotaryPositionEmbeddingPytorchV2(RotaryPositionEmbedding):
    """
    Rotary Position Embedding that works in the same way as the TransformerEngine RoPE
    (https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/attention.py)

    """

    def __init__(
        self,
        seq_len: int,
        training_type: str | None = None,
        *,
        dim: int,
        max_position_embeddings: int | None = None,
        original_max_position_embeddings: int | None = None,
        rope_theta: float = 10000.0,
        apply_yarn: bool = False,
        scale: float | None = None,
        extrapolation_factor: int = 1,
        attn_factor: int = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        rope_dim: str = "1D",
        latent_shape: tuple | None = None,
        original_latent_shape: tuple | None = None,
        pad_to_multiple_of: int | None = None,
    ):
        super().__init__(
            dim,
            max_position_embeddings,
            original_max_position_embeddings,
            rope_theta,
            apply_yarn,
            scale,
            extrapolation_factor,
            attn_factor,
            beta_fast,
            beta_slow,
            rope_dim,
            latent_shape,
            original_latent_shape,
            pad_to_multiple_of,
        )
        emb = self.create_rope_freqs(seq_len=seq_len, training_type=training_type)
        emb = emb.transpose(0, 1).contiguous()  # [seq, 1, 1, dim] -> [1, seq, 1, dim]
        assert emb.shape[0] == 1 and emb.shape[2] == 1, f"emb shape: {emb.shape}"
        # cos/sin first then dtype conversion for better precision
        self.register_buffer("cos_cached", torch.cos(emb), persistent=False)
        self.register_buffer("sin_cached", torch.sin(emb), persistent=False)

    def create_rope_freqs(
        self, seq_len: int, training_type: str = "2D"
    ) -> torch.Tensor:
        """
        Create rotary position embedding frequencies.

        Args:
            seq_len (int): Sequence length of a sample.

        Returns:
            torch.Tensor: The computed positional embeddings.
        """
        if self.rope_dim == "1D":
            freqs = super().forward(seq_len=seq_len)
            emb = torch.cat((freqs, freqs), dim=-1)
            emb = emb.reshape(emb.size(0), 1, 1, emb.size(1))

        elif self.rope_dim in ["2D", "3D"]:
            emb = super().forward(seq_len=seq_len)
            if training_type == "text_to_video":
                # since we added <bov> token at the beginning of the video for text2world, we also extend the position embedding by one token in the beginning
                bov_pe = torch.zeros((1, *emb.shape[1:]), device=emb.device)
                emb = torch.cat((bov_pe, emb), dim=0)
        else:
            raise ValueError(f"Invalid RoPE dimensionality: {self.rope_dim}")

        if (
            self.pad_to_multiple_of is not None
            and emb.shape[0] % self.pad_to_multiple_of != 0
        ):
            # Round up to the nearest multiple of pad_to_multiple_of
            pad_len = self.pad_to_multiple_of - emb.shape[0] % self.pad_to_multiple_of
            emb = torch.cat(
                (emb, torch.zeros((pad_len, *emb.shape[1:]), device=emb.device)), dim=0
            )

        return emb

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q, k: query, key tensors of shape [b, s, h, d]
        input_pos: optional tensor of positions to apply RoPE to, shape [s]
        seq_len: optional sequence length to apply RoPE to, used for inference

        """

        if q.dtype != self.cos_cached.dtype:
            self.cos_cached = self.cos_cached.to(q.dtype)
            self.sin_cached = self.sin_cached.to(q.dtype)

        cos_emb = self.cos_cached
        sin_emb = self.sin_cached
        if input_pos is not None:
            cos_emb = cos_emb[:, input_pos, :, :]
            sin_emb = sin_emb[:, input_pos, :, :]
        elif seq_len is not None:
            cos_emb = cos_emb[:, :seq_len, :, :]
            sin_emb = sin_emb[:, :seq_len, :, :]
        q = _apply_rotary_pos_emb_te(q, cos_emb, sin_emb)
        k = _apply_rotary_pos_emb_te(k, cos_emb, sin_emb)
        return q, k


# * --- Cosine-sine PE --- #


def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat_iter(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)


@torch.autocast(enabled=False, device_type="cuda")
def get_2d_sincos_pos_embed(
    embed_dim,
    grid_size: tuple[int, int] | int,
    cls_token=False,
    extra_tokens=0,
    pe_interpolation=1.0,
    base_size=16,
):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = to_2tuple(grid_size)  # type: ignore[assignment]
    grid_size = cast(tuple[int, int], grid_size)

    grid_h = (
        np.arange(grid_size[0], dtype=np.float32)
        / (grid_size[0] / base_size)
        / pe_interpolation
    )
    grid_w = (
        np.arange(grid_size[1], dtype=np.float32)
        / (grid_size[1] / base_size)
        / pe_interpolation
    )
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# * --- Axes PE --- #


# * --- Learnable DiT per-layer positional embedding --- * #
# adapted from Cosmos world model (diffusion model)

from einops import repeat
from torch.nn.init import trunc_normal_


def normalize(
    x: torch.Tensor, dim: Optional[list[int]] = None, eps: float = 0
) -> torch.Tensor:
    """
    Normalizes the input tensor along specified dimensions such that the average square norm of elements is adjusted.

    Args:
        x (torch.Tensor): The input tensor to normalize.
        dim (list, optional): The dimensions over which to normalize. If None, normalizes over all dimensions except the first.
        eps (float, optional): A small constant to ensure numerical stability during division.

    Returns:
        torch.Tensor: The normalized tensor.
    """
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


def repeat_at_rest_dims(x: torch.Tensor, insert_dim: int, rest_lens: list[int]):
    # when insert_dim=0, rest_lens=[128, 128]
    # x: [t, d] -> [t, h, w, d] (e.g., [32, 512] -> [32, 128, 128, 512])

    non_d_dims = len(rest_lens) + 1
    for i in range(non_d_dims):
        if i != insert_dim:
            x = x.unsqueeze(i)

    expand_shape = []
    for i in range(non_d_dims):
        if i != insert_dim:
            expand_shape.append(rest_lens[i - 1])
        else:
            expand_shape.append(-1)

    return x.expand(*expand_shape, -1)


class LearnablePosAxisEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        seq_len: list[int],
        interpolation: Literal["crop", "downsample"] = "crop",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.interpolation = interpolation
        self.ndim = len(seq_len)
        self.pos_embed_n_dims = nn.ParameterList(
            [nn.Parameter(torch.zeros(seq_len[i], dim)) for i in range(self.ndim)]
        )
        self.eps = eps

        # trunc normal init
        for i in range(self.ndim):
            trunc_normal_(self.pos_embed_n_dims[i], std=0.02)

    def forward(self, axials: tuple[int, ...]):
        # x: [b, *ndims, d]

        ndim_context = len(axials)
        assert len(self.pos_embed_n_dims) == ndim_context, (
            f"len(self.pos_embed) must be equal to ndim_context, but got {len(self.pos_embed_n_dims)=} and {ndim_context=}"
        )

        # interpolate
        if self.interpolation == "crop":
            pos_embed = torch.zeros(
                1,
            )  ## FIXME: get the shape of pos_embed
            for i in range(self.ndim):
                len_i = x.shape[i + 1]
                embed_i = self.pos_embed_n_dims[i][:len_i]  # [len_i, d]
                # repeat at the rest of context dims
                embed_i = repeat_at_rest_dims(embed_i, i, axials.pop(i))
                pos_embed = pos_embed + embed_i
        elif self.interpolation == "downsample":
            pos_embed = [
                self.pos_embed_n_dims[i][:: x.shape[i + 1]] for i in range(ndim_context)
            ]
        else:
            raise ValueError(f"Unknown interpolation method: {self.interpolation}")

        return normalize(pos_embed, dim=-1, eps=self.eps)


# helper functions


def exists(v):
    return v is not None


class AxialPositionalEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        axial_shape: tuple[int, ...],
        axial_dims: tuple[int, ...] | None = None,
    ):
        super().__init__()

        self.dim = dim
        self.shape = axial_shape
        self.max_seq_len = reduce(mul, axial_shape, 1)

        self.summed = not exists(axial_dims)
        axial_dims = ((dim,) * len(axial_shape)) if self.summed else axial_dims

        assert len(self.shape) == len(axial_dims), (
            "number of axial dimensions must equal the number of dimensions in the shape"
        )
        assert self.summed or not self.summed and sum(axial_dims) == dim, (
            f"axial dimensions must sum up to the target dimension {dim}"
        )

        self.weights = nn.ParameterList([])

        for ind, (shape, axial_dim) in enumerate(zip(self.shape, axial_dims)):
            ax_shape = [1] * len(self.shape)
            ax_shape[ind] = shape
            ax_shape = (1, *ax_shape, axial_dim)  # [1, h, w, d]
            ax_emb = nn.Parameter(
                torch.zeros(ax_shape)
            )  # in the original implementation, they use normal_(0, 1)
            ax_emb = nn.init.trunc_normal_(ax_emb, std=0.02)
            self.weights.append(ax_emb)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        assert seq_len <= self.max_seq_len, (
            f"Sequence length ({seq_len}) must be less than the maximum sequence length allowed ({self.max_seq_len})"
        )

        embs = []

        for ax_emb in self.weights:
            axial_dim = ax_emb.shape[-1]
            expand_shape = (batch, *self.shape, axial_dim)
            emb = ax_emb.expand(expand_shape).reshape(
                batch, self.max_seq_len, axial_dim
            )
            embs.append(emb)

        pos_emb = sum(embs) if self.summed else torch.cat(embs, dim=-1)

        return pos_emb[:, :seq_len].to(x)


# wrapper for images


class AxialPositionalEmbedding2D(nn.Module):
    def __init__(
        self,
        dim,
        axial_shape: tuple[int, ...],
        axial_dims: tuple[int, ...] | None = None,
    ):
        super().__init__()
        assert len(axial_shape) == 2, "Axial shape must have 2 dimensions for images"
        self.pos_emb = AxialPositionalEmbedding(dim, axial_shape, axial_dims)

    def forward(self, img):
        img = rearrange(img, "b c h w -> b h w c")
        img, packed_shape = pack([img], "b * c")

        pos_emb = self.pos_emb(img)

        (pos_emb,) = unpack(pos_emb, packed_shape, "b * c")
        pos_emb = rearrange(pos_emb, "b h w c -> b c h w")
        return pos_emb


# * --- learnable factorized axisal positional embedding --- * #


class LearnablePosAxisEmbedding2D(nn.Module):
    def __init__(
        self,
        dim: int,
        seq_len: list[int, int],
        factorize: bool = True,
        interpolation: Literal["crop", "interpolate"] = "crop",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.interpolation = interpolation
        self.factorize = factorize
        assert interpolation in ["crop", "interpolate"], (
            "Unknown interpolation method, only support `crop` or `interpolate`"
        )
        assert len(seq_len) == 2, "seq_len must be a list of 2 integers"
        self.max_h, self.max_w = seq_len
        if factorize:
            self.pos_embed_h = nn.Parameter(torch.zeros(dim, self.max_h))
            self.pos_embed_w = nn.Parameter(torch.zeros(dim, self.max_w))
            # trunc normal init
            trunc_normal_(self.pos_embed_h, std=0.02)
            trunc_normal_(self.pos_embed_w, std=0.02)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, dim, self.max_h, self.max_w))
            trunc_normal_(self.pos_embed, std=0.02)
        self.eps = eps

    def forward_factorize(self, hw: tuple[int, int]):
        h, w = hw

        # interpolate
        _force_to_interp = h > self.max_h or w > self.max_w
        if self.interpolation == "crop" and not _force_to_interp:
            assert h <= self.max_h and w <= self.max_w, (
                "input height and width must be less than or equal to the positional embedding height and width"
            )
            pos_embed_h = self.pos_embed_h[:, :h]
            pos_embed_w = self.pos_embed_w[:, :w]
            # repeat
            embed = repeat(pos_embed_h, "d h -> b d h w", b=1, w=w) + repeat(
                pos_embed_w, "d w -> b d h w", b=1, h=h
            )
        elif self.interpolation == "interpolate" or _force_to_interp:
            _interp_fn = lambda x: torch.nn.functional.interpolate(
                x, size=(h, w), mode="bilinear", align_corners=True
            )
            embed = _interp_fn(
                repeat(self.pos_embed_h, "d h -> b d h w", b=1, w=w)
            ) + _interp_fn(repeat(self.pos_embed_w, "d w -> b d h w", b=1, h=h))
        else:
            raise ValueError(
                f"Unknown interpolation method: {self.interpolation}, "
                "or input height and width must be less than or equal to the positional embedding height and width"
            )

        # normalize
        embed = normalize(embed, dim=1, eps=self.eps)

        return embed

    def forward_non_factorize(self, hw: tuple[int, int]):
        h, w = hw

        if self.interpolation == "crop":
            pos_embed = self.pos_embed[:, :h, :w]
        elif self.interpolation == "interpolate":
            self.pos_embed
            pos_embed = torch.nn.functional.interpolate(
                self.pos_embed, size=(h, w), mode="bilinear"
            )

        return normalize(pos_embed, dim=-1)

    def forward(self, axial_dim: tuple[int, ...], flatten: bool = True):
        if self.factorize:
            pe = self.forward_factorize(axial_dim)
        else:
            pe = self.forward_non_factorize(axial_dim)

        if flatten:
            pe = rearrange(pe, "b d h w -> b (h w) d")

        return normalize(pe, dim=-1)


# * --- Continous MLP learnable factorized positional embedding --- * #

# mlp - continuously parameterizing each axial position


def MLP(dim_in, dim_out, depth=2, expansion=2):
    curr_dim = dim_in
    dim_hidden = int(expansion * max(dim_in, dim_out))

    layers = []

    for _ in range(depth):
        layers.append(nn.Linear(curr_dim, dim_hidden))
        layers.append(nn.SiLU())

        curr_dim = dim_hidden

    layers.append(nn.Linear(curr_dim, dim_out))
    return nn.Sequential(*layers)


# main class


class ContinuousAxialPositionalEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        axials: tuple[int, ...] | None = None,
        num_axial_dims: int | None = None,
        mlp_depth: int = 2,
        mlp_expansion: int = 2.0,
        interp_type: str = "linear",
    ):
        """
        ## Usage

        >>> import torch
        >>> from axial_positional_embedding import (
        ...     ContinuousAxialPositionalEmbedding,
        ... )
        >>> pos_emb = ContinuousAxialPositionalEmbedding(
        >>>     dim = 512,
        >>>     num_axial_dims = 3
        >>> )
        >>> tokens = torch.randn(
        ...     1,
        ...     8,
        ...     16,
        ...     32,
        ...     512,
        ... )  # say a video with 8 frames, 16 x 32 image dimension
        >>> axial_pos_emb = pos_emb(
        ...     (8, 16, 32)
        ... )  # pass in the size from above
        >>> tokens = (
        ...     axial_pos_emb
        ...     + tokens
        ... )  # add positional embedding to token embeddings
        """
        super().__init__()
        if exists(axials):
            self.num_axial_dims = len(axials)
        elif exists(num_axial_dims):
            self.num_axial_dims = num_axial_dims
        else:
            raise ValueError(
                "either axials or num_axial_dims can not be None at the same time"
            )

        # mlps for each axial dimension
        self.mlps = ModuleList(
            [
                MLP(1, dim, depth=mlp_depth, expansion=mlp_expansion)
                for _ in range(self.num_axial_dims)
            ]
        )
        # dummy buffer for device and dtype
        self.register_buffer("dummy", tensor(0), persistent=False)

        # max sequence length
        self.interp_type = interp_type
        max_seq_len = axials
        if max_seq_len is not None:
            assert len(max_seq_len) == self.num_axial_dims, (
                "max_seq_len must have the same length as the number of axial dimensions"
            )
            self.register_buffer(
                "max_seq_len", torch.tensor(max_seq_len)
            )  # may affect EMA
            # self.max_seq_len = max_seq_len
        else:
            self.max_seq_len = None

    @property
    def device(self):
        return self.dummy.device

    @property
    def dtype(self):
        return next(self.mlps.parameters()).dtype

    def combine_factorized(
        self,
        axial_embeds: list[Tensor],
        axial_dims: tuple[int, ...] | None = None,
        flatten=False,
    ):
        if not exists(axial_dims):
            axial_dims = tuple(axial_embed.shape[0] for axial_embed in axial_embeds)

        assert len(axial_dims) == len(axial_embeds)

        axial_embeds = [
            axial_embed[:axial_dim]
            for axial_embed, axial_dim in zip(axial_embeds, axial_dims)
        ]

        axial_embed, *rest_axial_embeds = axial_embeds

        for rest_axial_embed in rest_axial_embeds:
            axial_embed = axial_embed[..., None, :] + rest_axial_embed

        assert axial_embed.shape[:-1] == axial_dims

        if flatten:
            axial_embed = rearrange(axial_embed, "... d -> (...) d")

        return axial_embed

    def maybe_derive_outer_dim(
        self, max_seq_len, axial_dims: Tensor | Size | tuple[int, ...]
    ):
        ndims = self.num_axial_dims
        assert len(axial_dims) in (ndims, ndims - 1)

        if len(axial_dims) == ndims:
            return axial_dims

        stride = reduce(mul, (*axial_dims,))

        outer_dim = ceil(max_seq_len / stride)
        return (outer_dim, *axial_dims)

    def forward_with_seq_len(
        self,
        seq_len: int,
        axial_dims: Tensor | Size | tuple[int, ...] = (),
        *,
        factorized: list[Tensor] | None = None,
        return_factorized=False,
    ):
        if not exists(factorized):
            axial_dims = self.maybe_derive_outer_dim(seq_len, axial_dims)
            factorized = self.forward(axial_dims, return_factorized=True)

        axial_embeds = self.combine_factorized(factorized, flatten=True)

        axial_embeds = axial_embeds[:seq_len]

        if not return_factorized:
            return axial_embeds

        return axial_embeds, factorized

    def forward_with_pos(
        self,
        pos: Tensor,
        axial_dims: Tensor | Size | tuple[int, ...] = (),
    ):
        assert pos.dtype in (torch.int, torch.long)

        max_pos = pos.amax().item() + 1
        axial_dims = self.maybe_derive_outer_dim(max_pos, axial_dims)
        indices = torch.unravel_index(pos, axial_dims)

        axial_embed = 0.0

        for mlp, axial_index in zip(self.mlps, indices):
            axial_index = rearrange(axial_index, "... -> ... 1")
            axial_embed = axial_embed + mlp(axial_index.to(self.dtype))

        return axial_embed

    def make_seq_len_for_mlp(self, axial_dim: int, dim_i: int):
        max_seq_len = self.max_seq_len[dim_i] if self.max_seq_len is not None else None

        if (
            max_seq_len is None
            or self.interp_type == "unchange"
            or axial_dim <= max_seq_len
        ):
            embed = torch.arange(axial_dim, device=self.device, dtype=self.dtype)
        elif axial_dim > max_seq_len:
            if self.interp_type == "linear":
                embed = torch.linspace(
                    0,
                    max_seq_len - 1,
                    steps=axial_dim,
                    device=self.device,
                    dtype=self.dtype,
                )
            else:
                raise ValueError(
                    f"Unknown interpolation type: {self.interp_type}, only support `linear`"
                )
        else:
            raise NotImplementedError(
                "max_seq_len = {} and interp_type = {}".format(
                    max_seq_len, self.interp_type
                )
            )

        assert embed.shape[0] == axial_dim, (
            "axial dim must be equal to the length of the embedding, not got {} and {}".format(
                axial_dim, embed.shape[0]
            )
        )

        return embed

    def forward(
        self,
        axial_dims: Tensor | Size | tuple[int, ...] | None = None,
        return_factorized=False,  # whether to return list[Tensor] of factorized axial positional embeddings
        flatten=True,  # whether to flatten axial dims
        align_batch_size=True,  # whether to repeat the batch size
    ):
        axial_embeds = []

        for i, (mlp, axial_dim) in enumerate(zip(self.mlps, axial_dims)):
            seq = self.make_seq_len_for_mlp(axial_dim, i)
            axial_embed = mlp(rearrange(seq, "n -> n 1"))

            axial_embeds.append(axial_embed)

        if return_factorized:
            assert not flatten

            # needed for Transfusion
            return axial_embeds

        axial_embed = self.combine_factorized(axial_embeds, flatten=flatten)

        if align_batch_size:
            axial_embed = axial_embed.unsqueeze(0)

        return axial_embed


if __name__ == "__main__":
    # Example usage
    embed_dim = 16
    grid_size = 4

    # Test 2D position embedding
    pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size)
    print("Position Embedding Shape:", pos_embed.shape)
    print("--" * 20)

    # Test rotary embedding
    rope = RopePosEmbed(theta=10000, axes_dim=[0, 16, 16])
    ids = RopePosEmbed._prepare_latent_image_ids(1, 32, 32, "cuda", torch.float64)
    pos_embed = rope(ids)
    print("Rotary Position Embedding Shape:", pos_embed[0].shape, pos_embed[1].shape)
    print("--" * 20)

    q = k = torch.randn(1, 8, 1024, 32).cuda()
    q = apply_rotary_emb(q, pos_embed, use_real=True, use_real_unbind_dim=-1)
    print(q.shape)

    # Test Cosmos Rotary Position Embedding
    rope = RotaryPositionEmbeddingPytorchV2(
        seq_len=32 * 32,
        dim=32,
        apply_yarn=True,
        scale=2.0,
        latent_shape=(64, 64),
        original_latent_shape=(32, 32),
        rope_dim="2D",
        beta_fast=4,
        beta_slow=1,
    ).cuda()

    x = torch.randn(2, 3, 64, 64)
    print(f"freq_cis shaped: {rope.create_rope_freqs(64 * 64).shape}")
    q = torch.randn(2, 64 * 64, 8, 32).cuda()
    k = torch.randn(2, 64 * 64, 8, 32).cuda()
    q, k = rope(q, k)
    print(f"query shaped as {q.shape}")

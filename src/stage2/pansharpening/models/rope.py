import math
from itertools import repeat as repeat_iter
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat

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
    pos: Union[np.ndarray, int],
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
        self.get_inv_freq(torch.device(torch.cuda.current_device()))

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

    def compute_freqs(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the spatial frequencies for the latent tensor."""
        self.seq = torch.arange(self.max_seq_len_cached, dtype=torch.float).cuda()
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

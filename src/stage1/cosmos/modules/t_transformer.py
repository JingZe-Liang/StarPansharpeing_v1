"""
TODO: Implement it and fix all bugs.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import create_act_layer, create_conv2d, create_norm_layer

from src.utilities.transport.tim.transition import get_delta_time_embed

from .t_blocks.embeddings import TimestepEmbedder
from .transformer import TransformerTokenizer


class FlowTransformerConditioned(TransformerTokenizer):
    """
    Flow Transformer with time and context conditioning.

    This class extends TransformerTokenizer to support time-dependent conditioning
    and context embedding from z tokens, similar to UViTMiddleTransformer logic.
    """

    def __init__(
        self,
        in_chan,
        embed_dim,
        ctx_embed_dim: int,
        time_cond_type: str = "t",
        ctx_format="1d",
        out_chan=None,
        img_size=384,
        patch_size=16,
        out_patch_size=1,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        dropout=0.0,
        attention_dropout=0.0,
        norm_layer="rmsnorm",
        drop_path=0.0,
        attn_type="sdpa",
        n_reg_tokens: int = 0,
        projections={"input": None, "output": None},
        additional_pe=False,
        pe_type="learn",  # ['learn', 'rope']
        rope_kwargs={},
        head: str | None = None,
        last_norm: str | None = None,
        is_causal: bool = False,
        patcher_type: str = "patch_embedder",
        with_cls_token: bool = False,
        # others
        patch_prog_dims: list[int] | None = None,
        unpatch_prog_dims: list[int] | None = None,
        **kwargs,
    ):
        # Remove flow-specific parameters from kwargs before passing to parent
        flow_kwargs = {
            "ctx_embed_dim": ctx_embed_dim,
        }
        remaining_kwargs = {k: v for k, v in kwargs.items() if k not in flow_kwargs}

        # Call parent init with all the standard parameters
        super().__init__(
            in_chan=in_chan,
            embed_dim=embed_dim,
            out_chan=out_chan,
            img_size=img_size,
            patch_size=patch_size,
            out_patch_size=out_patch_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            dropout=dropout,
            attention_dropout=attention_dropout,
            norm_layer=norm_layer,
            drop_path=drop_path,
            attn_type=attn_type,
            n_reg_tokens=n_reg_tokens,
            projections=projections,
            additional_pe=additional_pe,
            pe_type=pe_type,
            rope_kwargs=rope_kwargs,
            head=head,
            last_norm=last_norm,
            is_causal=is_causal,
            patcher_type=patcher_type,
            with_cls_token=with_cls_token,
            patch_prog_dims=patch_prog_dims,
            unpatch_prog_dims=unpatch_prog_dims,
        )

        self.time_embed_dim = embed_dim
        self.ctx_embed_dim = ctx_embed_dim

        self.time_cond_type = time_cond_type
        self.use_delta_t_embed = time_cond_type in ["t-r", "r", "t,t-r", "r,t-r", "t,r,t-r"]  # fmt: skip

        # Setting up the modules
        self._setup_time_embedder(time_cond_type)
        self._setup_context_projection(ctx_format, ctx_embed_dim, embed_dim)

    def _setup_time_embedder(self, time_cond_type: str):
        # Setup time embedder
        self.time_embed = TimestepEmbedder(self.time_embed_dim)
        if self.use_delta_t_embed:
            self.delta_t_embed = TimestepEmbedder(self.time_embed_dim)

    def _setup_context_projection(
        self, ctx_format: str, ctx_embed_dim: int, embed_dim: int
    ):
        # Context embedding projection
        # Assume the context is 1d
        self.ctx_format = ctx_format
        if ctx_format == "1d":
            self.ctx_proj = nn.Sequential(
                nn.Linear(ctx_embed_dim, embed_dim),
                create_act_layer("silu"),
                nn.Linear(embed_dim, embed_dim),
            )
        elif ctx_format == "2d":
            self.ctx_proj = nn.Sequential(
                nn.Conv2d(ctx_embed_dim, embed_dim, kernel_size=3, padding=1),
                create_act_layer("silu"),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            )
        else:
            raise ValueError(f"Unsupported ctx_format: {ctx_format}")

    def _get_hw(self, x: torch.Tensor) -> tuple[int, int]:
        """Get height and width from token sequence."""
        L = x.shape[1]
        h = w = int(math.sqrt(L))
        return h, w

    def _interp_z_to_x(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if z.shape[1] == x.shape[1]:
            return z

        h, w = self._get_hw(x)
        z_h, z_w = self._get_hw(z)

        # Convert to 2D format
        z_2d = rearrange(z, "b (zh zw) c -> b c zh zw", zh=z_h, zw=z_w)

        # Interpolate to match x's spatial dimensions
        z_2d_interp = F.interpolate(
            z_2d,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )

        # Convert back to 1D format
        z_1d = rearrange(z_2d_interp, "b c zh zw -> b (zh zw) c")
        return z_1d

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        z: torch.Tensor,
        *,
        ret_2d_tokens=False,
        ret_all=True,
        get_intermidates=None,
        out_shape: torch.Size | tuple | None = None,
        **kwargs,
    ):
        """
        Forward pass of FlowTransformerConditioned.

        Parameters
        ----------
        x : torch.Tensor
            Input tokens with shape (batch_size, seq_len, embed_dim) or (batch_size, channels, height, width)
        t : torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            Time embeddings, can be a single tensor or tuple of tensors
        z : torch.Tensor
            Context tokens with shape (batch_size, seq_len_z, ctx_embed_dim)
        **kwargs
            Additional keyword arguments

        Returns
        -------
        torch.Tensor
            Processed tokens with conditioning applied
        """
        # Process input tokens through patch embedding if needed
        if x.ndim == 4:  # Input is image format (B, C, H, W)
            x_tokens, rope = self._forward_get_tokens(x)
        else:  # Input is already token format (B, L, C)
            x_tokens = x
            # Get positional embeddings
            h, w = self._get_hw(x)
            x_tokens, rope = self._with_pos_embed(x, hw=(h, w))

        # Process context tokens if provided
        if z is not None:
            ctx_emb = self.ctx_proj(z)
            ctx_emb = self._interp_z_to_x(x_tokens, ctx_emb)

            # Apply time conditioning if enabled
            if self.time_condition and t is not None:
                # Handle tuple time embeddings (e.g., for meanflow)
                if isinstance(t, (list, tuple)):
                    # Use the first time embedding for conditioning
                    t_emb = t[0] if len(t) > 0 else t
                else:
                    t_emb = t

                # Project time embedding and get scale/shift
                t_emb = self.time_emb_proj(t_emb)
                t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)  # Add spatial dimensions

                # Split into scale and shift components
                t_scale, t_shift = t_emb.chunk(2, dim=1)

                # Apply time modulation to context embeddings
                ctx_emb = ctx_emb * (1 + t_scale) + t_shift

            # Combine input tokens with conditioned context
            # This creates a residual connection where context modulates the input
            x_tokens = x_tokens + ctx_emb

        # Apply projection if needed (for decoder mode)
        x_tokens = self._forward_proj_in(x_tokens)

        # Process through transformer layers with conditioning
        for layer in self.layers:
            # Assuming attention layers can accept additional conditioning
            # This would require modification to the attention mechanism
            if (
                hasattr(layer, "forward")
                and "ctx_emb" in layer.forward.__code__.co_varnames
            ):
                # If the layer supports context conditioning
                x_tokens = layer(x_tokens, ctx_emb=ctx_emb, rope=rope)
            else:
                # Standard forward pass
                x_tokens = layer(x_tokens, rope=rope)

        # Apply output projection if needed
        x_tokens = self._forward_proj_out(x_tokens)

        # Apply final normalization
        if self.last_norm is not None:
            x_tokens = self.last_norm(x_tokens)

        # Apply head if required
        if self.head_required:
            grid_size = self._get_hw(x_tokens)
            output = self._to_output(
                x_tokens,
                grid_size=grid_size,
                ret_2d_tokens=ret_2d_tokens,
                out_shape=out_shape,
            )
        else:
            output = x_tokens

        # Handle return format based on ret_all flag
        if not ret_all and get_intermidates is None:
            return output
        else:
            # Return tuple like parent class
            out = {
                "head_out": output,
                "x_norm_patch_tokens": output[:, self.n_reg_tokens :]
                if self.n_reg_tokens > 0
                else output,
                "x_prenorm": x_tokens,
                "x_reg_tokens": x_tokens[:, : self.n_reg_tokens, :]
                if self.n_reg_tokens > 0
                else None,
                "grid_size": self._get_hw(x_tokens),
                "mask": None,
                "ids_restore": None,
                "intermidates": None,
            }
            return output, out

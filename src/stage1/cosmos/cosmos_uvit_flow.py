"""
Hyperspectral Transformer Tokenizer with hybrid CNN / Transformer / UViT architecture.
RMSNorm + SwiGLU + EVA attention with Rope.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, NamedTuple, Optional, Tuple, TypedDict, Union, override

import accelerate
import accelerate.utils
import torch
from loguru import logger
from torch import Tensor
from typing_extensions import Annotated

from src.utilities.config_utils import (
    dataclass_from_dict,
    function_config_to_basic_types,
)
from src.utilities.network_utils.network_loading import load_weights_with_shape_check
from src.utilities.transport.tim.transition import TransitionSchedule
from src.utilities.transport.tim.transports import OT_FM, Transport

from .cosmos_tokenizer import (
    ChannelDropConfig,
    ContinuousImageTokenizer,
    ContinuousTokenizerConfig,
    EncoderDecoderConfig,
)
from .modules import blocks as cosmos_block
from .modules.layers2d import Decoder, Encoder
from .modules.naflex import NaFlexVitCfg, NaFlexVitCfgAdpoted, Transformer
from .modules.uvit_decoder import UViTDecoder, UViTDecoderConfig

LossOutput = TypedDict(
    "LossOutput",
    {"flow_loss": torch.Tensor, "q_loss": torch.Tensor},
)
type DecoderOutput = tuple[Tensor, LossOutput, dict | None]


@dataclass
class UViTTokenizerConfig(ContinuousTokenizerConfig):
    name: str = "UViTFlowTokenizer"
    # Override default values for the UViT specific configuration
    vf_on_z_or_module: str = "z"  # UViT must use z
    model: EncoderDecoderConfig = field(default_factory=EncoderDecoderConfig)
    decoder: Any = field(default_factory=UViTDecoderConfig)
    act_checkpoint: bool = False


class CosmosFlowTokenizer(ContinuousImageTokenizer):
    _no_split_modules = ["EvaBlock", "ResnetBlock", "AttnBlock"]
    _vf_on_z_or_module = "z"  # must be z if using this model

    def __init__(
        self,
        tokenizer_cfg: UViTTokenizerConfig,
        transport: Transport,
        transition_schedule_kwargs: dict = {},
        # Additional semantic transformer encoders
        trans_enc_cfgs: Optional[Tuple[NaFlexVitCfg, NaFlexVitCfgAdpoted]] = None,
        trans_dec_cfgs: Optional[Tuple[NaFlexVitCfg, NaFlexVitCfgAdpoted]] = None,
    ):
        self.tokenizer_cfg = tokenizer_cfg
        super().__init__(tokenizer_cfg)

        self.grad_checkpointing = self.tokenizer_cfg.act_checkpoint
        self._build_transformer_encoder(tokenizer_cfg, trans_enc_cfgs, trans_dec_cfgs)
        self._build_transition_schedule(transport, transition_schedule_kwargs)

    def _build_encoder_decoder(
        self, cfg: UViTTokenizerConfig, model_cfg: EncoderDecoderConfig
    ):
        model_cfg.act_checkpoint = cfg.act_checkpoint
        cfg.decoder.use_act_ckpt = cfg.act_checkpoint

        encoder = Encoder(**asdict(model_cfg))  # CNN encoder # type: ignore
        # The decoder is Flow UViT decoder
        decoder = UViTDecoder(**asdict(cfg.decoder))  # UViT decoder
        return encoder, decoder

    def _build_transition_schedule(self, transport, transition_schedule_kwargs):
        self.transition_schedule = TransitionSchedule(
            transport,
            **transition_schedule_kwargs,
        )

    def _build_transformer_encoder(
        self, tokenizer_cfg, trans_enc_cfgs=None, trans_dec_cfgs=None
    ):
        if trans_enc_cfgs is not None:
            self.trans_enc_cfg1, self.trans_enc_cfg2 = (
                trans_enc_cfgs[0],
                trans_enc_cfgs[1],
            )

        # Transformer Encoder and Decoder
        self.semantic_enc_transformer = None
        if trans_enc_cfgs is not None:
            self.semantic_enc_transformer = Transformer(
                self.trans_enc_cfg1, self.trans_enc_cfg2
            )
            self.semantic_enc_transformer.set_grad_checkpointing(
                self.grad_checkpointing
            )

        self.semantic_transformer_dec = None
        if trans_dec_cfgs is not None:
            self.trans_dec_cfg1, self.trans_dec_cfg2 = trans_dec_cfgs
            self.semantic_transformer_dec = Transformer(
                self.trans_dec_cfg1, self.trans_dec_cfg2
            )
            self.semantic_transformer_dec.set_grad_checkpointing(
                self.grad_checkpointing
            )

    def load_pretrained(self, uni_tokenizer_path: str, directly_load=True, **kwargs):
        """Init the model from the pretrained only CNN weights."""
        weights = accelerate.utils.load_state_dict(uni_tokenizer_path)

        # Directly load all weights if specified
        if directly_load:
            missing_ks, unexp_ks = load_weights_with_shape_check(
                self, weights, load_strategy="search"
            )
            if len(missing_ks) > 0 or len(unexp_ks) > 0:
                logger.warning(
                    f"Directly Loading Missing keys: {missing_ks}, Unexpected keys: {unexp_ks}"
                )
            logger.info(f"Finished directly loading pretrained weights.")
            return

    def encode(
        self, x, use_quantizer=None
    ) -> Union[Tuple[Tensor, Tensor, Union[dict, NamedTuple]], Tensor]:
        """Encode the image into latent.
        Output the latent tensor or latent, quantizer loss and loss breakdowns
        if has a quantizer.
        """
        z = self.encoder.encoder(x)
        if self.semantic_enc_transformer is not None:
            z = self.semantic_enc_transformer(z)
        h = self.encoder.quant_conv(z)

        # Quantization
        maybe_q_ret = self.apply_quantizer(h, z, use_quantizer)
        if isinstance(maybe_q_ret, tuple):
            h, q_loss, loss_breakdown = maybe_q_ret
            # NOTE: if quantizer is used, the aug z is not applied
            return h, q_loss, loss_breakdown

        # z augmentions
        h = self.latent_aug(maybe_q_ret)
        return h

    def decode(
        self,
        x: torch.Tensor,
        h: Union[torch.Tensor, tuple],
        inp_shape: Annotated[Union[torch.Size, int, tuple], "bs,c,h,w or bs,c or c"],
        mode: Literal["step", "loop"] = "step",
        clamp=False,
        ema_model: Optional["CosmosFlowTokenizer"] = None,
        sample_kwargs: dict = dict(
            num_steps=8,
            stochasticity_ratio=0.0,
            sample_type="transition",
            cfg_scale=1.0,
        ),
        ret_trajectory=False,
    ) -> DecoderOutput:
        """Decode the latent into the corresponding channels image.
        Output the reconstructed image or recon, quantizer loss and loss breakdowns.
        """
        unquant_conv, decoder = self.decoder.quant_conv, self.decoder.decoder

        # Parse the latent from encoder or generator network
        q_loss = q_loss_breakdown = None
        if self.quantizer_type is not None and isinstance(h, (tuple, list)):
            h, q_loss, q_loss_breakdown = h
        assert torch.is_tensor(h), "h should be the (quantized) latent"

        # Decoder channels
        chan = (
            inp_shape[1]
            if isinstance(inp_shape, (torch.Size, tuple, list))
            else inp_shape
        )

        # Unquantization conv
        h = unquant_conv(h)

        # Apply semantic transformer decoder if it exists
        if self.semantic_transformer_dec is not None:
            h = self.semantic_transformer_dec(h)
        null_cond_h = decoder.null_cond_h if hasattr(decoder, "null_cond_h") else None  # fmt: skip

        # Decode using the CNN decoder
        flow_loss, loss_dict = None, {}
        if mode == "step":
            if self.transition_schedule.transport.enhance_target:
                assert ema_model is not None, "EMA model is required for enhance_target"
                assert hasattr(decoder, "null_cond_h"), (
                    "The decoder must have null_cond_h for CFG"
                )

            # training: h is condition, x is the input
            noise = torch.randn_like(x)

            flow_loss, _, loss_dict, breakdowns = self.transition_schedule(
                decoder,
                ema_model.decoder.decoder if ema_model is not None else None,
                self.decoder.decoder,
                batch_size=h.shape[0],
                x=x,
                z=noise,
                model_kwargs={"inp_shape": chan, "z": h},
                ema_kwargs={"inp_shape": chan, "z": h},
                null_kwargs={"inp_shape": chan, "z": null_cond_h},
                use_dir_loss=True,  # default as in dde paper
            )
            # back to x_0
            recon = breakdowns["x0_pred"]
        elif mode == "loop":
            # eval: loop to generate reconstruction image when h is the condition.
            # no CFG
            x_init = torch.randn_like(x)
            recon = self.transition_schedule.sample(
                decoder,
                # start sampling noise
                z=x_init,
                # conditions
                y=h,
                y_null=null_cond_h,  # since no CFG, y_null is None
                T_max=1.0,
                **sample_kwargs,
            )
            if not ret_trajectory:
                recon = recon[-1]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if clamp:
            recon = recon.clamp(-1, 1)

        # Returns
        if self.quantizer_type is not None:
            losses = {"flow_loss": flow_loss, "q_loss": q_loss}
            return recon, losses, q_loss_breakdown
        else:
            return (
                recon,
                {
                    "flow_loss": flow_loss,
                    "q_loss": torch.tensor(0.0, device=x.device),
                },
                None,
            )

    @override
    def forward(
        self,
        input: torch.Tensor,
        dec_mode: Literal["step", "loop"] = "step",
        clamp: bool = False,
        ema_model: Optional["CosmosFlowTokenizer"] | None = None,
        sample_kwargs: dict = dict(
            num_steps=8,
            stochasticity_ratio=0.0,
            sample_type="transition",
            cfg_scale=1.0,
        ),
        ret_trajectory: bool = False,
    ) -> DecoderOutput:
        """Forward pass of the CosmosFlowTokenizer.

        Args:
            input: Input tensor
            mode: "step" for training, "loop" for inference
            clamp: Whether to clamp output values
            ema_model: EMA model for training
            sample_kwargs: Sampling arguments for inference
            ret_trajectory: Whether to return full trajectory

        Returns:
            Reconstructed image or (image, trajectory) if ret_trajectory=True
        """
        if cosmos_block.compile_forward_fn and cosmos_block.compile_forward_fn:
            torch.compiler.cudagraph_mark_step_begin()  # ty: ignore

        latent = self.encode(input)
        dec = self.decode(
            input,
            latent,
            input.shape,
            mode=dec_mode,
            clamp=clamp,
            ema_model=ema_model,
            sample_kwargs=sample_kwargs,
            ret_trajectory=ret_trajectory,
        )
        return dec

    @classmethod
    @override
    @function_config_to_basic_types
    def create_model(
        cls,
        tokenizer_cfg,
        transport,
        transition_schedule_kwargs={},
        trans_enc_cfg1=None,
        trans_enc_cfg2=None,
        trans_dec_cfg1=None,
        trans_dec_cfg2=None,
    ):
        tokenizer_cfg = dataclass_from_dict(UViTTokenizerConfig, tokenizer_cfg)
        trans_enc_cfg = None
        if trans_enc_cfg1 is not None and trans_enc_cfg2 is not None:
            trans_enc_cfg = (
                dataclass_from_dict(NaFlexVitCfg, trans_enc_cfg1),
                dataclass_from_dict(NaFlexVitCfgAdpoted, trans_enc_cfg2),
            )
        trans_dec_cfg = None
        if trans_dec_cfg1 is not None and trans_dec_cfg2 is not None:
            trans_dec_cfg = (
                dataclass_from_dict(NaFlexVitCfg, trans_dec_cfg1),
                dataclass_from_dict(NaFlexVitCfgAdpoted, trans_dec_cfg2),
            )

        return cls(
            tokenizer_cfg,
            transport,
            transition_schedule_kwargs,
            trans_enc_cfg,
            trans_dec_cfg,
        )

    def peft_lora_modules(
        self, conv_stem_reinit=False, conv_stem_chan: int | None = None
    ) -> list[str]:
        assert not conv_stem_reinit, "Conv stem reinit not supported for Flow tokenizer"
        lora_modules = [
            "nin_shortcut.1",
            "conv",
            "conv1",
            "conv2",
            "quant_conv",
            "post_quant_conv",
            "conv_in",
            "conv_out",
            # uvit loras
            "ada_ctx_proj",
            "norm1",
            "norm2",
            "conv_shortcut",
            "proj_in",
            "attn1.to_q",
            "attn1.to_k",
            "attn1.to_v",
            "attn1.out_proj",
            "proj",
        ]
        return lora_modules

    def peft_fully_finetune_modules(
        self, add_norms: bool = False, conv_stem_reinit=False
    ) -> list[str]:
        if isinstance(self.cfg.model.in_channels, (list, tuple)):
            return super().peft_fully_finetune_modules(add_norms, conv_stem_reinit)

        # The nested convs are in loras, return empty list to skip
        return []


def test_model_forward_backward():
    """Test the forward and backward pass of the CosmosFlowTokenizer model.

    This function tests CNN encoder + UViT decoder (no transformers),
    verifying forward/backward passes in different modes.
    """
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("Warning: CUDA is not available, running on CPU. This may be slow.")

    # Simple test configuration for CNN + UViT only
    print("\n=== Testing CNN Encoder + UViT Decoder ===")

    # Model configuration
    resolution = 64
    batch_size = 2
    in_channels = 3
    z_channels = 16

    # Create tokenizer config
    tokenizer_cfg = {
        "model": {
            "attn_resolutions": [16],
            "channels": 32,
            "channels_mult": [1, 2, 4],
            "dropout": 0.0,
            "in_channels": in_channels,
            "spatial_compression": 8,
            "num_res_blocks": 2,
            "out_channels": in_channels,
            "resolution": resolution,
            "z_channels": z_channels,
            "latent_channels": z_channels,
            "act_checkpoint": False,
            "norm_type": "gn",
            "norm_groups": 32,
            "block_name": "res_block",
            "moe_token_mixer_type": "res_block",
            "hidden_factor": 2,
            "use_residual_factor": False,
            "patch_method": "haar",
            "patch_size": 1,
            "attn_type": "none",
            "padding_mode": "reflect",
        },
        "decoder": {
            "in_channels": in_channels,
            "z_dim": z_channels,
            "channels": 128,
            "ch_mult": (1, 2, 4, 4),
            "layers_per_block": 2,
            "num_attention_heads": 4,
            "mid_nlayers": 4,
            "dropout": 0.0,
            "image_size": (resolution, resolution),
        },
        "quantizer_type": None,
        "name": "CosmosFlowTokenizer",
    }

    # Create transport instance
    transport = OT_FM(
        P_mean=0.0,
        P_std=1.6,
        sigma_d=1.0,
    )

    # Create transition schedule kwargs
    transition_schedule_kwargs = {
        "diffusion_ratio": 0.5,
        "consistency_ratio": 0.1,
        "derivative_type": "dde",
        "differential_epsilon": 0.005,
        "weight_time_type": "sqrt",
        "weight_time_tangent": True,
    }

    # Create model with only CNN encoder and UViT decoder (no transformers)
    model = CosmosFlowTokenizer.create_model(
        tokenizer_cfg=tokenizer_cfg,
        transport=transport,
        transition_schedule_kwargs=transition_schedule_kwargs,
        # No transformer configs - only CNN + UViT
    )
    model = model.to(device)
    print(f"CosmosFlowTokenizer created successfully!")

    # Create dummy input
    x = torch.randn(batch_size, in_channels, resolution, resolution).to(device)
    print(f"Input shape: {x.shape}")

    # Test eval mode
    print("\n--- Testing Evaluation Mode ---")
    model.eval()
    with torch.no_grad():
        # Test encode only
        encoded = model.encode(x)
        if isinstance(encoded, tuple):
            encoded_tensor, q_loss, loss_breakdown = encoded
            print(f"Encoded shape: {encoded_tensor.shape}, Q-loss: {q_loss}")
        else:
            encoded_tensor = encoded
            print(f"Encoded shape: {encoded_tensor.shape}")
            q_loss = None

        # Test decode in different modes
        print("Testing decode mode 'step'...")
        decoded_step = model.decode(x, encoded_tensor, x.shape, mode="step")
        if isinstance(decoded_step, tuple):
            decoded_tensor, _, _ = decoded_step
            print(f"Decoded shape (step): {decoded_tensor.shape}")
        else:
            decoded_tensor = decoded_step
            print(f"Decoded shape (step): {decoded_tensor.shape}")

        print("Testing decode mode 'loop'...")
        decoded_loop = model.decode(
            x, encoded_tensor, x.shape, mode="loop", sample_kwargs={"num_steps": 4}
        )
        if isinstance(decoded_loop, tuple):
            decoded_tensor, _, _ = decoded_loop
            print(f"Decoded shape (loop): {decoded_tensor.shape}")
        else:
            decoded_tensor = decoded_loop
            print(f"Decoded shape (loop): {decoded_tensor.shape}")

        # Test forward pass with different modes
        print("Testing forward with dec_mode='step'...")
        output_step = model(x, dec_mode="step")
        print(
            f"Forward output shape (step): {output_step[0].shape if isinstance(output_step, tuple) else output_step.shape}"
        )

        print("Testing forward with dec_mode='loop'...")
        output_loop = model(x, dec_mode="loop", sample_kwargs={"num_steps": 4})
        print(
            f"Forward output shape (loop): {output_loop[0].shape if isinstance(output_loop, tuple) else output_loop.shape}"
        )

    # Test training mode with gradients
    print("\n--- Testing Training Mode ---")
    model.train()
    x = x.detach().requires_grad_(True)

    # Forward pass with step mode (training)
    # Note: step mode requires EMA model, so we skip it for simple testing
    # Instead, test encode/decode separately
    print("Testing encode...")
    encoded = model.encode(x)
    if isinstance(encoded, tuple):
        encoded_tensor, q_loss, loss_breakdown = encoded
        print(f"Encoded shape: {encoded_tensor.shape}, Q-loss: {q_loss}")
    else:
        encoded_tensor = encoded
        print(f"Encoded shape: {encoded_tensor.shape}")

    # Test decode in loop mode (doesn't require EMA)
    print("Testing decode in loop mode...")
    model.zero_grad()
    decoded, loss_dict, _ = model.decode(
        x, encoded_tensor, x.shape, mode="step", sample_kwargs={"num_steps": 4}
    )

    # Compute loss and backward
    flow_loss = loss_dict.get("flow_loss", 0.0) or 0.0
    assert torch.is_tensor(flow_loss)
    flow_loss.backward()

    print(f"Training - Flow loss: {flow_loss.item():.6f}")

    # Check gradients
    grad_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_count += 1
    print(f"Parameters with gradients: {grad_count}")

    if x.grad is not None:
        print(f"Input gradient norm: {x.grad.norm().item():.6f}")

    print("\n=== Test completed successfully ===")


if __name__ == "__main__":
    """
    python -m src.stage1.cosmos.cosmos_uvit_flow
    """
    test_model_forward_backward()

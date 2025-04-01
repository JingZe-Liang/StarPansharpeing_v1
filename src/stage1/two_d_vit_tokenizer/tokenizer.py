import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from src.stage1.discretization.collections.bsq import BinarySphericalQuantizer
from src.stage1.discretization.collections.simple_vq import VectorQuantizer
from src.stage1.two_d_vit_tokenizer.transformer_tokenizer import (
    TransformerDecoder,
    TransformerEncoder,
)
from src.stage1.utilities import LogitLaplaceLoss


class VITVQModel(nn.Module):
    def __init__(
        self,
        vitconfig,
        n_embed,
        embed_dim,
        l2_norm=False,
        logit_laplace=False,
        ckpt_path=None,
        ignore_keys=[],
        grad_checkpointing=False,
        selective_checkpointing=False,
        clamp_range=(0, 1),
        dvitconfig=None,
    ):
        super().__init__()
        self.encoder = TransformerEncoder(**vitconfig)
        dvitconfig = vitconfig if dvitconfig is None else dvitconfig
        self.decoder = TransformerDecoder(**dvitconfig, logit_laplace=logit_laplace)
        if self.training and grad_checkpointing:
            self.encoder.set_grad_checkpointing(True, selective=selective_checkpointing)
            self.decoder.set_grad_checkpointing(True, selective=selective_checkpointing)

        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.l2_norm = l2_norm
        self.setup_quantizer()

        self.quant_embed = nn.Linear(
            in_features=vitconfig["width"], out_features=embed_dim
        )
        self.post_quant_embed = nn.Linear(
            in_features=embed_dim, out_features=dvitconfig["width"]
        )
        self.l2_norm = l2_norm
        self.logit_laplace = logit_laplace
        self.clamp_range = clamp_range
        if self.logit_laplace:
            self.logit_laplace_loss = LogitLaplaceLoss()

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def setup_quantizer(self):
        self.quantizer = VectorQuantizer(
            self.n_embed,
            self.embed_dim,
            l2_norm=self.l2_norm,
            beta=0.25,
            input_format="blc",
        )

    def init_from_ckpt(self, ckpt_path, ignore_keys=[]):
        try:
            logger.info(f"Try EMA state_dict first...")
            state_dict = torch.load(ckpt_path, map_location="cpu")["ema_state_dict"]
            state_dict = {
                k[14:]: v
                for k, v in state_dict.items()
                if k.startswith("module.module.")
            }
        except (KeyError, AttributeError):
            logger.info(
                f"Failed to find EMA state_dict, try vanilla state_dict instead"
            )
            state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
            state_dict = {
                k[7:]: v for k, v in state_dict.items() if k.startswith("module.")
            }
        filtered_state_dict = {
            k: v
            for k, v in state_dict.items()
            if all([not k.startswith(ig) for ig in ignore_keys])
        }
        missing_keys, unexpected_keys = self.load_state_dict(
            filtered_state_dict, strict=False
        )
        logger.info(f"Restored from {ckpt_path}")
        logger.info(f"missing_keys: {missing_keys}")
        logger.info(f"unexpected_keys: {unexpected_keys}")

    def encode(self, x, skip_quantize=False):
        h = self.encoder(x)
        h = self.quant_embed(h)
        if skip_quantize:
            assert not self.training, "skip_quantize should be used in eval mode only."
            if self.l2_norm:
                h = F.normalize(h, dim=-1)
            return h, {}, {}
        quant, loss, info = self.quantizer(h)
        return quant, loss, info

    def decode(self, quant):
        h = self.post_quant_embed(quant)
        x = self.decoder(h)
        return x

    def clamp(self, x):
        if self.logit_laplace:
            dec, _ = x.chunk(2, dim=1)
            x = self.logit_laplace_loss.unmap(F.sigmoid(dec))
        else:
            x = x.clamp_(self.clamp_range[0], self.clamp_range[1])
        return x

    def forward(self, input, skip_quantize=False):
        if self.logit_laplace:
            input = self.logit_laplace_loss.inmap(input)
        quant, loss, info = self.encode(input, skip_quantize=skip_quantize)
        dec = self.decode(quant)
        if self.logit_laplace:
            dec, lnb = dec.chunk(2, dim=1)
            logit_laplace_loss = self.logit_laplace_loss(dec, lnb, input)
            info.update({"logit_laplace_loss": logit_laplace_loss})
            dec = self.logit_laplace_loss.unmap(F.sigmoid(dec))
        else:
            dec = dec.clamp_(self.clamp_range[0], self.clamp_range[1])
        return dec, loss, info

    def get_last_layer(self):
        return self.decoder.conv_out.weight


class VITBSQModel(VITVQModel):
    def __init__(
        self,
        vitconfig,
        embed_dim,
        embed_group_size=9,
        l2_norm=False,
        logit_laplace=False,
        ckpt_path=None,
        ignore_keys=[],
        grad_checkpointing=False,
        selective_checkpointing=False,
        clamp_range=(0, 1),
        dvitconfig=None,
        beta=0.0,
        gamma0=1.0,
        gamma=1.0,
        zeta=1.0,
        persample_entropy_compute="group",
        cb_entropy_compute="group",
        post_q_l2_norm=False,
        inv_temperature=1.0,
    ):
        # set quantizer params
        self.beta = beta  # commit loss
        self.gamma0 = gamma0  # entropy
        self.gamma = gamma  # entropy penalty
        self.zeta = zeta  # lpips
        self.embed_group_size = embed_group_size
        self.persample_entropy_compute = persample_entropy_compute
        self.cb_entropy_compute = cb_entropy_compute
        self.post_q_l2_norm = post_q_l2_norm
        self.inv_temperature = inv_temperature

        # call init
        super().__init__(
            vitconfig,
            2**embed_dim,
            embed_dim,
            l2_norm=l2_norm,
            logit_laplace=logit_laplace,
            ckpt_path=ckpt_path,
            ignore_keys=ignore_keys,
            grad_checkpointing=grad_checkpointing,
            selective_checkpointing=selective_checkpointing,
            clamp_range=clamp_range,
            dvitconfig=dvitconfig,
        )

    def setup_quantizer(self):
        self.quantizer = BinarySphericalQuantizer(
            self.embed_dim,
            self.beta,
            self.gamma0,
            self.gamma,
            self.zeta,
            group_size=self.embed_group_size,
            persample_entropy_compute=self.persample_entropy_compute,
            cb_entropy_compute=self.cb_entropy_compute,
            input_format="blc",
            l2_norm=self.post_q_l2_norm,
            inv_temperature=self.inv_temperature,
        )

    def encode(self, x, skip_quantize=False):
        h = self.encoder(x)
        h = self.quant_embed(h)
        if self.l2_norm:
            h = F.normalize(h, dim=-1)
        if skip_quantize:
            assert not self.training, "skip_quantize should be used in eval mode only."
            return h, {}, {}
        quant, loss, info = self.quantizer(h)
        return quant, loss, info


if __name__ == "__main__":
    # Test config from YAML
    vit_config = {
        "width": 768,
        "layers": 12,
        "heads": 12,
        "patch_size": 8,
        "image_size": 256,
        "mlp_ratio": 4,
        "drop_rate": 0.0,
    }

    torch.cuda.set_device(1)
    # Initialize model with YAML config
    model = VITBSQModel(
        vitconfig=vit_config,
        embed_dim=36,
        embed_group_size=1,
        l2_norm=True,
        persample_entropy_compute="analytical",
        post_q_l2_norm=True,
        logit_laplace=False,
        beta=0.25,
    ).cuda()

    # Test forward pass
    x = torch.randn(1, 3, 256, 256).cuda()  # Batch of 1, 3 channels, 256x256
    dec, loss, info = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {dec.shape}")
    print(f"Loss: {loss}")
    print(f"Info: {info}")

    # Test encode/decode
    model.eval()
    quant, _, _ = model.encode(x, skip_quantize=True)
    reconstructed = model.decode(quant)
    print(
        f"Reconstruction difference: {torch.abs(x - reconstructed).mean().item():.4f}"
    )

    # Test clamping
    test_values = torch.tensor([[-1.0, 0.5, 2.0]])
    clamped = model.clamp(test_values)
    print(f"Clamping test: {test_values} -> {clamped}")

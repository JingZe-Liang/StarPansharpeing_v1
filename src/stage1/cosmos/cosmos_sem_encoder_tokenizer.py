import math
import sys
from pathlib import Path
from typing import Literal, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from timm.layers import DropPath, trunc_normal_
from timm.layers.weight_init import lecun_normal_
from timm.models.convnext import _init_weights as init_weights_convnext
from timm.models.vision_transformer import get_init_weights_vit, named_apply

from src.utilities.config_utils import to_object_recursive

from .modules.blocks import AdaptiveInputConvLayer, AdaptiveOutputConvLayer
from .modules.layers2d import Decoder as LightWeightUpsampler
from .modules.transformer import (
    AdaptivePatchEmbedding,
    Attention,
    AttentionBlock,
    TransformerTokenizer,
)


def init_weights_vit_jax_custom(
    module: nn.Module, name: str = "", head_bias: float = 0.0
) -> None:
    """ViT weight initialization, matching JAX (Flax) impl.

    Args:
        module: Module to initialize.
        name: Module name for context.
        head_bias: Bias value for head layer.
    """
    if isinstance(module, nn.Linear):
        if name.startswith("head"):
            nn.init.trunc_normal_(module.weight, mean=0, std=0.08)
            nn.init.constant_(module.bias, head_bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.normal_(
                    module.bias, std=1e-6
                ) if "mlp" in name else nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        # lecun_normal_(module.weight)
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()


def load_repa_dino_v3_model(
    weight_path: str | Path | None = None,
    model_name: str | None = "dinov3_vitl16",
    pretrained_on: Literal["satellite", "web"] = "satellite",
    compile=False,
) -> torch.nn.Module | torch._dynamo.OptimizedModule:
    """
    import torch

    REPO_DIR = <PATH/TO/A/LOCAL/DIRECTORY/WHERE/THE/DINOV3/REPO/WAS/CLONED>

    # DINOv3 ViT models pretrained on web images
    dinov3_vits16 = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
    dinov3_vits16plus = torch.hub.load(REPO_DIR, 'dinov3_vits16plus', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
    dinov3_vitb16 = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
    dinov3_vitl16 = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
    dinov3_vith16plus = torch.hub.load(REPO_DIR, 'dinov3_vith16plus', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
    dinov3_vit7b16 = torch.hub.load(REPO_DIR, 'dinov3_vit7b16', source='local', weights=<CHECKPOINT/URL/OR/PATH>)

    # DINOv3 ConvNeXt models pretrained on web images
    dinov3_convnext_tiny = torch.hub.load(REPO_DIR, 'dinov3_convnext_tiny', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
    dinov3_convnext_small = torch.hub.load(REPO_DIR, 'dinov3_convnext_small', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
    dinov3_convnext_base = torch.hub.load(REPO_DIR, 'dinov3_convnext_base', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
    dinov3_convnext_large = torch.hub.load(REPO_DIR, 'dinov3_convnext_large', source='local', weights=<CHECKPOINT/URL/OR/PATH>)

    # DINOv3 ViT models pretrained on satellite imagery
    dinov3_vitl16 = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', weights=<CHECKPOINT/URL/OR/PATH>)
    dinov3_vit7b16 = torch.hub.load(REPO_DIR, 'dinov3_vit7b16', source='local', weights=<CHECKPOINT/URL/OR/PATH>)

    The pretrained weights are placed as follows:

    src/stage1/utilities/losses/dinov3/weights
    ├── remote_sensing_image_pretrained_SAT_493M
    │   ├── dinov3_vit7b16_pretrain_sat493m-a6675841.pth
    │   └── dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
    └── web_image_pretrained_lvd
        ├── dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth
        ├── dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth
        ├── dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth
        ├── dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth
        ├── dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
        ├── dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth
        ├── dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
        ├── dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth
        ├── dinov3_vits16_pretrain_lvd1689m-08c60483.pth
        └── download_dinov3_weights.py
    """
    # repo_dir = Path(__file__).parents[1] / "dinov3"
    repo_dir = Path("src/stage1/utilities/losses/dinov3")
    assert repo_dir.exists(), (
        f"DINOv3 repo directory {repo_dir} does not exist. Please git clone from https://github.com/facebookresearch/dinov3"
    )

    if model_name is None and weight_path is not None:
        stem = Path(weight_path).stem
        model_name = "_".join(stem.split("_", 2))
    elif weight_path is None and model_name is not None:
        model_type_dir = {
            "web": "web_image_pretrained_lvd",
            "satellite": "remote_sensing_image_pretrained_SAT_493M",
        }[pretrained_on]
        weight_dir = repo_dir / "weights" / model_type_dir
        # search the weight path
        paths = weight_dir.rglob("*.pth")
        for p in paths:
            # avoid 'dinov3_vits16' and 'dinov3_vits16plus'
            search_name = model_name + "_pretrain"
            if search_name in p.stem:
                weight_path = str(p)
                break
        assert weight_path is not None, (
            f"can not find weight {model_name=} at {weight_dir}"
        )
    elif weight_path is None and model_name is None:
        raise ValueError("Either model_name or weight_path must be specified.")

    assert weight_path is not None, f"{weight_path=} does not exists"
    logger.info(
        f"[Dino v3 in REPA]: use Dino v3 model: {model_name} loaded from {weight_path}."
    )
    assert Path(weight_path).exists(), "Dino v3 model weight path does not exists"
    sys.path.append(str(repo_dir))
    dino_model = torch.hub.load(
        repo_dir, model_name, source="local", weights=weight_path
    )
    dino_model = cast(nn.Module, dino_model)
    if compile:
        dino_model = torch.compile(dino_model, mode="reduce-overhead")
        dino_model = cast(torch._dynamo.OptimizedModule, dino_model)
        logger.info("[Dino v3 model]: compiled model done")
    return dino_model


class Dinov3RAE(nn.Module):
    _no_split_modules = ["AttentionBlock", "SelfAttentionBlock"]

    # scaling factor for evaluation
    scaling_factor: torch.Tensor | None = None
    shift_factor: torch.Tensor | None = None

    # Compatibility with trainer (actually not used)
    _use_repa_loss: bool = False
    _use_vf_loss: bool = False
    _vf_on_z_or_module: Literal["z", "module"] = "module"
    _hook_module: str = "decoder.decoder.mid.block_2"  # "decoder.decoder.up.1.block.2"
    _dino_feature_dim: int = 768  # [768, 1024]
    _hook_feature: torch.Tensor | None = None
    z: torch.Tensor | None = None  # the latent z
    supported_cached_hiddens: list[str] = ["z"]

    def __init__(self, cfg):
        super().__init__()
        self.encoder = load_repa_dino_v3_model(
            model_name=cfg.dino_cfg.model_name,
            weight_path=cfg.dino_cfg.weight_path,
            compile=cfg.dino_cfg.compile,
            pretrained_on=cfg.dino_cfg.pretrained_on,
        )

        self.in_chan = cfg.in_chan
        self.out_chan = cfg.out_chan
        self.img_size = cfg.img_size
        self.grid_size = cfg.img_size // 16  # dino patch size
        self.encoder_patch_size = self.encoder.patch_embed.patch_size[0]
        dec_cfg = cfg.decoder_cfg
        lw_up_cfg = cfg.light_upsampler_cfg

        # FIXME: Should add encoder too, the patcher is too large when using nested conv.
        self.decoder = TransformerTokenizer(
            in_chan=dec_cfg.in_chan,  # dino out channels
            embed_dim=dec_cfg.embed_dim,
            img_size=self.grid_size,
            patch_size=dec_cfg.patch_size,
            out_patch_size=dec_cfg.out_patch_size,  # 1
            depth=dec_cfg.depth,
            num_heads=dec_cfg.num_heads,
            mlp_ratio=dec_cfg.mlp_ratio,
            n_reg_tokens=dec_cfg.n_reg_tokens,
            norm_layer=dec_cfg.norm_layer,
            pe_type=dec_cfg.pe_type,
            out_chan=dec_cfg.out_chan,
            drop_path=dec_cfg.drop_path,
            rope_kwargs=to_object_recursive(dec_cfg.rope_kwargs),
            projections={"input": None, "output": None},
            patcher_type="linear" if dec_cfg.patch_size == 1 else "patch_embedder",
            # use light weight upsample layer
            # not unpatcher (c * patch_size ** 2) is too large
            head="linear",
        )
        self.light_upsampler = LightWeightUpsampler(
            out_channels=self.out_chan,  # lw_up_cfg.out_chan,
            channels=lw_up_cfg.channels,
            channels_mult=lw_up_cfg.channels_mult,
            num_res_blocks=lw_up_cfg.num_res_blocks,
            dropout=0.0,
            resolution=self.img_size,
            z_channels=dec_cfg.out_chan,  # out channel from transformer decoder
            attn_resolutions=[],  # no attention here
            spatial_compression=self.encoder_patch_size,
            upsample_type="RepeatConv",
            attn_type="none",
            padding_mode="reflect",
            norm_type="gn",
            norm_groups=32,
            patch_size=lw_up_cfg.patch_size,  # 1, haar patcher
            adaptive_mode="interp",
        )

        # RAE noising strategy
        self.noise_tau = cfg.noise_tau  # 0.8

        # Make the encoder patch embedder adaptive to input channels
        self._make_encoder_patch_embeder_adaptive(cfg)
        # freeze the encoder despite the patch embedder
        # self.encoder.requires_grad_(False)
        # self.encoder.patch_embed.requires_grad_(True)

        # Init decoder and upsampler weights
        named_apply(init_weights_vit_jax_custom, self.decoder)
        named_apply(init_weights_convnext, self.light_upsampler)
        logger.info(f"[Dinov3RAE]: init the decoder and light upsampler weights done.")

        # Self-distillation

    def _make_encoder_patch_embeder_adaptive(self, cfg):
        encoder_patcher = self.encoder.patch_embed
        ep = encoder_patcher.patch_size[0]

        # conv weights and bias
        epw = encoder_patcher.proj.weight  # (embed_dim, 3, k, k)
        epb = encoder_patcher.proj.bias  # (embed_dim,)
        embed_dim = epw.shape[0]

        # assert no norm
        assert isinstance(encoder_patcher.norm, nn.Identity), (
            "encoder_patcher.norm must be nn.Identity"
        )

        # set to adaptive conv
        adaptive_patcher = AdaptivePatchEmbedding(
            cfg.in_chan, embed_dim, patch_size=ep, output_fmt="NHWC"
        )

        # set the weights and bias
        # repeats the original weights
        assert cfg.in_chan > 3, (
            f"the input channels must be larger than 3, but got {cfg.in_chan}"
        )
        repeat_times = math.ceil(cfg.in_chan / 3)
        in_chans_l = repeat_times * 3
        adaptive_w = adaptive_patcher.proj.conv.weight  # (embed_dim, in_chan, k, k)
        adaptive_w.data.copy_(
            epw.repeat(1, repeat_times, 1, 1)[:, : cfg.in_chan]
        )  # (embed_dim, in_chan, k, k)
        adaptive_patcher.proj.conv.bias.data.copy_(epb)  # (embed_dim,)
        self.encoder.patch_embed = adaptive_patcher  # replace

        logger.info(
            f"[Dinov3RAE]: replace the encoder patch embedder to adpative patch embedder, "
            f"with input channels {cfg.in_chan}, patch size {ep}, "
            f"weights are partially copied."
        )

    def encode(self, x):
        h = self.encoder(x, is_training=True)["x_norm_patchtokens"]
        assert h.ndim == 3, (
            f"the encoder output should be (bs, n_patches, dim), but got {h.shape}"
        )
        if self.training and self.noise_tau > 0:
            h = self.noising(h)
        return h

    def noising(self, h):
        """
        Taken from RAE paper.
        Make the decoder more robust to noise since we are using the FM model.
        """
        noise_sigma = self.noise_tau * torch.rand(
            (h.size(0),) + (1,) * (len(h.shape) - 1), device=h.device
        )  # (bs, 1, 1, 1)
        noise = noise_sigma * torch.randn_like(h)
        return h + noise

    def decode(self, h, inp_shape, clamp=False):
        out_chan = (
            inp_shape[1] if isinstance(inp_shape, (torch.Size, tuple)) else inp_shape
        )
        dec_hidden = self.decoder(h, ret_2d_tokens=True, ret_all=False)
        dec = self.light_upsampler(dec_hidden, out_chan)
        return dec.clamp(-1.0, 1.0) if clamp else dec

    def forward(self, x):
        inp_shape = x.shape
        h = self.encode(x)
        # breakpoint()
        dec = self.decode(h, inp_shape, clamp=True)
        return dec

    def get_last_layer(self):
        return self.decoder.head.weight

    def parameters(self):
        return (
            list(self.encoder.patch_embed.parameters())
            + list(self.decoder.parameters())
            + list(self.decoder_out_layer.parameters())
        )

    def named_parameters(self):
        name_ps = []
        for n, p in super().named_parameters():
            if not n.startswith("encoder") or "patch_embed" in n:
                name_ps.append((n, p))
            else:
                logger.debug(f"[Dinov3RAE]: ignore {n} in named_parameters")
        return name_ps


@logger.catch()
def test_dinov3_encoder_tokenizer():
    from fvcore.nn import parameter_count_table
    from omegaconf import OmegaConf

    cfg_str_basic = "in_chan=256 out_chan=256 noise_tau=0.8 img_size=512"
    cfg_basic = OmegaConf.from_dotlist(cfg_str_basic.split(" "))

    cfg_str_dino = (
        "model_name=dinov3_vitb16 weight_path=null pretrained_on='web' compile=False"
    )
    cfg_dino = OmegaConf.from_dotlist(cfg_str_dino.split(" "))

    cfg_str_dec = (
        "in_chan=768 embed_dim=768 patch_size=1 out_patch_size=1 depth=16 num_heads=16 mlp_ratio=4 "
        "norm_layer='rmsnorm' pe_type='rope' n_reg_tokens=0 out_chan=768 drop_path=0.1 "
        "rope_kwargs.rope_theta=10000.0 "
    )
    cfg_dec = OmegaConf.from_dotlist(cfg_str_dec.split(" "))

    cfg_str_upsampler = (
        "out_chan=256 channels=512 channels_mult=[1,1,1,1] num_res_blocks=1 "
        "patch_size=1"
    )
    cfg_upsampler = OmegaConf.from_dotlist(cfg_str_upsampler.split(" "))

    cfg = OmegaConf.merge(
        cfg_basic,
        {
            "dino_cfg": cfg_dino,
            "decoder_cfg": cfg_dec,
            "light_upsampler_cfg": cfg_upsampler,
        },
    )
    # print(cfg)

    tokenizer = Dinov3RAE(cfg)
    print(tokenizer)

    # n_learnable_params = 0
    # for p in tokenizer.parameters():
    #     if p.requires_grad:
    #         n_learnable_params += p.numel()
    # print(f"n_learnable_params: {n_learnable_params / 1e6:.2f}M")

    print(parameter_count_table(tokenizer))

    x = torch.randn(1, 128, 512, 512)
    with torch.no_grad():
        out = tokenizer(x)
    print(out)


if __name__ == "__main__":
    """
    export LOVELY_TENSORS=1
    python -m src.stage1.cosmos.cosmos_sem_encoder_tokenizer
    """
    test_dinov3_encoder_tokenizer()

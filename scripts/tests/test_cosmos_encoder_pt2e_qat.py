import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.stage1.cosmos.cosmos_tokenizer import (
    ContinuousImageTokenizer,
    ContinuousTokenizerConfig,
    EncoderDecoderConfig,
    vae_f8_config,
)


def _build_small_tokenizer_from_vae_f8_config() -> ContinuousImageTokenizer:
    cfg = vae_f8_config(
        in_chans=3,
        latent_chans=4,
        z_chans=8,
        vae_factor=4,
        patch_size=1,
        quantizer_type=None,
        use_repa_loss=False,
    )
    cfg.model.channels = 16
    cfg.model.channels_mult = [2, 2]
    cfg.model.num_res_blocks = 1
    cfg.model.resolution = 16
    cfg.model.norm_groups = 8

    model_cfg_raw = OmegaConf.to_container(cfg.model, resolve=True)
    assert isinstance(model_cfg_raw, dict)
    model_cfg = EncoderDecoderConfig(**dict(model_cfg_raw))
    tokenizer_cfg = ContinuousTokenizerConfig(
        model=model_cfg,
        quantizer_type=None,
        use_repa_loss=False,
        use_vf_loss=False,
        z_factor=1,
    )
    return ContinuousImageTokenizer(tokenizer_cfg)


def _count_encoder_conv2d_modules(tokenizer: ContinuousImageTokenizer) -> int:
    return sum(1 for module in tokenizer.encoder.encoder.modules() if isinstance(module, nn.Conv2d))


def _encoder_conv_weight_keys(tokenizer: ContinuousImageTokenizer) -> set[str]:
    return {
        f"encoder.{name}.weight"
        for name, module in tokenizer.encoder.encoder.named_modules()
        if isinstance(module, nn.Conv2d)
    }


def _count_fake_quant_or_observer_modules(model: nn.Module) -> int:
    count = 0
    for _, module in model.named_modules():
        module_name = type(module).__name__
        if "FakeQuantize" in module_name or "Observer" in module_name:
            count += 1
    return count


def test_pt2e_qat_targets_full_tokenizer_encoder_only() -> None:
    tokenizer = _build_small_tokenizer_from_vae_f8_config().cpu().eval()

    expected_encoder_conv_count = _count_encoder_conv2d_modules(tokenizer)
    expected_encoder_weight_keys = _encoder_conv_weight_keys(tokenizer)
    assert expected_encoder_conv_count > 0
    assert isinstance(tokenizer.encoder.quant_conv, nn.Conv2d)

    qat_model = ContinuousImageTokenizer.quantization_aware_training_model(
        tokenizer,
        quantize_type="pt2e_qat_prepare",
        ignore_layer_names=[],
        example_inputs=(torch.randn(1, 3, 16, 16),),
    )

    assert qat_model is tokenizer
    assert isinstance(qat_model, ContinuousImageTokenizer)
    assert getattr(qat_model, "_pt2e_qat_stage", None) == "prepare"
    assert getattr(qat_model.encoder.encoder, "_pt2e_qat_stage", None) == "prepare"
    assert _count_fake_quant_or_observer_modules(qat_model) > 0

    for _ in range(2):
        enc_out = qat_model.encode(torch.randn(1, 3, 16, 16))
        assert enc_out.latent.shape == (1, 4, 4, 4)

    inter_feat_out = qat_model.encode_with_intermediate_features(torch.randn(1, 3, 16, 16))
    assert inter_feat_out.encoded.shape == (1, 4, 4, 4)

    int8_model = ContinuousImageTokenizer.quantization_aware_training_model(
        qat_model,
        quantize_type="pt2e_qat_convert",
        ignore_layer_names=[],
    )
    assert int8_model is qat_model
    assert getattr(int8_model, "_pt2e_qat_stage", None) == "convert"
    assert getattr(int8_model.encoder.encoder, "_pt2e_qat_stage", None) == "convert"

    graph_module = int8_model.encoder.encoder.graph_module
    quantized_targets = [
        str(node.target) for node in graph_module.graph.nodes if "quantized_decomposed" in str(node.target)
    ]
    assert len(quantized_targets) > 0

    converted_state_dict = graph_module.state_dict()
    encoder_weight_dtypes = {
        key: converted_state_dict[key].dtype for key in expected_encoder_weight_keys if key in converted_state_dict
    }
    assert len(encoder_weight_dtypes) == expected_encoder_conv_count
    assert all(dtype == torch.float32 for dtype in encoder_weight_dtypes.values())
    assert int8_model.encoder.quant_conv.weight.dtype == torch.float32

    per_tensor_act_quants = [target for target in quantized_targets if "quantize_per_tensor" in target]
    assert len(per_tensor_act_quants) >= expected_encoder_conv_count

    output = int8_model.encode(torch.randn(1, 3, 16, 16))
    assert output.latent.shape == (1, 4, 4, 4)


def test_create_model_applies_qat_prepare_to_encoder_only() -> None:
    cfg = vae_f8_config(
        in_chans=3,
        latent_chans=4,
        z_chans=8,
        vae_factor=4,
        patch_size=1,
        quantizer_type=None,
        use_repa_loss=False,
    )
    cfg.model.channels = 16
    cfg.model.channels_mult = [2, 2]
    cfg.model.num_res_blocks = 1
    cfg.model.resolution = 16
    cfg.model.norm_groups = 8
    cfg.quantize_to_int8 = True
    cfg.qat_quantize_type = "pt2e_qat_prepare"
    cfg.qat_ignore_layer_names = ["conv_in"]

    tokenizer = ContinuousImageTokenizer.create_model(config=cfg).cpu()

    assert isinstance(tokenizer, ContinuousImageTokenizer)
    assert getattr(tokenizer, "_pt2e_qat_stage", None) == "prepare"
    assert getattr(tokenizer.encoder.encoder, "_pt2e_qat_stage", None) == "prepare"

from pathlib import Path
import sys

import hydra
import torch.nn as nn
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def test_cosmos_hybrid_multi_teacher_cache_layers_from_wrapper_cfg():
    tokenizer_cfg = OmegaConf.load(
        "scripts/configs/tokenizer_gan/tokenizer/cosmos_hybrid_phi_s_encoder_decoder_vit.yaml"
    )
    wrapper_cfg = OmegaConf.load("scripts/configs/tokenizer_gan/hybrid_cosmos_tokenizer_phi_s_encoder_decoder_vit.yaml")

    tokenizer_cfg.trans_enc_cfg.depth = 2
    tokenizer_cfg.trans_dec_cfg.depth = 1
    tokenizer_cfg.trans_enc_cfg.drop_path_rate = 0.0
    tokenizer_cfg.trans_dec_cfg.drop_path_rate = 0.0

    tokenizer_cfg.cnn_cfg.use_repa_loss = wrapper_cfg.tokenizer.cnn_cfg.use_repa_loss
    tokenizer_cfg.cnn_cfg.use_vf_loss = wrapper_cfg.tokenizer.cnn_cfg.use_vf_loss
    tokenizer_cfg.cnn_cfg.quantizer_type = wrapper_cfg.tokenizer.cnn_cfg.quantizer_type
    tokenizer_cfg.trans_enc_cfg.pretrained_type = wrapper_cfg.tokenizer.trans_enc_cfg.pretrained_type
    tokenizer_cfg.distillation_cfg.phis_student_source = wrapper_cfg.tokenizer.distillation_cfg.phis_student_source
    tokenizer_cfg.distillation_cfg.semantic_feature_dim = wrapper_cfg.tokenizer.distillation_cfg.semantic_feature_dim
    tokenizer_cfg.distillation_cfg.teacher_proj_dims = wrapper_cfg.tokenizer.distillation_cfg.teacher_proj_dims

    model = hydra.utils.instantiate(tokenizer_cfg)

    assert model.low_lvl_repa_proj_is_multi is True
    assert model.sem_repa_proj_is_multi_layer_cached is True

    for teacher_name in model._teacher_names:
        teacher_proj = model._repa_proj[teacher_name]
        assert isinstance(teacher_proj["low_lvl_repa_proj"], nn.ModuleList)
        assert isinstance(teacher_proj["sem_repa_proj"], nn.ModuleList)

#!/bin/bash
export CUDA_VISIBLE_DEVICES=0 MODEL_COMPILED=0 DISABLE_XFORMERS=1
python -m accelerate.commands.launch --num_processes=1 \
  src/stage2/generative/Sana/train_scripts/train.py \
  --config_path src/stage2/generative/Sana/configs/sana_controlnet_config/Sana_600M_RS_tokenizer_controlnet.yaml \
  --work_dir src/stage2/generative/Sana/output \
  --name sana_controlnet_litdata_run2

#!/usr/bin

torchrun \
--nnodes=${nnodes} \
--node_rank=${node_rank} \
--master_addr=${master_addr} \
--master_port=${master_port} \
--nproc_per_node=${nproc_per_node} \
main.py \
--local_bs 64 \
--vae_local_bs 64 \
--vocab_size 32768 \
--num_codebooks 8 \
--report_wandb True \
--model 'vitamin_large' \
--exp_name 'unitok_pretrain' \
--vis_img_dir 'assets/vis_imgs/' \
"$@"

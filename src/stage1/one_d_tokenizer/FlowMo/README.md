## Flow to the Mode: Mode-Seeking Diffusion Autoencoders for State-of-the-Art Image Tokenization

This repo contains the code for our FlowMo model training and evaluation. Check out our paper for more details: https://www.arxiv.org/abs/2503.11056

<p align="center">
  <img src="demo.gif" alt="sample GIF" />
</p>

## Get the code
```
git clone https://github.com/kylesargent/FlowMo
cd FlowMo
```

## Install the requirements
```
conda create -n FlowMo python=3.13.2 pip
conda activate FlowMo
pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```
Note: The torch and cuda version above were what we used to produce the paper results. But we've tested torch 2.4, 2.5, 2.6 and attained similar performance with all.

## Prepare the data
The dataset is read directly from the standard public ImageNet tar files. I have created indices for these tarfiles so that there is no data preprocessing needed. Please download the datasets and indices with the commands below. If you don't donwload them at the toplevel (like FlowMo/*.tar), you need to modify the corresponding path in `flowmo/configs/base.yaml`.

```
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
wget https://huggingface.co/ksarge/FlowMo/resolve/main/imagenet_train_index_overall.json
wget https://huggingface.co/ksarge/FlowMo/resolve/main/imagenet_val_index_overall.json
```

## Train your models
FlowMo is trained in two stages. The first stage is standard diffusion autoencoder training. In the second stage, we drop the batch size and LR and backpropagate through the sampling chain with a sample-level loss. For more details, please check the paper. For post-training, it is recommended to save more checkpoints and to concurrently run the continuous evaluator. Then you can select the best checkpoint based on early-stopping to counteract eventual reward hacking. <strong>For post-training, please supply your checkpoint path from pre-training.</strong>

The training commands for FlowMo-Lo are below. It is recommended to pre-train FlowMo-Lo for ~130 epochs minimum to match the paper result, but you may increase `trainer.max_steps` for better performance.
```
torchrun --nproc-per-node=8 -m flowmo.train \
    --experiment-name "flowmo_lo_pretrain" \
    model.context_dim=18 model.codebook_size_for_entropy=9 \
    trainer.max_steps=1300000

torchrun --nproc-per-node=8 -m flowmo.train \
    --experiment-name "flowmo_lo_posttrain" \
    --resume-from-ckpt ... \
    model.context_dim=18 model.codebook_size_for_entropy=9 \
    trainer.max_steps=1325000
    opt.lr=0.00005 \
    data.batch_size=8 \
    opt.n_grad_acc=2 \
    model.posttrain_sample=true \
    opt.lpips_mode='resnet' \
    opt.lpips_weight=0.01 \
    trainer.log_every=100 \
    trainer.checkpoint_every=5000 \
    trainer.keep_every=5000 \
```
The training commands for FlowMo-Hi are below. It is recommended to pre-train FlowMo-Hi for ~80 epochs minimum to match the paper result, but you may increase `trainer.max_steps` for better performance. 
```
torchrun --nproc-per-node=8 -m flowmo.train \
    --experiment-name "flowmo_hi_pretrain" \
    model.context_dim=56 model.codebook_size_for_entropy=14 \
    trainer.max_steps=800000

torchrun --nproc-per-node=8 -m flowmo.train \
    --experiment-name "flowmo_hi_posttrain" \
    --resume-from-ckpt ... \
    model.context_dim=56 model.codebook_size_for_entropy=14 \
    trainer.max_steps=825000
    opt.lr=0.00005 \
    data.batch_size=8 \
    opt.n_grad_acc=2 \
    model.posttrain_sample=true \
    opt.lpips_mode='resnet' \
    opt.lpips_weight=0.01 \
    trainer.log_every=100 \
    trainer.checkpoint_every=5000 \
    trainer.keep_every=5000 \
```

## Evaluation
To evaluate an experiment (continuously as new checkpoints are added, or just latest checkpoint if continuous=False), run

```
torchrun --nproc-per-node=1 -m flowmo.evaluate \
    --experiment-name flowmo_lo_prettrain_eval \
    eval.eval_dir=results/flowmo_lo_prettrain \
    eval.continuous=true \
    model.context_dim=18 model.codebook_size_for_entropy=9
```

To reproduce the results of the paper, the commands below will reproduce the performance of FlowMo-Lo and FlowMo-Hi respectively, assuming you have already downloaded the necessary checkpoints (see next section).
```
torchrun --nproc-per-node=1 -m flowmo.evaluate \
    --experiment-name "flowmo_lo_posttrain_eval" \
    eval.eval_dir=results/flowmo_lo_posttrain \
    eval.continuous=false \
    eval.force_ckpt_path='flowmo_lo.pth' \
    model.context_dim=18 model.codebook_size_for_entropy=9

torchrun --nproc-per-node=1 -m flowmo.evaluate \
    --experiment-name "flowmo_hi_posttrain_eval" \
    eval.eval_dir=results/flowmo_hi_posttrain \
    eval.continuous=false \
    eval.force_ckpt_path='flowmo_hi.pth' \
    model.context_dim=56 model.codebook_size_for_entropy=14
```
To speed up eval, you may subsample the data by passing eval.subsample_rate=N to subsample the validation dataset by NX, so that 10 corresponds to 10x subsampling, etc. Note that this will lead to less accurate rFID estimates. Also, the evaluator is distributed, so if you increase --nproc-per-node the evaluation will finish correspondingly faster.


## Get and use the pre-trained models
If you want to evaluate the pre-trained models, you may download them like so:
```
wget https://huggingface.co/ksarge/FlowMo/resolve/main/flowmo_lo.pth
wget https://huggingface.co/ksarge/FlowMo/resolve/main/flowmo_hi.pth
```
The provided notebook `example.ipynb` shows how to use the FlowMo tokenizer to reconstruct images. Within the FlowMo conda environment, you can install a notebook kernel like so:
```
python3 -m ipykernel install --user --name FlowMo
```

## Resource requirements and smaller models
Our main two models (FlowMo-Lo, FlowMo-Hi) were trained on 8 H100 GPUs. However, if your computational resources are limited, you may attain comparable though slightly worse performance by reducing the width and increasing the patch size, by modifying the launch script to pass `model.patch_size=8` and `model.mup_width=4`, or alternatively modifying `configs/base.yaml` with those values.

Still, to reproduce the performance of the models in the paper, you will need to use the larger model configurations.

## Acknowledgement
Our code base was based off https://github.com/TencentARC/SEED-Voken. We also use code from https://github.com/markweberdev/maskbit and https://github.com/black-forest-labs/flux. Thanks for the great contributions.

## Citation
If you find FlowMo useful, please cite us.

```
@misc{sargent2025flowmodemodeseekingdiffusion,
      title={Flow to the Mode: Mode-Seeking Diffusion Autoencoders for State-of-the-Art Image Tokenization}, 
      author={Kyle Sargent and Kyle Hsu and Justin Johnson and Li Fei-Fei and Jiajun Wu},
      year={2025},
      eprint={2503.11056},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.11056}, 
}
```
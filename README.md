# Hyperspectral Image Tokenizer

We propose a hyperspectral image tokenizer pos-trained on NVIDIA Cosmos Image tokenizer. Two versions are train:

- Continuous latents: 1) generator _v.s._ discriminator; 2) diffusion or flow matching.
- Discrete latents: 1) VQ; 2) BSQ.

Base on them, we **will** train two versions of hyperspectral image generators:

there are two ways:

|  Continuous Latents | **Discrete** Latents |
| :---: | :---: |
| Diffusion/Flow Matching/IMM _v.s_ MAR (with diffusive head) | MaskGiT, Autoregressive, or Discrete Diffusion |

Let us see which is better.


## Codes are cooking.

We are cooking the codes, stay tuned.

## TODO
1. ~~add dino v2 discriminator from UniTok~~
2. ~~add SANA triton kernels (~~triton rms~~, triton attention)~~
3. cook diffusion tokenizer (maybe diffusion timestep $t$ cost the model capacity ??)
4. add thera (antialias neural heat field) code into encoder (maybe decoder, both?)
5. ~~add CAMEL optimizer from SANA~~
6. cook discrete tokenizer (**maybe later, it is hard to cook**)
7. IMM and GMFlow code
8. ~~add pansharpening code (in the latent space)~~
9. 1d tokenizer but not use diffusion/flow matching (not now)
10. ~~add Conv-LoRA (MoE) adaptors for Cosmos tokenizer~~
11. add Phase Consistency Model's multi-scale discriminator
12. ~~add nested channel drop~~
13. add maskbit training generator code
14. use maskbit autoencoder (the disc from maskbit, not stable)
15. use bsq-vit autoencoder
16. ~~test vgg-lpips loss in the hyperspectral dataset (if there is a linlayer in the checkpoint?)~~ (not working)
17. cooking the cosmos_f16 tokenizer (continuous latents)
18. add shortcut model code
19. may use the Sana diffusion model as the diffusion generator
20. add UMoE block
21. Mean flow model code compactbility


## Checks
- [x] Check using the pretrained Cosmos tokenizer on RGB images, whether it will cause the reconstructed image to be blurry.
> not blurry, it may due to the fact that I didn't train it well.
- [x] If we can use muon optimizer ?
> YES! The Muon optimizer is much more efficient, higher PSNR value, and more stable.

## Some Refs
1. [GMFlow: Gaussian Mixture Flow Matching Model](https://github.com/Lakonik/GMFlow?tab=readme-ov-file).
2. IMM codes: [official codes](https://github.com/lumalabs/imm), [unofficial codes](https://github.com/rosinality/inductive-moment-matching/blob/main/src/imm/loss.py).
3. [Conv-LoRA](https://github.com/autogluon/autogluon/blob/081e3c6e4134beb84637863624d8b68a5c15bac1/multimodal/src/autogluon/multimodal/models/adaptation_layers.py#L703) (MoE) adaptors may suit Cosmos tokenizer.

## Logs

2025/05/29
> Pretrained on DFC2019 dataset and finetune on other dataset (MMseg: PSNR 36 << PSNR 39 pretrained; OHS 30 ~= 30 pretrained).

- [ ] try to train diffbands tokenizer at pretrained stage and lora finetuned again on different datasets.


2025/05/25
> 1. Only convolution layers of tokenizer will make the norm and the max value of layers to be larger. **Add Attention or LiteMLA will fix this issue**.
> This issue happens not only in cosmos tokenizer but also in LDM AutoencoderKL.

- [x] ~~Try to find if add attention will fix the diffbands input low quality reconstrution.~~ (**does not work.**)
> seems that the attention layer does not fix the large norm issue.

1. use pixel(un)shuffle as downsample and upsample operators (No.2 runnning)
2. just padconv and repeatconv as downsample and upsample operators (No. 6 running).
3. use previous weight with resblock (w/o attention) can finetune good.


## Logs

2025/05/29
> Pretrained on DFC2019 dataset and finetune on other dataset (MMseg: PSNR 36 << PSNR 39 pretrained; OHS 30 ~= 30 pretrained).

- [ ] try to train diffbands tokenizer at pretrained stage and lora finetuned again on different datasets.


2025/05/21

> 1. MoE tokenizer does not work well using triton RMS norm (**may due to the gn?**)
> 2. The LoRA pretrained on MMSeg dataset can not converge well on WV3, GF2 ...datasets, may be due to the small dataset size.

2025/05/02

> 1. MARS optimzier will collapse when starting with a pretrained checkpointed tokenizer. But using AdamW will not collapose.
> 2. The DCAE with only convolution layers underperforms than Cosmos tokenizer (pure conv version).
> 3. DCAE with linear attention seems work? (but convergence is slow).

2025/05/01

seems that channel drop does not work well (slow convergence and low quality).


2025/04/30

*Performances*:

| Model | Dataset| PSNR|
| :---: | :---: | :---: |
| Cosmos_M_f8c16p4 | MMSeg_pix256_c12 | ~39 |
|Cosmos_L_f8c16p4 | MMSeg_pix256_c12 | ~38 (non-convergent) |
| Cosmos_M_BSQ_f8c36p4 | MMSeg_pix256_c12 | ~33 |


2025/04/27

> 1. The MMSeg dataset (with 12 channels) is reconsted blur (~37dB PSNR best, mostly, about 40dB can be visually clean). maybe the latent channel number, 16, is not large enaught. **Try 18 channel (f8z18p4)?**
> 2. The DCF2019 (with 8 channels) on f8z16p4 configuration works fine (~40dB PSNR), if training longer will reach 42dB? In the RGB channels, the NVIDIA orignial tokenizer can reach 46dB (creazy!).
> 3. Try **if enlarging the decoder will make the reconstruction better?**


2025/4/23
> 1. seems BN is vital to the discriminator. ln, gn will make the trianing collapse.
> 2. FSDP2 works fine.

2025/4/10
> 1. hinge loss discriminator works fine for post-training Cosmos tokenizer;
> 2. cooking Flowmo Flux-based diffusive(FM) tokenizer.


## Contributors

- Zihan Cao, UESTC (Core contributor)
- Jieyi Zhu, UESTC (Participant)
- Liang-Jian Deng, UESTC (Supervisor)

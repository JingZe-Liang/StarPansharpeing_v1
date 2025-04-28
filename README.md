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
2. add SANA triton kernels (~~triton rms~~, triton attention)
3. cook diffusion tokenizer (maybe diffusion timestep $t$ cost the model capacity ??)
4. add thera (antialias neural heat field) code into encoder (maybe decoder, both?)
5. ~~add CAMEL optimizer from SANA~~
6. cook discrete tokenizer (**maybe later, it is hard to cook**)
7. IMM and GMFlow code
8. add pansharpening code (in the latent space)
9. 1d tokenizer but not use diffusion/flow matching
10. ~~add Conv-LoRA (MoE) adaptors for Cosmos tokenizer~~
11. add Phase Consistency Model's multi-scale discriminator
12. ~~add nested channel drop~~
13. add maskbit training generator code
14. use maskbit autoencoder
15. use bsq-vit autoencoder
16. test vgg-lpips loss in the hyperspectral dataset (if there is a linlayer in the checkpoint?)


## Checks
- [x] Check using the pretrained Cosmos tokenizer on RGB images, whether it will cause the reconstructed image to be blurry.
> not blurry, it may due to the fact that I didn't train it well.
- [ ] If we can use muon optimizer ?

## Some Refs
1. [GMFlow: Gaussian Mixture Flow Matching Model](https://github.com/Lakonik/GMFlow?tab=readme-ov-file).
2. IMM codes: [official codes](https://github.com/lumalabs/imm), [unofficial codes](https://github.com/rosinality/inductive-moment-matching/blob/main/src/imm/loss.py).
3. [Conv-LoRA](https://github.com/autogluon/autogluon/blob/081e3c6e4134beb84637863624d8b68a5c15bac1/multimodal/src/autogluon/multimodal/models/adaptation_layers.py#L703) (MoE) adaptors may suit Cosmos tokenizer.

## Logs

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
- Yu Zhong, UESTC (Participant)
- Liang-Jian Deng, UESTC (Supervisor)

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
1. add dino v2 discriminator from UniTok
2. add SANA triton kernels (~~triton rms~~, triton attention)
3. cook diffusion tokenizer (maybe diffusion timestep $t$ cost the model capacity ??)
4. add thera (antialias neural heat field) code into encoder (maybe decoder, both?)
5. ~~add CAMEL optimizer from SANA~~
6. cook discrete tokenizer (**maybe later, it is hard to cook**)

## Checks
- [ ] Check using the pretrained Cosmos tokenizer on RGB images, whether it will cause the reconstructed image to be blurry.



## Contributors

- Zihan Cao, UESTC (Core contributor)
- Jieyi Zhu, UESTC (Participant)
- Yu Zhong, UESTC (Participant)
- Liang-Jian Deng, UESTC (Supervisor)



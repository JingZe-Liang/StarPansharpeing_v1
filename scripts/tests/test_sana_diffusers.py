# run `pip install git+https://github.com/huggingface/diffusers` before use Sana in diffusers
# import torch
# from diffusers import SanaPipeline

device = "cuda:1"

# pipe = SanaPipeline.from_pretrained(
#     "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
#     torch_dtype=torch.bfloat16,
# )
# breakpoint()
# pipe.to(device)

# pipe.vae.to(torch.bfloat16)
# pipe.text_encoder.to(torch.bfloat16)

prompt = 'a cyberpunk cat with a neon sign that says "Sana"'
# image = pipe(
#     prompt=prompt,
#     height=1024,
#     width=1024,
#     guidance_scale=4.5,
#     num_inference_steps=20,
#     generator=torch.Generator(device="cuda").manual_seed(42),
# )[0]

# image[0].save("sana.png")

import torch
from transformers import pipeline

max_tokens = 512
default_prompt = f"""
You will act as a remote sensing analyst to describe the content of the image.
Please provide a detailed description of the image in 1 to 3 sentences, including the following aspects:
1. The general content of the image, such as the type of scene (e.g., urban, rural, forest, water body, etc.).
2. The specific objects or features present in the image, such as buildings, roads, vegetation, water bodies, etc.
3. The spatial distribution of the objects or features, such as their arrangement, placements, density, and patterns.
4. Any other relevant information that can help understand the image, such as the time of capture, weather conditions, or any notable geographical feature or events.
Please ensure that your description is clear, concise, and informative, the total words should **not exceed {max_tokens}**.
Do not use the markdown format.
Do not use the bullet points or numbered lists.
Do not include any personal opinions or subjective views.
"""


pipe = pipeline(
    "image-text-to-text",
    model="google/gemma-3n-e4b-it",
    device="cuda:1",
    torch_dtype=torch.bfloat16,
)
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": default_prompt}],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "data/Disaterm3/train_images/train_images/beriut_explosion_post_6_9.png",
            },
            {"type": "text", "text": "What animal is on the candy?"},
        ],
    },
]

output = pipe(text=messages, max_new_tokens=max_tokens)
print(output[0]["generated_text"][-1]["content"])

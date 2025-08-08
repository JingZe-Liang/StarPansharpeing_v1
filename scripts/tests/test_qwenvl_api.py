import asyncio
import base64
import platform

import toml
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

# def get_qwen25vl_max_api():
#     client = OpenAI(
#         # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
#         api_key=toml.load("env.toml")["DASHSCOPE_API_KEY"],
#         base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#     )

#     completion = client.chat.completions.create(
#         model="qwen-plus",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": "你是谁？"},
#         ],
#         extra_body={"enable_thinking": False},
#     )
#     print(completion.choices[0].message.content)


client = AsyncOpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=toml.load("env.toml")["DASHSCOPE_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


async def image_captioning(img_path):
    # Encode image to base64
    base64_image = encode_image_to_base64(img_path)

    response = await client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                    {"type": "text", "text": "这张图片里有什么？"},
                ],
            }
        ],
        model="qwen-vl-plus",
        extra_body={"enable_thinking": False},
    )
    # print(response.choices[0].message.content)
    return response


max_current_tasks = 5
semaphore = asyncio.Semaphore(max_current_tasks)


async def limited_image_captioning(img_path):
    async with semaphore:  # 获取信号量
        return await image_captioning(img_path)


async def main(image_paths: list[str]) -> list[ChatCompletion]:
    # 创建任务列表，但受信号量限制
    tasks = [limited_image_captioning(path) for path in image_paths]
    results = await asyncio.gather(*tasks)
    return results


# 运行示例 (取消注释并提供有效的图片路径)
results = asyncio.run(
    main(
        [
            "/Data4/cao/ZiHanCao/exps/HyperspectralTokenizer/scripts/tests/imgs/cat_memes.jpg",
            "scripts/tests/imgs/rs_mmseg_demo.jpg",
        ]
    )
)

for res in results:
    if res:
        print("---------------------")
        print(res.choices[0].message.content)


# get_qwen25vl_max_api()

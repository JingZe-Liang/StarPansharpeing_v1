"""
智谱AI风格的批量推理实现示例
基于常见的批量API设计模式
"""

import asyncio
import base64
import io
import os
from typing import Sequence, List, Dict, Any
from collections.abc import Sequence

import numpy as np
import toml
from openai import AsyncOpenAI
from PIL import Image


# 假设的批量处理接口
class BatchImageCaptioner:
    def __init__(self, max_batch_size: int = 10, max_concurrent_batches: int = 3):
        self.client = AsyncOpenAI(
            api_key=toml.load("env.toml")["DASHSCOPE_API_KEY"],
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.max_batch_size = max_batch_size
        self.max_concurrent_batches = max_concurrent_batches

    async def process_batch(self, images: List[Dict[str, Any]]) -> List[str]:
        """
        批量处理图像 - 模拟智谱AI的批量API
        """
        # 将批量请求分成多个子批次
        batches = [images[i : i + self.max_batch_size] for i in range(0, len(images), self.max_batch_size)]

        semaphore = asyncio.Semaphore(self.max_concurrent_batches)

        async def process_single_batch(batch):
            async with semaphore:
                # 构造批量请求
                batch_requests = []
                for img_data in batch:
                    base64_image = self._img_to_base64(img_data["image"])
                    batch_requests.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                                },
                                {"type": "text", "text": img_data.get("prompt", "Describe this image.")},
                            ],
                        }
                    )

                # 发送批量请求（这里模拟为多个并发请求）
                tasks = [self._single_image_captioning(request) for request in batch_requests]
                return await asyncio.gather(*tasks)

        # 并发处理所有批次
        batch_results = await asyncio.gather(*[process_single_batch(batch) for batch in batches])

        # 展平结果
        return [result for batch_result in batch_results for result in batch_result]

    async def _single_image_captioning(self, messages: Dict[str, Any]) -> str:
        """单个图像处理"""
        try:
            response = await self.client.chat.completions.create(
                messages=[messages],
                model="qwen-vl-plus",
                extra_body={"enable_thinking": False},
            )
            content = response.choices[0].message.content
            return content if content is not None else ""
        except Exception as e:
            print(f"Error in image_captioning: {e}")
            return ""

    def _img_to_base64(self, img: np.ndarray | str | Image.Image) -> str:
        """转换图像为base64"""
        if isinstance(img, str):
            with open(img, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        elif isinstance(img, Image.Image):
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            return base64.b64encode(img_bytes.getvalue()).decode("utf-8")

        assert img.ndim == 3 and img.dtype == np.uint8, "Image must be a 3D numpy array with dtype uint8."
        img_bytes = io.BytesIO()
        Image.fromarray(img).save(img_bytes, format="PNG")
        return base64.b64encode(img_bytes.getvalue()).decode("utf-8")


# 使用示例
async def main():
    # 初始化批量处理器
    batch_processor = BatchImageCaptioner(max_batch_size=5, max_concurrent_batches=2)

    # 准备批量数据
    batch_data = [
        {"image": "image1.jpg", "prompt": "Describe this rural scene."},
        {"image": "image2.jpg", "prompt": "What urban features are visible?"},
        {"image": "image3.jpg", "prompt": "Analyze the agricultural patterns."},
        # ... 更多图像
    ]

    # 批量处理
    results = await batch_processor.process_batch(batch_data)

    for i, result in enumerate(results):
        print(f"Image {i + 1}: {result}")


if __name__ == "__main__":
    asyncio.run(main())

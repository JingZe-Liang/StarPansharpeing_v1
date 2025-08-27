#!/usr/bin/env python3
"""
预编码Gemma2文本特征脚本
用于将QwenVL生成的caption编码为Gemma2文本特征并保存
"""

import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class Gemma2TextEncoder:
    def __init__(
        self,
        model_name="google/gemma-2-2b",
        max_length=300,
        cache_dir="/HardDisk/ZiHanCao/pretrained",
        device="cuda",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=cache_dir,
        ).to(device)
        self.max_length = max_length
        self.device = device

        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode_text(self, text):
        """编码单个文本"""
        if not text or not text.strip():
            # 空文本处理
            return {
                "caption_feature": np.zeros((self.max_length, 2304), dtype=np.float16),
                "attention_mask": np.zeros(self.max_length, dtype=np.int16),
                "text_length": 0,
            }

        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            # 移到GPU
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 获取文本特征 [batch_size, seq_len, hidden_size]
            outputs = self.model(**inputs)
            text_features = outputs.last_hidden_state

            # 获取attention mask
            attention_mask = inputs["attention_mask"]

            # 只保留有效token的特征 [n, 2304]
            valid_length = int(attention_mask[0].sum().item())
            valid_features = text_features[0, :valid_length].cpu().numpy()
            valid_mask = attention_mask[0, :valid_length].cpu().numpy()

            return {
                "caption_feature": valid_features,  # [n, 2304]
                "attention_mask": valid_mask,  # [n]
                "text_length": valid_length,
            }

    def process_captions(self, captions_json, output_dir):
        """批量处理caption"""
        os.makedirs(output_dir, exist_ok=True)

        with open(captions_json) as f:
            captions = json.load(f)

        results = {}
        for img_id, caption_dict in tqdm(captions.items(), desc="Encoding captions"):
            encoded_dict = {}

            for caption_type, caption_text in caption_dict.items():
                encoded = self.encode_text(caption_text)
                encoded_dict[caption_type] = encoded

                # 单独保存每个caption的npz
                npz_path = os.path.join(output_dir, f"{img_id}_{caption_type}.npz")
                np.savez_compressed(
                    npz_path,
                    caption_feature=encoded["caption_feature"],
                    attention_mask=encoded["attention_mask"],
                    text_length=encoded["text_length"],
                )

            results[img_id] = encoded_dict

        # 保存索引文件
        with open(os.path.join(output_dir, "encoded_captions.json"), "w") as f:
            json.dump(results, f, indent=2)

        print(f"Processed {len(captions)} images")
        return results


def main():
    parser = argparse.ArgumentParser(description="预编码Gemma2文本特征")
    parser.add_argument(
        "--captions_json", required=True, help="QwenVL生成的caption JSON文件路径"
    )
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--max_length", type=int, default=300, help="最大文本长度")
    parser.add_argument(
        "--model_name", default="google/gemma-2-2b", help="Gemma2模型名称"
    )
    parser.add_argument("--device", default="auto", help="设备(auto/cuda/cpu)")

    args = parser.parse_args()

    # 设置设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cpu":
        torch.set_default_device("cpu")

    print(f"Using device: {device}")
    print(f"Model: {args.model_name}")
    print(f"Max length: {args.max_length}")

    encoder = Gemma2TextEncoder(args.model_name, args.max_length)
    encoder.process_captions(args.captions_json, args.output_dir)

    print("Text encoding completed!")


def __test_model():
    """测试Gemma2模型加载和编码"""
    model_name = "google/gemma-2-2b"
    encoder = Gemma2TextEncoder(model_name=model_name, max_length=300)

    # 测试文本
    test_text = "This is a test caption for Gemma2 encoding."
    encoded = encoder.encode_text(test_text)

    print(f"Encoded text feature shape: {encoded['caption_feature'].shape}")
    print(f"Attention mask shape: {encoded['attention_mask'].shape}")
    print(f"Text length: {encoded['text_length']}")


if __name__ == "__main__":
    # main()
    __test_model()

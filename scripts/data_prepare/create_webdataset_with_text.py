#!/usr/bin/env python3
"""
创建包含预编码文本特征的WebDataset
将Gemma2编码的文本特征集成到WebDataset中
"""

import tarfile
import json
import os
import numpy as np
from tqdm import tqdm
import argparse
import io


def create_webdataset_with_encoded_text(
    image_dir,
    encoded_text_dir,
    output_tar,
    caption_types=["qwen"],
    image_extension=".png",
):
    """创建包含预编码文本特征的webdataset"""

    print(f"Creating webdataset: {output_tar}")
    print(f"Image dir: {image_dir}")
    print(f"Text dir: {encoded_text_dir}")
    print(f"Caption types: {caption_types}")

    # 收集所有编码的文本文件
    encoded_info = {}
    for cap_type in caption_types:
        for file in os.listdir(encoded_text_dir):
            if file.endswith(f"_{cap_type}.npz"):
                img_id = file.replace(f"_{cap_type}.npz", "")
                if img_id not in encoded_info:
                    encoded_info[img_id] = {}
                encoded_info[img_id][cap_type] = os.path.join(encoded_text_dir, file)

    print(f"Found {len(encoded_info)} images with encoded text")

    # 创建tar文件
    with tarfile.open(output_tar, "w") as tar:
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        processed = 0
        for img_file in tqdm(image_files, desc="Processing images"):
            img_id = os.path.splitext(img_file)[0]
            img_path = os.path.join(image_dir, img_file)

            if img_id not in encoded_info:
                print(f"Warning: No encoded text for {img_id}")
                continue

            # 添加图片
            tar.add(img_path, arcname=f"{img_id}{image_extension}")

            # 准备元数据
            text_data = {}

            # 处理每个caption类型
            for cap_type in caption_types:
                if cap_type in encoded_info[img_id]:
                    npz_path = encoded_info[img_id][cap_type]
                    data = np.load(npz_path)

                    # 保存为单独文件
                    feature_filename = f"{img_id}_{cap_type}_text_feature.npy"
                    mask_filename = f"{img_id}_{cap_type}_text_mask.npy"

                    # 临时保存，然后添加到tar
                    feature_path = f"/tmp/{feature_filename}"
                    mask_path = f"/tmp/{mask_filename}"

                    np.save(feature_path, data["caption_feature"].astype(np.float16))
                    np.save(mask_path, data["attention_mask"].astype(np.int16))

                    tar.add(feature_path, arcname=feature_filename)
                    tar.add(mask_path, arcname=mask_filename)

                    # 清理
                    os.remove(feature_path)
                    os.remove(mask_path)

                    text_data[cap_type] = {
                        "text_length": int(data["text_length"]),
                        "feature_shape": data["caption_feature"].shape,
                        "mask_shape": data["attention_mask"].shape,
                    }

            # 添加元数据json
            json_str = json.dumps(text_data, indent=2)
            json_bytes = json_str.encode("utf-8")
            json_info = tarfile.TarInfo(name=f"{img_id}.json")
            json_info.size = len(json_bytes)
            tar.addfile(json_info, fileobj=io.BytesIO(json_bytes))

            processed += 1

    print(f"Successfully processed {processed} images")
    print(f"Webdataset created: {output_tar}")


def validate_webdataset(tar_path):
    """验证webdataset内容"""
    print(f"Validating {tar_path}...")

    with tarfile.open(tar_path, "r") as tar:
        members = tar.getnames()

        # 统计文件类型
        images = [m for m in members if m.endswith((".png", ".jpg", ".jpeg"))]
        features = [m for m in members if m.endswith("_text_feature.npy")]
        masks = [m for m in members if m.endswith("_text_mask.npy")]
        metas = [m for m in members if m.endswith(".json")]

        print(f"Images: {len(images)}")
        print(f"Text features: {len(features)}")
        print(f"Text masks: {len(masks)}")
        print(f"Metadata: {len(metas)}")

        # 检查配对
        base_ids = [os.path.splitext(m)[0] for m in images]
        missing = []
        for img_id in base_ids[:5]:  # 检查前5个
            expected_feature = f"{img_id}_qwen_text_feature.npy"
            expected_mask = f"{img_id}_qwen_text_mask.npy"

            if expected_feature not in members or expected_mask not in members:
                missing.append(img_id)

        if missing:
            print(f"Missing pairs: {missing[:10]}...")
        else:
            print("All pairs validated successfully!")


def main():
    parser = argparse.ArgumentParser(description="创建包含文本特征的WebDataset")
    parser.add_argument("--image_dir", required=True, help="图片目录")
    parser.add_argument("--encoded_text_dir", required=True, help="编码文本目录")
    parser.add_argument("--output_tar", required=True, help="输出tar文件路径")
    parser.add_argument("--caption_types", nargs="+", default=["qwen"], help="caption类型列表")
    parser.add_argument("--image_extension", default=".png", help="图片扩展名")
    parser.add_argument("--validate", action="store_true", help="验证创建的tar文件")

    args = parser.parse_args()

    create_webdataset_with_encoded_text(
        args.image_dir,
        args.encoded_text_dir,
        args.output_tar,
        args.caption_types,
        args.image_extension,
    )

    if args.validate:
        validate_webdataset(args.output_tar)


if __name__ == "__main__":
    main()

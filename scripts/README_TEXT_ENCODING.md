# 文本特征预编码使用指南

本指南介绍如何使用Gemma2模型预编码文本特征，并集成到Sana WebDataset中。

## 完整流程

### 1. 生成Caption（使用QwenVL）
```bash
# 使用现有的QwenVL生成caption
python generate_captions.py \
  --images_dir /path/to/images \
  --output /path/to/captions_qwen.json
```

### 2. 预编码Gemma2文本特征
```bash
# 使用GPU（推荐）
python scripts/prepare_gemma2_text_features.py \
  --captions_json /path/to/captions_qwen.json \
  --output_dir /path/to/encoded_captions \
  --model_name google/gemma-2-2b \
  --max_length 300

# 使用CPU（显存不足时）
python scripts/prepare_gemma2_text_features.py \
  --captions_json /path/to/captions_qwen.json \
  --output_dir /path/to/encoded_captions \
  --model_name google/gemma-2-2b \
  --device cpu
```

### 3. 创建WebDataset
```bash
# 创建包含文本特征的tar文件
python scripts/create_webdataset_with_text.py \
  --image_dir /path/to/images \
  --encoded_text_dir /path/to/encoded_captions \
  --output_tar /path/to/dataset_with_text.tar \
  --caption_types qwen \
  --validate
```

## 文件结构

### 输入文件
```
images/
├── 00000001.png
├── 00000002.png
└── ...

captions_qwen.json
{
    "00000001": {"qwen": "a modern city with buildings"},
    "00000002": {"qwen": "forest area with green trees"}
}
```

### 输出文件
```
encoded_captions/
├── 00000001_qwen.npz
├── 00000002_qwen.npz
├── encoded_captions.json
└── ...

dataset_with_text.tar
├── 00000001.png
├── 00000001_qwen_text_feature.npy
├── 00000001_qwen_text_mask.npy
├── 00000001.json
└── ...
```

## Sana配置

### YAML配置示例
```yaml
data:
  type: "SanaWebDataset"
  data_dir: "/path/to/dataset_with_text.tar"
  load_text_feat: true
  load_vae_feat: false
  max_length: 300
  external_caption_suffixes: ['_qwen']
  external_clipscore_suffixes: []  # 不使用CLIP分数过滤
```

### 数据格式
- **文本特征**: `[300, 2304]` (float16)
- **attention_mask**: `[300]` (int16)
- **text_length**: 实际文本长度

## 内存优化

### 小显存机器
- 使用 `--device cpu` 参数
- 选择较小的模型: `--model_name google/gemma-2-2b-it`
- 减小 `--max_length 200`

### 批量处理
```bash
# 分批处理大量图片
python scripts/prepare_gemma2_text_features.py \
  --captions_json captions_batch1.json \
  --output_dir encoded_batch1/ \
  --device cuda:0
```

## 验证

### 检查输出
```bash
# 验证WebDataset
python scripts/create_webdataset_with_text.py \
  ... \
  --validate

# 检查单个文件
python -c "
import numpy as np
data = np.load('encoded_captions/00000001_qwen.npz')
print('Feature shape:', data['caption_feature'].shape)
print('Mask shape:', data['attention_mask'].shape)
print('Text length:', data['text_length'])
"
```

## 常见问题

1. **显存不足**: 使用CPU或减小batch_size
2. **文件缺失**: 确保caption和image ID匹配
3. **格式错误**: 检查图片格式和扩展名
4. **路径问题**: 使用绝对路径避免路径解析错误

## 支持的模型

- `google/gemma-2-2b` (默认)
- `google/gemma-2-2b-it`
- `google/gemma-2-9b` (需要更多显存)
- 其他兼容的transformers模型

# Condition Preparation for Hyperspectral Images

This script generates various condition maps and captions from hyperspectral images stored in WebDataset format.

## Features

- **Multiple Condition Types**: Generate HED edges, segmentation maps, sketches, MLSD line detection, and captions
- **Flexible Output Formats**: Save conditions as PNG, JPG, or SafeTensors
- **Caption Generation**: Generate captions using Qwen2.5-VL model
- **Batch Processing**: Efficiently process large datasets using WebDataset
- **Progress Tracking**: Real-time progress monitoring with tqdm
- **Configuration Support**: Use Hydra configs or command-line arguments

## Supported Conditions

1. **HED**: Holistically-Nested Edge Detection
2. **Segmentation**: SAM-based semantic segmentation
3. **Sketch**: Sketch/edge detection
4. **MLSD**: Mobile Line Segment Detection
5. **Caption**: Text descriptions using vision-language models
6. **Content**: Original RGB content

## Usage

### Using Hydra Configuration (Recommended)

```bash
# Use default configuration
python scripts/generative_condition_prepare.py --hydra

# Use specific configuration
python scripts/generative_condition_prepare.py --hydra config=hed_only

# Override specific parameters
python scripts/generative_condition_prepare.py --hydra \
    data.wds_paths=["your/data/path/*.tar"] \
    output.output_dir="your/output/dir" \
    processor.conditions=["hed","caption"]
```

### Using Command Line Arguments

```bash
python scripts/generative_condition_prepare.py \
    --wds_paths "data_local/MMSeg_YREB/hyper_images/*.tar" \
    --output_dir "data_local/MMSeg_YREB/conditions" \
    --tar_name "conditions_generated" \
    --tar_rel_path "conditions/conditions_generated.tar" \
    --conditions "all" \
    --rgb_channels 0 1 2 \
    --device "cuda" \
    --condition_save_format "png" \
    --caption_save_format "txt"
```

## Configuration Files

Configuration files are located in `scripts/configs/condition_preparation/`:

- `default_config.yaml`: Base configuration with all settings
- `hed_only.yaml`: Generate HED edge maps only
- `caption_only.yaml`: Generate captions only

### Configuration Structure

```yaml
data:
  _target_: src.data.hyperspectral_loader.get_hyperspectral_dataloaders
  wds_paths: ["path/to/your/*.tar"]
  batch_size: 1
  num_workers: 4
  shuffle: false
  to_neg_1_1: true

output:
  output_dir: "output/directory"
  tar_name: "output_tar_name"
  tar_rel_path: "relative/path/to/output.tar"

processor:
  conditions: "all"  # or ["hed", "segmentation", "sketch", "mlsd", "caption"]
  rgb_channels: [0, 1, 2]  # RGB channels from hyperspectral data
  device: "cuda"
  to_pil: true
  save_original_rgb: true
  condition_save_format: "png"  # png, jpg, or safetensors
  caption_save_format: "txt"    # txt, json, or safetensors
```

## Output Format

The script generates TAR files containing:

- `rgb.png`: Original RGB image (if `save_original_rgb=True`)
- `{condition_name}.{format}`: Condition maps (e.g., `hed.png`, `segmentation.png`)
- `caption.{format}`: Text captions (e.g., `caption.txt`)

## Memory and Performance Tips

1. **GPU Memory**: Use smaller batch sizes if running out of GPU memory
2. **Disk I/O**: Consider using SSD storage for faster processing
3. **Device Selection**: Specify GPU device using `processor.device: "cuda:0"`
4. **Condition Selection**: Generate only needed conditions to save time and space

## Dependencies

Ensure you have the following dependencies installed:

- torch
- webdataset
- PIL (Pillow)
- tqdm
- hydra-core
- opencv-python
- numpy
- safetensors

Plus the specific condition detectors:
- HED detector
- SAM detector
- Sketch detector
- MLSD detector
- Qwen2.5-VL model for captions

## Example Workflow

1. Prepare your hyperspectral data in WebDataset format
2. Create or modify a configuration file
3. Run the condition preparation script
4. Use the generated conditions for training generative models

```bash
# Generate all conditions
python scripts/generative_condition_prepare.py --hydra

# Generate only HED edges for faster processing
python scripts/generative_condition_prepare.py --hydra config=hed_only

# Generate captions for all images
python scripts/generative_condition_prepare.py --hydra config=caption_only
```

## Troubleshooting

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Missing Model Files**: Ensure all detector models are downloaded
3. **WebDataset Format Issues**: Check that input TAR files contain 'img' field
4. **RGB Channel Issues**: Adjust `rgb_channels` parameter for your data format

# Metrics Module for Stereo Matching

本模块提供立体匹配和语义分割的评估指标，兼容SemStereo实现。

## 📊 模块对比

### 1. **basic.py** (您的原始实现)
- ✅ 使用TorchMetrics框架
- ✅ 支持DDP训练
- ✅ 实现了EPE和D1指标
- ❌ **缺少相对阈值检查** (D1需要同时检查绝对误差>3px AND 相对误差>5%)
- ❌ 缺少Threshold metrics
- ❌ 缺少语义分割metrics

### 2. **SemStereo/utils/metrics.py** (原始SemStereo实现)
- ✅ 完整的D1实现 (3px & 5%双重阈值)
- ✅ EPE, Thres1px, Thres2px
- ✅ 完整的语义分割metrics (PA, MPA, mIoU, per-class IoU)
- ❌ 不使用TorchMetrics
- ❌ 不支持DDP自动聚合
- ❌ 代码风格较旧

### 3. **unified.py** (新的统一实现) ⭐
- ✅ **完全兼容SemStereo的所有metrics**
- ✅ 使用TorchMetrics框架,支持DDP
- ✅ 统一接口,同时计算立体匹配和分割指标
- ✅ 包含functional API (无状态) 和 class API (有状态)
- ✅ 完整文档和类型提示

## 📝 关键差异说明

### D1 Metric的正确实现

**SemStereo的D1 (正确):**
```python
E = torch.abs(D_gt - D_est)
err_mask = (E > 3) & (E / D_gt.abs() > 0.05)  # 双重条件
return torch.mean(err_mask.float())
```

**您的basic.py (不完整):**
```python
bad_pixels_map = error_map > threshold  # 只有绝对阈值
return d1_score
```

**unified.py (正确+完整):**
```python
bad_pixels = (error > 3.0) & (error / target_valid.abs() > 0.05)
return bad_pixels.float().mean()
```

## 🚀 使用方法

### 方法1: 使用Unified Class (推荐)

```python
from src.stage2.stereo_matching.metrics.unified import create_semstereo_metrics

# 创建metrics (US3D数据集设置)
metrics = create_semstereo_metrics(
    num_classes=6,
    min_disp=-64,
    max_disp=64,
)

# 在训练loop中
for batch in dataloader:
    outputs = model(batch['left'], batch['right'])

    # 更新metrics
    metrics.update(
        disp_pred=outputs['d_final'],
        disp_target=batch['disparity'],
        seg_pred=outputs['P_l'],
        seg_target=batch['label'],
    )

# Epoch结束时计算
results = metrics.compute()
print(f"EPE: {results['stereo']['EPE']:.3f}")
print(f"D1: {results['stereo']['D1']:.3f}")
print(f"mIoU: {results['seg']['mIoU']:.3f}")

# 重置state
metrics.reset()
```

### 方法2: 使用Functional API (无状态)

```python
from src.stage2.stereo_matching.metrics.unified import (
    compute_epe,
    compute_d1,
    compute_threshold_error,
)

# 计算单个batch的metrics
epe = compute_epe(
    pred=outputs['d_final'],
    target=batch['disparity'],
    min_disp=-64,
    max_disp=64,
)

d1 = compute_d1(
    pred=outputs['d_final'],
    target=batch['disparity'],
    min_disp=-64,
    max_disp=64,
)
```

### 方法3: 分别使用立体和分割metrics

```python
from src.stage2.stereo_matching.metrics.unified import (
    StereoMatchingMetrics,
    SemanticSegmentationMetrics,
)

# 只用立体匹配metrics
stereo_metrics = StereoMatchingMetrics(min_disp=-64, max_disp=64)
stereo_metrics.update(disp_pred, disp_target)
stereo_results = stereo_metrics.compute()

# 只用分割metrics
seg_metrics = SemanticSegmentationMetrics(num_classes=6)
seg_metrics.update(seg_pred, seg_target)
seg_results = seg_metrics.compute()
```

## 📊 输出Metrics说明

### 立体匹配 Metrics

| Metric | 全称 | 说明 | 范围 |
|--------|------|------|------|
| **EPE** | End-Point Error | 平均绝对误差 (像素) | [0, ∞) |
| **D1** | D1 Error | 误差>3px且>5%的像素比例 | [0, 1] |
| **Thres1px** | 1-pixel Threshold | 误差>1px的像素比例 | [0, 1] |
| **Thres2px** | 2-pixel Threshold | 误差>2px的像素比例 | [0, 1] |

### 语义分割 Metrics

| Metric | 全称 | 说明 | 范围 |
|--------|------|------|------|
| **PA** | Pixel Accuracy | 像素准确率 | [0, 1] |
| **MPA** | Mean Pixel Accuracy | 平均类别准确率 | [0, 1] |
| **mIoU** | Mean Intersection over Union | 平均IoU | [0, 1] |
| **F1** | F1 Score | F1分数 (=Dice) | [0, 1] |
| **Dice** | Dice Coefficient | Dice系数 | [0, 1] |
| **IoU_per_class** | Per-class IoU | 每个类别的IoU | Tensor[C] |
| **Acc_per_class** | Per-class Accuracy | 每个类别的准确率 | Tensor[C] |

## 🔄 迁移指南

### 从SemStereo迁移

```python
# 旧代码 (SemStereo)
from utils.metrics import EPE_metric, D1_metric, Thres_metric

epe = EPE_metric(disp_est, disp_gt, mask)
d1 = D1_metric(disp_est, disp_gt, mask)
thres1 = Thres_metric(disp_est, disp_gt, mask, 1.0)

# 新代码 (unified)
from src.stage2.stereo_matching.metrics.unified import compute_epe, compute_d1

# 自动生成mask
epe = compute_epe(disp_est, disp_gt, min_disp=-64, max_disp=64)
d1 = compute_d1(disp_est, disp_gt, min_disp=-64, max_disp=64)

# 或手动提供mask
epe = compute_epe(disp_est, disp_gt, mask=mask)
```

### 从basic.py迁移

```python
# 旧代码
from src.stage2.stereo_matching.metrics.basic import EndPointError, D1Error

epe_metric = EndPointError(min_disp=-64, max_disp=64)
d1_metric = D1Error(min_disp=-64, max_disp=64)

# 新代码 (更简单)
from src.stage2.stereo_matching.metrics.unified import UnifiedSemStereoMetrics

metrics = UnifiedSemStereoMetrics(
    num_classes=6,
    min_disp=-64,
    max_disp=64,
    compute_seg=True,  # 同时计算分割metrics
    compute_stereo=True,
)
```

## ⚠️ 注意事项

1. **D1 Metric的定义**:
   - 标准D1需要 **同时满足** 两个条件:
     - 绝对误差 > 3 pixels
     - 相对误差 > 5%
   - `basic.py`的实现只检查了绝对阈值,需要更新

2. **Mask生成**:
   - SemStereo: `mask = (gt < max_disp) & (gt >= min_disp)`
   - unified.py: 已自动处理,也支持手动传入mask

3. **DDP支持**:
   - `unified.py`的class-based metrics自动支持DDP
   - Functional API不支持DDP,需要手动聚合

4. **性能**:
   - Functional API: 更快,适合inference
   - Class-based API: 自动累积,适合training

## 📚 参考文献

- [SemStereo Paper](https://arxiv.org/abs/2412.12685)
- [KITTI Stereo Benchmark](http://www.cvlibs.net/datasets/kitti/eval_stereo.php)
- [TorchMetrics Documentation](https://torchmetrics.readthedocs.io/)

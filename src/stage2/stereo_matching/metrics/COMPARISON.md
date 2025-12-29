"""
Metrics Comparison: Your Implementation vs SemStereo

This document provides detailed comparison between:
1. Your basic.py implementation
2. SemStereo's metrics.py
3. The new unified.py implementation
"""

## ===================================================================
## PART 1: D1 Metric Comparison (最重要的差异!)
## ===================================================================

### SemStereo's D1_metric (CORRECT ✅)
```python
# From: SemStereo/utils/metrics.py, Line 39-43
def D1_metric(D_est, D_gt, mask):
    D_est, D_gt = D_est[mask], D_gt[mask]
    E = torch.abs(D_gt - D_est)
    # KEY: Both conditions must be true!
    err_mask = (E > 3) & (E / D_gt.abs() > 0.05)
    return torch.mean(err_mask.float())
```

**定义**: D1错误率 = 满足以下**两个条件**的像素比例:
1. 绝对误差 > 3 pixels **AND**
2. 相对误差 > 5% (error / |gt_disparity|)

这是KITTI和立体匹配领域的**标准评估指标**。

---

### Your basic.py's compute_d1 (INCOMPLETE ❌)
```python
# From: metrics/basic.py, Line 61-117
def compute_d1(est, gt, min_disp, max_disp, threshold=3.0):
    # ...
    error_map = torch.abs(est - gt)
    bad_pixels_map = error_map > threshold  # ❌ Only absolute threshold!
    # Missing: error / gt.abs() > 0.05
    # ...
```

**问题**:
- ❌ 只检查了绝对阈值 (error > 3.0)
- ❌ **缺少相对阈值检查** (error / |gt| > 0.05)
- 这会导致D1值不准确,无法与其他论文公平比较

---

### unified.py's compute_d1 (CORRECT ✅)
```python
# From: metrics/unified.py, Line 79-113
def compute_d1(pred, target, mask=None,
               abs_threshold=3.0, rel_threshold=0.05):
    # ...
    error = torch.abs(target_valid - pred_valid)

    # ✅ Both conditions!
    bad_pixels = (error > abs_threshold) & \
                 (error / target_valid.abs() > rel_threshold)

    return bad_pixels.float().mean()
```

**修复**: 完全实现了标准D1定义,兼容SemStereo。

## ===================================================================
## PART 2: EPE Metric Comparison
## ===================================================================

### SemStereo's EPE_metric ✅
```python
# From: SemStereo/utils/metrics.py, Line 57-59
def EPE_metric(D_est, D_gt, mask):
    D_est, D_gt = D_est[mask], D_gt[mask]
    return F.l1_loss(D_est, D_gt, size_average=True)
```

### Your basic.py's compute_epe ✅
```python
# From: metrics/basic.py, Line 6-58
def compute_epe(est, gt, min_disp, max_disp):
    # ... generate mask ...
    error_map = torch.abs(est - gt)
    masked_error = error_map[mask]
    average_epe = total_error / num_valid_pixels
    return total_error, num_valid_pixels, average_epe
```

### unified.py's compute_epe ✅
```python
# From: metrics/unified.py, Line 28-60
def compute_epe(pred, target, mask=None,
                min_disp=None, max_disp=None):
    # ... auto mask generation or use provided mask ...
    error = torch.abs(pred - target)
    return masked_error.mean()
```

**结论**: 所有实现在EPE上都是**正确的**,只是接口略有不同。

## ===================================================================
## PART 3: Threshold Metrics Comparison
## ===================================================================

### SemStereo's Thres_metric ✅
```python
# From: SemStereo/utils/metrics.py, Line 47-52
def Thres_metric(D_est, D_gt, mask, thres):
    D_est, D_gt = D_est[mask], D_gt[mask]
    E = torch.abs(D_gt - D_est)
    err_mask = E > thres
    return torch.mean(err_mask.float())
```

### Your basic.py ❌
**不存在** - 需要添加

### unified.py's compute_threshold_error ✅
```python
# From: metrics/unified.py, Line 116-151
def compute_threshold_error(pred, target, threshold,
                           mask=None, ...):
    error = torch.abs(target_valid - pred_valid)
    bad_pixels = error > threshold
    return bad_pixels.float().mean()
```

**新增**: unified.py添加了Thres1px和Thres2px支持。

## ===================================================================
## PART 4: Semantic Segmentation Metrics
## ===================================================================

### SemStereo's SegmentationMetric ✅
```python
# From: SemStereo/utils/metrics.py, Line 91-213
class SegmentationMetric(object):
    def __init__(self, numClass):
        self.confusionMatrix = np.zeros((self.numClass,)*2)

    def pixelAccuracy(self): ...
    def meanPixelAccuracy(self): ...
    def IoU(self): ...
    def meanIntersectionOverUnion(self): ...
```

**功能**:
- ✅ Pixel Accuracy (PA)
- ✅ Mean Pixel Accuracy (MPA)
- ✅ mean IoU (mIoU)
- ✅ Per-class IoU
- ✅ Per-class Accuracy
- ❌ 不支持DDP
- ❌ 使用numpy,不是纯PyTorch

### Your basic.py ❌
**不存在** - 完全缺失语义分割metrics

### unified.py's SemanticSegmentationMetrics ✅
```python
# From: metrics/unified.py, Line 357-461
class SemanticSegmentationMetrics(Metric):
    def __init__(self, num_classes, ignore_index=None):
        self.add_state("confmat", ...)  # ✅ DDP support

    def update(self, preds, target): ...

    def compute(self):
        # Returns: PA, MPA, mIoU, F1, Dice, IoU_per_class, Acc_per_class
```

**新增**:
- ✅ 完整的confusion matrix实现
- ✅ **支持DDP** (使用torchmetrics)
- ✅ 纯PyTorch,GPU加速
- ✅ **额外添加了F1和Dice**
- ✅ 更好的代码风格和类型提示

## ===================================================================
## PART 5: Feature Comparison Matrix
## ===================================================================

| Feature | basic.py | SemStereo | unified.py |
|---------|----------|-----------|------------|
| **Stereo Metrics** | | | |
| EPE (正确实现) | ✅ | ✅ | ✅ |
| D1 (完整:3px+5%) | ❌ | ✅ | ✅ |
| D1 (仅3px) | ✅ | - | - |
| Thres1px | ❌ | ✅ | ✅ |
| Thres2px | ❌ | ✅ | ✅ |
| **Segmentation Metrics** | | | |
| Pixel Accuracy | ❌ | ✅ | ✅ |
| Mean Pixel Acc | ❌ | ✅ | ✅ |
| mIoU | ❌ | ✅ | ✅ |
| Per-class IoU | ❌ | ✅ | ✅ |
| F1 Score | ❌ | ❌ | ✅ |
| Dice Coefficient | ❌ | ❌ | ✅ |
| **Technical Features** | | | |
| DDP Support | ✅ | ❌ | ✅ |
| TorchMetrics | ✅ | ❌ | ✅ |
| Type Hints | ✅ | ❌ | ✅ |
| Functional API | ✅ | ✅ | ✅ |
| Class API | ✅ | ✅ | ✅ |
| Unified Interface | ❌ | ❌ | ✅ |
| Pure PyTorch | ✅ | ❌(numpy) | ✅ |
| Documentation | ⚠️ | ⚠️ | ✅✅ |

## ===================================================================
## PART 6: Recommendations
## ===================================================================

### 立即修复 (Critical)

1. **修复basic.py的D1实现**:
   ```python
   # 当前 (错误)
   bad_pixels_map = error_map > threshold

   # 应该改为
   bad_pixels = (error > 3.0) & (error / gt.abs() > 0.05)
   ```

2. **添加Threshold metrics**:
   - 实现Thres1px和Thres2px
   - 用于更细粒度的误差分析

3. **添加语义分割metrics**:
   - 至少要有PA, MPA, mIoU
   - 用于评估multi-task模型

### 推荐使用 (Recommended)

**直接使用 `unified.py`** 作为您的主要metrics模块:

✅ **优势**:
- 完全兼容SemStereo
- 一个class同时计算所有metrics
- 支持DDP,开箱即用
- 代码风格现代,易维护
- 完整文档和测试

✅ **简单集成**:
```python
# 替换所有metrics导入
from src.stage2.stereo_matching.metrics.unified import (
    create_semstereo_metrics
)

# 一行代码创建所有metrics
metrics = create_semstereo_metrics(
    num_classes=6,
    min_disp=-64,
    max_disp=64
)
```

### 可选优化

1. **保留basic.py** (如果有特殊需求):
   - 但需要修复D1实现
   - 添加缺失的metrics

2. **性能优化**:
   - unified.py已经使用高效的confusion matrix累积
   - 如需更快,可以考虑CUDA kernel实现

3. **扩展metrics**:
   - 可以添加论文中的其他指标
   - 例如:分类别的D1错误率

## ===================================================================
## PART 7: Migration Guide
## ===================================================================

### Step 1: 安装unified.py
已完成 ✅

### Step 2: 更新训练代码

```python
# 旧代码
from src.stage2.stereo_matching.metrics.basic import (
    EndPointError, D1Error
)
epe_metric = EndPointError(min_disp=-64, max_disp=64)
d1_metric = D1Error(min_disp=-64, max_disp=64)

# 新代码
from src.stage2.stereo_matching.metrics.unified import (
    create_semstereo_metrics
)
metrics = create_semstereo_metrics()

# 使用
metrics.update(
    disp_pred=outputs['d_final'],
    disp_target=batch['disparity'],
    seg_pred=outputs['P_l'],
    seg_target=batch['label'],
)
results = metrics.compute()
```

### Step 3: 更新logger

```python
# 记录所有metrics
for metric_name, value in results['stereo'].items():
    logger.log({f'val/stereo/{metric_name}': value})

for metric_name, value in results['seg'].items():
    if isinstance(value, torch.Tensor) and value.dim() == 0:
        logger.log({f'val/seg/{metric_name}': value})
```

### Step 4: 验证结果

运行测试确保新metrics与SemStereo一致:
```bash
python src/stage2/stereo_matching/metrics/unified.py
```

## ===================================================================
## Summary
## ===================================================================

**主要发现**:
1. ❌ basic.py的D1实现**不完整**,缺少相对阈值检查
2. ❌ basic.py**缺少**Threshold和语义分割metrics
3. ✅ unified.py **完全修复**所有问题,并添加新功能

**建议**:
**立即采用 unified.py,逐步废弃 basic.py**

这样可以:
- 确保与SemStereo论文结果可比较
- 获得完整的multi-task评估能力
- 利用DDP加速训练
- 使用更好的代码组织和文档

# LeJEPA 与 Rectified LpJEPA：统一视角、原理与直接调用

这份文档的目标是把两个“minimal 脚本式实现”拆成你可以在任意训练循环里直接调用的模块，并解释它们背后的直觉与关键超参。

对应代码位置：
- `src/stage1/self_supervised/lejepa_aug.py`：已有 LeJEPA 的增强、`SIGReg`、`lejepa_loss(...)` 等函数式实现
- `src/stage1/self_supervised/lejepa_objectives.py`：新增的可调用类封装（`LeJEPALoss`、`RectifiedLpJEPALoss` 等）
- `src/stage1/self_supervised/rectified_lpjepa/`：上游 vendored 代码（包含更完整的 Lightning/Solo-learn 训练逻辑）

---

## 1. 共同结构：Invariance + 一个“防塌缩/塑形”的正则项

两者都可以看成在学一个表示 `z`，满足：
1) **同一张图（或同一 patch/token）的不同视角**，在表示空间里应该一致（invariance）；
2) **表示本身的分布**应该满足某种“结构/先验”，避免塌缩并引导更有用的几何或稀疏性质。

形式上都长得像：

`L = w_inv * L_inv(z1, z2) + w_reg * L_reg(z)`

区别主要在 `L_reg` 是什么、以及它对表示的偏好是什么。

---

## 2. LeJEPA：SIGReg（Sketch/Isotropic Gaussian Regularization）

### 2.1 直觉

LeJEPA 的 minimal 版本里，`SIGReg` 做的是一种“用随机投影 + 特征函数（cos/sin）去约束投影分布形状”的正则项。
它的目标不是强行把表示变成某个具体分布，而更像是**让多视角的表示在统计意义上更“各向同性/稳定”**，配合 invariance 项一起工作，降低塌缩风险并提升表示质量。

### 2.2 在本仓库里怎么直接用

你可以继续用函数式 API：`lejepa_loss(...)`。
为了更方便塞进你自己的训练 loop，我把它包成了 `torch.nn.Module`：`LeJEPALoss`。

示例（伪代码，核心是接口形状）：

```python
import torch
from src.stage1.self_supervised import LeJEPALoss

loss_fn = LeJEPALoss(lam=0.02, infonce_weight=None)

# global_emb: [Vg, B, D]；local_emb 可选: [Vl, B, D]
loss, metrics = loss_fn(global_emb, local_emb)
```

其中 `metrics` 会返回一个 dict，包含 `inv_loss`、`sigreg_loss`、（可选）`info_nce_loss` 等。

---

## 3. Rectified LpJEPA：RDMReg（用 SWD 匹配到“稀疏先验分布”）

### 3.1 直觉

Rectified LpJEPA 的正则项（RDMReg）做的是**分布匹配**：
把特征 `z` 的分布拉向一个“目标先验”——（可选）经过 ReLU 的 generalized Gaussian（典型是 p=1 的 Laplace）。

关键点有两个：

1) **末尾 ReLU 的 projector**：强制非负，天然会制造大量 0 激活，从而产生 L0 稀疏性；
2) **SWD（Sliced Wasserstein Distance）**：把高维分布投影到很多 1D 方向，在 1D 上通过排序计算 Wasserstein 距离，再对方向求平均。计算上相对友好，而且在经验上能作为“分布塑形”的强正则。

### 3.2 稀疏由什么控制？

在该方法里，“稀疏程度”主要由目标分布的参数控制：
- `mean_shift_value (mu)`：把目标分布整体向左/向右挪。`mu` 越小（更负），越多采样会被 ReLU 截断为 0，稀疏更强；
- `lp_norm_parameter (p)`：p=1 是 Laplace（更稀疏倾向），p=2 是 Gaussian；
- `chosen_sigma` 或 `sigma_mode`：控制尺度。`sigma_GN` 是让**rectify 之前**的 GN_p 有 unit variance；`sigma_RGN` 是解方程让 **ReLU(GN_p)** 的 variance 为 1（需要数值求解）。

### 3.3 在本仓库里怎么直接用

我把核心 objective 抽成了可调用模块 `RectifiedLpJEPALoss`，你只需要提供两个 view 的 projector 输出 `z1/z2`（形状 `[B, D]`）。

```python
import torch
from src.stage1.self_supervised import RectifiedLpJEPALoss

loss_fn = RectifiedLpJEPALoss(
    invariance_loss_weight=25.0,
    rdm_reg_loss_weight=125.0,
    target_distribution="rectified_lp_distribution",
    projection_vectors_type="random",
    num_projections=256,
    mean_shift_value=0.0,
    lp_norm_parameter=1.0,
    sigma_mode="sigma_GN",
)

loss, metrics = loss_fn(z1, z2)  # z1/z2: [B, D]
print(metrics.invariance_loss, metrics.rdmreg_loss)
```

你也可以把 `projection_vectors` 预先算好传进去（比如固定随机方向或复用某次 SVD 的方向）：

```python
loss, metrics = loss_fn(z1, z2, projection_vectors=proj_vecs)
```

### 3.4 Projector（末尾 ReLU）的复用

Rectified LpJEPA 通常使用“末尾 ReLU”的 3 层 MLP projector。我在 `lejepa_objectives.py` 里提供了：

```python
from src.stage1.self_supervised import create_rectified_lpjepa_projector

proj = create_rectified_lpjepa_projector(
    input_dim=backbone_dim,
    hidden_dim=2048,
    output_dim=256,
    rectified=True,
)
```

---

## 4. 你应该怎么选

如果你更关心“表示稳定 + 多视角一致”，并且希望正则项更像一个“温和的形状约束”，优先用 LeJEPA（SIGReg）。

如果你明确想要“稀疏且非负”的表示（更偏向字典学习/稀疏编码味道），并且愿意承担更强的分布匹配正则，优先用 Rectified LpJEPA（RDMReg）。

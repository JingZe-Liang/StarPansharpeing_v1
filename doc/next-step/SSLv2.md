# SSL RS 下一步

## 参考文章

**Revisiting Multi-Task Visual Representation Learning**
https://www.alphaxiv.org/overview/2601.13886?chatId=019bfbaf-514c-7c42-97b4-ccf3c3d98fe3

> Multi-task SSL pretraining, language grounding + ssl + dense

**OpenVision3**
Language (next token prediction, CLIP contrastive loss) + reconstruction loss (VAE recon and pixel recon).

---

## 1. Grounding 损失

RAM++ 提取实体名称，OWLv2-based 定位，最后利用 CLOC 的 prompt 模块，提取 box 特征与 patch 信息融合，最后做 siglip 的 loss 进行接地损失的计算。Query-token + 几层 cross attention 外加 box 的位置嵌入，这样直接拿到 pool token，与实体名称文字进行对齐计算（siglip loss）。

**另一种思路**：实体名称拿到之后，可以不用定位？直接用 SAM3 的开放词汇分割能力直接拿到 mask，拿到 mask 之后怎么对齐呢？可以直接把 mask 下采样到 grid size 的大小，作为 attention bias，这样尝试让 query 获取 mask 里面的信息，之后再跟实体文本进行对齐。

---

## 2. SAM Mask Loss (用于执行feature的形状一致性)

接 1 的想法，利用 SAM mask，可以进行 dense 信息在 SSL 训练的注入：可以采用以下两种方式之一来实现 $L_{Mask}$，作为"密集结构监督"：

### 方案 A：对比掩码损失（Contrastive Mask Loss）—— 推荐

这是最符合 MTV 对比学习范式的做法。

**做法**：对于 SAM 生成的每一个 Mask 区域 $M_k$：

1. 采样 Mask 内部的像素点作为正样本对（Positive Pairs）
2. 采样 Mask 外部（或不同 Mask）的像素点作为负样本对（Negative Pairs）
3. 使用 InfoNCE Loss 或类似 SigLIP 的 loss，强迫模型输出的特征图（Feature Map）在同一个 Mask 内部高度一致，在边界处发生突变

**优势**：这不依赖具体的语义类别（Class-Agnostic），纯粹学习"物体是什么形状"的结构信息，完美替代 Depth 的几何功能。

**具体做法**：既然没有类别标签（"Car", "Tree"），那我们就不要预测类别概率（Logits）。我们预测像素嵌入向量（Pixel Embeddings）。

- 属于同一个 Mask 的像素，其输出向量在空间中应该距离很近
- 属于不同 Mask 的像素，其输出向量应该互相排斥

**Head 结构设计**：完全复用 MTV 中的 DPT 融合模块，只修改最后一层卷积

- **输入**：多层 Backbone 特征融合后的特征图（比如 $H/4 \times W/4 \times C$）
- **结构**：Conv3x3 -> ReLU -> Conv1x1
- **输出**：输出维度不是 1（深度），也不是 $K$（类别数），而是一个低维嵌入空间 $E$（建议 $E=16$ 或 $32$）。输出 Tensor 形状为 $H \times W \times E$
- **归一化**：对最后一个维度做 L2 Normalization，把向量投射到超球面上

**Loss 设计（Discriminative Loss / InfoNCE）**：从 SAM 的 Mask 中随机采样

- Anchor：某个 Mask 内的一个像素
- Positive：同一个 Mask 内的另一个像素
- Negative：其他 Mask 的像素

让 Anchor 和 Positive 的余弦相似度接近 1，Anchor 和 Negative 的接近 -1。

---

### 方案 B：伪语义分割（Pseudo-Semantic Segmentation）

如果你不仅有 SAM 的 Mask，还能通过 CLIP 或 Grounding DINO 给 Mask 打上标签（如"Building", "Tree", "Water"）。

- **做法**：直接把任务变成一个标准的语义分割任务（Cross-Entropy Loss）
- **优势**：引入了额外的语义信息
- **劣势**：可能会引入标签噪声（Label Noise），且与 $L_{Grounding}$ 任务在语义层面有一定重叠

---

### 方案 C：MaskFormer（二分图匹配）

用 MaskFormer 架构来作为 MTV 的一个 Head，专门消化 SAM 提供的无类别 Mask（Class-Agnostic Masks），以下是为你定制的完整架构设计与流程。

#### 目标

MTV-RS MaskFormer Head

- **输入**：ViT Backbone 提取的多尺度特征
- **真值（GT）**：SAM 生成的 $M$ 个 Mask（无类别标签，只有形状）
- **输出**：预测 $N$ 个 Mask，并计算 Loss

---

#### 1. 架构设计（Architecture）

这个 Head 分为三个部分：Pixel Decoder、Transformer Decoder 和 Prediction Heads。

**第一步：Pixel Decoder（像素解码器）**

这部分负责生成高分辨率的像素特征图，供 Mask 预测使用。

- **输入**：Backbone 的多层特征（如第 3, 6, 9, 12 层）
- **操作**：使用 FPN 或简单的上采样融合（这其实就是 MTV 原文中 DPT 的前半部分）
- **输出**：一个高分辨率的像素嵌入图 $\mathcal{E}_{pixel} \in \mathbb{R}^{C_{emb} \times H \times W}$
  - 注：$C_{emb}$ 是嵌入维度，比如 256

**第二步：Transformer Decoder（Query 交互）**

这部分是 MaskFormer 的核心，用固定数量的 Query 去"寻找"物体。

- **输入**：
  1. 图像特征 $\mathcal{F}$（通常是 Pixel Decoder 中间层输出的低分辨率特征，为了省计算量）
  2. 可学习的 Queries ($Q$)：数量为 $N$ 个向量（比如 $N=100$）。这个 $N$ 必须大于你预期的图像块中最大的物体数量

- **操作**：
  - Self-Attention：Query 之间互相交流（处理遮挡、去重）
  - Cross-Attention：Query 去图像特征 $\mathcal{F}$ 中提取信息

- **输出**：更新后的 Query 嵌入 $Q' \in \mathbb{R}^{N \times C_{emb}}$

**第三步：Prediction Heads（预测头）**

MaskFormer 标准版有两个头：分类头和 Mask 头。但在你的场景下（SAM 无类别），需要做特殊修改：

**Mask Head（点积预测）**：

- 计算 Query 嵌入 $Q'$ 与像素嵌入 $\mathcal{E}_{pixel}$ 的点积
- 公式：$M_{pred} = \text{Sigmoid}(Q' \cdot \mathcal{E}_{pixel})$
- 输出：$N$ 个预测 Mask ($N \times H \times W$)

**Objectness Head（物体置信度头）—— 关键修改**：

- 因为 SAM 没有"猫/狗"标签，原本的 Softmax 分类器要改成二分类（Binary Classification）
- 预测：每个 Query 是否击中了一个真实的物体（Object vs. $\varnothing$）
- 输出：$N$ 个置信度分数 ($N \times 1$)

---

#### 2. 匹配与损失（Matching & Loss）

这是最关键的一步。因为模型输出了 $N$ 个预测，而你有 $M$ 个 GT Mask（且 $N > M$），且它们没有固定顺序，所以必须用匈牙利算法（Hungarian Algorithm）进行二分图匹配。

**流程**：

**构建代价矩阵（Cost Matrix）**

计算每一个预测 Mask ($i \in [1, N]$) 和每一个 GT Mask ($j \in [1, M]$) 之间的匹配代价 $\mathcal{C}_{ij}$。代价通常由三部分组成：

$$
\mathcal{C}_{ij} = -\lambda_{cls} \cdot P_i(\text{obj}) + \lambda_{mask} \cdot \mathcal{L}_{Focal}(m_i, m_j) + \lambda_{dice} \cdot \mathcal{L}_{Dice}(m_i, m_j)
$$

- $P_i(\text{obj})$：Query $i$ 预测为物体的概率
- $\mathcal{L}_{Focal}$：二值交叉熵，衡量像素是否对齐
- $\mathcal{L}_{Dice}$：衡量 Mask 重叠度（对大小物体平衡很重要）

**二分图匹配（Bipartite Matching）**

使用 `scipy.optimize.linear_sum_assignment` 找到总代价最小的一一对应关系。

- 结果：找到了 $M$ 个"最佳匹配"的 Query
- 剩下的 $N-M$ 个 Query 被归为"无物体（No Object）"

**计算最终 Loss**：

- 对于匹配上的 $M$ 个 Query：
  - 计算 Mask Loss (Focal + Dice)
  - 计算 Objectness Loss（目标是 1）

- 对于未匹配的 $N-M$ 个 Query：
  - 只计算 Objectness Loss（目标是 0）
  - 不计算 Mask Loss（因为它们对应的真值是空）

---

#### 3. 总结实现 Checklist

如果你要在 MTV 框架里写这个代码，你需要：

- **Backbone Interface**：从 ViT 取出多层特征 list
- **Pixel Decoder**：写一个简单的 FPN，输出 pixel_embeddings ($B, C, H, W$)
- **Transformer Decoder**：可以直接调用 PyTorch 的 `nn.TransformerDecoder`，定义 $N$ 个 learnable query parameter
- **Matcher**：直接复用 DETR 或 MaskFormer 开源库里的 `HungarianMatcher`
- **Loss Function**：定义 Dice Loss 和 Focal Loss。由于是多任务学习，记得给这个 Loss 一个权重系数（比如 $\lambda=1.0$ 或更高，因为分割很难）

> **这对遥感的好处**：这种基于 Query 的方法比简单的卷积 Head 更强，因为它能天然处理重叠物体（Instance Segmentation），而且对遥感图像中密集的、排列紧凑的小物体（如停车场里的车、港口里的集装箱）有更好的分离能力。虽然计算量比简单的卷积大，但对于追求 SOTA 的预训练模型来说，这是一个非常高级且合理的升级。

---

## 3. 全局语义监督 ($L_{VL}$)
使用 SigLIP 的基于 sigmoid 的对比学习来对齐全局图像和文本嵌入，将对齐视为密集的成对二元分类，以获得更稳定的梯度。

MTV 论文中直接引用并复用了 SigLIP 的损失函数公式。
我们可以通过对比 MTV 的描述和 SigLIP 的原文来确认这一点：
1. 核心公式一致
MTV 中的公式 (1)：
$$
\mathcal{L}_{VL} = - \frac{1}{|B|} \sum_{i=1}^{|B|} \sum_{j=1}^{|B|} \log \frac{1}{1 + e^{y_{ij} (- \tau \mathbf{v}_i \cdot \mathbf{t}_j + \beta)}}
$$
其中 $y_{ij} = 1$ 表示正样本对，$y_{ij} = -1$ 表示负样本对。这是标准的 Sigmoid Loss for Language-Image Pre-training (SigLIP) 损失函数。 SigLIP Objective

2. 核心思想一致
MTV 选择 SigLIP 而非传统的 Softmax (CLIP) 主要基于以下两点理由，这与 SigLIP 原文的论点完全一致：

解耦 Batch Size： Softmax 需要在整个 Batch 内做归一化，导致 Batch Size 越大计算越复杂（分母爆炸）。SigLIP 把每个 $(image, text)$ 对视为独立的二分类问题（是匹配/不匹配），因此损失函数的计算不依赖于 Batch Size 的大小。
更稳定的梯度： 在大规模分布式训练中，SigLIP 的梯度表现更平滑，尤其是在处理 noisy web data 时。

3. 实现上的微小差异 (Implementation Detail)
虽然数学原理完全一样，MTV 在工程实现上做了一个微小的调整以适配其多任务框架：

SigLIP 原文： 推荐使用一种特殊的 chunked 实现或 rotation 通信策略，以避免在 TPU/GPU 之间传输巨大的全量 Embedding，从而节省显存。
MTV 的调整： 作者发现，由于还要跑 SSL 和 Dense 任务，Text Encoder 的通信开销相对来说变得微不足道了。所以 MTV 没有使用 SigLIP 复杂的切块技巧，而是直接使用了简单的 Global All-Gather（把所有 GPU 上的 Text Feature 收集到一起）。

引文： "we utilize a more direct and implementation-efficient alternative: we synchronize textual embeddings across all devices using a differentiable all gather operation." Implementation Change



总结： 算法原理上 100% 原版 SigLIP，工程实现上做了简化（去掉了切块优化）。

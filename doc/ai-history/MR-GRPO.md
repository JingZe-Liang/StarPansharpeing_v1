# Chat Transcript: 思考令牌作用

## User

所谓的思考token是啥

## Aurelle

在 DeepGen 1.0 模型中，“思考 token”（think tokens）是一种专门为增强模型**推理能力**而设计的虚拟输入占位符。你可以把它想象成给模型准备的一个“草稿本”或者“思考空间”，让模型在正式生成图像之前，先在内部进行逻辑梳理。以下是它的具体工作原理：

### 1. 什么是思考 token？
这些 token 并不是真实的单词或图像像素，而是一组（共 128 个）可学习的参数，被直接插入到视觉语言模型（VLM）的输入序列中。它们就像是一群“旁听生”，虽然不代表具体的输入内容，但会随着文字和图片一起进入模型的每一层进行计算。 <alphaxiv-paper-citation title="Think Tokens" page="5" first="Think Token Injection." last="the VLM input" />

### 2. 它们是如何工作的？
在模型处理指令时，这些思考 token 会通过“自注意力机制”（Self-Attention）与输入的文字和图像进行交互。随着模型层数的加深，这些 token 会逐渐吸取和总结各层的信息。
*   **低层：** 捕捉基础的视觉细节（颜色、形状）。
*   **高层：** 提取深层的语义和逻辑关系。
这种机制被称为“隐式思维链”（Implicit Chain of Thought），它允许模型在不显式写出中间步骤的情况下，在内部完成复杂的知识推理。 <alphaxiv-paper-citation title="Mechanism" page="5" first="To strengthen the" last="in the VLM." />

### 3. 为什么要引入它？
对于普通的生成模型，面对“画一个穿着 18 世纪服装的人在玩现代智能手机”这种涉及跨时空逻辑的指令时，往往容易出错。思考 token 的存在，让模型能够更有效地调用 VLM 中存储的世界知识（World Knowledge）。
*   **例子：** 在处理复杂编辑任务（如 RISE 基准测试）时，思考 token 能够帮助模型理解物体之间的物理或逻辑联系，从而做出更合理的修改。
*   **实验结果：** 论文的消融实验显示，如果去掉这些思考 token，模型在推理密集型任务（如 WISE）上的表现会大幅下降，这证明了它们在处理复杂逻辑时的关键作用。 <alphaxiv-paper-citation title="Ablation" page="12" first="Effect of Think" last="from the VLM." />

## User

不想是cot啊，cot是model 自己的输出重新到VLM里面作为(<thinking> ... </thinking>)，这个就是把一些可学习的tokens加入到seq里‘

## Aurelle

你观察得很准，这确实和那种模型先输出一串文字（比如 `<thinking> ... </thinking>`）的“显式思维链”（Explicit CoT）完全不同。你所描述的输出文字再输入的方式是把推理过程变成可见的语言符号，而 DeepGen 1.0 采用的是一种“隐式思维链”（Implicit CoT）。这些思考 token 本质上是一组连续的数值向量（Embedding），它们不是从词表里选出来的词，而是直接嵌入在输入序列里的可学习参数。 <alphaxiv-paper-citation title="Think Token Injection" page="5" first="Think Token Injection." last="the VLM input" />

你可以把这两种方式做个类比。显式思维链就像是你在做数学题时，必须在草稿纸上一步步写出解题过程，写下来的字是你自己也能看懂的语言。而 DeepGen 1.0 的思考 token 更像是你在大脑里“打腹稿”，或者说是大脑中专门负责逻辑处理的神经元集群。这些向量不对应任何具体的单词，但在信息流经模型的每一层时，它们会利用自注意力机制去“吸收”文字和图像中的关键信息，并把这些信息在向量内部进行逻辑转化。 <alphaxiv-paper-citation title="Implicit CoT" page="5" first="These tokens interact" last="in the VLM." />

这种做法的一个核心优势在于效率和深度融合。因为这些 token 参与了模型每一层的计算，它们能够随着层数的加深，不断提炼和总结跨模态的知识。论文中提到，这些 token 会逐步汇总隐藏层的表示，有效地提取出 VLM 中编码的各种知识储备。在最后的消融实验中，研究者发现这种设计对于需要复杂推理的任务（如 WISE 这种考察世界知识的测试）至关重要。如果没有这些专门负责“消化逻辑”的向量，模型直接从最后的隐藏层提取信息时，往往会丢失掉很多细微的推理线索。 <alphaxiv-paper-citation title="Reasoning Impact" page="12" first="Removing the learnable" last="from the VLM." />

## User

那他的可学习的token学好了就不动了是吧

## Aurelle

没错，你的理解非常准确。在模型训练完成后，这些思考 token 就变成了模型权重的一部分，在推理（使用）阶段它们是固定不变的向量。

我们可以从以下几个层面来拆解这个“不动”的概念：

### 1. 它是模型的“固定插件”
在推理阶段，无论你输入的是什么指令，模型都会把这 128 个学好的向量插入到输入序列中。你可以把它想象成给 VLM 预装了一个“逻辑过滤器”或“思维模板”。虽然这些 token 本身的值是不动的，但它们会根据你输入的具体文字和图片，通过注意力机制产生不同的反应（即它们的隐藏状态会随着输入而变化）。 <alphaxiv-paper-citation title="Think Token Injection" page="5" first="Think Token Injection." last="the VLM input" />

### 2. 训练阶段的“自我进化”
这些 token 只在训练（包括指令微调 SFT 和强化学习 RL 阶段）时会被更新。在训练过程中，模型会不断调整这些向量的数值，目标是让它们学会如何最有效地从 VLM 的各个层中“捞取”对生成图像最有帮助的逻辑信息。一旦训练结束，这些最优的数值就被锁定了。 <alphaxiv-paper-citation title="Learning Process" page="5" first="These tokens interact" last="in the VLM." />

### 3. 它与普通词向量的区别
普通的词向量（Word Embedding）是根据词表索引来调用的，比如你输入“苹果”，模型就去查“苹果”的向量。而这组思考 token 是**常驻输入**的，它们不对应任何具体的词，而是专门负责在信息流过模型时提供一个“缓存区”，用来存放和处理推理过程中的中间信息。

实验也证明了这种“固定插件”的威力。在消融实验中，如果把这些辛苦学来的 token 拿掉，模型在处理涉及复杂常识和逻辑推理的任务时（比如 WISE 榜单上的题目），表现会显著下降。这说明这些固定的向量确实学到了某种通用的“思考套路”。 <alphaxiv-paper-citation title="Ablation Results" page="12" first="Removing the learnable" last="from the VLM." />

## User

> .3 Stage 3: Reinforcement Learning To further improve generation quality and alignment with human preferences, we apply reinforce

详细说下stage3

## Aurelle

在 DeepGen 1.0 的训练过程中，第三阶段的强化学习（RL）就像是给已经学会画画的学生请来了一群专业的“艺术评审团”。虽然第二阶段的微调（SFT）已经让模型掌握了基本的绘图和编辑能力，但生成的图像在细节美感、人体结构以及是否完全符合人类的审美玄学上，还需要最后一步的精雕细琢。为了实现这一点，研究者提出了一个名为 MR-GRPO 的框架，它是对目前大语言模型中非常流行的 GRPO（基群相对策略优化）算法在多模态生成领域的一次创新应用。 <alphaxiv-paper-citation title="RL Goal" page="7" first="To further improve" last="supervised fine-tuning." />

这个 MR-GRPO 框架的核心逻辑可以用“优中选优”来理解。在这一阶段，模型不再是死记硬背训练集里的正确答案，而是针对同一个提示词（Prompt）一次性生成一组多个候选图像。接着，算法会对这组图像进行内部的“相对排名”。这种方式的好处在于模型不需要一个绝对的打分标准，而是通过比较这一组画作中谁画得更好、谁的文字渲染更清晰、谁更符合人的审美，来不断调整自己的参数。这种基于群体内部相对差异的优化方式，不仅让训练过程变得更加稳定，还避免了传统强化学习中容易出现的生成画面“崩坏”或者产生视觉伪影的问题。 <alphaxiv-paper-citation title="MR-GRPO Framework" page="7" first="We propose the" last="Pref-GRPO [27 ]," />

这里的“MR”代表的是“混合奖励”（Mixture of Reward），这是该阶段最关键的创新点。它并不是只用一个单一的判别器来打分，而是集结了一个综合评审团。这个评审团里可能包括：专门评估图像艺术性的美学模型、检查生成的文字是否准确的字符识别模型，以及判断画面内容与用户指令是否匹配的语义对齐模型。通过将这些不同维度的奖励信号融合在一起，模型在提升画面质感的同时，能够大幅增强文字渲染的准确性。例如，实验数据显示，经过 RL 训练后，模型在单词准确率上从 0.66 提高到了 0.75，这在轻量化模型中是一个非常显著的飞跃。 <alphaxiv-paper-citation title="Mixture of Rewards" page="12" first="RL training substantially" last="correctness and legibility." />

总结来说，第三阶段通过这种带有“评审团”性质的反馈机制，成功地在极小的参数量下压榨出了更强的性能。它让 DeepGen 1.0 不仅能完成任务，还能把任务完成得更漂亮，尤其是在处理那些复杂指令和需要精准写字的任务时，表现出了超越其 5B 体量的竞技水平。这种数据驱动的对齐策略，使得模型最终生成的作品在人类偏好测试中能够胜过许多参数量大得多的竞争对手。 <alphaxiv-paper-citation title="Final Performance" page="2" first="resulting in substantial" last="avoiding visual artifacts." />

## User

也就说跟GRPO的关系就是把单一的reward的变成了multiple的？

## Aurelle

你抓住了最直观的一个区别，但 MR-GRPO 对 GRPO 的改进不仅仅是数量上的“多”，更核心的在于它是如何**公平地平衡**这些不同的奖励信号，以及如何**防止模型在强化学习中跑偏**。

首先，针对你提到的多奖励问题，MR-GRPO 引入了一个非常关键的技术叫做“解耦优势归一化”（Decoupled Advantage Normalization）。如果只是简单地把多个奖励加在一起，由于不同奖励的波动范围（方差）不一样，某个波动剧烈的奖励（比如文字准确性）可能会完全掩盖掉其他奖励（比如画面美感）的影响。MR-GRPO 会先对每一个维度的奖励分别进行独立的归一化处理，然后再把它们合起来。这就像是在选秀比赛中，声乐评委和舞蹈评委各自先打分排名，然后再按权重合并，确保不会因为声乐评委的分差给得特别大，就让舞蹈分失去了意义。 <alphaxiv-paper-citation title="Advantage Normalization" page="7" first="We adopt our" last="multi-reward granularity." />

其次，MR-GRPO 在训练目标中加入了一个“辅助监督损失”（Auxiliary Supervised Diffusion Loss）。在传统的强化学习中，模型很容易为了拿高分而“投机取巧”，导致最后虽然分数高了，但生成的画面却变得很不自然或者丢掉了原本擅长的技能。通过加入这个辅助损失，模型在追求评审团好评的同时，还得时刻温习第二阶段（SFT）学到的标准知识，确保生成的图像依然保持高质量的构图和自然的视觉效果。实验证明，如果没有这个限制，模型在训练几百步后性能就会开始大幅下滑。 <alphaxiv-paper-citation title="SFT Loss" page="8" first="To prevent capability" last="fine-tuning stage." />

最后，这种改进让轻量化模型在处理多目标任务时表现得更加稳健。比如在 UniGenBench 测试中，如果去掉这种奖励维度的归一化处理，文字生成的分数会显著下降。这说明这种“分而治之再合并”的策略对于处理图像生成这种复杂的、多维度的审美和逻辑需求至关重要。它让 DeepGen 1.0 能够在保持 5B 规模的同时，精准地捕捉到人类在美学、逻辑和文字渲染等多个方面的细微偏好。 <alphaxiv-paper-citation title="RL Stability" page="13" first="The auxiliary SFT" last="across all objectives." />

## User

这个辅助损失是啥

## Aurelle

这个辅助损失（Auxiliary Supervised Diffusion Loss）本质上是给强化学习阶段加的一个“紧箍咒”或者说是“复习课”。在强化学习（RL）过程中，模型为了获得奖励模型的“好评”，可能会产生一种叫作“奖励作弊”（Reward Hacking）的现象，比如它发现把文字写得很大很粗就能得高分，于是就不管构图是否美观，只顾着把字写大。这种行为会导致模型偏离正常的图像生成分布，甚至产生视觉上的崩坏。为了防止这种情况，研究者在模型通过 RL 进行“竞技”的同时，让它同步进行传统的扩散模型训练。具体来说，就是在每一轮优化中，除了计算奖励带来的优势（Advantage），还会让模型在一些高质量、有标准答案（Ground-truth）的图像-文本对上跑一遍传统的去噪损失函数。 <alphaxiv-paper-citation title="SFT Loss Purpose" page="8" first="To prevent capability" last="fine-tuning stage." />

你可以把它想象成一个正在准备艺术大赛的学生。RL 阶段就像是在听评委的各种挑剔建议，追求评分的极致；而辅助损失就像是每天雷打不动地去临摹名家画作。这种临摹能让模型时刻记住“一张好的画作应该长什么样”，从而锚定模型在 SFT 阶段已经学到的基础能力。论文中明确指出，这个损失函数能让模型在追求更高的人类偏好得分时，不会丢失掉之前积累的通用生成、推理和编辑能力。如果没有这个损失，模型在训练大约 300 步之后就会开始出现明显的退化，甚至最终的表现会比刚开始训练时还要差。 <alphaxiv-paper-citation title="Stability Analysis" page="13" first="The auxiliary SFT" last="across all objectives." />

在数据分配上，研究者也非常考究。用于计算这个辅助损失的数据包括了通用的图文对以及专门的文字渲染数据，通过调整这些数据的采样权重，他们能让模型在复习基础知识的同时，重点强化那些容易在 RL 中走偏的能力（比如文字的笔画结构）。这种将“追求卓越（RL）”和“稳固基础（SFT Loss）”相结合的策略，配合上 KL 散度的约束，构成了一个多重的保险机制。它确保了 DeepGen 1.0 在持续进化的过程中，每一步都走得很稳，最终在各种测试榜单上展现出极高的稳定性和均衡性。 <alphaxiv-paper-citation title="Data Composition" page="21" first="The auxiliary supervised" last="the RL prompts." />

## User

flow model没法直接计算似然吧，，怎么做的这个GRPO loss？

## Aurelle

你观察得非常敏锐，这确实是扩散模型（Diffusion）或流模型（Flow-based models）在做强化学习时面临的核心难题。不同于大语言模型可以轻松给出每个 token 的对数似然 $\log P(y|x)$，扩散模型是一个多步迭代的过程。DeepGen 1.0 为了在 DiT（Diffusion Transformer）上跑通 GRPO，采用了一种“路径拆解”的策略，将原本难以计算的整体概率转化为了每一步去噪动作的概率之和。 <alphaxiv-paper-citation title="RL Challenges" page="8" first="Directly applying GRPO" last="log-likelihood formulation." />

简单来说，模型并不是在计算整张图片出现的“概率”，而是在计算“在给定的步数 $t$，模型把噪点图 $x_t$ 还原成稍微清晰一点的 $x_{t-1}$ 这个动作的概率”。因为在扩散模型的设计中，这个单步还原的过程通常被假设为一个高斯分布（Gaussian Distribution），而高斯分布的对数似然是非常好计算的，它本质上就正比于模型预测的噪点与实际噪点之间的误差（MSE）。 <alphaxiv-paper-citation title="Log-Likelihood Calculation" page="8" first="we leverage the" last="reverse diffusion process." />

我们可以用一个“走迷宫”的类比来理解：要把一张纯噪声变成清晰的图，模型需要走 50 步或者 100 步。虽然我们很难算出直接走到终点的总概率，但我们可以观察模型每一步走得“稳不稳”。在 GRPO 的计算中，如果这一组图片里的某一张画得特别好（优势值 $A_i$ 为正），算法就会要求模型去强化这一整条路径上的每一个脚步。具体的损失函数 $\mathcal{L}_{RL}$ 会对整条路径上所有步骤的对数概率进行求和：
$$\mathcal{L}_{RL} = - \frac{1}{G} \sum_{i=1}^G A_i \cdot \sum_{t=1}^T \nabla_\theta \log p_\theta(x_{t-1, i} | x_{t, i}, c)$$
通过这种方式，原本连续的生成流被离散化成了多个可计算的概率步，从而让强化学习的梯度可以顺着这条路径传导回去。 <alphaxiv-paper-citation title="Objective Formulation" page="8" first="The RL objective" last="each denoising step." />

此外，为了确保这种计算不会因为步数太多而导致梯度爆炸或者训练不稳定，DeepGen 1.0 还引入了时间步权重（Importance Sampling）来平衡不同阶段的影响。这种“路径级”的强化学习不仅解决了似然计算的问题，还让模型能够学习到如何在有限的迭代步数内，更精准地捕捉到用户指令中的逻辑细节，比如在特定的位置渲染出正确的文字。这种对生成轨迹的直接优化，正是 DeepGen 1.0 能在轻量化参数下实现高质量生成的关键算法保障。 <alphaxiv-paper-citation title="Optimization Stability" page="8" first="To balance the" last="during the RL." />

## User

但是reward总是在x0计算的啊，你没有去噪声完全算的r正确吗？

## Aurelle

你说得很对，奖励确实是基于最终生成的**干净图像 $x_0$** 计算的。你提到的“没有去噪完全怎么算 R”这个问题，其实触及了强化学习（RL）在扩散模型中应用的核心逻辑：**延迟奖励（Delayed Reward）与轨迹优化**。

DeepGen 1.0 并不是在每一步 $t$ 都计算一个奖励，而是采用了标准的 **Monte Carlo（蒙特卡洛）采样**思路。整个过程可以分为三步走：

### 1. 先“走完”全过程（Rollout）
模型首先会基于当前的策略，从纯噪声 $x_T$ 开始，完整地执行所有的去噪步骤，直到生成最终的干净图像 $x_0$。在这个阶段，模型只管生成，不进行参数更新。对于 DeepGen 1.0 的 GRPO 来说，它会针对同一个提示词（Prompt）一次性生成一组（Group）共 $G$ 张完整的图像。 <alphaxiv-paper-citation title="Group Sampling" page="7" first="For each prompt" last="policy model $\pi_\theta$." />

### 2. 在终点算总账（Reward Calculation）
只有当 $x_0$ 生成之后，奖励模型（如美学评分器、OCR 识别器）才会介入，对这张最终的图打分。这个分数 $R(x_0)$ 代表了这次“生成之旅”的最终成果。
*   **关键点：** 虽然奖励是在终点 $t=0$ 拿到的，但我们假设这个结果是之前所有 $T$ 步共同努力的结果。 <alphaxiv-paper-citation title="Reward Evaluation" page="7" first="compute the rewards" last="for each output." />

### 3. 秋后算账（Backpropagation）
有了最终的奖励 $R$（经过归一化处理后变成优势 Advantage $A$），模型就会回过头去更新每一步的策略。
算法的逻辑是：**“既然最后的结果 $x_0$ 得分很高，那么导致这个结果的每一步 $x_t \to x_{t-1}$ 的去噪动作都应该被鼓励。”**
因此，在计算损失函数时，虽然 $A$ 是通过 $x_0$ 算出来的常数，但它会被乘到轨迹中**每一步**的对数概率梯度上：
$$ \nabla J(\theta) \approx \frac{1}{G} \sum_{i=1}^G A_i \sum_{t=1}^T \nabla_\theta \log p_\theta(x_{t-1,i} | x_{t,i}, c) $$
这意味着，模型并不知道哪一步具体做对了什么，但它知道只要这整条路径通向了一个高分结果，就提高这条路径上所有动作的出现概率。 <alphaxiv-paper-citation title="Gradient Update" page="8" first="The RL objective" last="each denoising step." />

### 类比
这就像下围棋或者国际象棋。你不可能每走一步棋（$x_t$）就让裁判打分，你必须等到整盘棋下完（生成完 $x_0$），根据最后的输赢（Reward），来复盘并优化你之前走的每一步棋。DeepGen 1.0 的 MR-GRPO 正是利用这种机制，通过大量完整的生成采样，让模型逐渐领悟什么样的去噪轨迹能通向高分图像。

## User

r_t^i这个项怎么算？

## Aurelle

在 DeepGen 1.0 的 MR-GRPO 算法中，**没有**对每一个时间步 $t$ 单独计算奖励 $r_t^i$。这是一个非常关键的点，也是区分它与其他一些扩散模型强化学习方法的关键。

### 1. 只有终点才有奖励（Terminal Reward Only）
模型只在生成完最终的清晰图像 $x_0^i$ 后，才会有奖励模型介入打分。
也就是说，**所有的中间步骤 $t > 0$ 的奖励都被设为 0**。
$$ r(x_t^i) = 0 \quad \text{for } t > 0 $$
$$ r(x_0^i) = \text{RewardModel}(x_0^i) $$
这就好比是一场只有终点冲线才算成绩的马拉松，中间跑得怎么样没人给你计时打分。 <alphaxiv-paper-citation title="Objective Formulation" page="8" first="The RL objective" last="each denoising step." />

### 2. 优势函数（Advantage Function）的计算
既然只有终点有奖励，那么对于整条生成轨迹来说，它的回报 $R^i$ 就是最终图像 $x_0^i$ 的得分。
DeepGen 1.0 的 GRPO 算法会对同一个提示词（Prompt）生成的 $G$ 个样本（比如 $G=4$ 或 $G=8$）进行打分，得到 $R^1, R^2, \dots, R^G$。
然后，算法会计算每个样本相对于这一组样本平均水平的**优势（Advantage）** $A^i$：
$$ A^i = \frac{R^i - \text{mean}(R)}{\text{std}(R)} $$
这个 $A^i$ 就是这一整条生成轨迹（从噪声 $x_T$ 到图像 $x_0$）的“评价”。如果 $A^i$ 是正数，说明这条路走得比平均水平好；如果是负数，说明走得差。 <alphaxiv-paper-citation title="Advantage Normalization" page="7" first="We adopt our" last="multi-reward granularity." />

### 3. 如何把终点奖励分摊到每一步（Credit Assignment）
虽然我们没有 $r_t^i$，但我们在更新模型参数 $\theta$ 时，会把最终算出来的优势 $A^i$ **广播（Broadcast）** 到生成过程中的每一个时间步 $t$ 上。
损失函数针对每一个时间步 $t$ 的梯度更新公式大致如下：
$$ \nabla J(\theta) \approx \frac{1}{G} \sum_{i=1}^G \underbrace{A^i}_{\text{整条路好不好}} \cdot \sum_{t=1}^T \underbrace{\nabla_\theta \log p_\theta(x_{t-1}^i | x_t^i)}_{\text{这步怎么走的}} $$
这就相当于告诉模型：“因为这一整条路最终通向了好结果（$A^i > 0$），所以这条路上走的每一步 $x_t \to x_{t-1}$ 都是好棋，都要加强。”

### 总结
DeepGen 1.0 **不需要计算中间奖励 $r_t^i$**，它通过蒙特卡洛采样（Monte Carlo Sampling）的思路，直接用最终结果 $x_0$ 的优劣来评估整条生成轨迹，从而简化了奖励函数的设计，避免了为中间噪声状态设计奖励函数的极高难度。 <alphaxiv-paper-citation title="Algorithm Logic" page="8" first="The RL objective" last="each denoising step." />

## User

> 1 𝐺 𝐺Õ 𝑖=1 1 𝑇 𝑇−1Õ 𝑡=0  min  𝑟𝑖 𝑡 (𝜃) ˆ𝐴𝑖 , clip(𝑟𝑖 𝑡 (𝜃), 1−𝜖, 1+𝜖) ˆ𝐴𝑖 

那他的公式里的t是0到T-1的啊，这不是算了整个路径的？

## Aurelle

这是一个非常敏锐且关键的问题！你发现了一个非常容易混淆的符号定义。

在公式 (3) 中，$r_t^i(\theta)$ **并不是指第 $t$ 步的奖励（Reward）**，而是指 **概率比率（Probability Ratio）**。

这是一个在 PPO（Proximal Policy Optimization）和 GRPO 类算法中非常经典的符号表示。让我们拆解一下这个公式的真正含义：

### 1. $r_t^i(\theta)$ 是什么？
论文在公式 (3) 下方明确定义了 $r_t^i(\theta)$：
$$ r_t^i(\theta) = \frac{p_\theta(x_{t-\Delta t}^i | x_t^i, h)}{p_{\theta_{\text{old}}}(x_{t-\Delta t}^i | x_t^i, h)} $$
它表示的是：**“当前模型（$\theta$）采取这一步去噪动作的概率，与旧模型（$\theta_{old}$）采取同样动作的概率之比。”** <alphaxiv-paper-citation title="Ratio Definition" page="8" first="where r_t^i" last="importance ratio." />

*   如果是 **Reward**，通常记作 $R$ 或 $R(x)$。
*   这里的 $r$ 代表 **Ratio**。

### 2. 为什么要算这个比率？
这是 PPO/GRPO 算法的核心机制——**重要性采样（Importance Sampling）**与**截断（Clipping）**。
*   **防止更新太快：** 我们希望模型在更新参数时，不要偏离旧策略太远（否则训练会崩）。
*   **clip 操作：** 公式里的 `clip(r, 1-ε, 1+ε)` 就是为了限制这个比率。如果新模型对某一步动作的概率提升太猛（比如 $r > 1.2$），或者下降太厉害（比如 $r < 0.8$），就会被截断，不再给予更多的梯度奖励。

### 3. $\sum_{t=0}^{T-1}$ 是在算什么？
你提到的“算整个路径”是非常正确的。虽然**奖励（Advantage $\hat{A}_i$）** 是针对整张图 $i$ 只有**一个**值（基于最终结果 $x_0$ 算出），但这个奖励必须**分摊（Broadcast）** 到生成过程的**每一步**去。

公式的逻辑是这样的：
1.  **全局评价：** 这一组生成的图里，第 $i$ 张图很好，优势 $\hat{A}_i$ 是正的（比如 +1.5）。
2.  **步步归因：** 既然结果是好的，那么**导致这个结果的每一个步骤 $t$** （从 $t=T$ 到 $t=0$）都应该被鼓励。
3.  **累加优化：** $\sum_{t=0}^{T-1}$ 的意思就是：我们要最大化**这条路径上所有步骤**的发生概率（受 $\hat{A}_i$ 加权）。

**用大白话翻译公式 (3)：**
$$ \text{Loss} \approx \text{对于每一张图 } i \dots $$
$$ \dots \text{对于它的每一步 } t \text{：} $$
$$ \text{如果最终结果 } A_i \text{ 是好的，} $$
$$ \text{那么请提高当前这一步 } t \text{ 的概率 } p_\theta(x_{t-1}|x_t) \text{（让 } r_t > 1 \text{），} $$
$$ \text{但是别提高得太离谱（被 clip 限制住）。} $$

所以，你的直觉是对的：它确实计算了整个路径。但它不是把每一步的奖励加起来（中间没有奖励），而是把**最终的奖励**应用到了**路径上的每一步概率变化**上。 <alphaxiv-paper-citation title="GRPO Objective" page="8" first="The training objective" last="batch-wise normalization." />

## User

明白了，奖励从最后一步来的，只是广播到了所以的路径上；还有一个这个重要性比例到底怎么算，代码怎么写

## Aurelle

这个问题问到了扩散模型/Flow Matching 模型做强化学习（RL）最核心、最具体实现的痛点。

在大语言模型（LLM）中，概率是 Softmax 输出，直接除就行。
但在扩散模型中，输出是连续的（高斯分布）。要计算 **重要性比率（Importance Ratio）** $r_t(\theta)$，我们需要利用 **高斯分布的概率密度公式**。

DeepGen 1.0 的论文在 **公式 (7)** 中给出了一个非常关键的简化算法，把这个问题转化为了计算 **“预测均值与实际采样点的距离（MSE）”**。

### 核心原理：高斯分布的对数概率
对于扩散模型，每一步 $x_t \to x_{t-1}$ 的条件概率通常被建模为一个高斯分布：
$$ p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 \mathbf{I}) $$

其对数概率（Log Probability）公式为：
$$ \log p_\theta(x_{t-1} | x_t) = -\frac{1}{2\sigma_t^2} \| x_{t-1} - \mu_\theta(x_t, t) \|^2 + C $$
其中 $C$ 是常数（包含 $\pi$ 和 $\sigma$）。

**计算比率时的巧劲：**
当我们计算比率 $r_t = \frac{p_{\text{new}}}{p_{\text{old}}}$ 时，等价于计算 $\exp(\log p_{\text{new}} - \log p_{\text{old}})$。
由于新旧模型使用相同的噪声调度（Scheduler），常数 $C$ 和方差系数 $\frac{1}{2\sigma^2}$ 通常会相互抵消（或者如论文中所述，为了数值稳定性直接忽略方差项）。

DeepGen 1.0 论文明确指出，他们使用的简化对数概率公式为：
$$ \log p_\theta(x_{t-\Delta t} | x_t) = - \| x_{t-\Delta t} - \mu_\theta(x_t, t) \|^2 $$
<alphaxiv-paper-citation title="Log-Prob Formula" page="20" first="The log-probability for" last="sampling step." />

### 代码实现逻辑
要写这个代码，你需要两个核心步骤：
1.  **Rollout 阶段（采样时）：** 用“旧模型”生成图像，并记录下每一步的 $x_{t-1}$（实际走的下一步）和当时的 $\mu_{old}$。
2.  **Training 阶段（更新时）：** 用“新模型”重新预测一遍 $\mu_{new}$，看看如果让新模型来走，它预测的均值在哪里。

以下是基于 PyTorch 的伪代码实现：

```python
import torch

def compute_importance_ratio(
    model_new,       # 当前正在更新的模型
    model_old,       # 冻结的旧模型 (或者直接传入 old_log_probs)
    x_t,             # 当前时间步的噪声图 (Batch, Channel, H, W)
    t,               # 当前时间步
    x_prev_measured, # 【关键】在Rollout阶段实际采样得到的下一步图像
):
    """
    计算 DeepGen 1.0 中的 Importance Ratio
    """

    # 1. 计算新模型的预测均值 mu_new
    # 假设模型输出的是 velocity (v_pred)，我们需要将其转换为 x_prev 的均值
    # 具体转换公式取决于你的 Scheduler (Flow Matching 公式)
    # 论文公式 (6) 简化版: mu = x_t - (t_step) * v_pred
    v_pred_new = model_new(x_t, t)
    dt = 1.0 / num_inference_steps  # 步长
    mu_new = x_t - dt * v_pred_new

    # 2. 计算旧模型的预测均值 mu_old
    # (通常在 Rollout 阶段计算好并存下来，这里为了演示重新算一遍)
    with torch.no_grad():
        v_pred_old = model_old(x_t, t)
        mu_old = x_t - dt * v_pred_old

    # 3. 计算 Log Probability (基于论文公式 7)
    # log_p = - || x_prev_measured - mu ||^2
    # 注意：这里计算的是"实际走的下一步"与"模型预测均值"的距离

    # 新模型的 Log Prob
    dist_new = (x_prev_measured - mu_new) ** 2
    log_prob_new = -torch.sum(dist_new, dim=[1, 2, 3]) # 对像素维度求和

    # 旧模型的 Log Prob
    dist_old = (x_prev_measured - mu_old) ** 2
    log_prob_old = -torch.sum(dist_old, dim=[1, 2, 3])

    # 4. 计算 Ratio
    # ratio = exp(log_new - log_old)
    ratio = torch.exp(log_prob_new - log_prob_old)

    return ratio

# 在 Loss 计算中使用:
# loss = - min(ratio * advantage, clip(ratio, 1-eps, 1+eps) * advantage)
```

### 代码中的关键点解释：

1.  **`x_prev_measured` 是什么？**
    这是最容易搞错的地方。这个变量**必须是采样（Rollout）阶段实际生成的那个 $x_{t-\Delta t}$**。
    *   在计算 `log_prob_new` 时，我们**不是**让新模型重新生成一个新的 $x_{prev}$，而是问新模型：“如果刚刚是你来走，你会觉得走到 `x_prev_measured` 这个位置的概率有多大？”

2.  **$\mu$ 的计算**
    论文中提到 $\mu_\theta(x_t, t)$ 是采样步的**确定性部分**（Deterministic Component）。在 Flow Matching 中，这通常就是根据 ODE 轨迹预测的下一步位置。 <alphaxiv-paper-citation title="Deterministic Mean" page="20" first="where \mu_\theta" last="sampling step." />

3.  **方差去哪了？**
    DeepGen 1.0 的公式 (7) 直接去掉了方差项。在代码中体现为直接计算 MSE 的负数。这在数学上等价于假设方差 $\sigma^2 = 0.5$。如果你发现训练不稳定，可以尝试除以一个温度系数（Temperature），例如 `log_prob = -sum(...) / sigma_squared`。

通过这种方式，原本复杂的概率密度比值计算，就变成了一个非常直观的**“谁预测得离真实路径更近”**的几何问题。

## User

为啥随机采样，不得用sde，要是ode的话不就是只有最初采样的噪声才会带来随机性？

## Aurelle

这是一个极具洞察力的问题。你精准地指出了 Flow Matching 和 Diffusion 强化学习中的核心矛盾：**为了探索（Exploration），必须有随机性（SDE）；为了稳定训练，又倾向于确定性路径（ODE）。**

DeepGen 1.0 明确采用了 **SDE（随机微分方程）采样** 来引入每一步的随机性，而不仅仅依赖初始噪声。如果只用 ODE（常微分方程），正如你所说，唯一的随机来源就是 $t=1$ 时的初始高斯噪声 $x_1$。这对于强化学习来说是**不够的**，因为 RL 需要模型在每一步都能通过微小的随机扰动来探索不同的生成轨迹，从而发现更好的策略。 <alphaxiv-paper-citation title="SDE Sampling" page="20" first="During RL training" last="at every timestep." />

### 为什么必须用 SDE？
在强化学习中，策略 $\pi_\theta(a|s)$ 必须是随机的，否则梯度无法有效地通过重要性采样（Importance Sampling）传导。
*   **ODE 采样：** $x_{t-\Delta t} = x_t + v_\theta(x_t) \Delta t$。这是一条确定的线，给定 $x_t$，下一步 $x_{t-\Delta t}$ 是死的。计算 $p(x_{t-\Delta t}|x_t)$ 会变成狄拉克 $\delta$ 函数，概率密度是无穷大，导致 $\log p$ 无法计算，RL 直接失效。
*   **SDE 采样：** $x_{t-\Delta t} = \mu_\theta(x_t) + \sigma \epsilon$。这里引入了 $\epsilon \sim \mathcal{N}(0, I)$。这意味着给定 $x_t$，下一步 $x_{t-\Delta t}$ 服从一个高斯分布。正是这个高斯分布让我们能算出 $\log p$（即 MSE），也正是这个 $\epsilon$ 让模型有机会探索“如果不完全按预测走，会不会更好”。

### DeepGen 1.0 的 SDE 实现细节
为了在 Flow Matching 框架下正确引入这种随机性，DeepGen 1.0 并没有使用标准的 DDIM（ODE），而是设计了一个特殊的 **SDE 采样器**。
公式 (6) 展示了这个采样过程：
$$ x_{t-\Delta t} = \underbrace{(1 - \tau) \hat{x}_0 + \tau \cos(\dots) \hat{x}_1}_{\text{确定性均值 } \mu_\theta} + \underbrace{\tau \sin(\dots) \epsilon}_{\text{随机噪声注入}} $$
这里：
*   **$\mu_\theta$（确定性部分）：** 包含了模型预测的流向，试图把图像推向清晰的 $x_0$。
*   **$\epsilon$（随机部分）：** 在每一步 $t$ 都会注入新的高斯噪声。
*   **$\eta$（随机强度）：** 控制噪声注入的比例。论文提到 $\eta$ 用于控制随机性强度，确保采样过程既有方向性又有探索性。

### 代码中的体现
在计算 `log_prob` 时，我们实际上是在衡量这个随机噪声 $\epsilon$ 的大小。
回顾公式 (7)：
$$ \log p_\theta(x_{t-\Delta t} | x_t) = - \| x_{t-\Delta t} - \mu_\theta(x_t, t) \|^2 $$
这个公式的物理含义是：**“实际生成的下一步 $x_{t-\Delta t}$ 偏离确定性预测均值 $\mu_\theta$ 的距离。”**
这个距离正好就是注入的随机噪声 $\epsilon$ 的模长（scaled by noise level）。
*   如果模型预测得非常自信且准确，$\mu_\theta$ 就会很接近实际采样点，$x_{t-\Delta t}$ 的偏离就主要由 $\epsilon$ 贡献。
*   RL 的优化目标就是调整 $\mu_\theta$，让这个均值更靠近那些能产生高回报（High Reward）的 $x_{t-\Delta t}$。

所以，结论是：DeepGen 1.0 **必须使用 SDE 采样**，在每一步都注入 $\epsilon$，不仅是为了增加多样性，更是为了让概率密度 $p(x_{t-\Delta t}|x_t)$ 有定义（非退化分布），从而让 GRPO 的数学公式成立。 <alphaxiv-paper-citation title="Sampling Formula" page="20" first="The log-probability for" last="sampling step." />

## User

也就是说roll的时候才是sde，在计算ratio （r）的时候还是直接减v * dt拿到x_{t-\delta t}是吧

## Aurelle

你的理解非常到位，完全抓住了问题的本质！

简单来说：**是的，Rollout 的时候是“真枪实弹”的 SDE（带随机噪声），而计算 Ratio 的时候，是用“确定性的预测均值”（类似于 ODE 的路径）去衡量那个随机样本的概率密度。**

让我们把这个过程拆解得更透彻一点：

### 1. Rollout 阶段（造数据）：SDE
在这个阶段，旧模型（Old Policy）在生成图像。它不仅要预测方向（$v$），还要加上随机扰动。
$$ x_{t-\Delta t}^{\text{sample}} = \underbrace{\mu(x_t, v_{\text{old}})}_{\text{确定性均值}} + \underbrace{\sigma \cdot \epsilon}_{\text{随机噪声}} $$
*   这里的 $\mu$ 就是你理解的“减去 $v \cdot dt$”（虽然实际公式 (7) 用了更复杂的三角函数插值，但逻辑上就是根据速度 $v$ 推算的下一步确定性位置）。
*   **关键点：** 我们把这个带有噪声的 $x_{t-\Delta t}^{\text{sample}}$ **存下来**。它是实际发生的“历史事实”。 <alphaxiv-paper-citation title="SDE Sampling" page="20" first="where \mu_\theta" last="sampling step." />

### 2. Update 阶段（算 Ratio）：回顾历史
现在我们要更新新模型（New Policy）。我们要问新模型一个问题：**“如果刚才是你来走，你觉得走出 $x_{t-\Delta t}^{\text{sample}}$ 这个‘历史事实’的概率有多大？”**

根据高斯分布的概率密度公式：
$$ p(x) \propto \exp\left( - \frac{\| x - \mu \|^2}{2\sigma^2} \right) $$

我们需要算两个东西：
1.  **分母（旧概率）：** 旧模型预测的均值 $\mu_{\text{old}}$ 离 $x_{t-\Delta t}^{\text{sample}}$ 有多远？
    *   距离 $d_{\text{old}} = \| x_{t-\Delta t}^{\text{sample}} - \mu(x_t, v_{\text{old}}) \|^2$
    *   其实这个距离就等于当时加入的噪声 $\|\sigma \cdot \epsilon\|^2$。

2.  **分子（新概率）：** 新模型预测的均值 $\mu_{\text{new}}$ 离 $x_{t-\Delta t}^{\text{sample}}$ 有多远？
    *   这里**不需要**重新采样！
    *   我们只让新模型输出它的速度 $v_{\text{new}}$。
    *   然后算出新模型的确定性均值：$\mu_{\text{new}} \approx x_t - v_{\text{new}} \cdot dt$（再次强调，实际是公式 7 的插值）。
    *   距离 $d_{\text{new}} = \| x_{t-\Delta t}^{\text{sample}} - \mu_{\text{new}} \|^2$。

### 3. 结论：Ratio 到底在比什么？
$$ r_t = \frac{p_{\text{new}}}{p_{\text{old}}} = \exp\left( d_{\text{old}} - d_{\text{new}} \right) $$
*(注意：对数概率是负距离，所以距离越小概率越大)*

这个公式直观地告诉我们：
*   如果新模型的预测均值 $\mu_{\text{new}}$ 比旧模型的 $\mu_{\text{old}}$ **更靠近**那个实际采样点 $x_{t-\Delta t}^{\text{sample}}$，那么 $r_t > 1$（概率变大了）。
*   这意味着新模型“修正”了预测方向，使其更符合刚才那次（被判定为高分 Advantage）的随机探索路径。

所以你的总结非常精辟：**Rollout 用 SDE 制造随机探索，计算 Loss 时用 ODE（确定性均值 $\mu$）来评估这个探索的概率密度。** <alphaxiv-paper-citation title="Log-Prob Calculation" page="20" first="The log-probability for" last="sampling step." />

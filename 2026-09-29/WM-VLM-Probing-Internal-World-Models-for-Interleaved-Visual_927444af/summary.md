---
title: "WM-VLM-Probing-Internal-World-Models-for-Interleaved-Visual"
source: https://arxiv.org/pdf/2609.34826v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:39:15"
field: "视觉-语言模型的推理能力"
keywords: ["world model", "interleaved reasoning", "visual-spatial reasoning", "latent generation", "Mixture-of-Transformers", "mental rotation", "rectified flow"]
innovations: ["提出WM-VLM架构，为VLM集成轻量级内部世界模型分支以生成中间视觉状态辅助空间推理", "设计两阶段训练策略（先学视觉状态预测再学利用其推理），解决联合训练失效问题", "程序化构建Tetris-2D/3D数据集并提供检索-因果验证协议证明模型真正依赖生成的视觉token"]
benchmarks: ["Tetris-2D", "Tetris-3D", "VSP-Nav"]
---

# 论文速读：WM-VLM: Probing Internal World Models for Interleaved Visual-Textual Reasoning

## 一句话总结
论文提出 WM-VLM，通过为 VLM 添加轻量级内部世界模型分支并采用两阶段训练策略，使模型能够生成中间视觉状态以辅助空间推理，在 Tetris-2D/3D 心理旋转任务上相比 SFT 骨干网络最高提升 39.25 个百分点。

## 研究问题与动机
- VLM 在空间推理任务上表现薄弱，主要原因在于传统 VLM 仅通过文本进行中间推理，难以保留解决空间问题所需的视觉信息。
- 现有方法要么操纵已有视觉输入（如裁剪/高亮），要么依赖外部世界模型，未能回答 VLM 本身能否学习生成中间视觉状态以提升空间推理。
- 标准 VLM 训练仅监督文本输出，缺乏对中间视觉状态预测的直接学习目标。
- 准确生成视觉状态不等于能利用其进行推理，需要研究如何教模型使用自身生成的视觉状态。

## 核心贡献（创新点）
- 提出 WM-VLM 架构，为预训练 VLM 集成基于 light MoT 的轻量级内部世界模型分支，通过路由机制区分干净图像 token（理解分支）与噪声图像 token（生成分支）。与已有工作的本质区别：不同于外部工具或独立世界模型，将世界建模能力内化到 VLM 内部。
- 设计两阶段训练策略：Stage 1 冻结理解分支、仅训练生成分支学习预测下一视觉状态（rectified flow 损失）；Stage 2 冻结视觉编码器、联合训练语言解码器与生成分支（交叉熵损失）。与已有工作的本质区别：解决联合训练失效问题，强制模型先学会视觉世界建模再进行推理。
- 程序化构建 Tetris-2D 和 Tetris-3D 数据集，提供可验证的中间视觉状态，用于分离评估视觉状态预测能力与下游推理能力。与已有工作的本质区别：解决缺乏同时满足"需预测新视觉状态、含可验证中间状态、可分离评估"三个条件的数据集的问题。
- 通过检索实验与扰动实验证明生成的视觉 token 对推理结果具有因果贡献，而非模型走捷径。与已有工作的本质区别：直接回应当前 latent visual reasoning 工作中关于"模型是否真正依赖视觉中间态"的质疑。

## 方法详解
- **架构设计**：基于 Mixture-of-Transformers (MoT) 的 light MoT 变体，包含理解分支（完整预训练 VLM，如 Qwen2.5-VL-7B-Instruct）和浅层生成分支（默认 k=4 层，与理解分支的连续 k 层对齐）。
- **Token 路由机制**：每个 token 按模态被路由——文本和干净图像 token 进入理解分支，噪声图像 token 进入生成分支；生成分支将噪声视觉 token 转换为干净视觉 token 后，所有 token 通过全局自注意力交互。
- **交错推理过程**：将交错视觉-文本推理建模为马尔可夫过程。在每个步骤 j，策略模型 π_θ 生成心理动作 T_j，内部世界模型 p_θ^wm 在嵌入空间预测结果视觉状态 ẑ_j，推理状态更新为 H_{j+1} = H_j ⊕ (T_j, ẑ_j)，最终生成文本摘要和答案。
- **两阶段训练**：Stage 1 使用 rectified flow 损失 L_flow（α=1, β=0）训练生成分支预测下一视觉状态；Stage 2 使用交叉熵损失 L_CE（α=0, β=1）联合训练语言解码器和生成分支。总损失为 L = αL_flow + βL_CE。
- **推理时的流匹配采样**：从噪声 z^(0) ~ N(0,I) 出发，通过 K 步 Euler 积分沿预测的速度场推进至 t=1，得到估计的视觉状态 ẑ_j = z̃_j^(1)。

## 实验与结果
- **数据集**：Tetris-2D（4k 训练、400 ID/500 OOD 测试）、Tetris-3D（16k 训练、400 easy ID/400 hard ID/500 OOD），以及 VSP-Nav 迷宫导航任务。
- **基线**：Qwen2.5-VL-7B-Instruct SFT、LatentUM、Mirage、ThinkMorph（基于 BAGEL）。
- **主要结果**：WM-VLM 在 Tetris-2D-ID 达到 87.50%（比 SFT 提升 +39.25pp）、Tetris-2D-OOD 达到 75.00%（+28.80pp）；Tetris-3D-SC-ID 达到 91.00%（+19.25pp）、C-ID 达到 66.25%（+21.75pp）、OOD 达到 71.80%（+16.30pp）。
- **检索相关性分析**：视觉 token 检索 top-1 正确率与答案准确率正相关（φ 系数在 Tetris-2D-ID 上为 0.298，Tetris-3D-SC-ID 上为 0.399）。
- **消融关键发现**：将生成 token 替换为全零或随机 embedding 导致准确率骤降 50-75pp；shuffle token 影响较小（-2~10pp）；两阶段训练远优于单阶段联合训练（后者精度接近随机猜测）；Stage 2 最优 loss 比例为 α=0.1, β=0.9。

## 相关工作脉络
- **LatentUM / Mirage**：分别生成离散/连续视觉 token 进行交错推理，但 Viveiros 等人指出模型可能通过 latent bypass 走捷径而不真正依赖视觉中间态；本文通过扰动实验直接验证因果依赖。
- **ThinkMorph (BAGEL)**：生成像素级图像并使用 flow-matching 损失，在 Tetris 任务上训练崩溃（0% 精度），说明大规模 interleaved 预训练数据对像素级生成至关重要，而本文的连续 latent 路径更稳健。
- **Visual Sketchpad / Pixel Reasoner**：使用外部工具编辑输入图像作为中间视觉状态；本文不依赖外部工具，而是让 VLM 内部预测新视觉状态。
- **MindJourney / DreamPlan**：调用外部世界模型模拟未来场景；本文探索 VLM 自身内化世界建模能力，无需外挂模型。
- **LatentUM (Jin et al., 2026)**：使用统一 latent-space 模型支持跨模态推理，本文与其共同点在于连续 latent 表示，但本文强调两阶段训练和内部世界模型的必要性。

## 局限性与未来方向
- 当前训练范式在端到端联合训练时失效，必须依赖两阶段 curriculum，限制了训练的简洁性和可扩展性。
- 实验数据集为程序化生成的空间旋转任务，覆盖范围有限，未验证于更复杂的真实世界空间推理或具身任务。
- 视觉 token 数量对 ID/OOD 性能有不同影响（ID 随 token 数增加而提升，OOD 在 2×2 时最优），最优配置尚未明确。
- 未来方向包括：使用具身轨迹数据（机器人或模拟器）进行预训练，使 VLM 能从物理交互中学习内部世界模型。

## 研究启发与可借鉴点
- **两阶段 curriculum 设计**：先让模型学会"想象"（生成视觉状态），再学会"利用想象推理"，这一训练顺序对融合生成与推理能力的任务具有普适参考价值。
- **检索-因果验证方法**：通过计算生成 token 与 ground truth 的相似度并分析其对答案准确率的因果关系，为检验 interleaved visual reasoning 模型是否真正使用视觉中间态提供了可复用的评估协议。
- **light MoT 架构的 token 路由思路**：通过模态感知的路由（干净/噪声 token 分别进入不同分支）实现多能力统一模型，可在其他多模态推理任务中借鉴。
- **可验证中间状态的程序化数据构建**：Tetris 数据集的设计原则（每个样本含可验证中间状态、可分离评估预测与推理）可为空间推理 benchmark 构建提供参考。

## 关键术语表
- **WM-VLM**：带有内部世界模型分支的视觉-语言模型，能够生成中间视觉状态辅助空间推理。
- **Light MoT (Mixture-of-Transformers)**：保留完整理解分支、使用浅层生成分支的 MoT 变体，生成层与理解分支的连续层对齐。
- **Rectified Flow**：一种生成建模方法，通过学习从噪声到目标分布的速度场来生成数据，本文用于视觉 latent 的预测。
- **Interleaved Visual-Textual Reasoning**：在推理过程中交替生成文本思考和视觉状态的推理范式。
- **Tetris-2D / Tetris-3D**：程序化生成的 2D/3D 心理旋转推理数据集，包含可验证的中间视觉状态。
- **Latent Bypass**：模型通过中间监督信号绕过真正依赖视觉中间态、直接走捷径的现象。
- **Internal World Model**：内化于模型内部的、能够根据当前状态和动作预测未来视觉状态的能力。
- **OOD (Out-of-Distribution)**：测试集中包含训练数据中未见过的物体形状或数量，用于评估泛化能力。

## 可复现要素
- 数据集：Tetris-2D 和 Tetris-3D 为程序化生成，具体构建算法见附录 Algorithm 1；VSP-Nav 来自 VSP benchmark。论文未提及是否公开代码/权重。
- 代码/权重：论文未明确声明开源。
- 关键超参：Stage 1 学习率 1e-4（Constant scheduler），Stage 2 学习率 1e-5（Cosine scheduler, warmup 0.03）；batch size 1 per device，8 GPU，BF16 精度；Flow/MSE loss 权重 Stage 1 为 1.0、Stage 2 为 0；CE loss 权重 Stage 1 为 0、Stage 2 为 1.0。

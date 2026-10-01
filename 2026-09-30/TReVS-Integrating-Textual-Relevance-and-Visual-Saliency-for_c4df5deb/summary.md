---
title: "TReVS-Integrating-Textual-Relevance-and-Visual-Saliency-for"
source: https://arxiv.org/pdf/2609.37581v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 11:13:25"
field: "多模态大模型高效推理"
keywords: ["视觉语言模型", "Token剪枝", "免训练优化", "多模态推理", "两阶段剪枝", "注意力方差"]
innovations: ["提出免训练两阶段剪枝框架TReVS，在LLM前融合文本相关性与视觉显著性", "发现高方差text-to-vision注意力头对查询更敏感，用于LLM内二次剪枝", "揭示预LLM阶段文本相关性与视觉显著性的负相关性，证明互补价值"]
benchmarks: ["GQA", "TextVQA", "POPE", "MME", "MMBench", "TGIF-QA", "MSVD-QA", "MSRVTT-QA"]
---

# 论文速读：TReVS-Integrating-Textual-Relevance-and-Visual-Saliency-for-Efficient-Vision-Language-Model-Token-Pruning

## 一句话总结
TReVS 是一个免训练的视觉-语言模型两阶段视觉 token 剪枝框架，通过在 LLM 前融合文本相关性信号（与视觉显著性互补）来保留查询相关证据，并在 LLM 浅层至中层利用高方差注意力头进一步剔除无关 token，在 LLaVA-1.5-7B 上剪掉 94.4% 视觉 token 的同时保留 92.8% 原始性能。

## 研究问题与动机
- **预 LLM 剪枝的信息瓶颈**：现有两阶段方法的第一阶段仅依赖视觉编码器显著性（[CLS] attention），会不可逆地丢弃查询相关的视觉证据，导致后续文本引导阶段缺乏关键信息。
- **文本相关性与视觉显著性互补**：两者在 token 排序上呈负 Spearman 相关，单一信号无法同时覆盖"视觉突出"和"查询相关"两类证据。
- **LLM 内部剪枝注意力信号的质量差异**：对所有注意力头平均会引入噪声；高方差的 text-to-vision 注意力头对查询变化更敏感，能提供更判别性的选择信号。
- **推理成本瓶颈**：LLaVA-NeXT 每图最多 2,880 tokens，Qwen2.5-VL 最多 16,384 tokens，视觉序列长度成为 VLM 推理的主要瓶颈。

## 核心贡献（创新点）
1. **揭示了预 LLM 阶段引入文本相关性的必要性**：发现 [CLS] 显著性与文本相关性在 token 排序上负相关，表明两者选择互补——这与仅用视觉显著性做第一阶段剪枝的方法本质不同。
2. **提出基于注意力方差的选择性头筛选机制**：发现高方差的 text-to-vision 注意力头对查询更敏感，能以免训练方式筛选出判别性更强的头子集——这与 FastV/PyramidDrop 等对所有头取平均的做法有本质区别。
3. **设计了端到端免训练两阶段剪枝框架 TReVS**：Stage 1 融合 [CLS] 显著性、余弦文本相关性和 FPS 多样性，Stage 2 在高方差头上做 LLM 内二次剪枝——整体无需微调，定位区别于 LearnPruner（需可学习预测器）等需要训练的方法。
4. **在多个基准上刷新 SOTA**：在 LLaVA-1.5-7B 上以 92.8% RelAcc. 超越 VScan（91.5%）和 DUET-VLM（91.6%）；在 Video-LLaVA-7B 上超越 DUET-VLM 达 3.2%。

## 方法详解
- **Stage 1（预 LLM 剪枝）**：
  - 视觉显著性分数 $s_i^v$：取 ViT 最后一层各 head 的 [CLS]-to-patch attention 均值（公式 3）。
  - 文本相关性分数 $s_i^t$：将视觉 token 经 Projector 对齐到文本空间，计算与所有 text token 嵌入的 rectified cosine similarity（公式 4），再 RMS 聚合（公式 5）。
  - 归一化与温度缩放：分别用 MAD 鲁棒归一化，并对 $s^v$ 和 $s^t$ 施加不同温度 $\tau_v=1.4, \tau_t=1.0$（公式 6）。
  - 统一融合分数：$s_i = \max(\hat{s}_i^v, \hat{s}_i^t) + \lambda \sqrt{\hat{s}_i^v \cdot \hat{s}_i^t}$，其中 $\lambda=1.0$ 为一致性奖励项（公式 7）。
  - 选 pivot token 后，用 Farthest Point Sampling (FPS) 从剩余候选中补 $K_d$ 个多样性 token，合并得 $K_1$ 个 token 送入 LLM。

- **Stage 2（LLM 内剪枝，在浅层至中层，论文默认第 8 层）**：
  - 按 text-to-vision 注意力方差 $u_h$（公式 8，对每个 head 跨 visual-token 位置求方差，再跨 text token 平均）选取方差最大的前一半 heads 作为高方差头集合 $\mathcal{H}^*$。
  - 在每个 visual token 上，取所有 text 位置的最大 attention，再对 $\mathcal{H}^*$ 内 head 取平均，得到 token 分数 $p_i$（公式 9），保留 Top-K 个 token 继续后续层计算。

## 实验与结果
- **模型与数据集**：
  - LLaVA-1.5-7B（576 tokens）：GQA、ScienceQA、TextVQA、POPE、MME、MMBench（EN/CN）
  - LLaVA-NeXT-7B（2,880 tokens）：GQA、TextVQA、MME、MMBench
  - Video-LLaVA-7B（2,048 tokens）：TGIF-QA、MSVD-QA、MSRVTT-QA
- **主要结果（LLaVA-1.5-7B，Table 1）**：
  - 32 tokens（保留 5.6%）：TReVS RelAcc. = **92.8%**，超越 VScan（91.5%）和 DUET-VLM（91.6%）；MME = 1651。
  - 64 tokens（保留 11.1%）：RelAcc. = **96.5%**。
  - 128 tokens（保留 22.2%）：RelAcc. = **98.8%**，MME = 1842。
- **LLaVA-NeXT-7B（Table 2）**：320 tokens 时 RelAcc. 96.6%；160 tokens 时 RelAcc. 93.0%；在 GQA 上略低于 VScan（0.5~0.6%）。
- **Video-LLaVA-7B（Table 3）**：保留 136 tokens（删 93.4%），RelAcc. **99.0%**，超越 DUET-VLM 3.2%。
- **效率（POPE，RTX 4090，Table 5）**：32 tokens 时 prefill 加速 2.2×，端到端加速 1.4×，KV-cache 减少 6.5×。
- **Ablation（Table 4）**：Stage 1 加入文本相关性后 RelAcc. 从 91.8%→92.7%；再用高方差头 Stage 2 提升至 93.2%。

## 相关工作脉络
1. **FastV（ECCV 2024）**：在 LLM 最浅层基于 attention 剪枝，但所有 visual token 先进 LLM，且对所有头取平均；TReVS 先在 LLM 前做 query-aware 预处理，再用高方差头精选。
2. **SparseVLM（ICML 2025）**：文本引导 attention + rank-adaptive sparsity，主要在 LLM 内部操作；TReVS 强调在 pre-LLM 阶段就引入文本相关性以弥补信息缺口。
3. **VisionZip（CVPR 2025）**：纯视觉显著性 + 合并策略，无文本引导；TReVS 的融合分数同时利用文本相关性和视觉显著性的互补性。
4. **DUET-VLM（CVPR 2026）**：局部聚类合并 + 逐层 cross-modal 剪枝；TReVS 免训练，用方差筛选头，设计更简单。
5. **LearnPruner（ICLR 2026）**：用可学习 predictor 替代 [CLS] attention 做 pre-LLM 剪枝；TReVS 无需任何训练/微调即可使用。
6. **DivPrune（CVPR 2025）/ VScan（TMLR 2026）**：均侧重视觉侧信号；TReVS 的核心定位是打破"pre-LLM 纯视觉、in-LLM 纯文本"的两极割裂。

## 局限性与未来方向
- **高分辨率全局推理任务（如 GQA）略有劣势**：TReVS 在高_resolution 输入下 GQA 成绩落后 0.5~0.6%，作者分析认为高方差头偏向稀疏细节而丢失全局上下文。
- **固定两阶段比例**：当前 $K_r:K_d=3:1$、$K_1:K_2=3:1$ 固定比例，对不同输入复杂度缺乏自适应。
- **仅限 LLaVA 系列验证**：主要在 LLaVA-1.5/NeXT、Video-LLaVA 上评估，尚未在 Qwen2.5-VL 等新型架构上验证。
- **未探索更深的 in-LLM 剪枝层**：作者建议未来可按输入复杂度和查询需求自适应分配两阶段 token 预算。

## 研究启发与可借鉴点
1. **文本相关性先验可用于其他预压缩任务**：本文证明 pre-LLM 阶段的文本对齐打分能与视觉显著性互补（负相关），该思路可直接迁移到视频 token 压缩、多图像场景的预筛选中。
2. **注意力方差作为免训练的头筛选信号**：公式 8 提出的方差指标实现简单、无需微调即可区分查询敏感头，可复用于其他 attention-based 剪枝或压缩方法。
3. **Fusion 公式可推广**：$\max + \text{reward}$ 的融合方式（公式 7）比简单加权更优雅，能同时保留任一信号突出的 token 并在双高时给予额外奖励，值得借鉴到多信号融合场景。
4. **负相关分析作为信号互补的论证范式**：通过 Spearman 相关性分析证明两个信号互补（Figure 2c），是一种值得学习的消融论证策略。
5. **高分辨率下的全局信息保留策略**：针对 GQA 类任务的精度下降，后续可探索分层级保留全局/局部 token 的混合策略。

## 关键术语表
- **TReVS**：Training-free framework integrating Textual Relevance and Visual Saliency for efficient visual token pruning。
- **[CLS] Attention**：ViT 中 [CLS] token 对各 patch token 的 attention，本文作为视觉显著性信号。
- **Textual Relevance Score**：将视觉 token 投影到文本空间后与 text token 嵌入的 rectified cosine similarity，经 RMS 聚合得到。
- **High-Variance Attention Head**：text-to-vision attention 方差较大的 LLM head，对查询变化更敏感。
- **Farthest Point Sampling (FPS)**：在特征空间中迭代选取距离已选集合最远的点，用于补充多样性 token。
- **RelAcc.**：Relative Accuracy，各 benchmark 剪枝后性能相对未剪枝模型的比值，取各 benchmark 平均值。
- **Pre-LLM / In-LLM Pruning**：分别在 LLM 输入前和 LLM 内部进行的视觉 token 削减阶段。
- **Query-Agnostic**：不依赖文本查询的剪枝策略，本文指出现有 pre-LLM 方法的共同缺陷。

## 可复现要素
- **数据集**：GQA、ScienceQA、TextVQA、POPE、MME、MMBench、TGIF-QA、MSVD-QA、MSRVTT-QA（均为公开基准）。
- **代码/权重**：论文未提及代码是否开源。
- **关键超参**：$K_r:K_d=3:1$，$K_1:K_2=3:1$，$\tau_v=1.4$，$\tau_t=1.0$，$\lambda=1.0$，Stage 2 在 LLM 第 8 层执行，保留 top-half 高方差头。

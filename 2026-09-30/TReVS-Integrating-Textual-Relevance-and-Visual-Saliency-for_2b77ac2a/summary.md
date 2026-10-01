---
title: "TReVS-Integrating-Textual-Relevance-and-Visual-Saliency-for"
source: https://arxiv.org/pdf/2609.37581v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 11:12:44"
field: "高效视觉语言模型推理"
keywords: ["visual token pruning", "vision-language models", "efficient inference", "two-stage pruning", "textual relevance", "attention variance"]
innovations: ["将文本相关性引入预LLM剪枝阶段，与视觉显著性互补以提升查询相关证据保留率", "利用高方差注意力头作为无训练的查询敏感head选择标准，提升in-LLM剪枝判别力"]
benchmarks: ["GQA", "TextVQA", "POPE", "MME", "MMBench", "ScienceQA", "TGIF-QA", "MSVD-QA", "MSRVTT-QA"]
---

# 论文速读：TReVS-Integrating-Textual-Relevance-and-Visual-Saliency-for- Efficient-Vision-Language-Model-Token-Pruning

## 一句话总结
TReVS 是一种无训练的两阶段视觉 token 剪枝框架，通过将文本相关性引入预 LLM 剪枝阶段以保留查询相关视觉证据，并利用高方差注意力头在 LLM 内部进行查询驱动的二次剪枝；在 LLaVA-1.5-7B 上以 94.4% 的剪枝率保留了 92.8% 的原始性能，优于现有最先进方法。

## 研究问题与动机
- 现有两阶段视觉 token 剪枝方法在预 LLM 阶段仅依赖视觉编码器显著性（如 [CLS] 注意力），可能过早丢弃对文本查询关键的视觉证据，造成不可逆的信息瓶颈。
- 视觉显著性与文本相关性产生的 token 排序呈负相关（ across datasets and budgets），说明两者提供互补信息，但未在现有方法中协同利用。
- 现有 in-LLM 剪枝方法通常对所有注意力头取平均，忽略了不同 head 对文本查询的敏感度差异，导致在浅层阶段注意力分散、精度损失大。
- 预 LLM 阶段与 in-LLM 阶段的信号不匹配形成信息瓶颈：一旦查询相关 token 在 pre-LLM 阶段被丢弃，后续阶段无法恢复。

## 核心贡献（创新点）
- **发现文本相关性对预 LLM 剪枝的互补价值**：证明将文本相关性引入预 LLM 剪枝可保留查询相关视觉证据，且与视觉显著性排序呈负相关，二者应协同使用；区别于以往仅依赖视觉信号的 pre-LLM 剪枝方法。
- **提出高方差注意力头作为查询敏感度的无训练代理**：发现文本到视觉注意力方差高的 head 对查询变化更敏感，能产生更具判别性的注意力信号；区别于以往对所有 head 取平均的做法。
- **设计 TReVS 两阶段无训练剪枝框架**：第一阶段融合视觉显著性、文本相关性与多样性进行预 LLM 剪枝，第二阶段在高方差 head 指导下进行 in-LLM 二次剪枝；区别于 DUET-VLM 等仅在前一阶段使用视觉聚类的方法。
- **在多个基准上实现 SOTA 性能**：在 LLaVA-1.5-7B 上剪枝 94.4% token 保留 92.8% 性能，在 Video-LLaVA-7B 上超越 DUET-VLM 3.2%；系统性验证了两阶段协同设计的有效性。

## 方法详解
**Stage 1：查询感知的预 LLM 视觉冗余削减**

- **视觉显著性得分** $s_i^v$：使用 ViT 最后一层的 [CLS]-to-patch 注意力，跨所有 $H$ 个 head 平均（公式 2-3）。
- **文本相关性得分** $s_i^t$：将 ViT 输出通过 projector 映射到文本空间 $Z$，计算每个 visual token 与所有 text token 的修正余弦相似度 $m_{j,i} = \max(\bar{t}_j^\top \bar{z}_i, 0)$，再经 RMS 聚合（公式 4-5）。
- **归一化与融合**：分别用 MAD 鲁棒归一化后，以温度 $\tau_v=1.4, \tau_t=1.0$ 缩放，融合公式为 $s_i = \max(\hat{s}_i^v, \hat{s}_i^t) + \lambda\sqrt{\hat{s}_i^v \cdot \hat{s}_i^t}$，$\lambda=1.0$（公式 6-7）。
- **多样性补充**：从剩余候选 token 中用 Farthest Point Sampling (FPS) 选取 $K_d$ 个多样性 token，与 top-$K_r$ pivot token 合并，共 $K_1=K_r+K_d$ 个 token 输入 LLM。

**Stage 2：查询驱动的 LLM 内部视觉 token 压缩**

- **高方差 head 选择**：在剪枝层 $l^*$（默认第 8 层后），计算每个 head 的文本到视觉注意力方差 $u_h$（公式 8），选取方差最大的前一半 head 构成 $\mathcal{H}^*$。
- **二次剪枝**：对每个 visual token 计算其 Across text positions 的最大注意力，经 $\mathcal{H}^*$ 平均后得分为 $p_i$（公式 9），保留 top-$K_2$ 个 token 继续后续 LLM 层推理。默认 $K_1:K_2=3:1$。

## 实验与结果
- **模型与基准**：LLaVA-1.5-7B（576 tokens）、LLaVA-NeXT-7B（2880 tokens）、Video-LLaVA-7B（2048 tokens）；评估 GQA、ScienceQA、TextVQA、POPE、MME、MMBench（EN/CN）、TGIF-QA、MSVD-QA、MSRVTT-QA。
- **LLaVA-1.5-7B 结果**：在 128/64/32 token 预算下 RelAcc 分别为 98.8% / 96.5% / 92.8%，全面优于 FastV、SparseVLM、DivPrune、VisionZip、VScan、DUET-VLM；MME 得分在所有预算下均最佳。
- **LLaVA-NeXT-7B 结果**：在 320/160 token 预算下 RelAcc 分别为 96.6% / 93.0%，GQA 上略逊于最优 0.5-0.6%（归因于高分辨率下全局推理需求）。
- **Video-LLaVA-7B 结果**：保留 136 tokens（剪枝 93.4%）时 RelAcc 达 99.0%，超越 DUET-VLM 3.2%。
- **效率**：32 token 时 prefill 加速 2.2×，端到端加速 1.4×，KV-cache 减少 6.5×。
- **最强结果**：LLaVA-1.5-7B 在 32 token 下 RelAcc 92.8%，较第二好的 DUET-VLM（91.6%）提升 1.2%。

## 相关工作脉络
- **VisionZip (CVPR 2025)**：基于视觉编码器显著性 + 上下文合并的 pre-LLM 剪枝，未引入文本查询信号，TReVS 在预 LLM 阶段补充了文本相关性。
- **FastV (ECCV 2024)**：在 LLM 浅层（Layer 2）进行注意力剪枝，所有 visual token 先完整进入 LLM；TReVS 通过 pre-LLM 剪枝提前削减冗余并保留查询相关证据。
- **SparseVLM (ICML 2025)**：结合文本引导注意力与秩自适应稀疏性，但 pre-LLM 阶段仍为纯视觉信号；TReVS 在 pre-LLM 阶段即引入文本相关性。
- **DivPrune (CVPR 2025)**：基于多样性的 pre-LLM 剪枝，忽视查询相关性；TReVS 证明了文本相关性与视觉显著性的负相关性及其互补价值。
- **DUET-VLM (CVPR 2026)**：两阶段方法，pre-LLM 使用局部聚类合并 token，in-LLM 使用 cross-modal 注意力；TReVS 的 pre-LLM 阶段直接融合文本相关性，信息保留更充分。
- **VScan (TMLR 2026)**：重新思考 VLM 视觉 token 缩减，但未将文本相关性引入 pre-LLM 阶段；TReVS 强调 pre-LLM 阶段查询引导的重要性。

## 局限性与未来方向
- 当前采用固定比例分配 pre-LLM 与 in-LLM 阶段的 token 数量（$K_r:K_d=3:1, K_1:K_2=3:1$），未根据输入复杂度或查询难度自适应调整。
- 高方差 head 的选择在所有样本中固定为前 50%，未考虑不同查询/图像下最优 head 子集的动态变化。
- 仅在 LLaVA 系列模型上验证，对 Qwen2.5-VL、Molmo2 等其他 VLM 架构的泛化性有待检验。
- 高分辨率图像下 GQA 性能略有下降，说明当前方法在全局多对象推理场景下保留的上下文可能不足。
- 论文指出未来工作可探索基于输入复杂度和查询需求的自适应 token 分配策略。

## 研究启发与可借鉴点
- **预 LLM 阶段引入文本相关性**是提升两阶段剪枝性能的关键设计原则，可迁移至其他视觉 token 剪枝或高效 VLM 研究中。
- **注意力方差作为查询敏感度的无训练代理指标**提供了简洁的 head 选择策略，可用于其他 attention-based 剪枝或稀疏化方法中。
- **多样性补充（FPS）与重要性排序相结合的 token 选择机制**值得借鉴，可在特征选择、关键点检测等任务中复用。
- **负相关性验证**（视觉显著性与文本相关性的 Spearman 相关）为方法设计提供了坚实的实证依据，可作为后续工作的验证范式。
- 可将 TReVS 的两阶段协同思想扩展到视频理解、多图像理解等更长视觉序列场景，探索自适应阶段间 token 分配策略。

## 关键术语表
- **Vision-Language Model (VLM)**：通过投影模块将预训练 LLM 与视觉编码器连接的 multimodal 模型，支持视觉问答、推理等任务。
- **Visual token pruning**：通过移除冗余视觉 token 以降低 VLM 推理计算开销的技术。
- **Pre-LLM pruning**：在进入 LLM 之前，基于视觉编码器信号对 visual token 进行的剪枝。
- **In-LLM pruning**：在 LLM 内部通过 text-to-vision cross-attention 进行的查询感知 token 剪枝。
- **[CLS] attention**：ViT 中 [CLS] token 对 patch token 的注意力分布，常用作视觉显著性信号。
- **High-variance attention head**：文本到视觉注意力方差较高的注意力头，对查询变化更敏感，具有更强的判别能力。
- **RelAcc (Relative Accuracy)**：剪枝后模型在各基准上性能相对于未剪枝模型的比值（平均）。
- **Farthest Point Sampling (FPS)**：迭代选取与已选集合余弦距离最远的 token，用于补充多样性视觉证据。

## 可复现要素
- **代码/权重**：论文未提及是否开源。
- **数据集**：GQA、ScienceQA-IMG、TextVQA、POPE、MME、MMBench（EN/CN）、TGIF-QA、MSVD-QA、MSRVTT-QA；均为公开基准。
- **关键超参**：$K_r:K_d=3:1$，$K_1:K_2=3:1$，$\tau_v=1.4$，$\tau_t=1.0$，$\lambda=1.0$，第二阶段在第 8 个 LLM 层后执行，选取方差最大的前 50% heads。
- **模型**：LLaVA-1.5-7B、LLaVA-NeXT-7B、Video-LLaVA-7B。

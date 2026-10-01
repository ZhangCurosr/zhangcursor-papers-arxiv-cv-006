---
title: "WHEN-TEXT-MATTERS-DESIGN-PRINCIPLES-FOR-VISUAL-TOKEN-PRUNING"
source: https://arxiv.org/pdf/2609.34861v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:38:39"
field: "多模态大模型推理效率优化"
keywords: ["Visual Token Pruning", "Vision-Language Models", "Training-free Optimization", "Text-to-Visual Attention", "Inference Acceleration"]
innovations: ["发现文本引导在decoder中间层最有效，提出两阶段解耦剪枝策略DeFT", "引入reserve fraction保留额外候选并在中点用text-to-visual attention重选", "在8个benchmark和3个VLM上验证，80%/90%剪枝下平均性能恢复较基线提升11.10/16.84pp"]
benchmarks: ["TextVQA", "ChartQA", "InfoVQA", "MMMU", "AI2D", "MMStar", "NoCaps", "TextCaps"]
---

# 论文速读：WHEN-TEXT-MATTERS-DESIGN-PRINCIPLES-FOR-VISUAL-TOKEN-PRUNING

## 一句话总结
本文发现文本引导的视觉 token 选择在 decoder 中间层比输入附近更有效，据此提出训练无关的两阶段剪枝方法 DeFT：先在 vision encoder 内用视觉注意力做早期剪枝并保留额外候选，再在 decoder 中点用 text-to-visual 注意力做最终重选；在 8 个基准和 3 个 VLM 上，80%/90% 剪枝率下平均性能恢复较最强基线分别提升 11.10 和 16.84 个百分点。

## 研究问题与动机
1. **核心问题**：VLM 推理中视觉 token 数量导致计算开销巨大，training-free 剪枝需在保留关键视觉信息与降低延迟之间取得平衡。
2. **现有方法的不足**：
   - **Image-based 剪枝**（如 ZOO-Prune）仅依赖视觉重要性，会丢弃任务相关但视觉上不突出的关键区域（如图 1 中计算器品牌名）。
   - **Text-guided 剪枝**（如 SparseVLM）在 early decoder 阶段应用文本引导时，难以捕捉复杂的 text-visual 关系（如图 1 中保留了"I was the victim"的支持信息但丢失了对应数值）。
3. **关键发现**：text-to-visual attention 对答案相关区域的识别能力随 decoder 深度增加而显著增强（图 2），过早使用文本引导反而有害（图 6a）。
4. **研究动机**：设计一种两阶段策略——早期纯视觉剪枝 + 延迟到 decoder 中点的文本重选——以兼顾效率与信息保留。

## 核心贡献（创新点）
1. **发现文本引导时机的重要性**：通过 region masking 重要性分析与 question-gain 实验，证明 text-to-visual attention 在 decoder 中间层（约 2/4 深度）对识别答案相关视觉区域最有效。
2. **提出 DeFT（Deferred Text-guided selection）两阶段剪枝框架**：vision-encoder attention 用于 pre-LLM 早期剪枝并保留额外候选，text-to-visual attention 在 decoder 中点做最终重选，本质区别在于将视觉与文本信号解耦到不同阶段。
3. **引入 reserve fraction α 控制候选保留量**：在 early pruning 阶段保留 K + ⌈α(N−K)⌉ 个候选（默认 α=0.2），使非视觉显著但任务相关的 token 有机会在中点被文本重选纳入最终集合。
4. **系统性验证与实用分析**：在 3 个 VLM（Qwen3-VL-4B/8B、LLaVA-OV-1.5-8B）和 8 个 benchmark 上全面评估，并分析了候选保留率（39–45% 最终 token 来自 reserve）、延迟 trade-off（图 7）及与 progressive pruning 方法的对比（表 7）。

## 方法详解
**两阶段架构（Figure 3）**：

1. **Early vision-guided pruning with reserve**（pre-LLM）：
   - 计算 head-averaged vision-encoder attention 作为视觉重要性分数 $s^{\mathrm{img}}$（公式 5）：有 class token 的编码器用 CLS-to-visual 注意力；无 class token 的用 visual self-attention。
   - 不直接剪到目标预算 K，而是保留 $M = \min\{N, K + \lceil\alpha(N-K)\rceil\}$ 个候选（公式 6），其中 $\alpha \in [0.1, 0.3]$，默认 0.2。
   - 这些候选通过 LLM decoder 直到中点 $L = D/2$。

2. **Deferred text-guided reselection**（decoder midpoint）：
   - 在中点计算 text-to-visual attention：以 input text tokens 为 query，visual candidates 为 key/value，head-averaged 后得到分数 $s^{\mathrm{text}}$（公式 7）。
   - 从候选集 $\mathcal{C}$ 中选 Top-K 作为最终视觉 token 集合 $\mathcal{S}$（公式 8）。

**关键设计原则**：
- 视觉分数仅用于构建候选池 $\mathcal{C}$，不参与最终选择。
- 最终选择完全依赖 text-to-visual attention，避免早期文本引导的噪声。
- 解码器计算预算用 token-block count 衡量：$B_{\mathrm{vis}} = L M + (D-L) K \approx D K + \alpha L (N-K)$（公式 9），用于与各 baseline 匹配计算量。

## 实验与结果
**模型与基线**：
- 模型：Qwen3-VL-4B、Qwen3-VL-8B、LLaVA-OneVision-1.5-8B
- 基线：FastV、SparseVLM、VisPruner、ZOO-Prune、RESTORE（均 training-free）

**评测基准**（8 个 full benchmark）：
- TextVQA（场景文字理解）、ChartQA（图表推理）、InfoVQA（信息图推理）
- AI2D、MMMU、MMStar（图表与通用多模态推理）
- NoCaps、TextCaps（开放描述）

**主要结果**（Avg. Rel. = 各任务 score 相对 Dense 的平均，越高越好）：
- **Qwen3-VL-4B**：70% 剪枝 94.05%、80% 剪枝 90.86%、90% 剪枝 82.91%
- **Qwen3-VL-8B**：70% 剪枝 96.84%、80% 剪枝 94.79%、90% 剪枝 90.65%
- **LLaVA-OV-1.5-8B**：70% 剪枝 95.20%、80% 剪枝 92.77%、90% 剪枝 86.47%
- **80% 剪枝平均提升**：较最强基线（ZOO-Prune/VisPruner）提升 **11.10 pp**
- **90% 剪枝平均提升**：较最强基线提升 **16.84 pp**
- 在 TextVQA、ChartQA、InfoVQA 上提升最显著，80% 剪枝分别超基线 8.16/11.59/15.97 pp

**延迟分析**（表 5，RTX A6000，Qwen3-VL-8B）：
- 80% 剪枝：LLM prefill 121.20ms（Dense 226.66ms），端到端 282.35ms
- 90% 剪枝：LLM prefill 108.37ms，端到端 268.50ms
- 与 FastV/SparseVLM 等基线处于同一量级，未引入额外模块开销

## 相关工作脉络
1. **FastV (Chen et al., 2024)**：vision-encoder attention 预剪枝，在 decoder 前两步后直接剪至目标 token 数；本文证明其在 ChartQA/InfoVQA 上因丢弃关键小 token 而表现较差（表 2–4）。
2. **SparseVLM (Zhang et al., 2024)**：text-to-visual attention 动态剪枝；本文指出其过早使用文本引导的问题（图 6a），且在中点决策更优。
3. **ZOO-Prune (Kim et al., 2026)**：zeroth-order gradient 估计视觉重要性；擅长保留视觉显著区域但忽略任务相关细节（图 1 案例）。
4. **VisPruner (Zhang et al., 2025)**：结合 visual importance、diversity、sensitivity；在 aggressive pruning 下性能下降明显。
5. **RESTORE (Cho et al., 2026)**：引入 position/attention correction；修正剪枝畸变但整体提升有限。
6. **PyramidDrop (Xing et al., 2024) / FlowCut (Tong et al., 2026)**：progressive pruning；表 7 中与本文在 matched token-block budget 下对比，DeFT 仍大幅领先。
7. **定位差异**：本文不依赖 trained selector 或额外模块，利用已有 model 内部 attention 信号，通过**阶段解耦**（early vision-only + deferred text-guided）实现简单高效的设计。

## 局限性与未来方向
1. **α 需手动调优**：reserve fraction α 直接影响性能-延迟 trade-off，目前通过固定范围实验（图 7）确定默认值 0.2，未探索自适应机制。
2. **仅在中点决策**：固定 $L = D/2$ 为 reselection 边界，不同模型/architecture 的最优深度可能不同（表 1 显示 D/4 和 2D/4 也有一定效果）。
3. **未评估跨语言/多语言场景**：所有实验基于英文 benchmark，文本引导的泛化性在低资源语言下未验证。
4. **扩展性待验证**：仅测试了 3 个主流 VLM，对新兴架构（如 multimodal MoE）的适用性未讨论。
5. **推理加速细节**： latency 仅测量了 input-to-first-token，未包含完整 generation 的 throughput 对比。

## 研究启发与可借鉴点
1. **阶段解耦设计模式**：将"快速粗筛"与"精确重选"分离到不同处理阶段，可迁移至其他需要权衡效率与精度的模型压缩场景（如 audio token pruning、3D point cloud reduction）。
2. **attention 信号的深度分析**：通过 region masking importance 与 question-gain 实验量化 attention 的有效性随深度的变化，是一种可复用的分析范式，可用于诊断其他 pruning/reduction 方法的瓶颈。
3. **Reserve 机制**：保留额外候选并在后期决策，相比立即剪枝可显著缓解 aggressive pruning 下的性能崩溃；可作为通用"安全缓冲区"设计融入其他剪枝框架。
4. **Budget-matched 公平对比**：用 token-block count 匹配各方法的计算量（公式 9），确保比较公平性，值得在效率优化工作中推广。
5. **与团队方向结合机会**：若团队关注 VLM 推理加速，可将 DeFT 作为 strong baseline 接入现有 pipeline；若关注多模态检索/ grounding，可借鉴其"延迟文本引导"思想优化视觉证据召回策略。

## 关键术语表
**Visual Token Pruning**：在不改变模型参数的情况下，通过选择性丢弃冗余视觉 token 来降低 VLM 推理计算开销的技术。

**Training-free Method**：无需额外训练或微调即可应用的推理优化方法，直接利用预训练模型内部信号（如 attention）进行决策。

**Text-to-Visual Attention**：LLM decoder 中文本 query 对视觉 key/value 的注意力权重，反映文本提示对图像各区域的关注程度。

**Reserve Fraction (α)**：控制 early pruning 阶段额外保留候选 token 比例的超参数，默认 0.2，越大则最终选择越灵活但推理成本越高。

**Avg. Rel. (Average Relative Performance)**：各 benchmark 任务 score 除以对应 Dense 模型 score 后的平均值，用于跨任务、跨剪枝率的综合性能比较。

**Token-Block Count (B_vis)**：衡量 decoder 阶段视觉 token 处理工作量的指标，计算方式为各阶段处理的 token 数乘以其通过的 block 数之和。

**Region Masking Importance**：通过逐区域 mask 并测量对正确回答 NLL 的影响来量化视觉区域重要性的分析方法。

**Progressive Pruning**：在不同 decoder 层逐步减少视觉 token 数量的剪枝策略，与本文的两阶段一次性重选形成对比。

## 可复现要素
- **数据集**：8 个公开 benchmark（TextVQA、ChartQA、InfoVQA、AI2D、MMMU、MMStar、NoCaps、TextCaps），均为已公开数据集。
- **代码**：论文声明源代码已公开，GitHub 链接：https://github.com/kmc3661/DeFT
- **模型权重**：Qwen3-VL-4B/8B、LLaVA-OneVision-1.5-8B 均为公开模型。
- **关键超参**：
  - 剪枝率：70%、80%、90%
  - Reserve fraction α：默认 0.2，实验范围 [0.1, 0.3]
  - Reselection depth：固定为 decoder 中点 $L = D/2$
  - Vision encoder attention 计算：head-averaged，CLS-based 或 self-attention 依模型而定
- **评估设置**：FlashAttention-2（Qwen）/ SDPA（LLaVA-OV），KV caching 关闭，与 baseline 共享输入预处理和生成设置。

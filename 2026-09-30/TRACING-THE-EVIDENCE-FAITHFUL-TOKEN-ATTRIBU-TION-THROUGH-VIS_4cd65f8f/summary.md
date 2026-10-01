---
title: "TRACING-THE-EVIDENCE-FAITHFUL-TOKEN-ATTRIBU-TION-THROUGH-VIS"
source: https://arxiv.org/pdf/2609.37656v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 11:12:24"
field: "多模态大模型可解释性"
keywords: ["token attribution", "vision-language models", "multimodal interpretability", "chain-of-thought reasoning", "Shapley value calibration", "multi-hop attribution"]
innovations: ["模态感知配对归因矩阵去除同模态共享分量以提取Token特异性贡献", "闭式Katz级数聚合所有直接和间接推理路径实现完整多跳归因", "基于双人Shapley值的跨模量分数校准实现图像与文本公平联合排名"]
benchmarks: ["MMStar", "MathVista", "MMMU", "MMMU-Pro", "MathVerse", "VisualPuzzles"]
---

# 论文速读：TRACING-THE-EVIDENCE-FAITHFUL-TOKEN-ATTRIBU-TION-THROUGH-VISION-LANGUAGE-REASONING

## 一句话总结
本文提出 VTRACE，一种面向大型视觉-语言模型（LVLM）推理过程的多模态 Token 归因框架，通过聚合直接和间接推理路径、并对跨模量进行校准，解决了现有方法对视觉证据的欠表征和路径追踪不完整两大问题，在六个视觉推理基准上显著提升归因忠实度。

## 研究问题与动机
1. **视觉证据在联合排名中被低估**：现有方法将图像和文本 Token 放在一起排序时，视觉证据普遍获得较低归因分，导致关键图像区域被文本 Token 掩盖。
2. **仅追踪有限路径导致视觉贡献被遗漏**：视觉证据可通过多个中间推理 Token 传播至最终答案，现有方法只追踪直接贡献或部分跳数路径，大量间接贡献被忽略。
3. **跨模量分数不可比**：图像 Token 和文本 Token 的原始归因分数尺度不同，难以在统一维度上进行公平排序。
4. **动机源于对推理忠实性的需求**：随着 LVLM 越来越多地生成中间 CoT 推理，如何追溯推理链中各输入证据的实际贡献，对可解释性和模型优化均有重要价值。

## 核心贡献（创新点）
1. **揭示两大多模归因挑战**：通过实证分析系统刻画了视觉证据在联合排名中被欠表征、以及在多跳推理路径中被低估的现象，为后续方法设计提供明确动机。
2. **提出模态感知配对归因矩阵**：通过从各模态内减去同模态平均输出值来提取 Token 特异性贡献，避免共享成分对跨模量比较的干扰。
3. **闭式聚合所有直接和间接推理路径**：将配对归因矩阵进行 Katz 形式的无穷级数求和（闭式解 $(I - \gamma \hat{W})^{-1} - I$），一次性捕获全部多跳贡献，而非仅追踪最高单一路径。
4. **基于 Shapley 值的跨模量校准**：利用两个玩家精确 Shapley 值估计图像和文本各自对响应似然的贡献比例，按比例重缩放图像 Token 分数，使跨模量排名公平可比。
5. **归因引导的后训练验证**：将 VTRACE 归因分数用于 GRPO 强化学习中的 credit assignment，成功提升 Qwen3-VL-4B 在多个基准上的平均性能，表明该方法可作为学习信号而非仅用于解释。

## 方法详解
VTRACE 分为三个阶段：

**阶段一：模态感知配对归因（Modality-Aware Pairwise Attribution）**
- 对 Transformer 第 $\ell$ 层中接收 Token $x_j$ 的注意力更新：$\Delta_j^{(\ell)} = \sum_{h=1}^{H} \sum_{i \le j} \alpha_{ji}^{(\ell,h)} f_i^{(\ell,h)}$
- 定义模态感知写入：$\tilde{f}_i^{(\ell,h)} = f_i^{(\ell,h)} - \frac{1}{|\mathcal{M}(i)|} \sum_{r \in \mathcal{M}(i)} f_r^{(\ell,h)}$，其中 $\mathcal{M}(i)$ 为与 $x_i$ 同模态的所有输入 Token 集合。这一步去除同模态共享平均分量，突出 Token 之间的区分性贡献。
- 配对归因权重：$W_{ij} = \frac{1}{L} \sum_{\ell=1}^{L} \frac{[\sum_h \alpha_{ji}^{(\ell,h)} \langle \tilde{f}_i^{(\ell,h)}, \Delta_j^{(\ell)} \rangle]_+}{\|\Delta_j^{(\ell)}\|_2}$，对 $i<j$；$W$ 为严格上三角矩阵。

**阶段二：组合式多跳归因（Compositional Multi-Hop Attribution）**
- 对归因矩阵做归一化：$\hat{W} = W / (\max_{i,j} W_{ij} + \epsilon)$
- 通过 Katz 形式聚合所有路径：$R = \sum_{\tau=1}^{T-1} \gamma^\tau \hat{W}^\tau = (I - \gamma \hat{W})^{-1} - I$，其中 $\gamma=1$ 为默认衰减系数。由于 $\hat{W}$ 严格上三角，级数在 $T-1$ 项后截断，可精确闭式求解。
- 每个 Token 的入流和出流：$\text{In}(x_r) = \sum_{i<r} R_{ir}$，$\text{Out}_{\mathcal{A}}(x_r) = \sum_{j \in \mathcal{A}} R_{rj}$，最终未校准分数：$u_r = (1 + \text{In}(x_r)) \cdot \text{Out}_{\mathcal{A}}(x_r)$。

**阶段三：跨模量校准（Cross-Modal Calibration）**
- 通过 teacher-forced log-probability 下降量测量各模态贡献：$D_S = \log p_\theta(y|X) - \log p_\theta(y|\text{pad}(S))$
- 用精确双人 Shapley 值分离交互贡献：$\phi_\mathcal{I} = \frac{1}{2}(D_\mathcal{I} + D_{\mathcal{I}\cup\mathcal{T}} - D_\mathcal{T})$，$\phi_\mathcal{T} = \frac{1}{2}(D_\mathcal{T} + D_{\mathcal{I}\cup\mathcal{T}} - D_\mathcal{I})$
- 图像份额 $s = \phi_\mathcal{I}/(\phi_\mathcal{I} + \phi_\mathcal{T})$，重缩放因子 $\lambda = \frac{s}{1-s} \cdot \frac{\sum_{j \in \mathcal{T}} u_j}{\sum_{i \in \mathcal{I}} u_i}$，最终 VTRACE 分数：图像 Token 乘以 $\lambda$，文本 Token 保持不变，保持各模态内部排序不变。

## 实验与结果
**数据集与基线**：六个视觉推理基准（MMStar、MathVista、MMMU、MMMU-Pro、MathVerse、VisualPuzzles）；七个基线方法（ReAGent、HETA、FlowTracer、IFR、Attn Rollout、AttnLRP、FlashTrace）。模型使用 Qwen3-VL-8B（主要）、Qwen3-VL-4B、InternVL3.5-8B。

**评估指标**：RISE 和 MAS 的插入（Insertion↑）和删除（Deletion↓）AUC。

**主要结果（Qwen3-VL-8B Joint 设置）**：
- VTRACE 在所有六个基准上均超越全部七个基线。
- 相比最强基线，VTRACE 平均 RISE 插入提升 **+7.1%**，删除降低 **-18.0%**。
- MAS 插入提升 **18.4%**，删除降低 **29.6%**。
- 在 Image 设置下，平均 RISE 插入提升 **+6.8%**，删除降低 **-9.3%**。
- 校正效应显著：去除校准模块后 Joint 设置下降最明显（Ins 从 0.610→0.602，Del 从 0.187→0.200）；去除多跳模块影响最大（Ins 从 0.610→0.470，下降约 23%）。

**泛化与鲁棒性**：在 Qwen3-VL-4B 和 InternVL3.5-8B 上一致最优；对正确和错误预测均有效。

**效率**：归因矩阵仅构建一次，后续可复用；3000 tokens 时显存 31 GB（对比 AttnLRP 的 87 GB）；从单目标扩展到全序列仅需 0.02s 额外开销。

**归因引导学习**：将 VTRACE 分数用于 GRPO 的 credit assignment（Top 40% Token 权重 1.5），Qwen3-VL-4B 在 8 个基准上平均性能从 63.99 提升至 **65.27**。

## 相关工作脉络
1. **Attention Rollout / ALTI**：通过层间注意力矩阵复合来追踪信息流，但仅衡量直接注意力传播，不处理多跳推理路径，且未考虑跨模量公平性。
2. **IFR（Information Flow Routes）**：构建 Token 级计算图追踪信息流，但仅覆盖单跳直接贡献，对分布于中间推理 Token 的视觉证据追踪不完整。
3. **FlashTrace**：递归追踪归因通过生成 Token 的传播，但仅沿高归因路径有限跳数传播，遗漏大量弱路径贡献；VTRACE 闭式求和所有路径以弥补此不足。
4. **FlowTracer**：构建注意力流网络并施加守恒约束，可用于 RL 反馈；VTRACE 在此基础上进一步解决跨模量分数校准和所有路径聚合问题。
5. **HETA**：基于 Hessian 灵敏度的一阶+二阶梯度归因，未考虑多模态归因的不平衡和推理路径聚合。
6. **AttnLRP**：将层间相关性传播扩展到 Transformer，但计算成本随序列长度线性增长，且未处理跨模量可比性问题。

## 局限性与未来方向
1. **当前仅评估单图视觉推理**，未涵盖多图推理、长上下文理解、Agent 交互等更广泛的 LVLM 应用场景。
2. **Shapley 校准需要额外前向传播**（3次掩码评估），虽然开销可控但影响了在极长序列上的可扩展性。
3. **归因引导学习的探索较初步**，仅验证了 GRPO 设定下的可行性，未深入探索更广泛的训练范式整合。
4. **硬推理场景下的归因边界情况**：如图17所示，在较难样本中 VTRACE 仍会将部分归因分配至背景区域，说明其效果受限于生成推理轨迹的质量。
5. 未来方向包括扩展至多图/长上下文、将归因更深入地整合到模型训练中、以及探索其他强化学习算法中的归因信号利用。

## 研究启发与可借鉴点
1. **模态内去中心化特征提取思路可迁移**：通过减去同模态均值来提取 Token 特异性贡献的方法，可推广到其他多模态（如语音+文本、视频+文本）归因任务，解决异质模量间的尺度不平衡问题。
2. **闭式多跳聚合替代递归追踪**：Katz 形式的无穷级数闭式解避免了逐跳递归计算，计算复杂度从 $O(T^2 \cdot \text{hops})$ 降至 $O(T^3)$（矩阵求逆），但得益于严格上三角结构实际更快，此设计模式可用于其他需要路径聚合的可解释性方法。
3. **归因分数驱动信用分配的强化学习思路值得借鉴**：将归因作为 credit assignment 信号引入 GRPO（而非传统 reward shaping），为模型可解释性与性能优化建立了直接桥梁，启发团队可将归因框架用于自身的 post-training 实验。
4. **RISE/MAS 联合评估协议**：同时报告插入和删除、RISE 和 MAS 四种 AUC 指标，全面衡量排序忠实度和幅度对齐性，可作为团队未来评估归因方法的标准化协议参考。

## 关键术语表
**Token Attribution**：将模型生成输出中每个 Token 的贡献归因于输入序列中的上游 Token，以揭示模型决策所依赖的证据来源。

**RISE（Random Input Sampling for Explanation）**：通过按归因分数排序逐步删除或恢复输入 Token，测量模型输出概率变化曲线下的面积来评估归因忠实度的扰动指标。

**MAS（Magnitude Aligned Scoring）**：在 RISE 基础上进一步衡量归因分数幅度与实际响应变化的对齐程度，同时关注排序和数值两部分忠实性。

**Compositional Multi-Hop（CMH）Attribution**：通过矩阵幂级数聚合从源 Token 到目标 Token 的所有直接和间接路径贡献，而非仅追踪最短或最高权重路径。

**Cross-Modal Calibration**：利用 Shapley 值估计各模态对模型输出的边际贡献比例，按比例重缩放各模态内部归因分数，实现跨模态公平排名。

**Modality-Aware Pairwise Attribution**：在计算 Token 对之间归因前，先减去该 Token 所属模态的平均贡献分量，从而提取 Token 特异性而非模态共享的贡献。

**Teacher-Forced Log-Probability**：在评估输入扰动影响时，使用原始生成序列作为目标，计算模型在该序列上的对数似然变化，避免生成分布偏移的干扰。

**GRPO（Group Relative Policy Optimization）**：一种无需 critic 网络的强化学习策略优化算法，VTRACE 在此框架中利用归因分数为 Top 40% 响应 Token 赋予更高权重以指导训练。

## 可复现要素
- **数据集**：MMStar、MathVista、MMMU、MMMU-Pro、MathVerse、VisualPuzzles 均为公开基准；HallusionBench、RealWorldQA、MathVision 亦公开可用。
- **代码**：论文提供了项目页面 https://vtrace-attribution.github.io/，但未在正文中明确声明 GitHub 仓库链接。
- **模型权重**：使用公开的 Qwen3-VL（4B/8B）和 InternVL3.5（2B/8B）权重。
- **关键超参**：Katz 衰减系数 $\gamma = 1.0$（默认）；训练时 rollout 数 $n=8$、温度 0.7、top-p 0.95；归因引导学习中 Top 40% Token 权重 1.5；推理时最大生成长度 2048 tokens。
- **硬件**：归因实验单卡 NVIDIA RTX PRO 6000 Blackwell 96GB；训练实验使用 AMD Instinct MI355X。

---
title: "TRACING-THE-EVIDENCE-FAITHFUL-TOKEN-ATTRIBU-TION-THROUGH-VIS"
source: https://arxiv.org/pdf/2609.37656v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 11:11:59"
field: "多模态大模型可解释性"
keywords: ["token attribution", "vision-language model", "multimodal interpretability", "chain-of-thought reasoning", "Shapley value calibration", "information flow tracing"]
innovations: ["提出VTRACE框架，结合模态中心化成对归因与Katz闭合形式多跳路径聚合", "引入跨模态Shapley校准实现图像与文本归因分数的公平统一排名", "探索归因信号用于GRPO后训练，将Qwen3-VL-4B平均性能提升1.28"]
benchmarks: ["MMStar", "MathVista", "MMMU", "MMMU-Pro", "MathVerse", "VisualPuzzles"]
---

# 论文速读：TRACING-THE-EVIDENCE-FAITHFUL-TOKEN-ATTRIBU-TION-THROUGH-VIS

## 一句话总结
论文提出 **VTRACE**，一个多模态 token 归因框架，通过模态感知的成对归因、所有直接/间接推理路径的闭合形式聚合，以及跨模态 Shapley 校准，解决 LVLM 中视觉证据被文本低估、且通过多跳推理传播的贡献被遗漏的问题，在六个视觉推理基准上显著优于七种现有基线。

## 研究问题与动机
1. **视觉证据在联合排名中被系统性低估**：在 LVLM 中，图像 token 与文本 token 一起参与归因排序时，视觉证据往往得分低于文本（如选项字母、标点），导致关键图像区域被排在次要位置。
2. **仅追踪部分推理路径会遗漏视觉贡献**：视觉证据可能通过多个中间推理 token 传播到最终答案（如图中的"帽徽→wearing→badge"路径），现有方法只追踪最高归因路径或单跳直接路径，造成重要视觉来源被严重低估。
3. **现有方法主要为纯文本 LM 设计**：将文本归因方法直接扩展到多模态场景时，由于跨模态注意力机制和表征差异，无法公平比较图像和文本 token 的贡献量级。
4. **多模态 reasoning 的归因需求**：LVLM 越来越多地生成中间推理链（CoT），需要能追踪从输入证据经过多跳推理到最终答案的完整信息流。

## 核心贡献（创新点）
1. **首次实证刻画多模态 token 归因的两个关键缺陷**：在六个基准上系统揭示视觉证据在联合排名中的系统性低估，以及多跳推理路径对视觉贡献被截断的影响。
2. **提出 VTRACE 多模态归因框架**：通过模态中心化的成对归因矩阵隔离 token 特异性贡献，再用 Katz 公式在闭合形式下聚合所有直接和间接路径，避免了递归或采样带来的误差。
3. **引入跨模态 Shapley 校准**：通过遮蔽实验估计图像和文本对响应似然的边际贡献，用两人合作博弈的 Shapley 值将两种模态的归因分数统一缩放，使跨模态排名具有可比性。
4. **将归因信号用于 RL 后训练**：探索性地使用 VTRACE 归因作为 GRPO 中响应 token 的信用分配信号，将平均性能从 63.99 提升至 65.27，证明归因不仅是解释工具，也可作为学习信号。

## 方法详解
VTRACE 分为三个阶段：

**1. 模态感知的成对 token 归因矩阵 W**
- 对每一层、每一注意力头，计算 source token $x_i$ 的输出投影值 $f_i^{(\ell,h)}$，减去同模态（图像或文本）均值，得到 centered write $\tilde{f}_i^{(\ell,h)}$，以去除模态内共享部分，突出 token 的差异化贡献。
- 成对归因由 source-centered write 与 receiver 位置更新方向的内积给出：$W_{ij} = \frac{1}{L}\sum_{\ell}\frac{[\sum_h \alpha_{ji}^{(\ell,h)}\langle\tilde{f}_i^{(\ell,h)},\Delta_j^{(\ell)}\rangle]_+}{\|\Delta_j^{(\ell)}\|_2}$，$W$ 为严格上三角矩阵。

**2. 组合多跳归因（CMH）**
- 对归一化矩阵 $\widehat{W}$，用 Katz 公式聚合所有长度的路径：$R = \sum_{\tau=1}^{T-1}\gamma^\tau\widehat{W}^\tau = (\mathbf{I}-\gamma\widehat{W})^{-1}-\mathbf{I}$（闭合形式，无需递归）。
- 每个 token 的得分由入流（incoming）和出流（outgoing）组合：$u_r = (1+\text{In}(x_r))\cdot\text{Out}_{\mathcal{A}}(x_r)$。

**3. 跨模态 Shapley 校准**
- 通过 teacher-forced log-prob 的下降量度量各模态对输出的贡献：$D_\mathcal{S} = \log p(y|X) - \log p(y|\text{pad}(\mathcal{S}))$。
- 用两人 Shapley 值分解图文交互贡献：$\phi_\mathcal{I} = \frac{1}{2}(D_\mathcal{I} + D_{\mathcal{I}\cup\mathcal{T}} - D_\mathcal{T})$，$\phi_\mathcal{T}$ 对称。
- 图像得分缩放因子：$\lambda = \frac{s}{1-s}\cdot\frac{\sum u_\mathcal{T}}{\sum u_\mathcal{I}}$，其中 $s=\phi_\mathcal{I}/(\phi_\mathcal{I}+\phi_\mathcal{T})$，保持模态内排序不变。

## 实验与结果
- **数据集**：MMStar、MathVista、MMMU、MMMU-pro、MathVerse、VisualPuzzles（六个视觉推理基准，覆盖感知、数学、知识推理）。
- **模型**：Qwen3-VL-8B（主要）、Qwen3-VL-4B、InternVL3.5-2B、InternVL3.5-8B。
- **基线**：ReAGent、HETA、FlowTracer、IFR、Attn Rollout、AttnLRP、FlashTrace（七种）。
- **评估指标**：RISE 和 MAS 的插入（↑）和删除（↓）AUC，Image-only 与 Joint 两种设置。
- **核心结果**：
  - **Joint 设置**：VTRACE 平均 RISE 插入 AUC 提升 **7.1%**，删除 AUC 降低 **18.0%**，在所有六个基准上均取得最佳。
  - **Image 设置**：平均插入提升 **6.8%**，删除降低 **9.3%**。
  - 在正确/错误预测样本上均持续优于最强基线，表明归因质量不依赖答案正确性。
  - 归因引导的 GRPO 后训练（Qwen3-VL-4B）：平均性能从 **63.99 → 65.27**（提升 **+1.28**）。
- **效率**：在 ~3000 token 序列上仅用 31 GB 显存，速度优于 AttnLRP（87 GB）等基线。

## 相关工作脉络
1. **Attention Rollout / ALTI**（Abnar & Zuidema, 2020; Ferrando et al., 2022b）：层间注意力聚合，最早期的 token-to-token 归因，未考虑生成推理过程中的多跳传播。
2. **AttnLRP**（Achtibat et al., 2024）：将 LRP 扩展到 Transformer，逐层传播相关性，但未解决跨模态公平排序问题。
3. **IFR**（Ferrando & Voita, 2024）：构建 token 状态图追踪信息流，但仅沿单条最高流路径回溯，容易遗漏分散证据。
4. **FlashTrace**（Pan et al., 2026）：递归追踪通过生成推理 token 的归因，但只在有限步数内传播，路径覆盖不完整。
5. **FlowTracer**（Dong et al., 2026）：构建守恒流图并用流得分指导 RL，关注的是流守恒约束而非全路径聚合。
6. **HETA**（Pramanik et al., 2026）：结合 Hessian 二阶敏感度，但对多模态场景的跨模态校准机制缺乏。
7. **GLIMPSE**（Shen, 2025）、**LVLM-Interpret**（Stan et al., 2024）：多模态归因方法，但主要聚焦响应级别聚合而非 token 级别的完整多跳追溯。

## 局限性与未来方向
1. **当前评估局限于单图视觉推理**：尚未扩展到多图推理、长上下文多模态理解、以及 agent 交互等更广泛场景。
2. **跨模态校准依赖 teacher-forced 遮蔽实验**：需要额外的前向计算（三步遮蔽），虽然开销可控，但在超长序列下可能成为瓶颈。
3. **归因引导的 RL 训练仅为初步探索**：仅使用了 top 40% token 加权，未深入探索更细粒度的信用分配策略或与其他 RL 算法的结合。
4. **在少数难例中（如 Figure 17）图像归因仍不够聚焦**：当视觉证据未在推理链中充分反映时，归因质量会有所下降。

## 研究启发与可借鉴点
1. **模态内中心化（modality centering）**是一种轻量且有效的去偏技巧：通过减去同模态均值来隔离 token 特异性，可在其他多模态归因任务中复用。
2. **Katz 闭合形式聚合多跳路径**避免了递归/采样的近似误差，且可并行计算，适用于任何基于注意力图的信息传播建模任务。
3. **Shapley 值用于跨模态校准**的思路清晰简洁，只需三步遮蔽前向即可估计模态贡献比，可推广到其他多模态模型的公平比较场景。
4. **归因信号作为 RL 信用分配**是一个新颖且有潜力的方向：VTRACE 的初步实验证明 attribution-guided GRPO 可提升模型性能，值得进一步探索与 PPO/RLHF 等方法的结合。

## 关键术语表
**VTRACE**：本文提出的多模态 token 归因框架，结合模态中心化、多跳路径聚合与跨模态校准。
**Modality-Aware Pairwise Attribution**：通过减去同模态均值来量化每个 token 相对于同类 token 的特异性贡献。
**Compositional Multi-Hop (CMH) Attribution**：利用 Katz 公式在闭合形式下聚合所有直接和间接推理路径的归因方法。
**Cross-Modal Calibration**：使用 Shapley 值将图像和文本的归因分数缩放到同一量级，实现公平跨模态排名。
**RISE Insertion/Deletion AUC**：按归因分数逐步恢复/遮蔽 token 时测量响应概率恢复/下降曲线下面积，越高/越低表示归因越忠实。
**Teacher-Forced Log-Probability Masking**：用 pad token 替换特定 token 集合，测量模型对原始生成 trace 的对数概率下降量作为模态贡献度量。
**Attribution-Guided GRPO**：将归因分数作为信用信号，在 Group Relative Policy Optimization 中对高贡献 token 赋予更大奖励权重。
**Attention Path Aggregation**：通过矩阵幂次 $(\widehat{W}^\tau)$ 累加所有长度为 $\tau$ 的信息传播路径的归因。

## 可复现要素
- **代码/权重**：项目页面 https://vtrace-attribution.github.io/（论文未明确声明 GitHub 仓库，但提供了 project page）
- **模型**：Qwen3-VL-8B/4B、InternVL3.5-8B（均为公开权重）
- **数据集**：MMStar、MathVista、MMMU、MMMU-pro、MathVerse、VisualPuzzles（均为公开 benchmark）
- **关键超参**：decay rate $\gamma=1.0$（默认）；GRPO 实验中 rollout 数 $n=8$，temperature=0.7，top-p=0.95，top 40% token 权重 1.5，训练 2 epochs
- **硬件**：单卡 NVIDIA RTX PRO 6000 Blackwell (96GB)，GRPO 用 AMD Instinct MI355X

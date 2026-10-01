---
title: "Similar-Choices-Different-Attention-Cross-Modal-Associations"
source: https://arxiv.org/pdf/2609.36475v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 16:46:59"
field: "多模态认知对齐评估"
keywords: ["cross-modal association", "bouba-kiki", "vision-language model", "eye-tracking", "attention alignment", "sound symbolism"]
innovations: ["选择与注意力解耦：证明VLM选择对齐不意味着空间注意力对齐", "中心偏差基线优于VLM saliency预测人类眼动", "gaze监督提升空间相关但不可迁移至选择性能"]
benchmarks: ["人类- VLM 匹配刺激对比（8,208 trials, 53 participants）", "Intended-category agreement (ICA)", "Spearman gaze-saliency correlation (ρ)", "Human-choice agreement"]
---

# 论文速读：Similar Choices, Different Attention: Cross-Modal Associations in Humans and Vision–Language Models

## 一句话总结
本文在相同刺激条件下直接比较了人类与视觉-语言模型（VLMs）的跨模态关联（如 bouba-kiki 效应），发现大模型虽能在选择上与人类对齐，但其注意力图与人类眼动的匹配度不及简单的"中心偏差"基线；微调小模型可使其选择达到人类水平，但注意力仍无法对齐人类注视模式。

## 研究问题与动机
- **核心问题**：VLMs 是否不仅在"选择"上与人类一致，而且在"注视哪里"（空间注意力）上与人类一致？
- **现有方法的不足**：
  - 先前研究对 VLMs 中是否存在跨模态关联结论矛盾，且人类与模型通常在**不同刺激或任务**上评估，缺乏直接对比。
  - 仅比较选择一致性（choice alignment）不足以判断模型是否真正捕捉了与人类相同的跨模态处理机制——"相似的选择"可能源于"不同的注意力"。
  - 缺乏在匹配条件下同时评估选择与空间注意力的研究。
- **研究动机**：bouba-kiki 效应是研究跨模态对应的标准测试床，若 VLMs 在新造伪词上能与人类对齐，说明其表征捕捉了词形与感知意义之间的非任意关联。

## 核心贡献（创新点）
1. **构建了人类与 VLMs 在相同刺激下的直接对比数据集**：收集了 53 名参与者的眼动与选择数据（8,208 trials），包含伪词与图像对，并公开 Release。
2. **揭示了"选择对齐"与"空间对齐"的解耦**：证明即使模型选择与人类高度一致（微调后约 73% 匹配人类实际选择），其注意力图与人类眼动的 Spearman 相关仍低于简单的高斯中心偏差基线。
3. **验证了 gaze 监督的局限性**：添加人类眼动作为注意力训练目标可将空间相关性从中心基线以下提升至 0.67–0.73，但该提升可由固定平均注视图复现，且不带来选择性能的显著改善。
4. **提出选择与注意力应联合评估的评估范式**：论证单一的选择一致性不足以证明人类对齐的跨模态处理，必须同时考察空间注意力模式。

## 方法详解
- **刺激设计**：
  - 语言刺激：4 个经典伪词（bouba/kiki/takete/maluma）、20 个英语 round-sharp 形容词、162 个基于语音学规律构建的两音节伪词（圆唇元音 + 浊辅音 → 圆形；不圆唇元音 + 清塞音 → 尖角）。
  - 视觉刺激：184 对 round-sharp 图像对，包括线描最小对（75对）、Gemini 生成的抽象图（92对）、前人文 献刺激（17对），每对准备左右两种摆放方向。
- **人类实验**：53 名参与者，Tobii Pro Fusion 眼动仪记录，每人完成三轮实验，共 8,208 次有效 trial；要求参与者朗读标签后口头选择 A（左）或 B（右）。
- **VLM 评估协议**：
  - 评估 6 个 embedding 模型（CLIP、SigLIP、SigLIP2）和 6 个 generative VLM（Gemma4、Qwen3-VL、Molmo2）。
  - Embedding 模型：计算图像-文本 cosine 相似度，取十种 caption 模板的平均，选分数高的图像。
  - Generative 模型：使用强制选择 prompt，从决策 token 的注意力图读取空间分布（取最后半层 self-attention heads 的平均）。
  - 空间对齐度量：对每张图像内部计算模型 saliency 与参与者眼动的 **Spearman ρ**，再平均两张图。
- **微调方法**：
  - 使用 LoRA 适配器（rank=16, scaling=32, dropout=0.05）在语言解码器上微调。
  - 损失函数：$\mathcal{L}_t = \text{CE}(y_t) + \lambda \frac{\sum_{i \in \{A,B\}} m_{t,i} \text{KL}[H_{t,i} \| A_{t,i}]}{\sum_i m_{t,i}}$
  - 训练条件：Choice-only（λ=0）、+Gaze（λ=1，使用 trial 特定的眼动图）、+AvgGaze（λ=1，使用固定平均眼动图）、+Gaze+Side（联合监督图像间分配）。
  - 在 Train 集（1,932 trials）上训练，Val 集选择 checkpoint，Test 集评估未见过的词和图像对。

## 实验与结果
- **数据集**：8,208 次有效 trial（53 名参与者），含 7,141 次伪词、910 次形容词、157 次经典词；Train/Val/Test 分割为 1,932/721/1,030 次伪词 trial。
- **Zero-shot 选择对齐**：
  - Embedding 模型在形容词上表现良好（ICA 0.75–0.79），但在伪词上接近随机；SigLIP-so400m 是唯一有明显信号的（ICA=0.57）。
  - 大参数 generative 模型在伪词上表现最佳：Gemma4-26B（ICA=0.62, ρ=0.58）、Qwen3-VL-30B（ICA=0.57, ρ=0.57）。
  - 小模型（Gemma4-E4B、Qwen3-VL-8B、Molmo2-4B）接近随机且严重侧向锁定（side-locked）。
- **Zero-shot 空间对齐**：
  - 所有模型的 saliency-gaze 相关均**低于**中心偏差基线（Center ρ ≈ 0.60–0.74）。
  - CLIP ViT-B/32 最高（ρ=0.59），但仍低于其 Center（0.74）；generative 模型中 Gemma4-E4B/26B 部分追踪眼动（ρ=0.34/0.27），其余接近零或为负。
- **微调后选择对齐**：
  - Choice-only 微调后，三个小模型的人类选择匹配从 ~50% 提升至约 **73%**，达到人类多数投票参考（72.86%）水平，且在未见词和图像对上传播良好。
- **微调后空间对齐**：
  - Choice-only 的空间相关仅 0.31/0.07/0.06，仍低于中心基线。
  - +Gaze 将空间相关提升至 **0.73/0.73/0.67**，但 +AvgGaze（固定平均图）达到几乎相同的 0.73/0.73/0.68，说明高相关性主要来自共享的空间先验而非 trial 特定的学习。
  - 添加 gaze 监督对选择性能无显著改善。
- **最强结果**：Gemma4-26B zero-shot 在伪词上取得最高人类选择匹配（Maj.=0.73, ρ=0.58）；微调后 +Gaze 条件达到最高空间相关（Gemma4-E4B ρ=0.733）。

## 相关工作脉络
- **Alper & Averbuch-Elor (2023)**：声称在 CLIP 嵌入空间中发现了 round-sharp 方向，伪词与图像投影到同一侧；本文指出其仅使用形容词作为语义基线，且未进行与人类的直接对比。
- **Kouwenhoven et al. (2025)**：报告 CLIP 中不存在跨模态关联，添加了多种 prompt 和 Grad-CAM 分析；本文与其一致认为 CLIP 在伪词上信号微弱，但强调需在匹配条件下对比人类。
- **Pani & Yang (2025) / Sood et al. (2020)**：前人研究表明 gaze 正则化可改善注意力对齐；本文验证了这一发现，但指出该改善不迁移至选择性能，且可由固定平均图复现。
- **Iida & Funakura (2024)**：在日本 VLMs 中未能复现 Alper 的结果，表明先前发现可能依赖英语设置。
- **Shimojo et al. (2003) / Krajbich et al. (2010)**：眼动逐渐累积于最终选择的图像（77–80% 一致），这使 per-trial 眼动成为有效的监督信号。
- **Raghu et al. (2021)**：视觉 Transformer 从早期层就开始全局整合信息，这解释了为何模型可能通过 holistic 表征而非局部形状特征做出与人类一致的选择。

## 局限性与未来方向
- 微调仅使用三个模型、一个优化 seed 和一个 gaze-loss weight，泛化性待验证。
- 注意力图和 Grad-CAM 是层相关的关联读取，非因果解释。
- 书面标签混淆了语音学与正字法贡献（参与者为英语母语者）。
- 参与者来自同一机构，共享 across splits，可能影响独立性假设。
- 未来方向：探索更多模型架构、多个 seed、不同语言的 sound symbolism、以及因果性的注意力干预实验。

## 研究启发与可借鉴点
1. **选择与注意力应联合评估**：在评估 VLM 的认知对齐时，仅看选择一致性可能产生误导，需同步考察空间注意力模式。
2. **中心偏差基线值得作为标准 baseline**：简单的高斯中心偏置在预测人类眼动上优于复杂模型的 saliency，提示模型可能依赖全局特征而非局部诊断性特征。
3. **固定平均 gaze 图的对照实验设计**：用 trial 特定的 gaze 与固定平均 gaze 对比，可有效分离"共享空间先验"与"trial 特定学习"的贡献。
4. **可迁移至其他认知评测场景**：该方法论可用于评估 VLM 在其他跨模态对应（如大小-音高、 texture-sound）中的人类对齐程度。
5. **微调范式可复用**：LoRA 适配器 + KL  gaze loss 的组合可用于其他需要同时优化选择与注意力的多模态任务。

## 关键术语表
- **Cross-modal association（跨模态关联）**：不同感官模态之间系统的特征配对，如声音与形状的对应。
- **Bouba-kiki effect（布巴-基基效应）**：人们倾向于将"bouba"与圆形形状、"kiki"与尖角形状配对的声音-形状对应现象。
- **Intended-category agreement (ICA)**：响应与标签预设的 round/sharp 类别一致的比例。
- **Spatial alignment（空间对齐）**：模型 saliency 图与人类眼动在图像内部的空间分布相似性。
- **Choice alignment（选择对齐）**：模型选择与人类选择在 trial 级别的一致性。
- **Grad-CAM**：基于梯度的可视化方法，用于生成嵌入模型的注意力图。
- **Decision-token attention**：从生成模型中预测答案 token 的位置读取的注意力图。
- **Side-locked**：模型几乎总是选择同一侧（左或右）的图像，而非基于内容做出选择。

## 可复现要素
- **数据集**：已声明公开，包含刺激材料和眼动数据。
- **代码/权重**：代码将在发表后提供；使用的 VLM 权重为开源模型（CLIP、SigLIP、Gemma4、Qwen3-VL、Molmo2）。
- **关键超参**：LoRA rank=16, scaling=32, dropout=0.05；AdamW lr=2×10⁻⁵, 5% warmup, cosine decay, batch=8, bf16；最多 6 轮 epoch, patience=2；gaze loss weight λ=0 或 1；眼动图高斯平滑 σ=40 screen pixels；中心基线 σ=0.125（归一化坐标）。

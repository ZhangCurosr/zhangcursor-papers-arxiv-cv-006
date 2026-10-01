---
title: "Similar-Choices-Different-Attention-Cross-Modal-Associations"
source: https://arxiv.org/pdf/2609.36475v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 16:46:58"
field: "多模态认知对齐评估"
keywords: ["cross-modal association", "bouba-kiki effect", "vision-language models", "eye tracking", "attention alignment", "sound symbolism"]
innovations: ["在同一刺激上直接对比人类与VLM的选择与空间注意力，揭示选择对齐与注意力对齐的解耦", "提出中心偏置基线证明VLM显著性图质量低于简单启发式", "通过+AvgGaze对照分离trial-specific眼动监督与静态空间先验的贡献"]
benchmarks: ["Human VLM comparison on bouba-kiki task", "Spatial alignment via Spearman correlation with gaze"]
---

# 论文速读：Similar-Choices-Different-Attention-Cross-Modal-Associations

## 一句话总结
本研究通过对比人类与VLMs在相同bouba-kiki任务上的选择与眼动注意力，发现较大VLMs虽能匹配人类选择，但其显著性图与人类注视点的对齐程度不如简单的中心偏置基线；微调模型可提升选择一致性，但注意力对齐仍落后，且仅匹配选择并不足以说明跨模态处理的人类对齐。

## 研究问题与动机
- 现有研究多使用不同刺激或任务对比人类与VLMs的跨模态关联（如bouba-kiki效应），缺乏严格匹配条件下的直接比较。
- 仅凭选择一致无法判断模型是否以与人类相同的方式关注图像特征——模型可能因非目标属性（如位置、颜色）而非形状本身做出正确选择。
- 需要回答：VLMs是否与人类一样做出跨模态配对？如果是，它们是否关注相同的视觉区域？
- 前人对CLIP等模型的研究结论矛盾（Alper & Averbuch-Elor 2023声称存在关联，Kouwenhoven et al. 2025则未复现），原因可能源于评估协议不一致。

## 核心贡献（创新点）
- 在同一刺激集上直接对比人类（n=53）与VLMs的选择和眼动数据，弥补了先前工作因刺激/任务不匹配导致的结论分歧。
- 引入空间注意力对齐（Saliency-gaze correlation）作为选择对齐之外的独立评估维度，揭示"选择正确≠注意力相似"的关键差异。
- 通过微调实验证明：人类选择对齐可泛化至未见词汇和图像，但注意力对齐仍需额外训练且仅靠平均注视图即可达到同等效果。
- 提出中心偏置基线（Center-bias baseline）作为简单对照，证明现有VLMs的空间显著性甚至不如固定高斯分布。
- 设计了多控制条件的微调实验（+Gaze、+AvgGaze、+Gaze+Side），分离了眼动监督中 trial-specific 信息与静态空间先验的贡献。

## 方法详解
- **刺激设计**：162个双音节伪词（基于音素规律构建，如 KEEKEE/KEETUH 为尖锐类，MOHMAH/LOONOO 为圆形类）+ 20个形容词 + 4个经典伪词（bouba/kiki/takete/maluma）；184对圆-尖图像（含线描图75对、Gemini生成抽象图92对、既往研究图像17对）。
- **人类实验**：53名参与者，8,208个trial，Tobii眼动仪记录，记录 gaze samples 与 fixation duration，选择与注视耦合率达 77-80%。
- **VLM评估**：6个嵌入模型（CLIP RN50/ViT-B-32、SigLIP-base/so400m、SigLIP2-base/so400m）和6个生成VLM（Gemma4-E4B/26B、Qwen3-VL-8B/30B、Molmo2-4B/8B）。嵌入模型用Grad-CAM，生成模型用决策token注意力（最后半层多头平均）。
- **空间对齐指标**：在每个图像内计算模型显著性图与人类注视点的Spearman ρ，再跨两图平均。
- **微调损失函数**：$\mathcal{L}_t = \text{CE}(y_t) + \lambda \frac{\sum_{i \in \{A,B\}} m_{t,i} \text{KL}[H_{t,i} \| A_{t,i}]}{\sum_i m_{t,i}}$，其中 CE 为选择交叉熵，KL 项加权注视分布与模型注意力的散度，λ=0（纯选择）或 λ=1（加入注视监督）。
- **基线对照**：中心偏置基线（每个图像中心放置 σ=0.125 的各向同性高斯）；人口均值图（训练集所有trial的注视平均）。

## 实验与结果
- **零样本选择对齐**：嵌入模型在形容词上ICA≈0.75，但在伪词上接近随机；生成VLMs中仅Gemma4-26B（ICA=0.62, ρ=0.58）和Qwen3-VL-30B（ICA=0.57, ρ=0.56）在伪词上显著优于人类分布。
- **零样本空间对齐**：所有模型的空间 ρ 均低于中心偏置基线（Center=0.60-0.74）；CLIP ViT-B/32 最佳（ρ=0.588）但仍低于基线（0.743）；Qwen3-VL-30B 和 Molmo2-8B 的 ρ 接近零或负值。
- **微调效果**：Choice-only 微调使小模型人类选择一致性从 ~50% 提升至 ~72-73%，达到人类多数投票参考水平（72.86%），并泛化至未见词和图像。
- **注视监督效果**：+Gaze 将小模型的空间 ρ 从 0.03-0.35 提升至 0.67-0.73，但 +AvgGaze（固定平均注视图）达到几乎相同的水平（0.67-0.73），ΔKL 提升仅 0.07-0.08 bits。
- **核心发现**：选择对齐与空间对齐彼此独立——微调选择不改善空间对齐，添加注视损失不改善选择对齐。

## 相关工作脉络
- **Alper & Averbuch-Elor (2023)**：声称CLIP在嵌入空间中存在圆-尖方向，但本研究指出其使用了不同刺激和形容词基线设定。
- **Kouwenhoven et al. (2025)**：未发现CLIP中跨模态关联证据，本研究在其基础上增加了空间对齐评估和匹配刺激对比。
- **Iida & Funakura (2024)**：在日本VLM上无法复现Alper的结果，本研究解释了这一差异可能源于英语导向的刺激设计。
- **Pani & Yang (2025)、Yu et al. (2017)、Sood et al. (2020)**：已有研究证明眼动正则化可改善注意力对齐，本研究确认了这一结论但进一步发现其对选择无迁移效应。
- **Shrestha et al. (2020)**：VQA中随机注视图可复现人类注视监督的增益，与本研究 AvgGaze 控制结果呼应。

## 局限性与未来方向
- 仅使用三个模型和一个优化种子，结果可能受训练随机性影响。
- 书面标签混淆了语音学与正字法贡献，无法分离两者。
- 参与者均为英语母语机构成员，样本代表性有限。
- Grad-CAM 和注意力图为相关读取，非因果解释。
- 未来工作可扩展至更多模型家族、不同语言伪词、以及因果归因分析。

## 研究启发与可借鉴点
- **多维度评估范式**：同时评估选择行为和空间注意力，避免单一指标误导结论，适用于各类VLM认知对齐研究。
- **简单基线对照设计**：中心偏置高斯作为"无信息"对照，有效揭示了模型显著性图的质量缺陷，值得在注意力评估中推广。
- **微调泛化测试**：在未见词汇和图像上测试微调后性能，而非仅在训练分布内评估，更能反映真实关联能力。
- **控制实验设计**：+AvgGaze 作为 +Gaze 的对照，分离了 trial-specific 信息与静态空间先验的贡献，这一实验设计可迁移至其他多模态监督学习场景。
- **位置鲁棒性检验**：通过图像左右交换评估模型是否真正依赖形状而非位置捷径，为评估协议提供标准化工具。

## 关键术语表
- **Cross-modal association（跨模态关联）**：不同感知模态间的系统性对应关系，如语音与形状、声音与大小的关联。
- **Bouba-kiki effect（布巴-奇奇效应）**：经典跨模态现象，人类普遍将"bouba"与圆形形状、"kiki"与尖角形状配对。
- **Intended-category agreement (ICA)**：响应与标签预期类别（圆/尖）一致的比率。
- **Decision-token attention（决策token注意力）**：生成VLM中预测A/B答案位置的注意力分布，反映模型决策时的视觉聚焦。
- **Grad-CAM**：基于梯度的视觉解释方法，通过特征图梯度定位图像中對分类贡献最大的区域。
- **Center-bias baseline（中心偏置基线）**：在每个图像中心放置固定高斯分布，作为无输入信息的注意力对照。
- **Spatial alignment（空间对齐）**：模型显著性图与人类注视点之间的相关性，衡量视觉注意力模式的一致性。
- **Choice alignment（选择对齐）**：模型输出与人类选择在统计层面的一致性程度。

## 可复现要素
- 数据集：8,208个人类trial（53参与者，眼动数据+选择数据），论文声明将公开刺激和眼动数据。
- 代码：论文声明将在发表后提供全部代码。
- 关键超参：LoRA rank=16, scaling=32, dropout=0.05；AdamW lr=2e-5, 5% warmup, cosine decay；batch size=8, bf16；最多6 epochs, patience=2。
- 输入分辨率：512×512；Grad-CAM使用各模型原生空间格子，不放大。
- 模型：CLIP RN50/ViT-B-32, SigLIP-base/so400m, SigLIP2-base/so400m, Gemma4-E4B/26B, Qwen3-VL-8B/30B, Molmo2-4B/8B。

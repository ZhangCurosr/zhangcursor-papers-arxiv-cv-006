---
title: "UNIAFFORD-TOKEN-ROUTED-MULTITASK-LEARNING-FOR-GENERALIZABLE"
source: https://arxiv.org/pdf/2609.37264v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 22:06:32"
field: "多模态物体功能感知与稠密预测"
keywords: ["affordance perception", "token routing", "multimodal MLLM", "2D-3D grounding", "zero-shot generalization", "semantic pseudo-pairing"]
innovations: ["提出 Token Router for Tasks 解耦路由与语言生成，使稠密损失直接塑造共享 MLLM 表示", "构建 UniAfford-Data 统一数据集与语义级伪配对机制，打通异构 2D/3D 标注", "在 OOD zero-shot 与模态隔离协议下分别达到 AGD20K/GEAL*/ReasonAff/PIAD 的 SOTA 或次优性能"]
benchmarks: ["AGD20K", "GEAL*", "ReasonAff", "PIAD", "PIADv2"]
---

# 论文速读：UNIAFFORD: TOKEN-ROUTED MULTITASK LEARNING FOR GENERALIZABLE 2D-3D AFFORDANCE PERCEPTION

## 一句话总结
论文提出了 **UniAfford** 框架与 **Token Router for Tasks** 范式，通过将像素级 2D 监督与点级 3D 监督统一于共享的 MLLM 语义空间中，实现了跨视觉‑几何空间的通用可迁移物体‑功能语义学习，在无目标微调下获得强 OOD zero‑shot 泛化与 SOTA 分支性能。

## 研究问题与动机
- **2D/3D 亲和感知割裂**：现有方法将 2D（图像像素掩码）与 3D（点云逐点标注）独立发展，数据集、标注格式、评估协议碎片化，阻碍跨模态语义迁移。
- **融合≠统一学习**：既有跨模态工作（如 IAGNet、GREAT）侧重输入端融合以改善 3D 定位，并未让像素级与点级稠密预测目标共同塑造共享表示。
- **路由依赖预定义标记**：LISA 等多任务 MLLM 通过生成特定分段 token 的隐藏状态调度分支，任务分发与语言生成耦合，难以灵活适应异构监督。
- **跨实例语义关联缺失**：不同来源的图像/点云样本可能共享相同功能语义，但缺乏实例级空间对应，需要一种轻量且可扩展的统一组织方式。

## 核心贡献（创新点）
1. **Token Router for Tasks**：直接从 MLLM 上下文隐状态预测分支分配，无需语言头生成预定义任务标记；分支特定的稠密预测损失可反向塑造共享表示。*与现有基于 marker 的分发机制本质区别在于将路由决策与语言生成解耦。*
2. **UniAfford 统一框架**：以 MLLM 为共享语义枢纽，配合模态感知路由器产生图像/点云查询，分别驱动 SAM 风格 2D 解码器与 SONATA 风格 3D 解码器，支持单模态与联合推理。*区别于以往“输入融合+单一输出”设计，本文实现双分支稠密预测共享语义表示。*
3. **UniAfford‑Data 数据集**：在统一物体‑功能分类法下整合 RAGNet/ReasonAff 像素级 2D 标注与 PIADv2/AGPIL 点级 3D 标注，通过语义级伪配对连接跨实例监督。*首次提供同时含 2D/3D 标注且以共享语义索引组织的统一数据集。*
4. **强 OOD zero‑shot 泛化与 SOTA 分支性能**：无需目标微调即在 AGD20K 与 GEAL\* 上取得领先零样本结果，并在模态隔离训练/评估下分别达到 ReasonAff、PIAD、PIADv2 的最优或次优性能。*证明异构联合监督可显著提升跨域与跨模态迁移能力。*

## 方法详解
- **统一编码**：文本经 tokenizer 编码，图像经 SigLIP 编码器，点云经预训练 SONATA 编码器，三者投影后动态拼接为统一 prefix 输入 MLLM（Qwen3‑VL）。
- **模态感知 Token Router**：对每个有效响应状态 $h_t$，路由头 $g_r$ 输出类别分布 $\{text, img, pc\}$，软概率用于可微监督，硬分配 $r_t = \arg\max p_{t,c}$ 决定分支。可用分支由输入与标注是否存在决定，不可用分支 logit 被屏蔽（$-\infty$）。
- **分支查询生成**：被分配至 img/pc 的状态分别经投影 $g_{img}, g_{pc}$ 得到查询序列 $Q^{img}, Q^{pc}$，按自回归顺序拼接并填充对齐。
- **2D 解码器（SAM‑style）**：将路由图像查询与 SAM 图像特征做投影余弦相似度，生成粗略热力图 $M^{img}$，经 prompt encoder 编码后送入 mask decoder 得到像素级预测 $\widehat{Y}^{2D}$。
- **3D 解码器（SONATA‑based）**：将路由点云查询与 SONATA 点特征做缩放余弦相似度，直接输出逐点亲和 logits $\widehat{Y}^{3D}$。
- **训练损失**：$\mathcal{L} = \lambda_{txt}\mathcal{L}_{txt} + m_{2D}\mathcal{L}_{2D} + m_{3D}\mathcal{L}_{3D} + \mathcal{L}_{router}$。$\mathcal{L}_{txt}$ 为排除 anchor 位置的标准语言建模 CE；$\mathcal{L}_{2D}$ 为 focal+Dice；$\mathcal{L}_{3D}$ 为 BCE+Dice；$\mathcal{L}_{router}$ 包含路由 CE、存在损失（noisy‑OR 估计分支激活）与稀疏损失（SmoothL1 控制期望 token 数）。
- **语义级伪配对**：图像与点云实例仅要求共享相同的物体‑功能标签组合，不要求同一物理实例或空间对齐，从而打通异构来源的标注。

## 实验与结果
- **数据集**：UniAfford‑Data（162 类物体、162 类功能、25k 张 2D 样本、69k 条 3D 样本），另有轻量 Sample 版用于消融。
- **基准与协议**：OOD zero‑shot 协议（混合训练后直接评测）：2D 在 AGD20K，3D 在 GEAL\*（LASO‑C 的 Scale/Jitter/Rotate severity=2 子集）；模态隔离协议分别评测单分支：2D 在 ReasonAff，3D 在 PIAD 与 PIADv2。
- **主要结果**：
  - **AGD20K（zero‑shot）**：gIoU 27.52 / cIoU 25.22，优于所有非推理型基线，SIM 0.37 最高。
  - **GEAL\*（zero‑shot）**：AUC 83.55 / mIoU 14.67 / SIM 0.565 / MAE 0.102，显著超越 OpenAD、IAGNet、GREAT 等零样本方法，且超过基于 GEAL 训练的 LASO/GEAL 参考模型的 AUC 与 SIM。
  - **ReasonAff（2D 隔离）**：gIoU 71.19 / cIoU 73.63，较 Affordance‑R1 的 67.41/62.72 分别提升 3.78/10.91。
  - **PIAD（3D 隔离）**：mIoU 14.25（较 DAG 的 9.73 绝对提升 4.52），AUC 77.33 最优，MAE 0.107 最优。
  - **PIADv2（3D 隔离）**：AUC 75.67 / mIoU 9.26，超越 GREAT。
- **消融**：路由替换为固定 anchor 后 2D gIoU 降至 62.28、3D mIoU 降至 17.46；单模态训练使 2D gIoU 降至 41.41、3D mIoU 降至 30.07；替换为 prompt 式耦合后 2D gIoU 36.97、3D mIoU 14.31，均证实各项设计必要性。

## 相关工作脉络
- **2D/3D 亲和感知独立研究**：如 UMD、AGD20K、3D‑AffordanceNet、PIADv2、LASO 等分别聚焦单一空间，本文通过共享分类法与数据集打破这种割裂。
- **语言引导的亲和感知**：LISA、AffordanceVLM、SeqAfford 等利用语言建立语义‑空间联系，但大多仅输出 2D 掩码或 3D 点级预测，未同时训练双分支稠密预测。
- **跨模态输入融合方法**：IAGNet、GREAT 将视觉与几何信息融合后主要针对 3D 输出，其融合模块针对特定任务定制，缺乏通用路由接口。
- **多任务 MLLM 接口**：UnifiedMLLM、u‑llava 等依赖预定义 marker 进行任务分发（类似 LISA 的 embedding‑as‑mask），本文路由直接基于隐状态分类，不与语言生成耦合。
- **扩散先验利用**：DAG 借助文本到图像扩散模型提取亲和先验，属生成式先验路径；本文完全在判别式 MLLM 框架内完成跨模态联合学习。

## 局限性与未来方向
- **计算开销较大**：依赖大型预训练 MLLM 与双解码器，3D 推理 FLOPs（1930.87 GFLOPs）显著高于专用 3D 基线（GREAT/IAGNet 约 13–21 GFLOPs）。
- **语义级配对而非实例级对齐**：图像与点云仅通过相同标签关联，缺乏几何一致性监督，限制了对空间一致性的建模。
- **评估局限于感知任务**：未涉及闭环机器人操作验证，实际交互性能待检验。
- **未来方向**：更高效 backbone/解码器设计、引入实例级配对的大规模数据、真实机器人平台评估、扩展至其他多任务稠密预测场景。

## 研究启发与可借鉴点
- **Token Router for Tasks 范式可迁移**：该解耦路由机制适用于任何需多分支稠密输出的多模态 MLLM 任务（如分割、检测、关键点对齐），避免强制语言头生成特殊标记。
- **语义级伪配对组织异构数据**：当多源标注存在尺度/视角/实例不一致时，以高层语义索引替代严格几何对齐，可大幅降低数据构建成本并扩大训练覆盖。
- **稠密损失反向塑造共享表示**：通过软路由将不同模态的稠密预测目标汇入同一 MLLM，可使语言表征同时承载视觉与几何功能语义，提升跨域泛化。
- **路由诊断接口**：利用语言头对已路由状态进行 token 解码，可直观检验表示是否保留有意义的功能语义，为多任务表示质量提供廉价分析手段。

## 关键术语表
- **Affordance perception**：识别物体上支持特定交互的可操作区域（如抓握、按压、乘坐）。
- **Token Router for Tasks**：基于 MLLM 上下文隐状态直接预测分支归属的多任务路由机制，与语言生成解耦。
- **UniAfford‑Data**：统一组织像素级 2D 与点级 3D 标注的数据集，采用共享物体‑功能索引与语义级伪配对。
- **Semantic‑level pseudo‑pairing**：仅按相同物体‑功能标签将不同来源的图像与点云实例关联，不要求空间或实例对应。
- **Modality‑aware router**：感知当前可用模态及其标注，对不可用分支屏蔽路由 logit 的结构化路由器。
- **SAM‑style 2D decoder**：利用投影相似度构建热力图并传入 SAM prompt encoder/mask decoder 完成像素级分割。
- **SONATA‑based 3D decoder**：通过缩放查询‑特征余弦相似度直接输出逐点亲和 logits 的点级预测头。
- **OOD zero‑shot generalization**：在训练分布外 benchmark 上不作任何目标微调即可迁移已有功能语义的能力。

## 可复现要素
- **数据集**：UniAfford‑Data（含 Sample 与 Final 两个规模版本）；论文提供项目页 https://4dvlab.github.io/UniAfford/，未明确声明代码/权重开源状态，需以项目页为准。
- **代码/权重**：项目页链接已给出；论文未明确提供 GitHub 仓库或权重下载链接，建议访问项目页获取最新信息。
- **关键超参**：MLLM（Qwen3‑VL）LoRA rank=8、scale=16、dropout=0.05；图像分辨率 1024×1024，点云采样 2048 点；混合训练学习率：MLLM 1e‑5、2D 解码器 5e‑6、3D 解码器 5e‑4、Router 1e‑3；优化器 AdamW，线性 warmup+cosine LR；loss 权重 $\lambda_{txt}=1.0,\lambda_e=0.5,\lambda_s=0.01$；硬件 NVIDIA B200 GPU。

---
title: "UNIAFFORD-TOKEN-ROUTED-MULTITASK-LEARNING-FOR-GENERALIZABLE"
source: https://arxiv.org/pdf/2609.37264v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 16:48:42"
field: "多模态具身感知"
keywords: ["affordance perception", "multimodal MLLM", "token routing", "2D-3D grounding", "zero-shot generalization", "dense prediction", "multitask learning"]
innovations: ["Token Router for Tasks：从MLLM隐状态学习分支路由，解耦任务分发与预定义marker生成", "UniAfford-Data：统一2D/3D affordance taxonomy与语义级pseudo-pairing，支持异构监督联合训练"]
benchmarks: ["AGD20K", "GEAL*", "ReasonAff", "PIAD", "PIADv2"]
---

# 论文速读：UNIAFFORD-TOKEN-ROUTED-MULTITASK-LEARNING-FOR-GENERALIZABLE

## 一句话总结
论文提出 **UniAfford** 框架与 **Token Router for Tasks** 范式，将 2D 像素级与 3D 点云级 affordance 感知统一到同一个 MLLM 语义枢纽中，通过可学习的 token 路由器将上下文隐状态按需分发至不同模态分支，实现无需目标域微调的跨数据集 zero-shot 泛化与 SOTA 分支级性能。

## 研究问题与动机
- 现有 2D affordance grounding 与 3D affordance grounding 长期作为独立任务发展，数据集、标注格式、评估协议互不兼容，导致跨模态语义难以迁移。
- 多模态融合方法仅停留在输入侧融合，缺乏共享的语义组织方式，难以让像素级与点级监督共同塑造统一的 affordance 表征。
- 现有 MLLM 多任务接口依赖预定义任务标记（如 `<seg>` 等 marker token）进行分支调度，任务分发与语言头生成强耦合，限制了 dense prediction loss 对共享表征的直接塑形能力。
- 如何构建一个统一框架，利用异构 2D/3D 监督联合学习可迁移的 object–affordance 语义，并在 OOD 零样本场景下实现跨数据集泛化？

## 核心贡献（创新点）
- **Token Router for Tasks 范式**：直接从 MLLM 上下文隐状态预测分支分配，无需语言头生成预定义任务 marker；训练时仅用通用锚点提供路由监督，推理时由学习到的路由器直接决定分支归属。与 LISA-style marker-based 路由的本质区别在于解耦了任务分发与词表生成。
- **UniAfford 统一框架**：以 MLLM 为共享语义枢纽，通过 modality-aware router 产出 image/point-cloud affordance queries，分别驱动 SAM-style 像素解码器与 SONATA-based 点云解码器，支持 image-only、pointcloud-only 及配对多模态输入的统一推理。
- **UniAfford-Data 数据集**：在共享 object–affordance 分类体系下整合 2D 像素标注与 3D 点标注，构建语义级 pseudo-pair，使不同来源的图像/点云样本可在共享任务索引下联合训练，无需实例级空间对应。
- **强 OOD zero-shot 泛化与 SOTA 分支性能**：在 AGD20K（2D）与 GEAL\*（3D）上无需目标域微调即取得零样本最强结果；在 ReasonAff/PIAD/PIADv2 等基准的 modality-isolated 训练/评估下同样刷新 SOTA。

## 方法详解
- **统一多模态编码**：文本指令 $X$、图像 $I$、点云 $P$ 分别通过 $E_{\text{txt}}^{\text{mllm}}$、SigLIP 编码器 + 投影层、SONATA 编码器 + 投影层映射到共享 MLLM 输入空间，动态拼接为 prefix $\breve{T}^{\text{in}}=[T^{\text{txt}}; T^{\text{img}}; T^{\text{pc}}]$，送入 MLLM 自回归生成隐状态序列 $H$。
- **Modality-aware Token Router**：对每个有效响应隐状态 $h_t$，路由头 $g_r$ 输出三类 logits $\{ \text{text, img, pc} \}$，经 availability-masked softmax 得到概率分布，hard assignment 为 $r_t=\arg\max_c p_{t,c}$。图像/点云分支投影为 $q_t^{\text{img}}=g_{\text{img}}(h_t)$、$q_t^{\text{pc}}=g_{\text{pc}}(h_t)$，按自回归顺序拼接为查询序列 $Q^{\text{img}}$、$Q^{\text{pc}}$。
- **Affordance Decoders**：2D 分支使用 SAM-style 解码器，通过可学习投影后的 query–feature 相似度构建 coarse heatmap $M_{u,v}^{\text{img}}=s_{\text{img}}\langle \text{Norm}(\phi_{\text{img}}(F_{u,v}^{\text{img}})),\text{Norm}(\gamma_{\text{img}}(q^{\text{img}}))\rangle$，再经 prompt encoder 与 mask decoder 输出像素级 logits $\widehat{Y}^{\text{2D}}$。3D 分支使用 SONATA-based 点解码器，直接计算点特征与 routed query 的缩放相似度 $\widehat{Y}_i^{\text{3D}}=s_{\text{pc}}\langle \text{Norm}(\phi_{\text{pc}}(F_i^{\text{pc}})),\text{Norm}(\gamma_{\text{pc}}(q^{\text{pc}}))\rangle$。
- **联合训练目标**：$\mathcal{L}=\lambda_{\text{txt}}\mathcal{L}_{\text{txt}}+m_{\text{2D}}\mathcal{L}_{\text{2D}}+m_{\text{3D}}\mathcal{L}_{\text{3D}}+\mathcal{L}_{\text{router}}$。语言损失仅对非锚点有效响应监督；2D 损失为 focal+Dice，3D 损失为 BCE+Dice；router 损失包含 token-level CE、existence BCE 与 sparsity SmoothL1，其中 existence 鼓励可用分支获得查询，sparsity 抑制冗余分配。锚点 token（`<img-aff>`、`<pc-aff>`）被排除在语言建模损失之外，仅提供路由监督。

## 实验与结果
- **数据集与协议**：UniAfford-Data 整合 RAGNet、ReasonAff（2D）、PIADv2、AGPIL（3D），共 162 类 object/affordance、25k 2D 样本、69k 3D 样本；评估采用 mixed-training OOD zero-shot 协议与 modality-isolated 协议。
- **2D 零样本（AGD20K）**：UniAfford 取得 gIoU=27.52、cIoU=25.22、SIM=0.37，超越所有非推理型 MLLM 基线并与推理型方法相当。
- **3D 零样本（GEAL\*）**：UniAfford 取得 AUC=83.55、mIoU=14.67、SIM=0.565、MAE=0.102，超过全部 zero-shot 基线及 LASO/GEAL reference 模型的 AUC 与 SIM。
- **分支独立 SOTA（ReasonAff/PIAD/PIADv2）**：ReasonAff 上 gIoU=71.19、cIoU=73.63，超越 Affordance-R1；PIAD 上 mIoU=14.25（较 DAG 的 9.73 提升 4.52 点）；PIADv2 上全面超越 GREAT。
- **消融**：learned routing 较 fixed-anchor 提升 2D gIoU 6.51、3D mIoU 17.10；joint 训练较 2D-only/3D-only 分别提升 2D gIoU 27.38、3D mIoU 4.49；decoder coupling 替换为 prompt-style 后 2D gIoU 降至 36.97、3D mIoU 降至 14.31。

## 相关工作脉络
- **2D/3D affordance grounding**（Do 2018; Vo 2023; Li 2024a）：本文统一两者共享 taxonomy，而非各自独立训练。
- **Language-guided affordance**（LISA Lai 2024; RAGNet Wu 2025a; DAG Liu 2025a; SeqAfford Yu 2025）：前者依赖 marker token 路由，本文解耦任务分发与词表生成。
- **Cross-modal 2D-3D learning**（IAGNet Yang 2023; GREAT Yang 2024）：侧重 3D 输出空间的输入融合，本文在输出侧同时支持像素与点级 dense 预测。
- **MLLM multitask routing**（UnifiedMLLM Li 2024b; u-llava Xu 2024）：采用预定义 marker 调度，本文引入 soft routing + structure losses 的通用范式。
- **SAM/SONATA decoders**（Kirillov 2023; Wu 2025c）：本文将其扩展为由 MLLM routed queries 驱动的 affordance 专用解码接口。

## 局限性与未来方向
- 依赖大型预训练 backbone（Qwen3-VL、SAM、SONATA），训练与推理成本较高，3D 推理 FLOPs 显著高于专用轻量模型。
- 语义级 pseudo-pairing 未建立实例级空间对应，限制了几何一致性监督的利用。
- 当前评测聚焦 affordance grounding benchmark，未进入闭环机器人操作验证。
- 未来将探索更高效 backbone/decoder、结合语义与实例级配对的大规模数据、真实机器人评估，以及推广至更广泛的 multitask dense prediction 场景。

## 研究启发与可借鉴点
- **Router-for-Tasks 范式可迁移**：将“任务分支分配从语言 marker 生成转向隐状态学习路由”的设计可直接复用于多任务 VLM/MLLM 系统，避免 hard-coded token 约束。
- **Heterogeneous supervision 塑造共享表征**：通过不同模态 dense loss 反向传播到 MLLM 隐层，而硬路由不参与梯度，这一解耦策略可推广至其他多任务视觉-语言联合训练。
- **Availability-masked routing 机制**：推理时按输入可用性动态屏蔽不可用分支 logits，可防止无效分支占用计算资源，适用于动态输入场景。
- **语言头诊断（Language-head diagnostics）**：将 routed states 投影回词表分析可解释性，是一种轻量且可复用的表征审计方法。
- **可与本团队结合的方向**：若团队关注具身感知或多模态 dense prediction，可将 Token Router 范式迁移至语言引导的 3D 分割/检测、或与其他 dense decoder（如 DETR-style）结合。

## 关键术语表
- **Affordance perception**：定位支持特定交互操作的物体区域（如抓取把柄、按压按钮）。
- **Token Router for Tasks**：一种通用多任务训练范式，直接从 MLLM 上下文隐状态预测分支分配，无需语言头生成预定义 task marker。
- **Semantic-level pseudo-pair**：在同一 object–affordance 分类下跨源匹配图像与点云样本，仅共享语义标签而不要求实例级几何对应。
- **Modality-aware router**：根据分支可用性对路由 logits 施加 masking，并按 hard assignment 将隐状态投影为查询序列。
- **SAM-style pixel decoder**：基于 SAM 的 prompt encoder + mask decoder，接收 routed image query 生成的 coarse heatmap 作为密集提示。
- **SONATA-based point decoder**：基于 SONATA 的点特征提取器配合 scaled similarity 计算，输出逐点 affordance logits。
- **Existence & Sparsity losses**：路由辅助正则项，前者通过 noisy-or 估计鼓励可用分支获得查询，后者平滑约束查询数量避免冗余。
- **Modality-isolated protocol**：仅使用单一模态输入与监督分别训练/评估各分支，衡量统一架构的独立分支能力。

## 可复现要素
- **数据集**：UniAfford-Data，整合 RAGNet、ReasonAff、PIADv2、AGPIL，含共享 object–affordance taxonomy 与语义 pseudo-pairing；论文未声明是否已开源，仅给出项目页面链接。
- **代码/权重**：项目页面 https://4dvlab.github.io/UniAfford/，论文未明确声明代码与权重开源状态。
- **关键超参**：MLLM 使用 Qwen3-VL + LoRA（rank=8, scale=16, dropout=0.05）；图像分辨率 1024×1024，点云采样 2048 点；MLLM LR=1e-5，2D decoder LR=5e-6/1e-5，3D decoder LR=5e-4/1e-4，Router LR=1e-3；$\lambda_{\text{txt}}=1.0$，$\lambda_r=1.0$，$\lambda_e=0.5$，$\lambda_s=0.01$。
- **硬件**：NVIDIA B200 GPU。
- **指标**：2D 用 gIoU、cIoU、$P_{50}$、$P_{50-95}$、KLD、SIM；3D 用 AUC、mIoU、SIM、MAE。

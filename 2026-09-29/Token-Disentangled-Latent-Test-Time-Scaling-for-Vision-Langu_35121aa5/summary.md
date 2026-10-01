---
title: "Token-Disentangled-Latent-Test-Time-Scaling-for-Vision-Langu"
source: https://arxiv.org/pdf/2609.35228v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:37:40"
field: "多模态大模型推理效率优化"
keywords: ["test-time scaling", "latent refinement", "multimodal large language model", "token routing", "vision-language reasoning", "chain-of-thought"]
innovations: ["将视觉与推理奖励按token角色分流至不相交隐式子集", "基于图像扰动敏感度与LM头熵的零训练token路由机制"]
benchmarks: ["MMStar", "RealWorldQA", "HallusionBench", "ScienceQA-IMG", "MathVista", "LogicVista"]
---

# 论文速读：Token-Disentangled Latent Test-Time Scaling for Vision-Language Reasoning

## 一句话总结
本文提出了一种针对冻结版多模态大语言模型（MLLM）的推理时隐式状态优化方法，通过将视觉反馈与推理反馈按token角色分流更新——图像敏感token接收视觉奖励、高熵token接收推理奖励——在六个多模态推理基准上分别较CoT提升+2.57和+1.51的宏平均准确率。

## 研究问题与动机
- **现有隐式测试时缩放方法的盲点**：既有方法对可编辑隐式前缀施加单一全局标量奖励，无法区分不同token在"感知证据"与"推理决策"上的角色差异。
- **多模态推理错误的根源异质性**：模型可能产生看似连贯的回答但视觉依据不足（感知瓶颈），也可能抓住视觉线索却无法区分答案（推理瓶颈），全局奖励无法定位真实错误源。
- **单奖励广播的风险**：用统一候选级奖励更新所有隐式位置，可能将推理信号误注入感知敏感token，或反之，导致两种能力相互侵蚀。
- **参数成本约束**：再次训练多模态模型成本高昂，希望在不更新参数的前提下，仅通过推理时额外计算来提升已训练好的冻结MLLM。

## 核心贡献（创新点）
1. **揭示多模态隐式测试时缩放的 token-role 盲点**：指出候选级标量奖励无法区分"视觉证据依赖"与"推理不确定性"两类角色，与已有工作（LatentSeek、DMLR、SoftCoT等）的本质区别在于首次将角色感知引入隐式搜索。
2. **Token-Disentangled Latent Refinement 框架**：通过图像敏感性打分选择视觉token、通过LM头熵标准化选择推理token，将 $R_{vis}$ 与 $R_{rea}$ 严格路由到不相交的子集，避免双信号坍缩为全局更新。
3. **双奖励设计（视觉 engagement + 文本先验校正推理得分）**：视觉奖励使用图像token注意力engagement相对初轮rollout的tanh边界差分；推理奖励结合有界似然得分与文本先验扣除，并叠加候选间margin，二者均非黑盒LLM-as-judge。
4. **跨架构/跨规模的零参训练一致性增益**：在Qwen2.5-VL-7B与InternVL3.5-8B两大主骨干，以及Qwen2.5-VL-3B、InternVL3.5-4B、LLaVA-OV-1.5-8B、MiMO-VL-RL-8B、Qwen3-VL-8B、Qwen2.5-VL-32B共8个开放权重重型骨干上，感知与推理两组平均均稳定提升，证明方法不依赖特定架构。
5. **严格的 matched-budget 公平比较**：与Self-consistency/Best-of-N/Reward-only/LatentSeek在解码候选数、GPU秒级别上对齐，证明增益来自隐式状态定向编辑而非更多采样。

## 方法详解
### 总体框架
给定多模态上下文 $\xi=(I,q)$，冻结MLLM先生成初始CoT rollout $y$ 及其隐式轨迹 $h_{1:T}$。取前 $L$ 个解码器最后一层隐向量作为可编辑隐前缀 $z^{(0)}=[h_1,\dots,h_L]$，随后进行 $K$ 步隐式更新。

### 隐式前缀与解码
每步基于当前 $z^{(k)}$ 经冻结LM头映射得到局部分布 $\pi_i(\nu|z_i^{(k)})=\text{softmax}(W_{lm}z_i^{(k)})_\nu$，从中采样4条候选前缀（1条greedy + 3条加 $\mathcal{N}(0,0.3^2)$ 噪声的变体），拼接原始prompt后由模型继续自回归生成候选回答 $x\in X^{(k)}$。

### Token 路由
- **视觉token**：对初轮rollout，分别在原图 $I$ 与灰块扰动图 $\tilde{I}$（patch blackening，patch=14，drop=0.5）下计算 teacher-forced log-prob $\ell_i,\tilde{\ell}_i$，敏感度 $\nu_i=\exp(\tilde{\ell}_i-\ell_i)-(\tilde{\ell}_i-\ell_i)-1$，取 Top-$\rho_\nu$（默认$\rho_\nu=0.4$）标记 $w_i^{vis}=1$。
- **推理token**：在可编辑切片内计算各位置LM头熵 $H_i$，标准化 $\bar{H}_i=(H_i-\mu_H)/(\sigma_H+\epsilon)$，在非视觉token中取 Top-$\rho_r$（默认$\rho_r=0.4$）标记 $w_i^{rea}=1$，两 mask 严格不相交。
- 未被选中的token保留anchor正则，不接受直接策略奖励。

### 双奖励设计
- **视觉奖励** $R_{vis}(x)=\tanh((E(x)-E(y))/\tau_{vis})$，$E(x)$ 为生成轨迹层面图像token attention engagement的均值（末层注意力，至多统计前 $N'=128$ 个token），初轮 $R_{vis}(y)=0$。
- **推理奖励** $R_{rea}(x)=\lambda_s \bar{r}(x)+\lambda_{mar} r_{mar}(c(x))$，其中 $S(c)=\log p_\theta(c|I,q)-\lambda_p \log p_\theta(c|q)$ 扣除非图像文本先验，$\bar{r}$ 为有界化后的per-step均值，$r_{mar}$ 为与同批候选的最大margin。默认 $\lambda_s=1.0,\lambda_{mar}=0.5,\lambda_p=1.0,\tau_{vis}=0.2$。

### 损失函数
$$\mathcal{L}= \underbrace{-\lambda_{pg}^{vis} R_{vis}(x_k^\star) S_{vis}^{(k)} -\lambda_{pg}^{rea} R_{rea}(x_k^\star) S_{rea}^{(k)}}_{\mathcal{L}_{policy}} + \underbrace{\lambda_{ent}\frac{\sum w_i^{rea} H_i}{\sum w_i^{rea}+\epsilon}}_{\mathcal{L}_{ent}} + \underbrace{\lambda_{anchor}\|z^{(k)}-z^{(0)}\|_2^2}_{\mathcal{L}_{anchor}}$$
其中 $S_{vis}^{(k)}=\sum_i w_i^{vis}\log\pi_i(\hat{y}_{k,i}^\star|z_i^{(k)})$，$S_{rea}^{(k)}$ 类似。默认 Adam(lr=0.05)、$K=4$、$\rho=0.5$（即 $L=\min(\lfloor 0.5T\rfloor,300)$）、$\lambda_{anchor}=0.05$、$\lambda_{ent}=0.01$。

### 候选选择与聚合
每步选 $x_k^\star=\arg\max_{x\in X^{(k)}} R_{rea}(x)$ 驱动下一步更新；最终集合 $\{y\}\cup X^{(1)}\cup\dots\cup X^{(K)}$ 按归一化短答案分组，取人数最多组（平局按组内最大奖励决），等价复合分 $10\cdot\text{count}+\max R$。

## 实验与结果
- **骨干模型**：主实验 Qwen2.5-VL-7B、InternVL3.5-8B；泛化实验覆盖 3B–32B 共8个开放权重模型（Qwen2.5-VL-3B、InternVL3.5-4B、LLaVA-OV-1.5-8B、MiMO-VL-RL-8B、Qwen3-VL-8B、Qwen2.5-VL-32B）。
- **基准**：感知类 MMStar、RealWorldQA、HallusionBench；推理类 ScienceQA-IMG、MathVista、LogicVista。
- **基线**：CoT、Self-consistency、Best-of-N、Reward-only、LatentSeek(reasoning/perception)、DMLR。
- **主要结果**：
  - Qwen2.5-VL-7B：感知 67.96（+2.55 vs CoT 65.41），推理 69.94（+2.59 vs 67.35），宏平均优于Best-of-N +1.62、优于Reward-only +1.95。
  - InternVL3.5-8B：感知 66.83（+0.99），推理 72.05（+2.04）。
  - LatentSeek双变体在感知上均低于CoT（64.83/64.35 vs 65.41），说明单奖励易退化；本文路由方案避免此问题。
  - MiMO-VL-RL-8B 提升最大（感知+8.16、推理+7.22），印证对RL后训练模型收益更高。
- **扩展分析**：
  - 组件消融：移除 $R_{rea}$ 损失最大（64.19/57.31 → base），$R_{vis}$ 次之；随机路由劣于硬路由（64.72/58.87）。
  - $K$ 扫表：$K=4$ 最优，$K\ge 8$ 饱和；$\rho_\nu=\rho_r=0.4$ 为 routing 最优；严格不相交优于允许重叠。
  - 视觉路由信号：图像敏感性 top-k 优于 visual-focus/cosine-prototype；默认 patch=14、drop=0.5 最稳。
  - 推理路由：neg-log-prob 略优于 token loss；random非视觉路由显著劣。
  - Matched-budget：本方法与Best-of-N/Reward-only在 gen-calls≈17、GPU sec≈44 持平，但 MathVista 72.10、LogicVista 46.98 最高，增益不来自额外解码token。

## 相关工作脉络
- **Output-space test-time scaling**（Self-consistency、Best-of-N、Large LLM Monkeys）：通过重复采样或候选选择利用额外推理预算；本文转向隐式空间定向编辑，同等预算下更经济。
- **Latent test-time scaling**（Coconut、SoftCoT/++、LatentSeek、LatentEvolve、s1）：在连续隐式状态上搜索；本文指出其全局奖励在多模态角色混淆的缺陷并提出 token-level 路由。
- **多模态 latent 改进**（DMLR、VaLR）：强调保留视觉信息；本文进一步分解为"感知参与"与"推理质量"两个独立奖励并分别路由，避免单信号互相侵蚀。
- **Token-level credit assignment**（ToR、PRCO、Spotlight on token perception）：训练时分离感知/推理token；本文将其思想迁移至推理时的隐式更新阶段，实现 zero-training 的 test-time 解耦。
- **Visual grounding / hallucination 诊断**（Eyes Wide Shut、HallusionBench、Llava-Critic）：揭示MLLM感知瓶颈；本文用图像敏感性打分显式定位受影响token并定向加固。

## 局限性与未来方向
- **额外推理开销**：每样本需多次 latent 更新、候选解码、图像扰动前向与注意力提取，虽与 Best-of-N 在同一量级但仍高于单次 CoT。
- **代理奖励非验证器**：$R_{vis}$ 与 $R_{rea}$ 均为 heuristics，模糊图像、欠定问题或需外部知识的场景下仍可能噪声较大；tanh 边界只能抑制不能完全消除。
- **硬 top-k 路由的局限**：当前采用严格 disjoint mask，软路由或可学习路由未探索；不对称场景（视觉主导 vs 推理主导）下固定比例 $\rho_\nu=\rho_r$ 未必最优。
- **仅适用于开放权重骨干**：需要暴露 decoder 最后一层 hidden states 并通过 LM head 进行投影解码，API-only 闭源模型不可用。
- **长交互/多轮扩展未验证**：当前框架针对单轮问答，未处理多轮对话或 longer multimodal interactions。

## 研究启发与可借鉴点
- **角色感知的隐式搜索范式**：将"token角色识别→奖励分流"的思路迁移到纯文本推理的 latent test-time scaling（如 Lang-Seek、SoftCoT++）同样有效，值得验证。
- **图像扰动敏感度作为感知信标**：raw-patch blackening 计算代价低、可微友好，可推广至任意冻结 VLM 的视觉token定位，配合其它任务（VQA、OCR、时序定位）均有复用价值。
- **双奖励互补的正则化组合**：anchor 惩罚 + entropy 正则 + 边界 tanh 奖励构成一套稳定的隐式更新组件，可抽象为通用"latent policy gradient with role routing"模板。
- **Matched-budget 公平比较规范**：在 gen-calls、GPU秒、decoded-token 三层同时对齐基线，避免"多采样=强"的伪优势，为本领域实验设计树立标杆。
- **RL-post-trained 模型的更大收益**：MiMO-VL-RL-8B 提升最大，暗示经过 RLVR 的模型隐式状态更易被 reward 信号引导，为 RLHF/RLVR 的推理时利用提供新视角。

## 关键术语表
- **Latent test-time scaling**：在冻结模型推理时编辑其隐式状态（而非输出token或权重）以利用额外计算提升性能。
- **Token-disentangled routing**：按token角色将不同奖励分流至不相交的隐式子集，避免单一全局信号造成的角色混淆。
- **Image sensitivity**：通过比较原始图与扰动图下 token 条件概率的变化量来度量该 token 对视觉证据的依赖程度。
- **Visual engagement reward**：以末层图像token attention 均值相对初轮的有界差分作为感知侧奖励。
- **Reasoning reward**：扣除文本先验后的有界对数似然得分叠加候选间 margin，衡量答案质量。
- **Anchor regularizer**：惩罚编辑后隐式前缀偏离初始 rollout 的 L2 距离，防止过度偏移。
- **Chain-of-thought rollout**：模型在给定 prompt 下首轮的自回归生成轨迹，作为后续隐式优化的起点与视觉 baseline。
- **Matched-budget evaluation**：在解码候选数、GPU秒、调用次数等维度与基线严格对齐的比较协议。

## 可复现要素
- **代码**：已开源 https://github.com/Qwen-Applications/TD-LTTS。
- **模型**：主实验骨干 Qwen2.5-VL-7B-Instruct、InternVL3.5-8B-Instruct；其余 3B–32B 骨干均为开源模型。
- **数据集**：MMStar、RealWorldQA、HallusionBench、ScienceQA-IMG、MathVista、LogicVista，均为公开 benchmark。
- **关键超参**：$K=4$ 步、$L=\min(\lfloor 0.5T\rfloor,300)$、Adam lr=0.05、$\rho_\nu=\rho_r=0.4$、$\tau_{vis}=0.2$、$\lambda_{anchor}=0.05$、$\lambda_{ent}=0.01$、噪声 $\sigma=0.3$、图像扰动 patch=14 drop=0.5。
- **计算环境**：论文未明确说明 GPU 型号与数量；GPU 秒为分片运行估算值。
- **其他细节**：editable states 为最后 decoder 层 hidden states（LM head 之前）；KV cache 不复用，每候选独立 generate() 调用。

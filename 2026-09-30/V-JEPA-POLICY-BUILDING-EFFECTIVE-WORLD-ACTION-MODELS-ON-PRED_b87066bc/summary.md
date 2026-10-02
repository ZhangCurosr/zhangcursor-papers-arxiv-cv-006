---
title: "V-JEPA-POLICY-BUILDING-EFFECTIVE-WORLD-ACTION-MODELS-ON-PRED"
source: https://arxiv.org/pdf/2609.37250v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 01:17:57"
field: "具身智能与视觉-动作模型"
keywords: ["World-Action Models", "V-JEPA", "Predictive Visual Latents", "Flow Matching", "Robot Manipulation", "Distribution Shift"]
innovations: ["构建于冻结预测性视觉潜空间的世界-动作模型，无需继承预训练生成器", "层间上下文 key-value 接口实现预测-动作双向耦合", "预测器仅在无动作标签视频数据上预训练并 transfer 到下游控制"]
benchmarks: ["LIBERO", "LIBERO-Plus", "RoboCasa-GR1"]
---

# 论文速读：V-JEPA POLICY: BUILDING EFFECTIVE WORLD-ACTION MODELS ON PREDICTIVE VISUAL LATENTS

## 一句话总结
论文提出 V-JEPA Policy，一种构建在冻结 V-JEPA 2.1 编码器预测性潜空间上的世界-动作模型（WAM），通过从零联合学习指令条件化的未来潜预测器与 flow-matching 动作专家，在无需继承完整预训练视觉生成模型的前提下实现了高效的机器人控制；进一步通过在 DROID 视频-指令对上的预测器预训练，显著提升了跨分布泛化性能。

## 研究问题与动机
1. **核心问题**：现有 WAM 方法依赖适配大规模预训练的视觉生成模型（如视频生成模型或图像编辑模型），是否必须继承完整的预训练视觉生成器才能实现有效的 WAM 学习？
2. **潜在假设**：大规模预测性预训练所习得的预测性视觉潜空间本身可能已提供足够的表征基础，无需继承生成模型。
3. **延伸问题**：同一预测性潜空间是否能支持从更广泛的 in-the-wild 视频中获取可迁移的未来建模知识，并 transferring 到下游控制任务？
4. **效率考量**：如何在不依赖大型生成背核的前提下，实现紧凑、低显存占用的 WAM 部署？

## 核心贡献（创新点）
1. **框架创新**：提出 V-JEPA Policy，将 WAM 直接构建于冻结 V-JEPA 2.1 编码器的预测性潜空间之上，从零联合学习未来潜预测器与 flow-matching 动作专家，无需继承预训练视觉生成模型——与 FastWAM/ImageWAM 等方法依赖适配生成模型形成本质区别。
2. **耦合机制设计**：设计"未来信息感知的上下文接口"，通过预测器的层间上下文 key-value 状态条件化动作专家（MoT 架构双向注意力交互），使未来预测直接塑造动作生成——区别于 JEPA-WAM 等先训练预测器再引入对齐阶段的 Pipeline。
3. **知识迁移验证**：证明预测器仅在 DROID 视频-指令对上以未来预测目标预训练（无动作标签），可将未来建模知识 transfer 到下游 WAM 训练，在 LIBERO-Plus 上将成功率从 79.25% 提升至 91.50%，且效果无法通过延长下游训练时间复现。
4. **系统基准对比**：在匹配下游训练预算下，对比判别式（DINOv2）、重建式（WAN2.2 VAE）和视频理解导向（InternVideo3）等多种视觉基座，确立预测性潜空间在分布偏移下的鲁棒性优势。

## 方法详解
### 整体架构
- **冻结视觉编码器**：使用 V-JEPA 2.1 ViT-L（约 0.3B 参数）作为固定视觉特征提取器，定义预测性视觉潜空间。
- **冻结文本编码器**：使用 T5-XXL 编码任务指令。
- **可训练模块**：未来潜预测器 $P_\phi$（约 0.5B 参数）+ flow-matching 动作专家 $v_\psi$（约 0.1B 参数），总计 0.9B 参数，0.6B 可训练。

### 未来潜预测（Sec 3.1）
- **输入构造**：每相机视角取两帧构成最小观测上下文（匹配 tubelet size=2），将未来 tubelet 位置设为 mask；目标表示由冻结编码器处理完整视频得到。
- **双向注意力机制**：上下文 token 与可学习未来 query token（$\Delta_t^+$）通过双向自注意力交互，使上下文状态融合未来演化信息。
- **条件注入**：指令（T5 特征）与本体状态通过 cross-attention 同时注入上下文 token 和未来 query token。
- **预测输出**：单前向传播输出未来潜预测 $\hat{\mathbf{Z}}_t^+$。

### 预测-动作耦合（Sec 3.2）
- **层间上下文接口**：提取预测器第 $j$ 层的上下文 key-value 状态 $(\mathbf{K}_c^{(j)}, \mathbf{V}_c^{(j)})$，保留 3D RoPE 旋转后的 key 以保持时空位置信息，形成接口 $\mathcal{C}_\phi = \{( \mathcal{R}_{3D}(\mathbf{K}_c^{(j)}), \mathbf{V}_c^{(j)} )\}_{j=1}^L$。
- **MoT 联合架构**：预测器与动作专家共享层数、兼容 attention head 维度，动作 query 对上下文 key-value 和自身 action token 进行 joint attention：
  $$\mathbf{O}_a^{(j)} = \text{softmax}\left(\mathbf{Q}_a^{(j)}[\mathcal{R}_{3D}(\mathbf{K}_c^{(j)}); \mathbf{K}_a^{(j)}]^\top / \sqrt{d_h}\right)[\mathbf{V}_c^{(j)}; \mathbf{V}_a^{(j)}]$$
- **Flow-matching 动作生成**：动作专家建模连续动作 chunk，通过 conditional flow matching 学习速度场 $v_\psi(\mathbf{a}_t^\tau, \tau; \mathcal{C}_\phi, \ell, \mathbf{q}_t)$。

### 联合训练与推理（Sec 3.3）
- **未来预测损失**：
  $$\mathcal{L}_{\text{future}} = \mathbb{E}_{\mathcal{D}} \| P_\phi(\mathbf{Z}_t^c, \Delta_t^+ | \ell, \mathbf{q}_t) - \mathbf{Z}_t^+ \|_1$$
- **动作 flow-matching 损失**：
  $$\mathcal{L}_{\text{action}} = \mathbb{E} \| v_\psi(\mathbf{a}_t^\tau, \tau; \mathcal{C}_\phi, \ell, \mathbf{q}_t) - (\mathbf{a}_t - \epsilon) \|_2^2$$
- **联合目标**：$\mathcal{L} = \mathcal{L}_{\text{action}} + \lambda_{\text{future}} \mathcal{L}_{\text{future}}$，其中 $\lambda_{\text{future}} = 1$。
- **推理流程**：单步预测器前向生成 $\mathcal{C}_\phi$ 并缓存，动作专家从 Gaussian noise 经 10 步 Euler 积分采样至 clean action chunk。

## 实验与结果
### 基准与设置
- **LIBERO**：40 任务（Spatial/Object/Goal/Long 四套件），1693 演示数据，训练 10  epochs。
- **LIBERO-Plus**：10,030 扰动任务（7 类分布偏移），无额外微调。
- **RoboCasa-GR1**：24 个人形 tabletop 任务，50k steps，batch=256。
- **真实平台**：TianJi Marvin 双臂平台，Table Cleanup 与 Saucer Racking 各 20 次试验。

### 主要结果
| 方法 | 参数量 (B) | LIBERO | LIBERO-Plus | RoboCasa-GR1 | 真实任务 (平均) |
|------|-----------|--------|-------------|--------------|----------------|
| V-JEPA Policy (From Scratch) | 0.9 | **97.25%** | 79.25% | 50.92% | 45% |
| V-JEPA Policy (Pretrained Predictor) | 0.9 | **98.70%** | **91.50%** | **55.58%** | **80%** |
| FastWAM | 6.0 | 97.6% | 51.5% | — | 60%/35% |
| ImageWAM | 4.5 | 98.4% | 83.1% | — | — |
| PRTS | 5.0 | 98.4% | 96.6% | — | — |
| π0.5 | 3.3 | 96.9% | 92.4% | 37.0% | — |

- **最强结果**：预训练预测器变体在 LIBERO-Plus 达到 91.50%，较从头训练提升 12.25pp；真实双臂任务平均成功率 80%，较 45% 大幅提升。
- **效率优势**：峰值显存仅 4.66 GiB（RTX 4090），比 FastWAM 低 63.4%，推理延迟 178.17ms vs. 202.51ms。

### 消融分析
- **视觉基座对比**：V-JEPA 2.1 ViT-L 在 LIBERO-Plus 上较 DINOv2 提升 12.23pp（分布偏移下优势扩大）。
- **未来预测必要性**：移除未来 query 或未来损失均导致性能下降，证实显式未来潜监督不可替代。
- **尺度分析**：V-JEPA 2.1 比 V-JEPA 2 在分布偏移下提升更显著；ViT-L→ViT-G 增益有限，表明表征质量优于单纯扩大容量。

## 相关工作脉络
1. **JEPA 系列**：V-JEPA 通过特征空间预测学习视觉表征（Bardes et al., 2024; Assran et al., 2025; Mur-Labadia et al., 2026）；本文在其预测性潜空间上构建 WAM，区别于 DINO-WM（Zhou et al., 2024）和 action-conditioned V-JEPA 2（Assran et al., 2025）的规划用途。
2. **WAM 与视频生成**：FastWAM（Yuan et al., 2026）、ImageWAM（Zhang et al., 2026c）依赖适配预训练视频/图像生成模型；本文证明无需生成模型，仅凭预测性潜空间即可实现 WAM。
3. **VLA 与基础策略**：π0/π0.5（Black et al., 2024, 2025）、PRTS（Zhang et al., 2026b）、VLA-JEPA（Sun et al., 2026）；本文在更小参数量（0.9B vs. 2.3–7.8B）下实现可比性能，且无动作监督预训练。
4. **JEPA-WAM**（Lin et al., 2026）：使用 Qwen 初始化预测器并增加 V-L 对齐阶段；本文直接从零联合学习，简化训练流程。
5. **预测性表征优势**：与判别式（DINOv2/v3）、重建式（WAN2.2 VAE）、视频理解（InternVideo3）对比，确立预测性潜空间在分布鲁棒性上的独特价值。

## 局限性与未来方向
1. **下游训练预算敏感**：虽然预测器预训练能缓解，但直接从零训练时扩展下游 compute 仍出现泛化收益递减（Fig. 2a）。
2. **多相机设置依赖**：当前框架使用 2-4 个相机视图，未充分探索单目或稀疏视角下的扩展性。
3. **长程任务挑战**：在 RoboCasa-GR1 等长 horizon 任务上仍有较大提升空间（50.92%），高自由度人形控制仍需进一步优化。
4. **未来方向**：探索更大规模 DROID 类 in-the-wild 数据上的预测器预训练；扩展至多模态条件（深度、触觉等）；研究 predictor-only 预训练与其他视觉基座的兼容性。

## 研究启发与可借鉴点
1. **冻结预测性潜空间构建下游模块**：可复用 V-JEPA 预测器架构（含 3D RoPE、双向注意力）作为其他视觉-动作任务的通用预测骨干，无需重新设计特征提取器。
2. **层间上下文接口设计**：通过中间层 key-value 状态而非最终预测输出条件化下游模块，可作为一般性的"预测-行动"耦合范式迁移至其他 WAM 架构。
3. **预测器预训练 + 动作专家从零初始化**：分离"未来建模知识获取"与"动作生成"两个阶段，前者在大规模无动作标签数据上预训练，后者在任务数据上 joint fine-tuning，这一 split pretraining 策略值得在其他具身任务中验证。
4. **效率-性能 Pareto 分析框架**：论文展示的参数效率分析（Fig. 5）为后续工作提供了可复用的基准评估范式。

## 关键术语表
- **World-Action Model (WAM)**：耦合未来视觉状态预测与动作生成的机器人控制模型，通过预测性知识提升泛化能力。
- **V-JEPA (Visual Joint-Embedding Predictive Architecture)**：通过在特征空间进行 masked video prediction 学习视觉表征的自监督架构，避免像素级重建。
- **Flow Matching**：一种生成建模技术，学习从噪声到数据的连续流场，比 diffusion 更高效地采样。
- **Mixture-of-Transformers (MoT)**：多个 Transformer 模块共享注意力结构但参数分离的联合架构，用于耦合预测器与动作专家。
- **3D RoPE (Rotary Position Embedding)**：基于时空坐标的旋转位置编码，用于区分视觉潜空间中的 token 位置。
- **LIBERO-Plus**：LIBERO 的扩展基准，包含 7 类受控分布偏移（视角、光照、纹理等）的 10,030 个扰动任务。
- **DROID**：大规模 in-the-wild 机器人操作视频数据集，含语音指令与本体态标注。
- **Context Key-Value Interface**：预测器层间上下文 token 的 key-value 投影，作为预测信息传递给动作专家的接口。

## 可复现要素
- **数据集**：LIBERO（公开）、LIBERO-Plus（公开）、RoboCasa-GR1（公开）、DROID（公开）、TianJi Marvin 真实平台数据（论文未提及是否公开）。
- **代码**：已开源，https://github.com/breez3young/VJEPA-Policy。
- **模型权重**：V-JEPA 2.1 编码器（需从官方仓库获取）、T5-XXL（公开）、自定义预测器与动作专家权重（开源）。
- **关键超参**：学习率 $10^{-4}$，AdamW，warmup 5%，cosine decay；LIBERO 训练 10 epochs/21,360 steps，batch=128；RoboCasa-GR1 训练 50k steps，batch=256；预测器预训练 100k steps，batch=192。

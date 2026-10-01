---
title: "V-JEPA-POLICY-BUILDING-EFFECTIVE-WORLD-ACTION-MODELS-ON-PRED"
source: https://arxiv.org/pdf/2609.37250v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 22:06:59"
field: "具身智能与机器人策略学习"
keywords: ["World-Action Model", "V-JEPA", "Predictive Visual Latent", "Flow Matching", "Robot Policy", "Vision-Language-Action", "LIBERO-Plus", "Predictor Pretraining"]
innovations: ["在冻结的 V-JEPA 预测隐空间上从零联合训练未来预测器与 flow-matching 动作专家，无需继承完整视觉生成模型", "设计基于层状 context KV 状态的预测-动作耦合接口，实现未来建模知识端到端驱动动作生成"]
benchmarks: ["LIBERO", "LIBERO-Plus", "RoboCasa-GR1", "TianJi Marvin 真实双臂平台"]
---

# 论文速读：V-JEPA Policy: Building Effective World-Action Models on Predictive Visual Latents

## 一句话总结
论文提出 V-JEPA Policy，在冻结的 V-JEPA 2.1 预测视觉编码器构建世界-动作模型（WAM），通过单次下游阶段从零联合训练指令条件的未来潜在预测器与 flow-matching 动作专家，以仅 0.9B 参数（0.6B 可训练）在 LIBERO、LIBERO-Plus、RoboCasa-GR1 及真实双臂平台上取得与更大 WAM/VLA 基线相匹敌的性能，并验证了预测性视觉隐空间无需继承完整生成模型即可支撑有效 WAM 学习。

## 研究问题与动机
- **核心问题**：有效的 WAM 学习是否必须继承预训练的视觉生成模型（如视频生成器/图像编辑模型），还是仅凭大规模预测预训练得到的预测性视觉隐空间就足够？
- **现有方法不足**：主流 WAM 通过适配大规模预训练的视频生成或图像编辑模型获得，将预测知识与生成骨干同时迁移，成本高、参数量大；且未检验视觉基础本身对 WAM 学习的贡献。
- **知识迁移潜力未明**：同一隐空间能否吸收来自更广泛野外视频的未来建模知识（无需动作标签），并迁移到下游控制？
- **效率与泛化瓶颈**：在分布偏移下，不同类别视觉基础（判别式、重建式、视频理解式、预测式）的表现差异及其对 WAM 稳健性的影响尚未系统比较。

## 核心贡献（创新点）
- **提出 V-JEPA Policy 框架**：在冻结的 V-JEPA 预测编码器上直接联合训练未来预测器与 flow-matching 动作专家，无需继承完整预训练视觉生成模型。本质区别在于摒弃了视频生成器，仅保留预测隐空间作为 WAM 基础。
- **设计基于层状 context KV 状态的预测-动作耦合接口**：预测器的层状上下文 key-value 状态直接条件化动作专家，使未来建模知识端到端地塑造动作生成。区别在于此前工作大多用最终预测输出或外部想象循环连接视觉与动作模块。
- **系统比较不同视觉基础对 WAM 的影响**：在相同下游框架与训练预算下，证明预测性隐空间显著优于判别式（DINOv2）、重建式（WAN VAE）和视频理解式（InternVideo3）替代方案，尤其在分布偏移时差距扩大。
- **展示无动作标签的预测器预训练可迁移提升**：仅在 DROID 视频-指令对上预训练预测器，转移到下游 WAM 训练，LIBERO-Plus 成功率从 79.25% 提升至 91.50%，超过延长下游训练的收益。

## 方法详解
- **固定视觉隐空间**：冻结 V-JEPA 2.1 ViT-L 编码器，将每视角当前帧与前一时间帧配对为两帧观测上下文（匹配 temporal tubelet size=2），与未来观测 $o_t^+$ 拼接成完整 clip；编码后提取未来 tubelet 位置作为预测目标 $Z_t^+$，拼接多视角得到 $Z_t^c$ 与 $Z_t^+$。
- **未来查询预测器 $P_\phi$**：采用 V-JEPA 2 预测器架构，将 $Z_t^c$ 投影后与未来查询 token $\Delta_t^+$ 拼接，通过双向自注意力（3D RoPE + 可学习视角嵌入）联合更新上下文与未来查询，使上下文状态吸收未来演化信息；使用冻结 T5-XXL 编码指令 $\ell$，与本体状态 $\mathbf{q}_t$ 投影拼接形成条件序列，通过 cross-attention 注入每一预测器层。
- **预测-动作耦合接口 $\mathcal{C}_\phi$**：取预测器第 $j$ 层上下文位置的 key/value 对 $(K_c^{(j)}, V_c^{(j)})$，保留 keys 上的 3D RoPE 变换而 values 不旋转，堆叠全部 $L$ 层形成层状接口 $\mathcal{C}_\phi = \{(R_{3D}(K_c^{(j)}), V_c^{(j)})\}_{j=1}^L$，供动作专家使用。
- **Flow-matching 动作专家 $v_\psi$**：构建 MoT（Mixture-of-Transformers）架构，预测器与动作专家各 24 层、兼容 attention head 数/维度；动作 token 的 query 对上下文 KV 和自身动作 token 做联合注意力，预测器 token 不能反向 attends 动作 token。动作专家输出条件速度 $v_\psi(a_t^\tau, \tau; \mathcal{C}_\phi, \ell, q_t)$，flow-matching 目标为 $a_t - \epsilon$。
- **联合训练损失**：
  - 未来预测损失：$\mathcal{L}_{future} = \mathbb{E}_{\mathcal{D}} \|P_\phi(Z_t^c, \Delta_t^+ | \ell, q_t) - Z_t^+\|_1$
  - 动作 flow-matching 损失：$\mathcal{L}_{action} = \mathbb{E}\|v_\psi(a_t^\tau, \tau; \mathcal{C}_\phi, \ell, q_t) - (a_t - \epsilon)\|_2^2$
  - 总损失：$\mathcal{L} = \mathcal{L}_{action} + \lambda_{future}\mathcal{L}_{future}$，$\lambda_{future}=1$；动作损失可回传至预测器，二者联合优化。
- **推理**：采用 imagine-then-act 策略，每次决策一步执行单次预测器前向得到 $\mathcal{C}_\phi$ 并缓存，再由动作专家从 $\tau=0$ 到高斯噪声积分到 $\tau=1$（10 步 Euler），只评估动作专家。

## 实验与结果
- **数据集/基准**：LIBERO（40 任务，四套件）、LIBERO-Plus（10,030 扰动任务，七类偏移）、RoboCasa-GR1（24 类人形 tabletop 任务）、TianJi Marvin 真实双臂平台（Table Cleanup、Saucer Racking）。
- **参数规模**：总 0.9B，其中视觉编码器约 0.3B（冻结），预测器 0.5B，动作专家 0.1B；可训练 0.6B；无动作监督的 embodiment 预训练（P.T.）。
- **LIBERO**：V-JEPA Policy（from scratch）平均成功率 97.25%（Spatial 97.2 / Object 98.0 / Goal 97.0 / Long 96.8），接近 ImageWAM 98.4 / PRTS 98.4 等大模型，优于 FastWAM 97.6。
- **LIBERO-Plus**：from scratch 79.25%；预测器预训练后达 91.50%，显著优于 FastWAM 68.9 / π0.5 88.5，接近 ImageWAM 98.1；各偏移轴均提升，语言/纹理/初始状态提升最显著。
- **RoboCasa-GR1**：from scratch 50.92%（超 StarVLA-π 43.90 / π0.5 37.00），预训练后 55.58%；低于 ABot-M0（58.30）。
- **真实世界**：from scratch Table Cleanup 55%/Saucer Racking 35%；预训练后提升至 75%/85%，远超 FastWAM（60%/35%）和 π0.5（85%/50%）。
- **视觉基础对照**（相同下游配方）：V-JEPA 2.1 ViT-L 在 LIBERO 97.25%、LIBERO-Plus 79.25% 均最优；与最强判别式 DINOv2 差距从 2.35pp（in-distribution）扩大到 12.23pp（distribution shift）；重建式 WAN VAE LIBERO-Plus 仅 46.43%。
- **预训练 vs 延长训练**：60k 步 from scratch LIBERO-Plus 仅 81.64%，预训练后 10 epoch 即达 91.50%，扩展训练收益递减。
- **效率**：RTX 4090 上 mean latency 178.17ms / P95 186.81ms / peak VRAM 4.66 GiB，峰值显存较 FastWAM 低 63.4%。

## 相关工作脉络
- **JEPA / V-JEPA**：V-JEPA 系列在特征空间做自监督预测，学习时空动态而非像素重建；本文沿用其冻结编码器提供预测隐空间，区别在于后续模块从零训练而非继续预训练。
- **DINO-WM / Action-Conditioned V-JEPA 2**：已在预训练特征空间中学习动力学用于规划；本文直接在此隐空间上联合训练 WAM，省去额外对齐阶段。
- **VLA-JEPA（Sun et al., 2026）**：整合 VLM 主干与潜在预测；本文不依赖 VLM，仅用冻结 T5 作语言条件，框架更轻量。
- **JEPA-WAM（Lin et al., 2026）**：使用 Qwen 初始化的预测器并额外做视觉-语言对齐阶段；本文直接从下游数据训练，不做额外预训练。
- **FastWAM / ImageWAM / Cosmos Policy**：通过适配视频生成/图像编辑模型获得 WAM；本文去除生成骨干，证明预测隐空间本身即为充分基础。
- **Pi0 / Pi0.5 / PRTS / OpenVLA**：代表性 VLA/基础策略；本文在 0.9B 规模、无动作预训练条件下达到相似性能，突出参数效率与泛化优势。

## 局限性与未来方向
- 当前仅验证了单阶段联合训练与 DROID 单源预训练，未探索多源/多阶段预测器预训练（如结合更多动作-无动作混合数据）。
- 真实世界实验局限于两个双腕任务，未覆盖更广泛的长程操作场景；泛化边界尚需进一步评估。
- 冻结编码器限制了视觉表征的领域适配能力，针对特定机器人形态可能需要定制化编码器或轻微调策略。
- DROID 预训练使用 5Hz 采样与双视角，未来可探索更高频多视角下的未来建模知识吸收。

## 研究启发与可借鉴点
- **预测隐空间作为 WAM 基础的有效性验证**：为后续研究提供了"无需生成模型即可学习 WAM"的可行路径，可直接迁移到其它预测性表征（如未来发展的 V-JEPA 变体）上。
- **层状 context KV 接口设计**：将预测器的中间层状态而非最终输出用于动作条件化，使多层未来信息被充分利用；该接口可复用到其他"预测-决策"耦合架构中。
- **预测器预训练的跨任务迁移范式**：无动作标签的野外视频预测器预训练可直接作为下游 WAM 的初始化策略，低成本显著提升泛化；可推广到其它机器人操作数据集。
- **控制变量式的视觉基础对比实验设计**：固定下游骨架与训练预算、仅更换视觉编码器的实验设置，为后续评估新视觉表征在 WAM 中的适用性提供了标准评测范式。
- **参数效率的 Pareto 前沿分析**：将成功率与参数规模联合可视化，有助于后续工作在更紧凑模型上进行 trade-off 分析与复现对比。

## 关键术语表
- **World-Action Model (WAM)**：将未来视觉状态预测与动作生成耦合在一起的机器人控制模型，利用预测知识提升任务泛化。
- **V-JEPA**：由 Meta AI 提出的视频联合嵌入预测架构，通过自监督特征空间预测学习视觉表征，不重建像素。
- **Flow Matching**：一种基于连续流的最优传输生成建模方法，本文用于学习连续动作 chunk 的条件分布。
- **Mixture-of-Transformers (MoT)**：本文采用的架构，预测器与动作专家共享层数与 attention 维度但参数独立，通过联合注意力交互。
- **Future-Informed Context Interface**：预测器各层上下文位置的 key-value 状态集合，作为未来感知信息的中间表示，条件化动作专家。
- **3D RoPE**：将旋转位置编码扩展到三维（时间、高度、宽度），用于区分视觉隐空间中 token 的时空坐标。
- **LIBERO-Plus**：LIBERO 的扩展基准，包含七类受控分布偏移（视角、初始状态、语言、光照、纹理、噪声、布局），用于评估 WAM 鲁棒性。
- **DROID**：大规模野外机器人操作视频-指令数据集，本文用于无动作标签的预测器预训练。

## 可复现要素
- **代码**：已开源，地址 https://github.com/breez3young/VJEPA-Policy
- **数据集**：LIBERO、LIBERO-Plus、RoboCasa-GR1 均为公开基准；DROID 为公开野外数据集。
- **权重**：论文使用冻结的 V-JEPA 2.1 ViT-L/T5-XXL 编码器（官方预训练权重），预测器与动作专家为从头训练；论文未单独发布微调权重，但给出完整训练配方。
- **关键超参**：视觉编码器 ViT-L（304M）、预测器 24 层 hidden=1024、动作专家 24 层 hidden=512、attention head=16×64；LIBERO 训练 10 epoch（21,360 steps，batch=128），RoboCasa-GR1 训练 50k steps（batch=256），DROID 预训练 100k steps（batch=192），learning rate=1e-4，AdamW，cosine decay。

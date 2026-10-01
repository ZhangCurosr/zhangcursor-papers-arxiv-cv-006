---
title: "TRIANGULAR RESAMPLING FOR LONG-HORIZON MOTION GENERATION"
source: (unknown)
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:37:06"
field: "长周期文本到动作生成"
keywords: ["long-horizon motion generation", "diffusion model", "triangular denoising", "rollout training", "distribution matching", "training-inference mismatch", "human motion synthesis"]
innovations: ["提出 GT 钳位三角重采样（TR）以弥合三角去噪中部分去噪状态的训练–推理不匹配", "推导统一回放构造支持监督与分布匹配（TR-DMD）两种目标", "在 120 秒生成协议下于非 DMD 与 DMD 两组分别达到 SOTA FID AUC"]
benchmarks: ["HumanML3D 120-second evaluation (FID AUC, Matching Distance AUC, R-precision AUC and linear slopes)"]
---

# 论文速读：TRIANGULAR RESAMPLING FOR LONG-HORIZON MOTION GENERATION

## 一句话总结
本文提出三角重采样（Triangular Resampling, TR）后训练方法，通过为三角去噪调度中的部分去噪状态引入共享去噪阈值的 GT 钳位 rollout，缓解长周期文本到动作生成中因训练–推理状态不匹配导致的误差累积问题。

## 研究问题与动机
- **长周期生成的误差累积**：模型在短时 ground-truth 序列上训练，但推理时需反复基于自身预测生成 120 秒超长序列，微小误差会随时间传播放大。
- **三角去噪调度的状态不匹配**：FloodDiffusion 采用三角去噪，部分去噪状态（active window 内的未完成 token）在训练时直接由 GT 加噪构造，推理时却包含前置 Euler 更新累积的预测误差，标准训练未覆盖这一区域。
- **已有 rollout 训练的不足**：DART、MotionStreamer 等方法仅替换已完成的运动历史，未将 rollout 扩展到三角窗口内“正在去噪”的中间状态，无法弥合该部分不匹配。
- **无约束 rollout 的漂移风险**：完全自由 rollout 会使训练状态远离配对 GT，需要引入锚定机制防止分布过度偏移。

## 核心贡献（创新点）
1. **识别并形式化三角去噪中部分去噪状态的训练–推理不匹配**，指出仅需回放已完成历史不足以解决长周期误差累积，必须覆盖 active window 内的动态去噪过程。
2. **提出 GT 钳位三角重采样（TR）**：为每次回放采样一个共享阈值 $r$，去噪水平低于 $r$ 的状态强制使用噪声匹配的 GT 重建，高于 $r$ 则保留模型预测，从而在 GT 锚定与模型误差暴露之间取得可控平衡。
3. **推导出两种目标下的统一回放构造**：监督版本（TR）直接复用标准 flow-matching 损失；分布匹配版本（TR-DMD）结合 Rolling Forcing 的 DMD 更新配方，展示方法对多类优化目标的兼容性。
4. **在 HumanML3D 120 秒生成协议下建立系统性评测**：以 FID AUC 和线性退化斜率为核心指标，证明 TR 和 TR-DMD 分别在非 DMD 与 DMD 比较组内达到 SOTA，并给出详尽的超参消融与人工偏好评估。

## 方法详解
**三角去噪基础**
FloodDiffusion 使用线性 flow-matching 路径，全局去噪相位 $\tau$ 控制 latent 位置 $j$ 的干净系数 $\alpha_j(\tau)=\text{clip}(\tau-j/c,0,1)$。位置 $<m(\tau)$ 已干净并 commitment，$[m(\tau),n(\tau))$ 构成活跃去噪窗口，其余为噪声。每步 Euler 更新按实际 $\Delta\alpha_j$ 推进。

**GT-clamped 三角 rollout（第 4.1 节）**
- 回放概率 $\gamma$：每个 post-training 样本以概率 $\gamma$ 进入回放。
- 初始化：选定区间重置为高斯噪声，更早前缀保持干净。
- 共享阈值采样：$r=\text{sigmoid}(u+\log s)$，$u\sim\mathcal{N}(0,1)$，$s>0$ 控制钳制强度；同一回放内所有 token 与所有 Euler 步共用 $r$。
- 每步更新后钳位规则：
  $$
  \tilde{z}_j^{k+1} = 
  \begin{cases}
  \alpha_j^{k+1} z_j^{\text{GT}} + (1-\alpha_j^{k+1})\epsilon_j^{k+1}, & \alpha_j^{k+1}<r \\
  z_j^{k+1}, & \alpha_j^{k+1}\ge r
  \end{cases}
  $$
  低阈值处重新采样与当前三角噪声水平匹配的真实 latent，高阈值处保留模型预测；$\epsilon_j^{k+1}\sim\mathcal{N}(0,I)$ 为新生成噪声。
- 超参角色：$\gamma$ 控制回放频率，$s$ 控制从 GT 过渡到模型控制的程度；$s$ 越大越保守，$s\to\infty$ 退化为纯 GT 训练。

**优化目标**
- **监督 TR**：回放状态 $\tilde{z}$ 作为固定输入送入标准前向传播，从 $\tilde{z}$ 与配对 $z^{\text{GT}}$ 反推有效噪声 $\hat{\epsilon}_j$ 与目标速度 $v_j^\star=z_j^{\text{GT}}-\hat{\epsilon}_j$，损失为 flow-matching MSE：
  $$
  \mathcal{L}_{\text{TR}}(\theta)=\mathbb{E}\left[\frac{1}{|B(\tau)|}\sum_{j\in B(\tau)}\|v_{\theta,j}(\tilde{z},\alpha(\tau),c)-v_j^\star\|_2^2\right]
  $$
  $B(\tau)$ 为当前监督输出带。
- **TR-DMD**：沿用回放与钳位结构，但训练目标改为分布匹配。在允许模型控制的离散阶段随机采样一步，用梯度停止符号 $\text{sg}(\cdot)$ 记录输入 latent，再重新计算清洁预测：
  $$
  \hat{z}_{0,j} = \text{sg}(z_{\alpha_j}) + (1-\alpha_j)v_{\theta,j}(\text{sg}(z_\alpha),\alpha,c)
  $$
  基于这些 $\hat{z}_0$ 构建 fake sample，使用 Rolling Forcing 风格的 DMD 更新（冻结 real-score、可训练 fake-score，均来自 teacher EMA），generator 从监督 TR 检查点初始化。

## 实验与结果
- **数据集**：HumanML3D（训练+评测）、BABEL（训练）。使用官方 263 维表示、20 fps。
- **评测协议**：对 256 个冻结测试提示生成 120 秒序列，切分为 12 个不重叠 10 秒窗口，逐窗口计算 FID、Matching Distance、R-precision，报告归一化 AUC 与线性退化斜率；以 FID AUC 为主指标。
- **训练设置**：基于官方 FloodDiffusion  checkpoint，相同初始化、数据顺序、优化器与学习率调度，post-training 30k 步；TR-off 约 21 小时/GPU，主 TR $(\gamma=0.25,s=0.6)$ 约 29 小时/H200。
- **主要结果（Table 1）**：
  - 非 DMD 组：TR $(\gamma=0.25,s=0.6)$ 取得 **FID AUC=1.153**，较 TR-off（1.951）降低 **40.9%**；FID 斜率从 0.526 降至 **0.235/分钟**（降幅 55.3%）；Matching Distance AUC 3.786 vs 3.864；R-precision AUC 0.655 vs 0.648。
  - DMD 组：TR-DMD $(\gamma=1,s=0.6)$ 取得 **FID AUC=1.324**，优于 Rolling Forcing（1.351）；但相比同设置监督初始化（1.251）未进一步提升，说明 DMD 兼容性不带来额外保真收益。
  - 相对 Gaussian history noise $(\sigma=0.05)$：TR 将 FID AUC 再降 **18.2%**（1.410→1.153），体现结构化三角回放的优势。
- **消融（Table 2）**：
  - 阈值 $s$：全自 rollout（$s=0$）FID AUC 暴跌至 16.024；$s=0.6$ 最优（1.251）；$s\ge100$ 退化为 GT 训练（2.780）。
  - 回放比例 $\gamma$：非单调，$\gamma=0.25$ 最佳（1.153），$\gamma=1$ 次之（1.251），$\gamma=0.5$ 最差（1.495）。
  - Random-position clamp 与 curriculum 分别达 1.857、1.455，均不如固定 $s=0.6$ 连续前沿钳位。
- **人工评估（Table 3）**：30 提示两两偏好，TR 在动作质量（BT=0.302）与文本对齐（0.317）均居首，显著优于 TR-off 与 Rolling Forcing。

## 相关工作脉络
1. **FloodDiffusion**（Cai et al., 2026）：本文骨干网络，提出三角去噪调度与双向 active window 联合去噪；本文定位是在其训练分布上进行 post-training 修正，而非改进架构。
2. **DART**（Zhao et al., 2025）：基于 motion primitive 的阶段式 curriculum，从 GT 历史逐步过渡到全 rollout；差异在于 DART 替换已完成历史，TR 额外覆盖部分去噪的中间 latent 状态。
3. **MotionStreamer**（Xiao et al., 2025）：Two-Forward 策略，逐步用模型生成的 latent 替换历史 token；同样未处理三角窗口内“正在去噪”区域的动态误差。
4. **Self Forcing / Rolling Forcing**（Huang et al., 2025; Liu et al., 2026）：视频领域 DMD 方法，前者用 few-step 自 rollout，后者用交错噪声的滚动窗口；TR-DMD 借鉴其 DMD 配方但适配三角单 token commit 结构。
5. **Resampling Forcing**（Guo et al., 2025）：自教师-free 自重采样，对噪声污染的 GT 帧做自回归重采样；相似点是 detached 模型状态+GT 监督，区别是 TR 引入共享阈值与三角时序前沿。
6. **TEACH / DoubleTake / FlowMDM / T2LM**：长周期动作生成其他路线（分段组合、重叠精炼、位置编码、潜在合成）；本文不参与架构竞争，专注训练–推理状态对齐这一共性难题。

## 局限性与未来方向
- **计算开销**：多步回放使单步耗时增至约 3.3×（0.58s→1.95s/update），长周期后训练成本显著上升；作者建议借鉴 Self Forcing 的短随机 rollout、采样步监督与梯度截断来降低开销。
- **单一骨干与调度**：当前仅在 FloodDiffusion 三角调度上验证，未推广至其他 backbones 或不同去噪 schedule。
- **DMD 收益有限**：TR-DMD 在 FID AUC 上未超越监督 TR，说明在该任务与协议下分布匹配可能带来边际增益，需进一步探索何时 DMD 能发挥更大作用。
- **超参敏感性**：阈值分布形状（固定 $s$ vs curriculum vs random-position）对性能有影响，如何自动学习或自适应调度未深入讨论。

## 研究启发与可借鉴点
1. **训练–推理状态对齐的系统化思路**：识别“部分去噪中间状态”这一易被忽视的不匹配源，提示在任意自回归/滚动窗口模型中均需检查 active region 的分布偏移，而非仅关注已 commit 历史。
2. **共享阈值 GT 钳位机制**：用单一标量 $r$ 控制整条 rollout 轨迹中 GT 与模型的切换 frontier，实现连续前沿并保持因果性；该设计可迁移至视频扩散的 Rolling Forcing 类方法。
3. **非单调回放比例的发现**：$\gamma=0.25$ 优于 $\gamma=1$，提示“少而精”的自 rollout 暴露可能比全覆盖更有利于稳定训练，可作为后续调参的先验。
4. **适配运动域的统一评测协议**：120 秒序列+12 窗口分段指标+AUC/斜率双汇总，兼顾整体保真与时间退化趋势，值得在其它长周期生成任务中复用。
5. **TR-DMD 兼容架构**：同一回放构造可无缝切换监督/DMD 目标，为后续混合训练策略（如先监督预热再 DMD 精炼）提供模块化基础。

## 关键术语表
- **Triangular Resampling (TR)**：一种后训练技巧，通过共享去噪阈值的 GT 钳位回放三角去噪窗口，使模型在训练中接触部分去噪状态下的自身预测误差。
- **FloodDiffusion**：基于三角去噪调度与双向 active window 联合去噪的长周期运动扩散模型骨干。
- **GT clamp**：在 rollout 过程中，当 token 的去噪水平低于阈值时强制用噪声匹配的 ground-truth latent 替换模型输出，起到锚定作用。
- **Rollout**：用当前模型不带梯度地逐步推进多步去噪，生成一段包含模型预测误差的 latent 轨迹。
- **Distribution Matching Distillation (DMD)**：通过 real/fake score 网络对齐生成分布与教师分布的训练框架，此处用于 TR-DMD。
- **Rolling Forcing**：视频生成中的 DMD 变体，使用交错噪声的滚动窗口与 attention sink 保留机制进行更新。
- **FID AUC**：沿时间窗口计算的 FID 曲线下的归一化面积，用于汇总长周期生成的整体保真度。
- ** Bradley–Terry (BT) 评分**：基于成对偏好投票拟合的 log-strength 得分，用于量化人工评估中各方法的相对优劣。

## 可复现要素
- **数据集**：HumanML3D（训练与评测）、BABEL（训练）；评测使用官方 test split 中的 256 个冻结提示。
- **代码/权重**：论文声明将公开实现、训练配置、冻结提示清单、checkpoint 标识符与评测脚本（见 Reproducibility Statement）；目前基于官方 FloodDiffusion checkpoint 与预训练 VAE。
- **关键超参**：
  - 监督 TR：$\gamma=0.25$，$s=0.6$，30k post-training 步。
  - TR-DMD：$\gamma=1$，$s=0.6$，generator 学习率 $1.5\times10^{-6}$，fake-score 学习率 $4\times10^{-7}$，AdamW $\beta=(0,0.999)$，无 weight decay，score-time shift 5，梯度裁剪 10，generator EMA 0.99（从 200 iter 起），DMD 1,200 outer iter（1,200 fake-score update + 240 generator update）。
  - 窗口/调度：$c=5$，每单位相位 $N=10$ 次 Euler 步，每 2 步 commit 一个 token。
  - DMD 评分窗口：最多使用序列末尾 35 个有效 latent。

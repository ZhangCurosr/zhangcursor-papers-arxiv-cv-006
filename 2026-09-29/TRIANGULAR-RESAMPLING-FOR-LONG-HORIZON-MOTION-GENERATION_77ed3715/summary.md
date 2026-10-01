---
title: "TRIANGULAR-RESAMPLING-FOR-LONG-HORIZON-MOTION-GENERATION"
source: https://arxiv.org/pdf/2609.34697v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:36:50"
field: "长程文本到动作生成"
keywords: ["long-horizon motion generation", "triangular denoising", "rollout training", "diffusion model", "train-inference mismatch", "distribution matching distillation"]
innovations: ["提出 Triangular Resampling，将 rollout 训练扩展到三角形去噪窗内所有部分去噪状态，通过共享阈值 + GT clamp 控制漂移", "将 DMD 与三角形 replay 结合（TR-DMD），在逐 token commit 结构下实现分布匹配", "建立 120 秒序列 + 12 段 10 秒窗口 + FID AUC/退化斜率的统一长程评估协议"]
benchmarks: ["HumanML3D"]
---

# 论文速读：TRIANGULAR RESAMPLING FOR LONG-HORIZON MOTION GENERATION

## 一句话总结
论文提出 Triangular Resampling（TR），一种针对 FloodDiffusion 三角形去噪调度的**后训练方法**，通过在训练时对部分去噪状态进行模型自生成 rollout 并用去噪阈值 + GT clamp 控制漂移，缓解长程生成中误差累积问题。在 HumanML3D 上，TR 将 120 秒生成的 FID AUC 从 1.951 降至 1.153（-40.9%），在各自 DMD/非 DMD 比较组中均取得 SOTA。

## 研究问题与动机
- **训练-推理不匹配**：FloodDiffusion 训练时主动窗口状态直接由 GT 加噪构造，推理时却反复以模型自身预测为条件，微小误差沿时间累积。
- **既有 rollout 训练未覆盖"部分去噪"状态**：已有方法（DART、MotionStreamer 等）仅用已完成的历史替换，但三角形调度中活跃窗口内大量 token 处于**半去噪**阶段，同样会携带旧预测误差并影响后续更新。
- **自由 rollout 会严重漂移**：消融显示全自 rollout（无 GT clamp）FID AUC 高达 16.024，表明必须在 GT 锚定与模型误差暴露之间取得平衡。
- **评估协议缺失**：多数工作仅评估动作片段/过渡，缺乏对"同一文本提示下持续生成质量衰减"的量化刻画；本文提出 12 段 10 秒滑动窗口 + FID AUC / 退化斜率。

## 核心贡献（创新点）
1. **提出 Triangular Resampling (TR)**：将 rollout 训练扩展到三角形窗内全部状态（含部分去噪），通过共享去噪阈值 + GT clamp 控制漂移；与仅替换已完成历史的 DART/MotionStreamer 相比，覆盖了活跃的 denoising frontier。
2. **设计噪声匹配的 GT clamp 机制**：按当前三角形噪声水平重新采样 GT，保留与监督目标一致的噪声分布；与 Gaussian history noise 等粗暴加噪方案相比，维持了正确的 latent 分布。
3. **兼容两种训练目标（TR 与 TR-DMD）**：TR 保留原始 flow-matching 监督损失，TR-DMD 在此基础上引入 DMD 分布匹配；两者共用同一 replay 构造，扩展了方法适用范围。
4. **建立统一的长程评估协议**：以 120 秒序列 + 12 个非重叠 10 秒窗口 + FID AUC / 线性退化斜率为核心指标，使不同方法在同一时间维度下可比。
5. **系统消融揭示关键超参规律**：阈值 shift s=0.6 为最优固定值，γ=0.25 优于 γ=1 和 γ=0.5（非单调）， Curriculum 和 Random-position clamp 均不及固定阈值连续前沿。

## 方法详解
- **基座**：FloodDiffusion，三角形去噪调度：全局相位 τ，token j 的干净系数 α_j(τ)=clip(τ−j/c,0,1)，活跃窗口为 [m(τ), n(τ))，c=5、N=10，每 2 步 commit 一个 token。
- **Triangular Resampling replay 构造**：以概率 γ 选择一批次进入 replay；将该批次重置为零相位高斯噪声，以当前模型沿三角形调度做 K 步 Euler 更新（不追踪梯度）。
- **共享阈值采样**：每个 replay 样本采样一个全局阈值 r=σigmoid(u+log s)，u~N(0,1)，s 控制 clamp 强度；同一 r 对所有 token 和所有 Euler 步共享，保证**连续前沿**。
- **GT clamp 规则**（公式 7）：更新后若 α_j^{k+1}<r，则 $\tilde{z}_j^{k+1}=\alpha_j^{k+1} z_j^{GT}+(1-\alpha_j^{k+1})\epsilon_j^{k+1}$（噪声匹配的 GT 重新采样）；若 α_j^{k+1}≥r，保留模型预测 $\tilde{z}_j^{k+1}=z_j^{k+1}$。
- **监督损失（TR）**：从 replay 状态反算有效噪声 $\hat{\epsilon}_j=(\tilde{z}_j-\alpha_j z_j^{GT})/\max(1-\alpha_j,\epsilon)$，得到速度目标 $v_j^\star=z_j^{GT}-\hat{\epsilon}_j$，优化 $\mathcal{L}_{TR}=\mathbb{E}[\|v_\theta(\tilde{z},\alpha,c)-v^\star\|_2^2]$，在输出带 B(τ) 上计算。
- **TR-DMD**：在 replay 基础上选取一个离散阶段，对 clean prediction 做 $\hat{z}_{0,j}=sg(z_{\alpha_j})+(1-\alpha_j)v_\theta(sg(z_\alpha),\alpha,c)$，再以 Rolling Forcing 式 DMD 更新（frozen real-score + trainable fake-score）。评分窗口取最后 35 个合法 latent。
- **超参**：默认 (γ,s)=(0.25,0.6)；s=3 时 log-shift 更大，接近 GT 训练；s→∞ 趋近纯监督；s=0 退化为全自 rollout（效果极差）。

## 实验与结果
- **数据集**：HumanML3D + BABEL 训练；HumanML3D 测试集 256 提示，生成 120 秒（20fps，263-dim）。
- **评估协议**：12 个 10 秒非重叠窗口，报告 FID / Matching Distance / R-precision 的 AUC 与线性退化斜率。
- **主结果（非 DMD 组，Table 1）**：
  - TR (γ=0.25, s=0.6)：**FID AUC 1.153**，slope 0.235/min；MD AUC 3.786；R-prec AUC 0.655。
  - TR-off：FID AUC 1.951，slope 0.526/min；**TR 相对提升 40.9%**，slope 下降 55.3%。
  - Gaussian noise (σ=0.05)：FID AUC 1.410 → TR 相对再降 18.2%，证明结构化 triangular replay 优于粗粒度 history 加噪。
  - DART / MotionStreamer / Resampling Forcing 均显著落后（FID AUC 13.192 / 7.376 / 3.407）。
- **主结果（DMD 组，Table 1）**：
  - TR-DMD (γ=1, s=0.6)：**FID AUC 1.324**，优于 Rolling Forcing 的 1.351，为 DMD 组 SOTA。
  - Self Forcing 1.520；Self Gradient Forcing 4.005；Causal + Rolling 1.634。
- **Human evaluation（Table 3）**：TR 在 BT 得分上 motion quality 0.302、text alignment 0.317，均位列第一，分别领先 TR-off 约 0.57/0.53 分。
- **消融（Table 2）**：
  - 阈值 s 最优在 0.6；s=0（全 rollout）FID AUC 暴增至 16.024。
  - 非单调 γ：0.25 最优（1.153）< 1（1.251）< 0.5（1.495）。
  - Random-position clamp（1.857）和 Curriculum（3→0.6，1.455）均不及固定 s=0.6（1.251）。

## 相关工作脉络
- **DART (Zhao et al., 2025)**：基于运动原语的 curriculum rollout（GT→混合→全 rollout），但其单元是完整 primitive，不处理三角形窗口内的部分去噪状态。
- **MotionStreamer (Xiao et al., 2025)**：Two-Forward 训练中逐步用模型预测替换历史 token，同样仅覆盖已完成部分，不涉及活跃 denoising frontier。
- **Resampling Forcing (Guo et al., 2025)**：对视频做 teacher-free 自 resampling，保留原始 flow-matching 损失；本文与其相似但面向三角形调度、共享阈值 + 连续前沿。
- **Self Forcing / Rolling Forcing (Huang et al. 2025; Liu et al. 2026)**：视频领域 DMD 方法，以 multi-latent chunk 为单位；本文将其 DMD recipe 移植到逐 token commit 的 FloodDiffusion 结构。
- **Self Gradient Forcing (Zhuang et al., 2026) / Causal Forcing (Zhu et al., 2026)**：视频 DMD 变体，依赖 teacher ODE/AR 初始化；本文 TR-DMD 无需额外 teacher。
- **PRISM v1 (Ling et al., 2026)**：self-forcing + 分布匹配的流式生成，但采用 per-joint latent decomposition，与本文 joint triangular 调度正交。

## 局限性与未来方向
- **计算开销**：TR 每步约 1.95s（vs TR-off 0.58s，~3.3×），主要源于多步 detached replay。
- **短 rollout + 采样步监督 + 截断梯度**（受 Self Forcing 启发）被作者建议用于降低开销。
- **仅验证 FloodDiffusion 基座**：作者明确说明未拓展到其他 backbone、其他去噪调度及更多后训练数据配比。
- **TR-DMD 未超越监督 TR**：说明该任务下 DMD 增益有限，需进一步探索。
- **阈值固定策略**：虽然固定 s=0.6 最优，但未探索更复杂的 Curriculum（如与 prompt 难度相关）。

## 研究启发与可借鉴点
- **"部分去噪状态"的误差建模思路可迁移**：任何带有 staggered-noise 或 active-window 结构的生成模型（如流式视频扩散）均可参考"沿调度前沿做 replay + clamp"的思路。
- **共享阈值 + 连续前沿设计简洁且有效**：避免了逐 token 独立阈值的复杂优化，且消融证明优于 random-position clamp 与 Curriculum，是实用的工程选择。
- **噪声匹配的 GT clamp 保持分布一致**：直接复用当前 α level 对 GT 加噪而非简单 copy GT，避免监督信号偏移，适用于任何 flow-matching 场景。
- **长程评估协议（AUC + 斜率）可作为后续工作的标准 benchmark**，便于跨方法公平比较衰退行为。
- **TR-off 作为 same-backbone 对照的价值**：证明 FloodDiffusion 基座本身已强于多数同领域方法，凸显 TR 的贡献是在强基座上进一步弥合 train-inference gap。

## 关键术语表
- **Triangular Denoising Schedule**：FloodDiffusion 的调度方式，活跃窗口内各 token 的干净系数 α 随位置线性递增，形成"三角"形状。
- **Triangular Resampling (TR)**：本文提出的后训练技术，沿三角形调度 replay 多步 Euler 更新并以共享阈值 + GT clamp 控制漂移。
- **GT Clamp**：将低于阈值的状态替换为噪声匹配的 GT 重新采样，防止 rollout 过度偏离真实数据流形。
- **Shift Parameter s**：阈值采样分布 logit-normal 的平移参数，控制 GT 锚定的强度（s=0.6 为本实验最优）。
- **FID AUC**：沿 12 个 10 秒窗口对 FID 值积分得到的归一化曲线下面积，衡量整体长程生成保真度。
- **FID Degradation Slope**：FID 随时间（分钟）的线性退化斜率，衡量长程生成质量的衰减速度。
- **TR-DMD**：将 TR replay 与 Distribution Matching Distillation 结合的训练变体，使用 Rolling Forcing 风格的 DMD 更新。
- **Rollout**：以当前模型多步更新生成的 detached 状态序列，作为后续训练 step 的输入。

## 可复现要素
- **数据集**：HumanML3D（公开）+ BABEL（公开）；论文使用官方 263-dim 表征、20 fps。
- **代码/权重**：论文声明"将公开实现、训练配置、冻结提示清单、checkpoint 标识与评估脚本"（Reproducibility Statement）；未提及开源仓库链接。
- **关键超参**：γ=0.25, s=0.6（监督 TR 最优）；TR-DMD 使用 γ=1, s=0.6；c=5, N=10；监督训练 30k 步；DMD 阶段 1,200 轮外层迭代（generator EMA decay=0.99, lr_g=1.5e-6, lr_f=4e-7）。
- **硬件**：单卡 NVIDIA H200（监督 TR，~29h）；双卡 H200（TR-DMD，~13h）。

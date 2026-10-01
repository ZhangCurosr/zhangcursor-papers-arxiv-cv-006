---
title: "Targeted-Visual-Counterfactual-Explanations-for-Contrastive"
source: https://arxiv.org/pdf/2609.37638v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 11:13:35"
field: "视觉-语言模型可解释性"
keywords: ["反事实解释", "CLIP", "扩散模型", "视觉解释", "零样本分类", "inpainting", "Attribution"]
innovations: ["首个专为CLIP零样本分类设计的图像级自适应掩码反事实解释方法", "结合CLIP语义引导与Latent Diffusion Inpainting实现最小化靶向编辑", "揭示source-mask与difference-mask在有效性-保真度之间的系统性权衡"]
benchmarks: ["ImageNet-1k", "Food-101", "Oxford Pets", "CUB-200"]
---

# 论文速读：Targeted-Visual-Counterfactual-Explanations-for-Contrastive

## 一句话总结
本文提出 **MACE（Mask-guided Adaptive Counterfactual Explanations）**，是首个专为 CLIP 零样本分类设计的图像级反事实解释方法，通过自适应 Attribution Mask 选取可编辑区域，结合冻结 CLIP 的语义引导与 Latent Diffusion Inpainting，实现以最小改动将预测切换至目标类别。

## 研究问题与动机
1. **现有 CLIP 解释方法的局限**：CLIP 现有解释手段（注意力图、梯度图）仅能高亮重要区域，却无法展示"如何修改输入才能使模型输出目标类别"。
2. **现有视觉反事实方法不适配 CLIP**：主流视觉反事实方法针对传统 CNN 分类器优化类别 logits，而 CLIP 的零样本分类基于跨模态 cosine similarity，直接套用不适用。
3. **CLIP 引导的图像编辑方法目标不同**：StyleGAN-NADA、DiffusionCLIP 等以文本对齐为优化目标，不保证目标类成为 CLIP 全类别排序中的 top-1，也不追求最小局部改动。
4. **需要一种专门适配 CLIP 零样本分类的反事实生成机制**，在保持源图非编辑区域的同时，以最小语义修改实现预测转换。

## 核心贡献（创新点）
1. **首个面向 CLIP 零样本分类的图像级反事实方法**：MACE 联合使用 CLIP Attribution 定位可编辑区域与 CLIP 决策引导的扩散 Inpainting，而非仅依赖文本提示对齐。
2. **自适应掩码扩展机制**：从最小的可编辑区域（top 25% Attribution 值）出发，以 5% 步长逐步扩展 Mask，仅在必要时扩大编辑范围直至目标类别达成，保证改动最小化。
3. **CLIP 语义引导的扩散 Inpainting 策略**：在每一步去噪中，解码估计的 clean latent，以 composite 图像（masked 区域用生成结果、其余区域保留原图）计算 source-target 分数差并限制梯度至可编辑 latent 位置，兼顾有效性与原图保真。
4. **系统性揭示有效性–保真度的权衡关系**：Source-mask 变体在四个数据集上均获得最高 target top-1 成功率，Difference-mask 变体始终产生最小的像素/感知变化与最佳真实感得分。

## 方法详解
**整体流程**：输入源图像 $x$ 和目标类别 $y_t$ → CLIP Attribution 生成可编辑 Mask → 自适应扩展 → CLIP 引导的 Latent Diffusion Inpainting 生成反事实图像 $x_{cf}$。

**关键步骤**：

1. **Attribution 算子**：使用 CAV（Class Activation Values）作为标签条件归因算子 $\Phi(x, s_y) \to A_y$，得到两个 relevance score：
   - 源掩码：$R_{src} = A_s$（支持当前预测的区域）
   - 差值掩码：$R_{diff} = \text{ReLU}(A_s - A_t)$（源类证据强于目标类的区域）

2. **自适应掩码构造**：按阈值选取 Top $r_k$ 比例的 attribution 值生成二值 mask $M_k$，初始 $r_0 = 0.25$，每次以 5% 步长扩展，最多 20 次。选取第一个使 $\hat{y}(x_{cf}^{(k)}) = y_t$ 的最小成功 mask；若无成功则选 target-vs-competitor margin 最大的候选。

3. **CLIP 引导的扩散 Inpainting**：
   - 使用 Stable Diffusion v1.5 inpainting checkpoint，VAE 编码得到 latent $z_I$
   - Mask 投影到 latent 分辨率得 $M_z$，反向扩散初始化：mask 内用独立噪声，mask 外用加噪后原图 latent
   - Denoiser 以目标 prompt 的 conditioning embedding $h_{y_t}$ 为文本条件，拼接 $z_t$、$\widetilde{M}$、$z_{masked}$
   - **CLIP 语义引导**：每步解码 $\hat{x}_0^{(t)}$，构建 composite $\tilde{x}^{(t)} = M \odot \hat{x}_0^{(t)} + (1-M) \odot x$，最小化损失：
     $$\mathcal{L}_{CLIP}^{(t)} = s_{y_s}(\tilde{x}^{(t)}) - s_{y_t}(\tilde{x}^{(t)})$$
   - 梯度限制在 $M_z$ 可编辑位置，受保护位置保留参考 latent 轨迹

## 实验与结果
**数据集**：ImageNet-1k（验证集，934 对）、Food-101（1,003 对）、Oxford Pets（1,194 对）、CUB-200（1,595 对），共 4,726 个反事实任务。

**评估指标**：Validity（target top-1 成功率）、Proximity（$\ell_1^{norm}$、$\ell_2^{norm}$、LPIPS、SSIM）、Realism（FID、KID）。

**主要结果**：

| 数据集 | 方法 | Validity ↑ | $\ell_1^{norm}$ ↓ | FID ↓ |
|--------|------|-----------|-------------------|-------|
| ImageNet | MACE_s | **0.9700** | 0.0742 | 49.80 |
| ImageNet | MACE_d | 0.7805 | **0.0407** | **44.84** |
| ImageNet | SD-only | 0.7195 | 0.3346 | 84.17 |
| Food-101 | MACE_s | **0.9990** | 0.0554 | 23.57 |
| Food-101 | MACE_d | 0.8744 | **0.0402** | **22.33** |
| Food-101 | SD-only | 0.9701 | 0.3437 | 151.40 |
| Oxford Pets | MACE_s | **0.9925** | 0.0615 | 33.65 |
| Oxford Pets | MACE_d | 0.7730 | **0.0361** | **25.41** |
| Oxford Pets | SD-only | 0.8970 | 0.3207 | 126.15 |
| CUB-200 | MACE_s | **0.9455** | 0.0586 | 14.70 |
| CUB-200 | MACE_d | 0.8696 | **0.0302** | **9.06** |
| CUB-200 | SD-only | 0.5034 | 0.2804 | 51.24 |

**关键结论**：
- MACE_s 在全部四个数据集上获得最高 validity（最优提升：CUB-200 相对 SD-only 从 50.34% → 94.55%）
- MACE_d 在全部四个数据集上获得最低的 $\ell_1$、$\ell_2$、LPIPS 和最高的 SSIM、FID、KID
- 两变体均显著优于仅用 Stable Diffusion inpainting 的 baseline（以 ImageNet 为例，$\ell_1^{norm}$ 从 0.3346 降至 0.0742 / 0.0407）
- 存在 validity–proximity 权衡：MACE_s 有效性最高但改动更大，MACE_d 改动最小但有效性较低

## 相关工作脉络
1. **DiME / DVCE**：针对传统 CNN 分类器的扩散反事实方法，优化类别 logits 和相似性正则化，需大量适配才能用于 CLIP，本文直接适配 CLIP 的 cosine similarity 分数空间。
2. **CLIP Surgery / SpLiCE / gradient-based attribution**：观测性解释方法，能定位重要区域但不回答"修改该区域是否能改变预测"的因果问题。
3. **DiffusionCLIP / StyleCLIP / CF-CLIP**：以文本对齐或语义编辑为目标，不要求目标类成为 CLIP 零样本排名的 top-1，也不追求最小局部修改。
4. **CF-CLIP**：引入对比目标实现更局部化编辑，但优化目标是 prompt alignment 而非 top-1 预测转换。
5. **CPL / COMO / CF-VLM**：在训练阶段使用反事实样本增强模型能力，不针对单个预测提供图像级反事实解释。
6. **Goyal et al. (2019)**：早期通过替换判别区域生成反事实的方法，本文使用扩散先验实现更高真实感。

## 局限性与未来方向
1. **当前仅优化 pairwise source–target 分数差**，不保证目标类战胜所有其他类别；未来可采用多分类或 margin-based 目标直接优化 top-1。
2. **不同 mask fraction 产生独立的生成结果而非增量扩展**，固定 mask 策略下 mask 增大未必单调提升 validity（如 CUB-200 上 source mask），联合优化 mask size 与 CLIP 引导强度值得探索。
3. **强 CLIP 引导虽提升有效性并降低 $\ell_1$ 距离，但可能引入肉眼可见的 artifacts**，现有 $\ell_1$ 指标无法捕捉此类质量问题。
4. **细粒度分类（如 CUB-200）失败案例较多**：差值 mask 可能过小无法添加足够目标证据，或源类强证据在编辑后仍然残留。
5. **最大 mask 尺寸仍有限制**，未来可探索更大的最大 mask 上限或更强的类别特定引导。

## 研究启发与可借鉴点
1. **自适应 mask 扩展策略可迁移**：从最小可编辑区域出发逐步扩展直至满足条件的策略，适用于其他需要平衡"最小改动"与"目标达成"的生成式解释方法。
2. **CLIP 作为外部决策引导信号的设计**：在 diffusion denoising 过程中穿插 CLIP 分数差的梯度更新，并将梯度严格限制在 mask 区域内，这一设计可推广至其他 VLM 的反事实解释场景。
3. **Difference attribution mask 的设计思路**：$R_{diff} = \text{ReLU}(A_s - A_t)$ 直接定位"源类相对于目标类的判别证据"，比单纯使用源类 attribution 更精准，可作为通用的对抗性区域定位策略。
4. **复合图像的 CLIP 评估**：用 composite $\tilde{x} = M \odot \hat{x}_0 + (1-M) \odot x$ 而非完全生成的图像来评估 CLIP 分数，使得引导信号更准确地反映"部分编辑"对决策的影响。
5. **与团队方向的结合机会**：可将 MACE 的 CLIP 引导 Inpainting 机制迁移至多模态大模型的可视化解释、对抗鲁棒性分析，或与 CF-VLM 等训练时反事实生成方法结合形成端到端框架。

## 关键术语表
**MACE**：Mask-guided Adaptive Counterfactual Explanations，本文提出的面向 CLIP 零样本分类的靶向视觉反事实解释方法。
**CLIP Attribution（CAV）**：使用 Class Activation Values 作为标签条件归因算子，量化图像各空间位置对 CLIP 某类别分数 $s_y(x)$ 的贡献。
**Source–Target Difference Mask**：$R_{diff} = \text{ReLU}(A_s - A_t)$ 构建的掩码，仅选择源类证据显著强于目标类的区域用于编辑。
**Adaptive Mask Expansion**：从 top 25% attribution 值出发，以 5% 步长逐步扩展可编辑 mask 比例，直到目标类别达成或达到最大尝试次数。
**CLIP Semantic Guidance**：在扩散去噪的每一步，通过 CLIP source-target 分数差对可编辑 latent 位置施加梯度修正，确保生成内容指向目标类别。
**Proximity**：衡量反事实图像与源图像的相似度，包括 $\ell_1$、$\ell_2$、LPIPS（越低越好）和 SSIM（越高越好）。
**Validity**：反事实图像的 target top-1 成功率，即 CLIP 零样本分类是否输出目标类别。
**Composite Image**：编辑区域内用扩散生成结果、其余区域保留原图的合成图像，用于每一步 CLIP 引导评估。

## 可复现要素
- **数据集**：ImageNet-1k（官方验证集）、Food-101（官方测试集）、Oxford-IIIT Pet（官方测试集）、CUB-200-2011（官方测试集），均为公开数据集。
- **代码开源**：论文声明源代码已开源，地址 https://anonymous.4open.science/r/MACE-04BC/（匿名审稿链接）。
- **权重**：使用 OpenAI CLIP ViT-B/16 和 Stable Diffusion v1.5 inpainting checkpoint，均为公开权重。
- **关键超参**：初始 mask 比例 $r_0 = 0.25$，扩展步长 5%，最多 20 次尝试；使用 CAV 作为归因算子；CLIP guidance scale 在消融实验中从 0 开始变化。

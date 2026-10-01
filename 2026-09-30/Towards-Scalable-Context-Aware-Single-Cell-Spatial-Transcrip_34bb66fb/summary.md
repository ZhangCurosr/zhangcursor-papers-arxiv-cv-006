---
title: "Towards-Scalable-Context-Aware-Single-Cell-Spatial-Transcrip"
source: https://arxiv.org/pdf/2609.36429v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 11:15:44"
field: "计算病理与空间组学"
keywords: ["spatial transcriptomics", "single-cell prediction", "pathology foundation model", "histology image", "grid sampling", "cross-attention", "computational pathology"]
innovations: ["单次PFM前向+可微grid sampling实现所有细胞的位置特异特征提取，消除per-cell前向瓶颈", "距离衰减交叉注意力模块（2D RoPE+Gaussian bias）为每个细胞注入空间局部形态上下文", "端到端单细胞监督训练，避免spot-level弱监督不适定解耦，并揭示全局CLS token不利于细粒度空间预测"]
benchmarks: ["10X-Xenium-52 (HEST-1k)", "HVG/MPG/SVG top-k PCC", "ID/OOD cross-organ generalization"]
---

# 论文速读：Towards-Scalable-Context-Aware-Single-Cell-Spatial-Transcriptomics-Prediction-from-Histology-Images

## 一句话总结
本文提出 CELLO，首个可扩展的端到端单细胞水平 H&E 图像→空间转录组基因表达预测框架：一次病理基础模型前向传播即可通过 grid sampling 为同一图像内所有细胞同步提取特征，并用距离衰减交叉注意力注入空间局部形态上下文，在 52 对 Xenium–H&E 样本（覆盖 12 个器官、约 1000 万细胞）上达到 SOTA 预测精度，且推理速度比 DeepSpot2Cell 平均快 14.0×（不含上游细胞分割）。

---

## 研究问题与动机
- **现有方法多为 spot 级**：Visium 等平台每 spot 聚合数十个细胞信号，掩盖了细胞类型异质性、稀有细胞状态与细胞间互作，难以满足单细胞分辨率需求。
- **单细胞扩展面临两大困难**：① 直接裁剪单个细胞并独立做 PFM 前向（如 DeepSpot2Cell）计算量随细胞数线性增长，在百万级细胞 WSIs 上不可扩展；且裁剪+resize 会畸变细胞形态并丢失微环境上下文。② 基于分割 mask 的 UNet 类方法（如 GHIST）不依赖 PFM per-cell 前向，但缺乏现代病理基础模型学到的强形态表征能力，且误差会沿上游分割级联。
- **scale mismatch 本质**：PFM patch-level 表征天然混合多细胞，per-cell 独立编码与 pretraining 的 patch 对齐范式相悖；如何在保留 PFM 强视觉表征的同时实现逐细胞精确定位，是关键未解问题。
- **全局上下文不一定有用**：作者实验发现 CLS token（整张 slide 的全局摘要）反而显著降低性能（ID 平均下降 13.6%），说明单细胞基因表达预测主要由细粒度局部形态驱动，而非高层全局语义。

---

## 核心贡献（创新点）
1. **单通 PFM + grid sampling 的 scalable cell-query 机制**：每个 WSI 仅做一次 pathology foundation model 前向，再通过可微 bilinear grid sampling 在所有细胞坐标处同时抽取位置特异特征，消除 DeepSpot2Cell 式的 per-cell N 次前向瓶颈。与已有工作的本质区别在于把 PFM 输出当作连续 2D feature map 进行 sub-patch 精度的查询，而非离散裁剪。
2. **距离衰减交叉注意力模块（distance-decay cross-attention）**：在 cross-attention 中叠加 2D RoPE 编码相对空间关系 + Gaussian 距离衰减偏置 $b_{i,m} = -\|s_i^{loc}-c_m\|^2/(2p^2)$，使每个细胞优先关注邻近 patch token，注入生物合理的空间归纳偏置。相比 GHIST 等 mask-based 方法，该方法不依赖精确细胞边界且充分复用 PFM 表征。
3. **端到端单细胞监督训练范式**：直接在配对的单细胞 ST（Xenium）与 H&E 上训练，避免 DeepSpot2Cell 的弱 spot-level supervision 导致的不适定细胞profile解耦；同时引入辅助 patch-level Huber 损失（$\lambda=0.5$）以稳定训练。
4. **系统性的 encoder 对比与上下文消融**：在 UNI / Virchow2 / H-Optimus-0 三大 PFM 上定量对比，确认 Virchow2 最优（MPG HVG SVG PCC-50 分别领先 H0 28.9%/14.7%/28.1%）；并通过 CLS fusion 门控实验得出"局部形态优先于全局语义"的设计原则。
5. **公开的 10X-Xenium-52 benchmark 与开源基线**：基于 HEST-1k 构建涵盖 12 器官、约 10M 细胞的基准，包含严格的 ID/OOD 划分（4 个未见器官各 1 slide，Kidney/Liver 见同器官不同病程），并公开代码、权重与数据，推动该方向可复现。

---

## 方法详解
**整体架构**：$f_\theta: (\mathbf{V}, \mathbf{s}_i) \mapsto \hat{\mathbf{x}}_i \in \mathbb{R}_{\ge 0}^G$，其中 $\mathbf{V}$ 为 $224\times224$ H&E patch，$\mathbf{s}_i$ 为细胞在全 slide 像素坐标下的 centroid。

1. **视觉特征提取**：冻结的病理基础模型 $f_{\mathrm{enc}}$（默认 Virchow2）将 $\mathbf{V}$ 划分为 $p\times p$ 非重叠子 patch，输出 $M=L_g^2$ 个空间 token 组成 2D 特征图 $\mathbf{T}\in\mathbb{R}^{L_g\times L_g\times d}$（Virchow2: $p=14, d=1280$）。CLS token $\mathbf{z}_{\mathrm{cls}}$ 被用于 CLS-fusion 消融但不进入最终 CELLO。

2. **位置感知 Cell Query（Grid Sampling）**：
   - 将细胞全局坐标转换到 patch-local 坐标系：$\mathbf{s}_i^{\mathrm{loc}}=\mathbf{s}_i-\mathbf{o}$。
   - 归一化到 $[-1,1]$：$\tilde{\mathbf{s}}_i=2\mathbf{s}_i^{\mathrm{loc}}/L-1$。
   - 可微 bilinear interpolation 从 $\mathbf{T}$ 中采样：$\mathbf{q}_i^{(0)}=\mathtt{GridSample}(\mathbf{T},\tilde{\mathbf{s}}_i)\in\mathbb{R}^d$，实现 sub-patch 级连续位置查询。

3. **Cross-Attention with 2D RoPE + Distance Decay**：
   - 将 $\mathbf{q}_i^{(0)}$ 作为 query，对所有 spatial token 做 multi-head cross-attention。
   - **2D RoPE**：对 query/key 应用旋转位置编码，细胞坐标除以 token stride 转为连续 token-grid 坐标，与视觉 key 的 grid 坐标对齐。
   - **距离衰减偏置**：$b_{i,m}=-\|s_i^{\mathrm{loc}}-c_m\|^2/(2p^2)$，$c_m$ 为第 $m$ 个 token 的中心像素坐标；$p^2$ 归一化使尺度相对 token 感受野。
   - LayerNorm 后得到细胞嵌入 $\mathbf{z}_i$。

4. **Gene Expression Head**：两层 ReLU MLP $h_{\mathrm{cell}}$ 将 $\mathbf{z}_i$ 解码为 $\hat{\mathbf{x}}_i\in\mathbb{R}_{\ge 0}^G$。

5. **训练目标**：$\mathcal{L}=\mathcal{L}_{\mathrm{cell}}+\lambda\mathcal{L}_{\mathrm{patch}}$，$\lambda=0.5$。
   - $\mathcal{L}_{\mathrm{cell}}$：逐基因 Huber loss 在所有细胞与活跃基因上取平均。用 Huber 而非 MSE 以鲁棒于厚尾表达分布。
   - $\mathcal{L}_{\mathrm{patch}}$：辅助损失，目标为 patch 内所有细胞表达之和 $\mathbf{x}_{\mathrm{patch}}=\sum_i\mathbf{x}_i$，对 patch 级一致性施加正则。

6. **训练细节**：4×H100, DDP, batch=512, FP16; AdamW; backbone LR=$5\times10^{-6}$, head LR=$2\times10^{-4}$; cosine annealing $\eta_{\min}=10^{-8}$, 200 epochs; 渐进解冻：前 10% epoch 冻结，接下来 30% 线性升温至 $2\times10^{-5}$，之后全量微调; gradient clipping=1.0; seed=42。

---

## 实验与结果
**数据集**：HEST-1k 中 52 对 Xenium–H&E WSIs，12 器官、3 种健康状态（癌/病/健康），共 ~10M 细胞，36 train / 4 val / 12 test。Test 含 4 个完全未见器官（Brain/Bone/Heart/Ovary 各 1 slide）与 2 个跨条件 OOD（Kidney/Liver）。

**评估指标**：平均 Pearson 相关系数（PCC），按 top-k 基因子集报告：MPG（最可预测）、HVG（高变异）、SVG（空间变异），取 k∈{10,50,200}；另报告 full gene panel 与 MPG/SVG 完整结果。

**基线**：
- **UNet3+（GHIST 类）**：基于细胞 mask 的 CNN encoder，不依赖 PFM per-cell 前向。
- **DeepSpot2Cell**：per-cell 独立 PFM 前向 + spot-level 弱监督bags-of-cells 范式。
- **Grid**：CELLO 的消融版（仅 grid sampling 无距离衰减 cross-attention）。
- **Per-cell crop（附录 G）**：与 CELLO 同 backbone/head/split，但用 frozen Virchow2 对以细胞为中心 224×224 crop 的 CLS 作为表征。

**主要结果（HVG PCC-50 平均）**：
- **ID 设置**：CELLO 0.4390 vs DeepSpot2Cell 0.2773（+58.5% 相对提升）vs UNet3+ 0.2044；Bowel 器官 CELLO 达 0.6599（最深）。
- **OOD 设置**：CELLO 0.2801 vs DeepSpot2Cell 0.1635（+71.3%）vs UNet3+ 0.1395；Ovary CELLO 0.4049 显著领先。
- **Grid 消融**：ID +18.6%，OOD +16.2%，验证距离衰减交叉注意力的有效性。
- **Encoder 选择**：Virchow2 最优；对 HVG PCC-50，Virchow2 较 H0 提升 14.7%，较 UNI 提升 20.3%。
- **CLS fusion**：加入 CLS token 在所有三个 encoder 下均造成性能下降（ID Virchow2 平均 9 项指标从 0.4793→0.4143，-13.6%）。
- **Full gene panel**：CELLO 在全部测试器官与 ID/OOD 下均保持最高 PCC，证明不只是高信号基因的特例。
- **外部验证**：5 个未参与 10X-Xenium-52 的 HEST-1k Xenium 样本（Breast/Lung/Skin），零微调直接使用 CELLO，MPG/HVG/SVG PCC 均处于合理范围（Table 10）。

**计算效率**：
- CELLO 单 slide 推理 67.5±48.8 s（GPU 39.3 s），DeepSpot2Cell 947.2±1068.3 s，**平均 14.0× 加速**（median 11.7×，范围 3.5–26.3×）；速度与细胞密度强相关（Spearman ρ=0.97）。
- 计入 CellViT-SAM-H 细胞分割（平均 382.5 s/slide）后 CELLO 仍快 2.5×；与 UNet3+ 时间相当（median 1.12×），但精度更高。
- CELLO 耗时结构：Virchow2 前向占 81.8%，grid sampling 5%、cross-attention 16%、CLS fusion + gene head 合计 <4%。

---

## 相关工作脉络
1. **DeepSpot（He et al. 2020 / Nonchev et al. 2025）**：早期 spot 级回归方法，通过转移学习从 patch 特征映射到表达向量；定位差异：spot 级聚合，无法解析细胞异质性。
2. **DeepSpot2Cell（Nonchev et al. 2025）**：首个尝试单细胞的 PFM 方法，per-cell crop + spot-level 弱监督 bags-of-cells；本文在扩展性（N 次 vs 1 次前向）与监督信号（真单细胞 vs 弱 spot）上双重超越。
3. **GHIST（Fu et al. 2025, Nature Methods）**：基于 cell mask 的单细胞 aware 表示，保留空间结构但不用 PFM；本文表明强 PFM 表征 + 位置查询的组合优于纯 mask-based 路径。
4. **iStar / scstGCN / Zhang et al. 2024 (Bayesspace-like super-resolution)**：通过 superpixel 级反卷积提高分辨率；定位差异：输出仍是 high-dim superpixel map 而非 precise cell profile，且需要 ST 数据作为输入。
5. **Diffusion / Flow matching 生成式方法（Zhu et al. 2025; Huang et al. 2025）**：用扩散或 flow matching 模拟基因表达的多模态分布；定位差异：生成式建模 raw counts，CELLO 采用 Huber 回归更适合 Pearson 评测任务且效率更高。
6. **计算病理基础模型（UNI / Virchow2 / H-Optimus-0）**：自监督预训练获取强形态表征；本文将其作为共享特征提取器并与位置查询结合，区别于简单 fine-tune 或 freeze 后 per-cell 使用。
7. **上下文感知生物建模（PINNACLE / AlphaFold / SToFM / COMMOT）**：显式编码上下文先验（几何/进化/空间）提升表征；CELLO 延续此哲学，用 2D RoPE + 距离衰减作为空间归纳偏置。

---

## 局限性与未来方向
- **OOD 样本量小**：4 个未见器官各只有 1 slide，Kidney/Liver 仅跨疾病状态，导致 cross-organ 泛化能力估计精度有限，需更多配对数据。
- **依赖上游细胞分割**：推理需要细胞 centroid 坐标，实际部署仍需 CellViT-SAM-H 等检测器；分割误差会传递（实验显示 10 px jitter 仅影响 <1%，但 40 px 误差导致 -10.6% PCC）。
- **单器官/单疾病偏向**：训练集 Lung 有 19 个样本而 Bone/Heart/Ovary 仅 1 个，长尾分布可能影响少数类别的上限。
- **基因面板差异**：Xenium 面板各样本测 280–480 基因不等，统一预测 1915 基因 union vocab 会引入噪声；当前按 sample 内基因计算 loss/metric 可缓解，但未完全解决面板不对齐问题。
- **未探索多模态融合**：仅用 H&E 形态，未整合 IHC、蛋白组学等其他模态信息。
- **未来方向**：扩大配对数据覆盖更多器官/条件；改进细胞分割精度以降低误差级联；探索无坐标需求的自监督预训练；扩展到更多 ST 平台（Stereo-seq、MERFISH 等）。

---

## 研究启发与可借鉴点
1. **"共享特征图 + 位置查询"范式可迁移**：凡是需要对不规则目标（细胞/分子/感兴趣区域）在连续空间坐标上进行精确定位预测的任务，均可考虑一次大模型前向 + 可微 grid sampling 替代 per-target 独立前向，显著降低计算复杂度。
2. **距离衰减偏置作为空间归纳先验**：在 cross-attention 中叠加 Gaussian 距离先验（$b\propto -\|p-q\|^2/\sigma^2$）是一种轻量且可微的空间上下文注入方式，适用于任何具有明确空间邻域关系的多目标预测任务（如空间代谢组、蛋白质共定位、细胞图谱）。
3. **全局 CLS token 不一定有帮助**：对于需要细粒度空间推理的任务，引入 slide-level 摘要可能引入无关噪声；设计时应以局部形态表征为主，全局信息需谨慎融合。
4. **辅助 patch/region-level 监督可稳定训练**：CELLO 用 patch-level 求和损失 $\mathcal{L}_{\mathrm{patch}}$ 作为正则，类似思路可用于其他多实例学习场景，帮助模型对齐层级一致。
5. **系统消融应包含 encoder 选择与上下文策略**：本文对 UNI/Virchow2/H0 的公平对比（无数据增强、同 split、同 head）揭示了"更强 PFM 直接带来更好 cell embedding"这一反直觉结论，提示后续工作在选择 backbone 时不能仅凭预训练规模推断。

---

## 关键术语表
- **H&E（Hematoxylin and Eosin）**：苏木精-伊红染色，病理学最标准的组织形态学染色方法，呈现细胞核（蓝）与胞质/细胞外基质（粉红）对比。
- **Spatial Transcriptomics（ST，空间转录组）**：将基因表达谱映射到组织空间位置的分子技术，Xenium 为代表的高分辨率 in situ 测序平台。
- **Pathology Foundation Model（PFM，病理基础模型）**：在数百万张 H&E 图像上自监督预训练的 ViT 类编码器（如 UNI/Virchow2/H-Optimus-0），提供强形态表征。
- **Grid Sampling / Bilinear Interpolation**：从离散 2D 特征图上按连续坐标可微插值提取特征，实现 sub-patch 精度的位置查询。
- **Distance-Decay Cross-Attention**：在 cross-attention 注意力分上加 Gaussian 距离衰减偏置，使邻近视觉 token 获得更高权重，编码空间局部性先验。
- **2D RoPE（Rotary Position Embedding）**：将二维相对位置编码注入 attention 的旋转位置编码，无需额外参数即可捕获 cell-token 相对几何关系。
- **Huber Loss**：对异常值鲁棒的损失函数，结合 MSE 与 MAE 特性，适合表达值厚尾分布的回归任务。
- **In-Distribution / Out-of-Distribution（ID / OOD）**：ID 指测试时器官与疾病状态在训练中见过；OOD 指未见器官（Brain/Bone/Heart/Ovary 各 1 slide）或同器官但新疾病状态（Kidney/Liver）。

---

## 可复现要素
- **数据集**：10X-Xenium-52（来自 HEST-1k），52 对 Xenium–H&E WSIs；数据在 HuggingFace 公开（`hf.co/datasets/gaozijun/cello_data`）。
- **代码**：GitHub 开源（`github.com/zjgao02/CELLO`）。
- **模型权重**：HuggingFace 开源（`hf.co/gaozijun/CELLO`）。
- **关键超参**：
  - Patch 大小：224×224；stride 224（推理时非重叠）。
  - Encoder：Virchow2（$p=14, d=1280$）；UNI（$p=16, d=1024$）；H-Optimus-0（$p=14, d=768$）。
  - 优化器：AdamW；backbone LR $5\times10^{-6}$，head LR $2\times10^{-4}$；weight decay 0.01/0.05。
  - LR schedule：cosine annealing，$\eta_{\min}=10^{-8}$，200 epochs。
  - Batch size：512（4×H100，per-GPU 128）。
  - Loss 权重：$\lambda=0.5$（cell:patch）。
  - 渐进解冻：前 10% epoch 冻结 backbone，随后 30% epoch 线性升温至 $2\times10^{-5}$，剩余全量微调。
  - 梯度裁剪：1.0；FP16 mixed precision。
  - Seed：42。
- **不确定/未明确提及**：distance-decay 的 $\sigma^2=p^2$ 中归一化常数确认为 $p^2$；cross-attention 的 head 数与维度未在主文详述（见附录 C）；训练数据增强策略声明为"不使用任何 data augmentation"。

---

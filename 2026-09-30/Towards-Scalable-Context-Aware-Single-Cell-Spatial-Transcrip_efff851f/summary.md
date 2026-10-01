---
title: "Towards-Scalable-Context-Aware-Single-Cell-Spatial-Transcrip"
source: https://arxiv.org/pdf/2609.36429v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 11:15:31"
field: "计算病理与空间组学"
keywords: ["single-cell spatial transcriptomics", "histology-to-gene prediction", "pathology foundation model", "grid sampling", "cross-attention", "distance-decay", "scalable computational pathology"]
innovations: ["单次PFM前向+grid sampling实现所有细胞的亚token级位置查询", "distance-decay cross-attention显式注入局部形态上下文先验", "cell-level与patch-level双损失联合训练消除弱监督瓶颈"]
benchmarks: ["HEST-1k Xenium-52", "10X-Xenium single-cell paired WSI", "MPG/HVG/SVG top-k Pearson correlation"]
---

# 论文速读：CELLO — 可扩展的细胞级空间转录组预测

## 一句话总结
本文提出 **CELLO**，一种端到端可缩放框架，仅对整张 H&E 图像做**一次**病理基础模型前向传播，再经 grid sampling + distance-decay cross-attention 同步提取所有细胞的特异性特征，从而在约 1000 万细胞、12 个器官的 Xenium–H&E 配对数据上达到 SOTA，且比 DeepSpot2Cell 快 **14.0×**。

## 研究问题与动机
1. **现有方法停留在 spot 级**：Visium 等平台的每个 spot 汇聚数十个细胞，得到的表达谱是“平均化”信号，掩盖了细胞异质性。
2. **直接套用 PFM 会尺度失配**：病理基础模型（PFM）的 patch 级表征混合多细胞信息；逐细胞裁剪再独立编码不仅计算量随细胞数线性膨胀，还会**扭曲细胞形态并丢失微环境上下文**。
3. **纯分割方法的表征瓶颈**：GHIST 等基于 cell mask 的 UNet3+ 虽保留空间结构，但无法充分利用现代 PFM 学到的强形态学表征，且其预测质量被上游分割噪声绑定。
4. **弱监督 spot 级训练的不适配**：DeepSpot2Cell 等通过 spot 标签反传到单细胞的设置存在**细胞间表达无法解耦**的 ill-posed 问题，且需要 N 次 PFM 推理，难以扩展到百万细胞 WSI。

## 核心贡献（创新点）
1. **单次 PFM 前向 + grid sampling 同时提取所有细胞特征**：避免逐细胞独立编码，把 PFM 计算摊薄到 patch 级，实现端到端可缩放。
2. **distance-decay cross-attention 模块**：通过 2D RoPE 编码相对位置、通过高斯距离衰减 bias 让每个细胞优先关注其局部形态邻域，将空间先验显式注入表征。
3. **cell-level 与 patch-level 双损失联合训练**：auxiliary patch loss 以 patch 内所有细胞表达之和为目标，提供额外的局部一致性正则。
4. **首次系统评测单细胞级 H&E→Xenium 预测**：使用 HEST-1k 的 52 对 Xenium–WSI、约 1000 万细胞，涵盖 12 器官和 ID/OOD 双设定，证明 CELLO 在精度与效率上的全面优势。
5. **与已有工作的本质区别**：不依赖逐细胞 PFM 推理、不依赖 cell mask 质量、不依赖 spot 级弱监督；直接从**真实单细胞配对数据**端到端学习。

## 方法详解
**问题形式化**：给定 H&E 图像块 $\mathbf{V} \in \mathbb{R}^{L \times L \times 3}$ 和细胞坐标集合 $S = \{\mathbf{s}_i\}_{i=1}^N$，学习 $f_\theta: (\mathbf{V}, \mathbf{s}_i) \mapsto \hat{\mathbf{x}}_i \in \mathbb{R}^G_{\ge 0}$，预测每个细胞的 $G$ 维基因表达向量。

**视觉特征提取**：使用冻结的 PFM（默认 Virchow2）对整块 $\mathbf{V}$ 做单次前向，得到 CLS token $\mathbf{z}_{cls}$ 与 $M$ 个空间 token $\{\mathbf{t}_m\}$，reshape 为 $L_g \times L_g \times d$ 的特征图 $\mathbf{T}$，保持与图像的几何对应。

**位置感知 Cell Query**（Grid Sampling）：
- 将细胞全局坐标转换到 patch 局部坐标系：$\mathbf{s}_i^{\text{loc}} = \mathbf{s}_i - \mathbf{o}$。
- 归一化到 $[-1,1]$ 后通过可微双线性插值采样：$\mathbf{q}_i^{(0)} = \text{GridSample}(\mathbf{T}, \tilde{\mathbf{s}}_i)$，实现亚 token 粒度的连续位置查询。

**Cross-Attention 与 2D RoPE + 距离衰减**：
- 对 $\mathbf{q}_i^{(0)}$ 和所有空间 token $\mathbf{t}_m$ 做 multi-head cross-attention。
- **2D RoPE**：将细胞坐标除以 token stride 映射到连续 grid 坐标，与 visual key 的 token-center grid 对齐后注入相对位置。
- **距离衰减 bias**：$b_{i,m} = -\|\mathbf{s}_i^{\text{loc}} - \mathbf{c}_m\|^2 / (2p^2)$，使近处 token 获得更大 prior，scale 相对于 token 感受野归一化。
- LayerNorm 输出细胞嵌入 $\mathbf{z}_i$。

**Gene Head 与训练目标**：
- 两阶段 MLP（ReLU）解码：$\hat{\mathbf{x}}_i = h_{\text{cell}}(\mathbf{z}_i')$。
- 联合损失：$\mathcal{L} = \mathcal{L}_{\text{cell}} + \lambda \mathcal{L}_{\text{patch}}$，其中 $\mathbf{x}_{\text{patch}} = \sum_{i=1}^N \mathbf{x}_i$。使用 Huber loss（对重尾表达值更稳健）。

**训练细节**：4× H100、batch 512、FP16、AdamW；backbone 初始 10% epoch 冻结，随后线性解冻到 $2\times10^{-5}$，最终全微调；backbone LR=$5\times10^{-6}$，head LR=$2\times10^{-4}$。

## 实验与结果
**数据集**：HEST-1k 中 52 对 Xenium–WSI，12 器官（lung/breast/bowel/skin/pancreas/lymphoid/liver/kidney/brain/bone/heart/ovary），约 1000 万细胞，union vocabulary 1915 基因，每样本 280–480 基因。ID 设定 6 器官 + OOD 设定含 4 个完全未见器官。

**评估指标**：Pearson 相关系数（PCC），按 HVG/MPG/SVG top-k（k=10/50/200）报告；另有全基因 panel 评估。

| 方法 | ID avg. HVG-50 | OOD avg. HVG-50 | WSI 推理时间 |
|---|---|---|---|
| UNet3+ | 0.2044 | 0.1395 | ~74s |
| DeepSpot2Cell | 0.2773 | 0.1635 | ~947s |
| Grid（无 cross-attn） | 0.3700 | 0.2413 | ~68s |
| **CELLO** | **0.4390** | **0.2801** | **~68s** |

- CELLO 相对最强 baseline DeepSpot2Cell：ID 提升 **+58.5%**，OOD 提升 **+71.3%**（HVG-50）。
- 相对 Grid：ID +18.6%，OOD +16.2%，证明 distance-decay cross-attention 的独立价值。
- 编码器 ablation：Virchow2 > H0 > UNI；Virchow2 在 MPG/HVG/SVG 三类基因集均最优。
- **CLS token 融合实验**：加入 CLS token 反而下降 ~13.6%（ID）和 ~13.1%（OOD），证明单细胞表达主要受**局部形态上下文**驱动，全局语义帮助有限。
- 速度：CELLO vs DeepSpot2Cell = **14.0× 加速**（median 11.7×，范围 3.5–26.3×）；加入 cell segmentation 后仍快 2.5×。CELLO vs UNet3+ 速度相当（median 1.12×）。
- 全基因 panel：CELLO 在所有测试器官均显著领先。
- 外部验证（5 个未参与训练的 HEST-1k 样本）：在 breast IL/UC、lung LUAD、skin SKCM 上均取得合理 PCC，验证泛化能力。
- 定位鲁棒性：±10 px centroid jitter 仅导致 <1% PCC 下降，说明对细胞定位误差容忍度高。

## 相关工作脉络
1. **Spot-level 回归方法**（DeepSpot、M2ORT、ThiTogene 等）：在 Visium spot 级做 H&E→表达预测，无法解析单细胞异质性。CELLO 首次直接端到端训练于**真实单细胞配对**。
2. **Super-resolution 去卷积**（iStar、scstGCN、Bayesspace）：需要 ST 数据作为输入，输出超像素级表达图，仍依赖后处理近似单细胞。CELLO 无需 ST 输入，直接从 H&E 生成单细胞表达。
3. **DeepSpot2Cell**：唯一相近的细胞级方法，采用逐细胞 PFM 推理 + spot 弱监督反传。CELLO 通过 grid sampling 消除逐细胞推理瓶颈，且使用**真单细胞监督**。
4. **GHIST**：基于 UNet3+ 从 cell mask 提取表征，保留空间结构但 PFM 表征弱、被分割质量绑定。CELLO 不依赖 cell mask、充分利用强 PFM。
5. **计算病理 PFM**（UNI、Virchow2、H-Optimus-0）：本研究系统评测了三类编码器，证明更强预训练表征对单细胞预测有显著增益（Virchow2 比 H0 高 14.7% PCC-50）。
6. **生成式方法**（Diffusion-ST、Flow Matching-ST）：将基因表达视为随机过程建模；CELLO 坚持回归范式（Huber loss + PCC 评估），更适合下游定量分析。

## 局限性与未来方向
1. **OOD 评估受限于单 slide**：Brain/Bone/Heart/Ovary 每个仅 1 个测试样本，跨器官泛化精度估计不够稳定。
2. **仍依赖上游细胞分割**：推理时需要 cell centroid；虽然对 ±10 px jitter 鲁棒，但分割错误仍会导致预测退化。
3. **长尾器官覆盖不足**：训练集中 lung 占 22/52 样本，brain/bone/heart/ovary 仅各 1 个，模型对稀有器官预测能力待验证。
4. **疾病状态不平衡**：bone/heart 仅健康、pancreas/ovary 仅癌症，跨疾病泛化需谨慎解读。
5. **未来方向**：扩大多器官配对数据、探索无需 cell mask 的定位（如自注意力显式建模）、结合 receptor-ligand 等细胞互作先验、扩展到更多 ST 平台（Visium HD、MERFISH）。

## 研究启发与可借鉴点
1. **Grid Sampling 作为 patch 级→实例级的通用桥接**：任何需要在离散实例坐标上从共享 feature map 提取特征的场景（细胞检测、蛋白定位、切片级 ROI 分析）均可复用此设计，避免逐实例裁剪带来的计算爆炸。
2. **距离衰减 bias 的简单有效**：不需要额外参数即可将"近处 token 更重要"的生物学先验注入 attention；类似思路可迁移到 spatial proteomics、spatial metabolomics 等其他空间组学预测任务。
3. **"全局语义不如局部形态"的实证价值**：CLS token 融合反而下降的结果提醒同行——在需要精细局部推理的任务上，过度依赖全局池化 token 可能适得其反；应在设计初期通过 ablation 验证 context 来源。
4. **双损失（instance-level + region-level）的正则思路**：patch-level auxiliary loss 与 cell-level loss 联合训练，为多尺度监督提供了一个干净范式；类似思路可用于 spot+cell 混合监督、多分辨率预测等场景。
5. **与本体团队的结合点**：若团队关注**低资源翻译、长文本推理或细粒度视觉定位**，CELLO 的"单次大模型前向 + 位置敏感查询"范式值得借鉴；其开源代码（github.com/zjgao02/CELLO）提供了可直接复用的 PyTorch 实现。

## 关键术语表
- **Spatial Transcriptomics (ST)**：在组织切片上同时测量基因表达与空间位置的技术（如 10x Xenium、Visium）。
- **Pathology Foundation Model (PFM)**：在百万级 H&E 图像上自监督预训练的大规模视觉编码器（UNI、Virchow2、H-Optimus-0）。
- **Xenium**：10x Genomics 的 in situ 单细胞分辨率空间转录组平台，提供亚细胞级基因定位。
- **HEST-1k**：涵盖 1000+ 对 H&E–ST 全切片的大规模基准数据集（Jaume et al., NeurIPS 2024）。
- **Grid Sampling / Bilateral Interpolation**：从离散 token 特征图上通过可微双线性插值在连续坐标处采样，实现亚 token 粒度的位置查询。
- **2D RoPE (Rotary Position Embedding)**：将相对 2D 位置编码注入 attention logits，无需额外参数即建模空间关系。
- **Distance-Decay Cross-Attention**：在 attention score 上叠加高斯距离衰减 bias，使近邻 token 获得更高权重。
- **HVG / MPG / SVG**：高可变基因、高预测力基因、空间可变基因——三种常用的基因子集评估口径。

## 可复现要素
- **数据集**：HEST-1k 公开数据（Hugging Face：hf.co/datasets/gaozijun/cello_data），52 对 Xenium–WSI 配对已整理。
- **代码**：github.com/zjgao02/CELLO，基于 PyTorch DDP，支持 4× H100 训练。
- **模型权重**：Hugging Face Hub（hf.co/gaozijun/CELLO）。
- **关键超参**：patch 224×224、stride 224、PFM 用 Virchow2（patch 14×14、dim 1280）、AdamW、LR backbone $5\times10^{-6}$ / head $2\times10^{-4}$、batch 512、λ=0.5、200 epoch、seed=42。
- **训练硬件**：4× NVIDIA H100，FP16 mixed precision。
- **评估协议**：ID/OOD 双设定，PCC 在 HVG/MPG/SVG top-k 三口径报告，含全基因 panel 与 5 样本外部验证。

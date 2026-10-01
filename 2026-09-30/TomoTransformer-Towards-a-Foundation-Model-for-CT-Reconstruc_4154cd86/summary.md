---
title: "TomoTransformer-Towards-a-Foundation-Model-for-CT-Reconstruc"
source: https://arxiv.org/pdf/2609.37605v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 11:15:22"
field: "医学图像重建"
keywords: ["sparse-view CT", "foundation model", "transformer", "tomographic reconstruction", "zero-shot generalization", "disentangled representation"]
innovations: ["在解耦反投影空间中操作，实现视角数、角度配置和探测器尺寸的无关性", "几何注意力机制将物理先验融入transformer架构", "单模型跨多种医学CT解剖结构和自然图像泛化，含零样本到纳米级X射线成像"]
benchmarks: ["LoDoPaB-CT", "KiTS23", "COVID-19 CTSpine1K", "MSD-T10", "COLONOG", "HNSCC", "FFHQ", "ImageNet"]
---

# 论文速读：TomoTransformer: Towards a Foundation Model for CT Reconstruction

## 一句话总结
本文提出TomoTransformer，一个基于transformer的CT重建基础模型，通过在解耦反投影空间中处理局部滤波投影作为token，实现了对任意视角数、角度配置和探测器尺寸的泛化能力，显著优于现有方法并在零样本实验中展示了跨模态泛化潜力。

## 研究问题与动机
1. **协议依赖性限制部署**：现有监督学习方法每个模型仅针对特定采集几何（固定投影数和角度范围）训练，投影数量、角度或探测器分辨率改变时都需要重新训练。
2. **尺寸限制**：现有模型通常绑定固定图像和探测器尺寸，无法处理不同分辨率数据。
3. **泛化能力不足**：模型通常在单一窄数据分布（如单个解剖区域）上训练，缺乏跨解剖结构、材料和分辨率的泛化能力。
4. **ViewTrans等近期工作的不足**：虽然ViewTrans等尝试使用transformer预测缺失视图，但仍绑定固定探测器尺寸和图像分辨率。

## 核心贡献（创新点）
1. **CT重建基础模型**：提出TomoTransformer，同时摆脱对投影数量和角度配置以及探测器尺寸的依赖，消除了限制现有监督方法的协议依赖性。
2. **解耦空间中的正弦图补全**：在解耦反投影空间中建模正弦图补全问题，使视图插值几何上合理，产生灵活可扩展的架构。
3. **大规模多域训练**：在包含多样化医学CT解剖结构和自然图像的大规模数据集上训练，证明单一权重集可跨解剖结构、材料和分辨率泛化，包括零样本迁移到真实纳米级脑组织数据。
4. **超越专用基线**：匹配或超越针对每个稀疏级别单独训练的专用基线方法，并显著优于绑定固定探测器尺寸的ViewTrans。
5. **零样本去噪能力**：在干净图像上预训练的TomoTransformer可实现对未见噪声投影的零样本去噪，无需微调。

## 方法详解
**1. 解耦反投影空间**：
- 将FBP公式中的单视角反投影$\mathbf{b}_r(x,y) = \tilde{\mathbf{s}}_r(x\cos(\theta_r) + y\sin(\theta_r))$作为独立token
- 构建张量$\mathbf{B} = [\mathbf{b}_1, \mathbf{b}_2, ..., \mathbf{b}_R] \in \mathbb{R}^{N \times N \times R}$，形成解耦反投影空间
- 在此空间中，每个视角贡献在重建坐标$(x,y)$处被隔离，视图插值成为空间对齐的插值问题

**2. Transformer架构设计**：
- 编码器-解码器结构类似MAE，编码器仅处理已有token，解码器接收编码器输出和可学习token预测目标视图
- 位置编码：将角度$\theta_r$映射为傅里叶位置嵌入$\{(\sin(k\theta_r), \cos(k\theta_r))\}_{k=1}^{F}$，$F=32$频率
- 几何注意力机制：在注意力logits中加入成对角度差$\theta_r - \theta_{r'}$的 learned 函数

**3. 几何Tokenizer**：
- 单视角反投影由平行于角度$\theta_r$的法线的条纹组成
- 几何tokenizer沿条纹方向最近的对角线采样$K = \lceil P\sqrt{2} \rceil = 46$个等间距点
- 线性编码器$\mathbf{E}: \mathbb{R}^K \to \mathbb{R}^D$将1D特征映射到token嵌入，$D=512$
- 几何Detokenizer逆转此过程

**4. 局部patch处理**：
- 从全反投影提取$P \times P$局部patch（$P=32$像素）作为token
- 避免全图处理，降低内存消耗和计算开销
- 模型与完整图像尺寸N和解码器分辨率解耦

**5. 损失函数**：
$$\mathcal{L} = \frac{1}{|\mathcal{Q}|} \sum_{q \in \mathcal{Q}} \|\hat{\mathbf{t}}_q - \mathbf{t}_q\|_1$$
其中$\mathcal{Q}$为目标角度集合。

## 实验与结果
**训练数据集**：
- 约650k个2D切片：570k CT切片（来自LoDoPaB-CT、COVID-19 CTSpine1K、KiTS23、MSD、COLONOG、HNSCC）+ 80k自然图像（FFHQ、ImageNet）
- 使用各数据集原始分辨率，无需重采样到统一网格

**固定稀疏视角重建（64→256）**：
- TomoTransformer (fixed)在8个数据集上平均PSNR达39.66 dB，优于所有基线
- 在LoDoPaB上达38.19 dB，COVID-19上达40.23 dB，KiTS23上达43.94 dB
- TomoTransformer (varied)在相同设置下平均38.32 dB，仍匹配专用基线

**可变稀疏视角重建**：
- 128→256：TomoTransformer PSNR 42.40 dB / SSIM 0.959，ViewTrans仅37.75 dB / 0.914
- 64→256：TomoTransformer PSNR 37.68 dB / SSIM 0.910，ViewTrans仅33.07 dB / 0.832
- 在极稀疏条件下优势更显著

**零样本去噪**：
- 在45dB高斯噪声和泊松光子计数噪声下，未见过噪声的模型仍能有效重建
- Blind-spot re-masking和Complementary re-masking策略进一步提升质量

**真实实验数据**：
- 纳米级脑组织PXCT数据（768×768，448个轴切片，128→256/512视图）
- 128→256：PSNR 35.8 dB / SSIM 0.837
- 128→512：PSNR 38.8 dB / SSIM 0.916
- 零样本泛化到完全未知的模态和尺度

## 相关工作脉络
1. **FBPConvNet和U-Net变体**：早期的CNN重建方法，绑定固定采集协议，需针对特定稀疏级别训练。
2. **DuDoTrans、CTTR、TD-STrans**：近期基于transformer的重建方法，虽捕捉全局结构，但仍绑定固定探测器分辨率。
3. **ViewTrans**：最接近的竞品，直接在原始正弦图上操作，绑定固定探测器尺寸；本文通过解耦反投影空间突破此限制。
4. **扩散模型方法**：理论可适配任意采集协议，但需要数百至数千次迭代推理，计算成本高。
5. **迭代重建方法**：如TV最小化、小波稀疏性，速度慢且先验手工设计。

## 局限性与未来方向
1. **仅针对2D平行束CT**：当前模型仅适用于2D平行束几何，未扩展到锥束CT或曲面探测器。
2. **训练依赖模拟数据**：主要在模拟的平行束CT数据上训练，真实扫描仪数据的域偏移可能仍存在挑战。
3. **推理速度**：相比专用小网络，transformer架构推理成本较高。
4. **未来方向**：论文提到将探索将此架构扩展到cone-beam CT等其他断层扫描模态。

## 研究启发与可借鉴点
1. **解耦空间的巧妙设计**：将反投影操作显式分离，使视图插值成为空间对齐问题，这一思路可迁移到其他 tomography 模态或逆问题。
2. **几何先验融入注意力**：通过几何注意力机制（GAM）将物理先验直接注入transformer，比纯数据驱动更有效，可用于其他具有已知几何结构的任务。
3. **局部patch tokenize策略**：将大图像分解为固定大小patch处理，既降低内存又实现分辨率无关性，是构建视觉基础模型的有效技巧。
4. **零样本泛化验证范式**：在训练分布外数据（纳米级脑组织）上展示零样本能力，为模型实用性提供了强有力证明。
5. **混合数据训练策略**：结合医学CT和自然图像训练，利用纹理先验增强泛化，可扩展到其他模态。

## 关键术语表
**TomoTransformer**：一种基于transformer的CT重建基础模型，在解耦反投影空间中处理局部投影token。
**解耦反投影空间（Disentangled back-projection space）**：将FBP中每个视角的反投影分离存储的表示空间，使视图插值成为空间对齐问题。
**几何注意力机制（Geometric Attention Mechanism, GAM）**：在transformer注意力logits中加入角度差的learned函数，编码投影几何先验。
**傅里叶位置嵌入**：将连续角度映射为正弦/余弦特征序列的位置编码方式，支持任意角度泛化。
**几何Tokenizer**：利用单视角反投影的条纹结构，沿对角线采样1D特征的低维嵌入方法。
**零样本去噪（Zero-shot denoising）**：在从未见过噪声类型的测试数据上直接应用的去噪能力。
**Ptychographic X-ray computed tomography（PXCT）**：一种基于衍射的X射线层析成像技术，测量折射率而非吸收。
**Blind-spot re-masking**：将测量角度划分为多个子集并逐一遮蔽预测的去噪策略。

## 可复现要素
- **数据集**：LoDoPaB-CT、CTSpine1K、KiTS23、MSD、COLONOG、HNSCC、FFHQ、ImageNet（多为公开数据集）
- **代码/权重**：论文未明确提及代码和权重是否开源
- **关键超参**：patch大小$P=32$，embedding维度$D=512$，head数$H=8$，傅里叶频率$F=32$，AdamW优化器，学习率$3\times10^{-4}$，余弦衰减，训练$10^6$步
- **硬件**：单卡NVIDIA A100 GPU

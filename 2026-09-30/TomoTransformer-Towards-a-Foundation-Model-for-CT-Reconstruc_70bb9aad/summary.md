---
title: "TomoTransformer-Towards-a-Foundation-Model-for-CT-Reconstruc"
source: https://arxiv.org/pdf/2609.37605v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 11:15:23"
field: "医学图像重建与基础模型"
keywords: ["sparse-view CT", "foundation model", "Transformer reconstruction", "disentangled back-projection", "geometric attention", "zero-shot denoising", "protocol-agnostic"]
innovations: ["在解耦反投影空间中构建 Transformer，实现任意投影数、角度与检测器尺寸的统一重建", "引入几何注意力机制，将角度差以 learned bias 注入自注意力", "局部 patch token 设计使模型与分辨率解耦，支持零样本跨域迁移与隐式去噪"]
benchmarks: ["LoDoPaB-CT", "COVID-19 CTSpine1K", "KiTS23", "MSD-T10", "COLONOG", "HNSCC", "FFHQ", "ImageNet", "SLS 纳米级脑 PXCT 数据"]
---

# 论文速读：TomoTransformer-Towards-a-Foundation-Model-for-CT-Reconstruc

## 一句话总结
提出 TomoTransformer，一个基于 Transformer 的 CT 重建基础模型，通过将每个局部滤波反投影视为独立 token 并利用自注意力预测缺失视图，实现了对任意输入投影数量、任意角度位置和任意检测器分辨率的统一建模，无需重新训练即可泛化到不同扫描协议与真实实验数据。

## 研究问题与动机
- 现有监督深度学习方法针对固定采集几何（固定投影数、角度范围）训练，换协议即需重新训练或修改架构。
- 传统模型绑定于固定的图像尺寸与检测器分辨率，缺乏处理可变分辨率数据的灵活性。
- 单一模型难以跨不同解剖区域、材料类型和数据分布进行泛化，部署局限性大。
- 尽管近期出现 ViewTrans 等多用途模型，但仍受限于固定检测器尺寸和图像分辨率，无法真正应对协议变化。

## 核心贡献（创新点）
1. **提出协议无关的 Transformer 重建模型**：TomoTransformer 同时无视输入投影数量、角度配置及检测器尺寸，消除了现有监督方法的协议依赖性。
2. **解耦反投影空间中的 sinogram 补全**：将视图插值 formulation 为空间对齐的插值问题，使几何关系更自然，从而支持灵活可扩展的架构设计。
3. **大规模异质数据预训练与零样本泛化**：在包含多种医学 CT 解剖结构与自然图像的大规模数据集上训练，单一权重即可跨解剖、材料与分辨率泛化，并零样本迁移至真实纳米级脑组织数据。
4. **优于协议特定基线与多用途基线**：在多个稀疏视图基准上匹配或超越专门针对各稀疏级别训练的基线，并显著优于同期多用途模型 ViewTrans。
5. **零样本去噪能力**：仅在干净数据上预训练的 TomoTransformer 可直接用于未见噪声分布的稀疏投影去噪，无需微调。

## 方法详解
- **解耦反投影空间**：由滤波反投影公式导出单视角反投影 $\mathbf{b}_r(x,y) = \tilde{\mathbf{s}}_r(x\cos\theta_r + y\sin\theta_r)$，将所有视角堆叠为张量 $\mathbf{B} \in \mathbb{R}^{N\times N\times R}$，每个切片对应一个视角，像素坐标在不同视角间空间对齐。
- **Token 设计**：从单视角反投影中裁剪局部 $P\times P$ patch 作为 token（$P\ll N$），大幅降低内存与计算开销，并使模型与图像尺寸、检测器分辨率解耦。
- **几何 Tokenizer/ Detokenizer**：Tokenizer 沿 patch 对角线采样 $K=\lceil P\sqrt{2}\rceil$ 个点得到一维特征，经线性层映射为 token embedding；Detokenizer 逆向操作，将 embedding 恢复为一维特征后沿查询角度方向扫描填充 patch。
- **位置编码与几何注意力**：每个投影角度 $\theta_r$ 映射为 Fourier 位置编码 $[\sin(k\theta_r),\cos(k\theta_r)]_{k=1}^{F}$，并加入可学习门控缩放；此外，在注意力 logits 中加入两两角度差 $\Delta_{rr'}=\theta_r-\theta_{r'}$ 的 learned 函数 $g_\phi(\sin\Delta,\cos\Delta)$，形成几何注意力（GAM）。
- **Encoder‑Decoder 架构**：类似 MAE，Encoder 仅处理观测 token，Decoder 输入包含观测 token 的输出与可学习 mask token；采用 4 层 Transformer 块，隐藏维度 $D=512$，注意力头数 $H=8$。
- **训练损失**：$\mathcal{L}=\frac{1}{|\mathcal{Q}|}\sum_{q\in\mathcal{Q}}\|\hat{\mathbf{t}}_q-\mathbf{t}_q\|_1$，仅在目标视角上计算 $\ell_1$ 损失。

## 实验与结果
- **训练数据**：约 65 万张 2D 切片（57 万张医学 CT 来自 LoDoPaB‑CT、COVID‑19 CTSpine1K、KiTS23、MSD、COLONOG、HNSCC；8 万张自然图像来自 FFHQ 和 ImageNet），按原始分辨率使用，无需重采样。
- **固定稀疏视图（64→256）**：在 8 个数据集上评估，TomoTransformer (fixed) 平均 PSNR 达 **39.66 dB**，优于 U‑Net (36.86)、DRUNet (38.88)、NAFNet (39.37)、Restormer (39.47) 等基线；TomoTransformer (varied) 平均 PSNR 38.32 dB，仍接近协议特定基线。
- **可变稀疏视图（LoDoPaB‑CT，分辨率 362×362）**：TomoTransformer 在 128→256 上获得 **42.40 dB / 0.959 SSIM**，在 64→256 上获得 **37.68 dB / 0.910 SSIM**，大幅领先 ViewTrans（128→256: 37.75/0.914；64→256: 33.07/0.832）。
- **零样本去噪**：在 45 dB 高斯噪声与泊松光子计数噪声下，直接应用预训练模型即可获得比含噪 FBP 更干净的图像；结合盲斑重掩码（BS）与互补重掩码（CR）进一步提升质量。
- **真实实验数据**：应用于同步辐射纳米级脑组织 PXCT 数据（768×768，617 视角），128→256 得 **35.8 dB / 0.837 SSIM**，128→512 得 **38.8 dB / 0.916 SSIM**；ViewTrans 因绑定分辨率无法直接应用，协议特定基线产生严重块状伪影与纹理幻觉。

## 相关工作脉络
- **FBPConvNet / U‑Net 变体**：传统 CNN 后处理方法，针对固定几何与分辨率，缺乏灵活性。
- **DuDoTrans / CTTR / TD‑STrans**：Transformer 基重建方法，仍受限于固定投影数与检测器尺寸。
- **ViewTrans**：近期多用途 sinogram 补全模型，使用 raw sinogram 且 token 维度等于检测器宽度，绑定固定分辨率；TomoTransformer 通过解耦反投影空间与局部 patch token 突破此限制。
- **扩散模型方法**：通过生成先验实现任意协议推理，但需数百至数千次迭代评分，计算成本高，且先验与检测器尺寸相关。
- ** Learned Primal‑Dual / 展开迭代方法**：将 Radon 算子嵌入网络，提升数据一致性，但仍需处理整幅图像，难以扩展到大尺度数据。
- **Glimpse / Lofi**：同作者前期工作，探索局部重建与尺度泛化，TomoTransformer 进一步将局部 patch 与 Transformer 结合，实现真正的协议无关重建。

## 局限性与未来方向
- 模型在极端稀疏视图（如 < 64 视角）下的性能未充分评估，重建质量可能下降。
- 当前仅针对 2D 平行束 CT，尚未扩展到锥束 CT 或 3D 体数据。
- 尽管训练高效（单次前向传播），但 Transformer 的内存与计算开销仍可能限制其在超高分辨率体积上的直接部署。
- 零样本去噪依赖数据一致性保持，若输入投影噪声极大或存在系统误差，效果可能受限。
- 未来方向包括：拓展至锥束 CT 与其他断层成像模态（如超声、地震波）、探索更高效的架构变体、结合物理约束进一步提升稳定性。

## 研究启发与可借鉴点
- **解耦反投影空间的设计**：将 sinogram 补全转化为空间对齐的插值问题，为其他角度‑插值类任务提供了可迁移的表示思路。
- **几何注意力机制（GAM）**：将已知的角度差以 learned bias 形式注入注意力 logits，是一种轻量且有效的物理先验嵌入方式，可推广至任意角度相关序列建模。
- **局部 patch token 与分辨率无关推理**：通过固定尺寸 patch 处理任意大小图像，并结合重叠平均策略，实现了训练与推理分辨率的完全解耦，适合资源受限场景。
- **零样本去噪与重掩码技术**：在不引入额外噪声建模的情况下，利用预训练模型的结构先验实现隐式去噪，为无监督/零样本鲁棒重建提供了新思路。
- **跨域泛化实验范式**：在医学 CT 与自然图像混合数据上预训练，并在完全不同的纳米级 X 射线数据上验证零样本迁移，为跨模态基础模型建立标准评估流程。

## 关键术语表
- **稀疏视图 CT**：投影数远少于传统密集采样，导致反问题 ill‑posed，产生条纹伪影的重建场景。
- **解耦反投影空间**：将每个视角的滤波反投影单独保留为独立切片，使同一像素在不同视角间保持空间对齐。
- **TomoTransformer**：本文提出的基于 Transformer 的 CT 重建基础模型，支持任意投影数、角度与检测器尺寸。
- **ViewTrans**：基于 sinogram 的 Transformer 多用途重建模型，但受限于固定检测器尺寸。
- **几何注意力（GAM）**：在注意力 logits 中加入角度差的学习函数，显式编码投影间的几何关系。
- **盲斑重掩码（BS）**：将测量角度划分若干子集，逐次用其余角度预测缺失子集，以进一步去噪。
- **互补重掩码（CR）**：在 BS 基础上交换预测与测量角色，两次预测平均后同时去噪测量与预测视图。
- **过滤反投影（FBP）**：经典解析重建方法，通过 ramp 滤波与反投影得到初始重建，常用于生成 sparse‑view 起点。

## 可复现要素
- **数据集**：LoDoPaB‑CT、COVID‑19 CTSpine1K、KiTS23、MSD、COLONOG、HNSCC、FFHQ、ImageNet（部分公开，需申请或已有开源链接）。
- **代码/权重**：论文未明确提供开源代码或预训练权重声明；附录给出了详细的网络结构、训练超参数与实现细节。
- **关键超参**：Patch 大小 $P=32$，采样点数 $K=46$，隐藏维度 $D=512$，注意力头数 $H=8$，每头维度 $64$，Fourier 频率数 $F=32$；AdamW 优化器，初始学习率 $3\times10^{-4}$，余弦衰减，梯度裁剪 1.0，训练 $10^6$ 步，单卡 A100。

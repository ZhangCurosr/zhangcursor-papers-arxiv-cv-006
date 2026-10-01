---
title: "TEXTURE-SPACE-MATERIAL-DIFFUSION"
source: https://arxiv.org/pdf/2609.37654v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 16:47:57"
field: "纹理空间材质生成"
keywords: ["材质生成", "PBR材质", "纹理空间扩散", "纹理空间", "视频扩散模型", "逆渲染", "3D生成"]
innovations: ["首个完全在纹理空间生成完整PBR材质的扩散框架", "3D感知旋转位置编码（3D-aware RoPE）首次应用于纹理空间扩散", "训练无关的噪声滚动与覆盖率加权专家聚合实现8K/100+视图推理扩展"]
benchmarks: ["BlenderVault", "DTC（Digital Twin Catalog）", "TexVerse"]
---

# 论文速读：TEXTURE-SPACE-MATERIAL-DIFFUSION

## 一句话总结
本文提出了一种完全在二维纹理空间中进行扩散生成的框架，利用预训练视频扩散模型（Wan2.1）的先验知识，从单视图/多视图图像或文本条件生成高质量 PBR 材质图，解决了图像空间方法的视角一致性问题，并支持 8K 高分辨率和 100+ 输入视图的扩展推理。

## 研究问题与动机
- **视角一致性问题**：图像空间方法（如 MVDream、SV3D、Hi3D）生成多视图时，同一表面点在多视图中可能被预测为不同材质，导致纹理模糊、细节丢失、高光抑制。
- **物体空间方法成本高**：Trellis、LSRM、3DTopia-XL 等方法依赖稀疏体素或三平面等三维表示，内存和计算开销大，且需从头在合成数据上训练，无法利用大型视频扩散模型先验。
- **材质创作劳动密集型**：为复杂 3D 资产手工编写详细 PBR 材质（Albedo、Roughness、Metallic、Height）极其耗时，亟需自动化生成方案。
- **未知光照下的鲁棒重建挑战**：真实拍摄照片通常带有未知/非均匀光照，如何在缺乏控制光照条件下解耦材质与光照，并从稀疏视图中重建完整覆盖的材质贴图。

## 核心贡献（创新点）
1. **首个完全在纹理空间生成的 PBR 材质扩散框架**：将已知几何投影到 UV atlas，直接在二维纹理空间进行扩散去噪，天然保证视角一致性，避免图像空间方法的纹理接缝问题。
2. **3D 感知旋转位置编码（3D-aware RoPE）**：首次将 3D 世界空间坐标与帧 ID 编码进 RoPE，为单视图/文本到材质生成提供强几何先验，使注意力能捕捉纹理空间中空间距离远但世界空间相邻的区域。
3. **训练无关的推理扩展技术**：提出渐进式噪声滚动（progressive noise rolling）配合随机空间偏移隐藏接缝，以及覆盖感知专家聚合（coverage-weighted expert aggregation），将推理从 17 视图/2K 扩展至 100+ 视图/8K。
4. **支持多模态条件输入与神经材质扩展**：统一框架兼容多视图、单视图、文本、低分辨率材质图多种输入，并证明可无缝适配神经材质表示（proof-of-concept）。
5. **与已有工作的本质区别**：区别于 TEXGen（仅生成漫反射纹理，需卷积-点云交替_attention_）和 CLAY/VideoMat（在图像空间生成后投影到纹理空间），本文在纹理空间端到端生成完整 PBR 四件套（basecolor + height + roughness + metallicity）。

## 方法详解
- **基础架构**：以 Wan2.1-1.3B Diffusion Transformer（DiT）为骨干，使用 VAE 编码器 $\mathcal{E}$ 和解码器 $\mathcal{D}$。输入张量 $\mathbf{I}$ 包含 N 帧纹理空间视图（含世界空间位置图和法线图），目标潜在变量 $\mathbf{z}_0^{\mathrm{mat}}$ 由 base color 和 HRM（height, roughness, metallicity）两张 RGB 图经 $\mathcal{E}$ 编码后沿时序维拼接而成。
- **流匹配训练目标**：前向过程为 $\mathbf{z}_\tau^{\mathrm{mat}} = (1-\tau)\mathbf{z}_0^{\mathrm{mat}} + \tau\epsilon$，模型预测速度场 $v = \epsilon - \mathbf{z}_0^{\mathrm{mat}}$，损失为：
$$\mathcal{L}(\theta) = \mathbb{E}_{\mathbf{z}_0^{\mathrm{mat}}, \epsilon}\left[\left\|\mathbf{f}_\theta([\mathbf{z}_\tau^{\mathrm{mat}}, \mathbf{z}^{\mathrm{I}}, \mathbf{z}^{\mathrm{c}}]; \mathbf{c}_{\mathrm{prompt}}, \tau) - (\epsilon - \mathbf{z}_0^{\mathrm{mat}})\right\|_2^2\right]$$
文本条件经 T5-XXL 编码为 $\mathbf{c}_{\mathrm{prompt}}$。
- **数据增强提升鲁棒性**：训练时添加 Gaussian 噪声并用 FLUX.1-dev img2img 去噪（strength ∈ [0, 0.3]），以及随机高斯模糊（$\sigma \in [0, 15]$），使模型对输入视图不一致性具有韧性。
- **3D-aware RoPE**：标准 Wan2.1 RoPE 编码 $(f_{\mathrm{id}}, p_u, p_v)$，本文替换为 $(f_{\mathrm{id}}, p_x, p_y, p_z)$，将世界空间位置（归一化到 [0,1]）降采样后作为 RoPE 输入，训练前 2000 步线性混合两种 RoPE。
- **推理扩展**：
  - **噪声滚动**：生成 2K 后用最近邻上采样到 8K，添加 timestep=0.2 噪声，将输入拆分为非重叠 2K crop 并行评估，每步随机空间偏移隐藏接缝。
  - **专家聚合**：将输入视图分批，每批独立预测 flow，按 per-texel 覆盖率评分（观测越多越可信），保留 top-2 专家做覆盖率加权平均，避免全平均导致的过平滑。
- **训练配置**：15k 迭代，32 张 A100 GPU，梯度累积 4 步，有效 batch size 128，分辨率从 $512^2 \to 1024^2 \to 2048^2$ 逐步提升。

## 实验与结果
- **数据集**：合成集使用 BlenderVault（32 个 held-out 3D 模型），真实集使用 DTC（Digital Twin Catalog）8 个样本；训练数据来自 TexVerse（120k 视频，17 帧，2048×2048，Poly Haven 697 光照探针）。
- **多视图重建（Table 1）**：
  - 合成集（已知几何）：Ours PSNR **31.24** / SSIM **0.960** / LPIPS **0.0370**，优于 DiffPT（27.50 / 0.929 / 0.0739）和 LSRM（22.82 / 0.872 / 0.1210）。
  - 新光照重渲染（Synthetic relight）：Ours PSNR **30.46** / SSIM **0.950**，大幅领先 DiffPT（24.90）和 LSRM（21.82），证明良好的材质-光照解耦能力。
  - 真实 DTC 集：DiffPT 因过拟合原始光照在重建指标上略优（30.67 vs 29.03），但 Ours 在重渲染任务上显著更强。
- **生成任务（Table 2）**：
  - 单视图到材质：Ours CLIP-FID **1.520** / CMMD **0.0081** / LPIPS **0.0325**，显著优于 Trellis.2（2.227 / 0.0184 / 0.0395）、VideoMatGen（2.973）和 Hunyuan3D 2.1（3.419）。
  - 文本到材质：Ours CLIP-FID **3.338** / CMMD **0.0230** / LPIPS **0.0542**，优于 VideoMat（4.376 / 0.0247 / 0.0652）和 VideoMatGen（4.725 / 0.0330 / 0.0639）。
- **鲁棒性**：使用 VACE 视频模型生成的不一致视图（Figure 5），Ours PSNR 27.07 / SSIM 0.946，仍优于 DiffPT（23.37 / 0.907）。
- **扩展性（Figure 8）**：17 视图→117 视图，2K→8K，LPIPS 从 0.0668 提升至 0.0639，验证推理扩展技术有效。

## 相关工作脉络
1. **图像/视频扩散模型**（SDS-based DreamFusion、MVDream、SV3D、Hi3D、Wan2.1）：本文利用视频扩散先验，但工作在纹理空间而非图像空间，从根本上避免视角不一致。
2. **物体空间生成**（3DTopia-XL、Trellis、LSRM）：三者使用三维体素/结构化潜在变量联合生成几何和材质；本文固定几何，专注材质生成，复用二维预训练模型先验。
3. **逆渲染/材质重建**（DiffPT、IntrinsicAnything、CLAY）：DiffPT 基于可微路径追踪优化，过拟合光照；CLAY 在图像空间生成 canonical views 再投影。本文直接生成完整 UV 贴图，解耦更彻底。
4. **纹理空间生成先驱 TEXGen**：TEXGen 仅生成漫反射（albedo）贴图，需交替使用卷积和点云 attention；本文生成完整 PBR 四件套，纯纹理空间扩散。
5. **神经材质表示**（Generative Neural Materials、VideoNeuMat）：本文证明框架可扩展至神经材质（Yu et al. 2026），是未来方向的起点。

## 局限性与未来方向
- **推理效率低**：17 视图 2K 推理耗时 130s（GB300），117 视图达 2610s；作者指出可用视频模型加速和蒸馏技术改进。
- **Patch 粒度限制**：Wan2.1 以 16×16 为 attention 最小单元，当 patch 跨越多个纹理贴片区段时无法区分；需更高分辨率或 pixel-space diffusion 解决。
- **镜面反射建模有限**：PBR 中 specular 部分仍有挑战，作者建议通过更精细的数据策展改进；神经材质可能提供更优的镜面表达。
- **几何依赖**：当前假设已知有效 UV 参数化；若几何来自生成模型（如 LSRM），误差会传递至纹理空间。

## 研究启发与可借鉴点
1. **纹理空间扩散范式可迁移**：对于任何需要视角一致纹理的任务（如法线贴图生成、置换贴图生成、程序化材质生成），可将 2D/视频扩散模型适配到纹理空间，复用成熟先验。
2. **3D-aware RoPE 设计值得复用**：将世界空间坐标嵌入位置编码，替代或补充像素坐标 RoPE，可有效缓解 UV 展开不连续导致的注意力分散问题，适用于所有纹理生成任务。
3. **推理扩展策略可直接借用**：噪声滚动 + 随机空间偏移隐藏接缝、覆盖率加权专家聚合，是 transformer 高分辨率生成时的通用技巧，可在图像超分、医学图像生成等场景复用。
4. **数据增强策略**：用 FLUX img2img 添加噪声再重建、随机模糊增强，可提升模型对真实拍摄输入（含镜头模糊、压缩伪影）的鲁棒性。
5. **创新机会**：将本框架与神经辐射场（NeRF）或 3D Gaussian Splatting 结合，实现"纹理空间生成 + 神经材质表示"的混合管线；或扩展至动态材质（时间维度扩散）生成。

## 关键术语表
- **PBR（Physically Based Rendering）**：基于物理的渲染，通过 Albedo（basecolor）、Roughness、Metallic、Height 等贴图精确模拟材质光学属性。
- **UV Atlas / Texture Space**：将 3D 模型表面展开为二维贴图坐标的空间，是多边形网格纹理映射的标准表示。
- **Flow Matching**：扩散训练的一种形式化方法，将前向过程定义为干净数据与噪声之间的线性插值，模型预测速度场。
- **3D-aware RoPE（Rotary Positional Embedding）**：将 3D 世界空间坐标而非 2D 像素坐标编码进旋转位置嵌入，增强空间局部性感知。
- **Noise Rolling**：高分辨率扩散推理时，将输入切割为非重叠 crop 并行处理，并在每步随机偏移 crop 边界以隐藏接缝。
- **Coverage-weighted Expert Aggregation**：多视图推理时按每个 texel 被多少视图观测到的覆盖率加权聚合各 batch 的预测结果。
- **CRF（Camera Response Function）Adjustment**：拟合相机响应曲线以消除不同方法间因色调映射差异导致的光照偏差，使定量比较更公平。
- **Neural Material**：用神经网络隐式表示材质外观（如 fuzzy material），比传统 PBR 贴图更具表达力。

## 可复现要素
- **数据集**：TexVerse（训练）、BlenderVault（测试）、DTC（真实测试）；论文未声明代码开源，未提供模型权重下载链接。
- **代码/权重**：论文未提及开源声明（截至阅读时），需关注后续发布。
- **关键超参**：训练 15k 步，32×A100，有效 batch size 128，分辨率调度 $512^2 \to 1024^2 \to 2048^2$；Wan2.1-1.3B 骨干，T5-XXL 文本编码器；RoPE 前 2000 步线性混合； upscale 4×，噪声 timestep=0.2，50 步去噪。

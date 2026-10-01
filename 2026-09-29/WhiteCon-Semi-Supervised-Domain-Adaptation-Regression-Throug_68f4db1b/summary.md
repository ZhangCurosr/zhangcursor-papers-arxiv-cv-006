---
title: "WhiteCon-Semi-Supervised-Domain-Adaptation-Regression-Throug"
source: https://arxiv.org/pdf/2609.34078v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:40:19"
field: "域适应回归"
keywords: ["semi-supervised domain adaptation", "regression", "whitening transform", "consistency regularization", "domain shift"]
innovations: ["首个将白化变换应用于回归任务并提供OLS理论证明", "提出特征方差一致性正则化对齐不同增强强度的特征分布", "结合DWT和双一致性正则化的SSDAR统一框架"]
benchmarks: ["BIWI", "QMUL", "MPI3D"]
---

# 论文速读：WhiteCon-Semi-Supervised-Domain-Adaptation-Regression-Throug

## 一句话总结
论文提出WhiteCon，首个将白化变换（DWT）应用于回归任务并结合双一致性正则化的半监督域适应回归（SSDAR）方法，在BIWI、QMUL和MPI3D等基准数据集上实现SOTA性能。

## 研究问题与动机
1. **现有SSDA方法难以直接迁移到回归任务**：多数半监督域适应方法依赖分类概率熵，不适用于连续值预测的回归场景。
2. **UDAR方法忽视少量标注目标数据的价值**：即使少量目标域标签数据也能显著提升性能，但现有无监督方法未充分利用。
3. **回归任务的参数稳定性问题未被充分研究**：域偏移会导致回归参数方差增大，影响模型泛化能力。
4. **分类域适应方法在回归中的局限性**：基于类别分布的方法无法有效处理连续输出，缺乏针对回归特性的适配。

## 核心贡献（创新点）
1. **首个将DWT应用于回归任务并提供OLS理论支撑**：证明白化变换可降低回归参数方差，与分类中仅用于特征对齐的本质不同。
2. **提出特征方差一致性正则化（L_crf）**：通过对弱增强、强增强和mixup增强特征的方差对齐提升模型鲁棒性，不同于以往复杂计算或单一一致性方法。
3. **构建WhiteCon统一框架**：结合DWT的方差缩减效应与双一致性正则化的鲁棒性，在多个基准上实现SOTA。

## 方法详解
1. **特定域白化变换（DWT）**：通过Cholesky分解将特征协方差矩阵变换为单位矩阵（$\hat{Z} = W(Z - \bar{Z})$，其中$W = L^{-1}$），消除特征间相关性。在OLS假设下，白化后回归参数方差从$\sigma^2(Z^TZ)^{-1}$降至$\sigma^2I$。
2. **预测一致性正则化（L_cp）**：对无标签目标数据施加弱增强、强增强和mixup增强，使强增强和mixup预测接近弱增强预测（伪标签）：$L_{cp} = \lambda_1(\frac{1}{k}\sum(\tilde{y}^{weak} - \hat{y}^{strong})^2 + \frac{1}{k}\sum(\tilde{y}^{weak} - \hat{y}^{mix})^2)$。
3. **特征方差一致性正则化（L_crf）**：沿每个特征维度计算弱/强/mixup增强特征的方差向量，使用MAE对齐：$L_{crf} = \lambda_2(\frac{1}{d}\sum|Var_j^{weak} - Var_j^{strong}| + \frac{1}{d}\sum|Var_j^{weak} - Var_j^{mix}|)$。
4. **总损失函数**：$L_{total} = L_{sup} + L_{cp} + L_{crf}$，其中$L_{sup}$为源域和少量目标域有标签数据的MSE损失。

## 实验与结果
- **数据集**：BIWI（头姿态估计，5874女+9804男图像）、QMUL（头姿态，1504女+3595男）、MPI3D（物体位置，真实/写实/玩具三域）。
- **基线方法**：S+T、MMD、DANN、RSD、DARE-GRAM、DeepDAR、LIRR、Full_T（100%目标标签）。
- **BIWI结果**：WhiteCon平均$R^2$=0.86，MAE=4.71，优于第二的RSD（$R^2$=0.82，MAE=5.73）；F→M提升显著（$R^2$: 0.87 vs 0.80）。
- **QMUL结果**：平均$R^2$=0.83，MAE=9.35，优于第二MMD（$R^2$=0.79，MAE=10.52）。
- **MPI3D结果**：平均$R^2$=0.90，MAE=2.04，大幅领先DARE-GRAM（$R^2$=0.74，MAE=3.78），差距最大达0.16。
- **消融实验**：移除$L_{crf}$导致最大性能下降；三个组件均不可或缺。
- **参数方差验证**：WhiteCon实现最低方差（$1.273\times10^{-4}$），证实理论分析。
- **标签比例实验**：30%目标标签时$R^2$=0.91，超过Full_T（100%标签）的0.88。

## 相关工作脉络
1. **RSD（Chen et al., 2021）**：基于SVD的正交基对齐方法，主张保持特征尺度，但WhiteCon证明特征尺度相似非性能必要条件。
2. **DARE-GRAM（Nejjar et al., 2023）**：通过逆Gram矩阵对齐，需指定低秩超参，易过度调参；WhiteCon无此需求。
3. **DeepDAR（Singh & Chakraborty, 2020）**：首个SSDAR方法，使用MMD和图拉普拉斯，但MMD仅对齐特征分布未保证回归输出一致性。
4. **LIRR（Li et al., 2021）**：联合优化不变表示和风险，但无法解决回归参数在域偏移下的稳定性问题。
5. **UDAC中DWT应用（Roy et al., 2019）**：首次将白化变换用于分类域适应，本文首次将其扩展至回归并提供OLS理论证明。
6. **FixMatch（Sohn et al., 2020）**：强/弱增强一致性框架，本文借鉴其增强策略并扩展至回归场景。

## 局限性与未来方向
1. **模态限制**：目前仅在图像数据上验证，时间序列和表格数据等广泛使用的回归模态尚未探索。
2. **理论假设**：OLS推导假设线性回归器，实际深层网络的非线性映射可能偏离理论边界。
3. **增强策略依赖**：强增强和mixup策略针对图像设计，跨模态迁移需重新设计。
4. **未来方向**：论文指出将扩展至时间序列和表格数据，需设计模态特定的增强和适配策略。

## 研究启发与可借鉴点
1. **白化变换的理论延伸**：将DWT从分类扩展到回归并提供参数方差缩减的理论证明，为其他特征预处理技术提供理论验证范式。
2. **方差一致性正则化设计**：用简单的方差对齐替代复杂的一致性计算，为回归任务的自监督学习提供轻量化方案。
3. **双一致性框架的通用性**：预测+特征双一致性可同时约束输出和中间表示，可迁移至其他半监督学习场景。
4. **超参敏感性分析**：论文展示$\lambda_1$和$\lambda_2$均在{0.1-0.5}范围内最优，表明方法对超参不敏感，适合实际应用。

## 关键术语表
**SSDAR（Semi-Supervised Domain Adaptation Regression）**：半监督域适应回归，利用少量目标域标签结合大量源域标签和无标签目标数据进行回归任务域适应。

**DWT（Domain-specific Whitening Transform）**：特定域白化变换，通过Cholesky分解将特征协方差矩阵转换为单位矩阵，消除特征间相关性。

**OLS（Ordinary Least Squares）**：普通最小二乘法，回归问题的经典解法，本文用于推导白化变换对参数方差的影响。

**Consistency Regularization**：一致性正则化，强制模型对同一输入的不同增强版本产生一致输出的半监督学习技术。

**Mixup Augmentation**：混合增强，通过线性组合两个样本的特征和标签生成新样本的数据增强方法。

## 可复现要素
- **数据集**：BIWI、QMUL、MPI3D均为公开数据集
- **代码**：已开源（https://github.com/sejin-sim/WhiteCon）
- **骨干网络**：ResNet50
- **优化器**：AdamW，学习率0.001，batch size 48
- **训练轮数**：50 epochs，前20个epoch线性warmup
- **超参数范围**：$\lambda_1, \lambda_2 \in \{0.1, 0.2, 0.3, 0.4, 0.5\}$
- **目标标签比例**：主要实验使用5%，扩展实验测试5%/10%/20%/30%

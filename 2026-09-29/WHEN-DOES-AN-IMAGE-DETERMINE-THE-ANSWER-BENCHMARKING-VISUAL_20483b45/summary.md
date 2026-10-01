---
title: "WHEN-DOES-AN-IMAGE-DETERMINE-THE-ANSWER-BENCHMARKING-VISUAL"
source: https://arxiv.org/pdf/2609.34480v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:38:34"
field: "多模态视觉推理与评估"
keywords: ["visual question answering", "abstention", "answerability", "benchmark", "evidence sufficiency", "joint success", "executable witness"]
innovations: ["跨三领域（图表/合成/照片）的统一完整问题评估框架", "可执行见证机制为图表缺失信息提供程序级证据", "联合成功指标揭示决策正确但答案错误的隐藏失败"]
benchmarks: ["PlotQA", "CLEVR", "GQA"]
---

# 论文速读：WHEN-DOES-AN-IMAGE-DETERMINE-THE-ANSWER-BENCHMARKING-VISUAL

## 一句话总结
论文提出了一个跨图表、合成场景和照片三类视觉领域的基准测试，要求多模态模型在同一问题的不同证据版本下，对所有可回答问题给出正确答案、对缺失证据问题正确拒答，并以可执行见证（executable witnesses）提供标签证据。

## 研究问题与动机
- 可靠视觉问答要求：证据充分时给出正确答案，证据不足时应拒答（abstention），现有基准多只评估单视角准确率，掩盖了"完整问题组内一致性"的失败
- 现有基准如CertainlyUncertain、MM-AQA仅测缺失证据下的拒答；Causal VQA测答案保持/改变编辑；HallusionBench测跨视觉上下文正确性，但均未将支持性编辑与缺失拒答统一在同一组问题上联合评分
- 隐藏对象可能留下线索（residual cues），需审计证据移除后的残留信息
- 单视图平均值会掩盖"必须同时满足所有状态"的组级任务失败，评估单元从"视图"变为"组"后排名可能反转

## 核心贡献（创新点）
- **跨三领域的受控基准构建**：在PlotQA（图表数值）、CLEVR（合成场景属性/遮挡）、GQA（照片依赖掩码）各构建1,000组问题；区别在于不仅构造问答对，还用源程序/可见性操作严格绑定每种状态（F/S/C/M/I）
- **完整问题评估与显式标签证据**：定义联合成功指标J_k，要求组内所有支持答案和所有拒答均正确；图表提供可执行见证对（admissible worlds with identical pixels but different answers），场景用源程序标注并审计残留线索
- **实证诊断揭示不同失败模式**：在六配置72,000条回复上，最佳配置的组级成功仅57.0%（PlotQA）、43.5%（CLEVR）、33.7%（GQA）；同一问题决策全对但仍可能出现错误答案（265/835组）
- **评估单元决定模型排名**：GLM 5.3 Flash单视图失败更少但影响更多问题，Qwen 3.8 Flash Next总失败更多但集中在较少问题，组级评分导致排名反转
- **残差线索审计**：在GQA中发现376组位置类问题（left/right/top/bottom）可能通过掩码位置保留空间线索，排除后仍能量化模型对指定残差的敏感度

## 方法详解
**响应义务定义（五种状态）**
- FULL (F)：原始图像
- A_SAME (S)：保持答案的编辑
- A_CHANGED (C)：改变答案的编辑
- U_MISSING (M)：被标注为缺失识别证据
- U_INVALID (I)：查询系列被移除

**图表（PlotQA）构造与见证**
- 使用D15模板：在一个x位置选择带标签系列并返回值
- 构造见证对满足：$W_0, W_1 \in \mathcal{W}_c, P_q(W_0) \neq P_q(W_1), O_{c,M}(W_0) = O_{c,M}(W_1) = I$
- 即两个合法完整图表产生不同答案，但经可见性操作后渲染像素完全相同
- 检查器验证：完整渲染、固定轴可接纳性、两次查找一致、跨世界答案不同、干预后像素相等
- 1,000组完整证明，覆盖5,000个图表视图

**公式定义**
- 兼容答案集：$\mathcal{A}(I; q, c, M) = \{P_q(W) : W \in \mathcal{W}_c, O_{c,M}(W) = I, P_q(W) \text{ defined}\}$
- 联合成功：$J_k = \frac{1}{G}\sum_g [\prod_{s \in S_d}(1-f_{gs})][\prod_{s \in \mathcal{A}_d}c_{gs}]$
- 决策成功：$B_k = \frac{1}{G}\sum_g \prod_{s \in S_d}(1-f_{gs})$
- 每视图失败：$E = \frac{1}{Gk}\sum_{g,s}f_{gs}$

**CLEVR场景**
- 使用官方Blender资产（Blender 2.79b/Cycles）
- 程序识别答案保持和答案改变属性编辑；遮挡隐藏必要属性
- 1,000组含408短文本、327布尔、265整数答案

**GQA照片**
- 基于场景图程序识别依赖关系
- 掩码覆盖非依赖框（S状态）和依赖框（M状态）
- 启发式匹配尺寸、宽高比和位置
- 1,000组含709短文本、291布尔问题

**评估协议**
- 六种配置（Qwen 3.8 Flash Next、Gemma 4 31B、Qwen 3.8 27B、Gemma 4 26B A4B、GLM 5.3 Flash、Molmo2-8B）
- 使用vLLM，temperature=0，512-token输出上限
- 提示明确区分"困难阅读"与"证据不足"
- 输出JSON格式：{"answerable": bool, "answer": string|null, "reason": str}
- 统计使用10,000次bootstrap组重采样，配对比较

## 实验与结果
**数据集规模**
- PlotQA：1,000组，5,000视图（F/S/C/M/I）
- CLEVR：1,000组，4,000视图（F/S/C/M）
- GQA：1,000组，3,000视图（F/S/M）
- 总计12,000视图，七万二千条回复

**主要结果（Table 4）**

| 配置 | PlotQA J_5 | CLEVR J_4 | GQA J_3 |
|------|-----------|-----------|---------|
| Qwen 3.8 Flash Next | 57.0% | 43.5% | 33.7% |
| Gemma 4 31B | 55.6% | 8.9% | 28.9% |
| Qwen 3.8 27B | 34.1% | 19.2% | 28.8% |
| Gemma 4 26B A4B | 40.7% | 5.4% | 30.0% |
| GLM 5.3 Flash | 18.2% | 17.0% | 24.5% |
| Molmo2-8B | 6.1% | 11.8% | 22.3% |

**关键发现**
- 最佳配置（Qwen 3.8 Flash Next）在图表上达到96.2%单视图决策准确率，但835组中仍有265组含错误支持答案
- 假阳性接受：11.3%–61.2%范围，GLM 5.3 Flash最严重（61.2%接受认证缺失输入）
- 虚假拒绝：Gemma 4 31B在3,000个支持视图中有949次虚假拒绝
- GQA位置类子群（376组）残差分析显示掩码位置本身可保留空间线索
- 评估单元变化导致排名反转：GLM单视图失败更少（741 vs 770），但Qwen组级成功更高（399 vs 305）

## 相关工作脉络
- **VQA基础**：VQA/VQA v2建立照片问答；FigureQA、DVQA、ChartQA扩展到图表；本文扩展至证据变化评估
- **拒答与证据充分性**：Selective prediction（Geifman & El-Yaniv）；SQuAD 2.0、AbstentionBench；VizWiz、UNK-VQA、MoHoBench；CertainlyUncertain、MM-AQA——本文配对支持编辑与拒答目标
- **受控变化与联合成功**：Causal VQA（答案不变性/变化）；HallusionBench（联合评分）；DynaMath、MuirBench——本文跨三领域组合
- **结构化推理与标签证据**：VISREAS、Super-CLEVR、CLOSURE——本文固定问题、变化证据
- **图表与文档**：ChartQA、ChartBench、CharXiv、DocVQA、InfographicVQA——本文聚焦可执行见证而非逻辑推理
- **视觉数学**：MathVista、MATH-Vision、MathVerse——本文分离任务难度与证据充分性

## 局限性与未来方向
- 图表验证限于固定轴、单一深度查找程序，递归程序仅通过1,456个案例审计
- 照片级认证未验证完整世界观察相等性（equation 2），仅做残差线索分析
- 支持性编辑仅约束 blanket rejection，未测试外观匹配控制或纯文本控制
- 扩展至更丰富视觉任务需区分编辑外观依赖与非视觉信息依赖
- 组级评分对错误集中度的敏感性可能导致不同应用偏好不同指标

## 研究启发与可借鉴点
- **可执行见证构造法**：用约束求解器（Z3）+枚举验证标签正确性，方法可迁移至其他需要严格负样本的基准
- **联合成功指标设计**：J_k公式将"所有状态必须正确"形式化，避免平均值掩盖极端失败
- **评估单元效应诊断**：通过类权重λ分析展示排名如何随指标变化，为benchmark设计提供方法论
- **残差线索审计框架**：对掩码后残留信息进行分层分析（location stratum vs other），可用于其他视觉编辑基准
- **错误分解可视化**：Figure 3的组级错误分解（joint success / decision-only / M failure / other）可直接复用于新基准诊断

## 关键术语表
- **Answerability（答案可确定性）**：问题是否能从当前图像中唯一确定的属性
- **Abstention（拒答）**：模型在证据不足时正确声明无法回答的行为
- **Joint success ($J_k$)**：要求组内所有支持答案和所有拒答同时正确的联合指标
- **Executable witness（可执行见证）**：一对合法完整世界，产生不同答案但相同观察像素
- **Residual cue（残差线索）**：掩码后仍保留的、可能泄露答案的视觉信息
- **Complete task success（完整任务成功）**：组级评分，区别于单视图准确率
- **Source program（源程序）**：CLEVR/GQA中生成问题和答案的可执行程序
- **Visibility operation（可见性操作）**：遮挡/隐藏部分数据区域的渲染操作

## 可复现要素
- **数据集**：PlotQA primary split、CLEVR validation split、GQA validation split，共12,000视图
- **代码/权重**：基准、标签证据和评估代码已开源：https://huggingface.co/datasets/sungguk/visual-answerability (v1.0.0)
- **模型配置**：vLLM，temperature=0，512-token上限，六配置见Table 5
- **超参**：未特别提及训练超参（零样本评估）
- **答案解析**：图表用精确有理数相等；场景用NFKC Unicode归一化+精确字符串匹配
- **统计**：10,000次bootstrap重采样，配对比较，seed=20260915

---
title: "TIMELINE-BENCH-EVALUATING-AGENTS-ON-REALISTIC-VIDEO-EDITING"
source: https://arxiv.org/pdf/2609.35143v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:36:10"
field: "多模态 Agent 评测"
keywords: ["Video Editing", "AI Agent Benchmark", "Multimodal Evaluation", "Creative AI", "LLM-as-judge", "Timelime Bench"]
innovations: ["首个要求端到端视频剪辑交付物的 Agent 基准，包含 56 个来自专业来源的完整任务", "以 2,582 份专业编辑盲审判断校准的 LLM 质量测试，实现主观创意质量与自动评测的桥接", "系统揭示 Agent 以静态帧和转录文本感知素材、以缺陷检查替代创意评判的行为模式"]
benchmarks: ["TIMELINE-BENCH", "GDPval", "VEBench", "MEDit-Bench", "AgenticVBench"]
---

# 论文速读：TIMELINE-BENCH EVALUATING AGENTS ON REALISTIC VIDEO-EDITING TASKS, FROM RAW FOOTAGE TO FINAL CUT

## 一句话总结
TIMELINE-BENCH 是一个包含 56 个真实视频剪辑任务的评测基准，要求 AI Agent 将原始素材与任务简报转化为完整的成片交付物；测试结果表明，目前最强 Agent（GPT-6 Astra + Codex CLI + 剪辑指导）仅解决 26.8% 的任务，主要瓶颈在于创意打磨与节奏把控，而非技术合规性。

## 研究问题与动机
- **现有评测缺乏对"完整创意交付物"的要求**：当前 AI Agent 基准（如 GDPval、Agents' Last Exam）侧重软件工程和程序性任务完成，很少要求产出可供专业使用的创意成品。
- **视频剪辑是极高难度的评估环境**：需要解释任务简报、理解源素材、协调画面/声音/图形在时间轴上的相互关系，决策高度依赖且不可孤立完成。
- **现有视频相关基准仅覆盖剪辑的局部能力**：VEBench、MEDit-Bench 等评测单一组件（如镜头选择、识别剪辑手法），或仅评估 GUI 操作轨迹，无法衡量从原始素材到成片的端到端能力。
- **视频剪辑任务没有单一正确答案**：同一简报允许多种有效剪辑方案，因此仅靠程序化测试无法判定输出质量，需要引入人类校准的质量评估。

## 核心贡献（创新点）
- **提出端到端的视频剪辑 Agent 基准 TIMELINE-BENCH**：包含 56 个来自专业来源（EditStock、Cinestudy、定制拍摄）的完整剪辑任务，每个任务提供源素材、任务简报、Docker 执行环境和参考成片，与现有仅评估剪辑片段或 GUI 轨迹的基准形成本质区别。
- **设计多层次的"通过即解决"评测协议**：每任务包含交付格式测试、内容退化测试、简报合规测试（程序化 + 模型裁判）和人工校准的质量测试，允许不同有效剪辑方案并存，而非寻找唯一正确答案。
- **构建首个针对视频剪辑质量的盲审人类偏好数据集**：由 43 位专业视频编辑完成 2,582 份双盲对比判断，并据此校准了由 Gemini 3.8 Flash、GPT-6 Astra、Claude Opus 5.5 组成的多模态 LLM 评委评分体系，实现了客观测试与主观质量之间的桥接。
- **全面评测 16 种前沿 Agent 并系统分析失败模式**：首次对比了 10 种模型在统一 OpenCode harness 下的表现，以及 Codex CLI、Claude Code 两种官方 harness 的差异，并揭示了 Agent 以静态帧/转录文本感知素材、检查缺陷而非评判质量的核心行为特征。

## 方法详解
- **任务组成**：每个 TIMELINE-BENCH 任务包含五项要素——编辑简报（brief）、源素材（source assets，平均 33.0 小时）、Docker 容器（预装 FFmpeg、Python/Node、Remotion、HyperFrames、AssemblyAI 转录等工具）、一套测试脚本和时间上限（300 分钟）。任务以 Harbor 格式打包，确保输入可复现。
- **测试体系（四类）**：
  - **交付测试（Delivery Tests）**：4 项/任务，检查输出文件是否存在、视频流、音频流和时长是否符合简报规格。
  - **内容测试（Content Tests）**：6 项共享测试，拒绝退化剪辑（大部分静音、画面冻结、重复镜头、死寂、声道不平衡、人脸被裁切）。
  - **简报合规测试（Brief Tests）**：180 项（每任务 1–6 项），其中 153 项为程序化测试（PP-OCR 读取屏幕文字、Whisper 转录语音、代码匹配片段与音频），27 项由三个多模态 LLM 裁判（Gemini 3.8 Flash、GPT-6 Astra、Claude Opus 5.5）三分之二多数裁定。
  - **质量测试（Quality Test）**：三个 LLM 裁判独立对 Agent 剪辑与参考剪辑分别打分（1–10 分），计算综合得分 z-score 平均值；通过阈值由该合集的人类偏好率拟合确定，确保 Agent 输出必须超过参考剪辑质量才能通过。
- **人类偏好研究**：43 位专业编辑（平均≥2年从业经验）在双盲条件下对比每对剪辑（Agent vs 参考），回答"您会为该项目选择哪个版本"，答案解锁条件为两个视频均完整播放完毕；每对由三位不同编辑独立评判，共 2,582 份有效判断。
- **Agent 配置**：10 种模型在统一 OpenCode harness 下运行；GPT-6 Astra、GPT-5.6 Sol 在 Codex CLI 下运行；Claude Fable 5.1、Claude Opus 5 在 Claude Code 下运行；此外设有"带剪辑指导的 GPT-6 Astra"（注入 1,909 词通用剪辑建议）和"计算机使用模式的 GPT-6 Astra"（通过鼠标键盘操作 DaVinci Resolve GUI，无命令行）。

## 实验与结果
- **数据集**：56 个任务，来源包括 EditStock（11 个， purchased）、Cinestudy（15 个，publicly available）、定制项目（30 个，newly commissioned）；共约 33.0 小时原始素材，素材时长中位数为 12.4 分钟，交付时长窗口 25–315 秒。
- **评估基线**：16 个 Agent（10 模型 × OpenCode + 4 模型 × 各自官方 Harness + 2 个变体），每个 Agent 每任务运行一次，共 896 次运行。
- **主要结果**：
  - 最佳 Agent **GPT-6 Astra + Codex CLI + 剪辑指导** 解决 15/56 任务（**26.8%**），其余接近的有 Claude Opus 5 + Claude Code（23.2%）和 GPT-6 Astra + Codex CLI（21.4%）。
  - 所有 Agent 平均解决率：**14.0%**（95% CI: 11.7%–16.4%）。
  - 人类编辑在 83.5% 的评判中选择参考剪辑，Agent 胜出/平局率平均为 16.5%。
  - 质量测试通过阈值与人类偏好率的 Spearman 相关系数 ρ = 0.93，Pearson r = 0.93。
- **最强结果提升**：添加剪辑指导将 GPT-6 Astra 在 Codex CLI 下的解决率从 21.4% 提升至 26.8%；而切换到计算机使用模式反而降至 3.6%，显著低于代码模式（p < 0.001）。
- **成本与效率**：最佳 Agent 单次运行成本约 $9.76，耗时约 21 分钟；成本与解决率呈正相关（ρ = 0.79）。

## 相关工作脉络
- **VEBench（Deng et al., 2026）**：评测多模态模型识别剪辑手法或从长视频中选镜头，仅评估组件能力而非端到端交付，无人类质量评估。TIMELINE-BENCH 在此基础上覆盖完整生产工作流。
- **MEDit-Bench（Ogata et al., 2026）**：基于消息驱动的叙事剪辑评测，以与专业剪辑的时间重叠度为指标。本文强调视频剪辑允许多种有效方案，不能仅以时间重叠度量质量。
- **AgenticVBench（Cao et al., 2026）**：评估 Agent 在后制作软件中的 GUI 操作轨迹和技术故障。本文同时评测代码 Agent 和 GUI Agent，并重点分析创意质量差距。
- **GDPval（Patwardhan et al., 2026）**：以专家盲审评价 AI 在经济价值任务中的表现。TIMELINE-BENCH 借鉴了其双盲专家对比范式，但专为此类无唯一正确答案的创意任务设计了质量测试校准机制。
- **Terminal-Bench / SWE-bench Pro**：Outcome-graded Agent 基准，关注软件工程和终端操作的正确性。本文定位差异在于：创意任务的"正确"是多解的，需要人类校准的质量测试补充。
- **CutVerse（Hu et al., 2026）**：GUI Agent 媒体后期制作基准，以里程碑检查评估操作轨迹。本文指出单纯评估 GUI 轨迹无法衡量成片质量，强调最终视频输出评测的重要性。

## 局限性与未来方向
- 每个 Agent 每任务仅运行一次，结果区间未涵盖运行间方差。
- 质量测试基于同一批人类编辑的判断进行校准，有效于 Agent 粒度而非单条剪辑粒度（κ = 0.15）。
- 参考成片由专业编辑制作而非绝对"ground truth"，不同专业编辑可能有合理分歧。
- 源素材和参考成片未公开发布（仅向获批研究团队开放），限制了社区的直接复现和扩展。
- 未来方向：开发原生音视频感知能力（而非静态帧+转录文本）、引入剪辑打磨工具（调色、音效、转场）、赋予 Agent 自主判断创意质量的元认知能力。

## 研究启发与可借鉴点
- **"质量测试校准"范式值得迁移**：对于无唯一正确答案的创意型任务，可借鉴本文"以人类盲审数据拟合 LLM 裁判通过阈值"的方法，在 LLM 裁判与人类判断之间建立量化映射。
- **过程行为分析揭示根本差距**：本文对 Agent 运行轨迹的细粒度分析（感知/构建/验证各阶段占比、检查目标分布）揭示了"会合规但不会创作"的本质，这种过程分析框架可复用于其他创意 Agent 评测。
- **Harness 与模型解耦评测设计**：将 10 种模型置于统一 OpenCode harness 下运行，使得模型能力与 harness 效果的归因更加清晰，这种方法论可直接复用于其他 Agent 横向评测。
- **单缺陷控制（Single-defect controls）验证测试有效性**：通过故意制造单一缺陷（静音、黑屏开头、重新排序、冻结画面）来验证测试能否正确拒绝缺陷输出，这一质量控制策略可推广至其他自动化评测体系。
- **Rasch 模型分离 Agent 能力与任务难度**：利用 Rasch 模型将人类偏好投票分解为 Agent 能力和任务难度两个独立维度，发现任务难度差异是 Agent 能力差异的 2.6 倍，这一统计框架可用于更精细的基准分析。

## 关键术语表
- **TIMELINE-BENCH**：一个包含 56 个真实视频剪辑任务的 Agent 评测基准，要求从原始素材到成片交付的端到端完成。
- **Resolution Rate（任务解决率）**：Agent 通过所有测试（含质量测试）的任务比例，是本文的核心性能指标。
- **Win-or-tie Rate（胜/平率）**：人类编辑认为 Agent 剪辑优于或等同于参考剪辑的判断比例，用于衡量主观质量。
- **Quality Test（质量测试）**：由三个多模态 LLM 裁判独立打分后取 z-score 均值，与基于人类偏好拟合的通过阈值比较的自动质量评估。
- **Brief Test（简报合规测试）**：检查 Agent 输出是否满足简报中明确规定的要求（如使用特定旁白、保持场景顺序），共 180 项。
- **Harbor Format**：用于打包任务的标准化格式，固定包含简报、输入文件、交付测试和环境定义，确保任务可复现。
- **Computer Use Mode（计算机使用模式）**：Agent 通过屏幕截图和鼠标键盘控制 GUI 软件（如 DaVinci Resolve）完成编辑，而非使用命令行工具。
- **Curated Editorial Guidance（剪辑指导）**：向 Agent 注入的 1,909 词通用剪辑建议文档，包含阅读简报、规划剪辑、审查交付等操作指南，不含任务特定答案。

## 可复现要素
- **数据集**：56 个 Harbor 格式任务已发布（https://timelinebench.tensortest.com），但源素材和参考成片仅向获批研究团队开放，不公开重分发。
- **代码**：Verifier 和任务定义以 Apache-2.0 开源发布，代码镜像位于 https://timelinebench.tensortest.com/code；提供可重算自动评测结果和人类研究统计数据的脚本。
- **权重/模型**：使用商业模型 API（GPT-6 Astra、Claude Fable 5.1、Gemini 3.8 Flash 等），非开源权重；harness 版本：OpenCode 1.18.31、Codex CLI 0.155.1、Claude Code 2.1.278。
- **关键超参**：每任务运行时间上限 300 分钟；环境为 32 vCPUs + 256GB 内存的 Linux 容器；质量测试评分 1–10 分制，z-score 标准化；剪辑指导 1,909 词；三个 LLM 裁判独立评分取均值。

---
title: "TIMELINE-BENCH-EVALUATING-AGENTS-ON-REALISTIC-VIDEO-EDITING"
source: https://arxiv.org/pdf/2609.35143v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:36:02"
field: "多模态 Agent 评测"
keywords: ["视频编辑", "AI Agent", "Benchmark", "多模态评估", "创意工作流", "LLM-as-judge"]
innovations: ["提出TIMELINE-BENCH：首个从原始素材到成片交付的完整视频剪辑agent评测基准", "设计三阶段混合验证体系（交付/内容/简报测试+校准LLM质量测试）", "通过轨迹分析揭示agent在创作任务中的系统性失败模式（感知→构建失衡、自检仅查缺陷不查质量）"]
benchmarks: ["TIMELINE-BENCH"]
---

# 论文速读：TIMELINE-BENCH: EVALUATING AGENTS ON REALISTIC VIDEO-EDITING TASKS, FROM RAW FOOTAGE TO FINAL CUT

## 一句话总结
本文提出了 TIMELINE-BENCH，一个包含 56 个真实视频剪辑任务的评测基准，要求 AI agent 从原始素材和任务简报出发完成完整视频编辑。实验评估了 16 个 agent，最好的 GPT-6 Astra（配合 Codex 与专业剪辑指导）仅解决 15/56 个任务（26.8%），且 73% 的失败源于质量测试而非合规性测试。

## 研究问题与动机
- **现有 agent 评测缺乏对创意交付物的评估**：当前主流基准（如 SWE-bench、GDPval）关注软件工程和科学计算等专业工作，但很少要求模型产出完整的创意成品（如一支剪辑好的视频）。
- **视频剪辑是高度复杂的专业工作流**：需要理解简报、统筹画面/声音/音乐/字幕的时间线，且各决策相互依赖（改动一个镜头会影响叙事节奏），现有的宽泛 benchmark 难以细致刻画这种编辑能力。
- **已有视频基准的局限**：VEBench、MEDit-Bench 等只评测剪辑的某个组件（如镜头选择、剪切列表），而不评估从原始素材到成片交付的完整流程；AGenticVBench、CutVerse 等关注 GUI 操作或事后检查，而非专业创作质量。
- **需要可校准的主观质量评估**：视频编辑允许多种合法方案，仅靠程序化测试无法衡量作品质量，需要结合人类专业编辑偏好和经校准的 LLM 质量测试。

## 核心贡献（创新点）
1. **提出 TIMELINE-BENCH 基准**：56 个从真实影视制作中采集的完整剪辑任务，每个任务包含原始素材、任务简报、Docker 执行环境和验证测试，要求 agent 产出可交付的最终视频；与前人工作（如 VEBench、MEDit-Bench）仅评测单步操作不同，本文要求端到端交付。
2. **设计三阶段混合验证体系**：将交付测试（4 项）、内容测试（6 项）和简报测试（180 项）与经过 43 位专业编辑盲评校准的 LLM 质量测试相结合；相比 GDPval 的纯人工评估，本文以 LLM 质量测试实现可自动化复用的主观判断。
3. **系统化评测 16 个前沿 agent**：覆盖 10 个模型 × 多个 harness（OpenCode、Codex CLI、Claude Code）及 guided/computer-use 变体，共 896 次运行；这是首个在完整视频剪辑任务上大规模横向比较 coding agent 与 GUI agent 的研究。
4. **揭示 agent 失败的深层模式**：通过轨迹分析发现 agent 将 54% 动作用于感知素材、仅 9% 用于实际构建，且主要检查技术缺陷而非节奏/故事质量；这一诊断为后续创作型 agent 设计提供了明确方向。

## 方法详解
- **任务构成**：每个任务 = 剪辑简报 + 原始素材（约 33 小时的原始素材，来源：EditStock 11 个包、Cinestudy 15 个项目、委托专业机构制作 30 个项目）+ Docker 容器（含 FFmpeg、Remotion、HyperFrames、AssemblyAI 转录等工具）+ 测试集合 + 300 分钟时限。
- **交付测试（4 项/任务）**：检查输出文件存在性及视频流、时长、音频流是否符合简报规格。
- **内容测试（6 项全局）**：拒绝退化编辑——大部分静音（>50%）、大部分冻结（>60%）、重复素材（>1s）、静音段超过 5s、声道不平衡（>6dB）、人脸被裁切。
- **简报测试（180 项）**：其中 153 项为程序化测试（使用 OCR 读取屏幕文字、Whisper 转录语音、PySceneDetect 检测剪辑点来验证内容）；27 项为可见内容测试，由 Gemini 3.8 Flash / GPT-6 Astra / Claude Opus 5.5 三个 LLM judge 二比三多数决。
- **质量测试设计**：三位多模态 LLM judge（Gemini 3.8 Flash 观看视频+音频；GPT-6 Astra 和 Claude Opus 5.5 读取每秒一张的联系表+四个测量指标：cut/min、最长静止帧、集成响度 EBU R128、静音占比）各自独立评分（1–10），转换为 z-score 后取面板平均分。通过阈值 $t_C$（在各 collection 中拟合，使通过率不超越人类 win-or-tie 率）：Cinestudy=0.30、Commercial=0.52、EditStock=0.21、UGC=1.54（judge 标准差单位）。
- **人类偏好研究**：43 位专业视频编辑（均 ≥2 年经验）对 863 对编辑进行盲评，共 2,582 条有效判断；agent 的 win-or-tie 率 = 人类偏好 agent 或无显著偏好的比例。
- **任务可验证性保障**：参考剪辑与输入隔离（MD5 哈希无匹配），agent 运行轨迹审计确认无网页检索参考视频行为；每个任务配备机械 oracle（低工艺渲染）用于交付测试验证。

## 实验与结果
- **数据集**：56 个任务，分四大 collection（Cinestudy 15 个、EditStock 11 个、UGC 15 个、Commercial 15 个），原始素材总计约 33.0 小时，素材时长中位数 12.4 分钟，交付时长窗口 25–315 秒。
- **评测基线**：16 个 agent，含 10 个模型在 OpenCode 下运行、4 个模型额外在 Codex CLI 和 Claude Code 下运行，外加 curated guidance 和 computer use 两个变体；共 896 次运行（每 agent×每任务一次）。
- **主要结果（任务解决率）**：
  - 最佳：**GPT-6 Astra + Codex CLI + curated guidance，26.8%**（15/56）；
  - Claude Opus 5 + Claude Code：23.2%；
  - GPT-6 Astra + Codex CLI（无指导）：21.4%；
  - 所有 agent 平均：14.0%（95% CI 11.7–16.4%）。
- **人类偏好结果**：人类编辑在 83.5% 判断中偏好参考剪辑，agent 仅获 11.1%，无显著偏好 5.5%；整体 win-or-tie 率 16.5%。
- **质量测试与人类偏好高度一致**：Spearman $\rho = 0.93$，Pearson $r = 0.93$，平均偏差仅 3.0 个百分点。
- **失败分析**：771 次未解决运行中 562 次（73%）仅失败质量测试；中位差距为 0.82 judge 标准差（约 1 分）。
- **计算机使用模式显著降低性能**：GPT-6 Astra in Computer Use 解决率仅 3.6%，win-or-tie 率 8.2%，比 coding 模式（26.8%）低约 23 个百分点，差异显著（$p < 0.001$）。
- **成本与解决率正相关**：Spearman $\rho = 0.79$，但最佳 agent 每运行约 \$9.76，并非最昂贵。

## 相关工作脉络
1. **GDPval（Patwardhan et al., 2026）**：评估 AI 在经济价值任务上的表现，采用专家盲评；TIMELINE-BENCH 定位不同在于聚焦视频剪辑这一特定创意工作流，且引入可自动化的 LLM 质量测试。
2. **VEBench（Deng et al., 2026）**：评测 LMM 的真实世界视频编辑能力，但仅输出剪切列表或时间跨度，无完整交付物；本文要求端到端产出成片。
3. **MEDit-Bench（Ogata et al., 2026）**：基于消息驱动的叙事视频编辑，评测单条长视频的剪切方案；本文涵盖更广的任务类型（商业广告、纪录片、UGC 内容等）。
4. **AGenticVBench（Cao et al., 2026）**：评测 agent 的后期制作 GUI 操作，关注技术失败分析；本文同时评测 coding agent 与 GUI agent，并以专业编辑的主观质量作为核心指标。
5. **CutVerse（Hu et al., 2026）**：GUI agent 媒体后期剪辑基准，以里程碑检查为主；本文通过 LLM 校准的质量测试更贴近真实创作评判。
6. **SWE-bench / Terminal-Bench（Jimenez et al., 2024; Merrill et al., 2026）**：代码/终端类 agent 基准；本文将同类"端到端任务完成"范式扩展到创意视频编辑领域。

## 局限性与未来方向
- **每个 agent 每个任务仅运行一次**，结果忽略了运行间方差，置信区间较宽（约 ±11 个百分点）。
- **质量测试仅在 agent 层面有效**，而非单次编辑层面（κ=0.15，接近单个人类编辑与多数的一致性）。
- **参考剪辑是专业作品而非认证 ground truth**，不同 valid editorial solution 之间可能存在合理差异。
- **未来方向**：需要原生音视频感知能力（而非依靠截图和转录）、集成 finishing 工具（调色、音效设计、图形）、以及自主判断创作质量的能力。

## 研究启发与可借鉴点
1. **三阶段混合验证框架可直接迁移**：交付/内容/简报测试 + 校准 LLM 质量测试的组合，适用于其他需主观质量判断的创作型任务（如平面设计、音乐编排）。
2. **"check-then-re-render"循环值得借鉴**：轨迹分析显示该模式与更高的人类偏好率相关（+2.1 分/SD），后续 agent 设计可强制引入自检-修正迭代。
3. **跨 harness 对比实验设计**：同一模型在不同 harness（OpenCode/Codex/Claude Code）下运行，隔离模型能力与工具链影响，为后续 agent 研究提供干净的对照范式。
4. **Rasch 模型分解 agent 能力与任务难度**：分离 agent skill 与 task difficulty 的统计方法可复用于其他 benchmark 的深度分析。
5. **单缺陷控制验证测试有效性**：通过构造单一缺陷样本（静音、黑屏、重排段落等）验证测试的敏感度，可推广到其他需要测试验证的基准构建中。

## 关键术语表
- **TIMELINE-BENCH**：本文提出的视频剪辑 agent 评测基准，包含 56 个从原始素材到成片的完整剪辑任务。
- **Task Resolution Rate**：agent 成功解决（通过所有测试）的任务比例，是本文的核心度量指标。
- **Quality Test**：由三位 LLM judge 独立评分后经 z-score 标准化的质量评估测试，阈值以人类编辑偏好校准。
- **Brief Test**：检查 agent 输出是否满足简报明确要求的测试，包括程序化测试（OCR/ASR/剪辑点检测）和 LLM judge 多数决。
- **Win-or-Tie Rate**：人类编辑偏好 agent 剪辑或与参考剪辑无显著偏好的判断比例，用于衡量主观质量。
- **Curated Editorial Guidance**：向 agent 注入 1,909 词通用剪辑建议（无任务特异性答案）的实验设置，可提升解决率。
- **Computer Use Mode**：agent 通过视觉界面操控鼠键而非命令行工具编辑视频的模式，本文发现其性能显著低于 coding 模式。
- **Harbor Format**：本文使用的任务封装格式，固定任务简报、输入、交付测试，确保可复现执行。

## 可复现要素
- **数据集**：56 个 Harbor 格式任务已发布（https://timelinebench.tensortest.com），但原始媒体素材需申请访问（仅限研究用途，禁止训练模型或重新分发）。
- **代码/验证器**：Verifier 代码及Frozen Constants 已开源（Apache-2.0），提供自动评估脚本，可在无参考剪辑情况下对新提交评分。
- **结果数据**：全部 896 次运行的 per-run 结果已公开。
- **关键超参**：质量测试阈值（Cinestudy=0.30, Commercial=0.52, EditStock=0.21, UGC=1.54 judge 标准差）；每运行限时 300 分钟；32 vCPU / 256GB 内存环境。
- **模型配置**：所有 agent 以最高 reasoning effort 运行；具体模型标识符、harness 版本和启动命令见附录 H。

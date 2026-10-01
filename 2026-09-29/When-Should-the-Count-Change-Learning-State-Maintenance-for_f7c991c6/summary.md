---
title: "When-Should-the-Count-Change-Learning-State-Maintenance-for"
source: https://arxiv.org/pdf/2609.35416v1.pdf
model: agnes-2.5-flash
chunks: 5
summarized_at: "2026-10-01 03:42:30"
---

# 论文速读：When-Should-the-Count-Change-Learning-State-Maintenance-for

## 一句话总结
提出 **StaMina**，一种显式学习“何时改变计数”的状态维护框架，通过阶段条件转换、延迟身份接受与可微路径监督，在 Full/Stream 双模式下显著超越 Counting‑SFT 与 Gemini‑3‑Flash，并配套构建多源计数数据管道与 SVCBench 等评测体系。

## 研究问题与动机
- **核心问题**：连续视频计数缺乏对“新观察 vs 新对象/完成事件”的显式区分机制，导致模型在长序列中频繁产生虚假更新或重复计数。
- **现有方法不足**：传统 SFT 计数基线（如 Counting‑SFT）依赖隐式上下文拟合，无法维护稳定的计数状态；缺乏对合法转换路径的可微监督与边界验证。
- **评测缺口**：既有视频理解基准偏重通用问答，缺少针对计数适应、在线视频迁移与计数条件动作预测的系统化评估。
- **方法诉求**：需要一种可微递推的状态机式架构，支持因果前缀回放与流式增量处理，并在身份分配、可见性判断与事件完成间实现解耦。

## 核心贡献（创新点）
1. **状态条件转换学习框架**：通过门控 delta 记忆与类型化状态记录（可见候选/已接受身份/已完成事件）显式建模计数维护过程。→ 与依赖隐式上下文的 SFT 基线本质区别在于引入可验证的状态机与合法转换路径监督。
2. **阶段条件与路径目标联合训练**：在 count 端点约束下结合 phase conditioning 与 verified‑boundary supervision，同步优化硬前缀精确匹配与虚假更新抑制。→ 区别于纯端到端计数损失，本文通过离散边界标签与后验传递实现误差可控的状态转移。
3. **延迟身份接受的对象分支**：独立学习可见性与关联，利用 stop‑gradient 阻断 NEW/DEFER 类别的梯度，实现候选的一对一关联最大化。→ 与常规实时 ID 分配不同，该设计将身份决策推迟至状态稳定期，降低早期噪声累积。
4. **多源数据管道与配套基准**：组织 39.8K 空间查询与自然/周期事件注释，构建 SVCBench；在 OVO‑Bench/OVO‑S‑Bench 上验证双模式性能。→ 填补因果视频计数从数据构建、监督设计到系统评测的完整链路空白。
5. **轻量模块化实例化**：基于 Qwen3‑VL‑4B/8B‑Instruct 冻结主干，仅训练 Event‑head 与 delta memory，支持 Full 回放与 Stream 在线推理。→ 与从头训练或密集微调方法不同，强调参数效率与双模式统一架构。

## 方法详解
- **状态条件转换学习**：维护三类类型化槽位，通过统一的 Event‑head 缓存因果特征、候选关联与生命周期掩码，以逻辑评分替代隐式预测。
- **帧采样与 Read 操作**：帧预算 $B_f$ 下均匀采样 $m=\min(M,B_f)$ 帧；Stream 模式按 1 fps 取最新源帧并压制重复索引；区间端点从同次 replay 读取，标签保持物理查询时间。
- **端点损失**：初始就绪状态 $F_0(a,n)=1[a=a_{\mathrm{READY}},\,n=0]$，计算 $Z_j=\sum_a F_{k_j}(a,y_j)$，最小化 $-\sum_j\log Z_j$，将 phase 后验向后传递。
- **边界损失**：验证相位边记录集 $\mathcal{M}_B=\{(k,\ell,a^*,b^*)\}$，对合法转移或验证 hold site 计算：
  $$\mathcal{L}_{\mathrm{boundary}}=-\frac{1}{|\mathcal{M}_B|}\sum_{(k,\ell,a^\star,b^\star)\in\mathcal{M}_B}\log p_{k\ell}^o(b^\star\mid a^\star)$$
  boundary times/phases 作为监督信号而非运行时输入。
- **对象分支与关联模块**：32 个目标条件 pooling query，候选 $x_{ki}=\sum_j\alpha_{kij}v_{kj}$（注意力宽度 $D=256$）；可见性 $\nu_{ki}=\sigma(w_v^\top x_{k i}+b_v)$，余弦相似度 $\geq 0.95$ 时抑制冗余；关联概率：
  $$p_{ki}(c)=\mathrm{softmax}_{c\in\mathcal{R}_{ki}\cup\{\mathrm{NEW,DEFER}\}}\,\mathrm{g}_{\mathrm{id}}(x_{ki},e_d,\bar{q}_c)$$
  其中 $\bar{q}_c$ 对历史原型使用 $\mathrm{sg}$ 算子，NEW/DEFER 携带 trainable class embedding。
- **推理模式**：Full 模式重置运行时并回放因果前缀（最多 64/128 帧）；Stream 模式维持 4096 token 窗口、fast weights 与类型化状态进行 1 fps 增量处理。

## 实验与结果
- **数据集与基准**：自构建管道 15,521 分组/383 父组/39,829 空间查询；训练/开发集 47,882/5,710；评测覆盖 SVCBench、OVO‑Bench（RT/BT/FA 三组）、OVO‑S‑Bench（L1‑L4 四级）。
- **SVCBench GPA**：StaMina‑4B Full 41.9 / Stream 36.4；StaMina‑8B Full 44.9 / Stream 38.2。8B 版本较 Counting‑SFT 提升 Full +10.9、Stream +3.2 分；Full 模式超越 Gemini‑3‑Flash（37.0）。
- **消融（Table 2，冻结 8B 特征）**：加入阶段条件+路径目标后，硬精确匹配 49.9 → 64.4，虚假更新 0.8 → 0.4；identity supervision 显著减少 hard cumulative‑count error 与 re‑entry double‑counting。
- **OVO‑Bench/S 性能**：STAMINA‑8B Full OVO‑Bench RT/BT/FA 为 60.71/52.97/52.68，L‑Avg 55.46/52.08；Stream 模式对应 56.34/49.92/49.98。较 Qwen3‑VL‑8B Full 在 BT/FA 与 L2/L3 追踪指标上均获提升。
- **效率**：Mean query latency Stream 0.48 s / Full 1.72 s。
- **结论**：完整轨迹提供跨查询持久性监督而非仅增加 QA 行数；计数适应后历史/空间上下文追踪显著提升，实时感知（RT/L1）略有下降。

## 相关工作脉络
1. **流式视觉上下文/长视频理解**：Fu et al. (2025), Wu et al. (2024), Liu et al. (2026c), Niu et al. (2025), Li et al. (2026) 等，本文聚焦计数特异性状态维护而非泛化理解。
2. **测试时训练与 Fast‑Weight 层**：Sun et al. (2020, 2025), StreamTTT (Chen et al., 2026), Spatial TTT，本文引入类型化状态与可微转换头而非纯权重在线更新。
3. **现有视频计数/SFT 基线**：

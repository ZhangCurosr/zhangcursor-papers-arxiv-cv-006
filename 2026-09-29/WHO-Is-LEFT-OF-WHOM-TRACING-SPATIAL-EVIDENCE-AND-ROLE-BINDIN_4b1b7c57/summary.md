---
title: "WHO-Is-LEFT-OF-WHOM-TRACING-SPATIAL-EVIDENCE-AND-ROLE-BINDIN"
source: https://arxiv.org/pdf/2609.35486v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:39:42"
---

# 论文速读：WHO-Is-LEFT-OF-WHOM-TRACING-SPATIAL-EVIDENCE-AND-ROLE-BINDIN

## 一句话总结
本文通过配对扰动测试与激活修补技术，系统追踪了 VLM 与 LLM 在相对位置推理中对物体空间位置的表征传递路径及目标/参考角色绑定机制，揭示高实例准确率下普遍存在的位置交换与角色反转不一致性；进一步提取的稳定角色方向可实现零训练干预，在自然图像基准上显著提升推理一致性与准确率。

## 研究问题与动机
- **准确率掩盖一致性缺陷**：现有空间推理评测以实例级准确率为核心，但模型可能在单一查询上答对，却在物体位置交换（object-location swap）或目标/参考角色反转（target/reference role reversal）下给出错误答案，缺乏对内部表征稳健性的刻画。
- **双计算组件缺乏因果分离**：相对位置判断（如“X在Y的左边吗？”）需同时完成（i）恢复物体空间位置、（ii）绑定目标与参考角色；现有机制研究多停留在表征可视化，未在同一框架内剥离并验证这两条互补通路的因果贡献。
- **跨模态与跨架构表征对比空白**：已有工作（如 Kang et al., 2026; Cui et al., 2026）仅在图像输入条件下识别空间 ID，未系统比较视觉/文本输入、VLM/纯 LLM backbone 三者之间的位置编码与传递差异。
- **缺乏可迁移的轻量干预手段**：如何将合成场景中学到的表示结构推广至真实图像基准，并在不更新模型参数的前提下提升推理鲁棒性，仍是开放问题。

## 核心贡献（创新点）
- **提出配对一致性评测框架**：构造位置交换与角色反转两类扰动，证明高准确率不等于关系推理的一致性，弥补单一准确率指标的评估盲区。
- **建立跨模态源-查询位置因果链**：通过 activation patching 验证源端场景证据对查询端位置 ID 的因果影响，并将该机制从视觉条件推广至文本输入与 LLM backbone。
- **首次量化提取稳定角色方向**：识别出与目标/参考角色绑定的独立隐状态方向，证明其几何稳定性，且破坏该方向会显著削弱角色敏感型预测。
- **实现零训练方向干预迁移**：在合成数据上估计的角色方向无需微调即可泛化至 What'sUp 与 COCO-Spatial 等自然图像基准，在多数设置下同步提升准确率与两类配对一致性。

## 方法详解
- **配对扰动设计**：定义 $\tau_{\text{loc}}(x) = (S^{t \leftrightarrow r}, q(t, r))$（交换两物体位置但查询不变）与 $\tau_{\text{role}}(x) = (S, q(r, t))$（场景不变但交换目标/参考角色），二者均使正确答案反转，用于严格检验模型是否真正理解空间关系。
- **Activation Patching 定位**：在 clean/corrupt pair 上，将指定层 $\ell$ 与语义 token 组 $g$ 的残差流状态替换，计算 Clean-Answer Recovery Rate $\mathrm{CARR}_{\ell,g} = \mathbb{E}[\mathbf{1}\{M(x_{\mathrm{patch}})>0\}]$ 与边际恢复分数，追踪位置交换与角色反转下有效 patch 位置随深度的迁移路径。
- **Location ID 提取**：在源端（视觉 bbox/strip 或文本描述 span）与查询端（target/reference 提及 span）分别按属性去均值：$\bar{h}_{i,o,\ell}^g = h_{i,o,\ell}^g - \mu_{\rho,i,o,\ell}^g$，再按位置标签取原型 $\mathrm{ID}_{\pi,\ell}^g = \mathbb{E}[\bar{h}^g \mid \pi_{i,o}=\pi]$；构造空间轴 $v^g$ 为两端 ID 之差，验证其解码能力与几何有序性。
- **源到查询因果验证**：将 corrupt 源端状态 patch 入 clean 运行，测量下游层查询端位置比较分数偏移 $\Delta m^{\to \text{corr}}$ 与 corrupt-answer flip rate (CAFR)，证明源端表征对查询端位置信息的因果驱动。
- **Query-side Location-ID Steering**：在查询端 token 上沿真实位置 ID 差方向施加干预 $h_k^{(\ell)} \leftarrow h_k^{(\ell)} + \alpha(\mathrm{ID}'_{\pi_o} - \mathrm{ID}_{\pi_o})$，验证该方向对关系预测的决定性作用，并排除 norm-matched random/shuffled 控制方向。
- **Role Direction 估计与干预**：利用 Target-First (TF) 与 Reference-First (RF) 模板构造联合对象级角色对比 $d_{p,o}^{(\ell)}$，取平均得稳定方向 $d_{\text{role}}^{(\ell)}$；通过破坏性更新（$h_{\text{target}} \leftarrow h_{\text{target}} - \alpha d_{\text{role}}, h_{\text{reference}}

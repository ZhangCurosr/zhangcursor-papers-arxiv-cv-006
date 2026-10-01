---
title: "Unified-Trajectory-Matching-Policy-Optimization-Diverse-T2I"
source: https://arxiv.org/pdf/2609.34688v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 11:09:38"
field: "生成模型强化学习后训练"
keywords: ["trajectory matching", "diffusion RL post-training", "mode collapse", "text-to-image generation", "vision-language-action", "forward KL", "diversity preservation", "OOD generalization"]
innovations: ["统一group-wise前向KL轨迹分布匹配框架，避免reward最大化导致的mode collapse", "progress-conditioned coarse-to-fine T2I采样调度器提升探索与效率", "feedback-conditioned action-chunk VLA采样支持多策略保留与OOD泛化"]
benchmarks: ["GenEval", "OCR Accuracy", "PickScore", "LIBERO", "MetaWorld-MT50", "CALVIN", "WallDetour"]
---

# 论文速读：Unified-Trajectory-Matching-Policy-Optimization-Diverse-T2I

## 一句话总结
本文提出**Uni-TMPO**（Unified Trajectory Matching Policy Optimization），一种用于扩散/流策略的统一RL后训练框架，通过将标准化reward转化为group-wise Boltzmann目标分布并用forward KL匹配，有效避免reward-maximizing导致的mode collapse，在T2I图像生成和VLA操作泛化中同时提升reward、多样性与OOD鲁棒性。

## 研究问题与动机
- **Reward-maximizing RL导致mode collapse**：即使在reference KL或entropy正则下，最大化期望reward仍使策略坍缩到单一高分mode，T2I产生外观/风格相似的图像，VLA丢失备选成功策略。
- **现有正则化手段不足**：Reference KL和transition entropy无法持续保留不同的完整输出或行为模式；显式多样性奖励需要领域特定的评估器，难以统一。
- **proxy reward上升但感知质量下降（reward hacking）**：继续优化可能提升代理reward，同时降低图像多样性与视觉质量。
- **缺乏统一框架**：T2I与VLA的轨迹结构不同（完整去噪轨迹 vs 多chunk反馈条件轨迹），但两者均基于高斯噪声+随机转移log probability，适合统一的目标设计。

## 核心贡献（创新点）
- **Group-wise forward KL轨迹分布匹配**：将标准化reward转为目标分布$q$，与策略分布$p_\theta$做前向KL；与reward最大化本质区别在于**保留多模态概率质量而非选择单winner**。
- **Progress-conditioned coarse-to-fine T2I采样调度器**：早期在浅层分支探索全局结构，后期在深层细化局部细节；相比TreeGRPO随机窗口调度，迭代时间降低**10.2%**。
- **Feedback-conditioned action-chunk VLA采样**：每次planning根据当前observation查询action expert，轨迹score由所有chunk的stochastic flow log probability均值构成；与π_RL等baseline的本质区别在于**显式匹配多chunk完整轨迹的概率分布**。
- **统一目标+领域采样**：T2I与VLA共享同一优化目标，仅采样过程适配领域差异；相比DAG/DGFS等方法拟合中间flow或subtrajectory，Uni-TMPO**直接匹配完整终端reward induced分布**。
- **实证最佳reward–diversity–efficiency权衡与OOD泛化**：T2I LGMD比最强baseline高**28.4%**；VLA MetaWorld/task OOD平均提升**8.0 pp**、scene OOD提升**4.8 pp**；WallDetour与真实机器人blocked-target实验验证多策略保留价值。

## 方法详解
- **轨迹表示统一**：context-conditioned随机扩散/流轨迹$\tau$，T2I中$c$为prompt、$\tau$为完整去噪轨迹；VLA中$c=(\ell,\rho)$为语言指令与初始标识，$\tau$为多chunk物理rollout与stochastic action flow决策。
- **Group构建**：对每个context采样$K$条轨迹组成$\mathcal{G}(c)=\{\tau_i\}_{i=1}^K$，每条提供terminal reward $R_i$与differentiable trajectory score $s_{\theta,i}$。
- **Target分布**：组内标准化reward $\widetilde{R}_i$，构造Boltzmann目标 $q_i = \frac{\exp(\beta \widetilde{R}_i)}{\sum_j \exp(\beta \widetilde{R}_j)}$；温度$\beta$控制偏好强度，较大$\beta$趋近winners-take-all，适中值保留多轨迹概率。
- **Policy分布**：对轨迹score做组内softmax $p_{\theta,i} = \frac{\exp(s_{\theta,i})}{\sum_j \exp(s_{\theta,j})}$；共用项抵消，仅需相对概率。
- **Forward KL损失**：$\mathcal{L}_{TM}(\theta)=\frac{1}{|\mathcal{B}_{valid}|}\sum_{\mathcal{G}}\mathrm{KL}(q\|p_\theta)$；仅更新reward方差超过$\epsilon_R$的group。梯度形式 $\nabla_\theta \mathcal{L}_\mathcal{G}=\sum_i(p_{\theta,i}-q_i)\nabla_\theta s_{\theta,i}$，当$p<q$时提升该轨迹概率，反之降低。
- **T2I采样（Algorithm 2）**：共享前缀+progress-conditioned coarse-to-fine branching；在预定分支层$b_\eta$处branch因子$b$，记录stochastic transition log density $s^{\mathrm{img}}_{\theta,i}=\sum_{k\in\mathcal{K}^{\mathrm{sto}}_i}\log p_\theta(x_{i,k-1}|x_{i,k},c)$。
- **VLA采样（Algorithm 3）**：feedback-conditioned replanning；每步预测action chunk $\mathbf{A}^{\mathrm{pred}}_{i,t}$，执行prefix后获取新observation；chunk score $\ell^{\mathrm{chunk}}_{\theta,i,t}=\sum_{k}\log p_\theta(\mathbf{u}^{(k+1)}|\mathbf{u}^{(k)},\mathbf{o}_{i,t},\ell)$；轨迹score取均值 $s^{\mathrm{VLA}}_{\theta,i}=T_i^{-1}\sum_t \ell^{\mathrm{chunk}}_{\theta,i,t}$。
- **On-policy更新**：reward detach，每轮fresh采样；VLA中仅训练~300M action expert，视觉-语言backbone冻结。

## 实验与结果
- **T2I**：基于FLUX.1-dev + LoRA，三协议（GenEval/OCR/PickScore）。Uni-TMPO在GenEval达**0.954**、OCR **0.951**、PickScore **24.301**，LGMD最高（对比Flow-GRPO等提升**28.4%**），迭代时间**91.9s/76.3s/68.3s**，优于TreeGRPO且多样性未下降；Multi-reward（OCR+PickScore联合调度权重3:1→1:3）OCR 0.932、PickScore 23.897、LGMD 0.155。
- **VLA ID**：基于SFT初始化的OpenPI（$\pi_0$、$\pi_{0.5}$），LIBERO/MetaWorld-MT50/CALVIN-D上均超越Flow-SDE/Flow-Noise最强reward-maximization baseline（$\pi_0$: +0.2/+2.8/+2.2 pp；$\pi_{0.5}$: +0.3/+2.1/+2.2 pp）。
- **VLA OOD**：MetaWorld ML45（45 tasks训练、5 unseen eval）：$\pi_0$达**69.9%**、$\pi_{0.5}$达**59.3%**，分别超baseline **4.4 / 11.5 pp**；CALVIN ABC→D场景迁移Len-5：$\pi_0$ **61.9%**、$\pi_{0.5}$ **82.1%**，分别超baseline **6.6 / 3.0 pp**。
- **多策略保留（WallDetour/真实机器人）**：LEFT/RIGHT两路径reward相近但LEFT略优；Reward Maximization分配**0.95/0.05**、熵**0.29**；Uni-TMPO分配**0.55/0.45**、熵**0.99**；阻塞LEFT后，Reward Maximization失败而Uni-TMPO通过RIGHT完成。

## 相关工作脉络
- **Flow-GRPO / TreeGRPO**：基于online RL的reward最大化；与本文的关键差异为**不匹配完整轨迹分布**，易collapse到单mode。
- **DAG-DB / DGFS-SubTB**：拟合state-dependent flow或subtrajectory约束；本文**直接匹配完整轨迹终端reward分布**，在FLUX短随机轨迹设置下效率更高（迭代时间低42–49%）。
- **π_RL (Flow-SDE/Flow-Noise)**：VLA reward-maximization baseline；本文在其轨迹表述基础上扩展为**多chunk完整轨迹matching**，避免单一行为策略被强化。
- **GRAPE / ConRFT / SimpleVLA-RL**：偏好/离线价值/在线RL的VLA后训练；本文聚焦**统一目标+group-wise分布匹配**而非pairwise preference或一致性policy。
- **FlowRL**：为Boltzmann trajectory matching学习prompt-conditioned normalizer $Z_\phi$；本文**组内归一化消去normalizer**，无需额外学习器。
- **TMPO (Li et al., 2026)**：针对T2I的Softmax Trajectory Balance；本文统一框架在此基础上增加**VLA适配与统一forward-KL目标**。

## 局限性与未来方向
- **Group size依赖**：需足够$K$以使 diverse mode 出现于组内后才能被匹配；过小group可能限制多样性覆盖。
- **Scheduler经验性**：progress-conditioned coarse-to-fine调度器依赖预设schedule，泛化至其他模型深度/分支结构需重新调参。
- **仅验证扩散/流策略**：当前框架限于stochastic diffusion and flow policies，未涵盖确定性策略或其他生成架构。
- **仅T2I与VLA两域**：未扩展到视频、音频、3D、分子/蛋白设计等潜在应用。
- **未来方向**：作者提出可延伸至视频、音频、3D生成，以及分子与蛋白质设计、diffusion-based planning等。

## 研究启发与可借鉴点
- **Group-wise forward KL匹配可迁移**：任何基于stochastic transition的生成任务（如video/audio 3D diffusion）均可套用“reward→target distribution → policy softmax → KL匹配”的统一范式。
- **Progress-conditioned scheduler思想可复用**：早期探索全局结构、后期细化局部的分支调度策略，适用于任意具有逐步去噪/逐步细化特性的生成流程。
- **Feedback-conditioned replanning构建轨迹score**：VLA中每chunk取stochastic log probability均值，为多步、observation-dependent序列决策提供了可微分的groupwise score计算范式。
- **LGMD/Cosine diversity联合评估**：VAE latent近重复检测+DINOv2特征空间多样性可成为T2I/图像生成RL后训练的标配评估组合。
- **Blocked-target ablation可作为泛化诊断**：WallDetour与真实机器人阻塞实验为VLA多策略保留提供直观、可复现的OOD泛化验证范式。

## 关键术语表
- **Uni-TMPO**：Unified Trajectory Matching Policy Optimization，统一轨迹匹配策略优化，面向扩散/流策略的RL后训练框架。
- **Group-wise forward KL**：在每个context采样组内计算目标分布与策略分布的前向KL，驱动轨迹概率向reward诱导分布对齐。
- **Boltzmann target distribution**：利用$\exp(\beta R)$对组内trajectory reward进行归一化，构造概率目标分布$q$。
- **Progress-conditioned coarse-to-fine sampling**：T2I分支调度策略，早期在浅层扩散步骤分支以探索全局结构，后期在深层分支以细化局部细节。
- **Feedback-conditioned action-chunk sampling**：VLA在线重规划采样，基于当前observation反复查询action expert生成chunk并执行前缀。
- **Mode collapse**：reward最大化后策略坍缩至单一高分输出模式，丧失多样性与备选策略。
- **LGMD (Latent Group Mode Diversity)**：基于VAE latent空间的近重复图像检测指标，值越高表示组内多样性越好。
- **π_RL**：基于flow模型的VLA online RL后训练框架，本文使用的reward-maximization baseline。

## 可复现要素
- **数据集/环境**：T2I使用FLUX.1-dev + GenEval/OCR/PickScore基准；VLA使用RLinf + OpenPI (π₀/π₀.₅) + LIBERO/MetaWorld/CALVIN/WallDetour。论文未明确声明数据公开状态。
- **代码/权重**：论文未提及开源；模型名称为Uni-TMPO，未给出repo链接。
- **关键超参**：group size $K=27$（T2I三分支×三层）；分支因子$b=3$；temperature $\beta$（论文未列出具体值，见supplement）；threshold $\epsilon_R$；VLA rollout batch=64 environments、每组8条、8 epochs/iter共512 trajectories；action expert约300M参数，vision-language backbone冻结。

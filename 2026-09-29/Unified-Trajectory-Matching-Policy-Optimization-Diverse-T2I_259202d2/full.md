# Unified Trajectory Matching Policy Optimization: Diverse T2I Generation and VLA Generalization

Zhiyuan Ma, Member, IEEE, Jiaming Li, Lingzhen Li, Yu Liu, Xuekai Zhu, Dingkang Liang, Kaiyan Zhang, Jianjun Li, Bowen Zhou, Fellow, IEEE and Xiang Bai, Fellow, IEEE

Abstract—Reward-maximizing reinforcement learning (RL) is widely used to post-train stochastic diffusion and flow policies for text-to-image (T2I) generation. However, reward-maximizing RL causes policy mode collapse even under reference KL or entropy regularization, reducing the policy to a single high-reward mode. In T2I, this produces similar images and reward hacking. When extended to vision-language-action (VLA) models, the same collapse removes alternative successful strategies and weakens task and scene generalization. To address this limitation, we introduce Unified Trajectory Matching Policy Optimization (Uni-TMPO), a unified RL post-training framework for diffusion and flow policies. First, Uni-TMPO converts standardized rewards into a target distribution within each trajectory group and derives the policy distribution from trajectory log probabilities. Then, forward Kullback–Leibler optimization matches the two distributions instead of maximizing expected reward. A progress-conditioned coarse-to-fine scheduler efficiently constructs T2I trajectories. Within the unified framework, feedback-conditioned sampling uses updated observations to construct VLA trajectories. Extensive experiments show that Uni-TMPO achieves higher T2I rewards and VLA ID success rates than the strongest baselines. More importantly, it achieves the best T2I reward–diversity–efficiency trade-off and VLA generalization to held-out tasks and scenes, while real-robot evaluation demonstrates the value of multiple action strategies when the higher-reward target is blocked.

Index Terms—Diffusion policies, flow-matching policies, generative diversity, mode collapse, out-of-distribution generalization, reinforcement learning post-training, text-to-image generation, trajectory distribution matching, vision-language-action models.

## 1 INTRODUCTION

EWARD-maximizing RL has become a common postbeen widely studied in T2I generation, where it improves image composition, text rendering, and human preference [1]–[3]. More recently, the same paradigm has been extended to VLA action generation, where online RL improves task completion beyond imitation learning [4], [5]. Both applications start from Gaussian noise but sample trajectories differently. T2I produces one image through a complete denoising process, whereas VLA generates multiple action chunks as observations change. Although the sampling procedures differ, both policies assign probabilities to complete trajectories through stochastic diffusion or flow transitions. Each trajectory then receives a reward based on the final image or manipulation result. This shared structure provides a common basis for unified RL post-training and raises a central question: How can we achieve RL post-training for image and action generation within a unified framework while preserving diverse solution modes and supporting OOD generalization?

A central obstacle is policy mode collapse under rewardmaximizing post-training, where the policy collapses to a single high-reward mode. Sampling images or actions from independent Gaussian noise appears to introduce randomness, but it does not preserve distinct modes after reward-maximizing post-training. As shown in Figs. 4 and 6, the resulting policy still maps different noise initializations to similar final images or the same dominant action strategy. In T2I, different pixel-noise samples can produce images with similar object appearances, aesthetic styles, or spatial layouts. Proxy rewards may also rise even as perceptual quality or diversity declines, leading to reward hacking. The recent extension to VLA inherits the same problem. Different actionnoise samples can produce one dominant manipulation strategy. As a result, alternative successful routes or action sequences disappear, weakening task and scene generalization. Reference KL and transition entropy can regularize post-training, but they do not consistently retain distinct complete outputs or action strategies. Although explicit diversity rewards directly target this problem, they require domain-specific evaluators [6], [7]. A unified framework should therefore use a shared optimization objective to preserve multiple successful trajectories while supporting the different trajectory structures of T2I and VLA.

To answer this question, we introduce Unified Trajectory Matching Policy Optimization (Uni-TMPO), a unified RL post-training framework for stochastic diffusion and flow policies, as shown in Fig. 1. For both T2I and VLA, the framework takes a context-conditioned trajectory group, one reward per trajectory, and the corresponding stochastic transition log probabilities as its training inputs. Uni-TMPO replaces expected-reward maximization with trajectory distribution matching within each group sampled under the same context. Standardized rewards form the groupwise Boltzmann target distribution q, while accumulated transition log probabilities form the policy distribution p<sub>θ</sub>. Forward KL then moves p<sub>θ</sub> toward q and is evaluated exactly over the sampled trajectories. The target can assign substantial probability to multiple high-reward trajectories instead of selecting only the top-ranked sample. Matching this target raises the probability of high-reward trajectories while retaining diverse image modes and alternative action strategies represented in the group. With this objective fixed, trajectory construction is adapted to each application. The T2I implementation collects complete denoising trajectories with progress-conditioned coarse-to-fine sampling. It places branches at earlier denoising steps during the early training stage to explore global structures, then moves them to later steps to refine local details with less redundant denoising. Within the unified framework, feedback-conditioned actionchunk VLA sampling builds trajectories through repeated policy queries. At each query, the policy generates a chunk from the current observation and language instruction; executing its prefix yields the observation for the next query. The transition log probabilities of all chunks jointly determine the trajectory’s relative probability within the group. Together, the shared objective and domain-specific trajectory construction provide a unified framework for image and action generation.

![](images/38eafc8f2b93621630b81cc3b889347e1cb3c80e99390b441509ff4cf2785c31.jpg)  
Fig. 1. Overview of Uni-TMPO for T2I generation and VLA manipulation. Progress-conditioned coarse-to-fine sampling constructs T2I trajectory groups for each prompt, while feedback-conditioned action-chunk sampling constructs VLA trajectory groups from a shared initialization. Unlike scalar reward maximization, Uni-TMPO matches reward-induced and policy-induced allocations within each group, preserving diverse image modes and action strategies. The right panels summarize T2I performance and VLA OOD generalization.

The unified framework and its experimental findings are summarized as follows:

Unified RL post-training framework. We introduce a unified trajectory-matching framework for RL posttraining of stochastic diffusion and flow policies. It uses groupwise forward KL to match the rewardinduced and policy-induced distributions over complete trajectories. We apply this framework to T2I and VLA, with sampling adapted to each domain.

Progress-conditioned coarse-to-fine T2I sampling. The scheduler explores global structures early and refines local details later. Under the same trajectorymatching objective and settings, it achieves higher reward and diversity than the TreeGRPO scheduler while reducing iteration time by 10.2%.

Feedback-conditioned action-chunk VLA sampling. Each trajectory comprises action chunks generated under changing observations and a shared language instruction. Their transition log probabilities provide trajectory probabilities for the unified framework.

Extensive evaluation. Under single-reward and jointreward T2I optimization, Uni-TMPO achieves the best reward–diversity–efficiency trade-off, with LGMD up to 28.4% higher than the strongest baseline. It also achieves the best VLA ID results and exceeds the strongest reward-maximization baselines by an average of 8.0 percentage points on task OOD generalization and 4.8 points on scene OOD generalization. Real-robot evaluation further shows how multiple action strategies support task completion when the higher-reward target becomes unavailable.

## 2 RELATED WORK

## 2.1 T2I Post-Training and Generative Diversity

Diffusion and flow samplers expose sequential stochastic transitions that support policy optimization. DDPO formulates image denoising as a finite-horizon decision process. DPOK adds a reference-policy constraint, while Flow-GRPO applies online RL with group-relative rewards [1]–[3]. Later work improves exploration and stability through tree collection, regulated clipping, replay, and reward regularization [7]–[11]. DRaFT backpropagates differentiable rewards through denoising. Diffusion-DPO and D3PO instead learn from pairwise preferences [12]–[14]. These methods improve reward but often collapse the policy onto a narrow set of high-reward image modes, reducing generative diversity.

Beyond mode collapse, continued optimization can increase proxy rewards even as perceptual quality and diversity decline, leading to reward hacking [15]–[17]. T2I studies report this behavior together with the trade-off between image quality and diversity [7], [10], [18], [19]. To improve diversity, existing methods evaluate multiple images generated from the same prompt using a predefined reference distribution [6]. Uni-TMPO instead constructs the groupwise target distribution directly from the scalar rewards already used for conventional RL post-training.

## 2.2 VLA Post-Training and OOD Generalization

Many manipulation tasks admit multiple successful action sequences. ACT predicts temporally coherent action chunks, and Diffusion Policy models conditional multimodal action distributions through denoising [20], [21]. VLA models such as RT-1, RT-2, Open X-Embodiment, Octo, OpenVLA, π<sub>0</sub>, and $\pi _ { 0 . 5 }$ condition policies on vision and language [22]–[28].

RL post-training of stochastic generative policies has more recently been extended to VLA. GRAPE uses trajectory preferences, ConRFT combines offline values with online updates to consistency policies, and SimpleVLA-RL scales interaction across manipulation tasks [5], [29], [30]. π<sub>RL</sub> makes the action-flow policies of $\pi _ { 0 }$ and $\pi _ { 0 . 5 }$ stochastic and provides tractable transition log probabilities for onpolicy optimization [4]. Uni-TMPO uses the same policy formulation but matches complete trajectories composed of multiple action chunks. We therefore use π<sub>RL</sub> as the corresponding reward-maximization baseline. Recent work shows that RL can improve task success while reducing a generative robot policy to one behavior [31]. This loss of alternative strategies motivates trajectory matching for VLA.

Losing alternative action strategies is especially harmful under OOD conditions, because the behavior favored during RL may fail when the task or scene changes. Accordingly, VLA generalization is evaluated using held-out tasks, scene transfer, and controlled execution changes. Online RL can improve generalization under some shifts, although visual robustness remains challenging [32]. MetaWorld and CALVIN test held-out tasks and scene transfer, respectively [33], [34]. Controlled benchmarks such as the Colosseum further test policy responses when a previously successful behavior becomes invalid [35]. Together, these settings motivate our evaluation of task and scene OOD generalization.

## 2.3 Trajectory, Flow, and Distribution Matching

KL-constrained policy optimization has a long history. REPS derives exponential reward weighting under a relativeentropy constraint, and MPO alternates between a rewardweighted nonparametric target and a parametric projection [36], [37]. PPO optimizes a clipped likelihood-ratio surrogate, while GRPO replaces a learned value baseline with statistics from trajectories sampled under the same context [38], [39]. These objectives support reward optimization or constrained policy improvement, but they do not directly match the probabilities of complete trajectories.

In contrast, GFlowNets sample complete trajectories in proportion to reward, while Trajectory Balance enforces consistency over complete paths [40], [41]. FlowRL applies Boltzmann reward matching to LLM reasoning and learns a prompt-conditioned normalizer $Z _ { \phi }$ [42]. Uni-TMPO instead constructs the reward-induced and policy distributions within each group, removing the learned normalizer.

For diffusion models, DAG and Diffusion Generative Flow Samplers (DGFS) fit state-dependent flows with detailed-balance or subtrajectory constraints along a diffusion chain [43], [44]. Gradient-informed variants add local signals [45]. Uni-TMPO instead matches complete-trajectory probabilities directly to a target derived from terminal rewards. The forward KL is computed exactly over the two groupwise distributions.

For efficient T2I rollout collection, TreeGRPO combines tree sampling with a training-independent randomwindow schedule and group-relative reward maximization [8]. TMPO [46] introduced a Softmax Trajectory Balance method for T2I post-training. In contrast, Uni-TMPO introduces a unified RL post-training framework based on direct groupwise forward-KL optimization for stochastic diffusion and flow policies across T2I generation and VLA control. Both domains share the optimization objective while using domain-specific trajectory construction for T2I denoising trajectories and VLA action-chunk trajectories; see Sec. 5.

## 3 PRELIMINARIES

## 3.1 Stochastic Diffusion and Flow Trajectories

To define a common post-training interface, we represent both T2I and VLA outputs as context-conditioned stochastic trajectories. Let c denote a conditioning context and τ a trajectory generated by a stochastic diffusion or flow policy. In image generation, c is a text prompt. One trajectory records the complete denoising rollout from Gaussian image noise to one image. In VLA, the policy is queried again whenever a new observation is available. Each query maps Gaussian action noise to an action chunk, and a prefix is executed before the next query. The environment influences later observations and rewards. Only stochastic policy decisions contribute to the differentiable trajectory score. Although their raw inputs and outputs differ, both domains provide a conditioning context, a complete trajectory, an outcome reward, and the log probabilities of stochastic policy transitions.

Group-based post-training samples K trajectories for the same context and compares their rewards [3], [39]. For embodied rollouts, the shared context contains the language instruction and an initialization identifier. The trajectories therefore begin from the same task condition. This identifier defines the comparison group and is not an input to the action policy. Each sampler therefore returns the group representation $\{ ( \tau _ { i } , R _ { i } , \tilde { s _ { \theta , i } } ) \} _ { i = 1 } ^ { K } ,$ where $R _ { i }$ is the trajectory reward and $s _ { \theta , \ast }$ <sub>i</sub> is computed from its stochastic transition log probabilities. This shared representation enables the same groupwise update in both domains; only the construction of $\tau _ { i }$ and $s _ { \boldsymbol { \theta } , \cdot }$ <sub>i</sub> differs between T2I and VLA.

## 3.2 Reward Maximization with Multiple Valid Trajectories

Under this common trajectory representation, conventional post-training maximizes expected scalar reward. When several trajectories obtain the same maximal reward, this objective does not distinguish between assigning all probability to one trajectory and distributing probability across them. A single-mode policy can therefore remain optimal even when multiple successful trajectories are available. The formal statement and proof are given in the supplement.

This ambiguity matters when distinct high-reward trajectories correspond to different image modes or physical strategies. Relative advantages improve credit assignment but do not define a target distribution over useful trajectories. A unified framework must therefore support both trajectory structures while specifying which trajectories are preferred and how probability is distributed among them.

## 3.3 Boltzmann Reward Distributions

To define this target distribution from scalar rewards, we use Boltzmann weighting,

$$
q _ { R } ( \tau \mid c ) \propto \exp \bigl ( \beta R ( \tau , c ) \bigr ) ,\tag{1}
$$

where the inverse temperature $\beta > 0$ controls how strongly the distribution favors higher rewards. Small or moderate values retain probability on several useful trajectories. Large values assign most probability to the highest-reward trajectories. Exponential reward weighting also underlies relative entropy and maximum a posteriori policy search [36], [37]. We apply this principle within each sampled group. Section 4 constructs a standardized Boltzmann target over trajectories that share the same context.

## 4 UNIFIED TRAJECTORY-MATCHING FRAMEWORK

Using the groupwise Boltzmann distribution over complete trajectories, Uni-TMPO provides a unified RL post-training framework for stochastic diffusion and flow policies. Within each group, rewards define the target distribution $q ,$ while trajectory scores define the policy distribution $p _ { \theta } .$ . Uni-TMPO matches the two distributions through a groupwise forward-KL objective. T2I and VLA use domain-specific trajectory sampling and score computation, as detailed in Sec. 5.

## 4.1 Trajectory Groups

For each context c, the policy samples K trajectories,

$$
\mathcal { G } ( c ) = \{ \tau _ { i } \} _ { i = 1 } ^ { K } .\tag{2}
$$

In T2I, c is a prompt and each $\tau _ { i }$ is a complete image denoising trajectory. In $\mathrm { V L A } , c = ( \ell , \rho )$ contains a language instruction ℓ and an initialization identifier $\rho .$ Each $\tau _ { i }$ stores the physical rollout and the stochastic action flow decisions that generated its action chunks. Fixing c ensures that all trajectories address the same conditional problem.

Every trajectory provides a terminal or episodic reward $R _ { i }$ and a differentiable trajectory score $s _ { \theta , i } ~ = ~ s _ { \theta } ( \tau _ { i } ~ \mid ~ c )$ The reward evaluates the image or physical rollout. For continuous diffusion and flow transitions, the score is constructed from recorded transition log densities. Environment transitions are excluded, while intermediate stochastic transitions are optimized through the trajectory-level loss.

## 4.2 Reward-Induced Target Distribution

We standardize rewards within each group to remove offsets and reduce sensitivity to evaluator scales, obtaining ${ \widetilde { R } } _ { i }$ . We then construct the groupwise Boltzmann target distribution:

$$
\boxed { q _ { i } = \frac { \exp ( \beta \widetilde { R } _ { i } ) } { \sum _ { j = 1 } ^ { K } \exp ( \beta \widetilde { R } _ { j } ) } } , \qquad q \in \Delta ^ { K - 1 } .\tag{3}
$$

Here, $\Delta ^ { K - 1 } \ : = \ : \lbrace x \in \mathbb { R } _ { + } ^ { K } \ : : \ : \sum _ { i = 1 } ^ { K } x _ { i } \ : = \ : 1 \rbrace$ denotes the probability simplex over the K trajectories. Rewards are evaluator scores, not probabilities. Eq. (3) converts their ordering and relative gaps into a target distribution over the group. Higher-reward trajectories receive larger target probabilities, while trajectories with similar rewards receive similar probabilities. The inverse temperature $\beta$ controls the strength of this preference. Large values approach winnertake-all selection over trajectories.

This construction applies to binary task success, continuous image preference, and episodic robot rewards.

Relation to FlowRL. As reviewed in Sec. 2, FlowRL learns a prompt-conditioned normalizer $Z _ { \phi }$ for Boltzmann trajectory matching [42]. Uni-TMPO normalizes the reward target and trajectory probabilities within the same group, so the shared normalizer cancels.

## 4.3 Groupwise Trajectory Probabilities

We normalize the trajectory scores within each group to obtain the corresponding trajectory probabilities,

$$
\boxed  p _ { \theta , i } = \frac { \exp ( s _ { \theta , i } ) } { \sum _ { j = 1 } ^ { K } \exp ( s _ { \theta , j } ) } \Bigg \} , \qquad p _ { \theta } \in \Delta ^ { K - 1 } .\tag{4}
$$

Here $s _ { \boldsymbol { \theta } , i }$ denotes the trajectory score for trajectory $i ,$ and $p _ { \theta , i }$ is its relative probability within $\mathcal G ( c )$ . Log-softmax provides a stable implementation. Terms shared by every trajectory cancel; terms shared by only a subset still affect their relative probabilities. See Secs. 5.1 and 5.2 for the T2I and VLA scores.

Within each group, all trajectories share the same context and score definition, so only their relative probabilities are needed. Comparing $p _ { \theta }$ with the target $q$ identifies highreward trajectories that receive too little policy probability.

## 4.4 Forward-KL Distribution Matching

A group with constant reward provides no relative preference. We therefore define

$$
M _ { \mathcal { G } } = \mathbb { I } \bigg [ \operatorname* { m a x } _ { i } R _ { i } - \operatorname* { m i n } _ { i } R _ { i } > \epsilon _ { R } \bigg ]\tag{5}
$$

and optimize only the valid groups ${ \mathcal { B } } _ { \mathrm { v a l i d } } = \{ { \mathcal { G } } \in B : M _ { \mathcal { G } } =$ 1}. The trajectory-matching loss is

$$
\boxed { \mathcal { L } _ { \mathrm { T M } } ( \theta ) = \frac { 1 } { | \mathcal { B } _ { \mathrm { v a l i d } } | } \sum _ { \mathcal { G } \in \mathcal { B } _ { \mathrm { v a l i d } } } \mathrm { K L } ( q \mathcal { G } \| p _ { \theta , \mathcal { G } } ) } .\tag{6}
$$

If a minibatch contains no informative group, it produces no update from rewards. Rewards are detached during an update. For one group,

$$
\mathrm { K L } ( q \| p _ { \theta } ) = - \sum _ { i = 1 } ^ { K } q _ { i } \log p _ { \theta , i } + C _ { q } ,\tag{7}
$$

where $C _ { q }$ is constant with respect to $\theta .$ Differentiating the group softmax gives

$$
\nabla _ { \theta } \mathcal { L } _ { \mathcal { G } } = \sum _ { i = 1 } ^ { K } ( p _ { \theta , i } - q _ { i } ) \nabla _ { \theta } s _ { \theta , i } .\tag{8}
$$

$\mathrm { I f } \ p _ { \theta , i } < q _ { i }$ , gradient descent increases the relative probability of trajectory $i ; { \mathrm { i f } } \ p _ { \theta , i } > q _ { i }$ , it decreases that probability. The coefficients are bounded, sum to zero, and vanish when the distributions agree. Thus, the update corrects each sampled trajectory toward its target probability instead of selecting the group winner or adding an undirected entropy bonus.

When the trajectory score decomposes as

$$
s _ { \theta , i } = \sum _ { u \in \mathcal { U } _ { i } } w _ { i , u } \ell _ { \theta , i , u } ,\tag{9}
$$

the exact transition-level credit is

$$
\frac { \partial \mathcal { L } _ { \mathcal { G } } } { \partial \ell _ { \theta , i , u } } = w _ { i , u } ( p _ { \theta , i } - q _ { i } ) .\tag{10}
$$

Thus, one correction for trajectory i is propagated through its recorded stochastic decisions. Matching redistributes probability among the sampled trajectories. A mode must first appear in the group before it can receive probability. The toy experiment in Fig. 2 illustrates this behavior.

## 4.5 On-Policy Update

Alg. 1 summarizes the shared Uni-TMPO update.

Algorithm 1 Uni-TMPO on-policy group update.   
Input: Policy $\pi _ { \theta } ,$ , contexts ${ \mathcal { C } } ,$ group size K, temperature $\beta ,$   
threshold $\epsilon _ { R }$   
$\mathcal { L }  0 , N  0$   
for $c \in { \mathcal { C } }$ do   
$\mathcal { G } _ { c } = \{ \tau _ { i } \} _ { i = 1 } ^ { K } \sim \pi _ { \theta } ( \cdot \vert c )$   
$R _ { i } \gets \mathrm { s g } [ R ( \tau _ { i } , c ) ]$ ▷ detached reward   
$s _ { \theta , i }  s _ { \theta } ( \tau _ { i } \mid c )$ ▷ trajectory score   
if max<sub>i</sub> $R _ { i } - \operatorname* { m i n } _ { i } R _ { i } > \epsilon _ { R }$ then   
$q _ { c } \gets \mathrm { s o f t m a x } ( \beta \bar { \mathbf { R } } _ { c } )$ ▷ target distribution   
$p _ { c } \gets \mathrm { s o f t m a x } ( \mathbf { s } _ { \theta , c } ) \qquad \triangleright$ groupwise probabilities   
$\mathcal { L }  \mathcal { L } + \mathrm { K L } ( q _ { c } | | p _ { c } ) , N  \bar { N } + \bar { 1 }$   
end if   
end for   
$\theta  \theta - \eta \nabla _ { \theta } ( \mathcal { L } / N ) \mathrm { i f } \ N > 0$   
return π<sub>θ</sub>

Rewards and environment transitions remain nondifferentiable. In VLA experiments, gradients pass through the trainable action expert while the vision-language backbone stays frozen. Training uses fresh on-policy groups. A reference-policy KL, when used, remains separate from ${ \mathcal { L } } _ { \mathrm { T M } } .$ The supplement provides proofs and implementation details. The next section instantiates trajectory sampling and score computation for T2I and VLA.

## 5 DOMAIN-SPECIFIC TRAJECTORY SAMPLING

The groupwise forward-KL objective applies to both T2I and VLA, but each domain organizes stochastic transitions differently. We therefore use progress-conditioned coarse-tofine T2I sampling for complete denoising trajectories and feedback-conditioned action-chunk VLA sampling for robot rollouts. Each procedure also provides the trajectory scores required by the shared objective. Algorithms 2 and 3 summarize the T2I and VLA sampling procedures, respectively.

```latex
Algorithm 2 Progress-conditioned coarse-to-fine T2I sam
pling (shared-prefix branching).
Input: Prompt $c ,$ policy $p _ { \theta } ,$ depth ${ \overline { { S , } } }$ scheduled levels $B _ { \eta } ,$
factor b
$x _ { S } \sim \mathcal { N } ( 0 , I ) , \mathcal { F } _ { S } \gets \{ x _ { S } \}$
for $k = S , \ldots , 1$ do
if $\boldsymbol { k } \in B _ { \eta }$ then
$\mathcal { F } _ { k - 1 } \dot { } \gets \mathrm { B r a n c h } _ { b } ( \mathcal { F } _ { k } ; p _ { \theta } , c )$
Record stochastic transitions with log densities in $\zeta .$
else
$\mathcal { F } _ { k - 1 }  \mathrm { S o l v e r } _ { k } ( \mathcal { F } _ { k } )$
end if
end for
$\begin{array} { r } { s _ { \theta , i } ^ { \mathrm { i m g } }  \sum _ { k \in \mathcal { K } _ { i } ^ { \mathrm { s t o } } } \log p _ { \theta } ( x _ { i , k - 1 } \mid x _ { i , k } , c ) } \end{array}$
return $\{ ( \tau _ { i } , s _ { \theta , i } ^ { \mathrm { i \dot { m } g } } ) \} _ { i = 1 } ^ { K }$
```

## 5.1 Progress-Conditioned Coarse-to-Fine T2I Sampling

For a text prompt c, one generated image corresponds to one complete denoising rollout

$$
\tau _ { i } ^ { \mathrm { i m g } } = ( x _ { i , S } , \ldots , x _ { i , 0 } ) , \qquad x _ { i , S } \sim \mathcal { N } ( 0 , I ) .\tag{11}
$$

The complete rollout contains several stochastic decisions indexed by $\boldsymbol { \kappa } _ { i } ^ { \mathrm { s t o } }$ . Given $x _ { i , S , }$ , the image trajectory score is the sum of their transition log densities:

$$
s _ { \theta , i } ^ { \mathrm { i m g } } = \sum _ { k \in \mathcal { K } _ { i } ^ { \mathrm { s t o } } } \log p _ { \theta } ( x _ { i , k - 1 } \mid x _ { i , k } , c ) .\tag{12}
$$

All leaves use the same number of stochastic transitions, so their scores need no length normalization. The density of their common initial state cancels from the group softmax.

We use shared prefixes to avoid repeating common denoising steps and apply a progress-conditioned coarseto-fine scheduler to select branch levels [46]. TreeGRPO instead samples one contiguous stochastic window from a training-independent truncated-geometric schedule [8]. Our bounded Beta scheduler places branches earlier during early training to explore composition and global structure. As training proceeds, it moves the branches later to refine appearance and local details. Each leaf records the transition log probabilities along its complete trajectory for the forward-KL objective in Eq. (6). Shared transitions cancel only when they occur in every trajectory score.

![](images/e971bff11e7760eaac46bc30279cbca81f723eee5ee34cb6885a5e53e4cdae5a.jpg)  
Fig. 2. Five-mode distribution matching with a three-layer MLP. The left panels show the five-peak reward landscape and shared initialization. The remaining panels show intermediate and final samples and final mode mass under Uni-TMPO (top) and Reward Maximization (bottom). Uni-TMPO matches the target and preserves all five modes, whereas Reward Maximization collapses to the highest-reward mode.

The scheduler changes which complete trajectories enter the group but leaves the matching loss unchanged. It can therefore improve exploration and efficiency without changing how complete trajectories are optimized.

```latex
Algorithm 3 Feedback-conditioned action-chunk VLA sam
pling (sequential replanning).
Input: Context $( \ell , \rho ) .$ , policy π<sub>θ</sub>, group size K, replan hori
zon $H ^ { \prime } ,$ , budget L
for $i = 1 , \ldots , K$ do
$\mathbf { o } _ { i , 0 }  \mathrm { I n i t } ( \rho ) , t  0$
while $\tau _ { i }$ active and $\begin{array} { r } { \sum _ { v < t } H _ { i , v } ^ { \prime } < L } \end{array}$ do
$( { \bf A } _ { i , t } ^ { \mathrm { p r e d } } , \zeta _ { i , t } ) \gets$ FlowSample $\mathbf { \Sigma } _ { \theta } ( \mathbf { o } _ { i , t } , \ell )$
$\mathbf { A } _ { i , t } ^ { \mathrm { e x e c } } \gets \mathbf { A } _ { i , t } ^ { \mathrm { p r e d } } [ 0 ; H _ { i , t } ^ { \prime } )$
$\mathbf { o } _ { i , t + 1 }  \mathrm { E n v S t e p } ( \mathbf { o } _ { i , t } , \mathbf { A } _ { i , t } ^ { \mathrm { e x e c } } )$
$t \gets t + 1$ ▷ execute, observe, and replan
end while
$T _ { i } \gets t , R _ { i } \gets \mathrm { R e w a r d } ( \tau _ { i } )$
$s _ { \theta , i } ^ { \mathrm { V L A } }  T _ { i } ^ { - 1 } \sum _ { t = 0 } ^ { T _ { i } - 1 } \ell _ { \theta , i , t } ^ { \mathrm { c h u n k } }$
end for
return $\{ ( \tau _ { i } , R _ { i } , s _ { \theta , i } ^ { \mathrm { V L A } } ) \} _ { i = 1 } ^ { K }$
```

## 5.2 Feedback-Conditioned Action-Chunk VLA Sampling

For VL $\mathbf { \nabla } \cdot \mathbf { A } , c = ( \ell , \rho )$ contains the instruction ℓ and a shared initialization identifier $\rho .$ The identifier defines the common initial condition and is not given to the policy.

We use feedback-conditioned action-chunk VLA sampling to construct each trajectory. At valid query t, the action expert runs one complete action-flow inference conditioned on the current observation $\mathbf { o } _ { i , t }$ and instruction ℓ. It predicts the chunk $\mathbf { A } _ { i , t } ^ { \mathrm { p r e d } } = [ \mathbf { a } _ { i , t , 0 } , \ldots , \mathbf { a } _ { i , t , H - 1 } ]$ . The policy executes the prefix $\mathbf { A } _ { i , t } ^ { \mathrm { e x e c } } = \mathbf { A } _ { i , t } ^ { \mathrm { p r e d } } [ 0 ; H _ { i , t } ^ { \prime } )$ . The resulting observation conditions the next action chunk.

Because every executed prefix changes the next observation, one action chunk cannot represent the whole rollout. The trajectory score therefore aggregates the action-flow transition log probabilities across policy queries.

Here H is the prediction horizon, $H ^ { \prime }$ the replanning horizon, and $T _ { i }$ the number of valid queries. Each executed prefix satisfies $1 \leq H _ { i . t } ^ { \prime } \leq H ^ { \prime } \leq H$

Each chunk maps Gaussian action noise $\mathbf { u } _ { i , t } ^ { ( 0 ) }$ to $\mathbf { u } _ { i , t } ^ { ( S ) }$ through stochastic flow transitions indexed by $\mathcal { K } _ { i , t } ^ { \mathrm { s t o } }$ . Given the initial noise, observation, and instruction, its score is

$$
\ell _ { \theta , i , t } ^ { \mathrm { c h u n k } } = \sum _ { k \in \mathcal { K } _ { i , t } ^ { \mathrm { s t o } } } \log p _ { \theta } \Big ( \mathbf { u } _ { i , t } ^ { ( k + 1 ) } \mid \mathbf { u } _ { i , t } ^ { ( k ) } , \mathbf { o } _ { i , t } , \ell \Big ) .\tag{13}
$$

Rewards are evaluated on executed actions. The trajectory score uses the stochastic transitions that generated the full chunks. Environment transitions affect future observations and rewards but have no policy likelihood term.

A VLA trajectory contains $T _ { i }$ chunk-generation queries. To compare trajectories with different query counts, we

TABLE 1  
Comparison of FLUX.1-dev T2I post-training across compositional image generation, visual text rendering, and human preference alignment. The task-specific training reward is shown in parentheses after each task. Best and second-best results are bolded and underlined.
<table><tr><td>Method</td><td>Time (s)↓</td><td>GenEval↑</td><td>OCR↑</td><td>PickScore↑</td><td>HPS↑</td><td>ImgRwd↑</td><td>LGMD↑</td><td>Cos.Div.↑</td></tr><tr><td colspan="9">Compositional Image Generation (GenEval)</td></tr><tr><td>FLUX.1-dev</td><td></td><td>0.647</td><td></td><td>22.301</td><td>0.301</td><td>1.099</td><td>-0.031</td><td>0.211</td></tr><tr><td>DAG-DB</td><td>187.5</td><td>0.889</td><td></td><td>21.998</td><td>0.291</td><td>1.071</td><td>0.097</td><td>0.237</td></tr><tr><td>DGFS-SubTB</td><td>178.6</td><td>0.917</td><td></td><td>22.210</td><td>0.298</td><td>1.107</td><td>0.113</td><td>0.241</td></tr><tr><td>Flow-GRPO</td><td>160.8</td><td>0.946</td><td></td><td>22.113</td><td>0.289</td><td>1.074</td><td>-0.089</td><td>0.198</td></tr><tr><td>TreeGRPO</td><td>126.2</td><td>0.936</td><td></td><td>21.524</td><td>0.281</td><td>1.083</td><td>-0.281</td><td>0.184</td></tr><tr><td>GARDO</td><td>165.1</td><td>0.926</td><td></td><td>22.261</td><td>0.292</td><td>1.087</td><td>0.009</td><td>0.235</td></tr><tr><td>Uni-TMPO (ours)</td><td>91.9</td><td>0.954</td><td></td><td>22.967</td><td>0.305</td><td>1.163</td><td>0.136</td><td>0.248</td></tr><tr><td colspan="9">Visual Text Rendering (OCR Accuracy)</td></tr><tr><td>FLUX.1-dev</td><td></td><td></td><td>0.591</td><td>21.968</td><td>0.292</td><td>1.121</td><td>-0.040</td><td>0.215</td></tr><tr><td>DAG-DB</td><td>135.2</td><td></td><td>0.894</td><td>21.895</td><td>0.285</td><td>1.097</td><td>0.089</td><td>0.221</td></tr><tr><td>DGFS-SubTB</td><td>131.5</td><td></td><td>0.922</td><td>22.189</td><td>0.287</td><td>1.109</td><td>0.108</td><td>0.229</td></tr><tr><td>Flow-GRPO</td><td>121.5</td><td></td><td>0.944</td><td>21.382</td><td>0.288</td><td>1.096</td><td>-0.089</td><td>0.211</td></tr><tr><td>TreeGRPO</td><td>93.7</td><td></td><td>0.924</td><td>21.116</td><td>0.287</td><td>1.106</td><td>-0.289</td><td>0.183</td></tr><tr><td>GARDO</td><td>123.6</td><td></td><td>0.934</td><td>22.104</td><td>0.288</td><td>1.089</td><td>0.061</td><td>0.231</td></tr><tr><td>Uni-TMPO (ours)</td><td>76.3</td><td></td><td>0.951</td><td>22.281</td><td>0.316</td><td>1.118</td><td>0.115</td><td>0.235</td></tr><tr><td colspan="9">Human Preference Alignment (PickScore)</td></tr><tr><td>FLUX.1-dev</td><td></td><td></td><td></td><td>22.604</td><td>0.310</td><td>1.119</td><td>-0.056</td><td>0.214</td></tr><tr><td>DAG-DB</td><td>129.0</td><td></td><td></td><td>23.691</td><td>0.338</td><td>1.504</td><td>0.139</td><td>0.228</td></tr><tr><td>DGFS-SubTB</td><td>124.2</td><td></td><td></td><td>23.895</td><td>0.351</td><td>1.559</td><td>0.155</td><td>0.237</td></tr><tr><td>Flow-GRPO</td><td>109.1</td><td></td><td></td><td>24.226</td><td>0.381</td><td>1.594</td><td>-0.104</td><td>0.212</td></tr><tr><td>TreeGRPO</td><td>79.4</td><td></td><td></td><td>23.674</td><td>0.372</td><td>1.576</td><td>-0.281</td><td>0.179</td></tr><tr><td>GARDO</td><td>112.5</td><td></td><td></td><td>23.976</td><td>0.347</td><td>1.566</td><td>-0.057</td><td>0.209</td></tr><tr><td>Uni-TMPO (ours)</td><td>68.3</td><td></td><td></td><td>24.301</td><td>0.380</td><td>1.605</td><td>0.199</td><td>0.258</td></tr></table>

![](images/1d88c34951e030518770c75e065e461971ccb6daba0e128ad8ee71ae18e49c27.jpg)

![](images/6d6ad2ba2fe73efc3b683a9e65816f7ef16903956f5397e2b82b8ef733fbc34a.jpg)

![](images/38804426e5bc8e1b6aa6dbc93df229b988432eb64ce9612f0f3a0fc03566d941.jpg)  
Fig. 3. T2I reward–diversity–efficiency trade-off. Metrics are normalized across the post-training methods within each task. Larger values indicate better results on every axis; the Time axis is reversed because shorter iteration time is better. Uni-TMPO provides the strongest overall trade-off across compositional generation, visual text rendering, and human preference alignment.

average their chunk scores:

$$
s _ { \theta , i } ^ { \mathrm { V L A } } = \frac { 1 } { T _ { i } } \sum _ { t = 0 } ^ { T _ { i } - 1 } \ell _ { \theta , i , t } ^ { \mathrm { c h u n k } } .\tag{14}
$$

This length-normalized trajectory score is the log of the geometric mean of the valid chunk densities. Group softmax converts it into the relative trajectory probabilities used by the matching objective.

Masks exclude padded policy queries, deterministic flow steps, and padded action entries. The full noisy chunk is still supplied to the action expert. Tensor definitions and gradient details are given in the supplement.

![](images/7e258db78b8a1f3ed3cfd7109dc1277b7f461a7620a5b53d2f2641b3f87280b1.jpg)  
Fig. 4. T2I diversity under matched prompts. Each panel compares three stochastic samples per method under the same prompt. Compared with Flow-GRPO, Uni-TMPO produces broader variation in object appearance, aesthetic style, and spatial layout. Table 1 reports quantitative results.

Collection begins from a shared initialization. All slots use the same instruction and initial simulator state but receive independent action noise. After execution begins, their observations and action chunks evolve independently. Each trajectory records the physical episode used for reward and the stochastic decisions used for policy scoring.

## 5.3 Shared Objective Across Domains

The T2I and VLA probability calculations produce the same two groupwise distributions required by Uni-TMPO. Forward KL aligns the policy distribution with the rewardinduced target in both domains. The following experiments evaluate this unified objective in both domains.

## 6 EXPERIMENTS

We evaluate the same trajectory-matching objective under both trajectory structures. The T2I study measures reward, diversity, and training efficiency. The VLA study evaluates ID performance and task and scene OOD generalization.

## 6.1 T2I Generation

## 6.1.1 Experimental Setup

We use FLUX.1-dev [47] with LoRA [48]. We consider three main single-reward protocols. GenEval measures compositional correctness [49]. OCR accuracy evaluates visual text rendering [50]. PickScore measures human preference [51]. GenEval provides a near-binary correctness score. OCR accuracy rewards exact symbol sequences, whereas PickScore assigns continuous preference scores to individual images. The main comparison is reported in Table 1. Additional joint-reward experiments combine HPS-v2.1, ImageReward, and PickScore at equal weight or jointly optimize OCR and PickScore [52], [53]. Their full configurations and results are reported in the supplemental material.

For each prompt, we sample a group of K image trajectories with the same depth. The main K=27 setting uses three stochastic branch levels with branch factor three. Every leaf contributes the same number of transition log densities to Eq. (12), so path length cannot change its relative group probability. Uni-TMPO uses the progress-conditioned coarse-to-fine scheduler in Sec. 5.1, while TreeGRPO retains its random-window scheduler.

## 6.1.2 Metrics and Baselines

Task reward alone cannot diagnose mode collapse. We therefore report two diversity measures on matched prompt groups. LGMD detects near duplicates in VAE latent space [54]. Cosine diversity measures feature-space diversity using DINOv2 embeddings [55]. The qualitative comparison shows changes in object appearance, aesthetic style, and spatial layout. Within each protocol, all methods share inputs, generation seeds, and inference settings.

The distribution-matching baselines are DAG-DB and DGFS-SubTB [43], [44]. DAG-DB applies one-step detailed balance with a state-dependent flow, while DGFS-SubTB matches diffusion subtrajectories. Both fit intermediate flows along the denoising chain. The reward-maximization baselines are Flow-GRPO [3] and TreeGRPO [8]; GARDO [7] additionally regularizes diversity. TreeGRPO retains its training-independent random-window schedule, while Uni-TMPO uses the progress-conditioned scheduler in Sec. 5.1. All comparisons use the same prompts, seeds, group size, decoding steps, and number of generated images.

## 6.1.3 Main Results

Flow-GRPO and TreeGRPO improve the optimized task metric but reduce LGMD and cosine diversity relative to FLUX.1- dev in all three protocols. Uni-TMPO instead improves the primary task metric and both diversity measures. It reaches the highest GenEval score of 0.954 and the highest OCR accuracy of 0.951. In preference optimization, it obtains the highest ImageReward and PickScore and the second-highest HPS. These results show that the optimized reward can improve without reducing measured visual diversity.

DAG-DB and DGFS-SubTB retain positive LGMD, which supports distribution matching as a way to preserve visual modes. Both methods fit intermediate flows, and DGFS also combines constraints over subtrajectories. In the threetransition FLUX setting, Uni-TMPO directly matches complete trajectories to a target derived from terminal rewards. It achieves higher primary metrics and 42–49% lower iteration time than DGFS in this setting.

TABLE 2  
In-distribution VLA performance of $\pi _ { 0 }$ and $\pi _ { 0 . 5 }$ . Baseline configurations follow $\pi _ { \mathrm { R L } } [ 4 ] .$ . LIBERO and MetaWorld-MT50 report average task success, while CALVIN-D reports five-subtask sequence completion (Len-5). All values are percentages. Bold values indicate the best result for each backbone.
<table><tr><td rowspan="2">Backbone Method</td><td rowspan="2"></td><td colspan="5">LIBERO</td><td>MetaWorld-MT50</td><td>CALVIN-D</td></tr><tr><td>Spatial</td><td>Object</td><td>Goal</td><td>Long</td><td>Avg.</td><td>Avg.</td><td>Len-5</td></tr><tr><td rowspan="4">π0</td><td>SFT</td><td>65.3</td><td>64.4</td><td>49.8</td><td>51.2</td><td>57.6</td><td>50.8</td><td>57.5</td></tr><tr><td>Flow-SDE</td><td>98.4</td><td>99.4</td><td>96.2</td><td>90.2</td><td>96.1</td><td>78.1</td><td>61.7</td></tr><tr><td>Flow-Noise</td><td>99.0</td><td>99.2</td><td>98.2</td><td>93.8</td><td>97.6</td><td>85.8</td><td>59.9</td></tr><tr><td>Uni-TMPO (ours)</td><td>99.2</td><td>99.6</td><td>98.8</td><td>93.6</td><td>97.8</td><td>88.6</td><td>63.9</td></tr><tr><td rowspan="4">π0.5</td><td>SFT</td><td>84.6</td><td>95.4</td><td>84.6</td><td>43.9</td><td>77.1</td><td>43.8</td><td>61.3</td></tr><tr><td>Flow-SDE</td><td>99.6</td><td>100.0</td><td>98.8</td><td>93.0</td><td>97.9</td><td>70.7</td><td>87.0</td></tr><tr><td>Flow-Noise</td><td>99.6</td><td>100.0</td><td>99.6</td><td>94.0</td><td>98.3</td><td>66.1</td><td>84.5</td></tr><tr><td>Uni-TMPO (ours)</td><td>99.6</td><td>100.0</td><td>100.0</td><td>94.8</td><td>98.6</td><td>72.8</td><td>89.2</td></tr></table>

(a) Meta-World ML45  
![](images/d531eb2ad05cec7e2bc24e1bca8b0a6697ae618acf3b3cfd6bcaec2f6da8739b.jpg)

(b) CALVIN ABC→D  
![](images/10363d4533b5b7c39aac5d03ac4a2e6e065d6e3801cb9f6a223a8f7560607a68.jpg)

![](images/84f931bcd3cf2bff9aa1b84e11f5953610b400be32230d7807b7133fa72b2e3c.jpg)  
SFT Flow-SDE/Flow-Noise Uni-TMPO

![](images/e1c07e4652b73759387b6fc80c17113ed19c4635e0a3df07c93c775b0c9095be.jpg)

![](images/bfe22533f635f771820f596ec5c1dace7f96cdcc0b92c611a7404fb18a40bd0a.jpg)

![](images/e5808546fcbfcd0706d6cbb0984870004244b6857a693266a9ba5953dcacf06c.jpg)  
SFT Flow-SDE/Flow-Noise Uni-TMPO  
Fig. 5. Task and scene OOD generalization after VLA post-training. MetaWorld ML45 uses 45 tasks for online RL and evaluates five unseen tasks. CALVIN uses Scenes ABC for supervised initialization and online RL, then evaluates five-subtask completion (Len-5) in Scene D over 1,000 sequences. The lower panels compare SFT and post-RL performance for π<sub>0</sub> and π<sub>0.5</sub>.

Across the three protocols, Uni-TMPO provides the best reward–diversity–efficiency trade-off among the compared methods. Fig. 3 summarizes the task metrics, diversity measures, and iteration time on a common within-task scale. Training dynamics are reported in the supplemental material.

The supplement evaluates the progress-conditioned coarse-to-fine scheduler against fixed positions and the TreeGRPO scheduler. The Uni-TMPO objective and all other settings remain fixed, so the comparison isolates the effect of the scheduler during T2I post-training.

The category-level GenEval analysis in the supplement shows that the improvement is not confined to one compositional skill. Uni-TMPO is best or tied for best in all six categories, covering object composition, counting, color, position, and attribute binding.

Figure 4 compares three forms of diversity under matched prompts. For appearance, Uni-TMPO varies the television shape, screen style, and surrounding scene. For aesthetics, it varies the color palette, lighting, and visual style of the glowing-tree scene. For spatial layout, it varies the sign shape, orientation, viewpoint, and position within the forest scene. Flow-GRPO produces more similar samples in each case. The variations remain consistent with the shared prompt. These examples complement the quantitative diversity metrics reported in Table 1 across protocols.

## 6.1.4 Multi-Reward Evaluation

We also jointly optimize OCR and PickScore as two distinct reward signals. With weights scheduled from 3:1 to 1:3, Uni-TMPO reaches 0.932 OCR accuracy and 23.897 PickScore, with LGMD 0.155 and cosine diversity 0.248. Flow-GRPO obtains 0.928, 23.081, −0.049, and 0.209, respectively. Fixed 1:1 weighting gives the same ordering. The supplement reports the full comparison and ablations of the coarse-to-fine scheduler, reference KL, branch structure, and β.

![](images/04aa52db6afd7ce45865b1d38777e1c085feef08f21d8ec0622fff19c7e48e03.jpg)

Fig. 6. WallDetour experiment. LEFT is shorter than RIGHT, although both reach the target under the same 0.9 × Success + 0.1 × PathEficiency reward. Reward Maximization concentrates successful rollouts on LEFT (0.95/0.05), whereas Uni-TMPO retains both routes (0.55/0.45), with normalized binary route entropies of 0.29 and 0.99. When LEFT is blocked at evaluation, only Uni-TMPO reaches the target via RIGHT.  
![](images/e65a3a43edd864ed7512a80a352d999c50e02a47780e0ae4c71599e48f864665.jpg)  
Fig. 7. Real-robot dual-target placement. The robot is instructed to place the red block on either yellow target. The left target receives slightly higher reward because it is closer. In the unblocked setting, both Reward Maximization rollouts select the left target, whereas Uni-TMPO succeeds at both targets. After the left target is blocked, Reward Maximization fails, while Uni-TMPO succeeds through the alternative right-target strategy.

## 6.2 VLA Manipulation

## 6.2.1 Training and Evaluation Setup

The VLA experiments use SFT-initialized OpenPI policies in RLinf [4], [27], [56]. We compare Uni-TMPO with the Flow-SDE and Flow-Noise reward-maximization variants of π<sub>RL</sub>. During RL, the vision-language backbone and language model remain frozen. Only the approximately 300M-parameter action expert is optimized. A rollout batch contains 64 environments, forming eight groups of size eight with a shared instruction and initial state. Eight rollout epochs collect 512 trajectories per training iteration. For fairness, all methods use identical environments, resets, interaction budgets, and horizons. The supplement provides training and evaluation details.

The VLA study evaluates ID performance, task and scene OOD generalization, and multiple action strategies in WallDetour and a real-robot blocked-target experiment.

## 6.2.2 In-Distribution Performance

LIBERO [57] contains Spatial, Object, Goal, and Long suites. Each suite has ten tasks and 50 fixed resets per task. MetaWorld-MT50 [33] evaluates 50 tasks with ten trials each and reports a macro average over four difficulty groups. CALVIN [34] contains four scenes, A–D. Its ID protocol uses supervised data from Scenes ABC, followed by online RL and evaluation in Scene D (ABC-SFT → D-RL → D-eval). The metric is completion of a five-subtask sequence. These settings measure ID improvement after RL.

Table 2 reports the ID comparison. With π<sub>0</sub>, Uni-TMPO reaches 97.8% on LIBERO, 88.6% on MT50, and 63.9% on

CALVIN-D. These results exceed the strongest π<sub>RL</sub> rewardmaximization baseline by 0.2, 2.8, and 2.2 points, respectively. With $\pi _ { 0 . 5 } ,$ , the corresponding results are 98.6%, 72.8%, and 89.2%. The gains are 0.3, 2.1, and 2.2 points. LIBERO leaves little headroom under this protocol. MT50 and CALVIN-D reveal larger differences through multitask manipulation and long-horizon completion. The next experiments evaluate whether these gains extend to OOD generalization.

## 6.2.3 Out-of-Distribution Generalization

The OOD study evaluates two forms of generalization. MetaWorld ML45 performs online RL on 45 tasks and reports macro success on the five official unseen tasks.

CALVIN measures scene transfer. Supervised initialization and online RL use Scenes ABC, followed by evaluation in Scene D (ABC-SFT → ABC-RL → D-eval). The official evaluator processes 1,000 five-subtask sequences. We report Len-5, the percentage completing all five subtasks.

Both backbone comparisons include SFT, Flow-SDE, Flow-Noise, and Uni-TMPO. Fig. 5 compares the SFT and post-RL endpoints for both backbones. Flow-SDE and Flow-Noise overlap because they obtain identical aggregate OOD scores; the connecting segments show endpoint changes rather than training curves over successive iterations.

On MetaWorld, Uni-TMPO reaches 69.9% with $\pi _ { 0 }$ and 59.3% with $\pi _ { 0 . 5 }$ . These results improve over the corresponding SFT checkpoints by 20.0 and 18.1 points and exceed the corresponding π<sub>RL</sub> reward-maximization baselines by 4.4 and 11.5 points. The consistent gains across both backbones demonstrate that Uni-TMPO generalizes beyond the 45 training tasks to unseen manipulation tasks.

On CALVIN ABC→D, Uni-TMPO obtains 61.9% and 82.1% Len-5 success. With $\pi _ { 0 } ,$ , the flow baselines fall 2.2 points below SFT; Uni-TMPO improves by 4.4 points and finishes 6.6 points higher. With π<sub>0.5</sub>, Uni-TMPO improves over SFT by 20.8 points and exceeds the flow baselines by 3.0 points. Together, MetaWorld and CALVIN show consistent gains under task and scene changes.

WallDetour and real-robot evaluation. We use WallDetour and a real-robot dual-target placement task to evaluate whether multiple action strategies remain effective after a higher-reward option is blocked.

WallDetour provides two successful routes to the same target. LEFT is shorter and receives slightly higher reward, while RIGHT provides a longer alternative. Reward Maximization and Uni-TMPO start from the same SFT policy trained on both routes and use the same reward without route labels or diversity bonuses. Reward Maximization concentrates successful rollouts on LEFT, with LEFT/RIGHT mass 0.95/0.05 and normalized route entropy 0.29. Uni-TMPO retains both routes, reaching 0.55/0.45 and entropy 0.99. After LEFT is blocked, Reward Maximization fails, whereas Uni-TMPO completes the task through RIGHT.

We further evaluate the same effect in real-robot dualtarget placement, as shown in Fig. 7. The robot is instructed to place the red block on either yellow target. Both placements complete the instruction, but the shorter left placement receives slightly higher reward. In the unblocked setting, the displayed Reward Maximization rollouts both select the left target, whereas Uni-TMPO succeeds at both targets. After the left target is blocked, Reward Maximization fails, while Uni-TMPO completes the task using the right-target strategy.

Together, these experiments show that multiple action strategies support task completion after the higher-reward option becomes unavailable.

## 7 CONCLUSION

Reward-maximizing post-training collapses stochastic diffusion and flow policies onto a narrow set of high-reward trajectories. Uni-TMPO instead matches reward-induced and policy-induced probabilities over trajectories sampled under the same context. For T2I, complete-trajectory matching and the progress-conditioned coarse-to-fine scheduler provide the best reward–diversity–efficiency trade-off among the compared methods. The comparisons with DAG and DGFS further support direct terminal-reward matching for the short stochastic trajectories used in FLUX. The same objective extends to VLA trajectories composed of multiple action chunks. Uni-TMPO achieves the best ID performance and task and scene OOD generalization. WallDetour and the realrobot blocked-target experiment further show that multiple action strategies support task completion when the higherreward option becomes unavailable. Together, these results demonstrate that groupwise trajectory matching provides a unified RL post-training framework for image and action generation. T2I and VLA use different trajectory sampling procedures but share the same groupwise forward-KL update. Overall, the framework preserves diverse solution modes and supports OOD generalization across stochastic diffusion and flow policies.

Future work may extend the RL post-training framework to video, audio, and 3D generation; molecular and protein design; and diffusion-based planning.

## REFERENCES

[1] K. Black, M. Janner, Y. Du, I. Kostrikov, and S. Levine, “Training diffusion models with reinforcement learning,” in International Conference on Learning Representations (ICLR), 2024.

[2] Y. Fan et al., “DPOK: Reinforcement learning for fine-tuning textto-image diffusion models,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 36, 2023, pp. 79 858–79 885.

[3] J. Liu et al., “Flow-GRPO: Training flow matching models via online RL,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 38, 2025, pp. 40 783–40 818.

[4] K. Chen et al., “π<sub>RL</sub>: Online RL fine-tuning for flow-based visionlanguage-action models,” arXiv preprint arXiv:2510.25889, 2025.

[5] H. Li et al., “SimpleVLA-RL: Scaling VLA training via reinforcement learning,” in The Fourteenth International Conference on Learning Representations (ICLR), 2026.

[6] Z. Miao et al., “Training diffusion models towards diverse image generation with reinforcement learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 10 844–10 853.

[7] H. He et al., “GARDO: Reinforcing diffusion models without reward hacking,” arXiv preprint arXiv:2512.24138, 2025.

[8] Z. Ding and W. Ye, “TreeGRPO: Tree-advantage GRPO for online RL post-training of diffusion models,” in The Fourteenth International Conference on Learning Representations (ICLR), 2026.

[9] X. Fu et al., “Dynamic-TreeRPO: Breaking the independent trajectory bottleneck with structured sampling,” arXiv preprint arXiv:2509.23352, 2025.

[10] J. Wang et al., “GRPO-Guard: Mitigating implicit over-optimization in flow matching via regulated clipping,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026, pp. 5988–5998.

[11] L. Zhang et al., “OP-GRPO: Efficient off-policy GRPO for flowmatching models,” arXiv preprint arXiv:2604.04142, 2026.

[12] K. Clark, P. Vicol, K. Swersky, and D. J. Fleet, “Directly finetuning diffusion models on differentiable rewards,” in International Conference on Learning Representations (ICLR), 2024.

[13] B. Wallace et al., “Diffusion model alignment using direct preference optimization,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 8228–8238.

[14] K. Yang et al., “Using human feedback to fine-tune diffusion models without any reward model,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 8941–8951.

[15] A. Pan, K. Bhatia, and J. Steinhardt, “The effects of reward misspecification: Mapping and mitigating misaligned models,” in International Conference on Learning Representations (ICLR), 2022.

[16] J. Skalse, N. H. R. Howe, D. Krasheninnikov, and D. Krueger, “Defining and characterizing reward gaming,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 35, 2022, pp. 9460– 9471.

[17] L. Gao, J. Schulman, and J. Hilton, “Scaling laws for reward model overoptimization,” in Proceedings of the 40th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 202, 2023, pp. 10 835–10 866.

[18] J. Wu et al., “RewardDance: Reward scaling in visual generation,” arXiv preprint arXiv:2509.08826, 2025.

[19] Y. Hong, K.-C. Kao, H. Zhou, and C.-J. Hsieh, “Understanding reward hacking in text-to-image reinforcement learning,” arXiv preprint arXiv:2601.03468, 2026.

[20] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn, “Learning fine-grained bimanual manipulation with low-cost hardware,” in Proceedings of Robotics: Science and Systems, Daegu, Republic of Korea, July 2023.

[21] C. Chi et al., “Diffusion policy: Visuomotor policy learning via action diffusion,” in Proceedings of Robotics: Science and Systems, Daegu, Republic of Korea, July 2023.

[22] A. Brohan et al., “RT-1: Robotics transformer for real-world control at scale,” in Proceedings of Robotics: Science and Systems, Daegu, Republic of Korea, July 2023.

[23] B. Zitkovich et al., “RT-2: Vision-language-action models transfer web knowledge to robotic control,” in Proceedings of the 7th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, vol. 229, 2023, pp. 2165–2183.

[24] Open X-Embodiment Collaboration et al., “Open X-Embodiment: Robotic learning datasets and RT-X models,” in 2024 IEEE International Conference on Robotics and Automation (ICRA), 2024, pp. 6892–6903.

[25] D. Ghosh et al., “Octo: An open-source generalist robot policy,” in Proceedings of Robotics: Science and Systems, Delft, Netherlands, July 2024.

[26] M. J. Kim et al., “OpenVLA: An open-source vision-language-action model,” in Proceedings of the 8th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, vol. 270, 2025, pp. 2679–2713.

[27] K. Black et al., “π<sub>0</sub>: A vision-language-action flow model for general robot control,” in Proceedings of Robotics: Science and Systems, Los Angeles, CA, USA, June 2025.

[28] Physical Intelligence et al., “π<sub>0.5</sub>: a vision-language-action model with open-world generalization,” arXiv preprint arXiv:2504.16054, 2025.

[29] Z. Zhang et al., “GRAPE: Generalizing robot policy via preference alignment,” arXiv preprint arXiv:2411.19309, 2024.

[30] Y. Chen, S. Tian, S. Liu, Y. Zhou, H. Li, and D. Zhao, “ConRFT: A reinforced fine-tuning method for VLA models via consistency policy,” in Proceedings of Robotics: Science and Systems, Los Angeles, CA, USA, June 2025.

[31] A. Longhini, D. Emukpere, J.-M. Renders, and S. Kim, “Behavioral mode discovery for fine-tuning multimodal generative policies,” arXiv preprint arXiv:2605.11387, 2026.

[32] J. Liu et al., “What can RL bring to VLA generalization? An empirical study,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 38, 2025, pp. 97 121–97 151.

[33] T. Yu et al., “Meta-World: A benchmark and evaluation for multi-task and meta reinforcement learning,” in Proceedings of the Conference on Robot Learning, ser. Proceedings of Machine Learning Research, vol. 100, 2020, pp. 1094–1100.

[34] O. Mees, L. Hermann, E. Rosete-Beas, and W. Burgard, “CALVIN: A benchmark for language-conditioned policy learning for long-

horizon robot manipulation tasks,” IEEE Robotics and Automation Letters, vol. 7, no. 3, pp. 7327–7334, 2022.

[35] W. Pumacay, I. Singh, J. Duan, R. Krishna, J. Thomason, and D. Fox, “The Colosseum: A benchmark for evaluating generalization for robotic manipulation,” in Proceedings ofRobotics: Science and Systems, Delft, Netherlands, July 2024.

[36] J. Peters, K. Mulling, and Y. Alt ¨ un, “Relative entropy policy search,”¨ in Proceedings of the Twenty-Fourth AAAI Conference on Artificial Intelligence, 2010, pp. 1607–1612.

[37] A. Abdolmaleki, J. T. Springenberg, Y. Tassa, R. Munos, N. Heess, and M. Riedmiller, “Maximum a posteriori policy optimisation,” in International Conference on Learning Representations (ICLR), 2018.

[38] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” arXiv preprint arXiv:1707.06347, 2017.

[39] Z. Shao et al., “DeepSeekMath: Pushing the limits of mathematical reasoning in open language models,” arXiv preprint arXiv:2402.03300, 2024.

[40] E. Bengio, M. Jain, M. Korablyov, D. Precup, and Y. Bengio, “Flow network based generative models for non-iterative diverse candidate generation,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 34, 2021, pp. 27 381–27 394.

[41] N. Malkin, M. Jain, E. Bengio, C. Sun, and Y. Bengio, “Trajectory balance: Improved credit assignment in GFlowNets,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 35, 2022, pp. 5955–5967.

[42] X. Zhu et al., “FlowRL: Matching reward distributions for LLM reasoning,” in International Conference on Learning Representations (ICLR), 2026.

[43] D. Zhang et al., “Improving GFlowNets for text-to-image diffusion alignment,” Transactions on Machine Learning Research, 2025.

[44] D. Zhang, R. T. Q. Chen, C.-H. Liu, A. Courville, and Y. Bengio, “Diffusion generative flow samplers: Improving learning signals through partial trajectory optimization,” in International Conference on Learning Representations (ICLR), 2024.

[45] Z. Liu, T. Z. Xiao, W. Liu, Y. Bengio, and D. Zhang, “Efficient diversity-preserving diffusion alignment via gradient-informed GFlowNets,” in International Conference on Learning Representations (ICLR), 2025.

[46] J. Li et al., “TMPO: Trajectory matching policy optimization for diverse and efficient diffusion alignment,” arXiv preprint arXiv:2605.10983, 2026.

[47] Black Forest Labs, “FLUX,” https://github.com/black-forest-labs/ flux, 2024.

[48] E. J. Hu et al., “LoRA: Low-rank adaptation of large language models,” in International Conference on Learning Representations (ICLR), 2022.

[49] D. Ghosh, H. Hajishirzi, and L. Schmidt, “GenEval: An objectfocused framework for evaluating text-to-image alignment,” in Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track, vol. 36, 2023, pp. 52 132–52 152.

[50] J. Chen, Y. Huang, T. Lv, L. Cui, Q. Chen, and F. Wei, “TextDiffuser: Diffusion models as text painters,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 36, 2023, pp. 9353–9387.

[51] Y. Kirstain, A. Polyak, U. Singer, S. Matiana, J. Penna, and O. Levy, “Pick-a-Pic: An open dataset of user preferences for text-to-image generation,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 36, 2023, pp. 36 652–36 663.

[52] X. Wu et al., “Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis,” arXiv preprint arXiv:2306.09341, 2023.

[53] J. Xu et al., “ImageReward: Learning and evaluating human preferences for text-to-image generation,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 36, 2023, pp. 15 903– 15 935.

[54] D. P. Kingma and M. Welling, “Auto-encoding variational bayes,” in International Conference on Learning Representations (ICLR), 2014.

[55] M. Oquab et al., “DINOv2: Learning robust visual features without supervision,” Transactions on Machine Learning Research, 2024.

[56] H. Zang et al., “RLinf-VLA: A unified and efficient framework for reinforcement learning of vision-language-action models,” arXiv preprint arXiv:2510.06710, 2025.

[57] B. Liu et al., “LIBERO: Benchmarking knowledge transfer for lifelong robot learning,” in Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track, vol. 36, 2023, pp. 44 776–44 791.
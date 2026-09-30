# Socialality Anchors: Towards Group-bounded Trajectory Prediction

Ziqian Zou, Conghao Wong, Qinmu Peng, Xinge You\*

Huazhong University of Science and Technology, Wuhan, 430074, Hubei, China.

\*Corresponding author(s). E-mail(s): youxg@mail.hust.edu.cn; Contributing authors: ziqianzoulive@icloud.com; conghaowong@icloud.com; pengqinmu@hust.edu.cn;

## Abstract

Trajectory prediction is a key component for understanding human behavior patterns in dynamic scenes. Researchers have devoted substantial eforts to modeling social interactions, especially group-wise interactions, since group membership often reflects shared intention, coordinated motion, and stable mutual adaptation, thus providing a persistent and semantically meaningful social prior for forecasting. However, existing group modeling methods may rely on a fixed threshold and infer groups mainly from agents’ relative positions within the observation window, overlooking the fact that grouping rules should be agentspecific, temporally coherent, and context-adaptive across diverse personalities, culturalities, and evolving interaction contexts. Inspired by human social perception that alternates between interpersonal distance in boundary-sensitive situations and relative speed consistency in dynamic interactions, we propose Socialality, a human-inspired trajectory prediction framework with interpretable socialality anchors and an extended grouping window for stable, context-aware grouping inference. Concretely, Socialality introduces a duo-scalar-controlled grouping kernel Socialality that jointly leverages historical observations and short-term future trajectory previews to learn agent-specific grouping rules, and employs a group-wise perception mechanism to model in-group and out-of-group interactions in an intuitive and explainable manner. Furthermore, we conduct extensive experiments on standard benchmarks to demonstrate the performance gains of Socialality, and provide qualitative analyses and statistical studies of anchor distributions to verify the interpretability and stability of the proposed socialality anchors. Code repo: https://github.com/LivepoolQ/Socialality

Keywords: Trajectory Prediction, Grouping, Socialality Anchor, Human-inspired

## 1 Introduction

Understanding and forecasting intelligent agents’ behaviors in dynamic scenes is a critical capability for human cognition and for many vision-based intelligent systems. Trajectory prediction, as a representative task, aims to forecast socially acceptable future trajectories for each target agent from a short history of observations, while accounting for the temporal regularities of motion and the potential interactions among surrounding agents [1]. Such forecasting ability supports a broad range of applications, including behavior analyses [2, 3], navigation and planning [4], transportation and autonomous driving [5], as well as detection and multi-target tracking [6, 7].

![](images/d369528cee10f2670b3aa6b565473a3ca033e118e17ac4165e6180676a54da42.jpg)  
Fig. 1 By observing the agent-wise preferences when grouping with others over a period of time, we mainly focus on how to infer each target agent’s social boundary, which will anchor its group afiliation.

Despite recent progress, trajectory prediction remains challenging in crowded scenes [8], where explicit grouping phenomenon could be widely observed. Empirical observations show that 55-70% of pedestrians walk in groups of two or more members in crowds[9]. In this manuscript, we define group afiliation for a target agent as which neighboring agent is considered to be a group member of this target agent <sup>1</sup>. When forecasting the target agent’s trajectory in crowded scenes, its group afiliation provides an important prior for how it interacts with surrounding agents [9, 11]. For example, group members tend to adjust their speeds and directions jointly when avoiding obstacles, rather than going separate ways. We refer to such interactions conditioned by the group afiliation as the group-bounded social interactions. Modeling group-bounded social interactions between pedestrians has long been a central theme in this field.

To explicitly determine a target agent’s group afiliation, a natural thought is to infer its social boundary [12], which defines the acceptable range of viewing neighboring agents that are regarded as group members and reflects the target agent’s interaction preference with others. However, it should take a period of time to infer such social boundary or interaction preference, such as acceptable interpersonal distance and preferred motion pattern. Moreover, proxemics studies suggest that social boundaries often vary with diferent agents from diverse cultural backgrounds and are contextdependent[13, 14]. Considering the dificulty of modeling such heterogeneous social boundaries of diverse agents, most existing trajectory prediction methods choose to skip the social boundary inferring and directly model social interactions among pedestrians before forecasting. However, this modeling shortcut neglects the priors provided by explicit group afiliations anchored by social boundaries, flattening the distinction between social interactions among in-group pedestrians and those among pedestrians from diferent groups. This suggests that trajectory prediction requires structural priors provided by explicitly modeling pedestrians’ agent-specific, context-adaptive, and temporally coherent social boundaries to anchor their specific group afiliations.

Specifically, agent-specific indicates that social boundaries should be explicitly established in an agent-wise manner, rather than being embedded in node-level or feature-level representations of each agent. Context-adaptive indicates the adaptability of even the same agent’s social boundary, which could be completely diferent under diverse interaction contexts or in-group relations. Temporally coherent indicates that such social boundaries around the current time step are not instantaneously determined from isolated observations, but inferred from continuous motion consistency over a period of time, including observation and anticipation.

Current researchers have attempted to model such social boundaries or groups in two main ways, representing each agent with diferent nodes in graph-based methods or setting certain grouping rule or train additional networks to determine group membership. On the one hand, some graph-based methods use a leveled graph or a directed graph [11, 15] to model diferent group structures while handling social interactions simultaneously among agents. However, in many graph-based formulations, edges mainly encode pair-wise interaction intensity or latent relational dependency, which is utilized as the evidence of whether two corresponding agents belonging to the same group. The instantaneous interaction strength or latent relational dependency should not be regarded equal to being in the same group, since simply avoiding fastmoving bypassing pedestrian obviously indicates considerable interactions strength. Moreover, though these methods could indeed represent certain group-like structures through graph-based message passing or latent relation learning, such structures could only be implicitly reflected in these representations, serving as a by-product of the modeling of social interactions rather than a dedicated inference target. As a result, these methods may succeed in describing who interacts with whom, but they do not explicitly explain what social boundaries anchor diferent groups.

On the other hand, some methods try to represent groups as explicit structures. Among these methods, some rely on manually annotated group ground truths to additionally train grouping networks [16, 17], which output group structures first before final trajectory prediction training or evaluation. Such annotated methods take lots of efort and largely rely on annotators’ own subjective judgments of whether two agents belong to the same group. Our conference paper, GPCC [18], proposes an end-to-end training paradigm without human annotation process, which determines group membership through a long-term distance kernel according to a manually set grouping rule. Such group membership results provide grouping priors for the subsequent perception and interaction modeling process. However, its fixed threshold implicitly assumes that a universal social boundary is shared by all agents. Moreover, GPCC’s grouping assignment is purely observation-based, relying only on historical motion cues while ignoring anticipation of short-term future tendencies. These assumptions overlook the agent-specific, context-adaptive, and temporally coherent nature of social boundaries in real crowded scenes.

To address the above requirements, as shown in Fig. 1, interpreting diferent agents specific and context-adaptive preferred social boundaries to explicitly anchor pedestrian groups with temporal coherence has become the main consideration of this manuscript. Considering trajectory prediction as a human-centric task, it could be much easier to establish social boundaries in a human-inspired way that simulates or emulates how humans perceive social context. In social psychology [19], perceived groupness is often characterized by entitativity cues, including proximity, similarity, and common fate, which connects the social boundary requirements mentioned above. Here, the agent-specific requirement aligns with proxemics studies suggesting that interpersonal distance is shaped by both personal preferences and cultural norms [14]. For example, people from diferent cultural backgrounds may maintain notably diferent distances in public walking scenarios, distinctively forming diferent privacy boundaries. The context-adaptive requirement also aligns with human intuition. For example, when walking with colleagues or friends, pedestrians may preserve a moderate interpersonal distance and move relatively eficiently towards a shared destination. When walking with intimate companions, they may naturally keep a closer distance and move at a more relaxed pace. These suggest that social boundaries are conditioned on the agent-level social contexts, i.e., social relations, roles, and coordination patterns. Finally, the time-coherent requirement further ensures the social boundaries are modeled in a human-inspired manner. For example, a pair of agents may appear close over the whole observation window but show a tendency of gradual separation near the current time, suggesting that their group relation may be weakening. Similarly, agents that gradually approach and align their motion patterns may be forming a group, adopting the other agent into its own boundary space. In other words, it means that such boundary preference requires a “temporal thickness” [20], i.e., it requires both observation and anticipation to finalize.

Similar to physical or territorial boundaries that are often anchored through observable indicators, as shown in Fig. 1, social boundaries can also be inferred from behavioral cues that reveal how pedestrians maintain and adjust their grouping relations with others. Motivated by the social psychology theories mentioned above, we introduce a set of learnable group-bounded anchors, referred to as socialality anchors, to simulate such grouping cues, which enables the model to learn and capture diferent agents’ various ways of anchoring their groups before making trajectory decisions. In relatively static or boundary-sensitive situations, interpersonal distance is often prioritized, because spatial proximity serves as an immediate cue of social boundary and afiliation in proxemics and group perception [13]. By contrast, in dynamic interaction scenes, humans tend to rely more on relative speed or motion consistency to infer whether multiple agents share a common fate or collective intention [9]. Such human-inspired observations motivate us to treat distance and speed as the primary behavioral factor for learning diferent anchors. These anchors could map various boundary preferences into quantified scalars, each of which is supposed to represent a single property for forming or keeping agents in a group, thus finally modulating and customizing their own unique social boundaries. In addition, we extend the original observation-only grouping window in the long-term distance kernel to contain both observation and anticipation temporal segments to coherently capture motion tendency with temporal thickness, thus forecasting trajectories with unique grouping preferences for diferent agents under various interaction contexts. In essence, the proposed group-bounded anchors transform social boundary modeling into a grouping cues interpretation problem. Instead of justifying whether two pedestrians are close enough to be grouped, the model learns which cues should be activated to explain the group afiliation of each agent under the current context.

This manuscript is an extension of our previous conference paper [18]. The GPCC Model in our conference paper introduces a group-based trajectory prediction paradigm, where social interactions are handled based on the inferred group membership using the original long-term distance kernel. However, with a manually fixed threshold in the long-term distance kernel, GPCC’s grouping rule cannot adapt to diferent forecasting scenes or even agent-wise social interactions. In addition, it lacks an explicit mechanism to either represent or explain how grouping decisions remain stable when pedestrians exhibit short-term motion fluctuations, such as temporary spatial proximity or abrupt changes in motion state. To address these issues, we propose Socialality, a human-inspired extension that introduces interpretable socialality anchors and leverages an extended grouping window to interpret agent-specific and context-adaptive social boundaries with temporal coherence to anchor heterogeneous groups.

In summary, our contributions are listed as follows:

1. We propose a socialality anchors-controlled grouping kernel that simultaneously considers both agents’ historical observations and their short-term future trajectory previews to learn to interpret agent-specific, context-adaptive social boundaries with temporal coherence, thus obtaining the group priors.

2. We propose the Socialality trajectory prediction model that inherits the perception mechanism to explicitly takes the group priors interpreted from Socialality kernel into account to simulate group-wise social interactions modulated by the newly proposed socialality anchors in a human-inspired way.

3. We conduct experiments on standard benchmarks to demonstrate the performance gain of Socialality, and further provide qualitative analyses together with statistical studies of anchor distributions to verify the interpretability and stability of the proposed socialality anchors.

## 2 Related Works

## 2.1 Trajectory Prediction and Social Interactions

Trajectory prediction aims at predicting agents’ future movements based on their historical states and potential interactions [1]. Early trajectory forecasting and crowd navigation studies largely relied on hand-crafted rules, where social efects were explic itly encoded as physics-inspired constraints between agents. The Social Force model [21] formulates pedestrian motion as the superposition of attractive and repulsive forces, yielding emergent collision avoidance and flow patterns. In parallel, velocityobstacle formulations [22, 23] cast interaction as geometric feasibility in velocity space, enforcing collision avoidance through reciprocal kinematic constraints. With the devel opment of deep learning methods, researchers gradually treat pedestrian motion as a temporal sequence generation task. These works predict agents’ future positions via recurrent architectures combined with hand-crafted rules. Trajectory prediction models based on RNNs [24, 25] and LSTMs [1, 26, 27] were subsequently introduced to model temporal dependencies in trajectories while implicitly accounting for interagent dependencies. Among them, Social-LSTM [1] first introduced the social pooling mechanism, which aggregates neighboring hidden states to capture local social context around the target agent. This paradigm has since been widely adopted and extended in many follow-up works [28–30], serving as a foundation for learning interaction-aware representations.

More recently, graph-based and Transformer-based architectures have become mainstream for modeling interactions, enabling more flexible message passing and context aggregation across agents and scenes [31–37]. Recent studies have also moved towards more interpretable, human-inspired formulations of social interaction. Drawing inspiration from biological echolocation, Wong et al.[38, 39] propose an angle-based representation that characterizes surrounding agents through relative distance, velocity, and direction, yielding an explicit and structured interaction description. Bae et al.[40, 41] leverage the knowledge embedded in large language models by introducing specially designed numeric tokens, enabling the model to reason about spatial relations and expose interaction cues from an alternative, language-driven perspective. Moreover, motivated by resonance phenomena, Resonance [42] interprets social interaction as spectrum-level co-vibration among agents, providing a decomposition that helps explain the stochasticity arising in multi-agent dynamics.

## 2.2 Group Modeling Methods in Trajectory Prediction

Recent trajectory prediction studies increasingly incorporate group modeling to capture higher-level social relations beyond pairwise interactions. Most of these groupmodeling methods are graph-based. Agents are represented as nodes, and group relations are inferred via message passing on interaction graphs, where edges encode spatial proximity or other latent relational dependencies. Graph neural networks (GNNs) and their variants have been widely adopted to aggregate neighbor information and to form group-aware representations for forecasting [11]. To better reflect collective structures, some methods employ multi-scale graphs or hypergraph-style for mulations [15], enabling relation reasoning at both individual and group levels, and thus modeling collective motion patterns in a unified architecture.

Despite their efectiveness, existing graph-based group modeling methods still face several limitations when viewed from the perspective of social boundary establishment. On the one hand, their grouping inference is typically implicit and tightly coupled with representation learning, where group membership and grouping rules are often hidden in node features, edge weights, or message-passing processes rather than being explicitly exposed as agent-wise social boundaries. On the other hand, since group relations are inferred from local observations or instantaneous interaction graphs, the resulting group structures may be unstable over time and sensitive to short-term motion fluctuations. These limitations motivate alternative designs that explicitly establish lightweight, interpretable, and temporally coherent social boundaries, thereby providing more reliable group priors for subsequent interaction modeling.

## 2.3 Human-inspired Methods in Trajectory Prediction

Pedestrian trajectory prediction is a human-centric task [43], as it aims to forecast human motion in shared environments. Accordingly, beyond purely data-driven forecasting, a line of works explicitly draws inspiration from human perception and cognition to build more interpretable and human-like prediction mechanisms. Some approaches mimic how humans selectively attend to relevant neighbors and regions to filter and weight social cues, thereby making interaction reasoning more structured and interpretable [38, 44].

Moreover, some human-inspired trajectory prediction methods are motivated by how humans use their sensors to perceive the world. Consequently, instead of treating all surrounding agents as equally observable, a number of methods incorporate view-based mechanisms to approximate how humans perceive social cues. For example, Hasan et al.[45] leverage the visual frustum of attention to emphasize that head orientation can serve as an explicit prior for deciding which neighbors are likely to influence the future motion. Similarly, Liao et al.[46] introduce an adaptive visual sector that dynamically allocates attention over angular regions according to motion states, enabling the model to filter and weight social information in a more human-like way. In this human-inspired spirit, the perception mechanism in GPCC also adopts an egocentric, observation-driven design to modulate which social cues are emphasized when forming interaction representations, aligning grouping and interaction reasoning with the target agent’s local perceptual context. Moreover, pedestrian motion is largely shaped by intention and goal-directed planning [47]. This motivates goal-driven forecasting, which explicitly models destinations as structured variables and conditions when forecasting trajectories [48–51].

In summary, existing group-modeling methods have shown the importance of groupness in trajectory prediction. However, most of them infer group relations implicitly through graph construction or message passing, where group membership is usually hidden in learned representations rather than explicitly established as agent-wise social boundaries. Therefore, they may provide less transparent characterization of agent-specific and context-dependent grouping preferences, especially when distinguishing in-group companions from nearby but out-group pedestrians. Meanwhile, human-inspired studies have introduced selective perception and goal-directed planning into trajectory prediction, but how to explicitly establish temporally coherent social boundaries from human-centric grouping cues remains to be addressed. Built upon the perception mechanism in GPCC Model, our enhanced Socialality Model takes a step further by introducing learnable socialality anchors to capture diverse grouping preferences and establish explicit group-bounded social boundaries. This design yields lightweight, interpretable, and stabilizable group priors, leading to more efective group-aware interaction reasoning and improved trajectory forecasting.

![](images/32473ec9ad84cdd945af95c67fefa7a6393e4c2c70c7b2d83b0460a5b85e7087.jpg)  
Fig. 2 GPCC method illustration. GPCC uses the long-term distance kernel and the perception mechanism to model group-wise social interactions.

## 3 Method

In our conference paper [18], the proposed GPCC Model (short for GPCC) aims at forecasting trajectories conditioned by social interactions with structured group priors. In this manuscript, we propose the enhanced Socialality Model (short for Socialal-$i t y )$ to extend the original frozen groupings in GPCC. It jointly integrates retentions (historical reviews) and protentions (short-term future previews) corresponding to each target agent through two trainable and complementary social anchors, therefore better understanding and forecasting agent-specific, temporally coherent, and context-adaptive grouping rules. As illustrated in Fig. 2 and Fig. 3, both GPCC and Socialality Models share almost the same computational pipelines, including a grouping kernel and a perception mechanism, achieving the goal of learning to group neighboring agents, as well as sensing group-wise diferences in interaction preferences, or even human motions. This section first contrasts two grouping kernels, the long-term distance kernel and the enhanced Socialality kernel, then introduces the perception mechanism and the enhanced Socialality trajectory prediction model that builds upon these kernels.

## 3.1 Problem Formulation

We denote the 2D coordinate of any agent $i ( i \in \{ 1 , 2 , . . . , N _ { e } \} )$ at time step t as $\mathbf { p } _ { i } ^ { t } = ( x _ { i } ^ { t } , y _ { i } ^ { t } ) \in \mathbb { R } ^ { 2 }$ (the current observation step is set to $t = 0$ in this manuscript). Our considered trajectory is uniformly sampled at a fixed temporal interval, and we use $t _ { o }$ discrete observing time steps to forecast trajectories in the following $t _ { p }$ steps. For a clear presentation, we denote any trajectory (agent i) with $t _ { 2 }$ steps and ends at $t = t _ { 1 }$ using the symbol $\mathbf { x } _ { i } ( t _ { 2 } , t _ { 1 } ) = \left( \mathbf { p } _ { i } ^ { t _ { 1 } - t _ { 2 } + 1 } , \ldots , \mathbf { p } _ { i } ^ { t _ { 1 } - 1 } , \mathbf { p } _ { i } ^ { t _ { 1 } } \right) ^ { \top } \in \mathbb { R } ^ { t _ { 2 } \times 2 }$ . Correspondingly, the observed trajectory (ends at $t = 0$ , with $t _ { o }$ steps) is represented as ${ \bf X } _ { i } \left( t _ { o } , 0 \right) =$ $\mathbf { x } _ { i } ( t _ { o } , 0 ) \in \mathbb { R } ^ { t _ { o } \times 2 }$ . In this manuscript, we focus on designing a network $\mathcal { N }$ to forecast possible future trajectory(ies) $\hat { \mathbf { Y } } _ { i } \left( { \bar { t } } _ { p } , t _ { p } \right) = \hat { \mathbf { x } } _ { i } ( t _ { p } , t _ { p } ) \in \mathbb { R } ^ { t _ { p } \times { 2 } }$ (starts at $t = 1$ and ends at $t = t _ { p } ,$ , with $t _ { p }$ steps) for any agent i, using all $N _ { e }$ agents’ observed trajectories $\mathcal { X } =$ $\{ \mathbf { X } _ { i } \left( t _ { o } , 0 \right) | i = 1 , 2 , \ldots , N _ { e } \} , \ i . e . , \ \hat { \mathbf { Y } } _ { i } \left( t _ { p } , t _ { p } \right) = N \left( \mathcal { X } , i \right)$ . Summaries of other symbols and network structures are listed in Table. 1.

![](images/e7c817ee1fbe43505cb11413b33b4a84e943be3d7d003bf98e5129f2d15e18fe.jpg)  
Fig. 3 Socialality method illustration. We use a forecasting scene with three walking agents to illustrate the overall modeling. A social group of agent 0 and agent 1 can be observed in this case.

Table 1 Symbols and network structures of GPCC Model and the enhanced Socialality Model
<table><tr><td>Sym.</td><td>Descriptions</td><td>Shapes</td></tr><tr><td>€</td><td>An infinitesimal quantity</td><td></td></tr><tr><td> $\mathbf { d } _ { j }$ </td><td>Neighboring agent j&#x27;s heading direction</td><td>(2)</td></tr><tr><td> $K _ { f }$ </td><td>Multi-style trajectory generation number in Socialality Model</td><td></td></tr><tr><td> $\bar { K _ { g } }$ </td><td>Trajectory generation number in short-term prediction network  $( K _ { g } < K _ { f } )$ </td><td></td></tr><tr><td> $\hat { \mathbf { Y } } _ { i } ( t _ { p } , t _ { p } )$ </td><td>Agent i&#x27;s multi-styled predicted trajectories</td><td> $( K _ { f } , t _ { p } , 2 )$ </td></tr><tr><td> $\mathbf { Y } _ { i } ( t _ { p } , t _ { p } )$ </td><td>Agent  $i \ ' \mathrm { s }$  ground-truth future trajectory</td><td> $( t _ { p } , 2 )$ </td></tr><tr><td> $\hat { \mathbf { Y } } _ { i } ^ { l }$ </td><td>Agent i&#x27;s linearly fitted trajectory in prediction period</td><td> $( t _ { p } , 2 )$ </td></tr><tr><td> $d$ </td><td>Feature dimensions in most networks (d = 32)</td><td></td></tr><tr><td> $l ( \cdot )$ </td><td> $ \mathrm { f c ( 2 , R e L U ) }  \mathrm { f c } ( d , \mathrm { t a n h } )$ </td><td></td></tr><tr><td> $\dot { m ( \cdot ) }$ </td><td> $ \mathrm { f c \bigl ( } d , \mathrm { R e L U } \bigr )  \mathrm { f c \bigl ( } 2 d , \mathrm { R e L U } \bigr ) \ast 2  \mathrm { f c \bigl ( } 2 d \ast t _ { o } , \mathrm { R e L U } \bigr )  \mathrm { f c ( 2 , t a n h ) }$ </td><td></td></tr><tr><td> $e ( \cdot )$ </td><td> $ \mathrm { f c ( 2 , R e L U ) }  \mathrm { f c } ( d , \mathrm { t a n h } )$ </td><td></td></tr><tr><td>h(.)</td><td> $ \mathrm { f c ( 3 , R e L U ) }  \mathrm { f c ( } d , \mathrm { R e L U ) }  \mathrm { f c } ( 2 d , \mathrm { R e L U } )  \mathrm { f c } ( d , \mathrm { t a n h } )$ </td><td></td></tr><tr><td>n(·)</td><td> $ \mathrm { f c ( 5 d , R e L U )  f c ( 4 d , t a n h ) }$ </td><td></td></tr></table>

## 3.2 Grouping Kernels

For a specific agent i and all its neighboring agents $\mathcal { N } _ { i } \ ( \mathcal { N } _ { i } \subseteq \{ 1 , 2 , . . . , N _ { e } \} )$ $i . e . ,$ group candidates, grouping kernels justify whether each agent $j \in \mathcal { N } _ { i }$ belongs to the same group as the target agent. These kernels are the foundations of both the original GPCC and the proposed Socialality Models, providing direct grouping priors for understanding and forecasting social interactions when predicting trajectories, especially those latent group behaviors. In detail, the original GPCC adopts a static grouping kernel, named long-term distance kernel. However, real-world grouping relations may vary with diferent agents, keep continuous between adjacent periods, and significantly difer over time. The Socialality kernel is then proposed to address these limitations. We first introduce the original long-term distance kernel, then describe how the new Socialality kernel will be structured.

## 3.2.1 The Long-Term Distance Kernel

Considering all time steps during an observation window Ω, the original long-term distance kernel is a manual-rule controlled kernel to justify any target agent’s group members. It uses a threshold Γ to compare the distance summation during the observation window Ω between the target agent i and any agent j of its neighboring agents $\in \mathcal { N } _ { i }$ . Here, we assume that agent j will be considered to be a group member of agent i only if they are consistently close enough during the whole observation window Ω. This distance summation is calculated through the long-term distance function $D _ { i } ( j | \Omega )$ , represented as:

$$
D _ { i } ( j | \boldsymbol { \Omega } ) = \sum _ { t \in \boldsymbol { \Omega } } \left\| \mathbf { p } _ { j } ^ { t } - \mathbf { p } _ { i } ^ { t } \right\| _ { 2 } .\tag{1}
$$

Every time step is taken into account through the distance summation. Then the manually controlled threshold $\Gamma \geq 0$ will be statically applied on all agents and time steps to evaluate whether their distance summation exceeds the threshold Γ. The long-term distance grouping kernel is then formulated as

$$
\begin{array} { r } { \mathcal { K } _ { i } ( j | \Omega , \Gamma ) = \mathbb { I } \left[ D _ { i } ( j | \Omega ) \leq \Gamma \right] . } \end{array}\tag{2}
$$

Here, I[·] denotes an indicator. It equals to 1 if the condition is satisfied (where the long-term distance is smaller than the given threshold Γ), and to 0 otherwise. The original long-term distance kernel regards spatial distance as explicit manifestation of social relations, $i . e .$ , pedestrians who remain spatially close over time are assumed to have certain social afiliation. It helps subsequent interaction-modeling networks to evaluate the grouping efectiveness during whole observation, rather than few outlier moments or periods caused by certain social interactions such as temporary collision avoidances. Compared to former works that utilize graph-based methods [11, 15] to model group relations, the long-term distance kernel not only straightforwardly conducts a lightweight computation (the distance summation), but also provides an eficient grouping rule for the subsequent grouping stage.

Specifically, in the original GPCC Model, the grouping stage only concerns the observation window $\Omega _ { 0 } = \{ - t _ { o } + 1 , \ldots , - 1 , 0 \}$ for grouping agents, where $t ~ = ~ 0$ represents current time step. For our considered target agent i and the observation period $\Omega _ { 0 }$ , i’s group $\mathcal { G } _ { i }$ , i.e., the set of its in-group agents selected from all its candidate neighbors ${ \mathcal { N } } _ { i }$ , is computed as

$$
\mathcal { G } _ { i } = \left\{ j \in \mathcal { N } _ { i } \left| \mathcal { K } _ { i } ( j | \Omega _ { 0 } , \Gamma ) = 1 \right. \right\} .\tag{3}
$$

Correspondingly, we have its out-of-group agents’ set $\overline { { \mathcal { G } } } _ { i }$ :

$$
\overline { { \mathcal { G } } } _ { i } = \left\{ j \in \mathcal { N } _ { i } \left| \mathcal { K } _ { i } ( j | \Omega _ { 0 } , \Gamma ) = 0 \right. \right\} .\tag{4}
$$

Naturally, we have $\mathcal G _ { i } \cup \overline { { \mathcal G } } _ { i } = \mathcal N _ { i }$

## 3.2.2 The Socialality Kernel

The long-term distance kernel performs grouping using a manually-set fixed-distance threshold Γ. Notably, this static threshold Γ is used on all agents when grouping, meaning that the same grouping rule will be applied across agents and scenes. However, grouping rules are intuitively not determined by one single variable. On the one hand, the acceptable social distance between the same group’s members often varies with time, and the group-wise acceptable social distance may also be obviously diferentiated for diferent groups. On the other hand, a single threshold implicitly assumes that all group decisions can be governed by the same rule, thus overlooking a wide range of grouping cases. For example, for diferent target agents belonging to distinct groups, their group-averaged speeds may also be substantially adjusted for greater in-group interaction flexibility [21].

Corresponding to the diverse personalities and culturalities of human beings, we propose the Socialality kernel to learn diferent target agents’ agent-specific, temporally coherent, and context-adaptive grouping rules, thus emulating their socialalities for sensing and forming their own groups. Here, Agent-specific indicates that grouping rules will change for diferent target agents. Specifically, the Socialality kernel explicitly learns target agent’s own grouping anchors to model such agent-specific grouping rules. Temporally coherent indicates that grouping is perceived by target agent not only based on past observations, but also over a temporal neighborhood around the current time, thus enforcing the original grouping stage to be extended. Context-adaptive indicates that the grouping rules are not single-context-guided, but structured with multiple grouping anchors. Accordingly, the proposed Socialality kernel uses multiple grouping anchors on an extended grouping stage to learn the agent-specific, temporally coherent, and context-adaptive grouping rules, therefore conducting robust grouping assignments.

Grouping Window Ω. To learn the temporally coherent grouping rules, the proposed Socialality kernel takes a grouping window shifted from observation-only window Ω to an extended temporal neighborhood window around current time Ω<sup>˜</sup> $( \Omega \mapsto \tilde { \Omega } )$ into account. An extended grouping window $\tilde { \Omega }$ consists of an observation time window Ω and an anticipation time window $\hat { \Omega } , i . e . , \tilde { \Omega } = \Omega \cup \hat { \Omega }$ . Trajectories during observed time window Ω are easy to obtain, as the short-term previews (future trajectories) require a simple prediction network. Such previews aim to provide only temporally coherent grouping cues around the current time for the Socialality kernel, rather than extra supervision in any subsequent modeling process. This short-term prediction network is trained by trajectories only within observation time window $\Omega _ { 0 }$ . Specifically, the observation time window $\Omega _ { 0 }$ is divided into two parts, $\Omega _ { 1 } = \{ - ( t _ { h } + t _ { f } ) + 1 , \ldots , - t _ { f } \}$ and $\Omega _ { 2 } = \{ - t _ { f } + 1 , \ldots , - 1 , 0 \} \left( t _ { o } = t _ { h } + t _ { f } \right)$ . During the training stage, the network uses all considered agents’ trajectories during $\Omega _ { 1 }$ as the input, and their trajectories during $\Omega _ { 2 }$ as the groundtruth. We define $\mathbf { x } ( | \Omega | , \operatorname* { m a x } \Omega ) = \mathbf { x } ( \Omega )$ (ends at $t = \operatorname* { m a x } \Omega$ with |Ω| steps) for better illustration. The $b e s t - o f - K _ { g } \ [ 8 ] \ \ell _ { 2 }$ loss function (short for preview loss) is used to supervise this training process:

$$
L _ { p } = \operatorname* { m i n } _ { 1 \leq s \leq K _ { g } } \frac { 1 } { t _ { f } | \mathcal { N } _ { i } | } \sum _ { j \in \mathcal { N } _ { i } } \left\| \hat { \mathbf { X } } _ { j } ^ { s } ( \pmb { \Omega } _ { 2 } ) - \mathbf { X } _ { j } ( \pmb { \Omega } _ { 2 } ) \right\| _ { 2 } .\tag{5}
$$

The forecasted trajectories $\left\{ { \bf X } _ { j } ( \hat { \Omega } | j \in { \mathcal N } _ { i } ) \right\}$ of all considered agents are then predicted by this short-term prediction network<sup>2</sup>. Correspondingly, the extended trajectory of any agent i is denoted as:

$$
\mathbf { X } _ { i } ( \tilde { \Omega } ) = \operatorname { C o n c a t } ( \mathbf { X } _ { i } ( \Omega ) , \mathbf { X } _ { i } ( \hat { \Omega } ) )\tag{6}
$$

Socialality Anchors $\tau _ { a }$ and $\tau _ { b } .$ . For all directly perceivable grouping-rule-modifiers, in-group social interactions are commonly characterized by both spatial distance and walking speed [13, 21], reflecting both the target agent $i \mathrm { \ ' } \mathrm { s }$ individual preference and the collective dynamics of its group. This observation suggests that grouping rules are not governed by a single fixed threshold, but are modulated along two dimensions. To enable the model to reflect such grouping rules observed in human social behavior, the proposed Socialality kernel introduces two learnable socialality anchors, which serve as parametric containers for the grouping-rule modifiers. Specifically, the anchors $\tau ^ { a }$ and $\tau ^ { b }$ respectively regulate the acceptable social distance and walking speed, thereby providing basis for adaptively modulating grouping rules in a human-inspired manner. These socialality anchors are also applied to calculate three modulation coeficients in the whole trajectory prediction model, which we will introduce later.

Given the target agent $i \mathrm { \ ' } _ { \mathrm { S } }$ trajectory during an extended grouping window $\tilde { \Omega } ,$ we use the network $m ( \cdot )$ to compute its socialality anchors:

$$
\left[ \tau _ { i } ^ { a } ( \tilde { \Omega } ) , \tau _ { i } ^ { b } ( \tilde { \Omega } ) \right] = m \left( { \bf X } _ { i } ( \tilde { \Omega } ) \right) = \tau _ { i } ( \tilde { \Omega } )\tag{7}
$$

See Table. 1 for detailed structures of the embedding network $m ( \cdot )$

The anchor $\tau _ { i } ^ { a } ( \tilde { \Omega } )$ is aimed to modulate the acceptable social distance range between agent i and its group members. Considering target agent i and any neighbor j during an extended grouping stage Ω, the Socialality kernel assumes that the step-wise<sup>˜</sup> distance $d _ { i } ^ { t } ( j ) = \big \| \tilde { \mathbf { p } } _ { j } ^ { t } - \mathbf { \tilde { p } } _ { i } ^ { t } \big \| _ { 2 } , t \in \tilde { \Omega }$ between $j$ and i should remain within $\tau _ { i } ^ { a } ( \tilde { \Omega } )$ modulated agent i’s displacement $p _ { i } ( \tilde { \Omega } ) = \left. \mathbf { p } _ { i } ^ { \mathrm { m a x } \tilde { \Omega } } - \mathbf { p } _ { i } ^ { \mathrm { m i n } \tilde { \Omega } } \right. _ { 2 }$ . Specifically, these two socialality anchors’s values are constrained to $( - 1 , 1 )$ by tanh activation, and thus we use $1 + \tau _ { i } ^ { a } ( \tilde { \Omega } )$ to set reasonable modulation range (0 , 2). Formally,

$$
S _ { i } ^ { a } ( j , \tau _ { i } ^ { a } | \tilde { \Omega } ) = \mathbb { I } \left[ \operatorname* { m a x } _ { t \in \tilde { \Omega } } \frac { d _ { i } ^ { t } ( j ) } { p _ { i } ( \tilde { \Omega } ) } < 1 + \tau _ { i } ^ { a } ( \tilde { \Omega } ) \right] .\tag{8}
$$

Here, we use the maximum over t to enforce a step-wise constraint: a neighbor is considered in-group only if it does not go beyond a certain distance range on each step.

Similar to $\tau _ { i } ^ { a } ( \tilde { \Omega } ) , \tau _ { i } ^ { b } ( \tilde { \Omega } )$ directly modulates the acceptable walking speed diference between the target agent i and its neighboring agent $j \in \mathcal { N } _ { i }$ . Given an extended grouping stage $\tilde { \Omega } .$ , their walking speed diference is represented by the ratio $\rho _ { i } ( j | \tilde { \Omega } )$ of their displacements: $\rho _ { i } ( j | \tilde { \Omega } ) = { \bar { p } _ { j } ( \tilde { \Omega } ) } / ( p _ { i } ( \tilde { \Omega } ) + \epsilon )$ , where ϵ is an infinitesimal quantity. Formally, this $\tau _ { i } ^ { b } ( \tilde { \Omega } )$ -modulated condition is formulated as:

$$
\begin{array} { r } { S _ { i } ^ { b } ( j , \tau _ { i } ^ { b } | \tilde { \Omega } ) = \mathbb { I } \left[ \left| \rho _ { i } ( j | \tilde { \Omega } ) - 1 \right| < \left| \tau _ { i } ^ { b } ( \tilde { \Omega } ) \right| \right] . } \end{array}\tag{9}
$$

Here, $| \tau _ { i } ^ { b } ( \tilde { \Omega } ) |$ controls the tolerance of speed mismatch, where larger values indicate more permissive grouping.

Accordingly, the Socialality kernel is finally defined as

$$
\begin{array} { r } { \mathcal { K } _ { i } ( j , \tau _ { i } | \tilde { \Omega } ) = { \cal S } _ { i } ^ { a } ( j , \tau _ { i } ^ { a } | \tilde { \Omega } ) \cdot { \cal S } _ { i } ^ { b } ( j , \tau _ { i } ^ { b } | \tilde { \Omega } ) . } \end{array}\tag{10}
$$

In the proposed Socialality Model, the subsequent concerns the extended time window $\tilde { \Omega }$ for group priors. Similarly, for our considered target agent i and extended time window $\tilde { \Omega } ,$ i’s group $\mathcal { G } _ { i }$ and out-of-group agents’ set $\overline { { \mathcal { G } } } _ { i }$ are computed as

$$
\mathcal { G } _ { i } = \{ j \in \mathcal { N } _ { i } | \mathcal { K } _ { i } ( j , \tau _ { i } | \tilde { \Omega } ) = 1 \} ,\tag{11}
$$

$$
\overline { { \mathscr { G } } } _ { i } = \{ j \in \mathcal { N } _ { i } | \mathcal { K } _ { i } ( j , \tau _ { i } | \tilde { \Omega } ) = 0 \} .\tag{12}
$$

Accordingly, sets $\mathcal { G } _ { i }$ and $\overline { { \mathcal { G } } } _ { i }$ are visible to trajectory prediction models, acting as group priors for the further modeling of group-wise social interactions.

Compared to the long-term distance and the Socialality kernel, the improved Socialality kernel uses two socialality anchors to adaptively learn target agent i’s grouping rules during the extended grouping window. It enables the presentation of agents’ diverse personalities and culturalities, thus modeling the agent-specific, temporally coherent, and context-adaptive grouping rules. These grouping rules yield more stable group priors around the current step, which in turn improves group-wise social interaction modeling in the subsequent perception mechanism.

## 3.3 Perception Mechanism

Both the original GPCC and the proposed Socialality Models share a similar perception mechanism. In the proposed Socialality Model, the perception mechanism utilizes the grouping priors obtained from the Socialality kernel to model target agent i’s social interactions of both in-group and out-of-group agents. To learn diferent interactions between in-group agents and out-of-group agents, we use two diferent strategies to imitate how humans perceive such group-wise social interactions. Given the set of target agent $i \mathrm { \ ' } _ { \mathrm { S } }$ in-group agents $\mathcal { G } _ { i }$ and their trajectories during the extended grouping stage $\tilde { \Omega }$ , the perception mechanism uses a self-trajectory encoder $l ( \cdot )$ and a group-trajectory encoder $e ( \cdot )$ to get i’s self representation and group representation, represented as

$$
\mathbf { f } _ { i } ^ { e } = l \left( \mathbf { X } _ { i } ( \tilde { \pmb { \Omega } } ) \right) ,\tag{13}
$$

$$
\mathbf { f } _ { i } ^ { g } = e \left( \left\{ \mathbf { X } _ { j } ( \tilde { \Omega } ) | j \in \mathcal { G } _ { i } \right\} \right) .\tag{14}
$$

See Table. 1 for detailed structure of encoder $e ( \cdot )$ . This structurally ensures that the diferences between how humans perceive in-group and out-of-group agents are learned from perception level intuitively.

The other strategy is applied on the out-of-group agents. Inspired by human perception of surrounding agents, we design a strategy that aggregates sensory cues to form the target agent i’s out-of-group agents’ representations. In this manuscript, we focus on modeling social interactions between the target agent and its neighboring agents, and thus do not explicitly consider static obstacles or other environmental elements. Among human sensory modalities, vision plays a central role in perceiving surrounding agents and planning future movements [52]. Accordingly, pedestrians tend to be more responsive to neighboring agents that fall within their field of view (FOV), i.e., the angular region of space visible to an agent i given its current heading direction $\mathbf { d } _ { i } ,$ as such neighboring agents provide more salient social cues [21]. The angle of FOV is denoted as $\theta _ { \mathrm { f o v } }$

The target agent primarily perceives neighboring agents’ social interactions with itself through observations within its field of view (FOV). In contrast, agents outside the FOV provide only limited perceptual cues, but their presence can still be sensed through peripheral awareness to some extent. It has been observed that humans are still aware of their neighboring agents even though they fall outside of the FOV, which is supported by sensitivity to luminance changes, and non-visual cues such as the sound of footsteps [53].

Here we define the unit vector $\mathbf { d } _ { i } ^ { \mathbf { p } }$ , which represents any direction of relative position p to the target agent i’s current position $\mathbf { p } _ { i } ^ { 0 }$ , as ${ \bf d } _ { i } ^ { \mathbf { p } } = \left( \bar { \mathbf { p } } - \mathbf { p } _ { i } ^ { 0 } \right) / \left( \left\| \mathbf { p } - \mathbf { p } _ { i } ^ { 0 } \right\| _ { 2 } \right) , \mathbf { p } \neq$ $\mathbf { p } _ { i } ^ { 0 }$ . The target agent i’s heading direction $\mathbf { d } _ { i }$ is denoted as its moving direction between last two adjacent time steps $t \in \{ - 1 , 0 \}$

$$
\mathbf { d } _ { i } = \frac { \mathbf { p } _ { i } ^ { 0 } - \mathbf { p } _ { i } ^ { - 1 } } { \left\| \mathbf { p } _ { i } ^ { 0 } - \mathbf { p } _ { i } ^ { - 1 } \right\| _ { 2 } + \epsilon } .\tag{15}
$$

Here, ϵ also indicates an infinitesimal quantity. During the observation window $\Omega _ { 0 } =$ $\{ - t _ { o } + 1 , \ldots , - 1 , 0 \}$ , the $\mathbf { F O V } _ { i }$ of agent i is then defined as the symmetric angular region centered around its heading direction $\mathbf { d } _ { i }$ :

$$
\mathbf { F O V } _ { i } = \left\{ \mathbf { p } \in \mathbb { R } ^ { 2 } \biggm | | \angle \big ( \mathbf { d } _ { i } ^ { \mathbf { p } } , \mathbf { d } _ { i } \big ) | \leq \frac { \theta _ { \mathrm { f o v } } } { 2 } \right\} ,\tag{16}
$$

Here $\angle ( \cdot )$ indicates signed angle, where positive and negative values correspond to counterclockwise (left) and clockwise (right) rotations, respectively.

For the target agent i at current time step $t = 0 = \operatorname* { m a x } \Omega _ { 0 }$ , the set of its in-FOV neighboring agents is defined by

$$
\mathcal { F } _ { i } = \left\{ j \bigm | j \in \overline { { \mathcal { G } } } _ { i } , \mathbf { p } _ { j } ^ { \operatorname* { m a x } \Omega _ { 0 } } \in \mathbf { F O V } _ { i } \right\} .\tag{17}
$$

Knowing only whether a neighboring agent lies within the target agent i’s FOV<sub>i</sub> is insuficient to model fine-grained social interaction. For instance, a neighboring agent may appear within the target agent’s FOV and move toward it. If they stick to their original heading direction, a collision may happen in this case. However, they usually make an instant decision of adjusting their heading directions to avoid the incoming intersection, which has been a subconscious process for pedestrians [54]. Inspired by the angle-based method proposed in SocialCircle [38], we regard such instant decisionmaking process needs a fast left/right choice, which corresponds to which side of FOV this incoming pedestrian falls in. If it falls within left side of FOV (left-FOV), the target agent tends to steer slightly to the right and vice versa.

Accordingly eq. (17), we further split $\mathcal { F } _ { i }$ into two subsets of neighbors according to left or right direction turning between $\mathbf { d } _ { i } ^ { \mathbf { p } }$ and $\mathbf { d } _ { i }$ . As for the outside part of the $\mathbf { F O V } _ { i } ,$ we mark it as the rear region. Formally, we denote sets of the target agent i’s out-of-group agents positioned in the left-FOV<sub>i</sub>, right-FOV<sub>i</sub> and rear region as

$$
\mathcal { C } _ { i } ^ { s } \subseteq \overline { { \mathcal { G } } } _ { i } , \ s \in \{ \mathrm { r i g h t } , \mathrm { l e f t } , \mathrm { r e a r } \}\tag{18}
$$

respectively.

To model perceptual asymmetry of neighboring agents among these regions, which specifically indicates how the target agent perceives in-FOV, and out-of-FOV agents, we adopt an information selection strategy over neighboring agents’ motion cues. For all out-of-group agents $\overline { { \mathcal { G } } } _ { i }$ , we assume that the target agent i mainly relies on three intuitive cues to perceive how they move in the scene in a SocialCircle-like way [38]: their distance to $i ,$ their relative moving direction<sup>3</sup> to i and their walking speed. For neighbors in the rear region, all motion cues except distance are set to zero, since agents outside the $\mathbf { F O V } _ { i } \left( i . e . , \overline { { \mathcal { G } } } _ { i } \setminus \mathcal { F } _ { i } = \mathcal { C } _ { i } ^ { \mathrm { r e a r } } \right)$ provide much weaker directional and velocity information, while distance can still be perceived through peripheral awareness. This aligns with the fact mentioned before that pedestrians tend to rely more on motion cues from agents within their FOV, while paying less attention to agents outside them.

Here, we compute three region-level aggregated cues by averaging over neighboring agents in each region:

$$
\phi _ { q } ( \mathcal { C } _ { i } ^ { s } ) = \frac { 1 } { | \mathcal { C } _ { i } ^ { s } | } \sum _ { j \in \mathcal { C } _ { i } ^ { s } } g _ { i } ^ { q } ( j ) , \ q \in \{ \mathrm { d i s , d i r , v e l } \} ,\tag{19}
$$

where $g _ { i } ^ { q } ( j )$ provides an in-region measurement between the target agent i and its each neighbor $j$ in the corresponding region:

$$
g _ { i } ^ { \mathrm { d i s } } ( j ) = \left\| \mathbf { p } _ { j } ^ { 0 } - \mathbf { p } _ { i } ^ { 0 } \right\| _ { 2 } ,\tag{20}
$$

$$
g _ { i } ^ { \mathrm { d i r } } ( j ) = \angle \left( { \bf d } _ { i } , { \bf d } _ { j } \right) ,\tag{21}
$$

$$
g _ { i } ^ { \mathrm { v e l } } ( j ) = \left. \mathbf { p } _ { j } ^ { 0 } - \mathbf { p } _ { j } ^ { - t _ { o } + 1 } \right. _ { 2 } .\tag{22}
$$

According to eq. (15), d<sub>j</sub> also denotes agent $j ^ { \circ } \mathrm { s }$ heading direction here. We concatenate the motion cues from the right, left, and rear regions to form the perception vector $\mathbf { r } _ { i } .$ . An embedding layer $h ( \cdot )$ then maps $\mathbf { r } _ { i }$ into a high-dimensional feature:

$$
\mathbf { f } _ { i } ^ { \overline { { g } } } = h ( \mathbf { r } _ { i } ) \in \mathbb { R } ^ { 2 d } .\tag{23}
$$

See Table. 1 for detailed structure of embedding layer $h ( \cdot )$

In summary, the perception mechanism extracts a region-aware representation that encodes perceptual asymmetry in a human-inspired manner, providing structured social cues $\mathbf { f } _ { i } ^ { e } , \mathbf { f } _ { i } ^ { g }$ and $\mathbf { f } _ { i } ^ { \overline { { g } } }$ for subsequent overall modeling.

## 3.4 The Socialality Trajectory Prediction Model

Building upon the Socialality kernel and the perception mechanism introduced above, we now formulate the complete Socialality trajectory prediction model. The perception mechanism models target agent $i \mathrm { \ ' } _ { \mathrm { S } }$ group-wise interactions based on the group priors provided by Socialality kernel. Similar to original GPCC Model in our conference paper [18], we fuse the social cues, $i . e . , \mathbf { f } _ { i } ^ { g } , \mathbf { f } _ { i } ^ { \overline { { g } } }$ , and the self representation $\mathbf { f } _ { i } ^ { e }$ to get final feature $\mathbf { f } _ { i }$ as the input to backbone prediction models. In the GPCC Model, the three representations are simply concatenated, implicitly assuming equal contributions from each branch. However, such an assumption may bias the optimization toward a dominant representation. To address this issue, we introduce three learnable modulation coeficients $c _ { 1 } , \ c _ { 2 }$ , and $c _ { 3 }$ , which adaptively rescale the representations prior before concatenation. These modulation coeficients $c _ { 1 } , \ : c _ { 2 }$ , and $c _ { 3 }$ are calculated from socialality anchors $\tau _ { i } ^ { a } ( \tilde { \Omega } )$ and $\tau _ { i } ^ { b } ( \tilde { \Omega } )$ . Specifically, larger $\tau _ { i } ^ { a } ( \tilde { \Omega } )$ relaxes distance tolerance, so we down-weight distance-sensitive group cues $\mathbf { f } _ { i } ^ { g }$ accordingly, while larger $\tau _ { i } ^ { b } ( \tilde { \Omega } )$ indicates more flexible speed choice, corresponding to more consideration of agent’s self-cues $\mathbf { f } _ { i } ^ { e }$ . Moreover, larger $\tau _ { i } ^ { a } ( \tilde { \Omega } )$ and $\tau _ { i } ^ { b } ( \tilde { \Omega } )$ both diminish the impact of out-group social cues $\mathbf { f } _ { i } ^ { \overline { { g } } }$

$$
c _ { 1 } = 1 + \tau _ { i } ^ { b } ( \tilde { \Omega } ) , c _ { 2 } = \left( 1 + \tau _ { i } ^ { a } ( \tilde { \Omega } ) \right) ^ { - 1 } , c _ { 3 } = \frac { c _ { 2 } } { c _ { 1 } } .\tag{24}
$$

This structurally ensures that subsequent fusion stage is further intuitively regularized. The fusion of social cues, $i . e . , \ \mathbf { f } _ { i } ^ { g } , \ \mathbf { f } _ { i } ^ { \overline { { g } } }$ , and $\mathbf { f } _ { i } ^ { e }$ , to form final representation $\mathbf { f } _ { i }$ is formulated as

$$
\mathbf { f } _ { i } = n \left( \mathrm { C o n c a t } \left( \mathrm { c } _ { 1 } \mathbf { f } _ { \mathrm { i } } ^ { \mathrm { e } } , \mathrm { c } _ { 2 } \mathbf { f } _ { \mathrm { i } } ^ { \mathrm { g } } , \mathrm { c } _ { 3 } \mathbf { f } _ { \mathrm { i } } ^ { \overline { { \mathrm { g } } } } \right) \right) .\tag{25}
$$

See Table. 1 for detailed structure of encoder $n ( \cdot )$

We take Transformer [55] and multi-style trajectory generation module in MSN [56] as backbone trajectory prediction model. The Transformer encoder captures high-dimensional representations of $\mathbf { f } _ { i }$ by modeling temporal dependencies through self-attention over $t _ { o }$ observation steps. Conditioned on the encoded features, the Transformer decoder attends to the encoder outputs while taking the linearly fitted trajectory $\hat { \mathbf { Y } } _ { i } ^ { l }$ of the target agent as the value input. The resulting feature embedding is then fed into the trajectory generation module [56] to produce future trajectories.

Formally,

$$
\hat { \mathbf { Y } } _ { i } ^ { s } = B _ { \mathrm { p r e d i c t i o n } } \left( \mathbf { f } _ { i } , \hat { \mathbf { Y } } _ { i } ^ { l } \right) , 1 \leq s \leq K _ { f } .\tag{26}
$$

## 3.5 Training

We adopt an end-to-end training strategy to train the whole Socialality Model, including the short-term future prediction network mentioned in section 3.2.2. Similar to the original GPCC Model, basic $\ell _ { 2 }$ loss is also used, which is calculated as

$$
L _ { o } = \operatorname* { m i n } _ { 1 \leq s \leq K _ { f } } \frac { 1 } { t _ { p } } \left\| \hat { \mathbf { Y } } _ { i } ^ { s } ( t _ { p } , t _ { p } ) - { \mathbf { Y } } _ { i } ( t _ { p } , t _ { p } ) \right\| _ { 2 } .\tag{27}
$$

Accordingly, the proposed Socialality trajectory prediction model is trained with the weighted sum of the basic $\ell _ { 2 }$ loss $L _ { o }$ and the preview loss $L _ { p } { \mathrm { : } }$

$$
L = L _ { o } + \beta L _ { p } .\tag{28}
$$

## 4 Experiments

## 4.1 Experimental Settings

## 4.1.1 Datasets

ETH-UCY [6, 57] includes several videos captured in pedestrian walking scenes. We follow the leave-one-out [1] strategy to train and validate the proposed Socialality Model with $\{ t _ { o } , t _ { p } \} = \{ 8 , \bar { 1 2 } \}$ and a sampling interval of $T = 0$ .4s. In the short-term future prediction subnetwork, we set $\{ t _ { h } = 4 , t _ { f } = 4 \}$

Stanford Drone Dataset (SDD) [58] comprises 60 video recordings obtained from an aerial perspective of the campus. The agents, which fall into diferent categories (e.g., pedestrian, bicyclist, skateboarder, cart, car, and bus), have been labeled in pixels. Following the same settings as previous researchers [59], the 60% of videos were designated for training, the 20% for validation, and the 20% for testing. We also set $\{ t _ { o } , t _ { p } , T \} = \{ 8 , 1 2 , 0 . 4 \mathrm { s } \}$ and $\{ t _ { h } , t _ { f } \} = \{ 4 , 4 \}$

## 4.1.2 Metrics

Following previous researchers, we also use the minimum Average/Final Displacement Error over K generated trajectories $[ 1 , 6 , 8 ]$ to evaluate models’ performance, known as $\mathrm { \mathrm { \Omega } ^ { \mathrm { \tiny { c } } } m i n A D E } _ { K _ { f } } / \mathrm { m i n F D E } _ { K _ { f } } , \mathrm { \Omega }$ , where $K _ { f }$ is usually set to 20 when evaluating. For agent $i ,$ using subscript $k$ to denote the kth prediction, we have

$$
\mathrm { m i n A D E } _ { 2 0 } ( i ) = \operatorname* { m i n } _ { k \in \{ 1 , 2 , \ldots , K _ { f } \} } \frac { 1 } { t _ { p } } \sum _ { t = 1 } ^ { t _ { p } } \left\| \hat { { \mathbf p } } _ { i } ^ { t _ { k } } - { \mathbf p } _ { i } ^ { t } \right\| _ { 2 } ,\tag{29}
$$

$$
\mathrm { m i n F D E } _ { 2 0 } ( i ) = \operatorname* { m i n } _ { k \in \{ 1 , 2 , . . . , K _ { f } \} } \left\| \hat { { \mathbf p } } _ { i } ^ { t _ { p _ { k } } } - { \mathbf p } _ { i } ^ { t _ { p } } \right\| _ { 2 } .\tag{30}
$$

Table 2 Comparisons to the state-of-the-art methods on ETH-UCY dataset
<table><tr><td>Models</td><td>eth</td><td>hotel</td><td>univ</td><td>zaral</td><td>zara2</td><td>Avg.</td></tr><tr><td>GP-Graph-STG[11] (2022)</td><td>0.48/0.77</td><td>0.24/0.40</td><td>0.29/0.47</td><td>0.24/0.42</td><td>0.23/0.40</td><td>0.29/0.49</td></tr><tr><td>GP-Graph-PEC[11] (2022)</td><td>0.56/0.82</td><td>0.18/0.26</td><td>0.31/0.46</td><td>0.23/0.40</td><td>0.17/0.27</td><td>0.29/0.44</td></tr><tr><td>GroupNet+CVAE[15] (2022)</td><td>0.46/0.73</td><td>0.15/0.25</td><td>0.26/0.49</td><td>0.21/0.39</td><td>0.17/0.33</td><td>0.25/0.44</td></tr><tr><td>GroupNet+T++[15] (2022)</td><td>0.38/0.74</td><td>0.11/0.20</td><td>0.19/0.40</td><td>0.14/0.32</td><td>0.11/0.25</td><td>0.19/0.38</td></tr><tr><td>Y-net[51] (2021)</td><td>0.28/0.33</td><td>0.10/0.14</td><td>0.24/0.41</td><td>0.17/0.27</td><td>0.13/0.22</td><td>0.18/0.27</td></tr><tr><td>SEEM[60] (2023)</td><td>0.62/1.20</td><td>0.61/1.21</td><td>0.50/1.04</td><td>0.31/0.61</td><td>0.36/0.68</td><td>0.48/0.95</td></tr><tr><td>EqMotion[61] (2023)</td><td>0.40/0.61</td><td>0.12/0.18</td><td>0.23/0.43</td><td>0.18/0.32</td><td>0.13/0.23</td><td>0.21/0.35</td></tr><tr><td>MSN[56] (2023)</td><td>0.27/0.41</td><td>0.11/0.17</td><td>0.28/0.48</td><td>0.22/0.36</td><td>0.18/0.29</td><td>0.21/0.34</td></tr><tr><td>LMTraj-SUP [41] (2024)</td><td>0.65/1.04</td><td>0.26/0.46</td><td>0.57/1.16</td><td>0.51/1.01</td><td>0.38/0.74</td><td>0.48/0.88</td></tr><tr><td>ET+HG[62] (2024)</td><td>0.33/0.56</td><td>0.13/0.21</td><td>0.23/0.47</td><td>0.19/0.33</td><td>0.15/0.25</td><td>0.21/0.36</td></tr><tr><td>LG-Traj[63] (2024)</td><td>0.38/0.56</td><td>0.11/0.17</td><td>0.23/0.42</td><td>0.18/0.33</td><td>0.14/0.25</td><td>0.20/0.34</td></tr><tr><td>SocialCircle[38] (2024)</td><td>0.25/0.38</td><td>0.12/0.14</td><td>0.23/0.42</td><td>0.18/0.29</td><td>0.13/0.22</td><td>0.18/0.29</td></tr><tr><td>SocialCircle+[39] (2024)</td><td>0.25/0.39</td><td>0.10/0.15</td><td>0.24/0.42</td><td>0.18/0.28</td><td>0.13/0.22</td><td>0.18/0.29</td></tr><tr><td>UPDD[64] (2024)</td><td>0.22/0.42</td><td>0.17/0.30</td><td>0.14/0.28</td><td>0.16/0.30</td><td>0.14/0.31</td><td>0.17/0.32</td></tr><tr><td>E-V2-Net[65] (2025)</td><td>0.25/0.38</td><td>0.11/0.16</td><td>0.23/0.42</td><td>0.19/0.30</td><td>0.13/0.24</td><td>0.18/0.30</td></tr><tr><td>Resonance[42] (2025)</td><td>0.23/0.35</td><td>0.10/0.15</td><td>0.24/0.41</td><td>0.17/0.29</td><td>0.13/0.22</td><td>0.17/0.28</td></tr><tr><td>GPCC (Ours)</td><td>0.25/0.38</td><td>0.10/0.15</td><td>0.25/0.44</td><td>0.17/0.28</td><td>0.13/0.22</td><td>0.18/0.29</td></tr><tr><td>Socialality (Ours)</td><td>0.23/0.37</td><td>0.10/0.16</td><td>0.24/0.43</td><td>0.17/0.29</td><td>0.13/0.22</td><td>0.17/0.29</td></tr></table>

Metrics are reported in the form of “ADE/FDE” (best-of-20) in meters. Lower metrics indicate better prediction performance. Blue markers denote the best 3 results on each set.

The final metrics are averaged over every agent in each set, and we will refer to them as ${ } ^ { 6 4 } \mathrm { A D E / F D E ^ { \prime } }$ for clarity.

## 4.1.3 Implementation Details

Both original GPCC and the proposed Socialality Models are trained on one NVIDIA GeForce RTX 3090. Feature dimension of most subnetworks is set to d = 32. The preview loss ratio is set to $\beta = 0 . 4$ . The FOV angle of the perception mechanism is set to be 180<sup>◦</sup>(discussion of choosing the FOV angle in our conference paper [18]). Following [27], trajectories are preprocessed by moving to (0, 0). We use the Adam optimizer with a learning rate 0.0002 to train our models, and the batch size is 1000 for up to 200 epochs.

## 4.2 Comparisons to State-of-the-Art Methods

In this section, we compare Socialality Model with recent state-of-the-art trajectory forecasting methods on the ETH-UCY and SDD benchmarks. Results are summarized in Table. 2 and Table. 3.

On ETH-UCY, Socialality Model consistently outperforms our previous GPCC model across most scenes. In GPCC, grouping is determined by a static longterm distance threshold, which cannot adapt to diferent social configurations. By introducing the agent-specific Socialality kernel, Socialality Model produces more stable grouping decisions, leading to lower ADE and FDE. The improvement is more evident in crowded scenes such as univ and eth, where group formations frequently change and agents exhibit diverse motion patterns. Compared with representative graph-based group-aware methods such as GP-Graph-STGCNN (0.29/0.49) and GP-Graph-PECNet (0.29/0.44), Socialality Model reduces the average ADE by approximately 41%. It also outperforms recent interaction-based models such as LGTRAJ (0.20/0.34) and EigenTrajectory+HighGraph (0.21/0.36).

Table 3 Comparisons to the state-of-the-art methods on SDD dataset
<table><tr><td>Models</td><td>ADE/FDE</td><td>Models</td><td>ADE/FDE</td></tr><tr><td>SpecTGNN[66] (2021)</td><td>8.21/12.41</td><td>IMP[67] (2023)</td><td>8.98/15.54</td></tr><tr><td>Y-net[51] (2021)</td><td>7.85/11.85</td><td>LMTraj-SUP [41] (2024)</td><td>17.5/34.5</td></tr><tr><td>GP-Graph-STGCNN[11] (2022)</td><td>10.6/20.5</td><td>RAN[68] (2024)</td><td>10.97/19.95</td></tr><tr><td>GP-Graph-PECNet[11] (2022)</td><td>9.1/13.8</td><td>LG-Traj[63] (2024)</td><td>7.80/12.79</td></tr><tr><td>GroupNet+PECNet[15] (2022)</td><td>9.65/15.34</td><td>UPDD[64] (2024)</td><td>6.59/13.90</td></tr><tr><td>GroupNet+CVAE[15] (2022)</td><td>9.31/16.11</td><td>SocialCircle[38] (2024)</td><td>6.54/10.36</td></tr><tr><td>NSP-SFM[69] (2022)</td><td>6.52/10.61</td><td>SocialCircle+[39] (2024)</td><td>6.44/10.22</td></tr><tr><td>MUSE-VAE[70] (2022)</td><td>6.36/11.10</td><td>E-V2-Net[65] (2025)</td><td>6.57/10.49</td></tr><tr><td>FlowChain[71] (2023)</td><td>9.93/17.17</td><td>Resonance[42] (2025)</td><td>6.27/10.02</td></tr><tr><td>GPCC (Ours)</td><td>6.39/10.17</td><td>Socialality (Ours)</td><td>6.28/10.12</td></tr></table>

Metrics are reported in the form of “ADE/FDE” (best-of-20) in pixels rather than meters compared to ETH-UCY. Lower metrics indicate better prediction performance. Blue markers denote the best 3 results on each set.

SDD contains larger spatial layouts and more diverse motion scales. Socialality Model’s performance reaches 6.28/10.12, outperforming the original GPCC(6.39/10.17). Compared with earlier graph-based approaches, the improvement is more substantial. For instance, relative to GroupNet+PECNet (9.65/15.34) and GP-Graph-STGCNN (10.6/20.5), the ADE reduction exceeds 34% and 40%, respectively. Even against strong recent models such as SocialCircle (6.54/10.36) and NSPSFM (6.52/10.61), Socialality Model achieves lower prediction errors. Since SDD scenes often include agents with diferent motion intentions and spatial tolerances, a fixed threshold-based grouping strategy becomes less reliable. The proposed anchor-based kernel adjusts grouping rules according to each agent’s motion pattern, resulting in more accurate interaction modeling.

Notably, the proposed Socialality Model’s backbone architecture remains identical to the original GPCC. The main structural modification lies in replacing the static long-term distance kernel with the proposed Socialality kernel. The consistent improvements on both datasets indicate that agent-specific, temporally coherent, and context-adaptive grouping rules leads to better trajectory forecasting performance.

## 4.3 Ablation Analyses

In this section, we provide quantitative evaluations to further validate the proposed Socialality Model. First, we conduct separate ablations to verify whether grouping priors actually contribute to the prediction model. Then we conduct several further ablations to validate the contribution of each key component that modifies the Socialality kernel’s grouping rules, including the socialality anchors and the extended grouping window. Detailed ablation settings and results are reported in Table. 4 and Table. 5.

Grouping Kernels. The main diference among the three variations in Table. 4 is that whether the groups have been formed or how the groups are formed before forecasting. Variation a1 serves as the baseline for comparisons, with no grouping priors introduced and no group-aware social interactions considered. In contrast, grouping kernels are introduced in both variations a2 and a0 to provide direct grouping priors for the subsequent interaction modeling. The diference between a2 and a0 is that a2 uses the original long-term distance kernel, while a0 adopts our proposed Socialality kernel.

Table 4 Ablation studies on validating the grouping priors on ETH-UCY and SDD datasets
<table><tr><td>ID</td><td>Grouping Kernel</td><td>eth</td><td>hotel</td><td>univ</td><td>zaral</td><td>zara2</td><td>SDD</td></tr><tr><td>a1</td><td></td><td>0.510 /1.052</td><td>0.117/0.163</td><td>0.304 /0.575</td><td>0.188 /0.302</td><td>0.147 0.231</td><td>6.462/10.379</td></tr><tr><td>a2</td><td>Long-Term Distance</td><td>0.254/0.383</td><td>0.103/ /0.156</td><td>0.256/ /0.448</td><td>0.175 /0.284</td><td>0.134/0.225</td><td>6.394/10.173</td></tr><tr><td>a0</td><td>| Socialality</td><td>0.233/0.369</td><td>0.102/0.156</td><td>0.240/ /0.428</td><td>0.168/ /0.294</td><td>0.130/0.223</td><td>6.283/10.121</td></tr></table>

“-” denotes that no grouping kernel is enabled in the corresponding variation model. The variation that uses the Long-Term Distance Kernel is the same as our original GPCC Model. Blue markers denote the best results on each set. Red markers denote the worst results on each set.

Based on such setup and the results in Table. 4, we find that both variations a2 and a0 achieve 50.2%/63.6% significantly better ADE/FDE on eth, compared with the base variation a1. Compared with the unbounded variation a1, a0 finally achieves up to 40% ADE gains across these datasets, though these variations share the same backbone prediction model with exactly the same network capacity. This means that the explicit modeling of social interactions with specific grouping priors has brought considerable performance improvements in the group-bounded variations a0 and a2 over the unbounded a1, regardless of which grouping kernel has been used.

Moreover, the full Socialality Model (a0) further obtains up to 4.1% better ADE on eth and 4.5% FDE on univ than variation a2 (GPCC, which uses our original long-term distance kernel). Here, the most significant diference between such kernels is that the recommendations of group members rely on multiple adaptively collaborated anchors in the proposed Socialality kernel, rather than a single fixed threshold in the long-term distance kernel. We can infer from such performance diferences that the real world groupings are more likely to be determined dynamically and collaboratively. See our further discussions in section 4.4.2. These results indicate that stable performance gains could be obtained through such explicit grouping priors, while the proposed Socialality kernel still has the ability to further expand network capability by refining our original grouping rules in GPCC, especially in scenarios with more complex interactions and higher uncertainty like univ and eth.

Socialality Anchors. We next validate the grouping rules in the proposed Socialalityin Table. 5. The grouping rules are mainly shaped by two components, i.e., the proposed socialality anchors and the extended grouping window. The socialality anchors are designed to encode context-adaptive and agent-specific grouping preferences. In particular, they modulate the acceptable social distance and speed flexibility for grouping, directing diferent agents to follow diferent grouping rules. The extended grouping window is designed to provide temporally extended grouping cues around the current time step. Rather than making grouping decisions from a single instantaneous moment or only observed period, the extended grouping window allows the Socialality to incorporate grouping evidence from both reviews (observed window) and previews (short-term forecasted window).

Table 5 Ablation studies on validating the proposed grouping rules on ETH-UCY and SDD datasets
<table><tr><td>ID</td><td> $\tau ^ { a } ~ \tau ^ { b }$ </td><td></td><td>RPC</td><td>eth</td><td>hotel</td><td>univ</td><td>zara1</td><td>zara2</td><td>SDD</td></tr><tr><td>a1</td><td> $\mathbf { \Sigma } - \mathbf { \Sigma } \checkmark$ </td><td></td><td> $\checkmark { \checkmark }$ </td><td>0.252/0.398</td><td> $0 . 1 1 5 / 0 . 1 8 0$ </td><td> $0 . 2 5 1 / 0 . 4 5 1$ </td><td> $0 . 1 8 5 / 0 . 3 1 7$ </td><td> $0 . 1 4 1 / 0 . 2 4 1$ </td><td>6.431/10.452</td></tr><tr><td>a2</td><td> $\checkmark -$ </td><td></td><td> $\checkmark { \checkmark }$ </td><td>0.240/0.384</td><td>0.110/0.175</td><td>0.244/0.440</td><td>0.183/0.313</td><td>0.139/0.235</td><td>6.343/10.278</td></tr><tr><td>a3</td><td></td><td> $\checkmark 0$ </td><td> $\checkmark { \checkmark }$ </td><td>0.232/0.364</td><td>0.104/0.158</td><td>0.241/0.434</td><td>0.180/0.305</td><td>0.134/0.225</td><td>6.305/10.189</td></tr><tr><td>a4</td><td> $0 \checkmark$ </td><td></td><td> $\checkmark { \checkmark }$ </td><td>0.237/0.375</td><td>0.107/0.169</td><td>0.243/0.435</td><td>0.180/0.304</td><td>0.137/0.233</td><td>6.331/10.218</td></tr><tr><td>a5</td><td></td><td>00</td><td> $\checkmark { \checkmark }$ </td><td>0.234/0.370</td><td>0.106/0.166</td><td>0.242/0.434</td><td>0.180/0.308</td><td>0.132/0.222</td><td>6.313/10.262</td></tr><tr><td>a6</td><td> $\checkmark \blacktriangledown$ </td><td></td><td> $\mathbf { \Sigma } - \mathbf { \Sigma } \mathbf { - } \mathbf { \Sigma }$ </td><td>0.238/0.378</td><td>0.116/0.187</td><td>0.248/0.460</td><td>0.184/0.325</td><td>0.137/0.238</td><td>6.356/10.315</td></tr><tr><td>a7</td><td></td><td> $\checkmark \blacktriangledown$ </td><td> $\checkmark - \checkmark$ </td><td>0.231/0.363</td><td>0.113/0.179</td><td> $0 . 2 4 7 \dot { / } 0 . 4 4 1$ </td><td>0.183/0.317</td><td>0.135/0.230</td><td>6.260/10.020</td></tr><tr><td>a8</td><td></td><td> $\checkmark \blacktriangledown$ </td><td> $\mathbf { \Pi } - \mathbf { \Pi } \checkmark -$ </td><td>0.234/0.367</td><td>0.110/0.168</td><td> $0 . 2 4 4 \dot { / } 0 . 4 3 6$ </td><td>0.182/0.308</td><td>0.135/0.229</td><td>6.303/10.194</td></tr><tr><td>a0</td><td></td><td> $\checkmark \blacktriangledown$ </td><td> $\checkmark { \checkmark }$ </td><td>0.233/0.369</td><td>0.102/0.156</td><td> $0 . 2 4 0 / 0 . 4 2 8$ </td><td> $0 . 1 6 8 / 0 . 2 9 4$ </td><td>0.130/0.223</td><td> $6 . 2 8 3 / 1 0 . 1 2 1$ </td></tr></table>

τ<sup>a</sup> and $\tau ^ { b }$ denote whether the corresponding socialality anchor is activated during training. “R”, “P”, and ${ } ^ { \mathfrak { s o } }$ denote whether reviews, previews, or current time are included in the grouping window respectively. “0” denotes the corresponding socialality anchor is set to 0 during training. “-” denotes the corresponding socialality anchor is completely removed. Both anchors are activated in the proposed full Socialality Model. Blue markers denote the best results on each set. Red markers denote the worst results on each set.

(i) Overall Verification: Among all variations, the full Socialality (a0) achieves the best or near-best performance on almost all subsets. This result indicates that the complete grouping rules, jointly shaped by the two socialality anchors and the extended grouping window, are the most efective for trajectory forecasting. For example, on hotel, the full model enhances ADE/FDE from 0.115/0.180 in a1 to 0.102/0.156, corresponding to improvements of 11.3% and 13.3%, respectively. On univ, compared with the current-only window variant a6, the full model further reduces ADE/FDE from 0.248/0.460 to 0.240/0.428, yielding improvements of 3.2% and 7.0%. These observations suggest that both the agent-specific anchor design and the temporally extended grouping cues are important to the final prediction performance.

(ii) Anchor Verification: We evaluate the contribution of the proposed socialality anchors from two aspects, including the whether each anchor exists and the respective roles of distance anchor $\tau ^ { a }$ and speed anchor $\tau ^ { b }$

We first compare the variations where an anchor is set to 0 with those where the corresponding anchor is completely removed from the whole model. These two degradations are distinctively diferent. When an anchor is removed, it is discarded from the entire model pipeline. Accordingly, it is not involved in learning agent-specific grouping preferences in the grouping stage, nor can it contribute to the subsequent modulation of leveled interactions. In contrast, the anchor set to 0 loses its capability to learn specific grouping preferences and applies the same fixed anchor value to all neighboring agents instead. However, its associated constraint and modulation efect are still partially preserved in the model. In summary, setting an anchor to 0 mainly weakens the learning of agent-specific grouping rules, while removing it disables the corresponding constraint completely.

When both anchors are set to 0 (a5), the model still keeps competitive performance on most scenes, such as 0.234/0.370 on eth and 0.242/0.434 on univ, which are only slightly worse than the full Socialality model by 0.4%/0.3% and 0.8%/1.4% in ADE/FDE. In contrast, when one anchor is completely removed, the performance drops more clearly than merely setting it to 0. This diference can be seen from paired comparisons where only one variable is changed at a time. When $\tau ^ { a }$ is removed while $\tau ^ { b }$ is retained, a1 performs worse than a4, where $\tau ^ { a }$ is instead fixed to 0. For example, on eth, the ADE/FDE gets worse from 0.237/0.375 in a4 to 0.252/0.398 in a1, corresponding to degradations of 6.3% and 6.1%. Similarly, on hotel, the performance drops with degradations of 7.5% and 6.5% in a1. A similar trend can also be observed for $\bar { \boldsymbol { \tau } } ^ { b }$ by comparing a2 and a3. When $\tau ^ { b }$ is removed, a2 is consistently worse than a3, where $\tau ^ { \dot { b } }$ is fixed to 0. These observations suggest that retaining an anchor in the model with a fixed value is diferent from removing it completely, and the two degradations lead to systematically diferent performance.

We further verify the respective roles of $\tau ^ { a }$ and $\tau ^ { b }$ . By comparing a3 and a4, we can observe that a3 is also consistently better than enabling only $\tau ^ { b }$ while setting $\tau ^ { a }$ to 0 (a4) on several subsets. For example, on hotel, a3 improves over a4 from 0.107/0.169 to 0.104/0.158, corresponding to relative gains of 2.8% and 6.5% in ADE/FDE. On zara2, a3 further improves over a4 from 0.137/0.233 to 0.134/0.225, yielding gains of 2.2% and 3.4%, respectively. A similar trend can also be observed when one anchor is completely removed. For example, on hotel, a2 improves over a1 with the gains of 4.3% and 2.8%. On univ, the corresponding gains are 2.8% and 2.4%, with the performance improving from 0.251/0.451 to 0.244/0.440. On zara2, a2 further improves over a1 1.4% and 2.5%, on ADE/FDE respectively. We can infer from these observations that $\tau ^ { a }$ plays a more basic role in grouping, while $\tau ^ { b }$ mainly acts as a complementary refinement.

(iii) Grouping Window Verification: We further validate the role of the extended grouping window. Compared with the current-only variant a6, introducing reviews or previews consistently improves the prediction performance. When reviews are added, a7 improves over a6 2.6% and 4.3% in ADE/FDE on hotel. On univ, the corresponding gains are 0.4% and 4.1%, and on zara1 they are 0.5% and 2.5%. Similarly, when previews are introduced, a8 improves over a6 on hotel, univ, and zara1 by 5.2%/10.2%, 1.6%/5.2%, and 1.1%/5.2% in ADE/FDE, respectively.

Moreover, using only previews (a8) even performs slightly better than using reviews with the current time step (a7) on most subsets, including hotel (2.7%/6.1%), univ (1.2%/1.1%), zara1 (0.5%/2.8%), and zara2 (0.0%/0.4%) in ADE/FDE, respectively. This observation suggests that grouping evidence may be direction-sensitive. In many scenes, group relations may be more clearly reflected by whether agents are likely to continue moving coherently in the near future, rather than only by whether they remained close in the observed past.

When reviews, previews, and the current time step are all included together, the full model a0 achieves the best overall performance among all grouping-window variants. Compared with a7, the full model’s ADE/FDE further improves 9.7% and 12.8% on hotel. Compared with a8, the corresponding gains on zara1 are 7.7% and 4.5%. Overall, these results suggest that the extended grouping window improves the prediction performance.

Table 6 Ablation studies on validating the preview loss ratio $\beta$ on ETH-UCY and SDD datasets
<table><tr><td>ID</td><td> $\beta$ </td><td>eth</td><td>hotel</td><td>univ</td><td>zara1</td><td>zara2</td><td>SDD</td></tr><tr><td>a1</td><td>0</td><td>0.235/0.374</td><td>0.111/0.176</td><td>0.248/0.440</td><td>0.183/0.311</td><td>0.135/0.230</td><td>6.419/10.421</td></tr><tr><td>a2</td><td>0.2</td><td>0.236/0.377</td><td>0.109/0.168</td><td>0.239/0.424</td><td>0.177/0.295</td><td>0.137/0.231</td><td>6.274/10.148</td></tr><tr><td>a3</td><td>0.6</td><td>0.232/0.362</td><td>0.108/0.166</td><td>0.242/0.432</td><td>0.185/0.318</td><td>0.136/0.232</td><td>6.296/10.155</td></tr><tr><td>a4</td><td>0.8</td><td>0.234/0.365</td><td>0.104/0.157</td><td>0.242/0.431</td><td>0.176/0.302</td><td>0.132/0.233</td><td>6.344/10.241</td></tr><tr><td>a5</td><td>1.0</td><td>0.235/0.365</td><td>0.110/0.171</td><td>0.245/0.437</td><td>0.180/0.305</td><td>0.138/0.234</td><td>6.369/10.340</td></tr><tr><td>a0</td><td>0.4</td><td>0.233/0.369</td><td>0.102/0.156</td><td>0.240/0.428</td><td>0.168/0.294</td><td>0.130/0.223</td><td>6.283/10.121</td></tr></table>

The preview loss ratio $\beta$ is set with diferent values from 0 to 1.0. The default setting is $\beta = 0 . 4 .$ Blue markers denote the best results on each set.

Table 7 Ablation studies on validating the short-term prediction networks on ETH-UCY and SDD datasets
<table><tr><td>ID</td><td>Backbone</td><td>eth</td><td>hotel</td><td>univ</td><td>zaral</td><td>zara2</td><td>SDD</td></tr><tr><td>a1</td><td>Linear</td><td>0.236/0.372</td><td>0.111/0.172</td><td>0.242/0.435</td><td>0.182/0.307</td><td>0.137/0.233</td><td>6.375/10.194</td></tr><tr><td>a2</td><td>MLP</td><td>0.237/0.373</td><td>0.110/0.183</td><td>0.245/0.437</td><td>0.183/0.312</td><td>0.137/0.231</td><td>6.438/10.422</td></tr><tr><td>a0</td><td>Tran</td><td>0.233/0.369</td><td>0.102/0.156</td><td>0.240/0.428</td><td>0.168/0.294</td><td>0.130/0.223</td><td>6.283/10.121</td></tr></table>

Backbone indicates diferent backbones used in short-term prediction networks. “Linear” means simple linear fit is applied to generate short-term previews. “MLP” is used in another variation. The default setting in the full Socialality Model uses “Tran (Transformer)” as the prediction backbone. Blue markers denote the best results on each set.

Additional Verifications. We additionally evaluate the design of the short-term preview network. Results are summarized in Table. 6 and Table. 7.

(i) Preview Loss Ratio Verification: We first analyze the efect of the preview loss ratio β. Overall, diferent values of β lead to relatively moderate performance variations across scenes, while a moderate setting gives the best overall balance. When β = 0 (a1), i.e., without preview supervision, the performance degrades on several subsets. For example, on univ, the ADE/FDE is worse than a0 of 3.3%/2.8%. When $\beta$ is set to 0.2 (a2), the performance on univ becomes the best among all settings. When $\beta$ is increased to 0.6 (a3), eth reaches the best result of 0.232/0.362, but zara1 worsens 10.1% and 8.2% than a0. Further increasing β to 1.0 (a5) leads to another drop on multiple subsets, such as hotel, where the ADE/FDE worsens 7.8%/9.6%. We can infer from the above observations that preview supervision is beneficial, but its weight should remain within a moderate range.

(ii) Backbone Verification: We then validate the role of the short-term prediction network with diferent prediction backbones. Although the default Socialality uses a Transformer predictor, replacing it with simpler alternatives still yields reasonably competitive performance across scenes. For example, using a Linear predictor (a1) gives 0.236/0.372 on eth, which is only 1.3% and 0.8% worse than the full model, while using an MLP predictor (a2) leads to similarly small degradations of 1.7% and 1.1%. It should be noted that the main purpose of this short-term prediction network is not to rely on a specific predictor architecture. Instead, the key point is to introduce a short-term prediction network itself, so that the model can obtain preview cues for grouping.

## 4.4 Discussions

In this section, we further analyze and discuss the proposed Socialality Model through multiple validations. As defined in section 3, grouping kernels in the proposed trajectory prediction models determine whether each neighbor belongs to a target agent’s group. Specifically, for the same target agent, we focus on evaluating diferent grouping decisions of the proposed Socialality kernel compared with the original long-term distance kernel. Moreover, the core of the Socialality kernel is a pair of grouping anchors, i.e., socialality anchors, modulating acceptable social distances and walking speeds within each target agent’s group. We visualize and analyze how such agent-specific socialality anchors distribute in a 2D space at diferent scenes.

## 4.4.1 Overall Model Predictions

![](images/1aaa60267bd53aabe591a66f6091f496e12915125836c4a58821d215d358fb88.jpg)  
Fig. 4 Visualized predictions of the original GPCC Model and the proposed Socialality Model in diferent scenes.

Here, we visualize trajectories predicted by the proposed Socialality Model and the original GPCC Model under representative scenes from ETH-UCY and SDD, as shown in Fig. 4. Overall, we could observe that the Socialality Model produces more behaviorally plausible and scene-consistent future trajectories than GPCC Model. In (a1) and (b1), the original GPCC Model tends to generate less stable predictions for agents with weak motion cues, e.g., mostly generating sudden movements for nearly static agents. By contrast, the proposed Socialality Model yields more comprehensive yet still reasonable predictions, better reflecting the fact that such agents are more likely to remain locally stable instead of moving abruptly. In the SDD cases shown in (a7) and (b7), Socialality Model produces noticeably more multi-style predictions than the original GPCC Model. Such diversity better matches the highly heterogeneous motion patterns in SDD, where agents are often influenced by open spaces, loose interactions, and multiple navigable choices. In summary, the proposed model not only improves the rationality of prediction outcomes, but also enhances the expressiveness of trajectory modes under diferent social environments.

## 4.4.2 Discussions of Groupings

![](images/2f34c2be48e476918f61120a7a1068b55ccf231210d5b4eb3e888a65eb37c4fb.jpg)  
Fig. 5 Visualized grouping results of the original GPCC Model and the proposed Socialality Model in diferent scenes. GPCC uses the original observation-only grouping window, whereas Socialality uses the proposed extended grouping window.

Overall Validations. To qualitatively evaluate the evolving grouping capability of the proposed Socialality kernel, we visualize how Socialality kernel handles cases where the long-term distance kernel fails to process in Fig. 5. Here, we treat agents walking together, having social interactions (talking to each other, eye contact, and so on), and share similar destinations as likely group members.

It can be seen from the above figures that grouping decisions vary substantially across diferent prediction models even for the same target agent. By comparing (a1), (a2), (a3) and (b1), (b2), (b3) in Fig. 5, we can observe that one neighboring agent is considered to be within the group by Socialality Model, while the original GPCC Model regards them as out-of-group agents due to the static-thresholdcontrolled grouping kernel. The static nature of this threshold determines that the choice of this threshold can only be a trade-of<sup>4</sup>. On the one hand, we could set a relatively larger threshold value to let the long-term distance kernel cover more neighboring agents when grouping, but large threshold also means including more agents that are not the target agent’s group members. On the other hand, a relatively smaller threshold could be applied to strictly constrain the group “membership”, thus overlooking in-group agents that are slightly away from the target agents. With the newly proposed Socialality kernel, we could observe that a precise group assignment is conducted even in crowded scenes like (b2) and (b5).

Previous studies have shown that relatively large groups (with more than three members) tend to move in a “V”-shaped formation to ensure in-group social interactions [9]. Here, (a6) and (b6) represent a large-group scene, where we could observe five group members (including the target agent itself). In (a5), the long-term distance kernel simply assigns the nearest neighbor as a group member and fails to recognize the rest of the group members. In such large-group cases, the original long-term distance kernel could only conduct incomplete group assignments based on the distance threshold alone, which neglects the phenomenon that the acceptable social distance between any group member and the target agent varies a lot due to “V”-shaped formation. Notably, every in-group agent is included by the Socialality kernel, as shown in (b6).

The capability of including more agents for larger groups does not necessarily mean the Socialality kernel tends to include an out-of-group agent more easily. In (a7) and (b7), the target agent moves in a small range and almost stays still in front of the zara store. It seems that this target agent is waiting for someone by itself. In (a7), the longterm distance kernel justifies a neighboring agent walking out of the store as a group member of the target agent. However, the Socialality kernel succeeds in excluding this closely passing-by agent, as shown in (b7). Visualizations in Fig. 5 demonstrate the Socialality kernel’s efectiveness of capturing the agent-specific and context-adaptive grouping rules.

Counterfactual Validations. We further conduct a series of counterfactual analyses to verify whether the grouping rules are learned and whether they indeed afect the final prediction results. First, we manually add neighboring agents [39] to examine how the grouping kernel reacts to diferent social relations in Fig. 5. Then, we randomly disturb the grouping structure by assigning diferent ratios of agents into the same group, and evaluate how the prediction performance changes under these counterfactual grouping settings. The performance results are reported in Table. 8 and visualized in Fig. 7. Finally, as shown in Fig. 8, we visualize the conditioned predictions to examine whether changes in grouping relations lead to corresponding changes in the predicted trajectories.

![](images/e9d5e8bee6f0954aa81a54991c2a2668038e53a6021a7bf60c6933e12490515f.jpg)  
Fig. 6 Examples of grouping decisions of GPCC and Socialality with manually added neighbors. Subfigures (a1)-(a8) and (b1)-(b8) visualize how GPCC and Socialality respond to manual neighbors placed at diferent relative positions under the same scenes and target agents. Subfigures (c1)-(c6) present representative near-boundary grouping cases, including joining group, staying in group, leaving group, and out of group.

(i) Manual-neighbor Intervention: We first conduct manual-neighbor interventions [39] to examine whether the group assignment responds to controlled local social contexts. Specifically, we manually add a neighboring agent around the target agent and change its relative position and motion pattern. As illustrated in eq. (11), the proposed Socialality conducts group decisions $\mathcal { G } _ { i }$ according to each neighboring agent’s motion patterns within the extended grouping window Ω<sup>˜</sup> . If the manually inserted neighbor under diferent social context settings is reasonably classified, it indicates that the grouping rule is learned.

We visualize several controlled local social contexts in Fig. 6. Fig. 6 (a1)-(a8) and (b1)-(b8) show cases where the manual neighbor is placed under diferent co-walking settings with the target agent. Additional examples in Fig. 6 (c1)-(c6) illustrate more subtle changes of group assignments under varying motion patterns.

By observing Fig. 6 (a1)-(a4), we find that GPCC may apply an overly strict criterion when determining whether a neighbor should be classified as an in-group agent. In (a2) and (a3), the manual neighbor is already placed relatively close to the target agent and exhibits a plausible co-walking pattern, but it is still excluded from the group assignment. Only when the closeness reaches the extreme case in (a4) does GPCC include it in the target agent’s group. This suggests that the fixed-rule kernel in GPCC relies on a strict grouping criterion, which is inconsistent with many realistic social scenarios. In contrast, Socialality produces group assignments that better match human intuition in Fig. 6 (b1)-(b4), indicating that the learned grouping rule can adapt its acceptance boundary according to the local social context rather than relying on a universal fixed threshold.

A complementary failure case of GPCC is shown in Fig. 6 (a5)-(a8). Here, the manual neighbor passes by the target agent within a short period and then diverges, i.e., they do not maintain a sustained co-walking relationship after the brief encounter. Nevertheless, GPCC may mis-classify such a short-time passing neighbor as an ingroup agent. By comparing Fig. 6 (b5)-(b8), we observe that Socialality largely avoids this kind of incorrect group assignment. Even when the manual neighbor comes close for a short time, Socialality is less likely to classify it as an in-group agent if the subsequent motion is inconsistent with that of the target agent. This is consistent with the design of Socialality, where group assignment is determined not only by spatial closeness but also by motion consistency within the extended grouping window, yielding more temporally coherent grouping outcomes.

We further present cases in Fig. 6 (c1)-(c6) to reveal how Socialality adjusts its group assignment under subtle changes of the manual neighbor. In (c1), a neighbor that is initially slightly far away becomes an in-group agent after moving in parallel with the target agent for a period (Joining Group), while (c2) shows a stable Staying in Group status when such co-walking persists. In contrast, (c3) shows an opposite transition. A neighbor that initially appears to walk together changes direction and leaves the group, and Socialality correspondingly excludes it from the target agent’s group assignment (Leaving Group). Finally, Fig. 6 (c4)-(c6) highlights an even more subtle phenomenon. When the target agent moves relatively slowly, its preference to in-group social distance decreases noticeably. As a result, the manual neighbor in (c5) is already classified as an out-of-group agent, and this tendency is further emphasized in (c6). This behavior is also consistent with real-world intuition that for slow pedestrians, being together often indicates a tighter spatial positions. Overall, these manual-neighbor interventions provide basic evidence that the proposed Socialality learns better grouping rules than the long-term distance kernel in GPCC , which relies only on simple spatial proximity.

(ii) Random-grouping Intervention: We further conduct random-grouping interventions to examine whether the final prediction is sensitive to the correctness of group assignments. Diferent from the manual-neighbor intervention, which modifies the local social context around a specific target agent, this intervention directly alters the grouping structure at the scene level. Specifically, we introduce a random grouping ratio $r _ { g }$ to control the proportion of agents that are randomly assigned into the target agent’s group in a specific scene. We sample $r _ { g }$ from 0 to 1 with a step size of 0.2, as reported in Table. 8. It should be noted that $r _ { g } = 0$ does not denote the original learned grouping strategy. Instead, it removes the grouping strategy by treating no neighboring agent as an in-group member of the target agent. In contrast, $r _ { g } = 1$ denotes an edge intervention where all neighboring agents are forced into the same group, regardless of their actual social relations.

Table 8 Prediction performance under diferent random grouping ratio $r _ { g }$ setting on ETH-UCY dataset
<table><tr><td>ID</td><td> $r _ { g }$ </td><td>eth</td><td>hotel</td><td>univ</td><td>zara1</td><td>zara2</td></tr><tr><td>a1</td><td>0</td><td>0.237/0.375</td><td>0.105/0.160</td><td>0.249/0.445</td><td>0.187/0.324</td><td>0.160/0.283</td></tr><tr><td>a2</td><td>0.2</td><td>0.236/0.375</td><td>0.104/0.159</td><td>0.248/0.441</td><td>0.183/0.315</td><td>0.137/0.225</td></tr><tr><td>a3</td><td>0.4</td><td>0.238/0.376</td><td>0.105/0.160</td><td>0.248/0.441</td><td>0.182/0.313</td><td>0.137/0.226</td></tr><tr><td>a4</td><td>0.6</td><td>0.237/0.374</td><td>0.105/0.162</td><td>0.247/0.440</td><td>0.181/0.31</td><td>0.135/0.223</td></tr><tr><td>a5</td><td>0.8</td><td>0.237/0.374</td><td>0.105/0.162</td><td>0.247/0.441</td><td>0.180/0.305</td><td>0.133/0.223</td></tr><tr><td>a6</td><td>1</td><td>0.254/0.414</td><td>0.128/0.221</td><td>0.283/0.499</td><td>0.195/0.344</td><td>0.223/0.372</td></tr><tr><td>a0</td><td></td><td>0.233/0.369</td><td>0.102/0.156</td><td>0.240/0.428</td><td>0.168/0.294</td><td>0.130/0.223</td></tr></table>

“-” denotes that no random grouping ratio is used in the corresponding full a0 model. Blue markers denote the best results on each set. Red markers denote the worst results on each set.

By observing Table. $^ { 8 , }$ we can see that manually intervening the grouping results noticeably afects the final prediction performance. The full model a0, which uses the original learned grouping strategy without random intervention, achieves the best performance on all five subsets. This indicates that the learned grouping rule provides more reliable group relations than randomly assigned grouping structures. Compared with a0, the $r _ { g } = 0 \mathrm { s e t t i n g } \mathrm { ( a 1 ) }$ prediction performance is worse about 7.4% and 8.0%. This suggests that removing group-aware modeling weakens the final prediction, since the target agent can no longer explicitly perceive its in-group members through the proposed group-wise perception mechanism.

The most severe degradation appears when $r _ { g } = 1$ . In this case, all neighboring agents are forced into one group, leading to the worst results on all subsets. Compared with $r _ { g } = 0$ , the average ADE/FDE further worsens 15.5% and 16.6%. Compared with the full model a0, the performance drop becomes even larger, with the average ADE/FDE worsening 24.1% and 25.8%, respectively. This result indicates that simply considering all neighboring agents as one group is harmful, probably because it largely removes the distinction between in-group and out-of-group interactions. As a result, the target agent may lose important perception patterns toward diferent neighboring agents.

For better illustration, we visualize the ADE changes under diferent $r _ { g }$ settings in Fig. 7. Across diferent subsets, we observe a common pattern that the performance first slightly improves when $r _ { g }$ increases from 0 to intermediate values, but then drops sharply when $r _ { g }$ reaches 1. A possible explanation is that, when no grouping is used $( r _ { g } = 0 )$ , the model ignores in-group interactions between the target agent and its true companions. As $r _ { g }$ gradually increases, some random group assignments may accidentally cover part of the true group relations, which slightly improves the prediction performance. However, these random assignments are still not as reliable as the learned grouping strategy in a0. When $r _ { g } = 1$ , the grouping structure collapses into one group, introducing strong counterfactual priors into the group-wise perception process and leading to clear performance degradation.

Observations  
![](images/a09478f531ef48c1b2e1ad839ad58501793548ec4ad7f7298023296db649e320.jpg)  
Fig. 7 ADE under diferent random grouping ratio $r _ { g }$ setting on ETH-UCY dataset.

Overall, the random-grouping intervention provides quantitative counterfactual evidence that the grouping relation is an efective condition for final trajectory prediction. Removing grouping or randomly forcing agents into groups both weakens the prediction performance, showing that Socialality does not benefit from arbitrary group assignments, but from structured grouping relations.

Target/In-group Agents  
Groundtruths  
Current Positions  
Neighbor Observations  
![](images/57979c57ef9834f62b0025072818d745c2c098d4892a85319717dc2379137509.jpg)  
Fig. 8 Visualizations of Socialality final predictions under diferent counterfactual settings.

(iii) Conditioned-prediction Intervention: We finally conduct conditionedprediction interventions to directly examine how diferent grouping priors afect the final predicted trajectories. Diferent from the above random-grouping intervention, which provides quantitative evidence through ADE/FDE comparisons, here we visualize the prediction results under diferent grouping settings for a more intuitive analysis. The corresponding group members are marked by yellow circles in Fig. 8.

Specifically, we present prediction results under three settings, including the original grouping strategy without intervention, the $r _ { g } = 0$ setting, and the $r _ { g } = 1$ setting. By comparing Fig. 8 (a1), (b1), and (c1), we observe that the same scene can produce substantially diferent final predictions under diferent grouping priors. In this four-agent group case, when the other in-group agents are no longer treated as group members under $r _ { g } = 0$ setting, the predicted trajectories become more concentrated. A similar phenomenon can also be observed from Fig. 8 (a2), (b2), and (c2), although the efect appears in an opposite form. Here, the predictions under $r _ { g } = 0$ become more diverse. This result further indicates that grouping priors do not impose a fixed change on the prediction distribution, but instead act as conditions that reshape the prediction according to the scene context. A possible explanation is that, when the target agent belongs to a group, its future motion is conditioned not only by its own motion pattern but also by the preference of the whole group. Therefore, the presence or absence of group members may lead to visibly diferent prediction distributions.

A particularly interesting case is shown in Fig. 8 (a3), (b3), and (c3). In this scene, the target agent is originally walking alone. Accordingly, when setting $r _ { g } = 0$ , the final predicted trajectories remain almost unchanged, which is consistent with our expectation since no efective group prior is removed. However, under $r _ { g } = 1$ , when another neighboring agent is forcibly treated as belonging to the same group, an additional prediction branch appears toward that neighbor. Notably, this new trajectory mode implies that the target agent may even make an almost reversed turn to approach the manually assigned group member.

Overall, these conditioned-prediction visualizations provide direct qualitative evidence that grouping priors serve as efective conditions for the Socialality Model.

Grouping Window. There are two main components within the proposed Socialality kernel, i.e., socialality anchors and the extended grouping window. After validating the capability of learning agent-specific and context adaptive grouping rules of the socialality anchors, we further analyze how time-coherent grouping decisions are made when relying on diferent grouping windows. As mentioned in section 3, Socialality jointly integrates retentions (historical reviews) and protentions (short-term future previews) corresponding to each target as grouping window. Fig. 9 illustrates five temporal segments: the retention-only window, three mixed windows, and the protention-only window. These segments represent diferent stages at which group membership may emerge, stabilize, or dissolve.

For the two-agent case in Fig. 9 (a1)-(a5), using retention alone (a1) and using limited retentions (a2) fail to include the companion of the target agent. Using protention alone (a5) also fails, because short-term predictions remain uncertain and do not yet reveal stable joint motion. Only when both temporal cues are combined (a3) and (a4) does the model correctly group the two agents. This indicates that group membership is not fully observable from either historical evidence or short-term anticipation alone.

![](images/f17bbb5de56ac5da99a45d6fcb8d8f5b8ccacbcb87b62bb808b00d44459ba256.jpg)  
Fig. 9 Visualizations of diferent grouping results under diferent grouping windows. Subfigures (a1)–(a5) and (b1)–(b5) show two representative cases, where the grouping windows are constructed entionwith diferent observation-prediction compositions, including {8, 0}, {6, 2}, {4, 4}, {2, 6}, {0, 8} observed steps and anticipated steps. Yellow shaded regions visualize the grouping windows of current in-group agents.

A similar phenomenon appears in the five-agent V-shaped group shown in Fig. 9 (b1)-(b5). Retention alone (b1) mainly captures nearby agents due to distance variation within the formation. Protention alone (b5) provides insuficient stability to recover the full structure. When retention and protention are integrated (b2)-(b4), both motion consistency and directional coherence are captured, enabling correct grouping of all members.

These observations also present a limitation of the original long-term distance kernel. Since grouping is determined only from past spatial proximity with a fixed threshold, the kernel tends to miss group members when historical distances are not consistently small, as shown in Fig. 9 (a1) and (b1). Conversely, relying on short-term future prediction only is also insuficient, because predicted trajectories alone are not stable enough to represent full group structures, as seen in Fig. 9 (a5) and (b5).

The proposed Socialality kernel resolves this issue by jointly using observations and anticipations. Historical trajectories provide reliable evidence of sustained interaction, while predicted trajectories reveal whether agents will continue moving together. Combining both cues allows the kernel to include true group members that are slightly separated in the past, while avoiding agents that are only temporarily close. Therefore, the advantage of Socialality kernel is not merely extending the observation window, but enabling grouping decisions to depend on both past proximity and future motion consistency. Such grouping results in Fig. 9 validate the Socialality kernel’s capability of learning temporally coherent grouping rules.

## 4.4.3 Discussions of Socialality Anchors

It should be noted that the learned socialality anchors are not designed to directly represent diferent existing groups. Instead, they represent agent-specific grouping preferences, i.e., how each target agent adjusts its tolerance to social distance and motion inconsistency when determining group membership. Therefore, the following analyses focus on whether the learned anchor space is structurally meaningful, and their statistical properties.

The original GPCC Model relies on a single fixed threshold Γ in the long-term distance kernel, implicitly assuming that a universal grouping rule can be applied to all agents and scenes. However, as discussed in section 3.2.2, grouping decisions are not governed by one variable only. They are jointly shaped by acceptable social distance and moving speed [13, 21]. Motivated by this, Socialality replaces the fixed Γ with two learnable socialality anchors, $\tau _ { i } ^ { a }$ and $\bar { \tau _ { i } ^ { b } }$ . These anchors serve as parametric containers for agent-specific grouping-rule modifiers. Therefore, each target agent i can be represented as a point $( \tau _ { i } ^ { a } , \tau _ { i } ^ { b } )$ in a 2D space.

![](images/667fb89f2015797efd40cdc156cbab98bf63bae11368833dcdbce1be16c1dba6.jpg)  
Fig. 10 Visualizations of grouping decisions and anchors. Counterfactual interventions of grouping decisions with manually added neighbors are also presented.

Grouping Rules of Anchors. We first visualize a representative scene from the zara1 subset in Fig. 10. After the group assignment of the proposed Socialality kernel, agents 0–4 are highlighted in the scene, where agent 0 denotes the target agent.

As mentioned in section 3 and eq. (7), we compute the socialality anchors of all agents in this scene through the network $m ( \cdot )$ , and visualize their anchor coordinates in a 2D scatter plot. We can observe that the anchor coordinates $( \tau _ { j } ^ { a } , \tau _ { j } ^ { b } ) , j \in$ {0, 1, 2, 3, 4}, corresponding to the group members assigned by Socialality kernel, are also close to each other in the anchor space. This indicates that the learned anchors indeed capture the grouping preference of agents, and agents within the same group tend to share similar grouping preferences. For clearer illustration, we roughly use a yellow mask centered at the anchor coordinate of the target agent, i.e., agent 0, to cover the anchors of agents with the same group membership.

Through this visualization, we can also observe that one blue point close to the target agent in the anchor space does not correspond to an actual group member in the original zara1 scene. However, by checking their observed trajectories, we find that this agent still shows a motion pattern similar to the target agent, and may therefore share a similar grouping preference. This observation suggests that the relationship between anchors and group membership is asymmetric. Agents in the same group are likely to have nearby anchors, but nearby anchors do not necessarily imply that the corresponding agents belong to the same group. In other words, the anchor space represents grouping-rule preferences rather than direct group labels.

We further visualize the grouping results after adding diferentiated manual neighbors, together with the corresponding anchor of the manual neighbor and the original anchors. Similar to Fig. 6, we adopt four strategies to add manual neighbors, corresponding to typical group dynamics, including joining a group, staying in a group, leaving a group, and remaining out of a group. By comparing interventions #1 and #2 in Fig. 10, we can observe that the manual neighbor in #2, which consistently stays within the group of target agent 0, has an anchor coordinate closer to agent 0 than the manual neighbor in #1.

Intervention #3 visualizes a manual neighbor that gradually moves away from the target agent. This manual neighbor may be grouped with the target agent during an earlier period, but then goes separate ways after a certain point. Correspondingly, its anchor is no longer located inside the yellow masked region, but still remains relatively closer to the target agent than a completely irrelevant neighbor. Intervention #4 visualizes a manual neighbor that is clearly unrelated to target agent 0, with distinct diferences in moving direction, speed, and spatial relation. By comparing the anchor coordinates of the manual neighbors in #3 and $\# 4 ,$ we can observe that the manual neighbor in #4 is located in a clearly diferent region of the anchor space, far from the target agent. This further indicates that the anchor distance reflects the similarity of grouping preferences among diference agents.

Distribution of Anchors. Fig. 11 visualizes the anchor distributions on multiple subsets of ETH-UCY and SDD. A first notable phenomenon is that anchors do not spread uniformly. Instead, they concentrate along several narrow “wings” (thinmanifold shape), and the patterns could be observed to be varying with diferent datasets. For example, eth exhibits a compact diagonal wing where $\tau ^ { a }$ and $\tau ^ { b }$ increase coherently, suggesting a correlation between distance tolerance and speed flexibility across agents. In contrast, zara1 forms two separated wings, indicating that the model might prefer two regimes of grouping rules, which might share common tendency since two separated wings are spatially close and almost parallel. Within SDD, the structures become more diverse. For example, coupa0 shows a curved wing (a “C”-shape in 2D space), hyang1 displays multiple wings with a noticeable downward tail, and bookstore3 presents a multi-wing pattern where wings seem to grow from a dense core. Such wing-shaped concentrations imply that the anchors are not arbitrary free parameters; rather, they converge according to specific grouping-rule to shape diferent patterns.

Second, the number of wings vary substantially across scenes. ETH-UCY subsets tend to present one or two dominant wings (e.g., the single compact wing in eth and the two-wing structure in zara1), whereas SDD subsets often exhibit richer multi-wing structures (e.g., hyang1 and bookstore3). This diference suggests that the learned anchors capture scene-specific mixtures of motion modes. Specifically, fewer wings indicate more similar grouping behaviors, while multiple wings indicate that agents in the scene may follow multiple distinct movement and interaction styles, which demand diferent distance-speed anchor combination to yield stable grouping.

![](images/d6f97a6f5e614dcdff79d11f2b6d69fa1235fe54a2c9e38e91b137f1cd4e3f12.jpg)

![](images/496986a4ffbe10aa1ab98acf74876dc6dea5895d37df0a8511554694215f4d7c.jpg)

![](images/3ba547a4d0b09d1b600a761049c3ee5bf9129124aabc318d7d08824700eed6f5.jpg)

![](images/f8e9b051e5fd65677795cbd2d1798d840206b2068a3e47eea000dc4c8fba2e9d.jpg)

![](images/8909669309bdabb046575e82bea45a4ea510d20dd25f631a6ff847499a71f919.jpg)

![](images/c477040c8fff1f53904d8cb79363fef2734afcda408f4ab81d62c47587d0cfc9.jpg)  
Fig. 11 Visualization of socialality anchors in a 2D space across diferent scenes. Each subfigure shows the distribution of socialality anchors parameterized by $( \tau ^ { a } , \tau ^ { b } )$ for one scene. Each blue dot denotes one agent’s anchor sample, while the orange region indicate their density distribution.

Overall, the anchor distribution provides an interpretable illustration of how Socialality learns agent-specific and context-adaptive grouping rules, suggesting that human grouping preferences in social scenes can be parameterized by the two anchors that jointly regulate distance tolerance and speed flexibility in a 2D space.

Structural Analyses of Anchors. To connect the anchor distribution to physical behaviors, we map representative anchor locations back to the scene in Fig. 12. Here, we take socialality anchor distribution in zara1 as example. The anchor distribution forms a clear V-shape with two wings. We select several targets agent corresponding to the anchor coordinates on diferent parts of the V-shape pattern and visualize these target agents and their neighbors’ observed trajectories.

In Fig. 12, we can observe that targets (7, 8, and 9 in the right subfigure) consistently move to the right, while targets (1, 2, and 3 in the left subfigure) consistently move to the left. In contrast, samples (4, 5, and 6 in the middle subfigure) exhibit a slow-motion pattern. Both target agents (4) and (6) remain nearly stationary during the observation window. This separation emerges without any motion-mode supervision, since the anchors only afect grouping through the Socialality kernel.

![](images/255d00bb93bda688a3e0757e4d3212478b0caf6b82e9cd75558d0ed7e1f384c3.jpg)  
Fig. 12 Structural visualization of socialality anchors distributions evaluated on ETH-UCY zara1. Diferent colors indicate diferent regions in the same distribution figure. We choose three representative examples from each region here. Best viewed in color.

Within each wing of this V-shape pattern, we observe an ordering related to walking speed. For the right-moving-agent wing, the target agents (7) and (8) move faster than agent (9). For the left-moving-agent wing, the target agents (1) and (2) move faster than that of agent (3). In both cases, faster examples, i.e., (1), (2), (7), and (8), lie closer to the wing boundary, while slower examples, i.e., (3) and (9), lie closer to the base of the wing. For near-stationary targets (4), (5), and (6) in the middle subfigure, the displacement is very small. Accordingly, these target agents’ anchor positions occupy the bottom region of the V-shape. A plausible explanation is that higher speed increases neighborhood variability and short-lived encounters, requiring a diferent balance between distance tolerance $\left( \tau _ { i } ^ { a } \right)$ and speed-flexibility tolerance $( \tau _ { i } ^ { b } )$ , which shifts the anchor position along the wing.

The anchor distributions in the other two subsets in Fig. 13 reveal that such structural relations with motion modes is not unique to zara1. In univ (b), the anchor distribution is more dispersed and continuous, suggesting a more diverse mixture of local interaction regimes. Nevertheless, nearby anchor regions still correspond to target agents with similar motion tendencies and neighborhood configurations, indicating that local smoothness in anchor space is preserved. In SDD subset little0 (a), the anchor distribution exhibits a multi-wing structure rather than a simple two-wing pattern, implying that the learned anchors adapt to richer motion modes in that scene. Even in this more complex case, diferent wings remain associated with visually distinguishable motion behaviors as demonstrated in Fig. 13.

Overall, Fig. 12 and Fig. 13 provides direct structural evidence that the learned anchor space is behaviorally meaningful. This suggests that the two-anchor design ofers a compact yet flexible representation for agent-specific and context-adaptive grouping rules in real scenes.

Statistical Analyses of Anchors. In this section, we analyze the statistical and geometric structure of the learned socialality anchors. First, we compute the mean, variance, covariance, and correlation coeficient. Given $N _ { b }$ samples $\{ ( \tau _ { i } ^ { a } , \tau _ { i } ^ { b } ) \} _ { i = 1 } ^ { N _ { b } }$ , the

![](images/f7eb7f975fed2c5634ca86f8cae297a4bdcb9c3cd00fab5b02046e0b81cf24ea.jpg)  
Fig. 13 Structural visualization of socialality anchors distributions evaluated on SDD little0 (a) and ETH-UCY univ (b). Diferent colors indicate diferent regions in distribution figures. We choose several representative examples from each region here. Best viewed in color.

covariance is defined as

$$
\operatorname { C o v } ( \tau ^ { a } , \tau ^ { b } ) = \frac { 1 } { N _ { b } } \sum _ { i = 1 } ^ { N _ { b } } \left( \tau _ { i } ^ { a } - \mu _ { a } \right) \left( \tau _ { i } ^ { b } - \mu _ { b } \right) ,\tag{31}
$$

where $\mu _ { a }$ and $\mu _ { b }$ denote the means of $\tau ^ { a }$ and $\tau ^ { b } .$ , respectively. The corresponding correlation $\rho _ { a b }$ is calculated as

$$
\rho _ { a b } = \frac { \mathrm { C o v } ( \tau ^ { a } , \tau ^ { b } ) } { \sqrt { \mathrm { V a r } ( \tau ^ { a } ) \mathrm { V a r } ( \tau ^ { b } ) } } .\tag{32}
$$

Table. 9 lists correlation $\rho _ { a b }$ across diferent clips and training runs. Nonindependence implies that the correlation coeficient is not zero, while non-perfect correlation implies that the correlation coeficient is not equal to 1 or −1. From Table. 9, we can infer that the experimental results consistently bounded away from 0 and 1, which indicates that the two diferent anchors are neither completely independent nor perfectly correlated. From another perspective, this also supports our conclusion in the ablation study in Table. 5 that a fixed anchor (distance in GPCC) is insuficient to describe the space where grouping rules reside. The butterfly-like multi-wing shape in Fig. 14 and Fig. 11, where the manifold is neither a linear straight line nor randomly occupies the entire 2D space, further illustrates this phenomenon, i.e., the distribution lies on a specific butterfly-shaped manifold. This suggests that these two anchors are already capable of representing grouping rules for diferent agents, rather than requiring three or more anchors. In other words, additional anchors would be redundant. Introducing a third anchor would increase parameterization without introducing new statistically supported degrees of freedom.

Table 9 The correlation $\rho _ { a b }$ across diferent subsets and training runs.
<table><tr><td>clip</td><td>hyang0</td><td></td><td>hyang12</td><td>nexus11</td><td>nexus5</td><td>nexus10</td></tr><tr><td> $\rho _ { a b }$ </td><td>(train#1)</td><td>0.268</td><td>0.169</td><td>0.288</td><td>0.839</td><td>0.168</td></tr><tr><td></td><td>ρab (train#2)</td><td>0.435</td><td>-0.810</td><td>0.862</td><td>0.608</td><td>0.118</td></tr><tr><td> $\rho _ { a b }$ </td><td>(train#3)</td><td>0.518</td><td>0.465</td><td>0.356</td><td>0.839</td><td>0.309</td></tr><tr><td>clip</td><td></td><td>coupa3</td><td>bookstore0</td><td>gates0</td><td>deathCircle0</td><td>little0</td></tr><tr><td> $\rho _ { a b }$ </td><td>(train#1)</td><td>0.277</td><td>0.527</td><td>0.243</td><td>0.637</td><td>0.703</td></tr><tr><td> $\rho _ { a b }$ </td><td>(train#2)</td><td>-0.566</td><td>-0.489</td><td>0.642</td><td>0.548</td><td>-0.259</td></tr><tr><td> $\rho _ { a b }$ </td><td>(train#3)</td><td>-0.030</td><td>0.273</td><td>0.2436</td><td>0.638</td><td>0.703</td></tr></table>

To further understand the intrinsic dimensionality of the anchor space, we consider the covariance matrix

$$
\Sigma = \binom { \mathrm { V a r } ( \tau ^ { a } ) } { \mathrm { C o v } ( \tau ^ { a } , \tau ^ { b } ) } \stackrel { \mathrm { C o v } ( \tau ^ { a } , \tau ^ { b } ) } { \mathrm { V a r } ( \tau ^ { b } ) } \biggr ) .\tag{33}
$$

This suggests that the anchor distribution lies close to a low-dimensional manifold embedded in $\mathbb { R } ^ { 2 }$

Furthermore, from the polar coordinates perspective, in Fig. 11, we observe that the angular variable θ concentrates within a small angular range, while the radial magnitude r varies more freely. This separation between direction and magnitude indicates that the system primarily modulates along a dominant direction, with limited orthogonal variation.

The above findings indicate that the learned socialality representation occupies an intermediate regime: it is neither purely one-dimensional nor genuinely twodimensional. For intuitive illustration, we refer to this as “1.5D”. It should be noted that this terminology is descriptive rather than formal, and does not imply a strict fractal or Hausdorf dimension [72]. These phenomena further indicate that human spatial social activities might be closely related to the two factors corresponding to these two socialality anchors, distance and speed, and may also provide new attempts for studying human behaviors or inspiring other human-inspired research.

After establishing the statistical structure of the learned anchors, we further examine whether their geometric organization is stable across independent training runs. Here, we repeatedly train Socialality Model on the same dataset with random initializations, and visualize the resulting anchor distributions in Fig. 14.

We observe that although the exact patterns vary from run to run, the learned anchor distributions are not arbitrary. Such variations mainly afect the fine-grained appearance of the distributions, rather than their underlying structural information. Each dataset tends to repeatedly exhibit a consistent geometric pattern. For example, on zara1, the anchors consistently form two-wing-like structures like a letter $^ { 6 6 } \mathrm { V } ^ { 5 }$ on hotel, they remain concentrated within a relatively compact region with limited directional spread; and on zara2, they repeatedly organize into bent shapes, similar to zara1. Notably, the socialality anchors are learned without explicit geometric supervision. These results support the stabilizability of the proposed Socialality kernel. The anchor distributions are not fragile artifacts caused by a particular random seed, but reproducible structures that repeatedly emerge under the same dataset. This indicates that Socialality kernel can consistently capture intrinsic socialality patterns from data, thereby providing a stable basis for the Socialality Model.

![](images/928fc2ca510978c63f174d4aaed8cadeab44fb46889b93665a29f42091ea3387.jpg)  
Fig. 14 Anchor distributions learned from diferent independent runs. Each row corresponds to one scene (zara1, hotel, and zara2), while each subfigure shows the anchor distribution density.

Overall, these analyses suggest that the learned socialality anchors provide a compact and interpretable parameterization of agent-specific and context-adaptive grouping preferences. The anchor space captures how diferent agents modulate acceptable social distance and in-group common speed under diferent social contexts. The structured distributions and patterns across independent training runs together support the interpretability and stability of the proposed socialality anchor design.

## 5 Conclusion and Limitations

In this manuscript, we propose the Socialality kernel and the corresponding Socialality trajectory prediction model to further expand GPCC [18] by explicitly obtaining diferent agents’ agent-specific, temporally coherent, and context-adaptive social boundaries for grouping when forecasting trajectories. The Socialality kernel uses two learnable coeficients $( \tau _ { i } ^ { a } , \tau _ { i } ^ { b } )$ , i.e., socialality anchors, to jointly modulate acceptable social distance and speed flexibility within each group over an extended grouping window. The boundaries anchored by the socialality anchors determines the group afiliation, which serves as the group priors conditioned to interaction modeling in the subsequent perception mechanism. We validate the Socialality Model on standard trajectory prediction benchmarks and observe consistent improvements over strong baselines, indicating that the proposed grouping mechanism yields measurable performance gains. We further assess the proposed socialality anchors from an interpretability and robustness perspective. Through qualitative visualizations, the learned anchors exhibit coherent and scene-consistent structures that align with recognizable diferent motion modes. These analyses jointly validate the interpretability, stability, and practical efectiveness of the learned socialality anchors. The analyses of two socialality anchors indicate that human spatial social activities might be related to the two factors corresponding to these two socialality anchors, distance and speed, and may also provide new attempts for studying human behaviors or inspiring other human-inspired research.

Despite these promising results, our framework has certain technical limitations. Based on the current two-stage network structure, the proposed Socialality aims to obtain the group priors first before interaction modeling. This sequential paradigm of grouping before modeling efectively leverages the obtained group priors in pedestriandominated scenes. However, the impact of such group priors might be less evident in scenes where the primary agents are vehicles. This suggests that the proposed method may exhibit certain limitations when generalized to vehicle trajectory prediction datasets. Accordingly, exploring and designing a more universal prior to accommodate complex scenes in a human-inspired manner will be one of our future considerations.

## Declarations

• Conflict of interest statement: The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this manuscript.

• Funding Information: This work is supported by the National Natural Science Foundation of China (Grant 62172177).

• Data availability: There is no additional data associated with this manuscript. Our code is available at https://github.com/LivepoolQ/Socialality.

## References

[1] Alahi, A., Goel, K., Ramanathan, V., Robicquet, A., Fei-Fei, L., Savarese, S.: Social lstm: Human trajectory prediction in crowded spaces. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 961–971 (2016)

[2] Alahi, A., Ramanathan, V., Goel, K., Robicquet, A., Sadeghian, A.A., Fei-Fei, L., Savarese, S.: Learning to predict human behavior in crowded scenes. In: Group and Crowd Behavior for Computer Vision, pp. 183–207. Elsevier, Amsterdam (2017)

[3] Chai, Y., Sapp, B., Bansal, M., Anguelov, D.: Multipath: Multiple probabilistic anchor trajectory hypotheses for behavior prediction. arXiv preprint arXiv:1910.05449 (2019)

[4] Lee, N., Choi, W., Vernaza, P., Choy, C.B., Torr, P.H., Chandraker, M.: Desire: Distant future prediction in dynamic scenes with interacting agents. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 336–345 (2017)

[5] Chen, Y., Ivanovic, B., Pavone, M.: Scept: Scene-consistent, policy-based trajectory predictions for planning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 17103–17112 (2022)

[6] Pellegrini, S., Ess, A., Schindler, K., Van Gool, L.: You’ll never walk alone: Modeling social behavior for multi-target tracking. In: 2009 IEEE 12th International Conference on Computer Vision, pp. 261–268 (2009). IEEE

[7] Saleh, F., Aliakbarian, S., Salzmann, M., Gould, S.: Artist: Autoregressive trajectory inpainting and scoring for tracking. arXiv preprint arXiv:2004.07482 (2020)

[8] Gupta, A., Johnson, J., Fei-Fei, L., Savarese, S., Alahi, A.: Social gan: Socially acceptable trajectories with generative adversarial networks. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 2255– 2264 (2018)

[9] Moussa¨ıd, M., Perozo, N., Garnier, S., Helbing, D., Theraulaz, G.: The walking behaviour of pedestrian social groups and its impact on crowd dynamics. PloS one 5(4), 10047 (2010)

[10] Qiu, F., Hu, X.: Modeling group structures in pedestrian crowd simulation. Simulation Modelling Practice and Theory 18(2), 190–205 (2010)

[11] Bae, I., Park, J.-H., Jeon, H.-G.: Learning pedestrian group representations for multi-modal trajectory prediction. In: European Conference on Computer Vision, pp. 270–289 (2022). Springer

[12] Lamont, M., Moln´ar, V.: The study of boundaries in the social sciences. Annual review of sociology 28(1), 167–195 (2002)

[13] Hall, E.T.: The hidden dimension. Leonardo 6(1), 94 (1973)

[14] Sorokowska, A., Sorokowski, P., Hilpert, P., Cantarero, K., Frackowiak, T., Ahmadi, K., Alghraibeh, A.M., Aryeetey, R., Bertoni, A., Bettache, K., et al.: Preferred interpersonal distances: A global comparison. Journal of cross-cultural psychology 48(4), 577–592 (2017)

[15] Xu, C., Li, M., Ni, Z., Zhang, Y., Chen, S.: Groupnet: Multiscale hypergraph neural networks for trajectory prediction with relational reasoning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6498–6507 (2022)

[16] Yamaguchi, K., Berg, A.C., Ortiz, L.E., Berg, T.L.: Who are you with and where are you going? In: CVPR 2011, pp. 1345–1352 (2011). IEEE

[17] Solera, F., Calderara, S., Cucchiara, R.: Socially constrained structural learning for groups detection in crowd. IEEE transactions on pattern analysis and machine intelligence 38(5), 995–1008 (2015)

[18] Zou, Z., Wong, C., Xia, B., You, X.: Who walks with you matters: Perceiving social interactions with groups for pedestrian trajectory prediction. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4844–4853 (2025)

[19] Campbell, D.T.: Common fate, similarity, and other indices of the status of aggregates of persons as social entities. Behavioral science 3(1), 14 (1958)

[20] Husserl, E.: On the Phenomenology of the Consciousness of Internal Time (1893– 1917) vol. 4. Springer, Dordrecht (2012)

[21] Helbing, D., Molnar, P.: Social force model for pedestrian dynamics. Physical review E 51(5), 4282 (1995)

[22] Fiorini, P., Shiller, Z.: Motion planning in dynamic environments using velocity obstacles. The international journal of robotics research 17(7), 760–772 (1998)

[23] Berg, J., Lin, M., Manocha, D.: Reciprocal velocity obstacles for real-time multiagent navigation. In: 2008 IEEE International Conference on Robotics and Automation, pp. 1928–1935 (2008). Ieee

[24] Kim, B., Kang, C.M., Kim, J., Lee, S.H., Chung, C.C., Choi, J.W.: Probabilistic vehicle trajectory prediction over occupancy grid map via recurrent neural network. In: 2017 IEEE 20th International Conference on Intelligent Transportation Systems (ITSC), pp. 399–404 (2017). IEEE

[25] Sun, J., Jiang, Q., Lu, C.: Recursive social behavior graph for trajectory prediction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 660–669 (2020)

[26] Zhang, P., Ouyang, W., Zhang, P., Xue, J., Zheng, N.: Sr-lstm: State refinement for lstm towards pedestrian trajectory prediction. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 12085–12094 (2019)

[27] Zhang, P., Xue, J., Zhang, P., Zheng, N., Ouyang, W.: Social-aware pedestrian trajectory prediction via states refinement lstm. IEEE transactions on pattern analysis and machine intelligence 44(5), 2742–2759 (2022)

[28] Deo, N., Trivedi, M.M.: Convolutional social pooling for vehicle trajectory prediction. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pp. 1468–1476 (2018)

[29] Pei, Z., Qi, X., Zhang, Y., Ma, M., Yang, Y.-H.: Human trajectory prediction in crowded scene using social-afinity long short-term memory. Pattern Recognition 93, 273–282 (2019) https://doi.org/10.1016/j.patcog.2019.04.025

[30] Liu, Z., He, L., Yuan, L., Lv, K., Zhong, R., Chen, Y.: Stagp: Spatio-temporal adaptive graph pooling network for pedestrian trajectory prediction. IEEE Robotics and Automation Letters (2023)

[31] Vemula, A., Muelling, K., Oh, J.: Social attention: Modeling attention in human crowds. In: 2018 IEEE International Conference on Robotics and Automation (ICRA), pp. 1–7 (2018). IEEE

[32] Fernando, T., Denman, S., Sridharan, S., Fookes, C.: Soft+ hardwired attention: An lstm framework for human trajectory prediction and abnormal event detection. Neural networks 108, 466–478 (2018)

[33] Yuan, Y., Weng, X., Ou, Y., Kitani, K.M.: Agentformer: Agent-aware transformers for socio-temporal multi-agent forecasting. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 9813–9823 (2021)

[34] Zhao, H., Xu, W., Monfort, M., Choi, J., Baker, C., Zhao, Y., Wu, H., Torralba, A., Tenenbaum, J.B., Wu, B.: Tnt: Target-driven trajectory prediction. In: Advances in Neural Information Processing Systems (NeurIPS) (2020)

[35] Ivanovic, B., Pavone, M.: The trajectron: Probabilistic multi-agent trajectory modeling with dynamic spatiotemporal graphs. In: Proceedings of the IEEE International Conference on Computer Vision, pp. 2375–2384 (2019)

[36] Cao, D., Wang, Y., Duan, J., Zhang, C., Zhu, X., Huang, C., Tong, Y., Xu, B., Bai, J., Tong, J., et al.: Spectral temporal graph neural network for multivariate time-series forecasting. Advances in Neural Information Processing Systems 33, 17766–17778 (2020)

[37] Mohamed, A., Qian, K., Elhoseiny, M., Claudel, C.: Social-stgcnn: A social spatiotemporal graph convolutional neural network for human trajectory prediction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14424–14432 (2020)

[38] Wong, C., Xia, B., Zou, Z., Wang, Y., You, X.: Socialcircle: Learning the anglebased social interaction representation for pedestrian trajectory prediction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 19005–19015 (2024)

[39] Wong, C., Xia, B., Zou, Z., You, X.: Socialcircle+: Learning the angle-based conditioned interaction representation for pedestrian trajectory prediction. arXiv preprint arXiv:2409.14984 (2024)

[40] Bae, I., Lee, J., Jeon, H.-G.: Social reasoning-aware trajectory prediction via multimodal language model. IEEE Transactions on Pattern Analysis and Machine Intelligence (2025)

[41] Bae, I., Lee, J., Jeon, H.-G.: Can language beat numerical regression? languagebased multimodal trajectory prediction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 753–766 (2024)

[42] Wong, C., Zou, Z., Xia, B.: Resonance: Learning to predict social-aware pedestrian trajectories as co-vibrations. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 25788–25799 (2025)

[43] Jiang, J., Yan, K., Xia, X., Yang, B.: A survey of deep learning-based pedestrian trajectory prediction: Challenges and solutions. Sensors 25(3), 957 (2025)

[44] Sadeghian, A., Kosaraju, V., Sadeghian, A., Hirose, N., Rezatofighi, H., Savarese, S.: Sophie: An attentive gan for predicting paths compliant to social and physical constraints. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 1349–1358 (2019)

[45] Hasan, I., Setti, F., Tsesmelis, T., Del Bue, A., Cristani, M., Galasso, F.: ”seeing is believing”: Pedestrian trajectory forecasting using visual frustum of attention. In: 2018 IEEE Winter Conference on Applications of Computer Vision (WACV), pp. 1178–1185 (2018). IEEE

[46] Liao, H., Liu, S., Li, Y., Li, Z., Wang, C., Li, Y., Li, S.E., Xu, C.: Human observation-inspired trajectory prediction for autonomous driving in mixedautonomy trafic environments. In: 2024 IEEE International Conference on Robotics and Automation (ICRA), pp. 14212–14219 (2024). IEEE

[47] Rehder, E., Kloeden, H.: Goal-directed pedestrian prediction. In: Proceedings of the IEEE International Conference on Computer Vision Workshops, pp. 50–58 (2015)

[48] Guo, H., Liu, Y., Meng, Q., Li, J., Chen, H.: Goal-oriented pedestrian trajectory prediction considering spatial-temporal interactions. IEEE Transactions on Instrumentation and Measurement (2024)

[49] Wang, C., Wang, Y., Xu, M., Crandall, D.J.: Stepwise goal-driven networks for trajectory prediction. IEEE Robotics and Automation Letters 7(2), 2716–2723 (2022)

[50] Chiara, L.F., Coscia, P., Das, S., Calderara, S., Cucchiara, R., Ballan, L.: Goaldriven self-attentive recurrent networks for trajectory prediction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2518–2527 (2022)

[51] Mangalam, K., An, Y., Girase, H., Malik, J.: From goals, waypoints & paths to long term human trajectory forecasting. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 15233–15242 (2021)

[52] Patla, A.E., Prentice, S.D., Robinson, C., Neufeld, J.: Visual control of locomotion: strategies for changing direction and for going over obstacles. Journal of Experimental Psychology: Human Perception and Performance 17(3), 603 (1991)

[53] Wolfe, J.M., Cave, K.R., Franzel, S.L.: Guided search: an alternative to the feature integration model for visual search. Journal of Experimental Psychology: Human perception and performance 15(3), 419 (1989)

[54] Cutting, J.E., Vishton, P.M., Braren, P.A.: How we avoid collisions with stationary and moving objects. Psychological review 102(4), 627 (1995)

[55] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I.: Attention is all you need. In: Advances in Neural Information Processing Systems, pp. 5998–6008 (2017)

[56] Wong, C., Xia, B., Peng, Q., Yuan, W., You, X.: Msn: multi-style network for trajectory prediction. IEEE Transactions on Intelligent Transportation Systems 24, 9751–9766 (2023)

[57] Lerner, A., Chrysanthou, Y., Lischinski, D.: Crowds by example. Computer Graphics Forum 26(3), 655–664 (2007)

[58] Robicquet, A., Sadeghian, A., Alahi, A., Savarese, S.: Learning social etiquette: Human trajectory understanding in crowded scenes. In: European Conference on Computer Vision, pp. 549–565 (2016). Springer

[59] Liang, J., Jiang, L., Hauptmann, A.: Simaug: Learning robust representations from simulation for trajectory prediction. In: Proceedings of the European Conference on Computer Vision (ECCV) (2020)

[60] Wang, D., Liu, H., Wang, N., Wang, Y., Wang, H., Mcloone, S.: Seem: a sequence entropy energy-based model for pedestrian trajectory all-then-one prediction. IEEE transactions on pattern analysis and machine intelligence 45(1), 1070–1086 (2023)

[61] Xu, C., Tan, R.T., Tan, Y., Chen, S., Wang, Y.G., Wang, X., Wang, Y.: Eqmotion: Equivariant multi-agent motion prediction with invariant interaction reasoning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1410–1420 (2023)

[62] Kim, S., Chi, H.-g., Lim, H., Ramani, K., Kim, J., Kim, S.: Higher-order relational reasoning for pedestrian trajectory prediction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15251–15260 (2024)

[63] Chib, P.S., Singh, P.: Lg-traj: Llm guided pedestrian trajectory prediction. arXiv preprint arXiv:2403.08032 (2024)

[64] Liu, Y., Ye, Z., Wang, R., Li, B., Sheng, Q.Z., Yao, L.: Uncertainty-aware pedestrian trajectory prediction via distributional difusion. Knowledge-Based Systems, 111862 (2024)

[65] Xia, B., Wong, C., Xu, D., Peng, Q., You, X.: Another vertical view: A hierarchical network for heterogeneous trajectory prediction via spectrums. IEEE Transactions on Pattern Analysis and Machine Intelligence (2025)

[66] Cao, D., Li, J., Ma, H., Tomizuka, M.: Spectral temporal graph neural network for trajectory prediction. In: 2021 IEEE International Conference on Robotics and Automation (ICRA), pp. 1839–1845 (2021). IEEE

[67] Shi, L., Wang, L., Long, C., Zhou, S., Tang, W., Zheng, N., Hua, G.: Representing multimodal behaviors with mean location for pedestrian trajectory prediction. IEEE Transactions on Pattern Analysis and Machine Intelligence (2023)

[68] Dong, Y., Wang, L., Zhou, S., Hua, G., Sun, C.: Recurrent aligned network for generalized pedestrian trajectory prediction. arXiv preprint arXiv:2403.05810 (2024)

[69] Yue, J., Manocha, D., Wang, H.: Human trajectory prediction via neural social physics. In: European Conference on Computer Vision, pp. 376–394 (2022). Springer

[70] Lee, M., Sohn, S.S., Moon, S., Yoon, S., Kapadia, M., Pavlovic, V.: Musevae: Multi-scale vae for environment-aware long term trajectory prediction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2221–2230 (2022)

[71] Maeda, T., Ukita, N.: Fast inference and update of probabilistic density estimation on trajectory prediction. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 9795–9805 (2023)

[72] Falconer, K.: Fractal Geometry: Mathematical Foundations and Applications.

## Appendix A Nomenclature Table

To assist readers in better understanding our formulations, particularly the temporal slicing operations required in our framework, we provide a nomenclature table here.

Table A1 Nomenclature of the key variables in the proposed methodology
<table><tr><td>Symbol</td><td>Description</td></tr><tr><td colspan="2">General</td></tr><tr><td> $N _ { e }$ </td><td>Total number of agents in the scene</td></tr><tr><td> $i , j$ </td><td>Indices for the target agent and its neighboring agent  $t = 0 )$ </td></tr><tr><td>t</td><td>Time step (current observation step is set to</td></tr><tr><td> $\epsilon$ </td><td>An infinitesimal quantity</td></tr><tr><td> $K _ { f } , K _ { g }$ </td><td>Trajectory generation number for main and short-term prediction networks</td></tr><tr><td colspan="2">Time Windows</td></tr><tr><td> $t _ { o } , t _ { p }$ </td><td>Number of discrete observation and prediction time steps</td></tr><tr><td> $\Omega _ { 0 }$ </td><td>Observation time window  $\{ - t _ { o } + 1 , \ldots , - 1 , 0 \}$ </td></tr><tr><td> $\hat { \Omega }$ </td><td>Anticipation (short-term future preview) time window</td></tr><tr><td> $\tilde { \Omega }$ </td><td>Extended temporal grouping window  $( \tilde { \Omega } = \Omega \cup \hat { \Omega } )$ </td></tr><tr><td colspan="2">Trajectories</td></tr><tr><td> $\mathbf { p } _ { i } ^ { t } \in \mathbb { R } ^ { 2 }$ </td><td>2D coordinate of agent i at time step t</td></tr><tr><td> $\mathbf { d } _ { i }$ </td><td>Heading direction of agent i</td></tr><tr><td> $\mathbf { X } _ { i } , \mathbf { Y } _ { i }$ </td><td>Ground-truth observed and future trajectories of agent i</td></tr><tr><td> $\hat { \mathbf { Y } } _ { i }$ </td><td>Predicted multi-styled future trajectories</td></tr><tr><td> $d _ { i } ^ { t } ( j )$ </td><td>Step-wise distance between agent  $j$  and i at time t</td></tr><tr><td> $p _ { i } ( \tilde { \Omega } )$ </td><td>Displacement of agent i during time window Ω</td></tr><tr><td> $\rho _ { i } ( j | \tilde { \Omega } )$ </td><td>Walking speed difference ratio between agent j and i</td></tr><tr><td colspan="2">Groups and Perceptual Sets</td></tr><tr><td> ${ \mathcal { N } } _ { i }$ </td><td>Set of all neighboring agents (group candidates) for target agent i</td></tr><tr><td> $\mathcal { G } _ { i } , \overline { { \mathcal { G } } } _ { i }$   $\mathbf { F O V } _ { i }$ </td><td>Sets of in-group agents and out-of-group agents for agent i</td></tr><tr><td> ${ \mathcal { F } } _ { i }$ </td><td>Field of view angular region for agent i</td></tr><tr><td></td><td>Set of in-FOV neighboring agents for agent i</td></tr><tr><td> $\mathcal { C } _ { i } ^ { s }$ </td><td>Sets of out-of-group agents in region  $s \in \{ \mathrm { r i g h t } , \mathrm { l e f t } , \mathrm { r e a r } \}$ </td></tr><tr><td colspan="2">Grouping Kernels</td></tr><tr><td> $\tau _ { i } ^ { a } , \tau _ { i } ^ { b }$ </td><td>Socialality anchors modulating acceptable social distance and speed difference</td></tr><tr><td> $\Gamma$ </td><td>Static distance threshold in the long-term distance kernel</td></tr><tr><td> $\kappa _ { i }$ </td><td>Grouping kernel function</td></tr><tr><td> $S _ { i } ^ { a } , S _ { i } ^ { b }$ </td><td>Grouping condition indicators modulated by anchors  $\tau _ { i } ^ { a }$  and</td></tr><tr><td> $c _ { 1 } , c _ { 2 } , c _ { 3 }$ </td><td>Learnable modulation coefficients for social cues fusion</td></tr><tr><td colspan="2">Features and Representations</td></tr><tr><td> $\mathbf { f } _ { i } ^ { e } , \mathbf { f } _ { i } ^ { g } , \mathbf { f } _ { i } ^ { \overline { { g } } }$ </td><td>Self representation, in-group representation, and out-of-group representation</td></tr><tr><td> $\mathbf { r } _ { i }$ </td><td>Concatenated perception vector of region-level motion cues</td></tr><tr><td> $\mathbf { f } _ { i }$ </td><td>Final fused feature representation for trajectory prediction</td></tr></table>

## Appendix B Discussions of Short-term Prediction Network

## B.1 Ego Loss Ratio

![](images/14198453ec35f8f264ccbc7f18c5c6896056346fe5c365f3ea7cda805f5568d4.jpg)  
Fig. B1 Efect of the ego loss ratio $\beta$ on the average number of group members across diferent scenes. The x-axis denotes the ego loss ratio, the y-axis denotes the average number of group members, and each curve corresponds to one scene.

In this section, we further discuss the parameterization of the short-term prediction network<sup>1</sup> in Socialality. Fig. B1 investigates how the average number of inferred group members changes as we vary the ego loss ratio $\beta$ across diferent ETH-UCY and SDD subsets. The calculation of average number of inferred group members can be formulated as

$$
\mathbb { E } [ | \mathcal { G } _ { i } | ] = \sum _ { i \in \{ 1 , 2 , . . . , N _ { e } \} } \frac { | \mathcal { G } _ { i } | } { N _ { e } } .\tag{B1}
$$

As mentioned in the Method section, the original $\ell _ { 2 }$ loss is fixed with a unit coeficient, the ego loss ratio $\beta$ still substantially afects the learned grouping behaviors.

Grouping decisions depend critically on the stability of the ego-centric relative motion. When the ego loss ratio is too small, ego predictions are weakly supervised, thus remaining relatively noisy. As a result, the grouping kernel tends to make conservative and inconsistent membership decisions, preventing the formation of larger and coherent groups. Increasing the ego loss ratio to a moderate regime (around 0.4) efectively stabilizes ego predictions, providing a reliable motion reference that enables the grouping kernel to expand groups more confidently and consistently. This efect is most prominent on interaction-heavy subsets such as univ and zara1, where the grouping structures are more common.

Notably, further increasing the ego loss ratio beyond 0.4 does not further improve grouping. Instead, on subsets such as univ, the inferred group size decreases sharply when the ego loss ratio becomes large $( \mathrm { e . g . , 0 . 6 - 0 . 8 } )$ . In this case, grouping becomes more cautious that neighbors are less likely to be assigned into the target agent’s group, leading to smaller inferred groups. Therefore, the peak around 0.4 can be interpreted as the balance point between concentration on self and social behaviors.

Diferent subsets also exhibit distinct sensitivities to the ego loss ratio. Interactionheavy scenes such as zara1 and univ present clear peaks around 0.4. In contrast, subsets such as hotel, zara2, and hyang0 show relatively mild variations, implying that their social structures are less dependent on expanding group membership.

Overall, analyses above further validate the efectiveness of setting the ego loss ratio $\beta = 0 . 4$ . In summary, setting the ego loss ratio to 0.4 can be viewed as a tradeof across subsets. It is large enough to learn consistent grouping priors in highly interactive scenes, and meanwhile it does not cause noticeable degradation on subsets with weaker interactions.

## B.2 Short-term Prediction Intervention

As the short-term predictions are leveraged for grouping, it is necessary to analyze the reliability of the short-term predictions and its impact on the overall results. To systematically address this, we design a set of intervention experiments. Following the notation in Sec. 3.1, let $\hat { \mathbf { X } } _ { j } ( \hat { \Omega } )$ denote the original forecasted short-term trajectory for neighboring agent $j .$ We introduce three types of interventions to generate the intervened previews, which are then concatenated with observed reviews $\mathbf { X } _ { j } ( \Omega )$ to form the extended trajectory ${ \bf X } _ { j } ( \tilde { \Omega } )$ for the Socialality kernel. We formulate these interventions using the do-calculus as follows.

Noise Injection. The noise injection intervention aims to evaluate the model’s performance change to diferent degrees of short-term prediction deviations, which simulates the unreliable previews generated from short-term prediction network. Here, we inject zero-mean Gaussian noise into the short-term previews of agent j to perform this intervention. Formally, for all neighboring agent $j \in \mathcal N _ { i }$ 2

$$
\begin{array} { r l } & { d o ( \hat { \mathbf { X } } _ { j } ( \hat { \boldsymbol { \Omega } } ) = \hat { \mathbf { X } } _ { j } ^ { \mathrm { n o i s e } } ( \hat { \boldsymbol { \Omega } } ) ) , \quad \mathrm { w h e r e } } \\ & { \hat { \mathbf { X } } _ { j } ^ { \mathrm { n o i s e } } ( \hat { \boldsymbol { \Omega } } ) = \hat { \mathbf { X } } _ { j } ( \hat { \boldsymbol { \Omega } } ) + \boldsymbol { \epsilon } . } \end{array}\tag{B2}
$$

Here, ϵ is a matrix sharing the same shape with $\mathbf { X } _ { j } ( \hat { \Omega } )$ . Each element in matrix ϵ is an independently sampled noise, which satisfies $\epsilon _ { i j } \stackrel { \mathrm { i . i . d . } } { \sim } \mathcal { N } ( 0 , \sigma ^ { 2 } )$

Trajectory Swapping. The trajectory swapping intervention aims to test the efect of assigning entirely irrelevant previews to agents, which simulates replacing agents’ trajectories. Here, we introduce a shufle ratio $r _ { s } \in [ 0 , 1 ]$ to determine the proportion of agents among all neighbors ${ \mathcal { N } } _ { i }$ that are subjected to swapping. Specifically, we randomly sample a subset of neighboring agents $\mathcal { N } _ { i } ^ { \mathrm { s w a p } } \subseteq \mathcal { N } _ { i }$ . We then shufle the assigned trajectories among the selected agent $j \in \mathcal { N } _ { i } ^ { \mathrm { s w a p } } . \ \overline { { j } }$ indicates the original neighbor index is shufled from j to $\bar { j } \in \mathcal { N } _ { i } ^ { \mathrm { s w a p } }$ . We perform this intervention as follows:

$$
\begin{array} { r l } & { d o ( \hat { \mathbf { X } } _ { j } ( \hat { \Omega } ) = \hat { \mathbf { X } } _ { j } ^ { \mathrm { s w a p } } ( \hat { \Omega } ) ) , \quad \mathrm { w h e r e } } \\ & { \hat { \mathbf { X } } _ { j } ^ { \mathrm { s w a p } } ( \hat { \Omega } ) = \left\{ \begin{array} { l l } { \hat { \mathbf { X } } _ { \overline { { j } } } ( \hat { \Omega } ) , } & { \mathrm { i f ~ } j \in \mathcal { N } _ { i } ^ { \mathrm { s w a p } } } \\ { \hat { \mathbf { X } } _ { j } ( \hat { \Omega } ) , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{B3}
$$

Trajectory Flipping. The trajectory flipping intervention aims to simulate directional failures of the short-term prediction network output. Here, we introduce a flip ratio $r _ { f } \in [ 0 , 1 ]$ to determine the proportion of agents among all neighbors ${ \mathcal { N } } _ { i }$ that are subjected to flipping. Similarly, we also randomly sample a subset of neighboring agents $\mathcal { N } _ { i } ^ { \mathrm { { f l i p } } } \subseteq { \mathcal { N } } _ { i }$ . For any selected agent $j \in \mathcal { N } _ { i } ^ { \mathrm { f l i p } }$ , we mirror its predicted future trajectory according to its last observed position $\mathbf { p } _ { j } ^ { 0 }$ . Specifically, for any future step $t \in { \hat { \Omega } }$ , the flipped coordinate is symmetrically computed as $2 \mathbf { p } _ { j } ^ { 0 } - \hat { \mathbf { p } } _ { j } ^ { t }$ . We perform this intervention as follows:

$$
\begin{array} { r l } & { d o \left( \hat { \mathbf { X } } _ { j } ( \hat { \Omega } ) = \hat { \mathbf { X } } _ { j } ^ { \mathrm { H i p } } ( \hat { \Omega } ) \right) , \quad \mathrm { w h e r e } } \\ & { \hat { \mathbf { X } } _ { j } ^ { \mathrm { H i p } } ( \hat { \Omega } ) = \left\{ \begin{array} { l l } { \left\{ 2 \mathbf { p } _ { j } ^ { 0 } - \hat { \mathbf { p } } _ { j } ^ { t } \right\} _ { t \in \hat { \Omega } } , } & { \mathrm { i f ~ } j \in \mathcal { N } _ { i } ^ { \mathrm { H i p } } } \\ { \hat { \mathbf { X } } _ { j } ( \hat { \Omega } ) , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{B4}
$$

By conducting the interventions above, we can quantitatively observe how the short-term predictions afect the subsequent grouping and the trajectory prediction performance. We visualize some examples corresponding to each intervention manner to better illustrate how we intervene the short-term predictions in Fig. B2.

The efects of noise injection can be easily observed in Fig. B2 (a1)-(a3). We can observe that the previews, $i . e . ,$ the short-term predictions, in (a2) and (a3) exhibit varying degrees of fluctuation compared to the baseline in (a1). In (a2), when $\sigma = 0 . 1$ this fluctuation appears relatively moderate, whereas the previews in (a3) already display random coordinate jumps. The trajectory flipping applied to diferent proportions of neighbors can also be observed in Fig. B2 (c1)-(c3). Specifically, we validate the trajectory swapping intervention by visualizing the target agent’s group members. In (b2), swapping the trajectories of a subset of neighbors reduces the original two in-group agents to one, while swapping all of them in (b3) completely reduces the in-group members to zero. It should be noted that since a certain proportion of neighbors is randomly selected, the specific group priors shown in (b2) are only one of the possibilities. Next, we conduct quantitative experiments for each intervention strategy under various parameter settings to evaluate their impact on the overall performance. The intervention of the short-term predictions would directly change the subsequent grouping process. To quantify how the intervened variants’ group priors difer from the base Socialality, we propose the group consistency metric to calculate the ratio of the target agents that have the same group priors as the base model.

![](images/2ffa03bb88bd6942b949fda039e0b1c92710874be5cbfce03f8c37fdd5a050f5.jpg)  
Fig. B2 Intervention examples under diferent intervention parameters in each corresponding intervention method.

Table B2 Intervention experiments of the short-term predictions on the zara1 dataset
<table><tr><td>Intervention</td><td>ID</td><td>Param</td><td>zaral (ADE/FDE)</td><td>∆% (ADE/FDE)</td><td>Group Consistency</td></tr><tr><td>Noise Injection (σ)</td><td>n1</td><td>0.1</td><td>0.1710 /0.2940</td><td>+0.43% /+0.06%</td><td>89.98%</td></tr><tr><td></td><td>n2</td><td>0.5</td><td>0.1814 /0.3069</td><td>+6.29% / +4.45%</td><td>60.23%</td></tr><tr><td></td><td>n3</td><td>1.0</td><td>0.2019 0.3372</td><td>+18.29% / +14.74%</td><td>33.23%</td></tr><tr><td></td><td>n4</td><td>2.0</td><td>0.2219 0.3800</td><td>+30.05% / +29.31%</td><td>9.19%</td></tr><tr><td>Trajectory Swapping (rs)</td><td>s1</td><td>0.2</td><td>0.1884 /0.3274</td><td>+10.41% +11.42%</td><td>41.61%</td></tr><tr><td></td><td>s2</td><td>0.5</td><td>0.2136 /0.3771</td><td>+25.14% / +28.34%</td><td>11.04%</td></tr><tr><td></td><td>s3</td><td>0.8</td><td>0.2401 /0.4282</td><td>+40.67% / +45.74%</td><td>1.77%</td></tr><tr><td></td><td>s4</td><td>1.0</td><td>0.2556 /0.4556</td><td>+49.78% / +55.06%</td><td>0.35%</td></tr><tr><td>Trajectory Flipping (rf)</td><td>f1</td><td>0.2</td><td>0.1886 /0.3250</td><td>+10.51% / +10.59%</td><td>69.25%</td></tr><tr><td></td><td>f2</td><td>0.5</td><td>0.2126 /0.3629</td><td>+24.56% / +23.49%</td><td>41.00%</td></tr><tr><td></td><td>f3</td><td>0.8</td><td>0.2392 /0.4056</td><td>+40.15% / +38.05%</td><td>26.13%</td></tr><tr><td></td><td>f4</td><td>1.0</td><td>0.2558 /0.4316</td><td>+49.92% / +46.88%</td><td>21.22%</td></tr><tr><td>No intervention</td><td>| base |</td><td>0</td><td>0.1686 /0.2938</td><td>(base)</td><td>(base)</td></tr></table>

σ denotes the standard deviation of the injected Gaussian noise. r and $r _ { f }$ denote the ratios of agents whose previews are randomly exchanged or symmetrically flipped, respectively. Results are reported over 5 independent runs. The performance degradations ∆% and the group consistency of all variants are compared to the base Socialality model, which involves no intervention operation.

As shown in Table. B2, we observe a consistent degradation in prediction performance across all three intervention strategies as the intervention intensity increases. First, under the noise injection intervention, the model exhibits a certain degree of robustness against slight preview fluctuation, showing merely a 0.43%/0.06% ADE/FDE degradation when $\sigma = 0 . 1$ . However, as the injected noise becomes more severe, the performance drops significantly. Second, the structural interventions, i.e., trajectory swapping and flipping, inflict much more severe performance drops compared to the noise injection. Even when a small proportion of neighbors’ previews are exchanged or flipped $( r _ { s } = 0 . 2 \ \mathrm { o r } \ r _ { f } = 0 . 2 )$ , the ADE and FDE increase by over 10%. When the previews of all neighbors are entirely replaced or mirrored $( r _ { s } = 1 . 0$ or $r _ { f } = 1 . 0 )$ , the performance drops drastically by up to almost 50%. Furthermore, the newly introduced group consistency metric matches this performance deterioration. When a slight noise $( \sigma = 0 . 1 )$ is injected, the group consistency remains almost the same at 89.98%, which explains the minimal impact on the final predictions. However, structural interventions severely disrupt the grouping accuracy. Specifically, swapping just 20% of the neighbors’ previews $( r _ { s } = 0 . 2 )$ drastically decreases the group consistency to 41.61%, while completely exchanging them $( r _ { s } = 1 . 0 )$ leads to a consistency of merely 0.35%. Similarly, trajectory flipping significantly lowers the consistency down to 21.22%, confirming that the correct directionality of previews is also essential for grouping decisions. These comprehensive results demonstrate that our model explicitly relies on the structural and directional correctness of the short-term predictions. When these previews are unreliable or completely wrong, the subsequent grouping mechanism is heavily disrupted, which naturally leads to a substantial deterioration in the final trajectory prediction performance.

## Appendix C Further Discussions of Perception Mechanism

## C.1 The Adaptive FOV Variant

The diferent FOV setting experiments in GPCC [2] indicate that agents might be able to capture wider vision range under specific scenes such as competitive games or extreme high-pedestrian-density scenes through frequent head movement. In realworld scenes, the ego agent also possesses a zooming capability towards diferent objects in vision range. This motivates us to regard introducing such zooming mechanism into social interaction modeling as our further considerations. Here, we conducted some basic modifications to our base model. Accordingly, we modify the FOV module in our perception mechanism and carry out the prediction experiments over the original ETH-UCY and SDD datasets, as well as the NBA dataset [3]. The settings of ETH-UCY and SDD datasets are strictly equal to the vanilla Socialality model. As for the NBA dataset, the distributions of NBA dataset are completely diferent from the pedestrian datasets. Following the methodology proposed by Xu et al. [4, 5], we set the parameters to be $\{ t _ { o } , t _ { p } , T \} = \{ 5 , 1 0 , 0 . 4 \mathrm { s } \}$ and $\{ t _ { h } , t _ { f } \} = \{ 3 , 2 \}$ . We randomly select approximately 50000 samples, with 65% allocated for training, 25% for testing, and 10% for validation.

Specifically, we introduce a light-weight embedding network $m _ { \mathrm { f o v } } ( \cdot )$ to encode target player’s surrounding players’ relative directions into a normalized FOV angle $ { \hat { \theta } } _ { \mathrm { F O V } }$ Such normalized FOV angle $ { \hat { \theta } } _ { \mathrm { F O V } }$ corresponds to real FOV angle from $0 ^ { \circ }$ to $3 6 0 ^ { \circ }$ . For a target agent i, the vector $\mathbf { r } _ { i }$ represents each surrounding player’s relative position at the last observation step. Formally, its normalized adaptive FOV angle is calculated through

$$
\begin{array} { r } { \hat { \theta } _ { \mathrm { F O V } } ^ { i } = m _ { \mathrm { f o v } } ( \mathbf { r } _ { i } ) \in ( - 1 , 1 ) . } \end{array}\tag{C5}
$$

The light-weight embedding network $m _ { \mathrm { f o v } } ( \cdot )$ comprises a two-layer MLP and its output is constrained to (−1, 1) through the final Tanh activation. Then the real FOV angle is gained through

$$
\begin{array} { r } { \theta _ { \mathrm { F O V } } ^ { i } = \left( 1 + \hat { \theta } _ { \mathrm { F O V } } ^ { i } \right) \pi \in ( 0 , 2 \pi ) . } \end{array}\tag{C6}
$$

## C.2 Evaluation with NBA Dataset

The prediction performance of the original GPCC, Socialality, and the adaptive FOV variant Socialality+adaptive FOV on ETH-UCY and SDD datasets is reported in Table. C3. We make some further modifications and get some interesting results on NBA dataset, which is reported in Table. C4. We observe in Table. C3 that equipping the model with the full dynamic FOV zooming capability brings only marginal performance improvements on the current pedestrian datasets (ETH-UCY and SDD). Specifically, although the adaptive FOV variant achieves slightly better ADE and FDE on the eth subset, and FDE on the zara2 subset (with a 2% improvement on the FDE of zara2), compared to the vanilla Socialality in the ETH-UCY dataset, its performance on other subsets is even slightly worse than the original model. On the SDD dataset, it also underperforms the vanilla Socialality by 0.74%/0.69% in terms of ADE/FDE. These marginal efects validate our initial settings that in standard pedestrian crowds, which share similar motion patterns with indoor scenes like shopping malls and airports, a fixed 180<sup>◦</sup> FOV is an optimal and highly acceptable compromise without over-engineering.

Table C3 Prediction performance evaluated on ETH-UCY and SDD datasets
<table><tr><td>Models</td><td>eth</td><td>hotel</td><td>univ</td><td>zara1</td><td>zara2</td><td>SDD</td></tr><tr><td>GPCC</td><td>0.254/0.383</td><td>0.103/ /0.156</td><td>0.256/0.448</td><td>0.175/0.284</td><td>0.134/0.225</td><td>6.394/10.173</td></tr><tr><td>Socialality</td><td>0.233/0.369</td><td>0.102/0.156</td><td>0.240/0.428</td><td>0.168/0.294</td><td>0.130/0.223</td><td>6.283/10.121</td></tr><tr><td>Socialality+aF</td><td>0.232/0.365</td><td>0.104/0.156</td><td>0.245/0.435</td><td>0.171/0.300</td><td>0.132/0.218</td><td>6.330/10.195</td></tr></table>

Blue markers denote the best results on each set. Socialality+aF indicates the Socialality+adaptive FOV variant.

These results suggest that in pedestrian crowds environments (e.g., ETH-UCY and SDD), the reliance on an adaptive FOV might be relatively low, where pedestrians mostly focus on their immediate forward direction. However, the dynamics in NBA games are notably diferent due to frequent collaboration and intense competition. During a basketball game, each player must constantly notice the positions of all teammates and opponents across the entire court, as well as the ball. Under these circumstances, restricting the model with a fixed biological FOV angle might largely limit its prediction capability.

Furthermore, the definition of a group in the NBA dataset difers significantly from the social groups (e.g., families or friends walking together) observed in street scenes. In professional sports, groups are inherently determined by team afiliation. Compared to instantly judging whether a near neighbor belongs to the target player’s same team with motion cues, game players know well who are their teammates. The audience can also tell this by observing the jersey colors easily. Accordingly, we extract deterministic team ownership labels directly from the metadata of the raw NBA dataset to construct a binary teammate group mask, where neighbors belonging to the same team as the target player are considered in-group agents, while opponents and the ball are out-ofgroup agents. The results are shown in Table. C4.

Table C4 Comparisons to the state-of-the-art methods on NBA dataset
<table><tr><td>Models</td><td>tp = 5</td><td>tp = 10</td></tr><tr><td>Social-LSTM[6] (2016)</td><td>0.88/1.53</td><td>1.79/3.16</td></tr><tr><td>S-GAN[7] (2018)</td><td>0.85/1.36</td><td>1.62/2.51</td></tr><tr><td>Social-STGCNN[8] (2020)</td><td>0.75/0.99</td><td>1.59/2.37</td></tr><tr><td>GroupNet+NMMP[4] (2022)</td><td>0.69/1.08</td><td>1.25/1.80</td></tr><tr><td>GroupNet+CVAE[4] (2022)</td><td>0.62/0.95</td><td>1.13/1.69</td></tr><tr><td>SocialCircle[9] (2024)</td><td>0.67/0.90</td><td>1.18/1.46</td></tr><tr><td>GPCC (Ours)</td><td>0.62/0.87</td><td>1.19/1.58</td></tr><tr><td>Socialality (Ours)</td><td>0.624/0.875</td><td>1.212/1.641</td></tr><tr><td>Socialality+adaptive FOV (Ours)</td><td>0.618/0.862</td><td>1.198/1.614</td></tr><tr><td>Socialality+team (Ours)</td><td>0.605/0.829</td><td>1.172/1.521</td></tr></table>

Metrics are reported in the form of “ADE/FDE” (best-of-20) in meters. Lower metrics indicate better prediction performance. Blue markers denote the best 3 results on each set. Socialality+adaptive FOV replaces fixed FOV angle with adaptive FOV angle embedding. Socialality+team uses team ownership as group priors before the perception mechanism.

In Table. C4, our four methods (GPCC, Socialality, Socialality+adaptive FOV, and Socialality+team) achieve promising results compared to SOTA methods despite the completely diferent interaction patterns in the NBA dataset. These results validate the generalization capability of our approach. Moreover, comparing Socialality and Socialality+adaptive FOV shows that the adaptive variant performs better. This indicates that a fixed FOV angle limits the model’s prediction performance on the NBA dataset. Also, Socialality+team variant showing the best prediction performance coincidentally validates our basic motivation, which is acquiring the group priors before social interaction modeling (group-bounded in the title of this manuscript).

## C.3 Distributions of FOV

We also analyse the adaptive FOV angle θ<sub>FOV</sub> distributions across diferent datasets. The results are visualized in Fig. C3. For the typical pedestrian scene zara1, the average

![](images/74b94035e7879862e8875a033936899347b80dcc0d26617fec22023f48c85fa2.jpg)

![](images/525bacecbe4b31d67a8defe4ca7dff2839e1026697f4fb4b27991d6779d79639.jpg)

![](images/b0b58eafd5835ca515b2c738e5690b1c0a1cfae82e41eaac70621ccc2a9dc5d2.jpg)  
Fig. C3 Visualized distributions of FOV angle $\theta _ { \mathrm { F O V } }$ on diferent datasets gained through Socialality+adaptive FOV.

FOV angle of $1 4 9 . 5 ^ { \circ }$ aligns well with the biological human visual field. Meanwhile, in the univ dataset, the average FOV angle expands to $2 3 0 . 2 ^ { \circ }$ , suggesting the need for broader perception in denser or more complex environments. For the highly dynamic NBA scene, the average FOV angle significantly increases to $2 8 6 . 9 ^ { \circ }$ . Furthermore, for the NBA dataset, the majority of the samples concentrate within the highly-focused interval of $2 8 0 ^ { \circ }$ to $2 9 5 ^ { \circ }$ . It aligns with cognitive patterns in competitive sports that players must maintain wider surrounding perception range. They might achieve this through more eye movements and more frequent head rotations compared to that of pedestrians. Furthermore, these experiments above demonstrate that the model adaptively adjusts agents’ perception range based on the unique motion distributions of diferent scenes without any physical hard-coding.

## Appendix D Discussions of Considering Environmental Contexts

## D.1 Design of sa-map Variant

To investigate the impact of environmental contexts, we follow our previous work SocialCircle+ [10] to bring environmental information into consideration. Specifically, SocialCircle+ extends the base SocialCircle [9] by introducing a conditional branch that extracts a behavior-semantic segmentation map from the scene image. It treats the pixels in this map labeled from walkable to unwalkable as virtual agents to construct PhysicalCircle meta components, which capture environmental factors such as relative velocity and equivalent distance to obstacles. These environmental conditions are then adaptively fused with social components to simulate how physical environments afect agents’ interactive decisions. In our proposed Socialality, the perception mechanism uses group priors as conditions to model the final social interaction representation. Similarly, we adopt the virtual agent strategy from SocialCircle+ to simultaneously consider the environmental and social interactions. In SocialCircle+, we conceptually treat the physical environment as a collection of discrete and stationary virtual agents.

To illustrate how the environment comes into play with the grouping mechanisms, we naturally extend our framework by incorporating these virtual agents directly into the Socialality framework. For neighboring agents, Socialality divides them into in-group agents and out-of-group agents through Socialality, which represents social constraints. Similarly, we apply the exact same grouping strategy to these virtual agents representing the environmental constraints. Furthermore, these grouped virtual agents are also jointly processed by the perception mechanism. The in-group and out-of-group features output by the perception mechanism inherently encode the contextual information of the virtual agents surrounding the target agent. These features are then fused into a comprehensive representation that simultaneously accounts for social interactions with neighboring agents and environmental interactions with virtual agents. This fused representation is then fed into the final trajectory prediction backbone to generate future paths.

![](images/78937ef84dc2e3a184f3ce80a0cc6f11164d1a2ff568f5ba2eff4838e411e92b.jpg)  
Fig. D4 Method illustration when introducing environmental contexts to the proposed Socialality network.

Here, we specifically draw a method figure to better illustrate how we introduce the segmentation map and the virtual agents to the Socialality in Fig. D4. First, the segmentation map is obtained following the method in SocialCircle+ [10]. The segmentation map is hand-labeled at the pixel level to indicate the degree of walkability for corresponding regions. For the ETH-UCY dataset, the scenes are relatively simple and small in scale, allowing us to basically annotate the corresponding regions using just two labels, i.e., walkable or completely non-walkable. For the SDD dataset, the map range is larger and has more complex environmental conditions. For example, there are large areas of grass frequently appearing on the campus, where most agents generally will not walk under normal circumstances. However, through observation of the SDD dataset videos, there are also cases indicating that agents may walk onto the grass to rest, cross the grass to take a shortcut, or briefly step on the grass to avoid other agents. For regions similar to grass, we make a trade-of and assign them an intermediate value between walkable and non-walkable. We select several representative segmentation maps that we labeled here in Fig. D5. The complete segmentation maps and related detailed settings can be found in our previous paper SocialCircle+ [10].

![](images/8ec54fa21d8659ef670911e43d341daedd730ce262553e55986e37f46ca50e39.jpg)  
Fig. D5 Some examples of manually labeled segmentation maps on ETH-UCY and SDD datasets.

In both the original base Socialality model and the current sa-map variant, the perception mechanism uses group priors as conditions to perceive interactions with other agents, including real agents and virtual agents. Specifically, the coordinates of the pixels from the segmentation map are directly treated as the positions of the virtual agents. Since these virtual agents represent stationary environmental objects, their velocities are set to zero. Then, the proposed Socialality evaluates both the spatial distance and the relative velocity between the target agent and these virtual agents. In this way, the grouping mechanism seamlessly integrates the physical environment by treating it as a special type of social interaction. Comparing Fig. D4 with the method figure in the Method section, we can observe that the overall framework of the current sa-map model variant is the same as Socialality. The only diference is the input part. The sa-map variant uses mixed agents observed trajectories as input, while base Socialality model receives only trajectories from real agents. It should be noted that the mixed trajectories are also extended through the short-term prediction network.

Table D5 Prediction performance of sa-map model evaluated on ETH-UCY and SDD datasets
<table><tr><td>Models</td><td>S</td><td>eth</td><td>hotel</td><td>univ</td><td>zaral</td><td>zara2</td><td>SDD</td></tr><tr><td>sa-map</td><td></td><td>0.232/0.361</td><td>0.109/0.173</td><td>0.248/0.446</td><td>0.183/0.309</td><td>0.140/0.241</td><td>6.354/10.176</td></tr><tr><td>sa-map</td><td>125</td><td>0.238/0.375</td><td>0.115/0.168</td><td>0.244/0.439</td><td>0.186/0.319</td><td>0.139/0.245</td><td>6.310/10.106</td></tr><tr><td>sa-map</td><td></td><td>0.230/0.360</td><td>0.104/0.155</td><td>0.237/0.416</td><td>0.170/0.294</td><td>0.130/0.218</td><td>6.274/10.074</td></tr><tr><td>sa-map</td><td>10</td><td>0.234.0.369</td><td>0.107/0.160</td><td>0.239/0.430</td><td>0.181/0.304</td><td>0.134/0.233</td><td>6.303/10.102</td></tr><tr><td>Socialality</td><td>-</td><td>0.233/0.369</td><td>0.102/0.156</td><td>0.240/0.428</td><td>0.168/0.294</td><td>0.130/0.223</td><td>6.283/10.121</td></tr></table>

Stride value s indicates the pooling stride used when reducing virtual agent density. Larger stride means fewer virtual agents. The default maximum virtual agents in a scene is set to 100 × 100. When stride s is set to 5, it means the maximum virtual agents are 20 × 20. Blue markers denote the best results on each set.

## D.2 Quantitative Analyses

We evaluated the prediction performance of the sa-map model variant on the ETH-UCY and SDD datasets, and the results are presented in Table. D5. It should be noted that the sa-map variant has only 3.2% more parameters than the original Socialality. As shown in Table. D5, the performance of the sa-map variant varies with the pooling stride s which controls the virtual agent density. Setting a moderate stride of s = 5 achieves the overall best balance between environmental cues and social interactions, outperforming the baseline Socialality across most datasets. This demonstrates that incorporating environmental contexts into our grouping mechanism can efectively improve prediction accuracy. To further analyse the grouping mechanism of current samap model variant, we visualize several prediction and grouping examples comparing Socialality with the current sa-map model variant in Fig. D6 and Fig. D8. Here we choose s = 5 as the stride for sa-map variant since this setting leads to best prediction performance.

![](images/067da97702b85d7200d942c12a5677f3df227419d7b16172cd8bac1663d0ad6a.jpg)  
Fig. D6 Distributions of predicted trajectories provided by the base Socialality and the sa-map variant on the ETH-UCY zara1, SDD hyang0, and deathCircle2 scenes. Segmentation regions are noted by semi-transparent orange color on the figure.

## D.3 Qualitative Analyses

1) Overall Prediction: We can observe in Fig. D6 that the final predictions of the base model Socialality and the sa-map variant are significantly diferent. On the one hand, comparing (a1) and (a2) with (b1) and (b2) reveals that the distributions of the predicted trajectories from the sa-map variant are more concentrated, attempting to constrain the trajectories within walkable regions. Without the constraints of the segmentation map, the predicted trajectories of the Socialality clearly exhibit a more diverse distribution. This also applies to relatively slow-moving agents, as (a3) and (b3) demonstrate diferent prediction patterns for the future movement trends of a nearly stationary target agent with higher uncertainty. On the other hand, comparing (a4) and (b4) shows that the sa-map variant might not achieve this simply by imposing greater restrictions on the overall prediction distributions of the model, since the prediction distributions in (a4) and (b4) are relatively similar. It indicates that the model adaptively chooses whether to apply environmental constraints to the current target agent. The same phenomenon can be observed in the SDD dataset. For example, in the SDD hyang0 intersection scene shown in (a5), (a6) and (b5), (b6), the sa-map variant outputs trajectories that are more aligned with map restrictions compared to those of the base Socialality. This is also observed in (a7) and (a8), where the sa-map variant predicts that the target agent tends to avoid collisions with the building, while the Socialality provides much more diverse predictions.

Accordingly, the quantitative and the qualitative analyses above validate that the sa-map variant has successfully learned the impact of environmental constraints on the predicted trajectories. Given the segmentation map, the sa-map variant can output predicted trajectories that better align with the environmental context conditions. Here, we would like further analyze how grouping mechanism and group behavior will change after introducing environmental contexts as virtual agents.

![](images/413b59b6524cc9e1e49ab0d91d306cdd707dbcdbe3899bc000c4561e9e4b8bb4.jpg)  
Fig. D7 Examples of diferent group priors provided by the base Socialality and the sa-map variants.

2) Grouping Analyses: In Fig. D8, we visualize several examples to investigate whether the grouping results of the sa-map variant difer after introducing virtual agents. First, we can observe that the base Socialality and the sa-map (s = 5) variant can generally achieve reasonable grouping of real agents. For the case in (a3) and (b3), both models provide the same group priors to include two neighbors walking in front of the target agent<sup>2</sup>. However, compared with (b1)-(b3), Fig. D8 (a1)-(a3) exhibits a more inclusive grouping preference. First, in (a1) and (b1), we can observe that the base Socialality includes an approaching agent, while the sa-map variant decides that a two-agent group is suficient. This phenomenon can also be observed in (a2) and (b2). The diferent grouping preferences may reflect an inherent model tendency. After introducing environmental contexts as virtual agents, the model is more inclined to shrink its group boundary, thus excluding some neighbors that would have originally been divided into the in-group agent set. We would like to further analyze the reason behind this phenomenon.

![](images/8854fc1452e62eb471fcf1a18564f6662e4dbd7721afd59d3fcaf84a48e6cc7d.jpg)  
Fig. D8 Examples of diferent group priors (including virtual agents) provided by the base Socialality and the sa-map variants under diferent stride value.

Since the only diference between the base Socialality and the sa-map variant is that the latter utilizes virtual agents and treats them as real agents, we visualize several examples to investigate how these virtual agents change the model’s grouping decisions in Fig. D8. Moreover, we add a sa-map variant under s = 1 to provide more in-depth analyses of virtual agents. By comparing (b1) and (c1), we find that sa-map (s = 5) grouped some virtual agents around the target agent, while sa-map $( s \ = \ 1 )$ chooses to consider the target agent as individually moving. The sa-map (s = 1) variant exhibits an even more exclusive tendency compared to sa-map (s = 5). Example in (b2) and (c2) also indicates that sa-map $( s = 5 )$ determines some in-group virtual agents, whereas in (b2), sa-map $( s = 1 )$ decides that no virtual agent should be included as an in-group agent. Further comparing (b3) and (c3) reveals that samap $( s = 1 )$ might cluster a massive number of virtual agents into the group for a target agent walking on the edge of non-walkable regions. In contrast, sa-map $( s = 5 )$ conducts the grouping for a much larger non-walkable region by incorporating only 5 virtual agents. This validates the necessity of introducing the virtual agents maxpooling strategy. Under the default setting, the number of virtual agents is far greater than the number of real agents in the scene. If there is a non-walkable region around the target agent, the number of virtual agents will also be excessively large. However, the entire Socialality framework equally considers real and virtual agents. This forces the model to decrease the probability of the target agent including neighbors, thus avoiding the inclusion of too many virtual agents that are useless for the final prediction. Conversely, with a reasonable stride value, the number of virtual agents in the scene is moderate, allowing them to jointly participate with real agents as group priors of the target agent, thereby achieving better trajectory prediction performance.

![](images/140a34f009055ddad3eec76d4d65be333bc4194c8bfdfa1c48aa23cab61adacb.jpg)

![](images/2f821743238c33e7b8291fd3f0adf9b715e3d917a9f96e55f67d54a52aac2237.jpg)  
Fig. D9 Visualization of socialality anchor distributions provided by the base Socialality and the sa-map variant.

3) Anchor Analyses: We are also wondering whether the socialality anchors would change after introducing environmental contexts as virtual agents. In Fig. D9, we can observe that the anchor distribution in both base model and the sa-map $( s \ : = \ : 5 )$ variant exhibit a distinct “V”-shape structure. However, the direction and distribution of the “V”-shape structure have changed in the sa-map (s = 5) variant. In our manuscript, we analyzed the relationship between the agent-specific anchor and the grouping rule from agent-level and group level perspective. Meanwhile, we analyzed the anchor distributions across diferent dataset scenes and the corresponding agent behavior patterns in each region. Finally, we demonstrated why more anchors are not needed from the statistical analysis. We can observe in Fig. D9 that the anchor distribution of the sa-map $( s = 5 )$ variant is more scattered, with many agents’ anchor values falling into the middle region of the “V”-shape. Moreover, a large proportion of the anchor distribution in sa-map $( s = 5 )$ is concentrated in regions with a large speed tolerance $\tau _ { b } .$ , while the distance tolerance $\tau _ { a }$ is evenly distributed. This corresponds to the situation where the target agent requires a larger $\tau _ { b }$ to include stationary virtual agents as in-group agents. It indicates that the model can adaptively adjust the anchor distribution and learning pattern to determine how to group neighboring agents more efectively. Here, the anchor visualization of the sa-map $( s = 5 )$ variant further validates the efectiveness of our proposed socialality anchors in representing the grouping preferences of agents.

## Appendix E Higher-degree Group Members Analyses

## E.1 Definition of Higher-degree Group Members

We conducted two analyses on the higher degree members among the target agent’s neighbors. We have cited a representative method in our original manuscript studying higher-order social relations [11], which uses HighGraph to capture the higher-order dynamics of social interactions. Similarly, to explicitly capture this chain efect of grouping relations without relying on graph, we follow how Kim et al. [11] construct HighGraph to adopt a recursive approach to construct higher degree group members. Specifically, for a target agent $i ,$ there might exist an in-group set containing its ingroup neighbors, which we refer to as the first-degree group set, denoted as $\mathcal { G } _ { i } ^ { ( 1 ) }$ Subsequently, each agent $j$ within this first-degree group set $( j \in \mathcal { G } _ { i } ^ { ( 1 ) } )$ possesses its own first-degree group set $\mathcal { G } _ { j } ^ { ( 1 ) }$ . The union of these individual sets is defined as the second-degree group set $\begin{array} { r } { \mathcal { G } _ { i } ^ { ( 2 ) } = \bigcup _ { j \in \mathcal { G } _ { i } ^ { ( 1 ) } } \mathcal { G } _ { j } ^ { ( 1 ) } } \end{array}$ . The third-degree group set is then defined as $\begin{array} { r } { \mathcal { G } _ { i } ^ { ( 3 ) } = \bigcup _ { j \in \mathcal { G } _ { i } ^ { ( 2 ) } } \mathcal { G } _ { j } ^ { ( 2 ) } } \end{array}$ . Following the same logic, the p-degree group set can be calculated from the $( p - 1 )$ -degree group set:

$$
\mathcal { G } _ { i } ^ { ( p ) } = \bigcup _ { j \in \mathcal { G } _ { i } ^ { ( p - 1 ) } } \mathcal { G } _ { j } ^ { ( 1 ) } , p \ge 1 .\tag{E7}
$$

Specifically, when $\begin{array} { r } { p = 1 , \mathcal { G } _ { i } ^ { ( 1 ) } = \bigcup _ { j \in \mathcal { G } _ { \cdot } ^ { ( 0 ) } } \mathcal { G } _ { j } ^ { ( 0 ) } } \end{array}$ , which indicates that the zero-degree group set ${ \mathcal G } _ { i } ^ { ( 0 ) }$ is the target agent itself, $i . e . , \mathcal { G } _ { i } ^ { ( 0 ) } = \{ i \}$

To illustrate the relation of these neighbors more directly, we visualize an example of second-degree group members. As shown in Fig. E10, (a1) represents that agent B is target agent A’s in-group neighbor. (a2) shifts the target agent to B, where agent A and agent C are both agent B’s in-group agents. In this way, agent C is connected with agent A via agent B, thus making itself agent A’s second-degree group member.

![](images/d7a0ab022bf0e336b983d386697fbe3fbef9b3eb7829f197bb9857f79aa4c7d3.jpg)  
Fig. E10 Illustration of first-degree group neighbors and higher-degree group neighbors.

In our Socialality, we apply two diferent interaction modeling strategies on ingroup and out-of-group agents, as shown in original manuscript Sec. 3.3 (Perception Mechanism). To take higher-degree group members into account, we modify the ingroup agents with diferent degree group members. The original in-group agents means considering up to the first-degree group members, which contains both the target agent $\{ i \} = \mathcal { G } _ { i } ^ { ( 0 ) }$ itself and the first degree group set $\mathcal { G } _ { i } ^ { ( 1 ) }$ . Specifically, when taking secondorder group members into account, the in-group members become the combination of original in-group agents set $\{ i \} \cup \mathcal { G } _ { i } ^ { ( 1 ) }$ and the second-degree group set $\mathcal { G } _ { i } ^ { ( 2 ) }$ , i.e., $\{ i \} \cup \mathcal { G } _ { i } ^ { ( 1 ) } \cup \mathcal { G } _ { i } ^ { ( 2 ) }$ . Following the same reasoning, considering up to q-degree group members means considering all correspond degree group members set $\mathcal { U } _ { i } ^ { q }$ , which can be formulated as:

$$
\mathcal { U } _ { i } ^ { q } = \bigcup _ { p = 0 } ^ { q } \mathcal { G } _ { i } ^ { ( p ) }\tag{E8}
$$

## E.2 Diferent Degree Group Members Feature Energy

To calculate the impact of introducing these higher-degree group members to our network, we treat the modification of the in-group set as a structural intervention. Specifically, we perform the intervention operation $d o ( \mathcal { G } _ { i } = \mathcal { U } _ { i } ^ { q } )$ to represent considering up to q-degree group members. Accordingly, the remaining neighbors set becomes $\mathcal { N } _ { i } \setminus \mathcal { U } _ { i } ^ { q }$ , which are regarded as the out-of-group agents to be calculated through the perception mechanism. In the original manuscript, the in-group agents finally form the in-group feature $\mathbf { f } _ { i } ^ { g }$ , which is modulated by the socialality anchors and fused through an fusion encoder $n ( \cdot )$ . Under $d o ( \mathcal { G } _ { i } = \mathcal { U } _ { i } ^ { q } )$ intervention, we calculate the feature energy of the modulated in-group feature, defined as the squared sum of the intervened feature tensors prior to the final fusion layer. Formally,

$$
E _ { i } ^ { g } ( \boldsymbol { q } ) = \left( \| c _ { 2 } \mathbf { f } _ { i } ^ { g } ( d o ( \mathcal { G } _ { i } = \mathcal { U } _ { i } ^ { q } ) ) \| _ { 2 } \right) ^ { 2 } .\tag{E9}
$$

We can observe from Table. E6 that the feature energy $E _ { i } ^ { g } ( q )$ increases as we expand the degree q. Specifically, when extending the group from first-degree $( q = 1 )$ to second-degree $( q = 2 )$ , the in-group feature energy experiences a notable increase across all models $( \mathrm { e . g . , + 2 8 . 1 \% }$ in $\# 1 _ { \mathrm { z a r a 1 } }$ and $+ 3 2 . 8 \%$ in $\# 1 _ { \mathrm { s d d } } )$ . However, when we further intervene the in-group agents set to third-degree members $( q = 3 )$ , the gain of feature energy becomes marginal. For example, in $\# 1 _ { \mathrm { z a r a 1 } }$ , the energy only marginally increases from +28.1% to +29.8% (1.7%). This phenomenon indicates that the second-degree group members might indeed need consideration, while the third or higher-degree group members are relatively marginal to the target agent.

Table E6 Feature energy for up to q-degree group members set $\mathcal { U } _ { i } ^ { q }$ evaluated on ETH-UCY zara1 across 3 trained weights and SDD datasets
<table><tr><td> $E _ { i } ^ { g } \left( q \right)$ </td><td>一  $d o ( \mathcal { G } _ { i } = \mathcal { U } _ { i } ^ { q } )$  一</td><td> $\# 1 _ { \mathrm { z a r a 1 } }$ </td><td> $\# 2 _ { \mathrm { z a r a 1 } }$ </td><td> $\# 3 _ { \mathrm { z a r a 1 } }$ </td><td> $\# 1 _ { \mathrm { s d d } }$ </td></tr><tr><td> $q = 1$ </td><td> $\mathcal { U } _ { i } ^ { 1 }$ </td><td> $4 1 2 4 7 ( + 0 . 0 \% )$ </td><td> $5 1 6 6 2 ( + 0 . 0 \% )$ </td><td>55459(+0.0%)</td><td>239994(+0.0%)</td></tr><tr><td> $q = 2$ </td><td> $\mathcal { U } _ { i } ^ { 2 }$ </td><td> $5 2 8 1 8 \dot { ( } + 2 8 . 1 \dot { \% } )$ </td><td> $6 7 8 3 7 ( + 3 1 . 3 \% )$ </td><td> $6 8 7 8 3 \dot { ( } + 2 4 . 0 \dot { \% } )$ </td><td>318733(+32.8%)</td></tr><tr><td> $q = 3$ </td><td> $\mathcal { U } _ { i } ^ { 3 }$ </td><td> $5 3 5 2 7 ( + 2 9 . 8 \% )$ </td><td> $6 9 6 6 8 ( + 3 4 . 9 \% )$ </td><td> $7 4 6 4 6 ( + 3 4 . 6 \% )$ </td><td> $3 2 9 0 1 3 ( + 3 7 . 1 \% )$ </td></tr></table>

Table E7 Feature energy allocation for diferent degree group members sets evaluated on ETH-UCY zara1 across 3 independent training runs and SDD datasets
<table><tr><td> $\mathcal { U } _ { i } ^ { q }$  一</td><td> $\# 1 _ { \mathbf { z a r a 1 } }$ </td><td> $\# 2 _ { \mathrm { z a r a 1 } }$ </td><td> $\# 3 _ { \mathrm { z a r a 1 } }$ </td><td>一  $\# 1 _ { \mathrm { s d d } }$ </td></tr><tr><td> $q = 1$ </td><td>99.89%</td><td>70.58%</td><td>99.39%</td><td>36.04%</td></tr><tr><td> $q = 2$ </td><td>99.89%</td><td>98.45%</td><td>99.99%</td><td>58.44%</td></tr><tr><td> $q = 3$ </td><td>99.89%</td><td>100.00%</td><td>100.00%</td><td>73.37%</td></tr></table>

## E.3 Diferent Degree Group Members Feature Sensitivity

To validate whether the Socialality captures and utilizes information from these exact higher-degree group members, we further analyse the final fused representation $\mathbf { f } _ { i }$ in Eq. (25) before trajectory prediction backbones. For a target agent i, $\mathbf { X } _ { k } ( t _ { o } , 0 ) \in \mathbb { R } ^ { t _ { o } \times 2 }$ denote the historical input trajectory of a neighboring agent k. We define the feature sensitivity $\Delta E _ { i  k }$ exerted by agent k on target i using the feature energy change. Formally,

$$
\Delta E _ { i  k } = | E _ { i } ( \Delta \mathbf { X } _ { k } ) | = \| \mathbf { f } _ { i } ( \Delta \mathbf { X } _ { k } ) \| _ { 2 } ^ { 2 } .\tag{E10}
$$

By computing this feature sensitivity over all agents $j \in \mathcal N _ { i }$ and aggregating each belonging to diferent q-degree group members set $\mathcal { U } _ { i } ^ { q }$ , we can quantify the Socialality’s feature energy allocation ratio towards diferent degree group members.

As shown in Table E7, we can observe that our proposed Socialality consistently allocates a major share of feature energy to the first-degree group members $( q = 1$ ， $i . e . , \ : \mathcal { U } _ { i } ^ { 1 } = \{ i \} \cup \mathcal { G } _ { i } ^ { ( 1 ) } )$ in most runs corresponding to zara1. Although the in-group feature increases considerably when further including the second-degree group members, as shown in Table. E6 and Table. E7, the model allocates little concentration on them since the fused feature sensitivity does not increase accordingly from $q = 1$ to $q = 2$ . However, in SDD dataset, the model implicitly learns to allocate an higher energy sensitivity to higher-degree group members. This indicates that the perception mechanism could adaptively adjusts energy allocation with higher proportion of higher-degree group members and the remaining neighbors according to diferent scenes and distributions.

## Appendix F Prediction Backbone Verifications

The proposed Socialality first obtains group priors via Socialality and utilizes them as conditions to derive the social interaction representation. Finally, a trajectory prediction backbone is employed to output the predicted trajectories based on the final social interaction representation. To further verify the generalization capability of Socialal-$i t y ,$ we replace the original MSN backbone with diferent backbones that are capable of modeling social interactions, and evaluate them on the ETH-UCY dataset to verify whether the prediction performance of Socialality can be maintained across various backbones. Specifically, we would like to briefly introduce the selected backbones below. It should be noted that all backbones generate $K _ { f } = 2 0$ trajectories.

MSN [12] is a Transformer-based multi-style trajectory prediction network. To build our base Socialality model $( i . e . , \ s a \mathrm { - M S N } )$ , the original social-interaction modules in MSN are removed and replaced by our group-conditioned representations.

Transformer [13] is the simplest Transformer model used to predict trajectories. Since MSN is a Transformer-based framework, we remove the MSN-related components and only keep the transformer part. Notably, this is the only backbone variant that requires fewer parameters than the base Socialality.

E-V<sup>2</sup>-Net [14] introduces the Fourier transform to trajectory prediction. Similarly, its default social-interaction modules are removed when constructing the sa-ev variant.

Resonance [15] is motivated by the co-vibration phenomenon and forecasts trajectories as the superposition of independent vibrations separately. Here, we replace the social-interaction part of Resonance with our final social interaction representation to compute the agents’ reactions to social vibrations. Since each vibration reaction is calculated independently, the sa-re variant contains the highest number of parameters among all tested models.

Table F8 Prediction performance of the proposed Socialality Model modified with diferent backbone prediction models on ETH-UCY dataset
<table><tr><td>Model</td><td>Backbone</td><td></td><td></td><td>hotel</td><td>univ</td><td>zaral</td><td>zara2</td></tr><tr><td>sa-tran</td><td>Transformer [13]</td><td></td><td>0.272/0.454</td><td>0.128/0.212</td><td>0.277/0.516</td><td>0.174/0.306</td><td>0.137/0.230</td></tr><tr><td>sa-ev</td><td>E-V2-Net [14]</td><td></td><td>0.234/0.369</td><td>0.111/0.171</td><td>0.246/0.437</td><td>0.177/0.311</td><td>0.137/0.236</td></tr><tr><td>sa-re</td><td>Resonance [15]</td><td></td><td>0.227/0.356</td><td>0.106/0.169</td><td>0.234/0.425</td><td>0.180/0.303</td><td>0.130/0.220</td></tr><tr><td>base</td><td>MSN [12]</td><td></td><td>0.233/0.369</td><td>0.102/0.156</td><td>0.240/0.428</td><td>0.168/0.294</td><td>0.130/0.223</td></tr></table>

The base model, i.e., Socialality uses MSN [12] as the prediction backbone, which includes a transformer network. sa-tran only uses transformer [13] as the prediction backbone, removing the MSN components. sa-ev uses E-V<sup>2</sup>-net [14] as the prediction backbone. sa-re uses Resonance [15] as the prediction backbone. Blue markers denote the best results on each set.

Table F9 Parameter counts of the proposed Socialality Model modified with diferent backbone prediction models
<table><tr><td>Model</td><td>Backbone</td><td>Parameters</td></tr><tr><td>sa-tran</td><td>Transformer [13]</td><td>2,041,413 (-1.8%)</td></tr><tr><td>sa-ev</td><td>E-V2-Net [14]</td><td>3,601,093 (+73.2%)</td></tr><tr><td>sa-re</td><td>Resonance [15]</td><td>3,996,357 (+92.2%)</td></tr><tr><td>base</td><td>MSN [12]</td><td>2,079,577</td></tr></table>

In Table. F8, we can observe that all three model variants achieve prediction performance comparable to the base Socialality on the ETH-UCY dataset. Notably, sa-tran still obtains considerable prediction performance despite removing the MSNrelated modules and reducing the overall parameter count. Compared to the base model, the performance gap between sa-tran and the base model is minimal on the zara1 subset. Furthermore, sa-re achieves the best prediction accuracy, outperforming the base model in terms of ADE/FDE on the eth, univ, and zara2 datasets. These comprehensive evaluations robustly demonstrate that the performance gains of our framework are not tightly coupled to a specific backbone (such as MSN). Instead, the group-bounded representations generated by our Socialality serve as priors that can efectively enhance diverse trajectory prediction architectures.

## Appendix G Dataset Analyses

First, we would like to analyse the distributions of pairwise distances between agents across diferent subsets. Considering the ETH-UCY dataset lacks explicit ground-truth group annotations (SDD dataset does not have such annotation, either.), we further analyse the nearest-neighbor distance distribution across all subsets. The nearestneighbor distance indicates the shortest distance during all observation steps between any neighbor and the target agent.

Table G10 Statistics for pairwise and nearest neighbor distances across ETH-UCY subsets (in meters)
<table><tr><td>Statistic</td><td>eth</td><td>hotel</td><td>univ</td><td>zara1</td><td>zara2</td></tr><tr><td>Mean (p)</td><td>5.738</td><td>4.667</td><td>6.555</td><td>5.215</td><td>4.946</td></tr><tr><td>Median (p)</td><td>4.747</td><td>4.105</td><td>6.353</td><td>4.503</td><td>4.336</td></tr><tr><td>Mean (n)</td><td>1.881</td><td>1.652</td><td>0.803</td><td>1.571</td><td>1.231</td></tr><tr><td>Median (n)</td><td>1.098</td><td>1.184</td><td>0.632</td><td>0.911</td><td>0.756</td></tr></table>

Mean (p) and Median (p) indicate pairwise mean distance and pairwise median distance. Mean (n) and Median (n) indicate nearest-neighbor mean distance and nearest-neighbor median distance.

As presented in Table. G10, we can observe that the nearest-neighbor (n) distances significantly shift across diferent subsets. For example, the median nearest-neighbor distance in the hotel and eth subsets is noticeably larger than in univ and zara2. Notably, the median nearest-neighbor distance of the hotel scene(1.184) is almost twice as large as that of the univ (0.632). This indicates that the acceptable social distance is highly sensitive to the specific scene and crowd density. The univ dataset exhibits the largest pairwise median distance, implying a larger scene where agents are widely spread out globally. However, it presents the smallest nearest-neighbor median distance. This demonstrates that agents in univ move in small groups despite the vast global space. Even within the exact same physical location, i.e., zara1 and zara2, the median nearest-neighbor distances are distinctively diferent. This might be caused by diferent recording time. The observations above highly align with the classic sociology research we cited in our original manuscript, which demonstrated that appropriate social distance varies with diferent agents.

![](images/6655666c2d2d9a35e369d2f76802f2dcdca748aace4a47cc1c588bf2e8e431a3.jpg)

![](images/70f97feb81074b17c9b83bef213f3a76cc9d65e2a64d416a87e0c733482f1f34.jpg)  
Fig. G11 Distributions of pairwise distance and nearest-neighbor distance across ETH-UCY scenes.

We also visualize the distributions of both pairwise and nearest-neighbor distances in Fig. G11. We can observe that both the pairwise distance and nearest-neighbor distance distributions difer significantly across the ETH-UCY subsets. Although the pairwise distance distributions of zara1 and zara2 are highly similar in (a), their nearest-neighbor distance distributions exhibit substantial diferences in (b). The observations above highly align with the classic sociology research we cited in our Introduction section, which demonstrated that appropriate social distance varies with diferent agents.

Classic theories in sociology argue that the appropriate social distance is culturedependent rather than universal [16]. In detail, such cultural-dependence could be interpreted from several simple but intuitive perspectives. Like people with diferent personalities usually keep diferent group-wise motion tendencies, and people from diferent culturalities also have diferent acceptable average social distance in a group.

The statistics above provide empirical evidence as to why a fixed threshold mostly fails to capture diverse grouping patterns. If we adopt a fixed spatial threshold of 1.0 meter to determine group afiliation, it will be too strict in the hotel scene, where agents naturally maintain a larger social distance. Conversely, in a dense scene like univ, applying the exact same 1.0m threshold might classify a massive number of out-of-group neighbors as in-group members. Therefore, the statistics above further validate our motivation that a context-adaptive and agent-specific grouping boundary is necessary to handle the diferent real-world scenes with diverse spatial distributions.

## References

[1] Wong, C., Zou, Z., You, X.: Encore: Conditioning trajectory forecasting via biased ego rehearsals. arXiv preprint arXiv:2605.11463 (2026)

[2] Zou, Z., Wong, C., Xia, B., You, X.: Who walks with you matters: Perceiving social interactions with groups for pedestrian trajectory prediction. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4844–4853 (2025)

[3] Linou, K., Linou, D., Boer, M.: NBA player movements. https://github.com/linouk23/NBA-Player-Movements (2016)

[4] Xu, C., Li, M., Ni, Z., Zhang, Y., Chen, S.: Groupnet: Multiscale hypergraph neural networks for trajectory prediction with relational reasoning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6498–6507 (2022)

[5] Xu, C., Mao, W., Zhang, W., Chen, S.: Remember intentions: Retrospectivememory-based trajectory prediction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6488–6497 (2022)

[6] Alahi, A., Goel, K., Ramanathan, V., Robicquet, A., Fei-Fei, L., Savarese, S.: Social lstm: Human trajectory prediction in crowded spaces. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 961–971 (2016)

[7] Gupta, A., Johnson, J., Fei-Fei, L., Savarese, S., Alahi, A.: Social gan: Socially acceptable trajectories with generative adversarial networks. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 2255– 2264 (2018)

[8] Mohamed, A., Qian, K., Elhoseiny, M., Claudel, C.: Social-stgcnn: A social spatiotemporal graph convolutional neural network for human trajectory prediction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14424–14432 (2020)

[9] Wong, C., Xia, B., Zou, Z., Wang, Y., You, X.: Socialcircle: Learning the anglebased social interaction representation for pedestrian trajectory prediction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 19005–19015 (2024)

[10] Wong, C., Xia, B., Zou, Z., You, X.: Socialcircle+: Learning the angle-based conditioned interaction representation for pedestrian trajectory prediction. arXiv preprint arXiv:2409.14984 (2024)

[11] Kim, S., Chi, H.-g., Lim, H., Ramani, K., Kim, J., Kim, S.: Higher-order relational reasoning for pedestrian trajectory prediction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15251–15260 (2024)

[12] Wong, C., Xia, B., Peng, Q., Yuan, W., You, X.: Msn: multi-style network for trajectory prediction. IEEE Transactions on Intelligent Transportation Systems 24, 9751–9766 (2023)

[13] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I.: Attention is all you need. In: Advances in Neural Information Processing Systems, pp. 5998–6008 (2017)

[14] Xia, B., Wong, C., Xu, D., Peng, Q., You, X.: Another vertical view: A hierarchical network for heterogeneous trajectory prediction via spectrums. IEEE Transactions on Pattern Analysis and Machine Intelligence (2025)

[15] Wong, C., Zou, Z., Xia, B.: Resonance: Learning to predict social-aware pedestrian trajectories as co-vibrations. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 25788–25799 (2025)

[16] Hall, E.T.: The hidden dimension. Leonardo 6(1), 94 (1973)
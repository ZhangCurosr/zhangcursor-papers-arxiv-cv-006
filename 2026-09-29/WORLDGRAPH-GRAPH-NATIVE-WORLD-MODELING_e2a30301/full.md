# WORLDGRAPH: GRAPH-NATIVE WORLD MODELING

Zezhong Ding<sup>1,3,†</sup>, Yipeng Li<sup>2,3,†</sup>, Xike Xie<sup>1,2,3∗</sup>

<sup>1</sup>School of Artificial Intelligence and Data Science, University of Science and Technology of China (USTC) <sup>2</sup>School of Biomedical Engineering, USTC

<sup>3</sup>Data Darkness Lab, Suzhou Institute for Advanced Research, USTC

{zezhongding,liyipeng0131}@mail.ustc.edu.cn, xkxie@ustc.edu.cn

## ABSTRACT

World models infer latent states of an environment to capture its underlying dynamics and predict future evolution. Many real-world environments, however, are inherently relational and observed as evolving graphs, where entities, relations, and their properties change over time. Prior graph-related world models use graph structures to organize internal states or support task-specific reasoning, rather than treating an evolving graph itself as the modeled world. We instead study graph world modeling (GWM), where graph evolution itself constitutes the world dynamics. We formulate graph world modeling over observed graph evolution, latent graph states, and heterogeneous graph-transition predictions. Based on this formulation, we construct GWM-Zero, a benchmark covering node-, edge-, and graphlevel transitions over eight temporal graph datasets. We propose WorldGraph, which combines a state-aware graph transformer for multi-granularity structural and transition-conditioned evolution modeling with transition-aware GRPO using dynamic grouping and structure-aware verifiable rewards. Extensive experiments on GWM-Zero show that WorldGraph consistently outperforms representative graph representation, temporal graph learning, graph pretraining, and graph world-model baselines across all three transition granularities.

## 1 INTRODUCTION

World models (Ha & Schmidhuber, 2018; Ding et al., 2026) learn compact internal states that capture how an environment evolves to support prediction of its future. Many real-world environments, however, are inherently relational: entities interact through structured relations, and both the entities and relations may change over time. Graphs provide a natural representation of such relational environments. More importantly, when the environment itself is observed as an evolving graph (Kurenkov et al., 2023; Savinov et al., 2018), its evolution is the world dynamics to be modeled. This motivates graph world models (GWMs) that model graph states and their evolution.

At time step t, the world provides an observed graph $g _ { t }$ , while its transition to $g _ { t + 1 }$ may involve updates to nodes, edges, properties, or their combinations. We denote the graph change associated with this transition by $\Delta g _ { t }$ , and view an evolving graph at a high level as $\{ \bar { ( g _ { t } , \Delta g _ { t } ) } \} _ { t }$ . Beyond the observed graphs, a graph world model should infer a latent world state that captures the structural and dynamic information underlying such evolution. This raises a fundamental question:

## How should a world model represent the state underlying graph evolution and predict its heterogeneous transitions?

Earlier studies have introduced graphs into world modeling mainly as auxiliary structures. $\mathrm { L ^ { 3 } P }$ (Zhang et al., 2021) abstracts Markov decision process states and their reachability into a graph for long-horizon planning. C-SWM (Kipf et al., 2020) represents objects extracted from visual observations as a latent graph and uses a GNN to model their interactions. Feng et al. (Feng et al., 2025) use graphs to organize multi-modal structure and reasoning. More recently, Graph-JEPA (Skenderi et al., 2023) improves the world model’s structured representation via masked subgraph latent prediction, and Liu et al. (Liu et al., 2026) conceptualized these graph-enhanced world models based on relational inductive biases. Ourfocus is different: we consider graph-native worlds, where the observed environment is itself an evolving graph and graph evolution is the prediction target. This calls for a graph-native formulation that explicitly models the latent world state underlying the graph evolution and predicts heterogeneous graph transitions across tasks of node, edge, and graph levels. Unlike dynamic graph learning, which typically learns temporal representations for predefined downstream tasks, a GWM maintains a latent world state that summarizes graph evolution and uses it to model heterogeneous future transitions. Node-, edge-, and graph-level predictions can then be viewed as different observable projections of the same underlying graph-world dynamics. This raises two fundamental challenges for a general GWM: representing the latent state underlying graph evolution and modeling the heterogeneous transitions that drive it.

Challenge 1: State Modeling. A graph world state should capture not only the current graph structure but also the evolution leading to it. Different transitions may depend on different structural scales and historical contexts. For example, relation formation may require broader structural context, while local property changes may depend mainly on nearby neighborhoods and recent transitions. Existing message-passing-based GNNs (Kipf & Welling, 2017; Velickoviˇ c et al., 2018;´ Hamilton et al., 2017) mainly capture local structure, while graph transformers and GFMs (Wu et al., 2026; 2022; Rampasek et al., 2022; Wang et al., 2025b) provide broader structural context but´ focus on static graph representations. Dynamic graph learning methods (Rossi et al., 2020; Peng et al., 2025) capture temporal information, but do not explicitly model a world state from graph structure and preceding transitions. The first challenge is therefore to learn a world state that jointly captures multi-granularity graph structure and historical dynamics.

Challenge 2: Transition Modeling. Graph transitions are inherently heterogeneous: they may modify node properties, relations, or local structures, while different transition patterns can occur at highly imbalanced frequencies (Liu et al., 2023). Moreover, their significance is not determined by frequency alone: a rare change involving structurally important entities may have a substantial effect on subsequent graph evolution. A general GWM should thus avoid being dominated by frequent transitions while remaining sensitive to structurally consequential changes. The second challenge is to learn heterogeneous graph transitions while accounting for both transition frequency and structural relevance.

To address these challenges, we propose WorldGraph, which jointly models latent world states and heterogeneous transitions. For Challenge 1, we develop a state-aware graph transformer that constructs the world state by integrating multi-granularity structural encoding and historyaware state encoding. It combines hop- and path-level graph contexts to capture structural dependencies at different scales, while incorporating preceding graph transitions to model how the current graph state has evolved over

![](images/fcf8f56b653f75ef14be9e8002863564b7c3dad9e67911f8290988b50e7bbf34.jpg)  
Figure 1: World Model (Ha & Schmidhuber, 2018) vs. Our Proposed GWM

time. For Challenge 2, we develop a transition-aware GRPO with dynamic grouping and structureaware verifiable rewards. Dynamic grouping allocates more training signal to rare transitions, while structure-aware rewards emphasize changes involving structurally important and historically variable entities. Together, these techniques enable WorldGraph to model graph dynamics across node-, edge-, and graph-level transitions within a unified framework.

Our contributions are threefold. 1) We formulate graph-native world modeling through observed graph states, transitions, and latent world states, and construct GWM-Zero, a benchmark covering node-, edge-, and graph-level transition prediction tasks over 8 graph datasets. 2) We propose WorldGraph, a unified framework that instantiates this formulation with a state-aware graph transformer and transition-aware RL. 3) Extensive experiments demonstrate that WorldGraph achieves an average model quality improvement of 10.77% over the strongest baselines, with the largest improvement of 50.66% in F1 score on the TGBN-Trade (Huang et al., 2023) dataset.

## 2 PROBLEM FORMULATION

We formalize graph world modeling through observed graph states, transitions, and world states.

Observed Graph Evolution. At time step t, the observed graph state is represented as $\begin{array} { r l } { g _ { t } } & { { } = } \end{array}$ $( V _ { t } , E _ { t } , X _ { t } ) \in \mathsf { \bar { \mathcal { G } } }$ , where $V _ { t } , E _ { t } ,$ and $X _ { t }$ denote the entities, relations, and associated properties, respectively. The evolution from $g _ { t }$ to $g _ { t + 1 }$ forms a graph transition. We use $\Delta g _ { t }$ to denote the graph change associated with this transition, which may involve node or edge addition/deletion, property changes, or their combinations. Accordingly, an evolving graph world can be viewed as a sequence of graph states and their changes, $\{ ( g _ { t } , \bar { \Delta g _ { t } } ) \} _ { t = 1 } ^ { T }$

Latent World States. While $g _ { t }$ describes the graph observed at time $t ,$ the underlying dynamics of graph evolution are not directly observed. We therefore introduce a world state $\mathbf { s } _ { t } ~ \in ~ S$ that summarizes both the current graph structure and the preceding evolution:

$$
\mathbf { s } _ { t } = f _ { s } ( g _ { t } , \Delta g _ { t - 1 } , \mathbf { s } _ { t - 1 } ) ,\tag{1}
$$

where $f _ { s }$ denotes the state model.

Definition 1 (Graph World Model). A graph world model (GWM) models graph-world dynamics through a state model $f _ { s }$ and a transition model $f _ { \Delta }$ . Given the current graph observation $^ { g _ { t } , }$ , the preceding graph change $\Delta g _ { t - 1 }$ , and the previous world state $\mathbf { s } _ { t - 1 }$ , the state model in-$f e r s \textbf { s } _ { t } ~ = ~ f _ { s } ( g _ { t } , \Delta g _ { t - 1 } , \mathbf { s } _ { t - 1 } )$ , while the transition model predicts the next graph change as $\Delta g _ { t } = f _ { \Delta } ( g _ { t } , \Delta g _ { t - 1 } , \mathbf { s } _ { t } ) .$

The two components capture complementary aspects of graph-world dynamics. The state model captures the latent structural and dynamic information of the evolving graph world, while the transition model predicts its subsequent evolution.

A complete graph change $\Delta g _ { t }$ may contain multiple structural events and may be observed at different granularities. We therefore instantiate graph-transition prediction at three levels: node-level changes in entities and properties, edge-level changes in relations, and graph-level changes in local structures. Importantly, these tasks are different observable projections of the same graph-world transition $\Delta g _ { t }$ rather than separate forms of the world dynamics. Accordingly, graph world modeling consists of two coupled problems: state modeling for learning the latent world state $\mathbf { s } _ { t } ,$ and transition modeling, which predicts observable graph transitions from this state.

## 3 WORLDGRAPH

WorldGraph instantiates the two components of a GWM introduced in Section 2. For state modeling, it constructs $s _ { t }$ by combining multi-granularity structure in the current graph with transition-conditioned history. For transition modeling, it learns heterogeneous graph transitions with frequency-aware sampling and structure-aware rewards. Figure 2 gives an overview of the framework.

## 3.1 STATE-AWARE GRAPH TRANSFORMER

The latent world state $s _ { t }$ should summarize both what the current graph looks like and how it arrived there. WorldGraph constructs this state in two stages: a multi-granularity graph encoder captures the current structure, and a history-aware state encoder integrates the preceding evolution.

## 3.1.1 STAGE 1: MULTI-GRANULARITY GRAPH ENCODER

Different graph transitions depend on structural information at different scales<sup>1</sup>. We combine hopbased message passing and random-walk path sampling to obtain complementary structural views.

Hop-Level Message Passing. Given a maximum receptive field constraint $( \mathrm { i . e . }$ , the maximum number of hops $L _ { \operatorname* { m a x } } )$ and the graph state $g _ { t } = ( V _ { t } , E _ { t } , \bar { X _ { t } } )$ , for every node $v \in V _ { t }$ , we obtain the hoplevel node embeddings $\{ \mathbf { x } _ { v } ^ { ( \ell ) } \} _ { 1 \le \ell \le L _ { \mathrm { m a x } } }$ , where $\mathbf { x } _ { v } ^ { ( \ell ) }$ aggregates the node features from all neighbors within the ℓ-hop range of node v by iteratively applying message passing (Hamilton et al., 2017). Random-Walk Path Sampling. To further enhance the richness of structural representations, we also perform random-walk sampling to collect M paths starting from every node $v \in V _ { t }$ . Then, for each node $v \in V _ { t }$ , we obtain the path-level embeddings $\{ \mathbf { x } _ { v } ^ { \mathrm { p a t h } _ { i } } \} _ { i = 1 } ^ { M }$ , where $\mathbf { x } _ { v } ^ { \mathrm { p a t h } _ { i } }$ <sup>i</sup> aggregates all node features along $\mathrm { p a t h } _ { i }$ by applying an attention mechanism (Vaswani et al., 2017).

![](images/511c87ead589a2764a6668d0b2aff939594509cd512ec00eef004a5618ebce90.jpg)  
Figure 2: Overview of WorldGraph. The state-aware graph transformer constructs latent world states by integrating multi-granularity graph structure with transition-conditioned history. The transition-aware RL algorithm learns heterogeneous graph transitions through dynamic grouping and structure-aware verifiable rewards across node-, edge-, and graph-level tasks.

Multi-Granularity Attention Fusion. For each node $v \in V _ { t }$ at time t, after obtaining $\{ \mathbf { x } _ { v } ^ { ( \ell ) } \} _ { \ell = 1 } ^ { L _ { \mathrm { m a x } } }$ and $\{ \mathbf { x } _ { v } ^ { \mathrm { p a t h } _ { i } } \} _ { i = 1 } ^ { M }$ , we fuse these embeddings via an attention-based aggregation<sup>2</sup> to obtain a structureaware node embedding ${ \bf z } _ { v , t } .$ . For $g _ { t }$ , we further obtain the structure-aware graph embedding $\mathbf { z } _ { t }$ by applying mean pooling over the node embeddings $\{ \mathbf { z } _ { v , t } \} _ { v \in V _ { t } }$

## 3.1.2 STAGE 2: HISTORY-AWARE STATE ENCODER

The current graph structure alone does not reveal how the graph arrived at its present state. We therefore integrate preceding graph changes into the latent world state, while accounting for both their temporal distance and their relevance to recent evolution.

Temporal Distance-Aware Encoding. Because the states that are closer in time have a greater impact on the current transition (Wang et al., 2025a), we consider incorporating the temporal distance (i.e., the difference between the current time and historical time) into the latent states at first.

Given the current time t and a historical time $\tau < t ,$ , we obtain the transition representation of $\Delta g _ { \ i }$ as follows: $\mathbf { x } _ { \Delta g _ { \tau } } = \mathrm { T y p e } ( \Delta g _ { \tau } )$ , where $\mathrm { T y p e ( \cdot ) }$ encodes the graph transition $\mathrm { t y p e } ^ { 3 }$ . Then, for every node $v \in V _ { t }$ , we calculate the node memory embeddings $\{ \mathbf { m } _ { v , \tau } \}$ for $\tau < t ^ { 4 }$ , where

$$
\begin{array} { r } { \mathbf { m } _ { v , \tau } = \mathbf { W } _ { x } \mathbf { z } _ { v , \tau } + \mathbf { W } _ { z } \mathbf { z } _ { \tau } + \mathrm { D i s t } ( t , \tau ) + \mathbf { W } _ { a } \mathbf { x } _ { \Delta g _ { \tau } } , } \end{array}\tag{2}
$$

where $\mathbf { W } _ { x } , \mathbf { W } _ { z }$ , and ${ \mathbf W } _ { a }$ are learnable weight matrices used for projection. $\operatorname { D i s t } ( t , \tau )$ encodes the relative temporal distance $t - \tau ^ { 4 }$ . Then, the node memory embedding is used to obtain the latent state. ${ \bf W } _ { a } { \bf x } _ { \Delta g _ { 1 } }$ allows the node memory embedding to incorporate the graph transition type.

History-Aware Attention Fusion. For each node v at time $t ,$ we construct a query embedding ${ \bf q } _ { v , t }$ by concatenating $\begin{array} { r } { { \bf z } _ { v , t } , { \bf z } _ { t } , { \bf x } _ { \Delta g _ { t - 1 } } , } \end{array}$ and its previous latent state $\mathbf { s } _ { v , t - 1 }$ . Then, we aggregate its historical memory $\{ \mathbf { m } _ { v , \tau } \}$ with multi-head attention. For one attention head, the attention score from history $\tau$ to the current time t is: $\begin{array} { r } { \beta _ { v , t , \tau } = \frac { ( \mathbf { q } _ { v , t } \mathbf { W } _ { Q } ) ( \mathbf { m } _ { v , \tau } \mathbf { W } _ { K } ) ^ { \top } } { \sqrt { d } } + \mathbf { p } _ { t - \tau } + \Pi ( \Delta g _ { \tau } , \Delta g _ { t - 1 } ) } \end{array}$ , where $\mathbf { W } _ { Q }$ and ${ \bf W } _ { K }$ are learnable linear projection matrices, and d is the dimension of the projected node embedding. $\Pi ( \Delta g _ { \tau } , \Delta g _ { t - 1 } ) = \left( \mathbf { x } _ { \Delta g _ { \tau } } \right) ^ { \top } \mathbf { x } _ { \Delta g _ { t - 1 } }$ is the similarity between the two transitions. This similarity is used as the reinforcement to account for how previous transitions $( \Delta g _ { \tau } )$ influence recent dynamics $( \Delta g _ { t - 1 }$ , which is the most recently observed transition).

Then, we obtain $\beta _ { v , t , \tau }$ using exponential normalization across τ. Based on this, the history embedding for node v (one attention head) is $\begin{array} { r } { \mathbf { h } _ { v , t } = \sum _ { \tau } \tilde { \beta } _ { v , t , \tau } \left( \mathbf { m } _ { v , \tau } \mathbf { W } _ { V } \right) } \end{array}$ , where $\mathbf { W } _ { V }$ is the learnable linear projection matrix. The final history embedding is obtained by mixing multiple attention heads with an MLP, yielding $\tilde { \mathbf { h } } _ { v , t }$

Finally, we sum ${ \bf z } _ { v , t } ,$ the history embedding $\tilde { \mathbf { h } } _ { v , t } .$ , and the previous latent state ${ \bf s } _ { v , t - 1 }$ to obtain the latent node state at the current time ${ \bf s } _ { v , t }$ (LayerNorm (Ba et al., 2016) is used to stabilize training):

$$
\mathbf { s } _ { v , t } = \mathrm { L a y e r N o r m } ( \tilde { \mathbf { h } } _ { v , t } + \mathbf { z } _ { v , t } + \mathbf { s } _ { v , t - 1 } ) .\tag{3}
$$

## 3.1.3 THEORETICAL ANALYSIS OF STATE-AWARE GRAPH TRANSFORMER

Here, we provide a theoretical analysis of why our state-aware graph transformer has advantages over the current state-of-the-art graph transformer, i.e., SGFormer (Wu et al., 2026), in terms of state modeling. For this analysis, we start from the message-passing mechanisms of SGFormer and WorldGraph, and then derive an upper bound on the difference between the final structure-aware node embedding they produce and the ideal structure-aware node embedding (i.e., the embedding obtained when the sampled subgraph provides full coverage).

Since SGFormer is trained using a subgraph with a fixed hop size, let the obtained subgraph be $g ^ { ( \mathrm { S G } ) }$ . In contrast, our graph transformer uses multiple-hop subgraphs and random-walk sampled subgraphs. These subgraphs are denoted as $\{ g _ { i } ^ { ( \mathrm { O U R S } ) } \} _ { i = 1 } ^ { L _ { \operatorname* { m a x } } + M }$ . Ideally, the subgraphs are generated from neighborhoods spanning arbitrarily large hop distances, and the random-walk sampling is performed with a sufficiently large number of steps. Let ζ denote the (sufficient) number of subgraphs. We obtain $\{ g _ { i } ^ { ( \mathrm { I D E A L } ) } \} _ { i = 1 } ^ { \zeta }$ . By substituting these into ${ \bf s } _ { v , t }$ , we can obtain the latent states produced by different mechanisms, which we denote as $\mathbf { s } _ { v , t } ^ { ( \mathrm { S G } ) } , \mathbf { s } _ { v , t } ^ { ( \mathrm { O U R S } ) }$ , and $\mathbf { s } _ { v , t } ^ { ( \mathrm { I D E A L } ) }$

Theorem 1. The difference between $\mathbf { s } _ { v , t } ^ { ( \mathrm { O U R S } ) }$ and $\mathbf { s } _ { v , t } ^ { ( \mathrm { I D E A L } ) }$ is $\begin{array} { r } { \Delta _ { \mathrm { O U R S } } = \left. \mathbf { s } _ { v , t } ^ { ( \mathrm { O U R S } ) } - \mathbf { s } _ { v , t } ^ { ( \mathrm { I D E A L } ) } \right. _ { 2 } } \end{array}$ and the difference between $\mathbf { s } _ { v , t } ^ { ( \mathrm { S G } ) }$ and $\mathbf { s } _ { v , t } ^ { ( \mathrm { I D E A L } ) }$ is $\Delta _ { \mathrm { S G } } = \left\| \mathbf { s } _ { v , t } ^ { ( \mathrm { S G } ) } - \mathbf { s } _ { v , t } ^ { ( \mathrm { I D E A L } ) } \right\| _ { 2 }$ . The upper bound of ∆<sub>OURS</sub> is smaller than that of $\Delta _ { \mathrm { S G } }$

Proof. The detailed proof is provided in Appendix C.1.

## 3.2 TRANSITION-AWARE REINFORCEMENT LEARNING

Given the latent world state $s _ { t } ,$ the transition model predicts heterogeneous graph changes with highly imbalanced frequencies and varying structural relevance. To account for these differences during optimization, we formulate transition learning with verifiable rewards and adapt GRPO to graph-world transitions. Transition-aware dynamic grouping emphasizes rare transitions, while structure-aware rewards account for structural relevance.

## 3.2.1 TRANSITION-AWARE DYNAMIC GROUP SAMPLING

Here, we introduce transition-aware dynamic grouping: the group assignment at time step t is biased towards rare historical transitions (i.e., the transitions that appear less frequently in the history). This design increases the coverage of low-frequency transitions, providing them with more training signal. As a result, the algorithm can learn robust policies for rare transitions.

Dynamic Transition Frequency Estimation. To achieve such a dynamic group sampling algorithm, we need to estimate the transition frequency at first. At time $t ,$ let the transition-conditioned groups be $\mathcal { T } _ { t } .$ . Each group $J _ { i } \in \mathcal { I } _ { t }$ corresponds to a specific graph transition type and stores the transitions from time steps before t that belong to this transition type. For a transition group $J _ { i } \in { \mathcal { I } } _ { t } ,$ we define the size of group $J _ { i } \ : \mathrm { a s } \ : | J _ { i } |$ . Based on $| J _ { i } |$ , we can compute the frequency of transitions of different types before time step t as $\widehat { p } ( J _ { i } )$

We then assign a rarity weight to each transition group, so that groups appearing less frequently in the prefix receive larger weights: $\omega ( J _ { i } ) = ( \widehat { p } ( J _ { i } ) ) ^ { - \gamma } , \gamma > 0 ^ { 5 }$ , and the probability of selecting group

$J _ { i }$ is Prob $\begin{array} { r } { ( J _ { i } ) = \frac { \exp ( \omega ( J _ { i } ) ) } { \sum _ { J _ { i } \in \mathcal { I } _ { t } } \exp ( \omega ( J _ { j } ) ) } . } \end{array}$

Rarity-based Rollout Budget Allocation. Finally, to make earlier rare transitions receive a larger effective group size, we allocate the total rollout budget B across groups proportionally. We reserve the same minimum number of rollouts<sup>6</sup> for every group and use largest-remainder rounding for the remaining budget, such that $\begin{array} { r } { \sum _ { J _ { i } } B _ { t } ( J _ { i } ) = B _ { i } } \end{array}$ , where $B _ { t } ( J _ { i } )$ is the allocated number of rollouts for transition group $J _ { i }$ at time t. Thus, groups corresponding to transitions that are rarer in the history are assigned more rollout samples at time t, increasing the relative training signal for these frequency-imbalanced transition patterns.

## 3.2.2 STRUCTURE-AWARE REWARD DESIGN

Graph transitions differ in their structural relevance. We therefore weight the verifiable reward using two complementary signals: structural importance and historical variability, emphasizing transitions involving more relevant nodes.

Node Importance Calculation. For a node $v \in V _ { t }$ , we define a structural importance term based on its node degree (deg(·)): $\begin{array} { r } { \mathcal { T } _ { \mathbf { s } } ( v ) = \left( \deg ( v ) \right) ^ { \theta } } \end{array}$ , where $\theta > 0 ^ { 5 }$ controls how strongly we emphasize highly-connected nodes. Let $f _ { < t } ( v )$ be the empirical frequency that node v is involved in a change $( \mathrm { i . e . }$ , the fraction of time steps at which node v is modified), and define the variability term as: $\dot { \mathcal { T } } _ { \mathbf { v } } ( v ) = ( f _ { < t } ( v ) ) ^ { \eta }$ , with $\eta \bar { > } 0 ^ { 5 }$ . We combine the two terms into a single importance: $\mathcal { T } ( v ) =$ $\begin{array} { r l } { \mathbf { \mathcal { T } _ { s } } ( v ) { \boldsymbol { \cdot } } \mathbf { \mathcal { T } _ { v } } ( v ) } & { } \\ { \frac { ( \mathbf { \mathcal { T } _ { s } } ( v ^ { \prime } ) { \boldsymbol { \cdot } } \mathbf { \mathcal { T } _ { v ^ { \prime } } } ( v ^ { \prime } ) ) } { \sum _ { v ^ { \prime } \in V _ { t } } ( \mathbf { \mathcal { T } _ { s } } ( v ^ { \prime } ) \cdot \mathbf { \mathcal { T } _ { v } } ( v ^ { \prime } ) ) } } \end{array}$ . Based on the node importance, we design a structure-aware reward.

Task-Specific Structure-Aware Reward Design. Graph-transition tasks, including node-, edge-, and graph-level tasks, produce outputs at different granularities. We therefore define a verifiable reward consistent with the output and evaluation metric. Then, we define the RL reward as follows. For node-level $\mathrm { t a s k s } ^ { 7 }$ , we denote the predicted and ground-truth change node sets by $\Delta \hat { Q }$ and $\Delta Q$ respectively. A prediction is counted as a true positive only when it exactly matches a ground-truth change. Therefore, we define the precision P, recall R, and the structure-aware reward $R _ { \mathrm { s t r u c t u r e } } { : }$

$$
\mathcal { P } = \frac { \sum _ { v \in \Delta \hat { Q } \cap \Delta Q } \mathcal { T } ( v ) } { \sum _ { v \in \Delta \hat { Q } } \mathcal { T } ( v ) } , \quad \mathcal { R } = \frac { \sum _ { v \in \Delta \hat { Q } \cap \Delta Q } \mathcal { T } ( v ) } { \sum _ { v \in \Delta Q } \mathcal { T } ( v ) } , \quad R _ { \mathrm { s t r u c t u r e } } = \frac { 2 \mathcal { P } \mathcal { R } } { \mathcal { P } + \mathcal { R } } .\tag{4}
$$

Based on this, we design rewards for node-level, edge-level, and graph-level tasks as follows. Although the three graph-world tasks operate at different granularities, their rewards share the same principle: a prediction should identify the correct structural changes and estimate the corresponding property changes. We therefore use the unified reward $R = \lambda R _ { \mathrm { s t r u c t u r e } } + ( 1 - \lambda ) R _ { \mathrm { p r o p e r t y } }$ , where $R _ { \mathrm { p r o p e r t y } }$ is calculated based on the ground-truth property and the predicted property. λ balances the two terms. The concrete definitions for node-, edge-, and graph-level tasks are provided in Appendix B.3. Based on these rewards, we train the policy network $\Delta g _ { t } = f _ { \Delta } ( g _ { t } , \Delta \bar { g } _ { t - 1 } , \mathbf { s } _ { t } ) ^ { 8 }$ to maximize the expected cumulative return and obtain the transition $\Delta g _ { t }$ at time step t.

## 3.2.3 THEORETICAL ANALYSIS OF TRANSITION-AWARE RL ALGORITHM

In GRPO (Shao et al., 2024), the group size is the same for different transitions. The distribution of group sizes across different transitions can be viewed as uniform. In our transition-aware RL algorithm, our transition group-size distribution is determined based on the historical interaction frequency of each transition. Following (Kim & Lee, 2026), we use the effective variance ratio (EVR). A smaller EVR means lower gradient noise and more stable RL training.

Theorem 2. In GRPO, the group size is $\rho ^ { \mathrm { G R P O } } = C _ { a } ,$ , where $C _ { a }$ is a constantfor different graph transition types. In our RL algorithm, the group size corresponding to each transition type (the corresponding group is J<sub>i</sub>) is $\begin{array} { r } { \bar { \rho } ^ { \mathrm { O U R S } } = C _ { a } + \bar { \mathrm { P r o b } } ( J _ { i } ) \cdot \mathcal { B } . } \end{array}$ . Based on (Kim & Lee, 2026), for a group size $\rho _ { i } ,$ , an upper bound on EVR is $\begin{array} { r } { \mathrm { E V R } ( \rho _ { i } ) = 1 + \frac { \delta - 1 } { 2 \rho _ { i } } + M _ { \mathrm { E V R } } } \end{array}$ , where M<sub>EVR</sub> $\ge \mathcal { O } \left( \textstyle { \frac { 1 } { \rho _ { i } ^ { 2 } } } \right)$ is a constant and δ is the kurtosis of the reward distribution (Kim & Lee, 2026). Then, for a given transition type, the group produced by our RL algorithm has a smaller upper bound on the EVR than the group produced by GRPO.

Proof. The detailed proof is provided in Appendix C.2.

Table 1: Overall Performance on the 3 Graph-Transition Tasks. Add., Rem., and Sem. denote addition, deletion, and property-change F1, respectively. Higher is better for F1, Macro-F1, and NDCG@10, while lower is better for MAE and RMSE. The best results are shown in bold and the second best results are underlined. We use an existing Controller design (Dinella et al., 2020) to enable the baselines in the first three categories to predict node-, edge-, and graph-level changes.  
(a) Node-Level Tasks
<table><tr><td rowspan="2">Method</td><td rowspan="2">Type</td><td colspan="4">TGBN-Trade</td><td colspan="4"></td><td colspan="4"></td></tr><tr><td>Add. F1 Rem. F1 Sem. F1 NDCG@10|</td><td></td><td></td><td></td><td></td><td></td><td></td><td>Add. F1 Rem. F1 Sem. F1 NDCG@10|</td><td>Add. F1 Rem. F1 Sem. F1</td><td></td><td></td><td>NDCG@10</td></tr><tr><td>GCN</td><td>|MPNN</td><td>0.4718</td><td>0.1540</td><td>0.5741</td><td>0.6357</td><td>0.7112</td><td>0.8488</td><td>0.6375</td><td>0.2469</td><td>0.9706</td><td>0.8976</td><td>0.6284</td><td>0.2161</td></tr><tr><td>GAT</td><td>MPNN</td><td>0.5248</td><td>0.1385</td><td>0.6275</td><td>0.6386</td><td>0.7253</td><td>0.8598</td><td>0.6523</td><td>0.2699</td><td>0.9720</td><td>0.8972</td><td>0.6440</td><td>0.2237</td></tr><tr><td>GraphSAGE</td><td>MPNN</td><td>0.5864</td><td>0.0000</td><td>0.6595</td><td>0.6246</td><td>0.7256</td><td>0.8538</td><td>0.6496</td><td>0.2780</td><td>0.9705</td><td>0.8902</td><td>0.6435</td><td>0.2519</td></tr><tr><td>SGFormer</td><td>Graph Transformer</td><td>0.6874</td><td>0.0308</td><td>0.6732</td><td>0.6324</td><td>0.7252</td><td>0.8564</td><td>0.6497</td><td>0.2822</td><td>0.9731</td><td>0.9024</td><td>0.6445</td><td>0.2413</td></tr><tr><td>NodeFormer</td><td>Graph Transformer</td><td>0.7069</td><td>0.0878</td><td>0.6669</td><td>0.6336</td><td>0.7080</td><td>0.8394</td><td>0.6470</td><td>0.2344</td><td>0.8011</td><td>0.7329</td><td>0.5409</td><td>0.1275</td></tr><tr><td>GraphGPS</td><td>Graph Transformer</td><td>0.8033</td><td>0.0000</td><td>0.6542</td><td>0.6155</td><td>0.7302</td><td>0.8596</td><td>0.6533</td><td>0.2824</td><td>0.9684</td><td>0.8931</td><td>0.6449</td><td>0.2376</td></tr><tr><td>Graph-JEPA</td><td>Pretrained</td><td>0.5585</td><td>0.0419</td><td>0.5777</td><td>0.6357</td><td>0.6563</td><td>0.7634</td><td>0.6143</td><td>0.2487</td><td>0.9573</td><td>0.8861</td><td>0.5999</td><td>0.1520</td></tr><tr><td>MDGFM</td><td>Pretrained</td><td>0.4497</td><td>0.1517</td><td>0.5616</td><td>0.6406</td><td>0.7194</td><td>0.8426</td><td>0.6177</td><td>0.2107</td><td>0.9558</td><td>0.8607</td><td>0.6151</td><td>0.1676</td></tr><tr><td>TGN</td><td>Temporal Model</td><td>0.8492</td><td>0.0559</td><td>0.6666</td><td>0.6181</td><td>0.7378</td><td>0.8698</td><td>0.6505</td><td>0.2683</td><td>0.9726</td><td>0.8982</td><td>0.6457</td><td>0.2259</td></tr><tr><td>TIDFormer</td><td>Temporal Model</td><td>0.8442</td><td>0.1129</td><td>0.6740</td><td>0.6299</td><td>0.7409</td><td>0.8754</td><td>0.6529</td><td>0.2556</td><td>0.9705</td><td>0.9066</td><td>0.6429</td><td>0.2150</td></tr><tr><td>WorldGraph</td><td>|GWM</td><td>0.9245</td><td>0.6430</td><td>0.7117</td><td>0.6484</td><td>0.7460</td><td>0.8801</td><td>0.6557</td><td>0.4418</td><td>0.9755</td><td>0.9317</td><td>0.6458</td><td>0.3838</td></tr><tr><td>WorldGraph-Pretrain</td><td>GWM</td><td>0.9696</td><td>0.6606</td><td>0.7354</td><td>0.6469</td><td>0.7474</td><td>0.8849</td><td>0.6685</td><td>0.4448</td><td>0.9779</td><td>0.9561</td><td>0.6492</td><td>0.3901</td></tr></table>

(b) Edge-Level Tasks
<table><tr><td rowspan="2">Method</td><td rowspan="2">Type</td><td colspan="3">TGBN-Trade</td><td colspan="3">UN Vote</td><td colspan="3">Contact</td><td colspan="3">SocialEvo</td></tr><tr><td></td><td></td><td>|Add. F1 Rem. F1 Macro-F1 |</td><td>Add. F1</td><td>Rem. F1 Macro-F1</td><td></td><td></td><td>Add. F1 Rem. F1 Macro-F1</td><td></td><td></td><td>Add. F1 Rem. F1 Macro-F1</td><td></td></tr><tr><td>GCN</td><td>|MPNN</td><td>0.2633</td><td>0.2973</td><td>0.2803</td><td>0.1337</td><td>0.1025</td><td>0.1181</td><td>0.0146</td><td>0.7946</td><td>0.4046</td><td>0.2387</td><td>0.5554</td><td>0.3970</td></tr><tr><td>GAT</td><td>MPNN</td><td>0.2967</td><td>0.3841</td><td>0.3404</td><td>0.1393</td><td>0.0926</td><td>0.1159</td><td>0.0146</td><td>0.7518</td><td>0.3832</td><td>0.2337</td><td>0.5897</td><td>0.4117</td></tr><tr><td>GraphSAGE</td><td>MPNN</td><td>0.3114</td><td>0.3865</td><td>0.3489</td><td>0.1649</td><td>0.1031</td><td>0.1340</td><td>0.0170</td><td>0.8174</td><td>0.4172</td><td>0.2629</td><td>0.5556</td><td>0.4093</td></tr><tr><td>SGFormer</td><td>Graph Transformer</td><td>0.3174</td><td>0.3915</td><td>0.3544</td><td>0.1403</td><td>0.1058</td><td>0.1230</td><td>0.0141</td><td>0.8144</td><td>0.4142</td><td>0.2175</td><td>0.5233</td><td>0.3704</td></tr><tr><td>NodeFormer</td><td>Graph Transformer</td><td>0.3031</td><td>0.3797</td><td>0.3414</td><td>0.1610</td><td>0.0840</td><td>0.1225</td><td>0.0124</td><td>0.5443</td><td>0.2784</td><td>0.1532</td><td>0.3496</td><td>0.2514</td></tr><tr><td>GraphGPS</td><td>Graph Transformer</td><td>0.3153</td><td>0.3926</td><td>0.3540</td><td>0.1270</td><td>0.0952</td><td>0.1111</td><td>0.0272</td><td>0.8847</td><td>0.4559</td><td>0.2611</td><td>0.5198</td><td>0.3905</td></tr><tr><td>Graph-JEPA</td><td>Pretrained</td><td>0.2571</td><td>0.2933</td><td>0.2752</td><td>0.1308</td><td>0.1083</td><td>0.1196</td><td>0.0165</td><td>0.8659</td><td>0.4412</td><td>0.2180</td><td>0.5312</td><td>0.3746</td></tr><tr><td>MDGFM</td><td>Pretrained</td><td>0.2187</td><td>0.2975</td><td>0.2581</td><td>0.1188</td><td>0.1041</td><td>0.1114</td><td>0.0112</td><td>0.7174</td><td>0.3643</td><td>0.2285</td><td>0.5184</td><td>0.3734</td></tr><tr><td>TGN</td><td>Temporal Model</td><td>0.3202</td><td>0.4010</td><td>0.3606</td><td>0.1886</td><td>0.1112</td><td>0.1499</td><td>0.0221</td><td>0.8776</td><td>0.4498</td><td>0.2518</td><td>0.5691</td><td>0.4104</td></tr><tr><td>TIDFormer</td><td>Temporal Model</td><td>0.3217</td><td>0.4005</td><td>0.3611</td><td>0.2132</td><td>0.3089</td><td>0.2611</td><td>0.0329</td><td>0.8583</td><td>0.4456</td><td>0.2651</td><td>0.5219</td><td>0.3935</td></tr><tr><td>WorldGraph</td><td>|GWM</td><td>0.3952</td><td>0.5325</td><td>0.4640</td><td>0.4195</td><td>0.5395</td><td>0.4795</td><td>0.1628</td><td>0.9009</td><td>0.5319</td><td>0.3841</td><td>0.6567</td><td>0.5204</td></tr><tr><td>WorldGraph-Pretrain</td><td>GWM</td><td>0.3948</td><td>0.5344</td><td>0.4646</td><td>0.4165</td><td>0.5996</td><td>0.5080</td><td>0.1819</td><td>0.8991</td><td>0.5405</td><td>0.3890</td><td>0.6609</td><td>0.5249</td></tr></table>

(c) Graph-Level Tasks
<table><tr><td rowspan="2">Method</td><td rowspan="2">Type</td><td rowspan="2">MAE</td><td>Flights RMSE</td><td rowspan="2"></td><td rowspan="2">MAE</td><td rowspan="2">Contact RMSE</td><td rowspan="2">Macro-F1</td><td rowspan="2">MAE</td><td rowspan="2">Enron RMSE</td><td rowspan="2">Macro-F1</td></tr><tr><td>Macro-F1</td></tr><tr><td>GCN</td><td>MPNN</td><td>0.8047</td><td>1.2736</td><td>0.4902</td><td>0.6842</td><td>0.9884</td><td>0.4357</td><td>1.1674</td><td>1.7609</td><td>0.3955</td></tr><tr><td>GAT</td><td>MPNN</td><td>0.8354</td><td>1.3303</td><td>0.4173</td><td>0.7470</td><td>1.0338</td><td>0.4238</td><td>1.1887</td><td>1.8048</td><td>0.3724</td></tr><tr><td>GraphSAGE</td><td>MPNN</td><td>0.8163</td><td>1.2984</td><td>0.4763</td><td>0.6811</td><td>0.9832</td><td>0.4402</td><td>1.1702</td><td>1.7731</td><td>0.4052</td></tr><tr><td>SGFormer</td><td>Graph Transformer</td><td>0.8043</td><td>1.2768</td><td>0.4837</td><td>0.6823</td><td>0.9842</td><td>0.4334</td><td>1.1533</td><td>1.7428</td><td>0.3996</td></tr><tr><td>NodeFormer</td><td>Graph Transformer</td><td>0.7941</td><td>1.2835</td><td>0.3974</td><td>0.7554</td><td>1.0403</td><td>0.3970</td><td>1.1573</td><td>1.7733</td><td>0.4139</td></tr><tr><td>GraphGPS</td><td>Graph Transformer</td><td>0.7723</td><td>1.2482</td><td>0.4826</td><td>0.6378</td><td>0.9452</td><td>0.4482</td><td>1.1713</td><td>1.7567</td><td>0.4090</td></tr><tr><td>Graph-JEPA</td><td>Pretrained</td><td>0.8407</td><td>1.3243</td><td>0.4413</td><td>0.7109</td><td>1.0177</td><td>0.4160</td><td>1.2211</td><td>1.7961</td><td>0.3483</td></tr><tr><td>MDGFM</td><td>Pretrained</td><td>0.8440</td><td>1.3357</td><td>0.3904</td><td>0.7385</td><td>1.0326</td><td>0.4001</td><td>1.2257</td><td>1.8545</td><td>0.3703</td></tr><tr><td>TGN</td><td>Temporal Model</td><td>0.7956</td><td>1.2615</td><td>0.4796</td><td>0.6659</td><td>0.9713</td><td>0.4151</td><td>1.1724</td><td>1.7767</td><td>0.4155</td></tr><tr><td>TIDFormer</td><td>Temporal Model</td><td>0.7929</td><td>1.2480</td><td>0.4759</td><td>0.6935</td><td>0.9955</td><td>0.4250</td><td>1.1980</td><td>1.8047</td><td>0.4102</td></tr><tr><td>WorldGraph</td><td>|GWM</td><td>0.7069</td><td>1.1254</td><td>0.5840</td><td>0.5648</td><td>0.8972</td><td>0.5438</td><td>1.1103</td><td>1.5447</td><td>0.4255</td></tr><tr><td>WorldGraph-Pretrain</td><td>GWM</td><td>0.6841</td><td>1.0985</td><td>0.5891</td><td>0.5540</td><td>0.8865</td><td>0.5547</td><td>1.0983</td><td>1.5383</td><td>0.4244</td></tr></table>

## 4 EXPERIMENTS

In this section, we conduct extensive experiments to answer the following research questions: (RQ1): Compared with existing state-of-the-art methods for solving graph world modeling problems, does WorldGraph have advantages in terms of model quality?

(RQ2): How effective are the components of WorldGraph?

(RQ3): How sensitive is WorldGraph to its key model configurations?

(RQ4): Is it effective to integrate WorldGraph into the existing graph world model problem settings?

## 4.1 EXPERIMENTAL SETUP

Benchmark. We construct the graph world benchmark, GWM-Zero, at three levels, including node, edge, and graph, to comprehensively evaluate WorldGraph. The benchmark is built from TGBN-Trade (Huang et al., 2023), TGBN-Genre (Huang et al., 2023), TGBN-Reddit (Huang et al., 2023), UN Vote (Poursafaei et al., 2022), Flights (Poursafaei et al., 2022), Contact (Poursafaei et al., 2022), SocialEvo (Poursafaei et al., 2022), and Enron (Poursafaei et al., 2022).

Baselines. We select 13 baselines from four perspectives. (1) From the graph-representation perspective, we include message passing-based neural networks (MPNN)—GCN (Kipf & Welling,

![](images/ae0db0b2434dc7805680304d7e6641d2699c88dda4f48ca7f3e9284561a60c45.jpg)  
Figure 3: Ablation Study on Node-Level Tasks/TGBN-Trade and Graph-Level Tasks/Enron. The left panel reports the equal-weight average of the 4 TGBN-Trade metrics, while the right panel reports Macro-F1.

2017), GAT (Velickovi ˇ c et al., 2018), and GraphSAGE (Hamilton et al., 2017)—as well ´ as graph Transformers—SGFormer (Wu et al., 2026), NodeFormer (Wu et al., 2022), and GraphGPS (Rampasek et al., 2022). (2) From the temporal-modeling perspective, we compare with´ TGN (Rossi et al., 2020) and TIDFormer (Peng et al., 2025), which explicitly model time-stamped interactions and temporal dependencies. (3) From the graph-pretraining perspective, we include Graph-JEPA (Skenderi et al., 2023) and MDGFM (Wang et al., 2025b) to evaluate the benefit of self-supervised and multi-domain pretraining. (4) Under existing world-modeling problem settings, we include L<sup>3</sup>P (Zhang et al., 2021), C-SWM (Kipf et al., 2020), and GWM-E (Feng et al., 2025).

## 4.2 OVERALL PERFORMANCE (RQ1)

As shown in Table 1, across all 3 task levels, WorldGraph and WorldGraph-Pretrain (Pretraining details are provided in Appendix G.1.) consistently outperform the baselines. For node-level tasks, the gains are particularly pronounced for node deletion on TGBN-Trade, where sparse deletion events cause GraphSAGE and GraphGPS to incorrectly predict the deletion nodes, yielding Rem. F1 = 0, while WorldGraph-Pretrain reaches Rem. F1 = 0.6606, an absolute improvement of 50.66% over the second best method. For semantic-property prediction, WorldGraph-Pretrain improves over the second best methods by 16.24% on TGBN-Genre and 13.82% on TGBN-Reddit, showing that jointly modeling graph structure, historical states, and graph transitions is beneficial for heterogeneous node transition prediction tasks. For edge-level tasks, the advantage is more evident, with an average Macro-F1 improvement of 13.72%. Compared with the strongest baselines on the corresponding dataset, its improvements on UN Vote (↑24.69%), Contact (↑8.46%), and SocialEvo (↑11.37%) demonstrate that the learned transition policy can capture the graph changes that are difficult for representation-only and temporal baselines. For graph-level tasks, WorldGraph also obtains the lowest MAE and RMSE and the highest Macro-F1, with average absolute reductions of 0.0757 and 0.1376 in MAE and RMSE, respectively, and an average Macro-F1 improvement of 7.18% over the best external baseline for each dataset and metric. These indicate that its advantage extends from discrete node and edge edits to continuous local-structure evolution.

## 4.3 ABLATION STUDY (RQ2)

We evaluate five variants that respectively remove the multi-granularity graph encoder, the historyaware state encoder, the transition-aware RL mechanism, as well as its two internal components: dynamic group sampling and structure-aware reward. Figure 3 summarizes their overall performance on Node-Level Tasks/TGBN-Trade and Graph-Level Tasks/Enron. Removing any component degrades performance on both tasks. The largest reductions are caused by removing the multigranularity graph encoder or the history-aware state encoder, which lead to absolute drops of 8.83% and 10.20% on Node-Level Tasks/Trade and 2.04% and 2.11% on Graph-Level Tasks/Enron, respectively, confirming that both current graph structure and historical evolution are essential for representing a graph world. Removing the complete RL mechanism, or removing either of its two proposed components—dynamic group sampling or structure-aware reward—also consistently reduces performance, with absolute drops of 6.27/0.85%, 4.19/0.55%, and 4.84/0.36%, respectively. The results show that the two components make complementary contributions to the final performance.

## 4.4 SENSITIVITY ANALYSIS (RQ3)

Figure 4 shows the sensitivity of World-Graph to three key hyperparameters on Graph-Level Tasks/Flights (Macro-F1). Overall, the performance varies by only 0.31 percentage points across all settings (ranging narrowly from 58.09% to 58.40%), demonstrating that WorldGraph is robust to these hyperparameter configurations. Specifically, varying the maximm hop number mum hop number $L _ { \mathrm { m a x } }$ from 1 to 4 leads

![](images/c60ab1ed8fe16ef7fd32d626344c15d4b99f946c0f4f8a5a946da419c6d87b6e.jpg)

![](images/1a6c46ab162e0f4f64220b370c33856d6d352d3a4fbb612f2083c312b4193de2.jpg)

![](images/8cc24e1f6c5578484fc9acc63eb1aac3bb4eccd6d2b5e3a9c24b3200e63f4409.jpg)  
Figure 4: Sensitivity Analysis on Graph-Level Tasks/Flights. Performance remains remarkably stable across varying maximum hop numbers $L _ { \mathrm { m a x } }$ , random-walk numbers M, and GRPO rollout budgets $B .$

from 1 to 4 leads to small fluctuations between 58.09% and 58.40%, with $L _ { \operatorname* { m a x } { } } = 2$ achieving the peak performance. The random-walk number M also exhibits a very marginal impact: the Macro-F1 score remains virtually flat within the range of 58.27%–58.40% across $M \in \mathsf { \bar { \{ 2 , 4 , 6 , 8 \} } }$ . For the GRPO rollout budget B, performance stays stably within 58.28%– 58.40% without noticeable deviation as B increases from 8 to 14. These results confirm that World-Graph is not sensitive to the choice of these structural and optimization hyperparameters, further validating its stability.

## 4.5 COMPARISON ON EXISTING GWM PROBLEM SETTINGS (RQ4)

For $\mathrm { L ^ { 3 } P , }$ we evaluate long-horizon planning on PointMaze (Duan et al., 2016) and FetchPickAnd-Place (Plappert et al., 2018). Integrating World-Graph as a plug-in module provides graph-structured state and history information to the landmark planner while leaving the original environment, actor, and training objective unchanged. The resulting learning curves are shown in Figure 5. Over the final ten test epochs, L<sup>3</sup>P + WorldGraph leads L<sup>3</sup>P by 7.11 percentage points on PointMaze (94.88% vs. 87.77%) and by 2.43 points on FetchPickAndPlace (99.18% vs. 96.74%).

For C-SWM, we evaluate multi-step latent-state ranking on Pong (Brockman et al., 2016) and Space Invaders (SI) (Brockman et al., 2016) environments (env.) at 1, 5, and 10 steps. The plug-in augments C-SWM’s objectcentric transition model with graphaware state and transition information, while retaining its original object encoder, transition backbone, and contrastive training objective. As shown in Table 2, WorldGraph improves both H@1 and MRR at every horizon on both games. The gains become larger at longer horizons, reaching 31.33 in H@1 and 29.55 in MRR at 10 steps on SI.

![](images/7bf5c61154e488f221cab4a0985fe25d13ca16f46a469b91da35588ea051b21f.jpg)  
Figure 5: Comparison with $\mathbf { L ^ { 3 } P }$ on PointMaze and FetchPickAndPlace. Solid curves show the mean test success rate and shaded regions show the standard deviation.

Table 2: Comparison with C-SWM on Multi-Step Latent-State Ranking Tasks. Higher H@1 and MRR are better.
<table><tr><td rowspan="2">Env.</td><td rowspan="2">Model</td><td colspan="2">1 Step</td><td colspan="2">5 Steps</td><td colspan="2">10 Steps</td></tr><tr><td>H@1</td><td>MRR</td><td>H@1</td><td>MRR</td><td>H@1</td><td>MRR</td></tr><tr><td rowspan="2">Pong</td><td>C-SWM</td><td>35.00</td><td>51.45</td><td>11.33</td><td>27.24</td><td>6.67</td><td>19.03</td></tr><tr><td>+ WorldGraph</td><td>40.33</td><td>57.68</td><td>26.33</td><td>45.85</td><td>16.00</td><td>34.62</td></tr><tr><td rowspan="2">SI</td><td>C-SWM</td><td>63.67</td><td>75.49</td><td>41.33</td><td>56.66</td><td>29.67</td><td>46.16</td></tr><tr><td>+ WorldGraph</td><td>68.00</td><td>79.68</td><td>62.67</td><td>76.04</td><td>61.00</td><td>75.71</td></tr></table>

Table 3: Comparison with GWM-E on Node Classification (NC), Link Prediction (LP), and Graph Classification (GC) tasks (Accuracy/%). Higher accuracy is better.
<table><tr><td rowspan="2">Model</td><td colspan="2">Cora</td><td colspan="2">PubMed</td><td rowspan="2">HIV GC</td></tr><tr><td>NC</td><td> $\mathrm { L P }$ </td><td> $\mathrm { N C }$ </td><td>LP</td></tr><tr><td>GWM-E</td><td>83.03</td><td>94.31</td><td>84.22</td><td>94.01</td><td>93.86</td></tr><tr><td>WorldGraph</td><td>88.93</td><td>95.17</td><td>89.34</td><td>97.04</td><td>96.49</td></tr></table>

For comparison with GWM-E, we use its traditional graph prediction setting on Cora (McCallum et al., 2000), PubMed (Namata et al., 2012), and HIV (Wu et al., 2018), covering node classification, link prediction, and graph classification. We independently train WorldGraph on these tasks with its multi-granularity graph encoder and task-specific prediction heads. WorldGraph achieves higher accuracy in all five columns, with absolute gains of 5.90 and 0.86 on Cora node classification and link prediction, 5.12 and 3.03 on the corresponding PubMed tasks, and 2.63 on HIV graph classification.

## 5 CONCLUSION

In this paper, we study graph world modeling by formalizing latent world states and heterogeneous graph transitions from evolving graph observations. We introduce GWM-Zero to cover node-, edge-, and graph-level transition prediction across eight temporal graph datasets, and propose WorldGraph, which uses a state-aware graph transformer for multi-granularity state modeling and a transitionaware GRPO with dynamic grouping and structure-aware verifiable rewards for learning under frequency imbalance. Extensive experiments show that WorldGraph consistently achieves stronger performance across diverse multi-granularity graph transition prediction tasks. WorldGraph provides a general and effective framework for graph-native world modeling and opens up promising directions for learning richer latent dynamics and more controllable graph evolution in future work.

## REFERENCES

Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer normalization. 2016. URL https://arxiv.org/abs/1607.06450.

Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. Openai gym. 2016. URL https://arxiv.org/abs/1606.01540.

Maxime Burchi and Radu Timofte. Learning transformer-based world models with contrastive predictive coding. In International Conference on Learning Representations, volume 2025, pp. 6949–6970, 2025.

Weilin Cong, Si Zhang, Jian Kang, Baichuan Yuan, Hao Wu, Xin Zhou, Hanghang Tong, and Mehrdad Mahdavi. Do we really need complicated model architectures for temporal networks? In International Conference on Learning Representations, 2023.

Tal Daniel, Carl Qi, Dan Haramati, Amir Zadeh, Chuan Li, Aviv Tamar, Deepak Pathak, and David Held. Latent particle world models: Self-supervised object-centric stochastic dynamics modeling. 2026. URL https://arxiv.org/abs/2603.04553.

Elizabeth Dinella, Hanjun Dai, Ziyang Li, Mayur Naik, Le Song, and Ke Wang. Hoppity: Learning graph transformations to detect and fix bugs in programs. In International conference on learning representations, 2020.

Jingtao Ding, Yunke Zhang, Yu Shang, Yuheng Zhang, Zefang Zong, Jie Feng, Yuan Yuan, Hongyuan Su, Nian Li, Nicholas Sukiennik, Fengli Xu, and Yong Li. Understanding world or predicting future? A comprehensive survey of world models. ACM Comput. Surv., 58(3): 57:1–57:38, 2026.

Yan Duan, Xi Chen, Rein Houthooft, John Schulman, and Pieter Abbeel. Benchmarking deep reinforcement learning for continuous control. In International conference on machine learning, pp. 1329–1338. PMLR, 2016.

Tao Feng, Yexin Wu, Guanyu Lin, and Jiaxuan You. Graph world model. In International Conference on Machine Learning, volume 267, 2025.

David Ha and Jurgen Schmidhuber. World models. 2018. doi: 10.5281/ZENODO.1207631. URL¨ https://zenodo.org/record/1207631.

Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning latent dynamics for planning from pixels. In International conference on machine learning, pp. 2555–2565. PMLR, 2019.

Danijar Hafner, Timothy P. Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. In International Conference on Learning Representations, 2020.

William L. Hamilton, Rex Ying, and Jure Leskovec. Inductive representation learning on large graphs. In Advances in Neural Information Processing Systems, volume 30, 2017.

Nick Hansen, Hao Su, and Xiaolong Wang. Td-mpc2: Scalable, robust world models for continuous control. In International Conference on Learning Representations, volume 2024, pp. 47376– 47405, 2024.

Shenyang Huang, Farimah Poursafaei, Jacob Danovitch, Matthias Fey, Weihua Hu, Emanuele Rossi, Jure Leskovec, Michael M. Bronstein, Guillaume Rabusseau, and Reihaneh Rabbany. Temporal graph benchmark for machine learning on temporal graphs. Advances in Neural Information Processing Systems, 36, 2023.

Yinan Huang, William Lu, Joshua Robinson, Yu Yang, Muhan Zhang, Stefanie Jegelka, and Pan Li. On the stability of expressive positional encodings for graphs. In International Conference on Learning Representations, 2024.

Yaning Jia, Chunhui Zhang, and Soroush Vosoughi. Aligning relational learning with lipschitz fairness. In International Conference on Learning Representations, 2024.

Taehyeon Kim and Kyung-Taek Lee. Efficiency-aware group size optimization for grpo via multifidelity bayesian optimization. AI, 7(7):234, 2026.

Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations, 2017.

Thomas N. Kipf, Elise van der Pol, and Max Welling. Contrastive learning of structured world models. In 8th International Conference on Learning Representations, 2020.

Srijan Kumar, Xikun Zhang, and Jure Leskovec. Predicting dynamic embedding trajectory in temporal interaction networks. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining, pp. 1269–1278, 2019.

Andrey Kurenkov, Michael Lingelbach, Tanmay Agarwal, Emily Jin, Chengshu Li, Ruohan Zhang, Li Fei-Fei, Jiajun Wu, Silvio Savarese, and Roberto Mart´ın-Mart´ın. Modeling dynamic environments with scene graph memory. In International Conference on Machine Learning, volume 202, pp. 17976–17993. PMLR, 2023.

Jiawei Liu, Senqiao Yang, Mingjun Wang, Yu Wang, and Bei Yu. Graph world models: Concepts, taxonomy, and future directions, 2026. URL https://arxiv.org/abs/2604.27895.

Junfeng Liu, Min Zhou, Shuai Ma, and Lujia Pan. Mata\*: Combining learnable node matching with a\* algorithm for approximate graph edit distance computation. In Proceedings of the 32nd ACM international conference on information and knowledge management, pp. 1503–1512, 2023.

Xiaodong Lu, Leilei Sun, Tongyu Zhu, and Weifeng Lv. Improving temporal link prediction via temporal walk matrix projection. Advances in Neural Information Processing Systems, 37:141153– 141182, 2024.

Sohir Maskey, Raffaele Paolino, Fabian Jogl, Gitta Kutyniok, and Johannes F. Lutzeyer. Graph representational learning: When does more expressivity hurt generalization? In International Conference on Learning Representations, 2026.

Andrew Kachites McCallum, Kamal Nigam, Jason Rennie, and Kristie Seymore. Automating the construction of internet portals with machine learning. Information Retrieval, 3(2):127–163, 2000.

Galileo Namata, Ben London, Lise Getoor, Bert Huang, and U Edu. Query-driven active surveying for collective classification. In 10th international workshop on mining and learning with graphs, volume 8, pp. 1, 2012.

Jie Peng, Zhewei Wei, and Yuhang Ye. TIDFormer: Exploiting temporal and interactive dynamics makes a great dynamic graph transformer. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pp. 2245–2256, 2025.

Matthias Plappert, Marcin Andrychowicz, Alex Ray, Bob McGrew, Bowen Baker, Glenn Powell, Jonas Schneider, Josh Tobin, Maciek Chociej, Peter Welinder, Vikash Kumar, and Wojciech Zaremba. Multi-goal reinforcement learning: Challenging robotics environments and request for research. 2018. URL https://arxiv.org/abs/1802.09464.

Farimah Poursafaei, Shenyang Huang, Kellin Pelrine, and Reihaneh Rabbany. Towards better evaluation for dynamic link prediction. Advances in Neural Information Processing Systems, 35: 32928–32941, 2022.

Ladislav Rampasek, Michael Galkin, Vijay Prakash Dwivedi, Anh Tuan Luu, Guy Wolf, and Do-´ minique Beaini. Recipe for a general, powerful, scalable graph transformer. In Advances in Neural Information Processing Systems, 2022.

Emanuele Rossi, Ben Chamberlain, Fabrizio Frasca, Davide Eynard, Federico Monti, and Michael M. Bronstein. Temporal graph networks for deep learning on dynamic graphs, 2020. URL https://arxiv.org/abs/2006.10637.

Nikolay Savinov, Alexey Dosovitskiy, and Vladlen Koltun. Semi-parametric topological memory for navigation. In International Conference on Learning Representations, 2018.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/2402. 03300.

Weining Shi, Zhisen Wen, Qinggang Zhang, Chentao Zhang, and Zhihong Zhang. Temporal graph thumbnail: Robust representation learning with global evolutionary skeleton. In International Conference on Learning Representations, volume 2026, pp. 137964–137999, 2026.

Geri Skenderi, Hang Li, Jiliang Tang, and Marco Cristani. Graph-level representation learning with joint-embedding predictive architectures, 2023. URL https://arxiv.org/abs/2309. 16014.

Yuxing Tian, Yiyan Qi, and Fan Guo. Freedyg: Frequency enhanced continuous-time dynamic graph model for link prediction. In International Conference on Learning Representations, volume 2024, pp. 8431–8450, 2024.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Petar Velickoviˇ c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Li´ o, and Yoshua\` Bengio. Graph attention networks. In International Conference on Learning Representations, 2018.

Cheng Wan, Youjie Li, Ang Li, Nam Sung Kim, and Yingyan Lin. Bns-gcn: Efficient full-graph training of graph convolutional networks with partition-parallelism and random boundary node sampling. Proceedings of Machine Learning and Systems, 4:673–693, 2022.

Peihao Wang, Ruisi Cai, Yuehao Wang, Jiajun Zhu, Pragya Srivastava, Zhangyang Wang, and Pan Li. Understanding and mitigating bottlenecks of state space models through the lens of recency and over-smoothing. In International Conference on Learning Representations, volume 2025, pp. 82314–82342, 2025a.

Shuo Wang, Bokui Wang, Zhixiang Shen, Boyan Deng, and Zhao Kang. Multi-domain graph foundation models: Robust knowledge transfer via topology alignment. In Proceedings of the 42nd International Conference on Machine Learning, 2025b.

Yanbang Wang, Yen-Yu Chang, Yunyu Liu, Jure Leskovec, and Pan Li. Inductive representation learning in temporal networks via causal anonymous walks. In International Conference on Learning Representations, 2021.

Qitian Wu, Wentao Zhao, Zenan Li, David P. Wipf, and Junchi Yan. Nodeformer: A scalable graph structure learning transformer for node classification. In Advances in Neural Information Processing Systems, volume 35, 2022.

Qitian Wu, Kai Yang, Hengrui Zhang, David Wipf, and Junchi Yan. Sgformer: Simplifying and scaling graph transformers with single-layer attention and approximation-free linear complexity. IEEE Transactions on Pattern Analysis and Machine Intelligence, pp. 1–16, 2026.

Zhenqin Wu, Bharath Ramsundar, Evan N Feinberg, Joseph Gomes, Caleb Geniesse, Aneesh S Pappu, Karl Leswing, and Vijay Pande. Moleculenet: a benchmark for molecular machine learning. Chemical science, 9(2):513–530, 2018.

Da Xu, Chuanwei Ruan, Evren Korpeoglu, Sushant Kumar, and Kannan Achan. Inductive representation learning on temporal graphs. In International Conference on Learning Representations, 2020.

Le Yu, Leilei Sun, Bowen Du, and Weifeng Lv. Towards better dynamic graph learning: New architecture and unified library. Advances in Neural Information Processing Systems, 36:67686– 67700, 2023.

Lunjun Zhang, Ge Yang, and Bradly C. Stadie. World model as a graph: Learning latent landmarks for planning. In Marina Meila and Tong Zhang (eds.), Proceedings of the 38th International Conference on Machine Learning, volume 139, pp. 12611–12620. PMLR, 2021.

Shaobin Zhuang, Zhipeng Huang, Ying Zhang, Fangyikang Wang, Canmiao Fu, Binxin Yang, Chong Sun, Chen Li, and Yali Wang. Video-gpt via next clip diffusion. In International Conference on Learning Representations, volume 2026, pp. 156161–156191, 2026.

Tao Zou, Yuhao Mao, Junchen Ye, and Bowen Du. Repeat-aware neighbor sampling for dynamic graph learning. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 4722–4733, 2024.

1 Introduction 1   
2 Problem Formulation 3   
3 WorldGraph 3   
3.1 State-aware Graph Transformer . 3   
3.1.1 Stage 1: Multi-granularity Graph Encoder 3   
3.1.2 Stage 2: History-aware State Encoder 4   
3.1.3 Theoretical Analysis of State-Aware Graph Transformer 5   
3.2 Transition-Aware Reinforcement Learning . 5   
3.2.1 Transition-Aware Dynamic Group Sampling 5   
3.2.2 Structure-aware Reward Design 6   
3.2.3 Theoretical Analysis of Transition-Aware RL Algorithm 6   
4 Experiments 7   
4.1 Experimental Setup 7   
4.2 Overall Performance (RQ1) 8   
4.3 Ablation Study (RQ2) 8   
4.4 Sensitivity Analysis (RQ3) 9   
4.5 Comparison on Existing GWM Problem Settings (RQ4) 9   
5 Conclusion 10   
A Notations 16   
B Additional Technical Details 17   
B.1 Detailed Design of State-aware Graph Transformer 17   
B.1.1 Details of Multi-granularity Graph Encoder 17   
B.1.2 Details of History-aware State Encoder 18   
B.2 Detailed Design of Transition-aware RL Algorithm 18   
B.2.1 Details of Transition-Aware Dynamic Group Sampling . 18   
B.2.2 Structure-aware Reward Design 18   
B.3 Details of Task-specific Structure-aware Reward Calculation 18   
B.4 Pseudo-code of Transition-aware RL 19   
C Proof of Theorems 21   
C.1 Proof of Theorem 1 21   
C.2 Proof of Theorem 2 21   
D Experimental Validation of Theoretical Results 22   
D.1 Validation of the Graph-Encoder Bound 22   
D.2 Validation of the RL Variance Bound . 22   
E Time and Space Complexity Analysis of WorldGraph. 23   
E.1 Multi-Granularity Graph Encoding. 23   
E.2 History-Aware State Encoding. 23   
E.3 Transition-Aware RL. 23   
F Additional Datasets and Baselines Information 24   
F.1 Statistics of Datasets 24   
F.2 Statistics of Baselines 24   
F.3 Why We Choose These Datasets and Baselines 25   
G Experimental details 27   
G.1 Details of WorldGraph Pretraining 27   
G.2 Details of Task Information 27   
G.3 Details of Plug-in Integration and Traditional Graph Prediction . 27   
G.4 Details of Hardware Information 27   
G.5 Details of Parameter setup. 28   
H More Experiments 29   
H.1 Efficiency Analysis of WorldGraph. 29   
H.2 Multi-step Prediction Analysis. 29   
H.3 Additional Sensitivity Analysis 29   
H.4 Image-to-Graph Visual Rollout. 31   
I More Related Works 32   
I.1 World Models. 32   
I.2 Temporal Graph Learning. 32   
J Reproducibility and Code Availability 32   
K Limitations and Broader Impacts 32   
K.1 Limitations 32   
K.2 Broader Impacts . 32

## A NOTATIONS

We provide the detailed notation in Table 4.

Table 4: Core notations used in WorldGraph.
<table><tr><td>Symbol</td><td>Description</td></tr><tr><td>Graph World</td><td></td></tr><tr><td> $t , \bar { \tau } , T$ </td><td>Current time, historical time, and sequence length.</td></tr><tr><td> $g _ { t } = ( V _ { t } , E _ { t } , X _ { t } ) \in \mathcal { G }$ </td><td>Observed graph at time t, its node, edge, and property sets, and the graph</td></tr><tr><td> $\Delta g _ { t }$ </td><td>space. Graph change associated with the transition from  $g _ { t } \ \mathrm { t o } \ g _ { t + 1 } .$ </td></tr><tr><td> $\mathbf { s } _ { t } \in \mathcal { S } , \ f _ { s }$ </td><td>Latent world state, world-state space, and state model.</td></tr><tr><td> $f _ { \Delta }$ </td><td>Transition model that predicts  $\hat { \Delta } g _ { t }$  from the current graph, the preceding graph change, and  $\mathbf { s } _ { t } .$ </td></tr><tr><td> $\mathbf { z } _ { v , t } , \ \mathbf { z } _ { t } , \ \mathbf { s } _ { v , t }$ </td><td>Structure-aware node embedding, graph embedding, and node-level latent state at time t.</td></tr></table>

<table><tr><td colspan="2">State-Aware Graph Transformer</td></tr><tr><td> $L _ { \mathrm { m a x } } , \ \ell$ </td><td>Maximum hop number and hop index.</td></tr><tr><td> $\mathbf { x } _ { v } ^ { ( \ell ) } , \mathbf { x } _ { v } ^ { \mathrm { p a t h } _ { i } }$ </td><td>Hop-level and i-th random-walk-path embeddings of node v.</td></tr><tr><td> $M , { \mathrm { ~ p a t h } } _ { i }$ </td><td>Number of sampled random-walk paths and the i-th path.</td></tr><tr><td> $\mathbf { x } _ { \Delta g _ { \tau } } , \mathbf { \bar { \Psi } } \mathbf { \tilde { y p e } } ( \Delta g _ { \tau } )$ </td><td>Transition representation and learnable transition-type embedding for  $\Delta g _ { \tau } .$ </td></tr><tr><td> $\mathbf { m } _ { v , \tau } , \mathrm { D i s t } ( t , \tau ) , t _ { \mathrm { m a x } }$   $\mathbf { W } _ { x } , \mathbf { W } _ { z } , \mathbf { W } _ { a }$ </td><td>Historical memory of node  $v ,$  relative temporal-distance embedding, and maximum historical time span.</td></tr><tr><td></td><td>Learnable projections of node history, graph history, and transition-type information in the node memory.</td></tr><tr><td> $\mathbf { q } _ { v , t } , \ \beta _ { v , t , \tau } , \ \tilde { \beta } _ { v , t , \tau } , \ d$ </td><td>History-attention query, attention score, normalized attention weight, and projected embedding dimension.</td></tr><tr><td> $\mathbf { p } _ { t - \tau } , \Pi ( \Delta g _ { \tau } , \Delta g _ { t - 1 } )$ </td><td>Relative-time attention bias and similarity between a historical transition and the latest observed transition.</td></tr><tr><td> $\mathbf { W } _ { Q } , \mathbf { W } _ { K } , \mathbf { W } _ { V }$   $\mathbf { h } _ { v , t } , \ \hat { \mathbf { h } } _ { v , t }$ </td><td>Learnable query, key, and value projections for history attention.</td></tr><tr><td></td><td>Single-head history embedding and the final multi-head history embedding of node v.</td></tr><tr><td> $g ^ { ( \mathrm { S G } ) } , ~ g _ { i } ^ { ( \mathrm { O U R S } ) }$ </td><td>An SGFormer subgraph and a WorldGraph sampled subgraph.</td></tr><tr><td> $g _ { i } ^ { ( \mathrm { I D E A L } ) } , \zeta$ </td><td>An ideal full-coverage subgraph and the ideal coverage count.</td></tr><tr><td> $\mathbf { s } _ { v , t } ^ { \mathrm { ( S G ) } } , \ \mathbf { s } _ { v , t } ^ { \mathrm { ( O U R S ) } }$ </td><td></td></tr><tr><td></td><td>Final latent states induced by SGFormer and WorldGraph.</td></tr><tr><td> $\mathbf { s } _ { v , t } ^ { ( \mathrm { I D E A L } ) }$ </td><td>Final latent state induced by the ideal full-coverage encoder.</td></tr><tr><td> $\Delta _ { \mathrm { S G } } , \ \Delta _ { \mathrm { O U R S } }$ </td><td>Deviations of SGFormer and WorldGraph from the ideal latent state</td></tr></table>

<table><tr><td colspan="2">Transition-Aware RL Algorithm</td></tr><tr><td> $\mathcal { T } _ { t } , ~ J _ { i } , ~ | J _ { i } |$ </td><td>Set of transition-conditioned groups at time  $t ,$  its i-th group, and the group size.</td></tr><tr><td> $\widehat { p } ( J _ { i } ) , \ \omega ( J _ { i } ) , \ \gamma$ </td><td>Historical transition frequency, rarity weight, and rarity-weight exponent  $J _ { i } .$ </td></tr><tr><td> $\mathrm { P r o b } ( J _ { i } ) , \boldsymbol { B } , \boldsymbol { B } _ { t } ( J _ { i } )$ </td><td>of group Selection probability, total rollout budget, and budget allocated to group</td></tr><tr><td> $\rho ^ { \mathrm { G R P O } } , \rho ^ { \mathrm { O U R S } } , \rho _ { i } , C _ { a }$ </td><td> $J _ { i } .$  Group sizes under standard GRPO and Transition-aware RL, a generic</td></tr><tr><td> $\mathrm { E V R } ( \rho _ { i } ) , M _ { \mathrm { E V R } } , \delta$ </td><td>group size, and the base group-size constant. Effective variance ratio, its residual term, and reward-distribution kurtosis.</td></tr><tr><td> $\mathcal { T } _ { \mathbf { s } } ( v ) , \deg ( v ) , \theta$ </td><td>Degree-based structural importance, node degree, and degree-scaling ex-</td></tr><tr><td> $f _ { < t } ( v ) , \ T _ { \mathbf { v } } ( v ) , \ \eta$ </td><td>ponent. Historical change frequency, variability importance, and variability-</td></tr><tr><td> $\mathcal { T } ( v ) , V _ { t }$ </td><td>scaling exponent. Normalized importance of node v and the current node set over which it is</td></tr><tr><td> $\Delta \hat { Q } , \Delta Q$ </td><td>normalized. Predicted and ground-truth change sets.</td></tr><tr><td> $\mathcal { P } , \mathcal { R } , \dot { R } _ { \mathrm { s t r u c t u r e } }$ </td><td>Importance-weighted precision, recall, and their harmonic-mean structure</td></tr><tr><td> $\lambda , R _ { \mathrm { p r o p e r t y } } , R$ </td><td>reward. Structural/property reward balancing coefficient, property-change reward, and unified task reward.</td></tr></table>

## B ADDITIONAL TECHNICAL DETAILS

## B.1 DETAILED DESIGN OF STATE-AWARE GRAPH TRANSFORMER

## B.1.1 DETAILS OF MULTI-GRANULARITY GRAPH ENCODER

Details of Hop-Level Message Passing. Given the maximum hop number $L _ { \mathrm { m a x } }$ , for a given graph $g _ { t }$ , we perform message passing for $\ell = \{ 1 , 2 , \cdots , L _ { \mathrm { m a x } } \}$ over the neighbors of each node to obtain the initial node embeddings. Let $\mathcal { N } ( v )$ denote the neighbor set of node v. We compute the node embedding of v:

$$
\mathbf { x } _ { v } ^ { ( \ell ) } = \mathrm { R e L U } \left( \mathbf { W } _ { \mathrm { s e l f } } ^ { ( \ell ) } \mathbf { x } _ { v } ^ { ( \ell - 1 ) } + \frac { \sum _ { u \in \mathcal { N } ( v ) } w _ { u v } \mathbf { W } _ { \mathrm { n e i g h b o r } } ^ { ( \ell ) } \mathbf { x } _ { u } ^ { ( \ell - 1 ) } } { \operatorname* { m a x } \left( 1 , \sum _ { u \in \mathcal { N } ( v ) } w _ { u v } \right) } \right)\tag{5}
$$

, where $w _ { u v }$ is set to the currently observed edge weight for a weighted graph, and to 1 for an unweighted graph, in which case the neighborhood term reduces to mean aggregation. $\mathbf { W } _ { \mathrm { s e l f } } ^ { ( \ell ) }$ and $\mathbf { W } _ { \mathrm { n e i g h b o r } } ^ { ( \ell ) }$ denote the learnable weight matrices at layer ℓ. Thus, we obtain node embeddings computed through multi-layer message passing based on neighbors at different hop distances and, when available, their observed relation strengths.

Details of Random-Walk Path Sampling. We use projected scaled dot-product attention to aggregate the node embeddings along path . Specifically, for each node $v _ { j } \in \mathsf { p a t h } _ { i }$ , we compute the attention score with respect to the starting node $v _ { 0 }$ as

$$
\mathrm { S c o r e } _ { i , j } = \frac { \left( { \bf W } _ { \mathrm { p a t h } } { \bf x } _ { v _ { j } } ^ { ( L _ { \mathrm { m a x } } ) } \right) ^ { \top } \left( { \bf W } _ { \mathrm { p a t h } } { \bf x } _ { v _ { 0 } } ^ { ( L _ { \mathrm { m a x } } ) } \right) } { \sqrt { d } } ,\tag{6}
$$

where $\mathbf { W } _ { \mathrm { p a t h } }$ is a learnable projection for path attention and d is the dimension of the projected node embedding. We normalize the scores over the nodes on the same path:

$$
\alpha _ { i , j } = \frac { \exp ( \mathrm { S c o r e } _ { i , j } ) } { \sum _ { k = 0 } ^ { T _ { i } } \exp ( \mathrm { S c o r e } _ { i , k } ) } .\tag{7}
$$

The embedding corresponding to the i-th path is then obtained by weighted aggregation:

$$
\mathbf { x } _ { v } ^ { \mathrm { p a t h } _ { i } } = \sum _ { j = 0 } ^ { T _ { i } } \alpha _ { i , j } \mathbf { x } _ { v _ { j } } ^ { ( L _ { \operatorname* { m a x } } ) } .\tag{8}
$$

Thus, we derive $\mathbf { x } _ { v } ^ { \mathrm { { p a t h } } _ { i } }$ for different sampled paths path , capturing multi-step structural information around node v from different random-walk trajectories.

After obtaining the hop-level embeddings $\{ \mathbf { x } _ { v } ^ { ( \ell ) } \} _ { \ell = 1 } ^ { L _ { \mathrm { m a x } } }$ and the path-level embeddings $\{ \mathbf { x } _ { v } ^ { \mathrm { p a t h } _ { i } } \} _ { i = 1 } ^ { M }$ we fuse all structural representations of node v via an attention-based aggregation to obtain a structure-aware embedding ${ \bf z } _ { v , t }$

Details of Multi-Granularity Attention Fusion. Then, we collect all candidate representations into a unified set $\mathcal { S } _ { v } = \left\{ \mathbf { x } _ { v } ^ { ( 1 ) } , \ldots , \mathbf { x } _ { v } ^ { ( L _ { \operatorname* { m a x } } ) } , \mathbf { x } _ { v } ^ { \mathrm { { p a t h } _ { 1 } } } , \ldots , \mathbf { x } _ { v } ^ { \mathrm { { p a t h } _ { M } } } \right\}$ . Let $\{ \mathbf { u } _ { v , k } \} _ { k = 1 } ^ { L _ { \operatorname* { m a x } } + M }$ denote the candidates in $\boldsymbol { S _ { v } }$ , where $\mathbf { u } _ { v , k } = \mathbf { x } _ { v } ^ { ( k ) }$ for $1 \leq k \leq L _ { \operatorname* { m a x } }$ and $\mathbf { u } _ { v , k } = \mathbf { x } _ { v } ^ { \mathrm { p a t h } _ { k - L _ { \mathrm { m a x } } } }$ for $L _ { \mathrm { m a x } } < k \leq$ $L _ { \mathrm { m a x } } + M$ . We compute an unnormalized attention score between each candidate ${ \bf u } _ { v , k }$ and the lasthop embedding of node v as $\begin{array} { r } { \alpha _ { v , k } = \frac { ( \mathbf { W } _ { \mathrm { f u s i o n } } \mathbf { u } _ { v , k } ) ^ { \top } \left( \mathbf { W } _ { \mathrm { f u s i o n } } \mathbf { x } _ { v } ^ { ( L _ { \operatorname* { m a x } } ) } \right) } { \sqrt { \mathcal { A } } } } \end{array}$ , where $\mathbf { W _ { \mathrm { f u s i o n } } }$ is a learnable √d   
projection for fusing the hop- and path-level representations. We then normalize the scores over all candidates via a softmax: $\begin{array} { r } { \tilde { \alpha } _ { v , k } = \frac { \exp \left( \alpha _ { v , k } \right) } { \sum _ { k ^ { \prime } = 1 } ^ { L _ { \operatorname* { m a x } } + M } \exp \left( \alpha _ { v , k ^ { \prime } } \right) } } \end{array}$ . The final structure-aware embedding is retained as the node-level world state:

$$
\mathbf { z } _ { v , t } = \sum _ { k = 1 } ^ { L _ { \operatorname* { m a x } } + M } \tilde { \alpha } _ { v , k } \mathbf { u } _ { v , k } .\tag{9}
$$

## B.1.2 DETAILS OF HISTORY-AWARE STATE ENCODER

Details of Query Embedding. For each node v, we construct a query embedding from its latest node-level world state, the global graph summary, the latest transition, and its previous latent state: $\mathbf { q } _ { v , t } = [ \mathbf { z } _ { v , t } ; \mathbf { z } _ { t } ; \mathbf { x } _ { \Delta g _ { t - 1 } } ; \mathbf { s } _ { v , t - 1 } ]$ , and aggregate its historical memory $\{ \mathbf { m } _ { v , \tau } \}$ with multi-head attention.

Details of Multi-Head Attention The same operation is performed in parallel by all attention heads. For one attention head, the history embedding is obtained by weighted aggregation of the value projections:

$$
\mathbf { h } _ { v , t } = \sum _ { \tau } \widetilde { \beta } _ { v , t , \tau } \left( \mathbf { m } _ { v , \tau } \mathbf { W } _ { V } \right) .\tag{10}
$$

The outputs from all heads are concatenated and mixed by an MLP to obtain the final history embedding:

$$
\widetilde { \mathbf { h } } _ { v , t } = \mathrm { M L P } \left( \big [ \mathbf { h } _ { v , t } ^ { ( 1 ) } ; \mathbf { h } _ { v , t } ^ { ( 2 ) } ; \ldots ; \mathbf { h } _ { v , t } ^ { ( H ) } \big ] \right) .\tag{11}
$$

## B.2 DETAILED DESIGN OF TRANSITION-AWARE RL ALGORITHM

## B.2.1 DETAILS OF TRANSITION-AWARE DYNAMIC GROUP SAMPLING

Details of Rarity-based Rollout Budget Allocation Finally, to make earlier rare transitions receive a larger effective group size, we allocate the total rollout budget B across groups proportionally. We reserve the same minimum number of rollouts<sup>9</sup> for every group and use largest-remainder rounding for the remaining budget, such that $\begin{array} { r } { \sum _ { j \in \mathcal { I } _ { t } } B _ { t } ( j ) = B _ { \mathrm { \ell } } } \end{array}$ , where $B _ { t } ( j )$ is the allocated number of rollouts for transition group j at time step t. Thus, groups corresponding to transitions that are rarer in the prefix $\{ \Delta g _ { \tau } \} _ { \tau < t }$ are assigned more rollout samples at time $t ,$ increasing the relative training signal for these frequency-imbalanced modification patterns.

## B.2.2 STRUCTURE-AWARE REWARD DESIGN

Details of Node/Edge/Graph Importance Calculation. The three graph-world tasks produce outputs at different granularities. We therefore retain the same structural importance $\mathcal { T } ( v )$ , while defining a verifiable reward consistent with the output and evaluation metric of each task. For a changed edge $\boldsymbol { e } = \left( u , v \right)$ , its importance is determined by its two nodes: $\begin{array} { r } { \mathcal { I } ( e ) = \frac { \mathcal { I } ( u ) + \mathcal { I } ( v ) } { 2 } } \end{array}$ . For a changed local graph centered at node v, its importance is $\mathcal { T } ( v )$

Although the three graph-world tasks operate at different granularities, their rewards share the same principle: a prediction should identify the correct structural changes and estimate the corresponding representation changes. We therefore use the unified reward

$$
R = \lambda R _ { \mathrm { s t r u c t u r e } } + ( 1 - \lambda ) R _ { \mathrm { p r o p e r t y } } ,\tag{12}
$$

where $R _ { \mathrm { p r o p e r t y } }$ is calculated based on the ground-truth updated property and the predicted property.   
λ balances the two terms.

## B.3 DETAILS OF TASK-SPECIFIC STRUCTURE-AWARE REWARD CALCULATION

Node-level tasks. Let $V _ { t + 1 } ^ { \mathrm { a d d } } , V _ { t + 1 } ^ { \mathrm { d e l e t e } }$ , and $\Delta V _ { t + 1 } ^ { \mathrm { p r o p e r t y } }$ denote the ground-truth sets of added, deleted, and property-changed nodes, respectively; their hatted versions denote the corresponding predictions. For a property-changed node $v , \Delta \mathbf { x } _ { v , t + 1 }$ and $\widehat { \Delta \mathbf { x } } _ { v , t + 1 }$ denote its ground-truth and predicted property changes. The importance-weighted property reward is

$$
R _ { \mathrm { p r o p e r t y } } ^ { \mathrm { n o d e } } = \exp \left( - \frac { \sum _ { v \in \Delta V _ { t + 1 } ^ { \mathrm { p r o p e r t y } } } \mathbb { Z } ( v ) \ \mathrm { M A E } \left( \widehat { \Delta \mathbf { x } } _ { v , t + 1 } , \Delta \mathbf { x } _ { v , t + 1 } \right) } { \sum _ { v \in \Delta V _ { t + 1 } ^ { \mathrm { p r o p e r t y } } } \mathbb { Z } ( v ) } \right) .\tag{13}
$$

Here, $R _ { \mathrm { s t r u c t u r e } } ^ { \mathrm { ( n o d e ) } }$ averages the set rewards for node addition, node deletion, and property-change localization. Setting $\lambda _ { \mathrm { n o d e } } = 3 / 4$ in the unified reward gives equal weight to the four constituent terms:

$$
\begin{array} { r l } & { R _ { \mathrm { n o d e } } = \cfrac { 1 } { 4 } [ R _ { \mathrm { s t r u c t u r e } } ( \widehat { V } _ { t + 1 } ^ { \mathrm { a d d } } , V _ { t + 1 } ^ { \mathrm { a d d } } ) + R _ { \mathrm { s t r u c t u r e } } ( \widehat { V } _ { t + 1 } ^ { \mathrm { d e l e t e } } , V _ { t + 1 } ^ { \mathrm { d e l e t e } } )  } \\ & { \qquad +  R _ { \mathrm { s t r u c t u r e } } ( \widehat { \Delta V } _ { t + 1 } ^ { \mathrm { p r o p e r t y } } , \Delta V _ { t + 1 } ^ { \mathrm { p r o p e r t y } } ) + R _ { \mathrm { p r o p e r t y } } ^ { \mathrm { n o d e } } ] . } \end{array}\tag{14}
$$

Edge-level tasks. Let $E _ { t + 1 } ^ { \mathrm { a d d } } , E _ { t + 1 } ^ { \mathrm { d e l e t e } }$ , and $\Delta E _ { t + 1 } ^ { \mathrm { p r o p e r t y } }$ denote the ground-truth sets of added, deleted, and property-changed edges, respectively; their hatted versions denote the corresponding predictions. For a property-changed edge $e , \Delta \mathbf { x } _ { e , t + 1 }$ and $\widehat { \Delta \mathbf { x } } _ { e , t + 1 }$ denote its ground-truth and predicted property changes. The importance-weighted property reward is

$$
R _ { \mathrm { p r o p e r t y } } ^ { \mathrm { e d g e } } = \mathrm { e x p } \left( - \frac { \sum _ { e \in \Delta E _ { t + 1 } ^ { \mathrm { p r o p e r t y } } } \mathbb { Z } ( e ) \mathrm { ~ M A E } \left( \widehat { \Delta \mathbf { x } } _ { e , t + 1 } , \Delta \mathbf { x } _ { e , t + 1 } \right) } { \sum _ { e \in \Delta E _ { t + 1 } ^ { \mathrm { p r o p e r t y } } } \mathbb { Z } ( e ) } \right) .\tag{15}
$$

Here, $R _ { \mathrm { s t r u c t u r e } } ^ { \mathrm { ( e d g e ) } }$ averages the set rewards for edge addition, edge deletion, and property-change localization. With $\lambda _ { \mathrm { e d g e } } = 3 / 4$ , the edge-level reward is

$$
\begin{array} { r } { R _ { \mathrm { e d g e } } = \frac { 1 } { 4 } \big [ R _ { \mathrm { s t r u c t u r e } } \Big ( \widehat { E } _ { t + 1 } ^ { \mathrm { a d d } } , E _ { t + 1 } ^ { \mathrm { a d d } } \Big ) + R _ { \mathrm { s t r u c t u r e } } \Big ( \widehat { E } _ { t + 1 } ^ { \mathrm { d e l e t e } } , E _ { t + 1 } ^ { \mathrm { d e l e t e } } \Big ) } \\ { + R _ { \mathrm { s t r u c t u r e } } \Big ( \widehat { \Delta E } _ { t + 1 } ^ { \mathrm { p r o p e r t y } } , \Delta E _ { t + 1 } ^ { \mathrm { p r o p e r t y } } \Big ) + R _ { \mathrm { p r o p e r t y } } ^ { \mathrm { e d g e } } \big ] . } \end{array}\tag{16}
$$

Graph-level tasks. Let $\Delta V _ { t + 1 } ^ { \mathrm { g r a p h } }$ and $\widehat { \Delta V } _ { t + 1 } ^ { \mathrm { g r a p h } }$ denote the ground-truth and predicted sets of nodes whose local graphs change. The local-graph representation $\phi _ { v , t }$ contains node count, edge count, density, number of connected components, clustering coefficient, and triangle count. We use $\Delta \phi _ { v , t + 1 } = \phi _ { v , t + 1 } - \phi _ { v , t }$ and $\widehat { \Delta \phi } _ { v , t + 1 } = \widehat { \phi } _ { v , t + 1 } - \phi _ { v , t }$ for the ground-truth and predicted representation changes. When evaluating a sampled transition, the property-change magnitude is computed over its selected local-graph centres that also exhibit a ground-truth structural change. Their property reward is

$$
R _ { \mathrm { p r o p e r t y } } ^ { \mathrm { g r a p h } } = \exp \left( - \frac { \sum _ { v \in \Delta V _ { t + 1 } ^ { \mathrm { g r a p h } } } \mathcal { T } ( v ) \mathrm { M A E } \left( \widehat { \Delta \phi } _ { v , t + 1 } , \Delta \phi _ { v , t + 1 } \right) } { \sum _ { v \in \Delta V _ { t + 1 } ^ { \mathrm { g r a p h } } } \mathcal { T } ( v ) } \right) .\tag{17}
$$

With $\lambda _ { \mathrm { g r a p h } } \quad = \quad 1 / 2$ , the graph-level reward equally combines change localization and representation-change magnitude:

$$
R _ { \mathrm { g r a p h } } = \frac { 1 } { 2 } \big [ R _ { \mathrm { s t r u c t u r e } } \Big ( \widehat { \Delta V } _ { t + 1 } ^ { \mathrm { g r a p h } } , \Delta V _ { t + 1 } ^ { \mathrm { g r a p h } } \Big ) + R _ { \mathrm { p r o p e r t y } } ^ { \mathrm { g r a p h } } \big ] .\tag{18}
$$

In practice, we add a small number $( \mathrm { e . g . , 1 0 ^ { - 8 } } )$ when computing the reward to avoid the numerical instability caused by a zero denominator.

## B.4 PSEUDO-CODE OF TRANSITION-AWARE RL

Algorithm 1 summarizes the training and inference procedure of the proposed transition-aware reinforcement learning algorithm. At each training step, the input is the current graph $g _ { t }$ , its observed historical transitions and latent states, and the next graph $g _ { t + 1 }$ used only as supervision. The state model first encodes $g _ { t }$ and its history into $\mathbf { s } _ { t }$ . From transitions observed before $t ,$ we then construct transition-conditioned groups $\mathcal { T } _ { t }$ , with one group corresponding to each transition type observed in the history. We estimate the historical frequency $\widehat { p } _ { t } ( J _ { i } )$ of each group, convert it to the rarity weight $\omega _ { t } ( J _ { i } ) = \stackrel { \cdot } { p _ { t } } ( J _ { i } ) ^ { - \gamma }$ , and normalize these weights to obtain the group allocation probabilities. Every group receives the same base number $C _ { a }$ of rollouts, while the remaining budget is allocated according to these probabilities, giving relatively more samples to transition types that are less frequently observed.

For each allocated rollout, the first operation is constrained to the corresponding group; for example, in a node-level task, one group may start with node addition, another with node deletion, and another with a node feature change. The Controller then samples subsequent valid operations and target nodes or edges until it stops. The resulting operation sequence is decoded as a predicted change set $\Delta \hat { Q }$ (together with the task-specific representation change). Comparing it with the ground-truth change set $\Delta Q$ and representation change from $g _ { t + 1 }$ gives the structure reward $R _ { \mathrm { s t r u c t u r e } }$ and property reward $R _ { \mathrm { p r o p e r t y } }$ , which are combined into the scalar return $R = \lambda R _ { \mathrm { s t r u c t u r e } } + ( 1 - \lambda ) R _ { \mathrm { l } }$ <sub>property</sub>. We use GRPO to compare these returns only among rollouts that began with the same edit type. Its core idea is to avoid fitting a separate value or critic network: rewards are normalized within each transition-conditioned group to form group-relative advantages. A rollout with an above-group average return receives a positive advantage, whereas a below-average rollout receives a negative one; the clipped policy-ratio objective then increases the probability of the former and decreases the probability of the latter. At inference time, the trained Controller supplies the transition conditioning and the task-specific prediction heads decode the final output from the predicted future latent state.

Algorithm 1 Transition-aware RL Training and Inference   
Require: Chronological graph sequence $\{ g _ { t } \} _ { t = 1 } ^ { T }$ , state model $f _ { s } ,$ , transition Controller $f _ { \Delta }$ , rollout   
budget $B ,$ base group size $C _ { a } ,$ exponents $\gamma , \theta , \eta ,$ and reward coefficient λ   
Ensure: Trained WorldGraph model   
1: Initialize the state model, transition Controller, and task-specific prediction heads   
2: for $t = 1 , 2 , \dots , T - 1$ do   
3: Encode the current graph and its history to obtain $\mathbf { s } _ { t } = f _ { s } ( g _ { t } , \Delta g _ { t - 1 } , \mathbf { s } _ { t - 1 } )$   
4: Compute the degree-based and historical-variability importance terms and obtain $\mathcal { T } ( v )$ using   
θ and η   
5: Construct the transition-conditioned groups $\mathcal { T } _ { t }$ from transitions observed before time t   
6: for each group $J _ { i } \in \mathcal { I } _ { t }$ do   
7: Estimate its historical frequency $\widehat { p } ( J _ { i } )$ and compute $\omega ( J _ { i } ) = \widehat { p } ( J _ { i } ) ^ { - \gamma }$   
8: end for   
9: Compute Prob(J ) by normalizing the rarity weights over $\mathcal { T } _ { t }$   
10: Allocate $B _ { t } ( J _ { i } )$ rollouts to each group, reserving $C _ { a }$ rollouts per group and distributing the   
remaining budget according to Prob(J )   
11: for each group $J _ { i } \in \mathcal { I } _ { t }$ do   
12: for $b = 1 , 2 , \ldots , B _ { t } ( J _ { i } )$ do   
13: Force the first operation to belong to $J _ { i }$   
14: Sample subsequent valid operations and target nodes or edges from $f _ { \Delta }$ until the   
stopping decision   
15: Decode the sampled operation sequence as a predicted change set $\Delta \hat { Q }$   
16: Compare $\Delta \hat { Q }$ with the ground-truth change set $\Delta Q$ and compute $R _ { \mathrm { s t r u c t u r e } }$   
17: Compute the task-specific property reward R<sub>property</sub>   
18: Compute $R = \lambda R _ { \mathrm { s t r u c t u r e } } ^ { \mathrm { ~ - ~ } } + ( 1 - \lambda ) R _ { \mathrm { p r o p e r t y } } ^ { \mathrm { ~ ~ } }$   
19: end for   
20: end for   
21: Normalize the rewards within each transition-conditioned group to obtain group-relative   
advantages   
22: Update $f _ { \Delta }$ with the clipped GRPO objective using all sampled rollouts and their group  
relative advantages   
23: end for   
24: At inference time, use the trained Controller to provide the transition conditioning, and use the   
task-specific prediction heads to decode the final task output from the predicted future latent   
state   
25: return the trained WorldGraph model

## C PROOF OF THEOREMS

## C.1 PROOF OF THEOREM 1

Proof. From Equation 3, the final latent state can be written as

$$
\mathbf { s } _ { v , t } ^ { ( \mathrm { m e t h o d } ) } = \mathcal { F } \Big ( \mathbf { z } _ { v , t } ^ { ( \mathrm { m e t h o d } ) } \Big ) .\tag{19}
$$

$\mathcal { F }$ is a deterministic function induced by the structure and history information with the same $\Delta g _ { t - 1 }$ ${ \bf s } _ { v , t - 1 } { } ^ { 1 0 }$ , and model parameters.

Following existing theoretical analysis of graph learning methods (Huang et al., 2024; Maskey et al., 2026; Jia et al., 2024), we consider that the message function $\mathcal F ( \cdot )$ is C-Lipschitz. Then the deviation between a method and the ideal model is bounded by the induced deviation of the aggregated inputs:

$$
\begin{array} { r } { \Delta _ { \mathrm { m e t h o d } } = \left\| \mathbf { s } _ { v , t } ^ { ( \mathrm { m e t h o d } ) } - \mathbf { s } _ { v , t } ^ { ( \mathrm { I D E A L } ) } \right\| _ { 2 } \leq C _ { L } \cdot \left\| \mathbf { z } _ { v , t } ^ { ( \mathrm { m e t h o d } ) } - \mathbf { z } _ { v , t } ^ { ( \mathrm { I D E A L } ) } \right\| _ { 2 } } \end{array}
$$

where $C _ { L }$ is a constant.

Based on Equation 9, we can mode $\begin{array} { r } { \mathbf { z } _ { v , t } ^ { ( \mathrm { m e t h o d } ) } = \sum _ { i } \psi ( g _ { i } ) } \end{array}$ . In our state-aware graph transformer $( \mathrm { S G T } ) , \psi ( g _ { i } ) = \tilde { \alpha } _ { i } { \mathbf { u } _ { v , k } } .$ as defined in Equation 9. To avoid representation drift of the same subgraph instance induced by different hyperparameters in attention designs (e.g., the number of attention heads), we use a consistent attention encoder across SGFormer, SGT (ours), and the ideal model. Consequently, for any training subgraph $g _ { i } , \psi _ { \mathrm { S G F o r m e r } } ( g _ { i } ) = \psi _ { \mathrm { O U R S } } ( g _ { i } ) = \psi _ { \mathrm { I D E A L } } ( g _ { i } ) = \psi ( g _ { i } )$ Following (Wan et $\mathrm { a l . , } 2 0 2 2 )$ , we can bound $\psi ( g _ { i } )$ by a constant $C _ { p } . \ \tilde { \alpha } _ { i }$ is bounded by 1 because it is exponentially normalized.

Therefore, $\begin{array} { r } { \big \| { \mathbf z } ^ { ( \mathrm { m e t h o d } ) } - { \mathbf z } ^ { ( \mathrm { I D E A L } ) } \big \| _ { 2 } = \| \sum _ { g _ { i } \notin g ^ { ( \mathrm { m e t h o d } ) } } \psi ( g _ { i } ) \| _ { 2 } \leq \| \sum _ { g _ { i } \notin g ^ { ( \mathrm { m e t h o d } ) } } C _ { p } \| _ { 2 } } \end{array}$ . Then, the upper bound of $\Delta _ { \mathrm { O U R S } } = \lVert \mathbf { z } ^ { ( \mathrm { O U R S } ) } - \bar { \mathbf { z } } ^ { ( \mathrm { i D E A L } ) } \rVert _ { 2 } \mathrm { i s } \ ( \zeta - ( L _ { \operatorname* { m a x } } + \bar { M } ) ) \lVert C _ { p } \rVert _ { 2 }$ . The upper bound of $\Delta _ { \mathrm { S G F o r m e r } } = \left\| \mathbf { z } ^ { \mathrm { ( S G F o r m e r ) } } - \mathbf { z } ^ { \mathrm { ( I D E A L ) } } \right\| _ { 2 } \mathrm { i s } \left( \zeta - 1 \right) \| C _ { p } \| _ { 2 } > ( \zeta - ( L _ { \operatorname* { m a x } } + M ) ) \| C _ { p } \| _ { 2 }$ Therefore, the upper bound of $\Delta _ { \mathrm { O U R S } }$ is smaller than that of $\Delta _ { \mathrm { S G } }$

## C.2 PROOF OF THEOREM 2

Proof. Because GRPO uses a fixed group size, the group size corresponding to each transition type is a constant $C _ { a }$ . Hence,

$$
\rho ^ { \mathrm { G R P O } } = C _ { a } .\tag{20}
$$

Our RL algorithm uses a dynamic group size based on the transition frequencies. As discussed in Footnote 6, we assign a minimum number of rollouts to different transition types. In this case, each transition type corresponds to the same group size as in GRPO. Then, based on the transition frequency, we allocate additional rollout budget to transitions with larger/minimum counts (i.e., larger group sizes) for different transition types, which leads to

$$
\rho ^ { \mathrm { O U R S } } = C _ { a } + \mathrm { P r o b } ( J _ { i } ) \cdot \mathcal { B } > \rho ^ { \mathrm { G R P O } } = C _ { a } .\tag{21}
$$

From (Kim & Lee, 2026), $\begin{array} { r } { \mathrm { E V R } ( \rho ) = 1 + \frac { \delta - 1 } { 2 \rho } + M _ { \mathrm { E V R } } } \end{array}$ . Following (Kim & Lee, 2026), we consider $\delta > 1$ . Since $\operatorname { E V R } ( \rho )$ is monotonically decreasing with respect to $\rho ,$

$$
\mathrm { E V R } ( \rho ^ { \mathrm { O U R S } } ) < \mathrm { E V R } ( \rho ^ { \mathrm { G R P O } } ) .\tag{22}
$$

This completes the proof.

Table 5: Experimental validation of Theorem 1 on Graph-level tasks/Flights. Values are mean ± standard deviation over five checkpoints; lower is better.
<table><tr><td>Encoder setting</td><td> $\| \mathbf { z } _ { v , t } - \mathbf { z } _ { v , t } ^ { ( \mathrm { I D E A L } ) } \| _ { 2 }$ </td><td> $\| \mathbf { s } _ { v , t } - \mathbf { s } _ { v , t } ^ { ( \mathrm { I D E A L } ) } \| _ { 2 }$ </td></tr><tr><td>SGFormer (Single-view)</td><td> $1 1 . 2 7 4 3 \pm 0 . 3 4 0 0$ </td><td> $6 . 2 4 6 0 \pm 0 . 9 2 5 5$ </td></tr><tr><td>WorldGraph (Multi-view SGT)</td><td> $\mathbf { 3 . 8 2 1 0 \pm 0 . 5 3 6 2 }$ </td><td> $\mathbf { 1 . 5 5 6 8 \pm 0 . 3 6 9 1 }$ </td></tr></table>

Table 6: Experimental validation of Theorem 2 on Node-level tasks/TGBN-Trade. Mean EVR is the arithmetic mean over the three transition groups; lower is better.
<table><tr><td>Method</td><td>Mean EVR</td></tr><tr><td>Standard GRPO</td><td>1.5550</td></tr><tr><td>Transition-aware RL</td><td>1.3526</td></tr></table>

## D EXPERIMENTAL VALIDATION OF THEORETICAL RESULTS

## D.1 VALIDATION OF THE GRAPH-ENCODER BOUND

To empirically verify the ordering predicted by Theorem 1, we use five frozen WorldGraph checkpoints on Graph-level tasks/Flights and evaluate the same test transitions without further training. We construct a full-view reference representation using an expanded deterministic structural-view set, and compare it with (i) SGFormer, which retains only one structural view, and (ii) the default multi-view SGT encoder. We report the mean node-wise Euclidean distances to the reference representation and the corresponding hidden-state distances. As shown in Table 5, the default multi-view SGT has substantially smaller deviations than SGFormer for both ${ \bf z } _ { v , t }$ and ${ \bf s } _ { v , t }$ , consistent with the smaller deviation bound established in Appendix C.1.

## D.2 VALIDATION OF THE RL VARIANCE BOUND

To empirically validate the ordering predicted by Theorem 2, we conduct a controlled experiment on Node-level tasks/TGBN-Trade. Standard GRPO uses a fixed base group size $C _ { a }$ for each transition type. Transition-aware RL retains the same base group size, then allocates the remaining rollout slots according to the estimated transition rarity until the configured total budget is exhausted. We report the first-order EVR approximation $1 + ( \dot { \delta } - 1 ) / ( 2 \rho )$ . As shown in Table 6, Transition-aware RL has a 13.02% lower mean EVR than standard GRPO, consistent with the ordering established in Appendix C.2.

## E TIME AND SPACE COMPLEXITY ANALYSIS OF WORLDGRAPH.

Let $N = | V _ { t } |$ | and $E = | E _ { t } |$ denote the numbers of nodes and edges in the current graph, respectively, and let d be the maximum hidden dimension. We further use $L _ { \mathrm { m a x } }$ for the number of messagepassing hops, M for the number of random walks per node, $T _ { \mathrm { r w } }$ for their maximum length, $L _ { \mathrm { h i s t } }$ for the history-window length, $C _ { t }$ for the number of legal transition candidates at time t, and B for the total rollout budget. The time complexity of WorldGraph consists of the following three parts.

## E.1 MULTI-GRANULARITY GRAPH ENCODING.

The $L _ { \mathrm { m a x } }$ hop-level message-passing layers require $O \big ( L _ { \operatorname* { m a x } } ( E d + N d ^ { 2 } ) \big )$ time. Sampling M walks of length at most $T _ { \mathrm { r w } }$ from every node costs $O ( \dot { N } M T _ { \mathrm { r w } } )$ , while path encoding and attention fusion require $O \big ( N ( M T _ { \mathrm { r w } } + L _ { \mathrm { m a x } } + M ) d ^ { 2 } \big )$ . Therefore, the total time complexity of this component is

$$
O \big ( L _ { \mathrm { m a x } } E d + N ( L _ { \mathrm { m a x } } + M T _ { \mathrm { r w } } + M ) d ^ { 2 } \big ) .\tag{23}
$$

## E.2 HISTORY-AWARE STATE ENCODING.

For each node, the state encoder uses one current query to attend to at most $L _ { \mathrm { h i s t } }$ historical memories. The query, key, value, and feed-forward projections require $O ( N L _ { \mathrm { h i s t } } d ^ { 2 } )$ time, while computing and aggregating the attention weights requires $O ( N L _ { \mathrm { h i s t } } d )$ time. Hence, this component has complexity $\breve { O ( N L _ { \mathrm { h i s t } } d ^ { 2 } ) }$ .

## E.3 TRANSITION-AWARE RL.

Encoding the node states and scoring the legal node or edge candidates costs $O \big ( ( N + C _ { t } ) d ^ { 2 } \big )$ Dynamic group statistics and rollout allocation depend only on the small set of transition groups and are negligible compared with candidate scoring. The encoded graph state and Controller logits are shared across the $\boldsymbol { B }$ rollouts; batched transition-conditioned state updates and structure-aware reward computation require at most $O \big ( B ( N d ^ { 2 } + C _ { t } + N + E ) \big )$ time. For node- and graph-level tasks, $C _ { t }$ is typically linear in $N .$ . For edge-level tasks, $C _ { t }$ may reach $O ( N ^ { 2 } )$ when all legal node pairs are considered, in which case candidate construction and scoring dominate the computation.

Combining the three components, the per-transition training complexity is

$$
\ O \big ( L _ { \mathrm { m a x } } E d + N ( L _ { \mathrm { m a x } } + M T _ { \mathrm { r w } } + M + L _ { \mathrm { h i s t } } ) d ^ { 2 } + C _ { t } d ^ { 2 } + \mathcal { B } ( N d ^ { 2 } + C _ { t } + N + E ) \big ) .\tag{24}
$$

Since $L _ { \mathrm { m a x } } , M , T _ { \mathrm { r w } } , L _ { \mathrm { h i s t } }$ , and $\boldsymbol { B }$ are bounded hyperparameters, WorldGraph scales linearly with the observed graph size apart from the task-dependent transition-candidate set. Its space complexity is

$$
O ( E + N ( L _ { \mathrm { m a x } } + M T _ { \mathrm { r w } } + L _ { \mathrm { h i s t } } ) d + C _ { t } d + { \mathcal { B } } ( N d + C _ { t } ) ) ,\tag{25}
$$

accounting for graph storage, hop/path representations, historical memories, candidate states, and rollout tensors.

## F ADDITIONAL DATASETS AND BASELINES INFORMATION

## F.1 STATISTICS OF DATASETS

We evaluate GWM-Zero on eight real-world dynamic graph datasets. Their statistics are summarized in Table 7. Contact interactions are aggregated into causal six-hour snapshots, SocialEvo interactions into daily snapshots, and Enron interactions into weekly snapshots; the remaining datasets retain their released temporal granularity. We adopt the official TGB splits and chronological 70%/15%/15% splits otherwise. All preprocessing statistics and thresholds are fitted using the training split only, and predictions use only the observed graph history.

Table 7: Statistics of the eight real-world datasets used in GWM-Zero. The numbers of temporal events and snapshots are computed from the processed artifacts used by all methods.
<table><tr><td>Dataset</td><td>Domain</td><td>Nodes</td><td>Temporal Events</td><td>Snapshots</td><td>Edge Input</td><td>Snapshot Interval</td><td>Task(s)</td></tr><tr><td>TGBN-Trade</td><td>Economic interaction</td><td>255</td><td>468,245</td><td>31</td><td>Weighted</td><td>Annual</td><td>Node-Level Tasks, Edge-Level Tasks</td></tr><tr><td>TGBN-Genre</td><td>User preference</td><td>1,505</td><td>17,858,395</td><td>1,580</td><td>Weighted</td><td>Daily</td><td>Node-Level Tasks</td></tr><tr><td>TGBN-Reddit</td><td>Social interaction</td><td>11,766</td><td>27,174,118</td><td>1,090</td><td>Binary</td><td>Daily</td><td>Node-Level Tasks</td></tr><tr><td>UN Vote</td><td>Political interaction</td><td>201</td><td>1,035,742</td><td>72</td><td>Weighted</td><td>Annual</td><td>Edge-Level Tasks</td></tr><tr><td>Contact</td><td>Physical proximity</td><td>692</td><td>2,426,279</td><td>112</td><td>Binary</td><td>Six hours</td><td>Edge-Level Tasks, Graph-Level Tasks</td></tr><tr><td>SocialEvo</td><td>Social proximity</td><td>74</td><td>2,098,119</td><td>243</td><td>Binary</td><td>Daily</td><td>Edge-Level Tasks</td></tr><tr><td>Flights</td><td>Transportation</td><td>13,169</td><td>1,811,118</td><td>41</td><td>Binary</td><td>Monthly</td><td>Graph-Level Tasks</td></tr><tr><td>Enron</td><td>Communication</td><td>184</td><td>108,825</td><td>183</td><td>Binary</td><td>Weekly</td><td>Graph-Level Tasks</td></tr></table>

• TGBN-Trade (Huang et al., 2023) is a directed international trade network. Nodes denote countries and weighted temporal edges record trade relations. Its official dynamic country-property vectors support the node-level task, while its annual topology is also used for edge-level prediction.

• TGBN-Genre (Huang et al., 2023) is a weighted bipartite interaction graph between users and music-genre coordinates. The official daily genre-preference vectors are retained without target remapping or normalization and are used for node-level evolution prediction.

• TGBN-Reddit (Huang et al., 2023) is a large bipartite interaction graph between entities and subreddit coordinates. We use binary graph connectivity and retain the official 698-dimensional dynamic property vectors for the node-level task.

• UN Vote (Poursafaei et al., 2022) is an annual country interaction network. A directed weighted edge represents the number of co-yes voting relations between a country pair within an annual interval. It is used for edge addition and removal prediction.

• Contact (Poursafaei et al., 2022) records physical proximity among students. We aggregate the released events into causal six-hour binary snapshots and use the resulting sequence for both edgelevel and graph-level evaluation.

• SocialEvo (Poursafaei et al., 2022) records temporal social-proximity interactions. After removing self-loops, we aggregate interactions into daily binary snapshots and use them for edge-level evaluation.

• Flights (Poursafaei et al., 2022) is a directed airport network in which an edge denotes an observed flight connection. For graph-level evaluation, we use the released native temporal snapshots and predict the next snapshot’s local structural changes.

• Enron (Poursafaei et al., 2022) is a directed employee email network. We remove self-loops and aggregate the interactions into weekly binary snapshots for graph-level structural-evolution prediction.

## F.2 STATISTICS OF BASELINES

We compare WorldGraph with eleven representative baselines covering graph encoders, graph Transformers, temporal graph models, pretrained graph models, and graph world models. Every method receives the same current graph, released history, data split, and task targets. For compatibility with the graph-world tasks, each baseline is equipped with supervised transition and countprediction heads, while its original representation backbone is preserved.

• MPNN.

– GCN (Kipf & Welling, 2017) performs normalized neighborhood aggregation through graph convolution.

– GAT (Velickovi ˇ c et al., 2018) assigns learned attention weights to neighboring nodes during ´ message passing.

– GraphSAGE (Hamilton et al., 2017) constructs node representations using sampledneighborhood aggregation.

## • Graph Transformers.

– SGFormer (Wu et al., 2026) combines simplified global self-attention with graph propagation for scalable graph representation learning.

– NodeFormer (Wu et al., 2022) approximates all-pair node attention with kernelized message passing.

– GraphGPS (Rampasek et al., 2022) combines a local message-passing module with a global ´ Transformer module.

## • Temporal graph models.

– TGN (Rossi et al., 2020) maintains node-wise memories that are updated by time-stamped interactions and uses them for temporal prediction.

– TIDFormer (Peng et al., 2025) models temporal intervals and evolving dependencies through interval-aware attention.

## • Pretrained graph models.

– Graph-JEPA (Skenderi et al., 2023) improves the structured representation of a graph world model through train-only self-supervised latent prediction on graph patches, followed by finetuning of its encoder and task adapter.

– MDGFM (Wang et al., 2025b) transfers a leave-one-domain-out pretrained encoder that aligns topology across source domains before downstream adaptation.

## • Graph world model<sup>11</sup>.

– GWM-E (Feng et al., 2025)<sup>12</sup> adopts the publicly released embedding-based GWM (GWM-E). It repeatedly propagates node properties over the graph to obtain multi-hop embeddings, maps the embedding at each hop into graph tokens through hop-specific projectors, and inserts these tokens into the task prompt so that a frozen LLaMA decoder can produce the final prediction.

– L<sup>3</sup>P (Zhang et al., 2021) identifies landmark states in the environment and constructs a reachability graph over them. It performs long-horizon planning by composing transitions between these landmarks, reducing the accumulation of errors from step-by-step prediction.

– C-SWM (Kipf et al., 2020) encodes visual observations into object-centric latent nodes, uses a graph neural network to model their interactions and transitions, and learns the latent dynamics with a contrastive objective.

## F.3 WHY WE CHOOSE THESE DATASETS AND BASELINES

The selected datasets cover economic, preference, social, political, proximity, transportation, and communication networks. They vary substantially in scale, temporal granularity, graph type, and edge semantics, ranging from 74 to 13,169 nodes and from approximately 10<sup>5</sup> to $\phantom { - } \overline { { 2 . 7 } } \times \overline { { 1 } } 0 ^ { 7 }$ temporal events. Moreover, the same collection supports complementary node-, edge-, and graph-level predictions, allowing us to evaluate whether a graph world model can consistently capture evolution at different structural granularities rather than overfitting to one task type.

The baselines provide complementary comparisons. GCN, GAT, and GraphSAGE test conventional local message passing; SGFormer, NodeFormer, and GraphGPS test global graph attention; TGN and TIDFormer test dedicated temporal modeling; Graph-JEPA and MDGFM test whether graph pretraining alone is sufficient; L<sup>3</sup>P and C-SWM test graph-structured planning and object-centric visual dynamics, respectively; and GWM-E tests graph-token conditioning of an LLM for traditional graph prediction tasks. This selection therefore separates the contributions of graph encoding, temporal state modeling, pretraining, graph-structured world modeling, graph-token task conditioning, and transition-aware reinforcement learning.

## G EXPERIMENTAL DETAILS

## G.1 DETAILS OF WORLDGRAPH PRETRAINING

WorldGraph-Pretrain adopts same-task leave-one-dataset-out pretraining: for each target dataset, we exclude it and use chronological transitions from the remaining datasets under the same node-, edge-, or graph-level task. Given a current graph $g _ { t } .$ , we construct two views through property masking and independent random-walk sampling. Representations of the same node v in the two views form a positive pair, while representations of different sampled nodes serve as negatives; we optimize these pairs with a symmetric InfoNCE loss. We minimize the cosine distance between the two views’ corresponding graph embeddings $\mathbf { z } _ { t }$ and between their corresponding node-level latent states ${ \bf s } _ { v , t } ,$ , and align the predicted next-step node representations with those encoded from $g _ { t + 1 }$ . The resulting multi-granularity graph encoder, history-aware state encoder, and latent predictor initialize the complete WorldGraph model for downstream training.

## G.2 DETAILS OF TASK INFORMATION

(1) Node Level Tasks. We evaluate node addition, node deletion, and node property change on TGBN-Trade, TGBN-Genre, and TGBN-Reddit. Each released snapshot defines one discrete time step. A node is naturally added when it has no incident edge at the current step but has one at the next, and naturally removed in the reverse case. Because natural changes are sparse, we supplement them with shared nodes whose historical interaction activity is rising (for addition) or declining (for deletion), masking the node, its incident edges, and its property at the corresponding time step. Properties are given by the official dynamic node-property labels. For the nodes observed at both adjacent steps, we compute their Jaccard distance between the Top-10 positive property dimensions. We then sort these distances in ascending order and use the value at the 60th percentile as the threshold. If a node’s Jaccard distance across adjacent steps exceeds this threshold, we mark it as indicating a property change. We report F1 score for addition, deletion, and property change, and NDCG@10 for future property prediction on changed nodes.

(2) Edge Level Tasks. We evaluate edge addition and deletion on TGBN-Trade, UN Vote, Contact, and SocialEvo. Given the current graph, the model predicts which disconnected node pairs will form edges and which existing edges will disappear, and then reconstructs the next graph. We report F1 score for edge addition and deletion and their macro-F1 score.

(3) Graph Level Tasks. We evaluate subgraph-level structural property changes on Flights, Contact, and Enron datasets. We use all nodes incident to at least one edge in the current graph as center nodes. For each selected node, we construct its local subgraph in both the current and next graphs, and predict the change in its six-dimensional structural property vector, consisting of node count, edge count, density, number of connected components, clustering coefficient, and triangle count. We report MAE and RMSE for these changes and macro-F1 for classifying them as decreasing, unchanged, or increasing.

## G.3 DETAILS OF PLUG-IN INTEGRATION AND TRADITIONAL GRAPH PREDICTION

As discussed in Section 4.5, we evaluate WorldGraph in the following two settings. For L<sup>3</sup>P and C-SWM, we preserve each model’s original backbone and task-specific decoder while adding the multi-granularity graph encoder, history-aware state encoder, Controller, transition-aware dynamic group sampling, and structure-aware reward. Separately, we compare standalone WorldGraph with GWM-E on the latter’s traditional graph prediction tasks, where GWM-E values are taken from the original reported results.

## G.4 DETAILS OF HARDWARE INFORMATION

The main experiments are conducted on a server equipped with an Intel Xeon E5-2698 v4 CPU at 2.20 GHz (40 logical CPU cores), 256 GB RAM, and four NVIDIA Tesla V100 GPUs with 16 GB memory each. Each training process uses one GPU.

Table 8: Principal hyperparameters used by WorldGraph.
<table><tr><td>Hyperparameter</td><td>Node-level tasks</td><td>Edge-level tasks</td><td>Graph-level tasks</td></tr><tr><td>Latent-state dimension</td><td>64</td><td>32</td><td>64</td></tr><tr><td>Hidden-state dimension</td><td>64</td><td>32</td><td>64</td></tr><tr><td>Transition-representation dimension</td><td>32</td><td>16</td><td>32</td></tr><tr><td>Maximum historical time span</td><td>8</td><td>8</td><td>8</td></tr><tr><td>Dropout rate</td><td>0.1</td><td>0.1</td><td>0.1</td></tr><tr><td>Learning rate</td><td>10-3</td><td>10-3</td><td>10-3</td></tr><tr><td>Total rollout budget</td><td>12</td><td>16</td><td>12</td></tr></table>

## G.5 DETAILS OF PARAMETER SETUP.

All methods use the same chronological splits, graph inputs, history, and task targets. WorldGraph is optimized with AdamW using an eight-state history window. Checkpoints are selected on the validation set, and results are averaged over five runs. The main settings are summarized in Table 8.

All graph-state transitions are processed chronologically without access to future information, and every method uses the same task construction and evaluation protocol.

## H MORE EXPERIMENTS

## H.1 EFFICIENCY ANALYSIS OF WORLDGRAPH.

We compare the mean epoch time, computed by averaging the first five training epochs, for World-Graph and the eight baseline models on TGBN-Genre. The last two rows in Table 9 report only the pure reinforcement-learning overhead for standard GRPO and Transition-aware RL, respectively. WorldGraph w/o RL is somewhat slower than most representation and temporal baselines, but it is not the slowest; given the substantially stronger predictive performance reported in Table 1, this moderate computational cost is acceptable. For the RL comparison, Transition-aware RL incurs only 1.99 s more per epoch than standard GRPO (27.54 s compared with 25.55 s), indicating that the two key components of our RL design—transition-aware dynamic group sampling and structure-aware reward—introduce little additional runtime. Overall, these results show that WorldGraph provides a reasonable overall efficiency while delivering stronger predictive performance.

Table 9: Mean training time per epoch (seconds) on TGBN-Genre, computed over the first five training epochs.
<table><tr><td>Model</td><td>Time (s/epoch)</td></tr><tr><td>GCN</td><td>47.26</td></tr><tr><td>GAT</td><td>51.47</td></tr><tr><td>GraphSAGE</td><td>43.49</td></tr><tr><td>SGFormer</td><td>53.45</td></tr><tr><td>NodeFormer</td><td>72.11</td></tr><tr><td>GraphGPS</td><td>64.04</td></tr><tr><td>TGN</td><td>44.50</td></tr><tr><td>TIDFormer</td><td>55.10</td></tr><tr><td>WorldGraph w/o RL</td><td>66.89</td></tr><tr><td>Standard GRPO</td><td>25.55</td></tr><tr><td>Transition-aware RL</td><td>27.54</td></tr></table>

## H.2 MULTI-STEP PREDICTION ANALYSIS.

We evaluate multi-step prediction on the Graph-level tasks, where the model forecasts changes in the structural property vector of local subgraphs. Each model is first trained with the standard onestep Graph-level protocol. At evaluation time, it observes the graph at an initial time, predicts the next graph-state change, and then recursively feeds its own predicted graph state into the same statetransition and prediction modules for steps 2, 3, and 5; ground-truth future graphs are used only to construct the evaluation targets. Following the Graph-level evaluation protocol, we report MAE and RMSE for these structural-property changes and Macro-F1 for classifying them as decreasing, unchanged, or increasing. Table 10 summarizes the results. WorldGraph-Pretrain is the best method on all nine Contact and Enron metrics, and on eight of the nine Flights metrics. Its Macro-F1 improvement over the strongest external baseline is 1.78%/3.90%/4.12% on Flights, 6.36%/6.19%/7.60% on Contact, and 1.10%/1.45%/1.28% on Enron for Steps 2/3/5, respectively. At Step 5, it also reduces MAE/RMSE by 0.4075/0.5832 on Flights, 0.3861/0.5506 on Contact, and 0.3789/0.4997 on Enron relative to the best external values. These results show that our models retain a clear ad vantage as the prediction step increases, demonstrating effective multi-step graph-state prediction.

## H.3 ADDITIONAL SENSITIVITY ANALYSIS

Figure 6 shows the sensitivity of WorldGraph to four additional hyperparameters on Graph-Level Tasks/Flights (Macro-F1): the rarity weight exponent $\gamma ,$ , the degree importance exponent θ, the volatility importance exponent $\eta ,$ and the historical time span $t _ { \mathrm { m a x } }$ used by the history-aware state encoder. Overall, the four curves exhibit an inverted U-shaped trend, showing that moving either below or above the default configuration can degrade performance. The default values $\gamma = 1 . 0 0$ $\theta = 1 . 0 0 , \eta = 1 . 0 0$ , and $t _ { \mathrm { m a x } } = 8$ reach the common peak of 58.40%; for example, the rarity exponent gives 57.99% at 0.50 and 58.07% at 1.25, while the historical span gives 58.13%, 58.35%, and 58.25% at $t _ { \mathrm { m a x } } = 6$ , 10, and 12, respectively. The intermediate settings of $\theta$ and $\eta ,$ together with the default history span, provide a balance between emphasizing important nodes, preserving structural coverage, and retaining useful temporal context.

Table 10: Multi-step prediction results on the Graph-level tasks. Models recursively use their predicted graph states for steps 2, 3, and 5. MAE and RMSE are lower-is-better, whereas Macro-F1 is higher-is-better. The best results are shown in bold and the second best results are underlined.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Model</td><td colspan="3">Step 2</td><td colspan="3"> $\mathrm { S t e p } 3$ </td><td colspan="3"> $\mathtt { S t e p 5 }$ </td></tr><tr><td>MAE</td><td>RMSE</td><td>Macro-F1</td><td>MAE</td><td>RMSE</td><td>Macro-F1</td><td>MAE</td><td>RMSE</td><td>Macro-F1</td></tr><tr><td rowspan="12">Flights</td><td>GCN</td><td>0.9208</td><td>1.5297</td><td>0.4486</td><td>1.3097</td><td>2.0548</td><td>0.3970</td><td>2.2527</td><td>3.2212</td><td>0.3584</td></tr><tr><td>GAT</td><td>0.9080</td><td>1.4992</td><td>0.4314</td><td>1.1381</td><td>1.7816</td><td>0.3950</td><td>1.7601</td><td>2.5047</td><td>0.3372</td></tr><tr><td>GraphSAGE</td><td>0.8764</td><td>1.4712</td><td>0.4667</td><td>1.1325</td><td>1.8040</td><td>0.4418</td><td>1.6862</td><td>2.4935</td><td>0.3982</td></tr><tr><td>SGFormer</td><td>0.8789</td><td>1.4686</td><td>0.4735</td><td>1.1885</td><td>1.8848</td><td>0.4207</td><td>1.8938</td><td>2.7597</td><td>0.3861</td></tr><tr><td>NodeFormer</td><td>0.9694</td><td>1.5545</td><td>0.4004</td><td>1.3759</td><td>1.9958</td><td>0.3755</td><td>2.2803</td><td>2.9962</td><td>0.3449</td></tr><tr><td>GraphGPS</td><td>0.8799</td><td>1.4658</td><td>0.4834</td><td>1.2033</td><td>1.8941</td><td>0.4433</td><td>1.9271</td><td>2.8334</td><td>0.4045</td></tr><tr><td>TGN</td><td>0.9161</td><td>1.4559</td><td>0.4467</td><td>1.4293</td><td>2.0275</td><td>0.3755</td><td>3.0440</td><td>3.8768</td><td>0.3411</td></tr><tr><td>TIDFormer</td><td>0.8942</td><td>1.4159</td><td>0.4682</td><td>1.2958</td><td>1.9056</td><td>0.4236</td><td>2.5503</td><td>3.5035</td><td>0.3537</td></tr><tr><td>Graph-JEPA</td><td>0.9502</td><td>1.5329</td><td>0.4564</td><td>1.6205</td><td>2.3081</td><td>0.4047</td><td>4.1542</td><td>5.1787</td><td>0.3511</td></tr><tr><td>MDGFM</td><td>0.9480</td><td>1.5371</td><td>0.3762</td><td>1.3147</td><td>1.9468</td><td>0.3512</td><td>2.1888</td><td>2.9197</td><td>0.3284</td></tr><tr><td>WorldGraph</td><td>0.7248</td><td>1.2904</td><td>0.4889</td><td>0.8976</td><td>1.4458</td><td>0.4694</td><td>1.3700</td><td>2.0019</td><td>0.4376</td></tr><tr><td>WorldGraph-Pretrain</td><td>0.7251</td><td>1.2879</td><td>0.5012</td><td>0.8660</td><td>1.4123</td><td>0.4823</td><td>1.2787</td><td>1.9103</td><td>0.4457</td></tr><tr><td rowspan="14">Contact</td><td>GCN</td><td>0.8891</td><td>1.1747</td><td>0.4889</td><td>1.3626</td><td>1.7485</td><td>0.4343</td><td>2.3580</td><td>2.8872</td><td>0.3735</td></tr><tr><td>GAT</td><td>0.8579</td><td>1.1269</td><td>0.4862</td><td>1.2642</td><td>1.6818</td><td>0.4285</td><td>1.9412</td><td>2.7057</td><td>0.4048</td></tr><tr><td>GraphSAGE</td><td>0.8653</td><td>1.1392</td><td>0.5057</td><td>1.2461</td><td>1.6131</td><td>0.4657</td><td>1.6471</td><td>2.1265</td><td>0.4286</td></tr><tr><td>SGFormer</td><td>0.8160</td><td>1.1041</td><td>0.4935</td><td>1.1923</td><td>1.5899</td><td>0.4422</td><td>1.6425</td><td>2.2004</td><td>0.4057</td></tr><tr><td>NodeFormer</td><td>1.0261</td><td>1.3081</td><td>0.4596</td><td>1.7076</td><td>2.1143</td><td>0.4068</td><td>2.8412</td><td>3.3949</td><td>0.3751</td></tr><tr><td>GraphGPS</td><td>0.8513</td><td>1.1307</td><td>0.4971</td><td>1.3215</td><td>1.6957</td><td>0.4505</td><td>2.0084</td><td>2.5280</td><td>0.4244</td></tr><tr><td>TGN</td><td>1.0661</td><td>1.3535</td><td>0.4750</td><td>1.8515</td><td>2.3242</td><td>0.4089</td><td>3.5733</td><td>4.3327</td><td>0.3650</td></tr><tr><td>TIDFormer</td><td>0.9457</td><td>1.2559</td><td>0.4828</td><td>1.6557</td><td>2.1509</td><td>0.4324</td><td>3.3123</td><td>4.2178</td><td>0.3936</td></tr><tr><td>Graph-JEPA</td><td>0.9747</td><td>1.2581</td><td>0.4884</td><td>1.6361</td><td>2.1350</td><td>0.4388</td><td>3.0525</td><td>4.0853</td><td>0.4112</td></tr><tr><td>MDGFM</td><td>0.9567</td><td>1.2312</td><td>0.4642</td><td>1.5444</td><td>1.9985</td><td>0.4085</td><td>2.8350</td><td>3.6336</td><td>0.3985</td></tr><tr><td>WorldGraph WorldGraph-Pretrain</td><td>0.6756</td><td>0.9754</td><td>0.5412</td><td>0.8290</td><td>1.1253</td><td>0.4984</td><td>1.3388</td><td>1.6578</td><td>0.4919</td></tr><tr><td>GCN</td><td>0.6540</td><td>0.9542</td><td>0.5693</td><td>0.7796</td><td>1.0800</td><td>0.5276</td><td>1.2564</td><td>1.5759</td><td>0.5046</td></tr><tr><td>GAT</td><td>1.3042 1.2605</td><td>1.9075</td><td>0.3605</td><td>1.5877</td><td>2.1316</td><td>0.3126</td><td>2.2529</td><td>2.8909</td><td>0.2909</td></tr><tr><td>GraphSAGE</td><td>1.2994</td><td>1.9077 1.9115</td><td>0.3580</td><td>1.3940 1.5150</td><td>1.9723 2.0678</td><td>0.3290</td><td>1.6845</td><td>2.2817</td><td>0.2990</td></tr><tr><td>SGFormer</td><td>1.2876</td><td></td><td>0.3718</td><td></td><td></td><td>0.3369</td><td>1.9448</td><td>2.5415</td><td>0.3059</td></tr><tr><td></td><td>1.2769</td><td>1.8868</td><td>0.3794</td><td>1.5019</td><td>2.0407</td><td>0.3455</td><td>1.8557</td><td>2.4450</td><td>0.3211</td></tr><tr><td>NodeFormer</td><td></td><td>1.9222</td><td>0.3948</td><td>1.4932</td><td>2.0900</td><td>0.3688</td><td>1.9648</td><td>2.6402</td><td>0.3424</td></tr><tr><td>GraphGPS</td><td>1.2499</td><td>1.8518</td><td>0.3937</td><td>1.3978</td><td>1.9342</td><td>0.3546</td><td>1.7089</td><td>2.2872</td><td>0.3152</td></tr><tr><td>TGN</td><td>1.2283</td><td>1.8633</td><td>0.4151</td><td>1.3613</td><td>1.9201</td><td>0.3968</td><td>1.6981</td><td>2.2845</td><td>0.3809</td></tr><tr><td>TIDFormer</td><td>1.2737</td><td>1.8897</td><td>0.3761</td><td>1.4000</td><td>1.9510</td><td>0.3558</td><td>1.7766</td><td>2.3845</td><td>0.3299</td></tr><tr><td>Graph-JEPA</td><td>1.3478</td><td>1.9117</td><td>0.3394</td><td>1.6037</td><td>2.1119</td><td>0.3145</td><td>2.2833</td><td>2.9209</td><td>0.2981</td></tr><tr><td>MDGFM</td><td>1.3762</td><td>2.0157</td><td>0.3419</td><td>1.6272</td><td>2.1908</td><td>0.3015</td><td>2.2030</td><td>2.8013</td><td>0.2610</td></tr><tr><td>WorldGraph</td><td>1.2231</td><td>1.7935</td><td>0.4172</td><td>1.2508</td><td>1.7171</td><td>0.4108</td><td>1.3441</td><td>1.7830</td><td>0.3901</td></tr><tr><td>WorldGraph-Pretrain</td><td>1.2071</td><td>1.7874</td><td>0.4261</td><td>1.2332</td><td>1.7136</td><td>0.4113</td><td>1.3192</td><td>1.7820</td><td>0.3937</td></tr></table>

![](images/321252bbaee67bfef9fab6113073512910587987d715ae8484fa77999ef97653.jpg)

![](images/9e8ce2e55af5e1242c7d475182c074f1a61a9ddb1cfe48c6718a6427e769b894.jpg)

![](images/b932da42e3004f2aa9903e543be3bb9d4ef847da33d32371ad71a544fdbb2086.jpg)

![](images/9e3839a9e4f51af145adf1a8d2f790380bf6500d55697c1c3dc0703f48f49cc2.jpg)  
Figure 6: Sensitivity analysis on Graph-Level Tasks/Flights (Macro-F1) across the rarity weight exponent $\gamma ,$ degree importance exponent $\theta ,$ volatility importance exponent $\eta ,$ and historical time span $t _ { \mathrm { m a x } } .$

## H.4 IMAGE-TO-GRAPH VISUAL ROLLOUT.

We further evaluate WorldGraph on a five-object Shapes transition-conditioned visual rollout task. Each input image is encoded by a CNN object extractor and an MLP into five object nodes forming a complete graph. Given the recorded environment transition, the multi-granularity graph encoder, history-aware state encoder, Controller, and transition module predict the next graph-state change; the Controller is trained with transition-aware dynamic group sampling and structure-aware rewards. A separately trained CNN decoder then maps the updated graph state back to an image. During multi-step prediction, only the initial image and the recorded transition sequence are provided, and the model recursively uses its predicted graph state rather than any ground-truth future image. As shown in Figure 7, the decoded WorldGraph rollout closely matches the ground-truth sequence, with the object positions and their temporal changes remaining well aligned across the prediction steps. This result demonstrates that WorldGraph can also support computer-vision tasks through graph-state prediction and reconstruction.

![](images/71f00926485cb7ff37b56929f44f670f91735c5f88446fba390c7815841d86d1.jpg)  
Figure 7: Transition-conditioned image-to-graph-to-image rollout. The first row shows the ground truth sequence, and the second row shows the images decoded from the graph states predicted by WorldGraph.

## I MORE RELATED WORKS

## I.1 WORLD MODELS.

World models learn compact internal representations of environments and model their temporal evolution to support prediction, planning, and decision-making. The seminal World Models framework (Ha & Schmidhuber, 2018) combines visual representation learning, recurrent memory, and a controller, while PlaNet (Hafner et al., 2019) learns stochastic latent dynamics for online planning and Dreamer (Hafner et al., 2020) learns long-horizon behaviors through latent imagination. Recent studies further improve the scalability and temporal abstraction of world models through decoder free latent dynamics (Hansen et al., 2024), Transformer-based contrastive prediction (Burchi & Timofte, 2025), and autoregressive next-clip diffusion for visual prediction (Zhuang et al., 2026). Object-centric world models further decompose observations into explicit entities and model their stochastic dynamics (Daniel et al., 2026). However, although these representations improve prediction or make individual entities explicit, they generally leave evolving relations and structural dependencies implicit. Consequently, they struggle to model entity, relation, and attribute changes at different granularities, highlighting graph-structured representations and relational inductive biases as important foundations for modeling complex and evolving worlds (Liu et al., 2026).

## I.2 TEMPORAL GRAPH LEARNING.

Temporal graph learning models (Kumar et al., 2019; Rossi et al., 2020; Xu et al., 2020; Wang et al., 2021; Cong et al., 2023; Yu et al., 2023; Tian et al., 2024; Lu et al., 2024; Zou et al., 2024; Peng et al., 2025; Shi et al., 2026) time-varying interactions and connectivity, typically by encoding timestamped events or graph snapshots into temporal node representations. For example, TGN (Rossi et al., 2020) maintains node memories from timed events, TIDFormer (Peng et al., 2025) uses calendar-based temporal partitioning and interaction-level attention to capture temporal and interactive dynamics, and TGT (Shi et al., 2026) summarizes global evolutionary regularities with a temporal graph thumbnail. However, these methods primarily organize temporal information around interactions, sampled neighborhoods, or aggregate graph regularities and are mainly evaluated on node- or link-oriented prediction tasks. They are therefore not designed to explicitly represent coordinated changes to entities, relations, attributes, and local structures at multiple granularities. In contrast, WorldGraph models such structured graph changes as explicit transitions and jointly performs state modeling and node-, edge-, and graph-level transition prediction.

## J REPRODUCIBILITY AND CODE AVAILABILITY

To ensure the reproducibility of our results, we provide the source code of WorldGraph and the benchmark datasets (GWM-Zero) at https://github.com/USTC-DataDarknessLab/ Graph-Native\_World\_Modeling.

## K LIMITATIONS AND BROADER IMPACTS

## K.1 LIMITATIONS

In this work, we present an initial conceptualization of a graph-native world model. However, further work is still needed to better integrate information from diverse modalities, so as to more fully realize the potential of graph world models.

## K.2 BROADER IMPACTS

This work proposes a novel graph-native world modeling paradigm, with the potential to address existing limitations of conventional world models in learning structural information, thereby enabling more faithful modeling of relational dynamics in evolving graph worlds. Meanwhile, the core ideas of this work can be transferred to different modalities, enabling multimodal graph world modeling where diverse observations (e.g., images and text) are fused into an evolving graph representation.
# ThinkingGuard: Decoding Implicit Hazards via Step-by-Step Risk Atribution in Multimodal Large Language Models

Ruochen Zhang   
ruochen124@buaa.edu.cn   
Beihang University   
Beijing, China   
Jiahe Xie   
xiejiahe@buaa.edu.cn   
Beihang University   
Beijing, China

Yao Huang huangyao26@mails.tsinghua.edu.cn Tsinghua University Beijing, China

Yuanfang Guo   
andyguo@buaa.edu.cn   
Beihang University   
Beijing, China   
Jin Yan   
23373283@buaa.edu.cn   
Beihang University   
Beijing, China   
Yitong Sun   
yt\_sun@buaa.edu.cn   
Beihang University   
Beijing, China   
Jifan Ma   
23373442@buaa.edu.cn   
Beihang University   
Beijing, China   
Xingxing Wei<sup>✉</sup>   
xx\_wei@buaa.edu.cn   
Institute of Artificial Intelligence   
Beihang University   
Beijing, China   
State Key Laboratory of AI Safety   
Beijing, China

## Abstract

While Multimodal Large Language Models (MLLMs) are increasingly deployed in safety-critical domains, their reliability is threatened by multimodal implicit risks. Unlike explicit threats, these hazards emerge when individually benign text and neutral visual entities logically converge to induce unsafe outputs. Current detection methods fail to address this because they overlook the underlying risk activation mechanisms that govern cross-modal risk activation, leading to single-modality shortcut learning and hal lucinated rationalizations. To bridge this gap, we first construct TriggerBench, the first dataset explicitly modeling risk compositionality (5,600 instances). By formally isolating Key Elements and Trigger Elements to build counterfactual contrastive pairs, TriggerBench eliminates risk residues and forces models to perform genuine logical deduction rather than superficial pattern matching, which provides a rigorous foundation for both large-scale training and fine-grained evaluation. Building on this, we propose a Step-Supervised Structured Reasoning training framework and employ it to train ThinkingGuard, a specialized guard model. Inspired by Situation Awareness theory, we decouple implicit risk identification into progressive cognitive stages, and utilize a step-reward Monte Carlo Tree Search algorithm to explore optimal reasoning trajectories, which are then distilled into the model through Dual Constraint Preference Alignment. Extensive experiments across both standard and implicit safety benchmarks demonstrate that ThinkingGuard achieves strong performance. Project resources are available at https://github.com/FroggyChen/ThinkingGuard.

![](images/b8282c47d44c00f9aa18fbce6a7490db6f5598abbcbadc24940bc795a0382663.jpg)

CCS Concepts

• Security and privacy → Software and application security;   
• Computing methodologies → Computer vision.

## Keywords

multimodal implicit risk detection; multimodal safety; multimodal large language models; safety alignment; reasoning-based safety

Ruochen Zhang, Yao Huang, Yitong Sun, Jiahe Xie, Jin Yan, Jifan Ma, Yuanfang Guo, and Xingxing Wei. 2026. ThinkingGuard: Decoding Implicit Hazards via Step-by-Step Risk Attribution in Multimodal Large Language Models. In Proceedings ofthe 34th ACM International Conference on Multimedia (MM ’26), November 10–14, 2026, Rio de Janeiro, Brazil. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3767308.3835707

## 1 Introduction

The rapid advancement of Multimodal Large Language Models (MLLMs), such as GPT-5 [27], Claude-4.6, and Qwen3-VL [1], has driven their extensive deployment across safety-critical domains, including healthcare, finance, and education [2, 3, 14, 18, 19, 32]. While these models ofer remarkable capabilities, their integration into high-stakes applications has simultaneously surfaced a broad spectrum of safety vulnerabilities [16, 20], ranging from toxic content generation to sophisticated adversarial manipulations. Early defensive eforts primarily focused on intercepting explicit threats, such as overtly harmful image-text pairs and adversarial inputs [11, 25, 29, 31, 36]. Specialized detection methods, including HiddenDetect [15], JailDam [23], and Llama-Guard-Vision [6], have demonstrated commendable eficacy in these contexts by filtering out malicious inputs at the initial stage.

Despite these advancements, a more subtle threat class has emerged: Multimodal Implicit Risks. These hazards arise when individually benign text and visual elements interact to produce an unsafe implication. As illustrated in Fig. 1, neither the request to remove flying insects nor the kitchen scene is harmful in isolation; the risk is activated only by the interaction between the alcohol-based insecticide and the open flame. Because no single modality contains an explicit harmful signal, such risks often evade conventional safety filters. Recent methods therefore introduce cross-modal reasoning for safety detection [7, 9, 21, 22, 33, 34], but they still lack explicit modeling of the risk activation mechanism that links benign elements to unsafe outcomes.

![](images/af291212602b96fbea76088a6f43cd6d01342af8baa4994e340d937f050e0a6d.jpg)  
Figure 1: ThinkingGuard detects the hidden insecticide-flame interaction missed by previous methods.

In this work, we argue that reliably detecting implicit risks requires a model to perform logical risk attribution, tracing precisely how specific neutral visual triggers interact with latent text intent to activate a safety violation. This demands both a data foundation that faithfully encodes the compositional logic of implicit risks, and a reasoning capability that could produce grounded attributions rather than superficial pattern matches. Yet two fundamental gaps prevent existing approaches from achieving this. First, risk compositionality is absent from existing data. Current datasets are built by camouflaging explicit hazards rather than synthesizing risks from individually benign components, leaving single-modality residues that allow models to exploit statistical shortcuts and learn which elements look dangerous instead of how harmless elements become dangerous together. Second, existing trained reasoning chains are correlational rather than interactional. Without tailored supervision, existing training methods will lead to intermediate reasoning steps degenerating into post-hoc rationalization that justifies an implicitly formed verdict, rather than a genuine cross-modal relational reasoning chain grounded in visual evidence. This will cause trained guard models’ justifications to become hallucinated against subtle cross-modal interactions.

To address the data gap, we introduce TriggerBench, the first dataset to explicitly model the generative mechanism of implicit risks. We formally decouple neutral Key Elements, the entities that directly map to the text intent, from contextual Trigger Elements, the environmental cues that activate the risk upon cross-modal in teraction. Building on this decomposition, we construct contrastive safe/unsafe sample pairs via counterfactual trigger replacement: for each unsafe sample, we substitute only the trigger element with a benign environmental counterpart while strictly preserving the key element and the text query, yielding a matched safe sample that is visually near-identical yet semantically safe. These counterfactual pairs support both training and evaluation by isolating the logical conditions of risk activation and discouraging superficial shortcuts.

To address the reasoning gap, we propose a Step-Supervised Logical Reasoning training framework and employ it to train ThinkingGuard, a dedicated multimodal guard model capable of producing evidence-grounded safety judgments for implicit risk detection. Specifically, inspired by Situation Awareness (SA) theory [10, 28], we decompose implicit risk identification into a logical reasoning trajectory spanning intent summarization, entity extraction, attribute description, relational risk analysis, and comprehensive safety judgment, where each step corresponds to a distinct cognitive stage of situational understanding. To systematically search for high-quality training trajectories, we introduce SA-MCTS, a step-reward Monte Carlo Tree Search algorithm that scores each intermediate reasoning step along three dimensions: attribution accuracy over key and trigger elements, logical coherence with prior context, and visual grounding against the input image. The resulting optimal trajectories are then distilled into ThinkingGuard via Dual-Constraint Preference Alignment, which jointly optimizes full-trajectory logical consistency through trajectory-level DPO and sharpens attribution precision at the most critical reasoning pivot through crucial-step enhancement, enabling rigorous and hallucination-resistant safety judgments. The primary contributions are as follows:

• We construct TriggerBench, the first dataset to explicitly model risk compositionality for implicit multimodal hazards, comprising 5,600 image-text pairs across nine safety dimensions. By formally decoupling Key Elements from Trigger Elements and constructing contrastive pairs via counterfactual safe replacement, TriggerBench eliminates singlemodality risk residues and provides the first data foundation for evaluating and training cross-modal logical deduction.

• We propose ThinkingGuard, a specialized safety guard model for multimodal risk detection. By combining SA-MCTS-driven trajectory search with Dual-Constraint Preference Alignment, it internalizes precise risk attribution capabilities, producing evidence-grounded safety judgments.

• ThinkingGuard achieves strong performance across both implicit and general safety benchmarks, consistently outperforming leading proprietary models and specialized open-source detectors, demonstrating strong generalization and adversarial robustness.

## 2 Related Work

## 2.1 Implicit Hazard Datasets

Implicit hazard datasets aim to identify unsafe intent that is not explicit in any single modality but emerges from interactions among images, text, and contextual cues. Unlike explicit safety datasets, these benchmarks assess cross-modal and situational reasoning, since hazards often become apparent only after integrating multiple subtle signals. Recent benchmarks characterize this challenge from diferent perspectives [5, 17, 24, 30]. MSSBench focuses on situational safety, where the same query may be safe or unsafe depending on its visual context; it contains 1,820 image-query pairs with balanced safe and unsafe contexts [38]. MMIT studies multimodal implicit toxicity, where individually benign modalities become harmful when combined, and includes 2,100 statements and prompts across 7 risk categories, 31 subcategories, and 5 crossmodal correlation modes [9]. Broader benchmarks such as USB extend evaluation to diverse risk categories, modality combinations, and both vulnerability and oversensitivity settings [37]. Overall, existing datasets establish the dificulty of implicit multimodal safety, but mainly conceal risks within individual modalities rather than explicitly modeling their logical composition.

## 2.2 Multimodal Risk Detection

Existing multimodal risk detection methods can be broadly divided into direct classification and reasoning enhanced approaches. Direct methods, such as the OpenAI Moderation API and Llama Guard 3 Vision [6], formulate detection as category prediction or binary safe/unsafe classification. Although eficient and practical, they often rely on predefined taxonomies and surface level signals, limiting their ability to detect risks arising from subtle context or crossmodal interactions. Recent work therefore incorporates reasoning into multimodal safety detection [5, 35]. ShieldVLM performs crossmodal deliberative reasoning for implicit toxicity detection [9], while GuardReasoner and its multimodal extension combine structured reasoning with preference optimization for complex safety judgments [21]. Other methods focus on adversarial or jailbreak detection [12, 39]: JailDAM uses adaptive policy memory to detect unsafe prompts at inference time [23], whereas HiddenDetect identifies jailbreak attempts by monitoring hidden states without additional fine-tuning [15]. These developments reflect a shift from direct classification toward context aware, reasoning-based detection. Nevertheless, existing approaches still struggle with implicit risks involving fine-grained relations, long-range context, or latent trigger conditions. ThinkingGuard addresses this limitation through logical risk attribution and preference alignment.

## 3 TriggerBench

The core bottleneck in existing implicit risk detection lies in the lack of precision in risk attribution. Due to the absence of explicit malicious symbols, models often fall into two failure modes: (1) risk perception failure, leading to false negatives as the precise source of the risk cannot be located; and (2) shortcut learning, causing false positives by erroneously binding regular sensitive elements in the scene with danger labels. To overcome this, we construct TriggerBench, a fine-grained implicit risk dataset comprising 5,600 image-text pairs, with 3,864 samples for training and 1,736 for testing. It is built via a dual-phase pipeline: generating ruleviolating unsafe samples (Section 3.1) and constructing contrastive safe counterparts via counterfactual replacement (Section 3.2).

## 3.1 Rule-Guided Risk Synthesis

We observe that implicit multimodal risks typically manifest only under specific scenarios and physical constraints, despite both the text and image being completely benign when evaluated in isolation.

We define the image content that directly maps to the text intent as the Key Element $\left( E _ { K } \right)$ , and the contextual image content that induces the risk as the Trigger Element $( E _ { T } )$ . Consequently, we formally define the generative mechanism of implicit risks as a logical rule violation triggered by cross-modal interaction:

$$
\ X ( I , T ) = 1 [ \Psi ( E _ { K } \otimes E _ { c o n t e x t } , H ) | = \mathcal { V } _ { u n s a f e } ]\tag{1}
$$

where Ψ extracts the semantic consequence of the cross-modal interaction (⊗) between $E _ { K }$ and the visual context, coupled with the hidden text intent �. The risk is activated (evaluates to 1) only if this consequence logically entails (|=) a predefined set of safety violation rules $\mathcal { V } _ { u n s a f e } .$

Guided by this formulation, we design a multi-stage synthesis pipeline to systematically build TriggerBench from scratch. First, we leverage advanced closed-source models to design risk scenarios across nine major dimensions, including physical harm, illegal activities, privacy, property damage, ethical violations, region and belief, ofensiveness, misinformation and violence. For each scenario, the model plans a core unsafe tuple $\{ E _ { K } , E _ { T } , H \}$ . The model then writes the corresponding input text � and a fine-grained image description for the unsafe scene $( I _ { d e s } ) _ { : }$ , actively ensuring that neither the text nor the image description contains explicit harmful concepts. We formally define the construction constraint for each risk sample $X _ { j } ^ { i } \mathrm { \mathbf { : } }$

$$
\begin{array} { r l } { X _ { j } ^ { i } = \{ T , I \} \ } & { { \mathrm { s . t . } } \quad \Psi ( E _ { K } \otimes E _ { T } , H ) \ \lvert = \mathcal { V } _ { u n s a f e } , } \\ & { \ \Psi ( I ) , \Psi ( T ) \ \lvert \neq \mathcal { V } _ { u n s a f e } } \end{array}\tag{2}
$$

Specifically, the indices (�, �) represent the hierarchical structure of our benchmark: � denotes the index of the safety dimension $( i \in \{ 1 , . . . , 9 \} )$ as categorized in our taxonomy, while � identifies the specific unsafe sample generated within that dimension.

Next, we employ the state-of-the-art FLUX.2-klein-9b model to render the textual description $I _ { d e s }$ into a high-fidelity image �. To ensure strict semantic alignment, visual realism, and the precise instantiation of the intended cross-modal risks, all generated image-text pairs undergo rigorous manual filtering by domain experts. Detailed annotation procedures, quality-control criteria, and reliability analyses are provided in the supplementary material. This pipeline ultimately yields a high-quality foundational dataset $D = \{ X _ { j } ^ { i } \}$ comprising 2,800 implicit risk samples.

## 3.2 Construction of Triggered Risk Pairs

Lacking fine-grained risk attribution capabilities, existing models are often overly sensitive to neutral key entities $\left( E _ { K } \right)$ that frequently appear in unsafe samples (e.g., erroneously associating any scalpel with physical harm). To mitigate this issue and decouple the core entity from the risk, we propose a Counterfactual Safe Replacement strategy. Building upon the safe environmental counterparts $( E _ { T } ^ { \prime } )$ and their corresponding image descriptions $( \hat { I } _ { d e s } )$ that were premeditated during our initial synthesis pipeline, we construct a contrastive safe sample for each data point in �, forming a matched set of safe pairs $\hat { D } = \{ \hat { X } _ { i } ^ { i } \}$

Counterfactual Safe Replacement Mechanism. The core logic lies in substituting the unsafe trigger $E _ { T }$ with the benign environment $E _ { T } ^ { \prime } ,$ , while strictly preserving the original key element $E _ { K }$ and the text query � . We formalize this mechanism as:

$$
\hat { X } _ { j } ^ { i } = \{ T , \hat { I } \} \quad \mathrm { s . t . } \quad \Psi ( E _ { K } \otimes E _ { T } ^ { \prime } , H ) \ \not \vdash \mathcal { V } _ { u n s a f e }\tag{3}
$$

![](images/b0141b78a179a443d174e42edc19b340549bc5ec01274dd6c1f12fca87bf21a7.jpg)  
Figure 2: TriggerBench data construction pipeline. We first construct implicit risk samples by combining sensitive entities with risk triggering conditions, then generate counterfactual safe samples by replacing the trigger entity with a safe counterpart.

To physically realize this intervention, we again utilize FLUX.2 to render $\hat { I } _ { d e s }$ into the safe image <sup>ˆ</sup>�. During this process, we enforce strict visual consistency, ensuring that <sup>ˆ</sup>� retains the exact semantic presence of $E _ { K }$ from the unsafe image, altering solely the surrounding contextual trigger. This approach fundamentally strips away the implicit risk of the image-text combination while maintaining extremely high structural and semantic similarity. By confronting models with these counterfactual pairs $( X _ { j } ^ { i } , \hat { X } _ { j } ^ { i } )$ , we force the MLLM to learn the precise logical dependencies between specific triggers and risk activation. This efectively eradicates the model’s reliance on spurious statistical correlations, thereby laying a solid data foundation for our subsequent critical evaluation and preference alignment.

## 4 ThinkingGuard

Unlike explicit risks, implicit risks lack intuitive violation cues, posing severe challenges to the reasoning coherence and attribution precision of MLLMs. To address this, we propose a risk attribution framework based on reasoning trajectory search. We formally model this multi-step reasoning process in Section 4.1, detail the SA-MCTS-driven optimal trajectory exploration in Section 4.2, and introduce the Dual-Constraint Preference Alignment with crucialstep enhancement in Section 4.3.

## 4.1 Logical Risk Attribution

Inspired by Situation Awareness (SA) theory [10, 28], we view implicit risk identification not as a one-shot classification problem, but as a progressive process of situational understanding. SA character izes decision-making in dynamic environments as the perception of task-relevant elements, the comprehension of their meaning, and the projection of their future status. Following this perspective, we explicitly decompose MLLM-based risk analysis into a logical multi step reasoning trajectory, such that latent intent, perceptual cues, contextual relations, and final risk judgment can be progressively constructed rather than entangled in a single generation.

To achieve fine-grained deconstruction of implicit risks, we formalize the risk analysis process of MLLMs as a multi-step reasoning trajectory generation framework $\mathcal { F } = \{ S , \mathcal { A } , \mathcal { P } , \mathcal { R } \}$ . Under this framework, a complete risk reasoning process is modeled as a trajectory containing multiple discrete steps $\tau = ( a _ { 1 } , a _ { 2 } , a _ { 3 } , . . . , a _ { n } )$ , where each action $a _ { t } \in \mathcal { A }$ corresponds to the text generated by the MLLM at a specific analysis depth, and S represents the contextual state space during the analysis.

The initial state � ∈ S contains the image-text pair to be tested and a carefully designed system prompt template. During the trajectory evolution, the state transition mechanism P denotes text concatenation, i.e., $s _ { t } = s _ { t - 1 } \oplus a _ { t }$ (where ⊕ represents string concatenation). From the perspective of SA, this trajectory follows a goal-oriented situational reasoning process. It begins by establishing a task-specific interpretive frame through the summarization of the user’s latent intent, providing the semantic basis for subsequent analysis. Reasoning then advances to the perception stage, focusing on the multimodal extraction of key entities from the text-image pair and the refined description of suspicious attributes. Building upon this perceptual grounding, the process captures the comprehension of contextual meaning by evaluating entity relationships and potential triggers to unearth implicit risks. Finally, the trajectory performs projection-informed safety judgment, integrating intent with these relational cues to produce a comprehensive evaluation resulting in the final safety verdict and risk categorization.

To strictly constrain the generation paradigm of the MLLM, we inject analysis examples into � and assign exclusive stop-tokens for each action �<sub>�</sub> to achieve precise step segmentation. This design enforces the risk reasoning process to unfold as a sequence of explicit intermediate states, where situational understanding is progressively constructed rather than expressed only at the final step. Accordingly, the fine-grained reward function R provides process feedback for each intermediate state $s _ { t } ,$ , so that every stage of the trajectory can be evaluated according to its contribution to the evolving risk awareness. This step-level scoring mechanism not only guides the MCTS path optimization in Section 4.2 but also forms the basis for constructing preference pairs in Section 4.3.

![](images/3c3187a732dc0b2b1a3459f3a55bf0149d396c610d107005c966ca039434a80a.jpg)  
Figure 3: Framework overview. Given fine-grained data, we use SA-MCTS to explore and score reasoning trajectories with reward-guided expansion and pruning, and then synthesize both full-trajectory and crucial-step preference pairs for training.

## 4.2 SA-MCTS for Trajectory Exploration

Traditional MCTS relies on sparse terminal rewards typical of welldefined game environments [4, 8]. However, in the context of implicit multimodal risk detection, relying solely on a final risk judg ment is insuficient to validate the correctness of intermediate log ical leaps within the vast open-ended text generation space. To construct rigorous multi-step structured logic chains, we propose Situation-Aware MCTS (SA-MCTS). This mechanism shifts the focus from standard tree search to progressive expansion, finegrained process reward evaluation, and dynamic logical pruning.

Asymmetric expansion based on search space. Recognizing the varying semantic complexity across diferent reasoning depths, we implement a targeted asymmetric node expansion strategy. For the initial text intent summarization, the semantic search space is relatively constrained. Therefore, we allocate a smaller number of candidate branches to eficiently establish a stable semantic baseline. Conversely, for the subsequent stages, which involve intricate multimodal interactions and critical risk analysis step, the reasoning space expands exponentially. Accordingly, we dynamically increase the branching factor to facilitate broader heuristic exploration. This tailored expansion mechanism efectively balances computational eficiency with the necessary exploration breadth, avoiding redundant resource allocation in low-variance analytical steps.

Fine-Grained Process Reward Evaluation. To provide steplevel supervision for the reasoning trajectory, we employ an LLMbased process reward model. Given a fixed evaluation prompt template $\mathcal { P } _ { \mathrm { r e w a r d } }$ , the evaluator independently scores each candidate reasoning step $\left( { { s _ { t - 1 } } , { a _ { t } } } \right)$ as follows:

$$
\mathcal { R } _ { \mathrm { s t e p } } ( \boldsymbol { a } _ { t } \mid \boldsymbol { s } _ { t - 1 } ) = \mathrm { J u d g e } _ { \mathrm { L L M } } \left( \boldsymbol { \mathcal { P } } _ { \mathrm { r e w a r d } } ; \boldsymbol { s } _ { t - 1 } , \boldsymbol { a } _ { t } , E _ { K } , E _ { T } , I \right) ,\tag{4}
$$

where $\mathrm { J u d g e } _ { \mathrm { L L M } }$ denotes the LLM-based reward evaluator, which produces a scalar reward by jointly considering the three criteria specified in $\mathcal { P } _ { \mathrm { r e w a r d } } \colon$ attribution accuracy, coherence and conciseness, and visual hallucination. These criteria are assessed holistically rather than combined using manually specified weighting coeficients. Their design is motivated by our preliminary error analysis of MLLM reasoning trajectories. Further details and human validation are provided in the supplementary material.

Attribution Accuracy (Attr): Evaluates whether the current action correctly identifies and reasons about the key entity $E _ { K }$ and the trigger entity $E _ { T } ,$ ensuring that the reasoning follows the appropriate attribution path.

Coherence and Conciseness (Consist): Evaluates the logical consistency between the current action �<sub>�</sub> and the historical context $s _ { t - 1 }$ , while considering whether the reasoning is concise, unambiguous, and informative.

Hallucination Penalty (Hallu): Evaluates whether the factual claims in the current action are grounded in the input image �. Unsupported speculation or descriptions that deviate from the visual evidence result in a lower reward.

Dynamic Max-Q Backpropagation and Pruning. In the backpropagation phase, standard average-Q updates can dilute the value of highly insightful reasoning chains. To preserve the optimal attribution logic, we introduce a Dynamic Subtree Max-Q update strategy. The value � of a non-leaf node is determined directly by the maximum evaluation value discovered within its subtree:

$$
\begin{array} { r } { Q ( s , a ) = \left\{ \begin{array} { l l } { \mathcal { R } _ { s t e p } , } & { \mathrm { i f ~ c h i l d r e n } = 0 } \\ { \operatorname* { m a x } _ { a ^ { \prime } \in \mathrm { c h i l d r e n } } Q ( s \oplus a , a ^ { \prime } ) , } & { \mathrm { o t h e r w i s e } } \end{array} \right. } \end{array}\tag{5}
$$

Guided by the fine-grained $\mathcal { R } _ { s t e p }$ scores, a node whose reward falls below a predefined threshold is immediately terminated, and its branch is pruned due to logical contradictions or severe hallucinations. Max-Q backup then propagates the value of the best surviving descendant, allowing SA-MCTS to bypass logical deadends and focus on promising reasoning trajectories.

In summary, by integrating reasoning-aware asymmetric expansion and dynamic pruning, this SA-MCTS framework significantly improves search eficiency within the vast open-ended reasoning space. Guided by precise step-level rewards, it not only establishes rigorous attribution paths during the search phase but also generates high-quality, high-contrast trajectory pairs for subsequent preference alignment.

## 4.3 Dual-Constraint Preference Alignment

To distill the reasoning expertise discovered by SA-MCTS into the MLLM, we construct a fine-grained preference dataset and perform alignment via Crucial-step Enhancement.

Full-Trajectory Preference Construction. For both unsafe and benign samples, we harvest complete reasoning paths from the search trees. The path with the highest cumulative process reward is designated as the optimal chosen trajectory $\tau _ { w } = ( a _ { 1 } ^ { w } , \ldots , a _ { n } ^ { w } )$ . Conversely, we identify a rejected trajectory $\tau _ { l } = ( a _ { 1 } ^ { l } , \ldots , a _ { n } ^ { l } )$ that deviates into logical flaws or visual hallucinations, yet maintains a high initial semantic similarity to efectively challenge the model. This forms the trajectory-level preference dataset $\mathcal { D } _ { t r a j } = \{ ( x , \tau _ { w } , \tau _ { l } ) \}$ where � represents the initial image-text input.

Crucial-Step Enhancement for Risk Analysis. While fulltrajectory optimization guarantees overall consistency, implicit risk detection inherently hinges on specific analytical pivots. For unsafe samples, the core dificulty lies in uncovering the precise interaction between the key entity $E _ { K }$ and the implicit trigger $E _ { T }$ (typically occurring at relationship risk analysis & comprehensive evaluation). Errors at this critical junction lead to systemic attribution failure. In contrast, benign samples generally only require a consistent enumeration of facts. Recognizing this asymmetry, we introduce a step-level enhancement mechanism exclusively for unsafe data. We extract the specific state-action pairs at this critical pivot $t _ { c }$ from the search tree. By pairing the optimal critical analysis $a _ { c } ^ { w }$ against the lowest-scoring sibling branch $a _ { c } ^ { l }$ under the identical historical context $s _ { c - 1 }$ , we construct the crucial-step dataset $\mathcal { D } _ { \mathit { p i v o t } } = \{ ( s _ { c - 1 } , a _ { c } ^ { w } , a _ { c } ^ { l } ) \}$

Ultimately, this dual-constraint alignment empowers the MLLM to internalize the precise risk-unmasking mechanics without the need for expensive tree searching during inference.

## 5 Experiments

## 5.1 Experimental Settings

Implementation Details. We use Qwen3-VL-8B-Instruct [1] for eficient trajectory exploration and as the backbone for the MLLM detection model. SA-MCTS uses a maximum depth of 5, branching factor of 2, and 30 iterations; ofline search averages 352 s per sample on one A100 GPU. ThinkingGuard is trained on TriggerBench-Train with LoRA-DPO [13, 26] on four A100-80GB GPUs for approximately 90 min in total, sampling full-trajectory and crucial-step pairs at a 2:1 ratio. We use a learning rate of $3 \times 1 0 ^ { - 5 }$ , batch size of 32, LoRA rank/alpha of 8/16, and DPO $\beta = 0 . 1$ . Ultimately, we adopt a 2:1 ratio of full-trajectory to crucial-step pairs, as validated in the supplementary material.

Datasets and Metrics. We employ diverse benchmarks for evaluation, including the TriggerBench-Test, our proposed benchmark for trigger risk detection; implicit-hazard benchmarks (MSSBench, MMIT, and USB); and general-safety and jailbreak benchmarks (VLSBench, SafeBench, MMSafetyBench, and JailbreakV-28K-mini). On TriggerBench, we report dimension-wise F1-Unsafe and Recall, together with their averages over all nine safety dimensions. For other benchmarks containing both safe and unsafe samples, we report Accuracy, F1-Unsafe, F1-Safe, and Recall; for benchmarks containing only unsafe samples, we report Accuracy.

Baselines. We compare ThinkingGuard with representative closed-source multimodal models (GPT-5.1 and Claude-Sonnet-4- $6 ) _ { ; }$ a commercial moderation system (OpenAI Moderation API), open-source safety detectors (Llama-Guard3-Vision, ShieldVLM, and GuardReasoner-VL), and jailbreak detectors (HiddenDetect and JailDam). ShieldVLM and GuardReasoner-VL are evaluated using their publicly released checkpoints, both based on Qwen2.5-VL-7B. These baselines span black-box moderation and reasoning-based safety detection paradigms. Prompts and evaluations of closedsource models are provided in the supplementary material.

## 5.2 Research Questions (RQs) and Findings

RQ1: Can existing methods address the challenges posed by implicit hazards? Existing methods remain insuficient for implicit multimodal hazards. As shown in Table 1, TriggerBench reveals a clear limitation of current safety models when the risk is implicit and must be inferred from interactions among visual entities, contextual cues, and latent triggers. Specialized moderation systems and jailbreak detectors including OpenAI Moderation, HiddenDetect, and JailDam exhibit low performance with average scores below 0.25. These models are primarily optimized for the identification of explicit keywords or adversarial linguistic patterns. Because the textual queries in TriggerBench are often semantically benign, these detectors fail to recognize the risk situated in the situational coupling between the user intent and the physical context depicted in the image. Proprietary models such as GPT-5.1 and Claude-Sonnet-4-6 achieve average scores between 0.642 and 0.676. Although these models represent advanced general-purpose multimodal systems, their end-to-end architectures frequently overlook the subtle interactions between benign entities that lead to implicit hazards. Open-source detectors and reasoning-based models such as ShieldVLM and GuardReasoner-VL achieve scores of 0.634 and 0.361, respectively, demonstrating that neither advanced detection architectures nor the mere generation of reasoning steps ensures reliable implicit risk identification. This outcome indicates that reasoning remains insuficient without a verification mechanism to maintain the accuracy and logical consistency of intermediate analytical pivots, as such models sufer from logical hallucinations where the reasoning chain is unrelated to the actual visual evidence.

Compared with leading open-source and closed-source methods, ThinkingGuard achieves a superior average score of 0.764, outperforming all baseline categories. This result validates the effectiveness of the structured risk attribution process and the finegrained preference alignment. By decomposing complex scenarios into verified intermediate steps such as entity extraction and relationship risk analysis, the model grounds its safety judgment in a comprehensive understanding of contextual triggers. The integration of Step-DPO further aligns the reasoning trajectories with verified safety principles, efectively mitigating the logical flaws observed in alternative methodologies. The results confirm that the fine-grained alignment of the reasoning process is essential for identifying hazards that emerge from the interaction of multimodal entities. Additional qualitative examples are shown in Fig. 4, which further illustrate that ThinkingGuard can consistently identify latent risks across diverse implicit-hazard scenarios.

Table 1: Performance on TriggerBench across 9 safety dimensions. Bold indicates the best result and underlined indicates the second-best result. Abbreviations: PH = Physical Harm, IA = Illegal Activities, PRI = Privacy, PD = Property Damage, EV = Ethical Violations, RB = Region & Belief, OFF = Ofensiveness, MIS = Misinformation, VIO = Violence, and AVE. = Average.
<table><tr><td rowspan="3">Method</td><td colspan="10">TriggerBench</td></tr><tr><td>PH</td><td>IA</td><td>PRI</td><td>PD</td><td>EV</td><td>RB</td><td>OFF</td><td>MIS</td><td>VIO</td><td>AVE.</td></tr><tr><td>F1 / Recall</td><td></td><td>F1 / Recall</td><td>F1 / Recall</td><td>F1 / Recall</td><td>F1 /Recall</td><td>F1 / Recall</td><td>F1 / Recall</td><td>F1 / Recall</td><td>F1 / Recall</td></tr><tr><td colspan="10">Closed-source models</td></tr><tr><td>GPT-5.1</td><td>0.718 / 0.802</td><td>0.667  / 0.619</td><td>0.720 / 0.680</td><td>0.739 / 0.820</td><td>0.533 / 0.412</td><td>0.293 / 0.189</td><td>0.626 / 0.537</td><td>0.611 / 0.505</td><td>0.870 / 0.939</td><td>0.642 / 0.619</td></tr><tr><td>Claude-Sonnet-4.6</td><td>0.727 / 0.700</td><td>0.747 / 0.653</td><td>0.686 / 0.600</td><td>0.737 / 0.656</td><td>0.583 / 0.433</td><td>0.443 / 0.300</td><td>0.746 / 0.663</td><td>0.611 / 0.458</td><td>0.804 / 0.764</td><td>0.676 / 0.587</td></tr><tr><td>OpenAI Moderation</td><td>0.024 / 0.012</td><td>0.079 / 0.042</td><td>0.039 / 0.020</td><td>0.075 / 0.040</td><td>0.040 / 0.021</td><td>0.043 / 0.022</td><td>0.080 / 0.042</td><td>0.119 / 0.063</td><td>0.387 / 0.263</td><td>0.099 / 0.063</td></tr><tr><td colspan="10">Open-source detectors</td></tr><tr><td>Llama-Guard3-Vision</td><td>0.114 / 0.062</td><td>0.272 / 0.177</td><td>0.296 / 0.210</td><td>0.369 / 0.260</td><td>0.096 / 0.052</td><td>0.021 / 0.011</td><td>0.258 / 0.168</td><td>0.144 / 0.084</td><td>0.319 / 0.228</td><td>0.210 / 0.144</td></tr><tr><td>HiddenDetect</td><td>0.153 / 0.086</td><td>0.000 / 0.000</td><td>0.139 / 0.080</td><td>0.127 / 0.070</td><td>0.060 / 0.031</td><td>0.021 / 0.011</td><td>0.079 / 0.042</td><td>0.216 / 0.126</td><td>0.155 / 0.088</td><td>0.106 / 0.060</td></tr><tr><td>JailDam</td><td>0.196 / 0.123</td><td>0.241 / 0.177</td><td>0.239 / 0.160</td><td>0.200 / 0.140</td><td>0.235 / 0.165</td><td>0.252 / 0.178</td><td>0.320 / 0.274</td><td>0.225 / 0.168</td><td>0.265 / 0.193</td><td>0.242 / 0.176</td></tr><tr><td>ShieldVLM</td><td>0.673 / 0.827</td><td>0.628 / 0.740</td><td>0.630 / 0.760</td><td>0.669 / 0.940</td><td>0.598 / 0.629</td><td>0.608 / 0.689</td><td>0.601 / 0.642</td><td>0.619 / 0.642</td><td>0.679 / 0.816</td><td>0.634 / 0.744</td></tr><tr><td>GuardReasoner-VL</td><td>0.286 / 0.185</td><td>0.456 / 0.323</td><td>0.378 / 0.270</td><td>0.443 / 0.350</td><td>0.191 / 0.113</td><td>0.175 / 0.100</td><td>0.348 / 0.253</td><td>0.256 / 0.158</td><td>0.719 / 0.684</td><td>0.361 / 0.282</td></tr><tr><td>ThinkingGuard</td><td>0.740 / 0.951</td><td>0.781 / 0.875</td><td>0.697 / 0.700</td><td>0.757 / 0.980</td><td>0.766 / 0.876</td><td>0.711 / 0.767</td><td>0.720 / 0.705</td><td>0.881 / 0.947</td><td>0.821 / 0.982</td><td>0.764 / 0.866</td></tr></table>

Table 2: Performance and inference eficiency across implicit-risk, general-safety, and jailbreak benchmarks. MSSBench and MMIT report Accuracy, F1-U, F1-S, and Recall, while USB and the four rightmost benchmarks report Accuracy. Bold and underlined values indicate the best and second-best results.
<table><tr><td rowspan="3">Method</td><td colspan="2">Efficiency</td><td colspan="8">Implicit Risk</td><td colspan="3">General Safety</td><td></td><td>Jailbreak</td></tr><tr><td></td><td></td><td></td><td colspan="3">MSSBench</td><td colspan="4"></td><td>USB</td><td rowspan="2">VLS Bench Bench</td><td rowspan="2">Safe</td><td rowspan="2">Bench</td><td rowspan="2">MMSafety JailbreakV -28K-mini</td></tr><tr><td></td><td>Lat. ↓ Tok. ↓</td><td></td><td>Acc. F1-U</td><td></td><td>F1-S Rec.</td><td>Acc.</td><td>F1-U</td><td>F1-S</td><td>Rec. Acc.</td><td></td></tr><tr><td>GPT-5.1</td><td>13.4</td><td>429</td><td>0.618</td><td>0.423</td><td>0.715</td><td>0.280</td><td>0.748</td><td>0.677</td><td>0.793</td><td>0.529</td><td>0.860</td><td>0.855</td><td>0.916</td><td>0.540</td><td>0.907</td></tr><tr><td>Claude-Sonnet-4-6</td><td>17.5</td><td>473</td><td>0.577</td><td>0.274</td><td>0.701</td><td>0.160</td><td>0.740</td><td>0.658</td><td>0.791</td><td>0.500</td><td>0.737</td><td>0.721</td><td>0.832</td><td>0.390</td><td>0.825</td></tr><tr><td>OpenAI Moderation</td><td>1</td><td>一</td><td>0.503</td><td>0.013</td><td>0.668</td><td>0.007</td><td>0.500</td><td>0.000</td><td>0.667</td><td>0.500</td><td>0.397</td><td>0.302</td><td>0.678</td><td>0.105</td><td>0.629</td></tr><tr><td>Llama-Guard3-Vision</td><td></td><td></td><td>0.500</td><td>0.000</td><td>0.667</td><td>0.000</td><td>0.524</td><td>0.091</td><td>0.677</td><td>0.048</td><td>0.297</td><td>0.037</td><td>0.554</td><td>0.296</td><td>0.650</td></tr><tr><td>HiddenDetect</td><td></td><td></td><td>0.502</td><td>0.007</td><td>0.667</td><td>0.003</td><td>0.517</td><td>0.073</td><td>0.673</td><td>0.038</td><td>0.270</td><td>0.021</td><td>0.678</td><td>0.461</td><td>0.307</td></tr><tr><td>JailDam</td><td></td><td></td><td>0.553</td><td>0.509</td><td>0.590</td><td>0.463</td><td>0.545</td><td>0.600</td><td>0.474</td><td>0.681</td><td>0.530</td><td>0.095</td><td>0.615</td><td>0.644</td><td>0.761</td></tr><tr><td>ShieldVLM†</td><td>3.7</td><td>381</td><td>0.695</td><td>0.620</td><td>0.745</td><td>0.497</td><td>0.898</td><td>0.887</td><td>0.906</td><td>0.805</td><td>0.913</td><td>0.892</td><td>0.862</td><td>0.670</td><td>0.886</td></tr><tr><td>GuardReasoner-VL</td><td>2.4</td><td>162</td><td>0.507</td><td>0.051</td><td>0.667</td><td>0.027</td><td>0.569</td><td>0.243</td><td>0.699</td><td>0.138</td><td>0.040</td><td>0.424</td><td>0.908</td><td>0.332</td><td>0.796</td></tr><tr><td>ThinkingGuard</td><td>3.5</td><td>352</td><td>0.718</td><td>0.668</td><td>0.755</td><td>0.567</td><td>0.769</td><td>0.772</td><td>0.780</td><td>0.757</td><td>0.927</td><td>0.933</td><td>0.924</td><td>0.677</td><td>0.914</td></tr></table>

<sup>†</sup> ShieldVLM is trained on MMIT, which includes part of MSSBench and partially overlaps with the data sources of USB, and also uses data sourced from VLSBench and MMSafetyBench.

RQ2: When faced with of-distribution implicit hazards data or jailbreak data, are our methods still robust? Thinking-Guard demonstrates strong generalization capabilities across diverse implicit risk benchmarks and maintains high adversarial resilience in jailbreak evaluations. As reported in Table 2, the model achieves superior performance on MSSBench and USB. While ShieldVLM exhibits leading results on MMIT, its performance is subject to the data dependencies indicated in the corresponding table footnote. In contrast, ThinkingGuard maintains competitive accuracy without such prior exposure, suggesting that the reasoning-based approach facilitates the acquisition of transferable safety principles rather than overfitting to specific benchmark distributions. The consistent gains on MSSBench (0.718 Acc.) further indicate that the situational awareness framework enables the model to resolve risk dependencies across varied task characteristics.

In terms of eficiency, ThinkingGuard requires 3.5 seconds and generates 352 output tokens per sample under our evaluation setup, which is comparable to ShieldVLM. Although GuardReasoner-VL is faster and produces shorter responses, its substantially lower detection performance indicates a less favorable trade-of between accuracy and eficiency.

Furthermore, the results in Table 2 confirm that ThinkingGuard sets new state-of-the-art performance levels on general safety benchmarks, including VLSBench, SafeBench and JailbreakV-28K-mini. The high accuracy on jailbreak scenarios (0.914) is particularly noteworthy, as these attacks typically utilize complex linguistic wrappers to bypass standard safety filters. The efectiveness of ThinkingGuard in these settings is attributed to the mandatory intent summarization and risk attribution steps, which function as a logical filter to identify latent malicious intent beneath adversarial perturbations. By prioritizing structural comprehension over superficial pattern recognition, the proposed alignment strategy ensures a more stable and robust safety judgment process. Additional results on the robustness and transferability of ThinkingGuard are provided in the supplementary material.

![](images/307b905e1da0cf5fc4bb7f3bb2f8f2d898f308864453c6963fd58f72a126f9cf.jpg)  
Figure 4: Qualitative examples on four representative implicit-risk cases. ThinkingGuard performs step-by-step reasoning over the image-text pairs, uncovers the hidden unsafe semantics, and predicts the correct safety labels and risk categories.

Table 3: Ablation study on analysis generation, preference training, and step-DPO. Results are averaged by benchmark group; General Safety & Jailbreak reports accuracy.
<table><tr><td rowspan="2">Setting</td><td colspan="3">Implicit Hazards</td><td rowspan="2">General Safety &amp; Jailbreak</td></tr><tr><td>Acc.</td><td>F1-Unsafe</td><td>Rec.</td></tr><tr><td>w/o analysis</td><td>0.753</td><td>0.592</td><td>0.505</td><td>0.674</td></tr><tr><td>w/o training</td><td>0.670</td><td>0.686</td><td>0.588</td><td>0.697</td></tr><tr><td>SFT only</td><td>0.683</td><td>0.659</td><td>0.500</td><td>0.629</td></tr><tr><td>w/o step-DPO</td><td>0.756</td><td>0.779</td><td>0.699</td><td>0.790</td></tr><tr><td>Full</td><td>0.784</td><td>0.810</td><td>0.749</td><td>0.833</td></tr></table>

## 5.3 Ablation Study

To assess the specific contributions of the reasoning architecture and alignment strategies, we conduct an ablation study across implicit-hazard, general-safety, and jailbreak benchmarks, and re port the average performance within each evaluation group. As demonstrated in Table 3, the full configuration achieves the highest performance across all safety metrics, confirming that the integration of situational reasoning and fine-grained preference alignment is essential for implicit hazard detection.

Comparing the full model to the untrained Qwen3-VL-8B-Instruct baseline (w/o training) reveals a substantial performance gap. The base model achieves an accuracy of only 0.670 and a recall of 0.588 on implicit hazards, indicating that general-purpose multimodal capabilities are insuficient for resolving complex safety dependen cies without specialized alignment. Furthermore, the removal of the structured analysis component (w/o analysis) leads to a critical decrease in unsafe recall from 0.749 to 0.505. This result underscores that situational reasoning is the primary mechanism for identifying hazards that emerge from entity interactions. Without intermediate analytical steps, the model reverts to a black-box state that lacks the capacity to verify the logical connections between benign visual elements and latent risks.

The choice of alignment paradigm also significantly influences the final performance. The configuration utilizing standard supervised fine-tuning (SFT only) yields an accuracy of 0.683, which is only marginally superior to the untrained baseline. This suggests that simple imitation of reasoning formats is inadequate for internalizing the precise decision boundaries required for implicit risk attribution. Moreover, the exclusion of the step-level DPO refinement (w/o step-DPO) reduces the model’s accuracy to 0.756 and its F1-Unsafe score to 0.779. This outcome confirms that while global trajectory optimization ensures general consistency, the specific supervision of pivotal reasoning steps provides the necessary discriminatory power to resolve the most critical analytical junctions. These findings validate that the efectiveness of ThinkingGuard is derived from the joint contribution of structured situational analysis and the dual-constraint preference alignment strategy.

## 6 Conclusion

In this paper, we propose ThinkingGuard, a multimodal guard model for implicit risk detection that shifts defense from statistical correlation to causal attribution. Specifically, we first construct TriggerBench for implicit risk evaluation as well as training, which is the first dataset to explicitly model risk compositionality by decoupling neutral Key Elements from contextual Trigger Elements. Furthermore, we introduce a Step-Supervised Structured Reasoning training framework consisting of SA-MCTS and Dual-Constraint Preference Alignment to internalize grounded, hallucination-resistant reasoning into ThinkingGuard. Extensive experiments demonstrate that ThinkingGuard achieves superior performance, grounding multimodal safety in rigorous structured reasoning.

## Acknowledgments

This work was supported in part by the National Natural Science Foundation of China under Grant 62576020 and by the Open Funding Programs of the State Key Laboratory of AI Safety.

## References

[1] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. 2025. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631 (2025).

[2] Arne Bewersdorf, Christian Hartmann, Marie Hornberger, Kathrin Seßler, Maria Bannert, Enkelejda Kasneci, Gjergji Kasneci, Xiaoming Zhai, and Claudia Nerdel. 2025. Taking the next step with generative artificial intelligence: The transfor mative role of multimodal large language models in science education. Learning and Individual Diferences 118 (2025), 102601.

[3] Gagan Bhatia, Hasan Cavusoglu, Muhammad Abdul-Mageed, et al. 2024. Fintral: A family of gpt-4 level multimodal financial large language models. In Findings ofthe Association for Computational Linguistics: ACL 2024. 13064–13087.

[4] Cameron B. Browne, Edward Powley, Daniel Whitehouse, Simon M. Lucas, Peter I. Cowling, Philipp Rohlfshagen, Stephen Tavener, Diego Perez, Spyridon Samothrakis, and Simon Colton. 2012. A Survey of Monte Carlo Tree Search Methods. IEEE Transactions on Computational Intelligence and AI in Games 4, 1 (2012), 1–43.

[5] Wei Cai, Jian Zhao, Yuchu Jiang, Tianle Zhang, and Xuelong Li. 2025. Safe semantics, unsafe interpretations: Tackling implicit reasoning safety in large vision-language models. In Proceedings of the 33rd ACM International Conference on Multimedia. 13489–13491.

[6] Jianfeng Chi, Ujjwal Karn, Hongyuan Zhan, Eric Smith, Javier Rando, Yiming Zhang, Kate Plawiak, Zacharie Delpierre Coudert, Kartikeya Upasani, and Ma hesh Pasupuleti. 2024. Llama guard 3 vision: Safeguarding human-ai image understanding conversations. arXiv preprint arXiv:2411.10414 (2024).

[7] Marta Costa-jussà, David Dale, Maha Elbayad, and Bokai Yu. 2024. Added toxicity mitigation at inference time for multimodal and massively multilingual translation. In Proceedings of the 25th Annual Conference of the European Association for Machine Translation (Volume 1). 360–372.

[8] Rémi Coulom. 2006. Eficient selectivity and backup operators in Monte-Carlo tree search. In International conference on computers and games. Springer, 72–83.

[9] Shiyao Cui, QingLin Zhang, Xuan Ouyang, Renmiao Chen, Zhexin Zhang, Yida Lu, Hongning Wang, Han Qiu, and Minlie Huang. 2025. ShieldVLM: Safeguarding the Multimodal Implicit Toxicity via Deliberative Reasoning with LVLMs: Shield VLM. In Proceedings ofthe 33rd ACM International Conference on Multimedia. 11677–11686.

[10] Mica R Endsley. 2021. Situation awareness. Handbook of human factors and ergonomics (2021), 434–455.

[11] Xiangming Gu, Xiaosen Zheng, Tianyu Pang, Chao Du, Qian Liu, Ye Wang, Jing Jiang, and Min Lin. 2024. Agent smith: A single image can jailbreak one million multimodal llm agents exponentially fast. arXiv preprint arXiv:2402.08567 (2024).

[12] Lukas Helf, Felix Friedrich, Manuel Brack, Kristian Kersting, and Patrick Schramowski. 2024. Llavaguard: An open vlm-based framework for safeguarding vision datasets and models. arXiv preprint arXiv:2406.05113 (2024).

[13] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. LoRA: Low-Rank Adaptation of Large Language Models. arXiv preprint arXiv:2106.09685 (2021).

[14] Jimin Huang, Mengxi Xiao, Dong Li, Zihao Jiang, Yuzhe Yang, Yifei Zhang, Lingfei Qian, Yan Wang, Xueqing Peng, Yang Ren, et al. 2024. Open-finllms: Open multimodal large language models for financial applications. arXiv preprint arXiv:2408.11878 (2024).

[15] Yilei Jiang, Xinyan Gao, Tianshuo Peng, Yingshui Tan, Xiaoyong Zhu, Bo Zheng, and Xiangyu Yue. 2025. HiddenDetect: Detecting jailbreak attacks against multi modal large language models via monitoring hidden states. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 14880–14893.

[16] Bohan Jin, Shuhan Qi, Kehai Chen, Xinyi Guo, and Xuan Wang. 2025. Mdit-bench: Evaluating the dual-implicit toxicity in large multimodal models. In Findings of the Association for Computational Linguistics: ACL 2025. 12552–12574.

[17] Mukul Khanna, Ram Ramrakhya, Gunjan Chhablani, Sriram Yenamandra, Theophile Gervet, Matthew Chang, Zsolt Kira, Devendra Singh Chaplot, Dhruv Batra, and Roozbeh Mottaghi. 2024. Goat-bench: A benchmark for multi-modal lifelong navigation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 16373–16383.

[18] Unggi Lee, Yoorim Son, Jaeyoon Shin, Gyuri Byun, Yunseo Lee, Junbo Koh, Minji Jeon, and Hyeoncheol Kim. 2025. LLaVA-Docent-V2: Improving Data Quality and Pedagogical Data Generation to Train Large Multimodal Models for Art Appreciation Education. In International Conference on Intelligent Tutoring Systems. Springer, 213–228.

[19] Chunyuan Li, Clif Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, and Jianfeng Gao. 2023. Llava-med: Training a large language-and-vision assistant for biomedicine in one day. Advances in Neural Information Processing Systems 36 (2023), 28541–28564.

[20] Xin Liu, Yichen Zhu, Yunshi Lan, Chao Yang, and Yu Qiao. 2024. Safety of multimodal large language models on images and texts. arXiv preprint arXiv:2402.00357 (2024).

[21] Yue Liu, Shengfang Zhai, Mingzhe Du, Yulin Chen, Tri Cao, Hongcheng Gao, Cheng Wang, Xinfeng Li, Kun Wang, Junfeng Fang, et al. 2025. Guardreasoner-vl: Safeguarding vlms via reinforced reasoning. arXiv preprint arXiv:2505.11049 (2025).

[22] Krishanu Maity, AS Poornash, Sriparna Saha, and Pushpak Bhattacharyya. 2024. ToxVidLM: A multimodal framework for toxicity detection in code-mixed videos. In Findings ofthe Association for Computational Linguistics: ACL 2024. 11130– 11142.

[23] Yi Nian, Shenzhe Zhu, Yuehan Qin, Li Li, Ziyi Wang, Chaowei Xiao, and Yue Zhao. 2025. JailDAM: Jailbreak detection with adaptive memory for vision-language model. arXiv preprint arXiv:2504.03770 (2025).

[24] Shruti Palaskar, Leon Gatys, Mona Abdelrahman, Mar Jacobo, Larry Lindsey, Rutika Moharir, Gunnar Lund, Yang Xu, Navid Shiee, Jefrey Bigham, et al. 2025. VLSU: Mapping the Limits of Joint Multimodal Understanding for AI Safety. arXiv preprint arXiv:2510.18214 (2025).

[25] Xiangyu Qi, Kaixuan Huang, Ashwinee Panda, Peter Henderson, Mengdi Wang, and Prateek Mittal. 2024. Visual adversarial examples jailbreak aligned large language models. In Proceedings ofthe AAAI conference on artificial intelligence, Vol. 38. 21527–21536.

[26] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. 2023. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. arXiv preprint arXiv:2305.18290 (2023).

[27] Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. 2025. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267 (2025).

[28] Neville A Stanton, Peter RG Chambers, and John Piggott. 2001. Situational awareness and safety. Safety science 39, 3 (2001), 189–204.

[29] Ruofan Wang, Juncheng Li, Yixu Wang, Bo Wang, Xiaosen Wang, Yan Teng, Yingchun Wang, Xingjun Ma, and Yu-Gang Jiang. 2025. Ideator: Jailbreaking and benchmarking large vision-language models using themselves. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 8875–8884.

[30] Siyin Wang, Xingsong Ye, Qinyuan Cheng, Junwen Duan, Shimin Li, Jinlan Fu, Xipeng Qiu, and Xuan-Jing Huang. 2025. Safe inputs but unsafe output: Benchmarking cross-modality safety alignment of large vision-language models. In Findings ofthe Association for Computational Linguistics: NAACL 2025. 3563– 3605.

[31] Yu Wang, Xiaofei Zhou, Yichen Wang, Geyuan Zhang, and Tianxing He. 2025. Jailbreak large vision-language models through multi-modal linkage. In Proceedings ofthe 63rd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). 1466–1494.

[32] Hanguang Xiao, Feizhong Zhou, Xingyue Liu, Tianqi Liu, Zhipeng Li, Xin Liu, and Xiaoxuan Huang. 2025. A comprehensive survey of large language models and multimodal large language models in medicine. Information Fusion 117 (2025), 102888.

[33] Huahui Yi, Kun Wang, Qiankun Li, Miao Yu, Liang Lin, Gongli Xi, Hao Wu, Xuming Hu, Kang Li, and Yang Liu. 2025. SaFeR-VLM: Toward Safety-aware Fine-grained Reasoning in Multimodal Models. arXiv preprint arXiv:2510.06871 (2025).

[34] Jialin Yuan, Ye Yu, Gaurav Mittal, Matthew Hall, Sandra Sajeev, and Mei Chen. 2024. Rethinking multimodal content moderation from an asymmetric angle with mixed-modality. In Proceedings ofthe IEEE/CVF winter conference on applications ofcomputer vision. 8532–8542.

[35] Xu Zhang, Hao Li, and Zhichao Lu. 2025. CrossGuard: Safeguarding MLLMs against Joint-Modal Implicit Malicious Attacks. arXiv preprint arXiv:2510.17687 (2025).

[36] Shiji Zhao, Ranjie Duan, Fengxiang Wang, Chi Chen, Caixin Kang, Shouwei Ruan, Jialing Tao, YueFeng Chen, Hui Xue, and Xingxing Wei. 2025. Jailbreaking multimodal large language models via shufle inconsistency. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 2045–2054.

[37] Baolin Zheng, Guanlin Chen, Hongqiong Zhong, Qingyang Teng, Yingshui Tan, Zhendong Liu, Weixun Wang, Jiaheng Liu, Jian Yang, Huiyun Jing, et al. 2025. Usb: A comprehensive and unified safety evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2505.23793 (2025).

[38] Kaiwen Zhou, Chengzhi Liu, Xuandong Zhao, Anderson Compalas, Dawn Song, and Xin Eric Wang. 2024. Multimodal situational safety. arXiv preprint arXiv:2410.06172 (2024).

[39] Yongshuo Zong, Ondrej Bohdal, Tingyang Yu, Yongxin Yang, and Timothy Hospedales. 2024. Safety fine-tuning at (almost) no cost: A baseline for vision large language models. arXiv preprint arXiv:2402.02207 (2024).
# TRACING THE EVIDENCE: FAITHFUL TOKEN ATTRIBU-TION THROUGH VISION-LANGUAGE REASONING

Bowen Yuan<sup>∗</sup>, Danny Wang<sup>∗</sup>, Ruihong Qiu, Zijian Wang, Zi Huang The University of Queensland

{bowen.yuan,danny.wang,r.qiu,zijian.wang,helen.huang}@uq.edu.au

## ABSTRACT

Large vision-language models (LVLMs) exhibit strong reasoning capabilities, yet the visual and textual evidence supporting the generated responses remains difficult to identify. Faithful token attribution explains an LVLM’s response by assigning scores that rank image and prompt tokens by how much the model relies on them, such that removing higher-ranked tokens causes the likelihood of the generated response to drop more rapidly. However, existing token-attribution methods have been developed mainly for text-based language models, and our empirical study reveals two challenges when complex multimodal sources are involved. First, the joint image-text attribution can underrepresent visual evidence relative to text, obscuring the image regions supporting the response. Second, visual evidence may influence the generated response through multiple intermediate reasoning paths, while existing methods trace only a limited subset of these paths, causing important visual contributions to be underestimated. Motivated by these insights, we introduce VTRACE, a multimodal token-attribution framework that traces input contributions through intermediate reasoning and calibrates attribution scores across modalities. VTRACE constructs pairwise attributions that highlight token-specific contributions and aggregates all forward attribution paths in closed form to account for both direct and indirect contributions. Cross-modal calibration then rescales image and text attribution scores using modality contributions estimated from response-likelihood changes, enabling a unified ranking of input tokens. Evaluations against seven baselines across six visual reasoning benchmarks demonstrate the superior attribution faithfulness. Project page: https://vtrace-attribution.github.io/.

(b) top-k (%) image tokens revealed  
Evidence NOT Traced Evidence Traced  
![](images/b0e3ed34bf08c353691bf195685874bd30e0e6a990609e9cdd094862a07e026f.jpg)

![](images/7f23597d2e4cd0c32759c07ba78a178a945f5468d9b518b77a98e8e925946fd6.jpg)

Q: How many bars have values larger than 4?  
![](images/f27e4eec317790ee88fa5f1e6287a0f6dc337635eb3aab9066a85b0b8396a424.jpg)

![](images/182e95a8313d37c35e953db60e9767d4c78f816ea2223e5b6455205a7a5c9264.jpg)  
Figure 1: Relevant visual evidence can be under-ranked in multimodal attribution. Left: Top-20% attributed tokens in two examples, with color intensity indicating attribution strength. Black boxes mark the question-relevant region, enlarged below. Right: (a) percentage of attributed image tokens and (b) normalized answer probability recovery across different selected token budgets.

## 1 INTRODUCTION

Large vision-language models (LVLMs) are increasingly capable of solving complex visual reasoning tasks by generating intermediate reasoning before arriving at a final answer (Xu et al., 2025; Chen et al., 2024b; Zhang et al., 2024b). However, observing the answer alone does not reveal which visual and textual evidence contributes to the answer, or how such evidence is used throughout the reasoning process (Stan et al., 2024; Shen, 2025). Explaining these predictions requires identifying how input evidence contributes to predictions throughout the reasoning (Pan et al., 2026; Uppaal et al., 2026; Deng et al., 2025). Token attribution provides a way to estimate these contributions: for a generated token or a target span, token attribution assigns preceding tokens an attribution score reflecting the contribution to that output (Abnar & Zuidema, 2020; Ferrando et al., 2022b; Achtibat et al., 2024). A faithful attribution should accurately reflect how input evidence contributes to the selected tokens.

Existing attribution methods typically quantify token contributions using either model-internal signals or behavioral changes under intervention. Internal-signal-based methods trace information propagation through transformer attention or token interactions to attribute predictions to preceding tokens (Abnar & Zuidema, 2020; Ferrando et al., 2022b; Achtibat et al., 2024), whereas perturbationbased methods measure how modifying input tokens changes the model’s output distribution (Zhao & Shan, 2024). More recent methods extend attribution to the reasoning process, with FlashTrace (Pan et al., 2026) tracing influence through reasoning tokens and FlowTracer (Dong et al., 2026) modeling answer-directed information flow across the reasoning trace.

Despite the advances, attribution in LVLM reasoning presents two challenges. 1) Visual evidence generally receives lower attribution scores when image and text tokens are scored together. Our empirical analysis reveals an imbalance in joint image-text attribution, where visual evidence are underrepresented relative to text. Illustrated in Figure 1, textual tokens such as answer options often receive higher attribution scores than image patches with questionrelevant visual cues. Consequently, relevant image patches can be ranked below less relevant text tokens, obscuring critical visual evidence that supports the model’s reasoning. 2) Visual evidence can be underestimated when only some of the reasoning paths to the target are traced. Relevant visual evidence may support

![](images/996709fa053df947f55de6e990273367b6060722957cdada8ac850dfd696334b.jpg)  
Figure 2: Tracing only highly-attributed paths misses evidence that many weaker paths carry.

the target through multiple intermediate reasoning tokens. Shown in Figure 2, critical visual evidence from the officer’s cap propagates through tokens such as wearing and badge. Direct attribution (a) ranks the cap only 164th among 176 image patches, while tracing the highest-attribution path (b) improves it to 51st. Aggregating all reasoning paths (c), however, raises it to 2nd, showing that limited path tracing substantially under-ranks evidence by capturing only part of its propagated contribution.

To address these challenges, we introduce VTRACE, a multimodal token-attribution framework that traces visual and textual contributions through intermediate reasoning and calibrates attribution scores across modalities. To reduce the underrepresentation of visual evidence in joint image-text ranking, VTRACE isolates token-specific contributions by removing shared components within each modality, and calibrates aggregated image and text attributions according to each modality’s effect on response likelihood. To recover visual contributions distributed across intermediate reasoning, VTRACE constructs pairwise token attributions over the full sequence and aggregates all direct and indirect reasoning paths in closed form, tracing the accumulated contributions back to the input tokens. Together, our contributions are threefold:

• We empirically characterize two critical limitations when extending token attribution to LVLMs: incomplete recovery of visual contributions through intermediate reasoning and underrepresentation of visual evidence in joint image-text rankings.

• We propose VTRACE, which combines modality-aware pairwise attribution, closed-form path aggregation, and cross-modal calibration to trace source evidence through reasoning while making image and text attribution scores comparable.

• We evaluate against seven baselines across six visual reasoning benchmarks. In joint image-text evaluation, VTRACE improves mean RISE insertion AUC by 7.1% and reduces deletion AUC by 18.0%, demonstrating more faithful attribution with competitive computational efficiency. The attribution-guided post-training experiment improves model’s visual reasoning performance, suggesting the potential of VTRACE as a learning signal for LVLM reasoning.

## 2 RELATED WORK

Our work builds on three lines of research. 1) Token attribution: Existing methods estimate which source tokens influence a target token or output span using attention aggregation (Attention Rollout, ALTI) (Abnar & Zuidema, 2020; Ferrando et al., 2022b), relevance propagation (AttnLRP) (Achtibat et al., 2024), Hessian-based sensitivity (HETA) (Pramanik et al., 2026), or perturbation-based attribution (ReAGent) (Zhao & Shan, 2024). Graph-based approaches instead trace contribution paths, where IFR builds a graph over token states and components (Ferrando & Voita, 2024), FlashTrace recursively tracks attribution through generated tokens (Pan et al., 2026), and FlowTracer models conserved flow over an attention graph (Dong et al., 2026). 2) Multimodal attribution: Chefer et al. (Chefer et al., 2021) propagate relevance through self- and cross-attention to attribute multimodal predictions to their inputs. LVLM-Interpret (Stan et al., 2024) extends attention- and relevance-based analysis to autoregressive LVLMs, while GLIMPSE (Shen, 2025) aggregates gradient-weighted attention across layers and generated tokens for response-level attribution. 3) Attribution through LVLM reasoning: Multimodal-CoT and LLaVA-CoT generate intermediate reasoning from image and text before answering (Zhang et al., 2024b; Xu et al., 2025), motivating studies of whether such reasoning faithfully reflects the evidence used by the model (Chen et al., 2024b; Balasubramanian et al., 2025). Existing attribution methods, however, only partially capture how multimodal evidence propagates through intermediate reasoning and do not address attribution-scale differences between image and text tokens. VTRACE addresses both by aggregating direct and indirect influence across the reasoning trace and calibrating cross-modal attribution scores. Detailed review is in Appendix B.

## 3 CRITICAL CHALLENGES IN MULTIMODAL ATTRIBUTION

## 3.1 PRELIMINARIES

Autoregressive LVLM Generation. Given an image I and a text prompt $q ,$ an LVLM with parameters θ generates a response $y = ( y _ { 1 } , \dotsc , y _ { n } )$ containing intermediate reasoning and a final answer:

$$
p _ { \theta } ( y \mid I , q ) = \prod _ { t = 1 } ^ { n } p _ { \theta } ( y _ { t } \mid I , q , y _ { < t } ) .\tag{1}
$$

Under this autoregressive process, input evidence can contribute to later predictions both directly and through intermediate reasoning tokens. Our goal is to attribute a selected output token or span, ranging from an individual answer token to the complete response, to its supporting input evidence by accounting for these direct and indirect contributions.

Pairwise Token Attribution. Let $\begin{array} { r l } { X } & { { } = } \end{array}$ $( x _ { 1 } , \ldots , x _ { T } )$ denote the complete sequence of input and generated tokens, with I and $\tau$ indexing image and prompt-text tokens, respectively. We estimate direct token-to-token contributions using a pairwise attribution matrix $W \in \mathbb { R } _ { \geq 0 } ^ { T \times T }$ where $W _ { i j }$ scores the contribution from an earlier source token $x _ { i }$ to a later receiver token $x _ { j }$ To trace contributions forward through the sequence, we retain only entries with $i < j$ and set $W _ { i j } = 0$ otherwise. Thus, W is strictly upper triangular. A generated token can receive contributions from earlier tokens and subsequently contribute to later tokens, allowing input evidence to propagate through intermediate reasoning. Section 4.1 defines how the entries of W are computed.

![](images/f0be6aaa55f18da10080364c34318e7ba0e8772192062640f5cff3801d513968.jpg)

Top-10 tokens (image + question)  
![](images/73864ece49ee022993c943a74502c89cf96a6a9df5a1e555caf0559aff69b466.jpg)  
Figure 3: Existing methods overlook visual cues.

## 3.2 VISUAL EVIDENCE IS OBSCURED BY TEXTUAL ATTRIBUTION

In multimodal attribution, image and text tokens are jointly ranked according to their contributions, yet existing methods exhibit a strong attribution imbalanced toward text tokens. As shown in Figure 1 Right, existing methods select only a small fraction of image tokens among their highest-ranked tokens. This imbalance is also evident in Figure 3: when the model correctly counts two dogs, neither IFR nor FlashTrace includes an image patch in its top-10 tokens, while option letters, punctuation, and question tokens occupy much of the ranking. These observations reveal attribution imbalance that pushes important visual evidence below textual tokens in attribution, causing the resulting attribution to underrepresent the visual information in multimodal reasoning. In contrast, VTRACE aligns image and text scores by their effect on response likelihood, bringing seven image patches into the top 10, producing a ranking that better reflects the visual evidence required for the answer (Section 4.3).

## 3.3 LIMITED REASONING-PATH TRACING MARGINALIZES VISUAL EVIDENCE

Existing attribution methods trace only direct or limited-hop contributions, making it difficult to trace distributed evidence back to its original sources. For instance, IFR measures direct input to answer contributions (Ferrando & Voita, 2024), while FlashTrace recursively propagates attribution through a limited number of reweighted reasoning steps (Pan et al., 2026). However, evidence from input tokens can be progressively integrated into subsequent generated tokens during reasoning, which in turn contribute to later predictions. As a result, the final prediction receives strong attribution from intermediate reasoning tokens while assigning weak attribution to the upstream visual evidence from which they originated. This is depicted in the aforementioned Figure 2. VTRACE instead aggregates evidence over all direct and indirect paths (Section 4.2), allowing information propagated through the reasoning trace to be traced back to the crucial image and text tokens.

## 4 METHODOLOGY

To address these issues, VTRACE proceeds in three stages. It first constructs a modality-aware pairwise attribution matrix, then aggregates direct and indirect contributions through intermediate reasoning, and finally calibrates image and text attribution scores for joint ranking. Figure 4 presents the framework, and Algorithm is in Appendix Algorithm 1.

## 4.1 MODALITY-AWARE PAIRWISE TOKEN ATTRIBUTION

Tracing token-specific information. Consider layer ℓ of an LVLM with H attention heads and hidden dimension d. For a receiver token $x _ { j }$ , the attention block produces the update $\Delta _ { j } ^ { ( \ell ) }$

$$
\Delta _ { j } ^ { ( \ell ) } = \sum _ { h = 1 } ^ { H } \sum _ { i \le j } \alpha _ { j i } ^ { ( \ell , h ) } f _ { i } ^ { ( \ell , h ) } ,\tag{2}
$$

where $\alpha _ { j i } ^ { ( \ell , h ) }$ is the attention weight assigned by receiver token $x _ { j }$ to source token $x _ { i } ,$ and $f _ { i } ^ { ( \ell , h ) } =$ $v _ { i } ^ { ( \ell , h ) } O ^ { ( \ell , h ) } \in \mathbb { R } ^ { d }$ is the output-projected value vector of $x _ { i }$ at head h.

As shown in Section 3.2, directly transferring token-attribution signals across modalities leads to biased multimodal attribution. We instead characterize a source by the information it contributes over the common component of its own modality. For a source $x _ { i }$ in either modality, we define the modality-aware write $\hat { \widetilde { f } } _ { i } ^ { ( \ell , h ) } \ 1$ by subtracting its modality’s mean output-projected value:

$$
\widetilde { f } _ { i } ^ { ( \ell , h ) } = f _ { i } ^ { ( \ell , h ) } - \frac { 1 } { \left| \mathcal { M } ( i ) \right| } \sum _ { r \in \mathcal { M } ( i ) } f _ { r } ^ { ( \ell , h ) } ,\tag{3}
$$

where $\mathcal { M } ( i ) = \mathcal { T }$ for $i \in \mathcal { T }$ and $\mathcal { M } ( i ) = \mathcal { T }$ for $i \in \mathcal T$ , denoting the set of input tokens belonging to the same modality as token $x _ { i }$ . This removes the modality-wise mean component from each source write, emphasizing the component that distinguishes $x _ { i }$ from other tokens within the same modality.

To quantify the attribution from source $x _ { i }$ to receiver $x _ { j }$ , we measure how strongly the source-specific write aligns with the update at $x _ { j }$ . For $i < j$ , the corresponding entry of W is:

![](images/ffc39081deba4d8b8e3f11d30faccfe03e19a31cd1831e7b935240a6e74e4b43.jpg)  
Figure 4: Overview of VTRACE. VTRACE (1) measures each pair of tokens against the average contribution of its own modality and (2) aggregates all direct and indirect paths in closed form, (3) enabling image patches and words to be faithfully ranked together.

$$
W _ { i j } = \frac { 1 } { L } \sum _ { \ell = 1 } ^ { L } \frac { \left[ \sum _ { h = 1 } ^ { H } \alpha _ { j i } ^ { ( \ell , h ) } \left. \widetilde { f } _ { i } ^ { ( \ell , h ) } , \Delta _ { j } ^ { ( \ell ) } \right. \right] _ { + } } { \left\| \Delta _ { j } ^ { ( \ell ) } \right\| _ { 2 } } , \qquad i < j ,\tag{4}
$$

and $W _ { i j } = 0$ otherwise. The inner product $\langle \cdot , \cdot \rangle$ measures alignment between the source vector and the receiver-token update, while $[ \cdot ] _ { - }$ <sub>+</sub> retains positive contributions. Thus, $W \in \mathbb { R } _ { \geq 0 } ^ { T \times T }$ is a strictly upper-triangular matrix containing the direct pairwise token attribution weights.

## 4.2 COMPOSITIONAL MULTI-HOP ATTRIBUTION: DIRECT AND INDIRECT CONTRIBUTIONS

The pairwise attribution matrix W quantifies direct contributions between source and receiver tokens. A source token $x _ { i }$ , however, can contribute to a later target $x _ { j }$ either directly or through intermediate tokens, including generated reasoning tokens. We refer to aggregating all such direct and indirect paths as compositional multi-hop (CMH) attribution. VTRACE computes CMH attribution over $W$ using a Katz-based formulation (Katz, 1953):

$$
R _ { i j } = \gamma \widehat { W } _ { i j } + \gamma ^ { 2 } ( \widehat { W } ^ { 2 } ) _ { i j } + \gamma ^ { 3 } ( \widehat { W } ^ { 3 } ) _ { i j } + \cdot \cdot \cdot = \sum _ { \tau = 1 } ^ { T - 1 } \gamma ^ { \tau } ( \widehat { W } ^ { \tau } ) _ { i j } ,\tag{5}
$$

where $\begin{array} { r } { \widehat { W } = \frac { W } { \operatorname* { m a x } _ { i , j } W _ { i j } + \epsilon } } \end{array}$ is the normalized direct-attribution matrix, and $\gamma \geq 0$ weights paths by length. Here, $\widehat { W } _ { i j }$ captures direct attribution, while $( \widehat { W } ^ { \tau } ) _ { i j }$ aggregates attribution over all length-τ paths from $x _ { i }$ to $x _ { j }$ . The finite path expansion admits the exact closed form:

$$
R = \sum _ { \tau = 1 } ^ { T - 1 } \gamma ^ { \tau } \widehat { W } ^ { \tau } = ( \mathbf { I } - \gamma \widehat { W } ) ^ { - 1 } - \mathbf { I }\tag{6}
$$

The proof is in Appendix C. Each $R _ { i j }$ aggregates the direct and indirect attribution from source $x _ { i }$ to receiver $x _ { j }$ represented by $\widehat { W } _ { i j }$

Scoring tokens by incoming and outgoing attribution. The matrix R assigns attribution to pairs of tokens. To obtain one score for each token, we consider every token $x _ { r }$ that precedes the selected receiver positions $x _ { j } \in { \mathcal { A } }$ . Its incoming attribution is the total path attribution reaching $x _ { r }$ from earlier tokens, and its outgoing attribution is the total path attribution from $x _ { r }$ to the selected receivers:

$$
\mathrm { I n } ( x _ { r } ) = \sum _ { i < r } R _ { i r } , \qquad \mathrm { O u t } _ { \cal A } ( x _ { r } ) = \sum _ { j \in \cal A } R _ { r j } .\tag{7}
$$

We define the uncalibrated attribution score of $x _ { r }$ as: $u _ { r } = \left( 1 + \operatorname { I n } ( x _ { r } ) \right) \operatorname { O u t } _ { A } ( x _ { r } )$

This score includes paths that begin at $x _ { r }$ and paths that pass through x before reaching a selected receiver $x _ { j }$ . The added 1 preserves the contribution of a token with no incoming attribution.

## 4.3 ALIGNING IMAGE AND TEXT ATTRIBUTION SCORES

The multi-hop stage produces an uncalibrated attribution score $u _ { r }$ for each upstream token. To jointly attribute image and textual tokens, we calibrate their total attribution according to each modality’s contribution to the generated output. For a modality set S, we measure this contribution by the decrease in teacher-forced log-probability when its tokens are masked:

$$
D _ { S } = \log p _ { \theta } ( y \mid X ) - \log p _ { \theta } { \big ( } y \mid \operatorname { p a d } ( S ) { \big ) } , \qquad S \in \{ \mathcal { T } , \mathcal { T } , \mathcal { L } \cup \mathcal { T } \} .\tag{8}
$$

Here, pad(S) replaces positions in S with the model’s pad-token embedding, so $D _ { S }$ measures the contribution of $\bar { \boldsymbol { s } }$ to the output. To account for image-text interactions, we use exact two-player Shapley values to divide their joint contribution:

$$
\phi _ { \cal T } = \frac { 1 } { 2 } \left( D _ { \cal \bar { T } } + D _ { \cal \bar { T } \cup \cal \mathcal { T } } - D _ { \cal \bar { T } } \right) , \qquad \phi _ { \cal \bar { T } } = \frac { 1 } { 2 } \left( D _ { \cal \bar { T } } + D _ { \cal \bar { T } \cup \cal \bar { T } } - D _ { \cal \bar { Z } } \right) .\tag{9}
$$

Let $s = \phi _ { \overline { { \cal L } } } / ( \phi _ { \overline { { \cal L } } } + \phi _ { \overline { { \cal T } } } )$ be the image share. We then rescale image-token scores by:

$$
\lambda = \frac { s } { 1 - s } \frac { \sum _ { j \in \mathcal { T } } u _ { j } } { \sum _ { i \in \mathcal { T } } u _ { i } } ,\tag{10}
$$

and leave user-text scores unchanged. This matches the total image-text attribution ratio to their estimated modality contributions while preserving the ranking within each modality.

Final VTRACE attribution score. For each input token r, the final VTRACE attribution score is:

$$
u _ { \mathrm { V T R A C E } } \left\{ \begin{array} { l l } { \lambda u _ { r } , } & { r \in \mathcal { T } , } \\ { u _ { r } , } & { r \in \mathcal { T } , } \end{array} \right. \quad r \in \mathcal { T } \cup \mathcal { T } .\tag{11}
$$

Therefore, VTRACE uses $R _ { i j }$ to aggregate direct and indirect attribution from source $x _ { i }$ to target $x _ { j } ,$ and $u _ { \mathrm { V T R A C E } }$ as the final cross-modal attribution score to jointly rank image and prompt-text tokens.

## 5 EXPERIMENTS

Benchmark Datasets and Baseline Methods (Appendix E). We conduct experiments on 6 visual reasoning benchmarks to comphrehensively evaluate our method’s capability: MMStar (Chen et al., 2024a), MathVista (Lu et al., 2024), MMMU (Yue et al., 2024), MMMU-pro (Yue et al., 2025), MathVerse (Zhang et al., 2024a) and VisualPuzzles (Song et al., 2025). They cover visually dependent understanding, mathematical reasoning, and knowledge-based visual reasoning, providing diverse settings for evaluating attribution across multimodal reasoning processes. We compare our method against existing attribution methods, including ReAGent (Zhao & Shan, 2024), HETA (Pramanik et al., 2026), FlowTracer (Dong et al., 2026), IFR (Ferrando & Voita, 2024), Attn Rollout (Abnar & Zuidema, 2020), AttnLRP (Achtibat et al., 2024), FlashTrace (Pan et al., 2026).

Implementation Details. Experiments are conducted on multiple scales of Qwen3-VL (Bai et al., 2025) and InternVL3.5 (Wang et al., 2025) (default Qwen3-VL-8B). For each sample, we generate one response containing intermediate reasoning and a final answer, and keep the response fixed across all attribution methods to ensure fairness. The main evaluation attributes the complete response to the input tokens. Following prior work on attribution for reasoning models (Pan et al., 2026), we evaluate attribution faithfulness using RISE (Petsiuk et al., 2018) and MAS (Chase Walker et al., 2024) insertion and deletion metrics, where tokens are ranked by attribution score and progressively removed or restored. Deletion measures how quickly the response degrades when highly attributed tokens are removed, while insertion measures how quickly it recovers when they are restored. We report both metrics under image-only and joint settings, ranking visual tokens alone or visual and textual tokens together, respectively. For an input perturbed at step k, we score the model using the normalized likelihood of the original generated trace $\begin{array} { r } { y \colon f ( X ) _ { k } = \exp \Bigl ( \frac { 1 } { n _ { \mathrm { g e n } } } \sum _ { t = 1 } ^ { n _ { \mathrm { g e n } } } \log p _ { \theta } \bigl ( y _ { t } \mid X ^ { ( k ) } , y _ { < t } \bigr ) \Bigr ) } \end{array}$ where $X ^ { ( k ) }$ denotes the perturbed context at step k. Detailed implementation and evaluation setups are provided in Appendices F and G.

Table 1: RISE attribution faithfulness on Qwen3-VL-8B across six benchmarks under Image and Joint perturbation settings. We report RISE insertion (Ins.↑, higher is better) and deletion (Del.↓, lower is better) AUC. The Image variant perturbs image patch tokens only, while Joint perturbs both image and text tokens. Best and Runner-up are highlighted.
<table><tr><td>Dataset</td><td>Setting</td><td>RISE</td><td>ReAGent</td><td>HETA</td><td>FlowTracer</td><td>IFR</td><td>Attn Rollout</td><td>AttnLRP</td><td>FlashTrace</td><td>VTRACE</td></tr><tr><td rowspan="4">MMStar</td><td rowspan="2">Image</td><td>Ins.↑</td><td>0.505</td><td>0.497</td><td>0.532</td><td>0.539</td><td>0.539</td><td>0.563</td><td>0.555</td><td>0.600</td></tr><tr><td>Del.↓</td><td>0.458</td><td>0.463</td><td>0.412</td><td>0.409</td><td>0.439</td><td>0.384</td><td>0.394</td><td>0.352</td></tr><tr><td rowspan="2">Joint</td><td>Ins.↑</td><td>0.371</td><td>0.489</td><td>0.506</td><td>0.507</td><td>0.501</td><td>0.529</td><td>0.554</td><td>0.581</td></tr><tr><td>Del.↓</td><td>0.325</td><td>0.245</td><td>0.226</td><td>0.222</td><td>0.273</td><td>0.219</td><td>0.204</td><td>0.180</td></tr><tr><td rowspan="4">MathVista</td><td rowspan="2">Image</td><td>Ins.↑</td><td>0.500</td><td>0.514</td><td>0.579</td><td>0.591</td><td>0.584</td><td>0.599</td><td>0.600</td><td>0.662</td></tr><tr><td>Del.↓</td><td>0.447</td><td>0.444</td><td>0.368</td><td>0.363</td><td>0.394</td><td>0.358</td><td>0.354</td><td>0.311</td></tr><tr><td rowspan="2">Joint</td><td>Ins.↑</td><td>0.382</td><td>0.467</td><td>0.517</td><td>0.516</td><td>0.501</td><td>0.512</td><td>0.574</td><td>0.647</td></tr><tr><td>Del.↓</td><td>0.345</td><td>0.307</td><td>0.274</td><td>0.269</td><td>0.331</td><td>0.274</td><td>0.241</td><td>0.178</td></tr><tr><td rowspan="4">MMMU</td><td rowspan="2">Image</td><td>Ins.↑</td><td>0.543</td><td>0.564</td><td>0.595</td><td>0.610</td><td>0.627</td><td>0.612</td><td>0.618</td><td>0.665</td></tr><tr><td>Del.↓</td><td>0.482</td><td>0.480</td><td>0.438</td><td>0.425</td><td>0.444</td><td>0.419</td><td>0.417</td><td>0.379</td></tr><tr><td rowspan="2">Joint</td><td>Ins.↑</td><td>0.368</td><td>0.589</td><td>0.583</td><td>0.608</td><td>0.604</td><td>0.615</td><td>0.629</td><td>0.660</td></tr><tr><td>Del.↓</td><td>0.315</td><td>0.216</td><td>0.204</td><td>0.195</td><td>0.241</td><td>0.191</td><td>0.178</td><td>0.159</td></tr><tr><td rowspan="4">MMMU-Pro</td><td rowspan="2">Image</td><td>Ins.↑</td><td>0.527</td><td>0.542</td><td>0.570</td><td>0.572</td><td>0.604</td><td>0.593</td><td>0.583</td><td>0.645</td></tr><tr><td>Del.↓</td><td>0.472</td><td>0.472</td><td>0.442</td><td>0.435</td><td>0.444</td><td>0.413</td><td>0.429</td><td>0.378</td></tr><tr><td rowspan="2">Joint</td><td></td><td></td><td>0.558</td><td>0.532</td><td>0.539</td><td>0.554</td><td>0.559</td><td>0.584</td><td></td></tr><tr><td>Ins.↑ Del.↓</td><td>0.371 0.322</td><td>0.241</td><td>0.242</td><td>0.242</td><td>0.269</td><td>0.229</td><td>0.212</td><td>0.619 0.181</td></tr><tr><td rowspan="4">MathVerse</td><td rowspan="2">Image</td><td>Ins.↑</td><td>0.523</td><td>0.565</td><td>0.636</td><td>0.640</td><td>0.624</td><td>0.610</td><td>0.644</td><td>0.684</td></tr><tr><td>Del.↓</td><td>0.454</td><td>0.430</td><td>0.341</td><td>0.345</td><td>0.384</td><td>0.360</td><td>0.341</td><td>0.296</td></tr><tr><td rowspan="2">Joint</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Ins.↑ Del.↓</td><td>0.383 0.330</td><td>0.495 0.314</td><td>0.539 0.279</td><td>0.540 0.283</td><td>0.513 0.342</td><td>0.547 0.274</td><td>0.568 0.241</td><td>0.606 0.187</td></tr><tr><td rowspan="4">VisualPuzzles</td><td rowspan="2">Image</td><td>Ins.↑</td><td>0.456</td><td>0.461</td><td>0.468</td><td>0.511</td><td>0.525</td><td>0.531</td><td>0.510</td><td>0.557</td></tr><tr><td>Del.↓</td><td>0.409</td><td>0.405</td><td>0.379</td><td>0.354</td><td>0.380</td><td>0.324</td><td>0.358</td><td>0.309</td></tr><tr><td rowspan="2">Joint</td><td>Ins.↑</td><td>0.376</td><td>0.488</td><td>0.477</td><td>0.504</td><td>0.523</td><td>0.529</td><td>0.530</td><td>0.571</td></tr><tr><td>Del.↓</td><td>0.348</td><td>0.288</td><td>0.291</td><td>0.283</td><td>0.283</td><td>0.255</td><td>0.268</td><td>0.207</td></tr></table>

## 5.1 QUANTITATIVE ANALYSIS

VTRACE consistently outperforms existing attribution methods under both Image and Joint evaluation. As shown in Table 1, VTRACE outperforms all baselines across six benchmarks. Under the Image setting, VTRACE improves average RISE insertion by 6.8% and reduces deletion by 9.3%. Under the Joint setting, it improves insertion by 7.1% and reduces deletion by 18.0%. These consistent gains indicate that VTRACE more faithfully identifies and ranks the visual and textual tokens contributing

Table 2: Ablation study on different components
<table><tr><td rowspan="2">Method</td><td>Image</td><td>Joint</td></tr><tr><td>Ins. ↑ Del. ↓</td><td>Ins. ↑ Del. ↓</td></tr><tr><td>VTRACE</td><td>0.631 0.327</td><td>0.610 0.187</td></tr><tr><td>w/o centering</td><td>0.625 0.340</td><td>0.599 0.200</td></tr><tr><td>w/o multi-hop</td><td>0.578 0.362</td><td>0.470 0.273</td></tr><tr><td>w/o calibration</td><td>0.631 0.327</td><td>0.602 0.200</td></tr></table>

to the model response. Moreover, improvements in both insertion and deletion further show that highly ranked tokens are more effective at recovering the response when restored and disrupting it when removed. For clarity, the main table reports RISE only, with complete RISE and MAS results provided in Appendix H.1.

Ablation study. We conduct ablation study on each component of VTRACE, including modality centering, compositional multi-hop aggregation and cross-modal calibration. As reported in Table 2, removing each component lowers attribution faithfulness. Specifically, modality centering and multihop aggregation improve faithfulness in both variants. Calibration further improves Joint attribution while preserving the relative ranking of image tokens, complementing the token level attribution with cross-modal calibration. The sensitivity study and is in Appendix H.2.

Efficiency Analysis. VTRACE efficiently scales to full-trace attribution because it constructs the attribution matrix once and then reuses it to attribute every generated token. For all methods, we measure the full attribution runtime, from processing the fixed response to producing the final attribution scores. As a result, moving from one target span to all generated tokens adds only minimal overhead, increasing medium runtime from 0.24 s to 0.26 s (Figure 6(a)). Notably, this efficiency is achieved without sacrificing faithfulness, where Figure 6(b) shows that VTRACE attains the best faithfulness at substantially lower attribution time than baselines. Moreover, VTRACE scales favorably in memory and runtime, using 31 GB at around 3,000 tokens versus 87 GB for AttnLRP (Figure 6(c)) and remaining faster than baselines as sequence length grows (Figure 6(d)).

![](images/14a58d837aa29c60a212d80aa2ddbc1431230eae61e254b4c28b672e9112ff60.jpg)  
Figure 5: Attribution faithfulness across different LVLM sizes and families.

Table 3: Attribution faithfulness on correct and incorrect model predictions over benchmarks. We compare against the strongest baseline for each metric independently.
<table><tr><td></td><td></td><td colspan="4">Correct Predictions</td><td colspan="4">Incorrect Predictions</td></tr><tr><td>Variant</td><td>Method</td><td>Del. RISE↓</td><td>Del. MAS ↓</td><td>Ins. RISE ↑</td><td>Ins. MAS ↑</td><td>Del. RISE↓</td><td>Del. MAS ↓</td><td>Ins. RISE ↑</td><td>Ins. MAS ↑</td></tr><tr><td rowspan="2">Image</td><td>Best Baseline</td><td>0.377</td><td>0.508</td><td>0.590</td><td>0.451</td><td>0.424</td><td>0.579</td><td>0.603</td><td>0.460</td></tr><tr><td>VTRACE</td><td>0.333</td><td>0.468</td><td>0.644</td><td>0.514</td><td>0.398</td><td>0.550</td><td>0.636</td><td>0.496</td></tr><tr><td rowspan="2">Joint</td><td>Best Baseline</td><td>0.209</td><td>0.338</td><td>0.581</td><td>0.381</td><td>0.200</td><td>0.323</td><td>0.605</td><td>0.410</td></tr><tr><td>VTRACE</td><td>0.171</td><td>0.235</td><td>0.626</td><td>0.463</td><td>0.176</td><td>0.241</td><td>0.640</td><td>0.474</td></tr></table>

Table 4: Attribution-guided learning with VTRACE on Qwen3-VL-4B. Incorporating VTRACE attribution into GRPO improves the average performance across benchmarks.
<table><tr><td>Method</td><td>MMStar</td><td>MathVista</td><td>MMMU</td><td>MathVerse</td><td>MMMU-Pro</td><td>HallusionBench</td><td>RealWorldQA</td><td>MathVision</td><td>Avg.</td></tr><tr><td>Base</td><td>63.53</td><td>73.30</td><td>55.33</td><td>56.55</td><td>47.57</td><td>70.87</td><td>72.55</td><td>41.78</td><td>60.19</td></tr><tr><td>GRPO</td><td>69.53</td><td>77.40</td><td>61.89</td><td>62.64</td><td>51.68</td><td>72.77</td><td>72.29</td><td>43.75</td><td>63.99</td></tr><tr><td>+ VTRACE</td><td>70.20</td><td>77.20</td><td>65.00</td><td>65.13</td><td>51.45</td><td>73.29</td><td>73.20</td><td>46.71</td><td>65.27</td></tr></table>

VTRACE consistently improves attribution faithfulness across different LVLM architectures and sizes. Shown in Figure 5, VTRACE achieves the highest RISE insertion and lowest deletion scores on both Qwen3-VL-4B and InternVL3.5-8B across diverse benchmarks, indicating strong generalization across model families and sizes. Full results are in Appendix H.3.

VTRACE improves attribution faithfulness regardless of answer correctness. Shown in Table 3, VTRACE consistently outperforms the strongest baselines across different evaluation settings. This indicates that VTRACE can faithfully identify influential token relationships regardless of the model’s prediction correctness. Full results are in Appendix H.4.

VTRACE can also provide a meaningful learning signal for improving LVLM reasoning. Following (Dong et al., 2026), we use its token-level attribution for credit assignment in Group Relative Policy Optimization (GRPO), directly integrating traced reasoning flow into post-training. Shown in Table 4, attribution-guided GRPO improves the average performance of Qwen3-VL-4B from 63.99 to 65.27. This demonstrates that VTRACE captures token relationships that are useful for optimization, extending its value beyond interpretability to downstream reinforcement learning and post-training.

![](images/c549339dd4f492c589311603851b771c7a5b61961a9037d75402cfdea423222a.jpg)

![](images/0a87b02ef9d724a796fb9e98b54d97e93e756de8f989a85eaaf4cda9c9fc1b8d.jpg)

## 5.2 QUALITATIVE ANALYSIS

VTRACE achieves faster recovery under insertion and sharper degradation under deletion. Elaborated in Appendix H.5 Figure 11, VTRACE achieves faster recovery under insertion and sharper degradation under deletion.

![](images/bcc7030e6a9dedfb33471310816007b2ad701db435029f077083de41a4e2cbb7.jpg)

![](images/58dc6bd003a692d246b16fa3e89dc578d136a4a69163fe5d09e1f6f86d485737.jpg)  
Figure 6: Efficiency analysis.

![](images/309833175d7746d523934418dbb2d0ca183077a97c600879953b07b39a014388.jpg)

![](images/1aa717bd44c72a99636c2ab1089432c18d3040a9e73559f3fa9470f9c229cfaa.jpg)

![](images/3a535e78e5b99bc97eaca666e9b4237a417f94e4f101ba779c45d2cbe9d8b420.jpg)  
Figure 7: Qualitative study: VTRACE better traces reasoning tokens to key visual evidence.

![](images/e515cbb076ac4fe0fafda8a50c69d2273a6d4f962626c4e2dd9801a6e0b987df.jpg)  
Figure 8: Qualitative study: Attribution comparison between diverse hops and IFR.

Case study. VTRACE more clearly traces visual evidence through intermediate reasoning to the final response. Evident in Figure 7, VTRACE assigns stronger attribution to plant-related image patches, while the baseline focuses more on background regions. VTRACE’s token-level paths further connect plant-related reasoning tokens back to the corresponding visual patches, which the baseline largely misses. Notably, when top 10% tokens are recovered, image tokens account for around a quarter of VTRACE’s selected tokens, while baselines select almost none. The RISE insertion curve (bottom-right) further shows that cross-modal calibration helps VTRACE recover response likelihood with fewer inserted tokens. Moreover, Figure 8 shows that limited-hop attribution concentrates credit on nearby reasoning tokens and misses key visual evidence (e.g., IFR), reiterating the challenge we identified in Sec 3.3. By aggregating all reasoning paths, VTRACE traces this credit back to the image patches that support the answer. An extended analysis of the diverse hops and VTRACE is in Appendix H.6. Additional case studies are in Appendix I.

## 6 CONCLUSION

In this work, we investigate faithful multimodal token attribution for LVLM reasoning, with the aim of tracing generated predictions back to the faithful visual and textual evidence that supports the generation. Through empirical analysis, we find that existing methods struggle with contributions propagated through intermediate reasoning and underrepresent relevant visual evidence. To address these challenges, we propose VTRACE, which represents token attributions in a pairwise attribution matrix and considers both direct and indirect paths. VTRACE further calibrates image and text attribution to provide an unbiased cross-modal ranking. Experiments across six visual reasoning benchmarks consistently outperforms baseline methods, demonstrating that VTRACE provides more faithful multimodal attribution while maintaining efficiency.

## REFERENCES

Samira Abnar and Willem Zuidema. Quantifying attention flow in transformers. In ACL, 2020.

Reduan Achtibat, Sayed Mohammad Vakilzadeh Hatefi, Maximilian Dreyer, Aakriti Jain, Thomas Wiegand, Sebastian Lapuschkin, and Wojciech Samek. Attnlrp: Attention-aware layer-wise relevance propagation for transformers. arXiv preprint arXiv:2402.05602, 2024.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report, 2025. URL https://arxiv.org/abs/2511.21631.

Sriram Balasubramanian, Samyadeep Basu, and Soheil Feizi. A closer look at bias and chain-ofthought faithfulness of large (vision) language models. In EMNLP, 2025.

Dominic Simon Chase Walker, Kenny Chen, and Rickard Ewetz. Attribution quality metrics with magnitude alignment. In IJCAI, 2024.

Hila Chefer, Shir Gur, and Lior Wolf. Generic attention-model explainability for interpreting bi-modal and encoder-decoder transformers. In ICCV, 2021.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, and Feng Zhao. Are we on the right way for evaluating large vision-language models? In NeurIPS, 2024a.

Shuang Chen, Yue Guo, Zhaochen Su, Yafu Li, Yulun Wu, Jiacheng Chen, Jiayu Chen, Weijie Wang, Xiaoye Qu, and Yu Cheng. Advancing multimodal reasoning: From optimized cold start to staged reinforcement learning. arXiv preprint arXiv:2506.04207, 2025.

Yangyi Chen, Karan Sikka, Michael Cogswell, Heng Ji, and Ajay Divakaran. Measuring and improving chain-of-thought reasoning in vision-language models. In NAACL, 2024b.

Ailin Deng, Tri Cao, Zhirui Chen, and Bryan Hooi. Words or vision: Do vision-language models have blind faith in text? In CVPR, pp. 3867–3876. IEEE, 2025.

Zhichen Dong, Yang Li, Yuhan Sun, Weixun Wang, Yijia Luo, Zinian Peng, Taiheng Ye, Chao Yang, Wenbo Su, Yu Cheng, et al. How does reasoning flow? tracing attention-induced information flow for targeted rl in llms. arXiv preprint arXiv:2606.10646, 2026.

Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. A mathematical framework for transformer circuits. Transformer Circuits Thread, 2021. https://transformer-circuits.pub/2021/framework/index.html.

Javier Ferrando and Elena Voita. Information flow routes: Automatically interpreting language models at scale. In EMNLP, 2024.

Javier Ferrando, Gerard I. Gállego, Belen Alastruey, Carlos Escolano, and Marta R. Costa-jussà. Towards opening the black box of neural machine translation: Source and target interpretations of the transformer. In EMNLP, 2022a.

Javier Ferrando, Gerard I. Gállego, and Marta R. Costa-jussà. Measuring the mixing of contextual information in the transformer. In EMNLP, 2022b.

Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. Hallusionbench: An advanced diagnostic suite for entangled language hallucination and visual illusion in large visionlanguage models. In CVPR, 2024.

Leo Katz. A new status index derived from sociometric analysis. Psychometrika, 18(1):39–43, 1953. doi: 10.1007/BF02289025.

Sicong Leng, Jing Wang, Jiaxi Li, Hao Zhang, Zhiqiang Hu, Boqiang Zhang, Yuming Jiang, Hang Zhang, Xin Li, Lidong Bing, Deli Zhao, Wei Lu, Yu Rong, Aixin Sun, and Shijian Lu. Mmr1: Enhancing multimodal reasoning with variance-aware sampling and open resources, 2025. URL https://arxiv.org/abs/2509.21268.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In ICLR, 2024.

Wenbo Pan, Zhichao Liu, Xianlong Wang, Haining Yu, and Xiaohua Jia. Towards long-horizon interpretability: Efficient and faithful multi-token attribution for reasoning llms. arXiv preprint arXiv:2602.01914, 2026.

Vitali Petsiuk, Abir Das, and Kate Saenko. Rise: Randomized input sampling for explanation of black-box models. arXiv preprint arXiv:1806.07421, 2018.

Vishal Pramanik, Maisha Maliha, Nathaniel Bastian, and Sumit Jha. Hessian-enhanced token attribution (heta): Interpreting autoregressive llms. In ICLR, 2026.

Guanxi Shen. Glimpse: Holistic cross-modal explainability for large vision-language models. arXiv preprint arXiv:2506.18985, 2025.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Yueqi Song, Tianyue Ou, Yibo Kong, Zecheng Li, Graham Neubig, and Xiang Yue. Visualpuzzles: Decoupling multimodal reasoning evaluation from domain knowledge, 2025. URL https: //arxiv.org/abs/2504.10342.

Gabriela Ben Melech Stan, Estelle Aflalo, Raanan Yehezkel Rohekar, Anahita Bhiwandiwalla, Shao-Yen Tseng, Matthew Lyle Olson, Yaniv Gurwicz, Chenfei Wu, Nan Duan, and Vasudev Lal. Lvlm-intrepret: An interpretability tool for large vision-language models. In XAI4CV Workshop CVPR, 2024.

Rheeya Uppaal, Phu Mon Htut, Min Bai, Nikolaos Pappas, Zheng Qi, and Sandesh Swamy. Journey before destination: On the importance of visual faithfulness in slow thinking. In Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 4147–4168, 2026.

Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. In NeurIPS, 2024.

Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, Zhaokai Wang, Zhe Chen, Hongjie Zhang, Ganlin Yang, Haomin Wang, Qi Wei, Jinhui Yin, Wenhao Li, Erfei Cui, Guanzhou Chen, Zichen Ding, Changyao Tian, Zhenyu Wu, Jingjing Xie, Zehao Li, Bowen Yang, Yuchen Duan, Xuehui Wang, Zhi Hou, Haoran Hao, Tianyi Zhang, Songze Li, Xiangyu Zhao, Haodong Duan, Nianchen Deng, Bin Fu, Yinan He, Yi Wang, Conghui He, Botian Shi, Junjun He, Yingtong Xiong, Han Lv, Lijun Wu, Wenqi Shao, Kaipeng Zhang, Huipeng Deng, Biqing Qi, Jiaye Ge, Qipeng Guo, Wenwei Zhang, Songyang Zhang, Maosong Cao, Junyao Lin, Kexian Tang, Jianfei Gao, Haian Huang, Yuzhe Gu, Chengqi Lyu, Huanze Tang, Rui Wang, Haijun Lv, Wanli Ouyang, Limin Wang, Min Dou, Xizhou Zhu, Tong Lu, Dahua Lin, Jifeng Dai, Weijie Su, Bowen Zhou, Kai Chen, Yu Qiao, Wenhai Wang, and Gen Luo. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency, 2025. URL https://arxiv.org/abs/2508.18265.

xAI. Realworldqa: A benchmark for real-world spatial understanding. https://huggingface. co/datasets/xai-org/RealworldQA, 2024. Accessed: 2026-09-18.

Guowei Xu, Peng Jin, Ziang Wu, Hao Li, Yibing Song, Lichao Sun, and Li Yuan. Llava-cot: Let vision language models reason step-by-step. In ICCV, 2025.

Xiang Yue, Yuansheng Ni, Tianyu Zheng, Kai Zhang, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. MMMU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert AGI. In CVPR, 2024.

Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, et al. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark, 2025. URL https://arxiv. org/abs/2409.02813, 2025.

Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, Peng Gao, and Hongsheng Li. MATHVERSE: does your multi-modal LLM truly see the diagrams in visual math problems? In ECCV, 2024a.

Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. Multimodal chain-of-thought reasoning in language models. TMLR, 2024b.

Zhixue Zhao and Boxuan Shan. Reagent: A model-agnostic feature attribution method for generative language models. arXiv preprint arXiv:2402.00794, 2024.

## APPENDIX

We summarize the key structure of the Appendix as follow:

• Section A: Limitations

• Section B: Extended Related Work

• Section C: Proof of Equation (6)

• Section D: VTRACE Algorithm

• Section E: Baseline & Benchmark Descriptions

• Section F: Experimental Details

• Section G: Evaluation Details

• Section H: Extended Experiments

• Section I: Case Studies

## A LIMITATIONS

While VTRACE demonstrates strong effectiveness, several limitations remain. Our current evaluation focuses primarily on image-based visual reasoning, while modern LVLMs also operate in broader settings, including multi-image reasoning, long-context multimodal understanding, and longerhorizon agentic interaction. Extending VTRACE to these settings would further establish the generality of the proposed attribution framework. Another direction is to integrate attribution into model training to improve model visual reasoning behavior. Our preliminary attribution-guided learning experiments suggest that this direction is promising.

## B EXTENDED RELATED WORK

## B.1 TOKEN ATTRIBUTION AND INFORMATION FLOW

Token attribution traces how source tokens contribute to the generation of a target token or output span. Existing methods differ mainly in the model signals used to define this contribution. Attention Rollout composes attention matrices across layers to capture how information is mixed between token positions throughout the network (Abnar & Zuidema, 2020). ALTI similarly aggregates token interactions across layers, but derives them from the attention block while accounting for residual connections and layer normalization (Ferrando et al., 2022b). Chefer et al. (Chefer et al., 2021) propose to propagate relevance through attention and residual operations, while AttnLRP extends the layer-wise relevance propagation to Transformer attention and supports attribution to both input tokens and intermediate representations (Achtibat et al., 2024). HETA incorporates attention and value information together with Hessian-based sensitivity and KL divergence under token masking (Pramanik et al., 2026).

For generative models, attribution must additionally account for the role of previously generated tokens. A source token may affect the final prediction directly or indirectly through intermediate tokens in the generated sequence. ALTI+ study this setting in machine translation by tracing contributions from both the source sentence and the generated prefix (Ferrando et al., 2022a). IFR represents each prediction as a computation graph whose nodes correspond to token states and model components, allowing attribution to follow routes through internal representations (Ferrando & Voita, 2024).

Several recent approaches explicitly model indirect contribution through generated tokens. Flash-Trace aggregates attribution over output spans and recursively follows attribution absorbed by intermediate generated tokens through limited steps (Pan et al., 2026). This captures cases in which an early source influences the selected output through later reasoning tokens rather than through a single direct interaction. FlowTracer constructs an attention-based flow graph, assigns capacities to edges, reweights them according to their ability to reach the target region, and imposes local flow conservation (Dong et al., 2026). The resulting token-level flow scores are further used as feedback for reinforcement learning. Both methods therefore move beyond purely local attribution by considering routes through intermediate tokens. VTRACE instead aggregates multiple forward paths between source and receiver positions, allowing direct and multi-step effects to be represented within a single computation. The resulting incoming and outgoing contributions are then used to score earlier tokens with respect to selected receiver tokens.

A complementary family of methods estimates importance through perturbation or intervention rather than tracing internal connections. ReAGent repeatedly replaces subsets of input tokens and updates their importance according to changes in the model’s next-token prediction (Zhao & Shan, 2024). This requires only forward evaluations and does not depend on gradients or explicit access to internal attribution signals. VTRACE separates this role from its token-level attribution computation: the pairwise attribution matrix is obtained once from cached model states, while additional forward evaluations with replaced image or prompt embeddings are used only to calibrate the relative contribution of the two input modalities. In our experiments, we further evaluate attribution faithfulness by perturbing the input tokens ranked by each method and measuring the resulting change in model prediction.

## B.2 ATTRIBUTION ACROSS IMAGE AND TEXT

Attribution in large vision-language models introduces an additional challenge because evidence is distributed across heterogeneous visual and textual representations. A meaningful attribution must therefore identify important evidence within each modality fairly. Chefer et al. (Chefer et al., 2021) extend Transformer explanation methods to multimodal architectures containing self-attention, coattention, and encoder-decoder attention, propagating relevance through the attention interactions that connect visual and textual representations. LVLM-Interpret provides attention maps, relevance maps, and causal visualizations for inspecting how large vision-language models generate responses (Stan et al., 2024). GLIMPSE combines gradient-weighted attention with propagation across layers and aggregation over generated tokens to identify visual and textual evidence supporting a generated response (Shen, 2025).

## B.3 ATTRIBUTION FOR REASONING LVLM

Recent LVLMs increasingly generate explicit intermediate reasoning before producing a final answer. Multimodal-CoT generates a textual rationale conditioned on both image and language inputs and then uses this rationale to predict the response (Zhang et al., 2024b). LLaVA-CoT structures multimodal reasoning into stages including summarization, visual interpretation, logical reasoning, and conclusion, and performs search over intermediate stages during inference (Xu et al., 2025). These approaches make intermediate generated tokens an explicit part of the inference process rather than treating the model as producing only a final answer.

Prior work has also examined whether multimodal reasoning faithfully reflects the evidence used by the model. Chen et al. (Chen et al., 2024b) introduce a benchmark and metrics for evaluating reasoning consistency in vision-language models. Balasubramanian et al. (Balasubramanian et al., 2025) study whether generated reasoning reflects visual and textual biases introduced into the input. Their results show that models are less likely to acknowledge subtle visual cues than explicit textual cues, highlighting a potential mismatch between generated explanations and the evidence that actually affects the prediction. Such findings motivate attribution methods that inspect model-internal contributions rather than relying only on the content of the generated rationale.

VTRACE targets this setting by estimating how information propagates through multimodal generations. Given selected receiver tokens, which may correspond to reasoning steps, answer tokens, or an output span, it traces contributions from earlier image, prompt, and generated tokens through both direct and indirect paths. This provides a way to inspect which input tokens supports an intermediate reasoning step, how earlier reasoning tokens influence later ones, and which sources ultimately contribute to the final answer.

## C PROOF OF EQ. 6

Theorem 1 (Closed-form multi-path aggregation). Let $\widehat { W } \in \mathbb { R } ^ { T \times T }$ be the strictly upper-triangular matrix. Define:

$$
R = \sum _ { \tau = 1 } ^ { T - 1 } \gamma ^ { \tau } \widehat { W } ^ { \tau } .\tag{12}
$$

Then:

$$
R = ( { \bf I } - \gamma \widehat { W } ) ^ { - 1 } - { \bf I } .\tag{13}
$$

Proof. Because $\widehat { W }$ is strictly upper triangular, $\widehat { W } ^ { T } = \mathbf { 0 }$ . Now consider the finite matrix geometric series and its expansion:

$$
{ \cal { S } } = \sum _ { \tau = 0 } ^ { T - 1 } ( \widehat { \gamma \widehat { W } } ) ^ { \tau } = \mathbf { I } + \widehat { \gamma \widehat { W } } + \gamma ^ { 2 } \widehat { W } ^ { 2 } + \cdot \cdot \cdot + \gamma ^ { T - 1 } \widehat { W } ^ { T - 1 }\tag{14}
$$

$$
\mathbf { \Psi } = \mathbf { I } + \sum _ { \tau = 1 } ^ { T - 1 } \gamma ^ { \tau } \widehat { W } ^ { \tau } .\tag{15}
$$

Multiplying S by $\mathbf { I } - \gamma \widehat { W }$ gives:

$$
( \mathbf { I } - \gamma { \widehat { W } } ) S = S - \gamma { \widehat { W } } S .\tag{16}
$$

Substituting the expression for S, we have:

$$
\begin{array} { r l } & { ( \mathbf { I } - \gamma \widehat { W } ) S = \left( \mathbf { I } + \gamma \widehat { W } + \gamma ^ { 2 } \widehat { W } ^ { 2 } + \cdot \cdot \cdot + \gamma ^ { T - 1 } \widehat { W } ^ { T - 1 } \right) } \\ & { \phantom { \frac { 0 } { 0 } } - \left( \gamma \widehat { W } + \gamma ^ { 2 } \widehat { W } ^ { 2 } + \cdot \cdot \cdot + \gamma ^ { T } \widehat { W } ^ { T } \right) . } \end{array}\tag{17}
$$

Note that all the matching intermediate terms cancel, which leaves us with:

$$
( \mathbf { I } - \gamma \widehat { W } ) S = \mathbf { I } - \gamma ^ { T } \widehat { W } ^ { T } .\tag{18}
$$

Since $\widehat { W } ^ { T } = \mathbf { 0 }$ , this reduces to:

$$
( \mathbf { I } - \gamma { \widehat { W } } ) S = \mathbf { I } .\tag{19}
$$

Hence, we have:

$$
S = \mathbf { I } + \sum _ { \tau = 1 } ^ { T - 1 } \gamma ^ { \tau } \widehat W ^ { \tau } = ( \mathbf { I } - \gamma \widehat W ) ^ { - 1 } .\tag{20}
$$

Finally, subtracting I from both sides gives:

$$
R = \sum _ { \tau = 1 } ^ { T - 1 } \gamma ^ { \tau } \widehat { W } ^ { \tau } = ( \mathbf { I } - \gamma \widehat { W } ) ^ { - 1 } - \mathbf { I } ,\tag{21}
$$

as required.

## D VTRACE ALGORITHM

Algorithm 1 gives the full procedure in pseudocode. Its input is the cached state of a single forward pass over the frozen trace: the value vectors, the attention weights, and the residual stream before and after each attention block.

• Step 1 computes the modality-aware pairwise attribution matrix W from the cached model states, as described in Section 4.1.

• Step 2 normalizes W to obtain Wc, computes the multi-hop attribution matrix R in closed form, and scores each source token using its incoming attribution and outgoing attribution to the selected receivers, as described in Section 4.2.

• Step 3 is the modality calibration of Section 4.3: three further forward passes measure the loss increase when the image, the text, or both are replaced by padding, the two modality weights are the Shapley values of this two player game, and the image scores are rescaled so that their share of the total matches the image weight while every within-modality order is left unchanged. This ensures the image and text tokens are well calibrated to be ranked fairly and faithfully.

Calibration boundary conditions. Calibration is applied over image and query token positions when used in perturbation evaluation. Calibration is applied only when both modality contributions from Equation (9) and both sums of token attribution scores are positive. If any of the quantities is zero or negative, the original scores are maintained. When applied, calibration multiplies image scores by the positive factor and leaves text scores unchanged. The operation preserves the ranking within each modality, and matches their total attribution ratio to the estimated contribution ratio.

## Algorithm 1 VTRACE VTRACE

```julia
# Input: cached states of one forward pass (V, a, x_attn, x_pre),
<sup>#</sup><sub>#</sub> attention output projections Wo, receiver positions recv_pos,
image positions img_pos, user-text positions txt_pos, gamma
# Output: VTrace attribution u_VTrace
# 1. Compute the modality-aware pairwise attribution matrix
F[l,i,h] = V[l,i,h] @ Wo[l,h] # per-head output of i
d[l,j] = x_attn[l,j] - x_pre[l,j]
# center within modality for input tokens (image or prompt-text tokens)
F[l,i,h] -= mean_k(F[l,k,h], k in modality(i)), i in img_pos or txt_pos
e[l,i,j] = sum_h(a[l,h,j,i] <sub>*</sub> dot(F[l,i,h], d[l,j]))
W[i,j] = mean_l(relu(e[l,i,j]) / (norm(d[l,j]) + eps)), i < j
# 2. Katz-based multi-hop attribution
W_hat = W / (max(W) + eps)
R = inv(I - gamma <sub>*</sub> W_hat) - I
inflow[r] = sum_i(R[i,r])
outflow[r] = sum(R[r,j] for j in recv_pos)
u[r] = (1 + inflow[r]) <sub>*</sub> outflow[r]
u[min(recv_pos):] = 0 # sources must precede the receivers
# 3. Align image and text attribution scores
# damage(pos) = logp(gen | clean) - logp(gen | pad at pos)
D_img, D_txt = damage(img_pos), damage(txt_pos)
D_both = damage(img_pos + txt_pos)
phi_img = (D_img + D_both - D_txt) / 2
phi_txt = (D_txt + D_both - D_img) / 2
s = phi_img / (phi_img + phi_txt) # image share
lam = (s / (1 - s)) (sum(u[txt_pos]) / sum(u[img_pos]))
# calibrate the image contribution of the attribution
u_VTrace = u
u_VTrace[img_pos] = lam <sub>*</sub> u[img_pos]
return u_VTrace
```

## E BASELINE & BENCHMARK DESCRIPTIONS

## E.1 BASELINE METHODS

We compare VTRACE with seven attribution methods:

• ReAGent (Zhao & Shan, 2024) is a perturbation method, which repeatedly replaces input tokens with plausible alternatives drawn from a masked language model and scores each token by the change this causes in the probability of the generated text.

• HETA (Pramanik et al., 2026) is a gradient method that adds second order terms from the Hessian to first order gradient attribution, so that interactions between input tokens contribute to their scores. We follow the released implementation.

• Attention Rollout (Abnar & Zuidema, 2020) composes head-averaged attention across layers. At each layer, attention is combined with the identity matrix as $0 . 5 A ^ { ( \ell ) } + 0 . 5 I$ and row-normalized. The resulting layer products provide cumulative input-to-answer attribution.

• AttnLRP (Achtibat et al., 2024) applies layer-wise relevance propagation through attention blocks with relevance-conserving rules for softmax and attention matrix multiplication. It requires one backward pass per attributed target token, so its cost increases with response length.

• IFR (Ferrando & Voita, 2024) builds on ALTI (Ferrando et al., 2022b) to decompose the Transformer into token-level information flows. It measures how each token contributes to subsequent representations through attention and MLP blocks, tracing these contributions layer by layer.

• FlowTracer (Dong et al., 2026) constructs a flow network from head-averaged attention over a selected layer range and scores each input token by the flow it carries to the answer tokens.

• FlashTrace (Pan et al., 2026) uses span-wise aggregation and recursive attribution to trace importance through intermediate reasoning. At each hop, important reasoning tokens become weighted targets for the next attribution step, propagating influence backward toward the original input. Attribution across hops is then aggregated into the final input scores.

## E.2 BENCHMARK STATISTICS

We evaluate our method on diverse multimodal benchmarks. These benchmarks cover a broad range of multimodal capabilities, ranging from general visual perception to knowledge understanding and mathematical problem solving. A brief introduction for each benchmark is provided below.

• MMStar (Chen et al., 2024a) is a vision-indispensable benchmark designed to evaluate multimodal capabilities of LVLMs while reducing the text-only shortcuts effect. It contains 1500 samples covering six core capabilities: coarse perception, fine-grained perception, instance reasoning, logical reasoning, science and technology, and mathematics.

• MathVista (Lu et al., 2024) is a benchmark for mathematical reasoning in visual contexts. Its questions cover figure question answering, geometry problem solving, math word problems, textbook question answering and visual question answering, and span seven reasoning types: algebraic, arithmetic, geometric, logical, numeric commonsense, scientific and statistical reasoning. We use the testmini split of 1,000 samples.

• MathVerse (Zhang et al., 2024a) evaluates visual mathematical reasoning using problems from plane geometry, solid geometry, and functions. Each problem is provided in six variants with different amounts of textual and visual information. We use the testmini split.

• MMMU (Yue et al., 2024) consists of college level multimodal questions collected from exams, quizzes and textbooks. It spans six disciplines: art and design, business, science, health and medicine, humanities and social science, and technology and engineering, across 30 subjects and 30 image types such as charts, diagrams, tables, chemical structures and medical images. We use the validation split of 900 samples.

• MMMU-Pro (Yue et al., 2025) extends MMMU to provide a more challenging evaluation of multimodal reasoning. It removes questions that text-only models answer correctly, expands the candidate options, and adds a vision-only setting in which the question is embedded in the image, so that answering requires reading the image.

• VisualPuzzles (Song et al., 2025) is a benchmark that decouples multimodal reasoning from domain knowledge. It contains 1,168 puzzles adapted from logical reasoning questions of the Chinese Civil Service Examination, covering five reasoning categories: algorithmic, analogical, deductive, inductive and spatial reasoning.

We additionally test three visual reasoning benchmarks for our post-training experiments.

• HallusionBench (Guan et al., 2024) evaluates hallucination and visual reasoning in LVLMs using 346 images and 1,129 human-designed questions. It particularly emphasis on failures caused by visual illusion and language hallucination.

• RealWorldQA (xAI, 2024) evaluates multimodal understanding in real-world scenes. The benchmark includes anonymized vehicle-view images and other natural scenes, with questions focusing particularly on spatial relationships, object properties, and everyday visual understanding.

• MathVision (Wang et al., 2024) evaluates visual mathematical reasoning using problems from real mathematics competitions. It spans 16 mathematical disciplines and five difficulty levels, requiring models to jointly interpret visual content and perform advanced mathematical reasoning.

## F EXPERIMENTAL DETAILS

Models. We evaluate on the publicly released Qwen3-VL (Bai et al., 2025) checkpoints at 8B and 4B parameters and on InternVL3.5-8B (Wang et al., 2025).

Reasoning Trace Generation. The benchmarks provide the image, the question and the reference answer, but not the reasoning trace that we attribute. We therefore run each model once per question with greedy decoding and a budget of 2,048 max tokens, and record the full context: the image tokens, the question, the generated reasoning and the final answer. This trace is then frozen. Every attribution method is evaluated on the same trace, and every perturbation in the evaluation measures the likelihood of the same trace, so this avoids any discrepancy caused by the generated reasoning traces. For VTRACE, the decay rate γ = 1 by default. We highlight that we do not filter traces by answer correctness, and Table 3 reports the metrics separately for correct and incorrect predictions.

Hardware. All token attribution-based experiments run on a single NVIDIA RTX PRO 6000 Blackwell GPU with 96GB of memory. Post-training based experiments (e.g., GRPO) are conducted on AMD Instinct MI355X GPUs.

## F.1 ATTRIBUTION-GUIDED LEARNING DETAILS

We conduct the attribution-guided experiments Section 5 using the verl framework Sheng et al. (2024). Specifically, we fine-tune Qwen3-VL-4B-Instruct with Group Relative Policy Optimization (GRPO). For each training sample, we generate n = 8 rollout responses with a temperature of 0.7 and top-p = 0.95.

The training data are constructed by combining and filtering publicly available multimodal reasoning data from MMR1-RL Leng et al. (2025) and the reinforcement-learning split of ReVisual-R1 Chen et al. (2025). We follow the targeted-RL weighting strategy of Dong et al. (2026), where we use VTRACE attribution scores to identify the most important response tokens. The top 40% of response tokens are assigned a weight of 1.5, while all remaining tokens retain the default weight of 1.0. All models are trained for 2 epochs under the same training configuration.

## G EVALUATION DETAILS

Because there are no token-level ground-truth labels indicating which image regions or question tokens contribute to a generated reasoning trace, we evaluate attribution using perturbation-based metrics. Following prior attribution work, we use RISE (Petsiuk et al., 2018) and MAS (Chase Walker et al., 2024), considering both their deletion and insertion variants. These metrics progressively remove or restore input tokens according to their attribution scores and measure how the model’s confidence in the generated trace changes.

## G.1 PERTURBATION PROTOCOL

Deletable Tokens. We perturb only tokens from the multimodal input. Let I and T denote the imagetoken and question-token positions, respectively. We consider two evaluation settings: image, where only tokens in I are perturbed, and joint, where tokens in both I and T are perturbed. Chat-template tokens are excluded, and generated tokens are kept fixed because they constitute the reasoning trace being explained.

Perturbation schedule. For image tokens, perturbation is performed in pixel space. We first construct a Gaussian-blurred version of the image and replace the image regions corresponding to selected visual tokens with their blurred counterparts. The modified image is re-encoded by the LVLM at each perturbation step. For question tokens, selected tokens are replaced with the padding token.

Moreover, perturbing one token at a time would require a separate forward pass for every token. Following FlashTrace (Pan et al., 2026), we therefore evaluate attribution in proportional steps. Given attribution scores u, the $P$ deletable tokens are ranked by decreasing attribution magnitude $| u _ { i } |$ and partitioned into $K = \operatorname* { m i n } ( 2 0 , P )$ approximately equal-sized groups. Each evaluation thus requires $\bar { K } + 1$ forward passes.

• For deletion, evaluation starts from the original input and progressively perturbs groups from highest to lowest attribution. A faithful attribution should therefore cause the model response to decrease rapidly.

• For insertion, evaluation starts from the fully perturbed input and progressively restores the same groups in the same order. A faithful attribution should recover the model response rapidly.

## G.2 METRIC DEFINITIONS

Let

$$
f ( X ) = \exp \left( { \frac { 1 } { n _ { \mathrm { g e n } } } } \sum _ { t = 1 } ^ { n _ { \mathrm { g e n } } } \log p _ { \theta } ( y _ { t } \mid X , y _ { < t } ) \right)
$$

denote the length-normalised likelihood of the original generated trace under context X, as defined in Section 5.

Let π denote the ordering of the P deletable tokens by decreasing attribution magnitude $| u _ { i } |$ , with attribution scores u. We define $X _ { \mathrm { d e l } } ^ { ( k ) }$ as the input after perturbing the first k groups under this ordering, and $X _ { \mathrm { i n s } } ^ { ( k ) }$ as the fully perturbed input after restoring the first k groups, for $k = 0 , \ldots , K$ . We have:

$$
f _ { k } ^ { \mathrm { d e l } } = f \left( \boldsymbol X _ { \mathrm { d e l } } ^ { ( k ) } \right) , \qquad f _ { k } ^ { \mathrm { i n s } } = f \left( \boldsymbol X _ { \mathrm { i n s } } ^ { ( k ) } \right) .
$$

We normalize each response curve between its fully perturbed and clean endpoints and enforce its expected monotonic direction:

$$
r _ { k } ^ { \mathrm { d e l } } = \operatorname* { m i n } _ { j \le k } \frac { f _ { j } ^ { \mathrm { d e l } } - f _ { K } ^ { \mathrm { d e l } } } { f _ { 0 } ^ { \mathrm { d e l } } - f _ { K } ^ { \mathrm { d e l } } } , \qquad r _ { k } ^ { \mathrm { i n s } } = \operatorname* { m a x } _ { j \le k } \frac { f _ { j } ^ { \mathrm { i n s } } - f _ { 0 } ^ { \mathrm { i n s } } } { f _ { K } ^ { \mathrm { i n s } } - f _ { 0 } ^ { \mathrm { i n s } } } .
$$

This allows the normalized responses to be bounded to [0, 1]. We can define our metrics as:

## G.2.1 RISE

RISE (Petsiuk et al., 2018) measures the area under the normalized perturbation curve:

$$
\begin{array} { r } { \mathrm { R I S E } _ { \mathrm { d e l } } = \mathrm { A U C } \left( r ^ { \mathrm { d e l } } \right) \downarrow , \qquad \mathrm { R I S E } _ { \mathrm { i n s } } = \mathrm { A U C } \left( r ^ { \mathrm { i n s } } \right) \uparrow . } \end{array}
$$

A faithful attribution should remove important evidence early under deletion, causing the response to decrease rapidly, and restore it early under insertion, causing the response to recover rapidly. Thus, lower deletion RISE and higher insertion RISE indicate better faithfulness. Notably, RISE depends only on the attribution ranking and does not account for attribution magnitude.

## G.2.2 MAS

Magnitude Aligned Scoring (MAS) (Chase Walker et al., 2024) additionally evaluates whether attribution magnitude agrees with the observed model response. Let $\mathcal { G } _ { k }$ denote the token group modified at step k. The fraction of attribution mass restored after insertion step k is given by:

$$
m _ { k } ^ { \mathrm { i n s } } = \frac { \sum _ { j = 1 } ^ { k } \sum _ { i \in \mathcal { G } _ { j } } \left| \boldsymbol { u } _ { i } \right| } { \sum _ { i } \left| \boldsymbol { u } _ { i } \right| } ,
$$

while the mass remaining after deletion is $m _ { k } ^ { \mathrm { d e l } } = 1 - m _ { k } ^ { \mathrm { i n s } }$ . The mismatch between attribution mass and model response is thus given by:

$$
d _ { k } ^ { \mathrm { d e l } } = \left| r _ { k } ^ { \mathrm { d e l } } - m _ { k } ^ { \mathrm { d e l } } \right| , \qquad d _ { k } ^ { \mathrm { i n s } } = \left| r _ { k } ^ { \mathrm { i n s } } - m _ { k } ^ { \mathrm { i n s } } \right| .
$$

MAS incorporates this mismatch into the corresponding perturbation curve:

$$
\mathrm { M A S } _ { \mathrm { d e l } } = \mathrm { A U C } \left( r ^ { \mathrm { d e l } } + d ^ { \mathrm { d e l } } \right) \downarrow , \qquad \mathrm { M A S } _ { \mathrm { i n s } } = \mathrm { A U C } \left( r ^ { \mathrm { i n s } } - d ^ { \mathrm { i n s } } \right) \uparrow .
$$

MAS therefore considers both attribution ranking and magnitude, where a faithful attribution should assign attribution mass in proportion to the observed change in model response. As with RISE, lower deletion and higher insertion scores indicate better faithfulness. In implementation, the normalized response and MAS curves are clipped to [0, 1], and AUC is computed using the trapezoidal rule.

## G.3 SYSTEM PROMPT

A fixed system prompt is used to generate reasoning traces. Specifically, the system prompt instructs the model to reason step by step over the visual evidence before providing the final answer.

Prompt Template for Reasoning Trace Generation   
System:   
Look at the image carefully and reason step by step, then end with a line ’Final answer:   
<answer>’.   
User:   
{image} {question}

## H EXTENDED EXPERIMENTS

## H.1 MAIN RESULTS

Table 5 and Table 6 report the complete Image-variant and Joint-variant faithfulness results with both RISE and MAS under insertion and deletion on Qwen3-VL-8B. VTRACE consistently outperforms the baselines for both the image and joint variants. VTRACE improves MAS insertion scores by 9.9% and 18.4% for image and join variant respectively, and reduces the deletion scores by 7.7% and 29.6%. The results turn out VTRACE manages to identify more faithful tokens that are important for constructing the reasoning.

Table 5: Attribution faithfulness of the Image variant on Qwen3-VL-8B across six benchmarks. We report RISE and MAS insertion $( \mathrm { i n s }$ ↑, higher is better) and deletion $\mathrm { ( _ { d e l } }$ ↓, lower is better) AUC and misalignment scores. The Image variant perturbs image patch tokens only. Best results are highlighted in red bold, and second-best results are highlighted in blue underlining.  
Table 6: Attribution faithfulness of the Joint variant on Qwen3-VL-8B across six benchmarks. The Joint variant perturbs both image and text tokens.
<table><tr><td>Dataset</td><td>Metric</td><td>ReAGent</td><td>HETA</td><td>FlowTracer</td><td>IFR</td><td>Attn Rollout</td><td>AttnLRP</td><td>FlashTrace</td><td>VTRACE</td></tr><tr><td rowspan="4">MMStar</td><td> $\mathrm { R I S E } _ { \mathrm { i n s } } \uparrow$ </td><td>0.505</td><td>0.497</td><td>0.532</td><td>0.539</td><td>0.539</td><td>0.563</td><td>0.555</td><td>0.600</td></tr><tr><td> ${ \bf M A S } _ { \mathrm { i n s } }$  ←</td><td>0.336</td><td>0.329</td><td>0.368</td><td>0.384</td><td>0.399</td><td>0.392</td><td>0.410</td><td>0.460</td></tr><tr><td> $\mathrm { R I S E _ { \mathrm { d e l } } }$ </td><td>0.458</td><td>0.463</td><td>0.412</td><td>0.409</td><td>0.439</td><td>0.384</td><td>0.394</td><td>0.352</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } }$  Y</td><td>0.620</td><td>0.625</td><td>0.563</td><td>0.551</td><td>0.566</td><td>0.547</td><td>0.528</td><td>0.489</td></tr><tr><td rowspan="4">MathVista</td><td> $\mathrm { R I S E _ { i n s } } ~ \mathrm { \cdot }$ </td><td>0.500</td><td>0.514</td><td>0.579</td><td>0.591</td><td>0.584</td><td>0.599</td><td>0.600</td><td>0.662</td></tr><tr><td> ${ \bf M A S } _ { \mathrm { i n s } }$  个</td><td>0.334</td><td>0.340</td><td>0.421</td><td>0.449</td><td>0.452</td><td>0.427</td><td>0.464</td><td>0.535</td></tr><tr><td> $\mathrm { R I S E _ { \mathrm { d e l } } }$ </td><td>0.447</td><td>0.444</td><td>0.368</td><td>0.363</td><td>0.394</td><td>0.358</td><td>0.354</td><td>0.311</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } }$  Y</td><td>0.615</td><td>0.607</td><td>0.513</td><td>0.499</td><td>0.518</td><td>0.521</td><td>0.482</td><td>0.446</td></tr><tr><td rowspan="4">MMMU</td><td> ${ \mathrm { R I S E } } _ { \mathrm { i n s } }$  ←</td><td>0.543</td><td>0.564</td><td>0.595</td><td>0.610</td><td>0.627</td><td>0.612</td><td>0.618</td><td>0.665</td></tr><tr><td> ${ \bf M A S } _ { \mathrm { i n s } }$ </td><td>0.381</td><td>0.389</td><td>0.431</td><td>0.461</td><td>0.500</td><td>0.444</td><td>0.479</td><td>0.536</td></tr><tr><td> $\mathrm { R I S E _ { \mathrm { d e l } } }$ </td><td>0.482</td><td>0.480</td><td>0.438</td><td>0.425</td><td>0.444</td><td>0.419</td><td>0.417</td><td>0.379</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } }$  Y</td><td>0.639</td><td>0.643</td><td>0.606</td><td>0.576</td><td>0.580</td><td>0.582</td><td>0.561</td><td>0.522</td></tr><tr><td rowspan="4">MMMU-Pro</td><td> ${ \mathrm { R I S E } } _ { \mathrm { i n s } }$  ←</td><td>0.527</td><td>0.542</td><td>0.570</td><td>0.572</td><td>0.604</td><td>0.593</td><td>0.583</td><td>0.645</td></tr><tr><td> ${ \bf M A S } _ { \mathrm { i n s } }$ </td><td>0.363</td><td>0.359</td><td>0.399</td><td>0.422</td><td>0.468</td><td>0.428</td><td>0.432</td><td>0.511</td></tr><tr><td> $\mathrm { R I S E _ { \mathrm { d e l } } }$ </td><td>0.472</td><td>0.472</td><td>0.442</td><td>0.435</td><td>0.444</td><td>0.413</td><td>0.429</td><td>0.378</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } }$ </td><td>0.631</td><td>0.639</td><td>0.609</td><td>0.580</td><td>0.579</td><td>0.569</td><td>0.575</td><td>0.517</td></tr><tr><td rowspan="4">MathVerse</td><td> ${ \mathrm { R I S E } } _ { \mathrm { i n s } }$  ←</td><td>0.523</td><td>0.565</td><td>0.636</td><td>0.640</td><td>0.624</td><td>0.610</td><td>0.644</td><td>0.684</td></tr><tr><td> ${ \bf M A S } _ { \mathrm { i n s } }$  个</td><td>0.359</td><td>0.380</td><td>0.475</td><td>0.509</td><td>0.498</td><td>0.445</td><td>0.513</td><td>0.562</td></tr><tr><td> $\mathrm { R I S E _ { \mathrm { d e l } } }$ </td><td>0.454</td><td>0.430</td><td>0.341</td><td>0.345</td><td>0.384</td><td>0.360</td><td>0.341</td><td>0.296</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } }$  Y</td><td>0.614</td><td>0.605</td><td>0.500</td><td>0.479</td><td>0.512</td><td>0.527</td><td>0.476</td><td>0.435</td></tr><tr><td rowspan="4">VisualPuzzles</td><td> ${ \mathrm { R I S E } } _ { \mathrm { i n s } }$  ←</td><td>0.456</td><td>0.461</td><td>0.468</td><td>0.511</td><td>0.525</td><td>0.531</td><td>0.510</td><td>0.557</td></tr><tr><td> ${ \bf M A S } _ { \mathrm { i n s } }$ </td><td>0.296</td><td>0.262</td><td>0.272</td><td>0.337</td><td>0.354</td><td>0.346</td><td>0.333</td><td>0.374</td></tr><tr><td> $\mathrm { R I S E _ { \mathrm { d e l } } }$ </td><td>0.409</td><td>0.405</td><td>0.379</td><td>0.354</td><td>0.380</td><td>0.324</td><td>0.358</td><td>0.309</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } }$ </td><td>0.566</td><td>0.613</td><td>0.582</td><td>0.528</td><td>0.522</td><td>0.489</td><td>0.537</td><td>0.456</td></tr></table>

<table><tr><td>Dataset</td><td>Metric</td><td>ReAGent</td><td>HETA</td><td>FlowTracer</td><td>IFR</td><td>Attn Rollout</td><td>AttnLRP</td><td>FlashTrace</td><td>VTRACE</td></tr><tr><td rowspan="4">MMStar</td><td> $\mathrm { R I S E } _ { \mathrm { i n s } } \uparrow$ </td><td>0.371</td><td>0.489</td><td>0.506</td><td>0.507</td><td>0.501</td><td>0.529</td><td>0.554</td><td>0.581</td></tr><tr><td> ${ \bf M A S } _ { \mathrm { i n s } }$  ←</td><td>0.165</td><td>0.277</td><td>0.299</td><td>0.305</td><td>0.314</td><td>0.314</td><td>0.349</td><td>0.397</td></tr><tr><td> $\mathrm { R I S E _ { \mathrm { d e l } } }$ </td><td>0.325</td><td>0.245</td><td>0.226</td><td>0.222</td><td>0.273</td><td>0.219</td><td>0.204</td><td>0.180</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } }$  →</td><td>0.494</td><td>0.408</td><td>0.372</td><td>0.364</td><td>0.384</td><td>0.341</td><td>0.332</td><td>0.246</td></tr><tr><td rowspan="4">MathVista</td><td> ${ \mathrm { R I S E } } _ { \mathrm { i n s } }$  ←</td><td>0.382</td><td>0.467</td><td>0.517</td><td>0.516</td><td>0.501</td><td>0.512</td><td>0.574</td><td>0.647</td></tr><tr><td> ${ \bf M A S } _ { \mathrm { i n s } }$  ←</td><td>0.186</td><td>0.268</td><td>0.316</td><td>0.324</td><td>0.327</td><td>0.304</td><td>0.377</td><td>0.497</td></tr><tr><td> $\mathrm { R I S E _ { \mathrm { d e l } } }$ </td><td>0.345</td><td>0.307</td><td>0.274</td><td>0.269</td><td>0.331</td><td>0.274</td><td>0.241</td><td>0.178</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } }$  V</td><td>0.521</td><td>0.494</td><td>0.435</td><td>0.427</td><td>0.464</td><td>0.422</td><td>0.379</td><td>0.246</td></tr><tr><td rowspan="4">MMMU</td><td> ${ \mathrm { R I S E } } _ { \mathrm { i n s } }$  ←</td><td>0.368</td><td>0.589</td><td>0.583</td><td>0.608</td><td>0.604</td><td>0.615</td><td>0.629</td><td>0.660</td></tr><tr><td> ${ \bf M A S } _ { \mathrm { i n s } }$ </td><td>0.170</td><td>0.394</td><td>0.393</td><td>0.429</td><td>0.450</td><td>0.421</td><td>0.436</td><td>0.503</td></tr><tr><td> $\mathrm { R I S E _ { \mathrm { d e l } } }$ </td><td>0.315</td><td>0.216</td><td>0.204</td><td>0.195</td><td>0.241</td><td>0.191</td><td>0.178</td><td>0.159</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } }$ </td><td>0.468</td><td>0.356</td><td>0.343</td><td>0.323</td><td>0.334</td><td>0.294</td><td>0.292</td><td>0.215</td></tr><tr><td rowspan="4">MMMU-Pro</td><td> ${ \mathrm { R I S E } } _ { \mathrm { i n s } }$  ↑</td><td>0.371</td><td>0.558</td><td>0.532</td><td>0.539</td><td>0.554</td><td>0.559</td><td>0.584</td><td>0.619</td></tr><tr><td> ${ \bf M A S } _ { \mathrm { i n s } }$  ↑</td><td>0.166</td><td>0.348</td><td>0.338</td><td>0.351</td><td>0.379</td><td>0.349</td><td>0.384</td><td>0.446</td></tr><tr><td> $\mathrm { R I S E _ { \mathrm { d e l } } }$ </td><td>0.322</td><td>0.241</td><td>0.242</td><td>0.242</td><td>0.269</td><td>0.229</td><td>0.212</td><td>0.181</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } }$ </td><td>0.477</td><td>0.401</td><td>0.403</td><td>0.402</td><td>0.377</td><td>0.355</td><td>0.350</td><td>0.246</td></tr><tr><td rowspan="4">MathVerse</td><td> ${ \mathrm { R I S E } } _ { \mathrm { i n s } }$  ←</td><td>0.383</td><td>0.495</td><td>0.539</td><td>0.540</td><td>0.513</td><td>0.547</td><td>0.568</td><td>0.606</td></tr><tr><td> ${ \bf M A S } _ { \mathrm { i n s } }$  ↑</td><td>0.193</td><td>0.297</td><td>0.337</td><td>0.345</td><td>0.345</td><td>0.331</td><td>0.350</td><td>0.434</td></tr><tr><td> $\mathrm { R I S E _ { \mathrm { d e l } } }$ </td><td>0.330</td><td>0.314</td><td>0.279</td><td>0.283</td><td>0.342</td><td>0.274</td><td>0.241</td><td>0.187</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } }$  V</td><td>0.489</td><td>0.508</td><td>0.438</td><td>0.447</td><td>0.485</td><td>0.416</td><td>0.375</td><td>0.254</td></tr><tr><td rowspan="4">VisualPuzzles</td><td> ${ \mathrm { R I S E } } _ { \mathrm { i n s } }$  ←</td><td>0.376</td><td>0.488</td><td>0.477</td><td>0.504</td><td>0.523</td><td>0.529</td><td>0.530</td><td>0.571</td></tr><tr><td> ${ \bf M A S } _ { \mathrm { i n s } }$  ↑</td><td>0.187</td><td>0.264</td><td>0.260</td><td>0.303</td><td>0.323</td><td>0.321</td><td>0.319</td><td>0.366</td></tr><tr><td> $\mathrm { R I S E _ { \mathrm { d e l } } }$ </td><td>0.348</td><td>0.288</td><td>0.291</td><td>0.283</td><td>0.283</td><td>0.255</td><td>0.268</td><td>0.207</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } } .$ </td><td>0.494</td><td>0.487</td><td>0.479</td><td>0.459</td><td>0.430</td><td>0.400</td><td>0.434</td><td>0.292</td></tr></table>

![](images/d8dbe78b0a8fa389733e8bdfd4258efeba2146fc4792008e02c8dd9442b490ff.jpg)

![](images/d891315bbdd3c9ff320db272ce76777c3bbc7e10fea2ae584881322e59363f1b.jpg)

![](images/13478c01b4efa59fb4501bc3bc679513b779ea80ca0ec719127391ce09161015.jpg)  
Figure 9: Sensitivity analysis for VTRACE to the decay rate γ.  
Figure 10: Sensitivity analysis for IFR to the decay rate γ.

## H.2 HYPERPARAMETER SENSITIVITY STUDY

We conduct sensitivity study for decay rate $\gamma .$ As shown in Figure $9 , \gamma = 1 . 0$ achieves the strongest performance across benchmarks, indicating that propagating attribution through intermediate tokens helps recover source evidence missed by direct attribution. To examine whether the proposed multihop aggregation generalizes beyond VTRACE, we additionally apply it to the baseline method IFR Ferrando & Voita (2024). As shown in Figure 10, IFR exhibits a similar sensitivity pattern, with the strongest overall performance around $\gamma = 0 . 5$ . The results indicate that the proposed multi-hop aggregation can also enhance existing attribution methods.

## H.3 FULL RESULTS ON GENERALIZABILITY ACROSS LVLM FAMILIES AND SIZES

We report the complete attribution results on InternVL3.5-2B, Qwen3-VL-4B, and InternVL3.5-8B in Tables 7 to 9. The evaluation includes both RISE and MAS scores under the Image and Joint variant. Across different LVLM families and model scales, VTRACE consistently outperforms the baseline attribution methods and demonstrates strong generalizability.

Table 7: Attribution faithfulness on Qwen3-VL-4B.
<table><tr><td>Dataset</td><td>Metric</td><td>ReAGent</td><td>HETA</td><td>FlowTracer</td><td>IFR</td><td>Attn Rollout</td><td>AttnLRP</td><td>FlashTrace</td><td>VTRACE</td></tr><tr><td colspan="10">Image variant</td></tr><tr><td rowspan="4">MMStar</td><td> $\mathrm { R I S E } _ { \mathrm { i n s } } \mathrm { ~ , ~ }$ </td><td>0.531</td><td>0.531</td><td>0.531</td><td>0.525</td><td>0.555</td><td>0.575</td><td>0.539</td><td>0.614</td></tr><tr><td> ${ \bf M A S } _ { \mathrm { i n s } }$  ←</td><td>0.382</td><td>0.364</td><td>0.368</td><td>0.376</td><td>0.421</td><td>0.419</td><td>0.399</td><td>0.482</td></tr><tr><td>RISEdel↓</td><td>0.480</td><td>0.453</td><td>0.454</td><td>0.462</td><td>0.452</td><td>0.418</td><td>0.447</td><td>0.386</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } }$  →</td><td>0.631</td><td>0.620</td><td>0.614</td><td>0.609</td><td>0.586</td><td>0.577</td><td>0.587</td><td>0.531</td></tr><tr><td rowspan="4">MathVerse</td><td> ${ \mathrm { R I S E } } _ { \mathrm { i n s } }$  ↑</td><td>0.490</td><td>0.600</td><td>0.612</td><td>0.611</td><td>0.618</td><td>0.596</td><td>0.617</td><td>0.673</td></tr><tr><td> ${ \bf M A S } _ { \mathrm { i n s } }$  ←</td><td>0.321</td><td>0.434</td><td>0.453</td><td>0.474</td><td>0.498</td><td>0.438</td><td>0.483</td><td>0.553</td></tr><tr><td> $\mathrm { R I S E _ { \mathrm { d e l } } }$ </td><td>0.411</td><td>0.294</td><td>0.288</td><td>0.285</td><td>0.313</td><td>0.290</td><td>0.277</td><td>0.230</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } }$  →</td><td>0.570</td><td>0.425</td><td>0.415</td><td>0.398</td><td>0.417</td><td>0.422</td><td>0.388</td><td>0.345</td></tr><tr><td rowspan="4">VisualPuzzles</td><td> ${ \mathrm { R I S E } } _ { \mathrm { i n s } }$ </td><td>0.470</td><td>0.463</td><td>0.441</td><td>0.445</td><td>0.514</td><td>0.521</td><td>0.457</td><td>0.551</td></tr><tr><td>↑  ${ \bf M A S } _ { \mathrm { i n s } }$ </td><td>0.319</td><td>0.267</td><td>0.247</td><td>0.256</td><td>0.350</td><td>0.335</td><td>0.270</td><td>0.379</td></tr><tr><td> $\mathrm { R I S E _ { \mathrm { d e l } } }$ </td><td>0.425</td><td>0.381</td><td>0.406</td><td>0.405</td><td>0.379</td><td>0.345</td><td>0.398</td><td>0.321</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } }$  →</td><td>0.556</td><td>0.599</td><td>0.624</td><td>0.616</td><td>0.528</td><td>0.519</td><td>0.602</td><td>0.470</td></tr><tr><td colspan="10">Joint variant</td></tr><tr><td rowspan="4">MMStar</td><td> ${ \mathrm { R I S E } } _ { \mathrm { i n s } }$ </td><td>0.335</td><td>0.510</td><td>0.503</td><td>0.503</td><td>0.491</td><td>0.516</td><td>0.526</td><td>0.548</td></tr><tr><td>←  ${ \bf M A S } _ { \mathrm { i n s } }$  ←</td><td>0.149</td><td>0.310</td><td>0.300</td><td>0.301</td><td>0.326</td><td>0.314</td><td>0.317</td><td>0.369</td></tr><tr><td> $\mathrm { R I S E _ { \mathrm { d e l } } }$ </td><td>0.289</td><td>0.192</td><td>0.181</td><td>0.183</td><td>0.240</td><td>0.169</td><td>0.166</td><td>0.157</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } }$  →</td><td>0.417</td><td>0.311</td><td>0.296</td><td>0.304</td><td>0.329</td><td>0.249</td><td>0.274</td><td>0.214</td></tr><tr><td rowspan="4">MathVerse</td><td> ${ \mathrm { R I S E } } _ { \mathrm { i n s } }$  ←</td><td>0.323</td><td>0.533</td><td>0.550</td><td>0.554</td><td>0.480</td><td>0.534</td><td>0.559</td><td>0.579</td></tr><tr><td> ${ \bf M A S } _ { \mathrm { i n s } }$  ↑</td><td>0.142</td><td>0.345</td><td>0.344</td><td>0.356</td><td>0.326</td><td>0.328</td><td>0.333</td><td>0.398</td></tr><tr><td> $\mathrm { R I S E _ { \mathrm { d e l } } }$ </td><td>0.248</td><td>0.195</td><td>0.171</td><td>0.173</td><td>0.305</td><td>0.174</td><td>0.148</td><td>0.116</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } }$  →</td><td>0.403</td><td>0.310</td><td>0.260</td><td>0.266</td><td>0.414</td><td>0.245</td><td>0.229</td><td>0.198</td></tr><tr><td rowspan="4">VisualPuzzles</td><td> ${ \mathrm { R I S E } } _ { \mathrm { i n s } }$  ←</td><td>0.355</td><td>0.505</td><td>0.464</td><td>0.481</td><td>0.514</td><td>0.520</td><td>0.489</td><td>0.546</td></tr><tr><td> ${ \bf M A S } _ { \mathrm { i n s } }$ </td><td>0.174</td><td>0.281</td><td>0.243</td><td>0.260</td><td>0.328</td><td>0.313</td><td>0.259</td><td>0.344</td></tr><tr><td> $\mathrm { R I S E _ { \mathrm { d e l } } }$ </td><td>0.320</td><td>0.236</td><td>0.262</td><td>0.261</td><td>0.243</td><td>0.229</td><td>0.234</td><td>0.188</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } }$ </td><td>0.446</td><td>0.405</td><td>0.446</td><td>0.443</td><td>0.360</td><td>0.361</td><td>0.395</td><td>0.265</td></tr></table>

Table 8: Attribution faithfulness on InternVL3.5-2B.
<table><tr><td>Dataset</td><td>Metric</td><td>ReAGent</td><td>HETA</td><td>FlowTracer</td><td>IFR</td><td>Attn Rollout</td><td>AttnLRP</td><td>FlashTrace</td><td>VTRACE</td></tr><tr><td colspan="10">Image variant</td></tr><tr><td rowspan="4">MMStar</td><td> $\mathrm { R I S E } _ { \mathrm { i n s } } \uparrow$ </td><td>0.523</td><td>0.639</td><td>0.592</td><td>0.614</td><td>0.659</td><td>0.618</td><td>0.635</td><td>0.694</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { i n s } }$ </td><td>0.348</td><td>0.492</td><td>0.459</td><td>0.486</td><td>0.554</td><td>0.477</td><td>0.520</td><td>0.588</td></tr><tr><td> $\mathrm { R I S E _ { \mathrm { d e l } } }$ </td><td>0.478</td><td>0.381</td><td>0.411</td><td>0.388</td><td>0.387</td><td>0.384</td><td>0.372</td><td>0.327</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } }$ </td><td>0.650</td><td>0.533</td><td>0.547</td><td>0.524</td><td>0.510</td><td>0.532</td><td>0.499</td><td>0.461</td></tr><tr><td rowspan="4">MathVerse</td><td>RISEins ↑</td><td>0.405</td><td>0.653</td><td>0.623</td><td>0.659</td><td>0.680</td><td>0.616</td><td>0.672</td><td>0.709</td></tr><tr><td> ${ \bf M A S } _ { \mathrm { i n s } }$  ↑</td><td>0.245</td><td>0.512</td><td>0.487</td><td>0.540</td><td>0.581</td><td>0.478</td><td>0.561</td><td>0.601</td></tr><tr><td> $\mathrm { R I S E _ { \mathrm { d e l } } }$ </td><td>0.310</td><td>0.181</td><td>0.193</td><td>0.173</td><td>0.198</td><td>0.173</td><td>0.165</td><td>0.139</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } }$  Y</td><td>0.443</td><td>0.265</td><td>0.293</td><td>0.266</td><td>0.306</td><td>0.261</td><td>0.266</td><td>0.239</td></tr><tr><td rowspan="4">VisualPuzzles</td><td> ${ \mathrm { R I S E } } _ { \mathrm { i n s } }$ </td><td>0.464</td><td>0.562</td><td>0.511</td><td>0.512</td><td>0.599</td><td>0.544</td><td>0.534</td><td>0.641</td></tr><tr><td>←  $\mathbf { M A S } _ { \mathrm { i n s } }$ </td><td>0.260</td><td>0.368</td><td>0.328</td><td>0.322</td><td>0.449</td><td>0.356</td><td>0.355</td><td>0.492</td></tr><tr><td> $\mathrm { R I S E _ { \mathrm { d e l } } }$ </td><td>0.401</td><td>0.322</td><td>0.370</td><td>0.360</td><td>0.327</td><td>0.321</td><td>0.343</td><td>0.256</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } }$ </td><td>0.588</td><td>0.488</td><td>0.542</td><td>0.547</td><td>0.450</td><td>0.478</td><td>0.517</td><td>0.372</td></tr><tr><td colspan="10">Joint variant</td></tr><tr><td rowspan="4">MMStar</td><td> $\mathrm { R I S E } _ { \mathrm { i n s } } \uparrow$ </td><td>0.469</td><td>0.714</td><td>0.677</td><td>0.682</td><td>0.686</td><td>0.694</td><td>0.706</td><td>0.707</td></tr><tr><td> ${ \bf M A S } _ { \mathrm { i n s } }$  ↑</td><td>0.254</td><td>0.548</td><td>0.505</td><td>0.516</td><td>0.580</td><td>0.518</td><td>0.541</td><td>0.585</td></tr><tr><td> $\mathrm { R I S E _ { \mathrm { d e l } } }$ </td><td>0.372</td><td>0.191</td><td>0.204</td><td>0.199</td><td>0.241</td><td>0.192</td><td>0.194</td><td>0.189</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } }$ </td><td>0.587</td><td>0.321</td><td>0.339</td><td>0.333</td><td>0.316</td><td>0.308</td><td>0.327</td><td>0.291</td></tr><tr><td rowspan="4">MathVerse</td><td> $\mathrm { R I S E } _ { \mathrm { i n s } } \mathrm { ~ , ~ }$  ←</td><td>0.481</td><td>0.765</td><td>0.741</td><td>0.749</td><td>0.728</td><td>0.763</td><td>0.759</td><td>0.799</td></tr><tr><td> ${ \bf M A S } _ { \mathrm { i n s } }$  ←</td><td>0.252</td><td>0.626</td><td>0.603</td><td>0.618</td><td>0.642</td><td>0.627</td><td>0.610</td><td>0.722</td></tr><tr><td> $\mathrm { R I S E _ { \mathrm { d e l } } }$ </td><td>0.358</td><td>0.200</td><td>0.199</td><td>0.195</td><td>0.250</td><td>0.175</td><td>0.172</td><td>0.153</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } }$ </td><td>0.581</td><td>0.311</td><td>0.305</td><td>0.298</td><td>0.321</td><td>0.263</td><td>0.271</td><td>0.209</td></tr><tr><td rowspan="4">VisualPuzzles</td><td> ${ \mathrm { R I S E } } _ { \mathrm { i n s } }$  ←</td><td>0.433</td><td>0.668</td><td>0.605</td><td>0.603</td><td>0.668</td><td>0.613</td><td>0.631</td><td>0.690</td></tr><tr><td> ${ \bf M A S } _ { \mathrm { i n s } }$ </td><td>0.217</td><td>0.477</td><td>0.409</td><td>0.413</td><td>0.550</td><td>0.417</td><td>0.434</td><td>0.557</td></tr><tr><td> $\mathrm { R I S E _ { \mathrm { d e l } } }$ </td><td>0.350</td><td>0.221</td><td>0.254</td><td>0.249</td><td>0.242</td><td>0.219</td><td>0.237</td><td>0.193</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } }$ </td><td>0.556</td><td>0.351</td><td>0.409</td><td>0.403</td><td>0.310</td><td>0.332</td><td>0.385</td><td>0.256</td></tr></table>

Table 9: Attribution faithfulness on InternVL3.5-8B.
<table><tr><td>Dataset</td><td>Metric</td><td>ReAGent</td><td>HETA</td><td>FlowTracer</td><td>IFR</td><td>Attn Rollout</td><td>AttnLRP</td><td>FlashTrace</td><td>VTRACE</td></tr><tr><td colspan="10">Image variant</td></tr><tr><td rowspan="4">MMStar</td><td> $\mathrm { R I S E } _ { \mathrm { i n s } } \ \mathrm { \hat { 1 } }$ </td><td>0.529</td><td>0.684</td><td>0.637</td><td>0.679</td><td>0.681</td><td>0.646</td><td>0.694</td><td>0.726</td></tr><tr><td> ${ \bf M A S } _ { \mathrm { i n s } }$  ↑</td><td>0.364</td><td>0.570</td><td>0.506</td><td>0.572</td><td>0.580</td><td>0.517</td><td>0.592</td><td>0.633</td></tr><tr><td> $\mathrm { R I S E _ { \mathrm { d e l } } }$ </td><td>0.479</td><td>0.344</td><td>0.365</td><td>0.338</td><td>0.370</td><td>0.367</td><td>0.325</td><td>0.303</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } }$ </td><td>0.643</td><td>0.475</td><td>0.505</td><td>0.462</td><td>0.491</td><td>0.509</td><td>0.447</td><td>0.426</td></tr><tr><td rowspan="4">MathVerse</td><td> ${ \mathrm { R I S E } } _ { \mathrm { i n s } }$  ↑</td><td>0.421</td><td>0.718</td><td>0.695</td><td>0.732</td><td>0.733</td><td>0.662</td><td>0.742</td><td>0.753</td></tr><tr><td> ${ \bf M A S } _ { \mathrm { i n s } }$ </td><td>0.298</td><td>0.620</td><td>0.579</td><td>0.632</td><td>0.624</td><td>0.545</td><td>0.648</td><td>0.661</td></tr><tr><td> $\mathrm { R I S E _ { \mathrm { d e l } } }$ </td><td>0.322</td><td>0.170</td><td>0.174</td><td>0.157</td><td>0.174</td><td>0.189</td><td>0.154</td><td>0.144</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } }$ </td><td>0.456</td><td>0.260</td><td>0.248</td><td>0.284</td><td>0.331</td><td>0.269</td><td>0.270</td><td>0.251</td></tr><tr><td rowspan="4">VisualPuzzles</td><td> ${ \mathrm { R I S E } } _ { \mathrm { i n s } }$  ↑</td><td>0.415</td><td>0.565</td><td>0.494</td><td>0.543</td><td>0.622</td><td>0.540</td><td>0.550</td><td>0.617</td></tr><tr><td> ${ \bf M A S } _ { \mathrm { i n s } }$ </td><td>0.256</td><td>0.414</td><td>0.330</td><td>0.399</td><td>0.508</td><td>0.372</td><td>0.408</td><td>0.487</td></tr><tr><td> $\mathrm { R I S E _ { \mathrm { d e l } } }$ </td><td>0.379</td><td>0.278</td><td>0.345</td><td>0.304</td><td>0.270</td><td>0.278</td><td>0.296</td><td>0.239</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } }$ </td><td>0.529</td><td>0.404</td><td>0.512</td><td>0.455</td><td>0.378</td><td>0.404</td><td>0.438</td><td>0.353</td></tr><tr><td colspan="10">Joint variant</td></tr><tr><td rowspan="4">MMStar</td><td> ${ \mathrm { R I S E } } _ { \mathrm { i n s } }$ </td><td>0.400</td><td>0.727</td><td>0.700</td><td>0.715</td><td>0.537</td><td>0.693</td><td>0.708</td><td>0.760</td></tr><tr><td>←  ${ \bf M A S } _ { \mathrm { i n s } }$  ←</td><td>0.184</td><td>0.583</td><td>0.548</td><td>0.578</td><td>0.361</td><td>0.532</td><td>0.543</td><td>0.657</td></tr><tr><td> $\mathrm { R I S E _ { \mathrm { d e l } } }$ </td><td>0.286</td><td>0.135</td><td>0.130</td><td>0.123</td><td>0.245</td><td>0.132</td><td>0.101</td><td>0.103</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } }$ </td><td>0.457</td><td>0.212</td><td>0.204</td><td>0.193</td><td>0.341</td><td>0.194</td><td>0.157</td><td>0.149</td></tr><tr><td rowspan="4">MathVerse</td><td> ${ \mathrm { R I S E } } _ { \mathrm { i n s } }$  ←</td><td>0.457</td><td>0.786</td><td>0.775</td><td>0.780</td><td>0.673</td><td>0.760</td><td>0.794</td><td>0.854</td></tr><tr><td> ${ \bf M A S } _ { \mathrm { i n s } }$  ↑</td><td>0.272</td><td>0.675</td><td>0.675</td><td>0.685</td><td>0.540</td><td>0.639</td><td>0.681</td><td>0.800</td></tr><tr><td> $\mathrm { R I S E _ { \mathrm { d e l } } }$ </td><td>0.354</td><td>0.190</td><td>0.181</td><td>0.173</td><td>0.331</td><td>0.182</td><td>0.147</td><td>0.116</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } }$ </td><td>0.513</td><td>0.278</td><td>0.266</td><td>0.254</td><td>0.456</td><td>0.266</td><td>0.217</td><td>0.168</td></tr><tr><td rowspan="4">VisualPuzzles</td><td> ${ \mathrm { R I S E } } _ { \mathrm { i n s } }$  个</td><td>0.376</td><td>0.648</td><td>0.578</td><td>0.603</td><td>0.586</td><td>0.588</td><td>0.576</td><td>0.666</td></tr><tr><td> ${ \bf M A S } _ { \mathrm { i n s } }$ </td><td>0.181</td><td>0.470</td><td>0.407</td><td>0.447</td><td>0.420</td><td>0.401</td><td>0.359</td><td>0.531</td></tr><tr><td> $\mathrm { R I S E _ { \mathrm { d e l } } }$ </td><td>0.304</td><td>0.189</td><td>0.223</td><td>0.208</td><td>0.207</td><td>0.184</td><td>0.181</td><td>0.135</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } }$ </td><td>0.452</td><td>0.295</td><td>0.362</td><td>0.337</td><td>0.280</td><td>0.274</td><td>0.299</td><td>0.188</td></tr></table>

Table 10: Attribution faithfulness on correctly and incorrectly answered samples for the Joint variant. RISE and MAS are reported under insertion $( \mathrm { i n s }$ ↑, higher is better) and deletion $\mathrm { ( _ { d e l } \downarrow }$ , lower is better). Best results are highlighted in red bold, and second-best results are highlighted in blue underlining.
<table><tr><td>Dataset</td><td>Split</td><td>Metric</td><td>ReAGent</td><td>HETA</td><td>FlowTracer</td><td>IFR</td><td>Attn Rollout</td><td>AttnLRP</td><td>FlashTrace</td><td>VTRACE</td></tr><tr><td rowspan="7">MMStar</td><td rowspan="3">Correct</td><td> $\mathrm { R I S E _ { \mathrm { i n s } } } \ \mathrm { \uparrow }$ </td><td>0.371</td><td>0.482</td><td>0.504</td><td>0.504</td><td>0.497</td><td>0.521</td><td>0.552</td><td>0.581</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { i n s } } \mathrm { ~ , ~ }$ </td><td>0.166</td><td>0.271</td><td>0.297</td><td>0.303</td><td>0.310</td><td>0.306</td><td>0.347</td><td>0.400</td></tr><tr><td> $\mathrm { R I S E _ { \mathrm { d e l } } }$ </td><td>0.326</td><td>0.251</td><td>0.230</td><td>0.225</td><td>0.278</td><td>0.225</td><td>0.207</td><td>0.181</td></tr><tr><td></td><td> $\mathbf { M A S } _ { \mathrm { d e l } }$  →</td><td>0.494</td><td>0.417</td><td>0.377</td><td>0.369</td><td>0.390</td><td>0.351</td><td>0.336</td><td>0.248</td></tr><tr><td rowspan="4">Incorrect</td><td> ${ \mathrm { R I S E } } _ { \mathrm { i n s } }$  ←</td><td>0.371</td><td>0.511</td><td>0.514</td><td>0.514</td><td>0.513</td><td>0.551</td><td>0.559</td><td>0.580</td></tr><tr><td> ${ \bf M A S } _ { \mathrm { i n s } }$  ←</td><td>0.164</td><td>0.295</td><td>0.305</td><td>0.311</td><td>0.324</td><td>0.337</td><td>0.352</td><td>0.389</td></tr><tr><td> $\mathrm { R I S E _ { d e l } \downarrow }$ </td><td>0.322</td><td>0.227</td><td>0.216</td><td>0.212</td><td>0.257</td><td>0.202</td><td>0.195</td><td>0.177</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } } \downarrow$ </td><td>0.492</td><td>0.381</td><td>0.358</td><td>0.350</td><td>0.365</td><td>0.315</td><td>0.320</td><td>0.241</td></tr><tr><td rowspan="7">MathVista</td><td rowspan="4">Correct</td><td> ${ \mathrm { R I S E } } _ { \mathrm { i n s } }$  ←</td><td>0.376</td><td>0.460</td><td>0.510</td><td>0.507</td><td>0.492</td><td>0.501</td><td>0.567</td><td>0.644</td></tr><tr><td> ${ \bf M A S } _ { \mathrm { i n s } }$ </td><td>0.177</td><td>0.262</td><td>0.309</td><td>0.314</td><td>0.317</td><td>0.292</td><td>0.370</td><td>0.495</td></tr><tr><td> $\mathrm { R I S E _ { \mathrm { d e l } } }$ </td><td>0.341</td><td>0.308</td><td>0.276</td><td>0.272</td><td>0.336</td><td>0.276</td><td>0.242</td><td>0.174</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } }$ </td><td>0.522</td><td>0.496</td><td>0.438</td><td>0.430</td><td>0.470</td><td>0.424</td><td>0.381</td><td>0.241</td></tr><tr><td> ${ \mathrm { R I S E } } _ { \mathrm { i n s } }$ </td><td>← 0.410</td><td>0.501</td><td>0.551</td><td>0.560</td><td>0.542</td><td>0.565</td><td>0.606</td><td>0.657</td></tr><tr><td rowspan="4">Incorrect</td><td> ${ \bf M A S } _ { \mathrm { i n s } }$ </td><td>0.228</td><td>0.299</td><td>0.347</td><td>0.373</td><td>0.376</td><td>0.361</td><td>0.410</td><td>0.502</td></tr><tr><td> $\mathrm { R I S E _ { d e l } }$ </td><td>0.364</td><td>0.299</td><td>0.264</td><td>0.257</td><td>0.310</td><td>0.265</td><td>0.234</td><td>0.194</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } }$  十</td><td>0.519</td><td>0.484</td><td>0.420</td><td>0.409</td><td>0.437</td><td>0.413</td><td>0.370</td><td>0.269</td></tr><tr><td> ${ \mathrm { R I S E } } _ { \mathrm { i n s } }$   ${ \bf M A S } _ { \mathrm { i n s } }$ </td><td>← 0.368</td><td>0.583</td><td>0.576</td><td>0.599</td><td>0.593</td><td>0.606</td><td>0.623</td><td>0.654</td></tr><tr><td rowspan="6">MMMU</td><td rowspan="4">Correct</td><td></td><td>0.169</td><td>0.386</td><td>0.383</td><td>0.415</td><td>0.436</td><td>0.409</td><td>0.427</td><td>0.494</td></tr><tr><td> $\mathrm { R I S E _ { \mathrm { d e l } } }$ </td><td>0.314</td><td>0.217</td><td>0.206</td><td>0.197</td><td>0.246</td><td>0.193</td><td>0.179</td><td>0.159</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } }$  →</td><td>0.466</td><td>0.359</td><td>0.347</td><td>0.327</td><td>0.341</td><td>0.297</td><td>0.296</td><td>0.216</td></tr><tr><td> ${ \mathrm { R I S E } } _ { \mathrm { i n s } }$  ←</td><td>0.370</td><td>0.613</td><td>0.607</td><td>0.637</td><td>0.644</td><td>0.645</td><td>0.650</td><td>0.682</td></tr><tr><td rowspan="4"> ${ \bf M A S } _ { \mathrm { i n s } }$  Incorrect</td><td></td><td>0.176</td><td>0.425</td><td>0.428</td><td>0.473</td><td>0.500</td><td>0.464</td><td>0.468</td><td>0.532</td></tr><tr><td> $\mathrm { R I S E _ { \mathrm { d e l } } }$ </td><td>0.322</td><td>0.211</td><td>0.199</td><td>0.187</td><td>0.223</td><td>0.183</td><td>0.171</td><td>0.156</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } }$ </td><td>0.474</td><td>0.343</td><td>0.330</td><td>0.308</td><td>0.310</td><td>0.282</td><td>0.279</td><td>0.212</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## H.4 FULL RESULTS ON ROBUSTNESS TO PREDICTION CORRECTNESS

We evaluate attribution faithfulness separately on correctly and incorrectly answered samples to examine whether the attribution quality depends on prediction correctness. Tables 10 and 11 show the

Table 11: Attribution faithfulness on correctly and incorrectly answered samples for the Image variant. RISE and MAS are reported under insertion $\mathrm { { ( _ { i n s } } }$ ↑, higher is better) and deletion $( \mathrm { d e l \downarrow } ,$ lower is better). Best results are highlighted in red bold, and second-best results are highlighted in blue underlining.
<table><tr><td>Dataset</td><td>Split</td><td>Metric</td><td>ReAGent</td><td>HETA</td><td>FlowTracer</td><td>IFR</td><td>Attn Rollout</td><td>AttnLRP</td><td>FlashTrace</td><td>VTRACE</td></tr><tr><td rowspan="7">MMStar</td><td rowspan="4">Correct</td><td> ${ \mathrm { R I S E } } _ { \mathrm { i n s } }$  ←</td><td>0.499</td><td>0.493</td><td>0.534</td><td>0.542</td><td>0.541</td><td>0.560</td><td>0.556</td><td>0.603</td></tr><tr><td> ${ \bf M A S } _ { \mathrm { i n s } }$ </td><td>0.328</td><td>0.326</td><td>0.371</td><td>0.390</td><td>0.403</td><td>0.389</td><td>0.414</td><td>0.466</td></tr><tr><td>RISEdel↓</td><td>0.453</td><td>0.459</td><td>0.402</td><td>0.399</td><td>0.429</td><td>0.380</td><td>0.386</td><td>0.344</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } }$  V</td><td>0.616</td><td>0.618</td><td>0.549</td><td>0.536</td><td>0.552</td><td>0.539</td><td>0.514</td><td>0.478</td></tr><tr><td rowspan="4">Incorrect</td><td> ${ \mathrm { R I S E } } _ { \mathrm { i n s } }$  ←</td><td>0.525</td><td>0.509</td><td>0.526</td><td>0.531</td><td>0.531</td><td>0.572</td><td>0.550</td><td>0.591</td></tr><tr><td> ${ \bf M A S } _ { \mathrm { i n s } }$  个</td><td>0.357</td><td>0.336</td><td>0.360</td><td>0.369</td><td>0.388</td><td>0.399</td><td>0.399</td><td>0.442</td></tr><tr><td> $\mathrm { R I S E _ { \mathrm { d e l } } }$ </td><td>0.471</td><td>0.477</td><td>0.440</td><td>0.437</td><td>0.469</td><td>0.397</td><td>0.418</td><td>0.374</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } }$ </td><td>0.632</td><td>0.646</td><td>0.603</td><td>0.595</td><td>0.606</td><td>0.568</td><td>0.566</td><td>0.520</td></tr><tr><td rowspan="7">MathVista</td><td rowspan="4">Correct</td><td> ${ \mathrm { R I S E } } _ { \mathrm { i n s } }$  ←</td><td>0.496</td><td>0.509</td><td>0.576</td><td>0.586</td><td>0.580</td><td>0.597</td><td>0.597</td><td>0.663</td></tr><tr><td> ${ \bf M A S } _ { \mathrm { i n s } }$ </td><td>0.326</td><td>0.335</td><td>0.418</td><td>0.443</td><td>0.446</td><td>0.424</td><td>0.459</td><td>0.537</td></tr><tr><td>RISEdel↓</td><td>0.440</td><td>0.439</td><td>0.358</td><td>0.356</td><td>0.388</td><td>0.348</td><td>0.345</td><td>0.297</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } }$ </td><td>0.611</td><td>0.599</td><td>0.500</td><td>0.489</td><td>0.510</td><td>0.506</td><td>0.470</td><td>0.429</td></tr><tr><td>RISEins ↑</td><td>0.523</td><td>0.539</td><td>0.590</td><td>0.614</td><td>0.603</td><td>0.611</td><td>0.617</td><td>0.655</td></tr><tr><td rowspan="4">Incorrect</td><td>MASins ↑</td><td>0.373</td><td>0.364</td><td>0.434</td><td>0.480</td><td>0.478</td><td>0.442</td><td>0.485</td><td>0.523</td></tr><tr><td>RISEdel↓</td><td>0.483</td><td>0.469</td><td>0.416</td><td>0.396</td><td>0.422</td><td>0.405</td><td>0.394</td><td>0.373</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } } .$ </td><td>0.634</td><td>0.644</td><td>0.575</td><td>0.546</td><td>0.558</td><td>0.588</td><td>0.538</td><td>0.525</td></tr><tr><td rowspan="4"> ${ \mathrm { R I S E } } _ { \mathrm { i n s } }$  Correct</td><td></td><td>0.533</td><td>0.562</td><td>0.591</td><td>0.606</td><td>0.624</td><td>0.607</td><td>0.616</td><td>0.666</td></tr><tr><td> ${ \bf M A S } _ { \mathrm { i n s } }$  1</td><td>0.372</td><td>0.389</td><td>0.430</td><td>0.458</td><td>0.496</td><td>0.440</td><td>0.479</td><td>0.540</td></tr><tr><td> $\mathrm { R I S E _ { \mathrm { d e l } } }$ </td><td>0.470</td><td>0.463</td><td>0.421</td><td>0.409</td><td>0.428</td><td>0.403</td><td>0.400</td><td>0.359</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } }$ </td><td>0.626</td><td>0.623</td><td>0.585</td><td>0.555</td><td>0.560</td><td>0.566</td><td>0.539</td><td>0.497</td></tr><tr><td rowspan="4">MMMU</td><td>Incorrect</td><td>个 0.575</td><td>0.567</td><td>0.608</td><td>0.625</td><td>0.638</td><td>0.627</td><td>0.627</td><td>0.661</td></tr><tr><td rowspan="4"></td><td> ${ \mathrm { R I S E } } _ { \mathrm { i n s } }$ </td><td>0.412</td><td>0.390</td><td>0.435</td><td>0.472</td><td>0.513</td><td>0.456</td><td>0.481</td><td>0.523</td></tr><tr><td> ${ \bf M A S } _ { \mathrm { i n s } }$   $\mathrm { R I S E _ { \mathrm { d e l } } }$ </td><td>0.522</td><td>0.541</td><td>0.493</td><td>0.477</td><td>0.497</td><td>0.470</td><td>0.472</td><td>0.448</td></tr><tr><td> $\mathbf { M A S } _ { \mathrm { d e l } }$ </td><td>0.683</td><td>0.717</td><td>0.676</td><td>0.647</td><td>0.647</td><td>0.637</td><td>0.634</td><td>0.606</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

complete results for the Image and Joint Variants, where VTRACE achieves the best performance on both correct and incorrect predictions. The results indicate that the attribution gains arise from attribution method rather than answer correctness.

![](images/9a36ee8f80b59635518d83faf54819100e7bf36a858adf69318e64a82fa70fe3.jpg)  
Figure 11: Fine-grained insertion and deletion perturbation curves across benchmarks.

## H.5 FINE-GRAINED PERTURBATION CURVES

VTRACE produces more faithful rankings throughout the perturbation process, beyond what is captured by the AUC scores. Shown in Figure Figure 11, VTRACE achieves faster recovery under insertion and sharper degradation under deletion. The advantage is especially clear at early perturbation stages, where restoring only a small fraction of top-ranked tokens rapidly recovers the response, while removing them causes substantial degradation.

## H.6 SUMMING PATHS CARRIES CREDIT BACK TO THE INPUTS.

Figure 12a shows how far back a generated token’s credit comes from. Evidently, a single hop keeps credit near the receiver: under $\breve { W } .$ , 22% of a token’s credit falls on the token directly before it and

58% on the 20 tokens before it (IFR: 17% and 53%). Once VTRACE sums all direct and indirect paths (R), the token directly before keeps 5%, and 75% of the credit comes from sources more than<sup>image</sup> <sup>1.1%</sup> <sup>question</sup> <sup>20%</sup> <sup>output</sup> <sup>79%</sup> <sub>image</sub> <sub>2.8%</sub><sup>image</sup> <sup>1.1%</sup> <sup>question</sup> <sup>20%</sup> <sup>output</sup> <sup>79%</sup> <sub>image</sub> <sub>2.8%</sub> 20 positions back. For a generated token, those sources are mostly the question, the image and the<sup>3</sup> <sup>hops</sup> <sup>W3</sup> 2.8% 2.1% <sub>2.3%</sub><sup>3</sup> <sup>hops</sup> <sup>W3</sup> 2.8% 2.1% <sub>2.3%</sub> early reasoning. Figure 12b tracks how much of the answer’s credit lands on the image as longer paths are added. The direct edge gives the image 0.7% and IFR gives it 1.0%. The share rises with                                                    <sup>(a)</sup> every hop, to 1.6% with paths of up to two edges and 6.2% with paths of up to eight, and R reaches<sup>image</sup> <sup>2.8%</sup> <sup>question</sup> <sup>21%</sup> <sup>output</sup> <sup>76%image</sup> <sup>2.8%</sup> <sup>question</sup> <sup>21%</sup> <sup>output</sup> <sup>76%</sup> 8.5%. The direct edge therefore sees almost none of the image, and the image share is still risingall hops R = W + W<sup>2</sup> + W<sup>3</sup> + ⋯ (VTrace) 2.2% <sub>4.3%</sub>a l hops R = W + W<sup>2</sup> + W<sup>3</sup> + ⋯ (VTrace) 2.2% after eight hops. Figure 12c tests whether this extra credit is faithful. We rank the image patches andrc<sup>e</sup> + + + ⋯ = | the question tokens separately by each operator’s score, then remove them in that order (deletion) or                          <sup>para lelogram</sup> <sup>…</sup> <sup>(A</sup> <sup>…</sup> <sup>(D</sup> <sup>…</sup> <sup>78</sup> <sup>…</sup> <sup>78</sup> <sup>…</sup> <sup>choices</sup> <sup>…</sup> <sup>(A</sup> <sup>…</sup> <sup>(C</sup> <sup>…</sup> <sup>(D</sup> <sup>…</sup> <sup>78</sup> <sup>…</sup> <sup>answer</sup> <sup>…</sup> <sup>7</sup>    <sup>s</sup> add them back to a fully masked input (insertion) and measure the RISE. Notably, longer paths helpimage 6.1% question 15% output 79%image 6.1% question 15% output 79% in all four settings, and VTRACE, which aggregates all paths performs the best, as expected.IFR (1 hop, atention) <sub>3.2%</sub>IFR (1 hop, atention) <sub>3.2%</sub>

![](images/c234f1ae200deb405759bcc4598d47ced15427a85a04b73837285cd6e74bbc67.jpg)  
(a) Source distance

![](images/0b16c209c5323d067b8dcefdb39ca9874d2a93fd368b143de21d81a294a94042.jpg)  
(b) Image credit vs. hops

![](images/697950208ff4cea75784b45f51f7031d1fe2d8fc49a71dda2764d7e746e14979.jpg)  
(c) RISE by operator  
Figure 12: (a) How far back a generated token’s credit comes from. (b) The answer’s image credit as hops are summed. (c) Performance comparison between diverse hops, VTRACE (R) and IFR.

## I CASE STUDIES

Image token attribution. Figure 13 compares the top 20% image patches selected by each method against a perturbation-based reference. For each patch, we blur it and measure the resulting drop in the model’s trace likelihood, where a larger drop indicates greater reliance on that region. This reference map is shown in the second column. The reported score on the bottom right measures how much of the reference importance is captured by the selected patches, with higher values indicating better alignment. We demonstrate two better performing and two slightly behind cases of VTRACE. In the first two examples, VTRACE focuses more strongly on the relevant people and scene regions, scoring 0.50 and 0.42 compared with 0.46 and 0.35 for the strongest baseline. The third example is tied with FlowTracer. The fourth shows a slightly less aligned case, while both VTRACE and AttnLRP both captures the important balls, VTRACE also assigns attribution to the player and table edge. This suggests that VTRACE can miss important visual regions when they are not clearly reflected in the generated reasoning trace.

Joint image and text attribution. Figures 14 to 17 show four case studies, comparing IFR, a second baseline and VTRACE on the same fixed model response, with the boxed final answer (without loss of generality) selected as the target. In each row, the image on the left and the question text outlines the method’s top 20% of the image tokens, and dims the rest. The text on the right shows the question and the full response, with each token shaded by its attribution score: the darker the token, the more it contributed to the answer.

A faithful attribution should point to the evidence the answer relies on: the image regions the question is about, the key words of the question, and the reasoning steps that carry what the model saw to its answer. Evidently, VTRACE does this more consistently than the baselines. Its top patches concentrate on the relevant objects and labels, while the baselines often spread their patches over the background and the image border. In the text, VTRACE highlights the key question words and the reasoning steps that use visual evidence, whereas the baselines focus mostly on the answer options and the final sentence. Figure 17 shows a harder case, where VTRACE’s image selection is less focused. See the captions for detailed description.

Input

Measured Dependence

IFR

Best Baseline VTrace (Ours)

Q: What is the fraction of females facing the camera? Answer: C (0.8)

![](images/e1e3fde8e834639002d00580738b24324ff3ef53ff71ddff79a074bfccbe045d.jpg)

![](images/3f115fcec251f1d6e087cae69854ed4ff88de121f8edc322eacb4852c694b415.jpg)  
Q: What is the main color theme of the scene?

![](images/803f59ae0f29a8ac5b5667eafaae8362177c7541b82d86f735109f1e3c02d69a.jpg)

FlashTrace  
![](images/b7faaf2201a76b47c104f5aaf5448100f8e314566527826408174c897f355976.jpg)

![](images/5639e9cf500a9f02b031814f1ff3bfe4cbe2a59837e0881bcd802d9555d10fd6.jpg)  
Answer: D (Yellow)

![](images/344e3974910b0e81ff5b72fe1fb0a4c6296b95d663a8bc6c1cfd4788d9d98f44.jpg)

![](images/854e11614c602d4206fe3c2e8bb31441225dce6dbf787c3c28ed9e87b529297b.jpg)  
Q: What is the sport being played in the image?

![](images/277b21f28271eb5254125e78218cc4840a1dc05b55eedb7be36ec295c0672c40.jpg)  
Attn Rollout

![](images/ce6760e12155770ccc5a1ab9e7ad1c1c344db3fe0b4d3e612febcd316add0f50.jpg)

![](images/de409b269b495fffd22671893a4e82bbb885f81b270ead2ca029e59ba54cb0d3.jpg)  
Answer: B (Soccer)

![](images/4e90bc71c465e9f2226f79398aa69c1ebdb1d6d95f23bae2301c65d5856608a8.jpg)

![](images/1415de1446f22659721428c23cb70464d5177a43fa6df2b3dbc7f1f284b16ecc.jpg)

![](images/2139493c49a610d718ac24c2dc2888206e6123c65d8860626a69701917d5a9fc.jpg)

FlowTracer  
![](images/dc4c48a13292962103ceee58b39fcf6863573909f6b1228839c091510cf1dc46.jpg)

![](images/a6e0c96f004c1a3221da00a5c821bd6700c507827a7a4e83b278e8cabb990bd5.jpg)  
Q: What is the color of the ball closest to the man? Answer: A (White)

![](images/aa8232137a407ba338b91653a5f7696a14216576f78a1c489d1d62511f73a246.jpg)

![](images/d46de0b70fcb73609f9563a97be4631495be459b47fbf373a3e594e30b05431e.jpg)  
AttnLRP

![](images/8d08184d58adb4a56ba620445ea1a1edf9546c9b3b4ff6cd4a2c85b31b0b0eaa.jpg)

![](images/79fd2575696ff4ce0f2182eefd393fe44e6a956c180de7e9b36f20165d2ad67c.jpg)

![](images/d31998cb40f446fc389ff4bff165d71d62fe0d583c79946d387e8380921ae5d9.jpg)  
Figure 13: Image token attribution on four MMStar questions. Each method’s top 20% of patches is outlined, with the share of the measured dependence it captures in the corner (higher is better). The measured dependence is the likelihood drop from blurring each patch in turn. VTRACE leads on the first two questions, ties FlowTracer on the third, and trails AttnLRP on the fourth.

![](images/1c123847137c9a362b2eee239b80a2f8ed0a4cb820e925c4e94a72288222700c.jpg)

Q: What is the sport being played in the image? Options: A: Tennis, B: Soccer, C: Volleyball, D: Basketball A: Step1: Identify key elements in the image. - There is a large net with a goal frame, which is characteristic of a soccer goal. - A soccer ball is visible, positioned in front of the goal. - Two men are standing on a green field that resembles artificial turf, commonly used in soccer. - The players are wearing athletic attire, including shorts and cleats, typical for soccer. - The background has silhouettes of people and a large red figure, possibly a mascot or part of a training setup, but this does not change the sport. Step2: Eliminate incorrect options. - Tennis: Played on a hard or grass court with rackets and a ball, no net with a goal frame. - Volleyball: Played on a court with a net, but the net is lower and there is no goal frame or soccer ball. - Basketball: Played on a court with a hoop and backboard, not a net with a goal frame. Step3: Confirm the sport. - The presence of a soccer goal, a soccer ball, and players on a field strongly indicates the sport is soccer (football). Final answer: B: Soccer

FlowTracer  
![](images/96eca133e54f8274f642fc62065b9d219caa59b8bbe462e75f4671435911df57.jpg)

Q: What is the sport being played in the image? Options: A: Tennis, B: Soccer, C: Volleyball, D: Basketball A: Step1: Identify key elements in the image. - There is a large net with a goal frame, which is characteristic of a soccer goal. - A soccer ball is visible, positioned in front of the goal. - Two men are standing on a green field that resembles artificial turf, commonly used in soccer. - The players are wearing athletic attire, including shorts and cleats, typical for soccer. - The background has silhouettes of people and a large red figure, possibly a mascot or part of a training setup, but this does not change the sport. Step2: Eliminate incorrect options. - Tennis: Played on a hard or grass court with rackets and a ball, no net with a goal frame. - Volleyball: Played on a court with a net, but the net is lower and there is no goal frame or soccer ball. - Basketball: Played on a court with a hoop and backboard, not a net with a goal frame. Step3: Confirm the sport. - The presence of a soccer goal, a soccer ball, and players on a field strongly indicates the sport is soccer (football). Final answer: B: Soccer

![](images/f133179f7b965dc83389480c780d0c3328c6e132fc86ddb23f2b71ea547d0328.jpg)

Q: What is the sport being played in the image? Options: A: Tennis, B: Soccer, C: Volleyball, D: Basketball A: Step1: Identify key elements in the image. - There is a large net with a goal frame, which is characteristic of a soccer goal. - A soccer ball is visible, positioned in front of the goal. - Two men are standing on a green field that resembles artificial turf, commonly used in soccer. - The players are wearing athletic attire, including shorts and cleats, typical for soccer. - The background has silhouettes of people and a large red figure, possibly a mascot or part of a training setup, but this does not change the sport. Step2: Eliminate incorrect options. - Tennis: Played on a hard or grass court with rackets and a ball, no net with a goal frame. - Volleyball: Played on a court with a net, but the net is lower and there is no goal frame or soccer ball. - Basketball: Played on a court with a hoop and backboard, not a net with a goal frame. Step3: Confirm the sport. - The presence of a soccer goal, a soccer ball, and players on a field strongly indicates the sport is soccer (football). Final answer: B: Soccer

Figure 14: Recognizing the sport. The model answers “Soccer”. VTRACE places most of its top patches on the goal net, the object that identifies the sport, while IFR and FlowTracer spread more patches over the players and the grass. In the text, VTRACE scores the option “Soccer” and the concluding “soccer” highest, and also shades the scene the model describes, such as “large net”, “goal frame” and “soccer ball”; FlowTracer concentrates on the question wording.

![](images/d147d0e23ee87a5efd57d8ac5aa2ce0cd98f3e38628acc76314b33f691450338.jpg)

Q: Find the perimeter of the parallelogram. Choices: (A)32(B)39(C)46(D)78 A: To find the perimeter of the parallelogram, we need to add up the lengths of all four sides. From the image, we are given: - One side is23 ft. - The adjacent side is16 ft. In a parallelogram, opposite sides are equal in length. Therefore: - The side opposite the2 3 ft side is also23 ft. - The side opposite the16 ft side is also16 ft. So, the perimeter P is: P =23 ft +16 ft +23 ft +16 ftP =2 ×23 ft +2 ×16 ftP =46 ft +32 ftP =78 ftThe angle of60° and the dashed line (which appears to be the height) are not needed to calculate the perimeter, as perimeter only depends on the side lengths. Looking at the choices: (A)32(B)39(C)46(D)78The correct answer is78 ft. Final answer: D

![](images/6d38457d45b095c97faa52f4d4664af49104ec3452a8beb0729e710ecbab4e83.jpg)

Q: Find the perimeter of the parallelogram. Choices: (A)32(B)39(C)46(D)78 A: To find the perimeter of the parallelogram, we need to add up the lengths of all four sides. From the image, we are given: - One side is23 ft. - The adjacent side is16 ft. In a parallelogram, opposite sides are equal in length. Therefore: - The side opposite the2 3 ft side is also23 ft. - The side opposite the16 ft side is also16 ft. So, the perimeter P is: P =23 ft +16 ft +23 ft +16 ftP =2 ×23 ft +2 ×16 ftP =46 ft +32 ftP =78 ftThe angle of60° and the dashed line (which appears to be the height) are not needed to calculate the perimeter, as perimeter only depends on the side lengths. Looking at the choices: (A)32(B)39(C)46(D)78The correct answer is78 ft. Final answer: D

![](images/d0d402bb9997244012ef7f235756e1420af867887412385d4bf99d21cdafdabd.jpg)  
Figure 15: Reading the side lengths of a parallelogram. The model adds the side lengths shown in the image, $2 \times 2 3 + 2 \times 1 6 = 7 8 .$ , and answers (D). VTRACE’s top patches cover the “23 ft” and “16 ft” labels, while IFR’s lie mostly along the image border. In the text, IFR highlights little beyond the answer options, whereas VTRACE highlights “perimeter” in the question and the side lengths and sums in the reasoning that lead to the answer.

Q: Find the perimeter of the parallelogram. Choices: (A)32(B)39(C)46(D)78 A: To find the perimeter of the parallelogram, we need to add up the lengths of all four sides. From the image, we are given: - One side is23 ft. - The adjacent side is16 ft. In a parallelogram, opposite sides are equal in length. Therefore: - The side opposite the2 3 ft side is also23 ft. - The side opposite the16 ft side is also16 ft. So, the perimeter P is: P =23 ft +16 ft +23 ft +16 ftP =2 ×23 ft +2 ×16 ftP =46 ft +32 ftP =78 ftThe angle of60° and the dashed line (which appears to be the height) are not needed to calculate the perimeter, as perimeter only depends on the side lengths. Looking at the choices: (A)32(B)39(C)46(D)78The correct answer is78 ft. Final answer: D

IFR  
![](images/aacc9ed2fde6e818853a629ffbc94cd824437b34a618c0320801001cdbc3fab1.jpg)

Q: At Bloomington Consulting, the head of human resources examined how the number of employees with health care benefits varied in response to policy changes. According to the table, what was the rate of change between2014 and2015? (Unit: employees per year) Choices: (A) -1(B)1(C)0(D) -2   
A: To find the rate of change between2014 and2015, we need to calculate the change in the number of employees divided by the change in time. From the table: - In2014, the number of employees with health care benefits was23. - In2015, the number of employees with health care benefits was22. The change in employees = Employees in 2015 - Employees in2014 =22 -23 = -1. The change in time =2015 -2014 =1 year. Therefore, the rate of change = (Change in employees) / (Change in time) = -1 /1 = -1 employees per year. This means the number of employees with health care benefits decreased by1 per year between2014 and2015. Looking at the choices: (A) -1(B)1(C) 0(D) -2The correct answer is (A) -1. Final answer: A Q: At Bloomington Consulting, the head of human resources examined how the number of employees with health care benefits varied in response to policy changes. According to the table, what was the rate of change between2014 and2015? (Unit: employees per year) Choices: (A) -1(B)1(C)0(D) -2   
A: To find the rate of change between2014 and2015, we need to calculate the change in the number of employees divided by the change in time. From the table: - In2014, the number of employees with health care benefits was23. - In2015, the number of employees with health care benefits was22. The change in employees = Employees in 2015 - Employees in2014 =22 -23 = -1. The change in time =2015 -2014 =1 year. Therefore, the rate of change = (Change in employees) / (Change in time) = -1 /1 = -1 employees per year. This means the number of employees with health care benefits decreased by1 per year between2014 and2015. Looking at the choices: (A) -1(B)1(C) 0(D) -2The correct answer is (A) -1. Final answer: A Q: At Bloomington Consulting, the head of human resources examined how the number of employees with health care benefits varied in response to policy changes. According to the table, what was the rate of change between2014 and2015? (Unit: employees per year) Choices: (A) -1(B)1(C)0(D) -2   
A: To find the rate of change between2014 and2015, we need to calculate the change in the number of employees divided by the change in time. From the table: - In2014, the number of employees with health care benefits was23. - In2015, the number of employees with health care benefits was22. The change in employees = Employees in 2015 - Employees in2014 =22 -23 = -1. The change in time =2015 -2014 =1 year. Therefore, the rate of change = (Change in employees) / (Change in time) = -1 /1 = -1 employees per year. This means the number of employees with health care benefits decreased by1 per year between2014 and2015. Looking at the choices: (A) -1(B)1(C) 0(D) -2The correct answer is (A) -1. Final answer: A

Attn Rollout  
![](images/cf627624233d36f600504a5386740282214f8e327b6188988808d78f4b7fe85c.jpg)

![](images/d220d5513770d80a03aa05da8e806637ce3b8816f8e0894e98263e99bc65999c.jpg)

Figure 16: Reading values from a table. The model reads that the count fell from 23 in 2014 to 22 in 2015 and answers (A) −1. VTRACE’s top patches fall on the year column and the 2014 and 2015 rows, while Attention Rollout concentrates on the table’s title bar. In the text, Attention Rollout highlights setup words such as “Bloomington Consulting” and “calculate”, whereas VTRACE highlights the years, the change in employees and the result −1.

![](images/82ab7d4d605f2ccf1e6d51202593480ba5723050364a86a406f3baf5dfb7f675.jpg)  
Figure 17: A harder case: the ball closest to the man. The model answers (A) White. In the text, VTRACE highlights the key question words “ball closest” and “man” and the reasoning about the white cue ball. In the image, its top patches cover the balls on the table, including the white cue ball and the red ball at the man’s hand, but many also fall on the dark background above the table.
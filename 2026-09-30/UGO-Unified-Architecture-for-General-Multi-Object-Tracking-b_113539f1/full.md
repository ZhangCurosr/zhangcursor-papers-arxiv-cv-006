# UGO: Unified Architecture for General Multi-Object Tracking by Segmentation

Jer Pelhan, Alan Lukežic, Matej Kristanˇ Faculty of Computer and Information Science, University of Ljubljana jer.pelhan@fri.uni-lj.si

## Abstract

General multi-object tracking (GMOT) tracks all instances of a user-specified category from a single first-frame exemplar. Prior work relies on bounding boxes and surrogate training, and struggles with non-rigid objects, crowded scenes, and distractors. We introduce UGO, a unified GMOT tracker that pairs a pretrained exemplar-conditioned detection head with an instance-propagation head in a common architecture. A novel training-free, energy-minimization consolidation method converts overlapping proposals into exclusive pixel-wise masks and detections, resolving over-segmentation, duplicates, and conflicts. A hierarchical memory spanning global and instance levels improves recall and per-instance segmentation accuracy using a new memory management protocol. UGO sets a new state-ofthe-art on GMOT benchmarks and video object counting, and is competitive with specialist MOT methods, establishing a strong paradigm for unified, open-category multi-object tracking. The code will be available here.

## 1 Introduction

Given a single instance exemplar, e.g., specified by a bounding box in the first frame, the task of general multi-object tracking (GMOT) is to track all instances ofthe same category throughout the video. This extends two classical problems: single-object tracking (SOT) [16], which extracts a trajectory of a single selected instance of an arbitrary category, and multi-object tracking (MOT) [13], which detects and associates multiple instances of a predefined class using a category-specific detector. GMOT thus unifies the generalization requirement of SOT with the coverage of MOT, while operating without category-specific training data, making it a fundamentally challenging problem.

Most GMOT methods [15; 38; 41; 12] are direct adaptations of the classical online MOT paradigm, combining a detector, Kalman-based instance propagation, and Hungarian matching to associate detections with instance predictions. Alternatively, tracking and detection association is posed as an end-to-end trainable detection problem [23] in the context of query-based detectors [10]. Since sufficiently diverse GMOT training data is unavailable, these methods rely on surrogate training on detection datasets, which cannot capture the complex dynamics of instance interactions. Moreover, all current methods localize targets as bounding boxes, which may be adequate for approximating pedestrians, but poorly approximate general non-compact and deformable objects, making the association unreliable, particularly in crowded scenes with overlapping instances.

In general, all instances are better represented by segmentation masks, which enable pixel-level instance separation. This is explored in video object segmentation [42], where masks of all tracked instances are jointly inferred. However, architectures for joint instance prediction do not scale to a large number of objects and require dedicated training sets. Alternatively, recent video foundation models [33] hold the potential to independently track all instances, but these become unreliable unde strong instance interactions and require external initialization at each new instance.

![](images/3e5254bb13c2d449762001e68c962f47b049cbae912217b0bc71f25bb72164cc.jpg)  
Figure 1: First row: Tracker ID=1 oversegments an occluding fish tracked by ID=2, while a single detector fires on both. UGO resolves the oversegmentations and redundant detection by a novel consolidation module. Second row: The new paradigm enables robust general multi-object tracking by segmentation in cluttered dense scenes through occlusions.

We address the aforementioned issues, by proposing UGO, a Unified General multiple Object tracker, that unifies recent advances in general object detection and the expressiveness of the video segmentation foundation models within a single architecture. UGO applies a video foundation backbone [33] with two lightweight heads, one for detecting instances corresponding to the user-provided exemplar, and another for frame-to-frame identity propagation. Both use the same segmentation module to predict calibrated per-pixel object presence beliefs, which may contain oversegmentation, double detections, and duplicated tracks (Figure 1). These are resolved into unique, pixel-wise exclusive tracked instance segmentations and detections of new instances by a novel training-free consolidation module, formulated as an energy-minimization problem with a global exclusivity objective. The novel design avoids the need for extensive, diverse GMOT-specific training datasets to solve complex combinatorial optimization problems efficiently. To ensure robustness to distractors and to align the detector with within-category instance appearance variation, we propose a hierarchical adaptive memory (HAM) composed of a detector category-level memory and individual per-instance memories. The memories are updated by jointly considering the tracks of all instances. UGO shows insensitivity to the hyperparameters commonly applied in MOT/GMOT settings, confirming its robust design.

Our contributions are: (i) a new general multi-object tracking paradigm with independently applied single-object trackers and detectors, coupled through mask consolidation and memory feedback to achieve a global coordination; (ii) a novel training-free energy-based consolidation module that delivers per-pixel exclusive segmentation masks from per-instance tracks and detections; (iii) a hierarchical adaptive memory (HAM) with a management protocol based on explicit failure detection, ensuring robust instance detection and identity propagation in the presence of distractors. UGO unifies SOT, MOT, and GMOT in a single framework, achieving state-of-the-art performance and defining a new paradigm in category-agnostic multi-object tracking.

## 2 Related Work

Multi-Object Tracking (MOT). In classical MOT pipelines such as SORT [5], DeepSORT [38], Tracktor [4], and ByteTrack [44], detectors produce per-frame boxes or masks, which are then associated by motion or appearance cues. Mask-level variants [37] (MOTS) extend this pipeline to segmentation. The methods excel in closed-world setups with strong class-specific detectors available, while their stability hinges on detector recall – missed detections lead to trajectory fragmentation – thus they underperform on long-tail categories. Recent works such as TrackFormer [27] and TransTrack [35] use DETR-like queries for joint detection and identity propagation. However, as noted in [14], this creates a conflict between category semantics (detection) and instance specificity (association). In addition, most methods require per-category training and lack pixel-level exclusivity across similar instances, causing overlaps and ID switches in crowded scenes.

General single object trackers (SOT). Modern SOT enable tracking of any instance and are highly robust to appearance changes and occlusion [33; 36; 11], while recent memory mechanisms improve reliability under distractors [36; 17]. However, SOT lacks scene-level competition and cannot ensure mutual exclusivity among many look-alikes; naively running multiple SOT instances on MOT datasets is prone to identity swaps. But SOT methods cannot discover new instances appearing in the video, and optimize per-target propagation rather than frame-level joint reasoning over all instances.

General & open-category tracking. General multi-object tracking (GMOT) extends MOT to tracking any instance of an unseen category, specified by a single instance exemplar. The GMOT-40 benchmark [3] established a standardized evaluation setting, demonstrating that a straightforward combination of a one-shot detector [15] and a target association module [32; 41; 6] provides a simple, yet limited baseline. Subsequent methods, including PLGMOT [25] and S-DETR [23], introduce exemplar-conditioned detection and transformer-based association. While PLGMOT enables instance discovery, its tracking-by-detection design limits recall. S-DETR improves this via exemplar-conditioned queries and dynamic track reuse. However, both remain restricted to box-level reasoning and NMS-based conflict resolution, causing identity switches in crowded scenes. Recently, SAM3 [9] extended single-object tracking toward open-vocabulary tracking, integrating detection capabilities, but it still lacks explicit pixel-level interaction reasoning among tracked instances.

## 3 A Unified General Multiple Object Tracker

Given a sequence of N video frames $\{ \mathbf { I } _ { t } \} _ { t = 1 : N }$ and an instance category exemplar bounding box ${ \bf b } _ { 1 } ^ { E }$ provided in the first frame, the task is to track all instances of the same category in the video. We propose a unified general multiple object tracker (UGO) (Figure 2). At time-step t, UGO already tracks some of the instances from the previous frame and proceeds in the current frame as follows. The input image $\mathbf { I } _ { t } \in \mathbb { R } ^ { H \times W \times 3 }$ is encoded by the Hiera [34] backbone into $\mathbf { F } _ { t } \in \mathbb { R } ^ { h \times w \times c }$ , with spatial resolution h × w and c feature channels. The features are passed to two light-weight heads. The first is the SAM2 [33] video segmentation head that propagates tracked instances from the previous time-step (Section 3.1). The second head is based on GECO2 [30] and detects all new instances and potentially also those already tracked, matching the exemplar category (Section 3.2). Both heads apply a pretrained SAM2 mask decoder, delivering compatible per-pixel logits, which enables consolidation of detector and tracker outputs into mutually exclusive masks (Section 3.3). The consolidation influences memory updating: the resolved conflicts trigger the affected tracker memory updates. The individual trackers are thus implicitly coupled through shared pixel-level decisions, forming a feedback loop that reduces future overlaps and stabilizes identity tracking (Section 3.5). This substantially simplifies trajectory management (Section 3.4).

![](images/29739bdbdd08097fe5c94be682beb3a770368a3b6d3a72569c2f73faa63d75f2.jpg)  
Figure 2: UGO unifies detection and frame-to-frame association in a single network. Detector and instance tracker heads produce compatible per-pixel outputs, which enables optimization-based, training-free consolidation and robust tracking.

## 3.1 Frame-to-frame instance propagation

A visual model $\boldsymbol { \mathcal { M } } _ { t - 1 , k } ^ { T }$ is maintained for each k-th tracked instance in form of a memory (Section 3.5). For each instance k, SAM2 head attends the backbone features to the corresponding memory and produces a logit map $\mathbf { T } _ { t , k } \in \mathbb { R } ^ { H \times W }$ , which reflects a belief that each pixel contains the instance, conditioned on its visual model. The output of the frame-to-frame propagation module is thus a set of per-instance tracker logits $\mathcal { T } _ { t } = \{ \mathbf { T } _ { t , k } \} _ { k = 1 : N _ { \tau } }$

## 3.2 General instance localization

Since only a single user-provided exemplar of the tracked instance category is available in the first frame, all other instances (in the first frame and also entering in later frames) have to be detected to initialize new per-instance trackers. We utilize the general-object detector GECO2 [30], a light-weight detection head, operating on the SAM2 backbone. The object category is specified by exemplar memory $\mathcal { M } _ { t } ^ { D } = \mathbf { \bar { \{ b } }  _ { i } ^ { E } \Bigr \} _ { i = 1 : N _ { B } }$ , which is initialized by the first-frame exemplar ${ \bf b } _ { 1 } ^ { E }$ and is updated during tracking (Section 3.5). The detection head localizes the instances and applies the SAM2 segmentation head to produce detection logits $\mathcal { D } _ { t } = \{ \mathbf { D } _ { t , i } \} _ { i = 1 : N _ { \mathcal { D } } }$ , where $\mathbf { D } _ { t , i } \in \mathbb { R } ^ { H \times W }$ reflects the belief of each pixel belonging to the instance, conditioned on the memory.

## 3.3 Detector and tracker output consolidation

The use of the same pre-trained SAM2 segmentation head by the detector and tracker ensures calibrated and compatible logits in $\mathcal { T } _ { t }$ and $\mathcal { D } _ { t }$ , which could be naïvely converted to masks by thresholding at zero. However, these masks indicate all pixels that may belong to the detector/tracker instance, not accounting for other instances. Thus, the masks will not be mutually exclusive, with some objects covered by both detectors and trackers, multiple trackers including distractor objects, and masks of different objects potentially overlapping due to over-segmentation (Figure 3). Because of these nontrivial interactions, the classical MOT bipartite [18; 7] or quadratic [39] optimization, even if redesigned to operate at the per-mask level, and disregarding the significant compute complexity, cannot be applied for conflict resolving.

To address these issues, we propose a new formulation that ensures mutually exclusive masks. We start by defining the logits tensor $\mathbf { L } \in \mathbb { R } ^ { H \times W \times S }$ obtained by concatenating the $N _ { \mathcal { D } }$ detector and $N _ { \mathcal { T } }$ tracker logit maps with a small constant background logit map, $\mathrm { i . e . , \ } \mathbf { L } = \cot ( \mathcal { D } _ { t } , \mathcal { T } _ { t } , \mathbf { 1 } \lambda _ { \mathrm { B G } } )$ For notation compactness, let $\mathbf { L } ( \mathbf { x } , s )$ retrieve the logit value of map s at pixel position x. Next, let $\mathbf { Y } \in [ 1 , \ldots , S ] ^ { W \times H }$ denote the mutually-exclusive pixel labeling with $\mathbf { Y } ( \mathbf { \bar { x } } )$ retrieving the logit map identity assigned to pixel x.

The solution for Y should ensure (i) a unique label on each pixel, (ii) among the masks competing for the same pixels, the trackers should be preferred over detectors, and (iii) among the competing tracker masks, the trackers with masks supported by detections should be preferred. Such labeling can be obtained by minimizing the objective

$$
\hat { \mathcal { L } } ( \mathbf { Y } ) = - \sum _ { \mathbf { x } \in \Omega ; s = \mathbf { Y } ( \mathbf { x } ) } \log \Big ( \mathbf { L } ( \mathbf { x } , s ) \cdot \boldsymbol { \Theta } _ { 1 } ( s ) \cdot \boldsymbol { \Theta } _ { 2 } ( s ) \Big ) ,\tag{1}
$$

where $\Omega = \{ 1 , \dots , H \} \times \{ 1 , \dots , W \}$ denotes the set of pixel positions, and $\Theta _ { 1 } ( \cdot )$ and $\Theta _ { 2 } ( \cdot )$ are unitary potentials enforcing the required properties (i-iii) of the optimal labeling. The first potential $\Theta _ { 1 } ( \cdot )$ is defined as

$$
\Theta _ { 1 } ( s ) = \left\{ \begin{array} { l l } { \frac { 1 } { 2 } \theta _ { 0 } } & { ; s \le N _ { D } } \\ { \theta _ { 0 } + \displaystyle \operatorname* { m a x } _ { d \in \mathcal { D } _ { t } } \mathrm { I o U } \bigl ( \mathbf { Y } \equiv s , d \bigr ) } & { ; N _ { D } < s \le N _ { D } \tau } \\ { 1 } & { ; s \equiv S } \end{array} \right.\tag{2}
$$

where $N _ {  Ḋ \mathcal Ḋ T Ḍ Ḍ } = N _ {  Ḋ \mathcal Ḋ D Ḍ Ḍ } + N _ { \mathcal Ḋ T Ḍ } , \theta _ { 0 }$ is a constant enforcing preference of trackers over detections $( \theta _ { 0 }$ vs $\scriptstyle { \frac { 1 } { 2 } } \theta _ { 0 } )$ , and the IoU term prefers trackers better supported by the detections, assuming a labeling Y. The second potential $\Theta _ { 2 } ( \cdot )$ reduces the scores of instances whose masks induced by labeling Y significantly deviate from their initially computed masks, indicating the other instances have absorbed their pixels,

$$
\Theta _ { 2 } ( s ) = \left\{ { \begin{array} { l l } { \mathrm { I o U } ( \mathbf { Y } \equiv s , L ( : , s ) > 0 ) } & { ; s \leq N _ { \mathscr { D } } \tau } \\ { 1 } & { ; s \equiv S } \end{array} } \right. .\tag{3}
$$

Since (1) is not separable in $\mathbf { Y } ( \mathbf { x } )$ , we introduce auxiliary variables $\mathbf { A } _ { 1 } \in \mathbb { R } ^ { 1 \times S }$ and $\mathbf { A } _ { 2 } \in \mathbb { R } ^ { 1 \times S }$ leading to a surrogate objective

$$
\mathcal { L } ( \mathbf { Y } ) = - \sum _ { \mathbf { x } \in \Omega , s = \mathbf { Y } ( \mathbf { x } ) } \log ( \mathbf { L } ( \mathbf { x } , s ) \mathbf { A } _ { 1 } ( s ) \mathbf { A } _ { 2 } ( s ) ) + \sum _ { s = 1 } ^ { S } N _ { s } \left[ \log ^ { 2 } \frac { \mathbf { A } _ { 1 } ( s ) } { \Theta _ { 1 } ( s ) } + \log ^ { 2 } \frac { \mathbf { A } _ { 2 } ( s ) } { \Theta _ { 2 } ( s ) } \right] ,\tag{4}
$$

with $N _ { s }$ the number of pixels assigned to label s, which can be optimized by MM-based [19] iterations of the following steps (see supplementary material for full derivation).

Assuming a constant $\mathbf { Y } ^ { ( k - 1 ) }$ <sup>)</sup>, estimated at previous step, taking derivatives of (4) w.r.t. log ${ \bf A } _ { 1 }$ and log ${ \bf A } _ { 2 }$ , and equating to zero, leads to the update,

$$
\mathbf { A } _ { 1 } ^ { ( k ) } ( s ) \propto \Theta _ { 1 | \mathbf { Y } ^ { ( k - 1 ) } } ( s ) ; \mathbf { A } _ { 2 } ^ { ( k ) } ( s ) \propto \Theta _ { 2 | \mathbf { Y } ^ { ( k - 1 ) } } ( s ) .\tag{5}
$$

Keeping ${ \bf A } _ { 1 }$ and ${ \bf A } _ { 2 }$ constant, the cost is separable across pixels, leading to maximization over Y as

$$
\mathbf { Y } ^ { ( k ) } ( \mathbf { x } ) = \underset { s \in \{ 1 : S \} } { \operatorname { a r g m a x } } \big ( \mathbf { L } ( \mathbf { x } , s ) \cdot \mathbf { A } _ { 1 } ^ { ( k ) } ( s ) \cdot \mathbf { A } _ { 2 } ^ { ( k ) } ( s ) \big ) .\tag{6}
$$

The optimization thus gradually re-assigns pixels claimed by several detector/tracker instances to either of them, and if a particular instance is poorly supported by the remaining pixels, it is automatically turned off. The labels $\mathbf { Y } ^ { ( 0 ) }$ are initialized by per-pixel argmax over the logits $\mathbf { L } ,$ while the optimization converges within a few iterations.

Finally, severely reduced masks are removed: any label s in Y, whose mask IoU with the initialization mask is smaller than a fixed threshold, is reassigned to the background, i.e., $s = S$ . UGO uses a permissive threshold for trackers $\tau _ { \mathrm { l o } } = 0 . 2$ (since small overlap may be due to distractor correction), and, to ensure a high recall, we set the threshold for detector masks to a moderate level $\tau _ { \mathrm { m d } } = 0 . 6$ (ablated in supplementary material).

## 3.4 Instance track lifecycle management

After consolidation, only two types of mutually-exclusive masks remain: (i) masks corresponding to instance trackers and (ii) masks corresponding to newly detected instances. Standard MOT rules are applied next for trajectory management.

Initialization & Update. New single-target trackers are initialized on detection masks, while the instance trackers are updated with their consolidated masks.

Termination. A tracker is flagged for termination when its mask collapses due to drift, or when the object leaves the field of view. In practice, the flag is raised when consolidation returns an empty mask. If the condition is met for ten consecutive frames, the tracker is terminated.

Initialized track validation. To remove tracks initialized on a false positive detection, they are classified as valid only after frequently confirmed by the detector. As a standard rule, tracks with 30% of their masks confirmed are declared valid, where a tracker mask $M _ { t , k }$ is considered confirmed if its IoU with any of the detectors is greater than the moderate $\tau _ { \mathrm { m d } }$ , i.e.,

$$
\operatorname* { m a x } _ { d \in \mathcal { D } _ { t } } \mathrm { I o U } ( M _ { t , k } , d > 0 ) \geq \tau _ { \mathrm { m d } } .\tag{7}
$$

## 3.5 Hierarchical Adaptive Memory

UGO employs a hierarchical adaptive memory (HAM) to represent the targets at two levels of detail. The category-level memory is used by the detector, while per-instance-level memory is used by the tracker. The two are continually updated to improve detection capabilities and to ensure instance-level tracking robustness. Figure 5 overviews HAM.

Category-level memory $\mathcal { M } _ { t } ^ { D } = \{ { \bf b } _ { i } ^ { E } \} _ { i = 1 : N _ { B } } ,$ , always contains the user-provided exemplar ${ \bf b } _ { 1 } ^ { E }$ and holds a FIFO buffer with three slots, updated by the exemplars proposed from reliable per-instance trackers. At time-step t, we consider all trackers with at least ten frames long trajectories, whose current mask $M _ { t }$ highly overlaps with a detector, i.e., $\mathrm { I o U } ( M _ { t } , d > 0 ) \ge \bar { \tau } _ { \mathrm { h i } }$ , where $\tau _ { \mathrm { h i } } = 0 . 9$ Among these, the current mask-fitted-bounding-box from the most frequently confirmed (7) trajectory updates the category-level memory $\mathcal { M } _ { t } ^ { D }$

Instance-level memory $\mathcal { M } _ { t , k } ^ { T }$ stores segmented examples of k-th instance and is used for its localization by the tracker. As in [36], the memory is split into distractor-resolving memory (DRM), responsible for discriminating between the target and visually-similar objects, and recent appearance memory (RAM), responsible for frame-to-frame segmentation accuracy – each a FIFO buffer with three DRM and four RAM slots. Both are updated from non-empty tracker outputs, with RAM updated as in [36], while we introduce a new DRM update protocol. The standard assumption [36] that the initialization frame provides an iconic target with ground-truth segmentation does not hold in

![](images/4675ce6d82fa5976c7aa60ffebb461d27e99eee2cf1d6a3ceecd42b13fc4f3c8.jpg)  
Figure 3: Oversegmented object at $T = 1$ and tracking in subsequent frames with and without consolidation-driven DRM update. Individual colors indicate an ID, with dashed pattern indicating overlapping IDs. Consolidation-based memory update leads to distractor resolution, while ignoring it leads to tracker failure.

GMOT, where trackers are initialized from potentially inaccurate detections. Therefore, the initialization frame should not be retained in DRM indefinitely [33; 36]. We instead leverage detector–tracker consolidation to identify frames with distractors. DRM is updated when the tracker mask before and after consolidation no longer reflect a high agreement $( \mathrm { I o U } < \tau _ { \mathrm { h i } } )$ , while the consolidated mask is detector-confirmed $( \mathrm { E q . } 7 )$ , indicating correction of distractor over-segmentation. Figure 3 visualizes the benefits of using the proposed DRM updating scheme.

## 4 Experiments

We follow the standard GMOT evaluation protocol [3], where a single target exemplar is provided in the first frame. Performance is evaluated by MOT measures, with the primary being MOTA, which integrates false positives, false negatives, and identity switches. Where possible, we also report HOTA [26], which jointly measures detection (DetA) and association (AssA) accuracy and provides a more reliable overall assessment than MOTA [26]. Several auxiliary measures are used. Identity preservation is measured by IDP, IDR, and their harmonic mean IDF1. Trajectory quality is quantified by MT, PT, and ML, denoting the number of mostly tracked $( > 8 0 \% )$ , partially tracked (20-80%), or mostly lost (< 20%) identities, respectively. We also report false positives (FP), false negatives (FN), F1 score, identity switches (IDSw), and fragmentations (FM).

Implementation details. We use the pretrained SAM2.1 tracking module [33] with a pretrained GeCo2 [30] head, both operating on the same SAM2.1 Hiera-L backbone. The consolidation optimization (Section 3.3) runs for $N _ { \mathrm { i t e r } } = 3$ iterations with parameter $\theta _ { 0 } = 0 . 5$ . The same thresholds for low, moderate and high overlaps $( \tau _ { \mathrm { l o } } = 0 . 2 , \tau _ { \mathrm { m d } } = 0 . 6$ and $\tau _ { \mathrm { h i } } = 0 . 8 )$ are used in all experiments. UGO tracks 100 objects at $\sim 2 . 1 \ : \mathrm { F P S }$ on a single A100 GPU.

## 4.1 Comparison with general multi-object trackers

UGO is compared on GMOT-40 [3] benchmark with GMOT trackers, including the five versions of the current best state-of-the-art S-DETR [23] (Table 1). UGO consistently outperforms all competitors. It outperforms S-DETR [23] by 23% MOTA, reflecting joint improvements in detection quality, association accuracy, and long-term trajectory consistency. It also outperforms S-DETR [23] by 52% IDF1, indicating a substantially better instance identity preservation. We also evaluate SAM3 [9] with default parameters (see supplementary), which only supports prompt-based tracking – we thus initialize it with the category names of target objects. Despite this semantic advantage, UGO surpasses SAM3 by 6% in HOTA and 22% in MOTA, demonstrating markedly stronger instance coverage and overall tracking robustness.

UGO achieves the highest mostly tracked rate (MT), surpassing S-DETR [23] by 25% and the lowest (-39%) mostly lost rate (ML), demonstrating a superior tracking stability. This highlights the effectiveness of the proposed pixel-level consolidation and evidence-gated memory mechanisms in enforcing consistent, globally coordinated identity assignment. As illustrated in Figure 4, the proposed consolidation resolves overlapping masks into mutually exclusive, pixel-consistent instance segmentations. By enforcing competition at the pixel level rather than relying on box-level matching (standard in prior SOTA), consolidation resolves ambiguous shared regions and corrects over-segmentation errors before they accumulate temporally. This prevents identity switches and trajectory fragmentations, especially in dense scenes with visually similar instances.

Table 1: State-of-the-art comparison on the GMOT-40 [3] benchmark.
<table><tr><td></td><td>Detector</td><td>Tracker</td><td>IDF1↑</td><td>MT↑</td><td>ML↓</td><td>FP↓</td><td>FN↓</td><td>F1↑</td><td>IDSw↓</td><td>MOTA↑</td><td>HOTA↑</td></tr><tr><td rowspan="6">Opecuuu-lary</td><td rowspan="4">OVTrack [21]</td><td>DeepSORT [38]</td><td>21.2</td><td>165</td><td>1367</td><td>49984</td><td>160378</td><td>47.7</td><td>1470</td><td>20.2</td><td></td></tr><tr><td>ByteTrack [44]</td><td>20.6</td><td>164</td><td>1345</td><td>51356</td><td>156329</td><td>49.1</td><td>1669</td><td>19.9</td><td></td></tr><tr><td>BoT-SORT [1]</td><td>20.3</td><td>167</td><td>1328</td><td>45721</td><td>163378</td><td>47.1</td><td>3278</td><td>20.0</td><td></td></tr><tr><td>TbQ [23]-SwT</td><td>18.7</td><td>186</td><td>1304</td><td>50784</td><td>154893</td><td>49.7</td><td>6381</td><td>21.3</td><td></td></tr><tr><td>DeepSORT [38]</td><td>41.6</td><td>401</td><td>877</td><td>46610</td><td>141330</td><td>55.0</td><td>2892</td><td>25.5</td><td>-</td></tr><tr><td rowspan="3">GLIP-T [20]</td><td>ByteTrack [44]</td><td>45.1</td><td>447</td><td>746</td><td>52591</td><td>131759</td><td>57.5</td><td>2706</td><td>27.0</td><td></td></tr><tr><td>BoT-SORT [1]</td><td>49.1</td><td>553</td><td>643</td><td>51308</td><td>133462</td><td>57.1</td><td>4675</td><td>27.3</td><td></td></tr><tr><td>TbQ [23]-SwT</td><td>39.8</td><td>581</td><td>592</td><td>49470</td><td>136602</td><td>56.3</td><td>9972</td><td>27.5</td><td></td></tr><tr><td rowspan="9">Tem-Bpsed</td><td rowspan="3">SAM3 [9] GTrack [15]</td><td></td><td>72.8</td><td>1209</td><td>276</td><td>70994</td><td>56131</td><td>75.9</td><td>990</td><td>50.2</td><td>60.1</td></tr><tr><td>DeepSORT [38]</td><td>24.4</td><td>72</td><td>1363</td><td>9000</td><td>208818</td><td>30.4</td><td>1315</td><td>14.5</td><td></td></tr><tr><td>ByteTrack [44]</td><td>32.1</td><td>178</td><td>1069</td><td>23881</td><td>181829</td><td>42.0</td><td>1791</td><td>19.1</td><td></td></tr><tr><td rowspan="3"></td><td>BoT-SORT [1]</td><td>34.0</td><td>251</td><td>978</td><td>22229</td><td>176991</td><td>44.3</td><td>7375</td><td>19.4</td><td>一</td></tr><tr><td>TbQ [23]-SwT</td><td>27.4</td><td>213</td><td>1066</td><td>13507</td><td>182376</td><td>43.0</td><td>6407</td><td>20.6</td><td>-</td></tr><tr><td>DeepSORT [38]</td><td>41.8</td><td>382</td><td>773</td><td>47336</td><td>124257</td><td>60.6</td><td>5131</td><td>31.1</td><td>-</td></tr><tr><td rowspan="4">S-DETR [23]</td><td>ByteTrack [44]</td><td>41.4</td><td>331</td><td>764</td><td>53417</td><td>104765</td><td>65.7</td><td>4204</td><td>33.7</td><td></td></tr><tr><td>BoT-SORT [1]</td><td>47.5</td><td>431</td><td>674</td><td>45769</td><td>119288</td><td>62.4</td><td>6775</td><td>34.1</td><td></td></tr><tr><td>TbQ [23]-SwT</td><td>42.8</td><td>504</td><td>666</td><td>44882</td><td>107894</td><td>66.0</td><td>11664</td><td>35.9</td><td></td></tr><tr><td>TbQ [23]-SwB</td><td>51.3</td><td>1083</td><td>278</td><td>44390</td><td>68189</td><td>77.0</td><td>11252</td><td>50.0</td><td></td></tr><tr><td>UGO</td><td></td><td>77.09</td><td>1349</td><td>170</td><td>58165</td><td>39670</td><td></td><td>81.3 1136</td><td>61.4</td><td></td><td>63.8</td></tr></table>

Before Consolidation  
After Consolidation  
All trajectories (final output)  
![](images/b4c292c90f9760c4a5b7f51d74f17b10305b08fde12357174a8db51cde131353.jpg)  
Figure 4: Incorrect masks spreading over several objects are corrected by the proposed detectortracker consolidation method, enabling accurate memory updates and preventing drifts.

Compared to S-DETR, UGO reduces false negatives by 42% at a cost of 31% higher FP rate, while it overall produces a better temporal coverage of target-category instances, as observed in 6% higher F1 score. Inspection of S-DETR code revealed usage of sequence-specific thresholds to reduce false positives, while UGO relies on a fixed setting, indicating its gains come from architecture. Overall, UGO achieves state-of-the-art performance with strong cross-category generalization.

## 4.2 Comparison with specialist trackers

Next, we compare UGO on benchmarks with trackers specialized for individual object categories. These setups are particularly challenging for generalist UGO, since it has not been fine-tuned per category, unlike its competitors.

Evaluation on AnimalTrack. AnimalTrack benchmark [43] evaluates classical MOT trackers on 10 different animal species. The benchmark provides training sets with category-specific labels to train the MOT trackers for each test category. Since iconic exemplars are not provided, UGO randomly selects five ground-truth bounding boxes from the first frame to define the target category.

Table 2: State-of-the-art comparison on AnimalTrack [43].
<table><tr><td>Method</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>IDP↑</td><td>IDR↑</td><td>MT↑</td><td>PT</td><td>ML↓</td><td>F1↑</td><td>IDSw↓</td><td>FM↓</td></tr><tr><td>ByteTrack [44]</td><td>40.1</td><td>38.5</td><td>51.2</td><td>64.9</td><td>42.3</td><td>310</td><td>465</td><td>329</td><td>0.631</td><td>1309</td><td>3513</td></tr><tr><td>IOUTrack [6]</td><td>41.6</td><td>55.7</td><td>45.7</td><td>51.9</td><td>40.7</td><td>388</td><td>454</td><td>262</td><td>0.762</td><td>4639</td><td>5259</td></tr><tr><td>SORT [5]</td><td>42.8</td><td>55.6</td><td>49.2</td><td>58.5</td><td>42.4</td><td>333</td><td>470</td><td>301</td><td>0.749</td><td>2530</td><td>3730</td></tr><tr><td>OMC [22]</td><td>43.0</td><td>53.4</td><td>50.3</td><td>61.8</td><td>42.4</td><td>324</td><td>478</td><td>302</td><td>0.735</td><td>4938</td><td>7162</td></tr><tr><td>Tracktor++ [4]</td><td>44.2</td><td>55.2</td><td>51.0</td><td>58.5</td><td>45.1</td><td>364</td><td>472</td><td>268</td><td>0.751</td><td>1976</td><td>4149</td></tr><tr><td>TransTrack [35]</td><td>45.4</td><td>48.3</td><td>53.4</td><td>63.4</td><td>46.1</td><td>327</td><td>416</td><td>361</td><td>0.705</td><td>1978</td><td>6459</td></tr><tr><td>QDTrack [29]</td><td>47.0</td><td>55.7</td><td>56.3</td><td>65.6</td><td>49.3</td><td>367</td><td>420</td><td>317</td><td>0.752</td><td>1970</td><td>5656</td></tr><tr><td>SAM3 [9]</td><td>52.1</td><td>43.9</td><td>66.4</td><td>68.4</td><td>64.4</td><td>642</td><td>280</td><td>188</td><td>0.713</td><td>470</td><td>2588</td></tr><tr><td>UGO</td><td>59.4</td><td>53.8</td><td>71.7</td><td>67.6</td><td>76.3</td><td>700</td><td>254</td><td>156</td><td>0.783</td><td>463</td><td>2490</td></tr></table>

Table 3: Tracking performance on the MOT17. The colors denote UGO performing better/worse compared to the individual method, <sup>∗</sup> denotes public detection setup.
<table><tr><td>Method</td><td>HOTA</td><td>DetA</td><td>AssA</td></tr><tr><td>MPNTrack17</td><td>46.6 (7%)</td><td>46.2 (-2%)</td><td>47.3 (18%)</td></tr><tr><td>eTC17</td><td>45.1 (11%)</td><td>44.1 (3%)</td><td>46.4 (19%)</td></tr><tr><td>Tracktor++v2</td><td>45.1 (11%)</td><td>45.3 (0%)</td><td>45.0 (24%)</td></tr><tr><td>ByteTrack [44]</td><td>63.1 (-21%)</td><td>64.5 (-30%)</td><td>62.0 (-10%)</td></tr><tr><td>OC-Sort [8]</td><td>63.2 (-21%)</td><td>63.2 (-28%)</td><td>63.4 (-12%)</td></tr><tr><td>OC-Sort* [8]</td><td>54.6 (-8%)</td><td>47.8 (-5%)</td><td>57.6 (-3%)</td></tr><tr><td>UGO</td><td>50.0</td><td>45.3</td><td>55.6</td></tr></table>

Table 4: Video object counting results on Science-Count [2]. Methods marked with \* accept text prompts instead of exemplars.
<table><tr><td rowspan="2">Method</td><td colspan="2">Penguins</td><td colspan="2">Crystals</td></tr><tr><td>MAE↓</td><td>RMSE↓</td><td>MAE↓</td><td>RMSE↓</td></tr><tr><td>GDINOMASA* [24]</td><td>9.0</td><td>11.5</td><td>72.3</td><td>87.9</td></tr><tr><td>CountGDBByteTrack* [2]</td><td>4.3</td><td>5.5</td><td>71.6</td><td>88.1</td></tr><tr><td>CountVid [2]*</td><td>4.0</td><td>5.3</td><td>69.1</td><td>86.0</td></tr><tr><td>CountGDBByteTrack [2]</td><td>4.0</td><td>4.2</td><td>31.1</td><td>52.8</td></tr><tr><td>GeCosAM2.1 [31]</td><td>11.3</td><td>14.5</td><td>46.1</td><td>82.8</td></tr><tr><td>CountVid [2]</td><td>3.3</td><td>4.8</td><td>33.7</td><td>59.8</td></tr><tr><td>UGO</td><td>2.7</td><td>3.5</td><td>24.6</td><td>43.4</td></tr></table>

Table 2 shows that UGO outperforms all state-of-the-art methods. UGO surpasses the strongest specialist method QDTrack [29] by 26% HOTA and 27% IDF1. Remarkably, UGO tracks 2× more instances than competing trackers, while producing 4× fewer identity switches and half the number of fragmentations. We additionally evaluate SAM3 [9], following Sec. 4.1. Since SAM3 relies on text prompts rather than exemplar conditioning, it benefits from an explicit semantic specification of the target (e.g., “geese”), whereas UGO must infer the category purely from visual evidence in a few exemplars. Despite its semantic advantage and large-scale detection training, UGO surpasses SAM3 by 14% HOTA and 23% MOTA, demonstrating superior performance.

The primary limitation of UGO is a higher FP rate, resulting in a 3% lower MOTA than QDTrack [29]. Inspection reveals that a third of FPs originate from the goose\_3 sequence, where similar bird species are also detected, reflecting ambiguity in exemplar-based category specification. Nevertheless, UGO achieves superior identity quality and tracking performance, highlighting strong generalization without category-specific training.

Evaluation on MOT17. We next evaluate UGO on the human-centric MOT17 benchmark [28], again with the same setup as in AnimalTrack. Table 3 reports a comparison with three standard MOT baselines, and state-of-the-art OC-SORT [8]. UGO outperforms all MOT baseline specialists in HOTA, demonstrating strong joint detection-association reasoning, but lags behind the sota, which exploits in-domain category-specific training, in particular the excellent fine-tuned person detectors, which lead to superior detection accuracy (DetA in Table 3). This is evident from a 24% DetA drop of OC-SORT using a public human detector. Meanwhile, comparable association accuracy to that of OC-SORT<sup>∗</sup> indicates the effectiveness of the consolidation module responsible for association.

## 4.3 Comparison with video object counters

We finally evaluate UGO in video object counting performance on the recent ScienceCount benchmark [2]. The MAE and RMSE [2], computed by counting trajectories without final confirmationbased filtering, are reported in Table 4. UGO outperforms the sota CountVid [2] by 18%/27% in MAE and 25%/27% in RMSE on the Penguins and Crystals subsets, respectively. Notably, CountVid is a video counting specialist, employing a multistage, forward-backward tracking pipeline with separate detection and tracking backbones. While UGO processes the video in a single pass, its dominance over the counting specialist emphasizes the strong generalization capabilities.

Table 5: UGO ablation on GMOT-40 [3].
<table><tr><td>Method</td><td>HOTA</td><td>DetA</td><td>AssA</td><td>MOTA</td><td>FN</td><td>FP</td><td>IDSw</td><td>MT</td><td>PT</td><td>ML</td><td>FM</td><td>IDF1</td></tr><tr><td> $\mathrm { U G O } _ { \overline { { \mathrm { C O N S } } } }$ </td><td>50.98</td><td>49.11</td><td>54.36</td><td>43.12</td><td>61835</td><td>59794</td><td>24173</td><td>1084</td><td>671</td><td>189</td><td>8414</td><td>61.47</td></tr><tr><td> $\operatorname { U G O } _ { \operatorname { H M } } ^ { \sim }$ </td><td>52.35</td><td>39.42</td><td>70.55</td><td>5.88</td><td>34457</td><td>204395</td><td>2402</td><td>1460</td><td>328</td><td>156</td><td>4553</td><td>60.76</td></tr><tr><td> $\mathrm { U G O } _ { \overline { { \Theta _ { 1 } } } }$ </td><td>52.39</td><td>55.56</td><td>50.48</td><td>57.11</td><td>39900</td><td>64272</td><td>5773</td><td>1360</td><td>421</td><td>163</td><td>4478</td><td>56.86</td></tr><tr><td> $\mathrm { U G O } _ { \mathrm { D 4 S } }$ </td><td>61.00</td><td>53.80</td><td>70.49</td><td>57.33</td><td>51892</td><td>55349</td><td>2132</td><td>1252</td><td>430</td><td>262</td><td>4409</td><td>72.81</td></tr><tr><td> $\mathrm { U G O } _ { \overline { { \Theta _ { 2 } } } }$ </td><td>62.76</td><td>55.55</td><td>72.11</td><td>59.16</td><td>40059</td><td>63251</td><td>1375</td><td>1351</td><td>419</td><td>174</td><td>4633</td><td>75.66</td></tr><tr><td> $\mathrm { U G O } _ { \overline { { \mathrm { D R M } } } }$ </td><td>63.19</td><td>56.23</td><td>72.20</td><td>60.30</td><td>41057</td><td>59463</td><td>1244</td><td>1331</td><td>434</td><td>179</td><td>4554</td><td>76.10</td></tr><tr><td> $\mathrm { U G O } _ { \mathrm { C A L } } ^ { - }$ </td><td>63.73</td><td>56.57</td><td>72.98</td><td>61.03</td><td>40126</td><td>58646</td><td>1121</td><td>1344</td><td>423</td><td>177</td><td>4501</td><td>76.89</td></tr><tr><td> $\mathrm { U G O } _ { \mathrm { l e x } }$ </td><td>62.37</td><td>54.53</td><td>72.54</td><td>58.34</td><td>48908</td><td>56773</td><td>1113</td><td>1305</td><td>417</td><td>222</td><td>4406</td><td>75.07</td></tr><tr><td> $\mathrm { U G O } _ { 4 \mathrm { - r a n d - G T } }$ </td><td>63.17</td><td>55.52</td><td>73.13</td><td>58.84</td><td>37425</td><td>66933</td><td>1139</td><td>1372</td><td>429</td><td>143</td><td>4624</td><td>76.34</td></tr><tr><td>UGO</td><td>63.83</td><td>56.72</td><td>73.03</td><td>61.39</td><td>39670</td><td>58165</td><td>1136</td><td>1349</td><td>425</td><td>170</td><td>4473</td><td>77.09</td></tr></table>

## 4.4 Ablation study

UGO design choices are analyzed on GMOT-40 [3] in Table 5 and further in supplementary material.

Consolidation module. Several variations are considered (Table 5). The first, $\mathrm { U G O } _ { \overline { { \mathrm { C O N S } } } }$ , has the consolidation module removed, and each detection is associated by the instance tracker with highest corresponding IoU. $\mathrm { U G O } _ { \overline { { \mathrm { C O N S } } } }$ results in 20% HOTA and 30% MOTA drops compared to UGO and nearly 2× as many trajectory fragmentations, due to missed detections and weak temporal consistency. Next, the consolidation module is replaced by Hungarian matching [18] $( \mathrm { U G O } _ { \mathrm { H M } } )$ for optimal assignment between tracker and detector masks. $\mathrm { U G } \mathrm { \bar { O } } _ { \mathrm { H M } }$ results in 18% HOTA and 90% MOTA performance drops. The results verify the importance and robustness of the proposed consolidation module, which standard MOT-style matching strategies cannot replace.

Calibration of detector and tracker logits. Consolidation relies on direct competition between tracker and detector logits, both computed with the shared SAM2 mask decoder; we verify their compatibility by scaling detector logits to match the tracker’s mean amplitude $( \mathrm { U G O } _ { \mathrm { C A L } } )$ , on GMOT-40. Performance remains unchanged (Table 5), indicating that no additional calibration is required.

Consolidation module potentials. The potentials $\Theta _ { 1 } ( \cdot )$ and $\Theta _ { 2 } ( \cdot )$ in (1) control the consolidation labeling dynamics. Removing $\Theta _ { 1 } ( \cdot )$ , denoted by $\mathrm { U G O } _ { \overline { { \Theta _ { 1 } } } }$ decreases HOTA by 18% and MOTA by 7%, while removing $\Theta _ { 2 } ( \cdot )$ leads to 2% HOTA and 4% MOTA reduction (denoted by $\mathrm { U G O } _ { \overline { { \Theta _ { 2 } } } } )$ $\Theta _ { 1 } ( \cdot )$ is thus central for resolving competition, favoring trackers over detections and reinforcing detector-supported tracks, while $\Theta _ { 2 } ( \cdot )$ suppresses uncertain or unstable masks. Together, they ensure stable, pixel-level consistent identities even in cluttered and highly dynamic scenes. This validates the importance of the proposed potentials for strong performance.

Instance-level memory. Replacing the proposed instance memory (Section 3.5) by the original SAM2 [33] $( \mathrm { U G O } _ { \overline { { \mathrm { D R M } } } } )$ results in 2% MOTA and 1% HOTA drops, 10% more identity switches and more fragmentations. This supports importance of the proposed memory for identity stability. Replacing our memory management with that of [36] $\mathrm { ( U G O _ { D 4 S } ) }$ leads to 4% HOTA and 7% MOTA drops, confirming that the proposed memory management offers a more robust multi-object tracking.

Category-level memory. Removing FIFO and keeping only the initial exemplar in the memory $\mathrm { ( U G \bar { O } _ { 1 e x } ) }$ , leads to 5% MOTA and 2% HOTA drop (Table 5), indicating that adaptive exemplar updates not only enhance instance discovery but also improve long-term identity consistency by stable trajectories mining. Constructing memory from four randomly selected ground-truth boxes from the first frame and keeping it fixed throughout the sequence $\mathrm { ( U G O _ { 4 - r a n d - G T } ) }$ leads to 4% MOTA and 1% HOTA drops, further emphasizing the benefits of our on-the-fly trajectory mining.

## 5 Conclusion

We introduced UGO, a unified general multi-object tracker that integrates exemplar-conditioned detection and mask-based instance propagation within a single architecture. A novel consolidation module resolves conflicting and over-segmented outputs, yielding mutually exclusive masks, while hierarchical memory enables robust global and per-instance modeling. UGO achieves state-of-theart performance in GMOT and video counting, while remaining competitive with specialist MOT methods, thus confirming the potential of the new GMOT design paradigm.

## 6 Acknowledgements

This work was supported by the Slovenian Research Agency program P2-0214 and project J2-60054, as well as the supercomputing network SLING (ARNES, EuroHPC Vega - IZUM), and the Slovenian Ministry of MESY and EC/EuroHPC JU via the project SLAIF (grant number 101254461).

## References

[1] Nir Aharon, Roy Orfaig, and Ben-Zion Bobrovsky. Bot-sort: Robust associations multi-pedestrian tracking. arXiv preprint arXiv:2206.14651, 2022.

[2] N. Amini-Naieni and A. Zisserman. Open-world object counting in videos. In Associationfor Advancement ofArtificial Intelligence Conference (AAAI), 2026.

[3] Hexin Bai, Wensheng Cheng, Peng Chu, Juehuan Liu, Kai Zhang, and Haibin Ling. Gmot-40: A benchmark for generic multiple object tracking. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6719–6728, 2021.

[4] Philipp Bergmann, Tim Meinhardt, and Laura Leal-Taixe. Tracking without bells and whistles. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 941–951, 2019.

[5] Alex Bewley, Zongyuan Ge, Lionel Ott, Fabio Ramos, and Ben Upcroft. Simple online and realtime tracking. In Proceedings of the IEEE International Conference on Image Processing, pages 3464–3468, 2016.

[6] Erik Bochinski, Volker Eiselein, and Thomas Sikora. High-speed tracking-by-detection without using image information. In AVSS, 2017.

[7] Guillem Brasó and Laura Leal-Taixé. Learning a neural solver for multiple object tracking. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6247–6257, 2020.

[8] Jinkun Cao, Jiangmiao Pang, Xinshuo Weng, Rawal Khirodkar, and Kris Kitani. Observation-centric sort: Rethinking sort for robust multi-object tracking. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 9686–9696, 2023.

[9] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, Jie Lei, Tengyu Ma, Baishan Guo, Arpit Kalla, Markus Marks, Joseph Greer, Meng Wang, Peize Sun, Roman Rädle, Triantafyllos Afouras, Effrosyni Mavroudi, Katherine Xu, Tsung-Han Wu, Yu Zhou, Liliane Momeni, Rishi Hazra, Shuangrui Ding, Sagar Vaze, Francois Porcher, Feng Li, Siyuan Li, Aishwarya Kamath, Ho Kei Cheng, Piotr Dollár, Nikhila Ravi, Kate Saenko, Pengchuan Zhang, and Christoph Feichtenhofer. Sam 3: Segment anything with concepts, 2025.

[10] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In European conference on computer vision, pages 213–229. Springer, 2020.

[11] Ho Kei Cheng, Seoung Wug Oh, Brian Price, Joon-Young Lee, and Alexander Schwing. Putting the object back into video object segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3151–3161, 2024.

[12] Peng Chu and Haibin Ling. Famnet: Joint learning of feature, affinity and multi-dimensional assignment for online multiple object tracking. In 2019 IEEE/CVF International Conference on Computer Vision (ICCV), pages 6171–6180, 2019.

[13] Patrick Dendorfer, Aljosa Osep, Anton Milan, Konrad Schindler, Daniel Cremers, Ian Reid, Stefan Roth, and Laura Leal-Taixé. Motchallenge: A benchmark for single-camera multiple target tracking. International Journal ofComputer Vision, 129(4):845–881, 2021.

[14] Ruopeng Gao, Ji Qi, and Limin Wang. Multiple object tracking as id prediction. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 27883–27893, June 2025.

[15] Lianghua Huang, Xin Zhao, and Kaiqi Huang. Globaltrack: A simple and strong baseline for long-term tracking. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 34, pages 11037–11044, 2020.

[16] Matej Kristan, Jiri Matas, Aleš Leonardis, Tomas Vojir, Roman Pflugfelder, Gustavo Fernandez, Georg Nebehay, Fatih Porikli, and Luka Cehovin. A novel performance evaluation methodology for single-target<sup>ˇ</sup> trackers. IEEE Transactions on Pattern Analysis and Machine Intelligence, 38(11):2137–2155, 2016.

[17] Matej Kristan, Jiˇrí Matas, Pavel Tokmakov, Alan Lukežic, Michael Felsberg, Lukaˇ Cehovin Zajc, Khanh-<sup>ˇ</sup> Tung Tran, Xuan-Son Vu, Johanna Björklund, Michal Neoral, et al. The third visual object tracking segmentation vots2025 challenge results. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 7422–7440, 2025.

[18] Harold W Kuhn. The hungarian method for the assignment problem. Naval research logistics quarterly, 2(1-2):83–97, 1955.

[19] Kenneth Lange. MM Optimization Algorithms. Society for Industrial and Applied Mathematics, Philadelphia, PA, 2016.

[20] Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, et al. Grounded language-image pre-training. In

Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10965– 10975, 2022.

[21] Siyuan Li, Tobias Fischer, Lei Ke, Henghui Ding, Martin Danelljan, and Fisher Yu. Ovtrack: Openvocabulary multiple object tracking. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5567–5577, 2023.

[22] Chao Liang, Zhipeng Zhang, Xue Zhou, Bing Li, Yi Lu, and Weiming Hu. One more check: Making" fake background" be tracked again. In Associationfor the Advancement ofArtificial Intelligence (AAAI), 2022.

[23] Qiankun Liu, Yichen Li, Yuqi Jiang, and Ying Fu. Siamese-detr for generic multi-object tracking. IEEE Transactions on Image Processing, 33:3935–3949, 2024.

[24] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European conference on computer vision, pages 38–55. Springer, 2024.

[25] Wenxi Liu, Yuhao Lin, Qi Li, Yinhua She, Yuanlong Yu, Jia Pan, and Jason Gu. Prototype learning based generic multiple object tracking via point-to-box supervision. Pattern Recognition, 154:110588, 2024.

[26] Jonathon Luiten, Aljosa Osep, Patrick Dendorfer, Philip Torr, Andreas Geiger, Laura Leal-Taixé, and Bastian Leibe. Hota: A higher order metric for evaluating multi-object tracking. International journal of computer vision, 129(2):548–578, 2021.

[27] Tim Meinhardt, Alexander Kirillov, Laura Leal-Taixe, and Christoph Feichtenhofer. Trackformer: Multiobject tracking with transformers. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 8844–8854, 2022.

[28] Anton Milan, Laura Leal-Taixé, Ian Reid, Stefan Roth, and Konrad Schindler. MOT16: A benchmark for multi-object tracking. arXiv:1603.00831, 2016.

[29] Jiangmiao Pang, Linlu Qiu, Xia Li, Haofeng Chen, Qi Li, Trevor Darrell, and Fisher Yu. Quasi-dense similarity learning for multiple object tracking. In IEEE International Conference on Computer Vision and Pattern Recognition Conference (CVPR), 2021.

[30] Jer Pelhan, Alan Lukezic, and Matej Kristan. Generalized-scale object counting with gradual query aggregation. In Proceedings of the AAAI Conference on Artificial Intelligence, 2026.

[31] Jer Pelhan, Alan Lukežic, Vitjan Zavrtanik, and Matej Kristan. A novel unified architecture for low-shotˇ counting by detection and segmentation. In Advances in Neural Information Processing Systems, volume 37. Curran Associates, Inc., 2024.

[32] Viresh Ranjan, Udbhav Sharma, Thu Nguyen, and Minh Hoai. Learning to count everything. In CVPR, pages 3394–3403, 2021.

[33] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024.

[34] Chaitanya Ryali, Yuan-Ting Hu, Daniel Bolya, Chen Wei, Haoqi Fan, Po-Yao Huang, Vaibhav Aggarwal, Arkabandhu Chowdhury, Omid Poursaeed, Judy Hoffman, et al. Hiera: A hierarchical vision transformer without the bells-and-whistles. In International Conference on Machine Learning, pages 29441–29454. PMLR, 2023.

[35] Peize Sun, Jinkun Cao, Yi Jiang, Rufeng Zhang, Enze Xie, Zehuan Yuan, Changhu Wang, and Ping Luo. Transtrack: Multiple object tracking with transformer. arXiv:2012.15460, 2020.

[36] Jovana Videnovic, Alan Lukezic, and Matej Kristan. A distractor-aware memory for visual object tracking with SAM2. In Comp. Vis. Patt. Recognition, 2025.

[37] Paul Voigtlaender, Michael Krause, Aljosa Osep, Jonathon Luiten, Berin Balachandar Gnana Sekar, Andreas Geiger, and Bastian Leibe. Mots: Multi-object tracking and segmentation. In IEEE International Conference on Computer Vision and Pattern Recognition Conference (CVPR), 2019.

[38] Nicolai Wojke, Alex Bewley, and Dietrich Paulus. Simple online and realtime tracking with a deep association metric. In IEEE International Conference in Image Processing (ICIP), 2017.

[39] Stephen J. Wright. Continuous Optimization (Nonlinear and Linear Programming). Princeton University Press, 2015.

[40] Tong Tong Wu and Kenneth Lange. The mm alternative to em. Statistical Science, 25(4), Nov. 2010.

[41] Yu Xiang, Alexandre Alahi, and Silvio Savarese. Learning to track: Online multi-object tracking by decision making. In International Conference on Computer Vision (ICCV), 2015.

[42] Zongxin Yang, Yunchao Wei, and Yi Yang. Associating objects with transformers for video object segmentation. Advances in Neural Information Processing Systems, 34:2491–2502, 2021.

[43] Libo Zhang, Junyuan Gao, Zhen Xiao, and Heng Fan. Animaltrack: A benchmark for multi-animal tracking in the wild. International Journal ofComputer Vision, 131(2):496–513, 2023.

[44] Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, and Xinggang Wang. Bytetrack: Multi-object tracking by associating every detection box. In Proceedings of the European Conference on Computer Vision, pages 1–21, 2022.

## A The mask consolidation algorithm

We provide further details on derivation of the UGO mask consolidation algorithm for the detector and tracker masks. We first introduce the notations, then define the original cost function and the surrogate loss, and then proceed to derive the corresponding minimization algorithm.

Let $\Omega = \{ 1 , \dots , H \} \times \{ 1 , \dots , W \}$ be the pixel domain and let $S = N _ { \mathscr D } + N _ { \mathscr T } + 1$ denote the number of labels, i.e., potentially competing masks for explaining individual pixels (detections, trackers, and background). We form a logit tensor

$$
\mathbf { L } \in \mathbb { R } ^ { H \times W \times S } ,\tag{8}
$$

by concatenating all detector logits, all tracker logits, and a constant background logit map $( \mathrm { i } . \mathrm { e } . , \lambda _ { \mathrm { B G } } )$ Let ${ \bf L } ( { \bf x } , s )$ denote the value of the tensor at pixel location x for label s.

We seek a mutually-exclusive pixel labeling $\mathbf { Y } : \Omega  \{ 1 , \dots , S \}$ , with $\mathbf { Y } ( \mathbf { x } )$ reading out the label assigned to a pixel x. Further, let $N _ { s } ( { \bf Y } )$ be a function that counts the number of pixels assigned a label s in the labeling Y,

$$
N _ { s } ( \mathbf { Y } ) = \sum _ { \mathbf { x } \in \Omega } \mathbf { 1 } _ { [ \mathbf { Y } ( \mathbf { x } ) \equiv s ] } ,\tag{9}
$$

where $\mathbf { Y } ( \mathbf { x } ) \equiv s$ verifies that the label at pixel x is equal to s in the labeling Y.

## A.1 The labeling cost function

To enforce a desired behavior of the optimization (as discussed in the paper, Section 3.3), we introduce two label-wise potentials $\Theta _ { 1 } ( s ; \mathbf { Y } )$ and $\Theta _ { 2 } ( s ; \mathbf { Y } )$ – for brevity we will omit Y in the following, i.e., $\Theta _ { 1 } ( s )$ and $\Theta _ { 2 } ( s )$ . We define these potentials to encode: (i) a preference for trackers over detections, (ii) a preference for trackers with high detector support, and (iii) suppression of unstable labels whose final mask deviates from its initial mask. Concretely, we make the potentials depend on Y through IoU terms between masks induced by Y and (binarized) initial masks (see Section 3.3 in the paper for definitions). The labeling cost function to be optimized is thus

$$
\hat { \mathcal { L } } ( \mathbf { Y } ) = - \sum _ { \mathbf { x } \in \Omega ; s = \mathbf { Y } ( \mathbf { x } ) } \log \Big ( \mathbf { L } ( \mathbf { x } , s ) \cdot \boldsymbol { \Theta } _ { 1 } ( s ) \cdot \boldsymbol { \Theta } _ { 2 } ( s ) \Big ) ,\tag{10}
$$

where $\mathbf { L } ( \mathbf { x } , s ) \ > \ 0$ always holds, as the label set includes a background label with a constant nonnegative logit $\lambda _ { \mathrm { B G } } > 0$ . Consequently, any detector or tracker label on a non-positive logit cannot be selected over the background. In practice, the IoU-based potentials are also lower-bounded by a small $\epsilon > 0$ , ensuring $\Theta _ { 1 } \bar { ( } s ) \Theta _ { 2 } ( s ) > 0$ and making the logarithm well-defined.

## A.2 Reweighted surrogate objective

Due to nonlinear interaction between pixel labeling in $\Theta _ { 1 } ( s )$ and $\Theta _ { 2 } ( s )$ , the objective (10) does not adhere to simple optimization over $\dot { \mathbf { Y } } .$ . Thus, auxiliary variables $\dot { A _ { 1 } ( s ) }$ and $A _ { 2 } ( s )$ are introduced for $\Theta _ { 1 } ( s , \mathbf { Y } )$ and $\Theta _ { 2 } ( s , \mathbf { Y } )$ that decouple the logits from the latter and make pixel labeling fully separable. This leads to the following surrogate objective

$$
\begin{array} { r l r } & { } & { \mathcal { L } ( \mathbf { Y } , A _ { 1 } , A _ { 2 } ) = } \\ & { } & { - \displaystyle \sum _ { \mathbf { x } \in \Omega } \Big ( \log ( \mathbf { L } ( \mathbf { x } , \mathbf { Y } ( \mathbf { x } ) ) A _ { 1 } ( \mathbf { Y } ( \mathbf { x } ) ) A _ { 2 } ( \mathbf { Y } ( \mathbf { x } ) ) ) \Big ) } \\ & { } & { + \displaystyle \sum _ { s = 1 : S } N _ { s } \Big ( \big ( \log A _ { 1 } ( s ) - \log \Theta _ { 1 } ( s ) \big ) ^ { 2 } + } \\ & { } & { \big ( \log A _ { 2 } ( s ) - \log \Theta _ { 2 } ( s ) \big ) ^ { 2 } \Big ) , } \end{array}\tag{11}
$$

where the $N _ { s }$ is introduced to remove the influence of the mask size, i.e., such that masks correspond ing to large objects are not preferred over those of small objects.

This objective can be minimized by an iterated scheme that exchanges between estimation of $A _ { 1 } ( s )$ and $A _ { 2 } ( s )$ and estimation of Y, closely following the standard Majorize-Minorize [40] approach. In particular, the following steps are exchanged:

Step 1 assumes an estimate ${ \bf \ddot { Y } } ^ { ( k - 1 ) }$ from previous iteration and solves the following minimization:

$$
A _ { 1 } ^ { ( k ) } , A _ { 2 } ^ { ( k ) } = \arg \operatorname* { m i n } _ { A _ { 1 } , A _ { 2 } } \mathcal { L } ( \mathbf { Y } ^ { ( k - 1 ) } , A _ { 1 } , A _ { 2 } ) .\tag{12}
$$

Step 2 then fixes ${ A } _ { 1 } ^ { ( k ) }$ and ${ A } _ { 2 } ^ { ( k ) }$ , and solves the following minimization:

$$
\mathbf { Y } ^ { ( k ) } = \arg \operatorname* { m i n } _ { \mathbf { V } } \mathcal { L } ( \mathbf { Y } , A _ { 1 } ^ { ( k ) } , A _ { 2 } ^ { ( k ) } ) .\tag{13}
$$

## A.3 Step 1 minimization problem

The objective (11) is rewritten into

$$
\mathcal { L } ( \mathbf { Y } ^ { ( k - 1 ) } , A _ { 1 } , A _ { 2 } ) = \sum _ { s = 1 : S } \Big (\tag{14}
$$

$$
- \sum _ { \mathbf { x } \in \Omega } \Big [ \log ( A _ { 1 } ( s ) A _ { 2 } ( s ) ) 1 _ { [ \mathbf { Y } ^ { ( k - 1 ) } ( \mathbf { x } ) \equiv s ] } + \varepsilon ( s ) \Big ]\tag{15}
$$

$$
+ N _ { s } ^ { ( k - 1 ) } \Big ( ( \log A _ { 1 } ( s ) - \log \Theta _ { 1 } ^ { ( k - 1 ) } ( s ) ) ^ { 2 }\tag{16}
$$

$$
+ ( \log A _ { 2 } ( s ) - \log \Theta _ { 2 } ^ { ( k - 1 ) } ( s ) ) ^ { 2 } \Bigl ) \Bigl ) ,\tag{17}
$$

where $\varepsilon ( s )$ absorbs the terms not depending on $A _ { 1 }$ and $A _ { 2 }$ , while $\Theta _ { 1 } ^ { ( k - 1 ) } , \Theta _ { 1 } ^ { ( k - 1 ) }$ and $N _ { s } ^ { ( k - 1 ) }$ are evaluated at $\mathbf { Y } ^ { ( k - 1 ) }$

Then the updates for $A _ { 1 } ^ { ( k ) } ( s )$ and $A _ { 2 } ^ { ( k ) } ( s )$ are obtained by differentiating the objective and setting to zero, i.e.,

$$
\frac { \partial \mathcal { L } ( \mathbf { Y } ^ { ( k - 1 ) } , A _ { 1 } ( s ) , A _ { 2 } ( s ) ) } { \partial \mathrm { l o g } A _ { 1 } ( s ) } \equiv 0\tag{18}
$$

$$
\frac { \partial \mathcal { L } ( \mathbf { Y } ^ { ( k - 1 ) } , A _ { 1 } ( s ) , A _ { 2 } ( s ) ) } { \partial \log A _ { 2 } ( s ) } \equiv 0 .\tag{19}
$$

The derivative in (18) is defined as

$$
\frac { \partial \mathcal { L } } { \partial \log A _ { 1 } ( s ) } = - N _ { s } + 2 N _ { s } \log A _ { 1 } ( s ) - 2 N _ { s } \log \Theta _ { 1 } ^ { ( k - 1 ) } ( s ) ,\tag{20}
$$

which gives the following update for $A _ { 1 } ^ { ( k ) } ( s )$

$$
A _ { 1 } ^ { ( k ) } ( s ) = \Theta _ { 1 } ^ { ( k - 1 ) } ( s ) \cdot e ^ { 1 / 2 } ,\tag{21}
$$

while a similar derivation from (19) with $\frac { \partial \mathcal { L } } { \partial \log A _ { 2 } ( s ) } \equiv 0$ gives

$$
A _ { 2 } ^ { ( k ) } ( s ) = \Theta _ { 2 } ^ { ( k - 1 ) } ( s ) \cdot e ^ { 1 / 2 } .\tag{22}
$$

## A.4 Step 2 minimization problem

In the second step, the auxiliary variables are fixed, the labeling Y can be obtained from (11) by minimizing

$$
\begin{array} { r l } & { \mathbf { Y } ^ { ( k ) } = \underset { \mathbf { Y } } { \arg \operatorname* { m i n } } \mathcal { L } ( \mathbf { Y } , A _ { 1 } ^ { ( k ) } , A _ { 2 } ^ { ( k ) } ) } \\ & { \quad \quad = \arg \underset { \mathbf { Y } } { \arg \operatorname* { m a x } } \quad \underset { \mathbf { x } \in \Omega ; s = \mathbf { Y } ( x ) } { \sum } L ( \mathbf { x } , s ) A _ { 1 } ^ { ( k ) } ( s ) A _ { 2 } ^ { ( k ) } ( s ) , } \end{array}\tag{23}
$$

which decomposes over pixels, thus the labels can be computed for each pixel x separately:

$$
\mathbf { Y } ^ { ( k ) } ( \mathbf { x } ) = \arg \operatorname* { m a x } _ { \mathbf { Y } ( \mathbf { x } ) } { L ( \mathbf { x } , \mathbf { Y } ( \mathbf { x } ) ) A _ { 1 } ^ { ( k ) } ( \mathbf { Y } ( \mathbf { x } ) ) A _ { 2 } ^ { ( k ) } ( \mathbf { Y } ( \mathbf { x } ) ) } .\tag{24}
$$

## B HAM Architecture Details

The hierarchical adaptive memory (HAM), described in Section 3.5, consists of two complementary components: a category-level memory $\mathcal { M } _ { t } ^ { D }$ used for exemplar-conditioned detection, and perinstance memories $\{ \breve { M } _ { t , k } ^ { \breve { T } } \}$ used for instance propagation, see Figure 5. The category-level memory maintains a compact set of reliable exemplars, while each instance-level memory is decomposed into distractor-resolving (DRM) and recent appearance (RAM) buffers, enabling robust discrimination and temporal adaptation. The two levels are tightly coupled through consolidation: reliable tracks update $\dot { \mathcal { M } } _ { t } ^ { D }$ , while consolidation-driven corrections trigger updates of $\mathcal { M } _ { t , k } ^ { T }$ , ensuring consistent global and instance-level representations.

![](images/69769fa4b81e96c1c0a443454b72ae921d66ed6490c88816e7fa6b5cbd4045c2.jpg)  
Figure 5: HAM overview: stable tracks provide robust updates for detector category-level memory, while instance-level memories are updated robustly by the result of consolidation analysis.

Table 6: State-of-the-art comparison on AnimalTrack [43].
<table><tr><td>Method</td><td>HOTA↑</td><td>MOTA↑</td><td>IDF1↑</td><td>IDP↑</td><td></td><td>IDR↑ MT↑</td><td>PT</td><td>ML↓</td><td>FP↓</td><td>FN↓</td><td>Pr↑</td><td>Re↑</td><td>F1↑</td><td>IDSw↓</td><td>FM↓</td></tr><tr><td>JDE</td><td>26.8</td><td>27.3</td><td>31.0</td><td>51.0</td><td>22.0</td><td>106</td><td>414</td><td>584</td><td>17887</td><td>155623</td><td>0.830</td><td>0.360</td><td>0.502</td><td>3187</td><td>5031</td></tr><tr><td>FairMOT</td><td>30.6</td><td>29.0</td><td>38.8</td><td>62.8</td><td>28.0</td><td>143</td><td>462</td><td>499</td><td>17653</td><td>152624</td><td>0.837</td><td>0.372</td><td>0.515</td><td>2335</td><td>5447</td></tr><tr><td>Trackformer</td><td>31.0</td><td>20.4</td><td>36.5</td><td>40.9</td><td>32.8</td><td>230</td><td>491</td><td>383</td><td>70404</td><td>118724</td><td>0.639</td><td>0.512</td><td>0.568</td><td>4355</td><td>3725</td></tr><tr><td>TADAM</td><td>32.5</td><td>36.5</td><td>37.2</td><td>44.4</td><td>32.0</td><td>258</td><td>495</td><td>351</td><td>41728</td><td>110048</td><td>0.761</td><td>0.547</td><td>0.637</td><td>2538</td><td>4469</td></tr><tr><td>DeepSORT</td><td>32.8</td><td>41.4</td><td>35.2</td><td>49.7</td><td>27.2</td><td>213</td><td>452</td><td>439</td><td>14131</td><td>124747</td><td>0.893</td><td>0.487</td><td>0.630</td><td>3503</td><td>4527</td></tr><tr><td>ByteTrack</td><td>40.1</td><td>38.5</td><td>51.2</td><td>64.9</td><td>42.3</td><td>310</td><td>465</td><td>329</td><td>31591</td><td>116587</td><td>0.800</td><td>0.521</td><td>0.631</td><td>1309</td><td>3513</td></tr><tr><td>IOUTrack</td><td>41.6</td><td>55.7</td><td>45.7</td><td>51.9</td><td>40.7</td><td>388</td><td>454</td><td>262</td><td>25206</td><td>77847</td><td>0.868</td><td>0.680</td><td>0.762</td><td>4639</td><td>5259</td></tr><tr><td>SORT</td><td>42.8</td><td>55.6</td><td>49.2</td><td>58.5</td><td>42.4</td><td>333</td><td>470</td><td>301</td><td>19099</td><td>86257</td><td>0.891</td><td>0.645</td><td>0.749</td><td>2530</td><td>3730</td></tr><tr><td>OMC</td><td>43.0</td><td>53.4</td><td>50.3</td><td>61.8</td><td>42.4</td><td>324</td><td>478</td><td>302</td><td>15910</td><td>92570</td><td>0.904</td><td>0.619</td><td>0.735</td><td>4938</td><td>7162</td></tr><tr><td>Tracktor++</td><td>44.2</td><td>55.2</td><td>51.0</td><td>58.5</td><td>45.1</td><td>364</td><td>472</td><td>268</td><td>25477</td><td>81538</td><td>0.864</td><td>0.665</td><td>0.751</td><td>1976</td><td>4149</td></tr><tr><td>TransTrack</td><td>45.4</td><td>48.3</td><td>53.4</td><td>63.4</td><td>46.1</td><td>327</td><td>416</td><td>361</td><td>28553</td><td>95212</td><td>0.838</td><td>0.608</td><td>0.705</td><td>1978</td><td>6459</td></tr><tr><td>QDTrack</td><td>47.0</td><td>55.7</td><td>56.3</td><td>65.6</td><td>49.3</td><td>367</td><td>420</td><td>317</td><td>22696</td><td>83057</td><td>0.876</td><td>0.658</td><td>0.752</td><td>1970</td><td>5656</td></tr><tr><td>SAM3</td><td>52.1</td><td>43.9</td><td>66.4</td><td>68.4</td><td>64.4</td><td>642</td><td>280</td><td>188</td><td>60979</td><td>74880</td><td>0.734</td><td>0.692</td><td>0.713</td><td>470</td><td>2588</td></tr><tr><td>UGO</td><td>59.4</td><td>53.8</td><td>71.7</td><td>67.6</td><td>76.3</td><td>700</td><td>254</td><td>156</td><td>71695</td><td>40157</td><td>0.739</td><td>0.835</td><td>0.783</td><td>463</td><td>2490</td></tr></table>

## C Experimental Evaluation on Animal Track

Table 6 provides extended results on AnimalTrack, including false positives (FP), false negatives (FN), precision, recall, and F1 score (Table 6). The results show that UGO besides achieving the highest HOTA, it achieves the highest recall and F1 score among all compared trackers. In particular, UGO reduces FN (40157 FN) at least two-fold compared to all specialist methods, including QDTrack (83057 FN), demonstrating significantly better temporal coverage of target instances. Although UGO produces more false positives than some specialist detectors (e.g., OMC or QDTrack), it achieves the strongest overall F1 performance. This confirms that UGO favors consistent instance coverage and identity preservation over overly conservative detection suppression.

Comparison with SAM3 (text-prompted MOT). We additionally evaluate SAM3 [9] on AnimalTrack. Unlike UGO, SAM3 does not support exemplar-conditioned multi-object tracking in the GMOT sense, where a single first-frame exemplar defines the category for the entire video. While SAM3 allows image exemplars as prompts, these are applied within the same frame (e.g., for the detection of all objects of the same category in that frame) and are not designed, nor do they work, as a persistent cross-frame category specification. We therefore evaluate SAM3 using a text prompt specifying the ground-truth category for each sequence, which aligns with its standard usage in video tracking.

Text prompting offers a semantic advantage over exemplar-based specification, particularly for finegrained categories. A textual label such as “duck” or “goose” explicitly defines the target class, whereas a few visual exemplars may not clearly indicate whether only that species or also visually similar birds should be tracked. For example, bird species can appear highly similar in crowded scenes, making the category boundary ambiguous from a few exemplars. Moreover, SAM3 is trained on large-scale image and video segmentation data spanning a broad set of visual concepts, which supports strong generalization under semantic (text) prompting. In contrast, the few-shot detector used in UGO is not explicitly trained for the target categories and must rely solely on limited exemplar supervision. Despite this advantage, UGO outperforms SAM3 on the primary tracking metrics. This exposes an important constraint/limitation of UGO. Since the target category is specified only through visual exemplars, UGO cannot explicitly control the semantic granularity of the category to be tracked, which is inherited from GECO2 detector [30].

![](images/95a4022c8cbf69ae6b18e482cfdc74fd3105740c37176b6f0d74b01577633b86.jpg)  
Figure 6: Two sequences from AnimalTrack where UGO produces the most false detections. Red bounding boxes are overlaid to highlight the relevant regions. Ground-truth bounding boxes are shown in green – note that some instances are missing in the ground truth annotations.

In contrast, UGO must infer the category solely from visual evidence contained in a few exemplars, without access to explicit semantic supervision, which is a hard task in fine-grained classification Figure 6. Despite this advantage, UGO outperforms SAM3 on the primary tracking metrics. UGO outperforms SAM3 in HOTA by 14% and a remarkable 23% MOTA. While SAM3 produces fewer identity switches and fragmentations, it is because of lower detection recall (20% lower relatively) and reduced overall instance coverage. These results highlight that the proposed exemplar-driven consolidation and memory mechanism provides stronger identity consistency and more complete tracking, even without access to explicit semantic category labels.

For evaluating SAM3 [9], we use the default parameters of the official video predictor. Table 7 summarizes these thresholds. SAM3 uses a loose association threshold to avoid duplicate masklet creation, while a stricter threshold determines whether an existing masklet is sufficiently supported by detector evidence. Newly initialized masklets are further filtered during a hotstart period, in which unmatched masklets or those repeatedly duplicating existing trajectories are removed. In contrast, UGO omits many of these thresholds by using the proposed consolidation module, since duplicate handling, new instance discovery, even mask correction, and instance memory update logic are handled in a single unified step.

Table 7: Threshold summary for SAM3 [9].
<table><tr><td>Parameter</td><td>Default</td><td>Role</td></tr><tr><td>assoc_iou_thresh</td><td>0.1</td><td>Loose detection-to-track IoU threshold for deciding whether a detection is new.</td></tr><tr><td>trk_assoc_iou_thresh</td><td>0.5</td><td>Stricter IoU threshold for deciding whether a tracker masklet is matched.</td></tr><tr><td>new_det_thresh</td><td>0.7</td><td>Minimum detector score required to spawn a new object.</td></tr><tr><td>HIGH_CONF_THRESH</td><td>0.8</td><td>Threshold for high-confidence detector reconditioning.</td></tr><tr><td>iou_thresh_recondition</td><td>0.8</td><td>Default IoU threshold for detector-to- tracker memory refresh.</td></tr><tr><td colspan="3">suppress_overlapping_based _on_recent_occlusion_threshold 0.7</td></tr><tr><td colspan="2"></td><td>IoU threshold for suppressing highly over- lapping tracker masks based on recent oc-</td></tr><tr><td colspan="2">detector_tracker_confirmation0.5</td><td>clusion history. Ratio of detector-confirmed tracker masks required to keep a masklet; oth-</td></tr><tr><td>hotstart_delay</td><td>15</td><td>Number of frames used to judge early masklet stability.</td></tr><tr><td>hotstart_unmatch_thresh</td><td>8</td><td>Number of unmatched frames after which a new masklet is removed.</td></tr><tr><td>hotstart_dup_thresh</td><td>8</td><td>Number of duplicated frames before hot- start removal.</td></tr><tr><td>max_trk_keep_alive</td><td>30</td><td>Maximum keep-alive counter.</td></tr><tr><td>recondition_every_nth_frame</td><td>16</td><td>Periodic detector-based tracker refresh in- terval.</td></tr><tr><td>masklet_confirmation _consecutive_det_thresh</td><td>3</td><td>Number of detections required if confir-</td></tr><tr><td>mask logit threshold</td><td>0</td><td>mation is enabled. Detector/tracker mask logits are binarized</td></tr><tr><td>NO_OBJ_LOGIT</td><td>-10</td><td>with threshold  $> 0 .$  Logit assigned to suppressed masks be-</td></tr><tr><td></td><td></td><td>fore memory encoding.</td></tr></table>

## D Ablation Study

This section analyzes the sensitivity of UGO to the conservative gates used by the consolidation module and the memory/lifecycle rules. These gates control when masks are accepted, suppressed, or used for memory updates. All experiments are conducted on GMOT-40 [3] by varying one gate at a time while keeping the remaining gates fixed to the default configuration. Results are reported in Table 8.

Table 8: UGO ablation on GMOT-40 [3].
<table><tr><td>Method</td><td>HOTA</td><td>DetA</td><td>AssA</td><td>MOTA</td><td>FN</td><td>FP</td><td>IDSw</td><td>MT</td><td>PT</td><td>ML</td><td>FM</td><td>IDF1</td></tr><tr><td> $\theta _ { 0 } = 0 . 3$ </td><td>63.71</td><td>56.60</td><td>72.91</td><td>61.12</td><td>40195</td><td>58347</td><td>1130</td><td>1348</td><td>426</td><td>170</td><td>4494</td><td>76.85</td></tr><tr><td> $\theta _ { 0 } = 0 . 4$ </td><td>63.75</td><td>56.62</td><td>72.96</td><td>61.15</td><td>40000</td><td>58416</td><td>1165</td><td>1345</td><td>431</td><td>168</td><td>4500</td><td>76.92</td></tr><tr><td> $\theta _ { 0 } = 0 . 6$ </td><td>63.82</td><td>56.65</td><td>73.09</td><td>61.20</td><td>39769</td><td>58605</td><td>1092</td><td>1353</td><td>422</td><td>169</td><td>4463</td><td>77.03</td></tr><tr><td> $\theta _ { 0 } = 0 . 7$ </td><td>63.77</td><td>56.64</td><td>72.98</td><td>61.20</td><td>39695</td><td>58666</td><td>1107</td><td>1344</td><td>433</td><td>167</td><td>4487</td><td>77.00</td></tr><tr><td> $\theta _ { 0 } = 0 . 8$ </td><td>63.62</td><td>56.57</td><td>72.72</td><td>60.98</td><td>39996</td><td>58942</td><td>1082</td><td>1347</td><td>431</td><td>166</td><td>4492</td><td>76.78</td></tr><tr><td> $\tau _ { \mathrm { m d } } = 0 . 4 5$ </td><td>63.45</td><td>56.33</td><td>72.70</td><td>60.37</td><td>38370</td><td>61941</td><td>1269</td><td>1359</td><td>419</td><td>166</td><td>4575</td><td>76.49</td></tr><tr><td> $\tau _ { \mathrm { m d } } = 0 . 5$ </td><td>63.66</td><td>56.45</td><td>73.01</td><td>60.74</td><td>38617</td><td>60793</td><td>1219</td><td>1357</td><td>422</td><td>165</td><td>4532</td><td>76.80</td></tr><tr><td> $\tau _ { \mathrm { m d } } = 0 . 5 5$ </td><td>63.67</td><td>56.51</td><td>72.96</td><td>60.76</td><td>39616</td><td>59756</td><td>1212</td><td>1354</td><td>420</td><td>170</td><td>4506</td><td>76.75</td></tr><tr><td> $\tau _ { \mathrm { m d } } = 0 . 6 5$ </td><td>63.73</td><td>56.66</td><td>72.86</td><td>61.32</td><td>40446</td><td>57620</td><td>1091</td><td>1342</td><td>426</td><td>176</td><td>4492</td><td>76.90</td></tr><tr><td> $\tau _ { \mathrm { m d } } = 0 . 7$ </td><td>63.56</td><td>56.57</td><td>72.59</td><td>61.45</td><td>41299</td><td>56408</td><td>1103</td><td>1331</td><td>434</td><td>179</td><td>4466</td><td>76.73</td></tr><tr><td> $\tau _ { \mathrm { l o } } = 0 . 1$ </td><td>63.78</td><td>56.67</td><td>72.99</td><td>61.14</td><td>39461</td><td>58990</td><td>1171</td><td>1351</td><td>427</td><td>166</td><td>4469</td><td>76.98</td></tr><tr><td> $\tau _ { \mathrm { l o } } = 0 . 3$ </td><td>63.71</td><td>56.72</td><td>72.76</td><td>61.36</td><td>39771</td><td>58148</td><td>1137</td><td>1350</td><td>427</td><td>167</td><td>4469</td><td>76.88</td></tr><tr><td> $\tau _ { \mathrm { h i } } = 0 . 7 5$ </td><td>63.65</td><td>56.65</td><td>72.69</td><td>61.29</td><td>40050</td><td>58023</td><td>1162</td><td>1341</td><td>437</td><td>166</td><td>4488</td><td>76.81</td></tr><tr><td> $\tau _ { \mathrm { h i } } = 0 . 8 5$ </td><td>63.72</td><td>56.66</td><td>72.87</td><td>61.19</td><td>39762</td><td>58613</td><td>1118</td><td>1346</td><td>426</td><td>172</td><td>4482</td><td>76.82</td></tr><tr><td> $\tau _ { \mathrm { h i } } = 0 . 9 5$ </td><td>63.26</td><td>56.25</td><td>72.32</td><td>60.43</td><td>40144</td><td>60185</td><td>1108</td><td>1356</td><td>405</td><td>183</td><td>4340</td><td>76.19</td></tr><tr><td>UGO</td><td>63.83</td><td>56.72</td><td>73.03</td><td>61.39</td><td>39670</td><td>58165</td><td>1136</td><td>1349</td><td>425</td><td>170</td><td>4473</td><td>77.09</td></tr></table>

Tracker preference strength $\theta _ { 0 } .$ . The constant $\theta _ { 0 }$ controls the preference of trackers over detector logits in $\Theta _ { 1 } ( \cdot )$ , and it also sets the scale of the detector-support term that boosts tracker masks consistent with detections. We evaluate $\theta _ { 0 } \in \{ 0 . 3 , 0 . 4 , 0 . 6 , \overset { \cdot } { 0 . 7 } \}$ and compare it with the default setting 0.5 in UGO. As shown in Table $^ { 8 , }$ performance is essentially unchanged across this range: HOTA varies by less than 0.21 points around the default, and MOTA remains within ≈ 0.41 points of the default (with the largest drop observed at $\theta _ { 0 } = 0 . 8$ due to increased FP). This indicates that the consolidation labeling is insensitive to the exact strength of preference, provided that trackers are moderately favored.

Table 9: Single-target tracking performance on the DiDi dataset [36] measured using the following standard performance measures: tracking quality, accuracy and robustness.
<table><tr><td>Method</td><td>Quality</td><td>Accuracy</td><td>Robustness</td></tr><tr><td>SAM2.1</td><td>0.649</td><td>0.720</td><td>0.887</td></tr><tr><td>DAM4SAM</td><td>0.694 ①</td><td>0.727 ①</td><td>0.944 ①</td></tr><tr><td>UGO</td><td>0.685 ②</td><td>0.724②</td><td>0.932 ②</td></tr></table>

Overlap threshold $\tau _ { \mathrm { m d } } .$ The moderate threshold $\tau _ { \mathrm { m d } }$ is used for: (i) deciding whether a tracker mask is confirmed by any detector that gates trajectory verification, and (ii) initialization of new tracks. We evaluate $\bar { \tau } _ { \mathrm { m d } } \in \{ 0 . 4 5 , 0 . 5 , 0 . \bar { 5 } 5 , 0 . 6 5 , 0 . 7 \}$ . Lowering $\tau _ { \mathrm { m d } }$ relaxes confirmation, which increases false positives; this is reflected by the drop of 2% MOTA at $\tau _ { \mathrm { m d } } = 0 . 4 5$ . Increasing τ<sub>md</sub> makes confirmation stricter; at $\tau _ { \mathrm { m d } } = 0 . 7$ it starts to reduce MT and increase ML, rejecting correct trajectories as invalid. Overall, $\tau _ { \mathrm { m d } } = 0 . 6$ is the stable balance, yielding the least ML.

Tracker-collapse threshold $\tau _ { \mathrm { l o } } .$ After consolidation, labels whose final mask is inconsistent with the initial mask are removed. For trackers, a permissive threshold $\tau _ { \mathrm { l o } }$ is used to avoid terminating tracks whose masks were corrected by consolidation (e.g., distractor removal reduces overlap with the pre-consolidation mask). We evaluate $\tau _ { \mathrm { l o } } \in \{ 0 . 1 , 0 . 3 \}$ around the default $\tau _ { \mathrm { l o } } = 0 . 2 . \ \mathrm { A }$ smaller value (0.1) retains more corrected tracker masks, slightly increasing FP, whereas a larger value (0.3) prunes more aggressively but may remove some valid corrected tracks. The differences are minor (Table 8), confirming that consolidation already effectively suppresses unstable masks.

High overlap threshold $\tau _ { \mathrm { h i } } .$ . The high threshold $\tau _ { \mathrm { h i } }$ is used to identify high-agreement tracker– detector pairs for category-level memory updates and to detect pre/post-consolidation changes that trigger distractor-removed-driven DRM updates. We evaluate $\tau _ { \mathrm { h i } } \in \{ 0 . 7 5 , 0 . 8 5 , 0 . 9 5 \}$ around the default $\tau _ { \mathrm { h i } } = 0 . 8 .$ . Lowering $\tau _ { \mathrm { h i } }$ increases the number of candidate updates (more frequent memory refreshes) with a cost of a modest FN increase. Increasing $\tau _ { \mathrm { h i } }$ to 0.95 makes updates rare and reduces the opportunity to adapt memory, leading to a small drop in HOTA/MOTA/IDF1 (Table 8). The default $\tau _ { \mathrm { h i } } = 0 . 8$ provides a conservative update regime.

Overall robustness to thresholds. Across all ablations in Table 8, varying any single conservative gate leads to only marginal performance changes. The largest observed deviation from the default configuration is observed when we set $\tau _ { \mathrm { m d } } ~ = ~ 0 . 4 5$ , corresponding to merely 2% decrease in MOTA. Over the same sweeps, HOTA varies by at most less than 1%. These results demonstrate that the proposed thresholds function as robust gates rather than delicately tuned hyperparameters. Performance remains stable across a wide operating range, indicating that UGO does not depend on dataset-specific calibration or fine-grained threshold optimization.

## E Application to Single-target Tracking

Although UGO is primarily designed for tracking multiple objects within a specified category, it can be naturally adapted to the single-object tracking (SOT) setting, where only a single instance annotated in the first frame must be tracked throughout the sequence. To further test UGO’s cross-task generalization capability, we evaluate it on a recent challenging dataset for single-target tracking DiDi [36].

Only a minimal modification of the original algorithm is required: we disable instance termination for the initialized target. This adjustment ensures that the tracker remains active even during extended periods of occlusion or temporary disappearance - scenarios that occur more frequently in SOT than in gMOT - to enable target re-detection.

Table 9 compares UGO with the tracking foundation model SAM2.1 [33] and its recent state-of-theart SOT extension DAM4SAM [36]. UGO surpasses SAM2.1 by 5.5% and achieves performance comparable to DAM4SAM, with only 1.3% lower tracking quality. Considering that UGO is fundamentally a multi-target tracker, this result highlights its strong generalization capability across different tracking paradigms.

![](images/c3871d8b8abb0f708b6590fbacdfc670559bdc6bc3fd9a713ad2b1b97829a7a7.jpg)  
Figure 7: Qualitative examples of tracking with UGO. Each instance is represented by a colored mask and line, denoting its trajectory.

## F Qualitative Results

Additional qualitative results of multi-object tracking with UGO are presented in Figure 7. Each tracked instance is visualized using a semi-transparent colored mask, accompanied by a trajectory line indicating its motion over past frames. We show 16 video sequences from different datasets, displaying two representative frames per sequence. The examples span a wide range of object categories – including airplanes, balls, balloons, and various animals (e.g., birds, penguins, ducks, deer, bees, fish), as well as people – highlighting the strong cross-category generalization capability of UGO.
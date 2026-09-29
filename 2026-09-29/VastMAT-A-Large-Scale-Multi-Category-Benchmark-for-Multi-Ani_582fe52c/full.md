# VastMAT: A Large-Scale Multi-Category Benchmark for Multi-Animal Tracking

Zhizhen Li<sup>1,2,\*</sup>, Zan Wang<sup>3,\*</sup>, Huidong Peng<sup>2,\*</sup>, Bohan Tan<sup>4,\*</sup> , Shimin Shan<sup>1</sup>, Yu Liu<sup>1</sup>, Liang Peng<sup>2,†</sup>

<sup>1</sup>Dalian University of Technology <sup>2</sup> Wuhan University <sup>3</sup>University of North Texas

<sup>4</sup>The Hong Kong University of Science and Technology lizhizhen@mail.dlut.edu.cn, pengliang@whu.edu.cn <sup>\*</sup>Equal contribution. <sup>†</sup>Corresponding author.

## Abstract

Multi-animal tracking (MAT) supports the study of animal movement, behavior, and group interactions. However, general multi-object tracking (MOT) benchmarks primarily focus on pedestrians and vehicles, whereas dedicated MAT benchmarks remain limited in jointly supporting broad animal coverage, large-scale video data, and extensive within-video multi-instance association. To address this gap, we introduce VastMAT, which has four key characteristics: (1) Large scale. It comprises 2,947 videos with 1,002,562 annotated frames, totaling 27.85 hours. (2) Broad category coverage. These videos cover 337 animal categories with diverse morphologies and motion patterns. (3) Extensive instance annotations. It provides 3,663,248 bounding boxes and 22,883 identity trajectories—to our knowledge, the largest numbers of both among dedicated MAT benchmarks. (4) High-quality annotations. To ensure reliability, annotations undergo iterative expert review and correction, and quality is assessed through an independent reannotation audit. To systematically assess tracking performance and cross-category generalization, we establish Seen-category and category-disjoint Unseen-category protocols, and evaluate eight representative MOT methods under both protocols. Under these protocols, the highest baseline HOTA scores are 66.37% and 52.90%, respectively, highlighting the challenge of tracking unseen animals. To address the low-overlap association challenge revealed by our analysis, we propose Center-Distance-Augmented Association (CDA), a lightweight module that adaptively combines IoU with center similarity normalized by the boxes’ own scales. Without additional training, CDA improves TrackTrack’s HOTA by 1.58 and 1.31 percentage points under the two protocols, respectively. To facilitate further MAT research, we will publicly release our benchmark and code.

## 1 Introduction

Multi-animal tracking (MAT) is a branch of multi-object tracking (MOT) that localizes all visible animals in a video and maintains their identities across frames. The resulting identity-consistent trajectories support studies of animal movement, individual behavior, and group interactions, making reliable MAT valuable to biology, ecology, conservation, and animal husbandry. Yet research on tracking across diverse animal categories remains constrained by the scale, species diversity, and association density of existing benchmarks.

![](images/82b8f05825e89e18a867b8b951e3716c2ad1e64cd1ddd7f1a0a6873c63e8da8a.jpg)  
KITTI MOT17 MOT20 UAVDT-MOT ImageNet-Vid YT-VIS TA GMOT-40 GMOT-40-Anim. TAO-Anim. AnimalTrack SA-FARI VastMAT (Ours)  
Figure 1: Category and video counts of Vast-MAT and representative tracking datasets.

Evaluating tracking across diverse animals requires attention to both category variation and within-video identity association. Differences in morphology, motion, and imaging conditions demand benchmarks that test applicability across animal categories, including Unseen animals. Animal videos also range from a few independently moving individuals to dense interactions among similar-looking animals, where rapid displacement, non-rigid deformation, and occlusion complicate identity preservation. MAT benchmarks therefore need both continuous multi-instance identity annotations and complementary evaluations that distinguish these capabilities.

Existing MOT benchmarks primarily target pedestrians or vehicles, including MOT17, MOT20, KITTI, and UAVDT [9, 10, 13, 11]. DanceTrack and SportsMOT introduce similar appearances and complex motion but remain focused on humans [26, 7]. TAO, BURST, ImageNet-Vid, and YouTube-VIS broaden category and task coverage, yet animals constitute only part of these datasets, whose annotation frequencies, output formats, and evaluation protocols differ[8, 2, 24, 30]. A large general-purpose vocabulary thus does not necessarily provide broad coverage of animals, group interactions, or multi-instance association across animal categories.

Dedicated animal tracking benchmarks have advanced MAT but still struggle to combine video scale, category diversity, and multi-instance identity association. AnimalTrack provides 58 videos from 10 animal categories for association in dense animal groups, but its category and video coverage remain limited[34]. SA-FARI expands coverage to 11,609 camera-trap videos and 99 animal categories with pixel-level annotations, yet its average of approximately 1.4 identity trajectories per video differs from the association demands of dense group tracking[29]. MAT therefore needs a unified benchmark combining diverse animal categories, large-scale video data, and continuous multi-instance identity annotations.

To this end, we introduce VastMAT, comprising 337 animal categories, 2,947 videos, 1,002,562 annotated frames, 3,663,248 bounding boxes, and 22,883 identity trajectories over 27.85 hours. As Figure 2 illustrates, the dataset covers fish, birds, mammals, reptiles, amphibians, and other animals, with individual motion, category co-occurrence, and group interactions in underwater, terrestrial, aerial, and artificial environments. Videos vary in object scale, viewpoint, lighting, background, and camera motion, and present association challenges involving deformation, occlusion, and similar appearances. Figure 1 and Table 1 compare category coverage and dataset scale. With 7.76 trajectories per video on average, VastMAT combines broad category coverage with continuous multi-instance association to support tracking and Cross-category generalization evaluation.

![](images/45690773e655b3703c7f32371ed92e0a569824099eda33bf59913dd3737da489.jpg)  
Figure 2: Annotated examples across animal groups and scenes in VastMAT.

Table 1: Detailed comparison of VastMAT and representative tracking datasets. n/a indicates that the source does not provide a directly comparable value. GMOT-40-Anim. and TAO-Anim. are animal subsets recomputed from their parent datasets.
<table><tr><td>Benchmark</td><td></td><td>Videos Categories</td><td>Min. len.(s)</td><td>Avg. len.(s)</td><td>Max. len.(s)</td><td>Total len.(s)</td><td>Avg. tracks</td><td>Max. tracks</td><td>Total tracks</td><td>Frame rate</td><td>Ann. FPS</td><td>Total boxes</td><td>Total frames</td></tr><tr><td>KITTI [13]</td><td>50</td><td>5</td><td>n/a</td><td>10</td><td>n/a</td><td>498</td><td>52</td><td>n/a</td><td>2,600</td><td>30</td><td>10</td><td>80K</td><td>15K</td></tr><tr><td>MOT17 [9]</td><td>14</td><td>1</td><td>17</td><td>33</td><td>85</td><td>463</td><td>95</td><td>222</td><td>1,331</td><td>25</td><td>30</td><td>300K</td><td>11K</td></tr><tr><td>MOT20 [10]</td><td>8</td><td>1</td><td>17</td><td>66.8</td><td>133</td><td>535</td><td>479</td><td>1,211</td><td>3,833</td><td>25</td><td>30</td><td>2,102K</td><td>13K</td></tr><tr><td>UAVDT-MOT [11]</td><td>100</td><td>3</td><td>2.8</td><td>26.67</td><td>99</td><td>2,666.7</td><td>27.0</td><td>n/a</td><td>2,700</td><td>30</td><td>6</td><td>840K</td><td>40K</td></tr><tr><td>ImageNet-Vid [24]</td><td>5,354</td><td>30</td><td>0.2</td><td>12.1</td><td>219.7</td><td>64,547.9</td><td>n/a</td><td>n/a</td><td>n/a</td><td>25</td><td>25</td><td>n/a</td><td>1,614K</td></tr><tr><td>YT-VIS [30]</td><td>2,883</td><td>40</td><td>1</td><td>5.6</td><td>7.2</td><td>16,015.2</td><td>n/a</td><td>n/a</td><td>n/a</td><td>30</td><td>5</td><td>131K</td><td>480.5K</td></tr><tr><td>TAO [8]</td><td>2,907</td><td>833</td><td>n/a</td><td>36.8</td><td>n/a</td><td>106,978</td><td>6</td><td>10</td><td>17,287</td><td>30</td><td>1</td><td>333K</td><td>2,674K</td></tr><tr><td>GMOT-40 [3]</td><td>40</td><td>10</td><td>3</td><td>8.9</td><td>24.2</td><td>356.0</td><td>51</td><td>128</td><td>2,026</td><td>30</td><td>30</td><td>256K</td><td>9K</td></tr><tr><td>GMOT-40-Anim. [3]</td><td>12</td><td>3</td><td>3</td><td>7.1</td><td>24.2</td><td>85.5</td><td>70</td><td>128</td><td>837</td><td>30</td><td>30</td><td>63K</td><td>2.6K</td></tr><tr><td>TAO-Anim. [8]</td><td>39</td><td>39</td><td>1</td><td>22</td><td>93</td><td>859.0</td><td>4</td><td>10</td><td>250</td><td>30</td><td>1</td><td>3.4K</td><td>2.5K</td></tr><tr><td>AnimalTrack [34]</td><td>58</td><td>10</td><td>6.5</td><td>14.2</td><td>75.6</td><td>823.7</td><td>33</td><td>133</td><td>1,927</td><td>30</td><td>30</td><td>429K</td><td>24.7K</td></tr><tr><td>SA-FARI [29]</td><td>11,609</td><td>99</td><td>0.5</td><td>14.19</td><td>90</td><td>164,808</td><td>1.4</td><td>14</td><td>16,224</td><td>10-60</td><td>6</td><td>942K</td><td>988K</td></tr><tr><td>VastMAT(Ours)</td><td>2,947</td><td>337</td><td>4</td><td>34.02</td><td>100.5</td><td>100,256.2</td><td>7.76</td><td>152</td><td>22,883</td><td>10</td><td>10</td><td>3,663K</td><td>1,003K</td></tr></table>

We establish two protocols to evaluate tracking on new videos of Seen categories and generalization to Unseen animal categories. Each contains 2,632 training videos and 315 test videos. The Unseen-category protocol strictly separates 283 training categories from 54 test categories, assigning categories that co-occur in a video to the same split. We systematically evaluate eight representative MOT methods. DiffMOT achieves the highest baseline HOTA under both protocols, scoring 66.37 and 52.90, while the corresponding YOLOX-X detectors achieve animal detection AP scores of 72.34 and 54.07. Together with animal-group and video-level analyses, these results show that evaluating tracking across diverse animals requires examining object detection, identity association, and performance variation across scenes.

We further introduce Center-Distance-Augmented Association (CDA), a lightweight, training-free module that replaces the geometric matching score in existing trackers at inference time. CDA combines IoU with center similarity normalized by the mean box diagonal and adapts their weights to the number of high-confidence detections. It improves TrackTrack’s HOTA by 1.58 and 1.31 percentage points under the Seen- and Unseen-category protocols, respectively.

In summary, Our main contributions are as follows: ♠ We establish VastMAT, a large-scale MAT benchmark with 2,947 videos and 337 animal categories, and validate box-level and within-clip identity consistency through an independent reannotation audit. ♥ We define Seen-category and category-disjoint Unseen-category protocols, systematically evaluate eight MOT methods, and analyze tracking through detection, animal-group, and video-level results. ♣ We propose lightweight CDA to improve identity association through normalization by the boxes’ own scales and adaptive similarity fusion, and validate its effectiveness through geometric comparisons, fusionweight analyses, and cross-tracker experiments.

## 2 Related Work

Multi-object tracking benchmarks. MOT17, MOT20, KITTI, and UAVDT-MOT provide pedestrian and vehicle tracking evaluations [9, 10, 13, 11]. DanceTrack and SportsMOT introduce complex motion and similar appearances but remain humanfocused [26, 7]. ImageNet-Vid, YouTube-VIS, TAO, and BURST broaden category coverage across detection, segmentation, and tracking [24, 30, 8, 2], while GMOT-40 evaluates exemplar-guided generic tracking [3]. Their task definitions, annotation frequencies, and animal coverage differ, leaving limited support for continuous identity association across diverse animal groups.

Animal vision and tracking benchmarks. Animal Kingdom, MammalNet, and ChimpACT support animal behavior analysis [20, 6, 19]; AP-10K and APT-36K provide cross-species pose annotations [33, 31]; FishNet supports both fish recognition and detection, while Wildlife-71 and WildlifeReID-10k address individual reidentification [16, 15, 1]. Together, These resources primarily target behavior, pose, recognition, or image-level identity. Among MAT benchmarks, AnimalTrack emphasizes dense groups but has limited category and video coverage [34]. SA-FARI broadens coverage with pixel-level annotations but provides relatively sparse withinvideo identities [29]. VastMAT combines 337 animal categories and 2,947 videos with dense frame-by-frame boxes and persistent identities for multi-instance association and cross-category evaluation.

Multi-object tracking and geometric association. Tracking-by-detection typically combines motion and appearance cues to establish identity correspondences. ByteTrack associates high- and low-confidence detections to recover overlooked objects[36], Diff-MOT predicts nonlinear motion using diffusion models[18], and TrackTrack improves candidate matching and track initialization from a track-centric perspective[25]. Joint approaches, such as TransTrack and MOTIP, maintain identities through track queries and temporal identity prediction[27, 12]. In explicit candidate matching, geometric scores measure spatial compatibility between predicted and detected boxes. GIoU, DIoU, CIoU, and EIoU extend IoU using enclosing regions, center distance, or shape information[22, 38, 35]. To address low-overlap association caused by animal motion and deformation, CDA combines IoU with linear center similarity normalized by the boxes’ own scales and adjusts their weights to the detection count, requiring no additional training.

## 3 The VastMAT Dataset

## 3.1 Design Principles

VastMAT provides a unified, large-scale resource for multi-category MAT, animal detection, and Cross-category generalization. Its construction follows four principles. (i) Broad animal coverage. We aim to cover at least 300 animal categories with diverse morphologies and motion patterns across underwater, terrestrial, and aerial environments. (ii) Large scale. We expand video, annotated-frame, bounding-box, and trajectory counts to capture diverse imaging conditions and support training and evaluation. (iii) Extensive identity association. We retain videos with multiple animals, co-occurring categories, and dense interactions to test identity preservation under occlusion, entry and exit, and similar appearances. (iv) Reliable annotations. We standardize category labels, bounding boxes, and identity assignment, combining model-assisted annotation with repeated manual review and independent reannotation to validate consistency.

## 3.2 Data Collection

We curate animal names, synonyms, and potentially confusable categories, selecting 337 animal categories from more than 400 candidates (Appendix B.1). Experts, including doctoral and master’s students working on related topics, verify each category’s suitability for tracking. We then search YouTube for Creative Commons-licensed videos of each category, collecting more than 5,000 candidate clips. After reviewing their suitability for visual tracking, we retain 2947 sequences and sample them uniformly at 10 FPS. Figure 8 in the appendix shows the long-tailed distributions of videos, trajectories, and boxes across categories, reflecting their differing frequencies in public video sources. Appendix F details provenance, maintenance, and responsible use.

## 3.3 Annotation and Quality Validation

We annotate animal boxes, categories, and within-video identities using X-AnyLabeling 3.3.7 and SAM3 [28, 5]. Each video is assigned to one annotator, who initializes animal regions, propagates them with SAM3, and corrects missed objects, drift, and identity errors. Boxes cover visible animal regions; fully occluded animals are omitted. Reappearing animals retain their IDs when identity is verifiable and receive new IDs otherwise. Second, experts review object coverage, localization, categories, and identity continuity. Third, annotations without unanimous approval from the two- to three-expert review team are returned to the original annotator for correction. Review and correction are repeated over multiple rounds. Finally, masks are converted to AnimalTrackcompatible bounding boxes. Table 10 in the appendix defines the format, and Figure 2 shows examples.

![](images/1ef32545b0dc9a1eb23b6c64f7bd4646b1c4fa0ce81b08e870a8957d6e9198cf.jpg)  
(a) Target Motion

![](images/b1c0bb74cfded3bc6917bfa122be01489a81b8cb29c9a8b9daa7b38e27c2f816.jpg)

![](images/5f46a24bc1cfd3fc2d81792f16cb738bb73ed4fc1e9d0fb4aeb20cf7593c64e5.jpg)  
(b) Relative Area to Initial Object Box (c) Relative Aspect Ratio to Initial Object Box (d) IoU on Adjacent Frames

![](images/9dd53f1ab2127f9ae2ee131e3700ec1548ec64b6d7df7a1d9dce8b5e34ff2937.jpg)  
Figure 3: Motion and box-geometry distributions across tracking datasets.

We audit annotation quality by randomly sampling 48 of the 2947 videos across all six animal groups and selecting 2 non-overlapping, consecutive 20-frame clips per video, totaling 1,920 frames. Annotators trained under the same guidelines, but uninvolved in the original annotation and without access to it, independently reannotate these frames, followed by the same repeated review and correction stages. We then match the 1920 reannotated frames one-to-one with the originals. At IoU ≥ 0.5, boxmatching F1 is 96.03%, and mean IoU among matched boxes is 0.9555. All 7,632 matched box pairs agree in category. To assess temporal identity consistency, we check whether each original identity matched in at least two frames of a 20-frame clip always maps to the same reannotated identity. Of 427 eligible clip–identity pairs, 424 are consistent, yielding 99.30% within-clip identity consistency and a mean longest consecutive consistent span of 18.49 frames. These results support the reliability of object coverage, localization, and within-clip identities. Appendix C.2 provides full metrics and group-wise results.

## 3.4 Dataset Statistics and Characteristics

VastMAT provides 27.85 hours of annotations across 337 animal categories, averaging 7.76 trajectories per video and 3.65 objects per frame. Average trajectory and box counts in Table 3 are both computed per video. The 337 categories comprise 82 mammals, 85 birds, 117 fish, 4 amphibians, 7 reptiles, and 42 other animals; Appendix B.1 lists all categories and their protocol coverage.

VastMAT includes 607 videos with more than 10 identity trajectories and 827 multi-category videos, covering sparse individual motion, multi-instance interactions, and category co-occurrence. Appendix B provides further distribution and co-occurrence analyses, while Figure 11 illustrates identity discrimination

Table 2: IoU statistics by normalized displacement (Protocol 2). Event percentages may not sum to 100% due to rounding.
<table><tr><td>Displ.</td><td>Events</td><td>IoU=0</td><td>&lt; 0.1</td><td>&lt; 0.3</td><td>Median</td></tr><tr><td>≤ 0.25</td><td>95.47%</td><td>0.004%</td><td>0.020%</td><td>0.304%</td><td>0.894</td></tr><tr><td>0.25–0.5</td><td>3.25%</td><td>1.49%</td><td>5.58%</td><td>51.19%</td><td>0.296</td></tr><tr><td>0.5-1.0</td><td>0.94%</td><td>29.49%</td><td>71.62%</td><td>100.0%</td><td>0.050</td></tr><tr><td>&gt; 1.0</td><td>0.35%</td><td>100.0%</td><td>100.0%</td><td>100.0%</td><td>0.000</td></tr></table>

challenges involving visually similar categories.

Table 3: Split statistics for the two VastMAT evaluation protocols.
<table><tr><td>Split</td><td>Videos</td><td>Classes</td><td>Min. (s)</td><td>Avg. (s)</td><td>Max. (s)</td><td>Frames</td><td>Tracks</td><td>Boxes</td><td>Avg. tracks</td><td>Avg. boxes</td></tr><tr><td>Exp1 Train</td><td>2,632</td><td>337</td><td>4.0</td><td>34.556</td><td>100.5</td><td>909,526</td><td>20,782</td><td>3,389,474</td><td>7.90</td><td>1,287.79</td></tr><tr><td>Exp1 Test</td><td>315</td><td>316</td><td>5.1</td><td>29.535</td><td>100.0</td><td>93,036</td><td>2,101</td><td>273,774</td><td>6.67</td><td>869.12</td></tr><tr><td>Exp2 Train/Seen</td><td>2,632</td><td>283</td><td>4.0</td><td>34.554</td><td>100.5</td><td>909,451</td><td>20,734</td><td>3,351,099</td><td>7.88</td><td>1,273.21</td></tr><tr><td>Exp2 Test/Unseen</td><td>315</td><td>54</td><td>7.4</td><td>29.559</td><td>100.0</td><td>93,111</td><td>2,149</td><td>312,149</td><td>6.82</td><td>990.95</td></tr></table>

Figure 3 compares motion, relative area, relative aspect ratio, and consecutive-frame IoU for the same identity. Motion and IoU are computed between consecutive annotated frames, while area and aspect ratio are normalized to the first-frame box. VastMAT has a median Target Motion of 0.032, compared with 0.015, 0.008, and 0.010 for AnimalTrack, MOT17, and MOT20. Its 95th percentile is 0.284, approximately 3.0, 3.7, and 10.6 times the respective values. The fraction with IoU< 0.8 is 33.4%, compared with 8.6% for AnimalTrack. Together with broader area and aspect-ratio distributions, these results indicate greater temporal geometric variation in VastMAT.

Protocol 2 contains 278,390 valid GT box pairs with the same identity and a frameindex gap of at most 3. We define normalized displacement as center distance divided by the mean box diagonal. As Table 2 shows, median IoU is 0.894 for displacements up to 0.25, falling to 0.296 and 0.050 for 0.25–0.5 and 0.5–1.0, respectively. Zero overlap accounts for only 0.68% of all events; low overlap is concentrated in the large-displacement ranges.

## 3.5 Data Splits and Evaluation Protocols

Data splits. We refer to the data partitioning principle in [32]. Both protocols contain 2,632 training videos and 315 test videos but use different splits (See Table 3). All Protocol 1 test categories appear in the training set, whereas Protocol 2 strictly separates 283 training categories from 54 test categories. To keep co-occurring categories on the same side of the split, we assign entire connected components of the category co-occurrence graph, placing the largest component of 266 categories in training.

Evaluation protocols. For both detection and tracking, all valid animal objects are mapped to a single animal foreground class. Table 3 summarizes the splits. Protocol 2 test categories cover mammals, birds, fish, and other animals; all amphibians and reptiles remain in training and are evaluated only under Protocol 1. Table 9 in the appendix provides the full composition. Because training-category coverage and test-video composition both differ between protocols, cross-protocol gaps reflect their combined effects.

## 4 CDA: A Lightweight Association Module

## 4.1 Motivation and Overview

CDA is a lightweight, training-free module for geometric association at inference time. Rapid motion and deformation can produce low-overlap candidates(See Table 2). Figure 4 illustrates the zero-overlap case: nearby boxes retain positive CenterSim when their center distance is smaller than the mean box diagonal, yielding a positive fused score for $\alpha < 1$ . CDA combines this cue with IoU, normalizes center distance by object scale, and adapts fusion weights to detection count.

## 4.2 CDA Similarity Score

Given a track’s predicted box a and a candidate detection box b in the current frame, CDA defines the fused similarity as

$$
S ( a , b ) = \alpha \mathrm { I o U } ( a , b ) + ( 1 - \alpha ) \mathrm { C e n t e r S i m } ( a , b ) ,\tag{1}
$$

where $\alpha$ controls the relative contributions of region overlap and center proximity. Center similarity is defined as

$$
\operatorname { C e n t e r S i m } ( a , b ) = \operatorname { c l i p } \left( 1 - { \frac { d _ { \operatorname { c e n t e r } } } { ( \mathrm { d i a g } _ { a } + \mathrm { d i a g } _ { b } ) / 2 } } , 0 , 1 \right) .\tag{2}
$$

Here, $d _ { \mathrm { c e n t e r } }$ is the Euclidean distance between box centers, and diag and dia $\mathbf { g } _ { b }$ are their diagonal lengths. CenterSim equals 1 when the centers coincide and falls to 0 when their distance reaches the mean diagonal length. Thus, some spatially close but non-overlapping candidates retain nonzero similarity, supplying positional information when overlap is insufficient. When pose changes shift box boundaries while the centers remain

![](images/5da6bf52822298253d5c8c6889a319ddac846786c4ac754452bf03f2de70b0d2.jpg)  
(a) IoU only

![](images/9306a4c26a4ed7fdedec1cc9282c6747707c49a944a77bd92fc6c599f399e49b.jpg)  
(b) CDA  
Figure 4: Scoring non-overlapping candidates with (a) IoU and (b) CDA.

close, CenterSim complements overlap; IoU retains region-level spatial constraints that help distinguish competing candidates with similar center distances.

## 4.3 Box-Scale Normalization

The same pixel displacement has different implications for animals of different sizes. Equation (2) normalizes center distance by the mean box diagonal, making CenterSim invariant to uniform scaling of boxes and their positions. Averaging incorporates predicted and detected scales, so the distance reference is not determined solely by the smaller box. CDA draws on DIoU’s use of center distance to complement overlap[38], but uses the boxes’ scales and linear decay. Unlike the diagonal of the smallest enclosing rectangle, this denominator does not grow as the enclosing region expands with center separation, preserving a distance scale determined by object size. Section 5.3 compares geometric designs; Appendix E.1 defines normalized DIoU and candidate gating.

## 4.4 Detection-Count-Adaptive Fusion

With few animals, center proximity helps compensate for low overlap; dense scenes require stronger overlap constraints to distinguish competing objects. We approximate

group size using the number of high-confidence detections n in the current frame and define

$$
\alpha ( n ) = \mathrm { c l i p } \left( 0 . 5 + k ( n - N _ { \mathrm { r e f } } ) , 0 . 1 , 0 . 9 \right) .\tag{3}
$$

The defaults are $k = 0 . 0 5$ and $N _ { \mathrm { r e f } } = 1 0$ . The IoU weight is 0.1, 0.5, and 0.9 for $n \leq 2 .$ $n = 1 0 ,$ , and $n \geq 1 8$ , respectively. Restricting it to [0.1, 0.9] keeps both geometric cues active in the adaptive configuration. The rule increases the IoU contribution as detection count grows and favors center similarity when fewer objects are present.

CDA replaces the base tracker’s geometric association term: it computes S from predicted and current-frame detection boxes and uses 1 − S as the geometric cost. In TrackTrack, this cost is combined with the existing appearance, confidence, and direction terms; the remaining matching and track-update procedures are unchanged. CDA and the DIoU control use identical normalized-DIoU gating to compare scores for admitted candidates; Section 5.4 discusses changes in candidate acceptance in ByteTrack. CDA requires no additional training. Default parameters are selected on a development set drawn from the training split and fixed during testing. Appendix E provides the full association cost and implementation details.

## 5 Experiments

Evaluation metrics. We use TrackEval to compute HOTA, CLEAR, and Identity metrics [17, 4, 23]. HOTA jointly evaluates detection and association through DetA and AssA, which we also report in diagnostic experiments. MOTA aggregates false negatives, false positives, and identity switches, while IDF1 measures identity matching consistency. We additionally report IDP/IDR, MT/PT/ML, FP/FN, identity switches (IDs/IDSW), and track fragmentations (FM). Detection is evaluated using COCO-style AP, AP<sub>50</sub>, and $\mathsf { A P 7 5 }$ , with AP averaged over IoU thresholds from 0.50–0.95 in steps of 0.05.

## 5.1 Evaluated Methods

We evaluate the eight trackers in Table 4 under both protocols. They span motion matching, appearance representation, track queries, and temporal identity modeling, enabling comparison of these mechanisms in animal scenes. ByteTrack, DiffMOT, and TrackTrack use YOLOX-X trained on the corresponding protocol’s training split, followed by their respective association pipelines. End-to-end methods train their joint detection and association frameworks on the corresponding training split.

## 5.2 Evaluation Results

Overall performance. Table 4 shows that DiffMOT achieves the highest HOTA under both protocols, at 66.37 and 52.90, followed by TrackTrack at 65.10 and 52.61. TrackTrack also produces the fewest track fragmentations in both. All eight methods score lower on Protocol 2, with HOTA gaps of approximately 5.3–13.5 percentage points, indicating a substantial performance gap under the Unseen-category evaluation setting. MOTIP and TCEI produce more mostly tracked trajectories in Protocol 1 but

Table 4: Overall comparison of tracking algorithms on VastMAT under Protocols 1 and 2. Within each protocol, the best two results for each metric are highlighted in red and blue, respectively.
<table><tr><td></td><td>Tracker HOTA</td><td>MOTA</td><td>IDF1</td><td>IDP</td><td>IDR</td><td>MT</td><td>PT</td><td>ML↓</td><td>FP↓</td><td>FN↓</td><td>IDs↓</td><td>FM↓</td></tr><tr><td colspan="11">Protocol 1</td></tr><tr><td>TransTrack [27] [arXiv 2020]</td><td>41.33</td><td>39.05</td><td>44.33</td><td>54.84</td><td>37.19</td><td>519</td><td>1029</td><td>553</td><td>37200</td><td>125298</td><td>4363</td><td>11599</td></tr><tr><td>QDTrack [21] [CVPR 2021]</td><td>53.80</td><td>52.79</td><td>60.01</td><td>74.92</td><td>50.05</td><td>623</td><td>886</td><td>592</td><td>18419</td><td>109312</td><td>1523</td><td>8051</td></tr><tr><td>FairMOT [37] [IJCV 2021]</td><td>41.31</td><td>47.23</td><td>43.93</td><td>53.43</td><td>37.30</td><td>513</td><td>1038</td><td>550</td><td>24279</td><td>106904</td><td>13300</td><td>10728</td></tr><tr><td>ByteTrack [36] [ECCV 2022]</td><td>56.59</td><td>62.91</td><td>63.37</td><td>69.93</td><td>57.93</td><td>846</td><td>905</td><td>350</td><td>25824</td><td>72789</td><td>2934</td><td>5903</td></tr><tr><td>DiffMOT [18] [CVPR 2024]</td><td>66.37</td><td>68.38</td><td>69.21</td><td>77.44</td><td>62.56</td><td>1043</td><td>710</td><td>348</td><td>15915</td><td>68519</td><td>2142</td><td>5054</td></tr><tr><td>MOTIP [12] [CVPR 2025]</td><td>61.71</td><td>58.55</td><td>65.96</td><td>67.63</td><td>64.36</td><td>1179</td><td>700</td><td>222</td><td>48268</td><td>61505</td><td>3712</td><td>7586</td></tr><tr><td>TrackTrack [25] [CVPR 2025]</td><td>65.10</td><td>67.45</td><td>67.87</td><td>75.88</td><td>61.39</td><td>934</td><td>806</td><td>361</td><td>17428</td><td>69720</td><td>1962</td><td>4200</td></tr><tr><td>TCEI [14] [CVPR 2026]</td><td>61.89</td><td>58.95</td><td>66.28</td><td>67.72</td><td>64.91</td><td>1176</td><td>706</td><td>219</td><td>48668</td><td>60034</td><td>3672</td><td>7629</td></tr><tr><td colspan="11">Protocol 2</td></tr><tr><td>TransTrack [27] [arXiv 2020]</td><td>36.00</td><td>28.25</td><td>38.49</td><td>53.33</td><td>30.11</td><td>357</td><td>1116</td><td>676</td><td>41788</td><td>177662</td><td>4504</td><td>14274</td></tr><tr><td>QDTrack [21] [CVPR 2021]</td><td>41.64</td><td>35.23</td><td>45.56</td><td>69.83</td><td>33.80</td><td>371</td><td>942</td><td>836</td><td>19569</td><td>180682</td><td>1928</td><td>9606</td></tr><tr><td>FairMOT [37] [IJCV 2021]</td><td>32.85</td><td>34.33</td><td>34.82</td><td>49.60</td><td>26.83</td><td>335</td><td>1104</td><td>710</td><td>24095</td><td>167431</td><td>13474</td><td>12348</td></tr><tr><td>ByteTrack [36] [ECCV 2022]</td><td>45.17</td><td>45.76</td><td>50.37</td><td>64.86</td><td>41.17</td><td>604</td><td>936</td><td>609</td><td>26083</td><td>140109</td><td>3115</td><td>6727</td></tr><tr><td>DiffMOT [18] [CVPR 2024]</td><td>52.90</td><td>49.70</td><td>55.23</td><td>73.13</td><td>44.36</td><td>754</td><td>781</td><td>614</td><td>15851</td><td>138622</td><td>2545</td><td>6682</td></tr><tr><td>MOTIP [12] [CVPR 2025]</td><td>51.11</td><td>47.60</td><td>55.48</td><td>67.13</td><td>47.27</td><td>787</td><td>884</td><td>478</td><td>33595</td><td>125932</td><td>4033</td><td>11350</td></tr><tr><td>TrackTrack [25] [CVPR 2025]</td><td>52.61</td><td>49.62</td><td>54.92</td><td>71.80</td><td>44.47</td><td>689</td><td>830</td><td>630</td><td>18071</td><td>136878</td><td>2309</td><td>5329</td></tr><tr><td>TCEI [14] [CVPR 2026]</td><td>52.21</td><td>47.43</td><td>56.27</td><td>64.43</td><td>49.95</td><td>921</td><td>791</td><td>437</td><td>44604</td><td>114783</td><td>4706</td><td>10683</td></tr></table>

also more false positives. In Protocol 2, TCEI achieves the highest IDF1 of 56.27, but its HOTA remains below DiffMOT and TrackTrack, underscoring the need to consider detection, identity association, and error statistics together.

Animal-group performance. YOLOX-X achieves 72.34 and 54.07 AP under Protocols 1 and 2 (See Table 5), so cross-protocol tracking results should be interpreted alongside detector performance. In Protocol 2, birds achieve 74.67 AP and 70.59 mean Diff-MOT HOTA, versus 23.08 and 25.36 for other animals, revealing substantial variation across groups. Protocol 2 has no amphibian or reptile test categories (–). Appendix Tables 14, 15, and 16 provide further detection, recall, and tracking results.

Table 5: Overall detection metrics and group-wise AP (%) for YOLOX-X.
<table><tr><td>Metric / Group</td><td>P1</td><td>P2</td></tr><tr><td>Overall AP</td><td>72.34</td><td>54.07</td></tr><tr><td>Overall  $\mathrm { A P } _ { 5 0 }$ </td><td>84.22</td><td>67.47</td></tr><tr><td>Overall AP75</td><td>75.94</td><td>56.63</td></tr><tr><td>Mammals AP</td><td>74.38</td><td>54.62</td></tr><tr><td>Fish AP</td><td>72.68</td><td>68.36</td></tr><tr><td>Birds AP</td><td>75.77</td><td>74.67</td></tr><tr><td>Amphibians AP</td><td>78.33</td><td></td></tr><tr><td>Reptiles AP</td><td>25.24</td><td></td></tr><tr><td>Others AP</td><td>45.98</td><td>23.08</td></tr></table>

Video-level difficulty. Within each protocol, we split the 315 test videos equally into Easy, Medium, and

Hard groups by mean per-video HOTA across all eight baselines. DiffMOT leads all groups except Protocol 2 Hard, where TCEI and MOTIP achieve 29.08 and 28.27 HOTA, exceeding DiffMOT’s 24.73 (Appendix Table 18). This ranking change highlights differences obscured by aggregate scores. Table 17 and Figure 12 in the appendix report the full distributions.

Qualitative comparison. Figure 5 compares eight trackers on Flamingo-7 and Anthias-4, illustrating localization differences under overlapping animals and pose changes, and incomplete coverage of small underwater targets. Appendix A provides additional

![](images/d65b931013dc70fc2329c5d55c541225623a5eed17a5c3b005b0844bdf56813c.jpg)  
Figure 5: Qualitative comparison of eight trackers on Protocol 1.

comparisons.

## 5.3 CDA Ablation and Design Analysis

We analyze CDA with TrackTrack on Protocol 2, keeping cached detections, appearance features, track management, and post-processing identical. Original TrackTrack retains its scoring and gating, while DIoU, CDA, and the other geometric controls share normalized-DIoU gating at a threshold of 0.10 to compare association scores.

Main CDA ablation on Protocol 2. In Table 6, adaptive CDA (❻) improves HOTA from 52.61 to 53.92 over original TrackTrack (❶), with AssA and IDF1 gains of 1.41 and 2.14 percentage points. Compared with DIoU under identical gating (❷), HOTA and AssA improve by 1.07 and 2.24 points, while DetA changes

Table 6: Main CDA ablation on Protocol 2.
<table><tr><td></td><td>Configuration</td><td>HOTA</td><td>AssA</td><td>IDF1</td><td>IDSW↓</td></tr><tr><td>①</td><td>TrackTrack</td><td>52.61</td><td>56.97</td><td>54.92</td><td>2309</td></tr><tr><td>②</td><td>TrackTrack (DIoU)</td><td>52.85</td><td>56.14</td><td>55.82</td><td>2133</td></tr><tr><td>③ 4</td><td>CDA (α = 0)</td><td>53.83 53.83</td><td>58.29</td><td>57.03</td><td>2313</td></tr><tr><td>5</td><td>CDA (α = 0.25)</td><td></td><td>58.19</td><td>56.89</td><td>2054</td></tr><tr><td></td><td>CDA (α = 1)</td><td>53.56</td><td>58.12</td><td>56.28</td><td>2082</td></tr><tr><td>6</td><td>CDA-density (fwd)</td><td>53.92</td><td>58.38</td><td>57.06</td><td>2166</td></tr><tr><td>7</td><td>CDA-density (rev)</td><td>53.54</td><td>57.85</td><td>56.26</td><td>1973</td></tr><tr><td>8</td><td>CDA (α = 0.3824)</td><td>53.52</td><td>57.62</td><td>56.40</td><td>2048</td></tr></table>

only from 49.96 to 50.01, indicating that the gains primarily reflect better association. Pure center scoring (❸) and fixed fusion (❹) both achieve 53.83 HOTA, but incorporating IoU reduces IDSW from 2,313 to 2,054, showing that overlap information can reduce identity switches while preserving HOTA.

To examine the scale reference for center distance, we fix α = 0.5 and vary only the normalization denominator. Mean-diagonal normalization achieves 53.80 HOTA, exceeding enclosing-box normalization at

Table 7: CDA versus IoU variants on Protocol 2.
<table><tr><td>Configuration</td><td></td><td>HOTA</td><td>AssA</td><td>IDF1</td><td>IDSW↓</td></tr><tr><td>①</td><td>TrackTrack</td><td>52.61</td><td>56.97</td><td>54.92</td><td>2309</td></tr><tr><td>②</td><td>GIoU</td><td>53.50</td><td>57.51</td><td>56.56</td><td>2012</td></tr><tr><td>③</td><td>CIoU</td><td>52.96</td><td>56.40</td><td>56.00</td><td>2115</td></tr><tr><td>4</td><td>EIoU</td><td>53.61</td><td>57.97</td><td>56.39</td><td>1960</td></tr><tr><td>⑤</td><td>CDA (α = 0.25)</td><td>53.83</td><td>58.19</td><td>56.89</td><td>2054</td></tr></table>

53.18 and approaching minimum-diagonal normalization at 53.82 (See Table 19 in the appendix), supporting a distance reference based on the boxes’ own scales. Comparisons with standard IoU variants further assess the complete geometric score[22, 38, 35]. In Table 7, CDA (❺) achieves the highest HOTA, AssA, and IDF1, whereas EIoU (❹) yields the fewest IDSW. These rankings show that identity-switch counts and overall identity consistency capture different aspects of performance, motivating evaluation with complementary association metrics.

Adaptive fusion. We compare fixed weights with forward and reverse density rules to assess the effect of adapting weights to detection count. In Table 6, the forward rule, fwd (❻), achieves 53.92 HOTA, exceeding 53.83 for fixed $\pmb { \alpha = 0 . 2 5 \left( \pmb { \Theta } \right) }$ , the reverse rule, rev (❼), and the fixed reference $\alpha = 0 . 3 8 2 4$ averaged over diagnostic events (❽). The forward–reverse difference shows that the direction of weight adaptation matters, emphasizing IoU at higher detection counts and center similarity at lower counts. Nine density-parameter configurations outperform TrackTrack (See Table 23 in the appendix). Appendix E provides complete weight, post-processing, and candidate-pair diagnostics.

## 5.4 CDA Across Trackers and Protocols

Cross-tracker results. With fixed $\alpha = 0 . 2 5 ,$ , CDA improves HOTA by 1.22 and 1.36 percentage points for TrackTrack (❶ and ❷) and ByteTrack (❸ and ❹), respectively (See Table 8). Their IDF1 scores increase from 54.92 and 50.37 to 56.89 and 52.96, accompanied by higher AssA and fewer IDSW. DiffMOT (❺ and ❻) gains slightly in HOTA and IDF1 without corresponding improvements in AssA or IDSW; its weight sensitivity is analyzed below.

Candidate acceptance in Byte-Track. ByteTrack’s geometric cost $1 - \mathrm { I o U }$ and matching threshold of 0.6 imply $\mathrm { I o U } \geq 0 . 4$ . With CDA, this becomes $S \ge 0 . 4$ , admitting some low-overlap candidates. FN and IDSW decrease by 2,839 and 823, re-

<table><tr><td colspan="5">Table 8: Cross-tracker results on Protocol 2.</td></tr><tr><td></td><td>Configuration</td><td>HOTA</td><td>AssA</td><td>IDF1 IDSW↓</td></tr><tr><td>0</td><td>TrackTrack</td><td>52.61</td><td>56.97</td><td>54.92 2309</td></tr><tr><td>②</td><td>TrackTrack + CDA</td><td>53.83</td><td>58.19 56.89</td><td>2054</td></tr><tr><td>③</td><td>ByteTrack</td><td>45.17</td><td>48.55 50.37</td><td>3115</td></tr><tr><td>④</td><td>ByteTrack + CDA</td><td>46.53</td><td>51.27 52.96</td><td>2292</td></tr><tr><td>⑤</td><td>DiffMOT</td><td>52.90</td><td>57.61 55.23</td><td>2545</td></tr><tr><td>6</td><td>DiffMOT + CDA</td><td>52.93</td><td>57.30</td><td>55.54 2711</td></tr></table>

spectively, while FP increases by 6,743 and MOTA falls by 0.99 percentage points. Thus, the change in candidate acceptance improves object coverage and identity association but also introduces more false positives.

Tracker-specific fusion weights. Figure 13 shows that TrackTrack achieves higher HOTA when center similarity receives greater weight. Among the tested weights, DiffMOT performs best at $\alpha = 0 . 5$ , reaching 53.53 HOTA, a gain of 0.63 percentage points (See Table 21 in the appendix). These results indicate that suitable fusion weights depend on the base tracker.

Results on Seen categories. On Protocol 1, fixed and adaptive CDA improve Track-Track’s HOTA from 65.10 to 66.66 and 66.68, respectively; the adaptive variant also reduces IDSW from 1,962 to 1,652. Together with Protocol 2, these results show that CDA improves TrackTrack on both Seen and Unseen categories. Table 25 in the appendix reports the full metrics.

Inference cost. With detections and appearance features cached, tracking 93,111 frames takes 129.32s compared with 128.97s for the baseline, an increase of only 0.27%. Table 26 in the appendix provides a similarity-computation microbenchmark.

## 6 Conclusion

We introduce VastMAT, comprising 337 animal categories and 2947 videos with continuous multi-instance identity annotations and Seen- and Unseen-category evaluation protocols. Evaluating eight MOT methods reveals performance variation across animal groups and videos. We further propose CDA, which requires no additional training and improves TrackTrack’s HOTA by 1.58 and 1.31 percentage points under the two protocols. VastMAT provides a unified benchmark for detection, identity association, and Cross-category generalization across diverse animals.

## Limitations

VastMAT inherits category imbalance and scene biases from publicly available online videos, so benchmark results may not fully reflect performance on underrepresented animals and environments. Moreover, CDA relies on the base tracker’s detection candidates and motion predictions, and its gains vary across trackers and fusion settings.

## AI Use Statement

We used AI tools to assist with writing, language refinement, related-work discovery, and drafting portions of the manuscript. All AI-assisted content was reviewed and verified by the authors, who take full responsibility for the final paper.

## Ethics Statement

VastMAT is constructed from existing public videos, whose rights remain with their respective holders. Release materials will comply with individually verified licenses or permissions and specify separate terms for code, annotations, and third-party videos; the dataset license does not extend rights to third-party content. The project will provide a channel for copyright and privacy concerns. Verified issues will lead to access restrictions, corrections, or removal, with effects on the data and evaluation scope documented.

## Reproducibility Statement

At release, we will provide the category dictionary, split lists for both protocols and the training-side development set, annotation format and data preparation tools, CDA implementation, and unified evaluation code. We will also release baseline training and inference configurations, pretraining sources, predictions, and metric-generation scripts. Annotation-quality materials will include audit lists, independent reannotations, and scripts for box-level and within-clip identity consistency. Each result will be tied to explicit data, annotation, and evaluation versions. Appendix F describes media access policies.

## References

[1] Luka´s Adam, Vojtˇ echˇ Cerm<sup>ˇ</sup> ak, Kostas Papafitsoros, and Lukas Picek.´ WildlifeReID-10k: Wildlife re-identification dataset with 10k individual animals.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 2090–2100, 2025.

[2] Ali Athar, Jonathon Luiten, Paul Voigtlaender, Tarasha Khurana, Achal Dave, Bastian Leibe, and Deva Ramanan. BURST: A benchmark for unifying object recognition, segmentation and tracking in video. In Proceedings ofthe IEEE/CVF Winter Conference on Applications ofComputer Vision, pages 1674–1683, 2023.

[3] Hexin Bai, Wensheng Cheng, Peng Chu, Juehuan Liu, Kai Zhang, and Haibin Ling. GMOT-40: A benchmark for generic multiple object tracking. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6719–6728, 2021.

[4] Keni Bernardin and Rainer Stiefelhagen. Evaluating multiple object tracking performance: The CLEAR MOT metrics. EURASIP Journal on Image and Video Processing, page 246309, 2008.

[5] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, et al. SAM 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719, 2025.

[6] Jun Chen, Ming Hu, Darren J. Coker, Michael L. Berumen, Blair Costelloe, Sara Beery, Anna Rohrbach, and Mohamed Elhoseiny. MammalNet: A large-scale video benchmark for mammal recognition and behavior understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13052–13061, 2023.

[7] Yutao Cui, Chenkai Zeng, Xiaoyu Zhao, Yichun Yang, Gangshan Wu, and Limin Wang. Sportsmot: A large multi-object tracking dataset in multiple sports scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9921–9931, 2023.

[8] Achal Dave, Tarasha Khurana, Pavel Tokmakov, Cordelia Schmid, and Deva Ramanan. TAO: A large-scale benchmark for tracking any object. In Proceedings of the European Conference on Computer Vision, pages 436–454, 2020.

[9] Patrick Dendorfer, Aljosa Oˇ sep, Anton Milan, Konrad Schindler, Daniel Cremers,ˇ Ian Reid, Stefan Roth, and Laura Leal-Taixe. MOTChallenge: A benchmark for´ single-camera multiple target tracking. International Journal of Computer Vision, 129(4):845–881, 2021.

[10] Patrick Dendorfer, Hamid Rezatofighi, Anton Milan, Javen Shi, Daniel Cremers, Ian Reid, Stefan Roth, Konrad Schindler, and Laura Leal-Taixe. MOT20:´ A benchmark for multi-object tracking in crowded scenes. arXiv preprint arXiv:2003.09003, 2020.

[11] Dawei Du, Yuankai Qi, Hongyang Yu, Yi Yang, Kaiwen Duan, Guorong Li, Weigang Zhang, Qingming Huang, and Qi Tian. The unmanned aerial vehicle benchmark: Object detection and tracking. In Proceedings of the European Conference on Computer Vision, pages 370–386, 2018.

[12] Ruopeng Gao, Jinkun Qi, and Limin Wang. Multiple object tracking as ID prediction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 27883–27893, 2025.

[13] Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we ready for autonomous driving? the KITTI vision benchmark suite. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, pages 3354–3361, 2012.

[14] Wen Guo, Pengfei Zhao, Zongmeng Wang, Yufan Hu, and Junyu Gao. Dual-level adaptation for multi-object tracking: Building test-time calibration from experience and intuition. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 28190–28200, 2026.

[15] Bingliang Jiao, Lingqiao Liu, Liying Gao, Ruiqi Wu, Guosheng Lin, Peng Wang, and Yanning Zhang. Toward re-identifying any animal. In Advances in Neural Information Processing Systems, volume 36, pages 40042–40053, 2023.

[16] Faizan Farooq Khan, Xiang Li, Andrew J. Temple, and Mohamed Elhoseiny. FishNet: A large-scale dataset and benchmark for fish recognition, detection, and functional trait prediction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20496–20506, 2023.

[17] Jonathon Luiten, Aljosa Oˇ sep, Patrick Dendorfer, Philip H. S. Torr, Andreas Geiger,ˇ Laura Leal-Taixe, and Bastian Leibe. HOTA: A higher order metric for evaluating´ multi-object tracking. International Journal ofComputer Vision, 129(2):548–578, 2021.

[18] Weiyi Lv, Yuhang Huang, Ning Zhang, Ruei-Sung Lin, Mei Han, and Dan Zeng. DiffMOT: A real-time diffusion-based multiple object tracker with non-linear prediction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19321–19330, 2024.

[19] Xiaoxuan Ma, Stephan P. Kaufhold, Jiajun Su, Wentao Zhu, Jack Terwilliger, Andres Meza, Yixin Zhu, Federico Rossano, and Yizhou Wang. ChimpACT: A longitudinal dataset for understanding chimpanzee behaviors. Advances in Neural Information Processing Systems, 36:27501–27531, 2023.

[20] Xun Long Ng, Kian Eng Ong, Qichen Zheng, Yun Ni, Si Yong Yeo, and Jun Liu. Animal kingdom: A large and diverse dataset for animal behavior understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19023–19034, 2022.

[21] Jiangmiao Pang, Linlu Qiu, Xia Li, Haofeng Chen, Qi Li, Trevor Darrell, and Fisher Yu. Quasi-dense similarity learning for multiple object tracking. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 164–173, 2021.

[22] Hamid Rezatofighi, Nathan Tsoi, JunYoung Gwak, Amir Sadeghian, Ian Reid, and Silvio Savarese. Generalized intersection over union: A metric and a loss

for bounding box regression. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 658–666, 2019.

[23] Ergys Ristani, Francesco Solera, Roger Zou, Rita Cucchiara, and Carlo Tomasi. Performance measures and a data set for multi-target, multi-camera tracking. In Proceedings ofthe European Conference on Computer Vision Workshops, pages 17–35, 2016.

[24] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, et al. ImageNet large scale visual recognition challenge. International Journal ofComputer Vision, 115(3):211–252, 2015.

[25] Kyujin Shim, Kyungdon Ko, Youngmin Yang, and Changick Kim. Focusing on tracks for online multi-object tracking. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11687–11696, 2025.

[26] Peize Sun, Jinkun Cao, Yi Jiang, Zehuan Yuan, Song Bai, Kris Kitani, and Ping Luo. DanceTrack: Multi-object tracking in uniform appearance and diverse motion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20993–21002, 2022.

[27] Peize Sun, Jinkun Cao, Yi Jiang, Rufeng Zhang, Enze Xie, Zehuan Yuan, Changhu Wang, and Ping Luo. TransTrack: Multiple object tracking with transformer. arXiv preprint arXiv:2012.15460, 2020.

[28] Wei Wang. X-AnyLabeling: A unified desktop platform for ai-assisted data annotation. GitHub repository, CVHub, 2023.

[29] Dante Wasmuht, Otto Brookes, Maximilian Schall, Pablo Palencia, Christopher Beirne, Tilo Burghardt, et al. The SA-FARI dataset: Segment anything in footage of animals for recognition and identification. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21679–21689, 2026.

[30] Linjie Yang, Yuchen Fan, and Ning Xu. Video instance segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5188–5197, 2019.

[31] Yuxiang Yang, Junjie Yang, Yufei Xu, Jing Zhang, Long Lan, and Dacheng Tao. APT-36K: A large-scale benchmark for animal pose estimation and tracking. In Advances in Neural Information Processing Systems, volume 35, pages 17301– 17313, 2022.

[32] Kaining Ying, Hengrui Hu, and Henghui Ding. MOVE: Motion-guided fewshot video object segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025.

[33] Hang Yu, Yufei Xu, Jing Zhang, Wei Zhao, Ziyu Guan, and Dacheng Tao. AP-10K: A benchmark for animal pose estimation in the wild. In Proceedings ofthe Neural Information Processing Systems Track on Datasets and Benchmarks, 2021.

[34] Libo Zhang, Junyu Gao, Zhen Xiao, and Heng Fan. AnimalTrack: A benchmark for multi-animal tracking in the wild. International Journal ofComputer Vision, 131(2):496–513, 2023.

[35] Yi-Fan Zhang, Weiqiang Ren, Zhang Zhang, Zhen Jia, Liang Wang, and Tieniu Tan. Focal and efficient IOU loss for accurate bounding box regression. Neurocomputing, 506:146–157, 2022.

[36] Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, and Xinggang Wang. ByteTrack: Multi-object tracking by associating every detection box. In Proceedings ofthe European Conference on Computer Vision, pages 1–21, 2022.

[37] Yifu Zhang, Chunyu Wang, Xinggang Wang, Wenjun Zeng, and Wenyu Liu. FairMOT: On the fairness of detection and re-identification in multiple object tracking. International Journal ofComputer Vision, 129(11):3069–3087, 2021.

[38] Zhaohui Zheng, Ping Wang, Wei Liu, Jinze Li, Rongguang Ye, and Dongwei Ren. Distance-IoU loss: Faster and better learning for bounding box regression. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 12993–13000, 2020.

## Appendix

To better understand VastMAT and CDA in this work, we provides additional visual results, dataset documentation, annotation validation, and experimental analyses. It is organized as follows:

• A Qualitative Results

Visual comparisons of eight baseline trackers across animal scenes.

• B Dataset Categories and Statistics The category inventory, video distributions, co-occurrence, and visual similarity.

• C Annotation Protocol and Quality Assessment The annotation format, review procedure, and independent reannotation audit.

## • D Additional Benchmark Results

Detection, animal-group tracking, and video-level performance analyses.

• E CDA Implementation and Additional Experiments Association settings, design comparisons, sensitivity analyses, diagnostics, and runtime.

• F Dataset Documentation and Maintenance

Data access, versioning, issue reporting, and intended use.

## A Qualitative Results

The main text presents representative examples in Figure 5; this appendix provides more detailed qualitative comparisons in Figure 6.

Figure 6 compares eight baseline trackers with ground truth across diverse animal scenes. Each triplet shows three sampled frames from one sequence, with frame indices in the upper-left corners. Colors identify trackers according to the legend; red boxes denote ground truth. Hamster-6 illustrates motion blur and variation in predicted box extent, while Crocodile-23 and Red Panda-1 contain overlapping animals and partial occlusion. Bowerbird-13 and Cicada-11 show incomplete prediction coverage for some small or low-contrast targets. The underwater scene in Turtle-41 and wave clutter in Albatross-4 further illustrate the diversity of imaging conditions. Together, these examples highlight scene-specific challenges in object coverage and localization that complement the quantitative evaluation.In Figure 6, panels (c) and (g) are generated using Protocol 1 boxes, and ”Protocol 1 and $2 ^ { \circ }$ in the caption means that the video appears in the test sets of both protocols.

![](images/ae9a6a0c945bce6b2c0bab2a7f6c48155c7dde2b2d38b30bee726b61af05974c.jpg)  
Figure 6: Qualitative comparison of eight baseline trackers on selected VastMAT sequences.

## B Dataset Categories and Statistics

This section documents the category inventory and provides additional statistics on video composition, category co-occurrence, and visual similarity.

## B.1 Animal Categories and Grouping

VastMAT contains 337 animal categories. To summarize coverage and support group-wise evaluation, we assign them to six coarse groups: mammals, birds, fish, amphibians, reptiles, and other animals. Group membership follows the animal category rather than the filming environment. For example, Dolphin, Killer Whale, and Whale are mammals; Penguin is a bird; and Shark and Ray are fish. Other animals include the insects, arachnids,

Table 9: Animal groups in VastMAT and their Protocol 2 category counts.
<table><tr><td>Group</td><td>All</td><td>P2 Seen</td><td>P2 Unseen</td></tr><tr><td>Mammals</td><td>82</td><td>65</td><td>17</td></tr><tr><td>Birds</td><td>85</td><td>74</td><td>11</td></tr><tr><td>Fish</td><td>117</td><td>103</td><td>14</td></tr><tr><td>Amphibians</td><td>4</td><td>4</td><td>0</td></tr><tr><td>Reptiles</td><td>7</td><td>7</td><td>0</td></tr><tr><td>Others</td><td>42</td><td>30</td><td>12</td></tr><tr><td>Total</td><td>337</td><td>283</td><td>54</td></tr></table>

crustaceans, mollusks, and other invertebrates represented in the dataset.

Table 9 summarizes the six groups and their Protocol 2 training and test category counts. The complete category names are listed below; the accompanying category dictionary defines the mapping from category IDs to English names.

Mammals (82 categories). Alpaca, Anteater, Antelope, Ape, Armadillo, Babirusa, Baboon, Badger, Bat, Bear, Beaver, Bison, Buffalo, Camel, Cat, Cattle, Cheetah, Chimpanzee, Civet, Colugo, Deer, Dog, Dolphin, Donkey, Echidna, Elephant, Fox, Giraffe, Goat, Gorilla, Guinea Pig, Hamster, Hedgehog, Hippopotamus, Horse, Hyena, Kangaroo, Killer Whale, Koala, Lemur,

Leopard, Lion, Lynx, Manatee, Marmot, Mongoose, Monkey, Mouse, Orangutan, Otter, Panda, Pangolin, Pig, Pika, Platypus, Porcupine, Porpoise, Possum, Quokka, Rabbit, Racoon, Red Panda, Rhinoceros, Sea Lion, Seal, Sheep, Sloth, Slow Loris, Snow Leopard, Squirrel, Stoat, Tapir, Tiger, Treeshrew, Walrus, Whale, Wildebeest, Wolf, Wolverine, Wombat, Yak, Zebra.

Birds (85 categories). Albatross, Bittern, Bluethroat, Bowerbird, Bulbul, Bustard, Buzzard, Chicken, Cormorant, Cowbird, Crane, Crow, Cuckoo, Curassow, Dipper, Drongo, Duck, Eagle, Falcon, Finch, Flamingo, Frigatebird, Goldcrest, Goldeneye, Goose, Great Argus, Grebe, Gull, Harrier, Heron, Hoopoe, Hornbill, Hummingbird, Ibis, Kingfisher, Kiwi, Lapwing, Lark, Myna, Nightingale, Nuthatch, Oriole, Ostrich, Owl, Parrot, Peacock, Pelican, Penguin, Pigeon, Pipit, Plover, Puffin, Quail, Rail, Robin, Sandpiper, Secretarybird, Shearwater, Shoebill, Shorebird, Shrike, Sparrow, Starling, Stilt, Stork, Swallow, Swan, Tern, Thick-knee, Thrush, Tinamou, Tit, Toucan, Turkey, Vulture, Wagtail, Warbler, Waterhen, Woodpecker, Wren, Wryneck, Jays, Cardinal bird, Magpie, Blackbird.

Fish (117 categories). Angelfish, Anglerfish, Anthias, Arapaima, Archerfish, Arowana, Bala Shark, Barb, Barracuda, Bass, Batfish, Betta, Bichir, Billfish, Blind cavefish, Burbot, Butterflyfish, Cardinalfish, Carp, Catfish, Chimaera, Cichlid, Cleaner Wrasse, Climbing perch, Clownfish, Cod, Coelacanth, Boxfish, Damselfish, Danio, Datnoid, Discus, Dottyback, Drum fish, Eel, Electric eel, Elephantnose fish, Fairy Wrasse, Fangblenny, Flounder, Flowerhorn, Flying fish, Fusilier, Gar, Glassy fish, Goby, Goldfish, Gourami, Grayling, Grouper, Guppy, Hatchetfish, Hawkfish, Herring, Hillstream loach, Hogfish, Knifefish, Lionfish, Loach, Lungfish, Mandarinfish, Milkfish, Molly, Moorish Idol, Mudskipper, Needlefish, Ocean Sunfish, Oscar, Paddlefish, Paradise fish, Parrotfish, Payara, Perch, Pike, Pineapplefish, Piranha, Pufferfish, Rabbitfish, Rainbowfish, Ray, Rockfish, Royal Gramma, Sailfin Tang, Salmon, Scorpionfish, Sculpin, Seahorse, Shark, Silver Dollar, Snailfish, Snakehead, Snapper, Snipefish, Squirrelfish, Sturgeon, Sunfish, Sweetlips, Taimen, Tangs, Tench, Tetra, Tilapia, Toadfish, Trevally, Triggerfish, Trout, Trumpetfish, Tuna, Unicornfish, White cloud mountain minnow, Wolffish, Wrasse, Yellow Wrasse, Zander, Mackerel, Sixline Wrasse, Tarpon.

Amphibians (4 categories). Frog, Giant Salamander, Newt, Toad.

Reptiles (7 categories). Chameleon, Crocodile, Gecko, Lizard, Snake, Tortoise, Turtle.

Other animals (42 categories). Ant, Bee, Beetle, Brittle Star, Butterfly, Caterpillar, Centipede, Cicada, Cockroach, Crab, Cricket, Cuttlefish, Dragonfly, Earthworm, Feather Star, Firefly, Fly, Grasshopper, Horseshoe Crab, Jellyfish, Ladybug, Leaf Insect, Lobster, Mantis, Mealworm, Millipede, Mosquito, Moth, Nautilus, Octopus, Scallop, Scorpion, Sea Anemone, Sea Spider, Shrimp, Snail, Spider, Squid, Starfish, Stick Insect, Termite, Wasp.

## B.2 Video-Level Distributions

Figure 7 summarizes four video-level attributes: trajectory count, average active instances per frame, video duration, and valid category count. In Figure 7(a), 607 videos contain more than 10 trajectories, including 252 with at least 21, with a maximum of 152 per video. Figure 7(b) shows that 664 videos average more than 5 active instances per frame, including 63 with more than 20. In Figure 7(c), over half the videos last 20–60 seconds, while 145 exceed 90 seconds. Figure 7(d) shows 2,120 single-category and 827 multi-category videos, with up to 10 categories per video. VastMAT thus covers sparse scenes, dense groups, long sequences, and multi-category interactions.

![](images/0d6f2079549ac5325944fec69651470dd8f108b761a90cf5b4ef05d456e4633b.jpg)  
(a) Tracks per video

![](images/b618a9d9e59d02f7bad1cb4fd90de6d6d94dbbfb66e8d0d69337383bd363f822.jpg)  
(b) Active instances per frame

![](images/2c30f696845baa7effdf6e431eb2e992b626a35e4a534707b45bc18f4fcd43a1.jpg)  
(c) Duration at 10 FPS (s)

![](images/5330f3e7645bbdbb74ec703d501672788d6b1a6c3bc65eb9cfc7bac6a5355608.jpg)  
(d) Effective classes per video  
Figure 7: Video-level distributions of (a) trajectory count, (b) average active instances per frame, (c) duration, and (d) category count in VastMAT.

## B.3 Category Coverage and Co-occurrence Structure

Per-category distributions. Figure 8 shows per-category distributions of videos, trajectories, and bounding boxes.

![](images/493bcdc0e2ffb0b3d930cafa425e4f2c66d238cdecc6a77bee0f7c8d54360de6.jpg)  
(a) Videos per category

![](images/402287f45104a8ee7da55d077d46ddcdab6a8dba53a3d050b562319f5a545f0c.jpg)  
(b) Tracks per category

![](images/349ae8ff178bdf7643c09b5c85c08056f0a97723ca0c84ac677239fd45e17cf9.jpg)  
(c) Bounding boxes per category  
Figure 8: Per-category distributions of videos, trajectories, and bounding boxes in VastMAT.

Complete category coverage. Figure 9 visualizes the distribution of all 337 categories across six major animal groups. The inner donut chart reports the category counts and proportions: Fish (117 categories, 34.7%), Birds (85, 25.2%), Mammals (82, 24.3%), Other animals (42, 12.5%), Amphibians (4, 1.2%), and Reptiles (7, 2.1%). The outer ring further breaks down each major group into its constituent categories, showing complete coverage of the 337 categories.

![](images/7df7b959a8ffef3c3f221ebe298e6dd65841528883cf4a7f8a22894c7d6737cc.jpg)  
Figure 9: All 337 categories in VastMAT grouped into six major animal groups, with inner-ring proportions and outer-ring category labels.

Frequent categories and co-occurrence. Figure 10(a) shows distinct scene compositions across categories. Tangs, Clownfish, and Damselfish often appear as secondary categories in multi-category videos, whereas Duck and Monkey more often appear as the primary subject. Multi-category videos therefore increase scene complexity while broadening coverage of naturally co-occurring categories.

Figure 10(b) shows the connected components of the category co-occurrence graph. The largest contains 266 categories, with 3 additional two-category components and 65 isolated categories, indicating that most categories are linked through multi-category videos. This structure requires category-disjoint splits to assign entire connected components, preventing categories from the same video from entering both training and testing.

## B.4 Visual Similarity Across Categories

VastMAT includes visually similar animals from different categories among its 337 categories. Figure 11 shows six representative pairs: Silver Dollar and Piranha, and

Guppy and Molly, have similar body outlines; Grebe and Duck share head, neck, and torso structures in side views; Moorish Idol and Butterflyfish have similar stripes and body shapes; and Goldfish and Cichlid, and Yellow Sailfin Tang and Tangs, closely resemble one another in color and overall appearance.

Differences between these categories often lie in local shape and texture, which low resolution, motion blur, pose changes, and occlusion can obscure. In multi-category videos, similar-looking individuals may yield similar association features, complicating cross-frame identity discrimination. These examples illustrate instance-level appearance ambiguity in the benchmark.

![](images/24af4a3d0ad2f0b1849f07719ceba5f0eb596dc1916b2b52b20c6ef96558a52d.jpg)

Figure 10: Category statistics in VastMAT: (a) composition of the 20 most frequent categories and (b) connected components in the category co-occurrence graph.  
![](images/bfeded52e03a5d53b23772854cc65030d2a100bd01a51525323d88c3f5aa03c7.jpg)  
Figure 11: Representative visually similar category pairs in VastMAT: (a) Silver Dollar vs. Piranha, (b) Grebe vs. Duck, (c) Moorish Idol vs. Butterflyfish, (d) Goldfish vs. Cichlid (Parrot Cichlid), (e) Yellow Sailfin Tang vs. Tangs (Yellow Tang), and (f) Guppy vs. Molly.

## C Annotation Protocol and Quality Assessment

This section specifies the annotation interface and documents the inspection procedure and independent quality audit.

## C.1 Annotation Format

Annotations for each video are stored in a nine-column gt.txt file, with fields defined in Table 10.

Table 10: The nine-column annotation interface of gt.txt in VastMAT.
<table><tr><td>Position</td><td>Name</td><td>Description</td></tr><tr><td>1</td><td>Frame number</td><td>Index of the annotated frame; starts at 1.</td></tr><tr><td>2</td><td>Identifier</td><td>Video-local unique track ID; IDs need not be consecutive.</td></tr><tr><td>3</td><td>Box left</td><td>Left x-coordinate of the axis-aligned bounding box, in pixels.</td></tr><tr><td>4</td><td>Box top</td><td>Top y-coordinate of the axis-aligned bounding box, in pixels.</td></tr><tr><td>5</td><td>Box width</td><td>Width of the axis-aligned bounding box, in pixels.</td></tr><tr><td>6</td><td>Box height</td><td>Height of the axis-aligned bounding box, in pixels.</td></tr><tr><td>7</td><td>Confidence</td><td>Annotation flag; set to 1 for valid annotated targets.</td></tr><tr><td>8</td><td>Class</td><td>Global class ID mapped by the released category dictionary.</td></tr><tr><td>9</td><td>Visibility</td><td>Reserved visibility field; set to —1 when unspecified. This value does not mark an ignored target.</td></tr></table>

## C.2 Annotation Inspection and Quality Assessment

Manual inspection and correction. Annotation review examines object coverage, box localization, category labels, and identity continuity through continuous playback and frame-by-frame inspection. Playback reveals identity correspondences during entry, exit, occlusion, and interactions, while individual frames are checked for missed objects, duplicate annotations, and box placement. Corrections cover the affected frame and relevant neighboring frames. Propagation drift is checked along subsequent frames, and identity errors are traced along the associated trajectories.

Sampling and independent reannotation. We select 48 videos through stratified random sampling across all six animal groups, with the sample composition reported in Table 11. Two non-overlapping, consecutive 20-frame clips are randomly selected from each video, yielding 96 clips and 1,920 frames. Reannotators receive only images without annotation overlays and the shared guidelines, without access to the original annotation files. Reannotation follows the same rules for visible-region boxes, occlusion handling, category mapping, and within-clip identity preservation. 9 frames contain no boxes in either annotation set; Remaining 1,911 contain annotations from at least one set.

Agreement metrics. We use the Hungarian algorithm for one-to-one box matching within each frame, accepting matches with IoU ≥ 0.5. Let $N _ { \mathrm { o } } , N _ { \mathrm { r } }$ , and M denote the original box count, reannotated box count, and valid matched-pair count, respectively.

The two matching coverage rates and box-matching F1 are defined as

$$
C _ { 0 } = \frac { M } { N _ { \mathrm { o } } } , \qquad C _ { \mathrm { r } } = \frac { M } { N _ { \mathrm { r } } } , \qquad F _ { \mathrm { l } } = \frac { 2 M } { N _ { \mathrm { o } } + N _ { \mathrm { r } } } .\tag{4}
$$

Mean IoU is computed only over valid matched pairs, and category agreement compares their labels. These metrics quantify correspondence between the annotation sets, while unmatched boxes identify disagreements in object coverage or localization.

Overall and group-wise results.   
The audit contains 7,885 original   
and 8,010 independently rean  
notated boxes, with 7,632 valid   
matches. Matching coverage   
is 96.79% for the original an  
notations and 95.28% for rean  
notations; box-matching F1 is   
96.03%, and mean matched-box   
IoU is 0.9555. All 7,632 matched   
sets contain 253 and 378 unmatched boxes, respectively. sets contain 253 and 378 unmatc

Table 11: Independent reannotation audit by animal group. Coverage and F1 are percentages.
<table><tr><td>Group</td><td>Videos</td><td> $N _ { 0 }$ </td><td> $N _ { \mathrm { r } }$ </td><td>M</td><td> $C _ { 0 }$ </td><td> $C _ { \mathrm { r } }$ </td><td> $\overline { { F _ { 1 } } }$ </td></tr><tr><td>Mammals</td><td>8</td><td>1,093</td><td>1,116</td><td>1,079</td><td>98.72</td><td>96.68</td><td>97.69</td></tr><tr><td>Birds</td><td>8</td><td>1,029</td><td>1,032</td><td>996</td><td>96.79</td><td>96.51</td><td>96.65</td></tr><tr><td>Fish</td><td>8</td><td>1,948</td><td>1,941</td><td>1,878</td><td>96.41</td><td>96.75</td><td>96.58</td></tr><tr><td>Amphibians</td><td>7</td><td>914</td><td>926</td><td>884</td><td>96.72</td><td>95.46</td><td>96.09</td></tr><tr><td>Reptiles</td><td>8</td><td>601</td><td>715</td><td>575</td><td>95.67</td><td>80.42</td><td>87.39</td></tr><tr><td>Others</td><td>9</td><td>2,300</td><td>2,280</td><td>2,220</td><td>96.52</td><td>97.37</td><td>96.94</td></tr><tr><td>Overall</td><td>48</td><td>7,885</td><td>8,010</td><td>7,632</td><td>96.79</td><td>95.28</td><td>96.03</td></tr></table>

pairs agree in category. The original and reannotated

Table 11 reports group-wise results. Matching coverage exceeds 95% on both sides for all five groups except reptiles. Of the 140 unmatched reannotated reptile boxes, 134 come from Snake-5, Snake-61, and Lizard-

Table 12: Original-side matching coverage by object size.
<table><tr><td>Box area a (pixels2)</td><td>Original boxes</td><td>Unmatched</td><td> $C _ { \bf { 0 } } \left( \% \right)$ </td></tr><tr><td> $\overline { { a < 3 2 ^ { 2 } } }$ </td><td>310</td><td>59</td><td>80.97</td></tr><tr><td> $3 2 ^ { 2 } \leq a < 6 4 ^ { 2 }$ </td><td>1,120</td><td>60</td><td>94.64</td></tr><tr><td> $6 4 ^ { 2 } \leq a < 1 2 8 ^ { 2 }$ </td><td>1,508</td><td>66</td><td>95.62</td></tr><tr><td> $a \ge 1 2 8 ^ { 2 }$ </td><td>4,947</td><td>68</td><td>98.63</td></tr></table>

1, indicating that disagreement is concentrated in a few sequences. Overall metrics aggregate all audited boxes and describe agreement within this sample.

Object-scale analysis. We group original boxes by their pixel area in the audited images (See Table 12). Matching coverage on the original side increases with object size, from 80.97% for the smallest group to 98.63% for the largest. Objects with area below $3 2 ^ { 2 }$ pixels account for 3.93% of original boxes but 23.32% of unmatched original boxes, indicating that small objects warrant particular attention during annotation review.

Within-clip identity consistency. We assess temporal identity correspondence by mapping original identities to reannotated identities within each 20- frame clip. An original identity matched in at least two frames is consistent if every matched frame maps to the same reanno-

Table 13: Within-clip identity consistency in the independent reannotation audit.
<table><tr><td>Metric</td><td>Value</td></tr><tr><td>Analyzed clips</td><td>96</td></tr><tr><td>Eligible clip-identity pairs (matched in ≥ 2 frames)</td><td>427</td></tr><tr><td>Pairs mapped to a single reannotated identity</td><td>424</td></tr><tr><td>Within-clip identity consistency</td><td>99.30%</td></tr><tr><td>Mean majority coverage</td><td>99.81%</td></tr><tr><td>Mean longest consecutive consistent span</td><td>18.49 frames</td></tr><tr><td>Videos with consistent identities in all audited clips</td><td>46/48</td></tr></table>

tated identity. Within-clip identity consistency is the fraction of eligible clip–identity pairs satisfying this condition; the same identity in different audited clips is counted separately. We also report majority coverage and the longest consecutive consistent span to characterize the stability of within-clip correspondence.

The 96 audited clips contain 427 eligible clip–identity pairs, of which 424 map to a single reannotated identity, yielding 99.30% within-clip identity consistency. Mean majority coverage is 99.81%, and the mean longest consecutive consistent span is 18.49 frames. Among the 48 videos, 46 have a within-clip IDR of 1.00, while 2 fall below 1.00: the second clip of Crocodile-2 has an IDR of 0.50 (only 2 GT IDs), and the second clip of Bee-7 has an IDR of 0.93 (2 of 30 densely distributed GT IDs change identity). Both involve brief within-clip identity assignment changes in scenes with changing object counts or dense small objects, consistent with entry and exit under occlusion and candidate competition. Table 13 summarizes the clip-level identity results.

This metric evaluates 20-frame clips and complements box-level agreement; identity re-entry after prolonged occlusion can be further assessed on full trajectories.

## D Additional Benchmark Results

This section supplements Section 5.2 with detection results and tracking analyses by animal group and video difficulty.

## D.1 Additional Detection Results

## D.1.1 Group-wise Detection at Different Localization Thresholds

Table 14 reports detection precision for each animal group at IoU thresholds of 0.50 and 0.75, showing performance under different localization requirements.

## D.1.2 Average Recall at Different Prediction Limits

We complement the precision analysis in Table 5 with average recall at different prediction limits. Evaluation is categoryagnostic, mapping all valid animals to the animal foreground

<table><tr><td colspan="7">Table 14:  $\mathsf { A P } _ { 5 0 }$  and  $\mathsf { A P 7 5 }$  (%) by animal group.</td></tr><tr><td>Protocol</td><td>Metric</td><td>Mammals</td><td>Fish</td><td>Birds</td><td>Amph.</td><td>Rept. Others</td></tr><tr><td rowspan="2">P1</td><td> $\mathrm { A P } _ { 5 0 }$ </td><td>85.28</td><td>82.57</td><td>87.71</td><td>86.01</td><td>29.76 62.03</td></tr><tr><td> $\mathrm { A P } _ { 7 5 }$ </td><td>77.15</td><td>77.04</td><td>79.60</td><td>77.31 26.31</td><td>47.70</td></tr><tr><td rowspan="2">P2</td><td> $\mathrm { A P } _ { 5 0 }$ </td><td>69.15</td><td>80.84</td><td>89.73</td><td>一</td><td>一 33.82</td></tr><tr><td> $\mathrm { A P } _ { 7 5 }$ </td><td>57.04</td><td>72.38</td><td>77.86</td><td>一 一</td><td>23.33</td></tr></table>

class. Table 15 reports average recall with at most 1, 10, and 100 predicted boxes per frame, averaged over IoU thresholds from 0.50 to 0.95. All results are percentages.

Increasing the prediction limit from 1 to 10 improves average recall by 37.27 and 33.75 percentage points under the two protocols, reflecting the need for multiple candidates in multi-instance scenes. Raising the limit to 100 yields only 0.78 and 0.71 additional points, indicating limited aggregate recall gains from

Table 15: Average recall at different per-frame prediction limits under both protocols.
<table><tr><td>Protocol</td><td>AR@1</td><td>AR@10</td><td>AR@100</td></tr><tr><td>Protocol 1</td><td>42.40</td><td>79.67</td><td>80.45</td></tr><tr><td>Protocol 2</td><td>35.63</td><td>69.38</td><td>70.09</td></tr></table>

retaining more candidates under the current detector outputs and test distributions.

## D.2 Tracking Results by Animal Group

Table 16 reports tracking performance by animal group. In Protocol 2, all methods achieve higher HOTA on birds than on the “other animals” group, revealing substantial variation beyond overall scores. Together with the detection analysis in the main text, these results characterize performance differences across diverse animal categories.

## D.3 Per-Video HOTA and Stratified Statistics

This subsection provides full statistics for the videolevel long-tail analysis in Section 5.2. Protocol 1 and Protocol 2 each contain 315 test videos but use different videos, precluding paired video-level comparisons.

## D.3.1 Complete Video-Level HOTA Statistics

Table 16: Average HOTA by coarse animal group under Protocol 1 and Protocol 2. Groups follow the definition in Table 5.

Table 17 reports HOTA distributions for eight methods across the 315 test videos in each protocol. Mean is the arithmetic mean of per-video HOTA;

<table><tr><td>Method</td><td>Mammals</td><td>Fish</td><td>Birds</td><td>Amphibians</td><td>Reptiles</td><td>Others</td></tr><tr><td colspan="7">Protocol 1</td></tr><tr><td>DiffMOT</td><td>67.02</td><td>63.72</td><td>71.97</td><td>83.98</td><td>49.42</td><td>50.02</td></tr><tr><td>TrackTrack</td><td>66.05</td><td>62.58</td><td>69.96</td><td>83.17</td><td>48.48</td><td>50.37</td></tr><tr><td>MOTIP</td><td>63.58</td><td>57.00</td><td>68.28</td><td>78.21</td><td>51.82</td><td>49.91</td></tr><tr><td>TCEI</td><td>62.95</td><td>56.35</td><td>68.90</td><td>80.68</td><td>49.22</td><td>51.97</td></tr><tr><td>ByteTrack</td><td>57.65</td><td>51.05</td><td>60.50</td><td>80.44</td><td>42.42</td><td>47.45</td></tr><tr><td>QDTrack</td><td>54.73</td><td>50.13</td><td>58.35</td><td>72.09</td><td>35.92</td><td>35.22</td></tr><tr><td>TransTrack</td><td>42.14</td><td>35.70</td><td>46.70</td><td>64.29</td><td>23.83</td><td>17.69</td></tr><tr><td>FairMOT</td><td>41.13</td><td>34.19</td><td>42.13</td><td>66.86</td><td>23.74</td><td>29.51</td></tr><tr><td colspan="7">Protocol 2</td></tr><tr><td>DiffMOT</td><td>54.41</td><td>54.92</td><td>70.59</td><td>一</td><td>一</td><td>25.36</td></tr><tr><td>TrackTrack</td><td>54.18</td><td>53.42</td><td>68.22</td><td>一</td><td>一</td><td>26.73</td></tr><tr><td>TCEI</td><td>51.79</td><td>49.80</td><td>68.09</td><td>一</td><td>一</td><td>34.85</td></tr><tr><td>MOTIP</td><td>49.72</td><td>49.41</td><td>70.31</td><td></td><td>一</td><td>33.94</td></tr><tr><td>ByteTrack</td><td>45.92</td><td>43.37</td><td>58.32</td><td>一</td><td>一</td><td>24.19</td></tr><tr><td>QDTrack</td><td>41.09</td><td>40.54</td><td>56.67</td><td>一</td><td>一</td><td>19.22</td></tr><tr><td>TransTrack</td><td>33.24</td><td>32.61</td><td>49.97</td><td>一</td><td></td><td>13.95</td></tr><tr><td>FairMOT</td><td>29.38</td><td>26.99</td><td>43.46</td><td>1</td><td>1</td><td>18.11</td></tr></table>

Median, IQR, P10, and P90 denote the median, interquartile range, and 10th and 90th percentiles. The last two columns give the percentages of videos with HOTA below 20 and 40. These statistics weight videos equally, unlike the TrackEval aggregate results in the main comparison table.

In Protocol 2, DiffMOT has the highest mean per-video HOTA of 53.14. TCEI averages 51.95 but has a P10 of 23.18, exceeding DiffMOT’s 14.31. Thus, average performance and performance on low-scoring videos provide complementary information. Figure 12 shows the full empirical cumulative distributions of per-video HOTA.

## D.3.2 Grouping Videos by Performance

Within each protocol, we compute each video’s arithmetic mean HOTA across the eight baselines and sort videos in descending order. The top, middle, and bottom 105 videos form the Easy, Medium, and Hard groups, respectively. All methods share the same groups within a protocol, and Table 18 reports their arithmetic mean HOTA in each group. These groups are defined by the current baseline set’s test performance and are constructed separately because the protocols use different test videos.

Table 17: Full video-level HOTA statistics on the 315 test videos of each protocol.
<table><tr><td>Tracker</td><td>Mean</td><td>Median</td><td>IQR</td><td>P10</td><td>P90</td><td>%HOTA&lt;20</td><td>%HOTA&lt;40</td></tr><tr><td colspan="8">Protocol 1</td></tr><tr><td>TransTrack</td><td>38.45</td><td>37.40</td><td>29.95</td><td>10.77</td><td>65.97</td><td>20.63</td><td>55.24</td></tr><tr><td>QDTrack</td><td>51.84</td><td>53.13</td><td>28.32</td><td>22.42</td><td>79.73</td><td>7.94</td><td>27.62</td></tr><tr><td>FairMOT</td><td>37.90</td><td>36.19</td><td>29.63</td><td>14.19</td><td>66.55</td><td>21.27</td><td>57.46</td></tr><tr><td>ByteTrack</td><td>55.22</td><td>56.46</td><td>29.34</td><td>26.28</td><td>82.25</td><td>5.08</td><td>25.08</td></tr><tr><td>DiffMOT</td><td>65.19</td><td>68.11</td><td>27.39</td><td>34.33</td><td>92.74</td><td>3.81</td><td>13.02</td></tr><tr><td>MOTIP</td><td>61.26</td><td>61.87</td><td>30.07</td><td>33.87</td><td>91.31</td><td>4.76</td><td>15.87</td></tr><tr><td>TrackTrack</td><td>64.05</td><td>67.84</td><td>29.36</td><td>33.52</td><td>91.74</td><td>3.81</td><td>13.97</td></tr><tr><td>TCEI</td><td>61.27</td><td>62.15</td><td>28.84</td><td>31.94</td><td>89.98</td><td>3.81</td><td>17.14</td></tr><tr><td colspan="8">Protocol 2</td></tr><tr><td>TransTrack</td><td>33.39</td><td>31.91</td><td>27.54</td><td>7.48</td><td>63.51</td><td>32.70</td><td>65.71</td></tr><tr><td>QDTrack</td><td>40.62</td><td>40.15</td><td>31.56</td><td>8.53</td><td>69.66</td><td>22.22</td><td>49.52</td></tr><tr><td>FairMOT</td><td>29.91</td><td>27.86</td><td>23.74</td><td>6.86</td><td>55.35</td><td>35.56</td><td>75.56</td></tr><tr><td>ByteTrack</td><td>44.43</td><td>44.27</td><td>30.86</td><td>13.88</td><td>74.76</td><td>13.97</td><td>42.86</td></tr><tr><td>DiffMOT</td><td>53.14</td><td>55.95</td><td>35.39</td><td>14.31</td><td>84.77</td><td>13.02</td><td>30.16</td></tr><tr><td>MOTIP</td><td>51.27</td><td>51.10</td><td>29.60</td><td>21.33</td><td>82.34</td><td>8.89</td><td>30.79</td></tr><tr><td>TrackTrack</td><td>52.47</td><td>53.12</td><td>36.96</td><td>17.22</td><td>84.50</td><td>11.75</td><td>32.38</td></tr><tr><td>TCEI</td><td>51.95</td><td>51.63</td><td>29.59</td><td>23.18</td><td>82.13</td><td>7.94</td><td>28.89</td></tr></table>

![](images/60a3d89ffb996fb03a7ee063c28a210a9ec9d56db98e846c36ed5e00d5bb6f26.jpg)  
Figure 12: Empirical CDF of per-video HOTA for all methods under Protocol 1 (left) and Protocol 2 (right). The curves show the fraction of videos below a given HOTA threshold.

Table 18: Mean per-video HOTA on the Easy, Medium, and Hard groups.
<table><tr><td>Difficulty</td><td>DiffMOT</td><td>TrackTrack</td><td>MOTIP</td><td>TCEI</td><td>ByteTrack</td><td>QDTrack</td><td>FairMOT</td><td>TransTrack</td></tr><tr><td colspan="9">Protocol 1</td></tr><tr><td>Easy</td><td>86.03</td><td>84.78</td><td>83.89</td><td>83.06</td><td>75.43</td><td>71.27</td><td>57.51</td><td>56.46</td></tr><tr><td>Medium</td><td>68.14</td><td>67.21</td><td>61.37</td><td>62.30</td><td>56.81</td><td>54.16</td><td>37.36</td><td>37.93</td></tr><tr><td>Hard</td><td>41.39</td><td>40.15</td><td>38.51</td><td>38.47</td><td>33.42</td><td>30.09</td><td>18.83</td><td>20.95</td></tr><tr><td colspan="9">Protocol 2</td></tr><tr><td>Easy</td><td>79.11</td><td>77.90</td><td>74.89</td><td>74.86</td><td>66.80</td><td>62.81</td><td>48.76</td><td>54.03</td></tr><tr><td>Medium</td><td>55.59</td><td>55.43</td><td>50.64</td><td>51.91</td><td>45.63</td><td>41.54</td><td>28.53</td><td>31.19</td></tr><tr><td>Hard</td><td>24.73</td><td>24.09</td><td>28.27</td><td>29.08</td><td>20.85</td><td>17.51</td><td>12.44</td><td>14.96</td></tr></table>

Table 18 shows that DiffMOT achieves the highest mean HOTA in all three Protocol 1

groups. In Protocol 2, it leads the Easy and Medium groups, whereas TCEI and MOTIP achieve 29.08 and 28.27 on the Hard group, exceeding DiffMOT’s 24.73. Relative strengths therefore vary across video groups, providing information beyond aggregate metrics.

## E CDA Implementation and Additional Experiments

This section details CDA’s association cost and candidate gating, and examines its design through fusion-weight, densityparameter, and post-processing comparisons. Offline candidate-pair diagnostics characterize the complementarity of IoU and center similarity, while runtime measurements assess integration overhead.

Table 19: CenterSim denominator ablation on the Protocol 2 test set $( \alpha = 0 . 5 )$
<table><tr><td>Denominator</td><td>HOTA</td><td>AssA</td><td>IDF1</td><td>IDSW</td></tr><tr><td>mean (default)</td><td>53.80</td><td>58.25</td><td>56.81</td><td>1972</td></tr><tr><td>max</td><td>53.58</td><td>57.77</td><td>56.60</td><td>1992</td></tr><tr><td>min</td><td>53.82</td><td>58.32</td><td>56.62</td><td>1953</td></tr><tr><td>enclosing</td><td>53.18</td><td>56.89</td><td>56.05</td><td>2018</td></tr></table>

## E.1 Association Cost and Candidate Gating

CDA operates at inference time and requires no additional training. In TrackTrack, the association cost is

$$
C ( a , b ) = 0 . 5 0 \big ( 1 - S ( a , b ) \big ) + 0 . 5 0 D _ { \mathrm { a p p } } ( a , b ) + 0 . 1 0 D _ { \mathrm { c o n f } } ( a , b ) + 0 . 0 5 D _ { \mathrm { a n g l e } } ( a , b ) ,\tag{5}
$$

where $D _ { \mathrm { a p p } } , D _ { \mathrm { c o n f } } ,$ , and $D _ { \mathrm { a n g l e } }$ denote appearance cosine distance, confidence difference, and motion-direction difference, respectively; S is the fused similarity in Equation equation 1.

Original TrackTrack retains its original gating. The normalized-DIoU control and CDA variants share DIoU gating, rejecting a candidate pair when DIoU<sup>]</sup> $( a , b ) \leq 0 . 1 0 .$ Among these controls, CDA changes association costs while preserving the gating function and threshold.

$$
\widetilde { \mathrm { D I o U } } ( a , b ) = \mathrm { c l i p } \left( \frac { \mathrm { I o U } ( a , b ) - \operatorname* { m i n } ( ( d _ { \mathrm { c e n t e r } } / c ) ^ { 2 } , 1 ) + 1 } { 2 } , 0 , 1 \right) ,\tag{6}
$$

Here, c is the diagonal length of the smallest enclosing rectangle. Association parameters are max time lost=20 det thr=init thr=match thr=0.60, tai thr=0.5 and penalty $\scriptstyle - { \mathrm { p } } / { \mathrm { q } } = 0 . 2 0 / 0 . 4 0$ , with original AFLink post-processing enabled by default. Apart from the explicitly reported geometric similarity and gating configurations, cached detections, FastReID features, TPA, TAI, and all other association parameters remain identical.

![](images/bcfd93814ccdeed55c94961d151b77354197f35fe4dee3f2c7c5238686f9656a.jpg)  
Figure 13: Fusion-weight sensitivity of TrackTrack and DiffMOT.

## E.2 Normalization Denominator Comparison

Table 19 fixes $\alpha = 0 . 5$ and varies only the center-distance denominator, retaining the other settings from Section 5.3. It reports association metrics and identity-switch counts for each configuration.

## E.3 Fusion-Weight Sensitivity

Figure 13 compares TrackTrack and DiffMOT across fusion weights, with numerical results in Tables 20 and 21. TrackTrack uses identical DIoU gating and AFLink settings across weights. As α increases from 0 to 1, HOTA ranges from 53.56 to 53.83; $\alpha = 0$ and $\alpha = 0 . 2 5$ both achieve 53.83, but the latter reduces IDSW from 2313 to 2054. Increasing the IoU weight to $\alpha = 0 . 7 5$ further reduces IDSW to 1956, with 53.68 HOTA, illustrating the differing effects of fusion weights on overall association and identity switches.

<table><tr><td colspan="2">Table 20: TrackTrack fusion- weight sensitivity on Protocol 2.</td></tr><tr><td>α HOTA</td><td>∆ vs TrackTrack IDSW</td></tr><tr><td>0 53.83</td><td>+1.22 2313</td></tr><tr><td>0.25 53.83</td><td>+1.22 2054</td></tr><tr><td>0.5 53.80</td><td>+1.19 1972</td></tr><tr><td>0.75 53.68</td><td>+1.07 1956</td></tr><tr><td>1 53.56</td><td>+0.95 2082</td></tr></table>

DiffMOT achieves its highest HOTA of 53.53 at $\alpha = 0 . 5 $ , improving on the baseline by 0.63 percentage points and outperforming both pure center similarity and pure IoU. The trackers’ responses indicate that overlap and center proximity are complementary, with their effective balance depending on the base tracker’s motion prediction and association configuration.

Table 21: DiffMOT fusion-weight sensitivity on Protocol 2.
<table><tr><td>α</td><td>HOTA</td><td>AssA</td><td>IDSW</td><td>ΔHOTA</td></tr><tr><td>baseline (CDA off)</td><td>52.90</td><td>57.61</td><td>2545</td><td></td></tr><tr><td>0</td><td>52.54</td><td>56.44</td><td>3188</td><td>-0.36</td></tr><tr><td>0.25</td><td>52.93</td><td>57.30</td><td>2711</td><td>+0.03</td></tr><tr><td>0.5</td><td>53.53</td><td>58.63</td><td>2467</td><td>+0.63</td></tr><tr><td>0.75</td><td>53.35</td><td>58.35</td><td>2439</td><td>+0.45</td></tr><tr><td>1</td><td>52.87</td><td>57.54</td><td>2528</td><td>-0.03</td></tr></table>

## E.4 Additional CDA Analyses

Table 22: AFLink sensitivity of two CDA con-Sensitivity to AFLink Post-processing. figurations (Protocol 2).Table 22 compares fixed-weight and Table 22 compares fixed-weight and

adaptive CDA with and without AFLink. Without AFLink, their HOTA scores are 53.74 and 53.79, respectively; enabling it increases them to 53.83 and 53.92. AFLink provides

<table><tr><td>Config.</td><td>AFLink</td><td>HOTA</td><td>AssA</td><td>IDF1</td><td>IDSW</td></tr><tr><td>Fixed α = 0.25</td><td>On</td><td>53.83</td><td>58.19</td><td>56.89</td><td>2054</td></tr><tr><td>Fixed α = 0.25</td><td>Off</td><td>53.74</td><td>57.97</td><td>56.71</td><td>2068</td></tr><tr><td>Adaptive</td><td>On</td><td>53.92</td><td>58.38</td><td>57.06</td><td>2166</td></tr><tr><td>Adaptive</td><td>Off</td><td>53.79</td><td>58.11</td><td>56.83</td><td>2183</td></tr></table>

modest gains for both configurations, and adaptive CDA achieves higher HOTA in both post-processing settings.

Density-Adaptive Parameter Grid. We vary the slope k and reference detection count $N _ { \mathrm { r e f } }$ in Equation equation 3 to assess parameter sensitivity. Across the nine configurations in Table 23, HOTA ranges from 53.36 to 53.92, consistently above original TrackTrack’s 52.61. The defaults $k = 0 . 0 5$ and $N _ { \mathrm { r e f } } = 1 0$ , selected on the training-side development set and fixed for testing, achieve 53.92 HOTA and 58.38

AssA. These results characterize association performance across different adaptation rates and reference counts.

Additional Seen-Category Results. Table 25 reports full Protocol 1 results for original TrackTrack and fixedweight and adaptive CDA. The fixed configuration uses $\alpha = 0 . 2 5$ , while the adaptive configuration follows the fusion rule defined in the main text.

Table 23: Density-adaptive parameter grid on Protocol 2.

Offline Candidate-Pair Diagnostics. To examine the discriminative complementarity of CenterSim and IoU, we analyze 190,890 association events offline over the full test set. For each event with a correct candidate, we define $m _ { I } = \mathrm { I o U } ( g _ { \mathrm { p r e v } } , c ^ { * } ) - \mathrm { I o U } ( g _ { \mathrm { p r e v } } , w ^ { * } )$ and $m _ { C } =$ CenterSim $( g _ { \mathrm { p r e v } } , c ^ { * } ) - 0$ CenterSim $\left( g _ { \mathrm { p r e v } } , w ^ { * } \right)$ , where $g _ { \mathrm { p r e v } }$ is the track’s previous-frame GT box, $c ^ { * }$ is the correct

<table><tr><td> $k$ </td><td> $N _ { \mathrm { r e f } }$ </td><td>HOTA</td><td>AssA</td><td>IDSW</td></tr><tr><td>0.02</td><td>5</td><td>53.47</td><td>57.53</td><td>2035</td></tr><tr><td>0.02</td><td>10</td><td>53.54</td><td>57.58</td><td>2041</td></tr><tr><td>0.02</td><td>20</td><td>53.82</td><td>58.17</td><td>2136</td></tr><tr><td>0.05</td><td>5</td><td>53.36</td><td>57.32</td><td>2034</td></tr><tr><td>0.05</td><td>10</td><td>53.92</td><td>58.38</td><td>2166</td></tr><tr><td>0.05</td><td>20</td><td>53.89</td><td>58.29</td><td>2171</td></tr><tr><td>0.10</td><td>5</td><td>53.49</td><td>57.53</td><td>2066</td></tr><tr><td>0.10</td><td>10</td><td>53.82</td><td>58.17</td><td>2171</td></tr><tr><td>0.10</td><td>20</td><td>53.78</td><td>58.05</td><td>2175</td></tr></table>

candidate with the highest IoU of at least 0.5 with the current GT box, and $w ^ { * }$ is the strongest incorrect competitor, defined as the candidate other than $c ^ { * }$ with the highest IoU with $g _ { \mathrm { p r e v } }$

Table 24 reports candidatepair discrimination by normalized displacement. For displacements from 0.5 to 1.0, only CenterSim correctly distinguishes the selected pair in 11.01% of events, compared with 4.83% for only IoU. This diagnostic reveals different responses to low-

Table 24: Candidate-pair discrimination by normalized displacement (center distance divided by the mean box diagonal). Fractions indicate pairs correctly distinguished by only CenterSim or only IoU.
<table><tr><td>Norm. displ.</td><td>only_center</td><td>only_iou</td><td>Better cue</td></tr><tr><td>≤ 0.25</td><td>0.24%</td><td>1.12%</td><td>IoU</td></tr><tr><td>0.25–0.5</td><td>1.83%</td><td>4.96%</td><td>IoU</td></tr><tr><td>0.5-1.0</td><td>11.01%</td><td>4.83%</td><td>CenterSim (2.3×)</td></tr><tr><td>&gt; 1.0</td><td>5.94%</td><td>2.31%</td><td>CenterSim (2.6×)</td></tr></table>

overlap events. It measures discrimination between specified candidate pairs, while online tracking evaluates association over complete candidate sets.

Inference Cost. Table 26 reports the computational overhead of CDA’s geometric similarity. With detections and appearance features cached, processing 93,111

<table><tr><td colspan="5">Table 25: CDA results on Protocol 1.</td></tr><tr><td>Configuration</td><td>HOTA</td><td>AssA</td><td>IDF1</td><td>IDSW</td></tr><tr><td>TrackTrack (HMIoU)</td><td>65.10</td><td>65.26</td><td>67.87</td><td>1962</td></tr><tr><td>CDA (α = 0.25)</td><td>66.66</td><td>67.46</td><td>70.41</td><td>1662</td></tr><tr><td>CDA (adaptive)</td><td>66.68</td><td>67.49</td><td>70.49</td><td>1652</td></tr></table>

frames takes 129.32 s, compared with 128.97 s for the baseline, an increase of 0.27%. The microbenchmark isolates motion-similarity computation for a single association call.

## F Dataset Documentation and Maintenance

Data sources and release. VastMAT is built from publicly available videos. Release materials will record sequence provenance, clip time intervals, and applicable licenses or permissions, providing media files or annotations with source indices as permitted. Code, annotations, and third-party videos will have separately stated licenses and attribution requirements.

Version maintenance. Each release will specify data and annotation versions, video lists, and evaluation configurations. Version histories will document annotation corrections, sequence removals, and split changes. If

Table 26: CDA inference overhead. The microbenchmark measures motion-similarity computation for one association call (µs, single-threaded).
<table><tr><td>Association matrix</td><td>Pure IoU (µs)</td><td>CDA (µs)</td><td>Increase</td></tr><tr><td>20×40</td><td>561.6</td><td>595.8</td><td> $+ 3 4 . 2 \ : ( + 6 \% )$ </td></tr><tr><td> $5 0 \times 1 0 0$ </td><td>3419.9</td><td>3534.3</td><td>+114.4 (+3%)</td></tr><tr><td>100 × 200</td><td>13256.7</td><td>13614.7</td><td>+358.0 (+3%)</td></tr></table>

source videos become unavailable, affected sequences will be recorded; evaluations on the remaining subset will state their actual scope.

Issue reporting. The project will provide a channel for annotation errors, access issues, and copyright or privacy concerns. Verified issues will prompt corrections, access restrictions, or removal, with effects on the data and evaluation scope documented.

Intended use. VastMAT supports research on animal detection, within-video identity association, and Cross-category generalization. Its public-video sources introduce biases in category and scene distributions. Users should interpret aggregate metrics alongside category coverage, animal-group results, and video-level performance, and further validate models in their intended application environments.
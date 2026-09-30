# UniBuild: Unified Building Mapping From Multi-Source Optical Remote Sensing Imagery With Detail Decoding and Geometry

Regularization

Wei Huang, Member, IEEE, Chenying Liu, Member, IEEE, Yilei Shi, Member, IEEE, and Xiao Xiang Zhu, Fellow, IEEE

Abstract—Building extraction from optical remote sensing (RS) imagery is fundamental to urban mapping, yet existing methods are often dataset-specific and generalize poorly to unseen domains. Their practical use is also limited by insufficient detail recovery and weak geometric regularization, leading to blurred boundaries, irregular shapes, and merged adjacent buildings. To address these issues, we propose UniBuild, a unified building extraction framework for multi-source RGB optical RS imagery. First, a unified multi-dataset training scheme is constructed over heterogeneous RGB optical datasets to learn transferable building representations across sensors and resolutions. Second, a novel detail-preserving HR-DPT decoder is designed to integrate high-level semantic features with highresolution spatial features, enhancing building detail recovery. Third, geometry-aware regularization is introduced through a structure-tensor-based direction-aware loss for boundary direction consistency and a saddle-aware loss for suppressing false activations in narrow inter-building gaps under low-resolution conditions. We train and evaluate UniBuild on multi-source RGB optical datasets, including 10 public high-resolution datasets and two self-collected low-resolution datasets. Experiments show that UniBuild consistently improves building-region accuracy, boundary sharpness, and adjacent-building separation across diverse datasets. It also generalizes well to unseen domains and supports practical building extraction from RGB optical RS imagery up to 10 m resolution. The predicted masks can be further converted into GIS-compatible building footprints through simple polygonization. The trained model and inference code are released at https://github.com/zhu-xlab/UniBuild.

Index Terms—building extraction, remote sensing, visual foundation model, boundary awareness, geometric supervision

## I. INTRODUCTION

Building extraction is a fundamental task in optical remote sensing (RS) image understanding, providing essential spatial information for urban planning, disaster assessment, and map updating. With the rapid accumulation of multi-source optical RS imagery and the development of deep learning, building extraction performance has improved significantly. However, most existing methods are still developed and evaluated on specific datasets, making them difficult to apply to diverse RS imagery in real-world scenarios. Therefore, developing a unified building extraction model for multi-source optical RS imagery is highly desirable. However, achieving this goal remains challenging due to the following key issues.

First, cross-sensor and cross-resolution generalization is limited. Optical RS imagery from different platforms exhibits large variations in spatial resolution, imaging conditions, and appearance distributions, while single public datasets usually cover only limited data diversity. Models trained on such datasets therefore tend to learn data-specific representations and degrade when transferred to unseen sensors or resolutions.

Second, HR detail decoding remains insufficient. Building extraction requires accurate recovery of boundaries, corners, and small structures. Although visual foundation models (VFMs), such as DINOv2 and DINOv3 [1, 2], capture strong global semantics, they are still limited in local detail representation and boundary recovery, often producing blurred boundaries and irregular building shapes.

![](images/f6a5bde20bbef46d99a7e8d8d0b87d728550da062fe80dd290aa464e90af16a6.jpg)  
Fig. 1. Key properties expected from UniBuild for unified building extraction across multi-source RGB optical RS imagery.

Third, building geometry regularization remains insufficient. CE and Dice losses mainly optimize pixel-wise accuracy and region overlap, but provide limited constraints on building-specific geometry. On the one hand, weak boundary direction constraints can lead to fragmented, zigzag, or locally inconsistent contours. On the other hand, under LR conditions, mixed pixels and over-smoothed features can obscure narrow inter-building gaps, making these saddle areas prone to false foreground activation and causing adjacent buildings to merge. Therefore, unified building extraction requires geometry-aware regularization that jointly improves boundary consistency and preserves LR inter-building gaps.

Based on these observations, we propose UniBuild, a unified building extraction framework for multi-source RGB optical RS imagery, instead of training separate models for different data sources. As shown in Fig. 1, UniBuild is designed to be sensor-agnostic, resolution-adaptive, detaildecoding, and geometry-aware. It is jointly trained on multiple public and private datasets covering diverse RGB optical sensors and spatial resolutions. To recover HR details, we design an HR-DPT decoder that fuses high-level semantic features with high-resolution shallow features. To enhance geometric modeling, we further introduce a structure-tensorbased direction-aware loss for boundary direction consistency and a saddle-aware loss for suppressing false activations in narrow inter-building gaps under LR conditions.

We train and evaluate UniBuild on 10 public HR RGB building extraction datasets and two self-collected LR RGB datasets derived from 4.8 m Planet and 10 m Sentinel-2 imagery. By integrating HR, MR, and LR imagery into a single framework, UniBuild establishes a practical unified building extraction model for RGB optical RS imagery up to 10 m resolution. Experimental results show that UniBuild achieves stable improvements across datasets, especially in boundary quality, and demonstrates stronger cross-sensor and crossresolution generalization than baselines using generic feature extraction models and conventional pixel-level losses.

In addition, we released the trained UniBuild model and inference code, which can be directly applied to arbitrary RGB optical RS imagery without additional training or fine-tuning. For imagery coarser than 1 m, only a simple upsampling step to 1 m GSD is required before inference. The pipeline also integrates optional polygonization, enabling users to convert RGB imagery into GIS-ready building footprint shapefiles for downstream geospatial applications.

In summary, the contributions of this work are as follows:

• We propose UniBuild, a unified building extraction framework for optical RS imagery that produces building masks and optional vectorized footprints. UniBuild achieves robust cross-sensor and cross-resolution generalization for RGB optical imagery up to 10 m.

• We design a novel HR-DPT decoder that combines high-level semantic features with high-resolution shallow details to improve HR detail decoding.

• We introduce a novel direction-aware loss that regularizes local boundary direction consistency across all resolutions and strengthens geometric relationship modeling.

• We propose a novel saddle-aware loss to suppress false activations in LR inter-building gaps and improve adjacent-building separability.

## II. RELATED WORK

## A. Building Extraction from Optical Remote Sensing Imagery

Building extraction from optical RS imagery is a core geospatial vision task for urban monitoring, disaster assessment, map updating, and infrastructure analysis [3]. Recent reviews further summarize building extraction from geometrical and semantic perspectives [4], while large-scale open building datasets highlight the need for globally complete and structurally rich building representations [5]. Early automatic methods mainly relied on handcrafted spectral, texture, geometric, and height cues, often combined with rule-based reasoning or probabilistic graphical models [6, 7].

Deep learning has advanced building extraction into endto-end semantic segmentation. CNN-based encoder–decoder architectures such as FCN [8], U-Net [9], and SegNet [10] established the foundation for many RS building extraction pipelines. FCN-style RS methods demonstrated strong potential for building delineation [11, 12], and related studies further explored joint road–building extraction [13], multimodal HR image–LiDAR fusion [14], cross-source adaptive fusion [15, 16], and graph-based structural refinement [17]. Subsequent works improved building extraction through guided filtering, attention mechanisms, body–boundary decomposition, and multimodal feature fusion [14, 18–20].

More recently, transformer-based methods and VFMs have brought new opportunities by providing stronger long-range dependency modeling and transferable representations. Sparse Token Transformers demonstrated the effectiveness of transformers for RS building extraction [21], while large-scale self-supervised VFMs such as DINOv2 [1] and DINOv3 [2] showed strong transferability across domains and dense prediction tasks. Building extraction has also been investigated under self-supervised, semi-supervised, weakly supervised, unsupervised domain adaptation, and vectorized settings [20, 22–27]. However, most existing methods are still developed and validated on one or only a few datasets, making them insufficient for a unified setting where one shared model must generalize across heterogeneous RGB optical RS imagery with different sensors and spatial resolutions.

## B. High-Resolution Detail Decoding for Dense Prediction

HR detail recovery remains a key challenge in dense prediction, as encoder downsampling weakens spatial cues for boundary reconstruction. Classical encoder–decoder architectures alleviate this issue through skip connections and multistage upsampling [8–10], while multi-scale designs improve contextual aggregation and feature fusion [28, 29]. In RS imagery, semantic–spatial refinement has also proven effective for geometry-sensitive tasks such as road extraction [30].

With transformer-based dense prediction, decoder design becomes crucial for converting semantically strong but spatially compressed features into fine-grained outputs. Seg-Former supports efficient dense prediction with hierarchical transformer features [31], while DPT reassembles transformer features into multi-scale image-like representations for progressive refinement [32]. These studies show that dense prediction quality depends heavily on how high-level semantics are decoded into HR spatial predictions.

This issue is especially important for building extraction, where sharp corners and thin boundaries must be preserved. Although VFMs such as DINOv2 [1] and DINOv3 [2] provide strong transferable semantics, their predictions are still decoded from spatially compressed embeddings, which may lead to blurred contours, incomplete structures, and missing small buildings, particularly in LR imagery. Therefore, unified building extraction requires a decoder that combines high-level semantic embeddings with stronger HR features for boundary recovery and local detail restoration.

## C. Geometry-Aware Regularization for Building Extraction

Geometry-aware regularization has gained attention for constraining object structures, especially in building extraction, where predictions require regular boundaries, plausible shapes, and clear separation between adjacent buildings. Existing methods can be broadly grouped into boundary-aware and structure-/topology-aware regularization.

Boundary-aware methods explicitly encourage better alignment between predicted and reference contours. Representative examples include Boundary Loss [33] and Active Boundary Loss [34]. In optical RS building extraction, boundary-assisted learning [35], conditional random fields [7, 36], dual spatialgraph refinement [37], and edge-aware refinement networks such as BEARNet [38] have further demonstrated the benefit of boundary modeling for improving building morphology. However, these methods mainly emphasize contour sharpness or boundary alignment, while direction consistency along building boundaries is less explicitly modeled.

Structure- and topology-aware methods aim to encode higher-level geometric priors into segmentation. For example, clDice [39] preserves connectivity by introducing topologyaware supervision. In building extraction, adversarial shape learning [40] and vectorized building outline modeling [27] further highlight the importance of regular shapes and structurally meaningful predictions. These studies show that building extraction requires supervision beyond independent pixels, as buildings should exhibit coherent edges, plausible geometry, and clear separation from neighboring objects.

Despite these advances, existing methods still insufficiently model local directions along building boundaries and pay limited attention to narrow inter-building gaps, especially under LR conditions where mixed pixels and over-smoothed representations can merge adjacent buildings. More broadly, robust cross-sensor and cross-resolution generalization, HR detail decoding, and geometry-aware regularization for boundary consistency and gap preservation remain underexplored. These limitations motivate UniBuild, a unified framework with resolution-adaptive representation, detail-preserving decoding, and complementary geometry-aware supervision.

## III. UNIBUILD

This section presents UniBuild from four aspects: unified input representation and VFM feature extraction, the proposed HR-DPT decoder, direction-aware boundary regularization, and saddle-aware LR gap suppression, as shown in Fig. 2.

## A. Unified Input Representation and VFM Feature Extraction

UniBuild aims to learn a shared building extraction model from multi-source RGB optical RS imagery with heterogeneous sensors and spatial resolutions. To make such inputs compatible, we first construct a unified input representation.

1) Resolution-aware input normalization and batch grouping: Input images are divided into HR and LR groups using 1,m GSD as the threshold, which balances building-detail preservation and computational cost. HR images are processed at their original resolution, while LR images are bilinearly upsampled to 1 m GSD. All images are then cropped into $5 1 2 \times 5 1 2$ patches. During training, each mini-batch contains samples from only one resolution group, enabling resolutionspecific optimization. Specifically, saddle-aware regularization is activated only for LR batches, while direction-aware regularization is applied to both HR and LR batches.

2) Unified augmentation and binary supervision: Let $\textbf { I } \in$ $\mathbb { R } ^ { 3 \times H \times W }$ denote the input RGB image and $\mathbf { Y } \in \left. 0 , 1 ^ { H \times W } \right.$ the binary building mask. During training, we apply weak geometric augmentation and strong intensity augmentation. Although the datasets differ in resolution, annotation quality, and label taxonomy, all annotations are converted into a unified binary label space of building and background.

3) VFM-based multi-level feature extraction: Given an input image I, DINOv3 is adopted as the VFM backbone $B ( \cdot )$ . The image is tokenized and processed to obtain four intermediate feature maps:

$$
\left\{ { \bf F } _ { 1 } , { \bf F } _ { 2 } , { \bf F } _ { 3 } , { \bf F } _ { 4 } \right\} = B ( { \bf I } ) .\tag{1}
$$

These non-hierarchical DINOv3 features share the same spatial resolution of $H / 1 6 \times W / 1 6$ while encoding multi-depth semantic cues. However, such spatial compression limits the recovery of fine building boundaries, corners, and small structures, motivating the proposed HR-DPT decoder.

## B. HR-DPT Decoder

The VFM backbone provides strong semantic representations but limited spatial details. To recover HR building details, we propose an HR-DPT decoder with an LR semantic branch and a semantics-guided HR shallow branch. The LR branch performs DPT-style top-down decoding [32], while the HR branch preserves fine image details and injects them into the semantic stream through gated residual fusion.

1) LR semantic feature branch: Given the backbone features

$$
\mathbf { F } _ { t } \in \mathbb { R } ^ { C _ { b } \times H / 1 6 \times W / 1 6 } , \quad t \in \{ 1 , 2 , 3 , 4 \} ,\tag{2}
$$

we use $1 \times 1$ projection and scale-specific resizing to construct a semantic pyramid:

$$
\mathbf { E } _ { t } = { \mathcal { R } } _ { t } \left( { \mathrm { C o n v } } _ { t } ^ { 1 \times 1 } ( \mathbf { F } _ { t } ) \right) , \quad t \in \{ 1 , 2 , 3 , 4 \} .\tag{3}
$$

Here, $\mathrm { C o n v } _ { t } ^ { 1 \times 1 } ( \cdot )$ maps $C _ { b }$ to $C _ { t }$ , and $\mathcal { R } _ { t } ( \cdot )$ aligns each feature to its pyramid scale:

$$
\begin{array} { r l } & { \quad \mathbf { E } _ { 1 } \in \mathbb { R } ^ { C _ { 1 } \times H / 4 \times W / 4 } , \ \mathbf { E } _ { 2 } \in \mathbb { R } ^ { C _ { 2 } \times H / 8 \times W / 8 } , } \\ & { \quad \mathbf { E } _ { 3 } \in \mathbb { R } ^ { C _ { 3 } \times H / 1 6 \times W / 1 6 } , \ \mathbf { E } _ { 4 } \in \mathbb { R } ^ { C _ { 4 } \times H / 3 2 \times W / 3 2 } . } \end{array}\tag{4}
$$

![](images/568584d8d67307b030827bff07dd2e511a3164e8c45db7ee39d2a2af3ed86b2e.jpg)  
Fig. 2. Overview of UniBuild. Multi-source RGB optical RS images are processed by a DINOv3 backbone and the proposed HR-DPT decoder, while direction-aware and saddle-aware losses regularize boundary orientation and LR inter-building gaps.

The four resizing operations use stride-4 and stride-2 transposed convolutions, identity mapping, and stride-2 convolution, respectively. However, these semantic features are still derived from spatially compressed VFM embeddings, so an additional HR shallow branch is introduced to recover fine spatial details.

2) HR shallow feature branch: An HR shallow feature is first extracted from the input image:

$$
\mathbf { S } ^ { ( 0 ) } = \psi ( \mathbf { I } ) , \quad \mathbf { S } ^ { ( 0 ) } \in \mathbb { R } ^ { C _ { s } \times H / 2 \times W / 2 } ,\tag{5}
$$

where $\psi ( \cdot )$ is a stride-2 convolution followed by a refinement convolution. For each semantic level $\mathbf { E } _ { t }$ , we project and upsample it to the HR feature resolution:

$$
\bar { \mathbf { E } } _ { t } = \left( \operatorname { C o n v } _ { t } ^ { 1 \times 1 } ( \mathbf { E } _ { t } ) \right) ^ { \uparrow } , \quad t \in \{ 1 , 2 , 3 , 4 \} ,\tag{6}
$$

where $( \cdot ) ^ { \uparrow }$ denotes bilinear upsampling to $H / 2 \times W / 2$ . The semantic feature is then fused with the HR feature as:

$$
\mathbf { S } ^ { ( t ) } = \mathbf { S } ^ { ( t - 1 ) } + f _ { t } \left( [ \mathbf { S } ^ { ( t - 1 ) } , \bar { \mathbf { E } } _ { t } ] \right) , \quad t \in \{ 1 , 2 , 3 , 4 \} ,\tag{7}
$$

where $[ \cdot , \cdot ]$ denotes concatenation and $f _ { t } ( \cdot )$ consists of two $3 \times 3$ convolutional layers. This progressively injects semantic cues into HR features while preserving fine spatial structures. From $\mathbf { S } ^ { ( 4 ) }$ , we build an HR feature pyramid:

$$
\begin{array} { r } { \mathbf { S } _ { 1 } = \mathbf { S } ^ { ( 4 ) } , ~ \mathbf { S } _ { 2 } = \mathcal { D } _ { 2 } ( \mathbf { S } _ { 1 } ) , ~ } \\ { \mathbf { S } _ { 3 } = \mathcal { D } _ { 3 } ( \mathbf { S } _ { 2 } ) , ~ \mathbf { S } _ { 4 } = \mathcal { D } _ { 4 } ( \mathbf { S } _ { 3 } ) . } \end{array}\tag{8}
$$

Here, $\mathcal { D } _ { 2 } ( \cdot ) , \ \mathcal { D } _ { 3 } ( \cdot )$ , and $\mathcal { D } _ { 4 } ( \cdot )$ are stride-2 downsampling blocks. The resulting scales are

$$
\begin{array} { r l } & { \mathbf { S } _ { 1 } \in \mathbb { R } ^ { C _ { s } \times H / 2 \times W / 2 } , \ \mathbf { S } _ { 2 } \in \mathbb { R } ^ { C _ { s } \times H / 4 \times W / 4 } , } \\ & { \mathbf { S } _ { 3 } \in \mathbb { R } ^ { C _ { s } \times H / 8 \times W / 8 } , \ \mathbf { S } _ { 4 } \in \mathbb { R } ^ { C _ { s } \times H / 1 6 \times W / 1 6 } . } \end{array}\tag{9}
$$

Before top-down decoding, each semantic feature is projected into the unified decoder space by a linear layer $\rho _ { t } \mathbf { : }$

$$
\tilde { \mathbf { E } } _ { t } = \rho _ { t } \big ( \mathbf { E } _ { t } \big ) , \quad t \in \{ 1 , 2 , 3 , 4 \} .\tag{10}
$$

The projected semantic features have the following scales:

$$
\begin{array} { r l } & { \tilde { \mathbf { E } } _ { 1 } \in \mathbb { R } ^ { C \times H / 4 \times W / 4 } , \tilde { \mathbf { E } } _ { 2 } \in \mathbb { R } ^ { C \times H / 8 \times W / 8 } , } \\ & { \tilde { \mathbf { E } } _ { 3 } \in \mathbb { R } ^ { C \times H / 1 6 \times W / 1 6 } , \tilde { \mathbf { E } } _ { 4 } \in \mathbb { R } ^ { C \times H / 3 2 \times W / 3 2 } . } \end{array}\tag{11}
$$

Here, C is the unified decoder channel dimension.

3) HR-enhanced semantic decoding: The semantic branch is decoded in a DPT-style top-down manner. Starting from the coarsest feature, each refinement block upsamples the decoded feature and fuses it with a finer semantic feature. To compensate for missing HR details, the matched HR shallow feature $\mathbf { S } _ { t }$ is injected after each refinement step.

Let $\bar { \mathbf { P } } _ { t }$ and $\mathbf { P } _ { t }$ denote the decoded features before and after HR injection, respectively:

$$
\begin{array} { r } { \bar { \mathbf { P } } _ { 4 } = \mathrm { R e f i n e } _ { 4 } ( \tilde { \mathbf { E } } _ { 4 } ) , \quad } \\ { \bar { \mathbf { P } } _ { t } = \mathrm { R e f i n e } _ { t } ( \mathbf { P } _ { t + 1 } , \tilde { \mathbf { E } } _ { t } ) , \quad t \in \{ 3 , 2 , 1 \} , } \end{array}\tag{12}
$$

where Refine denotes the t-th refinement block in DPT.

To control the contribution of HR details, a gate map is calculated from the decoded semantic feature and the matched HR shallow feature:

$$
\mathbf { A } _ { t } = \sigma \left( \operatorname { C o n v } _ { t } ^ { 1 \times 1 } \left( \left[ \bar { \mathbf { P } } _ { t } , \mathbf { S } _ { t } \right] \right) \right) , \quad t \in \{ 4 , 3 , 2 , 1 \} .\tag{13}
$$

The gated residual HR injection is formulated as

$$
\mathbf { P } _ { t } = \bar { \mathbf { P } } _ { t } + \left( 1 + \mathbf { A } _ { t } \right) \odot \boldsymbol { \phi } _ { t } ( \mathbf { S } _ { t } ) , \quad t \in \{ 4 , 3 , 2 , 1 \} .\tag{14}
$$

Here, $\phi _ { t } ( \cdot )$ projects $\mathbf { S } _ { t }$ to the decoder dimension C. This module supplements semantic features with spatially HR details.

Finally, the finest decoded feature ${ \bf P } _ { 1 }$ is passed to a prediction head H and upsampled to the original resolution:

$$
\mathbf { P } = \mathrm { S o f t m a x } \left( ( \mathcal { H } ( \mathbf { P } _ { 1 } ) ) ^ { \uparrow } \right) , \quad \mathbf { P } \in [ 0 , 1 ] ^ { 2 \times H \times W } ,\tag{15}
$$

where the two classes are background and building.

## C. Direction-aware Consistency

As shown in Fig. 3, to improve boundary regularity, we introduce a direction-aware loss that aligns structure-tensorbased orientation between the predictions and labels, which is imposed only around boundary-support regions.

![](images/2e53d2db524aa72f15e76fe276b574297eb5d2fd89f10217cb1cfbafe2526ed5.jpg)  
Fig. 3. Visualization of the proposed direction-aware loss.

1) Boundary-focused supervision region: Let ${ \textbf { Y } } \in$ $\{ 0 , 1 \} ^ { H \times W }$ be the ground-truth label map, with ignored pixels excluded. We convert it into a two-channel one-hot map:

$$
\mathbf { Y } ^ { \mathrm { o h } } \in \{ 0 , 1 \} ^ { 2 \times H \times W } .\tag{16}
$$

Since directional structure is meaningful mainly around class transitions, a boundary-support mask B is constructed by detecting local label variation:

$$
\mathbf { B } = \operatorname* { m a x } _ { c \in \{ 0 , 1 \} } \mathbb { I } \left( \mathrm { M a x P } ^ { 3 \times 3 } ( \mathbf { Y } _ { c } ^ { \mathrm { o h } } ) - \mathrm { M i n P } ^ { 3 \times 3 } ( \mathbf { Y } _ { c } ^ { \mathrm { o h } } ) > 0 \right) .\tag{17}
$$

2) Structure-tensor-based direction representation: For the prediction branch, we use the probability map $\mathrm { ~ { ~ \bf ~ P ~ } ~ } \in { }$ $[ 0 , 1 ] ^ { 2 \times H \times W }$ . For the target branch, the one-hot label map is smoothed by a $3 \times 3$ average filter:

$$
\mathbf { Y } ^ { \mathrm { s m } } = \mathrm { A v g P o o l } ^ { 3 \times 3 } \left( \mathbf { Y } ^ { \mathrm { o h } } \right) .\tag{18}
$$

Let U denote either P or $\mathbf { Y } ^ { \mathrm { s m } }$ . Class-wise Sobel gradients are computed as

$$
U _ { x , c } = K _ { x } * U _ { c } , \quad U _ { y , c } = K _ { y } * U _ { c } , \quad c \in \{ 0 , 1 \} ,\tag{19}
$$

where

$$
K _ { x } = { \left[ \begin{array} { l l l } { - 1 } & { 0 } & { 1 } \\ { - 2 } & { 0 } & { 2 } \\ { - 1 } & { 0 } & { 1 } \end{array} \right] } , \quad K _ { y } = { \left[ \begin{array} { l l l } { - 1 } & { - 2 } & { - 1 } \\ { 0 } & { 0 } & { 0 } \\ { 1 } & { 2 } & { 1 } \end{array} \right] } .\tag{20}
$$

The structure tensor components are

$$
\begin{array} { c l } { { \displaystyle { \cal J } _ { x x } ( { \bf U } ) = \sum _ { c = 0 } ^ { 1 } U _ { x , c } ^ { 2 } , ~ { \cal J } _ { y y } ( { \bf U } ) = \sum _ { c = 0 } ^ { 1 } U _ { y , c } ^ { 2 } , } } \\ { { { \cal J } _ { x y } ( { \bf U } ) = \displaystyle \sum _ { c = 0 } ^ { 1 } U _ { x , c } U _ { y , c } . } } \end{array}\tag{21}
$$

We then use a double-angle representation to obtain a normalized orientation descriptor:

$$
\mathbf { O } ( \mathbf { U } ) = \frac { \left( J _ { x x } ( \mathbf { U } ) - J _ { y y } ( \mathbf { U } ) , 2 J _ { x y } ( \mathbf { U } ) \right) } { \sqrt { \left( J _ { x x } ( \mathbf { U } ) - J _ { y y } ( \mathbf { U } ) \right) ^ { 2 } + \left( 2 J _ { x y } ( \mathbf { U } ) \right) ^ { 2 } + \varepsilon } } .\tag{22}
$$

where $\varepsilon$ is a small positive constant, preventing division by zero. The corresponding orientation energy is

$$
\mathbf E ( \mathbf U ) = \sqrt { ( J _ { x x } ( \mathbf U ) - J _ { y y } ( \mathbf U ) ) ^ { 2 } + ( 2 J _ { x y } ( \mathbf U ) ) ^ { 2 } + \varepsilon } .\tag{23}
$$

3) Direction-aware boundary regularization: Prediction and target orientation descriptors are computed as

$$
{ \bf O } _ { p } = { \bf O } ( { \bf P } ) , \quad { \bf O } _ { y } = { \bf O } ( { \bf Y } ^ { \mathrm { s m } } ) .\tag{24}
$$

The normalized target orientation energy is

$$
\hat { \mathbf { E } } _ { y } = \frac { \mathbf { E } ( \mathbf { Y } ^ { \mathrm { s m } } ) } { \operatorname* { m a x } \mathbf { E } ( \mathbf { Y } ^ { \mathrm { s m } } ) + \varepsilon } ,\tag{25}
$$

which is combined with the boundary-support mask:

$$
\mathbf { W } _ { \mathrm { d i r } } = \mathbf { B } \odot \hat { \mathbf { E } } _ { y } .\tag{26}
$$

The direction-aware loss is

$$
\mathcal { L } _ { \mathrm { d i r } } = \frac { \sum _ { u , v } \mathbf { W } _ { \mathrm { d i r } } ( u , v ) \left[ 1 - \langle \mathbf { O } _ { p } ( u , v ) , \mathbf { O } _ { y } ( u , v ) \rangle \right] } { \sum _ { u , v } \mathbf { W } _ { \mathrm { d i r } } ( u , v ) + \varepsilon } ,\tag{27}
$$

where $\langle \cdot , \cdot \rangle$ is inner product between normalized orientation vectors. Better orientation alignment yields a smaller loss.

## D. Saddle-aware Suppression

While the direction-aware loss improves boundary orientation consistency, LR imagery still struggles to separate adjacent buildings. Therefore, we introduce a saddle-aware suppression loss to penalize false positives in building gaps.

1) Saddle mask construction: Let $\mathbf { Y } \in \{ 0 , 1 \} ^ { H \times W }$ denote the binary label map. Foreground and background masks are

$$
\mathbf { Y } _ { \mathrm { f g } } = \mathbb { I } ( \mathbf { Y } = 1 ) , \quad \mathbf { Y } _ { \mathrm { b g } } = \mathbb { I } ( \mathbf { Y } = 0 ) .\tag{28}
$$

A saddle pixel is defined as a background pixel supported by building pixels from two opposite directions. We detect such pixels using eight fixed $5 \times 5$ directional kernels for horizontal, vertical, and diagonal directions. For direction $d ,$ the foreground support response is

$$
R _ { d } ( u , v ) = \sum _ { ( \Delta u , \Delta v ) \in \Omega _ { d } } \mathbf { Y } _ { \mathrm { f g } } ( u + \Delta u , v + \Delta v ) ,\tag{29}
$$

where $\Omega _ { d }$ contains two offsets along direction d. For example,

$$
\Omega _ { \mathrm { l } } = \{ ( 0 , - 2 ) , ( 0 , - 1 ) \} , \quad \Omega _ { \mathrm { r } } = \{ ( 0 , 1 ) , ( 0 , 2 ) \} .\tag{30}
$$

The support indicator is

$$
H _ { d } ( u , v ) = \mathbb { I } \big ( R _ { d } ( u , v ) > \tau _ { \mathrm { s a d } } \big ) ,\tag{31}
$$

where $\tau _ { \mathrm { s a d } } = 0 . 5$ . Opposite-side support is then checked by

$$
\begin{array} { r l r } & { } & { C _ { \mathrm { l r } } = H _ { 1 } \wedge H _ { \mathrm { r } } , \ C _ { \mathrm { u d } } = H _ { \mathrm { u } } \wedge H _ { \mathrm { d } } , ~ } \\ & { } & { C _ { \mathrm { d i a g 1 } } = H _ { \mathrm { u l } } \wedge H _ { \mathrm { d r } } , \ C _ { \mathrm { d i a g 2 } } = H _ { \mathrm { u r } } \wedge H _ { \mathrm { d l } } . } \end{array}\tag{32}
$$

The final saddle mask is

$$
\mathbf { M } _ { \mathrm { s a d } } ( u , v ) = \mathbf { Y } _ { \mathrm { b g } } ( u , v ) \cdot \mathbb { I } \left( C _ { \mathrm { l r } } \vee C _ { \mathrm { u d } } \vee C _ { \mathrm { d i a g 1 } } \vee C _ { \mathrm { d i a g 2 } } \right)\tag{33}
$$

Thus, $\mathbf { M } _ { \mathrm { s a d } }$ selects background pixels between nearby buildings and is used only as a non-gradient spatial weighting mask.

2) Saddle-weighted Dice loss: Let $\mathbf { P } _ { \mathrm { f g } } \in [ 0 , 1 ] ^ { H \times W }$ denote the predicted building probability. For resolution group $r \in$ {HR, LR}, the false-positive weight is

$$
\mathbf { W } _ { \mathrm { f p } } ^ { ( r ) } = 1 + \alpha _ { r } \mathbf { M } _ { \mathrm { s a d } } ,\tag{34}
$$

where $\alpha _ { \mathrm { H R } } = 0$ and $\alpha _ { \mathrm { L R } } > 0 .$ . The Dice terms are

$$
\begin{array} { c } { T P = \displaystyle \sum \mathbf { P } _ { \mathrm { f g } } \mathbf { Y } _ { \mathrm { f g } } , } \\ { F P ^ { ( r ) } = \displaystyle \sum \mathbf { P } _ { \mathrm { f g } } \mathbf { Y } _ { \mathrm { b g } } \mathbf { W } _ { \mathrm { f p } } ^ { ( r ) } , } \\ { F N = \displaystyle \sum ( 1 - \mathbf { P } _ { \mathrm { f g } } ) \mathbf { Y } _ { \mathrm { f g } } . } \end{array}\tag{35}
$$

Only the false-positive term is reweighted, because the goal is to suppress false building activations in background gaps. The saddle-aware loss is

$$
\mathcal { L } _ { \mathrm { s a d } } ^ { ( r ) } = 1 - \frac { 2 T P + \varepsilon } { 2 T P + F P ^ { ( r ) } + F N + \varepsilon } .\tag{36}
$$

When $\alpha _ { \mathrm { H R } } = 0 ;$ , this reduces to standard soft Dice for HR samples; for LR samples, $\alpha _ { \mathrm { L R } } > 0$ explicitly penalizes false activations in narrow inter-building gaps.

## E. Overall Objective

The final training objective is

$$
\begin{array} { r } { \mathcal { L } ^ { ( r ) } = \mathcal { L } _ { \mathrm { c e } } + \lambda _ { \mathrm { d i r } } \mathcal { L } _ { \mathrm { d i r } } + \mathcal { L } _ { \mathrm { s a d } } ^ { ( r ) } , \quad r \in \{ \mathrm { H R } , \mathrm { L R } \} . } \end{array}\tag{37}
$$

Here two hyper-parameters are required, $\lambda _ { \mathrm { d i r } }$ for directionaware boundary regularization and $\alpha _ { \mathrm { L R } }$ for LR saddle-aware suppression, with $\alpha _ { \mathrm { H R } } ~ = ~ 0$ by default. Overall, directionaware loss regularizes boundary orientation consistency across all resolutions, while saddle-aware loss strengthens interbuilding gap preservation under LR conditions.

## IV. EXPERIMENTS

## A. Training Datasets and Experimental Settings

a) Training Datasets: To train UniBuild across different spatial resolutions, sensors, and geographic domains, we collect public and private building extraction datasets and unify their annotations into a binary label space of building and background, where unlabeled or invalid pixels are ignored. The training datasets cover very-high-resolution aerial imagery, high-resolution satellite imagery, and mediumresolution satellite imagery. HR datasets include Potsdam [41], INRIA [42], Alabama [43], OpenEarthMap [44], LoveDA [45], GF-7 [46], WHU-Mix [47], SpaceNet2 [48], Land-Cover.ai [49], and ORBITaL-Net [50]. In addition, we construct two large-scale building extraction datasets from Planet imagery [5, 51] and Sentinel-2 (ST-2) imagery [52] over globally distributed urban areas, with building annotations derived from OpenStreetMap (OSM). Although OSM-derived labels contain inevitable noise, including missing buildings, outdated annotations, and image–label misalignment, they provide valuable large-scale supervision for geographically diverse LR scenarios. Overall, the training data span 0.05 m to 10 m GSD, enabling UniBuild to learn building representations across substantially different object scales and imaging conditions, which are summarized in Table I.

b) Experimental Settings: For a fair comparison, all competing models are trained on the same single-dataset splits and evaluated using the same validation protocol, image size, data augmentation, optimizer settings, and metrics. All the models use a 512 × 512 input crop size. Each model is trained for 50 epochs using AdamW, with a base learning rate of $5 \times 1 0 ^ { - 6 }$ , a decoder learning-rate multiplier of $^ { 1 0 , }$ and a weight decay of 0.01. The batch size is set to 10. For multidataset training, datasets are grouped according to their spatial resolutions. For both HR and LR images, we first apply weak geometric augmentations, including random scaling, cropping, and flipping, followed by stronger intensity augmentations, such as color jittering, Gaussian blurring, sharpening, and grayscale conversion. We evaluate semantic building extraction using building IoU and F1 score. Since boundary quality is particularly important for building footprint extraction, we further report Boundary-IoU and Boundary-F1. The best checkpoint is selected based on the average of IoU and Boundary-IoU, jointly considering region accuracy and boundary quality.

## B. Comparison With SOTA Building Extraction Models

To assess the effectiveness of UniBuild, we compare it with representative general semantic segmentation models and recent building extraction models, using the hyperparameters specified in Sec. IV-D. The general segmentation baselines include U-Net [9] with a ResNet-34 backbone [53], DeepLabV3+ [54] with a ResNet-50 backbone [53], and SegFormer [31] with a MiT-B2 backbone. The buildingspecific baselines include BuildFormer [55] with a Swin-T backbone [56], BIENet [57] with a ResNet-50 backbone, and BOMSC-Net [58] with its original backbone design. We further include DINOv3-B-DPT as a strong foundation-model baseline, which uses the same DINOv3-Base backbone [2] as UniBuild but adopts a standard DPT decoder and CE loss. UniBuild is evaluated as a complete framework, including the proposed HR-DPT decoder and geometry-aware regularization losses. All comparison models are trained with CE loss under their standard segmentation settings. The UniBuild results in Table II are obtained from single-dataset checkpoints for each dataset, rather than from the multi-dataset joint training.

As shown in Table II, DINOv3-B-DPT already substantially outperforms most conventional segmentation models and building-specific architectures, demonstrating the strong representation ability of the pretrained visual foundation backbone. Built upon the same DINOv3-Base backbone, UniBuild further achieves the best overall performance across the four representative datasets, with clear gains in both region-level and boundary-aware metrics. The improvements in B-IoU and B-F1 are particularly important, as accurate boundaries are essential for downstream tasks such as building polygonization. These results indicate that the performance gains of UniBuild come not only from the foundation backbone, but also from the proposed HR-DPT decoder and geometryaware regularization. The isolated effects of these two components are further analyzed in Sec. IV-D. The qualitative examples in Fig. 4 further support these findings, showing that UniBuild produces more complete masks, sharper boundaries, and clearer separation between adjacent buildings, especially for small buildings, dense layouts, and LR imagery.

TABLE I  
TRAINING DATASETS IN OUR EXPERIMENTS. ALL DATASETS ARE CONVERTED TO BINARY BUILDING EXTRACTION LABELS.
<table><tr><td>Dataset</td><td>Region</td><td>Type</td><td>Sensor/source</td><td>GSD</td><td>Patch size</td><td>Train</td><td>Val</td><td>Test</td><td>Split ratio</td></tr><tr><td>Potsdam</td><td>Germany</td><td>Aerial</td><td>Orthophoto RGB</td><td>0.05 m</td><td> $5 1 2 \times 5 1 2$ </td><td>4,377</td><td>547</td><td>548</td><td>8:1:1</td></tr><tr><td>INRIA</td><td>Europe/USA</td><td>Aerial</td><td>Orthophoto RGB</td><td>0.30 m</td><td> $5 1 2 \times 5 1 2$ </td><td>14,400</td><td>1,800</td><td>1,800</td><td>8:1:1</td></tr><tr><td>LoveDA</td><td>China</td><td>Mixed</td><td>Google Earth</td><td>0.30 m</td><td> $5 1 2 \times 5 1 2$ </td><td>10,088</td><td>3,338</td><td>3,338</td><td>3:1:1</td></tr><tr><td>SpaceNet2</td><td>Multi-city</td><td>Satellite</td><td>WorldView-3</td><td>0.30 m</td><td>512 × 512</td><td>8,474</td><td>1,059</td><td>1,060</td><td>8:1:1</td></tr><tr><td>LandCover.ai</td><td>Poland</td><td>Aerial</td><td>Orthophoto RGB</td><td>0.30 m</td><td>512 × 512</td><td>8,539</td><td>1,067</td><td>1,068</td><td>8:1:1</td></tr><tr><td>OEM</td><td>Global</td><td>Mixed</td><td>Aerial/satellite/UAV, multi-source</td><td>0.25-0.5m</td><td>512 × 512</td><td>7,461</td><td>932</td><td>934</td><td>8:1:1</td></tr><tr><td>ORBITaLNet</td><td>Global</td><td>Satellite</td><td>Maxar VHR, mainly WorldView-2/3</td><td>0.47 m</td><td>512 × 512</td><td>115,443</td><td>6,413</td><td>6,414</td><td>18:1:1</td></tr><tr><td>Alabama</td><td>USA</td><td>Satellite</td><td>Bing Maps</td><td>0.50 m</td><td>512 × 512</td><td>32,640</td><td>4,080</td><td>4,080</td><td>8:1:1</td></tr><tr><td>WHU-Mix</td><td>New Zealand</td><td>Aerial</td><td>LINZ aerial imagery</td><td>0.50m</td><td>512 × 512</td><td>39,346</td><td>4,381</td><td>8,402</td><td>9:1:2</td></tr><tr><td>GF-7</td><td>China</td><td>Satellite</td><td>GaoFen-7</td><td>0.65 m</td><td>512 × 512</td><td>3,106</td><td>1,034</td><td>1,035</td><td>3:1:1</td></tr><tr><td>Planet</td><td>Global</td><td>Satellite</td><td>PlanetScope</td><td>4.80 m</td><td> $5 1 2 \times 5 1 2$ </td><td>80,000</td><td>10,000</td><td>10,000</td><td>8:1:1</td></tr><tr><td>Sentinel-2</td><td>Global</td><td>Satellite</td><td>Sentinel-2 RGB</td><td>10m</td><td>512 × 512</td><td>40,000</td><td>5,000</td><td>5,000</td><td>8:1:1</td></tr></table>

![](images/cc2924a15426c2460e41dc43ff634c875070c32aeaea642202a353834247002c.jpg)  
Fig. 4. Qualitative comparison with competing models, DINOv3-B-DPT, and UniBuild. UniBuild uses the corresponding single-dataset trained checkpoint for each dataset. Rows 1–4 show examples from INRIA, WHU-Mix, GF-7, and Planet (bilinearly upsampled to 1 m), respectively. For each row, the columns present the RGB image, predictions from different methods, and the ground-truth building label.

## C. Single-dataset vs. Multi-dataset Training

Two settings are compared: (1) single-dataset DINOv3-B-DPT trained independently on each dataset with CE loss; and (2) UniBuild, jointly trained on all 12 datasets with our HR-DPT decoder and geometry-aware regularizations. As shown in Table IV, UniBuild improves the average IoU and B-IoU across the 12 datasets from 74.03 and 61.06 to 76.27 and 65.60, respectively, demonstrating that UniBuild consistently enhances both region-level accuracy and boundary quality compared with independently trained DINOv3-B-DPT models.

We further evaluate the cross-dataset transferability of single-dataset DINOv3-B-DPT models and compare them with the multi-dataset UniBuild in Fig. 5. UniBuild achieves positive gains across all datasets, indicating that multi-dataset joint training does not compromise dataset-specific performance but enables one unified model to maintain consistently strong results across multi-source RS images. These improvements can be attributed to both the proposed architecture and the diversity of multi-dataset training. The HR-DPT decoder and geometry aware losses enhance detail recovery and boundary regularity, while multi-source datasets introduce broader scale, scene, and annotation variations. Such diversity reduces reliance on dataset-specific object sizes and background co-occurrence patterns, improves robustness to boundary ambiguity and label inconsistency, and helps the foundation backbone preserve more transferable building representations.

TABLE II  
COMPARISON WITH SOME REPRESENTATIVE BUILDING EXTRACTIONMODELS AND UNIBUILD UNDER SINGLE-DATASET TRAINING.
<table><tr><td>Dataset</td><td>Method</td><td>IoU</td><td>F1</td><td>B-IoU</td><td>B-F1</td><td>MIoU</td><td>MF1</td></tr><tr><td rowspan="9">INRIA</td><td>U-Net</td><td>75.31</td><td>85.92</td><td>51.59</td><td>55.04</td><td>63.45</td><td>70.48</td></tr><tr><td>DeepLabV3+</td><td>75.92</td><td>86.31</td><td>50.71</td><td>53.88</td><td>63.32</td><td>70.10</td></tr><tr><td>BuildFormer</td><td>77.59</td><td>87.38</td><td>55.34</td><td>58.52</td><td>66.46</td><td>72.95</td></tr><tr><td>SegFormer</td><td>77.52</td><td>87.34</td><td>53.03</td><td>56.16</td><td>65.27</td><td>71.75</td></tr><tr><td>BIENet</td><td>76.93</td><td>86.96</td><td>55.11</td><td>58.46</td><td>66.02</td><td>72.71</td></tr><tr><td>BOMSC-Net</td><td>76.31</td><td>86.57</td><td>53.18</td><td>56.45</td><td>64.75</td><td>71.51</td></tr><tr><td>DINOv3-B-DPT</td><td>82.17</td><td>90.21</td><td>66.91</td><td>70.04</td><td>74.54</td><td>80.12</td></tr><tr><td>UniBuild</td><td>83.29</td><td>90.88</td><td>70.86</td><td>74.09</td><td>77.08</td><td>82.49</td></tr><tr><td>U-Net</td><td>63.14</td><td>77.41</td><td>53.01</td><td>56.78</td><td>58.08</td><td>67.09</td></tr><tr><td rowspan="8">GF-7</td><td></td><td></td><td></td><td></td><td></td><td></td><td>65.66</td></tr><tr><td>DeepLabV3+</td><td>63.41</td><td>77.61</td><td>50.70</td><td>53.70</td><td>57.05</td><td></td></tr><tr><td>BuildFormer</td><td>69.04 68.33</td><td>81.68</td><td>60.12</td><td>63.16</td><td>64.58</td><td>72.42</td></tr><tr><td>SegFormer</td><td>66.98</td><td>81.19</td><td>56.48</td><td>59.55</td><td>62.40</td><td>70.37</td></tr><tr><td>BIENet</td><td>65.81</td><td>80.23</td><td>58.71</td><td>62.03</td><td>62.85</td><td>71.13</td></tr><tr><td>BOMSC-Net</td><td>76.27</td><td>79.38</td><td>56.04</td><td>59.28</td><td>60.93</td><td>69.33</td></tr><tr><td>DINOv3-B-DPT UniBuild</td><td>78.68</td><td>86.54 88.07</td><td>71.10 75.76</td><td>74.01 79.04</td><td>73.69 77.22</td><td>80.28 83.56</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="8">WHU-Mix</td><td>U-Net</td><td>78.49</td><td>87.95</td><td>55.83</td><td>59.58</td><td>67.16</td><td>73.76</td></tr><tr><td>DeepLabV3+</td><td>79.48</td><td>88.57</td><td>56.42</td><td>59.79</td><td>67.95</td><td>74.18</td></tr><tr><td>BuildFormer</td><td>81.03</td><td>89.52</td><td>61.43</td><td>64.83</td><td>71.23</td><td>77.18</td></tr><tr><td>SegFormer</td><td>80.70</td><td>89.32</td><td>59.44</td><td>62.84</td><td>70.07</td><td>76.08</td></tr><tr><td>BIENet</td><td>79.93</td><td>88.85</td><td>60.99</td><td>64.62</td><td>70.46</td><td>76.73</td></tr><tr><td>BOMSC-Net</td><td>79.21</td><td>88.40</td><td>58.22</td><td>61.87</td><td>68.71</td><td>75.13</td></tr><tr><td>DINOv3-B-DPT</td><td>84.64</td><td>91.68</td><td>70.17</td><td>73.51</td><td>77.41</td><td>82.59</td></tr><tr><td>UniBuild</td><td>85.26</td><td>92.05</td><td>70.96</td><td>74.42</td><td>78.11</td><td>83.23</td></tr><tr><td rowspan="8">Planet</td><td>U-Net</td><td>37.84</td><td>54.90</td><td>35.57</td><td>37.08</td><td>36.70</td><td>45.99</td></tr><tr><td>DeepLabV3+</td><td>39.99</td><td>57.13</td><td>34.92</td><td>36.46</td><td>37.46</td><td>46.80</td></tr><tr><td>BuildFormer</td><td>40.30</td><td>57.45</td><td>35.77</td><td>37.23</td><td>38.04</td><td>47.34</td></tr><tr><td>SegFormer</td><td>42.05</td><td>59.21</td><td>36.61</td><td>38.10</td><td>39.33</td><td>48.65</td></tr><tr><td>BIENet</td><td>38.34</td><td>55.43</td><td>34.51</td><td>35.91</td><td>36.43</td><td>45.67</td></tr><tr><td>BOMSC-Net</td><td>38.66</td><td>55.77</td><td>34.49</td><td>35.98</td><td>36.58</td><td>45.87</td></tr><tr><td>DINOv3-B-DPT</td><td>45.35</td><td>62.40</td><td>42.02</td><td>43.61</td><td>43.69</td><td>53.01</td></tr><tr><td>UniBuild</td><td>48.25</td><td>65.10</td><td>47.35</td><td>49.30</td><td>47.80</td><td>57.20</td></tr></table>

TABLE III

MODEL COMPLEXITY COMPARISON UNDER A 512 × 512 RGB INPUT. PM AND GP DENOTE THE NUMBER OF TRAINABLE PARAMETERS AND GFLOPS FOR THE ENTIRE MODEL, RESPECTIVELY.
<table><tr><td>Method</td><td>PM (M)</td><td>GP (G)</td></tr><tr><td>U-Net</td><td>24.44</td><td>31.29</td></tr><tr><td>DeepLabV3+</td><td>26.68</td><td>36.57</td></tr><tr><td>BuildFormer</td><td>32.90</td><td>81.72</td></tr><tr><td>SegFormer</td><td>27.35</td><td>56.58</td></tr><tr><td>BIENet</td><td>32.58</td><td>37.89</td></tr><tr><td>BOMSC-Net</td><td>26.75</td><td>32.85</td></tr><tr><td>DINOv3-B-DPT</td><td>96.62</td><td>120.78</td></tr><tr><td>UniBuild / DINOv3-B-HR-DPT</td><td>97.68</td><td>160.57</td></tr></table>

## D. Ablation Study

1) Overall Module Contribution: We first provide an overall ablation study on two HR datasets, INRIA and GF-7, and two MR/LR datasets, Planet and ST-2. All variants use the same DINOv3-Base backbone and follow identical data splits and training protocol. DPT serves as the baseline decoder, HR-DPT denotes the proposed decoder without geometry-aware regularization, and HR-DPT + Dir + Sad denotes the full UniBuild configuration. For direction-aware regularization, we set $\lambda _ { \mathrm { d i r } } = 0 . 5 .$ . For saddle-aware regularization, we adopt the resolution-adaptive strategy in Sec. III-D, where the saddle penalty is disabled for HR batches and set to $\alpha _ { \mathrm { L R } } = 1 5$ for LR batches. As summarized in Table V, HR-DPT improves both region and boundary metrics over DPT, confirming the benefit of HR detail recovery from spatially compressed VFM features. Direction-aware regularization further improves boundary regularity, while saddle-aware suppression brings clear gains on LR datasets by reducing false-positive connections between adjacent buildings. In particular, $\mathbf { M } _ { \mathrm { I o U } }$ increases from 45.77 to 47.80 on Planet and from 26.97 to 33.60 on ST-2 after adding saddle-aware suppression. Detailed analyses of each component are provided in the following subsections.

TABLE IV  
COMPARISON BETWEEN SINGLE-DATASET DINOV3-B-DPT MODELS ANDUNIBUILD. “B-IOU” AND “B-F1” DENOTE BOUNDARY-IOU ANDBOUNDARY-F1, RESPECTIVELY. $\mathbf { \hat { \Pi } } ^ { * } \mathbf { M } _ { \mathrm { I o U } } \mathbf { \Sigma } ^ { * }$ IS THE MEAN OF IOU ANDB-IOU, WHILE $\mathbf { \ddot { \Phi } M } _ { \mathrm { F 1 } } \mathbf { \vec { \Phi } } , \mathbf { \vec { \Phi } }$ IS THE MEAN OF F1 AND B-F1.
<table><tr><td>Dataset</td><td>Method</td><td>IoU</td><td>F1</td><td>B-IoU</td><td>B-F1</td><td> $\scriptstyle \mathbf { M } _ { \mathrm { I o U } }$ </td><td>MF1</td></tr><tr><td rowspan="2">Potsdam</td><td>DINOv3-B-DPT</td><td>90.63</td><td>95.11</td><td>51.56</td><td>54.85</td><td>71.10</td><td>74.98</td></tr><tr><td>UniBuild</td><td>93.04</td><td>96.39</td><td>60.79</td><td>64.80</td><td>76.92</td><td>80.59</td></tr><tr><td rowspan="2">INRIA</td><td>DINOv3-B-DPT</td><td>82.17</td><td>90.21</td><td>66.91</td><td>70.04</td><td>74.54</td><td>80.12</td></tr><tr><td>UniBuild</td><td>83.34</td><td>90.92</td><td>69.30</td><td>72.71</td><td>76.32</td><td>81.81</td></tr><tr><td rowspan="2">Alabama</td><td>DINOv3-B-DPT</td><td>79.01</td><td>88.27</td><td>80.42</td><td>83.09</td><td>79.72</td><td>85.68</td></tr><tr><td>UniBuild</td><td>80.51</td><td>89.20</td><td>82.94</td><td>85.54</td><td>81.72</td><td>87.37</td></tr><tr><td rowspan="2">OEM</td><td>DINOv3-B-DPT</td><td>83.03</td><td>90.73</td><td>74.83</td><td>77.95</td><td>78.93</td><td>84.34</td></tr><tr><td>UniBuild</td><td>84.22</td><td>91.43</td><td>78.07</td><td>81.54</td><td>81.14</td><td>86.49</td></tr><tr><td rowspan="2">WHU-Mix</td><td>DINOv3-B-DPT</td><td>84.64</td><td>91.68</td><td>70.17</td><td>73.51</td><td>77.41</td><td>82.59</td></tr><tr><td>UniBuild</td><td>85.42</td><td>92.13</td><td>71.18</td><td>74.58</td><td>78.30</td><td>83.35</td></tr><tr><td rowspan="2">GF-7</td><td>DINOv3-B-DPT</td><td>76.27</td><td>86.54</td><td>71.10</td><td>74.01</td><td>73.69</td><td>80.28</td></tr><tr><td>UniBuild</td><td>80.23</td><td>89.03</td><td>78.03</td><td>81.32</td><td>79.13</td><td>85.17</td></tr><tr><td rowspan="2">LoveDA</td><td>DINOv3-B-DPT</td><td>71.00</td><td>83.04</td><td>30.29</td><td>32.69</td><td>50.64</td><td>57.87</td></tr><tr><td>UniBuild</td><td>71.64</td><td>83.48</td><td>35.92</td><td>38.64</td><td>53.78</td><td>61.06</td></tr><tr><td rowspan="2">SpaceNet2</td><td>DINOv3-B-DPT</td><td>79.69</td><td>88.70</td><td>66.19</td><td></td><td></td><td>78.97</td></tr><tr><td>UniBuild</td><td>81.54</td><td>89.83</td><td>71.38</td><td>69.24 74.46</td><td>72.94 76.46</td><td>82.14</td></tr><tr><td rowspan="2">LandCover.ai</td><td>DINOv3-B-DPT</td><td>84.14</td><td>91.39</td><td>77.64</td><td>80.09</td><td>80.89</td><td>85.74</td></tr><tr><td>UniBuild</td><td>87.31</td><td>93.23</td><td>86.40</td><td>88.51</td><td>86.86</td><td>90.87</td></tr><tr><td rowspan="2">ORBITaLNet</td><td>DINOv3-B-DPT</td><td>85.39</td><td>92.12</td><td>80.02</td><td>83.40</td><td>82.70</td><td>87.76</td></tr><tr><td>UniBuild</td><td>85.60</td><td>92.24</td><td>80.76</td><td>84.02</td><td>83.18</td><td>88.13</td></tr><tr><td rowspan="2">Planet</td><td>DINOv3-B-DPT</td><td>45.34</td><td>62.40</td><td>42.02</td><td>43.61</td><td>43.68</td><td>53.00</td></tr><tr><td>UniBuild</td><td>47.91</td><td>64.78</td><td>43.91</td><td>45.84</td><td>45.91</td><td>55.31</td></tr><tr><td rowspan="2">ST-2</td><td>DINOv3-B-DPT</td><td>27.04</td><td>43.04</td><td>21.57</td><td>22.12</td><td>24.31</td><td>32.58</td></tr><tr><td>UniBuild</td><td>34.52</td><td>51.32</td><td>28.56</td><td>29.73</td><td>31.54</td><td>40.52</td></tr><tr><td rowspan="2">Average</td><td>DINOv3-B-DPT</td><td>74.03</td><td>83.60</td><td>61.06</td><td>63.72</td><td>67.54</td><td>73.66</td></tr><tr><td>UniBuild</td><td>76.27</td><td>85.33</td><td>65.60</td><td>68.47</td><td>70.94</td><td>76.90</td></tr></table>

TABLE V  
OVERALL ABLATION STUDY OF SINGLE-DATASET UNIBUILD.
<table><tr><td>Dataset</td><td>Method</td><td>IoU</td><td>F1</td><td>B-IoU</td><td>B-F1</td><td> $\scriptstyle \mathbf { M } _ { \mathrm { I o U } }$ </td><td>MF1</td></tr><tr><td rowspan="4">INRIA</td><td>DPT</td><td>82.17</td><td>90.21</td><td>66.91</td><td>70.04</td><td>74.54</td><td>80.13</td></tr><tr><td>HR-DPT</td><td>82.47</td><td>90.39</td><td>67.81</td><td>71.15</td><td>75.14</td><td>80.77</td></tr><tr><td>HR-DPT + Dir</td><td>83.13</td><td>90.79</td><td>70.54</td><td>73.79</td><td>76.84</td><td>82.29</td></tr><tr><td>HR-DPT + Dir + Sad</td><td>83.29</td><td>90.88</td><td>70.86</td><td>74.09</td><td>77.08</td><td>82.49</td></tr><tr><td rowspan="4">GF-7</td><td>DPT</td><td>76.27</td><td>86.54</td><td>71.10</td><td>74.01</td><td>73.69</td><td>80.28</td></tr><tr><td>HR-DPT</td><td>77.94</td><td>87.60</td><td>74.57</td><td>77.89</td><td>76.25</td><td>82.75</td></tr><tr><td> $\mathrm { H R - D P T + D i r }$ </td><td>78.48</td><td>87.94</td><td>75.70</td><td>78.99</td><td>77.09</td><td>83.47</td></tr><tr><td> $\mathrm { H R \mathrm { - } D P T } + \mathrm { D i r } + \mathrm { S a d }$ </td><td>78.68</td><td>88.07</td><td>75.76</td><td>79.04</td><td>77.22</td><td>83.56</td></tr><tr><td rowspan="4">Planet</td><td>DPT</td><td>45.35</td><td>62.40</td><td>42.02</td><td>43.61</td><td>43.69</td><td>53.01</td></tr><tr><td>HR-DPT</td><td>45.71</td><td>62.74</td><td>42.64</td><td>44.25</td><td>44.18</td><td>53.50</td></tr><tr><td>HR-DPT + Dir</td><td>45.44</td><td>62.48</td><td>46.10</td><td>48.02</td><td>45.77</td><td>55.25</td></tr><tr><td> $\mathrm { H R \mathrm { - } D P T } + \mathrm { D i r } + \mathrm { S a d }$ </td><td>48.25</td><td>65.10</td><td>47.35</td><td>49.30</td><td>47.80</td><td>57.20</td></tr><tr><td rowspan="4">ST-2</td><td>DPT</td><td>27.05</td><td>42.58</td><td>22.53</td><td>23.07</td><td>24.79</td><td>32.83</td></tr><tr><td>HR-DPT</td><td>28.21</td><td>44.00</td><td>23.18</td><td>23.83</td><td>25.70</td><td>33.92</td></tr><tr><td> $\mathrm { H R - D P T + D i r }$ </td><td>26.17</td><td>41.48</td><td>27.76</td><td>29.61</td><td>26.97</td><td>35.55</td></tr><tr><td> $\mathrm { H R \mathrm { - } D P T } + \mathrm { D i r } + \mathrm { S a d }$ </td><td>36.73</td><td>53.72</td><td>30.47</td><td>31.69</td><td>33.60</td><td>42.71</td></tr></table>

2) Decoder Effectiveness and Complexity: We compare the decoder-only parameters (PM) and GFLOPs (GP) of UPerNet [59], DPT [32], and HR-DPT under the same DINOv3-B backbone with a $5 1 2 \times 5 1 2$ input. As shown in Table VI, HR-DPT requires 12.0M parameters and 70.1 GFLOPs, slightly higher than DPT but much lower than UPerNet. With this limited additional cost, HR-DPT consistently outperforms DPT and achieves better boundary reconstruction, producing finer building structures and cleaner boundaries under the same CE supervision, as shown in Fig. 6.

![](images/0cb2e26dbb6a5971a105d8e5cc5f4c090e78a691c97e5a59986fbefcd1aded6b.jpg)  
Fig. 5. Cross-dataset IoU comparison between single-dataset DINOv3-B-DPT models and multi-dataset UniBuild. Rows indicate training datasets and columns indicate evaluation datasets. Gain reports the IoU improvement of UniBuild over the diagonal DINOv3-B-DPT result.

TABLE VI  
DECODER-ONLY COMPARISON UNDER A 512 × 512 INPUT. PM AND GP DENOTE PARAMETER NUMBER (M) AND FLOPS (G).
<table><tr><td>Dataset</td><td>Decoder</td><td>IoU</td><td>F1</td><td>B-IoU</td><td>B-F1</td><td> $\scriptstyle \mathbf { M } _ { \mathrm { I o U } }$  </td><td> $\mathbf { M } _ { \mathrm { F 1 } }$ </td><td>PM (M)</td><td>GP (G)</td></tr><tr><td rowspan="3">INRIA</td><td>UPerNet</td><td>82.25</td><td>90.26</td><td>66.83</td><td>69.97</td><td>74.54</td><td>80.12</td><td>38.1</td><td>213.1</td></tr><tr><td>DPT</td><td>82.17</td><td>90.21</td><td>66.91</td><td>70.04</td><td>74.54</td><td>80.13</td><td>11.0</td><td>30.1</td></tr><tr><td>HR-DPT</td><td>82.47</td><td>90.39</td><td>67.81</td><td>71.15</td><td>75.14</td><td>80.77</td><td>12.0</td><td>70.1</td></tr><tr><td rowspan="3">GF-7</td><td>UPerNet</td><td>76.82</td><td>86.89</td><td>72.48</td><td>75.46</td><td>74.65</td><td>81.18</td><td>38.1</td><td>213.1</td></tr><tr><td>DPT</td><td>76.27</td><td>86.54</td><td>71.10</td><td>74.01</td><td>73.69</td><td>80.28</td><td>11.0</td><td>30.1</td></tr><tr><td>HR-DPT</td><td>77.94</td><td>87.60</td><td>74.57</td><td>77.89</td><td>76.25</td><td>82.75</td><td>12.0</td><td>70.1</td></tr><tr><td rowspan="3">Planet</td><td>UPerNet</td><td>44.70</td><td>61.78</td><td>41.26</td><td>42.83</td><td>42.98</td><td>52.31</td><td>38.1</td><td>213.1</td></tr><tr><td>DPT</td><td>45.35</td><td>62.40</td><td>42.02</td><td>43.61</td><td>43.69</td><td>53.01</td><td>11.0</td><td>30.1</td></tr><tr><td>HR-DPT</td><td>45.71</td><td>62.74</td><td>42.64</td><td>44.25</td><td>44.18</td><td>53.50</td><td>12.0</td><td>70.1</td></tr><tr><td rowspan="3">ST-2</td><td>UPerNet</td><td>28.70</td><td>44.60</td><td>21.37</td><td>21.91</td><td>25.04</td><td>33.26</td><td>38.1</td><td>213.1</td></tr><tr><td>DPT</td><td>27.05</td><td>42.58</td><td>22.53</td><td>23.07</td><td>24.79</td><td>32.83</td><td>11.0</td><td>30.1</td></tr><tr><td>HR-DPT</td><td>28.21</td><td>44.00</td><td>23.18</td><td>23.83</td><td>25.70</td><td>33.92</td><td>12.0</td><td>70.1</td></tr></table>

3) Direction-Aware Consistency: We further study direction-aware regularization on top of HR-DPT. This loss uses structure-tensor-based orientation cues to encourage local boundary consistency and suppress fragmented or zigzag contour responses. As shown in Table VII, it significantly improves boundary-aware metrics. For example, on INRIA, B-IoU and B-F1 increase from 67.81 and 71.15 to 70.87 and 74.09, respectively. This indicates that the orientation constraint effectively improves the geometric consistency of building contours. On MR/LR datasets, direction-aware regularization also improves boundary metrics, but its effect is more sensitive due to indistinct boundaries and mixed pixels. Therefore, this loss is mainly used as a boundary refinement term. Considering the balance between boundary quality and region completeness, we set $\lambda _ { \mathrm { d i r } } ~ = ~ 0 . 5$ as the default setting. The results in Fig. 7 further show that direction-aware regularization produces straighter and more coherent building boundaries with fewer irregular contour responses.

![](images/58c0b9cba2e2d05811fe2f5187ff6d32f4af7726cb5b7972c5a79d28ef501e42.jpg)  
Fig. 6. Visual comparison of DPT and HR-DPT trained with CE loss on INRIA and Planet. Planet examples are bilinearly upsampled to 1 m.

TABLE VII  
EFFECT OF DIRECTION-AWARE REGULARIZATION WEIGHT $\lambda _ { \mathrm { d i r } }$ . “W/O DIR” DENOTES WITHOUT DIRECTION-AWARE REGULARIZATION.
<table><tr><td>Dataset</td><td> $\lambda _ { \mathrm { d i r } }$ </td><td>IoU</td><td>F1</td><td>B-IoU</td><td>B-F1</td><td> $\mathbf { M } _ { \mathrm { I o U } }$ </td><td>MF1</td></tr><tr><td rowspan="6">INRIA</td><td>w/o Dir</td><td>82.47</td><td>90.39</td><td>67.81</td><td>71.15</td><td>75.14</td><td>80.77</td></tr><tr><td>0.2</td><td>82.93</td><td>90.67</td><td>70.15</td><td>73.42</td><td>76.54</td><td>82.05</td></tr><tr><td>0.5</td><td>83.13</td><td>90.79</td><td>70.54</td><td>73.79</td><td>76.84</td><td>82.29</td></tr><tr><td>1.0</td><td>83.17</td><td>90.81</td><td>70.87</td><td>74.09</td><td>77.02</td><td>82.45</td></tr><tr><td>1.5</td><td>83.05</td><td>90.74</td><td>70.74</td><td>73.96</td><td>76.90</td><td>82.35</td></tr><tr><td>2.0</td><td>83.00</td><td>90.71</td><td>70.55</td><td>73.78</td><td>76.77</td><td>82.25</td></tr><tr><td rowspan="6">GF-7</td><td>w/o Dir</td><td>77.94</td><td>87.60</td><td>74.57</td><td>77.89</td><td>76.25</td><td>82.75</td></tr><tr><td>0.2</td><td>78.52</td><td>87.97</td><td>75.73</td><td>79.03</td><td>77.12</td><td>83.50</td></tr><tr><td>0.5</td><td>78.48</td><td>87.94</td><td>75.70</td><td>78.99</td><td>77.09</td><td>83.47</td></tr><tr><td>1.0</td><td>78.50</td><td>87.96</td><td>75.59</td><td>78.89</td><td>77.05</td><td>83.43</td></tr><tr><td>1.5</td><td>78.41</td><td>87.90</td><td>75.38</td><td>78.71</td><td>76.89</td><td>83.31</td></tr><tr><td>2.0</td><td>78.27</td><td>87.81</td><td>75.15</td><td>78.50</td><td>76.71</td><td>83.16</td></tr><tr><td rowspan="6">Planet</td><td>w/o Dir</td><td>45.71</td><td>62.74</td><td>42.64</td><td>44.25</td><td>44.18</td><td>53.50</td></tr><tr><td>0.2</td><td>45.70</td><td>62.73</td><td>46.06</td><td>47.94</td><td>45.88</td><td>55.34</td></tr><tr><td>0.5</td><td>45.44</td><td>62.48</td><td>46.10</td><td>48.02</td><td>45.77</td><td>55.25</td></tr><tr><td>1.0</td><td>44.85</td><td>61.93</td><td>46.71</td><td>48.60</td><td>45.78</td><td>55.27</td></tr><tr><td>1.5</td><td>44.84</td><td>61.92</td><td>46.31</td><td>48.20</td><td>45.58</td><td>55.06</td></tr><tr><td>2.0</td><td>44.93</td><td>62.01</td><td>46.81</td><td>48.70</td><td>45.87</td><td>55.36</td></tr><tr><td rowspan="6">ST-2</td><td>w/o Dir</td><td>28.21</td><td>44.00</td><td>23.18</td><td>23.83</td><td>25.70</td><td>33.92</td></tr><tr><td>0.2</td><td>26.31</td><td>41.66</td><td>25.52</td><td>26.98</td><td>25.92</td><td>34.32</td></tr><tr><td>0.5</td><td>26.17</td><td>41.48</td><td>27.76</td><td>29.61</td><td>26.97</td><td>35.55</td></tr><tr><td>1.0</td><td>26.20</td><td>41.52</td><td>26.03</td><td>27.30</td><td>26.11</td><td>34.41</td></tr><tr><td>1.5</td><td>25.30</td><td>40.39</td><td>26.45</td><td>27.69</td><td>25.88</td><td>34.04</td></tr><tr><td>2.0</td><td>23.97</td><td>38.68</td><td>28.00</td><td>29.96</td><td>25.99</td><td>34.32</td></tr></table>

4) Saddle-Aware Suppression: We then evaluate saddleaware suppression, where $\alpha _ { \mathrm { L R } }$ denotes the internal falsepositive penalty strength. Since the saddle penalty is activated only for LR batches, its main effect appears on Planet and ST-2. As shown in Table VIII, $\alpha _ { \mathrm { L R } } = 1 5$ achieves the best overall performance, with $\mathbf { M } _ { \mathrm { I o U } }$ improving from 45.77 to 47.84 on Planet and from 26.97 to 33.60 on ST-2. The qualitative results in Fig. 8 further show that saddle-aware suppression effectively preserves narrow gaps in LR imagery.

## E. Out-of-Domain Building Extraction Performance

To assess OOD performance, we directly apply UniBuild to unseen domains without fine-tuning. The evaluated datasets are summarized in Table IX. As shown in Table X, UniBuild achieves strong zero-shot performance on Waterloo, WHU-Satellite, and Massachusetts, and remains effective on the more challenging ISPRS-Pforzheim dataset, demonstrating good transferability to unseen aerial and satellite imagery.

![](images/223073fdcf7ec9521161c89ae8104a5544002ec0815a25cc9c6dd79a3509fa67.jpg)  
Fig. 7. Visual comparison of HR-DPT trained with CE loss and CE+Dir on INRIA and Planet. Planet examples are bilinearly upsampled to 1 m.  
TABLE VIII

EFFECT OF SADDLE-AWARE PENALTY STRENGTH. ${ } ^ { 6 6 } \mathrm { W } / \mathrm { O }$ $\mathbf { S A D } ^ { * }$ DENOTES DIRECTION-AWARE REGULARIZATION ONLY. THE SADDLE PENALTY IS DISABLED FOR HR DATASETS WITH $\alpha _ { \mathrm { { H R } } } = 0 ,$ WHILE $\alpha _ { \mathrm { L R } }$ CONTROLS THE FALSE-POSITIVE PENALTY IN LR SADDLE REGIONS.
<table><tr><td>Dataset</td><td>αHR</td><td> $\alpha _ { \mathrm { L R } }$ </td><td>IoU</td><td>F1</td><td>B-IoU</td><td>B-F1</td><td> $\mathbf { M } _ { \mathrm { I o U } }$ </td><td> $\mathbf { M } _ { \mathrm { F 1 } }$ </td></tr><tr><td rowspan="2">INRIA</td><td>w/o Sad</td><td></td><td>83.13</td><td>90.79</td><td>70.54</td><td>73.79</td><td>76.84</td><td>82.29</td></tr><tr><td>0</td><td>一</td><td>83.29</td><td>90.88</td><td>70.86</td><td>74.09</td><td>77.08</td><td>82.49</td></tr><tr><td rowspan="2">GF-7</td><td>w/o Sad</td><td></td><td>78.48</td><td>87.94</td><td>75.70</td><td>78.99</td><td>77.09</td><td>83.47</td></tr><tr><td>0</td><td></td><td>78.68</td><td>88.07</td><td>75.76</td><td>79.04</td><td>77.22</td><td>83.56</td></tr><tr><td rowspan="7">Planet</td><td>一</td><td>w/o Sad</td><td>45.44</td><td>62.48</td><td>46.10</td><td>48.02</td><td>45.77</td><td>55.25</td></tr><tr><td></td><td>0</td><td>49.47</td><td>66.19</td><td>43.68</td><td>45.73</td><td>46.57</td><td>55.96</td></tr><tr><td></td><td>5</td><td>49.19</td><td>65.94</td><td>45.36</td><td>47.40</td><td>47.28</td><td>56.67</td></tr><tr><td></td><td>10</td><td>48.94</td><td>65.72</td><td>46.74</td><td>48.75</td><td>47.84</td><td>57.24</td></tr><tr><td></td><td>15</td><td>48.25</td><td>65.10</td><td>47.35</td><td>49.30</td><td>47.80</td><td>57.20</td></tr><tr><td></td><td>20</td><td>47.30</td><td>64.22</td><td>48.23</td><td>50.13</td><td>47.77</td><td>57.18</td></tr><tr><td></td><td>25</td><td>46.91</td><td>63.86</td><td>47.52</td><td>49.40</td><td>47.21</td><td>56.63</td></tr><tr><td rowspan="7">ST-2</td><td></td><td>w/o Sad</td><td>26.17</td><td>41.48</td><td>27.76</td><td>29.61</td><td>26.97</td><td>35.55</td></tr><tr><td></td><td>0</td><td>37.97</td><td>55.04</td><td>28.25</td><td>29.44</td><td>33.11</td><td>42.24</td></tr><tr><td></td><td>5</td><td>37.48</td><td>54.52</td><td>29.25</td><td>30.46</td><td>33.37</td><td>42.49</td></tr><tr><td></td><td>10</td><td>36.88</td><td>53.89</td><td>29.41</td><td>30.59</td><td>33.15</td><td>42.24</td></tr><tr><td></td><td>15</td><td>36.73</td><td>53.72</td><td>30.47</td><td>31.69</td><td>33.60</td><td>42.71</td></tr><tr><td></td><td>20</td><td>36.22</td><td>53.17</td><td>29.88</td><td>31.11</td><td>33.05</td><td>42.14</td></tr><tr><td></td><td>25</td><td>35.57</td><td>52.47</td><td>30.40</td><td>31.64</td><td>32.99</td><td>42.06</td></tr></table>

Visual examples are provided in Fig. 9. The results show that UniBuild produces spatially coherent masks with meaningful instance-level separability. We further apply a simple polygonization procedure, including connected-component separation, contour simplification, dominant-direction constraints, and short-edge merging, to visualize the extracted building instance structures. Overall, UniBuild directly produces reliable building masks and polygons, demonstrating practical robustness across diverse unseen domains. However,

![](images/d5cdb04f0c8b4df9a4421d23970a4405f9b227c667a091274514728a04adae2c.jpg)  
Fig. 8. Visual comparison of saddle-weighted regularization, where green areas denote saddle regions. We compare CE+Dir, CE+Dir+Dice $( \alpha _ { \mathrm { L R } } = 0 )$ and CE+Dir+Saddle $( \alpha _ { \mathrm { L R } } = 1 5 )$ , with all examples upsampled to 1 m.

LR inputs may still limit accurate instance-level delineation when small or adjacent buildings are poorly resolved.

TABLE IX  
OOD DATASETS FOR ZERO-SHOT EVALUATION. FOR MASSACHUSETTS AND ISPRS-PFORZHEIM, 0.5 m AND 1.0 m DENOTE THE UPSAMPLED INFERENCE RESOLUTIONS FROM 1.0 m AND 5.8 m, RESPECTIVELY.
<table><tr><td>Dataset</td><td> $\operatorname { R e g i o n }$ </td><td> $\mathrm { T y p e }$ </td><td>Subset</td><td>Resolution</td><td>Image size / Samples</td></tr><tr><td>Waterloo [60]</td><td>Canada</td><td>Aerial</td><td>Validation set</td><td>0.12 m</td><td>512 × 512 / 6,887</td></tr><tr><td>WHU-Satellite [12]</td><td>Global</td><td>Satellite</td><td>All</td><td>0.3-2.5 m</td><td>512 × 512 / 204</td></tr><tr><td>Massachusetts [61]</td><td>USA</td><td>Aerial</td><td>Test set</td><td>1.0 m (0.5 m ↑)</td><td>1500 × 1500 / 10</td></tr><tr><td>ISPRS-Pforzheim [62]</td><td>Germany</td><td>Satellite</td><td>One ZY-3 scene</td><td>5.8 m (1.0 m ↑)</td><td>42815 × 14188 /  1</td></tr></table>

TABLE X

OOD EVALUATION RESULTS OF UNIBUILD. $\mathrm { ^ { 6 6 } B \mathrm { - } I o U ^ { 3 3 } }$ AND $\mathrm { ^ { * } B { - } F 1 } ^ { \prime \mathrm { 3 } }$ DENOTE BOUNDARY-IOU AND BOUNDARY-F1, RESPECTIVELY. $\mathrm { ^ { * } M _ { I o U } } ^ { \prime \ }$ IS THE MEAN OF IOU AND B-IOU, WHILE $\mathbf { \ddot { \Phi } M } _ { \mathrm { F 1 } } \mathbf { \Phi } ^ { \ast }$ IS THE MEAN OF F1 AND B-F1. MASSACHUSETTS-0.5 m DENOTES THE RESULT AFTER UPSAMPLING THE ORIGINAL 1.0 m IMAGERY BY A FACTOR OF TWO.
<table><tr><td>Dataset</td><td>IoU</td><td>F1</td><td>B-IoU</td><td>B-F1</td><td>_  $\mathbf { M } _ { \mathrm { I o U } }$ </td><td> $\mathbf { M } _ { \mathrm { F 1 } }$ </td></tr><tr><td>Waterloo</td><td>89.28</td><td>94.34</td><td>73.15</td><td>76.86</td><td>81.22</td><td>85.60</td></tr><tr><td>WHU-Satellite</td><td>73.80</td><td>84.92</td><td>70.45</td><td>74.12</td><td>72.12</td><td>79.52</td></tr><tr><td>Massachusetts-1.0 m</td><td>53.57</td><td>69.77</td><td>71.85</td><td>74.72</td><td>62.71</td><td>72.24</td></tr><tr><td>Massachusetts-0.5 m</td><td>69.40</td><td>81.94</td><td>64.67</td><td>67.56</td><td>67.04</td><td>74.75</td></tr><tr><td>ISPRS-Pforzheim</td><td>32.51</td><td>49.07</td><td>30.77</td><td>32.63</td><td>31.64</td><td>40.85</td></tr><tr><td>Average</td><td>63.71</td><td>76.01</td><td>62.18</td><td>65.17</td><td>62.95</td><td>70.59</td></tr></table>

The comparison between Massachusetts-1.0 m and -0.5 m highlights the scale-adaptive inference capability of UniBuild. By directly upsampling the input imagery from 1.0 m to 0.5 m before inference, UniBuild improves IoU from 53.57 to 69.40. This shows that UniBuild supports flexible inference-time resolution adjustment, allowing a practical trade-off between extraction accuracy and computational cost while adapting to different building scales without retraining.

## V. CONCLUSION

In this paper, we presented UniBuild, a unified building extraction framework for multi-source optical RS imagery.

![](images/c1dadd74825ce090171c851575add9c05b964c9be925aa21e9e341e4134b2bfe.jpg)  
Fig. 9. Qualitative OOD building extraction and polygonization examples of UniBuild on four unseen datasets. Each row corresponds to one OOD dataset, including Waterloo, WHU-Satellite, Massachusetts, and ISPRS-Pforzheim (bilinearly upsampled from 5.8 m to 1 m).

UniBuild integrates multi-dataset training, an HR-DPT decoder, and geometry-aware regularization to improve transferable building representation learning, boundary recovery, and separation of adjacent buildings. Extensive experiments verify the effectiveness of UniBuild against competing models and show that its components are complementary, producing more accurate building regions, sharper boundaries, and clearer building separation. Multi-dataset training and out-of-domain evaluations further demonstrate its potential as a practical unified model for multi-source optical building extraction, including challenging imagery up to 10 m GSD.

Despite these results, several limitations remain. First, building extraction from LR imagery, especially 10 m Sentinel-2 imagery, remains challenging because small buildings and fine footprint details are close to or below the sensor resolution. Second, multi-source annotations may suffer from missing buildings, outdated footprints, and image–label misalignment. third, the current footprint regularization is used as postprocessing rather than an end-to-end raster-to-vector component. Future work will focus on LR building modeling, noiserobust supervision, and end-to-end footprint vectorization.

## REFERENCES

[1] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby et al., “Dinov2: Learning robust visual features without supervision,” Transactions on Machine Learning Research Journal, 2024.

[2] O. Simeoni, H. V. Vo, M. Seitzer, F. Baldassarre,´ M. Oquab, C. Jose, V. Khalidov, M. Szafraniec, S. Yi, M. Ramamonjisoa et al., “Dinov3,” arXiv preprint arXiv:2508.10104, 2025.

[3] L. Luo, P. Li, and X. Yan, “Deep learning-based building extraction from remote sensing images: A comprehensive review,” Energies, vol. 14, no. 23, p. 7982, 2021.

[4] Q. Li, L. Mou, Y. Sun, Y. Hua, Y. Shi, and X. X. Zhu, “A review of building extraction from remote sensing imagery: Geometrical structures and semantic attributes,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–15, 2024.

[5] X. X. Zhu, S. Chen, F. Zhang, Y. Shi, and Y. Wang, “Globalbuildingatlas: an open global and complete dataset of building polygons, heights and LoD1 3d models,” Earth System Science Data, vol. 17, no. 12, pp. 6647–6668, 2025.

[6] M. Awrangjeb, M. Ravanbakhsh, and C. S. Fraser, “Automatic detection of residential buildings using LIDAR data and multispectral imagery,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 65, no. 5, pp. 457–467, 2010.

[7] E. Li, J. Femiani, S. Xu, X. Zhang, and P. Wonka, “Robust rooftop extraction from visible band images using higher order CRF,” IEEE Transactions on Geoscience and Remote Sensing, vol. 53, no. 8, pp. 4483–4495, 2015.

[8] J. Long, E. Shelhamer, and T. Darrell, “Fully convolutional networks for semantic segmentation,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2015, pp. 3431–3440.

[9] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in Medical Image Computing and Computer-Assisted Intervention, ser. Lecture Notes in Computer Science, vol. 9351. Springer, 2015, pp. 234–241.

[10] V. Badrinarayanan, A. Kendall, and R. Cipolla, “Segnet: A deep convolutional encoder-decoder architecture for image segmentation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 39, no. 12, pp. 2481–2495, 2017.

[11] K. Bittner, S. Cui, and P. Reinartz, “Building extraction from remote sensing data using fully convolutional networks,” The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, vol. 42, pp. 481–486, 2017.

[12] S. Ji, S. Wei, and M. Lu, “Fully convolutional networks for multisource building extraction from an open aerial and satellite imagery data set,” IEEE Transactions on Geoscience and Remote Sensing, vol. 57, no. 1, pp. 574– 586, 2019.

[13] R. Alshehhi, P. R. Marpu, W. L. Woon, and M. Dalla Mura, “Simultaneous extraction of roads and buildings in remote sensing imagery with convolutional neural networks,” ISPRS Journal ofPhotogrammetry and Remote Sensing, vol. 130, pp. 139–149, 2017.

[14] J. Huang, X. Zhang, Q. Xin, Y. Sun, and P. Zhang, “Automatic building extraction from high-resolution aerial images and lidar data using gated residual refinement network,” ISPRS journal of photogrammetry and remote sensing, vol. 151, pp. 91–105, 2019.

[15] Q. Song, F. Mo, K. Ding, L. Xiao, R. Dian, X. Kang, and S. Li, “Mcfnet: Multiscale cross-domain fusion network for hsi and lidar data joint classification,” IEEE Transactions on Geoscience and Remote Sensing, 2025.

[16] Q. Song, J. Peng, W. Song, B. Sun, R. Dian, and S. Li, “Multi-domain adaptive fusion network for multisource remote sensing data classification,” Science China Information Sciences, 2026.

[17] Y. Shi, Q. Li, and X. X. Zhu, “Building segmentation through a gated graph convolutional neural network with deep structured feature embedding,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 159, pp. 184– 197, 2020.

[18] Y. Xu, L. Wu, Z. Xie, and Z. Chen, “Building extraction in very high resolution remote sensing imagery using deep learning and guided filters,” Remote Sensing, vol. 10, no. 1, p. 144, 2018.

[19] Y. Li, D. Hong, C. Li, J. Yao, and J. Chanussot, “Hd-net: High-resolution decoupled network for building footprint extraction via deeply supervised body and boundary decomposition,” ISPRS journal of photogrammetry and remote sensing, vol. 209, pp. 51–65, 2024.

[20] W. Huang, Y. Shi, Z. Xiong, and X. X. Zhu, “Heightassisted semi-supervised building footprint extraction from optical remote sensing images,” IEEE Transactions on Geoscience and Remote Sensing, vol. 63, pp. 1–14, 2025.

[21] K. Chen, Z. Zou, and Z. Shi, “Building extraction from remote sensing images with sparse token transformers,” Remote Sensing, vol. 13, no. 21, p. 4441, 2021.

[22] Q. Zhu, Z. Li, T. Song, L. Yao, Q. Guan, and L. Zhang, “Unrestricted region and scale: Deep self-supervised building mapping framework across different cities from five continents,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 209, pp. 344–367, 2024.

[23] W. Huang, Y. Shi, Z. Xiong, and X. X. Zhu, “Adaptmatch: Adaptive matching for semisupervised binary segmentation of remote sensing images,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1–16, 2023.

[24] W. Huang, Z. Gu, Y. Shi, Z. Xiong, and X. X. Zhu, “Semi-supervised building footprint extraction using debiased pseudo-labels,” IEEE Transactions on Geoscience and Remote Sensing, vol. 63, 2024.

[25] C. Liu, C. M. Albrecht, Y. Wang, Q. Li, and X. X. Zhu, “Aio2: Online correction of object labels for deep learning with incomplete annotation in remote sensing image segmentation,” IEEE Transactions on Geoscience

and Remote Sensing, vol. 62, pp. 1–17, 2024.

[26] J. Chen, P. He, J. Zhu, Y. Guo, G. Sun, M. Deng, and H. Li, “Memory-contrastive unsupervised domain adaptation for building extraction of high-resolution remote sensing imagery,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1–15, 2023.

[27] Z. Du, H. Sui, Q. Zhou, M. Zhou, W. Shi, J. Wang, and J. Liu, “Vectorized building extraction from highresolution remote sensing images using spatial cognitive graph convolution model,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 213, pp. 53–71, 2024.

[28] T.-Y. Lin, P. Dollar, R. Girshick, K. He, B. Hariharan,´ and S. Belongie, “Feature pyramid networks for object detection,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017, pp. 2117–2125.

[29] H. Zhao, J. Shi, X. Qi, X. Wang, and J. Jia, “Pyramid scene parsing network,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017, pp. 2881–2890.

[30] Z. Yang, H. Yao, Q. Li, W. Ni, J. Wu, and Q. Wang, “Semantic–spatial feature refinement network for road extraction from remote sensing images,” IEEE Transactions on Geoscience and Remote Sensing, vol. 64, pp. 1–10, 2026.

[31] E. Xie, W. Wang, Z. Yu, A. Anandkumar, J. M. Alvarez, and P. Luo, “Segformer: Simple and efficient design for semantic segmentation with transformers,” Advances in Neural Information Processing Systems, vol. 34, pp. 12 077–12 090, 2021.

[32] R. Ranftl, A. Bochkovskiy, and V. Koltun, “Vision transformers for dense prediction,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 12 179–12 188.

[33] H. Kervadec, J. Bouchtiba, C. Desrosiers, E. Granger, J. Dolz, and I. Ben Ayed, “Boundary loss for highly unbalanced segmentation,” Medical Image Analysis, vol. 67, p. 101851, 2021.

[34] C. Wang, Y. Zhang, M. Cui, P. Ren, Y. Yang, X. Xie, X.-S. Hua, H. Bao, and W. Xu, “Active boundary loss for semantic segmentation,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 36, no. 2, pp. 2397–2405, 2022.

[35] S. He and W. Jiang, “Boundary-assisted learning for building extraction from optical remote sensing imagery,” Remote Sensing, vol. 13, no. 4, p. 760, 2021.

[36] Q. Zhu, Z. Li, Y. Zhang, and Q. Guan, “Building extraction from high spatial resolution remote sensing images via multiscale-aware and segmentation-prior conditional random fields,” Remote Sensing, vol. 12, no. 23, p. 3983, 2020.

[37] R. Deng, Z. Guo, Q. Chen, X. Sun, Q. Chen, H. Wang, and X. Liu, “A dual spatial-graph refinement network for building extraction from aerial images,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1–20, 2023.

[38] H. Lin, M. Hao, W. Luo, H. Yu, and N. Zheng, “Bearnet: A novel buildings edge-aware refined network for

building extraction from high-resolution remote sensing images,” IEEE Geoscience and Remote Sensing Letters, vol. 20, pp. 1–5, 2023.

[39] S. Shit, J. C. Paetzold, A. Sekuboyina, I. Ezhov, A. Unger, A. Zhylka, J. P. W. Pluim, U. Bauer, and B. H. Menze, “cldice: A novel topology-preserving loss function for tubular structure segmentation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 16 560–16 569.

[40] L. Ding, H. Tang, Y. Liu, Y. Shi, X. X. Zhu, and L. Bruzzone, “Adversarial shape learning for building extraction in vhr remote sensing images,” IEEE Transactions on Image Processing, vol. 31, pp. 678–690, 2021.

[41] F. Rottensteiner, G. Sohn, J. Jung, M. Gerke, C. Baillard, S. Benitez, and U. Breitkopf, “The ISPRS benchmark on urban object classification and 3d building reconstruction,” in ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences, vol. I-3, 2012, pp. 293–298.

[42] E. Maggiori, Y. Tarabalka, G. Charpiat, and P. Alliez, “Can semantic labeling methods generalize to any city? the INRIA aerial image labeling benchmark,” in 2017 IEEE International Geoscience and Remote Sensing Symposium (IGARSS). IEEE, 2017, pp. 3226–3229.

[43] D. Cao, “Alabama buildings segmentation,” 2022. [Online]. Available: https://www.kaggle.com/datasets/ meowmeowplus/alabama-buildings-segmentation

[44] J. Xia, N. Yokoya, B. Adriano, and C. Broni-Bediako, “OpenEarthMap: A benchmark dataset for global highresolution land cover mapping,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2023, pp. 6254–6264.

[45] J. Wang, Z. Zheng, A. Ma, X. Lu, and Y. Zhong, “LoveDA: A remote sensing land-cover dataset for domain adaptive semantic segmentation,” in Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, vol. 1, 2021.

[46] P. Chen, H. Huang, F. Ye, J. Liu, W. Li, J. Wang, Z. Wang, C. Liu, and N. Zhang, “A benchmark GaoFen-7 dataset for building extraction from satellite images,” Scientific Data, vol. 11, no. 1, p. 187, 2024.

[47] M. Luo, S. Ji, and S. Wei, “A diverse large-scale building dataset and a novel plug-and-play domain generalization method for building extraction,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 16, pp. 4122–4138, 2023.

[48] A. Van Etten, D. Lindenbaum, and T. M. Bacastow, “SpaceNet: A remote sensing dataset and challenge series,” arXiv preprint arXiv:1807.01232, 2018.

[49] A. Boguszewski, D. Batorski, N. Ziemba-Jankowska, T. Dziedzic, and A. Zambrzycka, “Landcover. ai: Dataset for automatic mapping of buildings, woodlands, water and roads from aerial imagery,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 1102–1110.

[50] B. Swan, J. Pyle, D. Roddy, H. L. Yang, A. Rose, and M. Laverdiere, “ORBITaL-Net: A labeled training library for large-scale building feature extraction,” Scientific

Data, vol. 12, p. 1650, 2025.

[51] P. Team, “Planet application program interface: In space for life on earth,” Planet, 2017.

[52] M. Drusch, U. Del Bello, S. Carlier, O. Colin, V. Fernandez, F. Gascon, B. Hoersch, C. Isola, P. Laberinti, P. Martimort, A. Meygret, F. Spoto, O. Sy, F. Marchese, and P. Bargellini, “Sentinel-2: ESA’s optical high-resolution mission for GMES operational services,” Remote Sensing of Environment, vol. 120, pp. 25–36, 2012.

[53] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 770–778.

[54] L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, and H. Adam, “Encoder-decoder with atrous separable convolution for semantic image segmentation,” in Proceedings of the European Conference on Computer Vision (ECCV), 2018, pp. 801–818.

[55] L. Wang, S. Fang, R. Li, and X. Meng, “Building extraction with vision transformer,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–11, 2022.

[56] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, “Swin transformer: Hierarchical vision transformer using shifted windows,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 10 012–10 022.

[57] Q. Tang, Y. Li, Y. Xu, and B. Du, “Enhancing building footprint extraction with partial occlusion by exploring building integrity,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, p. 5650814, 2024.

[58] Y. Zhou, Z. Chen, B. Wang, S. Li, H. Liu, D. Xu, and C. Ma, “Bomsc-net: Boundary optimization and multiscale context awareness based building extraction from high-resolution remote sensing imagery,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–17, 2022.

[59] T. Xiao, Y. Liu, B. Zhou, Y. Jiang, and J. Sun, “Unified perceptual parsing for scene understanding,” in Proceedings of the European conference on computer vision (ECCV), 2018, pp. 418–434.

[60] H. He, Z. Jiang, K. Gao, S. Narges Fatholahi, W. Tan, B. Hu, H. Xu, M. A. Chapman, and J. Li, “Waterloo building dataset: A city-scale vector building dataset for mapping building footprints using aerial orthoimagery,” Geomatica, vol. 75, no. 3, pp. 99–115, 2022.

[61] V. Mnih, “Machine learning for aerial image labeling,” Ph.D. dissertation, University of Toronto, 2013. [Online]. Available: https://www.cs.toronto.edu/<sup>∼</sup>vmnih/ docs/Mnih Volodymyr PhD Thesis.pdf

[62] ISPRS, “ISPRS Data Sets: ZY-3 Stuttgart,” https://www.isprs.org/resources/datasets/images/ zy-3-Stuttgart/Default.aspx.

[63] H. Chen and Z. Shi, “A spatial-temporal attention-based method and a new dataset for remote sensing image change detection,” Remote sensing, vol. 12, no. 10, p. 1662, 2020.

## VI. SUPPLEMENTAL MATERIALS

## A. Cross-Resolution Building Polygonization

Fig. 10 further demonstrates the practical image-to-map capability of UniBuild across different image sources and spatial resolutions. The examples include high-resolution Google Maps image crops from Fez, Morocco, with a spatial resolution of 0.7 m, and Planet imagery from Carluke, Scotland, with an original spatial resolution of 4.8 m, which is upsampled to 1.0 m before inference. Google Maps imagery represents an unseen data source, whereas Planet imagery is included in the training data but is evaluated here in a different geographic region. Given only RGB imagery, UniBuild can directly produce building masks, which are further converted into structured building polygons and exported as shapefiles using our simple and efficient polygonization algorithm provided in the released code. This enables an end-to-end workflow from newly acquired optical imagery to GIS-ready building vector data, making UniBuild useful for timely building map updating. The results also show that higher-resolution imagery leads to more detailed and accurate building polygons, while lowerresolution inputs are more suitable for large-scale building mapping than for fine-grained instance-level delineation. These results highlight the practical potential of UniBuild for GIS applications such as building map updating, urban monitoring, settlement mapping, and geospatial database enrichment.

![](images/45ba7b51e9d8264bfcb2f56544d856720932578a67711d5b7e17158d17643656.jpg)

![](images/20ee338ced36145339950812ad885d286305bc82b8103d6efa57cee4dda50dd6.jpg)

![](images/40c1ee12352e057261dd2ed234798a5d9aca74c94c6f02c711eae43b9126ae18.jpg)

![](images/132fbb56c109c50ddf8ab5ebf65c1ba5a1086581efba424f395b3b28d1cfff75.jpg)  
Fig. 10. Building polygonization examples across image sources and spatial resolutions. The examples include Google Maps imagery from Fez, Morocco, at approximately 0.7 m resolution, and Planet imagery from Carluke, Scotland, upsampled from approximately 4.8 m to 1.0 m before inference. From top to bottom, the rows show RGB images, UniBuild-derived building polygons, and corresponding OSM building polygons.

## B. Zero-shot Building Change Detection

Although the model is trained only for single-date building extraction, we further evaluate its zero-shot ability for building change detection on LEVIR-CD [63] at a resolution of 0.5 m. Given a bi-temporal image pair $( I _ { t _ { 1 } } , I _ { t _ { 2 } } )$ , we independently predict two-channel building probability maps $P _ { t _ { 1 } }$ and $P _ { t _ { 2 } }$ using UniBuild with a DINOv3-Base backbone trained on multiple datasets in Sec. IV-C. The binary building masks $M _ { t _ { 1 } }$ and $M _ { t _ { 2 } }$ are obtained by pixel-wise argmax over the background and building output channels. The change map is then produced by applying an XOR operation to the two predicted masks:

$$
C = M _ { t _ { 1 } } \oplus M _ { t _ { 2 } } .\tag{38}
$$

This protocol does not use any bi-temporal change labels during training, and therefore directly evaluates whether the learned building representation can support downstream temporal reasoning in a zero-shot manner.

As shown in Table XI, UniBuild achieves an F1 score of 85.14% and an IoU of 74.12% on LEVIR-CD without using any change-detection labels during training, with a qualitative samples shown in Fig. 11. This result indicates that the learned building representation has strong zero-shot generalization ability and can be directly transferred from single-date building extraction to bi-temporal building change detection.

TABLE XI  
ZERO-SHOT BUILDING CHANGE DETECTION RESULTS ON LEVIR-CD. THE MODEL IS TRAINED ONLY FOR SINGLE-DATE BUILDING EXTRACTION AND IS NOT TRAINED WITH BI-TEMPORAL CHANGE LABELS.
<table><tr><td>Dataset</td><td>Precision</td><td>Recall</td><td>F1</td><td>IoU</td></tr><tr><td>LEVIR-CD</td><td>88.26</td><td>82.23</td><td>85.14</td><td>74.12</td></tr></table>

![](images/9a6ae68189b55fb9b0c966b250affbba60782645d33bb69e68dc987db73dd8c4.jpg)  
Fig. 11. Zero-shot building change detection on LEVIR-CD.
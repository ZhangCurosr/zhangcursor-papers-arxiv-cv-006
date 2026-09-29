# VL-AcneSeg: A Vision-Language Framework for Region-Aware Acne Lesion Segmentation

Sukju Oh, Soo Ick Cho, Dae Hun Suh, and Sukkyu Sun

Abstract— Acne assessment is crucial for clinical decision-making, yet traditional grading and counting are subjective and fail to account for lesion size. While areabased assessment has emerged as a promising alternative, acne segmentation has continued to rely on generalpurpose architectures. To address this gap, we propose VL-AcneSeg, a multimodal framework for acne lesion segmentation that leverages CLIP and region-level text prompts to incorporate spatial priors, enabling lesions to be localized across the whole face. Because region-level prompts indicate which facial areas contain lesions, we report a single global prompt, which requires no such information, as our primary setting. On our internal clinical dataset, VL-AcneSeg achieves a Dice score of 0.5082 and an IoU of 0.3407 under this protocol, the highest among all compared methods, including recent vision-language segmentation methods that are themselves given region-level prompts; region-level prompting raises these to 0.5296 and 0.3602. Moreover, lesion area measurements derived from our segmentation correlate with IGA scores at a level comparable to expert annotations (Pearson r = 0.719 versus 0.658). Notably, our framework maintains consistent performance across external validation datasets, performing reliably even on uncontrolled smartphone images without requiring additional training or fine-tuning. By pairing a protocol that requires no lesion-location information with area-based severity estimation, this work provides a foundation for objective acne assessment outside the clinic. Our imple-

Input Prompt: “acne lesion on the left cheek”

![](images/9ec8ec78e6cb547106df906cdb711cf93c30fd7f176330119b394441b32e89e7.jpg)

![](images/0a1bbce2612013cd7808930a35968c6b66b43f8853cb3cd0c02e465dee984189.jpg)  
(a) Ground Truth

![](images/18ba5538e34675287b8d83a4c29d5816ee71aea2e4c3872f8c62816667bf40c0.jpg)  
(b) Localization  
(c) Refinement  
Fig. 1: Conceptual framework of VL-AcneSeg. Given a regional text prompt, VL-AcneSeg localizes potential lesion sites through CLIP-based semantic alignment (Middle) and then performs refinement to produce the segmentation mask (Right).

mentation is publicly available at: https://github.com/ sukjuoh/VL-AcneSeg

Index Terms— Acne Segmentation, Vision-Language Model, Medical Imaging.

## I. INTRODUCTION

Acne vulgaris is one of the most prevalent chronic skin disorders worldwide [2], ranking among the top three most common skin conditions and affecting up to 85% of adolescents and young adults during their lifetime [3]–[5]. Because it predominantly involves the facial region, where lesions are highly visible, acne can lead to considerable psychosocial stress [6]. Acne lesions are broadly categorized into non-inflammatory types (e.g., comedones), which can be removed through simple extraction, and inflammatory types (e.g., papules, pustules, nodules), which often require pharmacological treatment depending on severity. Since inflammatory lesions are more directly tied to disease progression and therapeutic decisions, developing automated tools that can precisely quantify them is a critical priority for objective acne management.

Severity is assessed in clinical practice by global grading or by lesion counting, and each has its own limitation. Global grading compares a patient’s presentation against standardized reference cases [7] and carries the inter- and intra-observer variability that entails [8], [9]. Counting individual lesions by type and region is more quantitative [10], [11] and has been automated by recent detection methods [1], [12], [13], but it records a lesion as present or absent, so a small papule and a large confluent nodule contribute equally [14], [15].

Area-based assessment is a promising alternative because it reflects lesion extent as well as lesion number. Gazeau et al. combined lesion areas with lesion-specific severity scores into an overall index, offering a more objective basis for severity assessment [16].

Despite this potential, acne segmentation itself has received limited attention, largely due to the scarcity of annotated data and the inherent difficulty of the task [17], [18]. Pixel-level annotation of numerous small lesions requires dermatological expertise and is both costly and time-consuming. In addition, variations in skin tone, illumination, and imaging conditions introduce substantial appearance variability, while lesions often have blurred boundaries, irregular shapes, and subtle contrasts with surrounding skin, with confounding factors such as moles, scars, and pigmentation further complicating discrimination.

Reflecting these challenges, previous studies on acne segmentation have typically relied on general-purpose architectures and conventional training strategies, often applied to cropped patches around lesions rather than full-face images [19]–[21]. While such approaches simplify the task, they limit the ability to capture the global distribution of lesions, which is essential for area-based severity assessment. Even fullface models [16], [22] have introduced little segmentation methodology tailored to the specific challenges of acne.

Vision-language models offer a way forward, since they allow simple textual cues—such as the approximate facial region in which lesions are present—to be supplied as spatial priors for segmentation. Existing vision-language segmentation methods, however, are built around a different question: they use text to specify which class to segment, and their alignment relies on the contrast between classes, which is absent when there is a single lesion class. What acne does offer is a face whose anatomy is stable and nameable, so text can instead specify where the target may appear. Such region-level descriptions carry only coarse spatial information at inference time, yet can narrow where the model searches for lesions across the face. Accordingly, we propose VL-AcneSeg, a multimodal framework that utilizes region-level text prompts to guide the segmentation of inflammatory lesions. As illustrated in Fig. 1, the model aligns these textual prompts with visual features via CLIP to identify potential lesion sites, which are then refined into precise segmentation masks. Under region-level prompting the design additionally yields a severity estimate for each facial region. Our contributions are as follows:

• We recast text-guided segmentation for a setting in which open-vocabulary alignment does not apply. Existing paradigms use text to specify which class to segment, which carries little information when there is a single lesion class; we instead use it to specify where the target may appear, so that the text channel supplies a spatial rather than a categorical constraint.

• We adapt the patch-wise similarity paradigm to acne through tiled high-resolution CLIP encoding, layertargeted V-V attention and CLS-token-guided crossattention, so that the alignment operates at the resolution the lesions demand.

• We report a deployment-realistic protocol that requires no reference-derived information, evaluate it on external clinical and smartphone datasets without retraining, and show that the resulting lesion areas track Investigator’s Global Assessment severity at the level of the reference annotations.

## II. RELATED WORK

## A. Acne Lesion Segmentation

Acne lesion segmentation has been explored only to a limited extent compared to other dermatological conditions. Existing studies often rely on cropped image patches rather than full-face images, limiting their ability to capture the global lesion distribution [19]–[21], and typically adopt standard architectures such as U-Net without acne-specific design [22]. Gazeau et al. [16] employed a conventional segmentation backbone within their area-based severity framework, but their focus was on the evaluation methodology rather than the segmentation model itself. Moreover, absolute segmentation metrics reported in this domain remain modest across studies, reflecting the intrinsic difficulty of distinguishing acne from visually similar skin features such as PIH, scars, and pores [16], [22]. A systematic review covering 2017–2025 screened 345 articles and identified 29 eligible studies, the majority addressing severity grading or lesion counting rather than pixel-level segmentation [18]. Pixel-level acne segmentation has therefore received limited attention, and the methods that do address it were not designed for the setting that areabased assessment requires: the whole face at once, with lesions that are numerous and easily confused with the marks they leave behind.

## B. Open-Vocabulary and Text-Guided Segmentation

Vision-language models such as CLIP [23] and ALIGN [24] learn joint image–text representations that support openvocabulary understanding and zero-shot transfer [25]–[27]. This has motivated their use in dense prediction, through transformer-based fusion [28]–[30] and open-vocabulary paradigms such as OVSeg [31] and ODISE [32], which match class-agnostic region proposals with text embeddings.

However, these paradigms face significant practical and logical hurdles in acne segmentation. For ROI-based methods, if a mask generator can successfully propose an ROI for a specialized single-class lesion, the subsequent text-alignment step becomes effectively redundant. Furthermore, the visual features within an acne lesion are often too localized and subtle to provide sufficient discriminative information for meaningful alignment, while random masking strategies like MaskCLIP [33] fail to maintain the structural integrity of such small-scale lesions during training. Moreover, the lack of distinct positivenegative pairs in localized skin imagery makes it difficult to establish a viable contrastive learning framework.

More recently, CAT-Seg [34] introduced a patch-wise cosine similarity paradigm that formulates segmentation as a direct alignment between CLIP’s visual and textual embeddings, with subsequent extensions such as SED [35] and ESC-Net [36] refining this formulation for natural-image openvocabulary settings; we compare against SED directly, whereas ESC-Net couples the cost volume with SAM, which enters our comparison as a separate fine-tuned baseline. The most fundamental distinction between our approach and CAT-Seg lies in the role of text: CAT-Seg consumes class-level prompts that specify what object to segment, whereas VL-AcneSeg consumes region-level prompts that specify where the target may appear. Building on this distinction, our framework repurposes the alignment itself: the text channel supplies a spatial constraint rather than a categorical one, which extends the formulation to dense single-class segmentation, a setting where prior open-vocabulary methods struggle for want of inter-class contrast. The architectural components that follow are adaptations required to make this alignment operate at the resolution the task demands. A complementary line of work prompts CLIP with visual rather than textual references: PBIP [37] constructs class prototypes from a training image bank and uses them as image prompts for weakly supervised histopathological segmentation. Such prototype prompts encode appearance, whereas the region-level prompts used here encode anatomical location, and the two forms of guidance are therefore complementary.

## C. Text-Guided Segmentation Model in the Medical Domain

In the medical domain, text has mainly been used to identify the target structure. LViT [38] integrates BERT [39] text embeddings into a hybrid CNN-Transformer framework, while VLM-based approaches such as MedCLIP-SAM [40], SegICL [41], and OMT-SAM [42] combine pretrained text encoders with segmentation frameworks for zero-shot or prompt-based segmentation of anatomical structures.

More recent studies have begun to exploit the spatial content of clinical text explicitly. TVE-Net [43] converts the location descriptions in radiology reports into a “text view” through a hand-designed positional probability function, assigning a lesion probability to each image region so that textual spatial information is injected into the segmentation network. STPNet [44] instead retrieves multi-scale textual descriptions from a curated medical text repository during training and thereby removes the need for text input at inference, while EviVLM [45] introduces evidential learning to quantify and mitigate the modality gap between image and text representations.

Our framework shares with TVE-Net the premise that textual location information can serve as a spatial prior, but realises it differently. TVE-Net maps free-text radiology reports to a probability map through a hand-designed positional function, whereas VL-AcneSeg keeps the prior inside the joint embedding space: fixed-template region prompts are encoded by the CLIP text encoder and aligned with patch embeddings through cosine similarity, so that the prior is a learned visual–textual correspondence rather than a predefined geometric mapping, and no narrative report is required. These advances have moreover addressed radiology, where lesions are comparatively large and few in number; to our knowledge text-guided segmentation has not been applied to acne, whose lesions are small, numerous and distributed across the whole face.

![](images/321832975b61430d730b519885909907e5a6aee15b87eaecb94d1a6bb1164719.jpg)  
Fig. 2: Facial regions were defined using Mediapipe Face Mesh [46], and lesion masks were assigned to each region according to their centroid location. Region-wise masks and text prompts (“acne lesion on the $\{ r e g i o n \} ^ { \ast } )$ were then generated to provide spatial guidance for segmentation.

## III. METHOD

## A. Data Preparation

We collected standardized clinical photographs from 258 patients (38.4% male; mean age 22.7 ± 5.97 years) who visited the Department of Dermatology, Seoul National University Hospital, using Canon EOS 550D and Nikon D7100 cameras. All participants were of Korean ethnicity, and Fitzpatrick phototype was not routinely recorded. The same cohort was described previously in the context of automated lesion detection and counting [1]; the present study is the first to derive pixel-level segmentation masks from these data. A total of 1,213 images carry 20,699 annotated acne lesions. The lesion inventory originates from the study of this cohort reported in [1], where two dermatology residents independently marked each lesion. For the present study these annotations were converted into pixel-level masks with LabelMe [47], and a board-certified dermatologist reviewed every image, corrected the delineations and adjudicated the final set. The reference standard is therefore fixed by a single senior reader rather than by agreement among readers. The annotations cover five lesion types (closed and open comedones, papules, nodules/cysts, and pustules). Because clinical acne severity assessment is based primarily on inflammatory lesions, and because comedones are frequently indistinguishable at the pixel level in standardized clinical photographs, this study targets inflammatory lesions only (papules, nodules/cysts, and pustules); patients without any inflammatory lesion were excluded, leaving 222 patients, 821 images and 4,655 annotated lesion instances.

![](images/3d80d60db40e4b82ce95d66adcf96bba29e7a5c7870a00e60d84f750fae82cac.jpg)  
Fig. 3: Overview of VL-AcneSeg. Overlapping tiles are encoded by CLIP and reassembled into a unified feature map, whose patch-wise cosine similarity with the text embeddings yields the localization map; intermediate vision features and the CLS token provide additional guidance. A Swin Transformer, a CNN bottleneck and a decoder then refine this map into the lesion mask.

To generate region masks and text prompts, each facial image was divided into anatomical regions (forehead, cheeks, nose, and chin) using MediaPipe Face Mesh landmarks [46]. As illustrated in Fig. 2, the center coordinate of each lesion mask was used to determine its corresponding facial region, and the mask was assigned accordingly.

For every region containing one or more lesions, an individual text prompt was generated following a fixed template format, “acne lesion on the {region}.” For example, if acne lesions were present on the forehead, chin, and neck, three separate prompts (“acne lesion on the forehead,” “acne lesion on the chin,” and “acne lesion on the neck”) were produced. These prompts indicate the approximate location of lesions

across facial regions.

## B. Model Overview

Our goal is to segment acne lesions from facial images by leveraging textual descriptions of facial regions as spatial priors. Given an input image and corresponding region-level text prompts, our model outputs a pixel-wise segmentation mask indicating the locations of acne lesions.

The overall architecture is illustrated in Fig. 3. The input image is divided into tiles, encoded by the CLIP vision encoder and reassembled into a visual feature map, whose patch-wise cosine similarity with the CLIP text embeddings yields a localization map that is then passed to a Swin Transformer.

Intermediate visual features and the CLS tokens provide additional guidance, and a cross-attention module conditioned on the text embeddings refines the representation before the mask decoder, so that region-level cues are combined with visual features in a patch-aligned manner.

## C. Localization: High-Resolution Features and Patch-wise Similarity

Acne lesions are typically small and have blurry boundaries, making high-resolution input crucial for precise segmentation. However, the CLIP vision encoder Φ (ViT-L/14@336px) does not support inputs larger than $3 3 6 \times 3 3 6$ . To overcome this limitation, we divide each input image $X \in \mathbb { R } ^ { 3 \times H \times W }$ into overlapping tiles of size $k \times k ~ ( k = 3 3 6 )$ and process each tile individually through Φ. The tiling layout follows from two constraints rather than from a search: the tile size is fixed at 336 by the encoder, and clinical facial photographs have a 3:4 width-to-height ratio, so tiling the image with square patches requires four rows and three columns. With the longer side resampled to 1024, the smallest such layout is the $4 \times 3$ configuration used here; coarser layouts reduce the pixel density per tile, while the next admissible refinement quadruples the number of CLIP forward passes for tiles that each cover less anatomical context. For each tile $P _ { i , j }$ , we obtain the CLS token and patch embeddings:

$$
[ z _ { i , j } ^ { \mathrm { c l s } } , Z _ { i , j } ] = \Phi ( P _ { i , j } ) ,\tag{1}
$$

where $Z _ { i , j } ~ \in ~ \mathbb { R } ^ { c \times h \times w }$ is the token grid for the (i, j)-th tile $( h ~ = ~ w ~ = ~ 2 4 , ~ c ~ = ~ 1 0 2 4 )$ . All tile embeddings are then reassembled into a single feature map S by placing them at their corresponding spatial locations and averaging the overlapping regions using a Gaussian weight window G:

$$
S = \frac { \sum _ { i , j } Z _ { i , j } \odot G _ { i , j } } { \sum _ { i , j } G _ { i , j } + \epsilon } ,\tag{2}
$$

where ⊙ denotes element-wise multiplication, $G _ { i , j }$ is a 2D Gaussian kernel of size $h \times w$ centered on tile (i, j) with standard deviation $\sigma = h / 4$ that downweights border patches to reduce stitching artifacts at tile boundaries, and $\epsilon = 1 0 ^ { - 6 }$ is a small constant ensuring numerical stability. Finally, the stitched feature map S is upsampled to a fixed resolution and normalized at each spatial location:

$$
S _ { \mathrm { n o r m } } ( x , y ) = \frac { S ( x , y ) } { \| S ( x , y ) \| _ { 2 } } .\tag{3}
$$

Lesions are localized by aligning visual features with the text prompts (Fig. 1 (b)). The $N _ { T }$ region prompts $\{ T _ { k } \} _ { k = 1 } ^ { N _ { T } }$ are passed through the CLIP text encoder Γ:

$$
\{ t _ { k } = \Gamma ( T _ { k } ) \in \mathbb { R } ^ { C } \} _ { k = 1 } ^ { N _ { T } } .\tag{4}
$$

Let $S \in \mathbb { R } ^ { C \times H \times W }$ denote the visual feature map extracted from the image encoder Φ. Before computing similarity, both the visual features $S$ and text embeddings $t _ { k }$ are normalized along the channel dimension. For each text embedding $t _ { k }$ , we compute a patch-wise cosine similarity map with the visual features as:

$$
M _ { k } ( h , w ) = \frac { S _ { h , w } ^ { \top } t _ { k } } { \| S _ { h , w } \| _ { 2 } \| t _ { k } \| _ { 2 } } ,\tag{5}
$$

yielding a set of similarity maps:

$$
M = \{ M _ { k } \} _ { k = 1 } ^ { N _ { T } } \in \mathbb { R } ^ { H \times W \times N _ { T } } ,\tag{6}
$$

each encoding the alignment between visual patches and one region-level description.

## D. Feature Enhancement

CLIP [23] aligns visual and textual representations through a contrastive objective, but its attention is biased toward large and salient regions and is often dominated by a few tokens, which suppresses the fine-grained local features on which small lesions depend—a limitation also noted in Anomaly-CLIP [48].

To address this issue, we adopt the V-V attention mechanism introduced by AnomalyCLIP, but redesign its deployment strategy for fine-grained lesion segmentation. Whereas AnomalyCLIP applies V-V attention across multiple intermediate layers for global anomaly detection, we apply it only at the final feature extraction layer. The reason is that V-V attention replaces query–key routing with value–value affinity, so each token is re-expressed as a weighted average of tokens with similar values. Applied once, this suppresses the few dominant tokens that otherwise absorb the attention mass; applied repeatedly, the same averaging acts as a low-pass filter over the token grid, and the patch-aligned spatial structure that dense prediction depends on is progressively smoothed away. Anomaly detection tolerates this because it requires only a coarse image-level score, whereas lesions a few pixels across do not. Table III confirms the distinction empirically. Specifically, let $V ~ \in ~ \mathbb { R } ^ { N \times d }$ denote the value embeddings extracted from this target layer of the CLIP vision encoder, where N is the number of tokens and $d$ is the embedding dimension. We compute a token-token affinity matrix through a scaled dot product between the value vectors:

$$
A _ { v v } = \mathrm { S o f t m a x } \left( \frac { V V ^ { \top } } { \sqrt { d } } \right) ,\tag{7}
$$

and use it to refine the visual features:

$$
V ^ { \prime } = A _ { v v } V .\tag{8}
$$

The refined features are then passed through the original output projection of that layer and added to the original layer output to form the final feature map.

Fine-grained local information diminishes with layer depth, so we additionally take the feature map of the 12th CLIP layer as guidance and fuse it before the bottleneck (Section III-E).

We further utilize CLS tokens to inject coarse spatial information derived from the image tiles. Since each image is divided into $4 \times 3$ tiles and individually processed by the CLIP vision encoder, we obtain one CLS token $c _ { n } \in \mathbb { R } ^ { C }$ for each tile $n = 1 , \ldots , N$ . Given the normalized text embeddings $\{ t _ { k } \} _ { k = 1 } ^ { N _ { T } }$ , we compute the cosine similarity between all CLS tokens and text embeddings to produce tile-text similarity weights:

$$
w _ { n , k } = \frac { \exp ( c _ { n } ^ { \top } t _ { k } ) } { \sum _ { n ^ { \prime } = 1 } ^ { N } \exp ( c _ { n ^ { \prime } } ^ { \top } t _ { k } ) } .\tag{9}
$$

These weights are then used to compute a weighted combination of CLS tokens for each text embedding, representing the pooling operation (P) illustrated in Fig. 3:

$$
\bar { c } _ { k } = \sum _ { n = 1 } ^ { N } w _ { n , k } c _ { n } .\tag{10}
$$

The resulting $\bar { c _ { k } }$ encodes spatial cues indicating which tiles are semantically aligned with each text prompt. Finally, $\bar { c _ { k } }$ is concatenated with the corresponding text embedding $t _ { k }$ , and this combined representation is used as the query for the crossattention module.

## E. Refinement and Mask Decoding

The patch-wise similarity maps $\begin{array} { r c l } { M } & { \in } & { \mathbb { R } ^ { B \times N _ { T } \times H \times W } } \end{array}$ are first passed through a convolution layer to introduce a channel dimension, resulting in $M ^ { \prime } = \mathrm { C o n v } ( M ) \in$ $\mathbb { R } ^ { ( B \times N _ { T } ) \times C _ { M } \times H \times W }$ . In parallel, the original CLIP vision encoder feature map $S$ is replicated and concatenated with $M ^ { \prime }$ to form the input feature $F _ { \mathrm { i n } } .$ This feature map is processed by a Swin Transformer with W-MHSA and SW-MHSA blocks to capture long-range spatial dependencies (Fig. 3).

The guidance feature $F _ { \mathrm { g u i d e } }$ , taken from the 12th layer of the CLIP vision encoder, is passed through a convolution layer, replicated $N _ { T }$ times and concatenated with $F _ { \mathrm { s w i n } }$ before the bottleneck:

$$
F _ { \mathrm { b o t t l e - i n } } = \mathrm { C o n c a t } [ F _ { \mathrm { s w i n } } , \mathrm { R e p e a t } _ { N _ { T } } ( F _ { \mathrm { g u i d e } } ) ] ,\tag{11}
$$

where $F _ { \mathrm { g u i d e } } ~ \in ~ \mathbb { R } ^ { B \times C _ { g } \times H ^ { \prime } \times W ^ { \prime } }$ . The bottleneck refines this representation to integrate spatial similarity cues with lowlevel visual features:

$$
\begin{array} { r } { F _ { \mathrm { b o t t l e } } = \mathrm { B o t t l e n e c k } ( F _ { \mathrm { b o t t l e . i n } } ) \in \mathbb { R } ^ { ( B \times N _ { T } ) \times C _ { b } \times H ^ { \prime } \times W ^ { \prime } } . } \end{array}\tag{12}
$$

The weighted CLS token $\bar { c } _ { k }$ and the text embedding $t _ { k }$ form the query $Q \colon$

$$
\begin{array} { r } { Q = \{ { \mathrm { C o n c a t } } [ \bar { c } _ { k } , t _ { k } ] \} _ { k = 1 } ^ { N _ { T } } \in \mathbb { R } ^ { N _ { T } \times C _ { q } } . } \end{array}\tag{13}
$$

which is broadcast to $Q ^ { \prime } \in \mathbb { R } ^ { ( B \times N _ { T } ) \times C _ { q } \times H ^ { \prime } \times W ^ { \prime } }$ and interacts with the refined visual features through cross-attention:

$$
F _ { \mathrm { f u s e d } } = \mathrm { C r o s s A t t n } ( Q ^ { \prime } , F _ { \mathrm { b o t t l e } } , F _ { \mathrm { b o t t l e } } ) .\tag{14}
$$

The mask decoder produces the segmentation mask ${ \hat { Y } } \colon$ the fused feature $F _ { \mathrm { f u s e d } }$ is processed through Multi-Head Self-Attention (MHSA) and Layer Normalization (LN), then concatenated with the similarity map $M ^ { \prime }$ to anchor the predictions to the initial localization results. The final mask is produced through learned upsampling via a Convtranspose layer followed by a convolution:

$$
\begin{array} { r } { \hat { Y } = \mathrm { C o n v } ( \mathrm { C o n v T r a n s p o s e ( } } \\ { \mathrm { C o n c a t ( L N ( M H S A ( } { F } _ { \mathrm { f u s e d } } ) ) , M ^ { \prime } ) ) ) . } \end{array}\tag{15}
$$

## F. Loss Function

We supervise the network with two complementary losses: a global segmentation loss over the full-face mask, and a regionspecific loss for each regional prediction.

For the global segmentation, we adopt the weighted IoU and weighted BCE losses from PraNet [49], which emphasize hard pixels by upweighting challenging regions:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { w I o U } } ( \hat { Y } , Y ) + \mathcal { L } _ { \mathrm { w B C E } } ( \hat { Y } , Y ) .\tag{16}
$$

## TABLE I

COMPOSITION OF THE INTERNAL SPLITS AND THE TWO EXTERNAL TEST SETS, COUNTED OVER THE INFLAMMATORY LESION MASKS USED IN THIS STUDY. THE FULL ANNOTATED COHORT COMPRISES 258 PATIENTS, 1,213 IMAGES AND 20,699 LESIONS ACROSS FIVE TYPES. THE INTERNAL DATASET WAS DIVIDED PATIENT-WISE, AND EACH EXTERNAL IMAGE ORIGINATES FROM A DISTINCT SUBJECT.
<table><tr><td>Split</td><td>Patients</td><td>Images</td><td>Lesions</td></tr><tr><td>Internal (Seoul National University Hospital)</td><td></td><td></td><td></td></tr><tr><td>Train</td><td>190</td><td>703</td><td>3,814</td></tr><tr><td>Validation</td><td>16</td><td>60</td><td>494</td></tr><tr><td>Test</td><td>16</td><td>58</td><td>347</td></tr><tr><td>Total</td><td>222</td><td>821</td><td>4,655</td></tr><tr><td>External (AI Hub)</td><td></td><td></td><td></td></tr><tr><td>Controlled</td><td>71</td><td>71</td><td>171</td></tr><tr><td>Real-world</td><td>56</td><td>56</td><td>101</td></tr></table>

For region-specific supervision, we apply focal loss [50] (with $\alpha ~ = ~ 0 . 7 5 )$ to each regional prediction $\hat { Y } _ { k }$ associated with a text prompt T<sub>k</sub>:

$$
\mathcal { L } _ { \mathrm { r e g i o n } } = \frac { 1 } { N _ { T } } \sum _ { k = 1 } ^ { N _ { T } } \mathcal { L } _ { \mathrm { f o c a l } } ( \hat { Y } _ { k } , Y _ { k } ) ,\tag{17}
$$

where $Y _ { k }$ is the ground-truth mask for the k-th facial region and $N _ { T }$ is the number of region-level prompts. This term supervises the similarity map M directly, encouraging alignment between visual patches and their regional prompts, while the focal weighting emphasizes small or ambiguous lesions.

The final objective combines both terms:

$$
\mathcal { L } = 0 . 6 \mathcal { L } _ { \mathrm { t o t a l } } + 0 . 4 \mathcal { L } _ { \mathrm { r e g i o n } } .\tag{18}
$$

## IV. EXPERIMENTS

## A. Setup

We evaluated our model on three datasets. The first is the internal test set described in Section III-A. The internal dataset was split into training, validation, and test sets at an 8:1:1 ratio on a patient-wise basis to prevent data leakage across multiple images from the same subject; the split was defined on the full cohort of 258 patients, and after excluding patients without inflammatory lesions the realized split comprised 190, 16 and 16 patients. Investigator’s Global Assessment (IGA) scores were not recorded at acquisition and were assigned retrospectively at the visit level by three board-certified dermatologists, for the internal test set only, since IGA is used solely for the clinical correlation analysis (Section IV-G) and never for model training or selection. The 58 internal test images correspond to 22 visits from 16 patients, of which 3, 7, 8 and 4 were graded 1 to 4 on the standard five-point scale and none as grade 0; the grading protocol is given in Section S1 of the Supplementary Material. The other two datasets were derived from the AI Hub Korean Skin Condition Measurement Dataset<sup>1</sup>, developed to support facial skin analysis for Korean individuals.

From this collection we constructed two external test sets, one for a controlled imaging setting and one for an uncontrolled, real-world setting. Because the collection contains many images without visible inflammatory lesions, candidates were screened with an acne detection model trained on the public ACNE04 dataset [51] and then sampled at random; the dermatologist who annotated the internal set reviewed the sampled images and excluded those without confirmed inflammatory acne, yielding 71 controlled-setting and 56 realworld images. The screening model predicts bounding boxes rather than masks and its outputs were used only for candidate selection, never in the annotations or in any evaluation. Pixellevel masks were newly constructed for the present study by the same board-certified dermatologist who adjudicated the internal set, so both sets share one reference reader; unlike the internal set, no prior box inventory was available here, so the lesions were located as well as delineated. Text prompts were generated as described in Section III-A, and the composition of every split is summarized in Table I. The two subsets differ from the internal cohort in lesion burden and probe different kinds of imaging shift: the controlled subset was acquired with digital cameras under a setup comparable to the internal cohort, whereas the real-world subset was acquired with smartphones. The screening procedure, the resulting sampling bias and the available metadata are detailed in Section S1 of the Supplementary Material.

## B. Evaluation Metrics

We used the Dice coefficient and Intersection over Union (IoU):

$$
{ \mathrm { D i c e } } = { \frac { 2 | Y \cap { \hat { Y } } | } { | Y | + | { \hat { Y } } | } } ,\tag{19}
$$

$$
\mathrm { I o U } = \frac { | Y \cap \hat { Y } | } { | Y \cup \hat { Y } | } .\tag{20}
$$

We further report pixel-wise Precision and Recall, together with lesion-level counts per image. A predicted connected component is counted as a true positive (TP/img) if it overlaps a reference lesion in at least one pixel, and as a false positive (FP/img) otherwise. Specificity is not reported: background pixels dominate facial images, so it exceeds 0.999 for every method and carries no information. Let Ω denote the set of valid image pixels, Y the ground-truth lesion mask and $\hat { Y }$ the predicted mask. The pixel-wise counts are defined as $\mathrm { T P = }$ $\mathsf { \bar { | } } Y \cap \hat { Y } | , \mathrm { F P } = | \hat { Y } \backslash Y | , \mathrm { F N } = | Y \backslash \hat { Y } |$ , and $\mathrm { T N } = | \Omega | - | Y \cup { \hat { Y } } |$ giving

$$
\mathrm { P r e c i s i o n } = \frac { \mathrm { T P } } { \mathrm { T P } + \mathrm { F P } } ,\tag{21}
$$

$$
{ \mathrm { R e c a l l } } = { \frac { \mathrm { T P } } { \mathrm { T P } + { \mathrm { F N } } } } .\tag{22}
$$

All metrics are aggregated over the entire test set by accumulating TP, FP, FN, and TN across images before computing the ratios (micro-averaging). Lesion-level recall, precision and F1 restricted to the smallest quartile of lesions (below $5 0 0 \mathrm { p x } ^ { 2 } )$ , together with a threshold-free comparison based on the area under the precision–recall curve, are reported for all methods in Table S5 of the Supplementary Material. Zero-padded regions (Section IV-C) are excluded from Ω, so the denominator is the same effective area for every method.

## C. Implementation and Training Protocol

The model was implemented in PyTorch and trained on an NVIDIA A6000 GPU (48 GB). We used the AdamW [52] optimizer with a learning rate of $2 \times 1 0 ^ { - 4 }$ for the main network and $2 \times 1 0 ^ { - 6 }$ for the CLIP encoders, for 40 epochs with a batch size of 4. The optimizer’s betas were set to (0.9, 0.999).

Baselines used the default hyperparameters of their original papers with no additional search, and were initialized from the pretrained weights those implementations provide— ViT-B/16 for TransUNet, Swin Transformer for Swin-UNet, MiT for SegFormer, ImageNet ResNet encoders for PSPNet, DeepLabV3+ and CE-Net, VMamba for Swin-UMamba†, and DeiT-Base [53] with Bio ClinicalBERT [54] for EviVLM— while U-Net, U-Net++ and nnU-Net were trained from random initialization. SAM was initialized from the pretrained ViT-L checkpoint, with its image encoder and mask decoder fully fine-tuned and the prompt encoder left unused, as no point or box prompts were supplied. Every model, including ours, used the identical patient-wise 8:1:1 split and no data augmentation. For each model we selected the checkpoint with the highest validation IoU, and fixed the binarization threshold at 0.5. CAT-Seg and SED share our input resolution and $4 \times 3$ tiling, so the three CLIP-based methods are compared at matched resolution. All others take the image resized to a height of 1024, zero-padded along the width where the architecture requires a fixed input shape, with padded regions excluded from the loss and the evaluation.

## D. Results

As summarized in Table II, VL-AcneSeg attains the highest IoU and Dice scores in every evaluation setting. While traditional architectures like U-Net achieve high recall, they tend to be indiscriminate and often fail to distinguish acne lesions from visually similar conditions such as scars, postinflammatory hyperpigmentation (PIH), and rashes. As shown in Fig. 4, baseline models frequently misclassify these nonacne conditions as lesions, whereas VL-AcneSeg correctly identifies true acne through vision-language alignment.

Because the vision-language baselines are all supplied with region-level prompts, they are compared against VL-AcneSeg under the same regional prompting condition, whereas the remaining methods, which take no text input, are compared against the global-prompt configuration; in both cases the difference is assessed by paired patient-level cluster bootstrap, with the two-sided $p$ value taken as twice the proportion of resamples in which the difference lies on the opposite side of zero from the point estimate. Every baseline other than SAM differs significantly: $p \le 0 . 0 0 5$ for the convolutional, Transformer and Mamba architectures, and 0.013, 0.034 and 0.048 for CAT-Seg, SED and EviVLM. SAM is the one method from which the proposed model is not distinguished $( p = 0 . 5 8 )$ . We therefore retrained it and the global-prompt configuration three times with different seeds. IoU was 0.3460 ± 0.0049 against $0 . 3 2 1 9 \pm 0 . 0 0 5 2$ and Dice $0 . 5 1 4 1 \pm 0 . 0 0 5 4$ against $0 . 4 8 7 1 \pm 0 . 0 0 6 5 ;$ the variation across seeds is about five times smaller than the gap between them, and the ordering held every time. The remaining configurations were trained once, and Table II reports that run. The direction of every comparison favours VL-AcneSeg and the ordering is preserved across all three datasets, although with 16 test patients the intervals around the smaller differences remain wide; these are reported in Table S4 of the Supplementary Material.

TABLE II  
OUANTITATIVE COMPARISON OF ACNE LESION SEGMENTATION ON THE INTERNAL AND EXTERNAL DATASETS. SHADED ROWS (\*) ARE VISION-LANGUAGE METHODS GIVEN REGION-LEVEL PROMPTS DERIVED FROM THE REFERENCE MASKS, AN ORACLE SETTING; CAT-SEG AND SED USE THE SAME TILING SCHEME AND INPUT RESOLUTION AS THE PROPOSED METHOD, SO THAT THE CLIP-BASED METHODS ARE COMPARED AT MATCHED RESOLUTION. ALL REMAINING ROWS, INCLUDING VL-ACNESEG (GLOBAL), REQUIRE NO REFERENCE-DERIVED INFORMATION. TP/IMG AND FP/IMG ARE THE NUMBERS OF PREDICTED LESION COMPONENTS PER IMAGE THAT DO OR DO NOT OVERLAP A REFERENCE LESION; NEITHER IS RANKED. POINT ESTIMATES ARE COMPUTED OVER ALL TEST-SET PIXELS; 95% CONFIDENCE INTERVALS ARE OBTAINED BY PATIENT-LEVEL CLUSTER BOOTSTRAP (1,000 RESAMPLES). BEST VALUES ARE IN BOLD AND SECOND-BEST ARE UNDERLINED.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Backbone</td><td colspan="5">Internal Data (Training &amp; Evaluation)</td><td colspan="6">External Data (Evaluation Only)</td><td colspan="5"></td></tr><tr><td>Dice</td><td>Precision Recall</td><td>TP/img FP/img</td><td></td><td>IoU</td><td></td><td>Controlled Setting (1)</td><td>Precision</td><td></td><td>Recall TP/img FP/img</td><td>IoU</td><td>Real-world Setting (2) Dice</td><td>Precision Recall</td><td></td><td></td><td>TP/img FP/img</td></tr><tr><td>VL-AcneSeg (Region)*</td><td>ViT-L</td><td>IoU 0.3602 [0.309, 0.399] 0.5296 [0.472, 0.570]</td><td>0.5393</td><td></td><td></td><td>0.4232 [0.376, 0.470]</td><td>Dice 0.5948 [0.546, 0.640]</td><td></td><td>0.5794</td><td>1.64</td><td>0.60</td><td>0.3044 [0.2476, 0.3483]</td><td>0.4882</td><td>0.4471</td><td>1.50</td><td></td><td>0.82</td></tr><tr><td>SED [35]*</td><td>ConvNeXt-L</td><td>0.3182 [0.255, 0.373] 0.4828 [0.406, 0.544]</td><td>0.5202</td><td>0.3917</td><td>4.29</td><td>2.48 0.3070 [0.263, 0.355]</td><td></td><td>0.4698 [0.416, 0.524]</td><td>0.6110 0.7509</td><td>0.3419 1.97</td><td>1.51</td><td>0.1650 [0.121, 0.229]</td><td>0.4667 [0.3969, 0.5166] 0.2833 [0.215, 0.373]</td><td>0.6365</td><td>0.1822</td><td>0.98</td><td>0.32</td></tr><tr><td>CAT-Seg [34]*</td><td>ViT-L</td><td>0.2909 [0.226, 0.374] 0.4507 [0.368, 0.545]</td><td>0.6291 0.5832</td><td>0.3673</td><td>4.09 3.09</td><td>3.74 1.45</td><td>0.2287 [0.202, 0.255]</td><td>0.3722 [0.336, 0.407]</td><td>0.4101</td><td>0.3407 1.70</td><td>4.44</td><td>0.1300 [0.098, 0.160]</td><td>0.2300 [0.179, 0.276]</td><td>0.2050</td><td>0.2622</td><td>0.82</td><td></td></tr><tr><td>EviVLM [45]*</td><td>ViT-B</td><td>0.3290 [0.261, 0.385] 0.4952 [0.414, 0.556]</td><td>0.5289</td><td>0.4654</td><td>4.57</td><td>5.72</td><td>0.2584 [0.213, 0.311]</td><td></td><td>0.7417</td><td>0.2839</td><td>0.24</td><td>0.2487 [0.182, 0.312]</td><td>0.3984 [0.308, 0.476]</td><td>0.5134</td><td>0.3254</td><td>0.66</td><td>0.46 0.82</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.4106 [0.351, 0.474]</td><td></td><td>1.00</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VL-AcneSeg (Global)</td><td>ViT-L</td><td>0.3407 [0.282, 0.392] 0.5082 [0.440, 0.563]</td><td>0.4938</td><td>0.5238</td><td>3.69</td><td>2.43 0.3814 [0.345, 0.425]</td><td></td><td>0.5522 [0.513, 0.597]</td><td>0.5011 0.6149</td><td>1.83</td><td>0.94</td><td>0.2671 [0.213, 0.313]</td><td>0.4216 [0.352, 0.477]</td><td>0.3277</td><td>0.5908 1.23</td><td>0.93</td><td></td></tr><tr><td>SAM [55]</td><td>ViT-L</td><td>0.3216 [0.251, 0.409] 0.4867 [0.401, 0.581]</td><td>0.5601</td><td>0.4303</td><td>3.21</td><td>1.66 0.3764 [0.3306, 0.4284]</td><td></td><td>0.5469 [0.4969, 0.5999]</td><td>0.7064 0.4462</td><td>1.54</td><td>0.64</td><td>0.2546 [0.1568, 0.2919]</td><td>0.4058 [0.2710, 0.4519]</td><td>0.3452</td><td>0.4923 0.91</td><td></td><td>1.34</td></tr><tr><td>U-Net [56]</td><td>CNN</td><td>0.1418 [0.107, 0.167] 0.2483 [0.193, 0.286]</td><td>0.1490</td><td>0.7443</td><td>4.86</td><td>19.07 0.1771 [0.1549, 0.1980]</td><td></td><td>0.3009 [0.2682, 0.3305]</td><td>0.1857 0.7931</td><td>2.13</td><td>13.34</td><td>0.0863 [0.0620, 0.1148]</td><td>0.1590 [0.1167, 0.2060]</td><td>0.0884 0.7869</td><td>1.43</td><td>9.82</td><td></td></tr><tr><td>PSPNet [57]</td><td>CNN</td><td>0.1802 [0.134, 0.212] 0.3053 [0.236, 0.350]</td><td>0.2337</td><td>0.4402</td><td>2.33</td><td>1.21 0.3368 [0.3015, 0.3724]</td><td></td><td>0.5038 [0.4633, 0.5427]</td><td>0.4157 0.6394</td><td>1.79</td><td>1.43</td><td>0.1199 [0.0814, 0.1599]</td><td>0.2141 [0.1506, 0.2757]</td><td>0.1375 0.4826</td><td>0.91</td><td>1.62</td><td></td></tr><tr><td>DeepLabV3+ [58]</td><td>CNN</td><td>0.1807 [0.118, 0.229] 0.3061 [0.211, 0.373]</td><td>0.2592</td><td>0.3737</td><td>2.22</td><td>4.71 0.2603 [0.2058, 0.3185]</td><td></td><td>0.4131 [0.3413, 0.4832]</td><td>0.5224 0.3416</td><td>1.23</td><td>0.56</td><td>0.1681 [0.1018, 0.2344]</td><td>0.2878 [0.1848, 0.3798]</td><td>0.4078 0.2224</td><td>0.52</td><td>0.38</td><td></td></tr><tr><td>U-Net++ [59]</td><td>CNN</td><td>0.1575 [0.103, 0.221] 0.2721 [0.186, 0.362]</td><td>0.3802</td><td>0.2118</td><td>3.34</td><td>11.29 0.1986 [0.1627, 0.2342]</td><td></td><td>0.3315 [0.2798, 0.3796]</td><td>0.4420 0.2650</td><td>1.71</td><td>4.37</td><td>0.2517 [0.1817, 0.3309]</td><td>0.4022 [0.3076, 0.4973]</td><td>0.3455</td><td>0.4822 1.39</td><td></td><td>3.88</td></tr><tr><td>CE-Net [60]</td><td>CNN</td><td>0.2215 [0.161, 0.278] 0.3626 [0.277, 0.434]</td><td>0.2836</td><td>0.5028</td><td>3.34</td><td>4.31 0.1248 [0.0810, 0.1922]</td><td></td><td>0.2220 [0.1499, 0.3224]</td><td>0.1495 0.4308</td><td>1.81</td><td>8.06</td><td>0.1234 [0.0748, 0.1900]</td><td>0.2197 [0.1392, 0.3194]</td><td>0.1306</td><td>0.6915 1.57</td><td></td><td>7.93</td></tr><tr><td>SegFormer [61]</td><td>Transformer</td><td>0.2047 [0.146, 0.263] 0.3399 [0.254, 0.417]</td><td>0.2865</td><td>0.4177</td><td>3.90</td><td>10.83 0.1328 [0.1119, 0.1538]</td><td>0.2345 [0.2013, 0.2667]</td><td></td><td>0.1537 0.4974</td><td>1.79</td><td>17.21</td><td>0.0222 [0.0142, 0.0317]</td><td>0.0434 [0.0281, 0.0614]</td><td>0.0233</td><td>0.3072 0.88</td><td></td><td>16.38</td></tr><tr><td>TransUNet [62]</td><td>Transformer</td><td>0.2624 [0.189, 0.352] 0.4157 [0.317, 0.520]</td><td>0.5210</td><td>0.3459</td><td>3.09</td><td>2.36 0.1360 [0.0937, 0.1845] 7.53</td><td></td><td>0.2394 [0.1713, 0.3115]</td><td>0.3155 0.1928</td><td>0.93</td><td>0.60</td><td>0.0328 [0.0097, 0.0794]</td><td>0.0634 [0.0191, 0.1471]</td><td>0.0426 0.1245</td><td>0.23</td><td></td><td>1.57</td></tr><tr><td>Swin-UNet [63] nnU-Net [64]</td><td>Transformer</td><td>0.2089 [0.150, 0.264] 0.3456 [0.262, 0.418]</td><td>0.3266</td><td>0.3670 0.5234</td><td>3.12 4.09</td><td>0.2005 [0.1563, 0.2476] 17.97</td><td>0.3341 [0.2703, 0.3969]</td><td></td><td>0.4446 0.2676</td><td>1.19</td><td>3.79</td><td>0.1454 [0.0737, 0.2346]</td><td>0.2539 [0.1373, 0.3800]</td><td>0.2390 0.2709 0.0575</td><td>0.73</td><td></td><td>3.84</td></tr><tr><td>Swin-UMamba† [65]</td><td>CNN Mamba</td><td>0.1909 [0.142, 0.242] 0.3206 [0.248, 0.390] 0.2877 [0.207, 0.355] 0.4468 [0.343, 0.524]</td><td>0.2311 0.3329</td><td>0.6793</td><td>4.98</td><td>0.2079 [0.177, 0.240] 0.3278 [0.295, 0.364]</td><td>0.3442 [0.301, 0.387] 0.4937 [0.456, 0.534]</td><td></td><td>0.2927 0.4177 0.4933 0.4941</td><td>1.53 1.69</td><td>3.50 2.13</td><td>0.0454 [0.017, 0.084] 0.2552 [0.188, 0.326]</td><td>0.0869 [0.033, 0.155] 0.4066 [0.316, 0.492]</td><td>0.1779 0.3566 0.4729</td><td>0.39 0.75</td><td></td><td>4.62 11.68</td></tr></table>

![](images/028df868eb5e3924a723b4bcc437ebc73dad9d2aed93edf6005992d11808b825.jpg)  
Fig. 4: Qualitative comparison across the three datasets: (A) internal, (B) external controlled, and (C) external smartphone images. The left column shows the reference annotation on the full face; the green box marks the region enlarged in the remaining columns, so that lesions a few pixels across remain visible at print size. Predicted masks are overlaid in red, and both prompting conditions of the proposed method are shown. A comparison including all baselines and both prompting conditions is given in Figs. S3–S5 of the Supplementary Material.

On external datasets, VL-AcneSeg maintains consistent performance across both controlled and real-world settings. Most baseline models show notable degradation, with Transformerbased architectures being particularly vulnerable to distribution shifts compared to CNN-based models, likely due to their weaker inductive biases [66]. The gap becomes more pronounced in the real-world smartphone setting, where CAT-Seg and SED degrade substantially and fall below several unimodal baselines, indicating that general-purpose VLM architectures do not readily transfer to domain-shifted clinical environments. VL-AcneSeg, by contrast, retains stable segmentation under unpredictable lighting and varying facial angles.

## E. Ablation Study

We evaluated the contributions of VL-AcneSeg’s modules and hyperparameters through three studies: (1) architectural components, (2) image patch resolutions, and (3) text prompt strategies. Results on the external datasets are included in Table III, and the corresponding patch-resolution comparison is given in Table S3 of the Supplementary Material.

TABLE III  
ABLATION STUDY ON THE MODEL COMPONENTS OF VL-ACNESEG. CONFIDENCE INTERVALS AND p VALUES ARE OBTAINED BY PAIRED CLUSTER BOOTSTRAP AGAINST THE FULL MODEL, AT THE PATIENT LEVEL FOR THE INTERNAL DATASET AND AT THE IMAGE LEVEL FOR THE EXTERNAL DATASETS. SPECIFICITY EXCEEDS 0.999 EVERYWHERE AND IS OMITTED. FP LESIONS/IMG COUNTS PREDICTED LESION COMPONENTS MATCHING NO REFERENCE LESION. BEST VALUES ARE IN BOLD AND SECOND-BEST ARE UNDERLINED.
<table><tr><td rowspan="2">Metrics</td><td rowspan="2">Full Model</td><td>w/o cls</td><td>w/o v-v</td><td>w/o region</td><td>all-layer</td></tr><tr><td>token</td><td>attention</td><td>mask loss</td><td>V-V</td></tr><tr><td>Internal</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>IoU</td><td>0.3602 [0.309, 0.399]</td><td>0.3563 [0.294, 0.398]</td><td>0.3420 [0.290, 0.380]</td><td>0.3510 [0.290, 0.394]</td><td>0.2832 [0.215, 0.336]</td></tr><tr><td>Dice</td><td>0.5296 [0.472, 0.570]</td><td>0.5254 [0.458, 0.568]</td><td>0.5097 [0.451, 0.552]</td><td>0.5196 [0.454, 0.565]</td><td>0.4414 [0.357, 0.505]</td></tr><tr><td>Precision</td><td>0.5202</td><td>0.5149</td><td>0.4936</td><td>0.4935</td><td>0.3700</td></tr><tr><td>Recall</td><td>0.5393</td><td>0.5363</td><td>0.5268</td><td>0.5486</td><td>0.5449</td></tr><tr><td>p-value</td><td></td><td>0.554</td><td>0.057</td><td>0.526</td><td>&lt;0.001</td></tr><tr><td>TP lesions/img FP lesions/img</td><td>4.29</td><td>3.49</td><td>3.52</td><td>3.55</td><td>4.38</td></tr><tr><td>External (Controlled)</td><td>2.48</td><td>1.90</td><td>2.27</td><td>2.09</td><td>3.57</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>IoU</td><td>0.4232 [0.376, 0.470]</td><td>0.4064 [0.364, 0.452]</td><td>0.4020 [0.359, 0.449]</td><td>0.4097 [0.366, 0.455]</td><td>0.3314 [0.294, 0.374]</td></tr><tr><td>Dice</td><td>0.5948 [0.546, 0.640]</td><td>0.5780 [0.534, 0.622]</td><td>0.5735 [0.528, 0.620]</td><td>0.5813 [0.536, 0.625]</td><td>0.4979 [0.454, 0.544]</td></tr><tr><td>Precision</td><td>0.6110</td><td>0.6107</td><td>0.5879</td><td>0.5947</td><td>0.4588</td></tr><tr><td>Recall</td><td>0.5794</td><td>0.5486</td><td>0.5597</td><td>0.5684</td><td>0.5442</td></tr><tr><td>p-value</td><td></td><td>0.119</td><td>0.114</td><td>0.332</td><td>&lt;0.001</td></tr><tr><td>TP lesions/img FP lesions/img</td><td>1.64</td><td>1.57</td><td>1.27</td><td>1.07</td><td>1.68</td></tr><tr><td>External (Real-world)</td><td>0.60</td><td>0.47</td><td>0.67</td><td>0.45</td><td>0.78</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>IoU</td><td>0.3044 [0.246, 0.351]</td><td>0.2873 [0.190, 0.365]</td><td>0.2819 [0.218, 0.343]</td><td>0.2905 [0.216, 0.352]</td><td>0.1627 [0.114, 0.217]</td></tr><tr><td>Dice</td><td>0.4667 [0.395, 0.520]</td><td>0.4464 [0.320, 0.535]</td><td>0.4398 [0.358, 0.511]</td><td>0.4502 [0.355, 0.520]</td><td>0.2798 [0.204, 0.356]</td></tr><tr><td>Precision</td><td>0.4882</td><td>0.4571</td><td>0.3785 0.5248</td><td>0.4269</td><td>0.1965</td></tr><tr><td>Recall</td><td>0.4471</td><td>0.4361 0.573</td><td>0.346</td><td>0.4763</td><td>0.4860</td></tr><tr><td>p-value TP lesions/img</td><td>1.50</td><td>1.34</td><td>0.95</td><td>0.520 0.83</td><td>&lt;0.001</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>1.55</td></tr><tr><td>FP lesions/img</td><td>0.82</td><td>0.77</td><td>0.56</td><td>0.48</td><td>0.93</td></tr></table>

TABLE IV

ABLATION STUDY ON DIFFERENT IMAGE PATCH RESOLUTIONS FOR CLIP-BASED PROCESSING IN VL-ACNESEG. BEST VALUES ARE IN BOLD AND SECOND-BEST ARE UNDERLINED.
<table><tr><td>Metrics</td><td>4 × 3 patches</td><td>3 × 2 patches</td><td>2 × 1 patches</td></tr><tr><td>IoU</td><td>0.3602</td><td>0.3194</td><td>0.2700</td></tr><tr><td>Dice</td><td>0.5296</td><td>0.4841</td><td>0.4253</td></tr><tr><td>Precision</td><td>0.5202</td><td>0.4308</td><td>0.4070</td></tr><tr><td>Recall</td><td>0.5393</td><td>0.5524</td><td>0.4452</td></tr><tr><td>TP lesions/img</td><td>4.29</td><td>3.67</td><td>3.14</td></tr><tr><td>FP lesions/img</td><td>2.48</td><td>2.72</td><td>1.71</td></tr></table>

1) Effectiveness of Architectural Components: As shown in Table III, the Full Model attains the highest IoU and Dice on all three datasets. Removing any single component lowers performance consistently but by a margin that the present test sets do not resolve individually, indicating that the CLS token, V-V attention and the region mask loss act in a complementary manner rather than through one dominant factor. The placement of V-V attention, by contrast, is clearly resolved: applying it across all layers, as in the original formulation, degrades performance on every dataset $( p < 0 . 0 1 )$ and raises the number of false positive lesions per image from 2.48 to 3.57 internally and from 0.82 to 0.93 in the real-world setting, consistent with the view that repeated value–value mixing erases the fine spatial detail on which small lesions depend.

2) Impact of Image Patch Resolution: We tested tiling configurations of 4 × 3, 3 × 2, and 2 × 1 patches. As summarized in Table IV, while the 3 × 2 configuration yields the highest Recall, it suffers from low Precision due to its inability to distinguish acne from scars. In contrast, 4 × 3 patches provide the most balanced results. Localization maps for the three configurations are shown in Fig. S1 of the Supplementary

Material, where finer granularity yields the detailed spatial priors needed to discriminate true lesions from confounding skin features. Furthermore, qualitative evidence of precise localization in unconstrained smartphone environments is provided in Fig. S2 of the Supplementary Material.

3) Role of the Text Prompt: To test whether the spatial prior rests on the semantics of the anatomical terms or on memorised region identifiers, we prepared two clinical synonyms for each of the fifteen regions and at inference replaced every prompt by one of the two at random, without retraining or adapting the model (Table S1 of the Supplementary Material). Accuracy is largely preserved: internal IoU moves from 0.3602 to 0.3422, with precision rising slightly and recall falling, a shift towards more conservative predictions rather than a loss of spatial grounding. Randomly pairing each image with the prompts belonging to another image is by contrast costly. On the controlled set IoU falls from 0.4232 to 0.2170, close to the level obtained without CLIP at all; internally the drop is more moderate, from 0.3602 to 0.3039. The prompt is therefore followed as a reference to a particular area rather than treated as an undifferentiated conditioning signal, although how much the correct reference is worth varies between the two datasets. Alternative prompt contents, adding a severity level or a lesion count to the region name, are compared in Table S2.

Read from the bottom, Table V separates where the performance originates. Removing CLIP altogether leaves internal IoU at 0.1694 and real-world IoU at 0.0180, precision collapsing while recall is preserved: without the pretrained visual representation the model no longer distinguishes active lesions from post-inflammatory hyperpigmentation, pores and scars. Restoring the CLIP vision encoder alone recovers 0.2358, and adding region conditioning a further 0.3602, so both contribute

## TABLE V

WHERE THE PERFORMANCE ORIGINATES. EACH ROW REMOVES ONE ELEMENT OF THE VISION–LANGUAGE PIPELINE. Unseen synonyms AND Shuffled text prompt KEEP THE CLIP TEXT ENCODER BUT SUBSTITUTE SYNONYMS ABSENT FROM TRAINING OR PAIR EACH IMAGE WITH THE PROMPTS OF ANOTHER IMAGE, TESTING WHETHER THE PROMPT IS READ AS A REFERENCE TO A PARTICULAR AREA. Learnable embedding AND

One-hot region encoding REPLACE THE TEXT EMBEDDING WITH A NON-LINGUISTIC REGION CODE, SEPARATING LANGUAGE FROM SPATIAL CONDITIONING. CLIP vision encoder only DISCARDS THE TEXT BRANCH ALTOGETHER, AND w/o CLIP REMOVES THE CLIP ENCODERS AS WELL,

SO THAT THE INTERVAL BETWEEN THE TWO ISOLATES THE CONTRIBUTION OF THE PRETRAINED VISUAL REPRESENTATION. THE LAST FOUR SETTINGS WERE RETRAINED.
<table><tr><td>Prompt representation</td><td>IoU</td><td>Dice</td><td>Precision</td><td>Recall</td></tr><tr><td>Internal</td></tr><tr><td>Regional text prompt 0.3602 [0.309, 0.399]</td><td></td><td>0.5296 [0.472, 0.570]</td><td>0.5202</td><td>0.5393</td></tr><tr><td>Unseen synonyms</td><td>0.3422 [0.291, 0.392] 0.3039 [0.251, 0.356]</td><td>0.5100 [0.451, 0.563]</td><td>0.5272</td><td>0.4938</td></tr><tr><td>Shuffled text prompt Learnable embedding</td><td>0.3565 [0.301, 0.399]</td><td>0.4661 [0.402, 0.525] 0.5256 [0.463, 0.570]</td><td>0.5425 0.4654</td><td>0.4085 0.6037</td></tr><tr><td>One-hot region encoding</td><td>0.3111 [0.260, 0.357]</td><td>0.4745 [0.413, 0.526]</td><td>0.4313</td><td>0.5274</td></tr><tr><td>CLIP vision encoder only</td><td>0.2358 [0.184, 0.270]</td><td></td><td>0.3642</td><td>0.4007</td></tr><tr><td>w/o CLIP (Swin only)</td><td>0.1694 [0.117, 0.219]</td><td>0.3816 [0.311, 0.426] 0.2897 [0.210, 0.359]</td><td>0.2047</td><td>0.4957</td></tr><tr><td>External (Controlled)</td></tr><tr><td>Regional text prompt</td><td>0.4232 [0.376, 0.470]</td><td>0.5948 [0.546, 0.640]</td><td>0.6110</td><td>0.5794</td></tr><tr><td>Unseen synonyms</td><td>0.3724 [0.337, 0.414]</td><td>0.5427 [0.504, 0.585]</td><td>0.6476</td><td>0.4671</td></tr><tr><td>Shuffled text prompt</td><td>0.2170 [0.171, 0.267] 0.4292 [0.393, 0.474]</td><td>0.3567 [0.292, 0.421]</td><td>0.5597</td><td>0.2617</td></tr><tr><td>Learnable embedding One-hot region encoding</td><td>0.3732 [0.335, 0.417]</td><td>0.6006 [0.565, 0.643]</td><td>0.6246</td><td>0.5784</td></tr><tr><td>CLIP vision encoder only</td><td>0.2354 [0.205, 0.272]</td><td>0.5435 [0.502, 0.588]</td><td>0.4885</td><td>0.6125</td></tr><tr><td>w/o CLIP (Swin only)</td><td>0.2150 [0.181, 0.251]</td><td>0.3811 [0.340, 0.427]</td><td>0.3638</td><td>0.4002</td></tr><tr><td></td><td></td><td>0.3539 [0.307, 0.401]</td><td>0.2725</td><td>0.5049</td></tr><tr><td>External (Real-world)</td><td>0.3044 [0.248, 0.348]</td><td></td><td></td><td></td></tr><tr><td>Regional text prompt</td></tr><tr><td>Unseen synonyms</td><td>0.2708 [0.199, 0.334]</td><td>0.4667 [0.397, 0.517]</td><td>0.4882</td><td>0.4471</td></tr><tr><td>Shuffled text prompt</td><td>0.2435 [0.175, 0.290]</td><td>0.4262 [0.332, 0.500] 0.3916 [0.298, 0.449]</td><td>0.4509 0.4685</td><td>0.4041</td></tr><tr><td>Learnable embedding</td><td>0.2900 [0.227, 0.347]</td><td>0.4496 [0.369, 0.515]</td><td>0.5289</td><td>0.3364</td></tr><tr><td>One-hot region encoding</td><td>0.2734 [0.212, 0.329]</td><td>0.4294 [0.350, 0.495]</td><td>0.3864</td><td>0.3910</td></tr><tr><td></td><td>0.1470 [0.107, 0.188]</td><td></td><td></td><td>0.4831</td></tr><tr><td>CLIP vision encoder only w/o CLIP (Swin only)</td><td>0.0180 [0.008, 0.035]</td><td>0.2563 [0.194, 0.316] 0.0353 [0.016, 0.067]</td><td>0.2447 0.0186</td><td>0.2692 0.3475</td></tr></table>

substantially, while the visual representation is what sustains performance under domain shift. Among the conditioning schemes, a fixed one-hot index reaches only 0.3111 whereas a learnable per-region vector matches the text prompts (0.3565 against 0.3602): what matters is not that a region is named but that its code can be aligned with the patch features, which an orthogonal index cannot be. Either source yields one vector per region that enters the same cosine-similarity localization, so with a fixed set of fifteen areas the two are mechanistically equivalent and differ only in that the text encoder also resolves terms outside that vocabulary—a distinction that becomes practical only where the vocabulary is open or prompts arrive as free-form clinical language.

Fig. 5 shows how prompts steer spatial attention. For distinct areas such as the forehead (b) and cheek (c) the response is regionally dominant, whereas in the narrow temple region (d) the model falls back on the broader concept of an acne lesion. The bottom row probes terms absent from training: singleword clinical synonyms transfer, with “frontal region” (e) concentrating over the forehead and “malar area” (f) isolating the mid-face while suppressing the forehead response seen in (b) and (e). The compositional expression “lateral forehead” (g) instead yields a diffuse map, indicating that a spatial modifier is not fully composed with the anatomical noun.

We further quantified the response to imperfect prompts (Fig. 6), withholding prompts for regions that do contain lesions and issuing prompts for regions that do not. The two behave asymmetrically. Withholding degrades accuracy steadily, from 0.3602 to 0.3100 IoU at 40% perturbation, and falls below the global-prompt baseline of 0.3407 at roughly 15%. Over-prompting is far better tolerated, declining only to 0.3438 at the same level and remaining above that baseline throughout the range examined. The asymmetry itself is informative: that omission costs more than over-prompting means the model attends less to regions it has not been told about, which is the behaviour the mechanism is intended to produce. Where the shuffled-prompt condition shows that the model follows the region it is given, the omission curve shows that it looks less closely at regions it is not given. The two error types are not symmetric in the pipeline either: a spurious region enters the localization stage as a candidate and can still be rejected during refinement, whereas a region that never enters it cannot be recovered downstream. Two adversarial cases in Fig. 7 illustrate both sides: lesions are still localized where prompts are withheld, since visual evidence remains, and under misleading prompts candidates appear only where textual guidance and visual evidence coincide. In practical terms a prompt source must therefore achieve high recall over lesion-bearing regions, while spurious regions carry comparatively little cost.

![](images/a3331f7aff3e2c0cf0424942023a139fb1e3a5f6759f9caa522721e49e5b1334.jpg)  
Fig. 5: Anatomical region discrimination through textguided spatial priors. Top row: region names used during training. Bottom row: anatomical terms never seen during training, paired column-wise with the region above.

![](images/0bcdfcdd3f8877c12c084ae2817e7d316a1116832f6418ecdeee99f40194e259.jpg)  
Fig. 6: Segmentation accuracy under prompt perturbation on the internal test set. The 58 test images carry 165 lesion-bearing (positive) region prompts and 705 lesion-free (negative) regions. Each pool was shuffled once with a fixed seed, and the perturbation level denotes the fraction of that pool that is withheld (FN) or added (FP); the perturbed sets are nested. When every prompt of an image is withheld, it falls back to the single global prompt, marked by the dashed line.

Original Result  
Adversarial Result  
![](images/a269c5c60f4d9c06744ecf76d494ec406aa3dc7251cc9db57963148cd72b7be7.jpg)  
Fig. 7: Model response under adversarial prompting. Prompts for lesion-bearing regions are omitted and misleading prompts are issued for lesion-free regions.

## F. Failure Case Analysis

The residual errors follow a consistent pattern (Fig. 8). In (A) the cheek and chin carry active lesions, post-inflammatory hyperpigmentation and faded scars together, and the model both marks the residual features as lesions (blue boxes) and misses small papules embedded in densely pigmented skin (orange boxes). In (B) small inflammatory lesions are interspersed with widespread red marks, and the same two errors recur: subtle active lesions are overlooked and isolated non-acne marks elsewhere on the face are labelled as acne. Both modes arise from the same ambiguity, since active lesions and their residue share colour, shape and texture in clinical photographs. Confusions of this kind, rather than gross localization failures, account for most of the remaining error.

![](images/bf1bb942878a10b622801ec53148deca661582d0603584040e99785b8ccecff6.jpg)  
Fig. 8: Failure case analysis of VL-AcneSeg. Red regions denote acne masks. Blue boxes mark false positives, predominantly PIH, scars and prominent pores; orange boxes mark false negatives, typically small or low-contrast lesions.

TABLE VI  
CORRELATION BETWEEN THE PREDICTED LESION-TO-FACE AREA RATIO AND IGA SEVERITY ON THE INTERNAL TEST SET (22 VISITS FROM 16 PATIENTS). p VALUES COMPARE EACH COEFFICIENT WITH THAT OF THE EXPERT ANNOTATIONS USING A TEST FOR DEPENDENT CORRELATIONS SHARING ONE VARIABLE; CONFIDENCE INTERVALS ARE OBTAINED BY PATIENT-LEVEL CLUSTER BOOTSTRAP.
<table><tr><td rowspan="2">Source of area ratio</td><td colspan="3">Pearson</td><td colspan="3">Spearman</td></tr><tr><td></td><td>r [95% CI]</td><td>p</td><td></td><td>ρ [95% CI]</td><td>p</td></tr><tr><td>Expert annotation (GT)</td><td>0.658</td><td>[0.441, 0.807]</td><td></td><td></td><td>0.710 [0.541, 0.793]</td><td></td></tr><tr><td>VL-AcneSeg (region)</td><td>0.719</td><td>[0.477, 0.848]</td><td>0.430</td><td>0.684</td><td>[0.485, 0.799]</td><td>0.719</td></tr><tr><td>VL-AcneSeg (global)</td><td></td><td>0.695 [0.442, 0.833]</td><td>0.621</td><td></td><td>0.695 [0.465, 0.843]</td><td>0.868</td></tr><tr><td>EviVLM</td><td></td><td>0.640 [0.361, 0.796]</td><td>0.839</td><td>0.660</td><td>[0.392, 0.809]</td><td>0.598</td></tr><tr><td>CAT-Seg</td><td></td><td>0.650 [0.407, 0.797]</td><td>0.924</td><td>0.685</td><td>[0.479, 0.812]</td><td>0.737</td></tr><tr><td>SED</td><td></td><td>0.630 [0.279, 0.808]</td><td>0.770</td><td>0.645</td><td>[0.372, 0.827]</td><td>0.526</td></tr><tr><td>SAM</td><td></td><td>0.650 [0.320, 0.820]</td><td>0.932</td><td>0.572</td><td>[0.299, 0.759]</td><td>0.187</td></tr><tr><td>nnU-Net</td><td></td><td>0.579 [0.179, 0.793]</td><td>0.435</td><td></td><td>0.530 [0.203, 0.776]</td><td>0.125</td></tr><tr><td>Swin-UMamba†</td><td></td><td>0.587 [0.155, 0.801]</td><td>0.501</td><td></td><td>0.510 [0.151, 0.766]</td><td>0.116</td></tr><tr><td>PSPNet</td><td></td><td>0.690 [0.373, 0.863]</td><td>0.738</td><td>0.571</td><td>[0.323, 0.755]</td><td>0.218</td></tr><tr><td>U-Net</td><td></td><td>0.630 [0.336, 0.795]</td><td>0.722</td><td>0.628</td><td>[0.376, 0.809]</td><td>0.372</td></tr><tr><td>U-Net++</td><td></td><td>0.630 [0.291, 0.844]</td><td>0.798</td><td>0.549</td><td>[0.200, 0.790]</td><td>0.254</td></tr><tr><td>TransUNet</td><td></td><td>0.610 [0.293, 0.782]</td><td>0.623</td><td></td><td>0.610 [0.383, 0.771]</td><td>0.244</td></tr><tr><td>DeepLabV3+</td><td></td><td>0.600 [0.221, 0.808]</td><td>0.589</td><td></td><td>0.496 [0.189, 0.728]</td><td>0.085</td></tr><tr><td>CE-Net</td><td></td><td>0.590 [0.229, 0.779]</td><td>0.512</td><td>0.553</td><td>[0.240, 0.778]</td><td>0.225</td></tr><tr><td>SegFormer</td><td></td><td>0.540 [0.182, 0.753]</td><td>0.286</td><td>0.527</td><td>[0.179, 0.780]</td><td>0.219</td></tr><tr><td>Swin-UNet</td><td></td><td>0.400 [0.003, 0.617]</td><td>0.057</td><td></td><td>0.462 [0.110, 0.722]</td><td>0.102</td></tr></table>

## G. Clinical Implementation

To assess clinical relevance beyond pixel-level metrics, we correlated the lesion-to-face area ratio of each model with Investigator’s Global Assessment (IGA) severity on the internal test set, pooling the views of a visit so that each of the 22 visits contributes one observation (Table VI). The area ratio derived from VL-AcneSeg tracks IGA at the level of the expert annotations themselves (Pearson 0.719 versus 0.658; Spearman 0.684 versus 0.710); the direction of this comparison reverses between the two coefficients and neither difference is significant (p = 0.430 and 0.719), so the predicted and reference ratios cannot be distinguished in their agreement with IGA on this sample.

Comparing methods, no coefficient differs significantly from that of the reference annotations at this sample size. On Spearman, however, the six highest values belong to the reference annotations and to the five vision-language configurations, with the convolutional and Transformer architectures below them even where their pixel-level accuracy is comparable. Within this group the global prompt, our primary setting, correlates as well as the region-level one, as expected for a measure defined over the whole face; region-level prompts instead yield a separate estimate for each facial area. A high coefficient alone, however, does not imply an accurate area estimate: U-Net reaches a moderate value because its inflated areas still scale with lesion burden, whereas VL-AcneSeg reaches a comparable coefficient with precision and recall balanced, so its estimates track severity without a systematic offset. These results support the premise that lesion area measured from a segmentation can serve as a quantitative severity measure.

This is also where an area-based measure differs from counting. A count treats every lesion as one unit irrespective of extent, so it saturates precisely where severity is greatest: confluent papules that have merged into a plaque contribute the same value as a few discrete ones, and a lesion that shrinks under treatment without resolving contributes the same value throughout. An area ratio varies continuously with both, which is the property area-based assessment relies on.

A Dice score near 0.53 is modest by the standards of organ segmentation, but the quantity entering the severity measure is the aggregate lesion area rather than the boundary of each lesion, and a Dice score at this level is sufficient for that area to track IGA as closely as the reference annotations do. The difficulty of the task is also reflected in human performance: in a reader study on this cohort, eight dermatologists and residents reached a median lesion-detection F1 of 0.31 for inflammatory lesions from images alone [1]. Applications requiring precise per-lesion delineation would need higher accuracy than we report here.

## H. Deployment Considerations

The framework runs in two settings. Under the global prompt no region information is required, which is why we report it as the primary setting. Under region-level prompting a clinician or the patient indicates which facial areas carry lesions—a coarse judgement, not the location of individual lesions—which raises accuracy and additionally gives a separate estimate for each area. The perturbation analysis indicates how accurate such prompts need to be. Withholding prompts for lesion-bearing regions falls below the globalprompt baseline at roughly 15% omission, whereas issuing prompts for lesion-free regions stays above it throughout the range examined. In practice, therefore, over-inclusive prompting is safe, whereas leaving affected regions unmarked is what degrades the result.

VL-AcneSeg is slower than most baselines (Table VII), which follows from the tiled high-resolution encoding on which the accuracy also depends. In absolute terms, however, one image takes 858 ms under the global prompt and at most 1.54 s with all fifteen regions, so cost is not a constraint at either setting. The tiled encoding accounts for roughly 83% of the computation but is shared across prompts, so each additional region prompt adds only 18.5 GFLOPs, or about 48 ms.

TABLE VII  
COMPUTATIONAL COST, MEASURED ON A SINGLE NVIDIA A6000 AT THE INPUT RESOLUTION USED THROUGHOUT. FOR VL-ACNESEG, all regions DENOTES PROMPTS ISSUED FOR ALL FIFTEEN FACIAL REGIONS; THE TILED CLIP ENCODING IS SHARED ACROSS PROMPTS.
<table><tr><td></td><td>Params (M)</td><td>FLOPs (G)</td><td>Time (ms)</td></tr><tr><td>VL-AcneSeg (global prompt)</td><td>432</td><td>1,417</td><td>858</td></tr><tr><td>VL-AcneSeg (all regions)</td><td>432</td><td>1,676</td><td>1,536</td></tr><tr><td>SAM (ViT-L)</td><td>312</td><td>1,312</td><td>594</td></tr><tr><td>EviVLM</td><td>130</td><td>567</td><td>176</td></tr><tr><td>Swin-UMamba†</td><td>27</td><td>25</td><td>64</td></tr><tr><td>U-Net</td><td>17</td><td>428</td><td>74</td></tr></table>

## I. Limitations and Future Directions

Several limitations should be acknowledged. First, regionlevel prompting presupposes knowledge of which facial areas contain lesions; we therefore report the global prompt, which requires none, as our primary setting. With a closed set of fifteen regions the pretrained text prior is also not fully exploited: unseen synonyms are resolved (Table V), but that capacity makes no difference to a deployment in which the vocabulary is fixed in advance.

Second, all internal and external data represent Korean populations, so the evidence does not extend to other ethnicities, skin tones or imaging devices, and claims of worldwide applicability would be unwarranted. This constraint is not particular to our work: the two public acne datasets are likewise drawn from a single ethnic group, and no published acne algorithm has yet undergone prospective clinical validation [18]. Fitzpatrick phototype was not recorded in the clinical dataset and is not distributed with the external collection, so performance could not be stratified by skin type; nor could it be stratified by severity, since the 22 graded visits leave strata too small for stable estimates. The same sample size widens the intervals around the smaller between-method differences, so their magnitude is estimated less precisely than their direction. Constructing pixel-level acne datasets requires substantial dermatological effort, and evaluation on multiethnic, multi-site data remains the most important direction for establishing broader applicability.

Third, Dice and IoU remain moderate relative to other medical segmentation tasks, and the residual error is dominated by the ambiguity between active lesions and their own residue described in Section IV-F. Fourth, the correlation with IGA does not by itself show that the method is ready for clinical use; establishing that would require a prospective study against expert assessment.

Finally, inter-annotator agreement was not quantified. The annotation followed an adjudication design in which a single board-certified dermatologist fixed the final label set, so agreement between readers is not the quantity that defines the reference standard here; the underlying lesion markings were moreover produced for an earlier study [1], and the drafts preceding adjudication were not retained as separate outputs. Quantifying observer variability on this task would help establish the practical ceiling for pixel-level acne segmentation and remains an objective for future work.

Several of these limitations suggest concrete next steps. The confusion is between an active lesion and its own residue, which are hard to separate in the spatial domain because they share colour, shape and texture; features that expose properties the RGB image does not make explicit, such as a frequency-domain representation, may provide evidence that spatial appearance alone does not. Our annotations distinguish five lesion types, and we plan to extend the model to a multiclass objective, which would also allow papules, pustules and nodules to be segmented jointly. Prompts that vary in what they describe—lesion type, extent or a clinician’s own phrasing—would put the text encoder to work, and are a natural direction once multi-class supervision is in place. More broadly, the reformulation itself is not specific to acne: wherever lesions are small and numerous but distributed over an anatomy that is stable and nameable, text can supply the same kind of spatial constraint.

## V. CONCLUSION

We introduced VL-AcneSeg, a multimodal framework for inflammatory acne lesion segmentation. Where openvocabulary methods use text to specify which class to segment, we use it to specify where the target may appear, so that the text channel supplies a spatial rather than a categorical constraint—a formulation suited to a task with a single lesion class distributed over a nameable anatomy. Because regionlevel prompts presuppose knowledge of lesion locations, we report a single global prompt as the primary setting and treat region-level prompting as an oracle upper bound.

Across internal and external datasets VL-AcneSeg attains the highest Dice and IoU among the compared methods, including vision-language baselines that are themselves given region-level prompts, and retains this under domain shift where general-purpose architectures transfer less well. Controlled substitutions indicate that the prompt is read as a reference to a particular area rather than as an undifferentiated signal, although on a closed region vocabulary a learned region vector is mechanistically equivalent to a text embedding. The lesion areas the model produces track clinical IGA severity as closely as the expert annotations do, supporting area-based severity assessment as an alternative to lesion counting. We believe VL-AcneSeg provides a practical foundation for automated acne management.

## CONFLICT OF INTERESTS

The authors declare that they have no conflict of interest.

[1] D. H. Kim, S. Sun, S. I. Cho, H.-J. Kong, J. W. Lee, J. H. Lee, and D. H. Suh, “Automated facial acne lesion detecting and counting algorithm for acne severity evaluation and its utility in assisting dermatologists,” American Journal of Clinical Dermatology, vol. 24, no. 4, pp. 649–659, 2023.

[2] A. Grada, S. Muddasani, A. B. Fleischer Jr, S. R. Feldman, and G. M. Peck, “Trends in office visits for the five most common skin diseases in the united states,” The Journal of Clinical and Aesthetic Dermatology, vol. 15, no. 5, p. E82, 2022.

[3] J. K. Tan and K. Bhate, “A global perspective on the epidemiology of acne,” British Journal of Dermatology, vol. 172, no. S1, pp. 3–12, 2015.

[4] M. Law, A. Chuh, A. Lee, and N. Molinari, “Acne prevalence and beyond: acne disability and its predictive factors among chinese late adolescents in hong kong,” Clinical and experimental dermatology, vol. 35, no. 1, pp. 16–21, 2010.

[5] K. Bhate and H. Williams, “Epidemiology of acne vulgaris,” British Journal of Dermatology, vol. 168, no. 3, pp. 474–485, 2013.

[6] S. Y. Park, M. Y. Park, D. H. Suh, H. H. Kwon, S. Min, S. J. Lee, W. J. Lee, M. W. Lee, H. H. Ahn, H. Kang et al., “Cross-sectional survey of awareness and behavioral pattern regarding acne and acne scar based on smartphone application,” International Journal of Dermatology, vol. 55, no. 6, pp. 645–652, 2016.

[7] S. I. Cho, J. H. Yang, and D. H. Suh, “Analysis of trends and status of physician-based evaluation methods in acne vulgaris from 2000 to 2019,” The Journal of Dermatology, vol. 48, no. 1, pp. 42–48, 2021.

[8] C. Beylot, M. Chivot, M. Faure, H. Pawin, F. Poli, J. Revuz, N. Auffret, D. Moyse, B. Dreno, and G. E. A. (GEA), “Inter-observer agreement´ on acne severity based on facial photographs,” Journal of the European Academy of Dermatology and Venereology, vol. 24, no. 2, pp. 196–198, 2010.

[9] T. Agnew, G. Furber, M. Leach, and L. Segal, “A comprehensive critique and review of published measures of acne severity,” The Journal of clinical and aesthetic dermatology, vol. 9, no. 7, p. 40, 2016.

[10] I. H. Bae, J. H. Kwak, C. H. Na, M. S. Kim, B. S. Shin, and H. Choi, “A comprehensive review of the acne grading scale in 2023,” Annals of Dermatology, vol. 36, no. 2, p. 65, 2024.

[11] J. K. Tan, K. Fung, and L. Bulger, “Reliability of dermatologists in acne lesion counts and global assessments,” Journal of cutaneous medicine and surgery, vol. 10, no. 4, pp. 160–165, 2006.

[12] Q. T. Huynh, P. H. Nguyen, H. X. Le, L. T. Ngo, N.-T. Trinh, M. T.-T. Tran, H. T. Nguyen, N. T. Vu, A. T. Nguyen, K. Suda et al., “Automatic acne object detection and acne severity grading using smartphone images and artificial intelligence,” Diagnostics, vol. 12, no. 8, p. 1879, 2022.

[13] J. Wang, C. Wang, Z. Wang, A. H. Hounye, Z. Li, M. Kong, M. Hou, J. Zhang, and M. Qi, “A novel automatic acne detection and severity quantification scheme using deep learning,” Biomedical Signal Processing and Control, vol. 84, p. 104803, 2023.

[14] D. Thiboutot, A. Longenecker, D. Canfield, and S. V. Patwardhan, “Parametric acne severity (pas) score and lesion counts from multimodality facial image analysis correlates strongly with investigator assessment.” Journal of drugs in dermatology: JDD, vol. 20, no. 6, pp. 642–647, 2021.

[15] M. H. Gold, A. Bhatia, A. Kaur, M. Doucette, and A. Kothare, “Picturebased acne lesion counts: A validation study to assess accuracy and reliability of acne lesion counts via photography,” Journal of Cosmetic Dermatology, vol. 21, no. 12, pp. 6965–6975, 2022.

[16] L. Gazeau, H. Nguyen, Z. Nguyen, M. Lebedeva, T. Nguyen, T.-D. To, J. Le Digabel, J. Filiol, G. Josse, C. Perlis et al., “Acneai: A new acne severity assessment method using digital images and deep learning,” in International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, 2024, pp. 68–78.

[17] M. Moncho-Santonja, S. Aparisi-Navarro, B. Defez, and G. Peris-Fajarnes, “Segmentation of acne vulgaris images techniques: a compar-´ ative and technical study,” Applied Sciences, vol. 13, no. 10, p. 6157, 2023.

[18] D. O. Traini, G. Palmisano, C. Guerriero, and K. Peris, “Artificial intelligence in the assessment and grading of acne vulgaris: a systematic review,” Journal of Personalized Medicine, vol. 15, no. 6, p. 238, 2025.

[19] N. Yadav, S. M. Alfayeed, A. Khamparia, B. Pandey, D. N. Thanh, and S. Pande, “Hsv model-based segmentation driven facial acne detection using deep learning,” Expert Systems, vol. 39, no. 3, p. e12760, 2022.

[20] M. S. Junayed, M. B. Islam, and N. Anjum, “A transformer-based versatile network for acne vulgaris segmentation,” in 2022 Innovations in Intelligent Systems and Applications Conference (ASYU). IEEE, 2022, pp. 1–6.

[21] S. Kim, H. Yoon, and J. Lee, “Semi-supervised facial acne segmentation using bidirectional copy–paste,” Diagnostics, vol. 14, no. 10, p. 1040, 2024.

[22] S. Kim, C. Lee, G. Jung, H. Yoon, J. Lee, and S. Yoo, “Facial acne segmentation based on deep learning with center point loss,” in 2023 IEEE 36th International Symposium on Computer-Based Medical Systems (CBMS). IEEE, 2023, pp. 678–683.

[23] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PmLR, 2021, pp. 8748–8763.

[24] C. Jia, Y. Yang, Y. Xia, Y.-T. Chen, Z. Parekh, H. Pham, Q. Le, Y.-H. Sung, Z. Li, and T. Duerig, “Scaling up visual and vision-language representation learning with noisy text supervision,” in International conference on machine learning. PMLR, 2021, pp. 4904–4916.

[25] K. Zhou, J. Yang, C. C. Loy, and Z. Liu, “Conditional prompt learning for vision-language models,” in Proceedings of the IEEE/CVF confer-

ence on computer vision and pattern recognition, 2022, pp. 16 816– 16 825.

[26] Y. Zhong, J. Yang, P. Zhang, C. Li, N. Codella, L. H. Li, L. Zhou, X. Dai, L. Yuan, Y. Li et al., “Regionclip: Region-based language-image pretraining,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 16 793–16 803.

[27] J. Jeong, Y. Zou, T. Kim, D. Zhang, A. Ravichandran, and O. Dabeer, “Winclip: Zero-/few-shot anomaly classification and segmentation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 19 606–19 616.

[28] H. Ding, C. Liu, S. Wang, and X. Jiang, “Vlt: Vision-language transformer and query generation for referring segmentation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 6, pp. 7900–7916, 2022.

[29] Z. Yang, J. Wang, Y. Tang, K. Chen, H. Zhao, and P. H. Torr, “Lavt: Language-aware vision transformer for referring image segmentation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 18 155–18 165.

[30] Z. Wang, Y. Lu, Q. Li, X. Tao, Y. Guo, M. Gong, and T. Liu, “Cris: Clip-driven referring image segmentation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 11 686–11 695.

[31] F. Liang, B. Wu, X. Dai, K. Li, Y. Zhao, H. Zhang, P. Zhang, P. Vajda, and D. Marculescu, “Open-vocabulary semantic segmentation with mask-adapted clip,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 7061–7070.

[32] J. Xu, S. Liu, A. Vahdat, W. Byeon, X. Wang, and S. De Mello, “Openvocabulary panoptic segmentation with text-to-image diffusion models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 2955–2966.

[33] C. Zhou, C. C. Loy, and B. Dai, “Extract free dense labels from clip,” in European conference on computer vision. Springer, 2022, pp. 696–712.

[34] S. Cho, H. Shin, S. Hong, A. Arnab, P. H. Seo, and S. Kim, “Catseg: Cost aggregation for open-vocabulary semantic segmentation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 4113–4123.

[35] B. Xie, J. Cao, J. Xie, F. S. Khan, and Y. Pang, “Sed: A simple encoderdecoder for open-vocabulary semantic segmentation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 3426–3436.

[36] M. Lee, S. Cho, J. Lee, S. Yang, H. Choi, I.-J. Kim, and S. Lee, “Effective sam combination for open-vocabulary semantic segmentation,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 26 081–26 090.

[37] Q. Tang, L. Fan, M. Pagnucco, and Y. Song, “Prototype-based image prompting for weakly supervised histopathological image segmentation,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2025, pp. 30 271–30 280.

[38] Z. Li, Y. Li, Q. Li, P. Wang, D. Guo, L. Lu, D. Jin, Y. Zhang, and Q. Hong, “Lvit: language meets vision transformer in medical image segmentation,” IEEE transactions on medical imaging, vol. 43, no. 1, pp. 96–107, 2023.

[39] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional transformers for language understanding,” in Proceedings of the 2019 conference of the North American chapter of the associationfor computational linguistics: human language technologies, volume 1 (long and short papers), 2019, pp. 4171–4186.

[40] T. Koleilat, H. Asgariandehkordi, H. Rivaz, and Y. Xiao, “Medclip-sam: Bridging text and image towards universal medical image segmentation,” in International conference on medical image computing and computerassisted intervention. Springer, 2024, pp. 643–653.

[41] L. Shen, F. Shang, X. Huang, Y. Yang, H. Huang, and S. Xiang, “Segicl: A multimodal in-context learning framework for enhanced segmentation in medical imaging,” arXiv preprint arXiv:2403.16578, 2024.

[42] W. Zhang, Z. Zhang, M. He, and J. Ye, “Organ-aware multi-scale medical image segmentation using text prompt engineering,” arXiv preprint arXiv:2503.13806, 2025.

[43] L. Fang, X. Li, Y. Xu, F. Zhang, and C. Zhang, “Driven by textual knowledge: A text-view enhanced knowledge transfer network for lung infection region segmentation,” Medical Image Analysis, vol. 103, p. 103625, 2025.

[44] D. Shan, Z. Li, Y. Li, Q. Li, J. Tian, and Q. Hong, “Stpnet: Scaleaware text prompt network for medical image segmentation,” IEEE Transactions on Image Processing, vol. 34, pp. 3169–3180, 2025.

[45] Q. Pan, Z. Li, G. Yang, Q. Yang, and B. Ji, “Evivlm: When evidential learning meets vision language model for medical image segmentation,” IEEE Transactions on Medical Imaging, 2025.

[46] C. Lugaresi, J. Tang, H. Nash, C. McClanahan, E. Uboweja, M. Hays, F. Zhang, C.-L. Chang, M. G. Yong, J. Lee et al., “Mediapipe: A framework for building perception pipelines,” arXiv preprint arXiv:1906.08172, 2019.

[47] B. C. Russell, A. Torralba, K. P. Murphy, and W. T. Freeman, “Labelme: a database and web-based tool for image annotation,” International journal of computer vision, vol. 77, no. 1, pp. 157–173, 2008.

[48] Q. Zhou, G. Pang, Y. Tian, S. He, and J. Chen, “Anomalyclip: Objectagnostic prompt learning for zero-shot anomaly detection,” pp. 49 705– 49 737, 2024.

[49] D.-P. Fan, G.-P. Ji, T. Zhou, G. Chen, H. Fu, J. Shen, and L. Shao, “Pranet: Parallel reverse attention network for polyp segmentation,” in International conference on medical image computing and computerassisted intervention. Springer, 2020, pp. 263–273.

[50] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, “Focal loss´ for dense object detection,” in Proceedings of the IEEE international conference on computer vision, 2017, pp. 2980–2988.

[51] X. Wu, N. Wen, J. Liang, Y.-K. Lai, D. She, M.-M. Cheng, and J. Yang, “Joint acne image grading and counting via label distribution learning,” in 2019 IEEE/CVF international conference on computer vision (ICCV). IEEE, 2019, pp. 10 641–10 650.

[52] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” arXiv preprint arXiv:1711.05101, 2017.

[53] H. Touvron, M. Cord, M. Douze, F. Massa, A. Sablayrolles, and H. Jegou, “Training data-efficient image transformers & distillation´ through attention,” in International conference on machine learning. PMLR, 2021, pp. 10 347–10 357.

[54] E. Alsentzer, J. Murphy, W. Boag, W.-H. Weng, D. Jindi, T. Naumann, and M. McDermott, “Publicly available clinical bert embeddings,” in Proceedings of the 2nd clinical natural language processing workshop, 2019, pp. 72–78.

[55] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo et al., “Segment anything,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 4015–4026.

[56] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in International Conference on Medical image computing and computer-assisted intervention. Springer, 2015, pp. 234–241.

[57] H. Zhao, J. Shi, X. Qi, X. Wang, and J. Jia, “Pyramid scene parsing network,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 2881–2890.

[58] L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, and H. Adam, “Encoderdecoder with atrous separable convolution for semantic image segmentation,” in Proceedings of the European conference on computer vision (ECCV), 2018, pp. 801–818.

[59] Z. Zhou, M. M. Rahman Siddiquee, N. Tajbakhsh, and J. Liang, “Unet++: A nested u-net architecture for medical image segmentation,” in International workshop on deep learning in medical image analysis. Springer, 2018, pp. 3–11.

[60] Z. Gu, J. Cheng, H. Fu, K. Zhou, H. Hao, Y. Zhao, T. Zhang, S. Gao, and J. Liu, “Ce-net: Context encoder network for 2d medical image segmentation,” IEEE transactions on medical imaging, vol. 38, no. 10, pp. 2281–2292, 2019.

[61] E. Xie, W. Wang, Z. Yu, A. Anandkumar, J. M. Alvarez, and P. Luo, “Segformer: Simple and efficient design for semantic segmentation with transformers,” Advances in neural information processing systems, vol. 34, pp. 12 077–12 090, 2021.

[62] J. Chen, Y. Lu, Q. Yu, X. Luo, E. Adeli, Y. Wang, L. Lu, A. L. Yuille, and Y. Zhou, “Transunet: Transformers make strong encoders for medical image segmentation,” arXiv preprint arXiv:2102.04306, 2021.

[63] H. Cao, Y. Wang, J. Chen, D. Jiang, X. Zhang, Q. Tian, and M. Wang, “Swin-unet: Unet-like pure transformer for medical image segmentation,” in European conference on computer vision. Springer, 2022, pp. 205–218.

[64] F. Isensee, P. F. Jaeger, S. A. Kohl, J. Petersen, and K. H. Maier-Hein, “nnu-net: a self-configuring method for deep learning-based biomedical image segmentation,” Nature methods, vol. 18, no. 2, pp. 203–211, 2021.

[65] J. Liu, H. Yang, H.-Y. Zhou, L. Yu, Y. Liang, Y. Yu, S. Zhang, H. Zheng, and S. Wang, “Swin-umamba†: Adapting mamba-based vision foundation models for medical image segmentation,” IEEE Transactions on Medical Imaging, vol. 44, no. 10, pp. 3898–3908, 2025.

[66] Z. Wang, Y. Bai, Y. Zhou, and C. Xie, “Can cnns be more robust than transformers?” arXiv preprint arXiv:2206.03452, 2022.

## SUPPLEMENTARY MATERIAL

## S1. DATASET DETAILS

## A. Internal cohort

The internal cohort comprises 258 patients (38.4% male; mean age 22.7 ± 5.97 years) who visited the Department of Dermatology, Seoul National University Hospital between February and July 2020. Photographs were acquired with Canon EOS 550D and Nikon D7100 cameras under standardized conditions. Fitzpatrick phototype was not routinely recorded in the clinical dataset; all participants were of Korean ethnicity, a population in which phototypes III and IV predominate.

## B. IGA grading protocol

Investigator’s Global Assessment scores were assigned retrospectively at the visit level: three board-certified dermatologists independently reviewed all images acquired at a given visit and graded that visit on the standard five-point scale (0, clear; 1, almost clear; 2, mild; 3, moderate; 4, severe). The final grade was determined by majority voting; when the three graders assigned three different grades, the case was resolved through joint discussion until consensus was reached. Among the 22 visits in the internal test set, 3, 7, 8 and 4 visits were graded as IGA 1, 2, 3 and 4 respectively, and no visit was graded as IGA 0.

## C. External subset construction

The AI Hub Korean Skin Condition Measurement Dataset includes 13,936 facial images, 84,688 skin condition measurement records and 125,424 labeled data entries. Because the collection contains many images without visible inflammatory lesions, we first applied an acne detection model trained on the public ACNE04 dataset as a screening step. A deliberately low confidence threshold (0.3) was used so that the screening favoured recall, retaining images with even weak detection responses. From this candidate pool we randomly sampled 100 images captured with digital cameras in a controlled environment and 100 images captured with smartphones under uncontrolled conditions. Each sampled image was then reviewed by the dermatologist who adjudicated the internal set, and those without confirmed inflammatory acne were excluded, yielding the final external test sets of 71 and 56 images. Consistent with the permissive screening threshold, only 71% and 56% of the sampled images were confirmed to contain inflammatory acne. A residual sampling bias nevertheless remains, restricted to images whose lesions the screening model failed to detect even at this low threshold, and this should be considered when interpreting the external validation results.

## D. External imaging conditions and metadata

Because the AI Hub collection was compiled for general facial skin analysis rather than from an acne patient population, the two external subsets differ from the internal cohort in lesion burden. They also differ in the type of imaging shift they probe. The controlled subset was acquired with digital cameras under a standardized setup comparable to that of the internal cohort and therefore mainly reflects a shift in population and acquisition site rather than in imaging device. The real-world subset was acquired with smartphone cameras and includes images taken with both front- and rear-facing cameras, introducing additional variability in resolution, field of view and colour processing. The AI Hub distribution does not specify the individual camera models, and it provides neither participant demographics nor Fitzpatrick phototypes; these characteristics therefore cannot be reported for the external subsets.

TABLE S1  
ANATOMICAL SYNONYMS USED IN THE UNSEEN-PROMPT EXPERIMENT.EACH TRAINING REGION NAME WAS REPLACED AT INFERENCE BY ONE OFTHE TWO ALTERNATIVES, CHOSEN AT RANDOM; NONE APPEARS INTRAINING.
<table><tr><td>Seen (training)</td><td>Unseen 1</td><td>Unseen 2</td></tr><tr><td>forehead</td><td>supraorbital area</td><td>frontal region</td></tr><tr><td>left cheek</td><td>left malar region</td><td>left buccal area</td></tr><tr><td>right cheek</td><td>right malar region</td><td>right buccal area</td></tr><tr><td>chin</td><td>mentum</td><td>mental region</td></tr><tr><td>nose</td><td>nasal region</td><td>nasal bridge</td></tr><tr><td>left temple</td><td>left pterion area</td><td>left temporal region</td></tr><tr><td>right temple</td><td>right pterion area</td><td>right temporal region</td></tr><tr><td>glabella</td><td>interciliary space</td><td>intercilium</td></tr><tr><td>left eyebrow</td><td>left supraorbital ridge</td><td>left superciliary arch</td></tr><tr><td>right eyebrow</td><td>right supraorbital ridge</td><td>right superciliary arch</td></tr><tr><td>left eye</td><td>left periorbital region</td><td>left orbital area</td></tr><tr><td>right eye</td><td>right periorbital region</td><td>right orbital area</td></tr><tr><td>lip</td><td>labial region</td><td>vermilion area</td></tr><tr><td>upper lip</td><td></td><td></td></tr><tr><td>neck</td><td>upper labial region cervical region</td><td>infranasal area anterior cervical area</td></tr></table>

TABLE S2

ABLATION STUDY ON DIFFERENT TEXT PROMPT STRATEGIES FOR VL-ACNESEG IN INTERNAL DATA. BEST VALUES ARE IN BOLD AND SECOND-BEST ARE UNDERLINED.
<table><tr><td>Metrics</td><td>Region Prompt</td><td>Level Prompt</td><td>Num Prompt</td></tr><tr><td>IoU</td><td>0.3602</td><td>0.3521</td><td>0.3594</td></tr><tr><td>Dice</td><td>0.5296</td><td>0.5208</td><td>0.5288</td></tr><tr><td>Precision</td><td>0.5202</td><td>0.4889</td><td>0.4940</td></tr><tr><td>Recall</td><td>0.5393</td><td>0.5572</td><td>0.5688</td></tr></table>

## S2. TEXT PROMPT DETAILS

Table S1 lists the anatomical synonyms used in the unseenprompt experiment reported in the main paper; none of the alternatives appears during training. Table S2 compares three prompt contents on the internal dataset: Region supplies spatial guidance only, Level adds a qualitative severity descriptor, and Num adds an exact lesion count per region. Adding count descriptors raises recall by widening the search space but lowers precision, reflecting a trade-off between spatial guidance and quantitative prompt engineering.

## S3. LOCALIZATION UNDER DIFFERENT PATCH RESOLUTIONS

Localization maps for the three tiling configurations evaluated in Section IV-E of the main paper. Coarser tiling spreads activation over confounding skin features, whereas the finer division keeps the response on the lesions themselves.

![](images/d1ce58003334a605a1a38473cdb8d3491093dc1a94695c83976de185180a9b4b.jpg)  
(a) GT

![](images/28bfd5e81d33c032bf38868b2caf8055098870e79a0c5f76637452f4d1eeefd7.jpg)

![](images/35fa7ae6d0f2da767eb56b87512885508e2fdbd8f8a3632660702ca804d4b61b.jpg)  
(c) 3 × 2

(b) 4 × 3  
![](images/7700f00d97fb51b53e335161afd667986c86741c86eef4c2125ab50b4530b82a.jpg)

![](images/a1f2319590c1f01d23306e983ba7d47590916c15558fe29cef8fbc799c4e5360.jpg)  
(d) 2 × 1  
(e) Global Prompt

Fig. S1: Ablation study on patch granularity and regional textual cues. Finer patch division improves localization precision. The model still localizes lesions under a global text prompt alone (e).  
![](images/ad6ec43648138492589fce87cdacc0f94ee9f90c6a7af87ea3c688450656b75c.jpg)  
(a) Ground Truth

![](images/b60f1954e1e00fbe796a37d6e8531aec0256073fe2cc000ced7d3b23166a9bd1.jpg)  
(b) Full Model

![](images/33c5638c648dd25013f264cbeaca753478be88fe59211332cfa15b1aa50650e2.jpg)  
(c) Global Prompt  
Fig. S2: Localization heatmaps in a smartphone environment. (b) The full model identifies lesion-suspected areas with high precision and sharpness. (c) Although the global text prompt configuration exhibits broader localization extending beyond facial boundaries, it still captures the actual lesion sites.

## S4. QUALITATIVE ANALYSIS OF LOCALIZATION IN SMARTPHONE IMAGE

Figure S2 shows localization maps for a smartphone image under the two prompting conditions. With region-level prompts the response is concentrated on the lesion sites. With a global prompt the activation is broader and occasionally extends beyond the facial boundary, but the lesion sites are still recovered, so the model remains usable where region information is unavailable.

## S5. PATCH RESOLUTION ON EXTERNAL DATASETS

TABLE S3  
EFFECT OF PATCH TILING RESOLUTION ON THE TWO EXTERNAL DATASETS. COMPONENT-WISE ABLATIONS ON THE EXTERNAL DATASETS ARE REPORTED IN THE MAIN PAPER. BEST VALUES ARE IN BOLD AND SECOND-BEST ARE UNDERLINED.
<table><tr><td rowspan="3">Ablation Model</td><td colspan="10">External Datasets (Evaluation Only)</td></tr><tr><td colspan="5">Controlled Setting (1)</td><td colspan="5">Real-world Setting (2)</td></tr><tr><td>IoU</td><td>Dice</td><td>Precision</td><td>Recall</td><td>TP/img</td><td>FP/img</td><td>IoU Dice</td><td>Precision</td><td>Recall</td><td>TP/img</td><td>FP/img</td></tr><tr><td>VL-AcneSeg (Full Model)</td><td>0.4232</td><td>0.5948</td><td>0.6110</td><td>0.5794</td><td>1.64 0.60</td><td>0.3044</td><td>0.4667</td><td>0.4882</td><td>0.4471</td><td>1.50</td><td>0.82</td></tr><tr><td>3 × 2 Patches</td><td>0.3812</td><td>0.5520</td><td>0.5664</td><td>0.5383</td><td>1.83 0.79</td><td>0.2443</td><td>0.3927</td><td>0.3787</td><td>0.4077</td><td>1.11</td><td>0.66</td></tr><tr><td>2 × 1 Patches</td><td>0.3019</td><td>0.4638</td><td>0.4593</td><td>0.4683</td><td>1.58</td><td>0.77</td><td>0.1213 0.2164</td><td>0.1354</td><td>0.5383</td><td>0.89</td><td>1.43</td></tr></table>

Table S3 shows that the gain from finer tiling observed on the internal dataset also holds on both external datasets.

## S6. CONFIDENCE INTERVALS FOR BETWEEN-METHOD DIFFERENCES

The main paper reports confidence intervals for each method separately, which do not indicate whether two methods differ, since the same patients contribute to both estimates. Table S4 therefore reports the difference itself, resampled in pairs so that the correlation between the two methods is preserved. Every difference is significant at the 0.05 level except that of SAM, whose interval is the only one to include zero; as reported in the main paper, that comparison was examined further by retraining both configurations with three seeds. The width of the intervals reflects the size of the test set: the differences against the conventional architectures are estimated to within a few hundredths of an IoU point, whereas those against the vision-language baselines, where the margin is smaller, are correspondingly less precise.

## S7. ADDITIONAL EVALUATION METRICS

This section reports two metrics that address aspects the main comparison leaves open. The first is small-lesion performance at the lesion level. Acne lesions vary widely in size, and the pixel-level metrics of the main paper are dominated by the larger ones simply because they contribute more pixels; a method could therefore score well while missing most of the small lesions. We isolate the smallest quartile and count lesions rather than pixels, so that each lesion carries equal weight. The second is the area under the precision–recall curve. All results in the main paper use a binarization threshold of 0.5, fixed for every method without per-model tuning, and AUPRC is computed from the probability maps before that threshold is applied, so it indicates whether the reported ordering depends on that choice.

Small lesions. All quantities in this block are computed at the lesion level, over connected components rather than pixels. Recall alone is misleading here. U-Net recovers 84.0% of the small lesions but only 6.0% of what it marks is one, and the same pattern holds for Swin-UMamba†, U-Net++, TransUNet and PSPNet, all of which reach recall comparable to or above the proposed method at less than half its precision. CAT-Seg sits at the other extreme, with the highest precision of any method and the lowest recall among the vision-language ones. Both configurations of VL-AcneSeg are in the upper range on both quantities, which is why they lead on F1, and the globalprompt configuration exceeds every vision-language baseline even though those baselines receive region-level prompts.

## TABLE S4

DIFFERENCES IN IOU AND DICE BETWEEN VL-ACNESEG AND EACH BASELINE ON THE INTERNAL TEST SET, WITH 95% CONFIDENCE INTERVALS FROM PAIRED PATIENT-LEVEL CLUSTER BOOTSTRAP OVER THE 16 TEST PATIENTS. A POSITIVE VALUE FAVOURS THE PROPOSED METHOD. VISION-LANGUAGE BASELINES RECEIVE REGION-LEVEL PROMPTS AND ARE THEREFORE COMPARED AGAINST THE REGION-LEVEL CONFIGURATION; THE REMAINING METHODS TAKE NO TEXT INPUT AND ARE COMPARED AGAINST THE GLOBAL PROMPT.

ADDITIONAL METRICS ON THE INTERNAL TEST SET. Small lesions ARE THOSE IN THE LOWEST QUARTILE OF THE LESION-AREA DISTRIBUTION OVER THE WHOLE ANNOTATED SET, CORRESPONDING TO AN AREA BELOW $5 0 0 \mathrm { P X } ^ { 2 } ;$ ; RECALL, PRECISION AND F1 ARE COMPUTED OVER THESE LESIONS AT THE LESION LEVEL. AUPRC IS THE AREA UNDER THE PRECISION–RECALL CURVE, COMPUTED FROM THE PREDICTED PROBABILITY MAPS AND THEREFORE INDEPENDENT OF THE BINARIZATION THRESHOLD. ROWS ARE ORDERED BY SMALL-LESION F1. BEST VALUES ARE IN BOLD AND SECOND-BEST ARE UNDERLINED.
<table><tr><td>Baseline</td><td>∆IoU [95% CI]</td><td>p</td><td>∆Dice [95% CI]</td><td>p</td></tr><tr><td colspan="5">Vision-language methods, compared against VL-AcneSeg (region)</td></tr><tr><td>EviVLM</td><td>+0.031 [0.001, 0.069]</td><td>0.048</td><td>+0.034 [0.001, 0.080]</td><td>0.046</td></tr><tr><td>CAT-Seg</td><td>+0.069 [0.012, 0.136]</td><td>0.013</td><td>+0.079 [0.011, 0.166]</td><td>0.015</td></tr><tr><td>SED</td><td>+0.042 [0.003, 0.097]</td><td>0.034</td><td>+0.047 [0.003, 0.112]</td><td>0.038</td></tr><tr><td colspan="5">Remaining methods, compared against VL-AcneSeg (global)</td></tr><tr><td>SAM</td><td>+0.019 [-0.037, 0.078]</td><td>0.580</td><td>+0.021 [-0.041, 0.091]</td><td>0.592</td></tr><tr><td>Swin-UMamba†</td><td>+0.053 [0.025, 0.076]</td><td>0.005</td><td>+0.061 [0.018, 0.085]</td><td>0.005</td></tr><tr><td>TransUNet</td><td>+0.078 [0.021, 0.144]</td><td>0.005</td><td>+0.092 [0.023, 0.182]</td><td>0.005</td></tr><tr><td>CE-Net</td><td>+0.119 [0.065, 0.170]</td><td>&lt;0.001</td><td>+0.146 [0.076, 0.211]</td><td>&lt;0.001</td></tr><tr><td>Swin-UNet</td><td>+0.132 [0.091, 0.170]</td><td>&lt;0.001</td><td>+0.163 [0.110, 0.217]</td><td>&lt;0.001</td></tr><tr><td>SegFormer</td><td>+0.136 [0.090, 0.177]</td><td>&lt;0.001</td><td>+0.168 [0.109, 0.228]</td><td>&lt;0.001</td></tr><tr><td>nnU-Net</td><td>+0.150 [0.110, 0.185]</td><td>&lt;0.001</td><td>+0.188 [0.136, 0.234]</td><td>&lt;0.001</td></tr><tr><td>PSPNet</td><td>+0.161 [0.132, 0.185]</td><td>&lt;0.001</td><td>+0.203 [0.172, 0.230]</td><td>&lt;0.001</td></tr><tr><td>DeepLabV3+</td><td>+0.160 [0.131, 0.194]</td><td>&lt;0.001</td><td>+0.202 [0.164, 0.253]</td><td>&lt;0.001</td></tr><tr><td>U-Net++</td><td>+0.183 [0.121, 0.236]</td><td>&lt;0.001</td><td>+0.236 [0.158, 0.311]</td><td>&lt;0.001</td></tr><tr><td>U-Net</td><td>+0.199 [0.165, 0.229]</td><td>&lt;0.001</td><td>+0.260 [0.225, 0.292]</td><td>&lt;0.001</td></tr></table>

TABLE S5

facial region so that lesions a few pixels across remain visible, and includes both prompting conditions of the proposed method.

<table><tr><td rowspan="2">Model</td><td colspan="3">Small lesions  $( \leq$  500 px2)</td><td rowspan="2">AUPRC</td></tr><tr><td>Recall</td><td>Precision</td><td>F1</td></tr><tr><td>VL-AcneSeg (region)</td><td>0.494</td><td>0.451</td><td>0.472</td><td>0.4138</td></tr><tr><td>VL-AcneSeg (global)</td><td>0.492</td><td>0.432</td><td>0.460</td><td>0.4068</td></tr><tr><td>SAM</td><td>0.457</td><td>0.432</td><td>0.444</td><td>0.3400</td></tr><tr><td>SED</td><td>0.431</td><td>0.426</td><td>0.428</td><td>0.4167</td></tr><tr><td>EviVLM</td><td>0.453</td><td>0.405</td><td>0.428</td><td>0.4037</td></tr><tr><td>CAT-Seg</td><td>0.358</td><td>0.466</td><td>0.405</td><td>0.3922</td></tr><tr><td>DeepLabV3+</td><td>0.381</td><td>0.392</td><td>0.386</td><td>0.3958</td></tr><tr><td>CE-Net</td><td>0.420</td><td>0.275</td><td>0.332</td><td>0.1655</td></tr><tr><td>Swin-UMamba†</td><td>0.554</td><td>0.214</td><td>0.309</td><td>0.3963</td></tr><tr><td>PSPNet</td><td>0.503</td><td>0.208</td><td>0.294</td><td>0.3261</td></tr><tr><td>U-Net++</td><td>0.531</td><td>0.194</td><td>0.284</td><td>0.1582</td></tr><tr><td>Swin-UNet</td><td>0.333</td><td>0.247</td><td>0.284</td><td>0.1882</td></tr><tr><td>TransUNet</td><td>0.506</td><td>0.182</td><td>0.268</td><td>0.1745</td></tr><tr><td>nnU-Net</td><td>0.420</td><td>0.169</td><td>0.241</td><td>0.3249</td></tr><tr><td>U-Net</td><td>0.840</td><td>0.060</td><td>0.112</td><td>0.3582</td></tr><tr><td>SegFormer</td><td>0.136</td><td>0.083</td><td>0.103</td><td>0.1840</td></tr></table>

Threshold independence. Both configurations of the proposed method fall in the upper range, so the advantage reported in the main paper does not depend on the choice of threshold. The ordering below them differs from the pixel-level one, since AUPRC also reflects how well a model ranks pixels at thresholds it is never evaluated at. The top group is tightly packed, so AUPRC should be read as separating the stronger methods from the weaker ones rather than as distinguishing between them.

## S8. COMPREHENSIVE QUALITATIVE COMPARISON OF ACNE SEGMENTATION

Figs. S3–S5 extend the qualitative comparison of the main paper to all baselines, on the internal, external controlled and external smartphone datasets. Each figure enlarges a single

![](images/d7c68f382d4023d2d07e065478c0e4419ee13f87f517cf53244e2e74f96ceb3b.jpg)  
Fig. S3: Comprehensive qualitative comparison on the internal dataset (Part I). The left panel shows the input with the reference annotation; the green box marks the region enlarged in the remaining panels. Predicted masks are overlaid in red.

![](images/228e0262808c1f4b00694e5b77a54a91aba53144cb5fde72a1c47e2383b68cd1.jpg)  
Fig. S4: Comprehensive qualitative comparison on the external controlled dataset (Part II).

Ground Truth  
VLAcneSeg (Region)  
VLAcneSeg (Global)  
SAM  
Swin-UMambat  
EviVLM  
![](images/b39b95dc171563effac54a979b3746aff62db31a83dccc447e141feb6b179f89.jpg)  
Fig. S5: Comprehensive qualitative comparison on the external smartphone dataset (Part III).
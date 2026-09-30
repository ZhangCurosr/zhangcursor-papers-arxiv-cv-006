# Task-Oriented Visual Feature Compression via Residual Vector Quantization for Device-Edge Multimodal Inference

Luning Pang, Cheng Yuan, Jiawei Shao, Member, IEEE, Mingtao Huang, and Yuan Shen, Senior Member, IEEE

Abstract—Large multimodal models (LMMs) support diverse visual understanding and reasoning tasks but are often impractical to run entirely on resource-constrained devices. Device– edge co-inference reduces device computation, yet transmitting visual data over bandwidth-limited uplinks can introduce substantial delay. Task-oriented feature compression (TOFC) reduces the payload through feature aggregation and entropy coding. However, continuous-feature coding remains costly, and queryagnostic aggregation may discard task-relevant local evidence. We propose query-guided task-oriented feature compression (Q-TOFC) for device–edge multimodal inference. Q-TOFC employs residual vector quantization (RVQ) to encode each merged feature as a compact sequence of codebook indices, reducing its representation cost and allowing more features to be transmitted. It further incorporates query relevance into feature aggregation and uses a quantization error compensation adapter to mitigate the distortion introduced by discrete quantization. Experiments on seven multimodal benchmarks show that Q-TOFC reduces the visual payload by 53.6% relative to TOFC while maintaining comparable average normalized task performance. End-to-end latency evaluations further demonstrate lower latency under bandwidth-constrained uplinks.

Index Terms—Device-edge co-inference, feature compression, large multimodal models, residual vector quantization, taskoriented communication.

## I. INTRODUCTION

Large multimodal models (LMMs) combine visual perception with language reasoning [1] and support applications ranging from robotic perception [2] and autonomous navigation [3] to conversational agents [4]. A typical LMM contains a vision encoder, such as CLIP [5] or SigLIP [6], a multimodal projector, and a large language model (LLM) that generates responses from visual and textual tokens [7]–[9]. Running this entire pipeline on a resource-constrained mobile device is often impractical [10], [11]. Device–edge co-inference therefore partitions the computation between the device and a nearby edge server [12], [13], but the partition also determines what visual data must traverse the wireless uplink.

Visual information can be offloaded at either the image or feature level. Image-level offloading sends a compressed image to the edge server, where the image is decoded and processed by the vision encoder. Conventional JPEG [14] and learned neural codecs [15]–[18] can substantially reduce the image bitstream, but they leave image decoding and visual feature extraction at the server. Their compression objectives operate in the image domain and are designed to preserve reconstructed image quality. Consequently, they do not directly exploit redundancy in the intermediate visual representations consumed by the LMM.

Feature-level co-inference instead executes the vision encoder on the device and transmits its intermediate representations, avoiding visual feature extraction at the server. The obstacle is the size of dense visual features. For the LLaVA-OneVision backbone used in our primary experiments, one SigLIP patch contains 729 features of dimension 1152 and exceeds 1.6 MiB in FP16. High-resolution inputs may contain multiple patches. Task-oriented feature compression (TOFC) [19] demonstrates that directly compressing these represen tations can provide a stronger communication–performance trade-off than image-level codecs. TOFC first applies density peaks clustering based on K nearest neighbors (DPC-KNN) to aggregate the visual features in each patch into a smaller set of merged representations, thereby reducing the number of features to be transmitted. It then entropy-codes the merged representations.

TOFC establishes the value of feature-level transmission, but its compression mechanism leaves two opportunities for improvement under a tight uplink budget. First, DPC-KNN selects cluster centers from the visual feature distribution without using the textual query, even though the evidence needed by the downstream LMM is query dependent. Second, very low payloads require reducing each patch to a small number of merged features. As fewer representations must summarize the original visual tokens, localized evidence such as a sign or small object can be lost. These observations motivate a different allocation of the communication budget: retain more query-relevant merged features while reducing the representation cost of each retained feature.

We propose query-guided task-oriented feature compression (Q-TOFC), a visual feature compression framework that combines query-aware token reduction with RVQ-based coding and feature refinement. A lightweight relevance score incorporates the textual query into the DPC-KNN centerranking criterion. The total number of selected centers remains unchanged, but visual features that are more relevant to the query are more likely to be selected as cluster centers. Q-

![](images/75534d21ba679209c3332dcb444af562a65a881dcf6afae7bd93458376a7c4f7.jpg)  
Fig. 1. Overview of Q-TOFC. On the user device, the vision and text encoders extract visual and query features. A lightweight query-aware scoring module estimates feature relevance, after which query-guided clustering produces a compact set of merged features. Multi-layer residual vector quantization (RVQ) maps each merged feature to discrete indices. On the edge server, codebook lookup and a quantization error compensation adapter (QECA) reconstruct the features for subsequent LMM inference.

TOFC adapts residual vector quantization (RVQ) [20], [21] to encode each merged visual feature as a compact sequence of codebook indices. Reducing the representation cost per feature allows the framework to retain more merged features at a low overall payload. The edge server reconstructs the features through codebook lookup, after which a quantization error compensation adapter (QECA) predicts a residual correction for the quantized representation. During end-to-end fine-tuning, QECA is jointly shaped by the language-modeling objective and feature-level VQ regularization, balancing downstream task supervision with feature fidelity.

We compare Q-TOFC with TOFC, ELIC, and JPEG on seven multimodal benchmarks using rate–performance and end-to-end latency evaluations. At comparable average normalized performance, Q-TOFC reduces the visual payload by 53.6% relative to TOFC. Under bandwidth-constrained uplinks, the lower payload also translates into lower endto-end latency despite the additional device-side encoding operations. Additional experiments with LLaVA-1.5 show that the same trend persists with a different vision encoder and image-patching scheme.

Our main contributions are summarized as follows:

• We propose a query-conditioned visual feature compression framework for device–edge LMM inference, where DPC-KNN density–diversity scores are combined with text–visual relevance to guide cluster-center selection.

• We design an RVQ-based codec that lowers the representation cost of each merged feature, enabling more query-relevant features to be retained under a constrained payload, and introduce QECA to refine the reconstructed representations.

• We validate Q-TOFC through extensive experiments against TOFC, ELIC, and JPEG on seven multimodal benchmarks. Q-TOFC reduces the visual payload by

53.6% relative to TOFC while maintaining comparable average normalized performance. In the primary LLaVA-OneVision setting, it further reduces end-to-end latency by up to 29.0% relative to TOFC under bandwidthconstrained uplinks.

• Component ablations show that query guidance and QECA provide complementary performance gains. The RVQ-depth study further characterizes the trade-off among communication cost, codebook storage, and task performance.

The remainder of the paper is organized as follows. Section II surveys related literature. Section III formalizes the system model. Section IV details the Q-TOFC framework. Section V reports experimental results, and Section VI concludes the paper.

## II. RELATED WORK

## A. Task-Oriented Communication

Conventional wireless communication systems treat every bit of the transmitted payload equally, aiming for faithful reconstruction of the source signal. Task-oriented communication departs from this philosophy by asking a different question: what information does the receiver actually need to perform well on its assigned task? The theoretical foundation is often traced to the information bottleneck (IB) principle [22], which formalizes the trade-off between compression and relevance. Practical instantiations of this idea have been explored in feature compression for device-edge inference [23], cooperative edge inference [24], robust task-oriented transmission [25], and video analytics over temporally correlated frames [26]. Task-oriented edge–cloud co-inference has also been extended to human action understanding by converting pose sequences into discrete motion tokens for transmission [27]. A common theme is that discarding taskirrelevant redundancy at the transmitter can shrink the payload by more than an order of magnitude with little impact on accuracy. TOFC [19] applies this principle to device–edge LMM inference. It places vision encoding and feature compression on the user device, while the edge server decodes the transmitted representation, applies the multimodal projector, and performs LLM inference.

## B. Neural Data Compression

Reducing the storage and transmission cost of images has a long history. Hand-crafted codecs such as JPEG [14] and JPEG 2000 [28] apply block-based frequency transforms followed by quantization and entropy coding. Over the past decade, learned compression methods have pushed rate–distortion curves well beyond these classical bounds. Balle et al. [15], [16] in-´ troduced the hyperprior architecture that jointly optimizes a nonlinear transform with a learned probability model, and subsequent work added autoregressive context modeling [17], [29], [30] and more expressive prior structures [18]. Despite their success, these methods are generally trained to minimize pixel-level or perceptual distortion metrics such as PSNR, MS-SSIM, and LPIPS. Their bit allocation may therefore be inefficient when the downstream consumer is an LMM rather than a human viewer. In contrast, the present work compresses the latent features produced by the vision encoder and jointly optimizes the compression modules using the downstream token-prediction loss and feature-level VQ losses.

## C. Vector Quantization for Representation Learning

Vector quantization (VQ) replaces a continuous vector with the nearest entry in a learned codebook, yielding a discrete, compact code. The seminal VQ-VAE [31] demonstrated that such discretized representations can still support high-fidelity generation when coupled with an appropriate decoder. In the audio domain, SoundStream [20] and EnCodec [21] introduced RVQ, which stacks multiple VQ layers: after the first layer quantizes the input, each subsequent layer quantizes the residual between the input and the sum of previously selected codewords. This coarse-to-fine strategy achieves high spectral fidelity with a small number of bits per frame. In the visual domain, VQGAN [32] combined VQ with a perceptual loss and a patch-wise discriminator to learn codebooks for image synthesis. We repurpose RVQ as a compression primitive for task-oriented visual feature transmission and couple it with a lightweight compensation adapter for downstream LMM inference.

## D. LMM Inference Acceleration

Processing hundreds of visual tokens through dozens of transformer layers is a major source of latency in LMM inference, motivating extensive research on visual token reduction. Attention-based methods such as FastV and MustDrop remove low-importance visual tokens during LLM inference [33], [34], while projector-side methods such as TokenPacker and LLaVA-Mini condense the vision-encoder output before it enters the LLM [35], [36]. Other methods exploit redundancy among visual tokens. LLaVA-PruMerge adaptively prunes and merges tokens, VisionZip retains dominant tokens together with contextual summaries, and FocusLLaVA adjusts visual granularity through coarse-to-fine selection [37]–[39].

TOFC [19] places token reduction on the user device before feature transmission. It uses DPC-KNN to group visual features and averages the features assigned to each cluster. Consequently, features that are not selected as cluster centers still contribute to the merged representations instead of being directly discarded. The resulting smaller feature set reduces both the communication cost and the number of visual tokens processed by the server-side LMM.

Q-TOFC builds on this merge-before-transmission design. The main difference lies in how the merged features are represented. TOFC applies learned entropy coding after reducing each patch to a small set of merged features, whereas Q-TOFC uses RVQ to encode each merged feature as a compact sequence of codebook indices. The lower representation cost per feature allows Q-TOFC to retain more merged features under a constrained communication budget.

## III. SYSTEM MODEL

We consider a device-edge co-inference system in which a resource-constrained user device communicates with an edge server over a wireless uplink channel, as depicted in Fig. 1. The user submits a multimodal query consisting of an image $\pmb { v } _ { \mathrm { i } } \in \mathbb { R } ^ { h \times w \times 3 }$ and a textual instruction in natural language. Because the instruction is short (tens of tokens), it is available to the on-device compression module and is also sent directly to the server without compression. The image, which dominates the transmission budget, is processed locally by the device as described below.

Following the design of recent LMMs [7], [8], the backbone preprocessing scheme converts the input image into $n _ { \mathrm { p } }$ encoder inputs, such as a single resized image or multiple image patches. The vision encoder $f _ { \mathrm { v i s } }$ transforms these inputs, denoted by $v _ { \mathrm { p } }$ , into a tensor of visual features:

$$
\begin{array} { r } { \pmb { X } = f _ { \mathrm { v i s } } ( \pmb { v } _ { \mathrm { p } } ) , \quad \pmb { X } \in \mathbb { R } ^ { n _ { \mathrm { p } } \times n _ { \mathrm { v } } \times d _ { \mathrm { v } } } , } \end{array}\tag{1}
$$

where $n _ { \mathrm { v } }$ and $d _ { \mathrm { v } }$ denote the number of features per patch and the feature dimension, respectively.

The raw visual features X are far too large for a bandwidthlimited uplink. We therefore apply a cascade of lightweight modules on the device: (i) a query-aware token scoring module that estimates the relevance of each visual feature to the textual query; (ii) a query-guided clustering module that groups the $n _ { \mathrm { v } }$ features in each patch into $n _ { \mathrm { c } }$ clusters and averages the features within each cluster to produce $n _ { \mathrm { c } }$ merged features; and (iii) an RVQ codec that maps each merged feature to a sequence of codebook indices. The resulting indices are transmitted to the edge server.

Upon receiving the indices, the edge server reconstructs the quantized features through codebook lookup and applies QECA, detailed in Section IV, to reduce the distortion introduced by quantization. The reconstructed features are projected into the word-embedding space of the LLM by a twolayer MLP and concatenated with the tokenized instruction.

The LLM then autoregressively generates the textual response, which is sent back to the user device.

Following the convention in task-oriented communication research [24], [26], our channel model focuses on the uplink bandwidth constraint rather than explicitly modeling physicallayer effects such as fading and modulation, as channel coding and signal processing are beyond the scope of this work.

## A. Communication Cost Model

We characterize the communication cost by the number of bits used to transmit the visual representation for each multimodal query. Because all methods transmit the same textual instruction, it is not included in the visual payload. If the device transmits the visual features produced by the vision encoder without compression, the payload is

$$
B _ { \mathrm { r a w } } = n _ { \mathrm { p } } n _ { \mathrm { v } } d _ { \mathrm { v } } b _ { \mathrm { f } } ,\tag{2}
$$

where $b _ { \mathrm { f } }$ is the number of bits used to represent each feature element.

For an L-layer RVQ, each layer produces one codebook index for every merged feature. Let $M _ { k }$ denote the number of entries in the codebook at layer k. The index produced by that layer requires $\lceil \log _ { 2 } M _ { k } \rceil$ bits. The number of index bits required for one merged feature is therefore

$$
B _ { \mathrm { t o k } } = \sum _ { k = 0 } ^ { L - 1 } \left\lceil \log _ { 2 } M _ { k } \right\rceil .\tag{3}
$$

The RVQ index payload of one request is

$$
B _ { \mathrm { Q } } = n _ { \mathrm { p } } n _ { \mathrm { c } } B _ { \mathrm { t o k } } ,\tag{4}
$$

where $n _ { \mathrm { c } }$ is the number of merged features per patch. The payload relative to uncompressed feature transmission is

$$
\frac { B _ { \mathrm { Q } } } { B _ { \mathrm { r a w } } } = \frac { n _ { \mathrm { c } } B _ { \mathrm { t o k } } } { n _ { \mathrm { v } } d _ { \mathrm { v } } b _ { \mathrm { f } } } .\tag{5}
$$

The communication cost is therefore determined by the patch count $n _ { \mathrm { p } } .$ , the number of merged features $n _ { \mathrm { c } } .$ and the RVQ configuration through $B _ { \mathrm { t o k } }$

Let $R _ { \mathrm { u } }$ denote the effective uplink throughput in bits per second. The transmission latency is

$$
T _ { \mathrm { t x } } = { \frac { B _ { \mathrm { Q } } } { R _ { \mathrm { u } } } } .\tag{6}
$$

For a selected value of $n _ { \mathrm { c } }$ and a given RVQ configuration, the transmission latency follows from the image patch count and the uplink throughput. The end-to-end latency considered in our evaluation is

$$
T _ { \mathrm { e 2 e } } = T _ { \mathrm { d e v } } + T _ { \mathrm { t x } } + T _ { \mathrm { e d g e } } ,\tag{7}
$$

where $T _ { \mathrm { d e v } }$ includes on-device visual processing and compression, and $T _ { \mathrm { e d g e } }$ includes reconstruction and the remaining LMM computation. This decomposition is important because a smaller payload can require additional device computation, while image-level codecs move vision encoding to the server. We therefore report all three components rather than using payload size as a proxy for end-to-end latency.

## B. Design Objective

The goal of Q-TOFC is not solely to minimize reconstruction error with respect to the original image or visual features. Instead, the codec should preserve the information required by the downstream LMM to answer the user’s query. Let D denote the distribution of image–query–answer tuples, and let $\mathcal { C } _ { \phi } ( \cdot , \cdot )$ denote the query-conditioned compression and reconstruction pipeline parameterized by $\phi .$ The ideal task-oriented objective can be written as

$$
\begin{array} { r l } { \underset { \phi } { \operatorname* { m i n } } } & { \mathbb { E } _ { \mathcal { D } } [ \mathcal { L } _ { \mathrm { t a s k } } ( F _ { \theta } ( \mathcal { C } _ { \phi } ( f _ { \mathrm { v i s } } ( \pmb { v } _ { \mathrm { i } } ) , \pmb { q } ) , \pmb { q } ) , \pmb { a } ) ] } \\ { \mathrm { s . t . } } & { B _ { \mathrm { Q } } \leq B _ { 0 } , } \end{array}\tag{8}
$$

where $F _ { \theta }$ is the server-side LMM, q is the textual instruction, a is the target answer, and $B _ { 0 }$ is the communication budget. This formulation makes two requirements explicit. First, token reduction should be conditioned on the query because relevance is task dependent. Second, the reconstruction module should be optimized through the LMM loss rather than solely through feature-level distortion. The following section instantiates these two requirements with query-guided clustering and QECA.

## IV. METHOD

This section presents the proposed Q-TOFC framework and its three key components. We first describe the queryguided clustering module that reduces the number of visual features while biasing cluster-center selection toward queryrelevant evidence (Section IV-A), then introduce the RVQ codec (Section IV-B), followed by QECA (Section IV-C), and finally discuss the training strategy (Section IV-D).

## A. Query-Guided Token Reduction

Modern vision encoders produce hundreds of features per image patch, many of which lie close together in the feature space and carry redundant information. Following the feature-merging design of TOFC [19], we adopt density peaks clustering based on $K$ nearest neighbors (DPC-KNN) [40], a training-free method that adaptively partitions the $n _ { \mathrm { v } }$ features in each patch into $n _ { \mathrm { c } }$ clusters based on their distribution in the latent space. The DPC-KNN density–diversity criterion is inherited from this prior design. Our modification is to incorporate query relevance into the center-ranking score. Because the textual query is available before compression, we further introduce a lightweight query-aware scoring mechanism that biases the cluster-center selection toward task-relevant visual tokens.

For the n-th patch, the local density of the i-th feature is computed from its K nearest neighbors as

$$
\rho _ { n , i } = \exp \Bigl ( - \frac { 1 } { K } \sum _ { \substack { \pmb { x } _ { n , j } \in \mathrm { K N N } ( \pmb { x } _ { n , i } , K ) } } \| \pmb { x } _ { n , i } - \pmb { x } _ { n , j } \| ^ { 2 } \Bigr ) .\tag{9}
$$

To encourage diversity among cluster centers in the feature space, the minimum distance between feature i and any feature with higher density is also computed:

$$
\delta _ { n , i } = \operatorname* { m i n } _ { j : \rho _ { n , j } > \rho _ { n , i } } \| \pmb { x } _ { n , i } - \pmb { x } _ { n , j } \| .\tag{10}
$$

The $n _ { \mathrm { c } }$ features with the largest $\rho _ { n , i } \times \delta _ { n , i }$ are selected as cluster centers, and the remaining features are assigned to their nearest center. Average pooling within each cluster yields the merged features $\pmb { Y } \in \mathbb { R } ^ { n _ { \mathrm { p } } \times n _ { \mathrm { c } } \times d _ { \mathrm { v } } }$

To condition center selection on the query, we use the text tower paired with the vision encoder to embed the user instruction. Contrastively pretrained vision–language encoders such as CLIP and SigLIP align their image and text representations in a shared multimodal space [5], [6]. These representations therefore provide a natural basis for estimating text–visual relevance. Given the embedded text tokens $\pmb { T } \in \mathbb { R } ^ { N _ { \mathrm { t x t } } \times d _ { \mathrm { t x t } } }$ the scoring module computes a relevance score for each visual feature:

$$
r _ { n , i } = \operatorname* { m a x } _ { j } \sin \bigl ( f _ { \mathrm { i m g } } ( \pmb { x } _ { n , i } ) , \ f _ { \mathrm { t x t } } ( \pmb { t } _ { j } ) \bigr ) ,\tag{11}
$$

where $f _ { \mathrm { i m g } }$ and $f _ { \mathrm { t x t } }$ are lightweight projection heads that map the image and text features into a shared low-dimensional relevance space, and $\mathrm { s i m } ( \cdot , \cdot )$ denotes cosine similarity. The maximum over text tokens lets each visual token match the most relevant textual token in the query, which is useful for localized tasks such as OCR, counting, and object-centric reasoning. We normalize the relevance scores within each patch as

$$
\tilde { r } _ { n , i } = \frac { r _ { n , i } - \operatorname* { m i n } _ { j } r _ { n , j } } { \operatorname* { m a x } _ { j } r _ { n , j } - \operatorname* { m i n } _ { j } r _ { n , j } + \epsilon } .\tag{12}
$$

The clustering score is then adjusted as

$$
\hat { s } _ { n , i } = s _ { n , i } \cdot ( 1 + \alpha \tilde { r } _ { n , i } ) ,\tag{13}
$$

where $s _ { n , i } = \rho _ { n , i } \times \delta _ { n , i }$ is the original DPC-KNN score and α is a hyperparameter controlling the strength of text guidance. When $\alpha = 0$ , the clustering reduces to the standard DPC-KNN formulation. The query-aware score preserves the density– diversity criterion of DPC-KNN through $s _ { n , i }$ while biasing the selected cluster centers toward features with higher text relevance.

## B. Residual Vector Quantization Codec

After clustering, each merged feature vector $\boldsymbol { y } \in \mathbb { R } ^ { d _ { \mathrm { v } } }$ must be represented compactly. Q-TOFC uses the established RVQ formulation [20], [21] to represent each merged feature as a sequence of discrete codebook indices through a coarse-to-fine approximation. The discrete codec is combined with queryguided clustering and task-driven quantization compensation for task-oriented visual feature transmission.

Let $\{ C ^ { ( k ) } \} _ { k = 0 } ^ { L - 1 }$ be a set of L learnable codebooks, where $C ^ { ( k ) } \in \mathbb { R } ^ { M _ { k } \times d _ { v } }$ contains $M _ { k }$ entries at layer k. Given an input vector y, our cosine-similarity implementation proceeds iteratively:

$$
\begin{array} { r l } & { \boldsymbol r ^ { ( 0 ) } = \boldsymbol y , } \\ & { \boldsymbol q ^ { ( k ) } = \arg \underset { c \in { \cal C } ^ { ( k ) } } { \operatorname* { m a x } } \cos ( \boldsymbol r ^ { ( k ) } , c ) , \quad k = 0 , 1 , \dots , L - 1 , } \end{array}\tag{14}
$$

$$
\pmb { r } ^ { ( k + 1 ) } = \pmb { r } ^ { ( k ) } - \pmb { q } ^ { ( k ) } ,\tag{15}
$$

```latex
Algorithm 1 On-device Q-TOFC encoding
Require: Image $v _ { \mathrm { i } } ,$ text query ${ \mathbf { } } q ,$ number of merged features
per patch $n _ { \mathrm { c } } ,$ RVQ codebooks $\{ C ^ { ( k ) } \} _ { k = 0 } ^ { L - 1 }$
Ensure: Ordered RVQ index sequence
1: Extract visual features $\begin{array} { r } { \pmb { X } = f _ { \mathrm { v i s } } ( \pmb { v } _ { \mathrm { i } } ) } \end{array}$
2: Extract text features T from the frozen text encoder
3: for $n = 1 , \ldots , n _ { \mathrm { p } }$ do
4: Compute DPC-KNN density scores $\rho _ { n , i }$ and diversity
scores $\delta _ { n , i }$
5: Compute query relevance scores $r _ { n , i }$ using (11)
6: Select $n _ { \mathrm { c } }$ cluster centers by the adjusted score in (13)
7: Average-pool features within each cluster to obtain
$\{ \pmb { y } _ { n , m } \} _ { m = 1 } ^ { n _ { \mathrm { c } } }$
8: for $m = 1 , \ldots , n _ { \mathrm { c } }$ do
9: Quantize ${ \pmb y } _ { n , m }$ through L RVQ layers using (14)–
(15)
10: Append the L fixed-width RVQ indices to the
output sequence
11: end for
12: end for
13: return ordered RVQ index sequence
```

where $\pmb { r } ^ { ( k ) }$ is the residual at layer k, and $\pmb q ^ { ( k ) }$ is the selected codeword. The reconstructed vector is the sum of all codewords:

$$
\hat { \pmb { y } } _ { \mathrm { r v q } } = \sum _ { k = 0 } ^ { L - 1 } { \pmb { q } } ^ { ( k ) } .\tag{16}
$$

The index at layer k requires $\lceil \log _ { 2 } M _ { k } \rceil$ bits. Summing the index lengths across all layers gives $B _ { \mathrm { t o k } }$ in (3).

For efficient codebook lookup, (14) is implemented as batched matrix multiplication after vector normalization. This choice emphasizes angular alignment between normalized feature and codeword representations and is consistent with the cosine-similarity formulation in (14).

During training, the non-differentiable codeword selection operation is bypassed with the straight-through estimator (STE) [41]: gradients are copied from the output directly to the input, enabling end-to-end backpropagation through the quantization bottleneck. The codebook at each RVQ layer is updated using exponential moving averages of the assignment counts and the residual vectors assigned to each entry [20], [31].

## C. Quantization Error Compensation Adapter

Multi-layer RVQ inevitably introduces quantization error: the sum of L codewords only approximates the original continuous vector. Because this approximation error can degrade downstream inference, we propose QECA to recover taskrelevant information lost during discretization.

As illustrated in Fig. 2, QECA consists of two components:

1) Per-layer projections: each codeword $\pmb q ^ { ( k ) }$ is independently projected through a lightweight network $g _ { k } ( \cdot ) =$ $\mathrm { G E L U } ( W _ { k } \mathbf { q } ^ { ( k ) } )$ that maps the $d _ { \mathrm { v } }$ -dimensional codeword to a hidden representation of dimension $d _ { \mathrm { h } }$

![](images/426a72c72491d6701b5b77ccb93a5043c6e374d3589b918a942eb7ecd9eb2b31.jpg)  
Fig. 2. Architecture of the quantization error compensation adapter. The projected RVQ-layer codewords are concatenated with $\hat { \pmb { y } } _ { \mathrm { r v q } }$ to predict a residual correction ∆, which is added to $\hat { \pmb { y } } _ { \mathrm { r v q } }$ to obtain the refined feature.

2) Shared compensation MLP: the projected representations from all layers are concatenated with the STEquantized vector $\hat { \pmb { y } } _ { \mathrm { r v q } }$ and fed into a two-layer MLP:

$$
\mathbf { \Delta } \Delta = \mathrm { M L P } \Big ( \big [ g _ { 0 } ( \pmb { q } ^ { ( 0 ) } ) ; \ \cdots ; \ g _ { L - 1 } ( \pmb { q } ^ { ( L - 1 ) } ) ; \ \hat { \pmb { y } } _ { \mathrm { r v q } } \big ] \Big ) ,\tag{17}
$$

where $[ \cdot ; \cdot ]$ denotes concatenation along the feature dimension, and the MLP consists of a linear layer → GELU → linear layer.

The final reconstructed feature is

$$
\hat { \pmb { y } } = \hat { \pmb { y } } _ { \mathrm { r v q } } + \pmb { \Delta } .\tag{18}
$$

The last linear layer of the compensation MLP is initialized with zero weights and biases, ensuring that QECA initially produces no residual correction $( \mathrm { i . e . , } \ \Delta = \mathbf { 0 } )$ . This initialization prevents the adapter from perturbing the initial quantized features and improves stability during early training.

QECA is further optimized through the next-token prediction loss together with the trainable LoRA adapters in both the vision encoder and the LLM (see Section IV-D). Therefore, its predicted residual is guided by downstream answer generation rather than by feature-level reconstruction objectives alone. This task-driven training enables QECA to compensate for RVQ-induced distortion in a manner aligned with the downstream inference objective.

## D. Training Strategy

The overall training proceeds in two stages.

Stage I — RVQ and QECA pretraining. The RVQ codebooks and QECA are first pretrained on visual features extracted from the training images, without involving the LLM or textual instructions. The training objective combines cosine similarity loss and L2 normalization loss between the reconstructed and original features, along with a commitment loss that encourages the input vectors to stay close to their assigned codewords:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { v q } } = \lambda _ { 1 } \big ( 1 - \cos ( \hat { y } , y ) \big ) + \lambda _ { 2 } \| \hat { y } _ { \mathrm { n o r m } } - y _ { \mathrm { n o r m } } \| ^ { 2 } + \lambda _ { 3 } \mathcal { L } _ { \mathrm { c o m m i t } } , } \end{array}\tag{19}
$$

where $\begin{array} { r } { \mathcal { L } _ { \mathrm { c o m m i t } } \ = \ \frac { 1 } { L } \sum _ { k = 0 } ^ { L - 1 } \| \pmb { r } ^ { ( k ) } - \mathrm { s g } [ \pmb { q } ^ { ( k ) } ] \| ^ { 2 } } \end{array}$ is the commitment loss, sg[·] denotes the stop-gradient operator, and $\lambda _ { 1 } , \lambda _ { 2 } , \lambda _ { 3 }$ are weighting coefficients.

Stage II — End-to-end task fine-tuning. After featurelevel pretraining, we integrate the RVQ codec, multimodal projector, query-aware scoring module, and QECA with the pretrained vision encoder and LLM. Low-rank adaptation (LoRA) adapters are attached to both the vision encoder and the LLM [42]. The visual-side adapters allow the feature distribution to adapt to the discrete RVQ bottleneck under the joint task and VQ objectives, while retaining the pretrained vision-encoder weights. The trainable components are jointly optimized on an instruction-tuning dataset using the standard autoregressive language modeling loss:

$$
\mathcal { L } _ { \mathrm { l m } } = - \sum _ { t = 1 } ^ { T } \log p _ { \theta } ( a _ { t } \mid \boldsymbol { x } _ { \mathrm { v i s } } , \boldsymbol { x } _ { \mathrm { t x t } } , a _ { < t } ) ,\tag{20}
$$

where $a _ { t }$ is the t-th token of the target answer, $\pmb { x } _ { \mathrm { v i s } }$ and ${ \bf { \mathcal { x } } } _ { \mathrm { { t x t } } }$ denote the visual and textual token sequences, respectively, and θ comprises the multimodal projector, query-aware scoring module, QECA, and the LoRA parameters in both the vision encoder and the LLM. The pretrained base weights of both the vision encoder and the LLM remain frozen, while their attached LoRA parameters are updated. The RVQ codebooks continue to be updated by EMA during this stage.

The VQ loss $\mathcal { L } _ { \mathrm { v q } }$ from (19) is added as a regularization term to prevent the reconstructed features from drifting too far from the originals:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { l m } } + \beta \cdot \mathcal { L } _ { \mathrm { v q } } . } \end{array}\tag{21}
$$

## E. Complexity Analysis

On the device, the additional operations introduced by $\mathrm { Q } \mathrm { - }$ TOFC are query-aware scoring, DPC-KNN feature merging, and RVQ index generation. Computing the visual–text similarity matrix scales as

$$
\mathcal { O } ( n _ { \mathrm { p } } n _ { \mathrm { v } } N _ { \mathrm { t x t } } d _ { \mathrm { s } } ) .\tag{22}
$$

where $d _ { \mathrm { s } }$ is the dimension of the shared relevance space. DPC-KNN computes pairwise feature distances within each patch,

giving $\mathcal { O } ( n _ { \mathrm { p } } n _ { \mathrm { v } } ^ { 2 } d _ { \mathrm { v } } )$ , while RVQ nearest-codeword search scales as

$$
{ \mathcal O } \left( n _ { \mathrm { p } } n _ { \mathrm { c } } d _ { \mathrm { v } } \sum _ { k = 0 } ^ { L - 1 } M _ { k } \right) ,\tag{23}
$$

where $M _ { k }$ is the size of the k-th codebook. On the server, codebook lookup and summation cost $\mathcal { O } ( n _ { \mathrm { p } } n _ { \mathrm { c } } L d _ { \mathrm { v } } )$ , followed by a QECA operation whose cost is linear in the number of merged features. The measured device-side encoding and endto-end costs of these operations are reported in Section V.

## V. EXPERIMENTAL RESULTS

## A. Experimental Setup

Backbone model. We use LLaVA-OneVision-7B [8] as the backbone LMM and SigLIP-SO400M [6] as its vision encoder. Each 384 × 384 image patch yields $n _ { \mathrm { v } } = 7 2 9$ visual features of dimension $d _ { \mathrm { v } } ~ = ~ 1 1 5 2$ . When represented in FP16, i.e., $b _ { \mathrm { f } } ~ = ~ 1 6$ , the uncompressed features of one patch require approximately 1.60 MiB.

Q-TOFC configuration. Unless otherwise stated, we set the number of RVQ layers to $L = 8 ,$ the first-stage codebook size to $M _ { 0 } = 1 6 { , } 3 8 4$ , and $M _ { k } = 4 { , } 0 9 6$ for the subsequent layers. The RVQ indices therefore require $B _ { \mathrm { t o k } } = 1 4 + 7 \times 1 2 = 9 8$ bits per merged feature. We vary $n _ { \mathrm { c } }$ to obtain the rate– performance curves. The latency comparison and RVQ-depth ablation use $n _ { \mathrm { c } } ~ = ~ 6 4$ . Because LLaVA-OneVision uses a variable number of image patches, we report average perrequest communication cost in KiB for each benchmark. The QECA hidden dimension is $d _ { \mathrm { h } } ~ = ~ 2 5 6$ . The text guidance strength is $\alpha = 0 . 5$ by default, and the effect of disabling query-aware scoring is studied in the ablation experiments.

Training. Both stages use the LLaVA-1.5 second-stage instruction-tuning dataset [43]. In Stage I, only the images from this dataset are used: visual features are extracted to pretrain the RVQ codebooks and QECA without involving the LMM or textual instructions. In Stage II, all 665K instructiontuning samples are used for task-driven fine-tuning. The RVQ codebooks are updated by EMA, while the multimodal projector, query-aware scoring module, QECA, and LoRA adapters in both the SigLIP vision encoder and the LLM are optimized by backpropagation. The LoRA rank is set to 64 for both the SigLIP vision encoder and the LLM, and Stage II training is performed for one epoch. We set $( \lambda _ { 1 } , \lambda _ { 2 } , \lambda _ { 3 } ) = ( 1 . 5 , 0 . 5 , 0 . 0 5 )$ in (19) and $\beta = 0 . 1 5$ in (21).

Evaluation benchmarks. Following the benchmark suite in [19], we evaluate on seven multimodal benchmarks: Real-WorldQA, MME [44], AI2D [45], MMBench [46], MMStar [47], ScienceQA [48], and MMMU [49]. All evaluations are conducted using the VLMEvalKit framework [50] to ensure a fair comparison. These benchmarks cover complementary forms of visual understanding rather than a single narrow task. RealWorldQA evaluates question answering grounded in realworld images, while MME covers a broad range of multimodal perception abilities. MMBench uses multiple-choice questions to assess diverse multimodal capabilities, whereas MMStar emphasizes samples that require visual information across six core capabilities. AI2D evaluates question answering over scientific diagrams, ScienceQA contains multimodal multiplechoice science questions, and MMMU requires college-level reasoning across multiple disciplines and heterogeneous visual inputs. The seven-benchmark average therefore aggregates performance across real-world scenes, diagrams, scientific questions, and knowledge-intensive reasoning tasks.

Baselines. We compare Q-TOFC against the following compression methods:

• TOFC [19]: Clustering-based feature merging with learnable entropy coding and the closest feature-level baseline.

• JPEG [14]: Standard image compression at varying quality levels.

• ELIC [18]: Neural image compression with autoregressive context modeling.

For image-level baselines, the compressed image is decoded and re-encoded by the vision encoder at the edge server. The resulting visual features are then merged using DPC-KNN to produce 128 merged features per patch, thereby controlling the visual-token workload of subsequent LLM inference. All methods use the same LMM backbone and evaluation framework, and the runtime of each operation is included on the side where it is executed.

Score aggregation and communication accounting. For each benchmark, we first compute the score of a compressed method using the benchmark-specific evaluation metric implemented in VLMEvalKit. The score is then normalized by the score of the uncompressed LLaVA-OneVision-7B backbone on the same benchmark. The average score reported in Fig. 3 is the arithmetic mean of these normalized scores across the seven benchmarks. This normalization prevents a benchmark with a numerically larger metric range from dominating the average.

Communication cost is reported as the visual payload transmitted per request in KiB. The textual query is excluded because it is identical across all compared methods. To obtain the rate–performance curves, we vary the number of merged features $n _ { \mathrm { c } }$ for Q-TOFC while keeping its RVQ configuration fixed. For TOFC, we train separate models with different rate–distortion loss weights. The ELIC operating points use the official pretrained checkpoints corresponding to different rate–distortion settings, while the JPEG operating points are generated by varying the quality factor. All resulting outputs are evaluated using the same benchmark scripts.

## B. Rate–Performance Trade-off

Fig. 3 shows the communication–performance trade-off obtained by sweeping the compression settings of each method. Q-TOFC occupies the low-communication region of the curve: even at 3.24 KiB it preserves 90.95% of the backbone’s average normalized score, and at 6.49 KiB it reaches 92.21%. The latter operating point essentially matches the 92.20% score of TOFC at 13.98 KiB while requiring only 46.4% of its payload. Across the plotted operating points, Q-TOFC also lies above the image-level JPEG and ELIC baselines in the low-communication region. Among the image-level baselines, ELIC consistently outperforms JPEG, demonstrating the benefit of learned neural image compression over conventional image coding under the evaluated settings. However, ELIC remains below the feature-level methods at comparable communication costs, and JPEG does not reach their higherperformance range within the displayed payloads. This comparison indicates that improving the image codec narrows the performance gap, but directly transmitting task-oriented visual features remains more effective for downstream LMM inference under limited communication budgets. Increasing the number of merged features steadily improves Q-TOFC performance, confirming that retaining more visual evidence benefits multimodal reasoning. The curve becomes flatter at the higherrate operating points, however, indicating diminishing returns from further increasing the number of merged features under the fixed 98-bit representation. Thus, varying the number of merged features provides a direct control knob: smaller values favor communication efficiency, whereas larger values recover additional task performance.

![](images/6d3b105171b01d59928db3da5e8f0241fe7c54b59249aca257d40b052c133628.jpg)  
Fig. 3. Communication–performance trade-off averaged over seven benchmarks with LLaVA-OneVision-7B. Each benchmark score is normalized by the corresponding uncompressed LLaVA-OneVision-7B score before averaging.

Fig. 4 presents the communication–performance trade-off on RealWorldQA. At a comparable accuracy level, Q-TOFC reduces the visual payload by 23.7% relative to TOFC. This result is consistent with the design of preserving a larger set of compact, query-relevant merged features instead of aggressively reducing the token count. The larger separation from JPEG is consistent with the sensitivity of fine edges and text strokes to pixel-domain compression. ELIC narrows this gap through learned image compression, but both imagelevel baselines remain below the feature-level methods on this benchmark, indicating the value of feature-space transmission for preserving task-relevant visual information. The qualitative examples in Section V-F examine this behavior for localized sign evidence at the cluster-center level.

Fig. 5 shows the communication–performance results on

![](images/7390d94ac81b1ffdb763b4fc1105d90c2a3f22d506d40e4664a8ea15f7d4eb7c.jpg)  
Fig. 4. Communication–performance trade-off on RealWorldQA with LLaVA-OneVision-7B. Communication cost is the average visual payload transmitted per request, and the dashed line indicates the uncompressed-backbone score.

![](images/0f823772d62a143b8060a3ccfba1bf3e6285322670958576dba1c64ed1d93e97.jpg)  
Fig. 5. Communication–performance trade-off on MME with LLaVA-OneVision-7B. Communication cost is the average visual payload transmitted per request, and the dashed line indicates the uncompressed-backbone score.

MME. Q-TOFC rises rapidly in the low-payload region and approaches the uncompressed backbone performance with only a few KiB per request. At its 5.05 KiB operating point, Q-TOFC surpasses the highest plotted scores of TOFC, ELIC, and JPEG while reducing the communication payload by 55.8%, 51.9%, and 66.8%, respectively. Among the baselines, TOFC achieves a better communication–performance tradeoff than ELIC and JPEG, demonstrating the advantage of directly coding task-oriented visual features over transmitting compressed images. Together with the RealWorldQA results, the MME curve shows that Q-TOFC maintains its communication–performance advantage across benchmarks with different visual information requirements.

<table><tr><td>Ours ELIC</td><td>Server-only</td><td> Data transmission</td><td>-0- MME score</td></tr><tr><td>TOFC JPEG</td><td> On-device computation</td><td> Server-side computation</td><td></td></tr></table>

![](images/fa1f565176125925e3a98e38766919f32bbd0682c47b1dd6735928d834ac414e.jpg)  
Fig. 6. End-to-end latency breakdown and MME scores with LLaVA-OneVision-7B at uplink rates of 25, 50, and 100 KB/s. Stacked bars show on-device computation, data transmission, and server-side computation, while markers indicate the corresponding MME scores. Q-TOFC and TOFC use $n _ { \mathrm { c } } = 6 4$ and $n _ { \mathrm { c } } = 1 6$ , respectively; JPEG uses $q = 2 0 ,$ , and ELIC uses λ = 0.15.

![](images/3d95acf0b3bb65a77d9318f4a7c2ff35b8b5f052f2dc5dfac00b264c2726f992.jpg)  
Fig. 7. Communication–performance trade-off averaged over seven benchmarks with LLaVA-1.5-7B. Each benchmark score is normalized by the corresponding uncompressed LLaVA-1.5-7B score before averaging.

## C. Latency Analysis

Fig. 6 breaks down the end-to-end latency under uplink rates of 25, 50, and 100 KB/s. For the latency comparison, we select operating points that yield similar MME scores across the compressed methods. The transmission latency is computed from the visual payload and the assumed uplink rate, while the device-side and server-side components are measured computation times. The device-side measurements use an NVIDIA Jetson AGX Orin, and the edge-side measurements use NVIDIA RTX 4090 hardware. All computation and end-to-end latency results are averaged over the evaluation requests.

Q-TOFC and TOFC have comparable device-side processing times, while Q-TOFC transmits a smaller payload and therefore requires substantially less transmission time. At 25 KB/s on MME, Q-TOFC lowers the end-to-end latency by 29.0%, 67.6%, and 54.2% relative to TOFC, ELIC, and JPEG, respectively. The comparison includes all method-specific processing operations: JPEG and ELIC perform image decoding and visual feature extraction at the edge server, whereas TOFC and Q-TOFC compress intermediate visual features on the device and reconstruct them at the server. All these operations are included in the reported end-to-end latency. For a fixed Q-TOFC configuration, each image patch produces the same number of RVQ index bits. Once the patch count is known, the request payload can therefore be determined before transmission, which may simplify uplink resource allocation and latency-aware scheduling in practical deployments. As the uplink rate increases, transmission accounts for a smaller fraction of the total latency, and the latency gap between Q-TOFC and TOFC consequently narrows. Nevertheless, Q-TOFC maintains the lowest end-to-end latency among the compressed methods across all evaluated uplink rates. The score markers in Fig. 6 confirm that this latency advantage is achieved while maintaining similar MME performance.

## D. Extension to LLaVA-1.5

To examine whether Q-TOFC generalizes beyond the primary LLaVA-OneVision backbone, we further evaluate it with LLaVA-1.5-7B [43], which uses a CLIP ViT-L/14-336 vision encoder [5]. Each input image is resized and padded to $3 3 6 \times 3 3 6$ pixels and processed as a single image patch, yielding $n _ { \mathrm { v } } = 5 7 6$ visual features of dimension $d _ { \mathrm { v } } = 1 0 2 4$ We vary the number of merged features from $n _ { \mathrm { c } } ~ = ~ 6 4 ~ \mathrm { t o }$

<table><tr><td>Ours ELIC</td><td>Server-only</td><td> Data transmission</td><td>-0- MME score</td></tr><tr><td>TOFC JPEG</td><td> On-device computation</td><td> Server-side computation</td><td></td></tr></table>

![](images/e5b33c08fcf93e6b2b0946df9b7f2c59a38953211076f235b79d390d3b4852a7.jpg)  
Fig. 8. End-to-end latency breakdown and MME scores with LLaVA-1.5-7B at uplink rates of 25, 50, and 100 KB/s. Stacked bars show on-device computation, data transmission, and server-side computation, while markers indicate the corresponding MME scores. Q-TOFC and TOFC use $n _ { \mathrm { c } } = 6 4$ and $n _ { \mathrm { c } } = 3 2 ,$ respectively; JPEG uses $q = 2 0 ,$ , and ELIC uses λ = 0.32.

256 and train a separate Q-TOFC instance using the same two-stage procedure described in Section V-A. The remaining compression and evaluation settings are unchanged. For the average score, each benchmark is normalized by the corresponding uncompressed LLaVA-1.5 result.

Fig. 7 shows that the communication advantage of Q-TOFC persists with the CLIP-based backbone. At comparable average normalized performance, Q-TOFC reduces the payload by 45.4%, 60.0%, and 81.6% relative to the closest-performance operating points of TOFC, ELIC, and JPEG, respectively. At the higher-performance operating points, Q-TOFC requires approximately half the payload of TOFC while maintaining similar performance. The absolute payloads are lower than those with LLaVA-OneVision because LLaVA-1.5 processes a single fixed-resolution image patch rather than a variable number of patches.

Fig. 8 reports the MME latency results with LLaVA-1.5-7B. Q-TOFC achieves the lowest end-to-end latency among the compressed methods at all three uplink rates while maintaining comparable MME performance. At 25 KB/s, Q-TOFC reduces the end-to-end latency by 48.2%, 81.3%, and 60.3% relative to TOFC, ELIC, and JPEG, respectively. Q-TOFC also maintains the lowest latency at the other evaluated uplink rates, while the score markers show that the compared methods achieve similar MME performance. Together with Fig. 7, these results show that Q-TOFC retains its communication and latency advantages when the vision encoder changes from SigLIP to CLIP and the image representation changes from variablepatch to fixed-resolution input.

## E. Ablation Studies

Fig. 9 evaluates the contributions of query-aware scoring and QECA under different merged-feature counts. The full configuration consistently achieves the highest normalized score across the entire communication–performance curve, indicating that the gains are not limited to a single operating point. Removing either query-aware scoring or QECA shifts the curve downward, while disabling both modules produces the largest performance degradation. The separation is more visible in the low-payload region and becomes smaller as more merged features are retained. This trend suggests that both modules are particularly useful when the compressed representation has limited capacity.

![](images/3a6513a39a046e2c7b2848cfd419d5d60031636832b97e64b8a0eba36b1893b4.jpg)  
Fig. 9. Ablation of query-aware token scoring and QECA with LLaVA-OneVision. Scores are averaged over seven benchmarks after normalization by the corresponding uncompressed-backbone scores. At each operating point, all variants use the same RVQ configuration and merged-feature count.

The two modules improve different stages of the compression pipeline. Query-aware scoring affects feature selection before quantization by increasing the likelihood that queryrelevant visual features are retained during merging. QECA operates after RVQ reconstruction and refines the quantized representations before they are processed by the LMM. Removing both modules therefore affects both feature selection and feature reconstruction. The consistently stronger performance of the full configuration supports using query-aware scoring and QECA together in Q-TOFC.

![](images/85a42360656cbef547286225b49fe51328cfe621ab1040a41265ebae1d4f1838.jpg)  
Fig. 10. Qualitative examples of query-guided cluster-center selection on RealWorldQA. With identical visual features and the same total number of selected centers, query guidance increases the number of centers in the displayed evidence region from three to five (left) and from two to four (right). The marker denote visual-token locations projected onto the image grid rather than pixel-level attention.

TABLE I  
EFFECT OF RVQ DEPTH UNDER THE 64-FEATURE SETTING.
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>L</td><td rowspan=1 colspan=1> $\overline { { B _ { \mathrm { t o k } } } }$ </td><td rowspan=1 colspan=1>CB storage (MB)</td><td rowspan=1 colspan=1>MME</td><td rowspan=1 colspan=1>MMB</td><td rowspan=1 colspan=1>MMStar</td><td rowspan=1 colspan=1>AI2D</td><td rowspan=1 colspan=1>RWQA</td><td rowspan=1 colspan=1>SQA</td><td rowspan=1 colspan=1>MMMU</td><td rowspan=1 colspan=1>Avg. norm</td></tr><tr><td rowspan=1 colspan=1>Backbone</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1590.10</td><td rowspan=1 colspan=1>82.13</td><td rowspan=1 colspan=1>61.87</td><td rowspan=1 colspan=1>82.70</td><td rowspan=1 colspan=1>69.02</td><td rowspan=1 colspan=1>95.34</td><td rowspan=1 colspan=1>48.00</td><td rowspan=1 colspan=1>100.00%</td></tr><tr><td rowspan=1 colspan=1>Q-TOFC</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>26</td><td rowspan=1 colspan=1>47.19</td><td rowspan=1 colspan=1>1498.72</td><td rowspan=1 colspan=1>73.53</td><td rowspan=1 colspan=1>46.07</td><td rowspan=1 colspan=1>72.18</td><td rowspan=1 colspan=1>61.57</td><td rowspan=1 colspan=1>80.54</td><td rowspan=1 colspan=1>44.00</td><td rowspan=1 colspan=1>87.27%</td></tr><tr><td rowspan=1 colspan=1>Q-TOFC</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>66.06</td><td rowspan=1 colspan=1>1509.10</td><td rowspan=1 colspan=1>76.54</td><td rowspan=1 colspan=1>47.93</td><td rowspan=1 colspan=1>72.11</td><td rowspan=1 colspan=1>63.27</td><td rowspan=1 colspan=1>82.50</td><td rowspan=1 colspan=1>44.67</td><td rowspan=1 colspan=1>89.15%</td></tr><tr><td rowspan=1 colspan=1>Q-TOFC</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>74</td><td rowspan=1 colspan=1>84.93</td><td rowspan=1 colspan=1>1527.44</td><td rowspan=1 colspan=1>76.97</td><td rowspan=1 colspan=1>48.80</td><td rowspan=1 colspan=1>73.21</td><td rowspan=1 colspan=1>63.79</td><td rowspan=1 colspan=1>83.40</td><td rowspan=1 colspan=1>45.22</td><td rowspan=1 colspan=1>90.18%</td></tr><tr><td rowspan=1 colspan=1>Q-TOFC</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>98</td><td rowspan=1 colspan=1>103.81</td><td rowspan=1 colspan=1>1557.90</td><td rowspan=1 colspan=1>76.89</td><td rowspan=1 colspan=1>49.13</td><td rowspan=1 colspan=1>73.48</td><td rowspan=1 colspan=1>63.40</td><td rowspan=1 colspan=1>86.17</td><td rowspan=1 colspan=1>45.40</td><td rowspan=1 colspan=1>90.95%</td></tr></table>

Table I further studies the effect of RVQ depth while fixing the number of merged features per patch to 64 and keeping both query-guided clustering and QECA enabled. For an Llayer RVQ, each merged feature requires $B _ { \mathrm { t o k } } = 1 4 + ( L -$ 1)×12 index bits under our codebook configuration. Increasing RVQ depth therefore raises both the number of index bits per merged feature and the device-side codebook storage, which grows from 47.19 MB at L = 2 to 103.81 MB at L = 8 when the codebooks are stored in FP16. Increasing the number of RVQ codebooks improves the average normalized score from 87.27% at L = 2 to 90.95% at L = 8, showing that residual refinement is important for preserving task-relevant visual features. Most of this improvement is obtained by the first six layers: increasing the depth from L = 6 to L = 8 adds 24 bits per merged feature and 18.88 MB of codebook storage, while improving the average normalized score by 0.77 percentage points. This diminishing return reveals a three-way trade-off among communication cost, codebook memory, and task performance. We use L = 8 as the default high-accuracy configuration, while the L = 6 result provides a lower-rate, lower-memory alternative.

## F. Qualitative Analysis of Query-Guided Center Selection

To complement the aggregate ablation results, Fig. 10 visualizes how query guidance changes cluster-center selection in two RealWorldQA examples. For each example, the query-agnostic and query-guided variants use the same visual features and select the same total number of cluster centers. The only difference is whether the normalized text–visual relevance score is applied to the DPC-KNN center-ranking criterion. The selected visual-token locations are projected onto the input-image grid. For readability, the figure enlarges the evidence region required by the question and displays only the centers within the same crop for both variants. These markers indicate projected visual-token locations rather than pixel-level attention.

In the first example, the question asks, “In which direction is the one-way sign in this scene facing?” Query guidance increases the number of selected centers in the displayed sign region from three to five, thereby allocating more representation capacity to evidence relevant to recognizing its direction. Without query guidance, the model answers “left.” With queryguided center selection, the answer changes to “right,” which matches the ground truth. In the second example, the question asks, “What is the speed limit on this road?” The number of centers in the displayed speed-limit-sign region increases from two to four. Correspondingly, the query-agnostic variant answers “25,” whereas the query-guided variant answers “20,” which is correct. Since the total number of selected centers is unchanged in both comparisons, these changes represent a reallocation toward the localized evidence identified by the query rather than an increase in the total number of centers. Although qualitative, the two cases connect this reallocation with corrected downstream predictions and illustrate the mechanism underlying the quantitative gains observed in Fig. 9.

## VI. CONCLUSION

This paper presented Q-TOFC, a task-oriented visual feature compression framework for device–edge collaborative multimodal inference. Q-TOFC uses query-aware feature scoring to guide cluster-center selection toward visual evidence relevant to the user’s textual query. It employs RVQ to represent each merged visual feature as a compact sequence of codebook indices, allowing more merged features to be retained at low communication cost. To mitigate the distortion introduced by multi-layer quantization, QECA refines the reconstructed features under the joint supervision of the task objective and feature-level VQ regularization. Experiments on seven multimodal benchmarks with LLaVA-OneVision showed that Q-TOFC reduces the visual payload by 53.6% relative to TOFC while maintaining comparable average normalized task performance. The lower payload further translates into reduced end-to-end latency under bandwidth-constrained uplinks. Additional experiments with LLaVA-1.5 demonstrate that these communication and latency advantages persist with a different vision encoder and image-patching scheme. Ablation studies confirm that query-aware feature scoring and QECA address complementary sources of information loss, while the RVQdepth study characterizes the trade-off among communication cost, codebook storage, and task performance. Future work will investigate dynamic selection of both RVQ depth and the number of merged features according to channel conditions.

## REFERENCES

[1] S. Yin, C. Fu, S. Zhao, K. Li, X. Sun, T. Xu, and E. Chen, “A survey on multimodal large language models,” Nat. Sci. Rev., vol. 11, no. 12, 2024, art. no. nwae403.

[2] D. Driess et al., “PaLM-E: An embodied multimodal language model,” Sci. Robot., vol. 8, no. 84, 2023, art. no. eadh2585.

[3] B. Li, Y. Wang, J. Mao, B. Ivanovic, S. Veer, K. Leung, and M. Pavone, “Driving everywhere with large language model policy adaptation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 14 948–14 957.

[4] L. Wang, C. Ma, X. Feng, Z. Zhang, H. Yang, J. Zhang, Z. Chen, J. Tang, X. Chen, Y. Lin, W. X. Zhao, Z. Wei, and J. Wen, “A survey on large language model based autonomous agents,” Front. Comput. Sci., vol. 18, no. 6, 2024, art. no. 186345.

[5] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, “Learning transferable visual models from natural language supervision,” in Proc. 38th Int. Conf. Mach. Learn. (ICML), vol. 139, 2021, pp. 8748–8763.

[6] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer, “Sigmoid Loss for language image pre-training,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023, pp. 11 975–11 986.

[7] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” in Adv. Neural Inf. Process. Syst., vol. 36, 2023, pp. 34 892–34 916.

[8] B. Li, Y. Zhang, D. Guo, R. Zhang, F. Li, H. Zhang, K. Zhang, P. Zhang, Y. Li, Z. Liu, and C. Li, “LLaVA-OneVision: Easy visual task transfer,” arXiv preprint arXiv:2408.03326, 2024.

[9] S. Bai et al., “Qwen2.5-VL technical report,” arXiv preprint arXiv:2502.13923, 2025.

[10] Y. Shi, K. Yang, T. Jiang, J. Zhang, and K. B. Letaief, “Communicationefficient edge AI: Algorithms and systems,” IEEE Commun. Surveys Tuts., vol. 22, no. 4, pp. 2167–2191, 2020.

[11] J. Shao and J. Zhang, “Communication-computation trade-off in resource-constrained edge inference,” IEEE Commun. Mag., vol. 58, no. 12, pp. 20–26, 2020.

[12] J. Shao and X. Li, “AI flow at the network edge,” IEEE Netw., vol. 40, no. 1, pp. 330–336, 2026.

[13] H. An, W. Hu, S. Huang, S. Huang, R. Li, Y. Liang, J. Shao, Y. Song, Z. Wang, C. Yuan, C. Zhang, H. Zhang, W. Zhuang, and X. Li, “AI flow: Perspectives, scenarios, and approaches,” Vicinagearth, vol. 3, no. 1, 2026, art. no. 1.

[14] G. K. Wallace, “The JPEG still picture compression standard,” Commun. ACM, vol. 34, no. 4, pp. 30–44, 1991.

[15] J. Balle, V. Laparra, and E. P. Simoncelli, “End-to-end optimized image´ compression,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2017.

[16] J. Balle, D. Minnen, S. Singh, S. J. Hwang, and N. Johnston, “Variational´ image compression with a scale hyperprior,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2018.

[17] Z. Cheng, H. Sun, M. Takeuchi, and J. Katto, “Learned image compression with discretized Gaussian mixture likelihoods and attention modules,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, pp. 7939–7948.

[18] D. He, Z. Yang, W. Peng, R. Ma, H. Qin, and Y. Wang, “ELIC: Efficient learned image compression with unevenly grouped spacechannel contextual adaptive coding,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022, pp. 5708–5717.

[19] C. Yuan, Z. Liu, J. Lv, J. Shao, Y. Jiang, J. Zhang, and X. Li, “Taskoriented feature compression for multimodal understanding via deviceedge co-inference,” IEEE Trans. Mobile Comput., vol. 25, no. 4, pp. 4762–4775, 2026.

[20] N. Zeghidour, A. Luebs, A. Omran, J. Skoglund, and M. Tagliasacchi, “SoundStream: An end-to-end neural audio codec,” IEEE/ACM Trans. Audio, Speech, Lang. Process., vol. 30, pp. 495–507, 2022.

[21] A. Defossez, J. Copet, G. Synnaeve, and Y. Adi, “High fidelity neural´ audio compression,” arXiv preprint arXiv:2210.13438, 2022.

[22] N. Tishby, F. C. Pereira, and W. Bialek, “The information bottleneck method,” in Proc. 37th Annu. Allerton Conf. Commun., Control, Comput., 1999, pp. 368–377.

[23] J. Shao and J. Zhang, “BottleNet++: An end-to-end approach for feature compression in device-edge co-inference systems,” in Proc. IEEE Int. Conf. Commun. Workshops (ICC Workshops), 2020, pp. 1–6.

[24] J. Shao, Y. Mao, and J. Zhang, “Task-oriented communication for multidevice cooperative edge inference,” IEEE Trans. Wireless Commun., vol. 22, no. 1, pp. 73–87, 2023.

[25] S. Xie, S. Ma, M. Ding, Y. Shi, M. Tang, and Y. Wu, “Robust information bottleneck for task-oriented communication with digital modulation,” IEEE J. Sel. Areas Commun., vol. 41, no. 8, pp. 2577–2591, 2023.

[26] J. Shao, X. Zhang, and J. Zhang, “Task-oriented communication for edge video analytics,” IEEE Trans. Wireless Commun., vol. 23, no. 5, pp. 4141–4154, 2024.

[27] J. Liu, C. Yuan, L. He, J. Zhang, and J. Shao, “Task-oriented communication for human action understanding via edge-cloud co-inference,” arXiv preprint arXiv:2605.07354, 2026.

[28] A. Skodras, C. Christopoulos, and T. Ebrahimi, “The JPEG 2000 still image compression standard,” IEEE Signal Process. Mag., vol. 18, no. 5, pp. 36–58, 2001.

[29] D. Minnen, J. Balle, and G. D. Toderici, “Joint autoregressive and´ hierarchical priors for learned image compression,” in Adv. Neural Inf. Process. Syst., vol. 31, 2018, pp. 10 794–10 803.

[30] D. He, Y. Zheng, B. Sun, Y. Wang, and H. Qin, “Checkerboard context model for efficient learned image compression,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2021, pp. 14 766–14 775.

[31] A. van den Oord, O. Vinyals, and K. Kavukcuoglu, “Neural discrete representation learning,” in Adv. Neural Inf. Process. Syst., vol. 30, 2017, pp. 6306–6315.

[32] P. Esser, R. Rombach, and B. Ommer, “Taming transformers for highresolution image synthesis,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2021, pp. 12 873–12 883.

[33] L. Chen et al., “An image is worth 1/2 tokens after layer 2: Plug-andplay inference acceleration for large vision-language models,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2024, pp. 19–35.

[34] T. Liu, L. Shi, R. Hong, Y. Hu, Q. Yin, and L. Zhang, “Multi-stage vision token dropping: Towards efficient multimodal large language model,” arXiv preprint arXiv:2411.10803, 2024.

[35] W. Li, Y. Yuan, J. Liu, D. Tang, S. Wang, J. Qin, J. Zhu, and L. Zhang, “TokenPacker: Efficient visual projector for multimodal LLM,” Int. J. Comput. Vis., vol. 133, no. 10, pp. 6794–6812, 2025.

[36] S. Zhang, Q. Fang, Z. Yang, and Y. Feng, “LLaVA-Mini: Efficient image and video large multimodal models with one vision token,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2025.

[37] Y. Shang, M. Cai, B. Xu, Y. J. Lee, and Y. Yan, “LLaVA-PruMerge: Adaptive token reduction for efficient large multimodal models,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2025, pp. 22 857–22 867.

[38] S. Yang, Y. Chen, Z. Tian, C. Wang, J. Li, B. Yu, and J. Jia, “VisionZip: Longer is better but not necessary in vision language models,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025, pp. 19 792–19 802.

[39] Y. Zhu, C. Xie, S. Liang, B. Zheng, and S. Guo, “FocusLLaVA: A coarse-to-fine approach for efficient and effective visual token compression,” arXiv preprint arXiv:2411.14228, 2024.

[40] M. Du, S. Ding, and H. Jia, “Study on density peaks clustering based on k-nearest neighbors and principal component analysis,” Knowl.-Based Syst., vol. 99, pp. 135–145, 2016.

[41] Y. Bengio, N. Leonard, and A. Courville, “Estimating or propagating´ gradients through stochastic neurons for conditional computation,” arXiv preprint arXiv:1308.3432, 2013.

[42] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “LoRA: Low-rank adaptation of large language models,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2022.

[43] H. Liu, C. Li, Y. Li, and Y. J. Lee, “Improved baselines with visual instruction tuning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 26 296–26 306.

[44] C. Fu, P. Chen, Y. Shen, Y. Qin, M. Zhang, X. Lin, J. Yang, X. Zheng, K. Li, X. Sun, Y. Wu, R. Ji, C. Shan, and R. He, “MME: A comprehensive evaluation benchmark for multimodal large language models,” in Adv. Neural Inf. Process. Syst., vol. 38, 2025, pp. 162 549–162 567.

[45] A. Kembhavi, M. Salvato, E. Kolve, M. Seo, H. Hajishirzi, and A. Farhadi, “A diagram is worth a dozen images,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2016, pp. 235–251.

[46] Y. Liu, H. Duan, Y. Zhang, B. Li, S. Zhang, W. Zhao, Y. Yuan, J. Wang, C. He, Z. Liu, K. Chen, and D. Lin, “MMBench: Is your multi-modal model an all-around player?” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2024, pp. 216–233.

[47] L. Chen, J. Li, X. Dong, P. Zhang, Y. Zang, Z. Chen, H. Duan, J. Wang, Y. Qiao, D. Lin, and F. Zhao, “Are we on the right way for evaluating large vision-language models?” in Adv. Neural Inf. Process. Syst., vol. 37, 2024, pp. 27 056–27 087.

[48] P. Lu, S. Mishra, T. Xia, L. Qiu, K.-W. Chang, S.-C. Zhu, O. Tafjord, P. Clark, and A. Kalyan, “Learn to explain: Multimodal reasoning via thought chains for science question answering,” in Adv. Neural Inf. Process. Syst., vol. 35, 2022, pp. 2507–2521.

[49] X. Yue, Y. Ni, T. Zheng, K. Zhang, R. Liu, G. Zhang, S. Stevens, D. Jiang, W. Ren, Y. Sun, C. Wei, B. Yu, R. Yuan, R. Sun, M. Yin, B. Zheng, Z. Yang, Y. Liu, W. Huang, H. Sun, Y. Su, and W. Chen, “MMMU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert AGI,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 9556–9567.

[50] H. Duan, J. Yang, Y. Qiao, X. Fang, L. Chen, Y. Liu, X. Dong, Y. Zang, P. Zhang, J. Wang, D. Lin, and K. Chen, “VLMEvalKit: An open-source toolkit for evaluating large multi-modality models,” in Proc. 32nd ACM Int. Conf. Multimedia (MM ’24), 2024, pp. 11 198–11 201.
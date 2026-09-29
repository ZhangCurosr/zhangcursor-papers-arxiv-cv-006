# WhiteCon: Semi-Supervised Domain Adaptation Regression Through Whitening Transform and Dual Consistency

Se Jin Sim<sup>[0009−0000−6028−1690]</sup> and Seoung Bum Kim<sup>[0000−0002−2205−8516]</sup>

School of Industrial and Management Engineering Korea University, Seoul, Republic of Korea {ssj259, sbkim1}@korea.ac.kr

Abstract. Domain adaptation is crucial for addressing distributional shifts that degrade model performance across domains. While most existing research has centered on classification, semi-supervised domain adaptation regression (SSDAR) for continuous-output tasks remains largely unexplored, particularly in practical scenarios with limited labeled target data. To address this gap, we propose semi-supervised domain adaptation regression through whitening transform and dual consistency (WhiteCon), which combines domain-specific whitening transform (DWT) and dual consistency regularization to enhance training stability and domain adaptation. DWT reduces the variance of the model parameters by transforming the feature covariance matrix into an identity matrix, thus stabilizing training under ordinary least squares assumptions. In addition, variance consistency regularization, as part of dual consistency regularization, aligns the variances of weak, strong, and mixup-augmented features to improve resilience against augmentation-induced perturbations. Empirical evaluations on various benchmark datasets under SSDAR settings demonstrate that the proposed White-Con achieves state-of-the-art performance compared to existing methods, effectively addressing domain shifts in regression tasks. The code for WhiteCon is available at https://github.com/sejin-sim/WhiteCon.

Keywords: semi-supervised domain adaptation, regression, feature whitening, consistency regularization, variance alignment.

## 1 Introduction

Domain adaptation is a critical area of research that addresses performance degradation caused by distribution shifts between different domains. This topic has recently gained substantial attention in various fields, such as computer vision, natural language processing, and medical data analysis [1-3]. When a model trained on a source domain is applied to a target domain, significant differences in data distributions across domains can degrade the model's generalization performance [4, 5]. This issue is particularly challenging when labeled data are limited, emphasizing the need for domain adaptation that can reduce distribution gaps to ensure reliable performance in diverse settings [6].

![](images/989275f51db2651dd53139e4029fc64cfff9527e7b4612810c1f0690440f271b.jpg)  
Fig. 1. Illustration of the OLS problem in SSDAR using WhiteCon.  represents the parameters of the regressor. (a) Domain shift between source and target data under OLS. (b) WhiteCon mitigates the domain shift in SSDAR under OLS.

Most existing research on domain adaptation has focused on classification problems, leading to the development of various methods for reducing domain differences. Some methods, such as statistical moment alignment, aim to minimize distributional discrepancies [7, 8]. Another effective strategy, adversarial training, uses a minimax game between feature extractors and domain classifiers to reduce domain discrepancies, yielding promising results [9, 10]. Although some classification methods can be extended to regression, their reliance on class probability distributions is poorly suited to continuous outputs, creating a significant challenge for regression-based domain adaptation [11].

In response to these challenges, unsupervised domain adaptation regression (UDAR) methods have been proposed [12, 13]. UDAR methods aim to reduce domain discrepancies by using only labeled source data and unlabeled target data, making them particularly useful when labeled target data is scarce. Several methods, including singular value decomposition and distribution matching in feature space, have been used to align source and target domains [11, 14]. However, even small amounts of labeled target data significantly improve performance and are often feasible to acquire [13, 15].

As a result, semi-supervised domain adaptation (SSDA) integrates small amounts of labeled target data with labeled source and unlabeled target data [16]. Similar to unsupervised domain adaptation (UDA), SSDA methods for classification are also challenging to apply directly to regression tasks because they often rely on entropy measures from class prediction probabilities, which do not align well with continuous regression outputs.

The earliest proposed semi-supervised domain adaptation regression (SSDAR) method [17] aligned statistical moments and used graph Laplacians for unlabeled data, while recent method [18] used domain-specific regressors for invariant risks and adversarial learning to prevent the model from distinguishing between source and target distributions. Although these methods can be applied to some extent without specific regression assumptions, they rely on classification-based approaches, which may limit their effectiveness in addressing the challenges of regression tasks. To address these challenges, it is crucial to develop a method that directly considers the intrinsic properties of regression, rather than simply adapting classification techniques.

Fig. 1 illustrates the challenges posed by domain shifts in SSDAR, highlighting the distributional gap between source and target domains. To tackle these challenges, we introduce semi-supervised domain adaptation regression through whitening transform and dual consistency (WhiteCon). WhiteCon effectively reduces domain shift and addresses the unique challenges of regression tasks by combining domain-specific whitening transform (DWT) and dual consistency regularization.

The first component, DWT, eliminates correlations between features by transforming the feature covariance matrix into an identity matrix. Building upon the success of DWT in classification tasks [19], to our knowledge we are the first to apply it to regression tasks, providing a mathematical justification under ordinary least squares (OLS) assumptions. Unlike classification where DWT only aids feature alignment, in regression DWT improves model performance through regressor parameter variance reduction. Second, we introduce dual consistency regularization, which enforces consistency across both predictions and feature representations. Specifically, the variance consistency regularization enforces variance alignment among features extracted from weakly, strongly, and mixup-augmented samples of unlabeled target data. By matching the variances of strong and mixup augmentations to those of weak augmentations, the model achieves consistency across augmentation intensities and improves robustness. In summary, our method combines the variance-reducing effect of DWT with the robustness provided by dual consistency regularization, creating an effective method for SSDAR. The contributions of this study are summarized as follows:

(1) To the best of our knowledge, this study presents the first application of DWT to regression tasks with mathematical justification under the OLS framework. We demonstrate that DWT reduces the variance of regression parameters, thus improving task performance and providing theoretical support for the effectiveness of the proposed method.

(2) We introduce feature variance consistency regularization that aligns feature variances across different augmentations to improve robustness in SSDAR. This variance alignment promotes consistent feature distributions, reducing sensitivity to perturbations and enhancing generalization. Unlike existing methods that rely on complex formulations or focus on only one aspect independently, our approach is simple and addresses both feature and prediction consistency simultaneously in an intuitive manner.

(3) We propose WhiteCon, a simple yet effective framework that combines DWT and dual consistency regularization for SSDAR. Our method achieves state-ofthe-art performance on various benchmark datasets and demonstrates its superiority through extensive evaluations, including ablation studies, regressor parameter variance analysis, varying proportions of labeled target data, and feature visualizations.

## Related work

## 2.1 Domain Adaptation for Classification

UDA has been widely studied, particularly for classification tasks. One common approach, moment matching, aims to reduce distributional discrepancies by aligning the statistical moments of distributions [7, 8]. Maximum mean discrepancy (MMD) [20], a prominent technique, measures divergence by comparing their features in a reproducing kernel Hilbert space. However, MMD focuses on aligning input feature distributions but may overlook the complex dependencies between input features and continuous targets in regression tasks.

Another approach is adversarial learning, inspired by generative adversarial networks [21], which produces domain-invariant representations by training features that are difficult to distinguish between domains [9]. This method minimizes distributional differences through a competitive process with a domain discriminator [10]. Domainadversarial neural networks (DANN) [6] align distributions through a gradient reversal layer. However, when domain discrepancies are substantial or highly complex, domaininvariant representations alone may not suffice to reduce the gap effectively.

The availability of labeled target data has increased interest in SSDA [22]. Saito et al. [13] demonstrated that UDA methods often struggled in SSDA settings. Various semi-supervised methods have applied consistency regularization, which aligns model predictions across augmented views of the same data to enhance stability and performance [23]. However, while effective for classification within SSDA, consistency regularization's effectiveness in regression remains unproven.

Most SSDA methods for classification rely on computing the entropy of class prediction probabilities using softmax [13, 24]. This reliance limits their applicability to regression tasks, which require models capable of handling continuous value predictions rather than class probability distributions.

## 2.2 Domain Adaptation for Regression

Cortes and Mohri [25] provided a theoretical analysis of domain adaptation for regression. Over time, several methods have been introduced for this task, often relying on importance weighting in non-deep learning models or focusing on creating invariant representations of features [18].

Recently, UDAR methods were proposed to address regression using only labeled source and unlabeled target data. Chen et al. [11] proposed representation subspace distance (RSD) for domain adaptation regression, which uses singular value decomposition to produce orthogonal bases while preserving feature scale. However, our empirical analysis reveals that the relationship between feature scale and performance was not always consistent. Nejjar et al. [14] proposed domain adaptation regression by aligning the inverse Gram matrices (DARE-GRAM) based on the closed-form OLS problem. However, deriving the pseudo-inverse matrix required specifying the lowrank property as a hyperparameter, potentially leading to excessive hyperparameter tuning. Wu et al. [26] proposed distribution-informed neural networks that build distribution-aware relationships using neural tangent kernel theory, but assume infinite-width networks that may not hold in practice. Dhaini et al. [27] proposed dictionary learning for subspace mapping, though performance depends on dictionary size and initialization.

SSDAR methods aim to address realistic scenarios where a small amount of labeled target data is available for model training. Singh and Chakraborty [17] proposed deep domain adaptation for regression (DeepDAR), the first SSDAR method using deep learning. They used MMD for distribution alignment and graph Laplacians for unlabeled target data. However, MMD only aligns feature distributions without ensuring consistent regression outputs, creating a weak connection to regression performance. Li et al. [18] proposed learning invariant representations and risks (LIRR) for both regression and classification, using domain-specific predictors and DANN. While theoretically grounded, LIRR's joint optimization prevents it from addressing regressionspecific parameter stability under domain shift.

## 3 Proposed Methods

## 3.1 Preliminaries

Roy et al. [19] proposed DWT for reducing the distribution gap between source and target domains by eliminating feature correlations in unsupervised domain adaptation classification (UDAC). Unlike batch normalization, DWT applies batch whitening to transform the feature covariance matrix into an identity matrix. This ensures that features from both domains are mapped into a spherical distribution, which helps the model generalize across both domains. Let  represent the original feature vectors, and Ż denote the decorrelated feature by applying a whitening matrix W. The whitening transformation is as follows:

$$
\hat { Z } = { \cal W } ( Z - \bar { Z } ) ,\tag{1}
$$

where <sup>̅</sup> is the mean vector of the features, and $W$ is computed from the covariance matrix $\Sigma = ( Z - \bar { Z } ) ( Z - \bar { Z } ) ^ { T }$ . Using the Cholesky decomposition, the covariance matrix Σ is decomposed as $\boldsymbol { \Sigma } = \boldsymbol { L } \boldsymbol { L } ^ { T }$ , where  is a lower triangular matrix. The whitening matrix is then obtained as $W = L ^ { - 1 }$ . By applying $W$ , the covariance matrix of the whitened feature $\hat { Z }$ becomes the identity matrix , as shown below:

$$
\operatorname { C o v } \left( { \hat { Z } } \right) = { \hat { Z } } { \hat { Z } } ^ { T } = W ( Z - { \bar { Z } } ) ( Z - { \bar { Z } } ) ^ { T } W ^ { T } = W \Sigma W ^ { T } = L ^ { - 1 } L L ^ { T } ( L ^ { T } ) ^ { - 1 } = I .\tag{2}
$$

Consequently, DWT removes correlations between features and aligns the distributions between the source and target domains, improving generalization and reducing the impact of domain shifts.

Let $S = \{ ( x _ { i } ^ { s } , y _ { i } ^ { s } ) \} _ { i = 1 } ^ { n }$ represent a set of labeled data from the source domain, where y denotes the continuous values associated with samples $x _ { i } ^ { s }$ . Similarly, let $T _ { u l } =$ $\left\{ ( x _ { i } ^ { t u l } ) \right\} _ { i = 1 } ^ { k }$ represent a set of unlabeled data from the target domain. Additionally, let

$T _ { l } = \{ ( x _ { i } ^ { t } , y _ { i } ^ { t } ) \} _ { i = } ^ { m }$ represent a small set of labeled data in the target domain, where m « n and typically $m \ll k$ , indicating that only a small amount of target data is labeled. A key challenge is domain shift, where the source domain distribution $P ( x ^ { s } )$ differs from the target domain distribution $P { \Big ( } x ^ { ( t , t u l ) } { \Big ) } , { \mathrm { i . e . , } } P ( x ^ { s } ) \not = P { \Big ( } x ^ { ( t , t u l ) } { \Big ) }$ . This domain shift complicates model performance in the target domain, necessitating adaptation to address the distribution gap.

Given the data, the feature extractor $f ( \cdot )$ transforms the source and target samples into features. The regressor $g ( \cdot )$ , implemented as a fully connected layer, uses these features to generate predictions for both domains, denoted as ${ \hat { y } } ^ { s } = g ( f ( x ^ { s } ) )$ and $\hat { y } ^ { t } = g ( f ( x ^ { t } ) )$ . To optimize the model using the labeled samples from both the source and target domains, the supervised loss $L _ { s u p }$ is defined as follows:

$$
\begin{array} { r } { L _ { s u p } = \frac { 1 } { 2 } \Big ( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( y _ { i } ^ { s } - \hat { y } _ { i } ^ { s } ) ^ { 2 } + \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \big ( y _ { i } ^ { t t } - \hat { y } _ { i } ^ { t t } \big ) ^ { 2 } \Big ) , } \end{array}\tag{3}
$$

where  and  represent the numbers of labeled samples in the source and target domains, respectively.

## 3.2 Motivations

Once features $Z$ are extracted, predicting the target  using regressor parameters $\beta$ can be formulated as an OLS problem [28]:

$$
Y = Z \beta + \epsilon ,\tag{4}
$$

where  is a random error. $\mathrm { W e }$ construct our regressor $g ( \cdot )$ as a single linear layer. This simplifies the mapping from features to predictions, clarifying the OLS problem. Under the OLS assumption, the variance of the parameter $\beta$ can be expressed as follows:

$$
\operatorname { V a r } \left( { \hat { \beta } } \right) = \sigma ^ { 2 } ( Z ^ { T } Z ) ^ { - 1 } ,\tag{5}
$$

where $\sigma ^ { 2 }$ represents the variance of the error . As shown in Equation (2), when features $Z$ are whitened using DWT in the feature extractor, the equation is modified as follows:

$$
\operatorname { V a r } \left( { \hat { \beta } } \right) = \sigma ^ { 2 } { \left( { \hat { Z } } ^ { T } { \hat { Z } } \right) } ^ { - 1 } = \sigma ^ { 2 } I ,\tag{6}
$$

where $\sigma ^ { 2 } I$ indicates that the variance of the regressor parameters is reduced. This reduction in parameter variance can help improve the performance of the regression tasks [29]. These equations are applicable to SSDAR where a small amount of labeled target data is available for model training. Furthermore, a previous study [11] has shown that batch normalization negatively impacts domain adaptation regression, with better performance observed when it is disabled.

![](images/b81df30ee9d34a38d7e7020e48f372e40594dc990a1e3d7d2a11efb23b6d24c2.jpg)  
Fig. 2. Overview of the proposed framework, WhiteCon. DWT removes correlations between features and reduces the variance of regression model parameters. The proposed loss $L _ { c r f }$ trains the model to ensure that the variances of strongly and mixup-augmented features $\left( \mathrm { V a r } ^ { s t r o n g } \right.$ and ${ \mathrm { V a r } } ^ { m i x } .$ ) align more closely with the variance of weakly augmented features $\mathrm { V a r } ^ { w e a k }$ , introducing consistency regularization for the augmented unlabeled features and enhancing model robustness.

## 3.3 Overview of the Proposed Method

Our method combines DWT and dual consistency regularization loss $L _ { d \sigma }$ to address SSDAR. $L _ { d c r }$ consists of prediction consistency regularization loss $L _ { c r p }$ and feature variance consistency regularization $L _ { c r f }$ . The total loss $\mathrm { L } _ { t o t a l }$ is given by:

$$
{ \cal L } _ { t o t a l } \ = { \cal L } _ { s u p } + { \cal L } _ { d c r } \ = { \cal L } _ { s u p } + { \cal L } _ { \alpha p } + { \cal L } _ { \alpha f } .\tag{7}
$$

Fig. 2 presents an overview of our proposed framework, WhiteCon. The framework takes source and target samples and applies weak, strong, and mixup augmentations to the unlabeled target samples. These augmented samples pass through a feature extractor with DWT, which removes feature correlations and stabilizes training by reducing the variance of regressor parameters. The loss $L _ { c r f }$ aligns the variances of strongly and mixup-augmented features with those of weakly augmented features, while $L _ { c p }$ enforces consistency across predictions from these augmentations.

## 3.4 Dual Consistency Regularization

As discussed in Section 2.1, consistency regularization has proven effective in domain adaptation. Inspired by the semi-supervised regression approach of Sim et al. [30], we apply weak, strong, and mixup augmentations to unlabeled target samples $x ^ { t u l }$ to obtain weak $x ^ { w e a k }$ , strong $x ^ { s t r o n g }$ and mixup $x ^ { m \dot { \kappa } }$ augmented unlabeled target samples. Here, $x ^ { m \dot { \alpha } }$ is a linear combination of weak and strong augmented samples.

Prediction Consistency Regularization. After applying these augmentations, the feature extractor $f ( \cdot )$ , which includes DWT, transforms the augmented unlabeled target samples into feature representations $\hat { Z } ^ { w e a k } , \hat { Z } ^ { s t r o n g }$ and $\hat { Z } ^ { m \dot { \alpha } }$ , respectively. Each of these feature representations belongs to $\mathbb { R } ^ { k \times d }$ , where  is the feature dimensionality. Using the regressor $g ( \cdot )$ , we obtain predictions $\tilde { y } ^ { w e a k } , \hat { y } ^ { s t r o n g }$ and $\hat { y } ^ { m \hat { \boldsymbol { \kappa } } }$ . We train the model to make the strongly augmented prediction strong and the mixup prediction $\hat { y } ^ { m \hat { \boldsymbol { \kappa } } }$ close to the weakly augmented prediction $\tilde { y } ^ { w e a k }$ as the pseudo-label. The prediction consistency regularization loss $L _ { c p }$ is defined as follows:

$$
\begin{array} { r } { L _ { \alpha p } \ = \lambda _ { 1 } \Big ( \frac { 1 } { k } \sum _ { i = 1 } ^ { k } \big ( \widetilde { y } _ { i } ^ { w e a k } - \widehat { y } _ { i } ^ { s t r o n g } \ \big ) ^ { 2 } + \frac { 1 } { k } \sum _ { i = 1 } ^ { k } \big ( \widetilde { y } _ { i } ^ { w e a k } - \widehat { y } _ { i } ^ { m i x } \big ) ^ { 2 } \Big ) , } \end{array}\tag{8}
$$

where $\lambda _ { 1 }$ is a hyperparameter for $L _ { c p }$ . Incorporating elements of a semi-supervised regression approach, $L _ { c p }$ promotes consistent predictions across different augmentations, which is important in SSDAR settings with limited labeled target data.

Feature Variance Consistency Regularization. Unlike classification tasks, regression tasks predict continuous values for unlabeled data, making it challenging to apply threshold-based functions for pseudo-label refinement as in classification [31]. Therefore, introducing additional consistency beyond prediction consistency regularization is critical [30]. Previous studies proposed feature consistency across augmented unlabeled targets through complex calculations, such as simultaneously regulating variance, invariance, and covariance, or contrastive learning [32]. To simplify and enhance consistency regularization, we use feature variance from augmentations. The variance of the weakly augmented features ${ \mathrm { V a r } } ^ { w e a k }$ is calculated as follows:

$$
\begin{array} { r } { \mathrm { V a r } ^ { w e a k } = \frac { 1 } { k } \sum _ { i = 1 } ^ { k } \bigl ( \hat { Z } _ { i } ^ { w e a k } - \mu ^ { w e a k } \bigr ) ^ { 2 } , } \end{array}\tag{9}
$$

where $\hat { Z } ^ { w e a k } \in \mathbb { R } ^ { k \times d }$ represents the weakly augmented features across  samples and  feature dimensions, and $\mu ^ { w e a k } \in \mathbb { R } ^ { 1 \times d }$ is the mean vector of these features. The variance is computed along each feature dimension, resulting in a variance vector with the same dimensionality  as the feature matrix. Similar to $\operatorname { V a r } ^ { w e a k }$ in Equation (9), the variances for strongly and mixup-augmented features $\left( \mathrm { V a r } ^ { s t r o n g } \right.$ and ${ \mathrm { V a r } } ^ { m { \ i } { \sigma } } )$ are also computed along each feature dimension. The model is trained to align the variances of the strongly $\mathrm { V a r } ^ { s t r o n g }$ and mixup-augmented features ${ \mathrm { V a r } } ^ { m { \dot { \alpha } } }$ with that of the weakly augmented features ${ \mathrm { V a r } } ^ { w e a k }$ . The feature consistency regularization loss $L _ { c r f }$ is expressed as follows:

$$
\begin{array} { r } { L _ { \sigma f } = \lambda _ { 2 } \left( \frac { 1 } { d } \sum _ { j = 1 } ^ { d } \bigl | \mathrm { V a r } _ { j } ^ { w e a k } - \mathrm { V a r } _ { j } ^ { s t v o n g } \ \bigr | + \frac { 1 } { d } \sum _ { j = 1 } ^ { d } \bigl | \mathrm { V a r } _ { j } ^ { w e a k } - \mathrm { V a r } _ { j } ^ { m \ i k } \bigr | \right) , } \end{array}\tag{10}
$$

where $\lambda _ { 2 }$ is a hyperparameter for $L _ { c r f }$ . This variance alignment promotes consistent feature distributions across augmentations, reducing sensitivity to perturbations and enhancing generalization. Mean absolute error (MAE) is used for $L _ { c r f }$ because it handles differences evenly and prevents large errors from disproportionately affecting the training, leading to more stable variance alignment [33]. Unlike existing methods that focus on either prediction or feature consistency independently, our dual consistency regularization simultaneously addresses both aspects in a unified and straightforward manner.

![](images/00c0508a5fcd284b7cd832233b7c887bf149fafc88ed615e3ef85ffb80b973a5.jpg)  
Fig. 3. Sample images from the three benchmark datasets, BIWI, QMUL, and MPI3D.

## 4 Experiments

## 4.1 Datasets

We used three benchmark datasets to address the SSDAR problem: BIWI [34] and QMUL [35] for head pose estimation, and MPI3D [36] for object position estimation. Sample images from each dataset are shown in Fig. 3. To satisfy the SSDAR setting, we randomly selected a small proportion (5%) of the target data as labeled target samples for training. Weak augmentation consisted of random cropping and Gaussian blur (excluding angle-altering transforms such as flips), and strong augmentation followed the FixMatch protocol.

BIWI. This dataset contains 5,874 images of 6 female (F) subjects and 9,804 images of 14 male (M) subjects, captured while turning their heads. We evaluated the dataset under two cases: M→F and F→M, predicting yaw, pitch, and roll angles.

QMUL. The dataset consists of 1,504 images of 8 female (F) subjects and 3,595 images of 40 male (M) subjects, also captured while turning their heads. We evaluated two cases: M→F and F→M, predicting yaw and pitch angles.

MPI3D. This dataset contains 1,036,800 3D object samples across three domains: real (RL), realistic (RC), and toy (T). The dataset has been used in UDAR settings [11, 14], and we adapted it for SSDAR, evaluating six cases: RL→RC, RL→T, RC→RL, RC→T, T→RL, and T→RC, predicting the vertical and horizontal axes.

## 4.2 Experimental Setup

Implementation Details. For the backbone model, we used ResNet50, which is widely used in domain adaptation regression and is suitable for the input image size [11, 14]. The source and target labels were scaled to the range [0, 1] using min-max normalization. The AdamW optimizer was used with a learning rate of 0.001 and a batch size of 48. The model was trained for 50 epochs, with a linear ramp-up for $L _ { c r p }$ and $L _ { c r f }$ over the first 20 epochs for stability. The optimal hyperparameters $\lambda _ { 1 }$ and $\lambda _ { 2 }$ were selected based on the highest validation score. An NVIDIA RTX 4090 GPU was used for all the experiments.

Compared Methods. We adapted the following methods as mentioned in Section 2 to the SSDAR setting for comparison: S+T, MMD, DANN, RSD, DARE-GRAM, Deep-DAR, LIRR, and Full\_T. S+T is a supervised learning method that only uses labeled source data and a small amount (5%) of labeled target data. Full\_T is a model trained with the full set (100%) of labeled target data. We included UDAC methods (MMD and DANN) that can be applied to regression tasks, UDAR methods (RSD and DARE-GRAM), and SSDAR methods (DeepDAR and LIRR). Methods that could not be implemented under the same experimental conditions were excluded from the comparison [26, 27]. To apply UDA methods to the SSDAR setting, we incorporated $L _ { s u p }$ to train the model using the labeled target samples [18, 23].

## 4.3 Results

BIWI. Table 1 shows that WhiteCon achieved the best performance with an average $R ^ { 2 }$ of 0.86 and MAE of 4.71. Performance differences were more pronounced in F→M (WhiteCon: $R ^ { 2 }$ 0.87 vs. RSD: $R ^ { 2 } \ : 0 . 8 0 )$ , while M→F results were comparable across methods $( R ^ { 2 } \colon 0 . 8 3  – 0 . 8 5 )$ . WhiteCon outperformed the second-best method, RSD (average R²: 0.82 (−0.04), MAE: 5.73 (+0.99)).

QMUL. WhiteCon achieved the best performance in both F→M and M→F cases, with an average $R ^ { 2 }$ of 0.83 and MAE of 9.35 (Table 1). Notably, most baseline methods showed performance similar to S+T (within 0.01 $R ^ { 2 }$ difference), suggesting saturation of existing approaches on QMUL. WhiteCon achieved meaningful gains over the second-best method, MMD (average $R ^ { 2 } { : 0 . 7 9 ( - 0 . 0 4 ) }$ , MAE: 10.52 (+1.17)).

MPI3D. As shown in Table 2, WhiteCon achieved the highest overall performance across all cases, with an average $R ^ { 2 } \ \mathrm { { o f } } \ 0 . 9 0 $ and MAE of 2.04. While scenarios with RL and RC as source domains achieved higher performance, the T→RL and T→RC cases proved challenging for comparative methods. Nevertheless, WhiteCon demonstrated robust performance even under these difficult conditions. The second-best method, DARE-GRAM (average $R ^ { 2 } \colon 0 . 7 4 \ ( - 0 . 1 6 )$ , MAE: 3.78 (+1.74)), lagged substantially behind, representing the largest performance gap observed across the three datasets.

## 4.4 Analyses

Ablation Study. To evaluate the contribution of each component, we removed each module individually across three datasets (Table 3). While the most influential component varied across datasets, removing $L _ { \sigma f } ~ \left( \mathrm { w } / { 0 } ~ L _ { c r f } \right)$ consistently led to the largest performance degradation, confirming its critical role in enhancing overall performance. Removing DWT (w/o DWT) or $L _ { c p } ~ \left( \mathrm { w } / \mathrm { o } L _ { \sigma p } \right)$ also led to performance drops of similar degree, indicating that both modules provide complementary contributions of comparable importance. These findings demonstrate that WhiteCon’s superior performance relies on the integration of all three components, with $L _ { c r f }$ being the most important factor in general.

Table 1. Performance comparison of the proposed method and comparative methods on BIWI and QMUL in terms of $R ^ { 2 }$ and MAE. Result is reported as mean ± standard deviation with three different random seeds. Bold and underline indicate the best and the second-best results. The Wilcoxon rank-sum test is used to verify significant differences between the best and secondbest average results, and is noted by p-value (\*: p-value < 0.05).
<table><tr><td rowspan="3">Method</td><td colspan="6">BIWI</td><td colspan="6">QMUL</td></tr><tr><td colspan="2">F→M</td><td colspan="2">M→F</td><td colspan="2">Average</td><td colspan="2">F→M</td><td colspan="2">M→F</td><td colspan="2">Average</td></tr><tr><td>R²↑</td><td>MAE↓</td><td>R²↑</td><td>MAE↓</td><td>R²↑</td><td>MAE↓</td><td>R²↑</td><td>MAE↓</td><td>R²↑</td><td>MAE↓</td><td>R²↑</td><td>MAE↓</td></tr><tr><td>S+T</td><td>0.79±00B</td><td>557±009</td><td>0.83±001</td><td>6.43±024</td><td>0.81±008</td><td>6.00±047</td><td>0.77±001</td><td>11.60±029</td><td>0.79±001</td><td>10.11+04</td><td>0.78±001</td><td>10.86±085</td></tr><tr><td>MMD</td><td>0.80±001</td><td>551+020</td><td>0.83±000</td><td>629±006</td><td>0.82+002</td><td>590±02</td><td>0.78±001</td><td>11.38±056</td><td>0.80±001</td><td>9.66±014</td><td>0.79±001</td><td>10.52+095</td></tr><tr><td>DANN</td><td>0.80±2</td><td>5.70±028</td><td>0.84±001</td><td>595±009</td><td>0.82+002</td><td>5.83±024</td><td>0.78±001</td><td>11.30±011</td><td>0.80±0Y2</td><td>996±023</td><td>0.79±02</td><td>10.63±070</td></tr><tr><td>RSD</td><td>0.79±001</td><td>5.68±007</td><td>0.85±001</td><td>5.79±021</td><td>0.82+008</td><td>5.73±016</td><td>0.78±00</td><td>11.27+010</td><td>0.79±001</td><td>10.30±061</td><td>0.79±001</td><td>10.79±066</td></tr><tr><td>DARE-GRAM</td><td>0.79±001</td><td>555±00</td><td>0.81+001</td><td>631+020</td><td>0.80±001</td><td>5.93±043</td><td>0.77±002</td><td>11.68±033</td><td>0.80±0m</td><td>9.77±022</td><td>0.78±002</td><td>10.72±100</td></tr><tr><td>DeepDAR</td><td>0.77±002</td><td>6.04±033</td><td>0.83±001</td><td>633±017</td><td>0.80±00B</td><td>6.19±030</td><td>0.77±002</td><td>11.35±036</td><td>0.79±02</td><td>10.33±02</td><td>0.78±002</td><td>10.84±057</td></tr><tr><td>LIRR</td><td>0.79±002</td><td>5.83±09</td><td>0.84001</td><td>6.11±009</td><td>0.81+00B</td><td>597±031</td><td>0.77±001</td><td>11.33±035</td><td>0.79±02</td><td>996±014</td><td>0.78±002</td><td>10.65±074</td></tr><tr><td>WhiteCon (ours)</td><td>087±001</td><td>407±013</td><td>0.85±001</td><td>535±007</td><td>*0.86±001</td><td>*471±066</td><td>0.82+001</td><td>9.74±040</td><td>083±001</td><td>895±007</td><td>*0.83±001</td><td>*935±04</td></tr><tr><td>Full_T</td><td>094±001</td><td>330±011</td><td>0.98±000</td><td>262±034</td><td>096±002</td><td>296±02</td><td>0.88±001</td><td>931±031</td><td>092±00</td><td>6.64±024</td><td>090±00B</td><td>798±137</td></tr></table>

Table 2. Performance comparison of the proposed method and comparative methods on MPI3D in terms of $R ^ { 2 }$ and MAE. Result is reported as mean ± standard deviation across two regression outputs with three different random seeds. Bold and underline indicate the best and the secondbest results. The Wilcoxon rank-sum test is used to verify significant differences between the best and second-best average results, and is noted by p-value (\*: p-value < 0.05).
<table><tr><td rowspan="2">Method</td><td colspan="2">RL→RC</td><td colspan="2">RL→T</td><td colspan="2">RC→RL</td><td colspan="2">RC→T</td><td colspan="2">T→RL</td><td colspan="2">T→RC</td><td colspan="2">Average</td></tr><tr><td>R²↑MAE↓</td><td></td><td>R²↑MAE↓</td><td></td><td>R²↑MAE↓</td><td></td><td>R²↑MAE↓</td><td></td><td>R²↑MAE↓</td><td></td><td>R²↑MAE↓</td><td></td><td></td><td>R²↑MAE↓</td></tr><tr><td>S+T</td><td>0.88x</td><td>2548</td><td>0.662</td><td>4.61±8</td><td>0.81#5</td><td>333</td><td>0.70k0</td><td>434xw</td><td>-0.03xm</td><td>9.05m</td><td>034m</td><td>6.6687</td><td>056m</td><td>5.0924</td></tr><tr><td>MMD</td><td>0.89</td><td>251-00</td><td>0.788</td><td>3.8686</td><td>0.82#16</td><td>3.146</td><td>0.79±</td><td>3.80k</td><td>0.09TB</td><td>8.49±5</td><td>031±0</td><td>7.22#</td><td>0.61±m0</td><td>4.8423</td></tr><tr><td>DANN</td><td>0.898</td><td>246m5</td><td>0.7706</td><td>3.85±8</td><td>0.83±m</td><td>341</td><td>0.81±18</td><td>3.57±8</td><td>0.03±TB</td><td>894±7</td><td>0.24m6</td><td>7.61m</td><td>0.60±m</td><td>4978</td></tr><tr><td>RSD</td><td>0.92±m</td><td>2138</td><td>0.7506</td><td>4.1815</td><td>0.84mB</td><td>3.04m</td><td>0.80T5</td><td>3.698</td><td>0.04m</td><td>8.77±8</td><td>034m</td><td>6.88HB1</td><td>0.61±g</td><td>4.78</td></tr><tr><td>DARE-GRAM</td><td>0.93HT</td><td>2.14m</td><td>0.89</td><td>281±18</td><td>0.90kT</td><td>242HT8</td><td>0.90xm</td><td>260x14</td><td>0.13m</td><td>8.28t</td><td>0.69</td><td>444m</td><td>0.74</td><td>3.7815</td></tr><tr><td>DeepDAR</td><td>0.89m</td><td>268Hm</td><td>0.74</td><td>428x8</td><td>0.76±TB</td><td>4.12+D4</td><td>0.79±4</td><td>3.79</td><td>0.0818</td><td>8572</td><td>024m8</td><td>7.76</td><td>0.58±8</td><td>520£21</td></tr><tr><td>LIRR</td><td>0.90xB</td><td>239±R0</td><td>0.70±14</td><td>441±</td><td>0.83T</td><td>3.22</td><td>0.79±5</td><td>3.85±6</td><td>0.04m</td><td>8.74m2</td><td>0364</td><td>7.00±B1</td><td>0.60xBl</td><td>494m</td></tr><tr><td>WhiteCon (ours)</td><td>0.980</td><td>1.0604</td><td>0.95H0N</td><td>19315</td><td>0.980</td><td>128M19</td><td>0.95</td><td>183</td><td>0.6156</td><td>432m05</td><td>0.95H</td><td>18204</td><td>*0.90m3</td><td>*204p</td></tr><tr><td>Full_T</td><td>0.98x0</td><td>127318</td><td>098m0</td><td>1332</td><td>0.98±1</td><td>121±</td><td>0.98±1</td><td>133m2</td><td>0.98xD</td><td>121¥D</td><td>098</td><td>127±8</td><td>0.98x0</td><td>127±0</td></tr></table>

Table 3. Contribution of each component (DWT, $L _ { c p }$ $L _ { c r f } )$ to the overall performance of WhiteCon on BIWI, QMUL, and MPI3D datasets in terms of MAE. Result is reported as mean ± standard deviation with three different random seeds. Bold indicates the component with the largest contribution to performance on each dataset, and underline denotes the second most influential component.
<table><tr><td>Method</td><td>DWT  $L _ { \sigma p }$ </td><td> $L _ { c r f }$ </td><td>BIWI Average MAE</td><td>QMUL Average MAE</td><td>MPI3D Average MAE</td></tr><tr><td>Proposed</td><td>√ √</td><td>√</td><td>4.71±0.65</td><td>9.35±0.49</td><td>2.04±1.09</td></tr><tr><td>w/o DWT</td><td>√</td><td>√</td><td>4.83±0.71</td><td>9.97±039</td><td>2.63±220</td></tr><tr><td>w/o  $L _ { \sigma p }$ </td><td>√</td><td>√</td><td>4.94±056</td><td>9.54±0.71</td><td>3.58±189</td></tr><tr><td>w/o  $\underline { { L _ { c r f } } }$ </td><td>√ V</td><td></td><td>5.01±053</td><td>10.69±0.76</td><td>2.32±134</td></tr></table>

![](images/46091927cef5a637d18d839d763015c64ea02c1f8f0e3985b79be12d3d3bf86c.jpg)  
Fig. 4. Comparison of regressor parameter variance for QMUL F→M. The -axis represents the different methods, the -axis shows parameter variance values.

![](images/6a7ad04c355217110194a05fd4264e2891ef91665057f1e0fb8c17a840954663.jpg)  
Fig. 5. Performance comparison with increasing proportions of labeled target data on QMUL F→M. The -axis shows the proportions of labeled target data; the -axis shows $R ^ { 2 } .$ . The dashed line indicates the upper bound (Full\_T) with 100% labeled target data.

![](images/005766837d97ba9670e760a9aaa1676ba591d55d63071bbf549cbaba4abef543.jpg)  
Fig. 6. Feature visualization results of comparison methods on MPI3D RL→RC. The color bar represents label values.

Variance of Regressor Parameters. To empirically validate the mathematical justification presented in Section 3.2, we conducted experiments to measure parameter variance across methods (Fig. 4). WhiteCon achieved the lowest variance of $1 . 2 7 3 { \times } 1 0 ^ { - 4 }$ compared to $1 . 4 3 8 \times 1 0 ^ { - 4 } - 1 . 5 1 5 \times 1 0 ^ { - 4 }$ for comparison methods. These results are consistent with Section 3.2, suggesting that lower parameter variance contributes to improved regression performance. Notably, while Full\_T achieved high performance despite higher variance due to the absence of domain shift, among domain adaptation methods, WhiteCon both minimized variance and achieved the highest performance.

Increasing Proportions of Labeled Target. To verify WhiteCon's robustness across varying amounts of labeled target data, we conducted experiments with 5%, 10%, 20%, and 30% proportions. Fig. 5 shows that WhiteCon consistently outperformed comparison methods across all proportions on QMUL F→M. Remarkably, WhiteCon with 30% labeled data $( R ^ { 2 } 0 . 9 1 )$ exceeded Full\_T trained on 100% data $( R ^ { 2 } 0 . 8 8 )$ .

Feature Visualization. To examine feature alignment in the regression context, where effective alignment should show continuous patterns along label values, we visualized features (from the feature extractor $f ( \cdot ) )$ on MPI3D RL→RC using t-distributed stochastic neighbor embedding in Fig. 6. (a) S+T showed features largely unaligned with label values. While (d) RSD and (f) DeepDAR demonstrated improved alignment, confusion between label values remained, with dark colors scattered among bright colors. In contrast, (h) WhiteCon showed the clearest alignment, effectively grouping features with similar label values and demonstrating its superior ability to reduce the domain gap.

## 5 Conclusion

In this study, we proposed WhiteCon, a method for addressing SSDAR problems, combining DWT and dual consistency regularization to reduce domain discrepancies. DWT improves model performance under OLS assumptions by transforming the feature covariance matrix into an identity matrix, thus reducing the variance of regression parameters. Variance consistency regularization aligns feature variances across augmentations, improving model robustness. Experimental results on multiple benchmark datasets demonstrate that WhiteCon achieves state-of-the-art performance, confirming its effectiveness. Future work will extend WhiteCon to underexplored modalities such as time series and tabular data, which are widely used in real-world regression applications but remain understudied in SSDAR, requiring the design of modality-specific augmentation and adaptation strategies.

## Acknowledgments

This research was supported by Brain Korea 21 FOUR, the Ministry of Science and ICT (MSIT) in Korea under the ITRC support program supervised by the Institute for Information Communication Technology Planning and Evaluation (IITP-2026-RS-2020-0-01749), and the National Research Foundation of Korea grant funded by the Korea government (RS-2022-00144190).

## References

1. Das A, Rabby ASA, Kowsar I, Rahman F. A deep learning-based unified solution for character recognition. In: Proceedings of the International Conference on Pattern Recognition, ICPR, p. 1671-7 (2022)

2. Pandey V, Panwar N, Kumbhar A, Roy PP, Iwamura M. Enhanced cross-task EEG classification: domain adaptation with EEGNet. In: Proceedings of the International Conference on Pattern Recognition, ICPR, p. 354–69 (2024)

3. Chen Y, Ouyang X, Zhu K, Agam G. Semi-supervised dual-domain adaptation for semantic segmentation. In: Proceedings of the International Conference on Pattern Recognition, ICPR, p. 230-7 (2022)

4. Wu X, Du C, Zhang H, Liu J, Zhang D, Zou H. Unsupervised domain adaptation for crossdevice iris liveness detection model transfer. In: Proceedings of the International Conference on Pattern Recognition, ICPR, p. 256-72 (2024)

5. Seraj MS, Chakraborty S. Multi-source deep domain adaptation for deepfake detection. In: Proceedings of the International Conference on Pattern Recognition, ICPR, p. 127-42 (2024)

6. Ganin Y, Ustinova E, Ajakan H, Germain P, Larochelle H, Laviolette F, et al. Domainadversarial training of neural networks. Journal of Machine Learning Research 17(59):1- 35 (2016)

7. Long M, Cao Y, Wang J, Jordan M. Learning transferable features with deep adaptation networks. In: Proceedings of the International Conference on Machine Learning, ICML, p. 97-105 (2015)

8. Long M, Zhu H, Wang J, Jordan MI. Deep transfer learning with joint adaptation networks. In: Proceedings of the International Conference on Machine Learning, ICML, p. 2208-17 (2017)

9. Hoffman J, Tzeng E, Park T, Zhu J-Y, Isola P, Saenko K, et al. Cycada: Cycle-consistent adversarial domain adaptation. In: Proceedings of the International Conference on Machine Learning, ICML, p. 1989-98 (2018)

10. Tzeng E, Hoffman J, Saenko K, Darrell T. Adversarial discriminative domain adaptation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR, p. 7167-76 (2017)

11. Chen X, Wang S, Wang J, Long M. Representation subspace distance for domain adaptation regression. In: Proceedings of the International Conference on Machine Learning, ICML, p. 1749-59 (2021)

12. Shu R, Bui H, Narui H, Ermon S. A DIRT-T Approach to unsupervised domain adaptation. In: Proceedings of the International Conference on Learning Representations, ICLR (2018)

13. Saito K, Kim D, Sclaroff S, Darrell T, Saenko K. Semi-supervised domain adaptation via minimax entropy. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, ICCV, p. 8050-8 (2019)

14. Nejjar I, Wang Q, Fink O. DARE-GRAM: Unsupervised domain adaptation regression by aligning inverse gram matrices. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR, p. 11744-54 (2023)

15. Hanneke S, Kpotufe S. On the value of target data in transfer learning. In: Proceedings of the International Conference on Neural Information Processing Systems, NeurIPS (2019)

16. French G, Mackiewicz M, Fisher M. Self-ensembling for visual domain adaptation. In: Proceedings of the International Conference on Learning Representations, ICLR (2017)

17. Singh A, Chakraborty S. Deep domain adaptation for regression. Development and Analysis of Deep Learning Architectures. 2020.

18. Li B, Wang Y, Zhang S, Li D, Keutzer K, Darrell T, et al. Learning invariant representations and risks for semi-supervised domain adaptation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR, p. 1104-13 (2021)

19. Roy S, Siarohin A, Sangineto E, Bulo SR, Sebe N, Ricci E. Unsupervised domain adaptation using feature-whitening and consensus loss. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR, p. 9471-80 (2019)

20. Long M, Zhu H, Wang J, Jordan MI. Unsupervised domain adaptation with residual transfer networks. In: Proceedings of the International Conference on Neural Information Processing Systems, NeurIPS (2016)

21. Goodfellow I, Pouget-Abadie J, Mirza M, Xu B, Warde-Farley D, Ozair S, et al. Generative adversarial nets. In: Proceedings of the International Conference on Neural Information Processing Systems, NeurIPS (2014)

22. Li L, Zhang Z. Semi-supervised domain adaptation by covariance matching. IEEE Transactions on Pattern Analysis and Machine Intelligence 41(11):2724-39 (2018)

23. Yu Y-C, Lin H-T. Semi-supervised domain adaptation with source label adaptation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR, p. 24100-9 (2023)

24. Li J, Li G, Shi Y, Yu Y. Cross-domain adaptive clustering for semi-supervised domain adaptation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR, p. 2505-14 (2021)

25. Cortes C, Mohri M. Domain adaptation in regression. In: Proceedings of the International Conference on Algorithmic Learning Theory, ALT, p. 308-23 (2011)

26. Wu J, He J, Wang S, Guan K, Ainsworth E. Distribution-informed neural networks for domain adaptation regression. In: Proceedings of the International Conference on Neural Information Processing Systems, NeurIPS (2022)

27. Dhaini M, Berar M, Honeine P, Van Exem A. Unsupervised domain adaptation for regression using dictionary learning. Knowledge-Based Systems 267:110439 (2023)

28. Goldberger AS. Econometric Theory. 1964.

29. Reddy TA, Andersen KK. An evaluation of classical steady-state off-line linear parameter estimation methods applied to chiller performance data. Hvac&R Research 8(1):101-24 (2002)

30. Sim S, Bae J, Kim SB. Robust semi-supervised regression for vehicle interior noise prediction. IEEE Access 12:60-72 (2024)

31. Dai W, Li X, Cheng K-T. Semi-supervised deep regression with uncertainty consistency and variational model ensembling via bayesian neural networks. In: Proceedings of the AAAI Conference on Artificial Intelligence, AAAI, p. 7304-13 (2023)

32. Bardes A, Ponce J, Lecun Y. VICReg: Variance-invariance-covariance regularization for self-supervised learning. In: Proceedings of the International Conference on Learning Representations, ICLR (2022)

33. Kwon B, Son H. Accurate path loss prediction using a neural network ensemble method. Sensors 24(1):304 (2024)

34. Fanelli G, Dantone M, Gall J, Fossati A, Gool L. Random forests for real time 3D face analysis. International Journal of Computer Vision 101(3):437–58 (2013)

35. Sherrah J, Gong S. Fusion of perceptual cues for robust tracking of head pose and position. Pattern Recognition 34(8):1565-72 (2001)

36. Gondal MW, Wuthrich M, Miladinovic D, Locatello F, Breidt M, Volchkov V, et al. On the transfer of inductive bias from simulation to the real world: a new disentanglement dataset. In: Proceedings of the International Conference on Neural Information Processing Systems, NeurIPS (2019)

# Appendix for WhiteCon: Semi-Supervised Domain Adaptation Regression Through Whitening Transform and Dual Consistency

Se Jin Sim<sup>[0009−0000−6028−1690]</sup> and Seoung Bum Kim<sup>[0000−0002−2205−8516]</sup>

School of Industrial and Management Engineering Korea University, Seoul, Republic of Korea {ssj259, sbkim1}@korea.ac.kr

![](images/37ba8a44d520360a5ca8528cfb50cc51b6f9e5893a5e41cfe91ceef10b4333a8.jpg)  
Fig. A1. Feature scale comparison across methods on BIWI F→M. The �-axis represents the different methods and the �-axis shows Frobenius norm values

## A-1. Feature Scale.

RSD argued based on experiments that maintaining a feature scale similar to supervised learning before domain adaptation yields optimal performance in domain adaptation regression. To investigate RSD's assertion about the importance of feature scale in domain adaptation regression, we conducted feature scale analysis using Frobenius norm. The Frobenius norm, an extension of the L2 norm for matrices, quantifies the scale of feature representations by taking the square root of the sum of the squared elements across all entries. Fig. A1 presents a feature scale comparison of methods on BIWI F→M. Although WhiteCon achieved the highest overall performance (as shown in Table 1), RSD showed a feature scale of 20.91, which is closest to that of S+T at 19.76. Additionally, MMD, which achieved the second-highest performance, had the secondlargest feature scale of 26.42 among the methods. This result contrasts with RSD’s assertion that optimal performance in domain adaptation regression is achieved when the feature scale is close to S+T, suggesting that a similar feature scale is not a strict requirement for achieving high performance.

Table A1. Training (source, labeled target, and unlabeled target), validation, and testing data splits across three benchmark datasets.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Case</td><td rowspan="2">Source</td><td colspan="4">Target</td></tr><tr><td>Labeled (K%) Unlabeled Validation</td><td></td><td></td><td>Testing</td></tr><tr><td rowspan="9">BIWI</td><td rowspan="5">F→M</td><td rowspan="5">3,447</td><td>50 (5%)</td><td rowspan="5">2,500</td><td rowspan="5">100</td><td rowspan="5">2,131</td></tr><tr><td>100 (10%)</td></tr><tr><td>200 (20%)</td></tr><tr><td>300 (30%)</td></tr><tr><td>1,000 (100%)</td></tr><tr><td rowspan="5">M→F</td><td rowspan="5">5,624</td><td>35 (5%)</td><td rowspan="5">1,500</td><td rowspan="5">70</td><td rowspan="5">1,247</td></tr><tr><td>70 (10%)</td></tr><tr><td>140 (20%)</td></tr><tr><td>210 (30%) 700 (100%)</td></tr><tr><td>44 (5%)</td></tr><tr><td rowspan="9">QMUL</td><td rowspan="5">F→M</td><td rowspan="5">1,504</td><td>89 (10%)</td><td rowspan="5">1,349</td><td rowspan="5">89</td><td rowspan="5">1,215</td></tr><tr><td>179 (20%)</td></tr><tr><td></td></tr><tr><td>268 (30%)</td></tr><tr><td>896 (100%)</td></tr><tr><td rowspan="5">M→F</td><td rowspan="5">2,500</td><td>15 (5%) 30 (10%)</td><td rowspan="5">600</td><td rowspan="5">30</td><td rowspan="5">569</td></tr><tr><td>61 (20%)</td></tr><tr><td>91 (30%)</td></tr><tr><td>305 (100%)</td></tr><tr><td></td></tr><tr><td rowspan="6">MPI3D</td><td>RL → RC</td><td rowspan="6">3,000</td><td>50 (5%)</td><td rowspan="6">2,000</td><td rowspan="6">100</td><td>10,437</td></tr><tr><td>RL → T</td><td>100 (10%)</td><td>10,348</td></tr><tr><td>RC → RL</td><td>200 (20%)</td><td>10,405 10,348</td></tr><tr><td>RC → T</td><td rowspan="3">300 (30%) 1,000 (100%)</td><td>10,405</td></tr><tr><td>T → RL</td><td>10,437</td></tr><tr><td>T → RC</td><td></td></tr></table>

## A-2. Datasets Settings

We used three benchmark datasets: BIWI and QMUL for head pose estimation, and MPI3D for object position estimation, under SSDAR settings. Table A1 shows the number of training, validation, and testing samples for each dataset in each case. We used 10% of the labeled target data for validation, which was based on SSDA methods for classification [1-3]. To simulate different proportions of labeled target data, denoted as K%, we used values including 5%, 10%, 20%, and 30%, as shown in Fig. 5.

Table A2. Details of hyperparameter search for each method. Optimal values were selected based on validation performance.
<table><tr><td colspan="2">MMD</td></tr><tr><td>Coefficient of MMD loss</td><td>{0.0005, 0.0001, 0.001, 0.01}</td></tr><tr><td colspan="2">DANN</td></tr><tr><td>Coefficient of DANN loss</td><td>{0.001, 0.01, 0.1, 1}</td></tr><tr><td colspan="2">RSD</td></tr><tr><td>Coefficient of RSD loss Coefficient of bases mismatch penalization</td><td>{0.0001, 0.001} {0.001, 0.01, 0.1}</td></tr><tr><td colspan="2">loss DARE-GRAM</td></tr><tr><td>Coefficient of angle loss</td><td>{0.001, 0.005, 0.05, 0.5, 0.1, 1}</td></tr><tr><td colspan="2">Coefficient of trade-off loss {0.00001, 0.0001, 0.0005, 0.001, 0.01 } DeepDAR</td></tr><tr><td>Coefficient of MMD loss Coefficient of semi-supervised loss</td><td>{0.001, 0.01, 0.1, 1, 10} {0.001, 0.01, 0.1, 1, 10}</td></tr><tr><td colspan="2">LIRR</td></tr><tr><td>Coefficient of DANN loss</td><td>{0.01, 10, 100}</td></tr><tr><td colspan="2">WhiteCon (ours)</td></tr><tr><td>Coefficient of prediction consistency regularization loss (λ1) Coefficient of feature variance consistency</td><td>{0.1, 0.2, 0.3, 0.4, 0.5}</td></tr></table>

![](images/060de444420409e71d02e921ab9d9539cb8f49fdb7c89412befc024454007be3.jpg)  
(a) BIWI F → M

![](images/5f9701f54f9e7432599e2f25fae2d07dbd009253a1d1d8f689e1539e8d6216c2.jpg)  
(b) QMUL F → M

![](images/3bc71d87ea271e43deea8c121ac1b05f3d2185efb0d19be08d244366fc7e5b1f.jpg)  
(c) MPI3D F → M  
Fig. A2. MAE results from WhiteCon hyperparameter search across three benchmark datasets. The �-axis represents $\lambda _ { 1 } ,$ , the �-axis represents $\lambda _ { 2 } ,$ , and the �-axis represents MAE.

## A-3. Hyperparameter Search

We conducted extensive hyperparameter optimization for all compared methods to ensure fair evaluation. Table A2 details the hyperparameter ranges examined for each method, with optimal values selected based on validation performance. For WhiteCon, we systematically explored $\lambda _ { 1 }$ and $\lambda _ { 2 }$ values in the range {0.1, 0.2, 0.3, 0.4, 0.5}. Fig. A2 shows the hyperparameter search results for WhiteCon. We conducted experiments on a representative case from each of the three benchmark datasets to determine optimal parameters. Based on MAE performance, the optimal hyperparameters were identified as BIWI (0.1, 0.1), QMUL (0.3, 0.5), and MPI3D (0.1, 0.1).

## A-4. Augmentation

The images from all benchmarks were resized to 224 × 224 pixels. All comparison methods were trained using weakly augmented images. Following the augmentation guidelines for image angle prediction from Hu et al. [4], our weak augmentation consisted of random cropping and Gaussian blur, while excluding transformations that could alter the angle, such as flips. The strong augmentation strategy was identical to the one used in FixMatch [5], a widely-used semi-supervised method. As mentioned in Section 3.4, the mixup augmentation is a linear combination of weakly and strongly augmented samples.

## References

1. Li J, Li G, Shi Y, Yu Y. Cross-domain adaptive clustering for semi-supervised domain adaptation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR, p. 2505-14 (2021)

2. Saito K, Kim D, Sclaroff S, Darrell T, Saenko K. Semi-supervised domain adaptation via minimax entropy. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, ICCV, p. 8050-8 (2019)

3. Yu Y-C, Lin H-T. Semi-supervised domain adaptation with source label adaptation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR, p. 24100-9 (2023)

4. Hu H-C, Wu X, Liu H, Wei T-R, Wu H-T. Full-range head pose geometric data augmentations. arXiv preprint arXiv:2408.01566 (2024)

5. Sohn K, Berthelot D, Carlini N, Zhang Z, Zhang H, Raffel CA, et al. FixMatch: Simplifying semi-supervised learning with consistency and confidence. In: Proceedings of the International Conference on Neural Information Processing Systems, NeurIPS (2020)
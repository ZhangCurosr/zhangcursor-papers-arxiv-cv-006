# Temporal-Aware Fusion for Robust Outdoor LiDAR Localization

Minghang Zhu\*   
Fujian Key Laboratory of Urban   
Intelligent Sensing and Computing,   
Xiamen University   
Xiamen, China   
mihoo@stu.xmu.edu.cn   
Chen Liu   
Fujian Key Laboratory of Urban   
Intelligent Sensing and Computing,   
Xiamen University   
Xiamen, China   
23020231154150@stu.xmu.edu.cn   
Zhijing Wang\*   
Fujian Key Laboratory of Urban   
Intelligent Sensing and Computing,   
Xiamen University   
Xiamen, China   
23020241154448@stu.xmu.edu.cn   
Yongshu Huang   
Fujian Key Laboratory of Urban   
Intelligent Sensing and Computing,   
Xiamen University   
Xiamen, China   
23020231154192@stu.xmu.edu.cn   
Sheng Ao\*   
Fujian Key Laboratory of Urban   
Intelligent Sensing and Computing,   
Xiamen University   
Xiamen, China   
aosh@xmu.edu.cn   
Yuxin Guo   
Fujian Key Laboratory of Urban   
Intelligent Sensing and Computing,   
Xiamen University   
Xiamen, China   
23020240157665@stu.xmu.edu.cn Wen Li   
School of Engineering Mathematics   
and Technology, University of Bristol Bristol, United Kingdom liwen777@stu.xmu.edu.cn   
Cheng Wang   
Fujian Key Laboratory of Urban   
Intelligent Sensing and Computing,   
Xiamen University   
Xiamen, China   
cwang@xmu.edu.cn

## Abstract

LiDAR relocalization aims to estimate the global 6-DoF pose of a sensor in the environment. However, existing regression-based approaches often encounter limitations in dynamic or ambiguous scenarios, as they typically prioritize single-frame inference, leaving the potential of spatio-temporal consistency across scans not fully explored. In this paper, we propose a Temporal-aware Localization framework (TempLoc) designed to enhance the robustness of outdoor localization by effectively modeling sequential consistency. Specifically, a Global Coordinate Estimation module is first introduced to predict point-wise global coordinates and associated uncertainties for each LiDAR scan. A Prior Coordinate Generation module is then presented to estimate inter-frame point correspondences by the attention mechanism. Lastly, an Uncertainty-Guided Coordinate Fusion module is deployed to integrate both predictions of point correspondence in an end-to-end fashion, yielding a more temporally consistent and accurate global 6-DoF pose. Experimental results on the NCLT and Oxford RobotCar benchmarks show that our TempLoc outperforms state-of-the-art methods by a large margin, demonstrating the effectiveness of temporal-aware correspondence modeling in LiDAR relocalization.

CCS Concepts « Computing methodologies — Vision for robotics.

Keywords   
LiDAR Relocalization, Scene Coordinate Regression, Temporal Con  
sistency, Uncertainty Estimation ACM Reference Format:   
Minghang Zhu, Zhijing Wang, Yuxin Guo, Chen Liu, Yongshu Huang, Wen Li, Sheng Ao, and Cheng Wang. 2026. Temporal-Aware Fusion for Robust Outdoor LiDAR Localization. In Proceedings of the 34th ACM International Conference on Multimedia (MM ’26), November 10-14, 2026, Rio de Janeiro, Brazil. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3767308. 3834933

## 1 Introduction

Accurate and robust LiDAR-based relocalization is a fundamental capability for autonomous systems [27, 58, 59] and robotics [36, 42, 48]. Given a pre-built 3D map, the goal of LiDAR-based localization is to estimate the global 6 degree of freedom (DoF) pose of sensor according to the captured real-time scans [2, 3, 35]. The primary difficulty lies in handling an environment with dynamic variations, occlusions, and sensor noise [1, 14, 24, 54].

Most existing LiDAR relocalization methods [7, 17] rely on a matching paradigm, where the input point cloud is aligned against a pre-built global map to estimate the current pose [26, 30]. While highly accurate, these methods require storing and querying largescale 3D maps, resulting in significant memory and communication overhead [13, 33, 37]. In contrast, recent regression-based methods [44, 45, 57] attempt to directly predict the absolute pose from a single point cloud scan using deep neural networks, bypassing the need for explicit map storage and achieving higher computational efficiency. These regression-based alternatives open up promising directions for deployment on resource-constrained platforms [53].

![](images/08880bd526964fd4ce013bba3a085e7e934bc7e5a73f2cf938ba9d5e4148b055.jpg)  
Figure 1: Mean position error comparisons on NCLT and QE-Oxford dataset. Our method (TempLoc) achieves superior localization accuracy on both datasets.

Nevertheless, the regression-based methods often struggle in complex or ambiguous environments [44, 51]. This is because single LiDAR scan is prone to measurement noise and observable geometric variations induced by spatial changes or dynamic objects. To alleviate this issue, a handful of work [55, 56] has been proposed by incorporating LiDAR sequences to enhance localization accuracy. However, they only roughly encode spatial-temporal global features, without explicitly modeling the point-level sequential consistency [12], which results in suboptimal localization performance.

In the 2D counterpart, recent advances in temporal relocalization have demonstrated that incorporating frame-to-frame correspondences at the pixel level can significantly improve performance [9, 49, 61]. For instance, KFNet [60] estimates per-pixel scene coordinates and model the temporal transition using optical flow, enabling recursive state estimation through Kalman filtering. While effective for 2D image, these approaches [8, 10, 50] cannot be directly extended to the 3D domain due to the unordered and irregular nature of point clouds in 3D scenes, coupled with challenges such as sparsity, occlusions, and interference from dynamic objects, temporal LiDAR-based localization faces unique difficulties.

This observation raises a critical question: can a similar paradigm be adapted to LiDAR relocalization? Motivated by this, we propose a new LiDAR relocalization framework that jointly performs measurement and prior estimation, integrating them through a learned fusion mechanism. Our method, named TempLoc, consists of three components: (1) the Global Coordinate Estimation that predicts per-point global coordinates and uncertainties from a single LIDAR scan, (2) the Prior Coordinate Generation that leverages a point cloud registration network to estimate inter-frame point-wise transitions, and (3) the Uncertainty-Guided Coordinate Fusion that employs an end-to-end differentiable fusion mechanism to generate accurate and temporally consistent point correspondences. In the end, the global 6-DoF pose is obtained via a RANSAC-based optimization using the refined 3D correspondences.

Benefiting from explicitly modeling the temporal evolution of scene coordinates, our method effectively mitigates the limitations of single-frame regression and significantly enhances localization robustness. As shown in Figure 1, our method achieves state-ofthe-art performance on two public available benchmarks. Notably, the localization accuracy of our method outperforms that of the strong baseline LightLoc [21] by nearly 30% on the NCLT [6] dataset. Overall, our contributions are three-fold:

e We propose a temporally-aware LiDAR relocalization framework that incorporates sequential constraints into point correspondence estimation by extending scene coordinate regression to the temporal domain.

e We introduce an uncertainty-aware coordinate estimation module to predicts per-point global coordinates from single LiDAR scan.

e Extensive experiments and ablations demonstrating the effectiveness of our method and providing the intuition for its architectural components.

## 2 Related Work

Map-based methods [19, 39] need to pre-build feature maps and perform localization via retrieval or registration, requiring substantial storage for large-scale outdoor deployments. In contrast, regression-based methods offer a map-free alternative by implicitly encoding maps within network parameters, enabling direct global pose prediction without any additional storage overhead. These regression-based methods can be further categorized into Absolute Pose Regression (APR) and Scene Coordinate Regression (SCR).

## 2.1 Absolute Pose Regression

Absolute pose regression aims to use deep networks to learn and memorize scene information, directly regressing the sensor’s global pose in an end-to-end manner. PoseNet [16], a classic solution for APR, applies a modified GoogleNet [38] to regress camera poses. The first proposed APR method for LiDAR is PointLoc [44], which uses PointNet++ [32] and a self-attention module to extract global features. PosePN [57] adopts a universal encoder and memoryaware regression to avoid redundant retraining and improve localization performance. HypLiLoc [43] combines 3D features and 2D spherical projection features to further improve performance. DiffLoc [22] introduces a diffusion model to obtain robust and accurate positioning through iterative denoising. These methods use single-frame data as input, which will results in many outliers.

Since sequential data can provide contextual information and time constraints, VLocNet [40] and its semantic variant VLoc-Net++ [34] use multi-task optimization strategy to learn shared features, LSG [49] integrates visual odometry constraints, and ViPR [29] integrates APR and relative pose estimation using LSTMs. For filtering and handling specific constraints, CoordiNet [28] eliminates positioning outliers based on Kalman filtering, while LiDARspecific schemes like STCLoc [56] and NIDALoc [55] apply spatialtemporal regularization and memory modules, respectively. Despite improved accuracy, these sequence-based methods rely on longterm sequence information, causing high computational overhead that hinders real-time performance.

![](images/b3eb719e736aff7e69b157d405cb3f0ec3e13a4e4068354986174bdf0f2e9aa9.jpg)  
Figure 2: Overview of the proposed framework. The Global Coordinate Estimation (GCE) module processes sequential point clouds from consecutive frames via a SCR module with uncertainty estimation, producing stable pose estimates. Subsequently, a Prior Coordinate Generation (PCG) module refines the pose estimates using soft correspondences. Then, Uncertainty-Guided Coordinate Fusion (UCF) module produces point clouds in the world coordinate system. Finally, the RANSAC algorithm is employed to solve for the point cloud localization pose.

## 2.2 Scene Coordinate Regression

Different from APR which directly regresses the pose, SCR regresses the point cloud coordinates in the world coordinate system and relies on RANSAC for pose estimation. SGLoc [23] applied SCR to LiDAR positioning for the first time and significantly improved the positioning performance. LiSA [51] applies diffusion-based knowledge distillation to enable the SCR network to have semantic understanding and focus on more important points. In order to reduce training time, LightLoc [21] proposes a universal encoder that substantially reduces training time without compromising model performance.

However, Scene Coordinate Regression (SCR) often yields numerous outliers in regressed point coordinates. Despite iterative denoising with RANSAC, these outliers can still compromise localization accuracy. To mitigate the impact of such outliers, we integrate uncertainty estimation. Currently, within a filter framework, we fuse temporal information from neighboring frames to achieve superior pose estimation and smoother trajectories.

## 3 Methodology

As shown in Figure 2, our method consists of three key modules: Global Coordinate Estimation (GCE), Prior Coordinate Generation (PCG), and Uncertainty-guided Coordinate Fusion (UCF). The GCE module produces 3D point correspondences for each LIDAR scan between the local and global coordinate systems, along with their associated uncertainty scores (Sec. 3.1). Simultaneously, the PCG module estimates 3D point correspondences between adjacent frames and provides corresponding credibility assessments (Sec. 3.2). Based on this, the UCF module integrates the outputs of the GCE and PCG to obtain refined 3D-3D correspondences between the local and global coordinate systems (Sec. 3.3). Finally, the global 6-DoF pose is computed through RANSAC.

![](images/8849962dd73ec1571279824efe41e084ec1ef18aef3bf7b6007c19ec5e6099a8.jpg)  
Figure 3: Global Coordinate Estimation Module. The GCE module takes sparse point cloud features extracted by the backbone network as input. It first extracts deep features through an initial residual block followed by two stacked residual blocks (ResBlock 0 and 1). Finally, three fully connected layers output the predicted point cloud in the world coordinate system along with the uncertainty for each point.

## 3.1 Global Coordinate Estimation

The SCR method is prone to producing inaccurate coordinate predictions in environments characterized by noise, dynamic disturbances, or insufficient structural information. However, existing LiDAR relocalization methods typically overlook the quality of per-point predictions. To address this, as shown in Fig 3, we incorporate a perpoint uncertainty estimation mechanism into the SCR architecture to assess the quality of each predicted global coordinate.

![](images/f2a227aad7979b0717deb7ea7ba7e1149f0e6f59aab5e70d968806030fff2ef6.jpg)  
Figure 4: Illustration of coordinate fusion. The GCE module produces the measurement estimation at ¢ time, while the PCG module provides the prior estimation at the same time. These two estimations are fused by the UCF module to generate more accurate point correspondences between the relative and world coordinate systems.

Prediction of uncertainty. For an input point cloud $\mathbf { P } \in \mathbb { R } ^ { N \times 3 }$ we utilize the LightLoc network [21] to regress scene coordinates $\mathbf { P } _ { p r e d } \in \mathbb { R } ^ { M \times 3 }$ in global reference frame. Simultaneously, the perpoint uncertainty score $\pmb { u } _ { p r e d } \in \mathbb { R } ^ { M \times 1 }$ can be predicted by shared MLPs. Note that, the ground-truth score $u _ { g t }$ is defined as:

$$
u _ { i } ^ { g t } = \left\{ \begin{array} { l l } { 0 , } & { \sum _ { i = 1 } ^ { M } \left\| p _ { i } ^ { p r e d } - p _ { i } ^ { g t } \right\| _ { 1 } < \tau , } \\ { 1 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{1}
$$

where $\pmb { p } ^ { g t }$ represents the ground-truth global coordinate of each point. The threshold r is not static and decays as the network training progresses.

This strategy is critical. In early training, a higher t permits the network to treat a broad range of predictions as confident, helping it focus on penalizing severe outliers and avoiding the trivial all-high-uncertainty solution. As the network converges and predictions improve, a smaller z tightens the criterion, encouraging finer-grained uncertainty estimation.

Loss Function. The total loss is composed of a regression loss and an uncertainty loss. The uncertainty prediction is supervised by a dedicated loss term, $\mathcal { L } _ { u n }$ We employ the Mean Squared Error loss, which computes the average of the squared differences between the predicted uncertainty $u _ { i } ^ { p r e d }$ and the ground truth label $u _ { i } ^ { g t }$

$$
\mathcal { L } _ { u n } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } ( u _ { i } ^ { p r e d } - u _ { i } ^ { g t } ) ^ { 2 } .\tag{2}
$$

Meanwhile, the primary regression objective, $\mathcal { L } _ { r e g } ,$ minimizes the mean distance L1 [11] between the predicted and ground truth coordinates:

$$
\mathcal { L } _ { r e g } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \left. \pmb { p } _ { i } ^ { p r e d } - \pmb { p } _ { i } ^ { g t } \right. _ { 1 } .\tag{3}
$$

$$
\mathcal { L } _ { G C E } = \mathcal { L } _ { r e g } + \mathcal { L } _ { u n } .
$$

The final loss $\mathcal { L } _ { G C E }$ is the sum of these two terms:

(4)

## 3.2 Prior Coordinate Generation

To effectively leverage temporal information, we design a prior coordinate generation module to generate a reliable world coordinate prior for the current frame. This process has two core stages: first, generating a soft correspondence and uncertainty in the current frame for each point in the previous frame using a registration network; second, using these correspondences to assign prior coordinates to points in the current frame through a neighborhood interpolation.

Soft Correspondence Generation. We adopt the PCAM [5] as the backbone network and additionally introduce a self-attention mechanism [41, 62] to enhance point-wise features within each point cloud, thereby generating discriminative feature descriptors.

For a point cloud $\mathbf { P } ^ { ( t ) }$ with initial features $\mathrm { F } ^ { \left( t \right) }$ we first compute its self-attention matrix $\mathbf { A } ^ { ( t , t ) }$ Each element $( \mathbf { A } ^ { ( t , t ) } ) _ { i j } ,$ representing the attention of point i on point $j ,$ is computed from the cosine similarity of intra-cloud feature pairs, followed by a Softmax normalization:

$$
s _ { i j } = \frac { \mathbf { \boldsymbol { F } } _ { i } ^ { ( t ) } \cdot ( \mathbf { \boldsymbol { F } } _ { j } ^ { ( t ) } ) ^ { T } } { | | \mathbf { \boldsymbol { F } } _ { i } ^ { ( t ) } | | _ { 2 } \cdot | | \mathbf { \boldsymbol { F } } _ { j } ^ { ( t ) } | | _ { 2 } } ,\tag{5}
$$

$$
( \mathbf { A } ^ { ( t , t ) } ) _ { i j } = { \frac { e ^ { s _ { i j } } } { \sum _ { k = 1 } ^ { N } e ^ { s _ { i k } } } } \quad .\tag{6}
$$

After obtaining the context-aware features, we use them to compute multi-level cross-attention from current frame $\mathrm { P } ^ { ( t ) }$ to previous frame $\mathbf { P } ^ { ( t - 1 ) }$ , We aggregate all L cross-attention matrices $\mathbf { A } _ { ( l ) } ^ { ( t - 1 , t ) }$ through element-wise multiplication to obtain a global attention matrix $\mathbf { A } _ { g l o b a l } ^ { ( t - 1 , t ) }$

$$
\mathbf { A } _ { g l o b a l } ^ { ( t - 1 , t ) } = \mathbf { A } _ { ( 1 ) } ^ { ( t - 1 , t ) } \odot \mathbf { A } _ { ( 2 ) } ^ { ( t - 1 , t ) } \odot \cdot \cdot \cdot \odot \mathbf { A } _ { ( L ) } ^ { ( t - 1 , t ) } .\tag{7}
$$

Based on this global attention matrix, we generate a soft correspondence m/ $( \pmb { p } _ { i } ^ { ( t - 1 ) } )$ in the current frame for each point $\pmb { p } _ { i } ^ { ( t - 1 ) }$ from the previous frame. This soft correspondence is a weighted average of all points in the current frame, with weights given by the corresponding row vector of the global attention matrix:

$$
m ( \pmb { p } _ { i } ^ { ( t - 1 ) } ) = \frac { \sum _ { j = 1 } ^ { N } ( \mathbf { A } _ { g l o b a l } ^ { ( t - 1 , t ) } ) _ { i j } \cdot \pmb { p } _ { j } ^ { ( t ) } } { \sum _ { k = 1 } ^ { N } ( \mathbf { A } _ { g l o b a l } ^ { ( t - 1 , t ) } ) _ { i k } } ,\tag{8}
$$

where $\pmb { p } _ { j } ^ { ( t ) }$ is the point in current frame cloud $\mathbf { P } ^ { ( t ) }$ , and $N$ is its number of points. This yields a set of pseudo-correspondence points $\{ m ( \pmb { p } _ { i } ^ { ( t - 1 ) } ) \}$ , where each point carries the temporal information of its origin point $\pmb { p } _ { i } ^ { ( t - 1 ) }$

Temporal Coordinate Propagation. The generated soft correspondence point m( $\mathbf { \Delta } _ { \pmb { P } _ { i } } ^ { ( t - 1 ) } )$ is a theoretical position and does not necessarily match any of the actual sampled points $\pmb { p } _ { j } ^ { ( t ) }$ in the current frame. To assign a prior coordinate to the actual points of the current frame, we use a neighborhood interpolation strategy.

Assume we have predicted the world coordinates of all points in the previous frame, denoted as $\{ \hat { \pmb { p } } _ { i } ^ { ( t - 1 ) } \}$ . For each actual point $\pmb { p } _ { j } ^ { ( t ) }$ in the current frame, we search for its k nearest neighbors within the pseudo-correspondence set $\{ m ( \pmb { p } _ { i } ^ { ( t - 1 ) } ) \}$ Let these k nearest pseudo-neighbors be $\{ m ( \pmb { p } _ { m _ { 1 } } ^ { ( t - 1 ) } ) , \ldots , m ( \pmb { p } _ { m _ { k } } ^ { ( t - 1 ) } ) \}$ . Since each pseudocorrespondence point $m ( { \pmb p } _ { m } ^ { ( t - 1 ) } )$ corresponds to a source point $\pmb { p } _ { m } ^ { ( t - 1 ) }$ from the previous frame, it can inherit the corresponding world coordinate $\hat { \pmb { p } } _ { m } ^ { ( t - 1 ) }$ Next, we compute the prior world coordinate $\tilde { \pmb { p } } _ { j } ^ { ( t ) }$ for the actual current-frame point $\pmb { p } _ { j } ^ { ( t ) }$ by performing a distance-weighted average of the world coordinates carried by these k pseudo-neighbors:

Table 1: Average translation error(m) and rotation error(°) on the QE-Oxford dataset. The best results are indicated in bold, and the second-best results are underlined.
<table><tr><td rowspan="7">QE-Oxford</td><td rowspan="2">Category</td><td>Methods</td><td>15-13-06-37</td><td>17-13-26-39</td><td>17-14-03-00</td><td>18-14-14-42</td><td>Average[m/°]</td></tr><tr><td>PointLoc (2022 Sens.J) [44]</td><td>10.75/2.36</td><td>11.07/2.21</td><td>11.53/1.92</td><td>9.82/2.07</td><td>10.79/2.14</td></tr><tr><td rowspan="6">APR</td><td>PosePN++ (2022 PR) [57]</td><td>4.54/1.83</td><td>6.44/1.78</td><td>4.89/1.55</td><td>4.64/1.61</td><td>5.13/1.69</td></tr><tr><td>PoseSOE (2022 PR) [57]</td><td>4.17/1.76</td><td>6.16/1.81</td><td>5.42/1.87</td><td>4.16/1.70</td><td>4.98/1.79</td></tr><tr><td>STCLoc (2022 TITS) [56]</td><td>5.14/1.27</td><td>6.12/1.21</td><td>5.32/1.08</td><td>4.76/1.19</td><td>5.34/1.19</td></tr><tr><td>NIDALoc (2023 TITS) [55]</td><td>3.71/1.50</td><td>5.40/1.40</td><td>3.94/1.30</td><td>4.08/1.30</td><td>4.28/1.38</td></tr><tr><td>HypLiLoc (2023 CVPR) [43]</td><td>5.03/1.46</td><td>4.31/1.43</td><td>3.61/1.11</td><td>2.61/1.09</td><td>3.89/1.27</td></tr><tr><td>SGLoc (2023 CVPR) [23]</td><td>1.79/1.67</td><td>1.81/1.76</td><td>1.33/1.59</td><td>1.19/1.39</td><td>1.53/1.60</td></tr><tr><td rowspan="4">SCR</td><td>LiSA (2024 CVPR) [51]</td><td>0.94/1.10</td><td>1.17/1.21</td><td>0.84/1.15</td><td>0.85/1.11</td><td>0.95/1.14</td></tr><tr><td>LightLoc (2025 CVPR) [21]</td><td>0.82/1.12</td><td>0.85/1.07</td><td>0.81/1.11</td><td>0.82/1.16</td><td>0.83/1.12</td></tr><tr><td>RALoc (2025 ICCV) [52]</td><td>1.52/1.28</td><td>1.71/1.27</td><td>1.37/1.20</td><td>1.41/1.23</td><td>1.51/1.24</td></tr><tr><td>TempLoc (Ours)</td><td>0.74/0.99</td><td>0.77/0.99</td><td>0.60/0.95</td><td>0.77/1.05</td><td>0.72/0.99</td></tr></table>

$$
\tilde { \pmb { p } } _ { j } ^ { ( t ) } = \sum _ { l = 1 } ^ { k } w _ { j , l } \cdot \hat { \pmb { p } } _ { m _ { l } } ^ { ( t - 1 ) } ,\tag{9}
$$

the weight is defined as $\begin{array} { r } { w _ { j , l } = \frac { 1 / d _ { j , l } } { \sum _ { s = 1 } ^ { k } 1 / d _ { j , s } } } \end{array}$ , where $d _ { j , l }$ represents the Euclidean distance from the actual point $\pmb { p } _ { j } ^ { ( t ) }$ to its I-th pseudoneighbor $m ( \pmb { p } _ { m _ { l } } ^ { ( t - 1 ) } )$ .

Loss Function. The propagated prior coordinates $\tilde { \mathbf { P } } ^ { ( t ) }$ and their associated uncertainties are jointly supervised by the loss $\mathcal { L } _ { P C G }$ which is analogous to the definitions in Equations (2)-(4).

## 3.3 Uncertainty-Guided Coordinate Fusion

For each LiDAR scan, we can obtain two groups of world coordinates: (1) the measurement estimation $( \hat { \mathbf { P } } ^ { ( t ) } , \hat { u } ^ { ( t ) } )$ comprising the coordinates and uncertainties directly predicted by the Global Coordinate Estimation, and (2) the prior estimation $( \tilde { \mathbf { P } } ^ { ( t ) } , \tilde { u } ^ { ( t ) } )$ obtained from the Prior Coordinate Generation. To obtain the final and more accurate coordinate estimation, we design the Uncertainty-Guided Coordinate Fusion module, as shown in Figure 4.

Inspired by the Kalman filtering [15, 47], which involves an adaptive weighted fusion based on the uncertainty of different sources, we design a data-driven and end-to-end differentiable fusion mechanism. Specifically, for each point, we concatenate its prior uncertainty $\tilde { u } _ { i } ^ { ( t ) }$ and measurement uncertainty $\hat { u } _ { i } ^ { ( t ) }$ and use a Softmax function to dynamically compute the fusion weights. The Softmax function ensures the weights are positive and sum to one, which provides a probabilistic interpretation for the fusion, and allowing the network to automatically learn how to balance the two sources based on data:

$$
[ \alpha _ { i } , \beta _ { i } ] = \mathrm { S o f t m a x } ( [ - \tilde { u } _ { i } ^ { ( t ) } , - \hat { u } _ { i } ^ { ( t ) } ] ) ,\tag{10}
$$

where $\alpha _ { i }$ is the weight assigned to the prior coordinate $\tilde { \pmb { p } } _ { i } ^ { ( t ) }$ and $\beta _ { i }$ is the weight for the measurement coordinate $\hat { \pmb { p } } _ { i } ^ { ( t ) }$ . The final fused coordinates $\bar { \pmb { p } } ^ { ( t ) }$ and uncertainty $\bar { u } ^ { ( t ) }$ are then computed by weighted summation:

$$
\bar { \pmb { p } } _ { i } ^ { ( t ) } = \alpha _ { i } \tilde { \pmb { p } } _ { i } ^ { ( t ) } + \beta _ { i } \hat { \pmb { p } } _ { i } ^ { ( t ) } ,\tag{11}
$$

$$
\bar { u } _ { i } ^ { ( t ) } = \alpha _ { i } \tilde { u } _ { i } ^ { ( t ) } + \beta _ { i } \hat { u } _ { i } ^ { ( t ) } .\tag{12}
$$

The weighted outputs are applied to both the coordinates and their associated uncertainty scores, resulting in the fused estimations with consistence and reliability.

Loss Function. During the training phase, the fused results are supervised by the loss $\mathcal { L } _ { F u s e } ,$ which is analogous to the losses in Equations (2)-(4). Finally, the overall loss of our framework can be formulated as:

$$
\mathcal { L } _ { f u l l } = \lambda _ { 1 } \mathcal { L } _ { G C E } + \lambda _ { 2 } \mathcal { L } _ { P C G } + \lambda _ { 3 } \mathcal { L } _ { F u s e } ,\tag{13}
$$

where $\lambda _ { 1 } , \lambda _ { 2 } ,$ and $\lambda _ { 3 }$ are set to 0.3, 0.3, and 0.4, respectively, and control the weights of the three losses.

## 4 Experiments

In this section, we first describe the experimental setup, including the benchmark datasets, training details, and baseline methods (Sec. 4.1). We then conduct extensive experiments to compare the proposed TempLoc with the state-of-the-art methods on typical benchmarks (Sec. 4.2). Lastly, a series of ablation studies are conducted (Sec. 4.3).

## 4.1 Experimental Setup

Benchmark Datasets. We conduct evaluation experiments on two widely-used benchmark datasets for outdoor LiDAR-based localization: the Oxford RobotCar Dataset [4] and the NCLT Dataset [6]. During evaluation, unified localization metrics are adopted: average translation error(m) and average rotation error(°).

Oxford RobotCar Dataset. This dataset was collected in central Oxford, UK, using a Velodyne-32 LiDAR sensor, with GPS/INS ground truth. The route length is 10km, covering an area of about 2.5km? of urban road environments. Consistent with other localization methods, we use four routes for training (11-14-02-26, 14-12-05- 52, 14-14-48-55,18-15-20-12) and four routes for testing (15-13-06-37,

Table 2: Average translation error(m) and rotation error (°) on the Oxford dataset. The best results are indicated in bold, and the second-best results are underlined.
<table><tr><td rowspan="7">Oxford</td><td>Category</td><td>Methods</td><td>15-13-06-37</td><td>17-13-26-39</td><td>17-14-03-00</td><td>18-14-14-42</td><td>Average[m/°]↓</td></tr><tr><td rowspan="6">APR</td><td>PointLoc (2022 Sen.J) [44] PosePN++ (2022 PR) [57]</td><td>12.42/2.26</td><td>13.14/2.50</td><td>12.91/1.92</td><td>11.31/1.98</td><td>12.45/2.17</td></tr><tr><td></td><td>9.59/1.92</td><td>10.66/1.92</td><td>9.01/1.51</td><td>8.44/1.71</td><td>9.43/1.77</td></tr><tr><td>PoseSOE (2022 PR) [57]</td><td>7.59/1.94</td><td>10.39/2.08</td><td>9.21/2.12</td><td>7.27/1.87</td><td>8.62/2.00</td></tr><tr><td>STCLoc (2022 TITS) [56]</td><td>6.93/1.48</td><td>7.55/1.23</td><td>7.44/1.24</td><td>6.13/1.15</td><td>7.01/1.28</td></tr><tr><td>NIDALoc (2023 TITS) [55]</td><td>5.45/1.40</td><td>7.63/1.56</td><td>6.68/1.26</td><td>4.80/1.18</td><td>6.14/1.35</td></tr><tr><td>HypLiLoc (2023 CVPR) [43]</td><td>6.88/1.09</td><td>6.79/1.29</td><td>5.82/0.97</td><td>3.45/0.84</td><td>5.74/1.05</td></tr><tr><td rowspan="4">SCR</td><td>SGLoc (2023 CVPR) [23] LiSA (2024 CVPR) [51]</td><td>3.01/1.91</td><td>4.07/2.07</td><td>3.37/1.89</td><td>2.12/1.66</td><td>3.14/1.88</td></tr><tr><td>LightLoc (2025 CVPR) [21]</td><td>2.36/1.29</td><td>3.47/1.43</td><td>3.19/1.34</td><td>1.95/1.23</td><td>2.74/1.32</td></tr><tr><td>RALoc (2025 ICCV) [52]</td><td>2.33/1.21</td><td>3.19/1.34</td><td>3.11/1.24</td><td>2.05/1.20</td><td>2.67/1.25</td></tr><tr><td>TempLoc (Ours)</td><td>3.19/4.10 2.22/1.05</td><td>3.87/3.96 3.08/1.12</td><td>3.32/3.87 2.94/1.06</td><td>2.59/3.71 1.99/1.07</td><td>3.24/3.91 2.55/1.07</td></tr></table>

Table 3: Average translation error(m) and rotation error(°) on the NCLT dataset. The best results are indicated in bold, and the second-best results are underlined. ' denotes the removal of certain erroneous test segments following the LightLoc, details are provided in Appendix Sec. 8.1.
<table><tr><td rowspan="7">NCLT</td><td>Category</td><td>Methods</td><td>2012-02-12</td><td>2012-02-19</td><td>2012-03-31</td><td>2012-05-26</td><td>Average[m/°]↓</td></tr><tr><td rowspan="6">APR</td><td>PointLoc (2022 Sens.J) [44]</td><td>7.23/4.88</td><td>6.31/3.89</td><td>6.71/4.32</td><td>9.55/5.21</td><td>7.45/4.58</td></tr><tr><td>PosePN++ (2022 PR)[57]</td><td>4.97/3.75</td><td>3.68/2.65</td><td>4.35/3.38</td><td>8.42/4.30</td><td>5.36/3.52</td></tr><tr><td>PoseSOE (2022 PR)[57]</td><td>13.09/8.05</td><td>6.16/4.51</td><td>5.24/4.56</td><td>13.27/7.85</td><td>9.44/6.24</td></tr><tr><td>STCLoc (2022 TITS) [56]</td><td>4.91/4.34</td><td>3.25/3.10</td><td>3.75/4.04</td><td>7.53/4.95</td><td>4.86/4.11</td></tr><tr><td>NIDALoc (2023 TITS) [55]</td><td>4.48/3.59</td><td>3.14/2.52</td><td>3.67/3.46</td><td>6.32/4.67</td><td>4.40/3.56</td></tr><tr><td>HypLiLoc (2023 CVPR) [43]</td><td>1.71/3.56</td><td>1.68/2.69</td><td>1.52/2.90</td><td>2.29/3.34</td><td>1.80/3.12</td></tr><tr><td rowspan="5">SCR</td><td rowspan="5"></td><td>SGLoc (2023 CVPR) [23]</td><td>1.20/3.08</td><td>1.20/3.05</td><td>1.12/3.28</td><td>3.48/4.43</td><td>1.75/3.46</td></tr><tr><td>LiSA (2024 CVPR) [51]</td><td>0.97/2.23</td><td>0.91/2.09</td><td>0.87/2.21</td><td>3.11/2.72</td><td>1.47/2.31</td></tr><tr><td>LightLoc (2025 CVPR) [21]</td><td>0.98/2.76</td><td>0.89/2.51</td><td>0.86/2.67</td><td>3.10/3.26</td><td>1.46/2.80</td></tr><tr><td>RALoc (2025 ICCV) [52]</td><td>1.61/4.71</td><td>1.61/4.88</td><td>1.58/4.71</td><td>3.51/5.40</td><td>2.07/4.92</td></tr><tr><td>TempLoc (Ours)</td><td>0.74/2.48</td><td>0.74/2.22</td><td>0.68/2.35</td><td>2.04/3.20</td><td>1.05/2.56</td></tr></table>

17-13-26-39, 17-14-03-00, 18-14-14-42). QE-Oxford [23] represents an enhanced version of the original Oxford Radar RobotCar dataset, where ground-truth poses are refined and corrected through trajectory alignment techniques, thereby yielding more accurate and reliable experimental results. We discuss this in detail in Appendix Sec. 8.2.

NCLT Dataset. This dataset was collected at the University of Michigan’s North Campus using a Segway robotic platform equipped with a Velodyne-32 LiDAR sensor. It encompasses both indoor and outdoor environments with seasonal variations. Ground truth is obtained using GPS refined by SLAM techniques. The average route length is approximately 5.5km. Following SGLoc [23], we select four routes for training (2012-01-22, 2012-02-02, 2012-02-18, 2012-05- 11) and four routes for testing (2012-02-12, 2012-02-19, 2012-03-31, 2012-05-26).

Training Details. The proposed TempLoc method is implemented with PyTorch[31]. To ensure fairness, the comparison methods utilize the officially provided code and pre-trained models. During training, the batch size is set to 64, and the Adam[18, 46] optimizer is used with an initial learning rate of 0.001. For the Oxford dataset, the point cloud is sampled with a voxel size of 0.25, while for the NCLT dataset, the voxel size is set to 0.3. The parameter T is related to learning rate decay, we reduce t (Equation (1)) by x0.7 every 6 epochs. All experiments are conducted on the platform with Intel Xeon CPU@2.30GHZ with two NVIDIA RTX 3090Ti GPUs. Further details are in the supplementary material.

Baselines. The proposed TempLoc method is compared with several state-of-the-arts, including both APR and SCR methods. Specifically, the APR baselines contain single-frame methods such as PointLoc [44], PosePN++ [57], PoseSOE [57], and HypLiLoc [43], as well as temporal approaches like STCLoc [56] and NIDALoc [55]. For SCR methods, we consider SGLoc [23], LiSA [51], LightLoc [21], and RALoc [52]. To verify the advantages of our map-free method over retrieval-based localization, we also compare our approach with BEVplace++ [25].

## 4.2 Evaluation

Evaluation on the Oxford Dataset. We first evaluate TempLoc on the QE-Oxford dataset. As shown in Table 1, TempLoc outperforms existing methods in average accuracy. Compared with the baseline LightLoc, it reduces translation and rotation errors from 0.83m / 1.12° to 0.72m / 0.99°. Against temporal-constraint-based methods STCLoc and NIDALoc, TempLoc improves translation error from 4.28m to 0.72m and orientation error from 1.38° to 0.99° using only two consecutive frames, demonstrating substantial gains with minimal temporal dependency.

![](images/c959bb19b0ae90a91c89d58a98e13699f416029780bf4df71013d84a01e60d9c.jpg)  
SGLoc(1.81m,1.76°)

![](images/4c3baa9106f3cdf6441ab6a49bb1ea856f8c1edece01ace3a64878a80ee056d6.jpg)  
LiSA(1.17m,1.21°)

![](images/7e75dffc7fa3d2877d573ed831f474f5aa30ce5f4176920e46a185c4e8f0bd96.jpg)  
LightLoc(0.85m,1.07°)

![](images/62bc8ca75f120835849ef849d5a6a09322832511159c7808e3bbee6720632583.jpg)  
RALoc(1.71m,1.27°)

![](images/d299348d445b027d632457e8fa7165bc408db97678a13bcbe9a3daea0353db63.jpg)  
TempLoc(0.77m,0.99’)

Figure 5: Visualization of localization results on the QE-Oxford dataset. The blue trajectories represent the predicted results, while black lines denote the ground truth. The best results are indicated in bold, and the second-best results are underlined.  
![](images/f107c71f348c142e9d9e0534283913c318e8f6cc83f478148361333926edce83.jpg)  
SGLoe(1.12m,3.28°)

![](images/8cb3504da11d1286c39e33746edf353ab9d54459d1f7242e2168343ff4df1938.jpg)  
LiSA(0.87m,2.21°)

![](images/0b5bb152a36403d69f41ed60953e59917df352c4b8e7bb7183f558208a174fd6.jpg)  
LightLoc(0.86m,2.67°)

![](images/526ac6f3968c15beb4c67c19e66afe6e3596bd1b3cc612cfdee062eac95a3e22.jpg)  
RALoc(1.58m,4.71°)

![](images/a1389153d1b055b509f0bbc3af15488be05734d3c33a53cfd47dbe4551124288.jpg)  
TempLoc(0.68m,2.35°)  
Figure 6: Visualization of localization results on the NCLT dataset. The blue trajectories represent the predicted results, while black lines denote the ground truth. The best results are indicated in bold, and the second-best results are underlined.

Table 4: Real-time, Overhead and Performance on NCLT.
<table><tr><td>Method</td><td>Real-time (4)</td><td>Test GPU (↓)</td><td>Storage (↓) Model+Map</td><td>Error(↓) [m/°]</td><td>Recall@1 &lt;5m(↓)</td></tr><tr><td>SGLoc(23&#x27;CVPR) [23]</td><td>75ms</td><td>5.2GB</td><td>425MB+0MB</td><td>3.48/4.43</td><td>95.2%</td></tr><tr><td>LightLoc(25&#x27;CVPR) [21]</td><td>48ms</td><td>2.1GB</td><td>70MB+0MB</td><td>3.10/3.26</td><td>96.5%</td></tr><tr><td>BEVplace++(25&#x27;TRO) [25]</td><td>54ms</td><td>1.95GB</td><td>17MB+841MB</td><td>11.83/4.64</td><td>92.7%</td></tr><tr><td>TempLoc (Ours)</td><td>68ms</td><td>2.3GB</td><td>77MB+0MB</td><td>2.04/3.20</td><td>98.3%</td></tr></table>

Figure 5 shows the localization results on the QE-Oxford dataset using trajectory 17-13-26-39. SGLoc, LiSA, RALoc, and LightLoc all suffer from noticeable jumps. In contrast, TempLoc produces a smooth trajectory without erroneous jumps, highlighting stronger robustness via temporally-aware feature fusion.

Due to QE-Oxford’s adoption of global trajectory alignment from SGLoc [23], the ground truth error in the Oxford dataset was corrected. To ensure experimental fairness, we re-conducted experiments on the Oxford localization baseline, as shown in Table 2. While all methods experienced a decline in accuracy, TempLoc consistently achieves the best performance.

Evaluation on the NCLT Dataset. We further evaluate TempLoc on the NCLT dataset. As shown in Table 3, TempLoc outperforms the second-best method LightLoc with a 28% translation improvement (1.46m — 1.05m). Although its rotation accuracy is 2.56°, lower than LiSA (2.31°), TempLoc still outperforms LiSA by 29% in translation accuracy $( 1 . 4 7 m  1 . 0 5 m )$ . Note that LiSA relies on a heavy semantic segmentation model [20] during training, yet its gain is far inferior to our temporal-aware fusion. Figure 6 visualizes the 2012-03-31 trajectory. SGLoc, LiSA, LightLoc, and RALoc exhibit severe jumps, whereas TempLoc yields a smooth trajectory with top accuracy (0.68m), demonstrating robust performance in complex campus environments.

Table 5: Localization Performance in Dynamic Scenes on the QE-Oxford Dataset (translation/rotation error in $[ \mathbf { m } / ^ { \circ } ] )$ |.
<table><tr><td>Method</td><td>Scene1</td><td>Scene2</td><td>Scene3</td><td>Scene4</td></tr><tr><td>SGLoc (2023 CVPR) [23]</td><td>2.31/1.23</td><td>3.15/1.56</td><td>1.77/1.06</td><td>7.61/2.05</td></tr><tr><td>LiSA (2024 CVPR) [51]</td><td>1.97/1.43</td><td>1.87/1.33</td><td>1.44/1.01</td><td>2.07/1.65</td></tr><tr><td>LightLoc (2025 CVPR) [21]</td><td>2.20/1.28</td><td>1.09/0.92</td><td>0.69/0.98</td><td>1.77/1.15</td></tr><tr><td>RALoc (2025 ICCV) [52]</td><td>1.93/1.34</td><td>1.00/1.04</td><td>1.48/1.22</td><td>3.47/1.78</td></tr><tr><td>TempLoc (Ours)</td><td>1.00/0.97</td><td>0.44/0.63</td><td>0.50/0.54</td><td>0.40/0.34</td></tr></table>

Real-time, Overhead and Performance. We further compare TempLoc with BEVplace++ [25] and other map-free baselines on NCLT. TempLoc reduces feature storage to 0 MB compared to 841 MB for BEVplace++, and maintains a latency of 68 ms within the 100 ms real-time limit using a 2.3 GB GPU footprint. Crucially, TempLoc outperforms BEVplace++, reducing translation error from 11.837 to 2.04m and achieving 98.3% recall@1 within 5m, demonstrating a robust trade-off between efficiency and localization accuracy.

Localization Analysis under Dynamic Interference. As shown in Table 5, we evaluate localization performance on the QE-Oxford sequences 17-14-03-00, which contain multiple dynamic vehicle scenarios. SGLoc, LightLoc, and RALoc suffer from significant performance degradation in the presence of moving vehicles, whereas TempLoc remains unaffected and maintains stable accuracy. This demonstrates the strong robustness of TempLoc in dynamic environments. We discuss this in detail in Appendix Sec.9.

Visualization Comparison and Analysis. As shown in Figure 7, we present a qualitative comparison on the QE-Oxford and NCLT datasets. Closer alignment between the predicted and groundtruth point clouds indicates higher localization accuracy. For the

![](images/520ec12c8d95ba30c0ab23feb6d4eae6bdc69facaf89737f4b01b0070e377153.jpg)  
Figure 7: Localization Visualization Comparison. Experiments were conducted on the QE-Oxford dataset and NCLT datasets. The ground-truth point clouds are depicted in black, while the point clouds at the predicted localization are rendered in green.

![](images/69dd9a31e38bf95aa53957c99e282910491e9e4a7d2630c1ecbdd57ee767a55c.jpg)  
Figure 8: Uncertainty estimation visualization in dynamic environments. We present the top-down view of the point cloud, with dynamic vehicles highlighted using black bounding boxes.

Table 6: Ablation study on different modules. UE: Uncertainty Estimation, PCG+UCF: Prior Coordinate Generation and Uncertainty-guided Fusion. (translation/rotation error in [m/°]) |.
<table><tr><td>UE</td><td>PCG+UCF</td><td>Oxford</td><td>QE-Oxford</td><td>NCLT</td></tr><tr><td></td><td></td><td>2.67m/1.25°</td><td>0.83m/1.12°</td><td>1.46m/2.80°</td></tr><tr><td>√</td><td></td><td>2.56m/1.09°</td><td>0.74m/1.00°</td><td>1.30m/2.41°</td></tr><tr><td>√</td><td>√</td><td>2.55m/1.07°</td><td>0.72m/0.99°</td><td>1.05m/2.56°</td></tr></table>

QE-Oxford dataset (the left of Figure 7), we select a challenging traffic intersection scene containing numerous dynamic vehicles. These moving objects introduce severe interference to the localization models, resulting in noticeable ghosting and misalignment in the predicted point clouds of all competing methods when compared with the ground truth, whereas TempLoc achieves near-perfect overlap with ground truth. In the densely vegetated NCLT scene (the right of Figure 7), TempLoc leverages its uncertainty estimation module to retain high-certainty structural points, outperforming baseline in point-cloud alignment.

## 4.3 Ablation Study

Uncertainty Estimation Visualization. To verify that TempLoc can learn more robust features against dynamic objects through temporal awareness and uncertainty estimation, we visualize the point clouds under dynamic scenes in Figure 8. Specifically, Vehicles in the raw point cloud are highlighted with black bounding boxes, and static structures such as buildings have black borders (Figure 8(a)). Figure 8(b) shows that stable structures like buildings exhibit high certainty, while dynamic vehicles show high uncertainty. These results indicate that the network effectively suppresses dynamic object interference, improving localization robustness.

Ablation Study on Different Modules. To validate the effectiveness of the uncertainty estimation (UE) module and the fusion module (PCG+UCF) in TempLoc, we conduct ablation experiments on the Oxford, QE-Oxford, and NCLT datasets, as shown in Table 6. Starting from the baseline, incorporating the uncertainty estimation module (UE) improves translation and rotation accuracy by 4.1%/10.8%/10.9% and 12.8%/10.7%/13.9% across the three datasets, respectively. The relatively modest translation improvement on Oxford is attributed to inherent ground-truth noise in that benchmark. The PCG+UCF module yields smaller gains on Oxford and QE-Oxford, as these datasets contain more stable structures. However, in the complex campus environment of NCLT, PCG+UCF significantly reduces translation error compared with the UE-only variant, boosting overall localization performance by 19% (from 1.30m to 1.05m). The ablation results demonstrate that each component plays a critical role in TempLoc.

## 5 Conclusion

In this paper, we present TempLoc, a novel LiDAR relocalization framework designed to overcome the robustness limitations of traditional single-frame methods in challenging dynamic outdoor environments. By explicitly modeling sequential consistency across consecutive LiDAR scans, TempLoc follows a measurement-predictionfusion paradigm for robust 6-DoF pose estimation without dense map storage or pre-built databases. Extensive experiments conducted on the Oxford, QE-Oxford, and NCLT benchmarks demonstrate that TempLoc consistently outperforms state-of-the-art methods in both translation and rotation accuracy. Crucially, TempLoc transitions from relying solely on instantaneous noisy observations to integrating historical and current information, enhancing robustness in complex outdoor scenarios with dynamic objects.

## Acknowledgments

This work was partially supported by the National Natural Science Foundation of China (No. 62501502).

## References

[1] Sheng Ao, Yulan Guo, Qingyong Hu, Bo Yang, Andrew Markham, and Zengping Chen, 2022. You only train once: Learning general and distinctive 3D local descriptors. IEEE Transactions on Pattern Analysis and Machine Intelligence 45, 3 (2022), 3949-3967.

[2 <sub>s</sub>) <sub>Sheng</sub> <sub>Ao,</sub> <sub>Qingyong</sub> <sub>Hu,</sub> <sub>Hanyun</sub> <sub>Wang,</sub> <sub>Kai</sub> <sub>Xu,</sub> <sub>and</sub> <sub>Yulan</sub> <sub>Guo.</sub> <sub>2023.</sub> <sub>Buffer:</sub> Balancing accuracy, efficiency, and generalizability in point cloud registration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 1255-1264.

Sheng Ao, Qingyong Hu, Bo Yang, Andrew Markham, and Yulan Guo. 2021. SpinNet: Learning a General Surface Descriptor for 3D Point Cloud Registration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 11753-11762.

Dan Barnes, Matthew Gadd, Paul Murcutt, Paul Newman, and Ingmar Posner. 2020. The Oxford Radar RobotCar Dataset: A Radar Extension to the Oxford RobotCar Dataset. In 2020 IEEE International Conference on Robotics and Automation (ICRA). 6433-6438.

[5 <sub>a</sub>y Anh-Quan Cao, Gilles Puy, Alexandre Boulch, and Renaud Marlet. 2021. PCAM: Product of Cross-Attention Matrices for Rigid Registration of Point Clouds. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 13229- 13238.

[6 e Nicholas Carlevaris-Bianco, Arash K Ushani, and Ryan M Eustice. 2016. University of Michigan North Campus long-term vision and lidar dataset. The International journal of Robotics Research 35, 9 (2016), 1023-1035.

[7 o <sub>Daniele</sub> <sub>Cattaneo,</sub> <sub>Matteo</sub> <sub>Vaghi,</sub> <sub>and</sub> <sub>Abhinav</sub> <sub>Valada.</sub> <sub>2022.</sub> <sub>Lcdnet:</sub> <sub>Deep</sub> <sub>loop</sub> closure detection and point cloud registration for lidar slam. IEEE Transactions on Robotics 38, 4 (2022), 2074-2093.

[8 <sub>e</sub>) Shuai Chen, Tommaso Cavallari, Victor Adrian Prisacariu, and Eric Brachmann. 2024. Map-relative pose regression for visual re-localization. In Proceedings of th IEEE/CVF Conference on Computer Vision and Pattern Recognition. 20665-20674.

o<sub>a</sub>) Ronald Clark, Sen Wang, Andrew Markham, Niki Trigoni, and Hongkai Wen. 2017. VidLoc: A Deep Spatio-Temporal Model for 6-DoF Video-Clip Relocalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2652-2660.

[10] Siyan Dong, Shuzhe Wang, Shaohui Liu, Lulu Cai, Qingnan Fan, Juho Kannala, and Yanchao Yang. 2025. Reloc3r: Large-Scale Training of Relative Camera Pose Regression for Generalizable, Fast, and Accurate Visual Localization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 16739-16752.

[11] Ross Girshick. 2015. Fast R-CNN. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 1440-1448.

[12] Peiyu Guan, Zhiqiang Cao, Junzhi Yu, Chao Zhou, and Min Tan. 2021. Scene coordinate regression network with global context-guided spatial feature transformation for visual relocalization. IEEE Robotics and Automation Letters 6, 3 (2021), 5737-5744.

[13] Shengyu Huang, Zan Gojcic, Mikhail Usvyatsov, Andreas Wieser, and Konrad Schindler. 2021. Predator: Registration of 3d point clouds with low overlap. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 4265-4274.

[14] Yongshu Huang, Chen Liu, Minghang Zhu, Sheng Ao, Chenglu Wen, and Cheng Wang. 2025. Difflo: Semantic-aware lidar odometry with diffusion-based refinement. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 17050-17059.

[15] R. E. Kalman. 1960. A New Approach to Linear Filtering and Prediction Problems. Journal of Basic Engineering 82, 1 (March 1960), 35-45.

[16] Alex Kendall, Matthew Grimes, and Roberto Cipolla. 2015. PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 2938-2946.

[17] Giseop Kim, Sunwook Choi, and Ayoung Kim. 2021. Scan context++: Structural place recognition robust to rotation and lateral variations in urban environments. IEEE Transactions on Robotics 38, 3 (2021), 1856-1874.

[18] Diederik P. Kingma and Jimmy Ba. 2015. Adam: A method for stochastic optimization. In International Conference on Learning Representations.

[19] Jacek Komorowski. 2021. Minkloc3d: Point cloud based large-scale place recognition. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 1789-1798.

[20] Xin Lai, Yukang Chen, Fanbin Lu, Jianhui Liu, and Jiaya Jia. 2023. Spherical Transformer for LiDAR-Based 3D Recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 17545-17555.

[21] Wen Li, Chen Liu, Shangshu Yu, Dunqiang Liu, Yin Zhou, Siqi Shen, Chenglu Wen, and Cheng Wang. 2025. LightLoc: Learning Outdoor LiDAR Localization at Light Speed. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 6680-6689.

[22] Wen Li, Yuyang Yang, Shangshu Yu, Guosheng Hu, Chenglu Wen, Ming Cheng, and Cheng Wang. 2024. DiffLoc: Diffusion Model for Outdoor LiDAR Localization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 15045-15054.

[23] Wen Li, Shangshu Yu, Cheng Wang, Guosheng Hu, Siqi Shen, and Chenglu Wen. 2023. SGLoc: Scene Geometry Encoding for Outdoor LiDAR Localization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 9286-9295.

[24 Chen Liu, Wen Li, Yongshu Huang, Minghang Zhu, Yuyang Yang, Dunqiang Liu, Sheng Ao, and Cheng Wang. [n. d.]. RCP-LO: A Relative Coordinate Prediction Framework for Generalizable Deep LiDAR Odometry. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40. 7078-7086.

[25 Lun Luo, Si-Yuan Cao, Xiaorui Li, Jintao Xu, Rui Ai, Zhu Yu, and Xieyuanli Chen. 2025. Bevplace++: Fast, robust, and lightweight lidar global localization for unmanned ground vehicles. IEEE Transactions on Robotics 41 (2025), 4479-4498.

[26 Lun Luo, Shuhang Zheng, Yixuan Li, Yongzhi Fan, Beinan Yu, Si-Yuan Cao, Junwei Li, and Hui-Liang Shen. 2023. BEVPlace: Learning LiDAR-based place recognition using bird’s eye view images. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 8666-8675.

Junyi Ma, Jun Zhang, Jintao Xu, Rui Ai, Weihao Gu, and Xieyuanli Chen. 2022. Overlaptransformer: An efficient and yaw-angle-invariant transformer network for lidar-based place recognition. IEEE Robotics and Automation Letters 7, 3 (2022), 6958-6965.

Arthur Moreau, Nathan Piasco, Dzmitry Tsishkou, Bogdan Stanciulescu, and Arnaud de La Fortelle. 2022. CoordiNet: uncertainty-aware pose regressor for reliable vehicle localization. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 1848-1857.

Felix Ott, Tobias Feigl, Christoffer Loffler, and Christopher Mutschler. 2020. ViPR: Visual-Odometry-aided Pose Regression for 6DoF Camera Localization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops. 187-198.

Yue Pan, Xingguang Zhong, Louis Wiesmann, Thorbjorn Posewsky, Jens Behley, and Cyrill Stachniss. 2024. PIN-SLAM: LiDAR SLAM using a point-based implicit neural representation for achieving global map consistency. IEEE Transactions on Robotics 40 (2024), 4045-4064.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. 2019. PyTorch: An Imperative Style, High-Performance Deep Learning Library. In Advances in Neural Information Processing Systems, Vol. 32. Curran Associates, Inc.

Charles R. Qi, Li Yi, Hao Su, and Leonidas J. Guibas. 2017. PointNet++: deep hierarchical feature learning on point sets in a metric space. In Advances in Neural Information Processing Systems, Vol. 30.

[33 Zheng Qin, Hao Yu, Changjian Wang, Yulan Guo, Yuxing Peng, and Kai Xu. 2022. Geometric Transformer for Fast and Robust Point Cloud Registration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 11133-11142.

[34 Noha Radwan, Abhinav Valada, and Wolfram Burgard. 2018. VLocNet++: Deep Multitask Learning for Semantic Visual Localization and Odometry. IEEE Robotics and Automation Letters 3, 4 (2018), 4407-4414.

[35 Yaqi Shen, Le Hui, Haobo Jiang, Jin Xie, and Jian Yang. 2022. Reliable Inlier Evaluation for Unsupervised Point Cloud Registration. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 36. 2198-2206.

oa Chenghao Shi, Xieyuanli Chen, Junhao Xiao, Bin Dai, and Huimin Lu. 2024. Fast and accurate deep loop closing and relocalization for reliable lidar slam. IEEE Transactions on Robotics 40 (2024), 2620-2640.

(37 Linus Svarm, Olof Engqvist, Fredrik Kahl, and Magnus Oskarsson. 2017. City-scale localization for cameras with known vertical direction. IEEE Transactions on Pattern Analysis and Machine Intelligence 39, 7 (2017), 1455-1461.

[38 Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. 2015. Going deeper with convolutions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 1-9.

[39 Mikaela Angelina Uy and Gim Hee Lee. 2018. Pointnetvlad: Deep point cloud based retrieval for large-scale place recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 4470-4479.

Abhinav Valada, Noha Radwan, and Wolfram Burgard. 2018. Deep Auxiliary Learning for Visual Localization and Odometry. In 2018 IEEE international conference on robotics and automation (ICRA). 6939-6946.

[41 Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, L ukasz Kaiser, and Illia Polosukhin. 2017. Attention is All you Need. In Advances in Neural Information Processing Systems, Vol. 30. Curran Associates, Inc.

[42 Changwei Wang, Shunpeng Chen, Yukun Song, Rongtao Xu, Zherui Zhang, Jiguang Zhang, Haoran Yang, Yu Zhang, Kexue Fu, Shide Du, et al. 2025. Focus on Local: Finding Reliable Discriminative Regions for Visual Place Recognition. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 39. 7536-7544.

[43 Sijie Wang. Qiyu Kang, Rui She, Wei Wang, Kai Zhao, Yang Song, and Wee Peng Tay. 2023. HypLiLoc: Towards Effective LiDAR Pose Regression with Hyperbolic Fusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5176-5185.

[44] Wei Wang, Bing Wang, Peijun Zhao, Changhao Chen, Ronald Clark, Bo Yang, Andrew Markham, and Niki Trigoni. 2022. PointLoc: Deep Pose Regressor for LiDAR Point Cloud Localization. IEEE Sensors Journal 22, 1 (2022), 959-968.

[45] Jianshi Wu, Minghang Zhu, Dunqiang Liu, Wen Li, Sheng Ao, Siqi Shen, Chenglu Wen, and Cheng Wang. 2026. LEADER: Learning Reliable Local-to-Global Correspondences for LiDAR Relocalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 9932-9942.

[46] Haoning Xi, Zhiqi Shao, David A Hensher, John D Nelson, Huaming Chen, and Kasun Wijayaratna. 2025. A multi-task Transformer with mixture-of-experts for personalized periodic predictions of individual travel behavior in multimodal public transport. Transportation Research Part C: Emerging Technologies 179 (2025), 105287.

[47] Haoning Xi, Yan Wang, Zhiqi Shao, Xiang Zhang, and S.Travis Waller. 2024. Optimizing mobility resource allocation in multiple MaaS subscription frameworks: A group method of data handling-driven self-adaptive harmony search algorithm. Annals of Operations Research (2024).

[48] Xuecheng Xu, Sha Lu, Jun Wu, Haojian Lu, Qiuguo Zhu, Yiyi Liao, Rong Xiong, and Yue Wang. 2023. Ring++: Roto-translation invariant gram for global localization on a sparse scan map. IEEE Transactions on Robotics 39, 6 (2023), 4616-4635.

[49] Fei Xue, Xin Wang, Zike Yan, Qiuyuan Wang, Junqiu Wang, and Hongbin Zha. 2019. Local Supports Global: Deep Camera Relocalization With Sequence Enhancement. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 2841-2850.

[50] Shen Yan, Yu Liu, Long Wang, Zehong Shen, Zhen Peng, Haomin Liu, Maojun Zhang, Guofeng Zhang, and Xiaowei Zhou. 2023. Long-Term Visual Localization with Mobile Sensors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 17245-17255.

[51] Bochun Yang, Zijun Li, Wen Li, Zhipeng Cai, Chenglu Wen, Yu Zang, Matthias Muller, and Cheng Wang. 2024. LiSA: LiDAR Localization with Semantic Awareness. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 15271-15280.

[52] Yuyang Yang, Wen Li, Sheng Ao, Qingshan Xu, Shangshu Yu, Yu Guo, Yin Zhou, Siqi Shen, and Cheng Wang. 2025. RALoc: Enhancing Outdoor LiDAR Localization via Rotation Awareness. In Proceedings of the IEEE/CVF International

Conference on Computer Vision. 3304-3313.

[53 Huan Yin, Xuecheng Xu, Sha Lu, Xieyuanli Chen, Rong Xiong, Shaojie Shen, Cyrill Stachniss, and Yue Wang. 2024. A survey on global lidar localization: Challenges, advances and open problems. International Journal of Computer Vision 132, 8 (2024), 3139-3171.

[54 <sub>Peng</sub> <sub>Yin,</sub> <sub>Jianhao</sub> <sub>Jiao,</sub> <sub>Shiqi</sub> <sub>Zhao,</sub> <sub>Lingyun</sub> <sub>Xu,</sub> <sub>Guoquan</sub> <sub>Huang,</sub> <sub>Howie</sub> <sub>Choset,</sub> Sebastian Scherer, and Jianda Han. 2025. General Place Recognition Survey: Toward Real-World Autonomy. JEEE Transactions on Robotics 41 (2025), 3019- 3038.

[55 Shangshu Yu, Xiaotian Sun, Wen Li, Chenglu Wen, Yunuo Yang, Bailu Si, Guosheng Hu, and Cheng Wang. 2024. Nidaloc: Neurobiologically inspired deep lidar localization. IEEE Transactions on Intelligent Transportation Systems 25, 5 (2024), 4278-4289.

[56 Shangshu Yu, Cheng Wang, Yitai Lin, Chenglu Wen, Ming Cheng, and Guosheng Hu. 2023. Stcloc: Deep lidar localization with spatio-temporal constraints. IEEE Transactions on Intelligent Transportation Systems 24, 1 (2023), 489-500.

[57 Shangshu Yu, Cheng Wang, Chenglu Wen, Ming Cheng, Minghao Liu, Zhihong Zhang, and Xin Li. 2022. LiDAR-based localization using universal encoding and memory-aware regression. Pattern Recognition 128 (2022), 108685.

Chongjian Yuan, Jiarong Lin, Zuhao Zou, Xiaoping Hong, and Fu Zhang. 2023. Std: Stable triangle descriptor for 3d place recognition. In 2023 IEEE International Conference on Robotics and Automation (ICRA). 1897-1903.

Yongjun Zhang, Pengcheng Shi, and Jiayuan Li. 2024. LiDAR-Based Place Recognition For Autonomous Driving: A Survey. Comput. Surveys 57, 4 (2024), 1-36.

Lei Zhou, Zixin Luo, Tianwei Shen, Jiahui Zhang, Mingmin Zhen, Yao Yao, Tian Fang, and Long Quan. 2020. KFNet: Learning Temporal Camera Relocalization Using Kalman Filtering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 4918-4927.

Junjie Zhu, Bingjun Luo, Tianyu Yang, Zewen Wang, Xibin Zhao, and Yue Gao. 2023. Knowledge Conditioned Variational Learning for One-Class Facial Expression Recognition. IEEE Transactions on Image Processing (2023).

[62 Junjie Zhu, Bingjun Luo, Sicheng Zhao, Shihui Ying, Xibin Zhao, and Yue Gao. 2020. Iexpressnet: Facial expression recognition with incremental classes. In Proceedings of the 28th ACM International Conference on Multimedia. 2899-2908.
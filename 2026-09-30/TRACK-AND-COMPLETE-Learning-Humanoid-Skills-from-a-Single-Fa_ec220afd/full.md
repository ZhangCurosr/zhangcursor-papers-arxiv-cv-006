# TRACK-AND-COMPLETE: Learning Humanoid Skills from a Single Failed Human Video

Sarmad Idrees<sup>1</sup> and Jongeun Choi<sup>1,∗</sup>

Project page: https://tracc-humanoid.github.io

![](images/72e64ff956551b4df4d3f626c63c94e0bdcca5d349aa4573b3ab6675e936540d.jpg)  
Fig. 1. From a single failed video, TRACC tracks the usable motion prefix (green) up to the Point-of-No-Return (PoNR), then optimizes the intended task outcome (red) without a successful task-motion demonstration.

Abstract— Learning humanoid skills from videos typically requires a successful human demonstration, which often demands custom data collection. Although failures have traditionally been treated only as negative examples in robot learning, they can still reveal a usable trajectory prefix before the task fails, as well as the intended outcome. To leverage this information from a failed-attempt video, we propose TRACC, a pipeline that imitates the useful portion of the motion trajectory and then completes the task based on the inferred task outcome. The usable motion prefix serves as prior knowledge until the failure occurs, after which the task-completion reward guides the policy to learn the intended task goal without requiring a successful task trajectory. We evaluate our method on six in-the-wild failed human tasks from the Oops! dataset. Our experimental results demonstrate the effectiveness of the proposed approach for learning from failed attempts when no successful demonstration is available. Thus, these findings establish failed human videos as a viable source of supervision for humanoid skill learning.

## I. INTRODUCTION

Traditionally, learning physics-based humanoid skills from videos has focused on imitating demonstrated successful actions [1], [3], [10], [11], [12]. The recovered motion from these videos is therefore treated as a reference trajectory, and the policy is trained to reproduce the motion for a successful task completion. However, relying only on successful attempts often requires custom data collection, which is a tedious and expensive process.

TABLE I  
COMPARISON OF DEMONSTRATION-BASED POLICY LEARNING METHODS.
<table><tr><td>Method</td><td></td><td>Embodiment Demonstration</td><td>Single video</td><td>Use of failure</td></tr><tr><td>VideoMimic [1]</td><td>Humanoid</td><td>Target video</td><td>Multi</td><td></td></tr><tr><td>MeshMimic [2]</td><td>Humanoid</td><td>Target video</td><td>√</td><td></td></tr><tr><td>HDMI [3]</td><td>Humanoid</td><td>Target video</td><td>√</td><td></td></tr><tr><td>OKAMI [4]</td><td>Humanoid</td><td>Target video</td><td>√</td><td></td></tr><tr><td>ORION [5]</td><td>Arm</td><td>Target video</td><td>√</td><td></td></tr><tr><td>LUCID [6]</td><td>Arm</td><td>Video corpus</td><td>Multi</td><td></td></tr><tr><td>Donut as I Do [7]</td><td>Arm</td><td>Failed motion</td><td></td><td>Negative</td></tr><tr><td>RL-VLM-F [8]</td><td>Arm</td><td>Obs. pairs</td><td></td><td>Negative</td></tr><tr><td>Goals from failure [9]</td><td>None</td><td>Failed videos</td><td>一</td><td>Infer goal</td></tr><tr><td>TRACC (ours)</td><td>Humanoid</td><td>Failed video</td><td>√</td><td>Prefix + goal</td></tr></table>

On the other hand, failed attempts were mainly treated as negative examples that the policy should learn to avoid [7], [8]. A failed attempt can still show a useful approach, preparation, or early interaction before the task goes wrong. It can also reveal what the person intended to achieve for the task completion. Treating the whole video as an imitation target also includes the failed ending, while treating the whole video only as a negative example discards its useful prefix. Therefore, rather than treating a failed attempt entirely as either an imitation target or a negative example, we utilize its usable trajectory as motion guidance and its intended goal

![](images/03aaaf0dd76ecd76fab25b0bff59a3372f264cd2dc3fb2d4f705858af2808659.jpg)  
Fig. 2. The overall architecture of the TRACC pipeline. Human motion is reconstructed from the failed video, retargeted to the humanoid, and aligned with the simulated scene to train the prefix tracker. A VLM infers the intended goal, proposes task-reward candidates, and estimates a PoNR window. The tracker is then fine-tuned to complete the intended task.

for task completion.

Using widely available failed attempts of human actions [13], we study whether a humanoid can learn a complete task from this partial information when no successful task motion is available. We propose TRACC, a pipeline that Tracks and Complete the task by first imitating usable motion prefix from unsuccessful demonstrations and then learning to achieve the intended outcome. Specifically, we divide a failed video into two forms of supervision: the usable trajectory prefix provides reference motion for how the task begins, while the inferred intent describes how the task should end. The human motion is recovered from the video and retargeted to a humanoid robot, and the intendedtask outcome is inferred through a vision-language model (VLM), which also proposes task-completion reward and estimates when reference guidance should be released.

To determine how much of the failed demonstration should be used as reference motion, we define the Point-of-No-Return (PoNR) as the last point in the trajectory where the demonstrated motion remains useful for completing the task, before the observed failure begins. Tracking reward fades as the policy approaches PoNR, then the task-completion reward takes control to guide the policy toward the inferred intended outcome (see Figure 1). Thus, the method utilizes the usable portion of the failed attempt without imitating its failed ending or requiring a successful completion trajectory. To the best of our knowledge, TRACC is the first method to learn whole-body humanoid skills from a single failed human video, using the pre-failure motion as positive guidance and an inferred outcome for completion.

We evaluate TRACC on six failed-action videos from the Oops! dataset [13]. The failed attempt videos are selected under the assumptions that at least some useful human motion prefix is available and that the intended final outcome is feasible in the simulation environment. The experiments show that our method successfully learns to track the initial guidance motion and subsequently complete the task when no successful task-completion trajectory is available.

The main contributions of this work are threefold:

1) We formulate a video-based humanoid learning problem that learns a complete task from a failed human demonstration by exploiting its usable motion prefix and intended outcome, without requiring any successful task trajectory.

2) We propose TRACC, which decomposes a failed demonstration into motion guidance and task intent, and combines retargeted human motion with a task reward to learn the intended task beyond the observed failure.

3) We introduce the Point-of-No-Return (PoNR) to identify the latest viable part of the failed trajectory and progressively transition the policy from motion tracking to task-completion reward.

## II. RELATED WORK

a) Video-Based Humanoid Learning: Physics-based imitation learning methods learn to mimic reference human motion [14] by tracking rewards and reference-state initialization [15], or learn motion style through an adversarial motion prior [16], [17], [18]. Video-based methods first recover this reference motion from a monocular video and then train the policy to imitate the demonstrated task action [10], [19], [11]. Later works transferred the retargeted motion to humanoid robots for learning from human video demonstrations [12], [1], [20], [2]. These methods require a successful demonstration of the task as a behavior worth reproducing. In this study, we argue that failed demonstrations also provide enough information to learn the intended task by utilizing only the part of a motion that remains useful before the observed failure.

b) Learning From Failed Demonstrations: Prior work has used failed or imperfect demonstrations in several ways. Failed examples can identify regions that a policy should avoid [7], confidence scores can reduce the influence of poor demonstrations [21], and rankings over suboptimal behavior can support reward inference [22]. Failed videos have also been used to infer goals for visual planning [9]. These approaches use failure as negative example, ranking, or only goal inference. However, learning a physics-based humanoid policy from a single failed monocular human video has not been explored. TRACC uses the pre-failure motion as positive trajectory information and the inferred goal to complete the task.

c) Language Models for Reward Generation: Language models have demonstrated major success in translating task descriptions into executable reward functions for reinforcement learning [23], [24], [25], and related work extends reward generation for sim-to-real transfer [26]. Video2Reward [27] generates imitation rewards from videos that show target behavior. Vision-language models have also been used directly as reward models or preference providers [28], [8]. Building on these advancements, we use a VLM to infer the intended final outcome from the failed action and generate a taskcompletion reward that guides the policy after the usable motion ends.

## III. TRACC: TRACK-AND-COMPLETE

The overall pipeline is illustrated in Figure 2. We treat a failed-attempt video sequence as a partial demonstration rather than a negative example. First, the human motion is recovered and aligned with respect to the simulation environment, while a VLM generates a reward function after inferring the intended outcome of the task. Second, a tracking policy is trained to imitate the motion up to the point where the failure begins to appear. Finally, we further fine-tune the policy to first follow the useful trajectory and then complete the intended outcome. Hereafter, we refer to this combined tracking and task-objective policy as a task-specific unified policy, where a separate policy is trained for each task.

## A. Problem Formulation

Given a failed attempt human video $V = \left( I _ { 1 } , \ldots , I _ { T } \right)$ , we aim to learn a policy π that reproduces the useful trajectory of the observed motion and completes the intended task. The retargeted motion reference is defined as $\hat { M } = ( { \hat { s } } _ { 1 } , \dots , { \hat { s } } _ { T } )$

Algorithm 1 TRACC : Track-and-Complete   
Require: Failed-attempt video demonstration V   
Ensure: Unified policy π , selected reward $r _ { \mathrm { t a s k } } ^ { * } ,$ and measured PoNR τ<sup>∗</sup>   
Construct motion and task supervision   
1: M<sup>ˆ</sup> ← RECONSTRUCTRETARGETALIGN $( V , \mathcal { E } )$   
2: $( \{ r _ { \mathrm { t a s k } } ^ { k } \} _ { k = 1 } ^ { K } , P o N R _ { w } ) \gets$ INFERINTENTANDREWARDS(V)   
Learn the reference prefix   
3: θ ← TRAINTRACKER(M<sup>ˆ</sup> ,uniform RSI)   
4: θ ← FINETUNEFROMSTART $( \theta _ { \mathrm { p r e } } , \hat { s } _ { 1 } )$   
Select a task reward   
5: k<sup>∗</sup> ← BESTBYSUCCESSRATE $( \{ r _ { \mathrm { t a s k } } ^ { k } \} _ { k = 1 } ^ { K } ) ; \quad \theta  \theta _ { \mathrm { p r e } }$   
Train one release-conditioned policy   
6: while not converged do   
7: for all parallel environments e do   
8: Reset to ˆs<sub>1</sub> and sample $\tau _ { e } \sim \mathcal { U } ( P o N R _ { w } )$   
9: Roll out π with time-to-release in the observation   
10: $\mathbf { i f } \ t < \tau _ { e }$ then   
11: Use task reward and weighted tracking reward   
12: else   
13: Mask the reference and use task reward only   
14: end if   
15: end for   
16: Update θ with PPO using Eq. 3   
17: end while   
18: Measure the PoNR τ<sup>∗</sup> from terminal success   
19: return $\pi _ { \theta } , r _ { \mathrm { t a s k } } ^ { k ^ { \ast } } , \tau ^ { \ast }$  
TABLE II

POLICY OBSERVATION SPACE.
<table><tr><td>Group</td><td>Component</td><td>Dim.</td><td>After τ</td></tr><tr><td>Proprioception</td><td>Root position, orientation, velocity</td><td>13</td><td>Observed</td></tr><tr><td></td><td>Joint position</td><td>29</td><td>Observed</td></tr><tr><td></td><td>Joint velocity</td><td>29</td><td>Observed</td></tr><tr><td></td><td>Previous action</td><td>29</td><td>Observed</td></tr><tr><td>Task state</td><td>Task object, goal, and scene state</td><td> $d _ { \mathrm { t a s k } }$ </td><td>Observed</td></tr><tr><td>Reference</td><td>Activity, phase, release time τ</td><td>3</td><td>Masked-out</td></tr><tr><td></td><td>Heading-relative root offset</td><td>3</td><td>Masked-out</td></tr><tr><td></td><td>Heading-relative hand and foot offsets</td><td>12</td><td>Masked-out</td></tr></table>

while the VLM produces an intended goal description, taskreward candidates $\{ r _ { \mathrm { t a s k } } ^ { k } \} _ { k = 1 } ^ { K } ,$ , and a PoNR window $P o N R _ { w } =$ $[ \tau _ { \operatorname* { m i n } } , \tau _ { \operatorname* { m a x } } ]$ . At time t, the policy observes the humanoid states, the task observations, and prefix reference trajectory, and outputs normalized joint-position targets.

The unified policy is trained by uniformly sampling a release time τ per episode from $P o N R _ { w }$ and maximizing

$$
\theta ^ { * } = \arg \operatorname* { m a x } _ { \theta } \mathbb { E } _ { \tau \sim \mathcal { U } [ \tau _ { \mathrm { m i n } } , \tau _ { \mathrm { m a x } } ] , \pi _ { \theta } } \left[ \sum _ { t = 0 } ^ { H - 1 } \gamma ^ { t } r _ { t } ( \tau ) \right] ,\tag{1}
$$

where θ denotes the trainable policy parameters, $\theta ^ { * }$ their optimized values, and $\pi _ { \theta }$ the resulting policy. Also, H is the episode horizon, $\gamma \in [ 0 , 1 )$ is the discount factor, and $r _ { t } ( \tau )$ is the reward at time t under release time τ.

## B. Motion Reconstruction

We process the failed-attempt video in three steps. We first utilize GVHMR [19] to recover a world-grounded 3D human trajectory from the monocular video. Afterwards, we retarget the reconstructed body motion to the 29-DoF Unitree G1 while preserving its root motion and whole-body configuration within the humanoid embodiment. The retargeted motion is then aligned with respect to the simulated task scene. Furthermore, we estimate the task-object trajectory from the video and use the corresponding scene geometry to place the humanoid reference consistently with the object. The resultant scene-aligned reference motion M<sup>ˆ</sup> is finally used by the tracking stage.

![](images/e94109ce0efeebc3ce34f37f975d772acb06903fbe33bb812912bb717d215d0d.jpg)  
Fig. 3. Task success vs. release time τ plots for each task. The shaded area denotes the VLM-proposed release window $P o N R _ { w } ,$ and the dashed line marks the measured PoNR τ<sup>∗</sup>.

## C. Prefix Reference Tracking

From the provided video V, we only utilize the usable prefix of M<sup>ˆ</sup> as a positive imitation target. Where, the usable reference cutoff is determined by the VLM estimated $P o N R _ { w } .$ Following the motion-tracking objective of DeepMimic [15], we define the tracking reward as a weighted sum of pose (q), joint-velocity (q˙), end-effector (ee), and root (root) tracking terms:

$$
r _ { \mathrm { t r a c k } } ( s _ { t } , \hat { s } _ { t } ) = \frac { 1 } { \sum _ { m \in \mathcal { M } } w _ { m } } \sum _ { m \in \mathcal { M } } w _ { m } \exp ( - k _ { m } e _ { m } ) ,\tag{2}
$$

with ${ \mathcal { M } } = \{ q , \dot { q } , \mathrm { e e } , \mathrm { r o o t } \}$ , where $s _ { t }$ and $\hat { s } _ { t }$ are the current humanoid states and reference states, $e _ { m }$ is the corresponding mean-squared tracking error, and $w _ { m }$ and $k _ { m }$ control the contribution and sensitivity of each term.

We first train the tracking policy with reference-state initialization (RSI), sampling the humanoid reset state uniformly along the reference motion trajectory. RSI exposes the policy directly to every phase of the motion instead of requiring an untrained controller to reach later reference states through a long rollout, a strategy established for physics-based motion tracking [15]. However, RSI alone can hide errors that accumulate when the motion is executed from the initial state. As our final goal is to learn a task-conditioned policy that must reliably track the usable trajectory from the initial state and then continue to complete the intended task after the usable trajectory ends, we further fine-tune the same tracker with resets restricted to the initial reference frame. This reset condition forces the policy to reach later prefix states through its own dynamics and matches the initial-state distribution used during the unified policy training.

## D. Intent Inference and Reward Generation

Building on prior work that shows language models can translate semantic task descriptions into executable reinforcement-learning rewards [23], [24], [25], the VLM is prompted with the frame samples from the V to infer the intended outcome, generate $K = 3$ reward candidates for task completion, and provide $P o N R _ { w }$ . Since the visual observation can work as the guidance, hence, we purposely request an interval (i.e. $P o N R _ { w } )$ rather than a single PoNR timestamp to preciesely predict when the demonstrated behavior begins to deteriorate.

During tracking and unified policy training, we sample release times τ uniformly from $P o N R _ { w }$ . This sampled τ acts as the boundary where the tracking reward should be faded out to learn the task completion instead of tracking the failure motion. After training, we evaluate the same policy at several fixed release times over the full video timeframe and define the measured PoNR $\tau ^ { * }$ as the latest tested time that produces the best terminal success rate. Thus, the VLMproposed $P o N R _ { w }$ serves as a visual training prior, whereas the final $\tau ^ { * }$ is measured from the trained policy’s success curve and reported in Sec. IV.

## E. Unified Policy Training

The unified policy is initialized with the prefix tracker policy and further fine-tuned to perform a full reference tracking and task-completion from the initial reset state. Each VLM reward proposal $r _ { \mathrm { t a s k } } ^ { k }$ is trained with same iteration budget, and best overall success reward proposal is selected for further unified policy training. Table III summarizes the selected task reward for each task.

<table><tr><td>Task</td><td>Selected reward terms and weights</td></tr><tr><td>Kick target Football</td><td>Pad approach (0.10); latched strike (0.30); × after the strike: two-foot contact (0.22), pelvis height (0.18), and stability and uprightness (0.15); not fallen (0.05). Foot–ball approach (0.12); ball contact (0.30); ball speed toward the goal (0.22); ball–goal distance (0.18); uprightness (0.14); not fallen (0.04); hovering</td></tr><tr><td></td><td>without contact penalty (—0.10).</td></tr><tr><td>Backflip</td><td>Landing-region approach (0.08); left and right foot contact with load (0.15 each); landing pelvis height (0.18); low linear (0.18) and angular (0.12) velocity; uprightness (0.09); latched inversion bonus (0.05).</td></tr><tr><td>Box jump Handstand</td><td>Feet near the box top (0.15); loaded top contact (0.30); standing height (0.25); pelvis over the box (0.10); controlled standing (0.10); uprightness (0.05); settling (0.05); non-top (−0.55) and invalid-contact (-0.10) penalties; reward set to 1 on success.</td></tr><tr><td></td><td>Hand-patch approach (0.12); left and right palm contact (0.15 each); bilateral palm contact (0.08); × after bilateral contact: inversion progress (0.18) and feet above the pelvis (0.22); pelvis height (0.18); settling (0.07).</td></tr><tr><td>Log walk</td><td>Progress toward the far end of the log (0.45); foot contact on the log top (0.30); pelvis height above the log (0.15); uprightness (0.10); hovering without contact penalty (−0.20).</td></tr></table>

TABLE IV  
TASK SUCCESS CRITERIA.
<table><tr><td>Task</td><td>Success criterion</td></tr><tr><td>Kick target</td><td>Foot strikes the target at ≥ 2.0 m/s, upright ≥ 0.80, settled, held 2.0 s.</td></tr><tr><td>Football</td><td>Ball enters the goal region, upright ≥ 0.80, settled, held 2.0 s.</td></tr><tr><td>Backflip</td><td>Root becomes fully inverted, then both feet in the landing region and in contact, upright ≥ 0.80, settled, held 2.0 s.</td></tr><tr><td>Box jump</td><td>Both feet on the box top, upright ≥ 0.80, settled, held 2.0 s.</td></tr><tr><td>Handstand</td><td>Both hands in the support region and in contact, feet at least 0.40 m above the pelvis, held 0.5 s.</td></tr><tr><td>Log walk</td><td>Net displacement along the log ≥ 1.0 m without falling.</td></tr></table>

To achieve unified tracking and task-completion objective training, we propose a combined reward function, where the task reward remains active throughout the episode, while the tracking reward guides the reference tracking and fades before the sampled release time τ:

$$
r _ { t } ( \tau ) = r _ { \mathrm { t a s k } } ( s _ { t } ) + \lambda _ { t r a c k } ( \tau ) r _ { \mathrm { t r a c k } } ( s _ { t } , \hat { s } _ { t } ) ,\tag{3}
$$

where $\lambda _ { t r a c k }$ is set to 0.5 at the beginning of each episode, and decreases linearly to zero over the final 10% before τ, and remains zero thereafter.

To facilitate this transition from tracking to task-completion learning, we provide release time τ as an observation to the policy. Therefore, the policy can anticipate when reference guidance will disappear instead of reacting to an unexpected observation shift. At and after τ, the entire reference trajectory block is masked to zero, and the policy is required to complete the task from robot states and task observations alone. Algorithm 1 summarizes the whole TRACC pipeline.

## IV. EXPERIMENTS

## A. Experimental Setup

a) Tasks and simulation: Following the assumptions described in Sec. I, we select six failed human videos from the Oops! dataset [13], including box jumping, log walking, target kicking, kicking a football into a goal, backflip, and handstand. Combinely, these six tasks cover impulsive wholebody motion, sustained narrow-support control, and object contact. All experiments use a 29-DoF Unitree G1 humanoid in Isaac Gym [29]. The policy outputs normalized jointposition targets at 30 Hz, and training uses 4,096 parallel environments. We use GPT-5 (gpt-5) as the VLM in all experiments [30]. To avoid evaluation bias, we define task success criteria manually instead of relying on the VLMgenerated success criteria (see Table IV).

![](images/401d75b6bfae97f81999f591d8f13e18574edab06747cab7909746f32a92e5a7.jpg)  
Fig. 4. Failed human demonstrations (top) and corresponding policy rollouts (bottom) for log walking, target kicking, and football kicking, ordered from top to bottom. Frames progress from left to right. Best visualized in zoomed view.

b) Policy and optimization: Our policy architecture follows the tokenized Transformer design used for physical human-scene interaction in TokenHSI [31]. Separate MLP encoders map proprioception, task, and reference motion states to 64-dimensional tokens. A four-layer, two-head Transformer encoder with a 512-dimensional feed-forward layer processes the three tokens, followed by a [1024,512] action head.

![](images/fdf800a1cd93e4dbca0ea46c3f0f4ef5cf37fb6b63ec4b77d9e9fe48c914389e.jpg)  
Fig. 5. Failed human demonstrations (top) and corresponding policy rollouts (bottom) for backflip (left) and handstand (right). Frames progress from left to right. Best visualized in zoomed view.

TABLE V  
COMPARISON RESULTS FOR UNIFIED POLICY VS. TASK-REWARD-ONLY POLICY.
<table><tr><td>Policy</td><td colspan="2">Kick target</td><td colspan="2">Football</td><td colspan="2">Backflip</td><td colspan="2">Box jump</td><td colspan="2">Handstand</td><td colspan="2">Log walk</td></tr><tr><td></td><td>Survival (%)</td><td>Success (%)</td><td>Survival (%)</td><td>Success (%)</td><td>Survival (%)</td><td>Success (%)</td><td>Survival (%)</td><td>Success (%)</td><td>Survival (%)</td><td>Success (%)</td><td>Survival (%)</td><td>Success (%)</td></tr><tr><td>Task reward only</td><td>100.0</td><td>0.0</td><td>100.0</td><td>0.0</td><td>100.0</td><td>0.0</td><td>94.6</td><td>0.0</td><td>100.0</td><td>0.0</td><td>100.0</td><td>0.0</td></tr><tr><td>Unified policy (ours)</td><td>100.0</td><td>99.3</td><td>100.0</td><td>100.0</td><td>99.7</td><td>41.7</td><td>98.6</td><td>75.8</td><td>100.0</td><td>56.3</td><td>100.0</td><td>100.0</td></tr></table>

![](images/1d82c032fc15d2ef7290722a03accf59a1d19f936c08229fb60c5ea2ac8f2435.jpg)  
Fig. 6. Normalized joint jerk around reference release on log walking.

We optimize all policies with PPO [32], using a 32-step rollout horizon, discount $\gamma = 0 . 9 9$ , GAE parameter 0.95, clipping ratio 0.2, and learning rate $2 \times 1 0 ^ { - 5 }$ . Each update uses 6-epochs and 8-minibatches. The prefix tracker is trained for 1,500 additional iterations and fine-tuned for 800 iterations from the initial reference frame. Each valid task reward candidate receives the same 300-iteration training budget, and the candidate with the highest terminal success is further fine-tuned for 1,500 iterations to train a unified policy.

Table II summarizes the policy observation. The task block contains the object, goal, and scene quantities required by each task, so its dimension $d _ { \mathrm { t a s k } }$ varies across tasks. The 18-dimensional reference block is masked-out to zero at τ.

## B. Evaluation Results

Figures 4 and 5 show representative policy rollouts. These figures illustrates that the usable prefix supplies task specific preparation, while the unified policy replaces the failed continuation with motion directed toward the intended outcome. Fig. 3 evaluates the corresponding per-task policies over fixed release times. The VLM-proposed PoNR window is the visual training prior introduced in Sec. III, whereas the measured $\mathrm { P o N R } ~ \tau ^ { * }$ is obtained from the trained policy’s curve. Since the release time τ is provided to the policy as an observation, task-success outside the sampled window $P o N R _ { w }$ shows the temporal generalization of the policy rather than dependence on one hand-selected release time.

a) Kick target: The recovered motion provide the intial step-in, support-leg alignment, and leg swing, whereas the reaching the target, balancing on the support leg, and recovering from the kick are learned from the task reward. PoNR curve shows that the success remains high beyond the $P o N R _ { w }$ , indicating that the policy learned to recover from failling-states. It fail to recover once the reference extends into the purely unusable trajectory motion, placing the measured PoNR $\tau ^ { * }$ after the visual window but before the trajectory becomes harmful.

b) Football: The early non-monotonic region indicates sensitivity to the state at which guidance is removed. Once the reference reaches a useful approach and leg-swing state, it successfully performs the task. Furthermore, Table V demonstrates that the unified policy outperforms the taskreward-only policy in terms of success rate, showing the importance of the motion prefix for task completion.

c) Backflip: The backflip task exposes a limitation of the selected reward rather than the success measure. Inversion contributes only 0.05 to the reward (Table III), whereas the success criterion requires a full inversion before landing (Table IV). When released early, the policy jumps backward and lands without performing a full backflip, achieving high survival but low task success. The PoNR curve reflects this behavior, as the success rate increases once the reference motion provides the inverted state. After this point, the policy only needs to land and settle instead of learning the full backflip.

d) Box jump: The policy rollout frames for box jump task is shown in Fig. 1. The approach and crouched loading motion prepare the humanoid for takeoff. The PoNR curve explains the behavior of intermediate releases, which expose a useful state from which the policy completes the jump and landing, while later releases sharply reduce success as the reference enters the failed continuation.

e) Handstand: The handstand task clearly separates visual failure from dynamic usefulness. The VLM places the release window near the early appearance of failure, approximately 2.5–2.8 s, whereas the greatest task success occurs near 6.7 s. This phenomenon arises because the later reference motion already produces an inverted hand-supported state, leaving the policy only to stabilize the pose for the remaining hold duration.

f) Log walk: Instead of simple walking action, we uplifted the task difficulty to learn to maintain balance on the narrow support. In this log walk task, the prefix trajectory provides the approach and initial balance-oriented steps, and the task completion reward requires to maintain balance and walk for ≥ 1.0 m to achieve success.

Overall, the VLM-proposed window can only be treated as a training prior rather than the final PoNR τ<sup>∗</sup>. The useful boundary depends on both the task and the trained policy and is therefore determined from the release-time curve.

## C. Ablation Studies

Existing video-to-humanoid methods require a target demonstration, while prior learning from failure methods use different inputs or learning objectives (see Table I). Therefore, no prior method directly matches our single failed video setting. We evaluate the two main design choices of our method through ablation studies.

a) Prefix guidance vs. task reward only: We perform an ablation to evaluate the effectiveness of prefix guidance. We compare our unified policy training with task-rewardonly policies by removing the prefix trajectory guidance and training only with the selected task rewards (see Table V). The reward-only policies remain stable and achieve high survival success, but fail to satisfy the terminal success criteria for all six tasks. In contrast, the unified policies produce non-zero success on every task, showing that the prefix provides taskspecific approach and interaction states that are difficult to discover from the task reward alone.

b) Unified policy vs. policy chaining: This ablation evaluates why TRACC uses one policy before and after τ. The natural alternative to our approach is a two-policy chaining mechanism [33], [34], where a prefix policy tracks the reference until τ and a separate completion policy trained with the task reward. We use log walking as a test scenario for this ablation because the narrow-support locomotion makes a control discontinuity immediately visible. We align rollouts at τ<sup>∗</sup> and measure the normalized joint jerk over the complete episode. As shown in Fig. 6, the two-policy chain rises from around 0.17 before release to 0.90 at the switch, whereas the unified policy changes from 0.30 to 0.42. The unified policy therefore has 2.1× lower jerk at release τ. The result specifically shows that retaining the same controller avoids the sharp discontinuity caused by switching between independently trained policies.

## V. DISCUSSION AND FUTURE WORK

Our results show that a failed video can provide useful supervision when its motion prefix and intended outcome are used in tandem. The release-time curves further show that visual failure does not always coincide with the point at which the recovered motion becomes dynamically harmful, supporting policy-conditioned evaluation of the PoNR. These findings make learning from failed demonstrations a promising direction, but the current approach is limited to one humanoid embodiment and non-dexterious humanoid-object interaction tasks.

Future work will extend motion reconstruction and retargeting to dexterous hand motion, enabling robotic manipulation tasks that require precise hand-object interaction. We also plan to move beyond a single humanoid and study failed demonstrations in multi-humanoid collaborative tasks.

## VI. CONCLUSION

We present TRACC, a framework that learns humanoid skills from a single failed human video without requiring a successful task motion demonstration. By treating the recovered prefix as partial supervision, inferring the intended outcome, and training one release time conditioned policy, TRACC retains useful preparation while replacing the failed continuation with task completion-related behavior. The experiments demonstrate the importance of prefix guidance, and show that a unified policy reduces the release discontinuity introduced by policy chaining. The results show that failed human videos can provide useful information for humanoid skill learning and establish a new setting for future research in learning from unsuccessful demonstrations.

## REFERENCES

[1] A. Allshire, H. Choi, J. Zhang, D. McAllister, A. Zhang, C. M. Kim, T. Darrell, P. Abbeel, J. Malik, and A. Kanazawa, “Visual imitation enables contextual humanoid control,” in Proceedings of the 9th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, vol. 305. PMLR, 2025, pp. 794–815, best Student Paper Award. [Online]. Available: https: //proceedings.mlr.press/v305/allshire25a.html

[2] Q. Zhang, J. Ma, P. Liu, S. Shi, Z. Su, Z. Wang, J. Sun, W. Cui, J. Yu, G. Han et al., “Meshmimic: Geometry-aware humanoid motion learning through 3d scene reconstruction,” arXiv preprint arXiv:2602.15733, 2026.

[3] H. Weng, Y. Li, N. Sobanbabu, Z. Wang, Z. Luo, T. He, D. Ramanan, and G. Shi, “Hdmi: Learning interactive humanoid whole-body control from human videos,” arXiv preprint arXiv:2509.16757, 2025.

[4] J. Li, Y. Zhu, Y. Xie, Z. Jiang, M. Seo, G. Pavlakos, and Y. Zhu, “Okami: Teaching humanoid robots manipulation skills through single video imitation,” in 8th Annual Conference on Robot Learning (CoRL), 2024.

[5] Y. Zhu, A. Lim, P. Stone, and Y. Zhu, “Vision-based manipulation from single human video with open-world object graphs,” Autonomous Robots, vol. 50, no. 27, 2026.

[6] H. Gupta, G. Shi, and W. Yuan, “Lucid: Learning embodiment-agnostic intent models from unstructured human videos for scalable dexterous robot skill acquisition,” arXiv preprint arXiv:2606.11628, 2026.

[7] D. H. Grollman and A. Billard, “Donut as i do: Learning from failed demonstrations,” in ICRA, 2011.

[8] Y. Wang, Z. Sun, J. Zhang, Z. Xian, E. Biyik, D. Held, and Z. Erickson, “Rl-vlm-f: Reinforcement learning from vision language foundation model feedback,” in Proceedings of the 41st International Conference on Machine Learning, 2024.

[9] D. Epstein and C. Vondrick, “Learning goals from failure,” in 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2021, pp. 11 189–11 199.

[10] X. B. Peng, A. Kanazawa, J. Malik, P. Abbeel, and S. Levine, “Sfv: reinforcement learning of physical skills from videos,” in ACM Trans. Graph. (SIGGRAPH Asia), 2018.

[11] Z. Luo, J. Cao, K. Kitani, W. Xu et al., “Perpetual humanoid control for real-time simulated avatars,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 10 895–10 904.

[12] T. He, Z. Luo, W. Xiao, C. Zhang, K. Kitani, C. Liu, and G. Shi, “Learning human-to-humanoid real-time whole-body teleoperation,” in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 8944–8951.

[13] D. Epstein, B. Chen, and C. Vondrick, “Oops! predicting unintentional action in video,” in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2020, pp. 916–926.

[14] N. Mahmood, N. Ghorbani, N. F. Troje, G. Pons-Moll, and M. J. Black, “Amass: Archive of motion capture as surface shapes,” in Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 5442–5451.

[15] X. B. Peng, P. Abbeel, S. Levine, and M. van de Panne, “DeepMimic: Example-guided deep reinforcement learning of physics-based character skills,” ACM Transactions on Graphics, vol. 37, no. 4, pp. 1–14, 2018. [Online]. Available: https://doi.org/10.1145/3197517.3201311

[16] X. B. Peng, Z. Ma, P. Abbeel, S. Levine, and A. Kanazawa, “Amp: adversarial motion priors for stylized physics-based character control,” in ACM Trans. Graph. (SIGGRAPH), 2021.

[17] X. B. Peng, Y. Guo, L. Halper, S. Levine, and S. Fidler, “Ase: Largescale reusable adversarial skill embeddings for physically simulated characters,” ACM Transactions On Graphics (TOG), vol. 41, no. 4, pp. 1–17, 2022.

[18] C. Tessler, Y. Kasten, Y. Guo, S. Mannor, G. Chechik, and X. B. Peng, “Calm: Conditional adversarial latent models for directable virtual characters,” in ACM SIGGRAPH 2023 conference proceedings, 2023, pp. 1–9.

[19] Z. Shen, H. Pi, Y. Xia, Z. Cen, S. Peng, Z. Hu, H. Bao, R. Hu, and X. Zhou, “World-grounded human motion recovery via gravity-view coordinates,” in SIGGRAPH Asia 2024 Conference Papers. ACM, 2024, pp. 1–11. [Online]. Available: https: //doi.org/10.1145/3680528.3687565

[20] Z. Li, C. Chi, B. Zhu, Y. Wei, S. Bai, Y. Ji, Y. Peng, T. Huang, P. Wang, Z. Wang et al., “Robomirror: Understand before you imitate for video to humanoid locomotion,” arXiv preprint arXiv:2512.23649, 2025.

[21] Y.-H. Wu, N. Charoenphakdee, H. Bao, V. Tangkaratt, and M. Sugiyama, “Imitation learning from imperfect demonstration,” in ICML, 2019. [Online]. Available: https://arxiv.org/abs/1901.09387

[22] D. S. Brown, W. Goo, P. Nagarajan, and S. Niekum, “Extrapolating beyond suboptimal demonstrations via inverse reinforcement learning from observations,” in ICML, 2019. [Online]. Available: https: //arxiv.org/abs/1904.06387

[23] W. Yu, N. Gileadi, C. Fu, S. Kirmani, K.-H. Lee, M. Gonzalez Arenas, H.-T. L. Chiang, T. Erez, L. Hasenclever, J. Humplik, B. Ichter, T. Xiao, P. Xu, A. Zeng, T. Zhang, N. Heess, D. Sadigh, J. Tan, Y. Tassa, and F. Xia, “Language to rewards for robotic skill synthesis,” in Proceedings of the 7th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, vol. 229. PMLR, 2023. [Online]. Available: https://proceedings.mlr.press/v229/yu23a.html

[24] T. Xie, S. Zhao, C. H. Wu, Y. Liu, Q. Luo, V. Zhong, Y. Yang, and T. Yu, “Text2reward: Reward shaping with language models for reinforcement learning,” in ICLR, 2023. [Online]. Available: https://arxiv.org/abs/2309.11489

[25] Y. J. Ma, W. Liang, G. Wang, D.-A. Huang, O. Bastani, D. Jayaraman, Y. Zhu, L. Fan, and A. Anandkumar, “Eureka: Human-level reward design via coding large language models,” in The Twelfth International Conference on Learning Representations (ICLR), 2024. [Online]. Available: https://openreview.net/forum?id=HfVc7hq3ag

[26] Y. J. Ma, W. Liang, H.-J. Wang, S. Wang, Y. Zhu, L. Fan, O. Bastani, and D. Jayaraman, “Dreureka: Language model guided sim-to-real transfer,” in RSS, 2024. [Online]. Available: https://arxiv.org/abs/2406.01967

[27] R. Zeng, D. Zhou, Q. Liang, J. Liu, H. Li, C. Huang, J. Li, X. Hu, and F. Sun, “Video2Reward: Generating reward function from videos for legged robot behavior learning,” in ECAI 2024 – 27th European Conference on Artificial Intelligence, ser. Frontiers in Artificial Intelligence and Applications. IOS Press, 2024, pp. 4369–4376. [Online]. Available: https://doi.org/10.3233/FAIA241014

[28] J. Rocamonde, V. Montesinos, E. Nava, E. Perez, and D. Lindner, “Vision-language models are zero-shot reward models for reinforcement learning,” in ICLR, 2023. [Online]. Available: https://arxiv.org/abs/ 2310.12921

[29] V. Makoviychuk, L. Wawrzyniak, Y. Guo, M. Lu, K. Storey, M. Macklin, D. Hoeller, N. Rudin, A. Allshire, A. Handa, and G. State, “Isaac Gym: High performance GPU-based physics simulation for robot learning,” in Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS), 2021. [Online]. Available: https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/ hash/28dd2c7955ce926456240b2ff0100bde-Abstract-round2.html

[30] OpenAI, “Openai gpt-5 system card,” CoRR, vol. abs/2601.03267, 2026. [Online]. Available: https://doi.org/10.48550/arXiv.2601.03267

[31] L. Pan, Z. Yang, Z. Dou, W. Wang, B. Huang, B. Dai, T. Komura, and J. Wang, “TokenHSI: Unified synthesis of physical human–scene interactions through task tokenization,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2025, pp. 5379–5391. [Online]. Available: https://arxiv.org/abs/2503.19901

[32] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” 2017. [Online]. Available: https://arxiv.org/abs/1707.06347

[33] Y. Lee, J. J. Lim, A. Anandkumar, and Y. Zhu, “Adversarial skill chaining for long-horizon robot manipulation via terminal state regularization,” in Proceedings of the 5th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, vol. 164. PMLR, 2022, pp. 406–416. [Online]. Available: https://proceedings.mlr.press/v164/lee22a.html

[34] I. Uchendu, T. Xiao, Y. Lu, B. Zhu, M. Yan, J. Simon, M. Bennice, C. Fu, C. Ma, J. Jiao, S. Levine, and K. Hausman, “Jump-start reinforcement learning,” in Proceedings of the 40th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 202. PMLR, 2023, pp. 34 556–34 583. [Online]. Available: https://proceedings.mlr.press/v202/uchendu23a. html
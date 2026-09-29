# UNCOVERING ORDINAL-MATCHING BIAS IN AUDIO-VISUAL LLMS

Jihoo Jung<sup>1</sup>, Youngjoon Jang<sup>2</sup>, Hyebin Cho<sup>1</sup>, Suho Yoo<sup>1</sup>, Joon Son Chung<sup>1</sup>

<sup>1</sup>Korea Advanced Institute of Science and Technology, South Korea <sup>2</sup>University of Oxford, United Kingdom

## ABSTRACT

This work aims to improve how audio-visual large language models (AVLLMs) associate speech with the correct visible speaker in multi-speaker scenes. We find that current AVLLMs frequently fail at this task, and analyze the nature of these fail ures. To this end, we construct a synthetic diagnostic dataset in which multiple visible speakers each utter a single word. Analysis on this corpus reveals a consistent error pattern across three recent open-source AVLLMs: models attribute utterances by simply matching the order of spoken sentences with the left-to-right, top-to-bottom arrangement of visible faces, rather than relying on audio-visual cues such as lip synchronization. We term this behavior ordinal-matching bias. We further show that this bias can be substantially mitigated through a simple remedy, Ordinal-Decoupled Fine-Tuning (OD-FT), in which models are fine-tuned on synthetic videos where spatial positions of speakers and speaking order are independently randomized. Despite using only 400 synthetic training videos, OD-FT not only suppresses ordinal-matching bias but also improves audio-visual understanding on real-world videos, yielding average gains of 8.27% for Qwen2.5-Omni and 2.57% for video-SALMONN2+ across three audio-visual benchmarks.

Index Terms— Audio-visual large language models, speaker attribution, bias

## 1. INTRODUCTION

Audio-visual large language models (AVLLMs) [1–4] extend large language models to jointly perceive vision and audio, enabling complex reasoning over sounding videos. These advances have also spurred interest in understanding the internal mechanisms of AVLLMs [5, 6]. As conversation-rich videos, including films, and daily vlogs, account for a significant fraction of contemporary video data, speaker-utterance attribution– attributing each utterance to the correct visible speaker–has become an essential capability. Despite strong performance on general audio-visual benchmarks, current AVLLMs perform poorly on speaker-utterance attribution, frequently assigning an utterance to the wrong visible person [7–10].

Evidence from closely related architectures suggests that such systematic failures often stem from underlying biases rather than random errors. LLMs, for instance, become unreliable when reasoning over long contexts because their attention is biased toward certain positions regardless of content [11– 13]. Similarly, large vision-language models (LVLMs) are unreliable over multiple images or long videos largely because their predictions depend heavily on where the visual inputs are positioned rather than on their actual content [14, 15]. This raises the possibility that speaker-attribution errors in AVLLMs follow a similar pattern: when associating auditory signals with visible speakers, the model may rely on spurious cues–the spatial arrangement of faces and the temporal order of utterances–instead of genuine audio-visual correspondence.

![](images/de0664edfc4da55c16761555548ca3be391b3b91dc2e812dd709ed83a1124d0d.jpg)  
Fig. 1: Position bias on curated two-speaker clips with video-SALMONN2+. Accuracy on “Who spoke first?” is substantially higher when the first speaker is on the left. This gap persists even for the same videos after horizontal flipping, indicating a strong bias.

To validate this hypothesis, we conduct a preliminary experiment on two-person conversation clips, where the two speakers appear on the left and right halves of the frame, using the dataset of [16]. Each clip is evaluated under two conditions: the original version and a horizontally flipped version, which reverses the visual spatial position of the speakers in each frame while preserving all semantic content. In both cases, we prompt the model with “Who spoke first?” and restrict the answer choices to “left” or “right.”. Fig. 1 reveals a systematic position-related bias. First, in both the original videos (solid bars) and the flipped videos (hatched bars), accuracy is substantially higher for left-speaker-first clips (blue) than for right-first clips (gray). Second, when comparing the original and horizontally flipped versions of the same video, the model’s accuracy changes substantially: it consistently achieves higher accuracy whenever the first speaker happens to be positioned on the left. Together, these findings reveal a left bias: the model inherently attributes the initial speech to the left-positioned speaker, motivating further analysis.

To analyze such behavior more systematically, we construct a controlled, synthetic video corpus featuring several speakers, each uttering a single word. By evaluating three recent open-source AVLLMs [1, 3, 4], we observe that all models consistently fail to correctly associate the visible speakers with their corresponding spoken utterances. Crucially, these failures are not random: they follow a systematic pattern, which we term ordinal-matching bias. Rather than relying on genuine audio-visual cues such as lip synchronization, the models tend to link the k-th temporal utterance with the k-th speaker in the visual positional order. For example, as depicted in Fig. 2, when queried about the panda located in the top-left position (the first visual subject), the model routinely associates it with the first spoken utterance (“Egypt”), even though the panda actually produces the second utterance (“Mexico”).

This diagnosis suggests a simple remedy: if AVLLMs rely on the spurious correspondence between visual position and utterance order, then training examples that explicitly decouple these two orders should discourage such a shortcut. Based on this intuition, we introduce Ordinal-Decoupled Fine-Tuning (OD-FT). In OD-FT, we fine-tune two AVLLMs [1, 3] on a tar geted set of just 400 synthetic videos where the speaker spatial positions and the temporal utterance orders are fully randomized. OD-FT substantially raises speaker-utterance attribution accuracy, effectively eliminating the ordinal-matching bias. Importantly, the gains are not confined to our synthetic setting: on three real-world audio-visual benchmarks, OD-FT improves Qwen2.5-Omni [3] by 8.27% and video-SALMONN2+ [1] by 2.57% on average.

## 2. IDENTIFYING ORDINAL-MATCHING BIAS

We construct a synthetic diagnostic corpus (Sec. 2.1) to investigate failure modes in speaker–utterance attribution, identify a systematic failure pattern that we term “ordinal-matching bias” (Sec. 2.2), and provide causal evidence of such bias (Sec. 2.3).

## 2.1. Analysis Setup

Diagnostic corpus. As illustrated in Fig. 2, we construct a diagnostic corpus of synthetic videos. Each video features $N \in$ $\{ 4 , 5 , 6 \}$ distinct animal speakers placed at fixed positions, with each speaker uttering a single country name in a randomized sequence.<sup>1</sup> The static speaker images are generated using Gemini3.1-Flash-Lite-Image [18] and subsequently animated with synchronized speech utilizing a recent talking-face generation model [19]. Formally, each video is represented as a set of N speaker-utterance pairs mapping spatially localized speak ers to their temporal speech sequence under a visual positional order σ: $\{ ( V _ { i } ^ { \bar { \sigma } } , A _ { j } ) \bar  \} _ { i , j = 1 } ^ { N }$ . Here, $V _ { i } ^ { \sigma }$ represents the visual speaker located at rank $i \in \{ 1 , \ldots , N \}$ under ordering σ. We employ two such positional orders: row-major $\sigma _ { \mathrm { r o w } } ( \mathrm { e . g . }$ , topleft → top-right → bottom-left → bottom-right) and columnmajor $\sigma _ { \mathrm { c o l } } ( \mathrm { e . g . }$ , top-left → bottom-left → top-right → bottomright). Furthermore, $A _ { j }$ denotes the j-th spoken utterance in the audio sequence $\mathbf { A } = \left( A _ { 1 } , A _ { 2 } , \ldots , A _ { N } \right)$ . For example, the $N = 4$ video in Fig. 2 under a row-major order is formulated as $\{ ( V _ { 1 } ^ { \sigma _ { \mathrm { r o w } } } , A _ { 2 } ) , ( V _ { 2 } ^ { \sigma _ { \mathrm { r o w } } } , A _ { 4 } ) , ( V _ { 3 } ^ { \sigma _ { \mathrm { r o w } } } , A _ { 3 } ) , ( V _ { 4 } ^ { \sigma _ { \mathrm { r o w } } } , A _ { 1 } ) \}$ , indicating, for instance, that the top-left speaker speaks second and the bottom-right speaks first.

![](images/167b967867d8b52c32f9ca8b996b85e48f039441a5cad3bd38275879bd9ebc76.jpg)  
Fig. 2: Illustration of ordinal-matching bias. SW: The model predicts that the 1st on-screen speaker (“panda”) produced the 1st utterance (“Egypt”), rather than its true utterance (“Mexico”). WS: The model predicts that the 2nd utterance (“Mexico”) was produced by the 2nd on-screen speaker (“tiger”), rather than its true source (“panda”).

Tasks. We evaluate two complementary open-ended QA tasks. In Says-What (SW), the model is given a target speaker $V _ { i } ^ { \sigma }$ identify the utterance $A _ { j }$ spoken by $V _ { i } ^ { \sigma }$ . Conversely, in Who-Says (WS), the model is given a target utterance $A _ { j } .$ , identify the visual speaker $V _ { i } ^ { \sigma }$ who uttered it.

Models. We evaluate three recent open-source AVLLMs-Qwen2.5-Omni (7B) [3], video-SALMONN2+ (7B) [1], and Qwen3-Omni (30B-A3B) [4]-and one proprietary model, Gem ini 3.7 Flash [17].

Metrics. We report two metrics. (i) Accuracy measures whether the model prediction matches the ground-truth speaker-utterance attribution. (ii) Ordinal-matching rate $\textstyle ( \operatorname { O M } _ { \sigma } )$ quantifies the model’s tendency to associate the k-th speaker $V _ { k } ^ { \sigma }$ with the k-th utterance $A _ { k }$ , regardless of the true correspondence. Formally, let D be a set of evaluation queries, $\hat { y } _ { q }$ the model prediction for query $q \in \mathcal { D }$ , and $\tilde { y } _ { q } ^ { \sigma }$ the prediction dictated by the ordinal-matching rule $( V _ { k } ^ { \sigma }  A _ { k } )$ The ordinal-matching rate is defined as

$$
\mathrm { O M } _ { \sigma } = \frac { 1 } { | \mathcal { D } | } \sum _ { q \in \mathcal { D } } \mathbf { 1 } \left[ \hat { y } _ { q } = \tilde { y } _ { q } ^ { \sigma } \right] .
$$

## 2.2. Analysis Results

Speaker-utterance attribution collapses to chance in opensource AVLLMs. As shown in Tab. 1, all three open-source

Table 1: Existence of ordinal-matching bias. For open-source AVLLMs, accuracy is low while $\mathrm { O M } _ { \sigma _ { \mathrm { r o w } } }$ is high, i.e., models attribute the k-th utterance to the speaker at the k-th position in row-major order, confirming ordinal-matching bias.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Metric</td><td colspan="3"> $N { = } 4$ </td><td colspan="3"> $N { = } 5$ </td><td colspan="3"> $N { = } 6$ </td></tr><tr><td>SW</td><td>WS</td><td>Chance</td><td>SW</td><td>WS</td><td>Chance</td><td>SW</td><td>WS</td><td>Chance</td></tr><tr><td rowspan="3"> $\mathrm { Q w e n 2 . 5 \mathrm { - O m n i } \ [ 3 ] \mathrm { 7 B } }$ </td><td>Accuracy 1 ↑</td><td>26.9</td><td>25.9</td><td>25.0</td><td>23.7</td><td>24.1</td><td>20.0</td><td>18.9</td><td>19.3</td><td>16.7</td></tr><tr><td> $\mathrm { O M } _ { \sigma _ { r o w } }$ </td><td>70.4</td><td>75.0</td><td>25.0</td><td>54.6</td><td>62.2</td><td>20.0</td><td>48.9</td><td>53.7</td><td>16.7</td></tr><tr><td> $\mathrm { O M } _ { \sigma _ { c o l } }$ </td><td>57.0</td><td>53.1</td><td>25.0</td><td>43.6</td><td>45.4</td><td>20.0</td><td>47.8</td><td>47.2</td><td>16.7</td></tr><tr><td rowspan="3">video  ${ \bf \cdot S A L M O N N } 2 + [ 1 ] 7 \mathrm { B }$ </td><td> $\operatorname { A c c u r a c y } \uparrow$ </td><td>34.0</td><td>27.8</td><td>25.0</td><td>27.7</td><td>25.0</td><td>20.0</td><td>27.0</td><td>24.7</td><td>16.7</td></tr><tr><td> $\mathrm { O M } _ { \sigma _ { r o w } }$ </td><td>67.4</td><td>53.0</td><td>25.0</td><td>60.6</td><td>49.4</td><td>20.0</td><td>49.5</td><td>46.3</td><td>16.7</td></tr><tr><td> $\mathrm { O M } _ { \sigma _ { c o l } }$ </td><td>51.5</td><td>54.9</td><td>25.0</td><td>45.4</td><td>41.4</td><td>20.0</td><td>40.6</td><td>40.6</td><td>16.7</td></tr><tr><td rowspan="3">Qwen3-Omni [4] 30B-A3B</td><td> $_ { \mathrm { A c c u r a c y } } \uparrow$ </td><td>25.9</td><td>28.2</td><td>25.0</td><td>20.2</td><td>25.8</td><td>20.0</td><td>18.7</td><td>19.4</td><td>16.7</td></tr><tr><td> $\mathrm { O M } _ { \sigma _ { r o w } }$ </td><td>68.4</td><td>59.4</td><td>25.0</td><td>54.6</td><td>53.7</td><td>20.0</td><td>49.7</td><td>47.3</td><td>16.7</td></tr><tr><td> $\mathrm { O M } _ { \sigma _ { c o l } }$ </td><td>42.5</td><td>37.8</td><td>25.0</td><td>43.5</td><td>33.2</td><td>20.0</td><td>37.0</td><td>31.0</td><td>16.7</td></tr><tr><td rowspan="3">Gemini 3.7 Flash [17]</td><td> $_ { \mathrm { A c c u r a c y } } \uparrow$ </td><td>97.0</td><td>98.1</td><td>25.0</td><td>97.0</td><td>96.8</td><td>20.0</td><td>95.6</td><td>95.6</td><td>16.7</td></tr><tr><td> $\mathrm { O M } _ { \sigma _ { r o w } }$ </td><td>24.6</td><td>24.9</td><td>25.0</td><td>19.8</td><td>19.5</td><td>20.0</td><td>16.7</td><td>16.5</td><td>16.7</td></tr><tr><td> $\mathrm { O M } _ { \sigma _ { c o l } }$ </td><td>24.5</td><td>24.6</td><td>25.0</td><td>19.8</td><td>19.9</td><td>20.0</td><td>16.1</td><td>16.6</td><td>16.7</td></tr></table>

AVLLMs perform at or near chance level across all speaker counts, indicating that they largely fail to recover the true speaker–utterance correspondence. In contrast, Gemini 3.7 Flash achieves substantially higher accuracy.

Failures are systematic, not random. The ordinal-matching rate of the open-source AVLLMs is substantially above chance for all $N \in \{ 4 , 5 , 6 \}$ , with stronger agreement under rowmajor $( \mathrm { O M } _ { \sigma _ { \mathrm { r o w } } } )$ than column-major order $( \mathrm { O M } _ { \sigma _ { \mathrm { c o l } } } )$ . That is, rather than relying on audio-visual cues, the models associate utterances with speakers by aligning the temporal order of utterances with the spatial order of speakers in row-major layout, which we attribute to the ordinal-matching bias.

## 2.3. Causal Validation via Positional Intervention

The preceding results provide observational evidence of ordinal-matching bias. To establish causality, we manipulate the model’s visual position encodings of each speakers and test whether this intervention alters model’s predictions.

Positional interventions. If the model relies on genuine audio– visual correspondence, altering the visual position encodings of speakers should have little effect. If it instead relies on ordinal-matching bias, its predictions should shift accordingly, aligning the temporal order of utterances with the intervened visual ordering: the ordinal-matching rate measured against the original on-screen ordering (σ) should drop, while the rate measured against the intervened position ordering $( \sigma ^ { \dagger } )$ remains high. Concretely, each video patch enters the vision encoder with an integer coordinate $( h , w )$ within its frame, from which its positional encoding is derived. We rewrite these coordinates per speaker region, identically across frames, in two ways. Swap exchanges the positional assignments of the top-left and top-right speaker regions, whereas Rotate cyclically permutes the assignments of all four regions, such that each region receives the coordinates of its clockwise neighbor. For each intervention, we measure ordinal-matching rate with respect

Table 2: Causal evidence of ordinal-matching bias (N=4). We intervene on the visual positional indices while preserving the original on-screen order (Swap, Rotate). Relative to the baseline, $\mathrm { O M } _ { \sigma _ { \mathrm { r o w } } }$ , defined with respect to the original on-screen order $\sigma _ { \mathrm { r o w } } .$ drops, whereas $\mathrm { O M } _ { \sigma _ { \mathrm { r o w } } ^ { \dagger } }$ , defined with respect to the injected coordinate order $\sigma _ { \mathrm { r o w } } ^ { \dagger } ,$ stays high.
<table><tr><td colspan="4">Qwen2.5-Omni video-SALMONN2+ Qwen3-Omni</td></tr><tr><td>Interv.</td><td>Metric</td><td>SW WS</td><td></td></tr><tr><td>SW</td><td></td><td></td><td>WS SW</td></tr><tr><td>Baseline  $\mathrm { O M } _ { \sigma _ { \mathrm { r o w } } }$ </td><td>70.4</td><td>75.0 67.4</td><td>53.0 68.4 59.4</td></tr><tr><td> $\mathrm { O M } _ { \sigma _ { \mathrm { r o w } } }$ </td><td>28.0↓ 32.9↓</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td>39.5↓</td><td>36.4↓ 34.6↓ 32.4↓</td></tr><tr><td> $\operatorname { S w a p }$   $\mathrm { O M } _ { \sigma _ { \mathrm { r o w } } ^ { \dagger } }$  68.8↑</td><td>75.5↑ 68.9↑ 55.9↑ 61.5↑</td></tr><tr><td></td><td>58.2↑</td></tr><tr><td> $\mathrm { O M } _ { \sigma _ { \mathrm { r o w } } }$   $2 4 . 8 \downarrow$  Rotate  $\mathrm { O M } _ { \sigma _ { \mathrm { r o w } } ^ { \dagger } }$  56.2↑</td><td> $1 5 . 0 \downarrow$  28.0↓ 19.8↓ 29.9↓ 28.1↓</td></tr><tr><td>61.4↑</td><td>55.9↑ 57.9↑ 43.1↑ 38.0↑</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr></table>

to two ordinal templates: the original row-major ordering of speakers on the screen $( O M _ { \sigma _ { r o w } } )$ and the ordering implied by the intervened positions $( O M _ { \sigma _ { r o w } ^ { \dagger } } )$  
Intervention results. As shown in Tab. 2, across all intervention types and models, $O M _ { \sigma _ { \mathrm { r o w } } }$ drops, while $O M _ { \sigma _ { \mathrm { r o w } } ^ { \dagger } }$ have high value; that ${ \mathrm { i s } } ,$ the models’ predictions follow the rewritten positions rather than the actual on-screen layout. This provides causal evidence for ordinal-matching bias.

## 3. MITIGATING ORDINAL-MATCHING BIAS

Having established that speaker-utterance attribution failures of AVLLMs stem from ordinal-matching bias, we investigate whether this bias can be mitigated through counterfactual finetuning (Sec. 3.1) and whether this improves real-world benchmark performance (Sec. 3.2).

## 3.1. Ordinal-Decoupled Fine-tuning Experiments

Fine-tuning corpus. We construct a synthetic fine-tuning corpus of 400 videos spanning two- and four-speaker layouts.

Table 3: OD-FT suppresses ordinal-matching bias and improves real-world understanding. OD-FT improves accuracy on the synthetic diagnostic corpus while reducing ordinal-matching bias to near-chance levels, and consistently improves performance across all three real-world benchmarks. In contrast, OC-FT provides little improvement on the synthetic corpus, exacerbates ordinal-matching bias, and produces smaller gains on real-world benchmarks.
<table><tr><td></td><td colspan="10">Synthetic diagnosis corpus</td><td colspan="3">Real-world benchmarks</td></tr><tr><td></td><td colspan="3"> $N = 4$ </td><td colspan="3"> $N = 5$ </td><td colspan="3"> $N = 6$ </td><td>Social Omni</td><td>Daily Omni</td><td></td><td>AV Speaker</td></tr><tr><td></td><td colspan="2">Accuracy ↑</td><td colspan="2"> $\mathrm { O M } _ { \sigma _ { r o w } } \downarrow$ </td><td colspan="2">Accuracy ↑</td><td colspan="2"> $\mathrm { O M } _ { \sigma _ { r o w } } \downarrow$ </td><td colspan="2">Accuracy ↑</td><td colspan="2"> $\mathrm { O M } _ { \sigma _ { r o w } } \downarrow$  Acc. ↑</td><td>Acc. ↑</td><td>Acc. ↑</td></tr><tr><td>Model</td><td>SW</td><td>WS</td><td>SW</td><td>WS</td><td>SW</td><td>WS</td><td>SW</td><td>WS SW</td><td>WS</td><td>SW</td><td>WS</td><td></td><td></td><td></td></tr><tr><td>Qwen2.5-Omni</td><td>26.9</td><td>25.9</td><td>70.4</td><td>75.0</td><td>23.7</td><td>24.1</td><td>54.6 62.2</td><td>18.9</td><td>19.3</td><td>48.9</td><td>53.7</td><td>38.9</td><td>64.3</td><td>44.8</td></tr><tr><td>+ OC-FT</td><td>25.0</td><td>25.8</td><td>98.5</td><td>98.1</td><td>20.3</td><td>21.7</td><td>78.3 69.6</td><td>17.2</td><td>18.1</td><td>66.6</td><td>58.7</td><td>39.5</td><td>67.2</td><td>45.8</td></tr><tr><td>+ OD-FT</td><td>97.0</td><td>97.9</td><td>26.5</td><td>25.9</td><td>94.8</td><td>92.4</td><td>21.5 20.3</td><td>91.9</td><td>83.7</td><td>19.1</td><td>16.4</td><td>54.8</td><td>69.6</td><td>48.4</td></tr><tr><td>video-SALMONN2+</td><td>34.0</td><td>27.8</td><td>67.4</td><td>53.0</td><td>27.7</td><td>25.0 60.6</td><td>49.4</td><td>27.0</td><td>24.7</td><td>49.5</td><td>46.3</td><td>43.2</td><td>65.1</td><td>44.7</td></tr><tr><td>+ OC-FT</td><td>34.0</td><td>31.9</td><td>80.6</td><td>67.0</td><td>32.4</td><td>31.9 63.7</td><td>60.2</td><td>39.9</td><td>39.4</td><td>55.1</td><td>51.2</td><td>44.6</td><td>67.8</td><td>47.0</td></tr><tr><td>+ OD-FT</td><td>75.0</td><td>45.9</td><td>29.8</td><td>24.6</td><td>79.2</td><td>53.5 22.9</td><td>21.9</td><td>89.9</td><td>80.8</td><td>18.5</td><td>15.9</td><td>45.6</td><td>68.2</td><td>46.9</td></tr></table>

Unlike the diagnostic dataset in Sec. 2.1, these videos feature human speakers. In Ordinal-Decoupled Fine-Tuning (OD-FT), we randomly permute the utterance order independently of the speakers’ spatial order, so that a speaker’s visual position does not correspond to their utterance order. To isolate the effect of this decoupling from fine-tuning itself, we additionally construct a control corpus, Ordinal-Coupled Fine-Tuning (OC-FT), using the same visual and audio content but preserving strict ordinal correspondence: the k-th speaker in row-major order always produces the k-th utterance.

Implementation details. We fine-tune Qwen2.5-Omni (7B) and video-SALMONN2+ (7B) using LoRA [20] with rank 8 and α=16. We optimize the models with AdamW using a learning rate of $1 0 ^ { - 4 }$ . Training is performed with a batch size of 1 and 8 gradient accumulation steps for two epochs.

## 3.2. Experimental Results

OD-FT suppresses ordinal-matching bias. We repeat the evaluation from Sec. 2.2 to test whether ordinal-matching bias is reduced after fine-tuning. As shown in the syntheticcorpus results of Tab. 3, OD-FT improves accuracy while reducing the ordinal-matching rate to near-chance levels. In contrast, OC-FT yields little improvement in accuracy and further strengthens the ordinal-matching bias.

OD-FT improves general performance on real-world datasets. We further test whether reducing ordinal-matching bias improves reasoning performance on real-world datasets. We evaluate on three conversation-centric audio–visual benchmarks: AVSpeaker [10], DailyOmni [21], and the speakerperception subset of SocialOmni [16]. As shown in the real-world benchmark results of Tab. 3, OD-FT improves performance across the three benchmarks, yielding average gains of 8.27% for Qwen2.5-Omni and 2.57% for video-SALMONN2+. It also generally outperforms OC-FT, suggesting that these gains are not merely a generic effect of fine-tuning but specifically arise from breaking the correspondence between speaker position and utterance order.

![](images/9956e645cee91d0021010a44735d8682372950991ebb2b60d67b89e82a15b670.jpg)  
Fig. 3: Effect of the decoupled share. SocialOmni accuracy increases with the fraction of decoupled examples in the finetuning corpus, from 0% (OC-FT) to 100% (OD-FT).

Ablation on the decoupled share. We fine-tune Qwen2.5- Omni on mixtures of the OC-FT and OD-FT corpora, varying the proportion of decoupled examples (25%, 50%, and 75%). As shown in Fig. 3, SocialOmni accuracy increases with the decoupled share, further confirming that the gain comes from breaking the ordinal correspondence.

## 4. CONCLUSION

Current open-source AVLLMs remain unreliable at attributing utterances to the correct visible speaker. We show that this failure is not random: models follow a systematic ordinalmatching bias, pairing the k-th utterance with the k-th speaker in visual position order. Importantly, this bias can be mitigated with minimal supervision. Ordinal-Decoupled Fine-Tuning on only 400 synthetic videos suppresses the bias, and the resulting improvements transfer to three real-world benchmarks. Although our evaluation centers on structured environments and clean speech, we leave dynamic multi-party extensions and root-cause analyses to future work. Ultimately, this study highlights the need for AVLLMs built on genuine audio–visual alignment rather than fragile shortcuts.

## 5. REFERENCES

[1] Changli Tang, Yixuan Li, Yudong Yang, Jimin Zhuang, Guangzhi Sun, Wei Li, Zejun Ma, and Chao Zhang, “video-SALMONN 2: Captioning-enhanced audio-visual large language models,” arXiv:2506.15220, 2025.

[2] Junbo Cui, Bokai Xu, Chongyi Wang, Tianyu Yu, Weiyue Sun, Yingjing Xu, Tianran Wang, Zhihui He, Wenshuo Ma, Tianchi Cai, et al., “Minicpm-o 4.5: Towards realtime full-duplex omni-modal interaction,” arXiv preprint arXiv:2604.27393, 2026.

[3] Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, et al., “Qwen2.5-omni technical report,” arXiv:2503.20215, 2025.

[4] Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, Yuanjun Lv, Yongqi Wang, Dake Guo, He Wang, Linhan Ma, Pei Zhang, Xinyu Zhang, Hongkun Hao, Zishan Guo, Baosong Yang, Bin Zhang, Ziyang Ma, Xipin Wei, Shuai Bai, Keqin Chen, Xuejing Liu, Peng Wang, Mingkun Yang, Dayiheng Liu, Xingzhang Ren, Bo Zheng, Rui Men, Fan Zhou, Bowen Yu, Jianxin Yang, Le Yu, Jingren Zhou, and Junyang Lin, “Qwen3-omni technical report,” arXiv:2509.17765, 2025.

[5] Jihoo Jung, Chaeyoung Jung, Ji-Hoon Kim, and Joon Son Chung, “Probing cross-modal information hubs in audiovisual LLMs,” in Proc. ICML, 2026.

[6] Suho Yoo, Youngjoon Jang, and Joon Son Chung, “On the nature of attention sink that shapes decoding strategy in omni-llms,” arXiv preprint arXiv:2603.14337, 2026.

[7] Jihoo Jung, Youngjoon Jang, and Joon Son Chung, “Who says what: Symbolic trimodal binding mechanisms in audio-visual llms,” arXiv preprint arXiv:2609.31193, 2026.

[8] Changli Tang, Tianyi Wang, Fengyun Rao, Jing Lyu, and Chao Zhang, “D-orca: Dialogue-centric optimization for robust audio-visual captioning,” arXiv preprint arXiv:2602.07960, 2026.

[9] Xinlong Chen, Weihong Lin, Jingyun Hua, Linli Yao, Yue Ding, Bozhou Li, Bohan Zeng, Yang Shi, Qiang Liu, Yuanxing Zhang, et al., “Diadem: Advancing dialogue descriptions in audiovisual video captioning for multimodal large language models,” arXiv preprint arXiv:2601.19267, 2026.

[10] Le Thien Phuc Nguyen, Zhuoran Yu, Samuel Low Yu Hang, Subin An, Jeongik Lee, Yohan Ban, SeungEun

Chung, Thanh-Huy Nguyen, JuWan Maeng, Soochahn Lee, et al., “See, hear, and understand: Benchmarking audiovisual human speech understanding in multimodal large language models,” arXiv preprint arXiv:2512.02231, 2025.

[11] Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang, “Lost in the middle: How language models use long contexts,” Transactions ofthe associationfor computational linguistics, vol. 12, pp. 157–173, 2024.

[12] Raphael Tang, Crystina Zhang, Xueguang Ma, Jimmy Lin, and Ferhan Türe, “Found in the middle: Permutation self-consistency improves listwise ranking in large language models,” in Proc. NAACL, 2024.

[13] Shengnan An, Zexiong Ma, Zeqi Lin, Nanning Zheng, Jian-Guang Lou, and Weizhu Chen, “Make your llm fully utilize the context,” Proc. NeurIPS, 2024.

[14] Xinyu Tian, Shu Zou, Zhaoyuan Yang, and Jing Zhang, “Identifying and mitigating position bias of multi-image vision-language models,” in Proc. CVPR, 2025.

[15] Hou Xia, Zheren Fu, Fangcan Ling, Jiajun Li, Yi Tu, Zhendong Mao, and Yongdong Zhang, “Videolevelgauge: Investigating contextual positional bias in video language models.,” in Proc. ICLR, 2026.

[16] Tianyu Xie, Jinfa Huang, Yuexiao Ma, Rongfang Luo, Yan Yang, Wang Chen, Yuhui Zeng, Ruize Fang, Yixuan Zou, Xiawu Zheng, et al., “SocialOmni: Benchmarking audio-visual social interactivity in omni models,” arXiv preprint arXiv:2603.16859, 2026.

[17] Gemini Team, Google DeepMind, “Gemini 3.7 Flash model card,” Tech. Rep., Google DeepMind, 2026.

[18] Google DeepMind, “Gemini 3.1 flash-lite image (nano banana 2 lite),” 2026.

[19] Rang Meng, Yan Wang, Weipeng Wu, Ruobing Zheng, Yuming Li, and Chenguang Ma, “Echomimicv3: 1.3 b parameters are all you need for unified multi-modal and multi-task human animation,” in Proc. AAAI, 2026.

[20] Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen, “LoRA: Low-rank adaptation of large language models,” in Proc. ICLR, 2022.

[21] Ziwei Zhou, Rui Wang, and Zuxuan Wu, “Daily-omni: Towards audio-visual reasoning with temporal alignment across modalities,” arXiv preprint arXiv:2505.17862, 2025.
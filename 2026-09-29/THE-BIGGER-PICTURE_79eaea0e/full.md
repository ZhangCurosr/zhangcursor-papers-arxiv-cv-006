Trustworthy synthetic visual media: Evidence across the media lifecycle

Zexi Jia<sup>1</sup>, Zhiqiang Yuan<sup>1</sup>, Jie Zhou<sup>1</sup>, and Jinchao Zhang<sup>1,\*</sup>

<sup>1</sup>Tencent WeChat AI, Beijing, China. E-mail: zxjia10@gmail.com; {seraphyuan, withtomzhou, dayerzhang}@tencent.com

<sup>\*</sup>Corresponding author: Jinchao Zhang (E-mail: dayerzhang@tencent.com).

## THE BIGGER PICTURE

Images and videos have long helped people understand what happened and how a work came into being. Generative systems complicate that role. Realistic media can now be produced and revised without leaving a stable history, so appearance no longer reveals whether a scene was captured, synthesized, or altered along the way. Trust must instead come from evidence that explains the path an asset has taken and the circumstances in which it was used. Some of this evidence can be recovered from the media, while some must be recorded during production and preserved as the asset circulates. This review brings those approaches together and asks when their claims remain meaningful after ordinary processing or deliberate manipulation. We argue that trustworthy media do not depend on one universal marker of authenticity. The evidence must suit the question at hand, reach the person making the judgment, and remain open to correction when better information emerges. The larger goal is to keep the history of media intelligible even as the media itself continues to change.

## SUMMARY

Synthetic media rarely reach viewers in the state in which they leave a model. Editing, platform processing, and republication can each alter what a reviewer can still verify. This review asks what can be established when an asset’s production history is incomplete. We introduce a claim-centered evidence framework that compares methods by the question they can answer, the evidence available to the verifier, and the conditions under which the answer remains valid. Detection can estimate synthetic origin from the file alone, but it cannot usually recover authorship or permission. Watermarks and signed provenance records preserve a stronger connection to production when participating systems maintain that connection. Rights mechanisms address a further question: whether protected material or identity was used with authorization and what response the evidence can support. Reported benchmarks show that detector performance declines with changes in generators, subject matter, and acquisition channels, while proactive signals often fail when the workflow departs from the one for which they were designed. Verification should therefore be evaluated across the full path an asset follows. A trustworthy system keeps each claim attached to the relevant part of the asset and makes uncertainty and shared dependencies visible. It also allows a decision to change when better evidence appears.

## KEYWORDS

Synthetic Media; Content Authenticity; Media Forensics; Provenance; Digital Watermarking; Creator Rights; Trustworthy Artificial Intelligence

## INTRODUCTION

A recompressed video reaches a newsroom after passing through several accounts and platforms. Part of the scene may have been captured, another part generated, and the soundtrack replaced, yet the file carries no dependable account of those changes. The newsroom faces several questions that cannot be collapsed into one label: which parts were altered, where this version came from, whether the people depicted consented to the relevant use, and what action the available evidence justifies. A realistic appearance answers none of them by itself. Human observers can perform at near-chance levels when distinguishing GAN-synthesized faces from real ones and may even judge synthetic faces as more trustworthy, underscoring that perceptual realism is not evidence of origin.<sup>1</sup>

The dificulty is not simply to classify the asset as real or fake. Diferent decisions require different claims: whether content was synthesized or altered, which system or workflow produced it, whether a record of that process can be trusted, and whether the people or material involved were used with authorization. These claims draw on diferent evidence. Forensic detectors infer traces from the received file; watermarks and fingerprints attempt to preserve a connection to production; signed provenance records describe declared events; and rights mechanisms connect technical observations to identity, ownership, or consent.<sup>2–8</sup> Cropping, recompression, editing, and republication weaken each connection diferently. What matters is which evidence survives that process and which conclusion it can still support.

Previous surveys have mapped media forensics, deepfake detection, and generated-content analysis in considerable detail.<sup>9–14</sup> More recent reviews examine watermarking, proactive defenses, and the security of provenance systems.<sup>15–19</sup> These method-centered accounts are valuable for comparing algorithms, but they do not always reveal whether two results support the same conclusion. We instead treat a bounded claim as the unit of analysis. This perspective connects technical verification to production history and authorized use while keeping those questions distinct.

Our organizing idea is simple: each method should be judged by the claim it can support. We ask what evidence a verifier can observe and trust, how that evidence changes as the asset travels, and what action it can reasonably justify. On this basis, we bring detection, watermarking, provenance, and rights mechanisms into one claim-centered framework. We also evaluate where their links to the asset break across the media lifecycle and retain the conditions behind quantitative results, so performance can be interpreted in the workflow in which it was measured.

## Scope of this review

We focus on synthetic images and videos, including assets that mix captured and generated material. Audio is included when it afects the interpretation of an audiovisual claim. Interactive and three-dimensional media enter the review when their changing state alters the evidence needed for a visual conclusion. We exclude text-only detection and misinformation research without a media-evidence component, as well as legal interpretation beyond what a technical record can establish. We draw on peer-reviewed research, recent preprints, and primary technical standards, using policy and product documents only to describe formal requirements or deployed capabilities. Quantitative results are compared directly only when their sources, metrics, and evaluation protocols are suficiently aligned. Results obtained under diferent protocols remain separate because their tasks and operating conditions do not support a meaningful pooled estimate.

Figure 1 traces the field’s move from artifact-based detection toward evidence that can be recorded during creation, preserved through distribution, and examined when a claim is disputed.

![](images/7702ea01db2b5f2ca65f984962e9df767c82798a9cae2b3170b67c03aca31e98.jpg)  
Figure 1: Evolution of evidence practices for synthetic visual media. Milestones trace the shift from post-hoc detection to evidence recorded during creation and maintained through later use. Dates indicate when each approach became prominent rather than fixed starting points.

The review begins by defining what counts as synthetic for a given claim and what a verifier can observe. It then moves from post-hoc detection to evidence recorded during production, preserved through distribution, and connected to rights and accountability. The final sections consider how to evaluate and combine incomplete or conflicting records, followed by a research agenda for systems that remain useful when their conclusions are challenged.

## GENERATIVE MEDIA AND THE EVIDENCE PROBLEM

## From generation capability to evidence demand

Generative AI has progressed from producing isolated images to taking part in complete media workflows. Variational autoencoders and adversarial models established learned image generation, and successive advances made their outputs more stable and controllable.<sup>20–25</sup> Difusion and related continuous-time formulations then became a flexible basis for high-quality synthesis and editing.<sup>26–32</sup> Once models could follow language and preserve user-supplied structure, <sup>Let</sup> <sup>X</sup> <sup>denote</sup> <sup>the</sup> <sup>media</sup> <sup>assets</sup> <sup>available</sup> <sup>to</sup> <sup>a</sup> <sup>verifier.</sup> <sup>We</sup> <sup>represent</sup> <sup>an</sup> <sup>asset</sup> <sup>x</sup> <sup>∈</sup> <sup>X</sup> <sup>as</sup> generation became less a standalone task than an interface for revising visual material.<sup>33–43</sup>

![](images/9a53147bdd40d99387c7295aa40a0fc6fb4cf2131c9957506979ff3d8c016e2f.jpg)  
Figure 2: Generative media now extends far beyond still-image synthesis. The timeline follows<sub>Figure 2: Expansion of generative media across visual modalities. Selected milestones are</sub> <sup>this</sup> <sup>change</sup> <sup>into</sup> <sup>longer</sup> <sup>audiovisual</sup> <sup>sequences</sup> <sup>and</sup> <sup>responsive</sup> <sup>worlds,</sup> <sup>where</sup> <sup>verification</sup> <sup>must</sup>arranged in three tracks: image generation, video and audio generation, and three-dimensional or <sup>account</sup> <sup>for</sup> <sup>content</sup> <sup>that</sup> <sup>evolves</sup> <sup>rather</sup> <sup>than</sup> <sup>a</sup> <sup>fixed</sup> <sup>file.</sup>interactive world models. Their increasing temporal, multimodal, and interactive scope broadens the conditions that verification must cover.

A similar shift is occurring in moving and interactive visual media. Video models now pro-<sup>x</sup> <sup>x</sup> <sup>x</sup> duce extended scenes in which voice, movement, and camera behavior develop together, while speech and music generators can replace the audio track that gives those scenes meaning.<sup>44–57</sup> Neural rendering carried generation into three-dimensional space, while world models introduced environments that respond to action.<sup>58–69</sup> These modalities matter here when they alter a visual claim: verification must follow not just a frame but a changing scene whose meaning may depend on sound, prior state, or user interaction.

Higher realism is only part of the change. Generation is now embedded in ordinary editing and distribution workflows, so an asset may pass through several models before it reaches a H = (o , o , . . . , o ), o <sub>∈ O</sub>, (2)viewer. The final result may be partly captured and partly synthetic. It may also be attributable to a particular model even when its use was unauthorized. Verification must recover enough of where O contains the operations through which athat history to answer the specific question at stake.

erations may also change the records available to a later verifier. Let G ⊂ O be the generativeFigure 2 provides the technological backdrop. As synthesis moves beyond static pixels, verioperations. An asset is synthetic with respect to task τ when generafication must follow the history of an asset as its form and use evolve.

## <sub>Syn(x; τ) = 1 ⇐⇒ ∃ o ∈ H</sub>Media assets and production histories

Verification concerns two objects: the asset available for inspection and the history that produced it. We represent that distinction with compact notation:

$$
\begin{array} { r } { x = ( p _ { x } , r _ { x } , c _ { x } ) , \qquad H _ { x } = ( o _ { 1 } , o _ { 2 } , \ldots , o _ { T } ) . } \end{array}\tag{<sup>ay</sup> (1}
$$

Here $p _ { x }$ is the observable payload, $r _ { x }$ contains attached or recoverable records, and $c _ { x }$ is the context available during review. The sequence $H _ { x }$ contains the operations through which the asset was produced and circulated. Two verifiers can see the same pixels yet reach diferent assessments when only one can validate the history behind them.

Whether an asset should be called synthetic depends on the question being asked. For a task τ , we treat x as synthetic only when a generative operation in $H _ { x }$ materially changes the property that τ evaluates. This includes fully generated media as well as works altered only in the part relevant to the claim. An edit may change the answer to one claim while leaving another, such as the identity of the depicted person, unchanged.

Figure 3 organizes the literature by the point at which evidence enters this production history and by the claim it can later support.

## A claim-centered evidence framework

Our claim-centered evidence framework begins with a bounded proposition about an asset and asks how the available signals change confidence in it. Keeping that proposition explicit places technical measurements and institutional judgments in the same analytical structure while preserving the distinction between them.

Let $q$ be one bounded claim about the hidden history of the asset, such as synthetic origin, source attribution, or authorized use. Under verification setting σ, a verifier observes a set of signals $S ( x , \sigma )$ from the payload, records, and context. We represent the verification process as

$$
x \longrightarrow S ( x , \sigma ) \longrightarrow E _ { v } ( q \mid x , \sigma ) \longrightarrow a .\tag{2}
$$

Each transition requires an explicit inference; none is automatic. Here, $E _ { v }$ is the verifier’s assessment of claim q, not a universal trust score. The action a depends on both the uncertainty in that assessment and the consequence of an error. A detector score may be suficient to prioritize review yet far too weak for a public accusation. Likewise, a valid manifest can support chain-of-custody without establishing that the depicted event occurred. Figure 4 expands this short chain.

## Evidence roles across the lifecycle

Four activities shape evidence across the lifecycle, each at a diferent moment. Detection works backward from observed media when little history survives. Provenance records part of that history before it is lost. Rights and accountability processes determine how the history bears on an afected party, and operational evaluation tests whether the evidence remains useful after distribution. None substitutes for the others: an asset can have a known source and uncertain permission, or a valid signed history that omits the event under dispute.

A layered system may combine signals from detection, provenance, rights records, and operational tests in $S ( x , \sigma )$ . Agreement is informative only when the dependencies among those signals are understood. Two apparently independent results may fail together after regeneration, while an authenticated record may conflict with another source or omit a relevant event. Any combined assessment should keep those dependencies and disagreements explicit instead of hiding them in one score.

## Evidence availability and survival

The verification setting σ summarizes four practical conditions: what the verifier can access, which issuers or systems it trusts, the path through which the asset has passed, and how prevalent the claim is expected to be. At one extreme, the verifier sees only pixels. Participating systems may expose progressively richer records, whereas an adversarial setting also allows those records to be removed or forged.

Evidence must also survive the media channel. Let $\Pi _ { T }$ describe the transformations likely along the relevant distribution path, and let $V _ { s }$ test whether signal s can still be recovered. Its

![](images/f350a50fc6d780eea6bb7e253750e08e3ad75df7fb70b7d822b2237ed443081f.jpg)  
Figure 3: Claim-centered roadmap of the evidence literature. Branches organize representative studies by lifecycle role: detection, origin and edit tracing, rights and identity protection, and operational evaluation. Within each branch, verification tasks are linked to relevant method families.

![](images/a6a93698851dceafa03077f74939f6f56d3701c97e24650f6a326b8637ce7670.jpg)  
Figure 4: Claim-centered evidence framework. Production history and observable evidence are interpreted within a verification setting to support bounded claims and proportionate actions. The lower band summarizes the four principles used to evaluate this process.

durability is

$$
D ( s \mid \sigma ) = \mathsf { P r } _ { T \sim \Pi _ { T } } \left[ V _ { s } ( T ( x ) ) = 1 \mid V _ { s } ( x ) = 1 \right] .\tag{3}
$$

A durability of 0.9 means that the signal is expected to remain recoverable after 90% of transformations drawn from the relevant distribution path, conditional on being detectable beforehand. This measure concerns the route through which media actually travels, not merely a fixed distortion suite. Survival is still insuficient when the signal lacks a trusted issuer, supports an ambiguous claim, or produces too many false positives for the expected prevalence of the claim.

## Principles for trustworthy evidence

The framework yields four practical principles. Claim specificity keeps origin, factual truth, attribution, and authorization separate. Evidence survival asks whether a signal remains useful after realistic transformation and handof. Evidence independence makes shared assumptions and failure modes visible when several signals are combined. Decision proportionality matches the strength and reviewability of the evidence to the consequence of acting on it.

These three equations provide only the notation needed for the argument. Equation (1) separates the observable asset from its hidden history, Equation (2) follows the path from signals to a bounded decision, and Equation (3) asks whether those signals survive the path the media actually takes. Two later equations apply the same logic to deployment problems where the numerical relationship is itself informative.

## DETECTING SYNTHETIC CONTENT FROM MEDIA

Detection is most useful when an image or video arrives without a trustworthy record of origin. It estimates whether the received content resembles the forms of synthesis or manipulation represented in its calibration data. In this setting, $S ( x , \sigma )$ consists mainly of measurements from $p _ { x } .$ while the production history remains hidden. The result may justify closer inspection or triage, but it cannot by itself establish source, authorship, permission, or factual truth.

## Artifacts, reconstruction, and learned cues

Early detectors looked for regularities introduced by image synthesis, especially traces of upsampling and characteristic frequency behavior.<sup>2,3,70,76,77</sup> These cues made generated-image detection possible, yet they often reflected the implementation that produced the image. A detector could recognize one pipeline more reliably than synthetic origin itself.

Difusion models shifted attention from fixed artifacts to reconstruction and inversion. An image that lies close to a learned manifold may be reconstructed diferently from a camera image, and several detectors exploit that diference.<sup>82–86</sup> Reconstruction behavior still depends on the model and on how the asset was edited, so the cue can change with the production pipeline.

Recent detectors combine local forensic evidence with broader visual representations, often learned from natural images or vision–language supervision.<sup>91,94,170–172</sup> This improves transfer across some generator families, but it can replace one shortcut with another when the training data makes content or style predictive of the label. No cue generalizes merely because it operates at a high level; its value depends on the shifts it has survived.

## Generalization is the central detection problem

Let e denote an environment defined by a generator, content domain, transformation channel, and acquisition process. A deployment-oriented detector should be judged by its weakest relevant environment, not only its average result:

$$
R _ { \mathsf { r o b } } ( f ) = \mathsf { s u p } \ R _ { e } ( f ) , \qquad R _ { e } ( f ) = \mathbb { E } _ { ( x , y ) \sim P _ { e } } [ \ell ( f ( x ) , y ) ] .\tag{4}
$$

Average accuracy over familiar generators says little about the worst environment. Moving to a new generator tests dependence on the synthesis process, whereas a new domain tests whether content has become a shortcut. Re-digitization changes the signal again by introducing a new acquisition channel. The word “generalization” is useful only when the shift being measured is stated.

Recent benchmarks widen the range of generators and acquisition conditions used for this test.<sup>73,89,90,93,94,173–175</sup> NTIRE 2026, for example, combines outputs from 42 generators with 36 transformations, while the SAFE Image Authenticity Challenge asks systems to detect, classify, and localize both partial and fully synthetic content.<sup>176,177</sup> Text-rich benchmarks expose another blind spot: performance varies sharply across layouts and can deteriorate after JPEG compression.<sup>178</sup>

Method development is moving in the same direction. Layer-transition consistency, editing fingerprints, and quality-aware aggregation each address a diferent source of failure: unseen generators, post-production history, or repeated online reposting.<sup>179–181</sup> Training on thousands of community-released generators adds further diversity, while out-of-the-box studies reveal how public detectors behave without benchmark-specific retraining.<sup>182,183</sup> Together, these studies expose a basic problem: a detector may learn file handling or dataset construction instead of synthesis. Benchmarks must expose those nuisance variables and control them where possible.

Table 1 organizes representative detectors by their evidence cue and backbone. Its three blocks separate conventional benchmark transfer, unseen-model and cross-domain transfer, and real-world re-digitization so that each numerical comparison retains a shared protocol.

Table 1: Image detection across shared evaluation protocols. The three blocks examine cross-generator transfer, model and domain transfer, and performance after sharing or redigitization. ForenSynths, Ojha, GenImage, and FakeForm report accuracy and average precision (Acc./AP, %); RRDataset reports accuracy (%).
<table><tr><td colspan="6">Benchmark transfer: mean Acc./AP across each test set</td></tr><tr><td>Method</td><td>Evidence cue</td><td>Backbone</td><td>ForenSynths</td><td>Ojha</td><td>Genlmage</td></tr><tr><td>CNNSpot (2020) ³</td><td>augmentation artifacts</td><td>ResNet-50</td><td>76.0/86.7</td><td>52.8/67.5</td><td>53.3/63.6</td></tr><tr><td>FreDect (2020) 76</td><td>Fourier spectrum</td><td>ResNet-50</td><td>78.9/76.6</td><td>54.5/49.6</td><td>41.2/46.7</td></tr><tr><td>LGrad (2023) 184</td><td>image gradients</td><td>ResNet-50</td><td>86.1/91.6</td><td>90.9/97.3</td><td>61.8/62.0</td></tr><tr><td>UnivFD (2023) 91</td><td>CLIP semantics</td><td>CLIP ViT-L/14</td><td>89.1/98.2</td><td>86.7/94.3</td><td>70.6/83.7</td></tr><tr><td>PatchCraft (2023) 185</td><td>texture-patch contrast</td><td>patch CNN</td><td>84.8/92.7</td><td>84.0/93.8</td><td>86.3/96.4</td></tr><tr><td>FreqNet (2024) 186</td><td>learned frequency cues</td><td>ResNet-50</td><td>91.5/97.9</td><td>89.6/94.9</td><td>74.1/82.6</td></tr><tr><td>NPR (2024) 74</td><td>neighbor residuals</td><td>ResNet-50</td><td>92.4/95.6</td><td>95.1/97.4</td><td>77.2/83.0</td></tr><tr><td>FatFormer (2024) 171</td><td>forgery-aware features</td><td>CLIP + transformer</td><td>98.2/99.5</td><td>93.6/98.4</td><td>76.7/86.7</td></tr><tr><td>SAFE (2025) 187</td><td>transformation stability</td><td>image encoder</td><td>96.0/98.7</td><td>95.7/99.0</td><td>95.5/99.1</td></tr><tr><td>CoDA (2026) 175</td><td>color-response statistics</td><td>compact ResNet</td><td>98.2/99.6</td><td>97.5/99.4</td><td>95.9/99.1</td></tr><tr><td colspan="6">FakeForm transfer: mean Acc./AP across 13 unseen models and 62 visual domains</td></tr><tr><td>Method</td><td>Evidence cue</td><td colspan="3">Backbone</td><td>Domain</td></tr><tr><td>CNNSpot (2020) ³</td><td>augmentation artifacts</td><td colspan="3">ResNet-50</td><td>50.4/52.3</td></tr><tr><td>UnivFD (2023) 91</td><td>CLIP semantics</td><td colspan="3">CLIP ViT-L/14</td><td>71.2/83.8</td></tr><tr><td>FreqNet (2024) 186</td><td>learned frequency cues</td><td colspan="3">ResNet-50</td><td>61.3/66.3</td></tr><tr><td>NPR (2024) 74</td><td>neighbor residuals</td><td colspan="3">ResNet-50</td><td>68.4/82.1</td></tr><tr><td>FatFormer (2024) 171</td><td>forgery-aware features</td><td colspan="3">CLIP + transformer</td><td>76.5/87.2</td></tr><tr><td>SAFE (2025) 187</td><td>transformation stability</td><td colspan="3">image encoder</td><td>58.6/67.1</td></tr><tr><td>AIDE (2025) 94</td><td>low/high-level fusion</td><td colspan="3">CLIP + CNN</td><td>67.0/83.0</td></tr><tr><td>AIGI-Holmes (2025) 172</td><td>artifacts + reasoning</td><td colspan="3">multimodal LLM compact ResNet</td><td>73.6/86.4</td></tr><tr><td>CoDA (2026) 175</td><td colspan="3">color-response statistics</td><td>91.0/93.0</td><td>77.7/88.1</td></tr><tr><td colspan="7">RRDataset: accuracy under the original and re-digitized channels (%)</td></tr><tr><td>Method Evidence cue</td><td colspan="6">Backbone</td></tr><tr><td>CNNSpot (2020) ³</td><td>augmentation artifacts</td><td colspan="3">ResNet-50</td><td></td><td>43.1</td></tr><tr><td>GramNet (2020) 188</td><td>texture statistics</td><td colspan="3">Gram-CNN</td><td></td><td>79.5</td></tr><tr><td>FreDect (2020) 76</td><td>Fourier spectrum</td><td colspan="3">ResNet-50</td><td></td><td>46.3</td></tr><tr><td>Fusing (2022) 189</td><td>global-local artifacts</td><td colspan="3">dual CNN</td><td></td><td>30.8</td></tr><tr><td>LGrad (2023) 184</td><td>image gradients</td><td colspan="3">ResNet-50</td><td></td><td>14.7</td></tr><tr><td>DNF (2023) 190</td><td>inverse-diffusion noise</td><td colspan="3">ResNet-50</td><td></td><td>0.1</td></tr><tr><td>DIRE (2023) 82</td><td>reconstruction error</td><td colspan="3">ResNet-50</td><td></td><td>1.4</td></tr><tr><td>UnivFD (2023) 91</td><td>CLIP semantics</td><td colspan="3">CLIP ViT-L/14</td><td></td><td>36.2</td></tr><tr><td>NPR (2024) 74</td><td>neighbor residuals</td><td colspan="3">ResNet-50</td><td>59.5 67.0</td><td>38.7</td></tr><tr><td>SSP (2024) 191</td><td>patch noise</td><td colspan="3">compact CNN</td><td>58.2</td><td>32.6</td></tr><tr><td>FreqNet (2024) 186</td><td>learned frequency cues</td><td colspan="3">ResNet-50</td><td>62.8</td><td>37.5</td></tr><tr><td>DRCT (2024) 86</td><td>reconstruction contrast</td><td colspan="3">ConvNeXt-B</td><td>89.6</td><td>64.3</td></tr><tr><td>C2P-CLIP (2025) 192</td><td>category prompts</td><td colspan="3">CLIP ViT-L/14</td><td>58.6</td><td>18.0</td></tr><tr><td>SAFE (2025) 187</td><td>transformation stability</td><td colspan="3">image encoder</td><td>64.6</td><td>2.3</td></tr><tr><td>AIDE (2025) 94</td><td>low/high-level fusion</td><td colspan="3">CLIP + CNN</td><td>78.4</td><td>76.0</td></tr></table>

The three blocks expose distinct and increasingly demanding forms of transfer. In the first, FatFormer falls from 98.2% accuracy on ForenSynths to 76.7% on GenImage, while GenImage accuracy across the compared methods spans 41.2–95.9%. The FakeForm block then compares unseen-model and cross-domain transfer under one evaluation protocol. The direction and size of the gap vary by detector: SAFE moves from 88.4% accuracy across unseen models to 58.6% across domains, and AIDE from 90.8% to 67.0%. The resulting gaps show that transfer across generators and transfer across visual domains are distinct properties. Re-acquisition can be still more disruptive. DRCT reaches 89.6% overall after adaptation to RRDataset, yet established frequency and reconstruction methods fall to 0.1–46.3% on re-digitized fakes. Accuracy and AP also diverge in several cells, showing that a useful ranking can coexist with a decision threshold that no longer transfers.

Within the framework, these values support a synthetic-origin claim only for the verification settings tested; they do not recover a source or an operation history. The sharpest drops occur when the setting changes what reaches the detector. For deployment, calibration and abstention matter more than the highest clean-test score.

## Temporal evidence in video

Video forensics grew out of face manipulation. Early methods tracked blinking, head motion, or localized blending, and successive benchmarks made those tests harder by varying identities, compression, and recording conditions.<sup>100,102–104,193–195</sup> Full-scene generators change the task. They synthesize the setting, camera motion, and action around a subject, so a face-specific cue may cover only a small part of the claim.

Current detectors draw on three temporal scales. Frame-level methods aggregate spatial traces that recur across a clip. Sequence models look for motion or appearance that changes implausibly between frames, while multimodal methods compare the visible action with speech, sound, or a textual account.<sup>110–113,196</sup> These signals are complementary rather than interchangeable. A frame artifact may identify a generator family without locating an altered interval, whereas a temporal inconsistency may reveal that the clip is unstable without identifying its source.

The reported results show how strongly the answer depends on the test construction. On the GVF benchmark, image detectors used without retraining achieved 35.0–53.6% mean accuracy across four generator subsets; retraining on one text-to-video source raised the strongest baselines only to 63.3–66.7%. DeCoF instead reported 85.9–96.4% mean accuracy across Gen-2, Pika, Sora, Veo, and Kling, depending on which open-source generator supplied the training data.<sup>110</sup> The larger AEGIS study presents a diferent picture. On its hard set, Qwen2.5-VL variants detected only 22–23% of synthetic videos without task-specific training; fine-tuning raised the 7B model’s macro-F1 from 0.43 to 0.82 in-domain but only from 0.52 to 0.55 on the hard split.<sup>112</sup> AIGVDBench broadens the comparison to 31 generators, more than 440,000 videos, and 33 detectors. Its task-specific results also change the ranking: I3D reaches 89.1% AUC on image-to-video outputs but 61.2% on closed-source generators, while TimeSformer moves from 75.5% to 86.5%.<sup>113</sup>

These numbers do not establish a stable ordering of video detectors. Duration, generation task, frame sampling, the choice of real videos, and whether the generator is open or proprietary all change what reaches the classifier. Video evaluation needs interval-level ground truth and a video-level decision rule, with separate results for face manipulation, partial editing, and fully generated scenes. Source attribution is a further claim: recent work can distinguish generation task, model family, and precise generator, but that conclusion remains tied to the candidate sources represented during training.<sup>197</sup>

## What passive detection can establish

Operational reliability depends on prevalence as well as benchmark accuracy. If synthetic media occur with prevalence π, a detector with true-positive rate TPR and false-positive rate FPR has positive predictive value

$$
{ \mathsf { P P V } } = { \frac { \pi { \mathsf { T P R } } } { \pi { \mathsf { T P R } } + ( 1 - \pi ) { \mathsf { F P R } } } } .\tag{5}
$$

For example, with 1% prevalence, 90% TPR, and 1% FPR, only about 48% of flagged items are expected to be synthetic. When π is small, even an apparently accurate detector can generate more false allegations than correct detections. Balanced-test accuracy is a poor deployment guide. The relevant question is whether the score remains calibrated at a suficiently low falsepositive rate and whether uncertain cases can be withheld for review.

Passive detection provides a probability about an observed asset under stated conditions. Because failure to find evidence of manipulation is not evidence that an asset is authentic, a negative result leaves open the possibility of an unseen generator.<sup>198</sup> A positive result, meanwhile, says little about source or intent. Detection is best used to trigger closer review when no stronger record survives.

## PRESERVING ORIGIN AND TRANSFORMATION HISTORY

Provenance preserves information that would otherwise have to be inferred after the fact. It can document origin, transformation, or continuity between an observed asset and a production record. In terms of the framework, it adds records to S(x, σ), but only when the verifier can reach and trust the relevant keys, issuers, or remote services. A surviving watermark may justify attribution to a generator or service without explaining what happened later, while a signed history can be authentic and still omit an event that matters. Provenance supports relationships that were recorded; it cannot rule out events that were not.

## Post-hoc watermarks and the robustness tradeof

Post-hoc watermarking inserts a machine-readable signal after creation, so content can be marked without changing the generator. Existing methods make diferent compromises between how much they encode, how visible the change is, and how well the signal survives.<sup>118–121,124,199</sup> Because the mark is added separately, its evidentiary value depends on what remains after the asset has passed through the rest of the workflow.

No watermark maximizes quality, capacity, and durability at once. A larger message may be easier to disrupt, while aggressive robustness training can leave a more visible trace. More importantly, surviving compression does not imply survival after the image has been regenerated in a way that preserves its semantics. Evaluation should follow the workflow the mark is expected to encounter and determine whether the recovered message still leads to an identifiable issuer and a clear claim.

## Watermarks integrated into generation

Generation-integrated methods use the synthesis process itself to carry evidence. Early work showed that a difusion model could introduce a persistent pattern through its noise, decoder, or latent trajectory.<sup>4,5,131,200,201</sup> Later systems moved beyond simple detection. Some localize altered regions or recover richer messages, while others are designed explicitly against removal and spoofing or adapt the idea to autoregressive image generators.<sup>126,127,202–204</sup>

Recent work makes the security question more explicit. MaxMark increases message capacity within the difusion latent, whereas ISTS changes both embedding and verification to address removal and forgery together.<sup>205,206</sup> These advances do not remove the need for adversarial evaluation. A recent audit of autoregressive watermarks shows that one marked reference image can be enough to enable removal or mimicry without access to model parameters or secret keys.<sup>207</sup>

Embedding during generation can mark every output of a participating service without a separate encoding step. Its coverage, however, is a property of deployment rather than of the decoder alone. A model can be run without the marking component, and diferent services may not recognize one another’s keys. Generation-time watermarking can provide strong positive evidence inside a participating ecosystem, but an absent mark remains inconclusive outside it.

Table 2 separates results obtained under shared evaluations from values retained from original protocols. This distinction prevents robustness under compression, semantic editing, and regeneration from being treated as the same property.

Table 2: Image watermarking across shared and original evaluation protocols. W-Bench uses TPR at 0.1% FPR, and UltraEdit uses bit accuracy; other blocks use the quality measures and test conditions of the original studies. WDP denotes watermark detection probability, and PSNR/SSIM denotes peak signal-to-noise ratio and structural similarity.
<table><tr><td colspan="7">Post-hoc methods: W-Bench</td></tr><tr><td>Method</td><td>Payload</td><td>PSNR/SSIM</td><td>Regeneration</td><td>Global edit</td><td>Local edit</td><td>Video</td></tr><tr><td>RivaGAN (2019) 119</td><td>32 bits</td><td>40.43/0.970</td><td>11.3</td><td>14.8</td><td>45.6</td><td>3.2</td></tr><tr><td>StegaStamp (2020) 120</td><td>100 bits</td><td>29.65/0.911</td><td>91.6</td><td>78.7</td><td>99.0</td><td>30.9</td></tr><tr><td>MBRS (2021) 125</td><td>30 bits</td><td>27.37/0.894</td><td>99.4</td><td>59.8</td><td>94.4</td><td>13.6</td></tr><tr><td>CIN (2022) 208</td><td>30 bits</td><td>43.19/0.985</td><td>48.3</td><td>45.6</td><td>58.7</td><td>2.9</td></tr><tr><td>PIMoG (2022) 209</td><td>30 bits</td><td>37.72/0.986</td><td>77.0</td><td>64.9</td><td>69.3</td><td>14.3</td></tr><tr><td>SepMark (2023) 210</td><td>30 bits</td><td>35.48/0.981</td><td>67.5</td><td>74.1</td><td>95.0</td><td>8.8</td></tr><tr><td>TrustMark (2023) 124</td><td>100 bits</td><td>41.27/0.991</td><td>21.7</td><td>69.0</td><td>68.2</td><td>39.6</td></tr><tr><td>VINE-B (2025) )199</td><td>100 bits</td><td>40.51/0.995</td><td>95.1</td><td>88.8</td><td>94.6</td><td>25.4</td></tr><tr><td>VINE-R (2025) 199</td><td>100 bits</td><td>37.34/0.993</td><td>99.8</td><td>93.0</td><td>96.5</td><td>36.3</td></tr><tr><td colspan="7">Generation-integrated methods: UltraEdit</td></tr><tr><td>Method</td><td>Payload</td><td>PSNR/SSIM</td><td>Prompt edit</td><td>Regeneration</td><td>Inpainting</td><td>LaMa</td></tr><tr><td>Stable Signature (2023) 5</td><td>48 bits</td><td>31.43/0.834</td><td>0.561</td><td>0.626</td><td>0.905</td><td>0.894</td></tr><tr><td>WOUAF (2024) 135</td><td>64 bits</td><td>30.71/0.847</td><td>0.587</td><td>0.601</td><td>0.874</td><td>0.883</td></tr><tr><td>LaWa (2024) 211</td><td>48 bits</td><td>35.14/0.821</td><td>0.591</td><td>0.629</td><td>0.892</td><td>0.897</td></tr><tr><td>GenPTW (2026) 126</td><td>64 bits</td><td>39.56/0.892</td><td>0.969</td><td>0.974</td><td>0.990</td><td>0.994</td></tr><tr><td colspan="7">Post-hoc methods: original protocols</td></tr><tr><td>Method</td><td>Payload</td><td>Quality</td><td>Test</td><td>Reported result</td><td></td><td></td></tr><tr><td>WAM (2025) 122</td><td>32 bits</td><td>46.05/1.000</td><td>common; paraphrase</td><td></td><td>0.96–0.98; 0.56–0.63 WDP</td><td></td></tr><tr><td>InvisMark (2025) 121</td><td>256 bits</td><td>51.0/0.998</td><td>manipulation suite</td><td>&gt; 97% bit accuracy</td><td></td><td></td></tr><tr><td>SynthID-Image (2025) 134</td><td>detection</td><td>deployed</td><td>Google services</td><td></td><td>&gt; 10 billion images/frames</td><td></td></tr><tr><td>PECCAVI (2026) 212</td><td>detection</td><td>29.84/0.930</td><td>common; paraphrase</td><td></td><td>0.91–0.99; 0.85–0.90 WDP</td><td></td></tr><tr><td colspan="7">Generation-integrated methods: original protocols</td></tr><tr><td>Method</td><td>Payload</td><td>Quality</td><td>Carrier</td><td>Reported result</td><td></td><td></td></tr><tr><td>Tree-Ring (2023) ⁴</td><td>detection</td><td>25.77/0.920</td><td>diffusion noise</td><td></td><td></td><td></td></tr><tr><td>Gaussian Shading (2024) 131</td><td>multi-bit</td><td>30.23/0.920</td><td>diffusion latent</td><td></td><td>0.92–0.98 / 0.68–0.77 0.93-0.99 / 0.71-0.81</td><td></td></tr><tr><td>ZoDiac (2024) 201</td><td>detection</td><td>28.47/0.920</td><td>latent optimization</td><td></td><td>0.89–0.92 / 0.70-0.81</td><td></td></tr><tr><td>MaxMark (2026) 205</td><td>high capacity</td><td>matched</td><td>diffusion latent</td><td></td><td>up to 46% bit-accuracy gain</td><td></td></tr><tr><td>ISTS (2026) 206</td><td>detection</td><td>prompt-conditioned</td><td>dynamic noise</td><td></td><td>0.936/0.58 removal AUC/TPR</td><td></td></tr><tr><td>GROW (2026) 202</td><td>16 bits</td><td>27.54/0.850</td><td>diffusion sampling</td><td></td><td>97.8% message accuracy</td><td></td></tr><tr><td>PAI (2026) 203</td><td>key</td><td>FID 29.27</td><td>diffusion trajectory</td><td></td><td>99.0% removal; 96.3% spoofing</td><td></td></tr><tr><td>IndexMark (2025) 204,213</td><td>detection</td><td>FID 4.49</td><td>autoregressive tokens</td><td></td><td>83.4% regeneration TPR</td><td></td></tr><tr><td>WMAR (2025) 204,214</td><td>detection</td><td>FID 4.23</td><td>tokens + tuning</td><td></td><td>75.8% regeneration TPR</td><td></td></tr><tr><td>NoisePrints (2025) 128</td><td>seed proof</td><td>distortion-free</td><td>noise + seed</td><td></td><td>third-party image/video proof</td><td></td></tr><tr><td>ClusterMark (2026) 204</td><td>detection</td><td>FID 4.85</td><td>clustered tokens</td><td></td><td>96.9% mean TPR (6 attacks)</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

The shared evaluations make the trade-ofs concrete. VINE-R is more robust than VINE-B but has lower PSNR, and its TPR still falls to 36.3% after image-to-video conversion. Under visual paraphrasing, WAM reports a detection probability of 0.56–0.63, whereas PECCAVI reports 0.85–0.90 by concentrating the mark in more stable regions. Generation-integrated methods can fare better under semantic editing: GenPTW reports 0.969–0.994 bit accuracy across UltraEdit, although only under its own 64-bit protocol. ImageDetectBench likewise compares passive and watermark-based detectors under eight routine and three adversarial perturbations; its strongest watermarks are more robust when their verifier is available, but this advantage depends on participation in the marking system.<sup>215</sup> No reported method combines broad generator coverage, eficient verification, and durable recovery across all of the workflows represented in Table 2. Even when a signal reconnects an asset to an issuer, source history, later edits, and authorization still depend on the records to which that signal leads.

## Attribution at model, service, and user levels

Attribution can begin with a trace in the output, a mark placed during generation, or a record retained by the service. Each ties the conclusion to a diferent point in the production chain.<sup>70,128,133</sup> Recent retrieval-based attribution avoids fixing the candidate set during detector training and can adapt to a new source from a few examples, while video attribution has begun to separate generator, version, task, and developer at several levels.<sup>197,216</sup> A model fingerprint may identify a family of generators, whereas an authenticated service record can narrow the claim to one deployment. Neither, by itself, identifies the person responsible for publishing the asset.

Technical attribution and human responsibility often diverge because models are redistributed and accounts do not map cleanly to individual action. The person who publishes an asset may not be the person who generated it. Attribution should stop at the narrowest level justified by the evidence and state what access made that conclusion possible. Corroborating sources can strengthen the technical link, but an institution must still connect that link to conduct before assigning responsibility.

## Signed records and evidence continuity

Content Credentials and C2PA represent provenance as signed, structured assertions. A record typically binds an asset identifier to a declared history, an issuer, a set of assertions, and a cryptographic signature. In the notation of Equation (1), it enriches $r _ { x }$ with a reviewable account of part of $H _ { x }$ . Cryptographic verification can show that the protected assertion has not changed since signing. It cannot reveal events that were never recorded or establish that the scene itself is true.<sup>6,137</sup>

No single carrier is likely to preserve a complete history. A manifest can describe an edit precisely but may disappear when the asset is screenshotted or transcoded. A watermark carries less information, yet can survive long enough to reconnect the payload to a richer remote record.<sup>138,139</sup> The two are most useful when they reinforce that connection rather than repeat the same claim.

Detection remains useful within this design. Missing credentials leave origin unresolved, while disagreement between the payload and a signed history gives a reviewer a concrete reason to investigate. Provenance works best when its claims can be inspected and challenged instead of being reduced to a badge. A reviewable connection is especially important when the dispute concerns a person’s rights.

## PROTECTING CREATORS, IDENTITY, AND ACCOUNTABILITY

A rights dispute begins where origin evidence stops. Knowing that an image is synthetic, or even which tool produced it, says little about permission. The relevant question is how a protected resource entered the production chain and whether the actor responsible was entitled to use it. Resolving that question requires a record of the relationship, not simply a label on the output.

One output can raise more than one claim because inclusion in a dataset, influence on a model, and reproduction in an output are not the same relationship. Authorization is a separate question. The observable evidence may include protected samples, dataset or service records, model responses, and claimant documentation; diferent verifiers may have access to diferent subsets of this evidence. Evidence that establishes one link should not stand in for the others. Its immediate role is usually to support investigation, notice, or review; stronger sanctions require a reproducible connection to the particular relationship under dispute and a route for the other party to respond.

## Rights claims and evidentiary requirements

A rights claim should name five elements: the actor or afected party, the operation at issue, the protected resource, the pipeline stage, and the institutional context. Specifying the operation clarifies how the resource entered the pipeline. This structure prevents evidence of dataset inclusion from being mistaken for evidence of reproduction or lack of permission. Each link remains a claim in its own right.

Technical tests become more persuasive when administrative records tie the result to a known resource and point in time. Conversely, a license matters only when it applies to the operation being examined. Strong cases connect these records and preserve the history of any challenge, rather than elevating one score above the rest.

## Training data authorization and evidence of use

Training-data governance begins before a model is trained. A later audit is far easier when the dataset preserves where material came from and the terms under which it was collected. Webscale aggregation often strips away that context, making authorization dificult to reconstruct after the model exists.<sup>140,144</sup> An opt-out registry can express a current preference, but it cannot reveal what happened to copies already in circulation.

After training, an external party may need to show that protected data were used in training or influenced the model. Existing tests infer this connection from the model’s behavior or from signals deliberately placed in protected examples.<sup>148–152,217</sup> Black-box membership inference can operate without access to model weights, but it still supports a dataset-membership claim rather than one about authorization or downstream influence.<sup>218</sup> The reliability of these tests changes sharply when few examples are marked or when the model has subsequently been adapted. DWBench also shows that a method can appear reliable for one claimant yet produce false claims when many owners test the same model.

Inclusion does not imply that a sample materially influenced the model or will reappear in an output. Memorization studies show that difusion models can reveal training examples under particular prompting and sampling conditions, especially when examples are repeated.<sup>7,153,154</sup> This finding characterizes model behavior, not authorization. Failure to extract an image also leaves membership unresolved, so an audit must state which relationship its result actually tests.

## Protecting creative work, style, and likeness

Creator-facing defenses alter protected media before a model can learn from or manipulate it. Some make the relevant identity or style harder to learn; others disrupt the behavior of a downstream model.<sup>8,157–161,163,164,219–221</sup> Their appeal is that a creator can act without the provider’s cooperation. The protection is nevertheless conditional on the pipeline anticipated during development.

Recent defenses cover more than one personalization method. IDGuardian targets both identity extraction and injection, allowing the same protected portrait to resist training-based and training-free personalization. DeepProtect focuses on face swapping, whereas StyleProtect updates only style-sensitive cross-attention layers during fine-tuning.<sup>222–224</sup> UniDef pursues transfer across editing models, while VisiLock replaces image perturbation with a visual key that authorizes an editing model.<sup>225,226</sup> These are distinct forms of control and should not be collapsed into a single protection score.

Style and identity claims involve diferent harms. Style can be recognizable across a body of work without any one image being copied. Identity misuse can likewise cause harm even when the result is not perfectly deceptive. Non-consensual generation services show that the loss of control itself can matter.<sup>165</sup> Protection should extend beyond one model and be paired with a record that can support a later complaint or review.

Protection efectiveness changes with the surrounding pipeline. A provider can add purification or replace an encoder, and routine platform processing may weaken the perturbation even without an adaptive attack. Results from a fixed pipeline establish efectiveness only against the models and processing steps tested, not permanent protection. Identity defenses often report face-detection failure rate (FDFR) or identity similarity (ISM), whereas style defenses rely on generation-specific or human judgments; these values should not be pooled. Protection is more durable when prevention is backed by records that can support later verification and remedy.

## Model ownership, service attribution, and user accountability

The model itself may also be the protected resource. Fingerprints and distributor-specific signatures can reveal whether an output or behavior descends from a protected model.<sup>133,135</sup> That lineage becomes harder to recover after merging, distillation, or fine-tuning. Cert-LAS begins to address this gap by certifying model-ownership verification under bounded parameter perturbations rather than reporting only empirical resistance.<sup>227</sup> A credible ownership result should state how much modification it survives and whether an independent party can reproduce it.

Service records can connect a request to an account and preserve the model version used, while a user-specific mark may carry part of that connection beyond the service. Neither identifies a person with certainty. Accounts can be shared or compromised, and the publisher of an asset may be diferent from whoever initiated generation. Accountability depends on documenting those handofs rather than treating an account identifier as proof of conduct.

Persistent identifiers help investigate abuse, but they can also expose confidential activity and vulnerable users. Rights infrastructure should reveal no more than the claim requires and retain that information only as long as its purpose justifies. Attribution should be possible when needed without making ordinary creative work permanently traceable.

## From technical signals to rights decisions

Technical evidence can connect an output to a protected resource, but it does not decide the dispute. Legal and platform processes interpret that connection in light of consent and the circumstances of use.<sup>142,147,166</sup> A watermark match may justify an investigation or support a notice; the outcome rests on the wider record.

A rights record should state who is making the claim, what relationship is being tested, and how the technical result was obtained. It should also include relevant permission records, known sources of error, and a procedure for the other party to respond. Table 3 retains unified datasetauditing results where they are available and separates creator, identity, and ownership mechanisms evaluated under diferent protocols.

Table 3: Rights, identity, and ownership protection mechanisms. In the two DWBench blocks, TPR is measured at 5% FPR and VSR is reported as 0 or 1; WR denotes the data-marking rate in each column header. The other two blocks use the evaluation measures reported by their respective studies.<sup>217</sup>
<table><tr><td colspan="6">Dataset auditing: CIFAR-10 and ResNet-18 (DWBench)</td></tr><tr><td>Method</td><td>Signal</td><td>TPR (WR=1%)</td><td>TPR (WR=0.01%)</td><td>VSR (WR=1%)</td><td>VSR (WR=0.01%)</td></tr><tr><td>Radioactive Data (2020) 228</td><td>feature tag</td><td>23.3</td><td>4.7</td><td>1</td><td>0</td></tr><tr><td>DVBW (2023) 229</td><td>backdoor test</td><td>92.4</td><td>89.4</td><td>1</td><td>1</td></tr><tr><td>DYTMark (2023) 230</td><td>clean-label mark</td><td>91.7</td><td>60.4</td><td>1</td><td>1</td></tr><tr><td>ImgDup (2024) 231</td><td>optimized twins</td><td>8.1</td><td>20.0</td><td>1</td><td>0</td></tr><tr><td colspan="6">Dataset auditing: Pokémon and Stable Diffusion 1.4 LoRA (DWBench)</td></tr><tr><td>Method</td><td>Signal</td><td>TPR (WR=10%)</td><td>TPR (WR=2%)</td><td>VSR (WR=10%)</td><td>VSR (WR=2%)</td></tr><tr><td>RIW (2023) 232</td><td>edit-resistant mark</td><td>70.6</td><td>40.6</td><td>0</td><td>0</td></tr><tr><td>GenWM (2023) 233</td><td>generative mark</td><td>82.7</td><td>12.8</td><td>0</td><td>0</td></tr><tr><td>DiffusionShield (2023) 234</td><td>multi-bit mark</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>DiagnoB (2024) 235</td><td>behavioral trigger</td><td>99.1</td><td>35.3</td><td>1</td><td>0</td></tr><tr><td>EnTruth (2024) 236</td><td>semantic mark</td><td>99.3</td><td>12.2</td><td>1</td><td>0</td></tr><tr><td>DwtWM (2024) 237</td><td>wavelet mark</td><td>77.8</td><td>38.1</td><td>1</td><td>0</td></tr><tr><td>AdvWM (2024) 237</td><td>adversarial mark</td><td>94.7</td><td>19.3</td><td>1</td><td>0</td></tr><tr><td>Siren (2025) 238</td><td>early-learning signal</td><td>98.3</td><td>68.7</td><td>1</td><td>0</td></tr><tr><td>FT-Shield (2025) 239</td><td>expert verifier</td><td>32.7</td><td>14.2</td><td>0</td><td>0</td></tr><tr><td colspan="6">Creator, style, and identity protection</td></tr><tr><td>Method</td><td>Protects</td><td>Strategy</td><td>Original evaluation</td><td></td><td></td></tr><tr><td>Fawkes (2020) 157</td><td>identity</td><td>feature cloak</td><td>PubFig: &gt; 95%; APIs: 100%</td><td></td><td></td></tr><tr><td>LowKey (2021) 158 Unlearnable Examples</td><td>identity</td><td>adversarial filter</td><td>Amazon: &lt; 1% recognition</td><td></td><td></td></tr><tr><td>(2021) 159</td><td>training data</td><td>training perturbation</td><td></td><td>CIFAR/faces: near-random accuracy</td><td></td></tr><tr><td>Mist (2023) 161</td><td>style</td><td>transferable perturbation</td><td></td><td>Cross-model transfer; survives purification</td><td></td></tr><tr><td>Glaze (2023) 8</td><td>style</td><td>style cloak</td><td></td><td>Stable Diffusion: &gt; 92%; adaptive &gt; 85%</td><td></td></tr><tr><td>PhotoGuard (2023) 163</td><td>editing</td><td>input immunization</td><td></td><td>Stable Diffusion: FID 167.6; SSIM 0.50</td><td></td></tr><tr><td>Anti-DreamBooth (2023) 164</td><td>identity</td><td>personalization defense</td><td></td><td>VGGFace2: FDFR 0.63/0.76; ISM 0.33/0.28</td><td></td></tr><tr><td>MetaCloak (2023) 219</td><td>identity</td><td>meta-poisoning</td><td>Replicate: black-box transfer</td><td></td><td></td></tr><tr><td>Nightshade (2024) 160</td><td>style</td><td>concept poisoning</td><td>SDXL: 70–80% at 50 poisons</td><td></td><td></td></tr><tr><td>SimAC (2024) 220</td><td>identity</td><td>timestep attack</td><td>CelebA: FDFR 96.9%</td><td></td><td></td></tr><tr><td>RID (2024) 221</td><td>identity</td><td>one-pass perturbation</td><td>A100: 0.12 s/image</td><td></td><td></td></tr><tr><td>IDGuardian (2026) 222</td><td>identity</td><td>dual-stage perturbation</td><td></td><td>VGGFace2: PSNR 32.19; SSIM 0.842</td><td></td></tr><tr><td>DeepProtect (2026) 223</td><td>identity</td><td>feature and attribute defense</td><td>Five face-swap pipelines</td><td></td><td></td></tr><tr><td>StyleProtect (2026) 224</td><td>style</td><td>selective attention update</td><td></td><td>WikiArt: 30 artists; cross-model tests</td><td></td></tr><tr><td>UniDef (2026) 225</td><td>editing</td><td>model-agnostic perturbation</td><td></td><td>Multiple models and editing tasks</td><td></td></tr><tr><td colspan="6">Ownership verification and attack forensics</td></tr><tr><td>Method</td><td>Claim</td><td>Evidence</td><td>Original evaluation</td><td></td><td></td></tr><tr><td>Data Taggants (2024) 149</td><td>dataset use</td><td></td><td></td><td></td><td></td></tr><tr><td>DOV4CL (2025) 151</td><td>pretraining use</td><td>secret-key tags relation shift</td><td></td><td>ImageNet: black-box; no accuracy loss 5 contrastive models: p &lt; 0.05</td><td></td></tr><tr><td>DOV4MM (2025) 152</td><td>pretraining use</td><td>reconstruction shift</td><td></td><td>14 masked models: p &lt; 0.05</td><td></td></tr><tr><td>Dataset-WM Eval. (2025) 150</td><td>diffusion data</td><td>removal benchmark</td><td></td><td></td><td></td></tr><tr><td>Distributor WIC (2025) 133</td><td>user attribution</td><td></td><td></td><td>Customized Stable Diffusion: complete removal</td><td></td></tr><tr><td>DWBench (2026) 217</td><td></td><td>output fingerprint</td><td></td><td>4 generators: hundreds of ms</td><td></td></tr><tr><td></td><td>dataset use</td><td>unified audit</td><td></td><td>25 methods: 7/10 fail below 1% WR</td><td></td></tr><tr><td>PAI (2026) 203</td><td>output ownership</td><td>keyed forensics</td><td></td><td>12 attacks: 98.43% mean</td><td></td></tr><tr><td>RecoverMark (2026) 240</td><td>face ownership</td><td>localize + recover</td><td></td><td>6 attacks: 99.9% ownership</td><td></td></tr><tr><td>PECCAVI (2026) 212</td><td>image ownership</td><td>stable-region mark</td><td></td><td>Paraphrase: WDP 0.90/0.85</td><td></td></tr><tr><td>Cert-LAS (2026) 227</td><td>model ownership</td><td>certified smoothing</td><td></td><td>Certified under parameter perturbation</td><td></td></tr></table>

DWBench shows that high sample-level TPR at 10% marking does not ensure dataset-level verification at 2% participation or with multiple claimants. The other blocks address diferent claims: creator-facing methods test disruption, whereas ownership methods test later identification. Their results are meaningful only with explicit access assumptions, false-positive control, and independent verification.

Table 4: Benchmarks and deployment conditions across the media lifecycle. Rows specify the claim, tested condition, primary metric, and verifier input needed to judge whether results are directly comparable.
<table><tr><td>Benchmark</td><td>Claim</td><td>Tested condition</td><td>Metric</td><td>Verifier input</td></tr><tr><td>Genlmage (2023) 93</td><td>synthetic origin</td><td>unseen generators and image classes</td><td>accuracy</td><td>image</td></tr><tr><td>FakeForm (2026) 175</td><td>synthetic origin</td><td>generator and domain transfer</td><td>accuracy</td><td>image</td></tr><tr><td>RRDataset (2025) 89</td><td>synthetic origin</td><td>sharing and re-digitization</td><td>overall and re-digitized accuracy</td><td>image</td></tr><tr><td>NTIRE (2026) 176</td><td>synthetic origin</td><td>42 generators; 36 transformations</td><td>ROC AUC</td><td>image</td></tr><tr><td>Out-of-box (2026) 183</td><td>detector selection</td><td>12 datasets; 291 generators</td><td>mean accuracy and rank</td><td>image + detector</td></tr><tr><td>Text-rich (2026) 178</td><td>synthetic origin</td><td>six layout domains; JPEG processing</td><td>category accuracy</td><td>image</td></tr><tr><td>AEGIS (2025) 112</td><td>synthetic origin</td><td>in-domain and hyper-realistic hard sets</td><td>accuracy and macro-F1</td><td>video or key frames</td></tr><tr><td>AIGVDBench (2026) 113</td><td>synthetic origin</td><td>31 generators; three generation tasks</td><td>AUC and accuracy</td><td>video</td></tr><tr><td>W-Bench (2025) 199 126</td><td>watermark survival</td><td>semantic, local, and video edits</td><td>TPR at 0.1% FPR</td><td>image + decoder</td></tr><tr><td>UltraEdit (2026) ImageDetectBench</td><td>message recovery</td><td>prompt edit, regeneration, inpainting</td><td>bit accuracy</td><td>image + decoder</td></tr><tr><td>(2026) 215</td><td>synthetic origin</td><td>eight routine and three adversarial attacks</td><td>effectiveness and efficiency</td><td>image or decoder</td></tr><tr><td>DWBench (2026) 217</td><td>dataset use</td><td>low marking rate; multiple owners</td><td>TPR and verification success</td><td>model queries + data</td></tr><tr><td>SAFE Challenge (2026) )177</td><td>authenticity and location</td><td>partial and full synthesis</td><td>detection and localization image</td><td></td></tr></table>

## From preference to remedy

A lifecycle rights system begins by attaching a preference or license to a resource that can still be identified later. When the resource enters a dataset or production workflow, that relationship becomes part of its record. A subsequent technical test can then reconnect the disputed model or output to the earlier history, giving a platform something concrete to review. No link is suficient by itself: a strong technical match cannot identify a rightsholder without a claimant record, and a valid license says little unless it applies to the disputed use.

Accountability requires enough of this chain to be reconstructed that a decision can be explained and both parties can respond. Before such evidence carries consequential weight, its reliability must be tested under the conditions in which it will be used.

## EVALUATING EVIDENCE IN REAL-WORLD PIPELINES

Evaluation must match a method to a specific claim and decision context. The same detector can be adequate for triage and unsafe for public attribution, while a valid manifest can document an edit without establishing that the depicted event occurred. Clean accuracy hides both distinctions. Any evaluation must state what the verifier can access, which conditions may change the signal, and what follows from an error.

For each method, claim, and verification setting, we record an evidence profile with five parts: required access and trust; transformations and adversaries examined; metrics and operating points; known failures; and practical cost. Two methods are directly comparable only where the relevant parts of these profiles align. This is why the preceding tables separate shared evaluations from values retained under original protocols.

Table 4 makes this alignment explicit. Each benchmark exposes a diferent break in the evidence chain, so the rows describe operating conditions rather than one leaderboard.

Across Tables 1–3, three sources of failure emerge. Payload evidence changes with the generator, subject, or acquisition channel. Proactive evidence can disappear when an asset leaves the participating workflow. Rights evidence becomes statistically weak when few protected samples are available or many claimants are tested. Calling all three problems “robustness” obscures what an evaluation needs to vary.

## Distribution shifts that alter evidence

“Generalization” is often used as though every distribution shift were equivalent. A new generator tests dependence on one synthesis process; unfamiliar subject matter exposes content shortcuts; editing and re-acquisition alter the signal after production. These shifts can move in opposite directions, as shown by the model-to-domain gaps in Table 1.<sup>73,89,93,94,175,176,178</sup>

A benchmark must also prevent the dataset from answering the question for the detector. Real and generated samples need comparable content, encoding, and processing histories, with explicit controls for metadata and compression. Video adds duration, motion, frame sampling, generation task, and audio. A face swap, a generated insert, and a fully synthesized scene require separate ground truth rather than one undiferentiated fake class.<sup>110–113</sup>

Aggregate performance becomes informative only when the individual environments remain visible. Alongside the mean, an evaluation needs the worst environment and the dispersion across environments. This reveals whether a method transfers broadly or succeeds because several easy conditions ofset one deployment-breaking failure.

## Routine processing and adaptive attacks

Ordinary media handling estimates whether evidence survives its expected route to a viewer. An adaptive attacker chooses an operation after learning how the verifier works. The same crop, recompression, or regeneration can belong to either setting; the distinction lies in why it was selected and what the attacker knows. Mixing the two produces a robustness number that describes neither deployment nor security.

Watermark security illustrates the distinction. Conventional distortion tests estimate survival during routine processing, whereas semantic regeneration can remove the signal while preserving what a viewer recognizes. Other attacks try to forge or transplant a convincing mark, creating false attribution instead of simple evasion.<sup>126,168,199,206,207,212</sup> Provenance and rights systems face the same broader threat: an attacker can target not just the signal but the records, identities, or verification procedure that give it meaning.<sup>241</sup>

Security evaluation needs a threat model that states the attacker’s knowledge, query access, compute, and control over registries or accounts. The target is the complete verification path, not only a detector or decoder. A watermark may survive while its issuer record is replaced, and a signature may remain valid while the signed history is incomplete.<sup>141,242</sup>

## Calibration for the intended decision

Evidence quality depends on the operating point and the prevalence of the claim. Accuracy and area under the curve can remain high while a low-prevalence deployment produces more false allegations than correct detections. Reports should include calibration at the low false-positive rates required in practice and an abstention region where the data provide too little support. Attribution faces the same problem when the true source is absent from the candidate set.

A threshold has meaning only through the action it triggers. A sensitive detector can prioritize internal review; a public label, loss of monetization, or named attribution requires successively stronger corroboration. Ownership tests add multiple-claim risk because repeated or adaptively selected claims inflate false discovery. Their protocols need an explicit claimant population and null hypothesis, and recovery of a mark remains separate from proof that the claimant holds the asserted right.

Calibration also needs to expose uneven coverage. Confidence can shift with the depicted population, language, genre, or capture device. A precise score is misleading when the calibration set contains little evidence for the case at hand.

![](images/0414fae9dda256ebe8ff6460adc43a5e8194b920bd40b41c4ab19aa3cb503dc5.jpg)  
Figure 5: Lifecycle evaluation from creation to contest. Evidence is disclosed stage by stage as an asset passes through the workflow, allowing the final decision to be tested against valid counterevidence.

## Longitudinal evaluation from creation to contest

Provenance needs a longitudinal test. The issue is not only whether a signal survives, but whether a later verifier can still reach the correct record and interpret it after the asset has been transformed, a key revoked, or an issuer’s status changed. C2PA supplies a structure for signed assertions, yet interoperability and incomplete histories remain empirical questions.<sup>6,137,138,141</sup>

Longitudinal evaluation must also account for how people and institutions interpret the evidence. A provenance label can draw attention to source history while still encouraging viewers to mistake a credible record for proof that the content is true. Wording, placement, and the treatment of missing credentials shape that interpretation.<sup>243–245</sup> Cost and access matter as well: a test that requires private weights or thousands of paid queries may be reproducible in a provider’s laboratory but unavailable to a newsroom or independent creator.

A deployment benchmark can bring these requirements together by following one asset from creation to a contested decision. It records the true operation history, sends the asset through a realistic workflow, and reveals only the evidence that each reviewer would normally receive. The final stage introduces a missing record, conflict, revocation, or valid counterevidence and tests whether the decision updates consistently with the known history and decision rule. Figure 5 summarizes this protocol.

Such a benchmark moves the endpoint from signal recovery to decision quality. It measures whether the claim remains calibrated, whether its supporting record remains reachable, how much review costs, and whether valid counterevidence produces an appropriate reversal.

These tests establish where each evidence profile is reliable and where its authority ends. Combining profiles requires a separate analysis because they may rest on diferent claims and assumptions.

## INTEGRATING EVIDENCE ACROSS THE MEDIA LIFECYCLE

Real cases rarely contain one signal. A detector may assess properties of the payload, a watermark may identify a service, and a signed record may describe one part of the production path. These signals cannot be averaged as though they were repeated measurements of the same fact. Integration begins by preserving which claim each signal supports, where it applies, and what common dependency could make several signals fail together.<sup>6,141</sup>

## Claim-centered case structure

For an asset, we organize the case as a claim graph with three kinds of nodes: signals contain observations and records, claims state the propositions being assessed, and actions represent possible decisions. Links record support, contradiction, and shared dependence. This graph is a detailed implementation of the short path in Equation (2), not a separate scoring model.

The graph prevents support for one proposition from being treated as support for another. A detector can support synthetic origin without identifying a source. A service record can identify where one version was created without establishing permission, and a signed manifest authenticates its assertions rather than the depicted event. The same case can contain supported, contradicted, and unresolved claims.<sup>6,137</sup>

## Dependence and localization

Signals corroborate one another most strongly when they observe diferent parts of the pipeline. A payload detector and a signed service record may do so; two detectors trained on the same data, or two records issued through the same authority, may not. Simple voting can otherwise count one shortcut or one compromised issuer several times.<sup>94,242</sup>

Signals that share a training source, issuer, or vulnerability can be grouped and assessed jointly. Several agreeing outputs from one dependency group should not count as several independent witnesses. Apparent agreement carries less weight when it can be traced to the same underlying failure mode.

Signals must also remain local to the content they describe. A video may be captured in one interval and generated in another, while its soundtrack follows a third history. Evidence should remain attached to the relevant region, interval, or modality. Localized marks and edit records can preserve that binding; the claim graph records how the components contribute to the finished work.<sup>122,126</sup>

## Uncertainty and credential status over time

Missing evidence does not have one meaning. A signal may never have been issued, may have disappeared during circulation, or may be present but no longer valid. Collapsing these states into “no credential” creates false suspicion and an easy route for evasion. A camera that never issued credentials is not equivalent to a marked output whose signal was stripped. Incomplete histories need an explicit unknown state.<sup>6,141</sup>

Conflict is a separate state. A manifest and watermark may name diferent services, or the payload may be inconsistent with the declared history. Authenticating one record does not reconcile it with an incompatible assertion. Suppressing the contradiction would turn uncertainty into false confidence.<sup>6,242</sup>

Time changes the meaning of a credential. A revoked key may indicate later compromise without invalidating every earlier asset, just as a withdrawn license may afect future reuse without rewriting the original creation history. Each evidentiary result needs its own timestamp, method version, calibration record, and credential status. Without that history, a once-valid conclusion can silently outlive the conditions that justified it.<sup>6</sup>

## Proportionate and revisable decisions

Evidence thresholds should rise with the consequence of the decision. A sensitive signal may be enough to prioritize inspection, but restrictions on reach or monetization require stronger calibration and review. Public attribution demands the strongest corroboration because the cost of a false conclusion is high and dificult to reverse.

Decision records show what the evidence supports, what remains uncertain, and which policy turned that assessment into action. Labels need the same precision because “AI-generated,” “credentials unavailable,” and “edited after capture” describe diferent states. The interface can expose the bounded claim without forcing every viewer to inspect the full technical record.<sup>243–245</sup>

An afected party also needs a route to add counterevidence. The case record should show how the new information changes the graph and why the decision is retained or reversed. Reproducing the first result is not enough; the system must also be able to correct it.

## Worked cases across the lifecycle

News and event verification. A newsroom receives a recompressed viral video with no attached credentials. Frame and temporal detectors identify intervals for closer inspection.<sup>89,112,175</sup> Reporters then trace the source and compare recoverable records with the claimed place and time. A verified capture device can support origin while the accompanying caption remains unresolved. The editorial record keeps those conclusions separate.

Service attribution in commercial media. A generated advertisement carries an invisible mark that leads to a signed service record. Matching the two can establish which service produced this version.<sup>5,6,126</sup> It does not establish that the advertiser was entitled to publish it. If embedded and remote records disagree, the conflict remains part of the case rather than being resolved by whichever score is larger.<sup>169,242</sup>

Creator or identity dispute. An artist or individual challenges an advertisement that imitates a protected work or likeness. Earlier records establish what is claimed and when it existed; a technical test examines whether the disputed model or output can be connected to it. Dataset audits become unstable when little protected material is marked or many claimants are tested, so their result cannot carry the case alone.<sup>217,241</sup> The response follows the relationship that can actually be supported and remains open to later evidence.

Across these cases, diferent tools must preserve the relationship between an asset component, a claim, and its supporting record. Detection remains the fallback when no record survives. A mark can reconnect an asset to a richer history, which may then be compared with authorization records. Institutions still decide what the evidence means for the case, but the technical record can keep the basis and limits of that decision open to inspection.<sup>141,217,241</sup>

## DISCUSSION AND RESEARCH AGENDA

The comparisons in this review reveal three bottlenecks that are often described as one robustness problem. Passive detectors fail when the observable signal changes. Proactive methods fail when participation, binding, or record access breaks. Rights mechanisms fail when the afected party cannot supply enough protected material or obtain meaningful access to the model. Because these failures occur at diferent points in the evidence chain, no single robustness measure can capture them.

## Synthetic status as a weak endpoint

As generation becomes an ordinary editing operation, the label “synthetic” carries less information on its own. A photographed scene can contain a generated object; a real performance can be paired with synthetic speech; a fully generated advertisement can be authorized and accurately labeled. The useful question is increasingly which operation materially changed the part of the media relevant to the decision.

This shift changes the endpoint of detection research. Binary origin estimates remain valuable when no history survives, but they become an entry point for investigation rather than a complete account of authenticity. Benchmarks can reflect this by distinguishing full synthesis, local generation, semantic editing, and harmless computational assistance, then attaching the result to the afected region or interval. The materiality threshold belongs to the claim, not to the mere presence of an AI operation.<sup>112,113,177</sup>

## Continuity beyond signal robustness

Robustness asks whether a signal can still be recovered after a transformation. Evidentiary continuity asks a harder question: after that transformation, does the recovered signal still support the same claim about the same component? A watermark copied to another image can be robust but misleading. A manifest retained after an undeclared edit can remain cryptographically valid while no longer describing the visible asset completely.

Benchmarks should evaluate transformation trajectories rather than isolated before-and-after files. A hidden operation log can record compression, cropping, semantic editing, regeneration, insertion into video, and recapture. At each stage, the verifier recovers only the claims that remain justified. A continuity curve would show where a payload binding, remote record, or credential stops supporting its original proposition, instead of compressing that history into one robustness average.<sup>6,141,199,212</sup>

Video and interactive media make continuity stateful. The object of verification is no longer one immutable file but a sequence of versions, components, and user actions. Persistent component identifiers, time-bounded assertions, and open mappings between local signals and remote records become more important than a stronger file-level mark.<sup>6,111–113,246</sup>

## Why more signals do not always add trust

Layered evidence is useful only when the layers fail diferently. Two detectors trained on the same benchmark and backbone are not two independent witnesses. A watermark and manifest issued by the same provider can share an account system, key registry, and revocation service. Agreement then reflects common infrastructure as much as independent corroboration.

This dependence creates concentration risk. A widely adopted issuer can make verification easier while also becoming a common point of failure or exclusion. Benchmark reports can make that risk visible by naming shared training sources, foundation encoders, registries, and authorities. Stress tests can then remove or compromise one dependency and measure how much of the conclusion remains supported.<sup>94,141,242</sup>

Independence also has an institutional dimension. A result that only the model provider can reproduce is not equivalent to evidence available to an afected creator, journalist, or court. Access, privacy, and review cost determine who can challenge the record and how much practical authority the signal deserves.

## Contestability as a technical property

Verification is often evaluated at the moment of the first decision. Real disputes continue after that point. New records appear, keys are revoked, a claimant supplies authorization, or an afected party shows that the detector was applied outside its calibrated domain. A trustworthy system retains enough state to explain the first decision and revisit it without erasing the earlier record.

This makes appeal and correction measurable. An evaluation can introduce valid counterevidence and record whether the claim, explanation, and action change appropriately. It can also measure the time and cost required for an independent party to trigger that review. A system that is accurate but practically impossible to challenge concentrates authority rather than establishing trust.<sup>243–245</sup>

The same infrastructure raises a privacy constraint. Persistent attribution can help resolve misuse while exposing creators and ordinary users to tracking. Evidence records need selective disclosure, purpose limits, and retention rules alongside durable verification. The goal is not to expose the entire history to every verifier, but to reveal enough to support the bounded claim at issue.

## A measurable research program

A near-term testbed could follow the same assets through several lifecycles while retaining a hidden operation history. Detection systems, watermark verifiers, provenance readers, and rights audits would each receive the evidence appropriate to their role. Routine distribution, an adaptive attack, a conflicting record, and valid counterevidence would be introduced in separate stages. This design allows each method to answer its own question while exposing shared dependencies.

The resulting metrics would describe more than signal recovery. Claim calibration measures whether confidence matches the stated proposition. Continuity measures how long the asset– claim binding survives. Dependency-adjusted corroboration discounts evidence with a common failure source. Reversal correctness measures whether valid counterevidence changes the outcome, and access cost records who can realistically obtain review. Together, these measures connect technical performance to the settings in which the result informs a consequential decision.

Over the longer term, evidence should be able to outlive one model or platform without becoming a universal tracking layer. This will require interoperable component identifiers, versioned verification procedures, privacy-preserving records, and independent routes for review. Progress should be measured by whether a system preserves a justified claim through change, acknowledges when the connection breaks, and corrects consequential decisions when stronger evidence arrives.

## CONCLUSIONS

Synthetic media are dificult to verify because the visible asset reveals only part of their history. We have treated each conclusion about that history as a bounded claim supported by observable evidence. Detection remains essential when only the payload survives. Recorded provenance can preserve a stronger connection to production, while rights mechanisms ask what that history means for the people afected by it. These approaches meet at diferent points in the same lifecycle rather than competing to produce one universal authenticity score.

Across the literature, clean accuracy remains a poor guide to evidentiary value. Detector performance declines when the synthesis process or acquisition channel changes. Watermarks and manifests are stronger only when their records remain reachable and verifiable, and rights audits are constrained by the limited access available to afected people. A result becomes meaningful when its claim and operating conditions are explicit. The framework, tables, and lifecycle protocol developed here provide a common way to describe those conditions.

Future infrastructure must preserve evidence as media changes without extending a narrow result to the entire asset. Each claim should remain tied to the relevant component and moment;

dependencies, disagreements, and missing records should remain visible. The same infrastructure must support dynamic media without turning attribution into routine surveillance. Most importantly, consequential decisions must remain open to revision when stronger evidence appears.

## DECLARATION OF INTERESTS

The authors declare no competing interests.

## References

[1] Nightingale, S. J. and Farid, H. AI-synthesized faces are indistinguishable from real faces and more trustworthy. Proceedings of the National Academy of Sciences, 119(8):e2120481119, 2022.

[2] Marra, F., Gragnaniello, D., Verdoliva, L., and Poggi, G. Do GANs leave artificial fingerprints? In IEEE Conference on Multimedia Information Processing and Retrieval, pp. 506–511, 2019.

[3] Wang, S.-Y., Wang, O., Zhang, R., Owens, A., and Efros, A. A. CNN-generated images are surprisingly easy to spot... for now. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8692–8701, 2020.

[4] Wen, Y., Kirchenbauer, J., Geiping, J., and Goldstein, T. Tree-Rings Watermarks: Invisible Fingerprints for Difusion Images. In Advances in Neural Information Processing Systems (NeurIPS), volume 36, pp. 58047–58063, 2023.

[5] Fernandez, P., Couairon, G., Jegou, H., Douze, M., and Furon, T. The Stable Signature: Rooting watermarks in latent difusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 22409–22420, 2023.

[6] Coalition for Content Provenance and Authenticity. Content Credentials: C2PA Technical Specification 2.4, 2026. URL https://spec.c2pa.org/specifications/specifications/2.4/specs/C2PA\_Specificati on.html.

[7] Somepalli, G., Singla, V., Goldblum, M., Geiping, J., and Goldstein, T. Difusion art or digital forgery? Investigating data replication in difusion models. In Proceedings ofthe IEEE/CVFConference on Computer Vision and Pattern Recognition (CVPR), pp. 6048–6058, 2023.

[8] Shan, S., Cryan, J., Wenger, E., Zheng, H., Hanocka, R., and Zhao, B. Y. Glaze: Protecting artists from style mimicry by text-to-image models. In 32nd USENIX Security Symposium (USENIX Security 23), pp. 2187–2204. USENIX Association, 2023.

[9] Verdoliva, L. Media forensics and deepfakes: An overview. IEEE Journal of Selected Topics in Signal Processing, 14(5):910–932, 2020.

[10] Tolosana, R., Vera-Rodriguez, R., Fierrez, J., Morales, A., and Ortega-Garcia, J. Deepfakes and beyond: A survey of face manipulation and fake detection. Information Fusion, 64:131–148, 2020.

[11] Mirsky, Y. and Lee, W. The Creation and Detection of Deepfakes. ACM Computing Surveys, 54(1):1–41, 2021.

[12] Nguyen, T. T., Nguyen, Q. V. H., Nguyen, D. T., Nguyen, D. T., Huynh-The, T., Nahavandi, S., Nguyen, T. T., Pham, Q.-V., and Nguyen, C. M. Deep learning for deepfakes creation and detection: A survey. Computer Vision and Image Understanding, 223:103525, 2022.

[13] Masood, M., Nawaz, M., Malik, K. M., Javed, A., Irtaza, A., and Malik, H. Deepfakes generation and detection: State-of-the-art, open challenges, countermeasures, and way forward. Applied Intelligence, 53 (4):3974–4026, 2023.

[14] Rana, M. S., Nobi, M. N., Murali, B., and Sung, A. H. Deepfake detection: A systematic literature review. IEEE Access, 10:25494–25513, 2022.

[15] Luo, H., Li, L., and Li, J. Digital watermarking technology for AI-generated images: A survey. Mathematics, 13(4):651, 2025.

[16] Cao, J., Li, Q., Zhang, Z., Ni, J., and Lu, R. Secure and robust watermarking for AI-generated images: A comprehensive survey, arXiv preprint arXiv:2510.02384, 2025.

[17] Kumar, N. and Singh, A. K. Artificial intelligence content detection techniques using watermarking: A survey. Image and Vision Computing, 163:105728, 2025.

[18] Lai, Z., Arif, S., Feng, C., Liao, G., and Wang, C. Enhancing deepfake detection: Proactive forensics techniques using digital watermarking. Computers, Materials and Continua, 82(1):73–102, 2025.

[19] Mahara, A. and Rishe, N. Methods and Trends in Detecting AI-Generated Images: A Comprehensive Review, arXiv preprint arXiv:2502.15176, 2025.

[20] Kingma, D. P. and Welling, M. Auto-encoding variational Bayes. In International Conference on Learning Representations (ICLR), 2014.

[21] Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., and Bengio, Y. Generative adversarial nets. In Advances in Neural Information Processing Systems (NeurIPS), volume 27, pp. 2672–2680, 2014.

[22] Radford, A., Metz, L., and Chintala, S. Unsupervised representation learning with deep convolutional generative adversarial networks. In International Conference on Learning Representations (ICLR), 2016.

[23] Brock, A., Donahue, J., and Simonyan, K. Large Scale GAN Training for High Fidelity Natural Image Synthesis. In International Conference on Learning Representations (ICLR), 2019.

[24] Karras, T., Laine, S., and Aila, T. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4396–4405, 2019.

[25] Karras, T., Laine, S., Aittala, M., Hellsten, J., Lehtinen, J., and Aila, T. Analyzing and improving the image quality of StyleGAN. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8107–8116, 2020.

[26] Ho, J., Jain, A., and Abbeel, P. Denoising difusion probabilistic models. In Advances in Neural Information Processing Systems (NeurIPS), volume 33, pp. 6840–6851, 2020.

[27] Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic diferential equations. In International Conference on Learning Representations (ICLR), 2021.

[28] Song, J., Meng, C., and Ermon, S. Denoising difusion implicit models. In International Conference on Learning Representations (ICLR), 2021.

[29] Nichol, A., Dhariwal, P., Ramesh, A., Shyam, P., Mishkin, P., McGrew, B., Sutskever, I., and Chen, M. GLIDE: Towards photorealistic image generation and editing with text-guided difusion models, arXiv preprint arXiv:2112.10741, 2021.

[30] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent difusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10674–10685, 2022.

[31] Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Muller, J., Penna, J., and Rombach, R. SDXL: Improving latent difusion models for high-resolution image synthesis, arXiv preprint arXiv:2307.01952, 2023.

[32] Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., Podell, D., Dockhorn, T., English, Z., and Rombach, R. Scaling rectified flow transformers for highresolution image synthesis. In Proceedings of the 41st International Conference on Machine Learning (ICML), volume 235, pp. 12606–12633. PMLR, 2024.

[33] Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning (ICML), volume 139 of Proceedings of Machine Learning Research, pp. 8748–8763. PMLR, 2021.

[34] Ramesh, A., Pavlov, M., Goh, G., Gray, S., Voss, C., Radford, A., Chen, M., and Sutskever, I. Zeroshot text-to-image generation. In Proceedings of the 38th International Conference on Machine Learning (ICML), volume 139 of Proceedings of Machine Learning Research, pp. 8821–8831. PMLR, 2021.

[35] Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., and Chen, M. Hierarchical text-conditional image generation with CLIP latents, arXiv preprint arXiv:2204.06125, 2022.

[36] Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E. L., Ghasemipour, K., Lopes, R. G., Ayan, B. K., Salimans, T., Ho, J., Fleet, D. J., and Norouzi, M. Photorealistic text-to-image difusion models with deep language understanding. In Advances in Neural Information Processing Systems (NeurIPS), volume 35, pp. 36479–36494, 2022.

[37] Zhang, L., Rao, A., and Agrawala, M. Adding Conditional Control to Text-to-Image Difusion Models. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pp. 3813–3824, 2023.

[38] Brooks, T., Holynski, A., and Efros, A. A. InstructPix2Pix: Learning to Follow Image Editing Instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 18392–18402, 2023.

[39] Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M., and Aberman, K. DreamBooth: Fine Tuning Textto-Image Difusion Models for Subject-Driven Generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 22500–22510, 2023.

[40] OpenAI. DALL-E 3 Is Now Available in ChatGPT Plus and Enterprise, 2023. URL https://openai.com/i ndex/dall-e-3-is-now-available-in-chatgpt-plus-and-enterprise/.

[41] Wu, C., Li, J., Zhou, J., Lin, J., Gao, K., Yan, K., Yin, S., Bai, S., Xu, X., Chen, Y., Chen, Y., Tang, Z., Zhang, Z., Wang, Z., Yang, A., Yu, B., Cheng, C., Liu, D., Li, D., Zhang, H., Meng, H., Wei, H., Ni, J., Chen, K., Cao, K., Peng, L., Qu, L., Wu, M., Wang, P., Yu, S., Wen, T., Feng, W., Xu, X., Wang, Y., Zhang, Y., Zhu, Y., Wu, Y., Cai, Y., and Liu, Z. Qwen-Image Technical Report, arXiv preprint arXiv:2508.02324,

2025.

[42] OpenAI. Introducing 4o Image Generation, 2025. URL https://openai.com/index/introducing-4o-i mage-generation/.

[43] Google DeepMind. Imagen 4, 2025. URL https://deepmind.google/models/imagen/.

[44] Oord, A. v. d., Dieleman, S., Zen, H., Simonyan, K., Vinyals, O., Graves, A., Kalchbrenner, N., Senior, A., and Kavukcuoglu, K. WaveNet: A Generative Model for Raw Audio, arXiv preprint arXiv:1609.03499, 2016.

[45] Dhariwal, P., Jun, H., Payne, C., Kim, J. W., Radford, A., and Sutskever, I. Jukebox: A Generative Model for Music, arXiv preprint arXiv:2005.00341, 2020.

[46] Kreuk, F., Synnaeve, G., Polyak, A., Singer, U., Défossez, A., Copet, J., Parikh, D., Taigman, Y., and Adi, Y. AudioGen: Textually Guided Audio Generation, arXiv preprint arXiv:2209.15352, 2022.

[47] Copet, J., Kreuk, F., Gat, I., Remez, T., Kant, D., Synnaeve, G., Adi, Y., and Défossez, A. Simple and Controllable Music Generation. In Advances in Neural Information Processing Systems (NeurIPS), volume 36, pp. 47704–47720, 2023.

[48] Agostinelli, A., Denk, T. I., Borsos, Z., Engel, J., Verzetti, M., Caillon, A., Huang, Q., Jansen, A., Roberts, A., Tagliasacchi, M., Sharifi, M., Zeghidour, N., and Frank, C. MusicLM: Generating Music From Text, arXiv preprint arXiv:2301.11325, 2023.

[49] Liu, H., Chen, Z., Yuan, Y., Mei, X., Liu, X., Mandic, D., Wang, W., and Plumbley, M. D. AudioLDM: Text-to-Audio Generation with Latent Difusion Models, arXiv preprint arXiv:2301.12503, 2023.

[50] Tulyakov, S., Liu, M.-Y., Yang, X., and Kautz, J. MoCoGAN: Decomposing Motion and Content for Video Generation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1526–1535, 2018.

[51] Ho, J., Chan, W., Saharia, C., Whang, J., Gao, R., Gritsenko, A., Kingma, D. P., Poole, B., Norouzi, M., Fleet, D. J., and Salimans, T. Imagen Video: High definition video generation with difusion models, arXiv preprint arXiv:2210.02303, 2022.

[52] Singer, U., Polyak, A., Hayes, T., Yin, X., An, J., Zhang, S., Hu, Q., Yang, H., Ashual, O., Gafni, O., Parikh, D., Gupta, S., and Taigman, Y. Make-A-Video: Text-to-video generation without text-video data, arXiv preprint arXiv:2209.14792, 2022.

[53] Blattmann, A., Dockhorn, T., Kulal, S., Mendelevitch, D., Kilian, M., Lorenz, D., Levi, Y., English, Z., Voleti, V., Letts, A., Jampani, V., and Rombach, R. Stable Video Difusion: Scaling latent video difusion models to large datasets, arXiv preprint arXiv:2311.15127, 2023.

[54] Polyak, A., Zohar, A., Brown, A., Tjandra, A., Sinha, A., Lee, A., Vyas, A., Shi, B., Ma, C.-Y., Chuang, C.-Y., Yan, D., et al. Movie Gen: A Cast of Media Foundation Models, arXiv preprint arXiv:2410.13720, 2024.

[55] Kong, W., Tian, Q., Zhang, Z., Min, R., Dai, Z., Zhou, J., Xiong, J., Li, X., Wu, B., Zhang, J., Wu, K., Lin, Q., Yuan, J., Long, Y., Wang, A., et al. HunyuanVideo: A Systematic Framework for Large Video Generative Models, arXiv preprint arXiv:2412.03603, 2024.

[56] Gao, Y., Guo, H., Hoang, T., Huang, W., Jiang, L., Kong, F., Li, H., Li, J., Li, L., Li, X., Li, X., Li, Y., Lin, S., Lin, Z., Liu, J., Liu, S., Nie, X., Qing, Z., Ren, Y., Sun, L., Tian, Z., Wang, R., Wang, S., Wei, G., Wu, G., Wu, J., Xia, R., Xiao, F., Xiao, X., Yan, J., Yang, C., Yang, J., Yang, R., Yang, T., Yang, Y., Ye, Z., Zeng, X., Zeng, Y., Zhang, H., Zhao, Y., Zheng, X., Zhu, P., Zou, J., and Zuo, F. Seedance 1.0: Exploring the Boundaries of Video Generation Models, arXiv preprint arXiv:2506.09113, 2025.

[57] Google DeepMind. Veo: Our Leading Video Generation Model, 2025. URL https://deepmind.google/ models/veo/.

[58] Ha, D. and Schmidhuber, J. World Models, arXiv preprint arXiv:1803.10122, 2018.

[59] Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., and Ng, R. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. In European Conference on Computer Vision (ECCV), pp. 405–421, 2020.

[60] Poole, B., Jain, A., Barron, J. T., and Mildenhall, B. DreamFusion: Text-to-3D using 2D Difusion, arXiv preprint arXiv:2209.14988, 2022.

[61] Nichol, A., Jun, H., Dhariwal, P., Mishkin, P., and Chen, M. Point-E: A System for Generating 3D Point Clouds from Complex Prompts, arXiv preprint arXiv:2212.08751, 2022.

[62] Lin, C.-H., Gao, J., Tang, L., Takikawa, T., Zeng, X., Huang, X., Kreis, K., Fidler, S., Liu, M.-Y., and Lin, T.-Y. Magic3D: High-Resolution Text-to-3D Content Creation. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 300–309, 2023.

[63] Jun, H. and Nichol, A. Shap-E: Generating Conditional 3D Implicit Functions, arXiv preprint arXiv:2305.02463, 2023.

[64] Kerbl, B., Kopanas, G., Leimkühler, T., and Drettakis, G. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Transactions on Graphics, 42(4):1–14, 2023.

[65] Bruce, J., Dennis, M., Edwards, A., Parker-Holder, J., Shi, Y., Hughes, E., Lai, M., Mavalankar, A., Steigerwald, R., Apps, C., Aytar, Y., Bechtle, S., Behbahani, F., Chan, S., Heess, N., Gonzalez, L., Osindero, S., Ozair, S., Reed, S., Zhang, J., Zolna, K., Clune, J., de Freitas, N., Singh, S., and Rocktäschel, T. Genie:

Generative Interactive Environments, arXiv preprint arXiv:2402.15391, 2024.

[66] Valevski, D., Leviathan, Y., Arar, M., and Fruchter, S. Difusion Models Are Real-Time Game Engines, arXiv preprint arXiv:2408.14837, 2024.

[67] NVIDIA, Agarwal, N., Ali, A., Bala, M., Balaji, Y., Barker, E., Cai, T., Chattopadhyay, P., Chen, Y., Cui, Y., Ding, Y., et al. Cosmos World Foundation Model Platform for Physical AI, arXiv preprint arXiv:2501.03575, 2025.

[68] Zhao, Z., Lai, Z., Lin, Q., Zhao, Y., Liu, H., Yang, S., Feng, Y., Yang, M., Zhang, S., Yang, X., et al. Hunyuan3D 2.0: Scaling Difusion Models for High Resolution Textured 3D Assets Generation, arXiv preprint arXiv:2501.12202, 2025.

[69] NVIDIA et al. Cosmos 3: Omnimodal World Models for Physical AI, arXiv preprint arXiv:2606.02800, 2026.

[70] Yu, N., Davis, L. S., and Fritz, M. Attributing fake images to GANs: Learning and analyzing GAN fingerprints. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 7556–7566, 2019.

[71] Chai, L., Bau, D., Lim, S.-N., and Isola, P. What makes fake images detectable? Understanding properties that generalize. In European Conference on Computer Vision (ECCV), pp. 103–120, 2020.

[72] Gragnaniello, D., Cozzolino, D., Marra, F., Poggi, G., and Verdoliva, L. Are GAN Generated Images Easy to Detect? A Critical Analysis of the State-of-the-Art. In 2021 IEEE International Conference on Multimedia and Expo (ICME), pp. 1–6. IEEE, 2021.

[73] Bammey, Q. Synthbuster: Towards detection of difusion model generated images. IEEE Open Journal of Signal Processing, 5:1–9, 2024.

[74] Tan, C., Liu, H., Zhao, Y., Wei, S., Gu, G., Liu, P., and Wei, Y. Rethinking the up-sampling operations in CNN-based generative network for generalizable deepfake detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 28130–28139, 2024.

[75] Bird, J. J. and Lotfi, A. CIFAKE: Image classification and explainable identification of AI-generated synthetic images. IEEE Access, 12:15642–15650, 2024.

[76] Frank, J., Eisenhofer, T., Schonherr, L., Fischer, A., Kolossa, D., and Holz, T. Leveraging frequency analysis for deep fake image recognition. In Proceedings of the 37th International Conference on Machine Learning (ICML), volume 119 of Proceedings of Machine Learning Research, pp. 3247–3258. PMLR, 2020.

[77] Durall, R., Keuper, M., and Keuper, J. Watch your up-convolution: CNN based generative deep neural networks are failing to reproduce spectral distributions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 7887–7896, 2020.

[78] Qian, Y., Yin, G., Sheng, L., Chen, Z., and Shao, J. Thinking in frequency: Face forgery detection by mining frequency-aware clues. In European Conference on Computer Vision (ECCV), pp. 86–103, 2020.

[79] Liu, H., Li, X., Zhou, W., Chen, Y., He, Y., Xue, H., Zhang, W., and Yu, N. Spatial-phase shallow learning: Rethinking face forgery detection in frequency domain. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 772–781, 2021.

[80] Mandelli, S., Bonettini, N., Bestagini, P., and Tubaro, S. Detecting GAN-generated images by orthogonal training of multiple CNNs. In IEEE International Conference on Image Processing (ICIP), pp. 3091–3095, 2022.

[81] Corvi, R., Cozzolino, D., Zingarini, G., Poggi, G., Nagano, K., and Verdoliva, L. On the detection of synthetic images generated by difusion models. In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1–5, 2023.

[82] Wang, Z., Bao, J., Zhou, W., Wang, W., Hu, H., Chen, H., and Li, H. DIRE for difusion-generated image detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 22388–22398, 2023.

[83] Cazenavette, G., Sud, A., Leung, T., and Usman, B. FakeInversion: Learning to detect images from unseen text-to-image models by inverting Stable Difusion. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10759–10769, 2024.

[84] Ricker, J., Lukovnikov, D., and Fischer, A. AEROBLADE: Training-free detection of latent difusion images using autoencoder reconstruction error. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9130–9140, 2024.

[85] Luo, Y., Du, J., Yan, K., and Ding, S. LaRE2: Latent reconstruction error based method for difusiongenerated image detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 17006–17015, 2024.

[86] Chen, B., Zeng, J., Yang, J., and Yang, R. DRCT: Difusion reconstruction contrastive training towards universal detection of difusion generated images. In Proceedings of the 41st International Conference on Machine Learning (ICML), volume 235 of Proceedings of Machine Learning Research, pp. 7621–7639. PMLR, 2024.

[87] Chu, B., Xu, X., Wang, X., Zhang, Y., You, W., and Zhou, L. FIRE: Robust detection of difusion-generated images via frequency-guided reconstruction error. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 12830–12839, 2025.

[88] Lin, H., Qin, J., Wu, X., Chen, T., and Yang, Z. Revisiting DIRE: Towards universal AI-generated image

detection. Neural Networks, 193:108084, 2026.

[89] Li, C., Wang, X., Li, M., Miao, B., Sun, P., Zhang, Y., Ji, X., and Zhu, Y. Bridging the gap between ideal and real-world evaluation: Benchmarking AI-generated image detection in challenging scenarios. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 20379–20389, 2025.

[90] Zhou, Y., He, X., Lin, K., Fan, B., Ding, F., and Li, B. Breaking latent prior bias in detectors for generalizable AIGC image detection. In Advances in Neural Information Processing Systems (NeurIPS), volume 38, pp. 34606–34636, 2025.

[91] Ojha, U., Li, Y., and Lee, Y. J. Towards universal fake image detectors that generalize across generative models. In Proceedings ofthe IEEE/CVFConference on ComputerVision and Pattern Recognition (CVPR), pp. 24480–24489, 2023.

[92] Sha, Z., Li, Z., Yu, N., and Zhang, Y. DE-FAKE: Detection and attribution of fake images generated by text-to-image generation models. In Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security, pp. 3418–3432, 2023.

[93] Zhu, M., Chen, H., Yan, Q., Huang, X., Lin, G., Li, W., Tu, Z., Hu, H., Hu, J., and Wang, Y. GenImage: A million-scale benchmark for detecting AI-generated image. In Advances in Neural Information Processing Systems (NeurIPS), volume 36, pp. 77771–77782, 2023.

[94] Yan, S., Li, O., Cai, J., Hao, Y., Jiang, X., Hu, Y., and Xie, W. A sanity check for AI-generated image detection. In International Conference on Learning Representations (ICLR), 2025.

[95] Ha, A. Y. J., Passananti, J., Bhaskar, R., Shan, S., Southen, R., Zheng, H., and Zhao, B. Y. Organic or difused: Can we distinguish human art from AI-generated images? In ACM Conference on Computer and Communications Security, pp. 4822–4836, 2024.

[96] Wen, H., He, Y., Huang, Z., Li, T., Yu, Z., Huang, X., Qi, L., Wu, B., Li, X., and Cheng, G. BusterX: MLLM-Powered AI-Generated Video Forgery Detection and Explanation, arXiv preprint arXiv:2505.12620, 2025.

[97] Pellegrini, L., Cozzolino, D., Pandolfini, S., Maltoni, D., Ferrara, M., Verdoliva, L., Prati, M., and Ramilli, M. AI-GenBench: A new ongoing benchmark for AI-generated image detection. In 2025 International Joint Conference on Neural Networks (IJCNN), pp. 1–9, 2025.

[98] Afchar, D., Nozick, V., Yamagishi, J., and Echizen, I. MesoNet: A compact facial video forgery detection network. In IEEE International Workshop on Information Forensics and Security (WIFS), pp. 1–7, 2018.

[99] Sabir, E., Cheng, J., Jaiswal, A., AbdAlmageed, W., Masi, I., and Natarajan, P. Recurrent convolutional strategies for face manipulation detection in videos. In IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pp. 80–87, 2019.

[100] Rossler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., and Nießner, M. FaceForensics++: Learning to detect manipulated facial images. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 1–11, 2019.

[101] Nguyen, H. H., Yamagishi, J., and Echizen, I. Capsule-forensics: Using capsule networks to detect forged images and videos. In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 2307–2311, 2019.

[102] Li, Y., Yang, X., Sun, P., Qi, H., and Lyu, S. Celeb-DF: A large-scale challenging dataset for deepfake forensics. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3204–3213, 2020.

[103] Dolhansky, B., Bitton, J., Pflaum, B., Lu, J., Howes, R., Wang, M., and Ferrer, C. C. The Deepfake Detection Challenge (DFDC) Dataset, arXiv preprint arXiv:2006.07397, 2020.

[104] Jiang, L., Li, R., Wu, W., Qian, C., and Loy, C. C. DeeperForensics-1.0: A large-scale dataset for realworld face forgery detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2886–2895, 2020.

[105] Li, Y., Chang, M.-C., and Lyu, S. In Ictu Oculi: Exposing AI created fake videos by detecting eye blinking. In IEEE International Workshop on Information Forensics and Security (WIFS), pp. 1–7, 2018.

[106] Li, L., Bao, J., Zhang, T., Yang, H., Chen, D., Wen, F., and Guo, B. Face X-ray for more general face forgery detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5000–5009, 2020.

[107] Li, Y., Zhu, D., Cui, X., and Lyu, S. Celeb-DF++: A Large-scale Challenging Video DeepFake Benchmark for Generalizable Forensics, arXiv preprint arXiv:2507.18015, 2025.

[108] Haliassos, A., Vougioukas, K., Petridis, S., and Pantic, M. Lips do not lie: A generalisable and robust approach to face forgery detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5039–5049, 2021.

[109] Zhao, H., Zhou, W., Chen, D., Wei, T., Zhang, W., and Yu, N. Multi-attentional deepfake detection. In Proceedings ofthe IEEE/CVFConference on Computer Vision and Pattern Recognition (CVPR), pp. 2185– 2194, 2021.

[110] Ma, L., Yan, Z., Guo, Q., Liao, Y., Yu, H., and Zhou, P. Detecting AI-Generated Video via Frame Consistency. In 2025 IEEE International Conference on Multimedia and Expo (ICME), pp. 1–6. IEEE, 2025.

[111] Chen, H., Hong, Y., Huang, Z., Xu, Z., Gu, Z., Li, Y., Lan, J., Zhu, H., Zhang, J., Wang, W., and Li, H. De-Mamba: AI-generated video detection on million-scale GenVideo benchmark. Science China Information Sciences, 69(6):162103, 2026.

[112] Li, J., Zhang, X., and Zhou, J. T. AEGIS: Authenticity evaluation benchmark for AI-generated video sequences. In Proceedings of the ACM International Conference on Multimedia (ACM MM), pp. 13346– 13353, 2025.

[113] Ma, L., Xue, Z., Wang, Y., Yan, Z., Xu, J., Jiang, X., Yu, H., Liao, Y., and Bi, Z. Your One-Stop Solution for AI-Generated Video Detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4458–4470, June 2026.

[114] Wang, Z., Bao, J., Zhou, W., Wang, W., and Li, H. AltFreezing for more general video face forgery detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4129–4138, 2023.

[115] Xu, Y., Liang, J., Jia, G., Yang, Z., Zhang, Y., and He, R. TALL: Thumbnail layout for deepfake video detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 22601–22611, 2023.

[116] Ni, Z., Yan, Q., Huang, M., Yuan, T., Tang, Y., Hu, H., Chen, X., and Wang, Y. GenVidBench: A 6-Million Benchmark for AI-Generated Video Detection, arXiv preprint arXiv:2501.11340, 2025.

[117] Park, K., Yang, Y., Yi, J., Zheng, S., Shen, Y., Han, D., Shan, C., Muaz, M., and Qiu, L. VidGuard-R1: AI-Generated Video Detection and Explanation via Reasoning MLLMs and RL, arXiv preprint arXiv:2510.02282, 2025.

[118] Zhu, J., Kaplan, R., Johnson, J., and Fei-Fei, L. HiDDeN: Hiding data with deep networks. In European Conference on Computer Vision (ECCV), pp. 682–697, 2018.

[119] Zhang, K. A., Xu, L., Cuesta-Infante, A., and Veeramachaneni, K. Robust Invisible Video Watermarking with Attention, arXiv preprint arXiv:1909.01285, 2019.

[120] Tancik, M., Mildenhall, B., and Ng, R. StegaStamp: Invisible hyperlinks in physical photographs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2114– 2123, 2020.

[121] Xu, R., Hu, M., Lei, D., Li, Y., Lowe, D., Gorevski, A., Wang, M., Ching, E., and Deng, A. InvisMark: Invisible and robust watermarking for AI-generated image provenance. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 909–918, 2025.

[122] Sander, T., Fernandez, P., Durmus, A., Furon, T., and Douze, M. Watermark Anything with Localized Messages. In International Conference on Learning Representations (ICLR), 2025.

[123] Bui, T., Agarwal, S., Yu, N., and Collomosse, J. RoSteALS: Robust Steganography using Autoencoder Latent Space. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 933–942, 2023.

[124] Bui, T., Agarwal, S., and Collomosse, J. TrustMark: Universal Watermarking for Arbitrary Resolution Images, arXiv preprint arXiv:2311.18297, 2023.

[125] Jia, Z., Fang, H., and Zhang, W. MBRS: Enhancing Robustness of DNN-based Watermarking by Mini-Batch of Real and Simulated JPEG Compression. In Proceedings ofthe 29th ACM International Conference on Multimedia, pp. 41–49, 2021.

[126] Gan, Z., Liu, C., Tang, Y., Wang, B., Cui, S., Wang, W., and Zhang, X. GenPTW: Latent Image Watermarking for Provenance Tracing and Tamper Localization. Proceedings ofthe AAAI Conference on Artificial Intelligence, 40(5):4085–4093, 2026.

[127] Yang, H., Liu, B., Xu, X., Xu, C., Yu, Y., Huang, Z., Wang, Y., and He, S. StableGuard: Towards unified copyright protection and tamper localization in latent difusion models. In Advances in Neural Information Processing Systems (NeurIPS), volume 38, pp. 16462–16490, 2025.

[128] Goren, N., Katzir, O., Nakarmi, A., Ronen, E., Sharif, M., and Patashnik, O. NoisePrints: Distortion-free watermarks for authorship in private difusion models, arXiv preprint arXiv:2510.13793, 2025.

[129] Liu, Y., Li, Z., Backes, M., Shen, Y., and Zhang, Y. Watermarking difusion model, arXiv preprint arXiv:2305.12502, 2023.

[130] Pan, L., Guan, S., Fu, Z., Si, L., Wang, H., Wang, Z., Li, H., Hu, X., King, I., Yu, P. S., Liu, A., and Wen, L. MarkDifusion: An open-source toolkit for generative watermarking of latent difusion models, arXiv preprint arXiv:2509.10569, 2025.

[131] Yang, Z., Zeng, K., Chen, K., Fang, H., Zhang, W., and Yu, N. Gaussian Shading: Provable Performance-Lossless Image Watermarking for Difusion Models. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 12162–12171, 2024.

[132] Google DeepMind. Identifying AI-generated images with SynthID, 2023. URL https://deepmind.googl e/discover/blog/identifying-ai-generated-images-with-synthid/.

[133] Fei, J., Dai, Y., Yang, W., and Xia, Z. Distributor-centric model watermarking for image generative models. Knowledge-Based Systems, 329:114422, 2025.

[134] Gowal, S., Bunel, R., Stimberg, F., Stutz, D., Ortiz-Jimenez, G., Kouridi, C., Vecerik, M., Hayes, J., Rebufi, S.-A., Bernard, P., Gamble, C., Horvath, M. Z., Kaczmarczyck, F., Kaskasoli, A., Petrov, A., Shumailov, I.,

Thotakuri, M., Wiles, O., Yung, J., Ahmed, Z., Martin, V., Rosen, S., Savcak, C., Senoner, A., Vyas, N., and Kohli, P. SynthID-Image: Image watermarking at internet scale, arXiv preprint arXiv:2510.09263, 2025.

[135] Kim, C., Min, K., Patel, M., Cheng, S., and Yang, Y. WOUAF: Weight Modulation for User Attribution and Fingerprinting in Text-to-Image Difusion Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8974–8983, 2024.

[136] Coalition for Content Provenance and Authenticity. C2PA Technical Specification 2.2, 2025. URL https: //spec.c2pa.org/specifications/specifications/2.2/.

[137] Coalition for Content Provenance and Authenticity. C2PA and Content Credentials Explainer 2.2, 2025. URL https://spec.c2pa.org/specifications/specifications/2.2/explainer/Explainer.html.

[138] National Security Agency and Australian Signals Directorate’s Australian Cyber Security Centre and Canadian Centre for Cyber Security and United Kingdom National Cyber Security Centre. Content Credentials: Strengthening Multimedia Integrity in the Generative AI Era, 2025. URL https://media.defense.gov/20 25/Jan/29/2003634788/-1/-1/0/CSI-CONTENT-CREDENTIALS.PDF.

[139] Content Authenticity Initiative. Content Authenticity Initiative: Content Credentials implementation resources, 2024. URL https://contentauthenticity.org/.

[140] Longpre, S., Mahari, R., Chen, A., Obeng-Marnu, N., Sileo, D., Brannon, W., Muennighof, N., Khazam, N., Kabbara, J., Perisetla, K., Wu, X., Shippole, E., Bollacker, K., Wu, T., Villa, L., Pentland, S., and Hooker, S. The Data Provenance Initiative: A Large Scale Audit of Dataset Licensing and Attribution in AI, arXiv preprint arXiv:2310.16787, 2023.

[141] Golaszewski, E., Krawetz, N., Sherman, A. T., Zieglar, E., Matukumalli, S. K., Yus, R., Kegley, C. L., Barthel, M., Bowman, W., Barot, B., and Kullman, K. Verifying Provenance of Digital Media: Why the C2PA Specifications Fall Short, arXiv preprint arXiv:2604.24890, 2026.

[142] European Union. Regulation (EU) 2024/1689 Laying Down Harmonised Rules on Artificial Intelligence, 2024. URL https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng.

[143] The White House. Fact Sheet: Biden-Harris Administration Secures Voluntary Commitments from Leading Artificial Intelligence Companies to Manage the Risks Posed by AI, 2023. URL https://bidenwhitehous e.archives.gov/briefing-room/statements-releases/2023/07/21/fact-sheet-biden-harris-adm inistration-secures-voluntary-commitments-from-leading-artificial-intelligence-compani es-to-manage-the-risks-posed-by-ai/.

[144] Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., and Crawford, K. Datasheets for Datasets. Communications of the ACM, 64(12):86–92, 2021.

[145] Balan, K., Agarwal, S., Jenni, S., Parsons, A., Gilbert, A., and Collomosse, J. EKILA: Synthetic Media Provenance and Attribution for Generative Art. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 913–922, 2023.

[146] Jewitt, J., Rajbahadur, G. K., Li, H., Adams, B., and Hassan, A. E. Permissive-Washing in the Open AI Supply Chain: A Large-Scale Audit of License Integrity. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2, pp. 2014–2025, 2026.

[147] United States Copyright Ofice. Copyright and artificial intelligence, Part 2: Copyrightability, 2025. URL https://www.copyright.gov/ai/.

[148] Dziedzic, A., Duan, H., Kaleem, M. A., Dhawan, N., Guan, J., Cattan, Y., Boenisch, F., and Papernot, N. Dataset Inference for Self-Supervised Models. In Advances in Neural Information Processing Systems (NeurIPS), volume 35, pp. 12058–12070, 2022.

[149] Bouaziz, W., Usunier, N., and El-Mhamdi, E.-M. Data Taggants: Dataset Ownership Verification via Harmless Targeted Data Poisoning. In International Conference on Learning Representations (ICLR), 2025.

[150] Wang, X., Sun, H., Sun, W., Xue, K., Zhou, W., Zhang, J., Sun, W., Zhu, D., Min, X., Jia, J., and Fang, Z. Evaluating dataset watermarking for fine-tuning traceability of customized difusion models: A comprehensive benchmark and removal approach, arXiv preprint arXiv:2511.19316, 2025.

[151] Xie, Y., Song, J., Xue, M., Zhang, H., Wang, X., Hu, B., Chen, G., and Song, M. Dataset Ownership Verification in Contrastive Pre-trained Models, arXiv preprint arXiv:2502.07276, 2025.

[152] Xie, Y., Song, J., Shan, Y., Zhang, X., Wan, Y., Zhang, S., Duan, J., and Song, M. Dataset Ownership Verification for Pre-trained Masked Models. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 3132–3142, 2025.

[153] Carlini, N., Hayes, J., Nasr, M., Jagielski, M., Sehwag, V., Tramèr, F., Balle, B., Ippolito, D., and Wallace, E. Extracting training data from difusion models. In 32nd USENIX Security Symposium (USENIX Security 23), pp. 5253–5270. USENIX Association, 2023.

[154] Webster, R. A reproducible extraction of training images from difusion models, arXiv preprint arXiv:2305.08694, 2023.

[155] Somepalli, G., Singla, V., Goldblum, M., Geiping, J., and Goldstein, T. Understanding and mitigating copying in difusion models. In Advances in Neural Information Processing Systems (NeurIPS), volume 36, pp. 47783–47803, 2023.

[156] Ding, W., Li, C. Y., Shan, S., Zhao, B. Y., and Zheng, H. Understanding implosion in text-to-image generative models. In ACM Conference on Computer and Communications Security, pp. 1211–1225, 2024.

[157] Shan, S., Wenger, E., Zhang, J., Li, H., Zheng, H., and Zhao, B. Y. Fawkes: Protecting privacy against unauthorized deep learning models. In 29th USENIX Security Symposium (USENIX Security 20), pp. 1589–1604. USENIX Association, 2020.

[158] Cherepanova, V., Goldblum, M., Foley, H., Duan, S., Dickerson, J., Taylor, G., and Goldstein, T. LowKey: Leveraging Adversarial Attacks to Protect Social Media Users from Facial Recognition. In ICLR, 2021.

[159] Huang, H., Ma, X., Erfani, S. M., Bailey, J., and Wang, Y. Unlearnable examples: Making personal data unexploitable. In International Conference on Learning Representations (ICLR), 2021.

[160] Shan, S., Ding, W., Passananti, J., Wu, S., Zheng, H., and Zhao, B. Y. Nightshade: Prompt-specific poisoning attacks on text-to-image generative models. In IEEE Symposium on Security and Privacy, pp. 807–825, 2024.

[161] Liang, C. and Wu, X. Mist: Towards Improved Adversarial Examples for Difusion Models, arXiv preprint arXiv:2305.12683, 2023.

[162] Huang, J., Guo, Z., Luo, G., Qian, Z., Li, S., and Zhang, X. Disentangled Style Domain for Implicit z-Watermark Towards Copyright Protection. In Advances in Neural Information Processing Systems (NeurIPS), volume 37, pp. 55810–55830, 2024.

[163] Salman, H., Khaddaj, A., Leclerc, G., Ilyas, A., and Madry, A. Raising the cost of malicious AI-powered image editing. In Proceedings of the 40th International Conference on Machine Learning (ICML), volume 202 of Proceedings of Machine Learning Research, pp. 29894–29918. PMLR, 2023.

[164] Van Le, T., Phung, H., Nguyen, T. H., Dao, Q., Tran, N. N., and Tran, A. Anti-DreamBooth: Protecting users from personalized text-to-image synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 2116–2127, 2023.

[165] Hawkins, W., Mittelstadt, B., and Russell, C. Deepfakes on Demand: The Rise of Accessible Non-Consensual Deepfake Image Generators. In Proceedings of the 2025 ACM Conference on Fairness, Accountability, and Transparency, pp. 1602–1614, 2025.

[166] United States Copyright Ofice. Copyright and artificial intelligence, Part 3: Generative AI training, 2025. URL https://www.copyright.gov/ai/.

[167] Wu, X., Li, X., and Ni, J. Robustness of Watermarking on Text-to-Image Difusion Models, arXiv preprint arXiv:2408.02035, 2024.

[168] Zhao, X., Zhang, K., Su, Z., Vasan, S., Grishchenko, I., Kruegel, C., Vigna, G., Wang, Y.-X., and Li, L. Invisible Image Watermarks Are Provably Removable Using Generative AI. In Advances in Neural Information Processing Systems (NeurIPS), volume 37, pp. 8643–8672, 2024.

[169] Müller, A., Lukovnikov, D., Thietke, J., Fischer, A., and Quiring, E. Black-Box Forgery Attacks on Semantic Watermarks for Difusion Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 20937–20946, 2025.

[170] Liu, B., Yang, F., Bi, X., Xiao, B., Li, W., and Gao, X. Detecting generated images by real images. In European Conference on Computer Vision (ECCV), pp. 95–110, 2022.

[171] Liu, H., Tan, Z., Tan, C., Wei, Y., Wang, J., and Zhao, Y. Forgery-aware adaptive transformer for generalizable synthetic image detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10770–10780, 2024.

[172] Zhou, Z., Luo, Y., Wu, Y., Sun, K., Ji, J., Yan, K., Ding, S., Sun, X., Wu, Y., and Ji, R. AIGI-Holmes: Towards explainable and generalizable AI-generated image detection via multimodal large language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 18746–18758, 2025.

[173] Boychev, D. and Cholakov, R. ImagiNet: A multi-content benchmark for synthetic image detection, arXiv preprint arXiv:2407.20020, 2024.

[174] Hong, Y., Feng, J., Chen, H., Lan, J., Zhu, H., Wang, W., and Zhang, J. WildFake: A large-scale and hierarchical dataset for AI-generated images detection. Proceedings of the AAAI Conference on Artificial Intelligence, 39(4):3500–3508, 2025.

[175] Jia, Z., Yuan, Z., Duan, X., Zhang, J., Zhou, J., and Jain, A. K. CoDA: Color distribution probing for eficient and generalizable AI-generated image detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, pp. 1–17, 2026.

[176] Gushchin, A., Abud, K., Shumitskaya, E., Filippov, A., Bychkov, G., Lavrushkin, S., Erofeev, M., Antsiferova, A., Chen, C., Tan, S., Timofte, R., Vatolin, D., et al. NTIRE 2026 Challenge on Robust AI-Generated Image Detection in the Wild, arXiv preprint arXiv:2604.11487, 2026.

[177] Nguyen, T., Crisman, J., Hostetler, J., Cassani, L., Davinroy, M., Bautista, P., and Stamm, M. The SAFE Image Authenticity Challenge: Detecting and Localizing Partial and Fully Synthetic Manipulations. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision Workshops, pp. 933– 945, 2026.

[178] Wang, Y., Wang, S., Zhang, W., and Ouyang, Y. A Multi-Domain Benchmark for Detecting AI-Generated Text-Rich Images from GPT-Image-2, arXiv preprint arXiv:2606.19259, 2026.

[179] Yang, Y., Li, F., Kong, S., Diao, Y., Gao, X., Shi, Z., and Wang, M. Layer Consistency Matters: Elegant Latent Transition Discrepancy for Generalizable Synthetic Image Detection. In Proceedings of the IEEE/CVF

Conference on Computer Vision and Pattern Recognition (CVPR), pp. 38111–38121, June 2026.

[180] Wu, H., Li, K., Li, Y., and Zhou, J. Editprint: General Digital Image Forensics via Editing Fingerprint with Self-Augmentation Training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 35483–35493, June 2026.

[181] Guillaro, F., De Rosa, V., Cozzolino, D., and Verdoliva, L. Quality-Aware Calibration for AI-Generated Image Detection in the Wild. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pp. 10698–10707, 2026.

[182] Park, J. and Owens, A. Community Forensics: Using thousands of generators to train fake image detectors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8245–8257, 2025.

[183] Ren, S., Zhou, Y., Shen, X., Zewde, K., Duong, T., Huang, G., et al. How well are open sourced AIgenerated image detection models out-of-the-box: A comprehensive benchmark study, arXiv preprint arXiv:2602.07814, 2026.

[184] Tan, C., Zhao, Y., Wei, S., Gu, G., and Wei, Y. Learning on gradients: Generalized artifacts representation for GAN-generated images detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 12105–12114, 2023.

[185] Zhong, N., Xu, Y., Li, S., Qian, Z., and Zhang, X. PatchCraft: Exploring texture patch for eficient AIgenerated image detection, arXiv preprint arXiv:2311.12397, 2023.

[186] Tan, C., Zhao, Y., Wei, S., Gu, G., Liu, P., and Wei, Y. Frequency-aware deepfake detection: Improving generalizability through frequency space domain learning. Proceedings ofthe AAAI Conference on Artificial Intelligence, 38(5):5052–5060, 2024.

[187] Li, O., Cai, J., Hao, Y., Jiang, X., Hu, Y., and Feng, F. Improving synthetic image detection towards generalization: An image transformation perspective. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 2405–2414, 2025.

[188] Liu, Z., Qi, X., and Torr, P. H. S. Global texture enhancement for fake face detection in the wild. In Proceedings ofthe IEEE/CVFConference on Computer Vision and Pattern Recognition (CVPR), pp. 8057– 8066, 2020.

[189] Ju, Y., Jia, S., Ke, L., Xue, H., Nagano, K., and Lyu, S. Fusing global and local features for generalized AI-synthesized image detection. In IEEE International Conference on Image Processing (ICIP), pp. 3465– 3469, 2022.

[190] Zhang, Y. and Xu, X. Difusion noise feature: Accurate and fast generated image detection. In Frontiers in Artificial Intelligence and Applications, 2025.

[191] Chen, J., Yao, J., and Niu, L. A single simple patch is all you need for AI-generated image detection, arXiv preprint arXiv:2402.01123, 2024.

[192] Tan, C., Tao, R., Liu, H., Gu, G., Wu, B., Zhao, Y., and Wei, Y. C2P-CLIP: Injecting category common prompt in CLIP to enhance generalization in deepfake detection. Proceedings of the AAAI Conference on Artificial Intelligence, 39(7):7184–7192, 2025.

[193] Güera, D. and Delp, E. J. Deepfake video detection using recurrent neural networks. In IEEE International Conference on Advanced Video and Signal Based Surveillance, pp. 1–6. IEEE, 2018.

[194] Dang, H., Liu, F., Stehouwer, J., Liu, X., and Jain, A. K. On the detection of digital face manipulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5780–5789, 2020.

[195] Haliassos, A., Mira, R., Petridis, S., and Pantic, M. Leveraging real talking faces via self-supervision for robust forgery detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 14930–14942, 2022.

[196] Bohacek, M. and Farid, H. Lost in Translation: Lip-Sync Deepfake Detection from Audio-Video Mismatch. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pp. 4315–4323, June 2024.

[197] Kundu, R., Mohanty, V., Xiong, H., Jia, S., Balachandran, A., and Roy-Chowdhury, A. K. SAGA: Source Attribution of Generative AI Videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 21273–21283, June 2026.

[198] Farid, H. Mitigating the harms of manipulated media: Confronting deepfakes and digital deception. PNAS Nexus, 4(7):pgaf194, 2025.

[199] Lu, S., Zhou, Z., Lu, J., Zhu, Y., and Kong, A. W.-K. Robust Watermarking Using Generative Priors Against Image Editing: From Benchmarking to Advances. In International Conference on Learning Representations (ICLR), 2025.

[200] Zhao, Y., Pang, T., Du, C., Yang, X., Cheung, N.-M., and Lin, M. A Recipe for Watermarking Difusion Models, arXiv preprint arXiv:2303.10137, 2023.

[201] Zhang, L., Liu, X., Martin, A. V., Bearfield, C. X., Brun, Y., and Guan, H. Attack-Resilient Image Watermarking Using Stable Difusion. In Advances in Neural Information Processing Systems (NeurIPS), volume 37, pp. 38480–38507, 2024.

[202] Luo, P., Jia, Z., Zhong, Y., Zhang, J., and Zhou, J. GROW: Watermark Generation with Progressive

Guidance for Difusion Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 35978–35987, June 2026.

[203] Liu, Q., Zhang, Y., Ba, Z., Shuai, C., Cheng, P., Zheng, T., and Wang, Z. Attack-Resistant Watermarking for AIGC Image Forensics via Difusion-Based Semantic Deflection. In International Conference on Learning Representations (ICLR), 2026.

[204] Lukovnikov, D., Müller, A., Quiring, E., and Fischer, A. ClusterMark: Towards Robust Watermarking for Autoregressive Image Generators with Visual Token Clustering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9213–9222, June 2026.

[205] Chang, X., Yang, Z., Zhuo, C., and Li, Y. MaxMark: High-Capacity Difusion-Native Watermarking via Robust and Invertible Latent Embedding. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9394–9403, June 2026.

[206] Zhu, Y., Wang, Y., and Gao, X.-S. Towards Robust Content Watermarking Against Removal and Forgery Attacks. In Proceedings ofthe IEEE/CVFConference on Computer Vision and Pattern Recognition (CVPR) Findings, pp. 8059–8069, 2026.

[207] Müller, A., Lukovnikov, D., Kodama, S., Pham, M., Jain, A., Petit, J., Cohen, N., and Fischer, A. On the Robustness of Watermarking for Autoregressive Image Generation, arXiv preprint arXiv:2604.11720, 2026.

[208] Ma, R., Guo, M., Hou, Y., Yang, F., Li, Y., Jia, H., and Xie, X. Towards Blind Watermarking: Combining Invertible and Non-Invertible Mechanisms. In Proceedings of the ACM International Conference on Multimedia (ACM MM), pp. 1532–1542, 2022.

[209] Fang, H., Jia, Z., Ma, Z., Chang, E.-C., and Zhang, W. PIMoG: An Efective Screen-Shooting Noise-Layer Simulation for Deep-Learning-Based Watermarking Network. In Proceedings of the ACM International Conference on Multimedia (ACM MM), pp. 2267–2275, 2022.

[210] Wu, X., Liao, X., and Ou, B. SepMark: Deep Separable Watermarking for Unified Source Tracing and Deepfake Detection. In Proceedings of the ACM International Conference on Multimedia (ACM MM), pp. 1190–1201, 2023.

[211] Rezaei, A., Akbari, M., Ranjbar Alvar, S., Fatemi, A., and Zhang, Y. LaWa: Using Latent Space for In-Generation Image Watermarking. In European Conference on Computer Vision (ECCV), pp. 118–136, 2024.

[212] Dixit, S., Aziz, A., Bajpai, S., Sharma, V., Chadha, A., Jain, V., and Das, A. PECCVAI: Overcoming the Brittleness of AI Image Watermarking Under Visual Paraphrasing Attacks. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 24471–24480, June 2026.

[213] Tong, Y., Pan, Z., Yang, S., and Zhou, K. Training-Free Watermarking for Autoregressive Image Generation, arXiv preprint arXiv:2505.14673, 2025.

[214] Jovanović, N., Labiad, I., Souček, T., Vechev, M., and Fernandez, P. Watermarking Autoregressive Image Generation, arXiv preprint arXiv:2506.16349, 2025.

[215] Guo, M., Hu, Y., Jiang, Z., Li, Z., Sadovnik, A., Daw, A., and Gong, N. Z. AI-generated Image Detection: Passive or Watermark? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pp. 400–410, 2026.

[216] Wang, H., Cheng, R., Han, C., and Gui, J. Attribution as Retrieval: Model-Agnostic AI-Generated Image Attribution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 14062–14072, June 2026.

[217] Ren, X., Yu, X., Du, L., Chen, M., Shu, Y., Su, Z., Gao, Y., and Zhang, Z. DWBench: Holistic Evaluation of Watermark for Dataset Copyright Auditing, arXiv preprint arXiv:2602.13541, 2026.

[218] Bohacek, M. and Farid, H. GenAI Confessions: Black-box Membership Inference for Generative Image Models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops, pp. 321–330, October 2025.

[219] Liu, Y., Fan, C., Dai, Y., Chen, X., Zhou, P., and Sun, L. MetaCloak: Preventing Unauthorized Subject-Driven Text-to-Image Difusion-Based Synthesis via Meta-Learning. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 24219–24228, 2024.

[220] Wang, F., Tan, Z., Wei, T., Wu, Y., and Huang, Q. SimAC: A Simple Anti-Customization Method for Protecting Face Privacy against Text-to-Image Synthesis of Difusion Models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 12047–12056, 2024.

[221] Guo, H., Nie, S., Du, C., Pang, T., Sun, H., and Li, C. Real-Time Identity Defenses against Malicious Personalization of Difusion Models, arXiv preprint arXiv:2412.09844, 2024.

[222] Xiong, L., Li, J., Li, Z., Jiang, W., and Fu, Z. No Way To Steal My Face: Proactive Defense Against Identity-Preserving Personalized Generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 20680–20690, June 2026.

[223] Lee, E., Back, S.-h., Kim, H.-I., and Yoo, S. B. DeepProtect: Proactive Face-Swapping Defense using Identity Blending and Attribute Distortion. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6569–6579, June 2026.

[224] Tang, Q., Krinsky, J., and Bharati, A. StyleProtect: Safeguarding Artistic Identity in Finetuned Difusion Models. In Proceedings ofthe IEEE/CVFConference on Computer Vision and Pattern Recognition (CVPR)

Workshops, pp. 10759–10769, 2026.

[225] Shao, M., Meng, L., Lv, X., Wu, M., Chen, X., Zhang, Q., Liu, C., Qiao, Y., and Dong, C. UniDef: Universal Defense Against Unauthorized Image Manipulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8631–8640, June 2026.

[226] Le, V. T. and Fu, Y. VisiLock: Authorizing Instruction-based Image Editing with Dual Score Distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 15710–15718, June 2026.

[227] Qi, L., Li, Y., Liang, S., Tu, Z., and Tao, D. Cert-LAS: Toward Certified Model Ownership Verification for Text-to-Image Difusion Models via Layer-Adaptive Smoothing, arXiv preprint arXiv:2605.29809, 2026.

[228] Sablayrolles, A., Douze, M., Schmid, C., and Jégou, H. Radioactive Data: Tracing Through Training. In Proceedings ofthe 37th International Conference on Machine Learning (ICML), volume 119 of Proceedings of Machine Learning Research, pp. 8326–8335. PMLR, 2020.

[229] Li, Y., Zhu, M., Yang, X., Jiang, Y., Wei, T., and Xia, S.-T. Black-Box Dataset Ownership Verification via Backdoor Watermarking. IEEE Transactions on Information Forensics and Security, 18:2318–2332, 2023.

[230] Tang, R., Feng, Q., Liu, N., Yang, F., and Hu, X. Did You Train on My Dataset? Towards Public Dataset Protection with Clean-Label Backdoor Watermarking. ACM SIGKDD Explorations Newsletter, 25(1):43–53, 2023.

[231] Huang, Z., Gong, N. Z., and Reiter, M. K. A General Framework for Data-Use Auditing of ML Models. In Proceedings of the 2024 ACM SIGSAC Conference on Computer and Communications Security, pp. 1300–1314. ACM, 2024.

[232] Tan, M., Wang, T., and Jha, S. A Somewhat Robust Image Watermark against Difusion-Based Editing Models, arXiv preprint arXiv:2311.13713, 2023.

[233] Ma, Y., Zhao, Z., He, X., Li, Z., Backes, M., and Zhang, Y. Generative Watermarking against Unauthorized Subject-Driven Image Synthesis, arXiv preprint arXiv:2306.07754, 2023.

[234] Cui, Y., Ren, J., Xu, H., He, P., Liu, H., Sun, L., Xing, Y., and Tang, J. DifusionShield: A Watermark for Copyright Protection against Generative Difusion Models. ACM SIGKDD Explorations Newsletter, 26(2): 60–75, 2025.

[235] Wang, Z., Chen, C., Lyu, L., Metaxas, D. N., and Ma, S. Diagnosis: Detecting Unauthorized Data Usages in Text-to-Image Difusion Models. In International Conference on Learning Representations (ICLR), 2024.

[236] Ren, J., Cui, Y., Chen, C., Xing, Y., Liu, H., and Lyu, L. EnTruth: Enhancing the Traceability of Unauthorized Dataset Usage in Text-to-Image Difusion Models with Minimal and Robust Alterations, arXiv preprint arXiv:2406.13933, 2024.

[237] Wang, S., Zhu, Y., Tong, W., and Zhong, S. Detecting Dataset Abuse in Fine-Tuning Stable Difusion Models for Text-to-Image Synthesis, arXiv preprint arXiv:2409.18897, 2024.

[238] Li, B., Wei, Y., Fu, Y., Wang, Z., Li, Y., Zhang, J., Wang, R., and Zhang, T. Towards Reliable Verification of Unauthorized Data Usage in Personalized Text-to-Image Difusion Models. In IEEE Symposium on Security and Privacy, pp. 2564–2582, 2025.

[239] Cui, Y., Ren, J., Lin, Y., Xu, H., He, P., Xing, Y., Lyu, L., Fan, W., Liu, H., and Tang, J. FT-Shield: A Watermark against Unauthorized Fine-Tuning in Text-to-Image Difusion Models. ACM SIGKDD Explorations Newsletter, 26(2):76–88, 2025.

[240] An, H., Ye, X., Hua, G., Tao, Y., Cao, H., Yu, X., and Fang, Y. RecoverMark: Robust Watermarking for Localization and Recovery of Manipulated Faces. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8587–8597, June 2026.

[241] Shao, S., Li, Y., Zheng, M., Hu, Z., Chen, Y., Li, B., He, Y., Guo, J., Tao, D., and Qin, Z. DATABench: Evaluating Dataset Auditing in Deep Learning from an Adversarial Perspective, arXiv preprint arXiv:2507.05622, 2025.

[242] Nemecek, A., He, H., Cheng, G., and Ayday, E. Authenticated Contradictions from Desynchronized Provenance and Watermarking. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, June 2026.

[243] Gamage, D., Sewwandi, D., Zhang, M., and Bandara, A. K. Labeling Synthetic Content: User Perceptions of Warning Label Designs for AI-Generated Content on Social Media. In Proceedings ofthe CHI Conference on Human Factors in Computing Systems (CHI), pp. 1–29, 2025.

[244] Feng, K. J. K., Ritchie, N., Blumenthal, P., Parsons, A., and Zhang, A. X. Examining the Impact of Provenance-Enabled Media on Trust and Accuracy Perceptions. Proceedings of the ACM on Human-Computer Interaction, 7(CSCW2):1–42, 2023.

[245] Trattner, C., Forstner, S. L., Starke, A. D., and Knudsen, E. C2PA Provenance Labels Increase Trust in Digital News Platforms Across Western Countries. Proceedings of the International AAAI Conference on Web and Social Media, 20(1):2267–2279, 2026.

[246] San Roman, R., Fernandez, P., Elsahar, H., Défossez, A., Furon, T., and Tran, T. Proactive Detection of Voice Cloning with Localized Watermarking. In Proceedings of the 41st International Conference on Machine Learning (ICML), volume 235 of Proceedings of Machine Learning Research, pp. 43180–43196. PMLR, 2024.
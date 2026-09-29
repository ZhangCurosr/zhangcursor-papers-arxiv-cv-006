# WorldWeave: Growing Persistent Geometric Worlds for Video Generation

Yifan Huang<sup>1</sup> Lifan Jiang<sup>1</sup> Qingyue Hao<sup>1</sup> Cheng Chen<sup>1</sup>

Boxi Wu<sup>2</sup> Xiaoxue Ren<sup>1</sup> Xiaofei He<sup>1</sup> Dehai Zhao<sup>1,†</sup>

<sup>1</sup>Zhejiang University <sup>2</sup>Daerwen AI

<sup>†</sup>Corresponding author.

## WorldWeave

Scene request: “Extend the town and connect its road.”

Camera request: “Follow the road and look across the river.”

![](images/c46d1941fe607363a4570bfe0e429522344c236b8933cd08a8b7bd977b523683.jpg)  
Figure 1: Overview of WorldWeave. Terrain completion extends metric terrain, agent planning organizes and validates coarse-to-fine world layouts, and rendering queries the persistent world state to guide video generation. Previously generated regions are preserved while new regions are progressively added.

## ABSTRACT

Despite rapid progress, world models still lack explicit, persistent structural memory, making it difficult to preserve consistent world structure during continual scene expansion and cross-view revisits. To address this limitation, we present WORLDWEAVE, a world generation framework that decouples world-state maintenance from visual rendering. Specifically, WORLDWEAVE combines continual elevation-map generation with agent-guided scene organization and stitching to build an expandable explicit 3D world that incrementally extends structural memory while preserving existing structure. First, its terrain module uses diffusion-based image outpainting to generate continuous metric elevation maps under neighborhood conditioning and boundary constraints. Next, an agent integrates user intent, terrain evidence, and cross-region connectivity constraints to construct scenes through hierarchical semantic planning, deterministic geometry compilation, and local revision. Finally, during visual generation, planned camera trajectories query world geometry through a readonly interface, producing depth sequences that guide video synthesis without writing the generated results back into the world state. As a result, structural memory remains independent of short-window video generation, enabling continual expansion without predefined map boundaries and providing a consistent geometric basis for observations across trajectories and repeated visits.

## 1 Introduction

Generative world models have advanced rapidly in scene synthesis, view prediction and interactive environment generation, moving from local visual content toward explorable worlds (Duan et al., 2026; Chen et al., 2026; Huang et al., 2026; Zhu et al., 2026). Continual exploration requires more than realistic, temporally coherent observations: explored regions must persist outside the current view, new regions must connect to the existing environment, and revisits must encounter stable spatial structure. This requires structural memory beyond the current generation window, preserving world geometry, object identities and spatial relations as a common reference across time and viewpoints.

Despite improved visual generation, methods that rely primarily on images, video or finite observation histories still face challenges in maintaining explicit, persistent structural memory (Xiao et al., 2025; Ren et al., 2025; Yin et al., 2026). When observations implicitly carry world information, scene continuation depends on the generator re-inferring past content; local visual coherence does not directly establish lasting structural consistency. Expansion and revisitation expose two requirements: new regions must continue terrain, roads and waterways without altering established structure, and different observation trajectories must access the same world without reconstructing its geometric relations on each visit. A persistent world must therefore support both spatial addressing and relational composition: an extension must locate adjoining terrain, object support and continuing routes in a shared coordinate system. Persistence therefore governs both construction and observation.

We propose WORLDWEAVE, a world generation framework that decouples world-state maintenance from visual rendering. An explicit 3D world in a shared metric coordinate system serves as structural memory, separating incremental construction from observation generation. The state stores terrain, regional organization, asset identities and transforms, and boundary interfaces inherited by later regions. Each expansion constructs an independent candidate from the committed state; only validated additions create a new version, leaving existing structural records unchanged. Visual generation reads a specified version, allowing different camera trajectories to reuse the same geometry. World structure thus persists independently of the current video window, while construction can continue without predefined map boundaries.

WorldWeave first uses diffusion-based image outpainting to generate complete metric terrain chunks from committed neighbors. A multiscale residual codec referenced to neighboring terrain maps elevation to an image representation; height and first-derivative constraints confined to the new chunk join it continuously to the existing surface. An agent then integrates user intent, terrain evidence and inherited interfaces to plan region roles, functional districts and asset relations. Deterministic solvers and geometry compilers realize these plans as spatial layouts and infrastructure connections. Program checks and matched-view geometry previews guide bounded local revision, after which accepted candidates are atomically committed as new world versions. Planned camera trajectories query this state for depthconditioned video synthesis, with no RGB write-back.

This separation also defines what must remain consistent across observations: the scene geometry and relations belong to the world, while appearance is synthesized for each viewing request. New construction can therefore inherit off-screen structure without recovering it from previous RGB frames. The same distinction supports controlled evaluation: camera motion should reveal new content while retaining the organization of previously observed regions. We therefore assess structural consistency and revisit memory alongside camera compliance and visual quality, connecting preservation of scene organization with the ability to explore it. Figure 1 illustrates how scene requests and camera requests operate on this shared spatial reference.

Our contributions are threefold:

• Persistent structural-memory framework. We decouple world-state maintenance from visual rendering, supporting incremental construction that preserves existing structure and shared geometric queries for observations across trajectories and revisits.

• Continual metric terrain expansion. We combine diffusion-based image outpainting, multiscale elevationresidual encoding, and new-side boundary constraints to generate complete metric terrain chunks and append them continuously.

• Agent-guided scene construction. We combine hierarchical semantic planning with deterministic geometry compilation, bounded local revision and validated commit to extend scenes following user intent, terrain conditions and inherited cross-region connections.

## 2 Related Work

Video world models and structural memory. Echo-WM combines metric camera control with geometry condition ing (Duan et al., 2026); Zing-0.5 uses text, keyboard control and causal caching (Chen et al., 2026). SolarWM-5B uses camera-conditioned causal training (Huang et al., 2026), while SANA-WM uses hybrid linear attention for minute-long generation (Zhu et al., 2026).

WorldMem retrieves frames by their states (Xiao et al., 2025); ViewCrafter, WVD and Geometry-as-context condition view synthesis on geometry (Yu et al., 2025b; Zhang et al., 2025; Hu et al., 2026). GEN3C unprojects previous observations into a 3D cache and renders it to guide generation (Ren et al., 2025), while WorldStereo and Lyra 2.0 connect views through geometric memory or correspondences (Zhang et al., 2026b; Shen et al., 2026). Alaya-EVOKE Turbo estimates geometry from generated frames and updates a camera-indexed world-state bank (Yin et al., 2026); AlayaWorld combines a 3D cache with compressed frame history (AlayaWorld Team et al., 2026). WorldWeave instead maintains memory through validated scene construction, with video generation reading committed geometry.

Persistent and expandable 3D worlds. Persistent Nature uses an extendable layout grid and camera-independent decoder (Chai et al., 2023). Text2Room, WonderJourney and WonderWorld grow scenes through view synthesis and geometric integration (Höllein et al., 2023; Yu et al., 2024, 2025a); ScenePainter tracks concept relations to limit semantic drift (Xia et al., 2025), and One2Scene establishes a shared Gaussian scaffold before generating novel views (Wang et al., 2026b). InfiniCube follows a construction-first approach, expanding semantic voxel worlds from HD maps and vehicle boxes and rendering guidance buffers for driving videos (Lu et al., 2025). XCube and LT3SD use sparse voxels and latent trees (Ren et al., 2024; Meng et al., 2025); BlockFusion, SceneFactor and NuiScene extend spatial blocks (Wu et al., 2024; Bokhovkin et al., 2025; Lee et al., 2025). We preserve committed structure and inherit terrain and scene interfaces.

Agent-guided incremental scene construction. Infinigen supplies procedural environments (Raistrick et al., 2023, 2024); 3D-GPT, SceneCraft and SceneX expose construction through language-driven programs (Sun et al., 2025; Hu et al., 2024; Zhou et al., 2025). Holodeck supports language- and vision-guided layouts (Yang et al., 2024; Bian et al., 2025); Scenethesis combines visual guidance with geometric constraints, while scene graphs and hierarchical motifs organize relations (Ling et al., 2026; Liu et al., 2025; Pun et al., 2026). WorldClaw builds region-aware terrain and places editable assets with rendering feedback (Guo et al., 2026). WorldGen imposes navigational structure to obtain traversable worlds (Wang et al., 2026a); Code2Worlds separates object generation from environmental orchestration and refines scene programs through feedback (Zhang et al., 2026a). Our planning inherits neighboring interfaces and uses deterministic compilation and bounded revision before committing additions.

Terrain generation and boundary continuity. Hydrology-based modeling organizes terrain around drainage (Génevaux et al., 2013). Learned approaches include sketch-conditioned diffusion with upscaling (Lochner et al., 2023), joint height–texture synthesis with separate latent encoders (Higo et al., 2025) and text-conditioned generation trained on geospatial data (Borne-Pons et al., 2025). InfiniteDiffusion supports seed-consistent random access through overlapping queries (Goslin, 2026). Our fixed-order expansion instead generates metric chunks against committed neighbors and joins them through new-side height and derivative constraints.

## 3 Method

## 3.1 Persistent Structural Memory

WorldWeave starts from a seed terrain chunk and maintains an explicit world in a shared metric coordinate system. This structural memory has committed version $M _ { k } = ( \mathcal T _ { k } , \mathcal P _ { k } , \mathcal T _ { k } , \bar { \mathcal T _ { k } } )$ after k accepted additions: $\mathcal { T } _ { k }$ stores elevation and boundary derivatives, $\mathcal { P } _ { k }$ stores regional organization and object relations, $\mathcal { T } _ { k }$ stores asset mesh references, identities and world transforms, and $\mathcal { T } _ { k }$ stores exposed boundary interfaces. These interfaces specify terrain attachment and the positions and types of road or water connections inherited by new regions.

An expansion request specifies intent u and frontier chunk q. Construction reads $M _ { k }$ and forms an independent candidate $\Delta M _ { q }$ through terrain expansion and hierarchical scene compilation (Figure 2), using the committed terrain and inherited interfaces. Validation and commit (Section 3.4) determine whether the candidate becomes a new version. A camera query selects an existing committed version; changing the observation does not change its structural records. The state links semantic and geometric descriptions through stable instance identities. Regional plans specify which parts of the terrain serve each function, while instance transforms place the corresponding assets in the same metric frame. Exposed interfaces transfer this organization across chunk boundaries: a new region receives both the neighboring surface and the road or water connections it must extend. This gives local construction a consistent global reference as the map grows.

![](images/e2883add8d49b1e2128be7081791d321614becbfdc69adcedd6ce25bf0e46b91.jpg)  
Figure 2: WorldWeave pipeline. Neighbor-conditioned terrain generation uses residual HDR encoding and new-side joining. The agent plans regions, districts and asset relations, while deterministic tools compile geometry and provide structural and visual evidence for revision. Validated additions extend persistent world state; camera trajectories query its geometry to condition video generation without writing RGB back into the world.

## 3.2 Continual Metric Terrain Expansion

A candidate terrain chunk contains 512 × 512 elevation samples at 0.5 m spacing, giving a nominal 256 × 256 m footprint. Its context comprises one to three committed axial neighbors: one side, two opposite sides, two adjacent sides, or three sides. We generate a complete target in one image-outpainting call and repeat this operation in a fixed expansion order, using previously committed outputs as subsequent context.

To encode metric height within the image model’s dynamic range, we fit a reference surface $b _ { q } ( x , y )$ from valid neighboring boundary bands. The unknown target does not enter this fit. For elevation $h _ { q } ,$ the residual $r _ { q }$ is encoded through three monotone channels:

$$
r _ { q } = h _ { q } - b _ { q } , \qquad c _ { q , j } = { \textstyle { \frac { 1 } { 2 } } } + { \textstyle { \frac { 1 } { 2 } } } \operatorname { t a n h } ( r _ { q } / s _ { j } ) , \qquad ( s _ { 1 } , s _ { 2 } , s _ { 3 } ) = ( 8 , 3 2 , 1 2 8 ) \mathrm { m } .\tag{1}
$$

Each encoded sample is repeated in a $2 \times 2$ image block before VAE encoding, adding redundancy without changing the metric sampling grid. The reference surface carries the elevation level and broad trend supplied by the context, leaving the residual channels to describe local relief. Small channel scales allocate greater sensitivity to subtle variations, whereas larger scales retain information over stronger relief. The same reference is restored after decoding, so neighboring chunks remain expressed in world units even though generation takes place in image space.

We adapt Qwen-Image-Edit-2509 (Wu et al., 2025) with rank-32 LoRA (Hu et al., 2021), freezing its VAE and text conditioner. Encoded neighbors and a fixed completion prompt condition generation. We arrange the context on a spatial canvas, retaining neighbor directions around a neutral target slot. This exposes opposite or adjacent boundary conditions in one layout, allowing the target to be completed jointly against all attached sides. Let $\mathcal { V } _ { q }$ index valid target loss locations, $\ell _ { q } ( x )$ denote the local diffusion loss, and $d _ { q } ( x )$ the metric distance to the nearest attached edge. We use

$$
\begin{array} { r l r } {  { w _ { q } ( x ) = 1 + \frac { 3 } { 2 } [ 1 + \cos ( \pi \operatorname* { m i n } \{ d _ { q } ( x ) / a , 1 \} ) ] , } } \\ & { } & { \mathscr { L } _ { q } = \frac { \sum _ { x \in \mathcal { V } _ { q } } w _ { q } ( x ) \ell _ { q } ( x ) } { \sum _ { x \in \mathcal { V } _ { q } } w _ { q } ( x ) } , \qquad a = 1 6 \mathrm { m } . } \end{array}\tag{2}
$$

Weights are evaluated on the loss grid. The nearest-edge distance implements the maximum over intersecting edge bands; normalization keeps loss scales comparable across targets.

After VAE decoding, each channel gives a residual estimate $\hat { r } _ { q , j } = s _ { j }$ atanh $\left( 2 \hat { c } _ { q , j } - 1 \right)$ , with channel values clipped inside (0, 1). We combine the repeated estimates using inverse-sensitivity weighting, downweighting near-saturated channels, and refine the result with five robust projection iterations onto the three-channel encoding curve. Adding $b _ { q }$

recovers the metric prediction $\hat { h } _ { q } = b _ { q } + \hat { r } _ { q }$ on the original sampling grid. For each attached side $e \in { \mathcal { E } } _ { q } ,$ the neighbor supplies height $g _ { e }$ and directional derivative $v _ { e }$ on the world-coordinate interface $\Gamma _ { e }$ , using a shared normal $n _ { e } .$ The joined surface satisfies

$$
\begin{array} { r } { h _ { q } ^ { + } = \hat { h } _ { q } + \delta _ { q } , \qquad \mathrm { s u p p } ( \delta _ { q } ) \subseteq \mathcal { B } _ { q } , \qquad } \\ { h _ { q } ^ { + } | _ { \Gamma _ { e } } = g _ { e } , \qquad \partial _ { n _ { e } } h _ { q } ^ { + } | _ { \Gamma _ { e } } = v _ { e } , \qquad e \in \mathcal { E } _ { q } . } \end{array}\tag{3}
$$

Here $B _ { q }$ includes the new-side height-transition band and connector cells between existing sample-center grids. New connector cells are bicubic Hermite patches whose old-facing edges inherit boundary heights and derivatives (Figure 3); slope correction uses an independent taper. Corrections vanish at their inner support boundaries, with $\delta _ { q } = \partial _ { n } \delta _ { q } = 0$ Committed geometry is preserved, and modification magnitude is reported separately from continuity (Appendix D). Connector cells bridge the gap between the old and new sample-center grids, providing a shared boundary for subsequent terrain and scene construction. Height correction adjusts the offset between chunks, whereas slope correction control the approach to the inherited boundary slope. Separate support widths let the surface satisfy both conditions while confining derivative-induced displacement near the seam.

## 3.3 Hierarchical Scene Planning and Compilation

Deterministic terrain analysis extracts slope, local relief, drainage and buildable-area evidence from the accepted elevation field. The agent combines this evidence with $u ,$ inherited interfaces $\bar { \mathcal { T } } _ { k } .$ and available asset capabilities to assign broad region roles, such as settlement, woodland or river corridor. A region solver converts these semantic assignments into spatial extents. After region review (Section 3.4), the agent specifies district roles, adjacency and access requirements. A deterministic solver resolves their extents and infrastructure under inherited road and water interfaces, producing functional or ecological subdivisions with the access, clearance and terrain conditions required by downstream assets.

![](images/70160c6f9a0d622fd121b3b99e1504c5b37d317bc1dbabf95327fb679cf233d6.jpg)

Within each district, the agent turns its functional requirements into an asset-relation plan. The plan assigns primary, companion and service roles, specifies shared spaces and orientation, and connects buildings and vegetation groups to the district’s roads and ecological zones. The typed catalog supplies each asset’s dimensions, support anchors, entrance direction and required context.

Figure 3: New-side terrain joining. Old samples remain fixed; newly owned connector cells inherit boundary height and normal derivative. The correction vanishes at the interior support boundary. Schematic.

These attributes connect semantic choices to the spatial constraints inherited from regional and district planning. The hierarchy passes explicit commitments between scales: region boundaries delimit admissible land use, district connections reserve routes through those regions, and asset relations specify how individual instances participate in that organization. The deterministic compiler resolves the plan into stable instance identities and metric transforms under the spatial and asset contracts in Appendix E. It fits assets to supporting surfaces, realizes infrastructure connections and vegetation groupings, and preserves access corridors and collision clearance. For $n _ { q }$ candidate instances with required geometric relations $\mathcal { R } _ { q } .$ , acceptance requires

$$
\begin{array} { r } { \mathcal { T } _ { q } = \{ ( a _ { i } , T _ { i } , \rho _ { i } , d _ { i } ) \} _ { i = 1 } ^ { n _ { q } } , \qquad g _ { r } ( \mathcal { T } _ { q } , h _ { q } ^ { + } ; M _ { k } ) = 1 \quad ( r \in \mathcal { R } _ { q } ) . } \end{array}\tag{4}
$$

Here $a _ { i }$ identifies catalog geometry, $T _ { i }$ its world transform, $\rho _ { i }$ its semantic role and $d _ { i }$ its district. Each predicate $g _ { r }$ checks a catalog-supported relation against the candidate and committed context. Failed predicates retain instance and relation identifiers, locating the plan entries to revise. The plan, geometry and checks form the scene portion of $\Delta M _ { q }$

## 3.4 Bounded Revision and Validated Commit

Region review checks inherited connections, coverage and terrain feasibility before district refinement. After compilation, program checks evaluate support, collision and connectivity in parallel with matched-view GPU geometry and depth previews. The tests establish geometric validity; matched viewpoints show whether the compiled arrangement realizes the intended regional organization. The agent receives both for the same candidate, allowing a revision to address the responsible functional area or obstructed connection while retaining unaffected content. Accepted regions then expose their outer interfaces as context for the next expansion, linking each local planning cycle to continued world construction. Review evidence is bound to the candidate revision: after an edit, program reports and matched-view previews are regenerated for that revision. This keeps acceptance tied to the geometry that will be committed.

Let $\pi _ { q } ^ { ( t ) }$ be the candidate plan after t revisions and K the revision budget. A failed review returns a structured delta $\delta \pi _ { q } ^ { ( t ) }$ targeting the responsible region, district or asset relation:

$$
\pi _ { q } ^ { ( t + 1 ) } = \mathrm { E d i t } ( \pi _ { q } ^ { ( t ) } , \delta \pi _ { q } ^ { ( t ) } ) , \qquad \Delta M _ { q } ^ { ( t + 1 ) } = \mathrm { C o m p i l e } ( \pi _ { q } ^ { ( t + 1 ) } ; M _ { k } ) , \qquad t < K .\tag{5}
$$

Edit updates the indicated plan entries; Compile recomputes affected geometry with $M _ { k }$ read-only. Reviews use fixed acceptance criteria and reject candidates that exhaust the revision budget.

When all geometric checks and agent reviews pass, the candidate is atomically published as a new world version. Let $\Omega _ { k }$ denote the terrain, semantic-plan, instance and boundary-geometry records already committed in $M _ { k }$ . The update appends accepted records while preserving those fields:

$$
M _ { k + 1 } = M _ { k } \oplus \Delta M _ { q } , \qquad \left. M _ { k + 1 } \right| _ { \Omega _ { k } } = M _ { k } \big | _ { \Omega _ { k } } .\tag{6}
$$

Here $\oplus$ denotes the validated addition, including new exposed interfaces for subsequent expansion. The equality preserves the terrain, organization and instance records reused by future expansions and observations. Visibility is computed from the complete geometry of the selected version.

## 3.5 Read-only Queries for Video Generation

Observation generation selects a committed world version independently of expansion. A viewing request specifies desired motion and targets, either supplied directly by the user or sampled by the agent from the user’s intent. The camera planner converts this request into intrinsics, poses and timing $C _ { 1 : T }$ , then checks the trajectory for clearance and the required viewpoints. GPU queries resolve nearest visible intersections across terrain and asset meshes, returning metric depth with its convention, validity mask and calibration. Per-pixel hit identities associate queried surfaces with persistent asset records. They bind named camera targets to world geometry when assessing visibility along a trajectory. The resulting RGB video visualizes this world version.

As the camera moves, visible surfaces are recomputed from this common geometry, providing spatially coordinated conditions throughout the trajectory. For video generator $F ,$ , let $A _ { F }$ adapt depths from query Q. Its supported optional style input $s _ { F }$ comprises text, reference images, or both:

$$
D _ { 1 : T } = Q ( M _ { k } , C _ { 1 : T } ) , \qquad V _ { 1 : T } = F \bigl ( A _ { F } ( D _ { 1 : T } ) , s _ { F } \bigr ) .\tag{7}
$$

A surface point X retains its world coordinates in the selected version. For two views in which it is visible, the homogeneous image coordinates obey

$$
\widetilde { \boldsymbol { x } } _ { t } ( \boldsymbol { X } ) \sim \boldsymbol { K } _ { t } \boldsymbol { R } _ { t } ^ { \intercal } ( \boldsymbol { X } - \boldsymbol { c } _ { t } ) , \qquad \widetilde { \boldsymbol { x } } _ { s } ( \boldsymbol { X } ) \sim \boldsymbol { K } _ { s } \boldsymbol { R } _ { s } ^ { \intercal } ( \boldsymbol { X } - \boldsymbol { c } _ { s } ) .\tag{8}
$$

Here $K _ { t } , R _ { t }$ and $c _ { t }$ denote intrinsics, camera-to-world rotation and camera center. Both projections refer to the same terrain or asset surface, supplying a shared geometric correspondence through viewpoint changes and revisits. The depth sequence encodes how camera motion changes visibility, including occlusions and the boundaries of terrain and assets.

The adapter converts metric depth to the generator’s required encoding, resolution and frame rate while preserving camera calibration. Available style references control appearance, while depth supplies the spatial structure along the requested trajectory. Generated RGB never writes back into world state. Appendix F specifies revision, publication and observation.

## 4 Experiments

## 4.1 Experimental Setup

Benchmark. We evaluate 120 tasks covering forward exploration, viewpoint changes, and occlusion or off-screen return. Terrain relief, vegetation biomes (e.g., forests and grasslands), and settlement layouts (e.g., villages and towns) vary across scenes. Seasonal appearance and lighting are varied as rendering attributes. Videos are evaluated over 15 seconds at 12 fps.

Baselines. We compare with six open-source world models: SANA-WM, Zing, SolarWM-5B, AlayaWorld, Alaya-EVOKE-Turbo and JoyAI-Echo-1.5 (WM) (Zhu et al., 2026; Chen et al., 2026; Huang et al., 2026; AlayaWorld Team et al., 2026; Yin et al., 2026; Duan et al., 2026), and four video baselines: Seedance 2.0 (closed-source), Wan3.0, Kling 3.0 and our base generator MiniMax-H3. Since world models commonly require a first frame and camera controls, we provide them with the same first frame as the WorldWeave-generated video and the camera trajectory used to produce its depth input. All models receive the same task and scene-description prompt. WorldWeave additionally receives depth queried from the constructed world.

![](images/756f3707fa9fc1096d194fc9b1aaab3a59e38fd111193fc228c9a3b418780641.jpg)  
Figure 4: World-grounded video generation across four scenes. Each row follows a viewing trajectory from left to right, showing revisitation or continued exploration. Insets show depth guidance, and numbered markers identify corresponding scene elements across views.

Metrics. We assess visual quality with IQ and AQ from VBench (Huang et al., 2024), and revisit memory and camera compliance with SMC and Cam. We jointly calibrate smoothness and geometric consistency for realized motion, yielding MN-MS, MC-GeCo (Gu et al., 2026), MC-MEt3R (Asim et al., 2025) and MC-GeoCon (Xue et al., 2026), to reduce systematic scoring biases across different camera-motion speeds. Appendix B defines these metrics and their motion calibration.

## 4.2 Main Quantitative Results

Table 1 compares visual quality, structural consistency and camera control. WorldWeave ranks fifth in IQ and second in AQ, while leading MN-MS, MC-GeCo, SMC, Cam, MC-MEt3R and MC-GeoCon. Its strongest gains therefore concern maintaining an explorable world: even with greater mean image-space motion than six comparison methods, WorldWeave retains leading structural-consistency and camera-compliance scores alongside competitive appearance and smooth motion. Figure 4 illustrates the corresponding video sequences across revisitation and exploration tasks.

The Base comparison examines the benefit of world-grounded conditioning within the same video-model family. Relative to MiniMax-H3, WorldWeave improves all eight measures. The largest gains are in SMC (+426.7%) and Cam (+42.2%), together with a 32.2% reduction in MC-GeoCon error. IQ and AQ also improve, supporting both spatial organization and appearance.

## 4.3 Structural Consistency and Camera Compliance

Geometric consistency. WorldWeave achieves the lowest MC-GeCo and MC-MEt3R errors, 0.0573 and 0.122, improving over MiniMax-H3 by 9.6% and 12.5%, respectively. The lower motion-calibrated errors indicate more coherent geometry and reprojected features across generated views. MC-GeoCon corroborates this result through geometric correspondences. These results support transferring spatial organization from shared depth guidance into generated video.

Memory across revisits. WorldWeave obtains the highest SMC score, 2.30 compared with 2.21 for the runner-up Echo-WM, while MiniMax-H3 scores 0.438. This improvement strengthens memory over disappearance and return. When a target leaves the view or becomes occluded, later observations must recover its identity and spatial relations. Reusing a committed world version provides the same target geometry and surrounding layout on return, reducing the need to reconstruct these relations from a limited visual history. Appendix G provides matched occlusion and return sequences in farmstead, town and woodland scenes.

Camera compliance. WorldWeave also ranks first in Cam at 72.3, ahead of Seedance 2.0 by 2.49 points. Relative to MiniMax-H3’s 50.8, this is a 42.2% increase in score, reflecting stronger adherence to the requested translations,

Table 1: World-video comparison. The first six methods are open-source world models; the remaining baselines are video models. Bold/underline indicate best/second-best scores; yellow highlights the three largest relative gains over MiniMax-H3 (Base), with signed percentage changes.
<table><tr><td rowspan="2">Method</td><td colspan="2">Visual quality</td><td colspan="2">Memory and camera control</td><td colspan="4">Structural consistency</td></tr><tr><td>Imaging Aesthetic quality↑ quality↑</td><td></td><td>Structural memory↑</td><td>Camera compliance↑</td><td colspan="4">MN-MS↑ MC-GeCo↓ MC-MEt3R↓ MC-GeoCon↓</td></tr><tr><td>SANA-WM</td><td>0.739</td><td>0.619</td><td>0.328</td><td>55.5</td><td>0.973</td><td>0.113</td><td>0.203</td><td>0.189</td></tr><tr><td>Zing</td><td>0.763</td><td>0.685</td><td>0.989</td><td>55.8</td><td>0.974</td><td>0.0698</td><td>0.129</td><td>0.142</td></tr><tr><td>SolarWM-5B</td><td>0.755</td><td>0.604</td><td>0.370</td><td>54.7</td><td>0.956</td><td>0.0786</td><td>0.166</td><td>0.217</td></tr><tr><td>AlayaWorld</td><td>0.784</td><td>0.641</td><td>1.51</td><td>56.4</td><td>0.896</td><td>0.926</td><td>0.239</td><td>0.295</td></tr><tr><td>EVOKE-Turbo</td><td>0.784</td><td>0.718</td><td>1.12</td><td>61.6</td><td>0.961</td><td>0.105</td><td>0.182</td><td>0.249</td></tr><tr><td>Echo-WM</td><td>0.791</td><td>0.655</td><td>2.21</td><td>63.5</td><td>0.968</td><td>0.0604</td><td>0.135</td><td>0.135</td></tr><tr><td>Seedance 2.0</td><td>0.717</td><td>0.617</td><td>2.14</td><td>69.8</td><td>0.980</td><td>0.101</td><td>0.170</td><td>0.307</td></tr><tr><td>Wan3.0</td><td>0.786</td><td>0.702</td><td>1.63</td><td>59.1</td><td>0.938</td><td>0.146</td><td>0.184</td><td>0.131</td></tr><tr><td>Kling 3.0</td><td>0.739</td><td>0.588</td><td>2.09</td><td>59.3</td><td>0.978</td><td>0.0745</td><td>0.155</td><td>0.163</td></tr><tr><td>MiniMax-H3 (Base)</td><td>0.725</td><td>0.617</td><td>0.438</td><td>50.8</td><td>0.979</td><td>0.0633</td><td>0.140</td><td>0.178</td></tr><tr><td>WorldWeave</td><td>0.781</td><td>0.704</td><td>2.30 (+426.7%) 72.3 (+42.2%)</td><td></td><td>0.981</td><td>0.0573</td><td>0.122</td><td>0.121 (-32.2%)</td></tr></table>

turns and returns. On the 120 matched tasks, its mean optical-flow amplitude exceeds those of SANA-WM, EVOKE-Turbo, AlayaWorld, Seedance 2.0, Wan3.0 and Kling 3.0 by 8.1–24.1%. Even with this greater realized motion, WorldWeave maintains leading structural-consistency scores, supporting exploration with coherent changes in viewpoint. Appendix B.2 contrasts favorable raw scores with texture degradation or limited motion.

## 4.4 Ablation Studies

![](images/6641d1214eb298aa94f7663ceb4b26da20dc44ec0123bb6b7b9d0317ecb662da.jpg)

Multi-image  
![](images/836b9eceb031823826fe2a5dce324a4f654a16fd2c04e89508f4fefda4c100df.jpg)  
Figure 5: Opposite-neighbor completion with canvas (top) and multi-image inputs (bottom). Additional cases: Appendix A.2.

We evaluate elevation encoding, terrain conditioning and joining, followed by planning-model substitution and videogeneration seed variation (Appendix A).

Metric elevation encoding. Using residual rather than absolute grayscale reduces mean VAE round-trip RMSE from 2.182 to 0.469 m (Table 2(a)). Multiscale residual channels further reduce it to 0.158 m, and the full codec reaches 0.136 m, a further 14.0% reduction. Referencing neighboring terrain removes the large elevation offset, while the channel scales retain sensitivity across different relief levels. These reductions demonstrate improved elevation-codec fidelity: context referencing and multiscale channels jointly preserve metric height through the frozen image VAE.

Spatially arranged terrain conditioning. Canvas conditioning yields lower boundary-height gaps in all four neighbor configurations (Table 2(b)), reducing them by 65.4–76.1% relative to independent multi-image references. The remaining gaps are 0.264–0.368 m for nominally 256 m chunks. Slope continuity also improves in the two- and three-neighbor settings, although the single-neighbor case favors multi-image input. This pattern supports arranging neighboring terrain in its spatial context, especially when several boundaries must be satisfied together. Figure 5 shows the opposing-neighbor case; Appendix A.2 compares all configurations. Training configurations and evaluation details are specified in Appendix A.1.

Continuous terrain joining. Table 2(c) compares four settings on 256 matched targets. Height-only joining closes the height gap but leaves slope residual 0.359; full derivative inheritance reduces both residuals below numerical tolerance. Independent slope taper preserves continuity while reducing mean whole-target modification from 0.2088 to 0.0825 m (60.5%). All joining variants connect every target and preserve valid existing heights and derivatives, supporting separate height and slope transition widths (Appendix D.4).

Table 2: Terrain ablations of (a) elevation encoding, (b) input organization and (c) terrain joining. ∆h denotes whole-target RMS modification.
<table><tr><td colspan="4">(a) Elevation encoding</td></tr><tr><td>Reference Multiscale Spatial subtraction</td><td>HDR</td><td>repetition</td><td>RMSE (m)↓</td></tr><tr><td></td><td></td><td></td><td>2.182</td></tr><tr><td></td><td>√</td><td></td><td>1.935</td></tr><tr><td>√</td><td>一</td><td></td><td>0.469</td></tr><tr><td>√</td><td>√</td><td>1</td><td>0.158</td></tr><tr><td>√</td><td>√</td><td>√</td><td>0.136</td></tr></table>

<table><tr><td colspan="5">(b) Terrain input</td></tr><tr><td rowspan="2">Input</td><td colspan="4">Height gap (m)↓ Slope gap (m/m)↓</td></tr><tr><td>Multi</td><td>Canvas</td><td>Multi</td><td>Canvas</td></tr><tr><td>One</td><td>0.763</td><td>0.264</td><td>0.240</td><td>0.278</td></tr><tr><td rowspan="3">Opposite 1.394 L-shaped 1.076</td><td></td><td>0.333</td><td>0.373</td><td>0.338</td></tr><tr><td></td><td>0.296</td><td>0.310</td><td>0.275</td></tr><tr><td>1.214</td><td>0.368</td><td>0.361</td><td>0.328</td></tr></table>

<table><tr><td>correction inheritance taper</td><td>Height Derivative Slope</td><td></td><td colspan="3">Hc ↓ (m) Sc ↓ (m/m) ∆h (m)</td></tr><tr><td>一</td><td></td><td></td><td>0.319</td><td>0.638</td><td>0</td></tr><tr><td>√</td><td></td><td></td><td>0</td><td>0.359</td><td>0.0773</td></tr><tr><td>√</td><td>√</td><td></td><td>0</td><td>0</td><td>0.2088</td></tr><tr><td>√</td><td>√</td><td>√</td><td>0</td><td>0</td><td>0.0825</td></tr></table>

Planning model. Replacing the planner changes settlement density and infrastructure while retaining the shared terrain and cross-region organization. Both planners continue inherited roads: the alternative favors woodland and small settlement groups, while the default extends a denser street network. Appendix A.3 compares full layouts with identical terrain and the same existing region.

Random-seed sensitivity. Across five fixed scene–trajectory pairs and five generation seeds, the seed-wise mean IQ, AQ and MN-MS have coefficients of variation below 0.4%; those of the three geometric metrics range from 3.1% to 6.3%. Thus, the aggregate computational measures remain comparatively stable as the noise realization changes. Event-based SMC and Cam vary more strongly. Appendix A.4 provides per-task scores, seed-wise means and dispersion, including the observed black-sky case under fixed geometry, first frame, prompt and depth.

## 4.5 User Study

![](images/2a54cf340ae4942f70bb6b95a07492e5a658d177222eccbb26daa179f384eeb8.jpg)

We conduct a blinded study of WorldWeave and six world models on 20 selected scene–trajectory cases (140 videos). Participants score camera, object, and structure/texture consistency from 1 to 5. Each video receives three or four ratings, totaling 436 evaluations. Table 3 reports video means averaged equally across cases (Appendix C). WorldWeave leads with 4.57, 4.73 and 4.52, compared with 3.92, 3.42 and 3.32 for Echo-WM. The gains over Echo-WM are 0.65, 1.31 and 1.20 points.

5 Conclusion Table 3: User ratings (1–5; higher is better).

We presented WorldWeave, a world generation framework that decouples persistent structural memory from visual synthesis. Continual metric terrain expansion and agent-guided scene compilation extend an explicit world while preserving previously committed structure. Hierarchical planning connects user intent, terrain evidence and inherited interfaces; deterministic compilation, bounded revision and validated publication turn these decisions into reusable geometry. Read-only camera queries then provide depth conditions for video generation without writing synthesized observations back into world state. Across the evaluated methods, WorldWeave leads the structural-consistency and camera-compliance measures while retaining competitive visual quality. Human ratings on the 20 selected cases also favor WorldWeave in camera, object and structure/texture consistency. These results support maintaining a shared geometric world independently of individual video-generation windows, including observations with greater realized motion than most baselines.

Limitations and future work. Our current focus is static geometry, and depth provides guidance rather than a hard rendering constraint. Future work will address dynamic interactions, long-horizon appearance memory and adaptive spatial detail for larger worlds.

## AI Use Statement

Generative AI tools assisted with drafting sections of the manuscript, organizing and polishing the writing, and literature retrieval and discovery, including identifying potentially relevant related work. The authors are responsible for checking AI-assisted text and references against the underlying sources, and for the accuracy, originality and integrity of the final manuscript.

## Ethics Statement

Our evaluation uses virtual environments and generated videos. Participants in the user study provide informed consent and evaluate anonymized method outputs. The terrain data and geometry assets are drawn from authored virtual environments, not presented as surveyed real-world geography. Third-party assets, pretrained models and services remain subject to their respective licenses and terms; redistribution requires the corresponding permissions. As with other visual generation systems, outputs could be used to create misleading depictions. Applications should clearly disclose synthetic content and preserve provenance, and should not treat generated worlds as verified representations of real locations.

## Reproducibility Statement

Section 3 defines the world representation, incremental construction and read-only observation interfaces; Section 4 specifies the evaluation setting and reports the principal comparisons. Appendices D–F detail terrain encoding and joining, scene compilation, revision and geometric queries. Appendix B defines the metrics and Appendix A.1 specifies ablation protocols; Appendices A.2 and A.4 provide matched terrain visualizations and per-task seed results. Appendix C specifies user-study selection and score aggregation.

## References

AlayaWorld Team, Kaipeng Zhang, Chuanhao Li, Yifan Zhan, Yongtao Ge, Yuanyang Yin, Jiaming Tan, Kang He, Liaoyuan Fan, Mingliang Zhai, Ruicong Liu, Xiaojie Xu, Xuangeng Chu, Zhen Li, Zhengyuan Lin, Zhixiang Wang, Zian Meng, and Zihui Gao. AlayaWorld: Interactive Long-Horizon World Modeling - Full Technical Report (v1.1), 2026. URL https://arxiv.org/abs/2608.13492. Published: arXiv preprint.

Mohammad Asim, Christopher Wewer, Thomas Wimmer, Bernt Schiele, and Jan Eric Lenssen. MEt3R: Measuring Multi-View Consistency in Generated Images. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6034–6044. IEEE, 2025. URL https://arxiv.org/abs/2501.06336.

Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. arXiv preprint arXiv:2301.08243, 2023. URL https://arxiv.org/abs/2301.08243.

Mahmoud Assran, Adrien Bardes, David Fan, et al. V-JEPA 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025. URL https://arxiv.org/abs/2506.09985.

Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video. arXiv preprint arXiv:2404.08471, 2024. URL https://arxiv.org/abs/2404.08471.

Zixuan Bian, Ruohan Ren, Yue Yang, and Chris Callison-Burch. HOLODECK 2.0: Vision-Language-Guided 3D World Generation with Editing. arXiv preprint arXiv:2508.05899, 2025. URL https://arxiv.org/abs/2508.05899.

Aleksey Bokhovkin, Quan Meng, Shubham Tulsiani, and Angela Dai. SceneFactor: Factored Latent 3D Diffusion for Controllable 3D Scene Generation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 628–639, 2025. URL https://openaccess.thecvf.com/content/CVPR2025/html/Bokh ovkin\_SceneFactor\_Factored\_Latent\_3D\_Diffusion\_for\_Controllable\_3D\_Scene\_Generation\_CV PR\_2025\_paper.html.

Paul Borne-Pons, Mikolaj Czerkawski, Rosalie Martin, and Romain Rouffet. MESA: Text-Driven Terrain Generation Using Latent Diffusion and Global Copernicus Data. arXiv preprint arXiv:2504.07210, 2025. URL https: //arxiv.org/abs/2504.07210.

Lucy Chai, Richard Tucker, Zhengqi Li, Phillip Isola, and Noah Snavely. Persistent Nature: A Generative Model of Unbounded 3D Worlds. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20863–20874. IEEE, 2023. URL https://arxiv.org/abs/2303.13515.

Mingyang Chen, Shengdong Chen, Xiaoxiao Fu, Bosheng Gong, Haoyuan Guo, Bowen Li, Jiawen Li, Kejun Li, Tianpeng Li, Yin Liu, et al. Zing-0.5: Toward Playable Worlds with Real-Time Joint Action and Text Control. arXiv preprint arXiv:2609.17909, 2026. URL https://arxiv.org/abs/2609.17909.

Nan Duan, Haoyang Huang, Weiyang Jin, Haoran Li, Yaowei Li, Yuming Li, Yijun Liu, Xin Lu, Xiaoxiao Ma, Yanwen Ma, et al. Long-Horizon Audio-Visual Generation for Persistent Stories and Interactive Worlds. arXiv preprint arXiv:2608.23383, 2026. URL https://arxiv.org/abs/2608.23383.

Jean-David Génevaux, Éric Galin, Eric Guérin, Adrien Peytavie, and Bedrich Benes. Terrain Generation Using Procedural Models Based on Hydrology. ACM Transactions on Graphics (TOG), 32(4):1–13, 2013. doi: 10.1145/24 61912.2461996. URL https://doi.org/10.1145/2461912.2461996.

Alexander Goslin. InfiniteDiffusion: Bridging Learned Fidelity and Procedural Utility for Open-World Terrain Generation. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pages 1–10, 2026. doi: 10.1145/3799902.3811080. URL https://doi.org/10.1 145/3799902.3811080.

Leslie Gu, Junhwa Hur, Charles Herrmann, Fangneng Zhan, Todd Zickler, Deqing Sun, and Hanspeter Pfister. GeCo: Evaluating Geometric Consistency for Video Generation via Motion and Structure. In The 19th European Conference on Computer Vision (ECCV), 2026. URL https://vcg.seas.harvard.edu/publications/geco.

Chunchao Guo, Jinpeng Li, Yang Li, and Zilong Huang. WorldClaw: Agentic 3D Open-World Generation at Scale. arXiv preprint arXiv:2608.05248, 2026. URL https://arxiv.org/abs/2608.05248.

Kazuki Higo, Toshiki Kanai, Yuki Endo, and Yoshihiro Kanamori. TerraFusion: Joint Generation of Terrain Geometry and Texture Using Latent Diffusion Models. Virtual Reality & Intelligent Hardware, 7(6):560–576, 2025. doi: 10.1016/j.vrih.2025.11.001. URL https://doi.org/10.1016/j.vrih.2025.11.001.

Lukas Höllein, Ang Cao, Andrew Owens, Justin Johnson, and Matthias Nießner. Text2Room: Extracting Textured 3D Meshes from 2D Text-to-Image Models. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 7875–7886. IEEE, 2023. URL https://openaccess.thecvf.com/content/ICCV2023/html/Hollei n\_Text2Room\_Extracting\_Textured\_3D\_Meshes\_from\_2D\_Text-to-Image\_Models\_ICCV\_2023\_paper. html.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-Rank Adaptation of Large Language Models. arXiv preprint arXiv:2106.09685, 2021. URL https://arxiv.org/abs/2106.09685.

JiaKui Hu, Jialun Liu, Liying Yang, Xinliang Zhang, Kaiwen Li, Shuang Zeng, Yuanwei Li, Haibin Huang, Chi Zhang, and Yanye Lu. Geometry-as-context: Modulating Explicit 3D in Scene-consistent Video Generation to Geometry Context. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4258–4268, 2026. URL https://openaccess.thecvf.com/content/CVPR2026/html/Hu\_Geometry-as-c ontext\_Modulating\_Explicit\_3D\_in\_Scene-consistent\_Video\_Generation\_to\_Geometry\_CVPR\_20 26\_paper.html.

Ziniu Hu, Ahmet Iscen, Aashi Jain, Thomas Kipf, Yisong Yue, David A Ross, Cordelia Schmid, and Alireza Fathi. SceneCraft: An LLM Agent for Synthesizing 3D Scenes as Blender Code. In ICML, pages 19252–19282, 2024. URL https://proceedings.mlr.press/v235/hu24g.html.

Junchao Huang, Guian Fang, Shengju Qian, Xianghao Kong, Zhuoran Zhao, Wei Huang, Yihua Du, Zixin Zhang, Justin Cui, Yuchao Gu, et al. SolarWM: Open Data and Scalable Training for Long-Horizon Video World Models. arXiv preprint arXiv:2609.02886, 2026. URL https://arxiv.org/abs/2609.02886.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. VBench: Comprehensive Benchmark Suite for Video Generative Models. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21807–21818. IEEE, 2024. URL https://arxiv.org/abs/2311.17982.

Han-Hung Lee, Qinghong Han, and Angel X Chang. NuiScene: Exploring Efficient Generation of Unbounded Outdoor Scenes. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 26509–26518. IEEE, 2025. URL https://openaccess.thecvf.com/content/ICCV2025/html/Lee\_NuiScene\_Exploring\_Efficie nt\_Generation\_of\_Unbounded\_Outdoor\_Scenes\_ICCV\_2025\_paper.html.

Lu Ling, Chen-Hsuan Lin, Tsung-Yi Lin, Yifan Ding, Yu Zeng, Yichen Sheng, Yunhao Ge, Ming-Yu Liu, Aniket Bera, and Max Li. Scenethesis: A Language and Vision Agentic Framework for 3D Scene Generation. In International Conference on Learning Representations, volume 2026, pages 136596–136629, 2026. URL https: //proceedings.iclr.cc/paper\_files/paper/2026/hash/dd172fa21119391253587a4310a1a426-Abs tract-Conference.html.

Yuheng Liu, Xinke Li, Yuning Zhang, Lu Qi, Xin Li, Wenping Wang, Chongshou Li, Xueting Li, and Ming-Hsuan Yang. Controllable 3D Outdoor Scene Generation via Scene Graphs. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 28052–28062. IEEE, 2025. URL https://openaccess.thecvf.com/cont ent/ICCV2025/html/Liu\_Controllable\_3D\_Outdoor\_Scene\_Generation\_via\_Scene\_Graphs\_ICCV\_2 025\_paper.html.

Joshua Lochner, James Gain, Simon Perche, Adrien Peytavie, Eric Galin, and Eric Guérin. Interactive Authoring of Terrain using Diffusion Models. In Computer Graphics Forum, volume 42, page e14941. Wiley Online Library, 2023. doi: 10.1111/cgf.14941. URL https://doi.org/10.1111/cgf.14941.

Yifan Lu, Xuanchi Ren, Jiawei Yang, Tianchang Shen, Zhangjie Wu, Jun Gao, Yue Wang, Siheng Chen, Mike Chen, Sanja Fidler, et al. InfiniCube: Unbounded and Controllable Dynamic 3D Driving Scene Generation with World-Guided Video Models. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 27272–27283. IEEE, 2025. URL https://openaccess.thecvf.com/content/ICCV2025/html/Lu\_InfiniCube\_Unboun ded\_and\_Controllable\_Dynamic\_3D\_Driving\_Scene\_Generation\_with\_ICCV\_2025\_paper.html.

Quan Meng, Lei Li, Matthias Nießner, and Angela Dai. LT3SD: Latent Trees for 3D Scene Diffusion. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 650–660. IEEE, 2025. URL https://openaccess.thecvf.com/content/CVPR2025/html/Meng\_LT3SD\_Latent\_Trees\_for\_3D\_Sce ne\_Diffusion\_CVPR\_2025\_paper.html.

Hou In Derek Pun, Hou In Ivan Tam, Austin T Wang, Xiaoliang Huo, Angel X Chang, and Manolis Savva. HSM: Hierarchical Scene Motifs for Multi-Scale Indoor Scene Generation. In 2026 International Conference on 3D Vision (3DV), pages 1356–1367. IEEE, 2026. URL https://3dlg-hcvc.github.io/hsm/.

Alexander Raistrick, Lahav Lipson, Zeyu Ma, Lingjie Mei, Mingzhe Wang, Yiming Zuo, Karhan Kayan, Hongyu Wen, Beining Han, Yihan Wang, et al. Infinite Photorealistic Worlds Using Procedural Generation. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12630–12641. IEEE, 2023. URL https://arxiv.org/abs/2306.09310.

Alexander Raistrick, Lingjie Mei, Karhan Kayan, David Yan, Yiming Zuo, Beining Han, Hongyu Wen, Meenal Parakh, Stamatis Alexandropoulos, Lahav Lipson, et al. Infinigen Indoors: Photorealistic Indoor Scenes using Procedural Generation. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21783–21794. IEEE, 2024. URL https://openaccess.thecvf.com/content/CVPR2024/html/Raistrick\_ Infinigen\_Indoors\_Photorealistic\_Indoor\_Scenes\_using\_Procedural\_Generation\_CVPR\_2024\_p aper.html.

Xuanchi Ren, Jiahui Huang, Xiaohui Zeng, Ken Museth, Sanja Fidler, and Francis Williams. XCube: Large-Scale 3D Generative Modeling using Sparse Voxel Hierarchies. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4209–4219. IEEE, 2024. URL https://openaccess.thecvf.com/conten t/CVPR2024/html/Ren\_XCube\_Large-Scale\_3D\_Generative\_Modeling\_using\_Sparse\_Voxel\_Hierarc hies\_CVPR\_2024\_paper.html.

Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin Nimier-David, Thomas Müller, Alexander Keller, Sanja Fidler, and Jun Gao. GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6121–6132. IEEE, 2025. URL https://openaccess.thecvf.com/content/CVPR2025/html/Ren\_GEN3C\_3D-Informe d\_World-Consistent\_Video\_Generation\_with\_Precise\_Camera\_Control\_CVPR\_2025\_paper.html.

Tianchang Shen, Sherwin Bahmani, Kai He, Sangeetha Grama Srinivasan, Tianshi Cao, Jiawei Ren, Ruilong Li, Zian Wang, Nicholas Sharp, Zan Gojcic, et al. Lyra 2.0: Explorable Generative 3D Worlds. arXiv preprint arXiv:2604.13036, 2026. URL https://arxiv.org/abs/2604.13036.

Chunyi Sun, Junlin Han, Weijian Deng, Xinlong Wang, Zishan Qin, and Stephen Gould. 3D-GPT: Procedural 3D Modeling with Large Language Models. In 2025 International Conference on 3D Vision (3DV), pages 1253–1263. IEEE, 2025. URL https://arxiv.org/abs/2310.12945.

Dilin Wang, Hyunyoung Jung, Tom Monnier, Kihyuk Sohn, Chuhang Zou, Xiaoyu Xiang, Yu-Ying Yeh, Di Liu, Zixuan Huang, Thu Nguyen-Phuoc, Yuchen Fan, Sergiu Oprea, Ziyan Wang, Roman Shapovalov, Nikolaos Sarafianos, Thibault Groueix, Antoine Toisoul, Prithviraj Dhar, Xiao Chu, Minghao Chen, Geon Yeong Park, Rakesh Ranjan, and Andrea Vedaldi. WorldGen: From Text to Traversable and Interactive 3D Worlds. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 27124–27135, June 2026a. URL https://openaccess.thecvf.com/content/CVPR2026/html/Wang\_WorldGen\_From\_Text\_to\_Travers able\_and\_Interactive\_3D\_Worlds\_CVPR\_2026\_paper.html.

Pengfei Wang, Liyi Chen, Zhiyuan Ma, Yanjun Guo, Guowen Zhang, and Lei Zhang. One2Scene: Geometric Consistent Explorable 3D Scene Generation from a Single Image. In International Conference on Learning Representations, 2026b. URL https://proceedings.iclr.cc/paper\_files/paper/2026/hash/9aaa0d7db14b9f0c2575 a1761b1ab76e-Abstract-Conference.html.

Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-Image Technical Report. arXiv preprint arXiv:2508.02324, 2025. URL https: //arxiv.org/abs/2508.02324.

Zhennan Wu, Yang Li, Han Yan, Taizhang Shang, Weixuan Sun, Senbo Wang, Ruikai Cui, Weizhe Liu, Hiroyuki Sato, Hongdong Li, et al. BlockFusion: Expandable 3D Scene Generation using Latent Tri-plane Extrapolation. ACM Transactions on Graphics (ToG), 43(4):1–17, 2024. doi: 10.1145/3658188. URL https://doi.org/10.1145/36 58188.

Chong Xia, Shengjun Zhang, Fangfu Liu, Chang Liu, Khodchaphun Hirunyaratsameewong, and Yueqi Duan. ScenePainter: Semantically Consistent Perpetual 3D Scene Generation with Concept Relation Alignment. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 28808–28817. IEEE, 2025. URL https://openaccess.thecvf.com/content/ICCV2025/html/Xia\_ScenePainter\_Semantically\_Cons istent\_Perpetual\_3D\_Scene\_Generation\_with\_Concept\_Relation\_ICCV\_2025\_paper.html.

Zeqi Xiao, Yushi Lan, Yifan Zhou, Wenqi Ouyang, Shuai Yang, Yanhong Zeng, and Xingang Pan. WorldMem: Long-term Consistent World Simulation with Memory. In Advances in Neural Information Processing Systems, volume 38, pages 49632–49652, 2025. doi: 10.52202/085713-1659. URL https://proceedings.neurips.cc /paper\_files/paper/2025/hash/470629a47e2d65ce0606c40055df5d26-Abstract-Conference.html.

Yifei Xue, Yuanchen Fei, Hao Zhang, Chenzhi Nie, Yizhen Lao, et al. Illusion or Integrity? Geometrical Consistency Metric for AIGC Video Quality Evaluation. arXiv preprint arXiv:2608.09594, 2026. URL https://arxiv.org/ abs/2608.09594.

Yue Yang, Fan-Yun Sun, Luca Weihs, Eli VanderBilt, Alvaro Herrasti, Winson Han, Jiajun Wu, Nick Haber, Ranjay Krishna, Lingjie Liu, et al. Holodeck: Language Guided Generation of 3D Embodied AI Environments. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16277–16287. IEEE, 2024. URL https://openaccess.thecvf.com/content/CVPR2024/html/Yang\_Holodeck\_Language\_Guided\_Gene ration\_of\_3D\_Embodied\_AI\_Environments\_CVPR\_2024\_paper.html.

Yuanyang Yin, Gongxuan Wang, Yifan Zhan, Chuanhao Li, Kaipeng Zhang, and Feng Zhao. Alaya-EVOKE: From Linear-Scaling Supervision to Endless World. arXiv preprint arXiv:2608.13546, 2026. URL https://arxiv.org/ abs/2608.13546.

Hong-Xing Yu, Haoyi Duan, Junhwa Hur, Kyle Sargent, Michael Rubinstein, William T Freeman, Forrester Cole, Deqing Sun, Noah Snavely, Jiajun Wu, et al. WonderJourney: Going from Anywhere to Everywhere. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6658–6667. IEEE, 2024. URL https://openaccess.thecvf.com/content/CVPR2024/html/Yu\_WonderJourney\_Going\_from\_Anywhe re\_to\_Everywhere\_CVPR\_2024\_paper.html.

Hong-Xing Yu, Haoyi Duan, Charles Herrmann, William T Freeman, and Jiajun Wu. WonderWorld: Interactive 3D Scene Generation from a Single Image. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5916–5926. IEEE, 2025a. URL https://openaccess.thecvf.com/content/CVPR2025/html/ Yu\_WonderWorld\_Interactive\_3D\_Scene\_Generation\_from\_a\_Single\_Image\_CVPR\_2025\_paper.html.

Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. ViewCrafter: Taming Video Diffusion Models for High-fidelity Novel View Synthesis. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025b. doi: 10.1109/TPAMI.2025.3613256. URL https://doi.org/10.1109/TPAMI.2025.3613256.

Qihang Zhang, Shuangfei Zhai, Miguel Angel Bautista Martin, Kevin Miao, Alexander Toshev, Joshua Susskind, and Jiatao Gu. World-consistent Video Diffusion with Explicit 3D Modeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21685–21695, 2025. URL https://openaccess .thecvf.com/content/CVPR2025/html/Zhang\_World-consistent\_Video\_Diffusion\_with\_Explicit\_ 3D\_Modeling\_CVPR\_2025\_paper.html.

Yi Zhang, Yunshuang Wang, Zeyu Zhang, and Hao Tang. Code2Worlds: Empowering Coding LLMs for 4D World Generation. arXiv preprint arXiv:2602.11757, 2026a. URL https://arxiv.org/abs/2602.11757.

Yisu Zhang, Chenjie Cao, Tengfei Wang, Xuhui Zuo, Junta Wu, Jianke Zhu, and Chunchao Guo. WorldStereo: Bridging Camera-Guided Video Generation and Scene Reconstruction via 3D Geometric Memories. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026b. URL https://openaccess.thecvf. com/content/CVPR2026/html/Zhang\_WorldStereo\_Bridging\_Camera-Guided\_Video\_Generation\_and \_Scene\_Reconstruction\_via\_3D\_CVPR\_2026\_paper.html.

Mengqi Zhou, Yuxi Wang, Jun Hou, Shougao Zhang, Yiwei Li, Chuanchen Luo, Junran Peng, and Zhaoxiang Zhang. SceneX: Procedural Controllable Large-scale Scene Generation. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 39, pages 10806–10814, 2025. URL https://arxiv.org/abs/2403.15698.

Haoyi Zhu, Haozhe Liu, Yuyang Zhao, Tian Ye, Junsong Chen, Jincheng Yu, Tong He, Song Han, and Enze Xie. SANA-WM: Efficient Minute-Scale World Modeling with Hybrid Linear Diffusion Transformer. arXiv preprint arXiv:2605.15178, 2026. URL https://arxiv.org/abs/2605.15178.

## A Supplementary Ablation Studies

## A.1 Ablation Protocol

The codec test compares five encodings on 64 matched tasks through one frozen VAE. Terrain input evaluation uses four neighbor configurations, 64 targets each and two protocols, matching encoding, inference steps and seeds. Canvas uses pooled-context training and multi-image models use context subsets; targets are in-domain authored RDR2 terrain. The default planning model is GPT-5.6-Sol (gpt-5.6-sol, low reasoning effort), compared with Qwen3.8-Max. For planning-model substitution, terrain, intent, asset catalog, compiler, revision budget and cameras are fixed. We compare relation realization, geometric validity and revision cost alongside asset layouts.

## A.2 Visual Comparison of Terrain Conditioning

Figure 6 complements Table 2(b) with matched terrain completions. Spatial canvas conditioning maintains more coherent relief across the marked interfaces in these examples, supporting the boundary-continuity gains in the quantitative comparison.

## A.3 Planning-model Comparison

Figure 7 compares full layouts with identical terrain and an unchanged existing region on the left. The alternative planner favors wooded open space and dispersed settlements; the default planner produces a denser settlement and road network. Both retain the inherited terrain interface and organized connections.

## A.4 Random-seed Stability

## A.4.1 Controlled Comparison

We evaluate five fixed scene–trajectory pairs with seeds 0, 1, 2, 42 and 100. The world version, first frame, prompt, depth and non-seed generation settings are held fixed. Smoothness and MC-GeCo use the main comparison’s frozen task-wise motion reference; MC-GeCo divides each video’s co-visible residual by its relative motion before aggregation. Table 4 reports all 25 controlled outputs together with the five main-experiment references. Seed dispersion is computed from the 25 controlled outputs.

## A.4.2 Metric Variation and Visual Outcomes

Table 5 first averages the same five tasks for each seed. Its standard deviation and coefficient of variation are computed across those five seed-wise means, using the population standard deviation. This separates aggregate seed sensitivity from differences between scenes.

Computational measures. The aggregate IQ, AQ and MN-MS vary narrowly across seeds, with CVs of 0.39%, 0.40% and 0.05%. The geometric scores have larger but still comparatively moderate CVs: 5.29% for MC-GeCo, 3.11% for MC-MEt3R and 6.33% for MC-GeoCon. These results support stability of the aggregate appearance and geometry measures under the tested noise changes. Per-task results remain important: averaging can reduce dispersion and does not imply that every individual video is equally stable.

Occasional sky corruption. Visual inspection identifies a black-sky interval in one of the 25 controlled outputs, the mountain scene with seed 100. Its IQ is 0.7667, compared with 0.8021–0.8044 for the other four seeds, and MC-MEt3R rises to 0.1576 from 0.1093–0.1180. The outcome is consistent with the base video generator occasionally transferring dark depth-conditioning regions into sky appearance, although the seed comparison alone does not isolate that cause. The affected video remains in all statistics. This observation identifies an appearance-synthesis failure under unchanged geometry; one occurrence in this small study does not establish its deployment frequency.

Event-based evaluation. SMC and Cam have CVs of 27.61% and 15.60%, respectively, and vary more than the computational measures. Their discrete disappearance, return and motion-clause decisions can amplify small changes in generated evidence; the visual judgments additionally involve a large-model evaluator. These scores therefore reflect event realization and evaluation sensitivity as well as visual consistency. The present experiment varies generation seeds, not repeated judgments of an identical video, so it does not attribute all dispersion to evaluator randomness. We report their full ranges alongside the more stable computational measures.

![](images/ce453009e2cacd00e9872ee1f99c4645ab871ed2ae0b809b57cf15ddf99c5917.jpg)  
Figure 6: Terrain completion with spatial canvas (left) and separate images (right). Each pair shares known terrain, elevation colors and illumination. Dashed lines mark context–target interfaces; gray regions are unavailable.

(a) Alternative planner  
![](images/eaceeacb11548a22afa0b4022aae5f669d8a4fe58f7c8f28082d17c7bf9c356a.jpg)

(b) Default planner  
![](images/d32b85bedefc43792bc7a042547d5ad06000394682a9176da7d3582a4a59bf01.jpg)  
Figure 7: Full planning-model comparison: (a) alternative and (b) default planner. Within each map, the shared existing region is on the left and the new region on the right. Terrain and drawing conventions are identical.

Table 4: Per-task seed comparison. Each scene contains one main-experiment reference and five controlled seed variants. All values are retained; <sup>†</sup> marks the observed black-sky case.
<table><tr><td>Version</td><td>Seed</td><td>IQ↑</td><td>AQ↑</td><td>MN-MS↑</td><td>MC-GeCo↓</td><td>SMC↑</td><td>Cam↑</td><td>MC-MEt3R↓</td><td>MC-GeoCon↓</td></tr><tr><td>Forest</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Reference</td><td>一</td><td>0.7927</td><td>0.5921</td><td>0.9853</td><td>0.0468</td><td>1.0000</td><td>97.62</td><td>0.1080</td><td>0.0768</td></tr><tr><td>Seed</td><td>0</td><td>0.7943</td><td>0.5928</td><td>0.9834</td><td>0.0463</td><td>1.0000</td><td>97.62</td><td>0.1075</td><td>0.1002</td></tr><tr><td>Seed</td><td>1</td><td>0.7917</td><td>0.5929</td><td>0.9825</td><td>0.0448</td><td>1.0000</td><td>97.62</td><td>0.1053</td><td>0.0957</td></tr><tr><td>Seed</td><td>2</td><td>0.7916</td><td>0.5993</td><td>0.9828</td><td>0.0431</td><td>1.0000</td><td>97.62</td><td>0.1095</td><td>0.0902</td></tr><tr><td>Seed</td><td>42</td><td>0.7896</td><td>0.5929</td><td>0.9847</td><td>0.0423</td><td>1.0000</td><td>97.62</td><td>0.1078</td><td>0.0860</td></tr><tr><td>Seed</td><td>100</td><td>0.7930</td><td>0.6025</td><td>0.9862</td><td>0.0374</td><td>1.0000</td><td>95.24</td><td>0.1085</td><td>0.0655</td></tr><tr><td>Rocky grassland</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Reference</td><td>1</td><td>0.7930</td><td>0.6765</td><td>0.9842</td><td>0.0687</td><td>1.0000</td><td>97.62</td><td>0.1100</td><td>0.0660</td></tr><tr><td>Seed</td><td>0</td><td>0.7859</td><td>0.6473</td><td>0.9850</td><td>0.0783</td><td>1.7500</td><td>97.62</td><td>0.1128</td><td>0.0871</td></tr><tr><td>Seed</td><td>1</td><td>0.7874</td><td>0.6554</td><td>0.9840</td><td>0.0881</td><td>0.0000</td><td>40.48</td><td>0.1110</td><td>0.0806</td></tr><tr><td>Seed</td><td>2</td><td>0.7873</td><td>0.6443</td><td>0.9862</td><td>0.0743</td><td>1.7500</td><td>97.62</td><td>0.1110</td><td>0.0814</td></tr><tr><td>Seed</td><td>42</td><td>0.7918</td><td>0.6631</td><td>0.9853</td><td>0.0628</td><td>0.0000</td><td>40.48</td><td>0.1132</td><td>0.0779</td></tr><tr><td>Seed</td><td>100</td><td>0.7854</td><td>0.6407</td><td>0.9859</td><td>0.0652</td><td>0.0000</td><td>32.14</td><td>0.1120</td><td>0.0805</td></tr><tr><td>Town</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Reference</td><td>一</td><td>0.7958</td><td>0.7683</td><td>0.9789</td><td>0.0513</td><td>2.0000</td><td>97.96</td><td>0.1357</td><td>0.1563</td></tr><tr><td>Seed</td><td>0</td><td>0.7960</td><td>0.7641</td><td>0.9789</td><td>0.0496</td><td>1.5000</td><td>97.96</td><td>0.1378</td><td>0.1880</td></tr><tr><td>Seed</td><td>1</td><td>0.7966</td><td>0.7795</td><td>0.9806</td><td>0.0526</td><td>0.0000</td><td>41.84</td><td>0.1335</td><td>0.1490</td></tr><tr><td>Seed</td><td>2</td><td>0.7963</td><td>0.7716</td><td>0.9795</td><td>0.0524</td><td>1.0000</td><td>97.96</td><td>0.1355</td><td>0.1790</td></tr><tr><td>Seed Seed</td><td>42</td><td>0.7974</td><td>0.7683</td><td>0.9805</td><td>0.0530</td><td>1.5000</td><td>97.96</td><td>0.1348</td><td>0.1988</td></tr><tr><td></td><td>100</td><td>0.7964</td><td>0.7714</td><td>0.9784</td><td>0.0502</td><td>1.5000</td><td>97.96</td><td>0.1363</td><td>0.1809</td></tr><tr><td>Mountain</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Reference</td><td>一</td><td>0.8030</td><td>0.7135</td><td>0.9809</td><td>0.0490</td><td>3.5000</td><td>97.96</td><td>0.1166</td><td>0.0985</td></tr><tr><td>Seed</td><td>0</td><td>0.8044</td><td>0.7259</td><td>0.9810</td><td>0.0460</td><td>3.5000</td><td>97.96</td><td>0.1180</td><td>0.1274</td></tr><tr><td>Seed</td><td>1</td><td>0.8027</td><td>0.7207</td><td>0.9811</td><td>0.0436</td><td>2.5000</td><td>97.96</td><td>0.1107</td><td>0.1027</td></tr><tr><td>Seed</td><td>2</td><td>0.8021</td><td>0.7190</td><td>0.9807</td><td>0.0475</td><td>3.5000</td><td>97.96</td><td>0.1118</td><td>0.1300</td></tr><tr><td>Seed Seed</td><td>42</td><td>0.8028</td><td>0.7297</td><td>0.9819</td><td>0.0462</td><td>3.3750</td><td>97.96</td><td>0.1093</td><td>0.0891</td></tr><tr><td></td><td>100†</td><td>0.7667</td><td>0.7153</td><td>0.9818</td><td>0.0431</td><td>3.5000</td><td>97.96</td><td>0.1576</td><td>0.0949</td></tr><tr><td>Winter</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Reference</td><td>一</td><td>0.7921</td><td>0.7029</td><td>0.9751</td><td>0.0525</td><td>0.0000</td><td>34.69</td><td>0.1764</td><td>0.1709</td></tr><tr><td>Seed</td><td>0</td><td>0.7922</td><td>0.6954</td><td>0.9772</td><td>0.0403</td><td>1.0000</td><td>97.96</td><td>0.1669</td><td>0.1817</td></tr><tr><td>Seed</td><td>1</td><td>0.7899</td><td>0.6930</td><td>0.9759</td><td>0.0451</td><td>0.0000</td><td>34.69</td><td>0.1634</td><td>0.1607</td></tr><tr><td>Seed</td><td>2</td><td>0.7886 0.7943</td><td>0.6785</td><td>0.9777</td><td>0.0384</td><td>0.0000</td><td>34.69</td><td>0.1638</td><td>0.1661</td></tr><tr><td>Seed</td><td>42</td><td></td><td>0.6975</td><td>0.9777</td><td>0.0388</td><td>0.0000</td><td>34.69</td><td>0.1612</td><td>0.1583</td></tr><tr><td>Seed</td><td>100</td><td>0.7914</td><td>0.6953</td><td>0.9783</td><td>0.0399</td><td>0.0000</td><td>34.69</td><td>0.1638</td><td>0.1555</td></tr></table>

Table 5: Seed-wise means over five tasks and their across-seed dispersion. The five main-experiment references are excluded. SD is the population standard deviation and CV is SD divided by the mean, expressed as a percentage.
<table><tr><td>Seed</td><td>IQ</td><td>AQ</td><td>MN-MS</td><td>MC-GeCo</td><td>SMC</td><td>Cam</td><td>MC-MEt3R</td><td>MC-GeoCon</td></tr><tr><td>0</td><td>0.7945</td><td>0.6851</td><td>0.9811</td><td>0.0521</td><td>1.7500</td><td>97.82</td><td>0.1286</td><td>0.1369</td></tr><tr><td>1</td><td>0.7937</td><td>0.6883</td><td>0.9808</td><td>0.0548</td><td>0.7000</td><td>62.52</td><td>0.1248</td><td>0.1177</td></tr><tr><td>2</td><td>0.7932</td><td>0.6825</td><td>0.9814</td><td>0.0512</td><td>1.4500</td><td>85.17</td><td>0.1263</td><td>0.1293</td></tr><tr><td>42</td><td>0.7952</td><td>0.6903</td><td>0.9820</td><td>0.0486</td><td>1.1750</td><td>73.74</td><td>0.1253</td><td>0.1220</td></tr><tr><td>100</td><td>0.7866</td><td>0.6850</td><td>0.9821</td><td>0.0472</td><td>1.2000</td><td>71.60</td><td>0.1356</td><td>0.1155</td></tr><tr><td>Mean</td><td>0.7926</td><td>0.6863</td><td>0.9815</td><td>0.0508</td><td>1.2550</td><td>78.17</td><td>0.1281</td><td>0.1243</td></tr><tr><td>SD</td><td>0.0031</td><td>0.0027</td><td>0.0005</td><td>0.0027</td><td>0.3466</td><td>12.19</td><td>0.0040</td><td>0.0079</td></tr><tr><td>CV (%)</td><td>0.39</td><td>0.40</td><td>0.05</td><td>5.29</td><td>27.61</td><td>15.60</td><td>3.11</td><td>6.33</td></tr></table>

## B Evaluation Metrics

## B.1 Metric Definitions

Visual quality. IQ averages MUSIQ frame-quality estimates, AQ averages CLIP-based aesthetic predictions, and MS measures agreement with AMT-interpolated frames, following VBench (Huang et al., 2024). All measures evaluate generated RGB under the same public task description.

Structural Memory Consistency (SMC). SMC requires dedicated camera trajectories that induce a disappearance– return event for a designated target. To enable this evaluation, we randomly select 40% of all samples and adapt their sampling trajectories and camera motions accordingly; SMC is computed on this subset. A method-blind Qwen3.5-27B evaluator samples each video at 1 Hz, identifies the named target and its neighbors, and selects the earliest usable reference. It verifies complete target absence, then compares the reference with the earliest and latest usable return views. Each pair receives anchored 1–5 grades for identity, structure, neighbor relations and texture: 5 denotes preservation, 3 a clear local change, and 1 replacement or unrecognizable structure. Legitimate viewpoint and lighting changes are allowed. With grades $g _ { j d }$ for return view $j$ and criterion $d ,$ the score is

$$
s _ { j } = \mathrm { C a p } \left( \frac { \sum _ { d \in D _ { j } } w _ { d } g _ { j d } } { \sum _ { d \in D _ { j } } w _ { d } } \right) , \qquad \mathrm { S M C } = \frac { 1 } { | \mathcal { R } | } \sum _ { j \in \mathcal { R } } s _ { j } , \qquad w = ( 0 . 3 0 , 0 . 3 0 , 0 . 2 5 , 0 . 1 5 ) .\tag{9}
$$

Here $D _ { j }$ contains graded criteria and R the return views. Identi $\operatorname { t y } ,$ structure and neighbor evidence are required; an unobservable texture grade is omitted. Identity grade 1 fixes the pair score at 1; grade 2 caps it at 2. Structure at most 2 caps it at 2.5, and neighbor relations at most 2 cap it at 3. A designated memory task scores zero when its required disappearance–return event is absent.

Camera Compliance (Cam). The public prompt is decomposed into motion and visibility clauses. Translation and turning are evaluated from estimated camera poses at 1 Hz; target framing, absence and return use visual evidence. An event clause receives 1 when observed and 0 otherwise; a sustained clause receives the fraction of intervals satisfying it. Clause degrees $d _ { c }$ are combined as

$$
\mathrm { C a m } = \frac { 1 0 0 o } { | { \mathcal C } | } \sum _ { c \in { \mathcal C } } d _ { c } , \qquad o = 1 \mathrm { ~ f o r ~ c o r r e c t ~ e v e n t ~ o r d e r } , \quad o = \frac { 1 } { 2 } \mathrm { ~ o t h e r w i s e } .\tag{10}
$$

Thus Cam rewards executing the requested observation sequence, while SMC evaluates what is preserved when the target returns.

Geometric consistency. GeCo combines motion and depth residuals (Gu et al., 2026); MEt3R compares reprojected image features (Asim et al., 2025); GeoCon-Bench measures correspondence inlier ratio and geometric fitting error (Xue et al., 2026). We use a correspondence-based local reproduction for GeoCon and apply the corrections below.

## B.2 Motion-Calibrated Consistency and Smoothness

Realized motion. Motion amplitude is the evaluator’s mean optical-flow norm, reflecting scene depth, camera rotation and translation, and scene motion. Across 120 paired tasks, WorldWeave exceeds the six baselines listed in the main text on 74–93 tasks each.

The task requires both scene preservation and camera movement. Absolute disagreement can decrease when a model barely changes its view, even though it misses the requested exploration. Figure 8 shows this mismatch: favorable raw scores coexist with distorted texture or nearly stationary views. We therefore measure geometric disagreement relative to the motion actually realized and evaluate camera compliance separately.

Relative residuals on common visibility. Let u be observed angular flow, u the flow induced by estimated camera motion and depth, and $z _ { w } , z _ { t }$ warped and target depths. On pixels that project inside the target and pass depth agreement, the co-visible GeCo residual uses

$$
e _ { m } = \frac { \| u - u _ { r } \| _ { 2 } } { \| u \| _ { 2 } + \| u _ { r } \| _ { 2 } + \epsilon } , \qquad e _ { z } = \frac { | z _ { w } - z _ { t } | } { | z _ { w } | + | z _ { t } | + \epsilon } .\tag{11}
$$

The flow denominator expresses deformation relative to observed and predicted displacement; the visibility mask prevents newly exposed content from being treated as a correspondence failure. Valid motion and depth cues are averaged equally, retaining GeCo’s single-cue fallback.

Task-relative motion normalization. For GeCo, e is the co-visible relative residual above; for MEt3R and GeoCon, it is the original feature error or combined geometric error. Each video v is normalized by realized motion m(v) relative

(a) Texture degradation: dev-winter-a WorldWeave

AlayaWorld  
![](images/bdfde158063a776d14892b6f22d88a6490fd9f2884f009124fb88a9105c41045.jpg)  
Figure 8: Motion and consistency are complementary. AlayaWorld’s low raw MEt3R error coexists with distorted snow texture; Wan3.0’s low raw GeCo error accompanies little camera movement. Times are shown; scores refer to whole videos.

to the fixed task-wise reference $m _ { \mathrm { r e f } }$ , then scores are averaged:

$$
\begin{array} { l l } { \displaystyle r _ { m } ( v ) = \frac { m ( v ) } { m _ { \mathrm { r e f } } } , } & { \displaystyle e _ { \mathrm { M C } } ( v ) = \frac { e ( v ) } { r _ { m } ( v ) } , } \\ { \displaystyle \bar { e } _ { \mathrm { M C } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } e _ { \mathrm { M C } } ( v _ { i } ) , } & { \displaystyle e _ { \mathrm { G e o C o n } } = \sqrt { \mathrm { m a x } ( 1 - \mathrm { I R } , \epsilon ) \mathrm { m a x } ( \mathrm { G E } , \epsilon ) } . } \end{array}\tag{12}
$$

This reports error per relative amount of exploration: small raw error receives less credit when accompanied by little movement. The reference is the task-wise median from the frozen comparison set, shared across methods and held fixed when adding the Base or seed variants. Aggregation uses the mean of video-wise ratios, not the ratio of aggregate errors and motion. Non-positive motion and invalid correspondences are excluded. GeoCon remains supplementary because the local reproduction is insensitive to the tested local warp. The geometric scores and Cam jointly assess preservation under completed camera motion.

Motion-normalized smoothness. We apply the same task-relative motion calibration to VBench’s smoothness error:

$$
\mathrm { M N - M S } ( v ) = 1 - \frac { 1 - \mathrm { M S } ( v ) } { r _ { m } ( v ) } .\tag{13}
$$

Higher is better. Scores are computed per video before aggregation and interpreted jointly with geometric consistency and camera compliance.

## C User-study Protocol

Cases and presentation. The study contains 20 matched scene–trajectory cases with seven methods per case. Cases were selected using automatic metrics and visual screening before human rating; the reported means characterize this curated evaluation set. Participants view anonymized videos under shared task instructions; method names and automatic scores are hidden. Each account is assigned ten distinct videos, with presentation order shuffled and allocation prioritizing rating coverage. Participation is voluntary and requires informed consent.

Rating criteria. Participants assign integer scores from 1 to 5, with higher scores indicating better consistency. Camera consistency evaluates the requested movement, turns and observation order. Object consistency evaluates preservation of object identity, shape and number. Structure/texture consistency evaluates stable scene layout and coherent surface details. The same three questions accompany every video.

Aggregation. All 140 videos meet the minimum of three ratings: 124 have three and 16 have four, yielding 436 video evaluations and 1,308 criterion scores. We first average ratings per video and criterion, then average the 20 cases with equal weight. Let $s _ { c m d r }$ be rating r for case $c ,$ method m and criterion $d ,$ with $n _ { c m }$ ratings per video. The method-level

mean is

$$
\bar { s } _ { c m d } = \frac { 1 } { n _ { c m } } \sum _ { r = 1 } ^ { n _ { c m } } { s _ { c m d r } } , \qquad \mu _ { d } ( m ) = \frac { 1 } { | { \mathcal { C } } | } \sum _ { c \in { \mathcal { C } } } { \bar { s } } _ { c m d } .\tag{14}
$$

Here C contains all 20 cases. Each case has equal weight, so videos with four ratings do not contribute more than those with three. Scores remain on the original 1–5 scale; higher values indicate stronger perceived consistency. WorldWeave attains the unique highest object mean in 19 of 20 cases, and attains or shares the highest camera and structure/texture means in 16 and 18 cases.

## D Metric Terrain Implementation

## D.1 Context Reference and Coordinate Convention

Let ∆ be the metric sample spacing and $o _ { q }$ the target origin. Sample centers are $\begin{array} { r } { x _ { i j } = o _ { q } + \Delta ( i + \frac { 1 } { 2 } , j + \frac { 1 } { 2 } ) } \end{array}$ . Rotation into a canonical neighbor arrangement acts jointly on elevation, validity and side labels; outputs are mapped back before joining. Let $\bar { B } _ { \mathrm { c t x } }$ collect samples from inward-facing context bands and $X _ { a } = ( 1 , x _ { a } , y _ { a } )$ . The reference plane is initialized by least squares and refined by reweighted fits:

$$
\begin{array} { r l } & { \quad \ : e _ { a } ^ { ( l ) } = X _ { a } \theta ^ { ( l ) } - h _ { a } , \quad \sigma _ { l } = \operatorname* { m a x } \{ \sigma _ { \operatorname* { m i n } } , 1 . 4 8 2 6 \mathrm { ~ m e d i a n } _ { a } | e _ { a } ^ { ( l ) } | \} , } \\ & { \quad \ : \omega _ { a } ^ { ( l ) } = \operatorname* { m i n } \{ 1 , 1 . 5 \sigma _ { l } / \operatorname* { m a x } ( | e _ { a } ^ { ( l ) } | , \epsilon ) \} , } \\ & { \quad \ : \theta ^ { ( l + 1 ) } = \arg \operatorname* { m i n } _ { \theta } \displaystyle \sum _ { a \in \mathcal { B } _ { \mathrm { c t x } } } ( \omega _ { a } ^ { ( l ) } ) ^ { 2 } ( X _ { a } \theta - h _ { a } ) ^ { 2 } . } \end{array}\tag{15}
$$

The squared weights reflect weighting both the design matrix and observations. Multiple neighbors share one fit. With a single neighbor, the normal trend is tapered away from the interface to prevent indefinite linear extrapolation. Writing the plane coefficients as $( a , b , c )$ in normal–tangential coordinates $( d , y )$ gives

$$
b _ { q } ( d , y ) = a + c y + b \psi _ { L } ( d ) , \qquad \psi _ { L } ( d ) = \left\{ { d - d ^ { 2 } / ( 2 L ) } , \begin{array} { l l } { 0 \leq d < L , } \\ { L / 2 , } \end{array} \right.\tag{16}
$$

The normal derivative therefore decreases continuously to zero. Multi-neighbor configurations use the joint plane without this single-edge taper. Context bands are 64 samples wide; the fit uses four reweighting passes, $\sigma _ { \mathrm { m i n } } = 1$ m and $L = 6 4 \mathrm { m }$ . Missing elevations are filled from the nearest valid sample for numerical processing, while their original validity remains excluded from supervision. Target elevations never enter the context fit.

## D.2 Redundant Encoding and Robust Metric Decoding

The channel map $\begin{array} { r } { f _ { j } ( r ) = \frac { 1 } { 2 } + \frac { 1 } { 2 } \operatorname { t a n h } ( r / s _ { j } ) } \end{array}$ defines a one-dimensional curve in RGB space. Spatial repetition adds image-space redundancy, and block averaging restores the metric grid after VAE decoding:

$$
( U _ { \rho } c ) _ { \rho i + a , \rho j + b } = c _ { i j } , \qquad ( A _ { \rho } \widetilde { c } ) _ { i j } = \rho ^ { - 2 } \sum _ { a , b = 0 } ^ { \rho - 1 } \widetilde { c } _ { \rho i + a , \rho j + b } , \qquad \rho = 2 .\tag{17}
$$

For an averaged decoded sample $c _ { j } ,$ clip channel endpoints before inversion. Define $y _ { j } = 2 c _ { j } - 1$ and use inversesensitivity weighting to initialize the residual:

$$
r _ { j } = s _ { j } \operatorname { a t a n h } ( y _ { j } ) , \quad G _ { j } = \frac { 2 s _ { j } } { \operatorname { m a x } ( 1 - y _ { j } ^ { 2 } , \epsilon ) } , \quad r ^ { ( 0 ) } = \frac { \sum _ { j } G _ { j } ^ { - 2 } r _ { j } } { \sum _ { j } G _ { j } ^ { - 2 } } .\tag{18}
$$

Saturated channels have larger inverse gain and receive less weight. The channels need not agree after image decoding, so a fixed robust projection brings the estimate back toward their common encoding curve. With $J _ { j } ( r ) = f _ { j } ^ { \prime } ( r )$ and $\eta _ { j } ^ { ( l ) } = \operatorname* { m i n } \{ 1 , \kappa / \operatorname* { m a x } ( | f _ { j } ( r ^ { ( l ) } ) - c _ { j } | , \epsilon ) \}$ , the scalar update is

$$
r ^ { ( l + 1 ) } = r ^ { ( l ) } - \frac { \sum _ { j } \eta _ { j } ^ { ( l ) } J _ { j } ( r ^ { ( l ) } ) [ f _ { j } ( r ^ { ( l ) } ) - c _ { j } ] } { \operatorname* { m a x } \{ \sum _ { j } \eta _ { j } ^ { ( l ) } J _ { j } ( r ^ { ( l ) } ) ^ { 2 } , \epsilon \} } , \qquad \widehat { h } _ { q } = b _ { q } + r ^ { ( L _ { d } ) } .\tag{19}
$$

We use $L _ { d } = 5$ and $\kappa = 2 / 2 5 5$ , clipping channel values to $[ 1 0 ^ { - 5 } , 1 - 1 0 ^ { - 5 } ]$ . The projection is determined by decoded channels and the analytic codec.

## D.3 Conditioning, Supervision and Boundary Acceptance

Let $K _ { q } , T _ { q } , V _ { q }$ denote known-context, target and valid masks on the spatial canvas. The condition retains encoded context and replaces target content with a neutral value. A separate region image distinguishes target, validity and unavailable locations:

$$
C _ { \mathrm { c o n d } } ( x ) = \left\{ \begin{array} { l l } { f ( H ( x ) - B ( x ) ) , } & { x \in K _ { q } , } \\ { ( 1 / 2 , 1 / 2 , 1 / 2 ) , } & { x \in T _ { q } , } \\ { ( 0 , 0 , 1 ) , } & { x \notin K _ { q } \cup T _ { q } , } \end{array} \right. \quad R _ { q } = ( T _ { q } V _ { q } , V _ { q } , 1 - K _ { q } - T _ { q } ) .\tag{20}
$$

Canvas slots preserve neighbor direction; half-neighbor bands are used where the canonical layout places opposite neighbors around the full target. Only target-valid locations contribute to the normalized loss in the main text. Distances for edge weights are evaluated in target-local metric coordinates, so image repetition and the loss-grid resolution do not change the physical correction-band width.

Joining is evaluated on the shared interface, not by equating distinct sample centers on opposite sides. Let $\gamma _ { e }$ be interface height and derivative extraction with a common normal. Its residual is

$$
\begin{array} { r } { E _ { \partial q } ( h ) = \{ \gamma _ { e } ( h ) - ( g _ { e } , v _ { e } ) \} _ { e \in \mathcal { E } _ { q } } , \quad \gamma _ { e } ( h ) = ( h | _ { \Gamma _ { e } } , \partial _ { n _ { e } } h | _ { \Gamma _ { e } } ) . } \end{array}\tag{21}
$$

The connector inherits height and derivatives at the boundary of each preserved old surface. Corrections vanish with their normal derivatives at their inner support boundaries, while corner reconciliation acts on new-owned nodes. Continuity and modification magnitude are measured separately; all targets remain in the joining comparison, including those requiring large corrections.

## D.4 Ownership-aware Continuous Joining

Committed geometry is defined over the cells between existing sample centers. The gap between neighboring center grids and uncovered corner cells belong to the new connector surface. Let $J = \left( h , h _ { x } , h _ { y } , h _ { x y } \right)$ denote nodal height and derivatives. Bicubic Hermite connector cells inherit J on their old-facing edges; new-owned nodes supply the other constraints. Adjacent cells therefore share height and first derivatives without forcing spatially distinct old samples to coincide.

Missing context is completed before connection, preserving all original valid values and derivatives. Missing regions use nearest-valid completion, with a finite constant fallback only for entirely missing tiles. Generated targets retain their actual finite mask. Height and normal-slope corrections are applied on the new side, with normalized corner weights to avoid summing multiple full-strength side corrections. The fixed variant uses a 16 m band for both terms; slope taper keeps the height band fixed and shortens the derivative term’s support, targeting at most 0.5 m additional excursion with a minimum support of one sample spacing.

The joining experiment retains all 256 targets (64 terrain targets under four neighbor configurations). Large corrections receive quality diagnostics rather than being rejected from the comparison. The reported modification is

$$
\Delta h _ { q } = \left( \frac { 1 } { N _ { q } } \sum _ { i = 1 } ^ { N _ { q } } [ h _ { q } ^ { + } ( x _ { i } ) - \widehat { h } _ { q } ( x _ { i } ) ] ^ { 2 } \right) ^ { 1 / 2 } .\tag{22}
$$

Here $N _ { q }$ counts the fixed full-target samples. $H _ { c }$ and $S _ { c }$ measure height and slope disagreement through continuous queries at the preserved geometry boundary; they are distinct from the raw midpoint-extrapolated H/S in Table 2(b). Zeros in Table 2(c) denote residuals below $\mathrm { \dot { 1 } 0 ^ { - 9 } }$ . Their numerical closure follows from derivative inheritance, while $\Delta h _ { q }$ quantifies the cost of modifying the candidate. Relative to fixed $C ^ { 1 }$ joining, slope taper reduces mean modification by 0.1263 m (target-clustered 95% bootstrap interval: 0.0829–0.1806 m). The largest pointwise change remains 35.37 m; exact boundary closure does not imply uniformly small modifications or gentle slopes throughout the connector.

## E Hierarchical Scene Compilation

## E.1 Terrain Evidence and Spatial Responsibilities

Evidence is computed before semantic allocation. For metric elevation h, the local gradient determines slope and the Laplacian summarizes curvature. After depression filling, drainage uses the steepest positive descent among adjacent cells:

$$
\begin{array} { l } { \displaystyle s ( \boldsymbol { x } ) = \| \nabla h ( \boldsymbol { x } ) \| _ { 2 } , \qquad k ( \boldsymbol { x } ) = \nabla ^ { 2 } h ( \boldsymbol { x } ) , } \\ { \displaystyle d ( \boldsymbol { x } ) = \arg \operatorname* { m a x } _ { y \in \mathcal { N } _ { 8 } ( \boldsymbol { x } ) } \frac { \widetilde { h } ( \boldsymbol { x } ) - \widetilde { h } ( \boldsymbol { y } ) } { \| \boldsymbol { x } - \boldsymbol { y } \| _ { 2 } } , \qquad A ( \boldsymbol { x } ) = 1 + \sum _ { y : d ( \boldsymbol { y } ) = \boldsymbol { x } } A ( \boldsymbol { y } ) . } \end{array}\tag{23}
$$

Here $\widetilde { h }$ is the depression-filled field; cells without positive descent have no receiver. Accumulation A counts upstream contributing cells. The agent uses these fields together with inherited interfaces and asset capabilities to assign regional roles; evidence does not itself impose a semantic label.

Regions assign spatial responsibility, while districts refine it. Let $R _ { a }$ be a region and $D _ { a b }$ its districts on the target domain $\Omega _ { q }$ . Coverage and containment require

$$
\bigcup _ { a } R _ { a } = \Omega _ { q } , \qquad D _ { a b } \subseteq R _ { a } , \qquad \bigcup _ { b } D _ { a b } = R _ { a } , \qquad \mathrm { i n t } ( D _ { a b } ) \cap \mathrm { i n t } ( D _ { a c } ) = \emptyset ( b \neq c ) .\tag{24}
$$

These conditions apply to ownership layers; roads, water corridors and vegetation overlays carry separate typed respon sibilities. Boundary ports additionally constrain the position, direction and type of continuations. Region feasibility is checked before district and instance refinement, so later placement operates on accepted spatial responsibilities.

## E.2 Typed Relations and Asset Transforms

An asset plan is a typed relation graph $\mathcal { G } _ { q } = ( \mathcal { V } _ { q } ^ { \mathrm { a s s e t } } , \mathcal { E } _ { q } ^ { \mathrm { r e l } } )$ . Nodes carry catalog choice, role and district ownership; edges describe shared space, orientation, support or access. The catalog provides local geometry, support anchors, entrance axes and connector endpoints. For local point $p$ of instance $i ,$ the compiled transform is

$$
p ^ { w } = A _ { i } p + t _ { i } , \qquad T _ { i } = \left[ \begin{array} { l l } { A _ { i } } & { t _ { i } } \\ { 0 } & { 1 } \end{array} \right] , \qquad { \mathcal { T } } _ { q } = \{ ( \mathrm { i d } _ { i } , \mathrm { a s s e t } _ { i } , T _ { i } , \mathrm { r o l e } _ { i } , \mathrm { d i s t r i c t } _ { i } ) \} _ { i } .\tag{25}
$$

Allowed scaling is determined by asset type. For a two-ended connector module, let $p _ { - } , p _ { + }$ be local endpoints and $q _ { - } , q _ { + }$ the target endpoints. With $F ( v )$ an orthonormal frame whose first axis follows v, alignment is

$$
\begin{array} { l } { \alpha = \frac { \| q _ { + } - q _ { - } \| _ { 2 } } { \| p _ { + } - p _ { - } \| _ { 2 } } , } \\ { A _ { i } = F ( q _ { + } - q _ { - } ) \operatorname { d i a g } ( \alpha , 1 , 1 ) F ( p _ { + } - p _ { - } ) ^ { \top } , } \\ { \quad t _ { i } = ( q _ { - } + q _ { + } ) / 2 - A _ { i } ( p _ { - } + p _ { + } ) / 2 . } \end{array}\tag{26}
$$

Only the longitudinal axis is scaled, and forbidden span changes are rejected. This transform aligns connector endpoints with their prescribed world positions. Other asset families use their own support and orientation contracts under the same typed interface.

## E.3 Geometric Validation and Repair Attribution

The compiler evaluates the predicates required by each asset contract. With world support anchors $S _ { i }$ , admissible support surface $s _ { i }$ , collision envelopes $B _ { i }$ , and access graph H, representative constraints are

$$
\begin{array} { r l } & { \underset { p \in S _ { i } } { \operatorname* { m a x } } \operatorname { d i s t } ( p , S _ { i } ) \leq \tau _ { i } ^ { \mathrm { s u p } } , } \\ & { \operatorname { d i s t } ( B _ { i } , B _ { j } ) \geq c _ { i j } \quad \mathrm { f o r ~ p a i r s ~ r e q u i r i n g ~ c l e a r a n c e } , } \\ & { \operatorname { R e a c h } _ { \mathcal { H } } ( \mathrm { e n t r y } _ { i } , \mathrm { r o a d } _ { i } ) = 1 \quad \mathrm { f o r ~ a s s e t s ~ r e q u i r i n g ~ r o a d ~ a c c e s s } . } \end{array}\tag{27}
$$

Support relations permit intended contact; clearance applies to incompatible object pairs. The tolerances and required predicates belong to the asset contract. Each failed predicate retains its instance, relation and district identifiers, which

localize the plan entries to revise. Relation graphs therefore serve both construction and failure attribution, while deterministic geometry tools remain responsible for metric realization.

## F Revision, Publication and Read-only Observation

## F.1 Bounded Candidate Revision

Program checks and visual review consume the same candidate revision. Let $g _ { j } ( \Delta M _ { q } )$ be its required geometric predicates and $a _ { \mathrm { r e g } } , a _ { \mathrm { v i s } }$ the region and compiled-scene review decisions. Publication requires every geometric predicate and both review decisions to pass:

$$
\mathrm { A c c e p t } ( \Delta M _ { q } ) = a _ { \mathrm { r e g } } \wedge a _ { \mathrm { v i s } } \wedge \bigwedge _ { j \in \mathcal { I } _ { \mathrm { r e q } } } g _ { j } ( \Delta M _ { q } ) .\tag{28}
$$

A revision delta identifies a set of editable candidate records $\mathcal { U } _ { t }$ . Unaffected semantic decisions remain fixed, while the compiler recomputes geometry that depends on edited records. The plan-level restriction and old-world protection are

$$
\pi _ { q } ^ { ( t + 1 ) } | _ { \mathcal { U } _ { t } ^ { c } } = \pi _ { q } ^ { ( t ) } | _ { \mathcal { U } _ { t } ^ { c } } , \qquad \mathrm { W r i t e S e t } ( \Delta M _ { q } ^ { ( t ) } ) \cap \Omega _ { k } = \emptyset .\tag{29}
$$

A failed revision does not relax acceptance predicates. When the budget is exhausted, the candidate is rejected and the previous version remains available. Each revision regenerates reports and matched previews for its own candidate.

## F.2 Versioned Publication

Publication binds terrain, instance transforms, plans and interface records into one version. Let v be the visible version identifier and $\widehat { M }$ the validated candidate world. Readers select a version before querying:

$$
\widehat { M } = M _ { k } \oplus \Delta M _ { q } , \qquad v _ { \mathrm { a f t e r } } = \left\{ { k + 1 , \atop \mathrm { ~ { \scriptsize ~ \ o t h e r w i s e } . } } \right. \ x \mathrm { c c e p t } ( \Delta M _ { q } ) = 1 \mathrm { ~ a n d ~ p u b l i c a t i o n ~ s u c c e e d s } ,\tag{30}
$$

New records expose interfaces for subsequent additions, while existing instance identities and transforms persist. A query retains its selected version throughout a trajectory. Adding geometry may change visibility in a later version, but does not rewrite the earlier version’s structural records.

## F.3 Camera Geometry and Video Conditions

A viewing request provides a trajectory or lets the agent sample one from the requested motion and targets. The resulting camera schedule is $C _ { t } = ( K _ { t } , \bar { R } _ { t } , c _ { t } , \bar { \tau } _ { t } )$ , where $R _ { t }$ maps camera axes to world axes, $c _ { t }$ is the camera center and $\tau _ { t }$ is the timestamp. Clearance and requested visibility events are evaluated against the selected world. For homogeneous pixel $\bar { u } = ( u , \bar { v } , 1 ) ^ { \top }$ , the normalized world ray is

$$
d _ { t } ( u ) = \frac { R _ { t } K _ { t } ^ { - 1 } \bar { u } } { \| R _ { t } K _ { t } ^ { - 1 } \bar { u } \| _ { 2 } } , \qquad \lambda _ { t } ^ { * } ( u ) = \operatorname* { m i n } \{ \lambda > 0 : c _ { t } + \lambda d _ { t } ( u ) \in \mathcal { G } ( M _ { v } ) \} .\tag{31}
$$

The geometry set $\mathcal { G } ( M _ { v } )$ includes terrain and transformed assets. The nearest valid intersection determines visibility, hit identity and depth. Ray distance and camera-axis depth are distinct quantities:

$$
X _ { t } ( u ) = c _ { t } + \lambda _ { t } ^ { * } ( u ) d _ { t } ( u ) , \qquad D _ { t } ^ { \mathrm { r a y } } ( u ) = \lambda _ { t } ^ { * } ( u ) , \qquad D _ { t } ^ { z } ( u ) = e _ { 3 } ^ { \top } R _ { t } ^ { \top } [ X _ { t } ( u ) - c _ { t } ] .\tag{32}
$$

Here $e _ { 3 } = ( 0 , 0 , 1 ) ^ { \top }$ is the camera’s optical-axis unit vector. Pixels without a valid hit carry an explicit validity mask. The adapter preserves the depth convention and calibration while converting resolution, temporal sampling and encoding to the video model’s interface. Depth conditions are paired with optional text or image rendering-style references supported by the selected generator. All queries and RGB synthesis operate on the selected world without a write-back path.

## F.4 Relation to Joint-Embedding Predictive Architectures

I-JEPA learns semantic image representations by predicting target-region embeddings from context (Assran et al., 2023). V-JEPA extends feature prediction to video, learning representations through a latent prediction objective without pixel reconstruction (Bardes et al., 2024). V-JEPA 2 further uses an action-conditioned predictor over learned representations for planning (Assran et al., 2025). These approaches support a broader distinction between internal world modeling and the synthesis of visual observations.

WorldWeave shares this motivation but implements the separation at the level of explicit world-state maintenance. Its persistent state comprises metric terrain, asset instances, semantic organization and cross-region interfaces. Hierarchical planning, geometric compilation and validated commits extend that state; a video generator receives depth queried from a selected version and optional appearance references. JEPA concerns learning predictive representations, whereas WorldWeave concerns constructing and preserving a geometrically addressable world for subsequent observation. The connection is conceptual rather than an adoption of the JEPA training objective: our contribution is the concrete state-construction, extension and read-only query mechanism.

## G Additional Visual Comparisons

Figures 9, 10 and 11 compare target persistence during occlusion and off-screen return in farmstead, town and woodland scenes. The sequences allow comparison of object identity, neighboring layout and completion of the requested camera motion. Each comparison includes all eleven methods, including MiniMax-H3 (Base).

Prompt A continuous photorealistic walk through a rural farm courtyard in stable spring daylight. First observe the raised timber farmhouse in the center, the roofed circular well in front of it and the low barn on its right. Walk around the outside of the near timber granary on the left. Its solid wall must completely hide the farmhouse briefly; after passing around it, observe the same farmhouse again from the changed position. Preserve the farmhouse roof, porch, steps, well and barn identities and their fixed spatial relationships. One location and one continuous camera move.

Frame 1  
Frame 20  
Frame 30  
Frame 40  
Frame 50  
Frame 60  
![](images/cecc7daeb10f023d07a19c850b5b8f361145e2953dbec05f2389292a9306a184.jpg)  
Figure 9: Occlusion-and-return comparison in a farmstead. Rows show different methods and columns follow the video sequence. Numbered markers identify the farmhouse, well and barn for comparison across disappearance and return.

## 1 Workshop 2 Chimney 3 Crates 4 Loading barn

Prompt A continuous photorealistic walking shot in stable early-autumn daylight. First clearly observe the timber carriage workshop with its roof cupola and separate tall rectangular brick chimney, together with the crates on its left and the low loading barn on its right. Walk continuously around the solid foreground warehouse on the left, keeping attention on the same subject. The solid occluder must briefly hide the entire subject; then reveal the same subject again from the changed position after passing the occluder. Preserve the subject's geometry, appearance and fixed neighbor arrangement throughout. One location, no scene cut.  
![](images/a169d9011ea9e5eaded5635180bf19b153f5161eca8507080a129d5e6cdb01b1.jpg)  
Figure 10: Occlusion-and-return comparison in a town. Rows show different methods and columns follow the video sequence. Markers track the workshop, chimney, crates and loading barn as the camera passes behind the warehouse.

Prompt A continuous gently translating short-arc walk along a conifer creek-side clearing in natural early-autumn daylight. First establish the gabled timber ranger cabin with a stone chimney, the short parallel-log creek crossing front-left and covered stacked-log rack on the right. Turn smoothly toward the populated work canopy, low workshop and separate annex until the origina cabin completely leaves the view. Continue translating, then look back from a new position at the same cabin, log crossing and wood rack. Preserve their structure, identity, materials and world-space adjacency, including the creek bank and boulder. The short bridge stays made of round logs, not a different plank or stone bridge. No cuts or abrupt swings.

![](images/e9b83cd7361958ca24cfe1f7adf28f0328eb74c430dd0519a03c8c2722f66a59.jpg)  
Figure 11: Off-screen-return comparison in a woodland scene. Rows show different methods and columns follow the video sequence. Markers track the ranger cabin, log crossing and wood rack during the turn-away and return sequence.